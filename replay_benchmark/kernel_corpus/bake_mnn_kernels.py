#!/usr/bin/env python3
"""Bake MNN OpenCL .cl kernels into standalone compilable files.

For each .cl in sources/mnn/<tag>/source/backend/opencl/execution/cl/,
extract every __kernel entry point and produce a baked .cl file with all
required macros (FLOAT, GLOBAL_SIZE_DIMS, DEAL_NON_UNIFORM, etc.) defined
for the FP32 path.

Only processes _buf.cl files (buffer-only, no image2d_t) to ensure
standalone compilability without OpenCL image support.

Usage:
  python3 bake_mnn_kernels.py --sources <corpus>/sources --operators <corpus>/operators
"""

import argparse
import os
import re
import sys
from pathlib import Path


MNN_FP32_PREAMBLE = """// Baked from MNN historical kernel. Copyright Alibaba Group Holding Limited.
// FP32 path: FLOAT=float, all macros expanded. See bake_mnn_kernels.py.
#ifndef MNN_BAKED_FP32_PREAMBLE
#define MNN_BAKED_FP32_PREAMBLE
typedef float FLOAT;
typedef float2 FLOAT2;
typedef float4 FLOAT4;
typedef float8 FLOAT8;
typedef float16 FLOAT16;
typedef float INPUT_TYPE;
typedef float OUTPUT_TYPE;
typedef float3 FLOAT3;
typedef float3 INPUT_TYPE3;
typedef float3 OUTPUT_TYPE3;
typedef float4 INPUT_TYPE4;
typedef float4 OUTPUT_TYPE4;
typedef float8 INPUT_TYPE8;
typedef float8 OUTPUT_TYPE8;
typedef float16 INPUT_TYPE16;
typedef float16 OUTPUT_TYPE16;
#define COMPUTE_FLOAT float
#define COMPUTE_FLOAT2 float2
#define COMPUTE_FLOAT3 float3
#define COMPUTE_FLOAT4 float4
#define COMPUTE_FLOAT8 float8
#define COMPUTE_FLOAT16 float16
#define CONVERT_FLOAT4(x) (convert_float4(x))
#define CONVERT_FLOAT(x) (convert_float(x))
#define CONVERT_FLOAT2(x) (convert_float2(x))
#define CONVERT_FLOAT3(x) (convert_float3(x))
#define CONVERT_FLOAT8(x) (convert_float8(x))
#define CONVERT_FLOAT16(x) (convert_float16(x))
#define CONVERT_COMPUTE_FLOAT2(x) (convert_float2(x))
#define CONVERT_COMPUTE_FLOAT3(x) (convert_float3(x))
#define CONVERT_COMPUTE_FLOAT4(x) (convert_float4(x))
#define CONVERT_COMPUTE_FLOAT8(x) (convert_float8(x))
#define CONVERT_COMPUTE_FLOAT16(x) (convert_float16(x))
#define CONVERT_OUTPUT4(x) (convert_float4(x))
#define CONVERT_INPUT4(x) (convert_float4(x))
#define CONVERT_OUTPUT8(x) (convert_float8(x))
#define CONVERT_INPUT8(x) (convert_float8(x))
#define CONVERT_OUTPUT3(x) (convert_float3(x))
#define CONVERT_INPUT3(x) (convert_float3(x))
#define CONVERT_OUTPUT16(x) (convert_float16(x))
#define CONVERT_INPUT16(x) (convert_float16(x))
#define AS_INPUT_DATA4(x) (convert_float4(x))
#define AS_INPUT_DATA8(x) (convert_float8(x))
#define AS_INPUT_DATA16(x) (convert_float16(x))
#define RI_F(img, smp, coord) read_imagef(img, smp, coord)
#define WI_F(img, coord, val) write_imagef(img, coord, val)
#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,
#define GLOBAL_SIZE_3_DIMS __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,
#define GLOBAL_SIZE_DIM2 __private const int global_size_dim0, __private const int global_size_dim1,
#define DEAL_NON_UNIFORM_DIM2(input1, input2) if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { return; }
#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3) if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { return; }
#endif
"""


def is_buf_kernel(filename):
    """Only process buffer kernels (no image2d_t dependency)."""
    return filename.endswith("_buf.cl") or filename.endswith("_subgroup_buf.cl")


def get_op_type(filename):
    """Derive op_type from filename."""
    stem = filename.replace(".cl", "")
    # Map common prefixes to op_type
    if stem.startswith("conv_2d"):
        return "conv"
    if stem.startswith("depthwise_conv2d"):
        return "depthwise_conv"
    if stem.startswith("depthwise_deconv"):
        return "depthwise_deconv"
    if stem.startswith("deconv"):
        return "deconv"
    if stem.startswith("matmul"):
        return "matmul"
    if stem.startswith("gemm"):
        return "gemm"
    if stem.startswith("gemv"):
        return "gemv"
    if stem.startswith("winograd"):
        return "winograd"
    if stem.startswith("binary"):
        return "binary"
    if stem.startswith("unary"):
        return "unary"
    if stem.startswith("raster"):
        return "raster"
    if stem.startswith("reduction"):
        return "reduction"
    if stem.startswith("pooling"):
        return "pooling"
    if stem.startswith("scale"):
        return "scale"
    if stem.startswith("softmax"):
        return "softmax"
    if stem.startswith("interp"):
        return "interp"
    if stem.startswith("cast"):
        return "cast"
    if stem.startswith("range"):
        return "range"
    if stem.startswith("select"):
        return "select"
    if stem.startswith("strassen"):
        return "strassen"
    if stem.startswith("splitgelu"):
        return "splitgelu"
    if stem.startswith("topkv2"):
        return "topkv2"
    if stem.startswith("layernorm"):
        return "layernorm"
    if stem.startswith("groupnorm"):
        return "groupnorm"
    if stem.startswith("attention"):
        return "attention"
    if stem.startswith("input_transe"):
        return "input_transe"
    if stem.startswith("grid_sample"):
        return "grid_sample"
    if stem.startswith("argmax"):
        return "argmax"
    if stem.startswith("loop"):
        return "loop"
    return stem


def strip_mnn_fp16_guard(source):
    """Remove #ifdef MNN_SUPPORT_FP16 / #pragma EXTENSION cl_khr_fp16 blocks.
    Keep the #else (FP32) branch if present, otherwise remove the block.
    """
    result = []
    lines = source.split("\n")
    skip_depth = 0
    keep_else = False
    for line in lines:
        stripped = line.strip()
        if skip_depth > 0:
            if stripped.startswith("#if"):
                skip_depth += 1
            elif stripped == "#endif":
                skip_depth -= 1
            elif stripped == "#else" and skip_depth == 1:
                skip_depth = 0
                keep_else = True
            continue
        if keep_else:
            if stripped.startswith("#if"):
                keep_else_count = keep_else_count + 1 if 'keep_else_count' in dir() else 1
            if stripped == "#endif":
                keep_else = False
                continue
            result.append(line)
            continue
        if stripped.startswith("#ifdef MNN_SUPPORT_FP16") or stripped.startswith("#ifndef MNN_SUPPORT_FP16"):
            skip_depth = 1
            continue
        if stripped.startswith("#pragma OPENCL EXTENSION cl_khr_fp16"):
            continue
        result.append(line)
    return "\n".join(result)


def extract_entries(source):
    """Find all __kernel void NAME(...) entry names."""
    # Match: __kernel void name( or __kernel __attribute__ void name(
    pattern = re.compile(r'__kernel\s+(?:__attribute__\s*\([^)]*\)\s+)?void\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
    return [m.group(1) for m in pattern.finditer(source)]


def bake_kernel(source_path, output_path, tag, is_pack4=False):
    """Bake a single .cl kernel file."""
    source = source_path.read_text(encoding="utf-8", errors="replace")
    # Strip FP16 guards
    source = strip_mnn_fp16_guard(source)
    # Remove original license/header comments
    lines = source.split("\n")
    while lines and (lines[0].strip().startswith("//") or lines[0].strip() == ""):
        lines.pop(0)
    source = "\n".join(lines)
    # Build full baked source
    # Note: OPERATOR/OPERATE are NOT defined here — they are operator-specific
    # and must be provided by the adapter via compileMacros (-DOPERATOR=...).
    preamble = MNN_FP32_PREAMBLE
    # 3.6.0 reduction_buf.cl uses OPERATE(out, in) as a two-argument function
    # macro. OpenCL -D does not support function-like macros, so bake the SUM
    # definition into the preamble. Other ops (MAX/MIN/PROD) are not baked
    # because they would need separate variants (not in corpus yet).
    if tag == "3.6.0" and source_path.name == "reduction_buf.cl":
        preamble = preamble + "#ifndef OPERATE\n#define OPERATE(a, b) ((a) + (b))\n#endif\n"
    result = preamble + source
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result, encoding="utf-8")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, help="Corpus sources/ root")
    parser.add_argument("--operators", required=True, help="Corpus operators/ root")
    args = parser.parse_args(argv)

    sources_root = Path(args.sources)
    operators_root = Path(args.operators)

    for tag in ["1.2.0", "3.6.0"]:
        cl_dir = sources_root / "mnn" / tag / "source/backend/opencl/execution/cl"
        if not cl_dir.is_dir():
            continue
        for cl_file in sorted(cl_dir.glob("*.cl")):
            if not is_buf_kernel(cl_file.name):
                continue
            op_type = get_op_type(cl_file.name)
            variant = cl_file.stem  # e.g. conv_2d_buf
            out_dir = operators_root / "opencl" / op_type / "mnn" / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / (variant + "_fp32.cl")
            try:
                bake_kernel(cl_file, out_file, tag)
                print(f"OK: {out_file.relative_to(operators_root)}")
            except Exception as e:
                print(f"FAIL: {cl_file.name}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
