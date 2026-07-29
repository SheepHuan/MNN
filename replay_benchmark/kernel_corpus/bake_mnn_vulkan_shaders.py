#!/usr/bin/env python3
"""Bake MNN Vulkan .comp shaders into standalone compilable files.

For each source .comp in sources/mnn/<tag>/source/backend/vulkan/buffer/execution/glsl/,
produce a baked .comp in operators/vulkan/<op_type>/mnn/<tag>/ with MNN Vulkan macros
expanded for the FP32 path, plus a pre-compiled .spv so device runs do not need
glslangValidator.

The MNN Vulkan shaders use ``FLOAT``/``FLOAT4`` macros (no ncnn-style sfp/afp layer).
Baking only prepends a ``#version 450`` preamble with these macro definitions; the
kernel body is preserved verbatim.

Usage:
  python3 bake_mnn_vulkan_shaders.py --sources <corpus>/sources \
      --operators <corpus>/operators [--tags 3.6.0]
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


# FP32 preamble: MNN Vulkan shaders reference FLOAT / FLOAT4 (and occasionally
# OUTPUT_TYPE in binary.comp, which is defined inside the shader itself).
MNN_VULKAN_FP32_PREAMBLE = """#version 450

#define FLOAT float
#define FLOAT2 vec2
#define FLOAT4 vec4
#define FLOAT8 mat4x2
#define FLOAT16 mat4

"""

LICENSE_HEADER = (
    "// Baked from MNN historical Vulkan shader. SPDX-License-Identifier: Apache-2.0.\n"
    "// FP32 path: FLOAT=float, FLOAT4=vec4. See bake_mnn_vulkan_shaders.py.\n"
)


# Curated mapping: op_type -> list of (source filename, output variant stem, extra -D macros).
# Multiple variants sharing one op_type are listed under the same key so they
# land in the same operators/vulkan/<op_type>/ directory (the extractor keys
# op_type off the directory name).
#
# The first batch covers the six buffer-class operators that already have
# dedicated OpenCL adapters in mnn/ops/, so the Vulkan branch can share
# validator and CaseSpec parameters.
#
# Macros listed in `macros` are baked into the SPIR-V at compile time (the
# Vulkan runner uses pre-compiled .spv and cannot apply -D at runtime, so
# every operator variant that is selected by a #ifdef must be baked as a
# separate .comp/.spv pair).
OP_MAP = {
    "unary": [("unary.comp", "vulkan_unary_buf_exp_fp32", ["EXP"])],
    "binary": [("binary.comp", "vulkan_binary_buf_add_fp32", ["ADD"])],
    "raster": [
        ("blit.comp", "vulkan_blit_c4_fp32", ["C4"]),
        ("nc4hw4Tonchw.comp", "vulkan_nc4hw4_to_nchw_fp32", []),
    ],
    "reduction": [("reduce.comp", "vulkan_reduce_buf_sum_fp32", ["SUM"])],
    "pooling": [
        ("maxpool.comp", "vulkan_maxpool_fp32", []),
        ("avgpool.comp", "vulkan_avgpool_fp32", []),
    ],
    "select": [("select.comp", "vulkan_select_fp32", [])],
    "range": [("range.comp", "vulkan_range_fp32", [])],
    "cast": [("cast_float_int.comp", "vulkan_cast_float_int_fp32", [])],
    "scale": [("scale.comp", "vulkan_scale_fp32", [])],
    "prelu": [("preluWithChannel.comp", "vulkan_prelu_fp32", [])],
    "argmax": [("argmax.comp", "vulkan_argmax_fp32", [])],
    "softmax": [("softmaxHeight_NHWC.comp", "vulkan_softmax_height_fp32", [])],
    "layernorm": [("norm.comp", "vulkan_norm_fp32", [])],
    "interp": [
        ("resizeNearest.comp", "vulkan_resize_nearest_fp32", []),
        ("resizeBilinear.comp", "vulkan_resize_bilinear_fp32", []),
    ],
    "grid_sample": [("gridSampleNearest.comp", "vulkan_grid_sample_nearest_fp32", [])],
}


def bake_shader(source_path: Path, out_dir: Path, variant: str,
                macros: list, glslang: str) -> int:
    """Bake one .comp + .spv. Returns 0 on success, nonzero on failure."""
    raw = source_path.read_text(encoding="utf-8", errors="replace")
    # MNN Vulkan shaders declare constants as `layout(set=0, binding=N) uniform
    # constBuffer { ... }`. The replay runner passes constants via push constants
    # (vkCmdPushConstants), which requires the shader to declare them as
    # `layout(push_constant) uniform`. Rewrite the declaration at bake time so
    # the pre-compiled SPIR-V matches the runner's push-constant path.
    raw = re.sub(
        r'layout\(set\s*=\s*0\s*,\s*binding\s*=\s*\d+\)\s*(readonly\s+)?uniform\s+\w+',
        'layout(push_constant) uniform constBuffer',
        raw,
    )
    baked = LICENSE_HEADER + MNN_VULKAN_FP32_PREAMBLE + raw
    out_dir.mkdir(parents=True, exist_ok=True)
    comp_path = out_dir / (variant + ".comp")
    spv_path = out_dir / (variant + ".spv")
    comp_path.write_text(baked, encoding="utf-8")

    # Pre-compile SPIR-V so device runs do not need glslangValidator.
    # Operator-selection macros (EXP/ADD/C4/SUM/...) are baked in here because
    # the Vulkan runner consumes pre-compiled .spv and cannot apply -D later.
    cmd = [
        glslang, "-V", "--target-env", "vulkan1.1",
        "-e", "main",
    ]
    for m in macros:
        cmd.append("-D{}".format(m))
    cmd += [str(comp_path), "-o", str(spv_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(
            "FAIL compile {}:\n{}\n".format(comp_path, proc.stderr.strip())
        )
        return proc.returncode
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True,
                        help="Corpus sources/ root (contains mnn/<tag>/...).")
    parser.add_argument("--operators", required=True,
                        help="Corpus operators/ root (output base).")
    parser.add_argument("--tags", nargs="*", default=["3.6.0"],
                        help="MNN source tags to bake (default: 3.6.0).")
    parser.add_argument("--glslang", default=shutil.which("glslangValidator") or "glslangValidator",
                        help="Path to glslangValidator (default: PATH lookup).")
    args = parser.parse_args(argv)

    sources_root = Path(args.sources)
    operators_root = Path(args.operators)
    failures = 0
    baked_count = 0

    for tag in args.tags:
        glsl_dir = sources_root / "mnn" / tag / "source" / "backend" / "vulkan" \
            / "buffer" / "execution" / "glsl"
        if not glsl_dir.is_dir():
            sys.stderr.write("WARN: no MNN Vulkan glsl dir for tag {}: {}\n".format(tag, glsl_dir))
            continue
        for op_type, variants in OP_MAP.items():
            for src_name, variant, macros in variants:
                src = glsl_dir / src_name
                if not src.is_file():
                    sys.stderr.write("WARN: missing source {} for op {} tag {}\n".format(src, op_type, tag))
                    continue
                out_dir = operators_root / "vulkan" / op_type / "mnn" / tag
                rc = bake_shader(src, out_dir, variant, macros, args.glslang)
            if rc != 0:
                failures += 1
            else:
                baked_count += 1
                print("OK {}/{}/{}/{}.comp (+.spv)".format(op_type, "mnn", tag, variant))

    print("\nBaked {} shader(s), {} failure(s).".format(baked_count, failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
