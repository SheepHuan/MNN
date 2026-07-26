#!/usr/bin/env python3
"""Bake ncnn Vulkan .comp shaders into standalone compilable files.

For each source .comp in sources/ncnn/<tag>/src/layer/vulkan/shader/,
produce a baked .comp in operators/vulkan/<op>/<tag>/ with all ncnn macros
expanded for the FP32 scalar (sfp) or pack4 (sfpvec4) path.

Usage:
  python3 bake_ncnn_shaders.py --sources <corpus>/sources --operators <corpus>/operators
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


# FP32 scalar (sfp) path macros — for shaders without _pack4 suffix.
NCNN_FP32_SCALAR_PREAMBLE = r"""
#define sfp float
#define sfpvec2 vec2
#define sfpvec4 vec4
#define sfpmat4 mat4
#define afp float
#define afpvec2 vec2
#define afpvec4 vec4
#define afpmat4 mat4
#define sfp2afp(v) (v)
#define afp2sfp(v) (v)
#define lfp float
#define lfp2afp(v) float(v)
#define lfp2afpvec2(v) vec2(v)
#define lfp2afpvec4(v) vec4(v)
#define buffer_ld1(buf,i) float(buf[i])
#define buffer_ld2(buf,i) vec2(buf[i])
#define buffer_ld4(buf,i) buf[i]
#define buffer_st1(buf,i,v) {buf[i]=float(v);}
#define buffer_st2(buf,i,v) {buf[i]=vec2(v);}
#define buffer_st4(buf,i,v) {buf[i]=v;}
#define buffer_cp1(buf,i,sbuf,si) {buf[i]=sbuf[si];}
#define buffer_cp2(buf,i,sbuf,si) {buf[i]=sbuf[si];}
#define buffer_cp4(buf,i,sbuf,si) {buf[i]=sbuf[si];}
#define buffer_cp1to4(buf,i,sbuf,si4) {buf[i].r=sbuf[si4.r];buf[i].g=sbuf[si4.g];buf[i].b=sbuf[si4.b];buf[i].a=sbuf[si4.a];}
#define buffer_cp4to1(buf,i4,sbuf,si) {buf[i4.r]=sbuf[si].r;buf[i4.g]=sbuf[si].g;buf[i4.b]=sbuf[si].b;buf[i4.a]=sbuf[si].a;}
#define buffer_sm1(buf,i) buf[i]
#define buffer_sm4(buf,i) buf[i]
#define psc(x) (x==0?p.x:x)
#define i8buffer_ld4(buf,i) unpackInt4x8(buf[i])
#define i8buffer_st4(buf,i,v) {buf[i]=packInt4x8(v);}
#define i8buffer_cp1(buf,i,sbuf,si) {buf[i]=sbuf[si];}
#define i8buffer_cp4(buf,i,sbuf,si) {buf[i]=sbuf[si];}
#define i8buffer_ld1(buf,i) int(buf[i])
#define i8buffer_st1(buf,i,v) {buf[i]=int8_t(v);}
#define sint8 int8_t
#define aint8 int
#define sint8vec4 ivec4
#define aint8vec4 ivec4
#define pack32(x) ((x).r | ((x).g << 8) | ((x).b << 16) | ((x).a << 24))
#define pack16(x) pack32(uvec2(x.r, x.g))
#define pack8(x) pack32(x)
#define unpack16(x) uvec4((x) & 0xFF, ((x) >> 8) & 0xFF, ((x) >> 16) & 0xFF, ((x) >> 24) & 0xFF)
#define unpack8(x) unpack16(x)
#define sfp2afpmat4(v) v
#define afp2sfpmat4(v) v
#define sfp2afpvec4(v) vec4(v)
#define afp2sfpvec4(v) v
#define afp2lfp(v) v
#define lfp2afpvec2(v) vec2(v)
#define sfp2afpvec2(v) vec2(v)
#define afp2sfpvec2(v) v
#define buffer_sm1(buf,i) buf[i]
#define buffer_sm4(buf,i) buf[i]
#define buffer_sm4(buf,i) buf[i]
#define psc(x) (x==0?p.x:x)
#define PAD 0
#define UNPACK_N 0
"""

# int8 macros for quantize/requantize/packing_int8 path
NCNN_INT8_PREAMBLE = r"""
#define sint8 int8_t
#define aint8 int
#define sint8vec4 ivec4
#define aint8vec4 ivec4
#define i8buffer_ld4(buf,i) unpackInt4x8(buf[i])
#define i8buffer_st4(buf,i,v) {buf[i]=packInt4x8(v);}
#define i8buffer_cp1(buf,i,sbuf,si) {buf[i]=sbuf[si];}
#define i8buffer_cp4(buf,i,sbuf,si) {buf[i]=sbuf[si];}
#define i8buffer_ld1(buf,i) int(buf[i])
#define i8buffer_st1(buf,i,v) {buf[i]=int8_t(v);}
#define pack32(x) ((x).r | ((x).g << 8) | ((x).b << 16) | ((x).a << 24))
#define pack16(x) pack32(uvec2(x.r, x.g))
#define pack8(x) pack32(x)
#define unpack16(x) uvec4((x) & 0xFF, ((x) >> 8) & 0xFF, ((x) >> 16) & 0xFF, ((x) >> 24) & 0xFF)
#define unpack8(x) unpack16(x)
#define unpackInt4x8(v) ivec4((v) & 0xFF, ((v) >> 8) & 0xFF, ((v) >> 16) & 0xFF, ((v) >> 24) & 0xFF)
#define packInt4x8(v) (int((uint(v.r) & 0xFF) | ((uint(v.g) & 0xFF) << 8) | ((uint(v.b) & 0xFF) << 16) | ((uint(v.a) & 0xFF) << 24)))
"""

# Additional macros for pack4 path (sfpvec4 based, vec4 storage)
NCNN_PACK4_EXTRA = r"""
#define buffer_ld4(buf,i) buf[i]
#define buffer_st4(buf,i,v) {buf[i]=v;}
"""

LICENSE_HEADER = (
    "// Baked from ncnn historical shader. SPDX-License-Identifier: BSD-3-Clause.\n"
    "// FP32 path: sfp=float, afp=float, macros expanded. See bake_ncnn_shaders.py.\n"
)

# Content of vulkan_activation.comp include, with NCNN_moltenvk guards stripped.
VULKAN_ACTIVATION_CONTENT = """
afp activation_afp(afp v, int activation_type, float activation_param_0, float activation_param_1)
{
    if (activation_type == 1) { v = max(v, afp(0.f)); }
    if (activation_type == 2) { const afp slope = afp(activation_param_0); v = v < afp(0.f) ? v * slope : v; }
    if (activation_type == 3) { const afp const_min = afp(activation_param_0); const afp const_max = afp(activation_param_1); v = clamp(v, const_min, const_max); }
    if (activation_type == 4) { v = afp(1.f) / (afp(1.f) + exp(-v)); }
    if (activation_type == 5) { v = v * tanh(log(exp(v) + afp(1.f))); }
    if (activation_type == 6) { const afp alpha = afp(activation_param_0); const afp beta = afp(activation_param_1); v = v * clamp(v * afp(alpha) + afp(beta), afp(0.f), afp(1.f)); }
    return v;
}

afpvec4 activation_afpvec4(afpvec4 v, int activation_type, float activation_param_0, float activation_param_1)
{
    if (activation_type == 1) { v = max(v, afp(0.f)); }
    if (activation_type == 2) { const afp slope = afp(activation_param_0); v = mix(v, v * slope, lessThan(v, afpvec4(0.f))); }
    if (activation_type == 3) { const afp const_min = afp(activation_param_0); const afp const_max = afp(activation_param_1); v = clamp(v, const_min, const_max); }
    if (activation_type == 4) { v = afp(1.f) / (afp(1.f) + exp(-v)); }
    if (activation_type == 5) { v = v * tanh(log(exp(v) + afp(1.f))); }
    if (activation_type == 6) { const afp alpha = afp(activation_param_0); const afp beta = afp(activation_param_1); v = v * clamp(v * afp(alpha) + afp(beta), afp(0.f), afp(1.f)); }
    return v;
}
"""


def expand_includes(source):
    """Inline #include directives with known shader content."""
    source = source.replace('#include "vulkan_activation.comp"',
                             VULKAN_ACTIVATION_CONTENT)
    return source

# Files requiring cooperative matrix support — these need ncnn_VK_KHR_cooperative_matrix
# or ncnn_VK_NV_cooperative_matrix at runtime. We bake them with the cooperative matrix
# path disabled (FP32 scalar fallback) where possible, or skip if no fallback exists.
# These are NOT skipped — they are baked with cooperative matrix guards removed.
COMPLEX_CM_FILES = {
    "convolution1d_1x1s1d1_cm.comp",
    "convolution1d_gemm_cm.comp",
    "convolution_1x1s1d1_cm.comp",
    "convolution_gemm_cm.comp",
    "convolution_winograd_gemm_cm.comp",
    "deconvolution_gemm_cm.comp",
    "gemm_cm.comp",
    "sdpa_cross_cm.comp",
    "sdpa_fa_cm.comp",
}


def is_pack4(filename):
    """Determine if shader uses pack4 (vec4) storage by filename."""
    return "pack4" in filename and "pack4to1" not in filename


def strip_feature_guards(source):
    """Remove #if NCNN_fp16_* / #if NCNN_int8_* feature guards.

    These guards toggle fp16/int8 storage. For FP32 baked path we remove
    the guard and keep only the #else (FP32) branch, or remove the entire
    block if no #else.
    """
    # Remove #if NCNN_fp16_storage / NCNN_fp16_arithmetic / NCNN_fp16_packed
    # / NCNN_fp16_uniform / NCNN_int8_* blocks, keeping #else branch.
    result = []
    lines = source.split("\n")
    skip_depth = 0
    keep_else_depth = -1
    for line in lines:
        stripped = line.strip()
        if skip_depth > 0:
            if stripped.startswith("#if"):
                skip_depth += 1
            elif stripped == "#endif":
                skip_depth -= 1
            elif stripped == "#else" and skip_depth == 1:
                skip_depth = 0
                keep_else_depth = 1
            continue
        if keep_else_depth > 0:
            if stripped.startswith("#if"):
                keep_else_depth += 1
            elif stripped.startswith("#endif"):
                keep_else_depth -= 1
                continue
            elif stripped == "#else":
                keep_else_depth = 0
                skip_depth = 1
                continue
            result.append(line)
            continue
        if re.match(r"#if\s+NCNN_fp16_", stripped) or re.match(r"#if\s+NCNN_int8_", stripped):
            # Check if there's an #else later
            skip_depth = 1
            continue
        if re.match(r"#if\s+NCNN_image_array", stripped):
            skip_depth = 1
            continue
        if re.match(r"#if\s+NCNN_moltenvk", stripped):
            skip_depth = 1
            continue
        if re.match(r"#if\s+ncnn_VK_", stripped):
            # Cooperative matrix / other Vulkan extension guards — remove entirely
            # (keep #elif/#else FP32 fallback if present, otherwise remove block)
            skip_depth = 1
            continue
        result.append(line)
    return "\n".join(result)


def bake_shader(source_path, output_path, is_pack4):
    """Bake a single .comp shader."""
    source = source_path.read_text(encoding="utf-8")
    # Strip feature guards
    source = strip_feature_guards(source)
    # Expand #include directives
    source = expand_includes(source)
    # Remove license comment block (first few // lines)
    lines = source.split("\n")
    while lines and (lines[0].strip().startswith("//") or lines[0].strip() == ""):
        lines.pop(0)
    source = "\n".join(lines)
    # Build preamble
    preamble = LICENSE_HEADER
    preamble += NCNN_FP32_SCALAR_PREAMBLE
    preamble += NCNN_INT8_PREAMBLE
    if is_pack4:
        preamble += NCNN_PACK4_EXTRA
    # Insert preamble after #version line
    version_match = re.search(r"^#version\s+\d+", source, re.MULTILINE)
    if version_match:
        pos = version_match.end()
        # Find end of line
        eol = source.find("\n", pos)
        if eol == -1:
            eol = len(source)
        result = source[:eol + 1] + preamble + source[eol + 1:]
    else:
        result = preamble + source
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result, encoding="utf-8")
    return True


def get_op_type(filename):
    """Derive op_type from filename by stripping suffixes."""
    stem = filename.replace(".comp", "")
    # Strip pack variants
    for suffix in ["_pack4to1", "_pack1to4", "_pack4"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, help="Corpus sources/ root")
    parser.add_argument("--operators", required=True, help="Corpus operators/ root")
    args = parser.parse_args(argv)

    sources_root = Path(args.sources)
    operators_root = Path(args.operators)

    for tag in ["20190611", "20260526"]:
        shader_dir = sources_root / "ncnn" / tag / "src/layer/vulkan/shader"
        if not shader_dir.is_dir():
            continue
        for comp_file in sorted(shader_dir.glob("*.comp")):
            op_type = get_op_type(comp_file.name)
            pack4 = is_pack4(comp_file.name)
            # Variant name = file stem (without .comp)
            variant = comp_file.stem
            # For _cm files, cooperative matrix guards are stripped (no coop mat support)
            out_dir = operators_root / "vulkan" / op_type / "ncnn" / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / (variant + ".comp")
            try:
                bake_shader(comp_file, out_file, pack4)
                print(f"OK: {out_file.relative_to(operators_root)}")
            except Exception as e:
                print(f"FAIL: {comp_file.name}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
