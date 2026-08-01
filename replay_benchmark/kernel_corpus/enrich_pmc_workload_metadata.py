#!/usr/bin/env python3
"""为可证明的 operator case 写入 shape 与理论 workload 元数据。

这里只描述算法语义，不读取 PMC，也不把 allocation bytes、launch geometry 或设备资源
冒充 algorithmic workload。无法可靠给出 FLOPs 的数据搬运/比较类 kernel 会显式记录
``not_applicable`` provenance，并省略数值字段。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path


SUPPORTED_OP_TYPES = (
    "avgpool",
    "conv_dw",
    "matmul",
    "maxpool",
    "reduction",
    "softmax",
    "transpose",
)

PROVENANCE_SOURCE = "replay_benchmark_op_adapter_contract"
PROVENANCE_METHOD = "semantic_formula"
THEORETICAL_STATUS = "theoretical_estimate"

# These variants convert between fp32 and fp16 storage.  A single inferred
# dtype would under-count one side of the transfer, so leave them unannotated
# until the schema carries per-tensor dtypes.
MIXED_HALF_TRANSPOSE_VARIANTS = frozenset({
    "cuda_packcommon_half_4_fp32",
    "cuda_packcommon_rearrange_half_4_fp32",
    "cuda_unpackcommon_rearrange_half_4_fp32",
})

REDUCTION_VARIANTS = frozenset({
    "cuda_reduction_max_fp16",
    "cuda_reduction_max_fp32",
    "cuda_reduction_mean_120_fp32",
    "cuda_reduction_mean_axis_fp16",
    "cuda_reduction_mean_axis_fp32",
    "cuda_reduction_mean_fp16",
    "cuda_reduction_mean_fp32",
    "cuda_reduction_min_fp16",
    "cuda_reduction_min_fp32",
    "cuda_reduction_prod_fp16",
    "cuda_reduction_prod_fp32",
    "cuda_reduction_sum_120_fp32",
    "cuda_reduction_sum_axis_fp16",
    "cuda_reduction_sum_axis_fp32",
    "cuda_reduction_sum_fp16",
    "cuda_reduction_sum_fp32",
})

TRANSPOSE_FORMAT_LAYOUTS = {
    "cuda_c4nhw4_2_nchw_fp32": ("c4nhw4", "nchw"),
    "cuda_c4nhw4_2_nhwc8_fp32": ("c4nhw4", "nhwc8"),
    "cuda_c4nhw4_2_nhwc_fp32": ("c4nhw4", "nhwc"),
    "cuda_nchw_2_c4nhw4_fp32": ("nchw", "c4nhw4"),
    "cuda_nchw_2_nhwc8_fp32": ("nchw", "nhwc8"),
    "cuda_nhwc8_2_c4nhw4_fp32": ("nhwc8", "c4nhw4"),
    "cuda_nhwc8_2_nchw_fp32": ("nhwc8", "nchw"),
    "cuda_nhwc8_2_nhwc_fp32": ("nhwc8", "nhwc"),
    "cuda_nhwc_2_c4nhw4_fp32": ("nhwc", "c4nhw4"),
    "cuda_nhwc_2_nhwc8_fp32": ("nhwc", "nhwc8"),
}

TRANSPOSE_PACK_LAYOUTS = {
    "cuda_packcommon_fp32": ("nchw", "packed_channel_last"),
    "cuda_packcommon_4_fp32": ("nchw", "packed_channel_last"),
    "cuda_unpackcommon_fp32": ("packed_channel_last", "nchw"),
    "cuda_unpackcommon_4_fp32": ("packed_channel_last", "nchw"),
}


def _params(case):
    result = dict(case.get("int_params") or {})
    result.update(case.get("float_params") or {})
    return result


def _dtype(case):
    declared = str(case.get("dtype") or "").lower()
    if declared and declared != "unknown":
        return declared
    text = "{} {}".format(case.get("variant", ""), case.get("name", "")).lower()
    if "bf16" in text or "bfloat16" in text:
        return "bfloat16"
    if "fp16" in text or "half" in text:
        return "float16"
    if "uint8" in text:
        return "uint8"
    if "int8" in text:
        return "int8"
    if "fp32" in text or "float" in text:
        return "float32"
    return "unknown"


def _element_bytes(dtype):
    return {
        "bfloat16": 2,
        "float16": 2,
        "float32": 4,
        "int8": 1,
        "uint8": 1,
        "int32": 4,
    }.get(dtype)


def _positive_int(params, key, default=None):
    raw = params.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = int(raw)
    return value if value > 0 else None


def _product(shape):
    return math.prod(shape)


def _provenance(formula, *, status=THEORETICAL_STATUS, notes=""):
    return {
        "status": status,
        "source": PROVENANCE_SOURCE,
        "method": PROVENANCE_METHOD,
        "formula": formula,
        "confidence": 1.0,
        "notes": notes,
    }


def _result(
    dtype,
    shapes,
    workload,
    workload_formulas,
    *,
    shape_notes=None,
    workload_notes=None,
    flops_not_applicable="",
):
    shape_notes = shape_notes or {}
    workload_notes = workload_notes or {}
    shape_provenance = {
        name: _provenance(
            "adapter-resolved logical tensor shape",
            notes=shape_notes.get(name, ""),
        )
        for name in shapes
    }
    workload_provenance = {
        name: _provenance(formula, notes=workload_notes.get(name, ""))
        for name, formula in workload_formulas.items()
    }
    if flops_not_applicable:
        workload_provenance["algorithmic_flops"] = _provenance(
            "",
            status="not_applicable",
            notes=flops_not_applicable,
        )
    return {
        "dtype": dtype,
        "shapes": shapes,
        "workload": workload,
        "shape_provenance": shape_provenance,
        "workload_provenance": workload_provenance,
    }


def _describe_matmul(case, params, dtype):
    variant = str(case.get("variant") or "")
    if variant == "cuda_general_batch_matmul_fp32":
        batch = _positive_int(params, "batch", 2)
        m = _positive_int(params, "m", 4)
        k = _positive_int(params, "k", 4)
        n = _positive_int(params, "n", 4)
    elif variant == "cuda_matmul_gemv_fp32":
        batch = _positive_int(params, "batch", 2)
        m = 1
        k = _positive_int(params, "k", 128)
        n = _positive_int(params, "n", 128)
    else:
        return None
    if None in (batch, m, k, n):
        return None
    scalar_bytes = _element_bytes(dtype)
    if scalar_bytes is None:
        return None
    a_shape = [batch, m, k]
    b_shape = [batch, k, n]
    output_shape = [batch, m, n]
    output_elements = _product(output_shape)
    return _result(
        dtype,
        {"input0": a_shape, "input1": b_shape, "output0": output_shape},
        {
            "output_elements": output_elements,
            "algorithmic_flops": 2 * batch * m * n * k,
            "algorithmic_bytes": scalar_bytes * (
                _product(a_shape) + _product(b_shape) + output_elements
            ),
        },
        {
            "output_elements": "batch * m * n",
            "algorithmic_flops": "2 * batch * m * n * k",
            "algorithmic_bytes": "(A elements + B elements + output elements) * scalar_bytes",
        },
    )


def _describe_softmax(case, params, dtype):
    variant = str(case.get("variant") or "")
    if variant not in {
        "cuda_softmax_fp32",
        "cuda_softmax_fp16",
        "cuda_softmax_warp32_fp32",
        "cuda_softmax_axis_reduce_fp32",
    }:
        return None
    outside = _positive_int(params, "outside", 4)
    axis = _positive_int(params, "axis", 16)
    inside = _positive_int(params, "inside", 1)
    scalar_bytes = _element_bytes(dtype)
    if None in (outside, axis, inside) or scalar_bytes is None:
        return None
    shape = [outside, axis, inside]
    elements = _product(shape)
    vectors = outside * inside
    return _result(
        dtype,
        {"input0": shape, "output0": shape},
        {
            "output_elements": elements,
            "algorithmic_flops": vectors * (3 * axis - 1),
            "algorithmic_bytes": 2 * elements * scalar_bytes,
        },
        {
            "output_elements": "outside * axis * inside",
            "algorithmic_flops": "outside * inside * (3 * axis - 1)",
            "algorithmic_bytes": "(input elements + output elements) * scalar_bytes",
        },
        workload_notes={
            "algorithmic_flops": (
                "lower bound only; exponential and max-comparison work are not counted"
            ),
        },
    )


def _describe_reduction(case, params, dtype):
    variant = str(case.get("variant") or "")
    if variant not in REDUCTION_VARIANTS:
        return None
    outside = _positive_int(params, "outside", 4)
    axis = _positive_int(params, "axis", 16)
    inside = _positive_int(params, "inside", 1)
    scalar_bytes = _element_bytes(dtype)
    if None in (outside, axis, inside) or scalar_bytes is None:
        return None
    input_shape = [outside, axis, inside]
    output_shape = [outside, inside]
    output_elements = _product(output_shape)
    workload = {
        "output_elements": output_elements,
        "algorithmic_bytes": scalar_bytes * (
            _product(input_shape) + output_elements
        ),
    }
    formulas = {
        "output_elements": "outside * inside",
        "algorithmic_bytes": "(input elements + output elements) * scalar_bytes",
    }
    not_applicable = ""
    if "sum" in variant or "prod" in variant:
        workload["algorithmic_flops"] = output_elements * (axis - 1)
        formulas["algorithmic_flops"] = "outside * inside * (axis - 1)"
    elif "mean" in variant:
        workload["algorithmic_flops"] = output_elements * axis
        formulas["algorithmic_flops"] = "outside * inside * ((axis - 1) additions + 1 division)"
    else:
        not_applicable = "min/max reduction is comparison work, not floating-point arithmetic"
    return _result(
        dtype,
        {"input0": input_shape, "output0": output_shape},
        workload,
        formulas,
        flops_not_applicable=not_applicable,
    )


def _pool_valid_samples(ih, iw, oh, ow, kernel, stride, pad):
    total = 0
    for oy in range(oh):
        for ox in range(ow):
            for fy in range(kernel):
                for fx in range(kernel):
                    iy = oy * stride - pad + fy
                    ix = ox * stride - pad + fx
                    if 0 <= iy < ih and 0 <= ix < iw:
                        total += 1
    return total


def _describe_pool(case, params, dtype, *, average):
    variant = str(case.get("variant") or "")
    allowed = (
        {"cuda_avgpool_fp32", "cuda_avgpool_120_fp32"}
        if average
        else {"cuda_maxpool_fp32", "cuda_maxpool_120_fp32"}
    )
    if variant not in allowed:
        return None
    batch = _positive_int(params, "batch", 1)
    channels = _positive_int(params, "channels") or _positive_int(params, "c", 8)
    ih = _positive_int(params, "h", 8)
    iw = _positive_int(params, "w", 8)
    kernel = _positive_int(params, "kernel_size", 3)
    stride = _positive_int(params, "stride", 2)
    pad = int(params.get("pad", 1))
    scalar_bytes = _element_bytes(dtype)
    if None in (batch, channels, ih, iw, kernel, stride) or scalar_bytes is None:
        return None
    oh = (ih + 2 * pad - kernel) // stride + 1
    # CudaAvgPoolFp32Kernel resolves ow from oh. The dedicated 1.2.0
    # adapter resolves width independently, as do both max-pool adapters.
    if average and variant == "cuda_avgpool_fp32":
        ow = oh
    else:
        ow = (iw + 2 * pad - kernel) // stride + 1
    if oh <= 0 or ow <= 0:
        return None
    legacy_nchw = (
        str(case.get("tag") or "") == "1.2.0"
        or variant in {"cuda_avgpool_120_fp32", "cuda_maxpool_120_fp32"}
    )
    if legacy_nchw:
        input_shape = [batch, channels, ih, iw]
        output_shape = [batch, channels, oh, ow]
        layout_note = (
            "1.2.0 pool adapter indexes tensors as NCHW "
            "[batch, channel, height, width]"
        )
    else:
        input_shape = [batch, ih, iw, channels]
        output_shape = [batch, oh, ow, channels]
        layout_note = (
            "C8 pool adapter indexes tensors in channel-last physical order "
            "[batch, height, width, packed_channel]"
        )
    output_elements = _product(output_shape)
    workload = {
        "output_elements": output_elements,
        "algorithmic_bytes": scalar_bytes * (
            _product(input_shape) + output_elements
        ),
    }
    formulas = {
        "output_elements": "batch * output_h * output_w * channels",
        "algorithmic_bytes": "(input elements + output elements) * scalar_bytes",
    }
    not_applicable = ""
    if average:
        valid_samples = _pool_valid_samples(ih, iw, oh, ow, kernel, stride, pad)
        workload["algorithmic_flops"] = batch * channels * valid_samples
        formulas["algorithmic_flops"] = "sum(valid samples + one division per output window)"
    else:
        not_applicable = "max pooling performs comparisons rather than floating-point arithmetic"
    return _result(
        dtype,
        {"input0": input_shape, "output0": output_shape},
        workload,
        formulas,
        shape_notes={"input0": layout_note, "output0": layout_note},
        flops_not_applicable=not_applicable,
    )


def _describe_conv_dw(case, params, dtype):
    if str(case.get("variant") or "") != "cuda_conv_dw_fp32":
        return None
    iw = _positive_int(params, "iw", 4)
    ih = _positive_int(params, "ih", 4)
    channels = _positive_int(params, "channels", 8)
    kw = _positive_int(params, "kw", 3)
    kh = kw
    sw = _positive_int(params, "sw", 1)
    sh = sw
    pw = int(params.get("pw", 1))
    ph = pw
    scalar_bytes = _element_bytes(dtype)
    if None in (iw, ih, channels, kw, kh, sw, sh) or scalar_bytes is None:
        return None
    ow = (iw + 2 * pw - kw) // sw + 1
    oh = (ih + 2 * ph - kh) // sh + 1
    if oh <= 0 or ow <= 0:
        return None
    tag = str(case.get("tag") or "")
    if tag == "1.2.0":
        input_shape = [1, channels, ih, iw]
        weight_shape = [channels, kh, kw]
        output_shape = [1, channels, oh, ow]
        input_layout_note = (
            "MNN 1.2.0 depthwise adapter indexes input/output as NCHW"
        )
        weight_layout_note = (
            "MNN 1.2.0 depthwise weight layout is "
            "[channel, kernel_h, kernel_w]"
        )
    elif tag == "2.0.4":
        input_shape = [1, ih, iw, channels]
        weight_shape = [channels, kh, kw]
        output_shape = [1, oh, ow, channels]
        input_layout_note = (
            "MNN 2.0.4 depthwise adapter indexes input/output in channel-last order"
        )
        weight_layout_note = (
            "MNN 2.0.4 depthwise weight layout is "
            "[channel, kernel_h, kernel_w]"
        )
    else:
        input_shape = [1, ih, iw, channels]
        weight_shape = [kh, kw, channels]
        output_shape = [1, oh, ow, channels]
        input_layout_note = (
            "MNN 2.2.3 and later depthwise adapter indexes input/output "
            "in channel-last order"
        )
        weight_layout_note = (
            "MNN 2.2.3 and later prepared weight layout is "
            "[kernel_h, kernel_w, packed_channel]"
        )
    bias_shape = [channels]
    output_elements = _product(output_shape)
    input_elements = _product(input_shape)
    weight_elements = _product(weight_shape)
    bias_elements = _product(bias_shape)
    if tag == "1.2.0":
        weight_bias_bytes = scalar_bytes * (weight_elements + bias_elements)
        byte_formula = (
            "(input + output elements) * scalar_bytes + "
            "(weight + bias elements) * fp32_bytes"
        )
        byte_note = "MNN 1.2.0 stores depthwise weight and bias as fp32"
    else:
        weight_bias_bytes = 2 * (weight_elements + bias_elements)
        byte_formula = (
            "(input + output elements) * scalar_bytes + "
            "(weight + bias elements) * fp16_bytes"
        )
        byte_note = (
            "MNN 2.0.4 and later adapters store depthwise weight and bias as fp16"
        )
    algorithmic_bytes = (
        scalar_bytes * (input_elements + output_elements) + weight_bias_bytes
    )
    return _result(
        dtype,
        {
            "input0": input_shape,
            "weight0": weight_shape,
            "bias0": bias_shape,
            "output0": output_shape,
        },
        {
            "output_elements": output_elements,
            "algorithmic_flops": 2 * output_elements * kh * kw,
            "algorithmic_bytes": algorithmic_bytes,
        },
        {
            "output_elements": "output_h * output_w * channels",
            "algorithmic_flops": "2 * output_elements * kernel_h * kernel_w",
            "algorithmic_bytes": byte_formula,
        },
        shape_notes={
            "input0": input_layout_note,
            "weight0": weight_layout_note,
            "bias0": "depthwise bias has one scalar per channel",
            "output0": input_layout_note,
        },
        workload_notes={
            "algorithmic_flops": (
                "FMA is counted as 2 FLOPs; bias and activation/clamp work are not counted"
            ),
            "algorithmic_bytes": (
                byte_note + "; adapter configuration buffers are not counted"
            ),
        },
    )


def _layout_shape(layout, batch, channels, area):
    if layout in {"nchw", "c4nhw4"}:
        return [batch, channels, area]
    if layout in {"nhwc", "nhwc8", "packed_channel_last"}:
        return [batch, area, channels]
    raise ValueError("unsupported logical layout {}".format(layout))


def _transpose_shapes(case, params):
    variant = str(case.get("variant") or "")
    if variant in {"cuda_nhwc2nchw_fp32", "cuda_nhwc2nchw_fp16"}:
        outside = _positive_int(params, "outside", 2)
        axis = _positive_int(params, "axis", 4)
        inside = _positive_int(params, "inside", 3)
        if None in (outside, axis, inside):
            return None
        return (
            [outside, inside, axis],
            [outside, axis, inside],
            "NHWC logical order [outside, area, channel]",
            "NCHW logical order [outside, channel, area]",
        )
    if variant in {"cuda_nchw2nhwc_fp32", "cuda_nchw2nhwc_fp16"}:
        outside = _positive_int(params, "outside", 2)
        axis = _positive_int(params, "axis", 4)
        inside = _positive_int(params, "inside", 3)
        if None in (outside, axis, inside):
            return None
        return (
            [outside, axis, inside],
            [outside, inside, axis],
            "NCHW logical order [outside, channel, area]",
            "NHWC logical order [outside, area, channel]",
        )
    if variant in {"cuda_transpose_fp32", "cuda_transpose_local_fp32"}:
        m = _positive_int(params, "m", 4)
        n = _positive_int(params, "n", 4)
        if None in (m, n):
            return None
        return (
            [m, n],
            [n, m],
            "matrix input shape [m, n]",
            "matrix transpose output shape [n, m]",
        )
    if variant == "cuda_transpose_bdl_to_bld_fp32":
        batch = _positive_int(params, "batch", 2)
        d = _positive_int(params, "d", 4)
        length = _positive_int(params, "l", 4)
        if None in (batch, d, length):
            return None
        return (
            [batch, d, length],
            [batch, length, d],
            "linear-attention input order [batch, dimension, length]",
            "linear-attention output order [batch, length, dimension]",
        )
    if variant == "cuda_nchw2nchw_fp32":
        count = _positive_int(params, "count")
        if count is None:
            return None
        return (
            [count],
            [count],
            "flat NCHW identity-copy extent",
            "flat NCHW identity-copy extent",
        )
    layouts = TRANSPOSE_FORMAT_LAYOUTS.get(variant)
    if layouts is None:
        layouts = TRANSPOSE_PACK_LAYOUTS.get(variant)
    if layouts is not None:
        batch = _positive_int(params, "batch", 1)
        channels = _positive_int(params, "c", 4)
        area = _positive_int(params, "area", 4)
        if None in (batch, channels, area):
            return None
        input_layout, output_layout = layouts
        note_suffix = (
            "; shape records the unpadded logical extent, not packed allocation bytes"
        )
        return (
            _layout_shape(input_layout, batch, channels, area),
            _layout_shape(output_layout, batch, channels, area),
            "{} logical layout{}".format(input_layout, note_suffix),
            "{} logical layout{}".format(output_layout, note_suffix),
        )
    return None


def _describe_transpose(case, params, dtype):
    variant = str(case.get("variant") or "")
    if variant in MIXED_HALF_TRANSPOSE_VARIANTS:
        return None
    shapes = _transpose_shapes(case, params)
    scalar_bytes = _element_bytes(dtype)
    if shapes is None or scalar_bytes is None:
        return None
    input_shape, output_shape, input_note, output_note = shapes
    elements = _product(input_shape)
    if elements != _product(output_shape):
        return None
    return _result(
        dtype,
        {"input0": input_shape, "output0": output_shape},
        {
            "output_elements": elements,
            "algorithmic_bytes": 2 * elements * scalar_bytes,
        },
        {
            "output_elements": "logical tensor element count",
            "algorithmic_bytes": "(input elements + output elements) * scalar_bytes",
        },
        shape_notes={"input0": input_note, "output0": output_note},
        flops_not_applicable="layout transformation is data movement, not floating-point arithmetic",
    )


def describe_case(case):
    """返回可审计的 workload 描述；不支持或不确定时返回 ``None``。"""

    if case.get("backend") != "cuda":
        return None
    op_type = str(case.get("op_type") or "")
    params = _params(case)
    dtype = _dtype(case)
    if op_type == "matmul":
        return _describe_matmul(case, params, dtype)
    if op_type == "softmax":
        return _describe_softmax(case, params, dtype)
    if op_type == "reduction":
        return _describe_reduction(case, params, dtype)
    if op_type == "avgpool":
        return _describe_pool(case, params, dtype, average=True)
    if op_type == "maxpool":
        return _describe_pool(case, params, dtype, average=False)
    if op_type == "conv_dw":
        return _describe_conv_dw(case, params, dtype)
    if op_type == "transpose":
        return _describe_transpose(case, params, dtype)
    return None


def _is_owned_provenance(value):
    return (
        isinstance(value, dict)
        and value.get("source") == PROVENANCE_SOURCE
        and value.get("method") == PROVENANCE_METHOD
    )


def _clear_owned_map(case, value_field, provenance_field):
    provenance = case.get(provenance_field)
    if not isinstance(provenance, dict):
        return
    owned_keys = [
        key for key, value in provenance.items() if _is_owned_provenance(value)
    ]
    values = case.get(value_field)
    if isinstance(values, dict):
        for key in owned_keys:
            values.pop(key, None)
        if not values:
            case.pop(value_field, None)
    for key in owned_keys:
        provenance.pop(key, None)
    if not provenance:
        case.pop(provenance_field, None)


def _clear_owned_metadata(case):
    _clear_owned_map(case, "shapes", "shape_provenance")
    _clear_owned_map(case, "workload", "workload_provenance")
    dtype_provenance = case.get("dtype_provenance")
    if _is_owned_provenance(dtype_provenance):
        case.pop("dtype", None)
        case.pop("dtype_provenance", None)


def _merge_owned_map(case, description, value_field, provenance_field):
    proposed_values = description[value_field]
    proposed_provenance = description[provenance_field]
    existing_values = case.get(value_field)
    existing_provenance = case.get(provenance_field)
    if existing_values is not None and not isinstance(existing_values, dict):
        raise ValueError(
            "case {} field {} must be an object".format(
                case.get("name", ""), value_field
            )
        )
    if existing_provenance is not None and not isinstance(existing_provenance, dict):
        raise ValueError(
            "case {} field {} must be an object".format(
                case.get("name", ""), provenance_field
            )
        )
    if existing_values is None:
        existing_values = {}
        case[value_field] = existing_values
    if existing_provenance is None:
        existing_provenance = {}
        case[provenance_field] = existing_provenance
    for key, value in proposed_values.items():
        if key in existing_values and existing_values[key] != value:
            raise ValueError(
                "case {} has conflicting {}.{} metadata".format(
                    case.get("name", ""), value_field, key
                )
            )
        if key not in existing_values and key in existing_provenance:
            raise ValueError(
                "case {} has {}.{} provenance without the proposed value".format(
                    case.get("name", ""), provenance_field, key
                )
            )
    for key, provenance in proposed_provenance.items():
        if key in proposed_values:
            continue
        if key in existing_values:
            raise ValueError(
                "case {} has numeric {}.{} metadata but the adapter contract "
                "marks it not applicable".format(
                    case.get("name", ""), value_field, key
                )
            )
        if key in existing_provenance and (
            not isinstance(existing_provenance[key], dict)
            or existing_provenance[key].get("status") != provenance.get("status")
        ):
            raise ValueError(
                "case {} has conflicting {}.{} semantic status".format(
                    case.get("name", ""), provenance_field, key
                )
            )
    for key, value in proposed_values.items():
        # An equal value without this tool's provenance belongs to the
        # manifest author. Preserve both the value and its provenance.
        if key in existing_values:
            continue
        existing_values[key] = value
        existing_provenance[key] = proposed_provenance[key]
    # A not-applicable fact intentionally has provenance without a numeric
    # value. Keep that semantic classification unless the manifest author has
    # already supplied either a value or provenance for the same key.
    for key, provenance in proposed_provenance.items():
        if key in proposed_values:
            continue
        if key in existing_values or key in existing_provenance:
            continue
        existing_provenance[key] = provenance
    if not existing_values:
        case.pop(value_field, None)
    if not existing_provenance:
        case.pop(provenance_field, None)


def _merge_description(case, description):
    if case.get("dtype") in (None, "", "unknown"):
        case["dtype"] = description["dtype"]
        case["dtype_provenance"] = _provenance(
            "dtype inferred from the exact adapter variant"
        )
    _merge_owned_map(case, description, "shapes", "shape_provenance")
    _merge_owned_map(case, description, "workload", "workload_provenance")


def enrich_document(document):
    working = copy.deepcopy(document)
    cases = working.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("operator case document must contain a cases array")
    enriched = 0
    by_op_type = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        _clear_owned_metadata(case)
        description = describe_case(case)
        if description is None:
            continue
        _merge_description(case, description)
        enriched += 1
        key = str(case.get("op_type") or "unknown")
        by_op_type[key] = by_op_type.get(key, 0) + 1
    document.clear()
    document.update(working)
    return enriched, dict(sorted(by_op_type.items()))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output")
    output.add_argument("--in-place", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_path = Path(args.input)
    document = json.loads(input_path.read_text(encoding="utf-8"))
    enriched, by_op_type = enrich_document(document)
    output_path = input_path if args.in_place else Path(args.output)
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(
        "enriched {} cases: {}".format(
            enriched, json.dumps(by_op_type, sort_keys=True)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
