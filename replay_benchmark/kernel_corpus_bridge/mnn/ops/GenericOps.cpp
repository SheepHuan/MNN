#include "GenericOps.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

void registerGenericOps() {
    static struct Reg {
        Reg() {
            auto& reg = OpAdapterRegistry::instance();
            auto add = [&](const GenericBufAdapter::Spec& s) {
                reg.registerAdapter(std::unique_ptr<OpAdapter>(new GenericBufAdapter(s)));
            };
            using TS = GenericBufAdapter::TagSpec;

            // cast_buf: (sizeConst×2, buf, buf, scalarInt) — 3.6.0
            add({"cast", "cast_buf_fp32", {
                TS{"3.6.0", "cast_buf", 1, 1, 2},
            }});

            // unary_buf_fp32: (sizeConst×2/3, buf, buf, scalarInt) — 1.2.0 + 3.6.0
            // OPERATOR set to "in" (identity) by GenericBufAdapter
            add({"unary", "unary_buf_fp32", {
                TS{"1.2.0", "unary_buf", 1, 1, 3},
                TS{"3.6.0", "unary_buf", 1, 1, 2},
            }});

            // reduction_buf_fp32: (sizeConst×2, buf, buf, scalarInt×3) — 1.2.0 + 3.6.0
            // OPERATE set to "(num+in)" by GenericBufAdapter
            add({"reduction", "reduction_buf_fp32", {
                TS{"1.2.0", "reduct_buf", 1, 1, 2},
                TS{"3.6.0", "reduct_buf", 1, 1, 2},
            }});

            // raster_buf_fp32: (sizeConst×2, buf) — 1.2.0 + 3.6.0 (entry=buffer_set_zero)
            add({"raster", "raster_buf_fp32", {
                TS{"1.2.0", "buffer_set_zero", 0, 1, 2},
                TS{"3.6.0", "buffer_set_zero", 0, 1, 2},
            }});

            // softmax_buf_fp32: 1.2.0 (sizeConst*2, buf, buf, int*2, int4), 3.6.0 (sizeConst*2, buf, buf, int*3)
            add({"softmax", "softmax_buf_fp32", {
                TS{"1.2.0", "softmax_channel", 1, 1, 2},
                TS{"3.6.0", "softmax_in1_buf", 1, 1, 2},
            }});

            // argmax_buf: (sizeConst×2, buf, buf, scalarInt×3) — 3.6.0
            add({"argmax", "argmax_buf_fp32", {
                TS{"3.6.0", "argmax_buf", 1, 1, 2},
            }});

            // matmul_buf_fp32: (sizeConst×2, buf×2, buf, scalarInt×3) — 1.2.0 + 3.6.0
            add({"matmul", "matmul_buf_fp32", {
                TS{"1.2.0", "matmul_buf", 2, 1, 2},
                TS{"3.6.0", "matmul_buf", 2, 1, 2},
            }});

            // matmul_local_buf_fp32: (scalarInt×3, buf, buf) — 3.6.0
            add({"matmul", "matmul_local_buf_fp32", {
                TS{"3.6.0", "matmul_local_buf", 1, 1, 2},
            }});

            // topkv2_buf: (sizeConst×2, buf×2, buf, scalarInt×3) — 3.6.0
            add({"topkv2", "topkv2_buf_fp32", {
                TS{"3.6.0", "topkv2_buf", 2, 1, 2},
            }});

            // scale_buf_fp32: 1.2.0 (sizeConst×2,buf×3,buf,int4), 3.6.0 (sizeConst×2,buf×3,buf,int×3)
            add({"scale", "scale_buf_fp32", {
                TS{"1.2.0", "scale_buf", 3, 1, 2},
                TS{"3.6.0", "scale_buf", 3, 1, 2},
            }});

            // interp_buf_fp32: (sizeConst×2, buf, buf, float×4, int×5) — 1.2.0 + 3.6.0
            add({"interp", "interp_buf_fp32", {
                TS{"1.2.0", "nearest_buf", 1, 1, 2},
                TS{"3.6.0", "nearest_buf", 1, 1, 2},
            }});

            // range_buf: (sizeConst×2, buf×2, buf, int) — 3.6.0
            add({"range", "range_buf_fp32", {
                TS{"3.6.0", "range_buf", 2, 1, 2},
            }});

            // select_buf: (sizeConst×2, buf×3, buf) — 3.6.0
            add({"select", "select_buf_fp32", {
                TS{"3.6.0", "select_buf", 3, 1, 2},
            }});

            // groupnorm_buf: (sizeConst×3, buf×3, buf, int×3, buf×2, float) — 3.6.0
            add({"groupnorm", "groupnorm_buf_fp32", {
                TS{"3.6.0", "groupnorm_plain_buf", 4, 1, 3},
            }});

            // layernorm_buf: (sizeConst×2, buf, buf, int, buf×2, float) — 3.6.0
            add({"layernorm", "layernorm_buf_fp32", {
                TS{"3.6.0", "layernorm_buf", 3, 1, 2},
            }});

            // winogradTransform_buf: 1.2.0 (sizeConst×2,buf,buf,int×8), 3.6.0 (sizeConst×2,buf,buf,int×4)
            add({"winograd", "winogradTransform_buf_fp32", {
                TS{"1.2.0", "winoTransSrcBuf2_3_1", 1, 1, 2},
                TS{"3.6.0", "winoTransWeightBuf2_3_1", 1, 1, 2},
            }});

            // winogradTransform_subgroup_buf: (sizeConst×2, buf, buf, int×8) — 3.6.0
            add({"winograd", "winogradTransform_subgroup_buf_fp32", {
                TS{"3.6.0", "winoTransSrcBuf2_3_1_c16_c16", 1, 1, 2},
            }});

            // unary_subgroup_buf: (sizeConst×2, buf, buf, int×8) — 3.6.0
            add({"unary", "unary_subgroup_buf_fp32", {
                TS{"3.6.0", "unary_buf_c4_c4", 1, 1, 2},
            }});

            // input_transe_buf: (sizeConst×3, buf, buf, int×7) — 3.6.0
            add({"input_transe", "input_transe_buf_fp32", {
                TS{"3.6.0", "conv_transe_c4_c1", 1, 1, 3},
            }});

            // splitgelu_buf: (sizeConst×2, buf×2, buf, int4) — 3.6.0
            add({"splitgelu", "splitgelu_buf_fp32", {
                TS{"3.6.0", "splitgelu_buf", 2, 1, 2},
            }});

            // strassen_binary_buf: (sizeConst×2, buf×3, buf, int×2) — 3.6.0
            add({"strassen", "strassen_binary_buf_fp32", {
                TS{"3.6.0", "binary_cfunction_buf", 3, 1, 2},
            }});

            // grid_sample_buf: (sizeConst×2, buf×2, buf, int×5) — 3.6.0
            add({"grid_sample", "grid_sample_buf_fp32", {
                TS{"3.6.0", "nearest_buf", 2, 1, 2},
            }});

            // conv_2d_buf: 1.2.0 (sizeConst×2,buf×3,buf,int2,int×2,int2×4,int×2), 3.6.0 (sizeConst×2,buf×4,buf,int×6)
            add({"conv", "conv_2d_buf_fp32", {
                TS{"1.2.0", "conv_2d_c4h1w1", 3, 1, 2},
                TS{"3.6.0", "conv_2d_1x1_local", 4, 1, 2},
            }});

            // conv_2d_c1_subgroup_buf: (sizeConst×2, buf×4, buf, int×13) — 3.6.0
            add({"conv", "conv_2d_c1_subgroup_buf_fp32", {
                TS{"3.6.0", "conv_2d_buf_subgroup_c1_c4_b2", 4, 1, 2},
            }});

            // conv_2d_c16_subgroup_buf: (sizeConst×2, buf×4, buf, int×13) — 3.6.0
            add({"conv", "conv_2d_c16_subgroup_buf_fp32", {
                TS{"3.6.0", "conv_2d_buf_subgroup_c16_c4_b2", 4, 1, 2},
            }});

            // conv_2d_int_buf: (sizeConst×2, buf×5, buf, int2, int×3, int2×4, int×3, float) — 3.6.0
            add({"conv", "conv_2d_int_buf_fp32", {
                TS{"3.6.0", "conv_2d_int_c4h1w1", 5, 1, 2},
            }});

            // depthwise_conv2d_buf: (sizeConst×2, buf×4, int2, int, int2×4, int×2) — 1.2.0 + 3.6.0
            add({"depthwise_conv", "depthwise_conv2d_buf_fp32", {
                TS{"1.2.0", "depthwise_conv2d_c4h1w4", 4, 1, 2},
                TS{"3.6.0", "depthwise_conv2d_c4h1w4", 4, 1, 2},
            }});

            // depthwise_conv2d_subgroup_buf: (sizeConst×2, buf×4, int2×3, buf, int×9) — 3.6.0
            add({"depthwise_conv", "depthwise_conv2d_subgroup_buf_fp32", {
                TS{"3.6.0", "depthwise_conv_2d_buf_c16_c16", 4, 1, 2},
            }});

            // pooling_buf: handled by PoolingOp
            // Skip — dedicated adapter exists.

            // pooling_subgroup_buf: (sizeConst×2, buf, int2×3, buf×2, int×9) — 3.6.0
            add({"pooling", "pooling_subgroup_buf_fp32", {
                TS{"3.6.0", "pooling_c4_c4", 1, 2, 2},
            }});

            // gemm_buf: 1.2.0 (sizeConst×2, buf×3, int, float), 3.6.0 (sizeConst×2, int×5, buf, buf)
            add({"gemm", "gemm_buf_fp32", {
                TS{"1.2.0", "gemm_buf", 3, 0, 2},
                TS{"3.6.0", "transpose_pad", 1, 1, 2},
            }});

            // gemm_conv1x1_buf: (sizeConst×2, buf×4, int×5, float) — 3.6.0
            add({"gemm", "gemm_conv1x1_buf_fp32", {
                TS{"3.6.0", "inverse_quant_weight", 4, 0, 2},
            }});

            // gemv_conv1x1_buf: (sizeConst×2, buf×6, int×7, float) — 3.6.0
            add({"gemv", "gemv_conv1x1_buf_fp32", {
                TS{"3.6.0", "gemv_conv_c8_buf", 6, 0, 2},
            }});


            // buffer_convert_buf: 1.2.0 (sizeConst×2, buf×2, int×3, buf), 3.6.0 (sizeConst×2, buf, int4, buf)
            add({"buffer_convert_buf", "buffer_convert_buf_fp32", {
                TS{"1.2.0", "nhwc_buffer_to_nc4hw4_buffer", 2, 1, 2},
                TS{"3.6.0", "buffer_convert_to_buffer", 1, 1, 2},
            }});

            // buffer_convert_subgroup_buf: (sizeConst×2, buf, int×4, buf, int×4) — 3.6.0
            add({"buffer_convert_subgroup_buf", "buffer_convert_subgroup_buf_fp32", {
                TS{"3.6.0", "nhwc_buffer_to_nc16hw16_buffer", 1, 1, 2},
            }});

            // attention_buf: complex, simplified — 3.6.0
            add({"attention", "attention_buf_fp32", {
                TS{"3.6.0", "rearrange_qkv", 1, 3, 2},
            }});

            // self_attention_buf: complex, simplified — 3.6.0
            add({"self_attention_buf", "self_attention_buf_fp32", {
                TS{"3.6.0", "split_transpose_qkv", 1, 3, 2},
            }});

            // linear_attention_buf: (sizeConst, buf×4, buf, int×5) — 3.6.0
            add({"linear_attention_buf", "linear_attention_buf_fp32", {
                TS{"3.6.0", "linear_attn_conv_silu", 4, 1, 2},
            }});

            // matmul_params_buf: no __kernel entries — skip
        }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
