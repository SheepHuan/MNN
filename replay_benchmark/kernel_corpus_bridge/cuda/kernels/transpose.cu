// transpose.cu - NHWC_2_NCHW / NCHW_2_NHWC (3.6.0) + NCHW_2_NHWC_212 (2.1.2)
//   kernels + shims
//   source/backend/cuda/execution/Transpose.cu
#include "corpus_common.cuh"

namespace MNN {
namespace Corpus {

// ============================================================================
// Transpose format conversion: source/backend/cuda/execution/Transpose.cu
// NHWC_2_NCHW and NCHW_2_NHWC are the simplest (no DivModFast dependency).
// ============================================================================
template <typename T0, typename T1>
__global__ void NHWC_2_NCHW(const T0* input, T1* output,
                            int total, int inside, int axis, int outside) {
    CUDA_KERNEL_LOOP(index, total) {
        int x = index % inside;
        int y = (index / inside) % axis;
        int z = index / (inside * axis);
        output[z * axis * inside + y * inside + x] = input[index];
    }
}
template <typename T0, typename T1>
__global__ void NCHW_2_NHWC(const T0* input, T1* output,
                            int total, int inside, int axis, int outside) {
    CUDA_KERNEL_LOOP(index, total) {
        int x = index % inside;
        int y = (index / inside) % axis;
        int z = index / (inside * axis);
        output[index] = input[z * axis * inside + y * inside + x];
    }
}

// ---- NCHW_2_NHWC 2.1.2: src_offset uses channel (not inChannelPack) ----
template <typename T0, typename T1>
__global__ void NCHW_2_NHWC_212(const T0* input, T1* output, const int maxCount, const int channel,
                                 const int area, const int channel_pack,
                                 MNN::Corpus::DivModFast d_oc, MNN::Corpus::DivModFast d_area) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        d_oc.divmod(index, temp, chnl_idx);
        d_area.divmod(temp, batch_idx, area_idx);
        int src_offset = (batch_idx * channel + chnl_idx) * area + area_idx;
        output[index] = (T1)input[src_offset];
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- Transpose format conversion ----
void mnn_corpus_nhwc2nchw_fp32(const float* input, float* output, int total, int inside, int axis, int outside,
                                int grid, int block, cudaStream_t stream) {
    MNN::Corpus::NHWC_2_NCHW<float, float><<<grid, block, 0, stream>>>(input, output, total, inside, axis, outside);
}
void mnn_corpus_nchw2nhwc_fp32(const float* input, float* output, int total, int inside, int axis, int outside,
                                int grid, int block, cudaStream_t stream) {
    MNN::Corpus::NCHW_2_NHWC<float, float><<<grid, block, 0, stream>>>(input, output, total, inside, axis, outside);
}

// ---- Transpose fp16 ----
void mnn_corpus_nhwc2nchw_fp16(const void* input, void* output, int total, int inside, int axis, int outside,
                                int grid, int block, cudaStream_t stream) {
    MNN::Corpus::NHWC_2_NCHW<half, half><<<grid, block, 0, stream>>>((const half*)input, (half*)output, total, inside, axis, outside);
}
void mnn_corpus_nchw2nhwc_fp16(const void* input, void* output, int total, int inside, int axis, int outside,
                                int grid, int block, cudaStream_t stream) {
    MNN::Corpus::NCHW_2_NHWC<half, half><<<grid, block, 0, stream>>>((const half*)input, (half*)output, total, inside, axis, outside);
}

// ---- NCHW_2_NHWC 2.1.2: src_offset uses channel ----
void mnn_corpus_nhwc2nchw_212_fp32(const float* input, float* output, int total, int channel, int area, int channel_pack,
                                    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(channel_pack);
    MNN::Corpus::DivModFast d_area(area);
    MNN::Corpus::NCHW_2_NHWC_212<float, float><<<grid, block, 0, stream>>>(input, output, total, channel, area, channel_pack, d_oc, d_area);
}

} // extern "C"
