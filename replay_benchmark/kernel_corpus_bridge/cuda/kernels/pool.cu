// pool.cu - MAXPOOL/AVGPOOL/GLOBAL_AVGPOOL/GLOBAL_MAXPOOL kernels + shims
//   source/backend/cuda/execution/PoolExecution.cu
//   source/backend/cuda/execution/bf16/PoolBf16.cuh (maxpool_C8_BF16/avgpool_C8_BF16)
//
// BF16 pool kernels are copied verbatim from MNN's PoolBf16.cuh. They use
// `#if (__CUDA_ARCH__ >= 800)` guards so the kernel body compiles to an empty
// function on sm75 (no bf16 compute instructions), but the type definitions
// (__nv_bfloat16) are available via cuda_bf16.h. On sm80+ the body executes.
// Corpus defines ENABLE_CUDA_BF16 (matching MNN's MNN_CUDA_BF16 option) so
// the kernels are always present in the binary.
#define ENABLE_CUDA_BF16
#include "corpus_common.cuh"
#include <cuda_bf16.h>

// 1.2.7-era pool/TensorCore used PACK_NUMBER=16 (corpus_common.cuh does not
// define PACK_NUMBER; other .cu files define their own). Legacy half/float
// C16 pool kernels below rely on this value.
#ifndef PACK_NUMBER
#define PACK_NUMBER 16
#endif
#define PACK_NUMBER_C2 (PACK_NUMBER/2)
// half2 min/max helpers used by the legacy maxpool_halfC16/avgpool_halfC16 kernels
#define HALF2_MIN half2(-65504, -65504)
#define MNN_CUDA_HALF2_MAX(a, b) do { (a).x = __hgt((a).x, (b).x) ? (a).x : (b).x; (a).y = __hgt((a).y, (b).y) ? (a).y : (b).y; } while (0)

namespace MNN {
namespace Corpus {

// ============================================================================
// Pool: source/backend/cuda/execution/PoolExecution.cu
// ============================================================================
template <typename T>
__global__ void maxpool_C8(const T* uInput, T* uOutput, const int ib, const int ic_p, const int ih, const int iw,
                           const int oh, const int ow, const int padX, const int padY, const int kernelX,
                           const int kernelY, const int strideX, const int strideY) {
    int total = ib * oh * ow * ic_p;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int ic_idx = i % ic_p;
        int tmp0 = i / ic_p;
        int ow_idx = tmp0 % ow;
        int tmp1 = tmp0 / ow;
        int ib_idx = tmp1 / oh;
        int oh_idx = tmp1 % oh;
        int iw_idx = ow_idx * strideX - padX;
        int ih_idx = oh_idx * strideY - padY;
        int sx = max(0, -iw_idx);
        int sy = max(0, -ih_idx);
        int ex = min(kernelX, iw - iw_idx);
        int ey = min(kernelY, ih - ih_idx);
        T maxValue = HALF_MIN;
        for (int fy = sy; fy < ey; ++fy) {
            for (int fx = sx; fx < ex; ++fx) {
                int currentX = iw_idx + fx;
                int currentY = ih_idx + fy;
                const T* input = (const T*)(uInput + ib_idx * ih * iw * ic_p + currentY * iw * ic_p + currentX * ic_p + ic_idx);
                T val = *input;
                maxValue = maxValue > val ? maxValue : val;
            }
        }
        T* dst = (T*)(uOutput + ib_idx * oh * ow * ic_p + oh_idx * ow * ic_p + ow_idx * ic_p + ic_idx);
        *dst = maxValue;
    }
}
template <typename T>
__global__ void avgpool_C8(const T* uInput, T* uOutput, const int ib, const int ic_p, const int ih, const int iw,
                           const int oh, const int ow, const int padX, const int padY, const int kernelX,
                           const int kernelY, const int strideX, const int strideY) {
    int total = ib * oh * ow * ic_p;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int ic_idx = i % ic_p;
        int tmp0 = i / ic_p;
        int ow_idx = tmp0 % ow;
        int tmp1 = tmp0 / ow;
        int ib_idx = tmp1 / oh;
        int oh_idx = tmp1 % oh;
        int iw_idx = ow_idx * strideX - padX;
        int ih_idx = oh_idx * strideY - padY;
        int sx = max(0, -iw_idx);
        int sy = max(0, -ih_idx);
        int ex = min(kernelX, iw - iw_idx);
        int ey = min(kernelY, ih - ih_idx);
        T div = (float)(ey - sy) * (float)(ex - sx);
        T sumValue = (T)0.0f;
        for (int fy = sy; fy < ey; ++fy) {
            for (int fx = sx; fx < ex; ++fx) {
                int currentX = iw_idx + fx;
                int currentY = ih_idx + fy;
                const T* input = (const T*)(uInput + ib_idx * ih * iw * ic_p + currentY * iw * ic_p + currentX * ic_p + ic_idx);
                T val = *input;
                sumValue += val;
            }
        }
        sumValue /= div;
        T* dst = (T*)(uOutput + ib_idx * oh * ow * ic_p + oh_idx * ow * ic_p + ow_idx * ic_p + ic_idx);
        *dst = sumValue;
    }
}
template <typename T>
__global__ void global_avgpool_C8(const T* input, T* output, const int outside, const int axis, const int inside,
                                  const int per_block_size, const int calc_multi_num) {
    int idx_outside = blockIdx.x / inside;
    int idx_inside = blockIdx.x - idx_outside * inside;
    const T* src = input + idx_outside * axis * inside + idx_inside;
    int tid = threadIdx.x;
    float local_src = 0.0;
    __shared__ float sumValue;
    for (int i = 0; i < calc_multi_num; i++) {
        if (tid + i * per_block_size < axis) {
            local_src += (float)(src[(tid + i * per_block_size) * inside]);
        }
    }
    float maxRes = blockReduceSum<float>(local_src);
    if (tid == 0) sumValue = maxRes;
    __syncthreads();
    output[idx_outside * inside + idx_inside] = (T)(sumValue / (float)axis);
}
template <typename T>
__global__ void global_maxpool_C8(const T* input, T* output, const int outside, const int axis, const int inside,
                                   const int per_block_size, const int calc_multi_num) {
    int idx_outside = blockIdx.x / inside;
    int idx_inside = blockIdx.x - idx_outside * inside;
    const T* src = input + idx_outside * axis * inside + idx_inside;
    int tid = threadIdx.x;
    float local_src = -FLT_MAX;
    __shared__ float maxValue;
    for (int i = 0; i < calc_multi_num; i++) {
        if (tid + i * per_block_size < axis) {
            local_src = max(local_src, (float)(src[(tid + i * per_block_size) * inside]));
        }
    }
    float maxRes = blockReduceMax<float>(local_src);
    if (tid == 0) maxValue = maxRes;
    __syncthreads();
    output[idx_outside * inside + idx_inside] = (T)maxValue;
}

// ============================================================================
// 1.2.0 tag kernels — NCHW layout (not NC4HW4 packed)
// ============================================================================
template <typename T>
__global__ void maxpool_120(const T* uInput, T* uOutput,
    int bc, int ih, int iw, int oh, int ow,
    int padX, int padY, int kernelX, int kernelY, int strideX, int strideY) {
    int total = bc * oh * ow;
    CUDA_KERNEL_LOOP(i, total) {
        int x = i % ow;
        int tmp = i / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int ix = x * strideX - padX;
        int iy = y * strideY - padY;
        int sx = max(0, -ix);
        int sy = max(0, -iy);
        int ex = min(kernelX, iw - ix);
        int ey = min(kernelY, ih - iy);
        T maxValue = (T)(-1000000);
        for (int fy = sy; fy < ey; ++fy) {
            for (int fx = sx; fx < ex; ++fx) {
                T inputColor = uInput[z * iw * ih + (iy + fy) * iw + (ix + fx)];
                maxValue = max(inputColor, maxValue);
            }
        }
        uOutput[z * ow * oh + y * ow + x] = maxValue;
    }
}
template <typename T>
__global__ void avgpool_120(const T* uInput, T* uOutput,
    int bc, int ih, int iw, int oh, int ow,
    int padX, int padY, int kernelX, int kernelY, int strideX, int strideY) {
    int total = bc * oh * ow;
    CUDA_KERNEL_LOOP(i, total) {
        int x = i % ow;
        int tmp = i / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int ix = x * strideX - padX;
        int iy = y * strideY - padY;
        int sx = max(0, -ix);
        int sy = max(0, -iy);
        int ex = min(kernelX, iw - ix);
        int ey = min(kernelY, ih - iy);
        T sumValue = (T)0;
        for (int fy = sy; fy < ey; ++fy) {
            for (int fx = sx; fx < ex; ++fx) {
                T inputColor = uInput[z * iw * ih + (iy + fy) * iw + (ix + fx)];
                sumValue = sumValue + inputColor;
            }
        }
        uOutput[z * ow * oh + y * ow + x] = sumValue / ((T)(ey - sy) * (T)(ex - sx));
    }
}

// ============================================================================
// BF16 pool kernels — faithful copy from source/backend/cuda/execution/bf16/PoolBf16.cuh
// Body guarded by #if (__CUDA_ARCH__ >= 800): sm75 compiles empty kernel,
// sm80+ executes. Type __nv_bfloat16 available via cuda_bf16.h on all archs.
// ============================================================================
template<typename T>
__global__ void maxpool_C8_BF16(const T* uInput, T* uOutput,
    const int ib, const int ic_p,
    const int ih, const int iw,
    const int oh, const int ow,
    const int padX, const int padY,
    const int kernelX, const int kernelY,
    const int strideX, const int strideY
) {
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    int total = ib * oh * ow * ic_p;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int ic_idx = i % ic_p;
        int tmp0 = i / ic_p;
        int ow_idx = tmp0 % ow;
        int tmp1 = tmp0 / ow;
        int ib_idx = tmp1 / oh;
        int oh_idx = tmp1 % oh;

        int iw_idx = ow_idx * strideX - padX;
        int ih_idx = oh_idx * strideY - padY;
        int sx = max(0, -iw_idx);
        int sy = max(0, -ih_idx);
        int ex = min(kernelX, iw - iw_idx);
        int ey = min(kernelY, ih - ih_idx);
        T maxValue = uInput[0];
        for (int fy=sy; fy<ey; ++fy) {
            for (int fx=sx; fx<ex; ++fx) {
                int currentX = iw_idx + fx;
                int currentY = ih_idx + fy;
                const T* input = (const T*)(uInput
                    + ib_idx * ih * iw * ic_p
                    + currentY * iw * ic_p
                    + currentX * ic_p
                    + ic_idx
                );
                T val = *input;
                maxValue = maxValue > val ? maxValue : val;
            }
        }
        T* dst = (T*)(uOutput
            + ib_idx * oh * ow * ic_p
            + oh_idx * ow * ic_p
            + ow_idx * ic_p
            + ic_idx
        );
        *dst = maxValue;
    }
    #endif
}

template<typename T>
__global__ void avgpool_C8_BF16(const T* uInput, T* uOutput,
    const int ib, const int ic_p,
    const int ih, const int iw,
    const int oh, const int ow,
    const int padX, const int padY,
    const int kernelX, const int kernelY,
    const int strideX, const int strideY
) {
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    int total = ib * oh * ow * ic_p;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int ic_idx = i % ic_p;
        int tmp0 = i / ic_p;
        int ow_idx = tmp0 % ow;
        int tmp1 = tmp0 / ow;
        int ib_idx = tmp1 / oh;
        int oh_idx = tmp1 % oh;

        int iw_idx = ow_idx * strideX - padX;
        int ih_idx = oh_idx * strideY - padY;
        int sx = max(0, -iw_idx);
        int sy = max(0, -ih_idx);
        int ex = min(kernelX, iw - iw_idx);
        int ey = min(kernelY, ih - ih_idx);
        T div = (float)(ey-sy)* (float)(ex-sx);
        T sumValue = (T)0.0f;
        for (int fy=sy; fy<ey; ++fy) {
            for (int fx=sx; fx<ex; ++fx) {
                int currentX = iw_idx + fx;
                int currentY = ih_idx + fy;
                const T* input = (const T*)(uInput
                    + ib_idx * ih * iw * ic_p
                    + currentY * iw * ic_p
                    + currentX * ic_p
                    + ic_idx
                );
                T val = *input;
                sumValue += val;
            }
        }
        sumValue /= div;
        T* dst = (T*)(uOutput
            + ib_idx * oh * ow * ic_p
            + oh_idx * ow * ic_p
            + ow_idx * ic_p
            + ic_idx
        );
        *dst = sumValue;
    }
    #endif
}

// ============================================================================
// 1.2.7-era legacy pool kernels (PACK_NUMBER=16, from legacy_kernels.cu)
// ============================================================================
__global__ void maxpool_halfC16(const half* uInput, half* uOutput,
    int bc,
    int ih, int iw,
    int oh, int ow,
    int padX, int padY,
    int kernelX, int kernelY,
    int strideX, int strideY
    ) {
    int total = bc * oh * ow * 8;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int x = i % ow;
        int tmp = i / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int zC = z / 8;
        int zR = z % 8;
        int ix = x * strideX - padX;
        int iy = y * strideY - padY;
        int sx = max(0, -ix);
        int sy = max(0, -iy);
        int ex = min(kernelX, iw - ix);
        int ey = min(kernelY, ih - iy);
        float div = (float)(ey-sy)* (float)(ex-sx);
        half2 sumValue = HALF2_MIN;
        for (int fy=sy; fy<ey; ++fy) {
            for (int fx=sx; fx<ex; ++fx) {
                int currentX = ix + fx;
                int currentY = iy + fy;
                const half2* input = (const half2*)(uInput
                    + zR * 2
                    + currentX * 16
                    + currentY * iw * 16
                    + zC * iw * ih * 16
                );
                half2 inputV = *input;
                MNN_CUDA_HALF2_MAX(sumValue, inputV);
            }
        }
        half2* dst = (half2*)(uOutput
            + zC * ow * oh * 16
            + y * ow * 16
            + x * 16
            + zR * 2
        );
        *dst = sumValue;
    }
}

__global__ void avgpool_halfC16(const half* uInput, half* uOutput,
    int bc,
    int ih, int iw,
    int oh, int ow,
    int padX, int padY,
    int kernelX, int kernelY,
    int strideX, int strideY
    ) {
    int total = bc * oh * ow * 8;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int x = i % ow;
        int tmp = i / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int zC = z / 8;
        int zR = z % 8;
        int ix = x * strideX - padX;
        int iy = y * strideY - padY;
        int sx = max(0, -ix);
        int sy = max(0, -iy);
        int ex = min(kernelX, iw - ix);
        int ey = min(kernelY, ih - iy);
        float div = (float)(ey-sy)* (float)(ex-sx);
        half2 sumValue = half2(0.0f, 0.0f);
        half2 mulValue = half2(1.0f / div, 1.0f/div);
        for (int fy=sy; fy<ey; ++fy) {
            for (int fx=sx; fx<ex; ++fx) {
                int currentX = ix + fx;
                int currentY = iy + fy;
                const half2* input = (const half2*)(uInput
                    + zR * 2
                    + currentX * 16
                    + currentY * iw * 16
                    + zC * iw * ih * 16
                );
                sumValue = __hadd2(sumValue, (*input) * mulValue);
            }
        }
        half2* dst = (half2*)(uOutput
            + zC * ow * oh * 16
            + y * ow * 16
            + x * 16
            + zR * 2
        );
        *dst = sumValue;
    }
}

__global__ void maxpool_floatC16(const float* uInput, float* uOutput,
    int bc,
    int ih, int iw,
    int oh, int ow,
    int padX, int padY,
    int kernelX, int kernelY,
    int strideX, int strideY
    ) {
    int total = bc * oh * ow * PACK_NUMBER;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int x = i % ow;
        int tmp = i / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int zC = z / PACK_NUMBER;
        int zR = z % PACK_NUMBER;
        int ix = x * strideX - padX;
        int iy = y * strideY - padY;
        int sx = max(0, -ix);
        int sy = max(0, -iy);
        int ex = min(kernelX, iw - ix);
        int ey = min(kernelY, ih - iy);
        float maxValue = -FLT_MAX;
        for (int fy=sy; fy<ey; ++fy) {
            for (int fx=sx; fx<ex; ++fx) {
                int currentX = ix + fx;
                int currentY = iy + fy;
                const float* input = (const float*)(uInput
                    + zR
                    + currentX * PACK_NUMBER
                    + currentY * iw * PACK_NUMBER
                    + zC * iw * ih * PACK_NUMBER
                );
                maxValue = max(maxValue, *input);
            }
        }
        float* dst = (float*)(uOutput
            + zC * ow * oh * PACK_NUMBER
            + y * ow * PACK_NUMBER
            + x * PACK_NUMBER
            + zR
        );
        *dst = maxValue;
    }
}

__global__ void avgpool_floatC16(const float* uInput, float* uOutput,
    int bc,
    int ih, int iw,
    int oh, int ow,
    int padX, int padY,
    int kernelX, int kernelY,
    int strideX, int strideY
    ) {
    int total = bc * oh * ow * PACK_NUMBER;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int x = i % ow;
        int tmp = i / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int zC = z / PACK_NUMBER;
        int zR = z % PACK_NUMBER;
        int ix = x * strideX - padX;
        int iy = y * strideY - padY;
        int sx = max(0, -ix);
        int sy = max(0, -iy);
        int ex = min(kernelX, iw - ix);
        int ey = min(kernelY, ih - iy);
        float div = (float)(ey-sy)* (float)(ex-sx);
        float sumValue = 0.0f;
        float mulValue = 1.0f/div;
        for (int fy=sy; fy<ey; ++fy) {
            for (int fx=sx; fx<ex; ++fx) {
                int currentX = ix + fx;
                int currentY = iy + fy;
                const float* input = (const float*)(uInput
                    + zR
                    + currentX * PACK_NUMBER
                    + currentY * iw * PACK_NUMBER
                    + zC * iw * ih * PACK_NUMBER
                );
                sumValue = sumValue + (*input) * mulValue;
            }
        }
        float* dst = (float*)(uOutput
            + zC * ow * oh * 16
            + y * ow * 16
            + x * 16
            + zR
        );
        *dst = sumValue;
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- Pool ----
void mnn_corpus_maxpool_fp32(const float* in, float* out, int ib, int ic_p, int ih, int iw, int oh, int ow,
                              int padX, int padY, int kx, int ky, int sx, int sy, int grid, int block,
                              cudaStream_t stream) {
    MNN::Corpus::maxpool_C8<float><<<grid, block, 0, stream>>>(in, out, ib, ic_p, ih, iw, oh, ow, padX, padY, kx, ky,
                                                                 sx, sy);
}
void mnn_corpus_maxpool_fp16(const void* in, void* out, int ib, int ic_p, int ih, int iw, int oh, int ow,
                              int padX, int padY, int kx, int ky, int sx, int sy, int grid, int block,
                              cudaStream_t stream) {
    MNN::Corpus::maxpool_C8<half><<<grid, block, 0, stream>>>((const half*)in, (half*)out, ib, ic_p, ih, iw, oh, ow,
                                                                 padX, padY, kx, ky, sx, sy);
}
void mnn_corpus_avgpool_fp32(const float* in, float* out, int ib, int ic_p, int ih, int iw, int oh, int ow,
                              int padX, int padY, int kx, int ky, int sx, int sy, int grid, int block,
                              cudaStream_t stream) {
    MNN::Corpus::avgpool_C8<float><<<grid, block, 0, stream>>>(in, out, ib, ic_p, ih, iw, oh, ow, padX, padY, kx, ky,
                                                                 sx, sy);
}
void mnn_corpus_avgpool_fp16(const void* in, void* out, int ib, int ic_p, int ih, int iw, int oh, int ow,
                              int padX, int padY, int kx, int ky, int sx, int sy, int grid, int block,
                              cudaStream_t stream) {
    MNN::Corpus::avgpool_C8<half><<<grid, block, 0, stream>>>((const half*)in, (half*)out, ib, ic_p, ih, iw, oh, ow,
                                                                 padX, padY, kx, ky, sx, sy);
}
void mnn_corpus_global_avgpool_fp32(const float* input, float* output, int outside, int axis, int inside,
                                    int per_block_size, int calc_multi_num, int grid, int block,
                                    cudaStream_t stream) {
    MNN::Corpus::global_avgpool_C8<float><<<grid, block, 0, stream>>>(input, output, outside, axis, inside,
                                                                       per_block_size, calc_multi_num);
}
void mnn_corpus_global_avgpool_fp16(const void* input, void* output, int outside, int axis, int inside,
                                    int per_block_size, int calc_multi_num, int grid, int block,
                                    cudaStream_t stream) {
    MNN::Corpus::global_avgpool_C8<half><<<grid, block, 0, stream>>>((const half*)input, (half*)output, outside, axis,
                                                                       inside, per_block_size, calc_multi_num);
}
void mnn_corpus_global_maxpool_fp32(const float* input, float* output, int outside, int axis, int inside,
                                     int per_block_size, int calc_multi_num, int grid, int block,
                                     cudaStream_t stream) {
    MNN::Corpus::global_maxpool_C8<float><<<grid, block, 0, stream>>>(input, output, outside, axis, inside,
                                                                       per_block_size, calc_multi_num);
}
void mnn_corpus_global_maxpool_fp16(const void* input, void* output, int outside, int axis, int inside,
                                     int per_block_size, int calc_multi_num, int grid, int block,
                                     cudaStream_t stream) {
    MNN::Corpus::global_maxpool_C8<half><<<grid, block, 0, stream>>>((const half*)input, (half*)output, outside, axis,
                                                                       inside, per_block_size, calc_multi_num);
}

// ---- 1.2.0 tag: Pool (NCHW layout, bc parameter) ----
void mnn_corpus_maxpool_120_fp32(const float* in, float* out, int bc, int ih, int iw, int oh, int ow,
                                 int padX, int padY, int kx, int ky, int sx, int sy,
                                 int grid, int block, cudaStream_t stream) {
    MNN::Corpus::maxpool_120<float><<<grid, block, 0, stream>>>(in, out, bc, ih, iw, oh, ow, padX, padY, kx, ky, sx, sy);
}
void mnn_corpus_avgpool_120_fp32(const float* in, float* out, int bc, int ih, int iw, int oh, int ow,
                                  int padX, int padY, int kx, int ky, int sx, int sy,
                                  int grid, int block, cudaStream_t stream) {
    MNN::Corpus::avgpool_120<float><<<grid, block, 0, stream>>>(in, out, bc, ih, iw, oh, ow, padX, padY, kx, ky, sx, sy);
}

// ---- BF16 pool (faithful copy; body executes on sm80+, empty on sm75) ----
void mnn_corpus_maxpool_c8_bf16(const void* input, void* output,
                                int ib, int ic_p, int ih, int iw, int oh, int ow,
                                int padX, int padY, int kx, int ky, int sx, int sy,
                                int grid, int block, cudaStream_t stream) {
    MNN::Corpus::maxpool_C8_BF16<__nv_bfloat16><<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output,
        ib, ic_p, ih, iw, oh, ow, padX, padY, kx, ky, sx, sy);
}
void mnn_corpus_avgpool_c8_bf16(const void* input, void* output,
                                int ib, int ic_p, int ih, int iw, int oh, int ow,
                                int padX, int padY, int kx, int ky, int sx, int sy,
                                int grid, int block, cudaStream_t stream) {
    MNN::Corpus::avgpool_C8_BF16<__nv_bfloat16><<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output,
        ib, ic_p, ih, iw, oh, ow, padX, padY, kx, ky, sx, sy);
}

} // extern "C"
