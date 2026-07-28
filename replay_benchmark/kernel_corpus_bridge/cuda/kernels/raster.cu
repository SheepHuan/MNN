// raster.cu - blitRegion (3.6.0) + blitRegion_241 (2.4.1)
//   + pack_c4_120 / unpack_c4_120 / SETZERO_120 / add_bias_120 (1.2.0) kernels + shims
//   source/backend/cuda/execution/Raster.cu
#include "corpus_common.cuh"

namespace MNN {
namespace Corpus {

// ============================================================================
// 1.2.0 tag: pack_c4 / unpack_c4 (NC4HW4 pack/unpack, 3.6.0 uses PACKCOMMON)
// ============================================================================
template <typename T>
__global__ void pack_c4_120(const T* input, T* output, int inside, int axis, int outside, int axisC4) {
    int total = inside * axis * outside;
    CUDA_KERNEL_LOOP(i, total) {
        int x = i % inside;
        int tmp = i / inside;
        int y = tmp % axis;
        int z = tmp / axis;
        int y4 = y / 4;
        int yR = y % 4;
        output[(z * axisC4 + y4) * inside * 4 + x * 4 + yR] = input[i];
    }
}
template <typename T>
__global__ void unpack_c4_120(const T* input, T* output, int inside, int axis, int outside, int axisC4) {
    int total = inside * axis * outside;
    CUDA_KERNEL_LOOP(i, total) {
        int x = i % inside;
        int tmp = i / inside;
        int y = tmp % axis;
        int z = tmp / axis;
        int y4 = y / 4;
        int yR = y % 4;
        output[i] = input[(z * axisC4 + y4) * inside * 4 + x * 4 + yR];
    }
}

// ============================================================================
// 1.2.0 tag: SETZERO (ScatterNd zero-fill)
// ============================================================================
template <typename T>
__global__ void SETZERO_120(const int n, T* outputPtr) {
    CUDA_KERNEL_LOOP(i, n) { outputPtr[i] = (T)0; }
}

// ============================================================================
// 1.2.0 tag: add_bias (MatMul bias addition)
// ============================================================================
template <typename T>
__global__ void add_bias_120(T* input, T* output, const T* bias, int e, int h) {
    CUDA_KERNEL_LOOP(i, e * h) {
        int hi = i % h;
        output[i] = input[i] + bias[hi];
    }
}

// ============================================================================
// Raster blitRegion: source/backend/cuda/execution/Raster.cu
// ============================================================================
template <typename T>
__global__ void blitRegion(const T* inputO, T* outputO, int count, int loopCount,
                           const int32_t* dstIndice, const int32_t* srcIndice,
                           int dstUseIndice, int srcUseIndice, int dstStep, int srcStep, int srcLimit,
                           int sizeZ, int sizeY, int sizeX,
                           int strideZ, int strideY, int strideX,
                           int dstStrideZ, int dstStrideY, int dstStrideX) {
    for (size_t fuseIndex = blockIdx.x * blockDim.x + threadIdx.x; fuseIndex < count; fuseIndex += blockDim.x * gridDim.x) {
        int x = fuseIndex % sizeX;
        int temp = fuseIndex / sizeX;
        int y = temp % sizeY;
        temp = temp / sizeY;
        int z = temp % sizeZ;
        int i = temp / sizeZ;
        int srcOffsetO = i * srcStep;
        if (srcUseIndice >= 0) srcOffsetO = srcIndice[i] * srcStep;
        int dstOffsetO = i * dstStep;
        if (dstUseIndice >= 0) dstOffsetO = dstIndice[i] * dstStep;
        if (srcOffsetO >= 0 && srcOffsetO < srcLimit) {
            const T* input = inputO + srcOffsetO;
            T* output = outputO + dstOffsetO;
            int srcOffset = z * strideZ + y * strideY + x * strideX;
            int dstOffset = z * dstStrideZ + y * dstStrideY + x * dstStrideX;
            output[dstOffset] = input[srcOffset];
        } else {
            T* output = outputO + dstOffsetO;
            int dstOffset = z * dstStrideZ + y * dstStrideY + x * dstStrideX;
            output[dstOffset] = (T)0;
        }
    }
}

// ---- blitRegion 2.4.1: no count param, triple nested ZYX loop ----
template <typename T>
__global__ void blitRegion_241(const T* inputO, T* outputO, int loopCount,
                                const int32_t* dstIndice, const int32_t* srcIndice,
                                int dstUseIndice, int srcUseIndice, int dstStep, int srcStep, int srcLimit,
                                int sizeZ, int sizeY, int sizeX,
                                int strideZ, int strideY, int strideX,
                                int dstStrideZ, int dstStrideY, int dstStrideX) {
    int total = loopCount;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)total; i += blockDim.x * gridDim.x) {
        int srcOffsetO = i * srcStep;
        if (srcUseIndice >= 0) srcOffsetO = srcIndice[i] * srcStep;
        int dstOffsetO = i * dstStep;
        if (dstUseIndice >= 0) dstOffsetO = dstIndice[i] * dstStep;
        if (srcOffsetO >= 0 && srcOffsetO < srcLimit) {
            const T* input = inputO + srcOffsetO;
            T* output = outputO + dstOffsetO;
            for (int z = 0; z < sizeZ; ++z)
                for (int y = 0; y < sizeY; ++y)
                    for (int x = 0; x < sizeX; ++x)
                        output[z*dstStrideZ + y*dstStrideY + x*dstStrideX] = input[z*strideZ + y*strideY + x*strideX];
        } else {
            T* output = outputO + dstOffsetO;
            for (int z = 0; z < sizeZ; ++z)
                for (int y = 0; y < sizeY; ++y)
                    for (int x = 0; x < sizeX; ++x)
                        output[z*dstStrideZ + y*dstStrideY + x*dstStrideX] = (T)0;
        }
    }
}

// ---- fuseblit (fused multi-region blit, 3.6.0) ----
template<typename T0, typename T1>
__global__ void fuseblit(const T0* input, T1* output, int fuseNum, int count, const int32_t* sliceOffset,
                         DivModFast sizeZ, DivModFast sizeY, DivModFast sizeX,
                         int strideZ, int strideY, int strideX,
                         int dstStrideZ, int dstStrideY, int dstStrideX) {
    for (size_t c = blockIdx.x * blockDim.x + threadIdx.x; c < (size_t)count; c += blockDim.x * gridDim.x) {
        int ix, tmp, iy, tmp2, iz, j;
        sizeX.divmod(c, tmp, ix);
        sizeY.divmod(tmp, tmp2, iy);
        sizeZ.divmod(tmp2, j, iz);
        int src_offset = sliceOffset[j] + iz * strideZ + iy * strideY + ix * strideX;
        int dst_offset = sliceOffset[fuseNum + j] + iz * dstStrideZ + iy * dstStrideY + ix * dstStrideX;
        output[dst_offset] = input[src_offset];
    }
}

// ---- fuseblit_4 (vec4 fused blit, 3.6.0) ----
__global__ void fuseblit_4(const int32_t* input, int32_t* output, int fuseNum, int count, const int32_t* sliceOffset,
                           DivModFast sizeZ, DivModFast sizeY, DivModFast sizeX,
                           int strideZ, int strideY, int dstStrideZ, int dstStrideY) {
    for (size_t c = blockIdx.x * blockDim.x + threadIdx.x; c < (size_t)count; c += blockDim.x * gridDim.x) {
        int ix, tmp, iy, tmp2, iz, j;
        sizeX.divmod(c, tmp, ix);
        sizeY.divmod(tmp, tmp2, iy);
        sizeZ.divmod(tmp2, j, iz);
        int src_offset = sliceOffset[j] + iz * strideZ + iy * strideY + (ix << 2);
        int dst_offset = sliceOffset[fuseNum + j] + iz * dstStrideZ + iy * dstStrideY + (ix << 2);
        int4* srcF = (int4*)(input + src_offset);
        int4* dstF = (int4*)(output + dst_offset);
        dstF[0] = srcF[0];
    }
}

// ---- fuseblitLimit (fused blit with boundary check, 3.6.0) ----
template<typename T0, typename T1>
__global__ void fuseblitLimit(const T0* input, T1* output, const FuseRegion* info, const int32_t* sliceOffset) {
    int sizeZ = info->size[0], sizeY = info->size[1], sizeX = info->size[2];
    int strideZ = info->srcStride[0], strideY = info->srcStride[1], strideX = info->srcStride[2];
    int dstStrideZ = info->dstStride[0], dstStrideY = info->dstStride[1], dstStrideX = info->dstStride[2];
    int fuseNum = info->fuseNumber;
    int count = fuseNum * sizeZ * sizeY * sizeX;
    for (size_t c = blockIdx.x * blockDim.x + threadIdx.x; c < (size_t)count; c += blockDim.x * gridDim.x) {
        int j = c / (sizeZ * sizeY * sizeX);
        int i = c % (sizeZ * sizeY * sizeX);
        int ix = i % sizeX;
        int tmp = i / sizeX;
        int iy = tmp % sizeY;
        int iz = tmp / sizeY;
        const int* srcOffsetPtr = sliceOffset + 8 * j;
        const int* dstOffsetPtr = sliceOffset + 8 * j + 4;
        T0 srcValue = (T0)0;
        int src_offset = srcOffsetPtr[3] + iz * strideZ + iy * strideY + ix * strideX;
        if (srcOffsetPtr[0] > iz && srcOffsetPtr[1] > iy && srcOffsetPtr[2] > ix) {
            srcValue = input[src_offset];
        }
        int dst_offset = dstOffsetPtr[3] + iz * dstStrideZ + iy * dstStrideY + ix * dstStrideX;
        if (dstOffsetPtr[0] > iz && dstOffsetPtr[1] > iy && dstOffsetPtr[2] > ix) {
            output[dst_offset] = srcValue;
        }
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- Raster blitRegion ----
void mnn_corpus_blitregion_fp32(const float* input, float* output, int count, int loopCount,
                                 const int32_t* dstIndice, const int32_t* srcIndice,
                                 int dstUseIndice, int srcUseIndice, int dstStep, int srcStep, int srcLimit,
                                 int sizeZ, int sizeY, int sizeX,
                                 int strideZ, int strideY, int strideX,
                                 int dstStrideZ, int dstStrideY, int dstStrideX,
                                 int grid, int block, cudaStream_t stream) {
    MNN::Corpus::blitRegion<float><<<grid, block, 0, stream>>>(input, output, count, loopCount,
        dstIndice, srcIndice, dstUseIndice, srcUseIndice, dstStep, srcStep, srcLimit,
        sizeZ, sizeY, sizeX, strideZ, strideY, strideX, dstStrideZ, dstStrideY, dstStrideX);
}

// ---- blitRegion 2.4.1: no count param ----
void mnn_corpus_blitregion_241_fp32(const float* input, float* output, int loopCount,
                                     const int32_t* dstIndice, const int32_t* srcIndice,
                                     int dstUseIndice, int srcUseIndice, int dstStep, int srcStep, int srcLimit,
                                     int sizeZ, int sizeY, int sizeX,
                                     int strideZ, int strideY, int strideX,
                                     int dstStrideZ, int dstStrideY, int dstStrideX,
                                     int grid, int block, cudaStream_t stream) {
    MNN::Corpus::blitRegion_241<float><<<grid, block, 0, stream>>>(input, output, loopCount,
        dstIndice, srcIndice, dstUseIndice, srcUseIndice, dstStep, srcStep, srcLimit,
        sizeZ, sizeY, sizeX, strideZ, strideY, strideX, dstStrideZ, dstStrideY, dstStrideX);
}

// ---- 1.2.0 tag: pack_c4 / unpack_c4 ----
void mnn_corpus_pack_c4_120_fp32(const float* input, float* output, int inside, int axis, int outside, int axisC4,
                                 int grid, int block, cudaStream_t stream) {
    MNN::Corpus::pack_c4_120<float><<<grid, block, 0, stream>>>(input, output, inside, axis, outside, axisC4);
}
void mnn_corpus_unpack_c4_120_fp32(const float* input, float* output, int inside, int axis, int outside, int axisC4,
                                   int grid, int block, cudaStream_t stream) {
    MNN::Corpus::unpack_c4_120<float><<<grid, block, 0, stream>>>(input, output, inside, axis, outside, axisC4);
}

// ---- 1.2.0 tag: SETZERO ----
void mnn_corpus_setzero_120_fp32(const int n, float* outputPtr, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SETZERO_120<float><<<grid, block, 0, stream>>>(n, outputPtr);
}

// ---- 1.2.0 tag: add_bias ----
void mnn_corpus_add_bias_120_fp32(float* input, float* output, const float* bias, int e, int h,
                                   int grid, int block, cudaStream_t stream) {
    MNN::Corpus::add_bias_120<float><<<grid, block, 0, stream>>>(input, output, bias, e, h);
}

// ---- fuseblit (fused multi-region blit, 3.6.0) ----
void mnn_corpus_fuseblit_fp32(const float* input, float* output, int fuseNum, int count, const int32_t* sliceOffset,
                              int sizeX_val, int sizeY_val, int sizeZ_val,
                              int strideZ, int strideY, int strideX,
                              int dstStrideZ, int dstStrideY, int dstStrideX,
                              int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_z(sizeZ_val), d_y(sizeY_val), d_x(sizeX_val);
    MNN::Corpus::fuseblit<float, float><<<grid, block, 0, stream>>>(
        input, output, fuseNum, count, sliceOffset, d_z, d_y, d_x,
        strideZ, strideY, strideX, dstStrideZ, dstStrideY, dstStrideX);
}

// ---- fuseblit_4 (vec4 fused blit, 3.6.0) ----
void mnn_corpus_fuseblit_4_fp32(const int32_t* input, int32_t* output, int fuseNum, int count, const int32_t* sliceOffset,
                                int sizeX_val, int sizeY_val, int sizeZ_val,
                                int strideZ, int strideY, int dstStrideZ, int dstStrideY,
                                int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_z(sizeZ_val), d_y(sizeY_val), d_x(sizeX_val);
    MNN::Corpus::fuseblit_4<<<grid, block, 0, stream>>>(
        input, output, fuseNum, count, sliceOffset, d_z, d_y, d_x,
        strideZ, strideY, dstStrideZ, dstStrideY);
}

// ---- fuseblitLimit (fused blit with boundary check, 3.6.0) ----
void mnn_corpus_fuseblit_limit_fp32(const float* input, float* output,
                                    const MNN::Corpus::FuseRegion* info, const int32_t* sliceOffset,
                                    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::fuseblitLimit<float, float><<<grid, block, 0, stream>>>(input, output, info, sliceOffset);
}

} // extern "C"
