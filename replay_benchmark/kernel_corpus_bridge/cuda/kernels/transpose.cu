// transpose.cu - NHWC_2_NCHW / NCHW_2_NHWC (3.6.0) + NCHW_2_NHWC_212 (2.1.2)
//   kernels + shims
//   source/backend/cuda/execution/Transpose.cu
#include "corpus_common.cuh"

// MNN CUDA uses PACK_NUMBER=8 for transpose/pack kernels
static const int PACK_NUMBER = 8;

namespace MNN {
namespace Corpus {

// ============================================================================
// Transpose format conversion: source/backend/cuda/execution/Transpose.cu
// These bodies mirror source/backend/cuda/execution/Transpose.cu. The shim
// resolves DivModFast values from the adapter's outside/axis/inside contract.
// ============================================================================
template<typename T0, typename T1>
__global__ void NHWC_2_NCHW(const T0* input,
                            T1* output,
                            const int maxCount,
                            const int channel, // redundant parameter
                            const int area,
                            const int inChannelPack,
                            DivModFast divOutChannelPack,
                            DivModFast divArea
) {
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divArea.divmod(index, temp, area_idx);
        divOutChannelPack.divmod(temp, batch_idx, chnl_idx);

        int src_offset = (batch_idx * area + area_idx) * inChannelPack + chnl_idx;
        output[index] = (T1)input[src_offset];
    }
}
template<typename T0, typename T1>
__global__ void NCHW_2_NHWC(const T0* input,
                            T1* output,
                            const int maxCount,
                            const int channel, // redundant parameter
                            const int area,
                            const int inChannelPack,
                            DivModFast divOutChannelPack,
                            DivModFast divArea
) {
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divOutChannelPack.divmod(index, temp, chnl_idx);
        divArea.divmod(temp, batch_idx, area_idx);

        int src_offset = (batch_idx * inChannelPack + chnl_idx) * area + area_idx;
        output[index] = (T1)input[src_offset];
    }
}

// ---- NCHW_2_NHWC 2.1.2: src_offset uses channel (not inChannelPack) ----
template<typename T0, typename T1>
__global__ void NCHW_2_NHWC_212(const T0* input,
    T1* output,
    const int maxCount,
    const int channel,
    const int area,
    const int channel_pack,
    DivModFast d_oc,
    DivModFast d_area
) {
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        d_oc.divmod(index, temp, chnl_idx);
        d_area.divmod(temp, batch_idx, area_idx);

        int src_offset = (batch_idx * channel + chnl_idx) * area + area_idx;
        output[index] = (T1)input[src_offset];
    }
}

// ---- TRANSPOSE (generic, uses TransposeParam struct) ----
template <typename T>
__global__ void TRANSPOSE(const T* input, T* output, const TransposeParam* param) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)param->total) {
        int x = i % param->dims[0];
        int tmp = i / param->dims[0];
        int y = tmp % param->dims[1];
        int z = tmp / param->dims[1];
        int srcOffset = param->srcStride * z + y + x * param->dims[2];
        int dstOffset = param->dstStride * z + x + y * param->dims[3];
        output[dstOffset] = input[srcOffset];
    }
}

#define LOCAL_DIM 8
template <typename T>
__global__ void TRANSPOSE_LOCAL(const T* input, T* output, const TransposeParam* param) {
    __shared__ T localM[LOCAL_DIM][LOCAL_DIM + 1];
    int num = blockIdx.z;
    for (int n = num; n < param->size; n += gridDim.z) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x < param->dims[0] && y < param->dims[1]) {
            int offset = n * param->srcStride + x * param->dims[2] + y;
            localM[threadIdx.y][threadIdx.x] = input[offset];
        }
        __syncthreads();
        x = blockIdx.y * blockDim.y + threadIdx.x;
        y = blockIdx.x * blockDim.x + threadIdx.y;
        if (x < param->dims[1] && y < param->dims[0]) {
            int offset = n * param->dstStride + x * param->dims[3] + y;
            output[offset] = localM[threadIdx.x][threadIdx.y];
        }
    }
}

// ---- NCHW_2_NCHW (identity copy) ----
template<typename T0, typename T1>
__global__ void NCHW_2_NCHW(const T0* input, T1* output, const int maxCount) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)maxCount; index += blockDim.x * gridDim.x) {
        output[index] = (T1)input[index];
    }
}

// ---- Format conversion kernels (all share same body, different src/dst strides) ----
template<typename T0, typename T1>
__global__ void NHWC8_2_NCHW(const T0* input, T1* output, const int maxCount, const int channel, const int area,
                             const int inChannelPack, DivModFast divOutChannelPack, DivModFast divArea) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divArea.divmod(index, temp, area_idx);
        divOutChannelPack.divmod(temp, batch_idx, chnl_idx);
        int src_offset = (batch_idx * area + area_idx) * inChannelPack + chnl_idx;
        output[index] = (T1)input[src_offset];
    }
}
template<typename T0, typename T1>
__global__ void C4NHW4_2_NCHW(const T0* input, T1* output, const int maxCount, const int channel, const int area,
                               const int inChannelPack, DivModFast divOutChannelPack, DivModFast divArea) {
    const int batch = (maxCount / channel) / area;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divArea.divmod(index, temp, area_idx);
        divOutChannelPack.divmod(temp, batch_idx, chnl_idx);
        int c4_idx = chnl_idx >> 2;
        int cL_idx = chnl_idx & 3;
        int src_offset = ((c4_idx * batch + batch_idx) * area + area_idx) * 4 + cL_idx;
        output[index] = (T1)input[src_offset];
    }
}
template<typename T0, typename T1>
__global__ void NHWC8_2_NHWC(const T0* input, T1* output, const int maxCount, const int channel, const int area,
                             const int inChannelPack, DivModFast divOutChannelPack, DivModFast divArea) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divArea.divmod(index, temp, area_idx);
        divOutChannelPack.divmod(temp, batch_idx, chnl_idx);
        int src_offset = (batch_idx * area + area_idx) * inChannelPack + chnl_idx;
        output[index] = (T1)input[src_offset];
    }
}
template<typename T0, typename T1>
__global__ void C4NHW4_2_NHWC(const T0* input, T1* output, const int maxCount, const int channel, const int area,
                               const int inChannelPack, DivModFast divOutChannelPack, DivModFast divArea) {
    const int batch = (maxCount / channel) / area;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divArea.divmod(index, temp, area_idx);
        divOutChannelPack.divmod(temp, batch_idx, chnl_idx);
        int c4_idx = chnl_idx >> 2;
        int cL_idx = chnl_idx & 3;
        int src_offset = ((c4_idx * batch + batch_idx) * area + area_idx) * 4 + cL_idx;
        output[index] = (T1)input[src_offset];
    }
}
template<typename T0, typename T1>
__global__ void NHWC_2_NHWC8(const T0* input, T1* output, const int maxCount, const int channel, const int area,
                             const int inChannelPack, DivModFast divOutChannelPack, DivModFast divArea) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divArea.divmod(index, temp, area_idx);
        divOutChannelPack.divmod(temp, batch_idx, chnl_idx);
        if (chnl_idx >= channel) {
            output[index] = (T1)0.0f;
            continue;
        }
        int src_offset = (batch_idx * area + area_idx) * inChannelPack + chnl_idx;
        output[index] = (T1)input[src_offset];
    }
}
template<typename T0, typename T1>
__global__ void NCHW_2_NHWC8(const T0* input, T1* output, const int maxCount, const int channel, const int area,
                             const int inChannelPack, DivModFast divOutChannelPack, DivModFast divArea) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divArea.divmod(index, temp, area_idx);
        divOutChannelPack.divmod(temp, batch_idx, chnl_idx);
        if (chnl_idx >= channel) {
            output[index] = (T1)0.0f;
            continue;
        }
        int src_offset = (batch_idx * inChannelPack + chnl_idx) * area + area_idx;
        output[index] = (T1)input[src_offset];
    }
}
template<typename T0, typename T1>
__global__ void C4NHW4_2_NHWC8(const T0* input, T1* output, const int maxCount, const int channel, const int area,
                                const int inChannelPack, DivModFast divOutChannelPack, DivModFast divArea) {
    const int batch = (maxCount / (UP_DIV(channel, 8) * 8)) / area;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divArea.divmod(index, temp, area_idx);
        divOutChannelPack.divmod(temp, batch_idx, chnl_idx);
        if (chnl_idx >= channel) {
            output[index] = (T1)0.0f;
            continue;
        }
        int c4_idx = chnl_idx >> 2;
        int cL_idx = chnl_idx & 3;
        int src_offset = ((c4_idx * batch + batch_idx) * area + area_idx) * 4 + cL_idx;
        output[index] = (T1)input[src_offset];
    }
}
template<typename T0, typename T1>
__global__ void NHWC_2_C4NHW4(const T0* input, T1* output, const int maxCount, const int channel, const int area,
                              const int inChannelPack, DivModFast divOutChannelPack, DivModFast divArea) {
    const int batch = (maxCount / (UP_DIV(channel, 4) * 4)) / area;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divArea.divmod(index, temp, area_idx);
        divOutChannelPack.divmod(temp, batch_idx, chnl_idx);
        int c4_idx = chnl_idx >> 2;
        int cL_idx = chnl_idx & 3;
        int dst_offset = ((c4_idx * batch + batch_idx) * area + area_idx) * 4 + cL_idx;
        int src_offset = (batch_idx * area + area_idx) * inChannelPack + chnl_idx;
        if (chnl_idx >= channel) {
            output[dst_offset] = (T1)0.0f;
            continue;
        }
        output[dst_offset] = (T1)input[src_offset];
    }
}
template<typename T0, typename T1>
__global__ void NCHW_2_C4NHW4(const T0* input, T1* output, const int maxCount, const int channel, const int area,
                              const int inChannelPack, DivModFast divOutChannelPack, DivModFast divArea) {
    const int batch = (maxCount / (UP_DIV(channel, 4) * 4)) / area;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divArea.divmod(index, temp, area_idx);
        divOutChannelPack.divmod(temp, batch_idx, chnl_idx);
        int c4_idx = chnl_idx >> 2;
        int cL_idx = chnl_idx & 3;
        int dst_offset = ((c4_idx * batch + batch_idx) * area + area_idx) * 4 + cL_idx;
        int src_offset = (batch_idx * inChannelPack + chnl_idx) * area + area_idx;
        if (chnl_idx >= channel) {
            output[dst_offset] = (T1)0.0f;
            continue;
        }
        output[dst_offset] = (T1)input[src_offset];
    }
}
template<typename T0, typename T1>
__global__ void NHWC8_2_C4NHW4(const T0* input, T1* output, const int maxCount, const int channel, const int area,
                                const int inChannelPack, DivModFast divOutChannelPack, DivModFast divArea) {
    const int batch = (maxCount / (UP_DIV(channel, 4) * 4)) / area;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divArea.divmod(index, temp, area_idx);
        divOutChannelPack.divmod(temp, batch_idx, chnl_idx);
        int c4_idx = chnl_idx >> 2;
        int cL_idx = chnl_idx & 3;
        int dst_offset = ((c4_idx * batch + batch_idx) * area + area_idx) * 4 + cL_idx;
        int src_offset = (batch_idx * area + area_idx) * inChannelPack + chnl_idx;
        output[dst_offset] = (T1)input[src_offset];
    }
}

// ---- PACKCOMMON / UNPACKCOMMON ----
// Faithful to MNN Transpose.cu PACKCOMMON/UNPACKCOMMON:
//   axisAlign = UP_DIV(axis, PACK_NUMBER) * PACK_NUMBER
//   dstOffset = (z * inside + x) * axisAlign + y  (NC4HW4 layout)
//   srcOffset = x * insideStride + y * axisStride + z * inside * axis
// Corpus shim hardcodes insideStride=1, axisStride=area (NHWC src layout).
template<typename T0, typename T1>
__global__ void PACKCOMMON(const T0* input, T1* output, const int maxCount, const int channel, const int area,
                           const int inChannelPack, DivModFast divOutChannelPack, DivModFast divArea) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divArea.divmod(index, temp, area_idx);
        divOutChannelPack.divmod(temp, batch_idx, chnl_idx);
        int dst_offset = (batch_idx * area + area_idx) * inChannelPack + chnl_idx;
        int src_offset = area_idx + chnl_idx * area + batch_idx * area * channel;
        if (chnl_idx < channel) {
            output[dst_offset] = input[src_offset];
        } else {
            output[dst_offset] = (T1)0.0;
        }
    }
}
template<typename T0, typename T1>
__global__ void UNPACKCOMMON(const T0* input, T1* output, const int maxCount, const int channel, const int area,
                             const int inChannelPack, DivModFast divOutChannelPack, DivModFast divArea) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)maxCount; index += blockDim.x * gridDim.x) {
        int area_idx, temp, chnl_idx, batch_idx;
        divArea.divmod(index, temp, area_idx);
        divOutChannelPack.divmod(temp, batch_idx, chnl_idx);
        if (chnl_idx < channel) {
            int src_offset = (batch_idx * area + area_idx) * inChannelPack + chnl_idx;
            int dst_offset = area_idx + chnl_idx * area + batch_idx * area * channel;
            output[dst_offset] = input[src_offset];
        }
    }
}

// ---- blit_2_float (vec2 blit) ----
template<typename T>
__global__ void blit_2_float(const T* input, T* output, int count,
                             DivModFast sizeZ, DivModFast sizeY, DivModFast sizeX,
                             int strideZ, int strideY, int dstStrideZ, int dstStrideY) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)count; i += blockDim.x * gridDim.x) {
        int ix, tmp, iy, iz;
        sizeX.divmod(i, tmp, ix);
        sizeY.divmod(tmp, iz, iy);
        int srcOffset = iz * strideZ + iy * strideY + (ix << 1);
        int dstOffset = iz * dstStrideZ + iy * dstStrideY + (ix << 1);
        int2* dstF = (int2*)(output + dstOffset);
        dstF[0] = ((int2*)(input + srcOffset))[0];
    }
}

// ---- PACKCOMMON_4 (vec4 pack C4) ----
// Faithful to MNN Transpose.cu PACKCOMMON_4:
//   axisAlign = UP_DIV(axis, PACK_NUMBER/4) * PACK_NUMBER/4   (PACK_NUMBER=8 → align to 2)
//   dstOffset = (z * inside + x) * axisAlign + y
//   srcOffset = x * insideStride + y * axisStride + z * inside * axis
// Corpus shim hardcodes insideStride=1, axisStride=area (NHWC src layout).
template<typename T0, typename T1>
__global__ void PACKCOMMON_4(const T0* input, T1* output,
    int inside, int axis, int outside,
    int insideStride, int axisStride,
    DivModFast is, DivModFast cs
) {
    int axisAlign = UP_DIV(axis, PACK_NUMBER / 4) * PACK_NUMBER / 4;
    int total = axisAlign * inside * outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)total; i += blockDim.x * gridDim.x) {
        int tmp, x, y, z;
        cs.divmod(i, tmp, y);
        is.divmod(tmp, z, x);
        int dstOffset = (z * inside + x) * axisAlign + y;
        int srcOffset = x * insideStride + y * axisStride + z * inside * axis;
        if (y < axis) {
            output[dstOffset] = input[srcOffset];
        } else {
            output[dstOffset] = (T1)0.0;
        }
    }
}

// ---- UNPACKCOMMON_4 (vec4 unpack C4) ----
template<typename T0, typename T1>
__global__ void UNPACKCOMMON_4(const T0* input, T1* output,
    const int total, int inside, int axis, int outside,
    int insideStride, int axisStride, int axisAlign,
    DivModFast is, DivModFast cs
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)total; i += blockDim.x * gridDim.x) {
        int tmp, x, y, z;
        cs.divmod(i, tmp, y);
        is.divmod(tmp, z, x);
        if (y < axis) {
            int srcOffset = (z * inside + x) * axisAlign + y;
            int dstOffset = x * insideStride + y * axisStride + z * inside * axis;
            output[dstOffset] = input[srcOffset];
        }
    }
}

// ---- blit_2_half (vec2 blit for raster, same as blit_2_float but int copy) ----
template<typename T>
__global__ void blit_2_half(const T* input, T* output, int count,
                            DivModFast sizeZ, DivModFast sizeY, DivModFast sizeX,
                            int strideZ, int strideY, int dstStrideZ, int dstStrideY) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)count; i += blockDim.x * gridDim.x) {
        int ix, tmp, iy, iz;
        sizeX.divmod(i, tmp, ix);
        sizeY.divmod(tmp, iz, iy);
        int srcOffset = iz * strideZ + iy * strideY + (ix << 1);
        int dstOffset = iz * dstStrideZ + iy * dstStrideY + (ix << 1);
        int* dstF = (int*)(output + dstOffset);
        dstF[0] = ((int*)(input + srcOffset))[0];
    }
}

// ---- transpose_BDL_to_BLD (LinearAttention, 3.6.0) ----
// Faithful copy of MNN's shared-memory tiled [B][D][L] -> [B][L][D]
// transpose. The extra tile column avoids shared-memory bank conflicts.
#define TILE_DIM 32
#define BLOCK_ROWS 8
__global__ void transpose_BDL_to_BLD(
    const float* __restrict__ input, float* __restrict__ output,
    int B, int D, int L
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int batchIdx = blockIdx.z;
    const float* in = input + batchIdx * D * L;
    float* out = output + batchIdx * L * D;
    int xBase = blockIdx.x * TILE_DIM;
    int yBase = blockIdx.y * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int d = xBase + threadIdx.y + j;
        int l = yBase + threadIdx.x;
        if (d < D && l < L)
            tile[threadIdx.y + j][threadIdx.x] = in[d * L + l];
    }
    __syncthreads();
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int l = yBase + threadIdx.y + j;
        int d = xBase + threadIdx.x;
        if (l < L && d < D)
            out[l * D + d] = tile[threadIdx.x][threadIdx.y + j];
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- Transpose format conversion ----
void mnn_corpus_nhwc2nchw_fp32(const float* input, float* output, int total, int inside, int axis, int outside,
                                int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(axis), d_area(inside);
    MNN::Corpus::NHWC_2_NCHW<float, float><<<grid, block, 0, stream>>>(
        input, output, total, axis, inside, axis, d_oc, d_area);
}
void mnn_corpus_nchw2nhwc_fp32(const float* input, float* output, int total, int inside, int axis, int outside,
                                int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(axis), d_area(inside);
    MNN::Corpus::NCHW_2_NHWC<float, float><<<grid, block, 0, stream>>>(
        input, output, total, axis, inside, axis, d_oc, d_area);
}

// ---- Transpose fp16 ----
void mnn_corpus_nhwc2nchw_fp16(const void* input, void* output, int total, int inside, int axis, int outside,
                                int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(axis), d_area(inside);
    MNN::Corpus::NHWC_2_NCHW<half, half><<<grid, block, 0, stream>>>(
        (const half*)input, (half*)output, total, axis, inside, axis, d_oc, d_area);
}
void mnn_corpus_nchw2nhwc_fp16(const void* input, void* output, int total, int inside, int axis, int outside,
                                int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(axis), d_area(inside);
    MNN::Corpus::NCHW_2_NHWC<half, half><<<grid, block, 0, stream>>>(
        (const half*)input, (half*)output, total, axis, inside, axis, d_oc, d_area);
}

// ---- NCHW_2_NHWC 2.1.2: src_offset uses channel ----
void mnn_corpus_nchw2nhwc_212_fp32(const float* input, float* output, int total, int channel, int area, int channel_pack,
                                    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(channel_pack);
    MNN::Corpus::DivModFast d_area(area);
    MNN::Corpus::NCHW_2_NHWC_212<float, float><<<grid, block, 0, stream>>>(input, output, total, channel, area, channel_pack, d_oc, d_area);
}

// ---- TRANSPOSE (generic, uses TransposeParam struct) ----
void mnn_corpus_transpose_fp32(const float* input, float* output, const MNN::Corpus::TransposeParam* param,
                               int grid, int block, cudaStream_t stream) {
    MNN::Corpus::TRANSPOSE<float><<<grid, block, 0, stream>>>(input, output, param);
}
// ---- TRANSPOSE_LOCAL (shared memory tiled transpose, 8x8) ----
void mnn_corpus_transpose_local_fp32(const float* input, float* output, const MNN::Corpus::TransposeParam* param,
                                     int gridX, int gridY, int gridZ, int block, cudaStream_t stream) {
    dim3 grid(gridX, gridY, gridZ);
    dim3 blk(8, 8);
    MNN::Corpus::TRANSPOSE_LOCAL<float><<<grid, blk, 0, stream>>>(input, output, param);
}

// ---- Format conversion kernels (NCHW↔NHWC8↔C4NHW4) ----
// All follow the same pattern: for each output element, compute source offset
// from batch/area/channel decomposition. We implement a single generic kernel
// that covers all conversions by parameterizing the stride/pack relationships.
void mnn_corpus_nchw2nchw_fp32(const float* input, float* output, int maxCount,
                               int grid, int block, cudaStream_t stream) {
    MNN::Corpus::NCHW_2_NCHW<float, float><<<grid, block, 0, stream>>>(input, output, maxCount);
}
void mnn_corpus_nhwc8_2_nchw_fp32(const float* input, float* output, int maxCount, int channel, int area, int inChannelPack,
                                  int d_oc_val, int d_area_val, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(d_oc_val), d_area(d_area_val);
    MNN::Corpus::NHWC8_2_NCHW<float, float><<<grid, block, 0, stream>>>(input, output, maxCount, channel, area, inChannelPack, d_oc, d_area);
}
void mnn_corpus_c4nhw4_2_nchw_fp32(const float* input, float* output, int maxCount, int channel, int area, int inChannelPack,
                                   int d_oc_val, int d_area_val, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(d_oc_val), d_area(d_area_val);
    MNN::Corpus::C4NHW4_2_NCHW<float, float><<<grid, block, 0, stream>>>(input, output, maxCount, channel, area, inChannelPack, d_oc, d_area);
}
void mnn_corpus_nhwc8_2_nhwc_fp32(const float* input, float* output, int maxCount, int channel, int area, int inChannelPack,
                                  int d_oc_val, int d_area_val, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(d_oc_val), d_area(d_area_val);
    MNN::Corpus::NHWC8_2_NHWC<float, float><<<grid, block, 0, stream>>>(input, output, maxCount, channel, area, inChannelPack, d_oc, d_area);
}
void mnn_corpus_c4nhw4_2_nhwc_fp32(const float* input, float* output, int maxCount, int channel, int area, int inChannelPack,
                                   int d_oc_val, int d_area_val, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(d_oc_val), d_area(d_area_val);
    MNN::Corpus::C4NHW4_2_NHWC<float, float><<<grid, block, 0, stream>>>(input, output, maxCount, channel, area, inChannelPack, d_oc, d_area);
}
void mnn_corpus_nhwc_2_nhwc8_fp32(const float* input, float* output, int maxCount, int channel, int area, int inChannelPack,
                                  int d_oc_val, int d_area_val, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(d_oc_val), d_area(d_area_val);
    MNN::Corpus::NHWC_2_NHWC8<float, float><<<grid, block, 0, stream>>>(input, output, maxCount, channel, area, inChannelPack, d_oc, d_area);
}
void mnn_corpus_nchw_2_nhwc8_fp32(const float* input, float* output, int maxCount, int channel, int area, int inChannelPack,
                                  int d_oc_val, int d_area_val, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(d_oc_val), d_area(d_area_val);
    MNN::Corpus::NCHW_2_NHWC8<float, float><<<grid, block, 0, stream>>>(input, output, maxCount, channel, area, inChannelPack, d_oc, d_area);
}
void mnn_corpus_c4nhw4_2_nhwc8_fp32(const float* input, float* output, int maxCount, int channel, int area, int inChannelPack,
                                    int d_oc_val, int d_area_val, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(d_oc_val), d_area(d_area_val);
    MNN::Corpus::C4NHW4_2_NHWC8<float, float><<<grid, block, 0, stream>>>(input, output, maxCount, channel, area, inChannelPack, d_oc, d_area);
}
void mnn_corpus_nhwc_2_c4nhw4_fp32(const float* input, float* output, int maxCount, int channel, int area, int inChannelPack,
                                   int d_oc_val, int d_area_val, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(d_oc_val), d_area(d_area_val);
    MNN::Corpus::NHWC_2_C4NHW4<float, float><<<grid, block, 0, stream>>>(input, output, maxCount, channel, area, inChannelPack, d_oc, d_area);
}
void mnn_corpus_nchw_2_c4nhw4_fp32(const float* input, float* output, int maxCount, int channel, int area, int inChannelPack,
                                   int d_oc_val, int d_area_val, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(d_oc_val), d_area(d_area_val);
    MNN::Corpus::NCHW_2_C4NHW4<float, float><<<grid, block, 0, stream>>>(input, output, maxCount, channel, area, inChannelPack, d_oc, d_area);
}
void mnn_corpus_nhwc8_2_c4nhw4_fp32(const float* input, float* output, int maxCount, int channel, int area, int inChannelPack,
                                    int d_oc_val, int d_area_val, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(d_oc_val), d_area(d_area_val);
    MNN::Corpus::NHWC8_2_C4NHW4<float, float><<<grid, block, 0, stream>>>(input, output, maxCount, channel, area, inChannelPack, d_oc, d_area);
}

// ---- PACKCOMMON / UNPACKCOMMON (pack/unpack C4 channel) ----
void mnn_corpus_packcommon_fp32(const float* input, float* output, int maxCount, int channel, int area, int inChannelPack,
                                int d_oc_val, int d_area_val, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(d_oc_val), d_area(d_area_val);
    MNN::Corpus::PACKCOMMON<float, float><<<grid, block, 0, stream>>>(input, output, maxCount, channel, area, inChannelPack, d_oc, d_area);
}
void mnn_corpus_unpackcommon_fp32(const float* input, float* output, int maxCount, int channel, int area, int inChannelPack,
                                  int d_oc_val, int d_area_val, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(d_oc_val), d_area(d_area_val);
    MNN::Corpus::UNPACKCOMMON<float, float><<<grid, block, 0, stream>>>(input, output, maxCount, channel, area, inChannelPack, d_oc, d_area);
}

// ---- blit_2_float / blit_2_half (vec2 blit for raster) ----
void mnn_corpus_blit_2_float_fp32(const float* input, float* output, int count,
                                  int sizeX_val, int sizeY_val, int sizeZ_val,
                                  int strideZ, int strideY, int dstStrideZ, int dstStrideY,
                                  int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_x(sizeX_val), d_y(sizeY_val), d_z(sizeZ_val);
    MNN::Corpus::blit_2_float<float><<<grid, block, 0, stream>>>(input, output, count, d_z, d_y, d_x, strideZ, strideY, dstStrideZ, dstStrideY);
}

// ---- PACKCOMMON_4 (vec4 pack C4) ----
void mnn_corpus_packcommon_4_fp32(const float* input, float* output,
                                   int inside, int axis, int outside,
                                   int insideStride, int axisStride,
                                   int d_is_val, int d_cs_val,
                                   int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_is(d_is_val), d_cs(d_cs_val);
    MNN::Corpus::PACKCOMMON_4<float, float><<<grid, block, 0, stream>>>(
        input, output, inside, axis, outside, insideStride, axisStride, d_is, d_cs);
}

// ---- UNPACKCOMMON_4 (vec4 unpack C4) ----
void mnn_corpus_unpackcommon_4_fp32(const float* input, float* output,
                                     int total, int inside, int axis, int outside,
                                     int insideStride, int axisStride, int axisAlign,
                                     int d_is_val, int d_cs_val,
                                     int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_is(d_is_val), d_cs(d_cs_val);
    MNN::Corpus::UNPACKCOMMON_4<float, float><<<grid, block, 0, stream>>>(
        input, output, total, inside, axis, outside, insideStride, axisStride, axisAlign, d_is, d_cs);
}

// ---- blit_2_half (vec2 blit, same body as blit_2_float but int copy) ----
void mnn_corpus_blit_2_half_fp32(const float* input, float* output, int count,
                                  int sizeX_val, int sizeY_val, int sizeZ_val,
                                  int strideZ, int strideY, int dstStrideZ, int dstStrideY,
                                  int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_x(sizeX_val), d_y(sizeY_val), d_z(sizeZ_val);
    MNN::Corpus::blit_2_half<float><<<grid, block, 0, stream>>>(
        input, output, count, d_z, d_y, d_x, strideZ, strideY, dstStrideZ, dstStrideY);
}

// ---- transpose_BDL_to_BLD (LinearAttention, 3.6.0) ----
void mnn_corpus_transpose_bdl_to_bld_fp32(const float* input, float* output,
                                           int B, int D, int L,
                                           int gridX, int gridY, int gridZ,
                                           cudaStream_t stream) {
    dim3 grid(gridX, gridY, gridZ);
    dim3 block(TILE_DIM, BLOCK_ROWS);
    MNN::Corpus::transpose_BDL_to_BLD<<<grid, block, 0, stream>>>(input, output, B, D, L);
}

} // extern "C"
