// transpose_half.cu - half-precision transpose/pack/unpack/fuseblit kernels for corpus
//   Kernels copied verbatim from:
//     - source/backend/cuda/execution/Transpose.cu
//       (PACKCOMMON_half_4, PACKCOMMON_REARRANGE_half_4, UNPACKCOMMON_REARRANGE_half_4)
//     - source/backend/cuda/execution/Raster.cu
//       (fuseblit_half_4, UNARY_HALF2_SIGMOID)
//   Each kernel body is copied EXACTLY as MNN upstream; only the surrounding
//   namespace, PACK_NUMBER const, and extern "C" fp32 shims are added.
#include "corpus_common.cuh"
#include <cuda_fp16.h>

namespace MNN {
namespace Corpus {

static const int PACK_NUMBER = 8;

// ============================================================================
// From source/backend/cuda/execution/Transpose.cu (lines 118-139)
// PACKCOMMON_half_4: pack 4-element half vector (half2 output) with {0,0} zero fill.
// Template T0,T1 — shim instantiates <const float*, half2*> for fp32 -> half2 pack.
// ============================================================================
template<typename T0, typename T1>
__global__ void PACKCOMMON_half_4(const T0 *input, T1 *output,
    int inside, int axis, int outside,
    int insideStride, int axisStride,
    DivModFast is, DivModFast cs
    ) {
    int axisAlign = UP_DIV(axis, PACK_NUMBER/ 4) * PACK_NUMBER / 4;;
    int total = axisAlign * inside * outside;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int tmp, x, y, z;
        cs.divmod(i, tmp, y);
        is.divmod(tmp, z, x);
        int dstOffset = (z * inside + x) * axisAlign + y;
        int srcOffset = x * insideStride + y * axisStride + z * inside * axis;
        if (y < axis) {
            output[dstOffset] = input[srcOffset];
        } else {
            output[dstOffset] = T1(0);
        }
    }
}

// ============================================================================
// From source/backend/cuda/execution/Transpose.cu (lines 164-202)
// PACKCOMMON_REARRANGE_half_4: shared-memory tile rearrange for half-4 pack.
// Non-template, operates on double* (8-byte = 4x half elements).
// ============================================================================
__global__ void PACKCOMMON_REARRANGE_half_4(const double *input, double *output,
    int inside, int axis, int outside,
    int insideStride, int axisStride
    ) {
    int axisAlign = UP_DIV(axis, PACK_NUMBER) * PACK_NUMBER / 4;

    int insideAlign = inside / 4;
    int axisNum = axis / 16;
    int insideNum = inside / 32;

    __shared__ double sharedData[128];
    int tid = blockIdx.x;
    int localIdx = threadIdx.x;

    int tmpI = tid / axisNum;
    int y = tid % axisNum;
    int x = tmpI % insideNum;
    int z = tmpI / insideNum;

    int x_mod = localIdx % 8; // [0 ~ 8]
    int y_mod = localIdx / 8; // [0 ~ 15]
    int srcOffset = (8*x+x_mod) + (z * axis + (16*y+y_mod)) * insideAlign;
    sharedData[localIdx] = input[srcOffset];// [C_16, HW4_8]

    __syncthreads();

    int oy_mod = localIdx % 4; // [0 ~ 3]
    int ox_mod = localIdx / 4; // [0 ~ 31]
    int dstOffset = (z * inside + 32*x+ox_mod) * axisAlign + 4*y+oy_mod;

    // [4*oy_mod, ox_mod]
    half tmp_data[4];
    tmp_data[0] = ((half*)sharedData)[(4*oy_mod+0) * 32 + ox_mod];
    tmp_data[1] = ((half*)sharedData)[(4*oy_mod+1) * 32 + ox_mod];
    tmp_data[2] = ((half*)sharedData)[(4*oy_mod+2) * 32 + ox_mod];
    tmp_data[3] = ((half*)sharedData)[(4*oy_mod+3) * 32 + ox_mod];

    output[dstOffset] = ((double*)tmp_data)[0];
}

// ============================================================================
// From source/backend/cuda/execution/Transpose.cu (lines 55-93)
// UNPACKCOMMON_REARRANGE_half_4: unpack counterpart of the above, shared-mem
// tile rearrange. Non-template, operates on double*.
// ============================================================================
__global__ void UNPACKCOMMON_REARRANGE_half_4(const double *input, double *output,
    int inside, int axis, int outside,
    int insideStride, int axisStride
    ) {
    int axisAlign = UP_DIV(axis, PACK_NUMBER) * PACK_NUMBER / 4;
    int insideAlign = inside / 4;
    int axisNum = axis / 16;
    int insideNum = inside / 32;

    __shared__ double sharedData[128];
    int tid = blockIdx.x;
    int localIdx = threadIdx.x;

    int tmpI = tid / axisNum;
    int y = tid % axisNum;
    int x = tmpI % insideNum;
    int z = tmpI / insideNum;

    int y_mod = localIdx % 4; // [0 ~ 3]
    int x_mod = localIdx / 4; // [0 ~ 31]
    int srcOffset = (z * inside + 32*x+x_mod) * axisAlign + 4*y+y_mod;
    sharedData[localIdx] = input[srcOffset];// [HW_32, C4_4]

    __syncthreads();

    int oy_mod = localIdx % 16; // [0 ~ 15]
    int ox_mod = localIdx / 16; // [0 ~ 7]

    // [4*ox_mod, oy_mod]
    half tmp_data[4];
    tmp_data[0] = ((half*)sharedData)[(4*ox_mod+0) * 16 + oy_mod];
    tmp_data[1] = ((half*)sharedData)[(4*ox_mod+1) * 16 + oy_mod];
    tmp_data[2] = ((half*)sharedData)[(4*ox_mod+2) * 16 + oy_mod];
    tmp_data[3] = ((half*)sharedData)[(4*ox_mod+3) * 16 + oy_mod];

    int dstOffset = (8*x+ox_mod) + (z * axis + (16*y+oy_mod)) * insideAlign;

    output[dstOffset] = ((double*)tmp_data)[0];
}

// ============================================================================
// From source/backend/cuda/execution/Raster.cu (lines 381-398)
// fuseblit_half_4: half-4 fused multi-region blit, int2 (4x int16) vec copy.
// Non-template, operates on int16_t*.
// ============================================================================
__global__ void fuseblit_half_4(const int16_t *input, int16_t *output,
    int fuseNum, int count, const int32_t* sliceOffset,
    DivModFast sizeZ, DivModFast sizeY, DivModFast sizeX,
    int strideZ, int strideY,
    int dstStrideZ, int dstStrideY
    ) {
    for (size_t c = blockIdx.x * blockDim.x + threadIdx.x; c < count; c += blockDim.x * gridDim.x) {
        int ix, tmp, iy, tmp2, iz, j;
        sizeX.divmod(c, tmp, ix);
        sizeY.divmod(tmp, tmp2, iy);
        sizeZ.divmod(tmp2, j, iz);
        int src_offset = sliceOffset[j] + iz * strideZ + iy * strideY + (ix << 2);
        int dst_offset = sliceOffset[fuseNum+j] + iz * dstStrideZ + iy * dstStrideY + (ix << 2);
        int2* srcF = (int2 *)(input + src_offset);
        int2* dstF = (int2 *)(output + dst_offset);
        dstF[0] = srcF[0];
    }
}

// ============================================================================
// From source/backend/cuda/execution/Raster.cu (lines 138-149)
// UNARY_HALF2_SIGMOID: half2 vectorized sigmoid. Template T — shim
// instantiates <half2> for the fp32-shim entry point.
// ============================================================================
template<typename T>
__global__ void UNARY_HALF2_SIGMOID(const T *input, T *output,
        int count
        ) { 
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    half2 x = input[i];
    half2 one;
    one.x = 1.0;
    one.y = 1.0f;
    output[i] = __h2div(one, __hadd2(one, h2exp(__hneg2(x))));
  }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- PACKCOMMON_half_4 (fp32 corpus: scalar float variant) ----
// Corpus-only fallback: MNN instantiates this kernel only as
// `<const int2*, int2*>` (half2 packed, 4 bytes per element), launched from
// PackBuffer() when bytes==2. The corpus fp32 path cannot use half2 packed
// I/O (the replay runner operates on float* scalars), so we instantiate
// `<float, float>` with PACK_NUMBER/4 alignment (matching MNN's axisAlign
// formula) and replace the `{0, 0}` half2 zero-fill with `T1(0)`. Math is
// equivalent for the scalar case; the half2 vectorization benefit is lost
// (this is a correctness corpus, not a perf benchmark for this path).
void mnn_corpus_packcommon_half_4_fp32(const void* input, void* output,
    int inside, int axis, int outside, int insideStride, int axisStride,
    int d_is, int d_cs, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast is(d_is);
    MNN::Corpus::DivModFast cs(d_cs);
    MNN::Corpus::PACKCOMMON_half_4<float, float><<<grid, block, 0, stream>>>(
        (const float*)input, (float*)output,
        inside, axis, outside, insideStride, axisStride, is, cs);
}

// ---- PACKCOMMON_REARRANGE_half_4 (double* tile rearrange) ----
void mnn_corpus_packcommon_rearrange_half_4_fp32(const double* input, double* output,
    int inside, int axis, int outside, int insideStride, int axisStride,
    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::PACKCOMMON_REARRANGE_half_4<<<grid, block, 0, stream>>>(
        input, output, inside, axis, outside, insideStride, axisStride);
}

// ---- UNPACKCOMMON_REARRANGE_half_4 (double* tile rearrange) ----
void mnn_corpus_unpackcommon_rearrange_half_4_fp32(const double* input, double* output,
    int inside, int axis, int outside, int insideStride, int axisStride,
    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::UNPACKCOMMON_REARRANGE_half_4<<<grid, block, 0, stream>>>(
        input, output, inside, axis, outside, insideStride, axisStride);
}

// ---- fuseblit_half_4 (int16_t* vec2 fused blit) ----
void mnn_corpus_fuseblit_half_4_fp32(const void* input, void* output,
    int fuseNum, int count, const int32_t* sliceOffset,
    int sizeX_val, int sizeY_val, int sizeZ_val,
    int strideZ, int strideY, int dstStrideZ, int dstStrideY,
    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_z(sizeZ_val), d_y(sizeY_val), d_x(sizeX_val);
    MNN::Corpus::fuseblit_half_4<<<grid, block, 0, stream>>>(
        (const int16_t*)input, (int16_t*)output,
        fuseNum, count, sliceOffset, d_z, d_y, d_x,
        strideZ, strideY, dstStrideZ, dstStrideY);
}

// ---- UNARY_HALF2_SIGMOID (half2 vectorized sigmoid) ----
void mnn_corpus_unary_half2_sigmoid_fp32(const void* input, void* output, size_t count,
    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::UNARY_HALF2_SIGMOID<half2><<<grid, block, 0, stream>>>(
        (const half2*)input, (half2*)output, (int)count);
}

} // extern "C"
