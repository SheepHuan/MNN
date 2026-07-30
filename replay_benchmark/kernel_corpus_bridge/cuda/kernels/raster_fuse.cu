// raster_fuse.cu - Raster fused binary kernels for replay benchmark corpus
//   source/backend/cuda/execution/Raster.cu (Binary*/BinaryFuseAdd*/BinaryMid* macro families)
//
// Faithful copy of MNN macro templates. Each macro family is instantiated with
// ADD and MUL as representative operations. The macro logic is identical to
// MNN source — only the Func expression differs per operation.
#include "corpus_common.cuh"
#include <cuda_fp16.h>

namespace MNN {
namespace Corpus {

#define sign(y) ((y) > 0 ? 1 : ((y) < 0 ? -1 : 0))

// ============================================================================
// BINARY_FUNC: basic 3D-stride binary op (template <TIn, TOut>)
// ============================================================================
#define BINARY_FUNC(Name, Func)\
template<typename TIn, typename TOut>\
__global__ void Binary##Name(\
    const TIn *input0, const TIn* input1, TOut *output,\
    int sizeZ, int sizeY, int sizeX,\
    int strideZ, int strideY, int strideX,\
    int strideZ1, int strideY1, int strideX1,\
    int dstStrideZ, int dstStrideY, int dstStrideX, int activationType\
    ) { \
    int count = sizeZ * sizeY * sizeX;\
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {\
        int ix = i % sizeX;\
        int tmp = i / sizeX;\
        int iy = tmp % sizeY;\
        int iz = tmp / sizeY;\
        int srcOffset = iz * strideZ + iy * strideY + ix * strideX;\
        int srcOffset1 = iz * strideZ1 + iy * strideY1 + ix * strideX1;\
        int dstOffset = iz * dstStrideZ + iy * dstStrideY + ix * dstStrideX;\
        TIn x = input0[srcOffset];\
        TIn y = input1[srcOffset1];\
        TOut val = (TOut)(Func);\
        if(activationType == 1) {\
            val = (val < (TOut)0 ? (TOut)0 : val);\
        }\
        output[dstOffset] = val;\
    }\
}

BINARY_FUNC(ADD, x+y)
BINARY_FUNC(MUL, x*y)

// ============================================================================
// BINARY_FUSEADD_FUNC: binary op + atomicAdd to output (float only)
// ============================================================================
#define BINARY_FUSEADD_FUNC(Name, Func)\
__global__ void BinaryFuseAdd##Name(\
    const float *input0, const float* input1, float *output,\
    int sizeZ, int sizeY, int sizeX,\
    int strideZ, int strideY, int strideX,\
    int strideZ1, int strideY1, int strideX1,\
    int dstStrideZ, int dstStrideY, int dstStrideX\
    ) { \
    int count = sizeZ * sizeY * sizeX;\
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {\
        int ix = i % sizeX;\
        int tmp = i / sizeX;\
        int iy = tmp % sizeY;\
        int iz = tmp / sizeY;\
        int srcOffset = iz * strideZ + iy * strideY + ix * strideX;\
        int srcOffset1 = iz * strideZ1 + iy * strideY1 + ix * strideX1;\
        int dstOffset = iz * dstStrideZ + iy * dstStrideY + ix * dstStrideX;\
        float x = input0[srcOffset];\
        float y = input1[srcOffset1];\
        float val = (float)(Func);\
        atomicAdd(output + dstOffset, val);\
    }\
}

BINARY_FUSEADD_FUNC(ADD, x+y)
BINARY_FUSEADD_FUNC(MUL, x*y)

// ============================================================================
// BINARY_FUNC_FLOATMID: binary op with DivModFast + float intermediate (template)
// Includes BinaryMid, BinaryMid4, BinaryMidHalf2, BinaryMidLinear variants.
// For corpus, we implement BinaryMid (scalar) + BinaryMidLinear (1D) — the
// simplest variants. BinaryMid4/Half2 require PACK_NUMBER layout (deferred).
// ============================================================================
#define BINARY_FUNC_FLOATMID(Name, Func)\
template<typename TIn, typename TOut>\
__global__ void BinaryMid##Name(\
    const TIn *input0, const TIn* input1, TOut *output,\
    int sizeZ, int sizeY, int sizeX,\
    int strideZ, int strideY, int strideX,\
    int strideZ1, int strideY1, int strideX1,\
    int dstStrideZ, int dstStrideY, int dstStrideX, int activationType,\
    DivModFast d_sizeY, DivModFast d_sizeX\
    ) { \
    int count = sizeZ * sizeY * sizeX;\
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {\
        int ix, tmp, iy, iz;\
        d_sizeX.divmod(i, tmp, ix);\
        d_sizeY.divmod(tmp, iz, iy);\
        int srcOffset = iz * strideZ + iy * strideY + ix * strideX;\
        int srcOffset1 = iz * strideZ1 + iy * strideY1 + ix * strideX1;\
        int dstOffset = iz * dstStrideZ + iy * dstStrideY + ix * dstStrideX;\
        float x = input0[srcOffset];\
        float y = input1[srcOffset1];\
        float val = (float)(Func);\
        if(activationType == 1) {\
            val = (val < 0.0f ? 0.0f : val);\
        }\
        output[dstOffset] = val;\
    }\
}\
template<typename TIn, typename TOut>\
__global__ void BinaryMidLinear##Name(\
    const TIn *input0, const TIn* input1, TOut *output,\
    int sizeZ,\
    int strideZ,\
    int strideZ1,\
    int dstStrideZ,\
    int activationType\
    ) { \
    int count = sizeZ;\
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {\
        int iz = i;\
        int srcOffset = iz * strideZ;\
        int srcOffset1 = iz * strideZ1;\
        int dstOffset = iz * dstStrideZ;\
        float x = input0[srcOffset];\
        float y = input1[srcOffset1];\
        float val = (float)(Func);\
        if(activationType == 1) {\
            val = (val < 0.0f ? 0.0f : val);\
        }\
        output[dstOffset] = (TOut)val;\
    }\
}

BINARY_FUNC_FLOATMID(ADD, x+y)
BINARY_FUNC_FLOATMID(MUL, x*y)

// ============================================================================
// BINARY_FUNC_FLOATMID4: float4 vectorized binary op (template)
// BinaryMidLinear4: 1D, 4 elements per thread, float4 vectorized
// ============================================================================
#define BINARY_FUNC_FLOATMID4(Name, Func)\
template<typename TIn, typename TOut>\
__global__ void BinaryMidLinear4_##Name(\
    const TIn *input0, const TIn* input1, TOut *output,\
    int count_4, int activationType,\
    bool inp0Broadcast, bool inp1Broadcast\
    ) { \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count_4); i += blockDim.x * gridDim.x) {\
        int iz = i;\
        int srcOffset = iz << 2;\
        int srcOffset1 = iz << 2;\
        int dstOffset = iz << 2;\
        float4 xx = inp0Broadcast ? make_float4(input0[0], input0[0], input0[0], input0[0]) : ((float4 *)(input0+srcOffset))[0];\
        float4 yy = inp1Broadcast ? make_float4(input1[0], input1[0], input1[0], input1[0]) : ((float4 *)(input1+srcOffset1))[0];\
        float x = xx.x;\
        float y = yy.x;\
        TOut val = (TOut)(Func);\
        if(activationType == 1) {\
            val = (val < (TOut)0 ? (TOut)0 : val);\
        }\
        output[dstOffset] = val;\
        x = xx.y;\
        y = yy.y;\
        val = (TOut)(Func);\
        if(activationType == 1) {\
            val = (val < (TOut)0 ? (TOut)0 : val);\
        }\
        output[dstOffset+1] = val;\
        x = xx.z;\
        y = yy.z;\
        val = (TOut)(Func);\
        if(activationType == 1) {\
            val = (val < (TOut)0 ? (TOut)0 : val);\
        }\
        output[dstOffset+2] = val;\
        x = xx.w;\
        y = yy.w;\
        val = (TOut)(Func);\
        if(activationType == 1) {\
            val = (val < (TOut)0 ? (TOut)0 : val);\
        }\
        output[dstOffset+3] = val;\
    }\
}

BINARY_FUNC_FLOATMID4(ADD, x+y)
BINARY_FUNC_FLOATMID4(MUL, x*y)

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- BinaryADD (fp32, basic 3D-stride) ----
void mnn_corpus_binary_add_fp32(const float* in0, const float* in1, float* out,
                                 int sizeZ, int sizeY, int sizeX,
                                 int strideZ, int strideY, int strideX,
                                 int strideZ1, int strideY1, int strideX1,
                                 int dstStrideZ, int dstStrideY, int dstStrideX, int activationType,
                                 int grid, int block, cudaStream_t stream) {
    MNN::Corpus::BinaryADD<float, float><<<grid, block, 0, stream>>>(
        in0, in1, out, sizeZ, sizeY, sizeX,
        strideZ, strideY, strideX, strideZ1, strideY1, strideX1,
        dstStrideZ, dstStrideY, dstStrideX, activationType);
}
// ---- BinaryMUL (fp32, basic 3D-stride) ----
void mnn_corpus_binary_mul_fp32(const float* in0, const float* in1, float* out,
                                 int sizeZ, int sizeY, int sizeX,
                                 int strideZ, int strideY, int strideX,
                                 int strideZ1, int strideY1, int strideX1,
                                 int dstStrideZ, int dstStrideY, int dstStrideX, int activationType,
                                 int grid, int block, cudaStream_t stream) {
    MNN::Corpus::BinaryMUL<float, float><<<grid, block, 0, stream>>>(
        in0, in1, out, sizeZ, sizeY, sizeX,
        strideZ, strideY, strideX, strideZ1, strideY1, strideX1,
        dstStrideZ, dstStrideY, dstStrideX, activationType);
}

// ---- BinaryFuseAddADD (fp32, atomicAdd) ----
void mnn_corpus_binary_fuseadd_add_fp32(const float* in0, const float* in1, float* out,
                                        int sizeZ, int sizeY, int sizeX,
                                        int strideZ, int strideY, int strideX,
                                        int strideZ1, int strideY1, int strideX1,
                                        int dstStrideZ, int dstStrideY, int dstStrideX,
                                        int grid, int block, cudaStream_t stream) {
    MNN::Corpus::BinaryFuseAddADD<<<grid, block, 0, stream>>>(
        in0, in1, out, sizeZ, sizeY, sizeX,
        strideZ, strideY, strideX, strideZ1, strideY1, strideX1,
        dstStrideZ, dstStrideY, dstStrideX);
}
// ---- BinaryFuseAddMUL (fp32, atomicAdd) ----
void mnn_corpus_binary_fuseadd_mul_fp32(const float* in0, const float* in1, float* out,
                                        int sizeZ, int sizeY, int sizeX,
                                        int strideZ, int strideY, int strideX,
                                        int strideZ1, int strideY1, int strideX1,
                                        int dstStrideZ, int dstStrideY, int dstStrideX,
                                        int grid, int block, cudaStream_t stream) {
    MNN::Corpus::BinaryFuseAddMUL<<<grid, block, 0, stream>>>(
        in0, in1, out, sizeZ, sizeY, sizeX,
        strideZ, strideY, strideX, strideZ1, strideY1, strideX1,
        dstStrideZ, dstStrideY, dstStrideX);
}

// ---- BinaryMidADD (fp32, DivModFast) ----
void mnn_corpus_binarymid_add_fp32(const float* in0, const float* in1, float* out,
                                    int sizeZ, int sizeY, int sizeX,
                                    int strideZ, int strideY, int strideX,
                                    int strideZ1, int strideY1, int strideX1,
                                    int dstStrideZ, int dstStrideY, int dstStrideX, int activationType,
                                    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast dY(sizeY), dX(sizeX);
    MNN::Corpus::BinaryMidADD<float, float><<<grid, block, 0, stream>>>(
        in0, in1, out, sizeZ, sizeY, sizeX,
        strideZ, strideY, strideX, strideZ1, strideY1, strideX1,
        dstStrideZ, dstStrideY, dstStrideX, activationType, dY, dX);
}
// ---- BinaryMidMUL (fp32, DivModFast) ----
void mnn_corpus_binarymid_mul_fp32(const float* in0, const float* in1, float* out,
                                    int sizeZ, int sizeY, int sizeX,
                                    int strideZ, int strideY, int strideX,
                                    int strideZ1, int strideY1, int strideX1,
                                    int dstStrideZ, int dstStrideY, int dstStrideX, int activationType,
                                    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast dY(sizeY), dX(sizeX);
    MNN::Corpus::BinaryMidMUL<float, float><<<grid, block, 0, stream>>>(
        in0, in1, out, sizeZ, sizeY, sizeX,
        strideZ, strideY, strideX, strideZ1, strideY1, strideX1,
        dstStrideZ, dstStrideY, dstStrideX, activationType, dY, dX);
}

// ---- BinaryMidLinear4ADD (fp32, float4 vectorized) ----
void mnn_corpus_binarymidlinear4_add_fp32(const float* in0, const float* in1, float* out,
                                           int count_4, int activationType,
                                           int inp0Broadcast, int inp1Broadcast,
                                           int grid, int block, cudaStream_t stream) {
    MNN::Corpus::BinaryMidLinear4_ADD<float, float><<<grid, block, 0, stream>>>(
        in0, in1, out, count_4, activationType, inp0Broadcast != 0, inp1Broadcast != 0);
}
// ---- BinaryMidLinear4MUL (fp32, float4 vectorized) ----
void mnn_corpus_binarymidlinear4_mul_fp32(const float* in0, const float* in1, float* out,
                                           int count_4, int activationType,
                                           int inp0Broadcast, int inp1Broadcast,
                                           int grid, int block, cudaStream_t stream) {
    MNN::Corpus::BinaryMidLinear4_MUL<float, float><<<grid, block, 0, stream>>>(
        in0, in1, out, count_4, activationType, inp0Broadcast != 0, inp1Broadcast != 0);
}

} // extern "C"
