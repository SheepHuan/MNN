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
BINARY_FUNC(SUB, x-y)
BINARY_FUNC(MUL, x*y)
BINARY_FUNC(DIV, x/y)
BINARY_FUNC(MINIMUM, min(x, y))
BINARY_FUNC(MAXIMUM, max(x, y))
BINARY_FUNC(FLOORDIV, floor(x / y))
BINARY_FUNC(FLOORMOD, x - floor(x / y) * y)
BINARY_FUNC(SquaredDifference, (x-y)*(x-y))
BINARY_FUNC(POW, pow(x, y))

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
BINARY_FUSEADD_FUNC(SUB, x-y)
BINARY_FUSEADD_FUNC(MUL, x*y)
BINARY_FUSEADD_FUNC(DIV, x/y)
BINARY_FUSEADD_FUNC(MINIMUM, min(x, y))
BINARY_FUSEADD_FUNC(MAXIMUM, max(x, y))
BINARY_FUSEADD_FUNC(FLOORDIV, floor(x / y))
BINARY_FUSEADD_FUNC(FLOORMOD, x - floor(x / y) * y)
BINARY_FUSEADD_FUNC(SquaredDifference, (x-y)*(x-y))
BINARY_FUSEADD_FUNC(POW, pow(x, y))

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
BINARY_FUNC_FLOATMID(SUB, x-y)
BINARY_FUNC_FLOATMID(MUL, x*y)
BINARY_FUNC_FLOATMID(MUL_SILU, x*(y/(1.0f+exp(-y))))
BINARY_FUNC_FLOATMID(DIV, x/y)
BINARY_FUNC_FLOATMID(MINIMUM, min(x, y))
BINARY_FUNC_FLOATMID(MAXIMUM, max(x, y))
BINARY_FUNC_FLOATMID(FLOORDIV, floor(x / y))
BINARY_FUNC_FLOATMID(FLOORMOD, x - floor(x / y) * y)
BINARY_FUNC_FLOATMID(SquaredDifference, (x-y)*(x-y))
BINARY_FUNC_FLOATMID(POW, pow(x, y))

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
BINARY_FUNC_FLOATMID4(SUB, x-y)
BINARY_FUNC_FLOATMID4(MUL, x*y)
BINARY_FUNC_FLOATMID4(MUL_SILU, x*(y/(1.0f+exp(-y))))
BINARY_FUNC_FLOATMID4(DIV, x/y)
BINARY_FUNC_FLOATMID4(MINIMUM, min(x, y))
BINARY_FUNC_FLOATMID4(MAXIMUM, max(x, y))
BINARY_FUNC_FLOATMID4(FLOORDIV, floor(x / y))
BINARY_FUNC_FLOATMID4(FLOORMOD, x - floor(x / y) * y)
BINARY_FUNC_FLOATMID4(SquaredDifference, (x-y)*(x-y))
BINARY_FUNC_FLOATMID4(POW, pow(x, y))

// ============================================================================
// BinaryMid4: 3D-stride float4 vectorized (PACK_NUMBER=4 in X dim)
// 忠实复制 MNN Raster.cu:715-765 — ix<<2 + float4 load + stride 不含 X
// ============================================================================
#define BINARY_MID4_FUNC(Name, Func)\
template<typename TIn, typename TOut>\
__global__ void BinaryMid4_##Name(\
    const TIn *input0, const TIn* input1, TOut *output,\
    int sizeZ, int sizeY, int sizeX,\
    int strideZ, int strideY,\
    int strideZ1, int strideY1,\
    int dstStrideZ, int dstStrideY, int activationType,\
    DivModFast d_sizeY, DivModFast d_sizeX,\
    bool inp0Broadcast, bool inp1Broadcast\
    ) { \
    int count = sizeZ * sizeY * sizeX;\
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {\
        int ix, tmp, iy, iz;\
        d_sizeX.divmod(i, tmp, ix);\
        d_sizeY.divmod(tmp, iz, iy);\
        ix = ix << 2;\
        int srcOffset = iz * strideZ + iy * strideY + ix;\
        int srcOffset1 = iz * strideZ1 + iy * strideY1 + ix;\
        int dstOffset = iz * dstStrideZ + iy * dstStrideY + ix;\
        float4 xx = inp0Broadcast ? make_float4(input0[srcOffset-ix],input0[srcOffset-ix], input0[srcOffset-ix], input0[srcOffset-ix]) : ((float4 *)(input0+srcOffset))[0];\
        float4 yy = inp1Broadcast ? make_float4(input1[srcOffset1-ix],input1[srcOffset1-ix], input1[srcOffset1-ix], input1[srcOffset1-ix]) :((float4 *)(input1+srcOffset1))[0];\
        float x = xx.x;\
        float y = yy.x;\
        float val = (float)(Func);\
        if(activationType == 1) { val = (val < 0.0f ? 0.0f : val); }\
        output[dstOffset] = val;\
        x = xx.y; y = yy.y;\
        val = (float)(Func);\
        if(activationType == 1) { val = (val < 0.0f ? 0.0f : val); }\
        output[dstOffset+1] = val;\
        x = xx.z; y = yy.z;\
        val = (float)(Func);\
        if(activationType == 1) { val = (val < 0.0f ? 0.0f : val); }\
        output[dstOffset+2] = val;\
        x = xx.w; y = yy.w;\
        val = (float)(Func);\
        if(activationType == 1) { val = (val < 0.0f ? 0.0f : val); }\
        output[dstOffset+3] = val;\
    }\
}

BINARY_MID4_FUNC(ADD, x+y)
BINARY_MID4_FUNC(MUL, x*y)

// ============================================================================
// BinaryMidHalf2 / BinaryMidLinearHalf4: half2 vectorized variants (fp16 only)
// corpus 在 fp16 adapter 中实例化 <half, half>
// ============================================================================
#define BINARY_MIDHALF2_FUNC(Name, Func)\
template<typename TIn, typename TOut>\
__global__ void BinaryMidHalf2_##Name(\
    const TIn *input0, const TIn* input1, TOut *output,\
    int sizeZ, int sizeY, int sizeX,\
    int strideZ, int strideY,\
    int strideZ1, int strideY1,\
    int dstStrideZ, int dstStrideY, int activationType,\
    DivModFast d_sizeY, DivModFast d_sizeX,\
    bool inp0Broadcast, bool inp1Broadcast\
    ) { \
    int count = sizeZ * sizeY * sizeX;\
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {\
        int ix, tmp, iy, iz;\
        d_sizeX.divmod(i, tmp, ix);\
        d_sizeY.divmod(tmp, iz, iy);\
        ix = ix << 1;\
        int srcOffset = iz * strideZ + iy * strideY + ix;\
        int srcOffset1 = iz * strideZ1 + iy * strideY1 + ix;\
        int dstOffset = iz * dstStrideZ + iy * dstStrideY + ix;\
        half2 xx = inp0Broadcast ? make_half2(input0[srcOffset-ix], input0[srcOffset-ix]) : ((half2 *)(input0+srcOffset))[0];\
        half2 yy = inp1Broadcast ? make_half2(input1[srcOffset1-ix], input1[srcOffset1-ix]) : ((half2 *)(input1+srcOffset1))[0];\
        float x = xx.x; float y = yy.x;\
        float val = (float)(Func);\
        if(activationType == 1) { val = (val < 0.0f ? 0.0f : val); }\
        output[dstOffset] = val;\
        x = xx.y; y = yy.y;\
        val = (float)(Func);\
        if(activationType == 1) { val = (val < 0.0f ? 0.0f : val); }\
        output[dstOffset+1] = val;\
    }\
}

BINARY_MIDHALF2_FUNC(ADD, x+y)
BINARY_MIDHALF2_FUNC(MUL, x*y)

#define BINARY_MIDLINEARHALF4_FUNC(Name, Func)\
template<typename TIn, typename TOut>\
__global__ void BinaryMidLinearHalf4_##Name(\
    const TIn *input0, const TIn* input1, TOut *output,\
    int count_4, int activationType,\
    bool inp0Broadcast, bool inp1Broadcast\
    ) { \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count_4); i += blockDim.x * gridDim.x) {\
        int iz = i;\
        int srcOffset = iz << 2;\
        int srcOffset1 = iz << 2;\
        int dstOffset = iz << 2;\
        half2 xx = inp0Broadcast ? make_half2(input0[0], input0[0]) : ((half2 *)(input0+srcOffset))[0];\
        half2 yy = inp1Broadcast ? make_half2(input1[0], input1[0]) : ((half2 *)(input1+srcOffset1))[0];\
        float x = (float)xx.x; float y = (float)yy.x;\
        float val = (float)(Func);\
        if(activationType == 1) { val = (val < 0.0f ? 0.0f : val); }\
        output[dstOffset] = (TOut)val;\
        x = (float)xx.y; y = (float)yy.y;\
        val = (float)(Func);\
        if(activationType == 1) { val = (val < 0.0f ? 0.0f : val); }\
        output[dstOffset+1] = (TOut)val;\
        xx = inp0Broadcast ? make_half2(input0[0], input0[0]) : ((half2 *)(input0+srcOffset))[1];\
        yy = inp1Broadcast ? make_half2(input1[0], input1[0]) : ((half2 *)(input1+srcOffset1))[1];\
        x = (float)xx.x; y = (float)yy.x;\
        val = (float)(Func);\
        if(activationType == 1) { val = (val <  0.0f ? 0.0f  : val); }\
        output[dstOffset+2] = (TOut)val;\
        x = (float)xx.y; y = (float)yy.y;\
        val = (float)(Func);\
        if(activationType == 1) { val = (val < 0.0f ? 0.0f : val); }\
        output[dstOffset+3] = (TOut)val;\
    }\
}

BINARY_MIDLINEARHALF4_FUNC(ADD, x+y)
BINARY_MIDLINEARHALF4_FUNC(MUL, x*y)

} // namespace Corpus
} // namespace MNN

extern "C" {

// ============================================================================
// Shim 宏：批量生成 extern "C" 启动函数（fp32 实例化 <float,float>）
// ============================================================================

// Basic 3D-stride binary (BINARY_FUNC)
#define SHIM_BINARY(Op, OpLower) \
void mnn_corpus_binary_##OpLower##_fp32(const float* in0, const float* in1, float* out, \
    int sizeZ, int sizeY, int sizeX, \
    int strideZ, int strideY, int strideX, \
    int strideZ1, int strideY1, int strideX1, \
    int dstStrideZ, int dstStrideY, int dstStrideX, int activationType, \
    int grid, int block, cudaStream_t stream) { \
    MNN::Corpus::Binary##Op<float, float><<<grid, block, 0, stream>>>( \
        in0, in1, out, sizeZ, sizeY, sizeX, \
        strideZ, strideY, strideX, strideZ1, strideY1, strideX1, \
        dstStrideZ, dstStrideY, dstStrideX, activationType); \
}

// FuseAdd atomicAdd binary (BINARY_FUSEADD_FUNC)
#define SHIM_BINARY_FUSEADD(Op, OpLower) \
void mnn_corpus_binary_fuseadd_##OpLower##_fp32(const float* in0, const float* in1, float* out, \
    int sizeZ, int sizeY, int sizeX, \
    int strideZ, int strideY, int strideX, \
    int strideZ1, int strideY1, int strideX1, \
    int dstStrideZ, int dstStrideY, int dstStrideX, \
    int grid, int block, cudaStream_t stream) { \
    MNN::Corpus::BinaryFuseAdd##Op<<<grid, block, 0, stream>>>( \
        in0, in1, out, sizeZ, sizeY, sizeX, \
        strideZ, strideY, strideX, strideZ1, strideY1, strideX1, \
        dstStrideZ, dstStrideY, dstStrideX); \
}

// BinaryMid (BINARY_FUNC_FLOATMID): has DivModFast + activationType
#define SHIM_BINARYMID(Op, OpLower) \
void mnn_corpus_binarymid_##OpLower##_fp32(const float* in0, const float* in1, float* out, \
    int sizeZ, int sizeY, int sizeX, \
    int strideZ, int strideY, int strideX, \
    int strideZ1, int strideY1, int strideX1, \
    int dstStrideZ, int dstStrideY, int dstStrideX, int activationType, \
    int grid, int block, cudaStream_t stream) { \
    MNN::Corpus::DivModFast dY(sizeY), dX(sizeX); \
    MNN::Corpus::BinaryMid##Op<float, float><<<grid, block, 0, stream>>>( \
        in0, in1, out, sizeZ, sizeY, sizeX, \
        strideZ, strideY, strideX, strideZ1, strideY1, strideX1, \
        dstStrideZ, dstStrideY, dstStrideX, activationType, dY, dX); \
}

// BinaryMidLinear4 (BINARY_FUNC_FLOATMID4): float4 vectorized
#define SHIM_BINARYMIDLINEAR4(Op, OpLower) \
void mnn_corpus_binarymidlinear4_##OpLower##_fp32(const float* in0, const float* in1, float* out, \
    int count_4, int activationType, \
    int inp0Broadcast, int inp1Broadcast, \
    int grid, int block, cudaStream_t stream) { \
    MNN::Corpus::BinaryMidLinear4_##Op<float, float><<<grid, block, 0, stream>>>( \
        in0, in1, out, count_4, activationType, inp0Broadcast != 0, inp1Broadcast != 0); \
}

// ---- Binary* instantiations ----
SHIM_BINARY(ADD, add)
SHIM_BINARY(SUB, sub)
SHIM_BINARY(MUL, mul)
SHIM_BINARY(DIV, div)
SHIM_BINARY(MINIMUM, minimum)
SHIM_BINARY(MAXIMUM, maximum)
SHIM_BINARY(FLOORDIV, floordiv)
SHIM_BINARY(FLOORMOD, floormod)
SHIM_BINARY(SquaredDifference, squared_difference)
SHIM_BINARY(POW, pow)

// ---- BinaryFuseAdd* instantiations ----
SHIM_BINARY_FUSEADD(ADD, add)
SHIM_BINARY_FUSEADD(SUB, sub)
SHIM_BINARY_FUSEADD(MUL, mul)
SHIM_BINARY_FUSEADD(DIV, div)
SHIM_BINARY_FUSEADD(MINIMUM, minimum)
SHIM_BINARY_FUSEADD(MAXIMUM, maximum)
SHIM_BINARY_FUSEADD(FLOORDIV, floordiv)
SHIM_BINARY_FUSEADD(FLOORMOD, floormod)
SHIM_BINARY_FUSEADD(SquaredDifference, squared_difference)
SHIM_BINARY_FUSEADD(POW, pow)

// ---- BinaryMid* instantiations ----
SHIM_BINARYMID(ADD, add)
SHIM_BINARYMID(SUB, sub)
SHIM_BINARYMID(MUL, mul)
SHIM_BINARYMID(MUL_SILU, mul_silu)
SHIM_BINARYMID(DIV, div)
SHIM_BINARYMID(MINIMUM, minimum)
SHIM_BINARYMID(MAXIMUM, maximum)
SHIM_BINARYMID(FLOORDIV, floordiv)
SHIM_BINARYMID(FLOORMOD, floormod)
SHIM_BINARYMID(SquaredDifference, squared_difference)
SHIM_BINARYMID(POW, pow)

// ---- BinaryMidLinear4* instantiations ----
SHIM_BINARYMIDLINEAR4(ADD, add)
SHIM_BINARYMIDLINEAR4(SUB, sub)
SHIM_BINARYMIDLINEAR4(MUL, mul)
SHIM_BINARYMIDLINEAR4(MUL_SILU, mul_silu)
SHIM_BINARYMIDLINEAR4(DIV, div)
SHIM_BINARYMIDLINEAR4(MINIMUM, minimum)
SHIM_BINARYMIDLINEAR4(MAXIMUM, maximum)
SHIM_BINARYMIDLINEAR4(FLOORDIV, floordiv)
SHIM_BINARYMIDLINEAR4(FLOORMOD, floormod)
SHIM_BINARYMIDLINEAR4(SquaredDifference, squared_difference)
SHIM_BINARYMIDLINEAR4(POW, pow)

// ---- BinaryMid4 (float4, PACK_NUMBER=4 in X dim) ----
void mnn_corpus_binarymid4_add_fp32(const float* in0, const float* in1, float* out,
                                     int sizeZ, int sizeY, int sizeX,
                                     int strideZ, int strideY,
                                     int strideZ1, int strideY1,
                                     int dstStrideZ, int dstStrideY, int activationType,
                                     int inp0Broadcast, int inp1Broadcast,
                                     int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast dY(sizeY), dX(sizeX);
    MNN::Corpus::BinaryMid4_ADD<float, float><<<grid, block, 0, stream>>>(
        in0, in1, out, sizeZ, sizeY, sizeX,
        strideZ, strideY, strideZ1, strideY1,
        dstStrideZ, dstStrideY, activationType, dY, dX,
        inp0Broadcast != 0, inp1Broadcast != 0);
}
void mnn_corpus_binarymid4_mul_fp32(const float* in0, const float* in1, float* out,
                                     int sizeZ, int sizeY, int sizeX,
                                     int strideZ, int strideY,
                                     int strideZ1, int strideY1,
                                     int dstStrideZ, int dstStrideY, int activationType,
                                     int inp0Broadcast, int inp1Broadcast,
                                     int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast dY(sizeY), dX(sizeX);
    MNN::Corpus::BinaryMid4_MUL<float, float><<<grid, block, 0, stream>>>(
        in0, in1, out, sizeZ, sizeY, sizeX,
        strideZ, strideY, strideZ1, strideY1,
        dstStrideZ, dstStrideY, activationType, dY, dX,
        inp0Broadcast != 0, inp1Broadcast != 0);
}

// ---- BinaryMidHalf2 (fp16 half2, PACK_NUMBER=2) ----
void mnn_corpus_binarymidhalf2_add_fp16(const __half* in0, const __half* in1, __half* out,
                                         int sizeZ, int sizeY, int sizeX,
                                         int strideZ, int strideY,
                                         int strideZ1, int strideY1,
                                         int dstStrideZ, int dstStrideY, int activationType,
                                         int inp0Broadcast, int inp1Broadcast,
                                         int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast dY(sizeY), dX(sizeX);
    MNN::Corpus::BinaryMidHalf2_ADD<__half, __half><<<grid, block, 0, stream>>>(
        in0, in1, out, sizeZ, sizeY, sizeX,
        strideZ, strideY, strideZ1, strideY1,
        dstStrideZ, dstStrideY, activationType, dY, dX,
        inp0Broadcast != 0, inp1Broadcast != 0);
}
void mnn_corpus_binarymidhalf2_mul_fp16(const __half* in0, const __half* in1, __half* out,
                                         int sizeZ, int sizeY, int sizeX,
                                         int strideZ, int strideY,
                                         int strideZ1, int strideY1,
                                         int dstStrideZ, int dstStrideY, int activationType,
                                         int inp0Broadcast, int inp1Broadcast,
                                         int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast dY(sizeY), dX(sizeX);
    MNN::Corpus::BinaryMidHalf2_MUL<__half, __half><<<grid, block, 0, stream>>>(
        in0, in1, out, sizeZ, sizeY, sizeX,
        strideZ, strideY, strideZ1, strideY1,
        dstStrideZ, dstStrideY, activationType, dY, dX,
        inp0Broadcast != 0, inp1Broadcast != 0);
}

// ---- BinaryMidLinearHalf4 (fp16 half2 x2, 1D linear) ----
void mnn_corpus_binarymidlinearhalf4_add_fp16(const __half* in0, const __half* in1, __half* out,
                                                int count_4, int activationType,
                                                int inp0Broadcast, int inp1Broadcast,
                                                int grid, int block, cudaStream_t stream) {
    MNN::Corpus::BinaryMidLinearHalf4_ADD<__half, __half><<<grid, block, 0, stream>>>(
        in0, in1, out, count_4, activationType, inp0Broadcast != 0, inp1Broadcast != 0);
}
void mnn_corpus_binarymidlinearhalf4_mul_fp16(const __half* in0, const __half* in1, __half* out,
                                                int count_4, int activationType,
                                                int inp0Broadcast, int inp1Broadcast,
                                                int grid, int block, cudaStream_t stream) {
    MNN::Corpus::BinaryMidLinearHalf4_MUL<__half, __half><<<grid, block, 0, stream>>>(
        in0, in1, out, count_4, activationType, inp0Broadcast != 0, inp1Broadcast != 0);
}

} // extern "C"
