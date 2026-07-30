// conv_base.cu - Float22Half2 / Float22BFloat16 / Im2Col_FilterC / Im2Col_FilterC_Vec4 / WeightPackFill (3.6.0)
//   source/backend/cuda/execution/ConvBaseKernel.cu
//   Float22Half2: float2→half2 packing (4 floats → 4 halves per thread)
//   Float22BFloat16: float2→bfloat162 packing (requires sm80+)
//   Im2Col_FilterC: im2col for convolution (scalar version)
//   Im2Col_FilterC_Vec4: im2col with vec4 copy (precision-aware)
//   WeightPackFill: weight reordering [Co,Ci,KhKw]→[Co,KhKw,Ci]
#include "corpus_common.cuh"
#include <cuda_bf16.h>

// 1.2.7-era legacy im2col macros (from legacy_kernels.cu). conv_base.cu itself
// does not otherwise define PACK_NUMBER.
#ifndef PACK_NUMBER
#define PACK_NUMBER 16
#endif
#define PACK_NUMBER_C2 (PACK_NUMBER/2)
#define MATMULPACK 16
#define MATMULPACK2 (MATMULPACK * MATMULPACK)
#define BLOCK_INT4 2

namespace MNN {
namespace Corpus {

// 1.2.7-era MatMul + Im2Col parameter structs (from legacy_kernels.cu)
struct MatMulParam {
    int elh[3]; int elhPack[3]; int aStride[3]; int bStride[3]; int cStride[3];
    int aPStride[3]; int bPStride[3]; int batch; float minValue; float maxValue;
};
struct Im2ColParameter {
    int32_t padX; int32_t padY; int32_t dilateX; int32_t dilateY;
    int32_t strideX; int32_t strideY; int32_t kernelX; int32_t kernelY;
    int32_t icDiv4; int32_t kernelCountUnit; int32_t iw; int32_t ih;
    int32_t ow; int32_t oh; int32_t srcZStep; int32_t srcYStep;
    int32_t packCUnit; int32_t destICStride;
};

// ============================================================================
// Float22Half2: convert float array to half array (4 elements per thread)
// ============================================================================
__global__ void Float22Half2(const float* param, half* output, const size_t maxCount) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        float2* srcPtr = (float2*)(param + (index << 2));
        half2* dstPtr = (half2*)(output + (index << 2));
        dstPtr[0] = __float22half2_rn(srcPtr[0]);
        dstPtr[1] = __float22half2_rn(srcPtr[1]);
    }
}

// ============================================================================
// Float22BFloat16: convert float array to bfloat16 array (4 elements per thread)
// ============================================================================
__global__ void Float22BFloat16(const float* param, __nv_bfloat16* output, const size_t maxCount) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        float2* srcPtr = (float2*)(param + (index << 2));
        __nv_bfloat162* dstPtr = (__nv_bfloat162*)(output + (index << 2));
        dstPtr[0] = __float22bfloat162_rn(srcPtr[0]);
        dstPtr[1] = __float22bfloat162_rn(srcPtr[1]);
    }
#endif
}

// ============================================================================
// Im2Col_FilterC: im2col for convolution (scalar, per-element)
// ============================================================================
template<typename T0, typename T1>
__global__ void Im2Col_FilterC(
    const int sw, const int sh, const int dw, const int dh,
    const int pw, const int ph, const int icDiv4,
    const int iw, const int ih, const int ic,
    const size_t maxCount, const int pack, const int e, const int l, const int l_p,
    const T0* A, T1* AP,
    DivModFast d_lp, DivModFast d_ow, DivModFast d_oh, DivModFast d_fx, DivModFast d_ic
) {
    for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
        int eIndex, lpIndex;
        d_lp.divmod(indexO, eIndex, lpIndex);
        if (eIndex >= e || lpIndex >= l) {
            *(AP + indexO) = (T1)0.0f;
            continue;
        }
        int ox, oby, ob, oy, iz, kI, ksx, ksy;
        d_ow.divmod(eIndex, oby, ox);
        d_oh.divmod(oby, ob, oy);
        d_ic.divmod(lpIndex, kI, iz);
        d_fx.divmod(kI, ksy, ksx);
        size_t sx = ox * sw + ksx * dw - pw;
        size_t sy = oy * sh + ksy * dh - ph;
        const int ic_p = icDiv4 * pack;
        size_t dst_offset = eIndex * l_p + kI * ic + iz;
        if (sx >= 0 && sx < iw) {
            if (sy >= 0 && sy < ih) {
                size_t offset = ((ob * ih + sy) * iw + sx) * ic_p + iz;
                *(AP + dst_offset) = (T1)(*(A + offset));
                continue;
            }
        }
        *(AP + dst_offset) = (T1)0.0f;
    }
}

// ============================================================================
// Im2Col_FilterC_Vec4: im2col with vec4 copy (precision-aware).
//   precision: 0 = float→half4, 1 = float4→float4, 2 = half4→half4
// ============================================================================
template<typename T0, typename T>
__global__ void Im2Col_FilterC_Vec4(
    const int sw, const int sh, const int dw, const int dh,
    const int pw, const int ph, const int icDiv4,
    const int iw, const int ih, const int ic,
    const size_t maxCount, const int pack, const int e, const int l, const int l_p,
    const T0* A, T* AP, const int precision,
    DivModFast d_lp, DivModFast d_ow, DivModFast d_oh, DivModFast d_fx, DivModFast d_ic4
) {
    for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
        int eIndex, lpIndex;
        d_lp.divmod(indexO, eIndex, lpIndex);
        int ox, oby, ob, oy, iz_4, kI, ksx, ksy;
        d_ow.divmod(eIndex, oby, ox);
        d_oh.divmod(oby, ob, oy);
        d_ic4.divmod(lpIndex, kI, iz_4);
        d_fx.divmod(kI, ksy, ksx);
        size_t sx = ox * sw + ksx * dw - pw;
        size_t sy = oy * sh + ksy * dh - ph;
        const int ic_p = icDiv4 * pack;
        const int iz = iz_4 << 2;
        size_t dst_offset = eIndex * l_p + kI * ic + iz;
        if (sx >= 0 && sx < (size_t)iw && sy >= 0 && sy < (size_t)ih) {
            size_t src_offset = ((ob * ih + sy) * iw + sx) * ic_p + iz;
            if (precision == 1) {
                *((float4*)((float*)AP + dst_offset)) = *((float4*)((float*)A + src_offset));
            } else if (precision == 2) {
                // half4 -> half4 via int64 copy (faithful to MNN DATA_CONVERT_COPY)
                *((int64_t*)((half*)AP + dst_offset)) = *((int64_t*)((half*)A + src_offset));
            } else if (precision == 3) {
                // bf16: same int64 copy as precision==2 (MNN DATA_CONVERT_COPY L77)
                #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
                *((int64_t*)((half*)AP + dst_offset)) = *((int64_t*)((half*)A + src_offset));
                #endif
            } else if (precision == 0) {
                *((half2*)((half*)AP + dst_offset))     = __float22half2_rn(*((float2*)((float*)A + src_offset)));
                *((half2*)((half*)AP + dst_offset + 2)) = __float22half2_rn(*((float2*)((float*)A + src_offset + 2)));
            }
            continue;
        }
        if (precision == 1) {
            float4 zeros; zeros.x = zeros.y = zeros.z = zeros.w = 0.0f;
            *((float4*)((float*)AP + dst_offset)) = zeros;
        } else if (precision == 2 || precision == 0) {
            half2 zeros; zeros.x = (half)0.0f; zeros.y = (half)0.0f;
            *((half2*)((half*)AP + dst_offset))     = zeros;
            *((half2*)((half*)AP + dst_offset + 2)) = zeros;
        } else if (precision == 3) {
            // bf16 zero-fill: __nv_bfloat162 (MNN DATA_MEMSET_ZERO + extra bf16 path)
            #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
            __nv_bfloat162 zeros; zeros.x = (__nv_bfloat16)0.0f; zeros.y = (__nv_bfloat16)0.0f;
            *((__nv_bfloat162*)((__nv_bfloat16*)AP + dst_offset)) = zeros;
            *((__nv_bfloat162*)((__nv_bfloat16*)AP + dst_offset + 2)) = zeros;
            #endif
        }
    }
}

// ============================================================================
// WeightPackFill: weight reordering [Co,Ci,KhKw]→[Co,KhKw,Ci]
// ============================================================================
template<typename T0, typename T>
__global__ void WeightPackFill(const T0* param, T* output, const int khw, const size_t maxCount,
                               const int l, const int h, DivModFast d_lp, DivModFast d_ic) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int lpIndex, hpIndex, icIndex, khwIndex;
        d_lp.divmod(index, hpIndex, lpIndex);
        if (lpIndex >= l || hpIndex >= h) {
            output[index] = (T)0.0f;
            continue;
        }
        d_ic.divmod(lpIndex, khwIndex, icIndex);
        output[index] = param[hpIndex * l + icIndex * khw + khwIndex];
    }
}

// ============================================================================
// PackPadFill: matrix pad+pack preprocessing for MatMul (3.6.0)
// ============================================================================
template<typename T0, typename T1>
__global__ void PackPadFill(
    const T0* A, const T0* B,
    bool transA, bool transB,
    T1* tempA, T1* tempB, const int batchA, const int batchB,
    const int e, const int l, const int h,
    const int ep, const int lp, const int hp,
    DivModFast d_e, DivModFast d_l, DivModFast d_h,
    DivModFast d_lp, DivModFast d_lp2
) {
    T1 zero = (T1)0.0f;
    if ((char*)A != (char*)tempA) {
        if (transA) {
            const int maxCount = batchA * e * lp;
            for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                int bIndex, lpIndex, eIndex, tmp;
                d_lp.divmod(index, tmp, lpIndex);
                d_e.divmod(tmp, bIndex, eIndex);
                if (lpIndex >= l) { tempA[index] = zero; continue; }
                tempA[index] = A[bIndex * e * l + lpIndex * e + eIndex];
            }
        } else {
            if (l & 1 == 0) {
                // vec2 packed path (faithful to MNN): 2 elements per thread
                const int maxCount = batchA * e * (lp >> 1);
                for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                    int lp2Index, eIndex, bIndex, tmp;
                    d_lp2.divmod(index, tmp, lp2Index);
                    d_e.divmod(tmp, bIndex, eIndex);
                    if (lp2Index + lp2Index >= l) {
                        tempA[index + index] = zero;
                        tempA[index + index + 1] = zero;
                        continue;
                    }
                    tempA[index + index] = A[bIndex * e * l + eIndex * l + lp2Index + lp2Index];
                    tempA[index + index + 1] = A[bIndex * e * l + eIndex * l + lp2Index + lp2Index + 1];
                }
            } else {
                const int maxCount = batchA * e * lp;
                for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                    int lpIndex, eIndex, bIndex, tmp;
                    d_lp.divmod(index, tmp, lpIndex);
                    d_e.divmod(tmp, bIndex, eIndex);
                    if (lpIndex >= l || eIndex >= e) { tempA[index] = zero; continue; }
                    tempA[index] = A[bIndex * e * l + eIndex * l + lpIndex];
                }
            }
        }
    }
    if ((char*)B != (char*)tempB) {
        if (!transB) {
            const int maxCount = batchB * lp * h;
            if (h == hp) {
                for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                    int lpIndex, hpIndex, bIndex, tmp;
                    d_h.divmod(index, tmp, hpIndex);
                    d_lp.divmod(tmp, bIndex, lpIndex);
                    if (lpIndex >= l || hpIndex >= h) { tempB[index] = zero; continue; }
                    tempB[index] = B[bIndex * h * l + lpIndex * h + hpIndex];
                }
            } else {
                for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                    int lpIndex, hIndex, bIndex, tmp;
                    d_lp.divmod(index, tmp, lpIndex);
                    d_h.divmod(tmp, bIndex, hIndex);
                    if (lpIndex >= l || hIndex >= h) { tempB[index] = zero; continue; }
                    tempB[index] = B[bIndex * h * l + lpIndex * h + hIndex];
                }
            }
        } else {
            if (l & 1 == 0) {
                // vec2 packed path (faithful to MNN): 2 elements per thread
                const int maxCount = batchB * h * (lp >> 1);
                for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                    int lp2Index, hIndex, bIndex, tmp;
                    d_lp2.divmod(index, tmp, lp2Index);
                    d_h.divmod(tmp, bIndex, hIndex);
                    if (lp2Index + lp2Index >= l) {
                        tempB[index + index] = zero;
                        tempB[index + index + 1] = zero;
                        continue;
                    }
                    tempB[index + index] = B[bIndex * h * l + hIndex * l + lp2Index + lp2Index];
                    tempB[index + index + 1] = B[bIndex * h * l + hIndex * l + lp2Index + lp2Index + 1];
                }
            } else {
                const int maxCount = batchB * h * lp;
                for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
                    int lpIndex, hIndex, bIndex, tmp;
                    d_lp.divmod(index, tmp, lpIndex);
                    d_h.divmod(tmp, bIndex, hIndex);
                    if (lpIndex >= l || hIndex >= h) { tempB[index] = zero; continue; }
                    tempB[index] = B[bIndex * h * l + hIndex * l + lpIndex];
                }
            }
        }
    }
}

// ============================================================================
// WeightPackFill_Implicit: weight reordering [Co,Ci,KhKw]→[Cop,KhKw,Cip] for implicit conv
// ============================================================================
template<typename T>
__global__ void WeightPackFill_Implicit(const float* param, T* output,
                                        const int khw, const size_t maxCount,
                                        const int ci, const int co,
                                        DivModFast d_cip, DivModFast d_khw) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int copIndex, hpIndex, cipIndex, khwIndex;
        d_cip.divmod(index, hpIndex, cipIndex);
        d_khw.divmod(hpIndex, copIndex, khwIndex);
        if (cipIndex >= ci || copIndex >= co) {
            output[index] = (T)0.0f;
            continue;
        }
        output[index] = param[(copIndex * ci + cipIndex) * khw + khwIndex];
    }
}

// ============================================================================
// 1.2.7-era legacy im2col kernels (from legacy_kernels.cu)
// ============================================================================
__global__ void Im2Col1x1(const Im2ColParameter* param,
    const MatMulParam* matmulParam,
    const float* A,
    half* AP,
    DivModFast eAlignD,
    DivModFast owD,
    DivModFast ohD
    ) {
    int eAlign = matmulParam->elhPack[0] * MATMULPACK;
    int lAlign = matmulParam->elhPack[1];
    int maxCount = eAlign * lAlign * BLOCK_INT4;
    int kernelCount = 1;
    for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
        int index = indexO >> 1;
        int lR = indexO & 1;
        int eIndex, lIndex;
        eAlignD.divmod(index, lIndex, eIndex);
        int eU = eIndex >> 4;
        int eR = eIndex & 15;
        int dstOffset = eU * matmulParam->elhPack[1] * (MATMULPACK * MATMULPACK) + lIndex * (MATMULPACK * MATMULPACK) + eR * MATMULPACK + lR * 8;
        int4* dst = (int4*)(AP + dstOffset);
        if (eIndex >= matmulParam->elh[0]) {
            *dst = {0, 0, 0, 0};
            continue;
        }
        // Compute for source
        int ox, oy, ob;
        owD.divmod(eIndex, oy, ox);
        ohD.divmod(oy, ob, oy);
        int sz = lIndex;
        int sx = ox * param->strideX - param->padX;
        int sy = oy * param->strideY - param->padY;
        if (sx >= 0 && sx < param->iw) {
            if (sy >=0 && sy < param->ih) {
                int offset = sz * param->srcZStep + (ob * param->iw * param->ih + sy * param->iw + sx) * PACK_NUMBER + lR * 8;
                float2* srcF = (float2*)(A + offset);
                half2* dstH = (half2*)dst;
                dstH[0] = __float22half2_rn(srcF[0]);
                dstH[1] = __float22half2_rn(srcF[1]);
                dstH[2] = __float22half2_rn(srcF[2]);
                dstH[3] = __float22half2_rn(srcF[3]);
                continue;
            }
        }
        *dst = {0, 0, 0, 0};
    }
}

__global__ void Im2Col1x1_OPT(const Im2ColParameter* param,
    const MatMulParam* matmulParam,
    const int maxCount, 
    const float* A,
    half* AP,
    DivModFast eAlignD,
    DivModFast owD,
    DivModFast ohD
    ) {
    for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
        int index = indexO >> 3;
        int lR = indexO & 7;
        int eIndex, lIndex;
        eAlignD.divmod(index, lIndex, eIndex);
        int eU = eIndex >> 4;
        int eR = eIndex & 15;
        int dstOffset = ((eU * matmulParam->elhPack[1] + lIndex) << 8) + (eR << 4) + (lR << 1);

        int offset = lIndex * param->srcZStep + (eIndex << 4) + (lR << 1);
        float2* srcF = (float2*)(A + offset);
        half2* dstH = (half2*)(AP + dstOffset);
        dstH[0] = __float22half2_rn(srcF[0]);
    }
}

__global__ void Im2Col1x1_half(const Im2ColParameter* param,
    const MatMulParam* matmulParam,
    const half* A,
    half* AP,
    DivModFast eAlignD,
    DivModFast owD,
    DivModFast ohD
    ) {
int eAlign = matmulParam->elhPack[0] * MATMULPACK;
int lAlign = matmulParam->elhPack[1];
int maxCount = eAlign * lAlign * BLOCK_INT4;
int kernelCount = 1;
for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
    int index = indexO / BLOCK_INT4;
    int lR = indexO % BLOCK_INT4;
    int eIndex, lIndex;
    eAlignD.divmod(index, lIndex, eIndex);
    int eU = eIndex / MATMULPACK;
    int eR = eIndex % MATMULPACK;
    int dstOffset = eU * matmulParam->elhPack[1] * (MATMULPACK * MATMULPACK) + lIndex * (MATMULPACK * MATMULPACK) + eR * MATMULPACK + lR * 8;
    int4* dst = (int4*)(AP + dstOffset);
    if (eIndex >= matmulParam->elh[0]) {
        *dst = {0, 0, 0, 0};
        continue;
    }
    // Compute for source
    int ox, oy, ob;
    owD.divmod(eIndex, oy, ox);
    ohD.divmod(oy, ob, oy);
    int sz = lIndex;
    int sx = ox * param->strideX - param->padX;
    int sy = oy * param->strideY - param->padY;
    if (sx >= 0 && sx < param->iw) {
        if (sy >=0 && sy < param->ih) {
            int offset = sz * param->srcZStep + (ob * param->iw * param->ih + sy * param->iw + sx) * PACK_NUMBER + lR * 8;
            int4* src = (int4*)(A + offset);
            *dst = *src;
            continue;
        }
    }
    *dst = {0, 0, 0, 0};
}
}

__global__ void Im2Col1x1_half_OPT(const Im2ColParameter* param,
const MatMulParam* matmulParam,
const int maxCount, 
const half* A,
half* AP,
DivModFast eAlignD,
DivModFast owD,
DivModFast ohD
) {
for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
    int index = indexO >> 3;
    int lR = indexO & 7;
    int eIndex, lIndex;
    eAlignD.divmod(index, lIndex, eIndex);
    int eU = eIndex >> 4;
    int eR = eIndex & 15;
    int dstOffset = ((eU * matmulParam->elhPack[1] + lIndex) << 8) + (eR << 4) + (lR << 1);

    int offset = lIndex * param->srcZStep + (eIndex << 4) + (lR << 1);
    int* srcF = (int*)(A + offset);
    int* dstH = (int*)(AP + dstOffset);
    dstH[0] = srcF[0];
}
}

__global__ void Im2Col_half(const Im2ColParameter* param,
    const MatMulParam* matmulParam,
    const int maxCount,
    const half* A,
    half* AP,
    DivModFast d_eA,
    DivModFast d_ow,
    DivModFast d_oh,
    DivModFast d_fxy,
    DivModFast d_fx
    ) {
int eAlign = matmulParam->elhPack[0] << 4;
int lAlign = matmulParam->elhPack[1];
int kernelCount = param->kernelX * param->kernelY;
for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
    size_t index = indexO >> 1;
    size_t lR = indexO & 1;
    int eIndex, lIndex;
    d_eA.divmod(index, lIndex, eIndex);
    size_t eU = eIndex >> 4;
    size_t eR = eIndex & 15;
    size_t dstOffset = ((((eU * matmulParam->elhPack[1] + lIndex) << 4) + eR) << 4) + (lR << 3);
    int4* dst = (int4*)(AP + dstOffset);
    if (eIndex >= matmulParam->elh[0]) {
        *dst = {0, 0, 0, 0};
        continue;
    }
    // Compute for source
    int ox, oby, ob, oy, sz, kI, ksx, ksy;
    d_ow.divmod(eIndex, oby, ox);
    d_oh.divmod(oby, ob, oy);
    d_fxy.divmod(lIndex, sz, kI);
    d_fx.divmod(kI, ksy, ksx);

    size_t sx = ox * param->strideX + ksx * param->dilateX - param->padX;
    size_t sy = oy * param->strideY + ksy * param->dilateY - param->padY;
    if (sx >= 0 && sx < param->iw) {
        if (sy >=0 && sy < param->ih) {
            size_t offset = sz * param->srcZStep + (((ob * param->ih + sy) * param->iw + sx) << 4) + lR * 8;
            int4* src = (int4*)(A + offset);
            *dst = *src;
            continue;
        }
    }
    *dst = {0, 0, 0, 0};
}
}

__global__ void Im2Col_half_OPT(const Im2ColParameter* param,
    const MatMulParam* matmulParam,
    const size_t maxCount,
    const half* A,
    half* AP,
    DivModFast d_eA,
    DivModFast d_ow,
    DivModFast d_oh,
    DivModFast d_fxy,
    DivModFast d_fx
) {
size_t eAlign = matmulParam->elhPack[0] << 4;
size_t lAlign = matmulParam->elhPack[1];
size_t kernelCount = param->kernelX * param->kernelY;
for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
    size_t index = indexO >> 2;
    size_t lR = indexO & 3;
    int eIndex, lIndex;
    d_eA.divmod(index, lIndex, eIndex);
    size_t eU = eIndex >> 4;
    size_t eR = eIndex & 15;
    size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex) << 4) + eR) << 4) + (lR << 2);
    int2* dst = (int2*)(AP + dstOffset);
    if (eIndex >= matmulParam->elh[0]) {
        *dst = {0, 0};
        continue;
    }

    // Compute for source
    int ox, oby, ob, oy, sz, kI, ksx, ksy;
    d_ow.divmod(eIndex, oby, ox);
    d_oh.divmod(oby, ob, oy);
    d_fxy.divmod(lIndex, sz, kI);
    d_fx.divmod(kI, ksy, ksx);

    size_t sx = ox * param->strideX + ksx * param->dilateX - param->padX;
    size_t sy = oy * param->strideY + ksy * param->dilateY - param->padY;
    if (sx >= 0 && sx < param->iw) {
        if (sy >=0 && sy < param->ih) {
            size_t offset = sz * param->srcZStep + (((ob * param->ih + sy) * param->iw + sx) << 4) + (lR << 2);
            int2* src = (int2*)(A + offset);
            *dst = *src;
            continue;
        }
    }
    *dst = {0, 0};
}
}

__global__ void Im2Col_half_3x3S1D1P1_OPT2(const Im2ColParameter* param,
const MatMulParam* matmulParam,
const size_t maxCount,
const half* A,
half* AP,
DivModFast d_eA,
DivModFast d_ow,
DivModFast d_oh
) {
for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
size_t index = indexO >> 3;
size_t lR = indexO & 7;
int eIndex, lIndex;
d_eA.divmod(index, lIndex, eIndex);

int ix, oby, ob, iy;
d_ow.divmod(eIndex, oby, ix);
d_oh.divmod(oby, ob, iy);
size_t sz = lIndex;

size_t offset = sz * param->srcZStep + (((ob * param->ih + iy) * param->iw + ix) << 4) + (lR << 1);
int src = *((int*)(A + offset));

// Pixel (iy-1, ix-1)
if(iy-1 >=0 && ix-1 >=0) {
    size_t oeIndex = (ob * param->ih * param->iw + (iy-1) * param->iw + (ix-1));
    size_t eU = oeIndex >> 4;
    size_t eR = oeIndex & 15;
    size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + 8) << 4) + eR) << 4) + (lR << 1);
    int* dst = (int*)(AP + dstOffset);
    *dst = src;

    // Corner case
    if(iy-1 ==0) {
        size_t index[3] = {0, 1, 2};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix-1 ==0) {
        size_t index[3] = {0, 3, 6};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
}

// Pixel (iy-1, ix+0)
if(iy-1 >=0) {
    size_t oeIndex = (ob * param->ih * param->iw + (iy-1) * param->iw + (ix+0));
    size_t eU = oeIndex >> 4;
    size_t eR = oeIndex & 15;
    size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + 7) << 4) + eR) << 4) + (lR << 1);
    int* dst = (int*)(AP + dstOffset);
    *dst = src;

    // Corner case
    if(iy-1 ==0) {
        size_t index[3] = {0, 1, 2};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix ==0) {
        size_t index[3] = {0, 3, 6};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix == param->iw-1) {
        size_t index[3] = {2, 5, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
}

// Pixel (iy-1, ix+1)
if(iy-1 >=0 && ix+1 < param->iw) {
    size_t oeIndex = (ob * param->ih * param->iw + (iy-1) * param->iw + (ix+1));
    size_t eU = oeIndex >> 4;
    size_t eR = oeIndex & 15;
    size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + 6) << 4) + eR) << 4) + (lR << 1);
    int* dst = (int*)(AP + dstOffset);
    *dst = src;

    // Corner case
    if(iy-1 ==0) {
        size_t index[3] = {0, 1, 2};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix+1 == param->iw-1) {
        size_t index[3] = {2, 5, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
}

// Pixel (iy+0, ix-1)
if(ix-1 >=0) {
    size_t oeIndex = (ob * param->ih * param->iw + (iy+0) * param->iw + (ix-1));
    size_t eU = oeIndex >> 4;
    size_t eR = oeIndex & 15;
    size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + 5) << 4) + eR) << 4) + (lR << 1);
    int* dst = (int*)(AP + dstOffset);
    *dst = src;

    // Corner case
    if(iy ==0) {
        size_t index[3] = {0, 1, 2};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(iy == param->ih-1) {
        size_t index[3] = {6, 7, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix-1 ==0) {
        size_t index[3] = {0, 3, 6};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
}

// Pixel (iy, ix)
if(1) {
    size_t oeIndex = (ob * param->ih * param->iw + (iy+0) * param->iw + (ix+0));
    size_t eU = oeIndex >> 4;
    size_t eR = oeIndex & 15;
    size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + 4) << 4) + eR) << 4) + (lR << 1);
    int* dst = (int*)(AP + dstOffset);
    *dst = src;

    // Corner case
    if(iy ==0) {
        size_t index[3] = {0, 1, 2};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(iy == param->ih-1) {
        size_t index[3] = {6, 7, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix ==0) {
        size_t index[3] = {0, 3, 6};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix == param->iw-1) {
        size_t index[3] = {2, 5, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
}

// Pixel (iy, ix+1)
if(ix+1 < param->iw) {
    size_t oeIndex = (ob * param->ih * param->iw + (iy+0) * param->iw + (ix+1));
    size_t eU = oeIndex >> 4;
    size_t eR = oeIndex & 15;
    size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + 3) << 4) + eR) << 4) + (lR << 1);
    int* dst = (int*)(AP + dstOffset);
    *dst = src;

    // Corner case
    if(iy ==0) {
        size_t index[3] = {0, 1, 2};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(iy == param->ih-1) {
        size_t index[3] = {6, 7, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix+1 == param->iw-1) {
        size_t index[3] = {2, 5, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
}

// Pixel (iy+1, ix-1)
if(iy+1 < param->ih && ix-1 >=0) {
    size_t oeIndex = (ob * param->ih * param->iw + (iy+1) * param->iw + (ix-1));
    size_t eU = oeIndex >> 4;
    size_t eR = oeIndex & 15;
    size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + 2) << 4) + eR) << 4) + (lR << 1);
    int* dst = (int*)(AP + dstOffset);
    *dst = src;

    // Corner case
    if(iy+1 == param->ih-1) {
        size_t index[3] = {6, 7, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix-1 ==0) {
        size_t index[3] = {0, 3, 6};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }  
}

// Pixel (iy+1, ix)
if(iy+1 < param->ih) {
    size_t oeIndex = (ob * param->ih * param->iw + (iy+1) * param->iw + (ix+0));
    size_t eU = oeIndex >> 4;
    size_t eR = oeIndex & 15;
    size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + 1) << 4) + eR) << 4) + (lR << 1);
    int* dst = (int*)(AP + dstOffset);
    *dst = src;

    // Corner case
    if(iy+1 == param->ih-1) {
        size_t index[3] = {6, 7, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix ==0) {
        size_t index[3] = {0, 3, 6};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix == param->iw-1) {
        size_t index[3] = {2, 5, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
}

//Pixel (iy+1, ix+1)
if(iy+1 < param->ih && ix+1 < param->iw) {
    size_t oeIndex = (ob * param->ih * param->iw + (iy+1) * param->iw + (ix+1));
    size_t eU = oeIndex >> 4;
    size_t eR = oeIndex & 15;
    size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + 0) << 4) + eR) << 4) + (lR << 1);
    int* dst = (int*)(AP + dstOffset);
    *dst = src;

    // Corner case
    if(iy+1 == param->ih-1) {
        size_t index[3] = {6, 7, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix+1 == param->iw-1) {
        size_t index[3] = {2, 5, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << 4) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
}
}
}

__global__ void Im2Col(const Im2ColParameter* param,
    const MatMulParam* matmulParam,
    const float* A,
    half* AP) {
    int eAlign = matmulParam->elhPack[0] * MATMULPACK;
    int lAlign = matmulParam->elhPack[1];
    int maxCount = eAlign * lAlign * BLOCK_INT4;
    int kernelCount = param->kernelX * param->kernelY;
    for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
        int index = indexO / BLOCK_INT4;
        int lR = indexO % BLOCK_INT4;
        int eIndex = index % eAlign;
        int lIndex = index / eAlign;
        int eU = eIndex / MATMULPACK;
        int eR = eIndex % MATMULPACK;
        int dstOffset = eU * matmulParam->elhPack[1] * (MATMULPACK * MATMULPACK) + lIndex * (MATMULPACK * MATMULPACK) + eR * MATMULPACK + lR * 8;
        int4* dst = (int4*)(AP + dstOffset);
        if (eIndex >= matmulParam->elh[0]) {
            *dst = {0, 0, 0, 0};
            continue;
        }
        // Compute for source
        int ox = eIndex % param->ow;
        int oy = eIndex / param->ow;
        int ob = oy / param->oh;
        oy = oy % param->oh;
        int sz = lIndex / kernelCount;
        int kI = lIndex % kernelCount;
        int ksx = kI % param->kernelX;
        int ksy = kI / param->kernelX;

        int sx = ox * param->strideX + ksx * param->dilateX - param->padX;
        int sy = oy * param->strideY + ksy * param->dilateY - param->padY;
        if (sx >= 0 && sx < param->iw) {
            if (sy >=0 && sy < param->ih) {
                int offset = sz * param->srcZStep + (ob * param->iw * param->ih + sy * param->iw + sx) * PACK_NUMBER + lR * 8;
                float2* srcF = (float2*)(A + offset);
                half2* dstH = (half2*)dst;
                dstH[0] = __float22half2_rn(srcF[0]);
                dstH[1] = __float22half2_rn(srcF[1]);
                dstH[2] = __float22half2_rn(srcF[2]);
                dstH[3] = __float22half2_rn(srcF[3]);
                continue;
            }
        }
        *dst = {0, 0, 0, 0};
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- Float22Half2 ----
void mnn_corpus_float22half2_fp32(const float* input, void* output, size_t maxCount,
                                  int grid, int block, cudaStream_t stream) {
    MNN::Corpus::Float22Half2<<<grid, block, 0, stream>>>(input, (half*)output, maxCount);
}

// ---- Float22BFloat16 (sm80+) ----
void mnn_corpus_float22bfloat16_fp32(const float* input, void* output, size_t maxCount,
                                     int grid, int block, cudaStream_t stream) {
    MNN::Corpus::Float22BFloat16<<<grid, block, 0, stream>>>(input, (__nv_bfloat16*)output, maxCount);
}

// ---- Im2Col_FilterC (float→float, scalar) ----
void mnn_corpus_im2col_filterc_fp32(
    const float* A, float* AP,
    int sw, int sh, int dw, int dh, int pw, int ph,
    int icDiv4, int iw, int ih, int ic, size_t maxCount, int pack,
    int e, int l, int l_p,
    int d_lp_val, int d_ow_val, int d_oh_val, int d_fx_val, int d_ic_val,
    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_lp(d_lp_val), d_ow(d_ow_val), d_oh(d_oh_val), d_fx(d_fx_val), d_ic(d_ic_val);
    MNN::Corpus::Im2Col_FilterC<float, float><<<grid, block, 0, stream>>>(
        sw, sh, dw, dh, pw, ph, icDiv4, iw, ih, ic, maxCount, pack, e, l, l_p, A, AP,
        d_lp, d_ow, d_oh, d_fx, d_ic);
}

// ---- WeightPackFill (float→float) ----
void mnn_corpus_weight_pack_fill_fp32(
    const float* param, float* output, int khw, size_t maxCount,
    int l, int h, int d_lp_val, int d_ic_val,
    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_lp(d_lp_val), d_ic(d_ic_val);
    MNN::Corpus::WeightPackFill<float, float><<<grid, block, 0, stream>>>(
        param, output, khw, maxCount, l, h, d_lp, d_ic);
}

// ---- PackPadFill (MatMul preprocessing, fp32) ----
void mnn_corpus_pack_pad_fill_fp32(
    const float* A, const float* B,
    bool transA, bool transB,
    float* tempA, float* tempB,
    int batchA, int batchB, int e, int l, int h,
    int ep, int lp, int hp,
    int d_e_val, int d_l_val, int d_h_val, int d_lp_val, int d_lp2_val,
    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_e(d_e_val), d_l(d_l_val), d_h(d_h_val), d_lp(d_lp_val), d_lp2(d_lp2_val);
    MNN::Corpus::PackPadFill<float, float><<<grid, block, 0, stream>>>(
        A, B, transA, transB, tempA, tempB, batchA, batchB, e, l, h, ep, lp, hp,
        d_e, d_l, d_h, d_lp, d_lp2);
}

// ---- WeightPackFill_Implicit (implicit conv weight pack, fp32) ----
void mnn_corpus_weight_pack_fill_implicit_fp32(
    const float* param, float* output, int khw, size_t maxCount,
    int ci, int co, int d_cip_val, int d_khw_val,
    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_cip(d_cip_val), d_khw(d_khw_val);
    MNN::Corpus::WeightPackFill_Implicit<float><<<grid, block, 0, stream>>>(
        param, output, khw, maxCount, ci, co, d_cip, d_khw);
}

// ---- Im2Col_FilterC_Vec4 (float→float4, precision=1 for fp32 corpus) ----
void mnn_corpus_im2col_filterc_vec4_fp32(
    const float* A, float* AP,
    int sw, int sh, int dw, int dh, int pw, int ph,
    int icDiv4, int iw, int ih, int ic, size_t maxCount, int pack,
    int e, int l, int l_p, int precision,
    int d_lp_val, int d_ow_val, int d_oh_val, int d_fx_val, int d_ic4_val,
    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_lp(d_lp_val), d_ow(d_ow_val), d_oh(d_oh_val), d_fx(d_fx_val), d_ic4(d_ic4_val);
    MNN::Corpus::Im2Col_FilterC_Vec4<float, float><<<grid, block, 0, stream>>>(
        sw, sh, dw, dh, pw, ph, icDiv4, iw, ih, ic, maxCount, pack, e, l, l_p, A, AP, precision,
        d_lp, d_ow, d_oh, d_fx, d_ic4);
}

} // extern "C"
