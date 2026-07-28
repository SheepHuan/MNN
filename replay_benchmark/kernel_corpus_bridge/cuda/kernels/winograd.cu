// winograd.cu - Winograd convolution kernels for replay benchmark corpus
//   source/backend/cuda/execution/ConvWinogradExecution.cu  (WinoWeightReorder)
//   source/backend/cuda/execution/WinogradTrans.cuh          (WinoInputTrans / WinoTrans2Output + half2 variants)
//   source/backend/cuda/execution/ConvBaseKernel.cu          (Im2Col_FilterC_Vec4 + macros)
//
// fp32 shims provided:
//   mnn_corpus_wino_weight_reorder_fp32   -> WinoWeightReorder<float>
//   mnn_corpus_wino_input_trans_fp32       -> WinoInputTrans<float,float>
//   mnn_corpus_wino_trans2output_fp32      -> WinoTrans2Output<float>
// half2 / int4 variants deferred to CudaOpsFp16.cu.
#include "corpus_common.cuh"
#include <cuda_fp16.h>

namespace MNN {
namespace Corpus {

#define UNIT 2
static const int PACK_NUMBER = 8;

// ============================================================================
// WinoWeightReorder: reorder Winograd-transformed weights
//   [4x4, Cop, Cip, unitCi, unitCo] -> [4x4, Cop*unitCo, Cip*unitCi]
// From ConvWinogradExecution.cu lines 16-36.
// ============================================================================
template<typename T>
__global__ void WinoWeightReorder(const float* GgGt,
    T* GgGt_trans,
    const int block,
    const int co_pack,
    const int ci_pack,
    const int unitCi,
    const int unitCo
    ) {
    const int maxCount = block * co_pack * ci_pack;
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += gridDim.x * blockDim.x) {
        size_t tmp =  index / ci_pack;
        size_t ci_idx =  index % ci_pack;

        size_t block_idx =  tmp / co_pack;
        size_t co_idx = tmp % co_pack;
        // [4x4, Cop, Cip, unitCi, unitCo] -->> [4x4, Cop*unitCo, Cip*unitCi]
        size_t src_idx = block_idx * (co_pack*ci_pack) + (co_idx/unitCo) * (ci_pack*unitCo) + (ci_idx/unitCi) * (unitCi*unitCo) + (ci_idx%unitCi) * unitCo + (co_idx%unitCo);
        *(GgGt_trans + index) = *(GgGt + src_idx);
    }
}

// ============================================================================
// WinoInputTrans: Winograd input transform (BᵀdB), fp32 template version
// From WinogradTrans.cuh lines 7-178.
// ============================================================================
template<typename T0, typename T1>
__global__ void WinoInputTrans(const T0* input,
    T1* BtdB,
    const int unit,
    const int block,
    const int ci,
    const int ci_p8,
    const int maxCount,
    DivModFast lD,
    DivModFast whD,
    DivModFast wD,
    const int pad_x,
    const int pad_y,
    const int width,
    const int height
) {
    const int l = ci_p8;
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += gridDim.x * blockDim.x) {
        int e_idx, ci_idx, batch_idx, tmp, w_idx, h_idx;
        lD.divmod(index, e_idx, ci_idx);
        whD.divmod(e_idx, batch_idx, tmp);
        wD.divmod(tmp, h_idx, w_idx);

        const int sxStart = w_idx * unit - pad_x;
        const int syStart = h_idx * unit - pad_y;

        float S00, S10, S20, S30, S01, S11, S21, S31, S02, S12, S22, S32, S03, S13, S23, S33;
        
        int inp_offset = ((batch_idx * height + syStart) * width + sxStart) * ci_p8 + ci_idx;
        {
            int sx      = 0 + sxStart;
            int sy      = 0 + syStart;
            
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S00         = outBound ? 0.0f : (float)input[inp_offset];
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 0 + syStart;
            
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S10         = outBound ? 0.0f : (float)input[inp_offset+ci_p8];
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 0 + syStart;
            
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S20         = outBound ? 0.0f : (float)input[inp_offset+ci_p8+ci_p8];
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 0 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S30         = outBound ? 0.0f : (float)input[inp_offset+ci_p8+ci_p8+ci_p8];
        }
        {
            int sx      = 0 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S01         = outBound ? 0.0f : (float)input[inp_offset+width*ci_p8];
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S11         = outBound ? 0.0f : (float)input[inp_offset+(width+1)*ci_p8];
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S21         = outBound ? 0.0f : (float)input[inp_offset+(width+2)*ci_p8];
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S31         = outBound ? 0.0f : (float)input[inp_offset+(width+3)*ci_p8];
        }
        {
            int sx      = 0 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S02         = outBound ? 0.0f : (float)input[inp_offset+(width+width+0)*ci_p8];
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S12         = outBound ? 0.0f : (float)input[inp_offset+(width+width+1)*ci_p8];
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S22         = outBound ? 0.0f : (float)input[inp_offset+(width+width+2)*ci_p8];
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S32         = outBound ? 0.0f : (float)input[inp_offset+(width+width+3)*ci_p8];
        }
        {
            int sx      = 0 + sxStart;
            int sy      = 3 + syStart;
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S03         = outBound ? 0.0f : (float)input[inp_offset+(width+width+width+0)*ci_p8];
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 3 + syStart;
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S13         = outBound ? 0.0f : (float)input[inp_offset+(width+width+width+1)*ci_p8];
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 3 + syStart;
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S23         = outBound ? 0.0f : (float)input[inp_offset+(width+width+width+2)*ci_p8];
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 3 + syStart;
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S33         = outBound ? 0.0f : (float)input[inp_offset+(width+width+width+3)*ci_p8];
        }
        float m00 = +S00 - S02;
        float m10 = +S10 - S12;
        float m20 = +S20 - S22;
        float m30 = +S30 - S32;
        float m01 = +0.5f * (S01 + S02);
        float m11 = +0.5f * (S11 + S12);
        float m21 = +0.5f * (S21 + S22);
        float m31 = +0.5f * (S31 + S32);
        float m02 = +0.5f * (-S01 + S02);
        float m12 = +0.5f * (-S11 + S12);
        float m22 = +0.5f * (-S21 + S22);
        float m32 = +0.5f * (-S31 + S32);
        float m03 = -S01 + S03;
        float m13 = -S11 + S13;
        float m23 = -S21 + S23;
        float m33 = -S31 + S33;

        BtdB[0*maxCount + index]  = (T1)(+m00 - m20);
        BtdB[1*maxCount + index]  = (T1)(+0.5f * (m10 + m20));
        BtdB[2*maxCount + index]  = (T1)(+0.5f * (-m10 + m20));
        BtdB[3*maxCount + index]  = (T1)(-m10 + m30);
        BtdB[4*maxCount + index]  = (T1)(+m01 - m21);
        BtdB[5*maxCount + index]  = (T1)(+0.5f * (m11 + m21));
        BtdB[6*maxCount + index]  = (T1)(+0.5f * (-m11 + m21));
        BtdB[7*maxCount + index]  = (T1)(-m11 + m31);
        BtdB[8*maxCount + index]  = (T1)(+m02 - m22);
        BtdB[9*maxCount + index]  = (T1)(+0.5f * (m12 + m22));
        BtdB[10*maxCount + index] = (T1)(+0.5f * (-m12 + m22));
        BtdB[11*maxCount + index] = (T1)(-m12 + m32);
        BtdB[12*maxCount + index] = (T1)(+m03 - m23);
        BtdB[13*maxCount + index] = (T1)(+0.5f * (m13 + m23));
        BtdB[14*maxCount + index] = (T1)(+0.5f * (-m13 + m23));
        BtdB[15*maxCount + index] = (T1)(-m13 + m33);
    }
}

// ============================================================================
// WinoInputTrans_half2: Winograd input transform using half2 intrinsics
// From WinogradTrans.cuh lines 180-358.
// ============================================================================
__global__ void WinoInputTrans_half2(const half2* input,
    half2* BtdB,
    const int unit,
    const int block,
    const int ci,
    const int ci_p8,
    const int maxCount,
    DivModFast lD,
    DivModFast whD,
    DivModFast wD,
    const int pad_x,
    const int pad_y,
    const int width,
    const int height
) {
    const int l = ci_p8;
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += gridDim.x * blockDim.x) {
        int e_idx, ci_idx, batch_idx, tmp, w_idx, h_idx;
        lD.divmod(index, e_idx, ci_idx);
        whD.divmod(e_idx, batch_idx, tmp);
        wD.divmod(tmp, h_idx, w_idx);

        const int sxStart = w_idx * unit - pad_x;
        const int syStart = h_idx * unit - pad_y;

        half2 S00, S10, S20, S30, S01, S11, S21, S31, S02, S12, S22, S32, S03, S13, S23, S33;
        half2 zero;
        zero.x = 0.0f;
        zero.y = 0.0f;
        const int ci_div2 = (ci+1) >> 1;
        int inp_offset = ((batch_idx * height + syStart) * width + sxStart) * ci_p8 + ci_idx;
        {
            int sx      = 0 + sxStart;
            int sy      = 0 + syStart;
            
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci_div2);
            S00         = outBound ? zero : input[inp_offset];
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 0 + syStart;
            
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci_div2);
            S10         = outBound ? zero : input[inp_offset+ci_p8];
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 0 + syStart;
            
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci_div2);
            S20         = outBound ? zero : input[inp_offset+ci_p8+ci_p8];
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 0 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci_div2);
            S30         = outBound ? zero : input[inp_offset+ci_p8+ci_p8+ci_p8];
        }
        {
            int sx      = 0 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci_div2);
            S01         = outBound ? zero : input[inp_offset+width*ci_p8];
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci_div2);
            S11         = outBound ? zero : input[inp_offset+(width+1)*ci_p8];
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci_div2);
            S21         = outBound ? zero : input[inp_offset+(width+2)*ci_p8];
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci_div2);
            S31         = outBound ? zero : input[inp_offset+(width+3)*ci_p8];
        }
        {
            int sx      = 0 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci_div2);
            S02         = outBound ? zero : input[inp_offset+(width+width+0)*ci_p8];
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci_div2);
            S12         = outBound ? zero : input[inp_offset+(width+width+1)*ci_p8];
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci_div2);
            S22         = outBound ? zero : input[inp_offset+(width+width+2)*ci_p8];
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci_div2);
            S32         = outBound ? zero : input[inp_offset+(width+width+3)*ci_p8];
        }
        {
            int sx      = 0 + sxStart;
            int sy      = 3 + syStart;
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci_div2);
            S03         = outBound ? zero : input[inp_offset+(width+width+width+0)*ci_p8];
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 3 + syStart;
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci_div2);
            S13         = outBound ? zero : input[inp_offset+(width+width+width+1)*ci_p8];
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 3 + syStart;
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci_div2);
            S23         = outBound ? zero : input[inp_offset+(width+width+width+2)*ci_p8];
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 3 + syStart;
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci_div2);
            S33         = outBound ? zero : input[inp_offset+(width+width+width+3)*ci_p8];
        }
        half2 m00 = __hsub2(S00, S02);
        half2 m10 = __hsub2(S10, S12);
        half2 m20 = __hsub2(S20, S22);
        half2 m30 = __hsub2(S30, S32);

        half2 const_0_5;
        const_0_5.x = 0.5f;
        const_0_5.y = 0.5f;
        half2 m01 = __hmul2(const_0_5, __hadd2(S01, S02));
        half2 m11 = __hmul2(const_0_5, __hadd2(S11, S12));
        half2 m21 = __hmul2(const_0_5, __hadd2(S21, S22));
        half2 m31 = __hmul2(const_0_5, __hadd2(S31, S32));
        half2 m02 = __hmul2(const_0_5, __hsub2(S02, S01));
        half2 m12 = __hmul2(const_0_5, __hsub2(S12, S11));
        half2 m22 = __hmul2(const_0_5, __hsub2(S22, S21));
        half2 m32 = __hmul2(const_0_5, __hsub2(S32, S31));

        half2 m03 = __hsub2(S03, S01);
        half2 m13 = __hsub2(S13, S11);
        half2 m23 = __hsub2(S23, S21);
        half2 m33 = __hsub2(S33, S31);

        BtdB[0*maxCount + index]  = __hsub2(m00, m20);
        BtdB[1*maxCount + index]  = __hmul2(const_0_5, __hadd2(m10, m20));
        BtdB[2*maxCount + index]  = __hmul2(const_0_5, __hsub2(m20, m10));
        BtdB[3*maxCount + index]  = __hsub2(m30, m10);
        BtdB[4*maxCount + index]  = __hsub2(m01, m21);
        BtdB[5*maxCount + index]  = __hmul2(const_0_5, __hadd2(m11, m21));
        BtdB[6*maxCount + index]  = __hmul2(const_0_5, __hsub2(m21, m11));
        BtdB[7*maxCount + index]  = __hsub2(m31, m11);
        BtdB[8*maxCount + index]  = __hsub2(m02, m22);
        BtdB[9*maxCount + index]  = __hmul2(const_0_5, __hadd2(m12, m22));
        BtdB[10*maxCount + index] = __hmul2(const_0_5, __hsub2(m22, m12));
        BtdB[11*maxCount + index] = __hsub2(m32, m12);
        BtdB[12*maxCount + index] = __hsub2(m03, m23);
        BtdB[13*maxCount + index] = __hmul2(const_0_5, __hadd2(m13, m23));
        BtdB[14*maxCount + index] = __hmul2(const_0_5, __hsub2(m23, m13));
        BtdB[15*maxCount + index] = __hsub2(m33, m13);
    }
}

// ============================================================================
// WinoTrans2Output: Winograd output transform (AᵀM), fp32 template version
// From WinogradTrans.cuh lines 360-467.
// ============================================================================
template<typename T>
__global__ void WinoTrans2Output(const T* matmulData,
    const float* biasData,
    T* output,
    const int unit,
    const int block,
    const int co,
    const int co_p8,
    const int maxCount,
    DivModFast hD,
    DivModFast whD,
    DivModFast wD,
    const int width,
    const int height,
    const int activationType
) {
    const int h = co_p8;
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += gridDim.x * blockDim.x) {

        int e_idx, co_idx, batch_idx, tmp, w_idx, h_idx;
        hD.divmod(index, e_idx, co_idx);
        whD.divmod(e_idx, batch_idx, tmp);
        wD.divmod(tmp, h_idx, w_idx);

        float S00 = matmulData[index];
        float S10 = matmulData[index + maxCount];
        float S20 = matmulData[index + maxCount * 2];
        float S30 = matmulData[index + maxCount * 3];
        float S01 = matmulData[index + maxCount * 4];
        float S11 = matmulData[index + maxCount * 5];
        float S21 = matmulData[index + maxCount * 6];
        float S31 = matmulData[index + maxCount * 7];
        float S02 = matmulData[index + maxCount * 8];
        float S12 = matmulData[index + maxCount * 9];
        float S22 = matmulData[index + maxCount * 10];
        float S32 = matmulData[index + maxCount * 11];
        float S03 = matmulData[index + maxCount * 12];
        float S13 = matmulData[index + maxCount * 13];
        float S23 = matmulData[index + maxCount * 14];
        float S33 = matmulData[index + maxCount * 15];

        float m00 = +S00 + S01 + S02;
        float m10 = +S10 + S11 + S12;
        float m20 = +S20 + S21 + S22;
        float m30 = +S30 + S31 + S32;
        float m01 = +S01 - S02 + S03;
        float m11 = +S11 - S12 + S13;
        float m21 = +S21 - S22 + S23;
        float m31 = +S31 - S32 + S33;

        // write output
        float bias = biasData[co_idx];

        const int dxStart = w_idx * unit;
        const int dyStart = h_idx * unit;

        if(co_idx >= co_p8) {
            continue;
        }
        int out_offset = ((batch_idx * height + dyStart) * width + dxStart) * co_p8 + co_idx;

        /* if true */ {
            float res = bias + (float)(m00 + m10 + m20);
            if(activationType == 1) {
                res = max(res, 0.0f);
            }
            if(activationType == 2) {
                res = max(res, 0.0f);
                res = min(res, 6.0f);
            }
            output[out_offset] = (T)res;
        }
        if (dxStart + 1 < width) {
            float res = bias + (float)(m10 - m20 + m30);
            if(activationType == 1) {
                res = max(res, 0.0f);
            }
            if(activationType == 2) {
                res = max(res, 0.0f);
                res = min(res, 6.0f);
            }
            output[out_offset + co_p8] = (T)res;
        }
        if (dyStart + 1 < height) {
            float res = bias + (float)(m01 + m11 + m21);
            if(activationType == 1) {
                res = max(res, 0.0f);
            }
            if(activationType == 2) {
                res = max(res, 0.0f);
                res = min(res, 6.0f);
            }
            output[out_offset + width * co_p8] = (T)res;
        }
        if (dxStart + 1 < width && dyStart + 1 < height) {
            float res = bias + (float)(m11 - m21 + m31);
            if(activationType == 1) {
                res = max(res, 0.0f);
            }
            if(activationType == 2) {
                res = max(res, 0.0f);
                res = min(res, 6.0f);
            }
            output[out_offset + (width + 1) * co_p8] = (T)res;
        }

    }
}

// ============================================================================
// WinoTrans2Output_half2: Winograd output transform using half2 intrinsics
// From WinogradTrans.cuh lines 469-591.
// ============================================================================
__global__ void WinoTrans2Output_half2(const half2* matmulData,
    const float* biasData,
    half2* output,
    const int unit,
    const int block,
    const int co,
    const int co_p8,
    const int maxCount,
    DivModFast hD,
    DivModFast whD,
    DivModFast wD,
    const int width,
    const int height,
    const int activationType
) {
    const int h = co_p8;
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += gridDim.x * blockDim.x) {

        int e_idx, co_idx, batch_idx, tmp, w_idx, h_idx;
        hD.divmod(index, e_idx, co_idx);
        whD.divmod(e_idx, batch_idx, tmp);
        wD.divmod(tmp, h_idx, w_idx);

        auto S00 = matmulData[index];
        auto S10 = matmulData[index + maxCount];
        auto S20 = matmulData[index + maxCount * 2];
        auto S30 = matmulData[index + maxCount * 3];
        auto S01 = matmulData[index + maxCount * 4];
        auto S11 = matmulData[index + maxCount * 5];
        auto S21 = matmulData[index + maxCount * 6];
        auto S31 = matmulData[index + maxCount * 7];
        auto S02 = matmulData[index + maxCount * 8];
        auto S12 = matmulData[index + maxCount * 9];
        auto S22 = matmulData[index + maxCount * 10];
        auto S32 = matmulData[index + maxCount * 11];
        auto S03 = matmulData[index + maxCount * 12];
        auto S13 = matmulData[index + maxCount * 13];
        auto S23 = matmulData[index + maxCount * 14];
        auto S33 = matmulData[index + maxCount * 15];

        auto m00 = __hadd2(S00, __hadd2(S01, S02));
        auto m10 = __hadd2(S10, __hadd2(S11, S12));
        auto m20 = __hadd2(S20, __hadd2(S21, S22));
        auto m30 = __hadd2(S30, __hadd2(S31, S32));
        auto m01 = __hadd2(S03, __hsub2(S01, S02));
        auto m11 = __hadd2(S13, __hsub2(S11, S12));
        auto m21 = __hadd2(S23, __hsub2(S21, S22));
        auto m31 = __hadd2(S33, __hsub2(S31, S32));

        // write output
        half2 bias;
        bias.x = (half)biasData[2*co_idx];
        bias.y = (half)biasData[2*co_idx+1];

        const int dxStart = w_idx * unit;
        const int dyStart = h_idx * unit;

        if(co_idx >= co_p8) {
            continue;
        }
        int out_offset = ((batch_idx * height + dyStart) * width + dxStart) * co_p8 + co_idx;

        /* if true */ {
            half2 res = __hadd2(bias, __hadd2(__hadd2(m00, m10), m20));
            if(activationType == 1) {
                res.x = max(res.x, 0.0f);
                res.y = max(res.y, 0.0f);
            }
            if(activationType == 2) {
                res.x = max(res.x, 0.0f);
                res.y = max(res.y, 0.0f);
                res.x = min(res.x, 6.0f);
                res.y = min(res.y, 6.0f);
            }
            output[out_offset] = res;
        }
        if (dxStart + 1 < width) {
            half2 res = __hadd2(bias, __hadd2(__hsub2(m10, m20), m30));
            // float res = bias + (float)(m10 - m20 + m30);
            if(activationType == 1) {
                res.x = max(res.x, 0.0f);
                res.y = max(res.y, 0.0f);
            }
            if(activationType == 2) {
                res.x = max(res.x, 0.0f);
                res.y = max(res.y, 0.0f);
                res.x = min(res.x, 6.0f);
                res.y = min(res.y, 6.0f);
            }
            output[out_offset + co_p8] = res;
        }
        if (dyStart + 1 < height) {
            half2 res = __hadd2(bias, __hadd2(__hadd2(m01, m11), m21));
            // float res = bias + (float)(m01 + m11 + m21);
            if(activationType == 1) {
                res.x = max(res.x, 0.0f);
                res.y = max(res.y, 0.0f);
            }
            if(activationType == 2) {
                res.x = max(res.x, 0.0f);
                res.y = max(res.y, 0.0f);
                res.x = min(res.x, 6.0f);
                res.y = min(res.y, 6.0f);
            }
            output[out_offset + width * co_p8] = res;
        }
        if (dxStart + 1 < width && dyStart + 1 < height) {
            half2 res = __hadd2(bias, __hadd2(__hsub2(m11, m21), m31));
            // float res = bias + (float)(m11 - m21 + m31);
            if(activationType == 1) {
                res.x = max(res.x, 0.0f);
                res.y = max(res.y, 0.0f);
            }
            if(activationType == 2) {
                res.x = max(res.x, 0.0f);
                res.y = max(res.y, 0.0f);
                res.x = min(res.x, 6.0f);
                res.y = min(res.y, 6.0f);
            }
            output[out_offset + (width + 1) * co_p8] = res;
        }

    }
}

// (Im2Col_FilterC_Vec4 moved to conv_base.cu to avoid duplicate definition)

// NOTE: Im2Col_FilterC_Vec4 is defined in conv_base.cu (not here) to avoid
// duplicate symbol. The shim mnn_corpus_im2col_filterc_vec4_fp32 also lives
// in conv_base.cu.

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- WinoWeightReorder (float -> float) ----
void mnn_corpus_wino_weight_reorder_fp32(
    const float* GgGt, float* GgGt_trans,
    int block, int co_pack, int ci_pack, int unitCi, int unitCo,
    int grid, int block_, cudaStream_t stream) {
    MNN::Corpus::WinoWeightReorder<float><<<grid, block_, 0, stream>>>(
        GgGt, GgGt_trans, block, co_pack, ci_pack, unitCi, unitCo);
}

// ---- WinoInputTrans (float, float) ----
void mnn_corpus_wino_input_trans_fp32(
    const float* input, float* BtdB,
    int unit, int block, int ci, int ci_p8, int maxCount,
    int lD, int whD, int wD,
    int pad_x, int pad_y, int width, int height,
    int grid, int block_, cudaStream_t stream) {
    MNN::Corpus::DivModFast lD_(lD), whD_(whD), wD_(wD);
    MNN::Corpus::WinoInputTrans<float, float><<<grid, block_, 0, stream>>>(
        input, BtdB, unit, block, ci, ci_p8, maxCount, lD_, whD_, wD_,
        pad_x, pad_y, width, height);
}

// ---- WinoTrans2Output (float, bias float, output float) ----
void mnn_corpus_wino_trans2output_fp32(
    const float* matmulData, const float* biasData, float* output,
    int unit, int block, int co, int co_p8, int maxCount,
    int hD, int whD, int wD,
    int width, int height, int activationType,
    int grid, int block_, cudaStream_t stream) {
    MNN::Corpus::DivModFast hD_(hD), whD_(whD), wD_(wD);
    MNN::Corpus::WinoTrans2Output<float><<<grid, block_, 0, stream>>>(
        matmulData, biasData, output, unit, block, co, co_p8, maxCount,
        hD_, whD_, wD_, width, height, activationType);
}

// ---- WinoInputTrans (half2, half2) — defined but not launched in MNN ----
void mnn_corpus_wino_input_trans_half2_fp32(
    const void* input, void* BtdB,
    int unit, int block, int ci, int ci_p8, int maxCount,
    int lD, int whD, int wD,
    int pad_x, int pad_y, int width, int height,
    int grid, int block_, cudaStream_t stream) {
    MNN::Corpus::DivModFast lD_(lD), whD_(whD), wD_(wD);
    MNN::Corpus::WinoInputTrans_half2<<<grid, block_, 0, stream>>>(
        (const half2*)input, (half2*)BtdB, unit, block, ci, ci_p8, maxCount, lD_, whD_, wD_,
        pad_x, pad_y, width, height);
}

// ---- WinoTrans2Output (half2, bias float, output half2) ----
void mnn_corpus_wino_trans2output_half2_fp32(
    const void* matmulData, const float* biasData, void* output,
    int unit, int block, int co, int co_p8, int maxCount,
    int hD, int whD, int wD,
    int width, int height, int activationType,
    int grid, int block_, cudaStream_t stream) {
    MNN::Corpus::DivModFast hD_(hD), whD_(whD), wD_(wD);
    MNN::Corpus::WinoTrans2Output_half2<<<grid, block_, 0, stream>>>(
        (const half2*)matmulData, biasData, (half2*)output, unit, block, co, co_p8, maxCount,
        hD_, whD_, wD_, width, height, activationType);
}

} // extern "C"
