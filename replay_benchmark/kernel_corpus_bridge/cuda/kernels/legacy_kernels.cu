// legacy_kernels.cu - Historical kernels from deleted MNN versions (1.2.0-2.8.4)
//   These __global__ kernels existed in intermediate MNN releases but were removed
//   in later versions. Faithfully copied for cross-version replay coverage.
#include "corpus_common.cuh"
#include <cuda_fp16.h>
#include <mma.h>
#include <float.h>

using namespace nvcuda;
using half = __half;

namespace MNN {
namespace Corpus {

struct ConstBuffer_120 {
    int pad[2]; int kernelSize[2]; int stride[2]; int dilate[2];
    int inputSize[2]; int outputSize[2];
    int channel; int subChannel; int total; int activationType;
};
struct InputReorderParameter {
    int ic_stride; int ib_stride; int oc_stride; int ob_stride;
    int hw_size; int l_size; int h_size; int lpack_size; int hpack_size;
};
struct ReduceParam { int inside; int axis; int outside; };
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

#define HALF2_MIN half2(-65504, -65504)
#define MNN_CUDA_HALF2_MAX(a, b) do { (a).x = __hgt((a).x, (b).x) ? (a).x : (b).x; (a).y = __hgt((a).y, (b).y) ? (a).y : (b).y; } while (0)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4
#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2
#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)
#define CHUNK_L 4
#define CHUNK_E 4
#define CHUNK_H 4
#define BLOCK_INT4 2

// 1.2.7 used PACK_NUMBER=16 for pool/TensorCore (corpus_common.cuh defines 8)
#undef PACK_NUMBER
#define PACK_NUMBER 16
#define PACK_NUMBER_C2 (PACK_NUMBER/2)
#define MATMULPACK 16
#define MATMULPACK2 (MATMULPACK * MATMULPACK)

template <typename T>
__global__ void cutPad(const size_t size, const T* input, const int old_height,
                    const int old_width, const int height, const int width, const int pad_top,
                    const int pad_left, T* output) {
    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
        int block_num = pos / (width*height);
        int left = pos % (width*height);
        const int out_w = left % width;
        const int out_h = left / width % height;

        output[pos] = input[(block_num * old_height + out_h + pad_top) * old_width + out_w + pad_left];
    }
    return;
}

__global__ void DECONV_DW(const float* input, const float* kernel, const float* bias, float *output, const ConstBuffer_120* uConstant) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < uConstant->total; i += blockDim.x * gridDim.x) {
        {
            int iw = uConstant->inputSize[0];
            int ih = uConstant->inputSize[1];
            int c = uConstant->channel;
            int ow = uConstant->outputSize[0];
            int oh = uConstant->outputSize[1];
            int kw = uConstant->kernelSize[0];
            int kh = uConstant->kernelSize[1];
            int dw = uConstant->dilate[0];
            int dh = uConstant->dilate[1];
            int sw = uConstant->stride[0];
            int sh = uConstant->stride[1];
            int pw = uConstant->pad[0];
            int ph = uConstant->pad[1];

            int oz = i / (ow * oh);
            int tmp = i % (ow * oh);
            int oy = tmp / ow;
            int ox = tmp % ow;
            int kz = oz % uConstant->subChannel;
            
            int ix = ox + pw;
            int iy = oy + ph;
            float color = 0.0;
            if (bias != nullptr) {
                color = bias[kz];
            }

            int fx, fy, fz;
            for (fy=0; fy<kh; ++fy) {
                int sy = iy - fy*dh;
                int y = sy / sh;
                if (sy % sh == 0 && y >= 0 && y < ih) {
                    for (int fx=0; fx<kw; ++fx) {
                        int sx = ix - fx*dw;
                        int x = sx / sw;
                        if (sx % sw == 0 && x >= 0 && x < iw) {
                            float inputValue = input[0
                                + x
                                + y * iw
                                + oz * iw * ih
                            ];
                            float k = kernel[0
                                + fx
                                + fy * kw
                                + kz * kw * kh
                            ];
                            color  += k*inputValue;                            
                        }
                    }
                }
            }
            output[0
                + ox
                + oy * ow
                + oz * ow * oh
            ] = color;
        }
    }
    return;
}

template<typename T>
__global__ void SCATTERND(const int n, const int indicesLastDim, const int accNumber, const int* indicesPtr,
    const T* updatesPtr, T* outputPtr, const int32_t* dimsToCount) {
    CUDA_KERNEL_LOOP(index, n) {
        int pos = 0;
        for (int j = 0; j < indicesLastDim; ++j) {
            auto curIndex = (int)indicesPtr[index * indicesLastDim + j];
            // MNN_ASSERT(curIndex >= 0 && curIndex < output->length(j));
            pos += curIndex * dimsToCount[j];
        }
        for (int k = 0; k < accNumber; ++k) {
            float updateValue = updatesPtr[index * accNumber + k];
            atomicAdd(outputPtr + pos + k, updateValue);
        }
    }
}

__global__ void DeconvInputRerange(const int count,
        const InputReorderParameter* param,
        const float* Inp,
        __half* InpRe
        ) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        int l = param->l_size;
        int h = param->h_size;
        int lIndex = i % l;
        int hIndex = i / l;
        int lU = lIndex / 16;
        int lR = lIndex % 16;
        int hU = hIndex / 16;
        int hR = hIndex % 16;

        int bIndex = hIndex / param->hw_size;
        int hwIndex = hIndex % param->hw_size;

        float value = Inp[bIndex * param->ib_stride + lIndex * param->ic_stride + hwIndex];
        //inpRe[lIndex * param->oc_stride + bIndex * param->ob_stride + hwIndex] = value;

        //__half* dst = InpRe + lU * param->hpack_size * 16 * 16 + hU * 16 * 16 + hR + lR * 16;
        __half* dst = InpRe + hU * param->lpack_size * 16 * 16 + lU * 16 * 16 + lR + hR * 16;
        dst[0] = value;
    }
}

template <typename T>
__global__ void DIVSUM(const T *input, const T* maxV, T *output, const ReduceParam* param) {
    int inside = param->inside;
    int axis = param->axis;
    int outside = param->outside;
    int count = inside * axis * outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int tmp = i / inside;
        int x = i % inside;
        int y = tmp / axis;
        int c = tmp % axis;
        float sumValue = 0.0;
        const float basicInput = input[i];
        const float value = maxV[x + y * inside];
        output[i] = (T)(basicInput / value);
    }
    return;
}

template <typename T>
__global__ void EXPSUB(const T *input, const T* maxV, T *output, const ReduceParam* param) {
    int inside = param->inside;
    int axis = param->axis;
    int outside = param->outside;
    int count = inside * axis * outside;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        int tmp = i / inside;
        int x = i % inside;
        int y = tmp / axis;
        int c = tmp % axis;
        float sumValue = 0.0;
        const float basicInput = input[i];
        const float maxValue = maxV[x + y * inside];
        output[i] = (T)(exp(basicInput - maxValue));
    }
    return;
}

template <typename T>
__global__ void SPLIT_FusedKV(const size_t count, const T* fused_kv,
        T* ptr_k, T* ptr_v,
        int head_size
    ) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        //[B, S, H, 2, D] -> [B, S, H, D]
        const int bsh = i / head_size;
        const int d = i % head_size;

        ptr_k[i] =  fused_kv[(bsh * 2 + 0) * head_size + d];
        ptr_v[i] =  fused_kv[(bsh * 2 + 1) * head_size + d];
    }
}

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

template<typename T>
__global__ void GemmPrearrange(MatMulParam paramV,
        const T* OA,
        __half* OAP,
        const T* OB,
        __half* OBP,
        DivModFast lA
        ) {
    int b = blockIdx.x;
    auto param = &paramV;
    int lAlign = param->elhPack[1] * 16;
    int eAlign = param->elhPack[0] * 16;
    int hAlign = param->elhPack[2] * 16;
    __half* BP = OBP + b * param->elhPack[1] * param->elhPack[2] * 16 * 16;
    __half* AP = OAP + b * param->elhPack[1] * param->elhPack[0] * 16 * 16;
    const T* A = OA + b * param->elh[0] * param->elh[1];
    const T* B = OB + b * param->elh[2] * param->elh[1];
    int mc = param->elhPack[0] * param->elhPack[1] * 256;
    int e = param->elh[0];
    int l = param->elh[1];
    int h = param->elh[2];
    for (size_t index = threadIdx.x; index < mc && OA != nullptr; index += blockDim.x) {
        int lIndex, oIndex;
        lA.divmod(index, oIndex, lIndex);

        half value = 0.0;
        if (oIndex < e && lIndex < l) {
            value = A[oIndex * param->aStride[0] + lIndex * param->aStride[1]];
        }
        AP[index] = value;
    }
    mc = param->elhPack[2] * param->elhPack[1] * 256;
    for (size_t index = threadIdx.x; index < mc && OB != nullptr; index += blockDim.x) {
        int lIndex, oIndex;
        lA.divmod(index, oIndex, lIndex);
        half value = 0.0;
        if (oIndex < h && lIndex < l) {
            value = B[oIndex * param->bStride[2] + lIndex * param->bStride[1]];
        }
        BP[index] = value;
    }
}

template<typename T>
__global__ void GemmPrearrange_OPT(MatMulParam paramV, const int maxCount,
        const int AreaPackA, const int AreaPackB, const int AreaA, const int AreaB,
        const T* OA,
        __half* OAP,
        const T* OB,
        __half* OBP,
        DivModFast lA,
        DivModFast pM
        ) {
    int index, b;
    size_t indexT = blockIdx.x*blockDim.x+threadIdx.x;
    pM.divmod(indexT, b, index);
    int indexCopy = index;
    
    auto param = &paramV;
    int e = param->elh[0];
    int l = param->elh[1];
    int h = param->elh[2];
    for (; index < AreaPackA && OA != nullptr; index += blockDim.x*gridDim.x) {
        int lIndex, oIndex;
        lA.divmod(index, oIndex, lIndex);

        __half* AP = OAP + b * AreaPackA;
        const T* A = OA + b * AreaA;
        half value = 0.0;
        if (oIndex < e && lIndex < l) {
            value = A[oIndex * param->aStride[0] + lIndex * param->aStride[1]];
        }
        AP[index] = value;
    }

    index = indexCopy;
    for (; index < AreaPackB && OB != nullptr; index += blockDim.x*gridDim.x) {
        int lIndex, oIndex;
        lA.divmod(index, oIndex, lIndex);
        
        __half* BP = OBP + b * AreaPackB;
        const T* B = OB + b * AreaB;
        half value = 0.0;
        if (oIndex < h && lIndex < l) {
            value = B[oIndex * param->bStride[2] + lIndex * param->bStride[1]];
        }
        BP[index] = value;
    }
}

template<typename T, typename LayoutA, typename LayoutB>
__global__ void GemmPacked(const MatMulParam* param, T *bc, const half *ba, const half *bb, const T* biasPtr) {
    int eU = param->elhPack[0];
    int lU = param->elhPack[1];
    int hU = param->elhPack[2];
    int maxCount = eU * hU * warpSize * param->batch;
    extern __shared__ float sharedMemory[];
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int oIndex = index / warpSize;
        int subIndex = oIndex % (eU * hU);
        int bIndex = oIndex / (eU * hU);
        int wrapId = threadIdx.x / warpSize;
        int laneId = threadIdx.x % warpSize;
        int warpM = subIndex % eU;
        int warpN = subIndex / eU;
        T* c = bc + bIndex * param->elh[0] * param->elh[2];
        const half* a = ba + bIndex * param->elhPack[1] * param->elhPack[0] * 16 * 16;
        const half* b = bb + bIndex * param->elhPack[1] * param->elhPack[2] * 16 * 16;
        float* cache = sharedMemory + wrapId * 16 * 16;
        // Declare the fragments
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, LayoutA>
            a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, LayoutB>
            b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

        wmma::fill_fragment(acc_frag, 0.0f);
        const half* aStart = a + warpM * param->aPStride[0];
        const half* bStart = b + warpN * param->bPStride[0];
        //printf("GemmPacked: %d - %d - %d, numele: %d, %d\n", eU, lU, hU, a_frag.num_elements, b_frag.num_elements);
        // MLA
        for (int i = 0; i < lU; ++i) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, aStart + i * param->aPStride[1], param->aPStride[2]);
            wmma::load_matrix_sync(b_frag, bStart + i * param->bPStride[1], param->bPStride[2]);
            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        wmma::store_matrix_sync(cache, acc_frag, 16, wmma::mem_row_major);
        int eSta = warpM * 16;
        int eEnd = min(eSta + 16, param->elh[0]);
        int hSta = warpN * 16;
        int hEnd = min(hSta + 16, param->elh[2]);
        int eC = eEnd - eSta;
        int hC = hEnd - hSta;
        T* dstStart = c + hSta * param->cStride[2];
        if (nullptr != biasPtr) {
            for (int tId = laneId; tId < eC * hC; tId += warpSize) {
                int y = tId % eC;
                int x = tId / eC;
                int ye = y + eSta;
                float value = cache[16 * y + x];
                float biasValue = biasPtr[hSta + x];
                dstStart[ye * param->cStride[0] + x * param->cStride[2]] = value + biasValue;
            }
        } else {
            for (int tId = laneId; tId < eC * hC; tId += warpSize) {
                int y = tId % eC;
                int x = tId / eC;
                int ye = y + eSta;
                float value = cache[16 * y + x];
                dstStart[ye * param->cStride[0] + x * param->cStride[2]] = value;
            }
        }
    }
}

template<typename T>
__global__ void GemmPackedFull(const MatMulParam* param, const int iBlock, T *c, const half *a, const half *b, const T* biasPtr) {
    size_t eU = param->elhPack[0];
    size_t lU = param->elhPack[1];
    size_t hU = param->elhPack[2];
    size_t maxCount = eU * hU * warpSize;
    size_t wrapId = threadIdx.x / warpSize;
    size_t laneId = threadIdx.x % warpSize;
    extern __shared__ float sharedMemory[];
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        size_t subIndex = index / warpSize;
        size_t warpM = subIndex % eU;
        size_t warpN = subIndex / eU;
        T* cache = (T*)(sharedMemory + wrapId * 16 * 16);
        // Declare the fragments
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
            a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>
            b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, T> acc_frag;

        wmma::load_matrix_sync(acc_frag, biasPtr + 16 * warpN, 0, wmma::mem_row_major);
        const half* aStart = a + warpM * lU * 16 * 16;
        const half* bStart = b + warpN * lU * 16 * 16;
        //printf("GemmPacked: %d - %d - %d, numele: %d, %d\n", eU, lU, hU, a_frag.num_elements, b_frag.num_elements);
        // MLA
        for (size_t i = 0; i < lU; ++i) {
            wmma::load_matrix_sync(a_frag, aStart + i * 256, 16);
            wmma::load_matrix_sync(b_frag, bStart + i * 256, 16);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        for(size_t t=0; t<acc_frag.num_elements; t++){
            acc_frag.x[t] = max(acc_frag.x[t], param->minValue);
            acc_frag.x[t] = min(acc_frag.x[t], param->maxValue);
        }

        size_t eSta = (warpM + iBlock*eU) * 16;
        if(eSta >= (size_t)param->elh[0]) {
            continue;
        }
        size_t eEnd = ((eSta + (size_t)16) > (size_t)param->elh[0]) ? (size_t)param->elh[0] : (eSta + (size_t)16);

        size_t eC = eEnd - eSta;
        T* dstStart = (T*)(c + warpN * 16 * (size_t)param->elh[0] + eSta * 16);
        wmma::store_matrix_sync(cache, acc_frag, 16, wmma::mem_row_major);
        if (warpSize % 16 == 0) {
            size_t r = warpSize / 16;
            size_t x = laneId / r;
            size_t ysta = laneId % r;
            for (size_t y = ysta; y < eC; y+=r) {
                float value = *((T*)(cache + 16 * y + x));
                dstStart[y * 16 + x] = value;
            }
        } else {
            for (size_t tId = laneId; tId < eC * 16; tId += warpSize) {
                size_t y = tId % eC;
                size_t x = tId / eC;
                float value = *((T*)(cache + 16 * y + x));
                dstStart[y * 16 + x] = value;
            }
        }
    }
}

template<typename T>
__global__ void GemmPackedFull16x32(const MatMulParam* param, const int iBlock, T *c, const half *a, const half *b, const T* biasPtr) {
    size_t eU = param->elhPack[0];
    size_t lU = param->elhPack[1];
    size_t hU = param->elhPack[2];
    size_t threadCount = blockDim.x / warpSize;
    size_t maxCount = eU * hU;
    size_t wrapId = threadIdx.x / warpSize;
    size_t laneId = threadIdx.x % warpSize;
    extern __shared__ float sharedMemory[];
    T* cache = (T*)(sharedMemory + wrapId * 16 * 32);
    for (size_t index = blockIdx.x * threadCount + wrapId; index < maxCount; index += gridDim.x * threadCount) {
        size_t warpM = index % eU;
        size_t warpN = index / eU;
        // Declare the fragments
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
            MA0;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>
            MB0;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>
            MB1;
        wmma::fragment<wmma::accumulator, 16, 16, 16, T> MC0;
        wmma::fragment<wmma::accumulator, 16, 16, 16, T> MC1;

        wmma::load_matrix_sync(MC0, biasPtr + 32 * warpN + 0, 0, wmma::mem_row_major);
        wmma::load_matrix_sync(MC1, biasPtr + 32 * warpN + 16, 0, wmma::mem_row_major);
        const half* aStart = a + warpM * lU * 16 * 16;
        const half* bStart = b + warpN * lU * 16 * 32;
        //printf("GemmPacked: %d - %d - %d, numele: %d, %d\n", eU, lU, hU, a_frag.num_elements, b_frag.num_elements);
        // MLA
        for (size_t i = 0; i < lU; ++i) {
            wmma::load_matrix_sync(MA0, aStart + i * 256 + 0, 16);
            wmma::load_matrix_sync(MB0, bStart + i * 512, 16);
            wmma::load_matrix_sync(MB1, bStart + i * 512 + 256, 16);
            wmma::mma_sync(MC0, MA0, MB0, MC0);
            wmma::mma_sync(MC1, MA0, MB1, MC1);
        }
        for(size_t t=0; t<MC0.num_elements; t++){
            MC0.x[t] = max(MC0.x[t], param->minValue);
            MC0.x[t] = min(MC0.x[t], param->maxValue);
        }
        for(size_t t=0; t<MC1.num_elements; t++){
            MC1.x[t] = max(MC1.x[t], param->minValue);
            MC1.x[t] = min(MC1.x[t], param->maxValue);
        }
        size_t eSta = (warpM + iBlock*eU) * 16;
        if(eSta >= (size_t)param->elh[0]) {
            continue;
        }
        size_t eEnd = ((eSta + (size_t)16) > (size_t)param->elh[0]) ? (size_t)param->elh[0] : (eSta + (size_t)16);
        size_t eC = eEnd - eSta;
        T* dst0 = (T*)(c + warpN * 32 * (size_t)param->elh[0] + eSta * 16);
        T* dst1 = (T*)(c + (warpN * 32 + 16) * (size_t)param->elh[0] + eSta * 16);
        // First 8x32
        wmma::store_matrix_sync(cache, MC0, 16, wmma::mem_row_major);
        // Second 8x32
        wmma::store_matrix_sync(cache + 256, MC1, 16, wmma::mem_row_major);
        auto dst = dst0;
        auto src = cache;
        if (laneId >= 16) {
            dst = dst1;
            src = cache + 256;
        }
        size_t x = laneId % 16;
        for (size_t y = 0; y < eC; ++y) {
            dst[y * 16 + x] = src[y * 16 + x];
        }
    }
}

template<typename T>
__global__ void GemmPackedFull32x16(const MatMulParam* param, const int iBlock, T *c, const half *a, const half *b, const T* biasPtr) {
    size_t eU = param->elhPack[0];
    size_t lU = param->elhPack[1];
    size_t hU = param->elhPack[2];
    size_t threadCount = blockDim.x / warpSize;
    size_t maxCount = eU * hU;
    size_t wrapId = threadIdx.x / warpSize;
    size_t laneId = threadIdx.x % warpSize;
    extern __shared__ float sharedMemory[];
    T* cache = (T*)(sharedMemory + wrapId * 32 * 16);
    for (size_t index = blockIdx.x * threadCount + wrapId; index < maxCount; index += gridDim.x * threadCount) {
        size_t warpN = index % hU;
        size_t warpM = index / hU;
        // Declare the fragments
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
            MA0;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
            MA1;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>
            MB0;
        wmma::fragment<wmma::accumulator, 16, 16, 16, T> MC0;
        wmma::fragment<wmma::accumulator, 16, 16, 16, T> MC1;

        wmma::load_matrix_sync(MC0, biasPtr + 16 * warpN + 0, 0, wmma::mem_row_major);
        for(size_t t=0; t<MC0.num_elements; t++){
            MC1.x[t] = MC0.x[t];
        }

        const half* aStart = a + warpM * lU * 32 * 16;
        const half* bStart = b + warpN * lU * 16 * 16;
        //printf("GemmPacked: %d - %d - %d, numele: %d, %d\n", eU, lU, hU, a_frag.num_elements, b_frag.num_elements);
        // MLA
        for (size_t i = 0; i < lU; ++i) {
            wmma::load_matrix_sync(MA0, aStart + i * 512 + 0, 16);
            wmma::load_matrix_sync(MA1, aStart + i * 512 + 256, 16);
            wmma::load_matrix_sync(MB0, bStart + i * 256 + 0, 16);
            wmma::mma_sync(MC0, MA0, MB0, MC0);
            wmma::mma_sync(MC1, MA1, MB0, MC1);
        }
        for(size_t t=0; t<MC0.num_elements; t++){
            MC0.x[t] = max(MC0.x[t], param->minValue);
            MC0.x[t] = min(MC0.x[t], param->maxValue);
        }
        for(size_t t=0; t<MC1.num_elements; t++){
            MC1.x[t] = max(MC1.x[t], param->minValue);
            MC1.x[t] = min(MC1.x[t], param->maxValue);
        }
        size_t eSta = (warpM + iBlock*eU) * 32;
        if(eSta >= (size_t)param->elh[0]) {
            continue;
        }
        size_t eEnd = ((eSta + (size_t)16) > (size_t)param->elh[0]) ? (size_t)param->elh[0] : (eSta + (size_t)16);
        size_t eC = eEnd - eSta;
        T* dst0 = (T*)(c + warpN * 16 * (size_t)param->elh[0] + eSta * 16);
        T* dst1 = (T*)(dst0 + 256);
        // First 8x32
        wmma::store_matrix_sync(cache, MC0, 16, wmma::mem_row_major);
        // Second 8x32
        wmma::store_matrix_sync(cache + 256, MC1, 16, wmma::mem_row_major);
        auto dst = dst0;
        auto src = cache;
        if (laneId >= 16) {
            dst = dst1;
            src = cache + 256;
        }
        size_t x = laneId % 16;
        for (size_t y = 0; y < eC; ++y) {
            dst[y * 16 + x] = src[y * 16 + x];
        }
    }
}

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
} // extern "C"
