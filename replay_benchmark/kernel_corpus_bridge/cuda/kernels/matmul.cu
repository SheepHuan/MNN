// matmul.cu - GENERAL_BATCH_MATMUL + matmul_gemv_kernel (3.6.0)
//   source/backend/cuda/execution/MatMulExecution.cu
//   PackPadFill is a CUTLASS preprocessing kernel — not included here because
//   it cannot be validated independently without the CUTLASS GEMM that follows.
//   + 1.2.7-era tensor-core Gemm kernels (GemmPrearrange/GemmPacked*) from
//   legacy_kernels.cu — require <mma.h> + nvcuda::wmma fragments.
#include "corpus_common.cuh"
#include <mma.h>

using namespace nvcuda;
using half = __half;

// 1.2.7-era tensor-core matmul macros (from legacy_kernels.cu)
#ifndef PACK_NUMBER
#define PACK_NUMBER 16
#endif
#define PACK_NUMBER_C2 (PACK_NUMBER/2)
#define MATMULPACK 16
#define MATMULPACK2 (MATMULPACK * MATMULPACK)
#define BLOCK_INT4 2
#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4
#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2
#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)
#define CHUNK_L 4
#define CHUNK_E 4
#define CHUNK_H 4

namespace MNN {
namespace Corpus {

// 1.2.7-era MatMul parameter struct (from legacy_kernels.cu)
struct MatMulParam {
    int elh[3]; int elhPack[3]; int aStride[3]; int bStride[3]; int cStride[3];
    int aPStride[3]; int bPStride[3]; int batch; float minValue; float maxValue;
};

// ============================================================================
// GENERAL_BATCH_MATMUL: naive batched matmul for large-batch small-problem
//   [b,e,l] x [b,l,h] -> [b,e,h] (with all transpose combinations)
// ============================================================================
template<typename T0, typename T1>
__global__ void GENERAL_BATCH_MATMUL(
    const T0* A, const T0* B, const T0* bias,
    bool transA, bool transB,
    const int coefBatchA, const int coefBatchB,
    const int e, const int l, const int h,
    const int maxCount, T1* C,
    DivModFast d_e, DivModFast d_h
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int bIndex, hIndex, eIndex, tmp;
        d_h.divmod(index, tmp, hIndex);
        d_e.divmod(tmp, bIndex, eIndex);

        float sum = 0.0;
        if(!transA && !transB) {
            const T0* basePtrA = A + (coefBatchA * bIndex * e + eIndex) * l;
            const T0* basePtrB = B + (coefBatchB * bIndex * l + 0) * h + hIndex;
            T1* basePtrC       = C + (bIndex * e + eIndex) * h + hIndex;
            for(int i = 0; i < l; i++) {
                sum += (float)basePtrA[i] * (float)basePtrB[i * h];
            }
            if(bias != nullptr) {
                sum += (float)bias[hIndex];
            }
            basePtrC[0] = (T1)sum;
            return;
        }
        if(transA && !transB) {
            const T0* basePtrA = A + (coefBatchA * bIndex * l + 0) * e + eIndex;
            const T0* basePtrB = B + (coefBatchB * bIndex * l + 0) * h + hIndex;
            T1* basePtrC       = C + (bIndex * e + eIndex) * h + hIndex;
            for(int i = 0; i < l; i++) {
                sum += (float)basePtrA[i * e] * (float)basePtrB[i * h];
            }
            if(bias != nullptr) {
                sum += (float)bias[hIndex];
            }
            basePtrC[0] = (T1)sum;
            return;
        }
        if(transA && transB) {
            const T0* basePtrA = A + (coefBatchA * bIndex * l + 0) * e + eIndex;
            const T0* basePtrB = B + (coefBatchB * bIndex * h + hIndex) * l + 0;
            T1* basePtrC       = C + (bIndex * e + eIndex) * h + hIndex;
            for(int i = 0; i < l; i++) {
                sum += (float)basePtrA[i * e] * (float)basePtrB[i];
            }
            if(bias != nullptr) {
                sum += (float)bias[hIndex];
            }
            basePtrC[0] = (T1)sum;
            return;
        }
        if(!transA && transB) {
            const T0* basePtrA = A + (coefBatchA * bIndex * e + eIndex) * l + 0;
            const T0* basePtrB = B + (coefBatchB * bIndex * h + hIndex) * l + 0;
            T1* basePtrC       = C + (bIndex * e + eIndex) * h + hIndex;
            for(int i = 0; i < l; i++) {
                sum += (float)basePtrA[i] * (float)basePtrB[i];
            }
            if(bias != nullptr) {
                sum += (float)bias[hIndex];
            }
            basePtrC[0] = (T1)sum;
            return;
        }
    }
}

// ============================================================================
// matmul_gemv_kernel: GEMV for M=1 (decode stage optimization)
//   A: [batch, 1, L] or [batch, L, 1] if transA
//   B: [batch, L, H] or [batch, H, L] if transB
//   C: [batch, 1, H]
// ============================================================================
template<typename T>
__global__ void matmul_gemv_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    const T* __restrict__ bias,
    T* __restrict__ C,
    bool transA, bool transB,
    int coefBatchA, int coefBatchB,
    int l, int h, int batch
) {
    const int h_idx = blockIdx.x;
    const int b_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (h_idx >= h || b_idx >= batch) return;

    const int bA = coefBatchA * b_idx;
    const int bB = coefBatchB * b_idx;

    float sum = 0.0f;

    if (!transA && !transB) {
        const T* a_ptr = A + bA * l;
        for (int k = tid; k < l; k += blockDim.x) {
            sum += (float)a_ptr[k] * (float)B[bB * l * h + k * h + h_idx];
        }
    } else if (!transA && transB) {
        const T* a_ptr = A + bA * l;
        const T* b_ptr = B + bB * h * l + h_idx * l;
        for (int k = tid; k < l; k += blockDim.x) {
            sum += (float)a_ptr[k] * (float)b_ptr[k];
        }
    } else if (transA && !transB) {
        for (int k = tid; k < l; k += blockDim.x) {
            sum += (float)A[bA * l + k] * (float)B[bB * l * h + k * h + h_idx];
        }
    } else {
        const T* b_ptr = B + bB * h * l + h_idx * l;
        for (int k = tid; k < l; k += blockDim.x) {
            sum += (float)A[bA * l + k] * (float)b_ptr[k];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    __shared__ float warp_sums[32];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < num_warps; w++) {
            total += warp_sums[w];
        }
        if (bias != nullptr) {
            total += (float)bias[h_idx];
        }
        C[b_idx * h + h_idx] = (T)total;
    }
}

// ============================================================================
// 1.2.7-era legacy tensor-core Gemm kernels (from legacy_kernels.cu)
// ============================================================================
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

} // namespace Corpus
} // namespace MNN

extern "C" {

void mnn_corpus_general_batch_matmul_fp32(
    const float* A, const float* B, const float* bias,
    bool transA, bool transB,
    int coefBatchA, int coefBatchB,
    int e, int l, int h, int maxCount,
    float* C,
    int d_e_val, int d_h_val,
    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_e(d_e_val);
    MNN::Corpus::DivModFast d_h(d_h_val);
    MNN::Corpus::GENERAL_BATCH_MATMUL<float, float><<<grid, block, 0, stream>>>(
        A, B, bias, transA, transB, coefBatchA, coefBatchB, e, l, h, maxCount, C,
        d_e, d_h);
}

void mnn_corpus_matmul_gemv_fp32(
    const float* A, const float* B, const float* bias,
    float* C,
    bool transA, bool transB,
    int coefBatchA, int coefBatchB,
    int l, int h, int batch,
    int gridX, int gridY, int block, cudaStream_t stream) {
    dim3 grid(gridX, gridY);
    MNN::Corpus::matmul_gemv_kernel<float><<<grid, block, 0, stream>>>(
        A, B, bias, C, transA, transB, coefBatchA, coefBatchB, l, h, batch);
}

} // extern "C"
