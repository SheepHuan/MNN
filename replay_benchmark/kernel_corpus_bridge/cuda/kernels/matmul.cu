// matmul.cu - GENERAL_BATCH_MATMUL + matmul_gemv_kernel (3.6.0)
//   source/backend/cuda/execution/MatMulExecution.cu
//   PackPadFill is a CUTLASS preprocessing kernel — not included here because
//   it cannot be validated independently without the CUTLASS GEMM that follows.
#include "corpus_common.cuh"

namespace MNN {
namespace Corpus {

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
