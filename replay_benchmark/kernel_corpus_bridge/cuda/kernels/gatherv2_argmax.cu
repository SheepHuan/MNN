// gatherv2_argmax.cu - GATHERV2 / ARGMAX / ARGMIN / ARGMAX_120 kernels + shims
//   source/backend/cuda/execution/GatherV2Execution.cu
//   source/backend/cuda/execution/ArgMaxExecution.cu / ArgMinExecution.cu
#include "corpus_common.cuh"

namespace MNN {
namespace Corpus {

// ============================================================================
// GatherV2: source/backend/cuda/execution/GatherV2Execution.cu
// ============================================================================
template <typename T>
__global__ void GATHERV2(const int count, const int outside, const int inside, const int iNum, const int oNum,
                         const T* input, const int* indice, T* output) {
    CUDA_KERNEL_LOOP(i, count) {
        int x = i % inside;
        int y = i / inside;
        const int o = y / oNum;
        const int n = y % oNum;
        T* outPtr = output + inside * oNum * o;
        const T* inpPtr = input + inside * iNum * o;
        outPtr[n * inside + x] = inpPtr[indice[n] * inside + x];
    }
}

// ============================================================================
// ArgMax/ArgMin: source/backend/cuda/execution/ArgMaxExecution.cu / ArgMinExecution.cu
// ============================================================================
template <typename T>
__global__ void ARGMAX(const int count, const int outside, const int inside, const int dim,
                       const T* input, int* output) {
    CUDA_KERNEL_LOOP(i, count) {
        const int idx_out = i / inside;
        const int idx_in = i % inside;
        int* outPtr = output + idx_out * inside + idx_in;
        const T* inpPtr = input + idx_out * inside * dim + idx_in;
        int index = 0;
        T maxValue = inpPtr[0 * inside];
        for (int j = 1; j < dim; j++) {
            T value = inpPtr[j * inside];
            if (maxValue < value) { index = j; maxValue = value; }
        }
        outPtr[0] = index;
    }
}
template <typename T>
__global__ void ARGMIN(const int count, const int outside, const int inside, const int dim,
                       const T* input, int* output) {
    CUDA_KERNEL_LOOP(i, count) {
        const int o = i / inside;
        const int n = i % inside;
        int* outPtr = output + inside * o;
        const T* inpPtr = input + inside * dim * o;
        int index = 0;
        T minValue = inpPtr[n + 0 * inside];
        for (int j = 1; j < dim; j++) {
            T value = inpPtr[n + j * inside];
            if (minValue > value) { index = j; minValue = value; }
        }
        outPtr[n] = index;
    }
}

// ---- 1.2.0 tag: ARGMAX (output is T* not int*; stores index as float) ----
template <typename T>
__global__ void ARGMAX_120(const int count, const int outside, const int inside, const int dim,
                           const T* input, T* output) {
    CUDA_KERNEL_LOOP(i, count) {
        const int o = i / inside;
        const int n = i % inside;
        T* outPtr = output + inside * o;
        const T* inpPtr = input + inside * dim * o;
        int index = 0;
        T maxValue = inpPtr[0];
        for (int j = 1; j < dim; j++) {
            T value = inpPtr[j * inside];
            if (maxValue < value) { index = j; maxValue = value; }
        }
        outPtr[n] = (T)index;
    }
}

// ---- ARGMAX 1.2.7: T* output (index stored as T) ----
template <typename T>
__global__ void ARGMAX_127(const int count, const int outside, const int inside, const int dim,
                           const T* input, T* output) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)count; i += blockDim.x * gridDim.x) {
        const int o = i / inside;
        const int n = i % inside;
        T* outPtr = output + inside * o;
        const T* inpPtr = input + inside * dim * o;
        int index = 0;
        T maxValue = inpPtr[0];
        for (int j = 1; j < dim; j++) {
            T value = inpPtr[j * inside];
            if (maxValue < value) { index = j; maxValue = value; }
        }
        outPtr[n] = (T)index;
    }
}

// ---- ARGMAX 1.2.8 / 2.1.2: int* output, n+offset indexing ----
template <typename T>
__global__ void ARGMAX_128(const int count, const int outside, const int inside, const int dim,
                           const T* input, int* output) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)count; i += blockDim.x * gridDim.x) {
        const int o = i / inside;
        const int n = i % inside;
        int* outPtr = output + inside * o;
        const T* inpPtr = input + inside * dim * o;
        int index = 0;
        T maxValue = inpPtr[n + 0 * inside];
        for (int j = 1; j < dim; j++) {
            T value = inpPtr[n + j * inside];
            if (maxValue < value) { index = j; maxValue = value; }
        }
        outPtr[n] = index;
    }
}

// ---- ARGMAX 2.5.0: pointer-baked offset, outPtr[0] ----
template <typename T>
__global__ void ARGMAX_250(const int count, const int outside, const int inside, const int dim,
                           const T* input, int* output) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)count; i += blockDim.x * gridDim.x) {
        const int idx_out = i / inside;
        const int idx_in = i % inside;
        int* outPtr = output + idx_out * inside + idx_in;
        const T* inpPtr = input + idx_out * inside * dim + idx_in;
        int index = 0;
        T maxValue = inpPtr[0 * inside];
        for (int j = 1; j < dim; j++) {
            T value = inpPtr[j * inside];
            if (maxValue < value) { index = j; maxValue = value; }
        }
        outPtr[0] = index;
    }
}

// ---- ARGMAX_FIRST_STEP (2.5.0+, body stable to 3.6.0) ----
// Two-stage argmax for dim > 256. FIRST_STEP reduces dim into chunks of
// ARG_REDUCE_NUM, producing partial max values + indices.
#define ARG_REDUCE_NUM 256
template <typename T>
__global__ void ARGMAX_FIRST_STEP(const int count, const int outside, const int inside,
                            const int totalDims, const int dims, const int numDims,
                            const T *input, T *outputData, int *outputIndex) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)count; index += blockDim.x * gridDim.x) {
        const int idx_in = index % inside;
        const int tmp = index / inside;
        const int idx_out = tmp % outside;
        const int idx_num_dim = tmp / outside;

        const int idx_output = (idx_out * numDims + idx_num_dim) * inside + idx_in;
        const T* inpPtr = input + (idx_out * totalDims + idx_num_dim * dims) * inside + idx_in;
        int maxIndex = idx_num_dim * dims;
        T maxValue = inpPtr[0 * inside];
        for(int j=1; j<dims; j++) {
            const int idx_access = idx_num_dim * dims + j;
            if(idx_access < totalDims) {
                T value = inpPtr[j * inside];
                if(maxValue < value) {
                    maxIndex = idx_access;
                    maxValue = value;
                }
            }
        }
        outputData[idx_output] = maxValue;
        outputIndex[idx_output] = maxIndex;
    }
}

// ---- ARGMAX_SECOND_STEP 3.6.0 (fixed: correct baseInputIndex offset) ----
template <typename T>
__global__ void ARGMAX_SECOND_STEP(const int count, const int outside, const int inside, const int dims,
                            const T *inputData, const int *inputIndex, int *outputIndex) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)count; index += blockDim.x * gridDim.x) {
        const int idx_in = index % inside;
        const int idx_out = index / inside;

        int idx_output = idx_out * inside + idx_in;
        const T* inpPtr = inputData + idx_out * dims * inside + idx_in;
        const int* baseInputIndex = inputIndex + idx_out * dims * inside + idx_in;
        int maxIndex = baseInputIndex[0];
        T maxValue = inpPtr[0 * inside];
        for(int j=1; j<dims; j++) {
            T value = inpPtr[j * inside];
            if(maxValue < value) {
                maxIndex = baseInputIndex[j];
                maxValue = value;
            }
        }
        outputIndex[idx_output] = maxIndex;
    }
}

// ---- ARGMAX_SECOND_STEP 2.5.0-2.8.4 (buggy: no offset on inputIndex) ----
template <typename T>
__global__ void ARGMAX_SECOND_STEP_250(const int count, const int outside, const int inside, const int dims,
                            const T *inputData, const int *inputIndex, int *outputIndex) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)count; index += blockDim.x * gridDim.x) {
        const int idx_in = index % inside;
        const int idx_out = index / inside;

        int idx_output = idx_out * inside + idx_in;
        const T* inpPtr = inputData + idx_out * dims * inside + idx_in;
        int maxIndex = inputIndex[0];
        T maxValue = inpPtr[0 * inside];
        for(int j=1; j<dims; j++) {
            T value = inpPtr[j * inside];
            if(maxValue < value) {
                maxIndex = inputIndex[j];
                maxValue = value;
            }
        }
        outputIndex[idx_output] = maxIndex;
    }
}
#undef ARG_REDUCE_NUM

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- GatherV2 ----
void mnn_corpus_gatherv2_fp32(const int count, const int outside, const int inside, const int iNum, const int oNum,
                               const float* input, const int* indice, float* output, int grid, int block,
                               cudaStream_t stream) {
    MNN::Corpus::GATHERV2<float><<<grid, block, 0, stream>>>(count, outside, inside, iNum, oNum, input, indice, output);
}

// ---- ArgMax/ArgMin ----
void mnn_corpus_argmax_fp32(const int count, const int outside, const int inside, const int dim,
                             const float* input, int* output, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::ARGMAX<float><<<grid, block, 0, stream>>>(count, outside, inside, dim, input, output);
}
void mnn_corpus_argmin_fp32(const int count, const int outside, const int inside, const int dim,
                             const float* input, int* output, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::ARGMIN<float><<<grid, block, 0, stream>>>(count, outside, inside, dim, input, output);
}

// ---- fp16 variants ----
void mnn_corpus_gatherv2_fp16(const int count, const int outside, const int inside, const int iNum, const int oNum,
                               const void* input, const int* indice, void* output, int grid, int block,
                               cudaStream_t stream) {
    MNN::Corpus::GATHERV2<half><<<grid, block, 0, stream>>>(count, outside, inside, iNum, oNum, (const half*)input, indice, (half*)output);
}
void mnn_corpus_argmax_fp16(const int count, const int outside, const int inside, const int dim,
                              const void* input, int* output, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::ARGMAX<half><<<grid, block, 0, stream>>>(count, outside, inside, dim, (const half*)input, output);
}
void mnn_corpus_argmin_fp16(const int count, const int outside, const int inside, const int dim,
                              const void* input, int* output, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::ARGMIN<half><<<grid, block, 0, stream>>>(count, outside, inside, dim, (const half*)input, output);
}

// ---- 1.2.0 tag: ARGMAX (output is T* not int*) ----
void mnn_corpus_argmax_120_fp32(const int count, const int outside, const int inside, const int dim,
                                 const float* input, float* output, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::ARGMAX_120<float><<<grid, block, 0, stream>>>(count, outside, inside, dim, input, output);
}

// ---- ARGMAX 1.2.7 (T* output) ----
void mnn_corpus_argmax_127_fp32(const int count, const int outside, const int inside, const int dim,
                                  const float* input, float* output, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::ARGMAX_127<float><<<grid, block, 0, stream>>>(count, outside, inside, dim, input, output);
}
// ---- ARGMAX 1.2.8 / 2.1.2 (int* output, n+offset) ----
void mnn_corpus_argmax_128_fp32(const int count, const int outside, const int inside, const int dim,
                                  const float* input, int* output, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::ARGMAX_128<float><<<grid, block, 0, stream>>>(count, outside, inside, dim, input, output);
}
void mnn_corpus_argmax_212_fp32(const int count, const int outside, const int inside, const int dim,
                                  const float* input, int* output, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::ARGMAX_128<float><<<grid, block, 0, stream>>>(count, outside, inside, dim, input, output);
}
// ---- ARGMAX 2.5.0 (pointer-baked) ----
void mnn_corpus_argmax_250_fp32(const int count, const int outside, const int inside, const int dim,
                                  const float* input, int* output, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::ARGMAX_250<float><<<grid, block, 0, stream>>>(count, outside, inside, dim, input, output);
}

// ---- ARGMAX two-stage (dim > 256): 3.6.0 SECOND_STEP (fixed offset) ----
// Shim does FIRST_STEP + SECOND_STEP in sequence. Caller provides temp buffers.
void mnn_corpus_argmax_twostage_fp32(const int outside, const int inside, const int dim,
                                       const float* input, int* output,
                                       float* tempData, int* tempIndex,
                                       int grid1, int block1, int grid2, int block2,
                                       cudaStream_t stream) {
    const int ARG_REDUCE_NUM = 256;
    const int numDims = (dim + ARG_REDUCE_NUM - 1) / ARG_REDUCE_NUM;
    const int count1 = outside * inside * numDims;
    const int count2 = outside * inside;
    MNN::Corpus::ARGMAX_FIRST_STEP<float><<<grid1, block1, 0, stream>>>(
        count1, outside, inside, dim, ARG_REDUCE_NUM, numDims, input, tempData, tempIndex);
    MNN::Corpus::ARGMAX_SECOND_STEP<float><<<grid2, block2, 0, stream>>>(
        count2, outside, inside, numDims, tempData, tempIndex, output);
}

// ---- ARGMAX two-stage 2.5.0-2.8.4 (buggy SECOND_STEP: no offset) ----
void mnn_corpus_argmax_twostage_250_fp32(const int outside, const int inside, const int dim,
                                           const float* input, int* output,
                                           float* tempData, int* tempIndex,
                                           int grid1, int block1, int grid2, int block2,
                                           cudaStream_t stream) {
    const int ARG_REDUCE_NUM = 256;
    const int numDims = (dim + ARG_REDUCE_NUM - 1) / ARG_REDUCE_NUM;
    const int count1 = outside * inside * numDims;
    const int count2 = outside * inside;
    MNN::Corpus::ARGMAX_FIRST_STEP<float><<<grid1, block1, 0, stream>>>(
        count1, outside, inside, dim, ARG_REDUCE_NUM, numDims, input, tempData, tempIndex);
    MNN::Corpus::ARGMAX_SECOND_STEP_250<float><<<grid2, block2, 0, stream>>>(
        count2, outside, inside, numDims, tempData, tempIndex, output);
}

} // extern "C"
