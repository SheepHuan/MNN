// topkv2.cu - TopKAllRows / GetResultAllRows kernels + helpers + shim
//   source/backend/cuda/execution/TopKV2Execution.cu
#include "corpus_common.cuh"

namespace MNN {
namespace Corpus {

// ============================================================================
// TopKV2: source/backend/cuda/execution/TopKV2Execution.cu
// Full re-implementation of all device helpers + 2 kernels.
// ============================================================================
template<typename indexT, typename valueT>
__device__ inline void topk_siftDown(const int K, const int descendFlag, valueT* valuesThread, indexT* indicesThread) {
    int parent = 0;
    while (true) {
        int child = 2 * parent + 1;
        if (child >= K) break;
        if (child + 1 < K && (valueT)(descendFlag) * valuesThread[child + 1] < (valueT)(descendFlag) * valuesThread[child]) child++;
        if ((valueT)(descendFlag) * valuesThread[parent] > (valueT)(descendFlag) * valuesThread[child]) {
            valueT tmpV = valuesThread[parent]; valuesThread[parent] = valuesThread[child]; valuesThread[child] = tmpV;
            indexT tmpI = indicesThread[parent]; indicesThread[parent] = indicesThread[child]; indicesThread[child] = tmpI;
            parent = child;
        } else break;
    }
}

template<typename indexT, typename valueT>
__device__ void topk_TopKInThread(const valueT* inputDevice, indexT* indicesThread, valueT* valuesThread, const int K, const int numElePerRow, const valueT minValue, const int descendFlag) {
    for (int i = 0; i < K; i++) { indicesThread[i] = -1; valuesThread[i] = (valueT)(descendFlag) * minValue; }
    int idxFirstEleInRow = threadIdx.x + blockIdx.x * blockDim.x;
    for (indexT i = idxFirstEleInRow; i < numElePerRow; i += gridDim.x * blockDim.x) {
        valueT data = inputDevice[i];
        if ((valueT)(descendFlag) * data > (valueT)(descendFlag) * valuesThread[0]) {
            valuesThread[0] = data; indicesThread[0] = i; topk_siftDown<indexT, valueT>(K, descendFlag, valuesThread, indicesThread);
        }
    }
    for (int i = K - 1; i > 0; i--) {
        valueT tmpV = valuesThread[0]; valuesThread[0] = valuesThread[i]; valuesThread[i] = tmpV;
        indexT tmpI = indicesThread[0]; indicesThread[0] = indicesThread[i]; indicesThread[i] = tmpI;
        topk_siftDown<indexT, valueT>(i, descendFlag, valuesThread, indicesThread);
    }
}

template<typename indexT, typename valueT>
__device__ void topk_ReduceTopK(indexT* indicesArray, valueT* valuesArray, const int offset1, const int offset2, const int K, const int descendFlag) {
    indexT idx1 = offset1 + K - 1; indexT idx2 = offset2 + K - 1; indexT idxVirtual = offset1 + 2 * K - 1;
    while (idx2 >= offset2) {
        if (idx1 < offset1) {
            while (idxVirtual >= offset1) { indicesArray[idxVirtual] = indicesArray[offset2 + (idxVirtual - offset1)]; valuesArray[idxVirtual] = valuesArray[offset2 + (idxVirtual - offset1)]; idxVirtual--; }
            break;
        }
        if ((valueT)(descendFlag) * valuesArray[idx1] <= (valueT)(descendFlag) * valuesArray[idx2]) {
            if (idxVirtual <= offset1 + K - 1) { indicesArray[idxVirtual] = indicesArray[idx1]; valuesArray[idxVirtual] = valuesArray[idx1]; }
            idx1--;
        } else {
            if (idxVirtual <= offset1 + K - 1) { indicesArray[idxVirtual] = indicesArray[idx2]; valuesArray[idxVirtual] = valuesArray[idx2]; }
            idx2--;
        }
        idxVirtual--;
    }
}

template<typename indexT, typename valueT>
__device__ void topk_TopKOneRow(const valueT* inputDevice, indexT* indicesBlock, valueT* valuesBlock, indexT* tempIndicesDevice, valueT* tempValuesDevice, const int K, const int lengthRow, valueT minValue, const int descendFlag) {
    indexT* indicesThread = indicesBlock + threadIdx.x * K;
    valueT* valuesThread = valuesBlock + threadIdx.x * K;
    topk_TopKInThread<indexT, valueT>(inputDevice, indicesThread, valuesThread, K, lengthRow, minValue, descendFlag);
    __syncthreads();
    for (int stride = (blockDim.x >> 1); stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) topk_ReduceTopK<indexT, valueT>(indicesBlock, valuesBlock, threadIdx.x * K, (threadIdx.x + stride) * K, K, descendFlag);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int i = 0; i < K; i++) { tempIndicesDevice[K * blockIdx.x + i] = indicesBlock[i]; tempValuesDevice[K * blockIdx.x + i] = valuesBlock[i]; }
    }
}

template<typename indexT, typename valueT>
__global__ void TopKAllRows(const valueT* inputDevice, indexT* tempIndicesDevice, valueT* tempValuesDevice, const int K, const int lengthRow, valueT minValue, const int descendFlag) {
    extern __shared__ char smem[];
    indexT* indicesBlock = reinterpret_cast<indexT*>(smem);
    valueT* valuesBlock = reinterpret_cast<valueT*>(&smem[blockDim.x * K * sizeof(indexT)]);
    int idxRow = blockIdx.y;
    const valueT* inputDeviceThisRow = inputDevice + idxRow * lengthRow;
    indexT* tempIndicesDeviceThisRow = tempIndicesDevice + idxRow * gridDim.x * K;
    valueT* tempValuesDeviceThisRow = tempValuesDevice + idxRow * gridDim.x * K;
    topk_TopKOneRow<indexT, valueT>(inputDeviceThisRow, indicesBlock, valuesBlock, tempIndicesDeviceThisRow, tempValuesDeviceThisRow, K, lengthRow, minValue, descendFlag);
    __syncthreads();
}

template<typename indexT, typename valueT>
__device__ void topk_GetResultOneRow(indexT* outputIndicesDevice, valueT* outputValuesDevice, indexT* tempIndicesDevice, valueT* tempValuesDevice, indexT* finalIndices, valueT* finalValues, const int K, const int reduceLength, const int descendFlag) {
    if (threadIdx.x < reduceLength) {
        for (int i = 0; i < K; i++) { finalIndices[threadIdx.x * K + i] = tempIndicesDevice[threadIdx.x * K + i]; finalValues[threadIdx.x * K + i] = tempValuesDevice[threadIdx.x * K + i]; }
    }
    __syncthreads();
    int stride = blockDim.x >> 1;
    if ((threadIdx.x < stride) && (threadIdx.x + stride < reduceLength)) topk_ReduceTopK<indexT, valueT>(finalIndices, finalValues, threadIdx.x * K, (threadIdx.x + stride) * K, K, descendFlag);
    __syncthreads(); stride >>= 1;
    for (; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) topk_ReduceTopK<indexT, valueT>(finalIndices, finalValues, threadIdx.x * K, (threadIdx.x + stride) * K, K, descendFlag);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int i = 0; i < K; i++) { outputIndicesDevice[i] = finalIndices[i]; outputValuesDevice[i] = finalValues[i]; }
    }
}

template<typename indexT, typename valueT>
__global__ void GetResultAllRows(indexT* outputIndicesDevice, valueT* outputValuesDevice, indexT* tempIndicesDevice, valueT* tempValuesDevice, const int K, const int numBlockPerRow, const int descendFlag) {
    extern __shared__ char smem[];
    indexT* finalIndices = reinterpret_cast<indexT*>(smem);
    valueT* finalValues = reinterpret_cast<valueT*>(&smem[numBlockPerRow * K * sizeof(indexT)]);
    int idxRow = blockIdx.x;
    indexT* outputIndicesThisRow = outputIndicesDevice + idxRow * K;
    valueT* outputValuesThisRow = outputValuesDevice + idxRow * K;
    indexT* tempIndicesThisRow = tempIndicesDevice + idxRow * numBlockPerRow * K;
    valueT* tempValuesThisRow = tempValuesDevice + idxRow * numBlockPerRow * K;
    topk_GetResultOneRow<indexT, valueT>(outputIndicesThisRow, outputValuesThisRow, tempIndicesThisRow, tempValuesThisRow, finalIndices, finalValues, K, numBlockPerRow, descendFlag);
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- TopKV2 (two-stage: TopKAllRows + GetResultAllRows) ----
// Single shim that allocates temp buffer, launches both kernels, frees temp.
void mnn_corpus_topkv2_fp32(const float* input, int* outIndices, float* outValues,
                            int K, int lengthRow, int numRow, int descendFlag,
                            int grid1x, int grid1y, int block1, int smem1,
                            int grid2, int block2, int smem2,
                            cudaStream_t stream) {
    // Calculate how many blocks per row (based on grid1x)
    int numBlockPerRow = grid1x;
    int numBlockTotal = numBlockPerRow * numRow;
    // Allocate temp buffers
    int* tempIndices;
    float* tempValues;
    cudaMalloc(&tempIndices, numBlockTotal * K * sizeof(int));
    cudaMalloc(&tempValues, numBlockTotal * K * sizeof(float));
    dim3 g1((unsigned)grid1x, (unsigned)grid1y);
    dim3 b1((unsigned)block1);
    MNN::Corpus::TopKAllRows<int, float><<<g1, b1, smem1 * (sizeof(float) + sizeof(int)), stream>>>(
        input, tempIndices, tempValues, K, lengthRow, -1e30f, descendFlag);
    dim3 g2((unsigned)grid2);
    dim3 b2((unsigned)block2);
    MNN::Corpus::GetResultAllRows<int, float><<<g2, b2, smem2 * (sizeof(float) + sizeof(int)), stream>>>(
        outIndices, outValues, tempIndices, tempValues, K, numBlockPerRow, descendFlag);
    cudaStreamSynchronize(stream);
    cudaFree(tempIndices);
    cudaFree(tempValues);
}

} // extern "C"
