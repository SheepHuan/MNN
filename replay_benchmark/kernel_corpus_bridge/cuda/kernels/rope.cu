// rope.cu - ropeC4Kernel (3.6.0) kernel + shim
//   source/backend/cuda/execution/RoPEExecution.cu
//   Complete re-implementation including QNorm/KNorm (RMSNorm) path.
//   gamma=nullptr + useNorm=false when no norm is configured.
#include "corpus_common.cuh"

namespace MNN {
namespace Corpus {

template <typename T>
__global__ void ropeC4Kernel(const T* q, const T* k, const T* cos, const T* sin, T* qOut, T* kOut,
                             const float* qGamma, const float* kGamma,
                             int seqLen, int numHead, int kvNumHead, int headDim, int ropeHalfDim,
                             int qHiddenPack, int kHiddenPack, float qEps, float kEps,
                             bool qNorm, bool kNorm) {
    const int fullHead = numHead + kvNumHead;
    const int tokenHead = blockIdx.x;
    const int token = tokenHead / fullHead;
    const int combinedHead = tokenHead - token * fullHead;
    const bool isQ = combinedHead < numHead;
    const int head = isQ ? combinedHead : combinedHead - numHead;
    const int hiddenPack = isQ ? qHiddenPack : kHiddenPack;
    const T* input = isQ ? q : k;
    T* output = isQ ? qOut : kOut;
    const float* gamma = isQ ? qGamma : kGamma;
    const bool useNorm = isQ ? qNorm : kNorm;
    const float eps = isQ ? qEps : kEps;
    const int headCount = isQ ? numHead : kvNumHead;
    const int inputBase = token * hiddenPack + head * headDim;
    const int outputBase = (token * headCount + head) * headDim;

    __shared__ float squareSum[128];
    float scale = 1.0f;
    if (useNorm) {
        float localSum = 0.0f;
        for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
            float value = static_cast<float>(input[inputBase + d]);
            localSum += value * value;
        }
        squareSum[threadIdx.x] = localSum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                squareSum[threadIdx.x] += squareSum[threadIdx.x + stride];
            }
            __syncthreads();
        }
        scale = rsqrtf(squareSum[0] / static_cast<float>(headDim) + eps);
    }
    const int ropeDim = ropeHalfDim * 2;
    const int trigBase = token * ropeDim;

    for (int d = threadIdx.x; d < ropeHalfDim; d += blockDim.x) {
        float even = static_cast<float>(input[inputBase + d]);
        float odd = static_cast<float>(input[inputBase + d + ropeHalfDim]);
        if (useNorm) {
            even *= scale * gamma[d];
            odd *= scale * gamma[d + ropeHalfDim];
        }
        const float cEven = static_cast<float>(cos[trigBase + d]);
        const float cOdd = static_cast<float>(cos[trigBase + d + ropeHalfDim]);
        const float sEven = static_cast<float>(sin[trigBase + d]);
        const float sOdd = static_cast<float>(sin[trigBase + d + ropeHalfDim]);
        output[outputBase + d] = static_cast<T>(even * cEven - odd * sEven);
        output[outputBase + d + ropeHalfDim] = static_cast<T>(odd * cOdd + even * sOdd);
    }
    for (int d = ropeDim + threadIdx.x; d < headDim; d += blockDim.x) {
        float value = static_cast<float>(input[inputBase + d]);
        if (useNorm) {
            value *= scale * gamma[d];
        }
        output[outputBase + d] = static_cast<T>(value);
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

void mnn_corpus_rope_c4_fp32(const float* q, const float* k, const float* cos, const float* sin,
                             float* qOut, float* kOut, const float* qGamma, const float* kGamma,
                             int seqLen, int numHead, int kvNumHead, int headDim, int ropeHalfDim,
                             int qHiddenPack, int kHiddenPack, float qEps, float kEps,
                             bool qNorm, bool kNorm,
                             int grid, int block, cudaStream_t stream) {
    MNN::Corpus::ropeC4Kernel<float><<<grid, block, 0, stream>>>(
        q, k, cos, sin, qOut, kOut, qGamma, kGamma, seqLen, numHead, kvNumHead,
        headDim, ropeHalfDim, qHiddenPack, kHiddenPack, qEps, kEps, qNorm, kNorm);
}

} // extern "C"
