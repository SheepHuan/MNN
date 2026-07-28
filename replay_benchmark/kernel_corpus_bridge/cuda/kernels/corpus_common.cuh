// corpus_common.cuh - Shared device helpers for all kernels/*.cu files.
// Included at the top of each split kernel file so they share the same
// warp/block reduction primitives, CUDA_KERNEL_LOOP macro, DivModFast, and
// the ReduceParam_127 struct without ODR violations (all definitions here
// are inline/device-template, not host-linkable symbols).
//
// This file is included from within `namespace MNN { namespace Corpus {` —
// each kernels/<op>.cu opens that namespace, includes this header, then
// declares its __global__ kernels and extern "C" shims.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>
#include <float.h>

namespace MNN {
namespace Corpus {

// ============================================================================
// Device helpers (inlined from source/backend/cuda/execution/MNNCUDAFunction.cuh)
// ============================================================================
#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    }
    return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;
    val = warpReduceSum<T>(val);
    if (lane == 0) { shared[wid] = val; }
    __syncthreads();
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)0.0f;
    val = warpReduceSum(val);
    return val;
}

template <typename T>
__inline__ __device__ T warpReduceMax(T val) {
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    }
    return val;
}

template <typename T>
__inline__ __device__ T blockReduceMax(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;
    val = warpReduceMax<T>(val);
    if (lane == 0) { shared[wid] = val; }
    __syncthreads();
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)(-1.0e30f);
    val = warpReduceMax(val);
    return val;
}

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
#define HALF_MIN half(-65504)

// DivModFast (re-implemented here so each kernels/*.cu TU is self-contained
// without depending on CorpusKernels.cu's private copy).
struct DivModFast {
    uint32_t d_;
    uint32_t l_;
    uint32_t m_;
    DivModFast(int d = 1) {
        d_ = (d == 0) ? 1 : d;
        for (l_ = 0;; ++l_) { if ((1U << l_) >= d_) break; }
        uint64_t one = 1;
        uint64_t mm = ((one << 32) * ((one << l_) - d_)) / d_ + 1;
        m_ = static_cast<uint32_t>(mm);
    }
    __device__ __inline__ int div(int idx) const {
        uint32_t tm = __umulhi(m_, idx);
        return (tm + idx) >> l_;
    }
    __device__ __inline__ int mod(int idx) const { return idx - d_ * div(idx); }
    __device__ __inline__ void divmod(int idx, int& quo, int& rem) const { quo = div(idx); rem = idx - d_ * quo; }
};

// ReduceParam_127 (1.2.7 Reduction/SOFTMAX parameter struct — {inside, axis, outside}).
struct ReduceParam_127 { int inside; int axis; int outside; };

} // namespace Corpus
} // namespace MNN
