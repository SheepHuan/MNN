// range.cu - RANGE kernel + shims (source/backend/cuda/execution/RangeExecution.cu)
#include "corpus_common.cuh"

namespace MNN {
namespace Corpus {

template <typename T>
__global__ void RANGE(const int size, const T* input0, const T* input2, T* output) {
    CUDA_KERNEL_LOOP(i, size) {
        T start = input0[0];
        T step = input2[0];
        output[i] = start + (T)i * step;
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

void mnn_corpus_range_fp32(const int size, const float* start, const float* step, float* output, int grid, int block,
                           cudaStream_t stream) {
    MNN::Corpus::RANGE<float><<<grid, block, 0, stream>>>(size, start, step, output);
}
void mnn_corpus_range_i32(const int size, const int* start, const int* step, int* output, int grid, int block,
                          cudaStream_t stream) {
    MNN::Corpus::RANGE<int><<<grid, block, 0, stream>>>(size, start, step, output);
}
void mnn_corpus_range_fp16(const int size, const void* start, const void* step, void* output, int grid, int block,
                           cudaStream_t stream) {
    MNN::Corpus::RANGE<half><<<grid, block, 0, stream>>>(size, (const half*)start, (const half*)step, (half*)output);
}

} // extern "C"
