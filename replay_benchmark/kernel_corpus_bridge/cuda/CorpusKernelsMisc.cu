// CorpusKernelsMisc.cu - B-class kernel corpus (Gather/Argmax/Argmin/
// Interp/Transpose/GridSample). Split from CorpusKernels.cu to keep
// file sizes manageable. Compiled by nvcc into replay_cuda_corpus.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>
#include <float.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
#define HALF_MIN half(-65504)

// ============================================================================
// GatherV2: source/backend/cuda/execution/GatherV2Execution.cu
// ============================================================================
namespace MNN { namespace Corpus {

// Forward declarations of kernels defined in CorpusKernels.cu (same library)
template <typename T> __global__ void maxpool_C8(const T*, T*, int, int, int, int, int, int, int, int, int, int, int, int);
template <typename T> __global__ void avgpool_C8(const T*, T*, int, int, int, int, int, int, int, int, int, int, int, int);
template <typename T> __global__ void global_avgpool_C8(const T*, T*, int, int, int, int, int);
template <typename T> __global__ void global_maxpool_C8(const T*, T*, int, int, int, int, int);

// ============================================================================
// 1.2.0 tag kernels — NCHW layout (not NC4HW4 packed)
// ============================================================================
template <typename T>
__global__ void maxpool_120(const T* uInput, T* uOutput,
    int bc, int ih, int iw, int oh, int ow,
    int padX, int padY, int kernelX, int kernelY, int strideX, int strideY) {
    int total = bc * oh * ow;
    CUDA_KERNEL_LOOP(i, total) {
        int x = i % ow;
        int tmp = i / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int ix = x * strideX - padX;
        int iy = y * strideY - padY;
        int sx = max(0, -ix);
        int sy = max(0, -iy);
        int ex = min(kernelX, iw - ix);
        int ey = min(kernelY, ih - iy);
        T maxValue = (T)(-1000000);
        for (int fy = sy; fy < ey; ++fy) {
            for (int fx = sx; fx < ex; ++fx) {
                T inputColor = uInput[z * iw * ih + (iy + fy) * iw + (ix + fx)];
                maxValue = max(inputColor, maxValue);
            }
        }
        uOutput[z * ow * oh + y * ow + x] = maxValue;
    }
}
template <typename T>
__global__ void avgpool_120(const T* uInput, T* uOutput,
    int bc, int ih, int iw, int oh, int ow,
    int padX, int padY, int kernelX, int kernelY, int strideX, int strideY) {
    int total = bc * oh * ow;
    CUDA_KERNEL_LOOP(i, total) {
        int x = i % ow;
        int tmp = i / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int ix = x * strideX - padX;
        int iy = y * strideY - padY;
        int sx = max(0, -ix);
        int sy = max(0, -iy);
        int ex = min(kernelX, iw - ix);
        int ey = min(kernelY, ih - iy);
        T sumValue = (T)0;
        for (int fy = sy; fy < ey; ++fy) {
            for (int fx = sx; fx < ex; ++fx) {
                T inputColor = uInput[z * iw * ih + (iy + fy) * iw + (ix + fx)];
                sumValue = sumValue + inputColor;
            }
        }
        uOutput[z * ow * oh + y * ow + x] = sumValue / ((T)(ey - sy) * (T)(ex - sx));
    }
}

// ============================================================================
// 1.2.0 tag: Reduction (T accumulation, param order: inside, axis, outside)
// ============================================================================
template <typename T>
__global__ void SUM_120(const T* input, T* output, int inside, int axis, int outside) {
    int count = inside * outside;
    CUDA_KERNEL_LOOP(i, count) {
        int y = i / inside;
        int x = i % inside;
        T sumValue = (T)0;
        const T* basicInput = input + y * axis * inside + x;
        for (int v = 0; v < axis; ++v) sumValue += basicInput[v * inside];
        output[y * inside + x] = sumValue;
    }
}
template <typename T>
__global__ void MEAN_120(const T* input, T* output, int inside, int axis, int outside) {
    int count = inside * outside;
    CUDA_KERNEL_LOOP(i, count) {
        int y = i / inside;
        int x = i % inside;
        T sumValue = (T)0;
        const T* basicInput = input + y * axis * inside + x;
        for (int v = 0; v < axis; ++v) sumValue += basicInput[v * inside];
        output[y * inside + x] = sumValue / (T)axis;
    }
}

// DivModFast (re-defined here because it's in a separate TU from CorpusKernels.cu)
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

// ============================================================================
// Interp nearest: source/backend/cuda/execution/InterpExecution.cu
// ============================================================================
template <typename T>
__global__ void INTERP_NERAEST(const int total, const int c_p,
                                const int ih, const int iw, const int oh, const int ow,
                                const float scaleh, const float scalew, const float offseth, const float offsetw,
                                const T* in, T* out) {
    CUDA_KERNEL_LOOP(index, total) {
        int tmp0 = index / c_p;
        int c_idx = index % c_p;
        int x = tmp0 % ow;
        int tmp = tmp0 / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int ix = min(max(0, (int)floor((float)x * scalew + offsetw)), iw - 1);
        int iy = min(max(0, (int)floor((float)y * scaleh + offseth)), ih - 1);
        out[((z * oh + y) * ow + x) * c_p + c_idx] = in[((z * ih + iy) * iw + ix) * c_p + c_idx];
    }
}

// ============================================================================
// Transpose format conversion: source/backend/cuda/execution/Transpose.cu
// NHWC_2_NCHW and NCHW_2_NHWC are the simplest (no DivModFast dependency).
// ============================================================================
template <typename T0, typename T1>
__global__ void NHWC_2_NCHW(const T0* input, T1* output,
                            int total, int inside, int axis, int outside) {
    CUDA_KERNEL_LOOP(index, total) {
        int x = index % inside;
        int y = (index / inside) % axis;
        int z = index / (inside * axis);
        output[z * axis * inside + y * inside + x] = input[index];
    }
}
template <typename T0, typename T1>
__global__ void NCHW_2_NHWC(const T0* input, T1* output,
                            int total, int inside, int axis, int outside) {
    CUDA_KERNEL_LOOP(index, total) {
        int x = index % inside;
        int y = (index / inside) % axis;
        int z = index / (inside * axis);
        output[index] = input[z * axis * inside + y * inside + x];
    }
}

// ============================================================================
// GridSample: source/backend/cuda/execution/GridSampleExecution.cu
// Inline device helpers (getPosition/sample/CLAMP) are re-implemented here
// because they live in an anonymous namespace in the original .cu file.
// BorderMode is passed as int (0=ZEROS,1=BORDER,2=REFLECTION,3=CUBE).
// ============================================================================
inline __device__ float gsGetPosition(float x, int range, bool alignCorners) {
    float a = alignCorners ? 1.0f : 0.0f;
    float b = alignCorners ? 0.0f : 1.0f;
    return ((1.0f + x) * (range - a) - b) / 2.0f;
}
inline __device__ int gsClamp(int value, int minV, int maxV) {
    return min(max(value, minV), maxV);
}
inline __device__ int gsSample(int pos, int total, int paddingMode) {
    if (pos < 0 || pos >= total) {
        if (paddingMode == 0) return -1; // ZEROS
        pos = gsClamp(pos, 0, total - 1);
    }
    return pos;
}

template <typename T>
__global__ void GRID_SAMPLE_NEAREST(const int count, const T* input, const T* grid, T* output,
                                     const int input_height, const int input_width,
                                     const int output_height, const int output_width,
                                     const int channel, const int channel_pack,
                                     int paddingMode, bool alignCorners) {
    CUDA_KERNEL_LOOP(index, count) {
        int idx_cp = index % channel;
        int idx_nhw = index / channel;
        int idx_ow = idx_nhw % output_width;
        int idx_nh = idx_nhw / output_width;
        int idx_oh = idx_nh % output_height;
        int idx_ob = idx_nh / output_height;
        float pos_x = grid[idx_nhw * 2 + 0];
        float pos_y = grid[idx_nhw * 2 + 1];
        float in_grid_x = gsGetPosition(pos_x, input_width, alignCorners);
        float in_grid_y = gsGetPosition(pos_y, input_height, alignCorners);
        int in_pos_x = (int)floor(in_grid_x + 0.5f);
        int in_pos_y = (int)floor(in_grid_y + 0.5f);
        in_pos_x = gsSample(in_pos_x, input_width, paddingMode);
        in_pos_y = gsSample(in_pos_y, input_height, paddingMode);
        int dst_offset = ((idx_ob * output_height + idx_oh) * output_width + idx_ow) * channel_pack + idx_cp;
        if (in_pos_x == -1 || in_pos_y == -1) {
            output[dst_offset] = (T)0.0;
            continue;
        }
        output[dst_offset] = input[((idx_ob * input_height + in_pos_y) * input_width + in_pos_x) * channel_pack + idx_cp];
    }
}

// ============================================================================
// Interp bilinear / round: source/backend/cuda/execution/InterpExecution.cu
// ============================================================================
template <typename T>
__global__ void INTERP_BILINEAR(const int total, const int c_p,
                                const int ih, const int iw, const int oh, const int ow,
                                const float scaleh, const float scalew, const float offseth, const float offsetw,
                                const T* in, T* out) {
    CUDA_KERNEL_LOOP(index, total) {
        int tmp0 = index / c_p;
        int c_idx = index % c_p;
        int x = tmp0 % ow;
        int tmp = tmp0 / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        float fx = x * scalew + offsetw;
        int ix_0 = min(max(0, (int)floor(fx)), iw - 1);
        int ix_1 = min((int)ceil(fx), iw - 1);
        float fy = y * scaleh + offseth;
        int iy_0 = min(max(0, (int)floor(fy)), ih - 1);
        int iy_1 = min((int)ceil(fy), ih - 1);
        int index_00 = ((z * ih + iy_0) * iw + ix_0) * c_p + c_idx;
        int index_01 = ((z * ih + iy_0) * iw + ix_1) * c_p + c_idx;
        int index_10 = ((z * ih + iy_1) * iw + ix_0) * c_p + c_idx;
        int index_11 = ((z * ih + iy_1) * iw + ix_1) * c_p + c_idx;
        float factor_x = fx - ix_0;
        float factor_y = fy - iy_0;
        out[((z * oh + y) * ow + x) * c_p + c_idx] = (T)(
            (1.0f - factor_x) * (1.0f - factor_y) * (float)in[index_00]
            + factor_x * (1.0f - factor_y) * (float)in[index_01]
            + (1.0f - factor_x) * factor_y * (float)in[index_10]
            + factor_x * factor_y * (float)in[index_11]);
    }
}
template <typename T>
__global__ void INTERP_NERAEST_ROUND(const int total, const int c_p,
                                     const int ih, const int iw, const int oh, const int ow,
                                     const float scaleh, const float scalew, const float offseth, const float offsetw,
                                     const T* in, T* out) {
    CUDA_KERNEL_LOOP(index, total) {
        int tmp0 = index / c_p;
        int c_idx = index % c_p;
        int x = tmp0 % ow;
        int tmp = tmp0 / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int ix = min(max(0, (int)floor((float)x * scalew + offsetw + 0.499f)), iw - 1);
        int iy = min(max(0, (int)floor((float)y * scaleh + offseth + 0.499f)), ih - 1);
        out[((z * oh + y) * ow + x) * c_p + c_idx] = in[((z * ih + iy) * iw + ix) * c_p + c_idx];
    }
}

// ============================================================================
// GridSample bilinear: source/backend/cuda/execution/GridSampleExecution.cu
// ============================================================================
template <typename T>
__global__ void GRID_SAMPLE_BILINEAR(const int count, const T* input, const T* grid, T* output,
                                     const int input_height, const int input_width,
                                     const int output_height, const int output_width,
                                     const int channel, const int channel_pack,
                                     int paddingMode, bool alignCorners) {
    CUDA_KERNEL_LOOP(index, count) {
        int idx_cp = index % channel;
        int idx_nhw = index / channel;
        int idx_ow = idx_nhw % output_width;
        int idx_nh = idx_nhw / output_width;
        int idx_oh = idx_nh % output_height;
        int idx_ob = idx_nh / output_height;
        float pos_x = grid[idx_nhw * 2 + 0];
        float pos_y = grid[idx_nhw * 2 + 1];
        float in_grid_x = gsGetPosition(pos_x, input_width, alignCorners);
        float in_grid_y = gsGetPosition(pos_y, input_height, alignCorners);
        int in_pos_x0 = (int)floor(in_grid_x);
        int in_pos_y0 = (int)floor(in_grid_y);
        int in_pos_x1 = (int)ceil(in_grid_x);
        int in_pos_y1 = (int)ceil(in_grid_y);
        float x_weight = in_pos_x1 - in_grid_x;
        float y_weight = in_pos_y1 - in_grid_y;
        in_pos_x0 = gsSample(in_pos_x0, input_width, paddingMode);
        in_pos_y0 = gsSample(in_pos_y0, input_height, paddingMode);
        in_pos_x1 = gsSample(in_pos_x1, input_width, paddingMode);
        in_pos_y1 = gsSample(in_pos_y1, input_height, paddingMode);
        float in00 = (in_pos_y0 == -1 || in_pos_x0 == -1) ? 0.0f : (float)input[((idx_ob * input_height + in_pos_y0) * input_width + in_pos_x0) * channel_pack + idx_cp];
        float in01 = (in_pos_y0 == -1 || in_pos_x1 == -1) ? 0.0f : (float)input[((idx_ob * input_height + in_pos_y0) * input_width + in_pos_x1) * channel_pack + idx_cp];
        float in10 = (in_pos_y1 == -1 || in_pos_x0 == -1) ? 0.0f : (float)input[((idx_ob * input_height + in_pos_y1) * input_width + in_pos_x0) * channel_pack + idx_cp];
        float in11 = (in_pos_y1 == -1 || in_pos_x1 == -1) ? 0.0f : (float)input[((idx_ob * input_height + in_pos_y1) * input_width + in_pos_x1) * channel_pack + idx_cp];
        int dst_offset = ((idx_ob * output_height + idx_oh) * output_width + idx_ow) * channel_pack + idx_cp;
        output[dst_offset] = (T)(in00 * x_weight * y_weight + in01 * (1.0f - x_weight) * y_weight
                                  + in10 * x_weight * (1.0f - y_weight) + in11 * (1.0f - x_weight) * (1.0f - y_weight));
    }
}

// ============================================================================
// Reduction: source/backend/cuda/execution/ReductionTemplate.cuh
// ============================================================================
template <typename T>
__global__ void SUM_NAIVE(const T* input, T* output, const int outside, const int axis, const int inside) {
    int count = inside * outside;
    CUDA_KERNEL_LOOP(i, count) {
        int y = i / inside;
        int x = i % inside;
        float sumValue = 0.0f;
        const T* basicInput = input + y * axis * inside + x;
        for (int v = 0; v < axis; ++v) sumValue += (float)basicInput[v * inside];
        output[y * inside + x] = (T)sumValue;
    }
}
template <typename T>
__global__ void MEAN_NAIVE(const T* input, T* output, const int outside, const int axis, const int inside) {
    int count = inside * outside;
    CUDA_KERNEL_LOOP(i, count) {
        int y = i / inside;
        int x = i % inside;
        float sumValue = 0.0f;
        const T* basicInput = input + y * axis * inside + x;
        for (int v = 0; v < axis; ++v) sumValue += (float)basicInput[v * inside];
        output[y * inside + x] = (T)(sumValue / (float)axis);
    }
}
template <typename T>
__global__ void MINIMUM(const T* input, T* output, const int outside, const int axis, const int inside) {
    int count = inside * outside;
    CUDA_KERNEL_LOOP(i, count) {
        int y = i / inside;
        int x = i % inside;
        const T* basicInput = input + y * axis * inside + x;
        float res = (float)basicInput[0];
        for (int v = 1; v < axis; ++v) res = min((float)basicInput[v * inside], res);
        output[y * inside + x] = (T)res;
    }
}
template <typename T>
__global__ void MAXIMUM(const T* input, T* output, const int outside, const int axis, const int inside) {
    int count = inside * outside;
    CUDA_KERNEL_LOOP(i, count) {
        int y = i / inside;
        int x = i % inside;
        const T* basicInput = input + y * axis * inside + x;
        float res = (float)basicInput[0];
        for (int v = 1; v < axis; ++v) res = max((float)basicInput[v * inside], res);
        output[y * inside + x] = (T)res;
    }
}
template <typename T>
__global__ void PROD(const T* input, T* output, const int outside, const int axis, const int inside) {
    int count = inside * outside;
    CUDA_KERNEL_LOOP(i, count) {
        int y = i / inside;
        int x = i % inside;
        float sumValue = 1.0f;
        const T* basicInput = input + y * axis * inside + x;
        for (int v = 0; v < axis; ++v) sumValue *= (float)basicInput[v * inside];
        output[y * inside + x] = (T)sumValue;
    }
}

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

// ============================================================================
// 1.2.0 tag: ARGMAX (output is T* not int*; stores index as float)
// ============================================================================
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

// ============================================================================
// 1.2.0 tag: SCALE (scale/bias are T* not float*)
// ============================================================================
template <typename T>
__global__ void SCALE_120(const int n, const int channels, const int dim, const T* in, T* out,
                           const T* scaleData, const T* biasData) {
    CUDA_KERNEL_LOOP(index, n) {
        int c = (index / dim) % channels;
        out[index] = in[index] * scaleData[c] + biasData[c];
    }
}

// ============================================================================
// 1.2.0 tag: LAYERNORM (gamma/beta are T* not float*; no RMSNorm)
// ============================================================================
template <typename T>
__global__ void LAYERNORM_120(const int count, const int outside, const int inside, const float epsilon,
                               const T* in, T* out, const T* gamma_data, const T* beta_data) {
    CUDA_KERNEL_LOOP(i, count) {
        const int o = i / inside;
        const int index = i % inside;
        const T* inner_input = in + o * inside;
        T* inner_output = out + o * inside;
        T mean = (T)0;
        T sum = (T)0;
        for (int j = 0; j < inside; ++j) sum += inner_input[j];
        mean = sum / (T)inside;
        T square_sum = (T)0;
        for (int j = 0; j < inside; ++j) square_sum += (inner_input[j] - mean) * (inner_input[j] - mean);
        T variable = square_sum / (T)inside;
        variable = (T)1 / sqrt(variable + (T)epsilon);
        T res = (inner_input[index] - mean) * variable;
        if (gamma_data != nullptr && beta_data != nullptr) {
            res = res * gamma_data[index] + beta_data[index];
        }
        inner_output[index] = res;
    }
}

// ============================================================================
// 1.2.0 tag: PRELU (slope is T* not float*; div_factor instead of share_factor)
// ============================================================================
template <typename T>
__global__ void PRELU_120(const int n, const int channels, const int dim, const T* in, T* out,
                          const T* slopeData, int div_factor) {
    CUDA_KERNEL_LOOP(index, n) {
        int c = (index / dim) % channels / div_factor;
        out[index] = in[index] > 0 ? in[index] : in[index] * slopeData[c];
    }
}

// ============================================================================
// 1.2.0 tag: pack_c4 / unpack_c4 (NC4HW4 pack/unpack, 3.6.0 uses PACKCOMMON)
// ============================================================================
template <typename T>
__global__ void pack_c4_120(const T* input, T* output, int inside, int axis, int outside, int axisC4) {
    int total = inside * axis * outside;
    CUDA_KERNEL_LOOP(i, total) {
        int x = i % inside;
        int tmp = i / inside;
        int y = tmp % axis;
        int z = tmp / axis;
        int y4 = y / 4;
        int yR = y % 4;
        output[(z * axisC4 + y4) * inside * 4 + x * 4 + yR] = input[i];
    }
}
template <typename T>
__global__ void unpack_c4_120(const T* input, T* output, int inside, int axis, int outside, int axisC4) {
    int total = inside * axis * outside;
    CUDA_KERNEL_LOOP(i, total) {
        int x = i % inside;
        int tmp = i / inside;
        int y = tmp % axis;
        int z = tmp / axis;
        int y4 = y / 4;
        int yR = y % 4;
        output[i] = input[(z * axisC4 + y4) * inside * 4 + x * 4 + yR];
    }
}

// ============================================================================
// 1.2.0 tag: SETZERO (ScatterNd zero-fill)
// ============================================================================
template <typename T>
__global__ void SETZERO_120(const int n, T* outputPtr) {
    CUDA_KERNEL_LOOP(i, n) { outputPtr[i] = (T)0; }
}

// ============================================================================
// 1.2.0 tag: add_bias (MatMul bias addition)
// ============================================================================
template <typename T>
__global__ void add_bias_120(T* input, T* output, const T* bias, int e, int h) {
    CUDA_KERNEL_LOOP(i, e * h) {
        int hi = i % h;
        output[i] = input[i] + bias[hi];
    }
}

// ============================================================================
// 1.2.0 tag: INTERP (nearest, different signature from 3.6.0 INTERP_NERAEST)
// ============================================================================
template <typename T>
__global__ void INTERP_120(const int n, const int ih, const int iw, const int oh, const int ow,
                            const float scaleh, const float scalew, const float offseth, const float offsetw,
                            const T* in, T* out) {
    CUDA_KERNEL_LOOP(index, n) {
        int x = index % ow;
        int tmp = index / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int ix = min(max(0, (int)floor((float)x * scalew + offsetw)), iw - 1);
        int iy = min(max(0, (int)floor((float)y * scaleh + offseth)), ih - 1);
        out[(z * oh + y) * ow + x] = in[(z * ih + iy) * iw + ix];
    }
}

// ============================================================================
// 1.2.0 tag: INTERP_BILINEAR (different from 3.6.0: no c_p, uses n not total)
// ============================================================================
template <typename T>
__global__ void INTERP_BILINEAR_120(const int n, const int ih, const int iw, const int oh, const int ow,
                                     const float scaleh, const float scalew, const float offseth, const float offsetw,
                                     const T* in, T* out) {
    CUDA_KERNEL_LOOP(index, n) {
        int x = index % ow;
        int tmp = index / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        float fx = x * scalew + offsetw;
        int ix_0 = min(max(0, (int)floor(fx)), iw - 1);
        int ix_1 = min((int)ceil(fx), iw - 1);
        float fy = y * scaleh + offseth;
        int iy_0 = min(max(0, (int)floor(fy)), ih - 1);
        int iy_1 = min((int)ceil(fy), ih - 1);
        float factor_x = fx - ix_0;
        float factor_y = fy - iy_0;
        out[(z * oh + y) * ow + x] = (T)(
            (1.0f - factor_x) * (1.0f - factor_y) * (float)in[(z * ih + iy_0) * iw + ix_0]
            + factor_x * (1.0f - factor_y) * (float)in[(z * ih + iy_0) * iw + ix_1]
            + (1.0f - factor_x) * factor_y * (float)in[(z * ih + iy_1) * iw + ix_0]
            + factor_x * factor_y * (float)in[(z * ih + iy_1) * iw + ix_1]);
    }
}

// ============================================================================
// Raster blitRegion: source/backend/cuda/execution/Raster.cu
// ============================================================================
template <typename T>
__global__ void blitRegion(const T* inputO, T* outputO, int count, int loopCount,
                           const int32_t* dstIndice, const int32_t* srcIndice,
                           int dstUseIndice, int srcUseIndice, int dstStep, int srcStep, int srcLimit,
                           int sizeZ, int sizeY, int sizeX,
                           int strideZ, int strideY, int strideX,
                           int dstStrideZ, int dstStrideY, int dstStrideX) {
    for (size_t fuseIndex = blockIdx.x * blockDim.x + threadIdx.x; fuseIndex < count; fuseIndex += blockDim.x * gridDim.x) {
        int x = fuseIndex % sizeX;
        int temp = fuseIndex / sizeX;
        int y = temp % sizeY;
        temp = temp / sizeY;
        int z = temp % sizeZ;
        int i = temp / sizeZ;
        int srcOffsetO = i * srcStep;
        if (srcUseIndice >= 0) srcOffsetO = srcIndice[i] * srcStep;
        int dstOffsetO = i * dstStep;
        if (dstUseIndice >= 0) dstOffsetO = dstIndice[i] * dstStep;
        if (srcOffsetO >= 0 && srcOffsetO < srcLimit) {
            const T* input = inputO + srcOffsetO;
            T* output = outputO + dstOffsetO;
            int srcOffset = z * strideZ + y * strideY + x * strideX;
            int dstOffset = z * dstStrideZ + y * dstStrideY + x * dstStrideX;
            output[dstOffset] = input[srcOffset];
        } else {
            T* output = outputO + dstOffsetO;
            int dstOffset = z * dstStrideZ + y * dstStrideY + x * dstStrideX;
            output[dstOffset] = (T)0;
        }
    }
}

// ============================================================================
// GridSample 3D: NEAREST + BILINEAR
// ============================================================================
template <typename T>
__global__ void GRID_SAMPLE_NEAREST_3D(const int count, const T* input, const T* grid, T* output,
                                        int id, int ih, int iw, int od, int oh, int ow,
                                        int channel, int channel_pack, int paddingMode, bool alignCorners) {
    CUDA_KERNEL_LOOP(index, count) {
        int idx_cp = index % channel;
        int idx_nhw = index / channel;
        int idx_ow = idx_nhw % ow;
        int idx_nh = idx_nhw / ow;
        int idx_oh = idx_nh % oh;
        int idx_obd = idx_nh / oh;
        int idx_od = idx_obd % od;
        int idx_ob = idx_obd / od;
        float pos_x = grid[idx_nhw * 3 + 0];
        float pos_y = grid[idx_nhw * 3 + 1];
        float pos_z = grid[idx_nhw * 3 + 2];
        float igx = gsGetPosition(pos_x, iw, alignCorners);
        float igy = gsGetPosition(pos_y, ih, alignCorners);
        float igz = gsGetPosition(pos_z, id, alignCorners);
        int ipx = (int)floor(igx + 0.5f);
        int ipy = (int)floor(igy + 0.5f);
        int ipz = (int)floor(igz + 0.5f);
        ipx = gsSample(ipx, iw, paddingMode);
        ipy = gsSample(ipy, ih, paddingMode);
        ipz = gsSample(ipz, id, paddingMode);
        int dst = (((idx_ob * od + idx_od) * oh + idx_oh) * ow + idx_ow) * channel_pack + idx_cp;
        if (ipx == -1 || ipy == -1 || ipz == -1) { output[dst] = (T)0; continue; }
        output[dst] = input[(((idx_ob * id + ipz) * ih + ipy) * iw + ipx) * channel_pack + idx_cp];
    }
}
template <typename T>
__global__ void GRID_SAMPLE_BILINEAR_3D(const int count, const T* input, const T* grid, T* output,
                                        int id, int ih, int iw, int od, int oh, int ow,
                                        int channel, int channel_pack, int paddingMode, bool alignCorners) {
    CUDA_KERNEL_LOOP(index, count) {
        int idx_cp = index % channel;
        int idx_nhw = index / channel;
        int idx_ow = idx_nhw % ow;
        int idx_nh = idx_nhw / ow;
        int idx_oh = idx_nh % oh;
        int idx_obd = idx_nh / oh;
        int idx_od = idx_obd % od;
        int idx_ob = idx_obd / od;
        float pos_x = grid[idx_nhw * 3 + 0];
        float pos_y = grid[idx_nhw * 3 + 1];
        float pos_z = grid[idx_nhw * 3 + 2];
        float igx = gsGetPosition(pos_x, iw, alignCorners);
        float igy = gsGetPosition(pos_y, ih, alignCorners);
        float igz = gsGetPosition(pos_z, id, alignCorners);
        int ix0 = (int)floor(igx), ix1 = (int)ceil(igx);
        int iy0 = (int)floor(igy), iy1 = (int)ceil(igy);
        int iz0 = (int)floor(igz), iz1 = (int)ceil(igz);
        float xw = ix1 - igx, yw = iy1 - igy, zw = iz1 - igz;
        ix0 = gsSample(ix0, iw, paddingMode); iy0 = gsSample(iy0, ih, paddingMode); iz0 = gsSample(iz0, id, paddingMode);
        ix1 = gsSample(ix1, iw, paddingMode); iy1 = gsSample(iy1, ih, paddingMode); iz1 = gsSample(iz1, id, paddingMode);
        float v000 = (iz0==-1||iy0==-1||ix0==-1)?0:input[(((idx_ob*id+iz0)*ih+iy0)*iw+ix0)*channel_pack+idx_cp];
        float v001 = (iz0==-1||iy0==-1||ix1==-1)?0:input[(((idx_ob*id+iz0)*ih+iy0)*iw+ix1)*channel_pack+idx_cp];
        float v010 = (iz0==-1||iy1==-1||ix0==-1)?0:input[(((idx_ob*id+iz0)*ih+iy1)*iw+ix0)*channel_pack+idx_cp];
        float v011 = (iz0==-1||iy1==-1||ix1==-1)?0:input[(((idx_ob*id+iz0)*ih+iy1)*iw+ix1)*channel_pack+idx_cp];
        float v100 = (iz1==-1||iy0==-1||ix0==-1)?0:input[(((idx_ob*id+iz1)*ih+iy0)*iw+ix0)*channel_pack+idx_cp];
        float v101 = (iz1==-1||iy0==-1||ix1==-1)?0:input[(((idx_ob*id+iz1)*ih+iy0)*iw+ix1)*channel_pack+idx_cp];
        float v110 = (iz1==-1||iy1==-1||ix0==-1)?0:input[(((idx_ob*id+iz1)*ih+iy1)*iw+ix0)*channel_pack+idx_cp];
        float v111 = (iz1==-1||iy1==-1||ix1==-1)?0:input[(((idx_ob*id+iz1)*ih+iy1)*iw+ix1)*channel_pack+idx_cp];
        int dst = (((idx_ob*od+idx_od)*oh+idx_oh)*ow+idx_ow)*channel_pack+idx_cp;
        output[dst] = (T)(v000*xw*yw*zw + v001*(1-xw)*yw*zw + v010*xw*(1-yw)*zw + v011*(1-xw)*(1-yw)*zw
                           + v100*xw*yw*(1-zw) + v101*(1-xw)*yw*(1-zw) + v110*xw*(1-yw)*(1-zw) + v111*(1-xw)*(1-yw)*(1-zw));
    }
}

// ============================================================================
// Conv DepthWise: source/backend/cuda/execution/ConvDepthWiseExecution.cu
// Simplified fp32 re-implementation (processes 2 channels per thread like original).
// ============================================================================
template <typename T>
__global__ void CONV_DW(const T* input, const half* kernel, const half* bias, T* output,
                        float maxV, float minV, int iw, int ih, int c, int c_p,
                        int ow, int oh, int kw, int kh, int dw, int dh, int sw, int sh, int pw, int ph,
                        int total, DivModFast d_oc, DivModFast d_ow, DivModFast d_oh) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total / 2; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_2);
        d_ow.divmod(tmp1, tmp2, ox);
        d_oh.divmod(tmp2, ob, oy);
        int oz = oz_2 << 1;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        float color0 = bias[oz];
        float color1 = bias[oz + 1];
        int fxSta = max(0, (int)ceil(-(float)ix / dw));
        int fySta = max(0, (int)ceil(-(float)iy / dh));
        int fxEnd = min(kw, (int)ceil((float)(iw - ix) / dw));
        int fyEnd = min(kh, (int)ceil((float)(ih - iy) / dh));
        for (int fy = fySta; fy < fyEnd; ++fy) {
            for (int fx = fxSta; fx < fxEnd; ++fx) {
                int currentX = ix + fx * dw;
                int currentY = iy + fy * dh;
                const half* k0 = kernel + (oz * kh + fy) * kw + fx;
                const half* k1 = kernel + ((oz + 1) * kh + fy) * kw + fx;
                const T* inp = (const T*)(input + (ob * ih * iw + currentY * iw + currentX) * c_p + oz);
                float v0 = (float)inp[0];
                float v1 = (float)inp[1];
                color0 += v0 * (float)k0[0];
                color1 += v1 * (float)k1[0];
            }
        }
        color0 = max(minV, min(maxV, color0));
        color1 = max(minV, min(maxV, color1));
        T* dst0 = (T*)(output + (ob * oh * ow + oy * ow + ox) * c_p + oz);
        dst0[0] = (T)color0;
        dst0[1] = (T)color1;
    }
}

// ============================================================================
// Conv DepthWise 1.2.0: constBuffer struct param, NCHW layout, float kernel/bias
// (source/backend/cuda/execution/ConvDepthWiseExecution.cu @ tag 1.2.0)
// ============================================================================
struct ConvDwConstBuffer_120 {
    int pad[2];
    int kernelSize[2];
    int stride[2];
    int dilate[2];
    int inputSize[2];
    int outputSize[2];
    int channel;
    int subChannel;
    int total;
    int activationType;
};
__global__ void CONV_DW_120(const float* input, const float* kernel, const float* bias,
                             float* output, const ConvDwConstBuffer_120* u) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)u->total; i += blockDim.x * gridDim.x) {
        int iw = u->inputSize[0], ih = u->inputSize[1];
        int ow = u->outputSize[0], oh = u->outputSize[1];
        int kw = u->kernelSize[0], kh = u->kernelSize[1];
        int dw = u->dilate[0], dh = u->dilate[1];
        int sw = u->stride[0], sh = u->stride[1];
        int pw = u->pad[0], ph = u->pad[1];
        int acttype = u->activationType;
        int oz = i / (ow * oh);
        int tmp = i % (ow * oh);
        int oy = tmp / ow;
        int ox = tmp % ow;
        int kz = oz % u->subChannel;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        float color = (bias != nullptr) ? bias[kz] : 0.0f;
        for (int fy = 0; fy < kh; ++fy) {
            int sy = fy * dh + iy;
            if (sy >= ih || sy < 0) continue;
            for (int fx = 0; fx < kw; ++fx) {
                int sx = fx * dw + ix;
                if (sx >= iw || sx < 0) continue;
                float inputValue = input[sx + sy * iw + oz * iw * ih];
                float k = kernel[fx + fy * kw + kz * kw * kh];
                color += k * inputValue;
            }
        }
        color = (acttype == 1) ? max(0.0f, color) : (acttype == 2 ? (min(max(0.0f, color), 6.0f)) : color);
        output[ox + oy * ow + oz * ow * oh] = color;
    }
}

// ============================================================================
// Conv DepthWise 2.0.4: c_p-based indexing (signature same as 2.2.3 but
// single-channel-per-thread loop, no DivModFast). Reproduced from
// source/backend/cuda/execution/ConvDepthWiseExecution.cu @ tag 2.0.4.
// ============================================================================
template <typename T>
__global__ void CONV_DW_204(const T* input, const half* kernel, const half* bias, T* output,
                             float maxV, float minV, int iw, int ih, int c, int c_p,
                             int ow, int oh, int kw, int kh, int dw, int dh, int sw, int sh, int pw, int ph,
                             int total) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)total; index += blockDim.x * gridDim.x) {
        int oz = index % c_p;
        int tmp = index / c_p;
        int ox = tmp % ow;
        int oy = tmp / ow;
        int ob = oy / oh;
        int iy = oy % oh;
        int ix = ox * sw - pw;
        int iyv = iy * sh - ph;
        float color = (float)bias[oz];
        int fxSta = max(0, (int)ceil(-(float)ix / dw));
        int fySta = max(0, (int)ceil(-(float)iyv / dh));
        int fxEnd = min(kw, (int)ceil((float)(iw - ix) / dw));
        int fyEnd = min(kh, (int)ceil((float)(ih - iyv) / dh));
        for (int fy = fySta; fy < fyEnd; ++fy) {
            for (int fx = fxSta; fx < fxEnd; ++fx) {
                int currentX = ix + fx * dw;
                int currentY = iyv + fy * dh;
                const half* k = kernel + (oz * kh + fy) * kw + fx;
                const T* inp = (const T*)(input + (ob * ih * iw + currentY * iw + currentX) * c_p + oz);
                color += (float)inp[0] * (float)k[0];
            }
        }
        color = max(minV, min(maxV, color));
        T* dst = (T*)(output + (ob * oh * ow + oy * ow + ox) * c_p + oz);
        dst[0] = (T)color;
    }
}

} // namespace Corpus
} // namespace MNN

// ============================================================================
// B-class extern "C" launch shims
// ============================================================================
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

// ---- Interp nearest ----
void mnn_corpus_interp_nearest_fp32(const int total, const int c_p, int ih, int iw, int oh, int ow,
                                     float sh, float sw, float ohf, float owf,
                                     const float* in, float* out, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::INTERP_NERAEST<float><<<grid, block, 0, stream>>>(total, c_p, ih, iw, oh, ow, sh, sw, ohf, owf, in, out);
}

// ---- Transpose format conversion ----
void mnn_corpus_nhwc2nchw_fp32(const float* input, float* output, int total, int inside, int axis, int outside,
                                int grid, int block, cudaStream_t stream) {
    MNN::Corpus::NHWC_2_NCHW<float, float><<<grid, block, 0, stream>>>(input, output, total, inside, axis, outside);
}
void mnn_corpus_nchw2nhwc_fp32(const float* input, float* output, int total, int inside, int axis, int outside,
                                int grid, int block, cudaStream_t stream) {
    MNN::Corpus::NCHW_2_NHWC<float, float><<<grid, block, 0, stream>>>(input, output, total, inside, axis, outside);
}

// ---- GridSample nearest ----
void mnn_corpus_grid_sample_nearest_fp32(const int count, const float* input, const float* grid, float* output,
                                          int ih, int iw, int oh, int ow, int ch, int ch_p, int padMode, int align,
                                          int grid_, int block, cudaStream_t stream) {
    MNN::Corpus::GRID_SAMPLE_NEAREST<float><<<grid_, block, 0, stream>>>(
        count, input, grid, output, ih, iw, oh, ow, ch, ch_p, padMode, align != 0);
}

// ---- GridSample bilinear ----
void mnn_corpus_grid_sample_bilinear_fp32(const int count, const float* input, const float* grid, float* output,
                                           int ih, int iw, int oh, int ow, int ch, int ch_p, int padMode, int align,
                                           int grid_, int block, cudaStream_t stream) {
    MNN::Corpus::GRID_SAMPLE_BILINEAR<float><<<grid_, block, 0, stream>>>(
        count, input, grid, output, ih, iw, oh, ow, ch, ch_p, padMode, align != 0);
}

// ---- Interp bilinear / round ----
void mnn_corpus_interp_bilinear_fp32(const int total, const int c_p, int ih, int iw, int oh, int ow,
                                      float sh, float sw, float ohf, float owf,
                                      const float* in, float* out, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::INTERP_BILINEAR<float><<<grid, block, 0, stream>>>(total, c_p, ih, iw, oh, ow, sh, sw, ohf, owf, in, out);
}
void mnn_corpus_interp_nearest_round_fp32(const int total, const int c_p, int ih, int iw, int oh, int ow,
                                           float sh, float sw, float ohf, float owf,
                                           const float* in, float* out, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::INTERP_NERAEST_ROUND<float><<<grid, block, 0, stream>>>(total, c_p, ih, iw, oh, ow, sh, sw, ohf, owf, in, out);
}

// ---- Reduction (SUM/MAX/MIN/MEAN/PROD naive) ----
void mnn_corpus_reduction_sum_fp32(const float* input, float* output, int outside, int axis, int inside,
                                    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SUM_NAIVE<float><<<grid, block, 0, stream>>>(input, output, outside, axis, inside);
}
void mnn_corpus_reduction_mean_fp32(const float* input, float* output, int outside, int axis, int inside,
                                     int grid, int block, cudaStream_t stream) {
    MNN::Corpus::MEAN_NAIVE<float><<<grid, block, 0, stream>>>(input, output, outside, axis, inside);
}
void mnn_corpus_reduction_max_fp32(const float* input, float* output, int outside, int axis, int inside,
                                    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::MAXIMUM<float><<<grid, block, 0, stream>>>(input, output, outside, axis, inside);
}
void mnn_corpus_reduction_min_fp32(const float* input, float* output, int outside, int axis, int inside,
                                    int grid, int block, cudaStream_t stream) {
    MNN::Corpus::MINIMUM<float><<<grid, block, 0, stream>>>(input, output, outside, axis, inside);
}
void mnn_corpus_reduction_prod_fp32(const float* input, float* output, int outside, int axis, int inside,
                                     int grid, int block, cudaStream_t stream) {
    MNN::Corpus::PROD<float><<<grid, block, 0, stream>>>(input, output, outside, axis, inside);
}

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

// ---- Raster blitRegion ----
void mnn_corpus_blitregion_fp32(const float* input, float* output, int count, int loopCount,
                                 const int32_t* dstIndice, const int32_t* srcIndice,
                                 int dstUseIndice, int srcUseIndice, int dstStep, int srcStep, int srcLimit,
                                 int sizeZ, int sizeY, int sizeX,
                                 int strideZ, int strideY, int strideX,
                                 int dstStrideZ, int dstStrideY, int dstStrideX,
                                 int grid, int block, cudaStream_t stream) {
    MNN::Corpus::blitRegion<float><<<grid, block, 0, stream>>>(input, output, count, loopCount,
        dstIndice, srcIndice, dstUseIndice, srcUseIndice, dstStep, srcStep, srcLimit,
        sizeZ, sizeY, sizeX, strideZ, strideY, strideX, dstStrideZ, dstStrideY, dstStrideX);
}

// ---- fp16 variants of B-class kernels ----
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

// ---- GridSample 3D ----
void mnn_corpus_grid_sample_nearest_3d_fp32(const int count, const float* input, const float* grid, float* output,
                                            int id, int ih, int iw, int od, int oh, int ow,
                                            int ch, int ch_p, int padMode, int align,
                                            int grid_, int block, cudaStream_t stream) {
    MNN::Corpus::GRID_SAMPLE_NEAREST_3D<float><<<grid_, block, 0, stream>>>(
        count, input, grid, output, id, ih, iw, od, oh, ow, ch, ch_p, padMode, align != 0);
}
void mnn_corpus_grid_sample_bilinear_3d_fp32(const int count, const float* input, const float* grid, float* output,
                                              int id, int ih, int iw, int od, int oh, int ow,
                                              int ch, int ch_p, int padMode, int align,
                                              int grid_, int block, cudaStream_t stream) {
    MNN::Corpus::GRID_SAMPLE_BILINEAR_3D<float><<<grid_, block, 0, stream>>>(
        count, input, grid, output, id, ih, iw, od, oh, ow, ch, ch_p, padMode, align != 0);
}

// ---- Conv DepthWise fp32 (3.6.0 / 2.2.3+) ----
void mnn_corpus_conv_dw_fp32(const float* input, const half* kernel, const half* bias, float* output,
                             float maxV, float minV, int iw, int ih, int c, int c_p,
                             int ow, int oh, int kw, int kh, int dw, int dh, int sw, int sh, int pw, int ph,
                             int total, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::DivModFast d_oc(c_p / 2);
    MNN::Corpus::DivModFast d_ow(ow);
    MNN::Corpus::DivModFast d_oh(1);
    MNN::Corpus::CONV_DW<float><<<grid, block, 0, stream>>>(
        input, kernel, bias, output, maxV, minV, iw, ih, c, c_p, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph,
        total, d_oc, d_ow, d_oh);
}

// ---- Conv DepthWise 1.2.0: constBuffer struct, NCHW, float kernel/bias ----
void mnn_corpus_conv_dw_120_fp32(const float* input, const float* kernel, const float* bias, float* output,
                                  const MNN::Corpus::ConvDwConstBuffer_120* uConst, int grid, int block,
                                  cudaStream_t stream) {
    MNN::Corpus::CONV_DW_120<<<grid, block, 0, stream>>>(input, kernel, bias, output, uConst);
}

// ---- Conv DepthWise 2.0.4: c_p indexing, single-channel-per-thread ----
void mnn_corpus_conv_dw_204_fp32(const float* input, const half* kernel, const half* bias, float* output,
                                  float maxV, float minV, int iw, int ih, int c, int c_p,
                                  int ow, int oh, int kw, int kh, int dw, int dh, int sw, int sh, int pw, int ph,
                                  int total, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::CONV_DW_204<float><<<grid, block, 0, stream>>>(
        input, kernel, bias, output, maxV, minV, iw, ih, c, c_p, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total);
}

// ---- Transpose fp16 ----
void mnn_corpus_nhwc2nchw_fp16(const void* input, void* output, int total, int inside, int axis, int outside,
                                int grid, int block, cudaStream_t stream) {
    MNN::Corpus::NHWC_2_NCHW<half, half><<<grid, block, 0, stream>>>((const half*)input, (half*)output, total, inside, axis, outside);
}
void mnn_corpus_nchw2nhwc_fp16(const void* input, void* output, int total, int inside, int axis, int outside,
                                int grid, int block, cudaStream_t stream) {
    MNN::Corpus::NCHW_2_NHWC<half, half><<<grid, block, 0, stream>>>((const half*)input, (half*)output, total, inside, axis, outside);
}

// ---- 1.2.0 tag: Pool (NCHW layout, bc parameter) ----
void mnn_corpus_maxpool_120_fp32(const float* in, float* out, int bc, int ih, int iw, int oh, int ow,
                                 int padX, int padY, int kx, int ky, int sx, int sy,
                                 int grid, int block, cudaStream_t stream) {
    MNN::Corpus::maxpool_120<float><<<grid, block, 0, stream>>>(in, out, bc, ih, iw, oh, ow, padX, padY, kx, ky, sx, sy);
}
void mnn_corpus_avgpool_120_fp32(const float* in, float* out, int bc, int ih, int iw, int oh, int ow,
                                 int padX, int padY, int kx, int ky, int sx, int sy,
                                 int grid, int block, cudaStream_t stream) {
    MNN::Corpus::avgpool_120<float><<<grid, block, 0, stream>>>(in, out, bc, ih, iw, oh, ow, padX, padY, kx, ky, sx, sy);
}
// ---- 1.2.0 tag: Reduction (T accumulation, param order: inside, axis, outside) ----
void mnn_corpus_reduction_sum_120_fp32(const float* input, float* output, int inside, int axis, int outside,
                                       int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SUM_120<float><<<grid, block, 0, stream>>>(input, output, inside, axis, outside);
}
void mnn_corpus_reduction_mean_120_fp32(const float* input, float* output, int inside, int axis, int outside,
                                        int grid, int block, cudaStream_t stream) {
    MNN::Corpus::MEAN_120<float><<<grid, block, 0, stream>>>(input, output, inside, axis, outside);
}

// ---- 1.2.0 tag shims (other kernels) ----
void mnn_corpus_argmax_120_fp32(const int count, const int outside, const int inside, const int dim,
                                const float* input, float* output, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::ARGMAX_120<float><<<grid, block, 0, stream>>>(count, outside, inside, dim, input, output);
}
void mnn_corpus_scale_120_fp32(const int n, const int channels, const int dim, const float* in, float* out,
                                const float* scaleData, const float* biasData, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SCALE_120<float><<<grid, block, 0, stream>>>(n, channels, dim, in, out, scaleData, biasData);
}
void mnn_corpus_layernorm_120_fp32(const int count, const int outside, const int inside, const float epsilon,
                                   const float* in, float* out, const float* gamma, const float* beta,
                                   int grid, int block, cudaStream_t stream) {
    MNN::Corpus::LAYERNORM_120<float><<<grid, block, 0, stream>>>(count, outside, inside, epsilon, in, out, gamma, beta);
}
void mnn_corpus_prelu_120_fp32(const int n, const int channels, const int dim, const float* in, float* out,
                               const float* slopeData, int div_factor, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::PRELU_120<float><<<grid, block, 0, stream>>>(n, channels, dim, in, out, slopeData, div_factor);
}
void mnn_corpus_pack_c4_120_fp32(const float* input, float* output, int inside, int axis, int outside, int axisC4,
                                 int grid, int block, cudaStream_t stream) {
    MNN::Corpus::pack_c4_120<float><<<grid, block, 0, stream>>>(input, output, inside, axis, outside, axisC4);
}
void mnn_corpus_unpack_c4_120_fp32(const float* input, float* output, int inside, int axis, int outside, int axisC4,
                                   int grid, int block, cudaStream_t stream) {
    MNN::Corpus::unpack_c4_120<float><<<grid, block, 0, stream>>>(input, output, inside, axis, outside, axisC4);
}
void mnn_corpus_setzero_120_fp32(const int n, float* outputPtr, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::SETZERO_120<float><<<grid, block, 0, stream>>>(n, outputPtr);
}
void mnn_corpus_add_bias_120_fp32(float* input, float* output, const float* bias, int e, int h,
                                   int grid, int block, cudaStream_t stream) {
    MNN::Corpus::add_bias_120<float><<<grid, block, 0, stream>>>(input, output, bias, e, h);
}
void mnn_corpus_interp_120_fp32(const int n, int ih, int iw, int oh, int ow,
                                float sh, float sw, float ohf, float owf,
                                const float* in, float* out, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::INTERP_120<float><<<grid, block, 0, stream>>>(n, ih, iw, oh, ow, sh, sw, ohf, owf, in, out);
}
void mnn_corpus_interp_bilinear_120_fp32(const int n, int ih, int iw, int oh, int ow,
                                         float sh, float sw, float ohf, float owf,
                                         const float* in, float* out, int grid, int block, cudaStream_t stream) {
    MNN::Corpus::INTERP_BILINEAR_120<float><<<grid, block, 0, stream>>>(n, ih, iw, oh, ow, sh, sw, ohf, owf, in, out);
}

} // extern "C"
