//
//  FuseExecution.cpp
//  MNN
//
//  Created by MNN on 2023/06/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_CODEGEN_CUDA
#include "FuseExecution.hpp"
#include "FuseExecutionV2.hpp"
#endif
#include "UnaryExecution.hpp"
#include "BinaryExecution.hpp"
#include "MNNCUDADefine.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/TensorUtils.hpp"
#include "MNNCUDAFunction.cuh"
#include "core/KVMeta.hpp"
#include <chrono>
#ifdef MNN_CUDA_BENCH_OPS
#include <cublas_v2.h>
#endif
#include <cuda_fp16.h>
#include <float.h>
#ifdef MNN_CUDA_BENCH_OPS
#include <mma.h>
#endif
#include <sstream>
#include <string>
#include <vector>
#ifdef MNN_LOW_MEMORY
#include "weight_only_quant/ConvFpAIntBExecution.hpp"
#include "core/FileLoader.hpp"
#endif

namespace MNN {
namespace CUDA {

static int picSiluMulFeatureDim(const Tensor* output) {
    if (output == nullptr || output->dimensions() <= 0) {
        return 0;
    }
    if (output->dimensions() == 3) {
        return output->length(2);
    }
    return output->channel();
}

static int picSiluMulActiveRows(const Tensor* output) {
    const int channel = picSiluMulFeatureDim(output);
    const size_t count = CUDABackend::realSize(output);
    if (channel > 0 && count % static_cast<size_t>(channel) == 0) {
        return static_cast<int>(count / static_cast<size_t>(channel));
    }
    return static_cast<int>(count);
}

static bool usePicSiluMulFusedKernel(const Tensor* output) {
    constexpr int kMaxFusedRows = 8;
    constexpr int kMinChannel = 128;
    if (output == nullptr || output->dimensions() <= 0 || output->getType().code != halide_type_float) {
        return false;
    }
    return picSiluMulFeatureDim(output) >= kMinChannel && picSiluMulActiveRows(output) <= kMaxFusedRows;
}

__global__ void PicSiluMulFloatKernel(const float* gate, const float* up, float* output, size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        const float g = gate[i];
        const float u = up[i];
        const float silu = g > 87.0f ? g : (g < -87.0f ? 0.0f : g / (1.0f + __expf(-g)));
        output[i] = silu * u;
    }
}

__global__ void PicSiluMulHalf2Kernel(const half2* gate, const half2* up, half2* output, size_t count) {
    const half2 one = __float2half2_rn(1.0f);
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        const half2 g = gate[i];
        const half2 u = up[i];
        const half2 silu = __hmul2(g, __h2div(one, __hadd2(one, h2exp(__hneg2(g)))));
        output[i] = __hmul2(silu, u);
    }
}

__global__ void PicPackedSiluMulFloatKernel(const float* packed, float* output, int rows, int channels) {
    const int count = rows * channels;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        const int row = i / channels;
        const int c = i - row * channels;
        const int base = row * channels * 2;
        const float g = packed[base + c];
        const float u = packed[base + channels + c];
        const float silu = g > 87.0f ? g : (g < -87.0f ? 0.0f : g / (1.0f + __expf(-g)));
        output[i] = silu * u;
    }
}

__global__ void PicPackedSiluMulHalf2Kernel(const half2* packed, half2* output, int rows, int channelsHalf2) {
    const half2 one = __float2half2_rn(1.0f);
    const int count = rows * channelsHalf2;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        const int row = i / channelsHalf2;
        const int c = i - row * channelsHalf2;
        const int base = row * channelsHalf2 * 2;
        const half2 g = packed[base + c];
        const half2 u = packed[base + channelsHalf2 + c];
        const half2 silu = __hmul2(g, __h2div(one, __hadd2(one, h2exp(__hneg2(g)))));
        output[i] = __hmul2(silu, u);
    }
}

class PicSiluMulExecution : public Execution {
public:
    explicit PicSiluMulExecution(Backend* backend) : Execution(backend) {
        mFallbackSilu.reset(new UnaryExecution(UnaryOpOperation_SILU, backend));
        mFallbackMul.reset(new BinaryExecution(BinaryOpOperation_MUL, backend));
    }
    virtual ~PicSiluMulExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return INPUT_DATA_ERROR;
        }
        mUseHalf2Kernel = false;
        mUseFusedKernel = usePicSiluMulFusedKernel(outputs[0]);
        if (mUseFusedKernel && static_cast<CUDABackend*>(backend())->useFp16()) {
            const size_t count = CUDABackend::realSize(outputs[0]);
            mUseHalf2Kernel = (count % 2 == 0);
            mUseFusedKernel = mUseHalf2Kernel;
        }
        if (mUseFusedKernel) {
            mFallbackTemp.reset();
            return NO_ERROR;
        }
        mFallbackTemp.reset(Tensor::createDevice(outputs[0]->shape(), outputs[0]->getType(), outputs[0]->getDimensionType()));
        if (mFallbackTemp == nullptr || !backend()->onAcquireBuffer(mFallbackTemp.get(), Backend::DYNAMIC)) {
            return OUT_OF_MEMORY;
        }
        auto code = mFallbackSilu->onResize({inputs[0]}, {mFallbackTemp.get()});
        if (code != NO_ERROR) {
            backend()->onReleaseBuffer(mFallbackTemp.get(), Backend::DYNAMIC);
            return code;
        }
        code = mFallbackMul->onResize({mFallbackTemp.get(), inputs[1]}, outputs);
        if (code != NO_ERROR) {
            backend()->onReleaseBuffer(mFallbackTemp.get(), Backend::DYNAMIC);
            return code;
        }
        backend()->onReleaseBuffer(mFallbackTemp.get(), Backend::DYNAMIC);
        return NO_ERROR;
    }

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return INPUT_DATA_ERROR;
        }
        if (!mUseFusedKernel) {
            auto code = mFallbackSilu->onExecute({inputs[0]}, {mFallbackTemp.get()});
            if (code != NO_ERROR) {
                return code;
            }
            return mFallbackMul->onExecute({mFallbackTemp.get(), inputs[1]}, outputs);
        }
        auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
        const size_t count = CUDABackend::realSize(outputs[0]);
        if (count == 0) {
            return NO_ERROR;
        }
        const int threads = runtime->threads_num();
        if (mUseHalf2Kernel) {
            const size_t half2Count = count / 2;
            const int blocks = runtime->blocks_num(half2Count);
            PicSiluMulHalf2Kernel<<<blocks, threads>>>(
                reinterpret_cast<const half2*>(inputs[0]->deviceId()),
                reinterpret_cast<const half2*>(inputs[1]->deviceId()),
                reinterpret_cast<half2*>(outputs[0]->deviceId()), half2Count);
        } else {
            const int blocks = runtime->blocks_num(count);
            PicSiluMulFloatKernel<<<blocks, threads>>>(
                reinterpret_cast<const float*>(inputs[0]->deviceId()),
                reinterpret_cast<const float*>(inputs[1]->deviceId()),
                reinterpret_cast<float*>(outputs[0]->deviceId()), count);
        }
        checkKernelErrors;
        return NO_ERROR;
    }

private:
    bool mUseFusedKernel = true;
    bool mUseHalf2Kernel = false;
    std::shared_ptr<Tensor> mFallbackTemp;
    std::shared_ptr<Execution> mFallbackSilu;
    std::shared_ptr<Execution> mFallbackMul;
};

class PicPackedSiluMulExecution : public Execution {
public:
    explicit PicPackedSiluMulExecution(Backend* backend) : Execution(backend) {
    }
    virtual ~PicPackedSiluMulExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INPUT_DATA_ERROR;
        }
        mUseHalf2Kernel = false;
        mRows = picSiluMulActiveRows(outputs[0]);
        mChannels = picSiluMulFeatureDim(outputs[0]);
        const size_t inCount = CUDABackend::realSize(inputs[0]);
        const size_t outCount = CUDABackend::realSize(outputs[0]);
        if (mRows <= 0 || mChannels <= 0 || inCount != outCount * 2 || outputs[0]->getType().code != halide_type_float) {
            return NOT_SUPPORT;
        }
        if (static_cast<CUDABackend*>(backend())->useFp16()) {
            mUseHalf2Kernel = (mChannels % 2 == 0);
        }
        return NO_ERROR;
    }

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INPUT_DATA_ERROR;
        }
        auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
        const int threads = runtime->threads_num();
        if (mUseHalf2Kernel) {
            const int channelsHalf2 = mChannels / 2;
            const int count = mRows * channelsHalf2;
            if (count > 0) {
                PicPackedSiluMulHalf2Kernel<<<runtime->blocks_num(count), threads>>>(
                    reinterpret_cast<const half2*>(inputs[0]->deviceId()),
                    reinterpret_cast<half2*>(outputs[0]->deviceId()),
                    mRows, channelsHalf2);
            }
        } else {
            const int count = mRows * mChannels;
            if (count > 0) {
                PicPackedSiluMulFloatKernel<<<runtime->blocks_num(count), threads>>>(
                    reinterpret_cast<const float*>(inputs[0]->deviceId()),
                    reinterpret_cast<float*>(outputs[0]->deviceId()),
                    mRows, mChannels);
            }
        }
        checkKernelErrors;
        return NO_ERROR;
    }

private:
    bool mUseHalf2Kernel = false;
    int mRows = 0;
    int mChannels = 0;
};

#ifdef MNN_CUDA_BENCH_OPS
// Test-only direct-op probes for fused MLP/down experiments. These Extra types are
// compiled only for MNN_BUILD_TEST and must not be emitted by exporters or graph rewrites.
static int picBenchFeatureDim(const Tensor* tensor) {
    if (tensor == nullptr || tensor->dimensions() <= 0) {
        return 0;
    }
    if (tensor->dimensions() == 4 && tensor->length(1) == 1 && tensor->length(2) == 1) {
        return tensor->length(3);
    }
    return tensor->channel();
}

static bool runPicBenchDownFp16(cublasHandle_t handle, const void* weight, const void* input, void* output,
                                int rows, int ic, int icp, int oc, int ocp) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
#if CUDART_VERSION >= 11000
    const auto computeType = CUBLAS_COMPUTE_32F;
#else
    const auto computeType = CUDA_R_32F;
#endif
#if CUDART_VERSION >= 9000
    const auto algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
#else
    const auto algo = CUBLAS_GEMM_DEFAULT;
#endif
    return cublasGemmEx(handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        oc, rows, ic,
                        &alpha,
                        weight, CUDA_R_16F, icp,
                        input, CUDA_R_16F, icp,
                        &beta,
                        output, CUDA_R_16F, ocp,
                        computeType, algo) == CUBLAS_STATUS_SUCCESS;
}

template<int COLS, int GROUP, int KTILE>
__global__ void __launch_bounds__(COLS * GROUP, 2) PicStreamedPackedSiluDownKernel(
    const half* __restrict__ packed,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int rows, int inter, int interP, int hidden, int hiddenP
) {
    const int colBase = blockIdx.x * COLS;
    const int row = blockIdx.y;
    if (row >= rows) {
        return;
    }
    const int tid = threadIdx.x;
    const int colInTile = tid / GROUP;
    const int lane = tid - colInTile * GROUP;
    const int col = colBase + colInTile;
    __shared__ half act[KTILE];

    float acc = 0.0f;
    const half* rowPacked = packed + static_cast<size_t>(row) * static_cast<size_t>(inter) * 2;
    for (int kBase = 0; kBase < inter; kBase += KTILE) {
        if (tid < KTILE) {
            const int k = kBase + tid;
            half v = __float2half(0.0f);
            if (k < inter) {
                const float g = __half2float(rowPacked[k]);
                const float u = __half2float(rowPacked[inter + k]);
                const float silu = g > 87.0f ? g : (g < -87.0f ? 0.0f : g / (1.0f + __expf(-g)));
                v = __float2half_rn(silu * u);
            }
            act[tid] = v;
        }
        __syncthreads();
        if (col < hidden) {
            const half* w = weight + static_cast<size_t>(col) * static_cast<size_t>(interP) + kBase;
            #pragma unroll
            for (int kk = lane; kk < KTILE; kk += GROUP) {
                const int k = kBase + kk;
                if (k < inter) {
                    acc = fmaf(__half2float(act[kk]), __half2float(w[kk]), acc);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int offset = GROUP / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset, GROUP);
    }
    if (lane == 0 && col < hidden) {
        output[static_cast<size_t>(row) * static_cast<size_t>(hiddenP) + col] = __float2half_rn(acc);
    }
}

// Direct-op bench candidates only. Do not route exporter or production graphs to these PicBench* Extra types.
class PicBenchFusedPackedSiluDownExecution : public Execution {
public:
    PicBenchFusedPackedSiluDownExecution(Backend* backend, bool streamed) : Execution(backend), mStreamed(streamed) {
    }
    virtual ~PicBenchFusedPackedSiluDownExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() != 2 || outputs.size() != 1) {
            return INPUT_DATA_ERROR;
        }
        if (!static_cast<CUDABackend*>(backend())->useFp16()) {
            return NOT_SUPPORT;
        }
        mRows = outputs[0]->length(0);
        mHidden = picBenchFeatureDim(outputs[0]);
        const size_t packedCount = CUDABackend::realSize(inputs[0]);
        if (mRows <= 0 || mHidden <= 0 || packedCount == 0 ||
            packedCount % static_cast<size_t>(mRows * 2) != 0) {
            return NOT_SUPPORT;
        }
        mInter = static_cast<int>(packedCount / static_cast<size_t>(mRows * 2));
        mInterP = UP_DIV(mInter, 8) * 8;
        mHiddenP = UP_DIV(mHidden, 8) * 8;
        if (mInter <= 0 || mInter != mInterP || mHidden != mHiddenP ||
            CUDABackend::realSize(outputs[0]) < static_cast<size_t>(mRows) * static_cast<size_t>(mHiddenP) ||
            CUDABackend::realSize(inputs[1]) < static_cast<size_t>(mHiddenP) * static_cast<size_t>(mInterP)) {
            return NOT_SUPPORT;
        }
        if (!mStreamed) {
            mActivation.reset(Tensor::createDevice<float>({mRows, mInter, 1, 1}, Tensor::CAFFE));
            if (mActivation == nullptr || !backend()->onAcquireBuffer(mActivation.get(), Backend::DYNAMIC)) {
                return OUT_OF_MEMORY;
            }
            backend()->onReleaseBuffer(mActivation.get(), Backend::DYNAMIC);
        } else {
            mActivation.reset();
        }
        return NO_ERROR;
    }

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() != 2 || outputs.size() != 1) {
            return INPUT_DATA_ERROR;
        }
        auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
        auto* packed = reinterpret_cast<const half*>(inputs[0]->deviceId());
        auto* weight = reinterpret_cast<const half*>(inputs[1]->deviceId());
        auto* output = reinterpret_cast<half*>(outputs[0]->deviceId());
        if (mStreamed) {
            constexpr int kCols = 32;
            constexpr int kGroup = 8;
            constexpr int kTile = 128;
            dim3 grid(UP_DIV(mHidden, kCols), mRows);
            dim3 block(kCols * kGroup);
            PicStreamedPackedSiluDownKernel<kCols, kGroup, kTile><<<grid, block>>>(
                packed, weight, output, mRows, mInter, mInterP, mHidden, mHiddenP);
            checkKernelErrors;
            return NO_ERROR;
        }

        const int channelsHalf2 = mInter / 2;
        const int count = mRows * channelsHalf2;
        if (count > 0) {
            PicPackedSiluMulHalf2Kernel<<<runtime->blocks_num(count), runtime->threads_num()>>>(
                reinterpret_cast<const half2*>(packed),
                reinterpret_cast<half2*>(mActivation->deviceId()),
                mRows, channelsHalf2);
            checkKernelErrors;
        }
        if (!runPicBenchDownFp16(runtime->cublasHandle(), weight, reinterpret_cast<const void*>(mActivation->deviceId()),
                                 output, mRows, mInter, mInterP, mHidden, mHiddenP)) {
            return INVALID_VALUE;
        }
        return NO_ERROR;
    }

private:
    bool mStreamed = false;
    int mRows = 0;
    int mInter = 0;
    int mInterP = 0;
    int mHidden = 0;
    int mHiddenP = 0;
    std::shared_ptr<Tensor> mActivation;
};

template<int WARPS>
__global__ void __launch_bounds__(WARPS * 32, 2) PicWmmaPackedSiluDownKernel(
    const half* __restrict__ packed,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int rows, int inter, int interP, int hidden, int hiddenP
) {
    constexpr int kM = 16;
    constexpr int kN = 16;
    constexpr int kK = 16;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nBase = blockIdx.x * WARPS * kN + warp * kN;
    __shared__ half aTile[kM * kK];
    __shared__ float cTile[WARPS * kM * kN];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kM, kN, kK, half, nvcuda::wmma::row_major> aFrag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kM, kN, kK, half, nvcuda::wmma::col_major> bFrag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kM, kN, kK, float> cFrag;
    nvcuda::wmma::fill_fragment(cFrag, 0.0f);

    for (int kBase = 0; kBase < inter; kBase += kK) {
        for (int idx = threadIdx.x; idx < kM * kK; idx += blockDim.x) {
            const int row = idx / kK;
            const int kk = idx - row * kK;
            half value = __float2half(0.0f);
            if (row < rows && kBase + kk < inter) {
                const half* rowPacked = packed + static_cast<size_t>(row) * static_cast<size_t>(inter) * 2;
                const float gate = __half2float(rowPacked[kBase + kk]);
                const float up = __half2float(rowPacked[inter + kBase + kk]);
                const float silu = gate > 87.0f ? gate : (gate < -87.0f ? 0.0f : gate / (1.0f + __expf(-gate)));
                value = __float2half_rn(silu * up);
            }
            aTile[idx] = value;
        }
        __syncthreads();

        if (nBase < hidden) {
            nvcuda::wmma::load_matrix_sync(aFrag, aTile, kK);
            nvcuda::wmma::load_matrix_sync(bFrag,
                                           weight + static_cast<size_t>(nBase) * static_cast<size_t>(interP) + kBase,
                                           interP);
            nvcuda::wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }
        __syncthreads();
    }

    if (nBase < hidden) {
        float* cOut = cTile + warp * kM * kN;
        nvcuda::wmma::store_matrix_sync(cOut, cFrag, kN, nvcuda::wmma::mem_row_major);
        for (int idx = lane; idx < kM * kN; idx += 32) {
            const int row = idx / kN;
            const int col = idx - row * kN;
            if (row < rows && nBase + col < hidden) {
                output[static_cast<size_t>(row) * static_cast<size_t>(hiddenP) + nBase + col] =
                    __float2half_rn(cOut[idx]);
            }
        }
    }
}

class PicBenchWmmaPackedSiluDownExecution : public Execution {
public:
    explicit PicBenchWmmaPackedSiluDownExecution(Backend* backend) : Execution(backend) {
    }
    virtual ~PicBenchWmmaPackedSiluDownExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() != 2 || outputs.size() != 1) {
            return INPUT_DATA_ERROR;
        }
        if (!static_cast<CUDABackend*>(backend())->useFp16()) {
            return NOT_SUPPORT;
        }
        mRows = outputs[0]->length(0);
        mHidden = picBenchFeatureDim(outputs[0]);
        const size_t packedCount = CUDABackend::realSize(inputs[0]);
        if (mRows <= 0 || mRows > 16 || mHidden <= 0 || packedCount == 0 ||
            packedCount % static_cast<size_t>(mRows * 2) != 0) {
            return NOT_SUPPORT;
        }
        mInter = static_cast<int>(packedCount / static_cast<size_t>(mRows * 2));
        mInterP = UP_DIV(mInter, 16) * 16;
        mHiddenP = UP_DIV(mHidden, 16) * 16;
        if (mInter <= 0 || mInter != mInterP || mHidden != mHiddenP ||
            CUDABackend::realSize(outputs[0]) < static_cast<size_t>(mRows) * static_cast<size_t>(mHiddenP) ||
            CUDABackend::realSize(inputs[1]) < static_cast<size_t>(mHiddenP) * static_cast<size_t>(mInterP)) {
            return NOT_SUPPORT;
        }
        return NO_ERROR;
    }

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() != 2 || outputs.size() != 1) {
            return INPUT_DATA_ERROR;
        }
        constexpr int kWarps = 8;
        dim3 grid(UP_DIV(mHidden, 16 * kWarps));
        dim3 block(32 * kWarps);
        PicWmmaPackedSiluDownKernel<kWarps><<<grid, block>>>(
            reinterpret_cast<const half*>(inputs[0]->deviceId()),
            reinterpret_cast<const half*>(inputs[1]->deviceId()),
            reinterpret_cast<half*>(outputs[0]->deviceId()),
            mRows, mInter, mInterP, mHidden, mHiddenP);
        checkKernelErrors;
        return NO_ERROR;
    }

private:
    int mRows = 0;
    int mInter = 0;
    int mInterP = 0;
    int mHidden = 0;
    int mHiddenP = 0;
};
#endif

#ifdef MNN_LOW_MEMORY
struct PicGateUpConvSpec {
    std::string name;
    std::vector<int64_t> external;
    int ic = 0;
    int oc = 0;
    int quantBit = 4;
    int quantBlock = 64;
    int aMin = 1;
    int readType = 0;
    bool shapeInt32 = false;
    bool hasBias = false;
};

static const Attribute* findPicGateUpAttr(const Extra* extra, const char* key) {
    if (extra == nullptr || extra->attr() == nullptr || key == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr != nullptr && attr->key() != nullptr && attr->key()->str() == key) {
            return attr;
        }
    }
    return nullptr;
}

static std::string picGateUpAttrString(const Extra* extra, const std::string& key) {
    auto attr = findPicGateUpAttr(extra, key.c_str());
    if (attr == nullptr || attr->s() == nullptr) {
        return "";
    }
    return attr->s()->str();
}

static int picGateUpAttrInt(const Extra* extra, const std::string& key, int fallback = 0) {
    auto attr = findPicGateUpAttr(extra, key.c_str());
    if (attr == nullptr) {
        return fallback;
    }
    return attr->i();
}

static std::vector<int64_t> parsePicGateUpExternal(const std::string& value) {
    std::vector<int64_t> result;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            result.emplace_back(static_cast<int64_t>(std::stoll(item)));
        }
    }
    while (result.size() < 5) {
        result.emplace_back(0);
    }
    return result;
}

static PicGateUpConvSpec parsePicGateUpConvSpec(const Extra* extra, const std::string& prefix) {
    PicGateUpConvSpec spec;
    spec.name = picGateUpAttrString(extra, prefix + "_name");
    spec.external = parsePicGateUpExternal(picGateUpAttrString(extra, prefix + "_external"));
    spec.ic = picGateUpAttrInt(extra, prefix + "_in_features");
    spec.oc = picGateUpAttrInt(extra, prefix + "_out_features");
    spec.quantBit = picGateUpAttrInt(extra, prefix + "_quant_bit", 4);
    spec.quantBlock = picGateUpAttrInt(extra, prefix + "_quant_block", 64);
    spec.aMin = picGateUpAttrInt(extra, prefix + "_a_min", 1);
    spec.readType = picGateUpAttrInt(extra, prefix + "_read_type", 0);
    spec.shapeInt32 = picGateUpAttrInt(extra, prefix + "_shape_int32", 0) != 0;
    spec.hasBias = picGateUpAttrInt(extra, prefix + "_has_bias", 0) != 0;
    return spec;
}

static PicGateUpConvSpec parsePicLinearNhwcSpec(const Extra* extra) {
    PicGateUpConvSpec spec;
    spec.name = picGateUpAttrString(extra, "linear_name");
    spec.external = parsePicGateUpExternal(picGateUpAttrString(extra, "external"));
    spec.ic = picGateUpAttrInt(extra, "in_features");
    spec.oc = picGateUpAttrInt(extra, "out_features");
    spec.quantBit = picGateUpAttrInt(extra, "quant_bit", 4);
    spec.quantBlock = picGateUpAttrInt(extra, "quant_block", 64);
    spec.aMin = picGateUpAttrInt(extra, "a_min", 1);
    spec.readType = picGateUpAttrInt(extra, "read_type", 0);
    spec.shapeInt32 = picGateUpAttrInt(extra, "shape_int32", 0) != 0;
    spec.hasBias = picGateUpAttrInt(extra, "has_bias", 0) != 0;
    return spec;
}

static bool profilePicLinearNhwc() {
    const char* paged = ::getenv("MNN_PAGED_ATTENTION_PROFILE");
    if (paged != nullptr && paged[0] != '\0' && paged[0] != '0') {
        return true;
    }
    const char* graph = ::getenv("MNN_PIC_GRAPH_PROFILE");
    return graph != nullptr && graph[0] != '\0' && graph[0] != '0';
}

static std::vector<uint8_t> buildPicGateUpChildConvOp(const PicGateUpConvSpec& spec, const char* externalPath) {
    flatbuffers::FlatBufferBuilder builder;
    Convolution2DT conv;
    conv.common.reset(new Convolution2DCommonT);
    conv.common->padX = 0;
    conv.common->padY = 0;
    conv.common->kernelX = 1;
    conv.common->kernelY = 1;
    conv.common->strideX = 1;
    conv.common->strideY = 1;
    conv.common->dilateX = 1;
    conv.common->dilateY = 1;
    conv.common->padMode = PadMode_CAFFE;
    conv.common->group = 1;
    conv.common->outputCount = spec.oc;
    conv.common->inputCount = spec.ic;
    conv.common->relu = false;
    conv.common->relu6 = false;
    conv.common->hasOutputShape = false;
    conv.quanParameter.reset(new IDSTQuanT);
    conv.quanParameter->type = spec.quantBit == 16 ? 3 : 1;
    conv.quanParameter->useInt32 = false;
    conv.quanParameter->quantScale = 1.0f;
    conv.quanParameter->scaleIn = 0.0f;
    conv.quanParameter->scaleOut = 0.0f;
    conv.quanParameter->aMaxOrBits = spec.quantBit;
    conv.quanParameter->aMin = spec.aMin;
    conv.quanParameter->readType = spec.readType;
    conv.quanParameter->has_scaleInt = false;
    conv.quanParameter->shapeInt32 = spec.shapeInt32;
    conv.quanParameter->weightSize = 0;
    conv.external = spec.external;
    conv.bias.resize(std::max(0, spec.oc), 0.0f);
    if (externalPath != nullptr && spec.external.size() > 3 && spec.external[3] > 0) {
        FileLoader loader(externalPath);
        if (loader.valid()) {
            loader.offset(spec.external[0] + spec.external[1] + spec.external[2]);
            loader.read(reinterpret_cast<char*>(conv.bias.data()), spec.external[3]);
        }
    }
    auto convOffset = Convolution2D::Pack(builder, &conv);
    auto nameOffset = builder.CreateString(spec.name);
    flatbuffers::Offset<flatbuffers::String> externalPathOffset = 0;
    if (externalPath != nullptr) {
        externalPathOffset = builder.CreateString(externalPath);
    }
    OpBuilder opBuilder(builder);
    opBuilder.add_name(nameOffset);
    opBuilder.add_main(convOffset.Union());
    opBuilder.add_main_type(OpParameter_Convolution2D);
    opBuilder.add_type(OpType_Convolution);
    opBuilder.add_defaultDimentionFormat(MNN_DATA_FORMAT_NHWC);
    opBuilder.add_externalPath(externalPathOffset);
    builder.Finish(opBuilder.Finish());
    const uint8_t* ptr = builder.GetBufferPointer();
    return std::vector<uint8_t>(ptr, ptr + builder.GetSize());
}

template<typename T, int OC_PER_BLK, int MAX_BATCH>
__global__ void __launch_bounds__(128, 6) PicGateUpInt4V14MBKernel(
    const T* __restrict__ input,
    const uint8_t* __restrict__ gateKernel,
    const float2* __restrict__ gateParams,
    const T* __restrict__ gateBias,
    T* __restrict__ gateOutput,
    const uint8_t* __restrict__ upKernel,
    const float2* __restrict__ upParams,
    const T* __restrict__ upBias,
    T* __restrict__ upOutput,
    const float maxV, const float minV,
    const int batch, const int ic, const int icp,
    const int oc, const int ocp, const int numQg
) {
    const int ocBase = blockIdx.x * OC_PER_BLK;
    if (ocBase >= oc) {
        return;
    }
    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int nwarps = blockDim.x / 32;
    const int icPerGroup = ic / numQg;
    const int rowBytes = icp / 2;

    float gateAcc[MAX_BATCH * OC_PER_BLK];
    float upAcc[MAX_BATCH * OC_PER_BLK];
    #pragma unroll
    for (int i = 0; i < MAX_BATCH * OC_PER_BLK; ++i) {
        gateAcc[i] = 0.0f;
        upAcc[i] = 0.0f;
    }

    const int stride = blockDim.x * 8;
    for (int k = tid * 8; k < ic; k += stride) {
        const int gidx = k / icPerGroup;
        uint32_t gateW[OC_PER_BLK];
        uint32_t upW[OC_PER_BLK];
        float gateS[OC_PER_BLK], gateAdj[OC_PER_BLK];
        float upS[OC_PER_BLK], upAdj[OC_PER_BLK];
        #pragma unroll
        for (int oi = 0; oi < OC_PER_BLK; ++oi) {
            const int ocIdx = ocBase + oi;
            if (ocIdx < oc) {
                gateW[oi] = __ldg(reinterpret_cast<const uint32_t*>(
                    gateKernel + ocIdx * rowBytes + k / 2));
                upW[oi] = __ldg(reinterpret_cast<const uint32_t*>(
                    upKernel + ocIdx * rowBytes + k / 2));
                const float2 gp = __ldg(gateParams + ocIdx * numQg + gidx);
                const float2 up = __ldg(upParams + ocIdx * numQg + gidx);
                gateS[oi] = gp.x;
                gateAdj[oi] = gp.y;
                upS[oi] = up.x;
                upAdj[oi] = up.y;
            }
        }

        #pragma unroll
        for (int b = 0; b < MAX_BATCH; ++b) {
            if (b >= batch) {
                break;
            }
            const T* inPtr = input + b * icp;
            float inVals[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                inVals[j] = (float)inPtr[k + j];
            }
            const float inputSum = inVals[0] + inVals[1] + inVals[2] + inVals[3] +
                                   inVals[4] + inVals[5] + inVals[6] + inVals[7];
            #pragma unroll
            for (int oi = 0; oi < OC_PER_BLK; ++oi) {
                if (ocBase + oi >= oc) {
                    break;
                }
                float gateRaw = 0.0f;
                float upRaw = 0.0f;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t gatePacked = (gateW[oi] >> (j * 8)) & 0xFF;
                    const uint32_t upPacked = (upW[oi] >> (j * 8)) & 0xFF;
                    gateRaw += inVals[j * 2] * (float)(gatePacked >> 4);
                    gateRaw += inVals[j * 2 + 1] * (float)(gatePacked & 0x0F);
                    upRaw += inVals[j * 2] * (float)(upPacked >> 4);
                    upRaw += inVals[j * 2 + 1] * (float)(upPacked & 0x0F);
                }
                const int accIdx = b * OC_PER_BLK + oi;
                gateAcc[accIdx] = fmaf(gateAdj[oi], inputSum, fmaf(gateS[oi], gateRaw, gateAcc[accIdx]));
                upAcc[accIdx] = fmaf(upAdj[oi], inputSum, fmaf(upS[oi], upRaw, upAcc[accIdx]));
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < MAX_BATCH * OC_PER_BLK; ++i) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            gateAcc[i] += __shfl_xor_sync(0xffffffff, gateAcc[i], off);
            upAcc[i] += __shfl_xor_sync(0xffffffff, upAcc[i], off);
        }
    }

    __shared__ float gateSmem[MAX_BATCH * OC_PER_BLK][4];
    __shared__ float upSmem[MAX_BATCH * OC_PER_BLK][4];
    if (laneId == 0) {
        #pragma unroll
        for (int i = 0; i < MAX_BATCH * OC_PER_BLK; ++i) {
            gateSmem[i][warpId] = gateAcc[i];
            upSmem[i][warpId] = upAcc[i];
        }
    }
    __syncthreads();

    const int reduceCount = batch * OC_PER_BLK;
    if (tid < reduceCount) {
        const int b = tid / OC_PER_BLK;
        const int oi = tid % OC_PER_BLK;
        if (ocBase + oi < oc) {
            float gateResult = 0.0f;
            float upResult = 0.0f;
            for (int w = 0; w < nwarps; ++w) {
                gateResult += gateSmem[tid][w];
                upResult += upSmem[tid][w];
            }
            gateResult += (float)__ldg(&gateBias[ocBase + oi]);
            upResult += (float)__ldg(&upBias[ocBase + oi]);
            gateResult = fmaxf(fminf(gateResult, maxV), minV);
            upResult = fmaxf(fminf(upResult, maxV), minV);
            gateOutput[b * ocp + ocBase + oi] = (T)gateResult;
            upOutput[b * ocp + ocBase + oi] = (T)upResult;
        }
    }
}

class PicGateUpWeightOnlyExecution : public Execution {
public:
    PicGateUpWeightOnlyExecution(const Op* op, Backend* backend) : Execution(backend) {
        auto extra = op->main_as_Extra();
        mGateSpec = parsePicGateUpConvSpec(extra, "gate");
        mUpSpec = parsePicGateUpConvSpec(extra, "up");
        const char* externalPath = op->externalPath() != nullptr ? op->externalPath()->c_str() : nullptr;
        if (mGateSpec.ic <= 0 || mGateSpec.oc <= 0 || mUpSpec.ic != mGateSpec.ic || mUpSpec.oc != mGateSpec.oc) {
            mValid = false;
            return;
        }
        mGateOpBuffer = buildPicGateUpChildConvOp(mGateSpec, externalPath);
        mUpOpBuffer = buildPicGateUpChildConvOp(mUpSpec, externalPath);
        mGateOp = flatbuffers::GetRoot<Op>(mGateOpBuffer.data());
        mUpOp = flatbuffers::GetRoot<Op>(mUpOpBuffer.data());
        mGateResource.reset(new ConvFpAIntBExecution::Resource(backend, mGateOp));
        mUpResource.reset(new ConvFpAIntBExecution::Resource(backend, mUpOp));
        mGateConv.reset(new ConvFpAIntBExecution(backend, mGateOp, mGateResource));
        mUpConv.reset(new ConvFpAIntBExecution(backend, mUpOp, mUpResource));
    }

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (!mValid || inputs.size() != 1 || outputs.size() != 2) {
            return INPUT_DATA_ERROR;
        }
        auto code = mGateConv->onResize(inputs, {outputs[0]});
        if (code != NO_ERROR) {
            return code;
        }
        return mUpConv->onResize(inputs, {outputs[1]});
    }

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (!mValid || inputs.size() != 1 || outputs.size() != 2) {
            return INPUT_DATA_ERROR;
        }
        auto meta = static_cast<KVMeta*>(static_cast<CUDABackend*>(backend())->getMetaPtr());
        const bool decodeRepairActive = meta != nullptr && meta->pic_decode_repair_sparse_active;
        if (canUseScalarFused(inputs, outputs, decodeRepairActive)) {
            launchFused(inputs[0], outputs[0], outputs[1]);
            return NO_ERROR;
        }
        auto code = mGateConv->onExecute(inputs, {outputs[0]});
        if (code != NO_ERROR) {
            return code;
        }
        return mUpConv->onExecute(inputs, {outputs[1]});
    }

private:
    bool canUseScalarFused(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           bool decodeRepairActive) const {
        if (!decodeRepairActive) {
            return false;
        }
        if (!mGateResource->mIsWeightInt4 || !mUpResource->mIsWeightInt4 ||
            mGateResource->mGemvParams == nullptr || mUpResource->mGemvParams == nullptr) {
            return false;
        }
        if (!static_cast<CUDABackend*>(backend())->useFp16()) {
            return false;
        }
        const int batch = inputs[0]->batch();
        if (batch < 2 || batch > 8) {
            return false;
        }
        return outputs[0]->channel() == mGateSpec.oc && outputs[1]->channel() == mUpSpec.oc;
    }

    void launchFused(const Tensor* input, Tensor* gateOutput, Tensor* upOutput) {
        auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
        constexpr int kOcPerBlock = 2;
        const int batch = input->batch();
        const int ic = mGateSpec.ic;
        const int icp = UP_DIV(ic, 8) * 8;
        const int oc = mGateSpec.oc;
        const int ocp = UP_DIV(oc, 8) * 8;
        const int numQg = (mGateResource->mQuanC > 0) ? (mGateResource->mQuanC / oc) : 1;
        dim3 grid((oc + kOcPerBlock - 1) / kOcPerBlock);
        dim3 block(128);
        if (batch <= 3) {
            PicGateUpInt4V14MBKernel<half, kOcPerBlock, 3><<<grid, block>>>(
                reinterpret_cast<const half*>(input->deviceId()),
                reinterpret_cast<const uint8_t*>(mGateResource->mFilter), mGateResource->mGemvParams,
                reinterpret_cast<const half*>(mGateResource->mBias), reinterpret_cast<half*>(gateOutput->deviceId()),
                reinterpret_cast<const uint8_t*>(mUpResource->mFilter), mUpResource->mGemvParams,
                reinterpret_cast<const half*>(mUpResource->mBias), reinterpret_cast<half*>(upOutput->deviceId()),
                FLT_MAX, -FLT_MAX, batch, ic, icp, oc, ocp, numQg);
        } else {
            PicGateUpInt4V14MBKernel<half, kOcPerBlock, 8><<<grid, block>>>(
                reinterpret_cast<const half*>(input->deviceId()),
                reinterpret_cast<const uint8_t*>(mGateResource->mFilter), mGateResource->mGemvParams,
                reinterpret_cast<const half*>(mGateResource->mBias), reinterpret_cast<half*>(gateOutput->deviceId()),
                reinterpret_cast<const uint8_t*>(mUpResource->mFilter), mUpResource->mGemvParams,
                reinterpret_cast<const half*>(mUpResource->mBias), reinterpret_cast<half*>(upOutput->deviceId()),
                FLT_MAX, -FLT_MAX, batch, ic, icp, oc, ocp, numQg);
        }
        checkKernelErrors;
    }

    bool mValid = true;
    PicGateUpConvSpec mGateSpec;
    PicGateUpConvSpec mUpSpec;
    std::vector<uint8_t> mGateOpBuffer;
    std::vector<uint8_t> mUpOpBuffer;
    const Op* mGateOp = nullptr;
    const Op* mUpOp = nullptr;
    std::shared_ptr<ConvFpAIntBExecution::Resource> mGateResource;
    std::shared_ptr<ConvFpAIntBExecution::Resource> mUpResource;
    std::shared_ptr<ConvFpAIntBExecution> mGateConv;
    std::shared_ptr<ConvFpAIntBExecution> mUpConv;
};

template<typename T, int OC_PER_BLK, int MAX_BATCH>
__global__ void __launch_bounds__(128, 6) PicGateUpSiluInt4V14MBKernel(
    const T* __restrict__ input,
    const uint8_t* __restrict__ gateKernel,
    const float2* __restrict__ gateParams,
    const T* __restrict__ gateBias,
    const uint8_t* __restrict__ upKernel,
    const float2* __restrict__ upParams,
    const T* __restrict__ upBias,
    T* __restrict__ output,
    const float maxV, const float minV,
    const int batch, const int ic, const int icp,
    const int oc, const int ocp, const int numQg
) {
    const int ocBase = blockIdx.x * OC_PER_BLK;
    if (ocBase >= oc) {
        return;
    }
    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int nwarps = blockDim.x / 32;
    const int icPerGroup = ic / numQg;
    const int rowBytes = icp / 2;

    float gateAcc[MAX_BATCH * OC_PER_BLK];
    float upAcc[MAX_BATCH * OC_PER_BLK];
    #pragma unroll
    for (int i = 0; i < MAX_BATCH * OC_PER_BLK; ++i) {
        gateAcc[i] = 0.0f;
        upAcc[i] = 0.0f;
    }

    const int stride = blockDim.x * 8;
    for (int k = tid * 8; k < ic; k += stride) {
        const int gidx = k / icPerGroup;
        uint32_t gateW[OC_PER_BLK];
        uint32_t upW[OC_PER_BLK];
        float gateS[OC_PER_BLK], gateAdj[OC_PER_BLK];
        float upS[OC_PER_BLK], upAdj[OC_PER_BLK];
        #pragma unroll
        for (int oi = 0; oi < OC_PER_BLK; ++oi) {
            const int ocIdx = ocBase + oi;
            if (ocIdx < oc) {
                gateW[oi] = __ldg(reinterpret_cast<const uint32_t*>(
                    gateKernel + ocIdx * rowBytes + k / 2));
                upW[oi] = __ldg(reinterpret_cast<const uint32_t*>(
                    upKernel + ocIdx * rowBytes + k / 2));
                const float2 gp = __ldg(gateParams + ocIdx * numQg + gidx);
                const float2 up = __ldg(upParams + ocIdx * numQg + gidx);
                gateS[oi] = gp.x;
                gateAdj[oi] = gp.y;
                upS[oi] = up.x;
                upAdj[oi] = up.y;
            }
        }

        #pragma unroll
        for (int b = 0; b < MAX_BATCH; ++b) {
            if (b >= batch) {
                break;
            }
            const T* inPtr = input + b * icp;
            float inVals[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                inVals[j] = (float)inPtr[k + j];
            }
            const float inputSum = inVals[0] + inVals[1] + inVals[2] + inVals[3] +
                                   inVals[4] + inVals[5] + inVals[6] + inVals[7];
            #pragma unroll
            for (int oi = 0; oi < OC_PER_BLK; ++oi) {
                if (ocBase + oi >= oc) {
                    break;
                }
                float gateRaw = 0.0f;
                float upRaw = 0.0f;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t gatePacked = (gateW[oi] >> (j * 8)) & 0xFF;
                    const uint32_t upPacked = (upW[oi] >> (j * 8)) & 0xFF;
                    gateRaw += inVals[j * 2] * (float)(gatePacked >> 4);
                    gateRaw += inVals[j * 2 + 1] * (float)(gatePacked & 0x0F);
                    upRaw += inVals[j * 2] * (float)(upPacked >> 4);
                    upRaw += inVals[j * 2 + 1] * (float)(upPacked & 0x0F);
                }
                const int accIdx = b * OC_PER_BLK + oi;
                gateAcc[accIdx] = fmaf(gateAdj[oi], inputSum, fmaf(gateS[oi], gateRaw, gateAcc[accIdx]));
                upAcc[accIdx] = fmaf(upAdj[oi], inputSum, fmaf(upS[oi], upRaw, upAcc[accIdx]));
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < MAX_BATCH * OC_PER_BLK; ++i) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            gateAcc[i] += __shfl_xor_sync(0xffffffff, gateAcc[i], off);
            upAcc[i] += __shfl_xor_sync(0xffffffff, upAcc[i], off);
        }
    }

    __shared__ float gateSmem[MAX_BATCH * OC_PER_BLK][4];
    __shared__ float upSmem[MAX_BATCH * OC_PER_BLK][4];
    if (laneId == 0) {
        #pragma unroll
        for (int i = 0; i < MAX_BATCH * OC_PER_BLK; ++i) {
            gateSmem[i][warpId] = gateAcc[i];
            upSmem[i][warpId] = upAcc[i];
        }
    }
    __syncthreads();

    const int reduceCount = batch * OC_PER_BLK;
    if (tid < reduceCount) {
        const int b = tid / OC_PER_BLK;
        const int oi = tid % OC_PER_BLK;
        if (ocBase + oi < oc) {
            float gateResult = 0.0f;
            float upResult = 0.0f;
            for (int w = 0; w < nwarps; ++w) {
                gateResult += gateSmem[tid][w];
                upResult += upSmem[tid][w];
            }
            gateResult += (float)__ldg(&gateBias[ocBase + oi]);
            upResult += (float)__ldg(&upBias[ocBase + oi]);
            gateResult = fmaxf(fminf(gateResult, maxV), minV);
            upResult = fmaxf(fminf(upResult, maxV), minV);
            const float silu = gateResult > 87.0f ? gateResult :
                (gateResult < -87.0f ? 0.0f : gateResult / (1.0f + __expf(-gateResult)));
            output[b * ocp + ocBase + oi] = (T)(silu * upResult);
        }
    }
}

class PicGateUpSiluWeightOnlyExecution : public Execution {
public:
    PicGateUpSiluWeightOnlyExecution(const Op* op, Backend* backend) : Execution(backend) {
        auto extra = op->main_as_Extra();
        mGateSpec = parsePicGateUpConvSpec(extra, "gate");
        mUpSpec = parsePicGateUpConvSpec(extra, "up");
        const char* externalPath = op->externalPath() != nullptr ? op->externalPath()->c_str() : nullptr;
        if (mGateSpec.ic <= 0 || mGateSpec.oc <= 0 || mUpSpec.ic != mGateSpec.ic || mUpSpec.oc != mGateSpec.oc) {
            mValid = false;
            return;
        }
        mGateOpBuffer = buildPicGateUpChildConvOp(mGateSpec, externalPath);
        mUpOpBuffer = buildPicGateUpChildConvOp(mUpSpec, externalPath);
        mGateOp = flatbuffers::GetRoot<Op>(mGateOpBuffer.data());
        mUpOp = flatbuffers::GetRoot<Op>(mUpOpBuffer.data());
        mGateResource.reset(new ConvFpAIntBExecution::Resource(backend, mGateOp));
        mUpResource.reset(new ConvFpAIntBExecution::Resource(backend, mUpOp));
        mGateConv.reset(new ConvFpAIntBExecution(backend, mGateOp, mGateResource));
        mUpConv.reset(new ConvFpAIntBExecution(backend, mUpOp, mUpResource));
        mSilu.reset(new PicSiluMulExecution(backend));
    }

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (!mValid || inputs.size() != 1 || outputs.size() != 1) {
            return INPUT_DATA_ERROR;
        }
        mGateTemp.reset(Tensor::createDevice(outputs[0]->shape(), outputs[0]->getType(), outputs[0]->getDimensionType()));
        mUpTemp.reset(Tensor::createDevice(outputs[0]->shape(), outputs[0]->getType(), outputs[0]->getDimensionType()));
        if (mGateTemp == nullptr || mUpTemp == nullptr ||
            !backend()->onAcquireBuffer(mGateTemp.get(), Backend::DYNAMIC) ||
            !backend()->onAcquireBuffer(mUpTemp.get(), Backend::DYNAMIC)) {
            return OUT_OF_MEMORY;
        }
        auto code = mGateConv->onResize(inputs, {mGateTemp.get()});
        if (code == NO_ERROR) {
            code = mUpConv->onResize(inputs, {mUpTemp.get()});
        }
        if (code == NO_ERROR) {
            code = mSilu->onResize({mGateTemp.get(), mUpTemp.get()}, outputs);
        }
        backend()->onReleaseBuffer(mGateTemp.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mUpTemp.get(), Backend::DYNAMIC);
        return code;
    }

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (!mValid || inputs.size() != 1 || outputs.size() != 1) {
            return INPUT_DATA_ERROR;
        }
        auto meta = static_cast<KVMeta*>(static_cast<CUDABackend*>(backend())->getMetaPtr());
        const bool decodeRepairActive = meta != nullptr && meta->pic_decode_repair_sparse_active;
        if (canUseScalarFused(inputs, outputs, decodeRepairActive)) {
            launchFused(inputs[0], outputs[0]);
            return NO_ERROR;
        }
        auto code = mGateConv->onExecute(inputs, {mGateTemp.get()});
        if (code != NO_ERROR) {
            return code;
        }
        code = mUpConv->onExecute(inputs, {mUpTemp.get()});
        if (code != NO_ERROR) {
            return code;
        }
        return mSilu->onExecute({mGateTemp.get(), mUpTemp.get()}, outputs);
    }

private:
    bool canUseScalarFused(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           bool decodeRepairActive) const {
        if (!decodeRepairActive) {
            return false;
        }
        if (!mGateResource->mIsWeightInt4 || !mUpResource->mIsWeightInt4 ||
            mGateResource->mGemvParams == nullptr || mUpResource->mGemvParams == nullptr) {
            return false;
        }
        if (!static_cast<CUDABackend*>(backend())->useFp16()) {
            return false;
        }
        const int batch = inputs[0]->batch();
        if (batch < 2 || batch > 8) {
            return false;
        }
        return outputs[0]->channel() == mGateSpec.oc;
    }

    void launchFused(const Tensor* input, Tensor* output) {
        auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
        constexpr int kOcPerBlock = 2;
        const int batch = input->batch();
        const int ic = mGateSpec.ic;
        const int icp = UP_DIV(ic, 8) * 8;
        const int oc = mGateSpec.oc;
        const int ocp = UP_DIV(oc, 8) * 8;
        const int numQg = (mGateResource->mQuanC > 0) ? (mGateResource->mQuanC / oc) : 1;
        dim3 grid((oc + kOcPerBlock - 1) / kOcPerBlock);
        dim3 block(128);
        if (batch <= 3) {
            PicGateUpSiluInt4V14MBKernel<half, kOcPerBlock, 3><<<grid, block>>>(
                reinterpret_cast<const half*>(input->deviceId()),
                reinterpret_cast<const uint8_t*>(mGateResource->mFilter), mGateResource->mGemvParams,
                reinterpret_cast<const half*>(mGateResource->mBias),
                reinterpret_cast<const uint8_t*>(mUpResource->mFilter), mUpResource->mGemvParams,
                reinterpret_cast<const half*>(mUpResource->mBias),
                reinterpret_cast<half*>(output->deviceId()),
                FLT_MAX, -FLT_MAX, batch, ic, icp, oc, ocp, numQg);
        } else {
            PicGateUpSiluInt4V14MBKernel<half, kOcPerBlock, 8><<<grid, block>>>(
                reinterpret_cast<const half*>(input->deviceId()),
                reinterpret_cast<const uint8_t*>(mGateResource->mFilter), mGateResource->mGemvParams,
                reinterpret_cast<const half*>(mGateResource->mBias),
                reinterpret_cast<const uint8_t*>(mUpResource->mFilter), mUpResource->mGemvParams,
                reinterpret_cast<const half*>(mUpResource->mBias),
                reinterpret_cast<half*>(output->deviceId()),
                FLT_MAX, -FLT_MAX, batch, ic, icp, oc, ocp, numQg);
        }
        checkKernelErrors;
    }

    bool mValid = true;
    PicGateUpConvSpec mGateSpec;
    PicGateUpConvSpec mUpSpec;
    std::vector<uint8_t> mGateOpBuffer;
    std::vector<uint8_t> mUpOpBuffer;
    const Op* mGateOp = nullptr;
    const Op* mUpOp = nullptr;
    std::shared_ptr<ConvFpAIntBExecution::Resource> mGateResource;
    std::shared_ptr<ConvFpAIntBExecution::Resource> mUpResource;
    std::shared_ptr<ConvFpAIntBExecution> mGateConv;
    std::shared_ptr<ConvFpAIntBExecution> mUpConv;
    std::shared_ptr<Execution> mSilu;
    std::shared_ptr<Tensor> mGateTemp;
    std::shared_ptr<Tensor> mUpTemp;
};

class PicLinearNhwcWeightOnlyExecution : public Execution {
public:
    PicLinearNhwcWeightOnlyExecution(const Op* op, Backend* backend) : Execution(backend) {
        auto extra = op->main_as_Extra();
        mSpec = parsePicLinearNhwcSpec(extra);
        const char* externalPath = op->externalPath() != nullptr ? op->externalPath()->c_str() : nullptr;
        if (mSpec.ic <= 0 || mSpec.oc <= 0 || mSpec.external.size() < 3) {
            mValid = false;
            return;
        }
        mConvOpBuffer = buildPicGateUpChildConvOp(mSpec, externalPath);
        mConvOp = flatbuffers::GetRoot<Op>(mConvOpBuffer.data());
        mResource.reset(new ConvFpAIntBExecution::Resource(backend, mConvOp));
        mConv.reset(new ConvFpAIntBExecution(backend, mConvOp, mResource));
    }

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (!mValid || inputs.size() != 1 || outputs.size() != 1) {
            return INPUT_DATA_ERROR;
        }
        mRows = computeNhwcRows(inputs[0], mSpec.ic);
        const int outputRows = computeNhwcRows(outputs[0], mSpec.oc);
        if (!canUseNhwcCompact(inputs[0], outputs[0], mRows, outputRows)) {
            return NOT_SUPPORT;
        }
        forceNhwcFormat(inputs[0]);
        forceNhwcFormat(outputs[0]);
        return mConv->onResize(inputs, outputs);
    }

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (!mValid || inputs.size() != 1 || outputs.size() != 1) {
            return INPUT_DATA_ERROR;
        }
        forceNhwcFormat(inputs[0]);
        forceNhwcFormat(outputs[0]);
        const bool profile = profilePicLinearNhwc();
        uint64_t startUs = 0;
        if (profile) {
            cudaDeviceSynchronize();
            startUs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        }
        auto code = mConv->onExecute(inputs, outputs);
        if (profile) {
            cudaDeviceSynchronize();
            const auto endUs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
            auto meta = static_cast<KVMeta*>(static_cast<CUDABackend*>(backend())->getMetaPtr());
            MNN_PRINT("CUDAWeightOnlyConv profile op=pic_linear_nhwc_weight_only "
                      "name=%s batch=%d ic=%d oc=%d int4=%d static_dequant=%d "
                      "pic_decode_repair=%d total_us=%llu\n",
                      mSpec.name.c_str(), mRows, mSpec.ic, mSpec.oc,
                      mResource && mResource->mIsWeightInt4 ? 1 : 0,
                      mResource && mResource->mStaticDequantFilter != nullptr ? 1 : 0,
                      meta != nullptr && meta->pic_decode_repair_sparse_active ? 1 : 0,
                      static_cast<unsigned long long>(endUs - startUs));
        }
        return code;
    }

private:
    int computeNhwcRows(const Tensor* tensor, int channels) const {
        if (tensor == nullptr || tensor->dimensions() != 4 || channels <= 0 || tensor->length(3) != channels) {
            return 0;
        }
        const int d0 = tensor->length(0);
        const int d1 = tensor->length(1);
        const int d2 = tensor->length(2);
        if (d0 <= 0 || d1 <= 0 || d2 <= 0) {
            return 0;
        }
        return d0 * d1 * d2;
    }

    bool canUseNhwcCompact(const Tensor* input, const Tensor* output, int inputRows, int outputRows) const {
        if (input == nullptr || output == nullptr || input->dimensions() != 4 || output->dimensions() != 4) {
            return false;
        }
        if (!static_cast<CUDABackend*>(backend())->useFp16()) {
            return false;
        }
        if (mSpec.quantBit != 4 || mSpec.ic % 8 != 0 || mSpec.oc % 8 != 0) {
            return false;
        }
        if (inputRows <= 0 || outputRows <= 0 || inputRows != outputRows) {
            return false;
        }
        return input->length(3) == mSpec.ic && output->length(3) == mSpec.oc;
    }

    void forceNhwcFormat(Tensor* tensor) const {
        TensorUtils::getDescribe(tensor)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
    }

    bool mValid = true;
    int mRows = 0;
    PicGateUpConvSpec mSpec;
    std::vector<uint8_t> mConvOpBuffer;
    const Op* mConvOp = nullptr;
    std::shared_ptr<ConvFpAIntBExecution::Resource> mResource;
    std::shared_ptr<ConvFpAIntBExecution> mConv;
};
#endif

#ifdef MNN_CODEGEN_CUDA
FuseExecution::FuseExecution(const Op* op, Backend *backend) : Execution(backend) {
    // AUTOTIME;
    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime(); 
    auto extra = op->main_as_Extra();
    std::string source(reinterpret_cast<const char*>(extra->info()->data()));
    mSource = source;
    mName = extra->type()->c_str();
    mVectorize = extra->vector();
    // MNN_PRINT("\n\n%s\n\n%s \n\n", mSource.c_str(), mName);

    auto kernelInfoMap = static_cast<CUDABackend*>(backend)->kernelCuModuleMap();
    auto module = kernelInfoMap[std::pair<std::string, std::string>(mName, mSource)];
    MNN_CUDA_SAFE_CALL(cuModuleGetFunction(&mKernel, module, mName));
}

ErrorCode FuseExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime(); 
    auto output =outputs[0];
    auto format = TensorUtils::getDescribe(output)->dimensionFormat;
    auto dims = output->dimensions();
    batch = output->length(0);
    if (format == MNN_DATA_FORMAT_NHWC) {
        channel = output->length(dims-1);
        channel_pack = UP_DIV(channel, PACK_NUMBER) * PACK_NUMBER;
        area = 1;
        for(int i = 1; i < dims-1; i++) {
            area *= output->length(i);
        }

        if (mVectorize) { // Fast vectorize
            if(static_cast<CUDABackend*>(backend())->useFp16()) { // half2
                channel = channel / 2;
                channel_pack = channel_pack / 2; 
            } else { // float4
                channel = channel / 4;
                channel_pack = channel_pack / 4;
            }
        }
    } else if(format == MNN_DATA_FORMAT_NCHW || format == MNN_DATA_FORMAT_NC4HW4) {
        channel = output->length(1);
        channel_pack = UP_DIV(channel, PACK_NUMBER) * PACK_NUMBER;
        area = 1;
        for(int i = 2; i < dims; i++) {
            area *= output->length(i);
        }
    } else {
        MNN_ERROR("FuseExecution not support format:%d\n", format);
        MNN_ASSERT(false);
    }

    #if 0 // TODO : Optimize raster
    DivModFast d_area(area);
    DivModFast d_channel(channel);

    mDivChannelStorage = static_cast<CUDABackend*>(backend())->getStaticBufferPool()->alloc(sizeof(DivModFast));
    mDivAreaStorage = static_cast<CUDABackend*>(backend())->getStaticBufferPool()->alloc(sizeof(DivModFast));
    runtime->memcpy((uint8_t*)mDivAreaStorage.first + mDivAreaStorage.second, &d_area, sizeof(DivModFast), MNNMemcpyHostToDevice, true);
    runtime->memcpy((uint8_t*)mDivChannelStorage.first + mDivChannelStorage.second, &d_channel, sizeof(DivModFast), MNNMemcpyHostToDevice, true);
    #endif

    return NO_ERROR;
}

ErrorCode FuseExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto count = CUDABackend::realSize(outputs[0]);

    if(mVectorize) {
        if(static_cast<CUDABackend*>(backend())->useFp16()) { // half2
            count = count / 2;
        } else {
            count = count / 4;
        }
    }
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto& prop = runtime->prop();
    int threads_num = runtime->threads_num();//prop.maxThreadsPerBlock;
    int block_num = runtime->blocks_num(count);// prop.multiProcessorCount;

    std::vector<void *> args;

    if (static_cast<CUDABackend*>(backend())->useFp16()) {
        for (int i=0; i < inputs.size(); i++) {
            auto inputPtr = (const half*)inputs[i]->deviceId();
            args.emplace_back((void *)inputPtr);
        }
        for (int i=0; i < outputs.size(); i++) {
            auto outputPtr = (const half*)outputs[i]->deviceId();
            args.emplace_back((void *)outputPtr);
        }
    } else {
        for (int i=0; i < inputs.size(); i++) {
            auto inputPtr = (const float*)inputs[i]->deviceId();
            args.emplace_back((void *)inputPtr);
        }
        for (int i=0; i < outputs.size(); i++) {
            auto outputPtr = (const float*)outputs[i]->deviceId();
            args.emplace_back((void *)outputPtr);
        }
    }

    args.emplace_back((void *)count);

    //TODO : when can do not pass these params
    args.emplace_back((void *)batch);
    args.emplace_back((void *)area);
    args.emplace_back((void *)channel);
    args.emplace_back((void *)channel_pack);
    // args.emplace_back((void *)(DivModFast *)((uint8_t*)mDivAreaStorage.first + mDivAreaStorage.second));
    // args.emplace_back((void *)(DivModFast *)((uint8_t*)mDivChannelStorage.first + mDivChannelStorage.second));
    
    std::vector<void*> argsPtr;
    for(int i=0; i<args.size(); i++) {
        argsPtr.emplace_back(args.data() + i);
    }

    //printf("size  %p-%p-%p-%d   %p-%p-%p-%d\n\n", (const float*)inputs[0]->deviceId(), (const float*)inputs[1]->deargsviceId(), (const float*)outputs[0]->deviceId(), count, argsPtr[0], argsPtr[1], argsPtr[2], *((argsPtr[3])));

    MNN_CUDA_SAFE_CALL(
        cuLaunchKernel(mKernel,
        block_num, 1, 1, // grid dim
        threads_num, 1, 1, // block dim
        0, NULL, // shared mem 
        &(argsPtr[0]), 0)); // arguments

    return NO_ERROR;
}
#endif
class FuseCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto extra = op->main_as_Extra();
        if (extra != nullptr && extra->type() != nullptr && extra->type()->str() == "PicGateUpWeightOnly") {
#ifdef MNN_LOW_MEMORY
            return new PicGateUpWeightOnlyExecution(op, backend);
#else
            return nullptr;
#endif
        }
        if (extra != nullptr && extra->type() != nullptr && extra->type()->str() == "PicGateUpSiluWeightOnly") {
#ifdef MNN_LOW_MEMORY
            return new PicGateUpSiluWeightOnlyExecution(op, backend);
#else
            return nullptr;
#endif
        }
        if (extra != nullptr && extra->type() != nullptr && extra->type()->str() == "PicLinearNhwcWeightOnly") {
#ifdef MNN_LOW_MEMORY
            return new PicLinearNhwcWeightOnlyExecution(op, backend);
#else
            return nullptr;
#endif
        }
        if (extra != nullptr && extra->type() != nullptr && extra->type()->str() == "PicSiluMul") {
            return new PicSiluMulExecution(backend);
        }
        if (extra != nullptr && extra->type() != nullptr && extra->type()->str() == "PicPackedSiluMul") {
            return new PicPackedSiluMulExecution(backend);
        }
#ifdef MNN_CUDA_BENCH_OPS
        // Bench-only fused MLP/down candidates. Non-test builds intentionally do
        // not recognize these Extra types, so production/export graphs cannot select them.
        if (extra != nullptr && extra->type() != nullptr && extra->type()->str() == "PicBenchFusedPackedSiluDown") {
            return new PicBenchFusedPackedSiluDownExecution(backend, false);
        }
        if (extra != nullptr && extra->type() != nullptr && extra->type()->str() == "PicBenchStreamedPackedSiluDown") {
            return new PicBenchFusedPackedSiluDownExecution(backend, true);
        }
        if (extra != nullptr && extra->type() != nullptr && extra->type()->str() == "PicBenchWmmaPackedSiluDown") {
            return new PicBenchWmmaPackedSiluDownExecution(backend);
        }
#endif
#ifdef MNN_CODEGEN_CUDA
        if (FuseExecutionV2::check(op)) {
            return FuseExecutionV2::create(op, backend, inputs.size(), outputs.size());
        }
        return new FuseExecution(op, backend);
#else
        return nullptr;
#endif
    }
};

static CUDACreatorRegister<FuseCreator> __init(OpType_Extra);

};
};
