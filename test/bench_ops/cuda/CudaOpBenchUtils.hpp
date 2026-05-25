//
//  CudaOpBenchUtils.hpp
//  MNNTests
//

#pragma once

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <MNN/expr/Executor.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "core/Backend.hpp"
#include "core/Execution.hpp"
#include "core/KVMeta.hpp"
#include "core/PagedKVMeta.hpp"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace MNN {
namespace BenchOpsCuda {

struct OpHolder {
    flatbuffers::FlatBufferBuilder builder;

    explicit OpHolder(const OpT& op) {
        auto offset = Op::Pack(builder, &op);
        builder.Finish(offset);
    }

    const Op* get() const {
        return flatbuffers::GetRoot<Op>(builder.GetBufferPointer());
    }
};

struct BenchCase {
    const char* name;
    int batch;
    int qHeads;
    int kvHeads;
    int headDim;
    int seqLen;
    int pastLen;
    int warmup;
    int repeat;
};

inline bool checkCuda(cudaError_t code, const char* where) {
    if (code != cudaSuccess) {
        MNN_ERROR("%s failed: %s\n", where, cudaGetErrorString(code));
        return false;
    }
    return true;
}

class CudaEventPair {
public:
    CudaEventPair() {
        mValid = cudaEventCreate(&mStart) == cudaSuccess && cudaEventCreate(&mStop) == cudaSuccess;
    }
    ~CudaEventPair() {
        if (mStart != nullptr) {
            cudaEventDestroy(mStart);
        }
        if (mStop != nullptr) {
            cudaEventDestroy(mStop);
        }
    }

    bool measure(const std::function<ErrorCode()>& run, int warmup, int repeat, float* avgMs) {
        if (!mValid || avgMs == nullptr || repeat <= 0) {
            return false;
        }
        for (int i = 0; i < warmup; ++i) {
            auto code = run();
            if (code != NO_ERROR) {
                MNN_ERROR("warmup onExecute failed: %d\n", code);
                return false;
            }
        }
        if (!checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize before timing")) {
            return false;
        }
        if (!checkCuda(cudaEventRecord(mStart, 0), "cudaEventRecord(start)")) {
            return false;
        }
        for (int i = 0; i < repeat; ++i) {
            auto code = run();
            if (code != NO_ERROR) {
                MNN_ERROR("timed onExecute failed: %d\n", code);
                return false;
            }
        }
        if (!checkCuda(cudaEventRecord(mStop, 0), "cudaEventRecord(stop)")) {
            return false;
        }
        if (!checkCuda(cudaEventSynchronize(mStop), "cudaEventSynchronize(stop)")) {
            return false;
        }
        float totalMs = 0.0f;
        if (!checkCuda(cudaEventElapsedTime(&totalMs, mStart, mStop), "cudaEventElapsedTime")) {
            return false;
        }
        *avgMs = totalMs / static_cast<float>(repeat);
        return true;
    }

private:
    cudaEvent_t mStart = nullptr;
    cudaEvent_t mStop = nullptr;
    bool mValid = false;
};

class DirectOpBench {
public:
    explicit DirectOpBench(MNNForwardType type, void* meta = nullptr) : mType(type) {
        Backend::Info info;
        info.type = type;
        info.numThread = 1;

        auto status = MNNTestSuite::get()->pStaus;
        mConfig.memory = static_cast<BackendConfig::MemoryMode>(status.memory);
        mConfig.precision = static_cast<BackendConfig::PrecisionMode>(status.precision);
        mConfig.power = static_cast<BackendConfig::PowerMode>(status.power);
        info.user = &mConfig;

        auto creator = MNNGetExtraRuntimeCreator(type);
        if (creator == nullptr) {
            MNN_ERROR("runtime creator is not registered for backend %d\n", type);
            return;
        }
        mRuntime.reset(creator->onCreate(info));
        if (!mRuntime) {
            MNN_ERROR("failed to create runtime for backend %d\n", type);
            return;
        }
        mRuntime->pMeta = meta;
        mBackend.reset(mRuntime->onCreate(&mConfig));
        if (!mBackend) {
            MNN_ERROR("failed to create backend %d\n", type);
            return;
        }
    }

    bool valid() const {
        return mBackend != nullptr;
    }

    bool isCuda() const {
        return mType == MNN_FORWARD_CUDA;
    }

    Tensor* tensor(const std::vector<int>& shape, Tensor::DimensionType dim = Tensor::CAFFE) {
        auto t = std::shared_ptr<Tensor>(Tensor::createDevice<float>(shape, dim));
        if (!t || !mBackend->onAcquireBuffer(t.get(), Backend::STATIC)) {
            MNN_ERROR("failed to allocate tensor for backend %d\n", mType);
            return nullptr;
        }
        if (!zeroTensor(t.get())) {
            return nullptr;
        }
        mTensors.emplace_back(std::move(t));
        return mTensors.back().get();
    }

    std::unique_ptr<Execution> create(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                      const Op* op) {
        return std::unique_ptr<Execution>(mBackend->onCreate(inputs, outputs, op));
    }

    ErrorCode resize(Execution* exe, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
        if (exe == nullptr) {
            return INVALID_VALUE;
        }
        mBackend->onResizeBegin();
        auto code = exe->onResize(inputs, outputs);
        if (code != NO_ERROR) {
            return code;
        }
        return mBackend->onResizeEnd();
    }

    ErrorCode execute(Execution* exe, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
        if (exe == nullptr) {
            return INVALID_VALUE;
        }
        mBackend->onExecuteBegin();
        auto code = exe->onExecute(inputs, outputs);
        mBackend->onExecuteEnd();
        return code;
    }

    bool writeTensor(Tensor* tensor, const std::vector<float>& values) {
        if (tensor == nullptr || values.size() != static_cast<size_t>(tensor->elementSize())) {
            MNN_ERROR("writeTensor shape mismatch\n");
            return false;
        }
        if (isCuda()) {
            return checkCuda(cudaMemcpy(reinterpret_cast<void*>(tensor->deviceId()), values.data(),
                                        values.size() * sizeof(float), cudaMemcpyHostToDevice),
                             "cudaMemcpy(host->device)");
        }
        auto dst = tensor->host<float>();
        if (dst == nullptr) {
            MNN_ERROR("writeTensor host pointer is null\n");
            return false;
        }
        ::memcpy(dst, values.data(), values.size() * sizeof(float));
        return true;
    }

    bool readTensor(const Tensor* tensor, std::vector<float>* values) const {
        if (tensor == nullptr || values == nullptr) {
            return false;
        }
        values->resize(tensor->elementSize());
        if (isCuda()) {
            if (!checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize before readback")) {
                return false;
            }
            return checkCuda(cudaMemcpy(values->data(), reinterpret_cast<const void*>(tensor->deviceId()),
                                        values->size() * sizeof(float), cudaMemcpyDeviceToHost),
                             "cudaMemcpy(device->host)");
        }
        auto src = tensor->host<float>();
        if (src == nullptr) {
            MNN_ERROR("readTensor host pointer is null\n");
            return false;
        }
        ::memcpy(values->data(), src, values->size() * sizeof(float));
        return true;
    }

    bool zeroTensor(Tensor* tensor) {
        if (tensor == nullptr) {
            return false;
        }
        if (isCuda()) {
            auto bytes = static_cast<size_t>(tensor->elementSize()) * sizeof(float);
            return checkCuda(cudaMemset(reinterpret_cast<void*>(tensor->deviceId()), 0, bytes), "cudaMemset(tensor)");
        }
        auto bytes = static_cast<size_t>(tensor->elementSize()) * tensor->getType().bytes();
        ::memset(tensor->host<int8_t>(), 0, bytes);
        return true;
    }

private:
    MNNForwardType mType;
    BackendConfig mConfig;
    std::shared_ptr<Runtime> mRuntime;
    std::unique_ptr<Backend> mBackend;
    std::vector<std::shared_ptr<Tensor>> mTensors;
};

inline std::unique_ptr<OpHolder> makeLinearAttentionOp(int kvHeads, int qHeads, int headDim) {
    OpT op;
    op.type = OpType_LinearAttention;
    op.main.type = OpParameter_LinearAttentionParam;
    op.main.value = new LinearAttentionParamT;
    auto* param = op.main.AsLinearAttentionParam();
    param->attn_type = "gated_delta_rule";
    param->num_k_heads = kvHeads;
    param->num_v_heads = qHeads;
    param->head_k_dim = headDim;
    param->head_v_dim = headDim;
    param->use_qk_l2norm = true;
    return std::unique_ptr<OpHolder>(new OpHolder(op));
}

inline std::unique_ptr<OpHolder> makeAttentionOp(OpType type, bool kvCache) {
    OpT op;
    op.type = type;
    op.main.type = OpParameter_AttentionParam;
    op.main.value = new AttentionParamT;
    auto* param = op.main.AsAttentionParam();
    param->kv_cache = kvCache;
    param->layer_index = 0;
    param->kv_shared_layer_index = -1;
    return std::unique_ptr<OpHolder>(new OpHolder(op));
}

inline void setKVMeta(KVMeta& meta, int pastLen, int addLen, int removeLen = 0) {
    meta.previous = static_cast<size_t>(pastLen);
    meta.remove = static_cast<size_t>(removeLen);
    meta.reserve = nullptr;
    meta.n_reserve = 0;
    meta.add = static_cast<size_t>(addLen);
}

inline void setPagedMeta(PagedKVMeta& meta, int pastLen, int addLen, int removeLen = 0) {
    setKVMeta(meta, pastLen, addLen, removeLen);
    meta.logical_length = pastLen;
}

inline bool supportPerfPrecision() {
    auto precision = MNNTestSuite::get()->pStaus.precision;
    if (precision != BackendConfig::Precision_High && precision != BackendConfig::Precision_Normal) {
        MNN_PRINT("bench_ops/cuda/perf only uses fp32 tensors; run with precision=1 or precision=0.\n");
        return false;
    }
    return true;
}

inline bool supportAccuracyPrecision() {
    auto precision = MNNTestSuite::get()->pStaus.precision;
    if (precision != BackendConfig::Precision_High && precision != BackendConfig::Precision_Normal) {
        MNN_PRINT("bench_ops/cuda/accuracy only compares fp32 tensors; run with precision=1 or precision=0.\n");
        return false;
    }
    return true;
}

inline std::vector<float> makePattern(int size, float scale, float offset = 0.0f) {
    std::vector<float> data(size);
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>((i * 13 + 7) % 29 - 14) * scale + offset;
    }
    return data;
}

inline std::vector<float> makeGate(int size) {
    std::vector<float> data(size);
    for (int i = 0; i < size; ++i) {
        data[i] = -0.05f * static_cast<float>((i % 7) + 1);
    }
    return data;
}

inline std::vector<float> makeBeta(int size) {
    std::vector<float> data(size);
    for (int i = 0; i < size; ++i) {
        data[i] = 0.05f * static_cast<float>((i % 11) + 1);
    }
    return data;
}

inline std::vector<float> makeCausalMask(int queryLen, int kvLen) {
    std::vector<float> data(queryLen * kvLen);
    int gap = kvLen - queryLen;
    for (int q = 0; q < queryLen; ++q) {
        for (int k = 0; k < kvLen; ++k) {
            data[q * kvLen + k] = (k - q <= gap) ? 0.0f : -1.0e9f;
        }
    }
    return data;
}

inline bool compareVectors(const char* name, const std::vector<float>& got, const std::vector<float>& expected,
                           float absTol, float relTol) {
    if (got.size() != expected.size()) {
        MNN_ERROR("%s size mismatch: got %zu expected %zu\n", name, got.size(), expected.size());
        return false;
    }
    float maxAbs = 0.0f;
    float maxRel = 0.0f;
    int bad = 0;
    int badIndex = -1;
    for (size_t i = 0; i < got.size(); ++i) {
        float diff = std::fabs(got[i] - expected[i]);
        float denom = std::max(1.0f, std::fabs(expected[i]));
        float rel = diff / denom;
        if (diff > maxAbs) {
            maxAbs = diff;
            maxRel = rel;
        }
        if (diff > absTol && rel > relTol) {
            if (badIndex < 0) {
                badIndex = static_cast<int>(i);
            }
            ++bad;
        }
    }
    MNN_PRINT("[bench_ops/cuda/accuracy] %-28s max_abs=%.6g max_rel=%.6g bad=%d/%zu\n", name, maxAbs, maxRel,
              bad, got.size());
    if (bad > 0) {
        MNN_ERROR("%s failed first_bad=%d got=%.8f expected=%.8f\n", name, badIndex, got[badIndex],
                  expected[badIndex]);
        return false;
    }
    return true;
}

} // namespace BenchOpsCuda
} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
