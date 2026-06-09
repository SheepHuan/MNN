//
//  OpenCLAttentionPerf.cpp
//  MNNTests
//

#if defined(MNN_SUPPORT_TRANSFORMER_FUSE) && defined(MNN_OPENCL_ENABLED) && !defined(MNN_OPENCL_BUFFER_CLOSED)

#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include "core/Backend.hpp"
#include "core/Execution.hpp"
#include "core/PagedKVMeta.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace MNN;

namespace {

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

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value) : mName(name) {
        const char* oldValue = ::getenv(name);
        if (oldValue != nullptr) {
            mHadOldValue = true;
            mOldValue = oldValue;
        }
        if (value != nullptr) {
            ::setenv(name, value, 1);
        } else {
            ::unsetenv(name);
        }
    }
    ~ScopedEnvVar() {
        if (mHadOldValue) {
            ::setenv(mName.c_str(), mOldValue.c_str(), 1);
        } else {
            ::unsetenv(mName.c_str());
        }
    }

private:
    std::string mName;
    std::string mOldValue;
    bool mHadOldValue = false;
};

static bool checkCL(cl_int code, const char* where) {
    if (code != CL_SUCCESS) {
        MNN_ERROR("%s failed: %d\n", where, code);
        return false;
    }
    return true;
}

class DirectOpBenchOpenCL {
public:
    explicit DirectOpBenchOpenCL(void* meta = nullptr) {
        Backend::Info info;
        info.type = MNN_FORWARD_OPENCL;
        info.numThread = MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_NONE;

        auto status = MNNTestSuite::get()->pStaus;
        mConfig.memory = static_cast<BackendConfig::MemoryMode>(status.memory);
        mConfig.precision = static_cast<BackendConfig::PrecisionMode>(status.precision);
        mConfig.power = static_cast<BackendConfig::PowerMode>(status.power);
        info.user = &mConfig;

        auto creator = MNNGetExtraRuntimeCreator(MNN_FORWARD_OPENCL);
        if (creator == nullptr) {
            MNN_ERROR("runtime creator is not registered for OpenCL backend\n");
            return;
        }
        mRuntime.reset(creator->onCreate(info));
        if (!mRuntime) {
            MNN_ERROR("failed to create OpenCL runtime\n");
            return;
        }
        mRuntime->pMeta = meta;
        mBackend.reset(mRuntime->onCreate(&mConfig));
        if (!mBackend) {
            MNN_ERROR("failed to create OpenCL backend\n");
            return;
        }
        mOpenCLBackend = static_cast<OpenCL::OpenCLBackend*>(mBackend.get());
    }

    bool valid() const {
        return mOpenCLBackend != nullptr;
    }

    Tensor* tensor(const std::vector<int>& shape, Tensor::DimensionType dim = Tensor::CAFFE) {
        auto t = std::shared_ptr<Tensor>(Tensor::createDevice<float>(shape, dim));
        if (!t || !mBackend->onAcquireBuffer(t.get(), Backend::STATIC)) {
            MNN_ERROR("failed to allocate OpenCL tensor\n");
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
        auto& queue = mOpenCLBackend->getOpenCLRuntime()->commandQueue();
        return checkCL(queue.enqueueWriteBuffer(OpenCL::openCLBuffer(tensor), CL_TRUE, 0,
                                                values.size() * sizeof(float), values.data()),
                       "OpenCL enqueueWriteBuffer");
    }

    bool zeroTensor(Tensor* tensor) {
        if (tensor == nullptr) {
            return false;
        }
        std::vector<float> zeros(static_cast<size_t>(tensor->elementSize()), 0.0f);
        return writeTensor(tensor, zeros);
    }

    bool sync() const {
        return checkCL(mOpenCLBackend->getOpenCLRuntime()->commandQueue().finish(), "OpenCL queue finish");
    }

private:
    BackendConfig mConfig;
    std::shared_ptr<Runtime> mRuntime;
    std::unique_ptr<Backend> mBackend;
    OpenCL::OpenCLBackend* mOpenCLBackend = nullptr;
    std::vector<std::shared_ptr<Tensor>> mTensors;
};

class OpenCLWallTimer {
public:
    bool measure(const std::function<ErrorCode()>& run, const std::function<bool()>& sync, int warmup, int repeat,
                 float* avgMs) {
        if (avgMs == nullptr || repeat <= 0) {
            return false;
        }
        for (int i = 0; i < warmup; ++i) {
            auto code = run();
            if (code != NO_ERROR) {
                MNN_ERROR("warmup onExecute failed: %d\n", code);
                return false;
            }
        }
        if (!sync()) {
            return false;
        }
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeat; ++i) {
            auto code = run();
            if (code != NO_ERROR) {
                MNN_ERROR("timed onExecute failed: %d\n", code);
                return false;
            }
        }
        if (!sync()) {
            return false;
        }
        auto end = std::chrono::steady_clock::now();
        auto totalUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        *avgMs = static_cast<float>(totalUs) / (1000.0f * static_cast<float>(repeat));
        return true;
    }
};

static std::unique_ptr<OpHolder> makeAttentionOp(OpType type, bool kvCache) {
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

static void setPagedMeta(PagedKVMeta& meta, int pastLen, int addLen, int removeLen = 0) {
    meta.previous = static_cast<size_t>(pastLen);
    meta.remove = static_cast<size_t>(removeLen);
    meta.reserve = nullptr;
    meta.n_reserve = 0;
    meta.add = static_cast<size_t>(addLen);
    meta.logical_length = pastLen;
}

static std::vector<float> makePattern(int size, float scale, float offset = 0.0f) {
    std::vector<float> data(size);
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>((i * 13 + 7) % 29 - 14) * scale + offset;
    }
    return data;
}

static std::vector<float> makeCausalMask(int queryLen, int kvLen) {
    std::vector<float> data(queryLen * kvLen);
    int gap = kvLen - queryLen;
    for (int q = 0; q < queryLen; ++q) {
        for (int k = 0; k < kvLen; ++k) {
            data[q * kvLen + k] = (k - q <= gap) ? 0.0f : -1.0e9f;
        }
    }
    return data;
}

static bool supportPerfPrecision() {
    auto precision = MNNTestSuite::get()->pStaus.precision;
    if (precision != BackendConfig::Precision_High) {
        MNN_PRINT("bench_ops/opencl/perf/PagedAttention/V1V2 uses fp32 OpenCL buffers; run with precision=1.\n");
        return false;
    }
    return true;
}

static bool printPagedAttentionV1V2TimedResult(const char* impl, const BenchCase& c, float avgMs) {
    const int kvLen = c.pastLen + c.seqLen;
    const float ratio = kvLen > 0 ? static_cast<float>(c.seqLen) / static_cast<float>(kvLen) : 0.0f;
    MNN_PRINT("[bench_ops/opencl/perf/PagedAttention/V1V2] %-28s impl=%-9s B=%d qH=%d kvH=%d D=%d "
              "past=%d add=%d kv=%d q_over_kv=%.4f avg=%.4f ms\n",
              c.name, impl, c.batch, c.qHeads, c.kvHeads, c.headDim, c.pastLen, c.seqLen, kvLen, ratio, avgMs);
    ::fflush(stdout);
    return true;
}

static bool measurePagedAttentionCase(const BenchCase& c, float* avgMs) {
    if (avgMs == nullptr) {
        return false;
    }
    PagedKVMeta meta;
    const int capacity = c.pastLen + c.seqLen + 64;
    meta.beginRequest(capacity);
    DirectOpBenchOpenCL bench(&meta);
    if (!bench.valid()) {
        return false;
    }

    auto op = makeAttentionOp(OpType_PagedAttention, true);
    auto q = bench.tensor({c.batch, c.seqLen, c.qHeads, c.headDim});
    auto k = bench.tensor({c.batch, c.seqLen, c.kvHeads, c.headDim});
    auto v = bench.tensor({c.batch, c.seqLen, c.kvHeads, c.headDim});
    auto o = bench.tensor({c.batch, c.seqLen, c.qHeads, c.headDim});
    if (!q || !k || !v || !o) {
        return false;
    }
    auto qData = makePattern(c.batch * c.seqLen * c.qHeads * c.headDim, 0.003f, 0.001f);
    auto kData = makePattern(c.batch * c.seqLen * c.kvHeads * c.headDim, 0.002f, 0.002f);
    auto vData = makePattern(c.batch * c.seqLen * c.kvHeads * c.headDim, 0.004f, -0.001f);
    if (!bench.writeTensor(q, qData) || !bench.writeTensor(k, kData) || !bench.writeTensor(v, vData)) {
        return false;
    }

    std::vector<Tensor*> inputs = {q, k, v};
    if (c.pastLen == 0 && c.seqLen > 1) {
        auto mask = bench.tensor({c.seqLen, c.seqLen});
        auto maskData = makeCausalMask(c.seqLen, c.seqLen);
        if (!mask || !bench.writeTensor(mask, maskData)) {
            return false;
        }
        inputs.emplace_back(mask);
    }
    std::vector<Tensor*> outputs = {o};
    auto exe = bench.create(inputs, outputs, op->get());
    if (!exe) {
        MNN_ERROR("failed to create OpenCL PagedAttention execution\n");
        return false;
    }
    auto code = bench.resize(exe.get(), inputs, outputs);
    if (code != NO_ERROR) {
        MNN_ERROR("OpenCL PagedAttention onResize failed: %d\n", code);
        return false;
    }

    OpenCLWallTimer timer;
    auto run = [&]() {
        setPagedMeta(meta, c.pastLen, c.seqLen);
        return bench.execute(exe.get(), inputs, outputs);
    };
    if (!timer.measure(run, [&]() { return bench.sync(); }, c.warmup, c.repeat, avgMs)) {
        return false;
    }
    return true;
}

static std::vector<BenchCase> pagedAttentionV1V2RatioCases() {
    return {
        {"llama3.2-3B_kv2048_q1", 1, 24, 8, 128, 1, 2047, 5, 50},
        {"llama3.2-3B_kv2048_q2", 1, 24, 8, 128, 2, 2046, 5, 50},
        {"llama3.2-3B_kv2048_q4", 1, 24, 8, 128, 4, 2044, 5, 50},
        {"llama3.2-3B_kv2048_q8", 1, 24, 8, 128, 8, 2040, 5, 50},
        {"llama3.2-3B_kv2048_q16", 1, 24, 8, 128, 16, 2032, 3, 30},
        {"llama3.2-3B_kv2048_q32", 1, 24, 8, 128, 32, 2016, 3, 20},
        {"llama3.2-3B_kv2048_q64", 1, 24, 8, 128, 64, 1984, 3, 20},
        {"llama3.2-3B_kv2048_q128", 1, 24, 8, 128, 128, 1920, 3, 10},
        {"llama3.2-3B_kv2048_q256", 1, 24, 8, 128, 256, 1792, 2, 5},
        {"llama3.2-3B_kv2048_q512", 1, 24, 8, 128, 512, 1536, 2, 3},
        {"llama3.2-3B_kv2048_q1024", 1, 24, 8, 128, 1024, 1024, 1, 2},
        {"llama3.2-3B_kv2048_q1536", 1, 24, 8, 128, 1536, 512, 1, 1},
    };
}

static bool runPagedAttentionV1V2Case(const BenchCase& c) {
    struct ImplCase {
        const char* label;
        const char* impl;
        const char* forceV2Kernel;
    };
    const ImplCase impls[] = {
        {"v1", "v1", nullptr},
        {"v2_kernel", "v2", "1"},
        {"v2_route", "v2", nullptr},
    };
    for (auto impl : impls) {
        ScopedEnvVar implEnv("MNN_PAGED_ATTENTION_IMPL", impl.impl);
        ScopedEnvVar forceEnv("MNN_PAGED_ATTENTION_BENCH_FORCE_V2_KERNEL", impl.forceV2Kernel);
        float avgMs = 0.0f;
        if (!measurePagedAttentionCase(c, &avgMs)) {
            return false;
        }
        if (!printPagedAttentionV1V2TimedResult(impl.label, c, avgMs)) {
            return false;
        }
    }
    return true;
}

class OpenCLPagedAttentionV1V2Perf : public MNNTestCase {
public:
    bool run(int precision) override {
        if (!supportPerfPrecision()) {
            return true;
        }
        auto cases = pagedAttentionV1V2RatioCases();
        for (auto& c : cases) {
            if (!runPagedAttentionV1V2Case(c)) {
                return false;
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(OpenCLPagedAttentionV1V2Perf, "bench_ops/opencl/perf/PagedAttention/V1V2");

} // namespace

#endif
