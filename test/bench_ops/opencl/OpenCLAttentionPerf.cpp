//
//  OpenCLAttentionPerf.cpp
//  MNNTests
//

#if defined(MNN_SUPPORT_TRANSFORMER_FUSE) && !defined(MNN_OPENCL_BUFFER_CLOSED)

#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include "backend/opencl/execution/buffer/ConvBufAdrenoUtils.hpp"
#include "core/Backend.hpp"
#include "core/Execution.hpp"
#include "core/FileLoader.hpp"
#include "core/IDSTEncoder.hpp"
#include "core/PagedKVMeta.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
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

static int applyOpenCLTuneLevelOverride(int numThread) {
    const char* envValue = ::getenv("MNN_OPENCL_TUNE_LEVEL");
    if (envValue == nullptr || envValue[0] == '\0') {
        return numThread;
    }
    std::string level(envValue);
    std::transform(level.begin(), level.end(), level.begin(),
                   [](unsigned char c) { return static_cast<char>(::tolower(c)); });
    const int tuneMask = MNN_GPU_TUNING_NONE | MNN_GPU_TUNING_FAST | MNN_GPU_TUNING_NORMAL |
                         MNN_GPU_TUNING_HEAVY | MNN_GPU_TUNING_WIDE;
    int tuneFlag = 0;
    if (level == "none") {
        tuneFlag = MNN_GPU_TUNING_NONE;
    } else if (level == "fast") {
        tuneFlag = MNN_GPU_TUNING_FAST;
    } else if (level == "normal") {
        tuneFlag = MNN_GPU_TUNING_NORMAL;
    } else if (level == "heavy") {
        tuneFlag = MNN_GPU_TUNING_HEAVY;
    } else if (level == "wide") {
        tuneFlag = MNN_GPU_TUNING_WIDE;
    } else {
        return numThread;
    }
    return (numThread & ~tuneMask) | tuneFlag;
}

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
        info.numThread = applyOpenCLTuneLevelOverride(info.numThread);

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
        loadCacheFromEnv();
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

    template <typename T>
    Tensor* tensorTyped(const std::vector<int>& shape, Tensor::DimensionType dim = Tensor::CAFFE, T initValue = (T)0) {
        auto t = std::shared_ptr<Tensor>(Tensor::createDevice<T>(shape, dim));
        if (!t || !mBackend->onAcquireBuffer(t.get(), Backend::STATIC)) {
            MNN_ERROR("failed to allocate OpenCL tensor\n");
            return nullptr;
        }
        if (!fillTensor<T>(t.get(), initValue)) {
            return nullptr;
        }
        mTensors.emplace_back(std::move(t));
        return mTensors.back().get();
    }

    Tensor* tensor(const std::vector<int>& shape, Tensor::DimensionType dim = Tensor::CAFFE) {
        return tensorTyped<float>(shape, dim, 0.0f);
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

    template <typename T>
    bool writeTensorTyped(Tensor* tensor, const std::vector<T>& values) {
        if (tensor == nullptr || values.size() != static_cast<size_t>(tensor->elementSize())) {
            MNN_ERROR("writeTensor shape mismatch\n");
            return false;
        }
        std::shared_ptr<Tensor> hostTensor(Tensor::create<T>(tensor->shape(), (void*)values.data(),
                                                             tensor->getDimensionType()));
        if (!hostTensor) {
            MNN_ERROR("failed to create host tensor for write\n");
            return false;
        }
        return tensor->copyFromHostTensor(hostTensor.get());
    }

    bool writeTensor(Tensor* tensor, const std::vector<float>& values) {
        return writeTensorTyped<float>(tensor, values);
    }

    template <typename T>
    bool readTensorTyped(Tensor* tensor, std::vector<T>* values) {
        if (tensor == nullptr || values == nullptr) {
            return false;
        }
        std::shared_ptr<Tensor> hostTensor(Tensor::createHostTensorFromDevice(tensor, true), Tensor::destroy);
        if (!hostTensor) {
            MNN_ERROR("failed to create host tensor for read\n");
            return false;
        }
        values->assign(hostTensor->host<T>(), hostTensor->host<T>() + hostTensor->elementSize());
        return true;
    }

    template <typename T>
    bool fillTensor(Tensor* tensor, T value) {
        if (tensor == nullptr) {
            return false;
        }
        std::vector<T> values(static_cast<size_t>(tensor->elementSize()), value);
        return writeTensorTyped<T>(tensor, values);
    }

    bool zeroTensor(Tensor* tensor) {
        return fillTensor<float>(tensor, 0.0f);
    }

    bool sync() const {
        return checkCL(mOpenCLBackend->getOpenCLRuntime()->commandQueue().finish(), "OpenCL queue finish");
    }

    OpenCLRuntime* runtime() const {
        return mOpenCLBackend != nullptr ? mOpenCLBackend->getOpenCLRuntime() : nullptr;
    }

    OpenCL::OpenCLBackend* openCLBackend() const {
        return mOpenCLBackend;
    }

private:
    void loadCacheFromEnv() {
        const char* cacheFile = ::getenv("MNN_BENCH_OPENCL_CACHE_FILE");
        if (cacheFile == nullptr || cacheFile[0] == '\0') {
            return;
        }
        std::unique_ptr<FileLoader> loader(new FileLoader(cacheFile, true));
        if (!loader->valid() || !loader->read() || loader->size() == 0 || !loader->merge(mCacheBuffer)) {
            MNN_PRINT("OpenCL bench cache load failed: %s\n", cacheFile);
            return;
        }
        if (!mRuntime->onSetCache(mCacheBuffer.get(), mCacheBuffer.size())) {
            MNN_PRINT("OpenCL bench cache invalid, ignore: %s\n", cacheFile);
            return;
        }
        MNN_PRINT("OpenCL bench loaded cache: %s\n", cacheFile);
    }

    BackendConfig mConfig;
    std::shared_ptr<Runtime> mRuntime;
    std::unique_ptr<Backend> mBackend;
    OpenCL::OpenCLBackend* mOpenCLBackend = nullptr;
    AutoStorage<uint8_t> mCacheBuffer;
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

static bool measureWithPrepare(const std::function<bool()>& prepare,
                               const std::function<ErrorCode()>& run,
                               const std::function<bool()>& sync,
                               int warmup, int repeat, float* avgMs) {
    if (avgMs == nullptr || repeat <= 0) {
        return false;
    }
    for (int i = 0; i < warmup; ++i) {
        if (prepare && !prepare()) {
            MNN_ERROR("prepared sparse runtime warmup state failed\n");
            return false;
        }
        auto code = run();
        if (code != NO_ERROR) {
            MNN_ERROR("prepared sparse runtime warmup onExecute failed: %d\n", code);
            return false;
        }
        if (!sync()) {
            return false;
        }
    }
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
        if (prepare && !prepare()) {
            MNN_ERROR("prepared sparse runtime timed state failed\n");
            return false;
        }
        auto code = run();
        if (code != NO_ERROR) {
            MNN_ERROR("prepared sparse runtime timed onExecute failed: %d\n", code);
            return false;
        }
        if (!sync()) {
            return false;
        }
    }
    auto end = std::chrono::steady_clock::now();
    auto totalUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    *avgMs = static_cast<float>(totalUs) / (1000.0f * static_cast<float>(repeat));
    return true;
}

static std::unique_ptr<OpHolder> makeAttentionOp(OpType type, bool kvCache, int layerIndex = 0,
                                                 int kvSharedLayerIndex = -1) {
    OpT op;
    op.type = type;
    op.main.type = OpParameter_AttentionParam;
    op.main.value = new AttentionParamT;
    auto* param = op.main.AsAttentionParam();
    param->kv_cache = kvCache;
    param->layer_index = layerIndex;
    param->kv_shared_layer_index = kvSharedLayerIndex;
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

static int envInt(const char* name, int fallback) {
    const char* value = ::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::max(1, ::atoi(value));
}

static bool envFlag(const char* name, bool fallback = false) {
    const char* value = ::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return value[0] != '0';
}

static bool enabledByFilter(const char* envName, const char* name) {
    const char* filter = ::getenv(envName);
    if (filter == nullptr || filter[0] == '\0') {
        return true;
    }
    return std::string(name).find(filter) != std::string::npos;
}

static bool enabledRow(const char* envName, int row) {
    const char* filter = ::getenv(envName);
    if (filter == nullptr || filter[0] == '\0') {
        return true;
    }
    std::string spec(filter);
    size_t start = 0;
    while (start <= spec.size()) {
        size_t end = spec.find(',', start);
        std::string item = spec.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!item.empty()) {
            size_t dash = item.find('-');
            if (dash == std::string::npos) {
                if (::atoi(item.c_str()) == row) {
                    return true;
                }
            } else {
                int lo = ::atoi(item.substr(0, dash).c_str());
                int hi = ::atoi(item.substr(dash + 1).c_str());
                if (lo > hi) {
                    std::swap(lo, hi);
                }
                if (row >= lo && row <= hi) {
                    return true;
                }
            }
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return false;
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
        const char* forceV2Kernel;
    };
    const ImplCase impls[] = {
        {"v1", nullptr},
        {"v2_kernel", "1"},
    };
    for (auto impl : impls) {
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

enum class PicRuntimeSparseBenchMode {
    LaterCompact,
    ScoreFullQ,
};

struct PicRuntimeSparseBenchCase {
    const char* name;
    PicRuntimeSparseBenchMode mode;
    int batch;
    int qHeads;
    int kvHeads;
    int headDim;
    int kvLen;
    int activeLen;
    int layerIndex;
    bool evenScatter;
    int warmup;
    int repeat;
};

static const char* picRuntimeSparseBenchModeName(PicRuntimeSparseBenchMode mode) {
    switch (mode) {
        case PicRuntimeSparseBenchMode::LaterCompact:
            return "later_compact";
        case PicRuntimeSparseBenchMode::ScoreFullQ:
            return "score_fullq";
        default:
            return "unknown";
    }
}

static std::string envLabelOrAuto(const char* primary, const char* secondary = nullptr) {
    const char* value = ::getenv(primary);
    if ((value == nullptr || value[0] == '\0') && secondary != nullptr) {
        value = ::getenv(secondary);
    }
    return (value != nullptr && value[0] != '\0') ? std::string(value) : std::string("auto");
}

static std::vector<int> makePicRuntimeLogicalIndices(int activeLen, int kvLen, bool evenScatter) {
    std::vector<int> indices;
    if (activeLen <= 0 || kvLen <= 0) {
        return indices;
    }
    indices.resize(activeLen);
    if (!evenScatter || activeLen == 1) {
        for (int i = 0; i < activeLen; ++i) {
            indices[i] = std::min(i, kvLen - 1);
        }
        return indices;
    }
    for (int i = 0; i < activeLen; ++i) {
        indices[i] = static_cast<int>((static_cast<int64_t>(i) * (kvLen - 1)) / std::max(1, activeLen - 1));
    }
    return indices;
}

static bool printPicRuntimeSparseTimedResult(const PicRuntimeSparseBenchCase& c, float avgMs) {
    MNN_PRINT("[bench_ops/opencl/perf/PagedAttention/SparseRuntime] %-34s mode=%-13s "
              "variant=%-24s schedule=%-12s score_family=%-8s kv=%d active=%d qH=%d kvH=%d D=%d avg=%.4f ms\n",
              c.name, picRuntimeSparseBenchModeName(c.mode),
              envLabelOrAuto("MNN_BENCH_OPENCL_PAGED_SPARSE_FORCE_VARIANT",
                             "MNN_PAGED_ATTENTION_BENCH_FORCE_SPARSE_FLASH_VARIANT").c_str(),
              envLabelOrAuto("MNN_BENCH_OPENCL_PAGED_SPARSE_FORCE_SCHEDULE",
                             "MNN_PAGED_ATTENTION_BENCH_FORCE_SPARSE_FLASH_SCHEDULE").c_str(),
              envLabelOrAuto("MNN_BENCH_OPENCL_PAGED_SCORE_FORCE_FAMILY",
                             "MNN_PAGED_ATTENTION_BENCH_FORCE_SCORE_SPARSE_FAMILY").c_str(),
              c.kvLen, c.activeLen, c.qHeads, c.kvHeads, c.headDim, avgMs);
    ::fflush(stdout);
    return true;
}

static bool useAdrenoSparseRuntimeSingleShot(const DirectOpBenchOpenCL& bench) {
    auto runtime = bench.runtime();
    if (runtime == nullptr || runtime->getGpuType() != ADRENO) {
        return false;
    }
    const char* value = ::getenv("MNN_BENCH_OPENCL_PAGED_SPARSE_SINGLE_SHOT");
    if (value == nullptr || value[0] == '\0') {
        return true;
    }
    return value[0] != '0';
}

static bool runPicRuntimeSparseBenchCase(const PicRuntimeSparseBenchCase& c) {
    PagedKVMeta meta;
    meta.beginRequest(c.kvLen + 64);
    DirectOpBenchOpenCL bench(&meta);
    if (!bench.valid()) {
        return false;
    }
    const auto logicalIndices = makePicRuntimeLogicalIndices(c.activeLen, c.kvLen, c.evenScatter);
    const auto opType =
        c.mode == PicRuntimeSparseBenchMode::ScoreFullQ ? OpType_PicScoreAttention : OpType_PicSparseAttention;
    auto op = makeAttentionOp(opType, true, c.layerIndex);
    if (!op) {
        return false;
    }

    auto queryLen = c.mode == PicRuntimeSparseBenchMode::ScoreFullQ ? c.kvLen : c.activeLen;
    auto timedKvLen = c.mode == PicRuntimeSparseBenchMode::ScoreFullQ ? c.kvLen : c.activeLen;
    auto q = bench.tensor({c.batch, queryLen, c.qHeads, c.headDim});
    auto k = bench.tensor({c.batch, timedKvLen, c.kvHeads, c.headDim});
    auto v = bench.tensor({c.batch, timedKvLen, c.kvHeads, c.headDim});
    auto o = bench.tensor({c.batch, c.activeLen, c.qHeads, c.headDim});
    if (!q || !k || !v || !o) {
        return false;
    }
    auto qData = makePattern(c.batch * queryLen * c.qHeads * c.headDim, 0.003f, 0.001f);
    auto kvData = makePattern(c.batch * timedKvLen * c.kvHeads * c.headDim, 0.002f, 0.002f);
    auto vvData = makePattern(c.batch * timedKvLen * c.kvHeads * c.headDim, 0.004f, -0.001f);
    if (!bench.writeTensor(q, qData) || !bench.writeTensor(k, kvData) || !bench.writeTensor(v, vvData)) {
        return false;
    }

    std::vector<Tensor*> timedInputs = {q, k, v};
    std::vector<Tensor*> timedOutputs = {o};
    Tensor* budget = nullptr;
    Tensor* activeIndices = nullptr;
    if (c.mode == PicRuntimeSparseBenchMode::ScoreFullQ) {
        auto prepareScoreState = [&]() -> bool {
            meta.beginRequest(c.kvLen + 64);
            meta.cacheblend_score_active = true;
            meta.cacheblend_score_ready = true;
            meta.cacheblend_score_layer_idx = c.layerIndex;
            meta.cacheblend_score_pic_start = 0;
            meta.cacheblend_score_pic_token_count = c.kvLen;
            meta.cacheblend_score_top_k = c.activeLen;
            meta.cacheblend_score_selected_local_indices = logicalIndices;
            return true;
        };
        budget = bench.tensorTyped<int32_t>({1}, Tensor::CAFFE, c.activeLen);
        activeIndices = bench.tensorTyped<int32_t>({c.activeLen}, Tensor::CAFFE, 0);
        if (!budget || !activeIndices) {
            return false;
        }
        timedInputs.emplace_back(budget);
        timedOutputs.emplace_back(activeIndices);
        if (!prepareScoreState()) {
            return false;
        }

        auto exe = bench.create(timedInputs, timedOutputs, op->get());
        if (!exe) {
            MNN_ERROR("failed to create OpenCL PicScoreAttention execution\n");
            return false;
        }
        auto code = bench.resize(exe.get(), timedInputs, timedOutputs);
        if (code != NO_ERROR) {
            MNN_ERROR("OpenCL PicScoreAttention onResize failed: %d\n", code);
            return false;
        }
        if (useAdrenoSparseRuntimeSingleShot(bench)) {
            auto start = std::chrono::steady_clock::now();
            if (!prepareScoreState()) {
                return false;
            }
            code = bench.execute(exe.get(), timedInputs, timedOutputs);
            if (code != NO_ERROR) {
                MNN_ERROR("OpenCL PicScoreAttention single-shot onExecute failed: %d\n", code);
                return false;
            }
            if (!envFlag("MNN_PAGED_ATTENTION_PROFILE", false) && !bench.sync()) {
                return false;
            }
            auto end = std::chrono::steady_clock::now();
            const float avgMs = static_cast<float>(
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.0f;
            return printPicRuntimeSparseTimedResult(c, avgMs);
        }
        float avgMs = 0.0f;
        if (!measureWithPrepare(prepareScoreState,
                                [&]() { return bench.execute(exe.get(), timedInputs, timedOutputs); },
                                [&]() { return bench.sync(); },
                                c.warmup, c.repeat, &avgMs)) {
            return false;
        }
        return printPicRuntimeSparseTimedResult(c, avgMs);
    } else {
        auto prepareLaterState = [&]() -> bool {
            meta.beginRequest(c.kvLen + 64);
            meta.logical_length = c.kvLen;
            meta.previous = static_cast<size_t>(c.kvLen);
            meta.add = 0;
            meta.remove = 0;
            return meta.beginSparseQuery(logicalIndices, c.layerIndex);
        };
        auto fillQ = bench.tensor({c.batch, c.kvLen, c.qHeads, c.headDim});
        auto fillK = bench.tensor({c.batch, c.kvLen, c.kvHeads, c.headDim});
        auto fillV = bench.tensor({c.batch, c.kvLen, c.kvHeads, c.headDim});
        auto fillO = bench.tensor({c.batch, c.kvLen, c.qHeads, c.headDim});
        if (!fillQ || !fillK || !fillV || !fillO) {
            return false;
        }
        if (!bench.writeTensor(fillQ, makePattern(c.batch * c.kvLen * c.qHeads * c.headDim, 0.0015f, 0.0005f)) ||
            !bench.writeTensor(fillK, makePattern(c.batch * c.kvLen * c.kvHeads * c.headDim, 0.0010f, 0.0015f)) ||
            !bench.writeTensor(fillV, makePattern(c.batch * c.kvLen * c.kvHeads * c.headDim, 0.0020f, -0.0005f))) {
            return false;
        }
        std::vector<Tensor*> fillInputs = {fillQ, fillK, fillV};
        std::vector<Tensor*> fillOutputs = {fillO};
        auto fillOp = makeAttentionOp(OpType_PagedAttention, true, c.layerIndex);
        auto fillExe = bench.create(fillInputs, fillOutputs, fillOp->get());
        if (!fillExe) {
            MNN_ERROR("failed to create OpenCL PagedAttention fill execution\n");
            return false;
        }
        auto code = bench.resize(fillExe.get(), fillInputs, fillOutputs);
        if (code != NO_ERROR) {
            MNN_ERROR("OpenCL PagedAttention fill onResize failed: %d\n", code);
            return false;
        }
        if (bench.execute(fillExe.get(), fillInputs, fillOutputs) != NO_ERROR) {
            return false;
        }
        if (!envFlag("MNN_PAGED_ATTENTION_PROFILE", false) && !bench.sync()) {
            return false;
        }
        if (!prepareLaterState()) {
            return false;
        }
        auto sparseExe = bench.create(timedInputs, timedOutputs, op->get());
        if (!sparseExe) {
            MNN_ERROR("failed to create OpenCL PicSparseAttention sparse execution\n");
            return false;
        }
        code = bench.resize(sparseExe.get(), timedInputs, timedOutputs);
        if (code != NO_ERROR) {
            MNN_ERROR("OpenCL PicSparseAttention sparse onResize failed: %d\n", code);
            return false;
        }
        if (useAdrenoSparseRuntimeSingleShot(bench)) {
            auto start = std::chrono::steady_clock::now();
            if (!prepareLaterState()) {
                return false;
            }
            code = bench.execute(sparseExe.get(), timedInputs, timedOutputs);
            if (code != NO_ERROR) {
                MNN_ERROR("OpenCL PicSparseAttention single-shot onExecute failed: %d\n", code);
                return false;
            }
            if (!envFlag("MNN_PAGED_ATTENTION_PROFILE", false) && !bench.sync()) {
                return false;
            }
            auto end = std::chrono::steady_clock::now();
            const float avgMs = static_cast<float>(
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.0f;
            return printPicRuntimeSparseTimedResult(c, avgMs);
        }
        float avgMs = 0.0f;
        if (!measureWithPrepare(prepareLaterState,
                                [&]() { return bench.execute(sparseExe.get(), timedInputs, timedOutputs); },
                                [&]() { return bench.sync(); },
                                c.warmup, c.repeat, &avgMs)) {
            return false;
        }
        return printPicRuntimeSparseTimedResult(c, avgMs);
    }
}

static std::vector<PicRuntimeSparseBenchCase> picRuntimeSparseBenchCases() {
    return {
        {"MiniCPM5-1B_later_compact_scatter_ctx1024_q519", PicRuntimeSparseBenchMode::LaterCompact,
         1, 16, 2, 128, 1024, 519, 2, true, 3, 20},
        {"MiniCPM5-1B_later_compact_scatter_ctx2560_q269", PicRuntimeSparseBenchMode::LaterCompact,
         1, 16, 2, 128, 2560, 269, 2, true, 3, 16},
        {"MiniCPM5-1B_score_scatter_ctx1024_q519", PicRuntimeSparseBenchMode::ScoreFullQ,
         1, 16, 2, 128, 1024, 519, 1, true, 3, 20},
        {"llama3.2-3B_score_scatter_ctx1024_q519", PicRuntimeSparseBenchMode::ScoreFullQ,
         1, 24, 8, 128, 1024, 519, 1, true, 3, 20},
    };
}

class OpenCLPagedAttentionSparseRuntimePerf : public MNNTestCase {
public:
    bool run(int precision) override {
        (void)precision;
        auto cases = picRuntimeSparseBenchCases();
        for (const auto& c : cases) {
            if (!enabledByFilter("MNN_BENCH_OPENCL_PAGED_SPARSE_CASE", c.name)) {
                continue;
            }
            if (!runPicRuntimeSparseBenchCase(c)) {
                return false;
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(OpenCLPagedAttentionSparseRuntimePerf, "bench_ops/opencl/perf/PagedAttention/SparseRuntime");

struct WeightOnlyConvCase {
    const char* name;
    int rows;
    int ic;
    int oc;
    int quantBlock;
    int warmup;
    int repeat;
};

enum class WeightOnlyForcePath {
    Auto,
    GenericQuant,
    FPWeight
};

enum class WeightOnlyCompactDenseFamily : uint32_t {
    GenericQuant = 0,
    PicQuant = 1,
    FPWeight = 2,
    PicQuantB2 = 3,
    PicQuantC4 = 4,
    PicQuantWG64 = 5,
    PicQuantWG128 = 6,
    AdrenoBatchGemv = 7,
    PicQuantWG4X32 = 8,
    PicQuantWG8X16 = 9,
    PicQuantInputCache32 = 10,
    PicQuantInputCache64 = 11,
};

static std::unique_ptr<OpHolder> makeWeightOnlyLinearConvOp(int ic, int oc, int quantBlock) {
    OpT op;
    op.name = "bench_opencl_weight_only_linear";
    op.type = OpType_Convolution;
    op.defaultDimentionFormat = MNN_DATA_FORMAT_NCHW;
    op.main.type = OpParameter_Convolution2D;
    op.main.value = new Convolution2DT;

    auto* conv = op.main.AsConvolution2D();
    conv->common.reset(new Convolution2DCommonT);
    conv->common->padX = 0;
    conv->common->padY = 0;
    conv->common->kernelX = 1;
    conv->common->kernelY = 1;
    conv->common->strideX = 1;
    conv->common->strideY = 1;
    conv->common->dilateX = 1;
    conv->common->dilateY = 1;
    conv->common->padMode = PadMode_CAFFE;
    conv->common->group = 1;
    conv->common->outputCount = oc;
    conv->common->inputCount = ic;
    conv->common->relu = false;
    conv->common->relu6 = false;
    conv->bias.resize(oc, 0.0f);

    const int groups = UP_DIV(ic, quantBlock);
    std::vector<float> scale(static_cast<size_t>(oc) * static_cast<size_t>(groups), 0.03125f);
    std::vector<int8_t> quantWeight(static_cast<size_t>(oc) * static_cast<size_t>(ic));
    for (size_t i = 0; i < quantWeight.size(); ++i) {
        quantWeight[i] = static_cast<int8_t>((static_cast<int>(i * 13 + 7) & 15) - 8);
    }
    conv->quanParameter = IDSTEncoder::encode(nullptr, scale, ic, oc, false, quantWeight.data(), -8, 4, false);
    conv->quanParameter->aMin = 1;
    conv->quanParameter->readType = 0;
    conv->quanParameter->has_scaleInt = false;
    conv->quanParameter->weightSize = 0;
    return std::unique_ptr<OpHolder>(new OpHolder(op));
}

static bool requireMemoryLowOpenCL(const char* testName) {
    const int memory = MNNTestSuite::get()->pStaus.memory;
    if (memory != BackendConfig::Memory_Low) {
        MNN_PRINT("%s expects memory=2(Memory_Low) so OpenCL creates ConvBufLowMemoryExecution; got memory=%d.\n",
                  testName, memory);
        return false;
    }
    return true;
}

static WeightOnlyForcePath parseWeightOnlyForcePath() {
    const char* value = ::getenv("MNN_BENCH_OPENCL_WEIGHT_ONLY_FORCE");
    if (value == nullptr || value[0] == '\0') {
        return WeightOnlyForcePath::Auto;
    }
    std::string mode(value);
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) { return static_cast<char>(::tolower(c)); });
    if (mode == "0" || mode == "generic" || mode == "quant") {
        return WeightOnlyForcePath::GenericQuant;
    }
    if (mode == "1" || mode == "fp" || mode == "fpweight") {
        return WeightOnlyForcePath::FPWeight;
    }
    return WeightOnlyForcePath::Auto;
}

static const char* weightOnlyForcePathName(WeightOnlyForcePath path) {
    switch (path) {
        case WeightOnlyForcePath::GenericQuant:
            return "generic";
        case WeightOnlyForcePath::FPWeight:
            return "fp_weight";
        default:
            return "auto";
    }
}

static bool usePicCompactConvKey(int rows, int ic, int oc) {
    const int minChannel = std::min(ic, oc);
    return rows > 16 && rows <= 768 && ic >= 1024 && oc >= 1024 && rows <= minChannel;
}

static std::string makeConvSelectionTuneKey(int rows, int ic, int oc) {
    const bool compactLargeChannelSpillRows = rows > 512 && rows <= 576 && ic >= 1024 && oc >= 1024;
    const bool compactMidChannelRetuneRows =
        usePicCompactConvKey(rows, ic, oc) && rows >= 96 && rows < 192 && ic >= 1024 && oc >= 1024;
    std::string key = "convBufLowMemory_" + std::to_string(ic) + "_" + std::to_string(oc);
    if (compactLargeChannelSpillRows) {
        key += "_picspill_v2";
    } else if (compactMidChannelRetuneRows) {
        key += "_picmid_v1";
    }
    return key;
}

static std::string makeAdrenoCompactDenseFamilyTuneKey(int rows, int ic, int oc, int quantBit) {
    return MNN::OpenCL::ConvAdreno::compactDenseFamilyTuneKey(ic, oc, quantBit, rows);
}

static const char* weightOnlyCompactDenseFamilyName(uint32_t family) {
    switch (static_cast<WeightOnlyCompactDenseFamily>(family)) {
        case WeightOnlyCompactDenseFamily::GenericQuant:
            return "generic_quant";
        case WeightOnlyCompactDenseFamily::PicQuant:
            return "pic_quant";
        case WeightOnlyCompactDenseFamily::FPWeight:
            return "fp_weight";
        case WeightOnlyCompactDenseFamily::PicQuantB2:
            return "pic_quant_b2";
        case WeightOnlyCompactDenseFamily::PicQuantC4:
            return "pic_quant_c4";
        case WeightOnlyCompactDenseFamily::PicQuantWG64:
            return "pic_quant_wg64";
        case WeightOnlyCompactDenseFamily::PicQuantWG128:
            return "pic_quant_wg128";
        case WeightOnlyCompactDenseFamily::AdrenoBatchGemv:
            return "adreno_batch_gemv";
        case WeightOnlyCompactDenseFamily::PicQuantWG4X32:
            return "pic_quant_wg4x32";
        case WeightOnlyCompactDenseFamily::PicQuantWG8X16:
            return "pic_quant_wg8x16";
        case WeightOnlyCompactDenseFamily::PicQuantInputCache32:
            return "pic_quant_incache32";
        case WeightOnlyCompactDenseFamily::PicQuantInputCache64:
            return "pic_quant_incache64";
        default:
            return "unknown";
    }
}

static int readSelectedCompactDenseFamily(DirectOpBenchOpenCL& bench, int rows, int ic, int oc, int quantBit) {
    std::pair<std::vector<uint32_t>, uint32_t> tuneInfo;
    if (!MNN::OpenCL::getTunedInfo(makeAdrenoCompactDenseFamilyTuneKey(rows, ic, oc, quantBit),
                                   {static_cast<uint32_t>(rows), static_cast<uint32_t>(oc), static_cast<uint32_t>(ic)},
                                   tuneInfo, bench.runtime()) ||
        tuneInfo.first.empty()) {
        return -1;
    }
    return static_cast<int>(tuneInfo.first[0]);
}

static std::string weightOnlyStorageLabel(DirectOpBenchOpenCL& bench, const WeightOnlyConvCase& c) {
    const char* forcedStorage = ::getenv("MNN_BENCH_OPENCL_WEIGHT_ONLY_FORCE_STORAGE");
    if (forcedStorage != nullptr && forcedStorage[0] != '\0') {
        return forcedStorage;
    }
    const int actualPackCin = 4;
    bool useImage = UP_DIV(c.ic, actualPackCin) <= 16384 && ROUND_UP(c.oc, 8) <= 16384;
    if (MNN::OpenCL::ConvAdreno::preferCompactDenseWeightBuffer(bench.runtime(), 4, c.ic, c.oc)) {
        useImage = false;
    }
    return useImage ? "auto_image" : "auto_buffer";
}

static bool seedForcedConvSelection(DirectOpBenchOpenCL& bench, int rows, int ic, int oc, WeightOnlyForcePath path) {
    if (path == WeightOnlyForcePath::Auto) {
        return true;
    }
    std::pair<std::vector<uint32_t>, uint32_t> tuneInfo;
    tuneInfo.first = {path == WeightOnlyForcePath::FPWeight ? 1u : 0u};
    tuneInfo.second = 0u;
    MNN::OpenCL::setTunedInfo(makeConvSelectionTuneKey(rows, ic, oc), {static_cast<uint32_t>(rows)}, tuneInfo,
                              bench.runtime(), "gemm_conv1x1_buf");
    return true;
}

static int readSelectedConvRoute(DirectOpBenchOpenCL& bench, int rows, int ic, int oc) {
    std::pair<std::vector<uint32_t>, uint32_t> tuneInfo;
    if (!MNN::OpenCL::getTunedInfo(makeConvSelectionTuneKey(rows, ic, oc), {static_cast<uint32_t>(rows)}, tuneInfo,
                                   bench.runtime()) ||
        tuneInfo.first.empty()) {
        return -1;
    }
    return static_cast<int>(tuneInfo.first[0]);
}

static bool printWeightOnlyConvTimedResult(const WeightOnlyConvCase& c, WeightOnlyForcePath forcePath,
                                           int selectedFpWeight, int selectedFamily,
                                           const std::string& storageLabel, float avgMs) {
    const char* forcedFamily = ::getenv("MNN_BENCH_OPENCL_COMPACT_DENSE_FORCE_FAMILY");
    if (forcedFamily == nullptr || forcedFamily[0] == '\0') {
        forcedFamily = "-";
    }
    MNN_PRINT("[bench_ops/opencl/perf/WeightOnlyConv] %-22s rows=%d ic=%d oc=%d qblock=%d force=%-9s "
              "family=%-18s selected_family=%-18s storage=%-12s selected_fp_weight=%d avg=%.4f ms\n",
              c.name, c.rows, c.ic, c.oc, c.quantBlock, weightOnlyForcePathName(forcePath),
              forcedFamily,
              selectedFamily >= 0 ? weightOnlyCompactDenseFamilyName(static_cast<uint32_t>(selectedFamily)) : "-",
              storageLabel.c_str(),
              selectedFpWeight, avgMs);
    ::fflush(stdout);
    return true;
}

static bool runWeightOnlyConvCase(const WeightOnlyConvCase& c, WeightOnlyForcePath forcePath) {
    DirectOpBenchOpenCL bench;
    if (!bench.valid()) {
        return false;
    }
    auto input = bench.tensor({c.rows, c.ic, 1, 1}, Tensor::CAFFE);
    auto output = bench.tensor({c.rows, c.oc, 1, 1}, Tensor::CAFFE);
    if (input == nullptr || output == nullptr) {
        return false;
    }
    if (!bench.writeTensor(input, makePattern(input->elementSize(), 0.0078125f))) {
        return false;
    }
    if (!seedForcedConvSelection(bench, c.rows, c.ic, c.oc, forcePath)) {
        return false;
    }

    auto op = makeWeightOnlyLinearConvOp(c.ic, c.oc, c.quantBlock);
    std::vector<Tensor*> inputs = {input};
    std::vector<Tensor*> outputs = {output};
    auto exe = bench.create(inputs, outputs, op->get());
    if (!exe) {
        MNN_ERROR("failed to create OpenCL WeightOnlyConv execution for %s rows=%d ic=%d oc=%d\n",
                  c.name, c.rows, c.ic, c.oc);
        return false;
    }
    auto code = bench.resize(exe.get(), inputs, outputs);
    if (code != NO_ERROR) {
        MNN_ERROR("OpenCL WeightOnlyConv onResize failed for %s rows=%d ic=%d oc=%d: %d\n",
                  c.name, c.rows, c.ic, c.oc, code);
        return false;
    }
    const int selectedFpWeight = readSelectedConvRoute(bench, c.rows, c.ic, c.oc);
    const int selectedFamily = readSelectedCompactDenseFamily(bench, c.rows, c.ic, c.oc, 4);
    const std::string storageLabel = weightOnlyStorageLabel(bench, c);

    float avgMs = 0.0f;
    std::vector<float> syncOutput;
    OpenCLWallTimer timer;
    if (!timer.measure([&]() { return bench.execute(exe.get(), inputs, outputs); },
                       [&]() { return bench.readTensorTyped<float>(output, &syncOutput); }, c.warmup, c.repeat, &avgMs)) {
        return false;
    }
    return printWeightOnlyConvTimedResult(c, forcePath, selectedFpWeight, selectedFamily, storageLabel, avgMs);
}

static std::vector<WeightOnlyConvCase> weightOnlyConvBenchCases() {
    const int warmup = envInt("MNN_BENCH_OPENCL_WEIGHT_ONLY_WARMUP", 20);
    const int repeat = envInt("MNN_BENCH_OPENCL_WEIGHT_ONLY_REPEAT", 80);
    return {
        {"minicpm_hidden_to_ffn", 269, 1536, 4608, 64, warmup, repeat},
        {"minicpm_hidden_to_ffn", 115, 1536, 4608, 64, warmup, repeat},
        {"minicpm_hidden_to_ffn", 216, 1536, 4608, 64, warmup, repeat},
        {"minicpm_hidden_to_ffn", 317, 1536, 4608, 64, warmup, repeat},
        {"minicpm_hidden_to_ffn", 418, 1536, 4608, 64, warmup, repeat},
        {"minicpm_hidden_to_ffn", 519, 1536, 4608, 64, warmup, repeat},
        {"minicpm_hidden_to_ffn", 828, 1536, 4608, 64, warmup, repeat},
        {"minicpm_hidden_to_ffn", 1031, 1536, 4608, 64, warmup, repeat},
        {"minicpm_hidden_to_ffn", 1287, 1536, 4608, 64, warmup, repeat},
        {"minicpm_ffn_to_hidden", 269, 4608, 1536, 64, warmup, repeat},
        {"minicpm_ffn_to_hidden", 115, 4608, 1536, 64, warmup, repeat},
        {"minicpm_ffn_to_hidden", 216, 4608, 1536, 64, warmup, repeat},
        {"minicpm_ffn_to_hidden", 317, 4608, 1536, 64, warmup, repeat},
        {"minicpm_ffn_to_hidden", 418, 4608, 1536, 64, warmup, repeat},
        {"minicpm_ffn_to_hidden", 519, 4608, 1536, 64, warmup, repeat},
        {"minicpm_ffn_to_hidden", 828, 4608, 1536, 64, warmup, repeat},
        {"minicpm_ffn_to_hidden", 1031, 4608, 1536, 64, warmup, repeat},
        {"minicpm_ffn_to_hidden", 1287, 4608, 1536, 64, warmup, repeat},
        {"minicpm_hidden_to_attn", 269, 1536, 2048, 64, warmup, repeat},
        {"minicpm_hidden_to_attn", 115, 1536, 2048, 64, warmup, repeat},
        {"minicpm_hidden_to_attn", 216, 1536, 2048, 64, warmup, repeat},
        {"minicpm_hidden_to_attn", 317, 1536, 2048, 64, warmup, repeat},
        {"minicpm_hidden_to_attn", 418, 1536, 2048, 64, warmup, repeat},
        {"minicpm_hidden_to_attn", 519, 1536, 2048, 64, warmup, repeat},
        {"minicpm_hidden_to_attn", 828, 1536, 2048, 64, warmup, repeat},
        {"minicpm_hidden_to_attn", 1031, 1536, 2048, 64, warmup, repeat},
        {"minicpm_hidden_to_attn", 1287, 1536, 2048, 64, warmup, repeat},
        {"minicpm_attn_to_hidden", 269, 2048, 1536, 64, warmup, repeat},
        {"minicpm_attn_to_hidden", 115, 2048, 1536, 64, warmup, repeat},
        {"minicpm_attn_to_hidden", 216, 2048, 1536, 64, warmup, repeat},
        {"minicpm_attn_to_hidden", 317, 2048, 1536, 64, warmup, repeat},
        {"minicpm_attn_to_hidden", 418, 2048, 1536, 64, warmup, repeat},
        {"minicpm_attn_to_hidden", 519, 2048, 1536, 64, warmup, repeat},
        {"minicpm_attn_to_hidden", 828, 2048, 1536, 64, warmup, repeat},
        {"minicpm_attn_to_hidden", 1031, 2048, 1536, 64, warmup, repeat},
        {"minicpm_attn_to_hidden", 1287, 2048, 1536, 64, warmup, repeat},
    };
}

class OpenCLWeightOnlyConvPerf : public MNNTestCase {
public:
    bool run(int precision) override {
        (void)precision;
        if (!requireMemoryLowOpenCL("bench_ops/opencl/perf/WeightOnlyConv")) {
            return false;
        }
        const WeightOnlyForcePath forcePath = parseWeightOnlyForcePath();
        const auto cases = weightOnlyConvBenchCases();
        bool ok = true;
        for (const auto& c : cases) {
            if (!enabledByFilter("MNN_BENCH_OPENCL_WEIGHT_ONLY_CASE", c.name) ||
                !enabledRow("MNN_BENCH_OPENCL_WEIGHT_ONLY_ROWS", c.rows)) {
                continue;
            }
            ok = runWeightOnlyConvCase(c, forcePath) && ok;
        }
        return ok;
    }
};

MNNTestSuiteRegister(OpenCLWeightOnlyConvPerf, "bench_ops/opencl/perf/WeightOnlyConv");

enum class SparseQueryPattern {
    Prefix = 0,
    EvenScatter = 1,
};

struct SparseFlashBenchCase {
    const char* name;
    int batch;
    int qHeads;
    int kvHeads;
    int headDim;
    int querySeqLen;
    int outputSeqLen;
    int keySeqLen;
    bool queryRowsAreFull;
    SparseQueryPattern pattern;
    int warmup;
    int repeat;
};

struct SparseFlashBuffers {
    Tensor* query = nullptr;
    Tensor* key = nullptr;
    Tensor* value = nullptr;
    Tensor* sparseQuery = nullptr;
    Tensor* output = nullptr;
    std::shared_ptr<cl::Image2D> keyImage;
    std::shared_ptr<cl::Image2D> valueImage;
    int keyImageWidth = 0;
    int keyImageHeight = 0;
    int valueImageWidth = 0;
    int valueImageHeight = 0;
    std::vector<int32_t> sparseIndices;
};

struct SparseFlashImplResult {
    float avgMs = 0.0f;
    std::vector<float> output;
};

static std::vector<int32_t> makeSparseQueryIndices(int outputSeqLen, int keySeqLen, SparseQueryPattern pattern) {
    std::vector<int32_t> indices(outputSeqLen, 0);
    if (outputSeqLen <= 0 || keySeqLen <= 0) {
        return indices;
    }
    if (pattern == SparseQueryPattern::Prefix || outputSeqLen == 1) {
        for (int i = 0; i < outputSeqLen; ++i) {
            indices[i] = std::min(i, keySeqLen - 1);
        }
        return indices;
    }
    for (int i = 0; i < outputSeqLen; ++i) {
        const double t = outputSeqLen > 1 ? static_cast<double>(i) / static_cast<double>(outputSeqLen - 1) : 0.0;
        int idx = static_cast<int>(std::llround(t * static_cast<double>(keySeqLen - 1)));
        const int minAllowed = i > 0 ? static_cast<int>(indices[i - 1]) + 1 : 0;
        const int maxAllowed = keySeqLen - (outputSeqLen - i);
        idx = std::max(minAllowed, std::min(idx, maxAllowed));
        indices[i] = idx;
    }
    return indices;
}

static bool prepareSparseFlashBuffers(DirectOpBenchOpenCL& bench, const SparseFlashBenchCase& c,
                                      SparseFlashBuffers* buffers) {
    if (buffers == nullptr) {
        return false;
    }
    const int querySeqLen4 = ROUND_UP(c.querySeqLen, 4);
    const int keyMaxLen = ROUND_UP(c.keySeqLen, 4);
    const int valueMaxLen = keyMaxLen;
    const int queryElements = c.batch * c.qHeads * c.headDim * querySeqLen4;
    const int keyElements = c.batch * c.kvHeads * c.headDim * keyMaxLen;
    const int valueElements = c.batch * c.kvHeads * valueMaxLen * c.headDim;
    const int outputElements = c.batch * c.outputSeqLen * c.qHeads * c.headDim;

    buffers->query = bench.tensorTyped<float>({queryElements});
    buffers->key = bench.tensorTyped<float>({keyElements});
    buffers->value = bench.tensorTyped<float>({valueElements});
    buffers->sparseQuery = bench.tensorTyped<int32_t>({c.outputSeqLen});
    buffers->output = bench.tensorTyped<float>({outputElements});
    if (!buffers->query || !buffers->key || !buffers->value || !buffers->sparseQuery || !buffers->output) {
        return false;
    }

    auto queryData = makePattern(queryElements, 0.0025f, 0.0005f);
    auto keyData = makePattern(keyElements, 0.0020f, -0.0007f);
    auto valueData = makePattern(valueElements, 0.0030f, 0.0002f);
    buffers->sparseIndices = makeSparseQueryIndices(c.outputSeqLen, c.keySeqLen, c.pattern);
    if (!bench.writeTensor(buffers->query, queryData) ||
        !bench.writeTensor(buffers->key, keyData) ||
        !bench.writeTensor(buffers->value, valueData) ||
        !bench.writeTensorTyped<int32_t>(buffers->sparseQuery, buffers->sparseIndices) ||
        !bench.zeroTensor(buffers->output)) {
        return false;
    }
    return true;
}

static bool createBenchImageFromBuffer(DirectOpBenchOpenCL& bench, Tensor* buffer, int elementCount,
                                       int imageWidth, int imageHeight,
                                       std::shared_ptr<cl::Image2D>* image, int* width, int* height) {
    if (buffer == nullptr || image == nullptr || width == nullptr || height == nullptr || elementCount <= 0 ||
        imageWidth <= 0 || imageHeight <= 0) {
        return false;
    }
    auto runtime = bench.runtime();
    auto backend = bench.openCLBackend();
    if (runtime == nullptr || backend == nullptr) {
        return false;
    }
    const int pixelCount = UP_DIV(elementCount, 4);
    if (1LL * imageWidth * imageHeight < pixelCount) {
        MNN_ERROR("sparse flash bench image shape too small: pixels=%d width=%d height=%d\n",
                  pixelCount, imageWidth, imageHeight);
        return false;
    }
    const auto maxImage2D = runtime->getMaxImage2DSize();
    if (maxImage2D.size() < 2 || maxImage2D[0] == 0 || maxImage2D[1] == 0) {
        MNN_ERROR("invalid OpenCL max image size for sparse flash bench\n");
        return false;
    }
    if (imageWidth > static_cast<int>(maxImage2D[0]) || imageHeight > static_cast<int>(maxImage2D[1])) {
        MNN_ERROR("sparse flash bench image allocation exceeds device limit: pixels=%d width=%d height=%d max_h=%zu\n",
                  pixelCount, imageWidth, imageHeight, maxImage2D[1]);
        return false;
    }
    cl_int res = CL_SUCCESS;
    image->reset(new cl::Image2D(runtime->context(), CL_MEM_READ_ONLY,
                                 cl::ImageFormat(CL_RGBA, backend->fpType()),
                                 imageWidth, imageHeight, 0, nullptr, &res));
    if (!*image || res != CL_SUCCESS) {
        MNN_ERROR("failed to allocate sparse flash bench image %dx%d, code=%d\n", imageWidth, imageHeight, res);
        return false;
    }
    OpenCL::copyBufferToImage(runtime, OpenCL::openCLBuffer(buffer), **image, imageWidth, imageHeight,
                              backend->getPrecision());
    if (!checkCL(runtime->commandQueue().finish(), "OpenCL sparse flash bench copyBufferToImage finish")) {
        return false;
    }
    *width = imageWidth;
    *height = imageHeight;
    return true;
}

static bool prepareSparseFlashImageBuffers(DirectOpBenchOpenCL& bench, const SparseFlashBenchCase& c,
                                           SparseFlashBuffers* buffers) {
    if (!prepareSparseFlashBuffers(bench, c, buffers)) {
        return false;
    }
    const int keyMaxLen = ROUND_UP(c.keySeqLen, 4);
    const int keyElements = c.batch * c.kvHeads * c.headDim * keyMaxLen;
    const int valueElements = c.batch * c.kvHeads * keyMaxLen * c.headDim;
    const int keyImageWidth = std::max(1, keyMaxLen / 4);
    const int keyImageHeight = c.batch * c.kvHeads * c.headDim;
    const int valueImageWidth = std::max(1, c.headDim / 4);
    const int valueImageHeight = c.batch * c.kvHeads * keyMaxLen;
    return createBenchImageFromBuffer(bench, buffers->key, keyElements,
                                      keyImageWidth, keyImageHeight,
                                      &buffers->keyImage, &buffers->keyImageWidth, &buffers->keyImageHeight) &&
           createBenchImageFromBuffer(bench, buffers->value, valueElements,
                                      valueImageWidth, valueImageHeight,
                                      &buffers->valueImage, &buffers->valueImageWidth, &buffers->valueImageHeight);
}

static bool measureKernelEventMs(OpenCLRuntime* runtime,
                                 const std::function<bool(cl::Event*)>& launch,
                                 int warmup, int repeat, float* avgMs) {
    if (runtime == nullptr || avgMs == nullptr || repeat <= 0) {
        return false;
    }
    for (int i = 0; i < warmup; ++i) {
        cl::Event event;
        if (!launch(&event) || !checkCL(event.wait(), "OpenCL warmup event wait")) {
            return false;
        }
    }
    double totalEventUs = 0.0;
    double totalWallUs = 0.0;
    bool useWallClock = false;
    for (int i = 0; i < repeat; ++i) {
        cl::Event event;
        auto start = std::chrono::steady_clock::now();
        if (!launch(&event) || !checkCL(event.wait(), "OpenCL timed event wait")) {
            return false;
        }
        auto end = std::chrono::steady_clock::now();
        const double eventUs = runtime->getCostTime(&event);
        const double wallUs = static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        totalWallUs += wallUs;
        if (eventUs > 0.0) {
            totalEventUs += eventUs;
        } else {
            useWallClock = true;
        }
    }
    const double totalUs = useWallClock ? totalWallUs : totalEventUs;
    *avgMs = static_cast<float>(totalUs / (1000.0 * static_cast<double>(repeat)));
    return true;
}

static bool setSparseFlashKernelArgs(const std::shared_ptr<KernelWrap>& kernel,
                                     const std::vector<uint32_t>& gws,
                                     const SparseFlashBuffers& buffers,
                                     const SparseFlashBenchCase& c) {
    if (!kernel) {
        return false;
    }
    const int keyMaxLen = ROUND_UP(c.keySeqLen, 4);
    const int valueMaxLen = keyMaxLen;
    const float scale = 1.0f / std::sqrt(static_cast<float>(c.headDim));
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    ret |= kernel->get().setArg(idx++, gws[0]);
    ret |= kernel->get().setArg(idx++, gws[1]);
    ret |= kernel->get().setArg(idx++, gws[2]);
    ret |= kernel->get().setArg(idx++, OpenCL::openCLBuffer(buffers.query));
    ret |= kernel->get().setArg(idx++, OpenCL::openCLBuffer(buffers.key));
    ret |= kernel->get().setArg(idx++, OpenCL::openCLBuffer(buffers.value));
    ret |= kernel->get().setArg(idx++, OpenCL::openCLBuffer(buffers.sparseQuery));
    ret |= kernel->get().setArg(idx++, OpenCL::openCLBuffer(buffers.output));
    ret |= kernel->get().setArg(idx++, scale);
    ret |= kernel->get().setArg(idx++, c.querySeqLen);
    ret |= kernel->get().setArg(idx++, c.outputSeqLen);
    ret |= kernel->get().setArg(idx++, c.queryRowsAreFull ? 1 : 0);
    ret |= kernel->get().setArg(idx++, 0);
    ret |= kernel->get().setArg(idx++, c.outputSeqLen);
    ret |= kernel->get().setArg(idx++, c.keySeqLen);
    ret |= kernel->get().setArg(idx++, keyMaxLen);
    ret |= kernel->get().setArg(idx++, valueMaxLen);
    ret |= kernel->get().setArg(idx++, c.qHeads);
    ret |= kernel->get().setArg(idx++, c.kvHeads);
    ret |= kernel->get().setArg(idx++, c.headDim);
    return checkCL(ret, "setArg sparse flash bench");
}

static float maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return INFINITY;
    }
    float maxDiff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        maxDiff = std::max(maxDiff, std::fabs(a[i] - b[i]));
    }
    return maxDiff;
}

static const char* sparseFlashBenchProgramName(const char* kernelName) {
    if (kernelName == nullptr) {
        return nullptr;
    }
    if (::strcmp(kernelName, "sparse_flash_attention_row32_probe") == 0) {
        return "sparse_flash_row_probe_bench";
    }
    if (::strcmp(kernelName, "sparse_flash_attention_row32") == 0 ||
        ::strcmp(kernelName, "sparse_flash_attention_row64") == 0) {
        return "sparse_flash_row_bench";
    }
    if (::strcmp(kernelName, "mqtile_sparse_flash_hd64_q4k16") == 0 ||
        ::strcmp(kernelName, "mqtile_sparse_flash_hd128_q4k16") == 0 ||
        ::strcmp(kernelName, "mqtile_sparse_flash_hd128_q4k8") == 0 ||
        ::strcmp(kernelName, "mqtile_sparse_flash_hd128_q8k16") == 0) {
        return "sparse_flash_qtile_bench";
    }
    if (::strcmp(kernelName, "mqtile_sparse_flash_hd128_q4k8_kimg") == 0 ||
        ::strcmp(kernelName, "mqtile_sparse_flash_hd128_q8k16_kimg") == 0) {
        return "sparse_flash_qtile_kimg_bench";
    }
    if (::strcmp(kernelName, "mqtile_sparse_flash_hd128_q4k8_kvimg") == 0 ||
        ::strcmp(kernelName, "mqtile_sparse_flash_hd128_q8k16_kvimg") == 0) {
        return "sparse_flash_qtile_kvimg_bench";
    }
    return "attention_buf";
}

static const char* sparseFlashInlineSourceForKernel(const char* kernelName) {
    if (kernelName == nullptr) {
        return nullptr;
    }
    if (::strcmp(kernelName, "sparse_flash_inline_build_only_probe") == 0) {
        return R"CLC(
__kernel void sparse_flash_inline_build_only_probe() {
}
)CLC";
    }
    if (::strcmp(kernelName, "sparse_flash_inline_output_only_probe") == 0) {
        return R"CLC(
#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,
#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }
__kernel void sparse_flash_inline_output_only_probe(GLOBAL_SIZE_3_DIMS
                              __global FLOAT *output,
                              __private const int output_elements) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    const int index = (z * global_size_dim1 + y) * global_size_dim0 + x;
    if (index < output_elements) {
        output[index] = (FLOAT)0;
    }
}
)CLC";
    }
    if (::strcmp(kernelName, "sparse_flash_inline_row32_sig_probe") == 0) {
        return R"CLC(
#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,
#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }
#ifndef NUMHEAD_GROUP_SIZE
#define NUMHEAD_GROUP_SIZE 1
#endif
__kernel void sparse_flash_inline_row32_sig_probe(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *query,
                              __global const FLOAT *past_key,
                              __global const FLOAT *past_value,
                              __global const int *sparse_query,
                              __global FLOAT *output,
                              __private const float scale,
                              __private const int query_seq_len,
                              __private const int output_seq_len,
                              __private const int query_rows_are_full,
                              __private const int q_start,
                              __private const int q_piece_len,
                              __private const int key_seq_len,
                              __private const int key_max_len,
                              __private const int value_max_len,
                              __private const int head_num,
                              __private const int kv_head_num,
                              __private const int head_dim) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    int z = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    const int lid = get_local_id(0);
    if (get_local_size(0) != 32 || lid >= 32 || (head_dim != 64 && head_dim != 128)) {
        return;
    }
    const int q_index = q_start + y;
    if (y >= q_piece_len || q_index >= output_seq_len) {
        return;
    }
    const int b = z / head_num;
    const int h = z - b * head_num;
    const int kvh = h / NUMHEAD_GROUP_SIZE;
    if (kvh >= kv_head_num) {
        return;
    }
    if (lid == 0) {
        const int output_offset = ((b * output_seq_len + q_index) * head_num + h) * head_dim;
        output[output_offset] = (FLOAT)0;
    }
}
)CLC";
    }
    return nullptr;
}

static std::shared_ptr<KernelWrap> buildSparseFlashBenchKernel(DirectOpBenchOpenCL& bench,
                                                               OpenCLRuntime* runtime,
                                                               const SparseFlashBenchCase& c,
                                                               const char* kernelName) {
    if (runtime == nullptr || kernelName == nullptr) {
        return nullptr;
    }
    std::set<std::string> buildOptions;
    buildOptions.emplace("-DNUMHEAD_GROUP_SIZE=" + std::to_string(c.qHeads / c.kvHeads));
    if (const char* inlineSource = sparseFlashInlineSourceForKernel(kernelName)) {
        return runtime->buildKernelFromSource(inlineSource, kernelName, buildOptions,
                                              bench.openCLBackend()->getPrecision());
    }
    return runtime->buildKernel(sparseFlashBenchProgramName(kernelName), kernelName, buildOptions,
                                bench.openCLBackend()->getPrecision());
}

static bool sparseFlashImplEnabled(const char* label) {
    const char* filter = ::getenv("MNN_BENCH_OPENCL_SPARSE_FLASH_IMPL");
    if (filter == nullptr || filter[0] == '\0') {
        const std::string name = label == nullptr ? "" : std::string(label);
        const bool isProbe = name.find("probe") != std::string::npos;
        return !isProbe;
    }
    return std::string(label).find(filter) != std::string::npos;
}

static bool sparseFlashAllowNoReference() {
    const char* value = ::getenv("MNN_BENCH_OPENCL_SPARSE_FLASH_ALLOW_NO_REF");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool printSparseFlashTimedResult(const SparseFlashBenchCase& c, const char* impl, float avgMs,
                                        float row32AvgMs, float maxAbs);

static bool runSparseFlashBuildOnlyProbe(DirectOpBenchOpenCL& bench, const SparseFlashBenchCase& c,
                                         const char* kernelName, const char* label) {
    auto runtime = bench.runtime();
    if (runtime == nullptr) {
        return false;
    }
    auto kernel = buildSparseFlashBenchKernel(bench, runtime, c, kernelName);
    if (!kernel) {
        MNN_ERROR("failed to build OpenCL sparse flash build-only probe: %s\n", kernelName);
        return false;
    }
    return printSparseFlashTimedResult(c, label, 0.0f, 0.0f, 0.0f);
}

static bool runStandardOpenCLBuildOnlyProbe(DirectOpBenchOpenCL& bench, const SparseFlashBenchCase& c,
                                            const char* label) {
    auto runtime = bench.runtime();
    if (runtime == nullptr) {
        return false;
    }
    std::set<std::string> buildOptions;
    buildOptions.emplace("-DARGMAX");
    buildOptions.emplace("-DARGMAX_LOCAL_SIZE=4");
    auto kernel = runtime->buildKernel("argmax_buf", "argmax_buf", buildOptions,
                                       bench.openCLBackend()->getPrecision());
    if (!kernel) {
        MNN_ERROR("failed to build standard OpenCL probe kernel: argmax_buf\n");
        return false;
    }
    return printSparseFlashTimedResult(c, label, 0.0f, 0.0f, 0.0f);
}

static bool runSparseFlashImpl(DirectOpBenchOpenCL& bench, const SparseFlashBenchCase& c,
                               const SparseFlashBuffers& buffers, const char* kernelName,
                               const std::vector<uint32_t>& gws, const std::vector<uint32_t>& lws,
                               SparseFlashImplResult* result) {
    if (result == nullptr) {
        return false;
    }
    auto runtime = bench.runtime();
    if (runtime == nullptr) {
        return false;
    }
    if (c.kvHeads <= 0 || c.qHeads <= 0 || c.qHeads % c.kvHeads != 0) {
        MNN_ERROR("invalid sparse flash head grouping: qHeads=%d kvHeads=%d\n", c.qHeads, c.kvHeads);
        return false;
    }
    auto kernel = buildSparseFlashBenchKernel(bench, runtime, c, kernelName);
    if (!kernel) {
        MNN_ERROR("failed to build OpenCL sparse flash bench kernel: %s\n", kernelName);
        return false;
    }
    if (::strcmp(kernelName, "sparse_flash_inline_output_only_probe") == 0) {
        cl_int ret = CL_SUCCESS;
        uint32_t idx = 0;
        ret |= kernel->get().setArg(idx++, gws[0]);
        ret |= kernel->get().setArg(idx++, gws[1]);
        ret |= kernel->get().setArg(idx++, gws[2]);
        ret |= kernel->get().setArg(idx++, OpenCL::openCLBuffer(buffers.output));
        ret |= kernel->get().setArg(idx++, static_cast<int>(buffers.output->elementSize()));
        if (!checkCL(ret, "setArg sparse flash inline output-only probe")) {
            return false;
        }
    } else if (!setSparseFlashKernelArgs(kernel, gws, buffers, c)) {
        return false;
    }
    auto launch = [&](cl::Event* event) -> bool {
        OpenCL::run3DKernelDefault(kernel, gws, lws, runtime, event);
        return true;
    };
    if (!bench.zeroTensor(buffers.output)) {
        return false;
    }
    cl::Event refEvent;
    if (!launch(&refEvent) || !checkCL(refEvent.wait(), "OpenCL sparse flash reference wait")) {
        return false;
    }
    if (!bench.readTensorTyped<float>(buffers.output, &result->output)) {
        return false;
    }
    if (!measureKernelEventMs(runtime, launch, c.warmup, c.repeat, &result->avgMs)) {
        return false;
    }
    return true;
}

static bool setSparseFlashImageKVKernelArgs(const std::shared_ptr<KernelWrap>& kernel,
                                            const std::vector<uint32_t>& gws,
                                            const SparseFlashBuffers& buffers,
                                            const SparseFlashBenchCase& c) {
    if (!kernel || !buffers.keyImage || !buffers.valueImage) {
        return false;
    }
    const int keyMaxLen = ROUND_UP(c.keySeqLen, 4);
    const int valueMaxLen = keyMaxLen;
    const float scale = 1.0f / std::sqrt(static_cast<float>(c.headDim));
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    ret |= kernel->get().setArg(idx++, gws[0]);
    ret |= kernel->get().setArg(idx++, gws[1]);
    ret |= kernel->get().setArg(idx++, gws[2]);
    ret |= kernel->get().setArg(idx++, OpenCL::openCLBuffer(buffers.query));
    ret |= kernel->get().setArg(idx++, *buffers.keyImage);
    ret |= kernel->get().setArg(idx++, *buffers.valueImage);
    ret |= kernel->get().setArg(idx++, OpenCL::openCLBuffer(buffers.sparseQuery));
    ret |= kernel->get().setArg(idx++, OpenCL::openCLBuffer(buffers.output));
    ret |= kernel->get().setArg(idx++, scale);
    ret |= kernel->get().setArg(idx++, c.querySeqLen);
    ret |= kernel->get().setArg(idx++, c.outputSeqLen);
    ret |= kernel->get().setArg(idx++, c.queryRowsAreFull ? 1 : 0);
    ret |= kernel->get().setArg(idx++, 0);
    ret |= kernel->get().setArg(idx++, c.outputSeqLen);
    ret |= kernel->get().setArg(idx++, c.keySeqLen);
    ret |= kernel->get().setArg(idx++, keyMaxLen);
    ret |= kernel->get().setArg(idx++, valueMaxLen);
    ret |= kernel->get().setArg(idx++, buffers.keyImageWidth);
    ret |= kernel->get().setArg(idx++, buffers.valueImageWidth);
    ret |= kernel->get().setArg(idx++, c.qHeads);
    ret |= kernel->get().setArg(idx++, c.kvHeads);
    ret |= kernel->get().setArg(idx++, c.headDim);
    return checkCL(ret, "setArg sparse flash image-kv bench");
}

static bool setSparseFlashImageKeyKernelArgs(const std::shared_ptr<KernelWrap>& kernel,
                                             const std::vector<uint32_t>& gws,
                                             const SparseFlashBuffers& buffers,
                                             const SparseFlashBenchCase& c) {
    if (!kernel || !buffers.keyImage) {
        return false;
    }
    const int keyMaxLen = ROUND_UP(c.keySeqLen, 4);
    const int valueMaxLen = keyMaxLen;
    const float scale = 1.0f / std::sqrt(static_cast<float>(c.headDim));
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    ret |= kernel->get().setArg(idx++, gws[0]);
    ret |= kernel->get().setArg(idx++, gws[1]);
    ret |= kernel->get().setArg(idx++, gws[2]);
    ret |= kernel->get().setArg(idx++, OpenCL::openCLBuffer(buffers.query));
    ret |= kernel->get().setArg(idx++, *buffers.keyImage);
    ret |= kernel->get().setArg(idx++, OpenCL::openCLBuffer(buffers.value));
    ret |= kernel->get().setArg(idx++, OpenCL::openCLBuffer(buffers.sparseQuery));
    ret |= kernel->get().setArg(idx++, OpenCL::openCLBuffer(buffers.output));
    ret |= kernel->get().setArg(idx++, scale);
    ret |= kernel->get().setArg(idx++, c.querySeqLen);
    ret |= kernel->get().setArg(idx++, c.outputSeqLen);
    ret |= kernel->get().setArg(idx++, c.queryRowsAreFull ? 1 : 0);
    ret |= kernel->get().setArg(idx++, 0);
    ret |= kernel->get().setArg(idx++, c.outputSeqLen);
    ret |= kernel->get().setArg(idx++, c.keySeqLen);
    ret |= kernel->get().setArg(idx++, keyMaxLen);
    ret |= kernel->get().setArg(idx++, valueMaxLen);
    ret |= kernel->get().setArg(idx++, buffers.keyImageWidth);
    ret |= kernel->get().setArg(idx++, c.qHeads);
    ret |= kernel->get().setArg(idx++, c.kvHeads);
    ret |= kernel->get().setArg(idx++, c.headDim);
    return checkCL(ret, "setArg sparse flash image-key bench");
}

static bool runSparseFlashImageKVImpl(DirectOpBenchOpenCL& bench, const SparseFlashBenchCase& c,
                                      const SparseFlashBuffers& buffers, const char* kernelName,
                                      const std::vector<uint32_t>& gws, const std::vector<uint32_t>& lws,
                                      SparseFlashImplResult* result) {
    if (result == nullptr) {
        return false;
    }
    auto runtime = bench.runtime();
    if (runtime == nullptr) {
        return false;
    }
    if (c.kvHeads <= 0 || c.qHeads <= 0 || c.qHeads % c.kvHeads != 0) {
        MNN_ERROR("invalid sparse flash image-kv head grouping: qHeads=%d kvHeads=%d\n", c.qHeads, c.kvHeads);
        return false;
    }
    std::set<std::string> buildOptions;
    buildOptions.emplace("-DNUMHEAD_GROUP_SIZE=" + std::to_string(c.qHeads / c.kvHeads));
    auto kernel = runtime->buildKernel(sparseFlashBenchProgramName(kernelName), kernelName, buildOptions,
                                       bench.openCLBackend()->getPrecision());
    if (!kernel) {
        MNN_ERROR("failed to build OpenCL sparse flash image-kv bench kernel: %s\n", kernelName);
        return false;
    }
    if (!setSparseFlashImageKVKernelArgs(kernel, gws, buffers, c)) {
        return false;
    }
    auto launch = [&](cl::Event* event) -> bool {
        OpenCL::run3DKernelDefault(kernel, gws, lws, runtime, event);
        return true;
    };
    if (!bench.zeroTensor(buffers.output)) {
        return false;
    }
    cl::Event refEvent;
    if (!launch(&refEvent) || !checkCL(refEvent.wait(), "OpenCL sparse flash image-kv reference wait")) {
        return false;
    }
    if (!bench.readTensorTyped<float>(buffers.output, &result->output)) {
        return false;
    }
    if (!measureKernelEventMs(runtime, launch, c.warmup, c.repeat, &result->avgMs)) {
        return false;
    }
    return true;
}

static bool runSparseFlashImageKeyImpl(DirectOpBenchOpenCL& bench, const SparseFlashBenchCase& c,
                                       const SparseFlashBuffers& buffers, const char* kernelName,
                                       const std::vector<uint32_t>& gws, const std::vector<uint32_t>& lws,
                                       SparseFlashImplResult* result) {
    if (result == nullptr) {
        return false;
    }
    auto runtime = bench.runtime();
    if (runtime == nullptr) {
        return false;
    }
    if (c.kvHeads <= 0 || c.qHeads <= 0 || c.qHeads % c.kvHeads != 0) {
        MNN_ERROR("invalid sparse flash image-key head grouping: qHeads=%d kvHeads=%d\n", c.qHeads, c.kvHeads);
        return false;
    }
    std::set<std::string> buildOptions;
    buildOptions.emplace("-DNUMHEAD_GROUP_SIZE=" + std::to_string(c.qHeads / c.kvHeads));
    auto kernel = runtime->buildKernel(sparseFlashBenchProgramName(kernelName), kernelName, buildOptions,
                                       bench.openCLBackend()->getPrecision());
    if (!kernel) {
        MNN_ERROR("failed to build OpenCL sparse flash image-key bench kernel: %s\n", kernelName);
        return false;
    }
    if (!setSparseFlashImageKeyKernelArgs(kernel, gws, buffers, c)) {
        return false;
    }
    auto launch = [&](cl::Event* event) -> bool {
        OpenCL::run3DKernelDefault(kernel, gws, lws, runtime, event);
        return true;
    };
    if (!bench.zeroTensor(buffers.output)) {
        return false;
    }
    cl::Event refEvent;
    if (!launch(&refEvent) || !checkCL(refEvent.wait(), "OpenCL sparse flash image-key reference wait")) {
        return false;
    }
    if (!bench.readTensorTyped<float>(buffers.output, &result->output)) {
        return false;
    }
    if (!measureKernelEventMs(runtime, launch, c.warmup, c.repeat, &result->avgMs)) {
        return false;
    }
    return true;
}

static bool printSparseFlashTimedResult(const SparseFlashBenchCase& c, const char* impl, float avgMs,
                                        float row32AvgMs, float maxAbs) {
    const float speedupVsRow32 = row32AvgMs > 0.0f ? row32AvgMs / avgMs : 0.0f;
    MNN_PRINT("[bench_ops/opencl/perf/SparseFlash/MultiQTile] %-30s impl=%-30s "
              "full_q=%d q=%d kv=%d qH=%d kvH=%d D=%d avg=%.4f ms speedup_vs_row32=%.3f max_abs=%.6f\n",
              c.name, impl, c.queryRowsAreFull ? 1 : 0, c.outputSeqLen, c.keySeqLen,
              c.qHeads, c.kvHeads, c.headDim, avgMs, speedupVsRow32, maxAbs);
    ::fflush(stdout);
    return true;
}

static std::vector<SparseFlashBenchCase> sparseFlashBenchCases() {
    return {
        {"llama3.2-1B_later_prefix_ctx1024_q519", 1, 32, 8, 64, 519, 519, 1024, false,
         SparseQueryPattern::Prefix, 5, 30},
        {"llama3.2-1B_later_scatter_ctx1024_q519", 1, 32, 8, 64, 519, 519, 1024, false,
         SparseQueryPattern::EvenScatter, 5, 30},
        {"llama3.2-1B_score_scatter_ctx1024_q519", 1, 32, 8, 64, 1024, 519, 1024, true,
         SparseQueryPattern::EvenScatter, 5, 30},
        {"llama3.2-3B_later_prefix_ctx1024_q418", 1, 24, 8, 128, 418, 418, 1024, false,
         SparseQueryPattern::Prefix, 5, 30},
        {"llama3.2-3B_later_scatter_ctx1024_q418", 1, 24, 8, 128, 418, 418, 1024, false,
         SparseQueryPattern::EvenScatter, 5, 30},
        {"llama3.2-3B_later_prefix_ctx1024_q519", 1, 24, 8, 128, 519, 519, 1024, false,
         SparseQueryPattern::Prefix, 5, 30},
        {"llama3.2-3B_later_scatter_ctx1024_q519", 1, 24, 8, 128, 519, 519, 1024, false,
         SparseQueryPattern::EvenScatter, 5, 30},
        {"llama3.2-3B_score_scatter_ctx1024_q519", 1, 24, 8, 128, 1024, 519, 1024, true,
         SparseQueryPattern::EvenScatter, 5, 30},
        {"MiniCPM5-1B_later_prefix_ctx1024_q418", 1, 16, 2, 128, 418, 418, 1024, false,
         SparseQueryPattern::Prefix, 5, 30},
        {"MiniCPM5-1B_later_scatter_ctx1024_q418", 1, 16, 2, 128, 418, 418, 1024, false,
         SparseQueryPattern::EvenScatter, 5, 30},
        {"MiniCPM5-1B_later_prefix_ctx1024_q519", 1, 16, 2, 128, 519, 519, 1024, false,
         SparseQueryPattern::Prefix, 5, 30},
        {"MiniCPM5-1B_later_scatter_ctx1024_q519", 1, 16, 2, 128, 519, 519, 1024, false,
         SparseQueryPattern::EvenScatter, 5, 30},
        {"qwen3-8B_later_scatter_ctx1024_q519", 1, 32, 8, 128, 519, 519, 1024, false,
         SparseQueryPattern::EvenScatter, 5, 30},
        {"MiniCPM5-1B_later_scatter_ctx1522_q761", 1, 16, 2, 128, 761, 761, 1522, false,
         SparseQueryPattern::EvenScatter, 4, 20},
        {"MiniCPM5-1B_score_scatter_ctx1536_q319", 1, 16, 2, 128, 1536, 319, 1536, true,
         SparseQueryPattern::EvenScatter, 4, 20},
        {"llama3.2-3B_score_scatter_ctx1536_q319", 1, 24, 8, 128, 1536, 319, 1536, true,
         SparseQueryPattern::EvenScatter, 4, 20},
        {"MiniCPM5-1B_later_scatter_ctx2560_q269", 1, 16, 2, 128, 269, 269, 2560, false,
         SparseQueryPattern::EvenScatter, 4, 30},
        {"MiniCPM5-1B_score_scatter_ctx2560_q269", 1, 16, 2, 128, 2560, 269, 2560, true,
         SparseQueryPattern::EvenScatter, 4, 20},
    };
}

static bool runSparseFlashBenchCase(const SparseFlashBenchCase& c) {
    DirectOpBenchOpenCL bench;
    if (!bench.valid()) {
        return false;
    }
    if (bench.runtime() != nullptr && bench.runtime()->getGpuType() == ADRENO) {
        MNN_PRINT("[bench_ops/opencl/perf/SparseFlash/MultiQTile] skip %-30s on Adreno: "
                  "clCreateProgramWithSource is unstable in wrapped run_test.out for this bench; "
                  "use integrated paged-attention request profiling for variant attribution.\n",
                  c.name);
        ::fflush(stdout);
        return true;
    }
    if (sparseFlashImplEnabled("inline_build_only_probe")) {
        if (!runSparseFlashBuildOnlyProbe(bench, c, "sparse_flash_inline_build_only_probe",
                                          "inline_build_only_probe")) {
            return false;
        }
    }
    if (sparseFlashImplEnabled("standard_argmax_build_only_probe")) {
        if (!runStandardOpenCLBuildOnlyProbe(bench, c, "standard_argmax_build_only_probe")) {
            return false;
        }
    }
    SparseFlashBuffers buffers;
    if (!prepareSparseFlashBuffers(bench, c, &buffers)) {
        return false;
    }

    SparseFlashImplResult row32;
    const std::vector<uint32_t> row32Gws = {32u, static_cast<uint32_t>(c.outputSeqLen),
                                            static_cast<uint32_t>(c.batch * c.qHeads)};
    if (sparseFlashImplEnabled("inline_output_only_probe")) {
        SparseFlashImplResult probe;
        if (!runSparseFlashImpl(bench, c, buffers, "sparse_flash_inline_output_only_probe", row32Gws,
                                {32u, 1u, 1u}, &probe)) {
            return false;
        }
        if (!printSparseFlashTimedResult(c, "inline_output_only_probe", probe.avgMs, probe.avgMs, 0.0f)) {
            return false;
        }
    }
    if (sparseFlashImplEnabled("inline_row32_sig_probe")) {
        SparseFlashImplResult probe;
        if (!runSparseFlashImpl(bench, c, buffers, "sparse_flash_inline_row32_sig_probe", row32Gws,
                                {32u, 1u, 1u}, &probe)) {
            return false;
        }
        if (!printSparseFlashTimedResult(c, "inline_row32_sig_probe", probe.avgMs, probe.avgMs, 0.0f)) {
            return false;
        }
    }
    if (sparseFlashImplEnabled("current_row32_probe")) {
        SparseFlashImplResult probe;
        if (!runSparseFlashImpl(bench, c, buffers, "sparse_flash_attention_row32_probe", row32Gws, {32u, 1u, 1u},
                                &probe)) {
            return false;
        }
        if (!printSparseFlashTimedResult(c, "current_row32_probe", probe.avgMs, probe.avgMs, 0.0f)) {
            return false;
        }
    }
    if (sparseFlashImplEnabled("current_row32")) {
        if (!runSparseFlashImpl(bench, c, buffers, "sparse_flash_attention_row32", row32Gws, {32u, 1u, 1u}, &row32)) {
            return false;
        }
        if (!printSparseFlashTimedResult(c, "current_row32", row32.avgMs, row32.avgMs, 0.0f)) {
            return false;
        }
    }

    SparseFlashImplResult row64;
    const std::vector<uint32_t> row64Gws = {64u, static_cast<uint32_t>(c.outputSeqLen),
                                            static_cast<uint32_t>(c.batch * c.qHeads)};
    if (sparseFlashImplEnabled("current_row64")) {
        if (!runSparseFlashImpl(bench, c, buffers, "sparse_flash_attention_row64", row64Gws, {64u, 1u, 1u}, &row64)) {
            return false;
        }
        const float diff = row32.output.empty() ? 0.0f : maxAbsDiff(row32.output, row64.output);
        const float row32Avg = row32.avgMs > 0.0f ? row32.avgMs : row64.avgMs;
        if (!printSparseFlashTimedResult(c, "current_row64", row64.avgMs, row32Avg, diff)) {
            return false;
        }
    }

    const bool needRow32Reference =
        sparseFlashImplEnabled("mqtile_hd64_q4k16") ||
        sparseFlashImplEnabled("mqtile_hd128_q4k16") ||
        sparseFlashImplEnabled("mqtile_hd128_q4k8") ||
        sparseFlashImplEnabled("mqtile_hd128_q4k8_kimg") ||
        sparseFlashImplEnabled("mqtile_hd128_q4k8_kvimg") ||
        sparseFlashImplEnabled("mqtile_hd128_q8k16_kimg") ||
        sparseFlashImplEnabled("mqtile_hd128_q8k16_kvimg") ||
        sparseFlashImplEnabled("mqtile_hd128_q8k16");
    if (needRow32Reference && row32.output.empty()) {
        if (sparseFlashAllowNoReference()) {
            MNN_PRINT("[bench_ops/opencl/perf/SparseFlash/MultiQTile] case=%s allow_no_ref=1, skip current_row32 reference\n",
                      c.name);
        } else {
        MNN_ERROR("current_row32 must run before qtile sparse flash bench: case=%s\n", c.name);
        return false;
        }
    }

    if (c.headDim != 64 && c.headDim != 128) {
        MNN_ERROR("unsupported headDim for sparse flash mqtile bench: %d\n", c.headDim);
        return false;
    }
    auto runMqtile = [&](const char* kernelName, const char* label,
                         std::vector<uint32_t> gws, std::vector<uint32_t> lws) -> bool {
        if (!sparseFlashImplEnabled(label)) {
            return true;
        }
        SparseFlashImplResult result;
        if (!runSparseFlashImpl(bench, c, buffers, kernelName, gws, lws, &result)) {
            return false;
        }
        const bool hasReference = !row32.output.empty();
        const float diff = hasReference ? maxAbsDiff(row32.output, result.output) : 0.0f;
        if (hasReference && std::isfinite(diff) && diff > 1.0e-1f) {
            MNN_ERROR("sparse flash mqtile output mismatch too large: case=%s impl=%s max_abs=%f\n",
                      c.name, label, diff);
            return false;
        }
        const float refAvgMs = row32.avgMs > 0.0f ? row32.avgMs : result.avgMs;
        return printSparseFlashTimedResult(c, label, result.avgMs, refAvgMs, diff);
    };

    if (c.headDim == 64) {
        return runMqtile("mqtile_sparse_flash_hd64_q4k16", "mqtile_hd64_q4k16",
                         {16u, static_cast<uint32_t>(ROUND_UP(c.outputSeqLen, 4)),
                          static_cast<uint32_t>(c.batch * c.qHeads)},
                         {16u, 4u, 1u});
    }

    if (!runMqtile("mqtile_sparse_flash_hd128_q4k16", "mqtile_hd128_q4k16",
                   {16u, static_cast<uint32_t>(ROUND_UP(c.outputSeqLen, 4)),
                    static_cast<uint32_t>(c.batch * c.qHeads)},
                   {16u, 4u, 1u})) {
        return false;
    }
    if (!runMqtile("mqtile_sparse_flash_hd128_q4k8", "mqtile_hd128_q4k8",
                   {16u, static_cast<uint32_t>(ROUND_UP(c.outputSeqLen, 4)),
                    static_cast<uint32_t>(c.batch * c.qHeads)},
                   {16u, 4u, 1u})) {
        return false;
    }
    if (sparseFlashImplEnabled("mqtile_hd128_q4k8_kimg")) {
        SparseFlashBuffers imageBuffers;
        if (!prepareSparseFlashImageBuffers(bench, c, &imageBuffers)) {
            return false;
        }
        SparseFlashImplResult imageKey;
        if (!runSparseFlashImageKeyImpl(bench, c, imageBuffers, "mqtile_sparse_flash_hd128_q4k8_kimg",
                                        {16u, static_cast<uint32_t>(ROUND_UP(c.outputSeqLen, 4)),
                                         static_cast<uint32_t>(c.batch * c.qHeads)},
                                        {16u, 4u, 1u}, &imageKey)) {
            return false;
        }
        const bool hasReference = !row32.output.empty();
        const float diff = hasReference ? maxAbsDiff(row32.output, imageKey.output) : 0.0f;
        if (hasReference && std::isfinite(diff) && diff > 1.0e-1f) {
            MNN_ERROR("sparse flash image-key output mismatch too large: case=%s impl=%s max_abs=%f\n",
                      c.name, "mqtile_hd128_q4k8_kimg", diff);
            return false;
        }
        const float refAvgMs = row32.avgMs > 0.0f ? row32.avgMs : imageKey.avgMs;
        if (!printSparseFlashTimedResult(c, "mqtile_hd128_q4k8_kimg", imageKey.avgMs, refAvgMs, diff)) {
            return false;
        }
    }
    if (sparseFlashImplEnabled("mqtile_hd128_q4k8_kvimg")) {
        SparseFlashBuffers imageBuffers;
        if (!prepareSparseFlashImageBuffers(bench, c, &imageBuffers)) {
            return false;
        }
        SparseFlashImplResult imageKV;
        if (!runSparseFlashImageKVImpl(bench, c, imageBuffers, "mqtile_sparse_flash_hd128_q4k8_kvimg",
                                       {16u, static_cast<uint32_t>(ROUND_UP(c.outputSeqLen, 4)),
                                        static_cast<uint32_t>(c.batch * c.qHeads)},
                                       {16u, 4u, 1u}, &imageKV)) {
            return false;
        }
        const bool hasReference = !row32.output.empty();
        const float diff = hasReference ? maxAbsDiff(row32.output, imageKV.output) : 0.0f;
        if (hasReference && std::isfinite(diff) && diff > 1.0e-1f) {
            MNN_ERROR("sparse flash image-kv output mismatch too large: case=%s impl=%s max_abs=%f\n",
                      c.name, "mqtile_hd128_q4k8_kvimg", diff);
            return false;
        }
        const float refAvgMs = row32.avgMs > 0.0f ? row32.avgMs : imageKV.avgMs;
        if (!printSparseFlashTimedResult(c, "mqtile_hd128_q4k8_kvimg", imageKV.avgMs, refAvgMs, diff)) {
            return false;
        }
    }
    if (sparseFlashImplEnabled("mqtile_hd128_q8k16_kvimg")) {
        SparseFlashBuffers imageBuffers;
        if (!prepareSparseFlashImageBuffers(bench, c, &imageBuffers)) {
            return false;
        }
        SparseFlashImplResult imageKV;
        if (!runSparseFlashImageKVImpl(bench, c, imageBuffers, "mqtile_sparse_flash_hd128_q8k16_kvimg",
                                       {16u, static_cast<uint32_t>(ROUND_UP(c.outputSeqLen, 8)),
                                        static_cast<uint32_t>(c.batch * c.qHeads)},
                                       {16u, 8u, 1u}, &imageKV)) {
            return false;
        }
        const bool hasReference = !row32.output.empty();
        const float diff = hasReference ? maxAbsDiff(row32.output, imageKV.output) : 0.0f;
        if (hasReference && std::isfinite(diff) && diff > 1.0e-1f) {
            MNN_ERROR("sparse flash image-kv output mismatch too large: case=%s impl=%s max_abs=%f\n",
                      c.name, "mqtile_hd128_q8k16_kvimg", diff);
            return false;
        }
        const float refAvgMs = row32.avgMs > 0.0f ? row32.avgMs : imageKV.avgMs;
        if (!printSparseFlashTimedResult(c, "mqtile_hd128_q8k16_kvimg", imageKV.avgMs, refAvgMs, diff)) {
            return false;
        }
    }
    if (sparseFlashImplEnabled("mqtile_hd128_q8k16_kimg")) {
        SparseFlashBuffers imageBuffers;
        if (!prepareSparseFlashImageBuffers(bench, c, &imageBuffers)) {
            return false;
        }
        SparseFlashImplResult imageKey;
        if (!runSparseFlashImageKeyImpl(bench, c, imageBuffers, "mqtile_sparse_flash_hd128_q8k16_kimg",
                                        {16u, static_cast<uint32_t>(ROUND_UP(c.outputSeqLen, 8)),
                                         static_cast<uint32_t>(c.batch * c.qHeads)},
                                        {16u, 8u, 1u}, &imageKey)) {
            return false;
        }
        const bool hasReference = !row32.output.empty();
        const float diff = hasReference ? maxAbsDiff(row32.output, imageKey.output) : 0.0f;
        if (hasReference && std::isfinite(diff) && diff > 1.0e-1f) {
            MNN_ERROR("sparse flash image-key output mismatch too large: case=%s impl=%s max_abs=%f\n",
                      c.name, "mqtile_hd128_q8k16_kimg", diff);
            return false;
        }
        const float refAvgMs = row32.avgMs > 0.0f ? row32.avgMs : imageKey.avgMs;
        if (!printSparseFlashTimedResult(c, "mqtile_hd128_q8k16_kimg", imageKey.avgMs, refAvgMs, diff)) {
            return false;
        }
    }
    return runMqtile("mqtile_sparse_flash_hd128_q8k16", "mqtile_hd128_q8k16",
                     {16u, static_cast<uint32_t>(ROUND_UP(c.outputSeqLen, 8)),
                      static_cast<uint32_t>(c.batch * c.qHeads)},
                     {16u, 8u, 1u});
}

class OpenCLSparseFlashMultiQTilePerf : public MNNTestCase {
public:
    bool run(int precision) override {
        (void)precision;
        auto cases = sparseFlashBenchCases();
        for (const auto& c : cases) {
            if (!enabledByFilter("MNN_BENCH_OPENCL_SPARSE_FLASH_CASE", c.name)) {
                continue;
            }
            if (!runSparseFlashBenchCase(c)) {
                return false;
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(OpenCLSparseFlashMultiQTilePerf, "bench_ops/opencl/perf/SparseFlash/MultiQTile");

} // namespace

#endif
