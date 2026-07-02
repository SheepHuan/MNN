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
#include "backend/opencl/execution/buffer/ConvBufLowMemoryExecution.hpp"
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

static std::vector<int> rowsFromEnvOrDefault(const char* envName, std::vector<int> defaults) {
    const char* filter = ::getenv(envName);
    if (filter == nullptr || filter[0] == '\0') {
        return defaults;
    }
    std::vector<int> rows;
    std::string spec(filter);
    size_t start = 0;
    while (start <= spec.size()) {
        size_t end = spec.find(',', start);
        std::string item = spec.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!item.empty()) {
            size_t dash = item.find('-');
            if (dash == std::string::npos) {
                const int row = ::atoi(item.c_str());
                if (row > 0) {
                    rows.push_back(row);
                }
            } else {
                int lo = ::atoi(item.substr(0, dash).c_str());
                int hi = ::atoi(item.substr(dash + 1).c_str());
                if (lo > hi) {
                    std::swap(lo, hi);
                }
                for (int row = std::max(1, lo); row <= hi; ++row) {
                    rows.push_back(row);
                }
            }
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows.empty() ? defaults : rows;
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
    AdrenoDirectC4Gemv = 12,
    AdrenoBatchGemvC4Out = 13,
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

static std::unique_ptr<OpHolder> makePicSiluMulOp(const char* name) {
    OpT op;
    op.name = name != nullptr ? name : "bench_opencl_pic_silu_mul";
    op.type = OpType_Extra;
    op.defaultDimentionFormat = MNN_DATA_FORMAT_NCHW;
    op.main.type = OpParameter_Extra;
    op.main.value = new ExtraT;
    auto* extra = op.main.AsExtra();
    extra->type = "PicSiluMul";
    extra->engine = "MNN";
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
        case WeightOnlyCompactDenseFamily::AdrenoDirectC4Gemv:
            return "adreno_direct_c4_gemv";
        case WeightOnlyCompactDenseFamily::AdrenoBatchGemvC4Out:
            return "adreno_batch_gemv_c4out";
        default:
            return "unknown";
    }
}

static bool parseWeightOnlyCompactDenseFamilyName(const char* value, uint32_t* family) {
    if (value == nullptr || value[0] == '\0' || family == nullptr) {
        return false;
    }
    const std::pair<const char*, uint32_t> families[] = {
        {"generic_quant", static_cast<uint32_t>(WeightOnlyCompactDenseFamily::GenericQuant)},
        {"pic_quant", static_cast<uint32_t>(WeightOnlyCompactDenseFamily::PicQuant)},
        {"fp_weight", static_cast<uint32_t>(WeightOnlyCompactDenseFamily::FPWeight)},
        {"pic_quant_b2", static_cast<uint32_t>(WeightOnlyCompactDenseFamily::PicQuantB2)},
        {"pic_quant_c4", static_cast<uint32_t>(WeightOnlyCompactDenseFamily::PicQuantC4)},
        {"pic_quant_wg64", static_cast<uint32_t>(WeightOnlyCompactDenseFamily::PicQuantWG64)},
        {"pic_quant_wg128", static_cast<uint32_t>(WeightOnlyCompactDenseFamily::PicQuantWG128)},
        {"adreno_batch_gemv", static_cast<uint32_t>(WeightOnlyCompactDenseFamily::AdrenoBatchGemv)},
        {"pic_quant_wg4x32", static_cast<uint32_t>(WeightOnlyCompactDenseFamily::PicQuantWG4X32)},
        {"pic_quant_wg8x16", static_cast<uint32_t>(WeightOnlyCompactDenseFamily::PicQuantWG8X16)},
        {"pic_quant_incache32", static_cast<uint32_t>(WeightOnlyCompactDenseFamily::PicQuantInputCache32)},
        {"pic_quant_incache64", static_cast<uint32_t>(WeightOnlyCompactDenseFamily::PicQuantInputCache64)},
        {"adreno_direct_c4_gemv", static_cast<uint32_t>(WeightOnlyCompactDenseFamily::AdrenoDirectC4Gemv)},
        {"adreno_batch_gemv_c4out", static_cast<uint32_t>(WeightOnlyCompactDenseFamily::AdrenoBatchGemvC4Out)},
    };
    for (const auto& item : families) {
        if (::strcmp(value, item.first) == 0) {
            *family = item.second;
            return true;
        }
    }
    return false;
}

static bool isAdrenoTinyDenseFamilyShape(OpenCLRuntime* runtime, int rows, int ic, int oc, int quantBit) {
    return runtime != nullptr &&
           runtime->getGpuType() == ADRENO &&
           quantBit == 4 &&
           rows > 1 && rows <= 16 &&
           ic >= 1024 && oc >= 256;
}

static std::string makeAdrenoTinyDenseFamilyTuneKey(int rows, int ic, int oc, int quantBit) {
    return "adreno_tiny_dense_family_v1_q" + std::to_string(quantBit) +
           "_m" + std::to_string(rows) +
           "_ic" + std::to_string(ic) +
           "_oc" + std::to_string(oc);
}

static int adrenoTinyDenseHeuristicFamily(int rows) {
    if (rows <= 2) {
        return static_cast<int>(WeightOnlyCompactDenseFamily::PicQuantC4);
    }
    if (rows <= 4) {
        return static_cast<int>(WeightOnlyCompactDenseFamily::PicQuantInputCache64);
    }
    return static_cast<int>(WeightOnlyCompactDenseFamily::AdrenoBatchGemvC4Out);
}

static int readSelectedCompactDenseFamily(DirectOpBenchOpenCL& bench, int rows, int ic, int oc, int quantBit) {
    uint32_t forcedFamily = 0;
    if (parseWeightOnlyCompactDenseFamilyName(::getenv("MNN_BENCH_OPENCL_COMPACT_DENSE_FORCE_FAMILY"),
                                              &forcedFamily)) {
        return static_cast<int>(forcedFamily);
    }
    if (isAdrenoTinyDenseFamilyShape(bench.runtime(), rows, ic, oc, quantBit)) {
        std::pair<std::vector<uint32_t>, uint32_t> tinyInfo;
        if (MNN::OpenCL::getTunedInfo(makeAdrenoTinyDenseFamilyTuneKey(rows, ic, oc, quantBit),
                                      {static_cast<uint32_t>(rows), static_cast<uint32_t>(oc),
                                       static_cast<uint32_t>(ic)},
                                      tinyInfo, bench.runtime()) &&
            !tinyInfo.first.empty()) {
            return static_cast<int>(tinyInfo.first[0]);
        }
        return adrenoTinyDenseHeuristicFamily(rows);
    }
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

struct DecodeRepairMlpChainCase {
    const char* name;
    int rows;
    int hidden;
    int inter;
    int quantBlock;
    int warmup;
    int repeat;
};

static bool measureExecution(DirectOpBenchOpenCL& bench, Execution* exe,
                             const std::vector<Tensor*>& inputs,
                             const std::vector<Tensor*>& outputs,
                             Tensor* syncTensor, int warmup, int repeat,
                             float* avgMs) {
    std::vector<float> syncOutput;
    OpenCLWallTimer timer;
    return timer.measure([&]() { return bench.execute(exe, inputs, outputs); },
                         [&]() {
                             if (syncTensor != nullptr) {
                                 return bench.readTensorTyped<float>(syncTensor, &syncOutput);
                             }
                             return bench.sync();
                         },
                         warmup, repeat, avgMs);
}

static bool runDecodeRepairMlpChainCase(const DecodeRepairMlpChainCase& c) {
    DirectOpBenchOpenCL bench;
    if (!bench.valid()) {
        return false;
    }
    auto hidden = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    auto gate = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto up = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto act = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto out = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    if (hidden == nullptr || gate == nullptr || up == nullptr || act == nullptr || out == nullptr) {
        return false;
    }
    if (!bench.writeTensor(hidden, makePattern(hidden->elementSize(), 0.0078125f))) {
        return false;
    }

    auto gateOp = makeWeightOnlyLinearConvOp(c.hidden, c.inter, c.quantBlock);
    auto upOp = makeWeightOnlyLinearConvOp(c.hidden, c.inter, c.quantBlock);
    auto siluOp = makePicSiluMulOp("bench_opencl_decode_repair_mlp_silu_mul");
    auto downOp = makeWeightOnlyLinearConvOp(c.inter, c.hidden, c.quantBlock);

    std::vector<Tensor*> hiddenInput = {hidden};
    std::vector<Tensor*> gateOutput = {gate};
    std::vector<Tensor*> upOutput = {up};
    std::vector<Tensor*> siluInputs = {gate, up};
    std::vector<Tensor*> actOutput = {act};
    std::vector<Tensor*> downInput = {act};
    std::vector<Tensor*> downOutput = {out};

    auto gateExe = bench.create(hiddenInput, gateOutput, gateOp->get());
    auto upExe = bench.create(hiddenInput, upOutput, upOp->get());
    auto siluExe = bench.create(siluInputs, actOutput, siluOp->get());
    auto downExe = bench.create(downInput, downOutput, downOp->get());
    if (!gateExe || !upExe || !siluExe || !downExe) {
        MNN_ERROR("failed to create OpenCL DecodeRepairMlpChain executions for %s rows=%d\n", c.name, c.rows);
        return false;
    }
    auto code = bench.resize(gateExe.get(), hiddenInput, gateOutput);
    if (code == NO_ERROR) {
        code = bench.resize(upExe.get(), hiddenInput, upOutput);
    }
    if (code == NO_ERROR) {
        code = bench.resize(siluExe.get(), siluInputs, actOutput);
    }
    if (code == NO_ERROR) {
        code = bench.resize(downExe.get(), downInput, downOutput);
    }
    if (code != NO_ERROR) {
        MNN_ERROR("OpenCL DecodeRepairMlpChain onResize failed for %s rows=%d: %d\n", c.name, c.rows, code);
        return false;
    }

    const int gateFamily = readSelectedCompactDenseFamily(bench, c.rows, c.hidden, c.inter, 4);
    const int upFamily = readSelectedCompactDenseFamily(bench, c.rows, c.hidden, c.inter, 4);
    const int downFamily = readSelectedCompactDenseFamily(bench, c.rows, c.inter, c.hidden, 4);

    float gateMs = 0.0f;
    float upMs = 0.0f;
    float siluMs = 0.0f;
    float downMs = 0.0f;
    float chainMs = 0.0f;
    if (!measureExecution(bench, gateExe.get(), hiddenInput, gateOutput, gate, c.warmup, c.repeat, &gateMs)) {
        return false;
    }
    if (!measureExecution(bench, upExe.get(), hiddenInput, upOutput, up, c.warmup, c.repeat, &upMs)) {
        return false;
    }
    if (bench.execute(gateExe.get(), hiddenInput, gateOutput) != NO_ERROR ||
        bench.execute(upExe.get(), hiddenInput, upOutput) != NO_ERROR ||
        !bench.sync()) {
        return false;
    }
    if (!measureExecution(bench, siluExe.get(), siluInputs, actOutput, act, c.warmup, c.repeat, &siluMs)) {
        return false;
    }
    if (bench.execute(siluExe.get(), siluInputs, actOutput) != NO_ERROR || !bench.sync()) {
        return false;
    }
    if (!measureExecution(bench, downExe.get(), downInput, downOutput, out, c.warmup, c.repeat, &downMs)) {
        return false;
    }

    OpenCLWallTimer timer;
    std::vector<float> syncOutput;
    auto runChain = [&]() {
        auto chainCode = bench.execute(gateExe.get(), hiddenInput, gateOutput);
        if (chainCode != NO_ERROR) {
            return chainCode;
        }
        chainCode = bench.execute(upExe.get(), hiddenInput, upOutput);
        if (chainCode != NO_ERROR) {
            return chainCode;
        }
        chainCode = bench.execute(siluExe.get(), siluInputs, actOutput);
        if (chainCode != NO_ERROR) {
            return chainCode;
        }
        return bench.execute(downExe.get(), downInput, downOutput);
    };
    if (!timer.measure(runChain, [&]() { return bench.readTensorTyped<float>(out, &syncOutput); },
                       c.warmup, c.repeat, &chainMs)) {
        return false;
    }

    const float splitSumMs = gateMs + upMs + siluMs + downMs;
    MNN_PRINT("[bench_ops/opencl/perf/DecodeRepairMlpChain] %-24s rows=%d hidden=%d inter=%d qblock=%d "
              "gate_family=%-22s up_family=%-22s down_family=%-22s "
              "gate=%.4f up=%.4f silu=%.4f down=%.4f split_sum=%.4f chain=%.4f ms\n",
              c.name, c.rows, c.hidden, c.inter, c.quantBlock,
              gateFamily >= 0 ? weightOnlyCompactDenseFamilyName(static_cast<uint32_t>(gateFamily)) : "-",
              upFamily >= 0 ? weightOnlyCompactDenseFamilyName(static_cast<uint32_t>(upFamily)) : "-",
              downFamily >= 0 ? weightOnlyCompactDenseFamilyName(static_cast<uint32_t>(downFamily)) : "-",
              gateMs, upMs, siluMs, downMs, splitSumMs, chainMs);
    ::fflush(stdout);
    return true;
}

static const char* kDecodeRepairMlpFusedFloorSource = R"(
#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

inline int raw_i4(__global const uchar* weight, int index) {
    uchar packed = weight[index >> 1];
    int v = ((index & 1) == 0) ? (int)(packed >> 4) : (int)(packed & (uchar)15);
    return v - 8;
}

inline FLOAT8 raw_load_i4x8(__global const uchar* weight, int oc8, int k, int ic, FLOAT8 scale) {
    return (FLOAT8)(
        (FLOAT)raw_i4(weight, (oc8 + 0) * ic + k),
        (FLOAT)raw_i4(weight, (oc8 + 1) * ic + k),
        (FLOAT)raw_i4(weight, (oc8 + 2) * ic + k),
        (FLOAT)raw_i4(weight, (oc8 + 3) * ic + k),
        (FLOAT)raw_i4(weight, (oc8 + 4) * ic + k),
        (FLOAT)raw_i4(weight, (oc8 + 5) * ic + k),
        (FLOAT)raw_i4(weight, (oc8 + 6) * ic + k),
        (FLOAT)raw_i4(weight, (oc8 + 7) * ic + k)) * scale;
}

inline FLOAT8 raw_load_scale8(__global const FLOAT* scale, int oc8, int group, int groups) {
    return (FLOAT8)(
        scale[(oc8 + 0) * groups + group],
        scale[(oc8 + 1) * groups + group],
        scale[(oc8 + 2) * groups + group],
        scale[(oc8 + 3) * groups + group],
        scale[(oc8 + 4) * groups + group],
        scale[(oc8 + 5) * groups + group],
        scale[(oc8 + 6) * groups + group],
        scale[(oc8 + 7) * groups + group]);
}

inline FLOAT8 raw_silu_mul8(FLOAT8 gate, FLOAT8 up) {
    float8 gate_f = convert_float8(gate);
    float8 up_f = convert_float8(up);
    float8 fused = gate_f * native_recip((float8)1.0f + native_exp(-gate_f)) * up_f;
#ifdef MNN_SUPPORT_FP16
    return convert_half8(fused);
#else
    return fused;
#endif
}

__kernel void raw_gateup_silu_int4(__private int global_dim0,
                                   __private int global_dim1,
                                   __private int global_dim2,
                                   __global const FLOAT* input,
                                   __global const uchar* gate_weight,
                                   __global const FLOAT* gate_scale,
                                   __global const FLOAT* gate_bias,
                                   __global const uchar* up_weight,
                                   __global const FLOAT* up_scale,
                                   __global const FLOAT* up_bias,
                                   __global FLOAT* act,
                                   __private int rows,
                                   __private int hidden,
                                   __private int inter,
                                   __private int quant_block,
                                   __private int groups) {
    const int lid = get_local_id(0);
    const int oc = get_global_id(1);
    const int b4 = get_global_id(2);
    if (lid >= global_dim0 || oc >= global_dim1 || b4 >= global_dim2) {
        return;
    }
    const int row = b4 << 2;
    const int oc8 = oc << 3;
    FLOAT8 gate0 = 0;
    FLOAT8 gate1 = 0;
    FLOAT8 gate2 = 0;
    FLOAT8 gate3 = 0;
    FLOAT8 up0 = 0;
    FLOAT8 up1 = 0;
    FLOAT8 up2 = 0;
    FLOAT8 up3 = 0;
    __local FLOAT8 gate_sum0[WGS];
    __local FLOAT8 gate_sum1[WGS];
    __local FLOAT8 gate_sum2[WGS];
    __local FLOAT8 gate_sum3[WGS];
    __local FLOAT8 up_sum0[WGS];
    __local FLOAT8 up_sum1[WGS];
    __local FLOAT8 up_sum2[WGS];
    __local FLOAT8 up_sum3[WGS];
    for (int k = lid; k < hidden; k += WGS) {
        const int group = k / quant_block;
        FLOAT8 gate_w = raw_load_i4x8(gate_weight, oc8, k, hidden,
                                      raw_load_scale8(gate_scale, oc8, group, groups));
        FLOAT8 up_w = raw_load_i4x8(up_weight, oc8, k, hidden,
                                    raw_load_scale8(up_scale, oc8, group, groups));
        FLOAT in0 = input[row * hidden + k];
        FLOAT in1 = row + 1 < rows ? input[(row + 1) * hidden + k] : (FLOAT)0;
        FLOAT in2 = row + 2 < rows ? input[(row + 2) * hidden + k] : (FLOAT)0;
        FLOAT in3 = row + 3 < rows ? input[(row + 3) * hidden + k] : (FLOAT)0;
        gate0 = mad((FLOAT8)in0, gate_w, gate0);
        gate1 = mad((FLOAT8)in1, gate_w, gate1);
        gate2 = mad((FLOAT8)in2, gate_w, gate2);
        gate3 = mad((FLOAT8)in3, gate_w, gate3);
        up0 = mad((FLOAT8)in0, up_w, up0);
        up1 = mad((FLOAT8)in1, up_w, up1);
        up2 = mad((FLOAT8)in2, up_w, up2);
        up3 = mad((FLOAT8)in3, up_w, up3);
    }
    gate_sum0[lid] = gate0;
    gate_sum1[lid] = gate1;
    gate_sum2[lid] = gate2;
    gate_sum3[lid] = gate3;
    up_sum0[lid] = up0;
    up_sum1[lid] = up1;
    up_sum2[lid] = up2;
    up_sum3[lid] = up3;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = WGS >> 1; offset > 0; offset >>= 1) {
        if (lid < offset) {
            gate_sum0[lid] += gate_sum0[lid + offset];
            gate_sum1[lid] += gate_sum1[lid + offset];
            gate_sum2[lid] += gate_sum2[lid + offset];
            gate_sum3[lid] += gate_sum3[lid + offset];
            up_sum0[lid] += up_sum0[lid + offset];
            up_sum1[lid] += up_sum1[lid + offset];
            up_sum2[lid] += up_sum2[lid + offset];
            up_sum3[lid] += up_sum3[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        FLOAT8 gate_bias_v = vload8(0, gate_bias + oc8);
        FLOAT8 up_bias_v = vload8(0, up_bias + oc8);
        FLOAT8 fused0 = raw_silu_mul8(gate_sum0[0] + gate_bias_v, up_sum0[0] + up_bias_v);
        FLOAT8 fused1 = raw_silu_mul8(gate_sum1[0] + gate_bias_v, up_sum1[0] + up_bias_v);
        FLOAT8 fused2 = raw_silu_mul8(gate_sum2[0] + gate_bias_v, up_sum2[0] + up_bias_v);
        FLOAT8 fused3 = raw_silu_mul8(gate_sum3[0] + gate_bias_v, up_sum3[0] + up_bias_v);
        vstore8(fused0, 0, act + row * inter + oc8);
        if (row + 1 < rows) {
            vstore8(fused1, 0, act + (row + 1) * inter + oc8);
        }
        if (row + 2 < rows) {
            vstore8(fused2, 0, act + (row + 2) * inter + oc8);
        }
        if (row + 3 < rows) {
            vstore8(fused3, 0, act + (row + 3) * inter + oc8);
        }
    }
}

__kernel void raw_linear_int4(__private int global_dim0,
                              __private int global_dim1,
                              __private int global_dim2,
                              __global const FLOAT* input,
                              __global const uchar* weight,
                              __global const FLOAT* scale,
                              __global const FLOAT* bias,
                              __global FLOAT* output,
                              __private int rows,
                              __private int ic,
                              __private int oc_total,
                              __private int quant_block,
                              __private int groups) {
    const int lid = get_local_id(0);
    const int oc = get_global_id(1);
    const int b4 = get_global_id(2);
    if (lid >= global_dim0 || oc >= global_dim1 || b4 >= global_dim2) {
        return;
    }
    const int row = b4 << 2;
    const int oc8 = oc << 3;
    FLOAT8 sum0 = 0;
    FLOAT8 sum1 = 0;
    FLOAT8 sum2 = 0;
    FLOAT8 sum3 = 0;
    __local FLOAT8 local0[WGS];
    __local FLOAT8 local1[WGS];
    __local FLOAT8 local2[WGS];
    __local FLOAT8 local3[WGS];
    for (int k = lid; k < ic; k += WGS) {
        const int group = k / quant_block;
        FLOAT8 w = raw_load_i4x8(weight, oc8, k, ic, raw_load_scale8(scale, oc8, group, groups));
        FLOAT in0 = input[row * ic + k];
        FLOAT in1 = row + 1 < rows ? input[(row + 1) * ic + k] : (FLOAT)0;
        FLOAT in2 = row + 2 < rows ? input[(row + 2) * ic + k] : (FLOAT)0;
        FLOAT in3 = row + 3 < rows ? input[(row + 3) * ic + k] : (FLOAT)0;
        sum0 = mad((FLOAT8)in0, w, sum0);
        sum1 = mad((FLOAT8)in1, w, sum1);
        sum2 = mad((FLOAT8)in2, w, sum2);
        sum3 = mad((FLOAT8)in3, w, sum3);
    }
    local0[lid] = sum0;
    local1[lid] = sum1;
    local2[lid] = sum2;
    local3[lid] = sum3;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = WGS >> 1; offset > 0; offset >>= 1) {
        if (lid < offset) {
            local0[lid] += local0[lid + offset];
            local1[lid] += local1[lid + offset];
            local2[lid] += local2[lid + offset];
            local3[lid] += local3[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        FLOAT8 bias_v = vload8(0, bias + oc8);
        vstore8(local0[0] + bias_v, 0, output + row * oc_total + oc8);
        if (row + 1 < rows) {
            vstore8(local1[0] + bias_v, 0, output + (row + 1) * oc_total + oc8);
        }
        if (row + 2 < rows) {
            vstore8(local2[0] + bias_v, 0, output + (row + 2) * oc_total + oc8);
        }
        if (row + 3 < rows) {
            vstore8(local3[0] + bias_v, 0, output + (row + 3) * oc_total + oc8);
        }
    }
}
)";

static const char* kDecodeRepairMlpResourceGateUpSource = R"(
#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#define UCHAR4_TO_CHAR8_LOCAL(b, scale, offset, wei) \
    wei.s0 = (FLOAT)((b.s0 >> 4) - 8); \
    wei.s1 = (FLOAT)((b.s0 & 15) - 8); \
    wei.s2 = (FLOAT)((b.s1 >> 4) - 8); \
    wei.s3 = (FLOAT)((b.s1 & 15) - 8); \
    wei.s4 = (FLOAT)((b.s2 >> 4) - 8); \
    wei.s5 = (FLOAT)((b.s2 & 15) - 8); \
    wei.s6 = (FLOAT)((b.s3 >> 4) - 8); \
    wei.s7 = (FLOAT)((b.s3 & 15) - 8); \
    wei = wei * scale + offset;

inline FLOAT8 resource_silu_mul8(FLOAT8 gate, FLOAT8 up) {
    float8 gate_f = convert_float8(gate);
    float8 up_f = convert_float8(up);
    float8 fused = gate_f * native_recip((float8)1.0f + native_exp(-gate_f)) * up_f;
#ifdef MNN_SUPPORT_FP16
    return convert_half8(fused);
#else
    return fused;
#endif
}

__kernel void resource_gateup_silu_int4_image(__private int global_dim0,
                                              __private int global_dim1,
                                              __private int global_dim2,
                                              __global const FLOAT* input,
                                              __read_only image2d_t gate_weight,
                                              __global const FLOAT* gate_scale_offset,
                                              __global const FLOAT* gate_bias,
#ifdef RESOURCE_GATEUP_PAIR_OUTPUT
                                              __global FLOAT* gate_output,
                                              __read_only image2d_t up_weight,
                                              __global const FLOAT* up_scale_offset,
                                              __global const FLOAT* up_bias,
                                              __global FLOAT* up_output,
#else
                                              __read_only image2d_t up_weight,
                                              __global const FLOAT* up_scale_offset,
                                              __global const FLOAT* up_bias,
                                              __global FLOAT* fused_output,
#endif
                                              __private int rows,
                                              __private int dst_channel_align,
                                              __private int dst_channel_c4,
                                              __private int src_channel_align,
                                              __private int src_channel,
                                              __private int block_dim,
                                              __private float gate_coef,
                                              __private float up_coef) {
    const int lid = get_local_id(0);
    const int oc = get_global_id(1);
    const int b4 = get_global_id(2);
    if (lid >= global_dim0 || oc >= global_dim1 || b4 >= global_dim2) {
        return;
    }
    const int row = b4 << 2;
    const int oc8 = oc << 3;
    const int loop = (src_channel + 4 - 1) / 4;
    const int bhw4 = rows << 2;
    FLOAT8 gate0 = 0;
    FLOAT8 gate1 = 0;
    FLOAT8 gate2 = 0;
    FLOAT8 gate3 = 0;
    FLOAT8 up0 = 0;
    FLOAT8 up1 = 0;
    FLOAT8 up2 = 0;
    FLOAT8 up3 = 0;
    __local FLOAT8 gate_sum0[WGS];
    __local FLOAT8 gate_sum1[WGS];
    __local FLOAT8 gate_sum2[WGS];
    __local FLOAT8 gate_sum3[WGS];
    __local FLOAT8 up_sum0[WGS];
    __local FLOAT8 up_sum1[WGS];
    __local FLOAT8 up_sum2[WGS];
    __local FLOAT8 up_sum3[WGS];
    for (int j = lid; j < loop; j += WGS) {
        const int k4 = j << 2;
#ifdef ASYMMETRIC
        FLOAT8 gate_scale;
        FLOAT8 gate_offset;
        FLOAT8 up_scale;
        FLOAT8 up_offset;
        {
            FLOAT16 so = vload16(0, gate_scale_offset + oc8 * 2 + (k4 / block_dim) * dst_channel_c4 * 8) / (FLOAT16)gate_coef;
            gate_scale = so.s02468ace;
            gate_offset = so.s13579bdf;
            so = vload16(0, up_scale_offset + oc8 * 2 + (k4 / block_dim) * dst_channel_c4 * 8) / (FLOAT16)up_coef;
            up_scale = so.s02468ace;
            up_offset = so.s13579bdf;
        }
#else
        FLOAT8 gate_scale = vload8(0, gate_scale_offset + oc8 + (k4 / block_dim) * dst_channel_c4 * 4) / (FLOAT8)gate_coef;
        FLOAT8 up_scale = vload8(0, up_scale_offset + oc8 + (k4 / block_dim) * dst_channel_c4 * 4) / (FLOAT8)up_coef;
        FLOAT8 gate_offset = 0;
        FLOAT8 up_offset = 0;
#endif
        FLOAT4 in0 = vload4(0, input + row * src_channel_align + k4);
        FLOAT4 in1 = row + 1 < rows ? vload4(0, input + (row + 1) * src_channel_align + k4) : (FLOAT4)0;
        FLOAT4 in2 = row + 2 < rows ? vload4(0, input + (row + 2) * src_channel_align + k4) : (FLOAT4)0;
        FLOAT4 in3 = row + 3 < rows ? vload4(0, input + (row + 3) * src_channel_align + k4) : (FLOAT4)0;
        uchar16 gate_packed = as_uchar16(read_imagei(gate_weight, SAMPLER, (int2)(j, oc)));
        uchar16 up_packed = as_uchar16(read_imagei(up_weight, SAMPLER, (int2)(j, oc)));
        FLOAT8 gate_wei;
        FLOAT8 up_wei;
        UCHAR4_TO_CHAR8_LOCAL(gate_packed.s0123, gate_scale, gate_offset, gate_wei);
        UCHAR4_TO_CHAR8_LOCAL(up_packed.s0123, up_scale, up_offset, up_wei);
        gate0 = mad((FLOAT8)in0.s0, gate_wei, gate0); gate1 = mad((FLOAT8)in1.s0, gate_wei, gate1);
        gate2 = mad((FLOAT8)in2.s0, gate_wei, gate2); gate3 = mad((FLOAT8)in3.s0, gate_wei, gate3);
        up0 = mad((FLOAT8)in0.s0, up_wei, up0); up1 = mad((FLOAT8)in1.s0, up_wei, up1);
        up2 = mad((FLOAT8)in2.s0, up_wei, up2); up3 = mad((FLOAT8)in3.s0, up_wei, up3);
        UCHAR4_TO_CHAR8_LOCAL(gate_packed.s4567, gate_scale, gate_offset, gate_wei);
        UCHAR4_TO_CHAR8_LOCAL(up_packed.s4567, up_scale, up_offset, up_wei);
        gate0 = mad((FLOAT8)in0.s1, gate_wei, gate0); gate1 = mad((FLOAT8)in1.s1, gate_wei, gate1);
        gate2 = mad((FLOAT8)in2.s1, gate_wei, gate2); gate3 = mad((FLOAT8)in3.s1, gate_wei, gate3);
        up0 = mad((FLOAT8)in0.s1, up_wei, up0); up1 = mad((FLOAT8)in1.s1, up_wei, up1);
        up2 = mad((FLOAT8)in2.s1, up_wei, up2); up3 = mad((FLOAT8)in3.s1, up_wei, up3);
        UCHAR4_TO_CHAR8_LOCAL(gate_packed.s89ab, gate_scale, gate_offset, gate_wei);
        UCHAR4_TO_CHAR8_LOCAL(up_packed.s89ab, up_scale, up_offset, up_wei);
        gate0 = mad((FLOAT8)in0.s2, gate_wei, gate0); gate1 = mad((FLOAT8)in1.s2, gate_wei, gate1);
        gate2 = mad((FLOAT8)in2.s2, gate_wei, gate2); gate3 = mad((FLOAT8)in3.s2, gate_wei, gate3);
        up0 = mad((FLOAT8)in0.s2, up_wei, up0); up1 = mad((FLOAT8)in1.s2, up_wei, up1);
        up2 = mad((FLOAT8)in2.s2, up_wei, up2); up3 = mad((FLOAT8)in3.s2, up_wei, up3);
        UCHAR4_TO_CHAR8_LOCAL(gate_packed.scdef, gate_scale, gate_offset, gate_wei);
        UCHAR4_TO_CHAR8_LOCAL(up_packed.scdef, up_scale, up_offset, up_wei);
        gate0 = mad((FLOAT8)in0.s3, gate_wei, gate0); gate1 = mad((FLOAT8)in1.s3, gate_wei, gate1);
        gate2 = mad((FLOAT8)in2.s3, gate_wei, gate2); gate3 = mad((FLOAT8)in3.s3, gate_wei, gate3);
        up0 = mad((FLOAT8)in0.s3, up_wei, up0); up1 = mad((FLOAT8)in1.s3, up_wei, up1);
        up2 = mad((FLOAT8)in2.s3, up_wei, up2); up3 = mad((FLOAT8)in3.s3, up_wei, up3);
    }
    gate_sum0[lid] = gate0; gate_sum1[lid] = gate1; gate_sum2[lid] = gate2; gate_sum3[lid] = gate3;
    up_sum0[lid] = up0; up_sum1[lid] = up1; up_sum2[lid] = up2; up_sum3[lid] = up3;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = WGS / 2; i > 0; i >>= 1) {
        if (lid < i) {
            gate_sum0[lid] += gate_sum0[lid + i]; gate_sum1[lid] += gate_sum1[lid + i];
            gate_sum2[lid] += gate_sum2[lid + i]; gate_sum3[lid] += gate_sum3[lid + i];
            up_sum0[lid] += up_sum0[lid + i]; up_sum1[lid] += up_sum1[lid + i];
            up_sum2[lid] += up_sum2[lid + i]; up_sum3[lid] += up_sum3[lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        FLOAT8 gate_bias_v = vload8(0, gate_bias + oc8);
        FLOAT8 up_bias_v = vload8(0, up_bias + oc8);
        FLOAT8 gate_v0 = gate_sum0[0] + gate_bias_v;
        FLOAT8 gate_v1 = gate_sum1[0] + gate_bias_v;
        FLOAT8 gate_v2 = gate_sum2[0] + gate_bias_v;
        FLOAT8 gate_v3 = gate_sum3[0] + gate_bias_v;
        FLOAT8 up_v0 = up_sum0[0] + up_bias_v;
        FLOAT8 up_v1 = up_sum1[0] + up_bias_v;
        FLOAT8 up_v2 = up_sum2[0] + up_bias_v;
        FLOAT8 up_v3 = up_sum3[0] + up_bias_v;
#ifndef RESOURCE_GATEUP_PAIR_OUTPUT
        FLOAT8 fused0 = resource_silu_mul8(gate_v0, up_v0);
        FLOAT8 fused1 = resource_silu_mul8(gate_v1, up_v1);
        FLOAT8 fused2 = resource_silu_mul8(gate_v2, up_v2);
        FLOAT8 fused3 = resource_silu_mul8(gate_v3, up_v3);
#endif
        const int out_c = oc << 1;
        const int out_offset = out_c * bhw4 + row * 4;
        if (row + 3 < rows) {
#ifdef RESOURCE_GATEUP_PAIR_OUTPUT
            vstore16((FLOAT16)(gate_v0.s0123, gate_v1.s0123, gate_v2.s0123, gate_v3.s0123), 0, gate_output + out_offset);
            vstore16((FLOAT16)(up_v0.s0123, up_v1.s0123, up_v2.s0123, up_v3.s0123), 0, up_output + out_offset);
            if (oc8 + 4 < dst_channel_align) {
                const int out_offset_hi = out_offset + bhw4;
                vstore16((FLOAT16)(gate_v0.s4567, gate_v1.s4567, gate_v2.s4567, gate_v3.s4567), 0, gate_output + out_offset_hi);
                vstore16((FLOAT16)(up_v0.s4567, up_v1.s4567, up_v2.s4567, up_v3.s4567), 0, up_output + out_offset_hi);
            }
#else
            vstore16((FLOAT16)(fused0.s0123, fused1.s0123, fused2.s0123, fused3.s0123), 0, fused_output + out_offset);
            if (oc8 + 4 < dst_channel_align) {
                const int out_offset_hi = out_offset + bhw4;
                vstore16((FLOAT16)(fused0.s4567, fused1.s4567, fused2.s4567, fused3.s4567), 0, fused_output + out_offset_hi);
            }
#endif
        } else {
#ifdef RESOURCE_GATEUP_PAIR_OUTPUT
            vstore4(gate_v0.s0123, 0, gate_output + out_offset);
            vstore4(up_v0.s0123, 0, up_output + out_offset);
            if (row + 1 < rows) {
                vstore4(gate_v1.s0123, 0, gate_output + out_offset + 4);
                vstore4(up_v1.s0123, 0, up_output + out_offset + 4);
            }
            if (row + 2 < rows) {
                vstore4(gate_v2.s0123, 0, gate_output + out_offset + 8);
                vstore4(up_v2.s0123, 0, up_output + out_offset + 8);
            }
            if (oc8 + 4 < dst_channel_align) {
                const int out_offset_hi = out_offset + bhw4;
                vstore4(gate_v0.s4567, 0, gate_output + out_offset_hi);
                vstore4(up_v0.s4567, 0, up_output + out_offset_hi);
                if (row + 1 < rows) {
                    vstore4(gate_v1.s4567, 0, gate_output + out_offset_hi + 4);
                    vstore4(up_v1.s4567, 0, up_output + out_offset_hi + 4);
                }
                if (row + 2 < rows) {
                    vstore4(gate_v2.s4567, 0, gate_output + out_offset_hi + 8);
                    vstore4(up_v2.s4567, 0, up_output + out_offset_hi + 8);
                }
            }
#else
            vstore4(fused0.s0123, 0, fused_output + out_offset);
            if (row + 1 < rows) {
                vstore4(fused1.s0123, 0, fused_output + out_offset + 4);
            }
            if (row + 2 < rows) {
                vstore4(fused2.s0123, 0, fused_output + out_offset + 8);
            }
            if (oc8 + 4 < dst_channel_align) {
                const int out_offset_hi = out_offset + bhw4;
                vstore4(fused0.s4567, 0, fused_output + out_offset_hi);
                if (row + 1 < rows) {
                    vstore4(fused1.s4567, 0, fused_output + out_offset_hi + 4);
                }
                if (row + 2 < rows) {
                    vstore4(fused2.s4567, 0, fused_output + out_offset_hi + 8);
                }
            }
#endif
        }
    }
}
)";

static const char* kDecodeRepairMlpSiluDownC4Source = R"(
#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#define UCHAR4_TO_CHAR8_LOCAL(b, scale, offset, wei) \
    wei.s0 = (FLOAT)((b.s0 >> 4) - 8); \
    wei.s1 = (FLOAT)((b.s0 & 15) - 8); \
    wei.s2 = (FLOAT)((b.s1 >> 4) - 8); \
    wei.s3 = (FLOAT)((b.s1 & 15) - 8); \
    wei.s4 = (FLOAT)((b.s2 >> 4) - 8); \
    wei.s5 = (FLOAT)((b.s2 & 15) - 8); \
    wei.s6 = (FLOAT)((b.s3 >> 4) - 8); \
    wei.s7 = (FLOAT)((b.s3 & 15) - 8); \
    wei = wei * scale + offset;

inline FLOAT c4_silu_mul_scalar(FLOAT gate, FLOAT up) {
    float gate_f = (float)gate;
    float up_f = (float)up;
    float fused = gate_f * native_recip(1.0f + native_exp(-gate_f)) * up_f;
#ifdef MNN_SUPPORT_FP16
    return (half)fused;
#else
    return fused;
#endif
}

__kernel void resource_silu_down_c4_int4_image(__private int global_dim0,
                                               __private int global_dim1,
                                               __private int global_dim2,
                                               __global const FLOAT* gate,
                                               __global const FLOAT* up,
                                               __read_only image2d_t down_weight,
                                               __global const FLOAT* down_scale_offset,
                                               __global const FLOAT* down_bias,
                                               __global FLOAT* output,
                                               __private int rows,
                                               __private int dst_channel_align,
                                               __private int dst_channel_c4,
                                               __private int src_channel,
                                               __private int block_dim,
                                               __private float down_coef) {
    const int lid = get_local_id(0);
    const int oc = get_global_id(1);
    const int b4 = get_global_id(2);
    if (lid >= global_dim0 || oc >= global_dim1 || b4 >= global_dim2) {
        return;
    }
    const int row = b4 << 2;
    const int oc8 = oc << 3;
    const int loop = (src_channel + 4 - 1) / 4;
    const int bhw4 = rows << 2;
    FLOAT8 out0 = 0;
    FLOAT8 out1 = 0;
    FLOAT8 out2 = 0;
    FLOAT8 out3 = 0;
    __local FLOAT8 local0[WGS];
    __local FLOAT8 local1[WGS];
    __local FLOAT8 local2[WGS];
    __local FLOAT8 local3[WGS];
    for (int j = lid; j < loop; j += WGS) {
        const int k4 = j << 2;
#ifdef ASYMMETRIC
        FLOAT8 scale;
        FLOAT8 offset;
        {
            FLOAT16 so = vload16(0, down_scale_offset + oc8 * 2 + (k4 / block_dim) * dst_channel_c4 * 8) / (FLOAT16)down_coef;
            scale = so.s02468ace;
            offset = so.s13579bdf;
        }
#else
        FLOAT8 scale = vload8(0, down_scale_offset + oc8 + (k4 / block_dim) * dst_channel_c4 * 4) / (FLOAT8)down_coef;
        FLOAT8 offset = 0;
#endif
        const int input_offset = j * bhw4 + row * 4;
        FLOAT4 gate0 = vload4(0, gate + input_offset);
        FLOAT4 up0 = vload4(0, up + input_offset);
        FLOAT4 gate1 = row + 1 < rows ? vload4(0, gate + input_offset + 4) : (FLOAT4)0;
        FLOAT4 up1 = row + 1 < rows ? vload4(0, up + input_offset + 4) : (FLOAT4)0;
        FLOAT4 gate2 = row + 2 < rows ? vload4(0, gate + input_offset + 8) : (FLOAT4)0;
        FLOAT4 up2 = row + 2 < rows ? vload4(0, up + input_offset + 8) : (FLOAT4)0;
        FLOAT4 gate3 = row + 3 < rows ? vload4(0, gate + input_offset + 12) : (FLOAT4)0;
        FLOAT4 up3 = row + 3 < rows ? vload4(0, up + input_offset + 12) : (FLOAT4)0;
        uchar16 packed = as_uchar16(read_imagei(down_weight, SAMPLER, (int2)(j, oc)));
        FLOAT8 wei;
        UCHAR4_TO_CHAR8_LOCAL(packed.s0123, scale, offset, wei);
        out0 = mad((FLOAT8)c4_silu_mul_scalar(gate0.s0, up0.s0), wei, out0);
        out1 = mad((FLOAT8)c4_silu_mul_scalar(gate1.s0, up1.s0), wei, out1);
        out2 = mad((FLOAT8)c4_silu_mul_scalar(gate2.s0, up2.s0), wei, out2);
        out3 = mad((FLOAT8)c4_silu_mul_scalar(gate3.s0, up3.s0), wei, out3);
        UCHAR4_TO_CHAR8_LOCAL(packed.s4567, scale, offset, wei);
        out0 = mad((FLOAT8)c4_silu_mul_scalar(gate0.s1, up0.s1), wei, out0);
        out1 = mad((FLOAT8)c4_silu_mul_scalar(gate1.s1, up1.s1), wei, out1);
        out2 = mad((FLOAT8)c4_silu_mul_scalar(gate2.s1, up2.s1), wei, out2);
        out3 = mad((FLOAT8)c4_silu_mul_scalar(gate3.s1, up3.s1), wei, out3);
        UCHAR4_TO_CHAR8_LOCAL(packed.s89ab, scale, offset, wei);
        out0 = mad((FLOAT8)c4_silu_mul_scalar(gate0.s2, up0.s2), wei, out0);
        out1 = mad((FLOAT8)c4_silu_mul_scalar(gate1.s2, up1.s2), wei, out1);
        out2 = mad((FLOAT8)c4_silu_mul_scalar(gate2.s2, up2.s2), wei, out2);
        out3 = mad((FLOAT8)c4_silu_mul_scalar(gate3.s2, up3.s2), wei, out3);
        UCHAR4_TO_CHAR8_LOCAL(packed.scdef, scale, offset, wei);
        out0 = mad((FLOAT8)c4_silu_mul_scalar(gate0.s3, up0.s3), wei, out0);
        out1 = mad((FLOAT8)c4_silu_mul_scalar(gate1.s3, up1.s3), wei, out1);
        out2 = mad((FLOAT8)c4_silu_mul_scalar(gate2.s3, up2.s3), wei, out2);
        out3 = mad((FLOAT8)c4_silu_mul_scalar(gate3.s3, up3.s3), wei, out3);
    }
    local0[lid] = out0;
    local1[lid] = out1;
    local2[lid] = out2;
    local3[lid] = out3;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = WGS / 2; i > 0; i >>= 1) {
        if (lid < i) {
            local0[lid] += local0[lid + i];
            local1[lid] += local1[lid + i];
            local2[lid] += local2[lid + i];
            local3[lid] += local3[lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        FLOAT8 bias_v = vload8(0, down_bias + oc8);
        FLOAT8 out_v0 = local0[0] + bias_v;
        FLOAT8 out_v1 = local1[0] + bias_v;
        FLOAT8 out_v2 = local2[0] + bias_v;
        FLOAT8 out_v3 = local3[0] + bias_v;
        const int out_c = oc << 1;
        const int out_offset = out_c * bhw4 + row * 4;
        if (row + 3 < rows) {
            vstore16((FLOAT16)(out_v0.s0123, out_v1.s0123, out_v2.s0123, out_v3.s0123), 0, output + out_offset);
            if (oc8 + 4 < dst_channel_align) {
                const int out_offset_hi = out_offset + bhw4;
                vstore16((FLOAT16)(out_v0.s4567, out_v1.s4567, out_v2.s4567, out_v3.s4567), 0, output + out_offset_hi);
            }
        } else {
            vstore4(out_v0.s0123, 0, output + out_offset);
            if (row + 1 < rows) {
                vstore4(out_v1.s0123, 0, output + out_offset + 4);
            }
            if (row + 2 < rows) {
                vstore4(out_v2.s0123, 0, output + out_offset + 8);
            }
            if (oc8 + 4 < dst_channel_align) {
                const int out_offset_hi = out_offset + bhw4;
                vstore4(out_v0.s4567, 0, output + out_offset_hi);
                if (row + 1 < rows) {
                    vstore4(out_v1.s4567, 0, output + out_offset_hi + 4);
                }
                if (row + 2 < rows) {
                    vstore4(out_v2.s4567, 0, output + out_offset_hi + 8);
                }
            }
        }
    }
}
)";

static const char* kDecodeRepairMlpStreamedTileSource = R"(
#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#ifndef OC_TILE
#define OC_TILE 4
#endif
#define UCHAR4_TO_CHAR8_LOCAL(b, scale, offset, wei) \
    wei.s0 = (FLOAT)((b.s0 >> 4) - 8); \
    wei.s1 = (FLOAT)((b.s0 & 15) - 8); \
    wei.s2 = (FLOAT)((b.s1 >> 4) - 8); \
    wei.s3 = (FLOAT)((b.s1 & 15) - 8); \
    wei.s4 = (FLOAT)((b.s2 >> 4) - 8); \
    wei.s5 = (FLOAT)((b.s2 & 15) - 8); \
    wei.s6 = (FLOAT)((b.s3 >> 4) - 8); \
    wei.s7 = (FLOAT)((b.s3 & 15) - 8); \
    wei = wei * scale + offset;

inline FLOAT8 streamed_silu_mul8(FLOAT8 gate, FLOAT8 up) {
    float8 gate_f = convert_float8(gate);
    float8 up_f = convert_float8(up);
    float8 fused = gate_f * native_recip((float8)1.0f + native_exp(-gate_f)) * up_f;
#ifdef MNN_SUPPORT_FP16
    return convert_half8(fused);
#else
    return fused;
#endif
}

inline void streamed_accum_down4(FLOAT4 act0, FLOAT4 act1, FLOAT4 act2, FLOAT4 act3,
                                 __read_only image2d_t down_weight,
                                 __global const FLOAT* down_scale_offset,
                                 int oc_index,
                                 int k4,
                                 int dst_channel_c4,
                                 int down_block_dim,
                                 float down_coef,
                                 __private FLOAT8* out0,
                                 __private FLOAT8* out1,
                                 __private FLOAT8* out2,
                                 __private FLOAT8* out3) {
    const int oc8 = oc_index << 3;
#ifdef ASYMMETRIC
    FLOAT8 scale;
    FLOAT8 offset;
    {
        FLOAT16 so = vload16(0, down_scale_offset + oc8 * 2 + (k4 / down_block_dim) * dst_channel_c4 * 8) / (FLOAT16)down_coef;
        scale = so.s02468ace;
        offset = so.s13579bdf;
    }
#else
    FLOAT8 scale = vload8(0, down_scale_offset + oc8 + (k4 / down_block_dim) * dst_channel_c4 * 4) / (FLOAT8)down_coef;
    FLOAT8 offset = 0;
#endif
    uchar16 packed = as_uchar16(read_imagei(down_weight, SAMPLER, (int2)(k4, oc_index)));
    FLOAT8 wei;
    UCHAR4_TO_CHAR8_LOCAL(packed.s0123, scale, offset, wei);
    *out0 = mad((FLOAT8)act0.s0, wei, *out0);
    *out1 = mad((FLOAT8)act1.s0, wei, *out1);
    *out2 = mad((FLOAT8)act2.s0, wei, *out2);
    *out3 = mad((FLOAT8)act3.s0, wei, *out3);
    UCHAR4_TO_CHAR8_LOCAL(packed.s4567, scale, offset, wei);
    *out0 = mad((FLOAT8)act0.s1, wei, *out0);
    *out1 = mad((FLOAT8)act1.s1, wei, *out1);
    *out2 = mad((FLOAT8)act2.s1, wei, *out2);
    *out3 = mad((FLOAT8)act3.s1, wei, *out3);
    UCHAR4_TO_CHAR8_LOCAL(packed.s89ab, scale, offset, wei);
    *out0 = mad((FLOAT8)act0.s2, wei, *out0);
    *out1 = mad((FLOAT8)act1.s2, wei, *out1);
    *out2 = mad((FLOAT8)act2.s2, wei, *out2);
    *out3 = mad((FLOAT8)act3.s2, wei, *out3);
    UCHAR4_TO_CHAR8_LOCAL(packed.scdef, scale, offset, wei);
    *out0 = mad((FLOAT8)act0.s3, wei, *out0);
    *out1 = mad((FLOAT8)act1.s3, wei, *out1);
    *out2 = mad((FLOAT8)act2.s3, wei, *out2);
    *out3 = mad((FLOAT8)act3.s3, wei, *out3);
}

__kernel void resource_streamed_tile_mlp_int4_image(__private int global_dim0,
                                                    __private int global_dim1,
                                                    __private int global_dim2,
                                                    __global const FLOAT* input,
                                                    __read_only image2d_t gate_weight,
                                                    __global const FLOAT* gate_scale_offset,
                                                    __global const FLOAT* gate_bias,
                                                    __read_only image2d_t up_weight,
                                                    __global const FLOAT* up_scale_offset,
                                                    __global const FLOAT* up_bias,
                                                    __read_only image2d_t down_weight,
                                                    __global const FLOAT* down_scale_offset,
                                                    __global const FLOAT* down_bias,
                                                    __global FLOAT* output,
                                                    __private int rows,
                                                    __private int hidden_channel_align,
                                                    __private int hidden_channel_c4,
                                                    __private int src_channel_align,
                                                    __private int src_channel,
                                                    __private int inter_channel,
                                                    __private int inter_channel_c4,
                                                    __private int gate_block_dim,
                                                    __private int down_block_dim,
                                                    __private float gate_coef,
                                                    __private float up_coef,
                                                    __private float down_coef) {
    const int lid = get_local_id(0);
    const int oc_tile = get_global_id(1);
    const int b4 = get_global_id(2);
    if (lid >= global_dim0 || oc_tile >= global_dim1 || b4 >= global_dim2) {
        return;
    }
    const int row = b4 << 2;
    const int bhw4 = rows << 2;
    const int hidden_c8 = hidden_channel_align >> 3;
    const int inter_c8 = (inter_channel + 7) >> 3;

    FLOAT8 out0[OC_TILE];
    FLOAT8 out1[OC_TILE];
    FLOAT8 out2[OC_TILE];
    FLOAT8 out3[OC_TILE];
    for (int t = 0; t < OC_TILE; ++t) {
        const int oc_index = oc_tile * OC_TILE + t;
        FLOAT8 bias = oc_index < hidden_c8 ? vload8(0, down_bias + (oc_index << 3)) : (FLOAT8)0;
        out0[t] = bias;
        out1[t] = bias;
        out2[t] = bias;
        out3[t] = bias;
    }

    __local FLOAT8 gate_sum0[WGS];
    __local FLOAT8 gate_sum1[WGS];
    __local FLOAT8 gate_sum2[WGS];
    __local FLOAT8 gate_sum3[WGS];
    __local FLOAT8 up_sum0[WGS];
    __local FLOAT8 up_sum1[WGS];
    __local FLOAT8 up_sum2[WGS];
    __local FLOAT8 up_sum3[WGS];

    for (int ic8 = 0; ic8 < inter_c8; ++ic8) {
        const int inter_oc8 = ic8 << 3;
        FLOAT8 gate0 = 0;
        FLOAT8 gate1 = 0;
        FLOAT8 gate2 = 0;
        FLOAT8 gate3 = 0;
        FLOAT8 up0 = 0;
        FLOAT8 up1 = 0;
        FLOAT8 up2 = 0;
        FLOAT8 up3 = 0;
        for (int j = lid; j < (src_channel + 3) / 4; j += WGS) {
            const int k4 = j << 2;
#ifdef ASYMMETRIC
            FLOAT8 gate_scale;
            FLOAT8 gate_offset;
            FLOAT8 up_scale;
            FLOAT8 up_offset;
            {
                FLOAT16 so = vload16(0, gate_scale_offset + inter_oc8 * 2 + (k4 / gate_block_dim) * inter_channel_c4 * 8) / (FLOAT16)gate_coef;
                gate_scale = so.s02468ace;
                gate_offset = so.s13579bdf;
                so = vload16(0, up_scale_offset + inter_oc8 * 2 + (k4 / gate_block_dim) * inter_channel_c4 * 8) / (FLOAT16)up_coef;
                up_scale = so.s02468ace;
                up_offset = so.s13579bdf;
            }
#else
            FLOAT8 gate_scale = vload8(0, gate_scale_offset + inter_oc8 + (k4 / gate_block_dim) * inter_channel_c4 * 4) / (FLOAT8)gate_coef;
            FLOAT8 up_scale = vload8(0, up_scale_offset + inter_oc8 + (k4 / gate_block_dim) * inter_channel_c4 * 4) / (FLOAT8)up_coef;
            FLOAT8 gate_offset = 0;
            FLOAT8 up_offset = 0;
#endif
            FLOAT4 in0 = vload4(0, input + row * src_channel_align + k4);
            FLOAT4 in1 = row + 1 < rows ? vload4(0, input + (row + 1) * src_channel_align + k4) : (FLOAT4)0;
            FLOAT4 in2 = row + 2 < rows ? vload4(0, input + (row + 2) * src_channel_align + k4) : (FLOAT4)0;
            FLOAT4 in3 = row + 3 < rows ? vload4(0, input + (row + 3) * src_channel_align + k4) : (FLOAT4)0;
            uchar16 gate_packed = as_uchar16(read_imagei(gate_weight, SAMPLER, (int2)(j, ic8)));
            uchar16 up_packed = as_uchar16(read_imagei(up_weight, SAMPLER, (int2)(j, ic8)));
            FLOAT8 gate_wei;
            FLOAT8 up_wei;
            UCHAR4_TO_CHAR8_LOCAL(gate_packed.s0123, gate_scale, gate_offset, gate_wei);
            UCHAR4_TO_CHAR8_LOCAL(up_packed.s0123, up_scale, up_offset, up_wei);
            gate0 = mad((FLOAT8)in0.s0, gate_wei, gate0); gate1 = mad((FLOAT8)in1.s0, gate_wei, gate1);
            gate2 = mad((FLOAT8)in2.s0, gate_wei, gate2); gate3 = mad((FLOAT8)in3.s0, gate_wei, gate3);
            up0 = mad((FLOAT8)in0.s0, up_wei, up0); up1 = mad((FLOAT8)in1.s0, up_wei, up1);
            up2 = mad((FLOAT8)in2.s0, up_wei, up2); up3 = mad((FLOAT8)in3.s0, up_wei, up3);
            UCHAR4_TO_CHAR8_LOCAL(gate_packed.s4567, gate_scale, gate_offset, gate_wei);
            UCHAR4_TO_CHAR8_LOCAL(up_packed.s4567, up_scale, up_offset, up_wei);
            gate0 = mad((FLOAT8)in0.s1, gate_wei, gate0); gate1 = mad((FLOAT8)in1.s1, gate_wei, gate1);
            gate2 = mad((FLOAT8)in2.s1, gate_wei, gate2); gate3 = mad((FLOAT8)in3.s1, gate_wei, gate3);
            up0 = mad((FLOAT8)in0.s1, up_wei, up0); up1 = mad((FLOAT8)in1.s1, up_wei, up1);
            up2 = mad((FLOAT8)in2.s1, up_wei, up2); up3 = mad((FLOAT8)in3.s1, up_wei, up3);
            UCHAR4_TO_CHAR8_LOCAL(gate_packed.s89ab, gate_scale, gate_offset, gate_wei);
            UCHAR4_TO_CHAR8_LOCAL(up_packed.s89ab, up_scale, up_offset, up_wei);
            gate0 = mad((FLOAT8)in0.s2, gate_wei, gate0); gate1 = mad((FLOAT8)in1.s2, gate_wei, gate1);
            gate2 = mad((FLOAT8)in2.s2, gate_wei, gate2); gate3 = mad((FLOAT8)in3.s2, gate_wei, gate3);
            up0 = mad((FLOAT8)in0.s2, up_wei, up0); up1 = mad((FLOAT8)in1.s2, up_wei, up1);
            up2 = mad((FLOAT8)in2.s2, up_wei, up2); up3 = mad((FLOAT8)in3.s2, up_wei, up3);
            UCHAR4_TO_CHAR8_LOCAL(gate_packed.scdef, gate_scale, gate_offset, gate_wei);
            UCHAR4_TO_CHAR8_LOCAL(up_packed.scdef, up_scale, up_offset, up_wei);
            gate0 = mad((FLOAT8)in0.s3, gate_wei, gate0); gate1 = mad((FLOAT8)in1.s3, gate_wei, gate1);
            gate2 = mad((FLOAT8)in2.s3, gate_wei, gate2); gate3 = mad((FLOAT8)in3.s3, gate_wei, gate3);
            up0 = mad((FLOAT8)in0.s3, up_wei, up0); up1 = mad((FLOAT8)in1.s3, up_wei, up1);
            up2 = mad((FLOAT8)in2.s3, up_wei, up2); up3 = mad((FLOAT8)in3.s3, up_wei, up3);
        }
        gate_sum0[lid] = gate0; gate_sum1[lid] = gate1; gate_sum2[lid] = gate2; gate_sum3[lid] = gate3;
        up_sum0[lid] = up0; up_sum1[lid] = up1; up_sum2[lid] = up2; up_sum3[lid] = up3;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = WGS / 2; i > 0; i >>= 1) {
            if (lid < i) {
                gate_sum0[lid] += gate_sum0[lid + i]; gate_sum1[lid] += gate_sum1[lid + i];
                gate_sum2[lid] += gate_sum2[lid + i]; gate_sum3[lid] += gate_sum3[lid + i];
                up_sum0[lid] += up_sum0[lid + i]; up_sum1[lid] += up_sum1[lid + i];
                up_sum2[lid] += up_sum2[lid + i]; up_sum3[lid] += up_sum3[lid + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            FLOAT8 gate_bias_v = vload8(0, gate_bias + inter_oc8);
            FLOAT8 up_bias_v = vload8(0, up_bias + inter_oc8);
            FLOAT8 act0 = streamed_silu_mul8(gate_sum0[0] + gate_bias_v, up_sum0[0] + up_bias_v);
            FLOAT8 act1 = streamed_silu_mul8(gate_sum1[0] + gate_bias_v, up_sum1[0] + up_bias_v);
            FLOAT8 act2 = streamed_silu_mul8(gate_sum2[0] + gate_bias_v, up_sum2[0] + up_bias_v);
            FLOAT8 act3 = streamed_silu_mul8(gate_sum3[0] + gate_bias_v, up_sum3[0] + up_bias_v);
            const int down_k4 = ic8 << 1;
            for (int t = 0; t < OC_TILE; ++t) {
                const int oc_index = oc_tile * OC_TILE + t;
                if (oc_index < hidden_c8) {
                    streamed_accum_down4(act0.s0123, act1.s0123, act2.s0123, act3.s0123,
                                         down_weight, down_scale_offset, oc_index, down_k4,
                                         hidden_channel_c4, down_block_dim, down_coef,
                                         &out0[t], &out1[t], &out2[t], &out3[t]);
                    streamed_accum_down4(act0.s4567, act1.s4567, act2.s4567, act3.s4567,
                                         down_weight, down_scale_offset, oc_index, down_k4 + 1,
                                         hidden_channel_c4, down_block_dim, down_coef,
                                         &out0[t], &out1[t], &out2[t], &out3[t]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        for (int t = 0; t < OC_TILE; ++t) {
            const int oc_index = oc_tile * OC_TILE + t;
            if (oc_index >= hidden_c8) {
                continue;
            }
            const int out_c = oc_index << 1;
            const int out_offset = out_c * bhw4 + row * 4;
            if (row + 3 < rows) {
                vstore16((FLOAT16)(out0[t].s0123, out1[t].s0123, out2[t].s0123, out3[t].s0123), 0, output + out_offset);
                if ((oc_index << 3) + 4 < hidden_channel_align) {
                    const int out_offset_hi = out_offset + bhw4;
                    vstore16((FLOAT16)(out0[t].s4567, out1[t].s4567, out2[t].s4567, out3[t].s4567), 0, output + out_offset_hi);
                }
            } else {
                vstore4(out0[t].s0123, 0, output + out_offset);
                if (row + 1 < rows) {
                    vstore4(out1[t].s0123, 0, output + out_offset + 4);
                }
                if (row + 2 < rows) {
                    vstore4(out2[t].s0123, 0, output + out_offset + 8);
                }
                if ((oc_index << 3) + 4 < hidden_channel_align) {
                    const int out_offset_hi = out_offset + bhw4;
                    vstore4(out0[t].s4567, 0, output + out_offset_hi);
                    if (row + 1 < rows) {
                        vstore4(out1[t].s4567, 0, output + out_offset_hi + 4);
                    }
                    if (row + 2 < rows) {
                        vstore4(out2[t].s4567, 0, output + out_offset_hi + 8);
                    }
                }
            }
        }
    }
}
)";

static const char* kDecodeRepairMlpSiluNhwcSource = R"(
#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

inline FLOAT4 nhwc_silu_mul4(FLOAT4 gate, FLOAT4 up) {
    float4 gate_f = convert_float4(gate);
    float4 up_f = convert_float4(up);
    float4 fused = gate_f * native_recip((float4)1.0f + native_exp(-gate_f)) * up_f;
#ifdef MNN_SUPPORT_FP16
    return convert_half4(fused);
#else
    return fused;
#endif
}

__kernel void silu_c4_to_nhwc(__private int global_dim0,
                              __private int global_dim1,
                              __private int global_dim2,
                              __global const FLOAT* gate,
                              __global const FLOAT* up,
                              __global FLOAT* output,
                              __private int rows,
                              __private int channels,
                              __private int channel_align) {
    const int c4 = get_global_id(0);
    const int b4 = get_global_id(1);
    const int z = get_global_id(2);
    if (c4 >= global_dim0 || b4 >= global_dim1 || z >= global_dim2) {
        return;
    }
    const int row = b4 << 2;
    const int ch = c4 << 2;
    if (ch >= channels) {
        return;
    }
    const int bhw4 = rows << 2;
    const int in_offset = c4 * bhw4 + row * 4;
    FLOAT4 gate0 = vload4(0, gate + in_offset);
    FLOAT4 up0 = vload4(0, up + in_offset);
    vstore4(nhwc_silu_mul4(gate0, up0), 0, output + row * channel_align + ch);
    if (row + 1 < rows) {
        FLOAT4 gate1 = vload4(0, gate + in_offset + 4);
        FLOAT4 up1 = vload4(0, up + in_offset + 4);
        vstore4(nhwc_silu_mul4(gate1, up1), 0, output + (row + 1) * channel_align + ch);
    }
    if (row + 2 < rows) {
        FLOAT4 gate2 = vload4(0, gate + in_offset + 8);
        FLOAT4 up2 = vload4(0, up + in_offset + 8);
        vstore4(nhwc_silu_mul4(gate2, up2), 0, output + (row + 2) * channel_align + ch);
    }
    if (row + 3 < rows) {
        FLOAT4 gate3 = vload4(0, gate + in_offset + 12);
        FLOAT4 up3 = vload4(0, up + in_offset + 12);
        vstore4(nhwc_silu_mul4(gate3, up3), 0, output + (row + 3) * channel_align + ch);
    }
}
)";

static std::vector<uint8_t> makePackedI4Weights(int oc, int ic, int salt) {
    std::vector<uint8_t> packed(UP_DIV(oc * ic, 2), 0);
    for (int i = 0; i < oc * ic; ++i) {
        uint8_t q = static_cast<uint8_t>((i * 13 + salt) & 15);
        if ((i & 1) == 0) {
            packed[i >> 1] = static_cast<uint8_t>(q << 4);
        } else {
            packed[i >> 1] = static_cast<uint8_t>(packed[i >> 1] | q);
        }
    }
    return packed;
}

static int unpackI4Weight(const std::vector<uint8_t>& packed, int index) {
    uint8_t value = packed[index >> 1];
    int q = ((index & 1) == 0) ? static_cast<int>(value >> 4) : static_cast<int>(value & 15);
    return q - 8;
}

static std::vector<float> makeRawScale(int oc, int groups, int salt) {
    std::vector<float> scale(static_cast<size_t>(oc) * static_cast<size_t>(groups));
    for (size_t i = 0; i < scale.size(); ++i) {
        scale[i] = 0.00125f + static_cast<float>((static_cast<int>(i) * 7 + salt) % 11) * 0.000125f;
    }
    return scale;
}

static void computeRawMlpReference(const std::vector<float>& input,
                                   const std::vector<uint8_t>& gateWeight,
                                   const std::vector<float>& gateScale,
                                   const std::vector<float>& gateBias,
                                   const std::vector<uint8_t>& upWeight,
                                   const std::vector<float>& upScale,
                                   const std::vector<float>& upBias,
                                   const std::vector<uint8_t>& downWeight,
                                   const std::vector<float>& downScale,
                                   const std::vector<float>& downBias,
                                   int rows, int hidden, int inter, int quantBlock,
                                   std::vector<float>* act,
                                   std::vector<float>* output) {
    const int gateGroups = UP_DIV(hidden, quantBlock);
    const int downGroups = UP_DIV(inter, quantBlock);
    act->assign(static_cast<size_t>(rows) * inter, 0.0f);
    output->assign(static_cast<size_t>(rows) * hidden, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int j = 0; j < inter; ++j) {
            float gate = gateBias[j];
            float up = upBias[j];
            for (int k = 0; k < hidden; ++k) {
                const float x = input[r * hidden + k];
                const int group = k / quantBlock;
                gate += x * static_cast<float>(unpackI4Weight(gateWeight, j * hidden + k)) *
                        gateScale[j * gateGroups + group];
                up += x * static_cast<float>(unpackI4Weight(upWeight, j * hidden + k)) *
                      upScale[j * gateGroups + group];
            }
            (*act)[r * inter + j] = gate * (1.0f / (1.0f + std::exp(-gate))) * up;
        }
        for (int o = 0; o < hidden; ++o) {
            float sum = downBias[o];
            for (int j = 0; j < inter; ++j) {
                const int group = j / quantBlock;
                sum += (*act)[r * inter + j] *
                       static_cast<float>(unpackI4Weight(downWeight, o * inter + j)) *
                       downScale[o * downGroups + group];
            }
            (*output)[r * hidden + o] = sum;
        }
    }
}

static bool runRawKernel(OpenCLRuntime* runtime, const std::shared_ptr<KernelWrap>& kernel,
                         const std::vector<uint32_t>& gws, const std::vector<uint32_t>& lws) {
    if (runtime == nullptr || kernel == nullptr || gws.size() != 3 || lws.size() != 3) {
        return false;
    }
    std::vector<uint32_t> roundUp(3);
    for (int i = 0; i < 3; ++i) {
        roundUp[i] = ROUND_UP(gws[i], lws[i]);
    }
    cl_int ret = runtime->commandQueue().enqueueNDRangeKernel(kernel->get(), cl::NullRange,
                                                              cl::NDRange(roundUp[0], roundUp[1], roundUp[2]),
                                                              cl::NDRange(lws[0], lws[1], lws[2]));
    return checkCL(ret, "enqueue raw DecodeRepairMlpFusedFloor kernel");
}

static bool compareRawMlpOutput(const std::vector<float>& ref, const std::vector<float>& got,
                                float absTol, float relTol, float* maxAbs, float* maxRel,
                                int* badCount) {
    if (ref.size() != got.size() || maxAbs == nullptr || maxRel == nullptr || badCount == nullptr) {
        return false;
    }
    *maxAbs = 0.0f;
    *maxRel = 0.0f;
    *badCount = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float absErr = std::fabs(ref[i] - got[i]);
        const float relErr = absErr / std::max(1.0e-6f, std::fabs(ref[i]));
        *maxAbs = std::max(*maxAbs, absErr);
        *maxRel = std::max(*maxRel, relErr);
        if (absErr > absTol && relErr > relTol) {
            ++(*badCount);
        }
    }
    return true;
}

static bool runDecodeRepairMlpFusedFloorCase(const DecodeRepairMlpChainCase& c) {
    constexpr int kWgs = 64;
    DirectOpBenchOpenCL bench;
    if (!bench.valid()) {
        return false;
    }
    if (c.hidden % 8 != 0 || c.inter % 8 != 0 || c.hidden % c.quantBlock != 0 || c.inter % c.quantBlock != 0) {
        MNN_ERROR("DecodeRepairMlpFusedFloor expects hidden/inter multiples of 8 and qblock; got hidden=%d inter=%d qblock=%d\n",
                  c.hidden, c.inter, c.quantBlock);
        return false;
    }
    const int gateGroups = UP_DIV(c.hidden, c.quantBlock);
    const int downGroups = UP_DIV(c.inter, c.quantBlock);
    const auto inputData = makePattern(c.rows * c.hidden, 0.005f, 0.001f);
    const auto gateWeight = makePackedI4Weights(c.inter, c.hidden, 7);
    const auto upWeight = makePackedI4Weights(c.inter, c.hidden, 11);
    const auto downWeight = makePackedI4Weights(c.hidden, c.inter, 17);
    const auto gateScale = makeRawScale(c.inter, gateGroups, 3);
    const auto upScale = makeRawScale(c.inter, gateGroups, 5);
    const auto downScale = makeRawScale(c.hidden, downGroups, 9);
    const std::vector<float> gateBias(c.inter, 0.0f);
    const std::vector<float> upBias(c.inter, 0.0f);
    const std::vector<float> downBias(c.hidden, 0.0f);

    auto input = bench.tensorTyped<float>({c.rows * c.hidden});
    auto gateW = bench.tensorTyped<uint8_t>({static_cast<int>(gateWeight.size())});
    auto upW = bench.tensorTyped<uint8_t>({static_cast<int>(upWeight.size())});
    auto downW = bench.tensorTyped<uint8_t>({static_cast<int>(downWeight.size())});
    auto gateS = bench.tensorTyped<float>({static_cast<int>(gateScale.size())});
    auto upS = bench.tensorTyped<float>({static_cast<int>(upScale.size())});
    auto downS = bench.tensorTyped<float>({static_cast<int>(downScale.size())});
    auto gateB = bench.tensorTyped<float>({c.inter});
    auto upB = bench.tensorTyped<float>({c.inter});
    auto downB = bench.tensorTyped<float>({c.hidden});
    auto act = bench.tensorTyped<float>({c.rows * c.inter});
    auto output = bench.tensorTyped<float>({c.rows * c.hidden});
    if (!input || !gateW || !upW || !downW || !gateS || !upS || !downS ||
        !gateB || !upB || !downB || !act || !output) {
        return false;
    }
    if (!bench.writeTensor(input, inputData) ||
        !bench.writeTensorTyped<uint8_t>(gateW, gateWeight) ||
        !bench.writeTensorTyped<uint8_t>(upW, upWeight) ||
        !bench.writeTensorTyped<uint8_t>(downW, downWeight) ||
        !bench.writeTensor(gateS, gateScale) ||
        !bench.writeTensor(upS, upScale) ||
        !bench.writeTensor(downS, downScale) ||
        !bench.writeTensor(gateB, gateBias) ||
        !bench.writeTensor(upB, upBias) ||
        !bench.writeTensor(downB, downBias)) {
        return false;
    }

    auto runtime = bench.runtime();
    std::set<std::string> buildOptions;
    buildOptions.emplace("-DWGS=" + std::to_string(kWgs));
    auto gateupKernel = runtime->buildKernelFromSource(kDecodeRepairMlpFusedFloorSource, "raw_gateup_silu_int4",
                                                       buildOptions, bench.openCLBackend()->getPrecision());
    auto downKernel = runtime->buildKernelFromSource(kDecodeRepairMlpFusedFloorSource, "raw_linear_int4",
                                                     buildOptions, bench.openCLBackend()->getPrecision());
    if (gateupKernel == nullptr || downKernel == nullptr) {
        MNN_ERROR("failed to build DecodeRepairMlpFusedFloor raw kernels\n");
        return false;
    }
    std::vector<uint32_t> gateupGws = {
        kWgs,
        static_cast<uint32_t>(UP_DIV(c.inter, 8)),
        static_cast<uint32_t>(UP_DIV(c.rows, 4)),
    };
    std::vector<uint32_t> downGws = {
        kWgs,
        static_cast<uint32_t>(UP_DIV(c.hidden, 8)),
        static_cast<uint32_t>(UP_DIV(c.rows, 4)),
    };
    std::vector<uint32_t> lws = {kWgs, 1, 1};
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= gateupKernel->get().setArg(idx++, static_cast<int>(gateupGws[0]));
        ret |= gateupKernel->get().setArg(idx++, static_cast<int>(gateupGws[1]));
        ret |= gateupKernel->get().setArg(idx++, static_cast<int>(gateupGws[2]));
        ret |= gateupKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(input));
        ret |= gateupKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(gateW));
        ret |= gateupKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(gateS));
        ret |= gateupKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(gateB));
        ret |= gateupKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(upW));
        ret |= gateupKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(upS));
        ret |= gateupKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(upB));
        ret |= gateupKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(act));
        ret |= gateupKernel->get().setArg(idx++, c.rows);
        ret |= gateupKernel->get().setArg(idx++, c.hidden);
        ret |= gateupKernel->get().setArg(idx++, c.inter);
        ret |= gateupKernel->get().setArg(idx++, c.quantBlock);
        ret |= gateupKernel->get().setArg(idx++, gateGroups);
        if (!checkCL(ret, "setArg raw_gateup_silu_int4")) {
            return false;
        }
    }
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= downKernel->get().setArg(idx++, static_cast<int>(downGws[0]));
        ret |= downKernel->get().setArg(idx++, static_cast<int>(downGws[1]));
        ret |= downKernel->get().setArg(idx++, static_cast<int>(downGws[2]));
        ret |= downKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(act));
        ret |= downKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(downW));
        ret |= downKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(downS));
        ret |= downKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(downB));
        ret |= downKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(output));
        ret |= downKernel->get().setArg(idx++, c.rows);
        ret |= downKernel->get().setArg(idx++, c.inter);
        ret |= downKernel->get().setArg(idx++, c.hidden);
        ret |= downKernel->get().setArg(idx++, c.quantBlock);
        ret |= downKernel->get().setArg(idx++, downGroups);
        if (!checkCL(ret, "setArg raw_linear_int4")) {
            return false;
        }
    }

    auto runGateup = [&]() -> ErrorCode {
        return runRawKernel(runtime, gateupKernel, gateupGws, lws) ? NO_ERROR : COMPUTE_SIZE_ERROR;
    };
    auto runDown = [&]() -> ErrorCode {
        return runRawKernel(runtime, downKernel, downGws, lws) ? NO_ERROR : COMPUTE_SIZE_ERROR;
    };
    auto runChain = [&]() -> ErrorCode {
        auto code = runGateup();
        if (code != NO_ERROR) {
            return code;
        }
        return runDown();
    };

    float gateupMs = 0.0f;
    float downMs = 0.0f;
    float chainMs = 0.0f;
    OpenCLWallTimer timer;
    if (!timer.measure(runGateup, [&]() { return bench.sync(); }, c.warmup, c.repeat, &gateupMs)) {
        return false;
    }
    if (runGateup() != NO_ERROR || !bench.sync()) {
        return false;
    }
    if (!timer.measure(runDown, [&]() { return bench.sync(); }, c.warmup, c.repeat, &downMs)) {
        return false;
    }
    if (!timer.measure(runChain, [&]() { return bench.sync(); }, c.warmup, c.repeat, &chainMs)) {
        return false;
    }
    if (runChain() != NO_ERROR || !bench.sync()) {
        return false;
    }

    std::vector<float> gpuOutput;
    if (!bench.readTensorTyped<float>(output, &gpuOutput)) {
        return false;
    }
    std::vector<float> refAct;
    std::vector<float> refOutput;
    computeRawMlpReference(inputData, gateWeight, gateScale, gateBias, upWeight, upScale, upBias,
                           downWeight, downScale, downBias, c.rows, c.hidden, c.inter, c.quantBlock,
                           &refAct, &refOutput);
    float maxAbs = 0.0f;
    float maxRel = 0.0f;
    int badCount = 0;
    const float absTol = envInt("MNN_BENCH_OPENCL_MLP_FUSED_ABS_TOL_MILLI", 50) * 0.001f;
    const float relTol = envInt("MNN_BENCH_OPENCL_MLP_FUSED_REL_TOL_MILLI", 80) * 0.001f;
    if (!compareRawMlpOutput(refOutput, gpuOutput, absTol, relTol, &maxAbs, &maxRel, &badCount)) {
        return false;
    }
    MNN_PRINT("[bench_ops/opencl/perf/DecodeRepairMlpFusedFloor] %-24s rows=%d hidden=%d inter=%d qblock=%d "
              "gateup_silu=%.4f down=%.4f chain=%.4f ms max_abs=%.6f max_rel=%.6f bad=%d/%zu\n",
              c.name, c.rows, c.hidden, c.inter, c.quantBlock,
              gateupMs, downMs, chainMs, maxAbs, maxRel, badCount, gpuOutput.size());
    ::fflush(stdout);
    return badCount == 0;
}

static bool runDecodeRepairMlpResourceGateUpCase(const DecodeRepairMlpChainCase& c) {
    constexpr int kWgs = 64;
    DirectOpBenchOpenCL bench;
    if (!bench.valid()) {
        return false;
    }
    auto hidden = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    auto gate = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto up = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto splitAct = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto fusedAct = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto splitOut = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    auto fusedOut = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    auto inputNhwc = bench.tensorTyped<float>({ROUND_UP(c.hidden, 4) * ROUND_UP(c.rows, 4)});
    if (!hidden || !gate || !up || !splitAct || !fusedAct || !splitOut || !fusedOut || !inputNhwc) {
        return false;
    }
    if (!bench.writeTensor(hidden, makePattern(hidden->elementSize(), 0.0078125f))) {
        return false;
    }

    auto gateOp = makeWeightOnlyLinearConvOp(c.hidden, c.inter, c.quantBlock);
    auto upOp = makeWeightOnlyLinearConvOp(c.hidden, c.inter, c.quantBlock);
    auto siluOp = makePicSiluMulOp("bench_opencl_decode_repair_resource_gateup_silu_ref");
    auto downOp = makeWeightOnlyLinearConvOp(c.inter, c.hidden, c.quantBlock);
    std::vector<Tensor*> hiddenInput = {hidden};
    std::vector<Tensor*> gateOutput = {gate};
    std::vector<Tensor*> upOutput = {up};
    std::vector<Tensor*> siluInputs = {gate, up};
    std::vector<Tensor*> splitActOutput = {splitAct};
    std::vector<Tensor*> splitDownInput = {splitAct};
    std::vector<Tensor*> splitDownOutput = {splitOut};
    std::vector<Tensor*> fusedDownInput = {fusedAct};
    std::vector<Tensor*> fusedDownOutput = {fusedOut};

    auto gateExe = std::make_shared<OpenCL::ConvBufLowMemoryExecution>(hiddenInput, gateOutput, gateOp->get(),
                                                                       bench.openCLBackend());
    auto upExe = std::make_shared<OpenCL::ConvBufLowMemoryExecution>(hiddenInput, upOutput, upOp->get(),
                                                                     bench.openCLBackend());
    auto siluExe = bench.create(siluInputs, splitActOutput, siluOp->get());
    auto splitDownExe = std::make_shared<OpenCL::ConvBufLowMemoryExecution>(splitDownInput, splitDownOutput,
                                                                            downOp->get(), bench.openCLBackend());
    auto fusedDownExe = std::make_shared<OpenCL::ConvBufLowMemoryExecution>(fusedDownInput, fusedDownOutput,
                                                                            downOp->get(), bench.openCLBackend());
    if (!gateExe || !upExe || !siluExe || !splitDownExe || !fusedDownExe) {
        MNN_ERROR("failed to create resource-backed DecodeRepairMlp executions rows=%d\n", c.rows);
        return false;
    }
    auto code = bench.resize(gateExe.get(), hiddenInput, gateOutput);
    if (code == NO_ERROR) {
        code = bench.resize(upExe.get(), hiddenInput, upOutput);
    }
    if (code == NO_ERROR) {
        code = bench.resize(siluExe.get(), siluInputs, splitActOutput);
    }
    if (code == NO_ERROR) {
        code = bench.resize(splitDownExe.get(), splitDownInput, splitDownOutput);
    }
    if (code == NO_ERROR) {
        code = bench.resize(fusedDownExe.get(), fusedDownInput, fusedDownOutput);
    }
    if (code != NO_ERROR) {
        MNN_ERROR("resource-backed DecodeRepairMlp onResize failed rows=%d code=%d\n", c.rows, code);
        return false;
    }
    auto gateResource = gateExe->resource();
    auto upResource = upExe->resource();
    if (!gateResource || !upResource || !gateResource->mUseImage || !upResource->mUseImage ||
        !gateResource->mKernelImage || !upResource->mKernelImage ||
        !gateResource->mDequantScaleOffsetBuffer || !upResource->mDequantScaleOffsetBuffer ||
        !gateResource->mBias || !upResource->mBias) {
        MNN_ERROR("resource-backed DecodeRepairMlp requires image-backed gate/up resources rows=%d\n", c.rows);
        return false;
    }

    auto runtime = bench.runtime();
    auto preKernel = runtime->buildKernel("gemm_conv1x1_buf", "gemm_c4nhw4_to_nhwc", {},
                                          bench.openCLBackend()->getPrecision());
    std::set<std::string> buildOptions = gateResource->mBuildOptions;
    buildOptions.emplace("-DWGS=" + std::to_string(kWgs));
    buildOptions.emplace("-DQUANT_BIT=4");
    buildOptions.emplace("-DUSE_IMAGE");
    auto gateupKernel = runtime->buildKernelFromSource(kDecodeRepairMlpResourceGateUpSource,
                                                       "resource_gateup_silu_int4_image",
                                                       buildOptions, bench.openCLBackend()->getPrecision());
    if (!preKernel || !gateupKernel) {
        MNN_ERROR("failed to build resource-backed DecodeRepairMlp kernels rows=%d\n", c.rows);
        return false;
    }

    const int inputChannelAlign = ROUND_UP(c.hidden, 4);
    const int outputChannelAlign = ROUND_UP(c.inter, 4);
    const int dstChannelC4 = UP_DIV(c.inter, 4);
    const int blockDim = gateResource->mInputChannel / gateResource->mBlockSize;
    std::vector<uint32_t> preGws = {
        static_cast<uint32_t>(UP_DIV(c.rows, 4)),
        static_cast<uint32_t>(UP_DIV(c.hidden, 4)),
    };
    std::vector<uint32_t> preLws = {1, 1};
    std::vector<uint32_t> gateupGws = {
        kWgs,
        static_cast<uint32_t>(UP_DIV(c.inter, 8)),
        static_cast<uint32_t>(UP_DIV(c.rows, 4)),
    };
    std::vector<uint32_t> gateupLws = {kWgs, 1, 1};
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= preKernel->get().setArg(idx++, static_cast<int>(preGws[0]));
        ret |= preKernel->get().setArg(idx++, static_cast<int>(preGws[1]));
        ret |= preKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(hidden));
        ret |= preKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(inputNhwc));
        ret |= preKernel->get().setArg(idx++, c.rows);
        ret |= preKernel->get().setArg(idx++, c.hidden);
        ret |= preKernel->get().setArg(idx++, inputChannelAlign);
        if (!checkCL(ret, "setArg resource gateup preconvert")) {
            return false;
        }
    }
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= gateupKernel->get().setArg(idx++, static_cast<int>(gateupGws[0]));
        ret |= gateupKernel->get().setArg(idx++, static_cast<int>(gateupGws[1]));
        ret |= gateupKernel->get().setArg(idx++, static_cast<int>(gateupGws[2]));
        ret |= gateupKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(inputNhwc));
        ret |= gateupKernel->get().setArg(idx++, *gateResource->mKernelImage.get());
        ret |= gateupKernel->get().setArg(idx++, *gateResource->mDequantScaleOffsetBuffer.get());
        ret |= gateupKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(gateResource->mBias.get()));
        ret |= gateupKernel->get().setArg(idx++, *upResource->mKernelImage.get());
        ret |= gateupKernel->get().setArg(idx++, *upResource->mDequantScaleOffsetBuffer.get());
        ret |= gateupKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(upResource->mBias.get()));
        ret |= gateupKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(fusedAct));
        ret |= gateupKernel->get().setArg(idx++, c.rows);
        ret |= gateupKernel->get().setArg(idx++, outputChannelAlign);
        ret |= gateupKernel->get().setArg(idx++, dstChannelC4);
        ret |= gateupKernel->get().setArg(idx++, inputChannelAlign);
        ret |= gateupKernel->get().setArg(idx++, c.hidden);
        ret |= gateupKernel->get().setArg(idx++, blockDim);
        ret |= gateupKernel->get().setArg(idx++, gateResource->mCoef);
        ret |= gateupKernel->get().setArg(idx++, upResource->mCoef);
        if (!checkCL(ret, "setArg resource_gateup_silu_int4_image")) {
            return false;
        }
    }

    auto runPre = [&]() -> ErrorCode {
        std::vector<uint32_t> roundUp = {ROUND_UP(preGws[0], preLws[0]), ROUND_UP(preGws[1], preLws[1])};
        cl_int ret = runtime->commandQueue().enqueueNDRangeKernel(preKernel->get(), cl::NullRange,
                                                                  cl::NDRange(roundUp[0], roundUp[1]),
                                                                  cl::NDRange(preLws[0], preLws[1]));
        return checkCL(ret, "enqueue resource gateup preconvert") ? NO_ERROR : COMPUTE_SIZE_ERROR;
    };
    auto runGateup = [&]() -> ErrorCode {
        return runRawKernel(runtime, gateupKernel, gateupGws, gateupLws) ? NO_ERROR : COMPUTE_SIZE_ERROR;
    };
    auto runFusedGateup = [&]() -> ErrorCode {
        auto localCode = runPre();
        if (localCode != NO_ERROR) {
            return localCode;
        }
        return runGateup();
    };
    auto runSplitChain = [&]() -> ErrorCode {
        auto localCode = bench.execute(gateExe.get(), hiddenInput, gateOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        localCode = bench.execute(upExe.get(), hiddenInput, upOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        localCode = bench.execute(siluExe.get(), siluInputs, splitActOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        return bench.execute(splitDownExe.get(), splitDownInput, splitDownOutput);
    };
    auto runFusedChain = [&]() -> ErrorCode {
        auto localCode = runFusedGateup();
        if (localCode != NO_ERROR) {
            return localCode;
        }
        return bench.execute(fusedDownExe.get(), fusedDownInput, fusedDownOutput);
    };

    float splitChainMs = 0.0f;
    float gateupMs = 0.0f;
    float fusedChainMs = 0.0f;
    OpenCLWallTimer timer;
    if (!timer.measure(runSplitChain, [&]() { return bench.sync(); }, c.warmup, c.repeat, &splitChainMs)) {
        return false;
    }
    if (!timer.measure(runFusedGateup, [&]() { return bench.sync(); }, c.warmup, c.repeat, &gateupMs)) {
        return false;
    }
    if (!timer.measure(runFusedChain, [&]() { return bench.sync(); }, c.warmup, c.repeat, &fusedChainMs)) {
        return false;
    }
    if (runSplitChain() != NO_ERROR || runFusedChain() != NO_ERROR || !bench.sync()) {
        return false;
    }
    std::vector<float> splitData;
    std::vector<float> fusedData;
    if (!bench.readTensorTyped<float>(splitOut, &splitData) ||
        !bench.readTensorTyped<float>(fusedOut, &fusedData)) {
        return false;
    }
    float maxAbs = 0.0f;
    float maxRel = 0.0f;
    int badCount = 0;
    const float absTol = envInt("MNN_BENCH_OPENCL_MLP_RESOURCE_ABS_TOL_MILLI", 50) * 0.001f;
    const float relTol = envInt("MNN_BENCH_OPENCL_MLP_RESOURCE_REL_TOL_MILLI", 80) * 0.001f;
    if (!compareRawMlpOutput(splitData, fusedData, absTol, relTol, &maxAbs, &maxRel, &badCount)) {
        return false;
    }
    MNN_PRINT("[bench_ops/opencl/perf/DecodeRepairMlpResourceGateUp] %-24s rows=%d hidden=%d inter=%d qblock=%d "
              "split_chain=%.4f gateup_silu=%.4f fused_chain=%.4f delta=%.4f ms max_abs=%.6f max_rel=%.6f bad=%d/%zu\n",
              c.name, c.rows, c.hidden, c.inter, c.quantBlock,
              splitChainMs, gateupMs, fusedChainMs, fusedChainMs - splitChainMs,
              maxAbs, maxRel, badCount, fusedData.size());
    ::fflush(stdout);
    return badCount == 0;
}

static bool runDecodeRepairMlpSiluDownC4Case(const DecodeRepairMlpChainCase& c) {
    constexpr int kWgs = 64;
    DirectOpBenchOpenCL bench;
    if (!bench.valid()) {
        return false;
    }
    auto hidden = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    auto gate = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto up = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto splitAct = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto splitOut = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    auto fusedOut = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    if (!hidden || !gate || !up || !splitAct || !splitOut || !fusedOut) {
        return false;
    }
    if (!bench.writeTensor(hidden, makePattern(hidden->elementSize(), 0.0078125f))) {
        return false;
    }

    auto gateOp = makeWeightOnlyLinearConvOp(c.hidden, c.inter, c.quantBlock);
    auto upOp = makeWeightOnlyLinearConvOp(c.hidden, c.inter, c.quantBlock);
    auto siluOp = makePicSiluMulOp("bench_opencl_decode_repair_silu_down_c4_ref");
    auto downOp = makeWeightOnlyLinearConvOp(c.inter, c.hidden, c.quantBlock);
    std::vector<Tensor*> hiddenInput = {hidden};
    std::vector<Tensor*> gateOutput = {gate};
    std::vector<Tensor*> upOutput = {up};
    std::vector<Tensor*> siluInputs = {gate, up};
    std::vector<Tensor*> splitActOutput = {splitAct};
    std::vector<Tensor*> splitDownInput = {splitAct};
    std::vector<Tensor*> splitDownOutput = {splitOut};

    auto gateExe = std::make_shared<OpenCL::ConvBufLowMemoryExecution>(hiddenInput, gateOutput, gateOp->get(),
                                                                       bench.openCLBackend());
    auto upExe = std::make_shared<OpenCL::ConvBufLowMemoryExecution>(hiddenInput, upOutput, upOp->get(),
                                                                     bench.openCLBackend());
    auto siluExe = bench.create(siluInputs, splitActOutput, siluOp->get());
    auto splitDownExe = std::make_shared<OpenCL::ConvBufLowMemoryExecution>(splitDownInput, splitDownOutput,
                                                                            downOp->get(), bench.openCLBackend());
    if (!gateExe || !upExe || !siluExe || !splitDownExe) {
        MNN_ERROR("failed to create SiluDownC4 DecodeRepairMlp executions rows=%d\n", c.rows);
        return false;
    }
    auto code = bench.resize(gateExe.get(), hiddenInput, gateOutput);
    if (code == NO_ERROR) {
        code = bench.resize(upExe.get(), hiddenInput, upOutput);
    }
    if (code == NO_ERROR) {
        code = bench.resize(siluExe.get(), siluInputs, splitActOutput);
    }
    if (code == NO_ERROR) {
        code = bench.resize(splitDownExe.get(), splitDownInput, splitDownOutput);
    }
    if (code != NO_ERROR) {
        MNN_ERROR("SiluDownC4 DecodeRepairMlp onResize failed rows=%d code=%d\n", c.rows, code);
        return false;
    }
    auto downResource = splitDownExe->resource();
    if (!downResource || !downResource->mUseImage || !downResource->mKernelImage ||
        !downResource->mDequantScaleOffsetBuffer || !downResource->mBias) {
        MNN_ERROR("SiluDownC4 DecodeRepairMlp requires image-backed down resource rows=%d\n", c.rows);
        return false;
    }

    auto runtime = bench.runtime();
    std::set<std::string> buildOptions = downResource->mBuildOptions;
    buildOptions.emplace("-DWGS=" + std::to_string(kWgs));
    buildOptions.emplace("-DQUANT_BIT=4");
    buildOptions.emplace("-DUSE_IMAGE");
    auto siluDownKernel = runtime->buildKernelFromSource(kDecodeRepairMlpSiluDownC4Source,
                                                         "resource_silu_down_c4_int4_image",
                                                         buildOptions, bench.openCLBackend()->getPrecision());
    if (!siluDownKernel) {
        MNN_ERROR("failed to build SiluDownC4 DecodeRepairMlp kernel rows=%d\n", c.rows);
        return false;
    }

    const int outputChannelAlign = ROUND_UP(c.hidden, 4);
    const int dstChannelC4 = UP_DIV(c.hidden, 4);
    const int blockDim = downResource->mInputChannel / downResource->mBlockSize;
    std::vector<uint32_t> siluDownGws = {
        kWgs,
        static_cast<uint32_t>(UP_DIV(c.hidden, 8)),
        static_cast<uint32_t>(UP_DIV(c.rows, 4)),
    };
    std::vector<uint32_t> siluDownLws = {kWgs, 1, 1};
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= siluDownKernel->get().setArg(idx++, static_cast<int>(siluDownGws[0]));
        ret |= siluDownKernel->get().setArg(idx++, static_cast<int>(siluDownGws[1]));
        ret |= siluDownKernel->get().setArg(idx++, static_cast<int>(siluDownGws[2]));
        ret |= siluDownKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(gate));
        ret |= siluDownKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(up));
        ret |= siluDownKernel->get().setArg(idx++, *downResource->mKernelImage.get());
        ret |= siluDownKernel->get().setArg(idx++, *downResource->mDequantScaleOffsetBuffer.get());
        ret |= siluDownKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(downResource->mBias.get()));
        ret |= siluDownKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(fusedOut));
        ret |= siluDownKernel->get().setArg(idx++, c.rows);
        ret |= siluDownKernel->get().setArg(idx++, outputChannelAlign);
        ret |= siluDownKernel->get().setArg(idx++, dstChannelC4);
        ret |= siluDownKernel->get().setArg(idx++, c.inter);
        ret |= siluDownKernel->get().setArg(idx++, blockDim);
        ret |= siluDownKernel->get().setArg(idx++, downResource->mCoef);
        if (!checkCL(ret, "setArg resource_silu_down_c4_int4_image")) {
            return false;
        }
    }

    auto runSiluDown = [&]() -> ErrorCode {
        return runRawKernel(runtime, siluDownKernel, siluDownGws, siluDownLws) ? NO_ERROR : COMPUTE_SIZE_ERROR;
    };
    auto runSplitChain = [&]() -> ErrorCode {
        auto localCode = bench.execute(gateExe.get(), hiddenInput, gateOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        localCode = bench.execute(upExe.get(), hiddenInput, upOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        localCode = bench.execute(siluExe.get(), siluInputs, splitActOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        return bench.execute(splitDownExe.get(), splitDownInput, splitDownOutput);
    };
    auto runFusedChain = [&]() -> ErrorCode {
        auto localCode = bench.execute(gateExe.get(), hiddenInput, gateOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        localCode = bench.execute(upExe.get(), hiddenInput, upOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        return runSiluDown();
    };

    if (bench.execute(gateExe.get(), hiddenInput, gateOutput) != NO_ERROR ||
        bench.execute(upExe.get(), hiddenInput, upOutput) != NO_ERROR || !bench.sync()) {
        return false;
    }
    float splitChainMs = 0.0f;
    float siluDownMs = 0.0f;
    float fusedChainMs = 0.0f;
    OpenCLWallTimer timer;
    if (!timer.measure(runSplitChain, [&]() { return bench.sync(); }, c.warmup, c.repeat, &splitChainMs)) {
        return false;
    }
    if (!timer.measure(runSiluDown, [&]() { return bench.sync(); }, c.warmup, c.repeat, &siluDownMs)) {
        return false;
    }
    if (!timer.measure(runFusedChain, [&]() { return bench.sync(); }, c.warmup, c.repeat, &fusedChainMs)) {
        return false;
    }
    if (runSplitChain() != NO_ERROR || runFusedChain() != NO_ERROR || !bench.sync()) {
        return false;
    }
    std::vector<float> splitData;
    std::vector<float> fusedData;
    if (!bench.readTensorTyped<float>(splitOut, &splitData) ||
        !bench.readTensorTyped<float>(fusedOut, &fusedData)) {
        return false;
    }
    float maxAbs = 0.0f;
    float maxRel = 0.0f;
    int badCount = 0;
    const float absTol = envInt("MNN_BENCH_OPENCL_MLP_SILU_DOWN_C4_ABS_TOL_MILLI", 50) * 0.001f;
    const float relTol = envInt("MNN_BENCH_OPENCL_MLP_SILU_DOWN_C4_REL_TOL_MILLI", 80) * 0.001f;
    if (!compareRawMlpOutput(splitData, fusedData, absTol, relTol, &maxAbs, &maxRel, &badCount)) {
        return false;
    }
    MNN_PRINT("[bench_ops/opencl/perf/DecodeRepairMlpSiluDownC4] %-24s rows=%d hidden=%d inter=%d qblock=%d "
              "split_chain=%.4f silu_down=%.4f fused_chain=%.4f delta=%.4f ms "
              "max_abs=%.6f max_rel=%.6f bad=%d/%zu\n",
              c.name, c.rows, c.hidden, c.inter, c.quantBlock,
              splitChainMs, siluDownMs, fusedChainMs, fusedChainMs - splitChainMs,
              maxAbs, maxRel, badCount, fusedData.size());
    ::fflush(stdout);
    return badCount == 0;
}

static bool runDecodeRepairMlpGateUpSiluDownC4Case(const DecodeRepairMlpChainCase& c) {
    constexpr int kWgs = 64;
    DirectOpBenchOpenCL bench;
    if (!bench.valid()) {
        return false;
    }
    auto hidden = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    auto gate = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto up = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto pairGate = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto pairUp = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto splitAct = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto splitOut = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    auto fusedOut = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    auto inputNhwc = bench.tensorTyped<float>({ROUND_UP(c.hidden, 4) * ROUND_UP(c.rows, 4)});
    if (!hidden || !gate || !up || !pairGate || !pairUp || !splitAct || !splitOut || !fusedOut || !inputNhwc) {
        return false;
    }
    if (!bench.writeTensor(hidden, makePattern(hidden->elementSize(), 0.0078125f))) {
        return false;
    }

    auto gateOp = makeWeightOnlyLinearConvOp(c.hidden, c.inter, c.quantBlock);
    auto upOp = makeWeightOnlyLinearConvOp(c.hidden, c.inter, c.quantBlock);
    auto siluOp = makePicSiluMulOp("bench_opencl_decode_repair_gateup_silu_down_c4_ref");
    auto downOp = makeWeightOnlyLinearConvOp(c.inter, c.hidden, c.quantBlock);

    std::vector<Tensor*> hiddenInput = {hidden};
    std::vector<Tensor*> gateOutput = {gate};
    std::vector<Tensor*> upOutput = {up};
    std::vector<Tensor*> siluInputs = {gate, up};
    std::vector<Tensor*> splitActOutput = {splitAct};
    std::vector<Tensor*> splitDownInput = {splitAct};
    std::vector<Tensor*> splitDownOutput = {splitOut};

    auto gateExe = std::make_shared<OpenCL::ConvBufLowMemoryExecution>(hiddenInput, gateOutput, gateOp->get(),
                                                                       bench.openCLBackend());
    auto upExe = std::make_shared<OpenCL::ConvBufLowMemoryExecution>(hiddenInput, upOutput, upOp->get(),
                                                                     bench.openCLBackend());
    auto siluExe = bench.create(siluInputs, splitActOutput, siluOp->get());
    auto splitDownExe = std::make_shared<OpenCL::ConvBufLowMemoryExecution>(splitDownInput, splitDownOutput,
                                                                            downOp->get(), bench.openCLBackend());
    if (!gateExe || !upExe || !siluExe || !splitDownExe) {
        MNN_ERROR("failed to create GateUpSiluDownC4 DecodeRepairMlp executions rows=%d\n", c.rows);
        return false;
    }
    auto code = bench.resize(gateExe.get(), hiddenInput, gateOutput);
    if (code == NO_ERROR) {
        code = bench.resize(upExe.get(), hiddenInput, upOutput);
    }
    if (code == NO_ERROR) {
        code = bench.resize(siluExe.get(), siluInputs, splitActOutput);
    }
    if (code == NO_ERROR) {
        code = bench.resize(splitDownExe.get(), splitDownInput, splitDownOutput);
    }
    if (code != NO_ERROR) {
        MNN_ERROR("GateUpSiluDownC4 DecodeRepairMlp onResize failed rows=%d code=%d\n", c.rows, code);
        return false;
    }

    auto gateResource = gateExe->resource();
    auto upResource = upExe->resource();
    auto downResource = splitDownExe->resource();
    if (!gateResource || !upResource || !gateResource->mUseImage || !upResource->mUseImage ||
        !gateResource->mKernelImage || !upResource->mKernelImage ||
        !gateResource->mDequantScaleOffsetBuffer || !upResource->mDequantScaleOffsetBuffer ||
        !gateResource->mBias || !upResource->mBias) {
        MNN_ERROR("GateUpSiluDownC4 requires image-backed gate/up resources rows=%d\n", c.rows);
        return false;
    }
    if (!downResource || !downResource->mUseImage || !downResource->mKernelImage ||
        !downResource->mDequantScaleOffsetBuffer || !downResource->mBias) {
        MNN_ERROR("GateUpSiluDownC4 requires image-backed down resource rows=%d\n", c.rows);
        return false;
    }

    auto runtime = bench.runtime();
    auto preKernel = runtime->buildKernel("gemm_conv1x1_buf", "gemm_c4nhw4_to_nhwc", {},
                                          bench.openCLBackend()->getPrecision());
    std::set<std::string> gateupBuildOptions = gateResource->mBuildOptions;
    gateupBuildOptions.emplace("-DWGS=" + std::to_string(kWgs));
    gateupBuildOptions.emplace("-DQUANT_BIT=4");
    gateupBuildOptions.emplace("-DUSE_IMAGE");
    gateupBuildOptions.emplace("-DRESOURCE_GATEUP_PAIR_OUTPUT");
    auto gateupKernel = runtime->buildKernelFromSource(kDecodeRepairMlpResourceGateUpSource,
                                                       "resource_gateup_silu_int4_image",
                                                       gateupBuildOptions, bench.openCLBackend()->getPrecision());
    std::set<std::string> siluDownBuildOptions = downResource->mBuildOptions;
    siluDownBuildOptions.emplace("-DWGS=" + std::to_string(kWgs));
    siluDownBuildOptions.emplace("-DQUANT_BIT=4");
    siluDownBuildOptions.emplace("-DUSE_IMAGE");
    auto siluDownKernel = runtime->buildKernelFromSource(kDecodeRepairMlpSiluDownC4Source,
                                                         "resource_silu_down_c4_int4_image",
                                                         siluDownBuildOptions, bench.openCLBackend()->getPrecision());
    if (!preKernel || !gateupKernel || !siluDownKernel) {
        MNN_ERROR("failed to build GateUpSiluDownC4 DecodeRepairMlp kernels rows=%d\n", c.rows);
        return false;
    }

    const int inputChannelAlign = ROUND_UP(c.hidden, 4);
    const int interChannelAlign = ROUND_UP(c.inter, 4);
    const int interChannelC4 = UP_DIV(c.inter, 4);
    const int gateBlockDim = gateResource->mInputChannel / gateResource->mBlockSize;
    const int hiddenChannelAlign = ROUND_UP(c.hidden, 4);
    const int hiddenChannelC4 = UP_DIV(c.hidden, 4);
    const int downBlockDim = downResource->mInputChannel / downResource->mBlockSize;
    std::vector<uint32_t> preGws = {
        static_cast<uint32_t>(UP_DIV(c.rows, 4)),
        static_cast<uint32_t>(UP_DIV(c.hidden, 4)),
    };
    std::vector<uint32_t> preLws = {1, 1};
    std::vector<uint32_t> gateupGws = {
        kWgs,
        static_cast<uint32_t>(UP_DIV(c.inter, 8)),
        static_cast<uint32_t>(UP_DIV(c.rows, 4)),
    };
    std::vector<uint32_t> gateupLws = {kWgs, 1, 1};
    std::vector<uint32_t> siluDownGws = {
        kWgs,
        static_cast<uint32_t>(UP_DIV(c.hidden, 8)),
        static_cast<uint32_t>(UP_DIV(c.rows, 4)),
    };
    std::vector<uint32_t> siluDownLws = {kWgs, 1, 1};
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= preKernel->get().setArg(idx++, static_cast<int>(preGws[0]));
        ret |= preKernel->get().setArg(idx++, static_cast<int>(preGws[1]));
        ret |= preKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(hidden));
        ret |= preKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(inputNhwc));
        ret |= preKernel->get().setArg(idx++, c.rows);
        ret |= preKernel->get().setArg(idx++, c.hidden);
        ret |= preKernel->get().setArg(idx++, inputChannelAlign);
        if (!checkCL(ret, "setArg gateup silu-down preconvert")) {
            return false;
        }
    }
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= gateupKernel->get().setArg(idx++, static_cast<int>(gateupGws[0]));
        ret |= gateupKernel->get().setArg(idx++, static_cast<int>(gateupGws[1]));
        ret |= gateupKernel->get().setArg(idx++, static_cast<int>(gateupGws[2]));
        ret |= gateupKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(inputNhwc));
        ret |= gateupKernel->get().setArg(idx++, *gateResource->mKernelImage.get());
        ret |= gateupKernel->get().setArg(idx++, *gateResource->mDequantScaleOffsetBuffer.get());
        ret |= gateupKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(gateResource->mBias.get()));
        ret |= gateupKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(pairGate));
        ret |= gateupKernel->get().setArg(idx++, *upResource->mKernelImage.get());
        ret |= gateupKernel->get().setArg(idx++, *upResource->mDequantScaleOffsetBuffer.get());
        ret |= gateupKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(upResource->mBias.get()));
        ret |= gateupKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(pairUp));
        ret |= gateupKernel->get().setArg(idx++, c.rows);
        ret |= gateupKernel->get().setArg(idx++, interChannelAlign);
        ret |= gateupKernel->get().setArg(idx++, interChannelC4);
        ret |= gateupKernel->get().setArg(idx++, inputChannelAlign);
        ret |= gateupKernel->get().setArg(idx++, c.hidden);
        ret |= gateupKernel->get().setArg(idx++, gateBlockDim);
        ret |= gateupKernel->get().setArg(idx++, gateResource->mCoef);
        ret |= gateupKernel->get().setArg(idx++, upResource->mCoef);
        if (!checkCL(ret, "setArg resource_gateup_pair_int4_image")) {
            return false;
        }
    }
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= siluDownKernel->get().setArg(idx++, static_cast<int>(siluDownGws[0]));
        ret |= siluDownKernel->get().setArg(idx++, static_cast<int>(siluDownGws[1]));
        ret |= siluDownKernel->get().setArg(idx++, static_cast<int>(siluDownGws[2]));
        ret |= siluDownKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(pairGate));
        ret |= siluDownKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(pairUp));
        ret |= siluDownKernel->get().setArg(idx++, *downResource->mKernelImage.get());
        ret |= siluDownKernel->get().setArg(idx++, *downResource->mDequantScaleOffsetBuffer.get());
        ret |= siluDownKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(downResource->mBias.get()));
        ret |= siluDownKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(fusedOut));
        ret |= siluDownKernel->get().setArg(idx++, c.rows);
        ret |= siluDownKernel->get().setArg(idx++, hiddenChannelAlign);
        ret |= siluDownKernel->get().setArg(idx++, hiddenChannelC4);
        ret |= siluDownKernel->get().setArg(idx++, c.inter);
        ret |= siluDownKernel->get().setArg(idx++, downBlockDim);
        ret |= siluDownKernel->get().setArg(idx++, downResource->mCoef);
        if (!checkCL(ret, "setArg gateup resource_silu_down_c4_int4_image")) {
            return false;
        }
    }

    auto runPre = [&]() -> ErrorCode {
        std::vector<uint32_t> roundUp = {ROUND_UP(preGws[0], preLws[0]), ROUND_UP(preGws[1], preLws[1])};
        cl_int ret = runtime->commandQueue().enqueueNDRangeKernel(preKernel->get(), cl::NullRange,
                                                                  cl::NDRange(roundUp[0], roundUp[1]),
                                                                  cl::NDRange(preLws[0], preLws[1]));
        return checkCL(ret, "enqueue gateup silu-down preconvert") ? NO_ERROR : COMPUTE_SIZE_ERROR;
    };
    auto runGateupPair = [&]() -> ErrorCode {
        return runRawKernel(runtime, gateupKernel, gateupGws, gateupLws) ? NO_ERROR : COMPUTE_SIZE_ERROR;
    };
    auto runFusedGateupPair = [&]() -> ErrorCode {
        auto localCode = runPre();
        if (localCode != NO_ERROR) {
            return localCode;
        }
        return runGateupPair();
    };
    auto runSiluDown = [&]() -> ErrorCode {
        return runRawKernel(runtime, siluDownKernel, siluDownGws, siluDownLws) ? NO_ERROR : COMPUTE_SIZE_ERROR;
    };
    auto runSplitChain = [&]() -> ErrorCode {
        auto localCode = bench.execute(gateExe.get(), hiddenInput, gateOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        localCode = bench.execute(upExe.get(), hiddenInput, upOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        localCode = bench.execute(siluExe.get(), siluInputs, splitActOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        return bench.execute(splitDownExe.get(), splitDownInput, splitDownOutput);
    };
    auto runFusedChain = [&]() -> ErrorCode {
        auto localCode = runFusedGateupPair();
        if (localCode != NO_ERROR) {
            return localCode;
        }
        return runSiluDown();
    };

    float splitChainMs = 0.0f;
    float gateupPairMs = 0.0f;
    float siluDownMs = 0.0f;
    float fusedChainMs = 0.0f;
    OpenCLWallTimer timer;
    if (!timer.measure(runSplitChain, [&]() { return bench.sync(); }, c.warmup, c.repeat, &splitChainMs)) {
        return false;
    }
    if (!timer.measure(runFusedGateupPair, [&]() { return bench.sync(); }, c.warmup, c.repeat, &gateupPairMs)) {
        return false;
    }
    if (runFusedGateupPair() != NO_ERROR || !bench.sync()) {
        return false;
    }
    if (!timer.measure(runSiluDown, [&]() { return bench.sync(); }, c.warmup, c.repeat, &siluDownMs)) {
        return false;
    }
    if (!timer.measure(runFusedChain, [&]() { return bench.sync(); }, c.warmup, c.repeat, &fusedChainMs)) {
        return false;
    }
    if (runSplitChain() != NO_ERROR || runFusedChain() != NO_ERROR || !bench.sync()) {
        return false;
    }
    std::vector<float> splitData;
    std::vector<float> fusedData;
    if (!bench.readTensorTyped<float>(splitOut, &splitData) ||
        !bench.readTensorTyped<float>(fusedOut, &fusedData)) {
        return false;
    }
    float maxAbs = 0.0f;
    float maxRel = 0.0f;
    int badCount = 0;
    const float absTol = envInt("MNN_BENCH_OPENCL_MLP_GATEUP_SILU_DOWN_ABS_TOL_MILLI", 50) * 0.001f;
    const float relTol = envInt("MNN_BENCH_OPENCL_MLP_GATEUP_SILU_DOWN_REL_TOL_MILLI", 80) * 0.001f;
    if (!compareRawMlpOutput(splitData, fusedData, absTol, relTol, &maxAbs, &maxRel, &badCount)) {
        return false;
    }
    const int gateFamily = readSelectedCompactDenseFamily(bench, c.rows, c.hidden, c.inter, 4);
    const int downFamily = readSelectedCompactDenseFamily(bench, c.rows, c.inter, c.hidden, 4);
    MNN_PRINT("[bench_ops/opencl/perf/DecodeRepairMlpGateUpSiluDownC4] %-24s rows=%d hidden=%d inter=%d qblock=%d "
              "gate_family=%-22s down_family=%-22s split_chain=%.4f gateup_pair=%.4f silu_down=%.4f "
              "fused_chain=%.4f delta=%.4f ms max_abs=%.6f max_rel=%.6f bad=%d/%zu\n",
              c.name, c.rows, c.hidden, c.inter, c.quantBlock,
              gateFamily >= 0 ? weightOnlyCompactDenseFamilyName(static_cast<uint32_t>(gateFamily)) : "-",
              downFamily >= 0 ? weightOnlyCompactDenseFamilyName(static_cast<uint32_t>(downFamily)) : "-",
              splitChainMs, gateupPairMs, siluDownMs, fusedChainMs, fusedChainMs - splitChainMs,
              maxAbs, maxRel, badCount, fusedData.size());
    ::fflush(stdout);
    return badCount == 0;
}

static bool runDecodeRepairMlpStreamedTileCase(const DecodeRepairMlpChainCase& c) {
    constexpr int kWgs = 64;
    const int requestedOcTile = envInt("MNN_BENCH_OPENCL_MLP_STREAMED_TILE_OC", 4);
    const int ocTile = std::max(1, std::min(64, requestedOcTile));
    DirectOpBenchOpenCL bench;
    if (!bench.valid()) {
        return false;
    }
    auto hidden = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    auto gate = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto up = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto splitAct = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto splitOut = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    auto streamedOut = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    auto inputNhwc = bench.tensorTyped<float>({ROUND_UP(c.hidden, 4) * ROUND_UP(c.rows, 4)});
    if (!hidden || !gate || !up || !splitAct || !splitOut || !streamedOut || !inputNhwc) {
        return false;
    }
    if (!bench.writeTensor(hidden, makePattern(hidden->elementSize(), 0.0078125f))) {
        return false;
    }

    auto gateOp = makeWeightOnlyLinearConvOp(c.hidden, c.inter, c.quantBlock);
    auto upOp = makeWeightOnlyLinearConvOp(c.hidden, c.inter, c.quantBlock);
    auto siluOp = makePicSiluMulOp("bench_opencl_decode_repair_streamed_tile_ref");
    auto downOp = makeWeightOnlyLinearConvOp(c.inter, c.hidden, c.quantBlock);

    std::vector<Tensor*> hiddenInput = {hidden};
    std::vector<Tensor*> gateOutput = {gate};
    std::vector<Tensor*> upOutput = {up};
    std::vector<Tensor*> siluInputs = {gate, up};
    std::vector<Tensor*> splitActOutput = {splitAct};
    std::vector<Tensor*> splitDownInput = {splitAct};
    std::vector<Tensor*> splitDownOutput = {splitOut};

    auto gateExe = std::make_shared<OpenCL::ConvBufLowMemoryExecution>(hiddenInput, gateOutput, gateOp->get(),
                                                                       bench.openCLBackend());
    auto upExe = std::make_shared<OpenCL::ConvBufLowMemoryExecution>(hiddenInput, upOutput, upOp->get(),
                                                                     bench.openCLBackend());
    auto siluExe = bench.create(siluInputs, splitActOutput, siluOp->get());
    auto splitDownExe = std::make_shared<OpenCL::ConvBufLowMemoryExecution>(splitDownInput, splitDownOutput,
                                                                            downOp->get(), bench.openCLBackend());
    if (!gateExe || !upExe || !siluExe || !splitDownExe) {
        MNN_ERROR("failed to create StreamedTile DecodeRepairMlp executions rows=%d\n", c.rows);
        return false;
    }
    auto code = bench.resize(gateExe.get(), hiddenInput, gateOutput);
    if (code == NO_ERROR) {
        code = bench.resize(upExe.get(), hiddenInput, upOutput);
    }
    if (code == NO_ERROR) {
        code = bench.resize(siluExe.get(), siluInputs, splitActOutput);
    }
    if (code == NO_ERROR) {
        code = bench.resize(splitDownExe.get(), splitDownInput, splitDownOutput);
    }
    if (code != NO_ERROR) {
        MNN_ERROR("StreamedTile DecodeRepairMlp onResize failed rows=%d code=%d\n", c.rows, code);
        return false;
    }

    auto gateResource = gateExe->resource();
    auto upResource = upExe->resource();
    auto downResource = splitDownExe->resource();
    if (!gateResource || !upResource || !downResource ||
        !gateResource->mUseImage || !upResource->mUseImage || !downResource->mUseImage ||
        !gateResource->mKernelImage || !upResource->mKernelImage || !downResource->mKernelImage ||
        !gateResource->mDequantScaleOffsetBuffer || !upResource->mDequantScaleOffsetBuffer ||
        !downResource->mDequantScaleOffsetBuffer ||
        !gateResource->mBias || !upResource->mBias || !downResource->mBias) {
        MNN_ERROR("StreamedTile DecodeRepairMlp requires image-backed resources rows=%d\n", c.rows);
        return false;
    }

    auto runtime = bench.runtime();
    auto preKernel = runtime->buildKernel("gemm_conv1x1_buf", "gemm_c4nhw4_to_nhwc", {},
                                          bench.openCLBackend()->getPrecision());
    std::set<std::string> buildOptions = gateResource->mBuildOptions;
    buildOptions.insert(upResource->mBuildOptions.begin(), upResource->mBuildOptions.end());
    buildOptions.insert(downResource->mBuildOptions.begin(), downResource->mBuildOptions.end());
    buildOptions.emplace("-DWGS=" + std::to_string(kWgs));
    buildOptions.emplace("-DQUANT_BIT=4");
    buildOptions.emplace("-DUSE_IMAGE");
    buildOptions.emplace("-DOC_TILE=" + std::to_string(ocTile));
    auto streamedKernel = runtime->buildKernelFromSource(kDecodeRepairMlpStreamedTileSource,
                                                         "resource_streamed_tile_mlp_int4_image",
                                                         buildOptions, bench.openCLBackend()->getPrecision());
    if (!preKernel || !streamedKernel) {
        MNN_ERROR("failed to build StreamedTile DecodeRepairMlp kernels rows=%d\n", c.rows);
        return false;
    }

    const int inputChannelAlign = ROUND_UP(c.hidden, 4);
    const int hiddenChannelAlign = ROUND_UP(c.hidden, 4);
    const int hiddenChannelC4 = UP_DIV(c.hidden, 4);
    const int hiddenChannelC8 = UP_DIV(c.hidden, 8);
    const int interChannelC4 = UP_DIV(c.inter, 4);
    const int gateBlockDim = gateResource->mInputChannel / gateResource->mBlockSize;
    const int downBlockDim = downResource->mInputChannel / downResource->mBlockSize;
    std::vector<uint32_t> preGws = {
        static_cast<uint32_t>(UP_DIV(c.rows, 4)),
        static_cast<uint32_t>(UP_DIV(c.hidden, 4)),
    };
    std::vector<uint32_t> preLws = {1, 1};
    std::vector<uint32_t> streamedGws = {
        static_cast<uint32_t>(kWgs),
        static_cast<uint32_t>(UP_DIV(hiddenChannelC8, ocTile)),
        static_cast<uint32_t>(UP_DIV(c.rows, 4)),
    };
    std::vector<uint32_t> streamedLws = {static_cast<uint32_t>(kWgs), 1, 1};
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= preKernel->get().setArg(idx++, static_cast<int>(preGws[0]));
        ret |= preKernel->get().setArg(idx++, static_cast<int>(preGws[1]));
        ret |= preKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(hidden));
        ret |= preKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(inputNhwc));
        ret |= preKernel->get().setArg(idx++, c.rows);
        ret |= preKernel->get().setArg(idx++, c.hidden);
        ret |= preKernel->get().setArg(idx++, inputChannelAlign);
        if (!checkCL(ret, "setArg streamed tile preconvert")) {
            return false;
        }
    }
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= streamedKernel->get().setArg(idx++, static_cast<int>(streamedGws[0]));
        ret |= streamedKernel->get().setArg(idx++, static_cast<int>(streamedGws[1]));
        ret |= streamedKernel->get().setArg(idx++, static_cast<int>(streamedGws[2]));
        ret |= streamedKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(inputNhwc));
        ret |= streamedKernel->get().setArg(idx++, *gateResource->mKernelImage.get());
        ret |= streamedKernel->get().setArg(idx++, *gateResource->mDequantScaleOffsetBuffer.get());
        ret |= streamedKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(gateResource->mBias.get()));
        ret |= streamedKernel->get().setArg(idx++, *upResource->mKernelImage.get());
        ret |= streamedKernel->get().setArg(idx++, *upResource->mDequantScaleOffsetBuffer.get());
        ret |= streamedKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(upResource->mBias.get()));
        ret |= streamedKernel->get().setArg(idx++, *downResource->mKernelImage.get());
        ret |= streamedKernel->get().setArg(idx++, *downResource->mDequantScaleOffsetBuffer.get());
        ret |= streamedKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(downResource->mBias.get()));
        ret |= streamedKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(streamedOut));
        ret |= streamedKernel->get().setArg(idx++, c.rows);
        ret |= streamedKernel->get().setArg(idx++, hiddenChannelAlign);
        ret |= streamedKernel->get().setArg(idx++, hiddenChannelC4);
        ret |= streamedKernel->get().setArg(idx++, inputChannelAlign);
        ret |= streamedKernel->get().setArg(idx++, c.hidden);
        ret |= streamedKernel->get().setArg(idx++, c.inter);
        ret |= streamedKernel->get().setArg(idx++, interChannelC4);
        ret |= streamedKernel->get().setArg(idx++, gateBlockDim);
        ret |= streamedKernel->get().setArg(idx++, downBlockDim);
        ret |= streamedKernel->get().setArg(idx++, gateResource->mCoef);
        ret |= streamedKernel->get().setArg(idx++, upResource->mCoef);
        ret |= streamedKernel->get().setArg(idx++, downResource->mCoef);
        if (!checkCL(ret, "setArg resource_streamed_tile_mlp_int4_image")) {
            return false;
        }
    }

    auto runPre = [&]() -> ErrorCode {
        std::vector<uint32_t> roundUp = {ROUND_UP(preGws[0], preLws[0]), ROUND_UP(preGws[1], preLws[1])};
        cl_int ret = runtime->commandQueue().enqueueNDRangeKernel(preKernel->get(), cl::NullRange,
                                                                  cl::NDRange(roundUp[0], roundUp[1]),
                                                                  cl::NDRange(preLws[0], preLws[1]));
        return checkCL(ret, "enqueue streamed tile preconvert") ? NO_ERROR : COMPUTE_SIZE_ERROR;
    };
    auto runStreamKernel = [&]() -> ErrorCode {
        return runRawKernel(runtime, streamedKernel, streamedGws, streamedLws) ? NO_ERROR : COMPUTE_SIZE_ERROR;
    };
    auto runStreamChain = [&]() -> ErrorCode {
        auto localCode = runPre();
        if (localCode != NO_ERROR) {
            return localCode;
        }
        return runStreamKernel();
    };
    auto runSplitChain = [&]() -> ErrorCode {
        auto localCode = bench.execute(gateExe.get(), hiddenInput, gateOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        localCode = bench.execute(upExe.get(), hiddenInput, upOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        localCode = bench.execute(siluExe.get(), siluInputs, splitActOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        return bench.execute(splitDownExe.get(), splitDownInput, splitDownOutput);
    };

    if (runPre() != NO_ERROR || !bench.sync()) {
        return false;
    }
    float splitChainMs = 0.0f;
    float streamKernelMs = 0.0f;
    float streamChainMs = 0.0f;
    OpenCLWallTimer timer;
    if (!timer.measure(runSplitChain, [&]() { return bench.sync(); }, c.warmup, c.repeat, &splitChainMs)) {
        return false;
    }
    if (!timer.measure(runStreamKernel, [&]() { return bench.sync(); }, c.warmup, c.repeat, &streamKernelMs)) {
        return false;
    }
    if (!timer.measure(runStreamChain, [&]() { return bench.sync(); }, c.warmup, c.repeat, &streamChainMs)) {
        return false;
    }
    if (runSplitChain() != NO_ERROR || runStreamChain() != NO_ERROR || !bench.sync()) {
        return false;
    }
    std::vector<float> splitData;
    std::vector<float> streamedData;
    if (!bench.readTensorTyped<float>(splitOut, &splitData) ||
        !bench.readTensorTyped<float>(streamedOut, &streamedData)) {
        return false;
    }
    float maxAbs = 0.0f;
    float maxRel = 0.0f;
    int badCount = 0;
    const float absTol = envInt("MNN_BENCH_OPENCL_MLP_STREAMED_TILE_ABS_TOL_MILLI", 50) * 0.001f;
    const float relTol = envInt("MNN_BENCH_OPENCL_MLP_STREAMED_TILE_REL_TOL_MILLI", 80) * 0.001f;
    if (!compareRawMlpOutput(splitData, streamedData, absTol, relTol, &maxAbs, &maxRel, &badCount)) {
        return false;
    }
    const int gateFamily = readSelectedCompactDenseFamily(bench, c.rows, c.hidden, c.inter, 4);
    const int downFamily = readSelectedCompactDenseFamily(bench, c.rows, c.inter, c.hidden, 4);
    MNN_PRINT("[bench_ops/opencl/perf/DecodeRepairMlpStreamedTile] %-24s rows=%d hidden=%d inter=%d qblock=%d "
              "oc_tile=%d gate_family=%-22s down_family=%-22s split_chain=%.4f stream_kernel=%.4f "
              "stream_chain=%.4f delta=%.4f ms max_abs=%.6f max_rel=%.6f bad=%d/%zu\n",
              c.name, c.rows, c.hidden, c.inter, c.quantBlock, ocTile,
              gateFamily >= 0 ? weightOnlyCompactDenseFamilyName(static_cast<uint32_t>(gateFamily)) : "-",
              downFamily >= 0 ? weightOnlyCompactDenseFamilyName(static_cast<uint32_t>(downFamily)) : "-",
              splitChainMs, streamKernelMs, streamChainMs, streamChainMs - splitChainMs,
              maxAbs, maxRel, badCount, streamedData.size());
    ::fflush(stdout);
    return badCount == 0;
}

static bool runDecodeRepairMlpSiluNhwcDownCase(const DecodeRepairMlpChainCase& c) {
    const int kWgs = envInt("MNN_BENCH_OPENCL_MLP_SILU_NHWC_DOWN_WGS", 64);
    DirectOpBenchOpenCL bench;
    if (!bench.valid()) {
        return false;
    }
    auto hidden = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    auto gate = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto up = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto splitAct = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto splitOut = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    auto fusedOut = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    auto actNhwc = bench.tensorTyped<float>({ROUND_UP(c.inter, 4) * ROUND_UP(c.rows, 4)});
    if (!hidden || !gate || !up || !splitAct || !splitOut || !fusedOut || !actNhwc) {
        return false;
    }
    if (!bench.writeTensor(hidden, makePattern(hidden->elementSize(), 0.0078125f))) {
        return false;
    }

    auto gateOp = makeWeightOnlyLinearConvOp(c.hidden, c.inter, c.quantBlock);
    auto upOp = makeWeightOnlyLinearConvOp(c.hidden, c.inter, c.quantBlock);
    auto siluOp = makePicSiluMulOp("bench_opencl_decode_repair_silu_nhwc_down_ref");
    auto downOp = makeWeightOnlyLinearConvOp(c.inter, c.hidden, c.quantBlock);
    std::vector<Tensor*> hiddenInput = {hidden};
    std::vector<Tensor*> gateOutput = {gate};
    std::vector<Tensor*> upOutput = {up};
    std::vector<Tensor*> siluInputs = {gate, up};
    std::vector<Tensor*> splitActOutput = {splitAct};
    std::vector<Tensor*> splitDownInput = {splitAct};
    std::vector<Tensor*> splitDownOutput = {splitOut};

    auto gateExe = std::make_shared<OpenCL::ConvBufLowMemoryExecution>(hiddenInput, gateOutput, gateOp->get(),
                                                                       bench.openCLBackend());
    auto upExe = std::make_shared<OpenCL::ConvBufLowMemoryExecution>(hiddenInput, upOutput, upOp->get(),
                                                                     bench.openCLBackend());
    auto siluExe = bench.create(siluInputs, splitActOutput, siluOp->get());
    auto splitDownExe = std::make_shared<OpenCL::ConvBufLowMemoryExecution>(splitDownInput, splitDownOutput,
                                                                            downOp->get(), bench.openCLBackend());
    if (!gateExe || !upExe || !siluExe || !splitDownExe) {
        MNN_ERROR("failed to create SiluNhwcDown DecodeRepairMlp executions rows=%d\n", c.rows);
        return false;
    }
    auto code = bench.resize(gateExe.get(), hiddenInput, gateOutput);
    if (code == NO_ERROR) {
        code = bench.resize(upExe.get(), hiddenInput, upOutput);
    }
    if (code == NO_ERROR) {
        code = bench.resize(siluExe.get(), siluInputs, splitActOutput);
    }
    if (code == NO_ERROR) {
        code = bench.resize(splitDownExe.get(), splitDownInput, splitDownOutput);
    }
    if (code != NO_ERROR) {
        MNN_ERROR("SiluNhwcDown DecodeRepairMlp onResize failed rows=%d code=%d\n", c.rows, code);
        return false;
    }
    auto downResource = splitDownExe->resource();
    if (!downResource || !downResource->mUseImage || !downResource->mKernelImage ||
        !downResource->mDequantScaleOffsetBuffer || !downResource->mBias) {
        MNN_ERROR("SiluNhwcDown DecodeRepairMlp requires image-backed down resource rows=%d\n", c.rows);
        return false;
    }

    auto runtime = bench.runtime();
    auto siluNhwcKernel = runtime->buildKernelFromSource(kDecodeRepairMlpSiluNhwcSource,
                                                         "silu_c4_to_nhwc",
                                                         {}, bench.openCLBackend()->getPrecision());
    std::set<std::string> buildOptions = downResource->mBuildOptions;
    buildOptions.emplace("-DWGS=" + std::to_string(kWgs));
    buildOptions.emplace("-DQUANT_BIT=4");
    buildOptions.emplace("-DUSE_IMAGE");
    buildOptions.emplace("-DCOMPUTE_BATCH");
    buildOptions.emplace("-DOUTPUT_C4NHW4");
    buildOptions.emplace("-DOUTPUT_BHW=" + std::to_string(c.rows));
    auto downKernel = runtime->buildKernel("gemv_conv1x1_buf", "gemv_conv_c8_buf",
                                           buildOptions, bench.openCLBackend()->getPrecision());
    if (!siluNhwcKernel || !downKernel) {
        MNN_ERROR("failed to build SiluNhwcDown DecodeRepairMlp kernels rows=%d\n", c.rows);
        return false;
    }

    const int inputChannelAlign = ROUND_UP(c.inter, 4);
    const int outputChannelAlign8 = ROUND_UP(c.hidden, 8);
    const int outputChannelBlocks = UP_DIV(c.hidden, 4);
    const int inputChannelBlocks = UP_DIV(c.inter, 4);
    const int blockDim = downResource->mInputChannel / downResource->mBlockSize;
    std::vector<uint32_t> siluGws = {
        static_cast<uint32_t>(UP_DIV(c.inter, 4)),
        static_cast<uint32_t>(UP_DIV(c.rows, 4)),
        1,
    };
    std::vector<uint32_t> siluLws = {16, 1, 1};
    std::vector<uint32_t> downGws = {
        static_cast<uint32_t>(kWgs),
        static_cast<uint32_t>(UP_DIV(c.hidden, 8)),
        static_cast<uint32_t>(UP_DIV(c.rows, 4)),
    };
    std::vector<uint32_t> downLws = {static_cast<uint32_t>(kWgs), 1, 1};
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= siluNhwcKernel->get().setArg(idx++, static_cast<int>(siluGws[0]));
        ret |= siluNhwcKernel->get().setArg(idx++, static_cast<int>(siluGws[1]));
        ret |= siluNhwcKernel->get().setArg(idx++, static_cast<int>(siluGws[2]));
        ret |= siluNhwcKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(gate));
        ret |= siluNhwcKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(up));
        ret |= siluNhwcKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(actNhwc));
        ret |= siluNhwcKernel->get().setArg(idx++, c.rows);
        ret |= siluNhwcKernel->get().setArg(idx++, c.inter);
        ret |= siluNhwcKernel->get().setArg(idx++, inputChannelAlign);
        if (!checkCL(ret, "setArg silu_c4_to_nhwc")) {
            return false;
        }
    }
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= downKernel->get().setArg(idx++, static_cast<int>(downGws[0]));
        ret |= downKernel->get().setArg(idx++, static_cast<int>(downGws[1]));
        ret |= downKernel->get().setArg(idx++, static_cast<int>(downGws[2]));
        ret |= downKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(actNhwc));
        ret |= downKernel->get().setArg(idx++, *downResource->mKernelImage.get());
        ret |= downKernel->get().setArg(idx++, *downResource->mDequantScaleOffsetBuffer.get());
        ret |= downKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(downResource->mBias.get()));
        ret |= downKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(fusedOut));
        ret |= downKernel->get().setArg(idx++, outputChannelAlign8);
        ret |= downKernel->get().setArg(idx++, inputChannelAlign);
        ret |= downKernel->get().setArg(idx++, outputChannelBlocks);
        ret |= downKernel->get().setArg(idx++, inputChannelBlocks);
        ret |= downKernel->get().setArg(idx++, c.inter);
        ret |= downKernel->get().setArg(idx++, static_cast<int>(downResource->mBlockSize));
        ret |= downKernel->get().setArg(idx++, blockDim);
        ret |= downKernel->get().setArg(idx++, downResource->mCoef);
        if (!checkCL(ret, "setArg silu_nhwc direct down projection")) {
            return false;
        }
    }

    auto runSiluNhwc = [&]() -> ErrorCode {
        return runRawKernel(runtime, siluNhwcKernel, siluGws, siluLws) ? NO_ERROR : COMPUTE_SIZE_ERROR;
    };
    auto runDownProjection = [&]() -> ErrorCode {
        return runRawKernel(runtime, downKernel, downGws, downLws) ? NO_ERROR : COMPUTE_SIZE_ERROR;
    };
    auto runSplitChain = [&]() -> ErrorCode {
        auto localCode = bench.execute(gateExe.get(), hiddenInput, gateOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        localCode = bench.execute(upExe.get(), hiddenInput, upOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        localCode = bench.execute(siluExe.get(), siluInputs, splitActOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        return bench.execute(splitDownExe.get(), splitDownInput, splitDownOutput);
    };
    auto runFusedChain = [&]() -> ErrorCode {
        auto localCode = bench.execute(gateExe.get(), hiddenInput, gateOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        localCode = bench.execute(upExe.get(), hiddenInput, upOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        localCode = runSiluNhwc();
        if (localCode != NO_ERROR) {
            return localCode;
        }
        return runDownProjection();
    };

    if (bench.execute(gateExe.get(), hiddenInput, gateOutput) != NO_ERROR ||
        bench.execute(upExe.get(), hiddenInput, upOutput) != NO_ERROR ||
        runSiluNhwc() != NO_ERROR || !bench.sync()) {
        return false;
    }
    float splitChainMs = 0.0f;
    float siluNhwcMs = 0.0f;
    float directDownMs = 0.0f;
    float fusedChainMs = 0.0f;
    OpenCLWallTimer timer;
    if (!timer.measure(runSplitChain, [&]() { return bench.sync(); }, c.warmup, c.repeat, &splitChainMs)) {
        return false;
    }
    if (!timer.measure(runSiluNhwc, [&]() { return bench.sync(); }, c.warmup, c.repeat, &siluNhwcMs)) {
        return false;
    }
    if (!timer.measure(runDownProjection, [&]() { return bench.sync(); }, c.warmup, c.repeat, &directDownMs)) {
        return false;
    }
    if (!timer.measure(runFusedChain, [&]() { return bench.sync(); }, c.warmup, c.repeat, &fusedChainMs)) {
        return false;
    }
    if (runSplitChain() != NO_ERROR || runFusedChain() != NO_ERROR || !bench.sync()) {
        return false;
    }
    std::vector<float> splitData;
    std::vector<float> fusedData;
    if (!bench.readTensorTyped<float>(splitOut, &splitData) ||
        !bench.readTensorTyped<float>(fusedOut, &fusedData)) {
        return false;
    }
    float maxAbs = 0.0f;
    float maxRel = 0.0f;
    int badCount = 0;
    const float absTol = envInt("MNN_BENCH_OPENCL_MLP_SILU_NHWC_DOWN_ABS_TOL_MILLI", 50) * 0.001f;
    const float relTol = envInt("MNN_BENCH_OPENCL_MLP_SILU_NHWC_DOWN_REL_TOL_MILLI", 80) * 0.001f;
    if (!compareRawMlpOutput(splitData, fusedData, absTol, relTol, &maxAbs, &maxRel, &badCount)) {
        return false;
    }
    const int gateFamily = readSelectedCompactDenseFamily(bench, c.rows, c.hidden, c.inter, 4);
    const int downFamily = readSelectedCompactDenseFamily(bench, c.rows, c.inter, c.hidden, 4);
    MNN_PRINT("[bench_ops/opencl/perf/DecodeRepairMlpSiluNhwcDown] %-24s rows=%d hidden=%d inter=%d qblock=%d "
              "gate_family=%-22s down_family=%-22s split_chain=%.4f silu_nhwc=%.4f "
              "direct_down=%.4f fused_chain=%.4f delta=%.4f ms max_abs=%.6f max_rel=%.6f bad=%d/%zu\n",
              c.name, c.rows, c.hidden, c.inter, c.quantBlock,
              gateFamily >= 0 ? weightOnlyCompactDenseFamilyName(static_cast<uint32_t>(gateFamily)) : "-",
              downFamily >= 0 ? weightOnlyCompactDenseFamilyName(static_cast<uint32_t>(downFamily)) : "-",
              splitChainMs, siluNhwcMs, directDownMs, fusedChainMs, fusedChainMs - splitChainMs,
              maxAbs, maxRel, badCount, fusedData.size());
    ::fflush(stdout);
    return badCount == 0;
}

static bool runDecodeRepairMlpSharedInputGateUpCase(const DecodeRepairMlpChainCase& c) {
    const int kWgs = envInt("MNN_BENCH_OPENCL_MLP_SHARED_INPUT_WGS", 64);
    DirectOpBenchOpenCL bench;
    if (!bench.valid()) {
        return false;
    }
    auto hidden = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    auto gate = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto up = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto sharedGate = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto sharedUp = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto splitAct = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto sharedAct = bench.tensor({c.rows, c.inter, 1, 1}, Tensor::CAFFE);
    auto splitOut = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    auto sharedOut = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    auto sharedNhwcOut = bench.tensor({c.rows, c.hidden, 1, 1}, Tensor::CAFFE);
    auto inputNhwc = bench.tensorTyped<float>({ROUND_UP(c.hidden, 4) * ROUND_UP(c.rows, 4)});
    auto actNhwc = bench.tensorTyped<float>({ROUND_UP(c.inter, 4) * ROUND_UP(c.rows, 4)});
    if (!hidden || !gate || !up || !sharedGate || !sharedUp || !splitAct || !sharedAct ||
        !splitOut || !sharedOut || !sharedNhwcOut || !inputNhwc || !actNhwc) {
        return false;
    }
    if (!bench.writeTensor(hidden, makePattern(hidden->elementSize(), 0.0078125f))) {
        return false;
    }

    auto gateOp = makeWeightOnlyLinearConvOp(c.hidden, c.inter, c.quantBlock);
    auto upOp = makeWeightOnlyLinearConvOp(c.hidden, c.inter, c.quantBlock);
    auto siluOp = makePicSiluMulOp("bench_opencl_decode_repair_shared_input_gateup_silu");
    auto downOp = makeWeightOnlyLinearConvOp(c.inter, c.hidden, c.quantBlock);

    std::vector<Tensor*> hiddenInput = {hidden};
    std::vector<Tensor*> gateOutput = {gate};
    std::vector<Tensor*> upOutput = {up};
    std::vector<Tensor*> sharedGateOutput = {sharedGate};
    std::vector<Tensor*> sharedUpOutput = {sharedUp};
    std::vector<Tensor*> siluInputs = {gate, up};
    std::vector<Tensor*> sharedSiluInputs = {sharedGate, sharedUp};
    std::vector<Tensor*> splitActOutput = {splitAct};
    std::vector<Tensor*> sharedActOutput = {sharedAct};
    std::vector<Tensor*> splitDownInput = {splitAct};
    std::vector<Tensor*> sharedDownInput = {sharedAct};
    std::vector<Tensor*> splitDownOutput = {splitOut};
    std::vector<Tensor*> sharedDownOutput = {sharedOut};

    auto gateExe = std::make_shared<OpenCL::ConvBufLowMemoryExecution>(hiddenInput, gateOutput, gateOp->get(),
                                                                       bench.openCLBackend());
    auto upExe = std::make_shared<OpenCL::ConvBufLowMemoryExecution>(hiddenInput, upOutput, upOp->get(),
                                                                     bench.openCLBackend());
    auto siluExe = bench.create(siluInputs, splitActOutput, siluOp->get());
    auto sharedSiluExe = bench.create(sharedSiluInputs, sharedActOutput, siluOp->get());
    auto splitDownExe = std::make_shared<OpenCL::ConvBufLowMemoryExecution>(splitDownInput, splitDownOutput,
                                                                            downOp->get(), bench.openCLBackend());
    auto sharedDownExe = std::make_shared<OpenCL::ConvBufLowMemoryExecution>(sharedDownInput, sharedDownOutput,
                                                                             downOp->get(), bench.openCLBackend());
    if (!gateExe || !upExe || !siluExe || !sharedSiluExe || !splitDownExe || !sharedDownExe) {
        MNN_ERROR("failed to create SharedInputGateUp DecodeRepairMlp executions rows=%d\n", c.rows);
        return false;
    }
    auto code = bench.resize(gateExe.get(), hiddenInput, gateOutput);
    if (code == NO_ERROR) {
        code = bench.resize(upExe.get(), hiddenInput, upOutput);
    }
    if (code == NO_ERROR) {
        code = bench.resize(siluExe.get(), siluInputs, splitActOutput);
    }
    if (code == NO_ERROR) {
        code = bench.resize(sharedSiluExe.get(), sharedSiluInputs, sharedActOutput);
    }
    if (code == NO_ERROR) {
        code = bench.resize(splitDownExe.get(), splitDownInput, splitDownOutput);
    }
    if (code == NO_ERROR) {
        code = bench.resize(sharedDownExe.get(), sharedDownInput, sharedDownOutput);
    }
    if (code != NO_ERROR) {
        MNN_ERROR("SharedInputGateUp DecodeRepairMlp onResize failed rows=%d code=%d\n", c.rows, code);
        return false;
    }

    auto gateResource = gateExe->resource();
    auto upResource = upExe->resource();
    auto downResource = sharedDownExe->resource();
    if (!gateResource || !upResource || !gateResource->mUseImage || !upResource->mUseImage ||
        !gateResource->mKernelImage || !upResource->mKernelImage ||
        !gateResource->mDequantScaleOffsetBuffer || !upResource->mDequantScaleOffsetBuffer ||
        !gateResource->mBias || !upResource->mBias) {
        MNN_ERROR("SharedInputGateUp DecodeRepairMlp requires image-backed gate/up resources rows=%d\n", c.rows);
        return false;
    }
    if (!downResource || !downResource->mUseImage || !downResource->mKernelImage ||
        !downResource->mDequantScaleOffsetBuffer || !downResource->mBias) {
        MNN_ERROR("SharedInputGateUp DecodeRepairMlp requires image-backed down resource rows=%d\n", c.rows);
        return false;
    }

    auto runtime = bench.runtime();
    auto preKernel = runtime->buildKernel("gemm_conv1x1_buf", "gemm_c4nhw4_to_nhwc", {},
                                          bench.openCLBackend()->getPrecision());
    auto siluNhwcKernel = runtime->buildKernelFromSource(kDecodeRepairMlpSiluNhwcSource,
                                                         "silu_c4_to_nhwc",
                                                         {}, bench.openCLBackend()->getPrecision());
    std::set<std::string> buildOptions = gateResource->mBuildOptions;
    buildOptions.emplace("-DWGS=" + std::to_string(kWgs));
    buildOptions.emplace("-DQUANT_BIT=4");
    buildOptions.emplace("-DUSE_IMAGE");
    buildOptions.emplace("-DCOMPUTE_BATCH");
    buildOptions.emplace("-DOUTPUT_C4NHW4");
    buildOptions.emplace("-DOUTPUT_BHW=" + std::to_string(c.rows));
    auto gateKernel = runtime->buildKernel("gemv_conv1x1_buf", "gemv_conv_c8_buf",
                                           buildOptions, bench.openCLBackend()->getPrecision());
    auto upKernel = runtime->buildKernel("gemv_conv1x1_buf", "gemv_conv_c8_buf",
                                         buildOptions, bench.openCLBackend()->getPrecision());
    std::set<std::string> downBuildOptions = downResource->mBuildOptions;
    downBuildOptions.emplace("-DWGS=" + std::to_string(kWgs));
    downBuildOptions.emplace("-DQUANT_BIT=4");
    downBuildOptions.emplace("-DUSE_IMAGE");
    downBuildOptions.emplace("-DCOMPUTE_BATCH");
    downBuildOptions.emplace("-DOUTPUT_C4NHW4");
    downBuildOptions.emplace("-DOUTPUT_BHW=" + std::to_string(c.rows));
    auto downKernel = runtime->buildKernel("gemv_conv1x1_buf", "gemv_conv_c8_buf",
                                           downBuildOptions, bench.openCLBackend()->getPrecision());
    if (!preKernel || !siluNhwcKernel || !gateKernel || !upKernel || !downKernel) {
        MNN_ERROR("failed to build SharedInputGateUp DecodeRepairMlp kernels rows=%d\n", c.rows);
        return false;
    }

    const int inputChannelAlign = ROUND_UP(c.hidden, 4);
    const int outputChannelAlign8 = ROUND_UP(c.inter, 8);
    const int outputChannelBlocks = UP_DIV(c.inter, 4);
    const int inputChannelBlocks = UP_DIV(c.hidden, 4);
    const int gateBlockDim = gateResource->mInputChannel / gateResource->mBlockSize;
    const int upBlockDim = upResource->mInputChannel / upResource->mBlockSize;
    const int actChannelAlign = ROUND_UP(c.inter, 4);
    const int downOutputChannelAlign8 = ROUND_UP(c.hidden, 8);
    const int downOutputChannelBlocks = UP_DIV(c.hidden, 4);
    const int downInputChannelBlocks = UP_DIV(c.inter, 4);
    const int downBlockDim = downResource->mInputChannel / downResource->mBlockSize;
    std::vector<uint32_t> preGws = {
        static_cast<uint32_t>(UP_DIV(c.rows, 4)),
        static_cast<uint32_t>(UP_DIV(c.hidden, 4)),
    };
    std::vector<uint32_t> preLws = {1, 1};
    std::vector<uint32_t> projGws = {
        static_cast<uint32_t>(kWgs),
        static_cast<uint32_t>(UP_DIV(c.inter, 8)),
        static_cast<uint32_t>(UP_DIV(c.rows, 4)),
    };
    std::vector<uint32_t> projLws = {static_cast<uint32_t>(kWgs), 1, 1};
    std::vector<uint32_t> siluGws = {
        static_cast<uint32_t>(UP_DIV(c.inter, 4)),
        static_cast<uint32_t>(UP_DIV(c.rows, 4)),
        1,
    };
    std::vector<uint32_t> siluLws = {16, 1, 1};
    std::vector<uint32_t> downGws = {
        static_cast<uint32_t>(kWgs),
        static_cast<uint32_t>(UP_DIV(c.hidden, 8)),
        static_cast<uint32_t>(UP_DIV(c.rows, 4)),
    };
    std::vector<uint32_t> downLws = {static_cast<uint32_t>(kWgs), 1, 1};
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= preKernel->get().setArg(idx++, static_cast<int>(preGws[0]));
        ret |= preKernel->get().setArg(idx++, static_cast<int>(preGws[1]));
        ret |= preKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(hidden));
        ret |= preKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(inputNhwc));
        ret |= preKernel->get().setArg(idx++, c.rows);
        ret |= preKernel->get().setArg(idx++, c.hidden);
        ret |= preKernel->get().setArg(idx++, inputChannelAlign);
        if (!checkCL(ret, "setArg shared gateup preconvert")) {
            return false;
        }
    }
    auto setProjArgs = [&](const std::shared_ptr<KernelWrap>& kernel,
                           const std::shared_ptr<OpenCL::ConvBufResource>& resource,
                           Tensor* output, int blockDim, const char* name) -> bool {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= kernel->get().setArg(idx++, static_cast<int>(projGws[0]));
        ret |= kernel->get().setArg(idx++, static_cast<int>(projGws[1]));
        ret |= kernel->get().setArg(idx++, static_cast<int>(projGws[2]));
        ret |= kernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(inputNhwc));
        ret |= kernel->get().setArg(idx++, *resource->mKernelImage.get());
        ret |= kernel->get().setArg(idx++, *resource->mDequantScaleOffsetBuffer.get());
        ret |= kernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(resource->mBias.get()));
        ret |= kernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(output));
        ret |= kernel->get().setArg(idx++, outputChannelAlign8);
        ret |= kernel->get().setArg(idx++, inputChannelAlign);
        ret |= kernel->get().setArg(idx++, outputChannelBlocks);
        ret |= kernel->get().setArg(idx++, inputChannelBlocks);
        ret |= kernel->get().setArg(idx++, c.hidden);
        ret |= kernel->get().setArg(idx++, static_cast<int>(resource->mBlockSize));
        ret |= kernel->get().setArg(idx++, blockDim);
        ret |= kernel->get().setArg(idx++, resource->mCoef);
        return checkCL(ret, name);
    };
    if (!setProjArgs(gateKernel, gateResource, sharedGate, gateBlockDim, "setArg shared gate projection") ||
        !setProjArgs(upKernel, upResource, sharedUp, upBlockDim, "setArg shared up projection")) {
        return false;
    }
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= siluNhwcKernel->get().setArg(idx++, static_cast<int>(siluGws[0]));
        ret |= siluNhwcKernel->get().setArg(idx++, static_cast<int>(siluGws[1]));
        ret |= siluNhwcKernel->get().setArg(idx++, static_cast<int>(siluGws[2]));
        ret |= siluNhwcKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(sharedGate));
        ret |= siluNhwcKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(sharedUp));
        ret |= siluNhwcKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(actNhwc));
        ret |= siluNhwcKernel->get().setArg(idx++, c.rows);
        ret |= siluNhwcKernel->get().setArg(idx++, c.inter);
        ret |= siluNhwcKernel->get().setArg(idx++, actChannelAlign);
        if (!checkCL(ret, "setArg shared silu_c4_to_nhwc")) {
            return false;
        }
    }
    {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= downKernel->get().setArg(idx++, static_cast<int>(downGws[0]));
        ret |= downKernel->get().setArg(idx++, static_cast<int>(downGws[1]));
        ret |= downKernel->get().setArg(idx++, static_cast<int>(downGws[2]));
        ret |= downKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(actNhwc));
        ret |= downKernel->get().setArg(idx++, *downResource->mKernelImage.get());
        ret |= downKernel->get().setArg(idx++, *downResource->mDequantScaleOffsetBuffer.get());
        ret |= downKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(downResource->mBias.get()));
        ret |= downKernel->get().setArg(idx++, MNN::OpenCL::openCLBuffer(sharedNhwcOut));
        ret |= downKernel->get().setArg(idx++, downOutputChannelAlign8);
        ret |= downKernel->get().setArg(idx++, actChannelAlign);
        ret |= downKernel->get().setArg(idx++, downOutputChannelBlocks);
        ret |= downKernel->get().setArg(idx++, downInputChannelBlocks);
        ret |= downKernel->get().setArg(idx++, c.inter);
        ret |= downKernel->get().setArg(idx++, static_cast<int>(downResource->mBlockSize));
        ret |= downKernel->get().setArg(idx++, downBlockDim);
        ret |= downKernel->get().setArg(idx++, downResource->mCoef);
        if (!checkCL(ret, "setArg shared direct down projection")) {
            return false;
        }
    }

    auto runPre = [&]() -> ErrorCode {
        std::vector<uint32_t> roundUp = {ROUND_UP(preGws[0], preLws[0]), ROUND_UP(preGws[1], preLws[1])};
        cl_int ret = runtime->commandQueue().enqueueNDRangeKernel(preKernel->get(), cl::NullRange,
                                                                  cl::NDRange(roundUp[0], roundUp[1]),
                                                                  cl::NDRange(preLws[0], preLws[1]));
        return checkCL(ret, "enqueue shared gateup preconvert") ? NO_ERROR : COMPUTE_SIZE_ERROR;
    };
    auto runProjection = [&](const std::shared_ptr<KernelWrap>& kernel) -> ErrorCode {
        return runRawKernel(runtime, kernel, projGws, projLws) ? NO_ERROR : COMPUTE_SIZE_ERROR;
    };
    auto runSiluNhwc = [&]() -> ErrorCode {
        return runRawKernel(runtime, siluNhwcKernel, siluGws, siluLws) ? NO_ERROR : COMPUTE_SIZE_ERROR;
    };
    auto runDownProjection = [&]() -> ErrorCode {
        return runRawKernel(runtime, downKernel, downGws, downLws) ? NO_ERROR : COMPUTE_SIZE_ERROR;
    };
    auto runSharedGateUp = [&]() -> ErrorCode {
        auto localCode = runPre();
        if (localCode != NO_ERROR) {
            return localCode;
        }
        localCode = runProjection(gateKernel);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        return runProjection(upKernel);
    };
    auto runSplitChain = [&]() -> ErrorCode {
        auto localCode = bench.execute(gateExe.get(), hiddenInput, gateOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        localCode = bench.execute(upExe.get(), hiddenInput, upOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        localCode = bench.execute(siluExe.get(), siluInputs, splitActOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        return bench.execute(splitDownExe.get(), splitDownInput, splitDownOutput);
    };
    auto runSharedChain = [&]() -> ErrorCode {
        auto localCode = runSharedGateUp();
        if (localCode != NO_ERROR) {
            return localCode;
        }
        localCode = bench.execute(sharedSiluExe.get(), sharedSiluInputs, sharedActOutput);
        if (localCode != NO_ERROR) {
            return localCode;
        }
        return bench.execute(sharedDownExe.get(), sharedDownInput, sharedDownOutput);
    };
    auto runSharedNhwcChain = [&]() -> ErrorCode {
        auto localCode = runSharedGateUp();
        if (localCode != NO_ERROR) {
            return localCode;
        }
        localCode = runSiluNhwc();
        if (localCode != NO_ERROR) {
            return localCode;
        }
        return runDownProjection();
    };

    float splitChainMs = 0.0f;
    float sharedGateUpMs = 0.0f;
    float sharedChainMs = 0.0f;
    float sharedSiluNhwcMs = 0.0f;
    float sharedDirectDownMs = 0.0f;
    float sharedNhwcChainMs = 0.0f;
    OpenCLWallTimer timer;
    if (!timer.measure(runSplitChain, [&]() { return bench.sync(); }, c.warmup, c.repeat, &splitChainMs)) {
        return false;
    }
    if (!timer.measure(runSharedGateUp, [&]() { return bench.sync(); }, c.warmup, c.repeat, &sharedGateUpMs)) {
        return false;
    }
    if (!timer.measure(runSharedChain, [&]() { return bench.sync(); }, c.warmup, c.repeat, &sharedChainMs)) {
        return false;
    }
    if (runSharedGateUp() != NO_ERROR || runSiluNhwc() != NO_ERROR || !bench.sync()) {
        return false;
    }
    if (!timer.measure(runSiluNhwc, [&]() { return bench.sync(); }, c.warmup, c.repeat, &sharedSiluNhwcMs)) {
        return false;
    }
    if (!timer.measure(runDownProjection, [&]() { return bench.sync(); }, c.warmup, c.repeat, &sharedDirectDownMs)) {
        return false;
    }
    if (!timer.measure(runSharedNhwcChain, [&]() { return bench.sync(); }, c.warmup, c.repeat, &sharedNhwcChainMs)) {
        return false;
    }
    if (runSplitChain() != NO_ERROR || runSharedChain() != NO_ERROR ||
        runSharedNhwcChain() != NO_ERROR || !bench.sync()) {
        return false;
    }
    std::vector<float> splitData;
    std::vector<float> sharedData;
    std::vector<float> sharedNhwcData;
    if (!bench.readTensorTyped<float>(splitOut, &splitData) ||
        !bench.readTensorTyped<float>(sharedOut, &sharedData) ||
        !bench.readTensorTyped<float>(sharedNhwcOut, &sharedNhwcData)) {
        return false;
    }
    float maxAbs = 0.0f;
    float maxRel = 0.0f;
    int badCount = 0;
    float nhwcMaxAbs = 0.0f;
    float nhwcMaxRel = 0.0f;
    int nhwcBadCount = 0;
    const float absTol = envInt("MNN_BENCH_OPENCL_MLP_SHARED_INPUT_ABS_TOL_MILLI", 50) * 0.001f;
    const float relTol = envInt("MNN_BENCH_OPENCL_MLP_SHARED_INPUT_REL_TOL_MILLI", 80) * 0.001f;
    if (!compareRawMlpOutput(splitData, sharedData, absTol, relTol, &maxAbs, &maxRel, &badCount)) {
        return false;
    }
    if (!compareRawMlpOutput(splitData, sharedNhwcData, absTol, relTol,
                             &nhwcMaxAbs, &nhwcMaxRel, &nhwcBadCount)) {
        return false;
    }
    const int gateFamily = readSelectedCompactDenseFamily(bench, c.rows, c.hidden, c.inter, 4);
    const int downFamily = readSelectedCompactDenseFamily(bench, c.rows, c.inter, c.hidden, 4);
    MNN_PRINT("[bench_ops/opencl/perf/DecodeRepairMlpSharedInputGateUp] %-24s rows=%d hidden=%d inter=%d qblock=%d "
              "gate_family=%-22s down_family=%-22s split_chain=%.4f shared_gateup=%.4f "
              "shared_chain=%.4f delta=%.4f ms shared_silu_nhwc=%.4f shared_direct_down=%.4f "
              "shared_nhwc_chain=%.4f nhwc_delta=%.4f ms max_abs=%.6f max_rel=%.6f bad=%d/%zu "
              "nhwc_max_abs=%.6f nhwc_max_rel=%.6f nhwc_bad=%d/%zu\n",
              c.name, c.rows, c.hidden, c.inter, c.quantBlock,
              gateFamily >= 0 ? weightOnlyCompactDenseFamilyName(static_cast<uint32_t>(gateFamily)) : "-",
              downFamily >= 0 ? weightOnlyCompactDenseFamilyName(static_cast<uint32_t>(downFamily)) : "-",
              splitChainMs, sharedGateUpMs, sharedChainMs, sharedChainMs - splitChainMs,
              sharedSiluNhwcMs, sharedDirectDownMs, sharedNhwcChainMs, sharedNhwcChainMs - splitChainMs,
              maxAbs, maxRel, badCount, sharedData.size(),
              nhwcMaxAbs, nhwcMaxRel, nhwcBadCount, sharedNhwcData.size());
    ::fflush(stdout);
    return badCount == 0 && nhwcBadCount == 0;
}

static std::vector<DecodeRepairMlpChainCase> decodeRepairMlpChainCases() {
    const int warmup = envInt("MNN_BENCH_OPENCL_MLP_CHAIN_WARMUP", 20);
    const int repeat = envInt("MNN_BENCH_OPENCL_MLP_CHAIN_REPEAT", 80);
    const int hidden = envInt("MNN_BENCH_OPENCL_MLP_CHAIN_HIDDEN", 1536);
    const int inter = envInt("MNN_BENCH_OPENCL_MLP_CHAIN_INTER", 4608);
    const int quantBlock = envInt("MNN_BENCH_OPENCL_MLP_CHAIN_QBLOCK", 64);
    return {
        {"minicpm_decode_mlp", 2, hidden, inter, quantBlock, warmup, repeat},
        {"minicpm_decode_mlp", 4, hidden, inter, quantBlock, warmup, repeat},
        {"minicpm_decode_mlp", 6, hidden, inter, quantBlock, warmup, repeat},
        {"minicpm_decode_mlp", 8, hidden, inter, quantBlock, warmup, repeat},
    };
}

class OpenCLDecodeRepairMlpChainPerf : public MNNTestCase {
public:
    bool run(int precision) override {
        (void)precision;
        if (!requireMemoryLowOpenCL("bench_ops/opencl/perf/DecodeRepairMlpChain")) {
            return false;
        }
        bool ok = true;
        for (const auto& c : decodeRepairMlpChainCases()) {
            if (!enabledByFilter("MNN_BENCH_OPENCL_MLP_CHAIN_CASE", c.name) ||
                !enabledRow("MNN_BENCH_OPENCL_MLP_CHAIN_ROWS", c.rows)) {
                continue;
            }
            ok = runDecodeRepairMlpChainCase(c) && ok;
        }
        return ok;
    }
};

MNNTestSuiteRegister(OpenCLDecodeRepairMlpChainPerf, "bench_ops/opencl/perf/DecodeRepairMlpChain");

static std::vector<DecodeRepairMlpChainCase> decodeRepairMlpFusedFloorCases() {
    const int warmup = envInt("MNN_BENCH_OPENCL_MLP_FUSED_WARMUP",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_WARMUP", 20));
    const int repeat = envInt("MNN_BENCH_OPENCL_MLP_FUSED_REPEAT",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_REPEAT", 80));
    const int hidden = envInt("MNN_BENCH_OPENCL_MLP_FUSED_HIDDEN",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_HIDDEN", 1536));
    const int inter = envInt("MNN_BENCH_OPENCL_MLP_FUSED_INTER",
                             envInt("MNN_BENCH_OPENCL_MLP_CHAIN_INTER", 4608));
    const int quantBlock = envInt("MNN_BENCH_OPENCL_MLP_FUSED_QBLOCK",
                                  envInt("MNN_BENCH_OPENCL_MLP_CHAIN_QBLOCK", 64));
    return {
        {"minicpm_decode_mlp", 2, hidden, inter, quantBlock, warmup, repeat},
        {"minicpm_decode_mlp", 4, hidden, inter, quantBlock, warmup, repeat},
        {"minicpm_decode_mlp", 6, hidden, inter, quantBlock, warmup, repeat},
        {"minicpm_decode_mlp", 8, hidden, inter, quantBlock, warmup, repeat},
    };
}

class OpenCLDecodeRepairMlpFusedFloorPerf : public MNNTestCase {
public:
    bool run(int precision) override {
        (void)precision;
        if (!requireMemoryLowOpenCL("bench_ops/opencl/perf/DecodeRepairMlpFusedFloor")) {
            return false;
        }
        bool ok = true;
        for (const auto& c : decodeRepairMlpFusedFloorCases()) {
            if (!enabledByFilter("MNN_BENCH_OPENCL_MLP_FUSED_CASE", c.name) ||
                !enabledRow("MNN_BENCH_OPENCL_MLP_FUSED_ROWS", c.rows)) {
                continue;
            }
            ok = runDecodeRepairMlpFusedFloorCase(c) && ok;
        }
        return ok;
    }
};

MNNTestSuiteRegister(OpenCLDecodeRepairMlpFusedFloorPerf, "bench_ops/opencl/perf/DecodeRepairMlpFusedFloor");

static std::vector<DecodeRepairMlpChainCase> decodeRepairMlpResourceGateUpCases() {
    const int warmup = envInt("MNN_BENCH_OPENCL_MLP_RESOURCE_WARMUP",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_WARMUP", 20));
    const int repeat = envInt("MNN_BENCH_OPENCL_MLP_RESOURCE_REPEAT",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_REPEAT", 80));
    const int hidden = envInt("MNN_BENCH_OPENCL_MLP_RESOURCE_HIDDEN",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_HIDDEN", 1536));
    const int inter = envInt("MNN_BENCH_OPENCL_MLP_RESOURCE_INTER",
                             envInt("MNN_BENCH_OPENCL_MLP_CHAIN_INTER", 4608));
    const int quantBlock = envInt("MNN_BENCH_OPENCL_MLP_RESOURCE_QBLOCK",
                                  envInt("MNN_BENCH_OPENCL_MLP_CHAIN_QBLOCK", 64));
    return {
        {"minicpm_decode_mlp", 2, hidden, inter, quantBlock, warmup, repeat},
        {"minicpm_decode_mlp", 4, hidden, inter, quantBlock, warmup, repeat},
        {"minicpm_decode_mlp", 6, hidden, inter, quantBlock, warmup, repeat},
        {"minicpm_decode_mlp", 8, hidden, inter, quantBlock, warmup, repeat},
    };
}

class OpenCLDecodeRepairMlpResourceGateUpPerf : public MNNTestCase {
public:
    bool run(int precision) override {
        (void)precision;
        if (!requireMemoryLowOpenCL("bench_ops/opencl/perf/DecodeRepairMlpResourceGateUp")) {
            return false;
        }
        bool ok = true;
        for (const auto& c : decodeRepairMlpResourceGateUpCases()) {
            if (!enabledByFilter("MNN_BENCH_OPENCL_MLP_RESOURCE_CASE", c.name) ||
                !enabledRow("MNN_BENCH_OPENCL_MLP_RESOURCE_ROWS", c.rows)) {
                continue;
            }
            ok = runDecodeRepairMlpResourceGateUpCase(c) && ok;
        }
        return ok;
    }
};

MNNTestSuiteRegister(OpenCLDecodeRepairMlpResourceGateUpPerf,
                     "bench_ops/opencl/perf/DecodeRepairMlpResourceGateUp");

static std::vector<DecodeRepairMlpChainCase> decodeRepairMlpSiluDownC4Cases() {
    const int warmup = envInt("MNN_BENCH_OPENCL_MLP_SILU_DOWN_C4_WARMUP",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_WARMUP", 20));
    const int repeat = envInt("MNN_BENCH_OPENCL_MLP_SILU_DOWN_C4_REPEAT",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_REPEAT", 80));
    const int hidden = envInt("MNN_BENCH_OPENCL_MLP_SILU_DOWN_C4_HIDDEN",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_HIDDEN", 1536));
    const int inter = envInt("MNN_BENCH_OPENCL_MLP_SILU_DOWN_C4_INTER",
                             envInt("MNN_BENCH_OPENCL_MLP_CHAIN_INTER", 4608));
    const int quantBlock = envInt("MNN_BENCH_OPENCL_MLP_SILU_DOWN_C4_QBLOCK",
                                  envInt("MNN_BENCH_OPENCL_MLP_CHAIN_QBLOCK", 64));
    return {
        {"minicpm_decode_mlp", 2, hidden, inter, quantBlock, warmup, repeat},
        {"minicpm_decode_mlp", 4, hidden, inter, quantBlock, warmup, repeat},
        {"minicpm_decode_mlp", 6, hidden, inter, quantBlock, warmup, repeat},
        {"minicpm_decode_mlp", 8, hidden, inter, quantBlock, warmup, repeat},
    };
}

class OpenCLDecodeRepairMlpSiluDownC4Perf : public MNNTestCase {
public:
    bool run(int precision) override {
        (void)precision;
        if (!requireMemoryLowOpenCL("bench_ops/opencl/perf/DecodeRepairMlpSiluDownC4")) {
            return false;
        }
        bool ok = true;
        for (const auto& c : decodeRepairMlpSiluDownC4Cases()) {
            if (!enabledByFilter("MNN_BENCH_OPENCL_MLP_SILU_DOWN_C4_CASE", c.name) ||
                !enabledRow("MNN_BENCH_OPENCL_MLP_SILU_DOWN_C4_ROWS", c.rows)) {
                continue;
            }
            ok = runDecodeRepairMlpSiluDownC4Case(c) && ok;
        }
        return ok;
    }
};

MNNTestSuiteRegister(OpenCLDecodeRepairMlpSiluDownC4Perf,
                     "bench_ops/opencl/perf/DecodeRepairMlpSiluDownC4");

static std::vector<DecodeRepairMlpChainCase> decodeRepairMlpGateUpSiluDownC4Cases() {
    const int warmup = envInt("MNN_BENCH_OPENCL_MLP_GATEUP_SILU_DOWN_WARMUP",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_WARMUP", 20));
    const int repeat = envInt("MNN_BENCH_OPENCL_MLP_GATEUP_SILU_DOWN_REPEAT",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_REPEAT", 80));
    const int hidden = envInt("MNN_BENCH_OPENCL_MLP_GATEUP_SILU_DOWN_HIDDEN",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_HIDDEN", 1536));
    const int inter = envInt("MNN_BENCH_OPENCL_MLP_GATEUP_SILU_DOWN_INTER",
                             envInt("MNN_BENCH_OPENCL_MLP_CHAIN_INTER", 4608));
    const int quantBlock = envInt("MNN_BENCH_OPENCL_MLP_GATEUP_SILU_DOWN_QBLOCK",
                                  envInt("MNN_BENCH_OPENCL_MLP_CHAIN_QBLOCK", 64));
    return {
        {"minicpm_decode_mlp", 2, hidden, inter, quantBlock, warmup, repeat},
        {"minicpm_decode_mlp", 4, hidden, inter, quantBlock, warmup, repeat},
        {"minicpm_decode_mlp", 6, hidden, inter, quantBlock, warmup, repeat},
        {"minicpm_decode_mlp", 8, hidden, inter, quantBlock, warmup, repeat},
    };
}

class OpenCLDecodeRepairMlpGateUpSiluDownC4Perf : public MNNTestCase {
public:
    bool run(int precision) override {
        (void)precision;
        if (!requireMemoryLowOpenCL("bench_ops/opencl/perf/DecodeRepairMlpGateUpSiluDownC4")) {
            return false;
        }
        bool ok = true;
        for (const auto& c : decodeRepairMlpGateUpSiluDownC4Cases()) {
            if (!enabledByFilter("MNN_BENCH_OPENCL_MLP_GATEUP_SILU_DOWN_CASE", c.name) ||
                !enabledRow("MNN_BENCH_OPENCL_MLP_GATEUP_SILU_DOWN_ROWS", c.rows)) {
                continue;
            }
            ok = runDecodeRepairMlpGateUpSiluDownC4Case(c) && ok;
        }
        return ok;
    }
};

MNNTestSuiteRegister(OpenCLDecodeRepairMlpGateUpSiluDownC4Perf,
                     "bench_ops/opencl/perf/DecodeRepairMlpGateUpSiluDownC4");

static std::vector<DecodeRepairMlpChainCase> decodeRepairMlpStreamedTileCases() {
    const int warmup = envInt("MNN_BENCH_OPENCL_MLP_STREAMED_TILE_WARMUP",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_WARMUP", 20));
    const int repeat = envInt("MNN_BENCH_OPENCL_MLP_STREAMED_TILE_REPEAT",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_REPEAT", 80));
    const int hidden = envInt("MNN_BENCH_OPENCL_MLP_STREAMED_TILE_HIDDEN",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_HIDDEN", 1536));
    const int inter = envInt("MNN_BENCH_OPENCL_MLP_STREAMED_TILE_INTER",
                             envInt("MNN_BENCH_OPENCL_MLP_CHAIN_INTER", 4608));
    const int quantBlock = envInt("MNN_BENCH_OPENCL_MLP_STREAMED_TILE_QBLOCK",
                                  envInt("MNN_BENCH_OPENCL_MLP_CHAIN_QBLOCK", 64));
    return {
        {"minicpm_decode_mlp", 2, hidden, inter, quantBlock, warmup, repeat},
        {"minicpm_decode_mlp", 4, hidden, inter, quantBlock, warmup, repeat},
        {"minicpm_decode_mlp", 6, hidden, inter, quantBlock, warmup, repeat},
        {"minicpm_decode_mlp", 8, hidden, inter, quantBlock, warmup, repeat},
    };
}

class OpenCLDecodeRepairMlpStreamedTilePerf : public MNNTestCase {
public:
    bool run(int precision) override {
        (void)precision;
        if (!requireMemoryLowOpenCL("bench_ops/opencl/perf/DecodeRepairMlpStreamedTile")) {
            return false;
        }
        bool ok = true;
        for (const auto& c : decodeRepairMlpStreamedTileCases()) {
            if (!enabledByFilter("MNN_BENCH_OPENCL_MLP_STREAMED_TILE_CASE", c.name) ||
                !enabledRow("MNN_BENCH_OPENCL_MLP_STREAMED_TILE_ROWS", c.rows)) {
                continue;
            }
            ok = runDecodeRepairMlpStreamedTileCase(c) && ok;
        }
        return ok;
    }
};

MNNTestSuiteRegister(OpenCLDecodeRepairMlpStreamedTilePerf,
                     "bench_ops/opencl/perf/DecodeRepairMlpStreamedTile");

static std::vector<DecodeRepairMlpChainCase> decodeRepairMlpSiluNhwcDownCases() {
    const int warmup = envInt("MNN_BENCH_OPENCL_MLP_SILU_NHWC_DOWN_WARMUP",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_WARMUP", 20));
    const int repeat = envInt("MNN_BENCH_OPENCL_MLP_SILU_NHWC_DOWN_REPEAT",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_REPEAT", 80));
    const int hidden = envInt("MNN_BENCH_OPENCL_MLP_SILU_NHWC_DOWN_HIDDEN",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_HIDDEN", 1536));
    const int inter = envInt("MNN_BENCH_OPENCL_MLP_SILU_NHWC_DOWN_INTER",
                             envInt("MNN_BENCH_OPENCL_MLP_CHAIN_INTER", 4608));
    const int quantBlock = envInt("MNN_BENCH_OPENCL_MLP_SILU_NHWC_DOWN_QBLOCK",
                                  envInt("MNN_BENCH_OPENCL_MLP_CHAIN_QBLOCK", 64));
    std::vector<DecodeRepairMlpChainCase> cases;
    for (int rows : rowsFromEnvOrDefault("MNN_BENCH_OPENCL_MLP_SILU_NHWC_DOWN_ROWS", {2, 4, 6, 8})) {
        cases.push_back({"minicpm_decode_mlp", rows, hidden, inter, quantBlock, warmup, repeat});
    }
    return cases;
}

class OpenCLDecodeRepairMlpSiluNhwcDownPerf : public MNNTestCase {
public:
    bool run(int precision) override {
        (void)precision;
        if (!requireMemoryLowOpenCL("bench_ops/opencl/perf/DecodeRepairMlpSiluNhwcDown")) {
            return false;
        }
        bool ok = true;
        for (const auto& c : decodeRepairMlpSiluNhwcDownCases()) {
            if (!enabledByFilter("MNN_BENCH_OPENCL_MLP_SILU_NHWC_DOWN_CASE", c.name) ||
                !enabledRow("MNN_BENCH_OPENCL_MLP_SILU_NHWC_DOWN_ROWS", c.rows)) {
                continue;
            }
            ok = runDecodeRepairMlpSiluNhwcDownCase(c) && ok;
        }
        return ok;
    }
};

MNNTestSuiteRegister(OpenCLDecodeRepairMlpSiluNhwcDownPerf,
                     "bench_ops/opencl/perf/DecodeRepairMlpSiluNhwcDown");

static std::vector<DecodeRepairMlpChainCase> decodeRepairMlpSharedInputGateUpCases() {
    const int warmup = envInt("MNN_BENCH_OPENCL_MLP_SHARED_INPUT_WARMUP",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_WARMUP", 20));
    const int repeat = envInt("MNN_BENCH_OPENCL_MLP_SHARED_INPUT_REPEAT",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_REPEAT", 80));
    const int hidden = envInt("MNN_BENCH_OPENCL_MLP_SHARED_INPUT_HIDDEN",
                              envInt("MNN_BENCH_OPENCL_MLP_CHAIN_HIDDEN", 1536));
    const int inter = envInt("MNN_BENCH_OPENCL_MLP_SHARED_INPUT_INTER",
                             envInt("MNN_BENCH_OPENCL_MLP_CHAIN_INTER", 4608));
    const int quantBlock = envInt("MNN_BENCH_OPENCL_MLP_SHARED_INPUT_QBLOCK",
                                  envInt("MNN_BENCH_OPENCL_MLP_CHAIN_QBLOCK", 64));
    std::vector<DecodeRepairMlpChainCase> cases;
    for (int rows : rowsFromEnvOrDefault("MNN_BENCH_OPENCL_MLP_SHARED_INPUT_ROWS", {2, 4, 6, 8})) {
        cases.push_back({"minicpm_decode_mlp", rows, hidden, inter, quantBlock, warmup, repeat});
    }
    return cases;
}

class OpenCLDecodeRepairMlpSharedInputGateUpPerf : public MNNTestCase {
public:
    bool run(int precision) override {
        (void)precision;
        if (!requireMemoryLowOpenCL("bench_ops/opencl/perf/DecodeRepairMlpSharedInputGateUp")) {
            return false;
        }
        bool ok = true;
        for (const auto& c : decodeRepairMlpSharedInputGateUpCases()) {
            if (!enabledByFilter("MNN_BENCH_OPENCL_MLP_SHARED_INPUT_CASE", c.name) ||
                !enabledRow("MNN_BENCH_OPENCL_MLP_SHARED_INPUT_ROWS", c.rows)) {
                continue;
            }
            ok = runDecodeRepairMlpSharedInputGateUpCase(c) && ok;
        }
        return ok;
    }
};

MNNTestSuiteRegister(OpenCLDecodeRepairMlpSharedInputGateUpPerf,
                     "bench_ops/opencl/perf/DecodeRepairMlpSharedInputGateUp");

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
