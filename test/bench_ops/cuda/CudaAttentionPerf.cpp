//
//  CudaAttentionPerf.cpp
//  MNNTests
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "CudaOpBenchUtils.hpp"
#include <cstdio>
#include <cstdlib>
#include <cctype>

using namespace MNN;
using namespace MNN::BenchOpsCuda;

namespace {

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

static bool printTimedResult(const char* op, const BenchCase& c, float avgMs) {
    MNN_PRINT("[bench_ops/cuda/perf/%s] %-18s B=%d qH=%d kvH=%d D=%d past=%d add=%d avg=%.4f ms\n",
              op, c.name, c.batch, c.qHeads, c.kvHeads, c.headDim, c.pastLen, c.seqLen, avgMs);
    ::fflush(stdout);
    return true;
}

static bool printPagedAttentionV1V2TimedResult(const char* impl, const BenchCase& c, float avgMs) {
    const int kvLen = c.pastLen + c.seqLen;
    const float ratio = kvLen > 0 ? static_cast<float>(c.seqLen) / static_cast<float>(kvLen) : 0.0f;
    MNN_PRINT("[bench_ops/cuda/perf/PagedAttention/V1V2] %-28s impl=%-9s B=%d qH=%d kvH=%d D=%d "
              "past=%d add=%d kv=%d q_over_kv=%.4f avg=%.4f ms\n",
              c.name, impl, c.batch, c.qHeads, c.kvHeads, c.headDim, c.pastLen, c.seqLen, kvLen, ratio, avgMs);
    ::fflush(stdout);
    return true;
}

struct PagedAttentionSpecialShapeCase {
    BenchCase bench;
    int requestCapacity;
};

struct PagedAttentionPendingWriteCase {
    const char* name;
    int batch;
    int qHeads;
    int kvHeads;
    int headDim;
    int seqLen;
    int requestCapacity;
    int layers;
};

static bool printPagedAttentionSpecialTimedResult(const char* impl, const PagedAttentionSpecialShapeCase& c,
                                                  float avgMs) {
    const int kvLen = c.bench.pastLen + c.bench.seqLen;
    const float ratio = kvLen > 0 ? static_cast<float>(c.bench.seqLen) / static_cast<float>(kvLen) : 0.0f;
    MNN_PRINT("[bench_ops/cuda/perf/PagedAttention/SpecialShapes] %-28s impl=%-9s "
              "B=%d qH=%d kvH=%d D=%d past=%d add=%d kv=%d cap=%d q_over_kv=%.4f avg=%.4f ms\n",
              c.bench.name, impl, c.bench.batch, c.bench.qHeads, c.bench.kvHeads, c.bench.headDim,
              c.bench.pastLen, c.bench.seqLen, kvLen, c.requestCapacity, ratio, avgMs);
    ::fflush(stdout);
    return true;
}

static bool printPagedAttentionPendingWriteTimedResult(const char* impl, const PagedAttentionPendingWriteCase& c,
                                                       float avgMs) {
    MNN_PRINT("[bench_ops/cuda/perf/PagedAttention/PendingWriteSubgraph] %-28s impl=%-9s "
              "B=%d qH=%d kvH=%d D=%d add=%d cap=%d layers=%d avg=%.4f ms\n",
              c.name, impl, c.batch, c.qHeads, c.kvHeads, c.headDim, c.seqLen, c.requestCapacity, c.layers, avgMs);
    ::fflush(stdout);
    return true;
}

static std::string sanitizePrefixCacheName(const char* name) {
    std::string out = name != nullptr ? name : "paged_attention";
    for (char& ch : out) {
        if (!std::isalnum(static_cast<unsigned char>(ch))) {
            ch = '_';
        }
    }
    return out;
}

static std::unique_ptr<OpHolder> makePagedAttentionOpForLayer(int layerIndex) {
    OpT op;
    op.type = OpType_PagedAttention;
    op.main.type = OpParameter_AttentionParam;
    op.main.value = new AttentionParamT;
    auto* param = op.main.AsAttentionParam();
    param->kv_cache = true;
    param->layer_index = layerIndex;
    param->kv_shared_layer_index = -1;
    return std::unique_ptr<OpHolder>(new OpHolder(op));
}

static bool runLinearAttentionCase(const BenchCase& c) {
    DirectOpBench bench(MNN_FORWARD_CUDA);
    if (!bench.valid()) {
        return false;
    }
    const int convKernel = 4;
    const int keyDim = c.kvHeads * c.headDim;
    const int valDim = c.qHeads * c.headDim;
    const int qkvDim = 2 * keyDim + valDim;

    auto convW = bench.tensor({qkvDim, 1, convKernel});
    if (!convW) {
        return false;
    }
    auto convWData = makePattern(qkvDim * convKernel, 0.003f);
    if (!bench.writeTensor(convW, convWData)) {
        return false;
    }

    auto op = makeLinearAttentionOp(c.kvHeads, c.qHeads, c.headDim);
    std::unique_ptr<Execution> exe;
    if (c.pastLen > 0) {
        auto setupQkv = bench.tensor({c.batch, qkvDim, c.pastLen});
        auto setupGate = bench.tensor({c.batch, c.pastLen, c.qHeads});
        auto setupBeta = bench.tensor({c.batch, c.pastLen, c.qHeads});
        auto setupOut = bench.tensor({c.batch, c.pastLen, valDim});
        auto setupQkvData = makePattern(c.batch * qkvDim * c.pastLen, 0.01f);
        auto setupGateData = makeGate(c.batch * c.pastLen * c.qHeads);
        auto setupBetaData = makeBeta(c.batch * c.pastLen * c.qHeads);
        if (!setupQkv || !setupGate || !setupBeta || !setupOut ||
            !bench.writeTensor(setupQkv, setupQkvData) || !bench.writeTensor(setupGate, setupGateData) ||
            !bench.writeTensor(setupBeta, setupBetaData)) {
            return false;
        }
        std::vector<Tensor*> setupInputs = {setupQkv, setupGate, setupBeta, convW};
        std::vector<Tensor*> setupOutputs = {setupOut};
        exe = bench.create(setupInputs, setupOutputs, op->get());
        if (!exe) {
            MNN_ERROR("failed to create LinearAttention execution\n");
            return false;
        }
        auto code = bench.resize(exe.get(), setupInputs, setupOutputs);
        if (code != NO_ERROR) {
            MNN_ERROR("LinearAttention setup onResize failed: %d\n", code);
            return false;
        }
        code = bench.execute(exe.get(), setupInputs, setupOutputs);
        if (code != NO_ERROR) {
            MNN_ERROR("LinearAttention setup onExecute failed: %d\n", code);
            return false;
        }
        if (!checkCuda(cudaDeviceSynchronize(), "LinearAttention setup sync")) {
            return false;
        }
    }

    auto qkv = bench.tensor({c.batch, qkvDim, c.seqLen});
    auto gate = bench.tensor({c.batch, c.seqLen, c.qHeads});
    auto beta = bench.tensor({c.batch, c.seqLen, c.qHeads});
    auto out = bench.tensor({c.batch, c.seqLen, valDim});
    auto qkvData = makePattern(c.batch * qkvDim * c.seqLen, 0.012f, c.pastLen > 0 ? 0.001f : 0.0f);
    auto gateData = makeGate(c.batch * c.seqLen * c.qHeads);
    auto betaData = makeBeta(c.batch * c.seqLen * c.qHeads);
    if (!qkv || !gate || !beta || !out || !bench.writeTensor(qkv, qkvData) ||
        !bench.writeTensor(gate, gateData) || !bench.writeTensor(beta, betaData)) {
        return false;
    }

    std::vector<Tensor*> inputs = {qkv, gate, beta, convW};
    std::vector<Tensor*> outputs = {out};
    if (!exe) {
        exe = bench.create(inputs, outputs, op->get());
        if (!exe) {
            MNN_ERROR("failed to create LinearAttention execution\n");
            return false;
        }
    }
    auto code = bench.resize(exe.get(), inputs, outputs);
    if (code != NO_ERROR) {
        MNN_ERROR("LinearAttention onResize failed: %d\n", code);
        return false;
    }

    CudaEventPair timer;
    float avgMs = 0.0f;
    if (!timer.measure([&]() { return bench.execute(exe.get(), inputs, outputs); }, c.warmup, c.repeat, &avgMs)) {
        return false;
    }
    return printTimedResult("LinearAttention", c, avgMs);
}

static bool runAttentionCase(const BenchCase& c) {
    KVMeta meta;
    DirectOpBench bench(MNN_FORWARD_CUDA, &meta);
    if (!bench.valid()) {
        return false;
    }

    auto op = makeAttentionOp(OpType_Attention, true);
    std::unique_ptr<Execution> exe;

    if (c.pastLen > 0) {
        auto q = bench.tensor({c.batch, c.pastLen, c.qHeads, c.headDim});
        auto k = bench.tensor({c.batch, c.pastLen, c.kvHeads, c.headDim});
        auto v = bench.tensor({c.batch, c.pastLen, c.kvHeads, c.headDim});
        auto o = bench.tensor({c.batch, c.pastLen, c.qHeads, c.headDim});
        if (!q || !k || !v || !o) {
            return false;
        }
        std::vector<Tensor*> setupInputs = {q, k, v};
        std::vector<Tensor*> setupOutputs = {o};
        exe = bench.create(setupInputs, setupOutputs, op->get());
        if (!exe) {
            MNN_ERROR("failed to create Attention execution\n");
            return false;
        }
        auto code = bench.resize(exe.get(), setupInputs, setupOutputs);
        if (code != NO_ERROR) {
            MNN_ERROR("Attention setup onResize failed: %d\n", code);
            return false;
        }
        setKVMeta(meta, 0, c.pastLen);
        code = bench.execute(exe.get(), setupInputs, setupOutputs);
        if (code != NO_ERROR) {
            MNN_ERROR("Attention setup onExecute failed: %d\n", code);
            return false;
        }
        if (!checkCuda(cudaDeviceSynchronize(), "Attention setup sync")) {
            return false;
        }
    }

    auto q = bench.tensor({c.batch, c.seqLen, c.qHeads, c.headDim});
    auto k = bench.tensor({c.batch, c.seqLen, c.kvHeads, c.headDim});
    auto v = bench.tensor({c.batch, c.seqLen, c.kvHeads, c.headDim});
    auto o = bench.tensor({c.batch, c.seqLen, c.qHeads, c.headDim});
    if (!q || !k || !v || !o) {
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
    if (!exe) {
        exe = bench.create(inputs, outputs, op->get());
        if (!exe) {
            MNN_ERROR("failed to create Attention execution\n");
            return false;
        }
    }
    auto code = bench.resize(exe.get(), inputs, outputs);
    if (code != NO_ERROR) {
        MNN_ERROR("Attention onResize failed: %d\n", code);
        return false;
    }

    CudaEventPair timer;
    float avgMs = 0.0f;
    bool firstRun = true;
    auto run = [&]() {
        int logicalBeforeRun = firstRun ? c.pastLen : (c.pastLen + c.seqLen);
        setKVMeta(meta, logicalBeforeRun, c.seqLen, firstRun ? 0 : c.seqLen);
        firstRun = false;
        return bench.execute(exe.get(), inputs, outputs);
    };
    if (!timer.measure(run, c.warmup, c.repeat, &avgMs)) {
        return false;
    }
    return printTimedResult("Attention", c, avgMs);
}

static bool measurePagedAttentionCaseWithCapacity(const BenchCase& c, int requestCapacity, bool syncEachRun,
                                                  float* avgMs) {
    if (avgMs == nullptr) {
        return false;
    }
    PagedKVMeta meta;
    const int capacity = requestCapacity > 0 ? requestCapacity : (c.pastLen + c.seqLen + 64);
    meta.beginRequest(capacity);
    DirectOpBench bench(MNN_FORWARD_CUDA, &meta);
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
        MNN_ERROR("failed to create PagedAttention execution\n");
        return false;
    }
    auto code = bench.resize(exe.get(), inputs, outputs);
    if (code != NO_ERROR) {
        MNN_ERROR("PagedAttention onResize failed: %d\n", code);
        return false;
    }

    CudaEventPair timer;
    auto run = [&]() {
        setPagedMeta(meta, c.pastLen, c.seqLen);
        auto code = bench.execute(exe.get(), inputs, outputs);
        if (code != NO_ERROR) {
            return code;
        }
        if (syncEachRun && !checkCuda(cudaDeviceSynchronize(), "PagedAttention special-shape sync")) {
            return INVALID_VALUE;
        }
        return NO_ERROR;
    };
    if (!timer.measure(run, c.warmup, c.repeat, avgMs)) {
        return false;
    }
    return true;
}

static bool measurePagedAttentionCase(const BenchCase& c, float* avgMs) {
    return measurePagedAttentionCaseWithCapacity(c, 0, false, avgMs);
}

static bool runPagedAttentionCase(const BenchCase& c) {
    float avgMs = 0.0f;
    if (!measurePagedAttentionCase(c, &avgMs)) {
        return false;
    }
    return printTimedResult("PagedAttention", c, avgMs);
}

static std::vector<BenchCase> llamaAttentionPrefillCases() {
    return {
        {"llama3.2-1B_prefill_ctx512", 1, 32, 8, 64, 512, 0, 0, 1},
        {"llama3.2-3B_prefill_ctx512", 1, 24, 8, 128, 512, 0, 0, 1},
        {"llama3.2-8B_prefill_ctx512", 1, 32, 8, 128, 512, 0, 0, 1},
        {"llama3.2-1B_prefill_ctx1024", 1, 32, 8, 64, 1024, 0, 0, 1},
        {"llama3.2-3B_prefill_ctx1024", 1, 24, 8, 128, 1024, 0, 0, 1},
        {"llama3.2-8B_prefill_ctx1024", 1, 32, 8, 128, 1024, 0, 0, 1},
        {"llama3.2-1B_prefill_ctx2048", 1, 32, 8, 64, 2048, 0, 0, 1},
        {"llama3.2-3B_prefill_ctx2048", 1, 24, 8, 128, 2048, 0, 0, 1},
        {"llama3.2-8B_prefill_ctx2048", 1, 32, 8, 128, 2048, 0, 0, 1},
    };
}

static std::vector<BenchCase> llamaAttentionDecodeCases() {
    return {
        {"llama3.2-1B_decode_ctx512", 1, 32, 8, 64, 1, 512, 5, 50},
        {"llama3.2-3B_decode_ctx512", 1, 24, 8, 128, 1, 512, 5, 50},
        {"llama3.2-8B_decode_ctx512", 1, 32, 8, 128, 1, 512, 5, 50},
        {"llama3.2-1B_decode_ctx1024", 1, 32, 8, 64, 1, 1024, 5, 50},
        {"llama3.2-3B_decode_ctx1024", 1, 24, 8, 128, 1, 1024, 5, 50},
        {"llama3.2-8B_decode_ctx1024", 1, 32, 8, 128, 1, 1024, 5, 50},
        {"llama3.2-1B_decode_ctx2048", 1, 32, 8, 64, 1, 2048, 5, 50},
        {"llama3.2-3B_decode_ctx2048", 1, 24, 8, 128, 1, 2048, 5, 50},
        {"llama3.2-8B_decode_ctx2048", 1, 32, 8, 128, 1, 2048, 5, 50},
    };
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

static std::vector<PagedAttentionSpecialShapeCase> pagedAttentionSpecialShapeCases() {
    return {
        {{"minicpm5-1B_q1522_cap1536", 1, 16, 2, 128, 1522, 0, 0, 1}, 1536},
        {{"minicpm5-1B_q1522_cap2048", 1, 16, 2, 128, 1522, 0, 0, 1}, 2048},
        {{"minicpm5-1B_q1522_cap4096", 1, 16, 2, 128, 1522, 0, 0, 1}, 4096},
        {{"minicpm5-1B_q1521_cap4096", 1, 16, 2, 128, 1521, 0, 0, 1}, 4096},
        {{"minicpm5-1B_q1523_cap4096", 1, 16, 2, 128, 1523, 0, 0, 1}, 4096},
    };
}

static std::vector<PagedAttentionPendingWriteCase> pagedAttentionPendingWriteCases() {
    return {
        {"minicpm5-1B_q1522_cap4096_l8", 1, 16, 2, 128, 1522, 4096, 8},
        {"minicpm5-1B_q1522_cap2048_l8", 1, 16, 2, 128, 1522, 2048, 8},
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

static bool runPagedAttentionSpecialShapeCase(const PagedAttentionSpecialShapeCase& c) {
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
        if (!measurePagedAttentionCaseWithCapacity(c.bench, c.requestCapacity, true, &avgMs)) {
            return false;
        }
        if (!printPagedAttentionSpecialTimedResult(impl.label, c, avgMs)) {
            return false;
        }
    }
    return true;
}

static bool runPagedAttentionPendingWriteCase(const PagedAttentionPendingWriteCase& c) {
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
        PagedKVMeta meta;
        meta.beginRequest(c.requestCapacity);
        meta.layer_nums = c.layers;
        meta.file_flag = KVMeta::PendingWrite;
        meta.file_name = sanitizePrefixCacheName(c.name) + "_" + impl.label;

        DirectOpBench bench(MNN_FORWARD_CUDA, &meta);
        if (!bench.valid()) {
            return false;
        }
        RuntimeHint hint;
        hint.prefixcacheDirPath = ".cache/bench_ops";
        bench.setRuntimeHint(hint);

        auto q = bench.tensor({c.batch, c.seqLen, c.qHeads, c.headDim});
        auto k = bench.tensor({c.batch, c.seqLen, c.kvHeads, c.headDim});
        auto v = bench.tensor({c.batch, c.seqLen, c.kvHeads, c.headDim});
        auto o = bench.tensor({c.batch, c.seqLen, c.qHeads, c.headDim});
        auto mask = bench.tensor({c.seqLen, c.seqLen});
        auto qData = makePattern(c.batch * c.seqLen * c.qHeads * c.headDim, 0.01f);
        auto kData = makePattern(c.batch * c.seqLen * c.kvHeads * c.headDim, 0.011f);
        auto vData = makePattern(c.batch * c.seqLen * c.kvHeads * c.headDim, 0.009f);
        auto maskData = makeCausalMask(c.seqLen, c.seqLen);
        if (!q || !k || !v || !o || !mask || !bench.writeTensor(q, qData) || !bench.writeTensor(k, kData) ||
            !bench.writeTensor(v, vData) || !bench.writeTensor(mask, maskData)) {
            return false;
        }
        std::vector<Tensor*> inputs = {q, k, v, mask};
        std::vector<Tensor*> outputs = {o};
        std::vector<std::unique_ptr<OpHolder>> ops;
        std::vector<std::unique_ptr<Execution>> exes;
        ops.reserve(c.layers);
        exes.reserve(c.layers);
        for (int layerIndex = 0; layerIndex < c.layers; ++layerIndex) {
            ops.emplace_back(makePagedAttentionOpForLayer(layerIndex));
            auto exe = bench.create(inputs, outputs, ops.back()->get());
            if (!exe) {
                MNN_ERROR("failed to create PagedAttention pending-write execution: %s layer=%d impl=%s\n",
                          c.name, layerIndex, impl.label);
                return false;
            }
            auto code = bench.resize(exe.get(), inputs, outputs);
            if (code != NO_ERROR) {
                MNN_ERROR("PagedAttention pending-write onResize failed: %s layer=%d impl=%s code=%d\n",
                          c.name, layerIndex, impl.label, code);
                return false;
            }
            exes.emplace_back(std::move(exe));
        }

        CudaEventPair timer;
        float avgMs = 0.0f;
        auto run = [&]() {
            setPagedMeta(meta, 0, c.seqLen);
            meta.file_flag = KVMeta::PendingWrite;
            for (int layerIndex = 0; layerIndex < c.layers; ++layerIndex) {
                auto code = bench.execute(exes[layerIndex].get(), inputs, outputs);
                if (code != NO_ERROR) {
                    MNN_ERROR("PagedAttention pending-write onExecute failed: %s layer=%d impl=%s code=%d\n",
                              c.name, layerIndex, impl.label, code);
                    return code;
                }
                if (!checkCuda(cudaDeviceSynchronize(), "PagedAttention pending-write layer sync")) {
                    MNN_ERROR("PagedAttention pending-write sync failed: %s layer=%d impl=%s\n",
                              c.name, layerIndex, impl.label);
                    return INVALID_VALUE;
                }
            }
            return NO_ERROR;
        };
        if (!timer.measure(run, 0, 1, &avgMs)) {
            return false;
        }
        if (!printPagedAttentionPendingWriteTimedResult(impl.label, c, avgMs)) {
            return false;
        }
    }
    return true;
}

class CudaLinearAttentionPrefillPerf : public MNNTestCase {
public:
    bool run(int precision) override {
        if (!supportPerfPrecision()) {
            return true;
        }
        auto cases = llamaAttentionPrefillCases();
        for (auto& c : cases) {
            if (!runLinearAttentionCase(c)) {
                return false;
            }
        }
        return true;
    }
};

class CudaLinearAttentionDecodePerf : public MNNTestCase {
public:
    bool run(int precision) override {
        if (!supportPerfPrecision()) {
            return true;
        }
        auto cases = llamaAttentionDecodeCases();
        for (auto& c : cases) {
            if (!runLinearAttentionCase(c)) {
                return false;
            }
        }
        return true;
    }
};

class CudaAttentionPrefillPerf : public MNNTestCase {
public:
    bool run(int precision) override {
        if (!supportPerfPrecision()) {
            return true;
        }
        auto cases = llamaAttentionPrefillCases();
        for (auto& c : cases) {
            if (!runAttentionCase(c)) {
                return false;
            }
        }
        return true;
    }
};

class CudaAttentionDecodePerf : public MNNTestCase {
public:
    bool run(int precision) override {
        if (!supportPerfPrecision()) {
            return true;
        }
        auto cases = llamaAttentionDecodeCases();
        for (auto& c : cases) {
            if (!runAttentionCase(c)) {
                return false;
            }
        }
        return true;
    }
};

class CudaPagedAttentionPrefillPerf : public MNNTestCase {
public:
    bool run(int precision) override {
        if (!supportPerfPrecision()) {
            return true;
        }
        auto cases = llamaAttentionPrefillCases();
        for (auto& c : cases) {
            if (!runPagedAttentionCase(c)) {
                return false;
            }
        }
        return true;
    }
};

class CudaPagedAttentionDecodePerf : public MNNTestCase {
public:
    bool run(int precision) override {
        if (!supportPerfPrecision()) {
            return true;
        }
        auto cases = llamaAttentionDecodeCases();
        for (auto& c : cases) {
            if (!runPagedAttentionCase(c)) {
                return false;
            }
        }
        return true;
    }
};

class CudaPagedAttentionV1V2Perf : public MNNTestCase {
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

class CudaPagedAttentionSpecialShapePerf : public MNNTestCase {
public:
    bool run(int precision) override {
        if (!supportPerfPrecision()) {
            return true;
        }
        auto cases = pagedAttentionSpecialShapeCases();
        for (auto& c : cases) {
            if (!runPagedAttentionSpecialShapeCase(c)) {
                return false;
            }
        }
        return true;
    }
};

class CudaPagedAttentionPendingWritePerf : public MNNTestCase {
public:
    bool run(int precision) override {
        if (!supportPerfPrecision()) {
            return true;
        }
        auto cases = pagedAttentionPendingWriteCases();
        for (auto& c : cases) {
            if (!runPagedAttentionPendingWriteCase(c)) {
                return false;
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(CudaLinearAttentionPrefillPerf, "bench_ops/cuda/perf/LinearAttention/Prefill");
MNNTestSuiteRegister(CudaLinearAttentionDecodePerf, "bench_ops/cuda/perf/LinearAttention/Decode");
MNNTestSuiteRegister(CudaAttentionPrefillPerf, "bench_ops/cuda/perf/Attention/Prefill");
MNNTestSuiteRegister(CudaAttentionDecodePerf, "bench_ops/cuda/perf/Attention/Decode");
MNNTestSuiteRegister(CudaPagedAttentionPrefillPerf, "bench_ops/cuda/perf/PagedAttention/Prefill");
MNNTestSuiteRegister(CudaPagedAttentionDecodePerf, "bench_ops/cuda/perf/PagedAttention/Decode");
MNNTestSuiteRegister(CudaPagedAttentionV1V2Perf, "bench_ops/cuda/perf/PagedAttention/V1V2");
MNNTestSuiteRegister(CudaPagedAttentionSpecialShapePerf, "bench_ops/cuda/perf/PagedAttention/SpecialShapes");
MNNTestSuiteRegister(CudaPagedAttentionPendingWritePerf, "bench_ops/cuda/perf/PagedAttention/PendingWriteSubgraph");

} // namespace

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
