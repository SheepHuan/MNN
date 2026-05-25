//
//  CudaAttentionPerf.cpp
//  MNNTests
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "CudaOpBenchUtils.hpp"
#include <cstdio>

using namespace MNN;
using namespace MNN::BenchOpsCuda;

namespace {

static bool printTimedResult(const char* op, const BenchCase& c, float avgMs) {
    float tokens = static_cast<float>(c.seqLen);
    float us = avgMs * 1000.0f;
    MNN_PRINT("[bench_ops/cuda/perf/%s] %-18s B=%d qH=%d kvH=%d D=%d past=%d add=%d avg=%.4f ms %.2f us/token\n",
              op, c.name, c.batch, c.qHeads, c.kvHeads, c.headDim, c.pastLen, c.seqLen, avgMs,
              tokens > 0.0f ? us / tokens : 0.0f);
    ::fflush(stdout);
    return true;
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

static bool runPagedAttentionCase(const BenchCase& c) {
    PagedKVMeta meta;
    const int capacity = c.pastLen + c.seqLen + 64;
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
    float avgMs = 0.0f;
    auto run = [&]() {
        setPagedMeta(meta, c.pastLen, c.seqLen);
        return bench.execute(exe.get(), inputs, outputs);
    };
    if (!timer.measure(run, c.warmup, c.repeat, &avgMs)) {
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

MNNTestSuiteRegister(CudaLinearAttentionPrefillPerf, "bench_ops/cuda/perf/LinearAttention/Prefill");
MNNTestSuiteRegister(CudaLinearAttentionDecodePerf, "bench_ops/cuda/perf/LinearAttention/Decode");
MNNTestSuiteRegister(CudaAttentionPrefillPerf, "bench_ops/cuda/perf/Attention/Prefill");
MNNTestSuiteRegister(CudaAttentionDecodePerf, "bench_ops/cuda/perf/Attention/Decode");
MNNTestSuiteRegister(CudaPagedAttentionPrefillPerf, "bench_ops/cuda/perf/PagedAttention/Prefill");
MNNTestSuiteRegister(CudaPagedAttentionDecodePerf, "bench_ops/cuda/perf/PagedAttention/Decode");

} // namespace

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
