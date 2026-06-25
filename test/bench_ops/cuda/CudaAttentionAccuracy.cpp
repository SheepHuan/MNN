//
//  CudaAttentionAccuracy.cpp
//  MNNTests
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "CudaOpBenchUtils.hpp"

using namespace MNN;
using namespace MNN::BenchOpsCuda;

namespace {

struct LinearOutputs {
    std::vector<float> prefill;
    std::vector<float> decode;
};

struct PagedOutputs {
    std::vector<float> prefill;
    std::vector<float> decode;
};

struct AttentionOutputs {
    std::vector<float> prefill;
    std::vector<float> decode;
};

struct PagedDecodeCase {
    const char* name;
    int qHeads;
    int kvHeads;
    int headDim;
    int context;
};

struct PagedAttentionSpecialCompareCase {
    const char* name;
    int qHeads;
    int kvHeads;
    int headDim;
    int context;
    int requestCapacity;
};

static bool runLinearSequence(MNNForwardType type, LinearOutputs* outputs) {
    if (outputs == nullptr) {
        return false;
    }
    constexpr int batch = 1;
    constexpr int kvHeads = 2;
    constexpr int qHeads = 2;
    constexpr int headDim = 64;
    constexpr int prefillLen = 8;
    constexpr int decodeLen = 1;
    constexpr int convKernel = 4;
    const int keyDim = kvHeads * headDim;
    const int valDim = qHeads * headDim;
    const int qkvDim = 2 * keyDim + valDim;

    DirectOpBench bench(type);
    if (!bench.valid()) {
        return false;
    }
    auto op = makeLinearAttentionOp(kvHeads, qHeads, headDim);

    auto convWData = makePattern(qkvDim * convKernel, 0.003f);
    auto qkvData = makePattern(batch * qkvDim * prefillLen, 0.01f);
    auto gateData = makeGate(batch * prefillLen * qHeads);
    auto betaData = makeBeta(batch * prefillLen * qHeads);

    auto qkv = bench.tensor({batch, qkvDim, prefillLen});
    auto gate = bench.tensor({batch, prefillLen, qHeads});
    auto beta = bench.tensor({batch, prefillLen, qHeads});
    auto convW = bench.tensor({qkvDim, 1, convKernel});
    auto out = bench.tensor({batch, prefillLen, valDim});
    if (!qkv || !gate || !beta || !convW || !out || !bench.writeTensor(qkv, qkvData) ||
        !bench.writeTensor(gate, gateData) || !bench.writeTensor(beta, betaData) ||
        !bench.writeTensor(convW, convWData)) {
        return false;
    }

    std::vector<Tensor*> inputs = {qkv, gate, beta, convW};
    std::vector<Tensor*> outTensors = {out};
    auto exe = bench.create(inputs, outTensors, op->get());
    if (!exe) {
        MNN_ERROR("failed to create LinearAttention accuracy execution\n");
        return false;
    }
    auto code = bench.resize(exe.get(), inputs, outTensors);
    if (code != NO_ERROR) {
        MNN_ERROR("LinearAttention accuracy prefill onResize failed: %d\n", code);
        return false;
    }
    code = bench.execute(exe.get(), inputs, outTensors);
    if (code != NO_ERROR) {
        MNN_ERROR("LinearAttention accuracy prefill onExecute failed: %d\n", code);
        return false;
    }
    if (!bench.readTensor(out, &outputs->prefill)) {
        return false;
    }

    auto decodeQkvData = makePattern(batch * qkvDim * decodeLen, 0.012f, 0.001f);
    auto decodeGateData = makeGate(batch * decodeLen * qHeads);
    auto decodeBetaData = makeBeta(batch * decodeLen * qHeads);
    auto decodeQkv = bench.tensor({batch, qkvDim, decodeLen});
    auto decodeGate = bench.tensor({batch, decodeLen, qHeads});
    auto decodeBeta = bench.tensor({batch, decodeLen, qHeads});
    auto decodeOut = bench.tensor({batch, decodeLen, valDim});
    if (!decodeQkv || !decodeGate || !decodeBeta || !decodeOut || !bench.writeTensor(decodeQkv, decodeQkvData) ||
        !bench.writeTensor(decodeGate, decodeGateData) || !bench.writeTensor(decodeBeta, decodeBetaData)) {
        return false;
    }

    std::vector<Tensor*> decodeInputs = {decodeQkv, decodeGate, decodeBeta, convW};
    std::vector<Tensor*> decodeOutputs = {decodeOut};
    code = bench.resize(exe.get(), decodeInputs, decodeOutputs);
    if (code != NO_ERROR) {
        MNN_ERROR("LinearAttention accuracy decode onResize failed: %d\n", code);
        return false;
    }
    code = bench.execute(exe.get(), decodeInputs, decodeOutputs);
    if (code != NO_ERROR) {
        MNN_ERROR("LinearAttention accuracy decode onExecute failed: %d\n", code);
        return false;
    }
    return bench.readTensor(decodeOut, &outputs->decode);
}

static bool runPagedSequence(MNNForwardType type, PagedOutputs* outputs) {
    if (outputs == nullptr) {
        return false;
    }
    constexpr int batch = 1;
    constexpr int qHeads = 4;
    constexpr int kvHeads = 2;
    constexpr int headDim = 16;
    constexpr int prefillLen = 8;
    constexpr int decodeLen = 1;
    constexpr int capacity = prefillLen + decodeLen + 4;

    PagedKVMeta meta;
    meta.beginRequest(capacity);
    DirectOpBench bench(type, &meta);
    if (!bench.valid()) {
        return false;
    }
    auto op = makeAttentionOp(OpType_PagedAttention, true);

    auto qData = makePattern(batch * prefillLen * qHeads * headDim, 0.01f);
    auto kData = makePattern(batch * prefillLen * kvHeads * headDim, 0.011f);
    auto vData = makePattern(batch * prefillLen * kvHeads * headDim, 0.009f);
    auto q = bench.tensor({batch, prefillLen, qHeads, headDim});
    auto k = bench.tensor({batch, prefillLen, kvHeads, headDim});
    auto v = bench.tensor({batch, prefillLen, kvHeads, headDim});
    auto out = bench.tensor({batch, prefillLen, qHeads, headDim});
    if (!q || !k || !v || !out || !bench.writeTensor(q, qData) || !bench.writeTensor(k, kData) ||
        !bench.writeTensor(v, vData)) {
        return false;
    }
    auto mask = bench.tensor({prefillLen, prefillLen});
    auto maskData = makeCausalMask(prefillLen, prefillLen);
    if (!mask || !bench.writeTensor(mask, maskData)) {
        return false;
    }
    std::vector<Tensor*> inputs = {q, k, v, mask};
    std::vector<Tensor*> outTensors = {out};
    auto exe = bench.create(inputs, outTensors, op->get());
    if (!exe) {
        MNN_ERROR("failed to create PagedAttention accuracy execution\n");
        return false;
    }
    auto code = bench.resize(exe.get(), inputs, outTensors);
    if (code != NO_ERROR) {
        MNN_ERROR("PagedAttention accuracy prefill onResize failed: %d\n", code);
        return false;
    }
    setPagedMeta(meta, 0, prefillLen);
    code = bench.execute(exe.get(), inputs, outTensors);
    if (code != NO_ERROR) {
        MNN_ERROR("PagedAttention accuracy prefill onExecute failed: %d\n", code);
        return false;
    }
    if (!bench.readTensor(out, &outputs->prefill)) {
        return false;
    }

    auto decodeQData = makePattern(batch * decodeLen * qHeads * headDim, 0.012f, 0.001f);
    auto decodeKData = makePattern(batch * decodeLen * kvHeads * headDim, 0.010f, -0.002f);
    auto decodeVData = makePattern(batch * decodeLen * kvHeads * headDim, 0.008f, 0.003f);
    auto decodeQ = bench.tensor({batch, decodeLen, qHeads, headDim});
    auto decodeK = bench.tensor({batch, decodeLen, kvHeads, headDim});
    auto decodeV = bench.tensor({batch, decodeLen, kvHeads, headDim});
    auto decodeOut = bench.tensor({batch, decodeLen, qHeads, headDim});
    if (!decodeQ || !decodeK || !decodeV || !decodeOut || !bench.writeTensor(decodeQ, decodeQData) ||
        !bench.writeTensor(decodeK, decodeKData) || !bench.writeTensor(decodeV, decodeVData)) {
        return false;
    }
    std::vector<Tensor*> decodeInputs = {decodeQ, decodeK, decodeV};
    std::vector<Tensor*> decodeOutputs = {decodeOut};
    code = bench.resize(exe.get(), decodeInputs, decodeOutputs);
    if (code != NO_ERROR) {
        MNN_ERROR("PagedAttention accuracy decode onResize failed: %d\n", code);
        return false;
    }
    setPagedMeta(meta, prefillLen, decodeLen);
    code = bench.execute(exe.get(), decodeInputs, decodeOutputs);
    if (code != NO_ERROR) {
        MNN_ERROR("PagedAttention accuracy decode onExecute failed: %d\n", code);
        return false;
    }
    return bench.readTensor(decodeOut, &outputs->decode);
}

static bool runAttentionSequenceConfig(MNNForwardType type, const PagedDecodeCase& c, AttentionOutputs* outputs) {
    if (outputs == nullptr) {
        return false;
    }
    constexpr int batch = 1;
    constexpr int decodeLen = 1;

    KVMeta meta;
    DirectOpBench bench(type, &meta);
    if (!bench.valid()) {
        return false;
    }
    auto op = makeAttentionOp(OpType_Attention, true);

    auto qData = makePattern(batch * c.context * c.qHeads * c.headDim, 0.01f);
    auto kData = makePattern(batch * c.context * c.kvHeads * c.headDim, 0.011f);
    auto vData = makePattern(batch * c.context * c.kvHeads * c.headDim, 0.009f);
    auto q = bench.tensor({batch, c.context, c.qHeads, c.headDim});
    auto k = bench.tensor({batch, c.context, c.kvHeads, c.headDim});
    auto v = bench.tensor({batch, c.context, c.kvHeads, c.headDim});
    auto out = bench.tensor({batch, c.context, c.qHeads, c.headDim});
    if (!q || !k || !v || !out || !bench.writeTensor(q, qData) || !bench.writeTensor(k, kData) ||
        !bench.writeTensor(v, vData)) {
        return false;
    }
    auto mask = bench.tensor({c.context, c.context});
    auto maskData = makeCausalMask(c.context, c.context);
    if (!mask || !bench.writeTensor(mask, maskData)) {
        return false;
    }
    std::vector<Tensor*> inputs = {q, k, v, mask};
    std::vector<Tensor*> outTensors = {out};
    auto exe = bench.create(inputs, outTensors, op->get());
    if (!exe) {
        MNN_ERROR("failed to create Attention accuracy execution: %s\n", c.name);
        return false;
    }
    auto code = bench.resize(exe.get(), inputs, outTensors);
    if (code != NO_ERROR) {
        MNN_ERROR("Attention accuracy prefill onResize failed: %s code=%d\n", c.name, code);
        return false;
    }
    setKVMeta(meta, 0, c.context);
    code = bench.execute(exe.get(), inputs, outTensors);
    if (code != NO_ERROR) {
        MNN_ERROR("Attention accuracy prefill onExecute failed: %s code=%d\n", c.name, code);
        return false;
    }
    if (!bench.readTensor(out, &outputs->prefill)) {
        return false;
    }

    auto decodeQData = makePattern(batch * decodeLen * c.qHeads * c.headDim, 0.012f, 0.001f);
    auto decodeKData = makePattern(batch * decodeLen * c.kvHeads * c.headDim, 0.010f, -0.002f);
    auto decodeVData = makePattern(batch * decodeLen * c.kvHeads * c.headDim, 0.008f, 0.003f);
    auto decodeQ = bench.tensor({batch, decodeLen, c.qHeads, c.headDim});
    auto decodeK = bench.tensor({batch, decodeLen, c.kvHeads, c.headDim});
    auto decodeV = bench.tensor({batch, decodeLen, c.kvHeads, c.headDim});
    auto decodeOut = bench.tensor({batch, decodeLen, c.qHeads, c.headDim});
    if (!decodeQ || !decodeK || !decodeV || !decodeOut || !bench.writeTensor(decodeQ, decodeQData) ||
        !bench.writeTensor(decodeK, decodeKData) || !bench.writeTensor(decodeV, decodeVData)) {
        return false;
    }
    std::vector<Tensor*> decodeInputs = {decodeQ, decodeK, decodeV};
    std::vector<Tensor*> decodeOutputs = {decodeOut};
    code = bench.resize(exe.get(), decodeInputs, decodeOutputs);
    if (code != NO_ERROR) {
        MNN_ERROR("Attention accuracy decode onResize failed: %s code=%d\n", c.name, code);
        return false;
    }
    setKVMeta(meta, c.context, decodeLen);
    code = bench.execute(exe.get(), decodeInputs, decodeOutputs);
    if (code != NO_ERROR) {
        MNN_ERROR("Attention accuracy decode onExecute failed: %s code=%d\n", c.name, code);
        return false;
    }
    return bench.readTensor(decodeOut, &outputs->decode);
}

static bool runPagedSequenceConfig(MNNForwardType type, const PagedDecodeCase& c, AttentionOutputs* outputs) {
    if (outputs == nullptr) {
        return false;
    }
    constexpr int batch = 1;
    constexpr int decodeLen = 1;
    const int capacity = c.context + decodeLen + 4;

    PagedKVMeta meta;
    meta.beginRequest(capacity);
    DirectOpBench bench(type, &meta);
    if (!bench.valid()) {
        return false;
    }
    auto op = makeAttentionOp(OpType_PagedAttention, true);

    auto qData = makePattern(batch * c.context * c.qHeads * c.headDim, 0.01f);
    auto kData = makePattern(batch * c.context * c.kvHeads * c.headDim, 0.011f);
    auto vData = makePattern(batch * c.context * c.kvHeads * c.headDim, 0.009f);
    auto q = bench.tensor({batch, c.context, c.qHeads, c.headDim});
    auto k = bench.tensor({batch, c.context, c.kvHeads, c.headDim});
    auto v = bench.tensor({batch, c.context, c.kvHeads, c.headDim});
    auto out = bench.tensor({batch, c.context, c.qHeads, c.headDim});
    if (!q || !k || !v || !out || !bench.writeTensor(q, qData) || !bench.writeTensor(k, kData) ||
        !bench.writeTensor(v, vData)) {
        return false;
    }
    auto mask = bench.tensor({c.context, c.context});
    auto maskData = makeCausalMask(c.context, c.context);
    if (!mask || !bench.writeTensor(mask, maskData)) {
        return false;
    }
    std::vector<Tensor*> inputs = {q, k, v, mask};
    std::vector<Tensor*> outTensors = {out};
    auto exe = bench.create(inputs, outTensors, op->get());
    if (!exe) {
        MNN_ERROR("failed to create PagedAttention compare execution: %s\n", c.name);
        return false;
    }
    auto code = bench.resize(exe.get(), inputs, outTensors);
    if (code != NO_ERROR) {
        MNN_ERROR("PagedAttention compare prefill onResize failed: %s code=%d\n", c.name, code);
        return false;
    }
    setPagedMeta(meta, 0, c.context);
    code = bench.execute(exe.get(), inputs, outTensors);
    if (code != NO_ERROR) {
        MNN_ERROR("PagedAttention compare prefill onExecute failed: %s code=%d\n", c.name, code);
        return false;
    }
    if (!bench.readTensor(out, &outputs->prefill)) {
        return false;
    }

    auto decodeQData = makePattern(batch * decodeLen * c.qHeads * c.headDim, 0.012f, 0.001f);
    auto decodeKData = makePattern(batch * decodeLen * c.kvHeads * c.headDim, 0.010f, -0.002f);
    auto decodeVData = makePattern(batch * decodeLen * c.kvHeads * c.headDim, 0.008f, 0.003f);
    auto decodeQ = bench.tensor({batch, decodeLen, c.qHeads, c.headDim});
    auto decodeK = bench.tensor({batch, decodeLen, c.kvHeads, c.headDim});
    auto decodeV = bench.tensor({batch, decodeLen, c.kvHeads, c.headDim});
    auto decodeOut = bench.tensor({batch, decodeLen, c.qHeads, c.headDim});
    if (!decodeQ || !decodeK || !decodeV || !decodeOut || !bench.writeTensor(decodeQ, decodeQData) ||
        !bench.writeTensor(decodeK, decodeKData) || !bench.writeTensor(decodeV, decodeVData)) {
        return false;
    }
    std::vector<Tensor*> decodeInputs = {decodeQ, decodeK, decodeV};
    std::vector<Tensor*> decodeOutputs = {decodeOut};
    code = bench.resize(exe.get(), decodeInputs, decodeOutputs);
    if (code != NO_ERROR) {
        MNN_ERROR("PagedAttention compare decode onResize failed: %s code=%d\n", c.name, code);
        return false;
    }
    setPagedMeta(meta, c.context, decodeLen);
    code = bench.execute(exe.get(), decodeInputs, decodeOutputs);
    if (code != NO_ERROR) {
        MNN_ERROR("PagedAttention compare decode onExecute failed: %s code=%d\n", c.name, code);
        return false;
    }
    return bench.readTensor(decodeOut, &outputs->decode);
}

static bool runPagedSequenceConfigWithCapacity(MNNForwardType type, const PagedDecodeCase& c, int requestCapacity,
                                               AttentionOutputs* outputs) {
    if (outputs == nullptr) {
        return false;
    }
    constexpr int batch = 1;
    constexpr int decodeLen = 1;
    const int capacity = requestCapacity > 0 ? requestCapacity : (c.context + decodeLen + 4);

    PagedKVMeta meta;
    meta.beginRequest(capacity);
    DirectOpBench bench(type, &meta);
    if (!bench.valid()) {
        return false;
    }
    auto op = makeAttentionOp(OpType_PagedAttention, true);

    auto qData = makePattern(batch * c.context * c.qHeads * c.headDim, 0.01f);
    auto kData = makePattern(batch * c.context * c.kvHeads * c.headDim, 0.011f);
    auto vData = makePattern(batch * c.context * c.kvHeads * c.headDim, 0.009f);
    auto q = bench.tensor({batch, c.context, c.qHeads, c.headDim});
    auto k = bench.tensor({batch, c.context, c.kvHeads, c.headDim});
    auto v = bench.tensor({batch, c.context, c.kvHeads, c.headDim});
    auto out = bench.tensor({batch, c.context, c.qHeads, c.headDim});
    if (!q || !k || !v || !out || !bench.writeTensor(q, qData) || !bench.writeTensor(k, kData) ||
        !bench.writeTensor(v, vData)) {
        return false;
    }
    auto mask = bench.tensor({c.context, c.context});
    auto maskData = makeCausalMask(c.context, c.context);
    if (!mask || !bench.writeTensor(mask, maskData)) {
        return false;
    }
    std::vector<Tensor*> inputs = {q, k, v, mask};
    std::vector<Tensor*> outTensors = {out};
    auto exe = bench.create(inputs, outTensors, op->get());
    if (!exe) {
        MNN_ERROR("failed to create PagedAttention special compare execution: %s\n", c.name);
        return false;
    }
    auto code = bench.resize(exe.get(), inputs, outTensors);
    if (code != NO_ERROR) {
        MNN_ERROR("PagedAttention special compare prefill onResize failed: %s code=%d\n", c.name, code);
        return false;
    }
    setPagedMeta(meta, 0, c.context);
    code = bench.execute(exe.get(), inputs, outTensors);
    if (code != NO_ERROR) {
        MNN_ERROR("PagedAttention special compare prefill onExecute failed: %s code=%d\n", c.name, code);
        return false;
    }
    if (!bench.readTensor(out, &outputs->prefill)) {
        return false;
    }

    auto decodeQData = makePattern(batch * decodeLen * c.qHeads * c.headDim, 0.012f, 0.001f);
    auto decodeKData = makePattern(batch * decodeLen * c.kvHeads * c.headDim, 0.010f, -0.002f);
    auto decodeVData = makePattern(batch * decodeLen * c.kvHeads * c.headDim, 0.008f, 0.003f);
    auto decodeQ = bench.tensor({batch, decodeLen, c.qHeads, c.headDim});
    auto decodeK = bench.tensor({batch, decodeLen, c.kvHeads, c.headDim});
    auto decodeV = bench.tensor({batch, decodeLen, c.kvHeads, c.headDim});
    auto decodeOut = bench.tensor({batch, decodeLen, c.qHeads, c.headDim});
    if (!decodeQ || !decodeK || !decodeV || !decodeOut || !bench.writeTensor(decodeQ, decodeQData) ||
        !bench.writeTensor(decodeK, decodeKData) || !bench.writeTensor(decodeV, decodeVData)) {
        return false;
    }
    std::vector<Tensor*> decodeInputs = {decodeQ, decodeK, decodeV};
    std::vector<Tensor*> decodeOutputs = {decodeOut};
    code = bench.resize(exe.get(), decodeInputs, decodeOutputs);
    if (code != NO_ERROR) {
        MNN_ERROR("PagedAttention special compare decode onResize failed: %s code=%d\n", c.name, code);
        return false;
    }
    setPagedMeta(meta, c.context, decodeLen);
    code = bench.execute(exe.get(), decodeInputs, decodeOutputs);
    if (code != NO_ERROR) {
        MNN_ERROR("PagedAttention special compare decode onExecute failed: %s code=%d\n", c.name, code);
        return false;
    }
    return bench.readTensor(decodeOut, &outputs->decode);
}

static bool runPagedDecodeContext(MNNForwardType type, const PagedDecodeCase& c, std::vector<float>* output) {
    if (output == nullptr) {
        return false;
    }
    constexpr int batch = 1;
    constexpr int decodeLen = 1;
    const int capacity = c.context + decodeLen + 4;

    PagedKVMeta meta;
    meta.beginRequest(capacity);
    DirectOpBench bench(type, &meta);
    if (!bench.valid()) {
        return false;
    }
    auto op = makeAttentionOp(OpType_PagedAttention, true);

    auto qData = makePattern(batch * decodeLen * c.qHeads * c.headDim, 0.012f, 0.001f);
    auto kData = makePattern(batch * decodeLen * c.kvHeads * c.headDim, 0.010f, -0.002f);
    auto vData = makePattern(batch * decodeLen * c.kvHeads * c.headDim, 0.008f, 0.003f);
    auto q = bench.tensor({batch, decodeLen, c.qHeads, c.headDim});
    auto k = bench.tensor({batch, decodeLen, c.kvHeads, c.headDim});
    auto v = bench.tensor({batch, decodeLen, c.kvHeads, c.headDim});
    auto out = bench.tensor({batch, decodeLen, c.qHeads, c.headDim});
    if (!q || !k || !v || !out || !bench.writeTensor(q, qData) || !bench.writeTensor(k, kData) ||
        !bench.writeTensor(v, vData)) {
        return false;
    }

    std::vector<Tensor*> inputs = {q, k, v};
    std::vector<Tensor*> outputs = {out};
    auto exe = bench.create(inputs, outputs, op->get());
    if (!exe) {
        MNN_ERROR("failed to create PagedAttention context accuracy execution: %s\n", c.name);
        return false;
    }
    auto code = bench.resize(exe.get(), inputs, outputs);
    if (code != NO_ERROR) {
        MNN_ERROR("PagedAttention context accuracy onResize failed: %s code=%d\n", c.name, code);
        return false;
    }
    setPagedMeta(meta, c.context, decodeLen);
    code = bench.execute(exe.get(), inputs, outputs);
    if (code != NO_ERROR) {
        MNN_ERROR("PagedAttention context accuracy onExecute failed: %s code=%d\n", c.name, code);
        return false;
    }
    return bench.readTensor(out, output);
}

static std::vector<PagedDecodeCase> llamaPagedDecodeAccuracyCases() {
    return {
        {"PagedAttention/llama3.2-1B/ctx512/decode", 32, 8, 64, 512},
        {"PagedAttention/llama3.2-1B/ctx1024/decode", 32, 8, 64, 1024},
        {"PagedAttention/llama3.2-1B/ctx2048/decode", 32, 8, 64, 2048},
        {"PagedAttention/llama3.2-3B/ctx512/decode", 24, 8, 128, 512},
        {"PagedAttention/llama3.2-3B/ctx1024/decode", 24, 8, 128, 1024},
        {"PagedAttention/llama3.2-3B/ctx2048/decode", 24, 8, 128, 2048},
        {"PagedAttention/llama3.2-8B/ctx512/decode", 32, 8, 128, 512},
        {"PagedAttention/llama3.2-8B/ctx1024/decode", 32, 8, 128, 1024},
        {"PagedAttention/llama3.2-8B/ctx2048/decode", 32, 8, 128, 2048},
    };
}

static PagedDecodeCase minimalPagedAttentionCompareCase() {
    return {"PagedAttention/minimal/layer0/ctx8/decode", 4, 2, 16, 8};
}

static std::vector<PagedAttentionSpecialCompareCase> pagedAttentionSpecialCompareCases() {
    return {
        {"PagedAttention/minicpm5-1B/ctx1522/cap2048/decode", 16, 2, 128, 1522, 2048},
        {"PagedAttention/minicpm5-1B/ctx1522/cap4096/decode", 16, 2, 128, 1522, 4096},
        {"PagedAttention/minicpm5-1B/ctx1521/cap4096/decode", 16, 2, 128, 1521, 4096},
        {"PagedAttention/minicpm5-1B/ctx1523/cap4096/decode", 16, 2, 128, 1523, 4096},
    };
}

static std::string attentionCompareName(const char* backendName, const PagedDecodeCase& c, const char* stage) {
    std::string base = c.name;
    const std::string prefix = "PagedAttention/";
    if (base.find(prefix) == 0) {
        base = base.substr(prefix.size());
    }
    auto pos = base.rfind("/decode");
    if (pos != std::string::npos) {
        base.replace(pos, 7, std::string("/") + stage);
    } else {
        base += "/";
        base += stage;
    }

    std::string name = "PagedAttention_vs_Attention/";
    if (backendName != nullptr && backendName[0] != '\0') {
        name += backendName;
        name += "/";
    }
    name += base;
    return name;
}

static bool compareAttentionAndPagedOutputs(const char* backendName, const PagedDecodeCase& c,
                                            const AttentionOutputs& attention, const AttentionOutputs& paged) {
    auto prefillName = attentionCompareName(backendName, c, "prefill");
    auto decodeName = attentionCompareName(backendName, c, "decode");
    return compareVectors(prefillName.c_str(), paged.prefill, attention.prefill, 2.0e-3f, 2.0e-2f) &&
           compareVectors(decodeName.c_str(), paged.decode, attention.decode, 2.0e-3f, 2.0e-2f);
}

class CudaLinearAttentionAccuracy : public MNNTestCase {
public:
    bool run(int precision) override {
        if (!supportAccuracyPrecision()) {
            return true;
        }
        LinearOutputs cpu;
        LinearOutputs cuda;
        if (!runLinearSequence(MNN_FORWARD_CPU, &cpu) || !runLinearSequence(MNN_FORWARD_CUDA, &cuda)) {
            return false;
        }
        return compareVectors("LinearAttention/prefill", cuda.prefill, cpu.prefill, 2.0e-3f, 2.0e-2f) &&
               compareVectors("LinearAttention/decode", cuda.decode, cpu.decode, 2.0e-3f, 2.0e-2f);
    }
};

class CudaPagedAttentionAccuracy : public MNNTestCase {
public:
    bool run(int precision) override {
        if (!supportAccuracyPrecision()) {
            return true;
        }
        PagedOutputs cpu;
        PagedOutputs cuda;
        if (!runPagedSequence(MNN_FORWARD_CPU, &cpu) || !runPagedSequence(MNN_FORWARD_CUDA, &cuda)) {
            return false;
        }
        if (!compareVectors("PagedAttention/prefill", cuda.prefill, cpu.prefill, 2.0e-3f, 2.0e-2f) ||
            !compareVectors("PagedAttention/decode", cuda.decode, cpu.decode, 2.0e-3f, 2.0e-2f)) {
            return false;
        }
        auto cases = llamaPagedDecodeAccuracyCases();
        for (auto& c : cases) {
            std::vector<float> cpuContext;
            std::vector<float> cudaContext;
            if (!runPagedDecodeContext(MNN_FORWARD_CPU, c, &cpuContext) ||
                !runPagedDecodeContext(MNN_FORWARD_CUDA, c, &cudaContext) ||
                !compareVectors(c.name, cudaContext, cpuContext, 2.0e-3f, 2.0e-2f)) {
                return false;
            }
        }
        return true;
    }
};

class CudaPagedAttentionMinimalCompareAttentionAccuracy : public MNNTestCase {
public:
    bool run(int precision) override {
        if (!supportAccuracyPrecision()) {
            return true;
        }
        auto c = minimalPagedAttentionCompareCase();
        AttentionOutputs cudaAttention;
        AttentionOutputs cudaPaged;
        if (!runAttentionSequenceConfig(MNN_FORWARD_CUDA, c, &cudaAttention) ||
            !runPagedSequenceConfig(MNN_FORWARD_CUDA, c, &cudaPaged) ||
            !compareAttentionAndPagedOutputs("CUDA", c, cudaAttention, cudaPaged)) {
            return false;
        }
        return true;
    }
};

class CudaPagedAttentionCompareAttentionAccuracy : public MNNTestCase {
public:
    bool run(int precision) override {
        if (!supportAccuracyPrecision()) {
            return true;
        }
        auto cases = llamaPagedDecodeAccuracyCases();
        for (auto& c : cases) {
            AttentionOutputs attention;
            AttentionOutputs paged;
            if (!runAttentionSequenceConfig(MNN_FORWARD_CUDA, c, &attention) ||
                !runPagedSequenceConfig(MNN_FORWARD_CUDA, c, &paged)) {
                return false;
            }
            if (!compareAttentionAndPagedOutputs("", c, attention, paged)) {
                return false;
            }
        }
        return true;
    }
};

class CudaPagedAttentionSpecialCompareAttentionAccuracy : public MNNTestCase {
public:
    bool run(int precision) override {
        if (!supportAccuracyPrecision()) {
            return true;
        }
        auto cases = pagedAttentionSpecialCompareCases();
        for (auto& special : cases) {
            PagedDecodeCase base{special.name, special.qHeads, special.kvHeads, special.headDim, special.context};
            AttentionOutputs attention;
            AttentionOutputs paged;
            if (!runAttentionSequenceConfig(MNN_FORWARD_CUDA, base, &attention) ||
                !runPagedSequenceConfigWithCapacity(MNN_FORWARD_CUDA, base, special.requestCapacity, &paged)) {
                return false;
            }
            if (!compareAttentionAndPagedOutputs("special", base, attention, paged)) {
                return false;
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(CudaLinearAttentionAccuracy, "bench_ops/cuda/accuracy/LinearAttention");
MNNTestSuiteRegister(CudaPagedAttentionAccuracy, "bench_ops/cuda/accuracy/PagedAttention");
MNNTestSuiteRegister(CudaPagedAttentionMinimalCompareAttentionAccuracy,
                     "bench_ops/cuda/accuracy/PagedAttention/MinimalCompareAttention");
MNNTestSuiteRegister(CudaPagedAttentionCompareAttentionAccuracy,
                     "bench_ops/cuda/accuracy/PagedAttention/CompareAttention");
MNNTestSuiteRegister(CudaPagedAttentionSpecialCompareAttentionAccuracy,
                     "bench_ops/cuda/accuracy/PagedAttention/SpecialCompareAttention");

} // namespace

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
