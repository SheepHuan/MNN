//
//  CudaWeightOnlyConvPerf.cpp
//  MNNTests
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "CudaOpBenchUtils.hpp"
#include "core/IDSTEncoder.hpp"
#include "core/TensorUtils.hpp"

#include <cuda_fp16.h>
#include <cublas_v2.h>
#if __has_include(<cublasLt.h>)
#include <cublasLt.h>
#define MNN_BENCH_HAS_CUBLASLT 1
#else
#define MNN_BENCH_HAS_CUBLASLT 0
#endif
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <sstream>
#include <sys/stat.h>
#include <utility>
#include <vector>

using namespace MNN;
using namespace MNN::BenchOpsCuda;

namespace {

struct WeightOnlyConvCase {
    const char* name;
    int rows;
    int ic;
    int oc;
    int quantBlock;
    int warmup;
    int repeat;
};

struct GemmFloorCase {
    const char* name;
    int rows;
    int ic;
    int oc;
    int warmup;
    int repeat;
};

struct LinearConvertChainCase {
    const char* name;
    int rows;
    int ic;
    int oc;
    int quantBlock;
    int warmup;
    int repeat;
};

struct PicExternalPlan {
    std::vector<int64_t> external;
    int quantBit = 4;
    int quantBlock = 64;
    int aMin = 0;
    int readType = 0;
    bool shapeInt32 = false;
    bool hasBias = false;
};

class CudaDeviceBuffer {
public:
    CudaDeviceBuffer() = default;
    ~CudaDeviceBuffer() {
        if (mPtr != nullptr) {
            cudaFree(mPtr);
        }
    }

    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    bool alloc(size_t bytes) {
        if (!checkCuda(cudaMalloc(&mPtr, bytes), "cudaMalloc(CudaDeviceBuffer)")) {
            return false;
        }
        return checkCuda(cudaMemset(mPtr, 0, bytes), "cudaMemset(CudaDeviceBuffer)");
    }

    void* get() const {
        return mPtr;
    }

private:
    void* mPtr = nullptr;
};

class ScopedEnvVar {
public:
    explicit ScopedEnvVar(const char* key) : mKey(key != nullptr ? key : "") {
        const char* value = ::getenv(mKey.c_str());
        if (value != nullptr) {
            mHadOld = true;
            mOldValue = value;
        }
    }

    ~ScopedEnvVar() {
        if (mKey.empty()) {
            return;
        }
        if (mHadOld) {
            ::setenv(mKey.c_str(), mOldValue.c_str(), 1);
        } else {
            ::unsetenv(mKey.c_str());
        }
    }

    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
    std::string mKey;
    bool mHadOld = false;
    std::string mOldValue;
};

static int upDivInt(int x, int y) {
    return (x + y - 1) / y;
}

static std::unique_ptr<OpHolder> makeWeightOnlyLinearConvOp(int ic, int oc, int quantBlock) {
    OpT op;
    op.name = "bench_weight_only_linear";
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

    const int groups = upDivInt(ic, quantBlock);
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

static std::unique_ptr<OpHolder> makeExternalWeightOnlyLinearConvOp(const char* name, const std::string& externalPath,
                                                                    int ic, int oc, const PicExternalPlan& plan) {
    OpT op;
    op.name = name != nullptr ? name : "bench_external_weight_only_linear";
    op.type = OpType_Convolution;
    op.defaultDimentionFormat = MNN_DATA_FORMAT_NCHW;
    op.externalPath = externalPath;
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
    conv->common->hasOutputShape = false;
    conv->quanParameter.reset(new IDSTQuanT);
    if (plan.quantBit == 16) {
        conv->quanParameter->type = 3;
    } else {
        conv->quanParameter->quantScale = 1.0f;
        conv->quanParameter->scaleIn = 0.0f;
        conv->quanParameter->scaleOut = 0.0f;
        conv->quanParameter->useInt32 = false;
        conv->quanParameter->has_scaleInt = false;
        conv->quanParameter->shapeInt32 = plan.shapeInt32;
        conv->quanParameter->type = 1;
        conv->quanParameter->aMaxOrBits = plan.quantBit;
        conv->quanParameter->aMin = plan.aMin;
        conv->quanParameter->readType = plan.readType;
        conv->quanParameter->weightSize = 0;
    }
    conv->external = plan.external;
    conv->bias.resize(std::max(0, oc), 0.0f);
    return std::unique_ptr<OpHolder>(new OpHolder(op));
}

static std::unique_ptr<OpHolder> makePicSiluMulOp(const char* name) {
    OpT op;
    op.name = name != nullptr ? name : "bench_pic_silu_mul";
    op.type = OpType_Extra;
    op.defaultDimentionFormat = MNN_DATA_FORMAT_NCHW;
    op.main.type = OpParameter_Extra;
    op.main.value = new ExtraT;
    auto* extra = op.main.AsExtra();
    extra->type = "PicSiluMul";
    extra->engine = "MNN";
    auto attr = std::unique_ptr<AttributeT>(new AttributeT);
    attr->key = "name";
    attr->s = op.name;
    extra->attr.emplace_back(std::move(attr));
    return std::unique_ptr<OpHolder>(new OpHolder(op));
}

static std::unique_ptr<OpHolder> makePicPackedSiluMulOp(const char* name) {
    OpT op;
    op.name = name != nullptr ? name : "bench_pic_packed_silu_mul";
    op.type = OpType_Extra;
    op.defaultDimentionFormat = MNN_DATA_FORMAT_NCHW;
    op.main.type = OpParameter_Extra;
    op.main.value = new ExtraT;
    auto* extra = op.main.AsExtra();
    extra->type = "PicPackedSiluMul";
    extra->engine = "MNN";
    auto attr = std::unique_ptr<AttributeT>(new AttributeT);
    attr->key = "name";
    attr->s = op.name;
    extra->attr.emplace_back(std::move(attr));
    return std::unique_ptr<OpHolder>(new OpHolder(op));
}

static std::unique_ptr<OpHolder> makePicBenchPackedSiluDownOp(const char* type, const char* name) {
    // PicBench* Extra names are direct-op probes only; CUDA backend registers
    // them under MNN_CUDA_BENCH_OPS (MNN_BUILD_TEST=ON), not in production builds.
    OpT op;
    op.name = name != nullptr ? name : "bench_pic_fused_packed_silu_down";
    op.type = OpType_Extra;
    op.defaultDimentionFormat = MNN_DATA_FORMAT_NCHW;
    op.main.type = OpParameter_Extra;
    op.main.value = new ExtraT;
    auto* extra = op.main.AsExtra();
    extra->type = type != nullptr ? type : "PicBenchFusedPackedSiluDown";
    extra->engine = "MNN";
    auto attr = std::unique_ptr<AttributeT>(new AttributeT);
    attr->key = "name";
    attr->s = op.name;
    extra->attr.emplace_back(std::move(attr));
    return std::unique_ptr<OpHolder>(new OpHolder(op));
}

static std::string externalToString(const std::vector<int64_t>& external) {
    std::ostringstream os;
    for (size_t i = 0; i < external.size(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << external[i];
    }
    return os.str();
}

static void addStringAttr(ExtraT* extra, const std::string& key, const std::string& value) {
    auto attr = std::unique_ptr<AttributeT>(new AttributeT);
    attr->key = key;
    attr->s = value;
    extra->attr.emplace_back(std::move(attr));
}

static void addIntAttr(ExtraT* extra, const std::string& key, int value) {
    auto attr = std::unique_ptr<AttributeT>(new AttributeT);
    attr->key = key;
    attr->i = value;
    extra->attr.emplace_back(std::move(attr));
}

static void addPicGateUpConvAttrs(ExtraT* extra, const char* prefix, const char* name,
                                  int ic, int oc, const PicExternalPlan& plan) {
    const std::string p(prefix != nullptr ? prefix : "");
    addStringAttr(extra, p + "_name", name != nullptr ? name : p);
    addStringAttr(extra, p + "_external", externalToString(plan.external));
    addIntAttr(extra, p + "_in_features", ic);
    addIntAttr(extra, p + "_out_features", oc);
    addIntAttr(extra, p + "_quant_bit", plan.quantBit);
    addIntAttr(extra, p + "_quant_block", plan.quantBlock);
    addIntAttr(extra, p + "_a_min", plan.aMin);
    addIntAttr(extra, p + "_read_type", plan.readType);
    addIntAttr(extra, p + "_shape_int32", plan.shapeInt32 ? 1 : 0);
    addIntAttr(extra, p + "_has_bias", plan.hasBias ? 1 : 0);
}

static std::unique_ptr<OpHolder> makePicGateUpWeightOnlyExtraOp(const char* type, const char* name,
                                                                const std::string& externalPath,
                                                                int ic, int oc,
                                                                const PicExternalPlan& gatePlan,
                                                                const PicExternalPlan& upPlan) {
    OpT op;
    op.name = name != nullptr ? name : type;
    op.type = OpType_Extra;
    op.defaultDimentionFormat = MNN_DATA_FORMAT_NHWC;
    op.externalPath = externalPath;
    op.main.type = OpParameter_Extra;
    op.main.value = new ExtraT;
    auto* extra = op.main.AsExtra();
    extra->type = type != nullptr ? type : "PicGateUpWeightOnly";
    extra->engine = "MNN";
    addStringAttr(extra, "name", op.name);
    addIntAttr(extra, "in_features", ic);
    addIntAttr(extra, "out_features", oc);
    addPicGateUpConvAttrs(extra, "gate", "bench_gate", ic, oc, gatePlan);
    addPicGateUpConvAttrs(extra, "up", "bench_up", ic, oc, upPlan);
    return std::unique_ptr<OpHolder>(new OpHolder(op));
}

static std::unique_ptr<OpHolder> makePicLinearNhwcWeightOnlyExtraOp(const char* name,
                                                                    const std::string& externalPath,
                                                                    int ic, int oc,
                                                                    const PicExternalPlan& plan) {
    OpT op;
    op.name = name != nullptr ? name : "bench_pic_linear_nhwc_weight_only";
    op.type = OpType_Extra;
    op.defaultDimentionFormat = MNN_DATA_FORMAT_NHWC;
    op.externalPath = externalPath;
    op.main.type = OpParameter_Extra;
    op.main.value = new ExtraT;
    auto* extra = op.main.AsExtra();
    extra->type = "PicLinearNhwcWeightOnly";
    extra->engine = "MNN";
    addStringAttr(extra, "name", op.name);
    addStringAttr(extra, "linear_name", op.name);
    addStringAttr(extra, "external", externalToString(plan.external));
    addIntAttr(extra, "in_features", ic);
    addIntAttr(extra, "out_features", oc);
    addIntAttr(extra, "quant_bit", plan.quantBit);
    addIntAttr(extra, "quant_block", plan.quantBlock);
    addIntAttr(extra, "a_min", plan.aMin);
    addIntAttr(extra, "read_type", plan.readType);
    addIntAttr(extra, "shape_int32", plan.shapeInt32 ? 1 : 0);
    addIntAttr(extra, "has_bias", plan.hasBias ? 1 : 0);
    return std::unique_ptr<OpHolder>(new OpHolder(op));
}

static std::unique_ptr<OpHolder> makeRasterOp(const char* name) {
    OpT op;
    op.name = name != nullptr ? name : "bench_raster";
    op.type = OpType_Raster;
    op.defaultDimentionFormat = MNN_DATA_FORMAT_NCHW;
    op.main.type = OpParameter_NONE;
    op.main.value = nullptr;
    return std::unique_ptr<OpHolder>(new OpHolder(op));
}

static bool weightOnlyConvUsesPicRepairMeta() {
    const char* value = ::getenv("MNN_BENCH_WEIGHT_ONLY_PIC_REPAIR");
    if (value == nullptr || value[0] == '\0') {
        return true;
    }
    return std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "False") != 0 &&
           std::strcmp(value, "off") != 0 &&
           std::strcmp(value, "OFF") != 0 &&
           std::strcmp(value, "no") != 0 &&
           std::strcmp(value, "NO") != 0;
}

static bool runWeightOnlyConvCase(const WeightOnlyConvCase& c) {
    KVMeta meta;
    // Test-only switch:
    //   default / 1: preserve the existing decode-repair small-M tune path;
    //   0: measure normal WeightOnlyConv routing without PIC decode repair meta.
    meta.pic_decode_repair_sparse_active = weightOnlyConvUsesPicRepairMeta();
    DirectOpBench bench(MNN_FORWARD_CUDA, &meta);
    if (!bench.valid()) {
        return false;
    }

    auto input = bench.tensor({c.rows, c.ic, 1, 1}, Tensor::CAFFE, false);
    auto output = bench.tensor({c.rows, c.oc, 1, 1}, Tensor::CAFFE, false);
    if (!input || !output) {
        return false;
    }

    auto op = makeWeightOnlyLinearConvOp(c.ic, c.oc, c.quantBlock);
    std::vector<Tensor*> inputs = {input};
    std::vector<Tensor*> outputs = {output};
    auto exe = bench.create(inputs, outputs, op->get());
    if (!exe) {
        MNN_ERROR("failed to create WeightOnlyConv execution for %s rows=%d ic=%d oc=%d\n",
                  c.name, c.rows, c.ic, c.oc);
        return false;
    }
    auto code = bench.resize(exe.get(), inputs, outputs);
    if (code != NO_ERROR) {
        MNN_ERROR("WeightOnlyConv onResize failed for %s rows=%d ic=%d oc=%d: %d\n",
                  c.name, c.rows, c.ic, c.oc, code);
        return false;
    }

    CudaEventPair timer;
    float avgMs = 0.0f;
    if (!timer.measure([&]() { return bench.execute(exe.get(), inputs, outputs); }, c.warmup, c.repeat, &avgMs)) {
        return false;
    }
    MNN_PRINT("[bench_ops/cuda/perf/WeightOnlyConv] %-18s rows=%d ic=%d oc=%d qblock=%d avg=%.4f ms\n",
              c.name, c.rows, c.ic, c.oc, c.quantBlock, avgMs);
    ::fflush(stdout);
    return true;
}

static bool enabledByFilter(const char* name) {
    const char* filter = ::getenv("MNN_BENCH_WEIGHT_ONLY_CASE");
    if (filter == nullptr || filter[0] == '\0') {
        return true;
    }
    return std::string(name).find(filter) != std::string::npos;
}

static bool enabledRow(int row) {
    const char* filter = ::getenv("MNN_BENCH_WEIGHT_ONLY_ROWS");
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
                if (std::atoi(item.c_str()) == row) {
                    return true;
                }
            } else {
                int lo = std::atoi(item.substr(0, dash).c_str());
                int hi = std::atoi(item.substr(dash + 1).c_str());
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

static int envInt(const char* name, int fallback) {
    const char* value = ::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::max(1, std::atoi(value));
}

static int envIntAllowZero(const char* name, int fallback) {
    const char* value = ::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::max(0, std::atoi(value));
}

static bool requireMemoryLow(const char* testName) {
    const int memory = MNNTestSuite::get()->pStaus.memory;
    if (memory != BackendConfig::Memory_Low) {
        MNN_PRINT("%s expects memory=2(Memory_Low) so CUDA creates ConvFpAIntBExecution; got memory=%d.\n",
                  testName, memory);
        return false;
    }
    return true;
}

static bool ensureBenchCacheDirs() {
    ::mkdir(".cache", 0777);
    ::mkdir(".cache/bench_ops", 0777);
    return true;
}

static bool appendExternalInt4Linear(std::ofstream& out, int ic, int oc, int quantBlock,
                                     int patternSalt, PicExternalPlan* plan) {
    if (!out.good() || ic <= 0 || oc <= 0 || quantBlock <= 0 || plan == nullptr) {
        return false;
    }
    const int blockCount = upDivInt(ic, quantBlock);
    const size_t dataCount = static_cast<size_t>(ic) * static_cast<size_t>(oc);
    const size_t packedBytes = (dataCount + 1) / 2;
    const int64_t offset = static_cast<int64_t>(out.tellp());

    const uint8_t dim = 2;
    const uint16_t shape[2] = {static_cast<uint16_t>(oc), static_cast<uint16_t>(ic)};
    const int8_t sampleCount = 16;
    int8_t samples[16];
    for (int i = 0; i < 16; ++i) {
        samples[i] = static_cast<int8_t>(i - 8);
    }
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(shape), sizeof(shape));
    out.write(reinterpret_cast<const char*>(&sampleCount), sizeof(sampleCount));
    out.write(reinterpret_cast<const char*>(samples), sizeof(samples));

    std::vector<uint8_t> packed(packedBytes);
    for (size_t i = 0; i < packedBytes; ++i) {
        const int idx0 = static_cast<int>((((2 * i) * 13 + 7 + patternSalt) & 15));
        int idx1 = 0;
        if (2 * i + 1 < dataCount) {
            idx1 = static_cast<int>((((2 * i + 1) * 13 + 7 + patternSalt) & 15));
        }
        packed[i] = static_cast<uint8_t>((idx0 << 4) | idx1);
    }
    out.write(reinterpret_cast<const char*>(packed.data()), static_cast<std::streamsize>(packed.size()));

    std::vector<float> alpha(static_cast<size_t>(oc) * static_cast<size_t>(blockCount), 0.03125f);
    out.write(reinterpret_cast<const char*>(alpha.data()),
              static_cast<std::streamsize>(alpha.size() * sizeof(float)));
    if (!out.good()) {
        return false;
    }

    constexpr int headerBytes = 1 + 2 * 2 + 1 + 16;
    const int64_t weightLen = static_cast<int64_t>(headerBytes + packed.size());
    const int64_t alphaLen = static_cast<int64_t>(alpha.size() * sizeof(float));
    plan->external = {offset, weightLen, alphaLen, 0, 0};
    plan->quantBit = 4;
    plan->quantBlock = quantBlock;
    plan->aMin = 0;
    plan->readType = 0;
    plan->shapeInt32 = false;
    plan->hasBias = false;
    return true;
}

static std::string gateUpExternalWeightPath(int hidden, int inter, int quantBlock) {
    const char* explicitPath = ::getenv("MNN_BENCH_MLP_EXTERNAL_WEIGHT");
    if (explicitPath != nullptr && explicitPath[0] != '\0') {
        return explicitPath;
    }
    ensureBenchCacheDirs();
    char path[256];
    ::snprintf(path, sizeof(path), ".cache/bench_ops/decode_repair_gateup_%d_%d_q%d.weight",
               hidden, inter, quantBlock);
    return path;
}

static std::string linearExternalWeightPath(int ic, int oc, int quantBlock) {
    ensureBenchCacheDirs();
    char path[256];
    ::snprintf(path, sizeof(path), ".cache/bench_ops/decode_repair_linear_%d_%d_q%d.weight",
               ic, oc, quantBlock);
    return path;
}

static bool buildLinearExternalWeight(const std::string& path, int ic, int oc, int quantBlock,
                                      PicExternalPlan* plan) {
    std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!out.good()) {
        MNN_ERROR("failed to open %s for PicLinearNhwc external weight\n", path.c_str());
        return false;
    }
    if (!appendExternalInt4Linear(out, ic, oc, quantBlock, 9, plan)) {
        return false;
    }
    out.close();
    return out.good();
}

static bool buildGateUpExternalWeights(const std::string& path, int hidden, int inter, int quantBlock,
                                       PicExternalPlan* gatePlan, PicExternalPlan* upPlan) {
    std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!out.good()) {
        MNN_ERROR("failed to open %s for PicGateUp external weight\n", path.c_str());
        return false;
    }
    if (!appendExternalInt4Linear(out, hidden, inter, quantBlock, 0, gatePlan)) {
        return false;
    }
    if (!appendExternalInt4Linear(out, hidden, inter, quantBlock, 5, upPlan)) {
        return false;
    }
    out.close();
    return out.good();
}

static uint16_t floatToHalfBits(float value) {
    const __half h = __float2half(value);
    uint16_t bits = 0;
    ::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

static float halfBitsToFloat(uint16_t bits) {
    __half h;
    ::memcpy(&h, &bits, sizeof(bits));
    return __half2float(h);
}

static bool writeHalfPattern(Tensor* tensor) {
    if (tensor == nullptr) {
        return false;
    }
    std::vector<uint16_t> data(tensor->elementSize());
    for (size_t i = 0; i < data.size(); ++i) {
        const float value = static_cast<float>((static_cast<int>(i * 17 + 5) % 31) - 15) * 0.0078125f;
        data[i] = floatToHalfBits(value);
    }
    return checkCuda(cudaMemcpy(reinterpret_cast<void*>(tensor->deviceId()), data.data(),
                                data.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
                     "cudaMemcpy(writeHalfPattern)");
}

static bool readHalfTensor(const Tensor* tensor, std::vector<float>* values) {
    if (tensor == nullptr || values == nullptr) {
        return false;
    }
    std::vector<uint16_t> data(tensor->elementSize());
    if (!checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize before half readback") ||
        !checkCuda(cudaMemcpy(data.data(), reinterpret_cast<const void*>(tensor->deviceId()),
                              data.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
                   "cudaMemcpy(readHalfTensor)")) {
        return false;
    }
    values->resize(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        (*values)[i] = halfBitsToFloat(data[i]);
    }
    return true;
}

static bool compareHalfOutputs(const char* name, const std::vector<float>& got,
                               const std::vector<float>& expected, float absTol, float relTol) {
    if (got.size() != expected.size()) {
        MNN_ERROR("%s size mismatch: got %zu expected %zu\n", name, got.size(), expected.size());
        return false;
    }
    float maxAbs = 0.0f;
    float maxRel = 0.0f;
    int bad = 0;
    int badIndex = -1;
    for (size_t i = 0; i < got.size(); ++i) {
        const float diff = std::fabs(got[i] - expected[i]);
        const float denom = std::max(1.0f, std::fabs(expected[i]));
        const float rel = diff / denom;
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
    MNN_PRINT("[bench_ops/cuda/accuracy/Rows45Cublas] %-24s max_abs=%.6g max_rel=%.6g bad=%d/%zu\n",
              name, maxAbs, maxRel, bad, got.size());
    if (bad > 0) {
        MNN_ERROR("%s failed first_bad=%d got=%.8f expected=%.8f\n",
                  name, badIndex, got[badIndex], expected[badIndex]);
        return false;
    }
    return true;
}

static bool compareHalfOutputsForSuite(const char* suite, const char* name, const std::vector<float>& got,
                                       const std::vector<float>& expected, float absTol, float relTol) {
    if (got.size() != expected.size()) {
        MNN_ERROR("%s/%s size mismatch: got %zu expected %zu\n", suite, name, got.size(), expected.size());
        return false;
    }
    float maxAbs = 0.0f;
    float maxRel = 0.0f;
    int bad = 0;
    int badIndex = -1;
    for (size_t i = 0; i < got.size(); ++i) {
        const float diff = std::fabs(got[i] - expected[i]);
        const float denom = std::max(1.0f, std::fabs(expected[i]));
        const float rel = diff / denom;
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
    MNN_PRINT("[%s] %-24s max_abs=%.6g max_rel=%.6g bad=%d/%zu\n",
              suite, name, maxAbs, maxRel, bad, got.size());
    if (bad > 0) {
        MNN_ERROR("%s/%s failed first_bad=%d got=%.8f expected=%.8f\n",
                  suite, name, badIndex, got[badIndex], expected[badIndex]);
        return false;
    }
    return true;
}

static bool checkCublas(cublasStatus_t status, const char* where) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        MNN_ERROR("%s failed: %d\n", where, static_cast<int>(status));
        return false;
    }
    return true;
}

#if MNN_BENCH_HAS_CUBLASLT
static bool checkCublasLt(cublasStatus_t status, const char* where) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        MNN_ERROR("%s failed: %d\n", where, static_cast<int>(status));
        return false;
    }
    return true;
}

class CublasLtMatmulPlan {
public:
    CublasLtMatmulPlan() = default;
    ~CublasLtMatmulPlan() {
        if (mPreference != nullptr) {
            cublasLtMatmulPreferenceDestroy(mPreference);
        }
        if (mCDesc != nullptr) {
            cublasLtMatrixLayoutDestroy(mCDesc);
        }
        if (mBDesc != nullptr) {
            cublasLtMatrixLayoutDestroy(mBDesc);
        }
        if (mADesc != nullptr) {
            cublasLtMatrixLayoutDestroy(mADesc);
        }
        if (mOpDesc != nullptr) {
            cublasLtMatmulDescDestroy(mOpDesc);
        }
    }

    CublasLtMatmulPlan(const CublasLtMatmulPlan&) = delete;
    CublasLtMatmulPlan& operator=(const CublasLtMatmulPlan&) = delete;

    bool init(cublasLtHandle_t handle, int rows, int ic, int oc, size_t workspaceBytes, int maxHeuristics = 1) {
        cublasOperation_t transA = CUBLAS_OP_T;
        cublasOperation_t transB = CUBLAS_OP_N;
        if (!checkCublasLt(cublasLtMatmulDescCreate(&mOpDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                           "cublasLtMatmulDescCreate") ||
            !checkCublasLt(cublasLtMatmulDescSetAttribute(
                               mOpDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)),
                           "cublasLtMatmulDescSetAttribute(TRANSA)") ||
            !checkCublasLt(cublasLtMatmulDescSetAttribute(
                               mOpDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)),
                           "cublasLtMatmulDescSetAttribute(TRANSB)") ||
            !checkCublasLt(cublasLtMatrixLayoutCreate(&mADesc, CUDA_R_16F, ic, oc, ic),
                           "cublasLtMatrixLayoutCreate(A)") ||
            !checkCublasLt(cublasLtMatrixLayoutCreate(&mBDesc, CUDA_R_16F, ic, rows, ic),
                           "cublasLtMatrixLayoutCreate(B)") ||
            !checkCublasLt(cublasLtMatrixLayoutCreate(&mCDesc, CUDA_R_16F, oc, rows, oc),
                           "cublasLtMatrixLayoutCreate(C)") ||
            !checkCublasLt(cublasLtMatmulPreferenceCreate(&mPreference),
                           "cublasLtMatmulPreferenceCreate") ||
            !checkCublasLt(cublasLtMatmulPreferenceSetAttribute(
                               mPreference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                               &workspaceBytes, sizeof(workspaceBytes)),
                           "cublasLtMatmulPreferenceSetAttribute(workspace)")) {
            return false;
        }

        maxHeuristics = std::max(1, maxHeuristics);
        mHeuristics.resize(maxHeuristics);
        int returned = 0;
        if (!checkCublasLt(cublasLtMatmulAlgoGetHeuristic(
                               handle, mOpDesc, mADesc, mBDesc, mCDesc, mCDesc,
                               mPreference, maxHeuristics, mHeuristics.data(), &returned),
                           "cublasLtMatmulAlgoGetHeuristic") ||
            returned <= 0) {
            MNN_ERROR("cublasLt did not return a heuristic for rows=%d ic=%d oc=%d\n", rows, ic, oc);
            return false;
        }
        mHeuristics.resize(returned);
        mReady = true;
        return true;
    }

    int heuristicCount() const {
        return static_cast<int>(mHeuristics.size());
    }

    const cublasLtMatmulHeuristicResult_t& heuristic(int index) const {
        return mHeuristics[index];
    }

    bool run(cublasLtHandle_t handle, const void* weight, const void* input, void* output,
             void* workspace, size_t workspaceBytes, int heuristicIndex = 0) const {
        if (!mReady || heuristicIndex < 0 || heuristicIndex >= heuristicCount()) {
            return false;
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const auto& h = mHeuristics[heuristicIndex];
        return checkCublasLt(cublasLtMatmul(handle, mOpDesc,
                                            &alpha,
                                            weight, mADesc,
                                            input, mBDesc,
                                            &beta,
                                            output, mCDesc,
                                            output, mCDesc,
                                            &h.algo,
                                            workspace, workspaceBytes,
                                            0),
                             "cublasLtMatmul");
    }

private:
    cublasLtMatmulDesc_t mOpDesc = nullptr;
    cublasLtMatrixLayout_t mADesc = nullptr;
    cublasLtMatrixLayout_t mBDesc = nullptr;
    cublasLtMatrixLayout_t mCDesc = nullptr;
    cublasLtMatmulPreference_t mPreference = nullptr;
    std::vector<cublasLtMatmulHeuristicResult_t> mHeuristics;
    bool mReady = false;
};
#endif

static bool measureCudaBool(const std::function<bool()>& run, int warmup, int repeat, float* avgMs) {
    if (avgMs == nullptr || repeat <= 0) {
        return false;
    }
    for (int i = 0; i < warmup; ++i) {
        if (!run()) {
            return false;
        }
    }
    if (!checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize before timing")) {
        return false;
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)") ||
        !checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)")) {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        return false;
    }
    bool ok = checkCuda(cudaEventRecord(start, 0), "cudaEventRecord(start)");
    for (int i = 0; ok && i < repeat; ++i) {
        ok = run();
    }
    ok = ok && checkCuda(cudaEventRecord(stop, 0), "cudaEventRecord(stop)");
    ok = ok && checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float totalMs = 0.0f;
    ok = ok && checkCuda(cudaEventElapsedTime(&totalMs, start, stop), "cudaEventElapsedTime");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (!ok) {
        return false;
    }
    *avgMs = totalMs / static_cast<float>(repeat);
    return true;
}

#if MNN_BENCH_HAS_CUBLASLT
static bool runGemmLtFloorCase(cublasLtHandle_t handle, const GemmFloorCase& c, float* avgMs) {
    constexpr size_t elemBytes = 2;
    constexpr size_t workspaceBytes = 4ull * 1024ull * 1024ull;
    const int heuristicLimit = std::max(1, envInt("MNN_BENCH_CUBLASLT_HEURISTICS", 1));
    const bool sweep = envInt("MNN_BENCH_CUBLASLT_SWEEP", 0) != 0;
    CudaDeviceBuffer weight;
    CudaDeviceBuffer input;
    CudaDeviceBuffer output;
    CudaDeviceBuffer workspace;
    if (!weight.alloc(static_cast<size_t>(c.oc) * static_cast<size_t>(c.ic) * elemBytes) ||
        !input.alloc(static_cast<size_t>(c.ic) * static_cast<size_t>(c.rows) * elemBytes) ||
        !output.alloc(static_cast<size_t>(c.oc) * static_cast<size_t>(c.rows) * elemBytes) ||
        !workspace.alloc(workspaceBytes)) {
        return false;
    }
    CublasLtMatmulPlan plan;
    if (!plan.init(handle, c.rows, c.ic, c.oc, workspaceBytes, heuristicLimit)) {
        return false;
    }
    float bestMs = 0.0f;
    int bestIndex = 0;
    const int measureCount = sweep ? plan.heuristicCount() : 1;
    for (int i = 0; i < measureCount; ++i) {
        float ms = 0.0f;
        if (!measureCudaBool([&]() {
                return plan.run(handle, weight.get(), input.get(), output.get(), workspace.get(), workspaceBytes, i);
            }, c.warmup, c.repeat, &ms)) {
            return false;
        }
        const auto& h = plan.heuristic(i);
        if (!sweep) {
            MNN_PRINT("[bench_ops/cuda/perf/DecodeRepairMlpGemmLtFloor] %-18s rows=%d ic=%d oc=%d avg=%.4f ms\n",
                      c.name, c.rows, c.ic, c.oc, ms);
        } else {
            MNN_PRINT("[bench_ops/cuda/perf/DecodeRepairMlpGemmLtFloor] %-18s rows=%d ic=%d oc=%d "
                      "heuristic=%d/%d avg=%.4f ms workspace=%zu waves=%.3f state=%d\n",
                      c.name, c.rows, c.ic, c.oc, i, plan.heuristicCount(), ms,
                      static_cast<size_t>(h.workspaceSize), h.wavesCount, static_cast<int>(h.state));
        }
        if (i == 0 || ms < bestMs) {
            bestMs = ms;
            bestIndex = i;
        }
    }
    *avgMs = bestMs;
    if (sweep) {
        MNN_PRINT("[bench_ops/cuda/perf/DecodeRepairMlpGemmLtFloor] %-18s rows=%d ic=%d oc=%d "
                  "best_heuristic=%d best=%.4f ms returned=%d requested=%d\n",
                  c.name, c.rows, c.ic, c.oc, bestIndex, bestMs, plan.heuristicCount(), heuristicLimit);
    }
    ::fflush(stdout);
    return true;
}
#endif

static bool runFp16Gemm(cublasHandle_t handle, const void* weight, const void* input, void* output,
                        int rows, int ic, int oc) {
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
    return checkCublas(cublasGemmEx(handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    oc, rows, ic,
                                    &alpha,
                                    weight, CUDA_R_16F, oc,
                                    input, CUDA_R_16F, ic,
                                    &beta,
                                    output, CUDA_R_16F, oc,
                                    computeType, algo),
                       "cublasGemmEx(fp16 floor)");
}

static bool runFp16GemmRowsByOc(cublasHandle_t handle, const void* weight, const void* input, void* output,
                                int rows, int ic, int oc) {
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
    return checkCublas(cublasGemmEx(handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    rows, oc, ic,
                                    &alpha,
                                    input, CUDA_R_16F, rows,
                                    weight, CUDA_R_16F, ic,
                                    &beta,
                                    output, CUDA_R_16F, rows,
                                    computeType, algo),
                       "cublasGemmEx(fp16 rows_by_oc floor)");
}

static bool runFp16GemmProductionLayout(cublasHandle_t handle, const void* weight, const void* input, void* output,
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
    return checkCublas(cublasGemmEx(handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    oc, rows, ic,
                                    &alpha,
                                    weight, CUDA_R_16F, icp,
                                    input, CUDA_R_16F, icp,
                                    &beta,
                                    output, CUDA_R_16F, ocp,
                                    computeType, algo),
                       "cublasGemmEx(fp16 production layout floor)");
}

static bool runFp16GemmProductionLayoutChunk(cublasHandle_t handle, const void* weightBase, const void* input,
                                             void* outputBase, int rows, int ic, int icp, int ocChunk, int ocp,
                                             int ocOffset) {
    const auto* weight = static_cast<const uint8_t*>(weightBase) +
        static_cast<size_t>(ocOffset) * static_cast<size_t>(icp) * sizeof(uint16_t);
    auto* output = static_cast<uint8_t*>(outputBase) + static_cast<size_t>(ocOffset) * sizeof(uint16_t);
    return runFp16GemmProductionLayout(handle, weight, input, output, rows, ic, icp, ocChunk, ocp);
}

static bool runFp16GemmTransposedWeightLayout(cublasHandle_t handle, const void* weight, const void* input, void* output,
                                              int rows, int ic, int icp, int oc, int ocp, int computePolicy) {
    const float alpha32 = 1.0f;
    const float beta32 = 0.0f;
    const __half alpha16 = __float2half(1.0f);
    const __half beta16 = __float2half(0.0f);
    const void* alpha = &alpha32;
    const void* beta = &beta32;
#if CUDART_VERSION >= 11000
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    if (computePolicy == 1) {
        computeType = CUBLAS_COMPUTE_32F_FAST_16F;
    } else if (computePolicy == 2) {
        computeType = CUBLAS_COMPUTE_16F;
        alpha = &alpha16;
        beta = &beta16;
    }
#else
    const auto computeType = CUDA_R_32F;
#endif
#if CUDART_VERSION >= 9000
    const auto algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
#else
    const auto algo = CUBLAS_GEMM_DEFAULT;
#endif
    return checkCublas(cublasGemmEx(handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    oc, rows, ic,
                                    alpha,
                                    weight, CUDA_R_16F, ocp,
                                    input, CUDA_R_16F, icp,
                                    beta,
                                    output, CUDA_R_16F, ocp,
                                    computeType, algo),
                       "cublasGemmEx(fp16 transposed weight layout floor)");
}

static bool runFp16GemmProductionLayoutBeta(cublasHandle_t handle, const void* weight, const void* input, void* output,
                                            int rows, int ic, int icp, int oc, int ocp, float betaValue) {
    const float alpha = 1.0f;
    const float beta = betaValue;
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
    return checkCublas(cublasGemmEx(handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    oc, rows, ic,
                                    &alpha,
                                    weight, CUDA_R_16F, icp,
                                    input, CUDA_R_16F, icp,
                                    &beta,
                                    output, CUDA_R_16F, ocp,
                                    computeType, algo),
                       "cublasGemmEx(fp16 production layout beta floor)");
}

static ErrorCode cublasDownError(cublasHandle_t handle, const Tensor* weight, const Tensor* input, Tensor* output,
                                 int rows, int inter, int hidden) {
    const int interP = upDivInt(inter, 8) * 8;
    const int hiddenP = upDivInt(hidden, 8) * 8;
    if (!runFp16GemmProductionLayout(handle,
                                     reinterpret_cast<const void*>(weight->deviceId()),
                                     reinterpret_cast<const void*>(input->deviceId()),
                                     reinterpret_cast<void*>(output->deviceId()),
                                     rows, inter, interP, hidden, hiddenP)) {
        return INVALID_VALUE;
    }
    return NO_ERROR;
}

static bool runPicFusedSiluDownAccuracyCase(int rows, int hidden, int inter) {
    KVMeta meta;
    meta.pic_decode_repair_sparse_active = true;
    DirectOpBench bench(MNN_FORWARD_CUDA, &meta);
    if (!bench.valid()) {
        return false;
    }

    auto packedGateUp = bench.tensor({rows, inter * 2, 1, 1}, Tensor::CAFFE, false);
    auto downWeight = bench.tensor({hidden, inter, 1, 1}, Tensor::CAFFE, false);
    auto refActivation = bench.tensor({rows, inter, 1, 1}, Tensor::CAFFE, false);
    auto refOutput = bench.tensor({rows, hidden, 1, 1}, Tensor::CAFFE, false);
    auto materializedOutput = bench.tensor({rows, hidden, 1, 1}, Tensor::CAFFE, false);
    auto streamedOutput = bench.tensor({rows, hidden, 1, 1}, Tensor::CAFFE, false);
    auto wmmaOutput = bench.tensor({rows, hidden, 1, 1}, Tensor::CAFFE, false);
    if (!packedGateUp || !downWeight || !refActivation || !refOutput || !materializedOutput || !streamedOutput ||
        !wmmaOutput || !writeHalfPattern(packedGateUp) || !writeHalfPattern(downWeight)) {
        return false;
    }

    auto packedSiluOp = makePicPackedSiluMulOp("bench_pic_fused_silu_down_ref_silu");
    auto materializedOp = makePicBenchPackedSiluDownOp("PicBenchFusedPackedSiluDown",
                                                       "bench_pic_fused_silu_down_materialized");
    auto streamedOp = makePicBenchPackedSiluDownOp("PicBenchStreamedPackedSiluDown",
                                                   "bench_pic_fused_silu_down_streamed");
    auto wmmaOp = makePicBenchPackedSiluDownOp("PicBenchWmmaPackedSiluDown",
                                               "bench_pic_fused_silu_down_wmma");
    std::vector<Tensor*> siluInputs = {packedGateUp};
    std::vector<Tensor*> siluOutputs = {refActivation};
    std::vector<Tensor*> materializedInputs = {packedGateUp, downWeight};
    std::vector<Tensor*> materializedOutputs = {materializedOutput};
    std::vector<Tensor*> streamedInputs = {packedGateUp, downWeight};
    std::vector<Tensor*> streamedOutputs = {streamedOutput};
    std::vector<Tensor*> wmmaInputs = {packedGateUp, downWeight};
    std::vector<Tensor*> wmmaOutputs = {wmmaOutput};
    auto siluExe = bench.create(siluInputs, siluOutputs, packedSiluOp->get());
    auto materializedExe = bench.create(materializedInputs, materializedOutputs, materializedOp->get());
    auto streamedExe = bench.create(streamedInputs, streamedOutputs, streamedOp->get());
    auto wmmaExe = bench.create(wmmaInputs, wmmaOutputs, wmmaOp->get());
    if (!siluExe || !materializedExe || !streamedExe || !wmmaExe) {
        MNN_ERROR("PicFusedSiluDown failed to create execution rows=%d\n", rows);
        return false;
    }
    if (bench.resize(siluExe.get(), siluInputs, siluOutputs) != NO_ERROR ||
        bench.resize(materializedExe.get(), materializedInputs, materializedOutputs) != NO_ERROR ||
        bench.resize(streamedExe.get(), streamedInputs, streamedOutputs) != NO_ERROR ||
        bench.resize(wmmaExe.get(), wmmaInputs, wmmaOutputs) != NO_ERROR) {
        MNN_ERROR("PicFusedSiluDown resize failed rows=%d\n", rows);
        return false;
    }

    cublasHandle_t handle = nullptr;
    if (!checkCublas(cublasCreate(&handle), "cublasCreate(PicFusedSiluDown accuracy)")) {
        return false;
    }
#if CUDART_VERSION >= 9000
    checkCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode(PicFusedSiluDown accuracy)");
#endif
    bool ok = true;
    if (bench.execute(siluExe.get(), siluInputs, siluOutputs) != NO_ERROR ||
        cublasDownError(handle, downWeight, refActivation, refOutput, rows, inter, hidden) != NO_ERROR ||
        bench.execute(materializedExe.get(), materializedInputs, materializedOutputs) != NO_ERROR ||
        bench.execute(streamedExe.get(), streamedInputs, streamedOutputs) != NO_ERROR ||
        bench.execute(wmmaExe.get(), wmmaInputs, wmmaOutputs) != NO_ERROR) {
        ok = false;
    }
    cublasDestroy(handle);
    if (!ok) {
        return false;
    }

    std::vector<float> ref;
    std::vector<float> materialized;
    std::vector<float> streamed;
    std::vector<float> wmma;
    if (!readHalfTensor(refOutput, &ref) ||
        !readHalfTensor(materializedOutput, &materialized) ||
        !readHalfTensor(streamedOutput, &streamed) ||
        !readHalfTensor(wmmaOutput, &wmma)) {
        return false;
    }
    char name[128];
    ::snprintf(name, sizeof(name), "rows%d_materialized", rows);
    ok = compareHalfOutputsForSuite("bench_ops/cuda/accuracy/PicFusedSiluDown", name,
                                    materialized, ref, 0.08f, 0.08f) && ok;
    ::snprintf(name, sizeof(name), "rows%d_streamed", rows);
    ok = compareHalfOutputsForSuite("bench_ops/cuda/accuracy/PicFusedSiluDown", name,
                                    streamed, ref, 0.08f, 0.08f) && ok;
    ::snprintf(name, sizeof(name), "rows%d_wmma", rows);
    ok = compareHalfOutputsForSuite("bench_ops/cuda/accuracy/PicFusedSiluDown", name,
                                    wmma, ref, 0.08f, 0.08f) && ok;
    return ok;
}

static bool runPicFusedSiluDownPerfCase(int rows, int hidden, int inter, int warmup, int repeat) {
    constexpr int quantBlock = 64;
    KVMeta meta;
    meta.pic_decode_repair_sparse_active = true;
    DirectOpBench bench(MNN_FORWARD_CUDA, &meta);
    if (!bench.valid()) {
        return false;
    }

    auto input = bench.tensor({rows, 1, 1, hidden}, Tensor::TENSORFLOW, false);
    auto packedGateUp = bench.tensor({rows, 1, 1, inter * 2}, Tensor::TENSORFLOW, false);
    auto downWeight = bench.tensor({hidden, inter, 1, 1}, Tensor::CAFFE, false);
    auto activation = bench.tensor({rows, inter, 1, 1}, Tensor::CAFFE, false);
    auto refOutput = bench.tensor({rows, hidden, 1, 1}, Tensor::CAFFE, false);
    auto materializedOutput = bench.tensor({rows, hidden, 1, 1}, Tensor::CAFFE, false);
    auto streamedOutput = bench.tensor({rows, hidden, 1, 1}, Tensor::CAFFE, false);
    auto wmmaOutput = bench.tensor({rows, hidden, 1, 1}, Tensor::CAFFE, false);
    if (!input || !packedGateUp || !downWeight || !activation || !refOutput ||
        !materializedOutput || !streamedOutput || !wmmaOutput ||
        !writeHalfPattern(input) || !writeHalfPattern(downWeight)) {
        return false;
    }

    PicExternalPlan packedGateUpPlan;
    const std::string externalPath = linearExternalWeightPath(hidden, inter * 2, quantBlock);
    if (!buildLinearExternalWeight(externalPath, hidden, inter * 2, quantBlock, &packedGateUpPlan)) {
        return false;
    }

    auto gateUpOp = makePicLinearNhwcWeightOnlyExtraOp("bench_pic_fused_silu_down_packed_gateup",
                                                       externalPath, hidden, inter * 2, packedGateUpPlan);
    auto packedSiluOp = makePicPackedSiluMulOp("bench_pic_fused_silu_down_ref_silu");
    auto materializedOp = makePicBenchPackedSiluDownOp("PicBenchFusedPackedSiluDown",
                                                       "bench_pic_fused_silu_down_materialized");
    auto streamedOp = makePicBenchPackedSiluDownOp("PicBenchStreamedPackedSiluDown",
                                                   "bench_pic_fused_silu_down_streamed");
    auto wmmaOp = makePicBenchPackedSiluDownOp("PicBenchWmmaPackedSiluDown",
                                               "bench_pic_fused_silu_down_wmma");
    std::vector<Tensor*> gateUpInputs = {input};
    std::vector<Tensor*> gateUpOutputs = {packedGateUp};
    std::vector<Tensor*> siluInputs = {packedGateUp};
    std::vector<Tensor*> siluOutputs = {activation};
    std::vector<Tensor*> materializedInputs = {packedGateUp, downWeight};
    std::vector<Tensor*> materializedOutputs = {materializedOutput};
    std::vector<Tensor*> streamedInputs = {packedGateUp, downWeight};
    std::vector<Tensor*> streamedOutputs = {streamedOutput};
    std::vector<Tensor*> wmmaInputs = {packedGateUp, downWeight};
    std::vector<Tensor*> wmmaOutputs = {wmmaOutput};
    auto gateUpExe = bench.create(gateUpInputs, gateUpOutputs, gateUpOp->get());
    auto siluExe = bench.create(siluInputs, siluOutputs, packedSiluOp->get());
    auto materializedExe = bench.create(materializedInputs, materializedOutputs, materializedOp->get());
    auto streamedExe = bench.create(streamedInputs, streamedOutputs, streamedOp->get());
    auto wmmaExe = bench.create(wmmaInputs, wmmaOutputs, wmmaOp->get());
    if (!gateUpExe || !siluExe || !materializedExe || !streamedExe || !wmmaExe) {
        MNN_ERROR("PicFusedSiluDown perf failed to create execution rows=%d\n", rows);
        return false;
    }
    if (bench.resize(gateUpExe.get(), gateUpInputs, gateUpOutputs) != NO_ERROR ||
        bench.resize(siluExe.get(), siluInputs, siluOutputs) != NO_ERROR ||
        bench.resize(materializedExe.get(), materializedInputs, materializedOutputs) != NO_ERROR ||
        bench.resize(streamedExe.get(), streamedInputs, streamedOutputs) != NO_ERROR ||
        bench.resize(wmmaExe.get(), wmmaInputs, wmmaOutputs) != NO_ERROR) {
        MNN_ERROR("PicFusedSiluDown perf resize failed rows=%d\n", rows);
        return false;
    }

    cublasHandle_t handle = nullptr;
    if (!checkCublas(cublasCreate(&handle), "cublasCreate(PicFusedSiluDown perf)")) {
        return false;
    }
#if CUDART_VERSION >= 9000
    checkCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode(PicFusedSiluDown perf)");
#endif
    if (bench.execute(gateUpExe.get(), gateUpInputs, gateUpOutputs) != NO_ERROR ||
        bench.execute(siluExe.get(), siluInputs, siluOutputs) != NO_ERROR ||
        cublasDownError(handle, downWeight, activation, refOutput, rows, inter, hidden) != NO_ERROR ||
        bench.execute(materializedExe.get(), materializedInputs, materializedOutputs) != NO_ERROR ||
        bench.execute(streamedExe.get(), streamedInputs, streamedOutputs) != NO_ERROR ||
        bench.execute(wmmaExe.get(), wmmaInputs, wmmaOutputs) != NO_ERROR) {
        cublasDestroy(handle);
        return false;
    }

    CudaEventPair timer;
    float gateUpMs = 0.0f;
    float packedSiluMs = 0.0f;
    float downMs = 0.0f;
    float packedSiluDownMs = 0.0f;
    float materializedMs = 0.0f;
    float streamedMs = 0.0f;
    float wmmaMs = 0.0f;
    float baselineChainMs = 0.0f;
    float materializedChainMs = 0.0f;
    float streamedChainMs = 0.0f;
    float wmmaChainMs = 0.0f;
    if (!timer.measure([&]() { return bench.execute(gateUpExe.get(), gateUpInputs, gateUpOutputs); },
                       warmup, repeat, &gateUpMs) ||
        !timer.measure([&]() { return bench.execute(siluExe.get(), siluInputs, siluOutputs); },
                       warmup, repeat, &packedSiluMs) ||
        !timer.measure([&]() { return cublasDownError(handle, downWeight, activation, refOutput, rows, inter, hidden); },
                       warmup, repeat, &downMs) ||
        !timer.measure([&]() {
            auto code = bench.execute(siluExe.get(), siluInputs, siluOutputs);
            if (code != NO_ERROR) {
                return code;
            }
            return cublasDownError(handle, downWeight, activation, refOutput, rows, inter, hidden);
        }, warmup, repeat, &packedSiluDownMs) ||
        !timer.measure([&]() { return bench.execute(materializedExe.get(), materializedInputs, materializedOutputs); },
                       warmup, repeat, &materializedMs) ||
        !timer.measure([&]() { return bench.execute(streamedExe.get(), streamedInputs, streamedOutputs); },
                       warmup, repeat, &streamedMs) ||
        !timer.measure([&]() { return bench.execute(wmmaExe.get(), wmmaInputs, wmmaOutputs); },
                       warmup, repeat, &wmmaMs) ||
        !timer.measure([&]() {
            auto code = bench.execute(gateUpExe.get(), gateUpInputs, gateUpOutputs);
            if (code != NO_ERROR) {
                return code;
            }
            code = bench.execute(siluExe.get(), siluInputs, siluOutputs);
            if (code != NO_ERROR) {
                return code;
            }
            return cublasDownError(handle, downWeight, activation, refOutput, rows, inter, hidden);
        }, warmup, repeat, &baselineChainMs) ||
        !timer.measure([&]() {
            auto code = bench.execute(gateUpExe.get(), gateUpInputs, gateUpOutputs);
            if (code != NO_ERROR) {
                return code;
            }
            return bench.execute(materializedExe.get(), materializedInputs, materializedOutputs);
        }, warmup, repeat, &materializedChainMs) ||
        !timer.measure([&]() {
            auto code = bench.execute(gateUpExe.get(), gateUpInputs, gateUpOutputs);
            if (code != NO_ERROR) {
                return code;
            }
            return bench.execute(streamedExe.get(), streamedInputs, streamedOutputs);
        }, warmup, repeat, &streamedChainMs) ||
        !timer.measure([&]() {
            auto code = bench.execute(gateUpExe.get(), gateUpInputs, gateUpOutputs);
            if (code != NO_ERROR) {
                return code;
            }
            return bench.execute(wmmaExe.get(), wmmaInputs, wmmaOutputs);
        }, warmup, repeat, &wmmaChainMs)) {
        cublasDestroy(handle);
        return false;
    }
    cublasDestroy(handle);

    const float materializedDelta = materializedChainMs - baselineChainMs;
    const float streamedDelta = streamedChainMs - baselineChainMs;
    const float wmmaDelta = wmmaChainMs - baselineChainMs;
    MNN_PRINT("[bench_ops/cuda/perf/PicFusedSiluDown] rows=%d hidden=%d inter=%d "
              "gateup=%.4f ms packed_silu=%.4f ms down_cublas=%.4f ms "
              "packed_silu_down=%.4f ms materialized_fused=%.4f ms streamed_fused=%.4f ms wmma_fused=%.4f ms "
              "baseline_chain=%.4f ms materialized_chain=%.4f ms streamed_chain=%.4f ms wmma_chain=%.4f ms "
              "materialized_delta=%.4f ms streamed_delta=%.4f ms wmma_delta=%.4f ms "
              "materialized_gate=%s streamed_gate=%s wmma_gate=%s external=%s\n",
              rows, hidden, inter,
              gateUpMs, packedSiluMs, downMs, packedSiluDownMs, materializedMs, streamedMs, wmmaMs,
              baselineChainMs, materializedChainMs, streamedChainMs, wmmaChainMs,
              materializedDelta, streamedDelta, wmmaDelta,
              materializedDelta <= -0.25f ? "pass" : "fail",
              streamedDelta <= -0.25f ? "pass" : "fail",
              wmmaDelta <= -0.25f ? "pass" : "fail",
              externalPath.c_str());
    ::fflush(stdout);
    return true;
}

static bool runFp16GemmBatchedProductionLayout(cublasHandle_t handle, const void* const* deviceA,
                                               const void* const* deviceB, void* const* deviceC,
                                               int rows, int ic, int icp, int oc, int ocp, int batchCount) {
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
    return checkCublas(cublasGemmBatchedEx(handle,
                                           CUBLAS_OP_T, CUBLAS_OP_N,
                                           oc, rows, ic,
                                           &alpha,
                                           deviceA, CUDA_R_16F, icp,
                                           deviceB, CUDA_R_16F, icp,
                                           &beta,
                                           deviceC, CUDA_R_16F, ocp,
                                           batchCount,
                                           computeType, algo),
                       "cublasGemmBatchedEx(fp16 production layout floor)");
}

static bool runFp16GemmStridedBatchedProductionLayout(cublasHandle_t handle, const void* weight, const void* input,
                                                      void* output, int rows, int ic, int icp, int oc, int ocp,
                                                      int batchCount) {
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
    const long long strideA = static_cast<long long>(ocp) * static_cast<long long>(icp);
    const long long strideB = 0;
    const long long strideC = static_cast<long long>(ocp) * static_cast<long long>(rows);
    return checkCublas(cublasGemmStridedBatchedEx(handle,
                                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                                  oc, rows, ic,
                                                  &alpha,
                                                  weight, CUDA_R_16F, icp, strideA,
                                                  input, CUDA_R_16F, icp, strideB,
                                                  &beta,
                                                  output, CUDA_R_16F, ocp, strideC,
                                                  batchCount,
                                                  computeType, algo),
                       "cublasGemmStridedBatchedEx(fp16 production layout floor)");
}

static bool runGemmFloorCase(cublasHandle_t handle, const GemmFloorCase& c, float* avgMs) {
    CudaDeviceBuffer weight;
    CudaDeviceBuffer input;
    CudaDeviceBuffer output;
    const size_t elemBytes = 2;
    if (!weight.alloc(static_cast<size_t>(c.oc) * static_cast<size_t>(c.ic) * elemBytes) ||
        !input.alloc(static_cast<size_t>(c.ic) * static_cast<size_t>(c.rows) * elemBytes) ||
        !output.alloc(static_cast<size_t>(c.oc) * static_cast<size_t>(c.rows) * elemBytes)) {
        return false;
    }
    if (!measureCudaBool([&]() { return runFp16Gemm(handle, weight.get(), input.get(), output.get(), c.rows, c.ic, c.oc); },
                         c.warmup, c.repeat, avgMs)) {
        return false;
    }
    MNN_PRINT("[bench_ops/cuda/perf/DecodeRepairMlpGemmFloor] %-18s rows=%d ic=%d oc=%d avg=%.4f ms\n",
              c.name, c.rows, c.ic, c.oc, *avgMs);
    ::fflush(stdout);
    return true;
}

static bool runGemmFloorCaseRowsByOc(cublasHandle_t handle, const GemmFloorCase& c, float* avgMs) {
    CudaDeviceBuffer weight;
    CudaDeviceBuffer input;
    CudaDeviceBuffer output;
    const size_t elemBytes = 2;
    if (!weight.alloc(static_cast<size_t>(c.ic) * static_cast<size_t>(c.oc) * elemBytes) ||
        !input.alloc(static_cast<size_t>(c.rows) * static_cast<size_t>(c.ic) * elemBytes) ||
        !output.alloc(static_cast<size_t>(c.rows) * static_cast<size_t>(c.oc) * elemBytes)) {
        return false;
    }
    if (!measureCudaBool([&]() {
            return runFp16GemmRowsByOc(handle, weight.get(), input.get(), output.get(), c.rows, c.ic, c.oc);
        }, c.warmup, c.repeat, avgMs)) {
        return false;
    }
    MNN_PRINT("[bench_ops/cuda/perf/DecodeRepairMlpGemmFloor] %-18s rows=%d ic=%d oc=%d layout=rows_by_oc avg=%.4f ms\n",
              c.name, c.rows, c.ic, c.oc, *avgMs);
    ::fflush(stdout);
    return true;
}

static bool runGateUpBatchedGemmFloorCase(cublasHandle_t handle, int rows, int hidden, int inter,
                                          int warmup, int repeat) {
    constexpr size_t elemBytes = 2;
    const int ic = hidden;
    const int oc = inter;
    const int icp = upDivInt(ic, 8) * 8;
    const int ocp = upDivInt(oc, 8) * 8;
    const size_t weightBytes = static_cast<size_t>(ocp) * static_cast<size_t>(icp) * elemBytes;
    const size_t inputBytes = static_cast<size_t>(rows) * static_cast<size_t>(icp) * elemBytes;
    const size_t outputBytes = static_cast<size_t>(rows) * static_cast<size_t>(ocp) * elemBytes;

    CudaDeviceBuffer gateWeight;
    CudaDeviceBuffer upWeight;
    CudaDeviceBuffer packedWeight;
    CudaDeviceBuffer input;
    CudaDeviceBuffer gateOutput;
    CudaDeviceBuffer upOutput;
    CudaDeviceBuffer packedOutput;
    CudaDeviceBuffer pointerArrays;
    if (!gateWeight.alloc(weightBytes) ||
        !upWeight.alloc(weightBytes) ||
        !packedWeight.alloc(weightBytes * 2) ||
        !input.alloc(inputBytes) ||
        !gateOutput.alloc(outputBytes) ||
        !upOutput.alloc(outputBytes) ||
        !packedOutput.alloc(outputBytes * 2) ||
        !pointerArrays.alloc(sizeof(void*) * 6)) {
        return false;
    }

    const void* hostA[2] = {gateWeight.get(), upWeight.get()};
    const void* hostB[2] = {input.get(), input.get()};
    void* hostC[2] = {gateOutput.get(), upOutput.get()};
    if (!checkCuda(cudaMemcpy(pointerArrays.get(), hostA, sizeof(hostA), cudaMemcpyHostToDevice),
                   "cudaMemcpy(gateup A pointer array)") ||
        !checkCuda(cudaMemcpy(static_cast<uint8_t*>(pointerArrays.get()) + sizeof(hostA),
                              hostB, sizeof(hostB), cudaMemcpyHostToDevice),
                   "cudaMemcpy(gateup B pointer array)") ||
        !checkCuda(cudaMemcpy(static_cast<uint8_t*>(pointerArrays.get()) + sizeof(hostA) + sizeof(hostB),
                              hostC, sizeof(hostC), cudaMemcpyHostToDevice),
                   "cudaMemcpy(gateup C pointer array)")) {
        return false;
    }
    const void* const* deviceA = reinterpret_cast<const void* const*>(pointerArrays.get());
    const void* const* deviceB = reinterpret_cast<const void* const*>(
        static_cast<uint8_t*>(pointerArrays.get()) + sizeof(hostA));
    void* const* deviceC = reinterpret_cast<void* const*>(
        static_cast<uint8_t*>(pointerArrays.get()) + sizeof(hostA) + sizeof(hostB));

    float gateMs = 0.0f;
    float upMs = 0.0f;
    float batchedPtrMs = 0.0f;
    float batchedStridedMs = 0.0f;
    if (!measureCudaBool([&]() {
            return runFp16GemmProductionLayout(
                handle, gateWeight.get(), input.get(), gateOutput.get(), rows, ic, icp, oc, ocp);
        }, warmup, repeat, &gateMs)) {
        return false;
    }
    if (!measureCudaBool([&]() {
            return runFp16GemmProductionLayout(
                handle, upWeight.get(), input.get(), upOutput.get(), rows, ic, icp, oc, ocp);
        }, warmup, repeat, &upMs)) {
        return false;
    }
    if (!measureCudaBool([&]() {
            return runFp16GemmBatchedProductionLayout(
                handle, deviceA, deviceB, deviceC, rows, ic, icp, oc, ocp, 2);
        }, warmup, repeat, &batchedPtrMs)) {
        return false;
    }
    if (!measureCudaBool([&]() {
            return runFp16GemmStridedBatchedProductionLayout(
                handle, packedWeight.get(), input.get(), packedOutput.get(), rows, ic, icp, oc, ocp, 2);
        }, warmup, repeat, &batchedStridedMs)) {
        return false;
    }

    const float separateMs = gateMs + upMs;
    MNN_PRINT("[bench_ops/cuda/perf/DecodeRepairGateUpBatchedGemmFloor] rows=%d hidden=%d inter=%d "
              "gate=%.4f ms up=%.4f ms separate_sum=%.4f ms "
              "batched_ptr=%.4f ms ptr_delta=%.4f ms "
              "batched_strided=%.4f ms strided_delta=%.4f ms\n",
              rows, hidden, inter, gateMs, upMs, separateMs,
              batchedPtrMs, batchedPtrMs - separateMs,
              batchedStridedMs, batchedStridedMs - separateMs);
    ::fflush(stdout);
    return true;
}

static bool createTensorOpCublasHandle(cublasHandle_t* handle, cudaStream_t stream, const char* where) {
    if (handle == nullptr || !checkCublas(cublasCreate(handle), where)) {
        return false;
    }
#if CUDART_VERSION >= 9000
    if (!checkCublas(cublasSetMathMode(*handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode(TENSOR_OP)")) {
        return false;
    }
#endif
    return checkCublas(cublasSetStream(*handle, stream), "cublasSetStream");
}

static bool measureGateUpSequentialPair(cublasHandle_t handle, cudaStream_t stream,
                                        const void* gateWeight, const void* upWeight,
                                        const void* input, void* gateOutput, void* upOutput,
                                        int rows, int ic, int icp, int oc, int ocp,
                                        int warmup, int repeat, float* avgMs) {
    if (avgMs == nullptr || repeat <= 0) {
        return false;
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!checkCuda(cudaEventCreate(&start), "cudaEventCreate(seq start)") ||
        !checkCuda(cudaEventCreate(&stop), "cudaEventCreate(seq stop)")) {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        return false;
    }
    auto destroyEvents = [&]() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };
    auto runPair = [&]() -> bool {
        return runFp16GemmProductionLayout(handle, gateWeight, input, gateOutput, rows, ic, icp, oc, ocp) &&
               runFp16GemmProductionLayout(handle, upWeight, input, upOutput, rows, ic, icp, oc, ocp);
    };
    for (int i = 0; i < warmup; ++i) {
        if (!runPair()) {
            destroyEvents();
            return false;
        }
    }
    if (!checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(seq warmup)")) {
        destroyEvents();
        return false;
    }
    float totalMs = 0.0f;
    for (int i = 0; i < repeat; ++i) {
        if (!checkCuda(cudaEventRecord(start, stream), "cudaEventRecord(seq start)")) {
            destroyEvents();
            return false;
        }
        if (!runPair()) {
            destroyEvents();
            return false;
        }
        if (!checkCuda(cudaEventRecord(stop, stream), "cudaEventRecord(seq stop)") ||
            !checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(seq stop)")) {
            destroyEvents();
            return false;
        }
        float iterMs = 0.0f;
        if (!checkCuda(cudaEventElapsedTime(&iterMs, start, stop), "cudaEventElapsedTime(seq)")) {
            destroyEvents();
            return false;
        }
        totalMs += iterMs;
    }
    destroyEvents();
    *avgMs = totalMs / static_cast<float>(repeat);
    return true;
}

static bool measureGateUpParallelPair(cublasHandle_t gateHandle, cublasHandle_t upHandle,
                                      cudaStream_t gateStream, cudaStream_t upStream, cudaStream_t timingStream,
                                      const void* gateWeight, const void* upWeight,
                                      const void* input, void* gateOutput, void* upOutput,
                                      int rows, int ic, int icp, int oc, int ocp,
                                      int warmup, int repeat, float* avgMs) {
    if (avgMs == nullptr || repeat <= 0) {
        return false;
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t gateDone = nullptr;
    cudaEvent_t upDone = nullptr;
    cudaEvent_t stop = nullptr;
    if (!checkCuda(cudaEventCreate(&start), "cudaEventCreate(par start)") ||
        !checkCuda(cudaEventCreate(&gateDone), "cudaEventCreate(par gateDone)") ||
        !checkCuda(cudaEventCreate(&upDone), "cudaEventCreate(par upDone)") ||
        !checkCuda(cudaEventCreate(&stop), "cudaEventCreate(par stop)")) {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (gateDone != nullptr) {
            cudaEventDestroy(gateDone);
        }
        if (upDone != nullptr) {
            cudaEventDestroy(upDone);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        return false;
    }
    auto destroyEvents = [&]() {
        cudaEventDestroy(start);
        cudaEventDestroy(gateDone);
        cudaEventDestroy(upDone);
        cudaEventDestroy(stop);
    };
    auto runPair = [&]() -> bool {
        if (!checkCuda(cudaEventRecord(start, timingStream), "cudaEventRecord(par start)") ||
            !checkCuda(cudaStreamWaitEvent(gateStream, start, 0), "cudaStreamWaitEvent(gate start)") ||
            !checkCuda(cudaStreamWaitEvent(upStream, start, 0), "cudaStreamWaitEvent(up start)")) {
            return false;
        }
        if (!runFp16GemmProductionLayout(gateHandle, gateWeight, input, gateOutput, rows, ic, icp, oc, ocp) ||
            !runFp16GemmProductionLayout(upHandle, upWeight, input, upOutput, rows, ic, icp, oc, ocp)) {
            return false;
        }
        return checkCuda(cudaEventRecord(gateDone, gateStream), "cudaEventRecord(gate done)") &&
               checkCuda(cudaEventRecord(upDone, upStream), "cudaEventRecord(up done)") &&
               checkCuda(cudaStreamWaitEvent(timingStream, gateDone, 0), "cudaStreamWaitEvent(timing gate done)") &&
               checkCuda(cudaStreamWaitEvent(timingStream, upDone, 0), "cudaStreamWaitEvent(timing up done)");
    };
    for (int i = 0; i < warmup; ++i) {
        if (!runPair() ||
            !checkCuda(cudaEventRecord(stop, timingStream), "cudaEventRecord(par warm stop)") ||
            !checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(par warm stop)")) {
            destroyEvents();
            return false;
        }
    }
    float totalMs = 0.0f;
    for (int i = 0; i < repeat; ++i) {
        if (!runPair() ||
            !checkCuda(cudaEventRecord(stop, timingStream), "cudaEventRecord(par stop)") ||
            !checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(par stop)")) {
            destroyEvents();
            return false;
        }
        float iterMs = 0.0f;
        if (!checkCuda(cudaEventElapsedTime(&iterMs, start, stop), "cudaEventElapsedTime(par)")) {
            destroyEvents();
            return false;
        }
        totalMs += iterMs;
    }
    destroyEvents();
    *avgMs = totalMs / static_cast<float>(repeat);
    return true;
}

static bool runGateUpParallelGemmFloorCase(int rows, int hidden, int inter, int warmup, int repeat) {
    constexpr size_t elemBytes = 2;
    const int ic = hidden;
    const int oc = inter;
    const int icp = upDivInt(ic, 8) * 8;
    const int ocp = upDivInt(oc, 8) * 8;
    const size_t weightBytes = static_cast<size_t>(ocp) * static_cast<size_t>(icp) * elemBytes;
    const size_t inputBytes = static_cast<size_t>(rows) * static_cast<size_t>(icp) * elemBytes;
    const size_t outputBytes = static_cast<size_t>(rows) * static_cast<size_t>(ocp) * elemBytes;

    CudaDeviceBuffer gateWeight;
    CudaDeviceBuffer upWeight;
    CudaDeviceBuffer input;
    CudaDeviceBuffer gateOutput;
    CudaDeviceBuffer upOutput;
    if (!gateWeight.alloc(weightBytes) ||
        !upWeight.alloc(weightBytes) ||
        !input.alloc(inputBytes) ||
        !gateOutput.alloc(outputBytes) ||
        !upOutput.alloc(outputBytes)) {
        return false;
    }

    cudaStream_t seqStream = nullptr;
    cudaStream_t gateStream = nullptr;
    cudaStream_t upStream = nullptr;
    cudaStream_t timingStream = nullptr;
    cublasHandle_t seqHandle = nullptr;
    cublasHandle_t gateHandle = nullptr;
    cublasHandle_t upHandle = nullptr;
    auto cleanup = [&]() {
        if (seqHandle != nullptr) {
            cublasDestroy(seqHandle);
        }
        if (gateHandle != nullptr) {
            cublasDestroy(gateHandle);
        }
        if (upHandle != nullptr) {
            cublasDestroy(upHandle);
        }
        if (seqStream != nullptr) {
            cudaStreamDestroy(seqStream);
        }
        if (gateStream != nullptr) {
            cudaStreamDestroy(gateStream);
        }
        if (upStream != nullptr) {
            cudaStreamDestroy(upStream);
        }
        if (timingStream != nullptr) {
            cudaStreamDestroy(timingStream);
        }
    };
    if (!checkCuda(cudaStreamCreateWithFlags(&seqStream, cudaStreamNonBlocking), "cudaStreamCreate(seq)") ||
        !checkCuda(cudaStreamCreateWithFlags(&gateStream, cudaStreamNonBlocking), "cudaStreamCreate(gate)") ||
        !checkCuda(cudaStreamCreateWithFlags(&upStream, cudaStreamNonBlocking), "cudaStreamCreate(up)") ||
        !checkCuda(cudaStreamCreateWithFlags(&timingStream, cudaStreamNonBlocking), "cudaStreamCreate(timing)") ||
        !createTensorOpCublasHandle(&seqHandle, seqStream, "cublasCreate(seq)") ||
        !createTensorOpCublasHandle(&gateHandle, gateStream, "cublasCreate(gate)") ||
        !createTensorOpCublasHandle(&upHandle, upStream, "cublasCreate(up)")) {
        cleanup();
        return false;
    }

    float sequentialMs = 0.0f;
    float parallelMs = 0.0f;
    if (!measureGateUpSequentialPair(seqHandle, seqStream,
                                     gateWeight.get(), upWeight.get(), input.get(),
                                     gateOutput.get(), upOutput.get(),
                                     rows, ic, icp, oc, ocp, warmup, repeat, &sequentialMs) ||
        !measureGateUpParallelPair(gateHandle, upHandle, gateStream, upStream, timingStream,
                                   gateWeight.get(), upWeight.get(), input.get(),
                                   gateOutput.get(), upOutput.get(),
                                   rows, ic, icp, oc, ocp, warmup, repeat, &parallelMs)) {
        cleanup();
        return false;
    }
    cleanup();
    const float delta = parallelMs - sequentialMs;
    const float speedup = parallelMs > 0.0f ? sequentialMs / parallelMs : 0.0f;
    MNN_PRINT("[bench_ops/cuda/perf/DecodeRepairGateUpParallelGemmFloor] rows=%d hidden=%d inter=%d "
              "sequential_pair=%.4f ms parallel_pair=%.4f ms delta=%.4f ms speedup=%.3f\n",
              rows, hidden, inter, sequentialMs, parallelMs, delta, speedup);
    ::fflush(stdout);
    return true;
}

static bool measureGemmProductionLayoutOnStream(cublasHandle_t handle, cudaStream_t stream,
                                                const void* weight, const void* input, void* output,
                                                int rows, int ic, int icp, int oc, int ocp,
                                                int warmup, int repeat, float* avgMs) {
    if (avgMs == nullptr || repeat <= 0) {
        return false;
    }
    for (int i = 0; i < warmup; ++i) {
        if (!runFp16GemmProductionLayout(handle, weight, input, output, rows, ic, icp, oc, ocp)) {
            return false;
        }
    }
    if (!checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(gemm stream warmup)")) {
        return false;
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!checkCuda(cudaEventCreate(&start), "cudaEventCreate(gemm stream start)") ||
        !checkCuda(cudaEventCreate(&stop), "cudaEventCreate(gemm stream stop)")) {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        return false;
    }
    bool ok = checkCuda(cudaEventRecord(start, stream), "cudaEventRecord(gemm stream start)");
    for (int i = 0; ok && i < repeat; ++i) {
        ok = runFp16GemmProductionLayout(handle, weight, input, output, rows, ic, icp, oc, ocp);
    }
    ok = ok && checkCuda(cudaEventRecord(stop, stream), "cudaEventRecord(gemm stream stop)");
    ok = ok && checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(gemm stream stop)");
    float totalMs = 0.0f;
    ok = ok && checkCuda(cudaEventElapsedTime(&totalMs, start, stop), "cudaEventElapsedTime(gemm stream)");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (!ok) {
        return false;
    }
    *avgMs = totalMs / static_cast<float>(repeat);
    return true;
}

static bool measureQkvSequentialTriple(cublasHandle_t handle, cudaStream_t stream,
                                       const void* qWeight, const void* kWeight, const void* vWeight,
                                       const void* input, void* qOutput, void* kOutput, void* vOutput,
                                       int rows, int hidden, int hiddenPadded, int kv, int kvPadded,
                                       int warmup, int repeat, float* avgMs) {
    if (avgMs == nullptr || repeat <= 0) {
        return false;
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!checkCuda(cudaEventCreate(&start), "cudaEventCreate(qkv seq start)") ||
        !checkCuda(cudaEventCreate(&stop), "cudaEventCreate(qkv seq stop)")) {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        return false;
    }
    auto destroyEvents = [&]() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };
    auto runTriple = [&]() -> bool {
        return runFp16GemmProductionLayout(handle, qWeight, input, qOutput,
                                           rows, hidden, hiddenPadded, hidden, hiddenPadded) &&
               runFp16GemmProductionLayout(handle, kWeight, input, kOutput,
                                           rows, hidden, hiddenPadded, kv, kvPadded) &&
               runFp16GemmProductionLayout(handle, vWeight, input, vOutput,
                                           rows, hidden, hiddenPadded, kv, kvPadded);
    };
    for (int i = 0; i < warmup; ++i) {
        if (!runTriple()) {
            destroyEvents();
            return false;
        }
    }
    if (!checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(qkv seq warmup)")) {
        destroyEvents();
        return false;
    }
    float totalMs = 0.0f;
    for (int i = 0; i < repeat; ++i) {
        if (!checkCuda(cudaEventRecord(start, stream), "cudaEventRecord(qkv seq start)") ||
            !runTriple() ||
            !checkCuda(cudaEventRecord(stop, stream), "cudaEventRecord(qkv seq stop)") ||
            !checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(qkv seq stop)")) {
            destroyEvents();
            return false;
        }
        float iterMs = 0.0f;
        if (!checkCuda(cudaEventElapsedTime(&iterMs, start, stop), "cudaEventElapsedTime(qkv seq)")) {
            destroyEvents();
            return false;
        }
        totalMs += iterMs;
    }
    destroyEvents();
    *avgMs = totalMs / static_cast<float>(repeat);
    return true;
}

static bool measureQkvParallelTriple(cublasHandle_t qHandle, cublasHandle_t kHandle, cublasHandle_t vHandle,
                                     cudaStream_t qStream, cudaStream_t kStream, cudaStream_t vStream,
                                     cudaStream_t timingStream,
                                     const void* qWeight, const void* kWeight, const void* vWeight,
                                     const void* input, void* qOutput, void* kOutput, void* vOutput,
                                     int rows, int hidden, int hiddenPadded, int kv, int kvPadded,
                                     int warmup, int repeat, float* avgMs) {
    if (avgMs == nullptr || repeat <= 0) {
        return false;
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t qDone = nullptr;
    cudaEvent_t kDone = nullptr;
    cudaEvent_t vDone = nullptr;
    cudaEvent_t stop = nullptr;
    if (!checkCuda(cudaEventCreate(&start), "cudaEventCreate(qkv par start)") ||
        !checkCuda(cudaEventCreate(&qDone), "cudaEventCreate(qkv qDone)") ||
        !checkCuda(cudaEventCreate(&kDone), "cudaEventCreate(qkv kDone)") ||
        !checkCuda(cudaEventCreate(&vDone), "cudaEventCreate(qkv vDone)") ||
        !checkCuda(cudaEventCreate(&stop), "cudaEventCreate(qkv par stop)")) {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (qDone != nullptr) {
            cudaEventDestroy(qDone);
        }
        if (kDone != nullptr) {
            cudaEventDestroy(kDone);
        }
        if (vDone != nullptr) {
            cudaEventDestroy(vDone);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        return false;
    }
    auto destroyEvents = [&]() {
        cudaEventDestroy(start);
        cudaEventDestroy(qDone);
        cudaEventDestroy(kDone);
        cudaEventDestroy(vDone);
        cudaEventDestroy(stop);
    };
    auto runTriple = [&]() -> bool {
        if (!checkCuda(cudaEventRecord(start, timingStream), "cudaEventRecord(qkv par start)") ||
            !checkCuda(cudaStreamWaitEvent(qStream, start, 0), "cudaStreamWaitEvent(q start)") ||
            !checkCuda(cudaStreamWaitEvent(kStream, start, 0), "cudaStreamWaitEvent(k start)") ||
            !checkCuda(cudaStreamWaitEvent(vStream, start, 0), "cudaStreamWaitEvent(v start)")) {
            return false;
        }
        if (!runFp16GemmProductionLayout(qHandle, qWeight, input, qOutput,
                                         rows, hidden, hiddenPadded, hidden, hiddenPadded) ||
            !runFp16GemmProductionLayout(kHandle, kWeight, input, kOutput,
                                         rows, hidden, hiddenPadded, kv, kvPadded) ||
            !runFp16GemmProductionLayout(vHandle, vWeight, input, vOutput,
                                         rows, hidden, hiddenPadded, kv, kvPadded)) {
            return false;
        }
        return checkCuda(cudaEventRecord(qDone, qStream), "cudaEventRecord(q done)") &&
               checkCuda(cudaEventRecord(kDone, kStream), "cudaEventRecord(k done)") &&
               checkCuda(cudaEventRecord(vDone, vStream), "cudaEventRecord(v done)") &&
               checkCuda(cudaStreamWaitEvent(timingStream, qDone, 0), "cudaStreamWaitEvent(timing q done)") &&
               checkCuda(cudaStreamWaitEvent(timingStream, kDone, 0), "cudaStreamWaitEvent(timing k done)") &&
               checkCuda(cudaStreamWaitEvent(timingStream, vDone, 0), "cudaStreamWaitEvent(timing v done)");
    };
    for (int i = 0; i < warmup; ++i) {
        if (!runTriple() ||
            !checkCuda(cudaEventRecord(stop, timingStream), "cudaEventRecord(qkv warm stop)") ||
            !checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(qkv warm stop)")) {
            destroyEvents();
            return false;
        }
    }
    float totalMs = 0.0f;
    for (int i = 0; i < repeat; ++i) {
        if (!runTriple() ||
            !checkCuda(cudaEventRecord(stop, timingStream), "cudaEventRecord(qkv par stop)") ||
            !checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(qkv par stop)")) {
            destroyEvents();
            return false;
        }
        float iterMs = 0.0f;
        if (!checkCuda(cudaEventElapsedTime(&iterMs, start, stop), "cudaEventElapsedTime(qkv par)")) {
            destroyEvents();
            return false;
        }
        totalMs += iterMs;
    }
    destroyEvents();
    *avgMs = totalMs / static_cast<float>(repeat);
    return true;
}

static bool runQkvParallelGemmFloorCase(int rows, int hidden, int kv, int warmup, int repeat) {
    constexpr size_t elemBytes = 2;
    const int hiddenPadded = upDivInt(hidden, 8) * 8;
    const int kvPadded = upDivInt(kv, 8) * 8;
    const size_t qWeightBytes = static_cast<size_t>(hiddenPadded) * static_cast<size_t>(hiddenPadded) * elemBytes;
    const size_t kvWeightBytes = static_cast<size_t>(kvPadded) * static_cast<size_t>(hiddenPadded) * elemBytes;
    const size_t inputBytes = static_cast<size_t>(rows) * static_cast<size_t>(hiddenPadded) * elemBytes;
    const size_t qOutputBytes = static_cast<size_t>(rows) * static_cast<size_t>(hiddenPadded) * elemBytes;
    const size_t kvOutputBytes = static_cast<size_t>(rows) * static_cast<size_t>(kvPadded) * elemBytes;

    CudaDeviceBuffer qWeight;
    CudaDeviceBuffer kWeight;
    CudaDeviceBuffer vWeight;
    CudaDeviceBuffer input;
    CudaDeviceBuffer qOutput;
    CudaDeviceBuffer kOutput;
    CudaDeviceBuffer vOutput;
    if (!qWeight.alloc(qWeightBytes) ||
        !kWeight.alloc(kvWeightBytes) ||
        !vWeight.alloc(kvWeightBytes) ||
        !input.alloc(inputBytes) ||
        !qOutput.alloc(qOutputBytes) ||
        !kOutput.alloc(kvOutputBytes) ||
        !vOutput.alloc(kvOutputBytes)) {
        return false;
    }

    cudaStream_t seqStream = nullptr;
    cudaStream_t qStream = nullptr;
    cudaStream_t kStream = nullptr;
    cudaStream_t vStream = nullptr;
    cudaStream_t timingStream = nullptr;
    cublasHandle_t seqHandle = nullptr;
    cublasHandle_t qHandle = nullptr;
    cublasHandle_t kHandle = nullptr;
    cublasHandle_t vHandle = nullptr;
    auto cleanup = [&]() {
        if (seqHandle != nullptr) {
            cublasDestroy(seqHandle);
        }
        if (qHandle != nullptr) {
            cublasDestroy(qHandle);
        }
        if (kHandle != nullptr) {
            cublasDestroy(kHandle);
        }
        if (vHandle != nullptr) {
            cublasDestroy(vHandle);
        }
        if (seqStream != nullptr) {
            cudaStreamDestroy(seqStream);
        }
        if (qStream != nullptr) {
            cudaStreamDestroy(qStream);
        }
        if (kStream != nullptr) {
            cudaStreamDestroy(kStream);
        }
        if (vStream != nullptr) {
            cudaStreamDestroy(vStream);
        }
        if (timingStream != nullptr) {
            cudaStreamDestroy(timingStream);
        }
    };
    if (!checkCuda(cudaStreamCreateWithFlags(&seqStream, cudaStreamNonBlocking), "cudaStreamCreate(qkv seq)") ||
        !checkCuda(cudaStreamCreateWithFlags(&qStream, cudaStreamNonBlocking), "cudaStreamCreate(q)") ||
        !checkCuda(cudaStreamCreateWithFlags(&kStream, cudaStreamNonBlocking), "cudaStreamCreate(k)") ||
        !checkCuda(cudaStreamCreateWithFlags(&vStream, cudaStreamNonBlocking), "cudaStreamCreate(v)") ||
        !checkCuda(cudaStreamCreateWithFlags(&timingStream, cudaStreamNonBlocking), "cudaStreamCreate(qkv timing)") ||
        !createTensorOpCublasHandle(&seqHandle, seqStream, "cublasCreate(qkv seq)") ||
        !createTensorOpCublasHandle(&qHandle, qStream, "cublasCreate(q)") ||
        !createTensorOpCublasHandle(&kHandle, kStream, "cublasCreate(k)") ||
        !createTensorOpCublasHandle(&vHandle, vStream, "cublasCreate(v)")) {
        cleanup();
        return false;
    }

    float qMs = 0.0f;
    float kMs = 0.0f;
    float vMs = 0.0f;
    float sequentialMs = 0.0f;
    float parallelMs = 0.0f;
    if (!measureGemmProductionLayoutOnStream(seqHandle, seqStream,
                                             qWeight.get(), input.get(), qOutput.get(),
                                             rows, hidden, hiddenPadded, hidden, hiddenPadded,
                                             warmup, repeat, &qMs) ||
        !measureGemmProductionLayoutOnStream(seqHandle, seqStream,
                                             kWeight.get(), input.get(), kOutput.get(),
                                             rows, hidden, hiddenPadded, kv, kvPadded,
                                             warmup, repeat, &kMs) ||
        !measureGemmProductionLayoutOnStream(seqHandle, seqStream,
                                             vWeight.get(), input.get(), vOutput.get(),
                                             rows, hidden, hiddenPadded, kv, kvPadded,
                                             warmup, repeat, &vMs) ||
        !measureQkvSequentialTriple(seqHandle, seqStream,
                                    qWeight.get(), kWeight.get(), vWeight.get(), input.get(),
                                    qOutput.get(), kOutput.get(), vOutput.get(),
                                    rows, hidden, hiddenPadded, kv, kvPadded, warmup, repeat, &sequentialMs) ||
        !measureQkvParallelTriple(qHandle, kHandle, vHandle, qStream, kStream, vStream, timingStream,
                                  qWeight.get(), kWeight.get(), vWeight.get(), input.get(),
                                  qOutput.get(), kOutput.get(), vOutput.get(),
                                  rows, hidden, hiddenPadded, kv, kvPadded, warmup, repeat, &parallelMs)) {
        cleanup();
        return false;
    }
    cleanup();
    const float individualSumMs = qMs + kMs + vMs;
    const float delta = parallelMs - sequentialMs;
    const float speedup = parallelMs > 0.0f ? sequentialMs / parallelMs : 0.0f;
    MNN_PRINT("[bench_ops/cuda/perf/DecodeRepairQkvParallelGemmFloor] rows=%d hidden=%d kv=%d "
              "q=%.4f ms k=%.4f ms v=%.4f ms individual_sum=%.4f ms "
              "sequential_triple=%.4f ms parallel_triple=%.4f ms delta=%.4f ms speedup=%.3f\n",
              rows, hidden, kv, qMs, kMs, vMs, individualSumMs,
              sequentialMs, parallelMs, delta, speedup);
    ::fflush(stdout);
    return true;
}

static bool measureDownSplitNSequential(cublasHandle_t handle, cudaStream_t stream,
                                        const void* weight, const void* input, void* output,
                                        int rows, int ic, int icp, int oc, int ocp, int chunks,
                                        int warmup, int repeat, float* avgMs) {
    if (avgMs == nullptr || repeat <= 0 || chunks <= 0 || oc % chunks != 0) {
        return false;
    }
    const int ocChunk = oc / chunks;
    auto runChunks = [&]() -> bool {
        for (int c = 0; c < chunks; ++c) {
            if (!runFp16GemmProductionLayoutChunk(handle, weight, input, output,
                                                  rows, ic, icp, ocChunk, ocp, c * ocChunk)) {
                return false;
            }
        }
        return true;
    };
    for (int i = 0; i < warmup; ++i) {
        if (!runChunks()) {
            return false;
        }
    }
    if (!checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(down split seq warmup)")) {
        return false;
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!checkCuda(cudaEventCreate(&start), "cudaEventCreate(down split seq start)") ||
        !checkCuda(cudaEventCreate(&stop), "cudaEventCreate(down split seq stop)")) {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        return false;
    }
    bool ok = checkCuda(cudaEventRecord(start, stream), "cudaEventRecord(down split seq start)");
    for (int i = 0; ok && i < repeat; ++i) {
        ok = runChunks();
    }
    ok = ok && checkCuda(cudaEventRecord(stop, stream), "cudaEventRecord(down split seq stop)");
    ok = ok && checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(down split seq stop)");
    float totalMs = 0.0f;
    ok = ok && checkCuda(cudaEventElapsedTime(&totalMs, start, stop), "cudaEventElapsedTime(down split seq)");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (!ok) {
        return false;
    }
    *avgMs = totalMs / static_cast<float>(repeat);
    return true;
}

static bool measureDownSplitNParallel(const std::vector<cublasHandle_t>& handles,
                                      const std::vector<cudaStream_t>& streams,
                                      cudaStream_t timingStream,
                                      const void* weight, const void* input, void* output,
                                      int rows, int ic, int icp, int oc, int ocp, int chunks,
                                      int warmup, int repeat, float* avgMs) {
    if (avgMs == nullptr || repeat <= 0 || chunks <= 0 ||
        static_cast<int>(handles.size()) < chunks || static_cast<int>(streams.size()) < chunks ||
        oc % chunks != 0) {
        return false;
    }
    const int ocChunk = oc / chunks;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    std::vector<cudaEvent_t> done(chunks, nullptr);
    bool ok = checkCuda(cudaEventCreate(&start), "cudaEventCreate(down split par start)") &&
        checkCuda(cudaEventCreate(&stop), "cudaEventCreate(down split par stop)");
    for (int c = 0; ok && c < chunks; ++c) {
        ok = checkCuda(cudaEventCreate(&done[c]), "cudaEventCreate(down split par done)");
    }
    auto cleanupEvents = [&]() {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        for (auto event : done) {
            if (event != nullptr) {
                cudaEventDestroy(event);
            }
        }
    };
    if (!ok) {
        cleanupEvents();
        return false;
    }

    auto enqueueChunks = [&](int loopCount) -> bool {
        if (!checkCuda(cudaEventRecord(start, timingStream), "cudaEventRecord(down split par start)")) {
            return false;
        }
        for (int c = 0; c < chunks; ++c) {
            if (!checkCuda(cudaStreamWaitEvent(streams[c], start, 0), "cudaStreamWaitEvent(down split par start)")) {
                return false;
            }
        }
        for (int i = 0; i < loopCount; ++i) {
            for (int c = 0; c < chunks; ++c) {
                if (!runFp16GemmProductionLayoutChunk(handles[c], weight, input, output,
                                                      rows, ic, icp, ocChunk, ocp, c * ocChunk)) {
                    return false;
                }
            }
        }
        for (int c = 0; c < chunks; ++c) {
            if (!checkCuda(cudaEventRecord(done[c], streams[c]), "cudaEventRecord(down split par done)") ||
                !checkCuda(cudaStreamWaitEvent(timingStream, done[c], 0), "cudaStreamWaitEvent(down split par done)")) {
                return false;
            }
        }
        return true;
    };

    for (int i = 0; i < warmup; ++i) {
        if (!enqueueChunks(1) ||
            !checkCuda(cudaEventRecord(stop, timingStream), "cudaEventRecord(down split par warm stop)") ||
            !checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(down split par warm stop)")) {
            cleanupEvents();
            return false;
        }
    }
    if (!enqueueChunks(repeat) ||
        !checkCuda(cudaEventRecord(stop, timingStream), "cudaEventRecord(down split par stop)") ||
        !checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(down split par stop)")) {
        cleanupEvents();
        return false;
    }
    float totalMs = 0.0f;
    ok = checkCuda(cudaEventElapsedTime(&totalMs, start, stop), "cudaEventElapsedTime(down split par)");
    cleanupEvents();
    if (!ok) {
        return false;
    }
    *avgMs = totalMs / static_cast<float>(repeat);
    return true;
}

static bool runDownSplitNGemmFloorCase(int rows, int hidden, int inter, int chunks, int warmup, int repeat) {
    constexpr size_t elemBytes = 2;
    const int ic = inter;
    const int oc = hidden;
    const int icp = upDivInt(ic, 8) * 8;
    const int ocp = upDivInt(oc, 8) * 8;
    if (chunks <= 1 || oc % chunks != 0 || ((oc / chunks) % 8) != 0) {
        MNN_PRINT("[bench_ops/cuda/perf/DecodeRepairDownSplitNGemmFloor] rows=%d hidden=%d inter=%d "
                  "chunks=%d skipped invalid chunking\n", rows, hidden, inter, chunks);
        return true;
    }
    CudaDeviceBuffer weight;
    CudaDeviceBuffer input;
    CudaDeviceBuffer output;
    if (!weight.alloc(static_cast<size_t>(ocp) * static_cast<size_t>(icp) * elemBytes) ||
        !input.alloc(static_cast<size_t>(rows) * static_cast<size_t>(icp) * elemBytes) ||
        !output.alloc(static_cast<size_t>(rows) * static_cast<size_t>(ocp) * elemBytes)) {
        return false;
    }

    cudaStream_t seqStream = nullptr;
    cudaStream_t timingStream = nullptr;
    cublasHandle_t seqHandle = nullptr;
    std::vector<cudaStream_t> streams(chunks, nullptr);
    std::vector<cublasHandle_t> handles(chunks, nullptr);
    auto cleanup = [&]() {
        if (seqHandle != nullptr) {
            cublasDestroy(seqHandle);
        }
        for (auto handle : handles) {
            if (handle != nullptr) {
                cublasDestroy(handle);
            }
        }
        if (seqStream != nullptr) {
            cudaStreamDestroy(seqStream);
        }
        if (timingStream != nullptr) {
            cudaStreamDestroy(timingStream);
        }
        for (auto stream : streams) {
            if (stream != nullptr) {
                cudaStreamDestroy(stream);
            }
        }
    };
    if (!checkCuda(cudaStreamCreateWithFlags(&seqStream, cudaStreamNonBlocking), "cudaStreamCreate(down split seq)") ||
        !checkCuda(cudaStreamCreateWithFlags(&timingStream, cudaStreamNonBlocking), "cudaStreamCreate(down split timing)") ||
        !createTensorOpCublasHandle(&seqHandle, seqStream, "cublasCreate(down split seq)")) {
        cleanup();
        return false;
    }
    for (int c = 0; c < chunks; ++c) {
        char label[64];
        ::snprintf(label, sizeof(label), "cublasCreate(down split chunk %d)", c);
        if (!checkCuda(cudaStreamCreateWithFlags(&streams[c], cudaStreamNonBlocking), "cudaStreamCreate(down split chunk)") ||
            !createTensorOpCublasHandle(&handles[c], streams[c], label)) {
            cleanup();
            return false;
        }
    }

    float singleMs = 0.0f;
    float splitSeqMs = 0.0f;
    float splitParMs = 0.0f;
    if (!measureGemmProductionLayoutOnStream(seqHandle, seqStream,
                                             weight.get(), input.get(), output.get(),
                                             rows, ic, icp, oc, ocp, warmup, repeat, &singleMs) ||
        !measureDownSplitNSequential(seqHandle, seqStream,
                                     weight.get(), input.get(), output.get(),
                                     rows, ic, icp, oc, ocp, chunks, warmup, repeat, &splitSeqMs) ||
        !measureDownSplitNParallel(handles, streams, timingStream,
                                   weight.get(), input.get(), output.get(),
                                   rows, ic, icp, oc, ocp, chunks, warmup, repeat, &splitParMs)) {
        cleanup();
        return false;
    }
    cleanup();
    MNN_PRINT("[bench_ops/cuda/perf/DecodeRepairDownSplitNGemmFloor] rows=%d hidden=%d inter=%d "
              "chunks=%d chunk_oc=%d single=%.4f ms split_seq=%.4f ms split_parallel=%.4f ms "
              "parallel_delta=%.4f ms speedup=%.3f seq_delta=%.4f ms\n",
              rows, hidden, inter, chunks, oc / chunks,
              singleMs, splitSeqMs, splitParMs,
              splitParMs - singleMs, splitParMs > 0.0f ? singleMs / splitParMs : 0.0f,
              splitSeqMs - singleMs);
    ::fflush(stdout);
    return true;
}

class CudaPicFusedSiluDownAccuracy : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        if (precision != BackendConfig::Precision_Low && precision != BackendConfig::Precision_Normal) {
            MNN_PRINT("bench_ops/cuda/accuracy/PicFusedSiluDown expects precision=2(fp16) or 0(mix); got %d.\n",
                      precision);
            return false;
        }
        const int hidden = envInt("MNN_BENCH_MLP_HIDDEN", 3072);
        const int inter = envInt("MNN_BENCH_MLP_INTER", 8192);
        bool ok = true;
        for (int rows = 1; rows <= 8; ++rows) {
            if (enabledRow(rows)) {
                ok = runPicFusedSiluDownAccuracyCase(rows, hidden, inter) && ok;
            }
        }
        return ok;
    }
};

class CudaPicFusedSiluDownPerf : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        if (precision != BackendConfig::Precision_Low && precision != BackendConfig::Precision_Normal) {
            MNN_PRINT("bench_ops/cuda/perf/PicFusedSiluDown expects precision=2(fp16) or 0(mix); got %d.\n",
                      precision);
            return false;
        }
        if (!requireMemoryLow("bench_ops/cuda/perf/PicFusedSiluDown")) {
            return false;
        }
        const int warmup = envInt("MNN_BENCH_FUSED_MLP_WARMUP", envInt("MNN_BENCH_MLP_WARMUP", 20));
        const int repeat = envInt("MNN_BENCH_FUSED_MLP_REPEAT", envInt("MNN_BENCH_MLP_REPEAT", 80));
        const int hidden = envInt("MNN_BENCH_MLP_HIDDEN", 3072);
        const int inter = envInt("MNN_BENCH_MLP_INTER", 8192);
        bool ok = true;
        for (int rows = 1; rows <= 8; ++rows) {
            if (enabledRow(rows)) {
                ok = runPicFusedSiluDownPerfCase(rows, hidden, inter, warmup, repeat) && ok;
            }
        }
        return ok;
    }
};

class CudaWeightOnlyConvPerf : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        if (precision != BackendConfig::Precision_Low && precision != BackendConfig::Precision_Normal) {
            MNN_PRINT("bench_ops/cuda/perf/WeightOnlyConv expects precision=2(fp16) or 0(mix); got %d.\n",
                      precision);
            return false;
        }
        if (!requireMemoryLow("bench_ops/cuda/perf/WeightOnlyConv")) {
            return false;
        }
        std::vector<WeightOnlyConvCase> cases;
        const int warmup = envInt("MNN_BENCH_WEIGHT_ONLY_WARMUP", 20);
        const int repeat = envInt("MNN_BENCH_WEIGHT_ONLY_REPEAT", 100);
        const int hidden = envInt("MNN_BENCH_WEIGHT_ONLY_HIDDEN", 2048);
        const int inter = envInt("MNN_BENCH_WEIGHT_ONLY_INTER", 8192);
        const int kv = envInt("MNN_BENCH_WEIGHT_ONLY_KV", hidden == 3072 ? 1024 : 512);
        const int vocab = envInt("MNN_BENCH_WEIGHT_ONLY_VOCAB_OC", 128256);
        for (int rows = 1; rows <= 8; ++rows) {
            cases.push_back({"hidden_to_inter", rows, hidden, inter, 64, warmup, repeat});
            cases.push_back({"hidden_to_gateup_concat", rows, hidden, inter * 2, 64, warmup, repeat});
            cases.push_back({"inter_to_hidden", rows, inter, hidden, 64, warmup, repeat});
            cases.push_back({"hidden_to_hidden", rows, hidden, hidden, 64, warmup, repeat});
            cases.push_back({"hidden_to_kv", rows, hidden, kv, 64, warmup, repeat});
            cases.push_back({"hidden_to_qkv_concat", rows, hidden, hidden + 2 * kv, 64, warmup, repeat});
        }
        if (::getenv("MNN_BENCH_WEIGHT_ONLY_VOCAB") != nullptr) {
            for (int rows = 1; rows <= 2; ++rows) {
                cases.push_back({"hidden_to_vocab", rows, hidden, vocab, 64, std::max(1, warmup / 2), std::max(1, repeat / 2)});
            }
        }

        bool ok = true;
        for (const auto& c : cases) {
            if (enabledByFilter(c.name) && enabledRow(c.rows)) {
                ok = runWeightOnlyConvCase(c) && ok;
            }
        }
        return ok;
    }
};

static bool runRows45CublasAccuracyCase(const char* name, int rows, int ic, int oc, const char* policy) {
    constexpr int quantBlock = 64;
    ScopedEnvVar restoreRows45Policy("MNN_CUDA_PIC_INT4_ROWS45_CUBLAS");
    ScopedEnvVar restoreRows48LtPolicy("MNN_CUDA_PIC_INT4_ROWS48_CUBLASLT");
    KVMeta meta;
    meta.pic_decode_repair_sparse_active = true;
    DirectOpBench bench(MNN_FORWARD_CUDA, &meta);
    if (!bench.valid()) {
        return false;
    }

    auto input = bench.tensor({rows, ic, 1, 1}, Tensor::CAFFE, false);
    auto refOutput = bench.tensor({rows, oc, 1, 1}, Tensor::CAFFE, false);
    auto optOutput = bench.tensor({rows, oc, 1, 1}, Tensor::CAFFE, false);
    if (!input || !refOutput || !optOutput || !writeHalfPattern(input)) {
        return false;
    }

    auto op = makeWeightOnlyLinearConvOp(ic, oc, quantBlock);
    std::vector<Tensor*> refInputs = {input};
    std::vector<Tensor*> refOutputs = {refOutput};
    std::vector<Tensor*> optInputs = {input};
    std::vector<Tensor*> optOutputs = {optOutput};
    auto refExe = bench.create(refInputs, refOutputs, op->get());
    auto optExe = bench.create(optInputs, optOutputs, op->get());
    if (!refExe || !optExe) {
        MNN_ERROR("Rows45Cublas failed to create execution for %s rows=%d\n", name, rows);
        return false;
    }
    if (bench.resize(refExe.get(), refInputs, refOutputs) != NO_ERROR ||
        bench.resize(optExe.get(), optInputs, optOutputs) != NO_ERROR) {
        MNN_ERROR("Rows45Cublas resize failed for %s rows=%d\n", name, rows);
        return false;
    }

    ::setenv("MNN_CUDA_PIC_INT4_ROWS45_CUBLAS", "0", 1);
    ::setenv("MNN_CUDA_PIC_INT4_ROWS48_CUBLASLT", "0", 1);
    if (bench.execute(refExe.get(), refInputs, refOutputs) != NO_ERROR) {
        return false;
    }
    ::setenv("MNN_CUDA_PIC_INT4_ROWS45_CUBLAS", policy, 1);
    ::setenv("MNN_CUDA_PIC_INT4_ROWS48_CUBLASLT", "1", 1);
    if (bench.execute(optExe.get(), optInputs, optOutputs) != NO_ERROR) {
        return false;
    }

    std::vector<float> ref;
    std::vector<float> opt;
    if (!readHalfTensor(refOutput, &ref) || !readHalfTensor(optOutput, &opt)) {
        return false;
    }
    char caseName[128];
    ::snprintf(caseName, sizeof(caseName), "%s_rows%d_policy_%s", name, rows, policy);
    return compareHalfOutputs(caseName, opt, ref, 0.08f, 0.08f);
}

static bool runPicLinearNhwcAccuracyCase(const char* name, int rows, int ic, int oc) {
    constexpr int quantBlock = 64;
    KVMeta meta;
    meta.pic_decode_repair_sparse_active = true;
    DirectOpBench bench(MNN_FORWARD_CUDA, &meta);
    if (!bench.valid()) {
        return false;
    }

    PicExternalPlan linearPlan;
    const std::string externalPath = linearExternalWeightPath(ic, oc, quantBlock);
    if (!buildLinearExternalWeight(externalPath, ic, oc, quantBlock, &linearPlan)) {
        return false;
    }

    auto inputNchw = bench.tensor({rows, ic, 1, 1}, Tensor::CAFFE, false);
    auto inputNhwc = bench.tensor({rows, 1, 1, ic}, Tensor::TENSORFLOW, false);
    auto refOutput = bench.tensor({rows, oc, 1, 1}, Tensor::CAFFE, false);
    auto optOutput = bench.tensor({rows, 1, 1, oc}, Tensor::TENSORFLOW, false);
    if (!inputNchw || !inputNhwc || !refOutput || !optOutput ||
        !writeHalfPattern(inputNchw) || !writeHalfPattern(inputNhwc)) {
        return false;
    }

    auto refOp = makeExternalWeightOnlyLinearConvOp("bench_pic_linear_nhwc_ref",
                                                    externalPath, ic, oc, linearPlan);
    auto optOp = makePicLinearNhwcWeightOnlyExtraOp("bench_pic_linear_nhwc_opt",
                                                    externalPath, ic, oc, linearPlan);
    std::vector<Tensor*> refInputs = {inputNchw};
    std::vector<Tensor*> refOutputs = {refOutput};
    std::vector<Tensor*> optInputs = {inputNhwc};
    std::vector<Tensor*> optOutputs = {optOutput};
    auto refExe = bench.create(refInputs, refOutputs, refOp->get());
    auto optExe = bench.create(optInputs, optOutputs, optOp->get());
    if (!refExe || !optExe) {
        MNN_ERROR("PicLinearNhwc failed to create execution for %s rows=%d\n", name, rows);
        return false;
    }
    if (bench.resize(refExe.get(), refInputs, refOutputs) != NO_ERROR ||
        bench.resize(optExe.get(), optInputs, optOutputs) != NO_ERROR) {
        MNN_ERROR("PicLinearNhwc resize failed for %s rows=%d\n", name, rows);
        return false;
    }
    if (bench.execute(refExe.get(), refInputs, refOutputs) != NO_ERROR ||
        bench.execute(optExe.get(), optInputs, optOutputs) != NO_ERROR) {
        return false;
    }

    std::vector<float> ref;
    std::vector<float> opt;
    if (!readHalfTensor(refOutput, &ref) || !readHalfTensor(optOutput, &opt)) {
        return false;
    }
    char caseName[128];
    ::snprintf(caseName, sizeof(caseName), "pic_linear_nhwc_%s_rows%d", name, rows);
    return compareHalfOutputs(caseName, opt, ref, 0.08f, 0.08f);
}

class CudaRows45CublasAccuracy : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        if (precision != BackendConfig::Precision_Low) {
            MNN_PRINT("bench_ops/cuda/accuracy/Rows45Cublas expects precision=2(fp16); got %d.\n", precision);
            return false;
        }
        if (!requireMemoryLow("bench_ops/cuda/accuracy/Rows45Cublas")) {
            return false;
        }
        bool ok = true;
        for (int rows = 4; rows <= 5; ++rows) {
            ok = runRows45CublasAccuracyCase("gate", rows, 2048, 8192, "all") && ok;
            ok = runRows45CublasAccuracyCase("up", rows, 2048, 8192, "all") && ok;
            ok = runRows45CublasAccuracyCase("down", rows, 8192, 2048, "down") && ok;
            ok = runRows45CublasAccuracyCase("down_all", rows, 8192, 2048, "all") && ok;
        }
        for (int rows : {4, 6, 8}) {
            ok = runRows45CublasAccuracyCase("llama32_3b_gate", rows, 3072, 8192, "all") && ok;
            ok = runRows45CublasAccuracyCase("llama32_3b_up", rows, 3072, 8192, "all") && ok;
            ok = runRows45CublasAccuracyCase("llama32_3b_down", rows, 8192, 3072, "all") && ok;
        }
        for (int rows : {1, 2, 4, 6, 8}) {
            ok = runPicLinearNhwcAccuracyCase("llama32_3b_gate", rows, 3072, 8192) && ok;
            ok = runPicLinearNhwcAccuracyCase("llama32_3b_down", rows, 8192, 3072) && ok;
        }
        return ok;
    }
};

static bool runDecodeRepairMlpGemmFloorCase(cublasHandle_t handle, int rows, int hidden, int inter,
                                            int warmup, int repeat) {
    constexpr size_t elemBytes = 2;

    CudaDeviceBuffer input;
    CudaDeviceBuffer gate;
    CudaDeviceBuffer up;
    CudaDeviceBuffer output;
    CudaDeviceBuffer gateWeight;
    CudaDeviceBuffer upWeight;
    CudaDeviceBuffer downWeight;
    if (!input.alloc(static_cast<size_t>(hidden) * rows * elemBytes) ||
        !gate.alloc(static_cast<size_t>(inter) * rows * elemBytes) ||
        !up.alloc(static_cast<size_t>(inter) * rows * elemBytes) ||
        !output.alloc(static_cast<size_t>(hidden) * rows * elemBytes) ||
        !gateWeight.alloc(static_cast<size_t>(inter) * hidden * elemBytes) ||
        !upWeight.alloc(static_cast<size_t>(inter) * hidden * elemBytes) ||
        !downWeight.alloc(static_cast<size_t>(hidden) * inter * elemBytes)) {
        return false;
    }

    float gateMs = 0.0f;
    float upMs = 0.0f;
    float downMs = 0.0f;
    float chainMs = 0.0f;
    float gateupConcatMs = 0.0f;
    float gateRowsMs = 0.0f;
    float upRowsMs = 0.0f;
    float downRowsMs = 0.0f;
    float chainRowsMs = 0.0f;
    float gateupConcatRowsMs = 0.0f;
    GemmFloorCase gateCase = {"gate_fp16_gemm", rows, hidden, inter, warmup, repeat};
    GemmFloorCase upCase = {"up_fp16_gemm", rows, hidden, inter, warmup, repeat};
    GemmFloorCase downCase = {"down_fp16_gemm", rows, inter, hidden, warmup, repeat};
    GemmFloorCase gateupCase = {"gateup_concat_fp16", rows, hidden, inter * 2, warmup, repeat};
    if (!runGemmFloorCase(handle, gateCase, &gateMs) ||
        !runGemmFloorCase(handle, upCase, &upMs) ||
        !runGemmFloorCase(handle, downCase, &downMs) ||
        !runGemmFloorCase(handle, gateupCase, &gateupConcatMs)) {
        return false;
    }
    if (!runGemmFloorCaseRowsByOc(handle, gateCase, &gateRowsMs) ||
        !runGemmFloorCaseRowsByOc(handle, upCase, &upRowsMs) ||
        !runGemmFloorCaseRowsByOc(handle, downCase, &downRowsMs) ||
        !runGemmFloorCaseRowsByOc(handle, gateupCase, &gateupConcatRowsMs)) {
        return false;
    }

    if (!measureCudaBool([&]() {
            return runFp16Gemm(handle, gateWeight.get(), input.get(), gate.get(), rows, hidden, inter) &&
                   runFp16Gemm(handle, upWeight.get(), input.get(), up.get(), rows, hidden, inter) &&
                   runFp16Gemm(handle, downWeight.get(), gate.get(), output.get(), rows, inter, hidden);
        }, warmup, repeat, &chainMs)) {
        return false;
    }
    if (!measureCudaBool([&]() {
            return runFp16GemmRowsByOc(handle, gateWeight.get(), input.get(), gate.get(), rows, hidden, inter) &&
                   runFp16GemmRowsByOc(handle, upWeight.get(), input.get(), up.get(), rows, hidden, inter) &&
                   runFp16GemmRowsByOc(handle, downWeight.get(), gate.get(), output.get(), rows, inter, hidden);
        }, warmup, repeat, &chainRowsMs)) {
        return false;
    }

    MNN_PRINT("[bench_ops/cuda/perf/DecodeRepairMlpGemmFloor] rows=%d layout=oc_by_rows hidden=%d inter=%d "
              "gate=%.4f ms up=%.4f ms down=%.4f ms gateup_concat=%.4f ms "
              "projection_sum=%.4f ms chain_no_silu=%.4f ms\n",
              rows, hidden, inter, gateMs, upMs, downMs, gateupConcatMs,
              gateMs + upMs + downMs, chainMs);
    MNN_PRINT("[bench_ops/cuda/perf/DecodeRepairMlpGemmFloor] rows=%d layout=rows_by_oc hidden=%d inter=%d "
              "gate=%.4f ms up=%.4f ms down=%.4f ms gateup_concat=%.4f ms "
              "projection_sum=%.4f ms chain_no_silu=%.4f ms\n",
              rows, hidden, inter, gateRowsMs, upRowsMs, downRowsMs, gateupConcatRowsMs,
              gateRowsMs + upRowsMs + downRowsMs, chainRowsMs);
    ::fflush(stdout);
    return true;
}

class CudaDecodeRepairMlpGemmFloorPerf : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        if (precision != BackendConfig::Precision_Low && precision != BackendConfig::Precision_Normal) {
            MNN_PRINT("bench_ops/cuda/perf/DecodeRepairMlpGemmFloor expects precision=2(fp16) or 0(mix); got %d.\n",
                      precision);
            return false;
        }
        cublasHandle_t handle = nullptr;
        if (!checkCublas(cublasCreate(&handle), "cublasCreate")) {
            return false;
        }
#if CUDART_VERSION >= 9000
        checkCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode(TENSOR_OP)");
#endif
        const int warmup = envInt("MNN_BENCH_MLP_GEMM_WARMUP", envInt("MNN_BENCH_MLP_WARMUP", 20));
        const int repeat = envInt("MNN_BENCH_MLP_GEMM_REPEAT", envInt("MNN_BENCH_MLP_REPEAT", 80));
        const int hidden = envInt("MNN_BENCH_MLP_HIDDEN", 2048);
        const int inter = envInt("MNN_BENCH_MLP_INTER", 8192);
        bool ok = true;
        for (int rows = 1; rows <= 8; ++rows) {
            if (enabledRow(rows)) {
                ok = runDecodeRepairMlpGemmFloorCase(handle, rows, hidden, inter, warmup, repeat) && ok;
            }
        }
        cublasDestroy(handle);
        return ok;
    }
};

static bool runDecodeRepairMlpTiledGemmFloorCase(cublasHandle_t handle, int rows, int hidden, int inter,
                                                 int tile, int warmup, int repeat) {
    constexpr size_t elemBytes = 2;
    const int hiddenP = upDivInt(hidden, 8) * 8;
    const int interP = upDivInt(inter, 8) * 8;
    const int fullGateupOcp = upDivInt(inter * 2, 8) * 8;
    tile = std::max(1, std::min(tile, inter));
    if (inter % tile != 0) {
        MNN_ERROR("DecodeRepairMlpTiledGemmFloor requires tile to divide inter; inter=%d tile=%d\n", inter, tile);
        return false;
    }
    const int tileCount = inter / tile;
    const int tileP = upDivInt(tile, 8) * 8;
    const int gateupTileOcp = upDivInt(tile * 2, 8) * 8;

    CudaDeviceBuffer input;
    CudaDeviceBuffer fullGateupWeight;
    CudaDeviceBuffer fullGateupOutput;
    CudaDeviceBuffer fullActivation;
    CudaDeviceBuffer fullDownWeight;
    CudaDeviceBuffer output;
    CudaDeviceBuffer gateupTileWeight;
    CudaDeviceBuffer gateupTileOutput;
    CudaDeviceBuffer activationTile;
    CudaDeviceBuffer downTileWeight;
    if (!input.alloc(static_cast<size_t>(rows) * hiddenP * elemBytes) ||
        !fullGateupWeight.alloc(static_cast<size_t>(fullGateupOcp) * hiddenP * elemBytes) ||
        !fullGateupOutput.alloc(static_cast<size_t>(rows) * fullGateupOcp * elemBytes) ||
        !fullActivation.alloc(static_cast<size_t>(rows) * interP * elemBytes) ||
        !fullDownWeight.alloc(static_cast<size_t>(hiddenP) * interP * elemBytes) ||
        !output.alloc(static_cast<size_t>(rows) * hiddenP * elemBytes) ||
        !gateupTileWeight.alloc(static_cast<size_t>(gateupTileOcp) * hiddenP * elemBytes) ||
        !gateupTileOutput.alloc(static_cast<size_t>(rows) * gateupTileOcp * elemBytes) ||
        !activationTile.alloc(static_cast<size_t>(rows) * tileP * elemBytes) ||
        !downTileWeight.alloc(static_cast<size_t>(hiddenP) * tileP * elemBytes)) {
        return false;
    }

    auto runFullGateup = [&]() -> bool {
        return runFp16GemmProductionLayout(handle, fullGateupWeight.get(), input.get(), fullGateupOutput.get(),
                                           rows, hidden, hiddenP, inter * 2, fullGateupOcp);
    };
    auto runFullDown = [&]() -> bool {
        return runFp16GemmProductionLayout(handle, fullDownWeight.get(), fullActivation.get(), output.get(),
                                           rows, inter, interP, hidden, hiddenP);
    };
    auto runGateupTiles = [&]() -> bool {
        for (int i = 0; i < tileCount; ++i) {
            if (!runFp16GemmProductionLayout(handle, gateupTileWeight.get(), input.get(), gateupTileOutput.get(),
                                             rows, hidden, hiddenP, tile * 2, gateupTileOcp)) {
                return false;
            }
        }
        return true;
    };
    auto runDownTiles = [&]() -> bool {
        for (int i = 0; i < tileCount; ++i) {
            const float beta = i == 0 ? 0.0f : 1.0f;
            if (!runFp16GemmProductionLayoutBeta(handle, downTileWeight.get(), activationTile.get(), output.get(),
                                                 rows, tile, tileP, hidden, hiddenP, beta)) {
                return false;
            }
        }
        return true;
    };
    auto runTiledChain = [&]() -> bool {
        for (int i = 0; i < tileCount; ++i) {
            if (!runFp16GemmProductionLayout(handle, gateupTileWeight.get(), input.get(), gateupTileOutput.get(),
                                             rows, hidden, hiddenP, tile * 2, gateupTileOcp)) {
                return false;
            }
            const float beta = i == 0 ? 0.0f : 1.0f;
            if (!runFp16GemmProductionLayoutBeta(handle, downTileWeight.get(), activationTile.get(), output.get(),
                                                 rows, tile, tileP, hidden, hiddenP, beta)) {
                return false;
            }
        }
        return true;
    };

    float fullGateupMs = 0.0f;
    float fullDownMs = 0.0f;
    float fullChainMs = 0.0f;
    float tiledGateupMs = 0.0f;
    float tiledDownMs = 0.0f;
    float tiledChainMs = 0.0f;
    if (!measureCudaBool(runFullGateup, warmup, repeat, &fullGateupMs) ||
        !measureCudaBool(runFullDown, warmup, repeat, &fullDownMs) ||
        !measureCudaBool([&]() { return runFullGateup() && runFullDown(); }, warmup, repeat, &fullChainMs) ||
        !measureCudaBool(runGateupTiles, warmup, repeat, &tiledGateupMs) ||
        !measureCudaBool(runDownTiles, warmup, repeat, &tiledDownMs) ||
        !measureCudaBool(runTiledChain, warmup, repeat, &tiledChainMs)) {
        return false;
    }

    const float tiledVsFull = tiledChainMs - fullChainMs;
    const float downVsFull = tiledDownMs - fullDownMs;
    MNN_PRINT("[bench_ops/cuda/perf/DecodeRepairMlpTiledGemmFloor] rows=%d hidden=%d inter=%d tile=%d tiles=%d "
              "full_gateup=%.4f ms full_down=%.4f ms full_chain_no_silu=%.4f ms "
              "tiled_gateup=%.4f ms tiled_down_accum=%.4f ms tiled_chain_lower_bound=%.4f ms "
              "tiled_vs_full=%.4f ms tiled_down_vs_full_down=%.4f ms\n",
              rows, hidden, inter, tile, tileCount,
              fullGateupMs, fullDownMs, fullChainMs,
              tiledGateupMs, tiledDownMs, tiledChainMs,
              tiledVsFull, downVsFull);
    ::fflush(stdout);
    return true;
}

class CudaDecodeRepairMlpTiledGemmFloorPerf : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        if (precision != BackendConfig::Precision_Low && precision != BackendConfig::Precision_Normal) {
            MNN_PRINT("bench_ops/cuda/perf/DecodeRepairMlpTiledGemmFloor expects precision=2(fp16) or 0(mix); got %d.\n",
                      precision);
            return false;
        }
        cublasHandle_t handle = nullptr;
        if (!checkCublas(cublasCreate(&handle), "cublasCreate")) {
            return false;
        }
#if CUDART_VERSION >= 9000
        checkCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode(TENSOR_OP)");
#endif
        const int warmup = envInt("MNN_BENCH_MLP_GEMM_WARMUP", envInt("MNN_BENCH_MLP_WARMUP", 20));
        const int repeat = envInt("MNN_BENCH_MLP_GEMM_REPEAT", envInt("MNN_BENCH_MLP_REPEAT", 80));
        const int hidden = envInt("MNN_BENCH_MLP_HIDDEN", 2048);
        const int inter = envInt("MNN_BENCH_MLP_INTER", 8192);
        const int tile = envInt("MNN_BENCH_MLP_TILE", 1024);
        bool ok = true;
        for (int rows = 1; rows <= 8; ++rows) {
            if (enabledRow(rows)) {
                ok = runDecodeRepairMlpTiledGemmFloorCase(handle, rows, hidden, inter, tile, warmup, repeat) && ok;
            }
        }
        cublasDestroy(handle);
        return ok;
    }
};

static bool runDecodeRepairDownTransposedLayoutFloorCase(cublasHandle_t handle, int rows, int hidden, int inter,
                                                         int warmup, int repeat) {
    constexpr size_t elemBytes = 2;
    const int ic = inter;
    const int oc = hidden;
    const int icp = upDivInt(ic, 8) * 8;
    const int ocp = upDivInt(oc, 8) * 8;
    CudaDeviceBuffer input;
    CudaDeviceBuffer productionWeight;
    CudaDeviceBuffer transposedWeight;
    CudaDeviceBuffer output;
    if (!input.alloc(static_cast<size_t>(rows) * icp * elemBytes) ||
        !productionWeight.alloc(static_cast<size_t>(ocp) * icp * elemBytes) ||
        !transposedWeight.alloc(static_cast<size_t>(icp) * ocp * elemBytes) ||
        !output.alloc(static_cast<size_t>(rows) * ocp * elemBytes)) {
        return false;
    }

    float productionMs = 0.0f;
    float transposedMs = 0.0f;
    float transposedFast16Ms = 0.0f;
    float transposed16fMs = 0.0f;
    if (!measureCudaBool([&]() {
            return runFp16GemmProductionLayout(handle, productionWeight.get(), input.get(), output.get(),
                                               rows, ic, icp, oc, ocp);
        }, warmup, repeat, &productionMs)) {
        return false;
    }
    if (!measureCudaBool([&]() {
            return runFp16GemmTransposedWeightLayout(handle, transposedWeight.get(), input.get(), output.get(),
                                                     rows, ic, icp, oc, ocp, 0);
        }, warmup, repeat, &transposedMs)) {
        return false;
    }
    if (!measureCudaBool([&]() {
            return runFp16GemmTransposedWeightLayout(handle, transposedWeight.get(), input.get(), output.get(),
                                                     rows, ic, icp, oc, ocp, 1);
        }, warmup, repeat, &transposedFast16Ms)) {
        return false;
    }
    if (!measureCudaBool([&]() {
            return runFp16GemmTransposedWeightLayout(handle, transposedWeight.get(), input.get(), output.get(),
                                                     rows, ic, icp, oc, ocp, 2);
        }, warmup, repeat, &transposed16fMs)) {
        return false;
    }
    MNN_PRINT("[bench_ops/cuda/perf/DecodeRepairDownTransposedLayoutFloor] rows=%d hidden=%d inter=%d "
              "production_nt=%.4f ms transposed_nn=%.4f ms transposed_nn_fast16=%.4f ms transposed_nn_16f=%.4f ms "
              "transpose_delta=%.4f ms transpose_fast16_delta=%.4f ms transpose16f_delta=%.4f ms\n",
              rows, hidden, inter,
              productionMs, transposedMs, transposedFast16Ms, transposed16fMs,
              transposedMs - productionMs, transposedFast16Ms - productionMs, transposed16fMs - productionMs);
    ::fflush(stdout);
    return true;
}

class CudaDecodeRepairDownTransposedLayoutFloorPerf : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        if (precision != BackendConfig::Precision_Low && precision != BackendConfig::Precision_Normal) {
            MNN_PRINT("bench_ops/cuda/perf/DecodeRepairDownTransposedLayoutFloor expects precision=2(fp16) or 0(mix); got %d.\n",
                      precision);
            return false;
        }
        cublasHandle_t handle = nullptr;
        if (!checkCublas(cublasCreate(&handle), "cublasCreate")) {
            return false;
        }
#if CUDART_VERSION >= 9000
        checkCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode(TENSOR_OP)");
#endif
        const int warmup = envInt("MNN_BENCH_MLP_GEMM_WARMUP", envInt("MNN_BENCH_MLP_WARMUP", 30));
        const int repeat = envInt("MNN_BENCH_MLP_GEMM_REPEAT", envInt("MNN_BENCH_MLP_REPEAT", 120));
        const int hidden = envInt("MNN_BENCH_MLP_HIDDEN", 3072);
        const int inter = envInt("MNN_BENCH_MLP_INTER", 8192);
        bool ok = true;
        for (int rows = 1; rows <= 8; ++rows) {
            if (enabledRow(rows)) {
                ok = runDecodeRepairDownTransposedLayoutFloorCase(handle, rows, hidden, inter, warmup, repeat) && ok;
            }
        }
        cublasDestroy(handle);
        return ok;
    }
};

class CudaDecodeRepairDownSplitNGemmFloorPerf : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        if (precision != BackendConfig::Precision_Low && precision != BackendConfig::Precision_Normal) {
            MNN_PRINT("bench_ops/cuda/perf/DecodeRepairDownSplitNGemmFloor expects precision=2(fp16) or 0(mix); got %d.\n",
                      precision);
            return false;
        }
        const int warmup = envInt("MNN_BENCH_MLP_GEMM_WARMUP", envInt("MNN_BENCH_MLP_WARMUP", 30));
        const int repeat = envInt("MNN_BENCH_MLP_GEMM_REPEAT", envInt("MNN_BENCH_MLP_REPEAT", 120));
        const int hidden = envInt("MNN_BENCH_MLP_HIDDEN", 3072);
        const int inter = envInt("MNN_BENCH_MLP_INTER", 8192);
        std::vector<int> chunks = {2, 3, 4, 6, 8};
        const char* chunkEnv = ::getenv("MNN_BENCH_DOWN_NCHUNKS");
        if (chunkEnv != nullptr && chunkEnv[0] != '\0') {
            chunks.assign(1, envInt("MNN_BENCH_DOWN_NCHUNKS", 2));
        }
        bool ok = true;
        for (int rows = 1; rows <= 8; ++rows) {
            if (!enabledRow(rows)) {
                continue;
            }
            for (int chunkCount : chunks) {
                ok = runDownSplitNGemmFloorCase(rows, hidden, inter, chunkCount, warmup, repeat) && ok;
            }
        }
        return ok;
    }
};

class CudaDecodeRepairGateUpBatchedGemmFloorPerf : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        if (precision != BackendConfig::Precision_Low && precision != BackendConfig::Precision_Normal) {
            MNN_PRINT("bench_ops/cuda/perf/DecodeRepairGateUpBatchedGemmFloor expects precision=2(fp16) or 0(mix); got %d.\n",
                      precision);
            return false;
        }
        cublasHandle_t handle = nullptr;
        if (!checkCublas(cublasCreate(&handle), "cublasCreate")) {
            return false;
        }
#if CUDART_VERSION >= 9000
        checkCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode(TENSOR_OP)");
#endif
        const int warmup = envInt("MNN_BENCH_MLP_GEMM_WARMUP", envInt("MNN_BENCH_WEIGHT_ONLY_WARMUP", 20));
        const int repeat = envInt("MNN_BENCH_MLP_GEMM_REPEAT", envInt("MNN_BENCH_WEIGHT_ONLY_REPEAT", 80));
        const int hidden = envInt("MNN_BENCH_MLP_HIDDEN", envInt("MNN_BENCH_WEIGHT_ONLY_HIDDEN", 2048));
        const int inter = envInt("MNN_BENCH_MLP_INTER", envInt("MNN_BENCH_WEIGHT_ONLY_INTER", 8192));
        bool ok = true;
        for (int rows = 1; rows <= 8; ++rows) {
            if (enabledRow(rows)) {
                ok = runGateUpBatchedGemmFloorCase(handle, rows, hidden, inter, warmup, repeat) && ok;
            }
        }
        cublasDestroy(handle);
        return ok;
    }
};

class CudaDecodeRepairGateUpParallelGemmFloorPerf : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        if (precision != BackendConfig::Precision_Low && precision != BackendConfig::Precision_Normal) {
            MNN_PRINT("bench_ops/cuda/perf/DecodeRepairGateUpParallelGemmFloor expects precision=2(fp16) or 0(mix); got %d.\n",
                      precision);
            return false;
        }
        const int warmup = envInt("MNN_BENCH_MLP_GEMM_WARMUP", envInt("MNN_BENCH_WEIGHT_ONLY_WARMUP", 20));
        const int repeat = envInt("MNN_BENCH_MLP_GEMM_REPEAT", envInt("MNN_BENCH_WEIGHT_ONLY_REPEAT", 80));
        const int hidden = envInt("MNN_BENCH_MLP_HIDDEN", envInt("MNN_BENCH_WEIGHT_ONLY_HIDDEN", 2048));
        const int inter = envInt("MNN_BENCH_MLP_INTER", envInt("MNN_BENCH_WEIGHT_ONLY_INTER", 8192));
        bool ok = true;
        for (int rows = 1; rows <= 8; ++rows) {
            if (enabledRow(rows)) {
                ok = runGateUpParallelGemmFloorCase(rows, hidden, inter, warmup, repeat) && ok;
            }
        }
        return ok;
    }
};

class CudaDecodeRepairQkvParallelGemmFloorPerf : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        if (precision != BackendConfig::Precision_Low && precision != BackendConfig::Precision_Normal) {
            MNN_PRINT("bench_ops/cuda/perf/DecodeRepairQkvParallelGemmFloor expects precision=2(fp16) or 0(mix); got %d.\n",
                      precision);
            return false;
        }
        const int warmup = envInt("MNN_BENCH_MLP_GEMM_WARMUP", envInt("MNN_BENCH_WEIGHT_ONLY_WARMUP", 20));
        const int repeat = envInt("MNN_BENCH_MLP_GEMM_REPEAT", envInt("MNN_BENCH_WEIGHT_ONLY_REPEAT", 80));
        const int hidden = envInt("MNN_BENCH_MLP_HIDDEN", envInt("MNN_BENCH_WEIGHT_ONLY_HIDDEN", 3072));
        const int kv = envInt("MNN_BENCH_ATTN_KV", envInt("MNN_BENCH_WEIGHT_ONLY_KV", hidden == 3072 ? 1024 : 512));
        bool ok = true;
        for (int rows = 1; rows <= 8; ++rows) {
            if (enabledRow(rows)) {
                ok = runQkvParallelGemmFloorCase(rows, hidden, kv, warmup, repeat) && ok;
            }
        }
        return ok;
    }
};

#if MNN_BENCH_HAS_CUBLASLT
class CudaDecodeRepairMlpGemmLtFloorPerf : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        if (precision != BackendConfig::Precision_Low && precision != BackendConfig::Precision_Normal) {
            MNN_PRINT("bench_ops/cuda/perf/DecodeRepairMlpGemmLtFloor expects precision=2(fp16) or 0(mix); got %d.\n",
                      precision);
            return false;
        }
        cublasLtHandle_t handle = nullptr;
        if (!checkCublasLt(cublasLtCreate(&handle), "cublasLtCreate")) {
            return false;
        }
        const int warmup = envInt("MNN_BENCH_MLP_GEMM_WARMUP", envInt("MNN_BENCH_MLP_WARMUP", 20));
        const int repeat = envInt("MNN_BENCH_MLP_GEMM_REPEAT", envInt("MNN_BENCH_MLP_REPEAT", 80));
        const int hidden = envInt("MNN_BENCH_MLP_HIDDEN", 2048);
        const int inter = envInt("MNN_BENCH_MLP_INTER", 8192);
        bool ok = true;
        for (int rows = 1; rows <= 8; ++rows) {
            if (!enabledRow(rows)) {
                continue;
            }
            float gateMs = 0.0f;
            float upMs = 0.0f;
            float downMs = 0.0f;
            GemmFloorCase gateCase = {"gate_cublaslt", rows, hidden, inter, warmup, repeat};
            GemmFloorCase upCase = {"up_cublaslt", rows, hidden, inter, warmup, repeat};
            GemmFloorCase downCase = {"down_cublaslt", rows, inter, hidden, warmup, repeat};
            ok = runGemmLtFloorCase(handle, gateCase, &gateMs) && ok;
            ok = runGemmLtFloorCase(handle, upCase, &upMs) && ok;
            ok = runGemmLtFloorCase(handle, downCase, &downMs) && ok;
            MNN_PRINT("[bench_ops/cuda/perf/DecodeRepairMlpGemmLtFloor] rows=%d hidden=%d inter=%d "
                      "gate=%.4f ms up=%.4f ms down=%.4f ms projection_sum=%.4f ms\n",
                      rows, hidden, inter, gateMs, upMs, downMs, gateMs + upMs + downMs);
        }
        cublasLtDestroy(handle);
        return ok;
    }
};
#endif

static ErrorCode executeChain(DirectOpBench& bench,
                              const std::vector<std::pair<Execution*, std::pair<std::vector<Tensor*>, std::vector<Tensor*>>>>& chain) {
    for (const auto& item : chain) {
        auto code = bench.execute(item.first, item.second.first, item.second.second);
        if (code != NO_ERROR) {
            return code;
        }
    }
    return NO_ERROR;
}

static void configureFullCopyRaster(Tensor* src, Tensor* dst) {
    auto des = TensorUtils::getDescribe(dst);
    des->regions.resize(1);
    auto& region = des->regions[0];
    region.origin = src;
    region.src.offset = 0;
    region.dst.offset = 0;
    region.size[0] = src->batch();
    region.size[1] = src->channel();
    region.size[2] = src->height() * src->width();
    region.src.stride[0] = region.size[1] * region.size[2];
    region.src.stride[1] = region.size[2];
    region.src.stride[2] = 1;
    region.dst.stride[0] = region.size[1] * region.size[2];
    region.dst.stride[1] = region.size[2];
    region.dst.stride[2] = 1;
}

static bool runLinearConvertChainCase(const LinearConvertChainCase& c) {
    KVMeta meta;
    meta.pic_decode_repair_sparse_active = true;
    DirectOpBench bench(MNN_FORWARD_CUDA, &meta);
    if (!bench.valid()) {
        return false;
    }

    auto inputNchw = bench.tensor({c.rows, c.ic, 1, 1}, Tensor::CAFFE, false);
    auto inputC4 = bench.tensor({c.rows, c.ic, 1, 1}, Tensor::CAFFE_C4, false);
    auto convOutC4 = bench.tensor({c.rows, c.oc, 1, 1}, Tensor::CAFFE_C4, false);
    auto outputNchw = bench.tensor({c.rows, c.oc, 1, 1}, Tensor::CAFFE, false);
    auto noConvertOutput = bench.tensor({c.rows, c.oc, 1, 1}, Tensor::CAFFE, false);
    auto inputNhwc = bench.tensor({c.rows, 1, 1, c.ic}, Tensor::TENSORFLOW, false);
    auto outputNhwc = bench.tensor({c.rows, 1, 1, c.oc}, Tensor::TENSORFLOW, false);
    if (!inputNchw || !inputC4 || !convOutC4 || !outputNchw || !noConvertOutput || !inputNhwc || !outputNhwc) {
        return false;
    }

    configureFullCopyRaster(inputNchw, inputC4);
    configureFullCopyRaster(convOutC4, outputNchw);

    PicExternalPlan linearPlan;
    const std::string externalPath = linearExternalWeightPath(c.ic, c.oc, c.quantBlock);
    if (!buildLinearExternalWeight(externalPath, c.ic, c.oc, c.quantBlock, &linearPlan)) {
        return false;
    }

    auto preRasterOp = makeRasterOp("bench_linear_pre_convert_raster");
    auto postRasterOp = makeRasterOp("bench_linear_post_convert_raster");
    auto c4ConvOp = makeWeightOnlyLinearConvOp(c.ic, c.oc, c.quantBlock);
    auto nchwConvOp = makeWeightOnlyLinearConvOp(c.ic, c.oc, c.quantBlock);
    auto nhwcLinearOp = makePicLinearNhwcWeightOnlyExtraOp("bench_pic_linear_nhwc_weight_only",
                                                           externalPath, c.ic, c.oc, linearPlan);

    std::vector<Tensor*> preInputs = {inputNchw};
    std::vector<Tensor*> preOutputs = {inputC4};
    std::vector<Tensor*> c4ConvInputs = {inputC4};
    std::vector<Tensor*> c4ConvOutputs = {convOutC4};
    std::vector<Tensor*> postInputs = {convOutC4};
    std::vector<Tensor*> postOutputs = {outputNchw};
    std::vector<Tensor*> nchwConvInputs = {inputNchw};
    std::vector<Tensor*> nchwConvOutputs = {noConvertOutput};
    std::vector<Tensor*> nhwcLinearInputs = {inputNhwc};
    std::vector<Tensor*> nhwcLinearOutputs = {outputNhwc};

    auto preExe = bench.create(preInputs, preOutputs, preRasterOp->get());
    auto c4ConvExe = bench.create(c4ConvInputs, c4ConvOutputs, c4ConvOp->get());
    auto postExe = bench.create(postInputs, postOutputs, postRasterOp->get());
    auto nchwConvExe = bench.create(nchwConvInputs, nchwConvOutputs, nchwConvOp->get());
    auto nhwcLinearExe = bench.create(nhwcLinearInputs, nhwcLinearOutputs, nhwcLinearOp->get());
    if (!preExe || !c4ConvExe || !postExe || !nchwConvExe || !nhwcLinearExe) {
        MNN_ERROR("failed to create LinearConvertChain execution for %s rows=%d ic=%d oc=%d\n",
                  c.name, c.rows, c.ic, c.oc);
        return false;
    }
    if (bench.resize(preExe.get(), preInputs, preOutputs) != NO_ERROR ||
        bench.resize(c4ConvExe.get(), c4ConvInputs, c4ConvOutputs) != NO_ERROR ||
        bench.resize(postExe.get(), postInputs, postOutputs) != NO_ERROR ||
        bench.resize(nchwConvExe.get(), nchwConvInputs, nchwConvOutputs) != NO_ERROR ||
        bench.resize(nhwcLinearExe.get(), nhwcLinearInputs, nhwcLinearOutputs) != NO_ERROR) {
        MNN_ERROR("LinearConvertChain onResize failed for %s rows=%d ic=%d oc=%d\n",
                  c.name, c.rows, c.ic, c.oc);
        return false;
    }

    using ChainItem = std::pair<Execution*, std::pair<std::vector<Tensor*>, std::vector<Tensor*>>>;
    const std::vector<ChainItem> withConvertChain = {
        {preExe.get(), {preInputs, preOutputs}},
        {c4ConvExe.get(), {c4ConvInputs, c4ConvOutputs}},
        {postExe.get(), {postInputs, postOutputs}},
    };
    const std::vector<ChainItem> noConvertChain = {
        {nchwConvExe.get(), {nchwConvInputs, nchwConvOutputs}},
    };

    CudaEventPair timer;
    float preMs = 0.0f;
    float c4ConvMs = 0.0f;
    float postMs = 0.0f;
    float nchwConvMs = 0.0f;
    float nhwcLinearMs = 0.0f;
    float withConvertMs = 0.0f;
    float noConvertMs = 0.0f;
    if (!timer.measure([&]() { return bench.execute(preExe.get(), preInputs, preOutputs); }, c.warmup, c.repeat, &preMs) ||
        !timer.measure([&]() { return bench.execute(c4ConvExe.get(), c4ConvInputs, c4ConvOutputs); }, c.warmup, c.repeat, &c4ConvMs) ||
        !timer.measure([&]() { return bench.execute(postExe.get(), postInputs, postOutputs); }, c.warmup, c.repeat, &postMs) ||
        !timer.measure([&]() { return bench.execute(nchwConvExe.get(), nchwConvInputs, nchwConvOutputs); }, c.warmup, c.repeat, &nchwConvMs) ||
        !timer.measure([&]() { return bench.execute(nhwcLinearExe.get(), nhwcLinearInputs, nhwcLinearOutputs); }, c.warmup, c.repeat, &nhwcLinearMs) ||
        !timer.measure([&]() { return executeChain(bench, withConvertChain); }, c.warmup, c.repeat, &withConvertMs) ||
        !timer.measure([&]() { return executeChain(bench, noConvertChain); }, c.warmup, c.repeat, &noConvertMs)) {
        return false;
    }

    MNN_PRINT("[bench_ops/cuda/perf/LinearConvertChain] %-18s rows=%d ic=%d oc=%d qblock=%d "
              "pre_raster=%.4f ms c4_conv=%.4f ms post_raster=%.4f ms nchw_conv=%.4f ms "
              "nhwc_linear=%.4f ms with_convert_chain=%.4f ms no_convert_chain=%.4f ms "
              "nhwc_delta=%.4f ms nhwc_ratio=%.3f convert_delta=%.4f ms convert_ratio=%.3f external=%s\n",
              c.name, c.rows, c.ic, c.oc, c.quantBlock,
              preMs, c4ConvMs, postMs, nchwConvMs, nhwcLinearMs,
              withConvertMs, noConvertMs, nhwcLinearMs - withConvertMs,
              withConvertMs > 0.0f ? nhwcLinearMs / withConvertMs : 0.0f,
              withConvertMs - noConvertMs,
              noConvertMs > 0.0f ? withConvertMs / noConvertMs : 0.0f,
              externalPath.c_str());
    ::fflush(stdout);
    return true;
}

static bool runPicDecodeMlpCase(int rows, int hidden, int inter, int warmup, int repeat) {
    constexpr int quantBlock = 64;

    KVMeta meta;
    meta.pic_decode_repair_sparse_active = true;
    DirectOpBench bench(MNN_FORWARD_CUDA, &meta);
    if (!bench.valid()) {
        return false;
    }

    auto input = bench.tensor({rows, hidden, 1, 1}, Tensor::CAFFE, false);
    auto gate = bench.tensor({rows, inter, 1, 1}, Tensor::CAFFE, false);
    auto up = bench.tensor({rows, inter, 1, 1}, Tensor::CAFFE, false);
    auto swiglu = bench.tensor({rows, inter, 1, 1}, Tensor::CAFFE, false);
    auto packedGateUp = bench.tensor({rows, inter * 2, 1, 1}, Tensor::CAFFE, false);
    auto packedSwiglu = bench.tensor({rows, inter, 1, 1}, Tensor::CAFFE, false);
    auto output = bench.tensor({rows, hidden, 1, 1}, Tensor::CAFFE, false);
    auto packedOutput = bench.tensor({rows, hidden, 1, 1}, Tensor::CAFFE, false);
    auto fusedGate = bench.tensor({rows, inter, 1, 1}, Tensor::CAFFE, false);
    auto fusedUp = bench.tensor({rows, inter, 1, 1}, Tensor::CAFFE, false);
    auto fusedSwiglu = bench.tensor({rows, inter, 1, 1}, Tensor::CAFFE, false);
    auto fusedOutput = bench.tensor({rows, hidden, 1, 1}, Tensor::CAFFE, false);
    auto directSilu = bench.tensor({rows, inter, 1, 1}, Tensor::CAFFE, false);
    auto directOutput = bench.tensor({rows, hidden, 1, 1}, Tensor::CAFFE, false);
    if (!input || !gate || !up || !swiglu || !packedGateUp || !packedSwiglu || !output || !packedOutput ||
        !fusedGate || !fusedUp || !fusedSwiglu || !fusedOutput || !directSilu || !directOutput) {
        return false;
    }

    PicExternalPlan gatePlan;
    PicExternalPlan upPlan;
    const std::string externalPath = gateUpExternalWeightPath(hidden, inter, quantBlock);
    if (!buildGateUpExternalWeights(externalPath, hidden, inter, quantBlock, &gatePlan, &upPlan)) {
        return false;
    }

    auto gateOp = makeWeightOnlyLinearConvOp(hidden, inter, quantBlock);
    auto upOp = makeWeightOnlyLinearConvOp(hidden, inter, quantBlock);
    auto siluOp = makePicSiluMulOp("bench_pic_decode_mlp_silu");
    auto downOp = makeWeightOnlyLinearConvOp(inter, hidden, quantBlock);
    auto concatOp = makeWeightOnlyLinearConvOp(hidden, inter * 2, quantBlock);
    auto packedSiluOp = makePicPackedSiluMulOp("bench_pic_decode_mlp_packed_silu");
    auto packedDownOp = makeWeightOnlyLinearConvOp(inter, hidden, quantBlock);
    auto gateUpOp = makePicGateUpWeightOnlyExtraOp("PicGateUpWeightOnly", "bench_pic_decode_mlp_gateup_weight_only",
                                                   externalPath, hidden, inter, gatePlan, upPlan);
    auto fusedSiluOp = makePicSiluMulOp("bench_pic_decode_mlp_gateup_silu");
    auto fusedDownOp = makeWeightOnlyLinearConvOp(inter, hidden, quantBlock);
    auto directGateUpSiluOp = makePicGateUpWeightOnlyExtraOp("PicGateUpSiluWeightOnly",
                                                            "bench_pic_decode_mlp_gateup_silu_weight_only",
                                                            externalPath, hidden, inter, gatePlan, upPlan);
    auto directDownOp = makeWeightOnlyLinearConvOp(inter, hidden, quantBlock);

    std::vector<Tensor*> gateInputs = {input};
    std::vector<Tensor*> gateOutputs = {gate};
    std::vector<Tensor*> upInputs = {input};
    std::vector<Tensor*> upOutputs = {up};
    std::vector<Tensor*> siluInputs = {gate, up};
    std::vector<Tensor*> siluOutputs = {swiglu};
    std::vector<Tensor*> downInputs = {swiglu};
    std::vector<Tensor*> downOutputs = {output};
    std::vector<Tensor*> concatInputs = {input};
    std::vector<Tensor*> concatOutputs = {packedGateUp};
    std::vector<Tensor*> packedSiluInputs = {packedGateUp};
    std::vector<Tensor*> packedSiluOutputs = {packedSwiglu};
    std::vector<Tensor*> packedDownInputs = {packedSwiglu};
    std::vector<Tensor*> packedDownOutputs = {packedOutput};
    std::vector<Tensor*> gateUpInputs = {input};
    std::vector<Tensor*> gateUpOutputs = {fusedGate, fusedUp};
    std::vector<Tensor*> fusedSiluInputs = {fusedGate, fusedUp};
    std::vector<Tensor*> fusedSiluOutputs = {fusedSwiglu};
    std::vector<Tensor*> fusedDownInputs = {fusedSwiglu};
    std::vector<Tensor*> fusedDownOutputs = {fusedOutput};
    std::vector<Tensor*> directGateUpSiluInputs = {input};
    std::vector<Tensor*> directGateUpSiluOutputs = {directSilu};
    std::vector<Tensor*> directDownInputs = {directSilu};
    std::vector<Tensor*> directDownOutputs = {directOutput};

    auto gateExe = bench.create(gateInputs, gateOutputs, gateOp->get());
    auto upExe = bench.create(upInputs, upOutputs, upOp->get());
    auto siluExe = bench.create(siluInputs, siluOutputs, siluOp->get());
    auto downExe = bench.create(downInputs, downOutputs, downOp->get());
    auto concatExe = bench.create(concatInputs, concatOutputs, concatOp->get());
    auto packedSiluExe = bench.create(packedSiluInputs, packedSiluOutputs, packedSiluOp->get());
    auto packedDownExe = bench.create(packedDownInputs, packedDownOutputs, packedDownOp->get());
    auto gateUpExe = bench.create(gateUpInputs, gateUpOutputs, gateUpOp->get());
    auto fusedSiluExe = bench.create(fusedSiluInputs, fusedSiluOutputs, fusedSiluOp->get());
    auto fusedDownExe = bench.create(fusedDownInputs, fusedDownOutputs, fusedDownOp->get());
    auto directGateUpSiluExe = bench.create(directGateUpSiluInputs, directGateUpSiluOutputs, directGateUpSiluOp->get());
    auto directDownExe = bench.create(directDownInputs, directDownOutputs, directDownOp->get());
    if (!gateExe || !upExe || !siluExe || !downExe || !concatExe || !packedSiluExe || !packedDownExe ||
        !gateUpExe || !fusedSiluExe || !fusedDownExe || !directGateUpSiluExe || !directDownExe) {
        MNN_ERROR("failed to create PicDecodeMlp execution rows=%d\n", rows);
        return false;
    }
    if (bench.resize(gateExe.get(), gateInputs, gateOutputs) != NO_ERROR ||
        bench.resize(upExe.get(), upInputs, upOutputs) != NO_ERROR ||
        bench.resize(siluExe.get(), siluInputs, siluOutputs) != NO_ERROR ||
        bench.resize(downExe.get(), downInputs, downOutputs) != NO_ERROR ||
        bench.resize(concatExe.get(), concatInputs, concatOutputs) != NO_ERROR ||
        bench.resize(packedSiluExe.get(), packedSiluInputs, packedSiluOutputs) != NO_ERROR ||
        bench.resize(packedDownExe.get(), packedDownInputs, packedDownOutputs) != NO_ERROR ||
        bench.resize(gateUpExe.get(), gateUpInputs, gateUpOutputs) != NO_ERROR ||
        bench.resize(fusedSiluExe.get(), fusedSiluInputs, fusedSiluOutputs) != NO_ERROR ||
        bench.resize(fusedDownExe.get(), fusedDownInputs, fusedDownOutputs) != NO_ERROR ||
        bench.resize(directGateUpSiluExe.get(), directGateUpSiluInputs, directGateUpSiluOutputs) != NO_ERROR ||
        bench.resize(directDownExe.get(), directDownInputs, directDownOutputs) != NO_ERROR) {
        MNN_ERROR("PicDecodeMlp onResize failed rows=%d\n", rows);
        return false;
    }

    // Materialize dependencies once before measuring individual downstream ops.
    if (bench.execute(gateExe.get(), gateInputs, gateOutputs) != NO_ERROR ||
        bench.execute(upExe.get(), upInputs, upOutputs) != NO_ERROR ||
        bench.execute(siluExe.get(), siluInputs, siluOutputs) != NO_ERROR) {
        MNN_ERROR("PicDecodeMlp dependency warm execute failed rows=%d\n", rows);
        return false;
    }
    if (bench.execute(concatExe.get(), concatInputs, concatOutputs) != NO_ERROR ||
        bench.execute(packedSiluExe.get(), packedSiluInputs, packedSiluOutputs) != NO_ERROR) {
        MNN_ERROR("PicDecodeMlp packed dependency warm execute failed rows=%d\n", rows);
        return false;
    }
    if (bench.execute(gateUpExe.get(), gateUpInputs, gateUpOutputs) != NO_ERROR ||
        bench.execute(fusedSiluExe.get(), fusedSiluInputs, fusedSiluOutputs) != NO_ERROR ||
        bench.execute(directGateUpSiluExe.get(), directGateUpSiluInputs, directGateUpSiluOutputs) != NO_ERROR) {
        MNN_ERROR("PicDecodeMlp gateup fused dependency warm execute failed rows=%d\n", rows);
        return false;
    }

    float gateMs = 0.0f;
    float upMs = 0.0f;
    float siluMs = 0.0f;
    float downMs = 0.0f;
    float siluDownMs = 0.0f;
    float totalMs = 0.0f;
    float concatMs = 0.0f;
    float packedSiluMs = 0.0f;
    float packedDownMs = 0.0f;
    float packedSiluDownMs = 0.0f;
    float packedTotalMs = 0.0f;
    float gateUpWeightOnlyMs = 0.0f;
    float fusedSiluMs = 0.0f;
    float fusedDownMs = 0.0f;
    float fusedSiluDownMs = 0.0f;
    float fusedTotalMs = 0.0f;
    float directGateUpSiluMs = 0.0f;
    float directDownMs = 0.0f;
    float directTotalMs = 0.0f;
    CudaEventPair timer;
    if (!timer.measure([&]() { return bench.execute(gateExe.get(), gateInputs, gateOutputs); }, warmup, repeat, &gateMs)) {
        return false;
    }
    if (!timer.measure([&]() { return bench.execute(upExe.get(), upInputs, upOutputs); }, warmup, repeat, &upMs)) {
        return false;
    }
    if (!timer.measure([&]() { return bench.execute(siluExe.get(), siluInputs, siluOutputs); }, warmup, repeat, &siluMs)) {
        return false;
    }
    if (!timer.measure([&]() { return bench.execute(downExe.get(), downInputs, downOutputs); }, warmup, repeat, &downMs)) {
        return false;
    }
    if (!timer.measure([&]() { return bench.execute(concatExe.get(), concatInputs, concatOutputs); }, warmup, repeat, &concatMs)) {
        return false;
    }
    if (!timer.measure([&]() { return bench.execute(packedSiluExe.get(), packedSiluInputs, packedSiluOutputs); }, warmup, repeat, &packedSiluMs)) {
        return false;
    }
    if (!timer.measure([&]() { return bench.execute(packedDownExe.get(), packedDownInputs, packedDownOutputs); }, warmup, repeat, &packedDownMs)) {
        return false;
    }
    if (!timer.measure([&]() { return bench.execute(gateUpExe.get(), gateUpInputs, gateUpOutputs); }, warmup, repeat, &gateUpWeightOnlyMs)) {
        return false;
    }
    if (!timer.measure([&]() { return bench.execute(fusedSiluExe.get(), fusedSiluInputs, fusedSiluOutputs); }, warmup, repeat, &fusedSiluMs)) {
        return false;
    }
    if (!timer.measure([&]() { return bench.execute(fusedDownExe.get(), fusedDownInputs, fusedDownOutputs); }, warmup, repeat, &fusedDownMs)) {
        return false;
    }
    if (!timer.measure([&]() { return bench.execute(directGateUpSiluExe.get(), directGateUpSiluInputs, directGateUpSiluOutputs); }, warmup, repeat, &directGateUpSiluMs)) {
        return false;
    }
    if (!timer.measure([&]() { return bench.execute(directDownExe.get(), directDownInputs, directDownOutputs); }, warmup, repeat, &directDownMs)) {
        return false;
    }
    std::vector<std::pair<Execution*, std::pair<std::vector<Tensor*>, std::vector<Tensor*>>>> siluDownChain = {
        {siluExe.get(), {siluInputs, siluOutputs}},
        {downExe.get(), {downInputs, downOutputs}},
    };
    if (!timer.measure([&]() { return executeChain(bench, siluDownChain); }, warmup, repeat, &siluDownMs)) {
        return false;
    }
    std::vector<std::pair<Execution*, std::pair<std::vector<Tensor*>, std::vector<Tensor*>>>> chain = {
        {gateExe.get(), {gateInputs, gateOutputs}},
        {upExe.get(), {upInputs, upOutputs}},
        {siluExe.get(), {siluInputs, siluOutputs}},
        {downExe.get(), {downInputs, downOutputs}},
    };
    if (!timer.measure([&]() { return executeChain(bench, chain); }, warmup, repeat, &totalMs)) {
        return false;
    }
    std::vector<std::pair<Execution*, std::pair<std::vector<Tensor*>, std::vector<Tensor*>>>> packedSiluDownChain = {
        {packedSiluExe.get(), {packedSiluInputs, packedSiluOutputs}},
        {packedDownExe.get(), {packedDownInputs, packedDownOutputs}},
    };
    if (!timer.measure([&]() { return executeChain(bench, packedSiluDownChain); }, warmup, repeat, &packedSiluDownMs)) {
        return false;
    }
    std::vector<std::pair<Execution*, std::pair<std::vector<Tensor*>, std::vector<Tensor*>>>> packedChain = {
        {concatExe.get(), {concatInputs, concatOutputs}},
        {packedSiluExe.get(), {packedSiluInputs, packedSiluOutputs}},
        {packedDownExe.get(), {packedDownInputs, packedDownOutputs}},
    };
    if (!timer.measure([&]() { return executeChain(bench, packedChain); }, warmup, repeat, &packedTotalMs)) {
        return false;
    }
    std::vector<std::pair<Execution*, std::pair<std::vector<Tensor*>, std::vector<Tensor*>>>> fusedSiluDownChain = {
        {fusedSiluExe.get(), {fusedSiluInputs, fusedSiluOutputs}},
        {fusedDownExe.get(), {fusedDownInputs, fusedDownOutputs}},
    };
    if (!timer.measure([&]() { return executeChain(bench, fusedSiluDownChain); }, warmup, repeat, &fusedSiluDownMs)) {
        return false;
    }
    std::vector<std::pair<Execution*, std::pair<std::vector<Tensor*>, std::vector<Tensor*>>>> fusedChain = {
        {gateUpExe.get(), {gateUpInputs, gateUpOutputs}},
        {fusedSiluExe.get(), {fusedSiluInputs, fusedSiluOutputs}},
        {fusedDownExe.get(), {fusedDownInputs, fusedDownOutputs}},
    };
    if (!timer.measure([&]() { return executeChain(bench, fusedChain); }, warmup, repeat, &fusedTotalMs)) {
        return false;
    }
    std::vector<std::pair<Execution*, std::pair<std::vector<Tensor*>, std::vector<Tensor*>>>> directChain = {
        {directGateUpSiluExe.get(), {directGateUpSiluInputs, directGateUpSiluOutputs}},
        {directDownExe.get(), {directDownInputs, directDownOutputs}},
    };
    if (!timer.measure([&]() { return executeChain(bench, directChain); }, warmup, repeat, &directTotalMs)) {
        return false;
    }
    MNN_PRINT("[bench_ops/cuda/perf/PicDecodeMlp] rows=%d hidden=%d inter=%d qblock=%d "
              "gate=%.4f ms up=%.4f ms silu=%.4f ms down=%.4f ms silu_down=%.4f ms "
              "silu_down_over_down=%.4f ms sum=%.4f ms chain=%.4f ms\n",
              rows, hidden, inter, quantBlock, gateMs, upMs, siluMs, downMs,
              siluDownMs, siluDownMs - downMs, gateMs + upMs + siluMs + downMs, totalMs);
    MNN_PRINT("[bench_ops/cuda/perf/PicDecodeMlpPacked] rows=%d hidden=%d inter=%d qblock=%d "
              "concat=%.4f ms packed_silu=%.4f ms down=%.4f ms packed_silu_down=%.4f ms "
              "sum=%.4f ms chain=%.4f ms delta_vs_split_chain=%.4f ms\n",
              rows, hidden, inter, quantBlock, concatMs, packedSiluMs, packedDownMs,
              packedSiluDownMs, concatMs + packedSiluMs + packedDownMs,
              packedTotalMs, packedTotalMs - totalMs);
    MNN_PRINT("[bench_ops/cuda/perf/PicDecodeMlpGateUpWeightOnly] rows=%d hidden=%d inter=%d qblock=%d "
              "gateup=%.4f ms silu=%.4f ms down=%.4f ms silu_down=%.4f ms "
              "sum=%.4f ms chain=%.4f ms delta_vs_split_chain=%.4f ms external=%s\n",
              rows, hidden, inter, quantBlock, gateUpWeightOnlyMs, fusedSiluMs, fusedDownMs,
              fusedSiluDownMs, gateUpWeightOnlyMs + fusedSiluMs + fusedDownMs,
              fusedTotalMs, fusedTotalMs - totalMs, externalPath.c_str());
    MNN_PRINT("[bench_ops/cuda/perf/PicDecodeMlpGateUpSiluWeightOnly] rows=%d hidden=%d inter=%d qblock=%d "
              "gateup_silu=%.4f ms down=%.4f ms sum=%.4f ms chain=%.4f ms "
              "delta_vs_split_chain=%.4f ms\n",
              rows, hidden, inter, quantBlock, directGateUpSiluMs, directDownMs,
              directGateUpSiluMs + directDownMs, directTotalMs, directTotalMs - totalMs);
    ::fflush(stdout);
    return true;
}

class CudaPicDecodeMlpPerf : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        if (precision != BackendConfig::Precision_Low && precision != BackendConfig::Precision_Normal) {
            MNN_PRINT("bench_ops/cuda/perf/PicDecodeMlp expects precision=2(fp16) or 0(mix); got %d.\n",
                      precision);
            return false;
        }
        if (!requireMemoryLow("bench_ops/cuda/perf/PicDecodeMlp")) {
            return false;
        }
        const int warmup = envInt("MNN_BENCH_MLP_WARMUP", envInt("MNN_BENCH_WEIGHT_ONLY_WARMUP", 20));
        const int repeat = envInt("MNN_BENCH_MLP_REPEAT", envInt("MNN_BENCH_WEIGHT_ONLY_REPEAT", 80));
        const int hidden = envInt("MNN_BENCH_MLP_HIDDEN", 2048);
        const int inter = envInt("MNN_BENCH_MLP_INTER", 8192);
        bool ok = true;
        for (int rows = 1; rows <= 8; ++rows) {
            if (enabledRow(rows)) {
                ok = runPicDecodeMlpCase(rows, hidden, inter, warmup, repeat) && ok;
            }
        }
        return ok;
    }
};

class CudaLinearConvertChainPerf : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        if (precision != BackendConfig::Precision_Low && precision != BackendConfig::Precision_Normal) {
            MNN_PRINT("bench_ops/cuda/perf/LinearConvertChain expects precision=2(fp16) or 0(mix); got %d.\n",
                      precision);
            return false;
        }
        if (!requireMemoryLow("bench_ops/cuda/perf/LinearConvertChain")) {
            return false;
        }
        const int warmup = envInt("MNN_BENCH_LINEAR_CHAIN_WARMUP", envInt("MNN_BENCH_WEIGHT_ONLY_WARMUP", 20));
        const int repeat = envInt("MNN_BENCH_LINEAR_CHAIN_REPEAT", envInt("MNN_BENCH_WEIGHT_ONLY_REPEAT", 80));
        const int hidden = envInt("MNN_BENCH_MLP_HIDDEN", envInt("MNN_BENCH_WEIGHT_ONLY_HIDDEN", 3072));
        const int inter = envInt("MNN_BENCH_MLP_INTER", envInt("MNN_BENCH_WEIGHT_ONLY_INTER", 8192));
        const int kv = envInt("MNN_BENCH_ATTN_KV", 1024);
        constexpr int quantBlock = 64;
        bool ok = true;
        for (int rows = 1; rows <= 8; ++rows) {
            if (!enabledRow(rows)) {
                continue;
            }
            const LinearConvertChainCase cases[] = {
                {"mlp_gate_up", rows, hidden, inter, quantBlock, warmup, repeat},
                {"mlp_down", rows, inter, hidden, quantBlock, warmup, repeat},
                {"attn_q_o", rows, hidden, hidden, quantBlock, warmup, repeat},
                {"attn_k_v", rows, hidden, kv, quantBlock, warmup, repeat},
            };
            for (const auto& c : cases) {
                if (enabledByFilter(c.name)) {
                    ok = runLinearConvertChainCase(c) && ok;
                }
            }
        }
        return ok;
    }
};

// V16 INT8 __dp4a small-M accuracy: ref = V16 off (generic route), candidate = V16 on.
// Both use the same int4 weight-only op; only the env route differs.
static bool runV16Dp4aAccuracyCase(const char* name, int rows, int ic, int oc) {
    constexpr int quantBlock = 64;
    ScopedEnvVar restoreV16("MNN_CUDA_PIC_INT4_SMALLM_DP4A");
    ScopedEnvVar restoreV15("MNN_CUDA_PIC_INT4_SMALLM_V15");
    ScopedEnvVar restoreRows45("MNN_CUDA_PIC_INT4_ROWS45_CUBLAS");
    ScopedEnvVar restoreRows48("MNN_CUDA_PIC_INT4_ROWS48_CUBLASLT");
    KVMeta meta;
    meta.pic_decode_repair_sparse_active = true;
    DirectOpBench bench(MNN_FORWARD_CUDA, &meta);
    if (!bench.valid()) {
        return false;
    }

    auto input = bench.tensor({rows, ic, 1, 1}, Tensor::CAFFE, false);
    auto refOutput = bench.tensor({rows, oc, 1, 1}, Tensor::CAFFE, false);
    auto optOutput = bench.tensor({rows, oc, 1, 1}, Tensor::CAFFE, false);
    if (!input || !refOutput || !optOutput || !writeHalfPattern(input)) {
        return false;
    }

    auto op = makeWeightOnlyLinearConvOp(ic, oc, quantBlock);
    std::vector<Tensor*> refInputs = {input};
    std::vector<Tensor*> refOutputs = {refOutput};
    std::vector<Tensor*> optInputs = {input};
    std::vector<Tensor*> optOutputs = {optOutput};
    auto refExe = bench.create(refInputs, refOutputs, op->get());
    auto optExe = bench.create(optInputs, optOutputs, op->get());
    if (!refExe || !optExe) {
        MNN_ERROR("V16Dp4a failed to create execution for %s rows=%d\n", name, rows);
        return false;
    }
    if (bench.resize(refExe.get(), refInputs, refOutputs) != NO_ERROR ||
        bench.resize(optExe.get(), optInputs, optOutputs) != NO_ERROR) {
        MNN_ERROR("V16Dp4a resize failed for %s rows=%d\n", name, rows);
        return false;
    }

    ::setenv("MNN_CUDA_PIC_INT4_SMALLM_DP4A", "0", 1);
    ::setenv("MNN_CUDA_PIC_INT4_SMALLM_V15", "0", 1);
    ::setenv("MNN_CUDA_PIC_INT4_ROWS45_CUBLAS", "0", 1);
    ::setenv("MNN_CUDA_PIC_INT4_ROWS48_CUBLASLT", "0", 1);
    if (bench.execute(refExe.get(), refInputs, refOutputs) != NO_ERROR) {
        return false;
    }
    ::setenv("MNN_CUDA_PIC_INT4_SMALLM_DP4A", "all", 1);
    if (bench.execute(optExe.get(), optInputs, optOutputs) != NO_ERROR) {
        return false;
    }

    std::vector<float> ref;
    std::vector<float> opt;
    if (!readHalfTensor(refOutput, &ref) || !readHalfTensor(optOutput, &opt)) {
        return false;
    }
    char caseName[128];
    ::snprintf(caseName, sizeof(caseName), "v16_dp4a_%s_rows%d", name, rows);
    return compareHalfOutputs(caseName, opt, ref, 0.08f, 0.08f);
}

class CudaSmallMV16Accuracy : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        if (precision != BackendConfig::Precision_Low) {
            MNN_PRINT("bench_ops/cuda/accuracy/SmallMV16 expects precision=2(fp16); got %d.\n", precision);
            return false;
        }
        if (!requireMemoryLow("bench_ops/cuda/accuracy/SmallMV16")) {
            return false;
        }
        bool ok = true;
        for (int rows : {4, 6, 8}) {
            ok = runV16Dp4aAccuracyCase("llama32_3b_gate", rows, 3072, 8192) && ok;
            ok = runV16Dp4aAccuracyCase("llama32_3b_down", rows, 8192, 3072) && ok;
            ok = runV16Dp4aAccuracyCase("llama32_3b_q", rows, 3072, 3072) && ok;
            ok = runV16Dp4aAccuracyCase("qwen3_4b_gate", rows, 2560, 9728) && ok;
        }
        return ok;
    }
};

// V16 perf: same weight-only op, V16 dp4a route on.
static bool runV16Dp4aPerfCase(const WeightOnlyConvCase& c) {
    ScopedEnvVar restoreV16("MNN_CUDA_PIC_INT4_SMALLM_DP4A");
    KVMeta meta;
    meta.pic_decode_repair_sparse_active = true;
    DirectOpBench bench(MNN_FORWARD_CUDA, &meta);
    if (!bench.valid()) {
        return false;
    }

    auto input = bench.tensor({c.rows, c.ic, 1, 1}, Tensor::CAFFE, false);
    auto output = bench.tensor({c.rows, c.oc, 1, 1}, Tensor::CAFFE, false);
    if (!input || !output) {
        return false;
    }

    auto op = makeWeightOnlyLinearConvOp(c.ic, c.oc, c.quantBlock);
    std::vector<Tensor*> inputs = {input};
    std::vector<Tensor*> outputs = {output};
    auto exe = bench.create(inputs, outputs, op->get());
    if (!exe) {
        MNN_ERROR("failed to create V16Dp4a execution for %s rows=%d ic=%d oc=%d\n",
                  c.name, c.rows, c.ic, c.oc);
        return false;
    }
    if (bench.resize(exe.get(), inputs, outputs) != NO_ERROR) {
        MNN_ERROR("V16Dp4a onResize failed for %s rows=%d ic=%d oc=%d\n",
                  c.name, c.rows, c.ic, c.oc);
        return false;
    }

    ::setenv("MNN_CUDA_PIC_INT4_SMALLM_DP4A", "all", 1);
    CudaEventPair timer;
    float avgMs = 0.0f;
    auto run = [&]() { return bench.execute(exe.get(), inputs, outputs); };
    if (!timer.measure(run, c.warmup, c.repeat, &avgMs)) {
        return false;
    }
    MNN_PRINT("[bench_ops/cuda/perf/SmallMV16] %-18s rows=%d ic=%d oc=%d qblock=%d avg=%.4f ms\n",
              c.name, c.rows, c.ic, c.oc, c.quantBlock, avgMs);
    ::fflush(stdout);
    return true;
}

class CudaSmallMV16Perf : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        if (precision != BackendConfig::Precision_Low && precision != BackendConfig::Precision_Normal) {
            MNN_PRINT("bench_ops/cuda/perf/SmallMV16 expects precision=2(fp16) or 0(mix); got %d.\n", precision);
            return false;
        }
        if (!requireMemoryLow("bench_ops/cuda/perf/SmallMV16")) {
            return false;
        }
        std::vector<WeightOnlyConvCase> cases;
        const int warmup = envInt("MNN_BENCH_WEIGHT_ONLY_WARMUP", 20);
        const int repeat = envInt("MNN_BENCH_WEIGHT_ONLY_REPEAT", 100);
        const int hidden = envInt("MNN_BENCH_WEIGHT_ONLY_HIDDEN", 2048);
        const int inter = envInt("MNN_BENCH_WEIGHT_ONLY_INTER", 8192);
        const int kv = envInt("MNN_BENCH_WEIGHT_ONLY_KV", hidden == 3072 ? 1024 : 512);
        for (int rows : {4, 6, 8}) {
            cases.push_back({"hidden_to_inter", rows, hidden, inter, 64, warmup, repeat});
            cases.push_back({"hidden_to_gateup_concat", rows, hidden, inter * 2, 64, warmup, repeat});
            cases.push_back({"inter_to_hidden", rows, inter, hidden, 64, warmup, repeat});
            cases.push_back({"hidden_to_hidden", rows, hidden, hidden, 64, warmup, repeat});
            cases.push_back({"hidden_to_kv", rows, hidden, kv, 64, warmup, repeat});
            cases.push_back({"hidden_to_qkv_concat", rows, hidden, hidden + 2 * kv, 64, warmup, repeat});
        }
        bool ok = true;
        for (const auto& c : cases) {
            if (enabledByFilter(c.name) && enabledRow(c.rows)) {
                ok = runV16Dp4aPerfCase(c) && ok;
            }
        }
        return ok;
    }
};

} // namespace

MNNTestSuiteRegister(CudaRows45CublasAccuracy, "bench_ops/cuda/accuracy/Rows45Cublas");
MNNTestSuiteRegister(CudaSmallMV16Accuracy, "bench_ops/cuda/accuracy/SmallMV16");
MNNTestSuiteRegister(CudaPicFusedSiluDownAccuracy, "bench_ops/cuda/accuracy/PicFusedSiluDown");
MNNTestSuiteRegister(CudaWeightOnlyConvPerf, "bench_ops/cuda/perf/WeightOnlyConv");
MNNTestSuiteRegister(CudaSmallMV16Perf, "bench_ops/cuda/perf/SmallMV16");
MNNTestSuiteRegister(CudaDecodeRepairMlpGemmFloorPerf, "bench_ops/cuda/perf/DecodeRepairMlpGemmFloor");
MNNTestSuiteRegister(CudaDecodeRepairMlpTiledGemmFloorPerf, "bench_ops/cuda/perf/DecodeRepairMlpTiledGemmFloor");
MNNTestSuiteRegister(CudaDecodeRepairDownTransposedLayoutFloorPerf, "bench_ops/cuda/perf/DecodeRepairDownTransposedLayoutFloor");
MNNTestSuiteRegister(CudaDecodeRepairDownSplitNGemmFloorPerf, "bench_ops/cuda/perf/DecodeRepairDownSplitNGemmFloor");
MNNTestSuiteRegister(CudaDecodeRepairGateUpBatchedGemmFloorPerf, "bench_ops/cuda/perf/DecodeRepairGateUpBatchedGemmFloor");
MNNTestSuiteRegister(CudaDecodeRepairGateUpParallelGemmFloorPerf, "bench_ops/cuda/perf/DecodeRepairGateUpParallelGemmFloor");
MNNTestSuiteRegister(CudaDecodeRepairQkvParallelGemmFloorPerf, "bench_ops/cuda/perf/DecodeRepairQkvParallelGemmFloor");
#if MNN_BENCH_HAS_CUBLASLT
MNNTestSuiteRegister(CudaDecodeRepairMlpGemmLtFloorPerf, "bench_ops/cuda/perf/DecodeRepairMlpGemmLtFloor");
#endif
MNNTestSuiteRegister(CudaPicDecodeMlpPerf, "bench_ops/cuda/perf/PicDecodeMlp");
MNNTestSuiteRegister(CudaPicFusedSiluDownPerf, "bench_ops/cuda/perf/PicFusedSiluDown");
MNNTestSuiteRegister(CudaLinearConvertChainPerf, "bench_ops/cuda/perf/LinearConvertChain");

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
