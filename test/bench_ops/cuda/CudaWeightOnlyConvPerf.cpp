//
//  CudaWeightOnlyConvPerf.cpp
//  MNNTests
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "CudaOpBenchUtils.hpp"
#include "core/IDSTEncoder.hpp"

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
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

static bool runWeightOnlyConvCase(const WeightOnlyConvCase& c) {
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

static bool checkCublas(cublasStatus_t status, const char* where) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        MNN_ERROR("%s failed: %d\n", where, static_cast<int>(status));
        return false;
    }
    return true;
}

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
        for (int rows = 1; rows <= 8; ++rows) {
            cases.push_back({"hidden_to_inter", rows, 2048, 8192, 64, warmup, repeat});
            cases.push_back({"hidden_to_gateup_concat", rows, 2048, 16384, 64, warmup, repeat});
            cases.push_back({"inter_to_hidden", rows, 8192, 2048, 64, warmup, repeat});
            cases.push_back({"hidden_to_hidden", rows, 2048, 2048, 64, warmup, repeat});
            cases.push_back({"hidden_to_kv", rows, 2048, 512, 64, warmup, repeat});
            cases.push_back({"hidden_to_qkv_concat", rows, 2048, 3072, 64, warmup, repeat});
        }
        if (::getenv("MNN_BENCH_WEIGHT_ONLY_VOCAB") != nullptr) {
            for (int rows = 1; rows <= 2; ++rows) {
                cases.push_back({"hidden_to_vocab", rows, 2048, 128256, 64, std::max(1, warmup / 2), std::max(1, repeat / 2)});
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
    if (bench.execute(refExe.get(), refInputs, refOutputs) != NO_ERROR) {
        return false;
    }
    ::setenv("MNN_CUDA_PIC_INT4_ROWS45_CUBLAS", policy, 1);
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
        return ok;
    }
};

static bool runDecodeRepairMlpGemmFloorCase(cublasHandle_t handle, int rows, int warmup, int repeat) {
    constexpr int hidden = 2048;
    constexpr int inter = 8192;
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
        bool ok = true;
        for (int rows = 1; rows <= 8; ++rows) {
            if (enabledRow(rows)) {
                ok = runDecodeRepairMlpGemmFloorCase(handle, rows, warmup, repeat) && ok;
            }
        }
        cublasDestroy(handle);
        return ok;
    }
};

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

static bool runPicDecodeMlpCase(int rows, int warmup, int repeat) {
    constexpr int hidden = 2048;
    constexpr int inter = 8192;
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
    auto output = bench.tensor({rows, hidden, 1, 1}, Tensor::CAFFE, false);
    if (!input || !gate || !up || !swiglu || !output) {
        return false;
    }

    auto gateOp = makeWeightOnlyLinearConvOp(hidden, inter, quantBlock);
    auto upOp = makeWeightOnlyLinearConvOp(hidden, inter, quantBlock);
    auto siluOp = makePicSiluMulOp("bench_pic_decode_mlp_silu");
    auto downOp = makeWeightOnlyLinearConvOp(inter, hidden, quantBlock);

    std::vector<Tensor*> gateInputs = {input};
    std::vector<Tensor*> gateOutputs = {gate};
    std::vector<Tensor*> upInputs = {input};
    std::vector<Tensor*> upOutputs = {up};
    std::vector<Tensor*> siluInputs = {gate, up};
    std::vector<Tensor*> siluOutputs = {swiglu};
    std::vector<Tensor*> downInputs = {swiglu};
    std::vector<Tensor*> downOutputs = {output};

    auto gateExe = bench.create(gateInputs, gateOutputs, gateOp->get());
    auto upExe = bench.create(upInputs, upOutputs, upOp->get());
    auto siluExe = bench.create(siluInputs, siluOutputs, siluOp->get());
    auto downExe = bench.create(downInputs, downOutputs, downOp->get());
    if (!gateExe || !upExe || !siluExe || !downExe) {
        MNN_ERROR("failed to create PicDecodeMlp execution rows=%d\n", rows);
        return false;
    }
    if (bench.resize(gateExe.get(), gateInputs, gateOutputs) != NO_ERROR ||
        bench.resize(upExe.get(), upInputs, upOutputs) != NO_ERROR ||
        bench.resize(siluExe.get(), siluInputs, siluOutputs) != NO_ERROR ||
        bench.resize(downExe.get(), downInputs, downOutputs) != NO_ERROR) {
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

    float gateMs = 0.0f;
    float upMs = 0.0f;
    float siluMs = 0.0f;
    float downMs = 0.0f;
    float siluDownMs = 0.0f;
    float totalMs = 0.0f;
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
    MNN_PRINT("[bench_ops/cuda/perf/PicDecodeMlp] rows=%d hidden=%d inter=%d qblock=%d "
              "gate=%.4f ms up=%.4f ms silu=%.4f ms down=%.4f ms silu_down=%.4f ms "
              "silu_down_over_down=%.4f ms sum=%.4f ms chain=%.4f ms\n",
              rows, hidden, inter, quantBlock, gateMs, upMs, siluMs, downMs,
              siluDownMs, siluDownMs - downMs, gateMs + upMs + siluMs + downMs, totalMs);
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
        bool ok = true;
        for (int rows = 1; rows <= 8; ++rows) {
            if (enabledRow(rows)) {
                ok = runPicDecodeMlpCase(rows, warmup, repeat) && ok;
            }
        }
        return ok;
    }
};

} // namespace

MNNTestSuiteRegister(CudaRows45CublasAccuracy, "bench_ops/cuda/accuracy/Rows45Cublas");
MNNTestSuiteRegister(CudaWeightOnlyConvPerf, "bench_ops/cuda/perf/WeightOnlyConv");
MNNTestSuiteRegister(CudaDecodeRepairMlpGemmFloorPerf, "bench_ops/cuda/perf/DecodeRepairMlpGemmFloor");
MNNTestSuiteRegister(CudaPicDecodeMlpPerf, "bench_ops/cuda/perf/PicDecodeMlp");

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
