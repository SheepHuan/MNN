#if defined(MNN_SUPPORT_TRANSFORMER_FUSE) && !defined(MNN_OPENCL_BUFFER_CLOSED)

#include "MNNTestSuite.h"
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "MNN_generated.h"
#include "core/IDSTEncoder.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace MNN;
using namespace MNN::Express;

namespace {

struct WeightOnlyConvExprCase {
    const char* name;
    int rows;
    int ic;
    int oc;
    int quantBlock;
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

static int envInt(const char* name, int fallback) {
    const char* value = ::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::atoi(value);
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

static std::vector<float> makePattern(int size, float scale) {
    std::vector<float> data(size);
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>((i * 13 + 7) % 29 - 14) * scale;
    }
    return data;
}

static std::unique_ptr<OpT> makeWeightOnlyLinearConvOp(int ic, int oc, int quantBlock) {
    std::unique_ptr<OpT> op(new OpT);
    op->name = "opencl_weight_only_expr_bench";
    op->type = OpType_Convolution;
    op->defaultDimentionFormat = MNN_DATA_FORMAT_NCHW;
    op->main.type = OpParameter_Convolution2D;
    op->main.value = new Convolution2DT;

    auto* conv = op->main.AsConvolution2D();
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

    const int groups = (ic + quantBlock - 1) / quantBlock;
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
    return op;
}

static std::shared_ptr<Module> createModuleForCase(const WeightOnlyConvExprCase& c) {
    auto input = _Input({c.rows, c.ic, 1, 1}, NCHW, halide_type_of<float>());
    input->setName("input");
    auto op = makeWeightOnlyLinearConvOp(c.ic, c.oc, c.quantBlock);
    std::vector<VARP> inputs{input};
    auto output = Variable::create(Expr::create(std::move(op), std::move(inputs)));
    output->setName("output");
    std::unique_ptr<NetT> net(new NetT);
    std::vector<VARP> outputs{output};
    Variable::save(outputs, net.get());
    flatbuffers::FlatBufferBuilder builder;
    auto len = Net::Pack(builder, net.get());
    builder.Finish(len);
    return std::shared_ptr<Module>(Module::load({"input"}, {"output"}, builder.GetBufferPointer(), builder.GetSize()),
                                   Module::destroy);
}

static bool runCase(const WeightOnlyConvExprCase& c, int warmup, int repeat) {
    auto precision = static_cast<BackendConfig::PrecisionMode>(MNNTestSuite::get()->pStaus.precision);
    BackendConfig config;
    config.precision = precision;
    config.memory = BackendConfig::Memory_Low;
    int gpuMode = MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_NONE;
    gpuMode = applyOpenCLTuneLevelOverride(gpuMode);
    std::shared_ptr<Executor> executor(Executor::newExecutor(MNN_FORWARD_OPENCL, config, gpuMode));
    if (!executor) {
        MNN_ERROR("failed to create OpenCL executor\n");
        return false;
    }
    ExecutorScope scope(executor);
    executor->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, gpuMode);

    auto module = createModuleForCase(c);
    if (!module) {
        MNN_ERROR("failed to create module for %s rows=%d ic=%d oc=%d\n", c.name, c.rows, c.ic, c.oc);
        return false;
    }
    auto input = _Input({c.rows, c.ic, 1, 1}, NCHW, halide_type_of<float>());
    auto inputData = makePattern(c.rows * c.ic, 0.0078125f);
    {
        auto ptr = input->writeMap<float>();
        ::memcpy(ptr, inputData.data(), inputData.size() * sizeof(float));
        input->unMap();
    }

    for (int i = 0; i < warmup; ++i) {
        auto outputs = module->onForward({input});
        outputs[0]->readMap<float>();
    }
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
        auto outputs = module->onForward({input});
        outputs[0]->readMap<float>();
    }
    auto end = std::chrono::steady_clock::now();
    const double totalMs =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    const double avgMs = totalMs / static_cast<double>(repeat);
    MNN_PRINT("[bench_ops/opencl/dev/WeightOnlyConvExpr] %-22s rows=%d ic=%d oc=%d qblock=%d "
              "precision=%d gpu_mode=0x%x avg=%.4f ms\n",
              c.name, c.rows, c.ic, c.oc, c.quantBlock, static_cast<int>(precision), gpuMode, avgMs);
    ::fflush(stdout);
    return true;
}

static std::vector<WeightOnlyConvExprCase> allCases() {
    return {
        {"minicpm_hidden_to_ffn", 269, 1536, 4608, 64},
        {"minicpm_hidden_to_ffn", 115, 1536, 4608, 64},
        {"minicpm_hidden_to_ffn", 216, 1536, 4608, 64},
        {"minicpm_hidden_to_ffn", 317, 1536, 4608, 64},
        {"minicpm_hidden_to_ffn", 418, 1536, 4608, 64},
        {"minicpm_hidden_to_ffn", 519, 1536, 4608, 64},
        {"minicpm_hidden_to_ffn", 828, 1536, 4608, 64},
        {"minicpm_hidden_to_ffn", 1031, 1536, 4608, 64},
        {"minicpm_hidden_to_ffn", 1287, 1536, 4608, 64},
        {"minicpm_ffn_to_hidden", 269, 4608, 1536, 64},
        {"minicpm_ffn_to_hidden", 115, 4608, 1536, 64},
        {"minicpm_ffn_to_hidden", 216, 4608, 1536, 64},
        {"minicpm_ffn_to_hidden", 317, 4608, 1536, 64},
        {"minicpm_ffn_to_hidden", 418, 4608, 1536, 64},
        {"minicpm_ffn_to_hidden", 519, 4608, 1536, 64},
        {"minicpm_ffn_to_hidden", 828, 4608, 1536, 64},
        {"minicpm_ffn_to_hidden", 1031, 4608, 1536, 64},
        {"minicpm_ffn_to_hidden", 1287, 4608, 1536, 64},
        {"minicpm_hidden_to_attn", 269, 1536, 2048, 64},
        {"minicpm_hidden_to_attn", 115, 1536, 2048, 64},
        {"minicpm_hidden_to_attn", 216, 1536, 2048, 64},
        {"minicpm_hidden_to_attn", 317, 1536, 2048, 64},
        {"minicpm_hidden_to_attn", 418, 1536, 2048, 64},
        {"minicpm_hidden_to_attn", 519, 1536, 2048, 64},
        {"minicpm_hidden_to_attn", 828, 1536, 2048, 64},
        {"minicpm_hidden_to_attn", 1031, 1536, 2048, 64},
        {"minicpm_hidden_to_attn", 1287, 1536, 2048, 64},
        {"minicpm_attn_to_hidden", 269, 2048, 1536, 64},
        {"minicpm_attn_to_hidden", 115, 2048, 1536, 64},
        {"minicpm_attn_to_hidden", 216, 2048, 1536, 64},
        {"minicpm_attn_to_hidden", 317, 2048, 1536, 64},
        {"minicpm_attn_to_hidden", 418, 2048, 1536, 64},
        {"minicpm_attn_to_hidden", 519, 2048, 1536, 64},
        {"minicpm_attn_to_hidden", 828, 2048, 1536, 64},
        {"minicpm_attn_to_hidden", 1031, 2048, 1536, 64},
        {"minicpm_attn_to_hidden", 1287, 2048, 1536, 64},
    };
}

class OpenCLWeightOnlyConvExprPerf : public MNNTestCase {
public:
    bool run(int precision) override {
        (void)precision;
        const int warmup = std::max(1, envInt("MNN_BENCH_OPENCL_WEIGHT_ONLY_WARMUP", 5));
        const int repeat = std::max(1, envInt("MNN_BENCH_OPENCL_WEIGHT_ONLY_REPEAT", 20));
        bool ok = true;
        for (const auto& c : allCases()) {
            if (!enabledByFilter("MNN_BENCH_OPENCL_WEIGHT_ONLY_CASE", c.name) ||
                !enabledRow("MNN_BENCH_OPENCL_WEIGHT_ONLY_ROWS", c.rows)) {
                continue;
            }
            ok = runCase(c, warmup, repeat) && ok;
        }
        return ok;
    }
};

MNNTestSuiteRegister(OpenCLWeightOnlyConvExprPerf, "bench_ops/opencl/dev/WeightOnlyConvExpr");

} // namespace

#endif
