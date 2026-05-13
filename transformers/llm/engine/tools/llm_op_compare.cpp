#include "llm/llm.hpp"
#include "core/MNNFileUtils.h"

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace MNN;
using namespace MNN::Transformer;

namespace {

struct Options {
    std::string modelConfig;
    std::string promptFile;
    std::string baselineBackend = "cpu";
    std::string targetBackend = "opencl";
    std::string dumpDir = ".cache/llm-op-compare";
    std::string opFilter;
    int threads = 4;
    int attentionMode = 8;
    int dynamicOption = 0;
    int precision = 2;
    int memory = 2;
    int power = 0;
    int maxNewTokens = 0;
    int cpuSme2NeonDivisionRatio = 41;
    int cpuSmeCoreNum = 2;
    bool useMmap = false;
    bool useTemplate = true;
    bool dumpTargetTensors = false;
};

struct OpRecordMeta {
    size_t ordinal = 0;
    int outputIndex = 0;
    std::string opType;
    std::string opName;
    std::string dtype;
    std::vector<int> shape;
    size_t elementCount = 0;
    std::string tensorPath;
};

struct CompareSummary {
    size_t compared = 0;
    size_t matched = 0;
    size_t unsupported = 0;
    size_t metadataMismatch = 0;
    size_t shapeMismatch = 0;
    double worstMaxAbsDiff = 0.0;
    std::string worstTensor;
};

static void printUsage(const char* cmd) {
    std::cout
        << "Usage: " << cmd << " --model <config.json> --prompt-file <prompt.txt> [options]\n"
        << "Options:\n"
        << "  --baseline-backend <cpu|cuda|opencl>   default: cpu\n"
        << "  --target-backend <cpu|cuda|opencl>     default: opencl\n"
        << "  --dump-dir <dir>                       default: .cache/llm-op-compare\n"
        << "  --threads <n>                          default: 4\n"
        << "  --attention-mode <n>                   default: 8\n"
        << "  --dynamic-option <n>                   default: 0\n"
        << "  --precision <0|1|2>                    default: 2\n"
        << "  --memory <0|1|2>                       default: 2\n"
        << "  --power <0|1|2>                        default: 0\n"
        << "  --max-new-tokens <n>                   default: 0 (prefill only)\n"
        << "  --use-mmap <true|false>                default: false\n"
        << "  --use-template <true|false>            default: true\n"
        << "  --dump-target-tensors <true|false>     default: false\n"
        << "  --op-filter <regex>                    only capture matching op name/type\n";
}

static bool parseBool(const std::string& text, bool fallback = false) {
    if (text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "yes") {
        return true;
    }
    if (text == "0" || text == "false" || text == "FALSE" || text == "off" || text == "no") {
        return false;
    }
    return fallback;
}

static bool parseArgs(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto nextValue = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                return {};
            }
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            return false;
        }
        if (arg == "--model") {
            options.modelConfig = nextValue("--model");
        } else if (arg == "--prompt-file") {
            options.promptFile = nextValue("--prompt-file");
        } else if (arg == "--baseline-backend") {
            options.baselineBackend = nextValue("--baseline-backend");
        } else if (arg == "--target-backend") {
            options.targetBackend = nextValue("--target-backend");
        } else if (arg == "--dump-dir") {
            options.dumpDir = nextValue("--dump-dir");
        } else if (arg == "--threads") {
            options.threads = std::atoi(nextValue("--threads").c_str());
        } else if (arg == "--attention-mode") {
            options.attentionMode = std::atoi(nextValue("--attention-mode").c_str());
        } else if (arg == "--dynamic-option") {
            options.dynamicOption = std::atoi(nextValue("--dynamic-option").c_str());
        } else if (arg == "--precision") {
            options.precision = std::atoi(nextValue("--precision").c_str());
        } else if (arg == "--memory") {
            options.memory = std::atoi(nextValue("--memory").c_str());
        } else if (arg == "--power") {
            options.power = std::atoi(nextValue("--power").c_str());
        } else if (arg == "--max-new-tokens") {
            options.maxNewTokens = std::atoi(nextValue("--max-new-tokens").c_str());
        } else if (arg == "--use-mmap") {
            options.useMmap = parseBool(nextValue("--use-mmap"), false);
        } else if (arg == "--use-template") {
            options.useTemplate = parseBool(nextValue("--use-template"), true);
        } else if (arg == "--dump-target-tensors") {
            options.dumpTargetTensors = parseBool(nextValue("--dump-target-tensors"), false);
        } else if (arg == "--op-filter") {
            options.opFilter = nextValue("--op-filter");
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }
    if (options.modelConfig.empty() || options.promptFile.empty()) {
        return false;
    }
    return true;
}

static std::string readWholeFile(const std::string& path) {
    std::ifstream input(path);
    std::ostringstream output;
    output << input.rdbuf();
    return output.str();
}

static std::string sanitizeName(const std::string& text) {
    std::string result = text;
    for (auto& ch : result) {
        bool keep = (ch >= '0' && ch <= '9') ||
                    (ch >= 'a' && ch <= 'z') ||
                    (ch >= 'A' && ch <= 'Z') ||
                    ch == '_' || ch == '-' || ch == '.';
        if (!keep) {
            ch = '_';
        }
    }
    return result;
}

static std::string shapeString(const std::vector<int>& shape) {
    std::ostringstream os;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            os << 'x';
        }
        os << shape[i];
    }
    return os.str();
}

static std::vector<int> tensorShape(const Tensor* tensor) {
    std::vector<int> shape;
    shape.reserve(tensor->dimensions());
    for (int i = 0; i < tensor->dimensions(); ++i) {
        shape.push_back(tensor->length(i));
    }
    return shape;
}

static float fp16ToFloat(uint16_t value) {
    uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value & 0x7C00u) >> 10;
    uint32_t mantissa = value & 0x03FFu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03FFu;
            bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1Fu) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float result = 0.0f;
    ::memcpy(&result, &bits, sizeof(result));
    return result;
}

static std::string tensorDType(const Tensor* tensor) {
    auto type = tensor->getType();
    std::ostringstream os;
    if (type.code == halide_type_float) {
        os << "f";
    } else if (type.code == halide_type_int) {
        os << "i";
    } else if (type.code == halide_type_uint) {
        os << "u";
    } else {
        os << "type";
    }
    os << type.bits;
    return os.str();
}

static bool tensorToFloatVector(const Tensor* tensor, std::vector<float>& values) {
    auto dimType = tensor->getDimensionType();
    std::shared_ptr<Tensor> hostTensor(new Tensor(tensor, dimType));
    const Tensor* readable = tensor;
    if (tensor->copyToHostTensor(hostTensor.get())) {
        readable = hostTensor.get();
    }
    auto type = readable->getType();
    size_t count = readable->elementSize();
    values.resize(count);
    if (type.code == halide_type_float && type.bits == 32) {
        auto src = readable->host<float>();
        std::copy(src, src + count, values.begin());
        return true;
    }
    if (type.code == halide_type_float && type.bits == 16) {
        auto src = readable->host<uint16_t>();
        for (size_t i = 0; i < count; ++i) {
            values[i] = fp16ToFloat(src[i]);
        }
        return true;
    }
    if (type.code == halide_type_int && type.bits == 32) {
        auto src = readable->host<int32_t>();
        for (size_t i = 0; i < count; ++i) {
            values[i] = static_cast<float>(src[i]);
        }
        return true;
    }
    if (type.code == halide_type_int && type.bits == 8) {
        auto src = readable->host<int8_t>();
        for (size_t i = 0; i < count; ++i) {
            values[i] = static_cast<float>(src[i]);
        }
        return true;
    }
    if (type.code == halide_type_uint && type.bits == 8) {
        auto src = readable->host<uint8_t>();
        for (size_t i = 0; i < count; ++i) {
            values[i] = static_cast<float>(src[i]);
        }
        return true;
    }
    if (type.code == halide_type_uint && type.bits == 32) {
        auto src = readable->host<uint32_t>();
        for (size_t i = 0; i < count; ++i) {
            values[i] = static_cast<float>(src[i]);
        }
        return true;
    }
    values.clear();
    return false;
}

static void writeFloatVector(const std::string& path, const std::vector<float>& values) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
}

static bool readFloatVector(const std::string& path, size_t count, std::vector<float>& values) {
    std::ifstream input(path, std::ios::binary);
    if (!input.good()) {
        return false;
    }
    values.resize(count);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(count * sizeof(float)));
    return input.good() || input.eof();
}

static bool matchesFilter(const std::regex* filter, const std::string& type, const std::string& name) {
    if (filter == nullptr) {
        return true;
    }
    std::string text = type + ":" + name;
    return std::regex_search(text, *filter);
}

static std::string backendNameForConfig(const std::string& backend) {
    if (backend == "cpu" || backend == "cuda" || backend == "opencl" || backend == "vulkan") {
        return backend;
    }
    return "cpu";
}

static MNNForwardType backendForwardType(const std::string& backend) {
    if (backend == "cuda") {
        return MNN_FORWARD_CUDA;
    }
    if (backend == "opencl") {
        return MNN_FORWARD_OPENCL;
    }
    if (backend == "vulkan") {
        return MNN_FORWARD_VULKAN;
    }
    return MNN_FORWARD_CPU;
}

static Llm* buildLLM(const Options& options, const std::string& backendName) {
    auto llm = Llm::createLLM(options.modelConfig);
    std::map<int, std::string> level = {{0, "normal"}, {1, "high"}, {2, "low"}};
    std::map<bool, std::string> boolText = {{true, "true"}, {false, "false"}};
    bool ok = true;
    ok &= llm->set_config(R"({"async":false})");
    ok &= llm->set_config(R"({"reuse_kv":false})");
    ok &= llm->set_config(std::string("{\"backend_type\":\"") + backendName + "\"}");
    ok &= llm->set_config(std::string("{\"precision\":\"") + level[options.precision] + "\"}");
    ok &= llm->set_config(std::string("{\"memory\":\"") + level[options.memory] + "\"}");
    ok &= llm->set_config(std::string("{\"power\":\"") + level[options.power] + "\"}");
    ok &= llm->set_config(std::string("{\"thread_num\":") + std::to_string(options.threads) + "}");
    ok &= llm->set_config(std::string("{\"dynamic_option\":") + std::to_string(options.dynamicOption) + "}");
    ok &= llm->set_config(std::string("{\"attention_mode\":") + std::to_string(options.attentionMode) + "}");
    ok &= llm->set_config(std::string("{\"use_mmap\":") + boolText[options.useMmap] + "}");
    ok &= llm->set_config(R"({"tmp_path":"tmp"})");
    ok &= llm->set_config(std::string("{\"cpu_sme2_neon_division_ratio\":") +
                          std::to_string(options.cpuSme2NeonDivisionRatio) + "}");
    ok &= llm->set_config(std::string("{\"cpu_sme_core_num\":") + std::to_string(options.cpuSmeCoreNum) + "}");
    ok &= llm->set_config(std::string("{\"use_template\":") + boolText[options.useTemplate] + "}");
    ok &= llm->set_config(R"({"enable_debug":true})");
    if (!ok) {
        delete llm;
        return nullptr;
    }
    return llm;
}

class BaselineRecorder {
public:
    BaselineRecorder(const std::string& baseDir, const std::regex* filter)
        : mBaseDir(baseDir), mFilter(filter) {
        mTensorDir = mBaseDir + "/tensors";
        MNNCreateDir(mBaseDir.c_str());
        MNNCreateDir(mTensorDir.c_str());
        mMeta.open(mBaseDir + "/records.tsv");
        mMeta << "ordinal\toutput_index\top_type\top_name\tdtype\tshape\telement_count\tfile\n";
    }

    bool after(const std::vector<Tensor*>& outputs, const OperatorInfo* info) {
        if (info == nullptr || info->type() == "Copy") {
            return true;
        }
        if (!matchesFilter(mFilter, info->type(), info->name())) {
            return true;
        }
        for (int i = 0; i < static_cast<int>(outputs.size()); ++i) {
            auto tensor = outputs[i];
            tensor->wait(Tensor::MAP_TENSOR_READ, true);
            std::vector<float> values;
            if (!tensorToFloatVector(tensor, values)) {
                continue;
            }
            OpRecordMeta meta;
            meta.ordinal = mOrdinal;
            meta.outputIndex = i;
            meta.opType = info->type();
            meta.opName = info->name();
            meta.dtype = tensorDType(tensor);
            meta.shape = tensorShape(tensor);
            meta.elementCount = values.size();
            std::ostringstream name;
            name << std::setw(6) << std::setfill('0') << mOrdinal
                 << "_o" << i << "_" << sanitizeName(info->type())
                 << "_" << sanitizeName(info->name()) << ".bin";
            meta.tensorPath = mTensorDir + "/" + name.str();
            writeFloatVector(meta.tensorPath, values);
            mRecords.emplace_back(meta);
            mMeta << meta.ordinal << '\t'
                  << meta.outputIndex << '\t'
                  << meta.opType << '\t'
                  << meta.opName << '\t'
                  << meta.dtype << '\t'
                  << shapeString(meta.shape) << '\t'
                  << meta.elementCount << '\t'
                  << meta.tensorPath << '\n';
            ++mOrdinal;
        }
        return true;
    }

    const std::vector<OpRecordMeta>& records() const {
        return mRecords;
    }

private:
    std::string mBaseDir;
    std::string mTensorDir;
    const std::regex* mFilter = nullptr;
    std::ofstream mMeta;
    std::vector<OpRecordMeta> mRecords;
    size_t mOrdinal = 0;
};

class CompareRecorder {
public:
    CompareRecorder(const std::string& baseDir,
                    const std::vector<OpRecordMeta>& baseline,
                    const std::regex* filter,
                    bool dumpTargetTensors)
        : mBaseDir(baseDir), mBaseline(baseline), mFilter(filter), mDumpTargetTensors(dumpTargetTensors) {
        MNNCreateDir(mBaseDir.c_str());
        if (mDumpTargetTensors) {
            mTargetTensorDir = mBaseDir + "/target_tensors";
            MNNCreateDir(mTargetTensorDir.c_str());
        }
        mCompare.open(mBaseDir + "/compare.tsv");
        mCompare << "ordinal\toutput_index\top_type\top_name\tshape\tbaseline_shape\telement_count\tmax_abs_diff\tmean_abs_diff\trmse\tcosine\tstatus\n";
    }

    bool after(const std::vector<Tensor*>& outputs, const OperatorInfo* info) {
        if (info == nullptr || info->type() == "Copy") {
            return true;
        }
        if (!matchesFilter(mFilter, info->type(), info->name())) {
            return true;
        }
        for (int i = 0; i < static_cast<int>(outputs.size()); ++i) {
            auto tensor = outputs[i];
            tensor->wait(Tensor::MAP_TENSOR_READ, true);
            std::vector<float> values;
            if (!tensorToFloatVector(tensor, values)) {
                ++mSummary.unsupported;
                continue;
            }
            auto shape = tensorShape(tensor);
            std::string status = "ok";
            double maxAbsDiff = 0.0;
            double meanAbsDiff = 0.0;
            double rmse = 0.0;
            double cosine = 1.0;
            std::string baselineShape;
            if (mCursor >= mBaseline.size()) {
                status = "missing_baseline";
                ++mSummary.metadataMismatch;
            } else {
                const auto& ref = mBaseline[mCursor];
                baselineShape = shapeString(ref.shape);
                if (ref.outputIndex != i || ref.opType != info->type() || ref.opName != info->name()) {
                    status = "metadata_mismatch";
                    ++mSummary.metadataMismatch;
                } else if (ref.elementCount != values.size() || ref.shape != shape) {
                    status = "shape_mismatch";
                    ++mSummary.shapeMismatch;
                } else {
                    std::vector<float> baselineValues;
                    if (!readFloatVector(ref.tensorPath, ref.elementCount, baselineValues)) {
                        status = "baseline_read_failed";
                        ++mSummary.metadataMismatch;
                    } else {
                        compareVectors(baselineValues, values, maxAbsDiff, meanAbsDiff, rmse, cosine);
                        ++mSummary.matched;
                        if (maxAbsDiff > mSummary.worstMaxAbsDiff) {
                            mSummary.worstMaxAbsDiff = maxAbsDiff;
                            mSummary.worstTensor = info->type() + ":" + info->name() + ":" + std::to_string(i);
                        }
                    }
                }
            }
            if (mDumpTargetTensors) {
                std::ostringstream name;
                name << std::setw(6) << std::setfill('0') << mCursor
                     << "_o" << i << "_" << sanitizeName(info->type())
                     << "_" << sanitizeName(info->name()) << ".bin";
                writeFloatVector(mTargetTensorDir + "/" + name.str(), values);
            }
            mCompare << mCursor << '\t'
                     << i << '\t'
                     << info->type() << '\t'
                     << info->name() << '\t'
                     << shapeString(shape) << '\t'
                     << baselineShape << '\t'
                     << values.size() << '\t'
                     << maxAbsDiff << '\t'
                     << meanAbsDiff << '\t'
                     << rmse << '\t'
                     << cosine << '\t'
                     << status << '\n';
            ++mCursor;
            ++mSummary.compared;
        }
        return true;
    }

    const CompareSummary& summary() const {
        return mSummary;
    }

private:
    static void compareVectors(const std::vector<float>& baseline,
                               const std::vector<float>& target,
                               double& maxAbsDiff,
                               double& meanAbsDiff,
                               double& rmse,
                               double& cosine) {
        double absSum = 0.0;
        double squareSum = 0.0;
        double dot = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        maxAbsDiff = 0.0;
        for (size_t i = 0; i < baseline.size(); ++i) {
            double a = baseline[i];
            double b = target[i];
            double diff = std::abs(a - b);
            maxAbsDiff = std::max(maxAbsDiff, diff);
            absSum += diff;
            squareSum += diff * diff;
            dot += a * b;
            normA += a * a;
            normB += b * b;
        }
        meanAbsDiff = baseline.empty() ? 0.0 : absSum / static_cast<double>(baseline.size());
        rmse = baseline.empty() ? 0.0 : std::sqrt(squareSum / static_cast<double>(baseline.size()));
        if (normA <= 0.0 || normB <= 0.0) {
            cosine = 1.0;
        } else {
            cosine = dot / (std::sqrt(normA) * std::sqrt(normB));
        }
    }

private:
    std::string mBaseDir;
    std::string mTargetTensorDir;
    const std::vector<OpRecordMeta>& mBaseline;
    const std::regex* mFilter = nullptr;
    bool mDumpTargetTensors = false;
    std::ofstream mCompare;
    CompareSummary mSummary;
    size_t mCursor = 0;
};

template <typename CallbackFactory>
static bool runWithBackend(const Options& options,
                           const std::string& backendName,
                           const std::string& prompt,
                           CallbackFactory&& callbackFactory) {
    BackendConfig backendConfig;
    auto executor = Express::Executor::newExecutor(backendForwardType(backendName), backendConfig, 1);
    Express::ExecutorScope scope(executor);
    std::unique_ptr<Llm> llm(buildLLM(options, backendNameForConfig(backendName)));
    if (!llm) {
        MNN_ERROR("[llm_op_compare] failed to create llm for backend=%s\n", backendName.c_str());
        return false;
    }
    auto callbacks = callbackFactory();
    llm->setDebugCallback(std::move(callbacks.first), std::move(callbacks.second));
    if (!llm->load()) {
        MNN_ERROR("[llm_op_compare] failed to load model for backend=%s\n", backendName.c_str());
        return false;
    }
    llm->set_config(R"({"async":false})");
    std::ostringstream sink;
    llm->response(prompt, &sink, nullptr, options.maxNewTokens);
    auto context = llm->getContext();
    if (context != nullptr && context->status == LlmStatus::INTERNAL_ERROR) {
        MNN_ERROR("[llm_op_compare] backend=%s hit INTERNAL_ERROR\n", backendName.c_str());
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parseArgs(argc, argv, options)) {
        printUsage(argv[0]);
        return 1;
    }

    std::string prompt = readWholeFile(options.promptFile);
    if (prompt.empty()) {
        MNN_ERROR("[llm_op_compare] prompt file is empty: %s\n", options.promptFile.c_str());
        return 2;
    }

    std::unique_ptr<std::regex> filter;
    if (!options.opFilter.empty()) {
        filter.reset(new std::regex(options.opFilter));
    }

    std::string rootDir = options.dumpDir;
    MNNCreateDir(rootDir.c_str());
    std::string baselineDir = rootDir + "/baseline_" + sanitizeName(options.baselineBackend);
    std::string targetDir = rootDir + "/target_" + sanitizeName(options.targetBackend);

    BaselineRecorder baseline(baselineDir, filter.get());
    std::cout << "[llm_op_compare] capture baseline backend=" << options.baselineBackend << "\n";
    bool baselineOk = runWithBackend(
        options,
        options.baselineBackend,
        prompt,
        [&]() {
            TensorCallBackWithInfo before = [](const std::vector<Tensor*>&, const OperatorInfo*) {
                return true;
            };
            TensorCallBackWithInfo after = [&](const std::vector<Tensor*>& outputs, const OperatorInfo* info) {
                return baseline.after(outputs, info);
            };
            return std::make_pair(std::move(before), std::move(after));
        });
    if (!baselineOk) {
        return 3;
    }

    CompareRecorder compare(targetDir, baseline.records(), filter.get(), options.dumpTargetTensors);
    std::cout << "[llm_op_compare] compare target backend=" << options.targetBackend << "\n";
    bool targetOk = runWithBackend(
        options,
        options.targetBackend,
        prompt,
        [&]() {
            TensorCallBackWithInfo before = [](const std::vector<Tensor*>&, const OperatorInfo*) {
                return true;
            };
            TensorCallBackWithInfo after = [&](const std::vector<Tensor*>& outputs, const OperatorInfo* info) {
                return compare.after(outputs, info);
            };
            return std::make_pair(std::move(before), std::move(after));
        });
    if (!targetOk) {
        return 4;
    }

    const auto& summary = compare.summary();
    std::ofstream report(targetDir + "/summary.txt");
    report << "model=" << options.modelConfig << "\n";
    report << "prompt_file=" << options.promptFile << "\n";
    report << "baseline_backend=" << options.baselineBackend << "\n";
    report << "target_backend=" << options.targetBackend << "\n";
    report << "max_new_tokens=" << options.maxNewTokens << "\n";
    report << "compared=" << summary.compared << "\n";
    report << "matched=" << summary.matched << "\n";
    report << "unsupported=" << summary.unsupported << "\n";
    report << "metadata_mismatch=" << summary.metadataMismatch << "\n";
    report << "shape_mismatch=" << summary.shapeMismatch << "\n";
    report << "worst_max_abs_diff=" << summary.worstMaxAbsDiff << "\n";
    report << "worst_tensor=" << summary.worstTensor << "\n";
    report.close();

    std::cout << "[llm_op_compare] compared=" << summary.compared
              << " matched=" << summary.matched
              << " metadata_mismatch=" << summary.metadataMismatch
              << " shape_mismatch=" << summary.shapeMismatch
              << " worst_max_abs_diff=" << summary.worstMaxAbsDiff
              << " worst_tensor=" << summary.worstTensor << "\n";
    std::cout << "[llm_op_compare] report: " << targetDir << "/summary.txt\n";
    std::cout << "[llm_op_compare] compare table: " << targetDir << "/compare.tsv\n";
    return 0;
}
