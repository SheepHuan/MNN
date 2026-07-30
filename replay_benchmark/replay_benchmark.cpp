// MNN execution record and replay benchmark.
//
// The positional arguments intentionally follow benchmark/benchmark.cpp. The
// --record option adds execution recording and, together with an op selector,
// enables single-Execution replay.

#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>

#include "ReplayRecord.hpp"
#include "ReplayRunner.hpp"
#include "OpenCLPmuBenchmark.hpp"
#include "ModelPmuBenchmark.hpp"
#include "KernelCorpusBenchmark.hpp"
#include "core/Backend.hpp"
#include "revertMNNModel.hpp"

#if defined(MNN_REPLAY_HAS_OPENCL)
#include "backend/opencl/core/OpenCLBackend.hpp"
#endif

#include "rapidjson/document.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#include <direct.h>
#include <io.h>
#else
#include <dirent.h>
#endif

namespace {

using namespace MNN::Replay;
using MNN::Backend;
using MNN::BackendConfig;
using MNN::Interpreter;
using MNN::Tensor;

static std::vector<ModelFile> findModels(const std::string& path) {
    std::vector<ModelFile> result;
    if (fileExists(path)) {
        ModelFile model;
        model.path = path;
        const auto slash = path.find_last_of("/\\");
        model.name = slash == std::string::npos ? path : path.substr(slash + 1);
        result.emplace_back(std::move(model));
        return result;
    }
    if (!isDirectory(path)) {
        return result;
    }
#if defined(_MSC_VER)
    std::string pattern = joinPath(path, "*.mnn");
    struct _finddata_t fileInfo;
    intptr_t handle = _findfirst(pattern.c_str(), &fileInfo);
    if (handle == -1) {
        return result;
    }
    do {
        ModelFile model;
        model.name = fileInfo.name;
        model.path = joinPath(path, fileInfo.name);
        result.emplace_back(std::move(model));
    } while (_findnext(handle, &fileInfo) == 0);
    _findclose(handle);
#else
    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        return result;
    }
    while (auto* entry = readdir(dir)) {
        const std::string name = entry->d_name;
        if (name.size() < 4 || name.substr(name.size() - 4) != ".mnn") {
            continue;
        }
        ModelFile model;
        model.name = name;
        model.path = joinPath(path, name);
        result.emplace_back(std::move(model));
    }
    closedir(dir);
#endif
    std::sort(result.begin(), result.end(),
              [](const ModelFile& lhs, const ModelFile& rhs) { return lhs.path < rhs.path; });
    return result;
}

static bool parseOptions(int argc, const char* argv[], Options& options) {
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto requireValue = [&](std::string& value) -> bool {
            if (i + 1 >= argc) {
                return false;
            }
            value = argv[++i];
            return true;
        };
        if (arg == "--help" || arg == "-h") {
            return false;
        } else if (arg == "--record") {
            if (!requireValue(options.recordDir)) {
                return false;
            }
            options.record = true;
        } else if (arg == "--args") {
            if (!requireValue(options.argsFile)) {
                return false;
            }
        } else if (arg == "--model") {
            if (!requireValue(options.model)) {
                return false;
            }
        } else if (arg == "--op-id") {
            std::string value;
            if (!requireValue(value)) {
                return false;
            }
            options.opId = std::atoi(value.c_str());
            options.replay = true;
        } else if (arg == "--op-type") {
            if (!requireValue(options.opType)) {
                return false;
            }
            options.replay = true;
        } else if (arg == "--execution") {
            if (!requireValue(options.execution)) {
                return false;
            }
            options.replay = true;
        } else if (arg == "--variant") {
            if (!requireValue(options.variant)) {
                return false;
            }
            options.replay = true;
        } else if (arg == "--perf-counter-output") {
            if (!requireValue(options.perfCounterOutput)) {
                return false;
            }
            options.replay = true;
        } else if (arg == "--perf-counter-events") {
            if (!requireValue(options.perfCounterEvents)) {
                return false;
            }
            options.replay = true;
        } else if (arg == "--opencl-pmu-bench") {
            options.openclPmuBench = true;
        } else if (arg == "--model-pmu-bench") {
            options.modelPmuBench = true;
        } else if (arg == "--model-pmu-workload-runs") {
            std::string value;
            if (!requireValue(value)) return false;
            options.modelPmuWorkloadRuns = std::max(1, std::atoi(value.c_str()));
        } else if (arg == "--model-pmu-warmup-runs") {
            std::string value;
            if (!requireValue(value)) return false;
            options.modelPmuWarmupRuns = std::max(0, std::atoi(value.c_str()));
        } else if (arg == "--model-pmu-control-runs") {
            std::string value;
            if (!requireValue(value)) return false;
            options.modelPmuControlRuns = std::max(1, std::atoi(value.c_str()));
        } else if (arg == "--model-pmu-forward") {
            std::string value;
            if (!requireValue(value)) return false;
            options.modelPmuForward = std::atoi(value.c_str());
        } else if (arg == "--model-pmu-precision") {
            std::string value;
            if (!requireValue(value)) return false;
            options.precision = std::atoi(value.c_str());
        } else if (arg == "--model-pmu-prompt") {
            if (!requireValue(options.modelPmuPrompt)) return false;
        } else if (arg == "--opencl-pmu-list-events") {
            options.openclPmuBench = true;
            options.openclPmuListEvents = true;
        } else if (arg == "--opencl-pmu-case") {
            if (!requireValue(options.openclPmuCase)) {
                return false;
            }
        } else if (arg == "--opencl-pmu-iterations") {
            std::string value;
            if (!requireValue(value)) {
                return false;
            }
            options.openclPmuIterations = std::atoi(value.c_str());
        } else if (arg == "--opencl-pmu-workload-runs") {
            std::string value;
            if (!requireValue(value)) {
                return false;
            }
            options.openclPmuWorkloadRuns = std::max(1, std::atoi(value.c_str()));
        } else if (arg == "--opencl-pmu-warmup-runs") {
            std::string value;
            if (!requireValue(value)) {
                return false;
            }
            options.openclPmuWarmupRuns = std::max(0, std::atoi(value.c_str()));
        } else if (arg == "--opencl-pmu-size") {
            std::string value;
            if (!requireValue(value)) {
                return false;
            }
            options.openclPmuSize = static_cast<size_t>(std::strtoull(value.c_str(), nullptr, 10));
        } else if (arg == "--opencl-pmu-local-size") {
            std::string value;
            if (!requireValue(value)) {
                return false;
            }
            options.openclPmuLocalSize = static_cast<size_t>(std::strtoull(value.c_str(), nullptr, 10));
        } else if (arg == "--replay") {
            options.replay = true;
        } else if (arg == "--kernel-corpus-bench") {
            options.kernelCorpusBench = true;
        } else if (arg == "--kernel-corpus-root") {
            if (!requireValue(options.kernelCorpusRoot)) {
                return false;
            }
        } else if (arg == "--kernel-corpus-case") {
            if (!requireValue(options.kernelCorpusCase)) {
                return false;
            }
        } else if (arg == "--kernel-corpus-runs") {
            std::string value;
            if (!requireValue(value)) {
                return false;
            }
            options.kernelCorpusRuns = std::max(1, std::atoi(value.c_str()));
        } else if (arg == "--kernel-corpus-no-latency") {
            options.kernelCorpusMeasureLatency = false;
        } else {
            positional.emplace_back(arg);
        }
    }

    if (options.model.empty() && !positional.empty()) {
        options.model = positional[0];
    }
    if (positional.size() > 1)
        options.loop = std::atoi(positional[1].c_str());
    if (positional.size() > 2)
        options.warmup = std::atoi(positional[2].c_str());
    if (positional.size() > 3)
        options.forward = std::atoi(positional[3].c_str());
    if (positional.size() > 4)
        options.gpuMode = std::atoi(positional[4].c_str());
    if (positional.size() > 5)
        options.precision = std::atoi(positional[5].c_str());
    if (positional.size() > 6)
        options.sparsity = static_cast<float>(std::atof(positional[6].c_str()));
    if (positional.size() > 7)
        options.sparseBlockOC = std::atoi(positional[7].c_str());
    if (positional.size() > 8)
        options.testQuantizedModel = std::atoi(positional[8].c_str());
    if (positional.size() > 9)
        options.enableKleidiAI = std::atoi(positional[9].c_str()) != 0;

    if (!options.argsFile.empty()) {
        std::vector<uint8_t> argsBytes;
        if (!readBytes(options.argsFile, argsBytes)) {
            return false;
        }
        rapidjson::Document args;
        args.Parse(reinterpret_cast<const char*>(argsBytes.data()), argsBytes.size());
        if (args.HasParseError() || !args.IsObject()) {
            return false;
        }
        if (args.HasMember("execution_record") && args["execution_record"].IsObject()) {
            const auto& record = args["execution_record"];
            if (record.HasMember("enabled") && record["enabled"].IsBool()) {
                options.record = record["enabled"].GetBool() || options.record;
            }
            if (options.record && options.recordDir.empty() && record.HasMember("output_dir") &&
                record["output_dir"].IsString()) {
                options.recordDir = record["output_dir"].GetString();
            }
        }
    }

    if (options.openclPmuBench) {
        options.replay = false;
        options.record = false;
        return true;
    }
    if (options.kernelCorpusBench) {
        options.replay = false;
        options.record = false;
        return !options.kernelCorpusRoot.empty();
    }
    if (options.modelPmuBench) {
        options.replay = false;
        options.record = false;
        options.forward = options.modelPmuForward;
        return !options.model.empty() && !options.perfCounterOutput.empty() &&
               !options.perfCounterEvents.empty() && options.perfCounterEvents.find(',') == std::string::npos;
    }
    if (options.model.empty()) {
        return false;
    }
    if (!options.perfCounterOutput.empty() &&
        (options.recordDir.empty() ||
         (options.opId < 0 && options.opType.empty() && options.execution.empty() && options.variant.empty()))) {
        return false;
    }
    if (options.replay || options.record) {
        return !options.recordDir.empty();
    }
    return true;
}

static void printUsage(const char* program) {
    std::cout << "Usage:\n"
              << "  " << program
              << " model_or_dir [loop] [warmup] [forward] [gpu_mode] [precision] [sparsity] [sparse_block] [quantized] "
                 "[kleidiai] [--record dir | --args config.json]\n"
              << "  " << program << " --model model.mnn --record dir --op-id N [--execution Name] [--variant Name]\n"
              << "      [--perf-counter-output path.json] [--perf-counter-events name1,name2,...]\n"
              << "  " << program << " --opencl-pmu-bench [--opencl-pmu-case name|all]\n"
                 "      [--opencl-pmu-iterations N] [--opencl-pmu-workload-runs N]\n"
                 "      [--opencl-pmu-warmup-runs N] [--opencl-pmu-size bytes]\n"
                 "      [--opencl-pmu-local-size 32|64|128|256]\n"
                 "      [--perf-counter-events name1,name2,...|auto] [--perf-counter-output path.json]\n"
              << "  " << program << " --model-pmu-bench --model model.mnn --perf-counter-events event\n"
                 "      --perf-counter-output path.json [--model-pmu-workload-runs N]\n"
                 "      [--model-pmu-warmup-runs N] [--model-pmu-control-runs N]\n"
                 "      [--model-pmu-forward 3|7] [--model-pmu-prompt text]\n"
              << "  " << program << " --opencl-pmu-list-events\n"
              << "  " << program << " --kernel-corpus-bench --kernel-corpus-root <dir>\n"
              << "      [--kernel-corpus-case name] [--kernel-corpus-runs N]\n"
              << "      [--kernel-corpus-no-latency]\n"
              << "      [--perf-counter-output path.json] [--perf-counter-events name1,name2]\n"
              << "\n"
              << "forward: 0 CPU, 3 OpenCL, 7 Vulkan; precision: 0 normal, 1 high, 2 low FP16, 3 low BF16\n";
}

static bool recordModel(const ModelFile& model, const Options& options, bool multipleModels) {
    std::unique_ptr<Revert> revert(new Revert(model.path.c_str()));
    if (options.testQuantizedModel) {
        revert->initialize(0.0f, options.sparseBlockOC, false, true);
    } else {
        revert->initialize(options.sparsity, options.sparseBlockOC);
    }
    const auto* modelBuffer = revert->getBuffer();
    const size_t modelSize = revert->getBufferSize();
    std::shared_ptr<Interpreter> net(Interpreter::createFromBuffer(modelBuffer, modelSize), Interpreter::destroy);
    if (net == nullptr) {
        std::cerr << "Can't load model: " << model.path << std::endl;
        return false;
    }
    net->setSessionMode(Interpreter::Session_Debug);
    MNN::ScheduleConfig config;
    config.type = static_cast<MNNForwardType>(options.forward);
    config.numThread = options.gpuMode;
    BackendConfig backendConfig;
    backendConfig.precision = static_cast<BackendConfig::PrecisionMode>(options.precision);
    config.backendConfig = &backendConfig;
    MNN::Session* session = net->createSession(config);
    if (session == nullptr) {
        std::cerr << "Can't create session for model: " << model.path << std::endl;
        return false;
    }
    net->setSessionHint(Interpreter::CPU_ENABLE_KLEIDIAI, options.enableKleidiAI ? 1 : 0);
    const auto* backend = net->getBackend(session, net->getSessionInput(session, nullptr));
    const auto requestedForward = static_cast<MNNForwardType>(options.forward);
    const bool cpuExtension = requestedForward == MNN_FORWARD_CPU &&
                              backend != nullptr && backend->type() == MNN_FORWARD_CPU_EXTENSION;
    if (backend == nullptr || (backend->type() != requestedForward && !cpuExtension)) {
        std::cerr << "The model did not create the requested backend: " << forwardName(options.forward) << std::endl;
        return false;
    }
#if defined(MNN_REPLAY_HAS_OPENCL)
    const auto* openclBackend =
        backend->type() == MNN_FORWARD_OPENCL ? static_cast<const MNN::OpenCL::OpenCLBackend*>(backend) : nullptr;
#endif
    const ModelOpDatabase database(modelBuffer);
    const std::string modelRecordDir = multipleModels ? joinPath(options.recordDir, model.name) : options.recordDir;
    Options recordOptions = options;
    recordOptions.forward = backend->type();
    RecordWriter writer(modelRecordDir, recordOptions, database, model.path);

    auto inputs = net->getSessionInputAll(session);
    for (auto& item : inputs) {
        Tensor* tensor = item.second;
        if (tensor == nullptr) {
            continue;
        }
        void* ptr = tensor->map(Tensor::MAP_TENSOR_WRITE, tensor->getDimensionType());
        if (ptr == nullptr) {
            continue;
        }
        if (tensor->getType().code == halide_type_float && tensor->getType().bits == 32) {
            Revert::fillRandValue(reinterpret_cast<float*>(ptr), tensor->elementSize());
        } else {
            std::memset(ptr, 0, tensor->size());
        }
        tensor->unmap(Tensor::MAP_TENSOR_WRITE, tensor->getDimensionType(), ptr);
    }

    for (int i = 0; i < options.warmup; ++i) {
        if (net->runSession(session) != MNN::NO_ERROR) {
            return false;
        }
    }

    int recorded = 0;
    auto before = [&](const std::vector<Tensor*>& tensors, const MNN::OperatorInfo* info) {
        std::vector<ExecutionTrace> traces;
#if defined(MNN_REPLAY_HAS_OPENCL)
        if (openclBackend != nullptr) {
            for (const auto& trace : openclBackend->getReplayExecutionInfo()) {
                ExecutionTrace executionTrace;
                executionTrace.execution = trace.execution;
                executionTrace.executionName = trace.executionName;
                if (!trace.kernels.empty()) {
                    executionTrace.variant = trace.kernels.front().kernelName;
                }
                for (const auto& kernel : trace.kernels) {
                    KernelRecord record;
                    record.program = kernel.programName;
                    record.kernel = kernel.kernelName;
                    record.gws = kernel.globalWorkSize;
                    record.lws = kernel.localWorkSize;
                    executionTrace.kernels.emplace_back(std::move(record));
                }
                traces.emplace_back(std::move(executionTrace));
            }
        }
#endif
        if (!writer.begin(tensors, info, traces)) {
            return false;
        }
        return true;
    };
    auto after = [&](const std::vector<Tensor*>& tensors, const MNN::OperatorInfo* info) {
        writer.finish(tensors);
        recorded++;
        return true;
    };
    if (net->runSessionWithCallBackInfo(session, before, after, true) != MNN::NO_ERROR) {
        return false;
    }
    if (!writer.write()) {
        return false;
    }
    std::cout << "Recorded " << recorded << " ops to " << modelRecordDir << std::endl;
    return true;
}

static bool runCompatibleBenchmark(const ModelFile& model, const Options& options) {
    std::unique_ptr<Revert> revert(new Revert(model.path.c_str()));
    revert->initialize(options.sparsity, options.sparseBlockOC);
    std::shared_ptr<Interpreter> net(Interpreter::createFromBuffer(revert->getBuffer(), revert->getBufferSize()),
                                     Interpreter::destroy);
    if (net == nullptr) {
        return false;
    }
    net->setSessionMode(Interpreter::Session_Release);
    MNN::ScheduleConfig config;
    config.type = static_cast<MNNForwardType>(options.forward);
    config.numThread = options.gpuMode;
    BackendConfig backendConfig;
    backendConfig.precision = static_cast<BackendConfig::PrecisionMode>(options.precision);
    config.backendConfig = &backendConfig;
    auto* session = net->createSession(config);
    if (session == nullptr) {
        return false;
    }
    auto* input = net->getSessionInput(session, nullptr);
    auto* output = net->getSessionOutput(session, nullptr);
    if (input == nullptr || output == nullptr) {
        return false;
    }
    for (int i = 0; i < options.warmup; ++i) {
        if (net->runSession(session) != MNN::NO_ERROR) {
            return false;
        }
    }
    std::vector<float> costs;
    for (int i = 0; i < options.loop; ++i) {
        const auto start = std::chrono::steady_clock::now();
        if (net->runSession(session) != MNN::NO_ERROR) {
            return false;
        }
        const auto end = std::chrono::steady_clock::now();
        costs.emplace_back(std::chrono::duration<float, std::milli>(end - start).count());
    }
    float sum = 0.0f;
    for (float cost : costs)
        sum += cost;
    std::cout << model.name << " avg=" << (costs.empty() ? 0.0f : sum / costs.size()) << " ms" << std::endl;
    return true;
}

} // namespace

int main(int argc, const char* argv[]) {
    Options options;
    if (!parseOptions(argc, argv, options)) {
        printUsage(argv[0]);
        return 1;
    }
    if (options.openclPmuBench) {
        return runOpenCLPmuBenchmark(options) ? 0 : 1;
    }
    if (options.kernelCorpusBench) {
        MNN::Replay::KernelCorpus::Options kccOptions;
        kccOptions.corpusRoot = options.kernelCorpusRoot;
        kccOptions.caseFilter = options.kernelCorpusCase;
        kccOptions.runs = options.kernelCorpusRuns;
        kccOptions.measureLatency = options.kernelCorpusMeasureLatency;
        kccOptions.perfCounterOutput = options.perfCounterOutput;
        kccOptions.perfCounterEvents = options.perfCounterEvents;
        return MNN::Replay::KernelCorpus::runKernelCorpusBenchmark(kccOptions) ? 0 : 1;
    }
    if (options.modelPmuBench) {
        return runModelPmuBenchmark(options) ? 0 : 1;
    }
    if (options.replay) {
        return replayOp(options) ? 0 : 1;
    }
    if (options.record && options.forward != MNN_FORWARD_CPU && options.forward != MNN_FORWARD_OPENCL) {
        std::cerr << "--record supports CPU (0) and OpenCL (3) forward types" << std::endl;
        return 1;
    }
    const auto models = findModels(options.model);
    if (models.empty()) {
        std::cerr << "No MNN model found: " << options.model << std::endl;
        return 1;
    }
    bool success = true;
    for (const auto& model : models) {
        if (options.record) {
            success = recordModel(model, options, models.size() > 1) && success;
        } else {
            success = runCompatibleBenchmark(model, options) && success;
        }
    }
    return success ? 0 : 1;
}
