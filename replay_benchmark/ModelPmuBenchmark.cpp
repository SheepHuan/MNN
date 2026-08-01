#include "ModelPmuBenchmark.hpp"

#include "ReplayRecord.hpp"
#include "revertMNNModel.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <sys/stat.h>
#include <vector>

#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#if defined(MNN_REPLAY_HAS_PERFCOUNTER)
#include "MNNPerfCounter.hpp"
#endif

#if defined(MNN_REPLAY_HAS_LLM)
#include "llm/llm.hpp"
#include <sstream>
#endif

namespace MNN {
namespace Replay {

ModelPmuDelta makeModelPmuDelta(uint64_t control, uint64_t workload) {
    ModelPmuDelta result;
    result.control = control;
    result.workload = workload;
    if (workload >= control) {
        const uint64_t difference = workload - control;
        result.delta = difference > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
                           ? std::numeric_limits<int64_t>::max()
                           : static_cast<int64_t>(difference);
    } else {
        const uint64_t difference = control - workload;
        result.delta = difference > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
                           ? std::numeric_limits<int64_t>::min()
                           : -static_cast<int64_t>(difference);
    }
    return result;
}

bool validateModelPmuRunConfig(const ModelPmuRunConfig& config, std::string* error) {
    const char* message = nullptr;
    if (config.model.empty()) {
        message = "model-PMU mode requires a model path";
    } else if (config.forward != MNN_FORWARD_OPENCL && config.forward != MNN_FORWARD_VULKAN) {
        message = "model-PMU mode requires OpenCL or Vulkan forward";
    } else if (config.event.empty() || config.event.find(',') != std::string::npos) {
        message = "model-PMU mode requires exactly one PMU event";
    } else if (config.output.empty()) {
        message = "model-PMU mode requires an output path";
    } else if (config.warmupRuns < 0 || config.controlRuns < 1 || config.workloadRuns < 1) {
        message = "model-PMU run counts are invalid";
    }
    if (error != nullptr) {
        *error = message == nullptr ? std::string() : message;
    }
    return message == nullptr;
}

namespace {

static uint64_t monotonicNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch()).count());
}

static bool makeParentDirectories(const std::string& path) {
    const std::string::size_type slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return true;
    const std::string parent = path.substr(0, slash);
    if (parent.empty()) return true;
    std::string current;
    for (size_t i = 0; i < parent.size(); ++i) {
        current.push_back(parent[i]);
        if (parent[i] != '/' && parent[i] != '\\' && i + 1 != parent.size()) continue;
        if (current != "/" && mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) return false;
    }
    return mkdir(parent.c_str(), 0755) == 0 || errno == EEXIST;
}

static void addString(rapidjson::Value& object, const char* key, const std::string& value,
                      rapidjson::Document::AllocatorType& allocator) {
    object.AddMember(rapidjson::Value(key, allocator), rapidjson::Value(value.c_str(), allocator), allocator);
}

static bool writeReport(const Options& options, rapidjson::Document& document) {
    if (!makeParentDirectories(options.perfCounterOutput)) return false;
    FILE* file = std::fopen(options.perfCounterOutput.c_str(), "wb");
    if (file == nullptr) return false;
    char buffer[4096];
    rapidjson::FileWriteStream stream(file, buffer, sizeof(buffer));
    rapidjson::Writer<rapidjson::FileWriteStream> writer(stream);
    const bool result = document.Accept(writer);
    std::fclose(file);
    rapidjson::StringBuffer output;
    rapidjson::Writer<rapidjson::StringBuffer> outputWriter(output);
    document.Accept(outputWriter);
    std::cout << output.GetString() << std::endl;
    return result;
}

#if defined(MNN_REPLAY_HAS_PERFCOUNTER)

struct PmuSample {
    bool available = false;
    uint64_t value = 0;
    uint64_t startNs = 0;
    uint64_t endNs = 0;
    std::string error;
};

static bool assignUnsignedCounter(const MNN::PerfCounter::CounterValue& counter,
                                  PmuSample* sample) {
    if (sample == nullptr || counter.status != MNN::PerfCounter::CounterValueStatus::Valid ||
        counter.valueKind != MNN::PerfCounter::CounterValueKind::UnsignedInteger) {
        if (sample != nullptr) sample->error = "PMU counter did not return a valid uint64 value";
        return false;
    }
    sample->value = counter.value;
    sample->available = true;
    return true;
}

static std::unique_ptr<MNN::PerfCounter::Session> createPmuSession(const Options& options,
                                                                     MNN::PerfCounter::DeviceInfo* device,
                                                                     std::string* error) {
    MNN::PerfCounter::CounterSpec spec;
    spec.name = options.perfCounterEvents.c_str();
    const char* sessionError = nullptr;
    std::unique_ptr<MNN::PerfCounter::Session> session(
        MNN::PerfCounter::Session::create(&spec, 1, device, &sessionError));
    if (session == nullptr && error != nullptr) {
        *error = sessionError == nullptr ? "GPU PMU session creation failed" : sessionError;
    }
    return session;
}

static PmuSample sampleControl(const Options& options) {
    PmuSample result;
    MNN::PerfCounter::DeviceInfo device;
    std::string error;
    std::unique_ptr<MNN::PerfCounter::Session> session = createPmuSession(options, &device, &error);
    if (session == nullptr) {
        result.error = error;
        return result;
    }
    if (!session->start()) {
        result.error = session->error();
        return result;
    }
    result.startNs = monotonicNs();
    for (int i = 0; i < options.modelPmuControlRuns; ++i) {
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }
    result.endNs = monotonicNs();
    MNN::PerfCounter::CounterValue value;
    if (!session->stop(&value, 1)) {
        result.error = session->error();
        return result;
    }
    assignUnsignedCounter(value, &result);
    return result;
}

static PmuSample sampleWorkload(const Options& options, MNN::Interpreter* net, MNN::Session* session) {
    PmuSample result;
    MNN::PerfCounter::DeviceInfo device;
    std::string error;
    std::unique_ptr<MNN::PerfCounter::Session> pmu = createPmuSession(options, &device, &error);
    if (pmu == nullptr) {
        result.error = error;
        return result;
    }
    if (!pmu->start()) {
        result.error = pmu->error();
        return result;
    }
    result.startNs = monotonicNs();
    bool ok = true;
    for (int i = 0; i < options.modelPmuWorkloadRuns; ++i) {
        if (net->runSession(session) != MNN::NO_ERROR) {
            ok = false;
            result.error = "MNN Session run failed";
            break;
        }
    }
    if (ok) {
        const auto& outputs = net->getSessionOutputAll(session);
        for (const auto& item : outputs) {
            if (item.second == nullptr || item.second->wait(MNN::Tensor::MAP_TENSOR_READ, true) != 0) {
                ok = false;
                result.error = "MNN Session output synchronization failed";
                break;
            }
        }
    }
    result.endNs = monotonicNs();
    MNN::PerfCounter::CounterValue value;
    if (!pmu->stop(&value, 1)) {
        if (result.error.empty()) result.error = pmu->error();
        return result;
    }
    if (!ok) return result;
    assignUnsignedCounter(value, &result);
    return result;
}

#if defined(MNN_REPLAY_HAS_LLM)

static PmuSample sampleLlmWorkload(const Options& options, MNN::Transformer::Llm* llm) {
    PmuSample result;
    MNN::PerfCounter::DeviceInfo device;
    std::string error;
    std::unique_ptr<MNN::PerfCounter::Session> pmu = createPmuSession(options, &device, &error);
    if (pmu == nullptr) {
        result.error = error;
        return result;
    }
    if (!pmu->start()) {
        result.error = pmu->error();
        return result;
    }
    result.startNs = monotonicNs();
    bool ok = true;
    std::ostringstream sink;
    for (int i = 0; i < options.modelPmuWorkloadRuns; ++i) {
        llm->reset();
        sink.str(std::string());
        sink.clear();
        llm->response(options.modelPmuPrompt, &sink, nullptr, 1);
        if (llm->getContext() == nullptr ||
            llm->getContext()->status == MNN::Transformer::LlmStatus::INTERNAL_ERROR) {
            ok = false;
            result.error = "MNN LLM Session run failed";
            break;
        }
    }
    result.endNs = monotonicNs();
    MNN::PerfCounter::CounterValue value;
    if (!pmu->stop(&value, 1)) {
        if (result.error.empty()) result.error = pmu->error();
        return result;
    }
    if (!ok) return result;
    assignUnsignedCounter(value, &result);
    return result;
}

#endif

#endif

} // namespace

static bool isLlmModelPath(const std::string& path) {
    if (path.size() >= 5 && path.compare(path.size() - 5, 5, ".json") == 0) return true;
    const std::string::size_type slash = path.find_last_of("/\\");
    const std::string parent = slash == std::string::npos ? std::string() : path.substr(0, slash);
    return fileExists(joinPath(parent, "llm_config.json"));
}

static bool runLlmModelPmuBenchmark(const Options& options) {
    rapidjson::Document document;
    document.SetObject();
    auto& allocator = document.GetAllocator();
    addString(document, "format", "mnn-model-pmu-benchmark", allocator);
    document.AddMember("version", 1, allocator);
    addString(document, "model", options.model, allocator);
    addString(document, "workload_type", "llm_session", allocator);
    addString(document, "backend", forwardName(options.forward), allocator);
    document.AddMember("forward", options.forward, allocator);
    document.AddMember("precision", options.precision, allocator);
    addString(document, "status", "unavailable", allocator);
    addString(document, "error", "", allocator);

    const std::string modelName = options.model.substr(options.model.find_last_of("/\\") + 1);
    rapidjson::Value item(rapidjson::kObjectType);
    addString(item, "name", modelName, allocator);
    addString(item, "model", options.model, allocator);
    addString(item, "workload_type", "llm_session", allocator);
    addString(item, "backend", forwardName(options.forward), allocator);
    addString(item, "stage", "prefill_decode_one_token", allocator);
    item.AddMember("warmup_runs", options.modelPmuWarmupRuns, allocator);
    item.AddMember("control_runs", options.modelPmuControlRuns, allocator);
    item.AddMember("workload_runs", options.modelPmuWorkloadRuns, allocator);
    addString(item, "pmu_metric_name", options.perfCounterEvents, allocator);
    rapidjson::Value counters(rapidjson::kArrayType);
    std::string error;

    if (options.forward != MNN_FORWARD_OPENCL) {
        error = "LLM model-PMU currently supports OpenCL forward only";
    }

#if defined(MNN_REPLAY_HAS_LLM) && defined(MNN_REPLAY_HAS_PERFCOUNTER)
    MNN::Transformer::Llm* rawLlm = error.empty() ? MNN::Transformer::Llm::createLLM(options.model) : nullptr;
    std::unique_ptr<MNN::Transformer::Llm, void (*)(MNN::Transformer::Llm*)> llm(
        rawLlm, &MNN::Transformer::Llm::destroy);
    const char* llmPrecision = options.precision == MNN::BackendConfig::Precision_Normal ? "normal" : "low";
    const std::string runtimeConfig = std::string("{\"backend_type\":\"opencl\",\"precision\":\"") +
                                      llmPrecision + "\"}";
    if (llm == nullptr || !llm->set_config(runtimeConfig) || !llm->load()) {
        error = "MNN LLM engine could not load model package";
    }
    if (error.empty()) {
        std::ostringstream sink;
        for (int i = 0; i < options.modelPmuWarmupRuns; ++i) {
            llm->reset();
            sink.str(std::string());
            sink.clear();
            llm->response(options.modelPmuPrompt, &sink, nullptr, 1);
            if (llm->getContext() == nullptr ||
                llm->getContext()->status == MNN::Transformer::LlmStatus::INTERNAL_ERROR) {
                error = "MNN LLM warmup failed";
                break;
            }
        }
    }
    if (error.empty()) {
        const PmuSample control = sampleControl(options);
        const PmuSample workload = sampleLlmWorkload(options, llm.get());
        if (!control.available || !workload.available) {
            error = !workload.error.empty() ? workload.error : control.error;
        } else {
            const ModelPmuDelta delta = makeModelPmuDelta(control.value, workload.value);
            rapidjson::Value counter(rapidjson::kObjectType);
            addString(counter, "name", options.perfCounterEvents, allocator);
            counter.AddMember("control_delta", control.value, allocator);
            counter.AddMember("workload_delta", workload.value, allocator);
            counter.AddMember("delta_metric", delta.delta, allocator);
            counter.AddMember("valid", delta.delta != 0, allocator);
            counters.PushBack(counter, allocator);
            document["status"].SetString("ok", allocator);
        }
    }
#elif !defined(MNN_REPLAY_HAS_LLM)
    error = "MNN LLM engine was not compiled into replay_benchmark";
#else
    error = "MNNPerfCounter was not compiled into replay_benchmark";
#endif
    addString(item, "error", error, allocator);
    addString(item, "status", error.empty() ? "ok" : "unavailable", allocator);
    item.AddMember("counters", counters, allocator);
    document["error"].SetString(error.c_str(), allocator);
    rapidjson::Value cases(rapidjson::kArrayType);
    cases.PushBack(item, allocator);
    document.AddMember("cases", cases, allocator);
    return writeReport(options, document);
}

bool runModelPmuBenchmark(const Options& options) {
    ModelPmuRunConfig config;
    config.model = options.model;
    config.forward = options.forward;
    config.precision = options.precision;
    config.warmupRuns = options.modelPmuWarmupRuns;
    config.controlRuns = options.modelPmuControlRuns;
    config.workloadRuns = options.modelPmuWorkloadRuns;
    config.event = options.perfCounterEvents;
    config.output = options.perfCounterOutput;

    std::string validationError;
    if (!validateModelPmuRunConfig(config, &validationError)) {
        std::cerr << validationError << std::endl;
        return false;
    }
    if (isLlmModelPath(options.model)) {
        return runLlmModelPmuBenchmark(options);
    }

    rapidjson::Document document;
    document.SetObject();
    auto& allocator = document.GetAllocator();
    addString(document, "format", "mnn-model-pmu-benchmark", allocator);
    document.AddMember("version", 1, allocator);
    addString(document, "model", options.model, allocator);
    addString(document, "workload_type", "dnn_session", allocator);
    addString(document, "backend", forwardName(options.forward), allocator);
    document.AddMember("forward", options.forward, allocator);
    document.AddMember("precision", options.precision, allocator);
    addString(document, "status", "unavailable", allocator);
    addString(document, "error", "", allocator);

    rapidjson::Value cases(rapidjson::kArrayType);
    rapidjson::Value item(rapidjson::kObjectType);
    const std::string modelName = options.model.substr(options.model.find_last_of("/\\") + 1);
    addString(item, "name", modelName, allocator);
    addString(item, "model", options.model, allocator);
    addString(item, "workload_type", "dnn_session", allocator);
    addString(item, "backend", forwardName(options.forward), allocator);
    item.AddMember("warmup_runs", options.modelPmuWarmupRuns, allocator);
    item.AddMember("control_runs", options.modelPmuControlRuns, allocator);
    item.AddMember("workload_runs", options.modelPmuWorkloadRuns, allocator);
    addString(item, "pmu_metric_name", options.perfCounterEvents, allocator);

    std::string error;
    std::unique_ptr<Revert> revert;
    std::shared_ptr<MNN::Interpreter> net;
    MNN::Session* session = nullptr;
    bool modelReady = fileExists(options.model);
    if (!modelReady) error = "model file could not be read";
    if (modelReady) {
        revert.reset(new Revert(options.model.c_str()));
        if (options.testQuantizedModel) {
            revert->initialize(0.0f, options.sparseBlockOC, false, true);
        } else {
            revert->initialize(options.sparsity, options.sparseBlockOC);
        }
        net.reset(MNN::Interpreter::createFromBuffer(revert->getBuffer(), revert->getBufferSize()),
                  MNN::Interpreter::destroy);
        if (net == nullptr) error = "MNN Interpreter could not load model";
    }
    MNN::BackendConfig backendConfig;
    backendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(options.precision);
    MNN::ScheduleConfig schedule;
    schedule.type = static_cast<MNNForwardType>(options.forward);
    schedule.numThread = options.gpuMode;
    schedule.backendConfig = &backendConfig;
    if (error.empty()) {
        net->setSessionMode(MNN::Interpreter::Session_Release);
        session = net->createSession(schedule);
        if (session == nullptr) {
            error = std::string("MNN ") + forwardName(options.forward) + " Session could not be created";
        }
    }
    if (error.empty()) {
        MNN::Tensor* sessionInput = net->getSessionInput(session, nullptr);
        const MNN::Backend* backend = sessionInput == nullptr ? nullptr : net->getBackend(session, sessionInput);
        if (backend == nullptr || backend->type() != static_cast<MNNForwardType>(options.forward)) {
            error = std::string("MNN Session did not create requested ") + forwardName(options.forward) +
                    " backend";
        }
    }
    if (error.empty()) {
        const auto& inputs = net->getSessionInputAll(session);
        for (const auto& itemInput : inputs) {
            MNN::Tensor* tensor = itemInput.second;
            if (tensor == nullptr) continue;
            void* ptr = tensor->map(MNN::Tensor::MAP_TENSOR_WRITE, tensor->getDimensionType());
            if (ptr == nullptr) {
                error = "MNN Session input mapping failed";
                break;
            }
            std::memset(ptr, 0, tensor->size());
            tensor->unmap(MNN::Tensor::MAP_TENSOR_WRITE, tensor->getDimensionType(), ptr);
        }
    }
    if (error.empty()) {
        for (int i = 0; i < options.modelPmuWarmupRuns; ++i) {
            if (net->runSession(session) != MNN::NO_ERROR) {
                error = "MNN warmup Session run failed";
                break;
            }
        }
        if (error.empty()) {
            const auto& outputs = net->getSessionOutputAll(session);
            for (const auto& output : outputs) {
                if (output.second == nullptr || output.second->wait(MNN::Tensor::MAP_TENSOR_READ, true) != 0) {
                    error = "MNN warmup output synchronization failed";
                    break;
                }
            }
        }
    }

    rapidjson::Value counters(rapidjson::kArrayType);
#if defined(MNN_REPLAY_HAS_PERFCOUNTER)
    if (error.empty()) {
        const PmuSample control = sampleControl(options);
        const PmuSample workload = sampleWorkload(options, net.get(), session);
        if (!control.available || !workload.available) {
            error = !workload.error.empty() ? workload.error : control.error;
        } else {
            const ModelPmuDelta delta = makeModelPmuDelta(control.value, workload.value);
            rapidjson::Value counter(rapidjson::kObjectType);
            addString(counter, "name", options.perfCounterEvents, allocator);
            counter.AddMember("control_delta", control.value, allocator);
            counter.AddMember("workload_delta", workload.value, allocator);
            counter.AddMember("delta_metric", delta.delta, allocator);
            counter.AddMember("valid", delta.delta != 0, allocator);
            counter.AddMember("control_duration_ns", control.endNs >= control.startNs ? control.endNs - control.startNs : 0,
                              allocator);
            counter.AddMember("workload_duration_ns", workload.endNs >= workload.startNs ? workload.endNs - workload.startNs : 0,
                              allocator);
            counters.PushBack(counter, allocator);
            document["status"].SetString("ok", allocator);
        }
    }
#else
    error = "MNNPerfCounter was not compiled into replay_benchmark";
#endif
    addString(item, "error", error, allocator);
    addString(item, "status", error.empty() ? "ok" : "unavailable", allocator);
    item.AddMember("counters", counters, allocator);
    document["error"].SetString(error.c_str(), allocator);
    cases.PushBack(item, allocator);
    document.AddMember("cases", cases, allocator);
    if (!writeReport(options, document)) {
        std::cerr << "Can't write model PMU report: " << options.perfCounterOutput << std::endl;
        return false;
    }
    return true;
}

} // namespace Replay
} // namespace MNN
