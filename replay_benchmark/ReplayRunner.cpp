// MNN single Execution replay implementation.
#include "ReplayRunner.hpp"
#include "PerfCounterReport.hpp"

#if defined(MNN_REPLAY_HAS_PERFCOUNTER)
#include "MNNPerfCounter.hpp"
#endif

#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#if defined(MNN_REPLAY_HAS_OPENCL)
#include "backend/opencl/core/OpenCLBackend.hpp"
#endif

namespace MNN {
namespace Replay {

namespace {

static uint64_t monotonicNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}

static std::vector<std::string> splitCounterNames(const std::string& text) {
    std::vector<std::string> names;
    size_t begin = 0;
    while (begin < text.size()) {
        const size_t end = text.find(',', begin);
        const size_t length = end == std::string::npos ? text.size() - begin : end - begin;
        if (length != 0) {
            names.emplace_back(text.substr(begin, length));
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    if (names.empty()) {
        names = {"gpu_active_cycles", "compute_active_cycles", "compute_tasks", "l2_any_lookup",
                 "l2_ext_read", "l2_ext_write"};
    }
    return names;
}

#if defined(MNN_REPLAY_HAS_PERFCOUNTER)
static const char* perfVendorName(MNN::PerfCounter::GpuVendor vendor) {
    switch (vendor) {
        case MNN::PerfCounter::GpuVendor::Mali: return "Mali";
        case MNN::PerfCounter::GpuVendor::Adreno: return "Adreno";
        case MNN::PerfCounter::GpuVendor::Nvidia: return "NVIDIA";
        default: return "Unknown";
    }
}
#endif

} // namespace

bool replayOp(const Options& options) {
    RecordReader reader;
    std::string recordRoot = options.recordDir;
    if (!fileExists(joinPath(recordRoot, "record.json"))) {
        const auto slash = options.model.find_last_of("/\\");
        const auto modelName = slash == std::string::npos ? options.model : options.model.substr(slash + 1);
        const auto candidate = joinPath(recordRoot, modelName);
        if (fileExists(joinPath(candidate, "record.json"))) {
            recordRoot = candidate;
        }
    }
    if (!reader.load(recordRoot)) {
        return false;
    }
    recordRoot = reader.root();
    Options replayOptions = options;
    reader.applyRuntime(replayOptions);
#if defined(MNN_REPLAY_HAS_OPENCL) && defined(MNN_USE_LIB_WRAPPER)
    if (replayOptions.forward == MNN_FORWARD_OPENCL &&
        (MNN::OpenCLSymbolsOperator::createOpenCLSymbolsOperatorSingleInstance() == nullptr ||
         MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr() == nullptr)) {
        std::cerr << "OpenCL symbols are not available" << std::endl;
        return false;
    }
#elif !defined(MNN_REPLAY_HAS_OPENCL)
    if (replayOptions.forward == MNN_FORWARD_OPENCL) {
        std::cerr << "This build does not include OpenCL" << std::endl;
        return false;
    }
#endif
    const ReplayOpSpec* opSpec = reader.select(options);
    if (opSpec == nullptr) {
        std::cerr << "No unique op matched the selector" << std::endl;
        return false;
    }

    PerfCounterReport perfReport;
    perfReport.status = "unavailable";
    perfReport.model = options.model;
    perfReport.opId = opSpec->id;
    perfReport.opName = opSpec->name;
    perfReport.opType = opSpec->type;
    perfReport.execution = opSpec->execution;
    perfReport.variant = opSpec->variant;
    perfReport.backend = forwardName(replayOptions.forward);
    const bool perfRequested = !options.perfCounterOutput.empty();

    std::vector<uint8_t> modelBytes;
    if (!readBytes(options.model, modelBytes)) {
        std::cerr << "Can't read model: " << options.model << std::endl;
        return false;
    }
    ModelOpDatabase modelDatabase(modelBytes.data());
    const MNN::Op* modelOp = opSpec->name.empty() ? nullptr : modelDatabase.find(opSpec->name);

    std::vector<uint8_t> opBytes;
    const MNN::Op* op = nullptr;
    if (!opSpec->opFile.empty()) {
        if (!readBytes(joinPath(recordRoot, opSpec->opFile), opBytes)) {
            std::cerr << "Can't read recorded op: " << opSpec->opFile << std::endl;
            return false;
        }
        op = flatbuffers::GetRoot<MNN::Op>(opBytes.data());
    } else {
        op = modelOp;
    }
    if (op == nullptr) {
        std::cerr << "Selected op has no standalone op.fb and is not present in the model" << std::endl;
        return false;
    }
    if (modelOp != nullptr && modelOp->type() != op->type()) {
        std::cerr << "Recorded op type does not match the selected model op" << std::endl;
        return false;
    }

    BackendConfig backendConfig;
    backendConfig.precision = static_cast<BackendConfig::PrecisionMode>(replayOptions.precision);
    Backend::Info info;
    info.type = static_cast<MNNForwardType>(replayOptions.forward);
    info.numThread = replayOptions.gpuMode;
    info.user = &backendConfig;
    const MNNForwardType runtimeType = replayOptions.forward == MNN_FORWARD_CPU_EXTENSION
                                           ? MNN_FORWARD_CPU
                                           : static_cast<MNNForwardType>(replayOptions.forward);
    info.type = runtimeType;
    const auto* creator = MNN::MNNGetExtraRuntimeCreator(runtimeType);
    if (creator == nullptr) {
        std::cerr << "Requested runtime is not available: " << forwardName(replayOptions.forward) << std::endl;
        return false;
    }
    std::unique_ptr<Runtime> runtime(creator->onCreate(info));
    if (runtime == nullptr) {
        return false;
    }
    std::unique_ptr<Backend> backend(runtime->onCreate(&backendConfig, nullptr));
    if (backend == nullptr) {
        return false;
    }

    std::vector<std::unique_ptr<Tensor>> hostInputs;
    std::vector<std::unique_ptr<Tensor>> deviceInputs;
    std::vector<Tensor*> inputPointers;
    std::map<std::string, Tensor*> tensorsById;
    const bool cpuBackend = replayOptions.forward == MNN_FORWARD_CPU ||
                            replayOptions.forward == MNN_FORWARD_CPU_EXTENSION;
    const auto replayTensorStorage = cpuBackend ? Backend::STATIC : Backend::DYNAMIC_SEPERATE;
    for (const auto& spec : opSpec->inputs) {
        const auto type = typeFromName(spec.dtype);
        std::unique_ptr<Tensor> host(Tensor::create(spec.shape, type, nullptr, Tensor::CAFFE));
        std::vector<uint8_t> bytes;
        if (!readBytes(joinPath(recordRoot, spec.logicalFile), bytes) || host == nullptr ||
            host->size() != bytes.size()) {
            std::cerr << "Invalid input tensor: " << spec.id << std::endl;
            return false;
        }
        std::memcpy(host->buffer().host, bytes.data(), bytes.size());
        std::unique_ptr<Tensor> device(Tensor::createDevice(spec.shape, type, storageDimensionType(spec)));
        MNN::TensorUtils::setTensorChannelPack(device.get(), spec.channelPack);
        inputPointers.emplace_back(device.get());
        tensorsById[spec.id] = device.get();
        hostInputs.emplace_back(std::move(host));
        deviceInputs.emplace_back(std::move(device));
    }

    std::vector<std::unique_ptr<Tensor>> deviceOutputs;
    std::vector<Tensor*> outputPointers;
    for (const auto& spec : opSpec->outputs) {
        const auto type = typeFromName(spec.dtype);
        std::unique_ptr<Tensor> device(Tensor::createDevice(spec.shape, type, storageDimensionType(spec)));
        MNN::TensorUtils::setTensorChannelPack(device.get(), spec.channelPack);
        outputPointers.emplace_back(device.get());
        tensorsById[spec.id] = device.get();
        deviceOutputs.emplace_back(std::move(device));
    }

    for (size_t i = 0; i < opSpec->inputs.size(); ++i) {
        if (!restoreTensorRegions(inputPointers[i], opSpec->inputs[i], tensorsById, true)) {
            return false;
        }
    }
    for (size_t i = 0; i < opSpec->outputs.size(); ++i) {
        if (!restoreTensorRegions(outputPointers[i], opSpec->outputs[i], tensorsById, false)) {
            return false;
        }
    }

    std::unique_ptr<Execution> execution(backend->onCreate(inputPointers, outputPointers, op));
    if ((execution == nullptr || !execution->valid()) && replayOptions.forward == MNN_FORWARD_CPU_EXTENSION) {
        // ARM82 intentionally delegates unsupported low-precision ops to the
        // generic CPU backend in a normal Session. Mirror that fallback when
        // replaying an isolated recorded Execution.
        BackendConfig fallbackConfig = backendConfig;
        fallbackConfig.precision = BackendConfig::Precision_Normal;
        std::unique_ptr<Backend> fallbackBackend(runtime->onCreate(&fallbackConfig, nullptr));
        if (fallbackBackend != nullptr) {
            backend = std::move(fallbackBackend);
            execution.reset(backend->onCreate(inputPointers, outputPointers, op));
        }
    }
    if (execution == nullptr || !execution->valid()) {
        std::cerr << "Backend::onCreate failed for op " << opSpec->id << std::endl;
        return false;
    }

    for (size_t i = 0; i < opSpec->inputs.size(); ++i) {
        if (!backend->onAcquireBuffer(inputPointers[i], replayTensorStorage) ||
            !inputPointers[i]->copyFromHostTensor(hostInputs[i].get())) {
            return false;
        }
    }
    for (auto* output : outputPointers) {
        if (!backend->onAcquireBuffer(output, replayTensorStorage)) {
            return false;
        }
    }

#if defined(MNN_REPLAY_HAS_OPENCL)
    auto* openclBackend =
        replayOptions.forward == MNN_FORWARD_OPENCL ? static_cast<MNN::OpenCL::OpenCLBackend*>(backend.get()) : nullptr;
#endif
    backend->onResizeBegin();
    backend->onExecutionResizeBegin(op, execution.get());
    const auto resizeCode = execution->onResize(inputPointers, outputPointers);
    backend->onExecutionResizeEnd(op, execution.get());
    const auto resizeEndCode = backend->onResizeEnd();
    if (resizeCode != MNN::NO_ERROR || resizeEndCode != MNN::NO_ERROR) {
        std::cerr << "Execution resize failed" << std::endl;
        return false;
    }

#if defined(MNN_REPLAY_HAS_PERFCOUNTER)
    std::unique_ptr<MNN::PerfCounter::Session> perfSession;
    std::vector<std::string> counterNames;
    std::vector<MNN::PerfCounter::CounterSpec> counterSpecs;
    std::vector<MNN::PerfCounter::CounterValue> counterValues;
    MNN::PerfCounter::DeviceInfo perfDevice;
    if (perfRequested) {
        counterNames = splitCounterNames(options.perfCounterEvents);
        counterSpecs.resize(counterNames.size());
        for (size_t i = 0; i < counterNames.size(); ++i) {
            counterSpecs[i].name = counterNames[i].c_str();
        }
        const char* perfError = nullptr;
        perfSession.reset(MNN::PerfCounter::Session::create(counterSpecs.data(), counterSpecs.size(), &perfDevice,
                                                            &perfError));
        if (perfSession == nullptr) {
            perfReport.error = perfError == nullptr ? "Performance counter session creation failed" : perfError;
        } else {
            perfReport.vendor = perfVendorName(perfDevice.vendor);
            perfReport.productId = perfDevice.productId;
            perfReport.productName = perfDevice.productName == nullptr ? "" : perfDevice.productName;
            perfReport.driver = perfDevice.driverName == nullptr ? "" : perfDevice.driverName;
        }
    }
#else
    if (perfRequested) {
        perfReport.error = "MNNPerfCounter was not compiled into replay_benchmark";
    }
#endif

    if (!opSpec->execution.empty() && execution->getExecutionName() != nullptr &&
        opSpec->execution != execution->getExecutionName()) {
        std::cerr << "Execution mismatch: actual=" << execution->getExecutionName() << " expected=" << opSpec->execution
                  << std::endl;
        return false;
    }
#if defined(MNN_REPLAY_HAS_OPENCL)
    if (openclBackend != nullptr && !opSpec->variant.empty()) {
        const auto& traces = openclBackend->getReplayExecutionInfo();
        if (!traces.empty() && !traces.front().kernels.empty() &&
            traces.front().kernels.front().kernelName != opSpec->variant) {
            std::cerr << "Kernel variant mismatch: actual=" << traces.front().kernels.front().kernelName
                      << " expected=" << opSpec->variant << std::endl;
            return false;
        }
    }
#else
    if (!opSpec->variant.empty()) {
        std::cerr << "CPU replay record does not contain an OpenCL kernel variant" << std::endl;
        return false;
    }
#endif

    bool replayOk = true;
#if defined(MNN_REPLAY_HAS_PERFCOUNTER)
    bool perfStarted = false;
    if (perfSession != nullptr) {
        if (!perfSession->start()) {
            perfReport.error = perfSession->error();
            perfSession.reset();
        } else {
            perfStarted = true;
            perfReport.startNs = monotonicNs();
        }
    }
#endif
    backend->onExecuteBegin();
    const auto executeCode = execution->onExecute(inputPointers, outputPointers);
    backend->onExecuteEnd();
    if (executeCode != MNN::NO_ERROR) {
        std::cerr << "Execution failed: " << executeCode << std::endl;
        replayOk = false;
        perfReport.error = "Execution failed";
    }
#if defined(MNN_REPLAY_HAS_PERFCOUNTER)
    if (perfStarted) {
        bool synchronized = true;
        for (auto* output : outputPointers) {
            if (output == nullptr || output->wait(Tensor::MAP_TENSOR_READ, true) != 0) {
                synchronized = false;
                break;
            }
        }
        perfReport.synchronized = synchronized;
        counterValues.resize(counterSpecs.size());
        if (!synchronized || !perfSession->stop(counterValues.data(), counterValues.size())) {
            perfReport.error = synchronized ? perfSession->error() : "OpenCL output synchronization failed";
            replayOk = false;
        } else {
            perfReport.status = "ok";
            for (size_t i = 0; i < counterValues.size(); ++i) {
                PerfCounterValueRecord value;
                value.name = counterNames[i];
                value.integerValue = counterValues[i].value;
                value.floatingPointValue = counterValues[i].floatingPointValue;
                value.valueKind = counterValues[i].valueKind == MNN::PerfCounter::CounterValueKind::FloatingPoint
                                      ? PerfCounterValueKind::FloatingPoint
                                      : PerfCounterValueKind::UnsignedInteger;
                value.status = counterValues[i].status == MNN::PerfCounter::CounterValueStatus::Valid
                                   ? PerfCounterValueStatus::Valid
                                   : (counterValues[i].status == MNN::PerfCounter::CounterValueStatus::Overflow
                                          ? PerfCounterValueStatus::Overflow
                                          : PerfCounterValueStatus::Invalid);
                perfReport.counters.emplace_back(std::move(value));
            }
        }
        perfReport.endNs = monotonicNs();
    }
#endif
    for (int i = 0; i < static_cast<int>(outputPointers.size()); ++i) {
        if (!compareTensor(outputPointers[i], joinPath(recordRoot, opSpec->outputs[i].logicalFile))) {
            replayOk = false;
            if (perfRequested && perfReport.error.empty()) {
                perfReport.error = "Output comparison failed";
            }
        }
    }
    if (perfRequested && perfReport.status != "ok" && perfReport.error.empty()) {
        perfReport.error = "Performance counter collection was unavailable";
    }
    if (perfRequested && !perfReport.write(options.perfCounterOutput)) {
        std::cerr << "Can't write performance counter report: " << options.perfCounterOutput << std::endl;
        replayOk = false;
    }
    if (!replayOk) {
        return false;
    }
    std::cout << "Replay succeeded: op_id=" << opSpec->id << " type=" << opSpec->type
              << " execution=" << (execution->getExecutionName() == nullptr ? "" : execution->getExecutionName())
              << std::endl;
    return true;
}

} // namespace Replay
} // namespace MNN
