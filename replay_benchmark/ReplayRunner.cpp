// MNN single Execution replay implementation.
#include "ReplayRunner.hpp"

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
    const auto* creator = MNN::MNNGetExtraRuntimeCreator(info.type);
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
    const auto replayTensorStorage =
        replayOptions.forward == MNN_FORWARD_CPU ? Backend::STATIC : Backend::DYNAMIC_SEPERATE;
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
        if (!backend->onAcquireBuffer(device.get(), replayTensorStorage) || !device->copyFromHostTensor(host.get())) {
            return false;
        }
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
        if (!backend->onAcquireBuffer(device.get(), replayTensorStorage)) {
            return false;
        }
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
    if (execution == nullptr || !execution->valid()) {
        std::cerr << "Backend::onCreate failed for op " << opSpec->id << std::endl;
        return false;
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

    backend->onExecuteBegin();
    const auto executeCode = execution->onExecute(inputPointers, outputPointers);
    backend->onExecuteEnd();
    if (executeCode != MNN::NO_ERROR) {
        std::cerr << "Execution failed: " << executeCode << std::endl;
        return false;
    }
    for (int i = 0; i < static_cast<int>(outputPointers.size()); ++i) {
        if (!compareTensor(outputPointers[i], joinPath(recordRoot, opSpec->outputs[i].logicalFile))) {
            return false;
        }
    }
    std::cout << "Replay succeeded: op_id=" << opSpec->id << " type=" << opSpec->type
              << " execution=" << (execution->getExecutionName() == nullptr ? "" : execution->getExecutionName())
              << std::endl;
    return true;
}

} // namespace Replay
} // namespace MNN
