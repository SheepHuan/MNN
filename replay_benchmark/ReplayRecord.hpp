// MNN execution record format, serialization and tensor snapshot utilities.
#ifndef MNN_REPLAY_RECORD_HPP
#define MNN_REPLAY_RECORD_HPP

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include "MNN_generated.h"
#include "core/Backend.hpp"
#include "core/Execution.hpp"
#include "core/TensorUtils.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace MNN {
namespace Replay {

using MNN::Backend;
using MNN::BackendConfig;
using MNN::Execution;
using MNN::Runtime;
using MNN::Tensor;

struct Options {
    std::string model;
    std::string recordDir;
    std::string argsFile;
    bool record = false;
    int loop = 10;
    int warmup = 10;
    int forward = MNN_FORWARD_CPU;
    int gpuMode = 4;
    int precision = BackendConfig::Precision_Low;
    float sparsity = 0.0f;
    int sparseBlockOC = 1;
    int testQuantizedModel = 0;
    bool enableKleidiAI = false;
    bool replay = false;
    int opId = -1;
    std::string opType;
    std::string execution;
    std::string variant;
    std::string perfCounterOutput;
    std::string perfCounterEvents;
    bool openclPmuBench = false;
    std::string openclPmuCase = "all";
    int openclPmuIterations = 100;
    int openclPmuWorkloadRuns = 1;
    int openclPmuWarmupRuns = 2;
    size_t openclPmuSize = 0;
    size_t openclPmuLocalSize = 0;
};

struct TensorRecord {
    std::string id;
    std::string role;
    std::string dtype;
    std::string dimensionType;
    int dimensionFormat = 0;
    int channelPack = 0;
    std::vector<int> shape;
    std::string logicalFile;
    std::string storageFile;
    size_t logicalBytes = 0;
    size_t storageBytes = 0;
    struct Region {
        std::string origin;
        int srcOffset = 0;
        int dstOffset = 0;
        int srcStride[3] = {1, 1, 1};
        int dstStride[3] = {1, 1, 1};
        int size[3] = {1, 1, 1};
    };
    std::vector<Region> regions;
};

struct KernelRecord {
    std::string program;
    std::string kernel;
    std::vector<uint32_t> gws;
    std::vector<uint32_t> lws;
};

struct OpRecord {
    int id = -1;
    std::string name;
    std::string type;
    std::string opFile;
    std::string backend;
    std::string execution;
    std::string variant;
    std::vector<TensorRecord> inputs;
    std::vector<TensorRecord> outputs;
    std::vector<KernelRecord> kernels;
};

struct ExecutionTrace {
    const Execution* execution = nullptr;
    std::string executionName;
    std::string variant;
    std::vector<KernelRecord> kernels;
};

struct ModelFile {
    std::string name;
    std::string path;
};

bool isDirectory(const std::string& path);
bool fileExists(const std::string& path);
std::string joinPath(const std::string& root, const std::string& name);
bool writeBytes(const std::string& path, const void* data, size_t size);
bool readBytes(const std::string& path, std::vector<uint8_t>& data);
std::string typeName(const halide_type_t& type);
halide_type_t typeFromName(const std::string& name);
std::string dimensionTypeName(Tensor::DimensionType type);
Tensor::DimensionType dimensionTypeFromName(const std::string& name);
std::string forwardName(int forward);
std::string formatName(int format);
bool isFloatType(const halide_type_t& type);
std::unique_ptr<Tensor> hostCopy(const Tensor* tensor, Tensor::DimensionType dimensionType);
bool makeRecordDirectories(const std::string& root);
bool saveOp(const std::string& path, const MNN::Op* op);
std::string tensorFileStem(int opId, const char* role, int index);

class ModelOpDatabase {
public:
    explicit ModelOpDatabase(const void* buffer) {
        mNet = flatbuffers::GetRoot<MNN::Net>(buffer);
        if (mNet == nullptr || mNet->oplists() == nullptr) {
            return;
        }
        for (flatbuffers::uoffset_t i = 0; i < mNet->oplists()->size(); ++i) {
            const auto* op = mNet->oplists()->GetAs<MNN::Op>(i);
            if (op != nullptr && op->name() != nullptr) {
                mNamedOps[op->name()->str()] = op;
            }
        }
    }

    const MNN::Op* find(const std::string& name) const {
        auto iter = mNamedOps.find(name);
        return iter == mNamedOps.end() ? nullptr : iter->second;
    }

private:
    const MNN::Net* mNet = nullptr;
    std::map<std::string, const MNN::Op*> mNamedOps;
};

class RecordWriter {
public:
    RecordWriter(const std::string& root, const Options& options, const ModelOpDatabase& database,
                 const std::string& modelPath)
        : mRoot(root), mOptions(options), mDatabase(database), mModelPath(modelPath) {
        makeRecordDirectories(root);
    }

    bool begin(const std::vector<Tensor*>& inputs, const MNN::OperatorInfo* info,
               const std::vector<ExecutionTrace>& executionInfo) {
        if (info == nullptr) {
            return false;
        }
        mPending = OpRecord();
        mPending.id = static_cast<int>(mOps.size());
        mPending.name = info->name();
        mPending.type = info->type();
        mPending.backend = forwardName(mOptions.forward);

        const ExecutionTrace* matched = findExecutionInfo(info->execution(), executionInfo);
        const auto* modelOp = mDatabase.find(mPending.name);
        const auto* recordedOp = modelOp != nullptr ? modelOp : info->op();
        if (recordedOp != nullptr) {
            std::ostringstream name;
            name << "ops/" << std::setfill('0') << std::setw(6) << mPending.id << ".op.fb";
            mPending.opFile = name.str();
            if (!saveOp(joinPath(mRoot, mPending.opFile), recordedOp)) {
                mPending.opFile.clear();
            }
        }

        if (matched != nullptr) {
            mPending.execution = matched->executionName;
            mPending.variant = matched->variant;
            mPending.kernels = matched->kernels;
        } else if (info->execution() != nullptr && info->execution()->getExecutionName() != nullptr) {
            mPending.execution = info->execution()->getExecutionName();
        }

        for (int i = 0; i < static_cast<int>(inputs.size()); ++i) {
            const std::string id = tensorFileStem(mPending.id, "input", i);
            mTensorIds[inputs[i]] = id;
        }
        for (int i = 0; i < static_cast<int>(inputs.size()); ++i) {
            const std::string id = tensorFileStem(mPending.id, "input", i);
            mPending.inputs.emplace_back(dumpTensor(inputs[i], id, "input", i));
        }
        return true;
    }

    bool finish(const std::vector<Tensor*>& outputs) {
        for (int i = 0; i < static_cast<int>(outputs.size()); ++i) {
            const std::string id = tensorFileStem(mPending.id, "output", i);
            if (mTensorIds.find(outputs[i]) == mTensorIds.end()) {
                mTensorIds[outputs[i]] = id;
            }
        }
        for (int i = 0; i < static_cast<int>(outputs.size()); ++i) {
            const std::string id = tensorFileStem(mPending.id, "output", i);
            mPending.outputs.emplace_back(dumpTensor(outputs[i], id, "output", i));
        }
        mTensorIds.clear();
        mOps.emplace_back(std::move(mPending));
        return true;
    }

    bool write() const {
        rapidjson::Document document;
        document.SetObject();
        auto& allocator = document.GetAllocator();
        addString(document, "format", "mnn-execution-record", allocator);
        document.AddMember("version", 1, allocator);
        rapidjson::Value model(rapidjson::kObjectType);
        addString(model, "path", mModelPath, allocator);
        addString(model, "mnn_version", MNN::getVersion(), allocator);
        document.AddMember("model", model, allocator);
        rapidjson::Value runtime(rapidjson::kObjectType);
        addString(runtime, "backend", forwardName(mOptions.forward), allocator);
        runtime.AddMember("forward", mOptions.forward, allocator);
        runtime.AddMember("gpu_mode", mOptions.gpuMode, allocator);
        runtime.AddMember("precision", mOptions.precision, allocator);
        runtime.AddMember("memory", 0, allocator);
        document.AddMember("runtime", runtime, allocator);

        rapidjson::Value ops(rapidjson::kArrayType);
        for (const auto& op : mOps) {
            rapidjson::Value value(rapidjson::kObjectType);
            value.AddMember("op_id", op.id, allocator);
            addString(value, "name", op.name, allocator);
            addString(value, "op_type", op.type, allocator);
            addString(value, "op_file", op.opFile, allocator);
            addString(value, "backend", op.backend, allocator);
            addString(value, "execution", op.execution, allocator);
            addString(value, "variant", op.variant, allocator);
            addTensorArray(value, "inputs", op.inputs, allocator);
            addTensorArray(value, "outputs", op.outputs, allocator);
            rapidjson::Value kernels(rapidjson::kArrayType);
            for (const auto& kernel : op.kernels) {
                rapidjson::Value item(rapidjson::kObjectType);
                addString(item, "program", kernel.program, allocator);
                addString(item, "kernel", kernel.kernel, allocator);
                addUIntArray(item, "gws", kernel.gws, allocator);
                addUIntArray(item, "lws", kernel.lws, allocator);
                kernels.PushBack(item, allocator);
            }
            value.AddMember("kernels", kernels, allocator);
            ops.PushBack(value, allocator);
        }
        document.AddMember("ops", ops, allocator);

        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        document.Accept(writer);
        return writeBytes(joinPath(mRoot, "record.json"), buffer.GetString(), buffer.GetSize());
    }

private:
    static void addString(rapidjson::Value& object, const char* key, const std::string& value,
                          rapidjson::Document::AllocatorType& allocator) {
        rapidjson::Value stringValue;
        stringValue.SetString(value.c_str(), static_cast<rapidjson::SizeType>(value.size()), allocator);
        object.AddMember(rapidjson::Value(key, allocator), stringValue, allocator);
    }

    static void addString(rapidjson::Document& document, const char* key, const std::string& value,
                          rapidjson::Document::AllocatorType& allocator) {
        addString(static_cast<rapidjson::Value&>(document), key, value, allocator);
    }

    static void addIntArray(rapidjson::Value& object, const char* key, const std::vector<int>& values,
                            rapidjson::Document::AllocatorType& allocator) {
        rapidjson::Value array(rapidjson::kArrayType);
        for (int value : values) {
            array.PushBack(value, allocator);
        }
        object.AddMember(rapidjson::Value(key, allocator), array, allocator);
    }

    static void addUIntArray(rapidjson::Value& object, const char* key, const std::vector<uint32_t>& values,
                             rapidjson::Document::AllocatorType& allocator) {
        rapidjson::Value array(rapidjson::kArrayType);
        for (uint32_t value : values) {
            array.PushBack(value, allocator);
        }
        object.AddMember(rapidjson::Value(key, allocator), array, allocator);
    }

    static void addTensorArray(rapidjson::Value& object, const char* key, const std::vector<TensorRecord>& tensors,
                               rapidjson::Document::AllocatorType& allocator) {
        rapidjson::Value array(rapidjson::kArrayType);
        for (const auto& tensor : tensors) {
            rapidjson::Value item(rapidjson::kObjectType);
            addString(item, "id", tensor.id, allocator);
            addString(item, "role", tensor.role, allocator);
            addString(item, "dtype", tensor.dtype, allocator);
            addString(item, "dimension_type", tensor.dimensionType, allocator);
            addString(item, "dimension_format", formatName(tensor.dimensionFormat), allocator);
            item.AddMember("dimension_format_id", tensor.dimensionFormat, allocator);
            item.AddMember("channel_pack", tensor.channelPack, allocator);
            addIntArray(item, "shape", tensor.shape, allocator);
            addString(item, "logical_file", tensor.logicalFile, allocator);
            addString(item, "storage_file", tensor.storageFile, allocator);
            item.AddMember("logical_bytes", static_cast<uint64_t>(tensor.logicalBytes), allocator);
            item.AddMember("storage_bytes", static_cast<uint64_t>(tensor.storageBytes), allocator);
            rapidjson::Value regions(rapidjson::kArrayType);
            for (const auto& region : tensor.regions) {
                rapidjson::Value regionValue(rapidjson::kObjectType);
                addString(regionValue, "origin", region.origin, allocator);
                regionValue.AddMember("src_offset", region.srcOffset, allocator);
                regionValue.AddMember("dst_offset", region.dstOffset, allocator);
                addIntArray(regionValue, "src_stride", std::vector<int>(region.srcStride, region.srcStride + 3),
                            allocator);
                addIntArray(regionValue, "dst_stride", std::vector<int>(region.dstStride, region.dstStride + 3),
                            allocator);
                addIntArray(regionValue, "size", std::vector<int>(region.size, region.size + 3), allocator);
                regions.PushBack(regionValue, allocator);
            }
            item.AddMember("regions", regions, allocator);
            array.PushBack(item, allocator);
        }
        object.AddMember(rapidjson::Value(key, allocator), array, allocator);
    }

    const ExecutionTrace* findExecutionInfo(const Execution* execution,
                                            const std::vector<ExecutionTrace>& executionInfo) const {
        if (execution != nullptr) {
            for (const auto& info : executionInfo) {
                if (info.execution == execution) {
                    return &info;
                }
            }
        }
        return nullptr;
    }

    TensorRecord dumpTensor(const Tensor* tensor, const std::string& id, const std::string& role, int index) const {
        TensorRecord result;
        result.id = id;
        result.role = role;
        if (tensor == nullptr) {
            return result;
        }
        result.dtype = typeName(tensor->getType());
        result.dimensionType = dimensionTypeName(tensor->getDimensionType());
        result.shape = tensor->shape();
        auto* describe = MNN::TensorUtils::getDescribe(tensor);
        result.dimensionFormat = static_cast<int>(describe->dimensionFormat);
        result.channelPack = MNN::TensorUtils::getTensorChannelPack(tensor);

        auto logical = hostCopy(tensor, Tensor::CAFFE);
        auto storage = hostCopy(tensor, MNN::TensorUtils::getDimType(tensor));
        const std::string logicalName = "tensors/" + id + ".logical.bin";
        const std::string storageName = "tensors/" + id + ".storage.bin";
        result.logicalFile = logicalName;
        result.storageFile = storageName;
        if (logical != nullptr) {
            result.logicalBytes = logical->size();
            writeBytes(joinPath(mRoot, logicalName), logical->buffer().host, logical->size());
        }
        if (storage != nullptr) {
            result.storageBytes = storage->size();
            writeBytes(joinPath(mRoot, storageName), storage->buffer().host, storage->size());
        }
        for (const auto& region : describe->regions) {
            TensorRecord::Region regionRecord;
            const auto origin = mTensorIds.find(region.origin);
            // A region may refer to a Tensor from a previous Command. Its
            // current Tensor already has a complete snapshot, so retaining a
            // region with an unresolved origin would make the record invalid.
            if (origin == mTensorIds.end()) {
                continue;
            }
            regionRecord.origin = origin->second;
            regionRecord.srcOffset = region.src.offset;
            regionRecord.dstOffset = region.dst.offset;
            for (int i = 0; i < 3; ++i) {
                regionRecord.srcStride[i] = region.src.stride[i];
                regionRecord.dstStride[i] = region.dst.stride[i];
                regionRecord.size[i] = region.size[i];
            }
            result.regions.emplace_back(std::move(regionRecord));
        }
        return result;
    }

    std::string mRoot;
    const Options& mOptions;
    const ModelOpDatabase& mDatabase;
    std::string mModelPath;
    std::vector<OpRecord> mOps;
    OpRecord mPending;
    std::map<const Tensor*, std::string> mTensorIds;
};

struct ReplayTensorSpec {
    struct Region {
        std::string origin;
        int srcOffset = 0;
        int dstOffset = 0;
        int srcStride[3] = {1, 1, 1};
        int dstStride[3] = {1, 1, 1};
        int size[3] = {1, 1, 1};
    };
    std::string id;
    std::string dtype;
    std::string dimensionType;
    int dimensionFormat = MNN::MNN_DATA_FORMAT_NCHW;
    int channelPack = 4;
    std::vector<int> shape;
    std::string logicalFile;
    std::string storageFile;
    std::vector<Region> regions;
};

static Tensor::DimensionType storageDimensionType(const ReplayTensorSpec& spec) {
    switch (spec.dimensionFormat) {
        case MNN::MNN_DATA_FORMAT_NC4HW4:
            return Tensor::CAFFE_C4;
        case MNN::MNN_DATA_FORMAT_NHWC:
        case MNN::MNN_DATA_FORMAT_NHWC4:
            return Tensor::TENSORFLOW;
        case MNN::MNN_DATA_FORMAT_NCHW:
        default:
            return Tensor::CAFFE;
    }
}

struct ReplayOpSpec {
    int id = -1;
    std::string name;
    std::string type;
    std::string opFile;
    std::string backend;
    std::string execution;
    std::string variant;
    std::vector<ReplayTensorSpec> inputs;
    std::vector<ReplayTensorSpec> outputs;
};

class RecordReader {
public:
    bool load(const std::string& root) {
        mRoot = root;
        std::vector<uint8_t> bytes;
        if (!readBytes(joinPath(root, "record.json"), bytes)) {
            std::cerr << "Can't read record: " << joinPath(root, "record.json") << std::endl;
            return false;
        }
        rapidjson::Document document;
        document.Parse(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        if (document.HasParseError() || !document.IsObject()) {
            std::cerr << "Invalid record.json" << std::endl;
            return false;
        }
        if (!document.HasMember("format") || !document["format"].IsString() ||
            std::string(document["format"].GetString()) != "mnn-execution-record") {
            std::cerr << "Unsupported record format" << std::endl;
            return false;
        }
        if (!document.HasMember("version") || !document["version"].IsInt() || document["version"].GetInt() != 1) {
            std::cerr << "Unsupported record version" << std::endl;
            return false;
        }
        if (!document.HasMember("ops")) {
            std::cerr << "Invalid record.json: missing ops" << std::endl;
            return false;
        }
        if (document.HasMember("runtime") && document["runtime"].IsObject()) {
            const auto& runtime = document["runtime"];
            if (runtime.HasMember("forward") && runtime["forward"].IsInt()) {
                mForward = runtime["forward"].GetInt();
            }
            if (runtime.HasMember("gpu_mode") && runtime["gpu_mode"].IsInt()) {
                mGpuMode = runtime["gpu_mode"].GetInt();
            }
            if (runtime.HasMember("precision") && runtime["precision"].IsInt()) {
                mPrecision = runtime["precision"].GetInt();
            }
        }
        const auto& ops = document["ops"];
        if (!ops.IsArray()) {
            return false;
        }
        for (const auto& value : ops.GetArray()) {
            ReplayOpSpec op;
            op.id = value.HasMember("op_id") ? value["op_id"].GetInt() : -1;
            op.name = getString(value, "name");
            op.type = getString(value, "op_type");
            op.opFile = getString(value, "op_file");
            op.backend = getString(value, "backend");
            op.execution = getString(value, "execution");
            op.variant = getString(value, "variant");
            parseTensors(value, "inputs", op.inputs);
            parseTensors(value, "outputs", op.outputs);
            mOps.emplace_back(std::move(op));
        }
        return true;
    }

    const ReplayOpSpec* select(const Options& options) const {
        const ReplayOpSpec* selected = nullptr;
        for (const auto& op : mOps) {
            if (options.opId >= 0 && op.id != options.opId) {
                continue;
            }
            if (!options.opType.empty() && op.type != options.opType) {
                continue;
            }
            if (!options.execution.empty() && op.execution != options.execution) {
                continue;
            }
            if (!options.variant.empty() && op.variant != options.variant) {
                continue;
            }
            if (selected != nullptr) {
                std::cerr << "Selector matches more than one op" << std::endl;
                return nullptr;
            }
            selected = &op;
        }
        return selected;
    }

    const std::string& root() const { return mRoot; }

    void applyRuntime(Options& options) const {
        if (mForward >= 0) {
            options.forward = mForward;
        }
        if (mGpuMode >= 0) {
            options.gpuMode = mGpuMode;
        }
        if (mPrecision >= 0) {
            options.precision = mPrecision;
        }
    }

private:
    static std::string getString(const rapidjson::Value& object, const char* key) {
        return object.HasMember(key) && object[key].IsString() ? object[key].GetString() : "";
    }

    static void parseIntArray(const rapidjson::Value& object, const char* key, int* values) {
        if (!object.HasMember(key) || !object[key].IsArray()) {
            return;
        }
        const auto& array = object[key];
        for (rapidjson::SizeType i = 0; i < array.Size() && i < 3; ++i) {
            if (array[i].IsInt()) {
                values[i] = array[i].GetInt();
            }
        }
    }

    static void parseTensors(const rapidjson::Value& object, const char* key, std::vector<ReplayTensorSpec>& result) {
        if (!object.HasMember(key) || !object[key].IsArray()) {
            return;
        }
        for (const auto& value : object[key].GetArray()) {
            ReplayTensorSpec tensor;
            tensor.id = getString(value, "id");
            tensor.dtype = getString(value, "dtype");
            tensor.dimensionType = getString(value, "dimension_type");
            if (value.HasMember("dimension_format_id") && value["dimension_format_id"].IsInt()) {
                tensor.dimensionFormat = value["dimension_format_id"].GetInt();
            }
            if (value.HasMember("channel_pack") && value["channel_pack"].IsInt()) {
                tensor.channelPack = value["channel_pack"].GetInt();
            }
            tensor.logicalFile = getString(value, "logical_file");
            tensor.storageFile = getString(value, "storage_file");
            if (value.HasMember("shape") && value["shape"].IsArray()) {
                for (const auto& dim : value["shape"].GetArray()) {
                    tensor.shape.emplace_back(dim.GetInt());
                }
            }
            if (value.HasMember("regions") && value["regions"].IsArray()) {
                for (const auto& regionValue : value["regions"].GetArray()) {
                    ReplayTensorSpec::Region region;
                    region.origin = getString(regionValue, "origin");
                    if (regionValue.HasMember("src_offset") && regionValue["src_offset"].IsInt()) {
                        region.srcOffset = regionValue["src_offset"].GetInt();
                    }
                    if (regionValue.HasMember("dst_offset") && regionValue["dst_offset"].IsInt()) {
                        region.dstOffset = regionValue["dst_offset"].GetInt();
                    }
                    parseIntArray(regionValue, "src_stride", region.srcStride);
                    parseIntArray(regionValue, "dst_stride", region.dstStride);
                    parseIntArray(regionValue, "size", region.size);
                    tensor.regions.emplace_back(std::move(region));
                }
            }
            result.emplace_back(std::move(tensor));
        }
    }

    std::string mRoot;
    int mForward = -1;
    int mGpuMode = -1;
    int mPrecision = -1;
    std::vector<ReplayOpSpec> mOps;
};

bool compareTensor(const Tensor* actual, const std::string& expectedPath, float atol = 1e-4f, float rtol = 1e-3f);
bool restoreTensorRegions(Tensor* tensor, const ReplayTensorSpec& spec, const std::map<std::string, Tensor*>& tensors,
                          bool allowSnapshotFallback);

} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_RECORD_HPP
