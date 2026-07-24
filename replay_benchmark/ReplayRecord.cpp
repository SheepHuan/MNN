// MNN execution record implementation.
#include "ReplayRecord.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>

#if defined(_MSC_VER)
#include <direct.h>
#else
#include <dirent.h>
#endif

namespace MNN {
namespace Replay {

bool isDirectory(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && (st.st_mode & S_IFDIR) != 0;
}

bool fileExists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && (st.st_mode & S_IFREG) != 0;
}

bool makeDirectory(const std::string& path) {
    if (path.empty() || isDirectory(path)) {
        return true;
    }
    const auto separator = path.find_last_of("/\\");
    if (separator != std::string::npos && separator > 0 && !(separator == 2 && path.size() > 1 && path[1] == ':')) {
        const auto parent = path.substr(0, separator);
        if (!makeDirectory(parent)) {
            return false;
        }
    }
#if defined(_MSC_VER)
    return _mkdir(path.c_str()) == 0 || isDirectory(path);
#else
    return mkdir(path.c_str(), 0755) == 0 || isDirectory(path);
#endif
}

bool makeRecordDirectories(const std::string& root) {
    return makeDirectory(root) && makeDirectory(root + "/ops") && makeDirectory(root + "/tensors");
}

std::string joinPath(const std::string& root, const std::string& name) {
    if (root.empty()) {
        return name;
    }
    if (root.back() == '/' || root.back() == '\\') {
        return root + name;
    }
    return root + "/" + name;
}

bool writeBytes(const std::string& path, const void* data, size_t size) {
    std::ofstream output(path.c_str(), std::ios::binary);
    if (!output.good()) {
        return false;
    }
    if (size > 0) {
        output.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    }
    return output.good();
}

bool readBytes(const std::string& path, std::vector<uint8_t>& data) {
    std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
    if (!input.good()) {
        return false;
    }
    const std::streamsize size = input.tellg();
    if (size < 0) {
        return false;
    }
    data.resize(static_cast<size_t>(size));
    input.seekg(0, std::ios::beg);
    if (size > 0) {
        input.read(reinterpret_cast<char*>(data.data()), size);
    }
    return input.good() || input.eof();
}

std::string typeName(const halide_type_t& type) {
    std::ostringstream stream;
    if (type.code == halide_type_float) {
        stream << "float";
    } else if (type.code == halide_type_bfloat) {
        stream << "bfloat";
    } else if (type.code == halide_type_uint && type.bits == 1) {
        stream << "bool";
    } else if (type.code == halide_type_int) {
        stream << "int";
    } else if (type.code == halide_type_uint) {
        stream << "uint";
    } else {
        stream << "unknown";
    }
    stream << static_cast<int>(type.bits);
    if (type.lanes != 1) {
        stream << "x" << static_cast<int>(type.lanes);
    }
    return stream.str();
}

halide_type_t typeFromName(const std::string& name) {
    if (name == "float16") {
        return halide_type_t{halide_type_float, 16, 1};
    }
    if (name == "float32") {
        return halide_type_of<float>();
    }
    if (name == "int8") {
        return halide_type_of<int8_t>();
    }
    if (name == "int16") {
        return halide_type_of<int16_t>();
    }
    if (name == "int32") {
        return halide_type_of<int32_t>();
    }
    if (name == "uint8") {
        return halide_type_of<uint8_t>();
    }
    if (name == "uint16") {
        return halide_type_of<uint16_t>();
    }
    if (name == "uint32") {
        return halide_type_of<uint32_t>();
    }
    if (name == "bfloat16") {
        return halide_type_t{halide_type_bfloat, 16, 1};
    }
    if (name == "bool1") {
        return halide_type_of<bool>();
    }
    return halide_type_of<float>();
}

std::string dimensionTypeName(Tensor::DimensionType type) {
    switch (type) {
        case Tensor::TENSORFLOW:
            return "TENSORFLOW";
        case Tensor::CAFFE_C4:
            return "CAFFE_C4";
        case Tensor::CAFFE:
        default:
            return "CAFFE";
    }
}

Tensor::DimensionType dimensionTypeFromName(const std::string& name) {
    if (name == "TENSORFLOW") {
        return Tensor::TENSORFLOW;
    }
    if (name == "CAFFE_C4") {
        return Tensor::CAFFE_C4;
    }
    return Tensor::CAFFE;
}

std::string forwardName(int forward) {
    switch (static_cast<MNNForwardType>(forward)) {
        case MNN_FORWARD_CPU:
            return "CPU";
        case MNN_FORWARD_OPENCL:
            return "OPENCL";
        case MNN_FORWARD_VULKAN:
            return "VULKAN";
        case MNN_FORWARD_METAL:
            return "METAL";
        default:
            return "UNKNOWN";
    }
}

std::string formatName(int format) {
    switch (format) {
        case MNN::MNN_DATA_FORMAT_NHWC:
            return "NHWC";
        case MNN::MNN_DATA_FORMAT_NC4HW4:
            return "NC4HW4";
        case MNN::MNN_DATA_FORMAT_NHWC4:
            return "NHWC4";
        case MNN::MNN_DATA_FORMAT_NCHW:
        default:
            return "NCHW";
    }
}

bool isFloatType(const halide_type_t& type) {
    return type.code == halide_type_float;
}

std::unique_ptr<Tensor> hostCopy(const Tensor* tensor, Tensor::DimensionType dimensionType) {
    std::unique_ptr<Tensor> host(Tensor::create(tensor->shape(), tensor->getType(), nullptr, dimensionType));
    if (host == nullptr) {
        return nullptr;
    }
    auto* describe = MNN::TensorUtils::getDescribeOrigin(tensor);
    if (describe->getBackend() != nullptr) {
        if (!tensor->copyToHostTensor(host.get())) {
            return nullptr;
        }
    } else if (tensor->buffer().host != nullptr) {
        std::memcpy(host->buffer().host, tensor->buffer().host, std::min<size_t>(host->size(), tensor->size()));
    }
    return host;
}
bool saveOp(const std::string& path, const MNN::Op* op) {
    if (op == nullptr) {
        return false;
    }
    std::unique_ptr<MNN::OpT> opT(op->UnPack());
    if (opT == nullptr) {
        return false;
    }
    flatbuffers::FlatBufferBuilder builder(1024);
    const auto offset = MNN::Op::Pack(builder, opT.get());
    builder.Finish(offset);
    return writeBytes(path, builder.GetBufferPointer(), builder.GetSize());
}

std::string tensorFileStem(int opId, const char* role, int index) {
    std::ostringstream stream;
    stream << "op_" << opId << "_" << role << "_" << index;
    return stream.str();
}
bool compareTensor(const Tensor* actual, const std::string& expectedPath, float atol, float rtol) {
    std::vector<uint8_t> expected;
    if (!readBytes(expectedPath, expected)) {
        std::cerr << "Can't read expected tensor: " << expectedPath << std::endl;
        return false;
    }
    auto host = hostCopy(actual, Tensor::CAFFE);
    if (host == nullptr || host->size() != expected.size()) {
        std::cerr << "Tensor size mismatch, actual=" << (host == nullptr ? 0 : host->size())
                  << " expected=" << expected.size() << std::endl;
        return false;
    }
    const auto type = host->getType();
    if (!isFloatType(type) || type.bits != 32) {
        return std::memcmp(host->buffer().host, expected.data(), expected.size()) == 0;
    }
    const float* actualData = host->host<float>();
    const float* expectedData = reinterpret_cast<const float*>(expected.data());
    const size_t count = expected.size() / sizeof(float);
    float maxError = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float error = std::fabs(actualData[i] - expectedData[i]);
        maxError = std::max(maxError, error);
        if (error > atol + rtol * std::fabs(expectedData[i])) {
            std::cerr << "Output mismatch at " << i << ": actual=" << actualData[i] << " expected=" << expectedData[i]
                      << " max_error=" << maxError << std::endl;
            return false;
        }
    }
    std::cout << "max output error: " << maxError << std::endl;
    return true;
}

bool restoreTensorRegions(Tensor* tensor, const ReplayTensorSpec& spec, const std::map<std::string, Tensor*>& tensors,
                          bool allowSnapshotFallback) {
    if (spec.regions.empty()) {
        return true;
    }
    for (const auto& source : spec.regions) {
        const auto origin = tensors.find(source.origin);
        if (origin == tensors.end()) {
            if (allowSnapshotFallback) {
                // The logical snapshot was copied into a linear Tensor above.
                // This is equivalent to the old cross-Command region for an
                // isolated execution and does not need the original pointer.
                return true;
            }
            std::cerr << "Missing region origin tensor: " << source.origin << std::endl;
            return false;
        }
    }
    auto* describe = MNN::TensorUtils::getDescribe(tensor);
    describe->regions.clear();
    for (const auto& source : spec.regions) {
        const auto origin = tensors.find(source.origin);
        Tensor::InsideDescribe::Region region;
        region.origin = origin->second;
        region.src.offset = source.srcOffset;
        region.dst.offset = source.dstOffset;
        for (int i = 0; i < 3; ++i) {
            region.src.stride[i] = source.srcStride[i];
            region.dst.stride[i] = source.dstStride[i];
            region.size[i] = source.size[i];
        }
        describe->regions.emplace_back(region);
    }
    return true;
}

} // namespace Replay
} // namespace MNN
