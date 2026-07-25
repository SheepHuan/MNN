#include "PerfCounterReport.hpp"

#include "ReplayRecord.hpp"

#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/writer.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

namespace MNN {
namespace Replay {

namespace {

static bool makeParentDirectories(const std::string& path) {
    const std::string::size_type slash = path.find_last_of("/\\");
    if (slash == std::string::npos) {
        return true;
    }
    const std::string parent = path.substr(0, slash);
    if (parent.empty()) {
        return true;
    }
    std::string current;
    for (size_t i = 0; i < parent.size(); ++i) {
        current.push_back(parent[i]);
        if (parent[i] != '/' && parent[i] != '\\' && i + 1 != parent.size()) {
            continue;
        }
        if (!current.empty() && current != "/" && mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
            return false;
        }
    }
    if (mkdir(parent.c_str(), 0755) != 0 && errno != EEXIST) {
        return false;
    }
    return true;
}

static void addString(rapidjson::Value& object, const char* key, const std::string& value,
                      rapidjson::Document::AllocatorType& allocator) {
    rapidjson::Value k(key, allocator);
    rapidjson::Value v(value.c_str(), allocator);
    object.AddMember(k, v, allocator);
}

} // namespace

bool PerfCounterReport::write(const std::string& path) const {
    if (path.empty() || !makeParentDirectories(path)) {
        return false;
    }
    rapidjson::Document document;
    document.SetObject();
    auto& allocator = document.GetAllocator();
    addString(document, "format", "mnn-perf-counter", allocator);
    document.AddMember("version", 1, allocator);
    addString(document, "status", status, allocator);
    addString(document, "model", model, allocator);

    rapidjson::Value op(rapidjson::kObjectType);
    op.AddMember("op_id", opId, allocator);
    addString(op, "name", opName, allocator);
    addString(op, "op_type", opType, allocator);
    addString(op, "execution", execution, allocator);
    addString(op, "variant", variant, allocator);
    document.AddMember("op", op, allocator);

    rapidjson::Value runtime(rapidjson::kObjectType);
    addString(runtime, "backend", backend, allocator);
    addString(runtime, "vendor", vendor, allocator);
    runtime.AddMember("product_id", static_cast<uint64_t>(productId), allocator);
    addString(runtime, "product_name", productName, allocator);
    addString(runtime, "driver", driver, allocator);
    addString(runtime, "scope", "gpu_device_interval", allocator);
    runtime.AddMember("synchronized", synchronized, allocator);
    document.AddMember("runtime", runtime, allocator);

    rapidjson::Value measurement(rapidjson::kObjectType);
    measurement.AddMember("start_ns", static_cast<uint64_t>(startNs), allocator);
    measurement.AddMember("end_ns", static_cast<uint64_t>(endNs), allocator);
    measurement.AddMember("duration_ns", static_cast<uint64_t>(endNs >= startNs ? endNs - startNs : 0), allocator);
    document.AddMember("measurement", measurement, allocator);

    rapidjson::Value values(rapidjson::kArrayType);
    for (const auto& counter : counters) {
        rapidjson::Value value(rapidjson::kObjectType);
        addString(value, "name", counter.name, allocator);
        value.AddMember("value", static_cast<uint64_t>(counter.value), allocator);
        values.PushBack(value, allocator);
    }
    document.AddMember("counters", values, allocator);
    addString(document, "error", error, allocator);

    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) {
        return false;
    }
    char buffer[4096];
    rapidjson::FileWriteStream stream(file, buffer, sizeof(buffer));
    rapidjson::Writer<rapidjson::FileWriteStream> writer(stream);
    const bool result = document.Accept(writer);
    std::fclose(file);
    return result;
}

} // namespace Replay
} // namespace MNN
