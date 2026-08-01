// Measure CUDA kernel latency independently from PMU collection.
//
// This test intentionally passes --perf-counter-events none. It validates
// that latency collection does not open a CUPTI session and writes one JSON
// value per CUDA kernel-corpus case.

#include "tests/ReplayTest.hpp"
#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/writer.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

static std::string nextOutputPath() {
    static unsigned sequence = 0;
    char path[128];
    std::snprintf(path, sizeof(path), "/tmp/_kernel_latency_%ld_%u.json",
                  static_cast<long>(getpid()), sequence++);
    return path;
}

static bool readFile(const std::string& path, std::string* contents) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) return false;
    if (std::fseek(file, 0, SEEK_END) != 0) {
        std::fclose(file);
        return false;
    }
    const long size = std::ftell(file);
    if (size < 0 || std::fseek(file, 0, SEEK_SET) != 0) {
        std::fclose(file);
        return false;
    }
    contents->assign(static_cast<size_t>(size), '\0');
    const size_t read = contents->empty() ? 0 : std::fread(&(*contents)[0], 1, contents->size(), file);
    std::fclose(file);
    return read == contents->size();
}

static std::vector<std::string> discoverCudaCases(const std::string& corpusRoot) {
    std::string contents;
    if (!readFile(corpusRoot + "/operator_cases.json", &contents)) return {};
    rapidjson::Document document;
    document.Parse(contents.c_str(), contents.size());
    if (document.HasParseError() || !document.IsObject() || !document.HasMember("cases") ||
        !document["cases"].IsArray()) return {};

    std::vector<std::string> cases;
    for (const auto& item : document["cases"].GetArray()) {
        if (!item.IsObject() || !item.HasMember("backend") || !item["backend"].IsString() ||
            std::strcmp(item["backend"].GetString(), "cuda") != 0 ||
            !item.HasMember("name") || !item["name"].IsString()) continue;
        cases.emplace_back(item["name"].GetString());
    }
    return cases;
}

struct LaunchRecord {
    std::string kernelName;
    int64_t grid[3] = {0, 0, 0};
    int64_t block[3] = {0, 0, 0};
    uint64_t registersPerThread = 0;
    uint64_t staticSharedMemoryBytes = 0;
    uint64_t dynamicSharedMemoryBytes = 0;
    uint64_t localMemoryPerThreadBytes = 0;
    uint64_t localMemoryTotalBytes = 0;
};

struct OptionalNumber {
    bool present = false;
    double value = 0.0;
};

struct EnvironmentResult {
    std::string samplingSource;
    std::string samplingStatus;
    OptionalNumber gpuClockHzBefore;
    OptionalNumber gpuClockHzAfter;
    OptionalNumber temperatureCBefore;
    OptionalNumber temperatureCAfter;
};

struct LatencyResult {
    bool commandOk = false;
    bool hasLatency = false;
    bool pmuDisabled = false;
    bool hasStructuredMetadata = false;
    double latencyUs = 0.0;
    std::string launchSamplingStatus;
    std::vector<LaunchRecord> launchRecords;
    EnvironmentResult environment;
    std::string error;
};

static bool readDimension3(const rapidjson::Value& object, const char* key, int64_t result[3]) {
    if (!object.HasMember(key) || !object[key].IsArray() || object[key].Size() != 3) return false;
    for (rapidjson::SizeType i = 0; i < 3; ++i) {
        if (!object[key][i].IsInt64() && !object[key][i].IsUint64()) return false;
        result[i] = object[key][i].IsInt64()
            ? object[key][i].GetInt64() : static_cast<int64_t>(object[key][i].GetUint64());
        if (result[i] <= 0) return false;
    }
    return true;
}

static bool readNonnegativeUint64(const rapidjson::Value& object, const char* key, uint64_t* result) {
    if (!object.HasMember(key)) return false;
    if (object[key].IsUint64()) {
        *result = object[key].GetUint64();
        return true;
    }
    if (object[key].IsInt64() && object[key].GetInt64() >= 0) {
        *result = static_cast<uint64_t>(object[key].GetInt64());
        return true;
    }
    return false;
}

static OptionalNumber readOptionalNumber(const rapidjson::Value& object, const char* key) {
    OptionalNumber result;
    if (object.HasMember(key) && object[key].IsNumber()) {
        result.present = true;
        result.value = object[key].GetDouble();
    }
    return result;
}

static bool parseLaunchRecords(const rapidjson::Value& report, LatencyResult* result) {
    if (!report.HasMember("launch_sampling_status") || !report["launch_sampling_status"].IsString() ||
        !report.HasMember("launch_records") || !report["launch_records"].IsArray()) {
        result->error = "launch metadata is missing or malformed";
        return false;
    }
    result->launchSamplingStatus = report["launch_sampling_status"].GetString();
    for (const auto& item : report["launch_records"].GetArray()) {
        LaunchRecord record;
        if (!item.IsObject() || !item.HasMember("kernel_name") || !item["kernel_name"].IsString() ||
            !readDimension3(item, "grid", record.grid) ||
            !readDimension3(item, "block", record.block) ||
            !readNonnegativeUint64(item, "registers_per_thread", &record.registersPerThread) ||
            !readNonnegativeUint64(item, "static_shared_memory_bytes", &record.staticSharedMemoryBytes) ||
            !readNonnegativeUint64(item, "dynamic_shared_memory_bytes", &record.dynamicSharedMemoryBytes) ||
            !readNonnegativeUint64(item, "local_memory_per_thread_bytes", &record.localMemoryPerThreadBytes) ||
            !readNonnegativeUint64(item, "local_memory_total_bytes", &record.localMemoryTotalBytes)) {
            result->error = "launch record is malformed";
            return false;
        }
        record.kernelName = item["kernel_name"].GetString();
        if (record.kernelName.empty()) {
            result->error = "launch record kernel_name is empty";
            return false;
        }
        result->launchRecords.emplace_back(std::move(record));
    }
    if (result->launchSamplingStatus == "sampled" && result->launchRecords.empty()) {
        result->error = "launch sampling succeeded but returned no kernel records";
        return false;
    }
    return true;
}

static bool parseEnvironment(const rapidjson::Value& report, LatencyResult* result) {
    if (!report.HasMember("environment") || !report["environment"].IsObject()) {
        result->error = "environment metadata is missing or malformed";
        return false;
    }
    const auto& environment = report["environment"];
    if (!environment.HasMember("sampling_source") || !environment["sampling_source"].IsString() ||
        !environment.HasMember("sampling_status") || !environment["sampling_status"].IsString()) {
        result->error = "environment sampling status is missing or malformed";
        return false;
    }
    result->environment.samplingSource = environment["sampling_source"].GetString();
    result->environment.samplingStatus = environment["sampling_status"].GetString();
    result->environment.gpuClockHzBefore = readOptionalNumber(environment, "gpu_clock_hz_before");
    result->environment.gpuClockHzAfter = readOptionalNumber(environment, "gpu_clock_hz_after");
    result->environment.temperatureCBefore = readOptionalNumber(environment, "temperature_c_before");
    result->environment.temperatureCAfter = readOptionalNumber(environment, "temperature_c_after");
    return true;
}

static LatencyResult runLatency(const std::string& caseName, const std::string& corpusRoot) {
    LatencyResult result;
    const std::string outputPath = nextOutputPath();
    char command[8192];
    std::snprintf(command, sizeof(command),
                  "env LD_LIBRARY_PATH=.:source/backend/cuda:. ./replay_benchmark.out "
                  "--kernel-corpus-bench --kernel-corpus-root %s --kernel-corpus-case %s "
                  "--kernel-corpus-runs 1 --perf-counter-events none --perf-counter-output %s",
                  corpusRoot.c_str(), caseName.c_str(), outputPath.c_str());
    result.commandOk = std::system(command) == 0;
    if (!result.commandOk) {
        result.error = "latency benchmark command failed";
        std::remove(outputPath.c_str());
        return result;
    }

    std::string json;
    if (!readFile(outputPath, &json)) {
        result.error = "latency benchmark output is missing or unreadable";
        std::remove(outputPath.c_str());
        return result;
    }
    std::remove(outputPath.c_str());

    rapidjson::Document document;
    document.Parse(json.c_str(), json.size());
    if (document.HasParseError() || !document.IsObject() || !document.HasMember("cases") ||
        !document["cases"].IsArray() || document["cases"].Empty()) {
        result.error = "latency benchmark output is malformed";
        return result;
    }
    const auto& report = document["cases"].GetArray()[0];
    if (!report.IsObject() || !report.HasMember("pmu_status") || !report["pmu_status"].IsString() ||
        std::strcmp(report["pmu_status"].GetString(), "disabled") != 0) {
        result.error = "latency test did not disable PMU";
        return result;
    }
    result.pmuDisabled = true;
    if (!report.HasMember("latency_us") || !report["latency_us"].IsNumber()) {
        result.error = "latency_us is missing or malformed";
        return result;
    }
    result.latencyUs = report["latency_us"].GetDouble();
    result.hasLatency = result.latencyUs > 0.0;
    if (!result.hasLatency) result.error = "latency_us is not positive";
    if (result.hasLatency && parseLaunchRecords(report, &result) && parseEnvironment(report, &result)) {
        result.hasStructuredMetadata = true;
    }
    return result;
}

static std::string latencyOutputPath() {
    const char* value = std::getenv("REPLAY_KERNEL_LATENCY_OUTPUT");
    return value == nullptr || *value == '\0' ? "kernel_latency.json" : value;
}

static bool writeLatencyJson(const std::string& path,
                             const std::vector<std::string>& cases,
                             const std::vector<LatencyResult>& results) {
    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) return false;
    rapidjson::Document document;
    document.SetObject();
    auto& allocator = document.GetAllocator();
    for (size_t i = 0; i < cases.size(); ++i) {
        rapidjson::Value key(cases[i].c_str(), allocator);
        rapidjson::Value value(rapidjson::kObjectType);
        if (i < results.size() && results[i].hasLatency) {
            value.AddMember("latency_us", results[i].latencyUs, allocator);
        } else {
            rapidjson::Value nullValue(rapidjson::kNullType);
            value.AddMember("latency_us", nullValue, allocator);
        }
        if (i < results.size()) {
            rapidjson::Value launchStatus(results[i].launchSamplingStatus.c_str(), allocator);
            value.AddMember("launch_sampling_status", launchStatus, allocator);
            rapidjson::Value launches(rapidjson::kArrayType);
            for (const auto& record : results[i].launchRecords) {
                rapidjson::Value item(rapidjson::kObjectType);
                rapidjson::Value kernelName(record.kernelName.c_str(), allocator);
                item.AddMember("kernel_name", kernelName, allocator);
                rapidjson::Value grid(rapidjson::kArrayType);
                rapidjson::Value block(rapidjson::kArrayType);
                for (int d = 0; d < 3; ++d) {
                    grid.PushBack(record.grid[d], allocator);
                    block.PushBack(record.block[d], allocator);
                }
                item.AddMember("grid", grid, allocator);
                item.AddMember("block", block, allocator);
                item.AddMember("registers_per_thread", record.registersPerThread, allocator);
                item.AddMember("static_shared_memory_bytes", record.staticSharedMemoryBytes, allocator);
                item.AddMember("dynamic_shared_memory_bytes", record.dynamicSharedMemoryBytes, allocator);
                item.AddMember("local_memory_per_thread_bytes", record.localMemoryPerThreadBytes, allocator);
                item.AddMember("local_memory_total_bytes", record.localMemoryTotalBytes, allocator);
                launches.PushBack(item, allocator);
            }
            value.AddMember("launch_records", launches, allocator);

            rapidjson::Value environment(rapidjson::kObjectType);
            rapidjson::Value source(results[i].environment.samplingSource.c_str(), allocator);
            rapidjson::Value status(results[i].environment.samplingStatus.c_str(), allocator);
            environment.AddMember("sampling_source", source, allocator);
            environment.AddMember("sampling_status", status, allocator);
            if (results[i].environment.gpuClockHzBefore.present) {
                environment.AddMember("gpu_clock_hz_before", results[i].environment.gpuClockHzBefore.value, allocator);
            }
            if (results[i].environment.gpuClockHzAfter.present) {
                environment.AddMember("gpu_clock_hz_after", results[i].environment.gpuClockHzAfter.value, allocator);
            }
            if (results[i].environment.temperatureCBefore.present) {
                environment.AddMember("temperature_c_before", results[i].environment.temperatureCBefore.value, allocator);
            }
            if (results[i].environment.temperatureCAfter.present) {
                environment.AddMember("temperature_c_after", results[i].environment.temperatureCAfter.value, allocator);
            }
            value.AddMember("environment", environment, allocator);
            if (!results[i].error.empty()) {
                rapidjson::Value error(results[i].error.c_str(), allocator);
                value.AddMember("error", error, allocator);
            }
        }
        document.AddMember(key, value, allocator);
    }
    char buffer[4096];
    rapidjson::FileWriteStream stream(file, buffer, sizeof(buffer));
    rapidjson::Writer<rapidjson::FileWriteStream> writer(stream);
    const bool ok = document.Accept(writer);
    std::fclose(file);
    return ok;
}

} // namespace

TEST(KernelLatency, AllCudaCases) {
    const char* root = std::getenv("REPLAY_KERNEL_CORPUS_ROOT");
    const std::string corpusRoot = root == nullptr || *root == '\0'
        ? "../replay_benchmark/kernel_corpus" : root;
    auto cases = discoverCudaCases(corpusRoot);
    const char* caseFilter = std::getenv("REPLAY_KERNEL_LATENCY_CASE_FILTER");
    if (caseFilter != nullptr && *caseFilter != '\0') {
        std::vector<std::string> filtered;
        for (const auto& name : cases) {
            if (name.find(caseFilter) != std::string::npos) filtered.emplace_back(name);
        }
        cases.swap(filtered);
    }
    ASSERT_TRUE(!cases.empty());

    std::vector<LatencyResult> results;
    results.reserve(cases.size());
    size_t valid = 0;
    size_t structured = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        LatencyResult result = runLatency(cases[i], corpusRoot);
        if (!result.hasLatency) {
            std::printf("  [%zu/%zu] %s FAILED: %s\n", i + 1, cases.size(),
                        cases[i].c_str(), result.error.c_str());
        } else {
            ++valid;
            if (result.hasStructuredMetadata) ++structured;
            std::printf("  [%zu/%zu] %s %.3f us, launch=%s (%zu records), environment=%s\n",
                        i + 1, cases.size(), cases[i].c_str(), result.latencyUs,
                        result.launchSamplingStatus.c_str(), result.launchRecords.size(),
                        result.environment.samplingStatus.c_str());
        }
        results.emplace_back(result);
    }

    EXPECT_TRUE(writeLatencyJson(latencyOutputPath(), cases, results));
    EXPECT_EQ(valid, cases.size());
    EXPECT_EQ(structured, cases.size());
}

int main(int argc, char** argv) {
    return ReplayTest::RunAll(argc, argv);
}
