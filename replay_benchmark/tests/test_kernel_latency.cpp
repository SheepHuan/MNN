// Measure CUDA kernel latency independently from PMU collection.
//
// This test intentionally passes --perf-counter-events none. It validates
// that latency collection does not open a CUPTI session and writes one
// structured JSON record per CUDA kernel-corpus case.

#include "KernelCorpusBenchmark.hpp"
#include "tests/ReplayTest.hpp"
#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/writer.h"

#include <cerrno>
#include <cmath>
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

static uint32_t rotateRight(uint32_t value, uint32_t bits) {
    return (value >> bits) | (value << (32 - bits));
}

static std::string sha256Hex(const std::string& contents) {
    static const uint32_t constants[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    };
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };
    std::vector<uint8_t> message(contents.begin(), contents.end());
    const uint64_t bitLength = static_cast<uint64_t>(message.size()) * 8;
    message.push_back(0x80);
    while (message.size() % 64 != 56) message.push_back(0);
    for (int shift = 56; shift >= 0; shift -= 8) {
        message.push_back(static_cast<uint8_t>((bitLength >> shift) & 0xff));
    }

    for (size_t offset = 0; offset < message.size(); offset += 64) {
        uint32_t words[64] = {};
        for (size_t i = 0; i < 16; ++i) {
            const size_t position = offset + i * 4;
            words[i] = (static_cast<uint32_t>(message[position]) << 24) |
                       (static_cast<uint32_t>(message[position + 1]) << 16) |
                       (static_cast<uint32_t>(message[position + 2]) << 8) |
                       static_cast<uint32_t>(message[position + 3]);
        }
        for (size_t i = 16; i < 64; ++i) {
            const uint32_t sigma0 = rotateRight(words[i - 15], 7) ^
                                    rotateRight(words[i - 15], 18) ^
                                    (words[i - 15] >> 3);
            const uint32_t sigma1 = rotateRight(words[i - 2], 17) ^
                                    rotateRight(words[i - 2], 19) ^
                                    (words[i - 2] >> 10);
            words[i] = words[i - 16] + sigma0 + words[i - 7] + sigma1;
        }

        uint32_t a = state[0];
        uint32_t b = state[1];
        uint32_t c = state[2];
        uint32_t d = state[3];
        uint32_t e = state[4];
        uint32_t f = state[5];
        uint32_t g = state[6];
        uint32_t h = state[7];
        for (size_t i = 0; i < 64; ++i) {
            const uint32_t sum1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
            const uint32_t choose = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + sum1 + choose + constants[i] + words[i];
            const uint32_t sum0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    char digest[65];
    for (size_t i = 0; i < 8; ++i) {
        std::snprintf(digest + i * 8, 9, "%08x", state[i]);
    }
    return std::string(digest, 64);
}

static std::vector<std::string> discoverCudaCases(const std::string& corpusRoot,
                                                   std::string* sourceManifestSha256) {
    std::string contents;
    if (!readFile(corpusRoot + "/operator_cases.json", &contents)) return {};
    *sourceManifestSha256 = sha256Hex(contents);
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
    bool validationPassed = false;
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
    if (result->launchSamplingStatus != "sampled") {
        result->error = "launch sampling did not complete successfully";
        return false;
    }
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
    if (result->launchRecords.empty()) {
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
    if (result->environment.samplingSource.empty() || result->environment.samplingStatus != "sampled") {
        result->error = "environment sampling did not complete successfully";
        return false;
    }
    const OptionalNumber values[] = {
        result->environment.gpuClockHzBefore,
        result->environment.gpuClockHzAfter,
        result->environment.temperatureCBefore,
        result->environment.temperatureCAfter,
    };
    for (const auto& value : values) {
        if (!value.present || !std::isfinite(value.value) || value.value <= 0.0) {
            result->error = "environment sampling returned a missing or non-positive value";
            return false;
        }
    }
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
    if (!report.HasMember("valid") || !report["valid"].IsBool() ||
        !report["valid"].GetBool() ||
        !report.HasMember("validation_status") || !report["validation_status"].IsString() ||
        std::strcmp(report["validation_status"].GetString(), "validation_passed") != 0) {
        result.error = "kernel output validation failed";
        return result;
    }
    result.validationPassed = true;
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
                             const std::vector<LatencyResult>& results,
                             const std::string& sourceManifestSha256) {
    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) return false;
    rapidjson::Document document;
    document.SetObject();
    auto& allocator = document.GetAllocator();
    rapidjson::Value format("mnn-kernel-latency", allocator);
    rapidjson::Value manifestSha256(sourceManifestSha256.c_str(), allocator);
    document.AddMember("format", format, allocator);
    document.AddMember("version", 1, allocator);
    document.AddMember("source_manifest_sha256", manifestSha256, allocator);
    rapidjson::Value caseValues(rapidjson::kObjectType);
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
                environment.AddMember(
                    "temperature_c_before", results[i].environment.temperatureCBefore.value, allocator);
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
        caseValues.AddMember(key, value, allocator);
    }
    document.AddMember("cases", caseValues, allocator);
    char buffer[4096];
    rapidjson::FileWriteStream stream(file, buffer, sizeof(buffer));
    rapidjson::Writer<rapidjson::FileWriteStream> writer(stream);
    const bool ok = document.Accept(writer);
    std::fclose(file);
    return ok;
}

} // namespace

TEST(KernelLatency, CudaAdapterValidationIsTerminal) {
    using MNN::Replay::KernelCorpus::detail::AdapterValidationDecision;
    using MNN::Replay::KernelCorpus::detail::adapterValidationDecision;

    // These selected cases carry a legacy identity validator even though their
    // CUDA adapters validate a real layout transform. A failed adapter result
    // must be terminal; otherwise an identity output can be incorrectly rescued.
    const char* selectedCases[] = {
        "transpose_local_fp32_smoke",
        "c4nhw4_2_nchw_fp32_smoke",
    };
    for (const char* caseName : selectedCases) {
        (void)caseName;
        EXPECT_EQ(
            adapterValidationDecision(true, true, false),
            AdapterValidationDecision::Reject);
        EXPECT_EQ(
            adapterValidationDecision(true, true, true),
            AdapterValidationDecision::Accept);
    }

    // OpenCL/Vulkan compatibility remains unchanged while their validators
    // continue migrating from string dispatch to co-located adapter logic.
    EXPECT_EQ(
        adapterValidationDecision(true, false, false),
        AdapterValidationDecision::TryLegacy);
    EXPECT_EQ(
        adapterValidationDecision(false, false, false),
        AdapterValidationDecision::TryLegacy);
}

TEST(KernelLatency, AllCudaCases) {
    ASSERT_EQ(
        sha256Hex("abc"),
        std::string("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
    const char* root = std::getenv("REPLAY_KERNEL_CORPUS_ROOT");
    const std::string corpusRoot = root == nullptr || *root == '\0'
        ? "../replay_benchmark/kernel_corpus" : root;
    std::string sourceManifestSha256;
    auto cases = discoverCudaCases(corpusRoot, &sourceManifestSha256);
    const char* caseFilter = std::getenv("REPLAY_KERNEL_LATENCY_CASE_FILTER");
    if (caseFilter != nullptr && *caseFilter != '\0') {
        std::vector<std::string> filtered;
        for (const auto& name : cases) {
            if (name.find(caseFilter) != std::string::npos) filtered.emplace_back(name);
        }
        cases.swap(filtered);
    }
    ASSERT_TRUE(!cases.empty());
    ASSERT_EQ(sourceManifestSha256.size(), static_cast<size_t>(64));

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

    EXPECT_TRUE(writeLatencyJson(
        latencyOutputPath(), cases, results, sourceManifestSha256));
    EXPECT_EQ(valid, cases.size());
    EXPECT_EQ(structured, cases.size());
}

int main(int argc, char** argv) {
    return ReplayTest::RunAll(argc, argv);
}
