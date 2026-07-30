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

struct LatencyResult {
    bool commandOk = false;
    bool hasLatency = false;
    bool pmuDisabled = false;
    double latencyUs = 0.0;
    std::string error;
};

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
        if (i < results.size() && results[i].hasLatency) {
            document.AddMember(key, results[i].latencyUs, allocator);
        } else {
            rapidjson::Value nullValue(rapidjson::kNullType);
            document.AddMember(key, nullValue, allocator);
        }
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
    for (size_t i = 0; i < cases.size(); ++i) {
        LatencyResult result = runLatency(cases[i], corpusRoot);
        if (!result.hasLatency) {
            std::printf("  [%zu/%zu] %s FAILED: %s\n", i + 1, cases.size(),
                        cases[i].c_str(), result.error.c_str());
        } else {
            ++valid;
            std::printf("  [%zu/%zu] %s %.3f us\n", i + 1, cases.size(),
                        cases[i].c_str(), result.latencyUs);
        }
        results.emplace_back(result);
    }

    EXPECT_TRUE(writeLatencyJson(latencyOutputPath(), cases, results));
    EXPECT_EQ(valid, cases.size());
}

int main(int argc, char** argv) {
    return ReplayTest::RunAll(argc, argv);
}
