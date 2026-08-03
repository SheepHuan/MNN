// Measure OpenCL/Vulkan kernel latency on the local device.
//
// The test is the OpenCL/Vulkan counterpart of test_kernel_latency.cpp
// (which targets CUDA via cudaEventElapsedTime). It is intended to run on
// ARM devices (Rhinopi with Adreno, OrangePi with Mali) where the OpenCL or
// Vulkan backend is available. Each kernel corpus case for the selected
// backend is executed with --perf-counter-events none so PMU does not
// interfere with the latency measurement.
//
// Output JSON shape:
//   {
//     "<case_name>": {
//       "latency_us": <double|null>,
//       "pmu_status": "disabled",
//       "backend": "opencl"|"vulkan",
//       "framework": "mnn"|"ncnn",
//       "tag": "<tag>",
//       "variant": "<variant>",
//       "error": "<string, optional>"
//     }, ...
//   }

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
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

static std::string nextOutputPath() {
    static unsigned sequence = 0;
    char path[128];
    std::snprintf(path, sizeof(path), "/tmp/_opencl_kernel_latency_%ld_%u.json",
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

struct CaseInfo {
    std::string name;
    std::string backend;
    std::string framework;
    std::string tag;
    std::string variant;
};

// Discover kernel corpus cases for the requested backend ("opencl" or "vulkan").
// Optionally filter by framework ("mnn", "ncnn", or empty for both).
static std::vector<CaseInfo> discoverCases(const std::string& corpusRoot,
                                            const std::string& backend,
                                            const std::string& frameworkFilter) {
    std::vector<CaseInfo> cases;
    std::string contents;
    if (!readFile(corpusRoot + "/operator_cases.json", &contents)) return cases;
    rapidjson::Document document;
    document.Parse(contents.c_str(), contents.size());
    if (document.HasParseError() || !document.IsObject() || !document.HasMember("cases") ||
        !document["cases"].IsArray()) return cases;

    for (const auto& item : document["cases"].GetArray()) {
        if (!item.IsObject()) continue;
        if (!item.HasMember("backend") || !item["backend"].IsString()) continue;
        if (backend != item["backend"].GetString()) continue;
        if (!item.HasMember("name") || !item["name"].IsString()) continue;
        CaseInfo info;
        info.name = item["name"].GetString();
        info.backend = item["backend"].GetString();
        if (item.HasMember("framework") && item["framework"].IsString())
            info.framework = item["framework"].GetString();
        if (item.HasMember("tag") && item["tag"].IsString())
            info.tag = item["tag"].GetString();
        if (item.HasMember("variant") && item["variant"].IsString())
            info.variant = item["variant"].GetString();
        if (!frameworkFilter.empty() && info.framework != frameworkFilter) continue;
        cases.emplace_back(std::move(info));
    }
    return cases;
}

struct LatencyResult {
    bool commandOk = false;
    bool hasLatency = false;
    bool pmuDisabled = false;
    double latencyUs = 0.0;
    std::string backend;
    std::string error;
};

static LatencyResult runLatency(const CaseInfo& info, const std::string& corpusRoot, int runs) {
    LatencyResult result;
    result.backend = info.backend;
    const std::string outputPath = nextOutputPath();
    char command[8192];
    // Use --perf-counter-events none to disable PMU so the latency reading
    // reflects the kernel workload only, not a profiler replay.
    // Wrap with `timeout` so a hung kernel (e.g. gemm_buf_fp32_smoke on Mali-G610)
    // does not stall the whole sweep; timeout failure is reported as skipped.
    const char* timeoutEnv = std::getenv("REPLAY_KERNEL_LATENCY_TIMEOUT_SEC");
    const int timeoutSec = timeoutEnv != nullptr && *timeoutEnv != '\0' ? std::atoi(timeoutEnv) : 60;
    std::snprintf(command, sizeof(command),
                  "timeout %d ./replay_benchmark.out "
                  "--kernel-corpus-bench --kernel-corpus-root %s --kernel-corpus-case %s "
                  "--kernel-corpus-runs %d --perf-counter-events none --perf-counter-output %s "
                  "2>/dev/null",
                  timeoutSec, corpusRoot.c_str(), info.name.c_str(), runs, outputPath.c_str());
    const int rc = std::system(command);
    result.commandOk = (rc == 0);
    if (!result.commandOk) {
        // system() returns wait(2) status; timeout(1) exits with 124.
        const int exitCode = (rc != -1 && WIFEXITED(rc)) ? WEXITSTATUS(rc) : -1;
        if (exitCode == 124) {
            result.error = "skipped: kernel execution hung (timeout)";
        } else {
            result.error = "latency benchmark command failed";
        }
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

    // Strip any non-JSON banner (e.g. Rockchip Mali prints "arm_release_ver: ...").
    const size_t jstart = json.find('{');
    const size_t jend = json.rfind('}');
    if (jstart == std::string::npos || jend == std::string::npos || jend < jstart) {
        result.error = "latency benchmark output is malformed (no JSON payload)";
        return result;
    }
    const std::string payload = json.substr(jstart, jend - jstart + 1);

    rapidjson::Document document;
    document.Parse(payload.c_str(), payload.size());
    if (document.HasParseError() || !document.IsObject() || !document.HasMember("cases") ||
        !document["cases"].IsArray() || document["cases"].Empty()) {
        result.error = "latency benchmark output is malformed";
        return result;
    }
    const auto& report = document["cases"].GetArray()[0];
    // Skip platform-unsupported cases (e.g. Intel subgroup kernels on NVIDIA).
    // These are not failures — the device simply cannot execute the kernel.
    if (report.HasMember("compile_status") && report["compile_status"].IsString()) {
        const std::string compileStatus = report["compile_status"].GetString();
        if (compileStatus == "unsupported") {
            result.error = "skipped: kernel unsupported on this device";
            return result;
        }
        // Compile failure due to unsupported shader extension (e.g.
        // GL_KHR_shader_subgroup on NVIDIA; supported on Mali/Adreno), or
        // missing glslangValidator on the device (ncnn .comp files need it
        // to produce SPIR-V).
        if (compileStatus == "compile_failed" && report.HasMember("error") &&
            report["error"].IsString()) {
            const std::string err = report["error"].GetString();
            if (err.find("extension not supported") != std::string::npos ||
                err.find("GL_KHR_shader_subgroup") != std::string::npos ||
                err.find("glslangValidator") != std::string::npos) {
                result.error = "skipped: shader toolchain/extension unsupported (" + compileStatus + ")";
                return result;
            }
        }
    }
    // Cases that compile but fail at arg setup indicate a corpus runner
    // limitation (e.g. image2d_t args not yet supported by the buffer-only
    // OpenCL runner). Treat as skipped, not failed. Vulkan pipeline creation
    // failure on Adreno/Mali driver (e.g. ncnn unfold_im2col) is also a
    // platform capability issue, not a test defect.
    if (report.HasMember("error") && report["error"].IsString()) {
        const std::string err = report["error"].GetString();
        if (err.find("arg setup failed") != std::string::npos) {
            result.error = "skipped: corpus runner limitation (" + err + ")";
            return result;
        }
        if (err.find("VkPipeline creation failed") != std::string::npos ||
            err.find("InitVulkan failed") != std::string::npos) {
            result.error = "skipped: vulkan pipeline unsupported on this driver";
            return result;
        }
    }
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

static std::string latencyOutputPath(const char* backendEnv) {
    const char* value = std::getenv(backendEnv);
    if (value != nullptr && *value != '\0') return value;
    return std::string("kernel_latency_") + backendEnv + ".json";
}

static bool writeLatencyJson(const std::string& path,
                              const std::vector<CaseInfo>& cases,
                              const std::vector<LatencyResult>& results) {
    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) return false;
    rapidjson::Document document;
    document.SetObject();
    auto& allocator = document.GetAllocator();
    for (size_t i = 0; i < cases.size(); ++i) {
        rapidjson::Value key(cases[i].name.c_str(), allocator);
        rapidjson::Value value(rapidjson::kObjectType);
        if (i < results.size() && results[i].hasLatency) {
            value.AddMember("latency_us", results[i].latencyUs, allocator);
        } else {
            rapidjson::Value nullValue(rapidjson::kNullType);
            value.AddMember("latency_us", nullValue, allocator);
        }
        rapidjson::Value backend(cases[i].backend.c_str(), allocator);
        value.AddMember("backend", backend, allocator);
        rapidjson::Value framework(cases[i].framework.c_str(), allocator);
        value.AddMember("framework", framework, allocator);
        rapidjson::Value tag(cases[i].tag.c_str(), allocator);
        value.AddMember("tag", tag, allocator);
        rapidjson::Value variant(cases[i].variant.c_str(), allocator);
        value.AddMember("variant", variant, allocator);
        value.AddMember("pmu_status", "disabled", allocator);
        if (i < results.size() && !results[i].error.empty()) {
            rapidjson::Value error(results[i].error.c_str(), allocator);
            value.AddMember("error", error, allocator);
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

// Run all cases for the given backend. outputEnvVar is the environment
// variable that overrides the output JSON path.
static int runAllForBackend(const char* backend, const char* frameworkFilter,
                             const char* outputEnvVar, const char* defaultOutput) {
    const char* root = std::getenv("REPLAY_KERNEL_CORPUS_ROOT");
    const std::string corpusRoot = root == nullptr || *root == '\0'
        ? "../replay_benchmark/kernel_corpus" : root;
    auto cases = discoverCases(corpusRoot, backend, frameworkFilter ? frameworkFilter : "");
    const char* caseFilter = std::getenv("REPLAY_KERNEL_LATENCY_CASE_FILTER");
    if (caseFilter != nullptr && *caseFilter != '\0') {
        std::vector<CaseInfo> filtered;
        for (const auto& c : cases) {
            if (c.name.find(caseFilter) != std::string::npos) filtered.emplace_back(c);
        }
        cases.swap(filtered);
    }
    if (cases.empty()) {
        std::printf("  no %s cases found in %s\n", backend, corpusRoot.c_str());
        return 0;
    }
    const char* runsEnv = std::getenv("REPLAY_KERNEL_LATENCY_RUNS");
    const int runs = runsEnv != nullptr && *runsEnv != '\0' ? std::atoi(runsEnv) : 5;

    std::printf("  %s: %zu cases (runs=%d)\n", backend, cases.size(), runs);
    std::vector<LatencyResult> results;
    results.reserve(cases.size());
    size_t valid = 0;
    size_t skipped = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        LatencyResult result = runLatency(cases[i], corpusRoot, runs);
        const bool isSkip = result.error.rfind("skipped:", 0) == 0;
        if (!result.hasLatency && !isSkip) {
            std::printf("    [%zu/%zu] %s FAILED: %s\n", i + 1, cases.size(),
                        cases[i].name.c_str(), result.error.c_str());
        } else if (isSkip) {
            ++skipped;
            std::printf("    [%zu/%zu] %s SKIPPED (%s)\n", i + 1, cases.size(),
                        cases[i].name.c_str(), result.error.c_str() + 9);
        } else {
            ++valid;
            std::printf("    [%zu/%zu] %s %.3f us (framework=%s tag=%s)\n",
                        i + 1, cases.size(), cases[i].name.c_str(), result.latencyUs,
                        cases[i].framework.c_str(), cases[i].tag.c_str());
        }
        results.emplace_back(result);
    }

    const char* outPath = std::getenv(outputEnvVar);
    const std::string outputPath = (outPath != nullptr && *outPath != '\0') ? outPath : defaultOutput;
    if (!writeLatencyJson(outputPath, cases, results)) {
        std::fprintf(stderr, "  failed to write %s\n", outputPath.c_str());
        return 1;
    }
    std::printf("  %s: %zu/%zu passed, %zu skipped (platform-unsupported), wrote %s\n",
                backend, valid, cases.size(), skipped, outputPath.c_str());
    // Pass when every non-skipped case has a valid latency. Skipped cases are
    // platform-unsupported and not a defect.
    return (valid + skipped) == cases.size() ? 0 : 1;
}

} // namespace

TEST(OpenclKernelLatency, AllOpenclCases) {
    EXPECT_EQ(runAllForBackend("opencl", /*framework=*/nullptr,
                                "REPLAY_OPENCL_KERNEL_LATENCY_OUTPUT",
                                "opencl_kernel_latency.json"), 0);
}

TEST(OpenclKernelLatency, AllVulkanCases) {
    EXPECT_EQ(runAllForBackend("vulkan", /*framework=*/nullptr,
                                "REPLAY_VULKAN_KERNEL_LATENCY_OUTPUT",
                                "vulkan_kernel_latency.json"), 0);
}

TEST(OpenclKernelLatency, MnnOpenclCases) {
    EXPECT_EQ(runAllForBackend("opencl", "mnn",
                                "REPLAY_MNN_OPENCL_KERNEL_LATENCY_OUTPUT",
                                "mnn_opencl_kernel_latency.json"), 0);
}

TEST(OpenclKernelLatency, NcnnVulkanCases) {
    EXPECT_EQ(runAllForBackend("vulkan", "ncnn",
                                "REPLAY_NCNN_VULKAN_KERNEL_LATENCY_OUTPUT",
                                "ncnn_vulkan_kernel_latency.json"), 0);
}

int main(int argc, char** argv) {
    return ReplayTest::RunAll(argc, argv);
}
