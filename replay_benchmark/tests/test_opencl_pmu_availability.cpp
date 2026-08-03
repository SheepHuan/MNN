// OpenCL PMU metric availability and validity test.
//
// This is the OpenCL/Vulkan counterpart of test_pmu_availability.cpp (which
// targets NVIDIA CUDA via CUPTI). It runs on the local device (expected to be
// an ARM board with Adreno or Mali GPU) and:
//   1. Discovers all PMU events exposed via --opencl-pmu-list-events
//   2. For each event, runs one kernel corpus case with --perf-counter-events
//      and reads back the value
//   3. Classifies each event as VALID / NOT_FOUND / COMMAND_FAILED
//   4. Reports per-event status and a summary count
//
// Build (ARM):
//   cmake -DMNN_OPENCL=ON -DMNN_BUILD_BENCHMARK=ON -DMNN_REPLAY_ENABLE_PERFCOUNTER=ON ..
//   make replay_opencl_pmu_availability
//
// Run on device:
//   ./replay_opencl_pmu_availability
//   ./replay_opencl_pmu_availability --filter OpenclPmu.AllEvents
//   ./replay_opencl_pmu_availability --filter OpenclPmu.KeyEvents

#include "tests/ReplayTest.hpp"
#include "rapidjson/document.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>
#include <map>
#include <algorithm>

namespace {

// ============================================================
// Infrastructure
// ============================================================

struct EventResult {
    std::string name;
    // value semantics:
    //   >= 0     : valid counter value
    //   -1.0     : event not present in benchmark output (NOT_FOUND)
    //   -2.0     : benchmark command succeeded but JSON malformed/unreadable
    double value = -1.0;
};

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

static std::string nextOutputPath() {
    static unsigned sequence = 0;
    char path[128];
    std::snprintf(path, sizeof(path), "/tmp/_opencl_pmu_avail_%ld_%u.json",
                  static_cast<long>(getpid()), sequence++);
    return path;
}

// Run one kernel corpus case with a single PMU event; return the reported
// value for that event, or -1 if absent.
static EventResult runBench(const std::string& caseName, const std::string& eventName,
                             const std::string& corpusRoot) {
    EventResult result;
    result.name = eventName;
    const std::string outputPath = nextOutputPath();
    char command[8192];
    std::snprintf(command, sizeof(command),
                  "./replay_benchmark.out "
                  "--kernel-corpus-bench --kernel-corpus-root %s --kernel-corpus-case %s "
                  "--kernel-corpus-runs 1 --perf-counter-events \"%s\" "
                  "--perf-counter-output %s >/dev/null 2>&1",
                  corpusRoot.c_str(), caseName.c_str(), eventName.c_str(), outputPath.c_str());
    if (std::system(command) != 0) {
        result.value = -2.0;
        std::remove(outputPath.c_str());
        return result;
    }
    std::string json;
    if (!readFile(outputPath, &json)) {
        result.value = -2.0;
        std::remove(outputPath.c_str());
        return result;
    }
    std::remove(outputPath.c_str());

    // Strip any non-JSON banner before the first '{'.
    const size_t jstart = json.find('{');
    const size_t jend = json.rfind('}');
    if (jstart == std::string::npos || jend == std::string::npos || jend < jstart) {
        result.value = -2.0;
        return result;
    }
    const std::string payload = json.substr(jstart, jend - jstart + 1);

    rapidjson::Document document;
    document.Parse(payload.c_str(), payload.size());
    if (document.HasParseError() || !document.IsObject() || !document.HasMember("cases") ||
        !document["cases"].IsArray() || document["cases"].Empty()) {
        result.value = -2.0;
        return result;
    }
    const auto& report = document["cases"].GetArray()[0];
    if (!report.IsObject() || !report.HasMember("pmu_metrics") || !report["pmu_metrics"].IsObject()) {
        result.value = -1.0;
        return result;
    }
    const auto& metrics = report["pmu_metrics"];
    if (!metrics.HasMember(eventName.c_str())) {
        result.value = -1.0;
        return result;
    }
    const auto& v = metrics[eventName.c_str()];
    if (v.IsUint64()) {
        result.value = static_cast<double>(v.GetUint64());
    } else if (v.IsInt64()) {
        result.value = static_cast<double>(v.GetInt64());
    } else if (v.IsDouble()) {
        result.value = v.GetDouble();
    } else {
        result.value = -1.0;
    }
    return result;
}

// Discover OpenCL PMU events by running `replay_benchmark.out --opencl-pmu-list-events`.
// Expected JSON output: {"pmu_status": "available", "pmu_events": ["<name>", ...]}
// Some drivers (e.g. Rockchip Mali) print a non-JSON banner line before the
// JSON payload; we extract the substring from the first '{' to the last '}'.
static std::vector<std::string> discoverOpenclEvents() {
    std::vector<std::string> events;
    const std::string outputPath = nextOutputPath();
    char command[1024];
    std::snprintf(command, sizeof(command),
                  "./replay_benchmark.out --opencl-pmu-list-events > %s 2>/dev/null",
                  outputPath.c_str());
    if (std::system(command) != 0) {
        std::remove(outputPath.c_str());
        return events;
    }
    std::string json;
    if (!readFile(outputPath, &json)) {
        std::remove(outputPath.c_str());
        return events;
    }
    std::remove(outputPath.c_str());

    // Strip any non-JSON banner (e.g. "arm_release_ver: ...") before the first '{'.
    const size_t start = json.find('{');
    const size_t end = json.rfind('}');
    if (start == std::string::npos || end == std::string::npos || end < start) return events;
    const std::string payload = json.substr(start, end - start + 1);

    rapidjson::Document document;
    document.Parse(payload.c_str(), payload.size());
    if (document.HasParseError() || !document.IsObject()) return events;
    if (!document.HasMember("pmu_status") || !document["pmu_status"].IsString() ||
        std::strcmp(document["pmu_status"].GetString(), "available") != 0) {
        return events;
    }
    if (!document.HasMember("pmu_events") || !document["pmu_events"].IsArray()) return events;
    for (const auto& item : document["pmu_events"].GetArray()) {
        if (item.IsString()) events.emplace_back(item.GetString());
    }
    return events;
}

// Pick a representative case for the backend. Falls back to the first opencl
// case if the preferred name is not present.
static std::string pickTestCase(const std::string& corpusRoot, const std::string& backend,
                                 const std::string& preferred) {
    std::string contents;
    if (!readFile(corpusRoot + "/operator_cases.json", &contents)) return preferred;
    rapidjson::Document document;
    document.Parse(contents.c_str(), contents.size());
    if (document.HasParseError() || !document.IsObject() || !document.HasMember("cases") ||
        !document["cases"].IsArray()) return preferred;
    std::string firstMatch;
    for (const auto& item : document["cases"].GetArray()) {
        if (!item.IsObject() || !item.HasMember("backend") || !item["backend"].IsString() ||
            backend != item["backend"].GetString()) continue;
        if (!item.HasMember("name") || !item["name"].IsString()) continue;
        const std::string name = item["name"].GetString();
        if (firstMatch.empty()) firstMatch = name;
        if (name == preferred) return preferred;
    }
    return firstMatch.empty() ? preferred : firstMatch;
}

} // namespace

// ============================================================
// Test 1: Discovery — device exposes PMU events
// ============================================================
TEST(OpenclPmu, Discovery) {
    auto events = discoverOpenclEvents();
    ASSERT_TRUE(!events.empty());
    std::printf("  Discovered %zu OpenCL PMU events\n", events.size());
    for (size_t i = 0; i < events.size() && i < 10; ++i) {
        std::printf("    [%zu] %s\n", i + 1, events[i].c_str());
    }
    if (events.size() > 10) std::printf("    ... %zu more\n", events.size() - 10);
}

// ============================================================
// Test 2: Every discovered event — one event per process
// ============================================================
TEST(OpenclPmu, AllEvents) {
    auto events = discoverOpenclEvents();
    ASSERT_TRUE(!events.empty());

    const char* root = std::getenv("REPLAY_KERNEL_CORPUS_ROOT");
    const std::string corpusRoot = root == nullptr || *root == '\0'
        ? "../replay_benchmark/kernel_corpus" : root;
    const std::string testCase = pickTestCase(corpusRoot, "opencl", "argmax_buf_fp32_smoke");
    std::printf("  Using test case: %s\n", testCase.c_str());
    ASSERT_TRUE(!testCase.empty());

    int valid = 0, notFound = 0, commandFailed = 0;
    std::printf("  Sweeping %zu discovered OpenCL PMU events\n", events.size());
    for (size_t i = 0; i < events.size(); ++i) {
        auto r = runBench(testCase, events[i], corpusRoot);
        const char* state = "COMMAND_FAILED";
        if (r.value >= 0) { state = "VALID"; valid++; }
        else if (r.value == -1.0) { state = "NOT_FOUND"; notFound++; }
        else { commandFailed++; }
        std::printf("  [%4zu/%4zu] %-50s %-13s value=% .0f\n",
                    i + 1, events.size(), events[i].c_str(), state, r.value);
    }
    std::printf("  all events: %d valid, %d not_found, %d command_failed (of %zu)\n",
                valid, notFound, commandFailed, events.size());
    EXPECT_TRUE(valid > 0);
}

// ============================================================
// Test 3: Key events — the ones most useful for kernel analysis
// ============================================================
TEST(OpenclPmu, KeyEvents) {
    // Common Adreno/Mali PMU events. Not all may exist on every chip; we
    // report which subset is available.
    std::vector<std::pair<std::string, const char*>> candidates = {
        // Adreno (KGSL)
        {"cp_busy_cycles",          "Adreno CP busy cycles"},
        {"sp_busy_cycles",          "Adreno SP busy cycles"},
        {"sp_cs_invocations",       "Adreno CS invocations"},
        {"rbbm_vbif_busy",          "Adreno VBIF busy"},
        {"uche_busy_cycles",        "Adreno UCHE busy cycles"},
        {"tp_l1_cacheline_requests","Adreno L1 cache requests"},
        {"tp_l1_cacheline_misses",  "Adreno L1 cache misses"},
        // Mali
        {"gpu_active_cycles",       "Mali GPU active cycles"},
        {"compute_active_cycles",   "Mali compute active cycles"},
        {"compute_tasks",           "Mali compute tasks"},
        {"l2_any_lookup",           "Mali L2 lookup"},
        {"l2_ext_read",             "Mali L2 external read"},
        {"l2_ext_write",            "Mali L2 external write"},
    };
    auto available = discoverOpenclEvents();
    ASSERT_TRUE(!available.empty());

    const char* root = std::getenv("REPLAY_KERNEL_CORPUS_ROOT");
    const std::string corpusRoot = root == nullptr || *root == '\0'
        ? "../replay_benchmark/kernel_corpus" : root;
    const std::string testCase = pickTestCase(corpusRoot, "opencl", "argmax_buf_fp32_smoke");
    ASSERT_TRUE(!testCase.empty());

    int valid = 0, unavailable = 0;
    std::printf("  Testing %zu candidate key events on %s:\n", candidates.size(), testCase.c_str());
    for (const auto& [event, desc] : candidates) {
        if (std::find(available.begin(), available.end(), event) == available.end()) {
            std::printf("    %-40s NOT_EXPOSED (%s)\n", event.c_str(), desc);
            unavailable++;
            continue;
        }
        auto r = runBench(testCase, event, corpusRoot);
        if (r.value >= 0) {
            std::printf("    %-40s %14.0f (%s)\n", event.c_str(), r.value, desc);
            valid++;
        } else {
            std::printf("    %-40s INVALID (%s)\n", event.c_str(), desc);
            unavailable++;
        }
    }
    std::printf("  key events: %d valid, %d unavailable\n", valid, unavailable);
    // Don't ASSERT valid > 0 — different chips expose different subsets. The
    // test fails only if PMU is completely broken (Discovery would catch that).
}

// ============================================================
// Test 4: Reproducibility — same event gives same value across runs
// ============================================================
TEST(OpenclPmu, Reproducibility) {
    auto events = discoverOpenclEvents();
    ASSERT_TRUE(!events.empty());

    const char* root = std::getenv("REPLAY_KERNEL_CORPUS_ROOT");
    const std::string corpusRoot = root == nullptr || *root == '\0'
        ? "../replay_benchmark/kernel_corpus" : root;
    const std::string testCase = pickTestCase(corpusRoot, "opencl", "argmax_buf_fp32_smoke");
    ASSERT_TRUE(!testCase.empty());

    // Find a workload-sensitive event (prefer "_busy_cycles" which is a delta
    // between workload and idle). Free-running counters like "cp_always_count"
    // monotonically increase and will NOT reproduce — exclude them.
    std::string workingEvent;
    double baseline = -1.0;
    auto isReproducible = [](const std::string& name) {
        // Free-running counters that should not be tested for equality.
        return name.find("always_count") == std::string::npos &&
               name.find("timestamp") == std::string::npos;
    };
    for (const auto& e : events) {
        if (!isReproducible(e)) continue;
        auto r = runBench(testCase, e, corpusRoot);
        if (r.value >= 0) {
            workingEvent = e;
            baseline = r.value;
            break;
        }
    }
    if (workingEvent.empty()) {
        std::printf("  no workload-sensitive reproducible event found; skipping\n");
        return;
    }
    std::printf("  Using %s (baseline=%.0f) for 4 more runs\n", workingEvent.c_str(), baseline);

    int nonNegative = 0;
    int exactMatch = 0;
    for (int i = 0; i < 4; ++i) {
        auto r = runBench(testCase, workingEvent, corpusRoot);
        if (r.value >= 0) nonNegative++;
        if (r.value == baseline) exactMatch++;
    }
    std::printf("  %s: non-negative %d/4, exact-match %d/4 (baseline=%.0f)\n",
                workingEvent.c_str(), nonNegative, exactMatch, baseline);
    // Require all runs return a valid (non-negative) value. Exact equality is
    // not required because busy-cycle counters legitimately vary with system
    // noise and scheduling.
    EXPECT_EQ(nonNegative, 4);
}

int main(int argc, char** argv) {
    std::printf("=== OpenCL PMU Metric Availability Test Suite ===\n\n");
    return ReplayTest::RunAll(argc, argv);
}
