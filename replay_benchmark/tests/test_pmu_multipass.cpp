// test_pmu_multipass.cpp — PMU multi-pass accuracy test suite.
//
// 64 test combinations: 8 kernel variants × 8 metric groups.
// Each combination verifies the multi-pass report contract and compares only
// replay-invariant work counters with single-metric collection.
//
// This collector uses CUPTI AutoRange + KernelReplay without explicit cache
// purging. CUPTI restores device memory between passes, but hardware cache
// contents are not a reproducible part of that state. DRAM/L2 traffic, active
// warp integrals, and cycle counters can therefore legitimately change when
// the metric configuration changes. Those metrics are validated as finite,
// non-negative observations, but are not treated as cross-configuration
// numeric truth.
//
// Kernel variants (covering compute/memory/cache-bound workloads):
//   1. ConvDw    (depthwise conv, compute+memory)
//   2. Matmul    (GEMM, compute-bound)
//   3. Gemv      (GEMV, memory-bound)
//   4. Reduction  (sum reduction, memory-bound)
//   5. MaxPool   (max pooling, memory-bound)
//   6. Softmax   (softmax, compute+exp)
//   7. Layernorm (layernorm, compute)
//   8. Relu      (relu, pure memory)
//
// Metric groups (covering different hardware counter domains):
//   A. Compute       (sm__inst, sm__cycles)
//   B. Memory        (dram__bytes, read, write)
//   C. Cache         (l1tex__sectors, lts__sectors)
//   D. Mixed         (sm + dram + lts — triggers multi-pass)
//   E. InstrDetail   (sass fadd/fmul)
//   F. Warp          (warps_active, stall)
//   G. Throughput    (sm__cycles_active, dram__cycles_active)
//   H. AllCombined   (one from each domain — guaranteed multi-pass)
//
// Build: cmake -DMNN_CUDA=ON .. && make replay_pmu_test
// Run:   sudo LD_LIBRARY_PATH=build:build/source/backend/cuda ./build/replay_pmu_test
// Filter: ./replay_pmu_test --filter PMU.ConvDw_Mixed

#include "tests/ReplayTest.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unistd.h>

#include "rapidjson/document.h"

// ============================================================
// Infrastructure: subprocess + JSON parsing
// ============================================================

struct MetricResult {
    std::string name;
    double value;
    enum class Status {
        Valid,
        Unsupported,
        Overflow,
        CommandFailed,
        MalformedOutput,
    } status;
};

struct BenchResult {
    std::vector<MetricResult> metrics;
    std::string pmuStatus;
    uint64_t numPasses = 0;
    int commandStatus = -1;
    std::string error;
};

static const char* statusName(MetricResult::Status status) {
    switch (status) {
        case MetricResult::Status::Valid: return "Valid";
        case MetricResult::Status::Unsupported: return "Unsupported";
        case MetricResult::Status::Overflow: return "Overflow";
        case MetricResult::Status::CommandFailed: return "CommandFailed";
        case MetricResult::Status::MalformedOutput: return "MalformedOutput";
    }
    return "Unknown";
}

static std::string nextOutputPath() {
    static unsigned sequence = 0;
    char path[128];
    std::snprintf(path, sizeof(path), "/tmp/_pmu_test_%ld_%u.json",
                  static_cast<long>(getpid()), sequence++);
    return path;
}

static bool readFile(const std::string& path, std::string* contents) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return false;
    }
    long sz = ftell(f);
    if (sz < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return false;
    }
    contents->assign(static_cast<size_t>(sz), '\0');
    size_t read = contents->empty() ? 0 : fread(&(*contents)[0], 1, contents->size(), f);
    fclose(f);
    return read == contents->size();
}

static MetricResult parseMetric(const rapidjson::Value& caseResult, const std::string& metric) {
    if (!caseResult.IsObject() || !caseResult.HasMember("pmu_metric_statuses") ||
        !caseResult["pmu_metric_statuses"].IsObject()) {
        return {metric, -1.0, MetricResult::Status::MalformedOutput};
    }
    const auto& statuses = caseResult["pmu_metric_statuses"];
    if (!statuses.HasMember(metric.c_str()) || !statuses[metric.c_str()].IsString()) {
        return {metric, -1.0, MetricResult::Status::MalformedOutput};
    }
    const std::string status = statuses[metric.c_str()].GetString();
    if (status == "overflow") {
        return {metric, -1.0, MetricResult::Status::Overflow};
    }
    if (status == "invalid") {
        return {metric, -1.0, MetricResult::Status::Unsupported};
    }
    if (status != "valid") {
        return {metric, -1.0, MetricResult::Status::MalformedOutput};
    }
    if (!caseResult.HasMember("pmu_metrics") || !caseResult["pmu_metrics"].IsObject() ||
        !caseResult["pmu_metrics"].HasMember(metric.c_str()) ||
        !caseResult["pmu_metrics"][metric.c_str()].IsNumber()) {
        return {metric, -1.0, MetricResult::Status::MalformedOutput};
    }
    const double value = caseResult["pmu_metrics"][metric.c_str()].GetDouble();
    if (!std::isfinite(value) || value < 0.0) {
        return {metric, -1.0, MetricResult::Status::MalformedOutput};
    }
    return {metric, value, MetricResult::Status::Valid};
}

static BenchResult runBench(const std::string& caseName,
                            const std::vector<std::string>& metrics) {
    BenchResult result;
    const std::string outputPath = nextOutputPath();
    std::string metricStr;
    for (size_t i = 0; i < metrics.size(); ++i) {
        if (i > 0) metricStr += ",";
        metricStr += metrics[i];
    }

    char cmd[8192];
    std::snprintf(cmd, sizeof(cmd),
        "sudo env LD_LIBRARY_PATH=.:source/backend/cuda:. "
        "./replay_benchmark.out "
        "--kernel-corpus-bench "
        "--kernel-corpus-root ../replay_benchmark/kernel_corpus "
        "--kernel-corpus-case %s "
        "--kernel-corpus-runs 1 "
        "--perf-counter-events %s "
        "--perf-counter-output %s 2>/dev/null",
        caseName.c_str(), metricStr.c_str(), outputPath.c_str());

    result.commandStatus = system(cmd);
    if (result.commandStatus != 0) {
        result.error = "benchmark command failed";
        for (const auto& metric : metrics) {
            result.metrics.push_back({metric, -1.0, MetricResult::Status::CommandFailed});
        }
        std::remove(outputPath.c_str());
        return result;
    }

    std::string json;
    if (!readFile(outputPath, &json)) {
        result.error = "benchmark output is missing or unreadable";
        for (const auto& metric : metrics) {
            result.metrics.push_back({metric, -1.0, MetricResult::Status::MalformedOutput});
        }
        std::remove(outputPath.c_str());
        return result;
    }
    std::remove(outputPath.c_str());

    rapidjson::Document document;
    document.Parse(json.c_str(), json.size());
    if (document.HasParseError() || !document.IsObject() || !document.HasMember("cases") ||
        !document["cases"].IsArray() || document["cases"].Empty() ||
        !document["cases"][0].IsObject()) {
        result.error = "benchmark output JSON is malformed";
        for (const auto& metric : metrics) {
            result.metrics.push_back({metric, -1.0, MetricResult::Status::MalformedOutput});
        }
        return result;
    }
    const auto& caseResult = document["cases"][0];
    if (!caseResult.HasMember("num_passes") || !caseResult["num_passes"].IsUint64() ||
        caseResult["num_passes"].GetUint64() == 0) {
        result.error = "num_passes is missing, malformed, or zero";
        for (const auto& metric : metrics) {
            result.metrics.push_back({metric, -1.0, MetricResult::Status::MalformedOutput});
        }
        return result;
    }
    result.numPasses = caseResult["num_passes"].GetUint64();
    if (caseResult.HasMember("pmu_status") && caseResult["pmu_status"].IsString()) {
        result.pmuStatus = caseResult["pmu_status"].GetString();
    }
    if (result.pmuStatus.empty()) {
        result.error = "pmu_status is missing or malformed";
        for (const auto& metric : metrics) {
            result.metrics.push_back({metric, -1.0, MetricResult::Status::MalformedOutput});
        }
        return result;
    }

    for (const auto& metric : metrics) {
        result.metrics.push_back(parseMetric(caseResult, metric));
    }
    const bool pmuFailed = result.pmuStatus.find("failed") != std::string::npos ||
                           result.pmuStatus.find("error") != std::string::npos;
    if (pmuFailed) {
        result.error = "PMU status reports a collection failure";
        for (auto& metric : result.metrics) {
            if (metric.status == MetricResult::Status::Unsupported) {
                metric.status = MetricResult::Status::CommandFailed;
            }
        }
    }
    return result;
}

static bool validateMetricResult(const MetricResult& metric, const char* collectionKind) {
    const bool recognizedStatus =
        metric.status == MetricResult::Status::Valid ||
        metric.status == MetricResult::Status::Unsupported ||
        metric.status == MetricResult::Status::Overflow;
    if (!recognizedStatus) {
        printf("    %s %s (%s)\n", statusName(metric.status), metric.name.c_str(), collectionKind);
        EXPECT_TRUE(recognizedStatus);
        return false;
    }
    if (metric.status != MetricResult::Status::Valid) {
        return false;
    }
    const bool finiteNonNegative = std::isfinite(metric.value) && metric.value >= 0.0;
    if (!finiteNonNegative) {
        printf("    invalid numeric value for %s (%s): %.17g\n",
               metric.name.c_str(), collectionKind, metric.value);
    }
    EXPECT_TRUE(finiteNonNegative);
    return finiteNonNegative;
}

static bool isReplayInvariantMetric(const std::string& metric) {
    const bool isSum = metric.size() >= 4 &&
                       metric.compare(metric.size() - 4, 4, ".sum") == 0;
    return isSum && (metric.find("inst_executed") != std::string::npos ||
                     metric.find("l1tex__t_sectors") != std::string::npos);
}

static MetricResult runSingleMedian(const std::string& caseName, const std::string& metric) {
    std::vector<double> vals;
    MetricResult::Status lastStatus = MetricResult::Status::Unsupported;
    bool sawOverflow = false;
    bool sawCommandFailure = false;
    bool sawMalformed = false;
    for (int i = 0; i < 3; ++i) {
        BenchResult result = runBench(caseName, {metric});
        if (result.metrics.empty()) continue;
        lastStatus = result.metrics[0].status;
        if (validateMetricResult(result.metrics[0], "single") &&
            lastStatus == MetricResult::Status::Valid) {
            vals.push_back(result.metrics[0].value);
        }
        if (lastStatus == MetricResult::Status::Overflow) sawOverflow = true;
        if (lastStatus == MetricResult::Status::CommandFailed) sawCommandFailure = true;
        if (lastStatus == MetricResult::Status::MalformedOutput) sawMalformed = true;
    }
    if (vals.empty()) {
        MetricResult::Status status = MetricResult::Status::Unsupported;
        if (sawCommandFailure) status = MetricResult::Status::CommandFailed;
        else if (sawMalformed) status = MetricResult::Status::MalformedOutput;
        else if (sawOverflow) status = MetricResult::Status::Overflow;
        return {metric, -1.0, status};
    }
    std::sort(vals.begin(), vals.end());
    return {metric, vals[vals.size() / 2], MetricResult::Status::Valid};
}

// ============================================================
// Kernel case definitions (8 variants)
// ============================================================

struct KernelCase {
    const char* shortName;
    const char* benchCase;
    const char* description;
};

static const KernelCase kernels[] = {
    {"ConvDw",    "cuda_conv_dw_fp32_smoke",               "depthwise conv (compute+memory)"},
    {"Matmul",    "cuda_general_batch_matmul_fp32_smoke",  "GEMM (compute-bound)"},
    {"Gemv",      "cuda_matmul_gemv_fp32_smoke",           "GEMV (memory-bound)"},
    {"Reduction", "cuda_reduction_sum_fp32_smoke",         "reduction sum (memory-bound)"},
    {"MaxPool",   "cuda_maxpool_fp32_smoke",               "max pooling (memory-bound)"},
    {"Softmax",   "cuda_softmax_fp32_smoke",               "softmax (compute+exp)"},
    {"Layernorm", "cuda_layernorm_fp32_smoke",             "layernorm (compute)"},
    {"Relu",      "cuda_relu_fp32_smoke",                  "relu (pure memory)"},
};

// ============================================================
// Metric groups (8 groups)
// ============================================================

struct MetricGroup {
    const char* shortName;
    std::vector<std::string> metrics;
    bool expectMultiPass;
};

static const MetricGroup groups[] = {
    {"Compute", {
        "sm__inst_executed.sum", "sm__cycles_elapsed.sum",
    }, false},
    {"Memory", {
        "dram__bytes.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
    }, false},
    {"Cache", {
        "l1tex__t_sectors.sum", "lts__t_sectors.sum",
    }, false},
    {"Mixed", {
        "sm__inst_executed.sum", "sm__cycles_elapsed.sum",
        "dram__bytes.sum", "lts__t_sectors.sum",
    }, true},
    {"InstrDetail", {
        "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
        "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    }, false},
    {"Warp", {
        "sm__warps_active.sum",
        "smsp__warps_active.sum",
    }, true},
    {"Throughput", {
        "sm__cycles_active.sum", "dram__cycles_active.sum",
    }, true},
    {"AllCombined", {
        "sm__inst_executed.sum", "sm__cycles_elapsed.sum",
        "dram__bytes.sum", "dram__bytes_read.sum",
        "l1tex__t_sectors.sum", "lts__t_sectors.sum",
        "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
        "sm__warps_active.sum",
    }, true},
};

// ============================================================
// Shared validation logic
// ============================================================

static void checkMultiVsSingle(const std::string& caseName,
                                const std::vector<std::string>& metrics,
                                BenchResult* multiResult) {
    BenchResult multi = runBench(caseName, metrics);
    *multiResult = multi;
    if (multi.metrics.size() != metrics.size()) {
        EXPECT_EQ(multi.metrics.size(), metrics.size());
        return;
    }

    int validMetrics = 0;
    for (size_t i = 0; i < metrics.size(); ++i) {
        const MetricResult& multiMetric = multi.metrics[i];
        if (!validateMetricResult(multiMetric, "multi")) {
            printf("    %s %s (multi)\n", statusName(multiMetric.status), metrics[i].c_str());
            continue;
        }
        validMetrics++;
        if (!isReplayInvariantMetric(metrics[i])) {
            printf("    %-55s multi=%14.6g  observed only (cache-uncontrolled replay)\n",
                   metrics[i].c_str(), multiMetric.value);
            continue;
        }

        MetricResult singleMetric = runSingleMedian(caseName, metrics[i]);
        if (!validateMetricResult(singleMetric, "single median")) {
            printf("    %s %s (single)\n", statusName(singleMetric.status), metrics[i].c_str());
            EXPECT_TRUE(singleMetric.status == MetricResult::Status::Valid);
            continue;
        }
        EXPECT_EQ(multiMetric.value, singleMetric.value);
        printf("    %-55s multi=%14.6g single=%14.6g  replay-invariant\n",
               metrics[i].c_str(), multiMetric.value, singleMetric.value);
    }
    EXPECT_TRUE(validMetrics > 0);
}

static void runCombination(int ki, int gi) {
    const auto& kc = kernels[ki];
    const auto& mg = groups[gi];
    printf("  Kernel: %s (%s)\n", kc.shortName, kc.description);
    printf("  Group:  %s (%zu metrics)\n", mg.shortName, mg.metrics.size());
    BenchResult multi;
    checkMultiVsSingle(kc.benchCase, mg.metrics, &multi);
    if (!multi.error.empty()) printf("  Error: %s\n", multi.error.c_str());
    const std::string& status = multi.pmuStatus;
    printf("  Status: %s, passes=%llu\n", status.c_str(),
           static_cast<unsigned long long>(multi.numPasses));
    EXPECT_TRUE(multi.numPasses >= 1);
    EXPECT_TRUE(status.find("sampled") != std::string::npos);
    if (mg.expectMultiPass) {
        EXPECT_TRUE(multi.numPasses > 1);
    }
}

// ============================================================
// 64 tests: 8 kernels × 8 metric groups
// ============================================================

TEST(PMU, ConvDw_Compute)        { runCombination(0, 0); }
TEST(PMU, ConvDw_Memory)        { runCombination(0, 1); }
TEST(PMU, ConvDw_Cache)         { runCombination(0, 2); }
TEST(PMU, ConvDw_Mixed)         { runCombination(0, 3); }
TEST(PMU, ConvDw_InstrDetail)   { runCombination(0, 4); }
TEST(PMU, ConvDw_Warp)          { runCombination(0, 5); }
TEST(PMU, ConvDw_Throughput)    { runCombination(0, 6); }
TEST(PMU, ConvDw_AllCombined)   { runCombination(0, 7); }

TEST(PMU, Matmul_Compute)       { runCombination(1, 0); }
TEST(PMU, Matmul_Memory)        { runCombination(1, 1); }
TEST(PMU, Matmul_Cache)         { runCombination(1, 2); }
TEST(PMU, Matmul_Mixed)         { runCombination(1, 3); }
TEST(PMU, Matmul_InstrDetail)   { runCombination(1, 4); }
TEST(PMU, Matmul_Warp)          { runCombination(1, 5); }
TEST(PMU, Matmul_Throughput)    { runCombination(1, 6); }
TEST(PMU, Matmul_AllCombined)   { runCombination(1, 7); }

TEST(PMU, Gemv_Compute)         { runCombination(2, 0); }
TEST(PMU, Gemv_Memory)         { runCombination(2, 1); }
TEST(PMU, Gemv_Cache)          { runCombination(2, 2); }
TEST(PMU, Gemv_Mixed)          { runCombination(2, 3); }
TEST(PMU, Gemv_InstrDetail)    { runCombination(2, 4); }
TEST(PMU, Gemv_Warp)           { runCombination(2, 5); }
TEST(PMU, Gemv_Throughput)     { runCombination(2, 6); }
TEST(PMU, Gemv_AllCombined)    { runCombination(2, 7); }

TEST(PMU, Reduction_Compute)    { runCombination(3, 0); }
TEST(PMU, Reduction_Memory)     { runCombination(3, 1); }
TEST(PMU, Reduction_Cache)      { runCombination(3, 2); }
TEST(PMU, Reduction_Mixed)     { runCombination(3, 3); }
TEST(PMU, Reduction_InstrDetail){ runCombination(3, 4); }
TEST(PMU, Reduction_Warp)      { runCombination(3, 5); }
TEST(PMU, Reduction_Throughput){ runCombination(3, 6); }
TEST(PMU, Reduction_AllCombined){runCombination(3, 7); }

TEST(PMU, MaxPool_Compute)     { runCombination(4, 0); }
TEST(PMU, MaxPool_Memory)      { runCombination(4, 1); }
TEST(PMU, MaxPool_Cache)       { runCombination(4, 2); }
TEST(PMU, MaxPool_Mixed)       { runCombination(4, 3); }
TEST(PMU, MaxPool_InstrDetail) { runCombination(4, 4); }
TEST(PMU, MaxPool_Warp)        { runCombination(4, 5); }
TEST(PMU, MaxPool_Throughput)  { runCombination(4, 6); }
TEST(PMU, MaxPool_AllCombined) { runCombination(4, 7); }

TEST(PMU, Softmax_Compute)     { runCombination(5, 0); }
TEST(PMU, Softmax_Memory)      { runCombination(5, 1); }
TEST(PMU, Softmax_Cache)       { runCombination(5, 2); }
TEST(PMU, Softmax_Mixed)       { runCombination(5, 3); }
TEST(PMU, Softmax_InstrDetail) { runCombination(5, 4); }
TEST(PMU, Softmax_Warp)        { runCombination(5, 5); }
TEST(PMU, Softmax_Throughput)  { runCombination(5, 6); }
TEST(PMU, Softmax_AllCombined) { runCombination(5, 7); }

TEST(PMU, Layernorm_Compute)   { runCombination(6, 0); }
TEST(PMU, Layernorm_Memory)    { runCombination(6, 1); }
TEST(PMU, Layernorm_Cache)     { runCombination(6, 2); }
TEST(PMU, Layernorm_Mixed)     { runCombination(6, 3); }
TEST(PMU, Layernorm_InstrDetail){runCombination(6, 4); }
TEST(PMU, Layernorm_Warp)      { runCombination(6, 5); }
TEST(PMU, Layernorm_Throughput){ runCombination(6, 6); }
TEST(PMU, Layernorm_AllCombined){runCombination(6, 7); }

TEST(PMU, Relu_Compute)         { runCombination(7, 0); }
TEST(PMU, Relu_Memory)         { runCombination(7, 1); }
TEST(PMU, Relu_Cache)          { runCombination(7, 2); }
TEST(PMU, Relu_Mixed)          { runCombination(7, 3); }
TEST(PMU, Relu_InstrDetail)    { runCombination(7, 4); }
TEST(PMU, Relu_Warp)           { runCombination(7, 5); }
TEST(PMU, Relu_Throughput)     { runCombination(7, 6); }
TEST(PMU, Relu_AllCombined)    { runCombination(7, 7); }

// ============================================================
// Extra tests
// ============================================================

TEST(PMU, InstructionReproducibility_AllKernels) {
    int valid = 0;
    for (int k = 0; k < 8; ++k) {
        BenchResult r1 = runBench(kernels[k].benchCase, {"sm__inst_executed.sum"});
        if (r1.metrics.empty() || r1.metrics[0].status != MetricResult::Status::Valid) {
            MetricResult::Status status = r1.metrics.empty()
                ? MetricResult::Status::MalformedOutput : r1.metrics[0].status;
            printf("  %s %s\n", statusName(status), kernels[k].shortName);
            if (status != MetricResult::Status::Unsupported) EXPECT_TRUE(false);
            continue;
        }
        validateMetricResult(r1.metrics[0], "reproducibility first run");
        BenchResult r2 = runBench(kernels[k].benchCase, {"sm__inst_executed.sum"});
        if (r2.metrics.empty() || r2.metrics[0].status != MetricResult::Status::Valid) {
            MetricResult::Status status = r2.metrics.empty()
                ? MetricResult::Status::MalformedOutput : r2.metrics[0].status;
            printf("  %s %s (second run)\n", statusName(status), kernels[k].shortName);
            if (status != MetricResult::Status::Unsupported) EXPECT_TRUE(false);
            continue;
        }
        validateMetricResult(r2.metrics[0], "reproducibility second run");
        valid++;
        EXPECT_EQ(r1.metrics[0].value, r2.metrics[0].value);
        printf("  %-12s inst_executed.sum = %.0f\n", kernels[k].shortName, r1.metrics[0].value);
    }
    EXPECT_TRUE(valid > 0);
}

TEST(PMU, AutoSplitStatus) {
    BenchResult single = runBench("cuda_conv_dw_fp32_smoke", {"sm__inst_executed.sum"});
    std::string s1 = single.pmuStatus;
    EXPECT_TRUE(single.metrics.size() == 1 &&
                single.metrics[0].status == MetricResult::Status::Valid);
    if (!single.metrics.empty()) validateMetricResult(single.metrics[0], "auto-split single");
    EXPECT_TRUE(single.numPasses >= 1);
    EXPECT_TRUE(s1.find("sampled") != std::string::npos);

    BenchResult multi = runBench("cuda_conv_dw_fp32_smoke", {
        "sm__inst_executed.sum", "sm__cycles_elapsed.sum",
        "dram__bytes.sum", "lts__t_sectors.sum",
        "sm__warps_active.sum", "l1tex__t_sectors.sum",
    });
    std::string s2 = multi.pmuStatus;
    EXPECT_TRUE(!multi.metrics.empty());
    int validMetrics = 0;
    for (const auto& metric : multi.metrics) {
        if (validateMetricResult(metric, "auto-split multi")) {
            validMetrics++;
        }
    }
    EXPECT_TRUE(validMetrics > 0);
    EXPECT_TRUE(multi.numPasses > 1);
    EXPECT_TRUE(s2.find("sampled") != std::string::npos);
    printf("  single: %s, passes=%llu\n  multi:  %s, passes=%llu\n",
           s1.c_str(), static_cast<unsigned long long>(single.numPasses),
           s2.c_str(), static_cast<unsigned long long>(multi.numPasses));
}

int main(int argc, char** argv) {
    printf("=== Replay PMU Multi-pass Test Suite ===\n");
    printf("    64 combinations (8 kernels x 8 metric groups) + 2 extra tests\n\n");
    return ReplayTest::RunAll(argc, argv);
}
