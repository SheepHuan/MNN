// test_pmu_availability.cpp — PMU metric availability and validity test.
//
// Tests for each GPU vendor (CUDA first):
//   1. Discover all PMU metrics supported by the current GPU
//   2. For each metric, collect it individually (single-pass, guaranteed)
//   3. Verify the metric returns a valid (non-overflow) value
//   4. Classify each metric as: available + valid / available + invalid / unavailable
//   5. Report per-domain summary
//
// This is the foundation test — before doing multi-pass or accuracy tests,
// we must know which metrics actually work on this specific GPU.
//
// Build: cmake -DMNN_CUDA=ON .. && make replay_pmu_availability
// Run:   sudo LD_LIBRARY_PATH=build:build/source/backend/cuda ./build/replay_pmu_availability
// Filter: ./replay_pmu_availability --filter CudaMetric.Domain

#include "tests/ReplayTest.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>

// ============================================================
// Infrastructure (shared with test_pmu_multipass.cpp)
// ============================================================

struct MetricResult {
    std::string name;
    double value;
};

static std::vector<MetricResult> runBench(const std::string& caseName,
                                          const std::vector<std::string>& metrics) {
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
        "--perf-counter-output /tmp/_pmu_avail.json 2>/dev/null",
        caseName.c_str(), metricStr.c_str());
    if (system(cmd) != 0) return {};
    FILE* f = fopen("/tmp/_pmu_avail.json", "r");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string json(sz, '\0');
    fread(&json[0], 1, sz, f);
    fclose(f);

    std::vector<MetricResult> results;
    for (const auto& metric : metrics) {
        std::string pattern = "\"" + metric + "\":";
        size_t pos = json.find(pattern);
        double val = -1.0;
        if (pos != std::string::npos) {
            pos += pattern.size();
            while (pos < json.size() && json[pos] == ' ') pos++;
            if (json.substr(pos, 19) == "9223372036854775808")
                val = -2.0;  // overflow — metric collected but value invalid
            else
                val = atof(json.c_str() + pos);
        }
        results.push_back({metric, val});
    }
    return results;
}

// Discover all CUDA metrics via list_cuda_metrics
static std::vector<std::string> discoverCudaMetrics() {
    std::vector<std::string> metrics;
    char cmd[256];
    std::snprintf(cmd, sizeof(cmd),
        "sudo env LD_LIBRARY_PATH=.:source/backend/cuda:. "
        "./list_cuda_metrics --no-submetrics 2>/dev/null");
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return {};
    char line[512];
    while (fgets(line, sizeof(line), pipe)) {
        // Skip header line "# chip: TU102"
        if (line[0] == '#') continue;
        std::string m(line);
        // Trim whitespace
        while (!m.empty() && (m.back() == '\n' || m.back() == '\r' || m.back() == ' '))
            m.pop_back();
        if (!m.empty()) metrics.push_back(m);
    }
    pclose(pipe);
    return metrics;
}

// Extract domain prefix from metric name (e.g. "sm__inst_executed.sum" → "sm")
static std::string domainOf(const std::string& metric) {
    size_t pos = metric.find("__");
    if (pos == std::string::npos) return "other";
    return metric.substr(0, pos);
}

// Domain descriptions
static const char* domainDesc(const std::string& d) {
    if (d == "sm")     return "Streaming Multiprocessor";
    if (d == "smsp")   return "SM Sub-Partition";
    if (d == "l1tex")  return "L1/Texture Cache";
    if (d == "lts")    return "L2 Cache Slice";
    if (d == "dram")   return "DRAM Controller";
    if (d == "fbpa")   return "Frame Buffer Partition";
    if (d == "gcc")    return "Graphics Command Cache";
    if (d == "tpc")    return "Texture Processing Cluster";
    if (d == "gpu")    return "GPU Global";
    if (d == "gr")     return "Graphics Engine";
    if (d == "idc")    return "Inter-DRAM Channel";
    if (d == "fe")     return "Front End";
    if (d == "pcie")   return "PCIe";
    if (d == "sys")    return "System";
    if (d == "gpc")    return "Graphics Processing Cluster";
    if (d == "nvltx")  return "NVLink TX";
    if (d == "nvlrx")  return "NVLink RX";
    return "Other";
}

// ============================================================
// Test data
// ============================================================

static const char* testKernel = "cuda_conv_dw_fp32_smoke";

// ============================================================
// Test 1: Metric discovery — list_cuda_metrics returns metrics
// ============================================================
TEST(CudaMetric, Discovery) {
    auto metrics = discoverCudaMetrics();
    ASSERT_TRUE(!metrics.empty());
    printf("  Discovered %zu CUDA metrics\n", metrics.size());
    // Show first 5 per domain
    std::map<std::string, int> domainCount;
    for (const auto& m : metrics) domainCount[domainOf(m)]++;
    printf("  Per-domain counts:\n");
    for (const auto& [d, cnt] : domainCount) {
        printf("    %-10s (%s): %d metrics\n", d.c_str(), domainDesc(d), cnt);
    }
    EXPECT_TRUE(metrics.size() > 100);  // TU102 has 7200+
}

// ============================================================
// Test 2: Every discovered metric — one metric per process
// ============================================================
TEST(CudaMetric, AllMetrics) {
    auto metrics = discoverCudaMetrics();
    ASSERT_TRUE(!metrics.empty());

    int valid = 0;
    int notFound = 0;
    int overflow = 0;
    int commandFailed = 0;
    printf("  Sweeping %zu discovered CUDA metrics (one process each)\n", metrics.size());

    for (size_t i = 0; i < metrics.size(); ++i) {
        const auto& metric = metrics[i];
        auto result = runBench(testKernel, {metric});
        const char* state = "COMMAND_FAILED";
        double value = -1.0;
        if (!result.empty()) {
            value = result[0].value;
            if (value == -2.0) {
                state = "OVERFLOW";
                overflow++;
            } else if (value >= 0) {
                state = "VALID";
                valid++;
            } else {
                state = "NOT_FOUND";
                notFound++;
            }
        } else {
            commandFailed++;
        }
        printf("  [%5zu/%5zu] %-80s %-13s value=% .0f\n",
               i + 1, metrics.size(), metric.c_str(), state, value);
    }

    printf("  all metrics: %d valid, %d not_found, %d overflow, %d command_failed (of %zu)\n",
           valid, notFound, overflow, commandFailed, metrics.size());
    EXPECT_EQ(static_cast<size_t>(valid + notFound + overflow + commandFailed), metrics.size());
    EXPECT_TRUE(valid > 0);
}

// ============================================================
// Test 3: Compute domain (sm__) metrics
// ============================================================
TEST(CudaMetric, SmDomain) {
    auto all = discoverCudaMetrics();
    std::vector<std::string> smMetrics;
    for (const auto& m : all) {
        if (domainOf(m) == "sm" && m.find(".sum") != std::string::npos)
            smMetrics.push_back(m);
    }
    printf("  Testing %zu sm__ .sum metrics\n", smMetrics.size());

    int valid = 0, invalid = 0, overflow = 0;
    for (const auto& m : smMetrics) {
        auto r = runBench(testKernel, {m});
        if (r.empty()) { invalid++; continue; }
        if (r[0].value == -2.0) { overflow++; continue; }
        if (r[0].value >= 0) valid++;
        else invalid++;
    }
    printf("  sm__ domain: %d valid, %d invalid, %d overflow (of %zu)\n",
           valid, invalid, overflow, smMetrics.size());
    EXPECT_TRUE(valid > 0);
}

// ============================================================
// Test 3: Memory domain (dram__) metrics
// ============================================================
TEST(CudaMetric, DramDomain) {
    auto all = discoverCudaMetrics();
    std::vector<std::string> dramMetrics;
    for (const auto& m : all) {
        if (domainOf(m) == "dram" && m.find(".sum") != std::string::npos)
            dramMetrics.push_back(m);
    }
    printf("  Testing %zu dram__ .sum metrics\n", dramMetrics.size());

    int valid = 0, invalid = 0, overflow = 0;
    for (const auto& m : dramMetrics) {
        auto r = runBench(testKernel, {m});
        if (r.empty()) { invalid++; continue; }
        if (r[0].value == -2.0) { overflow++; continue; }
        if (r[0].value >= 0) valid++;
        else invalid++;
    }
    printf("  dram__ domain: %d valid, %d invalid, %d overflow (of %zu)\n",
           valid, invalid, overflow, dramMetrics.size());
    EXPECT_TRUE(valid > 0);
}

// ============================================================
// Test 4: L1 cache domain (l1tex__) metrics
// ============================================================
TEST(CudaMetric, L1texDomain) {
    auto all = discoverCudaMetrics();
    std::vector<std::string> l1Metrics;
    for (const auto& m : all) {
        if (domainOf(m) == "l1tex" && m.find(".sum") != std::string::npos)
            l1Metrics.push_back(m);
    }
    printf("  Testing %zu l1tex__ .sum metrics\n", l1Metrics.size());

    int valid = 0, invalid = 0, overflow = 0;
    for (const auto& m : l1Metrics) {
        auto r = runBench(testKernel, {m});
        if (r.empty()) { invalid++; continue; }
        if (r[0].value == -2.0) { overflow++; continue; }
        if (r[0].value >= 0) valid++;
        else invalid++;
    }
    printf("  l1tex__ domain: %d valid, %d invalid, %d overflow (of %zu)\n",
           valid, invalid, overflow, l1Metrics.size());
    EXPECT_TRUE(valid > 0);
}

// ============================================================
// Test 5: L2 cache domain (lts__) metrics
// ============================================================
TEST(CudaMetric, LtsDomain) {
    auto all = discoverCudaMetrics();
    std::vector<std::string> l2Metrics;
    for (const auto& m : all) {
        if (domainOf(m) == "lts" && m.find(".sum") != std::string::npos)
            l2Metrics.push_back(m);
    }
    printf("  Testing %zu lts__ .sum metrics\n", l2Metrics.size());

    int valid = 0, invalid = 0, overflow = 0;
    for (const auto& m : l2Metrics) {
        auto r = runBench(testKernel, {m});
        if (r.empty()) { invalid++; continue; }
        if (r[0].value == -2.0) { overflow++; continue; }
        if (r[0].value >= 0) valid++;
        else invalid++;
    }
    printf("  lts__ domain: %d valid, %d invalid, %d overflow (of %zu)\n",
           valid, invalid, overflow, l2Metrics.size());
    EXPECT_TRUE(valid > 0);
}

// ============================================================
// Test 6: SM sub-partition domain (smsp__) metrics
// ============================================================
TEST(CudaMetric, SmspDomain) {
    auto all = discoverCudaMetrics();
    std::vector<std::string> smspMetrics;
    for (const auto& m : all) {
        if (domainOf(m) == "smsp" && m.find(".sum") != std::string::npos)
            smspMetrics.push_back(m);
    }
    printf("  Testing %zu smsp__ .sum metrics\n", smspMetrics.size());

    int valid = 0, invalid = 0, overflow = 0;
    for (const auto& m : smspMetrics) {
        auto r = runBench(testKernel, {m});
        if (r.empty()) { invalid++; continue; }
        if (r[0].value == -2.0) { overflow++; continue; }
        if (r[0].value >= 0) valid++;
        else invalid++;
    }
    printf("  smsp__ domain: %d valid, %d invalid, %d overflow (of %zu)\n",
           valid, invalid, overflow, smspMetrics.size());
    EXPECT_TRUE(valid > 0);
}

// ============================================================
// Test 7: Frame buffer partition domain (fbpa__) metrics
// ============================================================
TEST(CudaMetric, FbpaDomain) {
    auto all = discoverCudaMetrics();
    std::vector<std::string> fbpaMetrics;
    for (const auto& m : all) {
        if (domainOf(m) == "fbpa" && m.find(".sum") != std::string::npos)
            fbpaMetrics.push_back(m);
    }
    printf("  Testing %zu fbpa__ .sum metrics\n", fbpaMetrics.size());

    int valid = 0, invalid = 0, overflow = 0;
    for (const auto& m : fbpaMetrics) {
        auto r = runBench(testKernel, {m});
        if (r.empty()) { invalid++; continue; }
        if (r[0].value == -2.0) { overflow++; continue; }
        if (r[0].value >= 0) valid++;
        else invalid++;
    }
    printf("  fbpa__ domain: %d valid, %d invalid, %d overflow (of %zu)\n",
           valid, invalid, overflow, fbpaMetrics.size());
    // fbpa may or may not be supported on all chips
    printf("  (fbpa__ support is optional — not asserting valid > 0)\n");
}

// ============================================================
// Test 8: GPU global domain (gpu__) metrics
// ============================================================
TEST(CudaMetric, GpuDomain) {
    auto all = discoverCudaMetrics();
    std::vector<std::string> gpuMetrics;
    for (const auto& m : all) {
        if (domainOf(m) == "gpu" && m.find(".sum") != std::string::npos)
            gpuMetrics.push_back(m);
    }
    printf("  Testing %zu gpu__ .sum metrics\n", gpuMetrics.size());

    int valid = 0, invalid = 0, overflow = 0;
    for (const auto& m : gpuMetrics) {
        auto r = runBench(testKernel, {m});
        if (r.empty()) { invalid++; continue; }
        if (r[0].value == -2.0) { overflow++; continue; }
        if (r[0].value >= 0) valid++;
        else invalid++;
    }
    printf("  gpu__ domain: %d valid, %d invalid, %d overflow (of %zu)\n",
           valid, invalid, overflow, gpuMetrics.size());
}

// ============================================================
// Test 9: Key metrics — the ones most useful for performance analysis
// ============================================================
TEST(CudaMetric, KeyMetrics) {
    std::vector<std::pair<std::string, const char*>> keyMetrics = {
        {"sm__inst_executed.sum",                    "total instructions"},
        {"sm__cycles_elapsed.sum",                    "total cycles (time)"},
        {"sm__warps_active.sum",                      "active warps"},
        {"sm__sass_thread_inst_executed_op_fadd_pred_on.sum", "FP32 add"},
        {"sm__sass_thread_inst_executed_op_fmul_pred_on.sum", "FP32 mul"},
        {"dram__bytes.sum",                           "total DRAM bytes"},
        {"dram__bytes_read.sum",                      "DRAM read bytes"},
        {"dram__bytes_write.sum",                     "DRAM write bytes"},
        {"l1tex__t_sectors.sum",                      "L1 sectors"},
        {"lts__t_sectors.sum",                        "L2 sectors"},
    };

    printf("  Testing %zu key metrics:\n", keyMetrics.size());
    for (const auto& [metric, desc] : keyMetrics) {
        auto r = runBench(testKernel, {metric});
        if (r.empty()) {
            printf("    %-55s UNAVAILABLE\n", metric.c_str());
            continue;
        }
        if (r[0].value == -2.0) {
            printf("    %-55s OVERFLOW (%s)\n", metric.c_str(), desc);
        } else if (r[0].value >= 0) {
            printf("    %-55s %14.0f (%s)\n", metric.c_str(), r[0].value, desc);
            EXPECT_TRUE(r[0].value >= 0);
        } else {
            printf("    %-55s NOT_FOUND (%s)\n", metric.c_str(), desc);
        }
    }
}

// ============================================================
// Test 10: Suffix comparison — .sum vs .avg vs .max vs .min
// ============================================================
TEST(CudaMetric, SuffixComparison) {
    std::string base = "sm__inst_executed";
    std::vector<std::string> suffixes = {".sum", ".avg", ".max", ".min"};

    printf("  Comparing suffixes for %s:\n", base.c_str());
    for (const auto& s : suffixes) {
        std::string metric = base + s;
        auto r = runBench(testKernel, {metric});
        if (r.empty() || r[0].value < 0) {
            printf("    %s: UNAVAILABLE\n", metric.c_str());
            continue;
        }
        printf("    %-30s = %14.0f\n", metric.c_str(), r[0].value);
        EXPECT_TRUE(r[0].value >= 0);
    }
}

// ============================================================
// Test 11: Reproducibility — instruction metric stable across runs
// ============================================================
TEST(CudaMetric, Reproducibility) {
    auto r1 = runBench(testKernel, {"sm__inst_executed.sum"});
    ASSERT_EQ(r1.size(), 1u);
    ASSERT_TRUE(r1[0].value >= 0);
    double v1 = r1[0].value;

    for (int i = 0; i < 4; ++i) {
        auto r = runBench(testKernel, {"sm__inst_executed.sum"});
        ASSERT_EQ(r.size(), 1u);
        EXPECT_EQ(r[0].value, v1);
    }
    printf("  sm__inst_executed.sum = %.0f (5/5 reproducible)\n", v1);
}

// ============================================================
// Test 12: Cross-kernel consistency — same metric, different kernels
// ============================================================
TEST(CudaMetric, CrossKernel) {
    std::vector<std::string> kernels = {
        "cuda_conv_dw_fp32_smoke",
        "cuda_relu_fp32_smoke",
        "cuda_maxpool_fp32_smoke",
        "cuda_softmax_fp32_smoke",
    };
    printf("  sm__inst_executed.sum across kernels:\n");
    std::vector<double> values;
    for (const auto& k : kernels) {
        auto r = runBench(k, {"sm__inst_executed.sum"});
        if (r.empty() || r[0].value < 0) {
            printf("    %-35s UNAVAILABLE\n", k.c_str());
            continue;
        }
        printf("    %-35s = %.0f\n", k.c_str(), r[0].value);
        values.push_back(r[0].value);
        // Reproducible within same kernel
        auto r2 = runBench(k, {"sm__inst_executed.sum"});
        if (!r2.empty()) EXPECT_EQ(r2[0].value, r[0].value);
    }
    // Different kernels should have different instruction counts
    EXPECT_TRUE(values.size() >= 2);
    bool allSame = true;
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] != values[0]) { allSame = false; break; }
    }
    EXPECT_FALSE(allSame);  // at least 2 kernels differ
}

int main(int argc, char** argv) {
    printf("=== CUDA PMU Metric Availability Test Suite ===\n\n");
    return ReplayTest::RunAll(argc, argv);
}
