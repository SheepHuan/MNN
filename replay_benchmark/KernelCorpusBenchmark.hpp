#ifndef MNN_REPLAY_KERNEL_CORPUS_BENCHMARK_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BENCHMARK_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace MNN {
namespace Replay {
namespace KernelCorpus {

struct Options {
    std::string corpusRoot;
    std::string operatorsFile = "operators.json";
    std::string casesFile = "operator_cases.json";
    std::string caseFilter;
    int runs = 1;
    bool measureLatency = true;
    std::string perfCounterOutput;
    std::string perfCounterEvents;
};

struct CaseReport {
    std::string framework;
    std::string tag;
    std::string backend;
    std::string opType;
    std::string variant;
    std::string caseName;
    std::string compileStatus;
    std::string dispatchStatus;
    std::string validationStatus;
    std::string pmuStatus;
    uint64_t controlDelta = 0;
    uint64_t workloadDelta = 0;
    // CUDA event latency for the workload, in microseconds. This is kept
    // separate from workloadDelta because workloadDelta is the first PMU
    // value when PMU sampling succeeds.
    double latencyUs = 0.0;
    bool responsive = false;
    bool valid = false;
    std::string error;

    // Per-case wall-clock timing (nanoseconds). workloadNs is total time for
    // `runs` dispatches; nsPerDispatch = workloadNs / runs.
    uint64_t workloadNs = 0;
    int runs = 0;

    // PMU metric values (name → value), populated when PMU is available
    std::vector<std::pair<std::string, uint64_t>> pmuMetrics;
    // Number of CUPTI passes required for the metric set (1=single-pass, N=replay)
    size_t numPasses = 1;
};

// Run the kernel corpus benchmark and return false only on unrecoverable
// initialization failure. Per-case failures are reported in the returned
// document, not by aborting the whole run.
bool runKernelCorpusBenchmark(const Options& options);

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BENCHMARK_HPP
