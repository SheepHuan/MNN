#ifndef MNN_REPLAY_KERNEL_CORPUS_BENCHMARK_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BENCHMARK_HPP

#include "CudaRuntimeMetadata.hpp"

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

enum class PmuMetricValueKind {
    UnsignedInteger,
    FloatingPoint,
};

enum class PmuMetricValueStatus {
    Invalid,
    Valid,
    Overflow,
};

struct PmuMetricValue {
    std::string name;
    uint64_t integerValue = 0;
    double floatingPointValue = 0.0;
    PmuMetricValueKind valueKind = PmuMetricValueKind::UnsignedInteger;
    PmuMetricValueStatus status = PmuMetricValueStatus::Invalid;
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

    // PMU metric values retain their native representation: Adreno/Mali raw
    // counters remain uint64 while CUDA/NVPW values remain double. Invalid or
    // overflowing metrics keep an explicit per-metric status and no JSON value.
    std::vector<PmuMetricValue> pmuMetrics;
    // Number of CUPTI passes required for the metric set (1=single-pass, N=replay)
    size_t numPasses = 1;

    // Actual CUDA kernel executions observed during the independent,
    // PMU-disabled latency workload. One adapter dispatch may emit multiple
    // launch records (for example a two-stage reduction).
    std::string launchSamplingStatus = "not_requested";
    std::vector<CudaLaunchRecord> launchRecords;

    // Current SM clock and GPU temperature sampled outside the timed workload
    // or profiler range. Missing values remain absent in JSON and are explained
    // by samplingStatus.
    CudaEnvironmentObservation environment;
};

namespace detail {

enum class AdapterValidationDecision {
    Accept,
    Reject,
    TryLegacy,
};

// CUDA adapters own the complete output semantics for their kernels. A failed
// CUDA adapter validation is terminal so a weaker legacy validator cannot
// rescue an output with the wrong layout. Non-CUDA adapters retain the legacy
// fallback while those paths are migrated.
inline AdapterValidationDecision adapterValidationDecision(
    bool hasAdapter, bool isCudaAdapter, bool adapterValidationPassed) {
    if (!hasAdapter) return AdapterValidationDecision::TryLegacy;
    if (adapterValidationPassed) return AdapterValidationDecision::Accept;
    return isCudaAdapter ? AdapterValidationDecision::Reject
                         : AdapterValidationDecision::TryLegacy;
}

} // namespace detail

// Run the kernel corpus benchmark and return false only on unrecoverable
// initialization failure. Per-case failures are reported in the returned
// document, not by aborting the whole run.
bool runKernelCorpusBenchmark(const Options& options);

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BENCHMARK_HPP
