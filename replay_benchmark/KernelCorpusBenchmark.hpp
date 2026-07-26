#ifndef MNN_REPLAY_KERNEL_CORPUS_BENCHMARK_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BENCHMARK_HPP

#include <cstddef>
#include <cstdint>
#include <string>
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
    bool responsive = false;
    bool valid = false;
    std::string error;
};

// Run the kernel corpus benchmark and return false only on unrecoverable
// initialization failure. Per-case failures are reported in the returned
// document, not by aborting the whole run.
bool runKernelCorpusBenchmark(const Options& options);

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BENCHMARK_HPP
