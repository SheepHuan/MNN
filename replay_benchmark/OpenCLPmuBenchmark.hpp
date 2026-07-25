#ifndef MNN_REPLAY_OPENCL_PMU_BENCHMARK_HPP
#define MNN_REPLAY_OPENCL_PMU_BENCHMARK_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace MNN {
namespace Replay {

struct Options;

enum class OpenCLPmuMemoryPath {
    Buffer,
    Image,
    Texture,
    Compute,
};

struct OpenCLPmuCaseInfo {
    const char* name;
    const char* concept;
    const char* kernel;
    const char* requiredExtension;
    OpenCLPmuMemoryPath path;
    bool expectedIncrease;
    uint32_t defaultIterations;
};

struct OpenCLPmuSignalResult {
    bool readable = false;
    bool responsive = false;
    bool discriminative = false;
    bool valid = false;
};

struct OpenCLPmuGeometry {
    size_t global[2] = {0, 0};
    size_t local[2] = {0, 0};
    bool usesLocalSize = false;
};

size_t openclPmuCaseCount();
const OpenCLPmuCaseInfo* openclPmuCases();
const OpenCLPmuCaseInfo* findOpenclPmuCase(const char* name);
std::vector<std::string> parseOpenclPmuCaseList(const std::string& text);
OpenCLPmuSignalResult classifyPmuSignal(uint64_t controlDelta, uint64_t workloadDelta,
                                        bool expectedIncrease, uint64_t threshold);
OpenCLPmuGeometry makeOpenclPmuGeometry(const OpenCLPmuCaseInfo& info, size_t count, size_t localSize);

// Returns false only when the benchmark mode cannot be initialized or its
// report cannot be written. Per-case capability and counter failures are kept
// in the report rather than aborting the whole run.
bool runOpenCLPmuBenchmark(const Options& options);

} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_OPENCL_PMU_BENCHMARK_HPP
