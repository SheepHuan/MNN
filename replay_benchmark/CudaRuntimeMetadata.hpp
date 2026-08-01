#ifndef MNN_REPLAY_CUDA_RUNTIME_METADATA_HPP
#define MNN_REPLAY_CUDA_RUNTIME_METADATA_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace MNN {
namespace Replay {
namespace KernelCorpus {

struct CudaLaunchRecord {
    std::string kernelName;
    int64_t grid[3] = {0, 0, 0};
    int64_t block[3] = {0, 0, 0};
    uint64_t registersPerThread = 0;
    uint64_t staticSharedMemoryBytes = 0;
    uint64_t dynamicSharedMemoryBytes = 0;
    uint64_t localMemoryPerThreadBytes = 0;
    uint64_t localMemoryTotalBytes = 0;
};

struct OptionalRuntimeNumber {
    bool present = false;
    double value = 0.0;
};

struct CudaEnvironmentObservation {
    std::string samplingSource = "nvml";
    std::string samplingStatus = "not_sampled";
    OptionalRuntimeNumber gpuClockHzBefore;
    OptionalRuntimeNumber gpuClockHzAfter;
    OptionalRuntimeNumber temperatureCBefore;
    OptionalRuntimeNumber temperatureCAfter;
};

// Collects the actual kernel launches observed by CUPTI Activity. It is used
// only for the independent, PMU-disabled latency workload so profiler replay
// cannot duplicate launch records.
class CudaLaunchMetadataCollector {
public:
    CudaLaunchMetadataCollector();
    ~CudaLaunchMetadataCollector();

    bool start();
    void stop();
    const std::vector<CudaLaunchRecord>& records() const;
    const std::string& status() const;

private:
    CudaLaunchMetadataCollector(const CudaLaunchMetadataCollector&) = delete;
    CudaLaunchMetadataCollector& operator=(const CudaLaunchMetadataCollector&) = delete;
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

// Samples the current SM clock and GPU temperature through NVML. Samples are
// deliberately taken outside CUDA event timing and CUPTI ranges.
class CudaEnvironmentSampler {
public:
    CudaEnvironmentSampler();
    ~CudaEnvironmentSampler();

    void sampleBefore();
    void sampleAfter();
    const CudaEnvironmentObservation& observation() const;

private:
    CudaEnvironmentSampler(const CudaEnvironmentSampler&) = delete;
    CudaEnvironmentSampler& operator=(const CudaEnvironmentSampler&) = delete;
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_CUDA_RUNTIME_METADATA_HPP
