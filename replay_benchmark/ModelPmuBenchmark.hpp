#ifndef MNN_REPLAY_MODEL_PMU_BENCHMARK_HPP
#define MNN_REPLAY_MODEL_PMU_BENCHMARK_HPP

#include <cstdint>
#include <string>

namespace MNN {
namespace Replay {

struct Options;

struct ModelPmuRunConfig {
    std::string model;
    int forward = 3;
    int precision = 0;
    int warmupRuns = 2;
    int controlRuns = 1;
    int workloadRuns = 5;
    std::string event;
    std::string output;
};

struct ModelPmuDelta {
    uint64_t control = 0;
    uint64_t workload = 0;
    int64_t delta = 0;
};

ModelPmuDelta makeModelPmuDelta(uint64_t control, uint64_t workload);
bool validateModelPmuRunConfig(const ModelPmuRunConfig& config, std::string* error);
bool runModelPmuBenchmark(const Options& options);

} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_MODEL_PMU_BENCHMARK_HPP
