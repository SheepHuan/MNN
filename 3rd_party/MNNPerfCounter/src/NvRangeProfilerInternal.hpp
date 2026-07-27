#ifndef MNN_PERF_COUNTER_NV_INTERNAL_HPP
#define MNN_PERF_COUNTER_NV_INTERNAL_HPP

// Internal header exposing NVIDIA Range Profiler helpers used by
// MNNPerfCounter.cpp when MNN_PERFCOUNTER_HAS_CUDA is defined. These are
// implementation details, not part of the public MNNPerfCounter API.

#include <cstdint>
#include <string>
#include <vector>

namespace MNN {
namespace PerfCounter {

// Build the NVPA config image from a list of metric names for the given chip.
// Returns false and sets *error on failure.
bool nvBuildConfigImage(const std::string& chipName,
                        const std::vector<std::string>& metricNames,
                        std::vector<uint8_t>* configImage,
                        std::string* error);

// Build the counterDataPrefix image for the given metrics/chip.
bool nvBuildCounterDataPrefix(const std::string& chipName,
                              const std::vector<std::string>& metricNames,
                              std::vector<uint8_t>* prefix,
                              std::string* error);

// Evaluate metric values from a decoded counterDataImage. Fills values[i]
// with the summed metric value across all ranges (for the metric at index i).
// Returns false on failure.
bool nvEvaluateMetrics(const std::string& chipName,
                       const std::vector<uint8_t>& counterDataImage,
                       const std::vector<std::string>& metricNames,
                       CounterValue* values, size_t count,
                       std::string* error);

// Range push/pop on an opaque CUpti_RangeProfiler_Object pointer. Used by
// Session::beginRange/endRange. rangeObj is CUpti_RangeProfiler_Object*.
bool nvBeginRange(void* rangeObj, const char* rangeName);
bool nvEndRange(void* rangeObj);

} // namespace PerfCounter
} // namespace MNN

#endif // MNN_PERFCOUNTER_NV_INTERNAL_HPP
