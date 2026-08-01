// Optional GPU performance-counter interface used by replay_benchmark.
#ifndef MNN_PERF_COUNTER_HPP
#define MNN_PERF_COUNTER_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace MNN {
namespace PerfCounter {

enum class GpuVendor { Unknown, Mali, Adreno, Nvidia };
enum class GpuFamily { Unknown, Midgard, Bifrost, Valhall, Arm5thGen, AdrenoLegacy, AdrenoA7xx, NvidiaTuring, NvidiaAmpere, NvidiaAda, NvidiaHopper, NvidiaOther };

struct DeviceInfo {
    GpuVendor vendor = GpuVendor::Unknown;
    GpuFamily family = GpuFamily::Unknown;
    uint64_t productId = 0;
    const char* productName = nullptr;
    const char* driverName = nullptr;
};

struct CounterSpec {
    const char* name = nullptr;
};

enum class CounterValueKind {
    UnsignedInteger,
    FloatingPoint,
};

enum class CounterValueStatus {
    Invalid,
    Valid,
    Overflow,
};

struct CounterValue {
    const char* name = nullptr;
    // Adreno and Mali expose native uint64 counters. Keep that channel exact
    // instead of routing it through double, which would lose values above
    // 2^53. NVIDIA's metrics evaluator natively returns double and uses the
    // floatingPointValue channel without an integer cast.
    uint64_t value = 0;
    double floatingPointValue = 0.0;
    CounterValueKind valueKind = CounterValueKind::UnsignedInteger;
    CounterValueStatus status = CounterValueStatus::Invalid;
};

struct CounterBinding {
    const char* name = nullptr;
    uint32_t group = 0;
    uint32_t selector = 0;
    uint32_t slotsAvailable = 0;
};

// Resolve a normalized counter name without opening a GPU device. This is
// useful to validate a profile and is also used by the deterministic tests.
bool resolveCounter(GpuVendor vendor, uint64_t productId, const char* name, CounterBinding* binding);

// Return all normalized counter names supported by the supplied GPU. The
// result is owned by the caller and contains no duplicate names.
std::vector<std::string> supportedCounterNames(const DeviceInfo& device);

// Decode the raw Mali GPU_ID register used by the ARM driver family.
bool identifyMali(uint64_t rawProductId, DeviceInfo* device);

// Identify an NVIDIA GPU from a CUDA device ordinal. Returns true and fills
// device on success. Uses CUPTI Range Profiler API internally.
bool identifyNvidia(int deviceOrdinal, DeviceInfo* device);

class Session {
public:
    static Session* create(const CounterSpec* specs, size_t count, DeviceInfo* device, const char** error);
    ~Session();

    bool start();
    bool stop(CounterValue* values, size_t count);

    // NVIDIA Range Profiler per-kernel range API (no-op on Mali/Adreno). For
    // NVIDIA sessions using CUPTI Range Profiler, start()/stop() demarcate the
    // profiling session. AutoRange sessions let CUPTI identify individual
    // kernels; beginRange()/endRange() are no-ops in that mode.
    // Returns false (no-op) if the backend has no range support.
    bool beginRange(const char* rangeName);
    bool endRange();

    const char* error() const;

    // Returns the number of CUPTI passes required for the configured metrics.
    // 1 = single-pass (no replay). >1 means metrics share hardware counter
    // slots and require N kernel replays. Returns 1 for non-NVIDIA backends.
    size_t numPasses() const;

private:
    Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    struct Impl;
    Impl* mImpl;
};

} // namespace PerfCounter
} // namespace MNN

#endif // MNN_PERF_COUNTER_HPP
