// Optional GPU performance-counter interface used by replay_benchmark.
#ifndef MNN_PERF_COUNTER_HPP
#define MNN_PERF_COUNTER_HPP

#include <cstddef>
#include <cstdint>

namespace MNN {
namespace PerfCounter {

enum class GpuVendor { Unknown, Mali, Adreno };

enum class GpuFamily { Unknown, Midgard, Bifrost, Valhall, Arm5thGen, AdrenoLegacy, AdrenoA7xx };

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

struct CounterValue {
    const char* name = nullptr;
    uint64_t value = 0;
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

// Decode the raw Mali GPU_ID register used by the ARM driver family.
bool identifyMali(uint64_t rawProductId, DeviceInfo* device);

class Session {
public:
    static Session* create(const CounterSpec* specs, size_t count, DeviceInfo* device, const char** error);
    ~Session();

    bool start();
    bool stop(CounterValue* values, size_t count);
    const char* error() const;

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
