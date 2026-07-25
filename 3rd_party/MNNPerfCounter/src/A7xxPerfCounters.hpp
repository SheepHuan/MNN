#ifndef MNN_PERF_COUNTER_A7XX_PERF_COUNTERS_HPP
#define MNN_PERF_COUNTER_A7XX_PERF_COUNTERS_HPP

#include <cstddef>
#include <cstdint>

namespace MNN {
namespace PerfCounter {
namespace detail {

struct A7xxEvent {
    const char* name;
    uint32_t group;
    uint32_t selector;
    uint32_t slots;
};

const A7xxEvent* a7xxEvents(size_t* count);

} // namespace detail
} // namespace PerfCounter
} // namespace MNN

#endif // MNN_PERF_COUNTER_A7XX_PERF_COUNTERS_HPP
