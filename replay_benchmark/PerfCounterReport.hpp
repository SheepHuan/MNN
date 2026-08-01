#ifndef MNN_REPLAY_PERF_COUNTER_REPORT_HPP
#define MNN_REPLAY_PERF_COUNTER_REPORT_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace MNN {
namespace Replay {

enum class PerfCounterValueKind {
    UnsignedInteger,
    FloatingPoint,
};

enum class PerfCounterValueStatus {
    Invalid,
    Valid,
    Overflow,
};

struct PerfCounterValueRecord {
    std::string name;
    uint64_t integerValue = 0;
    double floatingPointValue = 0.0;
    PerfCounterValueKind valueKind = PerfCounterValueKind::UnsignedInteger;
    PerfCounterValueStatus status = PerfCounterValueStatus::Invalid;
};

struct PerfCounterReport {
    std::string status;
    std::string model;
    int opId = -1;
    std::string opName;
    std::string opType;
    std::string execution;
    std::string variant;
    std::string backend;
    std::string vendor;
    uint64_t productId = 0;
    std::string productName;
    std::string driver;
    bool synchronized = false;
    uint64_t startNs = 0;
    uint64_t endNs = 0;
    std::vector<PerfCounterValueRecord> counters;
    std::string error;

    bool write(const std::string& path) const;
};

} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_PERF_COUNTER_REPORT_HPP
