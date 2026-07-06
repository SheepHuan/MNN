#ifndef TRANSFORMERS_PIC_LLM_ENGINE_SRC_LLM_INTERNAL_HPP
#define TRANSFORMERS_PIC_LLM_ENGINE_SRC_LLM_INTERNAL_HPP

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace MNN {
namespace Transformer {

inline bool picRequestProfileEnabled() {
    const char* value = std::getenv("MNN_PIC_REQUEST_PROFILE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

inline int64_t picRequestMonotonicUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline void picRequestProfileLog(const char* stage, int64_t elapsedUs, const std::string& detail = "") {
    if (!picRequestProfileEnabled()) {
        return;
    }
    std::fprintf(stderr, "MNN_PIC_REQUEST_PROFILE stage=%s cost_ms=%.3f", stage, elapsedUs / 1000.0);
    if (!detail.empty()) {
        std::fprintf(stderr, " %s", detail.c_str());
    }
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

inline bool picEnvFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

inline bool picDecodeRepairProfileEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("MNN_PIC_DECODE_REPAIR_PROFILE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

struct ScopedForcePrefillForward {
    bool& flag;
    bool old;

    explicit ScopedForcePrefillForward(bool& value) : flag(value), old(value) {
        flag = true;
    }

    ~ScopedForcePrefillForward() {
        flag = old;
    }
};

} // namespace Transformer
} // namespace MNN

#endif
