//
//  OpenCLPmcProfiler.cpp
//  MNN
//

#include "backend/opencl/core/OpenCLPmcProfiler.hpp"
#include "core/Macro.h"

#ifdef MNN_OPENCL_PMC_PROFILE
#include <device/product_id.hpp>
#include <hwcpipe/counter_database.hpp>
#include <hwcpipe/gpu.hpp>
#include <hwcpipe/sampler.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <regex.h>
#include <sstream>
#include <string>
#include <system_error>
#include <sys/stat.h>
#include <sys/types.h>
#include <utility>
#include <vector>
#endif

namespace MNN {
namespace OpenCL {

#ifdef MNN_OPENCL_PMC_PROFILE
namespace {

static bool _envFlagEnabled(const char* name, bool defaultValue) {
    const char* value = ::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    return value[0] != '0';
}

static uint64_t _nowUs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

static std::string _getEnvString(const char* name, const char* defaultValue = "") {
    const char* value = ::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue != nullptr ? defaultValue : "";
    }
    return value;
}

static int _getEnvInt(const char* name, int defaultValue) {
    const char* value = ::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    return ::atoi(value);
}

static std::string _lower(const std::string& value) {
    std::string out = value;
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(out[i])));
    }
    return out;
}

static std::string _compactAlnum(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        if (std::isalnum(c)) {
            out.push_back(static_cast<char>(std::tolower(c)));
        }
    }
    return out;
}

static bool _contains(const std::string& haystackLower, const char* needleLower) {
    return haystackLower.find(needleLower) != std::string::npos;
}

static std::vector<std::string> _split(const std::string& value, char delimiter) {
    std::vector<std::string> out;
    std::string current;
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == delimiter) {
            if (!current.empty()) {
                out.push_back(current);
            }
            current.clear();
            continue;
        }
        current.push_back(value[i]);
    }
    if (!current.empty()) {
        out.push_back(current);
    }
    return out;
}

static void _jsonString(std::ostream& os, const std::string& value) {
    os << '"';
    for (size_t i = 0; i < value.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        switch (c) {
            case '\\':
                os << "\\\\";
                break;
            case '"':
                os << "\\\"";
                break;
            case '\b':
                os << "\\b";
                break;
            case '\f':
                os << "\\f";
                break;
            case '\n':
                os << "\\n";
                break;
            case '\r':
                os << "\\r";
                break;
            case '\t':
                os << "\\t";
                break;
            default:
                if (c < 0x20) {
                    const char* hex = "0123456789abcdef";
                    os << "\\u00" << hex[(c >> 4) & 0xf] << hex[c & 0xf];
                } else {
                    os << static_cast<char>(c);
                }
                break;
        }
    }
    os << '"';
}

static void _jsonVector(std::ostream& os, const std::vector<uint32_t>& values) {
    os << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            os << ',';
        }
        os << values[i];
    }
    os << ']';
}

static bool _ensureParentDirectory(const std::string& path) {
    const size_t slash = path.find_last_of('/');
    if (slash == std::string::npos || slash == 0) {
        return true;
    }
    std::string dir = path.substr(0, slash);
    std::string current;
    if (!dir.empty() && dir[0] == '/') {
        current = "/";
    }
    std::vector<std::string> parts = _split(dir, '/');
    for (size_t i = 0; i < parts.size(); ++i) {
        if (parts[i].empty()) {
            continue;
        }
        if (!current.empty() && current[current.size() - 1] != '/') {
            current.push_back('/');
        }
        current += parts[i];
        if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
            return false;
        }
    }
    return true;
}

class PosixRegex {
public:
    PosixRegex() = default;
    ~PosixRegex() {
        if (mCompiled) {
            ::regfree(&mRegex);
        }
    }

    bool compile(const std::string& pattern) {
        if (mCompiled) {
            ::regfree(&mRegex);
            mCompiled = false;
        }
        mPattern = pattern;
        if (pattern.empty()) {
            mEnabled = false;
            mValid = true;
            return true;
        }
        mEnabled = true;
        const int rc = ::regcomp(&mRegex, pattern.c_str(), REG_EXTENDED | REG_NOSUB);
        if (rc != 0) {
            char buffer[256] = {};
            ::regerror(rc, &mRegex, buffer, sizeof(buffer));
            mError = buffer;
            mValid = false;
            return false;
        }
        mCompiled = true;
        mValid = true;
        return true;
    }

    bool enabled() const {
        return mEnabled;
    }

    bool valid() const {
        return mValid;
    }

    const std::string& error() const {
        return mError;
    }

    bool matches(const std::string& value) const {
        if (!mEnabled) {
            return true;
        }
        if (!mValid || !mCompiled) {
            return false;
        }
        return ::regexec(&mRegex, value.c_str(), 0, nullptr, 0) == 0;
    }

private:
    regex_t mRegex;
    bool mEnabled = false;
    bool mValid = true;
    bool mCompiled = false;
    std::string mPattern;
    std::string mError;
};

class LayerFilter {
public:
    void parse(const std::string& spec) {
        mAll = true;
        mRanges.clear();
        const std::string lower = _lower(spec);
        if (lower.empty() || lower == "all") {
            return;
        }
        mAll = false;
        std::vector<std::string> parts = _split(lower, ',');
        for (size_t i = 0; i < parts.size(); ++i) {
            const std::string& part = parts[i];
            const size_t dash = part.find('-');
            int begin = 0;
            int end = 0;
            if (dash == std::string::npos) {
                begin = ::atoi(part.c_str());
                end = begin;
            } else {
                begin = ::atoi(part.substr(0, dash).c_str());
                end = ::atoi(part.substr(dash + 1).c_str());
                if (end < begin) {
                    std::swap(begin, end);
                }
            }
            mRanges.push_back(std::make_pair(begin, end));
        }
    }

    bool matches(int layer) const {
        if (mAll) {
            return true;
        }
        for (size_t i = 0; i < mRanges.size(); ++i) {
            if (layer >= mRanges[i].first && layer <= mRanges[i].second) {
                return true;
            }
        }
        return false;
    }

private:
    bool mAll = true;
    std::vector<std::pair<int, int>> mRanges;
};

struct CounterInfo {
    hwcpipe_counter counter;
    std::string name;
    std::string units;
};

struct CounterValue {
    bool isFloat = false;
    uint64_t uintValue = 0;
    double floatValue = 0.0;
};

struct ActiveScope {
    uint64_t wallStartUs = 0;
    std::vector<CounterValue> beginValues;
};

static CounterValue _sampleValue(const hwcpipe::counter_sample& sample) {
    CounterValue value;
    if (sample.type == hwcpipe::counter_sample::type::float64) {
        value.isFloat = true;
        value.floatValue = sample.value.float64;
    } else {
        value.uintValue = sample.value.uint64;
    }
    return value;
}

static double _counterValueAsDouble(const CounterValue& value) {
    return value.isFloat ? value.floatValue : static_cast<double>(value.uintValue);
}

static void _jsonCounterValue(std::ostream& os, const CounterValue& value) {
    if (value.isFloat) {
        os << value.floatValue;
    } else {
        os << static_cast<unsigned long long>(value.uintValue);
    }
}

static void _jsonCounterDelta(std::ostream& os, const CounterValue& begin, const CounterValue& end) {
    if (begin.isFloat || end.isFloat) {
        os << (_counterValueAsDouble(end) - _counterValueAsDouble(begin));
        return;
    }
    const uint64_t delta = end.uintValue >= begin.uintValue ? end.uintValue - begin.uintValue : 0;
    os << static_cast<unsigned long long>(delta);
}

static bool _counterNameMatchesDefault(const std::string& nameLower) {
    if (nameLower == "gpu active cycles" ||
        nameLower == "gpu active raw cycles" ||
        nameLower == "any workload active cycles" ||
        nameLower == "compute queue active cycles" ||
        nameLower == "compute or binning phase active cycles" ||
        nameLower == "execution core active cycles" ||
        nameLower == "execution engine starvation cycles" ||
        nameLower == "arithmetic unit issue cycles" ||
        nameLower == "load/store unit read bytes from external memory" ||
        nameLower == "load/store unit write bytes to l2 memory system" ||
        nameLower == "load/store unit read bytes from l2 cache" ||
        nameLower == "texture unit read bytes from external memory" ||
        nameLower == "texture unit read bytes from l2 cache" ||
        nameLower == "output external read bytes" ||
        nameLower == "output external write bytes" ||
        nameLower == "output external read stall cycles" ||
        nameLower == "output external write stall cycles" ||
        nameLower == "l2 cache flush cycles") {
        return true;
    }
    return false;
}

static bool _counterNameMatchesAlias(const std::string& requestCompact, const std::string& nameLower) {
    if (requestCompact == "maligpuactivecy") {
        return nameLower == "gpu active cycles";
    }
    if (requestCompact == "malianyactivecy") {
        return nameLower == "any workload active cycles";
    }
    if (requestCompact == "malifragactivecy") {
        return nameLower == "fragment active cycles";
    }
    if (requestCompact == "malitileractivecy") {
        return nameLower == "tiler active cycles";
    }
    if (requestCompact == "maliextbusrdby") {
        return nameLower == "output external read bytes";
    }
    if (requestCompact == "maliextbuswrby") {
        return nameLower == "output external write bytes";
    }
    if (requestCompact == "maliscbuslsextrdby") {
        return nameLower == "load/store unit read bytes from external memory";
    }
    if (requestCompact == "maliscbuslswrby") {
        return nameLower == "load/store unit write bytes to l2 memory system";
    }
    if (requestCompact == "maliscbustexextrdby") {
        return nameLower == "texture unit read bytes from external memory";
    }
    return false;
}

class ProfilerState {
public:
    ~ProfilerState() {
        if (mSampler && mSamplingStarted) {
            std::error_code ec = mSampler->stop_sampling();
            (void)ec;
        }
    }

    bool enabledFor(const OpenCLPmcScopeMeta& meta) {
        std::lock_guard<std::mutex> lock(mMutex);
        readConfigLocked();
        if (!mEnabled) {
            return false;
        }
        if (!mLayerFilter.matches(meta.layer)) {
            return false;
        }
        const std::string op = meta.op != nullptr ? meta.op : "";
        const std::string phase = meta.phase != nullptr ? meta.phase : "";
        if (!mKernelRegex.matches(op) || !mPhaseRegex.matches(phase)) {
            return false;
        }
        return true;
    }

    uint64_t begin(OpenCLRuntime* runtime, cl::CommandQueue& queue, const OpenCLPmcScopeMeta& meta) {
        std::lock_guard<std::mutex> lock(mMutex);
        readConfigLocked();
        if (!mEnabled) {
            return 0;
        }
        if (!mLayerFilter.matches(meta.layer)) {
            return 0;
        }
        const std::string op = meta.op != nullptr ? meta.op : "";
        const std::string phase = meta.phase != nullptr ? meta.phase : "";
        if (!mKernelRegex.matches(op) || !mPhaseRegex.matches(phase)) {
            return 0;
        }
        ++mAcceptedScopes;
        if (mAcceptedScopes <= static_cast<uint64_t>(std::max(0, mWarmupSkip))) {
            return 0;
        }
        if (mMaxRecords > 0 && mWrittenRecords >= static_cast<uint64_t>(mMaxRecords)) {
            return 0;
        }
        if (mActiveToken != 0) {
            writeStatusLocked("busy_skip", "overlapping PMC scope skipped");
            return 0;
        }
        if (!ensureSamplerLocked()) {
            return 0;
        }
        if (mStrictFinish) {
            queue.finish();
        }
        std::vector<CounterValue> values;
        std::string error;
        if (!sampleCountersLocked(values, &error)) {
            writeStatusLocked("sample_failed", error);
            return 0;
        }
        const uint64_t token = ++mNextToken;
        ActiveScope scope;
        scope.wallStartUs = _nowUs();
        scope.beginValues.swap(values);
        mActiveScopes[token] = scope;
        mActiveToken = token;
        (void)runtime;
        return token;
    }

    void end(uint64_t token, OpenCLRuntime* runtime, cl::CommandQueue& queue,
             const OpenCLPmcScopeMeta& meta, cl::Event* event) {
        if (token == 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(mMutex);
        auto iter = mActiveScopes.find(token);
        if (iter == mActiveScopes.end()) {
            return;
        }
        if (mStrictFinish) {
            queue.finish();
        }
        std::vector<CounterValue> endValues;
        std::string error;
        if (!sampleCountersLocked(endValues, &error)) {
            writeStatusLocked("sample_failed", error);
            mActiveScopes.erase(iter);
            if (mActiveToken == token) {
                mActiveToken = 0;
            }
            return;
        }
        const uint64_t wallEndUs = _nowUs();
        const ActiveScope scope = iter->second;
        mActiveScopes.erase(iter);
        if (mActiveToken == token) {
            mActiveToken = 0;
        }
        double eventUs = 0.0;
        std::string eventStatus = "unavailable";
        if (event != nullptr) {
            cl_ulong start = 0;
            cl_ulong end = 0;
            const cl_int startRet = event->getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
            const cl_int endRet = event->getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
            if (startRet == CL_SUCCESS && endRet == CL_SUCCESS && end >= start) {
                eventUs = static_cast<double>(end - start) / 1000.0;
                eventStatus = "ok";
            }
        }
        writeRecordLocked(token, meta, scope, endValues, wallEndUs, eventUs, eventStatus);
        ++mWrittenRecords;
        (void)runtime;
    }

private:
    void readConfigLocked() {
        if (mConfigRead) {
            return;
        }
        mConfigRead = true;
        mEnabled = _envFlagEnabled("MNN_PIC_PMC_PROFILE", false);
        mStrictFinish = _envFlagEnabled("MNN_PIC_PMC_STRICT_FINISH", true);
        mOutputPath = _getEnvString("MNN_PIC_PMC_OUTPUT",
                                    "/mnt/ssd/code/.cache/mnn_opencl_pic/pmc/mnn_opencl_pmc.jsonl");
        mCounterRequest = _getEnvString("MNN_PIC_PMC_COUNTERS", "default");
        mMaxRecords = _getEnvInt("MNN_PIC_PMC_MAX_RECORDS", 2000);
        mWarmupSkip = _getEnvInt("MNN_PIC_PMC_WARMUP_SKIP", 0);
        mLayerFilter.parse(_getEnvString("MNN_PIC_PMC_LAYER", "all"));
        if (!mKernelRegex.compile(_getEnvString("MNN_PIC_PMC_KERNEL_REGEX", ""))) {
            writeStatusLocked("bad_kernel_regex", mKernelRegex.error());
        }
        if (!mPhaseRegex.compile(_getEnvString("MNN_PIC_PMC_PHASE_REGEX", ""))) {
            writeStatusLocked("bad_phase_regex", mPhaseRegex.error());
        }
    }

    bool ensureSamplerLocked() {
        if (mSamplerReady) {
            return true;
        }
        if (mSamplerAttempted) {
            return false;
        }
        mSamplerAttempted = true;
        if (!mEnabled) {
            return false;
        }
        hwcpipe::gpu gpu(0);
        if (!gpu) {
            writeStatusLocked("unsupported", "Mali GPU device 0 is missing or unsupported by libGPUCounters");
            return false;
        }
        hwcpipe::counter_database db;
        std::vector<CounterInfo> available;
        hwcpipe::counter_metadata meta;
        for (hwcpipe_counter counter : db.counters_for_gpu(gpu)) {
            const std::error_code ec = db.describe_counter(counter, meta);
            if (ec) {
                continue;
            }
            CounterInfo info;
            info.counter = counter;
            info.name = meta.name != nullptr ? meta.name : "";
            info.units = meta.units != nullptr ? meta.units : "";
            available.push_back(info);
        }
        std::vector<CounterInfo> selected = selectCountersLocked(available);
        if (selected.empty()) {
            writeStatusLocked("unsupported", "no requested GPU counters are available");
            return false;
        }
        hwcpipe::sampler_config config(gpu);
        for (size_t i = 0; i < selected.size(); ++i) {
            const std::error_code ec = config.add_counter(selected[i].counter);
            if (!ec) {
                mCounters.push_back(selected[i]);
            }
        }
        if (mCounters.empty()) {
            writeStatusLocked("unsupported", "libGPUCounters rejected all selected counters");
            return false;
        }
        mSampler.reset(new hwcpipe::sampler<>(config));
        std::error_code ec = mSampler->start_sampling();
        if (ec) {
            writeStatusLocked("unsupported", ec.message());
            mSampler.reset();
            return false;
        }
        mSamplingStarted = true;
        mSamplerReady = true;
        std::ostringstream reason;
        reason << "selected_counter_count=" << mCounters.size();
        writeStatusLocked("ready", reason.str());
        return true;
    }

    std::vector<CounterInfo> selectCountersLocked(const std::vector<CounterInfo>& available) const {
        const std::string requestLower = _lower(mCounterRequest);
        std::vector<CounterInfo> selected;
        if (requestLower.empty() || requestLower == "default") {
            for (size_t i = 0; i < available.size(); ++i) {
                if (_counterNameMatchesDefault(_lower(available[i].name))) {
                    selected.push_back(available[i]);
                    if (selected.size() >= 18) {
                        break;
                    }
                }
            }
            if (!selected.empty()) {
                return selected;
            }
            for (size_t i = 0; i < available.size() && selected.size() < 12; ++i) {
                const std::string nameLower = _lower(available[i].name);
                if (_contains(nameLower, "active cycles") ||
                    (_contains(nameLower, "external") && _contains(nameLower, "bytes")) ||
                    (_contains(nameLower, "l2") && _contains(nameLower, "bytes"))) {
                    selected.push_back(available[i]);
                }
            }
            return selected;
        }
        if (requestLower == "all") {
            return available;
        }

        PosixRegex counterRegex;
        const bool regexOk = counterRegex.compile(mCounterRequest);
        const std::vector<std::string> terms = _split(mCounterRequest, ',');
        for (size_t i = 0; i < available.size(); ++i) {
            const std::string nameLower = _lower(available[i].name);
            bool match = regexOk && counterRegex.matches(available[i].name);
            for (size_t t = 0; t < terms.size() && !match; ++t) {
                const std::string termLower = _lower(terms[t]);
                const std::string termCompact = _compactAlnum(terms[t]);
                if (!termLower.empty() && nameLower.find(termLower) != std::string::npos) {
                    match = true;
                } else if (_counterNameMatchesAlias(termCompact, nameLower)) {
                    match = true;
                }
            }
            if (match) {
                selected.push_back(available[i]);
            }
        }
        return selected;
    }

    bool sampleCountersLocked(std::vector<CounterValue>& values, std::string* error) {
        values.clear();
        if (!mSamplerReady || !mSampler) {
            if (error != nullptr) {
                *error = "sampler not initialized";
            }
            return false;
        }
        std::error_code ec = mSampler->sample_now();
        if (ec) {
            if (error != nullptr) {
                *error = ec.message();
            }
            return false;
        }
        values.reserve(mCounters.size());
        for (size_t i = 0; i < mCounters.size(); ++i) {
            hwcpipe::counter_sample sample;
            ec = mSampler->get_counter_value(mCounters[i].counter, sample);
            if (ec) {
                if (error != nullptr) {
                    *error = ec.message();
                }
                return false;
            }
            values.push_back(_sampleValue(sample));
        }
        return values.size() == mCounters.size();
    }

    void writeStatusLocked(const std::string& status, const std::string& reason) {
        if (status != "ready" && status != "bad_kernel_regex" && status != "bad_phase_regex") {
            if (mStatusWritten[status]) {
                return;
            }
            mStatusWritten[status] = true;
        }
        std::ostringstream os;
        os << "{\"pmc_status\":";
        _jsonString(os, status);
        os << ",\"reason\":";
        _jsonString(os, reason);
        os << "}\n";
        appendLineLocked(os.str());
    }

    void writeRecordLocked(uint64_t token, const OpenCLPmcScopeMeta& meta, const ActiveScope& scope,
                           const std::vector<CounterValue>& endValues, uint64_t wallEndUs,
                           double eventUs, const std::string& eventStatus) {
        std::ostringstream os;
        os.setf(std::ios::fixed);
        os.precision(3);
        os << "{\"pmc_status\":\"ok\",\"token\":" << token;
        os << ",\"op\":";
        _jsonString(os, meta.op != nullptr ? meta.op : "");
        os << ",\"phase\":";
        _jsonString(os, meta.phase != nullptr ? meta.phase : "");
        os << ",\"layer\":" << meta.layer;
        os << ",\"query\":" << meta.query;
        os << ",\"input_query\":" << meta.inputQuery;
        os << ",\"kv_len\":" << meta.kvLen;
        os << ",\"base_logical\":" << meta.baseLogical;
        os << ",\"lane\":" << meta.lane;
        os << ",\"q_tile\":" << meta.qTile;
        os << ",\"heads\":" << meta.heads;
        os << ",\"kv_heads\":" << meta.kvHeads;
        os << ",\"head_dim\":" << meta.headDim;
        os << ",\"gws\":";
        _jsonVector(os, meta.gws);
        os << ",\"lws\":";
        _jsonVector(os, meta.lws);
        os << ",\"wall_us\":" << (wallEndUs >= scope.wallStartUs ? wallEndUs - scope.wallStartUs : 0);
        os << ",\"event_us\":" << eventUs;
        os << ",\"event_status\":";
        _jsonString(os, eventStatus);
        os << ",\"dense_kv_work\":" << static_cast<unsigned long long>(meta.denseKvWork);
        os << ",\"causal_kv_work\":" << static_cast<unsigned long long>(meta.causalKvWork);
        os << ",\"append_count\":" << meta.appendCount;
        os << ",\"prepare_len\":" << meta.prepareLen;
        os << ",\"decode_prepare_inside_decode\":" << (meta.decodePrepareInsideDecode ? 1 : 0);
        os << ",\"counters\":[";
        const size_t n = std::min(scope.beginValues.size(), std::min(endValues.size(), mCounters.size()));
        for (size_t i = 0; i < n; ++i) {
            if (i > 0) {
                os << ',';
            }
            os << "{\"name\":";
            _jsonString(os, mCounters[i].name);
            os << ",\"units\":";
            _jsonString(os, mCounters[i].units);
            os << ",\"begin\":";
            _jsonCounterValue(os, scope.beginValues[i]);
            os << ",\"end\":";
            _jsonCounterValue(os, endValues[i]);
            os << ",\"delta\":";
            _jsonCounterDelta(os, scope.beginValues[i], endValues[i]);
            os << '}';
        }
        os << "]}\n";
        appendLineLocked(os.str());
    }

    void appendLineLocked(const std::string& line) {
        if (mOutputPath.empty()) {
            MNN_PRINT("%s", line.c_str());
            return;
        }
        if (!_ensureParentDirectory(mOutputPath)) {
            MNN_PRINT("OpenCL PMC profile failed to create output parent for %s\n", mOutputPath.c_str());
            MNN_PRINT("%s", line.c_str());
            return;
        }
        std::ofstream output(mOutputPath.c_str(), std::ios::out | std::ios::app);
        if (!output.good()) {
            MNN_PRINT("OpenCL PMC profile failed to open output %s\n", mOutputPath.c_str());
            MNN_PRINT("%s", line.c_str());
            return;
        }
        output << line;
    }

private:
    std::mutex mMutex;
    bool mConfigRead = false;
    bool mEnabled = false;
    bool mStrictFinish = true;
    int mMaxRecords = 2000;
    int mWarmupSkip = 0;
    std::string mOutputPath;
    std::string mCounterRequest;
    PosixRegex mKernelRegex;
    PosixRegex mPhaseRegex;
    LayerFilter mLayerFilter;

    bool mSamplerAttempted = false;
    bool mSamplerReady = false;
    bool mSamplingStarted = false;
    std::unique_ptr<hwcpipe::sampler<> > mSampler;
    std::vector<CounterInfo> mCounters;

    uint64_t mNextToken = 0;
    uint64_t mActiveToken = 0;
    uint64_t mAcceptedScopes = 0;
    uint64_t mWrittenRecords = 0;
    std::map<uint64_t, ActiveScope> mActiveScopes;
    std::map<std::string, bool> mStatusWritten;
};

static ProfilerState& _state() {
    static ProfilerState state;
    return state;
}

} // namespace
#endif

OpenCLPmcProfiler& OpenCLPmcProfiler::get() {
    static OpenCLPmcProfiler profiler;
    return profiler;
}

bool OpenCLPmcProfiler::enabledFor(const OpenCLPmcScopeMeta& meta) const {
#ifdef MNN_OPENCL_PMC_PROFILE
    return _state().enabledFor(meta);
#else
    (void)meta;
    return false;
#endif
}

uint64_t OpenCLPmcProfiler::begin(OpenCLRuntime* runtime, cl::CommandQueue& queue,
                                  const OpenCLPmcScopeMeta& meta) {
#ifdef MNN_OPENCL_PMC_PROFILE
    return _state().begin(runtime, queue, meta);
#else
    (void)runtime;
    (void)queue;
    (void)meta;
    return 0;
#endif
}

void OpenCLPmcProfiler::end(uint64_t token, OpenCLRuntime* runtime, cl::CommandQueue& queue,
                            const OpenCLPmcScopeMeta& meta, cl::Event* event) {
#ifdef MNN_OPENCL_PMC_PROFILE
    _state().end(token, runtime, queue, meta, event);
#else
    (void)token;
    (void)runtime;
    (void)queue;
    (void)meta;
    (void)event;
#endif
}

} // namespace OpenCL
} // namespace MNN
