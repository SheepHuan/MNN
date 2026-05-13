#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "backend/opencl/execution/buffer/AttentionBufExecution.hpp"
#include "core/MNNFileUtils.h"
#include "core/PagedKVCachePlan.hpp"
#include "core/PrefixCachePath.hpp"
#include "core/PrefixKVHostCache.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace MNN {
namespace OpenCL {

namespace {

constexpr const char* kOpenCLPrefixLayout = "opencl-buffer-canonical-no-rope-v1";

static size_t openCLNativeFileBytes(int tokenCount, int kvHeads, int headDim, int bytes) {
    int alignedTokens = ROUND_UP(tokenCount, 4);
    return static_cast<size_t>(alignedTokens) * kvHeads * headDim * bytes;
}

static double nowMs() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

static cl_int enqueueWaitForEvent(cl::CommandQueue& queue, const cl::Event& event) {
    if (event() == nullptr) {
        return CL_SUCCESS;
    }
    cl_event rawEvent = event();
    return clEnqueueWaitForEvents(queue(), 1, &rawEvent);
}

static cl_int openCLPrefixEnqueueMarker(cl::CommandQueue& queue, cl::Event& event) {
    cl_event rawEvent = nullptr;
    cl_int status = clEnqueueMarkerWithWaitList(queue(), 0, nullptr, &rawEvent);
    if (status == CL_SUCCESS && rawEvent != nullptr) {
        event = cl::Event(rawEvent);
    }
    return status;
}

static int alignToPage(int value, int pageSize) {
    return UP_DIV(value, pageSize) * pageSize;
}

static bool openCLPrefixEnvFlag(const char* name, bool defaultValue) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    std::string text(value);
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text == "1" || text == "true" || text == "on" || text == "yes";
}

static bool openCLPrefixProfileOverlapEnabled() {
    static bool enabled = openCLPrefixEnvFlag("MNN_OPENCL_PREFIX_PROFILE_OVERLAP", false);
    return enabled;
}

static bool openCLPrefixProfileVerboseEnabled() {
    static bool enabled = openCLPrefixEnvFlag("MNN_OPENCL_PREFIX_PROFILE_VERBOSE", false);
    return enabled;
}

static bool openCLPrefixUseHostPtrPagesEnabled() {
    static bool enabled = openCLPrefixEnvFlag("MNN_OPENCL_PREFIX_USE_HOST_PTR_PAGES", false);
    return enabled;
}

static int openCLPrefixProfilePairLimit() {
    static int limit = []() {
        const char* value = std::getenv("MNN_OPENCL_PREFIX_PROFILE_PAIR_LIMIT");
        if (value == nullptr || value[0] == '\0') {
            return 20;
        }
        return std::max(0, std::atoi(value));
    }();
    return limit;
}

struct OpenCLPrefixProfileInterval {
    double startMs = 0.0;
    double endMs = 0.0;
    int layer = -1;
    std::string name;
    std::string cache;
    uint64_t queueId = 0;
    size_t bytes = 0;
};

struct OpenCLPrefixCopyProfileRecord {
    int layer = -1;
    std::string cache;
    uint64_t queueId = 0;
    size_t bytes = 0;
    cl::Event firstEvent;
    cl::Event lastEvent;
};

struct OpenCLPrefixComputeProfileRecord {
    int layer = -1;
    uint64_t queueId = 0;
    cl::Event startEvent;
    cl::Event endEvent;
};

struct OpenCLPrefixKernelProfileRecord {
    int layer = -1;
    std::string name;
    uint64_t queueId = 0;
    cl::Event event;
};

struct OpenCLPrefixPageFillProfileRecord {
    int layer = -1;
    std::string cache;
    uint64_t queueId = 0;
    size_t bytes = 0;
    double diskReadMs = 0.0;
    double writeMs = 0.0;
    cl::Event startEvent;
    cl::Event endEvent;
};

struct OpenCLPrefixRequestProfile {
    int expectedLayers = 0;
    bool reported = false;
    std::vector<OpenCLPrefixCopyProfileRecord> copies;
    std::vector<OpenCLPrefixPageFillProfileRecord> pageFills;
    std::vector<OpenCLPrefixComputeProfileRecord> computes;
    std::vector<OpenCLPrefixKernelProfileRecord> kernels;
};

static std::mutex& openCLPrefixProfileMutex() {
    static std::mutex mutex;
    return mutex;
}

static std::unordered_map<uint64_t, OpenCLPrefixRequestProfile>& openCLPrefixProfiles() {
    static std::unordered_map<uint64_t, OpenCLPrefixRequestProfile> profiles;
    return profiles;
}

static uint64_t openCLPrefixQueueId(const cl::CommandQueue& queue) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(queue()));
}

static bool openCLPrefixEventInterval(const cl::Event& startEvent, const cl::Event& endEvent,
                                      double& startMs, double& endMs) {
    if (startEvent() == nullptr || endEvent() == nullptr) {
        return false;
    }
    cl_int waitErr = endEvent.wait();
    if (waitErr != CL_SUCCESS) {
        return false;
    }
    cl_ulong startNs = 0;
    cl_ulong endNs = 0;
    cl_int startErr = startEvent.getProfilingInfo(CL_PROFILING_COMMAND_START, &startNs);
    cl_int endErr = endEvent.getProfilingInfo(CL_PROFILING_COMMAND_END, &endNs);
    if (startErr != CL_SUCCESS || endErr != CL_SUCCESS || endNs < startNs) {
        return false;
    }
    startMs = static_cast<double>(startNs) / 1000000.0;
    endMs = static_cast<double>(endNs) / 1000000.0;
    return true;
}

static bool openCLPrefixEventInterval(const cl::Event& event, double& startMs, double& endMs) {
    return openCLPrefixEventInterval(event, event, startMs, endMs);
}

static double openCLPrefixUnionMs(std::vector<OpenCLPrefixProfileInterval> intervals) {
    if (intervals.empty()) {
        return 0.0;
    }
    std::sort(intervals.begin(), intervals.end(),
              [](const OpenCLPrefixProfileInterval& left, const OpenCLPrefixProfileInterval& right) {
        return left.startMs < right.startMs;
    });
    double total = 0.0;
    double start = intervals[0].startMs;
    double end = intervals[0].endMs;
    for (size_t i = 1; i < intervals.size(); ++i) {
        if (intervals[i].startMs <= end) {
            end = std::max(end, intervals[i].endMs);
        } else {
            total += std::max(0.0, end - start);
            start = intervals[i].startMs;
            end = intervals[i].endMs;
        }
    }
    total += std::max(0.0, end - start);
    return total;
}

static double openCLPrefixOverlapUnionMs(const std::vector<OpenCLPrefixProfileInterval>& left,
                                         const std::vector<OpenCLPrefixProfileInterval>& right) {
    std::vector<OpenCLPrefixProfileInterval> overlaps;
    for (const auto& a : left) {
        for (const auto& b : right) {
            double start = std::max(a.startMs, b.startMs);
            double end = std::min(a.endMs, b.endMs);
            if (end > start) {
                OpenCLPrefixProfileInterval interval;
                interval.startMs = start;
                interval.endMs = end;
                overlaps.emplace_back(interval);
            }
        }
    }
    return openCLPrefixUnionMs(std::move(overlaps));
}

static double openCLPrefixTotalMs(const std::vector<OpenCLPrefixProfileInterval>& intervals) {
    double total = 0.0;
    for (const auto& interval : intervals) {
        total += std::max(0.0, interval.endMs - interval.startMs);
    }
    return total;
}

static void openCLPrefixProfileExpect(uint64_t requestId, int expectedLayers) {
    if (!openCLPrefixProfileOverlapEnabled() || requestId == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(openCLPrefixProfileMutex());
    auto& profile = openCLPrefixProfiles()[requestId];
    profile.expectedLayers = std::max(profile.expectedLayers, expectedLayers);
}

static void openCLPrefixProfileRecordCopy(uint64_t requestId, int layer, const std::string& cache,
                                          uint64_t queueId, size_t bytes,
                                          const cl::Event& firstEvent, const cl::Event& lastEvent) {
    if (!openCLPrefixProfileOverlapEnabled() || requestId == 0 ||
        firstEvent() == nullptr || lastEvent() == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(openCLPrefixProfileMutex());
    auto& profile = openCLPrefixProfiles()[requestId];
    OpenCLPrefixCopyProfileRecord record;
    record.layer = layer;
    record.cache = cache;
    record.queueId = queueId;
    record.bytes = bytes;
    record.firstEvent = firstEvent;
    record.lastEvent = lastEvent;
    profile.copies.emplace_back(record);
}

static void openCLPrefixProfileRecordPageFill(uint64_t requestId, int layer, const std::string& cache,
                                              uint64_t queueId, size_t bytes,
                                              double diskReadMs, double writeMs,
                                              const cl::Event& startEvent, const cl::Event& endEvent) {
    if (!openCLPrefixProfileOverlapEnabled() || requestId == 0 ||
        startEvent() == nullptr || endEvent() == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(openCLPrefixProfileMutex());
    auto& profile = openCLPrefixProfiles()[requestId];
    OpenCLPrefixPageFillProfileRecord record;
    record.layer = layer;
    record.cache = cache;
    record.queueId = queueId;
    record.bytes = bytes;
    record.diskReadMs = diskReadMs;
    record.writeMs = writeMs;
    record.startEvent = startEvent;
    record.endEvent = endEvent;
    profile.pageFills.emplace_back(record);
}

static void openCLPrefixProfileRecordCompute(uint64_t requestId, int layer, uint64_t queueId,
                                             const cl::Event& startEvent, const cl::Event& endEvent) {
    if (!openCLPrefixProfileOverlapEnabled() || requestId == 0 ||
        startEvent() == nullptr || endEvent() == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(openCLPrefixProfileMutex());
    auto& profile = openCLPrefixProfiles()[requestId];
    OpenCLPrefixComputeProfileRecord record;
    record.layer = layer;
    record.queueId = queueId;
    record.startEvent = startEvent;
    record.endEvent = endEvent;
    profile.computes.emplace_back(record);
}

static void openCLPrefixProfileRecordKernel(uint64_t requestId, int layer, const std::string& name,
                                            uint64_t queueId, const cl::Event& event) {
    if (!openCLPrefixProfileOverlapEnabled() || !openCLPrefixProfileVerboseEnabled() ||
        requestId == 0 || event() == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(openCLPrefixProfileMutex());
    auto& profile = openCLPrefixProfiles()[requestId];
    OpenCLPrefixKernelProfileRecord record;
    record.layer = layer;
    record.name = name;
    record.queueId = queueId;
    record.event = event;
    profile.kernels.emplace_back(record);
}

static void openCLPrefixProfileMaybeReport(uint64_t requestId) {
    if (!openCLPrefixProfileOverlapEnabled() || requestId == 0) {
        return;
    }

    OpenCLPrefixRequestProfile snapshot;
    {
        std::lock_guard<std::mutex> lock(openCLPrefixProfileMutex());
        auto iter = openCLPrefixProfiles().find(requestId);
        if (iter == openCLPrefixProfiles().end() || iter->second.reported) {
            return;
        }
        int expectedLayers = iter->second.expectedLayers;
        if (expectedLayers > 0 && static_cast<int>(iter->second.computes.size()) < expectedLayers) {
            return;
        }
        iter->second.reported = true;
        snapshot = iter->second;
    }

    std::vector<OpenCLPrefixProfileInterval> copyIntervals;
    std::vector<OpenCLPrefixProfileInterval> pageFillIntervals;
    std::vector<OpenCLPrefixProfileInterval> computeIntervals;
    std::vector<OpenCLPrefixProfileInterval> kernelIntervals;
    bool profilingUnavailable = false;

    for (const auto& record : snapshot.copies) {
        OpenCLPrefixProfileInterval interval;
        interval.layer = record.layer;
        interval.cache = record.cache;
        interval.queueId = record.queueId;
        interval.bytes = record.bytes;
        if (!openCLPrefixEventInterval(record.firstEvent, record.lastEvent,
                                       interval.startMs, interval.endMs)) {
            profilingUnavailable = true;
            continue;
        }
        copyIntervals.emplace_back(interval);
    }
    for (const auto& record : snapshot.pageFills) {
        OpenCLPrefixProfileInterval interval;
        interval.layer = record.layer;
        interval.cache = record.cache;
        interval.queueId = record.queueId;
        interval.bytes = record.bytes;
        if (!openCLPrefixEventInterval(record.startEvent, record.endEvent,
                                       interval.startMs, interval.endMs)) {
            profilingUnavailable = true;
            continue;
        }
        pageFillIntervals.emplace_back(interval);
    }
    for (const auto& record : snapshot.computes) {
        OpenCLPrefixProfileInterval interval;
        interval.layer = record.layer;
        interval.queueId = record.queueId;
        if (!openCLPrefixEventInterval(record.startEvent, record.endEvent,
                                       interval.startMs, interval.endMs)) {
            profilingUnavailable = true;
            continue;
        }
        computeIntervals.emplace_back(interval);
    }
    for (const auto& record : snapshot.kernels) {
        OpenCLPrefixProfileInterval interval;
        interval.layer = record.layer;
        interval.name = record.name;
        interval.queueId = record.queueId;
        if (!openCLPrefixEventInterval(record.event, interval.startMs, interval.endMs)) {
            profilingUnavailable = true;
            continue;
        }
        kernelIntervals.emplace_back(interval);
    }

    double copyTotalMs = openCLPrefixTotalMs(copyIntervals);
    double pageFillTotalMs = openCLPrefixTotalMs(pageFillIntervals);
    double computeTotalMs = openCLPrefixTotalMs(computeIntervals);
    double copyUnionMs = openCLPrefixUnionMs(copyIntervals);
    double pageFillUnionMs = openCLPrefixUnionMs(pageFillIntervals);
    double computeUnionMs = openCLPrefixUnionMs(computeIntervals);
    double copyComputeOverlapMs = openCLPrefixOverlapUnionMs(copyIntervals, computeIntervals);
    double pageFillComputeOverlapMs = openCLPrefixOverlapUnionMs(pageFillIntervals, computeIntervals);
    double kernelComputeOverlapMs = openCLPrefixOverlapUnionMs(copyIntervals, kernelIntervals);
    double overlapRatio = copyUnionMs > 0.0 ? copyComputeOverlapMs / copyUnionMs : 0.0;
    double pageFillOverlapRatio = pageFillUnionMs > 0.0 ? pageFillComputeOverlapMs / pageFillUnionMs : 0.0;

    MNN_PRINT("[OpenCLPrefixAttentionPrefetchProfile] request_id=%llu request_overlap copy_events=%d page_fill_events=%d compute_events=%d kernel_events=%d expected_layers=%d copy_total_ms=%.3f page_fill_total_ms=%.3f compute_total_ms=%.3f copy_union_ms=%.3f page_fill_union_ms=%.3f compute_union_ms=%.3f copy_compute_overlap_ms=%.3f page_fill_compute_overlap_ms=%.3f kernel_compute_overlap_ms=%.3f overlap_ratio=%.3f page_fill_overlap_ratio=%.3f profiling_unavailable=%d\n",
              static_cast<unsigned long long>(requestId),
              static_cast<int>(copyIntervals.size()),
              static_cast<int>(pageFillIntervals.size()),
              static_cast<int>(computeIntervals.size()),
              static_cast<int>(kernelIntervals.size()),
              snapshot.expectedLayers,
              copyTotalMs,
              pageFillTotalMs,
              computeTotalMs,
              copyUnionMs,
              pageFillUnionMs,
              computeUnionMs,
              copyComputeOverlapMs,
              pageFillComputeOverlapMs,
              kernelComputeOverlapMs,
              overlapRatio,
              pageFillOverlapRatio,
              profilingUnavailable ? 1 : 0);

    int pairLimit = openCLPrefixProfilePairLimit();
    int pairCount = 0;
    for (const auto& copy : copyIntervals) {
        for (const auto& compute : computeIntervals) {
            double start = std::max(copy.startMs, compute.startMs);
            double end = std::min(copy.endMs, compute.endMs);
            double overlap = std::max(0.0, end - start);
            if (overlap <= 0.0) {
                continue;
            }
            if (pairCount++ >= pairLimit) {
                continue;
            }
            MNN_PRINT("[OpenCLPrefixAttentionPrefetchProfile] request_id=%llu request_overlap_pair copy_layer=%d copy_cache=%s compute_layer=%d overlap_ms=%.3f copy_queue_id=%llu compute_queue_id=%llu\n",
                      static_cast<unsigned long long>(requestId),
                      copy.layer,
                      copy.cache.c_str(),
                      compute.layer,
                      overlap,
                      static_cast<unsigned long long>(copy.queueId),
                      static_cast<unsigned long long>(compute.queueId));
        }
    }
    int pageFillPairCount = 0;
    for (const auto& fill : pageFillIntervals) {
        for (const auto& compute : computeIntervals) {
            double start = std::max(fill.startMs, compute.startMs);
            double end = std::min(fill.endMs, compute.endMs);
            double overlap = std::max(0.0, end - start);
            if (overlap <= 0.0) {
                continue;
            }
            if (pageFillPairCount++ >= pairLimit) {
                continue;
            }
            MNN_PRINT("[OpenCLPrefixAttentionPrefetchProfile] request_id=%llu page_fill_overlap_pair fill_layer=%d fill_cache=%s compute_layer=%d overlap_ms=%.3f fill_queue_id=%llu compute_queue_id=%llu\n",
                      static_cast<unsigned long long>(requestId),
                      fill.layer,
                      fill.cache.c_str(),
                      compute.layer,
                      overlap,
                      static_cast<unsigned long long>(fill.queueId),
                      static_cast<unsigned long long>(compute.queueId));
        }
    }
    int kernelPairCount = 0;
    for (const auto& copy : copyIntervals) {
        for (const auto& kernel : kernelIntervals) {
            double start = std::max(copy.startMs, kernel.startMs);
            double end = std::min(copy.endMs, kernel.endMs);
            double overlap = std::max(0.0, end - start);
            if (overlap <= 0.0) {
                continue;
            }
            if (kernelPairCount++ >= pairLimit) {
                continue;
            }
            MNN_PRINT("[OpenCLPrefixAttentionPrefetchProfile] request_id=%llu kernel_overlap copy_layer=%d copy_cache=%s kernel_layer=%d kernel=%s overlap_ms=%.3f copy_queue_id=%llu compute_queue_id=%llu\n",
                      static_cast<unsigned long long>(requestId),
                      copy.layer,
                      copy.cache.c_str(),
                      kernel.layer,
                      kernel.name.c_str(),
                      overlap,
                      static_cast<unsigned long long>(copy.queueId),
                      static_cast<unsigned long long>(kernel.queueId));
        }
    }
}

struct OpenCLPrefixSegmentPlacement {
    int logicalStart = 0;
    int physicalStart = 0;
    int tokenCount = 0;
    int physicalLength = 0;
    bool deviceCacheHit = false;
    double deviceCacheLookupMs = 0.0;
    std::string deviceCacheKey;
    std::string keyPath;
    std::string valuePath;
    std::string cacheName;
    cl::Event readyEvent;
};

struct OpenCLDevicePrefixDocumentRecord {
    int physicalStart = 0;
    int physicalLength = 0;
    int tokenCount = 0;
    int ropeDim = 0;
    float ropeTheta = 10000.0f;
    cl::Event readyEvent;
};

struct OpenCLDevicePrefixDocumentPool {
    std::mutex mutex;
    std::unordered_map<std::string, OpenCLDevicePrefixDocumentRecord> records;
    int nextPhysicalSlot = 0;
};

struct OpenCLPrefixInputShape {
    int batch = 0;
    int promptLen = 0;
    int kvHeads = 0;
    int headDim = 0;
};

static OpenCLPrefixInputShape openCLPrefixShapeFromInputs(const std::vector<Tensor*>& inputs) {
    OpenCLPrefixInputShape shape;
    auto query = inputs.empty() ? nullptr : inputs[0];
    auto key = inputs.size() > 1 ? inputs[1] : nullptr;
    if (query != nullptr && query->dimensions() > 0) {
        shape.batch = query->shape()[0];
    }
    if (key != nullptr && key->dimensions() > 3) {
        shape.promptLen = key->shape()[1];
        shape.kvHeads = key->shape()[2];
        shape.headDim = key->shape()[3];
    }
    return shape;
}

static bool openCLPrefixShapeValid(const OpenCLPrefixInputShape& shape) {
    return shape.batch > 0 && shape.promptLen > 0 && shape.kvHeads > 0 && shape.headDim > 0;
}

static OpenCLPrefixInputShape openCLPrefixShapeForRequest(const OpenCLPrefixInputShape& storedShape,
                                                          const OpenCLPrefixInputShape* fallbackShape,
                                                          const KVMeta* meta) {
    OpenCLPrefixInputShape shape = storedShape;
    if (fallbackShape != nullptr) {
        if (shape.batch <= 0) {
            shape.batch = fallbackShape->batch;
        }
        if (shape.promptLen <= 0) {
            shape.promptLen = fallbackShape->promptLen;
        }
        if (shape.kvHeads <= 0) {
            shape.kvHeads = fallbackShape->kvHeads;
        }
        if (shape.headDim <= 0) {
            shape.headDim = fallbackShape->headDim;
        }
    }
    if (meta != nullptr && meta->prefix_prompt_token_count > 0) {
        shape.promptLen = meta->prefix_prompt_token_count;
    }
    return shape;
}

static int openCLPrefixPrefetchThreadCountFromEnv() {
    const char* value = std::getenv("MNN_OPENCL_PREFIX_PREFETCH_THREADS");
    if (value != nullptr && value[0] != '\0' && std::atoi(value) != 1) {
        MNN_PRINT("[OpenCLPrefixAttentionPrefetch] ignore MNN_OPENCL_PREFIX_PREFETCH_THREADS=%s, force_single_worker=1\n",
                  value);
    }
    return 1;
}

class OpenCLPrefixPrefetchThreadPool {
public:
    using Task = std::function<void(cl::CommandQueue&)>;

    static std::shared_ptr<OpenCLPrefixPrefetchThreadPool> get(OpenCLRuntime* runtime) {
        if (runtime == nullptr) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(registryMutex());
        auto& registry = pools();
        auto iter = registry.find(runtime);
        if (iter != registry.end()) {
            return iter->second;
        }
        auto pool = std::shared_ptr<OpenCLPrefixPrefetchThreadPool>(
            new OpenCLPrefixPrefetchThreadPool(runtime, openCLPrefixPrefetchThreadCountFromEnv()));
        if (!pool->valid()) {
            return nullptr;
        }
        registry[runtime] = pool;
        return pool;
    }

    bool post(Task task) {
        if (!task || !mValid) {
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(mMutex);
            if (mStop) {
                return false;
            }
            mTasks.emplace(std::move(task));
        }
        mCondition.notify_one();
        return true;
    }

    int workerCount() const {
        return mWorkerCount;
    }

    bool valid() const {
        return mValid;
    }

private:
    OpenCLPrefixPrefetchThreadPool(OpenCLRuntime* runtime, int workerCount) : mWorkerCount(workerCount) {
        cl_int res = CL_SUCCESS;
        std::vector<cl::Device> devices = runtime->context().getInfo<CL_CONTEXT_DEVICES>(&res);
        if (res != CL_SUCCESS || devices.empty()) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention failed to get context device for prefetch queues: %d\n", res);
            return;
        }
        mQueues.resize(mWorkerCount);
        cl_command_queue_properties queueProperties =
            openCLPrefixProfileOverlapEnabled() ? CL_QUEUE_PROFILING_ENABLE : 0;
        for (int i = 0; i < mWorkerCount; ++i) {
            auto queue = std::make_shared<cl::CommandQueue>(runtime->context(), devices[0],
                                                            queueProperties, &res);
            if (res != CL_SUCCESS || queue == nullptr || (*queue)() == nullptr) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention failed to create prefetch queue %d: %d\n", i, res);
                return;
            }
            mQueues[i] = queue;
        }
        mValid = true;
        for (int i = 0; i < mWorkerCount; ++i) {
            mWorkers.emplace_back([this, i]() {
                while (true) {
                    Task task;
                    {
                        std::unique_lock<std::mutex> lock(mMutex);
                        mCondition.wait(lock, [this]() {
                            return mStop || !mTasks.empty();
                        });
                        if (mStop && mTasks.empty()) {
                            break;
                        }
                        task = std::move(mTasks.front());
                        mTasks.pop();
                    }
                    task(*mQueues[i]);
                }
            });
        }
        MNN_PRINT("[OpenCLPrefixAttentionPrefetch] thread_pool_workers=%d copy_queues=%d\n",
                  mWorkerCount, mWorkerCount);
    }

public:
    ~OpenCLPrefixPrefetchThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mStop = true;
        }
        mCondition.notify_all();
        for (auto& worker : mWorkers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

private:
    static std::mutex& registryMutex() {
        static std::mutex mutex;
        return mutex;
    }

    static std::unordered_map<OpenCLRuntime*, std::shared_ptr<OpenCLPrefixPrefetchThreadPool>>& pools() {
        static std::unordered_map<OpenCLRuntime*, std::shared_ptr<OpenCLPrefixPrefetchThreadPool>> value;
        return value;
    }

    int mWorkerCount = 4;
    bool mValid = false;
    std::vector<std::shared_ptr<cl::CommandQueue>> mQueues;
    std::vector<std::thread> mWorkers;
    std::queue<Task> mTasks;
    std::mutex mMutex;
    std::condition_variable mCondition;
    bool mStop = false;
};

static std::mutex& openCLDocumentPoolRegistryMutex() {
    static std::mutex mutex;
    return mutex;
}

static std::unordered_map<const KVCacheCLManager*, std::shared_ptr<OpenCLDevicePrefixDocumentPool>>&
openCLDocumentPoolRegistry() {
    static std::unordered_map<const KVCacheCLManager*, std::shared_ptr<OpenCLDevicePrefixDocumentPool>> value;
    return value;
}

static std::shared_ptr<OpenCLDevicePrefixDocumentPool> openCLDocumentPoolForCache(
        const std::shared_ptr<KVCacheCLManager>& cache) {
    if (cache == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(openCLDocumentPoolRegistryMutex());
    auto& registry = openCLDocumentPoolRegistry();
    auto iter = registry.find(cache.get());
    if (iter != registry.end()) {
        return iter->second;
    }
    auto pool = std::make_shared<OpenCLDevicePrefixDocumentPool>();
    registry.emplace(cache.get(), pool);
    return pool;
}

static std::mutex& openCLPrefixLayerCacheRegistryMutex() {
    static std::mutex mutex;
    return mutex;
}

struct OpenCLPrefixLayerCacheKey {
    OpenCLRuntime* runtime = nullptr;
    int layerIndex = -1;

    bool operator==(const OpenCLPrefixLayerCacheKey& other) const {
        return runtime == other.runtime && layerIndex == other.layerIndex;
    }
};

struct OpenCLPrefixLayerCacheKeyHash {
    size_t operator()(const OpenCLPrefixLayerCacheKey& key) const {
        size_t h0 = std::hash<void*>{}(key.runtime);
        size_t h1 = std::hash<int>{}(key.layerIndex);
        return h0 ^ (h1 + 0x9e3779b9 + (h0 << 6) + (h0 >> 2));
    }
};

static std::unordered_map<OpenCLPrefixLayerCacheKey, std::shared_ptr<KVCacheCLManager>,
                          OpenCLPrefixLayerCacheKeyHash>&
openCLPrefixLayerCacheRegistry() {
    static std::unordered_map<OpenCLPrefixLayerCacheKey, std::shared_ptr<KVCacheCLManager>,
                              OpenCLPrefixLayerCacheKeyHash> value;
    return value;
}

static std::string openCLDocumentPageCacheKey(const std::string& keyPath, const std::string& valuePath,
                                              const KVMeta::PrefixSegment& segment,
                                              int layerIndex, int runtimeBytes, int kvHeads, int headDim,
                                              int expectedRopeDim, float expectedRopeTheta) {
    std::string keyIdentity;
    std::string valueIdentity;
    size_t keySize = 0;
    size_t valueSize = 0;
    if (!MNN::detail::prefixKVFileIdentity(keyPath, keyIdentity, &keySize) ||
        !MNN::detail::prefixKVFileIdentity(valuePath, valueIdentity, &valueSize)) {
        return "";
    }
    return "backend=opencl|layout=" + segment.layout +
           "|cache=" + segment.cache_name +
           "|layer=" + std::to_string(layerIndex) +
           "|dtype=" + segment.dtype +
           "|runtime_bytes=" + std::to_string(runtimeBytes) +
           "|kv_heads=" + std::to_string(kvHeads) +
           "|head_dim=" + std::to_string(headDim) +
           "|tokens=" + std::to_string(segment.token_count) +
           "|rope_dim=" + std::to_string(expectedRopeDim) +
           "|rope_theta=" + std::to_string(expectedRopeTheta) +
           "|key=" + keyIdentity +
           "|value=" + valueIdentity +
           "|key_size=" + std::to_string(keySize) +
           "|value_size=" + std::to_string(valueSize);
}

} // namespace

class OpenCLPrefixAttentionBufExecution : public AttentionBufExecution {
public:
    OpenCLPrefixAttentionBufExecution(const MNN::Op* op, Backend* backend, bool kvCache, int layerIndex)
        : AttentionBufExecution(op, backend, kvCache), mLayerIndex(layerIndex) {
        adoptSharedKVCacheManager();
    }

    OpenCLPrefixAttentionBufExecution(std::shared_ptr<KVCacheCLManager> manager, const MNN::Op* op,
                                      Backend* backend, int layerIndex)
        : AttentionBufExecution(manager, op, backend), mLayerIndex(layerIndex) {
        adoptSharedKVCacheManager();
    }

    ~OpenCLPrefixAttentionBufExecution() override {
        unregisterForDevicePrefetch();
        waitDevicePrefetchTaskComplete();
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        mLastInputShape = openCLPrefixShapeFromInputs(inputs);
        ErrorCode code = NO_ERROR;
        if (isDirectSegmentsPrefix()) {
            setRecordClose closeRecord(mOpenCLBackend);
            code = AttentionBufExecution::onResize(inputs, outputs);
        } else {
            code = AttentionBufExecution::onResize(inputs, outputs);
        }
        if (code == NO_ERROR) {
            registerForDevicePrefetch();
        }
        return code;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (isDirectSegmentsPrefix()) {
            setRecordClose closeRecord(mOpenCLBackend);
            return AttentionBufExecution::onExecute(inputs, outputs);
        }
        return AttentionBufExecution::onExecute(inputs, outputs);
    }

    static bool submitAsyncDevicePrefetch(KVMeta* meta) {
        return submitDevicePrefetchForMeta(meta, nullptr);
    }

protected:
    virtual void onBeforeAttentionComputeEnqueue() override {
        mComputeProfileActive = false;
        mComputeProfileRequestId = 0;
        mComputeProfileLayer = -1;
        mComputeProfileQueueId = 0;
        mComputeProfileStartEvent = cl::Event();
        mComputeProfileEndEvent = cl::Event();
        if (!openCLPrefixProfileOverlapEnabled() || !isDirectSegmentsPrefix() ||
            mMeta == nullptr || mMeta->prefix_request_id == 0 ||
            mOpenCLBackend == nullptr || mOpenCLBackend->getOpenCLRuntime() == nullptr) {
            return;
        }
        auto* runtime = mOpenCLBackend->getOpenCLRuntime();
        runtime->setCommandQueueProfileEnable();
        auto& queue = runtime->commandQueue();
        cl_int status = openCLPrefixEnqueueMarker(queue, mComputeProfileStartEvent);
        if (status != CL_SUCCESS || mComputeProfileStartEvent() == nullptr) {
            MNN_PRINT("[OpenCLPrefixAttentionPrefetchProfile] layer=%d request_id=%llu compute_marker_start_error=%d profiling_unavailable=1\n",
                      mLayerIndex,
                      static_cast<unsigned long long>(mMeta->prefix_request_id),
                      status);
            runtime->setCommandQueueProfileDisable();
            return;
        }
        mComputeProfileActive = true;
        mComputeProfileRequestId = mMeta->prefix_request_id;
        mComputeProfileLayer = mLayerIndex;
        mComputeProfileQueueId = openCLPrefixQueueId(queue);
    }

    virtual bool onShouldProfileAttentionKernelEvents() const override {
        return openCLPrefixProfileOverlapEnabled() && isDirectSegmentsPrefix();
    }

    virtual void onAttentionKernelEvent(const std::string& name, const cl::Event& event) override {
        if (!mComputeProfileActive || event() == nullptr) {
            return;
        }
        openCLPrefixProfileRecordKernel(mComputeProfileRequestId, mComputeProfileLayer,
                                        name, mComputeProfileQueueId, event);
    }

    virtual void onAfterAttentionComputeEnqueue() override {
        if (!mComputeProfileActive || mOpenCLBackend == nullptr ||
            mOpenCLBackend->getOpenCLRuntime() == nullptr) {
            return;
        }
        auto* runtime = mOpenCLBackend->getOpenCLRuntime();
        auto& queue = runtime->commandQueue();
        cl_int status = openCLPrefixEnqueueMarker(queue, mComputeProfileEndEvent);
        if (status == CL_SUCCESS && mComputeProfileEndEvent() != nullptr) {
            openCLPrefixProfileRecordCompute(mComputeProfileRequestId, mComputeProfileLayer,
                                             mComputeProfileQueueId,
                                             mComputeProfileStartEvent, mComputeProfileEndEvent);
        } else {
            MNN_PRINT("[OpenCLPrefixAttentionPrefetchProfile] layer=%d request_id=%llu compute_marker_end_error=%d profiling_unavailable=1\n",
                      mComputeProfileLayer,
                      static_cast<unsigned long long>(mComputeProfileRequestId),
                      status);
        }
        runtime->setCommandQueueProfileDisable();
        openCLPrefixProfileMaybeReport(mComputeProfileRequestId);
        mComputeProfileActive = false;
    }

    virtual int onResizePrefixKVLength(const std::vector<Tensor*>& inputs, int seqlen) override {
        (void)seqlen;
        mResizeBasePastValid = false;
        mResizeBasePast = 0;
        mResizePrefixLength = 0;
        mResizeRequestId = 0;
        if (mMeta == nullptr || mMeta->file_flag != KVMeta::PendingReadSegments ||
            mMeta->prefix_segments.empty()) {
            return 0;
        }
        int segmentTotalTokens = mMeta->segment_total_tokens;
        if (segmentTotalTokens <= 0) {
            for (const auto& segment : mMeta->prefix_segments) {
                segmentTotalTokens += segment.token_count;
            }
        }
        int basePast = 0;
        if (segmentTotalTokens > 0 && mKVCacheCLManager != nullptr &&
            buildPrefixRuntimeKVBlockTablePlan(mPagedKVPlan, mMeta, seqlen)) {
            auto* runtime = mOpenCLBackend->getOpenCLRuntime();
            runtime->commandQueue().finish();
            if (mMeta->previous != mMeta->remove) {
                basePast = mKVCacheCLManager->pastKvLength();
                if (basePast <= 0) {
                    basePast = static_cast<int>(mMeta->previous - mMeta->remove);
                }
                if (!mKVCacheCLManager->pagedActive()) {
                    int headDim = inputs.size() > 1 && inputs[1] != nullptr &&
                                  inputs[1]->dimensions() > 3 ? inputs[1]->shape()[3] : 0;
                    int restoreRopeDim = !mMeta->prefix_segments.empty() &&
                                         mMeta->prefix_segments[0].rope_dim > 0 ?
                                         mMeta->prefix_segments[0].rope_dim : headDim;
                    float restoreRopeTheta = !mMeta->prefix_segments.empty() &&
                                             mMeta->prefix_segments[0].rope_theta > 0.0f ?
                                             mMeta->prefix_segments[0].rope_theta : 10000.0f;
                    if (!mKVCacheCLManager->ensurePagedStateForExistingHistory(
                            basePast, alignToPage(basePast, RuntimeKVBlockTablePlan::kDefaultPageSize),
                            RuntimeKVBlockTablePlan::kDefaultPageSize,
                            restoreRopeDim, restoreRopeTheta)) {
                        return 0;
                    }
                }
            }
            int ropeDim = 0;
            float ropeTheta = 10000.0f;
            if (!mMeta->prefix_segments.empty()) {
                ropeDim = mMeta->prefix_segments[0].rope_dim;
                ropeTheta = mMeta->prefix_segments[0].rope_theta > 0.0f ?
                    mMeta->prefix_segments[0].rope_theta : ropeTheta;
            }
            constexpr int pageSize = RuntimeKVBlockTablePlan::kDefaultPageSize;
            int physicalBase = basePast > 0 ? ROUND_UP(mKVCacheCLManager->pagedPhysicalLength(), pageSize) : 0;
            int logicalTotal = basePast + segmentTotalTokens + seqlen;
            int physicalTotal = physicalBase + mPagedKVPlan.physicalLength();
            mKVCacheCLManager->setHostPtrPagedEnabled(openCLPrefixUseHostPtrPagesEnabled());
            mKVCacheCLManager->discardPagedTablesForRewrite();
            if (mKVCacheCLManager->ensurePagedCapacity(physicalTotal, logicalTotal)) {
                mKVCacheCLManager->setPagedState(basePast + segmentTotalTokens, physicalTotal,
                                                 mPagedKVPlan.pageSize(), ropeDim, ropeTheta);
                mResizeBasePast = basePast;
                mResizePrefixLength = basePast + segmentTotalTokens;
                mResizeRequestId = mMeta->prefix_request_id;
                mResizeBasePastValid = true;
            }
        }
        return basePast + segmentTotalTokens;
    }

    virtual ErrorCode onPrepareKVCacheBeforeAppend(const std::vector<Tensor*>& inputs, bool& prepared,
                                                   int& appendKvSeqLen) override {
        prepared = false;
        appendKvSeqLen = inputs[1]->shape()[1];
        if (mMeta == nullptr || mMeta->file_flag != KVMeta::PendingReadSegments ||
            mMeta->prefix_segments.empty()) {
            return NO_ERROR;
        }

        if (mMeta->prefix_device_prefetch && mMeta->prefix_request_id != 0) {
            mLastInputShape = openCLPrefixShapeFromInputs(inputs);
            auto submitErr = submitDevicePrefetchForRequestIfNeeded(inputs);
            if (submitErr != NO_ERROR) {
                return submitErr;
            }
            bool prefetchHit = false;
            auto waitErr = consumeDevicePrefetch(prefetchHit, appendKvSeqLen);
            if (waitErr != NO_ERROR) {
                return waitErr;
            }
            if (prefetchHit) {
                prepared = true;
                return NO_ERROR;
            }
            MNN_PRINT("[OpenCLPrefixAttention] layer=%d request_id=%llu device_prefetch=1 device_prefetch_hit=0 fallback_sync=1\n",
                      mLayerIndex, static_cast<unsigned long long>(mMeta->prefix_request_id));
        }

        PrefixPrepareSummary summary;
        auto inputShape = openCLPrefixShapeForRequest(openCLPrefixShapeFromInputs(inputs), nullptr, mMeta);
        auto err = preparePrefixKVOnQueue(inputShape, mOpenCLBackend->getOpenCLRuntime()->defaultCommandQueue(),
                                          true, false, appendKvSeqLen, summary, nullptr);
        if (err != NO_ERROR) {
            return err;
        }
        prepared = summary.layerIndex >= 0;
        return NO_ERROR;
    }

    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (dst == nullptr) {
            return true;
        }
        auto* exe = new OpenCLPrefixAttentionBufExecution(mKVCacheCLManager, op, bn, mLayerIndex);
        exe->mMeta = mMeta;
        exe->mLastInputShape = mLastInputShape;
        exe->adoptSharedKVCacheManager();
        *dst = exe;
        return true;
    }

private:
    struct PrefixPrepareSummary {
        int layerIndex = -1;
        int basePast = 0;
        int segmentTotalTokens = 0;
        int promptTokens = 0;
        int pastAfterPrefix = 0;
    };

    struct SharedPrefetchRecord {
        std::mutex mutex;
        std::condition_variable condition;
        bool pending = false;
        bool ready = false;
        cl::Event event;
        bool hasEvent = false;
        ErrorCode status = NO_ERROR;
        int appendKvSeqLen = 0;
        PrefixPrepareSummary summary;
    };

    bool isDirectSegmentsPrefix() const {
        return mMeta != nullptr && mMeta->file_flag == KVMeta::PendingReadSegments &&
               !mMeta->prefix_segments.empty();
    }

    void adoptSharedKVCacheManager() {
        if (mLayerIndex < 0 || mKVCacheCLManager == nullptr ||
            mOpenCLBackend == nullptr || mOpenCLBackend->getOpenCLRuntime() == nullptr) {
            return;
        }
        OpenCLPrefixLayerCacheKey key;
        key.runtime = mOpenCLBackend->getOpenCLRuntime();
        key.layerIndex = mLayerIndex;
        std::lock_guard<std::mutex> lock(openCLPrefixLayerCacheRegistryMutex());
        auto& slot = openCLPrefixLayerCacheRegistry()[key];
        if (slot != nullptr) {
            mKVCacheCLManager = slot;
            return;
        }
        slot = mKVCacheCLManager;
    }

    ErrorCode preparePrefixKVOnQueue(const OpenCLPrefixInputShape& inputShape, cl::CommandQueue& queue,
                                     bool allowMetaLayerCursor, bool devicePrefetch,
                                     int& appendKvSeqLen, PrefixPrepareSummary& summary,
                                     cl::Event* readyEvent) {
        std::unique_lock<std::recursive_mutex> queueLock(openCLPrefixQueueMutex(), std::defer_lock);
        if (!devicePrefetch) {
            queueLock.lock();
        }

        int batch = inputShape.batch;
        int promptLen = inputShape.promptLen;
        int kvHeads = inputShape.kvHeads;
        int headDim = inputShape.headDim;
        if (batch != 1) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention direct_segments only supports batch=1 now\n");
            return NOT_SUPPORT;
        }
        if ((mMeta->remove > 0 || mMeta->n_reserve > 0) && mMeta->previous != mMeta->remove) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention direct_segments append does not support remove/reserve compaction yet\n");
            return NOT_SUPPORT;
        }
        if (mMeta->prefix_cache_dir.empty()) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention missing prefix_cache_dir in KVMeta\n");
            return INVALID_VALUE;
        }

        int segmentTotalTokens = mMeta->segment_total_tokens;
        if (segmentTotalTokens <= 0) {
            for (const auto& segment : mMeta->prefix_segments) {
                segmentTotalTokens += segment.token_count;
            }
            mMeta->segment_total_tokens = segmentTotalTokens;
        }
        if (segmentTotalTokens <= 0) {
            return INVALID_VALUE;
        }
        if (!buildPrefixRuntimeKVBlockTablePlan(mPagedKVPlan, mMeta, promptLen)) {
            return INVALID_VALUE;
        }

        int basePast = 0;
        if (mMeta->previous != mMeta->remove) {
            const int metaBasePast = static_cast<int>(mMeta->previous - mMeta->remove);
            const int managerPast = mKVCacheCLManager->pastKvLength();
            const bool useResizeBasePast = mResizeBasePastValid &&
                mResizeRequestId == mMeta->prefix_request_id;
            if (useResizeBasePast) {
                basePast = mResizeBasePast;
            } else if (mKVCacheCLManager->pagedActive() && managerPast >= segmentTotalTokens) {
                basePast = managerPast - segmentTotalTokens;
            } else {
                basePast = managerPast;
            }
            if (basePast <= 0) {
                basePast = metaBasePast;
            }
            if (mKVCacheCLManager->pagedActive()) {
                const int expectedPreparedPast = basePast + segmentTotalTokens;
                if (managerPast != basePast && managerPast != expectedPreparedPast) {
                    MNN_ERROR("[Error]: OpenCL PrefixAttention paged base history mismatch: "
                              "meta=%d cache=%d base_past=%d expected_prepared=%d request_id=%llu layer=%d\n",
                              metaBasePast, managerPast, basePast, expectedPreparedPast,
                              static_cast<unsigned long long>(mMeta->prefix_request_id), mLayerIndex);
                    return INVALID_VALUE;
                }
            }
            if (mLayerIndex == 0 && metaBasePast != basePast) {
                MNN_PRINT("[OpenCLPrefixAttention] base_past adjusted to runtime cache length: meta=%d cache=%d\n",
                          metaBasePast, basePast);
            }
            if (!mKVCacheCLManager->pagedActive()) {
                if (!mKVCacheCLManager->ensurePagedStateForExistingHistory(
                        basePast, alignToPage(basePast, RuntimeKVBlockTablePlan::kDefaultPageSize),
                        RuntimeKVBlockTablePlan::kDefaultPageSize, headDim, 10000.0f)) {
                    MNN_ERROR("[Error]: OpenCL PrefixAttention direct_segments append requires existing paged runtime KV cache; "
                              "reset before first direct_segments request. layer=%d manager=%p previous=%zu remove=%zu "
                              "base_past=%d manager_past=%d\n",
                              mLayerIndex, mKVCacheCLManager.get(), mMeta->previous, mMeta->remove,
                              basePast, mKVCacheCLManager->pastKvLength());
                    return NOT_SUPPORT;
                }
            }
        }

        int runtimeBytes = mKVCacheCLManager->byte();
        if (runtimeBytes != 2 && runtimeBytes != 4) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention only supports fp16/fp32 runtime KV cache\n");
            return NOT_SUPPORT;
        }

        int layerIndex = mLayerIndex;
        if (layerIndex < 0) {
            if (!allowMetaLayerCursor) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention device_prefetch requires explicit AttentionParam.layer_index\n");
                return NOT_SUPPORT;
            }
            layerIndex = mMeta->layer_index;
            if (mMeta->layer_nums > 0) {
                mMeta->layer_index = (mMeta->layer_index + 1) % mMeta->layer_nums;
            } else {
                mMeta->layer_index++;
            }
        }

        auto docPool = openCLDocumentPoolForCache(mKVCacheCLManager);
        if (docPool == nullptr) {
            return OUT_OF_MEMORY;
        }

        constexpr int pageSize = RuntimeKVBlockTablePlan::kDefaultPageSize;
        int physicalCursor = basePast > 0 ? std::max(mKVCacheCLManager->pagedPhysicalLength(), basePast) : 0;
        {
            std::lock_guard<std::mutex> lock(docPool->mutex);
            physicalCursor = std::max(physicalCursor, docPool->nextPhysicalSlot);
        }
        std::vector<OpenCLPrefixSegmentPlacement> placements;
        placements.reserve(mMeta->prefix_segments.size());
        int logicalCursor = basePast;
        int segmentIndex = 0;
        int ropeDimForRequest = 0;
        float ropeThetaForRequest = 10000.0f;
        for (const auto& segment : mMeta->prefix_segments) {
            if (segment.token_count <= 0) {
                segmentIndex++;
                continue;
            }
            if (segment.key_rope_state != KVMeta::KeyRopeCanonicalNoRope ||
                segment.rope_pairing != KVMeta::RopePairingHalf) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention requires canonical_no_rope half-split prefix cache: %s\n",
                          segment.cache_name.c_str());
                return NOT_SUPPORT;
            }
            if (segment.backend != "opencl" || segment.layout != kOpenCLPrefixLayout) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention requires backend-native OpenCL cache for %s, got backend=%s layout=%s\n",
                          segment.cache_name.c_str(), segment.backend.c_str(), segment.layout.c_str());
                return NOT_SUPPORT;
            }

            int ropeDim = segment.rope_dim > 0 ? segment.rope_dim : headDim;
            ropeDim = std::min(ropeDim, headDim);
            if (ropeDim <= 0 || (ropeDim % 2) != 0 || segment.rope_theta <= 0.0f) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention invalid RoPE metadata for cache %s\n",
                          segment.cache_name.c_str());
                return INVALID_VALUE;
            }
            if (ropeDimForRequest <= 0) {
                ropeDimForRequest = ropeDim;
                ropeThetaForRequest = segment.rope_theta;
            } else if (ropeDimForRequest != ropeDim ||
                       std::fabs(ropeThetaForRequest - segment.rope_theta) > 1e-5f) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention requires consistent RoPE metadata across direct_segments\n");
                return NOT_SUPPORT;
            }

            auto basePath = prefixCacheLayerBase(mMeta->prefix_cache_dir, "opencl", segment.cache_name, layerIndex);
            auto keyPath = basePath + ".k";
            auto valuePath = basePath + ".v";
            size_t keyFileSize = 0;
            size_t valueFileSize = 0;
            std::string keyIdentity;
            std::string valueIdentity;
            if (!MNN::detail::prefixKVFileIdentity(keyPath, keyIdentity, &keyFileSize) ||
                !MNN::detail::prefixKVFileIdentity(valuePath, valueIdentity, &valueFileSize)) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention missing native KV file for %s layer %d\n",
                          segment.cache_name.c_str(), layerIndex);
                return FILE_OPEN_FAILED;
            }
            size_t expectedBytes = openCLNativeFileBytes(segment.token_count, kvHeads, headDim, runtimeBytes);
            if (keyFileSize != expectedBytes || valueFileSize != expectedBytes) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention invalid native KV size for %s layer %d: key=%zu value=%zu expected=%zu\n",
                          segment.cache_name.c_str(), layerIndex, keyFileSize, valueFileSize, expectedBytes);
                return INVALID_VALUE;
            }
            auto cacheKey = openCLDocumentPageCacheKey(keyPath, valuePath, segment, layerIndex,
                                                       runtimeBytes, kvHeads, headDim,
                                                       ropeDim, segment.rope_theta);
            if (cacheKey.empty()) {
                return FILE_OPEN_FAILED;
            }

            OpenCLPrefixSegmentPlacement placement;
            placement.logicalStart = logicalCursor;
            placement.tokenCount = segment.token_count;
            placement.physicalLength = alignToPage(segment.token_count, pageSize);
            placement.deviceCacheKey = cacheKey;
            placement.keyPath = keyPath;
            placement.valuePath = valuePath;
            placement.cacheName = segment.cache_name;

            double lookupStart = nowMs();
            {
                std::lock_guard<std::mutex> lock(docPool->mutex);
                auto iter = docPool->records.find(cacheKey);
                if (iter != docPool->records.end() &&
                    iter->second.tokenCount == segment.token_count &&
                    iter->second.ropeDim == ropeDim &&
                    std::fabs(iter->second.ropeTheta - segment.rope_theta) <= 1e-5f) {
                    placement.deviceCacheHit = true;
                    placement.physicalStart = iter->second.physicalStart;
                    placement.physicalLength = iter->second.physicalLength;
                    placement.readyEvent = iter->second.readyEvent;
                } else {
                    int physicalStart = alignToPage(std::max(physicalCursor, docPool->nextPhysicalSlot), pageSize);
                    placement.physicalStart = physicalStart;
                    docPool->nextPhysicalSlot = physicalStart + placement.physicalLength;
                    physicalCursor = docPool->nextPhysicalSlot;
                }
            }
            placement.deviceCacheLookupMs = nowMs() - lookupStart;
            physicalCursor = std::max(physicalCursor, placement.physicalStart + placement.physicalLength);
            placements.emplace_back(placement);
            logicalCursor += segment.token_count;
            segmentIndex++;
        }

        if (logicalCursor != basePast + segmentTotalTokens) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention planned prefix length %d but expected %d\n",
                      logicalCursor - basePast, segmentTotalTokens);
            return INVALID_VALUE;
        }

        const int promptLogicalStart = basePast + segmentTotalTokens;
        const int promptPhysicalStart = alignToPage(physicalCursor, pageSize);
        physicalCursor = promptPhysicalStart + promptLen;
        const int requiredTotal = basePast + segmentTotalTokens + promptLen;
        const int requiredPhysicalTotal = alignToPage(physicalCursor, pageSize);
        {
            std::lock_guard<std::mutex> lock(docPool->mutex);
            docPool->nextPhysicalSlot = std::max(docPool->nextPhysicalSlot, requiredPhysicalTotal);
        }
        queue.finish();
        mKVCacheCLManager->setHostPtrPagedEnabled(openCLPrefixUseHostPtrPagesEnabled());
        mKVCacheCLManager->discardPagedTablesForRewrite();
        if (!mKVCacheCLManager->ensurePagedCapacity(requiredPhysicalTotal, requiredTotal)) {
            return OUT_OF_MEMORY;
        }

        int maxLen = ROUND_UP(mKVCacheCLManager->maxLength(), 4);
        mPendingHostUploads.clear();

        if (basePast > 0) {
            const auto& baseTokenTable = mKVCacheCLManager->pagedTokenTableHost();
            const auto& baseRopeTable = mKVCacheCLManager->pagedRopeTableHost();
            if (baseTokenTable.size() < static_cast<size_t>(basePast) ||
                baseRopeTable.size() < static_cast<size_t>(basePast) ||
                !mKVCacheCLManager->uploadPagedTableRange(0, basePast)) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention failed to restore base paged token table "
                          "base_past=%d token_host=%zu rope_host=%zu\n",
                          basePast, baseTokenTable.size(), baseRopeTable.size());
                return INVALID_VALUE;
            }
        }

        std::vector<int> tokenTable(requiredTotal - basePast, 0);
        std::vector<int> ropeTable(requiredTotal - basePast, 0);
        for (const auto& placement : placements) {
            for (int i = 0; i < placement.tokenCount; ++i) {
                int offset = placement.logicalStart + i - basePast;
                tokenTable[offset] = placement.physicalStart + i;
                ropeTable[offset] = 1;
            }
        }
        for (int i = 0; i < promptLen; ++i) {
            int offset = promptLogicalStart + i - basePast;
            tokenTable[offset] = promptPhysicalStart + i;
            ropeTable[offset] = 0;
        }
        cl_int tableStatus = CL_SUCCESS;
        cl::Event tableEvent;
        if (!mKVCacheCLManager->updatePagedTableHost(basePast, tokenTable, ropeTable)) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention failed to update host paged token table "
                      "base_past=%d write_tokens=%zu required_total=%d\n",
                      basePast, tokenTable.size(), requiredTotal);
            return INVALID_VALUE;
        }
        tableStatus |= queue.enqueueWriteBuffer(*mKVCacheCLManager->mutableTokenTable(), CL_TRUE,
                                                static_cast<size_t>(basePast) * sizeof(int),
                                                tokenTable.size() * sizeof(int), tokenTable.data());
        tableStatus |= queue.enqueueWriteBuffer(*mKVCacheCLManager->mutableRopeTable(), CL_TRUE,
                                                static_cast<size_t>(basePast) * sizeof(int),
                                                ropeTable.size() * sizeof(int), ropeTable.data(),
                                                nullptr, &tableEvent);
        if (tableStatus != CL_SUCCESS) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention failed to upload paged token table: %d\n", tableStatus);
            return INVALID_VALUE;
        }

        cl::Event lastEvent = tableEvent;
        bool hasLastEvent = tableEvent() != nullptr;
        logicalCursor = basePast;
        segmentIndex = 0;
        size_t placementIndex = 0;
        for (const auto& segment : mMeta->prefix_segments) {
            if (segment.token_count <= 0) {
                segmentIndex++;
                continue;
            }
            const auto& placement = placements[placementIndex++];
            int ropeDim = segment.rope_dim > 0 ? segment.rope_dim : headDim;
            ropeDim = std::min(ropeDim, headDim);

            bool hostCacheHit = false;
            double prefetchWaitMs = 0.0;
            double diskReadMs = 0.0;
            double uploadEnqueueMs = 0.0;
            double attentionWaitMs = 0.0;
            size_t uploadBytes = 0;
            size_t pageFillBytes = 0;
            bool directHostPages = false;
            const size_t elemBytes = static_cast<size_t>(runtimeBytes);
            cl::Event pageFillStartEvent;
            cl::Event pageFillEndEvent;
            bool hasPageFillStartEvent = false;

            if (placement.deviceCacheHit) {
                double waitStart = nowMs();
                cl_int waitStatus = enqueueWaitForEvent(queue, placement.readyEvent);
                attentionWaitMs = nowMs() - waitStart;
                if (waitStatus != CL_SUCCESS) {
                    MNN_ERROR("[Error]: OpenCL PrefixAttention failed to wait document page event: %d\n", waitStatus);
                    return INVALID_VALUE;
                }
            } else {
                if (openCLPrefixProfileOverlapEnabled()) {
                    cl_int markerStatus = openCLPrefixEnqueueMarker(queue, pageFillStartEvent);
                    queue.flush();
                    hasPageFillStartEvent = markerStatus == CL_SUCCESS && pageFillStartEvent() != nullptr;
                }
                auto keyRead = readPrefixKVHostCacheFile(placement.keyPath);
                auto valueRead = readPrefixKVHostCacheFile(placement.valuePath);
                if (!keyRead.ok || !valueRead.ok || keyRead.bytes == nullptr || valueRead.bytes == nullptr) {
                    MNN_ERROR("[Error]: OpenCL PrefixAttention failed to read prefix KV cache: %s / %s, key_error=%s value_error=%s\n",
                              placement.keyPath.c_str(), placement.valuePath.c_str(),
                              keyRead.error.c_str(), valueRead.error.c_str());
                    return FILE_OPEN_FAILED;
                }
                auto& keyBytes = *keyRead.bytes;
                auto& valueBytes = *valueRead.bytes;

                size_t expectedBytes = openCLNativeFileBytes(segment.token_count, kvHeads, headDim, runtimeBytes);
                if (keyBytes.size() != expectedBytes || valueBytes.size() != expectedBytes) {
                    MNN_ERROR("[Error]: OpenCL PrefixAttention invalid native KV size for %s layer %d: key=%zu value=%zu expected=%zu\n",
                              segment.cache_name.c_str(), layerIndex, keyBytes.size(), valueBytes.size(), expectedBytes);
                    return INVALID_VALUE;
                }

                double uploadStart = nowMs();
                cl_int uploadStatus = CL_SUCCESS;
                int alignedTokens = ROUND_UP(segment.token_count, 4);
                cl::Event firstUploadEvent;
                cl::Event lastUploadEvent;
                bool hasUploadProfileEvent = false;
                directHostPages = mKVCacheCLManager->hostPtrPagedEnabled() &&
                    mKVCacheCLManager->mutableKeyHostPtr() != nullptr &&
                    mKVCacheCLManager->mutableValueHostPtr() != nullptr;
                uint8_t* keyHostPtr = directHostPages ? mKVCacheCLManager->mutableKeyHostPtr() : nullptr;
                uint8_t* valueHostPtr = directHostPages ? mKVCacheCLManager->mutableValueHostPtr() : nullptr;
                const size_t runtimeBufferBytes = static_cast<size_t>(kvHeads) *
                    static_cast<size_t>(headDim) * static_cast<size_t>(maxLen) * elemBytes;
                void* mappedKeyPtr = nullptr;
                void* mappedValuePtr = nullptr;
                if (directHostPages) {
                    cl_int mapKeyStatus = CL_SUCCESS;
                    cl_int mapValueStatus = CL_SUCCESS;
                    mappedKeyPtr = queue.enqueueMapBuffer(*mKVCacheCLManager->mutableKey(), CL_TRUE,
                                                          CL_MAP_WRITE, 0, runtimeBufferBytes,
                                                          nullptr, nullptr, &mapKeyStatus);
                    mappedValuePtr = queue.enqueueMapBuffer(*mKVCacheCLManager->mutableValue(), CL_TRUE,
                                                            CL_MAP_WRITE, 0, runtimeBufferBytes,
                                                            nullptr, nullptr, &mapValueStatus);
                    if (mapKeyStatus != CL_SUCCESS || mapValueStatus != CL_SUCCESS ||
                        mappedKeyPtr == nullptr || mappedValuePtr == nullptr) {
                        MNN_ERROR("[Error]: OpenCL PrefixAttention failed to map host-ptr runtime KV pages: "
                                  "layer=%d cache=%s key_status=%d value_status=%d bytes=%zu\n",
                                  layerIndex, segment.cache_name.c_str(),
                                  mapKeyStatus, mapValueStatus, runtimeBufferBytes);
                        if (mappedKeyPtr != nullptr) {
                            queue.enqueueUnmapMemObject(*mKVCacheCLManager->mutableKey(), mappedKeyPtr);
                        }
                        if (mappedValuePtr != nullptr) {
                            queue.enqueueUnmapMemObject(*mKVCacheCLManager->mutableValue(), mappedValuePtr);
                        }
                        return INVALID_VALUE;
                    }
                    keyHostPtr = reinterpret_cast<uint8_t*>(mappedKeyPtr);
                    valueHostPtr = reinterpret_cast<uint8_t*>(mappedValuePtr);
                }
                for (const auto& ref : mPagedKVPlan.refs()) {
                    if (ref.source != RuntimeKVBlockRef::Document || ref.segmentIndex != segmentIndex ||
                        ref.tokenCount <= 0) {
                        continue;
                    }
                    for (int head = 0; head < kvHeads; ++head) {
                        for (int dim = 0; dim < headDim; ++dim) {
                            size_t srcOffset = ((static_cast<size_t>(head) * headDim + dim) *
                                                static_cast<size_t>(alignedTokens) +
                                                static_cast<size_t>(ref.sourceTokenStart)) * elemBytes;
                            size_t dstOffset = ((static_cast<size_t>(head) * headDim + dim) *
                                                static_cast<size_t>(maxLen) +
                                                static_cast<size_t>(placement.physicalStart + ref.sourceTokenStart)) *
                                               elemBytes;
                            if (directHostPages) {
                                ::memcpy(keyHostPtr + dstOffset, keyBytes.data() + srcOffset,
                                         static_cast<size_t>(ref.tokenCount) * elemBytes);
                            } else {
                                cl::Event writeEvent;
                                uploadStatus |= queue.enqueueWriteBuffer(*mKVCacheCLManager->mutableKey(), CL_FALSE,
                                                                         dstOffset,
                                                                         static_cast<size_t>(ref.tokenCount) * elemBytes,
                                                                         keyBytes.data() + srcOffset, nullptr, &writeEvent);
                                if (!hasUploadProfileEvent && writeEvent() != nullptr) {
                                    firstUploadEvent = writeEvent;
                                    hasUploadProfileEvent = true;
                                }
                                lastUploadEvent = writeEvent;
                                lastEvent = writeEvent;
                                hasLastEvent = writeEvent() != nullptr;
                            }
                        }
                        size_t srcValueOffset = (static_cast<size_t>(head) * static_cast<size_t>(alignedTokens) +
                                                 static_cast<size_t>(ref.sourceTokenStart)) *
                                                static_cast<size_t>(headDim) * elemBytes;
                        size_t dstValueOffset = (static_cast<size_t>(head) * static_cast<size_t>(maxLen) +
                                                 static_cast<size_t>(placement.physicalStart + ref.sourceTokenStart)) *
                                                static_cast<size_t>(headDim) * elemBytes;
                        if (directHostPages) {
                            ::memcpy(valueHostPtr + dstValueOffset, valueBytes.data() + srcValueOffset,
                                     static_cast<size_t>(ref.tokenCount) *
                                     static_cast<size_t>(headDim) * elemBytes);
                        } else {
                            cl::Event writeEvent;
                            uploadStatus |= queue.enqueueWriteBuffer(*mKVCacheCLManager->mutableValue(), CL_FALSE,
                                                                     dstValueOffset,
                                                                     static_cast<size_t>(ref.tokenCount) *
                                                                     static_cast<size_t>(headDim) * elemBytes,
                                                                     valueBytes.data() + srcValueOffset, nullptr, &writeEvent);
                            if (!hasUploadProfileEvent && writeEvent() != nullptr) {
                                firstUploadEvent = writeEvent;
                                hasUploadProfileEvent = true;
                            }
                            lastUploadEvent = writeEvent;
                            lastEvent = writeEvent;
                            hasLastEvent = writeEvent() != nullptr;
                        }
                    }
                }
                if (directHostPages && uploadStatus == CL_SUCCESS) {
                    cl::Event keyUnmapEvent;
                    cl::Event valueUnmapEvent;
                    uploadStatus |= queue.enqueueUnmapMemObject(*mKVCacheCLManager->mutableKey(),
                                                                mappedKeyPtr, nullptr, &keyUnmapEvent);
                    uploadStatus |= queue.enqueueUnmapMemObject(*mKVCacheCLManager->mutableValue(),
                                                                mappedValuePtr, nullptr, &valueUnmapEvent);
                    if (uploadStatus == CL_SUCCESS && valueUnmapEvent() != nullptr) {
                        lastEvent = valueUnmapEvent;
                        hasLastEvent = true;
                    }
                }
                if (uploadStatus != CL_SUCCESS) {
                    MNN_ERROR("[Error]: OpenCL PrefixAttention failed to upload prefix KV to paged runtime cache: "
                              "layer=%d cache=%s err=%d key_bytes=%zu value_bytes=%zu\n",
                              layerIndex, segment.cache_name.c_str(), uploadStatus,
                              keyBytes.size(), valueBytes.size());
                    return INVALID_VALUE;
                }
                uploadEnqueueMs = nowMs() - uploadStart;
                pageFillBytes = keyRead.file_size + valueRead.file_size;
                uploadBytes = directHostPages ? 0 : pageFillBytes;
                if (!directHostPages && hasUploadProfileEvent && lastUploadEvent() != nullptr) {
                    openCLPrefixProfileRecordCopy(mMeta->prefix_request_id, layerIndex,
                                                  placement.cacheName,
                                                  openCLPrefixQueueId(queue),
                                                  uploadBytes,
                                                  firstUploadEvent,
                                                  lastUploadEvent);
                }
                if (hasPageFillStartEvent) {
                    cl_int markerStatus = openCLPrefixEnqueueMarker(queue, pageFillEndEvent);
                    queue.flush();
                    if (markerStatus == CL_SUCCESS && pageFillEndEvent() != nullptr) {
                        openCLPrefixProfileRecordPageFill(mMeta->prefix_request_id, layerIndex,
                                                          placement.cacheName,
                                                          openCLPrefixQueueId(queue),
                                                          pageFillBytes,
                                                          keyRead.disk_read_ms + valueRead.disk_read_ms,
                                                          uploadEnqueueMs,
                                                          pageFillStartEvent,
                                                          pageFillEndEvent);
                    }
                }
                hostCacheHit = keyRead.host_cache_hit && valueRead.host_cache_hit;
                prefetchWaitMs = keyRead.prefetch_wait_ms + valueRead.prefetch_wait_ms;
                diskReadMs = keyRead.disk_read_ms + valueRead.disk_read_ms;
                mPendingHostUploads.emplace_back(keyRead.bytes);
                mPendingHostUploads.emplace_back(valueRead.bytes);

                if (hasLastEvent) {
                    OpenCLDevicePrefixDocumentRecord record;
                    record.physicalStart = placement.physicalStart;
                    record.physicalLength = placement.physicalLength;
                    record.tokenCount = segment.token_count;
                    record.ropeDim = ropeDim;
                    record.ropeTheta = segment.rope_theta;
                    record.readyEvent = lastEvent;
                    std::lock_guard<std::mutex> lock(docPool->mutex);
                    docPool->records[placement.deviceCacheKey] = record;
                }
            }

            const double waitMs = (!devicePrefetch && mMeta->prefix_device_prefetch) ?
                prefetchWaitMs : attentionWaitMs;
            MNN_PRINT("[OpenCLPrefixAttention] layer=%d cache=%s tokens=%d logical_start=%d physical_start=%d host_cache_hit=%d device_cache_hit=%d device_cache_lookup_ms=%.3f device_prefetch=%d device_prefetch_submit=%d device_prefetch_hit=0 prefetch_wait_ms=%.3f disk_read_ms=%.3f host_to_device_ms=%.3f materialize_ms=0.000 attention_wait_ms=%.3f upload_enqueue_bytes=%zu page_fill_bytes=%zu staging_path=0 direct_runtime_upload=0 shared_host_pages=%d direct_host_write=%d page_pool=1 on_read_rope=1 page_size=%d\n",
                      layerIndex, placement.cacheName.c_str(), segment.token_count,
                      placement.logicalStart, placement.physicalStart,
                      hostCacheHit ? 1 : 0,
                      placement.deviceCacheHit ? 1 : 0,
                      placement.deviceCacheLookupMs,
                      mMeta->prefix_device_prefetch ? 1 : 0,
                      devicePrefetch ? 1 : 0,
                      prefetchWaitMs,
                      diskReadMs,
                      uploadEnqueueMs,
                      waitMs,
                      uploadBytes,
                      pageFillBytes,
                      (mKVCacheCLManager != nullptr && mKVCacheCLManager->hostPtrPagedEnabled()) ? 1 : 0,
                      directHostPages ? 1 : 0,
                      pageSize);
            logicalCursor += segment.token_count;
            segmentIndex++;
        }

        queue.flush();

        if (logicalCursor != basePast + segmentTotalTokens) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention copied prefix length %d but expected %d\n",
                      logicalCursor - basePast, segmentTotalTokens);
            return INVALID_VALUE;
        }

        mKVCacheCLManager->setPagedState(basePast + segmentTotalTokens, requiredPhysicalTotal, pageSize,
                                         ropeDimForRequest > 0 ? ropeDimForRequest : headDim,
                                         ropeThetaForRequest);
        mKVCacheCLManager->setPastKvLength(basePast + segmentTotalTokens);
        if (layerIndex == 0) {
            MNN_PRINT("[OpenCLPrefixAttention] block_table base_past=%d prefix_tokens=%d prompt_tokens=%d required_total=%d physical_total=%d page_size=%d logical_blocks=%d physical_blocks=%d prefix_blocks=%d prompt_blocks=%d prompt_physical_start=%d device_prefetch=%d\n",
                      basePast, segmentTotalTokens, promptLen, requiredTotal, requiredPhysicalTotal,
                      pageSize,
                      UP_DIV(requiredTotal, pageSize),
                      UP_DIV(requiredPhysicalTotal, pageSize),
                      UP_DIV(segmentTotalTokens, pageSize),
                      UP_DIV(promptLen, pageSize),
                      promptPhysicalStart,
                      devicePrefetch ? 1 : 0);
        }
        if (readyEvent != nullptr && hasLastEvent) {
            *readyEvent = lastEvent;
        }
        appendKvSeqLen = promptLen;
        summary.layerIndex = layerIndex;
        summary.basePast = basePast;
        summary.segmentTotalTokens = segmentTotalTokens;
        summary.promptTokens = promptLen;
        summary.pastAfterPrefix = basePast + segmentTotalTokens;
        return NO_ERROR;
    }

    ErrorCode runDevicePrefetchOnQueue(const OpenCLPrefixInputShape& inputShape, cl::CommandQueue& queue,
                                       uint64_t requestId,
                                       const std::shared_ptr<SharedPrefetchRecord>& sharedRecord) {
        double start = nowMs();
        cl::Event event;
        int appendKvSeqLen = 0;
        PrefixPrepareSummary summary;
        auto err = preparePrefixKVOnQueue(inputShape, queue, false, true,
                                          appendKvSeqLen, summary, &event);
        {
            std::lock_guard<std::mutex> lock(mDevicePrefetchMutex);
            if (mDevicePrefetchRequestId == requestId) {
                mDevicePrefetchStatus = err;
                mDevicePrefetchReadyEvent = event;
                mDevicePrefetchHasEvent = event() != nullptr;
                mDevicePrefetchAppendKvSeqLen = appendKvSeqLen;
                mDevicePrefetchSummary = summary;
                mDevicePrefetchPending = false;
                mDevicePrefetchReady = true;
            }
        }
        mDevicePrefetchCondition.notify_all();

        if (sharedRecord != nullptr) {
            std::lock_guard<std::mutex> lock(sharedRecord->mutex);
            sharedRecord->event = event;
            sharedRecord->hasEvent = event() != nullptr;
            sharedRecord->status = err;
            sharedRecord->appendKvSeqLen = appendKvSeqLen;
            sharedRecord->summary = summary;
            sharedRecord->pending = false;
            sharedRecord->ready = true;
            sharedRecord->condition.notify_all();
        }
        if (err != NO_ERROR || summary.layerIndex < 0) {
            return err;
        }
        MNN_PRINT("[OpenCLPrefixAttentionPrefetch] layer=%d request_id=%llu submit_ms=%.3f prefix_tokens=%d prompt_tokens=%d manager=%p paged_active=%d manager_past=%d\n",
                  summary.layerIndex,
                  static_cast<unsigned long long>(requestId),
                  nowMs() - start,
                  summary.segmentTotalTokens,
                  summary.promptTokens,
                  mKVCacheCLManager.get(),
                  (mKVCacheCLManager != nullptr && mKVCacheCLManager->pagedActive()) ? 1 : 0,
                  mKVCacheCLManager != nullptr ? mKVCacheCLManager->pastKvLength() : 0);
        return NO_ERROR;
    }

    ErrorCode enqueueDevicePrefetchAsync(const OpenCLPrefixInputShape& inputShape, uint64_t requestId,
                                         const std::shared_ptr<SharedPrefetchRecord>& sharedRecord,
                                         const std::shared_ptr<OpenCLPrefixPrefetchThreadPool>& pool) {
        {
            std::lock_guard<std::mutex> lock(mDevicePrefetchMutex);
            if (mDevicePrefetchRequestId == requestId &&
                (mDevicePrefetchPending || mDevicePrefetchReady)) {
                return NO_ERROR;
            }
            mDevicePrefetchRequestId = requestId;
            mDevicePrefetchStatus = NO_ERROR;
            mDevicePrefetchReadyEvent = cl::Event();
            mDevicePrefetchHasEvent = false;
            mDevicePrefetchAppendKvSeqLen = 0;
            mDevicePrefetchSummary = PrefixPrepareSummary();
            mDevicePrefetchPending = true;
            mDevicePrefetchReady = false;
        }
        if (sharedRecord != nullptr) {
            std::lock_guard<std::mutex> lock(sharedRecord->mutex);
            sharedRecord->pending = true;
            sharedRecord->ready = false;
        }
        if (pool == nullptr || !pool->post([this, inputShape, requestId, sharedRecord](cl::CommandQueue& queue) {
                runDevicePrefetchOnQueue(inputShape, queue, requestId, sharedRecord);
            })) {
            {
                std::lock_guard<std::mutex> lock(mDevicePrefetchMutex);
                if (mDevicePrefetchRequestId == requestId) {
                    mDevicePrefetchStatus = INVALID_VALUE;
                    mDevicePrefetchPending = false;
                    mDevicePrefetchReady = true;
                }
            }
            mDevicePrefetchCondition.notify_all();
            if (sharedRecord != nullptr) {
                std::lock_guard<std::mutex> lock(sharedRecord->mutex);
                sharedRecord->status = INVALID_VALUE;
                sharedRecord->pending = false;
                sharedRecord->ready = true;
                sharedRecord->condition.notify_all();
            }
            return INVALID_VALUE;
        }
        return NO_ERROR;
    }

    ErrorCode consumeDevicePrefetch(bool& hit, int& appendKvSeqLen) {
        hit = false;
        if (mMeta == nullptr || mDevicePrefetchRequestId != mMeta->prefix_request_id ||
            mDevicePrefetchRequestId == 0) {
            if (!loadSharedDevicePrefetchRecord()) {
                return NO_ERROR;
            }
        }
        {
            std::unique_lock<std::mutex> lock(mDevicePrefetchMutex);
            while (mDevicePrefetchPending) {
                mDevicePrefetchCondition.wait(lock);
            }
        }
        if (mDevicePrefetchStatus != NO_ERROR) {
            return mDevicePrefetchStatus;
        }
        if (!mDevicePrefetchHasEvent || mDevicePrefetchReadyEvent() == nullptr) {
            return NO_ERROR;
        }
        double waitStart = nowMs();
        cl_int res = enqueueWaitForEvent(mOpenCLBackend->getOpenCLRuntime()->defaultCommandQueue(),
                                         mDevicePrefetchReadyEvent);
        double waitMs = nowMs() - waitStart;
        if (res != CL_SUCCESS) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention failed to enqueue device_prefetch event wait: %d\n", res);
            return INVALID_VALUE;
        }
        mKVCacheCLManager->setPastKvLength(mDevicePrefetchSummary.pastAfterPrefix);
        appendKvSeqLen = mDevicePrefetchAppendKvSeqLen;
        hit = true;
        MNN_PRINT("[OpenCLPrefixAttention] layer=%d request_id=%llu device_prefetch=1 device_prefetch_hit=1 attention_wait_ms=%.3f queue_event_wait=1 base_past=%d prefix_tokens=%d prompt_tokens=%d\n",
                  mDevicePrefetchSummary.layerIndex,
                  static_cast<unsigned long long>(mDevicePrefetchRequestId),
                  waitMs,
                  mDevicePrefetchSummary.basePast,
                  mDevicePrefetchSummary.segmentTotalTokens,
                  mDevicePrefetchSummary.promptTokens);
        return NO_ERROR;
    }

    bool loadSharedDevicePrefetchRecord() {
        if (mMeta == nullptr || mMeta->prefix_request_id == 0 || mLayerIndex < 0) {
            return false;
        }
        std::shared_ptr<SharedPrefetchRecord> record;
        {
            std::lock_guard<std::mutex> lock(registryMutex());
            auto metaIter = sharedPrefetchRecords().find(mMeta);
            if (metaIter == sharedPrefetchRecords().end()) {
                return false;
            }
            auto requestIter = metaIter->second.find(mMeta->prefix_request_id);
            if (requestIter == metaIter->second.end()) {
                return false;
            }
            auto layerIter = requestIter->second.find(mLayerIndex);
            if (layerIter == requestIter->second.end()) {
                return false;
            }
            record = layerIter->second;
        }
        if (record == nullptr) {
            return false;
        }
        {
            std::unique_lock<std::mutex> lock(record->mutex);
            while (record->pending) {
                record->condition.wait(lock);
            }
            if (!record->ready) {
                return false;
            }
            mDevicePrefetchRequestId = mMeta->prefix_request_id;
            mDevicePrefetchStatus = record->status;
            mDevicePrefetchReadyEvent = record->event;
            mDevicePrefetchHasEvent = record->hasEvent;
            mDevicePrefetchAppendKvSeqLen = record->appendKvSeqLen;
            mDevicePrefetchSummary = record->summary;
            mDevicePrefetchPending = false;
            mDevicePrefetchReady = true;
        }
        return true;
    }

    static bool submitDevicePrefetchForMeta(KVMeta* meta, const OpenCLPrefixInputShape* fallbackShape) {
        if (meta == nullptr || !meta->prefix_device_prefetch || meta->prefix_request_id == 0) {
            return false;
        }

        std::vector<OpenCLPrefixAttentionBufExecution*> targets;
        {
            std::lock_guard<std::mutex> lock(registryMutex());
            auto& submitted = submittedRequests();
            auto submittedIter = submitted.find(meta);
            if (submittedIter != submitted.end() && submittedIter->second == meta->prefix_request_id) {
                return true;
            }

            auto registryIter = registry().find(meta);
            if (registryIter != registry().end()) {
                std::unordered_map<int, OpenCLPrefixAttentionBufExecution*> byLayer;
                const bool needExistingPagedState = (meta->previous != meta->remove);
                auto preferCandidate = [needExistingPagedState](OpenCLPrefixAttentionBufExecution* current,
                                                                 OpenCLPrefixAttentionBufExecution* candidate) {
                    if (current == nullptr) {
                        return candidate;
                    }
                    if (candidate == nullptr) {
                        return current;
                    }
                    if (needExistingPagedState) {
                        const bool currentPaged = current->mKVCacheCLManager != nullptr &&
                                                  current->mKVCacheCLManager->pagedActive();
                        const bool candidatePaged = candidate->mKVCacheCLManager != nullptr &&
                                                    candidate->mKVCacheCLManager->pagedActive();
                        if (currentPaged != candidatePaged) {
                            return candidatePaged ? candidate : current;
                        }
                        const int currentPast = current->mKVCacheCLManager != nullptr ?
                            current->mKVCacheCLManager->pastKvLength() : 0;
                        const int candidatePast = candidate->mKVCacheCLManager != nullptr ?
                            candidate->mKVCacheCLManager->pastKvLength() : 0;
                        if (currentPast != candidatePast) {
                            return candidatePast > currentPast ? candidate : current;
                        }
                    }
                    return candidate;
                };
                for (auto* exe : registryIter->second) {
                    if (exe == nullptr || exe->mMeta != meta || exe->mLayerIndex < 0 ||
                        exe->mKVCacheCLManager == nullptr) {
                        continue;
                    }
                    auto iter = byLayer.find(exe->mLayerIndex);
                    byLayer[exe->mLayerIndex] = preferCandidate(
                        iter == byLayer.end() ? nullptr : iter->second, exe);
                }
                targets.reserve(byLayer.size());
                for (auto& iter : byLayer) {
                    targets.emplace_back(iter.second);
                }
            }
        }
        if (targets.empty()) {
            return false;
        }
        std::sort(targets.begin(), targets.end(), [](const OpenCLPrefixAttentionBufExecution* left,
                                                     const OpenCLPrefixAttentionBufExecution* right) {
            return left->mLayerIndex < right->mLayerIndex;
        });
        openCLPrefixProfileExpect(meta->prefix_request_id,
                                  meta->layer_nums > 0 ? meta->layer_nums : static_cast<int>(targets.size()));

        auto* first = targets.front();
        auto pool = first != nullptr && first->mOpenCLBackend != nullptr ?
            OpenCLPrefixPrefetchThreadPool::get(first->mOpenCLBackend->getOpenCLRuntime()) : nullptr;
        if (pool == nullptr || !pool->valid()) {
            MNN_PRINT("[OpenCLPrefixAttentionPrefetch] request_id=%llu unavailable_thread_pool=1 fallback_sync=1\n",
                      static_cast<unsigned long long>(meta->prefix_request_id));
            return false;
        }
        for (auto* target : targets) {
            auto inputShape = openCLPrefixShapeForRequest(target->mLastInputShape, fallbackShape, meta);
            if (!openCLPrefixShapeValid(inputShape)) {
                MNN_PRINT("[OpenCLPrefixAttentionPrefetch] request_id=%llu missing_input_shape=1 fallback_sync=1 layer=%d\n",
                          static_cast<unsigned long long>(meta->prefix_request_id),
                          target != nullptr ? target->mLayerIndex : -1);
                return false;
            }
        }

        std::unordered_map<int, std::shared_ptr<SharedPrefetchRecord>> recordsByLayer;
        {
            std::lock_guard<std::mutex> lock(registryMutex());
            auto& submitted = submittedRequests();
            auto submittedIter = submitted.find(meta);
            if (submittedIter != submitted.end() && submittedIter->second == meta->prefix_request_id) {
                return true;
            }
            sharedPrefetchRecords()[meta].clear();
            auto& requestRecords = sharedPrefetchRecords()[meta][meta->prefix_request_id];
            for (auto* target : targets) {
                if (target == nullptr) {
                    continue;
                }
                auto record = std::make_shared<SharedPrefetchRecord>();
                record->pending = true;
                requestRecords[target->mLayerIndex] = record;
                recordsByLayer[target->mLayerIndex] = record;
            }
            submitted[meta] = meta->prefix_request_id;
        }

        MNN_PRINT("[OpenCLPrefixAttentionPrefetch] request_id=%llu submit_layers=%zu mode=async workers=%d copy_queues=%d\n",
                  static_cast<unsigned long long>(meta->prefix_request_id),
                  targets.size(),
                  pool->workerCount(),
                  pool->workerCount());
        bool ok = true;
        for (auto* target : targets) {
            auto recordIter = recordsByLayer.find(target->mLayerIndex);
            auto record = recordIter != recordsByLayer.end() ? recordIter->second : nullptr;
            auto inputShape = openCLPrefixShapeForRequest(target->mLastInputShape, fallbackShape, meta);
            if (!openCLPrefixShapeValid(inputShape)) {
                ok = false;
                break;
            }
            auto err = target->enqueueDevicePrefetchAsync(inputShape, meta->prefix_request_id, record, pool);
            ok = (err == NO_ERROR) && ok;
        }
        if (!ok) {
            std::lock_guard<std::mutex> lock(registryMutex());
            submittedRequests().erase(meta);
            sharedPrefetchRecords().erase(meta);
        }
        return ok;
    }

    ErrorCode submitDevicePrefetchForRequestIfNeeded(const std::vector<Tensor*>& inputs) {
        if (mMeta == nullptr || !mMeta->prefix_device_prefetch || mMeta->prefix_request_id == 0) {
            return NO_ERROR;
        }
        if (mLayerIndex != 0) {
            return NO_ERROR;
        }
        auto inputShape = openCLPrefixShapeForRequest(openCLPrefixShapeFromInputs(inputs), nullptr, mMeta);
        (void)submitDevicePrefetchForMeta(mMeta, &inputShape);
        return NO_ERROR;
    }

    void registerForDevicePrefetch() {
        if (mRegisteredForDevicePrefetch || mMeta == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(registryMutex());
        auto& list = registry()[mMeta];
        if (std::find(list.begin(), list.end(), this) == list.end()) {
            list.emplace_back(this);
        }
        mRegisteredForDevicePrefetch = true;
    }

    void waitDevicePrefetchTaskComplete() {
        std::unique_lock<std::mutex> lock(mDevicePrefetchMutex);
        while (mDevicePrefetchPending) {
            mDevicePrefetchCondition.wait(lock);
        }
    }

    void unregisterForDevicePrefetch() {
        if (!mRegisteredForDevicePrefetch || mMeta == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(registryMutex());
        auto iter = registry().find(mMeta);
        if (iter != registry().end()) {
            auto& list = iter->second;
            list.erase(std::remove(list.begin(), list.end(), this), list.end());
            if (list.empty()) {
                registry().erase(iter);
                submittedRequests().erase(mMeta);
                sharedPrefetchRecords().erase(mMeta);
            }
        }
        mRegisteredForDevicePrefetch = false;
    }

    static std::mutex& registryMutex() {
        static std::mutex mutex;
        return mutex;
    }

    static std::unordered_map<KVMeta*, std::vector<OpenCLPrefixAttentionBufExecution*>>& registry() {
        static std::unordered_map<KVMeta*, std::vector<OpenCLPrefixAttentionBufExecution*>> value;
        return value;
    }

    static std::unordered_map<KVMeta*, uint64_t>& submittedRequests() {
        static std::unordered_map<KVMeta*, uint64_t> value;
        return value;
    }

    static std::unordered_map<KVMeta*, std::unordered_map<uint64_t, std::unordered_map<int, std::shared_ptr<SharedPrefetchRecord>>>>& sharedPrefetchRecords() {
        static std::unordered_map<KVMeta*, std::unordered_map<uint64_t, std::unordered_map<int, std::shared_ptr<SharedPrefetchRecord>>>> value;
        return value;
    }

    int mLayerIndex = -1;
    RuntimeKVBlockTablePlan mPagedKVPlan;
    std::vector<std::shared_ptr<std::vector<uint8_t>>> mPendingHostUploads;
    OpenCLPrefixInputShape mLastInputShape;
    bool mRegisteredForDevicePrefetch = false;
    std::mutex mDevicePrefetchMutex;
    std::condition_variable mDevicePrefetchCondition;
    bool mDevicePrefetchPending = false;
    bool mDevicePrefetchReady = false;
    uint64_t mDevicePrefetchRequestId = 0;
    cl::Event mDevicePrefetchReadyEvent;
    bool mDevicePrefetchHasEvent = false;
    ErrorCode mDevicePrefetchStatus = NO_ERROR;
    int mDevicePrefetchAppendKvSeqLen = 0;
    PrefixPrepareSummary mDevicePrefetchSummary;
    bool mComputeProfileActive = false;
    uint64_t mComputeProfileRequestId = 0;
    int mComputeProfileLayer = -1;
    uint64_t mComputeProfileQueueId = 0;
    cl::Event mComputeProfileStartEvent;
    cl::Event mComputeProfileEndEvent;
    bool mResizeBasePastValid = false;
    int mResizeBasePast = 0;
    int mResizePrefixLength = 0;
    uint64_t mResizeRequestId = 0;
};

class OpenCLPrefixAttentionBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        bool kvCache = true;
        int layerIndex = -1;
        auto param = op->main_as_AttentionParam();
        if (param != nullptr) {
            kvCache = param->kv_cache();
            layerIndex = param->layer_index();
        }
        OPENCL_CREATOR_CHECK(new OpenCLPrefixAttentionBufExecution(op, backend, kvCache, layerIndex));
    }
};

REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(OpenCLPrefixAttentionBufCreator, OpType_PrefixAttention, BUFFER);

namespace {
struct OpenCLPrefixDevicePrefetchSubmitterRegister {
    OpenCLPrefixDevicePrefetchSubmitterRegister() {
        registerPrefixDevicePrefetchSubmitter(&OpenCLPrefixAttentionBufExecution::submitAsyncDevicePrefetch);
    }
};

static OpenCLPrefixDevicePrefetchSubmitterRegister __init_prefix_device_prefetch_submitter;
} // namespace

} // namespace OpenCL
} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
