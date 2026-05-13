#include "AttentionExecution.hpp"
#include "core/MNNFileUtils.h"
#include "core/PagedKVCachePlan.hpp"
#include "core/PrefixCachePath.hpp"
#include "core/PrefixKVHostCache.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define MNN_CUDA_PREFIX_NVTX_ENABLED 1
#endif
#endif

#ifndef MNN_CUDA_PREFIX_NVTX_ENABLED
#define MNN_CUDA_PREFIX_NVTX_ENABLED 0
#endif

namespace MNN {
namespace CUDA {

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace {

constexpr int kCudaPagedPrefixPageSize = 64;
constexpr const char* kCudaPagedPrefixLayout = "cuda-paged-canonical-no-rope-v1";

template <typename T = void>
static inline T* getTensorDevicePtr(const Tensor* tensor) {
    if (!tensor || tensor->deviceId() == 0) {
        return nullptr;
    }
    return reinterpret_cast<T*>(tensor->deviceId());
}

template <typename T>
__device__ static T castPrefixScalar(float value) {
    return static_cast<T>(value);
}

template <>
__device__ __half castPrefixScalar<__half>(float value) {
    return __float2half(value);
}

template <typename T>
__global__ void applyCudaPagedPrefixRopeKernel(T* keyPages, int physicalStart, int logicalStart,
                                               int pageSize, int kvHeads, int headDim,
                                               int srcLen, int ropeDim, float ropeTheta) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int half = ropeDim / 2;
    int total = kvHeads * srcLen * half;
    if (index >= total) {
        return;
    }

    int freqDim = index % half;
    int token = (index / half) % srcLen;
    int head = index / (half * srcLen);
    int logicalSeq = logicalStart + token;
    int physicalSlot = physicalStart + token;
    int physicalBlock = physicalSlot / pageSize;
    int pageOffset = physicalSlot - physicalBlock * pageSize;
    size_t base = (((static_cast<size_t>(physicalBlock) * pageSize + pageOffset) *
                    kvHeads + head) * headDim);
    int leftDim = freqDim;
    int rightDim = freqDim + half;
    float left = static_cast<float>(keyPages[base + leftDim]);
    float right = static_cast<float>(keyPages[base + rightDim]);
    float invFreq = 1.0f / powf(ropeTheta, static_cast<float>(2 * freqDim) / static_cast<float>(ropeDim));
    float angle = static_cast<float>(logicalSeq) * invFreq;
    float cosValue = cosf(angle);
    float sinValue = sinf(angle);
    keyPages[base + leftDim] = castPrefixScalar<T>(left * cosValue - right * sinValue);
    keyPages[base + rightDim] = castPrefixScalar<T>(right * cosValue + left * sinValue);
}

template <typename T>
__global__ void importContiguousKVToPagedKernel(const T* pastKey, const T* pastValue,
                                                T* keyPages, T* valuePages,
                                                const int* blockTable,
                                                int batch, int tokenCount, int kvHeads,
                                                int headDim, int maxLen, int pageSize) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int token = blockIdx.y * blockDim.y + threadIdx.y;
    int bh = blockIdx.z * blockDim.z + threadIdx.z;
    if (d >= headDim || token >= tokenCount || bh >= batch * kvHeads) {
        return;
    }
    int b = bh / kvHeads;
    int h = bh % kvHeads;
    int physicalSlot = blockTable[token];
    int physicalBlock = physicalSlot / pageSize;
    int pageOffset = physicalSlot - physicalBlock * pageSize;

    size_t keySrc = ((static_cast<size_t>(token) * batch + b) * kvHeads + h) * headDim + d;
    size_t valueSrc = ((static_cast<size_t>(b) * kvHeads + h) * maxLen + token) * headDim + d;
    size_t keyDst = (((static_cast<size_t>(physicalBlock) * pageSize + pageOffset) *
                      batch + b) * kvHeads + h) * headDim + d;
    size_t valueDst = ((((static_cast<size_t>(physicalBlock) * batch + b) *
                         kvHeads + h) * pageSize + pageOffset) * headDim + d);
    keyPages[keyDst] = pastKey[keySrc];
    valuePages[valueDst] = pastValue[valueSrc];
}

template <typename T>
__global__ void loadCudaPagedPrefixKVKernel(const T* rawKeyPages, const T* rawValuePages,
                                            T* keyPages, T* valuePages, const int* blockTable,
                                            int pageSize, int kvHeads, int headDim,
                                            int srcLen, int dstStart,
                                            int ropeDim, float ropeTheta) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = kvHeads * srcLen * headDim;
    if (index >= total) {
        return;
    }

    int dim = index % headDim;
    int token = (index / headDim) % srcLen;
    int head = index / (headDim * srcLen);
    int dstSeq = dstStart + token;

    int srcBlock = token / pageSize;
    int srcPageOffset = token - srcBlock * pageSize;
    size_t srcKeyIndex = ((static_cast<size_t>(srcBlock) * pageSize + srcPageOffset) * kvHeads + head) *
                         headDim + dim;
    size_t srcValueIndex = ((static_cast<size_t>(srcBlock) * kvHeads + head) * pageSize + srcPageOffset) *
                           headDim + dim;
    float keyValue = static_cast<float>(rawKeyPages[srcKeyIndex]);
    float value = static_cast<float>(rawValuePages[srcValueIndex]);

    if (dim < ropeDim) {
        int half = ropeDim / 2;
        int pairDim = dim < half ? dim + half : dim - half;
        int freqDim = dim < half ? dim : dim - half;
        size_t pairIndex = ((static_cast<size_t>(srcBlock) * pageSize + srcPageOffset) * kvHeads + head) *
                           headDim + pairDim;
        float pairValue = static_cast<float>(rawKeyPages[pairIndex]);
        float left = dim < half ? keyValue : pairValue;
        float right = dim < half ? pairValue : keyValue;
        float invFreq = 1.0f / powf(ropeTheta, static_cast<float>(2 * freqDim) / static_cast<float>(ropeDim));
        float angle = static_cast<float>(dstSeq) * invFreq;
        float cosValue = cosf(angle);
        float sinValue = sinf(angle);
        keyValue = dim < half ? left * cosValue - right * sinValue : right * cosValue + left * sinValue;
    }

    int logicalBlock = dstSeq / pageSize;
    int pageOffset = dstSeq - logicalBlock * pageSize;
    int physicalBlock = blockTable[logicalBlock];
    keyPages[((static_cast<size_t>(physicalBlock) * pageSize + pageOffset) * kvHeads + head) *
             headDim + dim] = castPrefixScalar<T>(keyValue);
    valuePages[((static_cast<size_t>(physicalBlock) * kvHeads + head) * pageSize + pageOffset) *
               headDim + dim] = castPrefixScalar<T>(value);
}

static size_t cudaPagedFileBytes(int tokenCount, int pageSize, int kvHeads, int headDim, int bytes) {
    int blocks = UP_DIV(tokenCount, pageSize);
    return static_cast<size_t>(blocks) * pageSize * kvHeads * headDim * bytes;
}

static int alignToPage(int value, int pageSize) {
    return UP_DIV(value, pageSize) * pageSize;
}

static double nowMs() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

class CudaPrefixNvtxRange {
public:
    explicit CudaPrefixNvtxRange(const std::string& name) {
#if MNN_CUDA_PREFIX_NVTX_ENABLED
        nvtxRangePushA(name.c_str());
#endif
    }

    ~CudaPrefixNvtxRange() {
#if MNN_CUDA_PREFIX_NVTX_ENABLED
        nvtxRangePop();
#endif
    }
};

static std::string cudaPrefixNvtxName(const char* name, int layerIndex, uint64_t requestId = 0,
                                      size_t bytes = 0, const void* stream = nullptr) {
    std::ostringstream os;
    os << name;
    if (layerIndex >= 0) {
        os << " layer=" << layerIndex;
    }
    if (requestId != 0) {
        os << " request=" << requestId;
    }
    if (bytes != 0) {
        os << " bytes=" << bytes;
    }
    if (stream != nullptr) {
        os << " stream=" << stream;
    }
    return os.str();
}

static bool cudaPrefixEnvFlag(const char* name, bool defaultValue = false) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    if (::strcmp(value, "1") == 0 || ::strcmp(value, "true") == 0 ||
        ::strcmp(value, "TRUE") == 0 || ::strcmp(value, "on") == 0 ||
        ::strcmp(value, "ON") == 0 || ::strcmp(value, "yes") == 0 ||
        ::strcmp(value, "YES") == 0) {
        return true;
    }
    if (::strcmp(value, "0") == 0 || ::strcmp(value, "false") == 0 ||
        ::strcmp(value, "FALSE") == 0 || ::strcmp(value, "off") == 0 ||
        ::strcmp(value, "OFF") == 0 || ::strcmp(value, "no") == 0 ||
        ::strcmp(value, "NO") == 0) {
        return false;
    }
    return defaultValue;
}

static bool cudaPrefixProfileH2DEnabled() {
    static bool enabled = cudaPrefixEnvFlag("MNN_CUDA_PREFIX_PROFILE_H2D", false);
    return enabled;
}

static bool cudaPrefixProfileVerboseEnabled() {
    static bool enabled = cudaPrefixEnvFlag("MNN_CUDA_PREFIX_PROFILE_VERBOSE", false);
    return enabled;
}

static int cudaPrefixProfilePairLimit() {
    static int limit = []() {
        const char* value = std::getenv("MNN_CUDA_PREFIX_PROFILE_PAIR_LIMIT");
        if (value == nullptr || value[0] == '\0') {
            return 20;
        }
        return std::max(0, std::atoi(value));
    }();
    return limit;
}

static const char* cudaPrefixMemoryTypeName(cudaMemoryType type) {
    switch (type) {
        case cudaMemoryTypeUnregistered:
            return "unregistered";
        case cudaMemoryTypeHost:
            return "host";
        case cudaMemoryTypeDevice:
            return "device";
        case cudaMemoryTypeManaged:
            return "managed";
        default:
            return "unknown";
    }
}

static std::string cudaPrefixPointerAttributesSummary(const void* ptr) {
    if (ptr == nullptr) {
        return "null";
    }
    cudaPointerAttributes attr;
    ::memset(&attr, 0, sizeof(attr));
    auto err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        std::ostringstream os;
        os << "error=" << cudaGetErrorString(err);
        return os.str();
    }
    std::ostringstream os;
    os << "type=" << cudaPrefixMemoryTypeName(attr.type)
       << " device=" << attr.device
       << " device_ptr=" << attr.devicePointer
       << " host_ptr=" << attr.hostPointer;
    return os.str();
}

static unsigned long long cudaPrefixStreamId(cudaStream_t stream) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    unsigned long long streamId = 0;
    if (stream != nullptr && cudaStreamGetId(stream, &streamId) == cudaSuccess) {
        return streamId;
    }
#endif
    return 0;
}

struct CudaPrefixH2DProfileEvents {
    ~CudaPrefixH2DProfileEvents() {
        if (startEvent != nullptr) {
            cudaEventDestroy(startEvent);
        }
        if (keyEndEvent != nullptr) {
            cudaEventDestroy(keyEndEvent);
        }
        if (valueEndEvent != nullptr) {
            cudaEventDestroy(valueEndEvent);
        }
    }

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t keyEndEvent = nullptr;
    cudaEvent_t valueEndEvent = nullptr;
};

struct CudaPrefixComputeProfileEvents {
    ~CudaPrefixComputeProfileEvents() {
        if (startEvent != nullptr) {
            cudaEventDestroy(startEvent);
        }
        if (endEvent != nullptr) {
            cudaEventDestroy(endEvent);
        }
    }

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t endEvent = nullptr;
};

struct CudaPrefixSegmentPlacement {
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
    cudaEvent_t readyEvent = nullptr;
};

struct CudaPinnedPrefixKVEntry {
    ~CudaPinnedPrefixKVEntry() {
        if (pinned != nullptr) {
            cudaFreeHost(pinned);
        }
    }

    const void* data() const {
        if (pinned != nullptr) {
            return pinned;
        }
        return fallback != nullptr ? fallback->data() : nullptr;
    }

    void* pinned = nullptr;
    std::shared_ptr<std::vector<uint8_t>> fallback;
    bool ok = false;
    bool isPinned = false;
    bool sourceHostCacheHit = false;
    double prefetchWaitMs = 0.0;
    double diskReadMs = 0.0;
    double pinCopyMs = 0.0;
    size_t fileSize = 0;
    std::string error;
};

struct CudaPinnedPrefixKVRead {
    std::shared_ptr<CudaPinnedPrefixKVEntry> entry;
    bool ok = false;
    bool pinnedCacheHit = false;
    bool hostCacheHit = false;
    bool isPinned = false;
    double prefetchWaitMs = 0.0;
    double diskReadMs = 0.0;
    double pinCopyMs = 0.0;
    size_t fileSize = 0;
    std::string error;

    const void* data() const {
        return entry != nullptr ? entry->data() : nullptr;
    }
};

int cudaPrefixPrefetchThreadCountFromEnv() {
    const char* value = std::getenv("MNN_CUDA_PREFIX_PREFETCH_THREADS");
    if (value == nullptr || value[0] == '\0') {
        return 4;
    }
    int parsed = std::atoi(value);
    if (parsed == 4 || parsed == 8 || parsed == 16) {
        return parsed;
    }
    MNN_PRINT("[CUDAPrefixAttentionPrefetch] invalid MNN_CUDA_PREFIX_PREFETCH_THREADS=%s, fallback=4\n", value);
    return 4;
}

class CudaPrefixPrefetchThreadPool {
public:
    using Task = std::function<void(cudaStream_t)>;

    static CudaPrefixPrefetchThreadPool& get() {
        static CudaPrefixPrefetchThreadPool pool(cudaPrefixPrefetchThreadCountFromEnv());
        return pool;
    }

    bool post(Task task) {
        if (!task) {
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

private:
    explicit CudaPrefixPrefetchThreadPool(int workerCount) : mWorkerCount(workerCount) {
        int device = 0;
        cudaGetDevice(&device);
        mStreams.resize(mWorkerCount, nullptr);
        for (int i = 0; i < mWorkerCount; ++i) {
            auto err = cudaStreamCreateWithFlags(&mStreams[i], cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                MNN_ERROR("[Error]: CUDA PrefixAttention failed to create prefetch stream %d: %s\n",
                          i, cudaGetErrorString(err));
                mStreams[i] = nullptr;
            }
        }
        for (int i = 0; i < mWorkerCount; ++i) {
            mWorkers.emplace_back([this, i, device]() {
                cudaSetDevice(device);
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
                    task(mStreams[i]);
                }
            });
        }
        MNN_PRINT("[CUDAPrefixAttentionPrefetch] thread_pool_workers=%d copy_streams=%d\n",
                  mWorkerCount, mWorkerCount);
    }

    ~CudaPrefixPrefetchThreadPool() {
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
        for (auto stream : mStreams) {
            if (stream != nullptr) {
                cudaStreamDestroy(stream);
            }
        }
    }

    int mWorkerCount = 4;
    std::vector<cudaStream_t> mStreams;
    std::vector<std::thread> mWorkers;
    std::queue<Task> mTasks;
    std::mutex mMutex;
    std::condition_variable mCondition;
    bool mStop = false;
};

class CudaPinnedPrefixKVHostCache {
public:
    static CudaPinnedPrefixKVHostCache& get() {
        static CudaPinnedPrefixKVHostCache cache;
        return cache;
    }

    CudaPinnedPrefixKVRead readFile(const std::string& path) {
        CudaPinnedPrefixKVRead result;
        std::string identity;
        size_t fileSize = 0;
        if (!MNN::detail::prefixKVFileIdentity(path, identity, &fileSize)) {
            result.error = "prefix KV file does not exist: " + path;
            return result;
        }
        result.fileSize = fileSize;

        std::shared_future<std::shared_ptr<CudaPinnedPrefixKVEntry>> future;
        std::shared_ptr<std::promise<std::shared_ptr<CudaPinnedPrefixKVEntry>>> promise;
        {
            std::lock_guard<std::mutex> lock(mMutex);
            auto iter = mEntries.find(identity);
            if (iter != mEntries.end()) {
                future = iter->second;
                result.pinnedCacheHit = future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
            } else {
                promise = std::make_shared<std::promise<std::shared_ptr<CudaPinnedPrefixKVEntry>>>();
                future = promise->get_future().share();
                mEntries.emplace(identity, future);
            }
        }
        if (promise != nullptr) {
            promise->set_value(readPinnedNow(path));
        }

        double waitStart = nowMs();
        auto entry = future.get();
        result.prefetchWaitMs = nowMs() - waitStart;
        if (entry != nullptr) {
            result.entry = entry;
            result.ok = entry->ok;
            result.hostCacheHit = result.pinnedCacheHit || entry->sourceHostCacheHit;
            result.isPinned = entry->isPinned;
            if (!result.pinnedCacheHit) {
                result.prefetchWaitMs += entry->prefetchWaitMs;
                result.diskReadMs = entry->diskReadMs;
                result.pinCopyMs = entry->pinCopyMs;
            }
            result.fileSize = entry->fileSize;
            result.error = entry->error;
        }
        if (!result.ok) {
            std::lock_guard<std::mutex> lock(mMutex);
            mEntries.erase(identity);
        }
        return result;
    }

private:
    static std::shared_ptr<CudaPinnedPrefixKVEntry> readPinnedNow(const std::string& path) {
        auto entry = std::make_shared<CudaPinnedPrefixKVEntry>();
        auto source = readPrefixKVHostCacheFile(path);
        entry->sourceHostCacheHit = source.host_cache_hit;
        entry->prefetchWaitMs = source.prefetch_wait_ms;
        entry->diskReadMs = source.disk_read_ms;
        entry->fileSize = source.file_size;
        if (!source.ok || source.bytes == nullptr) {
            entry->error = source.error;
            return entry;
        }

        double pinStart = nowMs();
        void* pinned = nullptr;
        cudaError_t allocCode = cudaSuccess;
        {
            std::lock_guard<std::mutex> lock(pinnedAllocationMutex());
            allocCode = cudaMallocHost(&pinned, source.bytes->size());
        }
        if (allocCode == cudaSuccess && pinned != nullptr) {
            ::memcpy(pinned, source.bytes->data(), source.bytes->size());
            entry->pinned = pinned;
            entry->isPinned = true;
        } else {
            entry->fallback = source.bytes;
            entry->isPinned = false;
            entry->error = std::string("cudaMallocHost failed, fallback to pageable host memory: ") +
                           cudaGetErrorString(allocCode);
        }
        entry->pinCopyMs = nowMs() - pinStart;
        entry->ok = true;
        entry->fileSize = source.bytes->size();
        return entry;
    }

    static std::mutex& pinnedAllocationMutex() {
        static std::mutex mutex;
        return mutex;
    }

    std::mutex mMutex;
    std::unordered_map<std::string, std::shared_future<std::shared_ptr<CudaPinnedPrefixKVEntry>>> mEntries;
};

struct CudaDevicePrefixDocumentRecord {
    int physicalStart = 0;
    int physicalLength = 0;
    int tokenCount = 0;
    int ropeDim = 0;
    float ropeTheta = 10000.0f;
    cudaEvent_t readyEvent = nullptr;
};

struct CudaDevicePrefixDocumentPool {
    std::mutex mutex;
    std::unordered_map<std::string, CudaDevicePrefixDocumentRecord> records;
    int nextPhysicalSlot = 0;
};

static std::mutex& cudaDocumentPoolRegistryMutex() {
    static std::mutex mutex;
    return mutex;
}

static std::unordered_map<const AttentionExecution::SharedCache*, std::shared_ptr<CudaDevicePrefixDocumentPool>>&
cudaDocumentPoolRegistry() {
    static std::unordered_map<const AttentionExecution::SharedCache*, std::shared_ptr<CudaDevicePrefixDocumentPool>> value;
    return value;
}

static std::shared_ptr<CudaDevicePrefixDocumentPool> cudaDocumentPoolForCache(
        const std::shared_ptr<AttentionExecution::SharedCache>& cache) {
    if (cache == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(cudaDocumentPoolRegistryMutex());
    auto& registry = cudaDocumentPoolRegistry();
    auto iter = registry.find(cache.get());
    if (iter != registry.end()) {
        return iter->second;
    }
    auto pool = std::make_shared<CudaDevicePrefixDocumentPool>();
    registry.emplace(cache.get(), pool);
    return pool;
}

static std::string cudaDocumentPageCacheKey(const std::string& keyPath, const std::string& valuePath,
                                            const KVMeta::PrefixSegment& segment,
                                            int layerIndex, int precision, int kvHeads, int headDim,
                                            int expectedRopeDim, float expectedRopeTheta) {
    std::string keyIdentity;
    std::string valueIdentity;
    size_t keySize = 0;
    size_t valueSize = 0;
    if (!MNN::detail::prefixKVFileIdentity(keyPath, keyIdentity, &keySize) ||
        !MNN::detail::prefixKVFileIdentity(valuePath, valueIdentity, &valueSize)) {
        return "";
    }
    return "backend=cuda|layout=" + segment.layout +
           "|cache=" + segment.cache_name +
           "|layer=" + std::to_string(layerIndex) +
           "|dtype=" + segment.dtype +
           "|precision=" + std::to_string(precision) +
           "|kv_heads=" + std::to_string(kvHeads) +
           "|head_dim=" + std::to_string(headDim) +
           "|tokens=" + std::to_string(segment.token_count) +
           "|page_size=" + std::to_string(segment.page_size) +
           "|rope_dim=" + std::to_string(expectedRopeDim) +
           "|rope_theta=" + std::to_string(expectedRopeTheta) +
           "|key=" + keyIdentity +
           "|value=" + valueIdentity +
           "|key_size=" + std::to_string(keySize) +
           "|value_size=" + std::to_string(valueSize);
}

static std::string cudaPrefixRequestProfileKey(const KVMeta* meta, uint64_t requestId) {
    std::ostringstream os;
    os << meta << "#" << requestId;
    return os.str();
}

struct CudaPrefixProfileInterval {
    double startMs = 0.0;
    double endMs = 0.0;
    int layerIndex = -1;
    std::string cacheName;
    unsigned long long streamId = 0;
};

struct CudaPrefixRequestProfileH2D {
    int layerIndex = -1;
    uint64_t requestId = 0;
    std::string cacheName;
    unsigned long long streamId = 0;
    size_t keyBytes = 0;
    size_t valueBytes = 0;
    int physicalStart = 0;
    std::shared_ptr<CudaPrefixH2DProfileEvents> events;
};

struct CudaPrefixRequestProfileCompute {
    int layerIndex = -1;
    uint64_t requestId = 0;
    unsigned long long streamId = 0;
    std::shared_ptr<CudaPrefixComputeProfileEvents> events;
};

struct CudaPrefixRequestProfileState {
    KVMeta* meta = nullptr;
    uint64_t requestId = 0;
    int expectedComputeLayers = 0;
    std::vector<CudaPrefixRequestProfileH2D> h2d;
    std::vector<CudaPrefixRequestProfileCompute> compute;
    bool reporterLaunched = false;
};

static std::mutex& cudaPrefixRequestProfileMutex() {
    static std::mutex mutex;
    return mutex;
}

static std::unordered_map<std::string, std::shared_ptr<CudaPrefixRequestProfileState>>&
cudaPrefixRequestProfileRegistry() {
    static std::unordered_map<std::string, std::shared_ptr<CudaPrefixRequestProfileState>> registry;
    return registry;
}

static double cudaPrefixUnionMs(std::vector<CudaPrefixProfileInterval> intervals) {
    if (intervals.empty()) {
        return 0.0;
    }
    std::sort(intervals.begin(), intervals.end(), [](const auto& left, const auto& right) {
        return left.startMs < right.startMs;
    });
    double total = 0.0;
    double start = intervals[0].startMs;
    double end = intervals[0].endMs;
    for (size_t i = 1; i < intervals.size(); ++i) {
        if (intervals[i].endMs <= intervals[i].startMs) {
            continue;
        }
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

static double cudaPrefixOverlapUnionMs(const std::vector<CudaPrefixProfileInterval>& copyIntervals,
                                       const std::vector<CudaPrefixProfileInterval>& computeIntervals) {
    std::vector<CudaPrefixProfileInterval> overlaps;
    for (const auto& copy : copyIntervals) {
        for (const auto& compute : computeIntervals) {
            double start = std::max(copy.startMs, compute.startMs);
            double end = std::min(copy.endMs, compute.endMs);
            if (end > start) {
                CudaPrefixProfileInterval interval;
                interval.startMs = start;
                interval.endMs = end;
                overlaps.emplace_back(interval);
            }
        }
    }
    return cudaPrefixUnionMs(std::move(overlaps));
}

static void cudaPrefixRequestProfileMaybeLaunchLocked(
        const std::string& key,
        const std::shared_ptr<CudaPrefixRequestProfileState>& state);

static void cudaPrefixRequestProfileExpect(KVMeta* meta, uint64_t requestId, int expectedComputeLayers) {
    if (!cudaPrefixProfileH2DEnabled() || meta == nullptr || requestId == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(cudaPrefixRequestProfileMutex());
    auto& registry = cudaPrefixRequestProfileRegistry();
    auto key = cudaPrefixRequestProfileKey(meta, requestId);
    auto& state = registry[key];
    if (state == nullptr) {
        state = std::make_shared<CudaPrefixRequestProfileState>();
        state->meta = meta;
        state->requestId = requestId;
    }
    state->expectedComputeLayers = std::max(state->expectedComputeLayers, expectedComputeLayers);
    cudaPrefixRequestProfileMaybeLaunchLocked(key, state);
}

static void cudaPrefixRequestProfileRegisterH2D(KVMeta* meta, uint64_t requestId, int layerIndex,
                                                const std::string& cacheName,
                                                unsigned long long streamId,
                                                size_t keyBytes, size_t valueBytes,
                                                int physicalStart,
                                                const std::shared_ptr<CudaPrefixH2DProfileEvents>& events) {
    if (!cudaPrefixProfileH2DEnabled() || meta == nullptr || requestId == 0 || events == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(cudaPrefixRequestProfileMutex());
    auto& registry = cudaPrefixRequestProfileRegistry();
    auto key = cudaPrefixRequestProfileKey(meta, requestId);
    auto& state = registry[key];
    if (state == nullptr) {
        state = std::make_shared<CudaPrefixRequestProfileState>();
        state->meta = meta;
        state->requestId = requestId;
    }
    CudaPrefixRequestProfileH2D record;
    record.layerIndex = layerIndex;
    record.requestId = requestId;
    record.cacheName = cacheName;
    record.streamId = streamId;
    record.keyBytes = keyBytes;
    record.valueBytes = valueBytes;
    record.physicalStart = physicalStart;
    record.events = events;
    state->h2d.emplace_back(std::move(record));
    cudaPrefixRequestProfileMaybeLaunchLocked(key, state);
}

static void cudaPrefixRequestProfileRegisterCompute(
        KVMeta* meta, uint64_t requestId, int layerIndex,
        unsigned long long streamId,
        const std::shared_ptr<CudaPrefixComputeProfileEvents>& events) {
    if (!cudaPrefixProfileH2DEnabled() || meta == nullptr || requestId == 0 || events == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(cudaPrefixRequestProfileMutex());
    auto& registry = cudaPrefixRequestProfileRegistry();
    auto key = cudaPrefixRequestProfileKey(meta, requestId);
    auto& state = registry[key];
    if (state == nullptr) {
        state = std::make_shared<CudaPrefixRequestProfileState>();
        state->meta = meta;
        state->requestId = requestId;
    }
    for (const auto& compute : state->compute) {
        if (compute.layerIndex == layerIndex) {
            cudaPrefixRequestProfileMaybeLaunchLocked(key, state);
            return;
        }
    }
    CudaPrefixRequestProfileCompute record;
    record.layerIndex = layerIndex;
    record.requestId = requestId;
    record.streamId = streamId;
    record.events = events;
    state->compute.emplace_back(std::move(record));
    cudaPrefixRequestProfileMaybeLaunchLocked(key, state);
}

static bool cudaPrefixEventElapsed(cudaEvent_t start, cudaEvent_t end, double& outMs) {
    float ms = 0.0f;
    auto err = cudaEventElapsedTime(&ms, start, end);
    if (err != cudaSuccess) {
        return false;
    }
    outMs = static_cast<double>(ms);
    return true;
}

static void cudaPrefixRequestProfileMaybeLaunchLocked(
        const std::string& key,
        const std::shared_ptr<CudaPrefixRequestProfileState>& state) {
    if (state == nullptr || state->reporterLaunched ||
        state->expectedComputeLayers <= 0 ||
        static_cast<int>(state->compute.size()) < state->expectedComputeLayers) {
        return;
    }
    state->reporterLaunched = true;
    std::thread([key, state]() {
        bool timeout = false;
        bool queryError = false;
        std::vector<CudaPrefixRequestProfileH2D> h2dRecords;
        std::vector<CudaPrefixRequestProfileCompute> computeRecords;
        for (int retry = 0; retry < 5000; ++retry) {
            {
                std::lock_guard<std::mutex> lock(cudaPrefixRequestProfileMutex());
                h2dRecords = state->h2d;
                computeRecords = state->compute;
            }
            bool ready = static_cast<int>(computeRecords.size()) >= state->expectedComputeLayers;
            for (const auto& record : computeRecords) {
                if (record.events == nullptr || record.events->endEvent == nullptr) {
                    ready = false;
                    continue;
                }
                auto query = cudaEventQuery(record.events->endEvent);
                if (query == cudaErrorNotReady) {
                    ready = false;
                } else if (query != cudaSuccess) {
                    queryError = true;
                    MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] request_id=%llu request_overlap compute_query_error=%s layer=%d stream_id=%llu\n",
                              static_cast<unsigned long long>(state->requestId),
                              cudaGetErrorString(query),
                              record.layerIndex,
                              record.streamId);
                    ready = true;
                    break;
                }
            }
            for (const auto& record : h2dRecords) {
                if (record.events == nullptr || record.events->valueEndEvent == nullptr) {
                    ready = false;
                    continue;
                }
                auto query = cudaEventQuery(record.events->valueEndEvent);
                if (query == cudaErrorNotReady) {
                    ready = false;
                } else if (query != cudaSuccess) {
                    queryError = true;
                    MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] request_id=%llu request_overlap h2d_query_error=%s layer=%d cache=%s stream_id=%llu\n",
                              static_cast<unsigned long long>(state->requestId),
                              cudaGetErrorString(query),
                              record.layerIndex,
                              record.cacheName.c_str(),
                              record.streamId);
                    ready = true;
                    break;
                }
            }
            if (ready || queryError) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            if (retry == 4999) {
                timeout = true;
            }
        }

        cudaEvent_t reference = nullptr;
        if (!h2dRecords.empty() && h2dRecords[0].events != nullptr) {
            reference = h2dRecords[0].events->startEvent;
        }
        if (reference == nullptr && !computeRecords.empty() && computeRecords[0].events != nullptr) {
            reference = computeRecords[0].events->startEvent;
        }

        std::vector<CudaPrefixProfileInterval> copyIntervals;
        std::vector<CudaPrefixProfileInterval> computeIntervals;
        double copyTotalMs = 0.0;
        double computeTotalMs = 0.0;
        int copyReady = 0;
        int computeReady = 0;
        if (reference != nullptr && !queryError) {
            for (const auto& record : h2dRecords) {
                if (record.events == nullptr || record.events->startEvent == nullptr ||
                    record.events->valueEndEvent == nullptr ||
                    cudaEventQuery(record.events->valueEndEvent) != cudaSuccess) {
                    continue;
                }
                double startMs = 0.0;
                double endMs = 0.0;
                double durationMs = 0.0;
                if (!cudaPrefixEventElapsed(reference, record.events->startEvent, startMs) ||
                    !cudaPrefixEventElapsed(reference, record.events->valueEndEvent, endMs) ||
                    !cudaPrefixEventElapsed(record.events->startEvent, record.events->valueEndEvent, durationMs)) {
                    continue;
                }
                CudaPrefixProfileInterval interval;
                interval.startMs = startMs;
                interval.endMs = endMs;
                interval.layerIndex = record.layerIndex;
                interval.cacheName = record.cacheName;
                interval.streamId = record.streamId;
                copyIntervals.emplace_back(interval);
                copyTotalMs += durationMs;
                ++copyReady;
            }
            for (const auto& record : computeRecords) {
                if (record.events == nullptr || record.events->startEvent == nullptr ||
                    record.events->endEvent == nullptr ||
                    cudaEventQuery(record.events->endEvent) != cudaSuccess) {
                    continue;
                }
                double startMs = 0.0;
                double endMs = 0.0;
                double durationMs = 0.0;
                if (!cudaPrefixEventElapsed(reference, record.events->startEvent, startMs) ||
                    !cudaPrefixEventElapsed(reference, record.events->endEvent, endMs) ||
                    !cudaPrefixEventElapsed(record.events->startEvent, record.events->endEvent, durationMs)) {
                    continue;
                }
                CudaPrefixProfileInterval interval;
                interval.startMs = startMs;
                interval.endMs = endMs;
                interval.layerIndex = record.layerIndex;
                interval.streamId = record.streamId;
                computeIntervals.emplace_back(interval);
                computeTotalMs += durationMs;
                ++computeReady;
            }
        }

        double copyUnionMs = cudaPrefixUnionMs(copyIntervals);
        double computeUnionMs = cudaPrefixUnionMs(computeIntervals);
        double overlapMs = cudaPrefixOverlapUnionMs(copyIntervals, computeIntervals);
        double overlapRatio = copyUnionMs > 0.0 ? overlapMs / copyUnionMs : 0.0;
        MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] request_id=%llu request_overlap copy_events=%d compute_events=%d expected_layers=%d copy_total_gpu_ms=%.3f compute_total_gpu_ms=%.3f copy_union_ms=%.3f compute_union_ms=%.3f copy_compute_overlap_ms=%.3f overlap_ratio=%.3f timeout=%d query_error=%d\n",
                  static_cast<unsigned long long>(state->requestId),
                  copyReady,
                  computeReady,
                  state->expectedComputeLayers,
                  copyTotalMs,
                  computeTotalMs,
                  copyUnionMs,
                  computeUnionMs,
                  overlapMs,
                  overlapRatio,
                  timeout ? 1 : 0,
                  queryError ? 1 : 0);

        const int pairLimit = cudaPrefixProfilePairLimit();
        if (pairLimit > 0) {
            int printedPairs = 0;
            for (const auto& copy : copyIntervals) {
                for (const auto& compute : computeIntervals) {
                    if (copy.layerIndex == compute.layerIndex) {
                        continue;
                    }
                    double start = std::max(copy.startMs, compute.startMs);
                    double end = std::min(copy.endMs, compute.endMs);
                    double pairOverlap = std::max(0.0, end - start);
                    if (pairOverlap < 0.05) {
                        continue;
                    }
                    MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] request_id=%llu request_overlap_pair copy_layer=%d copy_cache=%s compute_layer=%d overlap_ms=%.3f copy_stream_id=%llu compute_stream_id=%llu\n",
                              static_cast<unsigned long long>(state->requestId),
                              copy.layerIndex,
                              copy.cacheName.c_str(),
                              compute.layerIndex,
                              pairOverlap,
                              copy.streamId,
                              compute.streamId);
                    if (++printedPairs >= pairLimit) {
                        break;
                    }
                }
                if (printedPairs >= pairLimit) {
                    break;
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(cudaPrefixRequestProfileMutex());
            auto& registry = cudaPrefixRequestProfileRegistry();
            auto iter = registry.find(key);
            if (iter != registry.end() && iter->second == state) {
                registry.erase(iter);
            }
        }
    }).detach();
}

} // namespace

class CUDAPrefixAttentionExecution : public AttentionExecution {
public:
    CUDAPrefixAttentionExecution(Backend* backend, bool kvCache, int layerIndex)
        : AttentionExecution(backend, kvCache), mLayerIndex(layerIndex) {
    }

    ~CUDAPrefixAttentionExecution() override {
        unregisterForDevicePrefetch();
        waitDevicePrefetchTaskComplete();
        clearDevicePrefetchEvent();
        clearComputeProfileEvents();
    }

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) override {
        auto code = AttentionExecution::onResize(inputs, outputs);
        if (code == NO_ERROR) {
            registerForDevicePrefetch();
        }
        return code;
    }

    static bool submitAsyncDevicePrefetch(KVMeta* meta) {
        return submitDevicePrefetchForMeta(meta, true);
    }

protected:
    virtual ErrorCode onPrepareKVCacheBeforeAppend(const std::vector<Tensor*>& inputs, cudaStream_t stream,
                                                   bool& prepared, int& appendKvSeqLen) override {
        prepared = false;
        appendKvSeqLen = mNewKvSeqLen;
        if (!mIsKVCacheEnabled || mMeta == nullptr ||
            mMeta->file_flag != KVMeta::PendingReadSegments || mMeta->prefix_segments.empty()) {
            return NO_ERROR;
        }
        if (mBatch != 1) {
            MNN_ERROR("[Error]: CUDA PrefixAttention direct_segments only supports batch=1 now\n");
            return NOT_SUPPORT;
        }
        if (mMeta->prefix_cache_dir.empty()) {
            MNN_ERROR("[Error]: CUDA PrefixAttention missing prefix_cache_dir in KVMeta\n");
            return INVALID_VALUE;
        }
        if ((mMeta->remove > 0 || mMeta->n_reserve > 0) && mMeta->previous != mMeta->remove) {
            MNN_ERROR("[Error]: CUDA PrefixAttention direct_segments append does not support remove/reserve compaction yet\n");
            return NOT_SUPPORT;
        }

        if (mMeta->prefix_device_prefetch && mMeta->prefix_request_id != 0) {
            auto submitErr = submitDevicePrefetchForRequestIfNeeded();
            if (submitErr != NO_ERROR) {
                return submitErr;
            }
            bool prefetchHit = false;
            auto waitErr = consumeDevicePrefetch(stream, prefetchHit, appendKvSeqLen);
            if (waitErr != NO_ERROR) {
                return waitErr;
            }
            if (prefetchHit) {
                appendKvSeqLen = mNewKvSeqLen;
            } else {
                MNN_PRINT("[CUDAPrefixAttention] layer=%d request_id=%llu device_prefetch=1 device_prefetch_hit=0 fallback_sync=1\n",
                          mLayerIndex, static_cast<unsigned long long>(mMeta->prefix_request_id));
            }
        }

        PrefixPrepareSummary summary;
        auto prepareErr = preparePagedPrefixKV(stream, true, false, appendKvSeqLen, summary);
        if (prepareErr != NO_ERROR) {
            return prepareErr;
        }
        prepared = true;
        return NO_ERROR;
    }

    virtual ErrorCode onProfileComputeStreamStart(cudaStream_t stream) override {
        if (!shouldProfilePrefixCompute()) {
            return NO_ERROR;
        }
        clearComputeProfileEvents();
        mComputeProfileRequestId = mMeta->prefix_request_id;
        mComputeProfileLayerIndex = mLayerIndex;
        mComputeProfileStream = stream;
        mComputeProfileStreamId = cudaPrefixStreamId(stream);
        mComputeProfileEvents = std::make_shared<CudaPrefixComputeProfileEvents>();
        auto startErr = cudaEventCreate(&mComputeProfileEvents->startEvent);
        auto endErr = cudaEventCreate(&mComputeProfileEvents->endEvent);
        if (startErr != cudaSuccess || endErr != cudaSuccess) {
            clearComputeProfileEvents();
            MNN_ERROR("[Error]: CUDA PrefixAttention failed to create compute profile events\n");
            return INVALID_VALUE;
        }
        mComputeStartEvent = mComputeProfileEvents->startEvent;
        mComputeEndEvent = mComputeProfileEvents->endEvent;
        auto recordErr = cudaEventRecord(mComputeStartEvent, stream);
        if (recordErr != cudaSuccess) {
            clearComputeProfileEvents();
            return INVALID_VALUE;
        }
        mComputeProfilePending = true;
        mComputeProfileReported = false;
        mComputeProfilePendingReported = false;
        mComputeProfileReporterLaunched = false;
        if (cudaPrefixProfileVerboseEnabled()) {
            MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] layer=%d request_id=%llu compute_profile_start stream=%p stream_id=%llu\n",
                      mComputeProfileLayerIndex,
                      static_cast<unsigned long long>(mComputeProfileRequestId),
                      mComputeProfileStream,
                      mComputeProfileStreamId);
        }
        return NO_ERROR;
    }

    virtual ErrorCode onProfileComputeStreamEnd(cudaStream_t stream) override {
        if (!mComputeProfilePending || mComputeEndEvent == nullptr) {
            return NO_ERROR;
        }
        auto recordErr = cudaEventRecord(mComputeEndEvent, stream);
        if (recordErr != cudaSuccess) {
            clearComputeProfileEvents();
            return INVALID_VALUE;
        }
        if (cudaPrefixProfileVerboseEnabled()) {
            MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] layer=%d request_id=%llu compute_profile_end stream=%p stream_id=%llu\n",
                      mComputeProfileLayerIndex,
                      static_cast<unsigned long long>(mComputeProfileRequestId),
                      mComputeProfileStream,
                      mComputeProfileStreamId);
        }
        cudaPrefixRequestProfileRegisterCompute(mMeta, mComputeProfileRequestId,
                                                mComputeProfileLayerIndex,
                                                mComputeProfileStreamId,
                                                mComputeProfileEvents);
        return NO_ERROR;
    }

    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (dst == nullptr) {
            return true;
        }
        auto exe = new CUDAPrefixAttentionExecution(bn, mIsKVCacheEnabled, mLayerIndex);
        exe->mCache = mCache;
        exe->mMeta = mMeta;
        *dst = exe;
        return true;
    }

private:
    struct PrefixPrepareSummary {
        int layerIndex = -1;
        int basePast = 0;
        int segmentTotalTokens = 0;
        int promptTokens = 0;
        int requiredTotal = 0;
        int requiredPhysicalTotal = 0;
        int promptPhysicalStart = 0;
    };

    struct H2DProfileRecord {
        int layerIndex = -1;
        uint64_t requestId = 0;
        std::string cacheName;
        cudaStream_t stream = nullptr;
        unsigned long long streamId = 0;
        size_t keyBytes = 0;
        size_t valueBytes = 0;
        int physicalStart = 0;
        std::shared_ptr<CudaPrefixH2DProfileEvents> events;
        cudaEvent_t startEvent = nullptr;
        cudaEvent_t keyEndEvent = nullptr;
        cudaEvent_t valueEndEvent = nullptr;
        bool reported = false;
        bool pendingReported = false;
    };

    ErrorCode preparePagedPrefixKV(cudaStream_t stream, bool allowMetaLayerCursor, bool devicePrefetch,
                                   int& appendKvSeqLen, PrefixPrepareSummary& summary) {
        CudaPrefixNvtxRange nvtx(cudaPrefixNvtxName("prefix_runtime_table_prepare",
                                                    mLayerIndex,
                                                    mMeta != nullptr ? mMeta->prefix_request_id : 0,
                                                    0, stream));
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

        appendKvSeqLen = mMeta->prefix_prompt_token_count;
        if (appendKvSeqLen <= 0) {
            appendKvSeqLen = static_cast<int>(mMeta->add);
        }
        if (appendKvSeqLen <= 0) {
            appendKvSeqLen = mNewKvSeqLen;
        }
        if (appendKvSeqLen <= 0 && devicePrefetch) {
            appendKvSeqLen = mMeta->prefix_prompt_token_count;
        }
        if (!buildPrefixRuntimeKVBlockTablePlan(mPagedKVPlan, mMeta, appendKvSeqLen)) {
            return INVALID_VALUE;
        }
        int basePast = 0;
        if (mMeta->previous != mMeta->remove) {
            basePast = mCache != nullptr ? mCache->mPastLength : 0;
            if (basePast <= 0) {
                basePast = static_cast<int>(mMeta->previous - mMeta->remove);
            }
            if (mCache == nullptr || !mCache->mPastKey || !mCache->mPastValue ||
                mCache->mPastKey->deviceId() == 0 || mCache->mPastValue->deviceId() == 0) {
                MNN_ERROR("[Error]: CUDA PrefixAttention direct_segments requires existing system/prompt KV cache for base history\n");
                return INVALID_VALUE;
            }
        }
        const int requiredTotal = basePast + segmentTotalTokens + appendKvSeqLen;
        const int basePhysicalReserved = alignToPage(basePast, kCudaPagedPrefixPageSize);
        const bool importBasePastToPaged = basePast > 0;

        int layerIndex = mLayerIndex;
        if (layerIndex < 0) {
            if (!allowMetaLayerCursor) {
                MNN_ERROR("[Error]: CUDA PrefixAttention device_prefetch requires explicit AttentionParam.layer_index\n");
                return NOT_SUPPORT;
            }
            layerIndex = mMeta->layer_index;
            if (mMeta->layer_nums > 0) {
                mMeta->layer_index = (mMeta->layer_index + 1) % mMeta->layer_nums;
            } else {
                mMeta->layer_index++;
            }
        }

        auto docPool = cudaDocumentPoolForCache(mCache);
        if (docPool == nullptr) {
            return OUT_OF_MEMORY;
        }

        std::vector<CudaPrefixSegmentPlacement> placements;
        placements.reserve(mMeta->prefix_segments.size());
        int logicalCursor = basePast;
        int physicalCursor = std::max(mCache != nullptr ? mCache->mPagedPhysicalLength : 0, basePhysicalReserved);
        {
            std::lock_guard<std::mutex> lock(docPool->mutex);
            physicalCursor = std::max(physicalCursor, docPool->nextPhysicalSlot);
        }
        int ropeDimForRequest = 0;
        float ropeThetaForRequest = 10000.0f;

        for (const auto& segment : mMeta->prefix_segments) {
            if (segment.token_count <= 0) {
                continue;
            }
            if (segment.key_rope_state != KVMeta::KeyRopeCanonicalNoRope ||
                segment.rope_pairing != KVMeta::RopePairingHalf) {
                MNN_ERROR("[Error]: CUDA PrefixAttention requires canonical_no_rope half-split prefix cache: %s\n",
                          segment.cache_name.c_str());
                return NOT_SUPPORT;
            }
            if (segment.backend != "cuda" || segment.layout != kCudaPagedPrefixLayout ||
                segment.page_size != kCudaPagedPrefixPageSize) {
                MNN_ERROR("[Error]: CUDA PrefixAttention requires backend-native paged cache for %s, got backend=%s layout=%s page_size=%d\n",
                          segment.cache_name.c_str(), segment.backend.c_str(), segment.layout.c_str(), segment.page_size);
                return NOT_SUPPORT;
            }
            int ropeDim = segment.rope_dim > 0 ? segment.rope_dim : mHeadDim;
            ropeDim = std::min(ropeDim, mHeadDim);
            if (ropeDim <= 0 || (ropeDim % 2) != 0 || segment.rope_theta <= 0.0f) {
                MNN_ERROR("[Error]: CUDA PrefixAttention invalid RoPE metadata for cache %s\n",
                          segment.cache_name.c_str());
                return INVALID_VALUE;
            }
            if (ropeDimForRequest == 0) {
                ropeDimForRequest = ropeDim;
                ropeThetaForRequest = segment.rope_theta;
            } else if (ropeDimForRequest != ropeDim || std::fabs(ropeThetaForRequest - segment.rope_theta) > 1e-5f) {
                MNN_ERROR("[Error]: CUDA PrefixAttention requires consistent RoPE metadata across direct_segments\n");
                return NOT_SUPPORT;
            }

            auto basePath = prefixCacheLayerBase(mMeta->prefix_cache_dir, "cuda", segment.cache_name, layerIndex);
            auto keyPath = basePath + ".k";
            auto valuePath = basePath + ".v";
            auto expectedBytes = cudaPagedFileBytes(segment.token_count, kCudaPagedPrefixPageSize,
                                                   mKvNumHead, mHeadDim, mPrecision);
            std::string keyIdentity;
            std::string valueIdentity;
            size_t keyFileSize = 0;
            size_t valueFileSize = 0;
            if (!MNN::detail::prefixKVFileIdentity(keyPath, keyIdentity, &keyFileSize) ||
                !MNN::detail::prefixKVFileIdentity(valuePath, valueIdentity, &valueFileSize)) {
                MNN_ERROR("[Error]: CUDA PrefixAttention missing native paged KV file for %s layer %d\n",
                          segment.cache_name.c_str(), layerIndex);
                return FILE_OPEN_FAILED;
            }
            if (keyFileSize != expectedBytes || valueFileSize != expectedBytes) {
                MNN_ERROR("[Error]: CUDA PrefixAttention invalid native paged KV size for %s layer %d: key=%zu value=%zu expected=%zu\n",
                          segment.cache_name.c_str(), layerIndex, keyFileSize, valueFileSize, expectedBytes);
                return INVALID_VALUE;
            }

            auto cacheKey = cudaDocumentPageCacheKey(keyPath, valuePath, segment, layerIndex,
                                                     mPrecision, mKvNumHead, mHeadDim,
                                                     ropeDim, segment.rope_theta);
            if (cacheKey.empty()) {
                return FILE_OPEN_FAILED;
            }
            CudaPrefixSegmentPlacement placement;
            placement.logicalStart = logicalCursor;
            placement.tokenCount = segment.token_count;
            placement.physicalLength = alignToPage(segment.token_count, kCudaPagedPrefixPageSize);
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
                    iter->second.physicalStart >= basePhysicalReserved &&
                    iter->second.ropeDim == ropeDim &&
                    std::fabs(iter->second.ropeTheta - segment.rope_theta) <= 1e-5f) {
                    placement.deviceCacheHit = true;
                    placement.physicalStart = iter->second.physicalStart;
                    placement.physicalLength = iter->second.physicalLength;
                    placement.readyEvent = iter->second.readyEvent;
                } else {
                    int physicalStart = alignToPage(std::max(std::max(physicalCursor, docPool->nextPhysicalSlot),
                                                             basePhysicalReserved),
                                                    kCudaPagedPrefixPageSize);
                    placement.physicalStart = physicalStart;
                    docPool->nextPhysicalSlot = physicalStart + placement.physicalLength;
                    physicalCursor = docPool->nextPhysicalSlot;
                }
            }
            placement.deviceCacheLookupMs = nowMs() - lookupStart;
            physicalCursor = std::max(physicalCursor, placement.physicalStart + placement.physicalLength);
            placements.emplace_back(placement);
            logicalCursor += segment.token_count;
        }

        const int promptLogicalStart = basePast + segmentTotalTokens;
        const int promptPhysicalStart = alignToPage(physicalCursor, kCudaPagedPrefixPageSize);
        physicalCursor = promptPhysicalStart + appendKvSeqLen;
        const int requiredPhysicalTotal = alignToPage(physicalCursor, kCudaPagedPrefixPageSize);
        {
            std::lock_guard<std::mutex> lock(docPool->mutex);
            docPool->nextPhysicalSlot = std::max(docPool->nextPhysicalSlot, requiredPhysicalTotal);
        }

        auto err = reallocKVCache_gpu(requiredTotal, mBatch, mKvNumHead, mHeadDim, stream);
        if (err != NO_ERROR) {
            return err;
        }
        auto pagedErr = ensurePagedKVCache_gpu(requiredPhysicalTotal, mBatch, mKvNumHead, mHeadDim, stream);
        if (pagedErr != NO_ERROR) {
            return pagedErr;
        }
        if (basePast > 0 && importBasePastToPaged) {
            std::vector<int> baseSlotTable(basePast);
            std::vector<int> baseRopeTable(basePast, 0);
            for (int i = 0; i < basePast; ++i) {
                baseSlotTable[i] = i;
            }
            auto tableCopy = cudaMemcpyAsync(getTensorDevicePtr<int>(mCache->mPagedBlockTable.get()),
                                             baseSlotTable.data(), baseSlotTable.size() * sizeof(int),
                                             cudaMemcpyHostToDevice, stream);
            auto ropeCopy = cudaMemcpyAsync(getTensorDevicePtr<int>(mCache->mPagedRopeTable.get()),
                                            baseRopeTable.data(), baseRopeTable.size() * sizeof(int),
                                            cudaMemcpyHostToDevice, stream);
            if (tableCopy != cudaSuccess || ropeCopy != cudaSuccess) {
                MNN_ERROR("[Error]: CUDA PrefixAttention failed to upload base system paged table: %s / %s\n",
                          cudaGetErrorString(tableCopy), cudaGetErrorString(ropeCopy));
                return INVALID_VALUE;
            }
            dim3 copyBlock(32, 8, 1);
            dim3 copyGrid(UP_DIV(mHeadDim, copyBlock.x),
                          UP_DIV(basePast, copyBlock.y),
                          UP_DIV(mBatch * mKvNumHead, copyBlock.z));
            if (mPrecision == 4) {
                importContiguousKVToPagedKernel<float><<<copyGrid, copyBlock, 0, stream>>>(
                    getTensorDevicePtr<float>(mCache->mPastKey.get()),
                    getTensorDevicePtr<float>(mCache->mPastValue.get()),
                    getTensorDevicePtr<float>(mCache->mPagedKey.get()),
                    getTensorDevicePtr<float>(mCache->mPagedValue.get()),
                    getTensorDevicePtr<int>(mCache->mPagedBlockTable.get()),
                    mBatch, basePast, mKvNumHead, mHeadDim, mCache->mMaxLength, kCudaPagedPrefixPageSize);
            } else if (mPrecision == 2) {
                importContiguousKVToPagedKernel<__half><<<copyGrid, copyBlock, 0, stream>>>(
                    getTensorDevicePtr<__half>(mCache->mPastKey.get()),
                    getTensorDevicePtr<__half>(mCache->mPastValue.get()),
                    getTensorDevicePtr<__half>(mCache->mPagedKey.get()),
                    getTensorDevicePtr<__half>(mCache->mPagedValue.get()),
                    getTensorDevicePtr<int>(mCache->mPagedBlockTable.get()),
                    mBatch, basePast, mKvNumHead, mHeadDim, mCache->mMaxLength, kCudaPagedPrefixPageSize);
            } else {
                return NOT_SUPPORT;
            }
            checkKernelErrors;
        }
        resetPagedKVCacheState(basePast);

        std::vector<int> slotTable;
        std::vector<int> ropeTable;
        slotTable.resize(std::max(0, requiredTotal - basePast));
        ropeTable.resize(slotTable.size(), 0);
        for (const auto& placement : placements) {
            for (int i = 0; i < placement.tokenCount; ++i) {
                int logicalOffset = placement.logicalStart + i - basePast;
                slotTable[logicalOffset] = placement.physicalStart + i;
                ropeTable[logicalOffset] = 1;
            }
        }
        for (int i = 0; i < appendKvSeqLen; ++i) {
            int logicalOffset = promptLogicalStart + i - basePast;
            slotTable[logicalOffset] = promptPhysicalStart + i;
            ropeTable[logicalOffset] = 0;
        }
        if (!slotTable.empty()) {
            auto tableCopy = cudaMemcpyAsync(getTensorDevicePtr<int>(mCache->mPagedBlockTable.get()) + basePast,
                                             slotTable.data(), slotTable.size() * sizeof(int),
                                             cudaMemcpyHostToDevice, stream);
            auto ropeCopy = cudaMemcpyAsync(getTensorDevicePtr<int>(mCache->mPagedRopeTable.get()) + basePast,
                                            ropeTable.data(), ropeTable.size() * sizeof(int),
                                            cudaMemcpyHostToDevice, stream);
            if (tableCopy != cudaSuccess || ropeCopy != cudaSuccess) {
                MNN_ERROR("[Error]: CUDA PrefixAttention failed to upload paged token tables: %s / %s\n",
                          cudaGetErrorString(tableCopy), cudaGetErrorString(ropeCopy));
                return INVALID_VALUE;
            }
        }
        mCache->mPagedTokenTableCustom = true;
        mCache->mPagedPhysicalLength = requiredPhysicalTotal;
        mCache->mPagedRopeDim = ropeDimForRequest;
        mCache->mPagedRopeTheta = ropeThetaForRequest;

        size_t placementIndex = 0;
        for (const auto& segment : mMeta->prefix_segments) {
            if (segment.token_count <= 0) {
                continue;
            }
            const auto& placement = placements[placementIndex++];
            int ropeDim = segment.rope_dim > 0 ? segment.rope_dim : mHeadDim;
            ropeDim = std::min(ropeDim, mHeadDim);
            const size_t expectedBytes = cudaPagedFileBytes(segment.token_count, kCudaPagedPrefixPageSize,
                                                           mKvNumHead, mHeadDim, mPrecision);

            bool hostCacheHit = false;
            bool pinnedHost = false;
            bool pinnedCacheHit = false;
            double prefetchWaitMs = 0.0;
            double diskReadMs = 0.0;
            double pinnedCopyMs = 0.0;
            double h2dEnqueueMs = 0.0;
            double materializeEnqueueMs = 0.0;
            double attentionWaitMs = 0.0;
            size_t uploadedBytes = 0;

            if (placement.deviceCacheHit) {
                if (placement.readyEvent != nullptr) {
                    double waitStart = nowMs();
                    auto waitErr = cudaStreamWaitEvent(stream, placement.readyEvent, 0);
                    attentionWaitMs = nowMs() - waitStart;
                    if (waitErr != cudaSuccess) {
                        MNN_ERROR("[Error]: CUDA PrefixAttention failed to wait document page event: %s\n",
                                  cudaGetErrorString(waitErr));
                        return INVALID_VALUE;
                    }
                }
            } else {
                auto keyRead = CudaPinnedPrefixKVHostCache::get().readFile(placement.keyPath);
                auto valueRead = CudaPinnedPrefixKVHostCache::get().readFile(placement.valuePath);
                if (!keyRead.ok || !valueRead.ok || keyRead.data() == nullptr || valueRead.data() == nullptr) {
                    MNN_ERROR("[Error]: CUDA PrefixAttention failed to read prefix KV cache: %s / %s, key_error=%s value_error=%s\n",
                              placement.keyPath.c_str(), placement.valuePath.c_str(), keyRead.error.c_str(), valueRead.error.c_str());
                    return FILE_OPEN_FAILED;
                }
                if (keyRead.fileSize != expectedBytes || valueRead.fileSize != expectedBytes) {
                    MNN_ERROR("[Error]: CUDA PrefixAttention invalid native paged KV size for %s layer %d: key=%zu value=%zu expected=%zu\n",
                              segment.cache_name.c_str(), layerIndex, keyRead.fileSize, valueRead.fileSize, expectedBytes);
                    return INVALID_VALUE;
                }

                double h2dStart = nowMs();
                auto* keyPages = getTensorDevicePtr<uint8_t>(mCache->mPagedKey.get());
                auto* valuePages = getTensorDevicePtr<uint8_t>(mCache->mPagedValue.get());
                size_t dstByteOffset = static_cast<size_t>(placement.physicalStart) * mKvNumHead * mHeadDim * mPrecision;
                auto keyCopy = cudaMemcpyAsync(keyPages + dstByteOffset, keyRead.data(), keyRead.fileSize, cudaMemcpyHostToDevice, stream);
                auto valueCopy = cudaMemcpyAsync(valuePages + dstByteOffset, valueRead.data(), valueRead.fileSize, cudaMemcpyHostToDevice, stream);
                if (keyCopy != cudaSuccess || valueCopy != cudaSuccess) {
                    MNN_ERROR("[Error]: CUDA PrefixAttention failed to copy prefix KV into runtime pages: %s / %s\n",
                              cudaGetErrorString(keyCopy), cudaGetErrorString(valueCopy));
                    return INVALID_VALUE;
                }
                h2dEnqueueMs = nowMs() - h2dStart;
                uploadedBytes = keyRead.fileSize + valueRead.fileSize;
                hostCacheHit = keyRead.hostCacheHit && valueRead.hostCacheHit;
                pinnedHost = keyRead.isPinned && valueRead.isPinned;
                pinnedCacheHit = keyRead.pinnedCacheHit && valueRead.pinnedCacheHit;
                prefetchWaitMs = keyRead.prefetchWaitMs + valueRead.prefetchWaitMs;
                diskReadMs = keyRead.diskReadMs + valueRead.diskReadMs;
                pinnedCopyMs = keyRead.pinCopyMs + valueRead.pinCopyMs;

                cudaEvent_t readyEvent = nullptr;
                auto eventErr = cudaEventCreateWithFlags(&readyEvent, cudaEventDisableTiming);
                if (eventErr == cudaSuccess) {
                    eventErr = cudaEventRecord(readyEvent, stream);
                }
                if (eventErr != cudaSuccess) {
                    if (readyEvent != nullptr) {
                        cudaEventDestroy(readyEvent);
                    }
                    MNN_ERROR("[Error]: CUDA PrefixAttention failed to record document page event: %s\n",
                              cudaGetErrorString(eventErr));
                    return INVALID_VALUE;
                }
                {
                    std::lock_guard<std::mutex> lock(docPool->mutex);
                    CudaDevicePrefixDocumentRecord record;
                    record.physicalStart = placement.physicalStart;
                    record.physicalLength = placement.physicalLength;
                    record.tokenCount = segment.token_count;
                    record.ropeDim = ropeDim;
                    record.ropeTheta = segment.rope_theta;
                    record.readyEvent = readyEvent;
                    docPool->records[placement.deviceCacheKey] = record;
                }
            }

            MNN_PRINT("[CUDAPrefixAttention] layer=%d cache=%s tokens=%d logical_start=%d physical_start=%d host_cache_hit=%d pinned_host=%d pinned_cache_hit=%d device_cache_hit=%d device_cache_lookup_ms=%.3f device_prefetch=%d device_prefetch_submit=%d prefetch_wait_ms=%.3f disk_read_ms=%.3f pinned_copy_ms=%.3f host_to_device_ms=%.3f materialize_ms=%.3f attention_wait_ms=%.3f h2d_enqueue_bytes=%zu staging_path=0 on_read_rope=1\n",
                      layerIndex, placement.cacheName.c_str(), segment.token_count,
                      placement.logicalStart, placement.physicalStart,
                      hostCacheHit ? 1 : 0,
                      pinnedHost ? 1 : 0,
                      pinnedCacheHit ? 1 : 0,
                      placement.deviceCacheHit ? 1 : 0,
                      placement.deviceCacheLookupMs,
                      devicePrefetch ? 1 : 0,
                      devicePrefetch ? 1 : 0,
                      prefetchWaitMs,
                      diskReadMs,
                      pinnedCopyMs,
                      h2dEnqueueMs,
                      materializeEnqueueMs,
                      attentionWaitMs,
                      uploadedBytes);
        }

        if (logicalCursor != basePast + segmentTotalTokens) {
            MNN_ERROR("[Error]: CUDA PrefixAttention copied prefix length %d but expected %d\n",
                      logicalCursor - basePast, segmentTotalTokens);
            return INVALID_VALUE;
        }

        mCache->mPastLength = basePast + segmentTotalTokens;
        mCache->mPagedLength = basePast + segmentTotalTokens;
        if (layerIndex == 0) {
            MNN_PRINT("[CUDAPrefixAttention] block_table base_past=%d prefix_tokens=%d prompt_tokens=%d required_total=%d physical_total=%d page_size=%d logical_blocks=%d physical_blocks=%d prefix_blocks=%d prompt_blocks=%d prompt_physical_start=%d device_prefetch=%d\n",
                      basePast, segmentTotalTokens, appendKvSeqLen, requiredTotal, requiredPhysicalTotal,
                      kCudaPagedPrefixPageSize,
                      UP_DIV(requiredTotal, kCudaPagedPrefixPageSize),
                      UP_DIV(requiredPhysicalTotal, kCudaPagedPrefixPageSize),
                      UP_DIV(segmentTotalTokens, kCudaPagedPrefixPageSize),
                      UP_DIV(appendKvSeqLen, kCudaPagedPrefixPageSize),
                      promptPhysicalStart,
                      devicePrefetch ? 1 : 0);
        }
        summary.layerIndex = layerIndex;
        summary.basePast = basePast;
        summary.segmentTotalTokens = segmentTotalTokens;
        summary.promptTokens = appendKvSeqLen;
        summary.requiredTotal = requiredTotal;
        summary.requiredPhysicalTotal = requiredPhysicalTotal;
        summary.promptPhysicalStart = promptPhysicalStart;
        return NO_ERROR;
    }

    ErrorCode reservePagedPrefixKVForDevicePrefetch(cudaStream_t stream, uint64_t requestId) {
        CudaPrefixNvtxRange nvtx(cudaPrefixNvtxName("prefix_prefetch_reserve", mLayerIndex,
                                                    requestId, 0, stream));
        if (mMeta == nullptr || mMeta->prefix_request_id != requestId || mCache == nullptr) {
            return NO_ERROR;
        }
        int segmentTotalTokens = mMeta->segment_total_tokens;
        if (segmentTotalTokens <= 0) {
            for (const auto& segment : mMeta->prefix_segments) {
                segmentTotalTokens += segment.token_count;
            }
        }
        if (segmentTotalTokens <= 0) {
            return INVALID_VALUE;
        }
        int appendKvSeqLen = mMeta->prefix_prompt_token_count;
        if (appendKvSeqLen <= 0) {
            appendKvSeqLen = static_cast<int>(mMeta->add);
        }
        if (appendKvSeqLen <= 0) {
            appendKvSeqLen = mNewKvSeqLen;
        }
        int basePast = 0;
        if (mMeta->previous != mMeta->remove) {
            basePast = mCache->mPastLength;
            if (basePast <= 0) {
                basePast = static_cast<int>(mMeta->previous - mMeta->remove);
            }
        }
        int physicalCursor = alignToPage(basePast, kCudaPagedPrefixPageSize);
        for (const auto& segment : mMeta->prefix_segments) {
            if (segment.token_count <= 0) {
                continue;
            }
            physicalCursor = alignToPage(physicalCursor, kCudaPagedPrefixPageSize) +
                             alignToPage(segment.token_count, kCudaPagedPrefixPageSize);
        }
        physicalCursor = alignToPage(physicalCursor, kCudaPagedPrefixPageSize) + appendKvSeqLen;
        const int requiredPhysicalTotal = alignToPage(physicalCursor, kCudaPagedPrefixPageSize);
        const int oldPagedPhysicalLength = mCache->mPagedPhysicalLength;
        const bool oldPagedTokenTableCustom = mCache->mPagedTokenTableCustom;
        auto err = ensurePagedKVCache_gpu(requiredPhysicalTotal, mBatch, mKvNumHead, mHeadDim, stream);
        mCache->mPagedPhysicalLength = oldPagedPhysicalLength;
        mCache->mPagedTokenTableCustom = oldPagedTokenTableCustom;
        return err;
    }

    ErrorCode preparePagedPrefixKVDocumentsOnly(cudaStream_t stream, uint64_t requestId,
                                                int& appendKvSeqLen, PrefixPrepareSummary& summary,
                                                std::vector<H2DProfileRecord>* h2dProfiles = nullptr) {
        CudaPrefixNvtxRange nvtx(cudaPrefixNvtxName("prefix_document_prefetch_prepare",
                                                    mLayerIndex, requestId, 0, stream));
        if (mMeta == nullptr || mMeta->prefix_request_id != requestId || mCache == nullptr) {
            return NO_ERROR;
        }
        int segmentTotalTokens = mMeta->segment_total_tokens;
        if (segmentTotalTokens <= 0) {
            for (const auto& segment : mMeta->prefix_segments) {
                segmentTotalTokens += segment.token_count;
            }
        }
        if (segmentTotalTokens <= 0) {
            return INVALID_VALUE;
        }

        appendKvSeqLen = mMeta->prefix_prompt_token_count;
        if (appendKvSeqLen <= 0) {
            appendKvSeqLen = static_cast<int>(mMeta->add);
        }
        if (appendKvSeqLen <= 0) {
            appendKvSeqLen = mNewKvSeqLen;
        }
        int basePast = 0;
        if (mMeta->previous != mMeta->remove) {
            basePast = mCache->mPastLength;
            if (basePast <= 0) {
                basePast = static_cast<int>(mMeta->previous - mMeta->remove);
            }
        }
        const int requiredTotal = basePast + segmentTotalTokens + appendKvSeqLen;
        const int basePhysicalReserved = alignToPage(basePast, kCudaPagedPrefixPageSize);

        int layerIndex = mLayerIndex;
        if (layerIndex < 0) {
            MNN_ERROR("[Error]: CUDA PrefixAttention device_prefetch requires explicit AttentionParam.layer_index\n");
            return NOT_SUPPORT;
        }

        auto docPool = cudaDocumentPoolForCache(mCache);
        if (docPool == nullptr) {
            return OUT_OF_MEMORY;
        }
        if (!mCache->mPagedKey || !mCache->mPagedValue ||
            mCache->mPagedKey->deviceId() == 0 || mCache->mPagedValue->deviceId() == 0) {
            MNN_ERROR("[Error]: CUDA PrefixAttention device_prefetch has no paged KV capacity for layer %d\n",
                      layerIndex);
            return INVALID_VALUE;
        }

        std::vector<CudaPrefixSegmentPlacement> placements;
        placements.reserve(mMeta->prefix_segments.size());
        int logicalCursor = basePast;
        int physicalCursor = basePhysicalReserved;
        {
            std::lock_guard<std::mutex> lock(docPool->mutex);
            physicalCursor = std::max(physicalCursor, docPool->nextPhysicalSlot);
        }
        int ropeDimForRequest = 0;
        float ropeThetaForRequest = 10000.0f;

        for (const auto& segment : mMeta->prefix_segments) {
            if (segment.token_count <= 0) {
                continue;
            }
            if (segment.key_rope_state != KVMeta::KeyRopeCanonicalNoRope ||
                segment.rope_pairing != KVMeta::RopePairingHalf) {
                return NOT_SUPPORT;
            }
            if (segment.backend != "cuda" || segment.layout != kCudaPagedPrefixLayout ||
                segment.page_size != kCudaPagedPrefixPageSize) {
                return NOT_SUPPORT;
            }
            int ropeDim = segment.rope_dim > 0 ? segment.rope_dim : mHeadDim;
            ropeDim = std::min(ropeDim, mHeadDim);
            if (ropeDim <= 0 || (ropeDim % 2) != 0 || segment.rope_theta <= 0.0f) {
                return INVALID_VALUE;
            }
            if (ropeDimForRequest == 0) {
                ropeDimForRequest = ropeDim;
                ropeThetaForRequest = segment.rope_theta;
            } else if (ropeDimForRequest != ropeDim ||
                       std::fabs(ropeThetaForRequest - segment.rope_theta) > 1e-5f) {
                return NOT_SUPPORT;
            }

            auto basePath = prefixCacheLayerBase(mMeta->prefix_cache_dir, "cuda", segment.cache_name, layerIndex);
            auto keyPath = basePath + ".k";
            auto valuePath = basePath + ".v";
            auto expectedBytes = cudaPagedFileBytes(segment.token_count, kCudaPagedPrefixPageSize,
                                                   mKvNumHead, mHeadDim, mPrecision);
            std::string keyIdentity;
            std::string valueIdentity;
            size_t keyFileSize = 0;
            size_t valueFileSize = 0;
            if (!MNN::detail::prefixKVFileIdentity(keyPath, keyIdentity, &keyFileSize) ||
                !MNN::detail::prefixKVFileIdentity(valuePath, valueIdentity, &valueFileSize)) {
                return FILE_OPEN_FAILED;
            }
            if (keyFileSize != expectedBytes || valueFileSize != expectedBytes) {
                return INVALID_VALUE;
            }

            auto cacheKey = cudaDocumentPageCacheKey(keyPath, valuePath, segment, layerIndex,
                                                     mPrecision, mKvNumHead, mHeadDim,
                                                     ropeDim, segment.rope_theta);
            if (cacheKey.empty()) {
                return FILE_OPEN_FAILED;
            }

            CudaPrefixSegmentPlacement placement;
            placement.logicalStart = logicalCursor;
            placement.tokenCount = segment.token_count;
            placement.physicalLength = alignToPage(segment.token_count, kCudaPagedPrefixPageSize);
            placement.deviceCacheKey = cacheKey;
            placement.keyPath = keyPath;
            placement.valuePath = valuePath;
            placement.cacheName = segment.cache_name;
            {
                std::lock_guard<std::mutex> lock(docPool->mutex);
                auto iter = docPool->records.find(cacheKey);
                if (iter != docPool->records.end() &&
                    iter->second.tokenCount == segment.token_count &&
                    iter->second.physicalStart >= basePhysicalReserved &&
                    iter->second.ropeDim == ropeDim &&
                    std::fabs(iter->second.ropeTheta - segment.rope_theta) <= 1e-5f) {
                    placement.deviceCacheHit = true;
                    placement.physicalStart = iter->second.physicalStart;
                    placement.physicalLength = iter->second.physicalLength;
                    placement.readyEvent = iter->second.readyEvent;
                } else {
                    int physicalStart = alignToPage(std::max(physicalCursor, docPool->nextPhysicalSlot),
                                                    kCudaPagedPrefixPageSize);
                    placement.physicalStart = physicalStart;
                    docPool->nextPhysicalSlot = physicalStart + placement.physicalLength;
                    physicalCursor = docPool->nextPhysicalSlot;
                }
            }
            physicalCursor = std::max(physicalCursor, placement.physicalStart + placement.physicalLength);
            placements.emplace_back(placement);
            logicalCursor += segment.token_count;
        }

        const int promptPhysicalStart = alignToPage(physicalCursor, kCudaPagedPrefixPageSize);
        const int requiredPhysicalTotal = alignToPage(promptPhysicalStart + appendKvSeqLen,
                                                     kCudaPagedPrefixPageSize);
        size_t placementIndex = 0;
        for (const auto& segment : mMeta->prefix_segments) {
            if (segment.token_count <= 0) {
                continue;
            }
            const auto& placement = placements[placementIndex++];
            if (placement.deviceCacheHit) {
                if (placement.readyEvent != nullptr) {
                    auto waitErr = cudaStreamWaitEvent(stream, placement.readyEvent, 0);
                    if (waitErr != cudaSuccess) {
                        return INVALID_VALUE;
                    }
                }
                continue;
            }

            const size_t expectedBytes = cudaPagedFileBytes(segment.token_count, kCudaPagedPrefixPageSize,
                                                           mKvNumHead, mHeadDim, mPrecision);
            auto keyRead = CudaPinnedPrefixKVHostCache::get().readFile(placement.keyPath);
            auto valueRead = CudaPinnedPrefixKVHostCache::get().readFile(placement.valuePath);
            if (!keyRead.ok || !valueRead.ok || keyRead.data() == nullptr || valueRead.data() == nullptr ||
                keyRead.fileSize != expectedBytes || valueRead.fileSize != expectedBytes) {
                return FILE_OPEN_FAILED;
            }

            const size_t h2dBytes = keyRead.fileSize + valueRead.fileSize;
            CudaPrefixNvtxRange h2dNvtx(cudaPrefixNvtxName("prefix_document_h2d_submit",
                                                           layerIndex, requestId, h2dBytes, stream));
            auto* keyPages = getTensorDevicePtr<uint8_t>(mCache->mPagedKey.get());
            auto* valuePages = getTensorDevicePtr<uint8_t>(mCache->mPagedValue.get());
            size_t dstByteOffset = static_cast<size_t>(placement.physicalStart) * mKvNumHead * mHeadDim * mPrecision;
            const bool profileH2D = cudaPrefixProfileH2DEnabled();
            const bool profileVerbose = cudaPrefixProfileVerboseEnabled();
            H2DProfileRecord profile;
            if (profileH2D) {
                profile.layerIndex = layerIndex;
                profile.requestId = requestId;
                profile.cacheName = placement.cacheName;
                profile.stream = stream;
                profile.streamId = cudaPrefixStreamId(stream);
                profile.keyBytes = keyRead.fileSize;
                profile.valueBytes = valueRead.fileSize;
                profile.physicalStart = placement.physicalStart;
                profile.events = std::make_shared<CudaPrefixH2DProfileEvents>();
                auto startErr = cudaEventCreate(&profile.events->startEvent);
                auto keyEndErr = cudaEventCreate(&profile.events->keyEndEvent);
                auto valueEndErr = cudaEventCreate(&profile.events->valueEndEvent);
                if (startErr != cudaSuccess || keyEndErr != cudaSuccess || valueEndErr != cudaSuccess) {
                    destroyH2DProfileRecord(profile);
                    MNN_ERROR("[Error]: CUDA PrefixAttention failed to create H2D profile events\n");
                    return INVALID_VALUE;
                }
                profile.startEvent = profile.events->startEvent;
                profile.keyEndEvent = profile.events->keyEndEvent;
                profile.valueEndEvent = profile.events->valueEndEvent;
                auto recordErr = cudaEventRecord(profile.startEvent, stream);
                if (recordErr != cudaSuccess) {
                    destroyH2DProfileRecord(profile);
                    return INVALID_VALUE;
                }
                if (profileVerbose) {
                    MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] layer=%d request_id=%llu cache=%s stream=%p stream_id=%llu physical_start=%d key_ptr_attr={%s} value_ptr_attr={%s} key_dst_attr={%s} value_dst_attr={%s}\n",
                              layerIndex,
                              static_cast<unsigned long long>(requestId),
                              placement.cacheName.c_str(),
                              stream,
                              profile.streamId,
                              placement.physicalStart,
                              cudaPrefixPointerAttributesSummary(keyRead.data()).c_str(),
                              cudaPrefixPointerAttributesSummary(valueRead.data()).c_str(),
                              cudaPrefixPointerAttributesSummary(keyPages + dstByteOffset).c_str(),
                              cudaPrefixPointerAttributesSummary(valuePages + dstByteOffset).c_str());
                }
            }
            cudaError_t keyCopy = cudaSuccess;
            if (profileVerbose) {
                CudaPrefixNvtxRange keyNvtx(cudaPrefixNvtxName("prefix_document_h2d_key_submit",
                                                               layerIndex, requestId, keyRead.fileSize, stream));
                keyCopy = cudaMemcpyAsync(keyPages + dstByteOffset, keyRead.data(), keyRead.fileSize,
                                          cudaMemcpyHostToDevice, stream);
            } else {
                keyCopy = cudaMemcpyAsync(keyPages + dstByteOffset, keyRead.data(), keyRead.fileSize,
                                          cudaMemcpyHostToDevice, stream);
            }
            if (profileH2D) {
                auto recordErr = cudaEventRecord(profile.keyEndEvent, stream);
                if (recordErr != cudaSuccess) {
                    destroyH2DProfileRecord(profile);
                    return INVALID_VALUE;
                }
            }
            cudaError_t valueCopy = cudaSuccess;
            if (profileVerbose) {
                CudaPrefixNvtxRange valueNvtx(cudaPrefixNvtxName("prefix_document_h2d_value_submit",
                                                                 layerIndex, requestId, valueRead.fileSize, stream));
                valueCopy = cudaMemcpyAsync(valuePages + dstByteOffset, valueRead.data(), valueRead.fileSize,
                                            cudaMemcpyHostToDevice, stream);
            } else {
                valueCopy = cudaMemcpyAsync(valuePages + dstByteOffset, valueRead.data(), valueRead.fileSize,
                                            cudaMemcpyHostToDevice, stream);
            }
            if (profileH2D) {
                auto recordErr = cudaEventRecord(profile.valueEndEvent, stream);
                if (recordErr != cudaSuccess) {
                    destroyH2DProfileRecord(profile);
                    return INVALID_VALUE;
                }
                cudaPrefixRequestProfileRegisterH2D(mMeta, requestId, layerIndex,
                                                    placement.cacheName, profile.streamId,
                                                    profile.keyBytes, profile.valueBytes,
                                                    profile.physicalStart, profile.events);
                if (h2dProfiles != nullptr) {
                    h2dProfiles->push_back(profile);
                } else {
                    destroyH2DProfileRecord(profile);
                }
            }
            if (keyCopy != cudaSuccess || valueCopy != cudaSuccess) {
                return INVALID_VALUE;
            }
            MNN_PRINT("[CUDAPrefixAttentionPrefetch] layer=%d cache=%s document_h2d_bytes=%zu stream=%p pinned_host=%d pinned_cache_hit=%d\n",
                      layerIndex,
                      placement.cacheName.c_str(),
                      h2dBytes,
                      stream,
                      (keyRead.isPinned && valueRead.isPinned) ? 1 : 0,
                      (keyRead.pinnedCacheHit && valueRead.pinnedCacheHit) ? 1 : 0);
            cudaEvent_t readyEvent = nullptr;
            auto eventErr = cudaEventCreateWithFlags(&readyEvent, cudaEventDisableTiming);
            if (eventErr == cudaSuccess) {
                eventErr = cudaEventRecord(readyEvent, stream);
            }
            if (eventErr != cudaSuccess) {
                if (readyEvent != nullptr) {
                    cudaEventDestroy(readyEvent);
                }
                return INVALID_VALUE;
            }
            {
                std::lock_guard<std::mutex> lock(docPool->mutex);
                CudaDevicePrefixDocumentRecord record;
                record.physicalStart = placement.physicalStart;
                record.physicalLength = placement.physicalLength;
                record.tokenCount = segment.token_count;
                record.ropeDim = segment.rope_dim > 0 ? std::min(segment.rope_dim, mHeadDim) : mHeadDim;
                record.ropeTheta = segment.rope_theta;
                record.readyEvent = readyEvent;
                docPool->records[placement.deviceCacheKey] = record;
            }
        }

        summary.layerIndex = layerIndex;
        summary.basePast = basePast;
        summary.segmentTotalTokens = segmentTotalTokens;
        summary.promptTokens = appendKvSeqLen;
        summary.requiredTotal = requiredTotal;
        summary.requiredPhysicalTotal = requiredPhysicalTotal;
        summary.promptPhysicalStart = promptPhysicalStart;
        return NO_ERROR;
    }

    ErrorCode runDevicePrefetchOnStream(cudaStream_t copyStream, uint64_t requestId) {
        CudaPrefixNvtxRange nvtx(cudaPrefixNvtxName("prefix_prefetch_worker",
                                                    mLayerIndex, requestId, 0, copyStream));
        auto start = nowMs();
        int appendKvSeqLen = 0;
        PrefixPrepareSummary summary;
        std::vector<H2DProfileRecord> h2dProfiles;
        auto err = preparePagedPrefixKVDocumentsOnly(copyStream, requestId, appendKvSeqLen, summary, &h2dProfiles);
        if (err != NO_ERROR) {
            destroyH2DProfileRecords(h2dProfiles);
            return err;
        }
        cudaEvent_t event = nullptr;
        auto eventErr = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        if (eventErr != cudaSuccess) {
            MNN_ERROR("[Error]: CUDA PrefixAttention failed to create device_prefetch event: %s\n",
                      cudaGetErrorString(eventErr));
            destroyH2DProfileRecords(h2dProfiles);
            return INVALID_VALUE;
        }
        eventErr = cudaEventRecord(event, copyStream);
        if (eventErr != cudaSuccess) {
            MNN_ERROR("[Error]: CUDA PrefixAttention failed to record device_prefetch event: %s\n",
                      cudaGetErrorString(eventErr));
            cudaEventDestroy(event);
            destroyH2DProfileRecords(h2dProfiles);
            return INVALID_VALUE;
        }
        {
            std::lock_guard<std::mutex> lock(mDevicePrefetchMutex);
            if (mDevicePrefetchRequestId != requestId) {
                cudaEventDestroy(event);
                destroyH2DProfileRecords(h2dProfiles);
                return NO_ERROR;
            }
            clearDevicePrefetchEvent();
            mDevicePrefetchReadyEvent = event;
            mDevicePrefetchH2DProfiles = h2dProfiles;
            h2dProfiles.clear();
            mDevicePrefetchAppendKvSeqLen = appendKvSeqLen;
            mDevicePrefetchSummary = summary;
        }
        MNN_PRINT("[CUDAPrefixAttentionPrefetch] layer=%d request_id=%llu submit_ms=%.3f prefix_tokens=%d prompt_tokens=%d physical_total=%d\n",
                  summary.layerIndex,
                  static_cast<unsigned long long>(requestId),
                  nowMs() - start,
                  summary.segmentTotalTokens,
                  summary.promptTokens,
                  summary.requiredPhysicalTotal);
        return NO_ERROR;
    }

    ErrorCode enqueueDevicePrefetch(cudaStream_t copyStream, uint64_t requestId) {
        {
            std::unique_lock<std::mutex> lock(mDevicePrefetchMutex);
            while (mDevicePrefetchPending) {
                mDevicePrefetchCondition.wait(lock);
            }
            if (mDevicePrefetchRequestId == requestId && mDevicePrefetchReady) {
                return mDevicePrefetchStatus;
            }
            clearDevicePrefetchEvent();
            mDevicePrefetchRequestId = requestId;
            mDevicePrefetchStatus = NO_ERROR;
            mDevicePrefetchAppendKvSeqLen = 0;
            mDevicePrefetchSummary = PrefixPrepareSummary();
            mDevicePrefetchReady = false;
        }
        auto err = runDevicePrefetchOnStream(copyStream, requestId);
        {
            std::lock_guard<std::mutex> lock(mDevicePrefetchMutex);
            if (mDevicePrefetchRequestId == requestId) {
                mDevicePrefetchStatus = err;
                mDevicePrefetchReady = true;
            }
        }
        mDevicePrefetchCondition.notify_all();
        return err;
    }

    ErrorCode enqueueDevicePrefetchAsync(uint64_t requestId) {
        {
            std::lock_guard<std::mutex> lock(mDevicePrefetchMutex);
            if (mDevicePrefetchRequestId == requestId &&
                (mDevicePrefetchPending || mDevicePrefetchReady)) {
                return NO_ERROR;
            }
            clearDevicePrefetchEvent();
            mDevicePrefetchRequestId = requestId;
            mDevicePrefetchStatus = NO_ERROR;
            mDevicePrefetchAppendKvSeqLen = 0;
            mDevicePrefetchSummary = PrefixPrepareSummary();
            mDevicePrefetchPending = true;
            mDevicePrefetchReady = false;
        }
        bool posted = CudaPrefixPrefetchThreadPool::get().post([this, requestId](cudaStream_t stream) {
            auto err = stream != nullptr ? runDevicePrefetchOnStream(stream, requestId) : INVALID_VALUE;
            {
                std::lock_guard<std::mutex> lock(mDevicePrefetchMutex);
                if (mDevicePrefetchRequestId == requestId) {
                    mDevicePrefetchStatus = err;
                    mDevicePrefetchPending = false;
                    mDevicePrefetchReady = true;
                }
            }
            mDevicePrefetchCondition.notify_all();
        });
        if (!posted) {
            std::lock_guard<std::mutex> lock(mDevicePrefetchMutex);
            if (mDevicePrefetchRequestId == requestId) {
                mDevicePrefetchStatus = INVALID_VALUE;
                mDevicePrefetchPending = false;
                mDevicePrefetchReady = true;
            }
            mDevicePrefetchCondition.notify_all();
            return INVALID_VALUE;
        }
        return NO_ERROR;
    }

    ErrorCode consumeDevicePrefetch(cudaStream_t stream, bool& hit, int& appendKvSeqLen) {
        CudaPrefixNvtxRange nvtx(cudaPrefixNvtxName("prefix_prefetch_consume",
                                                    mLayerIndex,
                                                    mMeta != nullptr ? mMeta->prefix_request_id : 0,
                                                    0, stream));
        hit = false;
        cudaEvent_t readyEvent = nullptr;
        PrefixPrepareSummary summary;
        uint64_t requestId = 0;
        {
            std::unique_lock<std::mutex> lock(mDevicePrefetchMutex);
            if (mMeta == nullptr || mDevicePrefetchRequestId != mMeta->prefix_request_id ||
                mDevicePrefetchRequestId == 0) {
                return NO_ERROR;
            }
            while (mDevicePrefetchPending) {
                mDevicePrefetchCondition.wait(lock);
            }
            if (mDevicePrefetchStatus != NO_ERROR) {
                return mDevicePrefetchStatus;
            }
            if (!mDevicePrefetchReady || mDevicePrefetchReadyEvent == nullptr) {
                return NO_ERROR;
            }
            readyEvent = mDevicePrefetchReadyEvent;
            appendKvSeqLen = mDevicePrefetchAppendKvSeqLen;
            summary = mDevicePrefetchSummary;
            requestId = mDevicePrefetchRequestId;
            reportH2DProfilesLocked(requestId);
        }
        double waitStart = nowMs();
        CudaPrefixNvtxRange waitNvtx(cudaPrefixNvtxName("prefix_prefetch_wait_event",
                                                        summary.layerIndex, requestId, 0, stream));
        auto waitErr = cudaStreamWaitEvent(stream, readyEvent, 0);
        double waitMs = nowMs() - waitStart;
        if (waitErr != cudaSuccess) {
            MNN_ERROR("[Error]: CUDA PrefixAttention failed to wait device_prefetch event: %s\n",
                      cudaGetErrorString(waitErr));
            return INVALID_VALUE;
        }
        hit = true;
        MNN_PRINT("[CUDAPrefixAttention] layer=%d request_id=%llu device_prefetch=1 device_prefetch_hit=1 attention_wait_ms=%.3f prefix_tokens=%d prompt_tokens=%d required_total=%d physical_total=%d prompt_physical_start=%d\n",
                  summary.layerIndex,
                  static_cast<unsigned long long>(requestId),
                  waitMs,
                  summary.segmentTotalTokens,
                  summary.promptTokens,
                  summary.requiredTotal,
                  summary.requiredPhysicalTotal,
                  summary.promptPhysicalStart);
        return NO_ERROR;
    }

    static bool submitDevicePrefetchForMeta(KVMeta* meta, bool asyncSubmit) {
        if (meta == nullptr || !meta->prefix_device_prefetch || meta->prefix_request_id == 0) {
            return false;
        }
        CudaPrefixNvtxRange nvtx(cudaPrefixNvtxName(asyncSubmit ? "prefix_prefetch_dispatch_async" :
                                                                  "prefix_prefetch_dispatch_sync",
                                                    -1, meta->prefix_request_id));
        std::vector<CUDAPrefixAttentionExecution*> targets;
        {
            std::lock_guard<std::mutex> lock(registryMutex());
            auto& submitted = submittedRequests();
            auto submittedIter = submitted.find(meta);
            if (submittedIter != submitted.end() && submittedIter->second == meta->prefix_request_id) {
                return true;
            }

            auto registryIter = registry().find(meta);
            if (registryIter != registry().end()) {
                std::unordered_map<int, CUDAPrefixAttentionExecution*> byLayer;
                for (auto* exe : registryIter->second) {
                    if (exe == nullptr || exe->mMeta != meta || exe->mLayerIndex < 0 ||
                        exe->mBatch <= 0 || exe->mKvNumHead <= 0 || exe->mHeadDim <= 0 ||
                        exe->mCache == nullptr) {
                        continue;
                    }
                    byLayer[exe->mLayerIndex] = exe;
                }
                targets.reserve(byLayer.size());
                for (auto& iter : byLayer) {
                    targets.emplace_back(iter.second);
                }
            }
            if (targets.empty()) {
                return false;
            }
            submitted[meta] = meta->prefix_request_id;
        }

        std::sort(targets.begin(), targets.end(), [](const CUDAPrefixAttentionExecution* left,
                                                     const CUDAPrefixAttentionExecution* right) {
            return left->mLayerIndex < right->mLayerIndex;
        });

        if (asyncSubmit) {
            auto reserveStream = devicePrefetchStream();
            if (reserveStream == nullptr) {
                std::lock_guard<std::mutex> lock(registryMutex());
                submittedRequests().erase(meta);
                return false;
            }
            for (auto* target : targets) {
                auto err = target->reservePagedPrefixKVForDevicePrefetch(reserveStream, meta->prefix_request_id);
                if (err != NO_ERROR) {
                    std::lock_guard<std::mutex> lock(registryMutex());
                    submittedRequests().erase(meta);
                    return false;
                }
            }
            auto syncErr = cudaStreamSynchronize(reserveStream);
            if (syncErr != cudaSuccess) {
                MNN_ERROR("[Error]: CUDA PrefixAttention failed to reserve device_prefetch paged KV: %s\n",
                          cudaGetErrorString(syncErr));
                std::lock_guard<std::mutex> lock(registryMutex());
                submittedRequests().erase(meta);
                return false;
            }
        }

        MNN_PRINT("[CUDAPrefixAttentionPrefetch] request_id=%llu submit_layers=%zu mode=%s workers=%d\n",
                  static_cast<unsigned long long>(meta->prefix_request_id),
                  targets.size(),
                  asyncSubmit ? "async" : "sync",
                  asyncSubmit ? CudaPrefixPrefetchThreadPool::get().workerCount() : 1);
        cudaPrefixRequestProfileExpect(meta, meta->prefix_request_id, static_cast<int>(targets.size()));
        if (asyncSubmit) {
            bool ok = true;
            for (auto* target : targets) {
                ok = target->enqueueDevicePrefetchAsync(meta->prefix_request_id) == NO_ERROR && ok;
            }
            return ok;
        }
        auto copyStream = devicePrefetchStream();
        if (copyStream == nullptr) {
            MNN_PRINT("[CUDAPrefixAttentionPrefetch] request_id=%llu unavailable_copy_stream=1 fallback_sync=1\n",
                      static_cast<unsigned long long>(meta->prefix_request_id));
            return false;
        }
        for (auto* target : targets) {
            auto err = target->enqueueDevicePrefetch(copyStream, meta->prefix_request_id);
            if (err != NO_ERROR) {
                return false;
            }
        }
        return true;
    }

    ErrorCode submitDevicePrefetchForRequestIfNeeded() {
        if (mMeta == nullptr || !mMeta->prefix_device_prefetch || mMeta->prefix_request_id == 0) {
            return NO_ERROR;
        }
        if (mLayerIndex != 0) {
            return NO_ERROR;
        }
        submitDevicePrefetchForMeta(mMeta, false);
        return NO_ERROR;
    }

    void clearDevicePrefetchEvent() {
        if (mDevicePrefetchReadyEvent != nullptr) {
            cudaEventDestroy(mDevicePrefetchReadyEvent);
            mDevicePrefetchReadyEvent = nullptr;
        }
        destroyH2DProfileRecords(mDevicePrefetchH2DProfiles);
    }

    static void destroyH2DProfileRecord(H2DProfileRecord& record) {
        record.events.reset();
        record.startEvent = nullptr;
        record.keyEndEvent = nullptr;
        record.valueEndEvent = nullptr;
    }

    static void destroyH2DProfileRecords(std::vector<H2DProfileRecord>& records) {
        for (auto& record : records) {
            destroyH2DProfileRecord(record);
        }
        records.clear();
    }

    void reportH2DProfilesLocked(uint64_t requestId) {
        if (!cudaPrefixProfileH2DEnabled() || !cudaPrefixProfileVerboseEnabled()) {
            return;
        }
        for (auto& record : mDevicePrefetchH2DProfiles) {
            if (record.requestId != requestId || record.reported || record.valueEndEvent == nullptr) {
                continue;
            }
            auto query = cudaEventQuery(record.valueEndEvent);
            if (query == cudaErrorNotReady) {
                if (!record.pendingReported) {
                    record.pendingReported = true;
                    MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] layer=%d request_id=%llu cache=%s h2d_gpu_ms=pending stream=%p stream_id=%llu key_bytes=%zu value_bytes=%zu physical_start=%d\n",
                              record.layerIndex,
                              static_cast<unsigned long long>(record.requestId),
                              record.cacheName.c_str(),
                              record.stream,
                              record.streamId,
                              record.keyBytes,
                              record.valueBytes,
                              record.physicalStart);
                }
                continue;
            }
            if (query != cudaSuccess) {
                MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] layer=%d request_id=%llu cache=%s h2d_gpu_ms=query_error:%s stream=%p stream_id=%llu key_bytes=%zu value_bytes=%zu physical_start=%d\n",
                          record.layerIndex,
                          static_cast<unsigned long long>(record.requestId),
                          record.cacheName.c_str(),
                          cudaGetErrorString(query),
                          record.stream,
                          record.streamId,
                          record.keyBytes,
                          record.valueBytes,
                          record.physicalStart);
                record.reported = true;
                continue;
            }
            float keyMs = 0.0f;
            float valueMs = 0.0f;
            float totalMs = 0.0f;
            auto keyErr = cudaEventElapsedTime(&keyMs, record.startEvent, record.keyEndEvent);
            auto valueErr = cudaEventElapsedTime(&valueMs, record.keyEndEvent, record.valueEndEvent);
            auto totalErr = cudaEventElapsedTime(&totalMs, record.startEvent, record.valueEndEvent);
            if (keyErr != cudaSuccess || valueErr != cudaSuccess || totalErr != cudaSuccess) {
                MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] layer=%d request_id=%llu cache=%s h2d_gpu_ms=elapsed_error stream=%p stream_id=%llu key_bytes=%zu value_bytes=%zu physical_start=%d\n",
                          record.layerIndex,
                          static_cast<unsigned long long>(record.requestId),
                          record.cacheName.c_str(),
                          record.stream,
                          record.streamId,
                          record.keyBytes,
                          record.valueBytes,
                          record.physicalStart);
                record.reported = true;
                continue;
            }
            MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] layer=%d request_id=%llu cache=%s h2d_gpu_ms=%.3f key_gpu_ms=%.3f value_gpu_ms=%.3f stream=%p stream_id=%llu key_bytes=%zu value_bytes=%zu physical_start=%d\n",
                      record.layerIndex,
                      static_cast<unsigned long long>(record.requestId),
                      record.cacheName.c_str(),
                      totalMs,
                      keyMs,
                      valueMs,
                      record.stream,
                      record.streamId,
                      record.keyBytes,
                      record.valueBytes,
                      record.physicalStart);
            record.reported = true;
        }
    }

    bool shouldProfilePrefixCompute() const {
        return cudaPrefixProfileH2DEnabled() &&
               mMeta != nullptr &&
               mMeta->file_flag == KVMeta::PendingReadSegments &&
               mMeta->prefix_device_prefetch &&
               mMeta->prefix_request_id != 0 &&
               mLayerIndex >= 0;
    }

    void clearComputeProfileEvents() {
        mComputeProfileEvents.reset();
        mComputeStartEvent = nullptr;
        mComputeEndEvent = nullptr;
        mComputeProfilePending = false;
        mComputeProfileReported = false;
        mComputeProfilePendingReported = false;
        mComputeProfileReporterLaunched = false;
        mComputeProfileRequestId = 0;
        mComputeProfileLayerIndex = -1;
        mComputeProfileStream = nullptr;
        mComputeProfileStreamId = 0;
    }

    void launchComputeH2DOverlapReporter() {
        if (mComputeProfileReporterLaunched || mComputeStartEvent == nullptr || mComputeEndEvent == nullptr) {
            return;
        }
        std::vector<H2DProfileRecord> h2dProfiles;
        {
            std::lock_guard<std::mutex> lock(mDevicePrefetchMutex);
            h2dProfiles = mDevicePrefetchH2DProfiles;
        }
        auto computeStart = mComputeStartEvent;
        auto computeEnd = mComputeEndEvent;
        int layerIndex = mComputeProfileLayerIndex;
        uint64_t requestId = mComputeProfileRequestId;
        cudaStream_t computeStream = mComputeProfileStream;
        unsigned long long computeStreamId = mComputeProfileStreamId;
        mComputeProfileReporterLaunched = true;
        std::thread([h2dProfiles, computeStart, computeEnd, layerIndex, requestId,
                     computeStream, computeStreamId]() mutable {
            bool ready = false;
            for (int retry = 0; retry < 200; ++retry) {
                auto query = cudaEventQuery(computeEnd);
                if (query == cudaSuccess) {
                    ready = true;
                    break;
                }
                if (query != cudaErrorNotReady) {
                    MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] layer=%d request_id=%llu compute_gpu_ms=query_error:%s stream=%p stream_id=%llu async_report=1\n",
                              layerIndex,
                              static_cast<unsigned long long>(requestId),
                              cudaGetErrorString(query),
                              computeStream,
                              computeStreamId);
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (!ready) {
                MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] layer=%d request_id=%llu compute_gpu_ms=pending stream=%p stream_id=%llu async_report=1\n",
                          layerIndex,
                          static_cast<unsigned long long>(requestId),
                          computeStream,
                          computeStreamId);
                return;
            }
            float computeMs = 0.0f;
            auto computeErr = cudaEventElapsedTime(&computeMs, computeStart, computeEnd);
            if (computeErr != cudaSuccess) {
                MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] layer=%d request_id=%llu compute_gpu_ms=elapsed_error:%s stream=%p stream_id=%llu async_report=1\n",
                          layerIndex,
                          static_cast<unsigned long long>(requestId),
                          cudaGetErrorString(computeErr),
                          computeStream,
                          computeStreamId);
                return;
            }

            double h2dTotalMs = 0.0;
            double selfOverlapMs = 0.0;
            int readyCopies = 0;
            int pendingCopies = 0;
            for (const auto& record : h2dProfiles) {
                if (record.requestId != requestId || record.valueEndEvent == nullptr) {
                    continue;
                }
                auto copyQuery = cudaEventQuery(record.valueEndEvent);
                if (copyQuery == cudaErrorNotReady) {
                    ++pendingCopies;
                    continue;
                }
                if (copyQuery != cudaSuccess) {
                    continue;
                }
                float copyMs = 0.0f;
                float computeStartRelMs = 0.0f;
                float computeEndRelMs = 0.0f;
                auto copyErr = cudaEventElapsedTime(&copyMs, record.startEvent, record.valueEndEvent);
                auto startErr = cudaEventElapsedTime(&computeStartRelMs, record.startEvent, computeStart);
                auto endErr = cudaEventElapsedTime(&computeEndRelMs, record.startEvent, computeEnd);
                if (copyErr != cudaSuccess || startErr != cudaSuccess || endErr != cudaSuccess) {
                    continue;
                }
                double overlapStart = std::max(0.0, static_cast<double>(computeStartRelMs));
                double overlapEnd = std::min(static_cast<double>(copyMs), static_cast<double>(computeEndRelMs));
                double overlap = std::max(0.0, overlapEnd - overlapStart);
                h2dTotalMs += copyMs;
                selfOverlapMs += overlap;
                ++readyCopies;
                MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] layer=%d request_id=%llu cache=%s h2d_compute_self copy_start_ms=0.000 copy_end_ms=%.3f compute_start_ms=%.3f compute_end_ms=%.3f overlap_ms=%.3f copy_stream_id=%llu compute_stream_id=%llu async_report=1\n",
                          layerIndex,
                          static_cast<unsigned long long>(requestId),
                          record.cacheName.c_str(),
                          copyMs,
                          computeStartRelMs,
                          computeEndRelMs,
                          overlap,
                          record.streamId,
                          computeStreamId);
            }
            MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] layer=%d request_id=%llu compute_gpu_ms=%.3f h2d_ready_copies=%d h2d_pending_copies=%d h2d_total_gpu_ms=%.3f h2d_compute_self_overlap_ms=%.3f stream=%p stream_id=%llu async_report=1\n",
                      layerIndex,
                      static_cast<unsigned long long>(requestId),
                      computeMs,
                      readyCopies,
                      pendingCopies,
                      h2dTotalMs,
                      selfOverlapMs,
                      computeStream,
                      computeStreamId);
        }).detach();
    }

    void reportComputeH2DOverlap(bool finalReport) {
        if (!cudaPrefixProfileH2DEnabled() || !mComputeProfilePending ||
            mComputeStartEvent == nullptr || mComputeEndEvent == nullptr ||
            mComputeProfileReported) {
            return;
        }
        auto computeQuery = cudaEventQuery(mComputeEndEvent);
        if (computeQuery == cudaErrorNotReady) {
            if (!mComputeProfilePendingReported && finalReport) {
                mComputeProfilePendingReported = true;
                MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] layer=%d request_id=%llu compute_gpu_ms=pending stream=%p stream_id=%llu final=%d\n",
                          mComputeProfileLayerIndex,
                          static_cast<unsigned long long>(mComputeProfileRequestId),
                          mComputeProfileStream,
                          mComputeProfileStreamId,
                          finalReport ? 1 : 0);
            }
            return;
        }
        if (computeQuery != cudaSuccess) {
            MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] layer=%d request_id=%llu compute_gpu_ms=query_error:%s stream=%p stream_id=%llu\n",
                      mComputeProfileLayerIndex,
                      static_cast<unsigned long long>(mComputeProfileRequestId),
                      cudaGetErrorString(computeQuery),
                      mComputeProfileStream,
                      mComputeProfileStreamId);
            mComputeProfileReported = true;
            return;
        }

        std::lock_guard<std::mutex> lock(mDevicePrefetchMutex);
        float computeMs = 0.0f;
        auto computeErr = cudaEventElapsedTime(&computeMs, mComputeStartEvent, mComputeEndEvent);
        if (computeErr != cudaSuccess) {
            MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] layer=%d request_id=%llu compute_gpu_ms=elapsed_error:%s stream=%p stream_id=%llu\n",
                      mComputeProfileLayerIndex,
                      static_cast<unsigned long long>(mComputeProfileRequestId),
                      cudaGetErrorString(computeErr),
                      mComputeProfileStream,
                      mComputeProfileStreamId);
            mComputeProfileReported = true;
            return;
        }

        double h2dTotalMs = 0.0;
        double selfOverlapMs = 0.0;
        int readyCopies = 0;
        int pendingCopies = 0;
        for (auto& record : mDevicePrefetchH2DProfiles) {
            if (record.requestId != mComputeProfileRequestId || record.valueEndEvent == nullptr) {
                continue;
            }
            auto copyQuery = cudaEventQuery(record.valueEndEvent);
            if (copyQuery == cudaErrorNotReady) {
                ++pendingCopies;
                continue;
            }
            if (copyQuery != cudaSuccess) {
                continue;
            }
            float copyMs = 0.0f;
            float computeStartRelMs = 0.0f;
            float computeEndRelMs = 0.0f;
            auto copyErr = cudaEventElapsedTime(&copyMs, record.startEvent, record.valueEndEvent);
            auto startErr = cudaEventElapsedTime(&computeStartRelMs, record.startEvent, mComputeStartEvent);
            auto endErr = cudaEventElapsedTime(&computeEndRelMs, record.startEvent, mComputeEndEvent);
            if (copyErr != cudaSuccess || startErr != cudaSuccess || endErr != cudaSuccess) {
                continue;
            }
            double overlapStart = std::max(0.0, static_cast<double>(computeStartRelMs));
            double overlapEnd = std::min(static_cast<double>(copyMs), static_cast<double>(computeEndRelMs));
            double overlap = std::max(0.0, overlapEnd - overlapStart);
            h2dTotalMs += copyMs;
            selfOverlapMs += overlap;
            ++readyCopies;
            MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] layer=%d request_id=%llu cache=%s h2d_compute_self copy_start_ms=0.000 copy_end_ms=%.3f compute_start_ms=%.3f compute_end_ms=%.3f overlap_ms=%.3f copy_stream_id=%llu compute_stream_id=%llu\n",
                      mComputeProfileLayerIndex,
                      static_cast<unsigned long long>(mComputeProfileRequestId),
                      record.cacheName.c_str(),
                      copyMs,
                      computeStartRelMs,
                      computeEndRelMs,
                      overlap,
                      record.streamId,
                      mComputeProfileStreamId);
        }
        MNN_PRINT("[CUDAPrefixAttentionPrefetchProfile] layer=%d request_id=%llu compute_gpu_ms=%.3f h2d_ready_copies=%d h2d_pending_copies=%d h2d_total_gpu_ms=%.3f h2d_compute_self_overlap_ms=%.3f stream=%p stream_id=%llu\n",
                  mComputeProfileLayerIndex,
                  static_cast<unsigned long long>(mComputeProfileRequestId),
                  computeMs,
                  readyCopies,
                  pendingCopies,
                  h2dTotalMs,
                  selfOverlapMs,
                  mComputeProfileStream,
                  mComputeProfileStreamId);
        mComputeProfileReported = true;
    }

    void waitDevicePrefetchTaskComplete() {
        std::unique_lock<std::mutex> lock(mDevicePrefetchMutex);
        while (mDevicePrefetchPending) {
            mDevicePrefetchCondition.wait(lock);
        }
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
            }
        }
        mRegisteredForDevicePrefetch = false;
    }

    static cudaStream_t devicePrefetchStream() {
        static cudaStream_t stream = []() {
            cudaStream_t created = nullptr;
            auto err = cudaStreamCreateWithFlags(&created, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                MNN_ERROR("[Error]: CUDA PrefixAttention failed to create device prefetch stream: %s\n",
                          cudaGetErrorString(err));
                return static_cast<cudaStream_t>(nullptr);
            }
            return created;
        }();
        return stream;
    }

    static std::mutex& registryMutex() {
        static std::mutex mutex;
        return mutex;
    }

    static std::unordered_map<KVMeta*, std::vector<CUDAPrefixAttentionExecution*>>& registry() {
        static std::unordered_map<KVMeta*, std::vector<CUDAPrefixAttentionExecution*>> value;
        return value;
    }

    static std::unordered_map<KVMeta*, uint64_t>& submittedRequests() {
        static std::unordered_map<KVMeta*, uint64_t> value;
        return value;
    }

    int mLayerIndex = -1;
    RuntimeKVBlockTablePlan mPagedKVPlan;
    bool mRegisteredForDevicePrefetch = false;
    std::mutex mDevicePrefetchMutex;
    std::condition_variable mDevicePrefetchCondition;
    bool mDevicePrefetchPending = false;
    bool mDevicePrefetchReady = false;
    uint64_t mDevicePrefetchRequestId = 0;
    cudaEvent_t mDevicePrefetchReadyEvent = nullptr;
    ErrorCode mDevicePrefetchStatus = NO_ERROR;
    int mDevicePrefetchAppendKvSeqLen = 0;
    PrefixPrepareSummary mDevicePrefetchSummary;
    std::vector<H2DProfileRecord> mDevicePrefetchH2DProfiles;
    std::shared_ptr<CudaPrefixComputeProfileEvents> mComputeProfileEvents;
    cudaEvent_t mComputeStartEvent = nullptr;
    cudaEvent_t mComputeEndEvent = nullptr;
    bool mComputeProfilePending = false;
    bool mComputeProfileReported = false;
    bool mComputeProfilePendingReported = false;
    bool mComputeProfileReporterLaunched = false;
    uint64_t mComputeProfileRequestId = 0;
    int mComputeProfileLayerIndex = -1;
    cudaStream_t mComputeProfileStream = nullptr;
    unsigned long long mComputeProfileStreamId = 0;
};

class CUDAPrefixAttentionCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        bool kvCache = true;
        int layerIndex = -1;
        if (op->main_type() == OpParameter_AttentionParam) {
            auto param = op->main_as_AttentionParam();
            if (param != nullptr) {
                kvCache = param->kv_cache();
                layerIndex = param->layer_index();
            }
        }
        return new CUDAPrefixAttentionExecution(backend, kvCache, layerIndex);
    }
};

static CUDACreatorRegister<CUDAPrefixAttentionCreator> __init_prefix_attention(OpType_PrefixAttention);

namespace {
struct CUDAPrefixDevicePrefetchSubmitterRegister {
    CUDAPrefixDevicePrefetchSubmitterRegister() {
        registerPrefixDevicePrefetchSubmitter(&CUDAPrefixAttentionExecution::submitAsyncDevicePrefetch);
    }
};

static CUDAPrefixDevicePrefetchSubmitterRegister __init_prefix_device_prefetch_submitter;
} // namespace

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

} // namespace CUDA
} // namespace MNN
