//
//  PrefixKVHostCache.hpp
//  MNN
//
//  Small host-side file cache and prefetch helper for PrefixAttention direct_segments.
//

#ifndef PrefixKVHostCache_hpp
#define PrefixKVHostCache_hpp

#include "KVMeta.hpp"
#include "MNNFileUtils.h"
#include "PrefixCachePath.hpp"

#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace MNN {

struct PrefixKVHostCacheRead {
    std::shared_ptr<std::vector<uint8_t>> bytes;
    bool ok = false;
    bool host_cache_hit = false;
    double prefetch_wait_ms = 0.0;
    double disk_read_ms = 0.0;
    size_t file_size = 0;
    std::string error;
};

namespace detail {

struct PrefixKVHostCacheEntry {
    std::shared_ptr<std::vector<uint8_t>> bytes;
    bool ok = false;
    double disk_read_ms = 0.0;
    size_t file_size = 0;
    std::string error;
};

inline double prefixKVNowMs() {
    using Clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

inline bool prefixKVFileIdentity(const std::string& path, std::string& identity, size_t* fileSize = nullptr) {
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
    WIN32_FILE_ATTRIBUTE_DATA info;
    if (!GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &info)) {
        return false;
    }
    uint64_t size = (static_cast<uint64_t>(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
    uint64_t mtime = (static_cast<uint64_t>(info.ftLastWriteTime.dwHighDateTime) << 32) |
                     info.ftLastWriteTime.dwLowDateTime;
#else
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        return false;
    }
    size_t size = static_cast<size_t>(info.st_size);
    int64_t mtime = static_cast<int64_t>(info.st_mtime);
#endif
    if (fileSize != nullptr) {
        *fileSize = static_cast<size_t>(size);
    }
    identity = path + "|" + std::to_string(static_cast<size_t>(size)) + "|" + std::to_string(mtime);
    return true;
}

inline std::shared_ptr<PrefixKVHostCacheEntry> prefixKVReadFileNow(const std::string& path) {
    auto entry = std::make_shared<PrefixKVHostCacheEntry>();
    double start = prefixKVNowMs();
    auto fd = MNNOpenFile(path.c_str(), MNN_FILE_READ);
    if (fd == INVALID_FILE) {
        entry->error = "failed to open prefix KV file: " + path;
        entry->disk_read_ms = prefixKVNowMs() - start;
        return entry;
    }
    size_t size = MNNGetFileSize(fd);
    if (size == INVALID_SIZE || size == 0) {
        MNNCloseFile(fd);
        entry->error = "invalid prefix KV file size: " + path;
        entry->disk_read_ms = prefixKVNowMs() - start;
        return entry;
    }
    entry->bytes = std::make_shared<std::vector<uint8_t>>(size);
    size_t readSize = MNNReadFile(fd, entry->bytes->data(), size);
    MNNCloseFile(fd);
    entry->file_size = size;
    entry->disk_read_ms = prefixKVNowMs() - start;
    if (readSize != size) {
        entry->bytes.reset();
        entry->error = "short read for prefix KV file: " + path;
        return entry;
    }
    entry->ok = true;
    return entry;
}

class PrefixKVHostCacheWorkerPool {
public:
    static PrefixKVHostCacheWorkerPool& get() {
        static PrefixKVHostCacheWorkerPool pool(threadCountFromEnv());
        return pool;
    }

    void post(std::function<void()>&& task) {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mTasks.emplace(std::move(task));
        }
        mCondition.notify_one();
    }

    int workerCount() const {
        return mWorkerCount;
    }

private:
    static int threadCountFromEnv() {
        const char* value = std::getenv("MNN_PREFIX_HOST_CACHE_THREADS");
        if (value == nullptr || value[0] == '\0') {
            return 4;
        }
        int parsed = std::atoi(value);
        return (parsed == 4 || parsed == 8 || parsed == 16) ? parsed : 4;
    }

    explicit PrefixKVHostCacheWorkerPool(int workerCount) : mWorkerCount(workerCount) {
        for (int i = 0; i < mWorkerCount; ++i) {
            mWorkers.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
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
                    task();
                }
            });
        }
    }

    ~PrefixKVHostCacheWorkerPool() {
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

    int mWorkerCount = 4;
    std::vector<std::thread> mWorkers;
    std::queue<std::function<void()>> mTasks;
    std::mutex mMutex;
    std::condition_variable mCondition;
    bool mStop = false;
};

inline std::shared_future<std::shared_ptr<PrefixKVHostCacheEntry>> prefixKVSubmitReadFile(const std::string& path) {
    auto promise = std::make_shared<std::promise<std::shared_ptr<PrefixKVHostCacheEntry>>>();
    auto future = promise->get_future().share();
    PrefixKVHostCacheWorkerPool::get().post([path, promise]() {
        promise->set_value(prefixKVReadFileNow(path));
    });
    return future;
}

} // namespace detail

class PrefixKVHostCache {
public:
    static PrefixKVHostCache& get() {
        static PrefixKVHostCache cache;
        return cache;
    }

    void prefetchFile(const std::string& path) {
        std::string identity;
        if (!detail::prefixKVFileIdentity(path, identity)) {
            return;
        }
        std::lock_guard<std::mutex> lock(mMutex);
        if (mEntries.find(identity) != mEntries.end()) {
            return;
        }
        mEntries.emplace(identity, detail::prefixKVSubmitReadFile(path));
    }

    PrefixKVHostCacheRead readFile(const std::string& path) {
        PrefixKVHostCacheRead result;
        std::string identity;
        size_t fileSize = 0;
        if (!detail::prefixKVFileIdentity(path, identity, &fileSize)) {
            result.error = "prefix KV file does not exist: " + path;
            return result;
        }
        result.file_size = fileSize;

        std::shared_future<std::shared_ptr<detail::PrefixKVHostCacheEntry>> future;
        {
            std::lock_guard<std::mutex> lock(mMutex);
            auto iter = mEntries.find(identity);
            if (iter != mEntries.end()) {
                future = iter->second;
                result.host_cache_hit =
                    future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
            } else {
                future = detail::prefixKVSubmitReadFile(path);
                mEntries.emplace(identity, future);
            }
        }

        double waitStart = detail::prefixKVNowMs();
        auto entry = future.get();
        result.prefetch_wait_ms = detail::prefixKVNowMs() - waitStart;
        if (entry != nullptr) {
            result.ok = entry->ok;
            result.bytes = entry->bytes;
            result.disk_read_ms = result.host_cache_hit ? 0.0 : entry->disk_read_ms;
            result.error = entry->error;
            result.file_size = entry->file_size;
        }
        if (!result.ok) {
            std::lock_guard<std::mutex> lock(mMutex);
            mEntries.erase(identity);
        }
        return result;
    }

private:
    std::mutex mMutex;
    std::unordered_map<std::string, std::shared_future<std::shared_ptr<detail::PrefixKVHostCacheEntry>>> mEntries;
};

inline PrefixKVHostCacheRead readPrefixKVHostCacheFile(const std::string& path) {
    return PrefixKVHostCache::get().readFile(path);
}

inline int prefixKVHostCacheWorkerCount() {
    return detail::PrefixKVHostCacheWorkerPool::get().workerCount();
}

inline void prefetchPrefixKVHostCacheFiles(const std::string& prefixCacheDir,
                                           int layerCount,
                                           const std::vector<KVMeta::PrefixSegment>& segments) {
    if (layerCount <= 0 || prefixCacheDir.empty() || segments.empty()) {
        return;
    }
    auto& cache = PrefixKVHostCache::get();
    for (const auto& segment : segments) {
        if (segment.cache_name.empty() || segment.backend.empty()) {
            continue;
        }
        for (int layer = 0; layer < layerCount; ++layer) {
            auto base = prefixCacheLayerBase(prefixCacheDir, segment.backend, segment.cache_name, layer);
            cache.prefetchFile(base + ".k");
            cache.prefetchFile(base + ".v");
        }
    }
}

} // namespace MNN

#endif // PrefixKVHostCache_hpp
