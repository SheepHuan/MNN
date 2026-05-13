//
//  SoftmaxBufExecution.cpp
//  MNN
//
//  Created by MNN on 2024/04/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "backend/opencl/execution/buffer/AttentionBufExecution.hpp"
#include "core/PrefixCachePath.hpp"
#include "half.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

namespace MNN {
namespace OpenCL {

namespace {

static float readOpenCLNativeScalar(const uint8_t* base, size_t index, int bytes) {
    if (bytes == 2) {
        uint16_t raw = 0;
        ::memcpy(&raw, base + index * 2, sizeof(raw));
        half_float::half value;
        ::memcpy(&value, &raw, sizeof(raw));
        return static_cast<float>(value);
    }
    float value = 0.0f;
    ::memcpy(&value, base + index * 4, sizeof(value));
    return value;
}

static void writeOpenCLNativeScalar(uint8_t* base, size_t index, int bytes, float value) {
    if (bytes == 2) {
        half_float::half halfValue(value);
        uint16_t raw = 0;
        ::memcpy(&raw, &halfValue, sizeof(raw));
        ::memcpy(base + index * 2, &raw, sizeof(raw));
        return;
    }
    ::memcpy(base + index * 4, &value, sizeof(value));
}

static cl::Buffer* createOpenCLKVBuffer(const cl::Context& context, size_t bufferSize,
                                        bool useHostPtrPages,
                                        std::vector<uint8_t>& hostStorage,
                                        cl_int& res) {
    if (useHostPtrPages) {
        hostStorage.resize(bufferSize);
        return new cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                              bufferSize, hostStorage.data(), &res);
    }
    hostStorage.clear();
    return new cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                          bufferSize, nullptr, &res);
}

} // namespace

std::recursive_mutex& openCLPrefixQueueMutex() {
    static std::recursive_mutex mutex;
    return mutex;
}

KVCacheCLManager::KVCacheCLManager(Backend *backend, bool kv_cahce) : mKVCache(kv_cahce){
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
}

void KVCacheCLManager::allocKVCache(const KVMeta* meta, int seqlen) {
    if (!mKVCache) {
        return;
    }
    if (meta != nullptr && meta->file_flag == KVMeta::PendingWrite) {
        mUseHostPtrPages = false;
        mPagedActive = false;
        mPagedDecodeStarted = false;
        mPagedLogicalLength = 0;
        mPagedPhysicalLength = 0;
        discardPagedTablesForRewrite();
        mPagedTokenTableHost.clear();
        mPagedRopeTableHost.clear();
    }
    mPastLength = meta != nullptr ? meta->previous : 0;
    if(mOpenCLBackend->getPrecision() != BackendConfig::Precision_High){
        mByte = 2;
    }
    reallocKVCache(meta, seqlen, false);
}

bool KVCacheCLManager::reallocKVCache(const KVMeta* meta, int seqlen, bool isExecute) {
    std::lock_guard<std::recursive_mutex> queueLock(openCLPrefixQueueMutex());
    if (!mKVCache) {
        return false;
    }
    if (isExecute && mPagedActive && meta != nullptr && meta->remove > 0) {
        mPagedActive = false;
        mPagedDecodeStarted = false;
        mPagedLogicalLength = 0;
        mPagedPhysicalLength = 0;
    }
    int kvSeqlen = meta->previous + seqlen - meta->remove + meta->computeReverseSize();
    int start = mPastLength - meta->remove;
    cl_int res;
    auto& queue = mOpenCLBackend->getOpenCLRuntime()->defaultCommandQueue();

    bool needHostPtrTransition = mUseHostPtrPages && mPastKey != nullptr && mPastValue != nullptr &&
        (mPastKeyHostStorage.empty() || mPastValueHostStorage.empty());
    bool needDefaultBufferTransition = !mUseHostPtrPages &&
        (!mPastKeyHostStorage.empty() || !mPastValueHostStorage.empty());
    // latest length larger than maxLen, or PrefixAttention switched this manager to host-ptr pages.
    if(kvSeqlen > mMaxLength || needHostPtrTransition || needDefaultBufferTransition){
        int copylen = mPagedActive ? mPagedPhysicalLength :
            (mPastLength - meta->remove + meta->computeReverseSize());
        bool needCopy = copylen > 0 && mPastKey != nullptr && mPastValue != nullptr && mMaxLength > 0;

        size_t oldSize = mKvNumHead * UP_DIV(mMaxLength, 4) * mHeadDim * 4 * mByte;
        size_t oldMaxlen = ROUND_UP(mMaxLength, 4);
        if (kvSeqlen > mMaxLength) {
            mMaxLength = kvSeqlen + mExpandChunk;
        }
        size_t newMaxlen = ROUND_UP(mMaxLength, 4);
        size_t bufferSize = UP_DIV(mMaxLength, 4) * mKvNumHead * mHeadDim * 4 * mByte;
        std::vector<uint8_t> newKeyHostStorage;
        std::vector<uint8_t> newValueHostStorage;
        // past_key: [1, numhead, headdim, maxlen]
        auto newKey = createOpenCLKVBuffer(mOpenCLBackend->getOpenCLRuntime()->context(), bufferSize,
                                           mUseHostPtrPages, newKeyHostStorage, res);
        if (res != CL_SUCCESS || newKey == nullptr) {
            MNN_ERROR("[Error]: KVCacheCLManager failed to allocate OpenCL key cache, host_ptr_pages=%d err=%d\n",
                      mUseHostPtrPages ? 1 : 0, res);
            return false;
        }
        // past_value: [1, numhead, maxlen, headdim]
        auto newValue = createOpenCLKVBuffer(mOpenCLBackend->getOpenCLRuntime()->context(), bufferSize,
                                             mUseHostPtrPages, newValueHostStorage, res);
        if (res != CL_SUCCESS || newValue == nullptr) {
            delete newKey;
            MNN_ERROR("[Error]: KVCacheCLManager failed to allocate OpenCL value cache, host_ptr_pages=%d err=%d\n",
                      mUseHostPtrPages ? 1 : 0, res);
            return false;
        }

        if (needCopy) {
            queue.finish();
        }
        if(needCopy){
            // copy key
            {
                size_t oldMaxlenSize = oldMaxlen * mByte;
                size_t newMaxlenSize = newMaxlen * mByte;
                char* newKeyPtr = mUseHostPtrPages ? reinterpret_cast<char*>(newKeyHostStorage.data()) :
                    (char*)queue.enqueueMapBuffer(*newKey, true, CL_MAP_WRITE, 0, bufferSize, nullptr, nullptr, &res);
                char* keyPtr = (!mPastKeyHostStorage.empty()) ? reinterpret_cast<char*>(mPastKeyHostStorage.data()) :
                    (char*)queue.enqueueMapBuffer(*mPastKey.get(), true, CL_MAP_READ, 0, oldSize, nullptr, nullptr, &res);
                if(newKeyPtr != nullptr && keyPtr != nullptr && res == CL_SUCCESS){
                    for(int i = 0; i < mKvNumHead * mHeadDim; ++i){
                        ::memcpy(newKeyPtr + i * newMaxlenSize, keyPtr + i * oldMaxlenSize, oldMaxlenSize);
                    }
                }else{
                    MNN_ERROR("[Error]: KVCacheCLManager failed to map OpenCL key cache during realloc "
                              "new_ptr=%p old_ptr=%p err=%d host_ptr_pages=%d\n",
                              newKeyPtr, keyPtr, res, mUseHostPtrPages ? 1 : 0);
                    if (!mUseHostPtrPages && newKeyPtr != nullptr) {
                        queue.enqueueUnmapMemObject(*newKey, newKeyPtr);
                    }
                    if (mPastKeyHostStorage.empty() && keyPtr != nullptr) {
                        queue.enqueueUnmapMemObject(*mPastKey.get(), keyPtr);
                    }
                    queue.finish();
                    delete newKey;
                    delete newValue;
                    return false;
                }
                if (!mUseHostPtrPages) {
                    queue.enqueueUnmapMemObject(*newKey, newKeyPtr);
                }
                if (mPastKeyHostStorage.empty()) {
                    queue.enqueueUnmapMemObject(*mPastKey.get(), keyPtr);
                }
            }

            // copy value
            {
                char* newValuePtr = mUseHostPtrPages ? reinterpret_cast<char*>(newValueHostStorage.data()) :
                    (char*)queue.enqueueMapBuffer(*newValue, true, CL_MAP_WRITE, 0, bufferSize, nullptr, nullptr, &res);
                char* valuePtr = (!mPastValueHostStorage.empty()) ? reinterpret_cast<char*>(mPastValueHostStorage.data()) :
                    (char*)queue.enqueueMapBuffer(*mPastValue.get(), true, CL_MAP_READ, 0, oldSize, nullptr, nullptr, &res);
                if(newValuePtr != nullptr && valuePtr != nullptr && res == CL_SUCCESS){
                    for(int i = 0; i < mKvNumHead; ++i){
                        for(int j = 0; j < copylen; ++j){
                            ::memcpy(newValuePtr + (i * newMaxlen + j) * mHeadDim * mByte, valuePtr + (i * oldMaxlen + j) * mHeadDim * mByte, mHeadDim * mByte);
                        }
                    }
                }else{
                    MNN_ERROR("[Error]: KVCacheCLManager failed to map OpenCL value cache during realloc "
                              "new_ptr=%p old_ptr=%p err=%d host_ptr_pages=%d\n",
                              newValuePtr, valuePtr, res, mUseHostPtrPages ? 1 : 0);
                    if (!mUseHostPtrPages && newValuePtr != nullptr) {
                        queue.enqueueUnmapMemObject(*newValue, newValuePtr);
                    }
                    if (mPastValueHostStorage.empty() && valuePtr != nullptr) {
                        queue.enqueueUnmapMemObject(*mPastValue.get(), valuePtr);
                    }
                    queue.finish();
                    delete newKey;
                    delete newValue;
                    return false;
                }
                if (!mUseHostPtrPages) {
                    queue.enqueueUnmapMemObject(*newValue, newValuePtr);
                }
                if (mPastValueHostStorage.empty()) {
                    queue.enqueueUnmapMemObject(*mPastValue.get(), valuePtr);
                }
            }
            queue.finish();
        }
        mPastKey.reset(newKey);
        mPastValue.reset(newValue);
        if (mUseHostPtrPages) {
            mPastKeyHostStorage = std::move(newKeyHostStorage);
            mPastValueHostStorage = std::move(newValueHostStorage);
        } else {
            mPastKeyHostStorage.clear();
            mPastValueHostStorage.clear();
        }
        // resize phase don't update mPastLength value, excute phase will update it
        if(isExecute){
            mPastLength = start;
        }
    }

    // Remove
    // resize phase don't remove kvcache, excute phase will do it
    if(isExecute){
        if (0 == meta->n_reserve) {
            mPastLength = start;
            return true;
        }

        size_t pastkvSize = mKvNumHead * UP_DIV(mMaxLength, 4) * mHeadDim * 4 * mByte;
        queue.finish();
        char *keyPtr = (char*)queue.enqueueMapBuffer(*mPastKey.get(), true, CL_MAP_READ | CL_MAP_WRITE, 0, pastkvSize, nullptr, nullptr, &res);
        char *valuePtr = (char*)queue.enqueueMapBuffer(*mPastValue.get(), true, CL_MAP_READ | CL_MAP_WRITE, 0, pastkvSize, nullptr, nullptr, &res);
        if (keyPtr == nullptr || valuePtr == nullptr || res != CL_SUCCESS) {
            MNN_ERROR("[Error]: KVCacheCLManager failed to map OpenCL KV cache for reserve/remove "
                      "key_ptr=%p value_ptr=%p err=%d pastkv_size=%zu\n",
                      keyPtr, valuePtr, res, pastkvSize);
            if (keyPtr != nullptr) {
                queue.enqueueUnmapMemObject(*mPastKey.get(), keyPtr);
            }
            if (valuePtr != nullptr) {
                queue.enqueueUnmapMemObject(*mPastValue.get(), valuePtr);
            }
            queue.finish();
            return false;
        }

        // TODO: need to ensure reserve info is sorted
        for (int n = 0; n < meta->n_reserve; ++n) {
            auto begin = meta->reserve[2 * n];
            auto length = meta->reserve[2 * n + 1];
            // past_key   : [mKvNumHead, mHeadDim, mMaxLength]
            // past_value : [mKvNumHead, mMaxLength, mHeadDim]

            auto copySrcIndex = start + begin;
            auto copyDstIndex = start;
            for(int i = 0; i <  mKvNumHead * mHeadDim; i++) {
                ::memcpy(keyPtr + (i * mMaxLength + copyDstIndex) * mByte, keyPtr + (i * mMaxLength + copySrcIndex) * mByte, length * mByte);
            }
            for(int i = 0; i <  mKvNumHead; i++) {
                for(int j = 0; j < length; j++) {
                    ::memcpy(valuePtr + (i * mMaxLength + copyDstIndex + j) * mHeadDim * mByte, valuePtr + (i * mMaxLength + copySrcIndex + j) * mHeadDim * mByte, mHeadDim * mByte);
                }
            }
            start += length;
        }
        mPastLength = (int)start;
        queue.enqueueUnmapMemObject(*mPastKey.get(), keyPtr);
        queue.enqueueUnmapMemObject(*mPastValue.get(), valuePtr);
        queue.finish();
    }
    return true;
}

bool KVCacheCLManager::ensureCapacity(int requiredTotal, bool isExecute) {
    std::lock_guard<std::recursive_mutex> queueLock(openCLPrefixQueueMutex());
    (void)isExecute;
    if (!mKVCache) {
        return false;
    }
    if (mOpenCLBackend->getPrecision() != BackendConfig::Precision_High) {
        mByte = 2;
    }
    bool needHostPtrTransition = mUseHostPtrPages && mPastKey != nullptr && mPastValue != nullptr &&
        (mPastKeyHostStorage.empty() || mPastValueHostStorage.empty());
    if (requiredTotal <= mMaxLength && !needHostPtrTransition) {
        return true;
    }

    cl_int res = CL_SUCCESS;
    auto& queue = mOpenCLBackend->getOpenCLRuntime()->defaultCommandQueue();
    int copylen = mPagedActive ? mPagedPhysicalLength : mPastLength;
    bool needCopy = copylen > 0 && mPastKey != nullptr && mPastValue != nullptr;
    size_t oldSize = mKvNumHead * UP_DIV(mMaxLength, 4) * mHeadDim * 4 * mByte;
    size_t oldMaxlen = ROUND_UP(mMaxLength, 4);
    if (requiredTotal > mMaxLength) {
        mMaxLength = requiredTotal + mExpandChunk;
    }
    size_t newMaxlen = ROUND_UP(mMaxLength, 4);
    size_t bufferSize = UP_DIV(mMaxLength, 4) * mKvNumHead * mHeadDim * 4 * mByte;
    std::vector<uint8_t> newKeyHostStorage;
    std::vector<uint8_t> newValueHostStorage;
    auto newKey = createOpenCLKVBuffer(mOpenCLBackend->getOpenCLRuntime()->context(), bufferSize,
                                       mUseHostPtrPages, newKeyHostStorage, res);
    if (res != CL_SUCCESS || newKey == nullptr) {
        MNN_ERROR("[Error]: KVCacheCLManager failed to allocate OpenCL paged key cache, host_ptr_pages=%d err=%d\n",
                  mUseHostPtrPages ? 1 : 0, res);
        return false;
    }
    auto newValue = createOpenCLKVBuffer(mOpenCLBackend->getOpenCLRuntime()->context(), bufferSize,
                                         mUseHostPtrPages, newValueHostStorage, res);
    if (res != CL_SUCCESS || newValue == nullptr) {
        delete newKey;
        MNN_ERROR("[Error]: KVCacheCLManager failed to allocate OpenCL paged value cache, host_ptr_pages=%d err=%d\n",
                  mUseHostPtrPages ? 1 : 0, res);
        return false;
    }

    if (needCopy) {
        queue.finish();
    }
    if (needCopy) {
        size_t oldMaxlenSize = oldMaxlen * mByte;
        size_t newMaxlenSize = newMaxlen * mByte;
        char* newKeyPtr = mUseHostPtrPages ? reinterpret_cast<char*>(newKeyHostStorage.data()) :
            (char*)queue.enqueueMapBuffer(*newKey, true, CL_MAP_WRITE, 0, bufferSize, nullptr, nullptr, &res);
        char* keyPtr = (!mPastKeyHostStorage.empty()) ? reinterpret_cast<char*>(mPastKeyHostStorage.data()) :
            (char*)queue.enqueueMapBuffer(*mPastKey.get(), true, CL_MAP_READ, 0, oldSize, nullptr, nullptr, &res);
        if (newKeyPtr != nullptr && keyPtr != nullptr && res == CL_SUCCESS) {
            for (int i = 0; i < mKvNumHead * mHeadDim; ++i) {
                ::memcpy(newKeyPtr + i * newMaxlenSize, keyPtr + i * oldMaxlenSize, copylen * mByte);
            }
        } else {
            MNN_ERROR("[Error]: KVCacheCLManager failed to map OpenCL key cache in ensureCapacity "
                      "new_ptr=%p old_ptr=%p err=%d host_ptr_pages=%d required_total=%d max_length=%d\n",
                      newKeyPtr, keyPtr, res, mUseHostPtrPages ? 1 : 0, requiredTotal, mMaxLength);
            if (!mUseHostPtrPages && newKeyPtr != nullptr) {
                queue.enqueueUnmapMemObject(*newKey, newKeyPtr);
            }
            if (mPastKeyHostStorage.empty() && keyPtr != nullptr) {
                queue.enqueueUnmapMemObject(*mPastKey.get(), keyPtr);
            }
            queue.finish();
            delete newKey;
            delete newValue;
            return false;
        }
        if (!mUseHostPtrPages) {
            queue.enqueueUnmapMemObject(*newKey, newKeyPtr);
        }
        if (mPastKeyHostStorage.empty()) {
            queue.enqueueUnmapMemObject(*mPastKey.get(), keyPtr);
        }

        char* newValuePtr = mUseHostPtrPages ? reinterpret_cast<char*>(newValueHostStorage.data()) :
            (char*)queue.enqueueMapBuffer(*newValue, true, CL_MAP_WRITE, 0, bufferSize, nullptr, nullptr, &res);
        char* valuePtr = (!mPastValueHostStorage.empty()) ? reinterpret_cast<char*>(mPastValueHostStorage.data()) :
            (char*)queue.enqueueMapBuffer(*mPastValue.get(), true, CL_MAP_READ, 0, oldSize, nullptr, nullptr, &res);
        if (newValuePtr != nullptr && valuePtr != nullptr && res == CL_SUCCESS) {
            for (int i = 0; i < mKvNumHead; ++i) {
                for (int j = 0; j < copylen; ++j) {
                    ::memcpy(newValuePtr + (i * newMaxlen + j) * mHeadDim * mByte,
                             valuePtr + (i * oldMaxlen + j) * mHeadDim * mByte, mHeadDim * mByte);
                }
            }
        } else {
            MNN_ERROR("[Error]: KVCacheCLManager failed to map OpenCL value cache in ensureCapacity "
                      "new_ptr=%p old_ptr=%p err=%d host_ptr_pages=%d required_total=%d max_length=%d\n",
                      newValuePtr, valuePtr, res, mUseHostPtrPages ? 1 : 0, requiredTotal, mMaxLength);
            if (!mUseHostPtrPages && newValuePtr != nullptr) {
                queue.enqueueUnmapMemObject(*newValue, newValuePtr);
            }
            if (mPastValueHostStorage.empty() && valuePtr != nullptr) {
                queue.enqueueUnmapMemObject(*mPastValue.get(), valuePtr);
            }
            queue.finish();
            delete newKey;
            delete newValue;
            return false;
        }
        if (!mUseHostPtrPages) {
            queue.enqueueUnmapMemObject(*newValue, newValuePtr);
        }
        if (mPastValueHostStorage.empty()) {
            queue.enqueueUnmapMemObject(*mPastValue.get(), valuePtr);
        }
        queue.finish();
    }

    mPastKey.reset(newKey);
    mPastValue.reset(newValue);
    if (mUseHostPtrPages) {
        mPastKeyHostStorage = std::move(newKeyHostStorage);
        mPastValueHostStorage = std::move(newValueHostStorage);
    } else {
        mPastKeyHostStorage.clear();
        mPastValueHostStorage.clear();
    }
    return true;
}

bool KVCacheCLManager::ensurePagedCapacity(int requiredPhysicalTotal, int logicalTableLength) {
    std::lock_guard<std::recursive_mutex> queueLock(openCLPrefixQueueMutex());
    if (!mKVCache) {
        return false;
    }
    if (requiredPhysicalTotal <= 0 || logicalTableLength <= 0) {
        return false;
    }
    if (!ensureCapacity(requiredPhysicalTotal, true)) {
        return false;
    }
    if (logicalTableLength > mPagedTableCapacity) {
        cl_int res = CL_SUCCESS;
        auto oldTokenTable = mPagedTokenTable;
        auto oldRopeTable = mPagedRopeTable;
        int oldCapacity = mPagedTableCapacity;
        int newCapacity = ROUND_UP(logicalTableLength + mExpandChunk, 64);
        auto tokenTable = std::shared_ptr<cl::Buffer>(new cl::Buffer(
            mOpenCLBackend->getOpenCLRuntime()->context(),
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            static_cast<size_t>(newCapacity) * sizeof(int), nullptr, &res));
        if (res != CL_SUCCESS || tokenTable == nullptr) {
            MNN_ERROR("[Error]: KVCacheCLManager failed to allocate OpenCL paged token table: %d\n", res);
            return false;
        }
        auto ropeTable = std::shared_ptr<cl::Buffer>(new cl::Buffer(
            mOpenCLBackend->getOpenCLRuntime()->context(),
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            static_cast<size_t>(newCapacity) * sizeof(int), nullptr, &res));
        if (res != CL_SUCCESS || ropeTable == nullptr) {
            MNN_ERROR("[Error]: KVCacheCLManager failed to allocate OpenCL paged rope table: %d\n", res);
            return false;
        }
        mPagedTokenTable = tokenTable;
        mPagedRopeTable = ropeTable;
        bool hadHostTables = mPagedTokenTableHost.size() >= static_cast<size_t>(oldCapacity) &&
                             mPagedRopeTableHost.size() >= static_cast<size_t>(oldCapacity);
        mPagedTokenTableHost.resize(newCapacity, 0);
        mPagedRopeTableHost.resize(newCapacity, 0);
        if (oldCapacity > 0 && hadHostTables) {
            int copyCount = std::min(oldCapacity, logicalTableLength);
            if (!uploadPagedTableRange(0, copyCount)) {
                return false;
            }
        } else if (oldTokenTable != nullptr && oldRopeTable != nullptr && oldCapacity > 0) {
            size_t copyBytes = static_cast<size_t>(std::min(oldCapacity, logicalTableLength)) * sizeof(int);
            auto& queue = mOpenCLBackend->getOpenCLRuntime()->defaultCommandQueue();
            res |= queue.enqueueCopyBuffer(*oldTokenTable, *mPagedTokenTable, 0, 0, copyBytes);
            res |= queue.enqueueCopyBuffer(*oldRopeTable, *mPagedRopeTable, 0, 0, copyBytes);
            res |= queue.finish();
            if (res != CL_SUCCESS) {
                MNN_ERROR("[Error]: KVCacheCLManager failed to copy OpenCL paged tables: %d\n", res);
                return false;
            }
        }
        mPagedTableCapacity = newCapacity;
    }
    return true;
}

void KVCacheCLManager::discardPagedTablesForRewrite() {
    mPagedTokenTable.reset();
    mPagedRopeTable.reset();
    mPagedTableCapacity = 0;
}

bool KVCacheCLManager::updatePagedTableHost(int logicalStart, const std::vector<int>& tokenTable,
                                            const std::vector<int>& ropeTable) {
    if (logicalStart < 0 || tokenTable.size() != ropeTable.size()) {
        return false;
    }
    size_t end = static_cast<size_t>(logicalStart) + tokenTable.size();
    if (end > mPagedTokenTableHost.size() || end > mPagedRopeTableHost.size()) {
        if (end > static_cast<size_t>(mPagedTableCapacity)) {
            return false;
        }
        mPagedTokenTableHost.resize(mPagedTableCapacity, 0);
        mPagedRopeTableHost.resize(mPagedTableCapacity, 0);
    }
    std::copy(tokenTable.begin(), tokenTable.end(), mPagedTokenTableHost.begin() + logicalStart);
    std::copy(ropeTable.begin(), ropeTable.end(), mPagedRopeTableHost.begin() + logicalStart);
    return true;
}

bool KVCacheCLManager::uploadPagedTableRange(int logicalStart, int tokenCount) {
    std::lock_guard<std::recursive_mutex> queueLock(openCLPrefixQueueMutex());
    if (logicalStart < 0 || tokenCount < 0 || tokenCount == 0) {
        return tokenCount == 0;
    }
    if (mPagedTokenTable == nullptr || mPagedRopeTable == nullptr) {
        return false;
    }
    size_t end = static_cast<size_t>(logicalStart) + static_cast<size_t>(tokenCount);
    if (end > mPagedTokenTableHost.size() || end > mPagedRopeTableHost.size()) {
        return false;
    }
    auto& queue = mOpenCLBackend->getOpenCLRuntime()->defaultCommandQueue();
    queue.finish();
    cl_int res = CL_SUCCESS;
    res |= queue.enqueueWriteBuffer(*mPagedTokenTable, CL_TRUE,
                                    static_cast<size_t>(logicalStart) * sizeof(int),
                                    static_cast<size_t>(tokenCount) * sizeof(int),
                                    mPagedTokenTableHost.data() + logicalStart);
    res |= queue.enqueueWriteBuffer(*mPagedRopeTable, CL_TRUE,
                                    static_cast<size_t>(logicalStart) * sizeof(int),
                                    static_cast<size_t>(tokenCount) * sizeof(int),
                                    mPagedRopeTableHost.data() + logicalStart);
    if (res != CL_SUCCESS) {
        MNN_ERROR("[Error]: KVCacheCLManager failed to upload OpenCL paged host table: %d "
                  "logical_start=%d token_count=%d table_capacity=%d\n",
                  res, logicalStart, tokenCount, mPagedTableCapacity);
        return false;
    }
    queue.finish();
    return true;
}

void KVCacheCLManager::setPagedState(int logicalLength, int physicalLength, int pageSize,
                                     int ropeDim, float ropeTheta) {
    mPagedActive = true;
    mPagedLogicalLength = logicalLength;
    mPagedPhysicalLength = physicalLength;
    mPagedPageSize = pageSize > 0 ? pageSize : mPagedPageSize;
    mPagedRopeDim = ropeDim;
    mPagedRopeTheta = ropeTheta;
    mPagedDecodeStarted = false;
}

bool KVCacheCLManager::ensurePagedStateForExistingHistory(int logicalLength, int physicalLength,
                                                          int pageSize, int ropeDim,
                                                          float ropeTheta) {
    if (mPagedActive) {
        return true;
    }
    int safePageSize = pageSize > 0 ? pageSize : mPagedPageSize;
    if (safePageSize <= 0) {
        safePageSize = 64;
    }
    int safePhysicalLength = std::max({
        mPagedPhysicalLength,
        physicalLength,
        ROUND_UP(mMaxLength, safePageSize)
    });
    if (!ensurePagedCapacity(safePhysicalLength, std::max(logicalLength, 1))) {
        return false;
    }
    if (mPastKey == nullptr || mPastValue == nullptr ||
        mPagedTokenTable == nullptr || mPagedRopeTable == nullptr) {
        return false;
    }
    if (logicalLength > 0) {
        std::vector<int> tokenTable(logicalLength);
        std::vector<int> ropeTable(logicalLength, 0);
        for (int i = 0; i < logicalLength; ++i) {
            tokenTable[i] = i;
        }
        if (!updatePagedTableHost(0, tokenTable, ropeTable) ||
            !uploadPagedTableRange(0, logicalLength)) {
            return false;
        }
    }
    setPagedState(logicalLength, safePhysicalLength, safePageSize,
                  ropeDim, ropeTheta > 0.0f ? ropeTheta : 10000.0f);
    return true;
}

bool KVCacheCLManager::ensurePagedAppendTable(int logicalStart, int tokenCount, bool isolateSourceStart) {
    if (!mPagedActive || tokenCount <= 0) {
        return true;
    }
    if (isolateSourceStart) {
        mPagedDecodeStarted = false;
    }
    int physicalStart = mPagedPhysicalLength;
    if (isolateSourceStart && physicalStart > 0 && mPagedPageSize > 0) {
        physicalStart = ROUND_UP(physicalStart, mPagedPageSize);
    }
    int requiredPhysical = physicalStart + tokenCount;
    int requiredLogical = logicalStart + tokenCount;
    if (!ensurePagedCapacity(requiredPhysical, requiredLogical)) {
        return false;
    }

    std::vector<int> tokenTable(tokenCount);
    std::vector<int> ropeTable(tokenCount, 0);
    for (int i = 0; i < tokenCount; ++i) {
        tokenTable[i] = physicalStart + i;
    }
    auto& queue = mOpenCLBackend->getOpenCLRuntime()->defaultCommandQueue();
    cl_int res = CL_SUCCESS;
    if (!updatePagedTableHost(logicalStart, tokenTable, ropeTable)) {
        return false;
    }
    res |= queue.enqueueWriteBuffer(*mPagedTokenTable, CL_TRUE,
                                    static_cast<size_t>(logicalStart) * sizeof(int),
                                    static_cast<size_t>(tokenCount) * sizeof(int),
                                    tokenTable.data());
    res |= queue.enqueueWriteBuffer(*mPagedRopeTable, CL_TRUE,
                                    static_cast<size_t>(logicalStart) * sizeof(int),
                                    static_cast<size_t>(tokenCount) * sizeof(int),
                                    ropeTable.data());
    if (res != CL_SUCCESS) {
        MNN_ERROR("[Error]: KVCacheCLManager failed to upload OpenCL paged append table: %d "
                  "logical_start=%d token_count=%d required_logical=%d table_capacity=%d "
                  "physical_start=%d required_physical=%d physical_length=%d\n",
                  res, logicalStart, tokenCount, requiredLogical, mPagedTableCapacity,
                  physicalStart, requiredPhysical, mPagedPhysicalLength);
        return false;
    }
    mPagedPhysicalLength = requiredPhysical;
    mPagedLogicalLength = requiredLogical;
    return true;
}

bool KVCacheCLManager::ensurePagedDecodeAppendTable(int logicalStart, int tokenCount) {
    bool isolateSourceStart = !mPagedDecodeStarted;
    if (!ensurePagedAppendTable(logicalStart, tokenCount, isolateSourceStart)) {
        return false;
    }
    mPagedDecodeStarted = true;
    return true;
}

ErrorCode AttentionBufExecution::saveOpenCLNativePrefixCache(const std::vector<Tensor *> &inputs, int totalKvLen) {
    std::lock_guard<std::recursive_mutex> queueLock(openCLPrefixQueueMutex());
    if (mMeta == nullptr || mMeta->file_flag != KVMeta::PendingWrite || mMeta->file_name.empty()) {
        return NO_ERROR;
    }
    if (inputs.size() < 2 || inputs[1] == nullptr) {
        return INVALID_VALUE;
    }
    auto key = inputs[1];
    if (key->dimensions() < 4) {
        return INVALID_VALUE;
    }
    int batch = key->shape()[0];
    int kvHeads = key->shape()[2];
    int headDim = key->shape()[3];
    if (batch != 1) {
        MNN_ERROR("[Error]: OpenCL prefix cache save only supports batch=1\n");
        return NOT_SUPPORT;
    }
    if (kvHeads <= 0 || headDim <= 0 || totalKvLen <= 0) {
        return INVALID_VALUE;
    }
    if (mMeta->key_rope_state != KVMeta::KeyRopeCanonicalNoRope) {
        MNN_ERROR("[Error]: OpenCL native prefix cache save requires canonical_no_rope key\n");
        return NOT_SUPPORT;
    }
    int ropeDim = mMeta->rope_dim > 0 ? mMeta->rope_dim : headDim;
    ropeDim = std::min(ropeDim, headDim);
    if (ropeDim <= 0 || (ropeDim % 2) != 0 || mMeta->rope_theta <= 0.0f ||
        mMeta->rope_pairing != KVMeta::RopePairingHalf) {
        MNN_ERROR("[Error]: OpenCL native prefix cache save got invalid RoPE metadata\n");
        return INVALID_VALUE;
    }
    auto* keyBuffer = mKVCacheCLManager->mutableKey();
    auto* valueBuffer = mKVCacheCLManager->mutableValue();
    if (keyBuffer == nullptr || valueBuffer == nullptr) {
        MNN_ERROR("[Error]: OpenCL native prefix cache save requires active runtime KV cache\n");
        return INVALID_VALUE;
    }

    int bytes = mKVCacheCLManager->byte();
    if (bytes != 2 && bytes != 4) {
        MNN_ERROR("[Error]: OpenCL native prefix cache save only supports fp16/fp32 KV\n");
        return NOT_SUPPORT;
    }
    int runtimeMaxLen = ROUND_UP(mKVCacheCLManager->maxLength(), 4);
    int diskMaxLen = ROUND_UP(totalKvLen, 4);
    if (runtimeMaxLen < totalKvLen || diskMaxLen <= 0) {
        return INVALID_VALUE;
    }
    size_t runtimeBytes = static_cast<size_t>(kvHeads) * headDim * runtimeMaxLen * bytes;
    size_t diskBytes = static_cast<size_t>(kvHeads) * headDim * diskMaxLen * bytes;

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    cl_int keyMapStatus = CL_SUCCESS;
    cl_int valueMapStatus = CL_SUCCESS;
    auto& queue = runtime->defaultCommandQueue();
    auto* mappedKey = static_cast<uint8_t*>(queue.enqueueMapBuffer(
        *keyBuffer, CL_TRUE, CL_MAP_READ, 0, runtimeBytes, nullptr, nullptr, &keyMapStatus));
    auto* mappedValue = static_cast<uint8_t*>(queue.enqueueMapBuffer(
        *valueBuffer, CL_TRUE, CL_MAP_READ, 0, runtimeBytes, nullptr, nullptr, &valueMapStatus));
    if (mappedKey == nullptr || mappedValue == nullptr ||
        keyMapStatus != CL_SUCCESS || valueMapStatus != CL_SUCCESS) {
        MNN_ERROR("[Error]: OpenCL native prefix cache failed to map runtime KV buffers\n");
        if (mappedKey != nullptr) {
            queue.enqueueUnmapMemObject(*keyBuffer, mappedKey);
        }
        if (mappedValue != nullptr) {
            queue.enqueueUnmapMemObject(*valueBuffer, mappedValue);
        }
        return INVALID_VALUE;
    }

    std::vector<uint8_t> keyHost(diskBytes, 0);
    std::vector<uint8_t> valueHost(diskBytes, 0);
    for (int head = 0; head < kvHeads; ++head) {
        for (int token = 0; token < diskMaxLen; ++token) {
            for (int dim = 0; dim < headDim; ++dim) {
                size_t dstKeyIndex = (static_cast<size_t>(head) * headDim + dim) * diskMaxLen + token;
                size_t dstValueIndex = (static_cast<size_t>(head) * diskMaxLen + token) * headDim + dim;
                if (token >= totalKvLen) {
                    writeOpenCLNativeScalar(keyHost.data(), dstKeyIndex, bytes, 0.0f);
                    writeOpenCLNativeScalar(valueHost.data(), dstValueIndex, bytes, 0.0f);
                    continue;
                }

                size_t srcKeyIndex = (static_cast<size_t>(head) * headDim + dim) * runtimeMaxLen + token;
                size_t srcValueIndex = (static_cast<size_t>(head) * runtimeMaxLen + token) * headDim + dim;
                float keyValue = readOpenCLNativeScalar(mappedKey, srcKeyIndex, bytes);
                float value = readOpenCLNativeScalar(mappedValue, srcValueIndex, bytes);
                if (dim < ropeDim) {
                    int halfDim = ropeDim / 2;
                    int pairDim = dim < halfDim ? dim + halfDim : dim - halfDim;
                    int freqDim = dim < halfDim ? dim : dim - halfDim;
                    size_t pairIndex = (static_cast<size_t>(head) * headDim + pairDim) * runtimeMaxLen + token;
                    float pairValue = readOpenCLNativeScalar(mappedKey, pairIndex, bytes);
                    float left = dim < halfDim ? keyValue : pairValue;
                    float right = dim < halfDim ? pairValue : keyValue;
                    float invFreq = 1.0f / std::pow(mMeta->rope_theta, static_cast<float>(2 * freqDim) / static_cast<float>(ropeDim));
                    float angle = static_cast<float>(token) * invFreq;
                    float cosValue = std::cos(angle);
                    float sinValue = -std::sin(angle);
                    keyValue = dim < halfDim ? left * cosValue - right * sinValue : right * cosValue + left * sinValue;
                }
                writeOpenCLNativeScalar(keyHost.data(), dstKeyIndex, bytes, keyValue);
                writeOpenCLNativeScalar(valueHost.data(), dstValueIndex, bytes, value);
            }
        }
    }

    queue.enqueueUnmapMemObject(*keyBuffer, mappedKey);
    queue.enqueueUnmapMemObject(*valueBuffer, mappedValue);
    queue.finish();

    if (!ensurePrefixCacheObjectDirs(mMeta->prefix_cache_dir, "opencl", mMeta->file_name)) {
        MNN_ERROR("[Error]: failed to create OpenCL prefix cache object dirs: %s\n",
                  mMeta->prefix_cache_dir.c_str());
        return FILE_OPEN_FAILED;
    }
    int layerIndex = mMeta->layer_index;
    auto base = prefixCacheLayerBase(mMeta->prefix_cache_dir, "opencl", mMeta->file_name, layerIndex);
    {
        std::ofstream out(base + ".k", std::ios::binary | std::ios::trunc);
        if (!out) {
            MNN_ERROR("[Error]: failed to write OpenCL prefix key cache: %s\n", (base + ".k").c_str());
            return FILE_OPEN_FAILED;
        }
        out.write(reinterpret_cast<const char*>(keyHost.data()), static_cast<std::streamsize>(keyHost.size()));
    }
    {
        std::ofstream out(base + ".v", std::ios::binary | std::ios::trunc);
        if (!out) {
            MNN_ERROR("[Error]: failed to write OpenCL prefix value cache: %s\n", (base + ".v").c_str());
            return FILE_OPEN_FAILED;
        }
        out.write(reinterpret_cast<const char*>(valueHost.data()), static_cast<std::streamsize>(valueHost.size()));
    }

    mMeta->layer_index++;
    if (mMeta->layer_nums > 0) {
        mMeta->layer_index %= mMeta->layer_nums;
    }
    return NO_ERROR;
}

void AttentionBufExecution::handleKVCache(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if(mHasMask) {
        auto mask = inputs[3];
        mIsAddMask = (mask->getType() == halide_type_of<float>());
    }

    auto query = inputs[0];
    auto key = inputs[1];
    auto shape = query->shape();

    int batch = shape[0];
    int seqlen = shape[1];
    int numHead = shape[2];
    int kvNumHead = key->shape()[2];
    int headDim = shape[3];

    if(nullptr == mMeta) {
        mPastKvSeqlen = 0;
        mKvSeqlen = seqlen;
        mKeyValueMaxlen = ROUND_UP(seqlen, 4);
        mDecodeTmpMaxlen = ROUND_UP(seqlen, 4);
        return;
    }
    mKVCacheCLManager->setArgs(numHead, kvNumHead, headDim);
    mKVCacheCLManager->allocKVCache(mMeta, seqlen);
    int prefixKvLength = onResizePrefixKVLength(inputs, seqlen);
    if (prefixKvLength > 0) {
        mKVCacheCLManager->ensureCapacity(prefixKvLength + seqlen, false);
        mKVCacheCLManager->setPastKvLength(prefixKvLength);
    }
    mKeyValueMaxlen = ROUND_UP(mKVCacheCLManager->maxLength(), 4);
    mDecodeTmpMaxlen = mKeyValueMaxlen;
    mPastKvSeqlen = mKVCacheCLManager->pastKvLength();
    mKvSeqlen = mPastKvSeqlen + seqlen;
}

ErrorCode AttentionBufExecution::init() {
    if(nullptr == mMeta) {
        return NO_ERROR;
    }
    //clear update arg vector, if prefill and decode use the same one
    mOpRecordUpdateInfo.clear();
    mRgQUpdateInfo.update_kernel_args.clear();
    mRgQUpdateInfo.update_global_size.clear();
    mRgQUpdateInfo.update_local_size.clear();
    mRgMUpdateInfo.update_kernel_args.clear();
    mRgMUpdateInfo.update_global_size.clear();
    mRgMUpdateInfo.update_local_size.clear();
    mRgUpdateInfo.update_kernel_args.clear();
    mRgUpdateInfo.update_global_size.clear();
    mRgUpdateInfo.update_local_size.clear();
    mQkUpdateInfo.update_kernel_args.clear();
    mQkUpdateInfo.update_global_size.clear();
    mQkUpdateInfo.update_local_size.clear();
    mSoftMaxUpdateInfo.update_kernel_args.clear();
    mSoftMaxUpdateInfo.update_global_size.clear();
    mSoftMaxUpdateInfo.update_local_size.clear();
    mRgVUpdateInfo.update_kernel_args.clear();
    mRgVUpdateInfo.update_global_size.clear();
    mRgVUpdateInfo.update_local_size.clear();
    mQkvUpdateInfo.update_kernel_args.clear();
    mQkvUpdateInfo.update_global_size.clear();
    mQkvUpdateInfo.update_local_size.clear();

    return NO_ERROR;
}

ErrorCode AttentionBufExecution::UpdateArgs(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs){
    if(nullptr == mMeta) {
        return NO_ERROR;
    }

    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto mask = inputs[3];
    auto shape = query->shape();

    int batch = shape[0];
    int seqlen = shape[1];
    int numHead = shape[2];
    int kvNumHead = key->shape()[2];
    int headDim = shape[3];
    int group_size = numHead / kvNumHead;
    float scale = 1.0 / sqrt(headDim);
    bool usePagedKV = mKVCacheCLManager != nullptr && mKVCacheCLManager->pagedActive();
    auto mask_shape = mask->shape();
    int dim = mask->dimensions();
    MNN_ASSERT(dim >= 2);
    int mask_seqlen = mask_shape[dim - 2];
    int maskKvlen  = mask_shape[dim - 1];
    mPastKvSeqlen = mKVCacheCLManager->pastKvLength();
    mKvSeqlen = mKVCacheCLManager->pastKvLength() + seqlen;
    mKVCacheCLManager->addKvLength(seqlen);
    // prefill
    if(mIsDecode == false){
        int maskKvlen = mKvSeqlen;
        int maskQlen = seqlen;
        if(mHasMask) {
            auto mask = inputs[3];
            auto mask_shape = mask->shape();
            int dim = mask->dimensions();
            MNN_ASSERT(dim >= 2);
            maskQlen = mask_shape[dim - 2];
            maskKvlen  = mask_shape[dim - 1];
        }
        // key value static memory has been changed, need reset args
        if(mKeyValueMaxlen != ROUND_UP(mKVCacheCLManager->maxLength(), 4)){
            mKeyValueMaxlen = ROUND_UP(mKVCacheCLManager->maxLength(), 4);
        }
        if(false == mLongPrefill){
            mGlobalWorkSizeQk0 = UP_DIV(mKvSeqlen, 4);
            mQkPrefillGlobal_size[1] = ROUND_UP(mGlobalWorkSizeQk0, std::max((uint32_t)1, mLocalWorkSizeQk[1]));
            mGlobalWorkSizeQk[1] = mQkPrefillGlobal_size[1];
            mTempQ.reset(Tensor::createDevice<float>({ROUND_UP(seqlen, 4) * ROUND_UP(headDim, 4) * batch * numHead}));
            mTempQK.reset(Tensor::createDevice<float>({ROUND_UP(seqlen, 4) * mKvSeqlen * numHead * batch}));
            mTempSoftMax.reset(Tensor::createDevice<float>({ROUND_UP(seqlen, 4) * mKvSeqlen * numHead * batch}));
            if(mIsAddMask) {
                mTempMask.reset(Tensor::createDevice<float>({ROUND_UP(maskQlen, 4) * ROUND_UP(maskKvlen, 4) * batch}));
            } else {
                mTempMask.reset(Tensor::createDevice<uint32_t>({ROUND_UP(maskQlen, 4) * ROUND_UP(maskKvlen, 4) * batch}));
            }
            mOpenCLBackend->onAcquireBuffer(mTempQ.get(), Backend::DYNAMIC_IN_EXECUTION);
            mOpenCLBackend->onAcquireBuffer(mTempMask.get(), Backend::DYNAMIC_IN_EXECUTION);
            mOpenCLBackend->onAcquireBuffer(mTempQK.get(), Backend::DYNAMIC_IN_EXECUTION);
            mOpenCLBackend->onAcquireBuffer(mTempSoftMax.get(), Backend::DYNAMIC_IN_EXECUTION);
            mOpenCLBackend->onReleaseBuffer(mTempQ.get(), Backend::DYNAMIC_IN_EXECUTION);
            mOpenCLBackend->onReleaseBuffer(mTempMask.get(), Backend::DYNAMIC_IN_EXECUTION);
            mOpenCLBackend->onReleaseBuffer(mTempQK.get(), Backend::DYNAMIC_IN_EXECUTION);
            mOpenCLBackend->onReleaseBuffer(mTempSoftMax.get(), Backend::DYNAMIC_IN_EXECUTION);
        }
        #ifndef ENABLE_OPENCL_TIME_PROFILER
        if(mOpenCLBackend->isUseRecordQueue()){
            if(mLongPrefill){
                mRgUpdateInfo.update_kernel_args[0].arg_value = &(*(mKVCacheCLManager->key()))();
                mRgUpdateInfo.update_kernel_args[1].arg_value = &(*(mKVCacheCLManager->value()))();
            }else{
                mRgQUpdateInfo.update_kernel_args[0].arg_value = &openCLDeferBuffer(mTempQ.get())();
                mRgUpdateInfo.update_kernel_args[0].arg_value = &(*(mKVCacheCLManager->key()))();
                mRgMUpdateInfo.update_kernel_args[0].arg_value = &openCLDeferBuffer(mTempMask.get())();
                mQkUpdateInfo.update_kernel_args[1].arg_value = &openCLDeferBuffer(mTempQ.get())();
                mQkUpdateInfo.update_kernel_args[2].arg_value = &(*(mKVCacheCLManager->key()))();
                if(mHasMask){
                    mQkUpdateInfo.update_kernel_args[3].arg_value = &openCLDeferBuffer(mTempMask.get())();
                    mQkUpdateInfo.update_kernel_args[4].arg_value = &openCLDeferBuffer(mTempQK.get())();
                }else{
                    mQkUpdateInfo.update_kernel_args[3].arg_value = &openCLDeferBuffer(mTempQK.get())();
                }
                mSoftMaxUpdateInfo.update_kernel_args[0].arg_value = &openCLDeferBuffer(mTempQK.get())();
                mSoftMaxUpdateInfo.update_kernel_args[1].arg_value = &openCLDeferBuffer(mTempSoftMax.get())();
                mRgVUpdateInfo.update_kernel_args[0].arg_value = &(*(mKVCacheCLManager->value()))();
                mQkvUpdateInfo.update_kernel_args[0].arg_value = &openCLDeferBuffer(mTempSoftMax.get())();
                mQkvUpdateInfo.update_kernel_args[1].arg_value = &(*(mKVCacheCLManager->value()))();
            }
        } else {
        #endif
            if(mLongPrefill){
                // rearrange key value
                cl_int ret = CL_SUCCESS;
                ret |= mKernel_rearrange_vec[0]->get().setArg(9, *mKVCacheCLManager->key());
                ret |= mKernel_rearrange_vec[0]->get().setArg(10, *mKVCacheCLManager->value());
                ret |= mKernel_rearrange_vec[0]->get().setArg(14, mKeyValueMaxlen);
                MNN_CHECK_CL_SUCCESS(ret, "reSetArg rearrange_k");
            }else{
                {
                    // rearrange query
                    cl_int ret = CL_SUCCESS;
                    ret |= mKernel_rearrangeQ->get().setArg(4, openCLDeferBuffer(mTempQ.get()));
                    MNN_CHECK_CL_SUCCESS(ret, "reSetArg rearrange_q");
                }
                {
                    // rearrange key
                    cl_int ret = CL_SUCCESS;
                    ret |= mKernel_rearrange->get().setArg(4, *mKVCacheCLManager->key());
                    if (usePagedKV) {
                        ret |= mKernel_rearrange->get().setArg(5, *mKVCacheCLManager->tokenTable());
                        ret |= mKernel_rearrange->get().setArg(6, *mKVCacheCLManager->ropeTable());
                        ret |= mKernel_rearrange->get().setArg(7, mPastKvSeqlen);
                        ret |= mKernel_rearrange->get().setArg(8, mKeyValueMaxlen);
                    } else {
                        ret |= mKernel_rearrange->get().setArg(5, mPastKvSeqlen);
                        ret |= mKernel_rearrange->get().setArg(6, mKeyValueMaxlen);
                    }
                    MNN_CHECK_CL_SUCCESS(ret, "reSetArg rearrange_k");
                }
                if(mHasMask){
                    // rearrange mask
                    cl_int ret = CL_SUCCESS;
                    ret |= mKernel_rearrangeMask->get().setArg(4, openCLDeferBuffer(mTempMask.get()));
                    MNN_CHECK_CL_SUCCESS(ret, "reSetArg rearrange_mask_shortprefill");
                }
                {
                    // matmul qk
                    mGlobalWorkSizeQk =  {static_cast<uint32_t>(UP_DIV(seqlen, 4)), static_cast<uint32_t>(UP_DIV(mKvSeqlen, 4)), static_cast<uint32_t>(numHead*batch)};
                    cl_int ret = CL_SUCCESS;
                    ret |= mKernel_qk->get().setArg(1, mGlobalWorkSizeQk0);
                    ret |= mKernel_qk->get().setArg(3, openCLDeferBuffer(mTempQ.get()));
                    ret |= mKernel_qk->get().setArg(4, *mKVCacheCLManager->key());
                    if (mHasMask) {
                        ret |= mKernel_qk->get().setArg(5, openCLDeferBuffer(mTempMask.get()));
                        ret |= mKernel_qk->get().setArg(6, openCLDeferBuffer(mTempQK.get()));
                        ret |= mKernel_qk->get().setArg(10, mKvSeqlen);
                        ret |= mKernel_qk->get().setArg(11, mKeyValueMaxlen);
                        if (usePagedKV) {
                            ret |= mKernel_qk->get().setArg(14, *mKVCacheCLManager->tokenTable());
                            ret |= mKernel_qk->get().setArg(15, *mKVCacheCLManager->ropeTable());
                            ret |= mKernel_qk->get().setArg(16, mKVCacheCLManager->pagedRopeDim());
                            ret |= mKernel_qk->get().setArg(17, mKVCacheCLManager->pagedRopeTheta());
                        }
                    } else {
                        ret |= mKernel_qk->get().setArg(5, openCLDeferBuffer(mTempQK.get()));
                        ret |= mKernel_qk->get().setArg(9, mKvSeqlen);
                        ret |= mKernel_qk->get().setArg(10, mKeyValueMaxlen);
                        if (usePagedKV) {
                            ret |= mKernel_qk->get().setArg(13, *mKVCacheCLManager->tokenTable());
                            ret |= mKernel_qk->get().setArg(14, *mKVCacheCLManager->ropeTable());
                            ret |= mKernel_qk->get().setArg(15, mKVCacheCLManager->pagedRopeDim());
                            ret |= mKernel_qk->get().setArg(16, mKVCacheCLManager->pagedRopeTheta());
                        }
                    }
                    MNN_CHECK_CL_SUCCESS(ret, "reSetArg matmul_qk_decode");
                    mGlobalWorkSizeQk[0] = ROUND_UP(mGlobalWorkSizeQk[0], std::max((uint32_t)1, mLocalWorkSizeQk[0]));
                    mGlobalWorkSizeQk[1] = ROUND_UP(mGlobalWorkSizeQk[1], std::max((uint32_t)1, mLocalWorkSizeQk[1]));
                    mGlobalWorkSizeQk[2] = ROUND_UP(mGlobalWorkSizeQk[2], std::max((uint32_t)1, mLocalWorkSizeQk[2]));
                }
                {
                    // softmax
                    cl_int ret = CL_SUCCESS;
                    ret |= mKernel_softmax->get().setArg(3, openCLDeferBuffer(mTempQK.get()));
                    ret |= mKernel_softmax->get().setArg(4, openCLDeferBuffer(mTempSoftMax.get()));
                    ret |= mKernel_softmax->get().setArg(7, mKvSeqlen);
                    MNN_CHECK_CL_SUCCESS(ret, "reSetArg softmax");
                }
                {
                    // rearrange value
                    cl_int ret = CL_SUCCESS;
                    ret |= mKernel_rearrangeV->get().setArg(4, *mKVCacheCLManager->value());
                    if (usePagedKV) {
                        ret |= mKernel_rearrangeV->get().setArg(5, *mKVCacheCLManager->tokenTable());
                        ret |= mKernel_rearrangeV->get().setArg(6, mPastKvSeqlen);
                        ret |= mKernel_rearrangeV->get().setArg(7, mKeyValueMaxlen);
                    } else {
                        ret |= mKernel_rearrangeV->get().setArg(5, mPastKvSeqlen);
                        ret |= mKernel_rearrangeV->get().setArg(6, mKeyValueMaxlen);
                    }
                    MNN_CHECK_CL_SUCCESS(ret, "reSetArg rearrange_v");
                }
                {
                    // qk * value
                    cl_int ret = CL_SUCCESS;
                    ret |= mKernel_qkv->get().setArg(3, openCLDeferBuffer(mTempSoftMax.get()));
                    ret |= mKernel_qkv->get().setArg(4, *mKVCacheCLManager->value());
                    ret |= mKernel_qkv->get().setArg(7, mKvSeqlen);
                    ret |= mKernel_qkv->get().setArg(8, mKeyValueMaxlen);
                    if (usePagedKV) {
                        ret |= mKernel_qkv->get().setArg(12, *mKVCacheCLManager->tokenTable());
                    }
                    MNN_CHECK_CL_SUCCESS(ret, "reSetArg matmul_qkv_decode");
                }
            }
        #ifndef ENABLE_OPENCL_TIME_PROFILER
        }
        #endif
        return NO_ERROR;
    }

    // Decode
    mKeyValueMaxlen = ROUND_UP(mKVCacheCLManager->maxLength(), 4);
    if(mKvSeqlen > mDecodeTmpMaxlen){
        mDecodeTmpMaxlen = mKeyValueMaxlen;
        mTempQK.reset(Tensor::createDevice<float>({mDecodeTmpMaxlen * numHead}));
        mTempSoftMax.reset(Tensor::createDevice<float>({mDecodeTmpMaxlen * numHead}));
        mOpenCLBackend->onAcquireBuffer(mTempQK.get(), Backend::DYNAMIC_IN_EXECUTION);
        mOpenCLBackend->onAcquireBuffer(mTempSoftMax.get(), Backend::DYNAMIC_IN_EXECUTION);
        mOpenCLBackend->onReleaseBuffer(mTempQK.get(), Backend::DYNAMIC_IN_EXECUTION);
        mOpenCLBackend->onReleaseBuffer(mTempSoftMax.get(), Backend::DYNAMIC_IN_EXECUTION);
    }
    mGlobalWorkSizeQk0 = UP_DIV(mKvSeqlen, 4);
    mQkGlobal_size[0] = ROUND_UP(mGlobalWorkSizeQk0, std::max((uint32_t)1, mLocalWorkSizeQk[0]));
    mGlobalWorkSizeQk[0] = mQkGlobal_size[0];

#ifndef ENABLE_OPENCL_TIME_PROFILER
    // use record, only update args
    if(mOpenCLBackend->isUseRecordQueue()){
        mRgUpdateInfo.update_kernel_args[0].arg_value = &(*(mKVCacheCLManager->key()))();
        mQkUpdateInfo.update_kernel_args[1].arg_value = &(*(mKVCacheCLManager->key()))();
        mQkUpdateInfo.update_kernel_args[2].arg_value = &openCLDeferBuffer(mTempQK.get())();
        mSoftMaxUpdateInfo.update_kernel_args[0].arg_value = &openCLDeferBuffer(mTempQK.get())();
        mSoftMaxUpdateInfo.update_kernel_args[1].arg_value = &openCLDeferBuffer(mTempSoftMax.get())();
        mRgVUpdateInfo.update_kernel_args[0].arg_value = &(*(mKVCacheCLManager->value()))();
        mQkvUpdateInfo.update_kernel_args[0].arg_value = &openCLDeferBuffer(mTempSoftMax.get())();
        mQkvUpdateInfo.update_kernel_args[1].arg_value = &(*(mKVCacheCLManager->value()))();
    } else {
#endif
        // not use record, need update args by using setArg
        {
            // rearrange key
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_rearrange->get().setArg(4, *mKVCacheCLManager->key());
            if (usePagedKV) {
                ret |= mKernel_rearrange->get().setArg(5, *mKVCacheCLManager->tokenTable());
                ret |= mKernel_rearrange->get().setArg(6, *mKVCacheCLManager->ropeTable());
                ret |= mKernel_rearrange->get().setArg(7, mPastKvSeqlen);
                ret |= mKernel_rearrange->get().setArg(8, mKeyValueMaxlen);
            } else {
                ret |= mKernel_rearrange->get().setArg(5, mPastKvSeqlen);
                ret |= mKernel_rearrange->get().setArg(6, mKeyValueMaxlen);
            }
            MNN_CHECK_CL_SUCCESS(ret, "reSetArg rearrange_k");
        }
        {
            // matmul qk
            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_qk->get().setArg(index++, mGlobalWorkSizeQk0);
            index++;
            index++;
            ret |= mKernel_qk->get().setArg(index++, *mKVCacheCLManager->key());
            ret |= mKernel_qk->get().setArg(index++, openCLDeferBuffer(mTempQK.get()));
            index++;
            ret |= mKernel_qk->get().setArg(index++, mKvSeqlen);
            ret |= mKernel_qk->get().setArg(index++, mKeyValueMaxlen);
            if (usePagedKV) {
                ret |= mKernel_qk->get().setArg(10, *mKVCacheCLManager->tokenTable());
                ret |= mKernel_qk->get().setArg(11, *mKVCacheCLManager->ropeTable());
                ret |= mKernel_qk->get().setArg(12, mKVCacheCLManager->pagedRopeDim());
                ret |= mKernel_qk->get().setArg(13, mKVCacheCLManager->pagedRopeTheta());
            }
            mGlobalWorkSizeQk[0] = ROUND_UP(mGlobalWorkSizeQk[0], std::max((uint32_t)1, mLocalWorkSizeQk[0]));
            mGlobalWorkSizeQk[1] = ROUND_UP(mGlobalWorkSizeQk[1], std::max((uint32_t)1, mLocalWorkSizeQk[1]));
            MNN_CHECK_CL_SUCCESS(ret, "reSetArg matmul_qk_decode");
        }
        {
            // softmax
            uint32_t index = 3;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_softmax->get().setArg(index++, openCLDeferBuffer(mTempQK.get()));
            ret |= mKernel_softmax->get().setArg(index++, openCLDeferBuffer(mTempSoftMax.get()));
            index++;
            index++;
            ret |= mKernel_softmax->get().setArg(index++, mKvSeqlen);
            MNN_CHECK_CL_SUCCESS(ret, "reSetArg softmax");
        }
        {
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_rearrangeV->get().setArg(4, *mKVCacheCLManager->value());
            if (usePagedKV) {
                ret |= mKernel_rearrangeV->get().setArg(5, *mKVCacheCLManager->tokenTable());
                ret |= mKernel_rearrangeV->get().setArg(6, mPastKvSeqlen);
                ret |= mKernel_rearrangeV->get().setArg(7, mKeyValueMaxlen);
            } else {
                ret |= mKernel_rearrangeV->get().setArg(5, mPastKvSeqlen);
                ret |= mKernel_rearrangeV->get().setArg(6, mKeyValueMaxlen);
            }

            MNN_CHECK_CL_SUCCESS(ret, "reSetArg rearrange_v");
        }
        // qk * value
        {
            uint32_t index = 2;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_qkv->get().setArg(index++, openCLDeferBuffer(mTempSoftMax.get()));
            ret |= mKernel_qkv->get().setArg(index++, *mKVCacheCLManager->value());
            index++;
            ret |= mKernel_qkv->get().setArg(index++, mKvSeqlen);
            ret |= mKernel_qkv->get().setArg(index++, mKeyValueMaxlen);
            if (usePagedKV) {
                ret |= mKernel_qkv->get().setArg(10, *mKVCacheCLManager->tokenTable());
            }
            MNN_CHECK_CL_SUCCESS(ret, "reSetArg matmul_qkv_decode");
        }
#ifndef ENABLE_OPENCL_TIME_PROFILER
    }
#endif
    return NO_ERROR;
}

int AttentionBufExecution::getLocalSize(int size, int maxGroupSize){
    int local_size = 1;
    while(local_size * 2 <= maxGroupSize && local_size * 2 <= size){
        local_size *= 2;
    }
    return local_size;
}

ErrorCode AttentionBufExecution::longPrefillResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs){

    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto shape = query->shape();

    int batch = shape[0];
    int seqlen = shape[1];
    int numHead = shape[2];
    int kvNumHead = key->shape()[2];
    int headDim = shape[3];
    int group_size = numHead / kvNumHead;
    float scale = 1.0 / sqrt(headDim);
    mKernelUsePagedKV = false;
    mKernelIsDecode = false;

    mAlignQ = 32;
    mAlignKV = 32;
    mAlignHDK = 4;
    mAlignHDN = 32;

    float useMemorySize = 1.0 * ROUND_UP(seqlen, mAlignQ) / 1024.0 * ROUND_UP(seqlen, mAlignKV) / 1024.0 * batch * numHead;
    // elementSize larger than 32M
    if(useMemorySize > 32.0) {
        mQseqSplitNum = useMemorySize >= 256.0 ? 8 : ((useMemorySize < 128.0) ? 2 : 4);
    }
    // splitPiecesSize need aligned to 32, make sure XgemmBatched globalsize be divisible by localsize
    int splitPiecesSize = ROUND_UP(seqlen, mAlignQ) / mQseqSplitNum;
    while((splitPiecesSize % 32) != 0){
        mAlignQ *= 2;
        splitPiecesSize = ROUND_UP(seqlen, mAlignQ) / mQseqSplitNum;
    }

    mKernel_rearrange_vec.resize(1); mGwsRearrgVec.resize(1); mLwsRearrgVec.resize(1);
    mKernel_mask_vec.resize(1);     mGwsMaskVec.resize(1);   mLwsMaskVec.resize(1);
    mKernel_qk_vec.resize(mQseqSplitNum);       mGwsQkVec.resize(mQseqSplitNum);     mLwsQkVec.resize(mQseqSplitNum);
    mKernel_softmax_vec.resize(mQseqSplitNum);   mGwsSoftMaxVec.resize(mQseqSplitNum); mLwsSoftMaxVec.resize(mQseqSplitNum);
    mKernel_trans_vec.resize(mQseqSplitNum);     mGwsTransVec.resize(mQseqSplitNum);  mLwsTransVec.resize(mQseqSplitNum);
    mKernel_qkv_vec.resize(mQseqSplitNum);      mGwsQkvVec.resize(mQseqSplitNum);    mLwsQkvVec.resize(mQseqSplitNum);
    mKernel_clip_vec.resize(1);     mGwsClipVec.resize(1);   mLwsClipVec.resize(1);

    mTempQ.reset(Tensor::createDevice<float>({ROUND_UP(seqlen, mAlignQ) * ROUND_UP(headDim, mAlignHDK) * batch * numHead}));
    mTempK.reset(Tensor::createDevice<float>({ROUND_UP(seqlen, mAlignKV) * ROUND_UP(headDim, mAlignHDK) * batch * numHead}));
    mTempV.reset(Tensor::createDevice<float>({ROUND_UP(seqlen, mAlignKV) * ROUND_UP(headDim, mAlignHDN) * batch * numHead}));
    if(mHasMask) {
        if(mIsAddMask) {
            mTempMask.reset(Tensor::createDevice<float>({ROUND_UP(seqlen, mAlignQ) * ROUND_UP(seqlen, mAlignKV) * batch}));
        } else {
            mTempMask.reset(Tensor::createDevice<uint32_t>({ROUND_UP(seqlen, mAlignQ) * ROUND_UP(seqlen, mAlignKV) * batch}));
        }
    }
    const int longPieceQkElements =
        ROUND_UP(seqlen, mAlignQ) * ROUND_UP(seqlen, mAlignKV) * batch * numHead / mQseqSplitNum;
    mTempQK.reset(Tensor::createDevice<float>({longPieceQkElements}));
    mTempSoftMax.reset(Tensor::createDevice<float>({longPieceQkElements}));
    mTempQKV.reset(Tensor::createDevice<float>({ROUND_UP(seqlen, mAlignQ) * ROUND_UP(headDim, mAlignHDN) * batch * numHead}));


    mOpenCLBackend->onAcquireBuffer(mTempQ.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempK.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempV.get(), Backend::DYNAMIC);
    if(mHasMask) {
        mOpenCLBackend->onAcquireBuffer(mTempMask.get(), Backend::DYNAMIC);
    }
    mOpenCLBackend->onAcquireBuffer(mTempQK.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempSoftMax.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempQKV.get(), Backend::DYNAMIC);

    mOpenCLBackend->onReleaseBuffer(mTempQ.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mTempK.get(), Backend::DYNAMIC);
    if(mHasMask) {
        mOpenCLBackend->onReleaseBuffer(mTempMask.get(), Backend::DYNAMIC);
    }
    mOpenCLBackend->onReleaseBuffer(mTempV.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mTempSoftMax.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mTempQK.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mTempQKV.get(), Backend::DYNAMIC);

    // query: [batch, seqLenQ, headNum, headDim] -> mTempQ: [batch*headNum, ROUND_UP(headDim, mAlignHDK), ROUND_UP(seqLenQ, mAlignQ)]
    // key: [batch, seqLenKV/4, headNum/group, headDim, seqLenKV_4] -> mTempK: [batch*headNum/group, ROUND_UP(headDim, mAlignHDK), ROUND_UP(seqLenKV, mAlignKV)]
    // value: [batch, seqLenKV/4, headNum/group, headDim, seqLenKV_4] -> mTempV: [batch*headNum/group, ROUND_UP(seqLenKV, mAlignKV), ROUND_UP(headDim, mAlignHDK]
    // key & value -> pastKey & pastValue (copy)
    int seq_idx = 0;
    // rearrange qkv
    {
        std::set<std::string> buildOption;
        if((headDim % 4) != 0){
            buildOption.emplace("-DHEADDIM_LEAVE");
        }
        // generate cache for every option
        {
            auto option = buildOption;
            auto kernel = runtime->buildKernel("attention_buf", "rearrange_qkv", option, mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        }
        {
            auto option = buildOption;
            option.emplace("-DSEQLEN_LEAVE");
            auto kernel = runtime->buildKernel("attention_buf", "rearrange_qkv", option, mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        }
        if((seqlen % 4) != 0){
            buildOption.emplace("-DSEQLEN_LEAVE");
        }
        if(nullptr != mMeta) {
            buildOption.emplace("-DSAVE_KV");
        }
        int seq_len_pack_q = ROUND_UP(seqlen, mAlignQ);
        int seq_len_pack_kv = ROUND_UP(mKvSeqlen, mAlignKV);

        int head_dim_pack_qk = ROUND_UP(headDim, mAlignHDK);
        int head_dim_pack_v = ROUND_UP(headDim, mAlignHDN);

        int tile[4] = {mAlignQ, mAlignKV, mAlignHDK, mAlignHDN};
        int shape[4] = {seqlen, mKvSeqlen, numHead, headDim};
        int param[4] = {group_size, batch, 0, 0};
        mKernel_rearrange_vec[seq_idx] = runtime->buildKernel("attention_buf", "rearrange_qkv", buildOption, mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_rearrange_vec[seq_idx]));

        mGwsRearrgVec[seq_idx] = {static_cast<uint32_t>(ALIMAX(UP_DIV(seq_len_pack_q, 4), UP_DIV(seq_len_pack_kv, 4))), \
            static_cast<uint32_t>(ALIMAX(UP_DIV(head_dim_pack_qk, 4), UP_DIV(head_dim_pack_v, 4))), \
            static_cast<uint32_t>(batch*numHead)};

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, mGwsRearrgVec[seq_idx][0]);
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, mGwsRearrgVec[seq_idx][1]);
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, mGwsRearrgVec[seq_idx][2]);
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, openCLBuffer(query));
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, openCLBuffer(key));
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, openCLBuffer(value));
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, openCLBuffer(mTempQ.get()));
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, openCLBuffer(mTempK.get()));
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, openCLBuffer(mTempV.get()));
        if(nullptr != mMeta) {
            ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, *mKVCacheCLManager->key());
            ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, *mKVCacheCLManager->value());
        }
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, tile);
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, shape);
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, param);
        ret |= mKernel_rearrange_vec[seq_idx]->get().setArg(index++, mKeyValueMaxlen);

        MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_qkv");
        mLwsRearrgVec[seq_idx] = localWS3DDefault(mGwsRearrgVec[seq_idx], maxWorkGroupSize, runtime, "rearrange_qkv", mKernel_rearrange_vec[seq_idx], mOpenCLBackend->getCLTuneLevel(), "attention_buf").first;
        mGwsRearrgVec[seq_idx][0] = ROUND_UP(mGwsRearrgVec[seq_idx][0], std::max((uint32_t)1, mLwsRearrgVec[seq_idx][0]));
        mGwsRearrgVec[seq_idx][1] = ROUND_UP(mGwsRearrgVec[seq_idx][1], std::max((uint32_t)1, mLwsRearrgVec[seq_idx][1]));
        mGwsRearrgVec[seq_idx][2] = ROUND_UP(mGwsRearrgVec[seq_idx][2], std::max((uint32_t)1, mLwsRearrgVec[seq_idx][2]));
        if(nullptr != mMeta) {
            mRgUpdateInfo.update_kernel_args.push_back({0, 9, sizeof(cl_mem), &(*(mKVCacheCLManager->key()))()});
            mRgUpdateInfo.update_kernel_args.push_back({0, 10, sizeof(cl_mem), &(*(mKVCacheCLManager->value()))()});
        }
        mRgUpdateInfo.update_kernel_args.push_back({0, 14, sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
        mOpRecordUpdateInfo.emplace_back(&mRgUpdateInfo);
        mOpenCLBackend->recordKernel3d(mKernel_rearrange_vec[seq_idx], mGwsRearrgVec[seq_idx], mLwsRearrgVec[seq_idx], &mRgUpdateInfo);
    }

    // mask rearaange
    if(mHasMask)
    {
        std::set<std::string> buildOption;

        int seq_len_pack_q = ROUND_UP(seqlen, mAlignQ);
        int seq_len_pack_kv = ROUND_UP(mKvSeqlen, mAlignKV);
        int shape[4] = {seqlen, mKvSeqlen, mAlignQ, mAlignKV};

        mKernel_mask_vec[seq_idx] = runtime->buildKernel("attention_buf", "rearrange_mask", buildOption, mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_mask_vec[seq_idx]));

        mGwsMaskVec[seq_idx] = {static_cast<uint32_t>(UP_DIV(seq_len_pack_q, 4)), \
            static_cast<uint32_t>(UP_DIV(seq_len_pack_kv, 4)), \
            static_cast<uint32_t>(batch)};

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_mask_vec[seq_idx]->get().setArg(index++, mGwsMaskVec[seq_idx][0]);
        ret |= mKernel_mask_vec[seq_idx]->get().setArg(index++, mGwsMaskVec[seq_idx][1]);
        ret |= mKernel_mask_vec[seq_idx]->get().setArg(index++, mGwsMaskVec[seq_idx][2]);
        ret |= mKernel_mask_vec[seq_idx]->get().setArg(index++, openCLBuffer(inputs[3]));
        ret |= mKernel_mask_vec[seq_idx]->get().setArg(index++, openCLBuffer(mTempMask.get()));
        ret |= mKernel_mask_vec[seq_idx]->get().setArg(index++, shape);

        MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_mask");
        mLwsMaskVec[seq_idx] = localWS3DDefault(mGwsMaskVec[seq_idx], maxWorkGroupSize, runtime, "rearrange_mask", mKernel_mask_vec[seq_idx], mOpenCLBackend->getCLTuneLevel(), "attention_buf").first;
        mGwsMaskVec[seq_idx][0] = ROUND_UP(mGwsMaskVec[seq_idx][0], std::max((uint32_t)1, mLwsMaskVec[seq_idx][0]));
        mGwsMaskVec[seq_idx][1] = ROUND_UP(mGwsMaskVec[seq_idx][1], std::max((uint32_t)1, mLwsMaskVec[seq_idx][1]));
        mGwsMaskVec[seq_idx][2] = ROUND_UP(mGwsMaskVec[seq_idx][2], std::max((uint32_t)1, mLwsMaskVec[seq_idx][2]));
        mOpenCLBackend->recordKernel3d(mKernel_mask_vec[seq_idx], mGwsMaskVec[seq_idx], mLwsMaskVec[seq_idx]);
    }

    for(int seq_idx = 0; seq_idx < mQseqSplitNum; seq_idx++) {
        // qk matmul
        {
            // Q : [batch*headNum, ROUND_UP(headDim, mAlignHDK), ROUND_UP(seqLenQ, mAlignQ) / mQseqSplitNum] -> [B, K, M]
            // K : [batch*headNum/group, ROUND_UP(headDim, mAlignHDK), ROUND_UP(seqLenKV, mAlignKV)] -> [B, K, N]
            // QV: [Batch * numHead, ROUND_UP(seqLenQ, mAlignQ) / mQseqSplitNum, ROUND_UP(seqLenKV, mAlignKV)]   -> [B, M, N]
            int loop = batch * numHead;
            int e_pack = ROUND_UP(seqlen, mAlignQ);
            int e_pack_piece = e_pack / mQseqSplitNum;
            int h_pack = ROUND_UP(mKvSeqlen, mAlignKV);
            int l_pack = ROUND_UP(headDim, mAlignHDK);

            std::set<std::string> buildOptions;

            int biasType = 0;
            std::vector<cl::Buffer> bufferVec = {openCLBuffer(mTempQ.get()), openCLBuffer(mTempK.get()), openCLBuffer(mTempQK.get())};
            if(mHasMask) {
                bufferVec.emplace_back(openCLBuffer(mTempMask.get()));
            }
            if(mIsAddMask) {
                biasType = 2;
            } else if(mHasMask) {
                biasType = 5;// int value mask
            }
            uint32_t layout = 14; // 10 means mix-precision, 4 means layout
            auto param = getGemmParams({(uint32_t)e_pack_piece, (uint32_t)h_pack, (uint32_t)l_pack, layout, (uint32_t)loop, (uint32_t)(biasType + 10*(group_size-1))}, bufferVec, mOpenCLBackend->getOpenCLRuntime(), mOpenCLBackend->getPrecision(), mOpenCLBackend->getCLTuneLevel());

            int KWG=param[0], KWI=param[1], MDIMA=param[2], MDIMC=param[3], MWG=param[4], NDIMB=param[5], NDIMC=param[6], NWG=param[7], SA=param[8], SB=param[9], STRM=param[10], STRN=param[11], VWM=param[12], VWN=param[13];
            buildOptions.emplace("-DKWG=" + std::to_string(KWG));
            buildOptions.emplace("-DKWI=" + std::to_string(KWI));
            buildOptions.emplace("-DMDIMA=" + std::to_string(MDIMA));
            buildOptions.emplace("-DMDIMC=" + std::to_string(MDIMC));
            buildOptions.emplace("-DMWG=" + std::to_string(MWG));
            buildOptions.emplace("-DNDIMB=" + std::to_string(NDIMB));
            buildOptions.emplace("-DNDIMC=" + std::to_string(NDIMC));
            buildOptions.emplace("-DNWG=" + std::to_string(NWG));
            buildOptions.emplace("-DSA=" + std::to_string(SA));
            buildOptions.emplace("-DSB=" + std::to_string(SB));
            buildOptions.emplace("-DSTRM=" + std::to_string(STRM));
            buildOptions.emplace("-DSTRN=" + std::to_string(STRN));
            buildOptions.emplace("-DVWM=" + std::to_string(VWM));
            buildOptions.emplace("-DVWN=" + std::to_string(VWN));
            if(layout >= 4) {
                buildOptions.emplace("-DOUTPUTMN");
            }

            int tileM = MWG;
            int tileN = NWG;
            int localM = MDIMC;
            int localN = NDIMC;

            if(mOpenCLBackend->getOpenCLRuntime()->getGpuType() == GpuType::ADRENO) {
                buildOptions.emplace("-DUSE_CL_MAD=1");
                buildOptions.emplace("-DRELAX_WORKGROUP_SIZE=1");
            }
            buildOptions.emplace("-DONLY_HAVE_ALPHA");
            if(biasType >= 1) {
                buildOptions.emplace("-DBIAS_TYPE=" + std::to_string(biasType));
            }

            buildOptions.emplace("-DPRECISION_COMPUTE=float -DCONVERT_PRECISION_COMPUTE=convert_float");
            buildOptions.emplace("-DPRECISION_COMPUTE2=float2 -DCONVERT_PRECISION_COMPUTE2=convert_float2");
            buildOptions.emplace("-DPRECISION_COMPUTE4=float4 -DCONVERT_PRECISION_COMPUTE4=convert_float4");
            buildOptions.emplace("-DPRECISION_COMPUTE8=float8 -DCONVERT_PRECISION_COMPUTE8=convert_float8");
            buildOptions.emplace("-DPRECISION_COMPUTE16=float16 -DCONVERT_PRECISION_COMPUTE16=convert_float16");

            mKernel_qk_vec[seq_idx] = mOpenCLBackend->getOpenCLRuntime()->buildKernel("matmul_params_buf", "XgemmBatched", buildOptions, mOpenCLBackend->getPrecision());

            int out_per_thread_m = tileM / localM;
            int out_per_thread_n = tileN / localN;

            mGwsQkVec[seq_idx] = {static_cast<uint32_t>(e_pack_piece/out_per_thread_m), static_cast<uint32_t>(h_pack/out_per_thread_n), static_cast<uint32_t>(loop)};
            mLwsQkVec[seq_idx] = {static_cast<uint32_t>(localM), static_cast<uint32_t>(localN), 1};

            float alpha = scale;
            float beta = 0.0f;
            int batch_offset_a = e_pack * l_pack;
            int batch_offset_b = h_pack * l_pack;
            int batch_offset_c = e_pack_piece * h_pack;

            int batch_offset[4] = {batch_offset_a, batch_offset_b, batch_offset_c, 0};
            // Q reads its logical slice from the full rearranged Q buffer, but the long-prefill
            // scratch buffers are reused piece by piece. So:
            // - A (Q) keeps the per-piece source offset.
            // - C (QK output scratch) stays piece-local at offset 0.
            // - E (mask / bias) still advances to the matching logical query slice.
            int base_ptr_offset[4] = {e_pack_piece * seq_idx, 0, 0, batch_offset_c * seq_idx};
            int stride[4] = {e_pack, h_pack, h_pack, h_pack};
            int group[4] = {1, group_size, 1, loop};

            int idx            = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, static_cast<int>(e_pack_piece));
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, static_cast<int>(h_pack));
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, static_cast<int>(l_pack));
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, alpha);
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, beta);
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, openCLBuffer(mTempQ.get()));
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, openCLBuffer(mTempK.get()));
            if(mHasMask) {
                ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, openCLBuffer(mTempMask.get()));
            }
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, openCLBuffer(mTempQK.get()));
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, batch_offset);
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, base_ptr_offset);
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, stride);
            ret |= mKernel_qk_vec[seq_idx]->get().setArg(idx++, group);
            MNN_CHECK_CL_SUCCESS(ret, "setArg Self-Attention batchmatmul qk Kernel");
            mOpenCLBackend->recordKernel3d(mKernel_qk_vec[seq_idx], mGwsQkVec[seq_idx], mLwsQkVec[seq_idx]);
        }

        // softmax
        {
            // QV:     [Batch * numHead, ROUND_UP(seqLenQ, mAlignQ) / mQseqSplitNum, ROUND_UP(seqLenKV, mAlignKV)]
            // Sotmax: [Batch * numHead, ROUND_UP(seqLenQ, mAlignQ) / mQseqSplitNum, ROUND_UP(seqLenKV, mAlignKV)]
            // axis  : 2 (last dim)
            int softmaxShape[4];
            softmaxShape[0] = batch*numHead;
            softmaxShape[1] = ROUND_UP(seqlen, mAlignQ) / mQseqSplitNum;
            softmaxShape[2] = ROUND_UP(mKvSeqlen, mAlignKV);

            auto MaxLocalSize = std::min(std::min(runtime->getMaxWorkItemSizes()[0], mMaxWorkGroupSize), static_cast<uint32_t>(256));
            int localSize = 64;

            std::set<std::string> buildOption;
            buildOption.emplace("-DSOFTMAX_LOCAL_SIZE=" + std::to_string(localSize));

            mKernel_softmax_vec[seq_idx] = runtime->buildKernel("self_attention_buf", "softmax_inside", buildOption, mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
            mGwsSoftMaxVec[seq_idx] =  {static_cast<uint32_t>(localSize), static_cast<uint32_t>(softmaxShape[1]), static_cast<uint32_t>(softmaxShape[0])};

            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_softmax_vec[seq_idx]->get().setArg(index++, mGwsSoftMaxVec[seq_idx][0]);
            ret |= mKernel_softmax_vec[seq_idx]->get().setArg(index++, mGwsSoftMaxVec[seq_idx][1]);
            ret |= mKernel_softmax_vec[seq_idx]->get().setArg(index++, mGwsSoftMaxVec[seq_idx][2]);
            ret |= mKernel_softmax_vec[seq_idx]->get().setArg(index++, openCLBuffer(mTempQK.get()));
            ret |= mKernel_softmax_vec[seq_idx]->get().setArg(index++, openCLBuffer(mTempSoftMax.get()));
            ret |= mKernel_softmax_vec[seq_idx]->get().setArg(index++, mKvSeqlen);
            ret |= mKernel_softmax_vec[seq_idx]->get().setArg(index++, softmaxShape);
            MNN_CHECK_CL_SUCCESS(ret, "setArg Attention softmax");

            mLwsSoftMaxVec[seq_idx] = {static_cast<uint32_t>(localSize), 1, 1};
            mOpenCLBackend->recordKernel3d(mKernel_softmax_vec[seq_idx], mGwsSoftMaxVec[seq_idx], mLwsSoftMaxVec[seq_idx]);
        }
        {
            // Sotmax: [Batch * numHead, ROUND_UP(seqLenQ, mAlignQ) / mQseqSplitNum, ROUND_UP(seqLenKV, mAlignKV)]
            // Trans:  [Batch * numHead, ROUND_UP(seqLenKV, mAlignKV), ROUND_UP(seqLenQ, mAlignQ) / mQseqSplitNum]
            int loop = batch * numHead;
            int transDimW = ROUND_UP(seqlen, mAlignQ) / mQseqSplitNum;
            int transDimH = ROUND_UP(mKvSeqlen, mAlignKV);

            std::set<std::string> buildOptions;
            mKernel_trans_vec[seq_idx] = runtime->buildKernel("self_attention_buf", "trans_3d_buf", buildOptions, mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mKernel_trans_vec[seq_idx]));

            mGwsTransVec[seq_idx] = {(uint32_t)transDimW/8, (uint32_t)transDimH/8, (uint32_t)(loop)};

            uint32_t index = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_trans_vec[seq_idx]->get().setArg(index++, mGwsTransVec[seq_idx][0]);
            ret |= mKernel_trans_vec[seq_idx]->get().setArg(index++, mGwsTransVec[seq_idx][1]);
            ret |= mKernel_trans_vec[seq_idx]->get().setArg(index++, mGwsTransVec[seq_idx][2]);
            ret |= mKernel_trans_vec[seq_idx]->get().setArg(index++, openCLBuffer(mTempSoftMax.get()));
            ret |= mKernel_trans_vec[seq_idx]->get().setArg(index++, openCLBuffer(mTempQK.get()));
            ret |= mKernel_trans_vec[seq_idx]->get().setArg(index++, loop);
            ret |= mKernel_trans_vec[seq_idx]->get().setArg(index++, transDimW);
            ret |= mKernel_trans_vec[seq_idx]->get().setArg(index++, transDimH);
            MNN_CHECK_CL_SUCCESS(ret, "setArg Attention transpose");
            mLwsTransVec[seq_idx] = localWS3DDefault(mGwsTransVec[seq_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "trans_3d_buf", mKernel_trans_vec[seq_idx], mOpenCLBackend->getCLTuneLevel(), "self_attention_buf").first;

            mGwsTransVec[seq_idx][0] = ROUND_UP(mGwsTransVec[seq_idx][0], std::max((uint32_t)1, mLwsTransVec[seq_idx][0]));
            mGwsTransVec[seq_idx][1] = ROUND_UP(mGwsTransVec[seq_idx][1], std::max((uint32_t)1, mLwsTransVec[seq_idx][1]));
            mGwsTransVec[seq_idx][2] = ROUND_UP(mGwsTransVec[seq_idx][2], std::max((uint32_t)1, mLwsTransVec[seq_idx][2]));

            mOpenCLBackend->recordKernel3d(mKernel_trans_vec[seq_idx], mGwsTransVec[seq_idx], mLwsTransVec[seq_idx]);
        }

        // qk * value
        {
            // Trans: [Batch * numHead, ROUND_UP(seqLenKV, mAlignKV), ROUND_UP(seqLenQ, mAlignQ) / mQseqSplitNum]   -> [B, K, M]
            // V :     [Batch * numHead / group, ROUND_UP(seqLenKV, mAlignKV), ROUND_UP(headDim, mAlignHDN)] -> [B, K, N]
            // QKV :   [Batch * numHead, ROUND_UP(headDim, mAlignHDN), ROUND_UP(seqLenQ, mAlignQ) / mQseqSplitNum] -> [B, N, M]

            int loop = batch * numHead;
            int e_pack = ROUND_UP(seqlen, mAlignQ);
            int e_pack_piece = e_pack / mQseqSplitNum;
            int l_pack = ROUND_UP(mKvSeqlen, mAlignKV);
            int h_pack = ROUND_UP(headDim, mAlignHDN);

            std::set<std::string> buildOptions;

            uint32_t layout = 0;
            auto param = getGemmParams({(uint32_t)e_pack_piece, (uint32_t)h_pack, (uint32_t)l_pack, layout, (uint32_t)loop, (uint32_t)0}, {openCLBuffer(mTempQK.get()), openCLBuffer(mTempV.get()), openCLBuffer(mTempQKV.get())}, mOpenCLBackend->getOpenCLRuntime(), mOpenCLBackend->getPrecision(), mOpenCLBackend->getCLTuneLevel());

            int KWG=param[0], KWI=param[1], MDIMA=param[2], MDIMC=param[3], MWG=param[4], NDIMB=param[5], NDIMC=param[6], NWG=param[7], SA=param[8], SB=param[9], STRM=param[10], STRN=param[11], VWM=param[12], VWN=param[13];
            buildOptions.emplace("-DKWG=" + std::to_string(KWG));
            buildOptions.emplace("-DKWI=" + std::to_string(KWI));
            buildOptions.emplace("-DMDIMA=" + std::to_string(MDIMA));
            buildOptions.emplace("-DMDIMC=" + std::to_string(MDIMC));
            buildOptions.emplace("-DMWG=" + std::to_string(MWG));
            buildOptions.emplace("-DNDIMB=" + std::to_string(NDIMB));
            buildOptions.emplace("-DNDIMC=" + std::to_string(NDIMC));
            buildOptions.emplace("-DNWG=" + std::to_string(NWG));
            buildOptions.emplace("-DSA=" + std::to_string(SA));
            buildOptions.emplace("-DSB=" + std::to_string(SB));
            buildOptions.emplace("-DSTRM=" + std::to_string(STRM));
            buildOptions.emplace("-DSTRN=" + std::to_string(STRN));
            buildOptions.emplace("-DVWM=" + std::to_string(VWM));
            buildOptions.emplace("-DVWN=" + std::to_string(VWN));
            if(layout >= 4) {
                buildOptions.emplace("-DOUTPUTMN");
            }

            int tileM = MWG;
            int tileN = NWG;
            int localM = MDIMC;
            int localN = NDIMC;

            if(mOpenCLBackend->getOpenCLRuntime()->getGpuType() == GpuType::ADRENO) {
                buildOptions.emplace("-DUSE_CL_MAD=1");
                buildOptions.emplace("-DRELAX_WORKGROUP_SIZE=1");
            }

            mKernel_qkv_vec[seq_idx] = mOpenCLBackend->getOpenCLRuntime()->buildKernel("matmul_params_buf", "XgemmBatched", buildOptions, mOpenCLBackend->getPrecision());

            int out_per_thread_m = tileM / localM;
            int out_per_thread_n = tileN / localN;

            mGwsQkvVec[seq_idx] = {static_cast<uint32_t>(e_pack_piece/out_per_thread_m), static_cast<uint32_t>(h_pack/out_per_thread_n), static_cast<uint32_t>(loop)};
            mLwsQkvVec[seq_idx] = {static_cast<uint32_t>(localM), static_cast<uint32_t>(localN), 1};

            float alpha = 1.0f;
            float beta = 0.0f;
            int batch_offset_a = e_pack_piece * l_pack;
            int batch_offset_b = h_pack * l_pack;
            int batch_offset_c = e_pack * h_pack;
            int batch_offset[4] = {batch_offset_a, batch_offset_b, batch_offset_c, 0};
            int base_ptr_offset[4] = {0, 0, e_pack_piece * seq_idx, 0};
            int stride[4] = {e_pack_piece, h_pack, e_pack, h_pack};
            int group[4] = {1, group_size, 1, loop};

            int idx            = 0;
            cl_int ret = CL_SUCCESS;
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, static_cast<int>(e_pack_piece));
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, static_cast<int>(h_pack));
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, static_cast<int>(l_pack));
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, alpha);
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, beta);
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, openCLBuffer(mTempQK.get()));
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, openCLBuffer(mTempV.get()));
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, openCLBuffer(mTempQKV.get()));
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, batch_offset);
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, base_ptr_offset);
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, stride);
            ret |= mKernel_qkv_vec[seq_idx]->get().setArg(idx++, group);
            MNN_CHECK_CL_SUCCESS(ret, "setArg Self-Attention batchmatmul qkv Kernel");
            mOpenCLBackend->recordKernel3d(mKernel_qkv_vec[seq_idx], mGwsQkvVec[seq_idx], mLwsQkvVec[seq_idx]);
        }
    }

    seq_idx = 0;
    // transpose to output
    {
        // QKV :   [Batch * numHead, ROUND_UP(headDim, mAlignHDN), ROUND_UP(seqLenQ, mAlignQ)] -> [B, N, M]
        // output: [batch, seqLenQ/4, headNum, headDim, seqLenQ_4]
        std::set<std::string> buildOption;

        mKernel_clip_vec[seq_idx] = runtime->buildKernel("attention_buf", "qkv_transpose_output", buildOption, mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_clip_vec[seq_idx]));

        mGwsClipVec[seq_idx] = {static_cast<uint32_t>(UP_DIV(seqlen, 4)), static_cast<uint32_t>(UP_DIV(headDim, 4)), static_cast<uint32_t>(batch*numHead)};

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, mGwsClipVec[seq_idx][0]);
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, mGwsClipVec[seq_idx][1]);
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, mGwsClipVec[seq_idx][2]);
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, openCLBuffer(mTempQKV.get()));
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, openCLBuffer(outputs[0]));
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, mAlignQ);
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, mAlignHDN);
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, seqlen);
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, numHead);
        ret |= mKernel_clip_vec[seq_idx]->get().setArg(index++, headDim);

        mLwsClipVec[seq_idx] = localWS3DDefault(mGwsClipVec[seq_idx], maxWorkGroupSize, runtime, "qkv_transpose_output", mKernel_clip_vec[seq_idx], mOpenCLBackend->getCLTuneLevel(), "attention_buf").first;
        mGwsClipVec[seq_idx][0] = ROUND_UP(mGwsClipVec[seq_idx][0], std::max((uint32_t)1, mLwsClipVec[seq_idx][0]));
        mGwsClipVec[seq_idx][1] = ROUND_UP(mGwsClipVec[seq_idx][1], std::max((uint32_t)1, mLwsClipVec[seq_idx][1]));
        mGwsClipVec[seq_idx][2] = ROUND_UP(mGwsClipVec[seq_idx][2], std::max((uint32_t)1, mLwsClipVec[seq_idx][2]));

        MNN_CHECK_CL_SUCCESS(ret, "setArg qkv_transpose_output");
        mOpenCLBackend->recordKernel3d(mKernel_clip_vec[seq_idx], mGwsClipVec[seq_idx], mLwsClipVec[seq_idx]);
    }
    mOpenCLBackend->endRecord(mRecording);

    return NO_ERROR;
}

ErrorCode AttentionBufExecution::prefillResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs){

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto shape = query->shape();

    int batch = shape[0];
    int seqlen = shape[1];
    int numHead = shape[2];
    int kvNumHead = key->shape()[2];
    int headDim = shape[3];
    int groupSize = numHead / kvNumHead;
    float scale = 1.0 / sqrt(headDim);
    bool usePagedKV = mKVCacheCLManager != nullptr && mKVCacheCLManager->pagedActive();
    mKernelUsePagedKV = usePagedKV;
    mKernelIsDecode = false;

    int maskKvlen = mKvSeqlen;
    int maskQlen = seqlen;

    if(mHasMask) {
        auto mask = inputs[3];
        auto mask_shape = mask->shape();
        int dim = mask->dimensions();
        MNN_ASSERT(dim >= 2);
        maskQlen = mask_shape[dim - 2];
        maskKvlen  = mask_shape[dim - 1];
        if(mIsAddMask) {
            mTempMask.reset(Tensor::createDevice<float>({ROUND_UP(maskQlen, 4) * ROUND_UP(maskKvlen, 4) * batch}));
        } else {
            mTempMask.reset(Tensor::createDevice<uint32_t>({ROUND_UP(maskQlen, 4) * ROUND_UP(maskKvlen, 4) * batch}));
        }
    }

    mTempQ.reset(Tensor::createDevice<float>({ROUND_UP(seqlen, 4) * ROUND_UP(headDim, 4) * numHead * batch}));
    mTempQK.reset(Tensor::createDevice<float>({ROUND_UP(seqlen, 4) * mKvSeqlen * numHead * batch}));
    mTempSoftMax.reset(Tensor::createDevice<float>({ROUND_UP(seqlen, 4) * mKvSeqlen * numHead * batch}));

    mOpenCLBackend->onAcquireBuffer(mTempQK.get(), Backend::DYNAMIC_IN_EXECUTION);
    mOpenCLBackend->onAcquireBuffer(mTempSoftMax.get(), Backend::DYNAMIC_IN_EXECUTION);
    mOpenCLBackend->onAcquireBuffer(mTempQ.get(), Backend::DYNAMIC_IN_EXECUTION);
    if(mHasMask){
        mOpenCLBackend->onAcquireBuffer(mTempMask.get(), Backend::DYNAMIC_IN_EXECUTION);
    }

    cl::Buffer keyBuffer, valueBuffer;
    if(nullptr != mMeta) {
        keyBuffer = *mKVCacheCLManager->key();
        valueBuffer = *mKVCacheCLManager->value();
    } else {
        mTempK.reset(Tensor::createDevice<float>({ROUND_UP(seqlen, 4) * ROUND_UP(headDim, 4) * numHead * batch}));
        mTempV.reset(Tensor::createDevice<float>({ROUND_UP(seqlen, 4) * ROUND_UP(headDim, 4) * numHead * batch}));
        mOpenCLBackend->onAcquireBuffer(mTempK.get(), Backend::DYNAMIC);
        mOpenCLBackend->onAcquireBuffer(mTempV.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mTempV.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mTempK.get(), Backend::DYNAMIC);
        keyBuffer = openCLBuffer(mTempK.get());
        valueBuffer = openCLBuffer(mTempV.get());
    }
    mOpenCLBackend->onReleaseBuffer(mTempQ.get(), Backend::DYNAMIC_IN_EXECUTION);
    mOpenCLBackend->onReleaseBuffer(mTempQK.get(), Backend::DYNAMIC_IN_EXECUTION);
    mOpenCLBackend->onReleaseBuffer(mTempSoftMax.get(), Backend::DYNAMIC_IN_EXECUTION);
    if(mHasMask){
        mOpenCLBackend->onReleaseBuffer(mTempMask.get(), Backend::DYNAMIC_IN_EXECUTION);
    }

    {
        // rearrange query
        std::set<std::string> buildOption;

        mKernel_rearrangeQ = runtime->buildKernel("attention_buf", "rearrange_q", buildOption, mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_rearrangeQ));

        mGlobalWorkSizeRearrgQ = {static_cast<uint32_t>(UP_DIV(seqlen, 4)), \
                                static_cast<uint32_t>(UP_DIV(headDim, 4)), \
                                static_cast<uint32_t>(numHead*batch)};

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_rearrangeQ->get().setArg(index++, mGlobalWorkSizeRearrgQ[0]);
        ret |= mKernel_rearrangeQ->get().setArg(index++, mGlobalWorkSizeRearrgQ[1]);
        ret |= mKernel_rearrangeQ->get().setArg(index++, mGlobalWorkSizeRearrgQ[2]);
        ret |= mKernel_rearrangeQ->get().setArg(index++, openCLBuffer(query));
        ret |= mKernel_rearrangeQ->get().setArg(index++, openCLDeferBuffer(mTempQ.get()));
        ret |= mKernel_rearrangeQ->get().setArg(index++, seqlen);
        ret |= mKernel_rearrangeQ->get().setArg(index++, headDim);
        ret |= mKernel_rearrangeQ->get().setArg(index++, numHead);

        MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_q");
        mLocalWorkSizeRearrgQ = localWS3DDefault(mGlobalWorkSizeRearrgQ, maxWorkGroupSize, runtime, "rearrange_q", mKernel_rearrangeQ, mOpenCLBackend->getCLTuneLevel(), "attention_buf").first;
        mGlobalWorkSizeRearrgQ[0] = ROUND_UP(mGlobalWorkSizeRearrgQ[0], std::max((uint32_t)1, mLocalWorkSizeRearrgQ[0]));
        mGlobalWorkSizeRearrgQ[1] = ROUND_UP(mGlobalWorkSizeRearrgQ[1], std::max((uint32_t)1, mLocalWorkSizeRearrgQ[1]));
        mGlobalWorkSizeRearrgQ[2] = ROUND_UP(mGlobalWorkSizeRearrgQ[2], std::max((uint32_t)1, mLocalWorkSizeRearrgQ[2]));
        mRgQUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &openCLDeferBuffer(mTempQ.get())()});
        mOpRecordUpdateInfo.emplace_back(&mRgQUpdateInfo);
        mOpenCLBackend->recordKernel3d(mKernel_rearrangeQ, mGlobalWorkSizeRearrgQ, mLocalWorkSizeRearrgQ, &mRgQUpdateInfo);
    }
    {
        // rearrange key
        std::set<std::string> buildOption;

        buildOption.emplace("-DOPENCL_PREFILL_ATTENTION");
        mKernel_rearrange = runtime->buildKernel("attention_buf", usePagedKV ? "rearrange_k_paged" : "rearrange_k",
                                                 buildOption, mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_rearrange));

        mGlobalWorkSizeRearrg = {static_cast<uint32_t>(UP_DIV(seqlen, 4)), \
                                static_cast<uint32_t>(UP_DIV(headDim, 4)), \
                                static_cast<uint32_t>(kvNumHead * batch)};

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_rearrange->get().setArg(index++, mGlobalWorkSizeRearrg[0]);
        ret |= mKernel_rearrange->get().setArg(index++, mGlobalWorkSizeRearrg[1]);
        ret |= mKernel_rearrange->get().setArg(index++, mGlobalWorkSizeRearrg[2]);
        ret |= mKernel_rearrange->get().setArg(index++, openCLBuffer(key));
        ret |= mKernel_rearrange->get().setArg(index++, keyBuffer);
        if (usePagedKV) {
            ret |= mKernel_rearrange->get().setArg(index++, *mKVCacheCLManager->tokenTable());
            ret |= mKernel_rearrange->get().setArg(index++, *mKVCacheCLManager->ropeTable());
        }
        ret |= mKernel_rearrange->get().setArg(index++, mPastKvSeqlen);
        ret |= mKernel_rearrange->get().setArg(index++, mKeyValueMaxlen);
        ret |= mKernel_rearrange->get().setArg(index++, seqlen);
        ret |= mKernel_rearrange->get().setArg(index++, kvNumHead);
        ret |= mKernel_rearrange->get().setArg(index++, numHead);
        ret |= mKernel_rearrange->get().setArg(index++, headDim);

        MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_k");
        mLocalWorkSizeRearrg = localWS3DDefault(mGlobalWorkSizeRearrg, maxWorkGroupSize, runtime, "rearrange_k", mKernel_rearrange, mOpenCLBackend->getCLTuneLevel(), "attention_buf").first;
        mGlobalWorkSizeRearrg[0] = ROUND_UP(mGlobalWorkSizeRearrg[0], std::max((uint32_t)1, mLocalWorkSizeRearrg[0]));
        mGlobalWorkSizeRearrg[1] = ROUND_UP(mGlobalWorkSizeRearrg[1], std::max((uint32_t)1, mLocalWorkSizeRearrg[1]));
        mGlobalWorkSizeRearrg[2] = ROUND_UP(mGlobalWorkSizeRearrg[2], std::max((uint32_t)1, mLocalWorkSizeRearrg[2]));
        if(nullptr != mMeta) {
            mRgUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &(*(mKVCacheCLManager->key()))()});
            if (usePagedKV) {
                mRgUpdateInfo.update_kernel_args.push_back({0, 5, sizeof(cl_mem), &(*(mKVCacheCLManager->tokenTable()))()});
                mRgUpdateInfo.update_kernel_args.push_back({0, 6, sizeof(cl_mem), &(*(mKVCacheCLManager->ropeTable()))()});
            }
        }
        mRgUpdateInfo.update_kernel_args.push_back({0, static_cast<uint32_t>(usePagedKV ? 7 : 5), sizeof(mPastKvSeqlen), &mPastKvSeqlen});
        mRgUpdateInfo.update_kernel_args.push_back({0, static_cast<uint32_t>(usePagedKV ? 8 : 6), sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
        mOpRecordUpdateInfo.emplace_back(&mRgUpdateInfo);
        mOpenCLBackend->recordKernel3d(mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg, &mRgUpdateInfo);
    }
    if (mHasMask){
        std::set<std::string> buildOption;
        if(mIsAddMask){
            buildOption.emplace("-DADD_MASK");
        } else if(mHasMask) {
            buildOption.emplace("-DSET_MASK");
        }
        mKernel_rearrangeMask = runtime->buildKernel("attention_buf", "rearrange_mask_shortprefill", buildOption, mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        mGlobalWorkSizeRearrgM =  {static_cast<uint32_t>(UP_DIV(maskQlen, 4)), static_cast<uint32_t>(UP_DIV(maskKvlen, 4)), static_cast<uint32_t>(batch)};
        auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_rearrangeMask));
        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_rearrangeMask->get().setArg(index++, mGlobalWorkSizeRearrgM[0]);
        ret |= mKernel_rearrangeMask->get().setArg(index++, mGlobalWorkSizeRearrgM[1]);
        ret |= mKernel_rearrangeMask->get().setArg(index++, mGlobalWorkSizeRearrgM[2]);
        ret |= mKernel_rearrangeMask->get().setArg(index++, openCLBuffer(inputs[3]));
        ret |= mKernel_rearrangeMask->get().setArg(index++, openCLDeferBuffer(mTempMask.get()));
        ret |= mKernel_rearrangeMask->get().setArg(index++, maskQlen);
        ret |= mKernel_rearrangeMask->get().setArg(index++, maskKvlen);
        MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_mask_shortprefill");
        mLocalWorkSizeRearrgM = localWS3DDefault(mGlobalWorkSizeRearrgM, maxWorkGroupSize, runtime, "rearrange_mask_shortprefill", mKernel_rearrangeMask, mOpenCLBackend->getCLTuneLevel(), "attention_buf").first;
        mGlobalWorkSizeRearrgM[0] = ROUND_UP(mGlobalWorkSizeRearrgM[0], std::max((uint32_t)1, mLocalWorkSizeRearrgM[0]));
        mGlobalWorkSizeRearrgM[1] = ROUND_UP(mGlobalWorkSizeRearrgM[1], std::max((uint32_t)1, mLocalWorkSizeRearrgM[1]));
        mGlobalWorkSizeRearrgM[2] = ROUND_UP(mGlobalWorkSizeRearrgM[2], std::max((uint32_t)1, mLocalWorkSizeRearrgM[2]));
        mRgMUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &openCLDeferBuffer(mTempMask.get())()});
        mOpRecordUpdateInfo.emplace_back(&mRgMUpdateInfo);
        mOpenCLBackend->recordKernel3d(mKernel_rearrangeMask, mGlobalWorkSizeRearrgM, mLocalWorkSizeRearrgM, &mRgMUpdateInfo);
    }
    {
        // matmul qk
        std::set<std::string> buildOption;
        if(mIsAddMask){
            buildOption.emplace("-DADD_MASK");
        } else if(mHasMask) {
            buildOption.emplace("-DSET_MASK");
        }
        buildOption.emplace("-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize));
        mKernel_qk = runtime->buildKernel("attention_buf", usePagedKV ? "matmul_qk_div_mask_prefill_paged" : "matmul_qk_div_mask_prefill",
                                          buildOption, mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        mGlobalWorkSizeQk =  {static_cast<uint32_t>(UP_DIV(seqlen, 4)), static_cast<uint32_t>(UP_DIV(mKvSeqlen, 4)), static_cast<uint32_t>(numHead*batch)};
        auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_qk));

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_qk->get().setArg(index++, mGlobalWorkSizeQk[0]);
        ret |= mKernel_qk->get().setArg(index++, mGlobalWorkSizeQk[1]);
        ret |= mKernel_qk->get().setArg(index++, mGlobalWorkSizeQk[2]);
        ret |= mKernel_qk->get().setArg(index++, openCLDeferBuffer(mTempQ.get()));
        ret |= mKernel_qk->get().setArg(index++, keyBuffer);
        if(mHasMask) {
            ret |= mKernel_qk->get().setArg(index++, openCLDeferBuffer(mTempMask.get()));
        }
        ret |= mKernel_qk->get().setArg(index++, openCLDeferBuffer(mTempQK.get()));
        ret |= mKernel_qk->get().setArg(index++, scale);
        ret |= mKernel_qk->get().setArg(index++, seqlen);
        ret |= mKernel_qk->get().setArg(index++, maskKvlen);
        ret |= mKernel_qk->get().setArg(index++, mKvSeqlen);
        ret |= mKernel_qk->get().setArg(index++, mKeyValueMaxlen);
        ret |= mKernel_qk->get().setArg(index++, numHead);
        ret |= mKernel_qk->get().setArg(index++, headDim);
        if (usePagedKV) {
            ret |= mKernel_qk->get().setArg(index++, *mKVCacheCLManager->tokenTable());
            ret |= mKernel_qk->get().setArg(index++, *mKVCacheCLManager->ropeTable());
            ret |= mKernel_qk->get().setArg(index++, mKVCacheCLManager->pagedRopeDim());
            ret |= mKernel_qk->get().setArg(index++, mKVCacheCLManager->pagedRopeTheta());
        }
        MNN_CHECK_CL_SUCCESS(ret, "setArg matmul_qk_div_mask_prefill");

        mLocalWorkSizeQk = localWS3DDefault(mGlobalWorkSizeQk, maxWorkGroupSize, runtime, "matmul_qk_div_mask_prefill", mKernel_qk, mOpenCLBackend->getCLTuneLevel(), "attention_buf").first;
        mGlobalWorkSizeQk[0] = ROUND_UP(mGlobalWorkSizeQk[0], std::max((uint32_t)1, mLocalWorkSizeQk[0]));
        mGlobalWorkSizeQk[1] = ROUND_UP(mGlobalWorkSizeQk[1], std::max((uint32_t)1, mLocalWorkSizeQk[1]));
        mGlobalWorkSizeQk[2] = ROUND_UP(mGlobalWorkSizeQk[2], std::max((uint32_t)1, mLocalWorkSizeQk[2]));
        mQkUpdateInfo.update_kernel_args.push_back({0, 1, sizeof(mGlobalWorkSizeQk0), &mGlobalWorkSizeQk0});
        mQkUpdateInfo.update_kernel_args.push_back({0, 3, sizeof(cl_mem), &openCLDeferBuffer(mTempQ.get())()});
        if(nullptr != mMeta) {
            mQkUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &(*(mKVCacheCLManager->key()))()});
        }
        if(mHasMask){
            mQkUpdateInfo.update_kernel_args.push_back({0, 5, sizeof(cl_mem), &openCLDeferBuffer(mTempMask.get())()});
            mQkUpdateInfo.update_kernel_args.push_back({0, 6, sizeof(cl_mem), &openCLDeferBuffer(mTempQK.get())()});
            mQkUpdateInfo.update_kernel_args.push_back({0, 10, sizeof(mKvSeqlen), &mKvSeqlen});
            mQkUpdateInfo.update_kernel_args.push_back({0, 11, sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
            if (usePagedKV) {
                mQkUpdateInfo.update_kernel_args.push_back({0, 14, sizeof(cl_mem), &(*(mKVCacheCLManager->tokenTable()))()});
                mQkUpdateInfo.update_kernel_args.push_back({0, 15, sizeof(cl_mem), &(*(mKVCacheCLManager->ropeTable()))()});
            }
        }else{
            mQkUpdateInfo.update_kernel_args.push_back({0, 5, sizeof(cl_mem), &openCLDeferBuffer(mTempQK.get())()});
            mQkUpdateInfo.update_kernel_args.push_back({0, 9, sizeof(mKvSeqlen), &mKvSeqlen});
            mQkUpdateInfo.update_kernel_args.push_back({0, 10, sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
            if (usePagedKV) {
                mQkUpdateInfo.update_kernel_args.push_back({0, 13, sizeof(cl_mem), &(*(mKVCacheCLManager->tokenTable()))()});
                mQkUpdateInfo.update_kernel_args.push_back({0, 14, sizeof(cl_mem), &(*(mKVCacheCLManager->ropeTable()))()});
            }
        }
        mQkPrefillGlobal_size[0] = mGlobalWorkSizeQk[0];
        mQkPrefillGlobal_size[1] = mGlobalWorkSizeQk[1];
        mQkPrefillGlobal_size[2] = mGlobalWorkSizeQk[2];
        mQkUpdateInfo.update_global_size.push_back({0, mQkPrefillGlobal_size});
        mOpRecordUpdateInfo.emplace_back(&mQkUpdateInfo);
        mOpenCLBackend->recordKernel3d(mKernel_qk, mGlobalWorkSizeQk, mLocalWorkSizeQk, &mQkUpdateInfo);
    }
    {
        // softmax
        int inside  = ROUND_UP(seqlen, 4);
        int outside = numHead * batch;
        int localSize = 64;

        std::set<std::string> buildOption;
        buildOption.emplace("-DSOFTMAX_LOCAL_SIZE=" + std::to_string(localSize));
        mKernel_softmax = runtime->buildKernel("softmax_buf", "softmax_v4_buf", buildOption, mOpenCLBackend->getPrecision());
        mGlobalWorkSizeSoftMax = {static_cast<uint32_t>(localSize), static_cast<uint32_t>(UP_DIV(inside, 4)), static_cast<uint32_t>(outside)};
        auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_softmax));

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_softmax->get().setArg(index++, mGlobalWorkSizeSoftMax[0]);
        ret |= mKernel_softmax->get().setArg(index++, mGlobalWorkSizeSoftMax[1]);
        ret |= mKernel_softmax->get().setArg(index++, mGlobalWorkSizeSoftMax[2]);
        ret |= mKernel_softmax->get().setArg(index++, openCLDeferBuffer(mTempQK.get()));
        ret |= mKernel_softmax->get().setArg(index++, openCLDeferBuffer(mTempSoftMax.get()));
        ret |= mKernel_softmax->get().setArg(index++, inside);
        ret |= mKernel_softmax->get().setArg(index++, outside);
        ret |= mKernel_softmax->get().setArg(index++, mKvSeqlen);
        MNN_CHECK_CL_SUCCESS(ret, "setArg softmax");

        mLocalWorkSizeSoftMax = {static_cast<uint32_t>(localSize), 1, 1};
        if(localSize == 1){
            mLocalWorkSizeSoftMax = localWS3DDefault(mGlobalWorkSizeSoftMax, maxWorkGroupSize, runtime, "softmax", mKernel_softmax, mOpenCLBackend->getCLTuneLevel(), "softmax_buf").first;
        }
        mGlobalWorkSizeSoftMax[0] = ROUND_UP(mGlobalWorkSizeSoftMax[0], std::max((uint32_t)1, mLocalWorkSizeSoftMax[0]));
        mGlobalWorkSizeSoftMax[1] = ROUND_UP(mGlobalWorkSizeSoftMax[1], std::max((uint32_t)1, mLocalWorkSizeSoftMax[1]));
        mGlobalWorkSizeSoftMax[2] = ROUND_UP(mGlobalWorkSizeSoftMax[2], std::max((uint32_t)1, mLocalWorkSizeSoftMax[2]));
        mSoftMaxUpdateInfo.update_kernel_args.push_back({0, 3, sizeof(cl_mem), &openCLDeferBuffer(mTempQK.get())()});
        mSoftMaxUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &openCLDeferBuffer(mTempSoftMax.get())()});
        mSoftMaxUpdateInfo.update_kernel_args.push_back({0, 7, sizeof(mKvSeqlen), &mKvSeqlen});
        mOpRecordUpdateInfo.emplace_back(&mSoftMaxUpdateInfo);
        mOpenCLBackend->recordKernel3d(mKernel_softmax, mGlobalWorkSizeSoftMax, mLocalWorkSizeSoftMax, &mSoftMaxUpdateInfo);
    }
    {
        // rearrange value
        std::set<std::string> buildOption;

        buildOption.emplace("-DOPENCL_PREFILL_ATTENTION");
        mKernel_rearrangeV = runtime->buildKernel("attention_buf", usePagedKV ? "rearrange_v_paged" : "rearrange_v",
                                                  buildOption, mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_rearrangeV));

        mGlobalWorkSizeRearrgV = {static_cast<uint32_t>(UP_DIV(headDim, 4)), \
            static_cast<uint32_t>(UP_DIV(seqlen, 4)), \
            static_cast<uint32_t>(kvNumHead * batch)};

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_rearrangeV->get().setArg(index++, mGlobalWorkSizeRearrgV[0]);
        ret |= mKernel_rearrangeV->get().setArg(index++, mGlobalWorkSizeRearrgV[1]);
        ret |= mKernel_rearrangeV->get().setArg(index++, mGlobalWorkSizeRearrgV[2]);
        ret |= mKernel_rearrangeV->get().setArg(index++, openCLBuffer(value));
        ret |= mKernel_rearrangeV->get().setArg(index++, valueBuffer);
        if (usePagedKV) {
            ret |= mKernel_rearrangeV->get().setArg(index++, *mKVCacheCLManager->tokenTable());
        }
        ret |= mKernel_rearrangeV->get().setArg(index++, mPastKvSeqlen);
        ret |= mKernel_rearrangeV->get().setArg(index++, mKeyValueMaxlen);
        ret |= mKernel_rearrangeV->get().setArg(index++, seqlen);
        ret |= mKernel_rearrangeV->get().setArg(index++, kvNumHead);
        ret |= mKernel_rearrangeV->get().setArg(index++, headDim);

        MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_v");
        mLocalWorkSizeRearrgV = localWS3DDefault(mGlobalWorkSizeRearrgV, maxWorkGroupSize, runtime, "rearrange_v", mKernel_rearrangeV, mOpenCLBackend->getCLTuneLevel(), "attention_buf").first;
        mGlobalWorkSizeRearrgV[0] = ROUND_UP(mGlobalWorkSizeRearrgV[0], std::max((uint32_t)1, mLocalWorkSizeRearrgV[0]));
        mGlobalWorkSizeRearrgV[1] = ROUND_UP(mGlobalWorkSizeRearrgV[1], std::max((uint32_t)1, mLocalWorkSizeRearrgV[1]));
        mGlobalWorkSizeRearrgV[2] = ROUND_UP(mGlobalWorkSizeRearrgV[2], std::max((uint32_t)1, mLocalWorkSizeRearrgV[2]));
        if(nullptr != mMeta) {
            mRgVUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &(*(mKVCacheCLManager->value()))()});
            if (usePagedKV) {
                mRgVUpdateInfo.update_kernel_args.push_back({0, 5, sizeof(cl_mem), &(*(mKVCacheCLManager->tokenTable()))()});
            }
        }
        mRgVUpdateInfo.update_kernel_args.push_back({0, static_cast<uint32_t>(usePagedKV ? 6 : 5), sizeof(mPastKvSeqlen), &mPastKvSeqlen});
        mRgVUpdateInfo.update_kernel_args.push_back({0, static_cast<uint32_t>(usePagedKV ? 7 : 6), sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
        mOpRecordUpdateInfo.emplace_back(&mRgVUpdateInfo);
        mOpenCLBackend->recordKernel3d(mKernel_rearrangeV, mGlobalWorkSizeRearrgV, mLocalWorkSizeRearrgV, &mRgVUpdateInfo);
    }
    // qk * value
    {
        std::set<std::string> buildOption;
        buildOption.emplace("-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize));
        mKernel_qkv = runtime->buildKernel("attention_buf", usePagedKV ? "matmul_qkv_prefill_paged" : "matmul_qkv_prefill",
                                           buildOption, mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_qkv));
        mGlobalWorkSizeQkv =  {static_cast<uint32_t>(UP_DIV(headDim, 8)), static_cast<uint32_t>(UP_DIV(seqlen, 4)), static_cast<uint32_t>(numHead*batch)};

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_qkv->get().setArg(index++, mGlobalWorkSizeQkv[0]);
        ret |= mKernel_qkv->get().setArg(index++, mGlobalWorkSizeQkv[1]);
        ret |= mKernel_qkv->get().setArg(index++, mGlobalWorkSizeQkv[2]);
        ret |= mKernel_qkv->get().setArg(index++, openCLDeferBuffer(mTempSoftMax.get()));
        ret |= mKernel_qkv->get().setArg(index++, valueBuffer);
        ret |= mKernel_qkv->get().setArg(index++, openCLBuffer(outputs[0]));
        ret |= mKernel_qkv->get().setArg(index++, seqlen);
        ret |= mKernel_qkv->get().setArg(index++, mKvSeqlen);
        ret |= mKernel_qkv->get().setArg(index++, mKeyValueMaxlen);
        ret |= mKernel_qkv->get().setArg(index++, numHead);
        ret |= mKernel_qkv->get().setArg(index++, kvNumHead);
        ret |= mKernel_qkv->get().setArg(index++, headDim);
        if (usePagedKV) {
            ret |= mKernel_qkv->get().setArg(index++, *mKVCacheCLManager->tokenTable());
        }
        MNN_CHECK_CL_SUCCESS(ret, "setArg matmul_qkv_prefill");

        mLocalWorkSizeQkv = localWS3DDefault(mGlobalWorkSizeQkv, maxWorkGroupSize, runtime, "matmul_qkv_prefill", mKernel_qkv, mOpenCLBackend->getCLTuneLevel(), "attention_buf").first;
        mGlobalWorkSizeQkv[0] = ROUND_UP(mGlobalWorkSizeQkv[0], std::max((uint32_t)1, mLocalWorkSizeQkv[0]));
        mGlobalWorkSizeQkv[1] = ROUND_UP(mGlobalWorkSizeQkv[1], std::max((uint32_t)1, mLocalWorkSizeQkv[1]));
        mGlobalWorkSizeQkv[2] = ROUND_UP(mGlobalWorkSizeQkv[2], std::max((uint32_t)1, mLocalWorkSizeQkv[2]));
        mQkvUpdateInfo.update_kernel_args.push_back({0, 3, sizeof(cl_mem), &openCLDeferBuffer(mTempSoftMax.get())()});
        if(nullptr != mMeta) {
            mQkvUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &(*(mKVCacheCLManager->value()))()});
            if (usePagedKV) {
                mQkvUpdateInfo.update_kernel_args.push_back({0, 12, sizeof(cl_mem), &(*(mKVCacheCLManager->tokenTable()))()});
            }
        }
        mQkvUpdateInfo.update_kernel_args.push_back({0, 7, sizeof(mKvSeqlen), &mKvSeqlen});
        mQkvUpdateInfo.update_kernel_args.push_back({0, 8, sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
        mOpRecordUpdateInfo.emplace_back(&mQkvUpdateInfo);
        mOpenCLBackend->recordKernel3d(mKernel_qkv, mGlobalWorkSizeQkv, mLocalWorkSizeQkv, &mQkvUpdateInfo);
    }
    mOpenCLBackend->endRecord(mRecording);

    return NO_ERROR;
}

ErrorCode AttentionBufExecution::decodeResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs){

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto shape = query->shape();

    int batch = shape[0];
    int seqlen = shape[1];
    int numHead = shape[2];
    int kvNumHead = key->shape()[2];
    int headDim = shape[3];
    int group_size = numHead / kvNumHead;
    float scale = 1.0 / sqrt(headDim);
    bool usePagedKV = mKVCacheCLManager != nullptr && mKVCacheCLManager->pagedActive();
    mKernelUsePagedKV = usePagedKV;
    mKernelIsDecode = true;


    cl::Buffer keyBuffer, valueBuffer;
    if(nullptr != mMeta) {
        keyBuffer = *mKVCacheCLManager->key();
        valueBuffer = *mKVCacheCLManager->value();
    } else {
        mTempK.reset(Tensor::createDevice<float>({ROUND_UP(seqlen, 4) * ROUND_UP(headDim, 4) * numHead * batch}));
        mTempV.reset(Tensor::createDevice<float>({ROUND_UP(seqlen, 4) * ROUND_UP(headDim, 4) * numHead * batch}));
        mOpenCLBackend->onAcquireBuffer(mTempK.get(), Backend::DYNAMIC);
        mOpenCLBackend->onAcquireBuffer(mTempV.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mTempV.get(), Backend::DYNAMIC);
        mOpenCLBackend->onReleaseBuffer(mTempK.get(), Backend::DYNAMIC);
        keyBuffer = openCLBuffer(mTempK.get());
        valueBuffer = openCLBuffer(mTempV.get());
    }

    mTempQK.reset(Tensor::createDevice<float>({mDecodeTmpMaxlen * numHead}));
    mTempSoftMax.reset(Tensor::createDevice<float>({mDecodeTmpMaxlen * numHead}));
    mOpenCLBackend->onAcquireBuffer(mTempQK.get(), Backend::DYNAMIC_IN_EXECUTION);
    mOpenCLBackend->onAcquireBuffer(mTempSoftMax.get(), Backend::DYNAMIC_IN_EXECUTION);
    mOpenCLBackend->onReleaseBuffer(mTempQK.get(), Backend::DYNAMIC_IN_EXECUTION);
    mOpenCLBackend->onReleaseBuffer(mTempSoftMax.get(), Backend::DYNAMIC_IN_EXECUTION);
    {
        // rearrange key
        std::set<std::string> buildOption;

        mKernel_rearrange = runtime->buildKernel("attention_buf", usePagedKV ? "rearrange_k_paged" : "rearrange_k",
                                                 buildOption, mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_rearrange));

        mGlobalWorkSizeRearrg = {static_cast<uint32_t>(1), \
                                static_cast<uint32_t>(UP_DIV(headDim, 4)), \
                                static_cast<uint32_t>(kvNumHead * batch)};

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_rearrange->get().setArg(index++, mGlobalWorkSizeRearrg[0]);
        ret |= mKernel_rearrange->get().setArg(index++, mGlobalWorkSizeRearrg[1]);
        ret |= mKernel_rearrange->get().setArg(index++, mGlobalWorkSizeRearrg[2]);
        ret |= mKernel_rearrange->get().setArg(index++, openCLBuffer(key));
        ret |= mKernel_rearrange->get().setArg(index++, keyBuffer);
        if (usePagedKV) {
            ret |= mKernel_rearrange->get().setArg(index++, *mKVCacheCLManager->tokenTable());
            ret |= mKernel_rearrange->get().setArg(index++, *mKVCacheCLManager->ropeTable());
        }
        ret |= mKernel_rearrange->get().setArg(index++, mPastKvSeqlen);
        ret |= mKernel_rearrange->get().setArg(index++, mKeyValueMaxlen);
        ret |= mKernel_rearrange->get().setArg(index++, seqlen);
        ret |= mKernel_rearrange->get().setArg(index++, kvNumHead);
        ret |= mKernel_rearrange->get().setArg(index++, numHead);
        ret |= mKernel_rearrange->get().setArg(index++, headDim);

        MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_k");
        mLocalWorkSizeRearrg = localWS3DDefault(mGlobalWorkSizeRearrg, maxWorkGroupSize, runtime, "rearrange_k", mKernel_rearrange, mOpenCLBackend->getCLTuneLevel(), "attention_buf").first;
        mGlobalWorkSizeRearrg[0] = ROUND_UP(mGlobalWorkSizeRearrg[0], std::max((uint32_t)1, mLocalWorkSizeRearrg[0]));
        mGlobalWorkSizeRearrg[1] = ROUND_UP(mGlobalWorkSizeRearrg[1], std::max((uint32_t)1, mLocalWorkSizeRearrg[1]));
        mGlobalWorkSizeRearrg[2] = ROUND_UP(mGlobalWorkSizeRearrg[2], std::max((uint32_t)1, mLocalWorkSizeRearrg[2]));
        if(nullptr != mMeta) {
            mRgUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &(*(mKVCacheCLManager->key()))()});
            if (usePagedKV) {
                mRgUpdateInfo.update_kernel_args.push_back({0, 5, sizeof(cl_mem), &(*(mKVCacheCLManager->tokenTable()))()});
                mRgUpdateInfo.update_kernel_args.push_back({0, 6, sizeof(cl_mem), &(*(mKVCacheCLManager->ropeTable()))()});
            }
            mRgUpdateInfo.update_kernel_args.push_back({0, static_cast<uint32_t>(usePagedKV ? 7 : 5), sizeof(mPastKvSeqlen), &mPastKvSeqlen});
            mRgUpdateInfo.update_kernel_args.push_back({0, static_cast<uint32_t>(usePagedKV ? 8 : 6), sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
            mOpRecordUpdateInfo.emplace_back(&mRgUpdateInfo);
            mOpenCLBackend->recordKernel3d(mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg, &mRgUpdateInfo);
        } else {
            mOpenCLBackend->recordKernel3d(mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg);
        }
    }
    {
        // matmul qk
        std::set<std::string> buildOption;
        buildOption.emplace("-DNUMHEAD_GROUP_SIZE=" + std::to_string(group_size));
        mKernel_qk = runtime->buildKernel("attention_buf", usePagedKV ? "matmul_qk_decode_paged" : "matmul_qk_decode",
                                          buildOption, mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        mGlobalWorkSizeQk =  {static_cast<uint32_t>(UP_DIV(mKvSeqlen, 4)), static_cast<uint32_t>(numHead)};
        auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_qk));

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_qk->get().setArg(index++, mGlobalWorkSizeQk[0]);
        ret |= mKernel_qk->get().setArg(index++, mGlobalWorkSizeQk[1]);
        ret |= mKernel_qk->get().setArg(index++, openCLBuffer(query));
        ret |= mKernel_qk->get().setArg(index++, keyBuffer);
        ret |= mKernel_qk->get().setArg(index++, openCLDeferBuffer(mTempQK.get()));
        ret |= mKernel_qk->get().setArg(index++, scale);
        ret |= mKernel_qk->get().setArg(index++, mKvSeqlen);
        ret |= mKernel_qk->get().setArg(index++, mKeyValueMaxlen);
        ret |= mKernel_qk->get().setArg(index++, numHead);
        ret |= mKernel_qk->get().setArg(index++, headDim);
        if (usePagedKV) {
            ret |= mKernel_qk->get().setArg(index++, *mKVCacheCLManager->tokenTable());
            ret |= mKernel_qk->get().setArg(index++, *mKVCacheCLManager->ropeTable());
            ret |= mKernel_qk->get().setArg(index++, mKVCacheCLManager->pagedRopeDim());
            ret |= mKernel_qk->get().setArg(index++, mKVCacheCLManager->pagedRopeTheta());
        }
        MNN_CHECK_CL_SUCCESS(ret, "setArg matmul_qk_decode");

        mLocalWorkSizeQk = localWS2DDefault(mGlobalWorkSizeQk, maxWorkGroupSize, runtime, "matmul_qk_decode", mKernel_qk, mOpenCLBackend->getCLTuneLevel(), "attention_buf").first;
        mGlobalWorkSizeQk[0] = ROUND_UP(mGlobalWorkSizeQk[0], std::max((uint32_t)1, mLocalWorkSizeQk[0]));
        mGlobalWorkSizeQk[1] = ROUND_UP(mGlobalWorkSizeQk[1], std::max((uint32_t)1, mLocalWorkSizeQk[1]));
        if(nullptr != mMeta) {
            mQkUpdateInfo.update_kernel_args.push_back({0, 0, sizeof(mGlobalWorkSizeQk0), &mGlobalWorkSizeQk0});
            mQkUpdateInfo.update_kernel_args.push_back({0, 3, sizeof(cl_mem), &(*(mKVCacheCLManager->key()))()});
            mQkUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &openCLDeferBuffer(mTempQK.get())()});
            mQkUpdateInfo.update_kernel_args.push_back({0, 6, sizeof(mKvSeqlen), &mKvSeqlen});
            mQkUpdateInfo.update_kernel_args.push_back({0, 7, sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
            if (usePagedKV) {
                mQkUpdateInfo.update_kernel_args.push_back({0, 10, sizeof(cl_mem), &(*(mKVCacheCLManager->tokenTable()))()});
                mQkUpdateInfo.update_kernel_args.push_back({0, 11, sizeof(cl_mem), &(*(mKVCacheCLManager->ropeTable()))()});
            }
            mQkGlobal_size[0] = mGlobalWorkSizeQk[0];
            mQkGlobal_size[1] = mGlobalWorkSizeQk[1];
            mQkUpdateInfo.update_global_size.push_back({0, mQkGlobal_size});
            mOpRecordUpdateInfo.emplace_back(&mQkUpdateInfo);
            mOpenCLBackend->recordKernel2d(mKernel_qk, mGlobalWorkSizeQk, mLocalWorkSizeQk, &mQkUpdateInfo);
        } else {
            mOpenCLBackend->recordKernel2d(mKernel_qk, mGlobalWorkSizeQk, mLocalWorkSizeQk);
        }
    }
    {
        // softmax
        int inside  = 1;
        int outside = numHead;
        int localSize = 64;

        std::set<std::string> buildOption;
        buildOption.emplace("-DSOFTMAX_LOCAL_SIZE=" + std::to_string(localSize));
        mKernel_softmax = runtime->buildKernel("softmax_buf", "softmax_in1_buf", buildOption, mOpenCLBackend->getPrecision());
        mGlobalWorkSizeSoftMax = {static_cast<uint32_t>(localSize), static_cast<uint32_t>(inside), static_cast<uint32_t>(outside)};
        auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_softmax));

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_softmax->get().setArg(index++, mGlobalWorkSizeSoftMax[0]);
        ret |= mKernel_softmax->get().setArg(index++, mGlobalWorkSizeSoftMax[1]);
        ret |= mKernel_softmax->get().setArg(index++, mGlobalWorkSizeSoftMax[2]);
        ret |= mKernel_softmax->get().setArg(index++, openCLDeferBuffer(mTempQK.get()));
        ret |= mKernel_softmax->get().setArg(index++, openCLDeferBuffer(mTempSoftMax.get()));
        ret |= mKernel_softmax->get().setArg(index++, inside);
        ret |= mKernel_softmax->get().setArg(index++, outside);
        ret |= mKernel_softmax->get().setArg(index++, mKvSeqlen);
        MNN_CHECK_CL_SUCCESS(ret, "setArg softmax");

        mLocalWorkSizeSoftMax = {static_cast<uint32_t>(localSize), 1, 1};
        if(localSize == 1){
            mLocalWorkSizeSoftMax = localWS3DDefault(mGlobalWorkSizeSoftMax, maxWorkGroupSize, runtime, "softmax", mKernel_softmax, mOpenCLBackend->getCLTuneLevel(), "softmax_buf").first;
        }
        mGlobalWorkSizeSoftMax[0] = ROUND_UP(mGlobalWorkSizeSoftMax[0], std::max((uint32_t)1, mLocalWorkSizeSoftMax[0]));
        mGlobalWorkSizeSoftMax[1] = ROUND_UP(mGlobalWorkSizeSoftMax[1], std::max((uint32_t)1, mLocalWorkSizeSoftMax[1]));
        mGlobalWorkSizeSoftMax[2] = ROUND_UP(mGlobalWorkSizeSoftMax[2], std::max((uint32_t)1, mLocalWorkSizeSoftMax[2]));
        if(nullptr != mMeta) {
            mSoftMaxUpdateInfo.update_kernel_args.push_back({0, 3, sizeof(cl_mem), &openCLDeferBuffer(mTempQK.get())()});
            mSoftMaxUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &openCLDeferBuffer(mTempSoftMax.get())()});
            mSoftMaxUpdateInfo.update_kernel_args.push_back({0, 7, sizeof(mKvSeqlen), &mKvSeqlen});
            mOpRecordUpdateInfo.emplace_back(&mSoftMaxUpdateInfo);
            mOpenCLBackend->recordKernel3d(mKernel_softmax, mGlobalWorkSizeSoftMax, mLocalWorkSizeSoftMax, &mSoftMaxUpdateInfo);
        } else {
            mOpenCLBackend->recordKernel3d(mKernel_softmax, mGlobalWorkSizeSoftMax, mLocalWorkSizeSoftMax);
        }
    }
    {
        // rearrange value
        std::set<std::string> buildOption;

        mKernel_rearrangeV = runtime->buildKernel("attention_buf", usePagedKV ? "rearrange_v_paged" : "rearrange_v",
                                                  buildOption, mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        auto maxWorkGroupSize  = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel_rearrangeV));

        mGlobalWorkSizeRearrgV = {static_cast<uint32_t>(UP_DIV(headDim, 4)), \
                                static_cast<uint32_t>(1), \
                                static_cast<uint32_t>(kvNumHead * batch)};

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_rearrangeV->get().setArg(index++, mGlobalWorkSizeRearrgV[0]);
        ret |= mKernel_rearrangeV->get().setArg(index++, mGlobalWorkSizeRearrgV[1]);
        ret |= mKernel_rearrangeV->get().setArg(index++, mGlobalWorkSizeRearrgV[2]);
        ret |= mKernel_rearrangeV->get().setArg(index++, openCLBuffer(value));
        ret |= mKernel_rearrangeV->get().setArg(index++, valueBuffer);
        if (usePagedKV) {
            ret |= mKernel_rearrangeV->get().setArg(index++, *mKVCacheCLManager->tokenTable());
        }
        ret |= mKernel_rearrangeV->get().setArg(index++, mPastKvSeqlen);
        ret |= mKernel_rearrangeV->get().setArg(index++, mKeyValueMaxlen);
        ret |= mKernel_rearrangeV->get().setArg(index++, seqlen);
        ret |= mKernel_rearrangeV->get().setArg(index++, kvNumHead);
        ret |= mKernel_rearrangeV->get().setArg(index++, headDim);

        MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_v");
        mLocalWorkSizeRearrgV = localWS3DDefault(mGlobalWorkSizeRearrgV, maxWorkGroupSize, runtime, "rearrange_v", mKernel_rearrangeV, mOpenCLBackend->getCLTuneLevel(), "attention_buf").first;
        mGlobalWorkSizeRearrgV[0] = ROUND_UP(mGlobalWorkSizeRearrgV[0], std::max((uint32_t)1, mLocalWorkSizeRearrgV[0]));
        mGlobalWorkSizeRearrgV[1] = ROUND_UP(mGlobalWorkSizeRearrgV[1], std::max((uint32_t)1, mLocalWorkSizeRearrgV[1]));
        mGlobalWorkSizeRearrgV[2] = ROUND_UP(mGlobalWorkSizeRearrgV[2], std::max((uint32_t)1, mLocalWorkSizeRearrgV[2]));
        if(nullptr != mMeta) {
            mRgVUpdateInfo.update_kernel_args.push_back({0, 4, sizeof(cl_mem), &(*(mKVCacheCLManager->value()))()});
            if (usePagedKV) {
                mRgVUpdateInfo.update_kernel_args.push_back({0, 5, sizeof(cl_mem), &(*(mKVCacheCLManager->tokenTable()))()});
            }
            mRgVUpdateInfo.update_kernel_args.push_back({0, static_cast<uint32_t>(usePagedKV ? 6 : 5), sizeof(mPastKvSeqlen), &mPastKvSeqlen});
            mRgVUpdateInfo.update_kernel_args.push_back({0, static_cast<uint32_t>(usePagedKV ? 7 : 6), sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
            mOpRecordUpdateInfo.emplace_back(&mRgVUpdateInfo);
            mOpenCLBackend->recordKernel3d(mKernel_rearrangeV, mGlobalWorkSizeRearrgV, mLocalWorkSizeRearrgV, &mRgVUpdateInfo);
        } else {
            mOpenCLBackend->recordKernel3d(mKernel_rearrangeV, mGlobalWorkSizeRearrgV, mLocalWorkSizeRearrgV);
        }
    }
    // qk * value
    {
        std::set<std::string> buildOption;
        buildOption.emplace("-DNUMHEAD_GROUP_SIZE=" + std::to_string(group_size));
        const int total_kernel = 2;
        std::string kernelName[total_kernel] = {
            usePagedKV ? "matmul_qkv_decode_b4_paged" : "matmul_qkv_decode_b4",
            usePagedKV ? "matmul_qkv_decode_b8_paged" : "matmul_qkv_decode_b8"
        };
        std::string unroll[total_kernel] = {"-DLOOP_UNROLL_4", "-DLOOP_UNROLL_8"};
        int itemC[total_kernel] = {4, 8};
        int actual_kernel = 2;
        std::shared_ptr<KernelWrap> kernel[total_kernel * total_kernel];
        std::vector<uint32_t> globalWorkSize[total_kernel * total_kernel];
        std::vector<uint32_t> localWorkSize[total_kernel * total_kernel];
        std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)

        for (int i = 0; i < actual_kernel; i++) {
            for(int j = 0; j < actual_kernel; j++){
                int knl_idx = i * total_kernel + j;
                auto option = buildOption;
                option.emplace(unroll[j]);
                kernel[knl_idx] = mOpenCLBackend->getOpenCLRuntime()->buildKernel("attention_buf", kernelName[i], option, mOpenCLBackend->getPrecision());
                uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));
                globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(headDim, itemC[i])), static_cast<uint32_t>(numHead)};
                uint32_t index = 0;
                cl_int ret = CL_SUCCESS;
                ret |= kernel[knl_idx]->get().setArg(index++, globalWorkSize[knl_idx][0]);
                ret |= kernel[knl_idx]->get().setArg(index++, globalWorkSize[knl_idx][1]);
                ret |= kernel[knl_idx]->get().setArg(index++, openCLDeferBuffer(mTempSoftMax.get()));
                ret |= kernel[knl_idx]->get().setArg(index++, valueBuffer);
                ret |= kernel[knl_idx]->get().setArg(index++, openCLBuffer(outputs[0]));
                ret |= kernel[knl_idx]->get().setArg(index++, mKvSeqlen);
                ret |= kernel[knl_idx]->get().setArg(index++, mKeyValueMaxlen);
                ret |= kernel[knl_idx]->get().setArg(index++, numHead);
                ret |= kernel[knl_idx]->get().setArg(index++, kvNumHead);
                ret |= kernel[knl_idx]->get().setArg(index++, headDim);
                if (usePagedKV) {
                    ret |= kernel[knl_idx]->get().setArg(index++, *mKVCacheCLManager->tokenTable());
                }
                MNN_CHECK_CL_SUCCESS(ret, "setArg matmul_qkv_decode");
                std::pair<std::vector<uint32_t>, int> retTune;
                retTune = localWS2DDefault(globalWorkSize[knl_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName[i] + unroll[j], kernel[knl_idx], mOpenCLBackend->getCLTuneLevel(), "attention_buf");
                if(min_cost.first > retTune.second) {
                    min_cost.first = retTune.second;
                    min_cost.second = knl_idx;
                    mLocalWorkSizeQkv = {retTune.first[0], retTune.first[1]};
                }
            }
        }
        int min_index  = min_cost.second / 2;
        int min_index_unroll  = min_cost.second % 2;
        mGlobalWorkSizeQkv = {globalWorkSize[min_cost.second][0], globalWorkSize[min_cost.second][1]};
        buildOption.emplace(unroll[min_index_unroll]);
        mKernel_qkv = runtime->buildKernel("attention_buf", kernelName[min_index], buildOption, mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);

        uint32_t index = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mKernel_qkv->get().setArg(index++, mGlobalWorkSizeQkv[0]);
        ret |= mKernel_qkv->get().setArg(index++, mGlobalWorkSizeQkv[1]);
        ret |= mKernel_qkv->get().setArg(index++, openCLDeferBuffer(mTempSoftMax.get()));
        ret |= mKernel_qkv->get().setArg(index++, valueBuffer);
        ret |= mKernel_qkv->get().setArg(index++, openCLBuffer(outputs[0]));
        ret |= mKernel_qkv->get().setArg(index++, mKvSeqlen);
        ret |= mKernel_qkv->get().setArg(index++, mKeyValueMaxlen);
        ret |= mKernel_qkv->get().setArg(index++, numHead);
        ret |= mKernel_qkv->get().setArg(index++, kvNumHead);
        ret |= mKernel_qkv->get().setArg(index++, headDim);
        if (usePagedKV) {
            ret |= mKernel_qkv->get().setArg(index++, *mKVCacheCLManager->tokenTable());
        }
        MNN_CHECK_CL_SUCCESS(ret, "setArg matmul_qkv_decode");

        mGlobalWorkSizeQkv[0] = ROUND_UP(mGlobalWorkSizeQkv[0], std::max((uint32_t)1, mLocalWorkSizeQkv[0]));
        mGlobalWorkSizeQkv[1] = ROUND_UP(mGlobalWorkSizeQkv[1], std::max((uint32_t)1, mLocalWorkSizeQkv[1]));
        if(nullptr != mMeta) {
            mQkvUpdateInfo.update_kernel_args.push_back({0, 2, sizeof(cl_mem), &openCLDeferBuffer(mTempSoftMax.get())()});
            mQkvUpdateInfo.update_kernel_args.push_back({0, 3, sizeof(cl_mem), &(*(mKVCacheCLManager->value()))()});
            if (usePagedKV) {
                mQkvUpdateInfo.update_kernel_args.push_back({0, 10, sizeof(cl_mem), &(*(mKVCacheCLManager->tokenTable()))()});
            }
            mQkvUpdateInfo.update_kernel_args.push_back({0, 5, sizeof(mKvSeqlen), &mKvSeqlen});
            mQkvUpdateInfo.update_kernel_args.push_back({0, 6, sizeof(mKeyValueMaxlen), &mKeyValueMaxlen});
            mOpRecordUpdateInfo.emplace_back(&mQkvUpdateInfo);
            mOpenCLBackend->recordKernel2d(mKernel_qkv, mGlobalWorkSizeQkv, mLocalWorkSizeQkv, &mQkvUpdateInfo);
        } else {
            mOpenCLBackend->recordKernel2d(mKernel_qkv, mGlobalWorkSizeQkv, mLocalWorkSizeQkv);
        }
    }
    mOpenCLBackend->endRecord(mRecording);

    return NO_ERROR;
}

// [Batch, q_seqlen, HeadNum, HeadDim] -> [Batch, kv_seqlen, HeadNum, HeadDim]
ErrorCode AttentionBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mOpenCLBackend->startRecord(mRecording);
    auto shape = inputs[0]->shape();

    int batch = shape[0];
    int seqlen = shape[1];
    int numHead = shape[2];
    int headDim = shape[3];
    int kvNumHead = inputs[1]->shape()[2];
    if(nullptr != mMeta) {
        // if has kv_cache, default has mask
//        MNN_ASSERT(inputs.size() > 3);
    }
    mHasMask = inputs.size() > 3;
    mIsDecode = seqlen == 1 && mMeta->add == 1;

    // reset updateArgs variable and kernel vector
    init();
    // handle kv_cache, like copy kv
    handleKVCache(inputs, outputs);

    mLongPrefill = false;
    if(mIsDecode) {
        return decodeResize(inputs, outputs);
    } else {
        if(mPastKvSeqlen == 0){
            std::pair<std::vector<uint32_t>, uint32_t> tuneInfo;
            std::string info = "attention_" + std::to_string(batch) + "_" + std::to_string(numHead) + "_" + std::to_string(headDim) + "_" + std::to_string(kvNumHead);
            if(seqlen > 16){
                if(getTunedInfo(info, {static_cast<unsigned int>(seqlen)}, tuneInfo, mOpenCLBackend->getOpenCLRuntime())){
                    mLongPrefill = tuneInfo.first[0];
                } else{
                    if (mOpenCLBackend->getCLTuneLevel() == Heavy || mOpenCLBackend->getCLTuneLevel() == Wide){
                        setRecordClose closeRecord(mOpenCLBackend);
                        // tunning choose use witch preill
                        prefillResize(inputs, outputs);
                        auto shortPrefillTime = getExecuteTime();
                        init();
                        mLongPrefill = true;
                        longPrefillResize(inputs, outputs);
                        auto longPrefillTime = getExecuteTime();
                        mLongPrefill = false;
                        if(longPrefillTime < shortPrefillTime){
                            mLongPrefill = true;
                        }
                        std::pair<std::vector<uint32_t>, uint32_t> tuneInfoTmp = std::make_pair<std::vector<uint32_t>, uint32_t>({mLongPrefill}, 0);
                        setTunedInfo(info, {static_cast<unsigned int>(seqlen)}, tuneInfoTmp, mOpenCLBackend->getOpenCLRuntime(), "attention_buf");
                        init();
                    }else{
                        if(seqlen > 512){
                            mLongPrefill = true;
                        }
                    }
                }
            }
        }
        if(mLongPrefill){
            longPrefillResize(inputs, outputs);
        }else{
            prefillResize(inputs, outputs);
        }
    }

    return NO_ERROR;
}

int AttentionBufExecution::getExecuteTime(){
    int executeTime = 0;
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if(mLongPrefill) {
        int seq_idx = 0;
        cl::Event event0, event1, event2, event3, event4, event5, event6;
        run3DKernelDefault(mKernel_rearrange_vec[seq_idx], mGwsRearrgVec[seq_idx], mLwsRearrgVec[seq_idx], mOpenCLBackend->getOpenCLRuntime(), &event0);
        executeTime += runtime->getEventTime(event0);
        if(mHasMask) {
            run3DKernelDefault(mKernel_mask_vec[seq_idx], mGwsMaskVec[seq_idx], mLwsMaskVec[seq_idx], mOpenCLBackend->getOpenCLRuntime(), &event1);
            executeTime += runtime->getEventTime(event1);
        }
        for(int seq_idx = 0; seq_idx < mQseqSplitNum; seq_idx++) {
            run3DKernelDefault(mKernel_qk_vec[seq_idx], mGwsQkVec[seq_idx], mLwsQkVec[seq_idx], mOpenCLBackend->getOpenCLRuntime(), &event2);
            executeTime += runtime->getEventTime(event2);
            run3DKernelDefault(mKernel_softmax_vec[seq_idx], mGwsSoftMaxVec[seq_idx], mLwsSoftMaxVec[seq_idx], mOpenCLBackend->getOpenCLRuntime(), &event3);
            executeTime += runtime->getEventTime(event3);
            run3DKernelDefault(mKernel_trans_vec[seq_idx], mGwsTransVec[seq_idx], mLwsTransVec[seq_idx], mOpenCLBackend->getOpenCLRuntime(), &event4);
            executeTime += runtime->getEventTime(event4);
            run3DKernelDefault(mKernel_qkv_vec[seq_idx], mGwsQkvVec[seq_idx], mLwsQkvVec[seq_idx], mOpenCLBackend->getOpenCLRuntime(), &event5);
            executeTime += runtime->getEventTime(event5);
        }
        seq_idx = 0;
        run3DKernelDefault(mKernel_clip_vec[seq_idx], mGwsClipVec[seq_idx], mLwsClipVec[seq_idx], mOpenCLBackend->getOpenCLRuntime(), &event6);
        executeTime += runtime->getEventTime(event6);
    } else{
        cl::Event event0, event1, event2, event3, event4, event5, event6;
        run3DKernelDefault(mKernel_rearrangeQ, mGlobalWorkSizeRearrgQ, mLocalWorkSizeRearrgQ, mOpenCLBackend->getOpenCLRuntime(), &event0);
        executeTime += runtime->getEventTime(event0);
        run3DKernelDefault(mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg, mOpenCLBackend->getOpenCLRuntime(), &event1);
        executeTime += runtime->getEventTime(event1);
        if(mHasMask) {
            run3DKernelDefault(mKernel_rearrangeMask, mGlobalWorkSizeRearrgM, mLocalWorkSizeRearrgM, mOpenCLBackend->getOpenCLRuntime(), &event2);
            executeTime += runtime->getEventTime(event2);
        }
        run3DKernelDefault(mKernel_qk, mGlobalWorkSizeQk, mLocalWorkSizeQk, mOpenCLBackend->getOpenCLRuntime(), &event3);
        executeTime += runtime->getEventTime(event3);
        run3DKernelDefault(mKernel_softmax, mGlobalWorkSizeSoftMax, mLocalWorkSizeSoftMax, mOpenCLBackend->getOpenCLRuntime(), &event4);
        executeTime += runtime->getEventTime(event4);
        run3DKernelDefault(mKernel_rearrangeV, mGlobalWorkSizeRearrgV, mLocalWorkSizeRearrgV, mOpenCLBackend->getOpenCLRuntime(), &event5);
        executeTime += runtime->getEventTime(event5);
        run3DKernelDefault(mKernel_qkv, mGlobalWorkSizeQkv, mLocalWorkSizeQkv, mOpenCLBackend->getOpenCLRuntime(), &event6);
        executeTime += runtime->getEventTime(event6);
    }
    return executeTime;
}

ErrorCode AttentionBufExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start AttentionBufExecution onExecute !\n");
#endif
    if(nullptr != mMeta){
        auto shape = inputs[0]->shape();
        int seqlen = shape[1];
        mKVCacheCLManager->reallocKVCache(mMeta, seqlen);
        bool cachePreparedBySubclass = false;
        int appendKvSeqLen = seqlen;
        auto prepareErr = onPrepareKVCacheBeforeAppend(inputs, cachePreparedBySubclass, appendKvSeqLen);
        if (prepareErr != NO_ERROR) {
            return prepareErr;
        }
        if (mKVCacheCLManager->pagedActive() && !cachePreparedBySubclass) {
            int logicalStart = mKVCacheCLManager->pastKvLength();
            bool appendOk = appendKvSeqLen > 1 ?
                mKVCacheCLManager->ensurePagedAppendTable(logicalStart, appendKvSeqLen, true) :
                mKVCacheCLManager->ensurePagedDecodeAppendTable(logicalStart, appendKvSeqLen);
            if (!appendOk) {
                return OUT_OF_MEMORY;
            }
        }
        bool usePagedKV = mKVCacheCLManager->pagedActive();
        bool needKernelRebuild = (usePagedKV != mKernelUsePagedKV) || (mIsDecode != mKernelIsDecode) ||
                                 (usePagedKV && mLongPrefill);
        if (needKernelRebuild) {
            init();
            mPastKvSeqlen = mKVCacheCLManager->pastKvLength();
            mKvSeqlen = mPastKvSeqlen + appendKvSeqLen;
            mKeyValueMaxlen = ROUND_UP(mKVCacheCLManager->maxLength(), 4);
            mDecodeTmpMaxlen = mKeyValueMaxlen;
            if (usePagedKV) {
                // Paged PrefixAttention uses the short kernels that consume the token table.
                mLongPrefill = false;
            }
            auto resizeErr = mIsDecode ? decodeResize(inputs, outputs) : prefillResize(inputs, outputs);
            if (resizeErr != NO_ERROR) {
                return resizeErr;
            }
        }
    }
    UpdateArgs(inputs, outputs);
    const bool captureKernelEvents = onShouldProfileAttentionKernelEvents();
    onBeforeAttentionComputeEnqueue();
#ifdef ENABLE_OPENCL_TIME_PROFILER
    if(mLongPrefill) {
        int seq_idx = 0;
        cl::Event event0, event1, event2, event3, event4, event5, event6;
        run3DKernelDefault(mKernel_rearrange_vec[seq_idx], mGwsRearrgVec[seq_idx], mLwsRearrgVec[seq_idx], mOpenCLBackend->getOpenCLRuntime(), &event0);
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_qkv", event0});
        onAttentionKernelEvent("rearrange_qkv", event0);
        if(mHasMask) {
            run3DKernelDefault(mKernel_mask_vec[seq_idx], mGwsMaskVec[seq_idx], mLwsMaskVec[seq_idx], mOpenCLBackend->getOpenCLRuntime(), &event1);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_mask", event1});
            onAttentionKernelEvent("rearrange_mask", event1);
        }
        for(int seq_idx = 0; seq_idx < mQseqSplitNum; seq_idx++) {
            run3DKernelDefault(mKernel_qk_vec[seq_idx], mGwsQkVec[seq_idx], mLwsQkVec[seq_idx], mOpenCLBackend->getOpenCLRuntime(), &event2);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"matmul_qk_div_mask", event2});
            onAttentionKernelEvent("matmul_qk_div_mask", event2);
            run3DKernelDefault(mKernel_softmax_vec[seq_idx], mGwsSoftMaxVec[seq_idx], mLwsSoftMaxVec[seq_idx], mOpenCLBackend->getOpenCLRuntime(), &event3);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"softmax", event3});
            onAttentionKernelEvent("softmax", event3);
            run3DKernelDefault(mKernel_trans_vec[seq_idx], mGwsTransVec[seq_idx], mLwsTransVec[seq_idx], mOpenCLBackend->getOpenCLRuntime(), &event4);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"transpose_softmax", event4});
            onAttentionKernelEvent("transpose_softmax", event4);
            run3DKernelDefault(mKernel_qkv_vec[seq_idx], mGwsQkvVec[seq_idx], mLwsQkvVec[seq_idx], mOpenCLBackend->getOpenCLRuntime(), &event5);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"matmul_qkv", event5});
            onAttentionKernelEvent("matmul_qkv", event5);
        }
        seq_idx = 0;
        run3DKernelDefault(mKernel_clip_vec[seq_idx], mGwsClipVec[seq_idx], mLwsClipVec[seq_idx], mOpenCLBackend->getOpenCLRuntime(), &event6);
        mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_output", event6});
        onAttentionKernelEvent("rearrange_output", event6);
    } else{
        if(mIsDecode){
            cl::Event event0, event1, event2, event3, event4;
            run3DKernelDefault(mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg, mOpenCLBackend->getOpenCLRuntime(), &event0);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_k", event0});
            onAttentionKernelEvent("rearrange_k", event0);
            runKernel2D(mKernel_qk, mGlobalWorkSizeQk, mLocalWorkSizeQk, mOpenCLBackend->getOpenCLRuntime(), &event1);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"matmul_qk_div_mask", event1});
            onAttentionKernelEvent("matmul_qk_div_mask", event1);
            run3DKernelDefault(mKernel_softmax, mGlobalWorkSizeSoftMax, mLocalWorkSizeSoftMax, mOpenCLBackend->getOpenCLRuntime(), &event2);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"softmax", event2});
            onAttentionKernelEvent("softmax", event2);
            run3DKernelDefault(mKernel_rearrangeV, mGlobalWorkSizeRearrgV, mLocalWorkSizeRearrgV, mOpenCLBackend->getOpenCLRuntime(), &event3);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_v", event3});
            onAttentionKernelEvent("rearrange_v", event3);
            runKernel2D(mKernel_qkv, mGlobalWorkSizeQkv, mLocalWorkSizeQkv, mOpenCLBackend->getOpenCLRuntime(), &event4);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"matmul_qkv", event4});
            onAttentionKernelEvent("matmul_qkv", event4);
        }else{
            cl::Event event0, event1, event2, event3, event4, event5, event6;
            run3DKernelDefault(mKernel_rearrangeQ, mGlobalWorkSizeRearrgQ, mLocalWorkSizeRearrgQ, mOpenCLBackend->getOpenCLRuntime(), &event0);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_q", event0});
            onAttentionKernelEvent("rearrange_q", event0);
            run3DKernelDefault(mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg, mOpenCLBackend->getOpenCLRuntime(), &event1);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_k", event1});
            onAttentionKernelEvent("rearrange_k", event1);
            if(mHasMask) {
                run3DKernelDefault(mKernel_rearrangeMask, mGlobalWorkSizeRearrgM, mLocalWorkSizeRearrgM, mOpenCLBackend->getOpenCLRuntime(), &event2);
                mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_mask_shortprefill", event2});
                onAttentionKernelEvent("rearrange_mask_shortprefill", event2);
            }
            run3DKernelDefault(mKernel_qk, mGlobalWorkSizeQk, mLocalWorkSizeQk, mOpenCLBackend->getOpenCLRuntime(), &event3);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"matmul_qk_div_mask", event3});
            onAttentionKernelEvent("matmul_qk_div_mask", event3);
            run3DKernelDefault(mKernel_softmax, mGlobalWorkSizeSoftMax, mLocalWorkSizeSoftMax, mOpenCLBackend->getOpenCLRuntime(), &event4);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"softmax", event4});
            onAttentionKernelEvent("softmax", event4);
            run3DKernelDefault(mKernel_rearrangeV, mGlobalWorkSizeRearrgV, mLocalWorkSizeRearrgV, mOpenCLBackend->getOpenCLRuntime(), &event5);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"rearrange_v", event5});
            onAttentionKernelEvent("rearrange_v", event5);
            run3DKernelDefault(mKernel_qkv, mGlobalWorkSizeQkv, mLocalWorkSizeQkv, mOpenCLBackend->getOpenCLRuntime(), &event6);
            mOpenCLBackend->getOpenCLRuntime()->pushEvent({"matmul_qkv", event6});
            onAttentionKernelEvent("matmul_qkv", event6);
        }
    }
#else
    if(mOpenCLBackend->isUseRecordQueue() && !captureKernelEvents){
        mOpenCLBackend->addRecord(mRecording, mOpRecordUpdateInfo);
        auto saveErr = saveOpenCLNativePrefixCache(inputs, mKvSeqlen);
        if (saveErr != NO_ERROR) {
            return saveErr;
        }
#ifdef LOG_VERBOSE
        MNN_PRINT("End AttentionBufExecution onExecute... \n");
#endif
        return NO_ERROR;
    }

    auto run3DProfiled = [&](const char* name, const ::std::shared_ptr<KernelWrap>& kernel,
                             const std::vector<uint32_t>& gws, const std::vector<uint32_t>& lws) {
        if (captureKernelEvents) {
            cl::Event event;
            run3DKernelDefault(kernel, gws, lws, mOpenCLBackend->getOpenCLRuntime(), &event);
            onAttentionKernelEvent(name, event);
        } else {
            run3DKernelDefault(kernel, gws, lws, mOpenCLBackend->getOpenCLRuntime());
        }
    };
    auto run2DProfiled = [&](const char* name, const ::std::shared_ptr<KernelWrap>& kernel,
                             const std::vector<uint32_t>& gws, const std::vector<uint32_t>& lws) {
        if (captureKernelEvents) {
            cl::Event event;
            runKernel2D(kernel, gws, lws, mOpenCLBackend->getOpenCLRuntime(), &event);
            onAttentionKernelEvent(name, event);
        } else {
            runKernel2D(kernel, gws, lws, mOpenCLBackend->getOpenCLRuntime());
        }
    };

    if(mLongPrefill) {
        int seq_idx = 0;
        run3DProfiled("rearrange_qkv", mKernel_rearrange_vec[seq_idx], mGwsRearrgVec[seq_idx], mLwsRearrgVec[seq_idx]);
        if(mHasMask) {
            run3DProfiled("rearrange_mask", mKernel_mask_vec[seq_idx], mGwsMaskVec[seq_idx], mLwsMaskVec[seq_idx]);
        }
        for(int seq_idx = 0; seq_idx < mQseqSplitNum; seq_idx++) {
            run3DProfiled("matmul_qk_div_mask", mKernel_qk_vec[seq_idx], mGwsQkVec[seq_idx], mLwsQkVec[seq_idx]);
            run3DProfiled("softmax", mKernel_softmax_vec[seq_idx], mGwsSoftMaxVec[seq_idx], mLwsSoftMaxVec[seq_idx]);
            run3DProfiled("transpose_softmax", mKernel_trans_vec[seq_idx], mGwsTransVec[seq_idx], mLwsTransVec[seq_idx]);
            run3DProfiled("matmul_qkv", mKernel_qkv_vec[seq_idx], mGwsQkvVec[seq_idx], mLwsQkvVec[seq_idx]);

        }
        seq_idx = 0;
        run3DProfiled("rearrange_output", mKernel_clip_vec[seq_idx], mGwsClipVec[seq_idx], mLwsClipVec[seq_idx]);
    } else{
        if(mIsDecode){
            run3DProfiled("rearrange_k", mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg);
            run2DProfiled("matmul_qk_div_mask", mKernel_qk, mGlobalWorkSizeQk, mLocalWorkSizeQk);
            run3DProfiled("softmax", mKernel_softmax, mGlobalWorkSizeSoftMax, mLocalWorkSizeSoftMax);
            run3DProfiled("rearrange_v", mKernel_rearrangeV, mGlobalWorkSizeRearrgV, mLocalWorkSizeRearrgV);
            run2DProfiled("matmul_qkv", mKernel_qkv, mGlobalWorkSizeQkv, mLocalWorkSizeQkv);
        }else{
            run3DProfiled("rearrange_q", mKernel_rearrangeQ, mGlobalWorkSizeRearrgQ, mLocalWorkSizeRearrgQ);
            run3DProfiled("rearrange_k", mKernel_rearrange, mGlobalWorkSizeRearrg, mLocalWorkSizeRearrg);
            if(mHasMask) {
            run3DProfiled("rearrange_mask_shortprefill", mKernel_rearrangeMask, mGlobalWorkSizeRearrgM, mLocalWorkSizeRearrgM);
            }
            run3DProfiled("matmul_qk_div_mask", mKernel_qk, mGlobalWorkSizeQk, mLocalWorkSizeQk);
            run3DProfiled("softmax", mKernel_softmax, mGlobalWorkSizeSoftMax, mLocalWorkSizeSoftMax);
            run3DProfiled("rearrange_v", mKernel_rearrangeV, mGlobalWorkSizeRearrgV, mLocalWorkSizeRearrgV);
            run3DProfiled("matmul_qkv", mKernel_qkv, mGlobalWorkSizeQkv, mLocalWorkSizeQkv);
        }
    }
#endif
    onAfterAttentionComputeEnqueue();

#ifdef LOG_VERBOSE
    MNN_PRINT("end AttentionBufExecution onExecute !\n");
#endif

    auto saveErr = saveOpenCLNativePrefixCache(inputs, mKvSeqlen);
    if (saveErr != NO_ERROR) {
        return saveErr;
    }
    return NO_ERROR;
}

AttentionBufExecution::AttentionBufExecution(const MNN::Op *op, Backend* backend, bool kv_cahce) : CommonExecution(backend, op) {
    mMeta = (KVMeta*)(backend->getMetaPtr());
    mKVCacheCLManager.reset(new KVCacheCLManager(backend, nullptr != mMeta));
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("softmax_buf", "softmax_buf", {"-DSOFTMAX_LOCAL_SIZE=512"}, mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL_CTOR(kernel);
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
}

AttentionBufExecution::AttentionBufExecution(std::shared_ptr<KVCacheCLManager> manager, const MNN::Op *op, Backend *backend) : CommonExecution(backend, op), mKVCacheCLManager(manager) {
    mMeta = (KVMeta*)(backend->getMetaPtr());
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("softmax_buf", "softmax_buf", {"-DSOFTMAX_LOCAL_SIZE=512"}, mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL_CTOR(kernel);
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
}

bool AttentionBufExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    if (bn->getMetaPtr() == backend()->getMetaPtr()) {
        *dst = new AttentionBufExecution(mKVCacheCLManager, op, bn);
    } else {
        *dst = new AttentionBufExecution(op, bn, true);
    }
    return true;
}

class AttentionBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        auto param = op->main_as_AttentionParam();
        OPENCL_CREATOR_CHECK(new AttentionBufExecution(op, backend, param->kv_cache()));
    }
};
REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(AttentionBufCreator, OpType_Attention, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif/* MNN_SUPPORT_TRANSFORMER_FUSE */
