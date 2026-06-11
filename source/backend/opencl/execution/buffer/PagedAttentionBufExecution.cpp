//
//  PagedAttentionBufExecution.cpp
//  MNN
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/PagedAttentionBufExecution.hpp"
#include "core/MNNFileUtils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <atomic>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace MNN {
namespace OpenCL {
namespace {

static int _reverseCount(const PagedKVMeta* meta) {
    if (meta == nullptr || meta->n_reserve <= 0 || meta->reserve == nullptr) {
        return 0;
    }
    return std::max(0, meta->computeReverseSize());
}

static bool _profilePagedAttention() {
    const char* value = ::getenv("MNN_PAGED_ATTENTION_PROFILE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static uint64_t _nowUs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

static bool _writeBinaryFile(const std::string& path, const std::vector<int8_t>& data) {
    std::ofstream os(path, std::ios::binary);
    if (!os.good()) {
        return false;
    }
    if (!data.empty()) {
        os.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
    return os.good();
}

struct ExternalLayerReadSegment {
    bool ok = false;
    bool directWritten = false;
    int sourceSlotStart = 0;
    std::string error;
    std::vector<int8_t> keyData;
    std::vector<int8_t> valueData;
};

struct ExternalLayerReadResult {
    bool ok = true;
    int layerIndex = -1;
    std::string requestKey;
    std::string error;
    std::vector<ExternalLayerReadSegment> segments;
};

struct ExternalLayerReadTask {
    std::string requestKey;
    int layerIndex = -1;
    std::shared_future<std::shared_ptr<ExternalLayerReadResult>> future;
};

static std::mutex gExternalLayerReadMutex;
static std::unordered_map<std::string, ExternalLayerReadTask> gExternalLayerReadTasks;

static bool _envFlagEnabled(const char* name, bool defaultValue) {
    const char* value = ::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    return value[0] != '0';
}

static bool _picOpenCLDebug() {
    return _envFlagEnabled("MNN_PIC_DECODE_DEBUG", false);
}

static bool _legacyB863976OpenCL() {
    return _envFlagEnabled("MNN_PAGED_ATTENTION_OPENCL_LEGACY_B863976", false);
}

static int _envIntValue(const char* name, int defaultValue) {
    const char* value = ::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value) {
        return defaultValue;
    }
    parsed = std::max<long>(std::numeric_limits<int>::min(), std::min<long>(std::numeric_limits<int>::max(), parsed));
    return static_cast<int>(parsed);
}

struct SparsePrefillPiece {
    SparsePrefillPiece() = default;
    SparsePrefillPiece(int qStartValue, int qLenValue, int activeKvLenValue)
        : qStart(qStartValue), qLen(qLenValue), activeKvLen(activeKvLenValue) {
    }
    int qStart = 0;
    int qLen = 0;
    int activeKvLen = 0;
};

static std::vector<SparsePrefillPiece> _buildFixedSparsePieces(const std::vector<int>& logicalIndices, int activeLen,
                                                               int kvLen, int qChunkLen) {
    std::vector<SparsePrefillPiece> pieces;
    if (activeLen <= 0 || kvLen <= 0 || qChunkLen <= 0) {
        return pieces;
    }
    pieces.reserve(static_cast<size_t>(UP_DIV(activeLen, qChunkLen)));
    for (int qStart = 0; qStart < activeLen; qStart += qChunkLen) {
        const int qPieceLen = std::min(qChunkLen, activeLen - qStart);
        int pieceMaxLogical = -1;
        if (static_cast<int>(logicalIndices.size()) >= qStart + qPieceLen) {
            for (int qi = 0; qi < qPieceLen; ++qi) {
                pieceMaxLogical = std::max(pieceMaxLogical, logicalIndices[static_cast<size_t>(qStart + qi)]);
            }
        } else {
            pieceMaxLogical = kvLen - 1;
        }
        pieces.push_back({qStart, qPieceLen, std::max(1, std::min(kvLen, pieceMaxLogical + 1))});
    }
    return pieces;
}

static std::vector<SparsePrefillPiece> _buildRangeAwareSparsePieces(const std::vector<int>& logicalIndices,
                                                                    int activeLen, int kvLen, int qChunkLen) {
    if (activeLen <= 0 || kvLen <= 0 || qChunkLen <= 4 ||
        static_cast<int>(logicalIndices.size()) < activeLen) {
        return _buildFixedSparsePieces(logicalIndices, activeLen, kvLen, qChunkLen);
    }
    const int groupCount = UP_DIV(activeLen, 4);
    const int maxGroupsPerPiece = std::max(1, qChunkLen / 4);
    const int basePieces = std::max(1, UP_DIV(activeLen, qChunkLen));
    constexpr int pieceMultiplier = 2;
    constexpr int maxPiecesEnv = 32;
    const int maxPieces = std::min(groupCount, std::min(maxPiecesEnv, std::max(basePieces, basePieces * pieceMultiplier)));
    constexpr uint64_t launchPenalty = 96;
    if (maxPieces <= basePieces) {
        return _buildFixedSparsePieces(logicalIndices, activeLen, kvLen, qChunkLen);
    }

    std::vector<int> groupMaxLogical(static_cast<size_t>(groupCount), -1);
    for (int group = 0; group < groupCount; ++group) {
        const int qStart = group * 4;
        const int qLen = std::min(4, activeLen - qStart);
        for (int qi = 0; qi < qLen; ++qi) {
            groupMaxLogical[static_cast<size_t>(group)] =
                std::max(groupMaxLogical[static_cast<size_t>(group)],
                         logicalIndices[static_cast<size_t>(qStart + qi)]);
        }
    }

    const uint64_t inf = std::numeric_limits<uint64_t>::max() / 4;
    std::vector<std::vector<uint64_t>> dp(static_cast<size_t>(maxPieces + 1),
                                         std::vector<uint64_t>(static_cast<size_t>(groupCount + 1), inf));
    std::vector<std::vector<int>> prev(static_cast<size_t>(maxPieces + 1),
                                       std::vector<int>(static_cast<size_t>(groupCount + 1), -1));
    dp[0][0] = 0;
    for (int piece = 1; piece <= maxPieces; ++piece) {
        for (int end = 1; end <= groupCount; ++end) {
            int maxLogical = -1;
            const int startMin = std::max(0, end - maxGroupsPerPiece);
            for (int start = end - 1; start >= startMin; --start) {
                maxLogical = std::max(maxLogical, groupMaxLogical[static_cast<size_t>(start)]);
                if (dp[static_cast<size_t>(piece - 1)][static_cast<size_t>(start)] == inf) {
                    continue;
                }
                const int activeKvLen = std::max(1, std::min(kvLen, maxLogical + 1));
                const uint64_t cost =
                    dp[static_cast<size_t>(piece - 1)][static_cast<size_t>(start)] +
                    static_cast<uint64_t>(end - start) * static_cast<uint64_t>(UP_DIV(activeKvLen, 4)) +
                    (piece > 1 ? launchPenalty : 0);
                if (cost < dp[static_cast<size_t>(piece)][static_cast<size_t>(end)]) {
                    dp[static_cast<size_t>(piece)][static_cast<size_t>(end)] = cost;
                    prev[static_cast<size_t>(piece)][static_cast<size_t>(end)] = start;
                }
            }
        }
    }

    int bestPieces = basePieces;
    uint64_t bestCost = inf;
    for (int piece = basePieces; piece <= maxPieces; ++piece) {
        const auto cost = dp[static_cast<size_t>(piece)][static_cast<size_t>(groupCount)];
        if (cost < bestCost) {
            bestCost = cost;
            bestPieces = piece;
        }
    }
    if (bestCost == inf) {
        return _buildFixedSparsePieces(logicalIndices, activeLen, kvLen, qChunkLen);
    }

    std::vector<int> boundaries;
    int cursor = groupCount;
    for (int piece = bestPieces; piece > 0; --piece) {
        const int start = prev[static_cast<size_t>(piece)][static_cast<size_t>(cursor)];
        if (start < 0) {
            return _buildFixedSparsePieces(logicalIndices, activeLen, kvLen, qChunkLen);
        }
        boundaries.emplace_back(cursor);
        cursor = start;
    }
    boundaries.emplace_back(0);
    std::reverse(boundaries.begin(), boundaries.end());

    std::vector<SparsePrefillPiece> pieces;
    pieces.reserve(static_cast<size_t>(bestPieces));
    for (int i = 0; i < bestPieces; ++i) {
        const int groupStart = boundaries[static_cast<size_t>(i)];
        const int groupEnd = boundaries[static_cast<size_t>(i + 1)];
        const int qStart = groupStart * 4;
        const int qLen = std::min(activeLen - qStart, (groupEnd - groupStart) * 4);
        if (qLen <= 0) {
            continue;
        }
        int maxLogical = -1;
        for (int group = groupStart; group < groupEnd; ++group) {
            maxLogical = std::max(maxLogical, groupMaxLogical[static_cast<size_t>(group)]);
        }
        pieces.push_back({qStart, qLen, std::max(1, std::min(kvLen, maxLogical + 1))});
    }
    if (pieces.empty()) {
        return _buildFixedSparsePieces(logicalIndices, activeLen, kvLen, qChunkLen);
    }
    return pieces;
}

static int _prefillQChunkLen(int seqLen, int kvLen, int batch, int numHeads, int layerCount) {
    if (seqLen <= 0 || kvLen <= 0 || batch <= 0 || numHeads <= 0) {
        return 0;
    }
    int chunk = _envIntValue("MNN_PAGED_ATTENTION_OPENCL_PREFILL_Q_CHUNK", -1);
    if (chunk == 0) {
        return seqLen;
    }
    if (chunk < 0) {
        const int maxAutoChunk = std::max(1, _envIntValue("MNN_PAGED_ATTENTION_OPENCL_PREFILL_Q_CHUNK_MAX", 512));
        const int budgetMb = std::max(64, _envIntValue("MNN_PAGED_ATTENTION_OPENCL_PREFILL_QK_BUDGET_MB", 5120));
        layerCount = std::max(1, layerCount);
        const size_t bytesPerQuery =
            static_cast<size_t>(2) * static_cast<size_t>(kvLen) * static_cast<size_t>(numHeads) *
            static_cast<size_t>(batch) * sizeof(float);
        const size_t budgetBytes = static_cast<size_t>(budgetMb) * 1024 * 1024;
        int budgetChunk = maxAutoChunk;
        if (bytesPerQuery > 0) {
            budgetChunk = static_cast<int>(
                std::max<size_t>(1, budgetBytes / (bytesPerQuery * static_cast<size_t>(layerCount))));
        }
        chunk = std::min(seqLen, std::min(maxAutoChunk, budgetChunk));
    }
    chunk = std::max(1, std::min(chunk, seqLen));
    if (chunk < seqLen) {
        chunk = ((chunk + 3) / 4) * 4;
    }
    return std::max(1, std::min(chunk, seqLen));
}

static size_t _fastPrefillScratchBytes(int seqLen, int kvLen, int batch, int numHeads, int kvHeads, int headDim,
                                       int qChunkLen) {
    const int seqPack = ROUND_UP(seqLen, 4);
    const int qChunkPack = ROUND_UP(std::max(1, std::min(qChunkLen, seqLen)), 4);
    const int kvPack = ROUND_UP(kvLen, 4);
    const int headPack4 = ROUND_UP(headDim, 4);
    const int headPack8 = ROUND_UP(headDim, 8);
    const size_t elemBytes = sizeof(float);
    size_t bytes = 0;
    bytes += static_cast<size_t>(seqPack) * headPack4 * numHeads * batch * elemBytes;
    bytes += static_cast<size_t>(kvPack) * headPack4 * kvHeads * batch * elemBytes;
    bytes += static_cast<size_t>(kvPack) * headPack8 * kvHeads * batch * elemBytes;
    bytes += static_cast<size_t>(seqPack) * kvPack * batch * elemBytes;
    bytes += static_cast<size_t>(qChunkPack) * kvLen * numHeads * batch * elemBytes;
    bytes += static_cast<size_t>(qChunkPack) * kvLen * numHeads * batch * elemBytes;
    return bytes;
}

static bool _useDirectValuePrefillForSparse(const PagedKVMeta* meta, int activeLen) {
    if (meta == nullptr) {
        return false;
    }
    if (!meta->cacheblend_score_ready || meta->cacheblend_score_pic_token_count <= 0) {
        return false;
    }
    constexpr int minActive = 384;
    if (activeLen < minActive) {
        return false;
    }
    const int selected = meta->cacheblend_score_selected_local_indices.empty()
        ? meta->cacheblend_score_top_k
        : static_cast<int>(meta->cacheblend_score_selected_local_indices.size());
    constexpr int minRatioPercent = 40;
    return selected > 0 &&
        selected * 100 >= meta->cacheblend_score_pic_token_count * minRatioPercent;
}

static uint32_t _sparseFlashLaneWidth(const PagedKVMeta* meta, int activeLen) {
    if (activeLen < 384) {
        return 64u;
    }
    if (meta != nullptr && meta->cacheblend_score_ready && meta->cacheblend_score_pic_token_count > 0) {
        const int selected = meta->cacheblend_score_selected_local_indices.empty()
            ? meta->cacheblend_score_top_k
            : static_cast<int>(meta->cacheblend_score_selected_local_indices.size());
        if (selected * 100 >= meta->cacheblend_score_pic_token_count * 50) {
            return 64u;
        }
    }
    return 32u;
}

static bool _useStaticFullPrefill(int seqLen, int kvLen, int batch, int numHeads, int kvHeads, int headDim) {
    if (seqLen <= 0 || kvLen <= 0 || batch <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0) {
        return false;
    }
    if (_envFlagEnabled("MNN_PAGED_ATTENTION_OPENCL_DISABLE_STATIC_FAST_PREFILL", false)) {
        return false;
    }
    const int forcedChunk = _envIntValue("MNN_PAGED_ATTENTION_OPENCL_PREFILL_Q_CHUNK", -1);
    if (forcedChunk > 0 && forcedChunk < seqLen) {
        return false;
    }
    const int maxStaticTokens =
        std::max(0, _envIntValue("MNN_PAGED_ATTENTION_OPENCL_STATIC_PREFILL_MAX_TOKENS", 2048));
    if (maxStaticTokens == 0 || seqLen > maxStaticTokens || kvLen > maxStaticTokens) {
        return false;
    }
    const int budgetMb =
        std::max(64, _envIntValue("MNN_PAGED_ATTENTION_OPENCL_STATIC_PREFILL_BUDGET_MB", 1024));
    const size_t budgetBytes = static_cast<size_t>(budgetMb) * 1024 * 1024;
    return _fastPrefillScratchBytes(seqLen, kvLen, batch, numHeads, kvHeads, headDim, seqLen) <= budgetBytes;
}

static bool _allowDirectPagedCacheOpenCL() {
    if (_legacyB863976OpenCL()) {
        return false;
    }
    if (!_envFlagEnabled("MNN_PAGED_ATTENTION_ZERO_COPY_CACHE", true)) {
        return false;
    }
    return true;
}

static bool _requireDirectPagedCacheOpenCL() {
    if (_legacyB863976OpenCL()) {
        return false;
    }
    if (!_envFlagEnabled("MNN_PAGED_ATTENTION_ZERO_COPY_CACHE", true)) {
        return false;
    }
    return !_envFlagEnabled("MNN_PAGED_ATTENTION_OPENCL_ALLOW_KV_STAGING_FALLBACK", false);
}

static bool _shouldUseDirectPagedCacheOpenCL() {
    return _allowDirectPagedCacheOpenCL();
}

static bool _slotTableIsIdentity(const PagedKVMeta* meta, int requiredSlots) {
    if (requiredSlots <= 0) {
        return false;
    }
    if (meta == nullptr) {
        return true;
    }
    if (meta->request_base != 0 || static_cast<int>(meta->slot_table_host.size()) < requiredSlots) {
        return false;
    }
    for (int i = 0; i < requiredSlots; ++i) {
        if (meta->slot_table_host[i] != i) {
            return false;
        }
    }
    return true;
}

static size_t _segmentTokenCount(const std::vector<PagedKVExternalSegment>& segments) {
    size_t count = 0;
    for (const auto& segment : segments) {
        count += segment.tokenCount;
    }
    return count;
}

static int _picCacheSourceSlotBase(const PagedKVMeta* meta, int kvLen) {
    if (meta == nullptr) {
        return kvLen;
    }
    return std::max(kvLen, meta->request_capacity);
}

static size_t _picCacheSourceSlotCount(const PagedKVMeta* meta) {
    if (meta == nullptr) {
        return 0;
    }
    return std::max({meta->external_source_slot_reserve,
                     _segmentTokenCount(meta->external_segments),
                     _segmentTokenCount(meta->cacheblend_score_segments)});
}

struct ExternalLayerMappedTarget {
    std::shared_ptr<Tensor> key;
    std::shared_ptr<Tensor> value;
    std::shared_ptr<cl::CommandQueue> queue;
    int batch = 0;
    int kvHeads = 0;
    int headDim = 0;
    int bytes = 0;
    int maxSlots = 0;
};

struct ExternalLayerMappedTargetRef {
    std::weak_ptr<Tensor> key;
    std::weak_ptr<Tensor> value;
    std::shared_ptr<cl::CommandQueue> queue;
    int batch = 0;
    int kvHeads = 0;
    int headDim = 0;
    int bytes = 0;
    int maxSlots = 0;
};

static std::mutex gExternalLayerMappedTargetMutex;
static std::unordered_map<std::string, ExternalLayerMappedTargetRef> gExternalLayerMappedTargets;

static std::string _externalLayerMappedTargetKey(const PagedKVMeta* meta, int layerIndex, int batch, int kvHeads,
                                                 int headDim, int bytes) {
    std::ostringstream os;
    os << reinterpret_cast<uintptr_t>(meta) << ":" << layerIndex << ":" << batch << ":" << kvHeads << ":"
       << headDim << ":" << bytes;
    return os.str();
}

static void _registerExternalLayerMappedTarget(const PagedKVMeta* meta, int layerIndex, int batch, int kvHeads,
                                               int headDim, int bytes, int maxSlots,
                                               const std::shared_ptr<Tensor>& key,
                                               const std::shared_ptr<Tensor>& value,
                                               cl::CommandQueue& queue) {
    if (!_shouldUseDirectPagedCacheOpenCL() || meta == nullptr || layerIndex < 0 || key == nullptr ||
        value == nullptr || key->deviceId() == 0 || value->deviceId() == 0) {
        return;
    }
    ExternalLayerMappedTargetRef ref;
    ref.key = key;
    ref.value = value;
    ref.queue.reset(new cl::CommandQueue(queue));
    ref.batch = batch;
    ref.kvHeads = kvHeads;
    ref.headDim = headDim;
    ref.bytes = bytes;
    ref.maxSlots = maxSlots;
    std::lock_guard<std::mutex> lock(gExternalLayerMappedTargetMutex);
    gExternalLayerMappedTargets[_externalLayerMappedTargetKey(meta, layerIndex, batch, kvHeads, headDim, bytes)] =
        std::move(ref);
}

static ExternalLayerMappedTarget _lookupExternalLayerMappedTarget(const PagedKVMeta* meta, int layerIndex, int batch,
                                                                  int kvHeads, int headDim, int bytes) {
    ExternalLayerMappedTarget target;
    if (!_shouldUseDirectPagedCacheOpenCL()) {
        return target;
    }
    std::lock_guard<std::mutex> lock(gExternalLayerMappedTargetMutex);
    auto iter = gExternalLayerMappedTargets.find(
        _externalLayerMappedTargetKey(meta, layerIndex, batch, kvHeads, headDim, bytes));
    if (iter == gExternalLayerMappedTargets.end()) {
        return target;
    }
    target.key = iter->second.key.lock();
    target.value = iter->second.value.lock();
    target.queue = iter->second.queue;
    target.batch = iter->second.batch;
    target.kvHeads = iter->second.kvHeads;
    target.headDim = iter->second.headDim;
    target.bytes = iter->second.bytes;
    target.maxSlots = iter->second.maxSlots;
    if (target.key == nullptr || target.value == nullptr || target.queue == nullptr) {
        gExternalLayerMappedTargets.erase(iter);
        target = ExternalLayerMappedTarget();
    }
    return target;
}

static int _lastExternalLayerIndex(const PagedKVMeta* meta) {
    if (meta == nullptr || meta->external_segments.empty()) {
        return -1;
    }
    int last = meta->layer_nums > 0 ? meta->layer_nums - 1 : -1;
    for (const auto& segment : meta->external_segments) {
        for (const auto& layer : segment.layers) {
            last = std::max(last, layer.layerIndex);
        }
    }
    return last;
}

static int _externalLayerRequiredSlots(const PagedKVMeta* meta, int kvLen) {
    const size_t sourceSlots = _picCacheSourceSlotCount(meta);
    if (sourceSlots == 0) {
        return kvLen;
    }
    const int sourceBase = _picCacheSourceSlotBase(meta, kvLen);
    if (sourceSlots > static_cast<size_t>(std::numeric_limits<int>::max() - sourceBase)) {
        return std::numeric_limits<int>::max();
    }
    return std::max(kvLen, sourceBase + static_cast<int>(sourceSlots));
}

static bool _externalLayerTargetReady(const ExternalLayerMappedTarget& target, int batch, int kvHeads, int headDim,
                                      int bytes, int requiredSlots) {
    return target.key != nullptr && target.value != nullptr && target.queue != nullptr &&
        target.batch == batch && target.kvHeads == kvHeads && target.headDim == headDim &&
        target.bytes == bytes && target.maxSlots >= requiredSlots;
}

static int _externalLayerReadWindow() {
    const char* value = ::getenv("MNN_PAGED_ATTENTION_PREFETCH_WINDOW");
    if (value == nullptr || value[0] == '\0') {
        return -1;
    }
    return std::atoi(value);
}

static std::string _externalLayerRequestKey(const PagedKVMeta* meta, int batch, int kvHeads, int headDim, int bytes,
                                            int kvLen) {
    std::ostringstream os;
    os << reinterpret_cast<uintptr_t>(meta) << ":" << (meta != nullptr ? meta->slot_table_version : 0) << ":"
       << batch << ":" << kvHeads << ":" << headDim << ":" << bytes << ":" << kvLen;
    if (meta != nullptr) {
        for (const auto& segment : meta->external_segments) {
            os << "|seg:" << segment.cacheName << ":" << segment.logicalStart << ":" << segment.tokenCount << ":"
               << segment.sourceTokenOffset << ":" << segment.sourceTokenCount;
            for (const auto& layer : segment.layers) {
                os << "|layer:" << layer.layerIndex << ":" << layer.keyPath << ":" << layer.valuePath << ":"
                   << layer.hasSourceOverride << ":" << layer.sourceTokenOffset << ":" << layer.sourceTokenCount;
            }
        }
    }
    return os.str();
}

static std::string _externalLayerTaskKey(const std::string& requestKey, int layerIndex) {
    return requestKey + "\n" + std::to_string(layerIndex);
}

static bool _readBinaryFileRange(const std::string& path, size_t offsetBytes, void* dst, size_t expectedBytes) {
    if (dst == nullptr || expectedBytes == 0) {
        return false;
    }
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is.good()) {
        return false;
    }
    auto fileSize = is.tellg();
    if (fileSize < 0) {
        return false;
    }
    const auto fileBytes = static_cast<uint64_t>(static_cast<std::streamoff>(fileSize));
    if (offsetBytes > fileBytes || fileBytes - offsetBytes < expectedBytes) {
        return false;
    }
    is.seekg(static_cast<std::streamoff>(offsetBytes), std::ios::beg);
    char* cursor = reinterpret_cast<char*>(dst);
    size_t remaining = expectedBytes;
    while (remaining > 0) {
        const auto chunk = static_cast<std::streamsize>(
            std::min<size_t>(remaining, static_cast<size_t>(std::numeric_limits<std::streamsize>::max())));
        is.read(cursor, chunk);
        if (is.gcount() != chunk) {
            return false;
        }
        cursor += chunk;
        remaining -= static_cast<size_t>(chunk);
    }
    return true;
}

static bool _readBinaryFileRangeToMappedCLBufferOffset(const std::string& path, size_t fileOffsetBytes,
                                                       cl::Buffer& buffer, size_t bufferOffsetBytes,
                                                       size_t expectedBytes, cl::CommandQueue& queue) {
    if (expectedBytes == 0) {
        return false;
    }
    cl_int error = CL_SUCCESS;
    void* ptr = queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_WRITE, bufferOffsetBytes, expectedBytes,
                                       nullptr, nullptr, &error);
    if (ptr == nullptr || error != CL_SUCCESS) {
        return false;
    }
    bool ok = _readBinaryFileRange(path, fileOffsetBytes, ptr, expectedBytes);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    cl_int unmap = queue.enqueueUnmapMemObject(buffer, ptr);
    cl_int finish = queue.finish();
    return ok && unmap == CL_SUCCESS && finish == CL_SUCCESS;
}

static bool _readKeySegmentToPagedCacheSourceSlotsOpenCL(const std::string& path, cl::Buffer& keyCache,
                                                         cl::CommandQueue& queue, int batch, int kvHeads,
                                                         int headDim, int bytes, size_t sourceTokenOffset,
                                                         size_t tokenCount, size_t sourceSlotStart,
                                                         int maxSlots) {
    if (batch <= 0 || kvHeads <= 0 || headDim <= 0 || bytes <= 0 || tokenCount == 0 ||
        sourceSlotStart + tokenCount > static_cast<size_t>(maxSlots)) {
        return false;
    }
    const size_t tokenBytes = static_cast<size_t>(batch) * kvHeads * headDim * bytes;
    return _readBinaryFileRangeToMappedCLBufferOffset(path, sourceTokenOffset * tokenBytes, keyCache,
                                                      sourceSlotStart * tokenBytes, tokenCount * tokenBytes,
                                                      queue);
}

enum class SourceCLReadStatus {
    Success,
    MapFailed,
    ReadFailed,
};

static SourceCLReadStatus _readBinaryFileRangeToMappedCLBuffer(const std::string& path, size_t offsetBytes,
                                                               cl::Buffer& buffer, size_t expectedBytes,
                                                               cl::CommandQueue& queue) {
    if (expectedBytes == 0) {
        return SourceCLReadStatus::ReadFailed;
    }
    cl_int error = CL_SUCCESS;
    void* ptr = queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_WRITE, 0, expectedBytes, nullptr, nullptr, &error);
    if (ptr == nullptr || error != CL_SUCCESS) {
        return SourceCLReadStatus::MapFailed;
    }
    bool ok = _readBinaryFileRange(path, offsetBytes, ptr, expectedBytes);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    cl_int unmap = queue.enqueueUnmapMemObject(buffer, ptr);
    cl_int finish = queue.finish();
    if (!ok) {
        return SourceCLReadStatus::ReadFailed;
    }
    return unmap == CL_SUCCESS && finish == CL_SUCCESS ? SourceCLReadStatus::Success : SourceCLReadStatus::MapFailed;
}

static bool _readBinaryFileRangeToCLBufferFallback(const std::string& path, size_t offsetBytes, cl::Buffer& buffer,
                                                   size_t expectedBytes, cl::CommandQueue& queue) {
    std::vector<int8_t> data(expectedBytes);
    if (!_readBinaryFileRange(path, offsetBytes, data.data(), expectedBytes)) {
        return false;
    }
    return queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, expectedBytes, data.data()) == CL_SUCCESS;
}

static bool _readBinaryFileRangeToSourceCLBuffer(const std::string& path, size_t offsetBytes, cl::Buffer& buffer,
                                                 size_t expectedBytes, cl::CommandQueue& queue) {
    auto status = _readBinaryFileRangeToMappedCLBuffer(path, offsetBytes, buffer, expectedBytes, queue);
    if (status == SourceCLReadStatus::Success) {
        return true;
    }
    if (status == SourceCLReadStatus::ReadFailed) {
        return false;
    }
    static std::once_flag fallbackOnce;
    std::call_once(fallbackOnce, []() {
        MNN_PRINT("OpenCLPagedAttention: mapped source CL buffer unavailable, fallback to enqueueWriteBuffer\n");
    });
    return _readBinaryFileRangeToCLBufferFallback(path, offsetBytes, buffer, expectedBytes, queue);
}

static bool _readExternalValueSegment(const std::string& path, void* dst, size_t expectedBytes, int batch,
                                      int kvHeads, size_t sourceTokenCount, size_t sourceTokenOffset,
                                      size_t tokenCount, int headDim, int bytes) {
    if (dst == nullptr || expectedBytes == 0 || batch <= 0 || kvHeads <= 0 || headDim <= 0 || bytes <= 0) {
        return false;
    }
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is.good()) {
        return false;
    }
    auto fileSize = is.tellg();
    if (fileSize < 0) {
        return false;
    }
    const auto fileBytes = static_cast<uint64_t>(static_cast<std::streamoff>(fileSize));
    const size_t tokenBytes = static_cast<size_t>(headDim) * bytes;
    const size_t segmentBytes = tokenCount * tokenBytes;
    const size_t requiredBytes = static_cast<size_t>(batch) * kvHeads * sourceTokenCount * tokenBytes;
    if (sourceTokenOffset + tokenCount > sourceTokenCount || fileBytes < requiredBytes ||
        expectedBytes < static_cast<size_t>(batch) * kvHeads * segmentBytes) {
        return false;
    }
    char* out = reinterpret_cast<char*>(dst);
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < kvHeads; ++h) {
            const size_t src = ((static_cast<size_t>(b) * kvHeads + h) * sourceTokenCount +
                                sourceTokenOffset) * tokenBytes;
            const size_t dstOffset = (static_cast<size_t>(b) * kvHeads + h) * segmentBytes;
            is.seekg(static_cast<std::streamoff>(src), std::ios::beg);
            is.read(out + dstOffset, static_cast<std::streamsize>(segmentBytes));
            if (is.gcount() != static_cast<std::streamsize>(segmentBytes)) {
                return false;
            }
        }
    }
    return true;
}

static SourceCLReadStatus _readExternalValueSegmentToMappedCLBuffer(
    const std::string& path, cl::Buffer& buffer, size_t expectedBytes, cl::CommandQueue& queue, int batch,
    int kvHeads, size_t sourceTokenCount, size_t sourceTokenOffset, size_t tokenCount, int headDim, int bytes) {
    if (expectedBytes == 0) {
        return SourceCLReadStatus::ReadFailed;
    }
    cl_int error = CL_SUCCESS;
    void* ptr = queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_WRITE, 0, expectedBytes, nullptr, nullptr, &error);
    if (ptr == nullptr || error != CL_SUCCESS) {
        return SourceCLReadStatus::MapFailed;
    }
    bool ok = _readExternalValueSegment(path, ptr, expectedBytes, batch, kvHeads, sourceTokenCount,
                                        sourceTokenOffset, tokenCount, headDim, bytes);
    auto unmap = queue.enqueueUnmapMemObject(buffer, ptr);
    auto finish = queue.finish();
    if (!ok) {
        return SourceCLReadStatus::ReadFailed;
    }
    return unmap == CL_SUCCESS && finish == CL_SUCCESS ? SourceCLReadStatus::Success : SourceCLReadStatus::MapFailed;
}

static bool _readExternalValueSegmentToCLBufferFallback(
    const std::string& path, cl::Buffer& buffer, size_t expectedBytes, cl::CommandQueue& queue, int batch,
    int kvHeads, size_t sourceTokenCount, size_t sourceTokenOffset, size_t tokenCount, int headDim, int bytes) {
    std::vector<int8_t> data(expectedBytes);
    if (!_readExternalValueSegment(path, data.data(), expectedBytes, batch, kvHeads, sourceTokenCount,
                                   sourceTokenOffset, tokenCount, headDim, bytes)) {
        return false;
    }
    return queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, expectedBytes, data.data()) == CL_SUCCESS;
}

static bool _readExternalValueSegmentToSourceCLBuffer(
    const std::string& path, cl::Buffer& buffer, size_t expectedBytes, cl::CommandQueue& queue, int batch,
    int kvHeads, size_t sourceTokenCount, size_t sourceTokenOffset, size_t tokenCount, int headDim, int bytes) {
    auto status = _readExternalValueSegmentToMappedCLBuffer(path, buffer, expectedBytes, queue, batch, kvHeads,
                                                            sourceTokenCount, sourceTokenOffset, tokenCount, headDim,
                                                            bytes);
    if (status == SourceCLReadStatus::Success) {
        return true;
    }
    if (status == SourceCLReadStatus::ReadFailed) {
        return false;
    }
    if (!_envFlagEnabled("MNN_PAGED_ATTENTION_OPENCL_ALLOW_KV_STAGING_FALLBACK", false)) {
        static std::once_flag mapRequiredOnce;
        std::call_once(mapRequiredOnce, []() {
            MNN_PRINT("OpenCLPagedAttention: mapped PagedCache source slots are required for PIC cache hydrate; "
                      "set MNN_PAGED_ATTENTION_OPENCL_ALLOW_KV_STAGING_FALLBACK=1 only for debug fallback\n");
        });
        return false;
    }
    static std::once_flag fallbackOnce;
    std::call_once(fallbackOnce, []() {
        MNN_PRINT("OpenCLPagedAttention: debug fallback copies PIC cache source data with enqueueWriteBuffer\n");
    });
    return _readExternalValueSegmentToCLBufferFallback(path, buffer, expectedBytes, queue, batch, kvHeads,
                                                       sourceTokenCount, sourceTokenOffset, tokenCount, headDim,
                                                       bytes);
}

static bool _readExternalValueSegmentToPagedCacheOpenCL(
    const std::string& path, cl::Buffer& valueCache, cl::CommandQueue& queue, int batch, int kvHeads, int maxSlots,
    size_t logicalStart, size_t sourceTokenCount, size_t sourceTokenOffset, size_t tokenCount, int headDim,
    int bytes) {
    if (batch <= 0 || kvHeads <= 0 || maxSlots <= 0 || headDim <= 0 || bytes <= 0 || tokenCount == 0 ||
        logicalStart + tokenCount > static_cast<size_t>(maxSlots)) {
        return false;
    }
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is.good()) {
        return false;
    }
    auto fileSize = is.tellg();
    if (fileSize < 0) {
        return false;
    }
    const auto fileBytes = static_cast<uint64_t>(static_cast<std::streamoff>(fileSize));
    const size_t tokenBytes = static_cast<size_t>(headDim) * bytes;
    const size_t segmentBytes = tokenCount * tokenBytes;
    const size_t requiredBytes = static_cast<size_t>(batch) * kvHeads * sourceTokenCount * tokenBytes;
    if (sourceTokenOffset + tokenCount > sourceTokenCount || fileBytes < requiredBytes) {
        return false;
    }
    bool ok = true;
    for (int b = 0; ok && b < batch; ++b) {
        for (int h = 0; h < kvHeads; ++h) {
            const size_t src = ((static_cast<size_t>(b) * kvHeads + h) * sourceTokenCount +
                                sourceTokenOffset) * tokenBytes;
            const size_t dst = ((static_cast<size_t>(b) * kvHeads + h) * static_cast<size_t>(maxSlots) +
                                logicalStart) * tokenBytes;
            cl_int error = CL_SUCCESS;
            void* ptr = queue.enqueueMapBuffer(valueCache, CL_TRUE, CL_MAP_WRITE, dst, segmentBytes, nullptr,
                                               nullptr, &error);
            if (ptr == nullptr || error != CL_SUCCESS) {
                ok = false;
                break;
            }
            is.seekg(static_cast<std::streamoff>(src), std::ios::beg);
            is.read(reinterpret_cast<char*>(ptr), static_cast<std::streamsize>(segmentBytes));
            if (is.gcount() != static_cast<std::streamsize>(segmentBytes)) {
                ok = false;
            }
            std::atomic_thread_fence(std::memory_order_seq_cst);
            cl_int unmap = queue.enqueueUnmapMemObject(valueCache, ptr);
            if (unmap != CL_SUCCESS) {
                ok = false;
                break;
            }
        }
    }
    cl_int finish = queue.finish();
    return ok && finish == CL_SUCCESS;
}

static bool _readExternalValueSegmentToPhysicalPagedCacheOpenCL(
    const std::string& path, cl::Buffer& valueCache, cl::CommandQueue& queue, const PagedKVMeta* meta, int batch,
    int kvHeads, int maxSlots, size_t logicalStart, size_t sourceTokenCount, size_t sourceTokenOffset,
    size_t tokenCount, int headDim, int bytes) {
    if (batch <= 0 || kvHeads <= 0 || maxSlots <= 0 || headDim <= 0 || bytes <= 0 || tokenCount == 0) {
        return false;
    }
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is.good()) {
        return false;
    }
    auto fileSize = is.tellg();
    if (fileSize < 0) {
        return false;
    }
    const auto fileBytes = static_cast<uint64_t>(static_cast<std::streamoff>(fileSize));
    const size_t tokenBytes = static_cast<size_t>(headDim) * bytes;
    const size_t requiredBytes = static_cast<size_t>(batch) * kvHeads * sourceTokenCount * tokenBytes;
    if (sourceTokenOffset + tokenCount > sourceTokenCount || fileBytes < requiredBytes) {
        return false;
    }
    bool ok = true;
    for (int b = 0; ok && b < batch; ++b) {
        for (int h = 0; ok && h < kvHeads; ++h) {
            size_t local = 0;
            while (local < tokenCount) {
                const size_t logical = logicalStart + local;
                const int slot = meta != nullptr ? meta->physicalSlot(logical) : static_cast<int>(logical);
                if (slot < 0 || slot >= maxSlots) {
                    ok = false;
                    break;
                }
                size_t run = 1;
                while (local + run < tokenCount) {
                    const size_t nextLogical = logicalStart + local + run;
                    const int nextSlot = meta != nullptr ? meta->physicalSlot(nextLogical)
                                                         : static_cast<int>(nextLogical);
                    if (nextSlot != slot + static_cast<int>(run) || nextSlot < 0 || nextSlot >= maxSlots) {
                        break;
                    }
                    ++run;
                }
                const size_t bytesToRead = run * tokenBytes;
                const size_t src = ((static_cast<size_t>(b) * kvHeads + h) * sourceTokenCount +
                                    sourceTokenOffset + local) * tokenBytes;
                const size_t dst = ((static_cast<size_t>(b) * kvHeads + h) * static_cast<size_t>(maxSlots) +
                                    static_cast<size_t>(slot)) * tokenBytes;
                cl_int error = CL_SUCCESS;
                void* ptr = queue.enqueueMapBuffer(valueCache, CL_TRUE, CL_MAP_WRITE, dst, bytesToRead, nullptr,
                                                   nullptr, &error);
                if (ptr == nullptr || error != CL_SUCCESS) {
                    ok = false;
                    break;
                }
                is.seekg(static_cast<std::streamoff>(src), std::ios::beg);
                is.read(reinterpret_cast<char*>(ptr), static_cast<std::streamsize>(bytesToRead));
                if (is.gcount() != static_cast<std::streamsize>(bytesToRead)) {
                    ok = false;
                }
                std::atomic_thread_fence(std::memory_order_seq_cst);
                cl_int unmap = queue.enqueueUnmapMemObject(valueCache, ptr);
                if (unmap != CL_SUCCESS) {
                    ok = false;
                    break;
                }
                local += run;
            }
        }
    }
    cl_int finish = queue.finish();
    return ok && finish == CL_SUCCESS;
}

static bool _readExternalLayerSegmentOpenCL(const PagedKVMeta* meta, const PagedKVExternalSegment& segment,
                                            int layerIndex, int batch, int kvHeads, int headDim, int bytes, int kvLen,
                                            int sourceSlotStart,
                                            const ExternalLayerMappedTarget* target,
                                            ExternalLayerReadSegment& out) {
    out = ExternalLayerReadSegment();
    if (segment.tokenCount == 0) {
        out.ok = true;
        return true;
    }
    auto layer = segment.layer(layerIndex);
    if (layer == nullptr) {
        out.error = "missing persistent PIC cache layer " + std::to_string(layerIndex) + " for cache " +
            segment.cacheName;
        return false;
    }
    const int segBatch = segment.batch > 0 ? segment.batch : batch;
    const int segKvHeads = segment.kvHeads > 0 ? segment.kvHeads : kvHeads;
    const int segHeadDim = segment.headDim > 0 ? segment.headDim : headDim;
    const int segBytes = segment.dtypeBytes > 0 ? segment.dtypeBytes : bytes;
    if (segBatch != batch || segKvHeads != kvHeads || segHeadDim != headDim || segBytes != bytes) {
        out.error = "persistent PIC cache shape mismatch at layer " + std::to_string(layerIndex);
        return false;
    }
    if (segment.keyRopeState != "canonical_no_rope" || segment.ropePairing != "half") {
        out.error = "persistent PIC cache key must be canonical_no_rope/half";
        return false;
    }
    if (segment.logicalStart + segment.tokenCount > static_cast<size_t>(kvLen)) {
        out.error = "persistent PIC cache range exceeds visible KV length at layer " + std::to_string(layerIndex);
        return false;
    }
    const size_t sourceTokenOffset = layer->hasSourceOverride ? layer->sourceTokenOffset : segment.sourceTokenOffset;
    const size_t sourceTokenCount = layer->hasSourceOverride && layer->sourceTokenCount > 0
        ? layer->sourceTokenCount
        : (segment.sourceTokenCount > 0 ? segment.sourceTokenCount : (sourceTokenOffset + segment.tokenCount));
    if (sourceTokenOffset + segment.tokenCount > sourceTokenCount) {
        out.error = "persistent PIC cache source range is invalid at layer " + std::to_string(layerIndex);
        return false;
    }
    const size_t maxInt = static_cast<size_t>(std::numeric_limits<int>::max());
    if (segment.logicalStart > maxInt || segment.tokenCount > maxInt || sourceTokenOffset > maxInt ||
        sourceTokenCount > maxInt) {
        out.error = "persistent PIC cache indices exceed int range at layer " + std::to_string(layerIndex);
        return false;
    }
    const size_t keyTokenBytes = static_cast<size_t>(batch) * kvHeads * headDim * bytes;
    const size_t keySegmentBytes = segment.tokenCount * keyTokenBytes;
    const bool requireDirect = _requireDirectPagedCacheOpenCL();
    const bool targetOk = target != nullptr && target->key != nullptr && target->value != nullptr &&
        target->queue != nullptr && target->batch == batch && target->kvHeads == kvHeads &&
        target->headDim == headDim && target->bytes == bytes && target->maxSlots >= kvLen &&
        sourceSlotStart >= 0 && static_cast<size_t>(sourceSlotStart) + segment.tokenCount <=
            static_cast<size_t>(target->maxSlots);
    if (requireDirect && !targetOk) {
        out.error = "OpenCL direct PagedCache target unavailable at layer " + std::to_string(layerIndex);
        return false;
    }
    if (targetOk) {
        auto& keyBuffer = openCLBuffer(target->key.get());
        auto& valueBuffer = openCLBuffer(target->value.get());
        const bool directKeyOk = _readKeySegmentToPagedCacheSourceSlotsOpenCL(
            layer->keyPath, keyBuffer, *target->queue, batch, kvHeads, headDim, bytes, sourceTokenOffset,
            segment.tokenCount, static_cast<size_t>(sourceSlotStart), target->maxSlots);
        const bool directValueOk = directKeyOk &&
            _readExternalValueSegmentToPhysicalPagedCacheOpenCL(
                layer->valuePath, valueBuffer, *target->queue, meta, batch, kvHeads, target->maxSlots,
                segment.logicalStart, sourceTokenCount, sourceTokenOffset, segment.tokenCount, headDim, bytes);
        if (directKeyOk && directValueOk) {
            out.directWritten = true;
            out.sourceSlotStart = sourceSlotStart;
            out.ok = true;
            return true;
        }
        if (requireDirect) {
            out.error = "OpenCL direct map read into PagedCache failed at layer " + std::to_string(layerIndex);
            return false;
        }
    }
    out.keyData.resize(keySegmentBytes);
    if (!_readBinaryFileRange(layer->keyPath, sourceTokenOffset * keyTokenBytes, out.keyData.data(),
                              keySegmentBytes)) {
        out.error = "failed to read persistent PIC cache key range at layer " + std::to_string(layerIndex);
        return false;
    }
    const size_t valueTokenBytes = static_cast<size_t>(headDim) * bytes;
    const size_t valueSegmentBytes = static_cast<size_t>(batch) * kvHeads * segment.tokenCount * valueTokenBytes;
    out.valueData.resize(valueSegmentBytes);
    if (!_readExternalValueSegment(layer->valuePath, out.valueData.data(), valueSegmentBytes, batch, kvHeads,
                                   sourceTokenCount, sourceTokenOffset, segment.tokenCount, headDim, bytes)) {
        out.error = "failed to read persistent PIC cache value range at layer " + std::to_string(layerIndex);
        return false;
    }
    out.ok = true;
    return true;
}

static std::shared_ptr<ExternalLayerReadResult> _readExternalLayerOpenCL(
    const PagedKVMeta* meta, std::string requestKey, std::vector<PagedKVExternalSegment> segments, int layerIndex,
    int batch, int kvHeads, int headDim, int bytes, int kvLen, ExternalLayerMappedTarget target) {
    auto result = std::make_shared<ExternalLayerReadResult>();
    result->requestKey = std::move(requestKey);
    result->layerIndex = layerIndex;
    result->segments.resize(segments.size());
    const bool hasTarget = _externalLayerTargetReady(
        target, batch, kvHeads, headDim, bytes, _externalLayerRequiredSlots(meta, kvLen));
    int sourceSlotCursor = _picCacheSourceSlotBase(meta, kvLen);
    for (size_t i = 0; i < segments.size(); ++i) {
        const int sourceSlotStart = sourceSlotCursor;
        sourceSlotCursor += static_cast<int>(segments[i].tokenCount);
        if (!_readExternalLayerSegmentOpenCL(meta, segments[i], layerIndex, batch, kvHeads, headDim, bytes, kvLen,
                                            sourceSlotStart, hasTarget ? &target : nullptr,
                                            result->segments[i])) {
            result->ok = false;
            result->error = result->segments[i].error;
            break;
        }
    }
    return result;
}

static void _scheduleExternalLayerReadsFrom(const PagedKVMeta* meta, int startLayer, int batch, int kvHeads,
                                            int headDim, int bytes, int kvLen) {
    if (_legacyB863976OpenCL()) {
        return;
    }
    if (meta == nullptr || meta->external_segments.empty()) {
        return;
    }
    const int lastLayer = _lastExternalLayerIndex(meta);
    if (startLayer < 0 || lastLayer < startLayer) {
        return;
    }
    const int requiredSlots = _externalLayerRequiredSlots(meta, kvLen);
    const int window = _externalLayerReadWindow();
    if (window == 0) {
        return;
    }
    const int endLayer = window > 0 ? std::min(lastLayer, startLayer + window - 1) : lastLayer;
    const std::string requestKey = _externalLayerRequestKey(meta, batch, kvHeads, headDim, bytes, kvLen);
    const std::string requestPrefix = requestKey + "\n";
    std::vector<std::pair<int, ExternalLayerMappedTarget>> layersToSchedule;
    {
        std::lock_guard<std::mutex> lock(gExternalLayerReadMutex);
        for (auto it = gExternalLayerReadTasks.begin(); it != gExternalLayerReadTasks.end();) {
            if (it->first.rfind(requestPrefix, 0) != 0) {
                it = gExternalLayerReadTasks.erase(it);
            } else {
                ++it;
            }
        }
        for (int layerIndex = startLayer; layerIndex <= endLayer; ++layerIndex) {
            if (meta->externalLayerLoaded(layerIndex)) {
                continue;
            }
            const auto key = _externalLayerTaskKey(requestKey, layerIndex);
            if (gExternalLayerReadTasks.find(key) == gExternalLayerReadTasks.end()) {
                auto target = _lookupExternalLayerMappedTarget(meta, layerIndex, batch, kvHeads, headDim, bytes);
                if (!_externalLayerTargetReady(target, batch, kvHeads, headDim, bytes, requiredSlots)) {
                    break;
                }
                layersToSchedule.emplace_back(layerIndex, std::move(target));
            }
        }
        auto segments = meta->external_segments;
        for (const auto& item : layersToSchedule) {
            const int layerIndex = item.first;
            auto target = item.second;
            const auto key = _externalLayerTaskKey(requestKey, layerIndex);
            auto future = std::async(std::launch::async,
                                     [meta, requestKey, segments, layerIndex, batch, kvHeads, headDim, bytes, kvLen,
                                      target]() {
                                         return _readExternalLayerOpenCL(meta, requestKey, segments, layerIndex,
                                                                         batch, kvHeads, headDim, bytes, kvLen,
                                                                         target);
                                     }).share();
            ExternalLayerReadTask task;
            task.requestKey = requestKey;
            task.layerIndex = layerIndex;
            task.future = std::move(future);
            gExternalLayerReadTasks[key] = std::move(task);
        }
    }
}

static std::shared_ptr<ExternalLayerReadResult> _takeExternalLayerRead(const PagedKVMeta* meta, int layerIndex,
                                                                       int batch, int kvHeads, int headDim, int bytes,
                                                                       int kvLen) {
    if (meta == nullptr || meta->external_segments.empty()) {
        return nullptr;
    }
    const auto requestKey = _externalLayerRequestKey(meta, batch, kvHeads, headDim, bytes, kvLen);
    const auto taskKey = _externalLayerTaskKey(requestKey, layerIndex);
    std::shared_future<std::shared_ptr<ExternalLayerReadResult>> future;
    {
        std::lock_guard<std::mutex> lock(gExternalLayerReadMutex);
        auto iter = gExternalLayerReadTasks.find(taskKey);
        if (iter == gExternalLayerReadTasks.end()) {
            return nullptr;
        }
        future = iter->second.future;
        gExternalLayerReadTasks.erase(iter);
    }
    if (!future.valid()) {
        return nullptr;
    }
    return future.get();
}

static int _ropeDimForExport(const PagedKVMeta* meta, int headDim) {
    if (meta == nullptr || meta->rope_dim <= 0) {
        return headDim;
    }
    return std::min(headDim, meta->rope_dim);
}

static bool _writeShapeFile(const std::string& path, int batch, int kvHeads, int headDim, int tokenCount, int bytes,
                            const PagedKVMeta* meta) {
    std::ofstream os(path, std::ios::binary);
    if (!os.good()) {
        return false;
    }
    int ropeDim = _ropeDimForExport(meta, headDim);
    os << "{\n"
       << "  \"format\": \"mnn-paged-attention-kv-shape-v1\",\n"
       << "  \"batch\": " << batch << ",\n"
       << "  \"kv_heads\": " << kvHeads << ",\n"
       << "  \"head_dim\": " << headDim << ",\n"
       << "  \"token_count\": " << tokenCount << ",\n"
       << "  \"dtype_bytes\": " << bytes << ",\n"
       << "  \"key_layout\": \"[token,batch,kv_head,head_dim]\",\n"
       << "  \"value_layout\": \"[batch,kv_head,token,head_dim]\",\n"
       << "  \"key_rope_state\": \"canonical_no_rope\",\n"
       << "  \"rope_pairing\": \"half\",\n"
       << "  \"rope_theta\": " << ((meta != nullptr && meta->rope_theta > 0.0f) ? meta->rope_theta : 10000.0f) << ",\n"
       << "  \"rope_dim\": " << ropeDim << ",\n"
       << "  \"rope_type\": \"" << ((meta != nullptr && !meta->rope_type.empty()) ? meta->rope_type : "default") << "\",\n"
       << "  \"rope_scaling_factor\": " << (meta != nullptr ? meta->rope_scaling_factor : 1.0f) << ",\n"
       << "  \"rope_scaling_low_freq_factor\": " << (meta != nullptr ? meta->rope_scaling_low_freq_factor : 1.0f) << ",\n"
       << "  \"rope_scaling_high_freq_factor\": " << (meta != nullptr ? meta->rope_scaling_high_freq_factor : 4.0f) << ",\n"
       << "  \"rope_scaling_original_max_position_embeddings\": "
       << (meta != nullptr ? meta->rope_scaling_original_max_position_embeddings : 0) << ",\n"
       << "  \"max_position_embeddings\": " << (meta != nullptr ? meta->max_position_embeddings : 0) << ",\n"
       << "  \"rope_attention_scaling\": " << (meta != nullptr ? meta->rope_attention_scaling : 1.0f) << "\n"
       << "}\n";
    return os.good();
}

} // namespace

PagedAttentionBufExecution::PagedAttentionBufExecution(const MNN::Op* op, Backend* backend)
    : CommonExecution(backend, op), mOpenCLBackend(static_cast<OpenCLBackend*>(backend)) {
    if (op != nullptr && op->type() == OpType_PicScoreAttention) {
        mPicAttentionMode = 1;
    } else if (op != nullptr && op->type() == OpType_PicSparseAttention) {
        mPicAttentionMode = 2;
    }
    auto param = op->main_as_AttentionParam();
    if (param != nullptr) {
        mLayerIndex = param->layer_index();
        mKVSharedLayerIndex = param->kv_shared_layer_index();
        mIsKVShared = mKVSharedLayerIndex >= 0;
    }
    mMeta = static_cast<PagedKVMeta*>(backend->getMetaPtr());
    mCache.reset(new SharedPagedCache);
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    mCopyKernel = runtime->buildKernel("paged_attention_buf", "copy_paged_kv", {}, mOpenCLBackend->getPrecision());
    mAttentionKernel = runtime->buildKernel("paged_attention_buf", "paged_attention", {}, mOpenCLBackend->getPrecision());
    mAttentionRowKernel = runtime->buildKernel("paged_attention_buf", "paged_attention_row", {},
                                               mOpenCLBackend->getPrecision());
    mPackPagedKVKernel = runtime->buildKernel("paged_attention_buf", "pack_paged_kv_prefill", {},
                                              mOpenCLBackend->getPrecision());
    mPackPagedKeyKernel = runtime->buildKernel("paged_attention_buf", "pack_paged_k_prefill", {},
                                               mOpenCLBackend->getPrecision());
    mHydrateExternalKernel = runtime->buildKernel("paged_attention_buf", "pic_page_attention_hydrate_kv", {},
                                                  mOpenCLBackend->getPrecision());
    mExportCanonicalKeyKernel = runtime->buildKernel("paged_attention_buf", "export_canonical_paged_key", {},
                                                     mOpenCLBackend->getPrecision());
    mCacheBlendScoreKernel = runtime->buildKernel("paged_attention_buf", "pic_cacheblend_value_score", {},
                                                  mOpenCLBackend->getPrecision());
    mCacheBlendTopKKernel = runtime->buildKernel("paged_attention_buf", "pic_cacheblend_topk", {},
                                                 mOpenCLBackend->getPrecision());
    mRearrangeQKernel = runtime->buildKernel("attention_buf", "rearrange_q", {}, mOpenCLBackend->getPrecision());
    mRearrangeMaskKernel = runtime->buildKernel("attention_buf", "rearrange_mask_shortprefill", {"-DADD_MASK"},
                                                mOpenCLBackend->getPrecision());
    mSoftmaxKernel = runtime->buildKernel("softmax_buf", "softmax_v4_buf", {"-DSOFTMAX_LOCAL_SIZE=64"},
                                          mOpenCLBackend->getPrecision());
    mZeroKernel = runtime->buildKernel("paged_attention_buf", "zero_output", {}, mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL_CTOR(mCopyKernel);
    OPENCL_CHECK_KERNEL_CTOR(mAttentionKernel);
    OPENCL_CHECK_KERNEL_CTOR(mAttentionRowKernel);
    OPENCL_CHECK_KERNEL_CTOR(mPackPagedKVKernel);
    OPENCL_CHECK_KERNEL_CTOR(mPackPagedKeyKernel);
    OPENCL_CHECK_KERNEL_CTOR(mHydrateExternalKernel);
    OPENCL_CHECK_KERNEL_CTOR(mExportCanonicalKeyKernel);
    OPENCL_CHECK_KERNEL_CTOR(mCacheBlendScoreKernel);
    OPENCL_CHECK_KERNEL_CTOR(mCacheBlendTopKKernel);
    OPENCL_CHECK_KERNEL_CTOR(mRearrangeQKernel);
    OPENCL_CHECK_KERNEL_CTOR(mRearrangeMaskKernel);
    OPENCL_CHECK_KERNEL_CTOR(mSoftmaxKernel);
    OPENCL_CHECK_KERNEL_CTOR(mZeroKernel);
}

ErrorCode PagedAttentionBufExecution::ensureCache(int maxSlots, int batch, int kvHeads, int headDim) {
    if (maxSlots <= 0 || batch <= 0 || kvHeads <= 0 || headDim <= 0) {
        return INVALID_VALUE;
    }
    if (mCache && mCache->key && mCache->value && mCache->slotTable && mCache->sparseQuery &&
        mCache->maxSlots == maxSlots && mCache->batch == batch && mCache->kvHeads == kvHeads &&
        mCache->headDim == headDim && mCache->bytes == mBytes) {
        if (!_legacyB863976OpenCL() && mLayerIndex >= 0) {
            _registerExternalLayerMappedTarget(mMeta, mLayerIndex, batch, kvHeads, headDim, mBytes, maxSlots,
                                               mCache->key, mCache->value,
                                               mOpenCLBackend->getOpenCLRuntime()->commandQueue());
        }
        return NO_ERROR;
    }
    if (!mCache) {
        mCache.reset(new SharedPagedCache);
    }
    if (mBytes == 4) {
        mCache->key.reset(Tensor::createDevice<float>({maxSlots, batch, kvHeads, headDim}));
        mCache->value.reset(Tensor::createDevice<float>({batch, kvHeads, maxSlots, headDim}));
    } else {
        mCache->key.reset(Tensor::createDevice<uint16_t>({maxSlots, batch, kvHeads, headDim}));
        mCache->value.reset(Tensor::createDevice<uint16_t>({batch, kvHeads, maxSlots, headDim}));
    }
    mCache->slotTable.reset(Tensor::createDevice<int>({maxSlots}));
    mCache->sparseQuery.reset(Tensor::createDevice<int>({maxSlots}));
    if (!mCache->key || !mCache->value || !mCache->slotTable || !mCache->sparseQuery) {
        return OUT_OF_MEMORY;
    }
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mCache->key.get(), Backend::STATIC));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mCache->value.get(), Backend::STATIC));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mCache->slotTable.get(), Backend::STATIC));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mCache->sparseQuery.get(), Backend::STATIC));
    const size_t keyElements = static_cast<size_t>(maxSlots) * batch * kvHeads * headDim;
    const size_t valueElements = static_cast<size_t>(batch) * kvHeads * maxSlots * headDim;
    if (keyElements > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        valueElements > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return OUT_OF_MEMORY;
    }
    auto& queue = mOpenCLBackend->getOpenCLRuntime()->commandQueue();
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mZeroKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= mZeroKernel->get().setArg(idx++, static_cast<int>(keyElements));
    MNN_CHECK_CL_SUCCESS(ret, "setArg zero_paged_key_cache");
    queue.enqueueNDRangeKernel(mZeroKernel->get(), cl::NullRange, cl::NDRange(keyElements), cl::NullRange);
    idx = 0;
    ret = CL_SUCCESS;
    ret |= mZeroKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= mZeroKernel->get().setArg(idx++, static_cast<int>(valueElements));
    MNN_CHECK_CL_SUCCESS(ret, "setArg zero_paged_value_cache");
    queue.enqueueNDRangeKernel(mZeroKernel->get(), cl::NullRange, cl::NDRange(valueElements), cl::NullRange);
    if (_shouldUseDirectPagedCacheOpenCL()) {
        queue.finish();
    }
    mCache->maxSlots = maxSlots;
    mCache->batch = batch;
    mCache->kvHeads = kvHeads;
    mCache->headDim = headDim;
    mCache->bytes = mBytes;
    mCache->slotTableVersion = -1;
    mCache->slotTableLength = 0;
    if (!_legacyB863976OpenCL() && mLayerIndex >= 0) {
        _registerExternalLayerMappedTarget(mMeta, mLayerIndex, batch, kvHeads, headDim, mBytes, maxSlots,
                                           mCache->key, mCache->value,
                                           mOpenCLBackend->getOpenCLRuntime()->commandQueue());
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::syncSlotTable(int requiredSlots) {
    if (!mCache || !mCache->slotTable || requiredSlots > mCache->maxSlots) {
        return OUT_OF_MEMORY;
    }
    std::vector<int> identity;
    const int* hostPtr = nullptr;
    int version = 0;
    if (mMeta != nullptr) {
        if (!mMeta->ensureLogicalCapacity(requiredSlots)) {
            return OUT_OF_MEMORY;
        }
        hostPtr = mMeta->slot_table_host.data();
        version = mMeta->slot_table_version;
    } else {
        identity.resize(requiredSlots);
        for (int i = 0; i < requiredSlots; ++i) {
            identity[i] = i;
        }
        hostPtr = identity.data();
    }
    if (mCache->slotTableVersion == version && mCache->slotTableLength >= requiredSlots && mMeta != nullptr) {
        return NO_ERROR;
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueWriteBuffer(
        openCLBuffer(mCache->slotTable.get()), CL_TRUE, 0, requiredSlots * sizeof(int), hostPtr);
    mCache->slotTableVersion = version;
    mCache->slotTableLength = requiredSlots;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::syncSparseQuery(int attnLen) {
    if (attnLen <= 0 || mMeta == nullptr || !mMeta->sparse_query_active) {
        return NO_ERROR;
    }
    if (static_cast<int>(mMeta->sparse_query_logical_indices.size()) < attnLen) {
        return INVALID_VALUE;
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueWriteBuffer(
        openCLBuffer(mCache->sparseQuery.get()), CL_TRUE, 0, attnLen * sizeof(int),
        mMeta->sparse_query_logical_indices.data());
    return NO_ERROR;
}

static ErrorCode _emitActiveIndicesOpenCL(PagedKVMeta* meta, int layerIndex, int kvLen, Tensor* output,
                                          OpenCLBackend* backend) {
    if (output == nullptr) {
        return NO_ERROR;
    }
    const int budget = static_cast<int>(output->elementSize());
    std::vector<int> active;
    if (meta != nullptr) {
        active = meta->buildBudgetActiveLogicalIndices(kvLen, budget);
    }
    if (active.empty() && budget > 0) {
        const int count = std::min(kvLen, budget);
        active.reserve(count);
        for (int i = 0; i < count; ++i) {
            active.emplace_back(i);
        }
    }
    if (static_cast<int>(active.size()) != budget) {
        MNN_ERROR("OpenCLPagedAttention layer %d active index count mismatch, budget=%d active=%d\n",
                  layerIndex, budget, static_cast<int>(active.size()));
        return INVALID_VALUE;
    }
    if (meta != nullptr && !active.empty() && !meta->activatePicRows(active, layerIndex, kvLen)) {
        return INVALID_VALUE;
    }
    if (!active.empty()) {
        backend->getOpenCLRuntime()->commandQueue().enqueueWriteBuffer(
            openCLBuffer(output), CL_TRUE, 0, active.size() * sizeof(int), active.data());
    }
    return NO_ERROR;
}

static ErrorCode _emitIdentityIndicesOpenCL(int count, Tensor* output, OpenCLBackend* backend) {
    if (output == nullptr) {
        return NO_ERROR;
    }
    count = std::min(count, static_cast<int>(output->elementSize()));
    if (count <= 0) {
        return NO_ERROR;
    }
    std::vector<int> indices(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        indices[static_cast<size_t>(i)] = i;
    }
    auto ret = backend->getOpenCLRuntime()->commandQueue().enqueueWriteBuffer(
        openCLBuffer(output), CL_TRUE, 0, indices.size() * sizeof(int), indices.data());
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureFastPrefillTemps(int seqLen, int kvLen, int qChunkLen,
                                                             bool staticWorkspace) {
    if (seqLen <= 0 || kvLen <= 0 || qChunkLen <= 0) {
        return INVALID_VALUE;
    }
    qChunkLen = staticWorkspace ? seqLen : std::min(qChunkLen, seqLen);
    if (!(mTempQ && mTempK && mTempV && mTempMask && mTempQK && mTempSoftmax &&
          mFastSeqLen == seqLen && mFastKvLen == kvLen && mFastQChunkLen == qChunkLen &&
          mFastStaticWorkspace == staticWorkspace)) {
        const int seqPack = ROUND_UP(seqLen, 4);
        const int qChunkPack = ROUND_UP(qChunkLen, 4);
        const int kvPack = ROUND_UP(kvLen, 4);
        const int headPack4 = ROUND_UP(mHeadDim, 4);
        const int headPack8 = ROUND_UP(mHeadDim, 8);
        mTempQ.reset(Tensor::createDevice<float>({seqPack * headPack4 * mNumHead * mBatch}));
        mTempK.reset(Tensor::createDevice<float>({kvPack * headPack4 * mKvNumHead * mBatch}));
        mTempV.reset(Tensor::createDevice<float>({kvPack * headPack8 * mKvNumHead * mBatch}));
        mTempMask.reset(Tensor::createDevice<float>({seqPack * kvPack * mBatch}));
        mTempQK.reset(Tensor::createDevice<float>({qChunkPack * kvLen * mNumHead * mBatch}));
        mTempSoftmax.reset(Tensor::createDevice<float>({qChunkPack * kvLen * mNumHead * mBatch}));
        if (!mTempQ || !mTempK || !mTempV || !mTempMask || !mTempQK || !mTempSoftmax) {
            return OUT_OF_MEMORY;
        }
        mFastSeqLen = seqLen;
        mFastKvLen = kvLen;
        mFastQChunkLen = qChunkLen;
        mFastStaticWorkspace = staticWorkspace;
        if (staticWorkspace) {
            OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempQ.get(), Backend::STATIC));
            OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempK.get(), Backend::STATIC));
            OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempV.get(), Backend::STATIC));
            OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempMask.get(), Backend::STATIC));
            OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempQK.get(), Backend::STATIC));
            OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempSoftmax.get(), Backend::STATIC));
        }
    }
    if (staticWorkspace) {
        return NO_ERROR;
    }
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempQ.get(), Backend::DYNAMIC_IN_EXECUTION));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempK.get(), Backend::DYNAMIC_IN_EXECUTION));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempV.get(), Backend::DYNAMIC_IN_EXECUTION));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempMask.get(), Backend::DYNAMIC_IN_EXECUTION));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempQK.get(), Backend::DYNAMIC_IN_EXECUTION));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempSoftmax.get(), Backend::DYNAMIC_IN_EXECUTION));
    mOpenCLBackend->onReleaseBuffer(mTempQ.get(), Backend::DYNAMIC_IN_EXECUTION);
    mOpenCLBackend->onReleaseBuffer(mTempK.get(), Backend::DYNAMIC_IN_EXECUTION);
    mOpenCLBackend->onReleaseBuffer(mTempV.get(), Backend::DYNAMIC_IN_EXECUTION);
    mOpenCLBackend->onReleaseBuffer(mTempMask.get(), Backend::DYNAMIC_IN_EXECUTION);
    mOpenCLBackend->onReleaseBuffer(mTempQK.get(), Backend::DYNAMIC_IN_EXECUTION);
    mOpenCLBackend->onReleaseBuffer(mTempSoftmax.get(), Backend::DYNAMIC_IN_EXECUTION);
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureSparseFlashKernel() {
    if (mHeadDim != 64 || mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    const int groupSize = mNumHead / mKvNumHead;
    if (mSparseFlashKernel32 && mSparseFlashKernel64 && mSparseFlashKernelGroupSize == groupSize) {
        return NO_ERROR;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    mSparseFlashKernel32 = runtime->buildKernel("attention_buf", "sparse_flash_attention_row32",
                                                {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                                                mOpenCLBackend->getPrecision());
    mSparseFlashKernel64 = runtime->buildKernel("attention_buf", "sparse_flash_attention_row64",
                                                {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                                                mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL(mSparseFlashKernel32);
    OPENCL_CHECK_KERNEL(mSparseFlashKernel64);
    mSparseFlashKernelGroupSize = groupSize;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureExternalTemps(size_t keyElements, size_t valueElements) {
    if (keyElements == 0 || valueElements == 0) {
        return INVALID_VALUE;
    }
    if (keyElements > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        valueElements > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return OUT_OF_MEMORY;
    }
    if (mExternalKey && mExternalValue && mExternalKeyElements >= keyElements &&
        mExternalValueElements >= valueElements) {
        return NO_ERROR;
    }
    if (mBytes == 2) {
        mExternalKey.reset(Tensor::createDevice<uint16_t>({static_cast<int>(keyElements)}));
        mExternalValue.reset(Tensor::createDevice<uint16_t>({static_cast<int>(valueElements)}));
    } else {
        mExternalKey.reset(Tensor::createDevice<float>({static_cast<int>(keyElements)}));
        mExternalValue.reset(Tensor::createDevice<float>({static_cast<int>(valueElements)}));
    }
    if (!mExternalKey || !mExternalValue) {
        return OUT_OF_MEMORY;
    }
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mExternalKey.get(), Backend::STATIC));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mExternalValue.get(), Backend::STATIC));
    mExternalKeyElements = keyElements;
    mExternalValueElements = valueElements;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureCacheBlendScoreTemps(int scoreCount, int indexCount) {
    if (scoreCount < 0 || indexCount < 0) {
        return INVALID_VALUE;
    }
    if (scoreCount == 0 && indexCount == 0) {
        return NO_ERROR;
    }
    if (mCacheBlendScores && mCacheBlendIndices && mCacheBlendScoreCount >= scoreCount &&
        mCacheBlendIndexCount >= indexCount) {
        return NO_ERROR;
    }
    if (scoreCount > 0) {
        mCacheBlendScores.reset(Tensor::createDevice<float>({scoreCount}));
        if (!mCacheBlendScores) {
            return OUT_OF_MEMORY;
        }
        OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mCacheBlendScores.get(), Backend::STATIC));
    }
    if (indexCount > 0) {
        mCacheBlendIndices.reset(Tensor::createDevice<int>({indexCount}));
        if (!mCacheBlendIndices) {
            return OUT_OF_MEMORY;
        }
        OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mCacheBlendIndices.get(), Backend::STATIC));
    }
    mCacheBlendScoreCount = scoreCount;
    mCacheBlendIndexCount = indexCount;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::hydrateExternalSegments(int layerIndex, int kvLen) {
    if (mMeta == nullptr || mMeta->external_segments.empty() || mMeta->externalLayerLoaded(layerIndex)) {
        return NO_ERROR;
    }
    const bool profile = _profilePagedAttention();
    const uint64_t startUs = profile ? _nowUs() : 0;
    size_t totalTokens = 0;
    size_t directTokens = 0;
    size_t fallbackTokens = 0;
    int directSegments = 0;
    const bool legacy = _legacyB863976OpenCL();
    auto& queue = mOpenCLBackend->getOpenCLRuntime()->commandQueue();
    if (!legacy && mCache != nullptr && mCache->key != nullptr && mCache->value != nullptr) {
        _registerExternalLayerMappedTarget(mMeta, layerIndex, mBatch, mKvNumHead, mHeadDim, mBytes,
                                           mCache->maxSlots, mCache->key, mCache->value, queue);
    }
    auto prefetched = legacy ? nullptr
                             : _takeExternalLayerRead(mMeta, layerIndex, mBatch, mKvNumHead, mHeadDim, mBytes, kvLen);
    if (prefetched != nullptr && !prefetched->ok) {
        MNN_ERROR("OpenCLPagedAttention async persistent PIC cache read failed at layer %d: %s\n", layerIndex,
                  prefetched->error.c_str());
        return INVALID_VALUE;
    }
    if (prefetched != nullptr && prefetched->segments.size() != mMeta->external_segments.size()) {
        MNN_ERROR("OpenCLPagedAttention async persistent PIC cache read segment count mismatch at layer %d\n",
                  layerIndex);
        return INVALID_VALUE;
    }
    auto directTarget = _lookupExternalLayerMappedTarget(mMeta, layerIndex, mBatch, mKvNumHead, mHeadDim, mBytes);
    const bool hasDirectTarget = directTarget.key != nullptr && directTarget.value != nullptr &&
                                 directTarget.queue != nullptr;
    int sourceSlotCursor = _picCacheSourceSlotBase(mMeta, kvLen);
    for (size_t segmentIndex = 0; segmentIndex < mMeta->external_segments.size(); ++segmentIndex) {
        const auto& segment = mMeta->external_segments[segmentIndex];
        const int sourceSlotStart = sourceSlotCursor;
        sourceSlotCursor += static_cast<int>(segment.tokenCount);
        totalTokens += segment.tokenCount;
        if (segment.tokenCount == 0) {
            continue;
        }
        auto layer = segment.layer(layerIndex);
        if (layer == nullptr) {
            MNN_ERROR("OpenCLPagedAttention layer %d missing persistent PIC cache for cache %s\n", layerIndex,
                      segment.cacheName.c_str());
            return INVALID_VALUE;
        }
        const int segBatch = segment.batch > 0 ? segment.batch : mBatch;
        const int segKvHeads = segment.kvHeads > 0 ? segment.kvHeads : mKvNumHead;
        const int segHeadDim = segment.headDim > 0 ? segment.headDim : mHeadDim;
        const int segBytes = segment.dtypeBytes > 0 ? segment.dtypeBytes : mBytes;
        if (segBatch != mBatch || segKvHeads != mKvNumHead || segHeadDim != mHeadDim || segBytes != mBytes) {
            return INVALID_VALUE;
        }
        if (segment.keyRopeState != "canonical_no_rope" || segment.ropePairing != "half") {
            return INVALID_VALUE;
        }
        if (segment.logicalStart + segment.tokenCount > static_cast<size_t>(kvLen)) {
            return INVALID_VALUE;
        }
        const size_t sourceTokenOffset = layer->hasSourceOverride ? layer->sourceTokenOffset
                                                                  : segment.sourceTokenOffset;
        const size_t sourceTokenCount = layer->hasSourceOverride && layer->sourceTokenCount > 0
            ? layer->sourceTokenCount
            : (segment.sourceTokenCount > 0 ? segment.sourceTokenCount : (sourceTokenOffset + segment.tokenCount));
        const size_t sourceEnd = sourceTokenOffset + segment.tokenCount;
        if (sourceEnd > sourceTokenCount) {
            return INVALID_VALUE;
        }
        const size_t maxInt = static_cast<size_t>(std::numeric_limits<int>::max());
        if (segment.logicalStart > maxInt || segment.tokenCount > maxInt || sourceTokenOffset > maxInt ||
            sourceTokenCount > maxInt) {
            MNN_ERROR("OpenCLPagedAttention persistent PIC cache indices exceed int range at layer %d\n", layerIndex);
            return INVALID_VALUE;
        }
        const size_t keyTokenBytes = static_cast<size_t>(mBatch) * mKvNumHead * mHeadDim * mBytes;
        const size_t keySourceOffsetBytes = sourceTokenOffset * keyTokenBytes;
        const size_t keySegmentBytes = segment.tokenCount * keyTokenBytes;
        const size_t valueTokenBytes = static_cast<size_t>(mHeadDim) * mBytes;
        const size_t valueSegmentBytes = static_cast<size_t>(mBatch) * mKvNumHead * segment.tokenCount *
                                         valueTokenBytes;
        ExternalLayerReadSegment syncRead;
        const ExternalLayerReadSegment* loadedSegment = nullptr;
        if (prefetched != nullptr) {
            loadedSegment = &prefetched->segments[segmentIndex];
            if (!loadedSegment->ok) {
                MNN_ERROR("OpenCLPagedAttention async persistent PIC cache segment read failed at layer %d: %s\n",
                          layerIndex, loadedSegment->error.c_str());
                return INVALID_VALUE;
            }
        } else {
            if (!_readExternalLayerSegmentOpenCL(mMeta, segment, layerIndex, mBatch, mKvNumHead, mHeadDim, mBytes,
                                                kvLen, sourceSlotStart,
                                                hasDirectTarget ? &directTarget : nullptr, syncRead)) {
                MNN_ERROR("OpenCLPagedAttention failed to read persistent PIC cache files for layer %d: %s\n",
                          layerIndex, syncRead.error.c_str());
                return INVALID_VALUE;
            }
            loadedSegment = &syncRead;
        }

        cl::Buffer* sourceKeyBuffer = nullptr;
        cl::Buffer* sourceValueBuffer = nullptr;
        int keySourceLogicalStart = 0;
        int hydrateValue = 1;
        if (loadedSegment != nullptr && loadedSegment->directWritten) {
            sourceKeyBuffer = &openCLBuffer(mCache->key.get());
            sourceValueBuffer = &openCLBuffer(mCache->value.get());
            keySourceLogicalStart = loadedSegment->sourceSlotStart;
            hydrateValue = 0;
            directTokens += segment.tokenCount;
            ++directSegments;
        } else {
            fallbackTokens += segment.tokenCount;
            auto err = ensureExternalTemps(keySegmentBytes / mBytes, valueSegmentBytes / mBytes);
            if (err != NO_ERROR) {
                return err;
            }
            auto& externalKeyBuffer = openCLBuffer(mExternalKey.get());
            auto& externalValueBuffer = openCLBuffer(mExternalValue.get());
            if (loadedSegment != nullptr) {
                if (loadedSegment->keyData.size() < keySegmentBytes ||
                    loadedSegment->valueData.size() < valueSegmentBytes) {
                    return INVALID_VALUE;
                }
                if (queue.enqueueWriteBuffer(externalKeyBuffer, CL_TRUE, 0, keySegmentBytes,
                                             loadedSegment->keyData.data()) != CL_SUCCESS ||
                    queue.enqueueWriteBuffer(externalValueBuffer, CL_TRUE, 0, valueSegmentBytes,
                                             loadedSegment->valueData.data()) != CL_SUCCESS) {
                    return INVALID_VALUE;
                }
            } else {
                if (!_readBinaryFileRangeToSourceCLBuffer(layer->keyPath, keySourceOffsetBytes, externalKeyBuffer,
                                                          keySegmentBytes, queue) ||
                    !_readExternalValueSegmentToSourceCLBuffer(
                        layer->valuePath, externalValueBuffer, valueSegmentBytes, queue, mBatch, mKvNumHead,
                        sourceTokenCount, sourceTokenOffset, segment.tokenCount, mHeadDim, mBytes)) {
                    return INVALID_VALUE;
                }
            }
            sourceKeyBuffer = &externalKeyBuffer;
            sourceValueBuffer = &externalValueBuffer;
        }

        int ropeDim = segment.ropeDim > 0 ? segment.ropeDim : mHeadDim;
        ropeDim = std::min(ropeDim, mHeadDim);
        ropeDim = (ropeDim / 2) * 2;
        const int ropeTypeLlama3 = segment.ropeType == "llama3" ? 1 : 0;
        const int oldContext = segment.ropeScalingOriginalMaxPositionEmbeddings > 0
            ? segment.ropeScalingOriginalMaxPositionEmbeddings
            : segment.maxPositionEmbeddings;
        const size_t totalElements = segment.tokenCount * static_cast<size_t>(mBatch) * mKvNumHead * mHeadDim;
        if (totalElements > maxInt) {
            MNN_ERROR("OpenCLPagedAttention persistent PIC cache hydrate element count exceeds int range at layer %d\n",
                      layerIndex);
            return INVALID_VALUE;
        }
        const int total = static_cast<int>(totalElements);
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mHydrateExternalKernel->get().setArg(idx++, *sourceKeyBuffer);
        ret |= mHydrateExternalKernel->get().setArg(idx++, *sourceValueBuffer);
        ret |= mHydrateExternalKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= mHydrateExternalKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= mHydrateExternalKernel->get().setArg(idx++, openCLBuffer(mCache->slotTable.get()));
        ret |= mHydrateExternalKernel->get().setArg(idx++, mBatch);
        ret |= mHydrateExternalKernel->get().setArg(idx++, mKvNumHead);
        ret |= mHydrateExternalKernel->get().setArg(idx++, mHeadDim);
        ret |= mHydrateExternalKernel->get().setArg(idx++, mCache->maxSlots);
        ret |= mHydrateExternalKernel->get().setArg(idx++, static_cast<int>(segment.logicalStart));
        ret |= mHydrateExternalKernel->get().setArg(idx++, static_cast<int>(segment.tokenCount));
        ret |= mHydrateExternalKernel->get().setArg(idx++, keySourceLogicalStart);
        ret |= mHydrateExternalKernel->get().setArg(idx++, hydrateValue);
        ret |= mHydrateExternalKernel->get().setArg(idx++, ropeDim);
        ret |= mHydrateExternalKernel->get().setArg(idx++, segment.ropeTheta);
        ret |= mHydrateExternalKernel->get().setArg(idx++, ropeTypeLlama3);
        ret |= mHydrateExternalKernel->get().setArg(idx++, segment.ropeScalingFactor);
        ret |= mHydrateExternalKernel->get().setArg(idx++, segment.ropeScalingLowFreqFactor);
        ret |= mHydrateExternalKernel->get().setArg(idx++, segment.ropeScalingHighFreqFactor);
        ret |= mHydrateExternalKernel->get().setArg(idx++, oldContext);
        ret |= mHydrateExternalKernel->get().setArg(idx++, segment.maxPositionEmbeddings);
        ret |= mHydrateExternalKernel->get().setArg(idx++, segment.ropeAttentionScaling);
        ret |= mHydrateExternalKernel->get().setArg(idx++, total);
        MNN_CHECK_CL_SUCCESS(ret, "setArg pic_page_attention_hydrate_kv");
        queue.enqueueNDRangeKernel(mHydrateExternalKernel->get(), cl::NullRange, cl::NDRange(total), cl::NullRange);
    }
    mMeta->markExternalLayerLoaded(layerIndex);
    if (profile) {
        queue.finish();
        MNN_PRINT("OpenCLPagedAttention profile op=hydrate layer=%d tokens=%d kv_len=%d async_read=%d "
                  "direct_segments=%d direct_tokens=%d fallback_tokens=%d us=%llu\n",
                  layerIndex, static_cast<int>(totalTokens), kvLen, prefetched != nullptr ? 1 : 0,
                  directSegments, static_cast<int>(directTokens), static_cast<int>(fallbackTokens),
                  static_cast<unsigned long long>(_nowUs() - startUs));
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runCacheBlendScoring(int layerIndex, int kvLen) {
    if (mMeta == nullptr || !mMeta->needsCacheBlendScoring(layerIndex)) {
        return NO_ERROR;
    }
    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t readUs = 0;
    uint64_t scoreKernelUs = 0;
    uint64_t topKUs = 0;
    uint64_t readbackUs = 0;
    const int picTokenCount = mMeta->cacheblend_score_pic_token_count;
    const int topK = mMeta->cacheblend_score_top_k;
    if (picTokenCount < 0 || topK < 0 || topK > picTokenCount ||
        mMeta->cacheblend_score_pic_start + picTokenCount > kvLen) {
        return INVALID_VALUE;
    }
    if (topK == 0 || picTokenCount == 0) {
        mMeta->setCacheBlendScoringResult({});
        return NO_ERROR;
    }
    auto err = ensureCacheBlendScoreTemps(picTokenCount, topK);
    if (err != NO_ERROR) {
        return err;
    }
    auto& queue = mOpenCLBackend->getOpenCLRuntime()->commandQueue();
    size_t scoreOffset = 0;
    int sourceSlotCursor = _picCacheSourceSlotBase(mMeta, kvLen);
    for (const auto& segment : mMeta->cacheblend_score_segments) {
        const int sourceSlotStart = sourceSlotCursor;
        sourceSlotCursor += static_cast<int>(segment.tokenCount);
        if (segment.tokenCount == 0) {
            continue;
        }
        auto layer = segment.layer(layerIndex);
        if (layer == nullptr) {
            return INVALID_VALUE;
        }
        const int segBatch = segment.batch > 0 ? segment.batch : mBatch;
        const int segKvHeads = segment.kvHeads > 0 ? segment.kvHeads : mKvNumHead;
        const int segHeadDim = segment.headDim > 0 ? segment.headDim : mHeadDim;
        const int segBytes = segment.dtypeBytes > 0 ? segment.dtypeBytes : mBytes;
        if (segBatch != mBatch || segKvHeads != mKvNumHead || segHeadDim != mHeadDim || segBytes != mBytes) {
            return INVALID_VALUE;
        }
        const size_t sourceTokenOffset = layer->hasSourceOverride ? layer->sourceTokenOffset
                                                                  : segment.sourceTokenOffset;
        const size_t sourceTokenCount = layer->hasSourceOverride && layer->sourceTokenCount > 0
            ? layer->sourceTokenCount
            : (segment.sourceTokenCount > 0 ? segment.sourceTokenCount : (sourceTokenOffset + segment.tokenCount));
        const size_t maxInt = static_cast<size_t>(std::numeric_limits<int>::max());
        if (sourceTokenOffset + segment.tokenCount > sourceTokenCount ||
            segment.logicalStart + segment.tokenCount > static_cast<size_t>(kvLen) ||
            scoreOffset + segment.tokenCount > static_cast<size_t>(picTokenCount) ||
            segment.logicalStart > maxInt || segment.tokenCount > maxInt || scoreOffset > maxInt) {
            return INVALID_VALUE;
        }
        if (sourceSlotStart < 0 || static_cast<size_t>(sourceSlotStart) + segment.tokenCount >
            static_cast<size_t>(mCache->maxSlots)) {
            return OUT_OF_MEMORY;
        }
        auto& cacheValueBuffer = openCLBuffer(mCache->value.get());
        uint64_t opStartUs = profileDetail ? _nowUs() : 0;
        if (!_readExternalValueSegmentToPagedCacheOpenCL(
                layer->valuePath, cacheValueBuffer, queue, mBatch, mKvNumHead, mCache->maxSlots,
                static_cast<size_t>(sourceSlotStart), sourceTokenCount, sourceTokenOffset, segment.tokenCount,
                mHeadDim, mBytes)) {
            return INVALID_VALUE;
        }
        if (profileDetail) {
            queue.finish();
            readUs += _nowUs() - opStartUs;
        }
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, openCLBuffer(mCache->slotTable.get()));
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, openCLBuffer(mCacheBlendScores.get()));
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, mBatch);
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, mKvNumHead);
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, mHeadDim);
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, mCache->maxSlots);
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, mCache->maxSlots);
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, sourceSlotStart);
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, static_cast<int>(segment.logicalStart));
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, static_cast<int>(segment.tokenCount));
        ret |= mCacheBlendScoreKernel->get().setArg(idx++, static_cast<int>(scoreOffset));
        MNN_CHECK_CL_SUCCESS(ret, "setArg pic_cacheblend_value_score");
        opStartUs = profileDetail ? _nowUs() : 0;
        ret = queue.enqueueNDRangeKernel(mCacheBlendScoreKernel->get(), cl::NullRange,
                                         cl::NDRange(static_cast<int>(segment.tokenCount)), cl::NullRange);
        MNN_CHECK_CL_SUCCESS(ret, "enqueue pic_cacheblend_value_score");
        if (profileDetail) {
            queue.finish();
            scoreKernelUs += _nowUs() - opStartUs;
        }
        scoreOffset += segment.tokenCount;
    }
    if (scoreOffset != static_cast<size_t>(picTokenCount)) {
        return INVALID_VALUE;
    }
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mCacheBlendTopKKernel->get().setArg(idx++, openCLBuffer(mCacheBlendScores.get()));
    ret |= mCacheBlendTopKKernel->get().setArg(idx++, openCLBuffer(mCacheBlendIndices.get()));
    ret |= mCacheBlendTopKKernel->get().setArg(idx++, picTokenCount);
    ret |= mCacheBlendTopKKernel->get().setArg(idx++, topK);
    MNN_CHECK_CL_SUCCESS(ret, "setArg pic_cacheblend_topk");
    constexpr int topKLocalSize = 256;
    uint64_t opStartUs = profileDetail ? _nowUs() : 0;
    ret = queue.enqueueNDRangeKernel(mCacheBlendTopKKernel->get(), cl::NullRange, cl::NDRange(topKLocalSize),
                                     cl::NDRange(topKLocalSize));
    MNN_CHECK_CL_SUCCESS(ret, "enqueue pic_cacheblend_topk");
    if (profileDetail) {
        queue.finish();
        topKUs += _nowUs() - opStartUs;
    }
    std::vector<int> selected(topK);
    opStartUs = profileDetail ? _nowUs() : 0;
    if (queue.enqueueReadBuffer(openCLBuffer(mCacheBlendIndices.get()), CL_TRUE, 0,
                                static_cast<size_t>(topK) * sizeof(int), selected.data()) != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    if (profileDetail) {
        readbackUs += _nowUs() - opStartUs;
    }
    std::vector<uint8_t> seen(static_cast<size_t>(picTokenCount), 0);
    for (int index : selected) {
        if (index < 0 || index >= picTokenCount || seen[static_cast<size_t>(index)] != 0) {
            return INVALID_VALUE;
        }
        seen[static_cast<size_t>(index)] = 1;
    }
    mMeta->setCacheBlendScoringResult(selected);
    if (profile) {
        queue.finish();
        const uint64_t totalUs = _nowUs() - startUs;
        MNN_PRINT("OpenCLPagedAttention profile op=cacheblend_score layer=%d pic_tokens=%d top_k=%d us=%llu "
                  "read_us=%llu score_kernel_us=%llu topk_us=%llu readback_us=%llu detail=%d\n",
                  layerIndex, picTokenCount, topK,
                  static_cast<unsigned long long>(totalUs),
                  static_cast<unsigned long long>(readUs),
                  static_cast<unsigned long long>(scoreKernelUs),
                  static_cast<unsigned long long>(topKUs),
                  static_cast<unsigned long long>(readbackUs),
                  profileDetail ? 1 : 0);
    }
    return NO_ERROR;
}

bool PagedAttentionBufExecution::canUseFastPrefill(const Tensor* mask, int baseLogical, int attnLen, int kvLen,
                                                   bool sparseQuery, bool externalHydrated, int* maskKeyLen) const {
    if (maskKeyLen != nullptr) {
        *maskKeyLen = 0;
    }
    if (mIsKVShared || mMeta == nullptr) {
        return false;
    }
    const bool hasExternal = !mMeta->external_segments.empty();
    if (hasExternal && !externalHydrated) {
        return false;
    }
    if (!hasExternal && !sparseQuery && (baseLogical != 0 || kvLen != attnLen)) {
        return false;
    }
    if (sparseQuery) {
        return false;
    }
    if (baseLogical < 0 || attnLen != mQuerySeqLen || attnLen <= 1 || kvLen < attnLen) {
        return false;
    }
    if (mHeadDim <= 0 || mHeadDim % 8 != 0 || mNumHead % mKvNumHead != 0) {
        return false;
    }
    if (mask == nullptr || mask->elementSize() <= 1 || mask->getType().code != halide_type_float) {
        return false;
    }
    const int maskElements = static_cast<int>(mask->elementSize());
    const int64_t fullMaskElements = static_cast<int64_t>(attnLen) * kvLen;
    const int64_t shortMaskElements = static_cast<int64_t>(attnLen) * attnLen;
    if (maskElements >= fullMaskElements) {
        if (maskKeyLen != nullptr) {
            *maskKeyLen = kvLen;
        }
        return true;
    }
    if (!sparseQuery && hasExternal && maskElements >= shortMaskElements) {
        if (maskKeyLen != nullptr) {
            *maskKeyLen = attnLen;
        }
        return true;
    }
    return false;
}

bool PagedAttentionBufExecution::canUseSparseFastPrefill(const Tensor* mask, int attnLen, int kvLen,
                                                         bool externalHydrated, bool queryRowsAreFull) const {
    if (_legacyB863976OpenCL()) {
        return false;
    }
    if (mIsKVShared || mMeta == nullptr || !mMeta->sparse_query_active || !externalHydrated) {
        return false;
    }
    if (attnLen <= 1 || kvLen <= 0 || mQuerySeqLen <= 0) {
        return false;
    }
    if (!queryRowsAreFull && attnLen != mQuerySeqLen) {
        return false;
    }
    if (queryRowsAreFull && mQuerySeqLen < attnLen) {
        return false;
    }
    if (queryRowsAreFull && mPicAttentionMode != 1) {
        return false;
    }
    if (mHeadDim != 64 || mNumHead % mKvNumHead != 0) {
        return false;
    }
    if (static_cast<int>(mMeta->sparse_query_logical_indices.size()) < attnLen) {
        return false;
    }
    if (mask != nullptr && mask->elementSize() > 1 && mPicAttentionMode != 1 && mPicAttentionMode != 2) {
        return false;
    }
    return mCache && mCache->key && mCache->value && mCache->slotTable && mCache->sparseQuery;
}

ErrorCode PagedAttentionBufExecution::runSparseFastPrefill(const std::vector<Tensor*>& inputs,
                                                           const std::vector<Tensor*>& outputs, int kvLen,
                                                           int attnLen, bool queryRowsAreFull) {
    auto query = inputs[0];
    auto output = outputs[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const int activeLen = attnLen;
    if (queryRowsAreFull && mPicAttentionMode != 1) {
        return INVALID_VALUE;
    }
    const int qStorageLen = queryRowsAreFull ? mQuerySeqLen : activeLen;
    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t rearrangeUs = 0;
    uint64_t packUs = 0;
    uint64_t flashUs = 0;
    uint64_t qkRectTiles = 0;
    uint64_t qkActiveTiles = 0;
    uint64_t qkRowTiles = 0;
    int flash32Pieces = 0;
    int flash64Pieces = 0;
    bool staticWorkspace = _useStaticFullPrefill(qStorageLen, kvLen, mBatch, mNumHead, mKvNumHead, mHeadDim);
    const bool singlePieceSparseFlash = mMeta != nullptr && mMeta->cacheblend_score_ready;
    const int layerCount = mMeta != nullptr && mMeta->layer_nums > 0 ? mMeta->layer_nums : 1;
    int qChunkLen = singlePieceSparseFlash
        ? activeLen
        : (staticWorkspace ? activeLen : _prefillQChunkLen(activeLen, kvLen, mBatch, mNumHead, layerCount));
    if (!singlePieceSparseFlash) {
        constexpr int sparseQChunkLimit = 64;
        if (sparseQChunkLimit > 0 && sparseQChunkLimit < qChunkLen) {
            qChunkLen = std::max(4, std::min(activeLen, ((sparseQChunkLimit + 3) / 4) * 4));
        }
    }
    auto pieces = singlePieceSparseFlash
        ? _buildFixedSparsePieces(mMeta->sparse_query_logical_indices, activeLen, kvLen, qChunkLen)
        : _buildRangeAwareSparsePieces(mMeta->sparse_query_logical_indices, activeLen, kvLen, qChunkLen);
    auto err = ensureFastPrefillTemps(qStorageLen, kvLen, qChunkLen, staticWorkspace);
    if (err != NO_ERROR) {
        return err;
    }
    err = ensureSparseFlashKernel();
    if (err != NO_ERROR) {
        return err;
    }
    auto run3D = [&](const std::shared_ptr<KernelWrap>& kernel, std::vector<uint32_t> gws,
                    const std::string& kernelName, const std::string& programName) {
        auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(kernel));
        auto lws = localWS3DDefault(gws, maxWorkGroupSize, runtime, kernelName, kernel,
                                    mOpenCLBackend->getCLTuneLevel(), programName).first;
        gws[0] = ROUND_UP(gws[0], std::max((uint32_t)1, lws[0]));
        gws[1] = ROUND_UP(gws[1], std::max((uint32_t)1, lws[1]));
        gws[2] = ROUND_UP(gws[2], std::max((uint32_t)1, lws[2]));
        run3DKernelDefault(kernel, gws, lws, runtime);
    };

    auto tempBuffer = [&](Tensor* tensor) -> cl::Buffer& {
        return staticWorkspace ? openCLBuffer(tensor) : openCLDeferBuffer(tensor);
    };

    const int kvPack = ROUND_UP(kvLen, 4);
    const float scale = (mMeta && mMeta->attn_scale > 0)
        ? mMeta->attn_scale
        : (1.0f / std::sqrt(static_cast<float>(mHeadDim)));
    const bool directValuePrefill = _useDirectValuePrefillForSparse(mMeta, activeLen) &&
        mCache != nullptr && mCache->value != nullptr &&
        mCache->maxSlots >= kvLen && _slotTableIsIdentity(mMeta, kvLen);
    cl::Buffer& qkvValueBuffer = directValuePrefill ? openCLBuffer(mCache->value.get()) : tempBuffer(mTempV.get());
    const int qkvValueMaxLen = directValuePrefill ? mCache->maxSlots : kvPack;
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    std::vector<uint32_t> gws;

    idx = 0;
    gws = {static_cast<uint32_t>(UP_DIV(qStorageLen, 4)), static_cast<uint32_t>(UP_DIV(mHeadDim, 4)),
           static_cast<uint32_t>(mNumHead * mBatch)};
    ret |= mRearrangeQKernel->get().setArg(idx++, gws[0]);
    ret |= mRearrangeQKernel->get().setArg(idx++, gws[1]);
    ret |= mRearrangeQKernel->get().setArg(idx++, gws[2]);
    ret |= mRearrangeQKernel->get().setArg(idx++, openCLBuffer(query));
    ret |= mRearrangeQKernel->get().setArg(idx++, tempBuffer(mTempQ.get()));
    ret |= mRearrangeQKernel->get().setArg(idx++, qStorageLen);
    ret |= mRearrangeQKernel->get().setArg(idx++, mHeadDim);
    ret |= mRearrangeQKernel->get().setArg(idx++, mNumHead);
    MNN_CHECK_CL_SUCCESS(ret, "setArg paged sparse flash rearrange_q");
    uint64_t opStartUs = profileDetail ? _nowUs() : 0;
    run3D(mRearrangeQKernel, gws, "rearrange_q", "attention_buf");
    if (profileDetail) {
        runtime->commandQueue().finish();
        rearrangeUs += _nowUs() - opStartUs;
    }

    idx = 0;
    gws = {static_cast<uint32_t>(UP_DIV(kvLen, 4)), static_cast<uint32_t>(UP_DIV(mHeadDim, 4)),
           static_cast<uint32_t>(mKvNumHead * mBatch)};
    ret = CL_SUCCESS;
    opStartUs = profileDetail ? _nowUs() : 0;
    if (directValuePrefill) {
        ret |= mPackPagedKeyKernel->get().setArg(idx++, gws[0]);
        ret |= mPackPagedKeyKernel->get().setArg(idx++, gws[1]);
        ret |= mPackPagedKeyKernel->get().setArg(idx++, gws[2]);
        ret |= mPackPagedKeyKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= mPackPagedKeyKernel->get().setArg(idx++, tempBuffer(mTempK.get()));
        ret |= mPackPagedKeyKernel->get().setArg(idx++, openCLBuffer(mCache->slotTable.get()));
        ret |= mPackPagedKeyKernel->get().setArg(idx++, mBatch);
        ret |= mPackPagedKeyKernel->get().setArg(idx++, kvLen);
        ret |= mPackPagedKeyKernel->get().setArg(idx++, mKvNumHead);
        ret |= mPackPagedKeyKernel->get().setArg(idx++, mHeadDim);
        ret |= mPackPagedKeyKernel->get().setArg(idx++, mCache->maxSlots);
        MNN_CHECK_CL_SUCCESS(ret, "setArg sparse flash pack_paged_k_prefill");
        run3D(mPackPagedKeyKernel, gws, "pack_paged_k_prefill", "paged_attention_buf");
    } else {
        ret |= mPackPagedKVKernel->get().setArg(idx++, gws[0]);
        ret |= mPackPagedKVKernel->get().setArg(idx++, gws[1]);
        ret |= mPackPagedKVKernel->get().setArg(idx++, gws[2]);
        ret |= mPackPagedKVKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= mPackPagedKVKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= mPackPagedKVKernel->get().setArg(idx++, tempBuffer(mTempK.get()));
        ret |= mPackPagedKVKernel->get().setArg(idx++, tempBuffer(mTempV.get()));
        ret |= mPackPagedKVKernel->get().setArg(idx++, openCLBuffer(mCache->slotTable.get()));
        ret |= mPackPagedKVKernel->get().setArg(idx++, mBatch);
        ret |= mPackPagedKVKernel->get().setArg(idx++, kvLen);
        ret |= mPackPagedKVKernel->get().setArg(idx++, mKvNumHead);
        ret |= mPackPagedKVKernel->get().setArg(idx++, mHeadDim);
        ret |= mPackPagedKVKernel->get().setArg(idx++, mCache->maxSlots);
        MNN_CHECK_CL_SUCCESS(ret, "setArg sparse flash pack_paged_kv_prefill");
        run3D(mPackPagedKVKernel, gws, "pack_paged_kv_prefill", "paged_attention_buf");
    }
    if (profileDetail) {
        runtime->commandQueue().finish();
        packUs += _nowUs() - opStartUs;
    }

    const int qSplitNum = static_cast<int>(pieces.size());
    for (const auto& piece : pieces) {
        const int qStart = piece.qStart;
        const int qPieceLen = piece.qLen;
        if (qPieceLen <= 0) {
            continue;
        }
        const int activeKvLen = std::max(1, std::min(kvLen, piece.activeKvLen));
        if (profileDetail) {
            qkRectTiles += static_cast<uint64_t>(UP_DIV(qPieceLen, 4)) * UP_DIV(activeKvLen, 4);
            if (mMeta != nullptr &&
                static_cast<int>(mMeta->sparse_query_logical_indices.size()) >= qStart + qPieceLen) {
                for (int qLocal4 = 0; qLocal4 < qPieceLen; qLocal4 += 4) {
                    int q4MaxLogical = -1;
                    const int qGroupLen = std::min(4, qPieceLen - qLocal4);
                    for (int qi = 0; qi < qGroupLen; ++qi) {
                        const int logical = mMeta->sparse_query_logical_indices[qStart + qLocal4 + qi];
                        q4MaxLogical = std::max(q4MaxLogical, logical);
                        qkRowTiles += UP_DIV(std::max(0, std::min(kvLen, logical + 1)), 4);
                    }
                    qkActiveTiles += UP_DIV(std::max(0, std::min(kvLen, q4MaxLogical + 1)), 4);
                }
            } else {
                qkActiveTiles += static_cast<uint64_t>(UP_DIV(qPieceLen, 4)) * UP_DIV(activeKvLen, 4);
                qkRowTiles += static_cast<uint64_t>(qPieceLen) * UP_DIV(activeKvLen, 4);
            }
        }

        const uint32_t flashLanes = _sparseFlashLaneWidth(mMeta, activeLen);
        auto flashKernel = flashLanes == 32u ? mSparseFlashKernel32 : mSparseFlashKernel64;
        if (flashLanes == 32u) {
            ++flash32Pieces;
        } else {
            ++flash64Pieces;
        }

        idx = 0;
        gws = {flashLanes, static_cast<uint32_t>(qPieceLen), static_cast<uint32_t>(mNumHead * mBatch)};
        ret = CL_SUCCESS;
        ret |= flashKernel->get().setArg(idx++, gws[0]);
        ret |= flashKernel->get().setArg(idx++, gws[1]);
        ret |= flashKernel->get().setArg(idx++, gws[2]);
        ret |= flashKernel->get().setArg(idx++, tempBuffer(mTempQ.get()));
        ret |= flashKernel->get().setArg(idx++, tempBuffer(mTempK.get()));
        ret |= flashKernel->get().setArg(idx++, qkvValueBuffer);
        ret |= flashKernel->get().setArg(idx++, openCLBuffer(mCache->sparseQuery.get()));
        ret |= flashKernel->get().setArg(idx++, openCLBuffer(output));
        ret |= flashKernel->get().setArg(idx++, scale);
        ret |= flashKernel->get().setArg(idx++, qStorageLen);
        ret |= flashKernel->get().setArg(idx++, activeLen);
        ret |= flashKernel->get().setArg(idx++, queryRowsAreFull ? 1 : 0);
        ret |= flashKernel->get().setArg(idx++, qStart);
        ret |= flashKernel->get().setArg(idx++, qPieceLen);
        ret |= flashKernel->get().setArg(idx++, activeKvLen);
        ret |= flashKernel->get().setArg(idx++, kvPack);
        ret |= flashKernel->get().setArg(idx++, qkvValueMaxLen);
        ret |= flashKernel->get().setArg(idx++, mNumHead);
        ret |= flashKernel->get().setArg(idx++, mKvNumHead);
        ret |= flashKernel->get().setArg(idx++, mHeadDim);
        MNN_CHECK_CL_SUCCESS(ret, flashLanes == 32u ? "setArg paged sparse flash attention row32"
                                                    : "setArg paged sparse flash attention row64");
        opStartUs = profileDetail ? _nowUs() : 0;
        run3DKernelDefault(flashKernel, gws, {flashLanes, 1u, 1u}, runtime);
        if (profileDetail) {
            runtime->commandQueue().finish();
            flashUs += _nowUs() - opStartUs;
        }
    }
    if (profile) {
        runtime->commandQueue().finish();
        int layerIndex = mLayerIndex >= 0 ? mLayerIndex : (mMeta != nullptr ? mMeta->layer_index : -1);
        const uint64_t totalUs = _nowUs() - startUs;
        if (profileDetail) {
            MNN_PRINT("OpenCLPagedAttention profile op=%s layer=%d "
                      "query=%d input_query=%d full_q=%d kv_len=%d q_chunk=%d q_split=%d static=%d "
                      "direct_value=%d lane32_pieces=%d lane64_pieces=%d us=%llu "
                      "rearrange_us=%llu pack_us=%llu flash_us=%llu "
                      "qk_rect_tiles=%llu qk_active_tiles=%llu qk_row_tiles=%llu\n",
                      queryRowsAreFull ? "score_flash_attention" : "sparse_flash_attention",
                      layerIndex, activeLen, mQuerySeqLen, queryRowsAreFull ? 1 : 0, kvLen, qChunkLen, qSplitNum,
                      staticWorkspace ? 1 : 0, directValuePrefill ? 1 : 0, flash32Pieces, flash64Pieces,
                      static_cast<unsigned long long>(totalUs),
                      static_cast<unsigned long long>(rearrangeUs),
                      static_cast<unsigned long long>(packUs),
                      static_cast<unsigned long long>(flashUs),
                      static_cast<unsigned long long>(qkRectTiles),
                      static_cast<unsigned long long>(qkActiveTiles),
                      static_cast<unsigned long long>(qkRowTiles));
        } else {
            MNN_PRINT("OpenCLPagedAttention profile op=%s layer=%d "
                      "query=%d input_query=%d full_q=%d kv_len=%d q_chunk=%d q_split=%d static=%d "
                      "direct_value=%d lane32_pieces=%d lane64_pieces=%d us=%llu\n",
                      queryRowsAreFull ? "score_flash_attention" : "sparse_flash_attention",
                      layerIndex, activeLen, mQuerySeqLen, queryRowsAreFull ? 1 : 0, kvLen, qChunkLen, qSplitNum,
                      staticWorkspace ? 1 : 0, directValuePrefill ? 1 : 0, flash32Pieces, flash64Pieces,
                      static_cast<unsigned long long>(totalUs));
        }
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runFastPrefill(const std::vector<Tensor*>& inputs,
                                                     const std::vector<Tensor*>& outputs, int kvLen, int maskKeyLen) {
    if (maskKeyLen <= 0 || maskKeyLen > kvLen) {
        return INVALID_VALUE;
    }
    auto query = inputs[0];
    auto mask = inputs[3];
    auto output = outputs[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const bool profile = _profilePagedAttention();
    const uint64_t startUs = profile ? _nowUs() : 0;
    const int layerCount = mMeta != nullptr && mMeta->layer_nums > 0 ? mMeta->layer_nums : 1;
    const bool legacy = _legacyB863976OpenCL();
    bool staticWorkspace = legacy ||
        _useStaticFullPrefill(mQuerySeqLen, kvLen, mBatch, mNumHead, mKvNumHead, mHeadDim);
    int qChunkLen = staticWorkspace ? mQuerySeqLen
                                    : _prefillQChunkLen(mQuerySeqLen, kvLen, mBatch, mNumHead, layerCount);
    auto err = ensureFastPrefillTemps(mQuerySeqLen, kvLen, qChunkLen, staticWorkspace);
    if (err != NO_ERROR && staticWorkspace && !legacy) {
        static std::once_flag fallbackOnce;
        std::call_once(fallbackOnce, []() {
            MNN_PRINT("OpenCLPagedAttention: static fast prefill workspace unavailable, fallback to q-split\n");
        });
        mTempQ.reset();
        mTempK.reset();
        mTempV.reset();
        mTempMask.reset();
        mTempQK.reset();
        mTempSoftmax.reset();
        mFastSeqLen = 0;
        mFastKvLen = 0;
        mFastQChunkLen = 0;
        mFastStaticWorkspace = false;
        staticWorkspace = false;
        qChunkLen = _prefillQChunkLen(mQuerySeqLen, kvLen, mBatch, mNumHead, layerCount);
        err = ensureFastPrefillTemps(mQuerySeqLen, kvLen, qChunkLen, false);
    }
    if (err != NO_ERROR) {
        return err;
    }
    if (!mQKKernel || !mQKVKernel || mFastKernelStatic != staticWorkspace || mFastKernelSparse) {
        const int groupSize = mNumHead / mKvNumHead;
        mQKKernel = runtime->buildKernel("attention_buf",
                                         staticWorkspace ? "matmul_qk_div_mask_prefill"
                                                         : "matmul_qk_div_mask_prefill_piece",
                                         {"-DADD_MASK", "-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                                         mOpenCLBackend->getPrecision());
        mQKVKernel = runtime->buildKernel("attention_buf",
                                          staticWorkspace ? "matmul_qkv_prefill" : "matmul_qkv_prefill_piece",
                                          {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                                          mOpenCLBackend->getPrecision());
        OPENCL_CHECK_KERNEL(mQKKernel);
        OPENCL_CHECK_KERNEL(mQKVKernel);
        mFastKernelStatic = staticWorkspace;
        mFastKernelSparse = false;
    }
    auto run3D = [&](const std::shared_ptr<KernelWrap>& kernel, std::vector<uint32_t> gws,
                    const std::string& kernelName, const std::string& programName) {
        auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(kernel));
        auto lws = localWS3DDefault(gws, maxWorkGroupSize, runtime, kernelName, kernel,
                                    mOpenCLBackend->getCLTuneLevel(), programName).first;
        gws[0] = ROUND_UP(gws[0], std::max((uint32_t)1, lws[0]));
        gws[1] = ROUND_UP(gws[1], std::max((uint32_t)1, lws[1]));
        gws[2] = ROUND_UP(gws[2], std::max((uint32_t)1, lws[2]));
        run3DKernelDefault(kernel, gws, lws, runtime);
    };

    auto tempBuffer = [&](Tensor* tensor) -> cl::Buffer& {
        return staticWorkspace ? openCLBuffer(tensor) : openCLDeferBuffer(tensor);
    };

    const int seqPack = ROUND_UP(mQuerySeqLen, 4);
    const int kvPack = ROUND_UP(kvLen, 4);
    const int headPack4 = ROUND_UP(mHeadDim, 4);
    const int headPack8 = ROUND_UP(mHeadDim, 8);
    const float scale = (mMeta && mMeta->attn_scale > 0)
        ? mMeta->attn_scale
        : (1.0f / std::sqrt(static_cast<float>(mHeadDim)));
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    std::vector<uint32_t> gws;

    idx = 0;
    gws = {static_cast<uint32_t>(UP_DIV(mQuerySeqLen, 4)), static_cast<uint32_t>(UP_DIV(mHeadDim, 4)),
           static_cast<uint32_t>(mNumHead * mBatch)};
    ret |= mRearrangeQKernel->get().setArg(idx++, gws[0]);
    ret |= mRearrangeQKernel->get().setArg(idx++, gws[1]);
    ret |= mRearrangeQKernel->get().setArg(idx++, gws[2]);
    ret |= mRearrangeQKernel->get().setArg(idx++, openCLBuffer(query));
    ret |= mRearrangeQKernel->get().setArg(idx++, tempBuffer(mTempQ.get()));
    ret |= mRearrangeQKernel->get().setArg(idx++, mQuerySeqLen);
    ret |= mRearrangeQKernel->get().setArg(idx++, mHeadDim);
    ret |= mRearrangeQKernel->get().setArg(idx++, mNumHead);
    MNN_CHECK_CL_SUCCESS(ret, "setArg paged fast rearrange_q");
    run3D(mRearrangeQKernel, gws, "rearrange_q", "attention_buf");

    idx = 0;
    gws = {static_cast<uint32_t>(UP_DIV(kvLen, 4)), static_cast<uint32_t>(UP_DIV(mHeadDim, 4)),
           static_cast<uint32_t>(mKvNumHead * mBatch)};
    ret = CL_SUCCESS;
    ret |= mPackPagedKVKernel->get().setArg(idx++, gws[0]);
    ret |= mPackPagedKVKernel->get().setArg(idx++, gws[1]);
    ret |= mPackPagedKVKernel->get().setArg(idx++, gws[2]);
    ret |= mPackPagedKVKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= mPackPagedKVKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= mPackPagedKVKernel->get().setArg(idx++, tempBuffer(mTempK.get()));
    ret |= mPackPagedKVKernel->get().setArg(idx++, tempBuffer(mTempV.get()));
    ret |= mPackPagedKVKernel->get().setArg(idx++, openCLBuffer(mCache->slotTable.get()));
    ret |= mPackPagedKVKernel->get().setArg(idx++, mBatch);
    ret |= mPackPagedKVKernel->get().setArg(idx++, kvLen);
    ret |= mPackPagedKVKernel->get().setArg(idx++, mKvNumHead);
    ret |= mPackPagedKVKernel->get().setArg(idx++, mHeadDim);
    ret |= mPackPagedKVKernel->get().setArg(idx++, mCache->maxSlots);
    MNN_CHECK_CL_SUCCESS(ret, "setArg pack_paged_kv_prefill");
    run3D(mPackPagedKVKernel, gws, "pack_paged_kv_prefill", "paged_attention_buf");

    idx = 0;
    gws = {static_cast<uint32_t>(UP_DIV(mQuerySeqLen, 4)), static_cast<uint32_t>(UP_DIV(maskKeyLen, 4)),
           static_cast<uint32_t>(mBatch)};
    ret = CL_SUCCESS;
    ret |= mRearrangeMaskKernel->get().setArg(idx++, gws[0]);
    ret |= mRearrangeMaskKernel->get().setArg(idx++, gws[1]);
    ret |= mRearrangeMaskKernel->get().setArg(idx++, gws[2]);
    ret |= mRearrangeMaskKernel->get().setArg(idx++, openCLBuffer(mask));
    ret |= mRearrangeMaskKernel->get().setArg(idx++, tempBuffer(mTempMask.get()));
    ret |= mRearrangeMaskKernel->get().setArg(idx++, mQuerySeqLen);
    ret |= mRearrangeMaskKernel->get().setArg(idx++, maskKeyLen);
    MNN_CHECK_CL_SUCCESS(ret, "setArg paged fast rearrange_mask_shortprefill");
    run3D(mRearrangeMaskKernel, gws, "rearrange_mask_shortprefill", "attention_buf");

    int qSplitNum = 1;
    if (staticWorkspace) {
        idx = 0;
        gws = {static_cast<uint32_t>(UP_DIV(mQuerySeqLen, 4)), static_cast<uint32_t>(UP_DIV(kvLen, 4)),
               static_cast<uint32_t>(mNumHead * mBatch)};
        ret = CL_SUCCESS;
        ret |= mQKKernel->get().setArg(idx++, gws[0]);
        ret |= mQKKernel->get().setArg(idx++, gws[1]);
        ret |= mQKKernel->get().setArg(idx++, gws[2]);
        ret |= mQKKernel->get().setArg(idx++, tempBuffer(mTempQ.get()));
        ret |= mQKKernel->get().setArg(idx++, tempBuffer(mTempK.get()));
        ret |= mQKKernel->get().setArg(idx++, tempBuffer(mTempMask.get()));
        ret |= mQKKernel->get().setArg(idx++, tempBuffer(mTempQK.get()));
        ret |= mQKKernel->get().setArg(idx++, scale);
        ret |= mQKKernel->get().setArg(idx++, mQuerySeqLen);
        ret |= mQKKernel->get().setArg(idx++, maskKeyLen);
        ret |= mQKKernel->get().setArg(idx++, kvLen);
        ret |= mQKKernel->get().setArg(idx++, kvPack);
        ret |= mQKKernel->get().setArg(idx++, mNumHead);
        ret |= mQKKernel->get().setArg(idx++, headPack4);
        MNN_CHECK_CL_SUCCESS(ret, "setArg paged fast matmul_qk_div_mask_prefill");
        run3D(mQKKernel, gws, "matmul_qk_div_mask_prefill", "attention_buf");

        idx = 0;
        gws = {64u, static_cast<uint32_t>(UP_DIV(seqPack, 4)), static_cast<uint32_t>(mNumHead * mBatch)};
        ret = CL_SUCCESS;
        ret |= mSoftmaxKernel->get().setArg(idx++, gws[0]);
        ret |= mSoftmaxKernel->get().setArg(idx++, gws[1]);
        ret |= mSoftmaxKernel->get().setArg(idx++, gws[2]);
        ret |= mSoftmaxKernel->get().setArg(idx++, tempBuffer(mTempQK.get()));
        ret |= mSoftmaxKernel->get().setArg(idx++, tempBuffer(mTempSoftmax.get()));
        ret |= mSoftmaxKernel->get().setArg(idx++, seqPack);
        ret |= mSoftmaxKernel->get().setArg(idx++, mNumHead * mBatch);
        ret |= mSoftmaxKernel->get().setArg(idx++, kvLen);
        MNN_CHECK_CL_SUCCESS(ret, "setArg paged fast softmax");
        run3DKernelDefault(mSoftmaxKernel, gws, {64u, 1u, 1u}, runtime);

        idx = 0;
        gws = {static_cast<uint32_t>(UP_DIV(mHeadDim, 8)), static_cast<uint32_t>(UP_DIV(mQuerySeqLen, 4)),
               static_cast<uint32_t>(mNumHead * mBatch)};
        ret = CL_SUCCESS;
        ret |= mQKVKernel->get().setArg(idx++, gws[0]);
        ret |= mQKVKernel->get().setArg(idx++, gws[1]);
        ret |= mQKVKernel->get().setArg(idx++, gws[2]);
        ret |= mQKVKernel->get().setArg(idx++, tempBuffer(mTempSoftmax.get()));
        ret |= mQKVKernel->get().setArg(idx++, tempBuffer(mTempV.get()));
        ret |= mQKVKernel->get().setArg(idx++, openCLBuffer(output));
        ret |= mQKVKernel->get().setArg(idx++, mQuerySeqLen);
        ret |= mQKVKernel->get().setArg(idx++, kvLen);
        ret |= mQKVKernel->get().setArg(idx++, kvPack);
        ret |= mQKVKernel->get().setArg(idx++, mNumHead);
        ret |= mQKVKernel->get().setArg(idx++, mKvNumHead);
        ret |= mQKVKernel->get().setArg(idx++, headPack8);
        MNN_CHECK_CL_SUCCESS(ret, "setArg paged fast matmul_qkv_prefill");
        run3D(mQKVKernel, gws, "matmul_qkv_prefill", "attention_buf");
    } else {
        qSplitNum = UP_DIV(mQuerySeqLen, qChunkLen);
        for (int piece = 0; piece < qSplitNum; ++piece) {
            const int qStart = piece * qChunkLen;
            const int qPieceLen = std::min(qChunkLen, mQuerySeqLen - qStart);
            if (qPieceLen <= 0) {
                continue;
            }
            const int qPiecePack = ROUND_UP(qPieceLen, 4);

            idx = 0;
            gws = {static_cast<uint32_t>(UP_DIV(qPieceLen, 4)), static_cast<uint32_t>(UP_DIV(kvLen, 4)),
                   static_cast<uint32_t>(mNumHead * mBatch)};
            ret = CL_SUCCESS;
            ret |= mQKKernel->get().setArg(idx++, gws[0]);
            ret |= mQKKernel->get().setArg(idx++, gws[1]);
            ret |= mQKKernel->get().setArg(idx++, gws[2]);
            ret |= mQKKernel->get().setArg(idx++, tempBuffer(mTempQ.get()));
            ret |= mQKKernel->get().setArg(idx++, tempBuffer(mTempK.get()));
            ret |= mQKKernel->get().setArg(idx++, tempBuffer(mTempMask.get()));
            ret |= mQKKernel->get().setArg(idx++, tempBuffer(mTempQK.get()));
            ret |= mQKKernel->get().setArg(idx++, scale);
            ret |= mQKKernel->get().setArg(idx++, mQuerySeqLen);
            ret |= mQKKernel->get().setArg(idx++, qStart);
            ret |= mQKKernel->get().setArg(idx++, qPieceLen);
            ret |= mQKKernel->get().setArg(idx++, maskKeyLen);
            ret |= mQKKernel->get().setArg(idx++, kvLen);
            ret |= mQKKernel->get().setArg(idx++, kvPack);
            ret |= mQKKernel->get().setArg(idx++, mNumHead);
            ret |= mQKKernel->get().setArg(idx++, headPack4);
            MNN_CHECK_CL_SUCCESS(ret, "setArg paged fast matmul_qk_div_mask_prefill_piece");
            run3D(mQKKernel, gws, "matmul_qk_div_mask_prefill_piece", "attention_buf");

            idx = 0;
            gws = {64u, static_cast<uint32_t>(UP_DIV(qPiecePack, 4)), static_cast<uint32_t>(mNumHead * mBatch)};
            ret = CL_SUCCESS;
            ret |= mSoftmaxKernel->get().setArg(idx++, gws[0]);
            ret |= mSoftmaxKernel->get().setArg(idx++, gws[1]);
            ret |= mSoftmaxKernel->get().setArg(idx++, gws[2]);
            ret |= mSoftmaxKernel->get().setArg(idx++, tempBuffer(mTempQK.get()));
            ret |= mSoftmaxKernel->get().setArg(idx++, tempBuffer(mTempSoftmax.get()));
            ret |= mSoftmaxKernel->get().setArg(idx++, qPiecePack);
            ret |= mSoftmaxKernel->get().setArg(idx++, mNumHead * mBatch);
            ret |= mSoftmaxKernel->get().setArg(idx++, kvLen);
            MNN_CHECK_CL_SUCCESS(ret, "setArg paged fast softmax");
            run3DKernelDefault(mSoftmaxKernel, gws, {64u, 1u, 1u}, runtime);

            idx = 0;
            gws = {static_cast<uint32_t>(UP_DIV(mHeadDim, 8)), static_cast<uint32_t>(UP_DIV(qPieceLen, 4)),
                   static_cast<uint32_t>(mNumHead * mBatch)};
            ret = CL_SUCCESS;
            ret |= mQKVKernel->get().setArg(idx++, gws[0]);
            ret |= mQKVKernel->get().setArg(idx++, gws[1]);
            ret |= mQKVKernel->get().setArg(idx++, gws[2]);
            ret |= mQKVKernel->get().setArg(idx++, tempBuffer(mTempSoftmax.get()));
            ret |= mQKVKernel->get().setArg(idx++, tempBuffer(mTempV.get()));
            ret |= mQKVKernel->get().setArg(idx++, openCLBuffer(output));
            ret |= mQKVKernel->get().setArg(idx++, mQuerySeqLen);
            ret |= mQKVKernel->get().setArg(idx++, qStart);
            ret |= mQKVKernel->get().setArg(idx++, qPieceLen);
            ret |= mQKVKernel->get().setArg(idx++, kvLen);
            ret |= mQKVKernel->get().setArg(idx++, kvPack);
            ret |= mQKVKernel->get().setArg(idx++, mNumHead);
            ret |= mQKVKernel->get().setArg(idx++, mKvNumHead);
            ret |= mQKVKernel->get().setArg(idx++, headPack8);
            MNN_CHECK_CL_SUCCESS(ret, "setArg paged fast matmul_qkv_prefill_piece");
            run3D(mQKVKernel, gws, "matmul_qkv_prefill_piece", "attention_buf");
        }
    }
    if (profile) {
        runtime->commandQueue().finish();
        int layerIndex = mLayerIndex >= 0 ? mLayerIndex : (mMeta != nullptr ? mMeta->layer_index : -1);
        MNN_PRINT("OpenCLPagedAttention profile op=prefill_attention_fast_qk_softmax_qkv layer=%d query=%d kv_len=%d "
                  "mask_key_len=%d q_chunk=%d q_split=%d static=%d us=%llu\n",
                  layerIndex, mQuerySeqLen, kvLen, maskKeyLen, qChunkLen, qSplitNum, staticWorkspace ? 1 : 0,
                  static_cast<unsigned long long>(_nowUs() - startUs));
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (inputs.size() < 3 || outputs.empty()) {
        if (_picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA resize invalid io inputs=%d outputs=%d mode=%d\n",
                      static_cast<int>(inputs.size()), static_cast<int>(outputs.size()), mPicAttentionMode);
        }
        return INVALID_VALUE;
    }
    mBytes = mOpenCLBackend->getPrecision() == BackendConfig::Precision_High ? 4 : 2;
    auto query = inputs[0];
    auto key = inputs[1];
    mBatch = query->length(0);
    mQuerySeqLen = query->length(1);
    mNumHead = query->length(2);
    mHeadDim = query->length(3);
    mKvNumHead = key->length(2);
    mNewKvSeqLen = key->length(1);
    if (mHeadDim <= 0 || mKvNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        if (_picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA resize invalid dims layer=%d mode=%d q=%d new_kv=%d heads=%d kv_heads=%d dim=%d\n",
                      mLayerIndex, mPicAttentionMode, mQuerySeqLen, mNewKvSeqLen, mNumHead, mKvNumHead, mHeadDim);
        }
        return INVALID_VALUE;
    }
    mQKKernel.reset();
    mQKVKernel.reset();
    int maxSlots = mNewKvSeqLen;
    if (mMeta != nullptr) {
        maxSlots = std::max(maxSlots, mMeta->request_capacity > 0 ? mMeta->request_capacity : mMeta->max_tokens);
        if (maxSlots <= 0) {
            maxSlots = static_cast<int>(mMeta->previous) + mNewKvSeqLen;
        }
        const size_t sourceSlots = _picCacheSourceSlotCount(mMeta);
        if (sourceSlots > 0) {
            const int sourceBase = _picCacheSourceSlotBase(mMeta, maxSlots);
            if (sourceSlots > static_cast<size_t>(std::numeric_limits<int>::max() - sourceBase)) {
                if (_picOpenCLDebug()) {
                    MNN_PRINT("PIC OpenCL PA resize source slot overflow layer=%d mode=%d max_slots=%d source_slots=%d\n",
                              mLayerIndex, mPicAttentionMode, maxSlots, static_cast<int>(sourceSlots));
                }
                return OUT_OF_MEMORY;
            }
            maxSlots = std::max(maxSlots, sourceBase + static_cast<int>(sourceSlots));
        }
    }
    auto err = ensureCache(maxSlots, mBatch, mKvNumHead, mHeadDim);
    const int layerIndex = mLayerIndex >= 0 ? mLayerIndex : (mMeta != nullptr ? mMeta->layer_index : 0);
    if (err == NO_ERROR && !_legacyB863976OpenCL() && mMeta != nullptr &&
        mMeta->shouldHydrateExternalLayer(layerIndex)) {
        size_t kept = mMeta->previous >= mMeta->remove ? (mMeta->previous - mMeta->remove) : 0;
        const int baseLogical = static_cast<int>(kept) + _reverseCount(mMeta);
        const int kvWriteLen = mMeta->add > 0
            ? static_cast<int>(std::min<size_t>(mMeta->add, mNewKvSeqLen))
            : mNewKvSeqLen;
        const bool plannedSparseLayer = mPicAttentionMode == 2 &&
            (mMeta->sparse_query_active || mMeta->cacheblend_score_active || mMeta->pic_graph_active_plan_ready);
        const int prefetchKvLen = plannedSparseLayer ? std::max(0, mMeta->logical_length)
                                                     : (baseLogical + kvWriteLen);
        if (prefetchKvLen > 0 && prefetchKvLen <= maxSlots) {
            _scheduleExternalLayerReadsFrom(mMeta, std::max(0, mMeta->external_hydrate_start_layer_idx),
                                            mBatch, mKvNumHead, mHeadDim, mBytes, prefetchKvLen);
        }
    }
    if (_picOpenCLDebug()) {
        const int outLen = outputs[0] != nullptr && outputs[0]->dimensions() > 1 ? outputs[0]->length(1) : -1;
        const int budget = inputs.size() > 4 && inputs[4] != nullptr && inputs[4]->host<int32_t>() != nullptr
            ? inputs[4]->host<int32_t>()[0]
            : -1;
        MNN_PRINT("PIC OpenCL PA resize layer=%d mode=%d q=%d new_kv=%d out=%d budget=%d max_slots=%d "
                  "request_active=%d capacity=%d previous=%d add=%d logical=%d source_segments=%d err=%d\n",
                  mLayerIndex, mPicAttentionMode, mQuerySeqLen, mNewKvSeqLen, outLen, budget, maxSlots,
                  mMeta != nullptr && mMeta->request_active ? 1 : 0,
                  mMeta != nullptr ? mMeta->request_capacity : -1,
                  mMeta != nullptr ? static_cast<int>(mMeta->previous) : -1,
                  mMeta != nullptr ? static_cast<int>(mMeta->add) : -1,
                  mMeta != nullptr ? mMeta->logical_length : -1,
                  mMeta != nullptr ? static_cast<int>(mMeta->external_segments.size()) : -1,
                  static_cast<int>(err));
    }
    return err;
}

ErrorCode PagedAttentionBufExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    const Tensor* mask = inputs.size() > 3 ? inputs[3] : nullptr;

    if (mMeta != nullptr && mMeta->request_capacity <= 0 && !mMeta->request_active) {
        mMeta->beginRequest(std::max(mNewKvSeqLen, mQuerySeqLen));
    }
    int layerIndex = mLayerIndex >= 0 ? mLayerIndex : (mMeta != nullptr ? mMeta->layer_index : 0);
    const bool decodeStep = mMeta != nullptr && mQuerySeqLen == 1 && mNewKvSeqLen == 1 &&
        mMeta->previous > 0 && !mMeta->cacheblend_score_active && !mMeta->pic_graph_active_plan_ready;
    const bool picRuntimeActive = mMeta != nullptr &&
        (mMeta->sparse_query_active || mMeta->cacheblend_score_active || mMeta->pic_graph_active_plan_ready);
    const int effectivePicAttentionMode = (decodeStep || !picRuntimeActive) ? 0 : mPicAttentionMode;
    if (!decodeStep && mMeta != nullptr && mMeta->sparseQueryBlockedBeforeLayer(layerIndex)) {
        MNN_ERROR("OpenCLPagedAttention layer %d received sparse query before sparse_start_layer=%d. "
                  "Run full prompt until the score layer, crop active hidden states, then resume sparse.\n",
                  layerIndex, mMeta->sparse_query_start_layer_idx);
        return INVALID_VALUE;
    }

    int reverse = _reverseCount(mMeta);
    int baseLogical = 0;
    int kvWriteLen = mNewKvSeqLen;
    int attnLen = output->length(1);
    bool sparseQuery = mMeta != nullptr && mMeta->sparseQueryActiveForLayer(layerIndex);
    const bool scoreAttention = effectivePicAttentionMode == 1;
    if (scoreAttention) {
        sparseQuery = false;
    }
    if (mMeta != nullptr) {
        size_t kept = mMeta->previous >= mMeta->remove ? (mMeta->previous - mMeta->remove) : 0;
        baseLogical = static_cast<int>(kept) + reverse;
        kvWriteLen = mMeta->add > 0 ? static_cast<int>(std::min<size_t>(mMeta->add, mNewKvSeqLen)) : mNewKvSeqLen;
    }
    if (sparseQuery) {
        if (static_cast<int>(mMeta->sparse_query_logical_indices.size()) < attnLen) {
            return INVALID_VALUE;
        }
        baseLogical = 0;
    } else if (effectivePicAttentionMode == 2) {
        MNN_ERROR("OpenCL PicSparseAttention layer %d requires active sparse rows from PicScoreAttention\n",
                  layerIndex);
        return INVALID_VALUE;
    }
    int kvLen = sparseQuery ? std::max(0, mMeta->logical_length) : (baseLogical + kvWriteLen);
    if (_picOpenCLDebug()) {
        MNN_PRINT("PIC OpenCL PA exec layer=%d mode=%d effective=%d runtime_active=%d decode=%d q=%d new_kv=%d "
                  "attn=%d kv_write=%d kv_len=%d base=%d sparse=%d score=%d cacheblend=%d graph=%d "
                  "request_active=%d capacity=%d max_slots=%d previous=%d add=%d logical=%d source_segments=%d "
                  "loaded_layers=%d outputs=%d\n",
                  layerIndex, mPicAttentionMode, effectivePicAttentionMode, picRuntimeActive ? 1 : 0,
                  decodeStep ? 1 : 0, mQuerySeqLen, mNewKvSeqLen, attnLen, kvWriteLen, kvLen, baseLogical,
                  sparseQuery ? 1 : 0, scoreAttention ? 1 : 0,
                  mMeta != nullptr && mMeta->cacheblend_score_active ? 1 : 0,
                  mMeta != nullptr && mMeta->pic_graph_active_plan_ready ? 1 : 0,
                  mMeta != nullptr && mMeta->request_active ? 1 : 0,
                  mMeta != nullptr ? mMeta->request_capacity : -1,
                  mCache != nullptr ? mCache->maxSlots : -1,
                  mMeta != nullptr ? static_cast<int>(mMeta->previous) : -1,
                  mMeta != nullptr ? static_cast<int>(mMeta->add) : -1,
                  mMeta != nullptr ? mMeta->logical_length : -1,
                  mMeta != nullptr ? static_cast<int>(mMeta->external_segments.size()) : -1,
                  mMeta != nullptr ? static_cast<int>(mMeta->external_loaded_layers.size()) : -1,
                  static_cast<int>(outputs.size()));
    }
    if (kvLen > mCache->maxSlots) {
        if (_picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA exec fail capacity layer=%d kv_len=%d max_slots=%d\n",
                      layerIndex, kvLen, mCache->maxSlots);
        }
        return OUT_OF_MEMORY;
    }
    auto err = syncSlotTable(kvLen);
    if (err != NO_ERROR) {
        if (_picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA exec fail slot_table layer=%d err=%d kv_len=%d\n",
                      layerIndex, static_cast<int>(err), kvLen);
        }
        return err;
    }
    err = syncSparseQuery(sparseQuery ? attnLen : 0);
    if (err != NO_ERROR) {
        if (_picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA exec fail sparse_query layer=%d err=%d attn=%d\n",
                      layerIndex, static_cast<int>(err), attnLen);
        }
        return err;
    }

    std::vector<int> physicalSlots(kvLen);
    for (int l = 0; l < kvLen; ++l) {
        physicalSlots[l] = mMeta ? mMeta->physicalSlot(l) : l;
    }
    auto& queue = mOpenCLBackend->getOpenCLRuntime()->commandQueue();
    if (!_legacyB863976OpenCL()) {
        _registerExternalLayerMappedTarget(mMeta, layerIndex, mBatch, mKvNumHead, mHeadDim, mBytes, mCache->maxSlots,
                                           mCache->key, mCache->value, queue);
    }
    if (!_legacyB863976OpenCL() && mMeta != nullptr && mMeta->shouldHydrateExternalLayer(layerIndex)) {
        _scheduleExternalLayerReadsFrom(mMeta, std::max(0, mMeta->external_hydrate_start_layer_idx),
                                        mBatch, mKvNumHead, mHeadDim, mBytes, kvLen);
    }
    const bool shouldHydrateExternal = mMeta != nullptr && mMeta->shouldHydrateExternalLayer(layerIndex);
    if (shouldHydrateExternal && !mMeta->externalLayerLoaded(layerIndex)) {
        auto hydrate = hydrateExternalSegments(layerIndex, kvLen);
        if (hydrate != NO_ERROR) {
            if (_picOpenCLDebug()) {
                MNN_PRINT("PIC OpenCL PA exec fail hydrate layer=%d err=%d kv_len=%d\n",
                          layerIndex, static_cast<int>(hydrate), kvLen);
            }
            return hydrate;
        }
    }

    if (!mIsKVShared && kvWriteLen > 0) {
        const int total = mBatch * kvWriteLen * mKvNumHead * mHeadDim;
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mCopyKernel->get().setArg(idx++, openCLBuffer(key));
        ret |= mCopyKernel->get().setArg(idx++, openCLBuffer(value));
        ret |= mCopyKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= mCopyKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= mCopyKernel->get().setArg(idx++, openCLBuffer(mCache->slotTable.get()));
        ret |= mCopyKernel->get().setArg(idx++, openCLBuffer(mCache->sparseQuery.get()));
        ret |= mCopyKernel->get().setArg(idx++, mBatch);
        ret |= mCopyKernel->get().setArg(idx++, mNewKvSeqLen);
        ret |= mCopyKernel->get().setArg(idx++, kvWriteLen);
        ret |= mCopyKernel->get().setArg(idx++, mKvNumHead);
        ret |= mCopyKernel->get().setArg(idx++, mHeadDim);
        ret |= mCopyKernel->get().setArg(idx++, baseLogical);
        ret |= mCopyKernel->get().setArg(idx++, mCache->maxSlots);
        ret |= mCopyKernel->get().setArg(idx++, sparseQuery ? 1 : 0);
        ret |= mCopyKernel->get().setArg(idx++, total);
        MNN_CHECK_CL_SUCCESS(ret, "setArg copy_paged_kv");
        queue.enqueueNDRangeKernel(mCopyKernel->get(), cl::NullRange, cl::NDRange(total), cl::NullRange);
    }

    err = runCacheBlendScoring(layerIndex, kvLen);
    if (err != NO_ERROR) {
        if (_picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA exec fail scoring layer=%d err=%d kv_len=%d\n",
                      layerIndex, static_cast<int>(err), kvLen);
        }
        return err;
    }
    if (scoreAttention && outputs.size() > 1) {
        err = _emitActiveIndicesOpenCL(mMeta, layerIndex, kvLen, outputs[1], mOpenCLBackend);
        if (err != NO_ERROR) {
            return err;
        }
        sparseQuery = mMeta != nullptr && mMeta->sparseQueryActiveForLayer(layerIndex);
        if (sparseQuery) {
            if (static_cast<int>(mMeta->sparse_query_logical_indices.size()) < attnLen) {
                return INVALID_VALUE;
            }
            baseLogical = 0;
            kvLen = std::max(0, mMeta->logical_length);
            err = syncSparseQuery(attnLen);
            if (err != NO_ERROR) {
                return err;
            }
        }
    } else if (outputs.size() > 1) {
        err = _emitIdentityIndicesOpenCL(attnLen, outputs[1], mOpenCLBackend);
        if (err != NO_ERROR) {
            return err;
        }
    }

    if (mMeta != nullptr && !mMeta->file_name.empty() && mMeta->file_flag == KVMeta::PendingWrite && kvLen > 0) {
        auto prefixDir = mOpenCLBackend->getRuntime()->hint().prefixcacheDirPath;
        MNNCreateDir(prefixDir.c_str());
        std::string basePath = MNNFilePathConcat(prefixDir, mMeta->file_name) + "_" + std::to_string(layerIndex);
        const size_t valueBytes = static_cast<size_t>(mBatch) * mKvNumHead * mCache->maxSlots * mHeadDim * mBytes;
        std::vector<int8_t> valueStorage(valueBytes);
        const size_t keyElements = static_cast<size_t>(kvLen) * mBatch * mKvNumHead * mHeadDim;
        if (keyElements > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return OUT_OF_MEMORY;
        }
        auto exportErr = ensureExternalTemps(keyElements, 1);
        if (exportErr != NO_ERROR) {
            return exportErr;
        }
        std::vector<int8_t> keyData(keyElements * mBytes);
        std::vector<int8_t> valueData(static_cast<size_t>(mBatch) * mKvNumHead * kvLen * mHeadDim * mBytes);
        int ropeDim = _ropeDimForExport(mMeta, mHeadDim);
        ropeDim = std::min(ropeDim, mHeadDim);
        ropeDim = (ropeDim / 2) * 2;
        const int oldContext = mMeta != nullptr && mMeta->rope_scaling_original_max_position_embeddings > 0
            ? mMeta->rope_scaling_original_max_position_embeddings
            : (mMeta != nullptr ? mMeta->max_position_embeddings : 0);
        uint32_t exportIdx = 0;
        cl_int exportRet = CL_SUCCESS;
        exportRet |= mExportCanonicalKeyKernel->get().setArg(exportIdx++, openCLBuffer(mCache->key.get()));
        exportRet |= mExportCanonicalKeyKernel->get().setArg(exportIdx++, openCLBuffer(mExternalKey.get()));
        exportRet |= mExportCanonicalKeyKernel->get().setArg(exportIdx++, openCLBuffer(mCache->slotTable.get()));
        exportRet |= mExportCanonicalKeyKernel->get().setArg(exportIdx++, mBatch);
        exportRet |= mExportCanonicalKeyKernel->get().setArg(exportIdx++, kvLen);
        exportRet |= mExportCanonicalKeyKernel->get().setArg(exportIdx++, mKvNumHead);
        exportRet |= mExportCanonicalKeyKernel->get().setArg(exportIdx++, mHeadDim);
        exportRet |= mExportCanonicalKeyKernel->get().setArg(exportIdx++, mCache->maxSlots);
        exportRet |= mExportCanonicalKeyKernel->get().setArg(exportIdx++, ropeDim);
        exportRet |= mExportCanonicalKeyKernel->get().setArg(
            exportIdx++, mMeta != nullptr && mMeta->rope_theta > 0.0f ? mMeta->rope_theta : 10000.0f);
        exportRet |= mExportCanonicalKeyKernel->get().setArg(
            exportIdx++, mMeta != nullptr && mMeta->rope_type == "llama3" ? 1 : 0);
        exportRet |= mExportCanonicalKeyKernel->get().setArg(
            exportIdx++, mMeta != nullptr ? std::max(mMeta->rope_scaling_factor, 1.0f) : 1.0f);
        exportRet |= mExportCanonicalKeyKernel->get().setArg(
            exportIdx++, mMeta != nullptr ? std::max(mMeta->rope_scaling_low_freq_factor, 1.0e-6f) : 1.0f);
        exportRet |= mExportCanonicalKeyKernel->get().setArg(
            exportIdx++, mMeta != nullptr ? std::max(mMeta->rope_scaling_high_freq_factor, 1.0e-6f) : 4.0f);
        exportRet |= mExportCanonicalKeyKernel->get().setArg(exportIdx++, oldContext);
        exportRet |= mExportCanonicalKeyKernel->get().setArg(
            exportIdx++, mMeta != nullptr ? mMeta->max_position_embeddings : 0);
        exportRet |= mExportCanonicalKeyKernel->get().setArg(
            exportIdx++, mMeta != nullptr ? mMeta->rope_attention_scaling : 1.0f);
        exportRet |= mExportCanonicalKeyKernel->get().setArg(exportIdx++, static_cast<int>(keyElements));
        MNN_CHECK_CL_SUCCESS(exportRet, "setArg export_canonical_paged_key");
        exportRet = queue.enqueueNDRangeKernel(mExportCanonicalKeyKernel->get(), cl::NullRange,
                                               cl::NDRange(static_cast<int>(keyElements)), cl::NullRange);
        MNN_CHECK_CL_SUCCESS(exportRet, "enqueue export_canonical_paged_key");
        const bool keyExportOk = queue.enqueueReadBuffer(openCLBuffer(mExternalKey.get()), CL_TRUE, 0,
                                                         keyData.size(), keyData.data()) == CL_SUCCESS;
        const bool valueExportOk = queue.enqueueReadBuffer(openCLBuffer(mCache->value.get()), CL_TRUE, 0,
                                                           valueBytes, valueStorage.data()) == CL_SUCCESS;
        for (int l = 0; l < kvLen; ++l) {
            int slot = physicalSlots[l];
            if (slot < 0 || slot >= mCache->maxSlots) {
                continue;
            }
            for (int b = 0; b < mBatch; ++b) {
                for (int h = 0; h < mKvNumHead; ++h) {
                    const int8_t* srcV = valueStorage.data() + ((b * mKvNumHead + h) * mCache->maxSlots + slot) * mHeadDim * mBytes;
                    int8_t* dstV = valueData.data() + ((b * mKvNumHead + h) * kvLen + l) * mHeadDim * mBytes;
                    ::memcpy(dstV, srcV, mHeadDim * mBytes);
                }
            }
        }
        if (!keyExportOk) {
            MNN_PRINT("OpenCLPagedAttention: failed to export key cache from GPU\n");
        } else if (!_writeBinaryFile(basePath + ".k", keyData)) {
            MNN_PRINT("OpenCLPagedAttention: failed to export key cache: %s\n", (basePath + ".k").c_str());
        }
        if (!valueExportOk) {
            MNN_PRINT("OpenCLPagedAttention: failed to export value cache from GPU\n");
        } else if (!_writeBinaryFile(basePath + ".v", valueData)) {
            MNN_PRINT("OpenCLPagedAttention: failed to export value cache: %s\n", (basePath + ".v").c_str());
        }
        if (!_writeShapeFile(basePath + ".json", mBatch, mKvNumHead, mHeadDim, kvLen, mBytes, mMeta)) {
            MNN_PRINT("OpenCLPagedAttention: failed to export shape metadata: %s\n", (basePath + ".json").c_str());
        }
        if (mLayerIndex < 0) {
            mMeta->layer_index = (mMeta->layer_index + 1) % std::max(1, mMeta->layer_nums);
        }
    }

    if (attnLen <= 0 || kvLen <= 0) {
        return NO_ERROR;
    }
    mScale = (mMeta && mMeta->attn_scale > 0) ? mMeta->attn_scale : (1.0f / std::sqrt(static_cast<float>(mHeadDim)));
    bool useMask = mask != nullptr && mask->elementSize() > 1 && mask->getType().code == halide_type_float;
    int maskElements = useMask ? static_cast<int>(mask->elementSize()) : 0;
    bool externalHydrated = !shouldHydrateExternal || mMeta->externalLayerLoaded(layerIndex);
    const bool queryRowsAreFull = sparseQuery && mQuerySeqLen > attnLen;
    int fastMaskKeyLen = 0;
    if (canUseFastPrefill(mask, baseLogical, attnLen, kvLen, sparseQuery, externalHydrated, &fastMaskKeyLen)) {
        return runFastPrefill(inputs, outputs, kvLen, fastMaskKeyLen);
    }
    if (canUseSparseFastPrefill(mask, attnLen, kvLen, externalHydrated, queryRowsAreFull)) {
        return runSparseFastPrefill(inputs, outputs, kvLen, attnLen, queryRowsAreFull);
    }
    const bool benchForceRowKernel = _envFlagEnabled("MNN_PAGED_ATTENTION_BENCH_FORCE_ROW_KERNEL", false) ||
        _envFlagEnabled("MNN_PAGED_ATTENTION_BENCH_FORCE_V2_KERNEL", false);
    const bool disableRowKernel = _envFlagEnabled("MNN_PAGED_ATTENTION_OPENCL_DISABLE_ROW_KERNEL", false);
    const bool rowKernelSupported = mHeadDim <= 256;
    const bool useRowKernel = rowKernelSupported && !disableRowKernel &&
        ((sparseQuery && attnLen >= 1) || benchForceRowKernel);
    const bool profileGeneric = _profilePagedAttention();
    const uint64_t genericStartUs = profileGeneric ? _nowUs() : 0;
    const int outputElements = mBatch * attnLen * mNumHead * mHeadDim;
    if (outputElements > 0) {
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mZeroKernel->get().setArg(idx++, openCLBuffer(output));
        ret |= mZeroKernel->get().setArg(idx++, outputElements);
        MNN_CHECK_CL_SUCCESS(ret, "setArg zero_output");
        queue.enqueueNDRangeKernel(mZeroKernel->get(), cl::NullRange, cl::NDRange(outputElements), cl::NullRange);
    }
    if (useRowKernel) {
        const int totalRows = mBatch * attnLen * mNumHead;
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mAttentionRowKernel->get().setArg(idx++, openCLBuffer(query));
        ret |= mAttentionRowKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= mAttentionRowKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= mAttentionRowKernel->get().setArg(idx++, openCLBuffer(output));
        ret |= mAttentionRowKernel->get().setArg(idx++, useMask ? openCLBuffer(mask) : openCLBuffer(output));
        ret |= mAttentionRowKernel->get().setArg(idx++, openCLBuffer(mCache->slotTable.get()));
        ret |= mAttentionRowKernel->get().setArg(idx++, openCLBuffer(mCache->sparseQuery.get()));
        ret |= mAttentionRowKernel->get().setArg(idx++, maskElements);
        ret |= mAttentionRowKernel->get().setArg(idx++, mBatch);
        ret |= mAttentionRowKernel->get().setArg(idx++, mQuerySeqLen);
        ret |= mAttentionRowKernel->get().setArg(idx++, attnLen);
        ret |= mAttentionRowKernel->get().setArg(idx++, attnLen);
        ret |= mAttentionRowKernel->get().setArg(idx++, mNumHead);
        ret |= mAttentionRowKernel->get().setArg(idx++, mKvNumHead);
        ret |= mAttentionRowKernel->get().setArg(idx++, mHeadDim);
        ret |= mAttentionRowKernel->get().setArg(idx++, baseLogical);
        ret |= mAttentionRowKernel->get().setArg(idx++, kvLen);
        ret |= mAttentionRowKernel->get().setArg(idx++, mCache->maxSlots);
        ret |= mAttentionRowKernel->get().setArg(idx++, mScale);
        ret |= mAttentionRowKernel->get().setArg(idx++, sparseQuery ? 1 : 0);
        ret |= mAttentionRowKernel->get().setArg(idx++, queryRowsAreFull ? 1 : 0);
        ret |= mAttentionRowKernel->get().setArg(idx++, totalRows);
        MNN_CHECK_CL_SUCCESS(ret, "setArg paged_attention_row");
        queue.enqueueNDRangeKernel(mAttentionRowKernel->get(), cl::NullRange, cl::NDRange(totalRows), cl::NullRange);
        if (profileGeneric) {
            queue.finish();
            MNN_PRINT("OpenCLPagedAttention profile op=row layer=%d query=%d attn=%d kv_write=%d kv_len=%d sparse=%d full_q=%d us=%llu\n",
                      layerIndex, mQuerySeqLen, attnLen, kvWriteLen, kvLen, sparseQuery ? 1 : 0,
                      queryRowsAreFull ? 1 : 0,
                      static_cast<unsigned long long>(_nowUs() - genericStartUs));
        }
        return NO_ERROR;
    }
    const int total = mBatch * attnLen * mNumHead * mHeadDim;
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mAttentionKernel->get().setArg(idx++, openCLBuffer(query));
    ret |= mAttentionKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= mAttentionKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= mAttentionKernel->get().setArg(idx++, openCLBuffer(output));
    ret |= mAttentionKernel->get().setArg(idx++, useMask ? openCLBuffer(mask) : openCLBuffer(output));
    ret |= mAttentionKernel->get().setArg(idx++, openCLBuffer(mCache->slotTable.get()));
    ret |= mAttentionKernel->get().setArg(idx++, openCLBuffer(mCache->sparseQuery.get()));
    ret |= mAttentionKernel->get().setArg(idx++, maskElements);
    ret |= mAttentionKernel->get().setArg(idx++, mBatch);
    ret |= mAttentionKernel->get().setArg(idx++, mQuerySeqLen);
    ret |= mAttentionKernel->get().setArg(idx++, attnLen);
    ret |= mAttentionKernel->get().setArg(idx++, attnLen);
    ret |= mAttentionKernel->get().setArg(idx++, mNumHead);
    ret |= mAttentionKernel->get().setArg(idx++, mKvNumHead);
    ret |= mAttentionKernel->get().setArg(idx++, mHeadDim);
    ret |= mAttentionKernel->get().setArg(idx++, baseLogical);
    ret |= mAttentionKernel->get().setArg(idx++, kvLen);
    ret |= mAttentionKernel->get().setArg(idx++, mCache->maxSlots);
    ret |= mAttentionKernel->get().setArg(idx++, mScale);
    ret |= mAttentionKernel->get().setArg(idx++, sparseQuery ? 1 : 0);
    ret |= mAttentionKernel->get().setArg(idx++, queryRowsAreFull ? 1 : 0);
    ret |= mAttentionKernel->get().setArg(idx++, total);
    MNN_CHECK_CL_SUCCESS(ret, "setArg paged_attention");
    queue.enqueueNDRangeKernel(mAttentionKernel->get(), cl::NullRange, cl::NDRange(total), cl::NullRange);
    if (profileGeneric) {
        queue.finish();
        MNN_PRINT("OpenCLPagedAttention profile op=generic layer=%d query=%d attn=%d kv_write=%d kv_len=%d sparse=%d full_q=%d us=%llu\n",
                  layerIndex, mQuerySeqLen, attnLen, kvWriteLen, kvLen, sparseQuery ? 1 : 0,
                  queryRowsAreFull ? 1 : 0,
                  static_cast<unsigned long long>(_nowUs() - genericStartUs));
    }
    return NO_ERROR;
}

bool PagedAttentionBufExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (dst == nullptr) {
        return true;
    }
    auto tmp = new PagedAttentionBufExecution(op, bn);
    tmp->mCache = mCache;
    tmp->mPicAttentionMode = mPicAttentionMode;
    auto param = op->main_as_AttentionParam();
    tmp->mIsKVShared = param != nullptr && param->kv_shared_layer_index() >= 0;
    *dst = tmp;
    return true;
}

class PagedAttentionBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        for (auto* input : inputs) {
            TensorUtils::setTensorSupportPack(input, false);
        }
        for (auto* output : outputs) {
            TensorUtils::setTensorSupportPack(output, false);
        }
        OPENCL_CREATOR_CHECK(new PagedAttentionBufExecution(op, backend));
    }
};

REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(PagedAttentionBufCreator, OpType_PagedAttention, BUFFER);
REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(PagedAttentionBufCreator, OpType_PicScoreAttention, BUFFER);
REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(PagedAttentionBufCreator, OpType_PicSparseAttention, BUFFER);

} // namespace OpenCL
} // namespace MNN

#endif /* MNN_OPENCL_BUFFER_CLOSED */
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
