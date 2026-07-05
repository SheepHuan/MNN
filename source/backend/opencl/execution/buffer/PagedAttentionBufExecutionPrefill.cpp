//
//  PagedAttentionBufExecutionPrefill.cpp
//  MNN
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/PagedAttentionBufExecution.hpp"
#include "backend/opencl/execution/buffer/PagedAttentionAdrenoUtils.hpp"
#include "backend/opencl/execution/buffer/PagedAttentionMaliUtils.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace MNN {
namespace OpenCL {
namespace {

static thread_local int gSteadyStateTuneMeasureDepth = 0;

static bool _profilePagedAttention() {
    const char* value = ::getenv("MNN_PAGED_ATTENTION_PROFILE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool _profilePagedAttentionTune() {
    const char* value = ::getenv("MNN_PAGED_ATTENTION_PROFILE_TUNE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool _tracePagedAttentionProgress() {
    const char* value = ::getenv("MNN_PAGED_ATTENTION_TRACE_PROGRESS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static int _pagedAttentionProfileLayerIndex(int layerIndex, const PagedKVMeta* meta) {
    return layerIndex >= 0 ? layerIndex : (meta != nullptr ? meta->layer_index : -1);
}

static uint64_t _nowUs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

static void _profileTuneCandidateStart(const char* op, int layerIndex, int queryLen, int kvLen,
                                       const char* candidate) {
    if (!_profilePagedAttention() && !_tracePagedAttentionProgress()) {
        return;
    }
    MNN_PRINT("OpenCLPagedAttention profile op=%s layer=%d query=%d kv_len=%d candidate=%s phase=begin\n",
              op, layerIndex, queryLen, kvLen, candidate != nullptr ? candidate : "unknown");
    std::fflush(stdout);
}

static void _profileTuneCandidateEnd(const char* op, int layerIndex, int queryLen, int kvLen,
                                     const char* candidate, bool ok, uint64_t elapsedUs) {
    if (!_profilePagedAttention() && !_tracePagedAttentionProgress()) {
        return;
    }
    MNN_PRINT("OpenCLPagedAttention profile op=%s layer=%d query=%d kv_len=%d candidate=%s phase=end ok=%d us=%llu\n",
              op, layerIndex, queryLen, kvLen, candidate != nullptr ? candidate : "unknown", ok ? 1 : 0,
              static_cast<unsigned long long>(elapsedUs));
    std::fflush(stdout);
}

template <typename Runner>
static bool _measureSteadyStateCandidate(OpenCLRuntime* runtime, Runner&& runner, uint64_t* elapsedUs) {
    if (runtime == nullptr || elapsedUs == nullptr) {
        return false;
    }
    if (gSteadyStateTuneMeasureDepth > 0) {
        const uint64_t t0 = _nowUs();
        auto timedErr = runner();
        if (timedErr != NO_ERROR) {
            return false;
        }
        runtime->commandQueue().finish();
        *elapsedUs = _nowUs() - t0;
        return true;
    }
    // Tune candidates can trigger deeper first-use work, such as nested family/variant tuning,
    // kernel build, or image mirror setup. Warm once, then measure the second run so the cache
    // records the steady-state path instead of a cold-start artifact.
    ++gSteadyStateTuneMeasureDepth;
    auto warmErr = runner();
    if (warmErr != NO_ERROR) {
        --gSteadyStateTuneMeasureDepth;
        return false;
    }
    runtime->commandQueue().finish();
    const uint64_t t0 = _nowUs();
    auto timedErr = runner();
    if (timedErr != NO_ERROR) {
        --gSteadyStateTuneMeasureDepth;
        return false;
    }
    runtime->commandQueue().finish();
    *elapsedUs = _nowUs() - t0;
    --gSteadyStateTuneMeasureDepth;
    return true;
}

static thread_local int gSparseQSplitChunkOverride = 0;
static thread_local bool gSparseQSplitTuneInProgress = false;
enum SparseFlashKernelVariant : uint32_t {
    kSparseFlashVariantRow32 = 0,
    kSparseFlashVariantRow64 = 1,
    kSparseFlashVariantMQTileHD64Q4K16 = 2,
    kSparseFlashVariantMQTileHD128Q4K16 = 3,
    kSparseFlashVariantMQTileHD128Q4K8 = 4,
    kSparseFlashVariantMQTileHD128Q8K16 = 5,
    kSparseFlashVariantMQTileHD128Q4K8KVImage = 6,
    kSparseFlashVariantMQTileHD128Q8K16KVImage = 7,
    kSparseFlashVariantMQTileHD128Q8K16KImage = 8,
    kSparseFlashVariantMQTileHD128Q4K8KImage = 9,
};
enum ScoreSparsePrefillFamily : uint32_t {
    kScoreSparseFamilyQSplit = 0,
    kScoreSparseFamilyFlash = 1,
};
enum CacheBlendTopKFamily : uint32_t {
    kCacheBlendTopKFamilyLegacy = 0,
    kCacheBlendTopKFamilyStage1024 = 1,
    kCacheBlendTopKFamilyStage2048 = 2,
};
enum SparseFlashScheduleVariant : uint32_t {
    kSparseFlashScheduleSinglePiece = 0,
    kSparseFlashScheduleRangeQ64 = 1,
    kSparseFlashScheduleRangeQ128 = 2,
};
enum TuneSelectionSource : uint32_t {
    kTuneSelectionSourceDefault = 0,
    kTuneSelectionSourceCache = 1,
    kTuneSelectionSourceOnlineTuned = 2,
    kTuneSelectionSourceBenchOverride = 3,
};
static thread_local int gSparseFlashVariantOverride = -1;
static thread_local bool gSparseFlashVariantTuneInProgress = false;
static thread_local int gSparseFlashScheduleOverride = -1;
static thread_local bool gSparseFlashScheduleTuneInProgress = false;
static bool _supportsAdrenoSparseFlashKVImage(OpenCLRuntime* runtime, int batch, int kvHeads, int headDim,
                                              int kvLen);
static bool _supportsAdrenoSparseFlashKImage(OpenCLRuntime* runtime, int batch, int kvHeads, int headDim,
                                             int kvLen);
static bool _preferAdrenoSparseFlashKImage(OpenCLRuntime* runtime, int batch, int kvHeads, int headDim,
                                           int kvLen);
static bool _allowAdrenoSparseFlashKVImageFallback(OpenCLRuntime* runtime, int batch, int kvHeads, int headDim,
                                                   int kvLen);
static bool _preferAdrenoScoreSparseFlash(OpenCLRuntime* runtime, int batch, int kvHeads, int headDim,
                                          int activeLen, int kvLen);
static bool _preferAdrenoLaterSparseQ4K8(OpenCLRuntime* runtime, int batch, int kvHeads, int headDim,
                                         int activeLen, int kvLen);
static bool _useAdrenoGemmTransNullLws(OpenCLRuntime* runtime, int seqLen, int kvLen, int batchHeads,
                                       std::vector<uint32_t>* lws);
static bool _useAdrenoGemmClipNullLws(OpenCLRuntime* runtime, int seqLen, int numHeads, int headDim,
                                      std::vector<uint32_t>* lws);

static bool _envFlagEnabled(const char* name, bool defaultValue) {
    const char* value = ::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    return value[0] != '0';
}

static const char* _tuneSelectionSourceName(uint32_t source) {
    switch (source) {
        case kTuneSelectionSourceCache:
            return "cache";
        case kTuneSelectionSourceOnlineTuned:
            return "online_tuned";
        case kTuneSelectionSourceBenchOverride:
            return "bench_override";
        case kTuneSelectionSourceDefault:
        default:
            return "default";
    }
}

static bool _legacyB863976OpenCL() {
    return _envFlagEnabled("MNN_PAGED_ATTENTION_OPENCL_LEGACY_B863976", false);
}

static std::string _openCLTuneDeviceKey(OpenCLRuntime* runtime) {
    if (runtime == nullptr) {
        return "unknown";
    }
    switch (runtime->getGpuType()) {
        case GpuType::MALI:
            return "mali";
        case GpuType::ADRENO:
            return _legacyB863976OpenCL() ? "adreno_legacy" : "adreno";
        case GpuType::RADEON:
            return "radeon";
        case GpuType::INTEL:
            return "intel";
        case GpuType::OTHER:
        default:
            return "other";
    }
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
    const int seqPack = ROUND_UP(seqLen, 32);
    const int qChunkPack = ROUND_UP(std::max(1, std::min(qChunkLen, seqLen)), 4);
    const int kvPack = ROUND_UP(kvLen, 32);
    const int headPack4 = ROUND_UP(headDim, 4);
    const int headPackV = ROUND_UP(headDim, 32);
    const size_t elemBytes = sizeof(float);
    size_t bytes = 0;
    bytes += static_cast<size_t>(seqPack) * headPack4 * numHeads * batch * elemBytes;
    bytes += static_cast<size_t>(kvPack) * headPack4 * kvHeads * batch * elemBytes;
    bytes += static_cast<size_t>(kvPack) * headPackV * kvHeads * batch * elemBytes;
    bytes += static_cast<size_t>(seqPack) * kvPack * batch * elemBytes;
    bytes += static_cast<size_t>(qChunkPack) * kvPack * numHeads * batch * elemBytes;
    bytes += static_cast<size_t>(qChunkPack) * kvPack * numHeads * batch * elemBytes;
    return bytes;
}

static bool _useDirectValuePrefillForSparse(const PagedKVMeta* meta, int activeLen, int headDim,
                                            OpenCLRuntime* runtime = nullptr) {
    if (meta == nullptr) {
        return false;
    }
    const bool adreno = runtime != nullptr && runtime->getGpuType() == GpuType::ADRENO && !_legacyB863976OpenCL();
    if (headDim >= 128 && !adreno) {
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

static bool _highBudgetSparsePlan(const PagedKVMeta* meta, int activeLen) {
    if (meta == nullptr) {
        return false;
    }
    constexpr int minActive = 384;
    if (activeLen < minActive) {
        return false;
    }
    int selected = 0;
    int picTokenCount = 0;
    if (meta->cacheblend_score_ready && meta->cacheblend_score_pic_token_count > 0) {
        selected = meta->cacheblend_score_selected_local_indices.empty()
            ? meta->cacheblend_score_top_k
            : static_cast<int>(meta->cacheblend_score_selected_local_indices.size());
        picTokenCount = meta->cacheblend_score_pic_token_count;
    } else if (meta->pic_graph_active_plan_ready && meta->pic_graph_pic_token_count > 0) {
        selected = static_cast<int>(meta->pic_graph_selected_local_indices.size());
        picTokenCount = meta->pic_graph_pic_token_count;
    } else {
        return false;
    }
    constexpr int minRatioPercent = 50;
    return selected > 0 && picTokenCount > 0 &&
           selected * 100 >= picTokenCount * minRatioPercent;
}

static int _sparseSelectedRatioPercent(const PagedKVMeta* meta) {
    if (meta == nullptr) {
        return 0;
    }
    int selected = 0;
    int picTokenCount = 0;
    if (meta->cacheblend_score_ready && meta->cacheblend_score_pic_token_count > 0) {
        selected = meta->cacheblend_score_selected_local_indices.empty()
            ? meta->cacheblend_score_top_k
            : static_cast<int>(meta->cacheblend_score_selected_local_indices.size());
        picTokenCount = meta->cacheblend_score_pic_token_count;
    } else if (meta->pic_graph_active_plan_ready && meta->pic_graph_pic_token_count > 0) {
        selected = static_cast<int>(meta->pic_graph_selected_local_indices.size());
        picTokenCount = meta->pic_graph_pic_token_count;
    }
    if (selected <= 0 || picTokenCount <= 0) {
        return 0;
    }
    return std::max(0, std::min(100, (selected * 100) / picTokenCount));
}

static bool _useFinerSparseQSplitPieces(const PagedKVMeta* meta, int activeLen) {
    return _highBudgetSparsePlan(meta, activeLen);
}

static int _sparseQSplitChunkLen(const PagedKVMeta* meta, int activeLen, int kvLen, int batch, int numHeads,
                                 int layerCount, bool staticWorkspace) {
    if (activeLen <= 0) {
        return 0;
    }
    if (gSparseQSplitChunkOverride > 0) {
        int overrideChunk = std::max(1, std::min(gSparseQSplitChunkOverride, activeLen));
        if (overrideChunk < activeLen) {
            overrideChunk = ((overrideChunk + 3) / 4) * 4;
        }
        return std::max(1, std::min(overrideChunk, activeLen));
    }
    int qChunkLen = staticWorkspace ? activeLen
                                    : _prefillQChunkLen(activeLen, kvLen, batch, numHeads, layerCount);
    constexpr int sparseQChunkLimit = 64;
    if (!staticWorkspace && sparseQChunkLimit > 0 && sparseQChunkLimit < qChunkLen) {
        qChunkLen = std::max(4, std::min(activeLen, ((sparseQChunkLimit + 3) / 4) * 4));
    }
    // HeadDim=128 q-split can become too coarse when static workspace keeps a whole 50% cacheblend budget
    // in one or two huge pieces. Keep the default path range-aware by capping the piece span modestly.
    if (staticWorkspace && _useFinerSparseQSplitPieces(meta, activeLen)) {
        constexpr int cacheblendStaticChunkCap = 192;
        qChunkLen = std::min(qChunkLen,
                             std::max(4, std::min(activeLen, ((cacheblendStaticChunkCap + 3) / 4) * 4)));
    }
    qChunkLen = std::max(1, std::min(qChunkLen, activeLen));
    if (qChunkLen < activeLen) {
        qChunkLen = ((qChunkLen + 3) / 4) * 4;
    }
    return std::max(1, std::min(qChunkLen, activeLen));
}

static bool _shouldTuneSparseQSplitChunk(const PagedKVMeta* meta, int activeLen, int headDim, bool staticWorkspace,
                                         bool profile, int tuneLevel) {
    if ((profile && !_profilePagedAttentionTune()) || meta == nullptr || !staticWorkspace) {
        return false;
    }
    if (headDim != 128 || activeLen < 384) {
        return false;
    }
    return tuneLevel == Heavy || tuneLevel == Wide;
}

static std::vector<int> _sparseQSplitChunkCandidates(int baseChunkLen, int activeLen) {
    std::vector<int> candidates;
    if (baseChunkLen <= 0 || activeLen <= 0) {
        return candidates;
    }
    auto append = [&](int candidate) {
        candidate = std::max(4, std::min(candidate, std::min(baseChunkLen, activeLen)));
        candidate = ((candidate + 3) / 4) * 4;
        candidate = std::max(4, std::min(candidate, std::min(baseChunkLen, activeLen)));
        if (std::find(candidates.begin(), candidates.end(), candidate) == candidates.end()) {
            candidates.emplace_back(candidate);
        }
    };
    append(baseChunkLen);
    append(64);
    append(96);
    append(128);
    append(160);
    append(192);
    std::sort(candidates.begin(), candidates.end());
    return candidates;
}

static uint32_t _sparseFlashLaneWidth(const PagedKVMeta* meta, int activeLen, int headDim) {
    if (headDim >= 128) {
        return 32u;
    }
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

static const char* _sparseFlashVariantName(uint32_t variant) {
    switch (variant) {
        case kSparseFlashVariantRow32:
            return "row32";
        case kSparseFlashVariantRow64:
            return "row64";
        case kSparseFlashVariantMQTileHD64Q4K16:
            return "mqtile_hd64_q4k16";
        case kSparseFlashVariantMQTileHD128Q4K16:
            return "mqtile_hd128_q4k16";
        case kSparseFlashVariantMQTileHD128Q4K8:
            return "mqtile_hd128_q4k8";
        case kSparseFlashVariantMQTileHD128Q8K16:
            return "mqtile_hd128_q8k16";
        case kSparseFlashVariantMQTileHD128Q4K8KImage:
            return "mqtile_hd128_q4k8_kimg";
        case kSparseFlashVariantMQTileHD128Q4K8KVImage:
            return "mqtile_hd128_q4k8_kvimg";
        case kSparseFlashVariantMQTileHD128Q8K16KVImage:
            return "mqtile_hd128_q8k16_kvimg";
        case kSparseFlashVariantMQTileHD128Q8K16KImage:
            return "mqtile_hd128_q8k16_kimg";
        default:
            return "unknown";
    }
}

static bool _parseSparseFlashVariantName(const char* name, uint32_t* variant) {
    if (name == nullptr || name[0] == '\0' || variant == nullptr) {
        return false;
    }
    const std::pair<const char*, uint32_t> variants[] = {
        {"row32", kSparseFlashVariantRow32},
        {"row64", kSparseFlashVariantRow64},
        {"mqtile_hd64_q4k16", kSparseFlashVariantMQTileHD64Q4K16},
        {"mqtile_hd128_q4k16", kSparseFlashVariantMQTileHD128Q4K16},
        {"mqtile_hd128_q4k8", kSparseFlashVariantMQTileHD128Q4K8},
        {"mqtile_hd128_q8k16", kSparseFlashVariantMQTileHD128Q8K16},
        {"mqtile_hd128_q4k8_kimg", kSparseFlashVariantMQTileHD128Q4K8KImage},
        {"mqtile_hd128_q4k8_kvimg", kSparseFlashVariantMQTileHD128Q4K8KVImage},
        {"mqtile_hd128_q8k16_kvimg", kSparseFlashVariantMQTileHD128Q8K16KVImage},
        {"mqtile_hd128_q8k16_kimg", kSparseFlashVariantMQTileHD128Q8K16KImage},
    };
    for (const auto& item : variants) {
        if (::strcmp(name, item.first) == 0) {
            *variant = item.second;
            return true;
        }
    }
    return false;
}

static bool _isSparseFlashKVImageVariant(uint32_t variant) {
    return variant == kSparseFlashVariantMQTileHD128Q4K8KVImage ||
           variant == kSparseFlashVariantMQTileHD128Q8K16KVImage;
}

static const char* _sparseFlashScheduleName(uint32_t schedule) {
    switch (schedule) {
        case kSparseFlashScheduleSinglePiece:
            return "single_piece";
        case kSparseFlashScheduleRangeQ64:
            return "range_q64";
        case kSparseFlashScheduleRangeQ128:
            return "range_q128";
        default:
            return "unknown";
    }
}

static bool _parseSparseFlashScheduleName(const char* name, uint32_t* schedule) {
    if (name == nullptr || name[0] == '\0' || schedule == nullptr) {
        return false;
    }
    const std::pair<const char*, uint32_t> schedules[] = {
        {"single_piece", kSparseFlashScheduleSinglePiece},
        {"range_q64", kSparseFlashScheduleRangeQ64},
        {"range_q128", kSparseFlashScheduleRangeQ128},
    };
    for (const auto& item : schedules) {
        if (::strcmp(name, item.first) == 0) {
            *schedule = item.second;
            return true;
        }
    }
    return false;
}

static bool _benchSparseFlashVariantOverride(uint32_t* variant) {
    return _parseSparseFlashVariantName(::getenv("MNN_BENCH_OPENCL_PAGED_SPARSE_FORCE_VARIANT"), variant) ||
           _parseSparseFlashVariantName(::getenv("MNN_PAGED_ATTENTION_BENCH_FORCE_SPARSE_FLASH_VARIANT"), variant);
}

static bool _benchSparseFlashScheduleOverride(uint32_t* schedule) {
    return _parseSparseFlashScheduleName(::getenv("MNN_BENCH_OPENCL_PAGED_SPARSE_FORCE_SCHEDULE"), schedule) ||
           _parseSparseFlashScheduleName(::getenv("MNN_PAGED_ATTENTION_BENCH_FORCE_SPARSE_FLASH_SCHEDULE"),
                                         schedule);
}

static bool _preferMaliLaterSparseQ8K16(OpenCLRuntime* runtime, int headDim, int kvLen, bool queryRowsAreFull) {
    return PagedAttentionMali::preferLaterSparseQ8K16(runtime, headDim, kvLen, queryRowsAreFull);
}

static bool _maliHeadDim128SparseVariantSupported(uint32_t variant, int headDim, OpenCLRuntime* runtime, int kvLen) {
    return PagedAttentionMali::headDim128SparseVariantSupported(
        runtime, headDim, kvLen, variant, kSparseFlashVariantRow32, kSparseFlashVariantMQTileHD128Q8K16);
}

static bool _sparseFlashVariantSupported(uint32_t variant, int headDim, OpenCLRuntime* runtime = nullptr,
                                         int batch = 0, int kvHeads = 0, int kvLen = 0,
                                         bool allowMaliExperimentalVariant = false) {
    if (!allowMaliExperimentalVariant &&
        runtime != nullptr && runtime->getGpuType() == GpuType::MALI && headDim == 128) {
        // Keep Mali routing separate from Adreno image/mqtile logic. Qwen3-4B
        // ctx1536/2048 validates q8k16 for later sparse layers; larger contexts
        // remain on the smaller row32 kernel unless explicitly bench-forced.
        return _maliHeadDim128SparseVariantSupported(variant, headDim, runtime, kvLen);
    }
    switch (variant) {
        case kSparseFlashVariantRow32:
        case kSparseFlashVariantRow64:
            return headDim == 64 || headDim == 128;
        case kSparseFlashVariantMQTileHD64Q4K16:
            return headDim == 64;
        case kSparseFlashVariantMQTileHD128Q4K16:
        case kSparseFlashVariantMQTileHD128Q4K8:
        case kSparseFlashVariantMQTileHD128Q8K16:
            return headDim == 128;
        case kSparseFlashVariantMQTileHD128Q4K8KImage:
            return headDim == 128 &&
                _supportsAdrenoSparseFlashKImage(runtime, batch, kvHeads, headDim, kvLen);
        case kSparseFlashVariantMQTileHD128Q8K16KImage:
            return headDim == 128 &&
                _supportsAdrenoSparseFlashKImage(runtime, batch, kvHeads, headDim, kvLen);
        case kSparseFlashVariantMQTileHD128Q4K8KVImage:
        case kSparseFlashVariantMQTileHD128Q8K16KVImage:
            return headDim == 128 &&
                _supportsAdrenoSparseFlashKVImage(runtime, batch, kvHeads, headDim, kvLen);
        default:
            return false;
    }
}

static std::vector<uint32_t> _sparseFlashVariantCandidates(int headDim, OpenCLRuntime* runtime = nullptr,
                                                           int batch = 0, int kvHeads = 0, int kvLen = 0,
                                                           bool queryRowsAreFull = false) {
    if (headDim == 64) {
        return {kSparseFlashVariantRow32, kSparseFlashVariantRow64, kSparseFlashVariantMQTileHD64Q4K16};
    }
    if (headDim == 128) {
        if (runtime != nullptr && runtime->getGpuType() == GpuType::MALI) {
            if (_preferMaliLaterSparseQ8K16(runtime, headDim, kvLen, queryRowsAreFull)) {
                return {kSparseFlashVariantMQTileHD128Q8K16};
            }
            return {kSparseFlashVariantRow32};
        }
        const bool adrenoScoreFlash = queryRowsAreFull && runtime != nullptr && runtime->getGpuType() == GpuType::ADRENO;
        if (adrenoScoreFlash) {
            std::vector<uint32_t> candidates = {
                kSparseFlashVariantRow64,
                kSparseFlashVariantMQTileHD128Q8K16,
            };
            if (_preferAdrenoSparseFlashKImage(runtime, batch, kvHeads, headDim, kvLen)) {
                candidates.emplace_back(kSparseFlashVariantMQTileHD128Q4K8KImage);
                candidates.emplace_back(kSparseFlashVariantMQTileHD128Q8K16KImage);
            } else if (_allowAdrenoSparseFlashKVImageFallback(runtime, batch, kvHeads, headDim, kvLen)) {
                candidates.emplace_back(kSparseFlashVariantMQTileHD128Q8K16KVImage);
            }
            return candidates;
        }
        std::vector<uint32_t> candidates = {
            kSparseFlashVariantRow32,
            kSparseFlashVariantRow64,
            kSparseFlashVariantMQTileHD128Q4K16,
            kSparseFlashVariantMQTileHD128Q4K8,
            kSparseFlashVariantMQTileHD128Q8K16,
        };
        if (_preferAdrenoSparseFlashKImage(runtime, batch, kvHeads, headDim, kvLen)) {
            candidates.emplace_back(kSparseFlashVariantMQTileHD128Q4K8KImage);
            candidates.emplace_back(kSparseFlashVariantMQTileHD128Q8K16KImage);
        } else if (_allowAdrenoSparseFlashKVImageFallback(runtime, batch, kvHeads, headDim, kvLen)) {
            candidates.emplace_back(kSparseFlashVariantMQTileHD128Q4K8KVImage);
            candidates.emplace_back(kSparseFlashVariantMQTileHD128Q8K16KVImage);
        }
        return candidates;
    }
    return {};
}

static bool _sparseFlashScheduleSupported(const PagedKVMeta* meta, uint32_t schedule, bool queryRowsAreFull) {
    switch (schedule) {
        case kSparseFlashScheduleSinglePiece:
            return meta != nullptr && meta->cacheblend_score_ready && queryRowsAreFull;
        case kSparseFlashScheduleRangeQ64:
        case kSparseFlashScheduleRangeQ128:
            return true;
        default:
            return false;
    }
}

static uint32_t _defaultSparseFlashSchedule(const PagedKVMeta* meta, bool queryRowsAreFull) {
    uint32_t forcedSchedule = 0;
    if (!queryRowsAreFull &&
        _benchSparseFlashScheduleOverride(&forcedSchedule) &&
        _sparseFlashScheduleSupported(meta, forcedSchedule, queryRowsAreFull)) {
        return forcedSchedule;
    }
    if (!queryRowsAreFull &&
        gSparseFlashScheduleOverride >= 0 &&
        _sparseFlashScheduleSupported(meta, static_cast<uint32_t>(gSparseFlashScheduleOverride), queryRowsAreFull)) {
        return static_cast<uint32_t>(gSparseFlashScheduleOverride);
    }
    if (meta != nullptr && meta->cacheblend_score_ready && queryRowsAreFull) {
        return kSparseFlashScheduleSinglePiece;
    }
    return kSparseFlashScheduleRangeQ64;
}

static std::vector<uint32_t> _sparseFlashScheduleCandidates(const PagedKVMeta* meta, int activeLen,
                                                            bool queryRowsAreFull) {
    std::vector<uint32_t> candidates;
    if (activeLen <= 0) {
        return candidates;
    }
    if (meta != nullptr && meta->cacheblend_score_ready && queryRowsAreFull) {
        candidates.emplace_back(kSparseFlashScheduleSinglePiece);
    }
    candidates.emplace_back(kSparseFlashScheduleRangeQ64);
    if (activeLen > 64) {
        candidates.emplace_back(kSparseFlashScheduleRangeQ128);
    }
    return candidates;
}

static bool _shouldTuneSparseFlashSchedule(const PagedKVMeta* meta, int activeLen, bool profile, int tuneLevel) {
    if ((profile && !_profilePagedAttentionTune()) || meta == nullptr || activeLen < 128) {
        return false;
    }
    return tuneLevel == Heavy || tuneLevel == Wide;
}

static int _sparseFlashScheduleQChunkLen(uint32_t schedule, int activeLen) {
    if (activeLen <= 0) {
        return 0;
    }
    int qChunkLen = activeLen;
    switch (schedule) {
        case kSparseFlashScheduleSinglePiece:
            return activeLen;
        case kSparseFlashScheduleRangeQ128:
            qChunkLen = 128;
            break;
        case kSparseFlashScheduleRangeQ64:
        default:
            qChunkLen = 64;
            break;
    }
    qChunkLen = std::max(1, std::min(qChunkLen, activeLen));
    if (qChunkLen < activeLen) {
        qChunkLen = std::max(4, ((qChunkLen + 3) / 4) * 4);
    }
    return std::max(1, std::min(qChunkLen, activeLen));
}

static uint32_t _defaultSparseFlashVariant(const PagedKVMeta* meta, int activeLen, int headDim,
                                           OpenCLRuntime* runtime = nullptr, bool queryRowsAreFull = false,
                                           int batch = 0, int kvHeads = 0, int kvLen = 0) {
    uint32_t forcedVariant = 0;
    if (!queryRowsAreFull &&
        _benchSparseFlashVariantOverride(&forcedVariant) &&
        _sparseFlashVariantSupported(forcedVariant, headDim, runtime, batch, kvHeads, kvLen,
                                     true /* allowMaliExperimentalVariant */)) {
        return forcedVariant;
    }
    if (!queryRowsAreFull &&
        gSparseFlashVariantOverride >= 0 &&
        _sparseFlashVariantSupported(static_cast<uint32_t>(gSparseFlashVariantOverride), headDim, runtime, batch,
                                     kvHeads, kvLen)) {
        return static_cast<uint32_t>(gSparseFlashVariantOverride);
    }
    if (headDim == 128) {
        if (runtime != nullptr && runtime->getGpuType() == GpuType::MALI) {
            if (_preferMaliLaterSparseQ8K16(runtime, headDim, kvLen, queryRowsAreFull)) {
                return kSparseFlashVariantMQTileHD128Q8K16;
            }
            return kSparseFlashVariantRow32;
        }
        if (queryRowsAreFull && runtime != nullptr && runtime->getGpuType() == GpuType::ADRENO) {
            // Adreno score-layer sparse attention is QK-dominant. Prefer the mixed path that keeps
            // V in buffer while serving K through the texture cache when the packed key image fits.
            if (_preferAdrenoSparseFlashKImage(runtime, batch, kvHeads, headDim, kvLen)) {
                return kSparseFlashVariantMQTileHD128Q8K16KImage;
            }
            if (_allowAdrenoSparseFlashKVImageFallback(runtime, batch, kvHeads, headDim, kvLen)) {
                return kSparseFlashVariantMQTileHD128Q8K16KVImage;
            }
            return kSparseFlashVariantMQTileHD128Q8K16;
        }
        if (_preferAdrenoLaterSparseQ4K8(runtime, batch, kvHeads, headDim, activeLen, kvLen)) {
            return kSparseFlashVariantMQTileHD128Q4K8;
        }
        if (runtime != nullptr && runtime->getGpuType() == GpuType::ADRENO &&
            _useDirectValuePrefillForSparse(meta, activeLen, headDim, runtime)) {
            // High-budget Adreno sparse layers can keep V on the identity-mapped paged buffer while
            // mirroring only K into an image. That avoids tempV pack/copy and still gives the QK loop
            // texture-cache-friendly K reads.
            if (_preferAdrenoSparseFlashKImage(runtime, batch, kvHeads, headDim, kvLen)) {
                return kSparseFlashVariantMQTileHD128Q8K16KImage;
            }
            if (_allowAdrenoSparseFlashKVImageFallback(runtime, batch, kvHeads, headDim, kvLen)) {
                return kSparseFlashVariantMQTileHD128Q8K16KVImage;
            }
            return kSparseFlashVariantMQTileHD128Q8K16;
        }
        return kSparseFlashVariantMQTileHD128Q4K16;
    }
    return _sparseFlashLaneWidth(meta, activeLen, headDim) == 32u ? kSparseFlashVariantRow32
                                                                   : kSparseFlashVariantRow64;
}

static bool _shouldTuneSparseFlashVariant(const PagedKVMeta* meta, int activeLen, int headDim, bool profile,
                                          int tuneLevel) {
    if ((profile && !_profilePagedAttentionTune()) || meta == nullptr || activeLen < 128) {
        return false;
    }
    if (headDim != 64 && headDim != 128) {
        return false;
    }
    return tuneLevel == Heavy || tuneLevel == Wide;
}

static uint64_t _sparseLogicalWork(const PagedKVMeta* meta, int activeLen, int kvLen);

static uint32_t _sparseLogicalWorkPermille(const PagedKVMeta* meta, int activeLen, int kvLen) {
    if (activeLen <= 0 || kvLen <= 0) {
        return 0;
    }
    const uint64_t work = _sparseLogicalWork(meta, activeLen, kvLen);
    if (work == 0) {
        return 0;
    }
    const uint64_t full = static_cast<uint64_t>(activeLen) * static_cast<uint64_t>(kvLen);
    if (full == 0) {
        return 0;
    }
    return static_cast<uint32_t>(std::min<uint64_t>(1000, (work * 1000u) / full));
}

static uint64_t _sparseLogicalWork(const PagedKVMeta* meta, int activeLen, int kvLen) {
    if (meta == nullptr || activeLen <= 0 || kvLen <= 0 ||
        static_cast<int>(meta->sparse_query_logical_indices.size()) < activeLen) {
        return 0;
    }
    uint64_t work = 0;
    for (int i = 0; i < activeLen; ++i) {
        const int logical = meta->sparse_query_logical_indices[static_cast<size_t>(i)];
        work += static_cast<uint64_t>(std::max(0, std::min(kvLen, logical + 1)));
    }
    return work;
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

static int _adrenoGemmPrefillQSplitNum(int seqLen, int kvLen, int batch, int numHeads) {
    constexpr int alignQ = 32;
    constexpr int alignKV = 32;
    if (seqLen <= 0 || kvLen <= 0 || batch <= 0 || numHeads <= 0) {
        return 1;
    }
    const int seqPack = ROUND_UP(seqLen, alignQ);
    const int kvPack = ROUND_UP(kvLen, alignKV);
    float useMemorySize = 1.0f * seqPack / 1024.0f * kvPack / 1024.0f * batch * numHeads;
    int split = 1;
    if (useMemorySize > 32.0f) {
        split = useMemorySize >= 256.0f ? 8 : ((useMemorySize < 128.0f) ? 2 : 4);
    }
    split = std::max(1, std::min(split, std::max(1, seqPack / alignQ)));
    while (split > 1 && ((seqPack / split) % alignQ) != 0) {
        split >>= 1;
    }
    return std::max(1, split);
}

static bool _useAdrenoGemmFullPrefill(OpenCLRuntime* runtime, int seqLen, int kvLen, int batch, int numHeads,
                                      int kvHeads, int headDim, int maskKeyLen) {
    return PagedAttentionAdreno::useAdrenoGemmFullPrefill(runtime, _legacyB863976OpenCL(), seqLen, kvLen, batch,
                                                          numHeads, kvHeads, headDim, maskKeyLen);
}

static bool _useAdrenoSourceSlotValueHydrate(OpenCLRuntime* runtime) {
    return PagedAttentionAdreno::useAdrenoSourceSlotValueHydrate(runtime, _legacyB863976OpenCL());
}

static bool _useAdrenoCacheBlendValueImage(OpenCLRuntime* runtime, int batch, int kvHeads, int tokenCount,
                                           int headDim) {
    return PagedAttentionAdreno::useAdrenoCacheBlendValueImage(runtime, _legacyB863976OpenCL(), batch, kvHeads,
                                                               tokenCount, headDim);
}

static bool _computeAdrenoCacheBlendValueImageShape(OpenCLRuntime* runtime, int batch, int kvHeads,
                                                    int tokenCount, int headDim,
                                                    int* imageWidth, int* imageHeight) {
    return PagedAttentionAdreno::computeAdrenoCacheBlendValueImageShape(runtime, _legacyB863976OpenCL(), batch,
                                                                        kvHeads, tokenCount, headDim,
                                                                        imageWidth, imageHeight);
}

static bool _supportsAdrenoSparseFlashKVImage(OpenCLRuntime* runtime, int batch, int kvHeads, int headDim,
                                              int kvLen) {
    return PagedAttentionAdreno::supportsAdrenoSparseFlashKVImage(runtime, _legacyB863976OpenCL(), batch, kvHeads,
                                                                  headDim, kvLen);
}

static bool _supportsAdrenoSparseFlashKImage(OpenCLRuntime* runtime, int batch, int kvHeads, int headDim,
                                             int kvLen) {
    return PagedAttentionAdreno::supportsAdrenoSparseFlashKImage(runtime, _legacyB863976OpenCL(), batch, kvHeads,
                                                                 headDim, kvLen);
}

static bool _computeAdrenoSparseFlashKImageShape(OpenCLRuntime* runtime, int batch, int kvHeads, int headDim,
                                                 int kvLen, int* imageWidth, int* imageHeight) {
    return PagedAttentionAdreno::computeAdrenoSparseFlashKImageShape(runtime, _legacyB863976OpenCL(), batch,
                                                                     kvHeads, headDim, kvLen,
                                                                     imageWidth, imageHeight);
}

static bool _computeAdrenoSparseFlashKVImageShapes(OpenCLRuntime* runtime, int batch, int kvHeads, int headDim,
                                                   int kvLen,
                                                   int* keyImageWidth, int* keyImageHeight,
                                                   int* valueImageWidth, int* valueImageHeight) {
    return PagedAttentionAdreno::computeAdrenoSparseFlashKVImageShapes(runtime, _legacyB863976OpenCL(), batch,
                                                                       kvHeads, headDim, kvLen,
                                                                       keyImageWidth, keyImageHeight,
                                                                       valueImageWidth, valueImageHeight);
}

static bool _preferAdrenoSparseFlashKImage(OpenCLRuntime* runtime, int batch, int kvHeads, int headDim,
                                           int kvLen) {
    return PagedAttentionAdreno::preferAdrenoSparseFlashKImage(runtime, _legacyB863976OpenCL(), batch, kvHeads,
                                                               headDim, kvLen);
}

static bool _allowAdrenoSparseFlashKVImageFallback(OpenCLRuntime* runtime, int batch, int kvHeads, int headDim,
                                                   int kvLen) {
    return PagedAttentionAdreno::allowAdrenoSparseFlashKVImageFallback(runtime, _legacyB863976OpenCL(), batch,
                                                                       kvHeads, headDim, kvLen);
}

static bool _preferAdrenoScoreSparseFlash(OpenCLRuntime* runtime, int batch, int kvHeads, int headDim,
                                          int activeLen, int kvLen) {
    return PagedAttentionAdreno::preferAdrenoScoreSparseFlash(runtime, _legacyB863976OpenCL(), batch, kvHeads,
                                                              headDim, activeLen, kvLen);
}

static bool _preferAdrenoLaterSparseQ4K8(OpenCLRuntime* runtime, int batch, int kvHeads, int headDim,
                                         int activeLen, int kvLen) {
    return PagedAttentionAdreno::preferAdrenoLaterSparseQ4K8(runtime, _legacyB863976OpenCL(), batch, kvHeads,
                                                             headDim, activeLen, kvLen);
}

static bool _useAdrenoGemmTransNullLws(OpenCLRuntime* runtime, int seqLen, int kvLen, int batchHeads,
                                       std::vector<uint32_t>* lws) {
    return PagedAttentionAdreno::useAdrenoGemmTransNullLws(runtime, _legacyB863976OpenCL(), seqLen, kvLen,
                                                           batchHeads, lws);
}

static bool _useAdrenoGemmClipNullLws(OpenCLRuntime* runtime, int seqLen, int numHeads, int headDim,
                                      std::vector<uint32_t>* lws) {
    return PagedAttentionAdreno::useAdrenoGemmClipNullLws(runtime, _legacyB863976OpenCL(), seqLen, numHeads,
                                                          headDim, lws);
}

static bool _rejectAdrenoSparseFlashVariantFromCache(uint32_t variant, OpenCLRuntime* runtime, int batch,
                                                     int kvHeads, int headDim, int activeLen, int kvLen,
                                                     bool queryRowsAreFull) {
    return PagedAttentionAdreno::rejectSparseFlashVariantFromCache(
        runtime, _legacyB863976OpenCL(), batch, kvHeads, headDim, activeLen, kvLen, queryRowsAreFull,
        variant, _isSparseFlashKVImageVariant(variant), kSparseFlashVariantMQTileHD128Q4K8);
}

static bool _rejectMaliSparseFlashVariantFromCache(uint32_t variant, OpenCLRuntime* runtime, int headDim,
                                                   int kvLen, bool queryRowsAreFull) {
    return PagedAttentionMali::rejectSparseFlashVariantFromCache(
        runtime, headDim, kvLen, queryRowsAreFull, variant,
        kSparseFlashVariantRow32, kSparseFlashVariantMQTileHD128Q8K16);
}

static bool _rejectAdrenoSparseFlashScheduleFromCache(uint32_t schedule, OpenCLRuntime* runtime, int batch,
                                                      int kvHeads, int headDim, int activeLen, int kvLen,
                                                      bool queryRowsAreFull) {
    return PagedAttentionAdreno::rejectSparseFlashScheduleFromCache(
        runtime, _legacyB863976OpenCL(), batch, kvHeads, headDim, activeLen, kvLen, queryRowsAreFull,
        schedule, kSparseFlashScheduleRangeQ128);
}

static bool _disableAdrenoStaticSparseFlashWorkspace(OpenCLRuntime* runtime, int numHeads, int kvHeads, int headDim,
                                                     int activeLen, int kvLen, bool queryRowsAreFull) {
    return PagedAttentionAdreno::disableStaticSparseFlashWorkspace(
        runtime, _legacyB863976OpenCL(), numHeads, kvHeads, headDim, activeLen, kvLen, queryRowsAreFull);
}

static void _appendGemmBuildOptions(std::set<std::string>& buildOptions, const std::vector<uint32_t>& param,
                                    uint32_t layout, bool adreno) {
    int KWG = param[0], KWI = param[1], MDIMA = param[2], MDIMC = param[3], MWG = param[4], NDIMB = param[5];
    int NDIMC = param[6], NWG = param[7], SA = param[8], SB = param[9], STRM = param[10], STRN = param[11];
    int VWM = param[12], VWN = param[13];
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
    if (layout >= 4) {
        buildOptions.emplace("-DOUTPUTMN");
    }
    if (adreno) {
        buildOptions.emplace("-DUSE_CL_MAD=1");
        buildOptions.emplace("-DRELAX_WORKGROUP_SIZE=1");
    }
}

} // namespace

ErrorCode PagedAttentionBufExecution::ensureSparseFlashKernel() {
    if ((mHeadDim != 64 && mHeadDim != 128) || mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    const int groupSize = mNumHead / mKvNumHead;
    if (mSparseFlashKernel32 && mSparseFlashKernel64 &&
        mSparseFlashKernelMQTileHD64Q4K16 && mSparseFlashKernelMQTileHD128Q4K16 &&
        mSparseFlashKernelMQTileHD128Q4K8 && mSparseFlashKernelMQTileHD128Q4K8KImage &&
        mSparseFlashKernelMQTileHD128Q4K8KVImage &&
        mSparseFlashKernelMQTileHD128Q8K16 && mSparseFlashKernelMQTileHD128Q8K16KImage &&
        mSparseFlashKernelMQTileHD128Q8K16KVImage &&
        mSparseFlashKernelGroupSize == groupSize) {
        return NO_ERROR;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    // row32/row64 refers to the lane width of the legacy sparse flash workgroup.
    // mqtile_* carries the explicit Q/K tile suffix so variant tuning can treat
    // tile shape as part of the stable kernel identity.
    mSparseFlashKernel32 = runtime->buildKernel("attention_buf", "sparse_flash_attention_row32",
                                                {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                                                mOpenCLBackend->getPrecision());
    mSparseFlashKernel64 = runtime->buildKernel("attention_buf", "sparse_flash_attention_row64",
                                                {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                                                mOpenCLBackend->getPrecision());
    mSparseFlashKernelMQTileHD64Q4K16 = runtime->buildKernel("attention_buf", "mqtile_sparse_flash_hd64_q4k16",
                                                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                                                             mOpenCLBackend->getPrecision());
    mSparseFlashKernelMQTileHD128Q4K16 =
        runtime->buildKernel("attention_buf", "mqtile_sparse_flash_hd128_q4k16",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    mSparseFlashKernelMQTileHD128Q4K8 =
        runtime->buildKernel("attention_buf", "mqtile_sparse_flash_hd128_q4k8",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    mSparseFlashKernelMQTileHD128Q4K8KImage =
        runtime->buildKernel("attention_buf", "mqtile_sparse_flash_hd128_q4k8_kimg",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    mSparseFlashKernelMQTileHD128Q4K8KVImage =
        runtime->buildKernel("attention_buf", "mqtile_sparse_flash_hd128_q4k8_kvimg",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    mSparseFlashKernelMQTileHD128Q8K16 =
        runtime->buildKernel("attention_buf", "mqtile_sparse_flash_hd128_q8k16",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    mSparseFlashKernelMQTileHD128Q8K16KImage =
        runtime->buildKernel("attention_buf", "mqtile_sparse_flash_hd128_q8k16_kimg",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    mSparseFlashKernelMQTileHD128Q8K16KVImage =
        runtime->buildKernel("attention_buf", "mqtile_sparse_flash_hd128_q8k16_kvimg",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL(mSparseFlashKernel32);
    OPENCL_CHECK_KERNEL(mSparseFlashKernel64);
    OPENCL_CHECK_KERNEL(mSparseFlashKernelMQTileHD64Q4K16);
    OPENCL_CHECK_KERNEL(mSparseFlashKernelMQTileHD128Q4K16);
    OPENCL_CHECK_KERNEL(mSparseFlashKernelMQTileHD128Q4K8);
    OPENCL_CHECK_KERNEL(mSparseFlashKernelMQTileHD128Q4K8KImage);
    OPENCL_CHECK_KERNEL(mSparseFlashKernelMQTileHD128Q4K8KVImage);
    OPENCL_CHECK_KERNEL(mSparseFlashKernelMQTileHD128Q8K16);
    OPENCL_CHECK_KERNEL(mSparseFlashKernelMQTileHD128Q8K16KImage);
    OPENCL_CHECK_KERNEL(mSparseFlashKernelMQTileHD128Q8K16KVImage);
    mSparseFlashKernelGroupSize = groupSize;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureAdrenoSparseFlashPackedKeyImage(int kvPack) {
    if (kvPack <= 0 || mBatch <= 0 || mKvNumHead <= 0 || mHeadDim != 128) {
        return INVALID_VALUE;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (!_supportsAdrenoSparseFlashKImage(runtime, mBatch, mKvNumHead, mHeadDim, kvPack)) {
        return NOT_SUPPORT;
    }
    const bool useFp32 = mBytes == static_cast<int>(sizeof(float));
    if (!mCopyBufferToImageLinearKernel || mCopyBufferToImageLinearUseFp32 != useFp32) {
        std::set<std::string> buildOptions;
        if (useFp32) {
            buildOptions.emplace("-DBUFFER_INP_FP32");
        }
        mCopyBufferToImageLinearKernel = runtime->buildKernel("copy_buffer_to_image2d", "copy_buffer_to_image2d",
                                                              buildOptions, mOpenCLBackend->getPrecision());
        if (mCopyBufferToImageLinearKernel == nullptr) {
            return NOT_SUPPORT;
        }
        mCopyBufferToImageLinearUseFp32 = useFp32;
    }
    int keyImageWidth = 0;
    int keyImageHeight = 0;
    if (!_computeAdrenoSparseFlashKImageShape(runtime, mBatch, mKvNumHead, mHeadDim, kvPack,
                                              &keyImageWidth, &keyImageHeight)) {
        return NOT_SUPPORT;
    }
    if (mSparseFlashPackedKeyImage != nullptr && mSparseFlashPackedKVLen >= kvPack &&
        mSparseFlashPackedKeyImageWidth == keyImageWidth &&
        mSparseFlashPackedKeyImageHeight == keyImageHeight &&
        mSparseFlashPackedImageBytes == mBytes) {
        return NO_ERROR;
    }
    if (!mAdrenoImagePool) {
        mAdrenoImagePool.reset(new ImagePool(runtime->context()));
    }
    auto keyImage = mAdrenoImagePool->alloc(keyImageWidth, keyImageHeight, mOpenCLBackend->fpType());
    if (keyImage == nullptr) {
        return OUT_OF_MEMORY;
    }
    mSparseFlashPackedKeyImage = keyImage;
    mSparseFlashPackedKVLen = kvPack;
    mSparseFlashPackedKeyImageWidth = keyImageWidth;
    mSparseFlashPackedKeyImageHeight = keyImageHeight;
    mSparseFlashPackedImageBytes = mBytes;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureAdrenoSparseFlashPackedKVImages(int kvPack) {
    if (kvPack <= 0 || mBatch <= 0 || mKvNumHead <= 0 || mHeadDim != 128) {
        return INVALID_VALUE;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (!_supportsAdrenoSparseFlashKVImage(runtime, mBatch, mKvNumHead, mHeadDim, kvPack)) {
        return NOT_SUPPORT;
    }
    const bool useFp32 = mBytes == static_cast<int>(sizeof(float));
    if (!mCopyBufferToImageLinearKernel || mCopyBufferToImageLinearUseFp32 != useFp32) {
        std::set<std::string> buildOptions;
        if (useFp32) {
            buildOptions.emplace("-DBUFFER_INP_FP32");
        }
        mCopyBufferToImageLinearKernel = runtime->buildKernel("copy_buffer_to_image2d", "copy_buffer_to_image2d",
                                                              buildOptions, mOpenCLBackend->getPrecision());
        if (mCopyBufferToImageLinearKernel == nullptr) {
            return NOT_SUPPORT;
        }
        mCopyBufferToImageLinearUseFp32 = useFp32;
    }
    int keyImageWidth = 0;
    int keyImageHeight = 0;
    int valueImageWidth = 0;
    int valueImageHeight = 0;
    if (!_computeAdrenoSparseFlashKVImageShapes(runtime, mBatch, mKvNumHead, mHeadDim, kvPack,
                                                &keyImageWidth, &keyImageHeight,
                                                &valueImageWidth, &valueImageHeight)) {
        return NOT_SUPPORT;
    }
    if (mSparseFlashPackedKeyImage != nullptr && mSparseFlashPackedValueImage != nullptr &&
        mSparseFlashPackedKVLen >= kvPack &&
        mSparseFlashPackedKeyImageWidth == keyImageWidth &&
        mSparseFlashPackedKeyImageHeight == keyImageHeight &&
        mSparseFlashPackedValueImageWidth == valueImageWidth &&
        mSparseFlashPackedValueImageHeight == valueImageHeight &&
        mSparseFlashPackedImageBytes == mBytes) {
        return NO_ERROR;
    }
    if (!mAdrenoImagePool) {
        mAdrenoImagePool.reset(new ImagePool(runtime->context()));
    }
    auto keyImage = mAdrenoImagePool->alloc(keyImageWidth, keyImageHeight, mOpenCLBackend->fpType());
    auto valueImage = mAdrenoImagePool->alloc(valueImageWidth, valueImageHeight, mOpenCLBackend->fpType());
    if (keyImage == nullptr || valueImage == nullptr) {
        return OUT_OF_MEMORY;
    }
    mSparseFlashPackedKeyImage = keyImage;
    mSparseFlashPackedValueImage = valueImage;
    mSparseFlashPackedKVLen = kvPack;
    mSparseFlashPackedKeyImageWidth = keyImageWidth;
    mSparseFlashPackedKeyImageHeight = keyImageHeight;
    mSparseFlashPackedValueImageWidth = valueImageWidth;
    mSparseFlashPackedValueImageHeight = valueImageHeight;
    mSparseFlashPackedImageBytes = mBytes;
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
        const int seqPack = ROUND_UP(seqLen, 32);
        const int qChunkPack = ROUND_UP(qChunkLen, 4);
        const int kvPack = ROUND_UP(kvLen, 32);
        const int headPack4 = ROUND_UP(mHeadDim, 4);
        const int headPackV = ROUND_UP(mHeadDim, 32);
        mTempQ.reset(Tensor::createDevice<float>({seqPack * headPack4 * mNumHead * mBatch}));
        mTempK.reset(Tensor::createDevice<float>({kvPack * headPack4 * mKvNumHead * mBatch}));
        mTempV.reset(Tensor::createDevice<float>({kvPack * headPackV * mKvNumHead * mBatch}));
        mTempMask.reset(Tensor::createDevice<float>({seqPack * kvPack * mBatch}));
        mTempQK.reset(Tensor::createDevice<float>({qChunkPack * kvPack * mNumHead * mBatch}));
        mTempSoftmax.reset(Tensor::createDevice<float>({qChunkPack * kvPack * mNumHead * mBatch}));
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

ErrorCode PagedAttentionBufExecution::ensureSparseFlashTemps(int seqLen, int kvLen, bool staticWorkspace) {
    if (seqLen <= 0 || kvLen <= 0) {
        return INVALID_VALUE;
    }
    const int seqPack = ROUND_UP(seqLen, 32);
    const int kvPack = ROUND_UP(kvLen, 32);
    const int headPack4 = ROUND_UP(mHeadDim, 4);
    const int headPackV = ROUND_UP(mHeadDim, 32);
    if (!(mTempQ && mTempK && mTempV &&
          mFastSeqLen == seqLen && mFastKvLen == kvLen && mFastQChunkLen == 0 &&
          mFastStaticWorkspace == staticWorkspace)) {
        mTempQ.reset(Tensor::createDevice<float>({seqPack * headPack4 * mNumHead * mBatch}));
        mTempK.reset(Tensor::createDevice<float>({kvPack * headPack4 * mKvNumHead * mBatch}));
        mTempV.reset(Tensor::createDevice<float>({kvPack * headPackV * mKvNumHead * mBatch}));
        // Sparse flash does not consume legacy qk/softmax scratch. Keeping those
        // static per layer can exhaust OrangePi memory at Qwen3-8B 2K+ high budgets.
        mTempMask.reset();
        mTempQK.reset();
        mTempSoftmax.reset();
        if (!mTempQ || !mTempK || !mTempV) {
            return OUT_OF_MEMORY;
        }
        mFastSeqLen = seqLen;
        mFastKvLen = kvLen;
        mFastQChunkLen = 0;
        mFastStaticWorkspace = staticWorkspace;
        if (staticWorkspace) {
            OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempQ.get(), Backend::STATIC));
            OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempK.get(), Backend::STATIC));
            OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempV.get(), Backend::STATIC));
        }
    }
    if (staticWorkspace) {
        return NO_ERROR;
    }
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempQ.get(), Backend::DYNAMIC_IN_EXECUTION));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempK.get(), Backend::DYNAMIC_IN_EXECUTION));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempV.get(), Backend::DYNAMIC_IN_EXECUTION));
    mOpenCLBackend->onReleaseBuffer(mTempQ.get(), Backend::DYNAMIC_IN_EXECUTION);
    mOpenCLBackend->onReleaseBuffer(mTempK.get(), Backend::DYNAMIC_IN_EXECUTION);
    mOpenCLBackend->onReleaseBuffer(mTempV.get(), Backend::DYNAMIC_IN_EXECUTION);
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureAdrenoGemmPrefillTemps(int seqLen, int kvLen, int qSplitNum) {
    if (seqLen <= 0 || kvLen <= 0 || qSplitNum <= 0 || mBatch <= 0 || mNumHead <= 0 || mHeadDim <= 0) {
        return INVALID_VALUE;
    }
    const int seqPack = ROUND_UP(seqLen, 32);
    const int headPack = ROUND_UP(mHeadDim, 32);
    const size_t qkvElements64 =
        static_cast<size_t>(seqPack) * headPack * mNumHead * mBatch;
    if (qkvElements64 > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return OUT_OF_MEMORY;
    }
    const int qkvElements = static_cast<int>(qkvElements64);
    const bool shapeChanged = mAdrenoGemmSeqLen != seqLen || mAdrenoGemmKvLen != kvLen ||
        mAdrenoGemmQSplitNum != qSplitNum || mAdrenoGemmQKVElements != qkvElements;
    if (shapeChanged) {
        mAdrenoGemmQKKernels.clear();
        mAdrenoGemmSoftmaxKernels.clear();
        mAdrenoGemmTransKernels.clear();
        mAdrenoGemmQKVKernels.clear();
        mAdrenoGemmClipKernel.reset();
        mAdrenoGemmSeqLen = seqLen;
        mAdrenoGemmKvLen = kvLen;
        mAdrenoGemmQSplitNum = qSplitNum;
        mAdrenoGemmQKVElements = qkvElements;
    }
    if (!mTempQKV || shapeChanged) {
        mTempQKV.reset(Tensor::createDevice<float>({qkvElements}));
        if (!mTempQKV) {
            return OUT_OF_MEMORY;
        }
    }
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempQKV.get(), Backend::DYNAMIC_IN_EXECUTION));
    mOpenCLBackend->onReleaseBuffer(mTempQKV.get(), Backend::DYNAMIC_IN_EXECUTION);
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
    if (mIsKVShared || mMeta == nullptr || !mMeta->sparse_query_active) {
        return false;
    }
    const bool scoreLayerFullQ = queryRowsAreFull && mPicAttentionMode == 1;
    if (!externalHydrated && !scoreLayerFullQ) {
        return false;
    }
    const bool allowSingleRowLaterPicSparse = !queryRowsAreFull && mPicAttentionMode == 2 && attnLen == 1;
    if (attnLen <= 0 || (!allowSingleRowLaterPicSparse && attnLen <= 1) || kvLen <= 0 || mQuerySeqLen <= 0) {
        return false;
    }
    if (!queryRowsAreFull && attnLen != mQuerySeqLen) {
        return false;
    }
    if (queryRowsAreFull && mQuerySeqLen < attnLen) {
        return false;
    }
    // Some graph-boundary exports keep later sparse layers in a full-Q storage
    // layout and rely on sparse_query logical rows to gather the active queries.
    // The sparse flash kernels already support that shape via query_rows_are_full,
    // so don't force those layers back to the row kernel.
    if ((mHeadDim != 64 && mHeadDim != 128) || mNumHead % mKvNumHead != 0) {
        return false;
    }
    if (static_cast<int>(mMeta->sparse_query_logical_indices.size()) < attnLen) {
        return false;
    }
    if (mask != nullptr && mask->elementSize() > 1 && mPicAttentionMode != 1 && mPicAttentionMode != 2) {
        return false;
    }
    return mCache && mCache->key && mCache->value && mCache->sparseQuery;
}

bool PagedAttentionBufExecution::canUseSparseQSplitPrefill(const Tensor* mask, int attnLen, int kvLen,
                                                           bool externalHydrated, bool queryRowsAreFull,
                                                           int* maskKeyLen) const {
    if (maskKeyLen != nullptr) {
        *maskKeyLen = 0;
    }
    if (_legacyB863976OpenCL()) {
        return false;
    }
    if (mIsKVShared || mMeta == nullptr || !mMeta->sparse_query_active) {
        return false;
    }
    const bool scoreLayerFullQ = queryRowsAreFull && mPicAttentionMode == 1;
    if (!externalHydrated && !scoreLayerFullQ) {
        return false;
    }
    if (attnLen <= 0 || kvLen <= 0 || mQuerySeqLen <= 0) {
        return false;
    }
    if (!queryRowsAreFull && attnLen != mQuerySeqLen) {
        return false;
    }
    if (queryRowsAreFull && mQuerySeqLen < attnLen) {
        return false;
    }
    if (mHeadDim != 128 || mNumHead % mKvNumHead != 0) {
        return false;
    }
    if (static_cast<int>(mMeta->sparse_query_logical_indices.size()) < attnLen) {
        return false;
    }
    if (!(mCache && mCache->key && mCache->value && mCache->sparseQuery)) {
        return false;
    }
    const bool ignoreFullCausalMask = mask != nullptr && mask->elementSize() > 1 &&
        mMeta->full_causal_attention_mask;
    // Budgeted sparse prefill already carries logical-row causality through
    // sparse_query/q_logical. For full-causal float masks, the gathered mask
    // rows are redundant on the q-split path and only add QK traffic.
    if (ignoreFullCausalMask) {
        return true;
    }
    if (mask == nullptr || mask->elementSize() <= 1) {
        return true;
    }
    if (mask->getType().code != halide_type_float) {
        return false;
    }
    const int qStorageLen = queryRowsAreFull ? mQuerySeqLen : attnLen;
    const int maskElements = static_cast<int>(mask->elementSize());
    const int64_t fullMaskElements = static_cast<int64_t>(qStorageLen) * kvLen;
    const int64_t shortMaskElements = static_cast<int64_t>(qStorageLen) * qStorageLen;
    if (maskElements >= fullMaskElements) {
        if (maskKeyLen != nullptr) {
            *maskKeyLen = kvLen;
        }
        return true;
    }
    if (maskElements >= shortMaskElements) {
        if (maskKeyLen != nullptr) {
            *maskKeyLen = qStorageLen;
        }
        return true;
    }
    return false;
}

ErrorCode PagedAttentionBufExecution::runSparseFastPrefill(const std::vector<Tensor*>& inputs,
                                                           const std::vector<Tensor*>& outputs, int kvLen,
                                                           int attnLen, bool queryRowsAreFull) {
    auto query = inputs[0];
    auto output = outputs[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const int activeLen = attnLen;
    const int qStorageLen = queryRowsAreFull ? mQuerySeqLen : activeLen;
    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const bool traceProgress = _tracePagedAttentionProgress();
    const int layerIndex = _pagedAttentionProfileLayerIndex(mLayerIndex, mMeta);
    const uint64_t startUs = profile ? _nowUs() : 0;
    if (traceProgress) {
        MNN_PRINT("OpenCLPagedAttention trace phase=enter op=%s layer=%d query=%d input_query=%d full_q=%d kv_len=%d head_dim=%d heads=%d kv_heads=%d\n",
                  queryRowsAreFull ? "score_flash_attention" : "sparse_flash_attention",
                  layerIndex, activeLen, mQuerySeqLen, queryRowsAreFull ? 1 : 0, kvLen, mHeadDim, mNumHead,
                  mKvNumHead);
        std::fflush(stdout);
    }
    uint64_t rearrangeUs = 0;
    uint64_t packUs = 0;
    uint64_t imageCopyUs = 0;
    uint64_t flashUs = 0;
    uint64_t qkRectTiles = 0;
    uint64_t qkActiveTiles = 0;
    uint64_t qkRowTiles = 0;
    int row32Pieces = 0;
    int row64Pieces = 0;
    int mqtileHD64Pieces = 0;
    int mqtileHD128Q4K16Pieces = 0;
    int mqtileHD128Q4K8Pieces = 0;
    int mqtileHD128Q4K8KImagePieces = 0;
    int mqtileHD128Q8K16Pieces = 0;
    int mqtileHD128Q8K16KImagePieces = 0;
    int mqtileHD128Q4K8KVImagePieces = 0;
    int mqtileHD128Q8K16KVImagePieces = 0;
    uint32_t sparseFlashSchedule = _defaultSparseFlashSchedule(mMeta, queryRowsAreFull);
    uint32_t sparseFlashScheduleSource = kTuneSelectionSourceDefault;
    uint32_t forcedSparseFlashSchedule = 0;
    const bool hasForcedSparseFlashSchedule =
        !queryRowsAreFull &&
        _benchSparseFlashScheduleOverride(&forcedSparseFlashSchedule) &&
        _sparseFlashScheduleSupported(mMeta, forcedSparseFlashSchedule, queryRowsAreFull);
    if (hasForcedSparseFlashSchedule) {
        sparseFlashSchedule = forcedSparseFlashSchedule;
    } else if (!queryRowsAreFull &&
               !gSparseFlashScheduleTuneInProgress && !gSparseFlashVariantTuneInProgress) {
        const std::string tuneKey =
            std::string("paged_sparse_flash_schedule_") + _openCLTuneDeviceKey(runtime) + "_" +
            std::to_string(mBatch) + "_" + std::to_string(mNumHead) + "_" +
            std::to_string(mKvNumHead) + "_" + std::to_string(mHeadDim) + "_" +
            std::to_string(queryRowsAreFull ? 1 : 0);
        const std::vector<uint32_t> tuneShape = {
            static_cast<uint32_t>(activeLen),
            static_cast<uint32_t>(kvLen),
            _sparseLogicalWorkPermille(mMeta, activeLen, kvLen),
            static_cast<uint32_t>(_sparseSelectedRatioPercent(mMeta)),
            static_cast<uint32_t>(queryRowsAreFull ? 1 : 0),
        };
        std::pair<std::vector<uint32_t>, uint32_t> tuneInfo;
        if (getTunedInfo(tuneKey, tuneShape, tuneInfo, runtime) && !tuneInfo.first.empty()) {
            const uint32_t tunedSchedule = tuneInfo.first[0];
            if (_sparseFlashScheduleSupported(mMeta, tunedSchedule, queryRowsAreFull) &&
                !_rejectAdrenoSparseFlashScheduleFromCache(tunedSchedule, runtime, mBatch, mKvNumHead, mHeadDim,
                                                           activeLen, kvLen, queryRowsAreFull)) {
                sparseFlashSchedule = tunedSchedule;
                sparseFlashScheduleSource = kTuneSelectionSourceCache;
            }
        } else if (_shouldTuneSparseFlashSchedule(mMeta, activeLen, profile,
                                                  mOpenCLBackend->getCLTuneLevel())) {
            const auto candidates = _sparseFlashScheduleCandidates(mMeta, activeLen, queryRowsAreFull);
            if (candidates.size() > 1) {
                bool tuned = false;
                uint64_t bestUs = std::numeric_limits<uint64_t>::max();
                uint32_t bestSchedule = sparseFlashSchedule;
                gSparseFlashScheduleTuneInProgress = true;
                for (uint32_t candidate : candidates) {
                    gSparseFlashScheduleOverride = static_cast<int>(candidate);
                    uint64_t candidateUs = 0;
                    const char* candidateName = _sparseFlashScheduleName(candidate);
                    _profileTuneCandidateStart("sparse_flash_schedule_tune", layerIndex, activeLen, kvLen,
                                               candidateName);
                    const bool measured = _measureSteadyStateCandidate(runtime, [&]() {
                            return runSparseFastPrefill(inputs, outputs, kvLen, attnLen, queryRowsAreFull);
                        }, &candidateUs);
                    _profileTuneCandidateEnd("sparse_flash_schedule_tune", layerIndex, activeLen, kvLen,
                                             candidateName, measured, candidateUs);
                    if (!measured) {
                        continue;
                    }
                    if (!tuned || candidateUs < bestUs) {
                        tuned = true;
                        bestUs = candidateUs;
                        bestSchedule = candidate;
                    }
                }
                gSparseFlashScheduleOverride = -1;
                gSparseFlashScheduleTuneInProgress = false;
                if (tuned) {
                    std::pair<std::vector<uint32_t>, uint32_t> bestInfo = std::make_pair(
                        std::vector<uint32_t>{bestSchedule},
                        static_cast<uint32_t>(std::min<uint64_t>(bestUs, std::numeric_limits<uint32_t>::max())));
                    setTunedInfo(tuneKey, tuneShape, bestInfo, runtime, "attention_buf");
                    sparseFlashSchedule = bestSchedule;
                    sparseFlashScheduleSource = kTuneSelectionSourceOnlineTuned;
                }
            }
        }
    }
    bool staticWorkspace = _useStaticFullPrefill(qStorageLen, kvLen, mBatch, mNumHead, mKvNumHead, mHeadDim);
    if (_disableAdrenoStaticSparseFlashWorkspace(runtime, mNumHead, mKvNumHead, mHeadDim, activeLen, kvLen,
                                                 queryRowsAreFull)) {
        staticWorkspace = false;
    }
    const bool singlePieceSparseFlash = sparseFlashSchedule == kSparseFlashScheduleSinglePiece;
    int qChunkLen = _sparseFlashScheduleQChunkLen(sparseFlashSchedule, activeLen);
    auto pieces = singlePieceSparseFlash
        ? _buildFixedSparsePieces(mMeta->sparse_query_logical_indices, activeLen, kvLen, qChunkLen)
        : _buildRangeAwareSparsePieces(mMeta->sparse_query_logical_indices, activeLen, kvLen, qChunkLen);
    auto err = ensureSparseFlashTemps(qStorageLen, kvLen, staticWorkspace);
    if (err != NO_ERROR) {
        return err;
    }
    err = ensureSparseFlashKernel();
    if (err != NO_ERROR) {
        return err;
    }
    uint32_t forcedSparseFlashVariant = 0;
    const bool sparseFlashBenchOverride =
        !queryRowsAreFull &&
        _benchSparseFlashVariantOverride(&forcedSparseFlashVariant) &&
        _sparseFlashVariantSupported(forcedSparseFlashVariant, mHeadDim, runtime, mBatch, mKvNumHead, kvLen,
                                     true /* allowMaliExperimentalVariant */);
    uint32_t sparseFlashVariant =
        _defaultSparseFlashVariant(mMeta, activeLen, mHeadDim, runtime, queryRowsAreFull, mBatch, mKvNumHead, kvLen);
    uint32_t sparseFlashVariantSource =
        (sparseFlashBenchOverride && sparseFlashVariant == forcedSparseFlashVariant)
            ? kTuneSelectionSourceBenchOverride
            : kTuneSelectionSourceDefault;
    if (!queryRowsAreFull && !gSparseFlashVariantTuneInProgress) {
        const std::string tuneKey =
            std::string("paged_sparse_flash_variant_") + _openCLTuneDeviceKey(runtime) + "_" +
            std::to_string(mBatch) + "_" + std::to_string(mNumHead) + "_" +
            std::to_string(mKvNumHead) + "_" + std::to_string(mHeadDim) + "_" +
            std::to_string(sparseFlashSchedule);
        const std::vector<uint32_t> tuneShape = {
            static_cast<uint32_t>(activeLen),
            static_cast<uint32_t>(kvLen),
            _sparseLogicalWorkPermille(mMeta, activeLen, kvLen),
            static_cast<uint32_t>(_sparseSelectedRatioPercent(mMeta)),
            static_cast<uint32_t>(queryRowsAreFull ? 1 : 0),
            sparseFlashSchedule,
        };
        std::pair<std::vector<uint32_t>, uint32_t> tuneInfo;
        if (getTunedInfo(tuneKey, tuneShape, tuneInfo, runtime) && !tuneInfo.first.empty()) {
            const uint32_t tunedVariant = tuneInfo.first[0];
            if (_sparseFlashVariantSupported(tunedVariant, mHeadDim, runtime, mBatch, mKvNumHead, kvLen) &&
                !_rejectMaliSparseFlashVariantFromCache(tunedVariant, runtime, mHeadDim, kvLen, queryRowsAreFull) &&
                !_rejectAdrenoSparseFlashVariantFromCache(tunedVariant, runtime, mBatch, mKvNumHead,
                                                          mHeadDim, activeLen, kvLen, queryRowsAreFull)) {
                sparseFlashVariant = tunedVariant;
                sparseFlashVariantSource = kTuneSelectionSourceCache;
            }
        } else if (_shouldTuneSparseFlashVariant(mMeta, activeLen, mHeadDim, profile,
                                                 mOpenCLBackend->getCLTuneLevel())) {
            const auto candidates =
                _sparseFlashVariantCandidates(mHeadDim, runtime, mBatch, mKvNumHead, kvLen, queryRowsAreFull);
            if (candidates.size() > 1) {
                bool tuned = false;
                uint64_t bestUs = std::numeric_limits<uint64_t>::max();
                uint32_t bestVariant = sparseFlashVariant;
                gSparseFlashVariantTuneInProgress = true;
                for (uint32_t candidate : candidates) {
                    gSparseFlashVariantOverride = static_cast<int>(candidate);
                    uint64_t candidateUs = 0;
                    const char* candidateName = _sparseFlashVariantName(candidate);
                    _profileTuneCandidateStart("sparse_flash_variant_tune", layerIndex, activeLen, kvLen,
                                               candidateName);
                    const bool measured = _measureSteadyStateCandidate(runtime, [&]() {
                            return runSparseFastPrefill(inputs, outputs, kvLen, attnLen, queryRowsAreFull);
                        }, &candidateUs);
                    _profileTuneCandidateEnd("sparse_flash_variant_tune", layerIndex, activeLen, kvLen,
                                             candidateName, measured, candidateUs);
                    if (!measured) {
                        continue;
                    }
                    if (!tuned || candidateUs < bestUs) {
                        tuned = true;
                        bestUs = candidateUs;
                        bestVariant = candidate;
                    }
                }
                gSparseFlashVariantOverride = -1;
                gSparseFlashVariantTuneInProgress = false;
                if (tuned) {
                    std::pair<std::vector<uint32_t>, uint32_t> bestInfo = std::make_pair(
                        std::vector<uint32_t>{bestVariant},
                        static_cast<uint32_t>(std::min<uint64_t>(bestUs, std::numeric_limits<uint32_t>::max())));
                    setTunedInfo(tuneKey, tuneShape, bestInfo, runtime, "attention_buf");
                    sparseFlashVariant = bestVariant;
                    sparseFlashVariantSource = kTuneSelectionSourceOnlineTuned;
                }
            }
        }
    }
    const char* sparseFlashVariantName = _sparseFlashVariantName(sparseFlashVariant);
    const char* sparseFlashScheduleName = _sparseFlashScheduleName(sparseFlashSchedule);
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
    const bool directValuePrefill = _useDirectValuePrefillForSparse(mMeta, activeLen, mHeadDim, runtime) &&
        mCache != nullptr && mCache->value != nullptr &&
        mCache->maxSlots >= kvLen;
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

    const bool useSparseFlashKImage =
        sparseFlashVariant == kSparseFlashVariantMQTileHD128Q4K8KImage ||
        sparseFlashVariant == kSparseFlashVariantMQTileHD128Q8K16KImage;
    const bool useSparseFlashKVImage =
        sparseFlashVariant == kSparseFlashVariantMQTileHD128Q4K8KVImage ||
        sparseFlashVariant == kSparseFlashVariantMQTileHD128Q8K16KVImage;
    if (useSparseFlashKImage || useSparseFlashKVImage) {
        if (directValuePrefill && !useSparseFlashKImage) {
            return INVALID_VALUE;
        }
        err = useSparseFlashKVImage ? ensureAdrenoSparseFlashPackedKVImages(kvPack)
                                    : ensureAdrenoSparseFlashPackedKeyImage(kvPack);
        if (err != NO_ERROR) {
            return err;
        }
    }

    idx = 0;
    gws = {static_cast<uint32_t>(UP_DIV(kvLen, 4)), static_cast<uint32_t>(UP_DIV(mHeadDim, 4)),
           static_cast<uint32_t>(mKvNumHead * mBatch)};
    ret = CL_SUCCESS;
    opStartUs = profileDetail ? _nowUs() : 0;
    if (directValuePrefill) {
        if (useSparseFlashKImage) {
            ret |= mPackPagedKeyToImageKernel->get().setArg(idx++, gws[0]);
            ret |= mPackPagedKeyToImageKernel->get().setArg(idx++, gws[1]);
            ret |= mPackPagedKeyToImageKernel->get().setArg(idx++, gws[2]);
            ret |= mPackPagedKeyToImageKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
            ret |= mPackPagedKeyToImageKernel->get().setArg(idx++, *mSparseFlashPackedKeyImage);
            ret |= mPackPagedKeyToImageKernel->get().setArg(idx++, mBatch);
            ret |= mPackPagedKeyToImageKernel->get().setArg(idx++, kvLen);
            ret |= mPackPagedKeyToImageKernel->get().setArg(idx++, mKvNumHead);
            ret |= mPackPagedKeyToImageKernel->get().setArg(idx++, mHeadDim);
            ret |= mPackPagedKeyToImageKernel->get().setArg(idx++, mCache->maxSlots);
            ret |= mPackPagedKeyToImageKernel->get().setArg(idx++, mSparseFlashPackedKeyImageWidth);
            MNN_CHECK_CL_SUCCESS(ret, "setArg sparse flash pack_paged_k_prefill_to_image");
            run3D(mPackPagedKeyToImageKernel, gws, "pack_paged_k_prefill_to_image", "paged_attention_buf");
        } else {
            ret |= mPackPagedKeyKernel->get().setArg(idx++, gws[0]);
            ret |= mPackPagedKeyKernel->get().setArg(idx++, gws[1]);
            ret |= mPackPagedKeyKernel->get().setArg(idx++, gws[2]);
            ret |= mPackPagedKeyKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
            ret |= mPackPagedKeyKernel->get().setArg(idx++, tempBuffer(mTempK.get()));
            ret |= mPackPagedKeyKernel->get().setArg(idx++, mBatch);
            ret |= mPackPagedKeyKernel->get().setArg(idx++, kvLen);
            ret |= mPackPagedKeyKernel->get().setArg(idx++, mKvNumHead);
            ret |= mPackPagedKeyKernel->get().setArg(idx++, mHeadDim);
            ret |= mPackPagedKeyKernel->get().setArg(idx++, mCache->maxSlots);
            MNN_CHECK_CL_SUCCESS(ret, "setArg sparse flash pack_paged_k_prefill");
            run3D(mPackPagedKeyKernel, gws, "pack_paged_k_prefill", "paged_attention_buf");
        }
    } else {
        ret |= mPackPagedKVKernel->get().setArg(idx++, gws[0]);
        ret |= mPackPagedKVKernel->get().setArg(idx++, gws[1]);
        ret |= mPackPagedKVKernel->get().setArg(idx++, gws[2]);
        ret |= mPackPagedKVKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= mPackPagedKVKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= mPackPagedKVKernel->get().setArg(idx++, tempBuffer(mTempK.get()));
        ret |= mPackPagedKVKernel->get().setArg(idx++, tempBuffer(mTempV.get()));
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
    if (useSparseFlashKImage || useSparseFlashKVImage) {
        if (!(directValuePrefill && useSparseFlashKImage)) {
            opStartUs = profileDetail ? _nowUs() : 0;
            uint32_t copyIdx = 0;
            cl_int copyRet = CL_SUCCESS;
            copyRet |= mCopyBufferToImageLinearKernel->get().setArg(copyIdx++, tempBuffer(mTempK.get()));
            copyRet |= mCopyBufferToImageLinearKernel->get().setArg(copyIdx++, *mSparseFlashPackedKeyImage);
            copyRet |= mCopyBufferToImageLinearKernel->get().setArg(copyIdx++, mSparseFlashPackedKeyImageWidth);
            copyRet |= mCopyBufferToImageLinearKernel->get().setArg(copyIdx++, mSparseFlashPackedKeyImageHeight);
            MNN_CHECK_CL_SUCCESS(copyRet, "setArg sparse flash copy packed key to image");
            copyIdx = 0;
            copyRet = runtime->commandQueue().enqueueNDRangeKernel(
                mCopyBufferToImageLinearKernel->get(), cl::NullRange,
                cl::NDRange(mSparseFlashPackedKeyImageWidth, mSparseFlashPackedKeyImageHeight), cl::NullRange);
            MNN_CHECK_CL_SUCCESS(copyRet, "enqueue sparse flash copy packed key to image");
            if (useSparseFlashKVImage) {
                copyIdx = 0;
                copyRet = CL_SUCCESS;
                copyRet |= mCopyBufferToImageLinearKernel->get().setArg(copyIdx++, tempBuffer(mTempV.get()));
                copyRet |= mCopyBufferToImageLinearKernel->get().setArg(copyIdx++, *mSparseFlashPackedValueImage);
                copyRet |= mCopyBufferToImageLinearKernel->get().setArg(copyIdx++, mSparseFlashPackedValueImageWidth);
                copyRet |= mCopyBufferToImageLinearKernel->get().setArg(copyIdx++, mSparseFlashPackedValueImageHeight);
                MNN_CHECK_CL_SUCCESS(copyRet, "setArg sparse flash copy packed value to image");
                copyRet = runtime->commandQueue().enqueueNDRangeKernel(
                    mCopyBufferToImageLinearKernel->get(), cl::NullRange,
                    cl::NDRange(mSparseFlashPackedValueImageWidth, mSparseFlashPackedValueImageHeight), cl::NullRange);
                MNN_CHECK_CL_SUCCESS(copyRet, "enqueue sparse flash copy packed value to image");
            }
            if (profileDetail) {
                runtime->commandQueue().finish();
                imageCopyUs += _nowUs() - opStartUs;
            }
        }
    }

    const int qSplitNum = static_cast<int>(pieces.size());
    if (profile) {
        MNN_PRINT("OpenCLPagedAttention profile op=%s layer=%d phase=begin "
                  "query=%d input_query=%d full_q=%d kv_len=%d q_chunk=%d q_split=%d "
                  "schedule=%s schedule_source=%s static=%d direct_value=%d variant=%s variant_source=%s\n",
                  queryRowsAreFull ? "score_flash_attention" : "sparse_flash_attention",
                  layerIndex, activeLen, mQuerySeqLen, queryRowsAreFull ? 1 : 0, kvLen, qChunkLen, qSplitNum,
                  sparseFlashScheduleName, _tuneSelectionSourceName(sparseFlashScheduleSource),
                  staticWorkspace ? 1 : 0, directValuePrefill ? 1 : 0, sparseFlashVariantName,
                  _tuneSelectionSourceName(sparseFlashVariantSource));
        std::fflush(stdout);
    }
    if (traceProgress) {
        MNN_PRINT("OpenCLPagedAttention trace phase=begin op=%s layer=%d query=%d input_query=%d full_q=%d kv_len=%d q_chunk=%d q_split=%d schedule=%s schedule_source=%s static=%d direct_value=%d variant=%s variant_source=%s\n",
                  queryRowsAreFull ? "score_flash_attention" : "sparse_flash_attention",
                  layerIndex, activeLen, mQuerySeqLen, queryRowsAreFull ? 1 : 0, kvLen, qChunkLen, qSplitNum,
                  sparseFlashScheduleName, _tuneSelectionSourceName(sparseFlashScheduleSource),
                  staticWorkspace ? 1 : 0, directValuePrefill ? 1 : 0, sparseFlashVariantName,
                  _tuneSelectionSourceName(sparseFlashVariantSource));
        std::fflush(stdout);
    }
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

        std::shared_ptr<KernelWrap> flashKernel;
        std::vector<uint32_t> flashLws;
        const char* setArgLabel = nullptr;
        switch (sparseFlashVariant) {
            case kSparseFlashVariantRow32:
                flashKernel = mSparseFlashKernel32;
                gws = {32u, static_cast<uint32_t>(qPieceLen), static_cast<uint32_t>(mNumHead * mBatch)};
                flashLws = {32u, 1u, 1u};
                setArgLabel = "setArg paged sparse flash attention row32";
                ++row32Pieces;
                break;
            case kSparseFlashVariantRow64:
                flashKernel = mSparseFlashKernel64;
                gws = {64u, static_cast<uint32_t>(qPieceLen), static_cast<uint32_t>(mNumHead * mBatch)};
                flashLws = {64u, 1u, 1u};
                setArgLabel = "setArg paged sparse flash attention row64";
                ++row64Pieces;
                break;
            case kSparseFlashVariantMQTileHD64Q4K16:
                flashKernel = mSparseFlashKernelMQTileHD64Q4K16;
                gws = {16u, static_cast<uint32_t>(ROUND_UP(qPieceLen, 4)),
                       static_cast<uint32_t>(mNumHead * mBatch)};
                flashLws = {16u, 4u, 1u};
                setArgLabel = "setArg paged sparse flash attention mqtile_hd64_q4k16";
                ++mqtileHD64Pieces;
                break;
            case kSparseFlashVariantMQTileHD128Q4K16:
                flashKernel = mSparseFlashKernelMQTileHD128Q4K16;
                gws = {16u, static_cast<uint32_t>(ROUND_UP(qPieceLen, 4)),
                       static_cast<uint32_t>(mNumHead * mBatch)};
                flashLws = {16u, 4u, 1u};
                setArgLabel = "setArg paged sparse flash attention mqtile_hd128_q4k16";
                ++mqtileHD128Q4K16Pieces;
                break;
            case kSparseFlashVariantMQTileHD128Q4K8:
                flashKernel = mSparseFlashKernelMQTileHD128Q4K8;
                gws = {16u, static_cast<uint32_t>(ROUND_UP(qPieceLen, 4)),
                       static_cast<uint32_t>(mNumHead * mBatch)};
                flashLws = {16u, 4u, 1u};
                setArgLabel = "setArg paged sparse flash attention mqtile_hd128_q4k8";
                ++mqtileHD128Q4K8Pieces;
                break;
            case kSparseFlashVariantMQTileHD128Q4K8KImage:
                flashKernel = mSparseFlashKernelMQTileHD128Q4K8KImage;
                gws = {16u, static_cast<uint32_t>(ROUND_UP(qPieceLen, 4)),
                       static_cast<uint32_t>(mNumHead * mBatch)};
                flashLws = {16u, 4u, 1u};
                setArgLabel = "setArg paged sparse flash attention mqtile_hd128_q4k8_kimg";
                ++mqtileHD128Q4K8KImagePieces;
                break;
            case kSparseFlashVariantMQTileHD128Q8K16:
                flashKernel = mSparseFlashKernelMQTileHD128Q8K16;
                gws = {16u, static_cast<uint32_t>(ROUND_UP(qPieceLen, 8)),
                       static_cast<uint32_t>(mNumHead * mBatch)};
                flashLws = {16u, 8u, 1u};
                setArgLabel = "setArg paged sparse flash attention mqtile_hd128_q8k16";
                ++mqtileHD128Q8K16Pieces;
                break;
            case kSparseFlashVariantMQTileHD128Q8K16KImage:
                flashKernel = mSparseFlashKernelMQTileHD128Q8K16KImage;
                gws = {16u, static_cast<uint32_t>(ROUND_UP(qPieceLen, 8)),
                       static_cast<uint32_t>(mNumHead * mBatch)};
                flashLws = {16u, 8u, 1u};
                setArgLabel = "setArg paged sparse flash attention mqtile_hd128_q8k16_kimg";
                ++mqtileHD128Q8K16KImagePieces;
                break;
            case kSparseFlashVariantMQTileHD128Q4K8KVImage:
                flashKernel = mSparseFlashKernelMQTileHD128Q4K8KVImage;
                gws = {16u, static_cast<uint32_t>(ROUND_UP(qPieceLen, 4)),
                       static_cast<uint32_t>(mNumHead * mBatch)};
                flashLws = {16u, 4u, 1u};
                setArgLabel = "setArg paged sparse flash attention mqtile_hd128_q4k8_kvimg";
                ++mqtileHD128Q4K8KVImagePieces;
                break;
            case kSparseFlashVariantMQTileHD128Q8K16KVImage:
                flashKernel = mSparseFlashKernelMQTileHD128Q8K16KVImage;
                gws = {16u, static_cast<uint32_t>(ROUND_UP(qPieceLen, 8)),
                       static_cast<uint32_t>(mNumHead * mBatch)};
                flashLws = {16u, 8u, 1u};
                setArgLabel = "setArg paged sparse flash attention mqtile_hd128_q8k16_kvimg";
                ++mqtileHD128Q8K16KVImagePieces;
                break;
            default:
                return INVALID_VALUE;
        }

        idx = 0;
        ret = CL_SUCCESS;
        ret |= flashKernel->get().setArg(idx++, gws[0]);
        ret |= flashKernel->get().setArg(idx++, gws[1]);
        ret |= flashKernel->get().setArg(idx++, gws[2]);
        ret |= flashKernel->get().setArg(idx++, tempBuffer(mTempQ.get()));
        if (useSparseFlashKVImage) {
            ret |= flashKernel->get().setArg(idx++, *mSparseFlashPackedKeyImage);
            ret |= flashKernel->get().setArg(idx++, *mSparseFlashPackedValueImage);
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
            ret |= flashKernel->get().setArg(idx++, mSparseFlashPackedKeyImageWidth);
            ret |= flashKernel->get().setArg(idx++, mSparseFlashPackedValueImageWidth);
            ret |= flashKernel->get().setArg(idx++, mNumHead);
            ret |= flashKernel->get().setArg(idx++, mKvNumHead);
            ret |= flashKernel->get().setArg(idx++, mHeadDim);
        } else if (useSparseFlashKImage) {
            ret |= flashKernel->get().setArg(idx++, *mSparseFlashPackedKeyImage);
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
            ret |= flashKernel->get().setArg(idx++, mSparseFlashPackedKeyImageWidth);
            ret |= flashKernel->get().setArg(idx++, mNumHead);
            ret |= flashKernel->get().setArg(idx++, mKvNumHead);
            ret |= flashKernel->get().setArg(idx++, mHeadDim);
        } else {
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
        }
        MNN_CHECK_CL_SUCCESS(ret, setArgLabel);
        opStartUs = profileDetail ? _nowUs() : 0;
        if (traceProgress) {
            MNN_PRINT("OpenCLPagedAttention trace phase=flash_dispatch op=%s layer=%d q_start=%d q_len=%d active_kv_len=%d kv_len=%d variant=%s gws=%u,%u,%u lws=%u,%u,%u\n",
                      queryRowsAreFull ? "score_flash_attention" : "sparse_flash_attention",
                      layerIndex, qStart, qPieceLen, activeKvLen, kvLen, sparseFlashVariantName,
                      gws[0], gws[1], gws[2], flashLws[0], flashLws[1], flashLws[2]);
            std::fflush(stdout);
        }
        run3DKernelDefault(flashKernel, gws, flashLws, runtime);
        if (profileDetail) {
            runtime->commandQueue().finish();
            flashUs += _nowUs() - opStartUs;
        }
    }
    auto rankErr = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, activeLen, queryRowsAreFull);
    if (rankErr != NO_ERROR) {
        return rankErr;
    }
    if (profile) {
        runtime->commandQueue().finish();
        const uint64_t totalUs = _nowUs() - startUs;
        if (profileDetail) {
            MNN_PRINT("OpenCLPagedAttention profile op=%s layer=%d "
                      "query=%d input_query=%d full_q=%d kv_len=%d q_chunk=%d q_split=%d schedule=%s "
                      "schedule_source=%s static=%d direct_value=%d variant=%s variant_source=%s row32_pieces=%d row64_pieces=%d "
                      "mqtile_hd64_pieces=%d mqtile_hd128_q4k16_pieces=%d "
                      "mqtile_hd128_q4k8_pieces=%d mqtile_hd128_q4k8_kimg_pieces=%d "
                      "mqtile_hd128_q8k16_pieces=%d mqtile_hd128_q8k16_kimg_pieces=%d "
                      "mqtile_hd128_q4k8_kvimg_pieces=%d mqtile_hd128_q8k16_kvimg_pieces=%d us=%llu "
                      "rearrange_us=%llu pack_us=%llu image_copy_us=%llu flash_us=%llu "
                      "qk_rect_tiles=%llu qk_active_tiles=%llu qk_row_tiles=%llu\n",
                      queryRowsAreFull ? "score_flash_attention" : "sparse_flash_attention",
                      layerIndex, activeLen, mQuerySeqLen, queryRowsAreFull ? 1 : 0, kvLen, qChunkLen, qSplitNum,
                      sparseFlashScheduleName, _tuneSelectionSourceName(sparseFlashScheduleSource),
                      staticWorkspace ? 1 : 0, directValuePrefill ? 1 : 0, sparseFlashVariantName,
                      _tuneSelectionSourceName(sparseFlashVariantSource),
                      row32Pieces, row64Pieces, mqtileHD64Pieces, mqtileHD128Q4K16Pieces,
                      mqtileHD128Q4K8Pieces, mqtileHD128Q4K8KImagePieces,
                      mqtileHD128Q8K16Pieces, mqtileHD128Q8K16KImagePieces,
                      mqtileHD128Q4K8KVImagePieces,
                      mqtileHD128Q8K16KVImagePieces,
                      static_cast<unsigned long long>(totalUs),
                      static_cast<unsigned long long>(rearrangeUs),
                      static_cast<unsigned long long>(packUs),
                      static_cast<unsigned long long>(imageCopyUs),
                      static_cast<unsigned long long>(flashUs),
                      static_cast<unsigned long long>(qkRectTiles),
                      static_cast<unsigned long long>(qkActiveTiles),
                      static_cast<unsigned long long>(qkRowTiles));
        } else {
            MNN_PRINT("OpenCLPagedAttention profile op=%s layer=%d "
                      "query=%d input_query=%d full_q=%d kv_len=%d q_chunk=%d q_split=%d schedule=%s "
                      "schedule_source=%s static=%d direct_value=%d variant=%s variant_source=%s row32_pieces=%d row64_pieces=%d "
                      "mqtile_hd64_pieces=%d mqtile_hd128_q4k16_pieces=%d "
                      "mqtile_hd128_q4k8_pieces=%d mqtile_hd128_q4k8_kimg_pieces=%d "
                      "mqtile_hd128_q8k16_pieces=%d mqtile_hd128_q8k16_kimg_pieces=%d "
                      "mqtile_hd128_q4k8_kvimg_pieces=%d mqtile_hd128_q8k16_kvimg_pieces=%d us=%llu\n",
                      queryRowsAreFull ? "score_flash_attention" : "sparse_flash_attention",
                      layerIndex, activeLen, mQuerySeqLen, queryRowsAreFull ? 1 : 0, kvLen, qChunkLen, qSplitNum,
                      sparseFlashScheduleName, _tuneSelectionSourceName(sparseFlashScheduleSource),
                      staticWorkspace ? 1 : 0, directValuePrefill ? 1 : 0, sparseFlashVariantName,
                      _tuneSelectionSourceName(sparseFlashVariantSource),
                      row32Pieces, row64Pieces, mqtileHD64Pieces, mqtileHD128Q4K16Pieces,
                      mqtileHD128Q4K8Pieces, mqtileHD128Q4K8KImagePieces,
                      mqtileHD128Q8K16Pieces, mqtileHD128Q8K16KImagePieces,
                      mqtileHD128Q4K8KVImagePieces,
                      mqtileHD128Q8K16KVImagePieces,
                      static_cast<unsigned long long>(totalUs));
        }
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runSparseQSplitPrefill(const std::vector<Tensor*>& inputs,
                                                             const std::vector<Tensor*>& outputs, int kvLen,
                                                             int attnLen, bool queryRowsAreFull, int maskKeyLen) {
    auto query = inputs[0];
    auto output = outputs[0];
    auto mask = inputs.size() > 3 ? inputs[3] : nullptr;
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const int activeLen = attnLen;
    const int qStorageLen = queryRowsAreFull ? mQuerySeqLen : activeLen;
    const bool ignoreFullCausalMask = mask != nullptr && mask->elementSize() > 1 &&
        mMeta != nullptr && mMeta->full_causal_attention_mask;
    const bool useMask = !ignoreFullCausalMask && mask != nullptr && mask->elementSize() > 1 &&
        mask->getType().code == halide_type_float && maskKeyLen > 0;
    const bool useCompactPackedScoreQ = queryRowsAreFull && !useMask;
    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const bool traceProgress = _tracePagedAttentionProgress();
    const int layerIndex = _pagedAttentionProfileLayerIndex(mLayerIndex, mMeta);
    const uint64_t startUs = profile ? _nowUs() : 0;
    if (traceProgress) {
        MNN_PRINT("OpenCLPagedAttention trace phase=enter op=%s layer=%d query=%d input_query=%d full_q=%d kv_len=%d mask_key_len=%d head_dim=%d heads=%d kv_heads=%d\n",
                  queryRowsAreFull ? "score_qsplit_attention" : "sparse_qsplit_attention",
                  layerIndex, activeLen, mQuerySeqLen, queryRowsAreFull ? 1 : 0, kvLen, maskKeyLen, mHeadDim,
                  mNumHead, mKvNumHead);
        std::fflush(stdout);
    }
    uint64_t rearrangeUs = 0;
    uint64_t packUs = 0;
    uint64_t maskUs = 0;
    uint64_t qkUs = 0;
    uint64_t softmaxUs = 0;
    uint64_t qkvUs = 0;
    uint64_t qkRectTiles = 0;
    uint64_t qkActiveTiles = 0;
    uint64_t qkRowTiles = 0;
    bool staticWorkspace = _useStaticFullPrefill(qStorageLen, kvLen, mBatch, mNumHead, mKvNumHead, mHeadDim);
    const int layerCount = mMeta != nullptr && mMeta->layer_nums > 0 ? mMeta->layer_nums : 1;
    int qChunkLen = _sparseQSplitChunkLen(mMeta, activeLen, kvLen, mBatch, mNumHead, layerCount, staticWorkspace);

    auto ensureTemps = [&](bool wantStaticWorkspace) {
        return ensureFastPrefillTemps(qStorageLen, kvLen, qChunkLen, wantStaticWorkspace);
    };
    auto err = ensureTemps(staticWorkspace);
    if (err != NO_ERROR && staticWorkspace) {
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
        qChunkLen = _sparseQSplitChunkLen(mMeta, activeLen, kvLen, mBatch, mNumHead, layerCount, false);
        err = ensureTemps(false);
    }
    if (err != NO_ERROR) {
        return err;
    }

    const int groupSize = mNumHead / mKvNumHead;
    if (!mQKKernel || !mQKVKernel || mFastKernelStatic || !mFastKernelSparse || mFastKernelAddMask != useMask) {
        std::set<std::string> qkBuildOptions;
        if (useMask) {
            qkBuildOptions.emplace("-DADD_MASK");
        }
        qkBuildOptions.emplace("-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize));
        mQKKernel = runtime->buildKernel("attention_buf", "matmul_qk_div_mask_prefill_piece_sparse",
                                         qkBuildOptions, mOpenCLBackend->getPrecision());
        mQKVKernel = runtime->buildKernel("attention_buf", "matmul_qkv_prefill_piece",
                                          {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                                          mOpenCLBackend->getPrecision());
        OPENCL_CHECK_KERNEL(mQKKernel);
        OPENCL_CHECK_KERNEL(mQKVKernel);
        mFastKernelStatic = false;
        mFastKernelSparse = true;
        mFastKernelAddMask = useMask;
    }

    if (!queryRowsAreFull && !gSparseQSplitTuneInProgress) {
        const int selectedRatioPercent = _sparseSelectedRatioPercent(mMeta);
        const std::string tuneKey =
            std::string("paged_sparse_qsplit_chunk_") + _openCLTuneDeviceKey(runtime) + "_" +
            std::to_string(mBatch) + "_" + std::to_string(mNumHead) + "_" +
            std::to_string(mKvNumHead) + "_" + std::to_string(mHeadDim) + "_" +
            std::to_string(queryRowsAreFull ? 1 : 0) + "_" + std::to_string(useMask ? 1 : 0);
        const std::vector<uint32_t> tuneShape = {
            static_cast<uint32_t>(activeLen),
            static_cast<uint32_t>(kvLen),
            static_cast<uint32_t>(selectedRatioPercent),
        };
        std::pair<std::vector<uint32_t>, uint32_t> tuneInfo;
        if (getTunedInfo(tuneKey, tuneShape, tuneInfo, runtime) && !tuneInfo.first.empty()) {
            int tunedChunk = static_cast<int>(tuneInfo.first[0]);
            tunedChunk = std::max(1, std::min(tunedChunk, activeLen));
            if (tunedChunk < activeLen) {
                tunedChunk = ((tunedChunk + 3) / 4) * 4;
            }
            tunedChunk = std::max(1, std::min(tunedChunk, activeLen));
            if (tunedChunk != qChunkLen) {
                gSparseQSplitTuneInProgress = true;
                gSparseQSplitChunkOverride = tunedChunk;
                auto tunedErr = runSparseQSplitPrefill(inputs, outputs, kvLen, attnLen, queryRowsAreFull, maskKeyLen);
                gSparseQSplitChunkOverride = 0;
                gSparseQSplitTuneInProgress = false;
                return tunedErr;
            }
        } else if (_shouldTuneSparseQSplitChunk(mMeta, activeLen, mHeadDim, staticWorkspace, profile,
                                                mOpenCLBackend->getCLTuneLevel())) {
            const auto candidates = _sparseQSplitChunkCandidates(qChunkLen, activeLen);
            if (candidates.size() > 1) {
                uint64_t bestUs = std::numeric_limits<uint64_t>::max();
                int bestChunk = qChunkLen;
                bool tuned = false;
                gSparseQSplitTuneInProgress = true;
                for (int candidate : candidates) {
                    gSparseQSplitChunkOverride = candidate;
                    uint64_t candidateUs = 0;
                    const std::string candidateName = std::to_string(candidate);
                    _profileTuneCandidateStart("sparse_qsplit_chunk_tune", layerIndex, activeLen, kvLen,
                                               candidateName.c_str());
                    const bool measured = _measureSteadyStateCandidate(runtime, [&]() {
                            return runSparseQSplitPrefill(inputs, outputs, kvLen, attnLen, queryRowsAreFull,
                                                          maskKeyLen);
                        }, &candidateUs);
                    _profileTuneCandidateEnd("sparse_qsplit_chunk_tune", layerIndex, activeLen, kvLen,
                                             candidateName.c_str(), measured, candidateUs);
                    if (!measured) {
                        continue;
                    }
                    if (!tuned || candidateUs < bestUs) {
                        tuned = true;
                        bestUs = candidateUs;
                        bestChunk = candidate;
                    }
                }
                gSparseQSplitChunkOverride = 0;
                gSparseQSplitTuneInProgress = false;
                if (tuned) {
                    std::pair<std::vector<uint32_t>, uint32_t> bestInfo = std::make_pair(
                        std::vector<uint32_t>{static_cast<uint32_t>(bestChunk)},
                        static_cast<uint32_t>(std::min<uint64_t>(bestUs, std::numeric_limits<uint32_t>::max())));
                    setTunedInfo(tuneKey, tuneShape, bestInfo, runtime, "attention_buf");
                    if (bestChunk != qChunkLen) {
                        gSparseQSplitTuneInProgress = true;
                        gSparseQSplitChunkOverride = bestChunk;
                        auto tunedErr = runSparseQSplitPrefill(inputs, outputs, kvLen, attnLen, queryRowsAreFull, maskKeyLen);
                        gSparseQSplitChunkOverride = 0;
                        gSparseQSplitTuneInProgress = false;
                        return tunedErr;
                    }
                }
            }
        }
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

    auto pieces = _buildRangeAwareSparsePieces(mMeta->sparse_query_logical_indices, activeLen, kvLen, qChunkLen);
    const int qSplitNum = static_cast<int>(pieces.size());
    if (profile) {
        MNN_PRINT("OpenCLPagedAttention profile op=%s layer=%d phase=begin "
                  "query=%d input_query=%d full_q=%d kv_len=%d mask_key_len=%d q_chunk=%d q_split=%d "
                  "static=%d compact_score_q=%d add_mask=%d\n",
                  queryRowsAreFull ? "score_qsplit_attention" : "sparse_qsplit_attention",
                  layerIndex, activeLen, mQuerySeqLen, queryRowsAreFull ? 1 : 0, kvLen, maskKeyLen, qChunkLen,
                  qSplitNum, staticWorkspace ? 1 : 0, useCompactPackedScoreQ ? 1 : 0, useMask ? 1 : 0);
        std::fflush(stdout);
    }
    if (traceProgress) {
        MNN_PRINT("OpenCLPagedAttention trace phase=begin op=%s layer=%d query=%d input_query=%d full_q=%d kv_len=%d mask_key_len=%d q_chunk=%d q_split=%d static=%d compact_score_q=%d add_mask=%d\n",
                  queryRowsAreFull ? "score_qsplit_attention" : "sparse_qsplit_attention",
                  layerIndex, activeLen, mQuerySeqLen, queryRowsAreFull ? 1 : 0, kvLen, maskKeyLen, qChunkLen,
                  qSplitNum, staticWorkspace ? 1 : 0, useCompactPackedScoreQ ? 1 : 0, useMask ? 1 : 0);
        std::fflush(stdout);
    }
    const int kvPack = ROUND_UP(kvLen, 32);
    const int headPack4 = ROUND_UP(mHeadDim, 4);
    const int headPack8 = ROUND_UP(mHeadDim, 8);
    const float scale = (mMeta && mMeta->attn_scale > 0)
        ? mMeta->attn_scale
        : (1.0f / std::sqrt(static_cast<float>(mHeadDim)));
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    std::vector<uint32_t> gws;
    uint64_t opStartUs = profileDetail ? _nowUs() : 0;

    idx = 0;
    gws = {static_cast<uint32_t>(UP_DIV(useCompactPackedScoreQ ? activeLen : qStorageLen, 4)),
           static_cast<uint32_t>(UP_DIV(mHeadDim, 4)),
           static_cast<uint32_t>(mNumHead * mBatch)};
    if (useCompactPackedScoreQ) {
        ret |= mRearrangeSparseQKernel->get().setArg(idx++, gws[0]);
        ret |= mRearrangeSparseQKernel->get().setArg(idx++, gws[1]);
        ret |= mRearrangeSparseQKernel->get().setArg(idx++, gws[2]);
        ret |= mRearrangeSparseQKernel->get().setArg(idx++, openCLBuffer(query));
        ret |= mRearrangeSparseQKernel->get().setArg(idx++, openCLBuffer(mCache->sparseQuery.get()));
        ret |= mRearrangeSparseQKernel->get().setArg(idx++, tempBuffer(mTempQ.get()));
        ret |= mRearrangeSparseQKernel->get().setArg(idx++, activeLen);
        ret |= mRearrangeSparseQKernel->get().setArg(idx++, mQuerySeqLen);
        ret |= mRearrangeSparseQKernel->get().setArg(idx++, mHeadDim);
        ret |= mRearrangeSparseQKernel->get().setArg(idx++, mNumHead);
        MNN_CHECK_CL_SUCCESS(ret, "setArg paged sparse qsplit rearrange_q_sparse");
        run3D(mRearrangeSparseQKernel, gws, "rearrange_q_sparse", "attention_buf");
    } else {
        ret |= mRearrangeQKernel->get().setArg(idx++, gws[0]);
        ret |= mRearrangeQKernel->get().setArg(idx++, gws[1]);
        ret |= mRearrangeQKernel->get().setArg(idx++, gws[2]);
        ret |= mRearrangeQKernel->get().setArg(idx++, openCLBuffer(query));
        ret |= mRearrangeQKernel->get().setArg(idx++, tempBuffer(mTempQ.get()));
        ret |= mRearrangeQKernel->get().setArg(idx++, qStorageLen);
        ret |= mRearrangeQKernel->get().setArg(idx++, mHeadDim);
        ret |= mRearrangeQKernel->get().setArg(idx++, mNumHead);
        MNN_CHECK_CL_SUCCESS(ret, "setArg paged sparse qsplit rearrange_q");
        run3D(mRearrangeQKernel, gws, "rearrange_q", "attention_buf");
    }
    if (profileDetail) {
        runtime->commandQueue().finish();
        rearrangeUs += _nowUs() - opStartUs;
    }

    idx = 0;
    gws = {static_cast<uint32_t>(UP_DIV(kvLen, 4)), static_cast<uint32_t>(UP_DIV(mHeadDim, 4)),
           static_cast<uint32_t>(mKvNumHead * mBatch)};
    ret = CL_SUCCESS;
    opStartUs = profileDetail ? _nowUs() : 0;
    ret |= mPackPagedKVKernel->get().setArg(idx++, gws[0]);
    ret |= mPackPagedKVKernel->get().setArg(idx++, gws[1]);
    ret |= mPackPagedKVKernel->get().setArg(idx++, gws[2]);
    ret |= mPackPagedKVKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= mPackPagedKVKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= mPackPagedKVKernel->get().setArg(idx++, tempBuffer(mTempK.get()));
    ret |= mPackPagedKVKernel->get().setArg(idx++, tempBuffer(mTempV.get()));
    ret |= mPackPagedKVKernel->get().setArg(idx++, mBatch);
    ret |= mPackPagedKVKernel->get().setArg(idx++, kvLen);
    ret |= mPackPagedKVKernel->get().setArg(idx++, mKvNumHead);
    ret |= mPackPagedKVKernel->get().setArg(idx++, mHeadDim);
    ret |= mPackPagedKVKernel->get().setArg(idx++, mCache->maxSlots);
    MNN_CHECK_CL_SUCCESS(ret, "setArg sparse qsplit pack_paged_kv_prefill");
    run3D(mPackPagedKVKernel, gws, "pack_paged_kv_prefill", "paged_attention_buf");
    if (profileDetail) {
        runtime->commandQueue().finish();
        packUs += _nowUs() - opStartUs;
    }

    if (useMask) {
        idx = 0;
        gws = {static_cast<uint32_t>(UP_DIV(qStorageLen, 4)), static_cast<uint32_t>(UP_DIV(maskKeyLen, 4)),
               static_cast<uint32_t>(mBatch)};
        ret = CL_SUCCESS;
        opStartUs = profileDetail ? _nowUs() : 0;
        ret |= mRearrangeMaskKernel->get().setArg(idx++, gws[0]);
        ret |= mRearrangeMaskKernel->get().setArg(idx++, gws[1]);
        ret |= mRearrangeMaskKernel->get().setArg(idx++, gws[2]);
        ret |= mRearrangeMaskKernel->get().setArg(idx++, openCLBuffer(mask));
        ret |= mRearrangeMaskKernel->get().setArg(idx++, tempBuffer(mTempMask.get()));
        ret |= mRearrangeMaskKernel->get().setArg(idx++, qStorageLen);
        ret |= mRearrangeMaskKernel->get().setArg(idx++, maskKeyLen);
        MNN_CHECK_CL_SUCCESS(ret, "setArg sparse qsplit rearrange_mask_shortprefill");
        run3D(mRearrangeMaskKernel, gws, "rearrange_mask_shortprefill", "attention_buf");
        if (profileDetail) {
            runtime->commandQueue().finish();
            maskUs += _nowUs() - opStartUs;
        }
    }

    for (const auto& piece : pieces) {
        const int qStart = piece.qStart;
        const int qPieceLen = piece.qLen;
        if (qPieceLen <= 0) {
            continue;
        }
        const int qPiecePack = ROUND_UP(qPieceLen, 4);
        const int activeKvLen = std::max(1, std::min(kvLen, piece.activeKvLen));
        if (profileDetail) {
            qkRectTiles += static_cast<uint64_t>(UP_DIV(qPieceLen, 4)) * UP_DIV(activeKvLen, 4);
            if (static_cast<int>(mMeta->sparse_query_logical_indices.size()) >= qStart + qPieceLen) {
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
            }
        }

        idx = 0;
        gws = {static_cast<uint32_t>(UP_DIV(qPieceLen, 4)), static_cast<uint32_t>(UP_DIV(activeKvLen, 4)),
               static_cast<uint32_t>(mNumHead * mBatch)};
        ret = CL_SUCCESS;
        ret |= mQKKernel->get().setArg(idx++, gws[0]);
        ret |= mQKKernel->get().setArg(idx++, gws[1]);
        ret |= mQKKernel->get().setArg(idx++, gws[2]);
        ret |= mQKKernel->get().setArg(idx++, tempBuffer(mTempQ.get()));
        ret |= mQKKernel->get().setArg(idx++, tempBuffer(mTempK.get()));
        if (useMask) {
            ret |= mQKKernel->get().setArg(idx++, tempBuffer(mTempMask.get()));
        }
        ret |= mQKKernel->get().setArg(idx++, openCLBuffer(mCache->sparseQuery.get()));
        ret |= mQKKernel->get().setArg(idx++, tempBuffer(mTempQK.get()));
        ret |= mQKKernel->get().setArg(idx++, scale);
        ret |= mQKKernel->get().setArg(idx++, useCompactPackedScoreQ ? activeLen : qStorageLen);
        ret |= mQKKernel->get().setArg(idx++, activeLen);
        ret |= mQKKernel->get().setArg(idx++, useCompactPackedScoreQ ? 0 : (queryRowsAreFull ? 1 : 0));
        ret |= mQKKernel->get().setArg(idx++, qStart);
        ret |= mQKKernel->get().setArg(idx++, qPieceLen);
        ret |= mQKKernel->get().setArg(idx++, maskKeyLen);
        ret |= mQKKernel->get().setArg(idx++, activeKvLen);
        ret |= mQKKernel->get().setArg(idx++, kvLen);
        ret |= mQKKernel->get().setArg(idx++, kvPack);
        ret |= mQKKernel->get().setArg(idx++, mNumHead);
        ret |= mQKKernel->get().setArg(idx++, headPack4);
        MNN_CHECK_CL_SUCCESS(ret, "setArg sparse qsplit matmul_qk_div_mask_prefill_piece_sparse");
        opStartUs = profileDetail ? _nowUs() : 0;
        if (traceProgress) {
            MNN_PRINT("OpenCLPagedAttention trace phase=qsplit_qk_dispatch op=%s layer=%d q_start=%d q_len=%d active_kv_len=%d kv_len=%d gws=%u,%u,%u\n",
                      queryRowsAreFull ? "score_qsplit_attention" : "sparse_qsplit_attention",
                      layerIndex, qStart, qPieceLen, activeKvLen, kvLen, gws[0], gws[1], gws[2]);
            std::fflush(stdout);
        }
        run3D(mQKKernel, gws, "matmul_qk_div_mask_prefill_piece_sparse", "attention_buf");
        if (profileDetail) {
            runtime->commandQueue().finish();
            qkUs += _nowUs() - opStartUs;
        }

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
        ret |= mSoftmaxKernel->get().setArg(idx++, activeKvLen);
        MNN_CHECK_CL_SUCCESS(ret, "setArg sparse qsplit softmax");
        opStartUs = profileDetail ? _nowUs() : 0;
        if (traceProgress) {
            MNN_PRINT("OpenCLPagedAttention trace phase=qsplit_softmax_dispatch op=%s layer=%d q_start=%d q_len=%d active_kv_len=%d kv_len=%d gws=%u,%u,%u\n",
                      queryRowsAreFull ? "score_qsplit_attention" : "sparse_qsplit_attention",
                      layerIndex, qStart, qPieceLen, activeKvLen, kvLen, gws[0], gws[1], gws[2]);
            std::fflush(stdout);
        }
        run3DKernelDefault(mSoftmaxKernel, gws, {64u, 1u, 1u}, runtime);
        if (profileDetail) {
            runtime->commandQueue().finish();
            softmaxUs += _nowUs() - opStartUs;
        }

        idx = 0;
        gws = {static_cast<uint32_t>(UP_DIV(mHeadDim, 8)), static_cast<uint32_t>(UP_DIV(qPieceLen, 4)),
               static_cast<uint32_t>(mNumHead * mBatch)};
        ret = CL_SUCCESS;
        ret |= mQKVKernel->get().setArg(idx++, gws[0]);
        ret |= mQKVKernel->get().setArg(idx++, gws[1]);
        ret |= mQKVKernel->get().setArg(idx++, gws[2]);
        ret |= mQKVKernel->get().setArg(idx++, tempBuffer(mTempSoftmax.get()));
        ret |= mQKVKernel->get().setArg(idx++, tempBuffer(mTempV.get()));
        ret |= mQKVKernel->get().setArg(idx++, openCLBuffer(mCache->sparseQuery.get()));
        ret |= mQKVKernel->get().setArg(idx++, openCLBuffer(output));
        ret |= mQKVKernel->get().setArg(idx++, activeLen);
        ret |= mQKVKernel->get().setArg(idx++, activeLen);
        ret |= mQKVKernel->get().setArg(idx++, qStart);
        ret |= mQKVKernel->get().setArg(idx++, qPieceLen);
        ret |= mQKVKernel->get().setArg(idx++, activeKvLen);
        ret |= mQKVKernel->get().setArg(idx++, kvPack);
        ret |= mQKVKernel->get().setArg(idx++, mNumHead);
        ret |= mQKVKernel->get().setArg(idx++, mKvNumHead);
        ret |= mQKVKernel->get().setArg(idx++, 1);
        ret |= mQKVKernel->get().setArg(idx++, headPack8);
        MNN_CHECK_CL_SUCCESS(ret, "setArg sparse qsplit matmul_qkv_prefill_piece");
        opStartUs = profileDetail ? _nowUs() : 0;
        if (traceProgress) {
            MNN_PRINT("OpenCLPagedAttention trace phase=qsplit_qkv_dispatch op=%s layer=%d q_start=%d q_len=%d active_kv_len=%d kv_len=%d gws=%u,%u,%u\n",
                      queryRowsAreFull ? "score_qsplit_attention" : "sparse_qsplit_attention",
                      layerIndex, qStart, qPieceLen, activeKvLen, kvLen, gws[0], gws[1], gws[2]);
            std::fflush(stdout);
        }
        run3D(mQKVKernel, gws, "matmul_qkv_prefill_piece", "attention_buf");
        if (profileDetail) {
            runtime->commandQueue().finish();
            qkvUs += _nowUs() - opStartUs;
        }
    }

    auto rankErr = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, activeLen, queryRowsAreFull);
    if (rankErr != NO_ERROR) {
        return rankErr;
    }

    if (profile) {
        runtime->commandQueue().finish();
        const uint64_t totalUs = _nowUs() - startUs;
        if (profileDetail) {
            MNN_PRINT("OpenCLPagedAttention profile op=%s layer=%d "
                      "query=%d input_query=%d full_q=%d kv_len=%d mask_key_len=%d q_chunk=%d q_split=%d "
                      "static=%d us=%llu rearrange_us=%llu pack_us=%llu mask_us=%llu qk_us=%llu "
                      "softmax_us=%llu qkv_us=%llu qk_rect_tiles=%llu qk_active_tiles=%llu qk_row_tiles=%llu\n",
                      queryRowsAreFull ? "score_qsplit_attention" : "sparse_qsplit_attention",
                      layerIndex, activeLen, mQuerySeqLen, queryRowsAreFull ? 1 : 0, kvLen, maskKeyLen, qChunkLen,
                      qSplitNum, staticWorkspace ? 1 : 0,
                      static_cast<unsigned long long>(totalUs),
                      static_cast<unsigned long long>(rearrangeUs),
                      static_cast<unsigned long long>(packUs),
                      static_cast<unsigned long long>(maskUs),
                      static_cast<unsigned long long>(qkUs),
                      static_cast<unsigned long long>(softmaxUs),
                      static_cast<unsigned long long>(qkvUs),
                      static_cast<unsigned long long>(qkRectTiles),
                      static_cast<unsigned long long>(qkActiveTiles),
                      static_cast<unsigned long long>(qkRowTiles));
        } else {
            MNN_PRINT("OpenCLPagedAttention profile op=%s layer=%d "
                      "query=%d input_query=%d full_q=%d kv_len=%d mask_key_len=%d q_chunk=%d q_split=%d "
                      "static=%d us=%llu\n",
                      queryRowsAreFull ? "score_qsplit_attention" : "sparse_qsplit_attention",
                      layerIndex, activeLen, mQuerySeqLen, queryRowsAreFull ? 1 : 0, kvLen, maskKeyLen, qChunkLen,
                      qSplitNum, staticWorkspace ? 1 : 0,
                      static_cast<unsigned long long>(totalUs));
        }
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runAdrenoGemmPrefill(const std::vector<Tensor*>& inputs,
                                                           const std::vector<Tensor*>& outputs, int kvLen,
                                                           int qSplitNum) {
    auto output = outputs[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (runtime == nullptr || runtime->getGpuType() != GpuType::ADRENO || qSplitNum <= 0 || mTempQKV == nullptr) {
        return INVALID_VALUE;
    }
    const int groupSize = mNumHead / mKvNumHead;
    const int loop = mBatch * mNumHead;
    const int ePack = ROUND_UP(mQuerySeqLen, 32);
    const int ePiece = ePack / qSplitNum;
    const int kvPack = ROUND_UP(kvLen, 32);
    const int headPackQK = ROUND_UP(mHeadDim, 4);
    const int headPackV = ROUND_UP(mHeadDim, 32);
    if (ePiece <= 0 || (ePiece % 32) != 0 || (kvPack % 32) != 0 || (headPackQK % 4) != 0 ||
        (headPackV % 32) != 0) {
        return INVALID_VALUE;
    }

    auto& qBuffer = openCLDeferBuffer(mTempQ.get());
    auto& kBuffer = openCLDeferBuffer(mTempK.get());
    auto& vBuffer = openCLDeferBuffer(mTempV.get());
    auto& maskBuffer = openCLDeferBuffer(mTempMask.get());
    auto& qkBuffer = openCLDeferBuffer(mTempQK.get());
    auto& softmaxBuffer = openCLDeferBuffer(mTempSoftmax.get());
    auto& qkvBuffer = openCLDeferBuffer(mTempQKV.get());

    if (mAdrenoGemmRearrangeQKernel == nullptr) {
        mAdrenoGemmRearrangeQKernel =
            runtime->buildKernel("paged_attention_buf", "rearrange_paged_q_gemm_prefill", {},
                                 mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        OPENCL_CHECK_KERNEL(mAdrenoGemmRearrangeQKernel);
    }
    if (mAdrenoGemmPackPagedKVKernel == nullptr) {
        mAdrenoGemmPackPagedKVKernel =
            runtime->buildKernel("paged_attention_buf", "pack_paged_kv_prefill_gemm", {},
                                 mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        OPENCL_CHECK_KERNEL(mAdrenoGemmPackPagedKVKernel);
    }
    if (mAdrenoGemmMaskKernel == nullptr) {
        mAdrenoGemmMaskKernel = runtime->buildKernel("attention_buf", "rearrange_mask", {},
                                                     mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        OPENCL_CHECK_KERNEL(mAdrenoGemmMaskKernel);
    }
    if (static_cast<int>(mAdrenoGemmQKKernels.size()) != qSplitNum ||
        static_cast<int>(mAdrenoGemmSoftmaxKernels.size()) != qSplitNum ||
        static_cast<int>(mAdrenoGemmTransKernels.size()) != qSplitNum ||
        static_cast<int>(mAdrenoGemmQKVKernels.size()) != qSplitNum ||
        mAdrenoGemmClipKernel == nullptr) {
        mAdrenoGemmQKKernels.assign(static_cast<size_t>(qSplitNum), nullptr);
        mAdrenoGemmSoftmaxKernels.assign(static_cast<size_t>(qSplitNum), nullptr);
        mAdrenoGemmTransKernels.assign(static_cast<size_t>(qSplitNum), nullptr);
        mAdrenoGemmQKVKernels.assign(static_cast<size_t>(qSplitNum), nullptr);
        for (int piece = 0; piece < qSplitNum; ++piece) {
            {
                std::set<std::string> buildOptions;
                constexpr uint32_t layout = 14;
                constexpr int biasType = 2;
                std::vector<cl::Buffer> buffers = {qBuffer, kBuffer, qkBuffer, maskBuffer};
                auto param = getGemmParams({static_cast<uint32_t>(ePiece), static_cast<uint32_t>(kvPack),
                                            static_cast<uint32_t>(headPackQK), layout,
                                            static_cast<uint32_t>(loop),
                                            static_cast<uint32_t>(biasType + 10 * (groupSize - 1))},
                                           buffers, runtime, mOpenCLBackend->getPrecision(),
                                           mOpenCLBackend->getCLTuneLevel());
                _appendGemmBuildOptions(buildOptions, param, layout, true);
                buildOptions.emplace("-DONLY_HAVE_ALPHA");
                buildOptions.emplace("-DBIAS_TYPE=" + std::to_string(biasType));
                buildOptions.emplace("-DPRECISION_COMPUTE=float -DCONVERT_PRECISION_COMPUTE=convert_float");
                buildOptions.emplace("-DPRECISION_COMPUTE2=float2 -DCONVERT_PRECISION_COMPUTE2=convert_float2");
                buildOptions.emplace("-DPRECISION_COMPUTE4=float4 -DCONVERT_PRECISION_COMPUTE4=convert_float4");
                buildOptions.emplace("-DPRECISION_COMPUTE8=float8 -DCONVERT_PRECISION_COMPUTE8=convert_float8");
                buildOptions.emplace("-DPRECISION_COMPUTE16=float16 -DCONVERT_PRECISION_COMPUTE16=convert_float16");
                mAdrenoGemmQKKernels[static_cast<size_t>(piece)] =
                    runtime->buildKernel("matmul_params_buf", "XgemmBatched", buildOptions,
                                         mOpenCLBackend->getPrecision());
                OPENCL_CHECK_KERNEL(mAdrenoGemmQKKernels[static_cast<size_t>(piece)]);
            }
            {
                std::set<std::string> buildOptions;
                buildOptions.emplace("-DSOFTMAX_LOCAL_SIZE=64");
                mAdrenoGemmSoftmaxKernels[static_cast<size_t>(piece)] =
                    runtime->buildKernel("self_attention_buf", "softmax_inside", buildOptions,
                                         mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
                OPENCL_CHECK_KERNEL(mAdrenoGemmSoftmaxKernels[static_cast<size_t>(piece)]);
            }
            {
                mAdrenoGemmTransKernels[static_cast<size_t>(piece)] =
                    runtime->buildKernel("self_attention_buf", "trans_3d_buf", {},
                                         mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
                OPENCL_CHECK_KERNEL(mAdrenoGemmTransKernels[static_cast<size_t>(piece)]);
            }
            {
                std::set<std::string> buildOptions;
                constexpr uint32_t layout = 0;
                std::vector<cl::Buffer> buffers = {qkBuffer, vBuffer, qkvBuffer};
                auto param = getGemmParams({static_cast<uint32_t>(ePiece), static_cast<uint32_t>(headPackV),
                                            static_cast<uint32_t>(kvPack), layout, static_cast<uint32_t>(loop),
                                            static_cast<uint32_t>(0)},
                                           buffers, runtime, mOpenCLBackend->getPrecision(),
                                           mOpenCLBackend->getCLTuneLevel());
                _appendGemmBuildOptions(buildOptions, param, layout, true);
                mAdrenoGemmQKVKernels[static_cast<size_t>(piece)] =
                    runtime->buildKernel("matmul_params_buf", "XgemmBatched", buildOptions,
                                         mOpenCLBackend->getPrecision());
                OPENCL_CHECK_KERNEL(mAdrenoGemmQKVKernels[static_cast<size_t>(piece)]);
            }
        }
        mAdrenoGemmClipKernel = runtime->buildKernel("attention_buf", "qkv_transpose_output", {},
                                                     mOpenCLBackend->getPrecision(), inputs[0], outputs[0]);
        OPENCL_CHECK_KERNEL(mAdrenoGemmClipKernel);
    }

    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    uint64_t qkUs = 0;
    uint64_t softmaxUs = 0;
    uint64_t transUs = 0;
    uint64_t qkvUs = 0;
    uint64_t clipUs = 0;
    uint64_t opStartUs = 0;
    cl_int ret = CL_SUCCESS;
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

    {
        std::vector<uint32_t> gws = {static_cast<uint32_t>(UP_DIV(ePack, 4)),
                                     static_cast<uint32_t>(UP_DIV(mHeadDim, 4)),
                                     static_cast<uint32_t>(loop)};
        uint32_t idx = 0;
        ret = CL_SUCCESS;
        ret |= mAdrenoGemmRearrangeQKernel->get().setArg(idx++, gws[0]);
        ret |= mAdrenoGemmRearrangeQKernel->get().setArg(idx++, gws[1]);
        ret |= mAdrenoGemmRearrangeQKernel->get().setArg(idx++, gws[2]);
        ret |= mAdrenoGemmRearrangeQKernel->get().setArg(idx++, openCLBuffer(inputs[0]));
        ret |= mAdrenoGemmRearrangeQKernel->get().setArg(idx++, qBuffer);
        ret |= mAdrenoGemmRearrangeQKernel->get().setArg(idx++, mQuerySeqLen);
        ret |= mAdrenoGemmRearrangeQKernel->get().setArg(idx++, mHeadDim);
        ret |= mAdrenoGemmRearrangeQKernel->get().setArg(idx++, mNumHead);
        ret |= mAdrenoGemmRearrangeQKernel->get().setArg(idx++, ePack);
        MNN_CHECK_CL_SUCCESS(ret, "setArg paged adreno gemm rearrange_q");
        run3D(mAdrenoGemmRearrangeQKernel, gws, "rearrange_paged_q_gemm_prefill", "paged_attention_buf");
    }
    {
        std::vector<uint32_t> gws = {static_cast<uint32_t>(UP_DIV(kvPack, 4)),
                                     static_cast<uint32_t>(UP_DIV(headPackV, 4)),
                                     static_cast<uint32_t>(mKvNumHead * mBatch)};
        uint32_t idx = 0;
        ret = CL_SUCCESS;
        ret |= mAdrenoGemmPackPagedKVKernel->get().setArg(idx++, gws[0]);
        ret |= mAdrenoGemmPackPagedKVKernel->get().setArg(idx++, gws[1]);
        ret |= mAdrenoGemmPackPagedKVKernel->get().setArg(idx++, gws[2]);
        ret |= mAdrenoGemmPackPagedKVKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= mAdrenoGemmPackPagedKVKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= mAdrenoGemmPackPagedKVKernel->get().setArg(idx++, kBuffer);
        ret |= mAdrenoGemmPackPagedKVKernel->get().setArg(idx++, vBuffer);
        ret |= mAdrenoGemmPackPagedKVKernel->get().setArg(idx++, mBatch);
        ret |= mAdrenoGemmPackPagedKVKernel->get().setArg(idx++, kvLen);
        ret |= mAdrenoGemmPackPagedKVKernel->get().setArg(idx++, mKvNumHead);
        ret |= mAdrenoGemmPackPagedKVKernel->get().setArg(idx++, mHeadDim);
        ret |= mAdrenoGemmPackPagedKVKernel->get().setArg(idx++, mCache->maxSlots);
        ret |= mAdrenoGemmPackPagedKVKernel->get().setArg(idx++, kvPack);
        ret |= mAdrenoGemmPackPagedKVKernel->get().setArg(idx++, headPackV);
        MNN_CHECK_CL_SUCCESS(ret, "setArg paged adreno gemm pack_paged_kv");
        run3D(mAdrenoGemmPackPagedKVKernel, gws, "pack_paged_kv_prefill_gemm", "paged_attention_buf");
    }
    {
        int maskShape[4] = {mQuerySeqLen, kvLen, 32, 32};
        std::vector<uint32_t> gws = {static_cast<uint32_t>(UP_DIV(ePack, 4)),
                                     static_cast<uint32_t>(UP_DIV(kvPack, 4)),
                                     static_cast<uint32_t>(mBatch)};
        uint32_t idx = 0;
        ret = CL_SUCCESS;
        ret |= mAdrenoGemmMaskKernel->get().setArg(idx++, gws[0]);
        ret |= mAdrenoGemmMaskKernel->get().setArg(idx++, gws[1]);
        ret |= mAdrenoGemmMaskKernel->get().setArg(idx++, gws[2]);
        ret |= mAdrenoGemmMaskKernel->get().setArg(idx++, openCLBuffer(inputs[3]));
        ret |= mAdrenoGemmMaskKernel->get().setArg(idx++, maskBuffer);
        ret |= mAdrenoGemmMaskKernel->get().setArg(idx++, maskShape);
        MNN_CHECK_CL_SUCCESS(ret, "setArg paged adreno gemm rearrange_mask");
        run3D(mAdrenoGemmMaskKernel, gws, "rearrange_mask", "attention_buf");
    }

    for (int piece = 0; piece < qSplitNum; ++piece) {
        const int qOffset = ePiece * piece;
        {
            auto kernel = mAdrenoGemmQKKernels[static_cast<size_t>(piece)];
            auto param = getGemmParams({static_cast<uint32_t>(ePiece), static_cast<uint32_t>(kvPack),
                                        static_cast<uint32_t>(headPackQK), static_cast<uint32_t>(14),
                                        static_cast<uint32_t>(loop),
                                        static_cast<uint32_t>(2 + 10 * (groupSize - 1))},
                                       {qBuffer, kBuffer, qkBuffer, maskBuffer}, runtime,
                                       mOpenCLBackend->getPrecision(), mOpenCLBackend->getCLTuneLevel());
            const int tileM = param[4];
            const int tileN = param[7];
            const int localM = param[3];
            const int localN = param[6];
            const int outPerThreadM = std::max(1, tileM / localM);
            const int outPerThreadN = std::max(1, tileN / localN);
            std::vector<uint32_t> gws = {static_cast<uint32_t>(ePiece / outPerThreadM),
                                         static_cast<uint32_t>(kvPack / outPerThreadN),
                                         static_cast<uint32_t>(loop)};
            std::vector<uint32_t> lws = {static_cast<uint32_t>(localM), static_cast<uint32_t>(localN), 1u};
            const float alpha = (mMeta && mMeta->attn_scale > 0)
                ? mMeta->attn_scale
                : (1.0f / std::sqrt(static_cast<float>(mHeadDim)));
            const float beta = 0.0f;
            int batchOffset[4] = {ePack * headPackQK, kvPack * headPackQK, ePiece * kvPack, 0};
            int basePtrOffset[4] = {qOffset, 0, 0, 0};
            int stride[4] = {ePack, kvPack, kvPack, kvPack};
            int group[4] = {1, groupSize, 1, loop};
            uint32_t idx = 0;
            ret = CL_SUCCESS;
            ret |= kernel->get().setArg(idx++, ePiece);
            ret |= kernel->get().setArg(idx++, kvPack);
            ret |= kernel->get().setArg(idx++, headPackQK);
            ret |= kernel->get().setArg(idx++, alpha);
            ret |= kernel->get().setArg(idx++, beta);
            ret |= kernel->get().setArg(idx++, qBuffer);
            ret |= kernel->get().setArg(idx++, kBuffer);
            ret |= kernel->get().setArg(idx++, maskBuffer);
            ret |= kernel->get().setArg(idx++, qkBuffer);
            ret |= kernel->get().setArg(idx++, batchOffset);
            ret |= kernel->get().setArg(idx++, basePtrOffset);
            ret |= kernel->get().setArg(idx++, stride);
            ret |= kernel->get().setArg(idx++, group);
            MNN_CHECK_CL_SUCCESS(ret, "setArg paged adreno gemm qk");
            opStartUs = profileDetail ? _nowUs() : 0;
            run3DKernelDefault(kernel, gws, lws, runtime);
            if (profileDetail) {
                runtime->commandQueue().finish();
                qkUs += _nowUs() - opStartUs;
            }
        }
        {
            auto kernel = mAdrenoGemmSoftmaxKernels[static_cast<size_t>(piece)];
            int softmaxShape[4] = {loop, ePiece, kvPack, 0};
            std::vector<uint32_t> gws = {64u, static_cast<uint32_t>(ePiece), static_cast<uint32_t>(loop)};
            uint32_t idx = 0;
            ret = CL_SUCCESS;
            ret |= kernel->get().setArg(idx++, gws[0]);
            ret |= kernel->get().setArg(idx++, gws[1]);
            ret |= kernel->get().setArg(idx++, gws[2]);
            ret |= kernel->get().setArg(idx++, qkBuffer);
            ret |= kernel->get().setArg(idx++, softmaxBuffer);
            ret |= kernel->get().setArg(idx++, kvLen);
            ret |= kernel->get().setArg(idx++, softmaxShape);
            MNN_CHECK_CL_SUCCESS(ret, "setArg paged adreno gemm softmax");
            opStartUs = profileDetail ? _nowUs() : 0;
            run3DKernelDefault(kernel, gws, {64u, 1u, 1u}, runtime);
            if (profileDetail) {
                runtime->commandQueue().finish();
                softmaxUs += _nowUs() - opStartUs;
            }
        }
        {
            auto kernel = mAdrenoGemmTransKernels[static_cast<size_t>(piece)];
            std::vector<uint32_t> gws = {static_cast<uint32_t>(ePiece / 8),
                                         static_cast<uint32_t>(kvPack / 8),
                                         static_cast<uint32_t>(loop)};
            std::vector<uint32_t> lws;
            if (!_useAdrenoGemmTransNullLws(runtime, ePiece, kvPack, loop, &lws)) {
                auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(kernel));
                lws = localWS3DDefault(gws, maxWorkGroupSize, runtime, "trans_3d_buf", kernel,
                                       mOpenCLBackend->getCLTuneLevel(), "self_attention_buf").first;
            }
            gws[0] = ROUND_UP(gws[0], std::max((uint32_t)1, lws[0]));
            gws[1] = ROUND_UP(gws[1], std::max((uint32_t)1, lws[1]));
            gws[2] = ROUND_UP(gws[2], std::max((uint32_t)1, lws[2]));
            uint32_t idx = 0;
            ret = CL_SUCCESS;
            ret |= kernel->get().setArg(idx++, gws[0]);
            ret |= kernel->get().setArg(idx++, gws[1]);
            ret |= kernel->get().setArg(idx++, gws[2]);
            ret |= kernel->get().setArg(idx++, softmaxBuffer);
            ret |= kernel->get().setArg(idx++, qkBuffer);
            ret |= kernel->get().setArg(idx++, loop);
            ret |= kernel->get().setArg(idx++, ePiece);
            ret |= kernel->get().setArg(idx++, kvPack);
            MNN_CHECK_CL_SUCCESS(ret, "setArg paged adreno gemm transpose");
            opStartUs = profileDetail ? _nowUs() : 0;
            run3DKernelDefault(kernel, gws, lws, runtime);
            if (profileDetail) {
                runtime->commandQueue().finish();
                transUs += _nowUs() - opStartUs;
            }
        }
        {
            auto kernel = mAdrenoGemmQKVKernels[static_cast<size_t>(piece)];
            auto param = getGemmParams({static_cast<uint32_t>(ePiece), static_cast<uint32_t>(headPackV),
                                        static_cast<uint32_t>(kvPack), static_cast<uint32_t>(0),
                                        static_cast<uint32_t>(loop), static_cast<uint32_t>(0)},
                                       {qkBuffer, vBuffer, qkvBuffer}, runtime, mOpenCLBackend->getPrecision(),
                                       mOpenCLBackend->getCLTuneLevel());
            const int tileM = param[4];
            const int tileN = param[7];
            const int localM = param[3];
            const int localN = param[6];
            const int outPerThreadM = std::max(1, tileM / localM);
            const int outPerThreadN = std::max(1, tileN / localN);
            std::vector<uint32_t> gws = {static_cast<uint32_t>(ePiece / outPerThreadM),
                                         static_cast<uint32_t>(headPackV / outPerThreadN),
                                         static_cast<uint32_t>(loop)};
            std::vector<uint32_t> lws = {static_cast<uint32_t>(localM), static_cast<uint32_t>(localN), 1u};
            const float alpha = 1.0f;
            const float beta = 0.0f;
            int batchOffset[4] = {ePiece * kvPack, headPackV * kvPack, ePack * headPackV, 0};
            int basePtrOffset[4] = {0, 0, ePiece * piece, 0};
            int stride[4] = {ePiece, headPackV, ePack, headPackV};
            int group[4] = {1, groupSize, 1, loop};
            uint32_t idx = 0;
            ret = CL_SUCCESS;
            ret |= kernel->get().setArg(idx++, ePiece);
            ret |= kernel->get().setArg(idx++, headPackV);
            ret |= kernel->get().setArg(idx++, kvPack);
            ret |= kernel->get().setArg(idx++, alpha);
            ret |= kernel->get().setArg(idx++, beta);
            ret |= kernel->get().setArg(idx++, qkBuffer);
            ret |= kernel->get().setArg(idx++, vBuffer);
            ret |= kernel->get().setArg(idx++, qkvBuffer);
            ret |= kernel->get().setArg(idx++, batchOffset);
            ret |= kernel->get().setArg(idx++, basePtrOffset);
            ret |= kernel->get().setArg(idx++, stride);
            ret |= kernel->get().setArg(idx++, group);
            MNN_CHECK_CL_SUCCESS(ret, "setArg paged adreno gemm qkv");
            opStartUs = profileDetail ? _nowUs() : 0;
            run3DKernelDefault(kernel, gws, lws, runtime);
            if (profileDetail) {
                runtime->commandQueue().finish();
                qkvUs += _nowUs() - opStartUs;
            }
        }
    }

    {
        std::vector<uint32_t> gws = {static_cast<uint32_t>(UP_DIV(mQuerySeqLen, 4)),
                                     static_cast<uint32_t>(UP_DIV(mHeadDim, 4)),
                                     static_cast<uint32_t>(loop)};
        std::vector<uint32_t> lws;
        if (!_useAdrenoGemmClipNullLws(runtime, mQuerySeqLen, mNumHead, mHeadDim, &lws)) {
            auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mAdrenoGemmClipKernel));
            lws = localWS3DDefault(gws, maxWorkGroupSize, runtime, "qkv_transpose_output",
                                   mAdrenoGemmClipKernel, mOpenCLBackend->getCLTuneLevel(), "attention_buf").first;
        }
        gws[0] = ROUND_UP(gws[0], std::max((uint32_t)1, lws[0]));
        gws[1] = ROUND_UP(gws[1], std::max((uint32_t)1, lws[1]));
        gws[2] = ROUND_UP(gws[2], std::max((uint32_t)1, lws[2]));
        uint32_t idx = 0;
        ret = CL_SUCCESS;
        ret |= mAdrenoGemmClipKernel->get().setArg(idx++, gws[0]);
        ret |= mAdrenoGemmClipKernel->get().setArg(idx++, gws[1]);
        ret |= mAdrenoGemmClipKernel->get().setArg(idx++, gws[2]);
        ret |= mAdrenoGemmClipKernel->get().setArg(idx++, qkvBuffer);
        ret |= mAdrenoGemmClipKernel->get().setArg(idx++, openCLBuffer(output));
        ret |= mAdrenoGemmClipKernel->get().setArg(idx++, 32);
        ret |= mAdrenoGemmClipKernel->get().setArg(idx++, 32);
        ret |= mAdrenoGemmClipKernel->get().setArg(idx++, mQuerySeqLen);
        ret |= mAdrenoGemmClipKernel->get().setArg(idx++, mNumHead);
        ret |= mAdrenoGemmClipKernel->get().setArg(idx++, mHeadDim);
        MNN_CHECK_CL_SUCCESS(ret, "setArg paged adreno gemm qkv_transpose_output");
        opStartUs = profileDetail ? _nowUs() : 0;
        run3DKernelDefault(mAdrenoGemmClipKernel, gws, lws, runtime);
        if (profileDetail) {
            runtime->commandQueue().finish();
            clipUs += _nowUs() - opStartUs;
        }
    }

    if (profile) {
        runtime->commandQueue().finish();
        int layerIndex = mLayerIndex >= 0 ? mLayerIndex : (mMeta != nullptr ? mMeta->layer_index : -1);
        if (profileDetail) {
            MNN_PRINT("OpenCLPagedAttention profile op=prefill_attention_adreno_gemm layer=%d "
                      "query=%d kv_len=%d q_split=%d piece=%d us_detail qk=%llu softmax=%llu "
                      "trans=%llu qkv=%llu clip=%llu\n",
                      layerIndex, mQuerySeqLen, kvLen, qSplitNum, ePiece,
                      static_cast<unsigned long long>(qkUs),
                      static_cast<unsigned long long>(softmaxUs),
                      static_cast<unsigned long long>(transUs),
                      static_cast<unsigned long long>(qkvUs),
                      static_cast<unsigned long long>(clipUs));
        } else {
            MNN_PRINT("OpenCLPagedAttention profile op=prefill_attention_adreno_gemm layer=%d "
                      "query=%d kv_len=%d q_split=%d piece=%d\n",
                      layerIndex, mQuerySeqLen, kvLen, qSplitNum, ePiece);
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
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t rearrangeUs = 0;
    uint64_t packUs = 0;
    uint64_t maskUs = 0;
    uint64_t qkUs = 0;
    uint64_t softmaxUs = 0;
    uint64_t qkvUs = 0;
    const int layerCount = mMeta != nullptr && mMeta->layer_nums > 0 ? mMeta->layer_nums : 1;
    const bool legacy = _legacyB863976OpenCL();
    const bool useAdrenoGemm = _useAdrenoGemmFullPrefill(runtime, mQuerySeqLen, kvLen, mBatch, mNumHead,
                                                         mKvNumHead, mHeadDim, maskKeyLen);
    const int adrenoGemmQSplitNum = useAdrenoGemm
        ? _adrenoGemmPrefillQSplitNum(mQuerySeqLen, kvLen, mBatch, mNumHead)
        : 1;
    bool staticWorkspace = !useAdrenoGemm && (legacy ||
        _useStaticFullPrefill(mQuerySeqLen, kvLen, mBatch, mNumHead, mKvNumHead, mHeadDim));
    int qChunkLen = staticWorkspace ? mQuerySeqLen
                                    : (useAdrenoGemm ? (ROUND_UP(mQuerySeqLen, 32) / adrenoGemmQSplitNum)
                                                     : _prefillQChunkLen(mQuerySeqLen, kvLen, mBatch, mNumHead,
                                                                         layerCount));
    auto err = ensureFastPrefillTemps(mQuerySeqLen, kvLen, qChunkLen, staticWorkspace);
    if (err == NO_ERROR && useAdrenoGemm) {
        err = ensureAdrenoGemmPrefillTemps(mQuerySeqLen, kvLen, adrenoGemmQSplitNum);
    }
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
    if (useAdrenoGemm) {
        auto gemmErr = runAdrenoGemmPrefill(inputs, outputs, kvLen, adrenoGemmQSplitNum);
        if (gemmErr == NO_ERROR && profile) {
            runtime->commandQueue().finish();
            int layerIndex = mLayerIndex >= 0 ? mLayerIndex : (mMeta != nullptr ? mMeta->layer_index : -1);
            MNN_PRINT("OpenCLPagedAttention profile op=prefill_attention_fast_qk_softmax_qkv layer=%d "
                      "query=%d kv_len=%d mask_key_len=%d q_chunk=%d q_split=%d static=0 adreno_gemm=1 us=%llu\n",
                      layerIndex, mQuerySeqLen, kvLen, maskKeyLen, qChunkLen, adrenoGemmQSplitNum,
                      static_cast<unsigned long long>(_nowUs() - startUs));
        }
        return gemmErr;
    }
    if (!useAdrenoGemm &&
        (!mQKKernel || !mQKVKernel || mFastKernelStatic != staticWorkspace || mFastKernelSparse ||
         !mFastKernelAddMask)) {
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
        mFastKernelAddMask = true;
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
    ret |= mPackPagedKVKernel->get().setArg(idx++, gws[0]);
    ret |= mPackPagedKVKernel->get().setArg(idx++, gws[1]);
    ret |= mPackPagedKVKernel->get().setArg(idx++, gws[2]);
    ret |= mPackPagedKVKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= mPackPagedKVKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= mPackPagedKVKernel->get().setArg(idx++, tempBuffer(mTempK.get()));
    ret |= mPackPagedKVKernel->get().setArg(idx++, tempBuffer(mTempV.get()));
    ret |= mPackPagedKVKernel->get().setArg(idx++, mBatch);
    ret |= mPackPagedKVKernel->get().setArg(idx++, kvLen);
    ret |= mPackPagedKVKernel->get().setArg(idx++, mKvNumHead);
    ret |= mPackPagedKVKernel->get().setArg(idx++, mHeadDim);
    ret |= mPackPagedKVKernel->get().setArg(idx++, mCache->maxSlots);
    MNN_CHECK_CL_SUCCESS(ret, "setArg pack_paged_kv_prefill");
    opStartUs = profileDetail ? _nowUs() : 0;
    run3D(mPackPagedKVKernel, gws, "pack_paged_kv_prefill", "paged_attention_buf");
    if (profileDetail) {
        runtime->commandQueue().finish();
        packUs += _nowUs() - opStartUs;
    }

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
    opStartUs = profileDetail ? _nowUs() : 0;
    run3D(mRearrangeMaskKernel, gws, "rearrange_mask_shortprefill", "attention_buf");
    if (profileDetail) {
        runtime->commandQueue().finish();
        maskUs += _nowUs() - opStartUs;
    }

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
        opStartUs = profileDetail ? _nowUs() : 0;
        run3D(mQKKernel, gws, "matmul_qk_div_mask_prefill", "attention_buf");
        if (profileDetail) {
            runtime->commandQueue().finish();
            qkUs += _nowUs() - opStartUs;
        }

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
        opStartUs = profileDetail ? _nowUs() : 0;
        run3DKernelDefault(mSoftmaxKernel, gws, {64u, 1u, 1u}, runtime);
        if (profileDetail) {
            runtime->commandQueue().finish();
            softmaxUs += _nowUs() - opStartUs;
        }

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
        opStartUs = profileDetail ? _nowUs() : 0;
        run3D(mQKVKernel, gws, "matmul_qkv_prefill", "attention_buf");
        if (profileDetail) {
            runtime->commandQueue().finish();
            qkvUs += _nowUs() - opStartUs;
        }
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
            opStartUs = profileDetail ? _nowUs() : 0;
            run3D(mQKKernel, gws, "matmul_qk_div_mask_prefill_piece", "attention_buf");
            if (profileDetail) {
                runtime->commandQueue().finish();
                qkUs += _nowUs() - opStartUs;
            }

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
            opStartUs = profileDetail ? _nowUs() : 0;
            run3DKernelDefault(mSoftmaxKernel, gws, {64u, 1u, 1u}, runtime);
            if (profileDetail) {
                runtime->commandQueue().finish();
                softmaxUs += _nowUs() - opStartUs;
            }

            idx = 0;
            gws = {static_cast<uint32_t>(UP_DIV(mHeadDim, 8)), static_cast<uint32_t>(UP_DIV(qPieceLen, 4)),
                   static_cast<uint32_t>(mNumHead * mBatch)};
            ret = CL_SUCCESS;
            ret |= mQKVKernel->get().setArg(idx++, gws[0]);
            ret |= mQKVKernel->get().setArg(idx++, gws[1]);
            ret |= mQKVKernel->get().setArg(idx++, gws[2]);
            ret |= mQKVKernel->get().setArg(idx++, tempBuffer(mTempSoftmax.get()));
            ret |= mQKVKernel->get().setArg(idx++, tempBuffer(mTempV.get()));
            ret |= mQKVKernel->get().setArg(idx++, openCLBuffer(mCache->sparseQuery.get()));
            ret |= mQKVKernel->get().setArg(idx++, openCLBuffer(output));
            ret |= mQKVKernel->get().setArg(idx++, mQuerySeqLen);
            ret |= mQKVKernel->get().setArg(idx++, mQuerySeqLen);
            ret |= mQKVKernel->get().setArg(idx++, qStart);
            ret |= mQKVKernel->get().setArg(idx++, qPieceLen);
            ret |= mQKVKernel->get().setArg(idx++, kvLen);
            ret |= mQKVKernel->get().setArg(idx++, kvPack);
            ret |= mQKVKernel->get().setArg(idx++, mNumHead);
            ret |= mQKVKernel->get().setArg(idx++, mKvNumHead);
            ret |= mQKVKernel->get().setArg(idx++, 0);
            ret |= mQKVKernel->get().setArg(idx++, headPack8);
            MNN_CHECK_CL_SUCCESS(ret, "setArg paged fast matmul_qkv_prefill_piece");
            opStartUs = profileDetail ? _nowUs() : 0;
            run3D(mQKVKernel, gws, "matmul_qkv_prefill_piece", "attention_buf");
            if (profileDetail) {
                runtime->commandQueue().finish();
                qkvUs += _nowUs() - opStartUs;
            }
        }
    }
    if (profile) {
        runtime->commandQueue().finish();
        int layerIndex = mLayerIndex >= 0 ? mLayerIndex : (mMeta != nullptr ? mMeta->layer_index : -1);
        if (profileDetail) {
            MNN_PRINT("OpenCLPagedAttention profile op=prefill_attention_fast_qk_softmax_qkv layer=%d query=%d kv_len=%d "
                      "mask_key_len=%d q_chunk=%d q_split=%d static=%d us=%llu detail_us rearrange=%llu pack=%llu "
                      "mask=%llu qk=%llu softmax=%llu qkv=%llu\n",
                      layerIndex, mQuerySeqLen, kvLen, maskKeyLen, qChunkLen, qSplitNum, staticWorkspace ? 1 : 0,
                      static_cast<unsigned long long>(_nowUs() - startUs),
                      static_cast<unsigned long long>(rearrangeUs), static_cast<unsigned long long>(packUs),
                      static_cast<unsigned long long>(maskUs), static_cast<unsigned long long>(qkUs),
                      static_cast<unsigned long long>(softmaxUs), static_cast<unsigned long long>(qkvUs));
        } else {
            MNN_PRINT("OpenCLPagedAttention profile op=prefill_attention_fast_qk_softmax_qkv layer=%d query=%d kv_len=%d "
                      "mask_key_len=%d q_chunk=%d q_split=%d static=%d us=%llu\n",
                      layerIndex, mQuerySeqLen, kvLen, maskKeyLen, qChunkLen, qSplitNum, staticWorkspace ? 1 : 0,
                      static_cast<unsigned long long>(_nowUs() - startUs));
        }
    }
    return NO_ERROR;
}

} // namespace OpenCL
} // namespace MNN

#endif /* MNN_OPENCL_BUFFER_CLOSED */
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
