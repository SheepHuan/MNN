//
//  PagedAttentionBufExecution.cpp
//  MNN
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/PagedAttentionBufExecution.hpp"
#include "backend/opencl/execution/buffer/PagedAttentionAdrenoUtils.hpp"
#include "backend/opencl/core/OpenCLPmcProfiler.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include "core/MNNFileUtils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace MNN {
namespace OpenCL {
namespace {

static thread_local int gSteadyStateTuneMeasureDepth = 0;

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

static OpenCLPmcScopeMeta _makeOpenCLPmcMeta(const char* op, const char* phase, int layer, int query,
                                             int inputQuery, int kvLen, int baseLogical, int lane, int qTile,
                                             int heads, int kvHeads, int headDim,
                                             const std::vector<uint32_t>& gws,
                                             const std::vector<uint32_t>& lws,
                                             uint64_t denseKvWork = 0, uint64_t causalKvWork = 0,
                                             int appendCount = 0, int prepareLen = 0,
                                             bool decodePrepareInsideDecode = false) {
    OpenCLPmcScopeMeta meta;
    meta.op = op;
    meta.phase = phase;
    meta.layer = layer;
    meta.query = query;
    meta.inputQuery = inputQuery;
    meta.kvLen = kvLen;
    meta.baseLogical = baseLogical;
    meta.lane = lane;
    meta.qTile = qTile;
    meta.heads = heads;
    meta.kvHeads = kvHeads;
    meta.headDim = headDim;
    meta.gws = gws;
    meta.lws = lws;
    meta.denseKvWork = denseKvWork;
    meta.causalKvWork = causalKvWork;
    meta.appendCount = appendCount;
    meta.prepareLen = prepareLen;
    meta.decodePrepareInsideDecode = decodePrepareInsideDecode;
    return meta;
}

template <typename Runner>
static void _runOpenCLPmcScope(OpenCLRuntime* runtime, const OpenCLPmcScopeMeta& meta, const Runner& runner) {
    if (runtime == nullptr) {
        runner(nullptr);
        return;
    }
    auto& pmc = OpenCLPmcProfiler::get();
    auto& queue = runtime->commandQueue();
    uint64_t token = 0;
    if (pmc.enabledFor(meta)) {
        token = pmc.begin(runtime, queue, meta);
    }
    cl::Event event;
    runner(token != 0 ? &event : nullptr);
    if (token != 0) {
        pmc.end(token, runtime, queue, meta, &event);
    }
}

static void _run3DKernelDefaultPmc(const std::shared_ptr<KernelWrap>& kernel,
                                   const std::vector<uint32_t>& gws,
                                   const std::vector<uint32_t>& lws,
                                   OpenCLRuntime* runtime,
                                   const OpenCLPmcScopeMeta& meta) {
    _runOpenCLPmcScope(runtime, meta, [&](cl::Event* eventPtr) {
        run3DKernelDefault(kernel, gws, lws, runtime, eventPtr);
    });
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

static bool _writeBinaryFile(const std::string& path, const int8_t* data, size_t bytes) {
    std::ofstream os(path, std::ios::binary);
    if (!os.good()) {
        return false;
    }
    if (data != nullptr && bytes > 0) {
        os.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    }
    return os.good();
}

static size_t _roundHostWorkspaceBytes(size_t bytes) {
    constexpr size_t kAlign = 64 * 1024 * 1024;
    if (bytes == 0) {
        return 0;
    }
    return ((bytes + kAlign - 1) / kAlign) * kAlign;
}

static bool _ensureHostWorkspace(std::vector<int8_t>& buffer, size_t bytes) {
    if (bytes == 0) {
        return false;
    }
    if (buffer.size() < bytes) {
        buffer.resize(_roundHostWorkspaceBytes(bytes));
    }
    return buffer.size() >= bytes;
}

struct OpenCLPagedExportWorkspace {
    std::mutex mutex;
    std::vector<int8_t> keyHost;
    std::vector<int8_t> valueStorageHost;
    std::vector<int8_t> valueDataHost;
};

static OpenCLPagedExportWorkspace& _openCLPagedExportWorkspace() {
    static OpenCLPagedExportWorkspace workspace;
    return workspace;
}

struct ExternalLayerReadSegment {
    bool ok = false;
    bool directWritten = false;
    bool valueAlreadyHydrated = false;
    int sourceSlotStart = 0;
    int valueSourceStart = 0;
    int valueSourceStride = 0;
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

struct CacheBlendTopKDispatch {
    uint32_t family = kCacheBlendTopKFamilyLegacy;
    bool useStage = false;
    int stage1LocalSize = 256;
    int stage2LocalSize = 256;
    int blockSize = 0;
    int blockCount = 0;
    int stageCandidateCount = 0;
    int stageSortSize = 0;
};

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

static bool _picOpenCLDebug() {
    return _envFlagEnabled("MNN_PIC_DECODE_DEBUG", false);
}

static bool _decodeGqaFusedKVEnabled() {
    static const bool enabled = _envFlagEnabled("MNN_PAGED_ATTENTION_DECODE_GQA_FUSED", false);
    return enabled;
}

static bool _decodeTransposedKEnabled() {
    static const bool enabled = _envFlagEnabled("MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K", false);
    return enabled;
}

static bool _decodeTransposedKAttentionOnlyEnabled() {
    static const bool enabled = _envFlagEnabled("MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K_ATTENTION_ONLY", false);
    return enabled;
}

static bool _decodeTransposedKReadonlyOnlyEnabled() {
    static const bool enabled = _envFlagEnabled("MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K_READONLY_ONLY", false);
    return enabled;
}

static bool _decodeIdentityAttentionOnlyBenchEnabled() {
    static const bool enabled = _envFlagEnabled("MNN_PAGED_ATTENTION_DECODE_IDENTITY_ATTENTION_ONLY_BENCH", false);
    return enabled;
}

static bool _decodeTransposedKAttentionOnlyBenchEnabled() {
    static const bool enabled =
        _envFlagEnabled("MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K_Q1_ATTENTION_ONLY_BENCH", false);
    return enabled;
}

static bool _decodeQ1SplitProfileEnabled() {
    static const bool enabled = _envFlagEnabled("MNN_PAGED_ATTENTION_DECODE_Q1_SPLIT_PROFILE", false);
    return enabled;
}

static bool _legacyB863976OpenCL();

static bool _decodeTransposedKSparseShapeSupported(OpenCLRuntime* runtime, int attnLen) {
    if (runtime == nullptr || attnLen <= 1 || attnLen > 8) {
        return false;
    }
    return !_legacyB863976OpenCL();
}

static bool _decodeTransposedKQ1NeedsDecodeKey() {
    return _decodeTransposedKReadonlyOnlyEnabled() || _decodeTransposedKAttentionOnlyEnabled() ||
           _decodeTransposedKAttentionOnlyBenchEnabled() || _decodeTransposedKEnabled();
}

static uint32_t _decodeHD128LaneWidth(uint64_t causalWorkPerRow) {
    const char* forced = ::getenv("MNN_PAGED_ATTENTION_DECODE_HD128_FORCE_LANE");
    if (forced != nullptr && forced[0] != '\0') {
        const int lane = ::atoi(forced);
        if (lane == 32 || lane == 64 || lane == 128) {
            return static_cast<uint32_t>(lane);
        }
    }
    return causalWorkPerRow >= 512u ? 128u : (causalWorkPerRow >= 256u ? 64u : 32u);
}

static uint32_t _decodeHD128SparseLaneWidth(OpenCLRuntime* runtime, uint64_t causalWorkPerRow, int attnLen) {
    const char* forced = ::getenv("MNN_PAGED_ATTENTION_DECODE_HD128_FORCE_LANE");
    if (forced != nullptr && forced[0] != '\0') {
        const int lane = ::atoi(forced);
        if (lane == 32 || lane == 64 || lane == 128) {
            return static_cast<uint32_t>(lane);
        }
    }
    if (runtime != nullptr && runtime->getGpuType() == GpuType::MALI && attnLen > 1) {
        return 64u;
    }
    return _decodeHD128LaneWidth(causalWorkPerRow);
}

static int _decodeHD128QTile(int attnLen) {
    const char* forced = ::getenv("MNN_PAGED_ATTENTION_DECODE_QTILE_FORCE");
    if (forced != nullptr && forced[0] != '\0') {
        const int qTile = ::atoi(forced);
        if (qTile == 2 || qTile == 4 || qTile == 8) {
            return qTile;
        }
    }
    return attnLen <= 2 ? 2 : (attnLen <= 4 ? 4 : 8);
}

static bool _compareCacheBlendTopK() {
    return _envFlagEnabled("MNN_PAGED_ATTENTION_COMPARE_CACHEBLEND_TOPK", false);
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

static const char* _scoreSparseFamilyName(uint32_t family) {
    switch (family) {
        case kScoreSparseFamilyQSplit:
            return "qsplit";
        case kScoreSparseFamilyFlash:
            return "flash";
        default:
            return "unknown";
    }
}

static const char* _cacheBlendTopKFamilyName(uint32_t family) {
    switch (family) {
        case kCacheBlendTopKFamilyLegacy:
            return "legacy";
        case kCacheBlendTopKFamilyStage1024:
            return "stage1024";
        case kCacheBlendTopKFamilyStage2048:
            return "stage2048";
        default:
            return "unknown";
    }
}

static int _cacheBlendTopKNextPow2(int value) {
    if (value <= 1) {
        return 1;
    }
    int rounded = 1;
    while (rounded < value && rounded < (1 << 30)) {
        rounded <<= 1;
    }
    return rounded;
}

static bool _cacheBlendTopKDispatchForFamily(OpenCLRuntime* runtime,
                                             const std::shared_ptr<KernelWrap>& stage1Kernel,
                                             const std::shared_ptr<KernelWrap>& stage2Kernel,
                                             int picTokenCount,
                                             int topK,
                                             uint32_t family,
                                             CacheBlendTopKDispatch* dispatch) {
    if (dispatch == nullptr || runtime == nullptr) {
        return false;
    }
    dispatch->family = family;
    dispatch->useStage = false;
    dispatch->stage1LocalSize = 256;
    dispatch->stage2LocalSize = 256;
    dispatch->blockSize = 0;
    dispatch->blockCount = 0;
    dispatch->stageCandidateCount = 0;
    dispatch->stageSortSize = 0;
    if (family == kCacheBlendTopKFamilyLegacy) {
        return true;
    }
    if (picTokenCount <= 1024 || topK <= 0) {
        return false;
    }
    if (runtime->getGpuType() == GpuType::MALI) {
        // Mali-G610 returns different or even invalid indices from the staged
        // path at high context lengths. Keep Mali on the deterministic legacy
        // top-k until the staged kernel is fixed.
        return false;
    }
    int blockSize = 0;
    switch (family) {
        case kCacheBlendTopKFamilyStage1024:
            blockSize = 1024;
            break;
        case kCacheBlendTopKFamilyStage2048:
            blockSize = 2048;
            break;
        default:
            return false;
    }
    if (topK > blockSize) {
        return false;
    }
    const int blockCount = UP_DIV(picTokenCount, blockSize);
    const int candidateCount = blockCount * topK;
    const int sortSize = _cacheBlendTopKNextPow2(candidateCount);
    constexpr int kTopKMaxSortSize = 4096;
    if (blockCount <= 0 || candidateCount <= 0 || sortSize > kTopKMaxSortSize) {
        return false;
    }
    const size_t stage1Bytes = static_cast<size_t>(blockSize) * (sizeof(float) + sizeof(int));
    const size_t stage2Bytes = static_cast<size_t>(sortSize) * (sizeof(float) + sizeof(int));
    size_t localMem = runtime->getMaxLocalMem();
    if (std::max(stage1Bytes, stage2Bytes) > localMem) {
        return false;
    }
    constexpr int kMaxPreferredLocalSize = 256;
    const int stage2LocalSize = std::max(1, std::min(kMaxPreferredLocalSize, sortSize));
    if (runtime->getMaxWorkGroupSize(stage1Kernel) < kMaxPreferredLocalSize ||
        runtime->getMaxWorkGroupSize(stage2Kernel) < stage2LocalSize) {
        return false;
    }
    dispatch->useStage = true;
    dispatch->stage1LocalSize = kMaxPreferredLocalSize;
    dispatch->stage2LocalSize = stage2LocalSize;
    dispatch->blockSize = blockSize;
    dispatch->blockCount = blockCount;
    dispatch->stageCandidateCount = candidateCount;
    dispatch->stageSortSize = sortSize;
    return true;
}

static std::vector<uint32_t> _cacheBlendTopKFamilyCandidates(OpenCLRuntime* runtime,
                                                             const std::shared_ptr<KernelWrap>& stage1Kernel,
                                                             const std::shared_ptr<KernelWrap>& stage2Kernel,
                                                             int picTokenCount,
                                                             int topK) {
    std::vector<uint32_t> candidates;
    if (picTokenCount <= 1024 || topK <= 0 || runtime == nullptr) {
        return candidates;
    }
    CacheBlendTopKDispatch dispatch;
    for (uint32_t family : {kCacheBlendTopKFamilyStage1024, kCacheBlendTopKFamilyStage2048}) {
        if (_cacheBlendTopKDispatchForFamily(runtime, stage1Kernel, stage2Kernel, picTokenCount, topK, family,
                                             &dispatch)) {
            candidates.emplace_back(family);
        }
    }
    return candidates;
}

static bool _shouldTuneCacheBlendTopKFamily(int picTokenCount, int topK, bool profile, int tuneLevel) {
    if ((profile && !_profilePagedAttentionTune()) || picTokenCount <= 1024 || topK <= 0) {
        return false;
    }
    return tuneLevel == Heavy || tuneLevel == Wide;
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
    return runtime != nullptr && runtime->getGpuType() == GpuType::MALI && headDim == 128 &&
        !queryRowsAreFull && kvLen > 0 && kvLen <= 2048;
}

static bool _maliHeadDim128SparseVariantSupported(uint32_t variant, int headDim, OpenCLRuntime* runtime, int kvLen) {
    if (runtime == nullptr || runtime->getGpuType() != GpuType::MALI || headDim != 128) {
        return false;
    }
    if (kvLen > 0 && kvLen <= 2048 && variant == kSparseFlashVariantMQTileHD128Q8K16) {
        return true;
    }
    return variant == kSparseFlashVariantRow32;
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
    if (!_isSparseFlashKVImageVariant(variant)) {
        if (!queryRowsAreFull &&
            _preferAdrenoLaterSparseQ4K8(runtime, batch, kvHeads, headDim, activeLen, kvLen)) {
            return variant != kSparseFlashVariantMQTileHD128Q4K8;
        }
        return false;
    }
    return _preferAdrenoSparseFlashKImage(runtime, batch, kvHeads, headDim, kvLen);
}

static bool _rejectMaliSparseFlashVariantFromCache(uint32_t variant, OpenCLRuntime* runtime, int headDim,
                                                   int kvLen, bool queryRowsAreFull) {
    if (runtime == nullptr || runtime->getGpuType() != GpuType::MALI || headDim != 128) {
        return false;
    }
    if (_preferMaliLaterSparseQ8K16(runtime, headDim, kvLen, queryRowsAreFull)) {
        return variant != kSparseFlashVariantMQTileHD128Q8K16;
    }
    return variant != kSparseFlashVariantRow32;
}

static bool _rejectAdrenoSparseFlashScheduleFromCache(uint32_t schedule, OpenCLRuntime* runtime, int batch,
                                                      int kvHeads, int headDim, int activeLen, int kvLen,
                                                      bool queryRowsAreFull) {
    if (queryRowsAreFull || schedule != kSparseFlashScheduleRangeQ128) {
        return false;
    }
    return _preferAdrenoLaterSparseQ4K8(runtime, batch, kvHeads, headDim, activeLen, kvLen);
}

static bool _disableAdrenoStaticSparseFlashWorkspace(OpenCLRuntime* runtime, int numHeads, int kvHeads, int headDim,
                                                     int activeLen, int kvLen, bool queryRowsAreFull) {
    if (queryRowsAreFull || runtime == nullptr || runtime->getGpuType() != GpuType::ADRENO ||
        _legacyB863976OpenCL()) {
        return false;
    }
    if (headDim != 128 || numHeads < 32 || kvHeads <= 0 || activeLen < 768 || kvLen < 2048) {
        return false;
    }
    return true;
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

static std::mutex& _openCLPagedCacheRegistryMutex() {
    static std::mutex mutex;
    return mutex;
}

static std::unordered_map<std::string, std::weak_ptr<PagedAttentionBufExecution::SharedPagedCache>>&
_openCLPagedCacheRegistry() {
    static std::unordered_map<std::string, std::weak_ptr<PagedAttentionBufExecution::SharedPagedCache>> registry;
    return registry;
}

static int _openCLPagedCacheLayerKey(int layerIndex, int kvSharedLayerIndex, const PagedKVMeta* meta) {
    if (kvSharedLayerIndex >= 0) {
        return kvSharedLayerIndex;
    }
    if (layerIndex >= 0) {
        return layerIndex;
    }
    return meta != nullptr ? meta->layer_index : -1;
}

static std::string _openCLPagedCacheRegistryKey(OpenCLBackend* backend, const PagedKVMeta* meta,
                                                int layerIndex, int kvSharedLayerIndex) {
    std::ostringstream os;
    os << reinterpret_cast<uintptr_t>(backend != nullptr ? backend->getOpenCLRuntime() : nullptr)
       << ":" << reinterpret_cast<uintptr_t>(meta)
       << ":" << _openCLPagedCacheLayerKey(layerIndex, kvSharedLayerIndex, meta);
    return os.str();
}

static bool _openCLPagedCacheCompatible(
    const std::shared_ptr<PagedAttentionBufExecution::SharedPagedCache>& cache,
    int maxSlots, int batch, int kvHeads, int headDim, int bytes) {
    return cache && cache->key && cache->decodeKey && cache->value && cache->sparseQuery &&
        cache->maxSlots >= maxSlots && cache->batch == batch && cache->kvHeads == kvHeads &&
        cache->headDim == headDim && cache->bytes == bytes;
}

static std::shared_ptr<PagedAttentionBufExecution::SharedPagedCache> _findOpenCLPagedCache(
    const std::string& key, int maxSlots, int batch, int kvHeads, int headDim, int bytes) {
    std::lock_guard<std::mutex> lock(_openCLPagedCacheRegistryMutex());
    auto& registry = _openCLPagedCacheRegistry();
    auto iter = registry.find(key);
    if (iter == registry.end()) {
        return nullptr;
    }
    auto cache = iter->second.lock();
    if (!_openCLPagedCacheCompatible(cache, maxSlots, batch, kvHeads, headDim, bytes)) {
        if (!cache) {
            registry.erase(iter);
        }
        return nullptr;
    }
    return cache;
}

static void _storeOpenCLPagedCache(
    const std::string& key,
    const std::shared_ptr<PagedAttentionBufExecution::SharedPagedCache>& cache) {
    if (!cache) {
        return;
    }
    std::lock_guard<std::mutex> lock(_openCLPagedCacheRegistryMutex());
    _openCLPagedCacheRegistry()[key] = cache;
}

struct OpenCLPagedDecodePrepareEntry {
    int layerIndex = -1;
    PagedAttentionBufExecution* execution = nullptr;
};

static bool _prepareOpenCLPagedDecode(PagedKVMeta* meta, int maxNewTokens);

static std::mutex& _openCLPagedDecodePrepareRegistryMutex() {
    static std::mutex mutex;
    return mutex;
}

static std::unordered_map<PagedKVMeta*, std::vector<OpenCLPagedDecodePrepareEntry>>&
_openCLPagedDecodePrepareRegistry() {
    static std::unordered_map<PagedKVMeta*, std::vector<OpenCLPagedDecodePrepareEntry>> registry;
    return registry;
}

static void _registerOpenCLPagedDecodePrepare(PagedKVMeta* meta, int layerIndex,
                                              PagedAttentionBufExecution* execution) {
    if (meta == nullptr || execution == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(_openCLPagedDecodePrepareRegistryMutex());
    auto& entries = _openCLPagedDecodePrepareRegistry()[meta];
    const auto exists = std::find_if(entries.begin(), entries.end(),
                                     [execution](const OpenCLPagedDecodePrepareEntry& entry) {
                                         return entry.execution == execution;
                                     }) != entries.end();
    if (!exists) {
        OpenCLPagedDecodePrepareEntry entry;
        entry.layerIndex = layerIndex;
        entry.execution = execution;
        entries.push_back(entry);
    }
    meta->decode_prepare_callback = _prepareOpenCLPagedDecode;
}

static void _unregisterOpenCLPagedDecodePrepare(PagedKVMeta* meta, PagedAttentionBufExecution* execution) {
    if (meta == nullptr || execution == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(_openCLPagedDecodePrepareRegistryMutex());
    auto& registry = _openCLPagedDecodePrepareRegistry();
    auto iter = registry.find(meta);
    if (iter == registry.end()) {
        return;
    }
    auto& entries = iter->second;
    entries.erase(std::remove_if(entries.begin(), entries.end(),
                                 [execution](const OpenCLPagedDecodePrepareEntry& entry) {
                                     return entry.execution == execution;
                                 }),
                  entries.end());
    if (entries.empty()) {
        registry.erase(iter);
        if (meta->decode_prepare_callback == _prepareOpenCLPagedDecode) {
            meta->decode_prepare_callback = nullptr;
        }
    }
}

static bool _prepareOpenCLPagedDecode(PagedKVMeta* meta, int maxNewTokens) {
    if (meta == nullptr) {
        return true;
    }
    const int kvLen = std::max(0, meta->logical_length);
    if (kvLen <= 0) {
        return true;
    }
    std::vector<OpenCLPagedDecodePrepareEntry> entries;
    {
        std::lock_guard<std::mutex> lock(_openCLPagedDecodePrepareRegistryMutex());
        auto iter = _openCLPagedDecodePrepareRegistry().find(meta);
        if (iter != _openCLPagedDecodePrepareRegistry().end()) {
            entries = iter->second;
        }
    }
    if (entries.empty()) {
        if (_picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA prepare decode missing executions meta=%p kv_len=%d max_new=%d\n",
                      static_cast<void*>(meta), kvLen, maxNewTokens);
        }
        return false;
    }
    std::sort(entries.begin(), entries.end(),
              [](const OpenCLPagedDecodePrepareEntry& a, const OpenCLPagedDecodePrepareEntry& b) {
                  if (a.layerIndex != b.layerIndex) {
                      return a.layerIndex < b.layerIndex;
                  }
                  return a.execution < b.execution;
              });
    int preparedLayers = 0;
    const uint64_t startUs = _profilePagedAttention() ? _nowUs() : 0;
    for (size_t i = 0; i < entries.size();) {
        const int layerIndex = entries[i].layerIndex;
        if (layerIndex < 0) {
            ++i;
            continue;
        }
        bool layerPrepared = false;
        ErrorCode layerErr = NOT_SUPPORT;
        while (i < entries.size() && entries[i].layerIndex == layerIndex) {
            if (entries[i].execution != nullptr) {
                const auto err = entries[i].execution->prepareDecodePrefix(kvLen);
                if (err == NO_ERROR) {
                    layerPrepared = true;
                    break;
                }
                if (err != NOT_SUPPORT) {
                    layerErr = err;
                }
            }
            ++i;
        }
        while (i < entries.size() && entries[i].layerIndex == layerIndex) {
            ++i;
        }
        if (!layerPrepared) {
            MNN_ERROR("OpenCLPagedAttention prepare decode failed layer=%d kv_len=%d err=%d\n",
                      layerIndex, kvLen, static_cast<int>(layerErr));
            return false;
        }
        ++preparedLayers;
    }
    if (meta->layer_nums > 0 && preparedLayers < meta->layer_nums) {
        MNN_ERROR("OpenCLPagedAttention prepare decode incomplete, prepared=%d expected=%d kv_len=%d\n",
                  preparedLayers, meta->layer_nums, kvLen);
        return false;
    }
    if (_profilePagedAttention()) {
        MNN_PRINT("OpenCLPagedAttention profile op=decode_prepare_request layers=%d kv_len=%d max_new=%d us=%llu\n",
                  preparedLayers, kvLen, maxNewTokens,
                  static_cast<unsigned long long>(_nowUs() - startUs));
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

static int _alignPicCacheSourceSlots(int slots) {
    constexpr int kSourceSlotAlignment = 32;
    if (slots <= 0) {
        return slots;
    }
    if (slots > std::numeric_limits<int>::max() - kSourceSlotAlignment) {
        return std::numeric_limits<int>::max();
    }
    return ROUND_UP(slots, kSourceSlotAlignment);
}

static size_t _segmentSourceSlotSpan(const std::vector<PagedKVExternalSegment>& segments) {
    size_t cursor = 0;
    for (const auto& segment : segments) {
        if (cursor > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return cursor;
        }
        cursor = static_cast<size_t>(_alignPicCacheSourceSlots(static_cast<int>(cursor)));
        cursor += segment.tokenCount;
    }
    if (cursor > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return cursor;
    }
    return static_cast<size_t>(_alignPicCacheSourceSlots(static_cast<int>(cursor)));
}

static int _picCacheSourceSlotBase(const PagedKVMeta* meta, int kvLen) {
    if (meta == nullptr) {
        return kvLen;
    }
    return _alignPicCacheSourceSlots(std::max(kvLen, meta->request_capacity));
}

static size_t _picCacheSourceSlotCount(const PagedKVMeta* meta) {
    if (meta == nullptr) {
        return 0;
    }
    return std::max({static_cast<size_t>(_alignPicCacheSourceSlots(
                         static_cast<int>(std::min<size_t>(
                             meta->external_source_slot_reserve,
                             static_cast<size_t>(std::numeric_limits<int>::max()))))),
                     _segmentSourceSlotSpan(meta->external_segments),
                     _segmentSourceSlotSpan(meta->cacheblend_score_segments)});
}

struct ExternalLayerMappedTarget {
    std::shared_ptr<Tensor> key;
    std::shared_ptr<Tensor> value;
    std::shared_ptr<cl::CommandQueue> queue;
    bool valueSourceHydrate = false;
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
    bool valueSourceHydrate = false;
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
                                               cl::CommandQueue& queue, bool valueSourceHydrate) {
    if (!_shouldUseDirectPagedCacheOpenCL() || meta == nullptr || layerIndex < 0 || key == nullptr ||
        value == nullptr || key->deviceId() == 0 || value->deviceId() == 0) {
        return;
    }
    ExternalLayerMappedTargetRef ref;
    ref.key = key;
    ref.value = value;
    ref.queue.reset(new cl::CommandQueue(queue));
    ref.valueSourceHydrate = valueSourceHydrate;
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
    target.valueSourceHydrate = iter->second.valueSourceHydrate;
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
    os << reinterpret_cast<uintptr_t>(meta) << ":" << (meta != nullptr ? meta->request_generation : 0) << ":"
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
    if (!_envFlagEnabled("MNN_PAGED_ATTENTION_OPENCL_ALLOW_KV_STAGING_FALLBACK", false)) {
        static std::once_flag mapRequiredOnce;
        std::call_once(mapRequiredOnce, []() {
            MNN_PRINT("OpenCLPagedAttention: mapped source CL buffer is required for PIC cache key hydrate; "
                      "set MNN_PAGED_ATTENTION_OPENCL_ALLOW_KV_STAGING_FALLBACK=1 only for debug fallback\n");
        });
        return false;
    }
    static std::once_flag fallbackOnce;
    std::call_once(fallbackOnce, []() {
        MNN_PRINT("OpenCLPagedAttention: debug fallback copies PIC cache key source with enqueueWriteBuffer\n");
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
                const int slot = static_cast<int>(logical);
                if (slot < 0 || slot >= maxSlots) {
                    ok = false;
                    break;
                }
                size_t run = 1;
                while (local + run < tokenCount) {
                    const size_t nextLogical = logicalStart + local + run;
                    const int nextSlot = static_cast<int>(nextLogical);
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
        bool directValueOk = false;
        bool valueAlreadyHydrated = false;
        if (directKeyOk && target->valueSourceHydrate) {
            directValueOk = _readExternalValueSegmentToPagedCacheOpenCL(
                layer->valuePath, valueBuffer, *target->queue, batch, kvHeads, target->maxSlots,
                static_cast<size_t>(sourceSlotStart), sourceTokenCount, sourceTokenOffset,
                segment.tokenCount, headDim, bytes);
        } else if (directKeyOk) {
            directValueOk = _readExternalValueSegmentToPhysicalPagedCacheOpenCL(
                layer->valuePath, valueBuffer, *target->queue, meta, batch, kvHeads, target->maxSlots,
                segment.logicalStart, sourceTokenCount, sourceTokenOffset, segment.tokenCount, headDim, bytes);
            valueAlreadyHydrated = directValueOk;
        }
        if (directKeyOk && directValueOk) {
            out.directWritten = true;
            out.valueAlreadyHydrated = valueAlreadyHydrated;
            out.sourceSlotStart = sourceSlotStart;
            out.valueSourceStart = target->valueSourceHydrate ? sourceSlotStart : 0;
            out.valueSourceStride = target->valueSourceHydrate ? target->maxSlots
                                                               : static_cast<int>(segment.tokenCount);
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
        sourceSlotCursor = _alignPicCacheSourceSlots(sourceSlotCursor);
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
    _registerOpenCLPagedDecodePrepare(mMeta, mLayerIndex, this);
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    mCopyKernel = runtime->buildKernel("paged_attention_buf", "copy_paged_kv", {}, mOpenCLBackend->getPrecision());
    mAttentionKernel = runtime->buildKernel("paged_attention_buf", "paged_attention", {}, mOpenCLBackend->getPrecision());
    mAttentionRowKernel = runtime->buildKernel("paged_attention_buf", "paged_attention_row", {},
                                               mOpenCLBackend->getPrecision());
    mPackPagedKVKernel = runtime->buildKernel("paged_attention_buf", "pack_paged_kv_prefill", {},
                                              mOpenCLBackend->getPrecision());
    mPackPagedKeyKernel = runtime->buildKernel("paged_attention_buf", "pack_paged_k_prefill", {},
                                               mOpenCLBackend->getPrecision());
    mPackPagedKeyToImageKernel = runtime->buildKernel("paged_attention_buf", "pack_paged_k_prefill_to_image", {},
                                                      mOpenCLBackend->getPrecision());
    mHydrateExternalKernel = runtime->buildKernel("paged_attention_buf", "pic_page_attention_hydrate_kv", {},
                                                  mOpenCLBackend->getPrecision());
    mHydrateExternalInplaceKernel = runtime->buildKernel("paged_attention_buf", "pic_page_attention_hydrate_kv_inplace",
                                                         {}, mOpenCLBackend->getPrecision());
    mExportCanonicalKeyKernel = runtime->buildKernel("paged_attention_buf", "export_canonical_paged_key", {},
                                                     mOpenCLBackend->getPrecision());
    mCacheBlendScoreKernel = runtime->buildKernel("paged_attention_buf", "pic_cacheblend_value_score", {},
                                                  mOpenCLBackend->getPrecision());
    mCacheBlendScoreImageKernel = runtime->buildKernel("paged_attention_buf",
                                                       "pic_cacheblend_value_score_cached_image", {},
                                                       mOpenCLBackend->getPrecision());
    mCacheBlendTopKKernel = runtime->buildKernel("paged_attention_buf", "pic_cacheblend_topk", {},
                                                 mOpenCLBackend->getPrecision());
    mCacheBlendTopKStage1Kernel = runtime->buildKernel("paged_attention_buf", "pic_cacheblend_topk_stage1", {},
                                                       mOpenCLBackend->getPrecision());
    mCacheBlendTopKStage2Kernel = runtime->buildKernel("paged_attention_buf", "pic_cacheblend_topk_stage2", {},
                                                       mOpenCLBackend->getPrecision());
    mRearrangeQKernel = runtime->buildKernel("attention_buf", "rearrange_q", {}, mOpenCLBackend->getPrecision());
    mRearrangeSparseQKernel = runtime->buildKernel("attention_buf", "rearrange_q_sparse", {},
                                                   mOpenCLBackend->getPrecision());
    mRearrangeMaskKernel = runtime->buildKernel("attention_buf", "rearrange_mask_shortprefill", {"-DADD_MASK"},
                                                mOpenCLBackend->getPrecision());
    mSoftmaxKernel = runtime->buildKernel("softmax_buf", "softmax_v4_buf", {"-DSOFTMAX_LOCAL_SIZE=64"},
                                          mOpenCLBackend->getPrecision());
    mDecodeSoftmaxKernel = runtime->buildKernel("softmax_buf", "softmax_in1_buf", {"-DSOFTMAX_LOCAL_SIZE=64"},
                                                mOpenCLBackend->getPrecision());
    mZeroKernel = runtime->buildKernel("paged_attention_buf", "zero_output", {}, mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL_CTOR(mCopyKernel);
    OPENCL_CHECK_KERNEL_CTOR(mAttentionKernel);
    OPENCL_CHECK_KERNEL_CTOR(mAttentionRowKernel);
    OPENCL_CHECK_KERNEL_CTOR(mPackPagedKVKernel);
    OPENCL_CHECK_KERNEL_CTOR(mPackPagedKeyKernel);
    OPENCL_CHECK_KERNEL_CTOR(mPackPagedKeyToImageKernel);
    OPENCL_CHECK_KERNEL_CTOR(mHydrateExternalKernel);
    OPENCL_CHECK_KERNEL_CTOR(mHydrateExternalInplaceKernel);
    OPENCL_CHECK_KERNEL_CTOR(mExportCanonicalKeyKernel);
    OPENCL_CHECK_KERNEL_CTOR(mCacheBlendScoreKernel);
    OPENCL_CHECK_KERNEL_CTOR(mCacheBlendScoreImageKernel);
    OPENCL_CHECK_KERNEL_CTOR(mCacheBlendTopKKernel);
    OPENCL_CHECK_KERNEL_CTOR(mCacheBlendTopKStage1Kernel);
    OPENCL_CHECK_KERNEL_CTOR(mCacheBlendTopKStage2Kernel);
    OPENCL_CHECK_KERNEL_CTOR(mRearrangeQKernel);
    OPENCL_CHECK_KERNEL_CTOR(mRearrangeSparseQKernel);
    OPENCL_CHECK_KERNEL_CTOR(mRearrangeMaskKernel);
    OPENCL_CHECK_KERNEL_CTOR(mSoftmaxKernel);
    OPENCL_CHECK_KERNEL_CTOR(mDecodeSoftmaxKernel);
    OPENCL_CHECK_KERNEL_CTOR(mZeroKernel);
}

PagedAttentionBufExecution::~PagedAttentionBufExecution() {
    _unregisterOpenCLPagedDecodePrepare(mMeta, this);
}

ErrorCode PagedAttentionBufExecution::ensureCache(int maxSlots, int batch, int kvHeads, int headDim) {
    if (maxSlots <= 0 || batch <= 0 || kvHeads <= 0 || headDim <= 0) {
        return INVALID_VALUE;
    }
    const std::string registryKey =
        _openCLPagedCacheRegistryKey(mOpenCLBackend, mMeta, mLayerIndex, mKVSharedLayerIndex);
    if (!_openCLPagedCacheCompatible(mCache, maxSlots, batch, kvHeads, headDim, mBytes)) {
        auto sharedCache = _findOpenCLPagedCache(registryKey, maxSlots, batch, kvHeads, headDim, mBytes);
        if (sharedCache) {
            mCache = sharedCache;
        }
    }
    if (mCache && mCache->key && mCache->value && mCache->sparseQuery &&
        mCache->decodeKey &&
        mCache->maxSlots >= maxSlots && mCache->batch == batch && mCache->kvHeads == kvHeads &&
        mCache->headDim == headDim && mCache->bytes == mBytes) {
        if (_picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA cache reuse layer=%d cache=%p required_slots=%d max_slots=%d ready=%d\n",
                      mLayerIndex, static_cast<void*>(mCache.get()), maxSlots, mCache->maxSlots,
                      mCache->decodeKeyReadyLength);
        }
        if (!_legacyB863976OpenCL() && mLayerIndex >= 0) {
            _registerExternalLayerMappedTarget(mMeta, mLayerIndex, batch, kvHeads, headDim, mBytes,
                                               mCache->maxSlots,
                                               mCache->key, mCache->value,
                                               mOpenCLBackend->getOpenCLRuntime()->commandQueue(),
                                               _useAdrenoSourceSlotValueHydrate(mOpenCLBackend->getOpenCLRuntime()));
        }
        return NO_ERROR;
    }
    if (!mCache) {
        mCache.reset(new SharedPagedCache);
    }
    if (mBytes == 4) {
        mCache->key.reset(Tensor::createDevice<float>({maxSlots, batch, kvHeads, headDim}));
        mCache->decodeKey.reset(Tensor::createDevice<float>({batch, kvHeads, headDim, maxSlots}));
        mCache->value.reset(Tensor::createDevice<float>({batch, kvHeads, maxSlots, headDim}));
    } else {
        mCache->key.reset(Tensor::createDevice<uint16_t>({maxSlots, batch, kvHeads, headDim}));
        mCache->decodeKey.reset(Tensor::createDevice<uint16_t>({batch, kvHeads, headDim, maxSlots}));
        mCache->value.reset(Tensor::createDevice<uint16_t>({batch, kvHeads, maxSlots, headDim}));
    }
    mCache->sparseQuery.reset(Tensor::createDevice<int>({maxSlots}));
    if (!mCache->key || !mCache->decodeKey || !mCache->value || !mCache->sparseQuery) {
        return OUT_OF_MEMORY;
    }
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mCache->key.get(), Backend::STATIC));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mCache->decodeKey.get(), Backend::STATIC));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mCache->value.get(), Backend::STATIC));
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
    ret |= mZeroKernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
    ret |= mZeroKernel->get().setArg(idx++, static_cast<int>(keyElements));
    MNN_CHECK_CL_SUCCESS(ret, "setArg zero_decode_key_cache");
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
    mCache->decodeKeyReadyLength = 0;
    mCache->sparseQueryHost.clear();
    _storeOpenCLPagedCache(registryKey, mCache);
    if (_picOpenCLDebug()) {
        MNN_PRINT("PIC OpenCL PA cache create layer=%d cache=%p max_slots=%d batch=%d kv_heads=%d dim=%d bytes=%d\n",
                  mLayerIndex, static_cast<void*>(mCache.get()), maxSlots, batch, kvHeads, headDim, mBytes);
    }
    if (!_legacyB863976OpenCL() && mLayerIndex >= 0) {
        _registerExternalLayerMappedTarget(mMeta, mLayerIndex, batch, kvHeads, headDim, mBytes, maxSlots,
                                           mCache->key, mCache->value,
                                           mOpenCLBackend->getOpenCLRuntime()->commandQueue(),
                                           _useAdrenoSourceSlotValueHydrate(mOpenCLBackend->getOpenCLRuntime()));
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::syncSparseQuery(int attnLen) {
    if (attnLen <= 0 || mMeta == nullptr || !mMeta->sparse_query_active) {
        return NO_ERROR;
    }
    if (!mCache || !mCache->sparseQuery || attnLen > mCache->maxSlots) {
        return OUT_OF_MEMORY;
    }
    if (static_cast<int>(mMeta->sparse_query_logical_indices.size()) < attnLen) {
        return INVALID_VALUE;
    }
    const int* hostPtr = mMeta->sparse_query_logical_indices.data();
    if (static_cast<int>(mCache->sparseQueryHost.size()) == attnLen &&
        std::equal(mCache->sparseQueryHost.begin(), mCache->sparseQueryHost.end(), hostPtr)) {
        return NO_ERROR;
    }
    auto ret = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueWriteBuffer(
        openCLBuffer(mCache->sparseQuery.get()), CL_TRUE, 0, static_cast<size_t>(attnLen) * sizeof(int),
        hostPtr);
    if (ret != CL_SUCCESS) {
        mCache->sparseQueryHost.clear();
        return INVALID_VALUE;
    }
    mCache->sparseQueryHost.assign(hostPtr, hostPtr + attnLen);
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::syncDecodeAttentionHeadIds() {
    if (mMeta == nullptr || mMeta->pic_decode_attention_head_ids.empty()) {
        mDecodeAttentionHeadIdsHost.clear();
        return NO_ERROR;
    }
    const auto& heads = mMeta->pic_decode_attention_head_ids;
    if (mDecodeAttentionHeadIds == nullptr ||
        mDecodeAttentionHeadIdCapacity < static_cast<int>(heads.size())) {
        mDecodeAttentionHeadIds.reset(Tensor::createDevice<int>({static_cast<int>(heads.size())}));
        if (!mDecodeAttentionHeadIds) {
            return OUT_OF_MEMORY;
        }
        OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mDecodeAttentionHeadIds.get(), Backend::STATIC));
        mDecodeAttentionHeadIdCapacity = static_cast<int>(heads.size());
        mDecodeAttentionHeadIdsHost.clear();
    }
    if (mDecodeAttentionHeadIdsHost != heads) {
        auto ret = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueWriteBuffer(
            openCLBuffer(mDecodeAttentionHeadIds.get()), CL_TRUE, 0, heads.size() * sizeof(int), heads.data());
        if (ret != CL_SUCCESS) {
            mDecodeAttentionHeadIdsHost.clear();
            return INVALID_VALUE;
        }
        mDecodeAttentionHeadIdsHost = heads;
    }
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

static ErrorCode _emitDecodeRecomputeIndicesOpenCL(PagedKVMeta* meta, int count, Tensor* output,
                                                   OpenCLBackend* backend) {
    if (output == nullptr) {
        return NO_ERROR;
    }
    count = std::min(count, static_cast<int>(output->elementSize()));
    if (count <= 0) {
        return NO_ERROR;
    }
    if (meta == nullptr || static_cast<int>(meta->sparse_query_logical_indices.size()) < count) {
        return INVALID_VALUE;
    }
    return _emitIdentityIndicesOpenCL(count, output, backend);
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

ErrorCode PagedAttentionBufExecution::ensureDecodeCausalKernel() {
    if (mHeadDim != 64 || mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    const int groupSize = mNumHead / mKvNumHead;
    if (mDecodeCausalKernel32 && mDecodeCausalKernel64 && mDecodeCausalKernelGroupSize == groupSize) {
        return NO_ERROR;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    mDecodeCausalKernel32 = runtime->buildKernel("attention_buf", "decode_causal_attention_row32",
                                                 {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                                                 mOpenCLBackend->getPrecision());
    mDecodeCausalKernel64 = runtime->buildKernel("attention_buf", "decode_causal_attention_row64",
                                                 {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                                                 mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL(mDecodeCausalKernel32);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernel64);
    mDecodeCausalKernelGroupSize = groupSize;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureDecodeCausalKernelHD128Identity() {
    if (mHeadDim != 128 || mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    const int groupSize = mNumHead / mKvNumHead;
    const bool needGqaFusedKV = groupSize > 1 && _decodeGqaFusedKVEnabled();
    if (mDecodeCausalKernelHD128IdentityRow32 && mDecodeCausalKernelHD128IdentityRow64 &&
        mDecodeCausalKernelHD128IdentityRow128 &&
        mDecodeCausalKernelHD128IdentityFusedKVRow32 && mDecodeCausalKernelHD128IdentityFusedKVRow64 &&
        mDecodeCausalKernelHD128IdentityFusedKVRow128 &&
        mDecodeQKIdentityKernel &&
        (!needGqaFusedKV ||
         (mDecodeCausalKernelHD128IdentityFusedKVGQARow32 &&
          mDecodeCausalKernelHD128IdentityFusedKVGQARow64 &&
          mDecodeCausalKernelHD128IdentityFusedKVGQARow128)) &&
        mDecodeCausalHD128IdentityKernelGroupSize == groupSize) {
        return NO_ERROR;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    mDecodeCausalKernelHD128IdentityRow32 =
        runtime->buildKernel("paged_decode_attention_buf", "decode_causal_attention_hd128_identity_row32",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128IdentityRow64 =
        runtime->buildKernel("paged_decode_attention_buf", "decode_causal_attention_hd128_identity_row64",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128IdentityRow128 =
        runtime->buildKernel("paged_decode_attention_buf", "decode_causal_attention_hd128_identity_row128",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128IdentityFusedKVRow32 =
        runtime->buildKernel("paged_decode_attention_buf", "decode_causal_attention_hd128_identity_fused_kv_row32",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128IdentityFusedKVRow64 =
        runtime->buildKernel("paged_decode_attention_buf", "decode_causal_attention_hd128_identity_fused_kv_row64",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128IdentityFusedKVRow128 =
        runtime->buildKernel("paged_decode_attention_buf", "decode_causal_attention_hd128_identity_fused_kv_row128",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    mDecodeQKIdentityKernel =
        runtime->buildKernel("paged_decode_attention_buf", "matmul_qk_decode_paged_identity_hd128",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    if (needGqaFusedKV) {
        mDecodeCausalKernelHD128IdentityFusedKVGQARow32 =
            runtime->buildKernel("paged_decode_attention_buf", "decode_causal_attention_hd128_identity_fused_kv_gqa_row32",
                                 {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                                 mOpenCLBackend->getPrecision());
        mDecodeCausalKernelHD128IdentityFusedKVGQARow64 =
            runtime->buildKernel("paged_decode_attention_buf", "decode_causal_attention_hd128_identity_fused_kv_gqa_row64",
                                 {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                                 mOpenCLBackend->getPrecision());
        mDecodeCausalKernelHD128IdentityFusedKVGQARow128 =
            runtime->buildKernel("paged_decode_attention_buf", "decode_causal_attention_hd128_identity_fused_kv_gqa_row128",
                                 {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                                 mOpenCLBackend->getPrecision());
    } else {
        mDecodeCausalKernelHD128IdentityFusedKVGQARow32.reset();
        mDecodeCausalKernelHD128IdentityFusedKVGQARow64.reset();
        mDecodeCausalKernelHD128IdentityFusedKVGQARow128.reset();
    }
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128IdentityRow32);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128IdentityRow64);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128IdentityRow128);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128IdentityFusedKVRow32);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128IdentityFusedKVRow64);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128IdentityFusedKVRow128);
    OPENCL_CHECK_KERNEL(mDecodeQKIdentityKernel);
    if (needGqaFusedKV) {
        OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128IdentityFusedKVGQARow32);
        OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128IdentityFusedKVGQARow64);
        OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128IdentityFusedKVGQARow128);
    }
    mDecodeCausalHD128IdentityKernelGroupSize = groupSize;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureDecodeTransposedKKernel() {
    if (mHeadDim != 128 || mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    const int groupSize = mNumHead / mKvNumHead;
    if (mDecodeKeyTransposeKernel && mDecodeKeyAppendKernel && mDecodeKeyAppendSparseKernel &&
        mDecodeQKTransposedKernel &&
        mDecodeQKVTransposedKernel && mDecodeCausalKernelHD128TransposedKFusedKVRow32 &&
        mDecodeCausalKernelHD128TransposedKFusedKVRow64 &&
        mDecodeCausalKernelHD128TransposedKFusedKVRow128 &&
        mDecodeCausalKernelHD128TransposedKReadonlyRow32 &&
        mDecodeCausalKernelHD128TransposedKReadonlyRow64 &&
        mDecodeCausalKernelHD128TransposedKReadonlyRow128 &&
        mDecodeCausalKernelHD128TransposedKSparseRow32 &&
        mDecodeCausalKernelHD128TransposedKSparseRow64 &&
        mDecodeCausalKernelHD128TransposedKSparseRow128 &&
        mDecodeCausalKernelHD128TransposedKQTile2Row32 &&
        mDecodeCausalKernelHD128TransposedKQTile2Row64 &&
        mDecodeCausalKernelHD128TransposedKQTile2Row128 &&
        mDecodeCausalKernelHD128TransposedKQTile4Row32 &&
        mDecodeCausalKernelHD128TransposedKQTile4Row64 &&
        mDecodeCausalKernelHD128TransposedKQTile4Row128 &&
        mDecodeCausalKernelHD128TransposedKQTile8Row32 &&
        mDecodeCausalKernelHD128TransposedKQTile8Row64 &&
        mDecodeCausalKernelHD128TransposedKQTile8Row128 &&
        (groupSize <= 1 ||
         (mDecodeCausalKernelHD128TransposedKFusedKVGQARow32 &&
          mDecodeCausalKernelHD128TransposedKFusedKVGQARow64 &&
          mDecodeCausalKernelHD128TransposedKFusedKVGQARow128)) &&
        mDecodeTransposedKKernelGroupSize == groupSize) {
        return NO_ERROR;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const std::set<std::string> options = {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)};
    mDecodeKeyTransposeKernel =
        runtime->buildKernel("paged_decode_attention_buf", "transpose_paged_key_to_decode_key", options,
                             mOpenCLBackend->getPrecision());
    mDecodeKeyAppendKernel =
        runtime->buildKernel("paged_decode_attention_buf", "append_decode_key_value_hd128", options,
                             mOpenCLBackend->getPrecision());
    mDecodeKeyAppendSparseKernel =
        runtime->buildKernel("paged_decode_attention_buf", "append_sparse_decode_key_value_hd128", options,
                             mOpenCLBackend->getPrecision());
    mDecodeQKTransposedKernel =
        runtime->buildKernel("paged_decode_attention_buf", "matmul_qk_decode_transposed_hd128", options,
                             mOpenCLBackend->getPrecision());
    mDecodeQKVTransposedKernel =
        runtime->buildKernel("paged_decode_attention_buf", "matmul_qkv_decode_value_hd128_b8", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKFusedKVRow32 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_fused_kv_row32", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKFusedKVRow64 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_fused_kv_row64", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKFusedKVRow128 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_fused_kv_row128", options,
                             mOpenCLBackend->getPrecision());
    if (groupSize > 1) {
        mDecodeCausalKernelHD128TransposedKFusedKVGQARow32 =
            runtime->buildKernel("paged_decode_attention_buf",
                                 "decode_causal_attention_hd128_transposed_k_fused_kv_gqa_row32", options,
                                 mOpenCLBackend->getPrecision());
        mDecodeCausalKernelHD128TransposedKFusedKVGQARow64 =
            runtime->buildKernel("paged_decode_attention_buf",
                                 "decode_causal_attention_hd128_transposed_k_fused_kv_gqa_row64", options,
                                 mOpenCLBackend->getPrecision());
        mDecodeCausalKernelHD128TransposedKFusedKVGQARow128 =
            runtime->buildKernel("paged_decode_attention_buf",
                                 "decode_causal_attention_hd128_transposed_k_fused_kv_gqa_row128", options,
                                 mOpenCLBackend->getPrecision());
    } else {
        mDecodeCausalKernelHD128TransposedKFusedKVGQARow32 = nullptr;
        mDecodeCausalKernelHD128TransposedKFusedKVGQARow64 = nullptr;
        mDecodeCausalKernelHD128TransposedKFusedKVGQARow128 = nullptr;
    }
    mDecodeCausalKernelHD128TransposedKReadonlyRow32 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_readonly_row32", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKReadonlyRow64 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_readonly_row64", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKReadonlyRow128 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_readonly_row128", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKSparseRow32 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_sparse_row32", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKSparseRow64 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_sparse_row64", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKSparseRow128 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_sparse_row128", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile2Row32 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q2_row32", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile2Row64 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q2_row64", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile2Row128 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q2_row128", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile4Row32 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q4_row32", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile4Row64 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q4_row64", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile4Row128 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q4_row128", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile8Row32 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q8_row32", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile8Row64 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q8_row64", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile8Row128 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q8_row128", options,
                             mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL(mDecodeKeyTransposeKernel);
    OPENCL_CHECK_KERNEL(mDecodeKeyAppendKernel);
    OPENCL_CHECK_KERNEL(mDecodeKeyAppendSparseKernel);
    OPENCL_CHECK_KERNEL(mDecodeQKTransposedKernel);
    OPENCL_CHECK_KERNEL(mDecodeQKVTransposedKernel);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKFusedKVRow32);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKFusedKVRow64);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKFusedKVRow128);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKReadonlyRow32);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKReadonlyRow64);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKReadonlyRow128);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKSparseRow32);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKSparseRow64);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKSparseRow128);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile2Row32);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile2Row64);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile2Row128);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile4Row32);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile4Row64);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile4Row128);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile8Row32);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile8Row64);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile8Row128);
    mDecodeTransposedKKernelGroupSize = groupSize;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureDecodeTransposedTemps(int kvLen) {
    if (kvLen <= 0 || mBatch <= 0 || mNumHead <= 0) {
        return INVALID_VALUE;
    }
    if (!(mTempQK && mTempSoftmax && mDecodeTransposedTempKvLen >= kvLen)) {
        const int kvPack = ROUND_UP(kvLen, 8);
        const int elements = kvPack * mNumHead * mBatch;
        mTempQK.reset(Tensor::createDevice<float>({elements}));
        mTempSoftmax.reset(Tensor::createDevice<float>({elements}));
        if (!mTempQK || !mTempSoftmax) {
            return OUT_OF_MEMORY;
        }
        mDecodeTransposedTempKvLen = kvPack;
    }
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempQK.get(), Backend::DYNAMIC_IN_EXECUTION));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempSoftmax.get(), Backend::DYNAMIC_IN_EXECUTION));
    mOpenCLBackend->onReleaseBuffer(mTempQK.get(), Backend::DYNAMIC_IN_EXECUTION);
    mOpenCLBackend->onReleaseBuffer(mTempSoftmax.get(), Backend::DYNAMIC_IN_EXECUTION);
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureDecodeKeyReady(int requiredLen, bool insideDecode) {
    mLastDecodeKeyRequiredLen = requiredLen;
    mLastDecodeKeyPrepareUs = 0;
    mLastDecodeKeyReadyHit = false;
    mLastDecodeKeyPrepared = false;
    mLastDecodeKeyPrepareInsideDecode = false;
    mLastDecodeKeySlotIdentity = false;
    mLastDecodeKeyPrefixStable = false;
    if (requiredLen <= 0) {
        mLastDecodeKeyReadyHit = true;
        mLastDecodeKeyPrefixStable = true;
        return NO_ERROR;
    }
    if (!mCache || !mCache->key || !mCache->decodeKey ||
        requiredLen > mCache->maxSlots) {
        return OUT_OF_MEMORY;
    }
    mLastDecodeKeySlotIdentity = true;
    mLastDecodeKeyPrefixStable = true;
    if (mCache->decodeKeyReadyLength >= requiredLen) {
        mLastDecodeKeyReadyHit = true;
        if (_picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA decode_key ready hit layer=%d cache=%p required=%d ready=%d\n",
                      mLayerIndex, static_cast<void*>(mCache.get()), requiredLen,
                      mCache->decodeKeyReadyLength);
        }
        return NO_ERROR;
    }
    if (_picOpenCLDebug()) {
        MNN_PRINT("PIC OpenCL PA decode_key prepare layer=%d cache=%p required=%d ready=%d max_slots=%d\n",
                  mLayerIndex, static_cast<void*>(mCache.get()), requiredLen, mCache->decodeKeyReadyLength,
                  mCache->maxSlots);
    }
    if (insideDecode) {
        mLastDecodeKeyPrepareInsideDecode = true;
        if (_picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA decode_key prepare blocked inside decode layer=%d required=%d ready=%d "
                      "prefix_stable=1\n",
                      mLayerIndex, requiredLen, mCache->decodeKeyReadyLength);
        }
        return INVALID_VALUE;
    }
    auto err = ensureDecodeTransposedKKernel();
    if (err != NO_ERROR) {
        return err;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto& queue = runtime->commandQueue();
    const bool profile = _profilePagedAttention();
    const uint64_t startUs = profile ? _nowUs() : 0;
    std::vector<uint32_t> gws = {
        static_cast<uint32_t>(requiredLen),
        static_cast<uint32_t>(mBatch * mKvNumHead * mHeadDim),
    };
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mDecodeKeyTransposeKernel->get().setArg(idx++, gws[0]);
    ret |= mDecodeKeyTransposeKernel->get().setArg(idx++, gws[1]);
    ret |= mDecodeKeyTransposeKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= mDecodeKeyTransposeKernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
    ret |= mDecodeKeyTransposeKernel->get().setArg(idx++, mBatch);
    ret |= mDecodeKeyTransposeKernel->get().setArg(idx++, mKvNumHead);
    ret |= mDecodeKeyTransposeKernel->get().setArg(idx++, mHeadDim);
    ret |= mDecodeKeyTransposeKernel->get().setArg(idx++, requiredLen);
    ret |= mDecodeKeyTransposeKernel->get().setArg(idx++, mCache->maxSlots);
    MNN_CHECK_CL_SUCCESS(ret, "setArg transpose_paged_key_to_decode_key");
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const int layerIndex = mLayerIndex >= 0 ? mLayerIndex : (mMeta != nullptr ? mMeta->layer_index : -1);
    cl_int enqueueRet = CL_SUCCESS;
    const auto pmcMeta = _makeOpenCLPmcMeta("decode_prepare_transpose_k", "prepare", layerIndex,
                                           requiredLen, mQuerySeqLen, requiredLen, -1, 0, 0,
                                           mNumHead, mKvNumHead, mHeadDim, gws, {0u, 0u},
                                           0, 0, 0, requiredLen, insideDecode);
    _runOpenCLPmcScope(runtime, pmcMeta, [&](cl::Event* eventPtr) {
        enqueueRet = queue.enqueueNDRangeKernel(mDecodeKeyTransposeKernel->get(), cl::NullRange,
                                                cl::NDRange(gws[0], gws[1]), cl::NullRange,
                                                nullptr, eventPtr);
    });
    MNN_CHECK_CL_SUCCESS(enqueueRet, "enqueue transpose_paged_key_to_decode_key");
    if (enqueueRet != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    mCache->decodeKeyReadyLength = requiredLen;
    mLastDecodeKeyPrefixStable = true;
    mLastDecodeKeyPrepared = true;
    if (profile) {
        queue.finish();
        mLastDecodeKeyPrepareUs = _nowUs() - startUs;
        MNN_PRINT("OpenCLPagedAttention profile op=decode_prepare_transpose_k layer=%d kv_len=%d "
                  "inside_decode=%d slot_identity=%d prefix_stable=%d us=%llu prepare_us=%llu\n",
                  layerIndex, requiredLen, insideDecode ? 1 : 0, 1,
                  mLastDecodeKeyPrefixStable ? 1 : 0,
                  static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                  static_cast<unsigned long long>(mLastDecodeKeyPrepareUs));
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureDecodeAttentionRankKernel() {
    if (mHeadDim != 128 || mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    const int groupSize = mNumHead / mKvNumHead;
    if (mDecodeAttentionRankScoreKernelHD128 &&
        mDecodeAttentionRankKernelGroupSize == groupSize) {
        return NO_ERROR;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    mDecodeAttentionRankScoreKernelHD128 =
        runtime->buildKernel("attention_buf", "decode_attention_pic_rank_score_hd128",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL(mDecodeAttentionRankScoreKernelHD128);
    mDecodeAttentionRankKernelGroupSize = groupSize;
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

ErrorCode PagedAttentionBufExecution::ensureCacheBlendScoreTemps(int scoreCount, int indexCount,
                                                                 int stageCandidateCount) {
    if (scoreCount < 0 || indexCount < 0) {
        return INVALID_VALUE;
    }
    if (stageCandidateCount < 0) {
        return INVALID_VALUE;
    }
    if (scoreCount == 0 && indexCount == 0 && stageCandidateCount == 0) {
        return NO_ERROR;
    }
    if (mCacheBlendScores && mCacheBlendIndices && mCacheBlendScoreCount >= scoreCount &&
        mCacheBlendIndexCount >= indexCount &&
        (stageCandidateCount == 0 ||
         (mCacheBlendStageValues && mCacheBlendStageIndices &&
          mCacheBlendStageCandidateCount >= stageCandidateCount))) {
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
    if (stageCandidateCount > 0) {
        mCacheBlendStageValues.reset(Tensor::createDevice<float>({stageCandidateCount}));
        if (!mCacheBlendStageValues) {
            return OUT_OF_MEMORY;
        }
        OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mCacheBlendStageValues.get(), Backend::STATIC));
        mCacheBlendStageIndices.reset(Tensor::createDevice<int>({stageCandidateCount}));
        if (!mCacheBlendStageIndices) {
            return OUT_OF_MEMORY;
        }
        OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mCacheBlendStageIndices.get(), Backend::STATIC));
    }
    mCacheBlendScoreCount = scoreCount;
    mCacheBlendIndexCount = indexCount;
    mCacheBlendStageCandidateCount = stageCandidateCount;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureAdrenoCacheBlendValueImage(int tokenCapacity) {
    if (tokenCapacity <= 0 || mBatch <= 0 || mKvNumHead <= 0 || mHeadDim <= 0) {
        return INVALID_VALUE;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (runtime == nullptr) {
        return INVALID_VALUE;
    }
    int imageWidth = 0;
    int imageHeight = 0;
    if (!_computeAdrenoCacheBlendValueImageShape(runtime, mBatch, mKvNumHead, tokenCapacity, mHeadDim,
                                                 &imageWidth, &imageHeight)) {
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
    if (mCacheBlendSourceValueImage != nullptr && mCacheBlendSourceValueTokenCapacity >= tokenCapacity &&
        mCacheBlendSourceValueImageHeight == imageHeight && mCacheBlendSourceValueBytes == mBytes) {
        return NO_ERROR;
    }
    if (!mAdrenoImagePool) {
        mAdrenoImagePool.reset(new ImagePool(runtime->context()));
    }
    auto image = mAdrenoImagePool->alloc(imageWidth, imageHeight, mOpenCLBackend->fpType());
    if (image == nullptr) {
        return OUT_OF_MEMORY;
    }
    mCacheBlendSourceValueImage = image;
    mCacheBlendSourceValueTokenCapacity = tokenCapacity;
    mCacheBlendSourceValueImageWidth = imageWidth;
    mCacheBlendSourceValueImageHeight = imageHeight;
    mCacheBlendSourceValueBytes = mBytes;
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
                                           mCache->maxSlots, mCache->key, mCache->value, queue,
                                           _useAdrenoSourceSlotValueHydrate(mOpenCLBackend->getOpenCLRuntime()));
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
        sourceSlotCursor = _alignPicCacheSourceSlots(sourceSlotCursor);
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
        int valueSourceStart = 0;
        int valueSourceStride = static_cast<int>(segment.tokenCount);
        int hydrateValue = 1;
        if (loadedSegment != nullptr && loadedSegment->directWritten) {
            sourceKeyBuffer = &openCLBuffer(mCache->key.get());
            sourceValueBuffer = &openCLBuffer(mCache->value.get());
            keySourceLogicalStart = loadedSegment->sourceSlotStart;
            valueSourceStart = loadedSegment->valueSourceStart;
            valueSourceStride = loadedSegment->valueSourceStride > 0
                ? loadedSegment->valueSourceStride
                : static_cast<int>(segment.tokenCount);
            hydrateValue = loadedSegment->valueAlreadyHydrated ? 0 : 1;
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
        if (loadedSegment != nullptr && loadedSegment->directWritten) {
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, mBatch);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, mKvNumHead);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, mHeadDim);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, mCache->maxSlots);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, static_cast<int>(segment.logicalStart));
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, static_cast<int>(segment.tokenCount));
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, keySourceLogicalStart);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, valueSourceStart);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, valueSourceStride);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, hydrateValue);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, ropeDim);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, segment.ropeTheta);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, ropeTypeLlama3);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, segment.ropeScalingFactor);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, segment.ropeScalingLowFreqFactor);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, segment.ropeScalingHighFreqFactor);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, oldContext);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, segment.maxPositionEmbeddings);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, segment.ropeAttentionScaling);
            ret |= mHydrateExternalInplaceKernel->get().setArg(idx++, total);
            MNN_CHECK_CL_SUCCESS(ret, "setArg pic_page_attention_hydrate_kv_inplace");
            queue.enqueueNDRangeKernel(mHydrateExternalInplaceKernel->get(), cl::NullRange, cl::NDRange(total),
                                       cl::NullRange);
        } else {
            ret |= mHydrateExternalKernel->get().setArg(idx++, *sourceKeyBuffer);
            ret |= mHydrateExternalKernel->get().setArg(idx++, *sourceValueBuffer);
            ret |= mHydrateExternalKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
            ret |= mHydrateExternalKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
            ret |= mHydrateExternalKernel->get().setArg(idx++, mBatch);
            ret |= mHydrateExternalKernel->get().setArg(idx++, mKvNumHead);
            ret |= mHydrateExternalKernel->get().setArg(idx++, mHeadDim);
            ret |= mHydrateExternalKernel->get().setArg(idx++, mCache->maxSlots);
            ret |= mHydrateExternalKernel->get().setArg(idx++, static_cast<int>(segment.logicalStart));
            ret |= mHydrateExternalKernel->get().setArg(idx++, static_cast<int>(segment.tokenCount));
            ret |= mHydrateExternalKernel->get().setArg(idx++, keySourceLogicalStart);
            ret |= mHydrateExternalKernel->get().setArg(idx++, valueSourceStart);
            ret |= mHydrateExternalKernel->get().setArg(idx++, valueSourceStride);
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
            queue.enqueueNDRangeKernel(mHydrateExternalKernel->get(), cl::NullRange, cl::NDRange(total),
                                       cl::NullRange);
        }
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

ErrorCode PagedAttentionBufExecution::prepareDecodePrefix(int kvLen) {
    if (mMeta == nullptr || kvLen <= 0) {
        return NO_ERROR;
    }
    if (mBatch <= 0 || mKvNumHead <= 0 || mHeadDim <= 0) {
        return NOT_SUPPORT;
    }
    const int layerIndex = mLayerIndex >= 0 ? mLayerIndex : mMeta->layer_index;
    if (layerIndex < 0) {
        return NOT_SUPPORT;
    }
    int maxSlots = std::max(kvLen, mMeta->request_capacity > 0 ? mMeta->request_capacity : mMeta->max_tokens);
    if (maxSlots <= 0) {
        maxSlots = kvLen;
    }
    const size_t sourceSlots = _picCacheSourceSlotCount(mMeta);
    if (sourceSlots > 0) {
        const int sourceBase = _picCacheSourceSlotBase(mMeta, maxSlots);
        if (sourceSlots > static_cast<size_t>(std::numeric_limits<int>::max() - sourceBase)) {
            return OUT_OF_MEMORY;
        }
        maxSlots = std::max(maxSlots, sourceBase + static_cast<int>(sourceSlots));
    }
    auto err = ensureCache(maxSlots, mBatch, mKvNumHead, mHeadDim);
    if (err != NO_ERROR) {
        return err;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto& queue = runtime->commandQueue();
    if (!_legacyB863976OpenCL()) {
        _registerExternalLayerMappedTarget(mMeta, layerIndex, mBatch, mKvNumHead, mHeadDim, mBytes,
                                           mCache->maxSlots, mCache->key, mCache->value, queue,
                                           _useAdrenoSourceSlotValueHydrate(runtime));
    }
    if (!_legacyB863976OpenCL() && mMeta->shouldHydrateExternalLayer(layerIndex)) {
        _scheduleExternalLayerReadsFrom(mMeta, std::max(0, mMeta->external_hydrate_start_layer_idx),
                                        mBatch, mKvNumHead, mHeadDim, mBytes, kvLen);
    }
    if (mMeta->shouldHydrateExternalLayer(layerIndex) && !mMeta->externalLayerLoaded(layerIndex)) {
        err = hydrateExternalSegments(layerIndex, kvLen);
        if (err != NO_ERROR) {
            return err;
        }
    }
    const bool q1DecodeKNeedsPrepare = _decodeTransposedKQ1NeedsDecodeKey();
    const bool repairDecodeKNeedsPrepare =
        mMeta->pic_decode_repair_tokens_per_step > 0 &&
        _decodeTransposedKSparseShapeSupported(runtime, mMeta->pic_decode_repair_tokens_per_step + 1);
    const bool needsDecodeKey =
        _decodeTransposedKEnabled() && mHeadDim == 128 &&
        (q1DecodeKNeedsPrepare || repairDecodeKNeedsPrepare);
    if (needsDecodeKey) {
        err = ensureDecodeKeyReady(kvLen, false);
        if (err != NO_ERROR) {
            return err;
        }
    }
    queue.finish();
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
    uint64_t imagePrepUs = 0;
    uint64_t scoreKernelUs = 0;
    uint64_t topKUs = 0;
    uint64_t readbackUs = 0;
    int imageSegments = 0;
    size_t imageTokens = 0;
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
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto topKFamilies = _cacheBlendTopKFamilyCandidates(runtime, mCacheBlendTopKStage1Kernel,
                                                        mCacheBlendTopKStage2Kernel, picTokenCount, topK);
    CacheBlendTopKDispatch topKDispatch;
    uint32_t topKFamilySource = kTuneSelectionSourceDefault;
    const uint32_t defaultTopKFamily =
        _cacheBlendTopKDispatchForFamily(runtime, mCacheBlendTopKStage1Kernel, mCacheBlendTopKStage2Kernel,
                                         picTokenCount, topK, kCacheBlendTopKFamilyStage2048, &topKDispatch)
        ? kCacheBlendTopKFamilyStage2048
        : (topKFamilies.empty() ? kCacheBlendTopKFamilyLegacy : topKFamilies.front());
    if (defaultTopKFamily != topKDispatch.family) {
        _cacheBlendTopKDispatchForFamily(runtime, mCacheBlendTopKStage1Kernel, mCacheBlendTopKStage2Kernel,
                                         picTokenCount, topK, defaultTopKFamily, &topKDispatch);
    }
    int stageCandidateCapacity = 0;
    for (uint32_t family : topKFamilies) {
        CacheBlendTopKDispatch candidateDispatch;
        if (_cacheBlendTopKDispatchForFamily(runtime, mCacheBlendTopKStage1Kernel, mCacheBlendTopKStage2Kernel,
                                             picTokenCount, topK, family, &candidateDispatch)) {
            stageCandidateCapacity = std::max(stageCandidateCapacity, candidateDispatch.stageCandidateCount);
        }
    }
    auto err = ensureCacheBlendScoreTemps(picTokenCount, topK, stageCandidateCapacity);
    if (err != NO_ERROR) {
        return err;
    }
    auto& queue = runtime->commandQueue();
    size_t scoreOffset = 0;
    int sourceSlotCursor = _picCacheSourceSlotBase(mMeta, kvLen);
    for (const auto& segment : mMeta->cacheblend_score_segments) {
        sourceSlotCursor = _alignPicCacheSourceSlots(sourceSlotCursor);
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
        const size_t valueTokenBytes = static_cast<size_t>(mHeadDim) * mBytes;
        const size_t valueSegmentBytes = static_cast<size_t>(mBatch) * mKvNumHead * segment.tokenCount *
                                         valueTokenBytes;
        auto& cacheValueBuffer = openCLBuffer(mCache->value.get());
        const bool preferImageScore = _useAdrenoCacheBlendValueImage(runtime, mBatch, mKvNumHead,
                                                                     static_cast<int>(segment.tokenCount), mHeadDim);
        bool usedImageScore = false;
        uint64_t opStartUs = profileDetail ? _nowUs() : 0;
        if (preferImageScore) {
            const size_t valueSegmentElements = valueSegmentBytes / static_cast<size_t>(mBytes);
            if (ensureExternalTemps(0, valueSegmentElements) == NO_ERROR &&
                ensureAdrenoCacheBlendValueImage(static_cast<int>(segment.tokenCount)) == NO_ERROR &&
                _readExternalValueSegmentToSourceCLBuffer(
                    layer->valuePath, openCLBuffer(mExternalValue.get()), valueSegmentBytes, queue, mBatch,
                    mKvNumHead, sourceTokenCount, sourceTokenOffset, segment.tokenCount, mHeadDim, mBytes)) {
                if (profileDetail) {
                    queue.finish();
                    readUs += _nowUs() - opStartUs;
                }
                uint32_t copyIdx = 0;
                cl_int copyRet = CL_SUCCESS;
                copyRet |= mCopyBufferToImageLinearKernel->get().setArg(copyIdx++, openCLBuffer(mExternalValue.get()));
                copyRet |= mCopyBufferToImageLinearKernel->get().setArg(copyIdx++, *mCacheBlendSourceValueImage);
                copyRet |= mCopyBufferToImageLinearKernel->get().setArg(copyIdx++, mCacheBlendSourceValueImageWidth);
                copyRet |= mCopyBufferToImageLinearKernel->get().setArg(copyIdx++, mCacheBlendSourceValueImageHeight);
                MNN_CHECK_CL_SUCCESS(copyRet, "setArg copy_buffer_to_image2d");
                if (copyRet == CL_SUCCESS) {
                    opStartUs = profileDetail ? _nowUs() : 0;
                    copyRet = queue.enqueueNDRangeKernel(mCopyBufferToImageLinearKernel->get(), cl::NullRange,
                                                         cl::NDRange(mCacheBlendSourceValueImageWidth,
                                                                     mCacheBlendSourceValueImageHeight),
                                                         cl::NullRange);
                    MNN_CHECK_CL_SUCCESS(copyRet, "enqueue copy_buffer_to_image2d");
                    if (copyRet == CL_SUCCESS) {
                        if (profileDetail) {
                            queue.finish();
                            imagePrepUs += _nowUs() - opStartUs;
                        }
                        uint32_t idx = 0;
                        cl_int ret = CL_SUCCESS;
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, *mCacheBlendSourceValueImage);
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, openCLBuffer(mCacheBlendScores.get()));
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, mBatch);
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, mKvNumHead);
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, mHeadDim);
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, mCache->maxSlots);
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, mCacheBlendSourceValueTokenCapacity);
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, mCacheBlendSourceValueImageWidth);
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, static_cast<int>(segment.logicalStart));
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, static_cast<int>(segment.tokenCount));
                        ret |= mCacheBlendScoreImageKernel->get().setArg(idx++, static_cast<int>(scoreOffset));
                        MNN_CHECK_CL_SUCCESS(ret, "setArg pic_cacheblend_value_score_cached_image");
                        if (ret == CL_SUCCESS) {
                            opStartUs = profileDetail ? _nowUs() : 0;
                            ret = queue.enqueueNDRangeKernel(mCacheBlendScoreImageKernel->get(), cl::NullRange,
                                                             cl::NDRange(static_cast<int>(segment.tokenCount)),
                                                             cl::NullRange);
                            MNN_CHECK_CL_SUCCESS(ret, "enqueue pic_cacheblend_value_score_cached_image");
                            if (ret == CL_SUCCESS) {
                                if (profileDetail) {
                                    queue.finish();
                                    scoreKernelUs += _nowUs() - opStartUs;
                                }
                                usedImageScore = true;
                                ++imageSegments;
                                imageTokens += segment.tokenCount;
                            }
                        }
                    }
                }
            }
        }
        if (!usedImageScore) {
            opStartUs = profileDetail ? _nowUs() : 0;
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
        }
        scoreOffset += segment.tokenCount;
    }
    if (scoreOffset != static_cast<size_t>(picTokenCount)) {
        return INVALID_VALUE;
    }
    auto runTopKFamily = [&](const CacheBlendTopKDispatch& dispatch) -> ErrorCode {
        cl_int localRet = CL_SUCCESS;
        if (!dispatch.useStage) {
            uint32_t idx = 0;
            localRet |= mCacheBlendTopKKernel->get().setArg(idx++, openCLBuffer(mCacheBlendScores.get()));
            localRet |= mCacheBlendTopKKernel->get().setArg(idx++, openCLBuffer(mCacheBlendIndices.get()));
            localRet |= mCacheBlendTopKKernel->get().setArg(idx++, picTokenCount);
            localRet |= mCacheBlendTopKKernel->get().setArg(idx++, topK);
            MNN_CHECK_CL_SUCCESS(localRet, "setArg pic_cacheblend_topk");
            if (localRet != CL_SUCCESS) {
                return INVALID_VALUE;
            }
            localRet = queue.enqueueNDRangeKernel(mCacheBlendTopKKernel->get(), cl::NullRange,
                                                  cl::NDRange(256), cl::NDRange(256));
            MNN_CHECK_CL_SUCCESS(localRet, "enqueue pic_cacheblend_topk");
            if (localRet != CL_SUCCESS) {
                return INVALID_VALUE;
            }
            return NO_ERROR;
        }
        uint32_t idx = 0;
        localRet |= mCacheBlendTopKStage1Kernel->get().setArg(idx++, openCLBuffer(mCacheBlendScores.get()));
        localRet |= mCacheBlendTopKStage1Kernel->get().setArg(idx++, openCLBuffer(mCacheBlendStageValues.get()));
        localRet |= mCacheBlendTopKStage1Kernel->get().setArg(idx++, openCLBuffer(mCacheBlendStageIndices.get()));
        localRet |= mCacheBlendTopKStage1Kernel->get().setArg(idx++, picTokenCount);
        localRet |= mCacheBlendTopKStage1Kernel->get().setArg(idx++, topK);
        localRet |= mCacheBlendTopKStage1Kernel->get().setArg(idx++, dispatch.blockSize);
        localRet |= mCacheBlendTopKStage1Kernel->get().setArg(
            idx++, cl::Local(static_cast<size_t>(dispatch.blockSize) * sizeof(float)));
        localRet |= mCacheBlendTopKStage1Kernel->get().setArg(
            idx++, cl::Local(static_cast<size_t>(dispatch.blockSize) * sizeof(int)));
        MNN_CHECK_CL_SUCCESS(localRet, "setArg pic_cacheblend_topk_stage1");
        if (localRet != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        localRet = queue.enqueueNDRangeKernel(mCacheBlendTopKStage1Kernel->get(), cl::NullRange,
                                              cl::NDRange(dispatch.blockCount * dispatch.stage1LocalSize),
                                              cl::NDRange(dispatch.stage1LocalSize));
        MNN_CHECK_CL_SUCCESS(localRet, "enqueue pic_cacheblend_topk_stage1");
        if (localRet != CL_SUCCESS) {
            return INVALID_VALUE;
        }

        idx = 0;
        localRet = CL_SUCCESS;
        localRet |= mCacheBlendTopKStage2Kernel->get().setArg(idx++, openCLBuffer(mCacheBlendStageValues.get()));
        localRet |= mCacheBlendTopKStage2Kernel->get().setArg(idx++, openCLBuffer(mCacheBlendStageIndices.get()));
        localRet |= mCacheBlendTopKStage2Kernel->get().setArg(idx++, openCLBuffer(mCacheBlendIndices.get()));
        localRet |= mCacheBlendTopKStage2Kernel->get().setArg(idx++, dispatch.stageCandidateCount);
        localRet |= mCacheBlendTopKStage2Kernel->get().setArg(idx++, topK);
        localRet |= mCacheBlendTopKStage2Kernel->get().setArg(idx++, dispatch.stageSortSize);
        localRet |= mCacheBlendTopKStage2Kernel->get().setArg(
            idx++, cl::Local(static_cast<size_t>(dispatch.stageSortSize) * sizeof(float)));
        localRet |= mCacheBlendTopKStage2Kernel->get().setArg(
            idx++, cl::Local(static_cast<size_t>(dispatch.stageSortSize) * sizeof(int)));
        MNN_CHECK_CL_SUCCESS(localRet, "setArg pic_cacheblend_topk_stage2");
        if (localRet != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        localRet = queue.enqueueNDRangeKernel(mCacheBlendTopKStage2Kernel->get(), cl::NullRange,
                                              cl::NDRange(dispatch.stage2LocalSize),
                                              cl::NDRange(dispatch.stage2LocalSize));
        MNN_CHECK_CL_SUCCESS(localRet, "enqueue pic_cacheblend_topk_stage2");
        if (localRet != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        return NO_ERROR;
    };
    if (_shouldTuneCacheBlendTopKFamily(picTokenCount, topK, profile, mOpenCLBackend->getCLTuneLevel()) &&
        !topKFamilies.empty()) {
        const std::string tuneKey =
            std::string("paged_cacheblend_topk_family_") + _openCLTuneDeviceKey(runtime) + "_" +
            std::to_string(mBatch) + "_" + std::to_string(mNumHead) + "_" +
            std::to_string(mKvNumHead) + "_" + std::to_string(mHeadDim);
        const std::vector<uint32_t> tuneShape = {
            static_cast<uint32_t>(picTokenCount),
            static_cast<uint32_t>(topK),
            static_cast<uint32_t>((static_cast<uint64_t>(topK) * 1000u) /
                                  static_cast<uint64_t>(std::max(1, picTokenCount))),
        };
        std::pair<std::vector<uint32_t>, uint32_t> tuneInfo;
        if (getTunedInfo(tuneKey, tuneShape, tuneInfo, runtime) && !tuneInfo.first.empty()) {
            CacheBlendTopKDispatch cachedDispatch;
            if (_cacheBlendTopKDispatchForFamily(runtime, mCacheBlendTopKStage1Kernel,
                                                 mCacheBlendTopKStage2Kernel, picTokenCount, topK,
                                                 tuneInfo.first[0], &cachedDispatch)) {
                topKDispatch = cachedDispatch;
                topKFamilySource = kTuneSelectionSourceCache;
            }
        } else if (topKFamilies.size() > 1) {
            bool tuned = false;
            uint64_t bestUs = std::numeric_limits<uint64_t>::max();
            CacheBlendTopKDispatch bestDispatch = topKDispatch;
            for (uint32_t family : topKFamilies) {
                CacheBlendTopKDispatch candidateDispatch;
                if (!_cacheBlendTopKDispatchForFamily(runtime, mCacheBlendTopKStage1Kernel,
                                                      mCacheBlendTopKStage2Kernel, picTokenCount, topK, family,
                                                      &candidateDispatch)) {
                    continue;
                }
                uint64_t candidateUs = 0;
                if (!_measureSteadyStateCandidate(runtime, [&]() {
                        return runTopKFamily(candidateDispatch);
                    }, &candidateUs)) {
                    continue;
                }
                if (!tuned || candidateUs < bestUs) {
                    tuned = true;
                    bestUs = candidateUs;
                    bestDispatch = candidateDispatch;
                }
            }
            if (tuned) {
                std::pair<std::vector<uint32_t>, uint32_t> bestInfo = std::make_pair(
                    std::vector<uint32_t>{bestDispatch.family},
                    static_cast<uint32_t>(std::min<uint64_t>(bestUs, std::numeric_limits<uint32_t>::max())));
                setTunedInfo(tuneKey, tuneShape, bestInfo, runtime, "attention_buf");
                topKDispatch = bestDispatch;
                topKFamilySource = kTuneSelectionSourceOnlineTuned;
            }
        }
        if (profile) {
            MNN_PRINT("OpenCLPagedAttention profile op=cacheblend_topk_family layer=%d pic_tokens=%d top_k=%d family=%s source=%s\n",
                      layerIndex, picTokenCount, topK, _cacheBlendTopKFamilyName(topKDispatch.family),
                      _tuneSelectionSourceName(topKFamilySource));
        }
    }
    uint64_t opStartUs = profileDetail ? _nowUs() : 0;
    err = runTopKFamily(topKDispatch);
    if (err != NO_ERROR && !topKDispatch.useStage) {
        return err;
    }
    if (err != NO_ERROR && topKDispatch.useStage) {
        MNN_ERROR("OpenCLPagedAttention cacheblend staged top-k failed, retrying legacy path layer=%d "
                  "pic_tokens=%d top_k=%d family=%s\n",
                  layerIndex, picTokenCount, topK, _cacheBlendTopKFamilyName(topKDispatch.family));
        CacheBlendTopKDispatch legacyDispatch;
        auto legacyErr = runTopKFamily(legacyDispatch);
        if (legacyErr != NO_ERROR) {
            return legacyErr;
        }
        topKDispatch = legacyDispatch;
        topKFamilySource = kTuneSelectionSourceDefault;
    }
    if (profileDetail) {
        queue.finish();
        topKUs += _nowUs() - opStartUs;
    }
    std::vector<int> selected(topK);
    auto readAndValidateSelected = [&](std::vector<int>* out, std::string* reason) -> ErrorCode {
        if (out == nullptr) {
            return INVALID_VALUE;
        }
        out->assign(static_cast<size_t>(topK), -1);
        cl_int readRet = queue.enqueueReadBuffer(openCLBuffer(mCacheBlendIndices.get()), CL_TRUE, 0,
                                                 static_cast<size_t>(topK) * sizeof(int), out->data());
        MNN_CHECK_CL_SUCCESS(readRet, "read pic_cacheblend_topk indices");
        if (readRet != CL_SUCCESS) {
            if (reason != nullptr) {
                *reason = std::string("read_failed ret=") + std::to_string(static_cast<int>(readRet));
            }
            return INVALID_VALUE;
        }
        std::vector<uint8_t> seen(static_cast<size_t>(picTokenCount), 0);
        for (int i = 0; i < topK; ++i) {
            const int index = (*out)[static_cast<size_t>(i)];
            if (index < 0 || index >= picTokenCount) {
                if (reason != nullptr) {
                    *reason = std::string("out_of_range at=") + std::to_string(i) +
                              " index=" + std::to_string(index);
                }
                return INVALID_VALUE;
            }
            if (seen[static_cast<size_t>(index)] != 0) {
                if (reason != nullptr) {
                    *reason = std::string("duplicate at=") + std::to_string(i) +
                              " index=" + std::to_string(index);
                }
                return INVALID_VALUE;
            }
            seen[static_cast<size_t>(index)] = 1;
        }
        return NO_ERROR;
    };
    opStartUs = profileDetail ? _nowUs() : 0;
    std::string invalidReason;
    err = readAndValidateSelected(&selected, &invalidReason);
    if (profileDetail) {
        readbackUs += _nowUs() - opStartUs;
    }
    if (_compareCacheBlendTopK() && topKDispatch.useStage) {
        const auto stagedErr = err;
        const auto stagedReason = invalidReason;
        const auto stagedSelected = selected;
        CacheBlendTopKDispatch legacyDispatch;
        std::vector<int> legacySelected;
        std::string legacyReason;
        auto legacyErr = runTopKFamily(legacyDispatch);
        if (legacyErr == NO_ERROR) {
            legacyErr = readAndValidateSelected(&legacySelected, &legacyReason);
        }
        auto summarize = [](const std::vector<int>& values) -> std::string {
            std::ostringstream os;
            os << "[";
            const int count = std::min<int>(static_cast<int>(values.size()), 16);
            for (int i = 0; i < count; ++i) {
                if (i > 0) {
                    os << ",";
                }
                os << values[static_cast<size_t>(i)];
            }
            if (static_cast<int>(values.size()) > count) {
                os << ",...";
            }
            os << "]";
            return os.str();
        };
        auto setEqual = [](std::vector<int> lhs, std::vector<int> rhs) -> bool {
            std::sort(lhs.begin(), lhs.end());
            std::sort(rhs.begin(), rhs.end());
            return lhs == rhs;
        };
        int firstMismatch = -1;
        const int compareCount = std::min<int>(static_cast<int>(stagedSelected.size()),
                                               static_cast<int>(legacySelected.size()));
        for (int i = 0; i < compareCount; ++i) {
            if (stagedSelected[static_cast<size_t>(i)] != legacySelected[static_cast<size_t>(i)]) {
                firstMismatch = i;
                break;
            }
        }
        if (firstMismatch < 0 && stagedSelected.size() != legacySelected.size()) {
            firstMismatch = compareCount;
        }
        const bool bothValid = stagedErr == NO_ERROR && legacyErr == NO_ERROR;
        const bool orderedEqual = bothValid && stagedSelected == legacySelected;
        const bool sameSet = bothValid && setEqual(stagedSelected, legacySelected);
        if (orderedEqual) {
            MNN_PRINT("OpenCLPagedAttention cacheblend top-k compare layer=%d pic_tokens=%d top_k=%d "
                      "staged=%s legacy=legacy ordered_equal=1 set_equal=1 sample=%s\n",
                      layerIndex, picTokenCount, topK, _cacheBlendTopKFamilyName(topKDispatch.family),
                      summarize(stagedSelected).c_str());
        } else {
            MNN_ERROR("OpenCLPagedAttention cacheblend top-k compare mismatch layer=%d pic_tokens=%d top_k=%d "
                      "staged=%s staged_err=%d staged_reason=%s legacy_err=%d legacy_reason=%s "
                      "ordered_equal=%d set_equal=%d first_mismatch=%d staged_sample=%s legacy_sample=%s\n",
                      layerIndex, picTokenCount, topK, _cacheBlendTopKFamilyName(topKDispatch.family),
                      static_cast<int>(stagedErr), stagedReason.c_str(), static_cast<int>(legacyErr),
                      legacyReason.c_str(), orderedEqual ? 1 : 0, sameSet ? 1 : 0, firstMismatch,
                      summarize(stagedSelected).c_str(), summarize(legacySelected).c_str());
        }
    }
    if (err != NO_ERROR && topKDispatch.useStage) {
        MNN_ERROR("OpenCLPagedAttention cacheblend staged top-k produced invalid selected rows, "
                  "retrying legacy path layer=%d pic_tokens=%d top_k=%d family=%s reason=%s\n",
                  layerIndex, picTokenCount, topK, _cacheBlendTopKFamilyName(topKDispatch.family),
                  invalidReason.c_str());
        CacheBlendTopKDispatch legacyDispatch;
        err = runTopKFamily(legacyDispatch);
        if (err != NO_ERROR) {
            return err;
        }
        topKDispatch = legacyDispatch;
        topKFamilySource = kTuneSelectionSourceDefault;
        opStartUs = profileDetail ? _nowUs() : 0;
        invalidReason.clear();
        err = readAndValidateSelected(&selected, &invalidReason);
        if (profileDetail) {
            readbackUs += _nowUs() - opStartUs;
        }
    }
    if (err != NO_ERROR) {
        MNN_ERROR("OpenCLPagedAttention cacheblend top-k selected rows invalid layer=%d pic_tokens=%d top_k=%d "
                  "path=%s reason=%s\n",
                  layerIndex, picTokenCount, topK, _cacheBlendTopKFamilyName(topKDispatch.family),
                  invalidReason.c_str());
        return err;
    }
    mMeta->setCacheBlendScoringResult(selected);
    if (profile) {
        queue.finish();
        const uint64_t totalUs = _nowUs() - startUs;
        MNN_PRINT("OpenCLPagedAttention profile op=cacheblend_score layer=%d pic_tokens=%d top_k=%d us=%llu "
                  "read_us=%llu image_prep_us=%llu score_kernel_us=%llu topk_us=%llu readback_us=%llu "
                  "detail=%d topk_path=%s image_segments=%d image_tokens=%d stage_candidates=%d "
                  "stage_block=%d stage_sort=%d\n",
                  layerIndex, picTokenCount, topK,
                  static_cast<unsigned long long>(totalUs),
                  static_cast<unsigned long long>(readUs),
                  static_cast<unsigned long long>(imagePrepUs),
                  static_cast<unsigned long long>(scoreKernelUs),
                  static_cast<unsigned long long>(topKUs),
                  static_cast<unsigned long long>(readbackUs),
                  profileDetail ? 1 : 0,
                  _cacheBlendTopKFamilyName(topKDispatch.family),
                  imageSegments,
                  static_cast<int>(imageTokens),
                  topKDispatch.useStage ? topKDispatch.stageCandidateCount : 0,
                  topKDispatch.useStage ? topKDispatch.blockSize : 0,
                  topKDispatch.useStage ? topKDispatch.stageSortSize : 0);
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
    if (mIsKVShared || mMeta == nullptr || !mMeta->sparse_query_active) {
        return false;
    }
    const bool scoreLayerFullQ = queryRowsAreFull && mPicAttentionMode == 1;
    if (!externalHydrated && !scoreLayerFullQ) {
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

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttention(const std::vector<Tensor*>& inputs,
                                                               const std::vector<Tensor*>& outputs, int kvLen,
                                                               int attnLen, int baseLogical, bool sparseQuery,
                                                               bool queryRowsAreFull) {
    if (attnLen <= 0 || kvLen <= 0 || mHeadDim != 64) {
        return INVALID_VALUE;
    }
    if (sparseQuery &&
        (mMeta == nullptr || static_cast<int>(mMeta->sparse_query_logical_indices.size()) < attnLen)) {
        return INVALID_VALUE;
    }
    auto err = ensureDecodeCausalKernel();
    if (err != NO_ERROR) {
        return err;
    }
    auto query = inputs[0];
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
    const uint32_t lanes = _sparseFlashLaneWidth(sparseQuery ? mMeta : nullptr, attnLen, mHeadDim);
    auto kernel = lanes == 32u ? mDecodeCausalKernel32 : mDecodeCausalKernel64;
    if (!kernel) {
        return INVALID_VALUE;
    }
    std::vector<uint32_t> gws = {
        lanes,
        static_cast<uint32_t>(attnLen),
        static_cast<uint32_t>(mNumHead * mBatch),
    };
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    ret |= kernel->get().setArg(idx++, gws[0]);
    ret |= kernel->get().setArg(idx++, gws[1]);
    ret |= kernel->get().setArg(idx++, gws[2]);
    ret |= kernel->get().setArg(idx++, openCLBuffer(query));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->sparseQuery.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= kernel->get().setArg(idx++, mScale);
    ret |= kernel->get().setArg(idx++, mBatch);
    ret |= kernel->get().setArg(idx++, mQuerySeqLen);
    ret |= kernel->get().setArg(idx++, attnLen);
    ret |= kernel->get().setArg(idx++, baseLogical);
    ret |= kernel->get().setArg(idx++, sparseQuery ? 1 : 0);
    ret |= kernel->get().setArg(idx++, queryRowsAreFull ? 1 : 0);
    ret |= kernel->get().setArg(idx++, kvLen);
    ret |= kernel->get().setArg(idx++, mCache->maxSlots);
    ret |= kernel->get().setArg(idx++, mNumHead);
    ret |= kernel->get().setArg(idx++, mKvNumHead);
    ret |= kernel->get().setArg(idx++, mHeadDim);
    MNN_CHECK_CL_SUCCESS(ret, lanes == 32u ? "setArg decode_causal_attention_row32"
                                           : "setArg decode_causal_attention_row64");
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const uint64_t denseWork = static_cast<uint64_t>(attnLen) * static_cast<uint64_t>(kvLen);
    uint64_t causalWork = 0;
    if (sparseQuery) {
        causalWork = _sparseLogicalWork(mMeta, attnLen, kvLen);
    } else {
        for (int i = 0; i < attnLen; ++i) {
            causalWork += static_cast<uint64_t>(std::max(0, std::min(kvLen, baseLogical + i + 1)));
        }
    }
    const int layerIndex = mLayerIndex >= 0 ? mLayerIndex : (mMeta != nullptr ? mMeta->layer_index : -1);
    const std::vector<uint32_t> lws = {lanes, 1u, 1u};
    const auto pmcMeta = _makeOpenCLPmcMeta("decode_causal_attention", "attention", layerIndex,
                                           attnLen, mQuerySeqLen, kvLen, baseLogical,
                                           static_cast<int>(lanes), 0, mNumHead, mKvNumHead, mHeadDim,
                                           gws, lws, denseWork, causalWork);
    _run3DKernelDefaultPmc(kernel, gws, lws, runtime, pmcMeta);
    if (profile) {
        runtime->commandQueue().finish();
        MNN_PRINT("OpenCLPagedAttention profile op=decode_causal_attention layer=%d "
                  "query=%d input_query=%d sparse=%d full_q=%d kv_len=%d lane=%u "
                  "dense_kv_work=%llu causal_kv_work=%llu us=%llu\n",
                  layerIndex, attnLen, mQuerySeqLen, sparseQuery ? 1 : 0, queryRowsAreFull ? 1 : 0,
                  kvLen, lanes,
                  static_cast<unsigned long long>(denseWork),
                  static_cast<unsigned long long>(causalWork),
                  static_cast<unsigned long long>(_nowUs() - startUs));
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128Identity(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int baseLogical, int layerIndex) {
    if (attnLen <= 0 || kvLen <= 0 || mHeadDim != 128 || mCache == nullptr ||
        !mCache->key || !mCache->value) {
        return INVALID_VALUE;
    }
    auto err = ensureDecodeCausalKernelHD128Identity();
    if (err != NO_ERROR) {
        return err;
    }
    auto query = inputs[0];
    auto output = outputs[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t attentionUs = 0;
    uint64_t rankUs = 0;
    uint64_t causalWork = 0;
    for (int i = 0; i < attnLen; ++i) {
        causalWork += static_cast<uint64_t>(std::max(0, std::min(kvLen, baseLogical + i + 1)));
    }
    const uint64_t causalWorkPerRow = attnLen > 0 ? causalWork / static_cast<uint64_t>(attnLen) : 0;
    const uint32_t lanes = _decodeHD128LaneWidth(causalWorkPerRow);
    auto kernel = lanes == 128u ? mDecodeCausalKernelHD128IdentityRow128 :
        (lanes == 64u ? mDecodeCausalKernelHD128IdentityRow64 : mDecodeCausalKernelHD128IdentityRow32);
    if (!kernel) {
        return INVALID_VALUE;
    }
    if (_decodeIdentityAttentionOnlyBenchEnabled() && _decodeQ1SplitProfileEnabled() && profile) {
        auto splitErr = runDecodeCausalAttentionHD128SplitProfile(inputs, outputs, kvLen, attnLen,
                                                                  baseLogical, layerIndex, false);
        if (splitErr == NO_ERROR) {
            return NO_ERROR;
        }
        if (_picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA decode identity split profile fallback layer=%d err=%d\n",
                      layerIndex, static_cast<int>(splitErr));
        }
    }
    if (mOpenCLBackend->isUseRecordQueue() && !profile &&
        (mMeta == nullptr || !mMeta->needsPicDecodeAttentionRankCapture(layerIndex))) {
        auto recordErr = runDecodeCausalAttentionHD128IdentityRecord(
            inputs, outputs, kvLen, attnLen, baseLogical, layerIndex, lanes, kernel);
        if (recordErr == NO_ERROR) {
            return NO_ERROR;
        }
    }
    std::vector<uint32_t> gws = {
        lanes,
        static_cast<uint32_t>(attnLen),
        static_cast<uint32_t>(mNumHead * mBatch),
    };
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    ret |= kernel->get().setArg(idx++, gws[0]);
    ret |= kernel->get().setArg(idx++, gws[1]);
    ret |= kernel->get().setArg(idx++, gws[2]);
    ret |= kernel->get().setArg(idx++, openCLBuffer(query));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= kernel->get().setArg(idx++, mScale);
    ret |= kernel->get().setArg(idx++, mBatch);
    ret |= kernel->get().setArg(idx++, mQuerySeqLen);
    ret |= kernel->get().setArg(idx++, attnLen);
    ret |= kernel->get().setArg(idx++, baseLogical);
    ret |= kernel->get().setArg(idx++, kvLen);
    ret |= kernel->get().setArg(idx++, mCache->maxSlots);
    ret |= kernel->get().setArg(idx++, mNumHead);
    ret |= kernel->get().setArg(idx++, mKvNumHead);
    ret |= kernel->get().setArg(idx++, mHeadDim);
    MNN_CHECK_CL_SUCCESS(ret, lanes == 128u ? "setArg decode_causal_attention_hd128_identity_row128" :
        (lanes == 64u ? "setArg decode_causal_attention_hd128_identity_row64"
                      : "setArg decode_causal_attention_hd128_identity_row32"));
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const uint64_t attentionStartUs = profileDetail ? _nowUs() : 0;
    const uint64_t denseWork = static_cast<uint64_t>(attnLen) * static_cast<uint64_t>(kvLen);
    const std::vector<uint32_t> lws = {lanes, 1u, 1u};
    const auto pmcMeta = _makeOpenCLPmcMeta("decode_causal_attention_hd128_identity", "attention", layerIndex,
                                           attnLen, mQuerySeqLen, kvLen, baseLogical,
                                           static_cast<int>(lanes), 0, mNumHead, mKvNumHead, mHeadDim,
                                           gws, lws, denseWork, causalWork);
    _run3DKernelDefaultPmc(kernel, gws, lws, runtime, pmcMeta);
    if (profileDetail) {
        runtime->commandQueue().finish();
        attentionUs = _nowUs() - attentionStartUs;
    }
    const uint64_t rankStartUs = profileDetail ? _nowUs() : 0;
    auto rankErr = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, attnLen, false);
    if (rankErr != NO_ERROR) {
        return rankErr;
    }
    if (profileDetail) {
        runtime->commandQueue().finish();
        rankUs = _nowUs() - rankStartUs;
    }
    if (profile) {
        runtime->commandQueue().finish();
        if (profileDetail) {
            MNN_PRINT("OpenCLPagedAttention profile op=decode_causal_attention_hd128_identity layer=%d "
                      "query=%d input_query=%d full_q=0 kv_len=%d lane=%u identity_slot=1 slot_identity=1 "
                      "dense_kv_work=%llu causal_kv_work=%llu us=%llu append_us=0 attention_us=%llu "
                      "rank_us=%llu attention_only=%d record_queue=0 "
                      "decode_key_ready_hit=0 decode_key_prepared=0 decode_prepare_inside_decode=0 "
                      "decode_prepare_us=0 prepare_us=0 decode_key_required=0 prefix_stable=1\n",
                      layerIndex, attnLen, mQuerySeqLen, kvLen, lanes,
                      static_cast<unsigned long long>(denseWork),
                      static_cast<unsigned long long>(causalWork),
                      static_cast<unsigned long long>(_nowUs() - startUs),
                      static_cast<unsigned long long>(attentionUs),
                      static_cast<unsigned long long>(rankUs),
                      _decodeIdentityAttentionOnlyBenchEnabled() ? 1 : 0);
        } else {
            MNN_PRINT("OpenCLPagedAttention profile op=decode_causal_attention_hd128_identity layer=%d "
                      "query=%d input_query=%d full_q=0 kv_len=%d lane=%u identity_slot=1 slot_identity=1 "
                      "dense_kv_work=%llu causal_kv_work=%llu us=%llu append_us=0 attention_us=%llu "
                      "rank_us=0 attention_only=%d record_queue=0 "
                      "decode_key_ready_hit=0 decode_key_prepared=0 decode_prepare_inside_decode=0 "
                      "decode_prepare_us=0 prepare_us=0 decode_key_required=0 prefix_stable=1\n",
                      layerIndex, attnLen, mQuerySeqLen, kvLen, lanes,
                      static_cast<unsigned long long>(denseWork),
                      static_cast<unsigned long long>(causalWork),
                      static_cast<unsigned long long>(_nowUs() - startUs),
                      static_cast<unsigned long long>(_nowUs() - startUs),
                      _decodeIdentityAttentionOnlyBenchEnabled() ? 1 : 0);
        }
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128IdentityFusedKV(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int baseLogical, int layerIndex) {
    if (attnLen <= 0 || kvLen <= 0 || mHeadDim != 128 || mCache == nullptr ||
        !mCache->key || !mCache->value || inputs.size() < 3) {
        return INVALID_VALUE;
    }
    auto err = ensureDecodeCausalKernelHD128Identity();
    if (err != NO_ERROR) {
        return err;
    }
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t attentionUs = 0;
    uint64_t rankUs = 0;
    uint64_t causalWork = 0;
    for (int i = 0; i < attnLen; ++i) {
        causalWork += static_cast<uint64_t>(std::max(0, std::min(kvLen, baseLogical + i + 1)));
    }
    const uint64_t causalWorkPerRow = attnLen > 0 ? causalWork / static_cast<uint64_t>(attnLen) : 0;
    const uint32_t lanes = _decodeHD128LaneWidth(causalWorkPerRow);
    auto kernel = lanes == 128u ? mDecodeCausalKernelHD128IdentityFusedKVRow128 :
        (lanes == 64u ? mDecodeCausalKernelHD128IdentityFusedKVRow64 : mDecodeCausalKernelHD128IdentityFusedKVRow32);
    if (!kernel) {
        return INVALID_VALUE;
    }
    std::vector<uint32_t> gws = {
        lanes,
        static_cast<uint32_t>(attnLen),
        static_cast<uint32_t>(mNumHead * mBatch),
    };
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    ret |= kernel->get().setArg(idx++, gws[0]);
    ret |= kernel->get().setArg(idx++, gws[1]);
    ret |= kernel->get().setArg(idx++, gws[2]);
    ret |= kernel->get().setArg(idx++, openCLBuffer(query));
    ret |= kernel->get().setArg(idx++, openCLBuffer(key));
    ret |= kernel->get().setArg(idx++, openCLBuffer(value));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= kernel->get().setArg(idx++, mScale);
    ret |= kernel->get().setArg(idx++, mBatch);
    ret |= kernel->get().setArg(idx++, mQuerySeqLen);
    ret |= kernel->get().setArg(idx++, attnLen);
    ret |= kernel->get().setArg(idx++, baseLogical);
    ret |= kernel->get().setArg(idx++, kvLen);
    ret |= kernel->get().setArg(idx++, mCache->maxSlots);
    ret |= kernel->get().setArg(idx++, mNumHead);
    ret |= kernel->get().setArg(idx++, mKvNumHead);
    ret |= kernel->get().setArg(idx++, mHeadDim);
    MNN_CHECK_CL_SUCCESS(ret, lanes == 128u ? "setArg decode_causal_attention_hd128_identity_fused_kv_row128" :
        (lanes == 64u ? "setArg decode_causal_attention_hd128_identity_fused_kv_row64"
                      : "setArg decode_causal_attention_hd128_identity_fused_kv_row32"));
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const uint64_t attentionStartUs = profileDetail ? _nowUs() : 0;
    const uint64_t denseWork = static_cast<uint64_t>(attnLen) * static_cast<uint64_t>(kvLen);
    const std::vector<uint32_t> lws = {lanes, 1u, 1u};
    const auto pmcMeta = _makeOpenCLPmcMeta("decode_causal_attention_hd128_identity_fused_kv", "attention",
                                           layerIndex, attnLen, mQuerySeqLen, kvLen, baseLogical,
                                           static_cast<int>(lanes), 0, mNumHead, mKvNumHead, mHeadDim,
                                           gws, lws, denseWork, causalWork);
    _run3DKernelDefaultPmc(kernel, gws, lws, runtime, pmcMeta);
    if (profileDetail) {
        runtime->commandQueue().finish();
        attentionUs = _nowUs() - attentionStartUs;
    }
    const uint64_t rankStartUs = profileDetail ? _nowUs() : 0;
    auto rankErr = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, attnLen, false);
    if (rankErr != NO_ERROR) {
        return rankErr;
    }
    if (profileDetail) {
        runtime->commandQueue().finish();
        rankUs = _nowUs() - rankStartUs;
    }
    if (profile) {
        runtime->commandQueue().finish();
        const uint64_t totalUs = _nowUs() - startUs;
        MNN_PRINT("OpenCLPagedAttention profile op=decode_causal_attention_hd128_identity_fused_kv layer=%d "
                  "query=%d input_query=%d full_q=0 kv_len=%d lane=%u identity_slot=1 slot_identity=1 "
                  "dense_kv_work=%llu causal_kv_work=%llu us=%llu append_us=0 attention_us=%llu "
                  "rank_us=%llu attention_only=0 record_queue=0 "
                  "decode_key_ready_hit=0 decode_key_prepared=0 decode_prepare_inside_decode=0 "
                  "decode_prepare_us=0 prepare_us=0 decode_key_required=0 prefix_stable=1\n",
                  layerIndex, attnLen, mQuerySeqLen, kvLen, lanes,
                  static_cast<unsigned long long>(denseWork),
                  static_cast<unsigned long long>(causalWork),
                  static_cast<unsigned long long>(totalUs),
                  static_cast<unsigned long long>(profileDetail ? attentionUs : totalUs),
                  static_cast<unsigned long long>(profileDetail ? rankUs : 0));
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128IdentityFusedKVGQA(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int baseLogical, int layerIndex) {
    if (attnLen != 1 || kvLen <= 0 || mHeadDim != 128 || mCache == nullptr ||
        !mCache->key || !mCache->value || inputs.size() < 3 || mKvNumHead <= 0 ||
        mNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    const int groupSize = mNumHead / mKvNumHead;
    // Keep this first grouped decode path on the small GQA groups used by the target models.
    if (groupSize <= 1 || groupSize > 8) {
        return INVALID_VALUE;
    }
    auto err = ensureDecodeCausalKernelHD128Identity();
    if (err != NO_ERROR) {
        return err;
    }
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t attentionUs = 0;
    uint64_t rankUs = 0;
    uint64_t causalWork = 0;
    for (int i = 0; i < attnLen; ++i) {
        causalWork += static_cast<uint64_t>(std::max(0, std::min(kvLen, baseLogical + i + 1)));
    }
    const uint64_t causalWorkPerRow = attnLen > 0 ? causalWork / static_cast<uint64_t>(attnLen) : 0;
    const uint32_t lanes = _decodeHD128LaneWidth(causalWorkPerRow);
    auto kernel = lanes == 128u ? mDecodeCausalKernelHD128IdentityFusedKVGQARow128 :
        (lanes == 64u ? mDecodeCausalKernelHD128IdentityFusedKVGQARow64
                      : mDecodeCausalKernelHD128IdentityFusedKVGQARow32);
    if (!kernel) {
        return INVALID_VALUE;
    }
    std::vector<uint32_t> gws = {
        lanes,
        static_cast<uint32_t>(attnLen),
        static_cast<uint32_t>(mKvNumHead * mBatch),
    };
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    ret |= kernel->get().setArg(idx++, gws[0]);
    ret |= kernel->get().setArg(idx++, gws[1]);
    ret |= kernel->get().setArg(idx++, gws[2]);
    ret |= kernel->get().setArg(idx++, openCLBuffer(query));
    ret |= kernel->get().setArg(idx++, openCLBuffer(key));
    ret |= kernel->get().setArg(idx++, openCLBuffer(value));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= kernel->get().setArg(idx++, mScale);
    ret |= kernel->get().setArg(idx++, mBatch);
    ret |= kernel->get().setArg(idx++, mQuerySeqLen);
    ret |= kernel->get().setArg(idx++, attnLen);
    ret |= kernel->get().setArg(idx++, baseLogical);
    ret |= kernel->get().setArg(idx++, kvLen);
    ret |= kernel->get().setArg(idx++, mCache->maxSlots);
    ret |= kernel->get().setArg(idx++, mNumHead);
    ret |= kernel->get().setArg(idx++, mKvNumHead);
    ret |= kernel->get().setArg(idx++, mHeadDim);
    MNN_CHECK_CL_SUCCESS(ret, lanes == 128u ? "setArg decode_causal_attention_hd128_identity_fused_kv_gqa_row128" :
        (lanes == 64u ? "setArg decode_causal_attention_hd128_identity_fused_kv_gqa_row64"
                      : "setArg decode_causal_attention_hd128_identity_fused_kv_gqa_row32"));
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const uint64_t attentionStartUs = profileDetail ? _nowUs() : 0;
    const uint64_t denseWork = static_cast<uint64_t>(attnLen) * static_cast<uint64_t>(kvLen);
    const std::vector<uint32_t> lws = {lanes, 1u, 1u};
    const auto pmcMeta = _makeOpenCLPmcMeta("decode_causal_attention_hd128_identity_fused_kv_gqa", "attention",
                                           layerIndex, attnLen, mQuerySeqLen, kvLen, baseLogical,
                                           static_cast<int>(lanes), 0, mNumHead, mKvNumHead, mHeadDim,
                                           gws, lws, denseWork, causalWork);
    _run3DKernelDefaultPmc(kernel, gws, lws, runtime, pmcMeta);
    if (profileDetail) {
        runtime->commandQueue().finish();
        attentionUs = _nowUs() - attentionStartUs;
    }
    const uint64_t rankStartUs = profileDetail ? _nowUs() : 0;
    auto rankErr = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, attnLen, false);
    if (rankErr != NO_ERROR) {
        return rankErr;
    }
    if (profileDetail) {
        runtime->commandQueue().finish();
        rankUs = _nowUs() - rankStartUs;
    }
    if (profile) {
        runtime->commandQueue().finish();
        const uint64_t totalUs = _nowUs() - startUs;
        MNN_PRINT("OpenCLPagedAttention profile op=decode_causal_attention_hd128_identity_fused_kv_gqa layer=%d "
                  "query=%d input_query=%d full_q=0 kv_len=%d lane=%u identity_slot=1 slot_identity=1 "
                  "group_size=%d kv_heads=%d dense_kv_work=%llu causal_kv_work=%llu us=%llu "
                  "append_us=0 attention_us=%llu rank_us=%llu attention_only=0 record_queue=0 "
                  "decode_key_ready_hit=0 decode_key_prepared=0 decode_prepare_inside_decode=0 "
                  "decode_prepare_us=0 prepare_us=0 decode_key_required=0 prefix_stable=1\n",
                  layerIndex, attnLen, mQuerySeqLen, kvLen, lanes, groupSize, mKvNumHead,
                  static_cast<unsigned long long>(denseWork),
                  static_cast<unsigned long long>(causalWork),
                  static_cast<unsigned long long>(totalUs),
                  static_cast<unsigned long long>(profileDetail ? attentionUs : totalUs),
                  static_cast<unsigned long long>(profileDetail ? rankUs : 0));
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128TransposedKFusedKVGQA(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int baseLogical, int layerIndex) {
    if (attnLen != 1 || kvLen <= 0 || baseLogical < 0 || mHeadDim != 128 || mCache == nullptr ||
        !mCache->key || !mCache->decodeKey || !mCache->value ||
        inputs.size() < 3 || mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    const int groupSize = mNumHead / mKvNumHead;
    if (groupSize <= 1 || groupSize > 8) {
        return INVALID_VALUE;
    }
    auto err = ensureDecodeTransposedKKernel();
    if (err != NO_ERROR) {
        return err;
    }
    err = ensureDecodeKeyReady(baseLogical, true);
    if (err != NO_ERROR) {
        return err;
    }

    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t attentionUs = 0;
    uint64_t rankUs = 0;
    uint64_t causalWork = 0;
    for (int i = 0; i < attnLen; ++i) {
        causalWork += static_cast<uint64_t>(std::max(0, std::min(kvLen, baseLogical + i + 1)));
    }
    const uint64_t causalWorkPerRow = attnLen > 0 ? causalWork / static_cast<uint64_t>(attnLen) : 0;
    const uint32_t lanes = _decodeHD128LaneWidth(causalWorkPerRow);
    auto kernel = lanes == 128u ? mDecodeCausalKernelHD128TransposedKFusedKVGQARow128 :
        (lanes == 64u ? mDecodeCausalKernelHD128TransposedKFusedKVGQARow64
                      : mDecodeCausalKernelHD128TransposedKFusedKVGQARow32);
    if (!kernel) {
        return INVALID_VALUE;
    }
    std::vector<uint32_t> gws = {
        lanes,
        static_cast<uint32_t>(attnLen),
        static_cast<uint32_t>(mKvNumHead * mBatch),
    };
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    ret |= kernel->get().setArg(idx++, gws[0]);
    ret |= kernel->get().setArg(idx++, gws[1]);
    ret |= kernel->get().setArg(idx++, gws[2]);
    ret |= kernel->get().setArg(idx++, openCLBuffer(query));
    ret |= kernel->get().setArg(idx++, openCLBuffer(key));
    ret |= kernel->get().setArg(idx++, openCLBuffer(value));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= kernel->get().setArg(idx++, mScale);
    ret |= kernel->get().setArg(idx++, mBatch);
    ret |= kernel->get().setArg(idx++, mQuerySeqLen);
    ret |= kernel->get().setArg(idx++, attnLen);
    ret |= kernel->get().setArg(idx++, baseLogical);
    ret |= kernel->get().setArg(idx++, kvLen);
    ret |= kernel->get().setArg(idx++, mCache->maxSlots);
    ret |= kernel->get().setArg(idx++, mNumHead);
    ret |= kernel->get().setArg(idx++, mKvNumHead);
    ret |= kernel->get().setArg(idx++, mHeadDim);
    MNN_CHECK_CL_SUCCESS(ret, lanes == 128u ?
        "setArg decode_causal_attention_hd128_transposed_k_fused_kv_gqa_row128" :
        (lanes == 64u ? "setArg decode_causal_attention_hd128_transposed_k_fused_kv_gqa_row64"
                      : "setArg decode_causal_attention_hd128_transposed_k_fused_kv_gqa_row32"));
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const uint64_t attentionStartUs = profileDetail ? _nowUs() : 0;
    const uint64_t denseWork = static_cast<uint64_t>(attnLen) * static_cast<uint64_t>(kvLen);
    const std::vector<uint32_t> lws = {lanes, 1u, 1u};
    const auto pmcMeta = _makeOpenCLPmcMeta("decode_causal_attention_hd128_transposed_k_fused_kv_gqa",
                                           "attention", layerIndex, attnLen, mQuerySeqLen, kvLen,
                                           baseLogical, static_cast<int>(lanes), 0, mNumHead,
                                           mKvNumHead, mHeadDim, gws, lws, denseWork, causalWork);
    _run3DKernelDefaultPmc(kernel, gws, lws, runtime, pmcMeta);
    if (profileDetail) {
        runtime->commandQueue().finish();
        attentionUs = _nowUs() - attentionStartUs;
    }
    mCache->decodeKeyReadyLength = std::max(mCache->decodeKeyReadyLength, baseLogical + 1);
    const uint64_t rankStartUs = profileDetail ? _nowUs() : 0;
    auto rankErr = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, attnLen, false);
    if (rankErr != NO_ERROR) {
        return rankErr;
    }
    if (profileDetail) {
        runtime->commandQueue().finish();
        rankUs = _nowUs() - rankStartUs;
    }
    if (profile) {
        runtime->commandQueue().finish();
        const uint64_t totalUs = _nowUs() - startUs;
        MNN_PRINT("OpenCLPagedAttention profile op=decode_causal_attention_hd128_transposed_k_fused_kv_gqa layer=%d "
                  "query=%d input_query=%d kv_len=%d lane=%u group_size=%d kv_heads=%d "
                  "dense_kv_work=%llu causal_kv_work=%llu us=%llu append_us=0 attention_us=%llu "
                  "rank_us=%llu decode_key_ready_hit=%d decode_key_prepared=%d "
                  "decode_prepare_inside_decode=%d decode_prepare_us=%llu prepare_us=%llu "
                  "decode_key_required=%d slot_identity=%d prefix_stable=%d record_queue=0\n",
                  layerIndex, attnLen, mQuerySeqLen, kvLen, lanes, groupSize, mKvNumHead,
                  static_cast<unsigned long long>(denseWork),
                  static_cast<unsigned long long>(causalWork),
                  static_cast<unsigned long long>(totalUs),
                  static_cast<unsigned long long>(profileDetail ? attentionUs : totalUs),
                  static_cast<unsigned long long>(profileDetail ? rankUs : 0),
                  mLastDecodeKeyReadyHit ? 1 : 0,
                  mLastDecodeKeyPrepared ? 1 : 0,
                  mLastDecodeKeyPrepareInsideDecode ? 1 : 0,
                  static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                  static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                  mLastDecodeKeyRequiredLen,
                  mLastDecodeKeySlotIdentity ? 1 : 0,
                  mLastDecodeKeyPrefixStable ? 1 : 0);
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128TransposedKFusedKV(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int baseLogical, int layerIndex) {
    if (attnLen != 1 || kvLen <= 0 || baseLogical < 0 || mHeadDim != 128 || mCache == nullptr ||
        !mCache->key || !mCache->decodeKey || !mCache->value || inputs.size() < 3 ||
        mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    auto err = ensureDecodeTransposedKKernel();
    if (err != NO_ERROR) {
        return err;
    }
    err = ensureDecodeKeyReady(baseLogical, true);
    if (err != NO_ERROR) {
        return err;
    }

    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const bool profile = _profilePagedAttention();
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t causalWork = 0;
    for (int i = 0; i < attnLen; ++i) {
        causalWork += static_cast<uint64_t>(std::max(0, std::min(kvLen, baseLogical + i + 1)));
    }
    const uint64_t causalWorkPerRow = attnLen > 0 ? causalWork / static_cast<uint64_t>(attnLen) : 0;
    const uint32_t lanes = _decodeHD128LaneWidth(causalWorkPerRow);
    auto kernel = lanes == 128u ? mDecodeCausalKernelHD128TransposedKFusedKVRow128 :
        (lanes == 64u ? mDecodeCausalKernelHD128TransposedKFusedKVRow64
                      : mDecodeCausalKernelHD128TransposedKFusedKVRow32);
    if (!kernel) {
        return INVALID_VALUE;
    }
    if (mOpenCLBackend->isUseRecordQueue() && !profile &&
        (mMeta == nullptr || !mMeta->needsPicDecodeAttentionRankCapture(layerIndex))) {
        auto recordErr = runDecodeCausalAttentionHD128TransposedKFusedKVRecord(
            inputs, outputs, kvLen, attnLen, baseLogical, layerIndex, lanes, kernel);
        if (recordErr == NO_ERROR) {
            return NO_ERROR;
        }
    }
    std::vector<uint32_t> gws = {
        lanes,
        static_cast<uint32_t>(attnLen),
        static_cast<uint32_t>(mNumHead * mBatch),
    };
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    ret |= kernel->get().setArg(idx++, gws[0]);
    ret |= kernel->get().setArg(idx++, gws[1]);
    ret |= kernel->get().setArg(idx++, gws[2]);
    ret |= kernel->get().setArg(idx++, openCLBuffer(query));
    ret |= kernel->get().setArg(idx++, openCLBuffer(key));
    ret |= kernel->get().setArg(idx++, openCLBuffer(value));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= kernel->get().setArg(idx++, mScale);
    ret |= kernel->get().setArg(idx++, mBatch);
    ret |= kernel->get().setArg(idx++, mQuerySeqLen);
    ret |= kernel->get().setArg(idx++, attnLen);
    ret |= kernel->get().setArg(idx++, baseLogical);
    ret |= kernel->get().setArg(idx++, kvLen);
    ret |= kernel->get().setArg(idx++, mCache->maxSlots);
    ret |= kernel->get().setArg(idx++, mNumHead);
    ret |= kernel->get().setArg(idx++, mKvNumHead);
    ret |= kernel->get().setArg(idx++, mHeadDim);
    MNN_CHECK_CL_SUCCESS(ret, lanes == 128u ?
        "setArg decode_causal_attention_hd128_transposed_k_fused_kv_row128" :
        (lanes == 64u ? "setArg decode_causal_attention_hd128_transposed_k_fused_kv_row64"
                      : "setArg decode_causal_attention_hd128_transposed_k_fused_kv_row32"));
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    uint64_t attentionUs = 0;
    uint64_t rankUs = 0;
    const uint64_t attentionStartUs = profileDetail ? _nowUs() : 0;
    const uint64_t denseWork = static_cast<uint64_t>(attnLen) * static_cast<uint64_t>(kvLen);
    const std::vector<uint32_t> lws = {lanes, 1u, 1u};
    const auto pmcMeta = _makeOpenCLPmcMeta("decode_causal_attention_hd128_transposed_k_fused_kv",
                                           "attention", layerIndex, attnLen, mQuerySeqLen, kvLen,
                                           baseLogical, static_cast<int>(lanes), 0, mNumHead,
                                           mKvNumHead, mHeadDim, gws, lws, denseWork, causalWork);
    _run3DKernelDefaultPmc(kernel, gws, lws, runtime, pmcMeta);
    if (profileDetail) {
        runtime->commandQueue().finish();
        attentionUs = _nowUs() - attentionStartUs;
    }
    mCache->decodeKeyReadyLength = std::max(mCache->decodeKeyReadyLength, baseLogical + 1);
    const uint64_t rankStartUs = profileDetail ? _nowUs() : 0;
    auto rankErr = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, attnLen, false);
    if (rankErr != NO_ERROR) {
        return rankErr;
    }
    if (profileDetail) {
        runtime->commandQueue().finish();
        rankUs = _nowUs() - rankStartUs;
    }

    if (profile) {
        runtime->commandQueue().finish();
        const uint64_t totalUs = _nowUs() - startUs;
        if (!profileDetail) {
            attentionUs = totalUs;
        }
        MNN_PRINT("OpenCLPagedAttention profile op=decode_causal_attention_hd128_transposed_k_fused_kv layer=%d "
                  "query=%d input_query=%d kv_len=%d lane=%u identity_slot=1 "
                  "dense_kv_work=%llu causal_kv_work=%llu us=%llu append_us=0 attention_us=%llu rank_us=%llu "
                  "decode_key_ready_hit=%d decode_key_prepared=%d "
                  "decode_prepare_inside_decode=%d decode_prepare_us=%llu prepare_us=%llu "
                  "decode_key_required=%d slot_identity=%d prefix_stable=%d record_queue=0 append_fused=1\n",
                  layerIndex, attnLen, mQuerySeqLen, kvLen, lanes,
                  static_cast<unsigned long long>(denseWork),
                  static_cast<unsigned long long>(causalWork),
                  static_cast<unsigned long long>(totalUs),
                  static_cast<unsigned long long>(attentionUs),
                  static_cast<unsigned long long>(rankUs),
                  mLastDecodeKeyReadyHit ? 1 : 0,
                  mLastDecodeKeyPrepared ? 1 : 0,
                  mLastDecodeKeyPrepareInsideDecode ? 1 : 0,
                  static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                  static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                  mLastDecodeKeyRequiredLen,
                  mLastDecodeKeySlotIdentity ? 1 : 0,
                  mLastDecodeKeyPrefixStable ? 1 : 0);
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128TransposedKFusedKVRecord(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int baseLogical, int layerIndex, uint32_t lanes, std::shared_ptr<KernelWrap> kernel) {
    if (!mOpenCLBackend->isUseRecordQueue() || _profilePagedAttention() || !kernel ||
        attnLen != 1 || kvLen <= 0 || mCache == nullptr || !mCache->key || !mCache->decodeKey ||
        !mCache->value || inputs.size() < 3 || outputs.empty() ||
        mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0 ||
        (mMeta != nullptr && mMeta->needsPicDecodeAttentionRankCapture(layerIndex))) {
        return INVALID_VALUE;
    }
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    const uint32_t heads = static_cast<uint32_t>(mNumHead * mBatch);
    const int groupSize = mNumHead / mKvNumHead;
    const bool needRecord = !mDecodeTransposedFusedRecordValid ||
        mDecodeTransposedFusedRecordLanes != lanes ||
        mDecodeTransposedFusedRecordAttnLen != attnLen ||
        mDecodeTransposedFusedRecordHeads != heads ||
        mDecodeTransposedFusedRecordGroupSize != groupSize;

    auto updateRecordArgs = [&]() {
        auto& args = mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args;
        if (args.size() < 12) {
            return false;
        }
        mDecodeTransposedFusedRecordQuerySeqLen = mQuerySeqLen;
        mDecodeTransposedFusedRecordBaseLogical = baseLogical;
        mDecodeTransposedFusedRecordKvLen = kvLen;
        mDecodeTransposedFusedRecordMaxSlots = mCache->maxSlots;
        mDecodeTransposedFusedRecordScale = mScale;
        args[0].arg_value = &openCLBuffer(query)();
        args[1].arg_value = &openCLBuffer(key)();
        args[2].arg_value = &openCLBuffer(value)();
        args[3].arg_value = &openCLBuffer(mCache->key.get())();
        args[4].arg_value = &openCLBuffer(mCache->value.get())();
        args[5].arg_value = &openCLBuffer(mCache->decodeKey.get())();
        args[6].arg_value = &openCLBuffer(output)();
        args[7].arg_value = &mDecodeTransposedFusedRecordScale;
        args[8].arg_value = &mDecodeTransposedFusedRecordQuerySeqLen;
        args[9].arg_value = &mDecodeTransposedFusedRecordBaseLogical;
        args[10].arg_value = &mDecodeTransposedFusedRecordKvLen;
        args[11].arg_value = &mDecodeTransposedFusedRecordMaxSlots;
        return true;
    };

    if (needRecord) {
        mDecodeTransposedFusedRecordGws0 = lanes;
        mDecodeTransposedFusedRecordGws1 = static_cast<uint32_t>(attnLen);
        mDecodeTransposedFusedRecordGws2 = heads;
        mDecodeTransposedFusedRecordQuerySeqLen = mQuerySeqLen;
        mDecodeTransposedFusedRecordBaseLogical = baseLogical;
        mDecodeTransposedFusedRecordKvLen = kvLen;
        mDecodeTransposedFusedRecordMaxSlots = mCache->maxSlots;
        mDecodeTransposedFusedRecordScale = mScale;

        std::vector<uint32_t> gws = {mDecodeTransposedFusedRecordGws0,
                                     mDecodeTransposedFusedRecordGws1,
                                     mDecodeTransposedFusedRecordGws2};
        std::vector<uint32_t> lws = {lanes, 1u, 1u};
        cl_int ret = CL_SUCCESS;
        uint32_t idx = 0;
        ret |= kernel->get().setArg(idx++, gws[0]);
        ret |= kernel->get().setArg(idx++, gws[1]);
        ret |= kernel->get().setArg(idx++, gws[2]);
        ret |= kernel->get().setArg(idx++, openCLBuffer(query));
        ret |= kernel->get().setArg(idx++, openCLBuffer(key));
        ret |= kernel->get().setArg(idx++, openCLBuffer(value));
        ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
        ret |= kernel->get().setArg(idx++, openCLBuffer(output));
        ret |= kernel->get().setArg(idx++, mDecodeTransposedFusedRecordScale);
        ret |= kernel->get().setArg(idx++, mBatch);
        ret |= kernel->get().setArg(idx++, mDecodeTransposedFusedRecordQuerySeqLen);
        ret |= kernel->get().setArg(idx++, attnLen);
        ret |= kernel->get().setArg(idx++, mDecodeTransposedFusedRecordBaseLogical);
        ret |= kernel->get().setArg(idx++, mDecodeTransposedFusedRecordKvLen);
        ret |= kernel->get().setArg(idx++, mDecodeTransposedFusedRecordMaxSlots);
        ret |= kernel->get().setArg(idx++, mNumHead);
        ret |= kernel->get().setArg(idx++, mKvNumHead);
        ret |= kernel->get().setArg(idx++, mHeadDim);
        MNN_CHECK_CL_SUCCESS(ret, lanes == 128u ?
            "record setArg decode_causal_attention_hd128_transposed_k_fused_kv_row128" :
            (lanes == 64u ? "record setArg decode_causal_attention_hd128_transposed_k_fused_kv_row64"
                          : "record setArg decode_causal_attention_hd128_transposed_k_fused_kv_row32"));
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }

        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.clear();
        mDecodeTransposedFusedRecordUpdateInfo.update_global_size.clear();
        mDecodeTransposedFusedRecordUpdateInfo.update_local_size.clear();
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 3, sizeof(cl_mem), &openCLBuffer(query)()});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 4, sizeof(cl_mem), &openCLBuffer(key)()});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 5, sizeof(cl_mem), &openCLBuffer(value)()});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 6, sizeof(cl_mem), &openCLBuffer(mCache->key.get())()});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 7, sizeof(cl_mem), &openCLBuffer(mCache->value.get())()});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 8, sizeof(cl_mem), &openCLBuffer(mCache->decodeKey.get())()});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 9, sizeof(cl_mem), &openCLBuffer(output)()});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 10, sizeof(mDecodeTransposedFusedRecordScale), &mDecodeTransposedFusedRecordScale});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 12, sizeof(mDecodeTransposedFusedRecordQuerySeqLen),
             &mDecodeTransposedFusedRecordQuerySeqLen});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 14, sizeof(mDecodeTransposedFusedRecordBaseLogical),
             &mDecodeTransposedFusedRecordBaseLogical});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 15, sizeof(mDecodeTransposedFusedRecordKvLen), &mDecodeTransposedFusedRecordKvLen});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 16, sizeof(mDecodeTransposedFusedRecordMaxSlots), &mDecodeTransposedFusedRecordMaxSlots});
        mDecodeTransposedFusedRecordUpdateInfos.clear();
        mDecodeTransposedFusedRecordUpdateInfos.emplace_back(&mDecodeTransposedFusedRecordUpdateInfo);

        mOpenCLBackend->startRecord(mRecording);
        mOpenCLBackend->recordKernel3d(kernel, gws, lws, &mDecodeTransposedFusedRecordUpdateInfo);
        mOpenCLBackend->endRecord(mRecording);
        mDecodeTransposedFusedRecordValid = true;
        mDecodeTransposedFusedRecordLanes = lanes;
        mDecodeTransposedFusedRecordAttnLen = attnLen;
        mDecodeTransposedFusedRecordHeads = heads;
        mDecodeTransposedFusedRecordGroupSize = groupSize;
    }
    if (!updateRecordArgs()) {
        return INVALID_VALUE;
    }
    mOpenCLBackend->addRecord(mRecording, mDecodeTransposedFusedRecordUpdateInfos);
    mCache->decodeKeyReadyLength = std::max(mCache->decodeKeyReadyLength, baseLogical + 1);
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::appendDecodeKeyValueHD128(const std::vector<Tensor*>& inputs,
                                                                int baseLogical, bool profileDetail,
                                                                uint64_t* appendUs) {
    if (baseLogical < 0 || mHeadDim != 128 || mCache == nullptr || !mCache->key || !mCache->decodeKey ||
        !mCache->value || !mDecodeKeyAppendKernel || inputs.size() < 3 ||
        mKvNumHead <= 0) {
        return INVALID_VALUE;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (runtime == nullptr) {
        return INVALID_VALUE;
    }
    auto key = inputs[1];
    auto value = inputs[2];
    auto& queue = runtime->commandQueue();
    const uint64_t appendStartUs = profileDetail ? _nowUs() : 0;
    const uint32_t appendGws0 = 128u;
    const uint32_t appendGws1 = static_cast<uint32_t>(mBatch * mKvNumHead);
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, appendGws0);
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, appendGws1);
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, openCLBuffer(key));
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, openCLBuffer(value));
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, mBatch);
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, mKvNumHead);
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, mHeadDim);
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, 0);
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, baseLogical);
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, mCache->maxSlots);
    MNN_CHECK_CL_SUCCESS(ret, "setArg append_decode_key_value_hd128");
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const std::vector<uint32_t> appendGws = {appendGws0, appendGws1};
    const std::vector<uint32_t> appendLws = {32u, 1u};
    const int layerIndex = mLayerIndex >= 0 ? mLayerIndex : (mMeta != nullptr ? mMeta->layer_index : -1);
    const auto pmcMeta = _makeOpenCLPmcMeta("append_decode_key_value_hd128", "append", layerIndex,
                                           1, mQuerySeqLen, baseLogical + 1, baseLogical, 32, 0,
                                           mNumHead, mKvNumHead, mHeadDim, appendGws, appendLws,
                                           0, 0, 1, 0, false);
    _runOpenCLPmcScope(runtime, pmcMeta, [&](cl::Event* eventPtr) {
        ret = queue.enqueueNDRangeKernel(mDecodeKeyAppendKernel->get(), cl::NullRange,
                                         cl::NDRange(appendGws0, appendGws1), cl::NDRange(32, 1),
                                         nullptr, eventPtr);
    });
    MNN_CHECK_CL_SUCCESS(ret, "enqueue append_decode_key_value_hd128");
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    mCache->decodeKeyReadyLength = std::max(mCache->decodeKeyReadyLength, baseLogical + 1);
    if (profileDetail) {
        queue.finish();
        if (appendUs != nullptr) {
            *appendUs = _nowUs() - appendStartUs;
        }
    } else if (appendUs != nullptr) {
        *appendUs = 0;
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128SplitProfile(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int baseLogical, int layerIndex, bool transposedK) {
    if (!_profilePagedAttention() || !_decodeQ1SplitProfileEnabled() || attnLen != 1 || kvLen <= 0 ||
        baseLogical + 1 != kvLen || mHeadDim != 128 || mCache == nullptr || !mCache->key ||
        !mCache->value || inputs.empty() || outputs.empty() || mKvNumHead <= 0 || mNumHead <= 0 ||
        mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    auto err = ensureDecodeTransposedKKernel();
    if (err != NO_ERROR) {
        return err;
    }
    if (!transposedK) {
        err = ensureDecodeCausalKernelHD128Identity();
        if (err != NO_ERROR) {
            return err;
        }
    } else if (!mCache->decodeKey || mCache->decodeKeyReadyLength < kvLen) {
        return INVALID_VALUE;
    }
    err = ensureDecodeTransposedTemps(kvLen);
    if (err != NO_ERROR) {
        return err;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (runtime == nullptr) {
        return INVALID_VALUE;
    }
    auto query = inputs[0];
    auto output = outputs[0];
    auto qkKernel = transposedK ? mDecodeQKTransposedKernel : mDecodeQKIdentityKernel;
    if (!query || !output || !qkKernel || !mDecodeQKVTransposedKernel || !mDecodeSoftmaxKernel) {
        return INVALID_VALUE;
    }
    auto& qkBuffer = openCLDeferBuffer(mTempQK.get());
    auto& softmaxBuffer = openCLDeferBuffer(mTempSoftmax.get());

    auto& queue = runtime->commandQueue();
    const uint64_t startUs = _nowUs();
    uint64_t qkUs = 0;
    uint64_t softmaxUs = 0;
    uint64_t qkvUs = 0;
    uint64_t rankUs = 0;
    const uint32_t outside = static_cast<uint32_t>(mBatch * mNumHead);
    const uint64_t denseWork = static_cast<uint64_t>(attnLen) * static_cast<uint64_t>(kvLen);
    const uint64_t causalWork = static_cast<uint64_t>(kvLen);

    {
        std::vector<uint32_t> gws = {static_cast<uint32_t>((kvLen + 3) / 4), outside};
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= qkKernel->get().setArg(idx++, gws[0]);
        ret |= qkKernel->get().setArg(idx++, gws[1]);
        ret |= qkKernel->get().setArg(idx++, openCLBuffer(query));
        ret |= qkKernel->get().setArg(
            idx++, transposedK ? openCLBuffer(mCache->decodeKey.get()) : openCLBuffer(mCache->key.get()));
        ret |= qkKernel->get().setArg(idx++, qkBuffer);
        ret |= qkKernel->get().setArg(idx++, mScale);
        ret |= qkKernel->get().setArg(idx++, mBatch);
        ret |= qkKernel->get().setArg(idx++, mQuerySeqLen);
        ret |= qkKernel->get().setArg(idx++, kvLen);
        ret |= qkKernel->get().setArg(idx++, mCache->maxSlots);
        ret |= qkKernel->get().setArg(idx++, mNumHead);
        ret |= qkKernel->get().setArg(idx++, mKvNumHead);
        ret |= qkKernel->get().setArg(idx++, mHeadDim);
        MNN_CHECK_CL_SUCCESS(ret, transposedK ? "setArg matmul_qk_decode_transposed_hd128"
                                              : "setArg matmul_qk_decode_paged_identity_hd128");
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        const uint64_t opStartUs = _nowUs();
        ret = queue.enqueueNDRangeKernel(qkKernel->get(), cl::NullRange, cl::NDRange(gws[0], gws[1]),
                                         cl::NullRange);
        MNN_CHECK_CL_SUCCESS(ret, transposedK ? "enqueue matmul_qk_decode_transposed_hd128"
                                              : "enqueue matmul_qk_decode_paged_identity_hd128");
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        queue.finish();
        qkUs = _nowUs() - opStartUs;
    }

    {
        std::vector<uint32_t> gws = {64u, 1u, outside};
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mDecodeSoftmaxKernel->get().setArg(idx++, gws[0]);
        ret |= mDecodeSoftmaxKernel->get().setArg(idx++, gws[1]);
        ret |= mDecodeSoftmaxKernel->get().setArg(idx++, gws[2]);
        ret |= mDecodeSoftmaxKernel->get().setArg(idx++, qkBuffer);
        ret |= mDecodeSoftmaxKernel->get().setArg(idx++, softmaxBuffer);
        ret |= mDecodeSoftmaxKernel->get().setArg(idx++, 1);
        ret |= mDecodeSoftmaxKernel->get().setArg(idx++, static_cast<int>(outside));
        ret |= mDecodeSoftmaxKernel->get().setArg(idx++, kvLen);
        MNN_CHECK_CL_SUCCESS(ret, "setArg decode split softmax_in1_buf");
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        const uint64_t opStartUs = _nowUs();
        run3DKernelDefault(mDecodeSoftmaxKernel, gws, {64u, 1u, 1u}, runtime);
        queue.finish();
        softmaxUs = _nowUs() - opStartUs;
    }

    {
        std::vector<uint32_t> gws = {static_cast<uint32_t>((mHeadDim + 7) / 8), outside};
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, gws[0]);
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, gws[1]);
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, softmaxBuffer);
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, openCLBuffer(output));
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, kvLen);
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, mCache->maxSlots);
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, mBatch);
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, mNumHead);
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, mKvNumHead);
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, mHeadDim);
        MNN_CHECK_CL_SUCCESS(ret, "setArg matmul_qkv_decode_value_hd128_b8");
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        const uint64_t opStartUs = _nowUs();
        ret = queue.enqueueNDRangeKernel(mDecodeQKVTransposedKernel->get(), cl::NullRange,
                                         cl::NDRange(gws[0], gws[1]), cl::NullRange);
        MNN_CHECK_CL_SUCCESS(ret, "enqueue matmul_qkv_decode_value_hd128_b8");
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        queue.finish();
        qkvUs = _nowUs() - opStartUs;
    }

    const uint64_t rankStartUs = _nowUs();
    auto rankErr = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, attnLen, false);
    if (rankErr != NO_ERROR) {
        return rankErr;
    }
    queue.finish();
    rankUs = _nowUs() - rankStartUs;

    const uint64_t attentionUs = qkUs + softmaxUs + qkvUs;
    const uint64_t totalUs = _nowUs() - startUs;
    MNN_PRINT("OpenCLPagedAttention profile op=decode_causal_attention_hd128_%s_split layer=%d "
              "query=%d input_query=%d full_q=0 kv_len=%d lane=0 identity_slot=1 slot_identity=1 "
              "dense_kv_work=%llu causal_kv_work=%llu us=%llu append_us=0 attention_us=%llu "
              "qk_us=%llu softmax_us=%llu qkv_us=%llu qk_only_us=%llu qkv_only_us=%llu rank_us=%llu "
              "attention_only=1 record_queue=0 decode_key_ready_hit=%d decode_key_prepared=%d "
              "decode_prepare_inside_decode=%d decode_prepare_us=%llu prepare_us=%llu "
              "decode_key_required=%d prefix_stable=%d split_profile=1\n",
              transposedK ? "transposed_k" : "identity", layerIndex, attnLen, mQuerySeqLen, kvLen,
              static_cast<unsigned long long>(denseWork), static_cast<unsigned long long>(causalWork),
              static_cast<unsigned long long>(totalUs), static_cast<unsigned long long>(attentionUs),
              static_cast<unsigned long long>(qkUs), static_cast<unsigned long long>(softmaxUs),
              static_cast<unsigned long long>(qkvUs), static_cast<unsigned long long>(qkUs),
              static_cast<unsigned long long>(qkvUs), static_cast<unsigned long long>(rankUs),
              transposedK && mLastDecodeKeyReadyHit ? 1 : 0,
              transposedK && mLastDecodeKeyPrepared ? 1 : 0,
              transposedK && mLastDecodeKeyPrepareInsideDecode ? 1 : 0,
              static_cast<unsigned long long>(transposedK ? mLastDecodeKeyPrepareUs : 0),
              static_cast<unsigned long long>(transposedK ? mLastDecodeKeyPrepareUs : 0),
              transposedK ? mLastDecodeKeyRequiredLen : 0,
              (!transposedK || mLastDecodeKeyPrefixStable) ? 1 : 0);
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128TransposedKAppendReadonly(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int baseLogical, int layerIndex, bool appendCurrent) {
    if (attnLen != 1 || kvLen <= 0 || baseLogical < 0 || mHeadDim != 128 || mCache == nullptr ||
        !mCache->key || !mCache->decodeKey || !mCache->value || inputs.size() < 3 ||
        mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    auto err = ensureDecodeTransposedKKernel();
    if (err != NO_ERROR) {
        return err;
    }
    const int requiredDecodeKeyLen = appendCurrent ? baseLogical : kvLen;
    err = ensureDecodeKeyReady(requiredDecodeKeyLen, true);
    if (err != NO_ERROR) {
        return err;
    }

    auto query = inputs[0];
    auto output = outputs[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (runtime == nullptr || output == nullptr) {
        return INVALID_VALUE;
    }

    auto& queue = runtime->commandQueue();
    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t appendUs = 0;
    uint64_t attentionUs = 0;
    uint64_t rankUs = 0;

    uint64_t causalWork = 0;
    for (int i = 0; i < attnLen; ++i) {
        causalWork += static_cast<uint64_t>(std::max(0, std::min(kvLen, baseLogical + i + 1)));
    }
    const uint64_t causalWorkPerRow = attnLen > 0 ? causalWork / static_cast<uint64_t>(attnLen) : 0;
    const uint32_t lanes = _decodeHD128LaneWidth(causalWorkPerRow);
    auto kernel = lanes == 128u ? mDecodeCausalKernelHD128TransposedKReadonlyRow128 :
        (lanes == 64u ? mDecodeCausalKernelHD128TransposedKReadonlyRow64
                      : mDecodeCausalKernelHD128TransposedKReadonlyRow32);
    if (!kernel) {
        return INVALID_VALUE;
    }
    if (appendCurrent && mOpenCLBackend->isUseRecordQueue() && !profile &&
        (mMeta == nullptr || !mMeta->needsPicDecodeAttentionRankCapture(layerIndex))) {
        auto recordErr = runDecodeCausalAttentionHD128TransposedKAppendReadonlyRecord(
            inputs, outputs, kvLen, attnLen, baseLogical, layerIndex, lanes, kernel);
        if (recordErr == NO_ERROR) {
            return NO_ERROR;
        }
    }

    if (appendCurrent) {
        auto appendErr = appendDecodeKeyValueHD128(inputs, baseLogical, profileDetail, &appendUs);
        if (appendErr != NO_ERROR) {
            return appendErr;
        }
    }

    if (_decodeQ1SplitProfileEnabled() && profile) {
        auto splitErr = runDecodeCausalAttentionHD128SplitProfile(inputs, outputs, kvLen, attnLen,
                                                                  baseLogical, layerIndex, true);
        if (splitErr == NO_ERROR) {
            return NO_ERROR;
        }
        if (_picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA decode transposed_k split profile fallback layer=%d err=%d\n",
                      layerIndex, static_cast<int>(splitErr));
        }
    }

    const uint64_t attentionStartUs = profileDetail ? _nowUs() : 0;
    std::vector<uint32_t> gws = {
        lanes,
        static_cast<uint32_t>(attnLen),
        static_cast<uint32_t>(mNumHead * mBatch),
    };
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    ret |= kernel->get().setArg(idx++, gws[0]);
    ret |= kernel->get().setArg(idx++, gws[1]);
    ret |= kernel->get().setArg(idx++, gws[2]);
    ret |= kernel->get().setArg(idx++, openCLBuffer(query));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= kernel->get().setArg(idx++, mScale);
    ret |= kernel->get().setArg(idx++, mBatch);
    ret |= kernel->get().setArg(idx++, mQuerySeqLen);
    ret |= kernel->get().setArg(idx++, attnLen);
    ret |= kernel->get().setArg(idx++, baseLogical);
    ret |= kernel->get().setArg(idx++, kvLen);
    ret |= kernel->get().setArg(idx++, mCache->maxSlots);
    ret |= kernel->get().setArg(idx++, mNumHead);
    ret |= kernel->get().setArg(idx++, mKvNumHead);
    ret |= kernel->get().setArg(idx++, mHeadDim);
    MNN_CHECK_CL_SUCCESS(ret, lanes == 128u ?
        "setArg decode_causal_attention_hd128_transposed_k_readonly_row128" :
        (lanes == 64u ? "setArg decode_causal_attention_hd128_transposed_k_readonly_row64"
                      : "setArg decode_causal_attention_hd128_transposed_k_readonly_row32"));
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const uint64_t denseWork = static_cast<uint64_t>(attnLen) * static_cast<uint64_t>(kvLen);
    const std::vector<uint32_t> lws = {lanes, 1u, 1u};
    const auto pmcMeta = _makeOpenCLPmcMeta("decode_causal_attention_hd128_transposed_k_readonly",
                                           "attention", layerIndex, attnLen, mQuerySeqLen, kvLen,
                                           baseLogical, static_cast<int>(lanes), 0, mNumHead,
                                           mKvNumHead, mHeadDim, gws, lws, denseWork, causalWork);
    _run3DKernelDefaultPmc(kernel, gws, lws, runtime, pmcMeta);
    if (profileDetail) {
        queue.finish();
        attentionUs = _nowUs() - attentionStartUs;
    }

    const uint64_t rankStartUs = profileDetail ? _nowUs() : 0;
    auto rankErr = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, attnLen, false);
    if (rankErr != NO_ERROR) {
        return rankErr;
    }
    if (profileDetail) {
        queue.finish();
        rankUs = _nowUs() - rankStartUs;
    }

    if (profile) {
        queue.finish();
        const uint64_t totalUs = _nowUs() - startUs;
        MNN_PRINT("OpenCLPagedAttention profile op=decode_causal_attention_hd128_transposed_k_readonly layer=%d "
                  "query=%d input_query=%d kv_len=%d lane=%u append_count=%d prepare_len=%d identity_slot=1 "
                  "dense_kv_work=%llu causal_kv_work=%llu us=%llu append_us=%llu attention_us=%llu "
                  "rank_us=%llu attention_only=1 append_fused=0 skip_append=%d "
                  "decode_key_ready_hit=%d decode_key_prepared=%d "
                  "decode_prepare_inside_decode=%d decode_prepare_us=%llu prepare_us=%llu "
                  "decode_key_required=%d slot_identity=%d prefix_stable=%d record_queue=0\n",
                  layerIndex, attnLen, mQuerySeqLen, kvLen, lanes, appendCurrent ? 1 : 0,
                  requiredDecodeKeyLen,
                  static_cast<unsigned long long>(denseWork),
                  static_cast<unsigned long long>(causalWork),
                  static_cast<unsigned long long>(totalUs),
                  static_cast<unsigned long long>(appendUs),
                  static_cast<unsigned long long>(attentionUs),
                  static_cast<unsigned long long>(rankUs),
                  appendCurrent ? 0 : 1,
                  mLastDecodeKeyReadyHit ? 1 : 0,
                  mLastDecodeKeyPrepared ? 1 : 0,
                  mLastDecodeKeyPrepareInsideDecode ? 1 : 0,
                  static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                  static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                  mLastDecodeKeyRequiredLen,
                  mLastDecodeKeySlotIdentity ? 1 : 0,
                  mLastDecodeKeyPrefixStable ? 1 : 0);
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128TransposedKAppendReadonlyRecord(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int baseLogical, int layerIndex, uint32_t lanes, std::shared_ptr<KernelWrap> attentionKernel) {
    if (!mOpenCLBackend->isUseRecordQueue() || _profilePagedAttention() || !attentionKernel ||
        attnLen != 1 || kvLen <= 0 || mCache == nullptr || !mCache->key || !mCache->decodeKey ||
        !mCache->value || !mDecodeKeyAppendKernel || inputs.size() < 3 ||
        outputs.empty() || mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0 ||
        (mMeta != nullptr && mMeta->needsPicDecodeAttentionRankCapture(layerIndex))) {
        return INVALID_VALUE;
    }
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    const uint32_t appendGws0 = 128u;
    const uint32_t appendGws1 = static_cast<uint32_t>(mBatch * mKvNumHead);
    const uint32_t heads = static_cast<uint32_t>(mNumHead * mBatch);
    const int groupSize = mNumHead / mKvNumHead;
    const bool needRecord = !mDecodeTransposedAppendReadonlyRecordValid ||
        mDecodeTransposedAppendReadonlyRecordLanes != lanes ||
        mDecodeTransposedAppendReadonlyRecordAttnLen != attnLen ||
        mDecodeTransposedAppendReadonlyRecordHeads != heads ||
        mDecodeTransposedAppendReadonlyRecordGroupSize != groupSize ||
        mDecodeTransposedAppendReadonlyRecordAppendGws1 != appendGws1;

    auto updateRecordArgs = [&]() {
        auto& appendArgs = mDecodeTransposedAppendRecordUpdateInfo.update_kernel_args;
        auto& attentionArgs = mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args;
        if (appendArgs.size() < 7 || attentionArgs.size() < 9) {
            return false;
        }
        mDecodeTransposedAppendReadonlyRecordQuerySeqLen = mQuerySeqLen;
        mDecodeTransposedAppendReadonlyRecordBaseLogical = baseLogical;
        mDecodeTransposedAppendReadonlyRecordKvLen = kvLen;
        mDecodeTransposedAppendReadonlyRecordMaxSlots = mCache->maxSlots;
        mDecodeTransposedAppendReadonlyRecordScale = mScale;
        appendArgs[0].arg_value = &openCLBuffer(key)();
        appendArgs[1].arg_value = &openCLBuffer(value)();
        appendArgs[2].arg_value = &openCLBuffer(mCache->key.get())();
        appendArgs[3].arg_value = &openCLBuffer(mCache->value.get())();
        appendArgs[4].arg_value = &openCLBuffer(mCache->decodeKey.get())();
        appendArgs[5].arg_value = &mDecodeTransposedAppendReadonlyRecordBaseLogical;
        appendArgs[6].arg_value = &mDecodeTransposedAppendReadonlyRecordMaxSlots;
        attentionArgs[0].arg_value = &openCLBuffer(query)();
        attentionArgs[1].arg_value = &openCLBuffer(mCache->value.get())();
        attentionArgs[2].arg_value = &openCLBuffer(mCache->decodeKey.get())();
        attentionArgs[3].arg_value = &openCLBuffer(output)();
        attentionArgs[4].arg_value = &mDecodeTransposedAppendReadonlyRecordScale;
        attentionArgs[5].arg_value = &mDecodeTransposedAppendReadonlyRecordQuerySeqLen;
        attentionArgs[6].arg_value = &mDecodeTransposedAppendReadonlyRecordBaseLogical;
        attentionArgs[7].arg_value = &mDecodeTransposedAppendReadonlyRecordKvLen;
        attentionArgs[8].arg_value = &mDecodeTransposedAppendReadonlyRecordMaxSlots;
        return true;
    };

    if (needRecord) {
        mDecodeTransposedAppendReadonlyRecordAppendGws0 = appendGws0;
        mDecodeTransposedAppendReadonlyRecordAppendGws1 = appendGws1;
        mDecodeTransposedAppendReadonlyRecordGws0 = lanes;
        mDecodeTransposedAppendReadonlyRecordGws1 = static_cast<uint32_t>(attnLen);
        mDecodeTransposedAppendReadonlyRecordGws2 = heads;
        mDecodeTransposedAppendReadonlyRecordQuerySeqLen = mQuerySeqLen;
        mDecodeTransposedAppendReadonlyRecordBaseLogical = baseLogical;
        mDecodeTransposedAppendReadonlyRecordKvLen = kvLen;
        mDecodeTransposedAppendReadonlyRecordMaxSlots = mCache->maxSlots;
        mDecodeTransposedAppendReadonlyRecordScale = mScale;

        std::vector<uint32_t> appendGws = {mDecodeTransposedAppendReadonlyRecordAppendGws0,
                                           mDecodeTransposedAppendReadonlyRecordAppendGws1};
        std::vector<uint32_t> appendLws = {32u, 1u};
        std::vector<uint32_t> attentionGws = {mDecodeTransposedAppendReadonlyRecordGws0,
                                              mDecodeTransposedAppendReadonlyRecordGws1,
                                              mDecodeTransposedAppendReadonlyRecordGws2};
        std::vector<uint32_t> attentionLws = {lanes, 1u, 1u};
        cl_int ret = CL_SUCCESS;
        uint32_t idx = 0;
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, appendGws[0]);
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, appendGws[1]);
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, openCLBuffer(key));
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, openCLBuffer(value));
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, mBatch);
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, mKvNumHead);
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, mHeadDim);
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, 0);
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, mDecodeTransposedAppendReadonlyRecordBaseLogical);
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, mDecodeTransposedAppendReadonlyRecordMaxSlots);
        MNN_CHECK_CL_SUCCESS(ret, "record setArg append_decode_key_value_hd128");
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        idx = 0;
        ret = CL_SUCCESS;
        ret |= attentionKernel->get().setArg(idx++, attentionGws[0]);
        ret |= attentionKernel->get().setArg(idx++, attentionGws[1]);
        ret |= attentionKernel->get().setArg(idx++, attentionGws[2]);
        ret |= attentionKernel->get().setArg(idx++, openCLBuffer(query));
        ret |= attentionKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= attentionKernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
        ret |= attentionKernel->get().setArg(idx++, openCLBuffer(output));
        ret |= attentionKernel->get().setArg(idx++, mDecodeTransposedAppendReadonlyRecordScale);
        ret |= attentionKernel->get().setArg(idx++, mBatch);
        ret |= attentionKernel->get().setArg(idx++, mDecodeTransposedAppendReadonlyRecordQuerySeqLen);
        ret |= attentionKernel->get().setArg(idx++, attnLen);
        ret |= attentionKernel->get().setArg(idx++, mDecodeTransposedAppendReadonlyRecordBaseLogical);
        ret |= attentionKernel->get().setArg(idx++, mDecodeTransposedAppendReadonlyRecordKvLen);
        ret |= attentionKernel->get().setArg(idx++, mDecodeTransposedAppendReadonlyRecordMaxSlots);
        ret |= attentionKernel->get().setArg(idx++, mNumHead);
        ret |= attentionKernel->get().setArg(idx++, mKvNumHead);
        ret |= attentionKernel->get().setArg(idx++, mHeadDim);
        MNN_CHECK_CL_SUCCESS(ret, lanes == 128u ?
            "record setArg decode_causal_attention_hd128_transposed_k_readonly_row128" :
            (lanes == 64u ? "record setArg decode_causal_attention_hd128_transposed_k_readonly_row64"
                          : "record setArg decode_causal_attention_hd128_transposed_k_readonly_row32"));
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }

        mDecodeTransposedAppendRecordUpdateInfo.update_kernel_args.clear();
        mDecodeTransposedAppendRecordUpdateInfo.update_global_size.clear();
        mDecodeTransposedAppendRecordUpdateInfo.update_local_size.clear();
        mDecodeTransposedAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 2, sizeof(cl_mem), &openCLBuffer(key)()});
        mDecodeTransposedAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 3, sizeof(cl_mem), &openCLBuffer(value)()});
        mDecodeTransposedAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 4, sizeof(cl_mem), &openCLBuffer(mCache->key.get())()});
        mDecodeTransposedAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 5, sizeof(cl_mem), &openCLBuffer(mCache->value.get())()});
        mDecodeTransposedAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 6, sizeof(cl_mem), &openCLBuffer(mCache->decodeKey.get())()});
        mDecodeTransposedAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 11, sizeof(mDecodeTransposedAppendReadonlyRecordBaseLogical),
             &mDecodeTransposedAppendReadonlyRecordBaseLogical});
        mDecodeTransposedAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 12, sizeof(mDecodeTransposedAppendReadonlyRecordMaxSlots),
             &mDecodeTransposedAppendReadonlyRecordMaxSlots});

        mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args.clear();
        mDecodeTransposedReadonlyRecordUpdateInfo.update_global_size.clear();
        mDecodeTransposedReadonlyRecordUpdateInfo.update_local_size.clear();
        mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args.push_back(
            {1, 3, sizeof(cl_mem), &openCLBuffer(query)()});
        mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args.push_back(
            {1, 4, sizeof(cl_mem), &openCLBuffer(mCache->value.get())()});
        mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args.push_back(
            {1, 5, sizeof(cl_mem), &openCLBuffer(mCache->decodeKey.get())()});
        mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args.push_back(
            {1, 6, sizeof(cl_mem), &openCLBuffer(output)()});
        mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args.push_back(
            {1, 7, sizeof(mDecodeTransposedAppendReadonlyRecordScale),
             &mDecodeTransposedAppendReadonlyRecordScale});
        mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args.push_back(
            {1, 9, sizeof(mDecodeTransposedAppendReadonlyRecordQuerySeqLen),
             &mDecodeTransposedAppendReadonlyRecordQuerySeqLen});
        mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args.push_back(
            {1, 11, sizeof(mDecodeTransposedAppendReadonlyRecordBaseLogical),
             &mDecodeTransposedAppendReadonlyRecordBaseLogical});
        mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args.push_back(
            {1, 12, sizeof(mDecodeTransposedAppendReadonlyRecordKvLen),
             &mDecodeTransposedAppendReadonlyRecordKvLen});
        mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args.push_back(
            {1, 13, sizeof(mDecodeTransposedAppendReadonlyRecordMaxSlots),
             &mDecodeTransposedAppendReadonlyRecordMaxSlots});
        mDecodeTransposedAppendReadonlyRecordUpdateInfos.clear();
        mDecodeTransposedAppendReadonlyRecordUpdateInfos.emplace_back(&mDecodeTransposedAppendRecordUpdateInfo);
        mDecodeTransposedAppendReadonlyRecordUpdateInfos.emplace_back(&mDecodeTransposedReadonlyRecordUpdateInfo);

        mOpenCLBackend->startRecord(mRecording);
        mOpenCLBackend->recordKernel2d(mDecodeKeyAppendKernel, appendGws, appendLws,
                                       &mDecodeTransposedAppendRecordUpdateInfo);
        mOpenCLBackend->recordKernel3d(attentionKernel, attentionGws, attentionLws,
                                       &mDecodeTransposedReadonlyRecordUpdateInfo);
        mOpenCLBackend->endRecord(mRecording);
        mDecodeTransposedAppendReadonlyRecordValid = true;
        mDecodeTransposedAppendReadonlyRecordLanes = lanes;
        mDecodeTransposedAppendReadonlyRecordAttnLen = attnLen;
        mDecodeTransposedAppendReadonlyRecordHeads = heads;
        mDecodeTransposedAppendReadonlyRecordGroupSize = groupSize;
    }
    if (!updateRecordArgs()) {
        return INVALID_VALUE;
    }
    mOpenCLBackend->addRecord(mRecording, mDecodeTransposedAppendReadonlyRecordUpdateInfos);
    mCache->decodeKeyReadyLength = std::max(mCache->decodeKeyReadyLength, baseLogical + 1);
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128TransposedKSparse(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int layerIndex) {
    if (attnLen <= 1 || attnLen > 8 || kvLen <= 0 || mHeadDim != 128 || mCache == nullptr ||
        !mCache->key || !mCache->decodeKey || !mCache->value ||
        !mCache->sparseQuery || inputs.size() < 3 || mMeta == nullptr ||
        static_cast<int>(mMeta->sparse_query_logical_indices.size()) < attnLen ||
        mNewKvSeqLen < attnLen || mKvNumHead <= 0 || mNumHead <= 0 ||
        mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    auto err = ensureDecodeTransposedKKernel();
    if (err != NO_ERROR) {
        return err;
    }
    const int appendCount = std::max(0, std::min(mMeta->pic_decode_recompute_append_count, attnLen));
    const int prepareLen = std::max(0, kvLen - appendCount);
    err = ensureDecodeKeyReady(prepareLen, true);
    if (err != NO_ERROR) {
        return err;
    }

    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (runtime == nullptr || output == nullptr) {
        return INVALID_VALUE;
    }

    auto& queue = runtime->commandQueue();
    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t appendUs = 0;
    uint64_t attentionUs = 0;
    uint64_t rankUs = 0;

    {
        const uint64_t appendStartUs = profileDetail ? _nowUs() : 0;
        std::vector<uint32_t> appendGws = {
            128u,
            static_cast<uint32_t>(attnLen),
            static_cast<uint32_t>(mBatch * mKvNumHead),
        };
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mDecodeKeyAppendSparseKernel->get().setArg(idx++, appendGws[0]);
        ret |= mDecodeKeyAppendSparseKernel->get().setArg(idx++, appendGws[1]);
        ret |= mDecodeKeyAppendSparseKernel->get().setArg(idx++, appendGws[2]);
        ret |= mDecodeKeyAppendSparseKernel->get().setArg(idx++, openCLBuffer(key));
        ret |= mDecodeKeyAppendSparseKernel->get().setArg(idx++, openCLBuffer(value));
        ret |= mDecodeKeyAppendSparseKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= mDecodeKeyAppendSparseKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= mDecodeKeyAppendSparseKernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
        ret |= mDecodeKeyAppendSparseKernel->get().setArg(idx++, openCLBuffer(mCache->sparseQuery.get()));
        ret |= mDecodeKeyAppendSparseKernel->get().setArg(idx++, mBatch);
        ret |= mDecodeKeyAppendSparseKernel->get().setArg(idx++, mNewKvSeqLen);
        ret |= mDecodeKeyAppendSparseKernel->get().setArg(idx++, attnLen);
        ret |= mDecodeKeyAppendSparseKernel->get().setArg(idx++, mKvNumHead);
        ret |= mDecodeKeyAppendSparseKernel->get().setArg(idx++, mHeadDim);
        ret |= mDecodeKeyAppendSparseKernel->get().setArg(idx++, mCache->maxSlots);
        MNN_CHECK_CL_SUCCESS(ret, "setArg append_sparse_decode_key_value_hd128");
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        const std::vector<uint32_t> appendLws = {32u, 1u, 1u};
        const auto pmcMeta = _makeOpenCLPmcMeta("append_sparse_decode_key_value_hd128", "append", layerIndex,
                                               attnLen, mQuerySeqLen, kvLen, -1, 32, 0,
                                               mNumHead, mKvNumHead, mHeadDim, appendGws, appendLws,
                                               0, 0, appendCount, prepareLen, false);
        _run3DKernelDefaultPmc(mDecodeKeyAppendSparseKernel, appendGws, appendLws, runtime, pmcMeta);
        mCache->decodeKeyReadyLength = std::max(mCache->decodeKeyReadyLength, kvLen);
        if (profileDetail) {
            queue.finish();
            appendUs = _nowUs() - appendStartUs;
        }
    }

    uint64_t causalWork = _sparseLogicalWork(mMeta, attnLen, kvLen);
    const uint64_t causalWorkPerRow = attnLen > 0 ? causalWork / static_cast<uint64_t>(attnLen) : 0;
    const uint32_t lanes = _decodeHD128SparseLaneWidth(runtime, causalWorkPerRow, attnLen);
    const int qTile = _decodeHD128QTile(attnLen);
    std::shared_ptr<KernelWrap> kernel;
    const char* kernelName = nullptr;
    if (qTile == 2) {
        kernel = lanes == 128u ? mDecodeCausalKernelHD128TransposedKQTile2Row128 :
            (lanes == 64u ? mDecodeCausalKernelHD128TransposedKQTile2Row64
                          : mDecodeCausalKernelHD128TransposedKQTile2Row32);
        kernelName = lanes == 128u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_q2_row128" :
            (lanes == 64u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_q2_row64"
                          : "setArg decode_causal_attention_hd128_transposed_k_qtile_q2_row32");
    } else if (qTile == 4) {
        kernel = lanes == 128u ? mDecodeCausalKernelHD128TransposedKQTile4Row128 :
            (lanes == 64u ? mDecodeCausalKernelHD128TransposedKQTile4Row64
                          : mDecodeCausalKernelHD128TransposedKQTile4Row32);
        kernelName = lanes == 128u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_q4_row128" :
            (lanes == 64u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_q4_row64"
                          : "setArg decode_causal_attention_hd128_transposed_k_qtile_q4_row32");
    } else {
        kernel = lanes == 128u ? mDecodeCausalKernelHD128TransposedKQTile8Row128 :
            (lanes == 64u ? mDecodeCausalKernelHD128TransposedKQTile8Row64
                          : mDecodeCausalKernelHD128TransposedKQTile8Row32);
        kernelName = lanes == 128u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_q8_row128" :
            (lanes == 64u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_q8_row64"
                          : "setArg decode_causal_attention_hd128_transposed_k_qtile_q8_row32");
    }
    if (!kernel) {
        return INVALID_VALUE;
    }
    const uint64_t attentionStartUs = profileDetail ? _nowUs() : 0;
    std::vector<uint32_t> gws = {
        lanes,
        static_cast<uint32_t>((attnLen + qTile - 1) / qTile),
        static_cast<uint32_t>(mNumHead * mBatch),
    };
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    ret |= kernel->get().setArg(idx++, gws[0]);
    ret |= kernel->get().setArg(idx++, gws[1]);
    ret |= kernel->get().setArg(idx++, gws[2]);
    ret |= kernel->get().setArg(idx++, openCLBuffer(query));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->sparseQuery.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= kernel->get().setArg(idx++, mScale);
    ret |= kernel->get().setArg(idx++, mBatch);
    ret |= kernel->get().setArg(idx++, mQuerySeqLen);
    ret |= kernel->get().setArg(idx++, attnLen);
    ret |= kernel->get().setArg(idx++, kvLen);
    ret |= kernel->get().setArg(idx++, mCache->maxSlots);
    ret |= kernel->get().setArg(idx++, mNumHead);
    ret |= kernel->get().setArg(idx++, mKvNumHead);
    ret |= kernel->get().setArg(idx++, mHeadDim);
    MNN_CHECK_CL_SUCCESS(ret, kernelName);
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const uint64_t denseWork = static_cast<uint64_t>(attnLen) * static_cast<uint64_t>(kvLen);
    const std::vector<uint32_t> lws = {lanes, 1u, 1u};
    const auto pmcMeta = _makeOpenCLPmcMeta("decode_causal_attention_hd128_transposed_k_sparse_qtile",
                                           "attention", layerIndex, attnLen, mQuerySeqLen, kvLen,
                                           -1, static_cast<int>(lanes), qTile, mNumHead, mKvNumHead,
                                           mHeadDim, gws, lws, denseWork, causalWork,
                                           appendCount, prepareLen, false);
    _run3DKernelDefaultPmc(kernel, gws, lws, runtime, pmcMeta);
    if (profileDetail) {
        queue.finish();
        attentionUs = _nowUs() - attentionStartUs;
    }

    const uint64_t rankStartUs = profileDetail ? _nowUs() : 0;
    auto rankErr = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, attnLen, false);
    if (rankErr != NO_ERROR) {
        return rankErr;
    }
    if (profileDetail) {
        queue.finish();
        rankUs = _nowUs() - rankStartUs;
    }

    if (profile) {
        queue.finish();
        const uint64_t totalUs = _nowUs() - startUs;
        if (profileDetail) {
            MNN_PRINT("OpenCLPagedAttention profile op=decode_causal_attention_hd128_transposed_k_sparse_qtile layer=%d "
                      "query=%d input_query=%d kv_len=%d lane=%u q_tile=%d append_count=%d prepare_len=%d "
                      "dense_kv_work=%llu causal_kv_work=%llu us=%llu append_us=%llu attention_us=%llu "
                      "rank_us=%llu decode_key_ready_hit=%d decode_key_prepared=%d "
                      "decode_prepare_inside_decode=%d decode_prepare_us=%llu prepare_us=%llu "
                      "decode_key_required=%d slot_identity=%d prefix_stable=%d identity_v=%d record_queue=0\n",
                      layerIndex, attnLen, mQuerySeqLen, kvLen, lanes, qTile, appendCount, prepareLen,
                      static_cast<unsigned long long>(denseWork),
                      static_cast<unsigned long long>(causalWork),
                      static_cast<unsigned long long>(totalUs),
                      static_cast<unsigned long long>(appendUs),
                      static_cast<unsigned long long>(attentionUs),
                      static_cast<unsigned long long>(rankUs),
                      mLastDecodeKeyReadyHit ? 1 : 0,
                      mLastDecodeKeyPrepared ? 1 : 0,
                      mLastDecodeKeyPrepareInsideDecode ? 1 : 0,
                      static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                      static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                      mLastDecodeKeyRequiredLen,
                      mLastDecodeKeySlotIdentity ? 1 : 0,
                      mLastDecodeKeyPrefixStable ? 1 : 0,
                      1);
        } else {
            MNN_PRINT("OpenCLPagedAttention profile op=decode_causal_attention_hd128_transposed_k_sparse_qtile layer=%d "
                      "query=%d input_query=%d kv_len=%d lane=%u q_tile=%d append_count=%d prepare_len=%d "
                      "dense_kv_work=%llu causal_kv_work=%llu us=%llu append_us=0 attention_us=%llu rank_us=0 "
                      "decode_key_ready_hit=%d decode_key_prepared=%d "
                      "decode_prepare_inside_decode=%d decode_prepare_us=%llu prepare_us=%llu "
                      "decode_key_required=%d slot_identity=%d prefix_stable=%d identity_v=%d record_queue=0\n",
                      layerIndex, attnLen, mQuerySeqLen, kvLen, lanes, qTile, appendCount, prepareLen,
                      static_cast<unsigned long long>(denseWork),
                      static_cast<unsigned long long>(causalWork),
                      static_cast<unsigned long long>(totalUs),
                      static_cast<unsigned long long>(totalUs),
                      mLastDecodeKeyReadyHit ? 1 : 0,
                      mLastDecodeKeyPrepared ? 1 : 0,
                      mLastDecodeKeyPrepareInsideDecode ? 1 : 0,
                      static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                      static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                      mLastDecodeKeyRequiredLen,
                      mLastDecodeKeySlotIdentity ? 1 : 0,
                      mLastDecodeKeyPrefixStable ? 1 : 0,
                      1);
        }
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128IdentityRecord(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int baseLogical, int layerIndex, uint32_t lanes, std::shared_ptr<KernelWrap> kernel) {
    if (!mOpenCLBackend->isUseRecordQueue() || _profilePagedAttention() || !kernel ||
        attnLen <= 0 || kvLen <= 0 || mCache == nullptr || !mCache->key || !mCache->value) {
        return INVALID_VALUE;
    }
    auto query = inputs[0];
    auto output = outputs[0];
    const uint32_t heads = static_cast<uint32_t>(mNumHead * mBatch);
    const bool needRecord = !mDecodeIdentityRecordValid ||
        mDecodeIdentityRecordLanes != lanes ||
        mDecodeIdentityRecordAttnLen != attnLen ||
        mDecodeIdentityRecordHeads != heads;

    auto updateRecordArgs = [&]() {
        auto& args = mDecodeIdentityRecordUpdateInfo.update_kernel_args;
        if (args.size() < 8) {
            return false;
        }
        mDecodeIdentityRecordQuerySeqLen = mQuerySeqLen;
        mDecodeIdentityRecordBaseLogical = baseLogical;
        mDecodeIdentityRecordKvLen = kvLen;
        mDecodeIdentityRecordMaxSlots = mCache->maxSlots;
        args[0].arg_value = &openCLBuffer(query)();
        args[1].arg_value = &openCLBuffer(mCache->key.get())();
        args[2].arg_value = &openCLBuffer(mCache->value.get())();
        args[3].arg_value = &openCLBuffer(output)();
        args[4].arg_value = &mDecodeIdentityRecordQuerySeqLen;
        args[5].arg_value = &mDecodeIdentityRecordBaseLogical;
        args[6].arg_value = &mDecodeIdentityRecordKvLen;
        args[7].arg_value = &mDecodeIdentityRecordMaxSlots;
        return true;
    };

    if (needRecord) {
        mDecodeIdentityRecordGws0 = lanes;
        mDecodeIdentityRecordGws1 = static_cast<uint32_t>(attnLen);
        mDecodeIdentityRecordGws2 = heads;
        mDecodeIdentityRecordQuerySeqLen = mQuerySeqLen;
        mDecodeIdentityRecordBaseLogical = baseLogical;
        mDecodeIdentityRecordKvLen = kvLen;
        mDecodeIdentityRecordMaxSlots = mCache->maxSlots;

        std::vector<uint32_t> gws = {mDecodeIdentityRecordGws0, mDecodeIdentityRecordGws1,
                                     mDecodeIdentityRecordGws2};
        std::vector<uint32_t> lws = {lanes, 1u, 1u};
        cl_int ret = CL_SUCCESS;
        uint32_t idx = 0;
        ret |= kernel->get().setArg(idx++, gws[0]);
        ret |= kernel->get().setArg(idx++, gws[1]);
        ret |= kernel->get().setArg(idx++, gws[2]);
        ret |= kernel->get().setArg(idx++, openCLBuffer(query));
        ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= kernel->get().setArg(idx++, openCLBuffer(output));
        ret |= kernel->get().setArg(idx++, mScale);
        ret |= kernel->get().setArg(idx++, mBatch);
        ret |= kernel->get().setArg(idx++, mDecodeIdentityRecordQuerySeqLen);
        ret |= kernel->get().setArg(idx++, attnLen);
        ret |= kernel->get().setArg(idx++, mDecodeIdentityRecordBaseLogical);
        ret |= kernel->get().setArg(idx++, mDecodeIdentityRecordKvLen);
        ret |= kernel->get().setArg(idx++, mDecodeIdentityRecordMaxSlots);
        ret |= kernel->get().setArg(idx++, mNumHead);
        ret |= kernel->get().setArg(idx++, mKvNumHead);
        ret |= kernel->get().setArg(idx++, mHeadDim);
        MNN_CHECK_CL_SUCCESS(ret, lanes == 128u ? "record setArg decode_causal_attention_hd128_identity_row128" :
            (lanes == 64u ? "record setArg decode_causal_attention_hd128_identity_row64"
                          : "record setArg decode_causal_attention_hd128_identity_row32"));
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }

        mDecodeIdentityRecordUpdateInfo.update_kernel_args.clear();
        mDecodeIdentityRecordUpdateInfo.update_global_size.clear();
        mDecodeIdentityRecordUpdateInfo.update_local_size.clear();
        mDecodeIdentityRecordUpdateInfo.update_kernel_args.push_back(
            {0, 3, sizeof(cl_mem), &openCLBuffer(query)()});
        mDecodeIdentityRecordUpdateInfo.update_kernel_args.push_back(
            {0, 4, sizeof(cl_mem), &openCLBuffer(mCache->key.get())()});
        mDecodeIdentityRecordUpdateInfo.update_kernel_args.push_back(
            {0, 5, sizeof(cl_mem), &openCLBuffer(mCache->value.get())()});
        mDecodeIdentityRecordUpdateInfo.update_kernel_args.push_back(
            {0, 6, sizeof(cl_mem), &openCLBuffer(output)()});
        mDecodeIdentityRecordUpdateInfo.update_kernel_args.push_back(
            {0, 9, sizeof(mDecodeIdentityRecordQuerySeqLen), &mDecodeIdentityRecordQuerySeqLen});
        mDecodeIdentityRecordUpdateInfo.update_kernel_args.push_back(
            {0, 11, sizeof(mDecodeIdentityRecordBaseLogical), &mDecodeIdentityRecordBaseLogical});
        mDecodeIdentityRecordUpdateInfo.update_kernel_args.push_back(
            {0, 12, sizeof(mDecodeIdentityRecordKvLen), &mDecodeIdentityRecordKvLen});
        mDecodeIdentityRecordUpdateInfo.update_kernel_args.push_back(
            {0, 13, sizeof(mDecodeIdentityRecordMaxSlots), &mDecodeIdentityRecordMaxSlots});
        mDecodeIdentityRecordUpdateInfos.clear();
        mDecodeIdentityRecordUpdateInfos.emplace_back(&mDecodeIdentityRecordUpdateInfo);

        mOpenCLBackend->startRecord(mRecording);
        mOpenCLBackend->recordKernel3d(kernel, gws, lws, &mDecodeIdentityRecordUpdateInfo);
        mOpenCLBackend->endRecord(mRecording);
        mDecodeIdentityRecordValid = true;
        mDecodeIdentityRecordLanes = lanes;
        mDecodeIdentityRecordAttnLen = attnLen;
        mDecodeIdentityRecordHeads = heads;
    }
    if (!updateRecordArgs()) {
        return INVALID_VALUE;
    }
    mOpenCLBackend->addRecord(mRecording, mDecodeIdentityRecordUpdateInfos);
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeAttentionRankCaptureOpenCL(const Tensor* query, int layerIndex,
                                                                          int kvLen, int attnLen,
                                                                          bool queryRowsAreFull) {
    if (mMeta == nullptr || !mMeta->needsPicDecodeAttentionRankCapture(layerIndex)) {
        return NO_ERROR;
    }
    if (query == nullptr || attnLen <= 0 || kvLen <= 0) {
        return NO_ERROR;
    }
    if (mHeadDim != 128 || mCache == nullptr || !mCache->key || !mCache->sparseQuery) {
        return NO_ERROR;
    }
    if (static_cast<int>(mMeta->sparse_query_logical_indices.size()) < attnLen) {
        return NO_ERROR;
    }
    const int picStart = mMeta->pic_decode_attention_pic_start;
    const int picTokenCount = mMeta->pic_decode_attention_pic_token_count;
    const int topM = std::min(mMeta->pic_decode_attention_top_m, picTokenCount);
    if (picStart < 0 || picTokenCount <= 0 || topM <= 0 || picStart + picTokenCount > kvLen ||
        mBatch <= 0 || mNumHead <= 0 || mKvNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return NO_ERROR;
    }
    const int qIndex = attnLen - 1;
    const int qLogical = mMeta->sparse_query_logical_indices[static_cast<size_t>(qIndex)];
    const int qRow = queryRowsAreFull ? qLogical : qIndex;
    if (qLogical < 0 || qLogical >= kvLen || qRow < 0 || qRow >= mQuerySeqLen ||
        picStart + picTokenCount > qLogical + 1) {
        return NO_ERROR;
    }
    if (!mMeta->pic_decode_attention_head_ids.empty()) {
        bool hasValidHead = false;
        for (int head : mMeta->pic_decode_attention_head_ids) {
            if (head >= 0 && head < mNumHead) {
                hasValidHead = true;
                break;
            }
        }
        if (!hasValidHead) {
            return NO_ERROR;
        }
    }
    auto err = ensureDecodeAttentionRankKernel();
    if (err != NO_ERROR) {
        return err;
    }
    err = ensureCacheBlendScoreTemps(picTokenCount, topM, 0);
    if (err != NO_ERROR) {
        return err;
    }
    err = syncDecodeAttentionHeadIds();
    if (err != NO_ERROR) {
        return err;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto& queue = runtime->commandQueue();
    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t scoreUs = 0;
    uint64_t topKUs = 0;
    uint64_t readbackUs = 0;

    cl::Buffer& headIdsBuffer = (mDecodeAttentionHeadIds && !mMeta->pic_decode_attention_head_ids.empty())
        ? openCLBuffer(mDecodeAttentionHeadIds.get())
        : openCLBuffer(mCache->sparseQuery.get());
    const int headIdCount = mMeta->pic_decode_attention_head_ids.empty()
        ? 0
        : static_cast<int>(mMeta->pic_decode_attention_head_ids.size());

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, openCLBuffer(query));
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, headIdsBuffer);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, openCLBuffer(mCacheBlendScores.get()));
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, mBatch);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, mQuerySeqLen);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, mNumHead);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, mKvNumHead);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, kvLen);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, mCache->maxSlots);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, picStart);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, picTokenCount);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, qRow);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, qLogical);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, headIdCount);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, mScale);
    MNN_CHECK_CL_SUCCESS(ret, "setArg decode_attention_pic_rank_score_hd128");
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const uint64_t scoreStartUs = profileDetail ? _nowUs() : 0;
    const size_t localSize = 128;
    const size_t globalSize =
        ((static_cast<size_t>(picTokenCount) + localSize - 1) / localSize) * localSize;
    {
        const std::vector<uint32_t> gws = {static_cast<uint32_t>(globalSize)};
        const std::vector<uint32_t> lws = {static_cast<uint32_t>(localSize)};
        const auto pmcMeta = _makeOpenCLPmcMeta("decode_attention_pic_rank_score_hd128", "rank_score",
                                               layerIndex, attnLen, mQuerySeqLen, kvLen, qLogical, 128, 0,
                                               headIdCount > 0 ? headIdCount : mNumHead, mKvNumHead, mHeadDim,
                                               gws, lws);
        _runOpenCLPmcScope(runtime, pmcMeta, [&](cl::Event* eventPtr) {
            ret = queue.enqueueNDRangeKernel(mDecodeAttentionRankScoreKernelHD128->get(), cl::NullRange,
                                             cl::NDRange(globalSize), cl::NDRange(localSize),
                                             nullptr, eventPtr);
        });
    }
    MNN_CHECK_CL_SUCCESS(ret, "enqueue decode_attention_pic_rank_score_hd128");
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    if (profileDetail) {
        queue.finish();
        scoreUs = _nowUs() - scoreStartUs;
    }

    idx = 0;
    ret = CL_SUCCESS;
    ret |= mCacheBlendTopKKernel->get().setArg(idx++, openCLBuffer(mCacheBlendScores.get()));
    ret |= mCacheBlendTopKKernel->get().setArg(idx++, openCLBuffer(mCacheBlendIndices.get()));
    ret |= mCacheBlendTopKKernel->get().setArg(idx++, picTokenCount);
    ret |= mCacheBlendTopKKernel->get().setArg(idx++, topM);
    MNN_CHECK_CL_SUCCESS(ret, "setArg decode_attention_rank_topk");
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const uint64_t topKStartUs = profileDetail ? _nowUs() : 0;
    {
        const std::vector<uint32_t> gws = {256u};
        const std::vector<uint32_t> lws = {256u};
        const auto pmcMeta = _makeOpenCLPmcMeta("decode_attention_rank_topk", "rank_topk",
                                               layerIndex, attnLen, mQuerySeqLen, kvLen, qLogical, 256, 0,
                                               headIdCount > 0 ? headIdCount : mNumHead, mKvNumHead, mHeadDim,
                                               gws, lws);
        _runOpenCLPmcScope(runtime, pmcMeta, [&](cl::Event* eventPtr) {
            ret = queue.enqueueNDRangeKernel(mCacheBlendTopKKernel->get(), cl::NullRange,
                                             cl::NDRange(256), cl::NDRange(256), nullptr, eventPtr);
        });
    }
    MNN_CHECK_CL_SUCCESS(ret, "enqueue decode_attention_rank_topk");
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    if (profileDetail) {
        queue.finish();
        topKUs = _nowUs() - topKStartUs;
    }

    std::vector<int> selected(static_cast<size_t>(topM), -1);
    const uint64_t readStartUs = profileDetail ? _nowUs() : 0;
    {
        const std::vector<uint32_t> gws = {static_cast<uint32_t>(topM)};
        const auto pmcMeta = _makeOpenCLPmcMeta("decode_attention_rank_readback", "readback",
                                               layerIndex, attnLen, mQuerySeqLen, kvLen, qLogical, 0, 0,
                                               headIdCount > 0 ? headIdCount : mNumHead, mKvNumHead, mHeadDim,
                                               gws, {0u});
        _runOpenCLPmcScope(runtime, pmcMeta, [&](cl::Event* eventPtr) {
            ret = queue.enqueueReadBuffer(openCLBuffer(mCacheBlendIndices.get()), CL_TRUE, 0,
                                          static_cast<size_t>(topM) * sizeof(int), selected.data(),
                                          nullptr, eventPtr);
        });
    }
    MNN_CHECK_CL_SUCCESS(ret, "read decode_attention_rank_topk indices");
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    if (profileDetail) {
        readbackUs = _nowUs() - readStartUs;
    }
    mMeta->setPicDecodeAttentionRankResult(selected, mMeta->pic_decode_attention_step_idx);
    if (profile) {
        queue.finish();
        MNN_PRINT("OpenCLPagedAttention profile op=decode_attention_rank layer=%d pic_tokens=%d top_m=%d "
                  "heads=%d q_logical=%d query=%d input_query=%d kv_len=%d lane=128 us=%llu rank_us=%llu "
                  "score_us=%llu topk_us=%llu readback_us=%llu append_us=0 attention_us=0 record_queue=0 "
                  "decode_key_ready_hit=0 decode_key_prepared=0 decode_prepare_inside_decode=0 "
                  "decode_prepare_us=0 prepare_us=0 slot_identity=%d prefix_stable=1\n",
                  layerIndex, picTokenCount, topM,
                  headIdCount > 0 ? headIdCount : mNumHead, qLogical, attnLen, mQuerySeqLen, kvLen,
                  static_cast<unsigned long long>(_nowUs() - startUs),
                  static_cast<unsigned long long>(_nowUs() - startUs),
                  static_cast<unsigned long long>(scoreUs),
                  static_cast<unsigned long long>(topKUs),
                  static_cast<unsigned long long>(readbackUs),
                  1);
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
        mMeta->beginRequest(std::max(std::max(mNewKvSeqLen, mQuerySeqLen), mMeta->max_tokens));
    }
    int layerIndex = mLayerIndex >= 0 ? mLayerIndex : (mMeta != nullptr ? mMeta->layer_index : 0);
    const bool decodeStep = mMeta != nullptr && mQuerySeqLen == 1 && mNewKvSeqLen == 1 &&
        mMeta->previous > 0 && !mMeta->cacheblend_score_active && !mMeta->pic_graph_active_plan_ready;
    const bool forcePlainSparseQuery = mMeta != nullptr && mMeta->sparse_query_active &&
        mMeta->sparse_query_force_plain_attention;
    const bool picRuntimeActive = mMeta != nullptr &&
        (mMeta->sparse_query_active || mMeta->cacheblend_score_active || mMeta->pic_graph_active_plan_ready);
    const int effectivePicAttentionMode =
        (decodeStep || !picRuntimeActive || forcePlainSparseQuery) ? 0 : mPicAttentionMode;
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
    const bool picDecodeRecompute = mMeta != nullptr && mMeta->pic_decode_recompute_active;
    const bool scoreAttention = effectivePicAttentionMode == 1 && !picDecodeRecompute;
    if (scoreAttention) {
        sparseQuery = false;
    }
    if (mMeta != nullptr) {
        size_t kept = mMeta->previous >= mMeta->remove ? (mMeta->previous - mMeta->remove) : 0;
        baseLogical = static_cast<int>(kept) + reverse;
        kvWriteLen = mMeta->add > 0 ? static_cast<int>(std::min<size_t>(mMeta->add, mNewKvSeqLen)) : mNewKvSeqLen;
    }
    if (sparseQuery) {
        if (!mMeta->validateSparseQueryRows(attnLen, mQuerySeqLen, false)) {
            MNN_ERROR("OpenCLPagedAttention layer %d invalid sparse rows before K/V write, query=%d attn=%d "
                      "active=%d logical_length=%d\n",
                      layerIndex, mQuerySeqLen, attnLen,
                      static_cast<int>(mMeta->sparse_query_logical_indices.size()), mMeta->logical_length);
            return INVALID_VALUE;
        }
        if (mNewKvSeqLen < attnLen) {
            MNN_ERROR("OpenCLPagedAttention layer %d sparse K/V rows %d smaller than active attention rows %d\n",
                      layerIndex, mNewKvSeqLen, attnLen);
            return INVALID_VALUE;
        }
        baseLogical = 0;
        kvWriteLen = attnLen;
    } else if (effectivePicAttentionMode == 2) {
        MNN_ERROR("OpenCL PicSparseAttention layer %d requires active sparse rows from PicScoreAttention\n",
                  layerIndex);
        return INVALID_VALUE;
    }
    int kvLen = sparseQuery ? std::max(0, mMeta->logical_length) : (baseLogical + kvWriteLen);
    bool useMask = mask != nullptr && mask->elementSize() > 1 && mask->getType().code == halide_type_float;
    const bool decodeCausalMask = !useMask || (mMeta != nullptr && mMeta->full_causal_attention_mask);
    const bool ordinaryDecodeCausal = decodeStep && !sparseQuery && attnLen == 1 &&
        mQuerySeqLen == 1 && mNewKvSeqLen == 1;
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    bool ordinaryDecodeHD128Identity = false;
    bool ordinaryDecodeFusedKV = false;
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
    auto err = syncSparseQuery(sparseQuery ? attnLen : 0);
    if (err != NO_ERROR) {
        if (_picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA exec fail sparse_query layer=%d err=%d attn=%d\n",
                      layerIndex, static_cast<int>(err), attnLen);
        }
        return err;
    }
    ordinaryDecodeHD128Identity =
        ordinaryDecodeCausal && decodeCausalMask && mHeadDim == 128 && runtime != nullptr;
    const bool externalReadyForFusedDecode =
        mMeta == nullptr || mMeta->external_segments.empty() ||
        static_cast<int>(mMeta->external_loaded_layers.size()) >= std::max(1, mMeta->layer_nums);
    ordinaryDecodeFusedKV =
        ordinaryDecodeHD128Identity && !mIsKVShared && kvWriteLen == 1 &&
        mCache != nullptr && mCache->key && mCache->value && inputs.size() >= 3 &&
        externalReadyForFusedDecode &&
        !_decodeIdentityAttentionOnlyBenchEnabled() &&
        (mMeta == nullptr || mMeta->file_flag != KVMeta::PendingWrite) &&
        (mMeta == nullptr || !mMeta->needsCacheBlendScoring(layerIndex));
    if (_picOpenCLDebug() && ordinaryDecodeCausal && mHeadDim == 128) {
        MNN_PRINT("PIC OpenCL PA decode hd128 gate layer=%d causal=%d mask=%d runtime=%d "
                  "slot_identity=1 fused=%d kv_shared=%d kv_write=%d inputs=%d external_ready=%d "
                  "file_flag=%d needs_score=%d kv_len=%d gpu=%d\n",
                  layerIndex, ordinaryDecodeCausal ? 1 : 0, decodeCausalMask ? 1 : 0,
                  runtime != nullptr ? 1 : 0,
                  ordinaryDecodeFusedKV ? 1 : 0, mIsKVShared ? 1 : 0, kvWriteLen,
                  static_cast<int>(inputs.size()), externalReadyForFusedDecode ? 1 : 0,
                  mMeta != nullptr ? static_cast<int>(mMeta->file_flag) : -1,
                  mMeta != nullptr && mMeta->needsCacheBlendScoring(layerIndex) ? 1 : 0,
                  kvLen, runtime != nullptr ? static_cast<int>(runtime->getGpuType()) : -1);
    }

    auto& queue = mOpenCLBackend->getOpenCLRuntime()->commandQueue();
    if (!_legacyB863976OpenCL()) {
        _registerExternalLayerMappedTarget(mMeta, layerIndex, mBatch, mKvNumHead, mHeadDim, mBytes, mCache->maxSlots,
                                           mCache->key, mCache->value, queue,
                                           _useAdrenoSourceSlotValueHydrate(mOpenCLBackend->getOpenCLRuntime()));
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
    const bool preEmitExternalHydrated = !shouldHydrateExternal || mMeta->externalLayerLoaded(layerIndex);
    const bool preEmitQueryRowsAreFull = sparseQuery && mQuerySeqLen > attnLen;
    const int decodeTransposedKSparseAppendCount =
        (mMeta != nullptr && picDecodeRecompute) ?
        std::max(0, std::min(mMeta->pic_decode_recompute_append_count, attnLen)) : 0;
    const int decodeTransposedKSparsePrepareLen = std::max(0, kvLen - decodeTransposedKSparseAppendCount);
    const bool decodeTransposedKEnabledGlobal = _decodeTransposedKEnabled();
    const bool decodeTransposedKQ1NeedsDecodeKey =
        decodeTransposedKEnabledGlobal && _decodeTransposedKQ1NeedsDecodeKey();
    const bool decodeTransposedKEnabled =
        decodeTransposedKEnabledGlobal && (attnLen > 1 || decodeTransposedKQ1NeedsDecodeKey);
    const bool decodeTransposedKSparseCandidate =
        decodeTransposedKEnabled && picDecodeRecompute && sparseQuery && decodeCausalMask &&
        mHeadDim == 128 && attnLen > 1 && attnLen <= 8 && !preEmitQueryRowsAreFull &&
        !mIsKVShared && preEmitExternalHydrated && runtime != nullptr &&
        _decodeTransposedKSparseShapeSupported(runtime, attnLen) &&
        mCache != nullptr && mCache->key && mCache->decodeKey && mCache->value &&
        mCache->sparseQuery && inputs.size() >= 3 &&
        (mMeta == nullptr || mMeta->file_flag != KVMeta::PendingWrite) &&
        (mMeta == nullptr || !mMeta->needsCacheBlendScoring(layerIndex));
    const bool decodeTransposedKSparsePrefixReady =
        decodeTransposedKSparseCandidate &&
        mCache->decodeKeyReadyLength >= decodeTransposedKSparsePrepareLen;
    const bool decodeTransposedKSparse =
        decodeTransposedKSparseCandidate && decodeTransposedKSparsePrefixReady;
    const bool decodeTransposedKQ1AttentionOnlyBenchCandidate =
        decodeTransposedKEnabled && _decodeTransposedKAttentionOnlyBenchEnabled() &&
        ordinaryDecodeHD128Identity && !mIsKVShared && kvWriteLen == 1 &&
        mCache != nullptr && mCache->key && mCache->decodeKey && mCache->value &&
        inputs.size() >= 3 && preEmitExternalHydrated &&
        (mMeta == nullptr || mMeta->file_flag != KVMeta::PendingWrite) &&
        (mMeta == nullptr || !mMeta->needsCacheBlendScoring(layerIndex));
    const bool decodeTransposedKQ1AttentionOnlyBenchPrefixReady =
        decodeTransposedKQ1AttentionOnlyBenchCandidate &&
        mCache->decodeKeyReadyLength >= baseLogical;
    const bool decodeTransposedKQ1AttentionOnlyBench =
        decodeTransposedKQ1AttentionOnlyBenchCandidate && decodeTransposedKQ1AttentionOnlyBenchPrefixReady;

    if (!mIsKVShared && kvWriteLen > 0 && !ordinaryDecodeFusedKV && !decodeTransposedKSparse &&
        !decodeTransposedKQ1AttentionOnlyBench) {
        const int total = mBatch * kvWriteLen * mKvNumHead * mHeadDim;
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mCopyKernel->get().setArg(idx++, openCLBuffer(key));
        ret |= mCopyKernel->get().setArg(idx++, openCLBuffer(value));
        ret |= mCopyKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= mCopyKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
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
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        cl_int enqueueRet = CL_SUCCESS;
        const std::vector<uint32_t> gws = {static_cast<uint32_t>(std::max(0, total))};
        const auto pmcMeta = _makeOpenCLPmcMeta("copy_paged_kv", "copy", layerIndex,
                                               kvWriteLen, mQuerySeqLen, kvLen, baseLogical, 0, 0,
                                               mNumHead, mKvNumHead, mHeadDim, gws, {0u},
                                               0, 0, kvWriteLen, 0, false);
        _runOpenCLPmcScope(runtime, pmcMeta, [&](cl::Event* eventPtr) {
            enqueueRet = queue.enqueueNDRangeKernel(mCopyKernel->get(), cl::NullRange, cl::NDRange(total),
                                                    cl::NullRange, nullptr, eventPtr);
        });
        MNN_CHECK_CL_SUCCESS(enqueueRet, "enqueue copy_paged_kv");
        if (enqueueRet != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        if (mCache != nullptr) {
            if (_picOpenCLDebug()) {
                MNN_PRINT("PIC OpenCL PA decode_key invalidate layer=%d cache=%p reason=copy_paged_kv "
                          "base=%d kv_write=%d kv_len=%d old_ready=%d\n",
                          layerIndex, static_cast<void*>(mCache.get()), baseLogical, kvWriteLen, kvLen,
                          mCache->decodeKeyReadyLength);
            }
            mCache->decodeKeyReadyLength = 0;
        }
    }

    const bool q1DecodeKNeedsPrepare = decodeTransposedKQ1NeedsDecodeKey;
    const bool repairDecodeKNeedsPrepare =
        mMeta != nullptr && mMeta->pic_decode_repair_tokens_per_step > 0 &&
        _decodeTransposedKSparseShapeSupported(runtime, mMeta->pic_decode_repair_tokens_per_step + 1);
    if (decodeTransposedKEnabled && (q1DecodeKNeedsPrepare || repairDecodeKNeedsPrepare) &&
        !decodeStep && !sparseQuery && !scoreAttention &&
        !picDecodeRecompute && mHeadDim == 128 && kvLen > 0 && mCache != nullptr &&
        mCache->key && mCache->decodeKey && runtime != nullptr) {
        auto prepareDecodeKeyErr = ensureDecodeKeyReady(kvLen, false);
        if (prepareDecodeKeyErr != NO_ERROR && _picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA decode transposed_k predecode prepare skipped layer=%d err=%d kv_len=%d\n",
                      layerIndex, static_cast<int>(prepareDecodeKeyErr), kvLen);
        }
    }

    err = runCacheBlendScoring(layerIndex, kvLen);
    if (err != NO_ERROR) {
        if (_picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA exec fail scoring layer=%d err=%d kv_len=%d\n",
                      layerIndex, static_cast<int>(err), kvLen);
        }
        return err;
    }
    if ((scoreAttention || picDecodeRecompute) && outputs.size() > 1) {
        err = picDecodeRecompute ? _emitDecodeRecomputeIndicesOpenCL(mMeta, attnLen, outputs[1], mOpenCLBackend)
                                 : _emitActiveIndicesOpenCL(mMeta, layerIndex, kvLen, outputs[1], mOpenCLBackend);
        if (err != NO_ERROR) {
            return err;
        }
        sparseQuery = mMeta != nullptr && mMeta->sparseQueryActiveForLayer(layerIndex);
        if (sparseQuery) {
            const bool allowFullQueryRows = scoreAttention && mQuerySeqLen > attnLen;
            if (!mMeta->validateSparseQueryRows(attnLen, mQuerySeqLen, allowFullQueryRows)) {
                MNN_ERROR("OpenCLPagedAttention layer %d invalid sparse rows after active-index emit, query=%d "
                          "attn=%d active=%d logical_length=%d allow_full_q=%d\n",
                          layerIndex, mQuerySeqLen, attnLen,
                          static_cast<int>(mMeta->sparse_query_logical_indices.size()), mMeta->logical_length,
                          allowFullQueryRows ? 1 : 0);
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
        const size_t keyElements = static_cast<size_t>(kvLen) * mBatch * mKvNumHead * mHeadDim;
        if (keyElements > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return OUT_OF_MEMORY;
        }
        auto exportErr = ensureExternalTemps(keyElements, 1);
        if (exportErr != NO_ERROR) {
            return exportErr;
        }
        const size_t keyBytes = keyElements * mBytes;
        const size_t compactValueBytes = static_cast<size_t>(mBatch) * mKvNumHead * kvLen * mHeadDim * mBytes;
        auto& exportWorkspace = _openCLPagedExportWorkspace();
        std::lock_guard<std::mutex> exportWorkspaceLock(exportWorkspace.mutex);
        if (!_ensureHostWorkspace(exportWorkspace.keyHost, keyBytes) ||
            !_ensureHostWorkspace(exportWorkspace.valueStorageHost, valueBytes) ||
            !_ensureHostWorkspace(exportWorkspace.valueDataHost, compactValueBytes)) {
            MNN_PRINT("OpenCLPagedAttention: failed to reserve export host workspace, key=%zu valueStorage=%zu value=%zu maxSlots=%d kvLen=%d\n",
                      keyBytes, valueBytes, compactValueBytes, mCache->maxSlots, kvLen);
            return OUT_OF_MEMORY;
        }
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
                                                         keyBytes, exportWorkspace.keyHost.data()) == CL_SUCCESS;
        const bool valueExportOk = queue.enqueueReadBuffer(openCLBuffer(mCache->value.get()), CL_TRUE, 0,
                                                           valueBytes, exportWorkspace.valueStorageHost.data()) == CL_SUCCESS;
        if (!keyExportOk || !valueExportOk) {
            MNN_PRINT("OpenCLPagedAttention: failed to export paged KV from GPU, key_ok=%d value_ok=%d key=%zu valueStorage=%zu value=%zu maxSlots=%d kvLen=%d\n",
                      keyExportOk ? 1 : 0, valueExportOk ? 1 : 0, keyBytes, valueBytes, compactValueBytes,
                      mCache->maxSlots, kvLen);
            return OUT_OF_MEMORY;
        }
        for (int l = 0; l < kvLen; ++l) {
            int slot = l;
            if (slot < 0 || slot >= mCache->maxSlots) {
                continue;
            }
            for (int b = 0; b < mBatch; ++b) {
                for (int h = 0; h < mKvNumHead; ++h) {
                    const int8_t* srcV = exportWorkspace.valueStorageHost.data() + ((b * mKvNumHead + h) * mCache->maxSlots + slot) * mHeadDim * mBytes;
                    int8_t* dstV = exportWorkspace.valueDataHost.data() + ((b * mKvNumHead + h) * kvLen + l) * mHeadDim * mBytes;
                    ::memcpy(dstV, srcV, mHeadDim * mBytes);
                }
            }
        }
        if (!_writeBinaryFile(basePath + ".k", exportWorkspace.keyHost.data(), keyBytes)) {
            MNN_PRINT("OpenCLPagedAttention: failed to export key cache: %s\n", (basePath + ".k").c_str());
            return FILE_CREATE_FAILED;
        }
        if (!_writeBinaryFile(basePath + ".v", exportWorkspace.valueDataHost.data(), compactValueBytes)) {
            MNN_PRINT("OpenCLPagedAttention: failed to export value cache: %s\n", (basePath + ".v").c_str());
            MNNRemoveFile((basePath + ".k").c_str());
            return FILE_CREATE_FAILED;
        }
        if (!_writeShapeFile(basePath + ".json", mBatch, mKvNumHead, mHeadDim, kvLen, mBytes, mMeta)) {
            MNN_PRINT("OpenCLPagedAttention: failed to export shape metadata: %s\n", (basePath + ".json").c_str());
            MNNRemoveFile((basePath + ".k").c_str());
            MNNRemoveFile((basePath + ".v").c_str());
            return FILE_CREATE_FAILED;
        }
        if (mLayerIndex < 0) {
            mMeta->layer_index = (mMeta->layer_index + 1) % std::max(1, mMeta->layer_nums);
        }
    }

    if (attnLen <= 0 || kvLen <= 0) {
        return NO_ERROR;
    }
    mScale = (mMeta && mMeta->attn_scale > 0) ? mMeta->attn_scale : (1.0f / std::sqrt(static_cast<float>(mHeadDim)));
    int maskElements = useMask ? static_cast<int>(mask->elementSize()) : 0;
    bool externalHydrated = !shouldHydrateExternal || mMeta->externalLayerLoaded(layerIndex);
    const bool queryRowsAreFull = scoreAttention && sparseQuery && mQuerySeqLen > attnLen;
    int fastMaskKeyLen = 0;
    if (ordinaryDecodeCausal && decodeCausalMask &&
        mCache != nullptr && mCache->key && mCache->value && mCache->sparseQuery) {
        if (ordinaryDecodeCausal && mHeadDim == 128 && runtime != nullptr) {
            if (decodeTransposedKQ1AttentionOnlyBench) {
                const bool profile = _profilePagedAttention();
                const bool profileDetail =
                    profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
                uint64_t setupAppendUs = 0;
                ErrorCode setupErr = ensureDecodeTransposedKKernel();
                if (setupErr == NO_ERROR) {
                    setupErr = ensureDecodeKeyReady(baseLogical, true);
                }
                if (setupErr == NO_ERROR) {
                    setupErr = appendDecodeKeyValueHD128(inputs, baseLogical, profileDetail, &setupAppendUs);
                }
                if (setupErr != NO_ERROR) {
                    if (_picOpenCLDebug()) {
                        MNN_PRINT("PIC OpenCL PA decode transposed_k q1 attention-only setup "
                                  "failed layer=%d err=%d\n",
                                  layerIndex, static_cast<int>(setupErr));
                    }
                    return setupErr;
                }
                if (profile) {
                    MNN_PRINT("OpenCLPagedAttention profile op=decode_transposed_k_q1_attention_only_setup "
                              "layer=%d query=%d input_query=%d kv_len=%d append_us=%llu "
                              "decode_key_ready_hit=%d decode_key_prepared=%d "
                              "decode_prepare_inside_decode=%d decode_prepare_us=%llu prepare_us=%llu "
                              "decode_key_required=%d slot_identity=%d prefix_stable=%d\n",
                              layerIndex, attnLen, mQuerySeqLen, kvLen,
                              static_cast<unsigned long long>(setupAppendUs),
                              mLastDecodeKeyReadyHit ? 1 : 0,
                              mLastDecodeKeyPrepared ? 1 : 0,
                              mLastDecodeKeyPrepareInsideDecode ? 1 : 0,
                              static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                              static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                              mLastDecodeKeyRequiredLen,
                              mLastDecodeKeySlotIdentity ? 1 : 0,
                              mLastDecodeKeyPrefixStable ? 1 : 0);
                }
                auto readonlyBenchErr = runDecodeCausalAttentionHD128TransposedKAppendReadonly(
                    inputs, outputs, kvLen, attnLen, baseLogical, layerIndex, false);
                if (readonlyBenchErr == NO_ERROR) {
                    return NO_ERROR;
                }
                if (_picOpenCLDebug()) {
                    MNN_PRINT("PIC OpenCL PA decode transposed_k q1 attention-only bench "
                              "failed layer=%d err=%d\n",
                              layerIndex, static_cast<int>(readonlyBenchErr));
                }
                return readonlyBenchErr;
            }
            if (ordinaryDecodeFusedKV) {
                if (decodeTransposedKQ1NeedsDecodeKey) {
                    if (_decodeTransposedKReadonlyOnlyEnabled()) {
                        auto readonlyOnlyErr = runDecodeCausalAttentionHD128TransposedKAppendReadonly(
                            inputs, outputs, kvLen, attnLen, baseLogical, layerIndex, false);
                        if (readonlyOnlyErr != NO_ERROR && _picOpenCLDebug()) {
                            MNN_PRINT("PIC OpenCL PA decode transposed_k readonly-only failed layer=%d err=%d\n",
                                      layerIndex, static_cast<int>(readonlyOnlyErr));
                        }
                        return readonlyOnlyErr;
                    }
                    if (_decodeTransposedKAttentionOnlyEnabled()) {
                        auto readonlyErr = runDecodeCausalAttentionHD128TransposedKAppendReadonly(
                            inputs, outputs, kvLen, attnLen, baseLogical, layerIndex, true);
                        if (readonlyErr != NO_ERROR && _picOpenCLDebug()) {
                            MNN_PRINT("PIC OpenCL PA decode transposed_k readonly failed layer=%d err=%d\n",
                                      layerIndex, static_cast<int>(readonlyErr));
                        }
                        return readonlyErr;
                    }
                    if (_decodeGqaFusedKVEnabled() && mKvNumHead > 0 && mNumHead > 0 &&
                        mNumHead % mKvNumHead == 0 && mNumHead / mKvNumHead > 1) {
                        auto transposedGqaErr = runDecodeCausalAttentionHD128TransposedKFusedKVGQA(
                            inputs, outputs, kvLen, attnLen, baseLogical, layerIndex);
                        if (transposedGqaErr == NO_ERROR) {
                            return NO_ERROR;
                        }
                        if (_picOpenCLDebug()) {
                            MNN_PRINT("PIC OpenCL PA decode transposed_k fused_gqa failed layer=%d err=%d\n",
                                      layerIndex, static_cast<int>(transposedGqaErr));
                        }
                    }
                    auto transposedErr = runDecodeCausalAttentionHD128TransposedKFusedKV(
                        inputs, outputs, kvLen, attnLen, baseLogical, layerIndex);
                    if (transposedErr != NO_ERROR && _picOpenCLDebug()) {
                        MNN_PRINT("PIC OpenCL PA decode transposed_k fused failed layer=%d err=%d\n",
                                  layerIndex, static_cast<int>(transposedErr));
                    }
                    return transposedErr;
                }
                if (_decodeGqaFusedKVEnabled()) {
                    auto groupedGqaErr = runDecodeCausalAttentionHD128IdentityFusedKVGQA(
                        inputs, outputs, kvLen, attnLen, baseLogical, layerIndex);
                    if (groupedGqaErr == NO_ERROR) {
                        return NO_ERROR;
                    }
                }
                return runDecodeCausalAttentionHD128IdentityFusedKV(inputs, outputs, kvLen, attnLen, baseLogical,
                                                                    layerIndex);
            }
            if (ordinaryDecodeHD128Identity) {
                return runDecodeCausalAttentionHD128Identity(inputs, outputs, kvLen, attnLen, baseLogical,
                                                             layerIndex);
            }
        }
        if (mHeadDim == 64) {
            return runDecodeCausalAttention(inputs, outputs, kvLen, attnLen, baseLogical, sparseQuery,
                                            queryRowsAreFull);
        }
    }
    if (decodeTransposedKSparse) {
        auto transposedSparseErr = runDecodeCausalAttentionHD128TransposedKSparse(
            inputs, outputs, kvLen, attnLen, layerIndex);
        if (transposedSparseErr == NO_ERROR) {
            return NO_ERROR;
        }
        if (_picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA decode transposed_k sparse failed layer=%d err=%d q=%d kv_len=%d\n",
                      layerIndex, static_cast<int>(transposedSparseErr), attnLen, kvLen);
        }
        return transposedSparseErr;
    }
    if (canUseFastPrefill(mask, baseLogical, attnLen, kvLen, sparseQuery, externalHydrated, &fastMaskKeyLen)) {
        return runFastPrefill(inputs, outputs, kvLen, fastMaskKeyLen);
    }
    const bool canSparseFastPrefill =
        canUseSparseFastPrefill(mask, attnLen, kvLen, externalHydrated, queryRowsAreFull);
    int sparseQSplitMaskKeyLen = 0;
    const bool canSparseQSplitPrefill =
        canUseSparseQSplitPrefill(mask, attnLen, kvLen, externalHydrated, queryRowsAreFull,
                                  &sparseQSplitMaskKeyLen);
    const bool picScoreSparse = queryRowsAreFull && mPicAttentionMode == 1 &&
        mMeta != nullptr && mMeta->sparse_query_active;
    const bool picLaterSparse = !queryRowsAreFull && mPicAttentionMode == 2 &&
        mMeta != nullptr && mMeta->sparse_query_active;
    if (picScoreSparse) {
        auto runtime = mOpenCLBackend->getOpenCLRuntime();
        const int layerIndex = _pagedAttentionProfileLayerIndex(mLayerIndex, mMeta);
        const bool useScoreFlash = (mHeadDim != 128) ||
            _preferAdrenoScoreSparseFlash(runtime, mBatch, mKvNumHead, mHeadDim, attnLen, kvLen);
        const uint32_t sparseFamily = useScoreFlash ? kScoreSparseFamilyFlash : kScoreSparseFamilyQSplit;
        if (_profilePagedAttention()) {
            MNN_PRINT("OpenCLPagedAttention profile op=score_sparse_family layer=%d query=%d kv_len=%d family=%s source=fixed_device\n",
                      layerIndex, attnLen, kvLen, _scoreSparseFamilyName(sparseFamily));
        }
        if (useScoreFlash) {
            if (!canSparseFastPrefill) {
                MNN_ERROR("OpenCL PicScoreAttention layer %d requires score flash path, "
                          "but sparse flash is unsupported for query=%d kv_len=%d head_dim=%d\n",
                          layerIndex, attnLen, kvLen, mHeadDim);
                return NOT_SUPPORT;
            }
            return runSparseFastPrefill(inputs, outputs, kvLen, attnLen, queryRowsAreFull);
        }
        if (!canSparseQSplitPrefill) {
            MNN_ERROR("OpenCL PicScoreAttention layer %d requires score qsplit path, "
                      "but qsplit is unsupported for query=%d kv_len=%d head_dim=%d\n",
                      layerIndex, attnLen, kvLen, mHeadDim);
            return NOT_SUPPORT;
        }
        return runSparseQSplitPrefill(inputs, outputs, kvLen, attnLen, queryRowsAreFull, sparseQSplitMaskKeyLen);
    }
    if (picLaterSparse) {
        if (!canSparseFastPrefill) {
            const int layerIndex = _pagedAttentionProfileLayerIndex(mLayerIndex, mMeta);
            MNN_ERROR("OpenCL PicSparseAttention layer %d requires sparse flash production path, "
                      "but sparse flash is unsupported for query=%d kv_len=%d head_dim=%d\n",
                      layerIndex, attnLen, kvLen, mHeadDim);
            return NOT_SUPPORT;
        }
        return runSparseFastPrefill(inputs, outputs, kvLen, attnLen, queryRowsAreFull);
    }
    if (canSparseFastPrefill) {
        return runSparseFastPrefill(inputs, outputs, kvLen, attnLen, queryRowsAreFull);
    }
    if (canSparseQSplitPrefill) {
        return runSparseQSplitPrefill(inputs, outputs, kvLen, attnLen, queryRowsAreFull, sparseQSplitMaskKeyLen);
    }
    const bool benchForceRowKernel = _envFlagEnabled("MNN_PAGED_ATTENTION_BENCH_FORCE_ROW_KERNEL", false) ||
        _envFlagEnabled("MNN_PAGED_ATTENTION_BENCH_FORCE_V2_KERNEL", false);
    const bool disableRowKernel = _envFlagEnabled("MNN_PAGED_ATTENTION_OPENCL_DISABLE_ROW_KERNEL", false);
    const bool rowKernelSupported = mHeadDim <= 256;
    const bool useRowKernel = rowKernelSupported && !disableRowKernel &&
        ((sparseQuery && attnLen >= 1) || benchForceRowKernel);
    const bool profileGeneric = _profilePagedAttention();
    const bool traceProgressGeneric = _tracePagedAttentionProgress();
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
        if (traceProgressGeneric) {
            MNN_PRINT("OpenCLPagedAttention trace phase=row_dispatch layer=%d query=%d attn=%d kv_write=%d kv_len=%d sparse=%d full_q=%d total_rows=%d\n",
                      layerIndex, mQuerySeqLen, attnLen, kvWriteLen, kvLen, sparseQuery ? 1 : 0,
                      queryRowsAreFull ? 1 : 0, totalRows);
            std::fflush(stdout);
        }
        queue.enqueueNDRangeKernel(mAttentionRowKernel->get(), cl::NullRange, cl::NDRange(totalRows), cl::NullRange);
        err = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, attnLen, queryRowsAreFull);
        if (err != NO_ERROR) {
            return err;
        }
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
    if (traceProgressGeneric) {
        MNN_PRINT("OpenCLPagedAttention trace phase=generic_dispatch layer=%d query=%d attn=%d kv_write=%d kv_len=%d sparse=%d full_q=%d total=%d\n",
                  layerIndex, mQuerySeqLen, attnLen, kvWriteLen, kvLen, sparseQuery ? 1 : 0,
                  queryRowsAreFull ? 1 : 0, total);
        std::fflush(stdout);
    }
    queue.enqueueNDRangeKernel(mAttentionKernel->get(), cl::NullRange, cl::NDRange(total), cl::NullRange);
    err = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, attnLen, queryRowsAreFull);
    if (err != NO_ERROR) {
        return err;
    }
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
