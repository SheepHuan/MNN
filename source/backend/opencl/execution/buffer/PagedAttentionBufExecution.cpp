//
//  PagedAttentionBufExecution.cpp
//  MNN
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/PagedAttentionBufExecution.hpp"
#include "backend/opencl/execution/buffer/PagedAttentionBufExecutionExternal.hpp"
#include "backend/opencl/execution/buffer/PagedAttentionAdrenoUtils.hpp"
#include "backend/opencl/execution/buffer/PagedAttentionMaliUtils.hpp"
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

static thread_local int gSparseQSplitChunkOverride = 0;
static thread_local bool gSparseQSplitTuneInProgress = false;
enum ScoreSparsePrefillFamily : uint32_t {
    kScoreSparseFamilyQSplit = 0,
    kScoreSparseFamilyFlash = 1,
};
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

static bool _picOpenCLDebug() {
    return _envFlagEnabled("MNN_PIC_DECODE_DEBUG", false);
}

static bool _decodeGqaFusedKVEnabled() {
    static const bool enabled = _envFlagEnabled("MNN_PAGED_ATTENTION_DECODE_GQA_FUSED", false);
    return enabled;
}

enum DecodeGatePolicy : uint32_t {
    kDecodeGateAuto = 0,
    kDecodeGateForceOff = 1,
    kDecodeGateForceOn = 2,
};

static DecodeGatePolicy _decodeGatePolicy(const char* name) {
    const char* value = ::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return kDecodeGateAuto;
    }
    return value[0] != '0' ? kDecodeGateForceOn : kDecodeGateForceOff;
}

static DecodeGatePolicy _decodeQ1TransposedKPolicy() {
    return _decodeGatePolicy("MNN_PAGED_ATTENTION_DECODE_Q1_TRANSPOSED_K");
}

static DecodeGatePolicy _decodeRepairSparseQTilePolicy() {
    return _decodeGatePolicy("MNN_PAGED_ATTENTION_DECODE_REPAIR_SPARSE_QTILE");
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

static bool _legacyB863976OpenCL();

static bool _decodeRepairSparseQTileShapeSupported(OpenCLRuntime* runtime, int attnLen) {
    if (runtime == nullptr || attnLen <= 0 || attnLen > 8) {
        return false;
    }
    return !_legacyB863976OpenCL();
}

static bool _decodeRepairSparseQTileDefaultEnabled(OpenCLRuntime* runtime, int attnLen) {
    if (!_decodeRepairSparseQTileShapeSupported(runtime, attnLen)) {
        return false;
    }
    return PagedAttentionMali::decodeRepairSparseQTileDefaultEnabled(runtime, _legacyB863976OpenCL(), attnLen) ||
           (runtime != nullptr && runtime->getGpuType() == GpuType::ADRENO);
}

static bool _decodeRepairSparseQTileEnabled(OpenCLRuntime* runtime, int attnLen) {
    const DecodeGatePolicy policy = _decodeRepairSparseQTilePolicy();
    if (policy == kDecodeGateForceOn) {
        return _decodeRepairSparseQTileShapeSupported(runtime, attnLen);
    }
    if (policy == kDecodeGateForceOff) {
        return false;
    }
    return _decodeRepairSparseQTileDefaultEnabled(runtime, attnLen);
}

static bool _decodeRepairFusedAppendEnabled() {
    // Disabled after OrangePi Qwen3-4B ctx1024 A/B on 2026-07-05:
    // even with a single-kernel record path, fused append regressed x0/x1 TPOT.
    return false;
}

static bool _decodeRepairQ1IdentityFusedKVEnabled() {
    // Disabled after Rhino/Adreno A/B: this x0-only fused identity-K diagnostic
    // left PIC x0 still +23..44 ms/token vs true normal and regressed Llama/Qwen.
    return false;
}

static bool _decodeRepairQ1GqaEnabled() {
    // Disabled after Rhino/Adreno A/B: reducing workgroups to kv_heads hurt
    // occupancy and regressed all tested x0 model/context pairs.
    return false;
}

static bool _decodeRepairKeyCacheQTileV2Enabled(OpenCLRuntime* runtime, int attnLen) {
    // Disabled after Rhino/Adreno A/B: skipping decode_key writes did not remove
    // the PIC x0 gap and regressed Llama/Qwen x0.
    (void)runtime;
    (void)attnLen;
    return false;
}

static bool _decodeQ1TransposedKVariantEnabled() {
    // q=1 production decode uses the identity fused-KV family. Keep transposed-K q=1
    // as an explicit A/B or profile variant until it consistently wins formal TPOT.
    if (_decodeTransposedKReadonlyOnlyEnabled() || _decodeTransposedKAttentionOnlyEnabled() ||
        _decodeTransposedKAttentionOnlyBenchEnabled()) {
        return true;
    }
    const DecodeGatePolicy policy = _decodeQ1TransposedKPolicy();
    if (policy == kDecodeGateForceOn) {
        return true;
    }
    return false;
}

static bool _legacyB863976OpenCL() {
    return _envFlagEnabled("MNN_PAGED_ATTENTION_OPENCL_LEGACY_B863976", false);
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


static bool _useAdrenoSourceSlotValueHydrate(OpenCLRuntime* runtime) {
    return PagedAttentionAdreno::useAdrenoSourceSlotValueHydrate(runtime, _legacyB863976OpenCL());
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

static bool _allowDirectPagedCacheOpenCL() {
    if (_legacyB863976OpenCL()) {
        return false;
    }
    if (!_envFlagEnabled("MNN_PAGED_ATTENTION_ZERO_COPY_CACHE", true)) {
        return false;
    }
    return true;
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
    const bool q1DecodeKNeedsPrepare = _decodeQ1TransposedKVariantEnabled();
    const int decodeRepairActiveRows = mMeta->pic_decode_repair_tokens_per_step + 1;
    const bool repairDecodeKNeedsPrepare =
        mMeta->pic_decode_repair_enabled &&
        _decodeRepairSparseQTileEnabled(runtime, decodeRepairActiveRows) &&
        !_decodeRepairKeyCacheQTileV2Enabled(runtime, decodeRepairActiveRows);
    const bool needsDecodeKey =
        mHeadDim == 128 && (q1DecodeKNeedsPrepare || repairDecodeKNeedsPrepare);
    if (needsDecodeKey) {
        err = ensureDecodeKeyReady(kvLen, false);
        if (err != NO_ERROR) {
            return err;
        }
    }
    queue.finish();
    return NO_ERROR;
}

// Disabled after Rhino/Adreno A/B on 2026-07-05:
// MNN_PAGED_ATTENTION_DECODE_REPAIR_Q1_IDENTITY_FUSED_KV did not remove the
// PIC x0 gap and regressed larger models; the implementation and record state
// were removed to keep the decode-repair path readable.

// Disabled after OrangePi Qwen3-4B ctx1024 A/B on 2026-07-05:
// fused append + single record regressed x0/x1 TPOT compared with the default
// two-kernel append+attention record path; the single-record implementation was removed.

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
    const int decodeRepairSparseQTileAppendCount =
        (mMeta != nullptr && picDecodeRecompute) ?
        std::max(0, std::min(mMeta->pic_decode_recompute_append_count, attnLen)) : 0;
    const int decodeRepairSparseQTilePrepareLen = std::max(0, kvLen - decodeRepairSparseQTileAppendCount);
    const bool decodeRepairSparseQTileEnabled =
        _decodeRepairSparseQTileEnabled(runtime, attnLen);
    const bool decodeRepairKeyCacheQTileV2 =
        decodeRepairSparseQTileEnabled && _decodeRepairKeyCacheQTileV2Enabled(runtime, attnLen) &&
        !_decodeRepairFusedAppendEnabled();
    const bool decodeQ1TransposedKVariant = _decodeQ1TransposedKVariantEnabled();
    const bool decodeNeedsTransposedK =
        (decodeRepairSparseQTileEnabled && !decodeRepairKeyCacheQTileV2) || decodeQ1TransposedKVariant;
    const bool decodeRepairSparseQTileCandidate =
        decodeRepairSparseQTileEnabled && picDecodeRecompute && sparseQuery && decodeCausalMask &&
        mHeadDim == 128 && attnLen >= 1 && attnLen <= 8 && !preEmitQueryRowsAreFull &&
        !mIsKVShared && preEmitExternalHydrated && runtime != nullptr &&
        _decodeRepairSparseQTileShapeSupported(runtime, attnLen) &&
        mCache != nullptr && mCache->key && (decodeRepairKeyCacheQTileV2 || mCache->decodeKey) && mCache->value &&
        mCache->sparseQuery && inputs.size() >= 3 &&
        (mMeta == nullptr || mMeta->file_flag != KVMeta::PendingWrite) &&
        (mMeta == nullptr || !mMeta->needsCacheBlendScoring(layerIndex));
    const bool decodeRepairSparseQTilePrefixReady =
        decodeRepairSparseQTileCandidate &&
        (decodeRepairKeyCacheQTileV2 || mCache->decodeKeyReadyLength >= decodeRepairSparseQTilePrepareLen);
    const bool decodeRepairSparseQTile = decodeRepairSparseQTileCandidate;
    const bool decodeQ1TransposedKRoute = ordinaryDecodeFusedKV && decodeQ1TransposedKVariant;
    const bool decodeTransposedKQ1AttentionOnlyBenchCandidate =
        _decodeTransposedKAttentionOnlyBenchEnabled() &&
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

    if (_picOpenCLDebug() && mHeadDim == 128 && (ordinaryDecodeCausal || picDecodeRecompute)) {
        MNN_PRINT("PIC OpenCL PA decode hd128 route layer=%d ordinary_q1=%d repair=%d q=%d kv_len=%d "
                  "q1_transposed_enabled=%d q1_transposed_route=%d "
                  "repair_qtile_enabled=%d repair_qtile_candidate=%d "
                  "repair_qtile_prefix_ready=%d repair_qtile_route=%d q1_attention_only_bench=%d "
                  "repair_keycache_qtile_v2=%d repair_q1_identity_fused_kv=%d repair_q1_gqa=%d "
                  "record_queue=%d q1_policy=%u repair_policy=%u "
                  "decode_key_ready=%d sparse_prepare_len=%d sparse_append=%d gqa_variant=%d\n",
                  layerIndex, ordinaryDecodeCausal ? 1 : 0, picDecodeRecompute ? 1 : 0, attnLen, kvLen,
                  decodeQ1TransposedKVariant ? 1 : 0, decodeQ1TransposedKRoute ? 1 : 0,
                  decodeRepairSparseQTileEnabled ? 1 : 0, decodeRepairSparseQTileCandidate ? 1 : 0,
                  decodeRepairSparseQTilePrefixReady ? 1 : 0, decodeRepairSparseQTile ? 1 : 0,
                  decodeTransposedKQ1AttentionOnlyBench ? 1 : 0,
                  (decodeRepairSparseQTile && decodeRepairKeyCacheQTileV2) ? 1 : 0,
                  (decodeRepairSparseQTile && attnLen == 1 && _decodeRepairQ1IdentityFusedKVEnabled()) ? 1 : 0,
                  (decodeRepairSparseQTile && attnLen == 1 && _decodeRepairQ1GqaEnabled() &&
                   mKvNumHead > 0 && mNumHead > 0 && mNumHead % mKvNumHead == 0 &&
                   mNumHead / mKvNumHead > 1 && mNumHead / mKvNumHead <= 8) ? 1 : 0,
                  mOpenCLBackend->isUseRecordQueue() ? 1 : 0,
                  static_cast<uint32_t>(_decodeQ1TransposedKPolicy()),
                  static_cast<uint32_t>(_decodeRepairSparseQTilePolicy()),
                  mCache != nullptr ? mCache->decodeKeyReadyLength : -1,
                  decodeRepairSparseQTilePrepareLen, decodeRepairSparseQTileAppendCount,
                  _decodeGqaFusedKVEnabled() ? 1 : 0);
    }

    if (!mIsKVShared && kvWriteLen > 0 && !ordinaryDecodeFusedKV && !decodeRepairSparseQTile &&
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

    const bool q1DecodeKNeedsPrepare = decodeQ1TransposedKVariant;
    const int decodeRepairActiveRowsForPrepare =
        mMeta != nullptr ? mMeta->pic_decode_repair_tokens_per_step + 1 : attnLen;
    const bool repairDecodeKNeedsPrepare =
        mMeta != nullptr && mMeta->pic_decode_repair_enabled &&
        _decodeRepairSparseQTileEnabled(runtime, decodeRepairActiveRowsForPrepare) &&
        !_decodeRepairKeyCacheQTileV2Enabled(runtime, decodeRepairActiveRowsForPrepare);
    if (decodeNeedsTransposedK && (q1DecodeKNeedsPrepare || repairDecodeKNeedsPrepare) &&
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
    err = runFusionRAGOnlineScoring(layerIndex, kvLen, query);
    if (err != NO_ERROR) {
        if (_picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA exec fail fusionrag scoring layer=%d err=%d kv_len=%d\n",
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
                if (decodeQ1TransposedKVariant) {
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
    if (decodeRepairSparseQTile) {
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
