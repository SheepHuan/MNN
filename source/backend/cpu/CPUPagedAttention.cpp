//
//  CPUPagedAttention.cpp
//  MNN
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "CPUPagedAttention.hpp"
#include "CPUBackend.hpp"
#include "compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/MNNFileUtils.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

namespace MNN {

static inline bool _picDebugEnabled() {
    return std::getenv("MNN_PIC_DECODE_DEBUG") != nullptr;
}

static inline float _pagedRead(const int8_t* ptr, int index, int bytes) {
#ifdef __aarch64__
    if (bytes == 2) {
        return static_cast<float>(((const __fp16*)ptr)[index]);
    }
#endif
    return ((const float*)ptr)[index];
}

static inline void _pagedWrite(int8_t* ptr, int index, float val, int bytes) {
#ifdef __aarch64__
    if (bytes == 2) {
        ((__fp16*)ptr)[index] = static_cast<__fp16>(val);
        return;
    }
#endif
    ((float*)ptr)[index] = val;
}

static inline int _reverseCount(const KVMeta* meta) {
    if (meta == nullptr || meta->n_reserve <= 0 || meta->reserve == nullptr) {
        return 0;
    }
    int revert = meta->computeReverseSize();
    return std::max(0, revert);
}

static inline float _readFloatMask(const Tensor* mask, int q, int logicalK, int queryLen, int kvLen, int bytes) {
    if (mask == nullptr || mask->elementSize() <= 1) {
        return 0.0f;
    }
    if (mask->getType().code != halide_type_float) {
        return 0.0f;
    }
    int elements = static_cast<int>(mask->elementSize());
    int maskCols = elements >= queryLen * kvLen ? kvLen : queryLen;
    if (maskCols <= 0) {
        return 0.0f;
    }
    int gap = kvLen - maskCols;
    if (logicalK < gap) {
        return 0.0f;
    }
    int col = logicalK - gap;
    int idx = q * maskCols + col;
    if (col < 0 || col >= maskCols || idx < 0 || idx >= elements) {
        return 0.0f;
    }
    return _pagedRead(mask->host<int8_t>(), idx, bytes);
}

static ErrorCode _emitActiveIndicesCPU(PagedKVMeta* meta, int layerIndex, int kvLen, Tensor* output,
                                       bool activateRows) {
    if (output == nullptr) {
        return NO_ERROR;
    }
    const int budget = static_cast<int>(output->elementSize());
    std::vector<int> active;
    if (meta != nullptr && meta->pic_decode_recompute_active && budget > 0) {
        if (static_cast<int>(meta->sparse_query_logical_indices.size()) < budget) {
            return INVALID_VALUE;
        }
        active.reserve(static_cast<size_t>(budget));
        for (int i = 0; i < budget; ++i) {
            active.emplace_back(i);
        }
    } else if (activateRows && meta != nullptr) {
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
        MNN_ERROR("CPUPagedAttention layer %d active index count mismatch, budget=%d active=%d\n",
                  layerIndex, budget, static_cast<int>(active.size()));
        return INVALID_VALUE;
    }
    if (activateRows && meta != nullptr && !active.empty() && !meta->activatePicRows(active, layerIndex, kvLen)) {
        return INVALID_VALUE;
    }
    auto ptr = output->host<int32_t>();
    if (ptr != nullptr && !active.empty()) {
        ::memcpy(ptr, active.data(), active.size() * sizeof(int32_t));
    }
    if (_picDebugEnabled()) {
        int first = active.empty() ? -1 : active.front();
        int second = active.size() > 1 ? active[1] : -1;
        int last = active.empty() ? -1 : active.back();
        std::fprintf(stderr,
                     "CPU PA active layer=%d budget=%d kvLen=%d activate=%d first=%d second=%d last=%d\n",
                     layerIndex, budget, kvLen, activateRows ? 1 : 0, first, second, last);
        std::fflush(stderr);
    }
    return NO_ERROR;
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

static bool _readBinaryFile(const std::string& path, std::vector<int8_t>& data) {
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is.good()) {
        return false;
    }
    auto size = is.tellg();
    if (size < 0) {
        return false;
    }
    data.resize(static_cast<size_t>(size));
    is.seekg(0, std::ios::beg);
    if (!data.empty()) {
        is.read(reinterpret_cast<char*>(data.data()), size);
    }
    return is.good() || is.eof();
}

static int _ropeDimForExport(const KVMeta* meta, int headDim) {
    if (meta == nullptr || meta->rope_dim <= 0) {
        return headDim;
    }
    return std::min(headDim, meta->rope_dim);
}

static float _ropeInvFreqForExport(const KVMeta* meta, int pairIndex, int ropeDim) {
    const float theta = (meta != nullptr && meta->rope_theta > 0.0f) ? meta->rope_theta : 10000.0f;
    float invFreq = std::pow(theta, -static_cast<float>(2 * pairIndex) / static_cast<float>(ropeDim));
    if (meta == nullptr || meta->rope_type != "llama3") {
        return invFreq;
    }
    const float factor = std::max(meta->rope_scaling_factor, 1.0f);
    const float lowFreqFactor = std::max(meta->rope_scaling_low_freq_factor, 1.0e-6f);
    const float highFreqFactor = std::max(meta->rope_scaling_high_freq_factor, 1.0e-6f);
    const int oldContext = meta->rope_scaling_original_max_position_embeddings > 0
        ? meta->rope_scaling_original_max_position_embeddings
        : meta->max_position_embeddings;
    if (oldContext <= 0 || factor == 1.0f || lowFreqFactor == highFreqFactor) {
        return invFreq;
    }
    constexpr float kTwoPi = 6.28318530717958647692f;
    const float wavelen = kTwoPi / invFreq;
    const float lowFreqWavelen = static_cast<float>(oldContext) / lowFreqFactor;
    const float highFreqWavelen = static_cast<float>(oldContext) / highFreqFactor;
    float scaled = wavelen > lowFreqWavelen ? invFreq / factor : invFreq;
    if (wavelen >= highFreqWavelen && wavelen <= lowFreqWavelen) {
        const float smooth = (static_cast<float>(oldContext) / wavelen - lowFreqFactor) /
                             (highFreqFactor - lowFreqFactor);
        scaled = (1.0f - smooth) * invFreq / factor + smooth * invFreq;
    }
    return scaled;
}

static float _ropeInvFreqForSegment(const PagedKVExternalSegment& segment, int pairIndex, int ropeDim) {
    const float theta = segment.ropeTheta > 0.0f ? segment.ropeTheta : 10000.0f;
    float invFreq = std::pow(theta, -static_cast<float>(2 * pairIndex) / static_cast<float>(ropeDim));
    if (segment.ropeType != "llama3") {
        return invFreq;
    }
    const float factor = std::max(segment.ropeScalingFactor, 1.0f);
    const float lowFreqFactor = std::max(segment.ropeScalingLowFreqFactor, 1.0e-6f);
    const float highFreqFactor = std::max(segment.ropeScalingHighFreqFactor, 1.0e-6f);
    const int oldContext = segment.ropeScalingOriginalMaxPositionEmbeddings > 0
        ? segment.ropeScalingOriginalMaxPositionEmbeddings
        : segment.maxPositionEmbeddings;
    if (oldContext <= 0 || factor == 1.0f || lowFreqFactor == highFreqFactor) {
        return invFreq;
    }
    constexpr float kTwoPi = 6.28318530717958647692f;
    const float wavelen = kTwoPi / invFreq;
    const float lowFreqWavelen = static_cast<float>(oldContext) / lowFreqFactor;
    const float highFreqWavelen = static_cast<float>(oldContext) / highFreqFactor;
    float scaled = wavelen > lowFreqWavelen ? invFreq / factor : invFreq;
    if (wavelen >= highFreqWavelen && wavelen <= lowFreqWavelen) {
        const float smooth = (static_cast<float>(oldContext) / wavelen - lowFreqFactor) /
                             (highFreqFactor - lowFreqFactor);
        scaled = (1.0f - smooth) * invFreq / factor + smooth * invFreq;
    }
    return scaled;
}

static void _inverseRopeKeyData(std::vector<int8_t>& keyData, int tokenCount, int batch, int kvHeads, int headDim,
                                int bytes, const KVMeta* meta) {
    int ropeDim = _ropeDimForExport(meta, headDim);
    ropeDim = std::min(ropeDim, headDim);
    ropeDim = (ropeDim / 2) * 2;
    if (ropeDim <= 0) {
        return;
    }
    const int half = ropeDim / 2;
    const float invAttentionScale = (meta != nullptr && meta->rope_attention_scaling > 0.0f)
        ? (1.0f / meta->rope_attention_scaling)
        : 1.0f;
    for (int l = 0; l < tokenCount; ++l) {
        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < kvHeads; ++h) {
                int base = ((l * batch + b) * kvHeads + h) * headDim;
                for (int p = 0; p < half; ++p) {
                    const float angle = static_cast<float>(l) * _ropeInvFreqForExport(meta, p, ropeDim);
                    const float c = std::cos(angle);
                    const float s = std::sin(angle);
                    const int first = base + p;
                    const int second = base + p + half;
                    const float y0 = _pagedRead(keyData.data(), first, bytes);
                    const float y1 = _pagedRead(keyData.data(), second, bytes);
                    _pagedWrite(keyData.data(), first, (y0 * c + y1 * s) * invAttentionScale, bytes);
                    _pagedWrite(keyData.data(), second, (y1 * c - y0 * s) * invAttentionScale, bytes);
                }
            }
        }
    }
}

static bool _writeShapeFile(const std::string& path, int batch, int kvHeads, int headDim, int tokenCount, int bytes,
                            const KVMeta* meta) {
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

static ErrorCode _restoreExternalSegmentsCPU(PagedKVMeta* meta, int layerIndex, int batch, int kvHeads, int headDim,
                                             int bytes, int maxSlots, int kvLen, int8_t* keyCache,
                                             int8_t* valueCache) {
    if (meta == nullptr || meta->external_segments.empty() || meta->externalLayerLoaded(layerIndex)) {
        return NO_ERROR;
    }
    for (const auto& segment : meta->external_segments) {
        auto layer = segment.layer(layerIndex);
        if (layer == nullptr) {
            MNN_ERROR("CPUPagedAttention layer %d missing external PIC KV for cache %s\n", layerIndex,
                      segment.cacheName.c_str());
            return INVALID_VALUE;
        }
        const int segBatch = segment.batch > 0 ? segment.batch : batch;
        const int segKvHeads = segment.kvHeads > 0 ? segment.kvHeads : kvHeads;
        const int segHeadDim = segment.headDim > 0 ? segment.headDim : headDim;
        const int segBytes = segment.dtypeBytes > 0 ? segment.dtypeBytes : bytes;
        if (segBatch != batch || segKvHeads != kvHeads || segHeadDim != headDim || segBytes != bytes) {
            MNN_ERROR("CPUPagedAttention external KV shape mismatch at layer %d, cache %s\n", layerIndex,
                      segment.cacheName.c_str());
            return INVALID_VALUE;
        }
        if (segment.keyRopeState != "canonical_no_rope" || segment.ropePairing != "half") {
            MNN_ERROR("CPUPagedAttention external KV must be canonical_no_rope/half, got %s/%s\n",
                      segment.keyRopeState.c_str(), segment.ropePairing.c_str());
            return INVALID_VALUE;
        }
        if (segment.logicalStart + segment.tokenCount > static_cast<size_t>(kvLen)) {
            MNN_ERROR("CPUPagedAttention external KV range exceeds visible KV length at layer %d\n", layerIndex);
            return INVALID_VALUE;
        }
        std::vector<int8_t> keyData;
        std::vector<int8_t> valueData;
        if (!_readBinaryFile(layer->keyPath, keyData) || !_readBinaryFile(layer->valuePath, valueData)) {
            MNN_ERROR("CPUPagedAttention failed to read external PIC KV files for layer %d\n", layerIndex);
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
        const size_t expectedKey = sourceTokenCount * static_cast<size_t>(batch) * kvHeads * headDim * bytes;
        const size_t expectedValue = static_cast<size_t>(batch) * kvHeads * sourceTokenCount * headDim * bytes;
        if (keyData.size() < expectedKey || valueData.size() < expectedValue) {
            MNN_ERROR("CPUPagedAttention external PIC KV file is too small at layer %d\n", layerIndex);
            return INVALID_VALUE;
        }
        int ropeDim = segment.ropeDim > 0 ? segment.ropeDim : headDim;
        ropeDim = std::min(ropeDim, headDim);
        ropeDim = (ropeDim / 2) * 2;
        const int half = ropeDim / 2;
        const float attentionScale = segment.ropeAttentionScaling > 0.0f ? segment.ropeAttentionScaling : 1.0f;
        for (size_t local = 0; local < segment.tokenCount; ++local) {
            const int logical = static_cast<int>(segment.logicalStart + local);
            const int slot = logical;
            if (slot < 0 || slot >= maxSlots) {
                return OUT_OF_MEMORY;
            }
            for (int b = 0; b < batch; ++b) {
                for (int h = 0; h < kvHeads; ++h) {
                    const int sourceToken = static_cast<int>(sourceTokenOffset + local);
                    const int keySrcBase = ((sourceToken * batch + b) * kvHeads + h) * headDim;
                    const int keyDstBase = ((slot * batch + b) * kvHeads + h) * headDim;
                    for (int p = 0; p < half; ++p) {
                        const float angle = static_cast<float>(logical) * _ropeInvFreqForSegment(segment, p, ropeDim);
                        const float c = std::cos(angle);
                        const float s = std::sin(angle);
                        const float x0 = _pagedRead(keyData.data(), keySrcBase + p, bytes);
                        const float x1 = _pagedRead(keyData.data(), keySrcBase + p + half, bytes);
                        _pagedWrite(keyCache, keyDstBase + p, (x0 * c - x1 * s) * attentionScale, bytes);
                        _pagedWrite(keyCache, keyDstBase + p + half, (x1 * c + x0 * s) * attentionScale, bytes);
                    }
                    for (int d = ropeDim; d < headDim; ++d) {
                        _pagedWrite(keyCache, keyDstBase + d, _pagedRead(keyData.data(), keySrcBase + d, bytes), bytes);
                    }
                    const int valueSrcTokenCount = static_cast<int>(sourceTokenCount);
                    const int valueSrcBase = ((b * kvHeads + h) * valueSrcTokenCount + sourceToken) * headDim;
                    const int valueDstBase = ((b * kvHeads + h) * maxSlots + slot) * headDim;
                    ::memcpy(valueCache + valueDstBase * bytes, valueData.data() + valueSrcBase * bytes,
                             headDim * bytes);
                }
            }
        }
    }
    meta->markExternalLayerLoaded(layerIndex);
    return NO_ERROR;
}

static ErrorCode _runCacheBlendScoringCPU(PagedKVMeta* meta, int layerIndex, int batch, int kvHeads, int headDim,
                                          int bytes, int maxSlots, int kvLen, const int8_t* valueCache) {
    if (meta == nullptr || !meta->needsCacheBlendScoring(layerIndex)) {
        return NO_ERROR;
    }
    const int picTokenCount = meta->cacheblend_score_pic_token_count;
    const int topK = meta->cacheblend_score_top_k;
    if (picTokenCount < 0 || topK < 0 || topK > picTokenCount ||
        meta->cacheblend_score_pic_start + picTokenCount > kvLen) {
        return INVALID_VALUE;
    }
    if (topK == 0 || picTokenCount == 0) {
        meta->setCacheBlendScoringResult({});
        return NO_ERROR;
    }
    std::vector<float> scores(static_cast<size_t>(picTokenCount), -std::numeric_limits<float>::max());
    size_t scoreOffset = 0;
    const size_t maxInt = static_cast<size_t>(std::numeric_limits<int>::max());
    for (const auto& segment : meta->cacheblend_score_segments) {
        if (segment.tokenCount == 0) {
            continue;
        }
        auto layer = segment.layer(layerIndex);
        if (layer == nullptr) {
            return INVALID_VALUE;
        }
        const int segBatch = segment.batch > 0 ? segment.batch : batch;
        const int segKvHeads = segment.kvHeads > 0 ? segment.kvHeads : kvHeads;
        const int segHeadDim = segment.headDim > 0 ? segment.headDim : headDim;
        const int segBytes = segment.dtypeBytes > 0 ? segment.dtypeBytes : bytes;
        if (segBatch != batch || segKvHeads != kvHeads || segHeadDim != headDim || segBytes != bytes) {
            return INVALID_VALUE;
        }
        const size_t sourceTokenOffset = layer->hasSourceOverride ? layer->sourceTokenOffset
                                                                  : segment.sourceTokenOffset;
        const size_t sourceTokenCount = layer->hasSourceOverride && layer->sourceTokenCount > 0
            ? layer->sourceTokenCount
            : (segment.sourceTokenCount > 0 ? segment.sourceTokenCount : (sourceTokenOffset + segment.tokenCount));
        if (sourceTokenOffset + segment.tokenCount > sourceTokenCount ||
            segment.logicalStart + segment.tokenCount > static_cast<size_t>(kvLen) ||
            scoreOffset + segment.tokenCount > static_cast<size_t>(picTokenCount) ||
            segment.logicalStart > maxInt || segment.tokenCount > maxInt || scoreOffset > maxInt) {
            return INVALID_VALUE;
        }
        std::vector<int8_t> valueData;
        if (!_readBinaryFile(layer->valuePath, valueData)) {
            return INVALID_VALUE;
        }
        const size_t valueElements = static_cast<size_t>(batch) * kvHeads * sourceTokenCount * headDim;
        const size_t cacheElements = static_cast<size_t>(batch) * kvHeads * maxSlots * headDim;
        if (valueElements > maxInt || cacheElements > maxInt) {
            return INVALID_VALUE;
        }
        const size_t expectedValue = valueElements * bytes;
        if (valueData.size() < expectedValue) {
            return INVALID_VALUE;
        }
        const float denom = static_cast<float>(std::max(1, batch * kvHeads * headDim));
        for (size_t local = 0; local < segment.tokenCount; ++local) {
            const size_t logical = segment.logicalStart + local;
            if (logical > maxInt) {
                return INVALID_VALUE;
            }
            const int slot = static_cast<int>(logical);
            if (slot < 0 || slot >= maxSlots) {
                scores[scoreOffset + local] = -std::numeric_limits<float>::max();
                continue;
            }
            const size_t sourceToken = sourceTokenOffset + local;
            float acc = 0.0f;
            for (int b = 0; b < batch; ++b) {
                for (int h = 0; h < kvHeads; ++h) {
                    const int refBase = ((b * kvHeads + h) * maxSlots + slot) * headDim;
                    const size_t cachedBase =
                        (static_cast<size_t>(b * kvHeads + h) * sourceTokenCount + sourceToken) * headDim;
                    for (int d = 0; d < headDim; ++d) {
                        acc += std::fabs(_pagedRead(valueCache, refBase + d, bytes) -
                                         _pagedRead(valueData.data(), static_cast<int>(cachedBase + d), bytes));
                    }
                }
            }
            scores[scoreOffset + local] = acc / denom;
        }
        scoreOffset += segment.tokenCount;
    }
    if (scoreOffset != static_cast<size_t>(picTokenCount)) {
        return INVALID_VALUE;
    }

    std::vector<int> selected;
    selected.reserve(topK);
    std::vector<uint8_t> used(static_cast<size_t>(picTokenCount), 0);
    for (int rank = 0; rank < topK; ++rank) {
        float best = -std::numeric_limits<float>::max();
        int bestIndex = -1;
        for (int i = 0; i < picTokenCount; ++i) {
            if (used[static_cast<size_t>(i)] != 0) {
                continue;
            }
            float value = scores[static_cast<size_t>(i)];
            if (std::isnan(value)) {
                value = -std::numeric_limits<float>::max();
            }
            if (bestIndex < 0 || value > best || (value == best && i < bestIndex)) {
                best = value;
                bestIndex = i;
            }
        }
        if (bestIndex < 0) {
            return INVALID_VALUE;
        }
        used[static_cast<size_t>(bestIndex)] = 1;
        selected.emplace_back(bestIndex);
    }
    meta->setCacheBlendScoringResult(selected);
    return NO_ERROR;
}

static ErrorCode _runFusionRAGOnlineScoringCPU(PagedKVMeta* meta, int layerIndex, int batch, int queryLen,
                                               int numHeads, int kvHeads, int headDim, int bytes, int maxSlots,
                                               int kvLen, const int8_t* queryInput, const int8_t* keyCache) {
    if (meta == nullptr || !meta->needsFusionRAGOnlineScoring(layerIndex)) {
        return NO_ERROR;
    }
    const int picStart = meta->fusionrag_online_score_pic_start;
    const int picTokenCount = meta->fusionrag_online_score_pic_token_count;
    const int topK = meta->fusionrag_online_score_top_k;
    if (picStart < 0 || picTokenCount < 0 || topK < 0 || topK > picTokenCount ||
        picStart + picTokenCount > kvLen || batch <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 ||
        numHeads % kvHeads != 0) {
        return INVALID_VALUE;
    }
    if (topK == 0 || picTokenCount == 0) {
        meta->setFusionRAGOnlineScoringResult({}, 0);
        return NO_ERROR;
    }
    std::vector<int> queryRows;
    std::vector<int> queryLogicals;
    if (!meta->fusionRAGOnlineQueryRows(queryRows, queryLogicals)) {
        meta->setFusionRAGOnlineScoringResult({}, 0);
        return NO_ERROR;
    }
    const int queryCount = static_cast<int>(queryRows.size());
    const int group = numHeads / kvHeads;
    const float scale = meta->attn_scale > 0 ? meta->attn_scale : (1.0f / std::sqrt(static_cast<float>(headDim)));
    const int reduceMode = meta->fusionrag_online_score_reduce_mode;
    std::vector<float> scores(static_cast<size_t>(picTokenCount), 0.0f);
    std::vector<float> logits(static_cast<size_t>(picTokenCount), -std::numeric_limits<float>::max());
    std::vector<float> headMax(static_cast<size_t>(picTokenCount), -std::numeric_limits<float>::max());

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < numHeads; ++h) {
            const int kvHead = h / group;
            if (reduceMode == PagedKVMeta::FusionRAGReduceMaxQueryMeanHead) {
                std::fill(headMax.begin(), headMax.end(), -std::numeric_limits<float>::max());
            }
            for (int qi = 0; qi < queryCount; ++qi) {
                const int qRow = queryRows[static_cast<size_t>(qi)];
                const int qLogical = queryLogicals[static_cast<size_t>(qi)];
                if (qRow < 0 || qRow >= queryLen || qLogical < 0 || qLogical >= kvLen) {
                    continue;
                }
                const int qBase = ((b * queryLen + qRow) * numHeads + h) * headDim;
                float maxLogit = -std::numeric_limits<float>::max();
                for (int local = 0; local < picTokenCount; ++local) {
                    const int logical = picStart + local;
                    if (logical < 0 || logical > qLogical || logical >= maxSlots) {
                        logits[static_cast<size_t>(local)] = -std::numeric_limits<float>::max();
                        continue;
                    }
                    const int kBase = ((logical * batch + b) * kvHeads + kvHead) * headDim;
                    float dot = 0.0f;
                    for (int d = 0; d < headDim; ++d) {
                        dot += _pagedRead(queryInput, qBase + d, bytes) * _pagedRead(keyCache, kBase + d, bytes);
                    }
                    const float logit = dot * scale;
                    logits[static_cast<size_t>(local)] = logit;
                    if (logit > maxLogit) {
                        maxLogit = logit;
                    }
                }
                if (!std::isfinite(maxLogit)) {
                    continue;
                }
                float sumExp = 0.0f;
                for (int local = 0; local < picTokenCount; ++local) {
                    const float logit = logits[static_cast<size_t>(local)];
                    if (!std::isfinite(logit)) {
                        continue;
                    }
                    sumExp += std::exp(logit - maxLogit);
                }
                if (!(sumExp > 0.0f)) {
                    continue;
                }
                const float queryScale = reduceMode == PagedKVMeta::FusionRAGReduceMeanQueryMeanHead
                    ? (1.0f / static_cast<float>(queryCount))
                    : 1.0f;
                for (int local = 0; local < picTokenCount; ++local) {
                    const float logit = logits[static_cast<size_t>(local)];
                    if (!std::isfinite(logit)) {
                        continue;
                    }
                    const float prob = std::exp(logit - maxLogit) / sumExp;
                    if (reduceMode == PagedKVMeta::FusionRAGReduceMaxQueryMeanHead) {
                        headMax[static_cast<size_t>(local)] =
                            std::max(headMax[static_cast<size_t>(local)], prob);
                    } else {
                        scores[static_cast<size_t>(local)] += prob * queryScale;
                    }
                }
            }
            if (reduceMode == PagedKVMeta::FusionRAGReduceMaxQueryMeanHead) {
                for (int local = 0; local < picTokenCount; ++local) {
                    const float value = headMax[static_cast<size_t>(local)];
                    if (std::isfinite(value)) {
                        scores[static_cast<size_t>(local)] += value;
                    }
                }
            }
        }
    }

    const float denom = static_cast<float>(std::max(1, batch * numHeads));
    for (float& score : scores) {
        score /= denom;
    }

    std::vector<int> selected;
    selected.reserve(topK);
    std::vector<uint8_t> used(static_cast<size_t>(picTokenCount), 0);
    for (int rank = 0; rank < topK; ++rank) {
        float best = -std::numeric_limits<float>::max();
        int bestIndex = -1;
        for (int i = 0; i < picTokenCount; ++i) {
            if (used[static_cast<size_t>(i)] != 0) {
                continue;
            }
            const float value = scores[static_cast<size_t>(i)];
            if (bestIndex < 0 || value > best || (value == best && i < bestIndex)) {
                best = value;
                bestIndex = i;
            }
        }
        if (bestIndex < 0) {
            return INVALID_VALUE;
        }
        used[static_cast<size_t>(bestIndex)] = 1;
        selected.emplace_back(bestIndex);
    }
    meta->setFusionRAGOnlineScoringResult(selected, queryCount);
    return NO_ERROR;
}

CPUPagedAttention::CPUPagedAttention(Backend* backend, const Op* op) : Execution(backend) {
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
    mCache.reset(new LayerCache);
}

ErrorCode CPUPagedAttention::ensureCache(int maxSlots, int batch, int kvHeads, int headDim) {
    if (maxSlots <= 0 || batch <= 0 || kvHeads <= 0 || headDim <= 0) {
        return INVALID_VALUE;
    }
    if (mCache && mCache->key && mCache->value && mCache->maxSlots == maxSlots && mCache->batch == batch &&
        mCache->kvHeads == kvHeads && mCache->headDim == headDim && mCache->bytes == mBytes) {
        return NO_ERROR;
    }

    if (!mCache) {
        mCache.reset(new LayerCache);
    }
    mCache->key.reset(Tensor::createDevice<int8_t>({maxSlots * batch * kvHeads * headDim * mBytes}));
    mCache->value.reset(Tensor::createDevice<int8_t>({batch * kvHeads * maxSlots * headDim * mBytes}));
    if (!mCache->key || !mCache->value) {
        return OUT_OF_MEMORY;
    }
    if (!backend()->onAcquireBuffer(mCache->key.get(), Backend::STATIC)) {
        return OUT_OF_MEMORY;
    }
    if (!backend()->onAcquireBuffer(mCache->value.get(), Backend::STATIC)) {
        return OUT_OF_MEMORY;
    }
    ::memset(mCache->key->host<int8_t>(), 0, mCache->key->elementSize());
    ::memset(mCache->value->host<int8_t>(), 0, mCache->value->elementSize());
    mCache->maxSlots = maxSlots;
    mCache->batch = batch;
    mCache->kvHeads = kvHeads;
    mCache->headDim = headDim;
    mCache->bytes = mBytes;
    return NO_ERROR;
}

ErrorCode CPUPagedAttention::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (inputs.size() < 3 || outputs.empty()) {
        return INVALID_VALUE;
    }
    auto core = static_cast<CPUBackend*>(backend())->functions();
    mBytes = core->bytes;
    auto query = inputs[0];
    auto key = inputs[1];
    int batch = query->length(0);
    int kvHeads = key->length(2);
    int headDim = query->length(3);
    int required = key->length(1);
    if (mMeta != nullptr) {
        required = std::max(required, mMeta->request_capacity > 0 ? mMeta->request_capacity : mMeta->max_tokens);
        if (required <= 0) {
            required = static_cast<int>(mMeta->previous) + key->length(1);
        }
    }
    return ensureCache(required, batch, kvHeads, headDim);
}

ErrorCode CPUPagedAttention::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    const Tensor* mask = inputs.size() > 3 ? inputs[3] : nullptr;
    auto output = outputs[0];

    int batch = query->length(0);
    int queryLen = query->length(1);
    int attnLen = output->length(1);
    int numHeads = query->length(2);
    int headDim = query->length(3);
    int newKvLen = key->length(1);
    int kvHeads = key->length(2);
    if (_picDebugEnabled()) {
        std::fprintf(stderr,
                     "CPU PA dims layer=%d opMode=%d q=[%d,%d,%d,%d] k=[%d,%d,%d,%d] "
                     "v=[%d,%d,%d,%d] out=[%d,%d,%d,%d] maskElems=%d metaPrev=%zu metaAdd=%zu metaRemove=%zu\n",
                     mLayerIndex, mPicAttentionMode, query->length(0), query->length(1), query->length(2),
                     query->length(3), key->length(0), key->length(1), key->length(2), key->length(3),
                     value->length(0), value->length(1), value->length(2), value->length(3),
                     output->length(0), output->length(1), output->length(2), output->length(3),
                     mask != nullptr ? static_cast<int>(mask->elementSize()) : 0,
                     mMeta != nullptr ? mMeta->previous : 0, mMeta != nullptr ? mMeta->add : 0,
                     mMeta != nullptr ? mMeta->remove : 0);
        std::fflush(stderr);
    }
    if (batch != key->length(0) || batch != value->length(0) || kvHeads <= 0 || numHeads % kvHeads != 0) {
        return INVALID_VALUE;
    }
    if (mMeta != nullptr && mMeta->request_capacity <= 0 && !mMeta->request_active) {
        mMeta->beginRequest(std::max(std::max(newKvLen, queryLen), mMeta->max_tokens));
    }

    int layerIndex = mLayerIndex >= 0 ? mLayerIndex : (mMeta != nullptr ? mMeta->layer_index : 0);
    if (mMeta != nullptr && mMeta->sparseQueryBlockedBeforeLayer(layerIndex)) {
        MNN_ERROR("CPUPagedAttention layer %d received sparse query before sparse_start_layer=%d. "
                  "Run full prompt until the score layer, crop active hidden states, then resume sparse.\n",
                  layerIndex, mMeta->sparse_query_start_layer_idx);
        return INVALID_VALUE;
    }

    int reverse = _reverseCount(mMeta);
    int baseLogical = 0;
    int kvInsertLen = newKvLen;
    const bool forcePlainSparseQuery = mMeta != nullptr && mMeta->sparse_query_active &&
        mMeta->sparse_query_force_plain_attention;
    const bool picRuntimeActive = mMeta != nullptr &&
        (mMeta->sparse_query_active || mMeta->cacheblend_score_active || mMeta->pic_graph_active_plan_ready);
    const int effectivePicAttentionMode = (picRuntimeActive && !forcePlainSparseQuery) ? mPicAttentionMode : 0;
    bool sparseQuery = mMeta != nullptr && mMeta->sparseQueryActiveForLayer(layerIndex);
    const bool picDecodeRecompute = mMeta != nullptr && mMeta->pic_decode_recompute_active;
    const bool scoreAttention = effectivePicAttentionMode == 1 && !picDecodeRecompute;
    if (scoreAttention) {
        sparseQuery = false;
    }
    if (mMeta != nullptr) {
        size_t kept = mMeta->previous >= mMeta->remove ? (mMeta->previous - mMeta->remove) : 0;
        baseLogical = static_cast<int>(kept) + reverse;
        kvInsertLen = mMeta->add > 0 ? static_cast<int>(std::min<size_t>(mMeta->add, newKvLen)) : newKvLen;
    }
    if (sparseQuery) {
        if (!mMeta->validateSparseQueryRows(attnLen, queryLen, false)) {
            MNN_ERROR("CPUPagedAttention layer %d invalid sparse rows before K/V write, query=%d attn=%d "
                      "active=%d logical_length=%d\n",
                      layerIndex, queryLen, attnLen,
                      static_cast<int>(mMeta->sparse_query_logical_indices.size()), mMeta->logical_length);
            return INVALID_VALUE;
        }
        if (newKvLen < attnLen) {
            MNN_ERROR("CPUPagedAttention layer %d sparse K/V rows %d smaller than active attention rows %d\n",
                      layerIndex, newKvLen, attnLen);
            return INVALID_VALUE;
        }
        baseLogical = 0;
        kvInsertLen = attnLen;
    } else if (effectivePicAttentionMode == 2) {
        MNN_ERROR("CPU PicSparseAttention layer %d requires active sparse rows from PicScoreAttention\n",
                  layerIndex);
        return INVALID_VALUE;
    }
    int kvLen = sparseQuery ? std::max(0, mMeta->logical_length) : (baseLogical + kvInsertLen);
    if (kvLen > mCache->maxSlots) {
        MNN_ERROR("CPUPagedAttention layer %d needs %d slots, cache capacity is %d\n", mLayerIndex, kvLen,
                  mCache->maxSlots);
        return OUT_OF_MEMORY;
    }
    if (mMeta != nullptr && !mMeta->ensureLogicalCapacity(kvLen)) {
        MNN_ERROR("CPUPagedAttention layer %d logical length %d exceeds request capacity %d\n", mLayerIndex, kvLen,
                  mMeta->request_capacity);
        return OUT_OF_MEMORY;
    }

    auto kCache = mCache->key->host<int8_t>();
    auto vCache = mCache->value->host<int8_t>();
    const auto qData = query->host<int8_t>();
    const auto kInput = key->host<int8_t>();
    const auto vInput = value->host<int8_t>();
    auto restore = _restoreExternalSegmentsCPU(mMeta, layerIndex, batch, kvHeads, headDim, mBytes,
                                               mCache->maxSlots, kvLen, kCache, vCache);
    if (restore != NO_ERROR) {
        return restore;
    }
    if (!mIsKVShared) {
        for (int b = 0; b < batch; ++b) {
            for (int l = 0; l < kvInsertLen; ++l) {
                int logical = sparseQuery ? mMeta->sparseLogicalIndex(l) : (baseLogical + l);
                if (logical < 0 || logical >= kvLen) {
                    return INVALID_VALUE;
                }
                int slot = logical;
                for (int h = 0; h < kvHeads; ++h) {
                    int inOffset = ((b * newKvLen + l) * kvHeads + h) * headDim;
                    int kOffset = ((slot * batch + b) * kvHeads + h) * headDim;
                    int vOffset = ((b * kvHeads + h) * mCache->maxSlots + slot) * headDim;
                    ::memcpy(kCache + kOffset * mBytes, kInput + inOffset * mBytes, headDim * mBytes);
                    ::memcpy(vCache + vOffset * mBytes, vInput + inOffset * mBytes, headDim * mBytes);
                }
            }
        }
    }
    auto cacheBlendScore = _runCacheBlendScoringCPU(mMeta, layerIndex, batch, kvHeads, headDim, mBytes,
                                                    mCache->maxSlots, kvLen, vCache);
    if (cacheBlendScore != NO_ERROR) {
        return cacheBlendScore;
    }
    auto fusionragScore = _runFusionRAGOnlineScoringCPU(mMeta, layerIndex, batch, queryLen, numHeads, kvHeads,
                                                        headDim, mBytes, mCache->maxSlots, kvLen, qData, kCache);
    if (fusionragScore != NO_ERROR) {
        return fusionragScore;
    }
    if (outputs.size() > 1) {
        auto emit = _emitActiveIndicesCPU(mMeta, layerIndex, kvLen, outputs[1], scoreAttention);
        if (emit != NO_ERROR) {
            return emit;
        }
        sparseQuery = mMeta != nullptr && mMeta->sparseQueryActiveForLayer(layerIndex);
        attnLen = output->length(1);
        if (sparseQuery) {
            const bool allowFullQueryRows = scoreAttention && queryLen > attnLen;
            if (!mMeta->validateSparseQueryRows(attnLen, queryLen, allowFullQueryRows)) {
                MNN_ERROR("CPUPagedAttention layer %d invalid sparse rows after active-index emit, query=%d "
                          "attn=%d active=%d logical_length=%d allow_full_q=%d\n",
                          layerIndex, queryLen, attnLen,
                          static_cast<int>(mMeta->sparse_query_logical_indices.size()), mMeta->logical_length,
                          allowFullQueryRows ? 1 : 0);
                return INVALID_VALUE;
            }
            baseLogical = 0;
            kvLen = std::max(0, mMeta->logical_length);
        }
    }
    if (mMeta != nullptr && !mMeta->file_name.empty() && mMeta->file_flag == KVMeta::PendingWrite && kvLen > 0) {
        auto prefixDir = static_cast<CPUBackend*>(backend())->getRuntime()->hint().prefixcacheDirPath;
        MNNCreateDir(prefixDir.c_str());
        int layerIndex = mLayerIndex >= 0 ? mLayerIndex : mMeta->layer_index;
        std::string basePath = MNNFilePathConcat(prefixDir, mMeta->file_name) + "_" + std::to_string(layerIndex);
        std::vector<int8_t> keyData(static_cast<size_t>(kvLen) * batch * kvHeads * headDim * mBytes);
        std::vector<int8_t> valueData(static_cast<size_t>(batch) * kvHeads * kvLen * headDim * mBytes);
        for (int l = 0; l < kvLen; ++l) {
            int slot = l;
            for (int b = 0; b < batch; ++b) {
                for (int h = 0; h < kvHeads; ++h) {
                    const int8_t* srcK = kCache + ((slot * batch + b) * kvHeads + h) * headDim * mBytes;
                    int8_t* dstK = keyData.data() + ((l * batch + b) * kvHeads + h) * headDim * mBytes;
                    ::memcpy(dstK, srcK, headDim * mBytes);

                    const int8_t* srcV = vCache + ((b * kvHeads + h) * mCache->maxSlots + slot) * headDim * mBytes;
                    int8_t* dstV = valueData.data() + ((b * kvHeads + h) * kvLen + l) * headDim * mBytes;
                    ::memcpy(dstV, srcV, headDim * mBytes);
                }
            }
        }
        _inverseRopeKeyData(keyData, kvLen, batch, kvHeads, headDim, mBytes, mMeta);
        if (!_writeBinaryFile(basePath + ".k", keyData)) {
            MNN_PRINT("CPUPagedAttention: failed to export key cache: %s\n", (basePath + ".k").c_str());
        }
        if (!_writeBinaryFile(basePath + ".v", valueData)) {
            MNN_PRINT("CPUPagedAttention: failed to export value cache: %s\n", (basePath + ".v").c_str());
        }
        if (!_writeShapeFile(basePath + ".json", batch, kvHeads, headDim, kvLen, mBytes, mMeta)) {
            MNN_PRINT("CPUPagedAttention: failed to export shape metadata: %s\n", (basePath + ".json").c_str());
        }
        if (mLayerIndex < 0) {
            mMeta->layer_index = (mMeta->layer_index + 1) % std::max(1, mMeta->layer_nums);
        }
    }
    ::memset(output->host<int8_t>(), 0, output->elementSize() * mBytes);
    if (attnLen <= 0 || kvLen <= 0) {
        return NO_ERROR;
    }

    mScale = (mMeta && mMeta->attn_scale > 0) ? mMeta->attn_scale : (1.0f / std::sqrt(static_cast<float>(headDim)));
    const int group = numHeads / kvHeads;
    const auto qInput = query->host<int8_t>();
    auto outPtr = output->host<int8_t>();
    int threadNum = static_cast<CPUBackend*>(backend())->threadNumber();
    const bool queryRowsAreFull = scoreAttention && sparseQuery && queryLen > attnLen;
    int totalWork = batch * attnLen * numHeads;
    MNN_CONCURRENCY_BEGIN(tId, threadNum) {
        std::vector<float> scores(kvLen);
        for (int index = (int)tId; index < totalWork; index += threadNum) {
            int h = index % numHeads;
            int tmp = index / numHeads;
            int q = tmp % attnLen;
            int b = tmp / attnLen;
            int kvHead = h / group;
            int qLogical = sparseQuery ? mMeta->sparseLogicalIndex(q) : (baseLogical + q);
            int qRow = queryRowsAreFull ? qLogical : q;
            int validLen = std::min(kvLen, qLogical + 1);
            float maxScore = -std::numeric_limits<float>::infinity();
            for (int k = 0; k < validLen; ++k) {
                int slot = k;
                float score = 0.0f;
                if (mBytes == 4) {
                    const float* qPtr = reinterpret_cast<const float*>(qInput) +
                        ((b * queryLen + qRow) * numHeads + h) * headDim;
                    const float* kPtr = reinterpret_cast<const float*>(kCache) +
                        ((slot * batch + b) * kvHeads + kvHead) * headDim;
                    for (int d = 0; d < headDim; ++d) {
                        score += qPtr[d] * kPtr[d];
                    }
                } else {
                    for (int d = 0; d < headDim; ++d) {
                        int qOffset = ((b * queryLen + qRow) * numHeads + h) * headDim + d;
                        int kOffset = ((slot * batch + b) * kvHeads + kvHead) * headDim + d;
                        score += _pagedRead(qInput, qOffset, mBytes) * _pagedRead(kCache, kOffset, mBytes);
                    }
                }
                score = score * mScale +
                        _readFloatMask(mask, queryRowsAreFull ? qLogical : q, k,
                                       queryRowsAreFull ? queryLen : attnLen, kvLen, mBytes);
                scores[k] = score;
                maxScore = std::max(maxScore, score);
            }
            float sum = 0.0f;
            for (int k = 0; k < validLen; ++k) {
                scores[k] = std::exp(scores[k] - maxScore);
                sum += scores[k];
            }
            float invSum = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int d = 0; d < headDim; ++d) {
                float acc = 0.0f;
                for (int k = 0; k < validLen; ++k) {
                    if (scores[k] <= 0.0f) {
                        continue;
                    }
                    int slot = k;
                    int vOffset = ((b * kvHeads + kvHead) * mCache->maxSlots + slot) * headDim + d;
                    float v = mBytes == 4 ? reinterpret_cast<const float*>(vCache)[vOffset] : _pagedRead(vCache, vOffset, mBytes);
                    acc += scores[k] * invSum * v;
                }
                int outOffset = (b * attnLen * numHeads * headDim) + q * numHeads * headDim + h * headDim + d;
                if (mBytes == 4) {
                    reinterpret_cast<float*>(outPtr)[outOffset] = acc;
                } else {
                    _pagedWrite(outPtr, outOffset, acc, mBytes);
                }
            }
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

bool CPUPagedAttention::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (dst == nullptr) {
        return true;
    }
    auto tmp = new CPUPagedAttention(bn, op);
    tmp->mCache = mCache;
    auto param = op->main_as_AttentionParam();
    tmp->mIsKVShared = param != nullptr && param->kv_shared_layer_index() >= 0;
    *dst = tmp;
    return true;
}

class CPUPagedAttentionCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUPagedAttention(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR_TRANSFORMER(CPUPagedAttentionCreator, OpType_PagedAttention);
REGISTER_CPU_OP_CREATOR_TRANSFORMER(CPUPagedAttentionCreator, OpType_PicScoreAttention);
REGISTER_CPU_OP_CREATOR_TRANSFORMER(CPUPagedAttentionCreator, OpType_PicSparseAttention);

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
