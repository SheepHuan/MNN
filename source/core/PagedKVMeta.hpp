//
//  PagedKVMeta.hpp
//  MNN
//

#ifndef PagedKVMeta_hpp
#define PagedKVMeta_hpp

#include "core/KVMeta.hpp"
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace MNN {

struct PagedKVExternalSegment {
    size_t logicalStart = 0;
    size_t tokenCount = 0;
    size_t sourceTokenOffset = 0;
    size_t sourceTokenCount = 0;
    std::string cacheName;
    int batch = 1;
    int kvHeads = 0;
    int headDim = 0;
    int dtypeBytes = 0;
    int ropeDim = 0;
    float ropeTheta = 10000.0f;
    float ropeScalingFactor = 1.0f;
    float ropeScalingLowFreqFactor = 1.0f;
    float ropeScalingHighFreqFactor = 4.0f;
    int ropeScalingOriginalMaxPositionEmbeddings = 0;
    int maxPositionEmbeddings = 0;
    float ropeAttentionScaling = 1.0f;
    std::string ropeType = "default";
    std::string keyRopeState = "canonical_no_rope";
    std::string ropePairing = "half";

    struct LayerFile {
        int layerIndex = 0;
        std::string keyPath;
        std::string valuePath;
        bool hasSourceOverride = false;
        size_t sourceTokenOffset = 0;
        size_t sourceTokenCount = 0;
    };
    std::vector<LayerFile> layers;

    const LayerFile* layer(int layerIndex) const {
        for (const auto& item : layers) {
            if (item.layerIndex == layerIndex) {
                return &item;
            }
        }
        return nullptr;
    }
};

struct PagedKVMeta : public KVMeta {
    bool paged_attention = true;
    bool request_active = false;
    int max_tokens = 0;
    int request_base = 0;
    int request_capacity = 0;
    int logical_length = 0;
    int slot_table_version = 0;
    int external_hydrate_start_layer_idx = 0;
    size_t external_source_slot_reserve = 0;
    std::vector<int> slot_table_host;
    std::vector<PagedKVExternalSegment> external_segments;
    std::vector<int> external_loaded_layers;
    bool sparse_query_active = false;
    int sparse_query_start_layer_idx = 0;
    bool pic_active_rows = false;
    int pic_active_start_layer_idx = 0;
    int pic_active_count = 0;
    std::vector<int> sparse_query_logical_indices;
    bool cacheblend_score_active = false;
    bool cacheblend_score_ready = false;
    int cacheblend_score_layer_idx = -1;
    int cacheblend_score_pic_start = 0;
    int cacheblend_score_pic_token_count = 0;
    int cacheblend_score_top_k = 0;
    std::vector<PagedKVExternalSegment> cacheblend_score_segments;
    std::vector<int> cacheblend_score_selected_local_indices;
    bool pic_graph_active_plan_ready = false;
    int pic_graph_score_layer_idx = -1;
    int pic_graph_pic_start = 0;
    int pic_graph_pic_token_count = 0;
    std::vector<int> pic_graph_selected_local_indices;

    void beginRequest(int capacity) {
        request_active = true;
        request_base = 0;
        request_capacity = std::max(0, capacity);
        max_tokens = request_capacity;
        logical_length = 0;
        previous = 0;
        remove = 0;
        add = 0;
        n_reserve = 0;
        reserve = nullptr;
        reserveHost.clear();
        external_hydrate_start_layer_idx = 0;
        external_source_slot_reserve = 0;
        external_segments.clear();
        external_loaded_layers.clear();
        sparse_query_active = false;
        sparse_query_start_layer_idx = 0;
        pic_active_rows = false;
        pic_active_start_layer_idx = 0;
        pic_active_count = 0;
        sparse_query_logical_indices.clear();
        finishCacheBlendScoring();
        finishPicGraphActivePlan();
        slot_table_host.resize(request_capacity);
        for (int i = 0; i < request_capacity; ++i) {
            slot_table_host[i] = request_base + i;
        }
        ++slot_table_version;
    }

    void finishRequest() {
        request_active = false;
        external_hydrate_start_layer_idx = 0;
        external_source_slot_reserve = 0;
        external_segments.clear();
        external_loaded_layers.clear();
        sparse_query_active = false;
        sparse_query_start_layer_idx = 0;
        pic_active_rows = false;
        pic_active_start_layer_idx = 0;
        pic_active_count = 0;
        sparse_query_logical_indices.clear();
        finishCacheBlendScoring();
        finishPicGraphActivePlan();
    }

    bool appendExternalSegments(const std::vector<PagedKVExternalSegment>& segments, size_t tokenCount) {
        if (!request_active) {
            return false;
        }
        auto start = static_cast<size_t>(logical_length);
        auto required = start + tokenCount;
        if (!ensureLogicalCapacity(required)) {
            return false;
        }
        size_t cursor = start;
        for (auto segment : segments) {
            segment.logicalStart = cursor;
            cursor += segment.tokenCount;
            external_segments.emplace_back(std::move(segment));
        }
        if (cursor != required) {
            return false;
        }
        logical_length = static_cast<int>(required);
        previous = required;
        external_loaded_layers.clear();
        return true;
    }

    bool reserveExternalSourceSlots(size_t tokenCount) {
        if (!request_active) {
            return false;
        }
        external_source_slot_reserve = std::max(external_source_slot_reserve, tokenCount);
        return true;
    }

    bool bindExternalSegments(const std::vector<PagedKVExternalSegment>& segments, int logicalLength) {
        if (!request_active || logicalLength < 0) {
            return false;
        }
        if (!ensureLogicalCapacity(static_cast<size_t>(logicalLength))) {
            return false;
        }
        external_segments.clear();
        external_segments.reserve(segments.size());
        for (const auto& segment : segments) {
            const size_t end = segment.logicalStart + segment.tokenCount;
            if (end > static_cast<size_t>(logicalLength)) {
                external_segments.clear();
                return false;
            }
            external_segments.emplace_back(segment);
        }
        external_loaded_layers.clear();
        logical_length = logicalLength;
        return true;
    }

    bool shouldHydrateExternalLayer(int layerIndex) const {
        return !external_segments.empty() && layerIndex >= external_hydrate_start_layer_idx;
    }

    bool externalLayerLoaded(int layerIndex) const {
        return std::find(external_loaded_layers.begin(), external_loaded_layers.end(), layerIndex) !=
               external_loaded_layers.end();
    }

    void markExternalLayerLoaded(int layerIndex) {
        if (!externalLayerLoaded(layerIndex)) {
            external_loaded_layers.emplace_back(layerIndex);
        }
    }

    bool beginSparseQuery(const std::vector<int>& logicalIndices, int sparseStartLayerIdx = 0) {
        if (!request_active || logicalIndices.empty()) {
            return false;
        }
        for (int index : logicalIndices) {
            if (index < 0 || index >= logical_length) {
                return false;
            }
        }
        sparse_query_active = true;
        sparse_query_start_layer_idx = std::max(0, sparseStartLayerIdx);
        pic_active_rows = true;
        pic_active_start_layer_idx = sparse_query_start_layer_idx;
        pic_active_count = static_cast<int>(logicalIndices.size());
        sparse_query_logical_indices = logicalIndices;
        return true;
    }

    void finishSparseQuery() {
        sparse_query_active = false;
        sparse_query_start_layer_idx = 0;
        pic_active_rows = false;
        pic_active_start_layer_idx = 0;
        pic_active_count = 0;
        sparse_query_logical_indices.clear();
        add = 0;
        remove = 0;
        n_reserve = 0;
        reserve = nullptr;
    }

    int sparseLogicalIndex(int queryIndex) const {
        if (!sparse_query_active || queryIndex < 0 ||
            queryIndex >= static_cast<int>(sparse_query_logical_indices.size())) {
            return -1;
        }
        return sparse_query_logical_indices[queryIndex];
    }

    bool sparseQueryActiveForLayer(int layerIndex) const {
        return sparse_query_active && layerIndex >= sparse_query_start_layer_idx;
    }

    bool sparseQueryBlockedBeforeLayer(int layerIndex) const {
        return sparse_query_active && layerIndex < sparse_query_start_layer_idx;
    }

    bool picActiveRowsForLayer(int layerIndex) const {
        return pic_active_rows && layerIndex >= pic_active_start_layer_idx && pic_active_count > 0;
    }

    int picActiveLogicalIndex(int activeIndex) const {
        if (!pic_active_rows || activeIndex < 0 ||
            activeIndex >= static_cast<int>(sparse_query_logical_indices.size())) {
            return -1;
        }
        return sparse_query_logical_indices[activeIndex];
    }

    std::vector<int> buildCacheBlendActiveLogicalIndices(int kvLen) const {
        std::vector<int> active;
        if (kvLen <= 0 || cacheblend_score_pic_start < 0 || cacheblend_score_pic_token_count < 0) {
            return active;
        }
        const int picStart = cacheblend_score_pic_start;
        const int picEnd = std::min(kvLen, picStart + cacheblend_score_pic_token_count);
        active.reserve(static_cast<size_t>(kvLen - cacheblend_score_pic_token_count) +
                       cacheblend_score_selected_local_indices.size());
        for (int logical = 0; logical < std::min(picStart, kvLen); ++logical) {
            active.emplace_back(logical);
        }
        std::vector<int> selected = cacheblend_score_selected_local_indices;
        std::sort(selected.begin(), selected.end());
        selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
        for (int local : selected) {
            const int logical = picStart + local;
            if (logical >= picStart && logical < picEnd) {
                active.emplace_back(logical);
            }
        }
        for (int logical = picEnd; logical < kvLen; ++logical) {
            active.emplace_back(logical);
        }
        return active;
    }

    std::vector<int> buildPicGraphActiveLogicalIndices(int kvLen) const {
        std::vector<int> active;
        if (!pic_graph_active_plan_ready || kvLen <= 0 || pic_graph_pic_start < 0 ||
            pic_graph_pic_token_count < 0) {
            return active;
        }
        const int picStart = pic_graph_pic_start;
        const int picEnd = std::min(kvLen, picStart + pic_graph_pic_token_count);
        active.reserve(static_cast<size_t>(kvLen - pic_graph_pic_token_count) +
                       pic_graph_selected_local_indices.size());
        for (int logical = 0; logical < std::min(picStart, kvLen); ++logical) {
            active.emplace_back(logical);
        }
        std::vector<int> selected = pic_graph_selected_local_indices;
        std::sort(selected.begin(), selected.end());
        selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
        for (int local : selected) {
            const int logical = picStart + local;
            if (logical >= picStart && logical < picEnd) {
                active.emplace_back(logical);
            }
        }
        for (int logical = picEnd; logical < kvLen; ++logical) {
            active.emplace_back(logical);
        }
        return active;
    }

    std::vector<int> buildBudgetActiveLogicalIndices(int kvLen, int budget) const {
        std::vector<int> active;
        if (kvLen <= 0 || budget <= 0) {
            return active;
        }
        if (pic_graph_active_plan_ready && pic_graph_score_layer_idx >= 0) {
            active = buildPicGraphActiveLogicalIndices(kvLen);
            if (!active.empty()) {
                return active;
            }
        }
        if (cacheblend_score_ready && cacheblend_score_layer_idx >= 0) {
            active = buildCacheBlendActiveLogicalIndices(kvLen);
            if (!active.empty()) {
                return active;
            }
        }
        const int count = std::min(kvLen, budget);
        active.reserve(count);
        for (int logical = 0; logical < count; ++logical) {
            active.emplace_back(logical);
        }
        return active;
    }

    bool activatePicRows(const std::vector<int>& logicalIndices, int sparseStartLayerIdx, int kvLen) {
        logical_length = std::max(logical_length, kvLen);
        return beginSparseQuery(logicalIndices, sparseStartLayerIdx);
    }

    bool beginPicGraphActivePlan(int picStart, int picTokenCount, int scoreLayerIdx,
                                 const std::vector<int>& selectedLocalIndices) {
        if (!request_active || picStart < 0 || picTokenCount < 0 || scoreLayerIdx < 0) {
            return false;
        }
        std::vector<int> selected;
        selected.reserve(selectedLocalIndices.size());
        for (int local : selectedLocalIndices) {
            if (local >= 0 && local < picTokenCount) {
                selected.emplace_back(local);
            }
        }
        std::sort(selected.begin(), selected.end());
        selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
        pic_graph_active_plan_ready = true;
        pic_graph_score_layer_idx = scoreLayerIdx;
        pic_graph_pic_start = picStart;
        pic_graph_pic_token_count = picTokenCount;
        pic_graph_selected_local_indices = std::move(selected);
        return true;
    }

    void finishPicGraphActivePlan() {
        pic_graph_active_plan_ready = false;
        pic_graph_score_layer_idx = -1;
        pic_graph_pic_start = 0;
        pic_graph_pic_token_count = 0;
        pic_graph_selected_local_indices.clear();
    }

    int graphActiveBudget(int seqLen) const {
        if (pic_graph_active_plan_ready && pic_graph_pic_token_count >= 0) {
            const int compact = seqLen - pic_graph_pic_token_count +
                                static_cast<int>(pic_graph_selected_local_indices.size());
            return std::max(0, std::min(seqLen, compact));
        }
        if (cacheblend_score_active && cacheblend_score_pic_token_count >= 0) {
            const int compact = seqLen - cacheblend_score_pic_token_count + cacheblend_score_top_k;
            return std::max(0, std::min(seqLen, compact));
        }
        return seqLen;
    }

    bool beginCacheBlendScoring(int picStart, int picTokenCount, int scoreLayerIdx, int topK,
                                const std::vector<PagedKVExternalSegment>& segments) {
        if (!request_active || picStart < 0 || picTokenCount < 0 || scoreLayerIdx < 0 || topK < 0) {
            return false;
        }
        if (picTokenCount == 0 || topK == 0) {
            cacheblend_score_active = true;
            cacheblend_score_ready = true;
            cacheblend_score_layer_idx = scoreLayerIdx;
            cacheblend_score_pic_start = picStart;
            cacheblend_score_pic_token_count = picTokenCount;
            cacheblend_score_top_k = 0;
            cacheblend_score_segments.clear();
            cacheblend_score_selected_local_indices.clear();
            return true;
        }
        size_t total = 0;
        cacheblend_score_segments.clear();
        cacheblend_score_segments.reserve(segments.size());
        for (auto segment : segments) {
            segment.logicalStart = static_cast<size_t>(picStart) + total;
            total += segment.tokenCount;
            cacheblend_score_segments.emplace_back(std::move(segment));
        }
        if (total != static_cast<size_t>(picTokenCount)) {
            finishCacheBlendScoring();
            return false;
        }
        cacheblend_score_active = true;
        cacheblend_score_ready = false;
        cacheblend_score_layer_idx = scoreLayerIdx;
        cacheblend_score_pic_start = picStart;
        cacheblend_score_pic_token_count = picTokenCount;
        cacheblend_score_top_k = std::min(topK, picTokenCount);
        cacheblend_score_selected_local_indices.clear();
        return true;
    }

    bool needsCacheBlendScoring(int layerIndex) const {
        return cacheblend_score_active && !cacheblend_score_ready && layerIndex == cacheblend_score_layer_idx;
    }

    void setCacheBlendScoringResult(const std::vector<int>& selectedLocalIndices) {
        cacheblend_score_selected_local_indices = selectedLocalIndices;
        cacheblend_score_ready = true;
    }

    void finishCacheBlendScoring() {
        cacheblend_score_active = false;
        cacheblend_score_ready = false;
        cacheblend_score_layer_idx = -1;
        cacheblend_score_pic_start = 0;
        cacheblend_score_pic_token_count = 0;
        cacheblend_score_top_k = 0;
        cacheblend_score_segments.clear();
        cacheblend_score_selected_local_indices.clear();
    }

    bool ensureLogicalCapacity(size_t required) {
        if (required <= slot_table_host.size()) {
            return true;
        }
        if (request_capacity <= 0 || required > static_cast<size_t>(request_capacity)) {
            return false;
        }
        auto old = slot_table_host.size();
        slot_table_host.resize(required);
        for (size_t i = old; i < required; ++i) {
            slot_table_host[i] = request_base + static_cast<int>(i);
        }
        ++slot_table_version;
        return true;
    }

    int physicalSlot(size_t logicalIndex) const {
        if (logicalIndex >= slot_table_host.size()) {
            return -1;
        }
        return slot_table_host[logicalIndex];
    }

    void syncPaged() {
        if (sparse_query_active) {
            previous = static_cast<size_t>(std::max(0, logical_length));
            n_reserve = 0;
            reserve = nullptr;
            remove = 0;
            add = 0;
            return;
        }
        int revertNumber = 0;
        for (int i = 0; i < n_reserve; ++i) {
            revertNumber += reserve[2 * i + 1];
        }
        auto nextLength = static_cast<int>(previous - remove + add + revertNumber);
        logical_length = std::max(0, nextLength);
        previous = static_cast<size_t>(logical_length);
        n_reserve = 0;
        reserve = nullptr;
        remove = 0;
        add = 0;
    }
};

} // namespace MNN

#endif // PagedKVMeta_hpp
