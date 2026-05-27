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
    std::vector<int> slot_table_host;
    std::vector<PagedKVExternalSegment> external_segments;
    std::vector<int> external_loaded_layers;
    bool sparse_query_active = false;
    std::vector<int> sparse_query_logical_indices;

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
        external_segments.clear();
        external_loaded_layers.clear();
        sparse_query_active = false;
        sparse_query_logical_indices.clear();
        slot_table_host.resize(request_capacity);
        for (int i = 0; i < request_capacity; ++i) {
            slot_table_host[i] = request_base + i;
        }
        ++slot_table_version;
    }

    void finishRequest() {
        request_active = false;
        external_segments.clear();
        external_loaded_layers.clear();
        sparse_query_active = false;
        sparse_query_logical_indices.clear();
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

    bool externalLayerLoaded(int layerIndex) const {
        return std::find(external_loaded_layers.begin(), external_loaded_layers.end(), layerIndex) !=
               external_loaded_layers.end();
    }

    void markExternalLayerLoaded(int layerIndex) {
        if (!externalLayerLoaded(layerIndex)) {
            external_loaded_layers.emplace_back(layerIndex);
        }
    }

    bool beginSparseQuery(const std::vector<int>& logicalIndices) {
        if (!request_active || logicalIndices.empty()) {
            return false;
        }
        for (int index : logicalIndices) {
            if (index < 0 || index >= logical_length) {
                return false;
            }
        }
        sparse_query_active = true;
        sparse_query_logical_indices = logicalIndices;
        return true;
    }

    void finishSparseQuery() {
        sparse_query_active = false;
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
