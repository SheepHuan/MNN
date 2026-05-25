//
//  PagedKVMeta.hpp
//  MNN
//

#ifndef PagedKVMeta_hpp
#define PagedKVMeta_hpp

#include "core/KVMeta.hpp"
#include <algorithm>
#include <string>
#include <vector>

namespace MNN {

struct PagedKVExternalSegment {
    size_t logicalStart = 0;
    size_t tokenCount = 0;
    std::string cacheName;
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
        slot_table_host.resize(request_capacity);
        for (int i = 0; i < request_capacity; ++i) {
            slot_table_host[i] = request_base + i;
        }
        ++slot_table_version;
    }

    void finishRequest() {
        request_active = false;
        external_segments.clear();
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
