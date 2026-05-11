//
//  PagedKVCachePlan.hpp
//  MNN
//
//  Host-side scaffold for moving runtime KV cache from one contiguous buffer to
//  fixed-size KV blocks plus a logical block table.
//

#ifndef PagedKVCachePlan_hpp
#define PagedKVCachePlan_hpp

#include "KVMeta.hpp"

#include <algorithm>
#include <vector>

namespace MNN {

struct RuntimeKVBlockRef {
    enum Source {
        Document = 0,
        Prompt = 1,
        Decode = 2,
    };

    int source = Document;
    int segmentIndex = -1;
    int logicalBlockIndex = 0;
    int physicalBlockIndex = -1;
    int physicalTokenStart = 0;
    int logicalTokenStart = 0;
    int sourceTokenStart = 0;
    int tokenCount = 0;
};

class RuntimeKVBlockTablePlan {
public:
    enum {
        kDefaultPageSize = 64,
    };

    void reset(int pageSize = kDefaultPageSize) {
        mPageSize = pageSize > 0 ? pageSize : kDefaultPageSize;
        mLogicalLength = 0;
        mPhysicalLength = 0;
        mRefs.clear();
    }

    bool appendDocumentSegment(int segmentIndex, int tokenCount) {
        return appendRange(RuntimeKVBlockRef::Document, segmentIndex, tokenCount);
    }

    bool appendPrompt(int tokenCount) {
        return appendRange(RuntimeKVBlockRef::Prompt, -1, tokenCount);
    }

    bool appendDecode(int tokenCount) {
        return appendRange(RuntimeKVBlockRef::Decode, -1, tokenCount);
    }

    int pageSize() const {
        return mPageSize;
    }

    int logicalLength() const {
        return mLogicalLength;
    }

    int logicalBlockCount() const {
        return (mLogicalLength + mPageSize - 1) / mPageSize;
    }

    int physicalLength() const {
        return mPhysicalLength;
    }

    int physicalBlockCount() const {
        return (mPhysicalLength + mPageSize - 1) / mPageSize;
    }

    const std::vector<RuntimeKVBlockRef>& refs() const {
        return mRefs;
    }

    bool empty() const {
        return mRefs.empty();
    }

private:
    bool appendRange(int source, int segmentIndex, int tokenCount) {
        if (tokenCount < 0) {
            return false;
        }
        alignPhysicalToBlock();
        int remaining = tokenCount;
        int sourceTokenStart = 0;
        while (remaining > 0) {
            int inBlockOffset = mPhysicalLength % mPageSize;
            int writable = std::min(mPageSize - inBlockOffset, remaining);
            RuntimeKVBlockRef ref;
            ref.source = source;
            ref.segmentIndex = segmentIndex;
            ref.logicalBlockIndex = mLogicalLength / mPageSize;
            ref.physicalBlockIndex = mPhysicalLength / mPageSize;
            ref.physicalTokenStart = mPhysicalLength;
            ref.logicalTokenStart = mLogicalLength;
            ref.sourceTokenStart = sourceTokenStart;
            ref.tokenCount = writable;
            mRefs.emplace_back(ref);
            mLogicalLength += writable;
            mPhysicalLength += writable;
            sourceTokenStart += writable;
            remaining -= writable;
        }
        return true;
    }

    void alignPhysicalToBlock() {
        if (mPhysicalLength > 0) {
            mPhysicalLength = ((mPhysicalLength + mPageSize - 1) / mPageSize) * mPageSize;
        }
    }

    int mPageSize = kDefaultPageSize;
    int mLogicalLength = 0;
    int mPhysicalLength = 0;
    std::vector<RuntimeKVBlockRef> mRefs;
};

inline bool buildPrefixRuntimeKVBlockTablePlan(RuntimeKVBlockTablePlan& plan, const KVMeta* meta,
                                               int promptTokenCount,
                                               int pageSize = RuntimeKVBlockTablePlan::kDefaultPageSize) {
    if (meta == nullptr || promptTokenCount < 0) {
        return false;
    }
    plan.reset(pageSize);
    for (int i = 0; i < static_cast<int>(meta->prefix_segments.size()); ++i) {
        const auto& segment = meta->prefix_segments[i];
        if (segment.token_count < 0) {
            return false;
        }
        if (!plan.appendDocumentSegment(i, segment.token_count)) {
            return false;
        }
    }
    return plan.appendPrompt(promptTokenCount);
}

} // namespace MNN

#endif // PagedKVCachePlan_hpp
