//
//  llm_pagedkv.cpp
//

#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>
#include <vector>

#include "core/PagedKVMeta.hpp"
#include "llm/llm.hpp"
#include "llm_internal.hpp"
#include "llmconfig.hpp"

namespace MNN {
namespace Transformer {
namespace {

bool buildBoundExternalSegments(const std::vector<MNN::PagedKVExternalSegment>& segments,
                                int picStart,
                                int picTokenCount,
                                const char* errorPrefix,
                                std::vector<MNN::PagedKVExternalSegment>& boundSegments) {
    boundSegments.clear();
    boundSegments.reserve(segments.size());
    size_t cursor = 0;
    for (auto segment : segments) {
        segment.logicalStart = static_cast<size_t>(picStart) + cursor;
        cursor += segment.tokenCount;
        boundSegments.emplace_back(std::move(segment));
    }
    if (cursor != static_cast<size_t>(picTokenCount)) {
        MNN_ERROR("%s segment token mismatch, segments=%d pic_tokens=%d\n",
                  errorPrefix,
                  static_cast<int>(cursor),
                  picTokenCount);
        return false;
    }
    return true;
}

bool collectNonPicActiveRows(const std::vector<int>& fullPromptTokenIds,
                             int picStart,
                             int picTokenCount,
                             const char* errorPrefix,
                             std::vector<int>& activeLogicalIndices,
                             std::vector<int>& activeTokenIds) {
    const int fullPromptLen = static_cast<int>(fullPromptTokenIds.size());
    const int picEnd = picStart + picTokenCount;
    activeLogicalIndices.clear();
    activeTokenIds.clear();
    activeLogicalIndices.reserve(static_cast<size_t>(fullPromptLen - picTokenCount));
    activeTokenIds.reserve(static_cast<size_t>(fullPromptLen - picTokenCount));
    for (int logical = 0; logical < picStart; ++logical) {
        activeLogicalIndices.emplace_back(logical);
        activeTokenIds.emplace_back(fullPromptTokenIds[static_cast<size_t>(logical)]);
    }
    for (int logical = picEnd; logical < fullPromptLen; ++logical) {
        activeLogicalIndices.emplace_back(logical);
        activeTokenIds.emplace_back(fullPromptTokenIds[static_cast<size_t>(logical)]);
    }
    if (activeLogicalIndices.empty()) {
        MNN_ERROR("%s has no non-PIC query rows\n", errorPrefix);
        return false;
    }
    return true;
}

void resetExternalSparsePrefillState(PagedKVMeta* paged, int fullPromptLen) {
    paged->finishSparseQuery();
    paged->finishCacheBlendScoring();
    paged->finishFusionRAGOnlineScoring();
    paged->finishPicGraphActivePlan();
    paged->external_hydrate_start_layer_idx = 0;
    paged->previous = 0;
    paged->logical_length = fullPromptLen;
    paged->remove = 0;
    paged->add = 0;
    paged->n_reserve = 0;
    paged->reserve = nullptr;
}

bool bindSparsePrefillInputs(PagedKVMeta* paged,
                             const std::vector<MNN::PagedKVExternalSegment>& boundSegments,
                             int fullPromptLen,
                             int picTokenCount,
                             const std::vector<int>& activeLogicalIndices,
                             const char* errorPrefix) {
    if (picTokenCount > 0 && !paged->reserveExternalSourceSlots(static_cast<size_t>(picTokenCount))) {
        MNN_ERROR("%s failed to reserve persistent PIC cache source slots\n", errorPrefix);
        return false;
    }
    if (!paged->bindExternalSegments(boundSegments, fullPromptLen)) {
        MNN_ERROR("%s failed to bind persistent PIC cache source segments\n", errorPrefix);
        return false;
    }
    if (!paged->beginSparseQuery(activeLogicalIndices, 0)) {
        MNN_ERROR("%s failed to bind active non-PIC rows\n", errorPrefix);
        return false;
    }
    paged->sparse_query_force_plain_attention = true;
    return true;
}

} // namespace

bool Llm::prefillWithForcedForward(const std::vector<int>& input_ids) {
    ScopedForcePrefillForward forcePrefillForward(mForcePrefillForward);
    return prefill(input_ids);
}

bool Llm::beginExternalPagedKVRequest() {
    return beginPagedRequestIfNeeded() ||
        (mConfig->paged_attention() && static_cast<PagedKVMeta*>(mMeta.get())->request_active);
}

bool Llm::reserveExternalPagedKVSourceSlots(size_t token_count) {
    if (token_count == 0) {
        return true;
    }
    if (!mConfig->paged_attention()) {
        MNN_ERROR("Persistent PIC cache source slot reservation requires paged_attention=true\n");
        return false;
    }
    auto paged = static_cast<PagedKVMeta*>(mMeta.get());
    if (paged == nullptr) {
        return false;
    }
    if (!paged->request_active) {
        paged->beginRequest(mConfig->paged_kv_max_tokens());
    }
    return paged->reserveExternalSourceSlots(token_count);
}

bool Llm::prefillCacheBlendGraphExternalPagedKV(const std::vector<int>& full_prompt_token_ids,
                                                const std::vector<MNN::PagedKVExternalSegment>& segments,
                                                int pic_start, int pic_token_count, int score_layer_idx,
                                                double recompute_ratio,
                                                std::vector<int>& selected_local_indices) {
    const int64_t totalStartUs = picRequestProfileEnabled() ? picRequestMonotonicUs() : 0;
    selected_local_indices.clear();
    if (!mConfig->paged_attention() || full_prompt_token_ids.empty()) {
        return false;
    }
    if (!mConfig->has_pic_recompute_budget()) {
        MNN_ERROR("Graph-level cacheblend prefill requires pic_recompute_budget input\n");
        return false;
    }
    if (pic_start < 0 || pic_token_count < 0 || score_layer_idx < 0 ||
        pic_start + pic_token_count > static_cast<int>(full_prompt_token_ids.size())) {
        return false;
    }
    int topK = 0;
    if (pic_token_count > 0 && recompute_ratio > 0.0) {
        topK = std::min(pic_token_count,
                        std::max(1, static_cast<int>(std::ceil(pic_token_count * recompute_ratio))));
    }
    auto paged = static_cast<PagedKVMeta*>(mMeta.get());
    if (paged == nullptr) {
        return false;
    }
    beginPagedRequestIfNeeded();
    paged->finishCacheBlendScoring();
    paged->finishFusionRAGOnlineScoring();
    paged->finishPicGraphActivePlan();
    paged->finishSparseQuery();

    std::vector<MNN::PagedKVExternalSegment> boundSegments;
    if (!buildBoundExternalSegments(segments, pic_start, pic_token_count, "Graph-level cacheblend", boundSegments)) {
        return false;
    }
    int64_t stageStartUs = picRequestProfileEnabled() ? picRequestMonotonicUs() : 0;
    if (!paged->bindExternalSegments(boundSegments, static_cast<int>(full_prompt_token_ids.size()))) {
        MNN_ERROR("Graph-level cacheblend failed to bind persistent PIC cache source segments\n");
        return false;
    }
    if (picRequestProfileEnabled()) {
        std::ostringstream os;
        os << "pic_start=" << pic_start
           << " pic_tokens=" << pic_token_count
           << " full_prompt_tokens=" << full_prompt_token_ids.size()
           << " segments=" << boundSegments.size();
        picRequestProfileLog("graph_cacheblend_bind_external_segments", picRequestMonotonicUs() - stageStartUs,
                             os.str());
    }
    paged->external_hydrate_start_layer_idx = score_layer_idx + 1;
    stageStartUs = picRequestProfileEnabled() ? picRequestMonotonicUs() : 0;
    if (!paged->beginCacheBlendScoring(pic_start, pic_token_count, score_layer_idx, topK, segments)) {
        MNN_ERROR("Graph-level cacheblend failed to start score-layer scoring\n");
        return false;
    }
    if (picRequestProfileEnabled()) {
        std::ostringstream os;
        os << "score_layer=" << score_layer_idx
           << " topk=" << topK
           << " pic_tokens=" << pic_token_count;
        picRequestProfileLog("graph_cacheblend_begin_scoring", picRequestMonotonicUs() - stageStartUs, os.str());
    }
    stageStartUs = picRequestProfileEnabled() ? picRequestMonotonicUs() : 0;
    const bool ok = prefill(full_prompt_token_ids);
    if (picRequestProfileEnabled()) {
        std::ostringstream os;
        os << "score_layer=" << score_layer_idx
           << " full_prompt_tokens=" << full_prompt_token_ids.size();
        picRequestProfileLog("graph_cacheblend_prefill_full_prompt", picRequestMonotonicUs() - stageStartUs, os.str());
    }
    const bool ready = paged->cacheblend_score_ready;
    if (ready) {
        selected_local_indices = paged->cacheblend_score_selected_local_indices;
    }
    paged->finishCacheBlendScoring();
    paged->finishSparseQuery();
    if (!ok || !ready || static_cast<int>(selected_local_indices.size()) != topK) {
        return false;
    }
    if (picRequestProfileEnabled()) {
        std::ostringstream os;
        os << "score_layer=" << score_layer_idx
           << " selected_count=" << selected_local_indices.size()
           << " topk=" << topK;
        picRequestProfileLog("prefill_cacheblend_graph_external_pagedkv_total",
                             picRequestMonotonicUs() - totalStartUs, os.str());
    }
    return true;
}

bool Llm::prefillFullReuseExternalPagedKV(const std::vector<int>& full_prompt_token_ids,
                                          const std::vector<MNN::PagedKVExternalSegment>& segments,
                                          int pic_start, int pic_token_count) {
    const int64_t totalStartUs = picRequestProfileEnabled() ? picRequestMonotonicUs() : 0;
    if (!mConfig->paged_attention() || full_prompt_token_ids.empty()) {
        return false;
    }
    const int fullPromptLen = static_cast<int>(full_prompt_token_ids.size());
    if (pic_start < 0 || pic_token_count < 0 || pic_start + pic_token_count > fullPromptLen) {
        return false;
    }
    const int picEnd = pic_start + pic_token_count;
    if (picEnd >= fullPromptLen) {
        MNN_ERROR("Full-reuse hydrate+suffix prefill requires at least one suffix token\n");
        return false;
    }
    std::vector<int> activeLogicalIndices;
    std::vector<int> activeTokenIds;
    if (!collectNonPicActiveRows(full_prompt_token_ids, pic_start, pic_token_count,
                                 "Full-reuse hydrate+suffix prefill",
                                 activeLogicalIndices,
                                 activeTokenIds)) {
        return false;
    }
    auto paged = static_cast<PagedKVMeta*>(mMeta.get());
    if (paged == nullptr) {
        return false;
    }
    beginPagedRequestIfNeeded(fullPromptLen);
    resetExternalSparsePrefillState(paged, fullPromptLen);

    std::vector<MNN::PagedKVExternalSegment> boundSegments;
    if (!buildBoundExternalSegments(segments, pic_start, pic_token_count,
                                    "Full-reuse hydrate+suffix", boundSegments)) {
        return false;
    }

    int64_t stageStartUs = picRequestProfileEnabled() ? picRequestMonotonicUs() : 0;
    if (!bindSparsePrefillInputs(paged, boundSegments, fullPromptLen, pic_token_count,
                                 activeLogicalIndices, "Full-reuse hydrate+suffix")) {
        return false;
    }
    if (picRequestProfileEnabled()) {
        std::ostringstream os;
        os << "pic_start=" << pic_start
           << " pic_tokens=" << pic_token_count
           << " active_tokens=" << activeTokenIds.size()
           << " logical_length=" << paged->logical_length
           << " previous=" << static_cast<int>(paged->previous)
           << " segments=" << boundSegments.size();
        picRequestProfileLog("full_reuse_bind_active_external_segments",
                             picRequestMonotonicUs() - stageStartUs, os.str());
    }

    stageStartUs = picRequestProfileEnabled() ? picRequestMonotonicUs() : 0;
    const bool ok = prefillWithForcedForward(activeTokenIds);
    paged->sparse_query_force_plain_attention = false;
    paged->finishSparseQuery();
    if (!ok || mContext == nullptr) {
        return false;
    }
    mContext->all_seq_len = fullPromptLen;
    mContext->prompt_len = fullPromptLen;
    mContext->history_tokens = full_prompt_token_ids;
    paged->logical_length = std::max(paged->logical_length, fullPromptLen);
    paged->previous = static_cast<size_t>(paged->logical_length);
    if (picRequestProfileEnabled()) {
        std::ostringstream os;
        os << "active_tokens=" << activeTokenIds.size()
           << " prelude_tokens=" << pic_start
           << " suffix_tokens=" << (fullPromptLen - picEnd)
           << " full_prompt_tokens=" << fullPromptLen
           << " logical_length=" << paged->logical_length;
        picRequestProfileLog("full_reuse_prefill_active_non_pic", picRequestMonotonicUs() - stageStartUs, os.str());
        picRequestProfileLog("prefill_full_reuse_external_pagedkv_total",
                             picRequestMonotonicUs() - totalStartUs, os.str());
    }
    return true;
}

bool Llm::prefillFusionRAGOnlineSelectExternalPagedKV(
    const std::vector<int>& full_prompt_token_ids,
    const std::vector<MNN::PagedKVExternalSegment>& segments,
    int pic_start, int pic_token_count, int capture_layer_idx, double recompute_ratio, int query_tail_tokens,
    int reduce_mode, std::vector<int>& selected_local_indices, int* selected_query_count) {
    const int64_t totalStartUs = picRequestProfileEnabled() ? picRequestMonotonicUs() : 0;
    selected_local_indices.clear();
    if (selected_query_count != nullptr) {
        *selected_query_count = 0;
    }
    if (!mConfig->paged_attention() || full_prompt_token_ids.empty()) {
        return false;
    }
    const int fullPromptLen = static_cast<int>(full_prompt_token_ids.size());
    if (pic_start < 0 || pic_token_count < 0 || capture_layer_idx < 0 || pic_start + pic_token_count > fullPromptLen) {
        return false;
    }
    const int picEnd = pic_start + pic_token_count;
    if (picEnd >= fullPromptLen) {
        MNN_ERROR("FusionRAG-online qcompute requires at least one suffix token after PIC tokens\n");
        return false;
    }
    int topK = 0;
    if (pic_token_count > 0 && recompute_ratio > 0.0) {
        topK = std::min(pic_token_count,
                        std::max(1, static_cast<int>(std::ceil(pic_token_count * recompute_ratio))));
    }
    std::vector<int> activeLogicalIndices;
    std::vector<int> activeTokenIds;
    if (!collectNonPicActiveRows(full_prompt_token_ids, pic_start, pic_token_count,
                                 "FusionRAG-online qcompute",
                                 activeLogicalIndices,
                                 activeTokenIds)) {
        return false;
    }
    auto paged = static_cast<PagedKVMeta*>(mMeta.get());
    if (paged == nullptr) {
        return false;
    }
    beginPagedRequestIfNeeded(fullPromptLen);
    resetExternalSparsePrefillState(paged, fullPromptLen);

    std::vector<MNN::PagedKVExternalSegment> boundSegments;
    if (!buildBoundExternalSegments(segments, pic_start, pic_token_count,
                                    "FusionRAG-online qcompute", boundSegments)) {
        return false;
    }
    int64_t stageStartUs = picRequestProfileEnabled() ? picRequestMonotonicUs() : 0;
    if (!bindSparsePrefillInputs(paged, boundSegments, fullPromptLen, pic_token_count,
                                 activeLogicalIndices, "FusionRAG-online qcompute")) {
        return false;
    }
    if (!paged->beginFusionRAGOnlineScoring(pic_start, pic_token_count, capture_layer_idx, topK, query_tail_tokens,
                                            reduce_mode)) {
        paged->sparse_query_force_plain_attention = false;
        paged->finishSparseQuery();
        return false;
    }
    if (picRequestProfileEnabled()) {
        std::ostringstream os;
        os << "capture_layer=" << capture_layer_idx
           << " topk=" << topK
           << " pic_tokens=" << pic_token_count
           << " active_tokens=" << activeTokenIds.size()
           << " query_tail_tokens=" << std::max(0, query_tail_tokens)
           << " reduce_mode=" << reduce_mode;
        picRequestProfileLog("fusionrag_online_qcompute_begin", picRequestMonotonicUs() - stageStartUs, os.str());
    }

    stageStartUs = picRequestProfileEnabled() ? picRequestMonotonicUs() : 0;
    const bool ok = prefillWithForcedForward(activeTokenIds);
    paged->sparse_query_force_plain_attention = false;
    const bool ready = paged->fusionrag_online_score_ready;
    if (ready) {
        selected_local_indices = paged->fusionrag_online_score_selected_local_indices;
        if (selected_query_count != nullptr) {
            *selected_query_count = paged->fusionrag_online_selected_query_count;
        }
    }
    paged->finishFusionRAGOnlineScoring();
    paged->finishSparseQuery();
    if (picRequestProfileEnabled()) {
        std::ostringstream os;
        os << "capture_layer=" << capture_layer_idx
           << " selected_count=" << selected_local_indices.size()
           << " selected_query_count=" << (selected_query_count != nullptr ? *selected_query_count : 0)
           << " topk=" << topK;
        picRequestProfileLog("fusionrag_online_qcompute_prefill", picRequestMonotonicUs() - stageStartUs, os.str());
    }
    if (!ok || !ready || static_cast<int>(selected_local_indices.size()) != topK) {
        return false;
    }
    if (picRequestProfileEnabled()) {
        std::ostringstream os;
        os << "capture_layer=" << capture_layer_idx
           << " selected_count=" << selected_local_indices.size()
           << " selected_query_count=" << (selected_query_count != nullptr ? *selected_query_count : 0);
        picRequestProfileLog("prefill_fusionrag_online_select_external_pagedkv_total",
                             picRequestMonotonicUs() - totalStartUs, os.str());
    }
    return true;
}

bool Llm::prefillFixedGraphExternalPagedKV(const std::vector<int>& full_prompt_token_ids,
                                           const std::vector<MNN::PagedKVExternalSegment>& segments,
                                           int pic_start, int pic_token_count, int score_layer_idx,
                                           const std::vector<int>& selected_local_indices) {
    if (!mConfig->paged_attention() || full_prompt_token_ids.empty()) {
        return false;
    }
    if (!mConfig->has_pic_recompute_budget()) {
        MNN_ERROR("Fixed graph-level PIC prefill requires pic_recompute_budget input\n");
        return false;
    }
    if (pic_start < 0 || pic_token_count < 0 || score_layer_idx < 0 ||
        pic_start + pic_token_count > static_cast<int>(full_prompt_token_ids.size())) {
        return false;
    }
    auto paged = static_cast<PagedKVMeta*>(mMeta.get());
    if (paged == nullptr) {
        return false;
    }
    beginPagedRequestIfNeeded();
    paged->finishPicGraphActivePlan();
    paged->finishSparseQuery();

    std::vector<MNN::PagedKVExternalSegment> boundSegments;
    if (!buildBoundExternalSegments(segments, pic_start, pic_token_count,
                                    "Fixed graph-level PIC", boundSegments)) {
        return false;
    }
    if (!paged->bindExternalSegments(boundSegments, static_cast<int>(full_prompt_token_ids.size()))) {
        MNN_ERROR("Fixed graph-level PIC failed to bind persistent PIC cache source segments\n");
        return false;
    }
    paged->external_hydrate_start_layer_idx = score_layer_idx + 1;
    if (!paged->beginPicGraphActivePlan(pic_start, pic_token_count, score_layer_idx, selected_local_indices)) {
        MNN_ERROR("Fixed graph-level PIC failed to bind active rows\n");
        return false;
    }
    const bool ok = prefill(full_prompt_token_ids);
    paged->finishPicGraphActivePlan();
    paged->finishSparseQuery();
    return ok;
}

} // namespace Transformer
} // namespace MNN
