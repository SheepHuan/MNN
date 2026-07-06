#include "pic_server_internal.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>

namespace pic {
namespace {

int envInt(const char* name, int fallback = 0) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

bool picRequestProfileEnabled() {
    return envInt("MNN_PIC_REQUEST_PROFILE", 0) > 0;
}

void picRequestProfileLog(const char* stage, int64_t elapsedUs, const std::string& detail = "") {
    if (!picRequestProfileEnabled()) {
        return;
    }
    std::fprintf(stderr, "MNN_PIC_REQUEST_PROFILE phase=end stage=%s cost_ms=%.3f", stage, elapsedUs / 1000.0);
    if (!detail.empty()) {
        std::fprintf(stderr, " %s", detail.c_str());
    }
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

void picRequestProfileBegin(const char* stage, const std::string& detail = "") {
    if (!picRequestProfileEnabled()) {
        return;
    }
    std::fprintf(stderr, "MNN_PIC_REQUEST_PROFILE phase=begin stage=%s", stage);
    if (!detail.empty()) {
        std::fprintf(stderr, " %s", detail.c_str());
    }
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

int64_t monotonicUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::string llmContextSuffix(const MNN::Transformer::Llm* llm) {
    const auto* context = llm != nullptr ? llm->getContext() : nullptr;
    if (context == nullptr) {
        return " [llm_context=null]";
    }
    std::ostringstream os;
    os << " [status=" << static_cast<int>(context->status)
       << ", current=" << context->current_token
       << ", all_seq=" << context->all_seq_len
       << ", prompt=" << context->prompt_len
       << ", gen_seq=" << context->gen_seq_len
       << ", output_tokens=" << context->output_tokens.size();
    if (!llm->lastError().empty()) {
        os << ", last_error=" << llm->lastError();
    }
    os << "]";
    return os.str();
}

std::string lowerAscii(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

} // namespace

bool isCacheBlendAlgorithm(const std::string& algorithm) {
    return algorithm == "cacheblend" || algorithm == "delta-v";
}

bool isKvshareAlgorithm(const std::string& algorithm) {
    return algorithm == "kvshare" || algorithm == "delta-a";
}

bool isFusionRAGOnlineAlgorithm(const std::string& algorithm) {
    return algorithm == "fusionrag_online";
}

bool isExternalRequestIndexAlgorithm(const std::string& algorithm) {
    return algorithm == "explicit" || algorithm == "cacheclip";
}

int fusionragReduceModeFromString(const std::string& mode) {
    const std::string normalized = lowerAscii(mode);
    if (normalized.empty() || normalized == "sum_query_mean_head") {
        return MNN::PagedKVMeta::FusionRAGReduceSumQueryMeanHead;
    }
    if (normalized == "mean_query_mean_head") {
        return MNN::PagedKVMeta::FusionRAGReduceMeanQueryMeanHead;
    }
    if (normalized == "max_query_mean_head") {
        return MNN::PagedKVMeta::FusionRAGReduceMaxQueryMeanHead;
    }
    return -1;
}

const char* fusionragReduceModeName(int mode) {
    switch (mode) {
        case MNN::PagedKVMeta::FusionRAGReduceMeanQueryMeanHead:
            return "mean_query_mean_head";
        case MNN::PagedKVMeta::FusionRAGReduceMaxQueryMeanHead:
            return "max_query_mean_head";
        case MNN::PagedKVMeta::FusionRAGReduceSumQueryMeanHead:
        default:
            return "sum_query_mean_head";
    }
}

std::vector<int> collectExplicitPicLocalIndices(const PreparedPicCache& pic, int preludeTokenCount) {
    const int picLength = static_cast<int>(pic.tokenIds.size());
    std::vector<int> localIndices;
    localIndices.reserve(pic.explicitPicLocalIndices.size() + pic.explicitLogicalIndices.size());
    for (int index : pic.explicitPicLocalIndices) {
        if (index >= 0 && index < picLength) {
            localIndices.emplace_back(index);
        }
    }
    for (int logical : pic.explicitLogicalIndices) {
        const int local = logical - preludeTokenCount;
        if (local >= 0 && local < picLength) {
            localIndices.emplace_back(local);
        }
    }
    std::sort(localIndices.begin(), localIndices.end());
    localIndices.erase(std::unique(localIndices.begin(), localIndices.end()), localIndices.end());
    return localIndices;
}

std::string plannerNameForAlgorithm(const std::string& algorithm) {
    if (isCacheBlendAlgorithm(algorithm)) {
        return "CacheBlendPlanner";
    }
    if (algorithm == "cacheclip") {
        return "CacheClipExternalPlanner";
    }
    if (isFusionRAGOnlineAlgorithm(algorithm)) {
        return "FusionRAGOnlinePlanner";
    }
    if (algorithm == "explicit") {
        return "ExplicitPlanner";
    }
    return "KvsharePlanner";
}

std::string sparseExecutionModeForAlgorithm(const std::string& algorithm) {
    if (isCacheBlendAlgorithm(algorithm)) {
        return "native-cacheblend-sparse-recompute";
    }
    if (algorithm == "cacheclip") {
        return "native-cacheclip-sparse-recompute";
    }
    if (isFusionRAGOnlineAlgorithm(algorithm)) {
        return "native-fusionrag-online-sparse-recompute";
    }
    if (algorithm == "explicit") {
        return "native-explicit-sparse-recompute";
    }
    return "native-kvshare-sparse-recompute";
}

std::string scoreKindForAlgorithm(const std::string& algorithm) {
    if (isCacheBlendAlgorithm(algorithm)) {
        return "layer_value_delta_mean_abs";
    }
    if (algorithm == "cacheclip") {
        return "cacheclip_aux_attention_grouped_request_indices";
    }
    if (isFusionRAGOnlineAlgorithm(algorithm)) {
        return "fusionrag_online_query_attention_request_indices";
    }
    if (algorithm == "explicit") {
        return "explicit_request_indices";
    }
    return "key_value_delta_influence_proxy";
}

PicExecutionPlan buildExecutionPlan(const PreparedPicCache& pic, int preludeTokenCount, int layerCount,
                                    const std::vector<int>* nativeSelectedLocalIndices,
                                    const json* scoreMetadata) {
    PicExecutionPlan plan;
    plan.selectionAlgorithm = pic.selectionAlgorithm;
    plan.decodeRefine = pic.decodeRefine;
    const int picStart = preludeTokenCount;
    const int picLength = static_cast<int>(pic.tokenIds.size());
    const int picEnd = picStart + picLength;
    const int effectiveScoreLayer = std::min(std::max(0, pic.scoreLayerIdx), std::max(0, layerCount - 1));
    auto fillAllRecompute = [&]() {
        plan.recomputeLogicalIndices.clear();
        for (int i = picStart; i < picEnd; ++i) {
            plan.recomputeLogicalIndices.emplace_back(i);
            plan.recomputeScores[std::to_string(i)] = 1.0;
        }
        plan.recomputeTokenCount = static_cast<int>(plan.recomputeLogicalIndices.size());
    };
    auto fillFullComputeFallback = [&](const std::string& plannerName, const std::string& scoreKind,
                                       const std::string& reason) {
        plan.plannerName = plannerName;
        plan.executionMode = "full-compute-fallback";
        plan.fullCompute = true;
        plan.scoreLayerIdx = layerCount;
        plan.fallbackReason = reason;
        fillAllRecompute();
        plan.prefillPicTokenIds = pic.tokenIds;
        plan.metadata = {
            {"score_kind", scoreKind},
            {"pic_token_count", picLength},
            {"reuse_token_count", 0},
            {"fallback_reason", plan.fallbackReason},
            {"native_sparse_recompute", false},
        };
    };
    if (pic.selectionAlgorithm == "full-reuse") {
        plan.plannerName = "FullReusePlanner";
        plan.executionMode = "native-full-reuse";
        plan.scoreLayerIdx = effectiveScoreLayer;
        plan.externalTokenIds = pic.tokenIds;
        plan.externalSegments = pic.segments;
        plan.metadata = {
            {"score_kind", "none"},
            {"pic_token_count", picLength},
            {"reuse_token_count", picLength},
            {"score_pass", "not_required_full_reuse"},
            {"suffix_query_source", "current_context"},
        };
        return plan;
    }
    if (pic.selectionAlgorithm == "full-compute") {
        plan.plannerName = "FullComputePlanner";
        plan.executionMode = "native-full-compute";
        plan.fullCompute = true;
        plan.scoreLayerIdx = layerCount;
        fillAllRecompute();
        plan.prefillPicTokenIds = pic.tokenIds;
        plan.metadata = {
            {"score_kind", "none"},
            {"pic_token_count", picLength},
            {"reuse_token_count", 0},
        };
        return plan;
    }
    if (pic.selectionAlgorithm == "epic") {
        plan.plannerName = "EpicPlanner";
        plan.executionMode = "native-epic-sparse-recompute";
        plan.scoreLayerIdx = effectiveScoreLayer;
        int recompute = picLength <= 0 || pic.recomputeRatio <= 0.0
            ? 0
            : std::min(picLength, std::max(1, static_cast<int>(std::ceil(picLength * pic.recomputeRatio))));
        plan.externalTokenIds = pic.tokenIds;
        plan.externalSegments = pic.segments;
        for (int i = 0; i < recompute; ++i) {
            int logical = picStart + i;
            plan.recomputeLogicalIndices.emplace_back(logical);
            plan.sparseLogicalIndices.emplace_back(logical);
            plan.sparseTokenIds.emplace_back(pic.tokenIds[i]);
            plan.recomputeScores[std::to_string(logical)] = static_cast<double>(recompute - i);
        }
        plan.recomputeTokenCount = recompute;
        plan.sparseRecompute = recompute > 0;
        plan.metadata = {
            {"score_kind", "pic_head_fixed_ratio"},
            {"score_source", "fixed_pic_head_contiguous_tokens"},
            {"score_pass", "not_required_fixed_pic_head"},
            {"pic_token_count", picLength},
            {"reuse_token_count", picLength - recompute},
            {"native_sparse_recompute", true},
            {"native_sparse_recompute_scope", "python_prefill_layer_plan"},
            {"pre_score_compute_layers", plan.scoreLayerIdx},
            {"pre_score_kv_source", "none_fixed_pic_head_selection"},
            {"post_score_reuse_kv_source", "cached_pic_kv"},
        };
        return plan;
    }
    plan.plannerName = plannerNameForAlgorithm(pic.selectionAlgorithm);
    if (nativeSelectedLocalIndices != nullptr || isExternalRequestIndexAlgorithm(pic.selectionAlgorithm)) {
        plan.executionMode = sparseExecutionModeForAlgorithm(pic.selectionAlgorithm);
        plan.scoreLayerIdx = effectiveScoreLayer;
        plan.externalTokenIds = pic.tokenIds;
        plan.externalSegments = pic.segments;
        std::vector<int> localIndices;
        if (isExternalRequestIndexAlgorithm(pic.selectionAlgorithm)) {
            localIndices = collectExplicitPicLocalIndices(pic, picStart);
        } else {
            localIndices = *nativeSelectedLocalIndices;
            std::sort(localIndices.begin(), localIndices.end());
            localIndices.erase(std::unique(localIndices.begin(), localIndices.end()), localIndices.end());
        }
        for (size_t rank = 0; rank < localIndices.size(); ++rank) {
            int local = localIndices[rank];
            if (local < 0 || local >= picLength) {
                continue;
            }
            int logical = picStart + local;
            plan.recomputeLogicalIndices.emplace_back(logical);
            plan.sparseLogicalIndices.emplace_back(logical);
            plan.sparseTokenIds.emplace_back(pic.tokenIds[local]);
            plan.recomputeScores[std::to_string(logical)] = static_cast<double>(localIndices.size() - rank);
        }
        plan.recomputeTokenCount = static_cast<int>(plan.recomputeLogicalIndices.size());
        plan.sparseRecompute = plan.recomputeTokenCount > 0;
        const std::string scoreKind = scoreKindForAlgorithm(pic.selectionAlgorithm);
        plan.metadata = {
            {"score_kind", scoreKind},
            {"pic_token_count", picLength},
            {"reuse_token_count", picLength - plan.recomputeTokenCount},
            {"native_sparse_recompute", true},
            {"native_sparse_recompute_scope", "python_prefill_layer_plan"},
            {"pre_score_compute_layers", plan.scoreLayerIdx},
            {"pre_score_kv_source", scoreMetadata != nullptr && plan.scoreLayerIdx > 0
                                         ? "request_full_reference_pagedcache_no_disk_write"
                                         : "none"},
            {"post_score_reuse_kv_source", "cached_pic_kv"},
        };
        if (scoreMetadata != nullptr) {
            plan.metadata["score_metadata"] = *scoreMetadata;
        }
        if (isKvshareAlgorithm(pic.selectionAlgorithm)) {
            plan.metadata["python_reference"] =
                "hf_pic_runtime kvshare uses attention_output gradient influence; MNN C++ currently uses a native K/V delta influence proxy because exported MNN modules do not expose score-layer query/attention-output autograd.";
        }
        return plan;
    }
    const std::string scoreKind = isCacheBlendAlgorithm(pic.selectionAlgorithm)
        ? "layer_value_delta_mean_abs_gpu_topk_unavailable"
        : "attention_output_error_kv_first_order_influence_gpu_topk_unavailable";
    fillFullComputeFallback(
        plan.plannerName, scoreKind,
        "Native CUDA/OpenCL score + GPU top-k is not implemented; legacy scratch .k/.v CPU scoring is disabled");
    return plan;
}

bool runPicPrefill(MNN::Transformer::Llm* llm, std::ostringstream& sink, const std::string& runtimeBackend,
                   const PreparedPicCache& pic, const std::vector<int>& fullPromptTokenIds,
                   int preludeTokenCount, int layerCount, bool modelSupportsGraphBoundary,
                   PicPrefillResult& result, std::string& error) {
    if (llm == nullptr) {
        error = "LLM is not loaded";
        return false;
    }

    result = PicPrefillResult();
    json scoreMetadata = json::object();
    bool graphBoundaryPrefillDone = false;
    bool fullReusePrefillDone = false;
    bool nativePicPrefillDone = false;
    const bool graphBoundaryEnabled =
        modelSupportsGraphBoundary &&
        (isCacheBlendAlgorithm(pic.selectionAlgorithm) || pic.selectionAlgorithm == "epic" ||
         isFusionRAGOnlineAlgorithm(pic.selectionAlgorithm) ||
         isExternalRequestIndexAlgorithm(pic.selectionAlgorithm));

    int64_t stageUs = 0;
    if (pic.selectionAlgorithm == "full-reuse") {
        result.nativeSelectedLocalIndices.clear();
        llm->reset();
        llm->generate_init(&sink, "");
        stageUs = monotonicUs();
        {
            std::ostringstream os;
            os << "selection_algorithm=" << pic.selectionAlgorithm
               << " prelude_tokens=" << preludeTokenCount
               << " pic_tokens=" << pic.tokenIds.size()
               << " suffix_tokens=" << (fullPromptTokenIds.size() - preludeTokenCount - pic.tokenIds.size());
            picRequestProfileBegin("prefill_full_reuse_external_pagedkv", os.str());
        }
        if (!llm->prefillFullReuseExternalPagedKV(
                fullPromptTokenIds, pic.segments, preludeTokenCount, static_cast<int>(pic.tokenIds.size()))) {
            llm->finishExternalPagedKVRequest();
            error = "Native full-reuse hydrate+suffix prefill failed on backend " + runtimeBackend +
                    llmContextSuffix(llm);
            return false;
        }
        {
            std::ostringstream os;
            os << "selection_algorithm=" << pic.selectionAlgorithm
               << " prelude_tokens=" << preludeTokenCount
               << " pic_tokens=" << pic.tokenIds.size()
               << " suffix_tokens=" << (fullPromptTokenIds.size() - preludeTokenCount - pic.tokenIds.size());
            picRequestProfileLog("prefill_full_reuse_external_pagedkv", monotonicUs() - stageUs, os.str());
        }
        fullReusePrefillDone = true;
        nativePicPrefillDone = true;
        scoreMetadata = {
            {"score_source", "none"},
            {"score_kind", "none"},
            {"score_pass", "not_required_full_reuse_hydrate_suffix"},
            {"graph_boundary", "none_full_reuse_hydrate_suffix"},
            {"selected_count", 0},
        };
    } else if (graphBoundaryEnabled && pic.selectionAlgorithm == "epic") {
        const int effectiveScoreLayer =
            std::min(std::max(0, pic.scoreLayerIdx), std::max(0, layerCount - 1));
        const int picTokenCount = static_cast<int>(pic.tokenIds.size());
        const int recompute = picTokenCount <= 0 || pic.recomputeRatio <= 0.0
            ? 0
            : std::min(picTokenCount, std::max(1, static_cast<int>(std::ceil(picTokenCount * pic.recomputeRatio))));
        result.nativeSelectedLocalIndices.clear();
        for (int i = 0; i < recompute; ++i) {
            result.nativeSelectedLocalIndices.emplace_back(i);
        }
        llm->reset();
        llm->generate_init(&sink, "");
        stageUs = monotonicUs();
        {
            std::ostringstream os;
            os << "selection_algorithm=" << pic.selectionAlgorithm
               << " effective_score_layer=" << effectiveScoreLayer
               << " selected_count=" << result.nativeSelectedLocalIndices.size();
            picRequestProfileBegin("prefill_fixed_graph_external_pagedkv", os.str());
        }
        if (!llm->prefillFixedGraphExternalPagedKV(
                fullPromptTokenIds, pic.segments, preludeTokenCount, picTokenCount, effectiveScoreLayer,
                result.nativeSelectedLocalIndices)) {
            llm->finishExternalPagedKVRequest();
            error = "Native graph-level epic prefill failed on backend " + runtimeBackend + llmContextSuffix(llm);
            return false;
        }
        {
            std::ostringstream os;
            os << "selection_algorithm=" << pic.selectionAlgorithm
               << " effective_score_layer=" << effectiveScoreLayer
               << " selected_count=" << result.nativeSelectedLocalIndices.size();
            picRequestProfileLog("prefill_fixed_graph_external_pagedkv", monotonicUs() - stageUs, os.str());
        }
        graphBoundaryPrefillDone = true;
        nativePicPrefillDone = true;
        scoreMetadata = {
            {"score_layer_idx", effectiveScoreLayer},
            {"score_source", "fixed_pic_head_contiguous_tokens"},
            {"score_kind", "pic_head_fixed_ratio"},
            {"score_pass", "not_required_graph_boundary"},
            {"graph_boundary", "score_layer_pic_score_attention"},
            {"selected_count", result.nativeSelectedLocalIndices.size()},
        };
    } else if (graphBoundaryEnabled && isFusionRAGOnlineAlgorithm(pic.selectionAlgorithm)) {
        const int effectiveScoreLayer =
            std::min(std::max(0, pic.scoreLayerIdx), std::max(0, layerCount - 1));
        const int fusionragCaptureLayer = pic.fusionragCaptureLayerIdx >= 0
            ? std::min(std::max(0, pic.fusionragCaptureLayerIdx), std::max(0, layerCount - 1))
            : std::max(0, layerCount - 1);
        const int fusionragReduceMode = fusionragReduceModeFromString(pic.fusionragScoreReduce);
        int fusionragSelectedQueryCount = 0;
        result.nativeSelectedLocalIndices.clear();

        llm->reset();
        llm->generate_init(&sink, "");
        stageUs = monotonicUs();
        {
            std::ostringstream os;
            os << "selection_algorithm=" << pic.selectionAlgorithm
               << " capture_layer=" << fusionragCaptureLayer
               << " effective_score_layer=" << effectiveScoreLayer
               << " recompute_ratio=" << pic.recomputeRatio
               << " query_tail_tokens=" << pic.fusionragQueryTailTokens
               << " reduce_mode=" << fusionragReduceModeName(fusionragReduceMode);
            picRequestProfileBegin("prefill_fusionrag_online_select_external_pagedkv", os.str());
        }
        if (!llm->prefillFusionRAGOnlineSelectExternalPagedKV(
                fullPromptTokenIds, pic.segments, preludeTokenCount, static_cast<int>(pic.tokenIds.size()),
                fusionragCaptureLayer, pic.recomputeRatio, pic.fusionragQueryTailTokens, fusionragReduceMode,
                result.nativeSelectedLocalIndices, &fusionragSelectedQueryCount)) {
            llm->finishExternalPagedKVRequest();
            error = "FusionRAG-online qcompute selector failed on backend " + runtimeBackend +
                    llmContextSuffix(llm);
            return false;
        }
        {
            std::ostringstream os;
            os << "selection_algorithm=" << pic.selectionAlgorithm
               << " capture_layer=" << fusionragCaptureLayer
               << " selected_count=" << result.nativeSelectedLocalIndices.size()
               << " selected_query_count=" << fusionragSelectedQueryCount;
            picRequestProfileLog("prefill_fusionrag_online_select_external_pagedkv",
                                 monotonicUs() - stageUs, os.str());
        }

        llm->reset();
        llm->generate_init(&sink, "");
        stageUs = monotonicUs();
        {
            std::ostringstream os;
            os << "selection_algorithm=" << pic.selectionAlgorithm
               << " effective_score_layer=" << effectiveScoreLayer
               << " selected_count=" << result.nativeSelectedLocalIndices.size();
            picRequestProfileBegin("prefill_fixed_graph_external_pagedkv", os.str());
        }
        if (!llm->prefillFixedGraphExternalPagedKV(
                fullPromptTokenIds, pic.segments, preludeTokenCount, static_cast<int>(pic.tokenIds.size()),
                effectiveScoreLayer, result.nativeSelectedLocalIndices)) {
            llm->finishExternalPagedKVRequest();
            error = "Native graph-level fusionrag_online prefill failed on backend " + runtimeBackend +
                    llmContextSuffix(llm);
            return false;
        }
        {
            std::ostringstream os;
            os << "selection_algorithm=" << pic.selectionAlgorithm
               << " effective_score_layer=" << effectiveScoreLayer
               << " selected_count=" << result.nativeSelectedLocalIndices.size();
            picRequestProfileLog("prefill_fixed_graph_external_pagedkv", monotonicUs() - stageUs, os.str());
        }
        graphBoundaryPrefillDone = true;
        nativePicPrefillDone = true;
        result.hasNativeSelectedLocalIndices = true;
        scoreMetadata = {
            {"score_layer_idx", effectiveScoreLayer},
            {"capture_layer_idx", fusionragCaptureLayer},
            {"score_source", "request_full_reuse_query_attention_over_cached_pic_keys"},
            {"score_kind", scoreKindForAlgorithm(pic.selectionAlgorithm)},
            {"score_pass", "request_qcompute_then_graph_boundary"},
            {"topk_location", "backend_host"},
            {"host_transfer", "selected_local_indices_only"},
            {"graph_boundary", "score_layer_pic_score_attention"},
            {"selected_count", result.nativeSelectedLocalIndices.size()},
            {"selected_query_count", fusionragSelectedQueryCount},
            {"query_tail_tokens", pic.fusionragQueryTailTokens},
            {"score_reduce", fusionragReduceModeName(fusionragReduceMode)},
        };
    } else if (graphBoundaryEnabled && isExternalRequestIndexAlgorithm(pic.selectionAlgorithm)) {
        const int effectiveScoreLayer =
            std::min(std::max(0, pic.scoreLayerIdx), std::max(0, layerCount - 1));
        result.nativeSelectedLocalIndices = collectExplicitPicLocalIndices(pic, preludeTokenCount);
        llm->reset();
        llm->generate_init(&sink, "");
        stageUs = monotonicUs();
        {
            std::ostringstream os;
            os << "selection_algorithm=" << pic.selectionAlgorithm
               << " effective_score_layer=" << effectiveScoreLayer
               << " selected_count=" << result.nativeSelectedLocalIndices.size();
            picRequestProfileBegin("prefill_fixed_graph_external_pagedkv", os.str());
        }
        if (!llm->prefillFixedGraphExternalPagedKV(
                fullPromptTokenIds, pic.segments, preludeTokenCount, static_cast<int>(pic.tokenIds.size()),
                effectiveScoreLayer, result.nativeSelectedLocalIndices)) {
            llm->finishExternalPagedKVRequest();
            error = "Native graph-level " + pic.selectionAlgorithm + " prefill failed on backend " +
                    runtimeBackend + llmContextSuffix(llm);
            return false;
        }
        {
            std::ostringstream os;
            os << "selection_algorithm=" << pic.selectionAlgorithm
               << " effective_score_layer=" << effectiveScoreLayer
               << " selected_count=" << result.nativeSelectedLocalIndices.size();
            picRequestProfileLog("prefill_fixed_graph_external_pagedkv", monotonicUs() - stageUs, os.str());
        }
        graphBoundaryPrefillDone = true;
        nativePicPrefillDone = true;
        result.hasNativeSelectedLocalIndices = true;
        const std::string scoreSource = pic.selectionAlgorithm == "cacheclip"
            ? "external_cacheclip_aux_attention_grouped"
            : (pic.selectionAlgorithm == "fusionrag_online"
                   ? "external_fusionrag_online_query_attention"
                   : "explicit_request_indices");
        scoreMetadata = {
            {"score_layer_idx", effectiveScoreLayer},
            {"score_source", scoreSource},
            {"score_kind", scoreKindForAlgorithm(pic.selectionAlgorithm)},
            {"score_pass", "external_selector_before_request"},
            {"topk_location", "external_selector"},
            {"host_transfer", "selected_local_indices_only"},
            {"graph_boundary", "score_layer_pic_score_attention"},
            {"selected_count", result.nativeSelectedLocalIndices.size()},
        };
    } else if (graphBoundaryEnabled) {
        const int effectiveScoreLayer =
            std::min(std::max(0, pic.scoreLayerIdx), std::max(0, layerCount - 1));
        llm->reset();
        llm->generate_init(&sink, "");
        stageUs = monotonicUs();
        {
            std::ostringstream os;
            os << "selection_algorithm=" << pic.selectionAlgorithm
               << " effective_score_layer=" << effectiveScoreLayer
               << " recompute_ratio=" << pic.recomputeRatio
               << " prompt_tokens=" << fullPromptTokenIds.size()
               << " pic_tokens=" << pic.tokenIds.size();
            picRequestProfileBegin("prefill_cacheblend_graph_external_pagedkv", os.str());
        }
        if (!llm->prefillCacheBlendGraphExternalPagedKV(
                fullPromptTokenIds, pic.segments, preludeTokenCount, static_cast<int>(pic.tokenIds.size()),
                effectiveScoreLayer, pic.recomputeRatio, result.nativeSelectedLocalIndices)) {
            llm->finishExternalPagedKVRequest();
            error = "Native graph-level cacheblend score/top-k failed on backend " + runtimeBackend +
                    llmContextSuffix(llm);
            return false;
        }
        {
            std::ostringstream os;
            os << "selection_algorithm=" << pic.selectionAlgorithm
               << " effective_score_layer=" << effectiveScoreLayer
               << " selected_count=" << result.nativeSelectedLocalIndices.size()
               << " recompute_ratio=" << pic.recomputeRatio;
            picRequestProfileLog("prefill_cacheblend_graph_external_pagedkv", monotonicUs() - stageUs, os.str());
        }
        graphBoundaryPrefillDone = true;
        nativePicPrefillDone = true;
        result.hasNativeSelectedLocalIndices = true;
        scoreMetadata = {
            {"score_layer_idx", effectiveScoreLayer},
            {"score_source", "request_full_reference_pagedcache_minus_cached_pic_value"},
            {"score_kind", "layer_value_delta_mean_abs"},
            {"score_pass", "request_full_prompt_graph_boundary"},
            {"topk_location", "backend_device"},
            {"host_transfer", "selected_local_indices_only"},
            {"graph_boundary", "score_layer_pic_score_attention"},
            {"selected_count", result.nativeSelectedLocalIndices.size()},
        };
    }

    stageUs = monotonicUs();
    picRequestProfileBegin("build_execution_plan");
    result.plan = buildExecutionPlan(
        pic, preludeTokenCount, layerCount,
        result.hasNativeSelectedLocalIndices ? &result.nativeSelectedLocalIndices : nullptr,
        result.hasNativeSelectedLocalIndices ? &scoreMetadata : nullptr);
    {
        std::ostringstream os;
        const int reuseTokenCount = result.plan.metadata.contains("reuse_token_count") &&
                result.plan.metadata["reuse_token_count"].is_number_integer()
            ? result.plan.metadata["reuse_token_count"].get<int>()
            : -1;
        os << "full_compute=" << (result.plan.fullCompute ? 1 : 0)
           << " sparse_recompute=" << (result.plan.sparseRecompute ? 1 : 0)
           << " recompute_tokens=" << result.plan.recomputeTokenCount
           << " reuse_tokens=" << reuseTokenCount
           << " score_layer=" << result.plan.scoreLayerIdx
           << " execution_mode=" << result.plan.executionMode;
        picRequestProfileLog("build_execution_plan", monotonicUs() - stageUs, os.str());
    }

    if (fullReusePrefillDone) {
        result.plan.executionMode = "native-full-reuse-hydrate-suffix";
        result.plan.sparseRecompute = false;
        result.plan.recomputeTokenCount = 0;
        result.plan.metadata["native_sparse_recompute"] = false;
        result.plan.metadata["score_pass"] = "not_required_full_reuse_hydrate_suffix";
        result.plan.metadata["graph_boundary"] = "none_full_reuse_hydrate_suffix";
        result.plan.metadata["graph_level_boundary"] = false;
        result.plan.metadata["full_reuse_dataflow"] =
            "persistent_pic_cache_source_to_current_request_pagedcache_then_suffix_prefill";
    } else if (graphBoundaryPrefillDone) {
        if (pic.selectionAlgorithm == "epic") {
            result.plan.executionMode = "native-epic-graph-boundary";
        } else if (pic.selectionAlgorithm == "cacheclip") {
            result.plan.executionMode = "native-cacheclip-graph-boundary";
        } else if (pic.selectionAlgorithm == "fusionrag_online") {
            result.plan.executionMode = "native-fusionrag-online-graph-boundary";
        } else if (pic.selectionAlgorithm == "explicit") {
            result.plan.executionMode = "native-explicit-graph-boundary";
        } else {
            result.plan.executionMode = "native-cacheblend-graph-boundary";
        }
        result.plan.sparseRecompute = false;
        result.plan.metadata["native_sparse_recompute_scope"] = "exported_graph_score_layer_boundary";
        result.plan.metadata["graph_level_boundary"] = true;
    }

    if (!nativePicPrefillDone && result.plan.sparseRecompute && result.plan.scoreLayerIdx > 0) {
        error = "PIC sparse prefill score_layer_idx=" + std::to_string(result.plan.scoreLayerIdx) +
                " requires a graph-boundary PIC model with pic_recompute_budget; "
                "legacy forwardVec(selected_tokens) sparse recompute from layer 0 has been removed";
        return false;
    }

    if (!nativePicPrefillDone) {
        llm->reset();
        llm->generate_init(&sink, "");
    }
    if (!nativePicPrefillDone && result.plan.fullCompute) {
        picRequestProfileBegin("prefill_full_compute_pic_chat");
        if (!llm->prefill(fullPromptTokenIds)) {
            llm->finishExternalPagedKVRequest();
            error = "Failed to prefill full-compute PIC chat request" + llmContextSuffix(llm);
            return false;
        }
        if (std::getenv("MNN_PIC_DECODE_DEBUG") != nullptr) {
            auto context = llm->getContext();
            std::fprintf(stderr, "PIC server prefill debug step=full_compute status=%d all_seq=%d\n",
                         context != nullptr ? static_cast<int>(context->status) : -999,
                         context != nullptr ? context->all_seq_len : -1);
            std::fflush(stderr);
        }
    } else if (!nativePicPrefillDone) {
        llm->finishExternalPagedKVRequest();
        error = "PIC " + pic.selectionAlgorithm +
                " requires a graph-boundary PIC model with pic_recompute_budget; "
                "legacy split prefix/append/suffix prefill has been removed";
        return false;
    }

    return true;
}

} // namespace pic
