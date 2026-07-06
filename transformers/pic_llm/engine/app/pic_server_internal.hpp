#pragma once

#include "core/PagedKVMeta.hpp"
#include "jsonhpp/json.hpp"
#include "llm/llm.hpp"

#include <sstream>
#include <string>
#include <vector>

namespace pic {

using json = nlohmann::json;

struct ExplicitCacheTokenSpan {
    int promptStart = -1;
    int sourceStart = 0;
    int tokenCount = 0;
};

struct PicDecodeRefineConfig {
    bool enabled = false;
    int tokensPerDecodeStep = 1;
    int topM = 32;
    double scoreThreshold = 1e-6;
    double scoreMargin = 0.0;
    double scoreDecay = 0.8;
    std::string selector = "top_hkvd";
    int attentionLayerIdx = -1;
    std::vector<int> attentionHeadIds;
};

struct PreparedPicCache {
    std::string id;
    std::string cacheName;
    std::string placeholder = "{{pic_cache}}";
    std::string selectionAlgorithm = "full-reuse";
    double recomputeRatio = 0.20;
    int scoreLayerIdx = 1;
    int fusionragCaptureLayerIdx = -1;
    int fusionragQueryTailTokens = 0;
    std::string fusionragScoreReduce = "sum_query_mean_head";
    bool hasExplicitRecomputeIndices = false;
    std::vector<int> explicitLogicalIndices;
    std::vector<int> explicitPicLocalIndices;
    std::vector<int> fullPromptTokenIds;
    std::vector<ExplicitCacheTokenSpan> promptSpans;
    std::vector<int> tokenIds;
    json textCaches = json::array();
    std::vector<MNN::PagedKVExternalSegment> segments;
    PicDecodeRefineConfig decodeRefine;
};

struct PicExecutionPlan {
    std::string selectionAlgorithm = "full-reuse";
    std::string plannerName = "FullReusePlanner";
    std::string executionMode = "native-full-reuse";
    std::string fallbackReason;
    int scoreLayerIdx = 0;
    int recomputeTokenCount = 0;
    std::vector<int> recomputeLogicalIndices;
    json recomputeScores = json::object();
    json metadata = json::object();
    std::vector<int> prefillPicTokenIds;
    std::vector<int> externalTokenIds;
    std::vector<MNN::PagedKVExternalSegment> externalSegments;
    std::vector<int> sparseLogicalIndices;
    std::vector<int> sparseTokenIds;
    bool fullCompute = false;
    bool sparseRecompute = false;
    PicDecodeRefineConfig decodeRefine;
};

struct PicPrefillResult {
    PicExecutionPlan plan;
    std::vector<int> nativeSelectedLocalIndices;
    bool hasNativeSelectedLocalIndices = false;
};

bool isCacheBlendAlgorithm(const std::string& algorithm);
bool isKvshareAlgorithm(const std::string& algorithm);
bool isFusionRAGOnlineAlgorithm(const std::string& algorithm);
bool isExternalRequestIndexAlgorithm(const std::string& algorithm);

int fusionragReduceModeFromString(const std::string& mode);
const char* fusionragReduceModeName(int mode);

std::vector<int> collectExplicitPicLocalIndices(const PreparedPicCache& pic, int preludeTokenCount);
std::string plannerNameForAlgorithm(const std::string& algorithm);
std::string sparseExecutionModeForAlgorithm(const std::string& algorithm);
std::string scoreKindForAlgorithm(const std::string& algorithm);

PicExecutionPlan buildExecutionPlan(const PreparedPicCache& pic, int preludeTokenCount, int layerCount,
                                    const std::vector<int>* nativeSelectedLocalIndices,
                                    const json* scoreMetadata);

bool runPicPrefill(MNN::Transformer::Llm* llm, std::ostringstream& sink, const std::string& runtimeBackend,
                   const PreparedPicCache& pic, const std::vector<int>& fullPromptTokenIds,
                   int preludeTokenCount, int layerCount, bool modelSupportsGraphBoundary,
                   PicPrefillResult& result, std::string& error);

} // namespace pic
