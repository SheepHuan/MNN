//
//  pic_server.cpp
//  MNN
//

#include "pic_server.hpp"

#include "core/PagedKVMeta.hpp"
#include "perfetto_c.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <system_error>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace pic {
namespace {

constexpr const char* kMetaFormat = "kvshare-prefix-cache-meta-v1";
constexpr const char* kTokensFormat = "kvshare-prefix-cache-tokens-v1";
constexpr const char* kDefaultPicPlaceholder = "{{pic_cache}}";

struct MnnLlmTraceInfo {
    std::string algorithm = "full-compute";
    std::string executionMode = "native-full-compute";
    double budgetRatio = 0.0;
    int64_t recomputeBudgetTokens = 0;
    int64_t promptTotalTokens = 0;
    int64_t preludeTokens = 0;
    int64_t picTokens = 0;
    int64_t suffixTokens = 0;
    int64_t maxTokens = 0;
    int64_t scoreLayerIdx = 0;
    bool hasPicCache = false;
};

struct MnnLlmPerfettoState {
    std::once_flag once;
    PerfettoTeCategory category = {
        &perfetto_atomic_false,
        nullptr,
        {"mnn.llm", "MNN LLM PIC server request stages", nullptr, 0},
        0,
    };
    PerfettoTeRegisteredTrack track = {};
};

MnnLlmPerfettoState& mnnLlmPerfettoState() {
    static MnnLlmPerfettoState state;
    return state;
}

void ensureMnnLlmPerfetto() {
    auto& state = mnnLlmPerfettoState();
    std::call_once(state.once, [&state]() {
        PerfettoProducerInitArgs args = PERFETTO_PRODUCER_INIT_ARGS_INIT();
        args.backends = PERFETTO_BACKEND_SYSTEM;
        args.shmem_size_hint_kb = 4096;
        PerfettoProducerInit(args);
        PerfettoTeCategoryRegister(&state.category);
        PerfettoTeInit();
        PerfettoTePublishCategories();
        PerfettoTeNamedTrackRegister(&state.track, "MNN LLM", 1, PerfettoTeGlobalTrackUuid(), true);
    });
}

class MnnLlmPerfettoSlice {
public:
    MnnLlmPerfettoSlice(const char* phase, const MnnLlmTraceInfo& info) {
        ensureMnnLlmPerfetto();
        auto& state = mnnLlmPerfettoState();
        PERFETTO_TE(state.category,
                    PERFETTO_TE_SLICE_BEGIN(phase),
                    PERFETTO_TE_REGISTERED_TRACK(&state.track),
                    PERFETTO_TE_ARG_STRING("phase", phase),
                    PERFETTO_TE_ARG_STRING("algorithm", info.algorithm.c_str()),
                    PERFETTO_TE_ARG_STRING("execution_mode", info.executionMode.c_str()),
                    PERFETTO_TE_ARG_DOUBLE("budget_ratio", info.budgetRatio),
                    PERFETTO_TE_ARG_INT64("recompute_budget_tokens", info.recomputeBudgetTokens),
                    PERFETTO_TE_ARG_INT64("prompt_total_tokens", info.promptTotalTokens),
                    PERFETTO_TE_ARG_INT64("prelude_token_count", info.preludeTokens),
                    PERFETTO_TE_ARG_INT64("pic_token_count", info.picTokens),
                    PERFETTO_TE_ARG_INT64("suffix_token_count", info.suffixTokens),
                    PERFETTO_TE_ARG_INT64("max_tokens", info.maxTokens),
                    PERFETTO_TE_ARG_INT64("score_layer_idx", info.scoreLayerIdx),
                    PERFETTO_TE_ARG_BOOL("has_pic_cache", info.hasPicCache));
        mActive = true;
    }

    MnnLlmPerfettoSlice(const MnnLlmPerfettoSlice&) = delete;
    MnnLlmPerfettoSlice& operator=(const MnnLlmPerfettoSlice&) = delete;

    ~MnnLlmPerfettoSlice() {
        if (!mActive) {
            return;
        }
        auto& state = mnnLlmPerfettoState();
        PERFETTO_TE(state.category,
                    PERFETTO_TE_SLICE_END(),
                    PERFETTO_TE_REGISTERED_TRACK(&state.track));
    }

private:
    bool mActive = false;
};

std::string traceAlgorithmName(const std::string& algorithm) {
    if (algorithm == "delta-v") {
        return "cacheblend";
    }
    if (algorithm == "delta-a") {
        return "kvshare";
    }
    return algorithm;
}

std::string traceExecutionModeForAlgorithm(const std::string& algorithm) {
    if (algorithm == "full-reuse") {
        return "native-full-reuse";
    }
    if (algorithm == "full-compute") {
        return "native-full-compute";
    }
    if (algorithm == "epic") {
        return "native-epic-sparse-recompute";
    }
    if (algorithm == "cacheblend" || algorithm == "delta-v") {
        return "native-cacheblend-sparse-recompute";
    }
    if (algorithm == "kvshare" || algorithm == "delta-a") {
        return "native-kvshare-sparse-recompute";
    }
    if (algorithm == "explicit") {
        return "native-explicit-sparse-recompute";
    }
    return algorithm;
}

double traceBudgetRatio(const std::string& algorithm, double requestedRatio) {
    if (algorithm == "full-reuse") {
        return 0.0;
    }
    if (algorithm == "full-compute") {
        return 1.0;
    }
    return requestedRatio;
}

int64_t traceRecomputeBudgetTokens(const std::string& algorithm, int picTokenCount, double requestedRatio) {
    if (algorithm == "full-reuse") {
        return 0;
    }
    if (algorithm == "full-compute") {
        return picTokenCount;
    }
    if (picTokenCount <= 0 || requestedRatio <= 0.0) {
        return 0;
    }
    return std::min<int64_t>(
        picTokenCount, std::max<int64_t>(1, static_cast<int64_t>(std::ceil(picTokenCount * requestedRatio))));
}

std::string sanitizeCachePart(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (unsigned char ch : value) {
        if (std::isalnum(ch) || ch == '.' || ch == '_' || ch == '-') {
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('_');
        }
    }
    while (!out.empty() && (out.front() == '.' || out.front() == '_')) {
        out.erase(out.begin());
    }
    while (!out.empty() && (out.back() == '.' || out.back() == '_')) {
        out.pop_back();
    }
    return out.empty() ? "cache" : out;
}

std::string jsonEscape(const std::string& value) {
    return json(value).dump();
}

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

uint32_t sha256RotateRight(uint32_t value, int bits) {
    return (value >> bits) | (value << (32 - bits));
}

std::string sha256Hex(const std::string& value) {
    static constexpr std::array<uint32_t, 64> k = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
        0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
        0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
        0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
        0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
        0xc67178f2u,
    };
    std::array<uint32_t, 8> h = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };

    std::vector<uint8_t> data(value.begin(), value.end());
    const uint64_t bitLen = static_cast<uint64_t>(data.size()) * 8u;
    data.push_back(0x80u);
    while ((data.size() % 64) != 56) {
        data.push_back(0u);
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        data.push_back(static_cast<uint8_t>((bitLen >> shift) & 0xffu));
    }

    for (size_t offset = 0; offset < data.size(); offset += 64) {
        std::array<uint32_t, 64> w = {};
        for (int i = 0; i < 16; ++i) {
            const size_t j = offset + static_cast<size_t>(i) * 4;
            w[i] = (static_cast<uint32_t>(data[j]) << 24) |
                   (static_cast<uint32_t>(data[j + 1]) << 16) |
                   (static_cast<uint32_t>(data[j + 2]) << 8) |
                   static_cast<uint32_t>(data[j + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = sha256RotateRight(w[i - 15], 7) ^ sha256RotateRight(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = sha256RotateRight(w[i - 2], 17) ^ sha256RotateRight(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = h[0];
        uint32_t b = h[1];
        uint32_t c = h[2];
        uint32_t d = h[3];
        uint32_t e = h[4];
        uint32_t f = h[5];
        uint32_t g = h[6];
        uint32_t hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t s1 = sha256RotateRight(e, 6) ^ sha256RotateRight(e, 11) ^ sha256RotateRight(e, 25);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = hh + s1 + ch + k[i] + w[i];
            const uint32_t s0 = sha256RotateRight(a, 2) ^ sha256RotateRight(a, 13) ^ sha256RotateRight(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + maj;
            hh = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }

    std::ostringstream os;
    os << std::hex << std::setfill('0');
    for (uint32_t part : h) {
        os << std::setw(8) << part;
    }
    return os.str();
}

std::string tokenIdsDigestInput(const std::vector<int>& tokenIds) {
    std::ostringstream os;
    for (size_t i = 0; i < tokenIds.size(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << tokenIds[i];
    }
    return os.str();
}

bool looksLikeFilePath(const std::string& value) {
    std::string candidate = value;
    candidate.erase(candidate.begin(), std::find_if(candidate.begin(), candidate.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    candidate.erase(std::find_if(candidate.rbegin(), candidate.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), candidate.end());
    if (candidate.empty() || candidate.find('\n') != std::string::npos || candidate.find('\r') != std::string::npos) {
        return false;
    }
    if (candidate.size() > 255) {
        return false;
    }
    if (candidate.rfind("/", 0) == 0 || candidate.rfind("./", 0) == 0 || candidate.rfind("../", 0) == 0 ||
        candidate.rfind("~/", 0) == 0 || candidate.rfind("file://", 0) == 0) {
        return true;
    }
    if (candidate.size() >= 3 && std::isalpha(static_cast<unsigned char>(candidate[0])) && candidate[1] == ':' &&
        (candidate[2] == '\\' || candidate[2] == '/')) {
        return true;
    }
    if (candidate.find('\\') != std::string::npos) {
        return true;
    }
    std::error_code ec;
    if (fs::exists(fs::path(candidate), ec)) {
        return true;
    }
    return candidate.find('/') != std::string::npos && fs::path(candidate).has_extension();
}

bool writeJsonFile(const fs::path& path, const json& value) {
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    std::ofstream os(path, std::ios::binary);
    if (!os.good()) {
        return false;
    }
    os << value.dump(2) << "\n";
    return os.good();
}

json readJsonFile(const fs::path& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is.good()) {
        return json();
    }
    std::stringstream buffer;
    buffer << is.rdbuf();
    return json::parse(buffer.str(), nullptr, false);
}

std::string absoluteString(const fs::path& path) {
    std::error_code ec;
    auto abs = fs::absolute(path, ec);
    return (ec ? path : abs).lexically_normal().string();
}

bool fileExistsNonEmpty(const fs::path& path) {
    std::error_code ec;
    return fs::exists(path, ec) && fs::is_regular_file(path, ec) && fs::file_size(path, ec) > 0;
}

size_t fileSizeOrZero(const fs::path& path) {
    std::error_code ec;
    auto size = fs::file_size(path, ec);
    return ec ? 0 : static_cast<size_t>(size);
}

struct KvShape {
    int batch = 1;
    int kvHeads = 0;
    int headDim = 0;
    int dtypeBytes = 0;
    int ropeDim = 0;
    double ropeTheta = 10000.0;
    std::string keyRopeState = "canonical_no_rope";
    std::string ropePairing = "half";
    std::string ropeType = "default";
    double ropeScalingFactor = 1.0;
    double ropeScalingLowFreqFactor = 1.0;
    double ropeScalingHighFreqFactor = 4.0;
    int ropeScalingOriginalMaxPositionEmbeddings = 0;
    int maxPositionEmbeddings = 0;
    double ropeAttentionScaling = 1.0;
};

struct TextCacheRef {
    std::string id;
    std::string cacheName;
    std::string metaPath;
};

struct PreparedTextCache {
    std::string id;
    std::string cacheName;
    std::string metaPath;
    std::vector<int> tokenIds;
    json source = json::object();
    json kvLayout = json::object();
    json kv = json::array();
    MNN::PagedKVExternalSegment segment;
};

struct ExplicitCacheTokenSpan {
    int promptStart = -1;
    int sourceStart = 0;
    int tokenCount = 0;
};

struct PreparedPicCache {
    std::string id;
    std::string cacheName;
    std::string placeholder = kDefaultPicPlaceholder;
    std::string selectionAlgorithm = "full-reuse";
    double recomputeRatio = 0.20;
    int scoreLayerIdx = 1;
    std::vector<int> explicitLogicalIndices;
    std::vector<int> explicitPicLocalIndices;
    std::vector<int> fullPromptTokenIds;
    std::vector<ExplicitCacheTokenSpan> promptSpans;
    std::vector<int> tokenIds;
    json textCaches = json::array();
    std::vector<MNN::PagedKVExternalSegment> segments;
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
};

KvShape kvShapeFromSegment(const MNN::PagedKVExternalSegment& segment) {
    KvShape shape;
    shape.batch = segment.batch;
    shape.kvHeads = segment.kvHeads;
    shape.headDim = segment.headDim;
    shape.dtypeBytes = segment.dtypeBytes;
    shape.ropeDim = segment.ropeDim;
    shape.ropeTheta = segment.ropeTheta;
    shape.ropeType = segment.ropeType;
    shape.ropeScalingFactor = segment.ropeScalingFactor;
    shape.ropeScalingLowFreqFactor = segment.ropeScalingLowFreqFactor;
    shape.ropeScalingHighFreqFactor = segment.ropeScalingHighFreqFactor;
    shape.ropeScalingOriginalMaxPositionEmbeddings = segment.ropeScalingOriginalMaxPositionEmbeddings;
    shape.maxPositionEmbeddings = segment.maxPositionEmbeddings;
    shape.ropeAttentionScaling = segment.ropeAttentionScaling;
    shape.keyRopeState = segment.keyRopeState;
    shape.ropePairing = segment.ropePairing;
    return shape;
}

KvShape kvShapeFromFirstSegment(const std::vector<MNN::PagedKVExternalSegment>& segments) {
    return segments.empty() ? KvShape() : kvShapeFromSegment(segments.front());
}

std::string pathJoinForMNN(const std::string& lhs, const std::string& rhs) {
    if (lhs.empty()) {
        return rhs;
    }
    if (lhs.back() == '/' || lhs.back() == '\\') {
        return lhs + rhs;
    }
    return lhs + "/" + rhs;
}

int jsonInt(const json& j, const std::string& key, int fallback = 0) {
    if (j.contains(key) && j[key].is_number_integer()) {
        return j[key].get<int>();
    }
    if (j.contains(key) && j[key].is_array() && !j[key].empty() && j[key][0].is_number_integer()) {
        return j[key][0].get<int>();
    }
    return fallback;
}

double jsonDouble(const json& j, const std::string& key, double fallback = 0.0) {
    if (j.contains(key) && j[key].is_number()) {
        return j[key].get<double>();
    }
    return fallback;
}

std::string jsonString(const json& j, const std::string& key, const std::string& fallback = "") {
    if (j.contains(key) && j[key].is_string()) {
        return j[key].get<std::string>();
    }
    return fallback;
}

bool jsonBool(const json& j, const std::string& key, bool fallback = false) {
    if (j.contains(key) && j[key].is_boolean()) {
        return j[key].get<bool>();
    }
    return fallback;
}

double clampDouble(double value, double low, double high) {
    return std::max(low, std::min(high, value));
}

std::vector<int> jsonIntVector(const json& value) {
    std::vector<int> out;
    if (!value.is_array()) {
        return out;
    }
    for (const auto& item : value) {
        if (item.is_number_integer()) {
            out.emplace_back(item.get<int>());
        }
    }
    return out;
}

bool jsonIntAlias(const json& j, const std::vector<const char*>& keys, int& value) {
    for (const auto* key : keys) {
        if (j.contains(key) && j[key].is_number_integer()) {
            value = j[key].get<int>();
            return true;
        }
    }
    return false;
}

std::vector<int> jsonIntVectorAlias(const json& j, const std::vector<const char*>& keys) {
    for (const auto* key : keys) {
        if (j.contains(key)) {
            return jsonIntVector(j[key]);
        }
    }
    return {};
}

bool parseExplicitCacheTokenSpans(const json& request, std::vector<ExplicitCacheTokenSpan>& spans,
                                  std::string& error) {
    spans.clear();
    const json* spanArray = nullptr;
    for (const auto* key : {"doc_cache_spans", "pic_token_spans", "cache_token_spans"}) {
        if (request.contains(key)) {
            spanArray = &request[key];
            break;
        }
    }
    if (spanArray != nullptr) {
        if (!spanArray->is_array() || spanArray->empty()) {
            error = "doc_cache_spans/pic_token_spans must be a non-empty array";
            return false;
        }
        for (const auto& item : *spanArray) {
            if (!item.is_object()) {
                error = "Each doc_cache_spans item must be an object";
                return false;
            }
            ExplicitCacheTokenSpan span;
            if (!jsonIntAlias(item, {"prompt_start", "pic_token_start", "doc_token_start", "logical_start"},
                              span.promptStart)) {
                error = "Each doc cache span needs prompt_start";
                return false;
            }
            jsonIntAlias(item, {"source_start", "source_token_start", "cache_token_start",
                                "pic_source_token_start"},
                         span.sourceStart);
            if (!jsonIntAlias(item, {"token_count", "pic_token_count", "doc_token_count", "count"},
                              span.tokenCount)) {
                error = "Each doc cache span needs token_count";
                return false;
            }
            spans.emplace_back(span);
        }
        return true;
    }

    auto indices = jsonIntVectorAlias(request, {"doc_token_indices", "pic_token_indices"});
    if (!indices.empty()) {
        std::sort(indices.begin(), indices.end());
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
        for (size_t i = 1; i < indices.size(); ++i) {
            if (indices[i] != indices[i - 1] + 1) {
                error = "doc_token_indices/pic_token_indices must be one contiguous span";
                return false;
            }
        }
        ExplicitCacheTokenSpan span;
        span.promptStart = indices.front();
        span.tokenCount = static_cast<int>(indices.size());
        jsonIntAlias(request, {"source_start", "source_token_start", "cache_token_start", "pic_source_token_start"},
                     span.sourceStart);
        spans.emplace_back(span);
        return true;
    }

    ExplicitCacheTokenSpan span;
    const bool hasStart =
        jsonIntAlias(request, {"prompt_start", "pic_token_start", "doc_token_start", "logical_start"},
                     span.promptStart);
    jsonIntAlias(request, {"source_start", "source_token_start", "cache_token_start", "pic_source_token_start"},
                 span.sourceStart);
    const bool hasCount =
        jsonIntAlias(request, {"token_count", "pic_token_count", "doc_token_count", "count"}, span.tokenCount);
    if (hasStart || hasCount) {
        if (!hasStart || !hasCount) {
            error = "Explicit doc cache token mapping needs both prompt_start and token_count";
            return false;
        }
        spans.emplace_back(span);
    }
    return true;
}

bool isSupportedPicAlgorithm(const std::string& algorithm) {
    return algorithm == "full-reuse" || algorithm == "full-compute" || algorithm == "cacheblend" ||
           algorithm == "epic" || algorithm == "kvshare" || algorithm == "delta-v" ||
           algorithm == "delta-a" || algorithm == "explicit";
}

std::string textCacheNameFromId(const std::string& id) {
    return "doc_" + sanitizeCachePart(id);
}

std::string compositeCacheName(const std::vector<TextCacheRef>& refs) {
    std::vector<std::string> parts;
    for (const auto& ref : refs) {
        if (!ref.id.empty()) {
            parts.emplace_back(sanitizeCachePart(ref.id));
        } else if (!ref.cacheName.empty()) {
            parts.emplace_back(sanitizeCachePart(ref.cacheName));
        } else {
            parts.emplace_back("text_cache");
        }
    }
    std::ostringstream os;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) {
            os << "_";
        }
        os << parts[i];
    }
    return os.str().empty() ? "pic_cache" : os.str();
}

bool parseTextCacheRefs(const json& request, std::vector<TextCacheRef>& refs, std::string& error) {
    json source = json::array();
    if (request.contains("text_cache_refs")) {
        source = request["text_cache_refs"];
    } else if (request.contains("text_caches")) {
        source = request["text_caches"];
    }
    if (!source.is_array() || source.empty()) {
        error = "pic_cache requires non-empty `text_cache_refs`";
        return false;
    }
    for (const auto& item : source) {
        if (!item.is_object()) {
            error = "Each text cache ref must be an object";
            return false;
        }
        if (item.value("type", "text") != "text") {
            error = "Text cache refs only support type=\"text\"";
            return false;
        }
        if (item.contains("content") || item.contains("path")) {
            error = "Text cache ref only accepts id/cache_name/meta_path; use /v1/prefill/text for content";
            return false;
        }
        TextCacheRef ref;
        ref.id = jsonString(item, "id");
        ref.cacheName = jsonString(item, "cache_name");
        ref.metaPath = jsonString(item, "meta_path");
        if (ref.id.empty() && ref.cacheName.empty() && ref.metaPath.empty()) {
            error = "Text cache ref needs id, cache_name, or meta_path";
            return false;
        }
        refs.emplace_back(std::move(ref));
    }
    return true;
}

std::string finishReasonForStatus(const MNN::Transformer::LlmContext* context, int maxTokens) {
    if (context == nullptr) {
        return "stop";
    }
    if (context->status == MNN::Transformer::LlmStatus::MAX_TOKENS_FINISHED ||
        (maxTokens > 0 && static_cast<int>(context->output_tokens.size()) >= maxTokens)) {
        return "length";
    }
    return "stop";
}

int64_t unixSecondsNow() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string trimStopStrings(std::string text, const json& stop) {
    std::vector<std::string> stops;
    if (stop.is_string()) {
        stops.emplace_back(stop.get<std::string>());
    } else if (stop.is_array()) {
        for (const auto& item : stop) {
            if (item.is_string()) {
                stops.emplace_back(item.get<std::string>());
            }
        }
    }
    for (const auto& item : stops) {
        if (item.empty()) {
            continue;
        }
        auto pos = text.find(item);
        if (pos != std::string::npos) {
            text.resize(pos);
        }
    }
    return text;
}

bool normalizeChatCompletionBatchRequest(
        const json& payload,
        std::vector<json>& requests,
        std::string& error) {
    requests.clear();
    bool explicitBatch = false;
    if (payload.is_array()) {
        explicitBatch = true;
        for (const auto& item : payload) {
            requests.emplace_back(item);
        }
    } else if (payload.is_object()) {
        const char* aliases[] = {"requests", "batch", "items", "inputs"};
        for (const char* alias : aliases) {
            if (!payload.contains(alias)) {
                continue;
            }
            explicitBatch = true;
            if (!payload[alias].is_array()) {
                error = std::string("Chat batch field `") + alias + "` must be an array";
                return false;
            }
            for (const auto& item : payload[alias]) {
                requests.emplace_back(item);
            }
            break;
        }
        if (!explicitBatch &&
            (payload.contains("messages") || payload.contains("full_prompt_token_ids") ||
             payload.contains("prompt_token_ids") || payload.contains("input_token_ids") ||
             payload.contains("pic_cache"))) {
            requests.emplace_back(payload);
        }
    }
    if (requests.empty()) {
        error = "ChatCompletionBatchRequest requires non-empty `requests` or a single request with `messages`";
        return false;
    }
    for (const auto& item : requests) {
        if (!item.is_object()) {
            error = "Each chat batch item must be a JSON object";
            return false;
        }
    }
    return true;
}

bool validateHomogeneousChatBatchRequest(const std::vector<json>& requests, std::string& error) {
    bool sawPic = false;
    bool sawFullCompute = false;
    for (const auto& item : requests) {
        const bool hasPic = item.contains("pic_cache") && !item["pic_cache"].is_null();
        sawPic = sawPic || hasPic;
        sawFullCompute = sawFullCompute || !hasPic;
    }
    if (sawPic && sawFullCompute) {
        error = "mnn_pic_server chat batch must be homogeneous: either every request has pic_cache "
                "or no request has pic_cache. Mixed PIC/full-compute batches are not supported.";
        return false;
    }
    return true;
}

void writeJson(httplib::Response& res, const json& body, int status = 200, int indent = 2) {
    res.status = status;
    res.set_content(body.dump(indent), "application/json");
}

void writeJsonError(httplib::Response& res, int status, const std::string& error) {
    writeJson(res, json({{"error", error}}), status, -1);
}

bool parseJsonBody(const httplib::Request& req, httplib::Response& res, json& body) {
    if (!json::accept(req.body)) {
        writeJsonError(res, 400, "Invalid JSON in request body");
        return false;
    }
    body = json::parse(req.body, nullptr, false);
    return true;
}

template <typename Handler>
void runLockedJsonEndpoint(const httplib::Request& req, httplib::Response& res, std::mutex& mutex,
                           Handler&& handler, int errorStatus = 400) {
    json request;
    if (!parseJsonBody(req, res, request)) {
        return;
    }

    json response;
    std::string error;
    bool ok = false;
    {
        std::lock_guard<std::mutex> lock(mutex);
        ok = handler(request, response, error);
    }
    if (!ok) {
        writeJsonError(res, errorStatus, error);
        return;
    }
    writeJson(res, response);
}

KvShape readKvShape(const fs::path& layersDir, const std::string& cacheName) {
    KvShape shape;
    auto sidecar = readJsonFile(layersDir / (cacheName + "_0.json"));
    if (sidecar.is_object()) {
        shape.batch = jsonInt(sidecar, "batch", shape.batch);
        shape.kvHeads = jsonInt(sidecar, "kv_heads", shape.kvHeads);
        shape.headDim = jsonInt(sidecar, "head_dim", shape.headDim);
        shape.dtypeBytes = jsonInt(sidecar, "dtype_bytes", shape.dtypeBytes);
        shape.ropeDim = jsonInt(sidecar, "rope_dim", shape.ropeDim);
        shape.ropeTheta = jsonDouble(sidecar, "rope_theta", shape.ropeTheta);
        shape.keyRopeState = jsonString(sidecar, "key_rope_state", shape.keyRopeState);
        shape.ropePairing = jsonString(sidecar, "rope_pairing", shape.ropePairing);
        shape.ropeType = jsonString(sidecar, "rope_type", shape.ropeType);
        shape.ropeScalingFactor = jsonDouble(sidecar, "rope_scaling_factor", shape.ropeScalingFactor);
        shape.ropeScalingLowFreqFactor =
            jsonDouble(sidecar, "rope_scaling_low_freq_factor", shape.ropeScalingLowFreqFactor);
        shape.ropeScalingHighFreqFactor =
            jsonDouble(sidecar, "rope_scaling_high_freq_factor", shape.ropeScalingHighFreqFactor);
        shape.ropeScalingOriginalMaxPositionEmbeddings = jsonInt(
            sidecar, "rope_scaling_original_max_position_embeddings",
            shape.ropeScalingOriginalMaxPositionEmbeddings);
        shape.maxPositionEmbeddings = jsonInt(sidecar, "max_position_embeddings", shape.maxPositionEmbeddings);
        shape.ropeAttentionScaling = jsonDouble(sidecar, "rope_attention_scaling", shape.ropeAttentionScaling);
    }
    return shape;
}

int countLayerFiles(const fs::path& layersDir, const std::string& cacheName) {
    int count = 0;
    for (;;) {
        auto keyPath = layersDir / (cacheName + "_" + std::to_string(count) + ".k");
        auto valuePath = layersDir / (cacheName + "_" + std::to_string(count) + ".v");
        if (!fileExistsNonEmpty(keyPath) || !fileExistsNonEmpty(valuePath)) {
            break;
        }
        ++count;
    }
    return count;
}

bool existingCacheMatches(const fs::path& metaPath, const std::string& contentSha256,
                          const std::vector<int>& tokenIds) {
    auto meta = readJsonFile(metaPath);
    if (!meta.is_object()) {
        return false;
    }
    if (meta.value("format", "") != kMetaFormat) {
        return false;
    }
    auto source = meta.value("source", json::object());
    const std::string storedSourceSha =
        source.is_object() ? source.value("token_ids_sha256", source.value("content_sha256", "")) : "";
    if (!source.is_object() || storedSourceSha != contentSha256) {
        return false;
    }
    auto kvLayout = meta.value("kv_layout", json::object());
    if (!kvLayout.is_object() || kvLayout.value("layout", "") != "mnn_paged_attention_raw_v1" ||
        kvLayout.value("key_rope_state", "") != "canonical_no_rope" ||
        jsonInt(kvLayout, "kv_heads", 0) <= 0 || jsonInt(kvLayout, "head_dim", 0) <= 0 ||
        !kvLayout.contains("rope_attention_scaling")) {
        return false;
    }
    if (!meta.contains("token_ids") || !meta["token_ids"].is_array()) {
        return false;
    }
    if (meta["token_ids"].size() != tokenIds.size()) {
        return false;
    }
    for (size_t i = 0; i < tokenIds.size(); ++i) {
        if (!meta["token_ids"][i].is_number_integer() || meta["token_ids"][i].get<int>() != tokenIds[i]) {
            return false;
        }
    }
    if (!meta.contains("kv") || !meta["kv"].is_array()) {
        return false;
    }
    for (const auto& entry : meta["kv"]) {
        if (!entry.is_object()) {
            return false;
        }
        auto keyPath = entry.value("key_path", "");
        auto valuePath = entry.value("value_path", "");
        if (keyPath.empty() || valuePath.empty() || !fileExistsNonEmpty(keyPath) || !fileExistsNonEmpty(valuePath)) {
            return false;
        }
    }
    return true;
}

json makeKvLayout(const json& config, int tokenCount, const KvShape& shape) {
    int hidden = jsonInt(config, "hidden_size");
    int heads = jsonInt(config, "num_attention_heads");
    int kvHeads = shape.kvHeads > 0 ? shape.kvHeads : jsonInt(config, "num_key_value_heads", heads);
    int headDim = shape.headDim > 0 ? shape.headDim : jsonInt(config, "head_dim", (heads > 0 ? hidden / heads : 0));
    int batch = std::max(1, shape.batch);
    double ropeTheta = shape.ropeTheta > 0.0 ? shape.ropeTheta : jsonDouble(config, "rope_theta", 10000.0);
    int ropeDim = shape.ropeDim > 0 ? shape.ropeDim : headDim;
    std::string precision = jsonString(config, "precision", "low");
    std::string dtype = shape.dtypeBytes == 4 || precision == "normal" || precision == "high" ? "float32" : "float16";
    return {
        {"backend", jsonString(config, "backend_type", "unknown")},
        {"layout", "mnn_paged_attention_raw_v1"},
        {"dtype", dtype},
        {"token_count", tokenCount},
        {"batch", batch},
        {"head_dim", headDim},
        {"kv_heads", kvHeads},
        {"key_layout", "[token,batch,kv_head,head_dim]"},
        {"value_layout", "[batch,kv_head,token,head_dim]"},
        {"key_shape", json::array({tokenCount, batch, kvHeads, headDim})},
        {"value_shape", json::array({batch, kvHeads, tokenCount, headDim})},
        {"key_rope_state", shape.keyRopeState},
        {"rope_pairing", shape.ropePairing},
        {"rope_theta", ropeTheta},
        {"rope_dim", ropeDim},
        {"rope_type", shape.ropeType},
        {"rope_scaling_factor", shape.ropeScalingFactor},
        {"rope_scaling_low_freq_factor", shape.ropeScalingLowFreqFactor},
        {"rope_scaling_high_freq_factor", shape.ropeScalingHighFreqFactor},
        {"rope_scaling_original_max_position_embeddings", shape.ropeScalingOriginalMaxPositionEmbeddings},
        {"max_position_embeddings", shape.maxPositionEmbeddings},
        {"rope_attention_scaling", shape.ropeAttentionScaling},
        {"source_position_base", 0},
        {"page_size", 0},
    };
}

fs::path resolveTextCacheMetaPath(const std::string& kvCacheDir, const std::string& backend,
                                  const TextCacheRef& ref) {
    if (!ref.metaPath.empty()) {
        return fs::path(ref.metaPath);
    }
    std::string cacheName = ref.cacheName.empty() ? textCacheNameFromId(ref.id) : sanitizeCachePart(ref.cacheName);
    return fs::path(kvCacheDir) / "objects" / backend / cacheName / "meta.json";
}

bool loadPreparedTextCache(const std::string& kvCacheDir, const std::string& backend, const TextCacheRef& ref,
                           PreparedTextCache& prepared, std::string& error) {
    auto metaPath = resolveTextCacheMetaPath(kvCacheDir, backend, ref);
    auto meta = readJsonFile(metaPath);
    if (!meta.is_object()) {
        error = "Failed to read text cache meta: " + metaPath.string();
        return false;
    }
    if (meta.value("format", "") != kMetaFormat || meta.value("type", "text") != "text") {
        error = "Unsupported text cache meta format/type: " + metaPath.string();
        return false;
    }
    auto kvLayout = meta.value("kv_layout", json::object());
    if (!kvLayout.is_object() || kvLayout.value("layout", "") != "mnn_paged_attention_raw_v1" ||
        kvLayout.value("key_rope_state", "") != "canonical_no_rope" ||
        kvLayout.value("rope_pairing", "") != "half") {
        error = "Text cache is not mnn_paged_attention_raw_v1 canonical_no_rope: " + metaPath.string();
        return false;
    }
    if (!kvLayout.contains("rope_attention_scaling")) {
        error = "Text cache metadata is missing rope_attention_scaling; rebuild: " + metaPath.string();
        return false;
    }
    if (kvLayout.value("backend", "") != backend && meta.value("backend", "") != backend) {
        error = "Text cache backend does not match current runtime: " + metaPath.string();
        return false;
    }
    if (!meta.contains("token_ids") || !meta["token_ids"].is_array() || meta["token_ids"].empty()) {
        error = "Text cache token_ids is empty: " + metaPath.string();
        return false;
    }
    prepared.id = meta.value("id", meta.value("cache_name", ""));
    prepared.cacheName = meta.value("cache_name", prepared.id);
    prepared.metaPath = absoluteString(metaPath);
    prepared.source = meta.value("source", json::object());
    prepared.kvLayout = kvLayout;
    prepared.kv = meta.value("kv", json::array());
    for (const auto& item : meta["token_ids"]) {
        if (!item.is_number_integer()) {
            error = "Text cache token_ids must be integers: " + metaPath.string();
            return false;
        }
        prepared.tokenIds.emplace_back(item.get<int>());
    }
    if (static_cast<int>(prepared.tokenIds.size()) != meta.value("token_count", static_cast<int>(prepared.tokenIds.size()))) {
        error = "Text cache token_count does not match token_ids: " + metaPath.string();
        return false;
    }
    if (!prepared.kv.is_array() || prepared.kv.empty()) {
        error = "Text cache KV layer list is empty: " + metaPath.string();
        return false;
    }

    auto shapePath = jsonString(prepared.kv[0], "shape_path");
    KvShape shape;
    if (!shapePath.empty()) {
        auto sidecar = readJsonFile(shapePath);
        if (sidecar.is_object()) {
            shape.batch = jsonInt(sidecar, "batch", shape.batch);
            shape.kvHeads = jsonInt(sidecar, "kv_heads", shape.kvHeads);
            shape.headDim = jsonInt(sidecar, "head_dim", shape.headDim);
            shape.dtypeBytes = jsonInt(sidecar, "dtype_bytes", shape.dtypeBytes);
            shape.ropeDim = jsonInt(sidecar, "rope_dim", shape.ropeDim);
            shape.ropeTheta = jsonDouble(sidecar, "rope_theta", shape.ropeTheta);
            shape.keyRopeState = jsonString(sidecar, "key_rope_state", shape.keyRopeState);
            shape.ropePairing = jsonString(sidecar, "rope_pairing", shape.ropePairing);
            shape.ropeType = jsonString(sidecar, "rope_type", shape.ropeType);
            shape.ropeScalingFactor = jsonDouble(sidecar, "rope_scaling_factor", shape.ropeScalingFactor);
            shape.ropeScalingLowFreqFactor =
                jsonDouble(sidecar, "rope_scaling_low_freq_factor", shape.ropeScalingLowFreqFactor);
            shape.ropeScalingHighFreqFactor =
                jsonDouble(sidecar, "rope_scaling_high_freq_factor", shape.ropeScalingHighFreqFactor);
            shape.ropeScalingOriginalMaxPositionEmbeddings =
                jsonInt(sidecar, "rope_scaling_original_max_position_embeddings",
                        shape.ropeScalingOriginalMaxPositionEmbeddings);
            shape.maxPositionEmbeddings = jsonInt(sidecar, "max_position_embeddings", shape.maxPositionEmbeddings);
            shape.ropeAttentionScaling = jsonDouble(sidecar, "rope_attention_scaling", shape.ropeAttentionScaling);
        }
    }
    if (shape.kvHeads <= 0) {
        shape.kvHeads = jsonInt(kvLayout, "kv_heads", 0);
    }
    if (shape.headDim <= 0) {
        shape.headDim = jsonInt(kvLayout, "head_dim", 0);
    }
    if (shape.dtypeBytes <= 0) {
        std::string dtype = kvLayout.value("dtype", "float16");
        shape.dtypeBytes = dtype == "float32" ? 4 : 2;
    }
    if (shape.ropeDim <= 0) {
        shape.ropeDim = jsonInt(kvLayout, "rope_dim", shape.headDim);
    }
    shape.ropeTheta = jsonDouble(kvLayout, "rope_theta", shape.ropeTheta);
    shape.ropeType = jsonString(kvLayout, "rope_type", shape.ropeType);
    shape.ropeScalingFactor = jsonDouble(kvLayout, "rope_scaling_factor", shape.ropeScalingFactor);
    shape.ropeScalingLowFreqFactor =
        jsonDouble(kvLayout, "rope_scaling_low_freq_factor", shape.ropeScalingLowFreqFactor);
    shape.ropeScalingHighFreqFactor =
        jsonDouble(kvLayout, "rope_scaling_high_freq_factor", shape.ropeScalingHighFreqFactor);
    shape.ropeScalingOriginalMaxPositionEmbeddings =
        jsonInt(kvLayout, "rope_scaling_original_max_position_embeddings",
                shape.ropeScalingOriginalMaxPositionEmbeddings);
    shape.maxPositionEmbeddings = jsonInt(kvLayout, "max_position_embeddings", shape.maxPositionEmbeddings);
    shape.ropeAttentionScaling = jsonDouble(kvLayout, "rope_attention_scaling", shape.ropeAttentionScaling);
    if (shape.kvHeads <= 0 || shape.headDim <= 0 || shape.dtypeBytes <= 0) {
        error = "Text cache KV shape is incomplete: " + metaPath.string();
        return false;
    }

    auto& segment = prepared.segment;
    segment.tokenCount = prepared.tokenIds.size();
    segment.sourceTokenCount = prepared.tokenIds.size();
    segment.cacheName = prepared.cacheName;
    segment.batch = shape.batch;
    segment.kvHeads = shape.kvHeads;
    segment.headDim = shape.headDim;
    segment.dtypeBytes = shape.dtypeBytes;
    segment.ropeDim = shape.ropeDim;
    segment.ropeTheta = static_cast<float>(shape.ropeTheta);
    segment.ropeScalingFactor = static_cast<float>(shape.ropeScalingFactor);
    segment.ropeScalingLowFreqFactor = static_cast<float>(shape.ropeScalingLowFreqFactor);
    segment.ropeScalingHighFreqFactor = static_cast<float>(shape.ropeScalingHighFreqFactor);
    segment.ropeScalingOriginalMaxPositionEmbeddings = shape.ropeScalingOriginalMaxPositionEmbeddings;
    segment.maxPositionEmbeddings = shape.maxPositionEmbeddings;
    segment.ropeAttentionScaling = static_cast<float>(shape.ropeAttentionScaling);
    segment.ropeType = shape.ropeType;
    segment.keyRopeState = shape.keyRopeState;
    segment.ropePairing = shape.ropePairing;
    for (const auto& entry : prepared.kv) {
        if (!entry.is_object()) {
            error = "Text cache KV entry must be an object: " + metaPath.string();
            return false;
        }
        auto keyPath = entry.value("key_path", "");
        auto valuePath = entry.value("value_path", "");
        if (keyPath.empty() || valuePath.empty() || !fileExistsNonEmpty(keyPath) || !fileExistsNonEmpty(valuePath)) {
            error = "Text cache KV file missing: " + metaPath.string();
            return false;
        }
        MNN::PagedKVExternalSegment::LayerFile layer;
        layer.layerIndex = jsonInt(entry, "layer_index", static_cast<int>(segment.layers.size()));
        layer.keyPath = keyPath;
        layer.valuePath = valuePath;
        segment.layers.emplace_back(std::move(layer));
    }
    return true;
}

bool preparePicCacheFromRequest(const std::string& kvCacheDir, const std::string& backend, const json& request,
                                PreparedPicCache& prepared, std::string& error) {
    if (!request.is_object()) {
        error = "PIC cache request must be a JSON object";
        return false;
    }
    if (request.contains("merge_mode") || request.contains("segments")) {
        error = "Legacy pic_cache fields are not supported; use text_cache_refs";
        return false;
    }
    prepared.placeholder = jsonString(request, "placeholder", kDefaultPicPlaceholder);
    prepared.selectionAlgorithm = jsonString(request, "selection_algorithm", "full-reuse");
    if (!isSupportedPicAlgorithm(prepared.selectionAlgorithm)) {
        error = "Unsupported selection_algorithm: " + prepared.selectionAlgorithm;
        return false;
    }
    prepared.recomputeRatio = clampDouble(jsonDouble(request, "pic_recompute_ratio", 0.20), 0.0, 1.0);
    const int defaultScoreLayer = 1;
    prepared.scoreLayerIdx = std::max(0, jsonInt(request, "pic_recompute_score_layer_idx", defaultScoreLayer));
    prepared.explicitLogicalIndices = jsonIntVector(request.value("pic_recompute_logical_indices", json::array()));
    prepared.explicitPicLocalIndices = jsonIntVector(request.value("pic_recompute_pic_local_indices", json::array()));
    prepared.fullPromptTokenIds = jsonIntVectorAlias(
        request, {"full_prompt_token_ids", "prompt_token_ids", "input_token_ids"});
    if (!parseExplicitCacheTokenSpans(request, prepared.promptSpans, error)) {
        return false;
    }

    std::vector<TextCacheRef> refs;
    if (!parseTextCacheRefs(request, refs, error)) {
        return false;
    }
    prepared.id = jsonString(request, "id", compositeCacheName(refs));
    prepared.cacheName = sanitizeCachePart("cmp_" + prepared.id);
    for (const auto& ref : refs) {
        PreparedTextCache text;
        if (!loadPreparedTextCache(kvCacheDir, backend, ref, text, error)) {
            return false;
        }
        prepared.tokenIds.insert(prepared.tokenIds.end(), text.tokenIds.begin(), text.tokenIds.end());
        prepared.segments.emplace_back(text.segment);
        prepared.textCaches.push_back({
            {"type", "text"},
            {"id", text.id},
            {"cache_name", text.cacheName},
            {"meta_path", text.metaPath},
            {"token_count", text.tokenIds.size()},
            {"kv_layout", text.kvLayout},
            {"kv", text.kv},
            {"source", text.source},
        });
    }
    if (prepared.tokenIds.empty()) {
        error = "Merged PIC cache has no tokens";
        return false;
    }
    return true;
}

std::vector<MNN::PagedKVExternalSegment> sliceExternalSegmentsRange(
    const std::vector<MNN::PagedKVExternalSegment>& segments, size_t sourceStart, size_t tokenCount) {
    std::vector<MNN::PagedKVExternalSegment> out;
    if (tokenCount == 0) {
        return out;
    }
    const size_t sourceEnd = sourceStart + tokenCount;
    size_t cursor = 0;
    for (const auto& segment : segments) {
        const size_t begin = cursor;
        const size_t end = cursor + segment.tokenCount;
        cursor = end;
        if (sourceEnd <= begin || sourceStart >= end) {
            continue;
        }
        const size_t overlapBegin = std::max(sourceStart, begin);
        const size_t overlapEnd = std::min(sourceEnd, end);
        auto sliced = segment;
        sliced.sourceTokenOffset += overlapBegin - begin;
        sliced.tokenCount = overlapEnd - overlapBegin;
        if (sliced.tokenCount > 0) {
            out.emplace_back(std::move(sliced));
        }
    }
    return out;
}

size_t externalSegmentTokenCount(const std::vector<MNN::PagedKVExternalSegment>& segments) {
    size_t count = 0;
    for (const auto& segment : segments) {
        count += segment.tokenCount;
    }
    return count;
}

bool applyExplicitPicPromptMapping(PreparedPicCache& pic,
                                   std::vector<int>& preludeTokenIds,
                                   std::vector<int>& suffixTokenIds,
                                   std::vector<int>& fullPromptTokenIds,
                                   json& tokenMapping,
                                   std::string& error) {
    if (pic.fullPromptTokenIds.empty()) {
        error = "PIC chat request must provide full_prompt_token_ids/prompt_token_ids; segmented placeholder tokenization is disabled";
        return false;
    }
    if (pic.promptSpans.empty()) {
        error = "PIC chat request must explicitly mark doc cache tokens with doc_cache_spans, pic_token_start/token_count, or pic_token_indices";
        return false;
    }
    auto spans = pic.promptSpans;
    std::sort(spans.begin(), spans.end(), [](const ExplicitCacheTokenSpan& lhs,
                                             const ExplicitCacheTokenSpan& rhs) {
        return lhs.promptStart < rhs.promptStart;
    });
    const auto sourceTokenIds = pic.tokenIds;
    const auto sourceSegments = pic.segments;
    const int fullTokenCount = static_cast<int>(pic.fullPromptTokenIds.size());
    const int sourceTokenCount = static_cast<int>(sourceTokenIds.size());
    if (fullTokenCount <= 0 || sourceTokenCount <= 0) {
        error = "Explicit PIC token mapping got empty full prompt or text cache tokens";
        return false;
    }

    const int picStart = spans.front().promptStart;
    int promptCursor = picStart;
    std::vector<int> mappedTokenIds;
    std::vector<MNN::PagedKVExternalSegment> mappedSegments;
    tokenMapping = json::array();
    for (const auto& span : spans) {
        if (span.promptStart < 0 || span.sourceStart < 0 || span.tokenCount <= 0) {
            error = "Doc cache span values must be non-negative and token_count must be positive";
            return false;
        }
        if (span.promptStart != promptCursor) {
            error = "Doc cache spans must form one contiguous prompt region for the current PIC graph path";
            return false;
        }
        if (span.promptStart + span.tokenCount > fullTokenCount) {
            error = "Doc cache span exceeds full_prompt_token_ids length";
            return false;
        }
        if (span.sourceStart + span.tokenCount > sourceTokenCount) {
            error = "Doc cache span exceeds persistent text cache token_ids length";
            return false;
        }
        for (int i = 0; i < span.tokenCount; ++i) {
            const int promptToken = pic.fullPromptTokenIds[span.promptStart + i];
            const int sourceToken = sourceTokenIds[span.sourceStart + i];
            if (promptToken != sourceToken) {
                error = "Doc cache span token mismatch at prompt index " +
                        std::to_string(span.promptStart + i) + ": prompt token " +
                        std::to_string(promptToken) + " != cache source token " +
                        std::to_string(sourceToken);
                return false;
            }
        }
        auto rangeSegments = sliceExternalSegmentsRange(
            sourceSegments, static_cast<size_t>(span.sourceStart), static_cast<size_t>(span.tokenCount));
        if (externalSegmentTokenCount(rangeSegments) != static_cast<size_t>(span.tokenCount)) {
            error = "Failed to map doc cache token span onto persistent KV segment files";
            return false;
        }
        mappedTokenIds.insert(mappedTokenIds.end(),
                              sourceTokenIds.begin() + span.sourceStart,
                              sourceTokenIds.begin() + span.sourceStart + span.tokenCount);
        mappedSegments.insert(mappedSegments.end(), rangeSegments.begin(), rangeSegments.end());
        tokenMapping.push_back({
            {"prompt_start", span.promptStart},
            {"source_start", span.sourceStart},
            {"token_count", span.tokenCount},
        });
        promptCursor += span.tokenCount;
    }

    preludeTokenIds.assign(pic.fullPromptTokenIds.begin(),
                           pic.fullPromptTokenIds.begin() + picStart);
    suffixTokenIds.assign(pic.fullPromptTokenIds.begin() + promptCursor,
                          pic.fullPromptTokenIds.end());
    fullPromptTokenIds = pic.fullPromptTokenIds;
    pic.tokenIds = std::move(mappedTokenIds);
    pic.segments = std::move(mappedSegments);
    return true;
}

PicExecutionPlan buildExecutionPlan(const PreparedPicCache& pic, int preludeTokenCount, int layerCount,
                                    const std::vector<int>* nativeSelectedLocalIndices = nullptr,
                                    const json* scoreMetadata = nullptr) {
    PicExecutionPlan plan;
    plan.selectionAlgorithm = pic.selectionAlgorithm;
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
    plan.plannerName = pic.selectionAlgorithm == "cacheblend" || pic.selectionAlgorithm == "delta-v"
        ? "CacheBlendPlanner"
        : (pic.selectionAlgorithm == "explicit" ? "ExplicitPlanner" : "KvsharePlanner");
    if (nativeSelectedLocalIndices != nullptr || pic.selectionAlgorithm == "explicit") {
        plan.executionMode = pic.selectionAlgorithm == "cacheblend" || pic.selectionAlgorithm == "delta-v"
            ? "native-cacheblend-sparse-recompute"
            : (pic.selectionAlgorithm == "explicit" ? "native-explicit-sparse-recompute"
                                                     : "native-kvshare-sparse-recompute");
        plan.scoreLayerIdx = effectiveScoreLayer;
        plan.externalTokenIds = pic.tokenIds;
        plan.externalSegments = pic.segments;
        std::vector<int> localIndices;
        if (pic.selectionAlgorithm == "explicit") {
            for (int index : pic.explicitPicLocalIndices) {
                if (index >= 0 && index < picLength) {
                    localIndices.emplace_back(index);
                }
            }
            for (int logical : pic.explicitLogicalIndices) {
                int local = logical - picStart;
                if (local >= 0 && local < picLength) {
                    localIndices.emplace_back(local);
                }
            }
            std::sort(localIndices.begin(), localIndices.end());
            localIndices.erase(std::unique(localIndices.begin(), localIndices.end()), localIndices.end());
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
        std::string scoreKind = pic.selectionAlgorithm == "cacheblend" || pic.selectionAlgorithm == "delta-v"
            ? "layer_value_delta_mean_abs"
            : (pic.selectionAlgorithm == "explicit" ? "explicit_request_indices"
                                                     : "key_value_delta_influence_proxy");
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
        if (pic.selectionAlgorithm == "kvshare" || pic.selectionAlgorithm == "delta-a") {
            plan.metadata["python_reference"] =
                "hf_pic_runtime kvshare uses attention_output gradient influence; MNN C++ currently uses a native K/V delta influence proxy because exported MNN modules do not expose score-layer query/attention-output autograd.";
        }
        return plan;
    }
    const std::string scoreKind = pic.selectionAlgorithm == "cacheblend" || pic.selectionAlgorithm == "delta-v"
        ? "layer_value_delta_mean_abs_gpu_topk_unavailable"
        : "attention_output_error_kv_first_order_influence_gpu_topk_unavailable";
    fillFullComputeFallback(
        plan.plannerName, scoreKind,
        "Native CUDA/OpenCL score + GPU top-k is not implemented; legacy scratch .k/.v CPU scoring is disabled");
    return plan;
}

json precisionRecoverySummary(const PicExecutionPlan& plan, double ratio, int requestedScoreLayerIdx) {
    return {
        {"selection_algorithm", plan.selectionAlgorithm},
        {"pic_recompute_ratio", ratio},
        {"pic_recompute_score_layer_idx", plan.scoreLayerIdx},
        {"requested_pic_recompute_score_layer_idx", requestedScoreLayerIdx},
        {"recompute_token_count", plan.recomputeTokenCount},
        {"recompute_logical_indices", plan.recomputeLogicalIndices},
        {"recompute_scores", plan.recomputeScores},
        {"execution_mode", plan.executionMode},
        {"fallback_reason", plan.fallbackReason},
        {"metadata", plan.metadata},
        {"decode_refine", {
            {"enabled", false},
            {"disabled_reason", "mnn_pic_server_prefill_only"},
            {"candidate_count", 0},
            {"refined_token_count", 0},
            {"refined_logical_indices", json::array()},
            {"steps", json::array()},
        }},
    };
}

} // namespace

PicServer::PicServer(PicServerConfig config) : mConfig(std::move(config)) {
}

bool PicServer::load() {
    mLlm.reset(MNN::Transformer::Llm::createLLM(mConfig.configPath));
    if (!mLlm) {
        std::cerr << "Failed to create LLM from " << mConfig.configPath << "\n";
        return false;
    }
    std::error_code ec;
    mConfig.kvCacheDir = absoluteString(mConfig.kvCacheDir);
    fs::create_directories(mConfig.kvCacheDir, ec);
    json runtimeConfig = {
        {"tmp_path", "tmp"},
        {"prefix_cache_path", mConfig.kvCacheDir},
    };
    const int pagedKvLimit = envInt("MNN_PIC_SERVER_PAGED_KV_MAX_TOKENS", 0);
    if (pagedKvLimit > 0) {
        runtimeConfig["paged_kv_max_tokens"] = pagedKvLimit;
        runtimeConfig["max_all_tokens"] = pagedKvLimit;
        std::cout << "PIC server test override: paged_kv_max_tokens=" << pagedKvLimit << "\n";
    }
    std::string config = runtimeConfig.dump();
    mLlm->set_config(config);
    if (!mLlm->load()) {
        std::cerr << "Failed to load LLM from " << mConfig.configPath << "\n";
        return false;
    }
    return true;
}

bool PicServer::start() {
    httplib::Server server;
    server.Options(".*", [this](const httplib::Request&, httplib::Response& res) {
        allowCors(res);
        res.status = 200;
    });
    server.Get("/", [this](const httplib::Request& req, httplib::Response& res) { handleRoot(req, res); });
    server.Get("/healthz", [this](const httplib::Request& req, httplib::Response& res) { handleHealth(req, res); });
    server.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) { handleModels(req, res); });
    server.Post("/reset", [this](const httplib::Request& req, httplib::Response& res) { handleReset(req, res); });
    server.Post("/v1/tune/update_cache", [this](const httplib::Request& req, httplib::Response& res) {
        handleUpdateRuntimeCache(req, res);
    });
    server.Post("/v1/prefill/text", [this](const httplib::Request& req, httplib::Response& res) {
        handlePrefillText(req, res);
    });
    server.Post("/v1/kv/pic_caches", [this](const httplib::Request& req, httplib::Response& res) {
        handlePicCaches(req, res);
    });
    server.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        handleChatCompletions(req, res);
    });
    server.Post("/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        handleChatCompletions(req, res);
    });

    std::cout << "Starting PIC server on http://" << mConfig.host << ":" << mConfig.port << "\n";
    return server.listen(mConfig.host.c_str(), mConfig.port);
}

void PicServer::allowCors(httplib::Response& res) const {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
}

void PicServer::handleRoot(const httplib::Request&, httplib::Response& res) {
    allowCors(res);
    auto body = runtimeInfo();
    body["endpoints"] = json::array({
        "/healthz",
        "/v1/models",
        "/reset",
        "/v1/tune/update_cache",
        "/v1/prefill/text",
        "/v1/kv/pic_caches",
        "/v1/chat/completions",
        "/chat/completions",
    });
    writeJson(res, body);
}

void PicServer::handleHealth(const httplib::Request&, httplib::Response& res) {
    allowCors(res);
    writeJson(res, json({{"status", "ok"}, {"model", mConfig.servedModelName}}), 200, -1);
}

void PicServer::handleModels(const httplib::Request&, httplib::Response& res) {
    allowCors(res);
    json body = {
        {"object", "list"},
        {"data", json::array({{
            {"id", mConfig.servedModelName},
            {"object", "model"},
            {"created", 0},
            {"owned_by", "kvshare-edge"},
        }})},
    };
    writeJson(res, body);
}

void PicServer::handleReset(const httplib::Request&, httplib::Response& res) {
    allowCors(res);
    std::lock_guard<std::mutex> lock(mMutex);
    if (mLlm) {
        mLlm->clearPrefixCacheFile();
        mLlm->finishExternalPagedKVRequest();
        mLlm->reset();
    }
    writeJson(res, json({
        {"status", "ok"},
        {"scope", "llm_request_state"},
        {"cleared", json::array({"prefix_cache_mode", "paged_pic_request", "context_history"})},
    }), 200, -1);
}

void PicServer::handleUpdateRuntimeCache(const httplib::Request&, httplib::Response& res) {
    allowCors(res);
    std::lock_guard<std::mutex> lock(mMutex);
    if (mLlm) {
        mLlm->updateRuntimeCache();
    }
    writeJson(res, json({
        {"status", "ok"},
        {"scope", "mnn_runtime_cache"},
        {"note", "runtime manager cache updated through MNN native cache flow"},
    }), 200, -1);
}

void PicServer::handlePrefillText(const httplib::Request& req, httplib::Response& res) {
    allowCors(res);
    runLockedJsonEndpoint(req, res, mMutex, [this](const json& request, json& response, std::string& error) {
        return buildTextCache(request, response, error);
    });
}

void PicServer::handlePicCaches(const httplib::Request& req, httplib::Response& res) {
    allowCors(res);
    runLockedJsonEndpoint(req, res, mMutex, [this](const json& request, json& response, std::string& error) {
        return buildPicCache(request, response, error);
    });
}

void PicServer::handleChatCompletions(const httplib::Request& req, httplib::Response& res) {
    allowCors(res);
    json request;
    if (!parseJsonBody(req, res, request)) {
        return;
    }
    std::vector<json> requests;
    json response;
    std::string error;
    bool ok = false;
    if (!normalizeChatCompletionBatchRequest(request, requests, error)) {
        writeJsonError(res, 400, error);
        return;
    }
    if (!validateHomogeneousChatBatchRequest(requests, error)) {
        writeJsonError(res, 400, error);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mMutex);
        ok = completeChatBatch(requests, response, error);
    }
    if (!ok) {
        writeJsonError(res, 500, error);
        return;
    }
    writeJson(res, response);
}

json PicServer::runtimeInfo() const {
    auto cfg = modelConfig();
    return {
        {"name", "mnn_pic_server"},
        {"backend", runtimeBackend()},
        {"attention_mode", jsonBool(cfg, "paged_attention", false) ? "paged" : "standard"},
        {"paged_attention", jsonBool(cfg, "paged_attention", false)},
        {"request_scoped_kv", true},
        {"kv_cache_dir", absoluteString(mConfig.kvCacheDir)},
        {"status", mLlm ? "ok" : "not_loaded"},
    };
}

json PicServer::modelConfig() const {
    if (!mLlm) {
        return json::object();
    }
    auto parsed = json::parse(mLlm->dump_config(), nullptr, false);
    return parsed.is_object() ? parsed : json::object();
}

std::string PicServer::runtimeBackend() const {
    auto cfg = modelConfig();
    auto backend = jsonString(cfg, "backend_type", "unknown");
    return "mnn_" + backend;
}

bool PicServer::buildTextCache(const json& request, json& response, std::string& error) {
    if (!mLlm) {
        error = "LLM is not loaded";
        return false;
    }
    if (!request.is_object()) {
        error = "Request must be a JSON object";
        return false;
    }
    if (request.value("type", "text") != "text") {
        error = "Only type=\"text\" is supported";
        return false;
    }
    if (request.contains("path")) {
        error = "`path` is not accepted. Send inline text in `content`.";
        return false;
    }
    if (!request.contains("id") || !request["id"].is_string() || request["id"].get<std::string>().empty()) {
        error = "Text cache request requires non-empty string field `id`";
        return false;
    }
    const bool hasContent = request.contains("content") && request["content"].is_string() &&
                            !request["content"].get<std::string>().empty();
    const bool hasExplicitTokenIds = request.contains("token_ids") && request["token_ids"].is_array();
    if (!hasContent && !hasExplicitTokenIds) {
        error = "Text cache request requires non-empty `content` or explicit non-empty `token_ids`";
        return false;
    }

    const std::string id = request["id"].get<std::string>();
    const std::string content = hasContent ? request["content"].get<std::string>() : "";
    if (hasContent && looksLikeFilePath(content)) {
        error = "`content` must be the actual text body, not a filesystem path";
        return false;
    }
    const bool force = request.value("force", false);
    const std::string cacheName = sanitizeCachePart(
        request.contains("cache_name") && request["cache_name"].is_string()
            ? request["cache_name"].get<std::string>()
            : std::string("doc_") + id);
    const std::string backend = runtimeBackend();
    fs::path cacheRoot = fs::path(mConfig.kvCacheDir) / "objects" / backend / cacheName;
    fs::path layersDir = cacheRoot / "layers";
    fs::path metaPath = cacheRoot / "meta.json";
    fs::path tokensPath = cacheRoot / "tokens.json";
    std::string fileStem = pathJoinForMNN(
        pathJoinForMNN(pathJoinForMNN(pathJoinForMNN("objects", backend), cacheName), "layers"),
        cacheName);

    std::vector<int> tokenIds;
    if (hasExplicitTokenIds) {
        tokenIds = jsonIntVector(request["token_ids"]);
    } else {
        tokenIds = mLlm->tokenizer_encode(content);
    }
    if (tokenIds.empty()) {
        error = hasExplicitTokenIds ? "Text cache token_ids must be a non-empty integer array"
                                    : "Text tokenization produced no tokens";
        return false;
    }
    const int textCacheTokenLimit = envInt("MNN_PIC_SERVER_MAX_TEXT_CACHE_TOKENS", 0);
    if (textCacheTokenLimit > 0 && static_cast<int>(tokenIds.size()) > textCacheTokenLimit) {
        error = "Text cache token count " + std::to_string(tokenIds.size()) +
                " exceeds MNN_PIC_SERVER_MAX_TEXT_CACHE_TOKENS=" + std::to_string(textCacheTokenLimit);
        return false;
    }
    const std::string sourceSha256 = hasExplicitTokenIds ? sha256Hex(tokenIdsDigestInput(tokenIds))
                                                        : sha256Hex(content);

    if (!force && existingCacheMatches(metaPath, sourceSha256, tokenIds)) {
        response = readJsonFile(metaPath);
        response["cache_hit"] = true;
        response["cache_status"] = "reused";
        return true;
    }

    std::error_code ec;
    fs::remove_all(cacheRoot, ec);
    fs::create_directories(layersDir, ec);
    if (ec) {
        error = "Failed to create cache directory: " + layersDir.string();
        return false;
    }

    mLlm->reset();
    mLlm->setPrefixCacheFile(fileStem);
    std::ostringstream sink;
    mLlm->response(tokenIds, &sink, "", 0);
    mLlm->clearPrefixCacheFile();
    mLlm->reset();

    auto cfg = modelConfig();
    int layerCount = jsonInt(cfg, "layer_nums", 0);
    int detectedLayers = countLayerFiles(layersDir, cacheName);
    if (layerCount <= 0) {
        layerCount = detectedLayers;
    }
    if (layerCount <= 0 || detectedLayers <= 0) {
        error = "Prefill finished but no layer KV files were exported";
        return false;
    }
    if (detectedLayers < layerCount) {
        error = "Expected " + std::to_string(layerCount) + " layer KV files, found " +
                std::to_string(detectedLayers);
        return false;
    }

    KvShape kvShape = readKvShape(layersDir, cacheName);
    json kvLayout = makeKvLayout(cfg, static_cast<int>(tokenIds.size()), kvShape);
    int layoutBatch = kvLayout.value("batch", 1);
    int layoutKvHeads = kvLayout.value("kv_heads", 0);
    int layoutHeadDim = kvLayout.value("head_dim", 0);
    json kvFiles = json::array();
    for (int layer = 0; layer < layerCount; ++layer) {
        fs::path keyPath = layersDir / (cacheName + "_" + std::to_string(layer) + ".k");
        fs::path valuePath = layersDir / (cacheName + "_" + std::to_string(layer) + ".v");
        kvFiles.push_back({
            {"layer_index", layer},
            {"key_path", absoluteString(keyPath)},
            {"value_path", absoluteString(valuePath)},
            {"key_bytes", fileSizeOrZero(keyPath)},
            {"value_bytes", fileSizeOrZero(valuePath)},
            {"file_format", "raw"},
            {"shape_path", absoluteString(layersDir / (cacheName + "_" + std::to_string(layer) + ".json"))},
            {"key_shape", json::array({static_cast<int>(tokenIds.size()), layoutBatch, layoutKvHeads, layoutHeadDim})},
            {"value_shape", json::array({layoutBatch, layoutKvHeads, static_cast<int>(tokenIds.size()), layoutHeadDim})},
        });
    }

    json tokensJson = {
        {"format", kTokensFormat},
        {"id", id},
        {"cache_name", cacheName},
        {"token_count", tokenIds.size()},
        {"token_ids", tokenIds},
    };
    if (!writeJsonFile(tokensPath, tokensJson)) {
        error = "Failed to write tokens metadata";
        return false;
    }

    response = {
        {"format", kMetaFormat},
        {"type", "text"},
        {"id", id},
        {"cache_name", cacheName},
        {"kv_cache_dir", absoluteString(mConfig.kvCacheDir)},
        {"meta_path", absoluteString(metaPath)},
        {"token_ids_path", absoluteString(tokensPath)},
        {"token_count", tokenIds.size()},
        {"token_ids", tokenIds},
        {"source", hasExplicitTokenIds
            ? json({{"kind", hasContent ? "inline_tokens_with_text" : "inline_tokens"},
                    {"token_ids_sha256", sourceSha256},
                    {"content_sha256", hasContent ? sha256Hex(content) : ""}})
            : json({{"kind", "inline"}, {"content_sha256", sourceSha256}})},
        {"backend", backend},
        {"attention_mode", jsonBool(cfg, "paged_attention", false) ? "paged" : "standard"},
        {"layer_count", layerCount},
        {"cache_hit", false},
        {"cache_status", "built"},
        {"kv_layout", kvLayout},
        {"kv", kvFiles},
    };
    if (!writeJsonFile(metaPath, response)) {
        error = "Failed to write cache metadata";
        return false;
    }
    return true;
}

bool PicServer::buildPicCache(const json& request, json& response, std::string& error) {
    if (!mLlm) {
        error = "LLM is not loaded";
        return false;
    }
    PreparedPicCache pic;
    const std::string backend = runtimeBackend();
    if (!preparePicCacheFromRequest(mConfig.kvCacheDir, backend, request, pic, error)) {
        return false;
    }
    auto cfg = modelConfig();
    KvShape shape = kvShapeFromFirstSegment(pic.segments);
    json kvLayout = makeKvLayout(cfg, static_cast<int>(pic.tokenIds.size()), shape);
    response = {
        {"format", kMetaFormat},
        {"type", "pic_cache"},
        {"id", pic.id},
        {"cache_name", pic.cacheName},
        {"kv_cache_dir", absoluteString(mConfig.kvCacheDir)},
        {"token_count", pic.tokenIds.size()},
        {"token_ids", pic.tokenIds},
        {"selection_algorithm", pic.selectionAlgorithm},
        {"pic_recompute_ratio", pic.recomputeRatio},
        {"pic_recompute_score_layer_idx", pic.scoreLayerIdx},
        {"pic_recompute_logical_indices", pic.explicitLogicalIndices},
        {"pic_recompute_pic_local_indices", pic.explicitPicLocalIndices},
        {"pic_decode_refine_enabled", false},
        {"pic_decode_refine_tokens_per_decode_step", 1},
        {"pic_decode_refine_top_m", 32},
        {"pic_decode_refine_score_threshold", 1e-6},
        {"pic_decode_refine_score_margin", 0.0},
        {"pic_decode_refine_score_decay", 0.8},
        {"text_caches", pic.textCaches},
        {"backend", backend},
        {"attention_mode", jsonBool(cfg, "paged_attention", false) ? "paged" : "standard"},
        {"layer_count", jsonInt(cfg, "layer_nums", 0)},
        {"kv_layout", kvLayout},
        {"note", "mnn_pic_server PIC loads mnn_paged_attention_raw_v1 canonical_no_rope KV tensors at runtime."},
    };
    return true;
}

bool PicServer::completeChatBatch(const std::vector<json>& requests, json& response, std::string& error) {
    if (!mLlm) {
        error = "LLM is not loaded";
        return false;
    }
    if (requests.empty()) {
        error = "ChatCompletionBatchRequest requires at least one request";
        return false;
    }

    const auto created = unixSecondsNow();
    const std::string responseId = "chatcmpl-batch-" + std::to_string(created);
    json data = json::array();
    int totalPromptTokens = 0;
    int totalCompletionTokens = 0;
    for (size_t index = 0; index < requests.size(); ++index) {
        json itemResponse;
        std::string itemError;
        if (!completeChatBatchItem(requests[index], itemResponse, itemError)) {
            error = "batch item " + std::to_string(index) + " failed: " + itemError;
            return false;
        }
        itemResponse["id"] = responseId + "-" + std::to_string(index);
        itemResponse["batch_index"] = index;
        if (itemResponse.contains("choices") && itemResponse["choices"].is_array() &&
            !itemResponse["choices"].empty() && itemResponse["choices"][0].is_object()) {
            itemResponse["choices"][0]["index"] = index;
        }
        if (itemResponse.contains("usage") && itemResponse["usage"].is_object()) {
            totalPromptTokens += jsonInt(itemResponse["usage"], "prompt_tokens", 0);
            totalCompletionTokens += jsonInt(itemResponse["usage"], "completion_tokens", 0);
        }
        data.push_back(std::move(itemResponse));
    }

    response = {
        {"id", responseId},
        {"object", "list"},
        {"created", created},
        {"model", mConfig.servedModelName},
        {"batch_execution", {
            {"mode", "request-scheduled-batch-entry"},
            {"native_paged_batch", false},
            {"batch_size", requests.size()},
        }},
        {"usage", {
            {"prompt_tokens", totalPromptTokens},
            {"completion_tokens", totalCompletionTokens},
            {"total_tokens", totalPromptTokens + totalCompletionTokens},
        }},
        {"data", data},
    };
    return true;
}

bool PicServer::completeChatBatchItem(const json& request, json& response, std::string& error) {
    if (!mLlm) {
        error = "LLM is not loaded";
        return false;
    }
    if (!request.is_object()) {
        error = "Chat completion request must be a JSON object";
        return false;
    }
    if (request.value("stream", false)) {
        error = "stream=true is not supported by mnn_pic_server yet";
        return false;
    }
    std::string requestedModel = jsonString(request, "model");
    if (!requestedModel.empty() && requestedModel != mConfig.servedModelName) {
        error = "This server only loaded model=" + mConfig.servedModelName + ", but request asked for " +
                requestedModel;
        return false;
    }
    const auto requestTokenIds =
        jsonIntVectorAlias(request, {"full_prompt_token_ids", "prompt_token_ids", "input_token_ids"});
    MNN::Transformer::ChatMessages messages;
    if (request.contains("messages")) {
        if (!request["messages"].is_array() || request["messages"].empty()) {
            error = "messages must be a non-empty array when provided";
            return false;
        }
        for (const auto& item : request["messages"]) {
            if (!item.is_object() || !item.contains("role") || !item["role"].is_string() ||
                !item.contains("content") || !item["content"].is_string()) {
                error = "Each message must contain string role/content";
                return false;
            }
            messages.emplace_back(item["role"].get<std::string>(), item["content"].get<std::string>());
        }
    } else if (requestTokenIds.empty()) {
        error = "Chat completion request needs messages or explicit full_prompt_token_ids/prompt_token_ids";
        return false;
    }
    int maxTokens = jsonInt(request, "max_tokens", -1);
    if (maxTokens < -1) {
        error = "max_tokens must be non-negative when provided";
        return false;
    }

    std::ostringstream sink;
    std::vector<int> outputTokens;
    int promptTokens = 0;
    int preludeTokensCount = 0;
    int picTokensCount = 0;
    json picInfo;
    auto cfg = modelConfig();
    int layerCount = jsonInt(cfg, "layer_nums", 0);

    if (!request.contains("pic_cache") || request["pic_cache"].is_null()) {
        std::vector<int> inputTokenIds = requestTokenIds;
        if (inputTokenIds.empty()) {
            std::string rendered = mLlm->apply_chat_template(messages, true);
            inputTokenIds = mLlm->tokenizer_encode(rendered);
        }
        const int prefillTokenLimit = envInt("MNN_PIC_SERVER_MAX_PREFILL_TOKENS", 0);
        if (prefillTokenLimit > 0 && static_cast<int>(inputTokenIds.size()) > prefillTokenLimit) {
            error = "Chat prefill token count " + std::to_string(inputTokenIds.size()) +
                    " exceeds MNN_PIC_SERVER_MAX_PREFILL_TOKENS=" + std::to_string(prefillTokenLimit);
            return false;
        }
        MnnLlmTraceInfo traceInfo;
        traceInfo.algorithm = "full-compute";
        traceInfo.executionMode = "native-full-compute";
        traceInfo.promptTotalTokens = static_cast<int64_t>(inputTokenIds.size());
        traceInfo.suffixTokens = static_cast<int64_t>(inputTokenIds.size());
        traceInfo.maxTokens = maxTokens;
        mLlm->reset();
        mLlm->generate_init(&sink, "");
        {
            MnnLlmPerfettoSlice prefillSlice("prefill", traceInfo);
            if (!mLlm->prefill(inputTokenIds)) {
                error = "Failed to prefill no-PIC chat request";
                return false;
            }
        }
        if (maxTokens != 0) {
            if (std::getenv("MNN_PIC_DECODE_DEBUG") != nullptr) {
                auto context = mLlm->getContext();
                std::fprintf(stderr, "PIC server decode debug branch=no_pic max_tokens=%d status=%d current=%d "
                                     "output_tokens=%d\n",
                             maxTokens, context != nullptr ? static_cast<int>(context->status) : -999,
                             context != nullptr ? context->current_token : -999,
                             context != nullptr ? static_cast<int>(context->output_tokens.size()) : -1);
                std::fflush(stderr);
            }
            MnnLlmPerfettoSlice decodeSlice("decode", traceInfo);
            outputTokens = mLlm->decode(maxTokens);
        }
        auto context = mLlm->getContext();
        if (context != nullptr) {
            if (maxTokens == 0) {
                outputTokens = context->output_tokens;
            }
            promptTokens = static_cast<int>(inputTokenIds.size());
        }
    } else {
        PreparedPicCache pic;
        if (!preparePicCacheFromRequest(mConfig.kvCacheDir, runtimeBackend(), request["pic_cache"], pic, error)) {
            return false;
        }
        std::vector<int> preludeTokenIds;
        std::vector<int> suffixTokenIds;
        std::string placeholder = pic.placeholder.empty() ? kDefaultPicPlaceholder : pic.placeholder;
        const std::string promptProtocol = "explicit-full-prompt-token-span-v1";
        if (pic.fullPromptTokenIds.empty()) {
            pic.fullPromptTokenIds = requestTokenIds;
        }
        if (pic.promptSpans.empty() &&
            !parseExplicitCacheTokenSpans(request, pic.promptSpans, error)) {
            return false;
        }
        std::vector<int> fullPromptTokenIds;
        json explicitTokenMapping = json::array();
        if (!applyExplicitPicPromptMapping(pic, preludeTokenIds, suffixTokenIds,
                                           fullPromptTokenIds, explicitTokenMapping, error)) {
            return false;
        }
        if (suffixTokenIds.empty()) {
            error = "PIC prompt suffix tokenization produced no tokens";
            return false;
        }
        const int preludeTraceTokens = static_cast<int>(preludeTokenIds.size());
        const int picTraceTokens = static_cast<int>(pic.tokenIds.size());
        const int suffixTraceTokens = static_cast<int>(suffixTokenIds.size());
        const int totalPrefillTokens = preludeTraceTokens + picTraceTokens + suffixTraceTokens;
        const int prefillTokenLimit = envInt("MNN_PIC_SERVER_MAX_PREFILL_TOKENS", 0);
        if (prefillTokenLimit > 0 && totalPrefillTokens > prefillTokenLimit) {
            error = "PIC prefill token count " + std::to_string(totalPrefillTokens) +
                    " exceeds MNN_PIC_SERVER_MAX_PREFILL_TOKENS=" + std::to_string(prefillTokenLimit);
            return false;
        }
        MnnLlmTraceInfo traceInfo;
        traceInfo.algorithm = traceAlgorithmName(pic.selectionAlgorithm);
        traceInfo.executionMode = traceExecutionModeForAlgorithm(pic.selectionAlgorithm);
        traceInfo.budgetRatio = traceBudgetRatio(pic.selectionAlgorithm, pic.recomputeRatio);
        traceInfo.recomputeBudgetTokens =
            traceRecomputeBudgetTokens(pic.selectionAlgorithm, picTraceTokens, pic.recomputeRatio);
        traceInfo.promptTotalTokens = totalPrefillTokens;
        traceInfo.preludeTokens = preludeTraceTokens;
        traceInfo.picTokens = picTraceTokens;
        traceInfo.suffixTokens = suffixTraceTokens;
        traceInfo.maxTokens = maxTokens;
        traceInfo.scoreLayerIdx = pic.scoreLayerIdx;
        traceInfo.hasPicCache = true;

        PicExecutionPlan plan;
        {
            MnnLlmPerfettoSlice prefillSlice("prefill", traceInfo);
            std::vector<int> nativeSelectedLocalIndices;
            json scoreMetadata = json::object();
            bool hasNativeSelectedLocalIndices = false;
            bool graphBoundaryPrefillDone = false;
            const bool graphBoundaryEnabled =
                jsonBool(modelConfig(), "pic_recompute_budget", false) &&
                (pic.selectionAlgorithm == "cacheblend" || pic.selectionAlgorithm == "delta-v" ||
                 pic.selectionAlgorithm == "epic");
            if (graphBoundaryEnabled && pic.selectionAlgorithm == "epic") {
                const int effectiveScoreLayer =
                    std::min(std::max(0, pic.scoreLayerIdx), std::max(0, layerCount - 1));
                const int recompute = picTraceTokens <= 0 || pic.recomputeRatio <= 0.0
                    ? 0
                    : std::min(picTraceTokens,
                               std::max(1, static_cast<int>(std::ceil(picTraceTokens * pic.recomputeRatio))));
                nativeSelectedLocalIndices.clear();
                for (int i = 0; i < recompute; ++i) {
                    nativeSelectedLocalIndices.emplace_back(i);
                }
                mLlm->reset();
                mLlm->generate_init(&sink, "");
                if (!mLlm->prefillFixedGraphExternalPagedKV(
                        fullPromptTokenIds, pic.segments, static_cast<int>(preludeTokenIds.size()),
                        static_cast<int>(pic.tokenIds.size()), effectiveScoreLayer, nativeSelectedLocalIndices)) {
                    mLlm->finishExternalPagedKVRequest();
                    error = "Native graph-level epic prefill failed on backend " + runtimeBackend();
                    return false;
                }
                graphBoundaryPrefillDone = true;
                scoreMetadata = {
                    {"score_layer_idx", effectiveScoreLayer},
                    {"score_source", "fixed_pic_head_contiguous_tokens"},
                    {"score_kind", "pic_head_fixed_ratio"},
                    {"score_pass", "not_required_graph_boundary"},
                    {"graph_boundary", "score_layer_pic_score_attention"},
                    {"selected_count", nativeSelectedLocalIndices.size()},
                };
            } else if (graphBoundaryEnabled) {
                const int effectiveScoreLayer =
                    std::min(std::max(0, pic.scoreLayerIdx), std::max(0, layerCount - 1));
                mLlm->reset();
                mLlm->generate_init(&sink, "");
                if (!mLlm->prefillCacheBlendGraphExternalPagedKV(
                        fullPromptTokenIds, pic.segments, static_cast<int>(preludeTokenIds.size()),
                        static_cast<int>(pic.tokenIds.size()), effectiveScoreLayer, pic.recomputeRatio,
                        nativeSelectedLocalIndices)) {
                    mLlm->finishExternalPagedKVRequest();
                    error = "Native graph-level cacheblend score/top-k failed on backend " + runtimeBackend();
                    return false;
                }
                graphBoundaryPrefillDone = true;
                hasNativeSelectedLocalIndices = true;
                scoreMetadata = {
                    {"score_layer_idx", effectiveScoreLayer},
                    {"score_source", "request_full_reference_pagedcache_minus_cached_pic_value"},
                    {"score_kind", "layer_value_delta_mean_abs"},
                    {"score_pass", "request_full_prompt_graph_boundary"},
                    {"topk_location", "backend_device"},
                    {"host_transfer", "selected_local_indices_only"},
                    {"graph_boundary", "score_layer_pic_score_attention"},
                    {"selected_count", nativeSelectedLocalIndices.size()},
                };
            } else if (pic.selectionAlgorithm == "cacheblend" || pic.selectionAlgorithm == "delta-v") {
                const int effectiveScoreLayer =
                    std::min(std::max(0, pic.scoreLayerIdx), std::max(0, layerCount - 1));
                mLlm->reset();
                std::ostringstream scoreSink;
                mLlm->generate_init(&scoreSink, "");
                if (!mLlm->selectCacheBlendExternalPagedKV(
                        fullPromptTokenIds, pic.segments, static_cast<int>(preludeTokenIds.size()),
                        static_cast<int>(pic.tokenIds.size()), effectiveScoreLayer, pic.recomputeRatio,
                        nativeSelectedLocalIndices)) {
                    error = "Native cacheblend score/top-k failed on backend " + runtimeBackend();
                    return false;
                }
                hasNativeSelectedLocalIndices = true;
                scoreMetadata = {
                    {"score_layer_idx", effectiveScoreLayer},
                    {"score_source", "request_full_reference_pagedcache_minus_cached_pic_value"},
                    {"score_kind", "layer_value_delta_mean_abs"},
                    {"score_pass", "request_full_reference_pagedcache_no_disk_write"},
                    {"topk_location", "backend_device"},
                    {"host_transfer", "selected_local_indices_only"},
                    {"selected_count", nativeSelectedLocalIndices.size()},
                };
            }
            plan = buildExecutionPlan(
                pic, static_cast<int>(preludeTokenIds.size()), layerCount,
                hasNativeSelectedLocalIndices ? &nativeSelectedLocalIndices : nullptr,
                hasNativeSelectedLocalIndices ? &scoreMetadata : nullptr);
            if (graphBoundaryPrefillDone) {
                plan.executionMode = pic.selectionAlgorithm == "epic" ? "native-epic-graph-boundary"
                                                                      : "native-cacheblend-graph-boundary";
                plan.sparseRecompute = false;
                plan.metadata["native_sparse_recompute_scope"] = "exported_graph_score_layer_boundary";
                plan.metadata["graph_level_boundary"] = true;
            }
            traceInfo.executionMode = plan.executionMode;
            traceInfo.recomputeBudgetTokens = plan.recomputeTokenCount;

            if (!graphBoundaryPrefillDone) {
                mLlm->reset();
                mLlm->generate_init(&sink, "");
                if (!plan.externalTokenIds.empty() &&
                    !mLlm->reserveExternalPagedKVSourceSlots(plan.externalTokenIds.size())) {
                    mLlm->finishExternalPagedKVRequest();
                    error = "Failed to reserve OpenCL PagedCache source slots for persistent PIC cache";
                    return false;
                }
            }
            if (!graphBoundaryPrefillDone && plan.fullCompute) {
                if (!mLlm->prefill(fullPromptTokenIds)) {
                    mLlm->finishExternalPagedKVRequest();
                    error = "Failed to prefill full-compute PIC chat request";
                    return false;
                }
                if (std::getenv("MNN_PIC_DECODE_DEBUG") != nullptr) {
                    auto context = mLlm->getContext();
                    std::fprintf(stderr, "PIC server prefill debug step=full_compute status=%d all_seq=%d\n",
                                 context != nullptr ? static_cast<int>(context->status) : -999,
                                 context != nullptr ? context->all_seq_len : -1);
                    std::fflush(stderr);
                }
            } else if (!graphBoundaryPrefillDone) {
                std::vector<int> firstPrefill;
                firstPrefill.reserve(preludeTokenIds.size() + plan.prefillPicTokenIds.size());
                firstPrefill.insert(firstPrefill.end(), preludeTokenIds.begin(), preludeTokenIds.end());
                firstPrefill.insert(firstPrefill.end(), plan.prefillPicTokenIds.begin(),
                                    plan.prefillPicTokenIds.end());
                if (!firstPrefill.empty()) {
                    if (!mLlm->prefill(firstPrefill)) {
                        mLlm->finishExternalPagedKVRequest();
                        error = "Failed to prefill prelude/PIC prefix tokens";
                        return false;
                    }
                    if (std::getenv("MNN_PIC_DECODE_DEBUG") != nullptr) {
                        auto context = mLlm->getContext();
                        std::fprintf(stderr, "PIC server prefill debug step=first_prefill status=%d all_seq=%d "
                                             "tokens=%d\n",
                                     context != nullptr ? static_cast<int>(context->status) : -999,
                                     context != nullptr ? context->all_seq_len : -1,
                                     static_cast<int>(firstPrefill.size()));
                        std::fflush(stderr);
                    }
                } else {
                    mLlm->beginExternalPagedKVRequest();
                }
                if (!plan.externalTokenIds.empty() &&
                    !mLlm->appendExternalPagedKV(plan.externalTokenIds, plan.externalSegments)) {
                    mLlm->finishExternalPagedKVRequest();
                    error = "Failed to bind persistent PIC cache source into current request PagedCache";
                    return false;
                }
                if (std::getenv("MNN_PIC_DECODE_DEBUG") != nullptr) {
                    auto context = mLlm->getContext();
                    std::fprintf(stderr, "PIC server prefill debug step=bind_pic_source status=%d all_seq=%d "
                                         "tokens=%d\n",
                                 context != nullptr ? static_cast<int>(context->status) : -999,
                                 context != nullptr ? context->all_seq_len : -1,
                                 static_cast<int>(plan.externalTokenIds.size()));
                    std::fflush(stderr);
                }
                if (plan.sparseRecompute &&
                    !mLlm->recomputeExternalPagedKV(plan.sparseLogicalIndices, plan.sparseTokenIds,
                                                    plan.scoreLayerIdx)) {
                    mLlm->finishExternalPagedKVRequest();
                    error = "Failed to sparse-recompute selected PIC KV tokens";
                    return false;
                }
                if (std::getenv("MNN_PIC_DECODE_DEBUG") != nullptr && plan.sparseRecompute) {
                    auto context = mLlm->getContext();
                    std::fprintf(stderr, "PIC server prefill debug step=sparse_recompute status=%d all_seq=%d\n",
                                 context != nullptr ? static_cast<int>(context->status) : -999,
                                 context != nullptr ? context->all_seq_len : -1);
                    std::fflush(stderr);
                }
                if (!mLlm->prefill(suffixTokenIds)) {
                    mLlm->finishExternalPagedKVRequest();
                    error = "Failed to prefill PIC prompt suffix tokens";
                    return false;
                }
                if (std::getenv("MNN_PIC_DECODE_DEBUG") != nullptr) {
                    auto context = mLlm->getContext();
                    std::fprintf(stderr, "PIC server prefill debug step=suffix status=%d all_seq=%d tokens=%d\n",
                                 context != nullptr ? static_cast<int>(context->status) : -999,
                                 context != nullptr ? context->all_seq_len : -1,
                                 static_cast<int>(suffixTokenIds.size()));
                    std::fflush(stderr);
                }
            }
        }
        if (maxTokens != 0) {
            if (std::getenv("MNN_PIC_DECODE_DEBUG") != nullptr) {
                auto context = mLlm->getContext();
                std::fprintf(stderr, "PIC server decode debug branch=pic max_tokens=%d execution=%s status=%d "
                                     "current=%d output_tokens=%d\n",
                             maxTokens, traceInfo.executionMode.c_str(),
                             context != nullptr ? static_cast<int>(context->status) : -999,
                             context != nullptr ? context->current_token : -999,
                             context != nullptr ? static_cast<int>(context->output_tokens.size()) : -1);
                std::fflush(stderr);
            }
            MnnLlmPerfettoSlice decodeSlice("decode", traceInfo);
            outputTokens = mLlm->decode(maxTokens);
        } else {
            auto context = mLlm->getContext();
            if (context != nullptr) {
                outputTokens = context->output_tokens;
            }
        }
        mLlm->finishExternalPagedKVRequest();
        promptTokens = static_cast<int>(suffixTokenIds.size());
        preludeTokensCount = static_cast<int>(preludeTokenIds.size());
        picTokensCount = static_cast<int>(pic.tokenIds.size());

        std::vector<int> recomputePicLocalIndices;
        recomputePicLocalIndices.reserve(plan.recomputeLogicalIndices.size());
        for (int logical : plan.recomputeLogicalIndices) {
            int local = logical - preludeTokensCount;
            if (local >= 0 && local < static_cast<int>(pic.tokenIds.size())) {
                recomputePicLocalIndices.emplace_back(local);
            }
        }

        KvShape shape = kvShapeFromFirstSegment(pic.segments);
        picInfo = {
            {"format", kMetaFormat},
            {"type", "pic_cache"},
            {"id", pic.id},
            {"cache_name", sanitizeCachePart("pic_" + pic.id)},
            {"kv_cache_dir", absoluteString(mConfig.kvCacheDir)},
            {"token_count", pic.tokenIds.size()},
            {"token_ids", pic.tokenIds},
            {"system_prompt_token_count", preludeTokensCount},
            {"prelude_token_count", preludeTokensCount},
            {"prompt_token_count", promptTokens},
            {"text_caches", pic.textCaches},
            {"prompt_protocol", promptProtocol},
            {"token_alignment", "explicit_request_verified"},
            {"full_prompt_token_count", fullPromptTokenIds.size()},
            {"doc_cache_token_mapping", explicitTokenMapping},
            {"placeholder", placeholder},
            {"selection_algorithm", pic.selectionAlgorithm},
            {"pic_recompute_ratio", pic.recomputeRatio},
            {"pic_recompute_score_layer_idx", pic.scoreLayerIdx},
            {"pic_recompute_logical_indices", plan.recomputeLogicalIndices},
            {"pic_recompute_pic_local_indices", recomputePicLocalIndices},
            {"pic_decode_refine_enabled", false},
            {"pic_decode_refine_tokens_per_decode_step", 1},
            {"pic_decode_refine_top_m", 32},
            {"pic_decode_refine_score_threshold", 1e-6},
            {"pic_decode_refine_score_margin", 0.0},
            {"pic_decode_refine_score_decay", 0.8},
            {"precision_recovery", precisionRecoverySummary(plan, pic.recomputeRatio, pic.scoreLayerIdx)},
            {"kv_layout", makeKvLayout(cfg, static_cast<int>(pic.tokenIds.size()), shape)},
            {"backend", runtimeBackend()},
            {"attention_mode", jsonBool(cfg, "paged_attention", false) ? "paged" : "standard"},
            {"layer_count", layerCount},
            {"note", "mnn_pic_server consumed PIC through paged KV slots; non-native scoring algorithms use metadata-marked fallback."},
        };
    }

    auto context = mLlm->getContext();
    std::string text = trimStopStrings(sink.str(), request.value("stop", json()));
    const int completionTokens = static_cast<int>(outputTokens.size());
    const int totalPromptTokens = promptTokens + preludeTokensCount + picTokensCount;
    response = {
        {"id", "chatcmpl-" + std::to_string(unixSecondsNow())},
        {"object", "chat.completion"},
        {"created", unixSecondsNow()},
        {"model", mConfig.servedModelName},
        {"choices", json::array({{
            {"index", 0},
            {"message", {{"role", "assistant"}, {"content", text}}},
            {"finish_reason", finishReasonForStatus(context, maxTokens)},
        }})},
        {"usage", {
            {"prompt_tokens", totalPromptTokens},
            {"completion_tokens", completionTokens},
            {"total_tokens", totalPromptTokens + completionTokens},
        }},
    };
    if (!picInfo.is_null()) {
        response["pic_cache"] = picInfo;
    }
    return true;
}

} // namespace pic
