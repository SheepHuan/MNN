#include "llm/llm.hpp"
#include "../src/tokenizer/tokenizer.hpp"
#include "../src/ujson.hpp"

#include <MNN/Tensor.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using MNN::Transformer::Llm;
using MNN::Transformer::Tokenizer;

namespace fs = std::filesystem;

namespace {

struct Span {
    int start = 0;
    int end = 0;
};

struct CachedChunkMeta {
    std::vector<int> auxInputIds;
    std::vector<Span> auxOffsets;
    std::vector<Span> primaryTokenOffsets;
    std::vector<int> auxChunkIndices;
    int chunkCharStart = 0;
    int pastTokenCount = 0;
    bool cacheHit = false;
};

[[noreturn]] void printErrorAndExit(const std::string& error);

int64_t monotonicUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

double elapsedSeconds(int64_t startUs) {
    return static_cast<double>(monotonicUs() - startUs) / 1000000.0;
}

std::string readAllStdin() {
    std::ostringstream os;
    os << std::cin.rdbuf();
    return os.str();
}

std::string jsonString(const ujson::json& value, const char* key, const std::string& fallback = "") {
    if (!value.contains(key) || value[key].is_null()) {
        return fallback;
    }
    return value[key].get<std::string>();
}

int jsonInt(const ujson::json& value, const char* key, int fallback = 0) {
    if (!value.contains(key) || value[key].is_null()) {
        return fallback;
    }
    return value[key].get<int>();
}

double jsonDouble(const ujson::json& value, const char* key, double fallback = 0.0) {
    if (!value.contains(key) || value[key].is_null()) {
        return fallback;
    }
    return value[key].get<double>();
}

bool fileExists(const fs::path& path) {
    std::error_code ec;
    return fs::exists(path, ec);
}

std::string fnv1aHex(const std::string& text) {
    uint64_t hash = 1469598103934665603ull;
    for (unsigned char ch : text) {
        hash ^= static_cast<uint64_t>(ch);
        hash *= 1099511628211ull;
    }
    std::ostringstream os;
    os << std::hex << std::setfill('0') << std::setw(16) << hash;
    return os.str();
}

std::vector<int> jsonIntArray(const ujson::json& value, const char* key) {
    std::vector<int> out;
    if (!value.contains(key) || !value[key].is_array()) {
        return out;
    }
    for (size_t i = 0; i < value[key].size(); ++i) {
        out.emplace_back(value[key][i].get<int>());
    }
    return out;
}

std::string resolveConfigPath(const std::string& spec) {
    fs::path path(spec);
    if (fileExists(path) && fs::is_regular_file(path) && path.extension() == ".json") {
        return fs::absolute(path).string();
    }
    if (fileExists(path) && fs::is_directory(path)) {
        fs::path config = path / "config.json";
        if (fileExists(config)) {
            return fs::absolute(config).string();
        }
    }
    return spec;
}

ujson::json parseJsonFile(const fs::path& path) {
    std::ifstream is(path);
    if (!is.is_open()) {
        return ujson::json();
    }
    std::ostringstream os;
    os << is.rdbuf();
    return ujson::json::parse(os.str());
}

std::string resolveTokenizerPath(const std::string& spec, const std::string& fallbackConfig = "") {
    auto resolveFromDir = [](const fs::path& dir) -> std::string {
        for (const char* name : {"tokenizer.mtok", "tokenizer.txt"}) {
            fs::path candidate = dir / name;
            if (fileExists(candidate)) {
                return fs::absolute(candidate).string();
            }
        }
        return "";
    };

    if (!spec.empty()) {
        fs::path path(spec);
        if (fileExists(path) && fs::is_directory(path)) {
            auto resolved = resolveFromDir(path);
            if (!resolved.empty()) {
                return resolved;
            }
        }
        if (fileExists(path) && fs::is_regular_file(path)) {
            if (path.extension() == ".json") {
                auto config = parseJsonFile(path);
                std::string tokenizerFile = jsonString(config, "tokenizer_file", "");
                if (!tokenizerFile.empty()) {
                    return fs::absolute(path.parent_path() / tokenizerFile).string();
                }
            }
            return fs::absolute(path).string();
        }
    }

    if (!fallbackConfig.empty()) {
        fs::path configPath(resolveConfigPath(fallbackConfig));
        auto config = parseJsonFile(configPath);
        std::string tokenizerFile = jsonString(config, "tokenizer_file", "");
        if (!tokenizerFile.empty()) {
            return fs::absolute(configPath.parent_path() / tokenizerFile).string();
        }
        auto resolved = resolveFromDir(configPath.parent_path());
        if (!resolved.empty()) {
            return resolved;
        }
    }
    return spec;
}

std::unique_ptr<Tokenizer> loadTokenizer(const std::string& spec, const std::string& fallbackConfig = "") {
    std::string path = resolveTokenizerPath(spec, fallbackConfig);
    Tokenizer* raw = Tokenizer::createTokenizer(path);
    if (raw == nullptr) {
        printErrorAndExit("failed to load tokenizer from " + path);
    }
    return std::unique_ptr<Tokenizer>(raw);
}

std::string decodeVisibleText(Tokenizer* tokenizer, const std::vector<int>& ids) {
    std::string out;
    out.reserve(ids.size() * 4);
    for (int id : ids) {
        if (tokenizer->is_special(id)) {
            continue;
        }
        out += tokenizer->decode(id);
    }
    return out;
}

std::vector<Span> tokenOffsets(Tokenizer* tokenizer, const std::vector<int>& ids) {
    std::vector<Span> offsets;
    offsets.reserve(ids.size());
    int cursor = 0;
    for (int id : ids) {
        if (tokenizer->is_special(id)) {
            offsets.push_back({cursor, cursor});
            continue;
        }
        std::string piece = tokenizer->decode(id);
        int start = cursor;
        cursor += static_cast<int>(piece.size());
        offsets.push_back({start, cursor});
    }
    return offsets;
}

bool spansOverlap(int leftStart, int leftEnd, int rightStart, int rightEnd) {
    return leftStart < rightEnd && rightStart < leftEnd;
}

std::string dumpVectorKey(const std::vector<int>& values) {
    std::ostringstream os;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            os << ',';
        }
        os << values[i];
    }
    return os.str();
}

std::string cacheKey(const std::string& primaryTokenizerPath,
                     const std::string& auxTokenizerPath,
                     const std::string& auxConfigPath,
                     const std::vector<int>& prefixTokenIds,
                     const std::vector<int>& docTokenIds) {
    std::ostringstream os;
    os << primaryTokenizerPath << '\n'
       << auxTokenizerPath << '\n'
       << auxConfigPath << '\n'
       << dumpVectorKey(prefixTokenIds) << '\n'
       << dumpVectorKey(docTokenIds) << '\n';
    return fnv1aHex(os.str());
}

ujson::json spansToJson(const std::vector<Span>& spans) {
    ujson::json out = ujson::json::array();
    for (const auto& span : spans) {
        ujson::json item = ujson::json::array();
        item.push_back(span.start);
        item.push_back(span.end);
        out.push_back(item);
    }
    return out;
}

std::vector<Span> spansFromJson(const ujson::json& value) {
    std::vector<Span> out;
    if (!value.is_array()) {
        return out;
    }
    out.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        if (!value[i].is_array() || value[i].size() < 2) {
            continue;
        }
        out.push_back({value[i][0].get<int>(), value[i][1].get<int>()});
    }
    return out;
}

std::vector<int> encodeWithoutPrefix(Tokenizer* tokenizer, const std::string& text) {
    std::vector<int> encoded = tokenizer->encode(text);
    std::vector<int> prefixOnly = tokenizer->encode("");
    if (encoded.size() >= prefixOnly.size()) {
        encoded.erase(encoded.begin(), encoded.begin() + static_cast<long>(prefixOnly.size()));
    }
    return encoded;
}

CachedChunkMeta loadOrCreateChunkMeta(const ujson::json& request,
                                      Tokenizer* primaryTokenizer,
                                      Tokenizer* auxTokenizer,
                                      const std::string& auxConfigPath,
                                      const std::string& primaryTokenizerPath,
                                      const std::string& auxTokenizerPath,
                                      const fs::path& cacheRoot) {
    std::vector<int> prefixTokenIds = jsonIntArray(request, "prefix_token_ids");
    std::vector<int> docTokenIds = jsonIntArray(request, "doc_token_ids");
    std::string key = cacheKey(primaryTokenizerPath, auxTokenizerPath, auxConfigPath, prefixTokenIds, docTokenIds);
    fs::path metaPath = cacheRoot / (key + ".json");

    CachedChunkMeta meta;
    meta.cacheHit = fileExists(metaPath);
    if (meta.cacheHit) {
        auto cached = parseJsonFile(metaPath);
        meta.auxInputIds = jsonIntArray(cached, "aux_input_ids");
        meta.auxChunkIndices = jsonIntArray(cached, "aux_chunk_indices");
        meta.auxOffsets = spansFromJson(cached["aux_offsets"]);
        meta.primaryTokenOffsets = spansFromJson(cached["primary_token_offsets"]);
        meta.chunkCharStart = jsonInt(cached, "chunk_char_start", 0);
        meta.pastTokenCount = jsonInt(cached, "past_token_count", static_cast<int>(meta.auxInputIds.size()));
        if (!meta.auxInputIds.empty()) {
            return meta;
        }
        meta.cacheHit = false;
    }

    std::string prefixText = decodeVisibleText(primaryTokenizer, prefixTokenIds);
    std::string docText = decodeVisibleText(primaryTokenizer, docTokenIds);
    std::string combinedText = prefixText + docText;
    meta.chunkCharStart = static_cast<int>(prefixText.size());
    int chunkCharEnd = meta.chunkCharStart + static_cast<int>(docText.size());
    meta.primaryTokenOffsets = tokenOffsets(primaryTokenizer, docTokenIds);
    meta.auxInputIds = auxTokenizer->encode(combinedText);
    meta.auxOffsets = tokenOffsets(auxTokenizer, meta.auxInputIds);
    meta.pastTokenCount = static_cast<int>(meta.auxInputIds.size());
    for (int i = 0; i < static_cast<int>(meta.auxOffsets.size()); ++i) {
        const auto& span = meta.auxOffsets[static_cast<size_t>(i)];
        if (span.end <= span.start) {
            continue;
        }
        if (spansOverlap(span.start, span.end, meta.chunkCharStart, chunkCharEnd)) {
            meta.auxChunkIndices.emplace_back(i);
        }
    }
    if (meta.auxChunkIndices.empty()) {
        printErrorAndExit("CacheClip auxiliary tokenizer produced no chunk tokens for the cached document span");
    }

    ujson::json out;
    out["aux_input_ids"] = ujson::json::array();
    for (int value : meta.auxInputIds) {
        out["aux_input_ids"].push_back(value);
    }
    out["aux_chunk_indices"] = ujson::json::array();
    for (int value : meta.auxChunkIndices) {
        out["aux_chunk_indices"].push_back(value);
    }
    out["aux_offsets"] = spansToJson(meta.auxOffsets);
    out["primary_token_offsets"] = spansToJson(meta.primaryTokenOffsets);
    out["chunk_char_start"] = meta.chunkCharStart;
    out["past_token_count"] = meta.pastTokenCount;
    std::ofstream os(metaPath);
    os << out.dump();
    return meta;
}

std::vector<float> attentionScoresForChunk(const float* data,
                                           int heads,
                                           int queryCount,
                                           int kvCount,
                                           int queryStart,
                                           const std::vector<int>& auxChunkIndices) {
    std::vector<float> scores(auxChunkIndices.size(), 0.0f);
    if (heads <= 0 || queryCount <= 0 || kvCount <= 0 || auxChunkIndices.empty()) {
        return scores;
    }
    int usedQueryStart = std::max(0, std::min(queryStart, queryCount));
    int usedQueryCount = std::max(1, queryCount - usedQueryStart);
    float scale = 1.0f / static_cast<float>(heads * usedQueryCount);
    for (size_t chunkIndex = 0; chunkIndex < auxChunkIndices.size(); ++chunkIndex) {
        int kvIndex = auxChunkIndices[chunkIndex];
        if (kvIndex < 0 || kvIndex >= kvCount) {
            continue;
        }
        float acc = 0.0f;
        for (int head = 0; head < heads; ++head) {
            for (int query = usedQueryStart; query < queryCount; ++query) {
                size_t offset = static_cast<size_t>(head) * queryCount * kvCount +
                                static_cast<size_t>(query) * kvCount +
                                static_cast<size_t>(kvIndex);
                acc += data[offset];
            }
        }
        scores[chunkIndex] = acc * scale;
    }
    return scores;
}

std::vector<float> projectAuxScoresToPrimary(const std::vector<float>& auxScores,
                                             const std::vector<int>& auxChunkIndices,
                                             const std::vector<Span>& auxOffsets,
                                             const std::vector<Span>& primaryTokenOffsets,
                                             int chunkCharStart) {
    std::vector<float> projected;
    projected.reserve(primaryTokenOffsets.size());
    for (const auto& tokenSpan : primaryTokenOffsets) {
        int globalStart = chunkCharStart + tokenSpan.start;
        int globalEnd = chunkCharStart + tokenSpan.end;
        float best = 0.0f;
        for (size_t i = 0; i < auxChunkIndices.size() && i < auxScores.size(); ++i) {
            int auxIndex = auxChunkIndices[i];
            if (auxIndex < 0 || auxIndex >= static_cast<int>(auxOffsets.size())) {
                continue;
            }
            const auto& auxSpan = auxOffsets[static_cast<size_t>(auxIndex)];
            if (spansOverlap(auxSpan.start, auxSpan.end, globalStart, globalEnd)) {
                best = std::max(best, auxScores[i]);
            }
        }
        projected.emplace_back(best);
    }
    return projected;
}

int recomputeCount(int tokenCount, double ratio, int minTokens) {
    if (tokenCount <= 0 || ratio <= 0.0) {
        return 0;
    }
    int count = static_cast<int>(std::floor(static_cast<double>(tokenCount) * std::clamp(ratio, 0.0, 1.0)));
    count = std::max(count, std::max(1, minTokens));
    return std::min(count, tokenCount);
}

std::vector<int> selectTopRatioOffsets(const std::vector<float>& scores, double ratio, int minTokens) {
    std::vector<std::pair<float, int>> indexed;
    indexed.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        indexed.emplace_back(scores[i], static_cast<int>(i));
    }
    int count = recomputeCount(static_cast<int>(indexed.size()), ratio, minTokens);
    if (count <= 0) {
        return {};
    }
    std::sort(indexed.begin(), indexed.end(), [](const auto& left, const auto& right) {
        if (left.first != right.first) {
            return left.first > right.first;
        }
        return left.second < right.second;
    });
    std::vector<int> selected;
    selected.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        selected.emplace_back(indexed[static_cast<size_t>(i)].second);
    }
    std::sort(selected.begin(), selected.end());
    return selected;
}

std::vector<int> slidingWindowGroupOffsets(const std::vector<int>& candidates,
                                           int tokenCount,
                                           int windowTokens,
                                           int minCandidates) {
    if (tokenCount <= 0 || candidates.empty()) {
        return {};
    }
    int window = std::max(1, windowTokens);
    int threshold = std::max(1, minCandidates);
    std::vector<int> dedup = candidates;
    dedup.erase(std::unique(dedup.begin(), dedup.end()), dedup.end());
    std::vector<int> groupedFlags(static_cast<size_t>(tokenCount), 0);
    std::vector<int> candidateFlags(static_cast<size_t>(tokenCount), 0);
    for (int value : dedup) {
        if (value >= 0 && value < tokenCount) {
            candidateFlags[static_cast<size_t>(value)] = 1;
        }
    }
    for (int start : dedup) {
        if (start < 0 || start >= tokenCount) {
            continue;
        }
        int end = std::min(tokenCount, start + window);
        int density = 0;
        for (int offset = start; offset < end; ++offset) {
            density += candidateFlags[static_cast<size_t>(offset)];
        }
        if (density >= threshold) {
            for (int offset = start; offset < end; ++offset) {
                groupedFlags[static_cast<size_t>(offset)] = 1;
            }
        }
    }
    std::vector<int> grouped;
    for (int i = 0; i < tokenCount; ++i) {
        if (groupedFlags[static_cast<size_t>(i)] != 0) {
            grouped.emplace_back(i);
        }
    }
    return grouped;
}

std::string jsonEscape(const std::string& text) {
    return ujson::json(text).dump();
}

[[noreturn]] void printErrorAndExit(const std::string& error) {
    ujson::json out;
    out["error"] = error;
    std::cout << out.dump() << std::endl;
    std::exit(1);
}

} // namespace

int main() {
        ujson::json request = ujson::json::parse(readAllStdin());
        if (request.is_null()) {
            printErrorAndExit("invalid selector payload");
        }

        const int groupWindowTokens = jsonInt(request, "group_window_tokens", 8);
        const int groupMinCandidates = jsonInt(request, "group_min_candidates", 5);
        if (groupMinCandidates > groupWindowTokens) {
            printErrorAndExit("group_min_candidates must be <= group_window_tokens");
        }

        const int64_t bootstrapStartUs = monotonicUs();
        const std::string auxConfigPath = resolveConfigPath(jsonString(request, "aux_model"));
        const std::string primaryTokenizerPath = resolveTokenizerPath(jsonString(request, "primary_tokenizer"));
        const std::string auxTokenizerPath = resolveTokenizerPath(
            jsonString(request, "aux_tokenizer"),
            auxConfigPath);

        auto primaryTokenizer = loadTokenizer(primaryTokenizerPath);
        auto auxTokenizer = loadTokenizer(auxTokenizerPath, auxConfigPath);

        std::unique_ptr<Llm, void (*)(Llm*)> llm(Llm::createLLM(auxConfigPath), Llm::destroy);
        if (!llm) {
            printErrorAndExit("failed to create auxiliary MNN LLM from " + auxConfigPath);
        }
        int threadNum = 4;
        if (const char* env = std::getenv("CACHECLIP_AUX_THREAD_NUM")) {
            int parsed = std::atoi(env);
            if (parsed > 0) {
                threadNum = parsed;
            }
        }
        bool enablePrefixCache = false;
        if (const char* env = std::getenv("CACHECLIP_ENABLE_AUX_PREFIX_CACHE")) {
            enablePrefixCache = std::atoi(env) != 0;
        }
        fs::path cacheRoot(jsonString(request, "aux_cache_root"));
        fs::create_directories(cacheRoot);
        std::ostringstream configOs;
        configOs << "{\"backend_type\":\"cpu\",\"thread_num\":" << threadNum
                 << ",\"cacheclip_query_attention\":true";
        if (enablePrefixCache) {
            configOs << ",\"prefix_cache_path\":"
                     << jsonEscape(cacheRoot.string());
        }
        configOs << "}";
        llm->set_config(configOs.str());
        if (!llm->load()) {
            printErrorAndExit("failed to load auxiliary MNN model from " + auxConfigPath);
        }
        const double bootstrapElapsed = elapsedSeconds(bootstrapStartUs);

        const int64_t startedUs = monotonicUs();
        const int64_t cacheStartedUs = monotonicUs();
        CachedChunkMeta chunkMeta = loadOrCreateChunkMeta(
            request,
            primaryTokenizer.get(),
            auxTokenizer.get(),
            auxConfigPath,
            primaryTokenizerPath,
            auxTokenizerPath,
            cacheRoot);
        double cachePrepareElapsed = elapsedSeconds(cacheStartedUs);

        std::string queryText = decodeVisibleText(primaryTokenizer.get(), jsonIntArray(request, "query_token_ids"));
        std::vector<int> queryIds = encodeWithoutPrefix(auxTokenizer.get(), queryText);
        int queryTokenCount = static_cast<int>(queryIds.size());
        if (queryTokenCount <= 0) {
            ujson::json out;
            out["selected_pic_local_indices"] = ujson::json::array();
            out["selector_latency_s"] = 0.0;
            out["selector_cache_hit"] = chunkMeta.cacheHit;
            out["candidate_count"] = 0;
            out["selected_count"] = 0;
            out["cache_path"] = (cacheRoot / (cacheKey(primaryTokenizerPath, auxTokenizerPath, auxConfigPath,
                                                       jsonIntArray(request, "prefix_token_ids"),
                                                       jsonIntArray(request, "doc_token_ids")) + ".json")).string();
            out["aux_device"] = "cpu";
            out["aux_model"] = auxConfigPath;
            out["primary_tokenizer"] = primaryTokenizerPath;
            ujson::json timings;
            timings["bootstrap_load_s"] = bootstrapElapsed;
            timings["cache_prepare_s"] = cachePrepareElapsed;
            timings["query_attention_s"] = 0.0;
            timings["projection_s"] = 0.0;
            timings["ranking_grouping_s"] = 0.0;
            timings["total_s"] = 0.0;
            out["timings"] = timings;
            std::cout << out.dump() << std::endl;
            return 0;
        }

        const int auxMaxLength = jsonInt(request, "aux_max_length", 0);
        if (auxMaxLength > 0 && chunkMeta.pastTokenCount + queryTokenCount > auxMaxLength) {
            printErrorAndExit("CacheClip auxiliary input exceeds aux_max_length");
        }

        std::string key = cacheKey(primaryTokenizerPath, auxTokenizerPath, auxConfigPath,
                                   jsonIntArray(request, "prefix_token_ids"),
                                   jsonIntArray(request, "doc_token_ids"));
        llm->reset();
        bool prefixCacheExists = false;
        if (enablePrefixCache) {
            prefixCacheExists = llm->setPrefixCacheFile(key);
        }
        llm->generate(chunkMeta.auxInputIds, 0);
        cachePrepareElapsed = elapsedSeconds(cacheStartedUs);

        const int queryTailTokens = jsonInt(request, "query_tail_tokens", 0);
        const int queryStart = queryTailTokens > 0 ? std::max(0, queryTokenCount - queryTailTokens) : 0;
        const int64_t queryStartedUs = monotonicUs();
        llm->generate(queryIds, 0);
        const double queryElapsed = elapsedSeconds(queryStartedUs);

        int attnIndex = llm->getOutputIndex("cacheclip_query_attention");
        const auto outputs = llm->getOutputs();
        if (attnIndex < 0 || static_cast<size_t>(attnIndex) >= outputs.size() || outputs[static_cast<size_t>(attnIndex)] == nullptr) {
            printErrorAndExit("cacheclip_query_attention output missing; export aux model with --cacheclip_query_attention");
        }
        auto attentionVar = outputs[static_cast<size_t>(attnIndex)];
        auto* tensor = const_cast<MNN::Tensor*>(attentionVar->getTensor());
        if (tensor == nullptr) {
            printErrorAndExit("cacheclip attention tensor is null");
        }
        tensor->wait(MNN::Tensor::MAP_TENSOR_READ, true);
        auto* info = attentionVar->getInfo();
        if (info == nullptr || info->dim.size() != 4) {
            printErrorAndExit("cacheclip attention tensor shape must be [B,H,Q,KV]");
        }
        int batch = info->dim[0];
        int heads = info->dim[1];
        int qLen = info->dim[2];
        int kvLen = info->dim[3];
        if (batch != 1) {
            printErrorAndExit("cacheclip attention tensor batch must be 1");
        }
        const float* attentionData = attentionVar->readMap<float>();
        if (attentionData == nullptr) {
            printErrorAndExit("failed to map cacheclip attention tensor");
        }

        const int64_t projectStartedUs = monotonicUs();
        std::vector<float> auxScores = attentionScoresForChunk(
            attentionData,
            heads,
            qLen,
            kvLen,
            queryStart,
            chunkMeta.auxChunkIndices);
        std::vector<float> projected = projectAuxScoresToPrimary(
            auxScores,
            chunkMeta.auxChunkIndices,
            chunkMeta.auxOffsets,
            chunkMeta.primaryTokenOffsets,
            chunkMeta.chunkCharStart);
        const double projectElapsed = elapsedSeconds(projectStartedUs);

        const int64_t selectStartedUs = monotonicUs();
        std::vector<int> candidateOffsets = selectTopRatioOffsets(
            projected,
            jsonDouble(request, "ratio", 0.0),
            jsonInt(request, "min_tokens", 1));
        std::vector<int> selectedOffsets = slidingWindowGroupOffsets(
            candidateOffsets,
            static_cast<int>(projected.size()),
            groupWindowTokens,
            groupMinCandidates);
        const double selectElapsed = elapsedSeconds(selectStartedUs);
        const double totalElapsed = elapsedSeconds(startedUs);

        ujson::json out;
        out["selected_pic_local_indices"] = ujson::json::array();
        for (int value : selectedOffsets) {
            out["selected_pic_local_indices"].push_back(value);
        }
        out["selector_latency_s"] = totalElapsed;
        out["selector_cache_hit"] = enablePrefixCache && prefixCacheExists && chunkMeta.cacheHit;
        out["candidate_count"] = static_cast<int>(candidateOffsets.size());
        out["selected_count"] = static_cast<int>(selectedOffsets.size());
        out["cache_path"] = (cacheRoot / (key + ".json")).string();
        out["aux_device"] = "cpu";
        out["aux_model"] = auxConfigPath;
        out["primary_tokenizer"] = primaryTokenizerPath;
        ujson::json timings;
        timings["bootstrap_load_s"] = bootstrapElapsed;
        timings["cache_prepare_s"] = cachePrepareElapsed;
        timings["query_attention_s"] = queryElapsed;
        timings["projection_s"] = projectElapsed;
        timings["ranking_grouping_s"] = selectElapsed;
        timings["total_s"] = totalElapsed;
        out["timings"] = timings;
        std::cout << out.dump() << std::endl;
        return 0;
}
