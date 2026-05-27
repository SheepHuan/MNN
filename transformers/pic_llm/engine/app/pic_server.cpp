//
//  pic_server.cpp
//  MNN
//

#include "pic_server.hpp"

#include "core/PagedKVMeta.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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

struct PreparedPicCache {
    std::string id;
    std::string cacheName;
    std::string placeholder = kDefaultPicPlaceholder;
    std::string selectionAlgorithm = "full-reuse";
    double recomputeRatio = 0.20;
    int scoreLayerIdx = 1;
    std::vector<int> explicitLogicalIndices;
    std::vector<int> explicitPicLocalIndices;
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

float halfToFloat(uint16_t h) {
    const uint16_t hExp = h & 0x7c00u;
    const uint16_t hSig = h & 0x03ffu;
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t out = 0;
    if (hExp == 0) {
        if (hSig == 0) {
            out = sign;
        } else {
            uint16_t sig = hSig;
            int exp = -1;
            do {
                ++exp;
                sig <<= 1;
            } while ((sig & 0x0400u) == 0);
            sig &= 0x03ffu;
            const uint32_t fExp = static_cast<uint32_t>(127 - 15 - exp) << 23;
            out = sign | fExp | (static_cast<uint32_t>(sig) << 13);
        }
    } else if (hExp == 0x7c00u) {
        out = sign | 0x7f800000u | (static_cast<uint32_t>(hSig) << 13);
    } else {
        const uint32_t fExp = static_cast<uint32_t>((hExp >> 10) + (127 - 15)) << 23;
        out = sign | fExp | (static_cast<uint32_t>(hSig) << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

float readRawFloat(const std::vector<int8_t>& data, size_t elementIndex, int dtypeBytes) {
    const size_t byteOffset = elementIndex * static_cast<size_t>(dtypeBytes);
    if (byteOffset + static_cast<size_t>(dtypeBytes) > data.size()) {
        return 0.0f;
    }
    if (dtypeBytes == 4) {
        float value = 0.0f;
        std::memcpy(&value, data.data() + byteOffset, sizeof(value));
        return value;
    }
    uint16_t h = 0;
    std::memcpy(&h, data.data() + byteOffset, sizeof(h));
    return halfToFloat(h);
}

bool readBinaryFile(const fs::path& path, std::vector<int8_t>& data) {
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
    if (!source.is_object() || source.value("content_sha256", "") != contentSha256) {
        return false;
    }
    auto kvLayout = meta.value("kv_layout", json::object());
    if (!kvLayout.is_object() || kvLayout.value("layout", "") != "mnn_paged_attention_raw_v1" ||
        kvLayout.value("key_rope_state", "") != "canonical_no_rope" ||
        jsonInt(kvLayout, "kv_heads", 0) <= 0 || jsonInt(kvLayout, "head_dim", 0) <= 0) {
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
    prepared.scoreLayerIdx = std::max(0, jsonInt(request, "pic_recompute_score_layer_idx", 1));
    prepared.explicitLogicalIndices = jsonIntVector(request.value("pic_recompute_logical_indices", json::array()));
    prepared.explicitPicLocalIndices = jsonIntVector(request.value("pic_recompute_pic_local_indices", json::array()));

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

std::vector<MNN::PagedKVExternalSegment> sliceExternalSegments(
    const std::vector<MNN::PagedKVExternalSegment>& segments, size_t skipTokens) {
    std::vector<MNN::PagedKVExternalSegment> out;
    size_t cursor = 0;
    for (const auto& segment : segments) {
        size_t begin = cursor;
        size_t end = cursor + segment.tokenCount;
        cursor = end;
        if (skipTokens >= end) {
            continue;
        }
        auto sliced = segment;
        size_t localSkip = skipTokens > begin ? skipTokens - begin : 0;
        sliced.sourceTokenOffset += localSkip;
        sliced.tokenCount -= localSkip;
        if (sliced.tokenCount > 0) {
            out.emplace_back(std::move(sliced));
        }
    }
    return out;
}

std::vector<int> selectTopRatioLocalIndices(const std::vector<double>& scores, double ratio);

PicExecutionPlan buildExecutionPlan(const PreparedPicCache& pic, int preludeTokenCount, int layerCount,
                                    const std::vector<double>* scoreValues = nullptr,
                                    const json* scoreMetadata = nullptr) {
    PicExecutionPlan plan;
    plan.selectionAlgorithm = pic.selectionAlgorithm;
    const int picStart = preludeTokenCount;
    const int picLength = static_cast<int>(pic.tokenIds.size());
    const int picEnd = picStart + picLength;
    auto fillAllRecompute = [&]() {
        plan.recomputeLogicalIndices.clear();
        for (int i = picStart; i < picEnd; ++i) {
            plan.recomputeLogicalIndices.emplace_back(i);
            plan.recomputeScores[std::to_string(i)] = 1.0;
        }
        plan.recomputeTokenCount = static_cast<int>(plan.recomputeLogicalIndices.size());
    };
    if (pic.selectionAlgorithm == "full-reuse") {
        plan.plannerName = "FullReusePlanner";
        plan.executionMode = "native-full-reuse";
        plan.scoreLayerIdx = 0;
        plan.externalTokenIds = pic.tokenIds;
        plan.externalSegments = pic.segments;
        plan.metadata = {
            {"score_kind", "none"},
            {"pic_token_count", picLength},
            {"reuse_token_count", picLength},
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
        plan.scoreLayerIdx = std::min(std::max(0, pic.scoreLayerIdx), std::max(0, layerCount - 1));
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
            {"pic_token_count", picLength},
            {"reuse_token_count", picLength - recompute},
            {"native_sparse_recompute", true},
            {"native_sparse_recompute_scope", "python_prefill_layer_plan"},
            {"pre_score_compute_layers", plan.scoreLayerIdx},
            {"pre_score_kv_source", "full_prompt_reference"},
            {"post_score_reuse_kv_source", "cached_pic_kv"},
        };
        return plan;
    }
    plan.plannerName = pic.selectionAlgorithm == "cacheblend" || pic.selectionAlgorithm == "delta-v"
        ? "CacheBlendPlanner"
        : (pic.selectionAlgorithm == "explicit" ? "ExplicitPlanner" : "KvsharePlanner");
    if (scoreValues != nullptr || pic.selectionAlgorithm == "explicit") {
        plan.executionMode = pic.selectionAlgorithm == "cacheblend" || pic.selectionAlgorithm == "delta-v"
            ? "native-cacheblend-sparse-recompute"
            : (pic.selectionAlgorithm == "explicit" ? "native-explicit-sparse-recompute"
                                                     : "native-kvshare-sparse-recompute");
        plan.scoreLayerIdx = std::min(std::max(0, pic.scoreLayerIdx), std::max(0, layerCount - 1));
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
            localIndices = selectTopRatioLocalIndices(*scoreValues, pic.recomputeRatio);
        }
        for (size_t rank = 0; rank < localIndices.size(); ++rank) {
            int local = localIndices[rank];
            int logical = picStart + local;
            plan.recomputeLogicalIndices.emplace_back(logical);
            plan.sparseLogicalIndices.emplace_back(logical);
            plan.sparseTokenIds.emplace_back(pic.tokenIds[local]);
            double score = scoreValues != nullptr && local >= 0 && local < static_cast<int>(scoreValues->size())
                ? (*scoreValues)[local]
                : static_cast<double>(localIndices.size() - rank);
            plan.recomputeScores[std::to_string(logical)] = score;
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
            {"pre_score_kv_source", "full_prompt_reference"},
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
    plan.executionMode = "full-compute-fallback";
    plan.fullCompute = true;
    plan.scoreLayerIdx = layerCount;
    plan.fallbackReason =
        "MNN C++ PIC server does not yet expose HF-style full prompt Q/K/V scoring and sparse hidden-state recompute";
    fillAllRecompute();
    plan.prefillPicTokenIds = pic.tokenIds;
    plan.metadata = {
        {"score_kind", pic.selectionAlgorithm == "cacheblend" || pic.selectionAlgorithm == "delta-v"
                           ? "layer_value_delta_mean_abs_unavailable"
                           : "attention_output_error_kv_first_order_influence_unavailable"},
        {"pic_token_count", picLength},
        {"reuse_token_count", 0},
        {"fallback_reason", plan.fallbackReason},
    };
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

const json* findLayerEntry(const json& kv, int layerIndex) {
    if (!kv.is_array()) {
        return nullptr;
    }
    for (const auto& entry : kv) {
        if (entry.is_object() && jsonInt(entry, "layer_index", -1) == layerIndex) {
            return &entry;
        }
    }
    return nullptr;
}

bool buildFullPromptReference(MNN::Transformer::Llm* llm, const std::string& kvCacheDir,
                              const std::vector<int>& fullTokens, const std::string& scratchName,
                              int layerCount, std::string& fileStem, std::string& error) {
    if (llm == nullptr || fullTokens.empty()) {
        error = "Full prompt reference needs a loaded LLM and non-empty token list";
        return false;
    }
    fileStem = pathJoinForMNN(pathJoinForMNN(pathJoinForMNN("scratch", sanitizeCachePart(scratchName)), "layers"), "full");
    fs::path scratchRoot = fs::path(kvCacheDir) / "scratch" / sanitizeCachePart(scratchName);
    std::error_code ec;
    fs::remove_all(scratchRoot, ec);
    fs::create_directories(scratchRoot / "layers", ec);
    if (ec) {
        error = "Failed to create full prompt scratch directory: " + scratchRoot.string();
        return false;
    }
    llm->reset();
    llm->setPrefixCacheFile(fileStem);
    std::ostringstream sink;
    llm->response(fullTokens, &sink, "", 0);
    llm->reset();
    for (int layer = 0; layer < layerCount; ++layer) {
        fs::path valuePath = fs::path(kvCacheDir) / (fileStem + "_" + std::to_string(layer) + ".v");
        if (!fileExistsNonEmpty(valuePath)) {
            error = "Full prompt reference did not export layer " + std::to_string(layer) + ": " + valuePath.string();
            return false;
        }
    }
    return true;
}

bool scorePicKvDelta(const PreparedPicCache& pic, const std::string& kvCacheDir, const std::string& fullFileStem,
                     int fullTokenCount, int picStart, int scoreLayerIdx, bool includeKey, bool includeValue,
                     std::vector<double>& scores, json& metadata, std::string& error) {
    scores.assign(pic.tokenIds.size(), 0.0);
    if (pic.tokenIds.empty()) {
        return true;
    }
    fs::path fullKeyPath = fs::path(kvCacheDir) / (fullFileStem + "_" + std::to_string(scoreLayerIdx) + ".k");
    fs::path fullValuePath = fs::path(kvCacheDir) / (fullFileStem + "_" + std::to_string(scoreLayerIdx) + ".v");
    fs::path fullShapePath = fs::path(kvCacheDir) / (fullFileStem + "_" + std::to_string(scoreLayerIdx) + ".json");
    auto fullShape = readJsonFile(fullShapePath);
    if (!fullShape.is_object()) {
        error = "Missing full prompt reference shape sidecar: " + fullShapePath.string();
        return false;
    }
    int batch = jsonInt(fullShape, "batch", 1);
    int kvHeads = jsonInt(fullShape, "kv_heads", 0);
    int headDim = jsonInt(fullShape, "head_dim", 0);
    int dtypeBytes = jsonInt(fullShape, "dtype_bytes", 0);
    if (batch <= 0 || kvHeads <= 0 || headDim <= 0 || dtypeBytes <= 0) {
        error = "Full prompt reference shape sidecar is incomplete: " + fullShapePath.string();
        return false;
    }
    std::vector<int8_t> fullKey;
    std::vector<int8_t> fullValue;
    if (includeKey && !readBinaryFile(fullKeyPath, fullKey)) {
        error = "Failed to read full prompt key reference: " + fullKeyPath.string();
        return false;
    }
    if (includeValue && !readBinaryFile(fullValuePath, fullValue)) {
        error = "Failed to read full prompt value reference: " + fullValuePath.string();
        return false;
    }
    const double invDen = 1.0 / static_cast<double>(std::max(1, batch * kvHeads * headDim));
    size_t picCursor = 0;
    for (size_t segmentIndex = 0; segmentIndex < pic.segments.size(); ++segmentIndex) {
        const auto& segment = pic.segments[segmentIndex];
        const auto& textCache = pic.textCaches[segmentIndex];
        json kvEntries = textCache.value("kv", json::array());
        auto layerEntry = findLayerEntry(kvEntries, scoreLayerIdx);
        if (layerEntry == nullptr) {
            error = "Text cache is missing score layer " + std::to_string(scoreLayerIdx);
            return false;
        }
        if (segment.batch != batch || segment.kvHeads != kvHeads || segment.headDim != headDim ||
            segment.dtypeBytes != dtypeBytes) {
            error = "Text cache shape does not match full prompt reference at score layer";
            return false;
        }
        std::vector<int8_t> cachedKey;
        std::vector<int8_t> cachedValue;
        if (includeKey && !readBinaryFile(layerEntry->value("key_path", ""), cachedKey)) {
            error = "Failed to read cached PIC key for score layer";
            return false;
        }
        if (includeValue && !readBinaryFile(layerEntry->value("value_path", ""), cachedValue)) {
            error = "Failed to read cached PIC value for score layer";
            return false;
        }
        const size_t sourceTokenCount = segment.sourceTokenCount > 0 ? segment.sourceTokenCount : segment.tokenCount;
        for (size_t local = 0; local < segment.tokenCount; ++local) {
            const size_t globalPicLocal = picCursor + local;
            const int fullToken = picStart + static_cast<int>(globalPicLocal);
            const size_t sourceToken = segment.sourceTokenOffset + local;
            if (fullToken < 0 || fullToken >= fullTokenCount || sourceToken >= sourceTokenCount) {
                error = "PIC scoring token index is out of range";
                return false;
            }
            double score = 0.0;
            for (int b = 0; b < batch; ++b) {
                for (int h = 0; h < kvHeads; ++h) {
                    for (int d = 0; d < headDim; ++d) {
                        if (includeValue) {
                            const size_t fullOffset =
                                ((static_cast<size_t>(b) * kvHeads + h) * fullTokenCount + fullToken) *
                                    headDim +
                                d;
                            const size_t cachedOffset =
                                ((static_cast<size_t>(b) * kvHeads + h) * sourceTokenCount + sourceToken) *
                                    headDim +
                                d;
                            score += std::abs(readRawFloat(fullValue, fullOffset, dtypeBytes) -
                                              readRawFloat(cachedValue, cachedOffset, dtypeBytes));
                        }
                        if (includeKey) {
                            const size_t fullOffset =
                                ((static_cast<size_t>(fullToken) * batch + b) * kvHeads + h) * headDim + d;
                            const size_t cachedOffset =
                                ((sourceToken * static_cast<size_t>(batch) + b) * kvHeads + h) * headDim + d;
                            score += std::abs(readRawFloat(fullKey, fullOffset, dtypeBytes) -
                                              readRawFloat(cachedKey, cachedOffset, dtypeBytes));
                        }
                    }
                }
            }
            scores[globalPicLocal] = score * invDen;
        }
        picCursor += segment.tokenCount;
    }
    metadata["score_layer_idx"] = scoreLayerIdx;
    metadata["score_source"] = "full_prompt_reference_minus_cached_pic_kv";
    metadata["score_dtype_bytes"] = dtypeBytes;
    metadata["score_batch"] = batch;
    metadata["score_kv_heads"] = kvHeads;
    metadata["score_head_dim"] = headDim;
    return true;
}

bool attachFullReferenceBeforeScoreLayer(PicExecutionPlan& plan, const std::string& kvCacheDir,
                                         const std::string& fullFileStem, int fullTokenCount, int picStart,
                                         std::string& error) {
    if (plan.fullCompute || plan.externalSegments.empty() || plan.scoreLayerIdx <= 0) {
        return true;
    }
    if (fullFileStem.empty() || fullTokenCount <= 0) {
        error = "Sparse PIC execution needs full prompt reference KV before score layer";
        return false;
    }
    size_t picCursor = 0;
    for (auto& segment : plan.externalSegments) {
        for (auto& layer : segment.layers) {
            if (layer.layerIndex >= plan.scoreLayerIdx) {
                continue;
            }
            fs::path keyPath = fs::path(kvCacheDir) / (fullFileStem + "_" + std::to_string(layer.layerIndex) + ".k");
            fs::path valuePath = fs::path(kvCacheDir) / (fullFileStem + "_" + std::to_string(layer.layerIndex) + ".v");
            if (!fileExistsNonEmpty(keyPath) || !fileExistsNonEmpty(valuePath)) {
                error = "Missing full prompt reference KV for pre-score layer " + std::to_string(layer.layerIndex);
                return false;
            }
            layer.keyPath = absoluteString(keyPath);
            layer.valuePath = absoluteString(valuePath);
            layer.hasSourceOverride = true;
            layer.sourceTokenOffset = static_cast<size_t>(picStart) + picCursor;
            layer.sourceTokenCount = static_cast<size_t>(fullTokenCount);
        }
        picCursor += segment.tokenCount;
    }
    plan.metadata["pre_score_compute_layers"] = plan.scoreLayerIdx;
    plan.metadata["pre_score_kv_source"] = "full_prompt_reference";
    return true;
}

std::vector<int> selectTopRatioLocalIndices(const std::vector<double>& scores, double ratio) {
    std::vector<int> indices(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        indices[i] = static_cast<int>(i);
    }
    if (scores.empty() || ratio <= 0.0) {
        return {};
    }
    int topK = std::min(static_cast<int>(scores.size()),
                        std::max(1, static_cast<int>(std::ceil(scores.size() * ratio))));
    std::stable_sort(indices.begin(), indices.end(), [&](int lhs, int rhs) {
        if (scores[lhs] == scores[rhs]) {
            return lhs < rhs;
        }
        return scores[lhs] > scores[rhs];
    });
    indices.resize(topK);
    std::sort(indices.begin(), indices.end());
    return indices;
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
    std::string config = std::string("{\"tmp_path\":\"tmp\",\"prefix_cache_path\":") +
                         jsonEscape(mConfig.kvCacheDir) + "}";
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
        "/v1/prefill/text",
        "/v1/kv/pic_caches",
        "/v1/chat/completions",
        "/chat/completions",
    });
    res.set_content(body.dump(2), "application/json");
}

void PicServer::handleHealth(const httplib::Request&, httplib::Response& res) {
    allowCors(res);
    res.set_content(json({{"status", "ok"}, {"model", mConfig.servedModelName}}).dump(), "application/json");
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
    res.set_content(body.dump(2), "application/json");
}

void PicServer::handleReset(const httplib::Request&, httplib::Response& res) {
    allowCors(res);
    std::lock_guard<std::mutex> lock(mMutex);
    if (mLlm) {
        mLlm->reset();
    }
    res.set_content(json({{"status", "ok"}}).dump(), "application/json");
}

void PicServer::handlePrefillText(const httplib::Request& req, httplib::Response& res) {
    allowCors(res);
    if (!json::accept(req.body)) {
        res.status = 400;
        res.set_content(json({{"error", "Invalid JSON in request body"}}).dump(), "application/json");
        return;
    }
    auto request = json::parse(req.body, nullptr, false);
    json response;
    std::string error;
    bool ok = false;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        ok = buildTextCache(request, response, error);
    }
    if (!ok) {
        res.status = 400;
        res.set_content(json({{"error", error}}).dump(), "application/json");
        return;
    }
    res.set_content(response.dump(2), "application/json");
}

void PicServer::handlePicCaches(const httplib::Request& req, httplib::Response& res) {
    allowCors(res);
    if (!json::accept(req.body)) {
        res.status = 400;
        res.set_content(json({{"error", "Invalid JSON in request body"}}).dump(), "application/json");
        return;
    }
    auto request = json::parse(req.body, nullptr, false);
    json response;
    std::string error;
    bool ok = false;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        ok = buildPicCache(request, response, error);
    }
    if (!ok) {
        res.status = 400;
        res.set_content(json({{"error", error}}).dump(), "application/json");
        return;
    }
    res.set_content(response.dump(2), "application/json");
}

void PicServer::handleChatCompletions(const httplib::Request& req, httplib::Response& res) {
    allowCors(res);
    if (!json::accept(req.body)) {
        res.status = 400;
        res.set_content(json({{"error", "Invalid JSON in request body"}}).dump(), "application/json");
        return;
    }
    auto request = json::parse(req.body, nullptr, false);
    json response;
    std::string error;
    bool ok = false;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        ok = completeChat(request, response, error);
    }
    if (!ok) {
        res.status = 500;
        res.set_content(json({{"error", error}}).dump(), "application/json");
        return;
    }
    res.set_content(response.dump(2), "application/json");
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
    if (!request.contains("content") || !request["content"].is_string() ||
        request["content"].get<std::string>().empty()) {
        error = "Text cache request requires non-empty string field `content`";
        return false;
    }

    const std::string id = request["id"].get<std::string>();
    const std::string content = request["content"].get<std::string>();
    if (looksLikeFilePath(content)) {
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

    auto tokenIds = mLlm->tokenizer_encode(content);
    if (tokenIds.empty()) {
        error = "Text tokenization produced no tokens";
        return false;
    }
    const std::string contentSha256 = sha256Hex(content);

    if (!force && existingCacheMatches(metaPath, contentSha256, tokenIds)) {
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
        {"source", {{"kind", "inline"}, {"content_sha256", contentSha256}}},
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
    KvShape shape;
    if (!pic.segments.empty()) {
        const auto& segment = pic.segments.front();
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
    }
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

bool PicServer::completeChat(const json& request, json& response, std::string& error) {
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
    if (!request.contains("messages") || !request["messages"].is_array() || request["messages"].empty()) {
        error = "messages must be a non-empty array";
        return false;
    }
    MNN::Transformer::ChatMessages messages;
    for (const auto& item : request["messages"]) {
        if (!item.is_object() || !item.contains("role") || !item["role"].is_string() ||
            !item.contains("content") || !item["content"].is_string()) {
            error = "Each message must contain string role/content";
            return false;
        }
        messages.emplace_back(item["role"].get<std::string>(), item["content"].get<std::string>());
    }
    int maxTokens = jsonInt(request, "max_tokens", -1);
    if (maxTokens == 0) {
        error = "max_tokens must be positive when provided";
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
        mLlm->reset();
        mLlm->response(messages, &sink, "", maxTokens);
        auto context = mLlm->getContext();
        if (context != nullptr) {
            outputTokens = context->output_tokens;
            promptTokens = context->prompt_len;
        }
    } else {
        PreparedPicCache pic;
        if (!preparePicCacheFromRequest(mConfig.kvCacheDir, runtimeBackend(), request["pic_cache"], pic, error)) {
            return false;
        }
        std::vector<int> preludeTokenIds;
        std::vector<int> suffixTokenIds;
        bool hasPlaceholder = false;
        std::string promptProtocol = "legacy-system-prelude";
        std::string rendered = mLlm->apply_chat_template(messages, true);
        std::string placeholder = pic.placeholder.empty() ? kDefaultPicPlaceholder : pic.placeholder;
        auto placeholderPos = rendered.find(placeholder);
        if (placeholderPos != std::string::npos) {
            hasPlaceholder = true;
            promptProtocol = "chat-template-placeholder-v1";
            auto preludeText = rendered.substr(0, placeholderPos);
            auto suffixText = rendered.substr(placeholderPos + placeholder.size());
            if (!preludeText.empty()) {
                preludeTokenIds = mLlm->tokenizer_encode(preludeText);
            }
            suffixTokenIds = mLlm->tokenizer_encode(suffixText);
        } else {
            MNN::Transformer::ChatMessages systemMessages;
            MNN::Transformer::ChatMessages promptMessages;
            for (const auto& message : messages) {
                if (message.first == "system") {
                    systemMessages.emplace_back(message);
                } else {
                    promptMessages.emplace_back(message);
                }
            }
            if (promptMessages.empty()) {
                error = "pic_cache requires at least one non-system message";
                return false;
            }
            if (!systemMessages.empty()) {
                preludeTokenIds = mLlm->tokenizer_encode(mLlm->apply_chat_template(systemMessages, false));
            }
            suffixTokenIds = mLlm->tokenizer_encode(mLlm->apply_chat_template(promptMessages, true));
            if (!preludeTokenIds.empty()) {
                auto emptyPrefix = mLlm->tokenizer_encode("");
                if (!emptyPrefix.empty() && suffixTokenIds.size() >= emptyPrefix.size() &&
                    std::equal(emptyPrefix.begin(), emptyPrefix.end(), suffixTokenIds.begin())) {
                    suffixTokenIds.erase(suffixTokenIds.begin(), suffixTokenIds.begin() + emptyPrefix.size());
                }
            }
        }
        if (suffixTokenIds.empty()) {
            error = "PIC prompt suffix tokenization produced no tokens";
            return false;
        }
        std::vector<int> fullPromptForScoring;
        fullPromptForScoring.reserve(preludeTokenIds.size() + pic.tokenIds.size() + suffixTokenIds.size());
        fullPromptForScoring.insert(fullPromptForScoring.end(), preludeTokenIds.begin(), preludeTokenIds.end());
        fullPromptForScoring.insert(fullPromptForScoring.end(), pic.tokenIds.begin(), pic.tokenIds.end());
        fullPromptForScoring.insert(fullPromptForScoring.end(), suffixTokenIds.begin(), suffixTokenIds.end());

        std::vector<double> scoreValues;
        json scoreMetadata = json::object();
        bool hasNativeScores = false;
        std::string fullFileStem;
        auto ensureFullPromptReference = [&](const std::string& reason) -> bool {
            if (!fullFileStem.empty()) {
                return true;
            }
            std::ostringstream tokenDigestInput;
            for (int token : fullPromptForScoring) {
                tokenDigestInput << token << ",";
            }
            std::string scratchName = sanitizeCachePart(reason) + "_" + sanitizeCachePart(pic.id) + "_" +
                                      sha256Hex(tokenDigestInput.str()).substr(0, 12);
            return buildFullPromptReference(mLlm.get(), mConfig.kvCacheDir, fullPromptForScoring, scratchName,
                                            layerCount, fullFileStem, error);
        };
        if (pic.selectionAlgorithm == "cacheblend" || pic.selectionAlgorithm == "delta-v" ||
            pic.selectionAlgorithm == "kvshare" || pic.selectionAlgorithm == "delta-a") {
            if (!ensureFullPromptReference("pic_score")) {
                return false;
            }
            const int effectiveScoreLayer =
                std::min(std::max(0, pic.scoreLayerIdx), std::max(0, layerCount - 1));
            const bool includeKey = pic.selectionAlgorithm == "kvshare" || pic.selectionAlgorithm == "delta-a";
            const bool includeValue = true;
            if (!scorePicKvDelta(pic, mConfig.kvCacheDir, fullFileStem,
                                 static_cast<int>(fullPromptForScoring.size()),
                                 static_cast<int>(preludeTokenIds.size()), effectiveScoreLayer,
                                 includeKey, includeValue, scoreValues, scoreMetadata, error)) {
                return false;
            }
            hasNativeScores = true;
        }
        PicExecutionPlan plan = buildExecutionPlan(pic, static_cast<int>(preludeTokenIds.size()), layerCount,
                                                   hasNativeScores ? &scoreValues : nullptr,
                                                   hasNativeScores ? &scoreMetadata : nullptr);
        if (!plan.fullCompute && plan.scoreLayerIdx > 0) {
            if (!ensureFullPromptReference("pic_prescore")) {
                return false;
            }
            if (!attachFullReferenceBeforeScoreLayer(plan, mConfig.kvCacheDir, fullFileStem,
                                                     static_cast<int>(fullPromptForScoring.size()),
                                                     static_cast<int>(preludeTokenIds.size()), error)) {
                return false;
            }
        }
        mLlm->reset();
        mLlm->generate_init(&sink, "");
        if (plan.fullCompute) {
            std::vector<int> fullPrompt;
            fullPrompt.reserve(preludeTokenIds.size() + pic.tokenIds.size() + suffixTokenIds.size());
            fullPrompt.insert(fullPrompt.end(), preludeTokenIds.begin(), preludeTokenIds.end());
            fullPrompt.insert(fullPrompt.end(), pic.tokenIds.begin(), pic.tokenIds.end());
            fullPrompt.insert(fullPrompt.end(), suffixTokenIds.begin(), suffixTokenIds.end());
            outputTokens = mLlm->generate(fullPrompt, maxTokens);
        } else {
            std::vector<int> firstPrefill;
            firstPrefill.reserve(preludeTokenIds.size() + plan.prefillPicTokenIds.size());
            firstPrefill.insert(firstPrefill.end(), preludeTokenIds.begin(), preludeTokenIds.end());
            firstPrefill.insert(firstPrefill.end(), plan.prefillPicTokenIds.begin(), plan.prefillPicTokenIds.end());
            if (!firstPrefill.empty()) {
                mLlm->generate(firstPrefill, 0);
            } else {
                mLlm->beginExternalPagedKVRequest();
            }
            if (!plan.externalTokenIds.empty() &&
                !mLlm->appendExternalPagedKV(plan.externalTokenIds, plan.externalSegments)) {
                mLlm->finishExternalPagedKVRequest();
                error = "Failed to append external PIC KV into paged request";
                return false;
            }
            if (plan.sparseRecompute &&
                !mLlm->recomputeExternalPagedKV(plan.sparseLogicalIndices, plan.sparseTokenIds)) {
                mLlm->finishExternalPagedKVRequest();
                error = "Failed to sparse-recompute selected PIC KV tokens";
                return false;
            }
            outputTokens = mLlm->generate(suffixTokenIds, maxTokens);
            mLlm->finishExternalPagedKVRequest();
        }
        promptTokens = static_cast<int>(suffixTokenIds.size());
        preludeTokensCount = static_cast<int>(preludeTokenIds.size());
        picTokensCount = static_cast<int>(pic.tokenIds.size());

        KvShape shape;
        if (!pic.segments.empty()) {
            const auto& segment = pic.segments.front();
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
        }
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
            {"placeholder", hasPlaceholder ? placeholder : ""},
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
