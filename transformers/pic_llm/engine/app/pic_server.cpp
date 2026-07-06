//
//  pic_server.cpp
//  MNN
//

#include "pic_server.hpp"
#include "pic_server_internal.hpp"

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
#include <unordered_map>
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
    if (algorithm == "cacheclip") {
        return "native-cacheclip-sparse-recompute";
    }
    if (algorithm == "fusionrag_online") {
        return "native-fusionrag-online-sparse-recompute";
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

bool picGraphProfileEnabled() {
    return envInt("MNN_PIC_GRAPH_PROFILE", 0) > 0;
}

std::string envString(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

std::string lowerAscii(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

std::string picGraphProfileScope() {
    return lowerAscii(envString("MNN_PIC_GRAPH_PROFILE_SCOPE"));
}

bool picGraphProfileScopeEnabled(const char* phase) {
    if (!picGraphProfileEnabled()) {
        return false;
    }
    const std::string scope = picGraphProfileScope();
    const std::string phaseName = lowerAscii(phase == nullptr ? "request" : phase);
    if (scope.empty() || scope == "request" || scope == "all") {
        return phaseName == "request";
    }
    if (scope == "decode" || scope == "decode-only") {
        return phaseName == "decode";
    }
    if (scope == "prefill" || scope == "prefill-only") {
        return phaseName == "prefill";
    }
    return phaseName == "request";
}

bool picGraphProfileProgressEnabled() {
    return envInt("MNN_PIC_GRAPH_PROFILE_PROGRESS", 0) > 0;
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

std::string tensorShapes(const std::vector<MNN::Tensor*>& tensors) {
    std::ostringstream os;
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (i > 0) {
            os << "|";
        }
        const auto* tensor = tensors[i];
        if (tensor == nullptr) {
            os << "null";
            continue;
        }
        os << "[";
        const auto shape = tensor->shape();
        for (size_t j = 0; j < shape.size(); ++j) {
            if (j > 0) {
                os << "x";
            }
            os << shape[j];
        }
        os << "]";
    }
    return os.str();
}

struct PicGraphProfileRecord {
    std::string name;
    std::string type;
    std::string inputShapes;
    std::string outputShapes;
    int calls = 0;
    int64_t totalUs = 0;
    int64_t maxUs = 0;
    float flops = 0.0f;
};

struct PicGraphProfilePending {
    bool valid = false;
    int64_t startUs = 0;
    std::string name;
    std::string type;
    std::string inputShapes;
    float flops = 0.0f;
};

class PicGraphProfiler {
public:
    static PicGraphProfiler& get() {
        static PicGraphProfiler profiler;
        return profiler;
    }

    void beginRequest(const std::string& label) {
        if (!picGraphProfileEnabled()) {
            return;
        }
        std::lock_guard<std::mutex> lock(mMutex);
        mActive = true;
        mLabel = label;
        mRequestId++;
        mRecords.clear();
        mTotalUs = 0;
        mTotalCalls = 0;
    }

    void endRequest() {
        if (!picGraphProfileEnabled()) {
            return;
        }
        std::vector<PicGraphProfileRecord> records;
        std::unordered_map<std::string, PicGraphProfileRecord> typeRecords;
        std::string label;
        int requestId = 0;
        int64_t totalUs = 0;
        int totalCalls = 0;
        {
            std::lock_guard<std::mutex> lock(mMutex);
            if (!mActive) {
                return;
            }
            label = mLabel;
            requestId = mRequestId;
            totalUs = mTotalUs;
            totalCalls = mTotalCalls;
            records.reserve(mRecords.size());
            for (const auto& iter : mRecords) {
                records.emplace_back(iter.second);
                auto& typeRecord = typeRecords[iter.second.type];
                typeRecord.type = iter.second.type;
                typeRecord.name = iter.second.type;
                typeRecord.calls += iter.second.calls;
                typeRecord.totalUs += iter.second.totalUs;
                typeRecord.maxUs = std::max(typeRecord.maxUs, iter.second.maxUs);
                typeRecord.flops += iter.second.flops;
            }
            mActive = false;
        }
        std::sort(records.begin(), records.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.totalUs != rhs.totalUs) {
                return lhs.totalUs > rhs.totalUs;
            }
            return lhs.name < rhs.name;
        });
        std::vector<PicGraphProfileRecord> types;
        types.reserve(typeRecords.size());
        for (const auto& iter : typeRecords) {
            types.emplace_back(iter.second);
        }
        std::sort(types.begin(), types.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.totalUs != rhs.totalUs) {
                return lhs.totalUs > rhs.totalUs;
            }
            return lhs.type < rhs.type;
        });
        const int topN = envInt("MNN_PIC_GRAPH_PROFILE_TOP", 40);
        std::fprintf(stderr,
                     "MNN_PIC_GRAPH_PROFILE_SUMMARY request=%d label=%s total_ms=%.3f calls=%d unique_ops=%zu "
                     "unique_types=%zu\n",
                     requestId, label.c_str(), totalUs / 1000.0, totalCalls, records.size(), types.size());
        for (int i = 0; i < static_cast<int>(types.size()) && i < topN; ++i) {
            const auto& record = types[i];
            std::fprintf(stderr,
                         "MNN_PIC_GRAPH_PROFILE_TYPE rank=%d type=%s total_ms=%.3f max_ms=%.3f calls=%d\n",
                         i + 1, record.type.c_str(), record.totalUs / 1000.0, record.maxUs / 1000.0,
                         record.calls);
        }
        for (int i = 0; i < static_cast<int>(records.size()) && i < topN; ++i) {
            const auto& record = records[i];
            std::fprintf(stderr,
                         "MNN_PIC_GRAPH_PROFILE_OP rank=%d type=%s name=%s total_ms=%.3f max_ms=%.3f calls=%d "
                         "inputs=%s outputs=%s\n",
                         i + 1, record.type.c_str(), record.name.c_str(), record.totalUs / 1000.0,
                         record.maxUs / 1000.0, record.calls, record.inputShapes.c_str(),
                         record.outputShapes.c_str());
        }
        std::fflush(stderr);
    }

    bool before(const std::vector<MNN::Tensor*>& inputs, const MNN::OperatorInfo* info) {
        if (!picGraphProfileEnabled() || info == nullptr) {
            return true;
        }
        if (!isActive()) {
            return true;
        }
        auto& pending = pendingOp();
        pending.valid = true;
        pending.startUs = monotonicUs();
        pending.name = info->name();
        pending.type = info->type();
        pending.inputShapes = tensorShapes(inputs);
        pending.flops = info->flops();
        if (picGraphProfileProgressEnabled()) {
            std::fprintf(stderr,
                         "MNN_PIC_GRAPH_PROFILE_PROGRESS phase=begin type=%s name=%s inputs=%s\n",
                         pending.type.c_str(), pending.name.c_str(), pending.inputShapes.c_str());
            std::fflush(stderr);
        }
        return true;
    }

    bool after(const std::vector<MNN::Tensor*>& outputs, const MNN::OperatorInfo* info) {
        if (!picGraphProfileEnabled() || info == nullptr) {
            return true;
        }
        auto& pending = pendingOp();
        if (!pending.valid) {
            return true;
        }
        for (auto* output : outputs) {
            if (output != nullptr) {
                output->wait(MNN::Tensor::MAP_TENSOR_READ, true);
            }
        }
        const int64_t costUs = std::max<int64_t>(0, monotonicUs() - pending.startUs);
        const std::string outputShapes = tensorShapes(outputs);
        if (picGraphProfileProgressEnabled()) {
            std::fprintf(stderr,
                         "MNN_PIC_GRAPH_PROFILE_PROGRESS phase=end type=%s name=%s cost_ms=%.3f inputs=%s outputs=%s\n",
                         pending.type.c_str(), pending.name.c_str(), costUs / 1000.0, pending.inputShapes.c_str(),
                         outputShapes.c_str());
            std::fflush(stderr);
        }
        const std::string key = pending.name + "\n" + pending.type + "\n" + pending.inputShapes + "\n" + outputShapes;
        {
            std::lock_guard<std::mutex> lock(mMutex);
            if (mActive) {
                auto& record = mRecords[key];
                if (record.calls == 0) {
                    record.name = pending.name;
                    record.type = pending.type;
                    record.inputShapes = pending.inputShapes;
                    record.outputShapes = outputShapes;
                }
                record.calls++;
                record.totalUs += costUs;
                record.maxUs = std::max(record.maxUs, costUs);
                record.flops += pending.flops;
                mTotalUs += costUs;
                mTotalCalls++;
            }
        }
        pending.valid = false;
        return true;
    }

private:
    PicGraphProfiler() = default;

    bool isActive() {
        std::lock_guard<std::mutex> lock(mMutex);
        return mActive;
    }

    static PicGraphProfilePending& pendingOp() {
        thread_local PicGraphProfilePending pending;
        return pending;
    }

    std::mutex mMutex;
    bool mActive = false;
    int mRequestId = 0;
    std::string mLabel;
    std::unordered_map<std::string, PicGraphProfileRecord> mRecords;
    int64_t mTotalUs = 0;
    int mTotalCalls = 0;
};

class PicGraphProfileRequestScope {
public:
    explicit PicGraphProfileRequestScope(std::string label, const char* phase = "request")
        : mEnabled(picGraphProfileScopeEnabled(phase)) {
        if (mEnabled) {
            std::ostringstream scopedLabel;
            scopedLabel << "phase=" << (phase == nullptr ? "request" : phase) << "," << label;
            PicGraphProfiler::get().beginRequest(scopedLabel.str());
        }
    }

    PicGraphProfileRequestScope(const PicGraphProfileRequestScope&) = delete;
    PicGraphProfileRequestScope& operator=(const PicGraphProfileRequestScope&) = delete;

    ~PicGraphProfileRequestScope() {
        if (mEnabled) {
            PicGraphProfiler::get().endRequest();
        }
    }

private:
    bool mEnabled = false;
};

std::string picGraphProfileLabel(const json& request) {
    std::ostringstream os;
    os << "mode=";
    if (request.contains("pic_cache") && request["pic_cache"].is_object()) {
        const auto& pic = request["pic_cache"];
        os << (pic.contains("selection_algorithm") && pic["selection_algorithm"].is_string()
                   ? pic["selection_algorithm"].get<std::string>()
                   : "pic");
        os << ",ratio="
           << (pic.contains("pic_recompute_ratio") && pic["pic_recompute_ratio"].is_number()
                   ? pic["pic_recompute_ratio"].get<double>()
                   : 0.0);
        os << ",score_layer="
           << (pic.contains("pic_recompute_score_layer_idx") && pic["pic_recompute_score_layer_idx"].is_number_integer()
                   ? pic["pic_recompute_score_layer_idx"].get<int>()
                   : 1);
    } else {
        os << "full-compute";
    }
    os << ",max_tokens="
       << (request.contains("max_tokens") && request["max_tokens"].is_number_integer()
               ? request["max_tokens"].get<int>()
               : -1);
    for (const char* key : {"full_prompt_token_ids", "prompt_token_ids", "input_token_ids"}) {
        if (request.contains(key) && request[key].is_array()) {
            os << ",tokens=" << request[key].size();
            break;
        }
    }
    return os.str();
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

bool parsePicDecodeRefineConfig(const json& request, PicDecodeRefineConfig& config, std::string& error) {
    config = PicDecodeRefineConfig();
    if (!request.contains("decode_refine")) {
        return true;
    }
    if (!request["decode_refine"].is_object()) {
        error = "decode_refine must be an object when provided";
        return false;
    }
    const json& src = request["decode_refine"];
    config.enabled = jsonBool(src, "enabled", config.enabled);
    config.tokensPerDecodeStep = std::max(0, jsonInt(src, "tokens_per_decode_step",
                                                     config.tokensPerDecodeStep));
    config.topM = std::max(0, jsonInt(src, "top_m", config.topM));
    config.scoreThreshold = std::max(0.0, jsonDouble(src, "score_threshold", config.scoreThreshold));
    config.scoreMargin = std::max(0.0, jsonDouble(src, "score_margin", config.scoreMargin));
    config.scoreDecay = clampDouble(jsonDouble(src, "score_decay", config.scoreDecay), 0.0, 1.0);
    config.selector = jsonString(src, "selector", config.selector);
    config.attentionLayerIdx = jsonInt(src, "attention_layer_idx", config.attentionLayerIdx);
    config.attentionHeadIds = jsonIntVector(src.value("attention_head_ids", json::array()));

    if (config.selector != "top_hkvd" && config.selector != "lagged_attention_hkvd") {
        error = "Unsupported decode refine selector: " + config.selector;
        return false;
    }
    if (config.enabled && config.selector == "lagged_attention_hkvd" && config.attentionLayerIdx < 0) {
        error = "decode_refine selector lagged_attention_hkvd requires attention_layer_idx";
        return false;
    }
    return true;
}

json decodeRefineSummary(const PicDecodeRefineConfig& config, const std::string& disabledReason = "") {
    json summary = {
        {"enabled", config.enabled},
        {"tokens_per_decode_step", config.tokensPerDecodeStep},
        {"top_m", config.topM},
        {"score_threshold", config.scoreThreshold},
        {"score_margin", config.scoreMargin},
        {"score_decay", config.scoreDecay},
        {"selector", config.selector},
        {"attention_layer_idx", config.attentionLayerIdx},
        {"attention_head_ids", config.attentionHeadIds},
        {"candidate_count", 0},
        {"refined_token_count", 0},
        {"refined_logical_indices", json::array()},
        {"steps", json::array()},
    };
    if (!disabledReason.empty()) {
        summary["disabled_reason"] = disabledReason;
    } else if (!config.enabled) {
        summary["disabled_reason"] = "not_requested";
    }
    return summary;
}

json decodeRefineRuntimeSummary(const PicDecodeRefineConfig& config, MNN::Transformer::Llm* llm,
                                const std::string& disabledReason = "") {
    json summary = decodeRefineSummary(config, disabledReason);
    if (!config.enabled || !disabledReason.empty() || llm == nullptr) {
        return summary;
    }
    const auto steps = llm->picDecodeRepairStepLogicalIndices();
    std::vector<int> refinedLogicalIndices;
    json stepSummaries = json::array();
    refinedLogicalIndices.reserve(steps.size() * static_cast<size_t>(std::max(0, config.tokensPerDecodeStep)));
    for (size_t i = 0; i < steps.size(); ++i) {
        const auto& step = steps[i];
        refinedLogicalIndices.insert(refinedLogicalIndices.end(), step.begin(), step.end());
        stepSummaries.push_back({
            {"step_idx", static_cast<int>(i)},
            {"repair_token_count", step.size()},
            {"repair_logical_indices", step},
        });
    }
    summary["candidate_count"] = llm->picDecodeRepairCandidateCount();
    summary["refined_token_count"] = refinedLogicalIndices.size();
    summary["refined_logical_indices"] = refinedLogicalIndices;
    summary["steps"] = std::move(stepSummaries);
    summary["runtime"] = "mnn_token_id_sparse_decode";
    return summary;
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
           algorithm == "delta-a" || isFusionRAGOnlineAlgorithm(algorithm) ||
           isExternalRequestIndexAlgorithm(algorithm);
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
    res.set_content(body.dump(indent, ' ', false, json::error_handler_t::replace), "application/json");
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
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
    try {
        std::lock_guard<std::mutex> lock(mutex);
        ok = handler(request, response, error);
    } catch (const std::exception& e) {
        ok = false;
        error = e.what();
    } catch (...) {
        ok = false;
        error = "Unhandled exception";
    }
#else
    {
        std::lock_guard<std::mutex> lock(mutex);
        ok = handler(request, response, error);
    }
#endif
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

std::string uniqueBuildSuffix() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return ".building-" + std::to_string(static_cast<long long>(now));
}

bool replaceDirectoryWithBuiltCache(const fs::path& finalRoot, const fs::path& buildRoot, std::string& error) {
    std::error_code ec;
    fs::path backupRoot = finalRoot;
    backupRoot += ".previous-" + std::to_string(
        static_cast<long long>(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::remove_all(backupRoot, ec);
    ec.clear();
    if (fs::exists(finalRoot, ec)) {
        fs::rename(finalRoot, backupRoot, ec);
        if (ec) {
            error = "Failed to move existing cache aside: " + finalRoot.string() + ": " + ec.message();
            return false;
        }
    }
    ec.clear();
    fs::rename(buildRoot, finalRoot, ec);
    if (ec) {
        const auto renameError = ec.message();
        std::error_code restoreEc;
        if (fs::exists(backupRoot, restoreEc)) {
            fs::rename(backupRoot, finalRoot, restoreEc);
        }
        error = "Failed to publish built cache: " + buildRoot.string() + " -> " + finalRoot.string() +
            ": " + renameError;
        return false;
    }
    ec.clear();
    fs::remove_all(backupRoot, ec);
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
    prepared.scoreLayerIdx = std::max(0, jsonInt(request, "pic_recompute_score_layer_idx", 1));
    if (isFusionRAGOnlineAlgorithm(prepared.selectionAlgorithm)) {
        prepared.fusionragCaptureLayerIdx = jsonInt(
            request, "fusionrag_score_layers", jsonInt(request, "fusionrag_score_layer_idx", -1));
        prepared.fusionragQueryTailTokens = std::max(0, jsonInt(request, "fusionrag_query_tail_tokens", 0));
        prepared.fusionragScoreReduce = jsonString(request, "fusionrag_score_reduce", "sum_query_mean_head");
        if (fusionragReduceModeFromString(prepared.fusionragScoreReduce) < 0) {
            error = "Unsupported fusionrag_score_reduce: " + prepared.fusionragScoreReduce;
            return false;
        }
    }
    if (!parsePicDecodeRefineConfig(request, prepared.decodeRefine, error)) {
        return false;
    }
    prepared.hasExplicitRecomputeIndices = request.contains("pic_recompute_logical_indices") ||
                                           request.contains("pic_recompute_pic_local_indices");
    prepared.explicitLogicalIndices = jsonIntVector(request.value("pic_recompute_logical_indices", json::array()));
    prepared.explicitPicLocalIndices = jsonIntVector(request.value("pic_recompute_pic_local_indices", json::array()));
    if (prepared.selectionAlgorithm == "cacheclip" && !prepared.hasExplicitRecomputeIndices) {
        error = prepared.selectionAlgorithm +
                " requires pic_recompute_pic_local_indices or pic_recompute_logical_indices from an external selector";
        return false;
    }
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

json precisionRecoverySummary(const PicExecutionPlan& plan, double ratio, int requestedScoreLayerIdx,
                              const json* decodeRefine = nullptr) {
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
        {"decode_refine", decodeRefine != nullptr ? *decodeRefine : decodeRefineSummary(plan.decodeRefine)},
    };
}

json performanceSummary(const MNN::Transformer::LlmContext* context, int completionTokens,
                        int activeTokensPerDecodeStep, int64_t requestWallUs) {
    int64_t prefillUs = 0;
    int64_t decodeUs = 0;
    int64_t sampleUs = 0;
    int64_t ttfaUs = 0;
    if (context != nullptr) {
        prefillUs = context->prefill_us;
        decodeUs = context->decode_us;
        sampleUs = context->sample_us;
        ttfaUs = context->ttfa_us;
    }
    // ArGeneration reuses the prefill logits for the first generated token and
    // records decode_us for the following next-logits forwards.
    const int measuredDecodeTokens = decodeUs > 0 ? std::max(1, completionTokens - 1) : 0;
    const double prefillS = static_cast<double>(prefillUs) / 1000000.0;
    const double decodeS = static_cast<double>(decodeUs) / 1000000.0;
    const double sampleS = static_cast<double>(sampleUs) / 1000000.0;
    const double wallS = static_cast<double>(requestWallUs) / 1000000.0;
    const double decodeTpotMs = measuredDecodeTokens > 0
        ? static_cast<double>(decodeUs) / 1000.0 / static_cast<double>(measuredDecodeTokens)
        : 0.0;
    const double decodeTps = decodeS > 0.0
        ? static_cast<double>(measuredDecodeTokens) / decodeS
        : 0.0;
    const double sampleTpotMs = completionTokens > 0
        ? static_cast<double>(sampleUs) / 1000.0 / static_cast<double>(completionTokens)
        : 0.0;
    const double wallMinusDecodeS = std::max(0.0, wallS - decodeS);
    const double wallMinusDecodeSampleS = std::max(0.0, wallS - decodeS - sampleS);
    return {
        {"prefill_us", prefillUs},
        {"prefill_latency_s", prefillS},
        {"decode_us", decodeUs},
        {"decode_latency_s", decodeS},
        {"decode_measured_tokens", measuredDecodeTokens},
        {"decode_tpot_ms", decodeTpotMs},
        {"decode_tps", decodeTps},
        {"completion_tokens", completionTokens},
        {"active_tokens_per_decode_step", activeTokensPerDecodeStep},
        {"sample_us", sampleUs},
        {"sample_latency_s", sampleS},
        {"sample_tpot_ms", sampleTpotMs},
        {"ttfa_us", ttfaUs},
        {"request_wall_us", requestWallUs},
        {"request_wall_s", wallS},
        {"wall_minus_decode_s", wallMinusDecodeS},
        {"wall_minus_decode_sample_s", wallMinusDecodeSampleS},
    };
}

} // namespace

PicServer::PicServer(PicServerConfig config) : mConfig(std::move(config)) {
}

bool PicServer::load() {
    return loadLlmInstance(nullptr);
}

bool PicServer::loadLlmInstance(std::string* error) {
    std::unique_ptr<MNN::Transformer::Llm> llm(MNN::Transformer::Llm::createLLM(mConfig.configPath));
    if (!llm) {
        std::string message = "Failed to create LLM from " + mConfig.configPath;
        if (error != nullptr) {
            *error = message;
        } else {
            std::cerr << message << "\n";
        }
        return false;
    }
    std::error_code ec;
    mConfig.kvCacheDir = absoluteString(mConfig.kvCacheDir);
    fs::create_directories(mConfig.kvCacheDir, ec);
    const std::string runtimeCacheDir = absoluteString(
        mConfig.runtimeCacheDir.empty()
            ? (fs::path(mConfig.kvCacheDir) / "runtime_cache").string()
            : mConfig.runtimeCacheDir);
    mConfig.runtimeCacheDir = runtimeCacheDir;
    fs::create_directories(runtimeCacheDir, ec);
    json runtimeConfig = {
        {"tmp_path", runtimeCacheDir},
        {"prefix_cache_path", mConfig.kvCacheDir},
    };
    const int pagedKvLimit = envInt("MNN_PIC_SERVER_PAGED_KV_MAX_TOKENS", 0);
    if (pagedKvLimit > 0) {
        runtimeConfig["paged_kv_max_tokens"] = pagedKvLimit;
        runtimeConfig["max_all_tokens"] = pagedKvLimit;
        std::cout << "PIC server test override: paged_kv_max_tokens=" << pagedKvLimit << "\n";
    }
    if (picGraphProfileEnabled()) {
        runtimeConfig["enable_debug"] = true;
        std::cout << "PIC server graph profile enabled: MNN_PIC_GRAPH_PROFILE=1\n";
    }
    std::string config = runtimeConfig.dump();
    llm->set_config(config);
    if (picGraphProfileEnabled()) {
        llm->setDebugCallback(
            [](const std::vector<MNN::Tensor*>& inputs, const MNN::OperatorInfo* info) {
                return PicGraphProfiler::get().before(inputs, info);
            },
            [](const std::vector<MNN::Tensor*>& outputs, const MNN::OperatorInfo* info) {
                return PicGraphProfiler::get().after(outputs, info);
            });
    }
    if (!llm->load()) {
        std::string message = "Failed to load LLM from " + mConfig.configPath;
        if (error != nullptr) {
            *error = message;
        } else {
            std::cerr << message << "\n";
        }
        return false;
    }
    mLlm = std::move(llm);
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

void PicServer::handleReset(const httplib::Request& req, httplib::Response& res) {
    allowCors(res);
    bool releaseModuleClones = false;
    bool reloadModel = false;
    if (!req.body.empty()) {
        json request;
        if (!parseJsonBody(req, res, request)) {
            return;
        }
        if (!request.is_null() && !request.is_object()) {
            writeJsonError(res, 400, "Reset request body must be a JSON object");
            return;
        }
        if (request.is_object()) {
            releaseModuleClones = request.value("release_module_clones", false) ||
                                  request.value("release_runtime_buffers", false);
            reloadModel = request.value("reload_model", false);
        }
    }
    std::lock_guard<std::mutex> lock(mMutex);
    size_t releasedModuleClones = 0;
    bool reloaded = false;
    json cleared = json::array({"prefix_cache_mode", "paged_pic_request", "context_history"});
    if (mLlm) {
        mLlm->clearPrefixCacheFile();
        mLlm->finishExternalPagedKVRequest();
        mLlm->reset();
        if (releaseModuleClones) {
            releasedModuleClones = mLlm->releaseForwardModuleClones();
            cleared.push_back("module_clones");
            cleared.push_back("runtime_buffer_free_lists");
        }
    }
    if (reloadModel) {
        std::string error;
        mLlm.reset();
        if (!loadLlmInstance(&error)) {
            writeJsonError(res, 500, error);
            return;
        }
        reloaded = true;
        cleared.push_back("llm_model_runtime");
    }
    writeJson(res, json({
        {"status", "ok"},
        {"scope", "llm_request_state"},
        {"cleared", cleared},
        {"released_module_clones", releasedModuleClones},
        {"reloaded_model", reloaded},
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
    const auto decodeModel = jsonString(cfg, "llm_decode_model", "");
    const auto decodeWeight = jsonString(cfg, "llm_decode_weight", "");
    return {
        {"name", "mnn_pic_server"},
        {"backend", runtimeBackend()},
        {"attention_mode", jsonBool(cfg, "paged_attention", false) ? "paged" : "standard"},
        {"paged_attention", jsonBool(cfg, "paged_attention", false)},
        {"pic_dualgraph_decode", !decodeModel.empty()},
        {"llm_decode_model", decodeModel},
        {"llm_decode_weight", decodeWeight},
        {"llm_decode_shared_weight", jsonBool(cfg, "llm_decode_shared_weight", false)},
        {"request_scoped_kv", true},
        {"kv_cache_dir", absoluteString(mConfig.kvCacheDir)},
        {"runtime_cache_dir", absoluteString(mConfig.runtimeCacheDir)},
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
    const std::string buildName = cacheName + uniqueBuildSuffix();
    fs::path buildRoot = cacheRoot.parent_path() / buildName;
    fs::path buildLayersDir = buildRoot / "layers";
    fs::path buildMetaPath = buildRoot / "meta.json";
    fs::path buildTokensPath = buildRoot / "tokens.json";
    std::string buildFileStem = pathJoinForMNN(
        pathJoinForMNN(pathJoinForMNN(pathJoinForMNN("objects", backend), buildName), "layers"),
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
    fs::remove_all(buildRoot, ec);
    ec.clear();
    fs::create_directories(buildLayersDir, ec);
    if (ec) {
        error = "Failed to create cache directory: " + buildLayersDir.string();
        return false;
    }

    mLlm->reset();
    if (!mLlm->beginTextCacheExport(buildFileStem)) {
        error = "Failed to start PIC text cache export";
        fs::remove_all(buildRoot, ec);
        return false;
    }
    std::ostringstream sink;
    mLlm->response(tokenIds, &sink, "", 0);
    auto contextAfterPrefill = mLlm->getContext();
    const bool prefillFailed =
        contextAfterPrefill == nullptr ||
        contextAfterPrefill->status == MNN::Transformer::LlmStatus::INTERNAL_ERROR ||
        contextAfterPrefill->status == MNN::Transformer::LlmStatus::TIMEOUT ||
        contextAfterPrefill->status == MNN::Transformer::LlmStatus::USER_CANCEL;
    if (prefillFailed) {
        error = "Failed to build persistent text cache" + llmContextSuffix(mLlm.get());
        mLlm->clearTextCacheExport();
        mLlm->finishExternalPagedKVRequest();
        mLlm->reset();
        fs::remove_all(buildRoot, ec);
        return false;
    }
    mLlm->clearTextCacheExport();
    mLlm->finishExternalPagedKVRequest();
    mLlm->reset();

    auto cfg = modelConfig();
    int layerCount = jsonInt(cfg, "layer_nums", 0);
    int detectedLayers = countLayerFiles(buildLayersDir, cacheName);
    if (layerCount <= 0) {
        layerCount = detectedLayers;
    }
    if (layerCount <= 0 || detectedLayers <= 0) {
        error = "Prefill finished but no layer KV files were exported";
        fs::remove_all(buildRoot, ec);
        return false;
    }
    if (detectedLayers < layerCount) {
        error = "Expected " + std::to_string(layerCount) + " layer KV files, found " +
                std::to_string(detectedLayers);
        fs::remove_all(buildRoot, ec);
        return false;
    }

    KvShape kvShape = readKvShape(buildLayersDir, cacheName);
    json kvLayout = makeKvLayout(cfg, static_cast<int>(tokenIds.size()), kvShape);
    int layoutBatch = kvLayout.value("batch", 1);
    int layoutKvHeads = kvLayout.value("kv_heads", 0);
    int layoutHeadDim = kvLayout.value("head_dim", 0);
    json kvFiles = json::array();
    for (int layer = 0; layer < layerCount; ++layer) {
        fs::path keyPath = layersDir / (cacheName + "_" + std::to_string(layer) + ".k");
        fs::path valuePath = layersDir / (cacheName + "_" + std::to_string(layer) + ".v");
        fs::path buildKeyPath = buildLayersDir / (cacheName + "_" + std::to_string(layer) + ".k");
        fs::path buildValuePath = buildLayersDir / (cacheName + "_" + std::to_string(layer) + ".v");
        kvFiles.push_back({
            {"layer_index", layer},
            {"key_path", absoluteString(keyPath)},
            {"value_path", absoluteString(valuePath)},
            {"key_bytes", fileSizeOrZero(buildKeyPath)},
            {"value_bytes", fileSizeOrZero(buildValuePath)},
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
    if (!writeJsonFile(buildTokensPath, tokensJson)) {
        error = "Failed to write tokens metadata";
        fs::remove_all(buildRoot, ec);
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
    if (!writeJsonFile(buildMetaPath, response)) {
        error = "Failed to write cache metadata";
        fs::remove_all(buildRoot, ec);
        return false;
    }
    if (!replaceDirectoryWithBuiltCache(cacheRoot, buildRoot, error)) {
        fs::remove_all(buildRoot, ec);
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
        {"decode_refine", decodeRefineSummary(pic.decodeRefine,
                                              pic.decodeRefine.enabled
                                                  ? "not_prepared_until_chat_decode"
                                                  : "")},
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
    const int64_t requestStartUs = monotonicUs();
    PicGraphProfileRequestScope graphProfileScope(picGraphProfileLabel(request));
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
    int activeTokensPerDecodeStep = 1;
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
                error = "Failed to prefill no-PIC chat request" + llmContextSuffix(mLlm.get());
                return false;
            }
        }
        if (maxTokens != 0) {
            if (!mLlm->preparePagedDecode(maxTokens)) {
                mLlm->finishExternalPagedKVRequest();
                error = "Failed to prepare PagedAttention decode state before no-PIC decode" +
                        llmContextSuffix(mLlm.get());
                return false;
            }
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
            PicGraphProfileRequestScope decodeGraphProfileScope(picGraphProfileLabel(request), "decode");
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
        int64_t stageUs = monotonicUs();
        picRequestProfileBegin("prepare_pic_cache");
        if (!preparePicCacheFromRequest(mConfig.kvCacheDir, runtimeBackend(), request["pic_cache"], pic, error)) {
            return false;
        }
        {
            std::ostringstream os;
            os << "selection_algorithm=" << pic.selectionAlgorithm
               << " score_layer=" << pic.scoreLayerIdx
               << " pic_tokens=" << pic.tokenIds.size()
               << " segments=" << pic.segments.size();
            picRequestProfileLog("prepare_pic_cache", monotonicUs() - stageUs, os.str());
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
        stageUs = monotonicUs();
        picRequestProfileBegin("apply_explicit_pic_prompt_mapping");
        if (!applyExplicitPicPromptMapping(pic, preludeTokenIds, suffixTokenIds,
                                           fullPromptTokenIds, explicitTokenMapping, error)) {
            return false;
        }
        {
            std::ostringstream os;
            os << "prelude_tokens=" << preludeTokenIds.size()
               << " pic_tokens=" << pic.tokenIds.size()
               << " suffix_tokens=" << suffixTokenIds.size()
               << " full_prompt_tokens=" << fullPromptTokenIds.size();
            picRequestProfileLog("apply_explicit_pic_prompt_mapping", monotonicUs() - stageUs, os.str());
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
        std::vector<int> nativeSelectedLocalIndices;
        bool hasNativeSelectedLocalIndices = false;
        {
            MnnLlmPerfettoSlice prefillSlice("prefill", traceInfo);
            PicPrefillResult prefillResult;
            if (!runPicPrefill(
                    mLlm.get(), sink, runtimeBackend(), pic, fullPromptTokenIds,
                    static_cast<int>(preludeTokenIds.size()), layerCount,
                    jsonBool(cfg, "pic_recompute_budget", false), prefillResult, error)) {
                return false;
            }
            plan = std::move(prefillResult.plan);
            nativeSelectedLocalIndices = std::move(prefillResult.nativeSelectedLocalIndices);
            hasNativeSelectedLocalIndices = prefillResult.hasNativeSelectedLocalIndices;
            traceInfo.executionMode = plan.executionMode;
            traceInfo.recomputeBudgetTokens = plan.recomputeTokenCount;
        }
        if (pic.decodeRefine.enabled && maxTokens != 0) {
            std::vector<int> seedSelectedPicLocalIndices;
            seedSelectedPicLocalIndices.reserve(plan.recomputeLogicalIndices.size());
            for (int logical : plan.recomputeLogicalIndices) {
                const int local = logical - static_cast<int>(preludeTokenIds.size());
                if (local >= 0 && local < static_cast<int>(pic.tokenIds.size())) {
                    seedSelectedPicLocalIndices.emplace_back(local);
                }
            }
            std::vector<int> rankedPicLocalIndices = nativeSelectedLocalIndices;
            if (rankedPicLocalIndices.empty()) {
                rankedPicLocalIndices = seedSelectedPicLocalIndices;
            }
            if (!mLlm->preparePicDecodeRepair(
                    static_cast<int>(preludeTokenIds.size()), pic.tokenIds, rankedPicLocalIndices,
                    seedSelectedPicLocalIndices, pic.decodeRefine.tokensPerDecodeStep,
                    pic.decodeRefine.selector, pic.decodeRefine.attentionLayerIdx,
                    pic.decodeRefine.attentionHeadIds, pic.decodeRefine.topM)) {
                error = "Failed to prepare PIC decode_refine runtime state";
                return false;
            }
            activeTokensPerDecodeStep = pic.decodeRefine.tokensPerDecodeStep + 1;
            plan.metadata["decode_refine_runtime"] = "mnn_token_id_sparse_decode";
            plan.metadata["decode_refine_ranking_source"] =
                hasNativeSelectedLocalIndices ? "native_selected_indices_then_pic_order"
                                              : "seed_selected_indices_then_pic_order";
            plan.metadata["decode_refine_selection_source"] =
                pic.decodeRefine.selector == "lagged_attention_hkvd"
                    ? "lagged_attention_compact_rank_filtered_hkvd_then_top_hkvd_fill"
                    : "top_hkvd";
            plan.metadata["decode_refine_attention_layer_idx"] = pic.decodeRefine.attentionLayerIdx;
            plan.metadata["decode_refine_attention_head_ids"] = pic.decodeRefine.attentionHeadIds;
            plan.metadata["decode_refine_attention_top_m"] = pic.decodeRefine.topM;
        }
        if (maxTokens != 0) {
            if (!mLlm->preparePagedDecode(maxTokens)) {
                mLlm->finishExternalPagedKVRequest();
                error = "Failed to prepare PagedAttention decode state before PIC decode" +
                        llmContextSuffix(mLlm.get());
                return false;
            }
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
            PicGraphProfileRequestScope decodeGraphProfileScope(picGraphProfileLabel(request), "decode");
            outputTokens = mLlm->decode(maxTokens);
        } else {
            auto context = mLlm->getContext();
            if (context != nullptr) {
                outputTokens = context->output_tokens;
            }
        }
        json decodeRefineInfo = decodeRefineSummary(pic.decodeRefine);
        if (pic.decodeRefine.enabled) {
            const std::string disabledReason = maxTokens == 0 ? "max_tokens_zero_prefill_only" : "";
            decodeRefineInfo = decodeRefineRuntimeSummary(pic.decodeRefine, mLlm.get(), disabledReason);
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
            {"decode_refine", decodeRefineInfo},
            {"precision_recovery", precisionRecoverySummary(plan, pic.recomputeRatio, pic.scoreLayerIdx,
                                                            &decodeRefineInfo)},
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
        {"performance", performanceSummary(context, completionTokens, activeTokensPerDecodeStep,
                                           monotonicUs() - requestStartUs)},
    };
    if (!picInfo.is_null()) {
        response["pic_cache"] = picInfo;
    }
    return true;
}

} // namespace pic
