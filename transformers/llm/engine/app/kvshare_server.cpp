#include "kvshare_server.hpp"

#include <MNN/AutoTime.hpp>
#include "core/PrefixCachePath.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

using MNN::Transformer::ChatMessages;
using MNN::Transformer::Llm;
using MNN::Transformer::LlmStatus;

namespace kvshare {

namespace {

std::string currentTimeString() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    return std::to_string(seconds);
}

void allowCors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
}

std::string absolutePath(const std::filesystem::path& path) {
    std::error_code ec;
    auto abs = std::filesystem::absolute(path, ec);
    return ec ? path.string() : abs.string();
}

std::string sanitizeCachePart(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    for (unsigned char ch : value) {
        if (std::isalnum(ch) || ch == '_' || ch == '-' || ch == '.') {
            result.push_back(static_cast<char>(ch));
        } else {
            result.push_back('_');
        }
    }
    return result.empty() ? "unnamed" : result;
}

std::string documentCacheName(const std::string& id) {
    return "doc_" + sanitizeCachePart(id);
}

std::string compositeCacheName(const json& segments) {
    std::string name = "segments";
    if (segments.is_array()) {
        for (const auto& segment : segments) {
            name += "_";
            name += sanitizeCachePart(segment.value("id", segment.value("cache_name", "unknown")));
        }
    }
    return name;
}

constexpr const char* kDocumentTokenizerPrefixPolicy = "strip-empty-encode-prefix-v1";
constexpr const char* kDefaultKVPrefixPlaceholder = "{{kv_prefix}}";

int stripTokenizerPrefixTokens(std::vector<int>& tokens, const std::vector<int>& prefixTokens) {
    if (tokens.empty() || prefixTokens.empty() || tokens.size() <= prefixTokens.size()) {
        return 0;
    }
    if (!std::equal(prefixTokens.begin(), prefixTokens.end(), tokens.begin())) {
        return 0;
    }
    tokens.erase(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(prefixTokens.size()));
    return static_cast<int>(prefixTokens.size());
}

ChatMessages collectSystemMessages(const ChatMessages& messages) {
    ChatMessages result;
    for (const auto& message : messages) {
        if (message.first == "system") {
            result.emplace_back(message);
        }
    }
    return result;
}

ChatMessages collectNonSystemMessages(const ChatMessages& messages) {
    ChatMessages result;
    for (const auto& message : messages) {
        if (message.first != "system") {
            result.emplace_back(message);
        }
    }
    return result;
}

bool splitRenderedPromptAtPlaceholder(const std::string& renderedPrompt,
                                      const std::string& placeholder,
                                      std::string& prelude,
                                      std::string& suffix,
                                      std::string& error) {
    if (placeholder.empty()) {
        error = "kv_prefix.placeholder must not be empty";
        return false;
    }
    auto pos = renderedPrompt.find(placeholder);
    if (pos == std::string::npos) {
        return false;
    }
    if (renderedPrompt.find(placeholder, pos + placeholder.size()) != std::string::npos) {
        error = "kv_prefix placeholder appears more than once in the rendered chat prompt";
        return false;
    }
    prelude = renderedPrompt.substr(0, pos);
    suffix = renderedPrompt.substr(pos + placeholder.size());
    return true;
}

std::filesystem::path cacheObjectDir(const std::string& cacheDir, const std::string& backend,
                                     const std::string& cacheName) {
    return std::filesystem::path(MNN::prefixCacheObjectDir(cacheDir, backend, cacheName));
}

std::filesystem::path metaPathForCache(const std::string& cacheDir, const std::string& backend,
                                       const std::string& cacheName) {
    return std::filesystem::path(MNN::prefixCacheMetaPath(cacheDir, backend, cacheName));
}

std::filesystem::path tokensPathForCache(const std::string& cacheDir, const std::string& backend,
                                         const std::string& cacheName) {
    return std::filesystem::path(MNN::prefixCacheTokensPath(cacheDir, backend, cacheName));
}

std::filesystem::path cacheBasePath(const std::string& cacheDir, const std::string& backend,
                                    const std::string& cacheName, int layer) {
    return std::filesystem::path(MNN::prefixCacheLayerBase(cacheDir, backend, cacheName, layer));
}

bool writeTextFile(const std::filesystem::path& path, const std::string& content, std::string& error) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        error = "Failed to create directory: " + path.parent_path().string();
        return false;
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        error = "Failed to open file for write: " + path.string();
        return false;
    }
    out << content;
    return static_cast<bool>(out);
}

bool readTextFile(const std::filesystem::path& path, std::string& content, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "Failed to open file for read: " + path.string();
        return false;
    }
    std::ostringstream os;
    os << in.rdbuf();
    content = os.str();
    return true;
}

int layerCountFromConfig(Llm* llm) {
    int layers = 32;
    if (llm == nullptr) {
        return layers;
    }
    auto config = json::parse(llm->dump_config(), nullptr, false);
    if (config.is_object() && config.contains("layer_nums") && config["layer_nums"].is_number_integer()) {
        layers = config["layer_nums"].get<int>();
    }
    return layers;
}

double numberFromJson(const json& object, const std::string& key, double defaultValue) {
    if (object.is_object() && object.contains(key) && object[key].is_number()) {
        return object[key].get<double>();
    }
    return defaultValue;
}

int intFromJson(const json& object, const std::string& key, int defaultValue) {
    if (object.is_object() && object.contains(key) && object[key].is_number_integer()) {
        return object[key].get<int>();
    }
    return defaultValue;
}

bool isValidCudaPrefixPrefetchThreads(int threads) {
    return threads == 4 || threads == 8 || threads == 16;
}

bool isValidOpenCLPrefixPrefetchThreads(int threads) {
    return threads == 1;
}

bool isValidPrefixHostCacheThreads(int threads) {
    return threads == 4 || threads == 8 || threads == 16;
}

json ropeLayoutFromConfig(Llm* llm, const std::string& backend, std::string& error) {
    json config = json::object();
    if (llm != nullptr) {
        config = json::parse(llm->dump_config(), nullptr, false);
    }
    if (!config.is_object()) {
        config = json::object();
    }

    auto modelType = config.value("model_type", std::string(""));
    bool isMrope = config.value("is_mrope", false);
    if (isMrope || modelType == "chatglm" || modelType == "chatglm2" ||
        modelType == "ernie4_5" || modelType == "glm_ocr") {
        error = "direct_segments canonical RoPE v1 only supports half-split text RoPE; unsupported model_type=" + modelType;
        return json::object();
    }

    int headDim = intFromJson(config, "head_dim", 0);
    int kvHeads = intFromJson(config, "kv_heads", 0);
    if (headDim <= 0 && config.contains("key_value_shape") && config["key_value_shape"].is_array() &&
        !config["key_value_shape"].empty()) {
        const auto& shape = config["key_value_shape"];
        if (shape.back().is_number_integer()) {
            headDim = shape.back().get<int>();
        }
        if (shape.size() >= 2 && shape[shape.size() - 2].is_number_integer()) {
            kvHeads = shape[shape.size() - 2].get<int>();
        }
    }
    if (headDim <= 0) {
        double attnScale = numberFromJson(config, "attn_scale", 0.0);
        if (attnScale > 0.0) {
            double inferred = 1.0 / (attnScale * attnScale);
            int rounded = static_cast<int>(std::lround(inferred));
            if (rounded > 0 && std::abs(inferred - static_cast<double>(rounded)) < 1e-3) {
                headDim = rounded;
            }
        }
    }
    int ropeDim = intFromJson(config, "rotary_dim", headDim);
    if (ropeDim <= 0) {
        ropeDim = headDim;
    }

    double ropeTheta = numberFromJson(config, "rope_theta", 10000.0);
    double ropeRatio = numberFromJson(config, "rope_ratio", 0.0);
    if (ropeRatio > 0.0) {
        ropeTheta *= ropeRatio;
    }
    if (config.contains("rope_parameters") && config["rope_parameters"].is_object()) {
        const auto& ropeParameters = config["rope_parameters"];
        bool perLayerType = false;
        for (auto it = ropeParameters.begin(); it != ropeParameters.end(); ++it) {
            if (it.value().is_object()) {
                perLayerType = true;
                break;
            }
        }
        if (!perLayerType) {
            ropeTheta = numberFromJson(ropeParameters, "rope_theta", ropeTheta);
            double partialRotaryFactor = numberFromJson(ropeParameters, "partial_rotary_factor", 1.0);
            if (partialRotaryFactor > 0.0 && ropeDim > 0) {
                ropeDim = static_cast<int>(ropeDim * partialRotaryFactor);
            }
        }
    }

    std::string format = "mnn-cpu-flash-prefix-kv-v1";
    std::string layout = "cpu-flash-packed-canonical-no-rope-v1";
    int pageSize = 0;
    if (backend == "cuda") {
        format = "mnn-cuda-paged-prefix-kv-v1";
        layout = "cuda-paged-canonical-no-rope-v1";
        pageSize = 64;
    } else if (backend == "opencl") {
        format = "mnn-opencl-prefix-kv-v1";
        layout = "opencl-buffer-canonical-no-rope-v1";
    }

    json result = {
        {"format", format},
        {"format_version", "v1"},
        {"backend", backend},
        {"layout", layout},
        {"key_rope_state", "canonical_no_rope"},
        {"rope_state", "canonical_no_rope"},
        {"rope_pairing", "half"},
        {"head_dim", headDim},
        {"rope_dim", ropeDim},
        {"rope_theta", ropeTheta},
        {"source_position_base", 0},
        {"dtype", "runtime"},
        {"valid_token_count_field", "token_count"},
        {"note", "Key is saved without position RoPE; direct_segments applies RoPE again with concatenated global positions."}
    };
    if (pageSize > 0) {
        result["page_size"] = pageSize;
    }
    if (kvHeads > 0) {
        result["kv_heads"] = kvHeads;
    }
    return result;
}

bool kvLayoutMatchesBackend(const json& kvLayout, const json& segmentMeta, const std::string& backend) {
    if (!kvLayout.is_object()) {
        return false;
    }
    auto layoutBackend = kvLayout.value("backend", segmentMeta.value("backend", std::string("")));
    auto format = kvLayout.value("format", std::string(""));
    auto layout = kvLayout.value("layout", std::string(""));
    if (backend == "cpu") {
        return (layoutBackend.empty() || layoutBackend == "cpu") &&
               format == "mnn-cpu-flash-prefix-kv-v1";
    }
    if (backend == "cuda") {
        return layoutBackend == "cuda" &&
               format == "mnn-cuda-paged-prefix-kv-v1" &&
               layout == "cuda-paged-canonical-no-rope-v1" &&
               kvLayout.value("page_size", 0) == 64;
    }
    if (backend == "opencl") {
        return layoutBackend == "opencl" &&
               format == "mnn-opencl-prefix-kv-v1" &&
               layout == "opencl-buffer-canonical-no-rope-v1";
    }
    return false;
}

json kvFilesForCache(const std::string& cacheDir, const std::string& backend,
                     const std::string& cacheName, int layers) {
    json files = json::array();
    for (int i = 0; i < layers; ++i) {
        auto base = cacheBasePath(cacheDir, backend, cacheName, i);
        files.push_back({
            {"layer", i},
            {"key_path", absolutePath(base.string() + ".k")},
            {"value_path", absolutePath(base.string() + ".v")}
        });
    }
    return files;
}

bool prefixCacheReady(const std::string& cacheDir, const std::string& backend,
                      const std::string& cacheName, int layers) {
    for (int i = 0; i < layers; ++i) {
        auto base = cacheBasePath(cacheDir, backend, cacheName, i);
        if (!std::filesystem::exists(base.string() + ".k") ||
            !std::filesystem::exists(base.string() + ".v")) {
            return false;
        }
    }
    return true;
}

void removePrefixCacheFiles(const std::string& cacheDir, const std::string& backend,
                            const std::string& cacheName, int layers) {
    std::error_code ec;
    for (int i = 0; i < layers; ++i) {
        auto base = cacheBasePath(cacheDir, backend, cacheName, i);
        std::filesystem::remove(base.string() + ".k", ec);
        std::filesystem::remove(base.string() + ".v", ec);
    }
    std::filesystem::remove(metaPathForCache(cacheDir, backend, cacheName), ec);
    std::filesystem::remove(tokensPathForCache(cacheDir, backend, cacheName), ec);
    std::filesystem::remove_all(cacheObjectDir(cacheDir, backend, cacheName), ec);
}

bool loadJsonFile(const std::filesystem::path& path, json& result, std::string& error) {
    std::string content;
    if (!readTextFile(path, content, error)) {
        return false;
    }
    result = json::parse(content, nullptr, false);
    if (result.is_discarded()) {
        error = "Invalid JSON file: " + path.string();
        return false;
    }
    return true;
}

bool writeJsonFile(const std::filesystem::path& path, const json& value, std::string& error) {
    return writeTextFile(path, value.dump(2), error);
}

std::vector<int> tokensFromMeta(const json& meta, const std::string& cacheDir, std::string& error) {
    if (meta.contains("token_ids") && meta["token_ids"].is_array()) {
        return meta["token_ids"].get<std::vector<int>>();
    }
    if (!meta.contains("token_ids_path") || !meta["token_ids_path"].is_string()) {
        error = "Cache meta has no token_ids or token_ids_path.";
        return {};
    }
    json tokensJson;
    auto tokensPath = std::filesystem::path(meta["token_ids_path"].get<std::string>());
    if (!tokensPath.is_absolute()) {
        tokensPath = std::filesystem::path(cacheDir) / tokensPath;
    }
    if (!loadJsonFile(tokensPath, tokensJson, error)) {
        return {};
    }
    if (!tokensJson.contains("token_ids") || !tokensJson["token_ids"].is_array()) {
        error = "Token file missing token_ids: " + tokensPath.string();
        return {};
    }
    return tokensJson["token_ids"].get<std::vector<int>>();
}

bool isAllowedAttentionMode(int mode) {
    switch (mode) {
        case 8:
        case 9:
        case 10:
        case 13:
        case 14:
            return true;
        default:
            return false;
    }
}

std::string trimLeadingWhitespace(const std::string& str) {
    auto it = std::find_if(str.begin(), str.end(), [](unsigned char ch) { return !std::isspace(ch); });
    return std::string(it, str.end());
}

std::string getR1AssistantString(std::string assistantContent) {
    std::size_t pos = assistantContent.find("</think>");
    if (pos != std::string::npos) {
        assistantContent.erase(0, pos + std::string("</think>").length());
    }
    return trimLeadingWhitespace(assistantContent) + "<|end_of_sentence|>";
}

std::string getR1UserString(const std::string& userContent) {
    return "<|User|>" + userContent + "<|Assistant|>";
}

void tuningPrepare(Llm* llm) {
    MNN_PRINT("Prepare for tuning opt Begin\n");
    llm->tuning(MNN::Transformer::OP_ENCODER_NUMBER, {1, 5, 10, 20, 30, 50, 100});
    MNN_PRINT("Prepare for tuning opt End\n");
}

} // namespace

void Utf8StreamProcessor::processStream(const char* str, size_t len) {
    utf8Buffer_.append(str, len);

    size_t i = 0;
    std::string completeChars;
    while (i < utf8Buffer_.size()) {
        int length = utf8CharLength(static_cast<unsigned char>(utf8Buffer_[i]));
        if (length == 0 || i + static_cast<size_t>(length) > utf8Buffer_.size()) {
            break;
        }
        completeChars.append(utf8Buffer_, i, static_cast<size_t>(length));
        i += static_cast<size_t>(length);
    }
    utf8Buffer_ = utf8Buffer_.substr(i);
    if (!completeChars.empty()) {
        callback_(completeChars);
    }
}

int Utf8StreamProcessor::utf8CharLength(unsigned char byte) {
    if ((byte & 0x80) == 0) return 1;
    if ((byte & 0xE0) == 0xC0) return 2;
    if ((byte & 0xF0) == 0xE0) return 3;
    if ((byte & 0xF8) == 0xF0) return 4;
    return 0;
}

KvShareServer::KvShareServer(Options options) : options_(std::move(options)) {
}

bool KvShareServer::init() {
    if (!isAllowedAttentionMode(options_.attentionMode)) {
        std::cerr << "Unsupported attention_mode: " << options_.attentionMode << std::endl;
        return false;
    }
    if (options_.backend != "cpu" && options_.backend != "cuda" && options_.backend != "opencl") {
        std::cerr << "Unsupported backend: " << options_.backend << std::endl;
        return false;
    }
    if (options_.cudaPrefixPrefetchThreads != 0 &&
        !isValidCudaPrefixPrefetchThreads(options_.cudaPrefixPrefetchThreads)) {
        std::cerr << "Unsupported cuda prefix prefetch thread count: "
                  << options_.cudaPrefixPrefetchThreads << " (expected 4, 8, or 16)" << std::endl;
        return false;
    }
    if (options_.openclPrefixPrefetchThreads != 0 &&
        !isValidOpenCLPrefixPrefetchThreads(options_.openclPrefixPrefetchThreads)) {
        std::cerr << "Unsupported opencl prefix prefetch thread count: "
                  << options_.openclPrefixPrefetchThreads << " (expected 1)" << std::endl;
        return false;
    }
    if (options_.prefixHostCacheThreads != 0 &&
        !isValidPrefixHostCacheThreads(options_.prefixHostCacheThreads)) {
        std::cerr << "Unsupported prefix host cache thread count: "
                  << options_.prefixHostCacheThreads << " (expected 4, 8, or 16)" << std::endl;
        return false;
    }
    if (options_.configPath.empty()) {
        std::cerr << "Missing config path" << std::endl;
        return false;
    }
    if (options_.cudaPrefixPrefetchThreads != 0) {
        setenv("MNN_CUDA_PREFIX_PREFETCH_THREADS",
               std::to_string(options_.cudaPrefixPrefetchThreads).c_str(), 1);
    }
    if (options_.openclPrefixPrefetchThreads != 0) {
        setenv("MNN_OPENCL_PREFIX_PREFETCH_THREADS",
               std::to_string(options_.openclPrefixPrefetchThreads).c_str(), 1);
    }
    if (options_.prefixHostCacheThreads != 0) {
        setenv("MNN_PREFIX_HOST_CACHE_THREADS",
               std::to_string(options_.prefixHostCacheThreads).c_str(), 1);
    }
    llm_.reset(Llm::createLLM(options_.configPath));
    if (!llm_) {
        std::cerr << "Failed to create LLM for config: " << options_.configPath << std::endl;
        return false;
    }

    auto prefixCachePath = std::filesystem::path(options_.prefixCacheDir);
    auto cacheRoot = prefixCachePath.has_parent_path() ? prefixCachePath.parent_path() : std::filesystem::path(".cache");
    auto tmpPath = cacheRoot / "kvshare_tmp";
    std::error_code ec;
    std::filesystem::create_directories(prefixCachePath, ec);
    if (ec) {
        std::cerr << "Failed to create prefix cache directory: " << ec.message() << std::endl;
        return false;
    }
    std::filesystem::create_directories(tmpPath, ec);
    if (ec) {
        std::cerr << "Failed to create tmp cache directory: " << ec.message() << std::endl;
        return false;
    }

    bool setSuccess = true;
    json runtimeConfig = {
        {"backend_type", options_.backend},
        {"tmp_path", tmpPath.string()},
        {"kvcache_mmap", true},
        {"prefix_cache_path", options_.prefixCacheDir},
        {"reuse_kv", options_.reuseKv}
    };
    setSuccess &= llm_->set_config(runtimeConfig.dump());
    setSuccess &= llm_->set_config(std::string("{\"thread_num\":") + std::to_string(options_.threads) + "}");
    setSuccess &= llm_->set_config(std::string("{\"attention_mode\":") + std::to_string(options_.attentionMode) + "}");
    if (!options_.isR1) {
        setSuccess &= llm_->set_config(R"({"use_template":true})");
    } else {
        setSuccess &= llm_->set_config(R"({"use_template":false})");
    }
    if (!setSuccess) {
        std::cerr << "Failed to configure LLM" << std::endl;
        return false;
    }

    std::cout << "kvshare_server init:"
              << " config=" << options_.configPath
              << " backend=" << options_.backend
              << " host=" << options_.host
              << " port=" << options_.port
              << " threads=" << options_.threads
              << " attention_mode=" << options_.attentionMode
              << " reuse_kv=" << (options_.reuseKv ? "true" : "false")
              << " prefix_cache_dir=" << options_.prefixCacheDir
              << " prefix_host_cache_threads="
              << (options_.prefixHostCacheThreads == 0 ? std::string("env/default") : std::to_string(options_.prefixHostCacheThreads))
              << " cuda_prefix_prefetch_threads="
              << (options_.cudaPrefixPrefetchThreads == 0 ? std::string("env/default") : std::to_string(options_.cudaPrefixPrefetchThreads))
              << " opencl_prefix_prefetch_threads="
              << (options_.openclPrefixPrefetchThreads == 0 ? std::string("env/default") : std::to_string(options_.openclPrefixPrefetchThreads))
              << std::endl;
    {
        AUTOTIME;
        if (!llm_->load()) {
            std::cerr << "Failed to load LLM" << std::endl;
            return false;
        }
    }
    tuningPrepare(llm_.get());
    return true;
}

int KvShareServer::start() {
    httplib::Server server;

    server.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
        allowCors(res);
        res.status = 200;
    });

    server.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        allowCors(res);
        json response = {
            {"name", "kvshare_server"},
            {"backend", options_.backend},
            {"attention_mode", options_.attentionMode},
            {"prefix_cache_dir", absolutePath(options_.prefixCacheDir)},
            {"status", "ok"}
        };
        res.set_content(response.dump(), "application/json");
    });

    server.Post("/reset", [&](const httplib::Request&, httplib::Response& res) {
        allowCors(res);
        std::lock_guard<std::mutex> lock(mutex_);
        resetSessionLocked();
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    server.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        allowCors(res);
        ChatRequest chatRequest;
        std::string error;
        if (!parseChatRequest(req, chatRequest, error)) {
            res.status = 400;
            res.set_content(json({{"error", error}}).dump(), "application/json");
            return;
        }
        std::cout << "POST /v1/chat/completions"
                  << " stream=" << (chatRequest.stream ? "true" : "false")
                  << " messages=" << chatRequest.messages.size()
                  << std::endl;
        if (chatRequest.stream) {
            answerStreaming(chatRequest, res);
            return;
        }
        answer(chatRequest, res);
    });

    server.Post("/v1/kv/documents", [&](const httplib::Request& req, httplib::Response& res) {
        allowCors(res);
        handleDocumentCacheRequest(req, res);
    });

    server.Post("/v1/kv/prefix_caches", [&](const httplib::Request& req, httplib::Response& res) {
        allowCors(res);
        handlePrefixCacheRequest(req, res);
    });

    std::cout << "Starting kvshare_server on http://" << options_.host << ":" << options_.port << std::endl;
    server.listen(options_.host.c_str(), options_.port);
    return 0;
}

bool KvShareServer::parseChatRequest(const httplib::Request& req, ChatRequest& chatRequest, std::string& error) const {
    if (!json::accept(req.body)) {
        error = "Invalid JSON in request body.";
        return false;
    }
    json requestJson = json::parse(req.body, nullptr, false);
    if (requestJson.is_discarded()) {
        error = "Invalid JSON in request body.";
        return false;
    }
    if (!requestJson.contains("messages")) {
        error = "Missing required field: messages";
        return false;
    }
    if (!parseMessages(requestJson["messages"], chatRequest.messages, error)) {
        return false;
    }
    if (requestJson.contains("kv_prefix")) {
        if (!requestJson["kv_prefix"].is_object()) {
            error = "Field kv_prefix must be an object";
            return false;
        }
        chatRequest.kvPrefix = requestJson["kv_prefix"];
    }
    chatRequest.model = requestJson.value("model", "undefined-model");
    chatRequest.stream = requestJson.value("stream", false);
    if (requestJson.contains("max_tokens") && !requestJson["max_tokens"].is_number_integer()) {
        error = "Field max_tokens must be an integer";
        return false;
    }
    chatRequest.maxTokens = requestJson.value("max_tokens", -1);
    auto parseFloatField = [&](const char* key, std::optional<float>& target) -> bool {
        if (!requestJson.contains(key)) {
            return true;
        }
        if (!requestJson[key].is_number()) {
            error = std::string("Field ") + key + " must be a number";
            return false;
        }
        target = requestJson[key].get<float>();
        return true;
    };
    auto parseIntField = [&](const char* key, std::optional<int>& target) -> bool {
        if (!requestJson.contains(key)) {
            return true;
        }
        if (!requestJson[key].is_number_integer()) {
            error = std::string("Field ") + key + " must be an integer";
            return false;
        }
        target = requestJson[key].get<int>();
        return true;
    };
    if (!parseFloatField("temperature", chatRequest.temperature) ||
        !parseFloatField("top_p", chatRequest.topP) ||
        !parseIntField("top_k", chatRequest.topK) ||
        !parseFloatField("presence_penalty", chatRequest.presencePenalty) ||
        !parseFloatField("frequency_penalty", chatRequest.frequencyPenalty) ||
        !parseFloatField("repetition_penalty", chatRequest.repetitionPenalty)) {
        return false;
    }
    return true;
}

bool KvShareServer::parseMessages(const json& messagesJson, ChatMessages& messages, std::string& error) const {
    if (!messagesJson.is_array()) {
        error = "Field messages must be an array";
        return false;
    }
    for (const auto& item : messagesJson) {
        if (!item.is_object()) {
            error = "Each message must be an object";
            return false;
        }
        if (!item.contains("role") || !item["role"].is_string()) {
            error = "Each message must contain a string role";
            return false;
        }
        if (!item.contains("content") || !item["content"].is_string()) {
            error = "Each message must contain a string content";
            return false;
        }
        messages.emplace_back(item["role"].get<std::string>(), item["content"].get<std::string>());
    }
    if (messages.empty()) {
        error = "messages must not be empty";
        return false;
    }
    return true;
}

bool KvShareServer::handleDocumentCacheRequest(const httplib::Request& req, httplib::Response& res) {
    if (!json::accept(req.body)) {
        res.status = 400;
        res.set_content(R"({"error":"Invalid JSON in request body."})", "application/json");
        return false;
    }
    json requestJson = json::parse(req.body, nullptr, false);
    json responseJson;
    std::string error;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!buildDocumentCacheLocked(requestJson, responseJson, error)) {
            res.status = 400;
            res.set_content(json({{"error", error}}).dump(), "application/json");
            return false;
        }
    }
    res.set_content(responseJson.dump(2), "application/json");
    return true;
}

bool KvShareServer::handlePrefixCacheRequest(const httplib::Request& req, httplib::Response& res) {
    if (!json::accept(req.body)) {
        res.status = 400;
        res.set_content(R"({"error":"Invalid JSON in request body."})", "application/json");
        return false;
    }
    json requestJson = json::parse(req.body, nullptr, false);
    json responseJson;
    std::string error;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!composePrefixCacheLocked(requestJson, responseJson, error)) {
            res.status = 400;
            res.set_content(json({{"error", error}}).dump(), "application/json");
            return false;
        }
    }
    res.set_content(responseJson.dump(2), "application/json");
    return true;
}

bool KvShareServer::buildDocumentCacheLocked(const json& requestJson, json& responseJson, std::string& error) {
    if ((options_.backend != "cpu" && options_.backend != "cuda" && options_.backend != "opencl") ||
        options_.attentionMode != 8) {
        error = "Document KV cache currently requires backend=cpu, backend=cuda or backend=opencl, and attention_mode=8";
        return false;
    }
    if (!requestJson.is_object()) {
        error = "Document cache request must be an object";
        return false;
    }
    if (!requestJson.contains("id") || !requestJson["id"].is_string()) {
        error = "Document cache request missing string field: id";
        return false;
    }
    std::string id = requestJson["id"].get<std::string>();
    std::string content;
    json source = json::object();
    if (requestJson.contains("content")) {
        if (!requestJson["content"].is_string()) {
            error = "Field content must be a string";
            return false;
        }
        content = requestJson["content"].get<std::string>();
        source["kind"] = "inline";
    } else if (requestJson.contains("path")) {
        if (!requestJson["path"].is_string()) {
            error = "Field path must be a string";
            return false;
        }
        auto path = std::filesystem::path(requestJson["path"].get<std::string>());
        if (!readTextFile(path, content, error)) {
            return false;
        }
        source["kind"] = "path";
        source["path"] = absolutePath(path);
    } else {
        error = "Document cache request needs content or path";
        return false;
    }
    if (content.empty()) {
        error = "Document content must not be empty";
        return false;
    }

    std::string cacheName = requestJson.value("cache_name", documentCacheName(id));
    cacheName = sanitizeCachePart(cacheName);
    bool force = requestJson.value("force", false);
    int layers = layerCountFromConfig(llm_.get());
    json kvLayout = ropeLayoutFromConfig(llm_.get(), options_.backend, error);
    if (!error.empty()) {
        return false;
    }
    std::error_code ec;
    std::filesystem::create_directories(cacheObjectDir(options_.prefixCacheDir, options_.backend, cacheName) / "layers", ec);
    if (ec) {
        error = "Failed to create prefix cache directory: " + options_.prefixCacheDir;
        return false;
    }

    std::vector<int> tokens = llm_->tokenizer_encode(content);
    const auto tokenizerPrefixTokens = llm_->tokenizer_encode("");
    const int strippedTokenizerPrefixTokens = stripTokenizerPrefixTokens(tokens, tokenizerPrefixTokens);
    if (tokens.empty()) {
        error = "Document tokenization produced no tokens";
        return false;
    }

    bool reusedExistingKvCache = false;
    if (force) {
        removePrefixCacheFiles(options_.prefixCacheDir, options_.backend, cacheName, layers);
    }
    if (!force && prefixCacheReady(options_.prefixCacheDir, options_.backend, cacheName, layers)) {
        json existingMeta;
        std::string metaError;
        auto existingMetaPath = metaPathForCache(options_.prefixCacheDir, options_.backend, cacheName);
        if (!loadJsonFile(existingMetaPath, existingMeta, metaError) ||
            !existingMeta.contains("kv_layout") || !existingMeta["kv_layout"].is_object() ||
            existingMeta["kv_layout"].value("key_rope_state", std::string("")) != "canonical_no_rope" ||
            existingMeta.value("tokenizer_prefix_policy", std::string("")) != kDocumentTokenizerPrefixPolicy ||
            !kvLayoutMatchesBackend(existingMeta["kv_layout"], existingMeta, options_.backend)) {
            // 老版本或其他 backend 的缓存不能冒充当前 backend-native canonical key。
            removePrefixCacheFiles(options_.prefixCacheDir, options_.backend, cacheName, layers);
        } else {
            reusedExistingKvCache = true;
        }
    }
    if (!prefixCacheReady(options_.prefixCacheDir, options_.backend, cacheName, layers)) {
        reusedExistingKvCache = false;
        llm_->reset();
        llm_->clearPrefixCacheFile();
        llm_->setPrefixCacheWriteFile(cacheName, MNN::Transformer::PrefixCacheFlagCanonicalNoRopeKey);
        llm_->generate_init(nullptr, "\n");
        llm_->generate(tokens, 0);
        auto* context = llm_->getContext();
        if (context != nullptr && context->status == LlmStatus::INTERNAL_ERROR) {
            llm_->clearPrefixCacheFile();
            error = "LLM internal error while building document cache";
            return false;
        }
        llm_->clearPrefixCacheFile();
        llm_->reset();
    }

    if (!prefixCacheReady(options_.prefixCacheDir, options_.backend, cacheName, layers)) {
        error = "Prefix cache files were not created for document: " + id;
        return false;
    }

    auto tokensPath = tokensPathForCache(options_.prefixCacheDir, options_.backend, cacheName);
    json tokensJson = {
        {"format", "mnn-prefix-cache-tokens-v1"},
        {"id", id},
        {"cache_name", cacheName},
        {"tokenizer_prefix_policy", kDocumentTokenizerPrefixPolicy},
        {"stripped_tokenizer_prefix_tokens", strippedTokenizerPrefixTokens},
        {"token_count", tokens.size()},
        {"token_ids", tokens}
    };
    if (!writeJsonFile(tokensPath, tokensJson, error)) {
        return false;
    }

    auto metaPath = metaPathForCache(options_.prefixCacheDir, options_.backend, cacheName);
    responseJson = {
        {"format", "mnn-prefix-cache-meta-v1"},
        {"type", "document"},
        {"id", id},
        {"cache_name", cacheName},
        {"prefix_cache_dir", absolutePath(options_.prefixCacheDir)},
        {"meta_path", absolutePath(metaPath)},
        {"token_ids_path", absolutePath(tokensPath)},
        {"tokenizer_prefix_policy", kDocumentTokenizerPrefixPolicy},
        {"stripped_tokenizer_prefix_tokens", strippedTokenizerPrefixTokens},
        {"token_count", tokens.size()},
        {"token_ids", tokens},
        {"source", source},
        {"backend", options_.backend},
        {"attention_mode", options_.attentionMode},
        {"layer_count", layers},
        {"cache_hit", reusedExistingKvCache},
        {"cache_status", reusedExistingKvCache ? "reused" : "built"},
        {"kv_layout", kvLayout},
        {"kv", kvFilesForCache(options_.prefixCacheDir, options_.backend, cacheName, layers)}
    };
    if (!writeJsonFile(metaPath, responseJson, error)) {
        return false;
    }
    return true;
}

bool KvShareServer::resolvePrefixSegmentsLocked(const json& segmentsJson, bool requireCanonicalNoRopeKey, bool requirePrefixCacheFiles,
                                                std::vector<int>& mergedTokens,
                                                std::vector<MNN::Transformer::PrefixCacheSegment>& prefixSegments,
                                                json& resolvedSegments, std::string& error) {
    if (!segmentsJson.is_array() || segmentsJson.empty()) {
        error = "Prefix cache request needs a non-empty segments array";
        return false;
    }
    int layers = layerCountFromConfig(llm_.get());
    mergedTokens.clear();
    prefixSegments.clear();
    resolvedSegments = json::array();

    for (const auto& segment : segmentsJson) {
        if (!segment.is_object()) {
            error = "Each segment must be an object";
            return false;
        }
        json segmentMeta;
        std::string resolvedMetaPath;
        if (segment.contains("content") || segment.contains("path")) {
            error = "direct_segments segments must reference an existing backend-native document cache; build content/path with /v1/kv/documents first";
            return false;
        } else {
            std::filesystem::path metaPath;
            if (segment.contains("meta_path")) {
                if (!segment["meta_path"].is_string()) {
                    error = "Segment meta_path must be a string";
                    return false;
                }
                metaPath = segment["meta_path"].get<std::string>();
            } else {
                std::string type = segment.value("type", "document");
                std::string segmentId = segment.value("id", segment.value("cache_name", ""));
                if (segmentId.empty()) {
                    error = "Segment needs id, cache_name, meta_path, content, or path";
                    return false;
                }
                std::string segmentCacheName = segment.value("cache_name", type == "document" ? documentCacheName(segmentId) : segmentId);
                metaPath = metaPathForCache(options_.prefixCacheDir, options_.backend, sanitizeCachePart(segmentCacheName));
            }
            if (!loadJsonFile(metaPath, segmentMeta, error)) {
                return false;
            }
            resolvedMetaPath = segmentMeta.value("meta_path", absolutePath(metaPath));
        }
        if (!segmentMeta.contains("cache_name") || !segmentMeta["cache_name"].is_string()) {
            error = "Segment meta missing cache_name";
            return false;
        }
        auto cacheName = sanitizeCachePart(segmentMeta["cache_name"].get<std::string>());
        json kvLayout = segmentMeta.value("kv_layout", json::object());
        if (requireCanonicalNoRopeKey) {
            if (!kvLayout.is_object() ||
                kvLayout.value("key_rope_state", std::string("")) != "canonical_no_rope" ||
                kvLayout.value("rope_pairing", std::string("")) != "half") {
                error = "Segment prefix cache is not canonical_no_rope; rebuild it with force=true: " + cacheName;
                return false;
            }
            if (!kvLayoutMatchesBackend(kvLayout, segmentMeta, options_.backend)) {
                error = "Segment prefix cache layout does not match backend=" + options_.backend +
                        "; rebuild or convert it for this backend: " + cacheName;
                return false;
            }
        }
        if (requirePrefixCacheFiles && !prefixCacheReady(options_.prefixCacheDir, options_.backend, cacheName, layers)) {
            error = "Segment prefix cache files are not ready: " + cacheName;
            return false;
        }

        error.clear();
        auto segmentTokens = tokensFromMeta(segmentMeta, options_.prefixCacheDir, error);
        if (!error.empty() && segmentTokens.empty()) {
            return false;
        }
        if (segmentTokens.empty()) {
            error = "Segment token cache is empty: " + cacheName;
            return false;
        }
        if (segmentMeta.contains("token_count") && segmentMeta["token_count"].is_number_integer() &&
            segmentMeta["token_count"].get<int>() != static_cast<int>(segmentTokens.size())) {
            error = "Segment token_count does not match token_ids length: " + cacheName;
            return false;
        }

        MNN::Transformer::PrefixCacheSegment prefixSegment;
        prefixSegment.cache_name = cacheName;
        prefixSegment.token_count = static_cast<int>(segmentTokens.size());
        prefixSegment.token_ids = segmentTokens;
        if (requireCanonicalNoRopeKey) {
            prefixSegment.key_rope_state = "canonical_no_rope";
            prefixSegment.rope_dim = kvLayout.value("rope_dim", 0);
            prefixSegment.rope_theta = static_cast<float>(kvLayout.value("rope_theta", 10000.0));
            prefixSegment.rope_pairing = kvLayout.value("rope_pairing", std::string("half"));
            prefixSegment.source_position_base = kvLayout.value("source_position_base", 0);
            prefixSegment.backend = kvLayout.value("backend", segmentMeta.value("backend", std::string("")));
            prefixSegment.layout = kvLayout.value("layout", kvLayout.value("format", std::string("")));
            prefixSegment.dtype = kvLayout.value("dtype", std::string("runtime"));
            prefixSegment.page_size = kvLayout.value("page_size", 0);
        }
        prefixSegments.emplace_back(std::move(prefixSegment));
        mergedTokens.insert(mergedTokens.end(), segmentTokens.begin(), segmentTokens.end());

        json resolved = {
            {"type", segmentMeta.value("type", segment.value("type", "cache"))},
            {"id", segmentMeta.value("id", segment.value("id", ""))},
            {"cache_name", cacheName},
            {"meta_path", resolvedMetaPath},
            {"token_count", segmentTokens.size()},
            {"kv_layout", kvLayout},
            {"kv", segmentMeta.value("kv", json::array())}
        };
        if (segmentMeta.contains("source")) {
            resolved["source"] = segmentMeta["source"];
        }
        resolvedSegments.push_back(resolved);
    }
    return true;
}

bool KvShareServer::composePrefixCacheLocked(const json& requestJson, json& responseJson, std::string& error) {
    if ((options_.backend != "cpu" && options_.backend != "cuda" && options_.backend != "opencl") ||
        options_.attentionMode != 8) {
        error = "direct_segments prefix cache metadata requires backend=cpu, backend=cuda or backend=opencl, and attention_mode=8";
        return false;
    }
    if (!requestJson.is_object()) {
        error = "Prefix cache request must be an object";
        return false;
    }
    if (!requestJson.contains("segments") || !requestJson["segments"].is_array() || requestJson["segments"].empty()) {
        error = "Prefix cache request needs a non-empty segments array";
        return false;
    }
    if (!requestJson.contains("merge_mode") || requestJson.value("merge_mode", std::string("")) != "direct_segments") {
        error = "/v1/kv/prefix_caches only supports merge_mode=direct_segments";
        return false;
    }

    int layers = layerCountFromConfig(llm_.get());
    const auto& segmentsJson = requestJson["segments"];
    std::string id = requestJson.value("id", compositeCacheName(segmentsJson));
    std::string cacheName = requestJson.value("cache_name", "cmp_" + sanitizeCachePart(id));
    cacheName = sanitizeCachePart(cacheName);

    std::vector<int> mergedTokens;
    std::vector<MNN::Transformer::PrefixCacheSegment> prefixSegments;
    json resolvedSegments = json::array();
    json directKvLayout = ropeLayoutFromConfig(llm_.get(), options_.backend, error);
    if (!error.empty()) {
        return false;
    }
    if (!resolvePrefixSegmentsLocked(segmentsJson, true, true, mergedTokens, prefixSegments, resolvedSegments, error)) {
        return false;
    }
    if (mergedTokens.empty()) {
        error = "Merged prefix cache has no tokens";
        return false;
    }

    responseJson = {
        {"format", "mnn-prefix-cache-meta-v1"},
        {"type", "direct_segments"},
        {"id", id},
        {"cache_name", cacheName},
        {"prefix_cache_dir", absolutePath(options_.prefixCacheDir)},
        {"token_count", mergedTokens.size()},
        {"token_ids", mergedTokens},
        {"segments", resolvedSegments},
        {"merge_mode", "direct_segments"},
        {"backend", options_.backend},
        {"attention_mode", options_.attentionMode},
        {"layer_count", layers},
        {"kv_layout", directKvLayout},
        {"note", options_.backend + " direct_segments requires existing backend-native canonical_no_rope prefix KV files; materialization happens inside PrefixAttention."}
    };
    return true;
}

bool KvShareServer::applySamplingConfigLocked(const ChatRequest& request, std::string& error) {
    json samplingConfig;
    samplingConfig["sampler_type"] = "mixed";
    if (request.temperature.has_value()) {
        if (*request.temperature <= 0.0f) {
            error = "Field temperature must be > 0";
            return false;
        }
        samplingConfig["temperature"] = *request.temperature;
    }
    if (request.topP.has_value()) {
        if (*request.topP <= 0.0f || *request.topP > 1.0f) {
            error = "Field top_p must be in (0, 1]";
            return false;
        }
        samplingConfig["top_p"] = *request.topP;
    }
    if (request.topK.has_value()) {
        if (*request.topK <= 0) {
            error = "Field top_k must be > 0";
            return false;
        }
        samplingConfig["top_k"] = *request.topK;
    }
    if (request.presencePenalty.has_value()) {
        samplingConfig["presence_penalty"] = *request.presencePenalty;
    }
    if (request.frequencyPenalty.has_value()) {
        samplingConfig["frequency_penalty"] = *request.frequencyPenalty;
    }
    if (request.repetitionPenalty.has_value()) {
        if (*request.repetitionPenalty <= 0.0f) {
            error = "Field repetition_penalty must be > 0";
            return false;
        }
        samplingConfig["repetition_penalty"] = *request.repetitionPenalty;
    }
    if (samplingConfig.size() == 1) {
        return true;
    }
    if (!llm_->set_config(samplingConfig.dump())) {
        error = "Failed to apply sampling config";
        return false;
    }
    std::cout << "sampling config applied: " << samplingConfig.dump() << std::endl;
    return true;
}

void KvShareServer::answer(const ChatRequest& request, httplib::Response& res) {
    bool ok = false;
    std::string error;
    json prefixInfo = json::object();
    auto answerText = runCompletionLocked(request, nullptr, nullptr, ok, error, &prefixInfo);
    if (!ok) {
        res.status = 500;
        res.set_content(json({{"error", error}}).dump(), "application/json");
        return;
    }

    auto* context = llm_->getContext();
    int prefixTokens = prefixInfo.value("token_count", 0);
    int preludeTokens = prefixInfo.value("prelude_token_count",
                                          prefixInfo.value("system_prompt_token_count", 0));
    json responseJson = {
        {"id", "chatcmpl-" + currentTimeString()},
        {"object", "chat.completion"},
        {"created", static_cast<int>(std::time(nullptr))},
        {"model", request.model},
        {"choices", json::array({
            {
                {"index", 0},
                {"message", {{"role", "assistant"}, {"content", answerText}}},
                {"finish_reason", "stop"}
            }
        })},
        {"usage", {
            {"prompt_tokens", context->prompt_len + prefixTokens + preludeTokens},
            {"completion_tokens", context->gen_seq_len},
            {"total_tokens", context->prompt_len + prefixTokens + preludeTokens + context->gen_seq_len}
        }}
    };
    if (!prefixInfo.empty()) {
        responseJson["kv_prefix"] = prefixInfo;
    }
    res.set_content(responseJson.dump(), "application/json");
}

void KvShareServer::answerStreaming(const ChatRequest& request, httplib::Response& res) {
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    res.set_chunked_content_provider("text/event-stream", [this, request](size_t, httplib::DataSink& sink) {
        bool ok = false;
        std::string error;
        Utf8StreamProcessor processor([&sink, &request](const std::string& utf8Char) {
            bool isEop = utf8Char.find("<eop>") != std::string::npos;
            json sseJson = {
                {"id", "chatcmpl-" + currentTimeString()},
                {"object", "chat.completion.chunk"},
                {"created", static_cast<int>(std::time(nullptr))},
                {"model", request.model},
                {"choices", json::array({
                    {
                        {"delta", isEop ? json::object() : json({{"content", utf8Char}})},
                        {"index", 0},
                        {"finish_reason", isEop ? "stop" : ""}
                    }
                })}
            };
            std::string chunk = "data: " + sseJson.dump() + "\n\n";
            sink.os.write(chunk.c_str(), static_cast<std::streamsize>(chunk.size()));
            sink.os.flush();
        });
        {
            LlmStreamBuffer streamBuffer([&processor](const char* str, size_t len) {
                processor.processStream(str, len);
            });
            std::ostream outputStream(&streamBuffer);
            json prefixInfo;
            runCompletionLocked(request, &outputStream, "<eop>", ok, error, &prefixInfo);
        }
        if (!ok) {
            json errorJson = {{"error", error}};
            std::string chunk = "data: " + errorJson.dump() + "\n\n";
            sink.os.write(chunk.c_str(), static_cast<std::streamsize>(chunk.size()));
            sink.os.flush();
        }
        std::string done = "data: [DONE]\n\n";
        sink.os.write(done.c_str(), static_cast<std::streamsize>(done.size()));
        sink.os.flush();
        sink.done();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return false;
    });
}

bool KvShareServer::applyKvPrefixLocked(const json& kvPrefix, const ChatMessages& messages, std::ostream* output, const char* endWith, int maxTokens, json& prefixInfo, std::string& error) {
    if (!kvPrefix.is_object() || kvPrefix.empty()) {
        return false;
    }
    if (!kvPrefix.contains("segments") || !kvPrefix["segments"].is_array() || kvPrefix["segments"].empty()) {
        error = "kv_prefix requires a non-empty segments array";
        return false;
    }
    if (!kvPrefix.contains("merge_mode") || kvPrefix.value("merge_mode", std::string("")) != "direct_segments") {
        error = "kv_prefix only supports merge_mode=direct_segments";
        return false;
    }
    if (kvPrefix.contains("device_prefetch") && !kvPrefix["device_prefetch"].is_boolean()) {
        error = "kv_prefix.device_prefetch must be boolean when provided";
        return false;
    }
    const bool devicePrefetch = kvPrefix.value("device_prefetch", false);
    if ((options_.backend != "cpu" && options_.backend != "cuda" && options_.backend != "opencl") || options_.attentionMode != 8) {
        error = "direct_segments currently requires backend=cpu, backend=cuda or backend=opencl, and attention_mode=8";
        return false;
    }

    std::string directCacheName = kvPrefix.value("cache_name", "direct_" + sanitizeCachePart(kvPrefix.value("id", compositeCacheName(kvPrefix["segments"]))));
    directCacheName = sanitizeCachePart(directCacheName);
    json directKvLayout = ropeLayoutFromConfig(llm_.get(), options_.backend, error);
    if (!error.empty()) {
        return false;
    }
    std::vector<int> prefixTokens;
    std::vector<MNN::Transformer::PrefixCacheSegment> prefixSegments;
    json resolvedSegments = json::array();
    auto resolveStart = std::chrono::steady_clock::now();
    if (!resolvePrefixSegmentsLocked(kvPrefix["segments"], true, true, prefixTokens, prefixSegments, resolvedSegments, error)) {
        return false;
    }
    double resolveMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - resolveStart).count();
    if (prefixTokens.empty()) {
        error = "kv_prefix resolved to an empty token cache";
        return false;
    }

    const std::string placeholder = kvPrefix.value("placeholder", std::string(kDefaultKVPrefixPlaceholder));
    std::vector<int> preludeTokens;
    std::vector<int> promptTokens;
    std::string promptProtocol = "legacy-system-prefix";
    auto renderedPrompt = llm_->apply_chat_template(messages);
    std::string preludeText;
    std::string suffixText;
    bool hasPlaceholder = splitRenderedPromptAtPlaceholder(renderedPrompt, placeholder, preludeText, suffixText, error);
    if (!error.empty()) {
        return false;
    }
    if (hasPlaceholder) {
        promptProtocol = "chat-template-placeholder-v1";
        if (!preludeText.empty()) {
            preludeTokens = llm_->tokenizer_encode(preludeText);
        }
        promptTokens = llm_->tokenizer_encode(suffixText);
        auto tokenizerPrefixTokens = llm_->tokenizer_encode("");
        stripTokenizerPrefixTokens(promptTokens, tokenizerPrefixTokens);
        if (preludeTokens.empty() && !preludeText.empty()) {
            error = "direct_segments prompt prelude tokenization produced no tokens";
            return false;
        }
    } else {
        auto systemMessages = collectSystemMessages(messages);
        auto promptMessages = collectNonSystemMessages(messages);
        if (promptMessages.empty()) {
            error = "direct_segments requires at least one non-system message for the current prompt suffix";
            return false;
        }

        if (!systemMessages.empty()) {
            auto systemPrompt = llm_->apply_chat_template(systemMessages, false);
            if (systemPrompt.empty()) {
                error = "Failed to apply system prompt chat template";
                return false;
            }
            preludeTokens = llm_->tokenizer_encode(systemPrompt);
            if (preludeTokens.empty()) {
                error = "System prompt tokenization produced no tokens";
                return false;
            }
        }

        auto prompt = llm_->apply_chat_template(promptMessages);
        if (prompt.empty()) {
            error = "Failed to apply current prompt chat template";
            return false;
        }
        promptTokens = llm_->tokenizer_encode(prompt);
        if (!preludeTokens.empty()) {
            auto tokenizerPrefixTokens = llm_->tokenizer_encode("");
            stripTokenizerPrefixTokens(promptTokens, tokenizerPrefixTokens);
        }
    }
    if (promptTokens.empty()) {
        error = "Prompt tokenization produced no tokens";
        return false;
    }

    llm_->clearPrefixCacheFile();
    llm_->generate_init(output, endWith);
    if (!preludeTokens.empty()) {
        llm_->generate(preludeTokens, 0);
    }
    auto prefetchSubmitStart = std::chrono::steady_clock::now();
    if (!llm_->setPrefixCacheSegments(prefixSegments, devicePrefetch, static_cast<int>(promptTokens.size()))) {
        llm_->clearPrefixCacheSegments();
        error = "Failed to set direct segment prefix cache";
        return false;
    }
    double prefetchSubmitMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prefetchSubmitStart).count();
    llm_->generate(promptTokens, maxTokens);
    llm_->clearPrefixCacheSegments();

    prefixInfo = {
        {"format", "mnn-prefix-cache-meta-v1"},
        {"type", "direct_segments"},
        {"id", kvPrefix.value("id", compositeCacheName(kvPrefix["segments"]))},
        {"cache_name", directCacheName},
        {"prefix_cache_dir", absolutePath(options_.prefixCacheDir)},
        {"token_count", prefixTokens.size()},
        {"token_ids", prefixTokens},
        {"system_prompt_token_count", preludeTokens.size()},
        {"prelude_token_count", preludeTokens.size()},
        {"prompt_token_count", promptTokens.size()},
        {"segments", resolvedSegments},
        {"merge_mode", "direct_segments"},
        {"prompt_protocol", promptProtocol},
        {"placeholder", hasPlaceholder ? placeholder : ""},
        {"device_prefetch", devicePrefetch},
        {"kv_layout", directKvLayout},
        {"backend", options_.backend},
        {"attention_mode", options_.attentionMode},
        {"layer_count", layerCountFromConfig(llm_.get())},
        {"resolve_ms", resolveMs},
        {"prefetch_submit_ms", prefetchSubmitMs},
        {"note", options_.backend + " PrefixAttention consumed backend-native prefix KV via direct_segments."}
    };
    return true;
}

std::string KvShareServer::runCompletionLocked(const ChatRequest& request, std::ostream* output, const char* endWith, bool& ok, std::string& error, json* prefixInfo) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto messages = maybeConvertToR1(request.messages);
    if (!applySamplingConfigLocked(request, error)) {
        ok = false;
        std::cout << "completion failed: " << error << std::endl;
        return {};
    }
    json localPrefixInfo = json::object();
    if (!request.kvPrefix.empty()) {
        if (!applyKvPrefixLocked(request.kvPrefix, messages, output, endWith, request.maxTokens, localPrefixInfo, error)) {
            ok = false;
            std::cout << "completion failed: " << error << std::endl;
            return {};
        }
    } else {
        llm_->reset();
        llm_->clearPrefixCacheFile();
        llm_->response(messages, output, endWith, request.maxTokens);
    }
    auto* context = llm_->getContext();
    if (context == nullptr) {
        error = "Missing LLM context";
        ok = false;
        std::cout << "completion failed: " << error << std::endl;
        return {};
    }
    if (context->status == LlmStatus::INTERNAL_ERROR) {
        error = "LLM internal error";
        ok = false;
        std::cout << "completion failed: " << error << std::endl;
        return {};
    }
    const double prefillSeconds = static_cast<double>(context->prefill_us) / 1e6;
    const double decodeSeconds = static_cast<double>(context->decode_us) / 1e6;
    const double prefillTps = prefillSeconds > 0.0 ? static_cast<double>(context->prompt_len) / prefillSeconds : 0.0;
    const double decodeTps = decodeSeconds > 0.0 ? static_cast<double>(context->gen_seq_len) / decodeSeconds : 0.0;
    std::string answer = context->generate_str;
    if (prefixInfo != nullptr) {
        *prefixInfo = localPrefixInfo;
    }
    updateSessionLocked(request.messages, answer);
    hasActiveSession_ = true;
    ok = true;
    std::cout << "completion ok"
              << " prompt_tokens=" << context->prompt_len
              << " completion_tokens=" << context->gen_seq_len
              << std::endl;
    std::cout << "prefill stats"
              << " tokens=" << context->prompt_len
              << " elapsed_s=" << prefillSeconds
              << " tps=" << prefillTps
              << std::endl;
    std::cout << "decode stats"
              << " tokens=" << context->gen_seq_len
              << " elapsed_s=" << decodeSeconds
              << " tps=" << decodeTps
              << std::endl;
    return answer;
}

ChatMessages KvShareServer::maybeConvertToR1(const ChatMessages& prompts) const {
    if (!options_.isR1) {
        return prompts;
    }
    ChatMessages result;
    result.emplace_back("system", "<|begin_of_sentence|>You are a helpful assistant.");
    if (prompts.empty()) {
        return result;
    }
    auto iter = prompts.begin();
    for (; iter != prompts.end() - 1; ++iter) {
        if (iter->first == "system") {
            continue;
        }
        if (iter->first == "assistant") {
            result.emplace_back("assistant", getR1AssistantString(iter->second));
        } else if (iter->first == "user") {
            result.emplace_back("user", getR1UserString(iter->second));
        }
    }
    if (iter->first == "user") {
        result.emplace_back("user", getR1UserString(iter->second));
    } else {
        result.emplace_back("assistant", getR1AssistantString(iter->second));
    }
    return result;
}

void KvShareServer::updateSessionLocked(const ChatMessages& requestMessages, const std::string& assistantMessage) {
    sessionMessages_ = requestMessages;
    sessionMessages_.emplace_back("assistant", assistantMessage);
}

void KvShareServer::resetSessionLocked() {
    if (llm_) {
        llm_->reset();
    }
    sessionMessages_.clear();
    hasActiveSession_ = false;
}

} // namespace kvshare

int main(int argc, char** argv) {
    kvshare::KvShareServer::Options options;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-c") {
            if (++i >= argc) {
                std::cerr << "Missing value for -c" << std::endl;
                return 1;
            }
            options.configPath = argv[i];
        } else if (arg == "--host") {
            if (++i >= argc) {
                std::cerr << "Missing value for --host" << std::endl;
                return 1;
            }
            options.host = argv[i];
        } else if (arg == "--port") {
            if (++i >= argc) {
                std::cerr << "Missing value for --port" << std::endl;
                return 1;
            }
            options.port = std::atoi(argv[i]);
        } else if (arg == "--threads") {
            if (++i >= argc) {
                std::cerr << "Missing value for --threads" << std::endl;
                return 1;
            }
            options.threads = std::atoi(argv[i]);
        } else if (arg == "--attention-mode") {
            if (++i >= argc) {
                std::cerr << "Missing value for --attention-mode" << std::endl;
                return 1;
            }
            options.attentionMode = std::atoi(argv[i]);
        } else if (arg == "--backend") {
            if (++i >= argc) {
                std::cerr << "Missing value for --backend" << std::endl;
                return 1;
            }
            options.backend = argv[i];
        } else if (arg == "--prefix-cache-dir") {
            if (++i >= argc) {
                std::cerr << "Missing value for --prefix-cache-dir" << std::endl;
                return 1;
            }
            options.prefixCacheDir = argv[i];
        } else if (arg == "--prefix-host-cache-threads") {
            if (++i >= argc) {
                std::cerr << "Missing value for --prefix-host-cache-threads" << std::endl;
                return 1;
            }
            options.prefixHostCacheThreads = std::atoi(argv[i]);
            if (!kvshare::isValidPrefixHostCacheThreads(options.prefixHostCacheThreads)) {
                std::cerr << "Invalid value for --prefix-host-cache-threads: " << argv[i]
                          << " (expected 4, 8, or 16)" << std::endl;
                return 1;
            }
        } else if (arg == "--cuda-prefix-prefetch-threads") {
            if (++i >= argc) {
                std::cerr << "Missing value for --cuda-prefix-prefetch-threads" << std::endl;
                return 1;
            }
            options.cudaPrefixPrefetchThreads = std::atoi(argv[i]);
            if (!kvshare::isValidCudaPrefixPrefetchThreads(options.cudaPrefixPrefetchThreads)) {
                std::cerr << "Invalid value for --cuda-prefix-prefetch-threads: " << argv[i]
                          << " (expected 4, 8, or 16)" << std::endl;
                return 1;
            }
        } else if (arg == "--opencl-prefix-prefetch-threads") {
            if (++i >= argc) {
                std::cerr << "Missing value for --opencl-prefix-prefetch-threads" << std::endl;
                return 1;
            }
            options.openclPrefixPrefetchThreads = std::atoi(argv[i]);
            if (!kvshare::isValidOpenCLPrefixPrefetchThreads(options.openclPrefixPrefetchThreads)) {
                std::cerr << "Invalid value for --opencl-prefix-prefetch-threads: " << argv[i]
                          << " (expected 1)" << std::endl;
                return 1;
            }
        } else if (arg == "--reuse-kv") {
            if (++i >= argc) {
                std::cerr << "Missing value for --reuse-kv" << std::endl;
                return 1;
            }
            std::string value = argv[i];
            std::transform(value.begin(), value.end(), value.begin(), ::tolower);
            if (value == "1" || value == "true" || value == "yes" || value == "on") {
                options.reuseKv = true;
            } else if (value == "0" || value == "false" || value == "no" || value == "off") {
                options.reuseKv = false;
            } else {
                std::cerr << "Invalid value for --reuse-kv: " << argv[i] << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    if (options.configPath.empty()) {
        std::cerr << "Usage: kvshare_server -c <config.json> [--host 127.0.0.1] [--port 9091] [--threads 4] [--attention-mode 8] [--backend cpu] [--prefix-cache-dir .cache/kvshare/prefixcache] [--prefix-host-cache-threads 4|8|16] [--cuda-prefix-prefetch-threads 4|8|16] [--opencl-prefix-prefetch-threads 1] [--reuse-kv false]" << std::endl;
        return 1;
    }
    std::string lowerModelName = options.configPath;
    std::transform(lowerModelName.begin(), lowerModelName.end(), lowerModelName.begin(), ::tolower);
    options.isR1 = lowerModelName.find("deepseek-r1") != std::string::npos;

    kvshare::KvShareServer server(std::move(options));
    if (!server.init()) {
        return 1;
    }
    return server.start();
}
