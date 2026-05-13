#pragma once

#include "httplib.h"
#include "jsonhpp/json.hpp"
#include "llm/llm.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <streambuf>
#include <string>
#include <vector>

namespace kvshare {

using nlohmann::json;

class LlmStreamBuffer : public std::streambuf {
public:
    using CallBack = std::function<void(const char* str, size_t len)>;

    explicit LlmStreamBuffer(CallBack callback) : callback_(std::move(callback)) {
    }

protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        if (callback_) {
            callback_(s, static_cast<size_t>(n));
        }
        return n;
    }

private:
    CallBack callback_;
};

class Utf8StreamProcessor {
public:
    explicit Utf8StreamProcessor(std::function<void(const std::string&)> callback) : callback_(std::move(callback)) {
    }

    void processStream(const char* str, size_t len);

private:
    static int utf8CharLength(unsigned char byte);

private:
    std::string utf8Buffer_;
    std::function<void(const std::string&)> callback_;
};

class KvShareServer {
public:
    struct Options {
        std::string host = "127.0.0.1";
        int port = 9091;
        int attentionMode = 8;
        int threads = 4;
        bool isR1 = false;
        bool reuseKv = false;
        int prefixHostCacheThreads = 0;
        int cudaPrefixPrefetchThreads = 0;
        int openclPrefixPrefetchThreads = 0;
        std::string backend = "cpu";
        std::string configPath;
        std::string prefixCacheDir = ".cache/kvshare/prefixcache";
    };

    explicit KvShareServer(Options options);

    bool init();
    int start();

private:
    struct ChatRequest {
        std::string model = "undefined-model";
        bool stream = false;
        int maxTokens = -1;
        std::optional<float> temperature;
        std::optional<float> topP;
        std::optional<int> topK;
        std::optional<float> presencePenalty;
        std::optional<float> frequencyPenalty;
        std::optional<float> repetitionPenalty;
        MNN::Transformer::ChatMessages messages;
        json kvPrefix = json::object();
    };

    bool parseChatRequest(const httplib::Request& req, ChatRequest& chatRequest, std::string& error) const;
    bool parseMessages(const json& messagesJson, MNN::Transformer::ChatMessages& messages, std::string& error) const;
    bool handleDocumentCacheRequest(const httplib::Request& req, httplib::Response& res);
    bool handlePrefixCacheRequest(const httplib::Request& req, httplib::Response& res);
    bool applySamplingConfigLocked(const ChatRequest& request, std::string& error);
    void answer(const ChatRequest& request, httplib::Response& res);
    void answerStreaming(const ChatRequest& request, httplib::Response& res);
    std::string runCompletionLocked(const ChatRequest& request, std::ostream* output, const char* endWith, bool& ok, std::string& error, json* prefixInfo = nullptr);
    MNN::Transformer::ChatMessages maybeConvertToR1(const MNN::Transformer::ChatMessages& prompts) const;
    void updateSessionLocked(const MNN::Transformer::ChatMessages& requestMessages, const std::string& assistantMessage);
    void resetSessionLocked();
    bool buildDocumentCacheLocked(const json& requestJson, json& responseJson, std::string& error);
    bool composePrefixCacheLocked(const json& requestJson, json& responseJson, std::string& error);
    bool resolvePrefixSegmentsLocked(const json& segmentsJson, bool requireCanonicalNoRopeKey, bool requirePrefixCacheFiles,
                                     std::vector<int>& mergedTokens,
                                     std::vector<MNN::Transformer::PrefixCacheSegment>& prefixSegments,
                                     json& resolvedSegments, std::string& error);
    bool applyKvPrefixLocked(const json& kvPrefix, const MNN::Transformer::ChatMessages& messages, std::ostream* output, const char* endWith, int maxTokens, json& prefixInfo, std::string& error);

private:
    Options options_;
    std::unique_ptr<MNN::Transformer::Llm> llm_;
    mutable std::mutex mutex_;
    MNN::Transformer::ChatMessages sessionMessages_;
    bool hasActiveSession_ = false;
};

} // namespace kvshare
