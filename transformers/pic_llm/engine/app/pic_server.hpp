//
//  pic_server.hpp
//  MNN
//

#pragma once

#include "httplib.h"
#include "jsonhpp/json.hpp"
#include "llm/llm.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace pic {

using json = nlohmann::json;

struct PicServerConfig {
    std::string configPath;
    std::string host = "0.0.0.0";
    int port = 9091;
    std::string kvCacheDir = ".cache/kvshare/kvcache";
    std::string servedModelName = "mnn-pic-model";
};

class PicServer {
public:
    explicit PicServer(PicServerConfig config);

    bool load();
    bool start();

private:
    void allowCors(httplib::Response& res) const;
    void handleRoot(const httplib::Request& req, httplib::Response& res);
    void handleHealth(const httplib::Request& req, httplib::Response& res);
    void handleModels(const httplib::Request& req, httplib::Response& res);
    void handleReset(const httplib::Request& req, httplib::Response& res);
    void handlePrefillText(const httplib::Request& req, httplib::Response& res);
    void handlePicCaches(const httplib::Request& req, httplib::Response& res);
    void handleChatCompletions(const httplib::Request& req, httplib::Response& res);

    bool buildTextCache(const json& request, json& response, std::string& error);
    bool buildPicCache(const json& request, json& response, std::string& error);
    bool completeChatBatch(const std::vector<json>& requests, json& response, std::string& error);
    bool completeChatBatchItem(const json& request, json& response, std::string& error);
    json runtimeInfo() const;
    json modelConfig() const;
    std::string runtimeBackend() const;

    PicServerConfig mConfig;
    std::unique_ptr<MNN::Transformer::Llm> mLlm;
    mutable std::mutex mMutex;
};

} // namespace pic
