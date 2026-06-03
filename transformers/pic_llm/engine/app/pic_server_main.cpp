//
//  pic_server_main.cpp
//  MNN
//

#include "pic_server.hpp"

#include <cstdlib>
#include <iostream>

namespace {

void printUsage(const char* argv0) {
    std::cout << "Usage: " << argv0
              << " --config config.json [--host 0.0.0.0] [--port 9091]"
              << " [--kv-cache-dir .cache/kvshare/kvcache] [--model name]\n";
}

} // namespace

int main(int argc, const char* argv[]) {
    pic::PicServerConfig config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto needValue = [&](const char* name) -> const char* {
            if (++i >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                printUsage(argv[0]);
                std::exit(1);
            }
            return argv[i];
        };
        if (arg == "--config" || arg == "-c") {
            config.configPath = needValue(arg.c_str());
        } else if (arg == "--host") {
            config.host = needValue(arg.c_str());
        } else if (arg == "--port") {
            config.port = std::atoi(needValue(arg.c_str()));
        } else if (arg == "--kv-cache-dir") {
            config.kvCacheDir = needValue(arg.c_str());
        } else if (arg == "--model") {
            config.servedModelName = needValue(arg.c_str());
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (config.configPath.empty() && arg.find('-') != 0) {
            config.configPath = arg;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    if (config.configPath.empty()) {
        printUsage(argv[0]);
        return 1;
    }
    if (config.port <= 0) {
        std::cerr << "Invalid port: " << config.port << "\n";
        return 1;
    }

    pic::PicServer server(config);
    if (!server.load()) {
        return 1;
    }
    if (!server.start()) {
        std::cerr << "Failed to start PIC server\n";
        return 1;
    }
    return 0;
}
