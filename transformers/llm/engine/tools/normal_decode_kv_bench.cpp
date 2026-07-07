#include "llm/llm.hpp"

#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace MNN::Transformer;

struct Args {
    std::string config;
    std::string backend = "opencl";
    int prompt = 512;
    int generate = 128;
    int stepTokens = 1;
    int threads = 4;
    int precision = 2;
    int memory = 2;
    int power = 0;
    std::string readyFile;
    std::string goFile;
    std::string jsonFile;
};

static void usage(const char* prog) {
    std::fprintf(stderr,
                 "usage: %s -m config.json -a opencl -p prompt_tokens -n decode_tokens "
                 "[--step-tokens n] --ready-file path --go-file path -j result.json\n",
                 prog);
}

static bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto needValue = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "-m" || arg == "--model") {
            auto v = needValue(arg.c_str());
            if (!v) return false;
            args.config = v;
        } else if (arg == "-a" || arg == "--backend") {
            auto v = needValue(arg.c_str());
            if (!v) return false;
            args.backend = v;
        } else if (arg == "-p" || arg == "--prompt") {
            auto v = needValue(arg.c_str());
            if (!v) return false;
            args.prompt = std::atoi(v);
        } else if (arg == "-n" || arg == "--generate") {
            auto v = needValue(arg.c_str());
            if (!v) return false;
            args.generate = std::atoi(v);
        } else if (arg == "--step-tokens") {
            auto v = needValue(arg.c_str());
            if (!v) return false;
            args.stepTokens = std::atoi(v);
        } else if (arg == "-t" || arg == "--threads") {
            auto v = needValue(arg.c_str());
            if (!v) return false;
            args.threads = std::atoi(v);
        } else if (arg == "-c" || arg == "--precision") {
            auto v = needValue(arg.c_str());
            if (!v) return false;
            args.precision = std::atoi(v);
        } else if (arg == "--memory") {
            auto v = needValue(arg.c_str());
            if (!v) return false;
            args.memory = std::atoi(v);
        } else if (arg == "--power") {
            auto v = needValue(arg.c_str());
            if (!v) return false;
            args.power = std::atoi(v);
        } else if (arg == "--ready-file") {
            auto v = needValue(arg.c_str());
            if (!v) return false;
            args.readyFile = v;
        } else if (arg == "--go-file") {
            auto v = needValue(arg.c_str());
            if (!v) return false;
            args.goFile = v;
        } else if (arg == "-j" || arg == "--json") {
            auto v = needValue(arg.c_str());
            if (!v) return false;
            args.jsonFile = v;
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return false;
        }
    }
    return !args.config.empty() && args.prompt > 0 && args.generate > 0 && args.stepTokens > 0;
}

static int backendId(const std::string& backend) {
    if (backend == "metal") return 1;
    if (backend == "cuda") return 2;
    if (backend == "opencl") return 3;
    return 0;
}

static bool configureLlm(Llm* llm, const Args& args) {
    std::map<int, std::string> level = {{0, "normal"}, {1, "high"}, {2, "low"}};
    std::map<int, std::string> backend = {{0, "cpu"}, {1, "metal"}, {2, "cuda"}, {3, "opencl"}};
    const int backendType = backendId(args.backend);
    bool ok = true;
    ok &= llm->set_config("{\"async\":false}");
    ok &= llm->set_config("{\"reuse_kv\":false}");
    ok &= llm->set_config("{\"precision\":\"" + level[args.precision] + "\"}");
    ok &= llm->set_config("{\"memory\":\"" + level[args.memory] + "\"}");
    ok &= llm->set_config("{\"power\":\"" + level[args.power] + "\"}");
    ok &= llm->set_config("{\"backend_type\":\"" + backend[backendType] + "\"}");
    ok &= llm->set_config("{\"thread_num\":" + std::to_string(args.threads) + "}");
    return ok;
}

static void writeFile(const std::string& path, const std::string& text) {
    if (path.empty()) {
        return;
    }
    std::ofstream os(path);
    os << text;
}

static void waitForGo(const std::string& path) {
    if (path.empty()) {
        return;
    }
    while (true) {
        std::ifstream is(path);
        if (is.good()) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) {
        usage(argv[0]);
        return 2;
    }

    MNN::BackendConfig backendConfig;
    auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
    MNN::Express::ExecutorScope scope(executor);

    std::unique_ptr<Llm, void (*)(Llm*)> llm(Llm::createLLM(args.config), Llm::destroy);
    if (!llm || !configureLlm(llm.get(), args)) {
        std::fprintf(stderr, "failed to create/configure LLM\n");
        return 3;
    }

    MNN::Timer loadTimer;
    if (!llm->load()) {
        std::fprintf(stderr, "failed to load LLM\n");
        return 4;
    }
    const double loadS = loadTimer.durationInUs() / 1000000.0;

    llm->set_config("{\"async\":false}");
    llm->set_config("{\"max_new_tokens\":1}");
    std::vector<int> tokens(args.prompt, 16);
    auto* context = llm->getContext();

    MNN::Timer prefillTimer;
    llm->response(tokens, nullptr, nullptr, 0);
    const double prefillWallS = prefillTimer.durationInUs() / 1000000.0;
    const int64_t prefillUs = context->prefill_us;
    if (context->status == LlmStatus::INTERNAL_ERROR) {
        std::fprintf(stderr, "prefill failed\n");
        return 5;
    }

    writeFile(args.readyFile, "ready\n");
    waitForGo(args.goFile);

    const int64_t decodeUsBefore = context->decode_us;
    const int generatedBefore = context->gen_seq_len;
    MNN::Timer decodeWallTimer;
    const int64_t prefillUsBeforeDecode = context->prefill_us;
    int generated = 0;
    if (args.stepTokens == 1) {
        for (; generated < args.generate && !llm->stoped(); ++generated) {
            llm->generate(1);
            if (context->status == LlmStatus::INTERNAL_ERROR) {
                std::fprintf(stderr, "decode failed at token %d\n", generated);
                return 6;
            }
        }
    } else {
        while (generated < args.generate) {
            const int step = std::min(args.stepTokens, args.generate - generated);
            std::vector<int> stepTokens(step, 16);
            llm->generate(stepTokens, 0);
            if (context->status == LlmStatus::INTERNAL_ERROR) {
                std::fprintf(stderr, "chunk input failed at token %d step %d\n", generated, step);
                return 6;
            }
            generated += step;
        }
    }
    const double decodeWallS = decodeWallTimer.durationInUs() / 1000000.0;
    const int64_t decodeUs = args.stepTokens == 1
        ? context->decode_us - decodeUsBefore
        : context->prefill_us - prefillUsBeforeDecode;
    const int generatedDelta = context->gen_seq_len - generatedBefore;
    const int measuredTokens = generatedDelta > 0 ? generatedDelta : generated;
    const double decodeKernelS = decodeUs / 1000000.0;
    const double tps = decodeKernelS > 0.0 ? measuredTokens / decodeKernelS : 0.0;
    const double tpotMs = measuredTokens > 0 ? decodeUs / 1000.0 / measuredTokens : 0.0;

    char buffer[2048];
    std::snprintf(buffer, sizeof(buffer),
                  "{"
                  "\"model_config\":\"%s\","
                  "\"backend\":\"%s\","
                  "\"prompt_len\":%d,"
                  "\"step_tokens\":%d,"
                  "\"generated_tokens\":%d,"
                  "\"target_generated_tokens\":%d,"
                  "\"load_time_s\":%.9f,"
                  "\"prefill_wall_s\":%.9f,"
                  "\"prefill_us\":%lld,"
                  "\"decode_wall_s\":%.9f,"
                  "\"decode_us\":%lld,"
                  "\"decode_tpot_ms\":%.9f,"
                  "\"decode_tps\":%.9f"
                  "}\n",
                  args.config.c_str(), args.backend.c_str(), args.prompt, args.stepTokens, measuredTokens, args.generate,
                  loadS, prefillWallS, static_cast<long long>(prefillUs), decodeWallS,
                  static_cast<long long>(decodeUs), tpotMs, tps);
    if (!args.jsonFile.empty()) {
        writeFile(args.jsonFile, buffer);
    }
    std::printf("%s", buffer);
    return measuredTokens > 0 ? 0 : 7;
}
