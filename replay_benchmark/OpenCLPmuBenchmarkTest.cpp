#include "OpenCLPmuBenchmark.hpp"
#include "ModelPmuBenchmark.hpp"
#include "ReplayRecord.hpp"

#if defined(MNN_REPLAY_HAS_OPENCL)
#include "backend/opencl/core/OpenCLBackend.hpp"
#endif

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "rapidjson/document.h"

using namespace MNN::Replay;

int main() {
    assert(openclPmuCaseCount() >= 14);
    assert(findOpenclPmuCase("buffer_reuse_small") != nullptr);
    if (findOpenclPmuCase("buffer_stride") == nullptr || findOpenclPmuCase("buffer_pchase") == nullptr ||
        findOpenclPmuCase("buffer_vec4") == nullptr || findOpenclPmuCase("fp32_throughput") == nullptr ||
        findOpenclPmuCase("fp16_throughput") == nullptr) {
        return 7;
    }
    if (findOpenclPmuCase("image_stride_x") == nullptr || findOpenclPmuCase("image_stride_y") == nullptr ||
        findOpenclPmuCase("constant_bandwidth") == nullptr || findOpenclPmuCase("local_bandwidth") == nullptr ||
        findOpenclPmuCase("local_barrier") == nullptr || findOpenclPmuCase("atomic_contended") == nullptr ||
        findOpenclPmuCase("atomic_distributed") == nullptr) {
        return 8;
    }
    assert(findOpenclPmuCase("texture_linear") != nullptr);
    assert(findOpenclPmuCase("local_memory") != nullptr);
    assert(findOpenclPmuCase("does_not_exist") == nullptr);

    const auto selected = parseOpenclPmuCaseList("buffer_reuse_small,texture_linear");
    assert(selected.size() == 2);
    assert(selected[0] == "buffer_reuse_small");
    assert(parseOpenclPmuCaseList("all").size() == openclPmuCaseCount());

    const auto bufferGeometry = makeOpenclPmuGeometry(*findOpenclPmuCase("buffer_stream_large"), 262145, 128);
    assert(bufferGeometry.usesLocalSize);
    assert(bufferGeometry.global[0] == 262272);
    assert(bufferGeometry.global[1] == 1);
    assert(bufferGeometry.local[0] == 128);
    assert(bufferGeometry.local[1] == 1);

    const auto pointerChaseGeometry = makeOpenclPmuGeometry(*findOpenclPmuCase("buffer_pchase"), 262145, 128);
    assert(!pointerChaseGeometry.usesLocalSize);
    assert(pointerChaseGeometry.global[0] == 1);
    assert(pointerChaseGeometry.local[0] == 0);

    const auto imageGeometry = makeOpenclPmuGeometry(*findOpenclPmuCase("image_reuse"), 262145, 128);
    assert(!imageGeometry.usesLocalSize);
    assert(imageGeometry.global[0] == 256);
    assert(imageGeometry.global[1] == 256);
    assert(imageGeometry.local[0] == 0);
    assert(imageGeometry.local[1] == 0);

    const auto responsive = classifyPmuSignal(10, 100, true, 20);
    assert(responsive.readable && responsive.responsive && responsive.discriminative && responsive.valid);
    const auto quiet = classifyPmuSignal(10, 12, true, 20);
    assert(quiet.readable && !quiet.responsive && !quiet.valid);

    const auto positiveModelDelta = makeModelPmuDelta(100, 275);
    assert(positiveModelDelta.control == 100 && positiveModelDelta.workload == 275 &&
           positiveModelDelta.delta == 175);
    const auto negativeModelDelta = makeModelPmuDelta(275, 100);
    assert(negativeModelDelta.delta == -175);
    ModelPmuRunConfig validModelConfig;
    validModelConfig.model = "model.mnn";
    validModelConfig.event = "gpu_active_cycles";
    validModelConfig.output = "/tmp/model-pmu.json";
    std::string modelConfigError;
    assert(validateModelPmuRunConfig(validModelConfig, &modelConfigError));
    validModelConfig.forward = MNN_FORWARD_VULKAN;
    assert(validateModelPmuRunConfig(validModelConfig, &modelConfigError));
    validModelConfig.event = "gpu_active_cycles,compute_tasks";
    assert(!validateModelPmuRunConfig(validModelConfig, &modelConfigError));

#if defined(MNN_REPLAY_HAS_OPENCL)
    // This crosses the replay executable -> MNN_CL shared-library boundary.
    // It must use the same OpenCLSymbols singleton as the backend library.
    auto* symbols = MNN::OpenCLSymbolsOperator::createOpenCLSymbolsOperatorSingleInstance();
    if (symbols == nullptr || MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr() == nullptr) {
        return 1;
    }
    MNN::BackendConfig config;
    config.precision = MNN::BackendConfig::Precision_Normal;
    MNN::Backend::Info info;
    info.type = MNN_FORWARD_OPENCL;
    info.user = &config;
    MNN::OpenCL::CLRuntime runtime(info);
    assert(!runtime.isCLRuntimeError());

    // Device integration coverage is opt-in because host CI may not have a
    // GPU OpenCL implementation. This must include the image path: malformed
    // image sampler arguments can crash some Adreno drivers in setArg().
    if (std::getenv("MNN_RUN_OPENCL_PMU_RUNTIME_TEST") != nullptr) {
        MNN::Replay::Options options;
        options.openclPmuBench = true;
        options.openclPmuCase = "image_reuse";
        options.openclPmuIterations = 1;
        options.openclPmuWorkloadRuns = 3;
        options.openclPmuWarmupRuns = 2;
        options.openclPmuSize = 262144;
        options.perfCounterEvents = "gpu_active_cycles";
        options.perfCounterOutput = "/tmp/mnn-opencl-pmu-phase1.json";
        if (!runOpenCLPmuBenchmark(options)) {
            return 2;
        }
        std::ifstream input(options.perfCounterOutput);
        std::stringstream contents;
        contents << input.rdbuf();
        rapidjson::Document report;
        report.Parse(contents.str().c_str());
        if (report.HasParseError() || !report.IsObject() || !report.HasMember("cases") ||
            !report["cases"].IsArray() || report["cases"].Empty()) {
            return 3;
        }
        const auto& item = report["cases"][0];
        if (!item.IsObject() || !item.HasMember("workload_runs") ||
            !item["workload_runs"].IsInt() || item["workload_runs"].GetInt() != 3 ||
            !item.HasMember("warmup_runs") || !item["warmup_runs"].IsInt() ||
            item["warmup_runs"].GetInt() != 2) {
            return 4;
        }
        if (!item.HasMember("status") || !item["status"].IsString() ||
            std::string(item["status"].GetString()) != "ok" ||
            !item.HasMember("error") || !item["error"].IsString() ||
            item["error"].GetStringLength() != 0) {
            return 5;
        }
        if (!item.HasMember("global_size") || !item["global_size"].IsObject() ||
            !item["global_size"].HasMember("x") || !item["global_size"]["x"].IsUint64() ||
            item["global_size"]["x"].GetUint64() != 256 ||
            !item["global_size"].HasMember("y") || !item["global_size"]["y"].IsUint64() ||
            item["global_size"]["y"].GetUint64() != 256 ||
            !item.HasMember("local_size") || !item["local_size"].IsObject() ||
            !item["local_size"].HasMember("x") || !item["local_size"]["x"].IsUint64() ||
            item["local_size"]["x"].GetUint64() != 0 ||
            !item.HasMember("uses_local_size") || item["uses_local_size"].GetBool()) {
            return 6;
        }
        if (!item.HasMember("prepare_count") || !item["prepare_count"].IsInt() ||
            item["prepare_count"].GetInt() != 1 ||
            !item.HasMember("sampled_workload_dispatches") ||
            !item["sampled_workload_dispatches"].IsInt() ||
            item["sampled_workload_dispatches"].GetInt() != 3) {
            return 7;
        }
    }
#endif
    return 0;
}
