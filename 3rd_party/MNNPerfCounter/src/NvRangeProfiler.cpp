// NvRangeProfiler.cpp - NVIDIA CUPTI Range Profiler backend for MNNPerfCounter.
//
// Provides per-kernel precise hardware counter collection on NVIDIA GPUs via
// the CUPTI Range Profiler API (cupti_range_profiler.h) backed by NV Perf Host
// API (nvperf_host.h / nvperf_cuda_host.h). Does not depend on any CUPTI
// sample helper code — all NVPA calls are issued directly.
//
// Flow:
//   1. cuInit + cuptiProfilerInitialize + cuptiDeviceGetChipName
//   2. Build config image from metric names via NVPA RawMetricsConfig
//   3. Build counterDataPrefix via NVPA CounterDataBuilder
//   4. Allocate counterDataImage + scratch via cuptiProfilerCounterDataImage*
//   5. start()  -> cuptiRangeProfilerEnable + SetConfig + Start
//   6. AutoRange + KernelReplay lets CUPTI identify and replay the target kernel
//   7. stop()   -> Stop + Disable + DecodeData
//   8. Evaluate metrics via NVPW_MetricsEvaluator_EvaluateToGpuValues per range

#include "MNNPerfCounter.hpp"
#include "NvRangeProfilerInternal.hpp"

#include <cuda.h>
#include <cupti.h>
#include <cupti_profiler_target.h>
#include <cupti_range_profiler.h>
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace MNN {
namespace PerfCounter {

namespace {

const char* nvpaStatusString(NVPA_Status s) {
    switch (s) {
        case NVPA_STATUS_SUCCESS: return "NVPA_STATUS_SUCCESS";
        case NVPA_STATUS_INVALID_ARGUMENT: return "NVPA_STATUS_INVALID_ARGUMENT";
        case NVPA_STATUS_INVALID_METRIC_ID: return "NVPA_STATUS_INVALID_METRIC_ID";
        case NVPA_STATUS_UNSUPPORTED_GPU: return "NVPA_STATUS_UNSUPPORTED_GPU";
        case NVPA_STATUS_NOT_INITIALIZED: return "NVPA_STATUS_NOT_INITIALIZED";
        case NVPA_STATUS_NOT_LOADED: return "NVPA_STATUS_NOT_LOADED";
        case NVPA_STATUS_NOT_SUPPORTED: return "NVPA_STATUS_NOT_SUPPORTED";
        case NVPA_STATUS_OUT_OF_MEMORY: return "NVPA_STATUS_OUT_OF_MEMORY";
        default: return "NVPA_STATUS_UNKNOWN";
    }
}

#define CUPTI_CHECK(expr, errout) do { \
    CUptiResult _r = (expr); \
    if (_r != CUPTI_SUCCESS) { \
        const char* _s = nullptr; \
        cuptiGetResultString(_r, &_s); \
        if (errout) *(errout) = std::string("CUPTI error: ") + (_s ? _s : "unknown"); \
        return false; \
    } \
} while (0)

#define NVPA_CHECK(expr, errout) do { \
    NVPA_Status _r = (expr); \
    if (_r != NVPA_STATUS_SUCCESS) { \
        if (errout) *(errout) = std::string("NVPA error: ") + nvpaStatusString(_r); \
        return false; \
    } \
} while (0)

// Parse a metric name like "sm__cycles_elapsed.avg" into base name + submetric
// flags. The NVPA API accepts the dotted form directly; we just pass it through.
// keepInstances/isolated default to true (per kernel).
void parseMetricName(const std::string& full, std::string* base, bool* isolated, bool* keepInstances) {
    *base = full;
    *isolated = true;
    *keepInstances = true;
}

// Step 2: generate config image from metric names.
} // namespace

bool nvBuildConfigImage(const std::string& chipName,
                       const std::vector<std::string>& metricNames,
                       std::vector<uint8_t>* configImage,
                       size_t* numPassesOut,
                       std::string* error) {
    // Create metrics evaluator to resolve raw dependencies.
    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params sizeParams = {
        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    sizeParams.pChipName = chipName.c_str();
    NVPA_CHECK(NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&sizeParams), error);

    std::vector<uint8_t> scratch(sizeParams.scratchBufferSize);
    NVPW_CUDA_MetricsEvaluator_Initialize_Params initParams = {
        NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
    initParams.scratchBufferSize = scratch.size();
    initParams.pScratchBuffer = scratch.data();
    initParams.pChipName = chipName.c_str();
    NVPA_CHECK(NVPW_CUDA_MetricsEvaluator_Initialize(&initParams), error);
    NVPW_MetricsEvaluator* evaluator = initParams.pMetricsEvaluator;

    // Collect raw metric dependencies for all requested metrics.
    std::vector<const char*> rawNames;
    bool isolated = true, keepInstances = true;
    for (const auto& m : metricNames) {
        std::string base;
        parseMetricName(m, &base, &isolated, &keepInstances);
        NVPW_MetricEvalRequest req;
        NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params conv = {
            NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
        conv.pMetricsEvaluator = evaluator;
        conv.pMetricName = base.c_str();
        conv.pMetricEvalRequest = &req;
        conv.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
        NVPA_CHECK(NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&conv), error);

        NVPW_MetricsEvaluator_GetMetricRawDependencies_Params deps = {
            NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE};
        deps.pMetricsEvaluator = evaluator;
        deps.pMetricEvalRequests = &req;
        deps.numMetricEvalRequests = 1;
        deps.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
        deps.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
        NVPA_CHECK(NVPW_MetricsEvaluator_GetMetricRawDependencies(&deps), error);
        std::vector<const char*> depPtrs(deps.numRawDependencies);
        deps.ppRawDependencies = depPtrs.data();
        NVPA_CHECK(NVPW_MetricsEvaluator_GetMetricRawDependencies(&deps), error);
        for (auto p : depPtrs) rawNames.push_back(p);
    }

    // Destroy evaluator now that raw deps are resolved.
    NVPW_MetricsEvaluator_Destroy_Params destroyEval = {
        NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE};
    destroyEval.pMetricsEvaluator = evaluator;
    NVPW_MetricsEvaluator_Destroy(&destroyEval);

    // Build NVPA_RawMetricRequest list (dedup not strictly required).
    std::vector<NVPA_RawMetricRequest> reqs;
    for (auto* name : rawNames) {
        NVPA_RawMetricRequest r = {NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE};
        r.pMetricName = name;
        r.isolated = isolated;
        r.keepInstances = keepInstances;
        reqs.push_back(r);
    }

    // Create CUDA raw metrics config and add metrics.
    NVPW_CUDA_RawMetricsConfig_Create_V2_Params createParams = {
        NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE};
    createParams.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
    createParams.pChipName = chipName.c_str();
    NVPA_CHECK(NVPW_CUDA_RawMetricsConfig_Create_V2(&createParams), error);
    NVPA_RawMetricsConfig* cfg = createParams.pRawMetricsConfig;

    NVPW_RawMetricsConfig_BeginPassGroup_Params beginPG = {
        NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE};
    beginPG.pRawMetricsConfig = cfg;
    NVPA_CHECK(NVPW_RawMetricsConfig_BeginPassGroup(&beginPG), error);

    NVPW_RawMetricsConfig_AddMetrics_Params addParams = {
        NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE};
    addParams.pRawMetricsConfig = cfg;
    addParams.pRawMetricRequests = reqs.data();
    addParams.numMetricRequests = reqs.size();
    NVPA_CHECK(NVPW_RawMetricsConfig_AddMetrics(&addParams), error);

    NVPW_RawMetricsConfig_EndPassGroup_Params endPG = {
        NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE};
    endPG.pRawMetricsConfig = cfg;
    NVPA_CHECK(NVPW_RawMetricsConfig_EndPassGroup(&endPG), error);

    NVPW_RawMetricsConfig_GenerateConfigImage_Params gen = {
        NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE};
    gen.pRawMetricsConfig = cfg;
    NVPA_CHECK(NVPW_RawMetricsConfig_GenerateConfigImage(&gen), error);

    // Query actual number of passes required by this metric configuration.
    // This accounts for raw-counter dependency conflicts — metrics sharing
    // the same hardware counter slot require multiple passes (kernel replays).
    // numIsolatedPasses is the key value: it reflects actual replay passes.
    if (numPassesOut) {
        NVPW_RawMetricsConfig_GetNumPasses_Params passParams = {
            NVPW_RawMetricsConfig_GetNumPasses_Params_STRUCT_SIZE};
        passParams.pRawMetricsConfig = cfg;
        if (NVPW_RawMetricsConfig_GetNumPasses(&passParams) == NVPA_STATUS_SUCCESS) {
            *numPassesOut = passParams.numPipelinedPasses + passParams.numIsolatedPasses;
        }
    }

    NVPW_RawMetricsConfig_GetConfigImage_Params get = {
        NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE};
    get.pRawMetricsConfig = cfg;
    get.bytesAllocated = 0;
    get.pBuffer = nullptr;
    NVPA_CHECK(NVPW_RawMetricsConfig_GetConfigImage(&get), error);
    configImage->resize(get.bytesCopied);
    get.bytesAllocated = configImage->size();
    get.pBuffer = configImage->data();
    NVPA_CHECK(NVPW_RawMetricsConfig_GetConfigImage(&get), error);

    NVPW_RawMetricsConfig_Destroy_Params destroyCfg = {
        NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE};
    destroyCfg.pRawMetricsConfig = cfg;
    NVPW_RawMetricsConfig_Destroy(&destroyCfg);
    return true;
}

// Step 3: generate counterDataPrefix from metric names.
bool nvBuildCounterDataPrefix(const std::string& chipName,
                             const std::vector<std::string>& metricNames,
                             std::vector<uint8_t>* prefix,
                             std::string* error) {
    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params sizeParams = {
        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    sizeParams.pChipName = chipName.c_str();
    NVPA_CHECK(NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&sizeParams), error);

    std::vector<uint8_t> scratch(sizeParams.scratchBufferSize);
    NVPW_CUDA_MetricsEvaluator_Initialize_Params initParams = {
        NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
    initParams.scratchBufferSize = scratch.size();
    initParams.pScratchBuffer = scratch.data();
    initParams.pChipName = chipName.c_str();
    NVPA_CHECK(NVPW_CUDA_MetricsEvaluator_Initialize(&initParams), error);
    NVPW_MetricsEvaluator* evaluator = initParams.pMetricsEvaluator;

    std::vector<NVPA_RawMetricRequest> reqs;
    bool isolated = true, keepInstances = true;
    for (const auto& m : metricNames) {
        std::string base;
        parseMetricName(m, &base, &isolated, &keepInstances);
        NVPW_MetricEvalRequest mer;
        NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params conv = {
            NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
        conv.pMetricsEvaluator = evaluator;
        conv.pMetricName = base.c_str();
        conv.pMetricEvalRequest = &mer;
        conv.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
        NVPA_CHECK(NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&conv), error);

        NVPW_MetricsEvaluator_GetMetricRawDependencies_Params deps = {
            NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE};
        deps.pMetricsEvaluator = evaluator;
        deps.pMetricEvalRequests = &mer;
        deps.numMetricEvalRequests = 1;
        deps.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
        deps.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
        NVPA_CHECK(NVPW_MetricsEvaluator_GetMetricRawDependencies(&deps), error);
        std::vector<const char*> depPtrs(deps.numRawDependencies);
        deps.ppRawDependencies = depPtrs.data();
        NVPA_CHECK(NVPW_MetricsEvaluator_GetMetricRawDependencies(&deps), error);
        for (auto p : depPtrs) {
            NVPA_RawMetricRequest r = {NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE};
            r.pMetricName = p;
            r.isolated = isolated;
            r.keepInstances = keepInstances;
            reqs.push_back(r);
        }
    }

    NVPW_MetricsEvaluator_Destroy_Params d = {NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE};
    d.pMetricsEvaluator = evaluator;
    NVPW_MetricsEvaluator_Destroy(&d);

    // CounterDataBuilder.
    NVPW_CounterDataBuilder_Create_Params cbCreate = {NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE};
    cbCreate.pChipName = chipName.c_str();
    NVPA_CHECK(NVPW_CounterDataBuilder_Create(&cbCreate), error);
    NVPA_CounterDataBuilder* builder = cbCreate.pCounterDataBuilder;

    NVPW_CounterDataBuilder_AddMetrics_Params addParams = {NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE};
    addParams.pCounterDataBuilder = builder;
    addParams.pRawMetricRequests = reqs.data();
    addParams.numMetricRequests = reqs.size();
    NVPA_CHECK(NVPW_CounterDataBuilder_AddMetrics(&addParams), error);

    NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getPrefix = {
        NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE};
    getPrefix.pCounterDataBuilder = builder;
    getPrefix.bytesAllocated = 0;
    getPrefix.pBuffer = nullptr;
    NVPA_CHECK(NVPW_CounterDataBuilder_GetCounterDataPrefix(&getPrefix), error);
    prefix->resize(getPrefix.bytesCopied);
    getPrefix.bytesAllocated = prefix->size();
    getPrefix.pBuffer = prefix->data();
    NVPA_CHECK(NVPW_CounterDataBuilder_GetCounterDataPrefix(&getPrefix), error);

    NVPW_CounterDataBuilder_Destroy_Params cbDestroy = {NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE};
    cbDestroy.pCounterDataBuilder = builder;
    NVPW_CounterDataBuilder_Destroy(&cbDestroy);
    return true;
}

GpuFamily nvidiaFamilyFromComputeCapability(int major, int minor) {
    // sm_75 = Turing, sm_80/86 = Ampere, sm_89 = Ada, sm_90 = Hopper
    if (major == 7 && minor == 5) return GpuFamily::NvidiaTuring;
    if (major == 8) return GpuFamily::NvidiaAmpere;
    if (major == 8 && minor == 9) return GpuFamily::NvidiaAda;
    if (major == 9) return GpuFamily::NvidiaHopper;
    return GpuFamily::NvidiaOther;
}

bool identifyNvidia(int deviceOrdinal, DeviceInfo* device) {
    if (device == nullptr) return false;
    if (cuInit(0) != CUDA_SUCCESS) return false;
    int count = 0;
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS || deviceOrdinal >= count) return false;
    CUdevice dev;
    if (cuDeviceGet(&dev, deviceOrdinal) != CUDA_SUCCESS) return false;

    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    char name[256] = {0};
    cuDeviceGetName(name, sizeof(name), dev);

    device->vendor = GpuVendor::Nvidia;
    device->family = nvidiaFamilyFromComputeCapability(major, minor);
    device->productId = (uint64_t(major) << 8) | uint64_t(minor);
    device->productName = nullptr; // name is stack-allocated; caller must copy
    device->driverName = "CUPTI Range Profiler";
    // Stash the device name into a static so productName stays valid.
    static thread_local std::string nameStore;
    nameStore = name;
    device->productName = nameStore.c_str();
    return true;
}

bool nvEvaluateMetrics(const std::string& chipName,
                       const std::vector<uint8_t>& counterDataImage,
                       const std::vector<std::string>& metricNames,
                       CounterValue* values, size_t count,
                       std::string* error) {
    if (counterDataImage.empty()) {
        if (error) *error = "counterDataImage is empty";
        return false;
    }
    std::fprintf(stderr, "[nvEval] counterDataImage=%zu bytes, %zu metrics, count=%zu\n",
                 counterDataImage.size(), metricNames.size(), count);
    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params sizeParams = {
        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    sizeParams.pChipName = chipName.c_str();
    NVPA_CHECK(NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&sizeParams), error);
    std::vector<uint8_t> scratch(sizeParams.scratchBufferSize);
    NVPW_CUDA_MetricsEvaluator_Initialize_Params initParams = {
        NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
    initParams.scratchBufferSize = scratch.size();
    initParams.pScratchBuffer = scratch.data();
    initParams.pChipName = chipName.c_str();
    NVPA_CHECK(NVPW_CUDA_MetricsEvaluator_Initialize(&initParams), error);
    NVPW_MetricsEvaluator* evaluator = initParams.pMetricsEvaluator;

    NVPW_CounterData_GetNumRanges_Params nr = {NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE};
    nr.pCounterDataImage = counterDataImage.data();
    NVPA_CHECK(NVPW_CounterData_GetNumRanges(&nr), error);

    bool anyOk = true;
    for (size_t i = 0; i < metricNames.size() && i < count; ++i) {
        std::string base;
        bool isolated = true, keepInstances = true;
        parseMetricName(metricNames[i], &base, &isolated, &keepInstances);
        NVPW_MetricEvalRequest mer;
        NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params conv = {
            NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
        conv.pMetricsEvaluator = evaluator;
        conv.pMetricName = base.c_str();
        conv.pMetricEvalRequest = &mer;
        conv.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
        if (NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&conv) != NVPA_STATUS_SUCCESS) {
            anyOk = false;
            continue;
        }
        double summed = 0.0;
        for (size_t r = 0; r < nr.numRanges; ++r) {
            NVPW_MetricsEvaluator_SetDeviceAttributes_Params sda = {
                NVPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE};
            sda.pMetricsEvaluator = evaluator;
            sda.pCounterDataImage = counterDataImage.data();
            sda.counterDataImageSize = counterDataImage.size();
            if (NVPW_MetricsEvaluator_SetDeviceAttributes(&sda) != NVPA_STATUS_SUCCESS) {
                anyOk = false; continue;
            }
            double v = 0.0;
            NVPW_MetricsEvaluator_EvaluateToGpuValues_Params ev = {
                NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE};
            ev.pMetricsEvaluator = evaluator;
            ev.pMetricEvalRequests = &mer;
            ev.numMetricEvalRequests = 1;
            ev.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
            ev.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
            ev.pCounterDataImage = counterDataImage.data();
            ev.counterDataImageSize = counterDataImage.size();
            ev.rangeIndex = r;
            ev.isolated = true;
            ev.pMetricValues = &v;
            if (NVPW_MetricsEvaluator_EvaluateToGpuValues(&ev) != NVPA_STATUS_SUCCESS) {
                anyOk = false; continue;
            }
            summed += v;
        }
        values[i].name = metricNames[i].c_str();
        values[i].value = static_cast<uint64_t>(summed);
    }
    NVPW_MetricsEvaluator_Destroy_Params d = {NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE};
    d.pMetricsEvaluator = evaluator;
    NVPW_MetricsEvaluator_Destroy(&d);
    return anyOk;
}

// Range push/pop wrappers operating on opaque state (called by Session::Impl
// in MNNPerfCounter.cpp). rangeObj is CUpti_RangeProfiler_Object*.
bool nvBeginRange(void* rangeObj, const char* rangeName) {
    CUpti_RangeProfiler_PushRange_Params p = {CUpti_RangeProfiler_PushRange_Params_STRUCT_SIZE};
    p.pRangeProfilerObject = static_cast<CUpti_RangeProfiler_Object*>(rangeObj);
    p.pRangeName = rangeName;
    return cuptiRangeProfilerPushRange(&p) == CUPTI_SUCCESS;
}

bool nvEndRange(void* rangeObj) {
    CUpti_RangeProfiler_PopRange_Params p = {CUpti_RangeProfiler_PopRange_Params_STRUCT_SIZE};
    p.pRangeProfilerObject = static_cast<CUpti_RangeProfiler_Object*>(rangeObj);
    return cuptiRangeProfilerPopRange(&p) == CUPTI_SUCCESS;
}

} // namespace PerfCounter
} // namespace MNN
