// list_cuda_metrics.cu - Enumerate all NVIDIA GPU metrics supported by the
// local GPU via the NV Perf Host API. Prints one metric name per line to
// stdout (status info to stderr), suitable for piping into --perf-counter-events.
//
// Build:
//   nvcc -std=c++17 -arch=sm_75 list_cuda_metrics.cu -lcupti -lcuda -o list_cuda_metrics
//
// Usage:
//   ./list_cuda_metrics                    # all metrics (with submetrics)
//   ./list_cuda_metrics --no-submetrics    # base metrics only
//   ./list_cuda_metrics --chip <name>      # for a specific chip name

#include <cuda.h>
#include <cupti.h>
#include <cupti_target.h>
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const char* rollupOpString(NVPW_RollupOp op) {
    switch (op) {
        case NVPW_ROLLUP_OP_AVG: return ".avg";
        case NVPW_ROLLUP_OP_MAX: return ".max";
        case NVPW_ROLLUP_OP_MIN: return ".min";
        case NVPW_ROLLUP_OP_SUM: return ".sum";
        default: return "";
    }
}

static const char* submetricString(NVPW_Submetric s) {
    switch (s) {
        case NVPW_SUBMETRIC_PEAK_SUSTAINED: return ".peak_sustained";
        case NVPW_SUBMETRIC_PEAK_SUSTAINED_ACTIVE: return ".peak_sustained_active";
        case NVPW_SUBMETRIC_PEAK_SUSTAINED_ACTIVE_PER_SECOND: return ".peak_sustained_active.per_second";
        case NVPW_SUBMETRIC_PEAK_SUSTAINED_ELAPSED: return ".peak_sustained_elapsed";
        case NVPW_SUBMETRIC_PEAK_SUSTAINED_ELAPSED_PER_SECOND: return ".peak_sustained_elapsed.per_second";
        case NVPW_SUBMETRIC_PEAK_SUSTAINED_FRAME: return ".peak_sustained_frame";
        case NVPW_SUBMETRIC_PEAK_SUSTAINED_FRAME_PER_SECOND: return ".peak_sustained_frame.per_second";
        case NVPW_SUBMETRIC_PEAK_SUSTAINED_REGION: return ".peak_sustained_region";
        case NVPW_SUBMETRIC_PEAK_SUSTAINED_REGION_PER_SECOND: return ".peak_sustained_region.per_second";
        case NVPW_SUBMETRIC_PER_CYCLE_ACTIVE: return ".per_cycle_active";
        case NVPW_SUBMETRIC_PER_CYCLE_ELAPSED: return ".per_cycle_elapsed";
        case NVPW_SUBMETRIC_PER_CYCLE_IN_FRAME: return ".per_cycle_in_frame";
        case NVPW_SUBMETRIC_PER_CYCLE_IN_REGION: return ".per_cycle_in_region";
        case NVPW_SUBMETRIC_PER_SECOND: return ".per_second";
        case NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_ACTIVE: return ".pct_of_peak_sustained_active";
        case NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_ELAPSED: return ".pct_of_peak_sustained_elapsed";
        case NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_FRAME: return ".pct_of_peak_sustained_frame";
        case NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_REGION: return ".pct_of_peak_sustained_region";
        case NVPW_SUBMETRIC_MAX_RATE: return ".max_rate";
        case NVPW_SUBMETRIC_PCT: return ".pct";
        case NVPW_SUBMETRIC_RATIO: return ".ratio";
        case NVPW_SUBMETRIC_NONE:
        default: return "";
    }
}

static std::string getChipName(int deviceOrdinal) {
    CUpti_Device_GetChipName_Params p = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    p.deviceIndex = deviceOrdinal;
    if (cuptiDeviceGetChipName(&p) != CUPTI_SUCCESS) return "";
    return p.pChipName ? p.pChipName : "";
}

int main(int argc, char** argv) {
    bool listSubMetrics = true;
    std::string chipOverride;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-submetrics") == 0) listSubMetrics = false;
        else if (std::strcmp(argv[i], "--chip") == 0 && i + 1 < argc) chipOverride = argv[++i];
        else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf("Usage: %s [--no-submetrics] [--chip <name>]\n", argv[0]);
            return 0;
        }
    }

    if (cuInit(0) != CUDA_SUCCESS) { std::fprintf(stderr, "cuInit failed\n"); return 1; }
    int count = 0;
    cuDeviceGetCount(&count);
    if (count == 0) { std::fprintf(stderr, "no CUDA device\n"); return 1; }

    std::string chip = chipOverride.empty() ? getChipName(0) : chipOverride;
    if (chip.empty()) { std::fprintf(stderr, "could not get chip name\n"); return 1; }
    std::fprintf(stderr, "# chip: %s\n", chip.c_str());

    NVPW_InitializeHost_Params init = {NVPW_InitializeHost_Params_STRUCT_SIZE};
    if (NVPW_InitializeHost(&init) != NVPA_STATUS_SUCCESS) {
        std::fprintf(stderr, "NVPW_InitializeHost failed\n"); return 1;
    }

    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params sz = {
        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    sz.pChipName = chip.c_str();
    if (NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&sz) != NVPA_STATUS_SUCCESS) {
        std::fprintf(stderr, "CalculateScratchBufferSize failed\n"); return 1;
    }
    std::vector<uint8_t> scratch(sz.scratchBufferSize);
    NVPW_CUDA_MetricsEvaluator_Initialize_Params ei = {
        NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
    ei.scratchBufferSize = scratch.size();
    ei.pScratchBuffer = scratch.data();
    ei.pChipName = chip.c_str();
    if (NVPW_CUDA_MetricsEvaluator_Initialize(&ei) != NVPA_STATUS_SUCCESS) {
        std::fprintf(stderr, "MetricsEvaluator_Initialize failed\n"); return 1;
    }
    NVPW_MetricsEvaluator* ev = ei.pMetricsEvaluator;

    int total = 0;
    for (int mt = 0; mt < NVPW_METRIC_TYPE__COUNT; ++mt) {
        NVPW_MetricsEvaluator_GetMetricNames_Params gmn = {
            NVPW_MetricsEvaluator_GetMetricNames_Params_STRUCT_SIZE};
        gmn.metricType = static_cast<NVPW_MetricType>(mt);
        gmn.pMetricsEvaluator = ev;
        if (NVPW_MetricsEvaluator_GetMetricNames(&gmn) != NVPA_STATUS_SUCCESS) continue;

        for (size_t i = 0; i < gmn.numMetrics; ++i) {
            size_t beginIdx = gmn.pMetricNameBeginIndices[i];
            const char* base = &gmn.pMetricNames[beginIdx];
            for (int ro = 0; ro < NVPW_ROLLUP_OP__COUNT; ++ro) {
                std::string name = base;
                if (mt != NVPW_METRIC_TYPE_RATIO) name += rollupOpString(static_cast<NVPW_RollupOp>(ro));
                if (listSubMetrics) {
                    NVPW_MetricsEvaluator_GetSupportedSubmetrics_Params gs = {
                        NVPW_MetricsEvaluator_GetSupportedSubmetrics_Params_STRUCT_SIZE};
                    gs.metricType = static_cast<NVPW_MetricType>(mt);
                    gs.pMetricsEvaluator = ev;
                    if (NVPW_MetricsEvaluator_GetSupportedSubmetrics(&gs) == NVPA_STATUS_SUCCESS) {
                        for (size_t sm = 0; sm < gs.numSupportedSubmetrics; ++sm) {
                            std::string sub = name + submetricString(
                                static_cast<NVPW_Submetric>(gs.pSupportedSubmetrics[sm]));
                            std::printf("%s\n", sub.c_str());
                            ++total;
                        }
                    } else {
                        std::printf("%s\n", name.c_str()); ++total;
                    }
                } else {
                    std::printf("%s\n", name.c_str()); ++total;
                }
            }
        }
    }
    NVPW_MetricsEvaluator_Destroy_Params d = {NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE};
    d.pMetricsEvaluator = ev;
    NVPW_MetricsEvaluator_Destroy(&d);
    std::fprintf(stderr, "# %d metrics\n", total);
    return 0;
}
