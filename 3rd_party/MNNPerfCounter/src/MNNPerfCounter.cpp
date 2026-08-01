#include "MNNPerfCounter.hpp"
#include "A7xxPerfCounters.hpp"

#if defined(MNN_PERFCOUNTER_HAS_CUDA)
#include "NvRangeProfilerInternal.hpp"
#include <cuda.h>
#include <cupti.h>
#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <cupti_range_profiler.h>
#endif

#include "driver_ioctl.h"
#include "hwcpipe/counter_database.hpp"
#include "hwcpipe/gpu.hpp"
#include "hwcpipe/sampler.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace MNN {
namespace PerfCounter {

namespace {

struct Event {
    const char* name;
    uint32_t group;
    uint32_t selector;
    uint32_t slots;
};

// KGSL's public group ids are part of the userspace ioctl ABI. They are
// intentionally kept as an explicit table instead of being inferred from the
// order of Mesa's JSON groups.
enum AdrenoGroup {
    kKgslCp = 0,
    kKgslRbbm = 1,
    kKgslPc = 2,
    kKgslVfd = 3,
    kKgslHlsq = 4,
    kKgslVpc = 5,
    kKgslTse = 6,
    kKgslRas = 7,
    kKgslUche = 8,
    kKgslTp = 9,
    kKgslSp = 10,
    kKgslRb = 11,
};

static const Event kA7xxEvents[] = {
    {"rbbm_busy_cycles", kKgslRbbm, 0, 4},
    {"hlsq_busy_cycles", kKgslHlsq, 0, 6},
    {"hlsq_stall_cycles_uche", kKgslHlsq, 1, 6},
    {"uche_busy_cycles", kKgslUche, 0, 12},
    {"uche_vbif_latency_cycles", kKgslUche, 2, 12},
    {"uche_vbif_latency_samples", kKgslUche, 3, 12},
    {"uche_vbif_read_beats_sp", kKgslUche, 8, 12},
    {"uche_read_requests_sp", kKgslUche, 13, 12},
    {"tp_busy_cycles", kKgslTp, 0, 12},
    {"sp_busy_cycles", kKgslSp, 0, 24},
    {"sp_alu_working_cycles", kKgslSp, 1, 24},
    {"sp_stall_cycles_uche", kKgslSp, 5, 24},
    {"sp_non_execution_cycles", kKgslSp, 7, 24},
    {"sp_lm_load_instructions", kKgslSp, 27, 24},
    {"sp_lm_store_instructions", kKgslSp, 28, 24},
    {"sp_gm_load_instructions", kKgslSp, 30, 24},
    {"sp_gm_store_instructions", kKgslSp, 31, 24},
    {"sp_gm_atomics", kKgslSp, 32, 24},
    {"sp_cs_instructions", kKgslSp, 56, 24},
    {"sp_lm_bank_conflicts", kKgslSp, 61, 24},
    {"sp_working_eu_cs_stage", kKgslSp, 77, 24},
    {"sp_any_eu_working_cs_stage", kKgslSp, 78, 24},
    {"sp_gm_load_latency_cycles", kKgslSp, 82, 24},
    {"sp_gm_load_latency_samples", kKgslSp, 83, 24},
    {"sp_executable_waves", kKgslSp, 84, 24},
    {"sp_cs_invocations", kKgslSp, 100, 24},
    {"gpu_active_cycles", kKgslSp, 0, 24},
    {"compute_active_cycles", kKgslSp, 0, 24},
    {"compute_tasks", kKgslSp, 100, 24},
    {"l2_any_lookup", kKgslUche, 13, 12},
    {"l2_ext_read", kKgslUche, 8, 12},
    {"l2_ext_write", kKgslUche, 9, 12},
};

static const Event kAdrenoCommonEvents[] = {
    {"rbbm_busy_cycles", 1, 9, 4},
    {"hlsq_busy_cycles", 4, 0, 6},
    {"uche_busy_cycles", 8, 0, 12},
    {"uche_vbif_latency_cycles", 8, 2, 12},
    {"uche_vbif_latency_samples", 8, 3, 12},
    {"uche_vbif_read_beats_sp", 8, 8, 12},
    {"uche_read_requests_sp", 8, 13, 12},
    {"tp_busy_cycles", 9, 0, 12},
    {"sp_busy_cycles", 10, 0, 24},
    {"sp_stall_cycles_uche", 10, 5, 24},
    {"sp_gm_load_instructions", 10, 30, 24},
    {"sp_gm_store_instructions", 10, 31, 24},
    {"sp_vs_instructions", 10, 91, 24},
    {"sp_fs_instructions", 10, 92, 24},
    {"sp_cs_instructions", 10, 104, 24},
    {"gpu_active_cycles", 10, 0, 24},
    {"compute_active_cycles", 10, 0, 24},
    {"compute_tasks", 10, 104, 24},
    {"l2_any_lookup", 8, 13, 12},
    {"l2_ext_read", 8, 8, 12},
    {"l2_ext_write", 8, 9, 12},
};

struct MaliAlias {
    const char* name;
    hwcpipe_counter counter;
};

static const MaliAlias kMaliAliases[] = {
    {"gpu_active_cycles", MaliGPUActiveCy},
    {"compute_active_cycles", MaliNonFragActiveCy},
    {"compute_tasks", MaliNonFragTask},
    {"l2_any_lookup", MaliL2CacheLookup},
    {"l2_ext_read", MaliSCBusLSL2RdBt},
    {"l2_ext_write", MaliSCBusLSWrBt},
};

static const char* kUnavailable = "GPU performance counter backend unavailable";

static const Event* findEvent(const Event* events, size_t count, const char* name) {
    if (name == nullptr) {
        return nullptr;
    }
    for (size_t i = 0; i < count; ++i) {
        if (std::strcmp(events[i].name, name) == 0) {
            return events + i;
        }
    }
    return nullptr;
}

static bool isAdrenoA7xx(uint64_t productId) {
    return productId >= 700 && productId < 800;
}

static bool rawMaliProduct(uint64_t raw, hwcpipe::device::product_id* product) {
    const auto result = hwcpipe::device::product_id_from_raw_gpu_id(raw);
    if (result.first) {
        return false;
    }
    *product = result.second;
    return true;
}

static const char* maliProductName(hwcpipe::device::product_id product) {
    using P = hwcpipe::device::product_id;
    switch (product) {
        case P::t60x: return "Mali-T60x";
        case P::t62x: return "Mali-T62x";
        case P::t720: return "Mali-T720";
        case P::t760: return "Mali-T760";
        case P::t820: return "Mali-T820";
        case P::t830: return "Mali-T830";
        case P::t860: return "Mali-T860";
        case P::t880: return "Mali-T880";
        case P::g31: return "Mali-G31";
        case P::g51: return "Mali-G51";
        case P::g52: return "Mali-G52";
        case P::g71: return "Mali-G71";
        case P::g72: return "Mali-G72";
        case P::g76: return "Mali-G76";
        case P::g57: return "Mali-G57";
        case P::g57_2: return "Mali-G57.2";
        case P::g68: return "Mali-G68";
        case P::g77: return "Mali-G77";
        case P::g78: return "Mali-G78";
        case P::g78ae: return "Mali-G78AE";
        case P::g310: return "Mali-G310";
        case P::g510: return "Mali-G510";
        case P::g610: return "Mali-G610";
        case P::g615: return "Mali-G615";
        case P::g710: return "Mali-G710";
        case P::g715: return "Mali-G715";
        case P::g720: return "Mali-G720";
        case P::g620: return "Mali-G620";
        case P::g725: return "Mali-G725";
        case P::g625: return "Mali-G625";
        case P::g1_ultra: return "Mali-G1 Ultra";
        case P::g1_premium: return "Mali-G1 Premium";
        case P::g1_pro: return "Mali-G1 Pro";
    }
    return nullptr;
}

static GpuFamily maliFamily(hwcpipe::device::product_id product) {
    switch (hwcpipe::device::get_gpu_family(product)) {
        case hwcpipe::device::gpu_family::midgard: return GpuFamily::Midgard;
        case hwcpipe::device::gpu_family::bifrost: return GpuFamily::Bifrost;
        case hwcpipe::device::gpu_family::valhall: return GpuFamily::Valhall;
        case hwcpipe::device::gpu_family::fifthgen: return GpuFamily::Arm5thGen;
    }
    return GpuFamily::Unknown;
}

static bool findMaliCounter(uint64_t rawProductId, const char* name, hwcpipe_counter* counter) {
    hwcpipe::device::product_id product;
    if (!rawMaliProduct(rawProductId, &product) || name == nullptr) {
        return false;
    }

    hwcpipe::sampler_config config(product, 0);
    for (const auto& alias : kMaliAliases) {
        if (std::strcmp(alias.name, name) == 0) {
            if (config.add_counter(alias.counter)) {
                return false;
            }
            *counter = alias.counter;
            return true;
        }
    }

    // Also accept every descriptive counter in Arm's database. This keeps the
    // wrapper complete as the upstream database grows without exposing its API.
    hwcpipe::counter_database database;
    hwcpipe::counter_metadata metadata;
    for (uint32_t i = 0; i < 1024; ++i) {
        const auto candidate = static_cast<hwcpipe_counter>(i);
        if (database.describe_counter(candidate, metadata) || metadata.name == nullptr ||
            std::strcmp(metadata.name, name) != 0) {
            continue;
        }
        if (config.add_counter(candidate)) {
            return false;
        }
        *counter = candidate;
        return true;
    }
    return false;
}

static void setError(std::string* error, const char* message) {
    if (error != nullptr) {
        *error = message == nullptr ? kUnavailable : message;
    }
}

static void appendUnique(std::vector<std::string>* names, const char* name) {
    if (names == nullptr || name == nullptr || *name == '\0') {
        return;
    }
    if (std::find(names->begin(), names->end(), name) == names->end()) {
        names->emplace_back(name);
    }
}

} // namespace

struct Session::Impl {
    struct AdrenoCounter {
        CounterSpec spec;
        CounterBinding binding;
        hpc_gpu_adreno_ioctl_counter_read_counter_t raw;
        uint64_t previous = 0;
    };

    GpuVendor vendor = GpuVendor::Unknown;
    DeviceInfo device;
    std::string error;
    int adrenoFd = -1;
    std::vector<AdrenoCounter> adrenoCounters;
    std::unique_ptr<hwcpipe::sampler<>> maliSampler;
    std::vector<hwcpipe_counter> maliCounters;
    std::vector<CounterSpec> maliSpecs;
    // NVIDIA Range Profiler state (only used when vendor == Nvidia).
    std::string nvChipName;
    std::vector<std::string> nvMetricNames;
    std::vector<uint8_t> nvConfigImage;
    std::vector<uint8_t> nvCounterDataImage;
    std::vector<uint8_t> nvCounterDataScratch;
    void* nvRangeObj = nullptr;  // CUpti_RangeProfiler_Object*
    void* nvCtx = nullptr;       // CUcontext
    int nvRangeDepth = 0;
    size_t nvNumPasses = 1;      // actual passes needed (from GetNumPasses)
    bool nvAutoRange = false;    // CUPTI owns range selection and kernel replay
    bool started = false;
};

bool identifyMali(uint64_t rawProductId, DeviceInfo* device) {
    if (device == nullptr) {
        return false;
    }
    hwcpipe::device::product_id product;
    if (!rawMaliProduct(rawProductId, &product)) {
        return false;
    }
    device->vendor = GpuVendor::Mali;
    device->family = maliFamily(product);
    device->productId = rawProductId;
    device->productName = maliProductName(product);
    device->driverName = "Arm Mali HWCpipe (vinstr/kinstr_prfcnt)";
    return device->productName != nullptr;
}

bool resolveCounter(GpuVendor vendor, uint64_t productId, const char* name, CounterBinding* binding) {
    if (binding == nullptr || name == nullptr) {
        return false;
    }
    *binding = CounterBinding();
    if (vendor == GpuVendor::Adreno) {
        const Event* event = nullptr;
        if (isAdrenoA7xx(productId)) {
            size_t count = 0;
            const auto* events = detail::a7xxEvents(&count);
            for (size_t i = 0; i < count; ++i) {
                if (std::strcmp(events[i].name, name) == 0) {
                    binding->name = events[i].name;
                    binding->group = events[i].group;
                    binding->selector = events[i].selector;
                    binding->slotsAvailable = events[i].slots;
                    return true;
                }
            }
            event = findEvent(kA7xxEvents, sizeof(kA7xxEvents) / sizeof(kA7xxEvents[0]), name);
        } else if (productId >= 500 && productId < 700) {
            event = findEvent(kAdrenoCommonEvents, sizeof(kAdrenoCommonEvents) / sizeof(kAdrenoCommonEvents[0]), name);
        }
        if (event == nullptr) {
            return false;
        }
        binding->name = event->name;
        binding->group = event->group;
        binding->selector = event->selector;
        binding->slotsAvailable = event->slots;
        return true;
    }
    if (vendor == GpuVendor::Mali) {
        hwcpipe_counter counter;
        if (!findMaliCounter(productId, name, &counter)) {
            return false;
        }
        binding->name = name;
        binding->selector = static_cast<uint32_t>(counter);
        return true;
    }
    return false;
}

std::vector<std::string> supportedCounterNames(const DeviceInfo& device) {
    std::vector<std::string> names;
    if (device.vendor == GpuVendor::Adreno && isAdrenoA7xx(device.productId)) {
        size_t count = 0;
        const auto* events = detail::a7xxEvents(&count);
        for (size_t i = 0; i < count; ++i) {
            appendUnique(&names, events[i].name);
        }
        static const char* const normalized[] = {
            "gpu_active_cycles", "compute_active_cycles", "compute_tasks", "l2_any_lookup", "l2_ext_read",
            "l2_ext_write",
        };
        for (const char* name : normalized) {
            CounterBinding binding;
            if (resolveCounter(device.vendor, device.productId, name, &binding)) {
                appendUnique(&names, name);
            }
        }
        return names;
    }
    if (device.vendor != GpuVendor::Mali) {
        return names;
    }

    hwcpipe::device::product_id product;
    if (!rawMaliProduct(device.productId, &product)) {
        return names;
    }
    hwcpipe::gpu gpu(0);
    if (!gpu) {
        return names;
    }
    hwcpipe::counter_database database;
    for (const auto candidate : database.counters_for_gpu(gpu)) {
        hwcpipe::counter_metadata metadata;
        if (database.describe_counter(candidate, metadata) || metadata.name == nullptr) {
            continue;
        }
        hwcpipe::sampler_config config(product, 0);
        if (config.add_counter(candidate)) {
            continue;
        }
        appendUnique(&names, metadata.name);
    }
    static const char* const normalized[] = {
        "gpu_active_cycles", "compute_active_cycles", "compute_tasks", "l2_any_lookup", "l2_ext_read",
        "l2_ext_write",
    };
    for (const char* name : normalized) {
        CounterBinding binding;
        if (resolveCounter(device.vendor, device.productId, name, &binding)) {
            appendUnique(&names, name);
        }
    }
    return names;
}

Session::Session() : mImpl(new Impl()) {}

Session::~Session() {
    if (mImpl == nullptr) {
        return;
    }
    if (mImpl->started) {
        if (mImpl->vendor == GpuVendor::Adreno) {
            for (const auto& counter : mImpl->adrenoCounters) {
                hpc_gpu_adreno_ioctl_deactivate_counter(mImpl->adrenoFd, counter.binding.group, counter.binding.selector);
            }
        } else if (mImpl->maliSampler) {
            const auto ignored = mImpl->maliSampler->stop_sampling();
            (void)ignored;
        }
    }
    if (mImpl->adrenoFd >= 0) {
        hpc_gpu_adreno_ioctl_close_gpu_device(mImpl->adrenoFd);
    }
    delete mImpl;
}

Session* Session::create(const CounterSpec* specs, size_t count, DeviceInfo* device, const char** error) {
    if (error != nullptr) {
        *error = nullptr;
    }
    if (specs == nullptr || count == 0) {
        if (error != nullptr) *error = "At least one counter is required";
        return nullptr;
    }

    std::unique_ptr<Session> session(new Session());
    Impl& impl = *session->mImpl;
    impl.adrenoFd = hpc_gpu_adreno_ioctl_open_gpu_device();
    if (impl.adrenoFd >= 0) {
        const uint32_t gpuId = hpc_gpu_adreno_ioctl_get_gpu_device_id(impl.adrenoFd);
        if (gpuId >= 500 && gpuId < 800) {
            impl.vendor = GpuVendor::Adreno;
            impl.device.vendor = GpuVendor::Adreno;
            impl.device.productId = gpuId;
            impl.device.family = isAdrenoA7xx(gpuId) ? GpuFamily::AdrenoA7xx : GpuFamily::AdrenoLegacy;
            impl.device.productName = isAdrenoA7xx(gpuId) ? "Adreno A7xx" : "Adreno A5xx/A6xx";
            impl.device.driverName = "KGSL";
            for (size_t i = 0; i < count; ++i) {
                CounterBinding binding;
                if (!resolveCounter(GpuVendor::Adreno, gpuId, specs[i].name, &binding)) {
                    setError(&impl.error, "Counter is not supported by the detected Adreno GPU");
                    return nullptr;
                }
                Impl::AdrenoCounter counter;
                counter.spec = specs[i];
                counter.binding = binding;
                counter.raw.group_id = binding.group;
                counter.raw.countable_selector = binding.selector;
                counter.raw.value = 0;
                impl.adrenoCounters.emplace_back(counter);
            }
            if (device != nullptr) *device = impl.device;
            return session.release();
        }
        hpc_gpu_adreno_ioctl_close_gpu_device(impl.adrenoFd);
        impl.adrenoFd = -1;
    }

    hwcpipe::gpu gpu(0);
    if (!gpu) {
        // Try NVIDIA via CUPTI Range Profiler before giving up.
#if defined(MNN_PERFCOUNTER_HAS_CUDA)
        DeviceInfo nvDev;
        if (identifyNvidia(0, &nvDev)) {
            impl.vendor = GpuVendor::Nvidia;
            impl.device = nvDev;
            impl.nvMetricNames.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                impl.nvMetricNames.emplace_back(specs[i].name ? specs[i].name : "");
            }
            if (device != nullptr) *device = impl.device;
            return session.release();
        }
#endif
        if (error != nullptr) *error = kUnavailable;
        return nullptr;
    }
    const auto constants = gpu.get_constants();
    if (!identifyMali(constants.gpu_id, &impl.device)) {
        if (error != nullptr) *error = "Unknown Mali GPU product ID";
        return nullptr;
    }
    impl.vendor = GpuVendor::Mali;
    hwcpipe::sampler_config config(gpu);
    for (size_t i = 0; i < count; ++i) {
        hwcpipe_counter counter;
        if (!findMaliCounter(constants.gpu_id, specs[i].name, &counter) || config.add_counter(counter)) {
            if (error != nullptr) *error = "Counter is not supported by the detected Mali GPU";
            return nullptr;
        }
        impl.maliCounters.emplace_back(counter);
        impl.maliSpecs.emplace_back(specs[i]);
    }
    impl.maliSampler.reset(new hwcpipe::sampler<>(config));
    if (!impl.maliSampler || !(*impl.maliSampler)) {
        if (error != nullptr) *error = "Mali performance counter sampler could not be created";
        return nullptr;
    }
    if (device != nullptr) *device = impl.device;
    return session.release();
}

bool Session::start() {
    if (mImpl == nullptr || mImpl->started) {
        return false;
    }
    if (mImpl->vendor == GpuVendor::Adreno) {
        for (auto& counter : mImpl->adrenoCounters) {
            if (hpc_gpu_adreno_ioctl_activate_counter(mImpl->adrenoFd, counter.binding.group, counter.binding.selector) < 0) {
                mImpl->error = "KGSL counter activation failed";
                return false;
            }
        }
        std::vector<uint64_t> values(mImpl->adrenoCounters.size());
        std::vector<hpc_gpu_adreno_ioctl_counter_read_counter_t> raw(mImpl->adrenoCounters.size());
        for (size_t i = 0; i < raw.size(); ++i) raw[i] = mImpl->adrenoCounters[i].raw;
        if (hpc_gpu_adreno_ioctl_query_counters(mImpl->adrenoFd, static_cast<uint32_t>(raw.size()), raw.data(), values.data()) < 0) {
            mImpl->error = "KGSL counter read failed";
            return false;
        }
        for (size_t i = 0; i < raw.size(); ++i) mImpl->adrenoCounters[i].previous = values[i];
    } else if (mImpl->maliSampler) {
        if (mImpl->maliSampler->start_sampling()) {
            mImpl->error = "Mali counter accumulation start failed";
            return false;
        }
    } else if (mImpl->vendor == GpuVendor::Nvidia) {
#if defined(MNN_PERFCOUNTER_HAS_CUDA)
        // Lazy-build config/counterData images on first start() so create()
        // stays cheap when the session is only probed.
        if (mImpl->nvConfigImage.empty()) {
            // Resolve chipName via CUPTI.
            CUpti_Device_GetChipName_Params chipParams = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
            chipParams.deviceIndex = 0; // single-device assumption
            if (cuptiDeviceGetChipName(&chipParams) != CUPTI_SUCCESS) {
                mImpl->error = "cuptiDeviceGetChipName failed";
                return false;
            }
            mImpl->nvChipName = chipParams.pChipName ? chipParams.pChipName : "";
            CUpti_Profiler_Initialize_Params initParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
            if (cuptiProfilerInitialize(&initParams) != CUPTI_SUCCESS) {
                mImpl->error = "cuptiProfilerInitialize failed";
                return false;
            }
            size_t numPasses = 1;
            if (!nvBuildConfigImage(mImpl->nvChipName, mImpl->nvMetricNames, &mImpl->nvConfigImage, &numPasses, &mImpl->error)) {
                return false;
            }
            // If this metric configuration needs multiple passes (kernel replays
            // due to raw-counter conflicts), log it so the caller knows the
            // values come from N separate kernel executions.
            if (numPasses > 1) {
                std::fprintf(stderr, "[PMU] metric config requires %zu passes (metrics share hardware counter slots)\n", numPasses);
            }
            mImpl->nvNumPasses = numPasses;
            std::vector<uint8_t> prefix;
            if (!nvBuildCounterDataPrefix(mImpl->nvChipName, mImpl->nvMetricNames, &prefix, &mImpl->error)) {
                return false;
            }
            // Allocate counterDataImage + scratch via CounterDataImageOptions.
            CUpti_Profiler_CounterDataImageOptions opts = {CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE};
            opts.pCounterDataPrefix = prefix.data();
            opts.counterDataPrefixSize = prefix.size();
            opts.maxNumRanges = 1; // caller-driven; one range per beginRange/endRange per session
            opts.maxNumRangeTreeNodes = 1;
            opts.maxRangeNameLength = 256;

            CUpti_Profiler_CounterDataImage_CalculateSize_Params sizeParams = {CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};
            sizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
            sizeParams.pOptions = &opts;
            if (cuptiProfilerCounterDataImageCalculateSize(&sizeParams) != CUPTI_SUCCESS) {
                mImpl->error = "cuptiProfilerCounterDataImageCalculateSize failed";
                return false;
            }
            mImpl->nvCounterDataImage.resize(sizeParams.counterDataImageSize);
            CUpti_Profiler_CounterDataImage_Initialize_Params initImg = {CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
            initImg.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
            initImg.pOptions = &opts;
            initImg.counterDataImageSize = mImpl->nvCounterDataImage.size();
            initImg.pCounterDataImage = mImpl->nvCounterDataImage.data();
            if (cuptiProfilerCounterDataImageInitialize(&initImg) != CUPTI_SUCCESS) {
                mImpl->error = "cuptiProfilerCounterDataImageInitialize failed";
                return false;
            }
            CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params sbs = {CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
            sbs.counterDataImageSize = mImpl->nvCounterDataImage.size();
            sbs.pCounterDataImage = mImpl->nvCounterDataImage.data();
            if (cuptiProfilerCounterDataImageCalculateScratchBufferSize(&sbs) != CUPTI_SUCCESS) {
                mImpl->error = "scratch size failed";
                return false;
            }
            mImpl->nvCounterDataScratch.resize(sbs.counterDataScratchBufferSize);
            CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params isb = {CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
            isb.counterDataImageSize = mImpl->nvCounterDataImage.size();
            isb.pCounterDataImage = mImpl->nvCounterDataImage.data();
            isb.counterDataScratchBufferSize = mImpl->nvCounterDataScratch.size();
            isb.pCounterDataScratchBuffer = mImpl->nvCounterDataScratch.data();
            if (cuptiProfilerCounterDataImageInitializeScratchBuffer(&isb) != CUPTI_SUCCESS) {
                mImpl->error = "scratch init failed";
                return false;
            }
        }
        // Get current CUDA context.
        CUcontext cuCtx = nullptr;
        if (cuCtxGetCurrent(&cuCtx) != CUDA_SUCCESS) {
            mImpl->error = "cuCtxGetCurrent failed";
            return false;
        }
        mImpl->nvCtx = cuCtx;
        CUpti_RangeProfiler_Enable_Params en = {CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE};
        en.ctx = cuCtx;
        if (cuptiRangeProfilerEnable(&en) != CUPTI_SUCCESS) {
            mImpl->error = "cuptiRangeProfilerEnable failed";
            return false;
        }
        mImpl->nvRangeObj = en.pRangeProfilerObject;
        CUpti_RangeProfiler_SetConfig_Params sc = {CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE};
        sc.pRangeProfilerObject = en.pRangeProfilerObject;
        sc.pConfig = mImpl->nvConfigImage.data();
        sc.configSize = mImpl->nvConfigImage.size();
        sc.counterDataImageSize = mImpl->nvCounterDataImage.size();
        sc.pCounterDataImage = mImpl->nvCounterDataImage.data();
        // The corpus benchmark emits one target kernel while this session is
        // active. AutoRange + KernelReplay lets CUPTI identify that kernel and
        // replay all required passes without application-side replay.
        sc.range = CUPTI_AutoRange;
        sc.replayMode = CUPTI_KernelReplay;
        sc.maxRangesPerPass = 1;
        sc.numNestingLevels = 1;
        sc.minNestingLevel = 1;
        sc.passIndex = 0;
        if (cuptiRangeProfilerSetConfig(&sc) != CUPTI_SUCCESS) {
            mImpl->error = "cuptiRangeProfilerSetConfig failed";
            return false;
        }
        mImpl->nvAutoRange = true;
        CUpti_RangeProfiler_Start_Params st = {CUpti_RangeProfiler_Start_Params_STRUCT_SIZE};
        st.pRangeProfilerObject = en.pRangeProfilerObject;
        CUptiResult sr = cuptiRangeProfilerStart(&st);
        if (sr != CUPTI_SUCCESS) {
            const char* es = nullptr;
            cuptiGetResultString(sr, &es);
            mImpl->error = std::string("cuptiRangeProfilerStart failed: ") + (es ? es : "unknown");
            return false;
        }
#else
        mImpl->error = kUnavailable;
        return false;
#endif
    } else {
        mImpl->error = kUnavailable;
        return false;
    }
    mImpl->started = true;
    return true;
}

bool Session::stop(CounterValue* values, size_t count) {
    size_t requiredCount = 0;
    if (mImpl != nullptr) {
        if (mImpl->vendor == GpuVendor::Adreno) {
            requiredCount = mImpl->adrenoCounters.size();
        } else if (mImpl->maliSampler) {
            requiredCount = mImpl->maliCounters.size();
        } else if (mImpl->vendor == GpuVendor::Nvidia) {
            requiredCount = mImpl->nvMetricNames.size();
        }
    }
    if (mImpl == nullptr || !mImpl->started || values == nullptr || count < requiredCount) {
        return false;
    }
    bool ok = true;
    if (mImpl->vendor == GpuVendor::Adreno) {
        std::vector<uint64_t> current(mImpl->adrenoCounters.size());
        std::vector<hpc_gpu_adreno_ioctl_counter_read_counter_t> raw(mImpl->adrenoCounters.size());
        for (size_t i = 0; i < raw.size(); ++i) raw[i] = mImpl->adrenoCounters[i].raw;
        if (hpc_gpu_adreno_ioctl_query_counters(mImpl->adrenoFd, static_cast<uint32_t>(raw.size()), raw.data(), current.data()) < 0) {
            mImpl->error = "KGSL counter read failed";
            ok = false;
        } else {
            for (size_t i = 0; i < raw.size(); ++i) {
                values[i].name = mImpl->adrenoCounters[i].spec.name;
                values[i].value = current[i] >= mImpl->adrenoCounters[i].previous ? current[i] - mImpl->adrenoCounters[i].previous : 0;
                values[i].floatingPointValue = 0.0;
                values[i].valueKind = CounterValueKind::UnsignedInteger;
                values[i].status = CounterValueStatus::Valid;
            }
        }
        for (const auto& counter : mImpl->adrenoCounters) {
            if (hpc_gpu_adreno_ioctl_deactivate_counter(mImpl->adrenoFd, counter.binding.group, counter.binding.selector) < 0) ok = false;
        }
    } else if (mImpl->maliSampler) {
        if (mImpl->maliSampler->sample_now()) {
            mImpl->error = "Mali counter read failed";
            ok = false;
        } else {
            hwcpipe::counter_sample sample;
            for (size_t i = 0; i < mImpl->maliCounters.size(); ++i) {
                if (mImpl->maliSampler->get_counter_value(mImpl->maliCounters[i], sample) || sample.type != hwcpipe::counter_sample::type::uint64) {
                    ok = false;
                    continue;
                }
                values[i].name = mImpl->maliSpecs[i].name;
                values[i].value = sample.value.uint64;
                values[i].floatingPointValue = 0.0;
                values[i].valueKind = CounterValueKind::UnsignedInteger;
                values[i].status = CounterValueStatus::Valid;
            }
        }
        if (mImpl->maliSampler->stop_sampling()) ok = false;
    } else if (mImpl->vendor == GpuVendor::Nvidia) {
#if defined(MNN_PERFCOUNTER_HAS_CUDA)
        // Stop the range profiler session and evaluate collected metrics.
        CUpti_RangeProfiler_Object* rangeObj = static_cast<CUpti_RangeProfiler_Object*>(mImpl->nvRangeObj);
        CUpti_RangeProfiler_Stop_Params st = {CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE};
        st.pRangeProfilerObject = rangeObj;
        CUptiResult sr = cuptiRangeProfilerStop(&st);
        if (sr != CUPTI_SUCCESS) {
            const char* es = nullptr;
            cuptiGetResultString(sr, &es);
            mImpl->error = std::string("cuptiRangeProfilerStop failed: ") + (es ? es : "unknown");
            ok = false;
        }
        CUpti_RangeProfiler_Disable_Params dis = {CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE};
        dis.pRangeProfilerObject = rangeObj;
        cuptiRangeProfilerDisable(&dis); // best-effort
        mImpl->nvRangeObj = nullptr;

        if (ok) {
            // Evaluate metrics from the counterDataImage (which holds all
            // collected ranges). nvEvaluateMetrics sums across ranges and
            // fills values[i] for metric i.
            std::string evalErr;
            if (!nvEvaluateMetrics(mImpl->nvChipName, mImpl->nvCounterDataImage, mImpl->nvMetricNames,
                                   values, count, &evalErr)) {
                mImpl->error = "nvEvaluateMetrics failed: " + evalErr;
                ok = false;
            }
        }
#else
        mImpl->error = kUnavailable;
        ok = false;
#endif
    }
    mImpl->started = false;
    return ok;
}

const char* Session::error() const {
    return mImpl == nullptr ? kUnavailable : mImpl->error.c_str();
}

size_t Session::numPasses() const {
    return mImpl == nullptr ? 1 : mImpl->nvNumPasses;
}

bool Session::beginRange(const char* rangeName) {
    if (mImpl == nullptr || !mImpl->started || mImpl->vendor != GpuVendor::Nvidia) return false;
    if (mImpl->nvRangeObj == nullptr) return false;
    if (mImpl->nvAutoRange) {
        (void)rangeName;
        return true;
    }
#if defined(MNN_PERFCOUNTER_HAS_CUDA)
    if (!nvBeginRange(mImpl->nvRangeObj, rangeName)) {
        mImpl->error = "cuptiRangeProfilerPushRange failed";
        return false;
    }
    ++mImpl->nvRangeDepth;
    return true;
#else
    (void)rangeName;
    return false;
#endif
}

bool Session::endRange() {
    if (mImpl == nullptr || mImpl->vendor != GpuVendor::Nvidia) return false;
    if (mImpl->nvAutoRange) return true;
    if (mImpl->nvRangeDepth == 0) return false;
#if defined(MNN_PERFCOUNTER_HAS_CUDA)
    if (!nvEndRange(mImpl->nvRangeObj)) {
        mImpl->error = "cuptiRangeProfilerPopRange failed";
        return false;
    }
    --mImpl->nvRangeDepth;
    return true;
#else
    return false;
#endif
}

} // namespace PerfCounter
} // namespace MNN
