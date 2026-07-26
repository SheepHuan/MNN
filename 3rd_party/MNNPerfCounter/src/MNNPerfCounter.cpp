#include "MNNPerfCounter.hpp"
#include "A7xxPerfCounters.hpp"

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
    } else {
        mImpl->error = kUnavailable;
        return false;
    }
    mImpl->started = true;
    return true;
}

bool Session::stop(CounterValue* values, size_t count) {
    if (mImpl == nullptr || !mImpl->started || values == nullptr || count < mImpl->adrenoCounters.size() + mImpl->maliCounters.size()) {
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
            }
        }
        if (mImpl->maliSampler->stop_sampling()) ok = false;
    }
    mImpl->started = false;
    return ok;
}

const char* Session::error() const {
    return mImpl == nullptr ? kUnavailable : mImpl->error.c_str();
}

} // namespace PerfCounter
} // namespace MNN
