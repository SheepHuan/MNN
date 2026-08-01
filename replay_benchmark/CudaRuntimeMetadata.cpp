#include "CudaRuntimeMetadata.hpp"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <sstream>

#include <cuda_runtime.h>

#if defined(MNN_REPLAY_HAS_CUPTI_ACTIVITY)
#include <cupti_activity.h>
#endif

#if defined(MNN_REPLAY_HAS_NVML)
#include <nvml.h>
#endif

namespace MNN {
namespace Replay {
namespace KernelCorpus {

namespace {

#if defined(MNN_REPLAY_HAS_CUPTI_ACTIVITY)
static std::string cuptiErrorString(CUptiResult result) {
    const char* text = nullptr;
    cuptiGetResultString(result, &text);
    return text == nullptr ? "unknown CUPTI error" : text;
}
#endif

} // namespace

struct CudaLaunchMetadataCollector::Impl {
    std::vector<CudaLaunchRecord> records;
    std::string status = "unavailable:CUPTI Activity was not built";
    bool started = false;

#if defined(MNN_REPLAY_HAS_CUPTI_ACTIVITY)
    static Impl*& activeCollector() {
        static Impl* collector = nullptr;
        return collector;
    }

    static std::mutex& activityMutex() {
        static std::mutex mutex;
        return mutex;
    }

    static bool& callbacksRegistered() {
        static bool registered = false;
        return registered;
    }

    static void CUPTIAPI requestBuffer(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
        constexpr size_t kBufferSize = 1024 * 1024;
        void* storage = nullptr;
        if (posix_memalign(&storage, 8, kBufferSize) != 0) storage = nullptr;
        *buffer = static_cast<uint8_t*>(storage);
        *size = storage == nullptr ? 0 : kBufferSize;
        *maxNumRecords = 0;
        if (storage == nullptr) {
            std::lock_guard<std::mutex> lock(activityMutex());
            if (activeCollector() != nullptr) {
                activeCollector()->status = "unavailable:CUPTI Activity buffer allocation failed";
            }
        }
    }

    static void CUPTIAPI completeBuffer(CUcontext, uint32_t, uint8_t* buffer,
                                        size_t, size_t validSize) {
        if (buffer == nullptr) return;
        Impl* collector = nullptr;
        {
            std::lock_guard<std::mutex> lock(activityMutex());
            collector = activeCollector();
        }
        if (collector != nullptr && validSize > 0) {
            CUpti_Activity* activity = nullptr;
            CUptiResult result = CUPTI_SUCCESS;
            while ((result = cuptiActivityGetNextRecord(buffer, validSize, &activity)) == CUPTI_SUCCESS) {
                if (activity->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL &&
                    activity->kind != CUPTI_ACTIVITY_KIND_KERNEL) {
                    continue;
                }
                const auto* kernel = reinterpret_cast<const CUpti_ActivityKernel11*>(activity);
                CudaLaunchRecord record;
                record.kernelName = kernel->name == nullptr ? "unknown" : kernel->name;
                record.grid[0] = kernel->gridX;
                record.grid[1] = kernel->gridY;
                record.grid[2] = kernel->gridZ;
                record.block[0] = kernel->blockX;
                record.block[1] = kernel->blockY;
                record.block[2] = kernel->blockZ;
                record.registersPerThread = kernel->registersPerThread;
                record.staticSharedMemoryBytes = static_cast<uint64_t>(std::max(0, kernel->staticSharedMemory));
                record.dynamicSharedMemoryBytes = static_cast<uint64_t>(std::max(0, kernel->dynamicSharedMemory));
                record.localMemoryPerThreadBytes = kernel->localMemoryPerThread;
                record.localMemoryTotalBytes = kernel->localMemoryTotal_v2;
                std::lock_guard<std::mutex> lock(activityMutex());
                if (activeCollector() == collector) collector->records.emplace_back(std::move(record));
            }
            if (result != CUPTI_ERROR_MAX_LIMIT_REACHED) {
                std::lock_guard<std::mutex> lock(activityMutex());
                if (activeCollector() == collector) {
                    collector->status = "partial:" + cuptiErrorString(result);
                }
            }
        }
        std::free(buffer);
    }
#endif
};

CudaLaunchMetadataCollector::CudaLaunchMetadataCollector() : mImpl(new Impl) {
}

CudaLaunchMetadataCollector::~CudaLaunchMetadataCollector() {
    stop();
}

bool CudaLaunchMetadataCollector::start() {
#if !defined(MNN_REPLAY_HAS_CUPTI_ACTIVITY)
    return false;
#else
    if (mImpl->started) return true;
    mImpl->records.clear();
    {
        std::lock_guard<std::mutex> lock(Impl::activityMutex());
        if (Impl::activeCollector() != nullptr) {
            mImpl->status = "unavailable:CUPTI Activity collector is busy";
            return false;
        }
        if (!Impl::callbacksRegistered()) {
            CUptiResult result = cuptiActivityRegisterCallbacks(&Impl::requestBuffer, &Impl::completeBuffer);
            if (result != CUPTI_SUCCESS) {
                mImpl->status = "unavailable:" + cuptiErrorString(result);
                return false;
            }
            Impl::callbacksRegistered() = true;
        }
        Impl::activeCollector() = mImpl.get();
    }
    CUptiResult result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (result != CUPTI_SUCCESS) {
        std::lock_guard<std::mutex> lock(Impl::activityMutex());
        if (Impl::activeCollector() == mImpl.get()) Impl::activeCollector() = nullptr;
        mImpl->status = "unavailable:" + cuptiErrorString(result);
        return false;
    }
    mImpl->started = true;
    mImpl->status = "started";
    return true;
#endif
}

void CudaLaunchMetadataCollector::stop() {
#if defined(MNN_REPLAY_HAS_CUPTI_ACTIVITY)
    if (!mImpl->started) return;
    CUptiResult flushResult = cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    CUptiResult disableResult = cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    size_t dropped = 0;
    CUptiResult droppedResult = cuptiActivityGetNumDroppedRecords(nullptr, 0, &dropped);
    {
        std::lock_guard<std::mutex> lock(Impl::activityMutex());
        if (Impl::activeCollector() == mImpl.get()) Impl::activeCollector() = nullptr;
    }
    mImpl->started = false;
    if (flushResult != CUPTI_SUCCESS) {
        mImpl->status = "partial:flush failed: " + cuptiErrorString(flushResult);
    } else if (disableResult != CUPTI_SUCCESS) {
        mImpl->status = "partial:disable failed: " + cuptiErrorString(disableResult);
    } else if (droppedResult == CUPTI_SUCCESS && dropped > 0) {
        mImpl->status = "partial:dropped_records=" + std::to_string(dropped);
    } else if (mImpl->records.empty()) {
        mImpl->status = "unavailable:no kernel activity records";
    } else if (mImpl->status == "started") {
        mImpl->status = "sampled";
    }
#endif
}

const std::vector<CudaLaunchRecord>& CudaLaunchMetadataCollector::records() const {
    return mImpl->records;
}

const std::string& CudaLaunchMetadataCollector::status() const {
    return mImpl->status;
}

struct CudaEnvironmentSampler::Impl {
    CudaEnvironmentObservation observation;

#if defined(MNN_REPLAY_HAS_NVML)
    nvmlDevice_t device = nullptr;
    bool initialized = false;
    std::string lastError;

    Impl() {
        nvmlReturn_t result = nvmlInit_v2();
        if (result != NVML_SUCCESS) {
            observation.samplingStatus = std::string("unavailable:") + nvmlErrorString(result);
            return;
        }
        initialized = true;
        int cudaDevice = 0;
        char pciBusId[32] = {0};
        cudaError_t cudaResult = cudaGetDevice(&cudaDevice);
        if (cudaResult != cudaSuccess ||
            cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), cudaDevice) != cudaSuccess) {
            observation.samplingStatus = "unavailable:cannot map CUDA device to PCI bus ID";
            return;
        }
        result = nvmlDeviceGetHandleByPciBusId_v2(pciBusId, &device);
        if (result != NVML_SUCCESS) {
            observation.samplingStatus = std::string("unavailable:") + nvmlErrorString(result);
            device = nullptr;
            return;
        }
        observation.samplingStatus = "ready";
    }

    ~Impl() {
        if (initialized) nvmlShutdown();
    }

    void sample(OptionalRuntimeNumber* clock, OptionalRuntimeNumber* temperature) {
        if (device == nullptr) return;
        unsigned int clockMHz = 0;
        nvmlReturn_t clockResult = nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &clockMHz);
        if (clockResult == NVML_SUCCESS) {
            clock->present = true;
            clock->value = static_cast<double>(clockMHz) * 1000000.0;
        } else {
            lastError = nvmlErrorString(clockResult);
        }
        nvmlReturn_t temperatureResult = NVML_ERROR_NOT_SUPPORTED;
        double temperatureC = 0.0;
#if defined(nvmlTemperature_v1)
        nvmlTemperature_t temperatureInfo = {};
        temperatureInfo.version = nvmlTemperature_v1;
        temperatureInfo.sensorType = NVML_TEMPERATURE_GPU;
        temperatureResult = nvmlDeviceGetTemperatureV(device, &temperatureInfo);
        temperatureC = static_cast<double>(temperatureInfo.temperature);
#else
        unsigned int legacyTemperatureC = 0;
        temperatureResult = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &legacyTemperatureC);
        temperatureC = static_cast<double>(legacyTemperatureC);
#endif
        if (temperatureResult == NVML_SUCCESS) {
            temperature->present = true;
            temperature->value = temperatureC;
        } else {
            lastError = nvmlErrorString(temperatureResult);
        }
    }

    void updateStatus() {
        const int present = static_cast<int>(observation.gpuClockHzBefore.present) +
                            static_cast<int>(observation.gpuClockHzAfter.present) +
                            static_cast<int>(observation.temperatureCBefore.present) +
                            static_cast<int>(observation.temperatureCAfter.present);
        if (present == 4) {
            observation.samplingStatus = "sampled";
        } else if (present > 0) {
            observation.samplingStatus = lastError.empty() ? "partial" : "partial:" + lastError;
        } else if (device != nullptr) {
            observation.samplingStatus = lastError.empty() ? "unavailable:no NVML values" : "unavailable:" + lastError;
        }
    }
#else
    Impl() {
        observation.samplingStatus = "unavailable:NVML was not built";
    }

    void sample(OptionalRuntimeNumber*, OptionalRuntimeNumber*) {
    }

    void updateStatus() {
    }
#endif
};

CudaEnvironmentSampler::CudaEnvironmentSampler() : mImpl(new Impl) {
}

CudaEnvironmentSampler::~CudaEnvironmentSampler() = default;

void CudaEnvironmentSampler::sampleBefore() {
    mImpl->sample(&mImpl->observation.gpuClockHzBefore, &mImpl->observation.temperatureCBefore);
    mImpl->updateStatus();
}

void CudaEnvironmentSampler::sampleAfter() {
    mImpl->sample(&mImpl->observation.gpuClockHzAfter, &mImpl->observation.temperatureCAfter);
    mImpl->updateStatus();
}

const CudaEnvironmentObservation& CudaEnvironmentSampler::observation() const {
    return mImpl->observation;
}

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
