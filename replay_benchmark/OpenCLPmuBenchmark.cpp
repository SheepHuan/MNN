#include "OpenCLPmuBenchmark.hpp"

#include "ReplayRecord.hpp"
#include "OpenCLPmuBufferKernels.hpp"
#include "OpenCLPmuComputeKernels.hpp"
#include "OpenCLPmuImageKernels.hpp"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <sys/stat.h>
#include <vector>

#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/writer.h"

#if defined(MNN_REPLAY_HAS_OPENCL)
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/runtime/OpenCLWrapper.hpp"
#endif

#if defined(MNN_REPLAY_HAS_PERFCOUNTER)
#include "MNNPerfCounter.hpp"
#endif

namespace MNN {
namespace Replay {

namespace {

static const OpenCLPmuCaseInfo kCases[] = {
    {"buffer_reuse_small", "global buffer L1 locality", "pmu_buffer_reuse", nullptr, OpenCLPmuMemoryPath::Buffer, true, 100},
    {"buffer_reuse_l2", "global buffer L2 locality", "pmu_buffer_reuse", nullptr, OpenCLPmuMemoryPath::Buffer, true, 80},
    {"buffer_stride", "global buffer strided access", "pmu_buffer_stride", nullptr, OpenCLPmuMemoryPath::Buffer, true, 80},
    {"buffer_pchase", "global buffer pointer chasing", "pmu_buffer_pchase", nullptr, OpenCLPmuMemoryPath::Buffer, true, 60},
    {"buffer_vec4", "vectorized global buffer access", "pmu_buffer_vec4", nullptr, OpenCLPmuMemoryPath::Buffer, true, 80},
    {"buffer_stream_large", "global buffer L2 miss and external traffic", "pmu_buffer_stream", nullptr, OpenCLPmuMemoryPath::Buffer, true, 40},
    {"buffer_write", "global buffer write traffic", "pmu_buffer_write", nullptr, OpenCLPmuMemoryPath::Buffer, true, 80},
    {"buffer_int32", "integer arithmetic and buffer access", "pmu_int_compute", nullptr, OpenCLPmuMemoryPath::Compute, true, 100},
    {"buffer_fp32", "FP32 arithmetic and buffer access", "pmu_fp32_compute", nullptr, OpenCLPmuMemoryPath::Compute, true, 100},
    {"fp32_throughput", "FP32 arithmetic throughput", "pmu_fp32_throughput", nullptr, OpenCLPmuMemoryPath::Compute, true, 100},
    {"buffer_fp16", "FP16 arithmetic and buffer access", "pmu_fp16_compute", "cl_khr_fp16", OpenCLPmuMemoryPath::Compute, true, 100},
    {"fp16_throughput", "FP16 arithmetic throughput", "pmu_fp16_throughput", "cl_khr_fp16", OpenCLPmuMemoryPath::Compute, true, 100},
    {"constant_memory", "constant address space reads", "pmu_constant_compute", nullptr, OpenCLPmuMemoryPath::Compute, true, 100},
    {"constant_bandwidth", "constant memory working-set bandwidth", "pmu_constant_bandwidth", nullptr, OpenCLPmuMemoryPath::Compute, true, 100},
    {"local_memory", "local address space and barriers", "pmu_local_memory", nullptr, OpenCLPmuMemoryPath::Compute, true, 80},
    {"local_bandwidth", "local memory bandwidth without loop barriers", "pmu_local_bandwidth", nullptr, OpenCLPmuMemoryPath::Compute, true, 80},
    {"local_barrier", "barrier synchronization cost", "pmu_local_barrier", nullptr, OpenCLPmuMemoryPath::Compute, true, 80},
    {"global_atomic", "global atomic memory ordering", "pmu_atomic", nullptr, OpenCLPmuMemoryPath::Compute, true, 50},
    {"atomic_contended", "contended global atomic updates", "pmu_atomic_contended", nullptr, OpenCLPmuMemoryPath::Compute, true, 50},
    {"atomic_distributed", "distributed global atomic updates", "pmu_atomic_distributed", nullptr, OpenCLPmuMemoryPath::Compute, true, 50},
    {"image_reuse", "image object 2D locality", "pmu_image_read", nullptr, OpenCLPmuMemoryPath::Image, true, 80},
    {"image_write", "image object write traffic", "pmu_image_write", nullptr, OpenCLPmuMemoryPath::Image, true, 60},
    {"texture_nearest", "texture cache nearest sampling", "pmu_texture_read_nearest", nullptr, OpenCLPmuMemoryPath::Texture, true, 80},
    {"texture_linear", "texture cache linear filtering", "pmu_texture_read_linear", nullptr, OpenCLPmuMemoryPath::Texture, true, 80},
    {"image_stride_x", "image horizontal stride", "pmu_image_stride_x", nullptr, OpenCLPmuMemoryPath::Image, true, 80},
    {"image_stride_y", "image vertical stride", "pmu_image_stride_y", nullptr, OpenCLPmuMemoryPath::Image, true, 80},
};

// Keep the default profile small enough to fit every A7xx physical group in a
// single KGSL configuration. A custom --perf-counter-events profile is split
// into additional batches below when it exceeds a group's slot count.
static const char* const kDefaultAdrenoCounters[] = {
    "cp_busy_cycles",
    "rbbm_vbif_busy",
    "hlsq_busy_cycles",
    "uche_busy_cycles",
    "uche_vbif_read_beats_sp",
    "tp_busy_cycles",
    "tp_l1_cacheline_requests",
    "tp_l1_cacheline_misses",
    "sp_busy_cycles",
    "sp_alu_working_cycles",
    "sp_stall_cycles_uche",
    "sp_lm_load_instructions",
    "sp_lm_store_instructions",
    "sp_lm_atomics",
    "sp_gm_load_instructions",
    "sp_gm_store_instructions",
    "sp_gm_atomics",
    "sp_cs_instructions",
    "sp_lm_bank_conflicts",
    "sp_working_eu_cs_stage",
    "sp_any_eu_working_cs_stage",
    "sp_gm_load_latency_cycles",
    "sp_gm_load_latency_samples",
    "sp_executable_waves",
    "sp_cs_invocations",
    "rb_busy_cycles",
};

static const char* const kPortableCounters[] = {
    "gpu_active_cycles",
    "compute_active_cycles",
    "compute_tasks",
    "l2_any_lookup",
    "l2_ext_read",
    "l2_ext_write",
};

static uint64_t monotonicNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch()).count());
}

static bool hasExtension(const std::string& extensions, const char* extension) {
    return extension == nullptr || extensions.find(extension) != std::string::npos;
}

static bool makeParentDirectories(const std::string& path) {
    const std::string::size_type slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return true;
    const std::string parent = path.substr(0, slash);
    if (parent.empty()) return true;
    std::string current;
    for (size_t i = 0; i < parent.size(); ++i) {
        current.push_back(parent[i]);
        if (parent[i] != '/' && parent[i] != '\\' && i + 1 != parent.size()) continue;
        if (current != "/" && mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) return false;
    }
    return mkdir(parent.c_str(), 0755) == 0 || errno == EEXIST;
}

static void addString(rapidjson::Value& object, const char* key, const std::string& value,
                      rapidjson::Document::AllocatorType& allocator) {
    rapidjson::Value k(key, allocator);
    rapidjson::Value v(value.c_str(), allocator);
    object.AddMember(k, v, allocator);
}

static void addGeometry(rapidjson::Value& object, const OpenCLPmuGeometry& geometry,
                        rapidjson::Document::AllocatorType& allocator) {
    rapidjson::Value global(rapidjson::kObjectType);
    global.AddMember("x", static_cast<uint64_t>(geometry.global[0]), allocator);
    global.AddMember("y", static_cast<uint64_t>(geometry.global[1]), allocator);
    rapidjson::Value local(rapidjson::kObjectType);
    local.AddMember("x", static_cast<uint64_t>(geometry.local[0]), allocator);
    local.AddMember("y", static_cast<uint64_t>(geometry.local[1]), allocator);
    object.AddMember("global_size", global, allocator);
    object.AddMember("local_size", local, allocator);
    object.AddMember("uses_local_size", geometry.usesLocalSize, allocator);
}

static const char* pathName(OpenCLPmuMemoryPath path) {
    switch (path) {
        case OpenCLPmuMemoryPath::Buffer: return "buffer";
        case OpenCLPmuMemoryPath::Image: return "image";
        case OpenCLPmuMemoryPath::Texture: return "texture";
        case OpenCLPmuMemoryPath::Compute: return "compute";
    }
    return "unknown";
}

#if defined(MNN_REPLAY_HAS_PERFCOUNTER)

struct PmuBatch {
    std::vector<std::string> names;
};

struct PmuInterval {
    bool available = false;
    uint64_t startNs = 0;
    uint64_t endNs = 0;
    uint32_t dispatches = 0;
    std::vector<MNN::PerfCounter::CounterValue> values;
    std::string error;
};

static std::vector<std::string> defaultPmuCounterNames(const MNN::PerfCounter::DeviceInfo& device) {
    const char* const* names = kPortableCounters;
    size_t count = sizeof(kPortableCounters) / sizeof(kPortableCounters[0]);
    if (device.vendor == MNN::PerfCounter::GpuVendor::Adreno &&
        device.family == MNN::PerfCounter::GpuFamily::AdrenoA7xx) {
        names = kDefaultAdrenoCounters;
        count = sizeof(kDefaultAdrenoCounters) / sizeof(kDefaultAdrenoCounters[0]);
    }
    return std::vector<std::string>(names, names + count);
}

static std::vector<std::string> parsePmuCounterList(const std::string& text) {
    std::vector<std::string> result;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find(',', start);
        if (end == std::string::npos) end = text.size();
        size_t first = start;
        while (first < end && text[first] == ' ') ++first;
        size_t last = end;
        while (last > first && text[last - 1] == ' ') --last;
        if (last > first) result.emplace_back(text.substr(first, last - first));
        start = end + 1;
    }
    return result;
}

static bool discoverPmuDevice(MNN::PerfCounter::DeviceInfo* device, std::string* error) {
    MNN::PerfCounter::CounterSpec spec;
    spec.name = "gpu_active_cycles";
    const char* sessionError = nullptr;
    std::unique_ptr<MNN::PerfCounter::Session> session(
        MNN::PerfCounter::Session::create(&spec, 1, device, &sessionError));
    if (session == nullptr) {
        if (error != nullptr) *error = sessionError == nullptr ? "GPU PMU is unavailable" : sessionError;
        return false;
    }
    return true;
}

static std::vector<PmuBatch> makePmuBatches(const std::vector<std::string>& names,
                                            const MNN::PerfCounter::DeviceInfo& device,
                                            std::vector<std::string>* unsupported) {
    std::vector<PmuBatch> batches;
    std::map<uint32_t, size_t> groupCounts;
    for (const auto& name : names) {
        MNN::PerfCounter::CounterBinding binding;
        if (device.vendor == MNN::PerfCounter::GpuVendor::Mali) {
            if (!MNN::PerfCounter::resolveCounter(device.vendor, device.productId, name.c_str(), &binding)) {
                if (unsupported != nullptr) unsupported->push_back(name);
                continue;
            }
            if (batches.empty()) batches.emplace_back();
            batches.front().names.push_back(name);
            continue;
        }
        if (!MNN::PerfCounter::resolveCounter(device.vendor, device.productId, name.c_str(), &binding)) {
            if (unsupported != nullptr) unsupported->push_back(name);
            continue;
        }
        size_t batchIndex = 0;
        for (; batchIndex < batches.size(); ++batchIndex) {
            const size_t used = groupCounts[static_cast<uint32_t>(batchIndex) << 16 | binding.group];
            if (used < binding.slotsAvailable) break;
        }
        if (batchIndex == batches.size()) batches.emplace_back();
        batches[batchIndex].names.push_back(name);
        groupCounts[static_cast<uint32_t>(batchIndex) << 16 | binding.group]++;
    }
    return batches;
}

#endif

} // namespace

size_t openclPmuCaseCount() {
    return sizeof(kCases) / sizeof(kCases[0]);
}

const OpenCLPmuCaseInfo* openclPmuCases() {
    return kCases;
}

const OpenCLPmuCaseInfo* findOpenclPmuCase(const char* name) {
    if (name == nullptr) return nullptr;
    for (size_t i = 0; i < openclPmuCaseCount(); ++i) {
        if (std::strcmp(kCases[i].name, name) == 0) return kCases + i;
    }
    return nullptr;
}

std::vector<std::string> parseOpenclPmuCaseList(const std::string& text) {
    std::vector<std::string> result;
    if (text.empty() || text == "all") {
        for (size_t i = 0; i < openclPmuCaseCount(); ++i) result.emplace_back(kCases[i].name);
        return result;
    }
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find(',', start);
        if (end == std::string::npos) end = text.size();
        if (end > start) result.emplace_back(text.substr(start, end - start));
        start = end + 1;
    }
    return result;
}

OpenCLPmuSignalResult classifyPmuSignal(uint64_t controlDelta, uint64_t workloadDelta,
                                        bool expectedIncrease, uint64_t threshold) {
    OpenCLPmuSignalResult result;
    result.readable = true;
    const uint64_t difference = workloadDelta >= controlDelta ? workloadDelta - controlDelta : controlDelta - workloadDelta;
    result.responsive = difference > threshold;
    result.discriminative = result.responsive;
    result.valid = expectedIncrease ? (result.responsive && workloadDelta > controlDelta)
                                    : (result.responsive && workloadDelta < controlDelta);
    return result;
}

OpenCLPmuGeometry makeOpenclPmuGeometry(const OpenCLPmuCaseInfo& info, size_t count, size_t localSize) {
    OpenCLPmuGeometry geometry;
    if (std::strcmp(info.name, "buffer_pchase") == 0) {
        geometry.global[0] = 1;
        geometry.global[1] = 1;
        return geometry;
    }
    if (info.path == OpenCLPmuMemoryPath::Image || info.path == OpenCLPmuMemoryPath::Texture) {
        geometry.global[0] = 256;
        geometry.global[1] = 256;
        return geometry;
    }
    const size_t effectiveLocalSize = localSize == 0 ? 64 : localSize;
    geometry.usesLocalSize = true;
    geometry.local[0] = effectiveLocalSize;
    geometry.local[1] = 1;
    geometry.global[0] = ((count + effectiveLocalSize - 1) / effectiveLocalSize) * effectiveLocalSize;
    geometry.global[1] = 1;
    return geometry;
}

#if defined(MNN_REPLAY_HAS_OPENCL)

namespace {

struct OpenCLPmuRuntime {
    std::unique_ptr<MNN::OpenCL::CLRuntime> runtime;
    std::unique_ptr<MNN::Backend> backend;
    MNN::OpenCLRuntime* opencl = nullptr;
    cl::Device device;
    std::string extensions;
    bool imageSupport = false;
};

static bool initializeOpenCL(OpenCLPmuRuntime* output, std::string* error) {
    if (output == nullptr) return false;
#if defined(MNN_USE_LIB_WRAPPER)
    if (MNN::OpenCLSymbolsOperator::createOpenCLSymbolsOperatorSingleInstance() == nullptr) {
        if (error != nullptr) *error = "OpenCL symbols operator is unavailable";
        return false;
    }
    MNN::OpenCLSymbols* symbols = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr();
    if (symbols == nullptr) {
        if (error != nullptr) *error = "OpenCL symbols are unavailable";
        return false;
    }
    if (symbols->isError()) {
        if (error != nullptr) *error = "OpenCL symbol loading failed";
        return false;
    }
    if (symbols->clGetPlatformIDs == nullptr) {
        if (error != nullptr) *error = "OpenCL clGetPlatformIDs symbol is missing";
        return false;
    }
#endif
    BackendConfig config;
    config.precision = BackendConfig::Precision_Normal;
    Backend::Info info;
    info.type = MNN_FORWARD_OPENCL;
    info.numThread = 4;
    info.user = &config;
    output->runtime.reset(new MNN::OpenCL::CLRuntime(info));
    if (output->runtime == nullptr || output->runtime->isCLRuntimeError()) {
        if (error != nullptr) *error = "OpenCL runtime initialization failed";
        return false;
    }
    BackendConfig backendConfig;
    backendConfig.precision = BackendConfig::Precision_Normal;
    backendConfig.memory = BackendConfig::Memory_Normal;
    output->backend.reset(output->runtime->onCreate(&backendConfig, nullptr));
    if (output->backend == nullptr) {
        if (error != nullptr) *error = "OpenCL backend initialization failed";
        return false;
    }
    output->opencl = static_cast<MNN::OpenCL::OpenCLBackend*>(output->backend.get())->getOpenCLRuntime();
    if (output->opencl == nullptr) {
        if (error != nullptr) *error = "OpenCL runtime handle is unavailable";
        return false;
    }
    std::vector<cl::Device> devices = output->opencl->context().getInfo<CL_CONTEXT_DEVICES>();
    if (devices.empty()) {
        if (error != nullptr) *error = "OpenCL context has no device";
        return false;
    }
    output->device = devices[0];
    output->extensions = output->device.getInfo<CL_DEVICE_EXTENSIONS>();
    output->imageSupport = output->device.getInfo<CL_DEVICE_IMAGE_SUPPORT>() != 0;
    return true;
}

static bool buildProgram(const OpenCLPmuRuntime& runtime, const std::vector<const char*>& sources,
                         cl::Program* program, std::string* error) {
    cl_int status = CL_SUCCESS;
    cl::Program::Sources sourceStrings;
    for (const char* source : sources) sourceStrings.emplace_back(source);
    *program = cl::Program(runtime.opencl->context(), sourceStrings, &status);
    if (status != CL_SUCCESS) {
        if (error != nullptr) *error = "OpenCL program creation failed";
        return false;
    }
    status = program->build(std::vector<cl::Device>(1, runtime.device));
    if (status != CL_SUCCESS) {
        std::string log;
        program->getBuildInfo(runtime.device, CL_PROGRAM_BUILD_LOG, &log);
        if (error != nullptr) *error = log.empty() ? "OpenCL program build failed" : log;
        return false;
    }
    return true;
}

static const char* imageKernelSource(const char* kernel) {
    if (kernel == nullptr) return nullptr;
    if (std::strcmp(kernel, "pmu_image_read") == 0) return kOpenCLPmuImageReadKernel;
    if (std::strcmp(kernel, "pmu_texture_read_nearest") == 0) return kOpenCLPmuTextureNearestKernel;
    if (std::strcmp(kernel, "pmu_texture_read_linear") == 0) return kOpenCLPmuTextureLinearKernel;
    if (std::strcmp(kernel, "pmu_image_write") == 0) return kOpenCLPmuImageWriteKernel;
    if (std::strcmp(kernel, "pmu_image_stride_x") == 0) return kOpenCLPmuImageStrideXKernel;
    if (std::strcmp(kernel, "pmu_image_stride_y") == 0) return kOpenCLPmuImageStrideYKernel;
    return nullptr;
}

static size_t caseBytes(const OpenCLPmuCaseInfo& info, const Options& options, uint64_t cacheSize) {
    if (options.openclPmuSize != 0) return options.openclPmuSize;
    if (std::strcmp(info.name, "buffer_reuse_small") == 0) return 32 * 1024;
    if (std::strcmp(info.name, "buffer_reuse_l2") == 0) return std::max<size_t>(512 * 1024, static_cast<size_t>(cacheSize) * 2);
    if (std::strcmp(info.name, "buffer_stream_large") == 0) return 16 * 1024 * 1024;
    return 1024 * 1024;
}

struct PreparedOpenCLPmuCase {
    PreparedOpenCLPmuCase() = default;
    cl::Kernel kernel;
    cl::Buffer source;
    cl::Buffer output;
    cl::Buffer sink;
    cl::Buffer halfSource;
    cl::Buffer halfOutput;
    cl::Buffer weights;
    cl_mem image = nullptr;
    cl_sampler sampler = nullptr;
    cl::NDRange globalRange;
    cl::NDRange localRange;
    OpenCLPmuGeometry geometry;
    uint64_t workingSetBytes = 0;
    uint64_t strideBytes = 0;
    uint32_t vectorWidth = 0;
    uint32_t counterCount = 0;
    uint32_t localBytes = 0;
    uint32_t imageWidth = 0;
    uint32_t imageHeight = 0;
    bool useLocalRange = false;
    uint32_t controlRepeats = 1;

    ~PreparedOpenCLPmuCase() {
        if (sampler != nullptr) {
            clReleaseSampler(sampler);
            sampler = nullptr;
        }
        if (image != nullptr) {
            clReleaseMemObject(image);
            image = nullptr;
        }
    }

    PreparedOpenCLPmuCase(const PreparedOpenCLPmuCase&) = delete;
    PreparedOpenCLPmuCase& operator=(const PreparedOpenCLPmuCase&) = delete;
};

static bool prepareCase(const OpenCLPmuCaseInfo& info, const Options& options, OpenCLPmuRuntime* runtime,
                        cl::Program& program, cl::Program* fp16Program, cl::Program* imageProgram,
                        std::unique_ptr<PreparedOpenCLPmuCase>* prepared, std::string* error) {
    if (prepared == nullptr) return false;
    prepared->reset();
    MNN::OpenCLRuntime* opencl = runtime->opencl;
    cl::CommandQueue& queue = opencl->commandQueue();
    const size_t bytes = std::min<size_t>(caseBytes(info, options, opencl->deviceGlobalMemeryCacheSize()),
                                          static_cast<size_t>(opencl->maxAllocSize()));
    const size_t count = std::max<size_t>(1024, bytes / sizeof(float));
    const uint32_t iterations = options.openclPmuIterations > 0 ? static_cast<uint32_t>(options.openclPmuIterations)
                                                                  : info.defaultIterations;
    cl_int status = CL_SUCCESS;
    const bool imageRead = std::strcmp(info.kernel, "pmu_image_read") == 0 ||
                           std::strcmp(info.kernel, "pmu_texture_read_nearest") == 0 ||
                           std::strcmp(info.kernel, "pmu_texture_read_linear") == 0 ||
                           std::strcmp(info.kernel, "pmu_image_stride_x") == 0 ||
                           std::strcmp(info.kernel, "pmu_image_stride_y") == 0;
    const bool imageKernel = info.path == OpenCLPmuMemoryPath::Image || info.path == OpenCLPmuMemoryPath::Texture;
    size_t width = 256;
    size_t height = 256;
    if (imageKernel && options.openclPmuSize != 0) {
        const size_t pixels = std::max<size_t>(1, bytes / (4 * sizeof(float)));
        width = std::max<size_t>(1, std::min<size_t>(1024, static_cast<size_t>(std::sqrt(static_cast<double>(pixels)))));
        height = (pixels + width - 1) / width;
    }
    const size_t outputCount = imageRead ? width * height * 4 : count;
    std::vector<float> host(std::max(count, outputCount), 1.0f);
    std::vector<float> imageHost(outputCount, 1.0f);
    std::unique_ptr<PreparedOpenCLPmuCase> result(new PreparedOpenCLPmuCase);
    result->controlRepeats = iterations;
    result->source = cl::Buffer(opencl->context(), CL_MEM_READ_ONLY, count * sizeof(float), nullptr, &status);
    result->output = cl::Buffer(opencl->context(), CL_MEM_READ_WRITE, outputCount * sizeof(float), nullptr, &status);
    result->sink = cl::Buffer(opencl->context(), CL_MEM_READ_WRITE, sizeof(uint32_t), nullptr, &status);
    if (status != CL_SUCCESS) {
        if (error != nullptr) *error = "OpenCL buffer allocation failed";
        return false;
    }
    status = queue.enqueueWriteBuffer(result->source, CL_TRUE, 0, count * sizeof(float), host.data());
    if (status == CL_SUCCESS) {
        status = queue.enqueueWriteBuffer(result->output, CL_TRUE, 0, outputCount * sizeof(float), host.data());
    }
    uint32_t zero = 0;
    if (status == CL_SUCCESS) {
        status = queue.enqueueWriteBuffer(result->sink, CL_TRUE, 0, sizeof(zero), &zero);
    }
    if (status != CL_SUCCESS) {
        if (error != nullptr) *error = "OpenCL input initialization failed";
        return false;
    }

    if (imageKernel && imageProgram == nullptr) {
        if (error != nullptr) *error = "OpenCL image program is unavailable";
        return false;
    }
    const bool fp16 = std::strcmp(info.kernel, "pmu_fp16_compute") == 0 ||
                      std::strcmp(info.kernel, "pmu_fp16_throughput") == 0;
    if (fp16) {
        if (fp16Program == nullptr) return false;
        result->kernel = cl::Kernel(*fp16Program, info.kernel, &status);
    } else {
        cl::Program* kernelProgram = imageKernel && imageProgram != nullptr ? imageProgram : &program;
        result->kernel = cl::Kernel(*kernelProgram, info.kernel, &status);
    }
    if (status != CL_SUCCESS) {
        if (error != nullptr) *error = "OpenCL kernel creation failed";
        return false;
    }

    const size_t requestedLocalSize = options.openclPmuLocalSize == 0 ? 64 : options.openclPmuLocalSize;
    if (requestedLocalSize != 32 && requestedLocalSize != 64 && requestedLocalSize != 128 &&
        requestedLocalSize != 256) {
        if (error != nullptr) *error = "OpenCL PMU local size must be one of 32, 64, 128, or 256";
        return false;
    }
    const bool vectorKernel = std::strcmp(info.kernel, "pmu_buffer_vec4") == 0;
    const bool pointerChaseKernel = std::strcmp(info.kernel, "pmu_buffer_pchase") == 0;
    const size_t dispatchCount = vectorKernel ? std::max<size_t>(1, count / 4) : (pointerChaseKernel ? 1 : count);
    OpenCLPmuGeometry geometry = makeOpenclPmuGeometry(info, dispatchCount, requestedLocalSize);
    if (imageKernel) {
        geometry.global[0] = width;
        geometry.global[1] = height;
    }
    if (geometry.usesLocalSize &&
        requestedLocalSize > runtime->device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()) {
        if (error != nullptr) *error = "OpenCL PMU local size exceeds CL_DEVICE_MAX_WORK_GROUP_SIZE";
        return false;
    }
    if (std::strcmp(info.kernel, "pmu_buffer_reuse") == 0) {
        const uint32_t active = static_cast<uint32_t>(std::min<size_t>(count, std::strcmp(info.name, "buffer_reuse_small") == 0 ? 8192 : count));
        status |= result->kernel.setArg(0, result->source);
        status |= result->kernel.setArg(1, result->output);
        status |= result->kernel.setArg(2, static_cast<uint32_t>(count));
        status |= result->kernel.setArg(3, active);
        status |= result->kernel.setArg(4, iterations);
    } else if (std::strcmp(info.kernel, "pmu_buffer_stride") == 0) {
        size_t active = 1;
        while ((active << 1) <= count) active <<= 1;
        const uint32_t stride = 16;
        status |= result->kernel.setArg(0, result->source); status |= result->kernel.setArg(1, result->output);
        status |= result->kernel.setArg(2, static_cast<uint32_t>(count));
        status |= result->kernel.setArg(3, static_cast<uint32_t>(active - 1));
        status |= result->kernel.setArg(4, stride);
        status |= result->kernel.setArg(5, iterations);
        result->workingSetBytes = static_cast<uint64_t>(active * sizeof(float));
        result->strideBytes = static_cast<uint64_t>(stride * sizeof(float));
    } else if (std::strcmp(info.kernel, "pmu_buffer_pchase") == 0) {
        std::vector<uint32_t> next(count);
        for (size_t i = 0; i < count; ++i) next[i] = static_cast<uint32_t>((i * 2654435761u + 1013904223u) % count);
        status = queue.enqueueWriteBuffer(result->source, CL_TRUE, 0, next.size() * sizeof(uint32_t), next.data());
        status |= result->kernel.setArg(0, result->source); status |= result->kernel.setArg(1, result->output);
        status |= result->kernel.setArg(2, static_cast<uint32_t>(count)); status |= result->kernel.setArg(3, iterations);
        result->workingSetBytes = static_cast<uint64_t>(count * sizeof(uint32_t));
        result->strideBytes = sizeof(uint32_t);
    } else if (std::strcmp(info.kernel, "pmu_buffer_vec4") == 0) {
        status |= result->kernel.setArg(0, result->source); status |= result->kernel.setArg(1, result->output);
        status |= result->kernel.setArg(2, static_cast<uint32_t>(dispatchCount)); status |= result->kernel.setArg(3, iterations);
        result->workingSetBytes = static_cast<uint64_t>(dispatchCount * 4 * sizeof(float));
        result->vectorWidth = 4;
    } else if (std::strcmp(info.kernel, "pmu_buffer_stream") == 0) {
        status |= result->kernel.setArg(0, result->source); status |= result->kernel.setArg(1, result->output);
        status |= result->kernel.setArg(2, static_cast<uint32_t>(count)); status |= result->kernel.setArg(3, iterations);
    } else if (std::strcmp(info.kernel, "pmu_buffer_write") == 0) {
        status |= result->kernel.setArg(0, result->output); status |= result->kernel.setArg(1, static_cast<uint32_t>(count)); status |= result->kernel.setArg(2, iterations);
    } else if (std::strcmp(info.kernel, "pmu_int_compute") == 0 || std::strcmp(info.kernel, "pmu_fp32_compute") == 0 ||
               std::strcmp(info.kernel, "pmu_fp32_throughput") == 0) {
        status |= result->kernel.setArg(0, result->source); status |= result->kernel.setArg(1, result->output);
        status |= result->kernel.setArg(2, static_cast<uint32_t>(count)); status |= result->kernel.setArg(3, iterations);
    } else if (fp16) {
        result->halfSource = cl::Buffer(opencl->context(), CL_MEM_READ_ONLY, count * sizeof(uint16_t), nullptr, &status);
        result->halfOutput = cl::Buffer(opencl->context(), CL_MEM_READ_WRITE, count * sizeof(uint16_t), nullptr, &status);
        std::vector<uint16_t> halfHost(count, 0x3c00);
        if (status == CL_SUCCESS) {
            status = queue.enqueueWriteBuffer(result->halfSource, CL_TRUE, 0, halfHost.size() * sizeof(uint16_t), halfHost.data());
        }
        status |= result->kernel.setArg(0, result->halfSource); status |= result->kernel.setArg(1, result->halfOutput);
        status |= result->kernel.setArg(2, static_cast<uint32_t>(count)); status |= result->kernel.setArg(3, iterations);
    } else if (std::strcmp(info.kernel, "pmu_constant_compute") == 0) {
        result->weights = cl::Buffer(opencl->context(), CL_MEM_READ_ONLY, 16 * sizeof(float), nullptr, &status);
        if (status == CL_SUCCESS) {
            status = queue.enqueueWriteBuffer(result->weights, CL_TRUE, 0, 16 * sizeof(float), host.data());
        }
        status |= result->kernel.setArg(0, result->output); status |= result->kernel.setArg(1, result->weights);
        status |= result->kernel.setArg(2, static_cast<uint32_t>(count)); status |= result->kernel.setArg(3, iterations);
    } else if (std::strcmp(info.kernel, "pmu_constant_bandwidth") == 0) {
        size_t tableCount = 16;
        while ((tableCount << 1) <= count && tableCount < 4096) tableCount <<= 1;
        result->weights = cl::Buffer(opencl->context(), CL_MEM_READ_ONLY, tableCount * sizeof(float), nullptr, &status);
        std::vector<float> table(tableCount, 1.0f);
        if (status == CL_SUCCESS) {
            status = queue.enqueueWriteBuffer(result->weights, CL_TRUE, 0, table.size() * sizeof(float), table.data());
        }
        status |= result->kernel.setArg(0, result->output); status |= result->kernel.setArg(1, result->weights);
        status |= result->kernel.setArg(2, static_cast<uint32_t>(count));
        status |= result->kernel.setArg(3, static_cast<uint32_t>(tableCount - 1)); status |= result->kernel.setArg(4, iterations);
        result->workingSetBytes = static_cast<uint64_t>(tableCount * sizeof(float));
    } else if (std::strcmp(info.kernel, "pmu_local_memory") == 0 ||
               std::strcmp(info.kernel, "pmu_local_bandwidth") == 0 ||
               std::strcmp(info.kernel, "pmu_local_barrier") == 0) {
        status |= result->kernel.setArg(0, result->output); status |= result->kernel.setArg(1, static_cast<uint32_t>(count));
        status |= result->kernel.setArg(2, iterations);
        status |= result->kernel.setArg(3, geometry.local[0] * sizeof(float), nullptr);
        result->localBytes = static_cast<uint32_t>(geometry.local[0] * sizeof(float));
    } else if (std::strcmp(info.kernel, "pmu_atomic") == 0) {
        status |= result->kernel.setArg(0, result->sink); status |= result->kernel.setArg(1, static_cast<uint32_t>(1)); status |= result->kernel.setArg(2, iterations);
        result->counterCount = 1;
    } else if (std::strcmp(info.kernel, "pmu_atomic_contended") == 0) {
        status |= result->kernel.setArg(0, result->sink); status |= result->kernel.setArg(1, iterations);
        result->counterCount = 1;
    } else if (std::strcmp(info.kernel, "pmu_atomic_distributed") == 0) {
        const uint32_t counterCount = static_cast<uint32_t>(count);
        status |= result->kernel.setArg(0, result->output); status |= result->kernel.setArg(1, counterCount);
        status |= result->kernel.setArg(2, iterations);
        result->counterCount = counterCount;
    } else if (std::strcmp(info.kernel, "pmu_image_read") == 0 ||
               std::strcmp(info.kernel, "pmu_texture_read_nearest") == 0 ||
               std::strcmp(info.kernel, "pmu_texture_read_linear") == 0 ||
               std::strcmp(info.kernel, "pmu_image_write") == 0 ||
               std::strcmp(info.kernel, "pmu_image_stride_x") == 0 ||
               std::strcmp(info.kernel, "pmu_image_stride_y") == 0) {
        if (!runtime->imageSupport) {
            if (error != nullptr) *error = "OpenCL image support is unavailable";
            return false;
        }
        cl::ImageFormat format(CL_RGBA, CL_FLOAT);
        const bool imageWrite = std::strcmp(info.kernel, "pmu_image_write") == 0;
        const cl_mem_flags imageFlags = imageWrite ? CL_MEM_WRITE_ONLY : CL_MEM_READ_ONLY;
        result->imageWidth = static_cast<uint32_t>(width);
        result->imageHeight = static_cast<uint32_t>(height);
        result->workingSetBytes = static_cast<uint64_t>(width * height * 4 * sizeof(float));
        if (std::strcmp(info.kernel, "pmu_image_stride_x") == 0 || std::strcmp(info.kernel, "pmu_image_stride_y") == 0) {
            result->strideBytes = static_cast<uint64_t>(4 * 4 * sizeof(float));
        }
        result->image = clCreateImage2D(opencl->context()(), imageFlags, &format, width, height, 0, nullptr, &status);
        if (status != CL_SUCCESS) {
            if (error != nullptr) *error = "OpenCL image allocation failed";
            return false;
        }
        if (!imageWrite) {
            size_t origin[3] = {0, 0, 0};
            size_t region[3] = {width, height, 1};
            status = clEnqueueWriteImage(queue(), result->image, CL_TRUE, origin, region, 0, 0,
                                         imageHost.data(), 0, nullptr, nullptr);
        }
        if (imageWrite) {
            status |= result->kernel.setArg(0, sizeof(result->image), &result->image); status |= result->kernel.setArg(1, static_cast<uint32_t>(width));
            status |= result->kernel.setArg(2, static_cast<uint32_t>(height)); status |= result->kernel.setArg(3, iterations);
        } else if (std::strcmp(info.kernel, "pmu_image_read") == 0 ||
                   std::strcmp(info.kernel, "pmu_image_stride_x") == 0 ||
                   std::strcmp(info.kernel, "pmu_image_stride_y") == 0) {
            result->sampler = clCreateSampler(opencl->context()(), CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST, &status);
            status |= result->kernel.setArg(0, sizeof(result->image), &result->image); status |= result->kernel.setArg(1, result->output);
            status |= result->kernel.setArg(2, sizeof(result->sampler), &result->sampler);
            status |= result->kernel.setArg(3, static_cast<uint32_t>(width)); status |= result->kernel.setArg(4, static_cast<uint32_t>(height));
            if (std::strcmp(info.kernel, "pmu_image_stride_x") == 0 || std::strcmp(info.kernel, "pmu_image_stride_y") == 0) {
                status |= result->kernel.setArg(5, static_cast<uint32_t>(4)); status |= result->kernel.setArg(6, iterations);
            } else {
                status |= result->kernel.setArg(5, iterations);
            }
        } else {
            status |= result->kernel.setArg(0, sizeof(result->image), &result->image); status |= result->kernel.setArg(1, result->output);
            status |= result->kernel.setArg(2, static_cast<uint32_t>(width)); status |= result->kernel.setArg(3, static_cast<uint32_t>(height));
            status |= result->kernel.setArg(4, iterations);
        }
    }
    if (status != CL_SUCCESS) {
        if (error != nullptr) *error = "OpenCL kernel argument setup failed";
        return false;
    }
    result->geometry = geometry;
    result->useLocalRange = geometry.usesLocalSize;
    result->localRange = geometry.usesLocalSize ? cl::NDRange(geometry.local[0]) : cl::NullRange;
    result->globalRange = geometry.usesLocalSize ? cl::NDRange(geometry.global[0]) : cl::NDRange(width, height);
    *prepared = std::move(result);
    return true;
}

static bool enqueuePreparedCase(OpenCLPmuRuntime* runtime, PreparedOpenCLPmuCase* prepared, std::string* error) {
    if (runtime == nullptr || prepared == nullptr) {
        if (error != nullptr) *error = "Prepared OpenCL case is unavailable";
        return false;
    }
    cl::CommandQueue& queue = runtime->opencl->commandQueue();
    const cl::NDRange localRange = prepared->useLocalRange ? prepared->localRange : cl::NullRange;
    const cl_int status = queue.enqueueNDRangeKernel(prepared->kernel, cl::NullRange, prepared->globalRange, localRange);
    if (status != CL_SUCCESS) {
        if (error != nullptr) *error = "OpenCL benchmark kernel enqueue failed";
        return false;
    }
    if (queue.finish() != CL_SUCCESS) {
        if (error != nullptr) *error = "OpenCL queue finish failed";
        return false;
    }
    return true;
}

static bool enqueuePreparedControl(OpenCLPmuRuntime* runtime, PreparedOpenCLPmuCase* prepared,
                                   cl::Kernel& controlKernel, cl::Buffer& sink, std::string* error) {
    if (runtime == nullptr || prepared == nullptr) {
        if (error != nullptr) *error = "Prepared OpenCL control is unavailable";
        return false;
    }
    cl::CommandQueue& queue = runtime->opencl->commandQueue();
    cl_int status = controlKernel.setArg(0, sink);
    status |= controlKernel.setArg(1, prepared->controlRepeats);
    if (status != CL_SUCCESS) {
        if (error != nullptr) *error = "OpenCL control kernel argument setup failed";
        return false;
    }
    const cl::NDRange localRange = prepared->useLocalRange ? prepared->localRange : cl::NullRange;
    status = queue.enqueueNDRangeKernel(controlKernel, cl::NullRange, prepared->globalRange, localRange);
    if (status != CL_SUCCESS) {
        if (error != nullptr) *error = "OpenCL control kernel enqueue failed";
        return false;
    }
    if (queue.finish() != CL_SUCCESS) {
        if (error != nullptr) *error = "OpenCL control queue finish failed";
        return false;
    }
    return true;
}

#if defined(MNN_REPLAY_HAS_PERFCOUNTER)

static PmuInterval samplePmuInterval(const PmuBatch& batch, OpenCLPmuRuntime* runtime,
                                     const Options& options, PreparedOpenCLPmuCase* prepared,
                                     cl::Kernel& controlKernel,
                                     cl::Buffer& controlSink, bool control) {
    PmuInterval interval;
    if (batch.names.empty()) {
        interval.error = "No supported counters in this PMU batch";
        return interval;
    }
    std::vector<MNN::PerfCounter::CounterSpec> specs(batch.names.size());
    for (size_t i = 0; i < batch.names.size(); ++i) specs[i].name = batch.names[i].c_str();
    MNN::PerfCounter::DeviceInfo ignoredDevice;
    const char* sessionError = nullptr;
    std::unique_ptr<MNN::PerfCounter::Session> session(
        MNN::PerfCounter::Session::create(specs.data(), specs.size(), &ignoredDevice, &sessionError));
    if (session == nullptr) {
        interval.error = sessionError == nullptr ? "GPU PMU session creation failed" : sessionError;
        return interval;
    }
    if (!session->start()) {
        interval.error = session->error();
        return interval;
    }
    interval.startNs = monotonicNs();
    std::string workloadError;
    bool ok = true;
    if (control) {
        ok = enqueuePreparedControl(runtime, prepared, controlKernel, controlSink, &workloadError);
        interval.dispatches = ok ? 1 : 0;
    } else {
        const int workloadRuns = std::max(1, options.openclPmuWorkloadRuns);
        for (int run = 0; run < workloadRuns && ok; ++run) {
            ok = enqueuePreparedCase(runtime, prepared, &workloadError);
            if (ok) ++interval.dispatches;
        }
    }
    interval.endNs = monotonicNs();
    interval.values.resize(batch.names.size());
    if (!ok || !session->stop(interval.values.data(), interval.values.size())) {
        interval.error = ok ? session->error() : workloadError;
        return interval;
    }
    interval.available = true;
    return interval;
}

#endif

} // namespace

#endif

bool runOpenCLPmuBenchmark(const Options& options) {
    rapidjson::Document document;
    document.SetObject();
    auto& allocator = document.GetAllocator();
    addString(document, "format", "mnn-opencl-pmu-benchmark", allocator);
    document.AddMember("version", 1, allocator);
    addString(document, "status", "unavailable", allocator);
    rapidjson::Value cases(rapidjson::kArrayType);

#if !defined(MNN_REPLAY_HAS_OPENCL)
    rapidjson::Value errors(rapidjson::kArrayType);
    rapidjson::Value message("OpenCL was not compiled into replay_benchmark", allocator);
    errors.PushBack(message, allocator);
    document.AddMember("cases", cases, allocator);
    document.AddMember("errors", errors, allocator);
#else
    OpenCLPmuRuntime runtime;
    std::string error;
    if (!initializeOpenCL(&runtime, &error)) {
        rapidjson::Value errors(rapidjson::kArrayType);
        errors.PushBack(rapidjson::Value(error.c_str(), allocator), allocator);
        document.AddMember("cases", cases, allocator);
        document.AddMember("errors", errors, allocator);
    } else {
        addString(document, "device", runtime.device.getInfo<CL_DEVICE_NAME>(), allocator);
        addString(document, "extensions", runtime.extensions, allocator);
        cl::Program program;
        const std::vector<const char*> coreSources = {kOpenCLPmuBufferKernels, kOpenCLPmuComputeKernels};
        if (!buildProgram(runtime, coreSources, &program, &error)) {
            rapidjson::Value errors(rapidjson::kArrayType);
            errors.PushBack(rapidjson::Value(error.c_str(), allocator), allocator);
            document.AddMember("cases", cases, allocator);
            document.AddMember("errors", errors, allocator);
        } else {
            cl::Program fp16Program;
            bool fp16Ready = false;
            if (hasExtension(runtime.extensions, "cl_khr_fp16")) {
                std::vector<const char*> fp16Sources = {kOpenCLPmuFp16Kernels, kOpenCLPmuFp16ThroughputKernels};
                fp16Ready = buildProgram(runtime, fp16Sources, &fp16Program, nullptr);
            }
#if defined(MNN_REPLAY_HAS_PERFCOUNTER)
            MNN::PerfCounter::DeviceInfo perfDevice;
            std::string perfError;
            bool perfReady = discoverPmuDevice(&perfDevice, &perfError);
            std::vector<std::string> requestedCounters;
            std::vector<std::string> unsupportedCounters;
            std::vector<PmuBatch> pmuBatches;
            if (perfReady) {
                requestedCounters = (options.perfCounterEvents.empty() || options.perfCounterEvents == "auto")
                                        ? defaultPmuCounterNames(perfDevice)
                                        : parsePmuCounterList(options.perfCounterEvents);
                pmuBatches = makePmuBatches(requestedCounters, perfDevice, &unsupportedCounters);
                if (pmuBatches.empty()) {
                    perfReady = false;
                    perfError = "No requested performance counter is supported by the detected GPU";
                }
                addString(document, "pmu_vendor",
                          perfDevice.vendor == MNN::PerfCounter::GpuVendor::Adreno
                              ? "Adreno"
                              : (perfDevice.vendor == MNN::PerfCounter::GpuVendor::Mali ? "Mali" : "Unknown"),
                          allocator);
                document.AddMember("pmu_product_id", static_cast<uint64_t>(perfDevice.productId), allocator);
                addString(document, "pmu_product_name", perfDevice.productName == nullptr ? "" : perfDevice.productName,
                          allocator);
                addString(document, "pmu_driver", perfDevice.driverName == nullptr ? "" : perfDevice.driverName,
                          allocator);
            }
            addString(document, "pmu_status", perfReady ? "available" : "unavailable", allocator);
            addString(document, "pmu_error", perfError, allocator);
#else
            const bool perfReady = false;
#endif
            cl_int controlStatus = CL_SUCCESS;
            cl::Kernel controlKernel(program, "pmu_control", &controlStatus);
            cl::Buffer controlSink(runtime.opencl->context(), CL_MEM_READ_WRITE,
                                    sizeof(uint32_t), nullptr, &controlStatus);
            if (controlStatus == CL_SUCCESS) {
                uint32_t zero = 0;
                controlStatus = runtime.opencl->commandQueue().enqueueWriteBuffer(
                    controlSink, CL_TRUE, 0, sizeof(zero), &zero);
            }
            const bool controlReady = controlStatus == CL_SUCCESS;
            const std::vector<std::string> selected = parseOpenclPmuCaseList(options.openclPmuCase);
            for (const auto& selectedName : selected) {
                const OpenCLPmuCaseInfo* info = findOpenclPmuCase(selectedName.c_str());
                if (info == nullptr) continue;
                rapidjson::Value item(rapidjson::kObjectType);
                addString(item, "name", info->name, allocator);
                addString(item, "concept", info->concept, allocator);
                addString(item, "path", pathName(info->path), allocator);
                rapidjson::Value status("ok", allocator);
                std::string caseError;
                bool workloadReady = true;
                cl::Program imageProgram;
                std::unique_ptr<PreparedOpenCLPmuCase> preparedCase;
                bool imageProgramReady = true;
                if (info->path == OpenCLPmuMemoryPath::Image || info->path == OpenCLPmuMemoryPath::Texture) {
                    const char* imageSource = imageKernelSource(info->kernel);
                    imageProgramReady = imageSource != nullptr &&
                                         buildProgram(runtime, std::vector<const char*>(1, imageSource),
                                                      &imageProgram, &caseError);
                    if (!imageProgramReady) {
                        status.SetString("error", allocator);
                        workloadReady = false;
                    }
                }
                if (info->requiredExtension != nullptr && !hasExtension(runtime.extensions, info->requiredExtension)) {
                    status.SetString("skipped", allocator);
                    caseError = std::string("Missing extension: ") + info->requiredExtension;
                    workloadReady = false;
                } else if ((std::strcmp(info->kernel, "pmu_fp16_compute") == 0 ||
                            std::strcmp(info->kernel, "pmu_fp16_throughput") == 0) && !fp16Ready) {
                    status.SetString("skipped", allocator);
                    caseError = "FP16 kernel is unavailable";
                    workloadReady = false;
                }
                if (workloadReady &&
                    !prepareCase(*info, options, &runtime, program, fp16Ready ? &fp16Program : nullptr,
                                 imageProgramReady ? &imageProgram : nullptr, &preparedCase, &caseError)) {
                    status.SetString("error", allocator);
                    workloadReady = false;
                }
                rapidjson::Value counterResults(rapidjson::kArrayType);
                uint64_t controlDurationNs = 0;
                uint64_t workloadDurationNs = 0;
                uint32_t sampledWorkloadDispatches = 0;
                uint32_t dispatches = 0;
                bool pmuMeasured = false;
#if defined(MNN_REPLAY_HAS_PERFCOUNTER)
                if (workloadReady && perfReady && controlReady) {
                    for (int warmup = 0; warmup < std::max(0, options.openclPmuWarmupRuns); ++warmup) {
                        if (!enqueuePreparedCase(&runtime, preparedCase.get(), &caseError)) {
                            status.SetString("error", allocator);
                            workloadReady = false;
                            break;
                        }
                    }
                }
                if (workloadReady && perfReady && controlReady) {
                    for (const auto& batch : pmuBatches) {
                        const PmuInterval control = samplePmuInterval(batch, &runtime, options, preparedCase.get(),
                                                                      controlKernel, controlSink, true);
                        const PmuInterval workload = samplePmuInterval(batch, &runtime, options, preparedCase.get(),
                                                                       controlKernel, controlSink, false);
                        controlDurationNs += control.endNs >= control.startNs ? control.endNs - control.startNs : 0;
                        workloadDurationNs += workload.endNs >= workload.startNs ? workload.endNs - workload.startNs : 0;
                        sampledWorkloadDispatches += workload.dispatches;
                        dispatches += workload.dispatches;
                        if (!control.available || !workload.available || control.values.size() != batch.names.size() ||
                            workload.values.size() != batch.names.size()) {
                            if (caseError.empty()) {
                                caseError = !workload.error.empty() ? workload.error : control.error;
                            }
                            continue;
                        }
                        pmuMeasured = true;
                        for (size_t i = 0; i < batch.names.size(); ++i) {
                            const OpenCLPmuSignalResult signal = classifyPmuSignal(
                                control.values[i].value, workload.values[i].value, info->expectedIncrease, 0);
                            rapidjson::Value result(rapidjson::kObjectType);
                            addString(result, "name", batch.names[i], allocator);
                            result.AddMember("control_delta", static_cast<uint64_t>(control.values[i].value), allocator);
                            result.AddMember("workload_delta", static_cast<uint64_t>(workload.values[i].value), allocator);
                            result.AddMember("readable", signal.readable, allocator);
                            result.AddMember("responsive", signal.responsive, allocator);
                            result.AddMember("discriminative", signal.discriminative, allocator);
                            result.AddMember("valid", signal.valid, allocator);
                            counterResults.PushBack(result, allocator);
                        }
                    }
                }
#endif
                if (workloadReady && !pmuMeasured) {
                    if (!enqueuePreparedCase(&runtime, preparedCase.get(), &caseError)) {
                        status.SetString("error", allocator);
                        workloadReady = false;
                    } else {
                        dispatches = 1;
                    }
                }
                if (workloadReady && perfReady && !pmuMeasured && caseError.empty()) {
                    caseError = "PMU did not produce a complete control/workload sample";
                }
                item.AddMember("status", status, allocator);
                if (preparedCase != nullptr) {
                    addGeometry(item, preparedCase->geometry, allocator);
                    if (preparedCase->workingSetBytes != 0) {
                        item.AddMember("working_set_bytes", preparedCase->workingSetBytes, allocator);
                    }
                    if (preparedCase->strideBytes != 0) {
                        item.AddMember("stride_bytes", preparedCase->strideBytes, allocator);
                    }
                    if (preparedCase->vectorWidth != 0) {
                        item.AddMember("vector_width", preparedCase->vectorWidth, allocator);
                    }
                    if (preparedCase->counterCount != 0) {
                        item.AddMember("counter_count", preparedCase->counterCount, allocator);
                    }
                    if (preparedCase->localBytes != 0) {
                        item.AddMember("local_bytes", preparedCase->localBytes, allocator);
                    }
                    if (preparedCase->imageWidth != 0) {
                        item.AddMember("image_width", preparedCase->imageWidth, allocator);
                        item.AddMember("image_height", preparedCase->imageHeight, allocator);
                    }
                }
                item.AddMember("iterations", options.openclPmuIterations > 0 ? options.openclPmuIterations : static_cast<int>(info->defaultIterations), allocator);
                item.AddMember("workload_runs", std::max(1, options.openclPmuWorkloadRuns), allocator);
                item.AddMember("warmup_runs", std::max(0, options.openclPmuWarmupRuns), allocator);
                item.AddMember("prepare_count", preparedCase == nullptr ? 0 : 1, allocator);
                item.AddMember("sampled_workload_dispatches", static_cast<uint32_t>(sampledWorkloadDispatches), allocator);
                item.AddMember("dispatches", static_cast<uint32_t>(dispatches), allocator);
                item.AddMember("pmu_available", perfReady, allocator);
                item.AddMember("pmu_measured", pmuMeasured, allocator);
                item.AddMember("control_duration_ns", static_cast<uint64_t>(controlDurationNs), allocator);
                item.AddMember("workload_duration_ns", static_cast<uint64_t>(workloadDurationNs), allocator);
                item.AddMember("counters", counterResults, allocator);
                rapidjson::Value unsupported(rapidjson::kArrayType);
#if defined(MNN_REPLAY_HAS_PERFCOUNTER)
                for (const auto& counter : unsupportedCounters) {
                    rapidjson::Value value(counter.c_str(), allocator);
                    unsupported.PushBack(value, allocator);
                }
#endif
                item.AddMember("unsupported_counters", unsupported, allocator);
                addString(item, "error", caseError, allocator);
                cases.PushBack(item, allocator);
            }
            document["status"].SetString("ok", allocator);
            document.AddMember("cases", cases, allocator);
            rapidjson::Value errors(rapidjson::kArrayType);
            document.AddMember("errors", errors, allocator);
        }
    }
#endif
    if (options.perfCounterOutput.empty()) {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        document.Accept(writer);
        std::cout << buffer.GetString() << std::endl;
        return true;
    }
    if (!makeParentDirectories(options.perfCounterOutput)) return false;
    FILE* file = std::fopen(options.perfCounterOutput.c_str(), "wb");
    if (file == nullptr) return false;
    char buffer[4096];
    rapidjson::FileWriteStream stream(file, buffer, sizeof(buffer));
    rapidjson::Writer<rapidjson::FileWriteStream> writer(stream);
    const bool ok = document.Accept(writer);
    std::fclose(file);
    return ok;
}

} // namespace Replay
} // namespace MNN
