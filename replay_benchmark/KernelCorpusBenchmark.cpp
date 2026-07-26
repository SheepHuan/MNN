#include "KernelCorpusBenchmark.hpp"
#include "ReplayRecord.hpp"
#include "kernel_corpus_bridge/Bridge.hpp"
#include "kernel_corpus_bridge/mnn/MnnBridge.hpp"
#include "kernel_corpus_bridge/ncnn/NcnnBridge.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include <sys/stat.h>

#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/writer.h"

#if defined(MNN_REPLAY_HAS_OPENCL)
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/runtime/OpenCLWrapper.hpp"
#endif

#if defined(MNN_REPLAY_HAS_VULKAN)
#include "backend/vulkan/component/VulkanInstance.hpp"
#include "backend/vulkan/component/VulkanDevice.hpp"
#include "backend/vulkan/component/VulkanCommandPool.hpp"
#include "backend/vulkan/component/VulkanBuffer.hpp"
#include "backend/vulkan/component/VulkanMemoryPool.hpp"
#include "backend/vulkan/vulkan/vulkan_wrapper.h"
#include <unistd.h>
#endif

#if defined(MNN_REPLAY_HAS_PERFCOUNTER)
#include "MNNPerfCounter.hpp"
#endif

namespace MNN {
namespace Replay {
namespace KernelCorpus {

namespace {

static const char* kAdrenoCounters[] = {
    "cp_busy_cycles", "rbbm_vbif_busy", "hlsq_busy_cycles", "uche_busy_cycles",
    "tp_busy_cycles", "tp_l1_cacheline_requests", "tp_l1_cacheline_misses",
    "sp_busy_cycles", "sp_cs_invocations", "sp_gm_load_instructions",
    "sp_gm_store_instructions", "sp_lm_load_instructions", "sp_working_eu_cs_stage",
};
static const char* kMaliCounters[] = {
    "gpu_active_cycles", "compute_active_cycles", "compute_tasks",
    "l2_any_lookup", "l2_ext_read", "l2_ext_write",
};
static const char* kPortableCounters[] = {
    "gpu_active_cycles", "compute_active_cycles", "compute_tasks",
};

static bool discoverPmuDevice(MNN::PerfCounter::DeviceInfo* device, std::string* error) {
    MNN::PerfCounter::CounterSpec spec;
    spec.name = "gpu_active_cycles";
    const char* sessionError = nullptr;
    std::unique_ptr<MNN::PerfCounter::Session> session(
        MNN::PerfCounter::Session::create(&spec, 1, device, &sessionError));
    if (session == nullptr) {
        if (error) *error = sessionError == nullptr ? "GPU PMU is unavailable" : sessionError;
        return false;
    }
    return true;
}

static std::vector<std::string> selectPmuCounters(const MNN::PerfCounter::DeviceInfo& device) {
    if (device.vendor == MNN::PerfCounter::GpuVendor::Adreno) {
        return std::vector<std::string>(kAdrenoCounters,
            kAdrenoCounters + sizeof(kAdrenoCounters) / sizeof(kAdrenoCounters[0]));
    }
    if (device.vendor == MNN::PerfCounter::GpuVendor::Mali) {
        return std::vector<std::string>(kMaliCounters,
            kMaliCounters + sizeof(kMaliCounters) / sizeof(kMaliCounters[0]));
    }
    return std::vector<std::string>(kPortableCounters,
        kPortableCounters + sizeof(kPortableCounters) / sizeof(kPortableCounters[0]));
}

struct PmuScope {
    bool started = false;
    std::vector<std::string> names;
    std::unique_ptr<MNN::PerfCounter::Session> session;
    std::vector<MNN::PerfCounter::CounterValue> values;
    uint64_t workloadDelta = 0;
    std::string status = "unavailable";
};

static PmuScope beginPmu() {
    PmuScope s;
#if defined(MNN_REPLAY_HAS_PERFCOUNTER)
    MNN::PerfCounter::DeviceInfo device;
    std::string err;
    if (!discoverPmuDevice(&device, &err)) {
        s.status = "unavailable";
        return s;
    }
    s.names = selectPmuCounters(device);
    std::vector<MNN::PerfCounter::CounterSpec> specs;
    for (const auto& name : s.names) {
        MNN::PerfCounter::CounterSpec sp;
        sp.name = name.c_str();
        specs.push_back(sp);
    }
    const char* pmuErr = nullptr;
    MNN::PerfCounter::DeviceInfo sessionDevice;
    s.session.reset(MNN::PerfCounter::Session::create(specs.data(), specs.size(), &sessionDevice, &pmuErr));
    if (s.session && s.session->start()) {
        s.started = true;
        s.status = "started";
    } else {
        s.status = "start_failed";
    }
#endif
    return s;
}

static void endPmu(PmuScope& s) {
#if defined(MNN_REPLAY_HAS_PERFCOUNTER)
    if (!s.started) return;
    s.values.resize(s.names.size());
    if (s.session && s.session->stop(s.values.data(), s.values.size())) {
        s.status = "sampled";
        if (!s.values.empty()) s.workloadDelta = s.values[0].value;
    } else {
        s.status = "stop_failed";
    }
#endif
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
        if (!current.empty() && current != "/") mkdir(current.c_str(), 0755);
    }
    return true;
}

static void addString(rapidjson::Value& object, const char* key, const std::string& value,
                      rapidjson::Document::AllocatorType& allocator) {
    rapidjson::Value k(key, allocator);
    rapidjson::Value v(value.c_str(), allocator);
    object.AddMember(k, v, allocator);
}

static bool writeDoc(const Options& options, rapidjson::Document& document) {
    if (options.perfCounterOutput.empty()) {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        document.Accept(writer);
        std::cout << buffer.GetString() << std::endl;
        return true;
    }
    makeParentDirectories(options.perfCounterOutput);
    FILE* file = std::fopen(options.perfCounterOutput.c_str(), "wb");
    if (!file) return false;
    char buf[4096];
    rapidjson::FileWriteStream stream(file, buf, sizeof(buf));
    rapidjson::Writer<rapidjson::FileWriteStream> writer(stream);
    const bool ok = document.Accept(writer);
    std::fclose(file);
    return ok;
}

static CaseReport buildReport(const AdaptedCase& ac, const std::string& compileStatus,
                              const std::string& error = "") {
    CaseReport r;
    r.framework = ac.framework;
    r.tag = ac.tag;
    r.backend = ac.backend;
    r.opType = ac.opType;
    r.variant = ac.variant;
    r.caseName = ac.caseName;
    r.compileStatus = compileStatus;
    r.dispatchStatus = "not_run";
    r.validationStatus = "not_validated";
    r.pmuStatus = "unavailable";
    r.error = error;
    return r;
}

// ---- Validator dispatch ----

static bool runValidator(const AdaptedCase& ac, const std::vector<float>& output) {
    if (ac.validator == "all_zero_fp32") return validateAllZeroFp32(output);
    if (ac.validator == "exp_fp32") return validateExpFp32(ac.validatorInputA, output);
    if (ac.validator == "matmul_fp32") {
        const int M = ac.m > 0 ? ac.m : 16;
        const int N = ac.n > 0 ? ac.n : 16;
        const int K = ac.k > 0 ? ac.k : 16;
        return validateMatmulFp32(M, N, K, ac.validatorInputA, ac.validatorInputB, output);
    }
    if (ac.validator == "sigmoid_fp32") return validateSigmoidFp32(ac.validatorInputA, output);
    if (ac.validator == "tanh_fp32") return validateTanhFp32(ac.validatorInputA, output);
    if (ac.validator == "permute_identity_fp32") {
        return validatePermuteIdentityFp32(ac.w, ac.h, ac.c, ac.validatorInputA, output);
    }
    return false;
}

} // namespace

#if defined(MNN_REPLAY_HAS_OPENCL)

namespace {

struct OpenCLRuntimeHolder {
    std::unique_ptr<MNN::OpenCL::CLRuntime> clRuntime;
    std::unique_ptr<MNN::Backend> backend;
    MNN::OpenCLRuntime* opencl = nullptr;
    cl::Device device;
};

static bool initOpenCL(OpenCLRuntimeHolder* holder, std::string* error) {
#if defined(MNN_USE_LIB_WRAPPER)
    if (MNN::OpenCLSymbolsOperator::createOpenCLSymbolsOperatorSingleInstance() == nullptr) {
        if (error) *error = "OpenCL symbols operator unavailable";
        return false;
    }
    MNN::OpenCLSymbols* symbols = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr();
    if (!symbols || symbols->isError()) {
        if (error) *error = "OpenCL symbol loading failed";
        return false;
    }
#endif
    BackendConfig config;
    config.precision = BackendConfig::Precision_Normal;
    Backend::Info info;
    info.type = MNN_FORWARD_OPENCL;
    info.numThread = 4;
    info.user = &config;
    holder->clRuntime.reset(new MNN::OpenCL::CLRuntime(info));
    if (!holder->clRuntime || holder->clRuntime->isCLRuntimeError()) {
        if (error) *error = "OpenCL runtime init failed";
        return false;
    }
    BackendConfig bc;
    bc.precision = BackendConfig::Precision_Normal;
    bc.memory = BackendConfig::Memory_Normal;
    holder->backend.reset(holder->clRuntime->onCreate(&bc, nullptr));
    if (!holder->backend) {
        if (error) *error = "OpenCL backend creation failed";
        return false;
    }
    holder->opencl = static_cast<MNN::OpenCL::OpenCLBackend*>(holder->backend.get())->getOpenCLRuntime();
    if (!holder->opencl) {
        if (error) *error = "OpenCL runtime handle unavailable";
        return false;
    }
    auto devices = holder->opencl->context().getInfo<CL_CONTEXT_DEVICES>();
    if (devices.empty()) {
        if (error) *error = "OpenCL context has no device";
        return false;
    }
    holder->device = devices[0];
    return true;
}

// Generic OpenCL runner: consumes AdaptedCase only.
static CaseReport runOpenCL(OpenCLRuntimeHolder* holder, const AdaptedCase& ac, int runs) {
    CaseReport report = buildReport(ac, "compile_failed");
    MNN::OpenCLRuntime* opencl = holder->opencl;
    cl::CommandQueue& queue = opencl->commandQueue();
    cl_int status = CL_SUCCESS;

    cl::Program::Sources sources;
    sources.emplace_back(ac.source.c_str(), ac.source.size());
    cl::Program program(opencl->context(), sources, &status);
    if (status != CL_SUCCESS) { report.error = "OpenCL program creation failed"; return report; }
    status = program.build(std::vector<cl::Device>(1, holder->device));
    if (status != CL_SUCCESS) {
        std::string log;
        program.getBuildInfo(holder->device, CL_PROGRAM_BUILD_LOG, &log);
        report.error = log.empty() ? "OpenCL build failed" : log;
        return report;
    }
    report.compileStatus = "compiled";

    cl::Kernel kernel(program, ac.entry.c_str(), &status);
    if (status != CL_SUCCESS) {
        report.error = "OpenCL kernel creation failed: " + ac.entry;
        return report;
    }

    // Create buffers from AdaptedCase.buffers
    std::vector<cl::Buffer> clBuffers;
    for (const auto& b : ac.buffers) {
        const cl_mem_flags flags = b.isOutput
            ? (CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR)
            : (CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR);
        clBuffers.emplace_back(opencl->context(), flags, b.sizeBytes, nullptr, &status);
        if (status != CL_SUCCESS) { report.error = "OpenCL buffer alloc failed"; return report; }
        if (!b.initialData.empty()) {
            status = queue.enqueueWriteBuffer(clBuffers.back(), CL_TRUE, 0, b.sizeBytes, b.initialData.data());
            if (status != CL_SUCCESS) { report.error = "OpenCL buffer write failed"; return report; }
        }
    }

    // Set args in AdaptedCase.args order
    for (size_t i = 0; i < ac.args.size(); ++i) {
        const auto& a = ac.args[i];
        switch (a.kind) {
            case AdaptedArg::SizeConst: {
                const int val = static_cast<int>(ac.globalSize[a.whichDim]);
                status |= kernel.setArg(i, val);
                break;
            }
            case AdaptedArg::Buffer: {
                status |= kernel.setArg(i, clBuffers[a.bufferIndex]);
                break;
            }
            case AdaptedArg::Scalar: {
                if (a.scalarType == AdaptedArg::Int) status |= kernel.setArg(i, a.intVal);
                else status |= kernel.setArg(i, a.floatVal);
                break;
            }
        }
    }
    if (status != CL_SUCCESS) { report.error = "OpenCL arg setup failed"; return report; }

    // warmup (outside PMU)
    for (int i = 0; i < ac.warmupRuns; ++i) {
        cl::NDRange g = ac.dims == 2
            ? cl::NDRange(ac.globalSize[0], ac.globalSize[1])
            : cl::NDRange(ac.globalSize[0], ac.globalSize[1], ac.globalSize[2]);
        status = queue.enqueueNDRangeKernel(kernel, cl::NullRange, g, cl::NullRange);
        if (status != CL_SUCCESS) { report.error = "OpenCL warmup dispatch failed"; return report; }
    }
    if (queue.finish() != CL_SUCCESS) { report.error = "OpenCL warmup finish failed"; return report; }
    report.dispatchStatus = "dispatched";

    // workload + PMU
    PmuScope pmu = beginPmu();
    for (int r = 0; r < runs; ++r) {
        cl::NDRange g = ac.dims == 2
            ? cl::NDRange(ac.globalSize[0], ac.globalSize[1])
            : cl::NDRange(ac.globalSize[0], ac.globalSize[1], ac.globalSize[2]);
        status = queue.enqueueNDRangeKernel(kernel, cl::NullRange, g, cl::NullRange);
        if (status != CL_SUCCESS) {
            report.error = "OpenCL workload dispatch failed";
            report.dispatchStatus = "dispatch_failed";
            return report;
        }
    }
    queue.finish();
    endPmu(pmu);
    report.pmuStatus = pmu.status;
    report.workloadDelta = pmu.workloadDelta;

    // readback output buffers
    std::vector<float> output;
    for (size_t i = 0; i < ac.buffers.size(); ++i) {
        if (!ac.buffers[i].isOutput) continue;
        output.resize(ac.buffers[i].sizeBytes / sizeof(float));
        status = queue.enqueueReadBuffer(clBuffers[i], CL_TRUE, 0, ac.buffers[i].sizeBytes, output.data());
        if (status != CL_SUCCESS) {
            report.error = "OpenCL readback failed";
            report.validationStatus = "readback_failed";
            return report;
        }
    }

    // validation
    const bool valid = runValidator(ac, output);
    report.validationStatus = valid ? "validation_passed" : "validation_failed";
    report.valid = valid;
    report.responsive = report.workloadDelta > 0;
    return report;
}

} // namespace

#endif // MNN_REPLAY_HAS_OPENCL

#if defined(MNN_REPLAY_HAS_VULKAN)

namespace {

struct VulkanRuntimeHolder {
    std::shared_ptr<VulkanInstance> instance;
    std::shared_ptr<VulkanDevice> device;
    std::shared_ptr<VulkanCommandPool> cmdPool;
    bool valid = false;
    std::string error;
};

static bool initVulkan(VulkanRuntimeHolder* holder) {
    if (!InitVulkan()) {
        holder->error = "InitVulkan failed (libvulkan.so load failed)";
        return false;
    }
    holder->instance = std::make_shared<VulkanInstance>();
    if (!holder->instance->supportVulkan()) { holder->error = "Vulkan not supported"; return false; }
    uint32_t gpuCount = 0;
    if (holder->instance->enumeratePhysicalDevices(gpuCount, nullptr) != VK_SUCCESS || gpuCount == 0) {
        holder->error = "No Vulkan physical device"; return false;
    }
    std::vector<VkPhysicalDevice> gpus(gpuCount);
    holder->instance->enumeratePhysicalDevices(gpuCount, gpus.data());
    holder->device = std::make_shared<VulkanDevice>(holder->instance);
    if (!holder->device->success()) { holder->error = "VulkanDevice creation failed"; return false; }
    holder->cmdPool = std::make_shared<VulkanCommandPool>(*holder->device);
    holder->valid = true;
    return true;
}

static bool compileGlslToSpirv(const std::string& source, std::vector<uint32_t>* spirv, std::string* error) {
    char tmpFile[] = "/tmp/mnn_kc_XXXXXX.comp";
    int fd = mkstemps(tmpFile, 5);
    if (fd < 0) { if (error) *error = "Cannot create temp GLSL file"; return false; }
    write(fd, source.c_str(), source.size());
    close(fd);
    std::string spvFile = std::string(tmpFile) + ".spv";
    std::string cmd = "glslangValidator -V --entry-point main " + std::string(tmpFile) + " -o " + spvFile + " 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) { unlink(tmpFile); if (error) *error = "Cannot run glslangValidator"; return false; }
    std::string log; char buf[1024];
    while (fgets(buf, sizeof(buf), pipe)) log += buf;
    int rc = pclose(pipe);
    unlink(tmpFile);
    if (rc != 0) { if (error) *error = "glslangValidator failed: " + log; unlink(spvFile.c_str()); return false; }
    std::ifstream spv(spvFile, std::ios::binary);
    if (!spv) { if (error) *error = "Cannot open SPIR-V"; return false; }
    spv.seekg(0, std::ios::end);
    size_t size = spv.tellg();
    spv.seekg(0, std::ios::beg);
    spirv->resize(size / sizeof(uint32_t));
    spv.read(reinterpret_cast<char*>(spirv->data()), size);
    spv.close();
    unlink(spvFile.c_str());
    return true;
}

// Generic Vulkan runner: consumes AdaptedCase only.
static CaseReport runVulkan(VulkanRuntimeHolder* holder, const AdaptedCase& ac, int runs) {
    CaseReport report = buildReport(ac, "compile_failed");
    if (!holder->valid) { report.compileStatus = "unsupported"; report.error = holder->error; return report; }

    VulkanDevice& dev = *holder->device;
    VulkanCommandPool& cmdPool = *holder->cmdPool;

    // 1. SPIR-V: prefer pre-compiled (ac.spirv), else glslangValidator
    std::vector<uint32_t> spirv;
    std::string compileError;
    if (!ac.spirv.empty()) {
        spirv = ac.spirv;
    } else if (!compileGlslToSpirv(ac.source, &spirv, &compileError)) {
        report.error = compileError;
        return report;
    }
    report.compileStatus = "compiled";

    VkShaderModule shaderModule;
    if (dev.createShaderModule(shaderModule, spirv.size() * sizeof(uint32_t), spirv.data()) != VK_SUCCESS) {
        report.error = "VkShaderModule creation failed"; return report;
    }

    // 3. Descriptor set layout from ac.buffers + ac.vulkanBindings
    const int numBindings = static_cast<int>(ac.buffers.size());
    std::vector<VkDescriptorSetLayoutBinding> bindings(numBindings);
    for (int i = 0; i < numBindings; ++i) {
        bindings[i].binding = ac.vulkanBindings.empty() ? i : ac.vulkanBindings[i];
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }
    VkDescriptorSetLayout setLayout;
    if (dev.createDescriptorSetLayout(setLayout, numBindings, bindings.data()) != VK_SUCCESS) {
        report.error = "VkDescriptorSetLayout creation failed";
        dev.destroyShaderModule(shaderModule);
        return report;
    }

    // 4. Pipeline layout (push constant 128B built-in)
    VkPipelineLayout pipelineLayout;
    if (dev.createPipelineLayout(pipelineLayout, setLayout) != VK_SUCCESS) {
        report.error = "VkPipelineLayout creation failed";
        dev.destroyDescriptorSetLayout(setLayout);
        dev.destroyShaderModule(shaderModule);
        return report;
    }

    // 5. Specialization info
    std::vector<VkSpecializationMapEntry> specEntries;
    std::vector<uint32_t> specData;
    VkSpecializationInfo specInfo;
    memset(&specInfo, 0, sizeof(specInfo));
    if (!ac.specConstants.empty()) {
        for (const auto& sc : ac.specConstants) {
            specEntries.push_back({static_cast<uint32_t>(sc.first),
                                   static_cast<uint32_t>(specData.size() * sizeof(uint32_t)),
                                   sizeof(uint32_t)});
            specData.push_back(sc.second);
        }
        specInfo.mapEntryCount = specEntries.size();
        specInfo.pMapEntries = specEntries.data();
        specInfo.dataSize = specData.size() * sizeof(uint32_t);
        specInfo.pData = specData.data();
    }

    // 6. Compute pipeline
    VkPipeline pipeline;
    if (dev.createComputePipeline(pipeline, shaderModule, pipelineLayout, VK_NULL_HANDLE, &specInfo) != VK_SUCCESS) {
        report.error = "VkPipeline creation failed";
        dev.destroyPipelineLayout(pipelineLayout);
        dev.destroyDescriptorSetLayout(setLayout);
        dev.destroyShaderModule(shaderModule);
        return report;
    }

    // 7. Descriptor pool + set
    VkDescriptorPoolSize poolSize;
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = numBindings;
    VkDescriptorPool descPool;
    if (dev.createDescriptorPool(descPool, 1, &poolSize) != VK_SUCCESS) {
        report.error = "VkDescriptorPool creation failed";
        dev.destroyPipeline(pipeline);
        dev.destroyPipelineLayout(pipelineLayout);
        dev.destroyDescriptorSetLayout(setLayout);
        dev.destroyShaderModule(shaderModule);
        return report;
    }
    VkDescriptorSet descSet;
    if (dev.allocateDescriptorSet(descSet, descPool, setLayout) != VK_SUCCESS) {
        report.error = "VkDescriptorSet allocation failed";
        dev.destroyDescriptorPool(descPool);
        dev.destroyPipeline(pipeline);
        dev.destroyPipelineLayout(pipelineLayout);
        dev.destroyDescriptorSetLayout(setLayout);
        dev.destroyShaderModule(shaderModule);
        return report;
    }

    // 8. Create buffers + write descriptor set
    VulkanMemoryPool memPool(dev, true);
    std::vector<std::unique_ptr<VulkanBuffer>> vkBuffers;
    std::vector<VkDescriptorBufferInfo> bufInfos(numBindings);
    for (int i = 0; i < numBindings; ++i) {
        const auto& b = ac.buffers[i];
        const void* hostData = b.initialData.empty() ? nullptr : b.initialData.data();
        vkBuffers.emplace_back(new VulkanBuffer(memPool, false, b.sizeBytes, hostData,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
        bufInfos[i].buffer = vkBuffers.back()->buffer();
        bufInfos[i].offset = 0;
        bufInfos[i].range = VK_WHOLE_SIZE;
    }
    std::vector<VkWriteDescriptorSet> writes(numBindings);
    for (int i = 0; i < numBindings; ++i) {
        memset(&writes[i], 0, sizeof(VkWriteDescriptorSet));
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descSet;
        writes[i].dstBinding = ac.vulkanBindings.empty() ? i : ac.vulkanBindings[i];
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    dev.updateDescriptorSets(numBindings, writes.data(), 0, nullptr);

    // warmup
    for (int i = 0; i < ac.warmupRuns; ++i) {
        VulkanCommandPool::Buffer* cmdbuf = cmdPool.allocBuffer();
        cmdbuf->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        cmdbuf->bindPipeline(pipeline);
        cmdbuf->bindDescriptorSets(pipelineLayout, 0, 1, &descSet);
        if (!ac.pushConstants.empty()) {
            cmdbuf->pushConstants(pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                                 ac.pushConstants.size(), ac.pushConstants.data());
        }
        cmdbuf->dispatch(ac.globalSize[0], ac.globalSize[1], ac.globalSize[2]);
        cmdbuf->end();
        cmdPool.submitAndWait(cmdbuf->get());
        delete cmdbuf;
    }
    report.dispatchStatus = "dispatched";

    // For in-place buffers (sigmoid/tanh), warmup already transformed the
    // data. Re-write initial data before workload so validation compares
    // against the original input.
    for (size_t i = 0; i < ac.buffers.size(); ++i) {
        if (!ac.buffers[i].initialData.empty() && ac.buffers[i].isOutput) {
            void* mapped = vkBuffers[i]->map();
            std::memcpy(mapped, ac.buffers[i].initialData.data(), ac.buffers[i].sizeBytes);
            vkBuffers[i]->unmap();
        }
    }

    // workload + PMU
    PmuScope pmu = beginPmu();
    for (int r = 0; r < runs; ++r) {
        // For in-place buffers, re-write initial data before each dispatch so
        // the output after N runs equals a single transformation of the input.
        for (size_t i = 0; i < ac.buffers.size(); ++i) {
            if (!ac.buffers[i].initialData.empty() && ac.buffers[i].isOutput) {
                void* mapped = vkBuffers[i]->map();
                std::memcpy(mapped, ac.buffers[i].initialData.data(), ac.buffers[i].sizeBytes);
                vkBuffers[i]->unmap();
            }
        }
        VulkanCommandPool::Buffer* cmdbuf = cmdPool.allocBuffer();
        cmdbuf->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        cmdbuf->bindPipeline(pipeline);
        cmdbuf->bindDescriptorSets(pipelineLayout, 0, 1, &descSet);
        if (!ac.pushConstants.empty()) {
            cmdbuf->pushConstants(pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                                 ac.pushConstants.size(), ac.pushConstants.data());
        }
        cmdbuf->dispatch(ac.globalSize[0], ac.globalSize[1], ac.globalSize[2]);
        cmdbuf->end();
        cmdPool.submitAndWait(cmdbuf->get());
        delete cmdbuf;
    }
    endPmu(pmu);
    report.pmuStatus = pmu.status;
    report.workloadDelta = pmu.workloadDelta;

    // readback output buffers
    std::vector<float> output;
    for (size_t i = 0; i < ac.buffers.size(); ++i) {
        if (!ac.buffers[i].isOutput) continue;
        output.resize(ac.buffers[i].sizeBytes / sizeof(float));
        void* mapped = vkBuffers[i]->map();
        memcpy(output.data(), mapped, ac.buffers[i].sizeBytes);
        vkBuffers[i]->unmap();
    }

    // validation
    const bool valid = runValidator(ac, output);
    report.validationStatus = valid ? "validation_passed" : "validation_failed";
    report.valid = valid;
    report.responsive = report.workloadDelta > 0;

    // cleanup
    dev.destroyDescriptorPool(descPool);
    dev.destroyPipeline(pipeline);
    dev.destroyPipelineLayout(pipelineLayout);
    dev.destroyDescriptorSetLayout(setLayout);
    dev.destroyShaderModule(shaderModule);
    return report;
}

} // namespace

#endif // MNN_REPLAY_HAS_VULKAN

bool runKernelCorpusBenchmark(const Options& options) {
    // Force bridge registration (function-local static, no global dynamic init).
    registerMnnBridge();
    registerNcnnBridge();

    rapidjson::Document document;
    document.SetObject();
    auto& allocator = document.GetAllocator();
    addString(document, "format", "mnn-kernel-corpus-benchmark", allocator);
    document.AddMember("version", 1, allocator);
    addString(document, "status", "ok", allocator);
    rapidjson::Value casesArray(rapidjson::kArrayType);
    rapidjson::Value errors(rapidjson::kArrayType);

    const std::string operatorsPath = joinPath(options.corpusRoot, options.operatorsFile);
    const std::string casesPath = joinPath(options.corpusRoot, options.casesFile);
    std::vector<uint8_t> opBytes, caseBytes;
    if (!readBytes(operatorsPath, opBytes) || !readBytes(casesPath, caseBytes)) {
        errors.PushBack(rapidjson::Value("Cannot load operators.json or operator_cases.json", allocator), allocator);
        document.AddMember("cases", casesArray, allocator);
        document.AddMember("errors", errors, allocator);
        return writeDoc(options, document);
    }
    rapidjson::Document opDoc, caseDoc;
    opDoc.Parse(reinterpret_cast<const char*>(opBytes.data()), opBytes.size());
    caseDoc.Parse(reinterpret_cast<const char*>(caseBytes.data()), caseBytes.size());
    if (opDoc.HasParseError() || !opDoc.IsObject() || !opDoc.HasMember("operators") ||
        caseDoc.HasParseError() || !caseDoc.IsObject() || !caseDoc.HasMember("cases")) {
        errors.PushBack(rapidjson::Value("Invalid operators.json or operator_cases.json", allocator), allocator);
        document.AddMember("cases", casesArray, allocator);
        document.AddMember("errors", errors, allocator);
        return writeDoc(options, document);
    }

    // Build operator lookup
    struct OpInfo {
        std::string file, entry, language;
    };
    std::map<std::string, OpInfo> opMap;
    for (const auto& v : opDoc["operators"].GetArray()) {
        std::string key = std::string(v["backend"].GetString()) + "/" +
                          v["op_type"].GetString() + "/" +
                          v["framework"].GetString() + "/" +
                          v["tag"].GetString() + "/" +
                          v["variant"].GetString();
        OpInfo info;
        info.file = v["file"].GetString();
        info.entry = v["entry"].GetString();
        info.language = v["language"].GetString();
        opMap[key] = info;
    }

    // Init runtimes lazily
#if defined(MNN_REPLAY_HAS_OPENCL)
    OpenCLRuntimeHolder openclHolder;
    bool openclReady = false;
    std::string openclError;
#endif
#if defined(MNN_REPLAY_HAS_VULKAN)
    VulkanRuntimeHolder vulkanHolder;
    bool vulkanReady = false;
    bool vulkanInited = false;
#endif

    for (const auto& v : caseDoc["cases"].GetArray()) {
        CaseSpec spec;
        spec.name = v.HasMember("name") && v["name"].IsString() ? v["name"].GetString() : "";
        spec.backend = v.HasMember("backend") && v["backend"].IsString() ? v["backend"].GetString() : "";
        spec.opType = v.HasMember("op_type") && v["op_type"].IsString() ? v["op_type"].GetString() : "";
        spec.framework = v.HasMember("framework") && v["framework"].IsString() ? v["framework"].GetString() : "";
        spec.tag = v.HasMember("tag") && v["tag"].IsString() ? v["tag"].GetString() : "";
        spec.variant = v.HasMember("variant") && v["variant"].IsString() ? v["variant"].GetString() : "";
        spec.dtype = v.HasMember("dtype") && v["dtype"].IsString() ? v["dtype"].GetString() : "";
        spec.validator = v.HasMember("validator") && v["validator"].IsString() ? v["validator"].GetString() : "";
        spec.warmupRuns = v.HasMember("warmup_runs") && v["warmup_runs"].IsInt() ? v["warmup_runs"].GetInt() : 2;
        spec.workloadRuns = v.HasMember("workload_runs") && v["workload_runs"].IsInt() ? v["workload_runs"].GetInt() : 5;
        spec.elementCount = v.HasMember("element_count") && v["element_count"].IsInt() ? v["element_count"].GetInt() : 0;
        spec.m = v.HasMember("m") && v["m"].IsInt() ? v["m"].GetInt() : 0;
        spec.n = v.HasMember("n") && v["n"].IsInt() ? v["n"].GetInt() : 0;
        spec.k = v.HasMember("k") && v["k"].IsInt() ? v["k"].GetInt() : 0;
        spec.w = v.HasMember("w") && v["w"].IsInt() ? v["w"].GetInt() : 0;
        spec.h = v.HasMember("h") && v["h"].IsInt() ? v["h"].GetInt() : 0;
        spec.c = v.HasMember("c") && v["c"].IsInt() ? v["c"].GetInt() : 0;
        spec.orderType = v.HasMember("order_type") && v["order_type"].IsInt() ? v["order_type"].GetInt() : 0;

        if (!options.caseFilter.empty() && spec.name != options.caseFilter) continue;

        CaseReport report;

        // Find operator file
        const std::string key = spec.backend + "/" + spec.opType + "/" + spec.framework + "/" +
                                spec.tag + "/" + spec.variant;
        auto opIt = opMap.find(key);
        if (opIt == opMap.end()) {
            AdaptedCase stub;
            stub.framework = spec.framework; stub.tag = spec.tag; stub.backend = spec.backend;
            stub.opType = spec.opType; stub.variant = spec.variant; stub.caseName = spec.name;
            report = buildReport(stub, "not_found", "Operator not in operators.json");
            goto emit;
        }

        {
            // Read source
            std::vector<uint8_t> srcBytes;
            if (!readBytes(joinPath(options.corpusRoot, opIt->second.file), srcBytes)) {
                AdaptedCase stub;
                stub.framework = spec.framework; stub.tag = spec.tag; stub.backend = spec.backend;
                stub.opType = spec.opType; stub.variant = spec.variant; stub.caseName = spec.name;
                report = buildReport(stub, "source_missing", "Cannot read " + opIt->second.file);
                goto emit;
            }
            const std::string source(reinterpret_cast<const char*>(srcBytes.data()), srcBytes.size());

            // Bridge adapt
            const Bridge* bridge = findBridge(spec.framework, spec.tag);
            if (bridge == nullptr) {
                AdaptedCase stub;
                stub.framework = spec.framework; stub.tag = spec.tag; stub.backend = spec.backend;
                stub.opType = spec.opType; stub.variant = spec.variant; stub.caseName = spec.name;
                report = buildReport(stub, "unsupported", "No bridge for " + spec.framework + "/" + spec.tag);
                goto emit;
            }
            AdaptedCase ac = bridge->adapt(spec, source, opIt->second.file, options.corpusRoot);
            ac.entry = opIt->second.entry;

            // Run
            if (spec.backend == "opencl") {
#if defined(MNN_REPLAY_HAS_OPENCL)
                if (!openclReady) {
                    openclReady = initOpenCL(&openclHolder, &openclError);
                    if (openclReady) {
                        addString(document, "opencl_device", openclHolder.device.getInfo<CL_DEVICE_NAME>(), allocator);
                    } else {
                        addString(document, "opencl_error", openclError, allocator);
                    }
                }
                if (openclReady) {
                    report = runOpenCL(&openclHolder, ac, std::max(1, options.runs));
                } else {
                    report = buildReport(ac, "unsupported", openclError);
                }
#else
                report = buildReport(ac, "unsupported", "OpenCL not compiled into replay_benchmark");
#endif
            } else if (spec.backend == "vulkan") {
#if defined(MNN_REPLAY_HAS_VULKAN)
                if (!vulkanInited) {
                    vulkanReady = initVulkan(&vulkanHolder);
                    vulkanInited = true;
                    if (vulkanReady) {
                        addString(document, "vulkan_device", vulkanHolder.device->proty().deviceName, allocator);
                    } else {
                        addString(document, "vulkan_error", vulkanHolder.error, allocator);
                    }
                }
                if (vulkanReady) {
                    report = runVulkan(&vulkanHolder, ac, std::max(1, options.runs));
                } else {
                    report = buildReport(ac, "unsupported", vulkanHolder.error);
                }
#else
                report = buildReport(ac, "unsupported", "Vulkan not compiled into replay_benchmark");
#endif
            } else {
                report = buildReport(ac, "unsupported", "Unknown backend: " + spec.backend);
            }
        }

    emit:
        {
            rapidjson::Value caseValue(rapidjson::kObjectType);
            addString(caseValue, "framework", report.framework, allocator);
            addString(caseValue, "tag", report.tag, allocator);
            addString(caseValue, "backend", report.backend, allocator);
            addString(caseValue, "operator", report.opType, allocator);
            addString(caseValue, "variant", report.variant, allocator);
            addString(caseValue, "case", report.caseName, allocator);
            addString(caseValue, "compile_status", report.compileStatus, allocator);
            addString(caseValue, "dispatch_status", report.dispatchStatus, allocator);
            addString(caseValue, "validation_status", report.validationStatus, allocator);
            addString(caseValue, "pmu_status", report.pmuStatus, allocator);
            caseValue.AddMember("control_delta", report.controlDelta, allocator);
            caseValue.AddMember("workload_delta", report.workloadDelta, allocator);
            caseValue.AddMember("responsive", report.responsive, allocator);
            caseValue.AddMember("valid", report.valid, allocator);
            if (!report.error.empty()) addString(caseValue, "error", report.error, allocator);
            casesArray.PushBack(caseValue, allocator);
        }
    }

    document.AddMember("cases", casesArray, allocator);
    document.AddMember("errors", errors, allocator);
    return writeDoc(options, document);
}

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
