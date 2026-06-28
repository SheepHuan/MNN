#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "MNNTestSuite.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backend/opencl/core/OpenCLBackend.hpp"
#include "core/Macro.h"
#include "core/Backend.hpp"

using namespace MNN;

namespace {

struct RawHybridCase {
    const char* name;
    bool keyImage;
    bool valueImage;
    bool scramble;
    int qRows;
    int kvRows;
    int channels;
    int span;
};

static int envInt(const char* name, int fallback) {
    const char* value = ::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::atoi(value);
}

static int applyOpenCLTuneLevelOverride(int numThread) {
    const char* envValue = ::getenv("MNN_OPENCL_TUNE_LEVEL");
    if (envValue == nullptr || envValue[0] == '\0') {
        return numThread;
    }
    std::string level(envValue);
    std::transform(level.begin(), level.end(), level.begin(),
                   [](unsigned char c) { return static_cast<char>(::tolower(c)); });
    const int tuneMask = MNN_GPU_TUNING_NONE | MNN_GPU_TUNING_FAST | MNN_GPU_TUNING_NORMAL |
                         MNN_GPU_TUNING_HEAVY | MNN_GPU_TUNING_WIDE;
    int tuneFlag = 0;
    if (level == "none") {
        tuneFlag = MNN_GPU_TUNING_NONE;
    } else if (level == "fast") {
        tuneFlag = MNN_GPU_TUNING_FAST;
    } else if (level == "normal") {
        tuneFlag = MNN_GPU_TUNING_NORMAL;
    } else if (level == "heavy") {
        tuneFlag = MNN_GPU_TUNING_HEAVY;
    } else if (level == "wide") {
        tuneFlag = MNN_GPU_TUNING_WIDE;
    } else {
        return numThread;
    }
    return (numThread & ~tuneMask) | tuneFlag;
}

static bool enabledByFilter(const char* envName, const char* name) {
    const char* filter = ::getenv(envName);
    if (filter == nullptr || filter[0] == '\0') {
        return true;
    }
    return std::string(name).find(filter) != std::string::npos;
}

static const char* kRawHybridProbeSource = R"CLC(
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

inline int raw_hybrid_slot(int base, int cb, int i, int kv_rows, int scramble) {
    if (scramble == 0) {
        return (base + i) % kv_rows;
    }
    return (base * 17 + i * 131 + cb * 3 + 7) % kv_rows;
}

__kernel void RawHybridImgKVSpan(__global const float* query,
                                 __global const int* slot_idx,
                                 __read_only image2d_t key_img,
                                 __read_only image2d_t value_img,
                                 __global float* output,
                                 __private const int channel_blocks,
                                 __private const int q_rows,
                                 __private const int kv_rows,
                                 __private const int span,
                                 __private const int scramble) {
    const int cb = get_global_id(0);
    const int q = get_global_id(1);
    if (cb >= channel_blocks || q >= q_rows) {
        return;
    }
    const int qoff = ((q * channel_blocks) + cb) << 2;
    const float4 qv = vload4(0, query + qoff);
    const int base = slot_idx[q];
    float4 acc = (float4)0;
    for (int i = 0; i < span; ++i) {
        const int slot = raw_hybrid_slot(base, cb, i, kv_rows, scramble);
        const int2 coord = (int2)(cb, slot);
        const float4 kv = read_imagef(key_img, SAMPLER, coord);
        const float4 vv = read_imagef(value_img, SAMPLER, coord);
        acc = mad(qv, kv, acc);
        acc += vv;
    }
    vstore4(acc, 0, output + qoff);
}

__kernel void RawHybridBufKVSpan(__global const float* query,
                                 __global const int* slot_idx,
                                 __global const float* key_buf,
                                 __global const float* value_buf,
                                 __global float* output,
                                 __private const int channel_blocks,
                                 __private const int q_rows,
                                 __private const int kv_rows,
                                 __private const int span,
                                 __private const int scramble) {
    const int cb = get_global_id(0);
    const int q = get_global_id(1);
    if (cb >= channel_blocks || q >= q_rows) {
        return;
    }
    const int qoff = ((q * channel_blocks) + cb) << 2;
    const float4 qv = vload4(0, query + qoff);
    const int base = slot_idx[q];
    float4 acc = (float4)0;
    for (int i = 0; i < span; ++i) {
        const int slot = raw_hybrid_slot(base, cb, i, kv_rows, scramble);
        const int off = ((slot * channel_blocks) + cb) << 2;
        const float4 kv = vload4(0, key_buf + off);
        const float4 vv = vload4(0, value_buf + off);
        acc = mad(qv, kv, acc);
        acc += vv;
    }
    vstore4(acc, 0, output + qoff);
}

__kernel void RawHybridImgKeyBufValueSpan(__global const float* query,
                                          __global const int* slot_idx,
                                          __read_only image2d_t key_img,
                                          __global const float* value_buf,
                                          __global float* output,
                                          __private const int channel_blocks,
                                          __private const int q_rows,
                                          __private const int kv_rows,
                                          __private const int span,
                                          __private const int scramble) {
    const int cb = get_global_id(0);
    const int q = get_global_id(1);
    if (cb >= channel_blocks || q >= q_rows) {
        return;
    }
    const int qoff = ((q * channel_blocks) + cb) << 2;
    const float4 qv = vload4(0, query + qoff);
    const int base = slot_idx[q];
    float4 acc = (float4)0;
    for (int i = 0; i < span; ++i) {
        const int slot = raw_hybrid_slot(base, cb, i, kv_rows, scramble);
        const int2 coord = (int2)(cb, slot);
        const int off = ((slot * channel_blocks) + cb) << 2;
        const float4 kv = read_imagef(key_img, SAMPLER, coord);
        const float4 vv = vload4(0, value_buf + off);
        acc = mad(qv, kv, acc);
        acc += vv;
    }
    vstore4(acc, 0, output + qoff);
}

__kernel void RawHybridBufKeyImgValueSpan(__global const float* query,
                                          __global const int* slot_idx,
                                          __global const float* key_buf,
                                          __read_only image2d_t value_img,
                                          __global float* output,
                                          __private const int channel_blocks,
                                          __private const int q_rows,
                                          __private const int kv_rows,
                                          __private const int span,
                                          __private const int scramble) {
    const int cb = get_global_id(0);
    const int q = get_global_id(1);
    if (cb >= channel_blocks || q >= q_rows) {
        return;
    }
    const int qoff = ((q * channel_blocks) + cb) << 2;
    const float4 qv = vload4(0, query + qoff);
    const int base = slot_idx[q];
    float4 acc = (float4)0;
    for (int i = 0; i < span; ++i) {
        const int slot = raw_hybrid_slot(base, cb, i, kv_rows, scramble);
        const int2 coord = (int2)(cb, slot);
        const int off = ((slot * channel_blocks) + cb) << 2;
        const float4 kv = vload4(0, key_buf + off);
        const float4 vv = read_imagef(value_img, SAMPLER, coord);
        acc = mad(qv, kv, acc);
        acc += vv;
    }
    vstore4(acc, 0, output + qoff);
}
)CLC";

#define CHECK_CL(error, info) \
    do { \
        if ((error) != CL_SUCCESS) { \
            std::fprintf(stderr, "CL ERROR %d: %s\n", static_cast<int>(error), info); \
            return false; \
        } \
    } while (false)

static std::vector<RawHybridCase> allCases() {
    return {
        {"mixed_imgimg_span_q269_kv2560_c2048", true, true, false, 269, 2560, 2048, 8},
        {"mixed_imgkey_bufvalue_span_q269_kv2560_c2048", true, false, false, 269, 2560, 2048, 8},
        {"mixed_bufkey_imgvalue_span_q269_kv2560_c2048", false, true, false, 269, 2560, 2048, 8},
        {"mixed_bufbuf_span_q269_kv2560_c2048", false, false, false, 269, 2560, 2048, 8},
        {"mixed_imgimg_scramble_q269_kv2560_c2048", true, true, true, 269, 2560, 2048, 8},
        {"mixed_imgkey_bufvalue_scramble_q269_kv2560_c2048", true, false, true, 269, 2560, 2048, 8},
        {"mixed_bufkey_imgvalue_scramble_q269_kv2560_c2048", false, true, true, 269, 2560, 2048, 8},
        {"mixed_bufbuf_scramble_q269_kv2560_c2048", false, false, true, 269, 2560, 2048, 8},
        {"mixed_imgimg_span_q519_kv2560_c2048", true, true, false, 519, 2560, 2048, 8},
        {"mixed_imgkey_bufvalue_span_q519_kv2560_c2048", true, false, false, 519, 2560, 2048, 8},
        {"mixed_bufkey_imgvalue_span_q519_kv2560_c2048", false, true, false, 519, 2560, 2048, 8},
        {"mixed_bufbuf_span_q519_kv2560_c2048", false, false, false, 519, 2560, 2048, 8},
        {"mixed_imgimg_scramble_q519_kv2560_c2048", true, true, true, 519, 2560, 2048, 8},
        {"mixed_imgkey_bufvalue_scramble_q519_kv2560_c2048", true, false, true, 519, 2560, 2048, 8},
        {"mixed_bufkey_imgvalue_scramble_q519_kv2560_c2048", false, true, true, 519, 2560, 2048, 8},
        {"mixed_bufbuf_scramble_q519_kv2560_c2048", false, false, true, 519, 2560, 2048, 8},
        {"mixed_imgimg_span_q269_kv2048_c1536", true, true, false, 269, 2048, 1536, 8},
        {"mixed_imgkey_bufvalue_span_q269_kv2048_c1536", true, false, false, 269, 2048, 1536, 8},
        {"mixed_bufkey_imgvalue_span_q269_kv2048_c1536", false, true, false, 269, 2048, 1536, 8},
        {"mixed_bufbuf_span_q269_kv2048_c1536", false, false, false, 269, 2048, 1536, 8},
        {"mixed_imgimg_scramble_q269_kv2048_c1536", true, true, true, 269, 2048, 1536, 8},
        {"mixed_imgkey_bufvalue_scramble_q269_kv2048_c1536", true, false, true, 269, 2048, 1536, 8},
        {"mixed_bufkey_imgvalue_scramble_q269_kv2048_c1536", false, true, true, 269, 2048, 1536, 8},
        {"mixed_bufbuf_scramble_q269_kv2048_c1536", false, false, true, 269, 2048, 1536, 8},
    };
}

static std::vector<float> makeFloatPattern(size_t size, int seed) {
    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>((static_cast<int>(i * 13 + seed) % 29) - 14) * 0.03125f;
    }
    return data;
}

static std::vector<int> makeSlotIndex(int qRows, int kvRows, bool scramble) {
    std::vector<int> slot(qRows);
    const int usable = std::max(1, kvRows - 8);
    for (int i = 0; i < qRows; ++i) {
        if (scramble) {
            slot[i] = (i * 37 + 13) % usable;
        } else {
            slot[i] = i % usable;
        }
    }
    return slot;
}

static bool validLws(const std::array<size_t, 2>& lws, const std::vector<uint32_t>& maxItems, uint64_t maxWorkGroupSize) {
    if (lws[0] == 0 || lws[1] == 0) {
        return false;
    }
    if (!maxItems.empty() && lws[0] > maxItems[0]) {
        return false;
    }
    if (maxItems.size() > 1 && lws[1] > maxItems[1]) {
        return false;
    }
    return lws[0] * lws[1] <= maxWorkGroupSize;
}

class ProbeOpenCLContext {
public:
    explicit ProbeOpenCLContext(int precision) {
        Backend::Info info;
        info.type = MNN_FORWARD_OPENCL;
        info.numThread = MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_NONE;
        info.numThread = applyOpenCLTuneLevelOverride(info.numThread);

        auto status = MNNTestSuite::get()->pStaus;
        mConfig.memory = static_cast<BackendConfig::MemoryMode>(status.memory);
        mConfig.precision = static_cast<BackendConfig::PrecisionMode>(precision);
        mConfig.power = static_cast<BackendConfig::PowerMode>(status.power);
        info.user = &mConfig;

        auto creator = MNNGetExtraRuntimeCreator(MNN_FORWARD_OPENCL);
        if (creator == nullptr) {
            MNN_ERROR("failed to get OpenCL runtime creator for raw hybrid probe\n");
            return;
        }
        mRuntime.reset(creator->onCreate(info));
        if (!mRuntime) {
            MNN_ERROR("failed to create OpenCL runtime for raw hybrid probe\n");
            return;
        }
        mBackend.reset(mRuntime->onCreate(&mConfig));
        if (!mBackend) {
            MNN_ERROR("failed to create OpenCL backend for raw hybrid probe\n");
            return;
        }
        mOpenCLBackend = static_cast<OpenCL::OpenCLBackend*>(mBackend.get());
        if (mOpenCLBackend == nullptr) {
            MNN_ERROR("failed to resolve OpenCL backend for raw hybrid probe\n");
            return;
        }
    }

    bool valid() const {
        return mOpenCLBackend != nullptr;
    }

    OpenCL::OpenCLBackend* backend() const {
        return mOpenCLBackend;
    }

    OpenCLRuntime* runtime() const {
        return mOpenCLBackend != nullptr ? mOpenCLBackend->getOpenCLRuntime() : nullptr;
    }

private:
    BackendConfig mConfig;
    std::shared_ptr<Runtime> mRuntime;
    std::unique_ptr<Backend> mBackend;
    OpenCL::OpenCLBackend* mOpenCLBackend = nullptr;
};

static const char* kernelNameForCase(const RawHybridCase& c) {
    if (c.keyImage && c.valueImage) {
        return "RawHybridImgKVSpan";
    }
    if (c.keyImage) {
        return "RawHybridImgKeyBufValueSpan";
    }
    if (c.valueImage) {
        return "RawHybridBufKeyImgValueSpan";
    }
    return "RawHybridBufKVSpan";
}

static const char* storageName(const RawHybridCase& c) {
    if (c.keyImage && c.valueImage) {
        return "imgkey_imgvalue";
    }
    if (c.keyImage) {
        return "imgkey_bufvalue";
    }
    if (c.valueImage) {
        return "bufkey_imgvalue";
    }
    return "bufkey_bufvalue";
}

static bool runCase(const RawHybridCase& c, int precision, int warmup, int repeat) {
    ProbeOpenCLContext probe(precision);
    if (!probe.valid() || probe.runtime() == nullptr) {
        return false;
    }
    if (probe.runtime()->getGpuType() == ADRENO) {
        MNN_PRINT("[bench_ops/opencl/perf/RawHybridProbe] skip %-38s on Adreno: direct raw CL buffer/image "
                  "probing is unstable in wrapped run_test.out; use WeightOnlyConv/FuseExprProbe instead.\n",
                  c.name);
        ::fflush(stdout);
        return true;
    }
    auto* runtime = probe.runtime();
    auto& context = runtime->context();
    auto& queue = runtime->commandQueue();
    cl_int res = CL_SUCCESS;

    const int channelBlocks = c.channels / 4;
    const size_t qFloatCount = static_cast<size_t>(c.qRows) * static_cast<size_t>(channelBlocks) * 4;
    const size_t kvFloatCount = static_cast<size_t>(c.kvRows) * static_cast<size_t>(channelBlocks) * 4;
    const size_t qBytes = qFloatCount * sizeof(float);
    const size_t kvBytes = kvFloatCount * sizeof(float);
    const size_t slotBytes = static_cast<size_t>(c.qRows) * sizeof(int);

    auto queryHost = makeFloatPattern(qFloatCount, 5);
    auto keyHost = makeFloatPattern(kvFloatCount, 11);
    auto valueHost = makeFloatPattern(kvFloatCount, 19);
    auto slotHost = makeSlotIndex(c.qRows, c.kvRows, c.scramble);
    std::vector<float> outputHost(qFloatCount, 0.0f);

    res = CL_SUCCESS;
    cl::Buffer queryBuf(context, CL_MEM_READ_ONLY, qBytes, nullptr, &res);
    CHECK_CL(res, "create query buffer");
    cl::Buffer slotBuf(context, CL_MEM_READ_ONLY, slotBytes, nullptr, &res);
    CHECK_CL(res, "create slot buffer");
    cl::Buffer outputBuf(context, CL_MEM_READ_WRITE, qBytes, nullptr, &res);
    CHECK_CL(res, "create output buffer");
    cl::Buffer keyBuf(context, CL_MEM_READ_ONLY, kvBytes, nullptr, &res);
    CHECK_CL(res, "create key buffer");
    cl::Buffer valueBuf(context, CL_MEM_READ_ONLY, kvBytes, nullptr, &res);
    CHECK_CL(res, "create value buffer");

    cl::ImageFormat imageFormat(CL_RGBA, CL_FLOAT);
    cl::Image2D keyImg(context, CL_MEM_READ_ONLY, imageFormat, channelBlocks, c.kvRows, 0, nullptr, &res);
    CHECK_CL(res, "create key image");
    cl::Image2D valueImg(context, CL_MEM_READ_ONLY, imageFormat, channelBlocks, c.kvRows, 0, nullptr, &res);
    CHECK_CL(res, "create value image");

    res = queue.enqueueWriteBuffer(queryBuf, CL_TRUE, 0, qBytes, queryHost.data());
    CHECK_CL(res, "write query buffer");
    res = queue.enqueueWriteBuffer(slotBuf, CL_TRUE, 0, slotBytes, slotHost.data());
    CHECK_CL(res, "write slot buffer");
    res = queue.enqueueWriteBuffer(keyBuf, CL_TRUE, 0, kvBytes, keyHost.data());
    CHECK_CL(res, "write key buffer");
    res = queue.enqueueWriteBuffer(valueBuf, CL_TRUE, 0, kvBytes, valueHost.data());
    CHECK_CL(res, "write value buffer");
    const std::array<size_t, 3> origin = {0, 0, 0};
    const std::array<size_t, 3> region = {static_cast<size_t>(channelBlocks), static_cast<size_t>(c.kvRows), 1};
    const size_t imageRowPitch = static_cast<size_t>(channelBlocks) * 4 * sizeof(float);
    res = queue.enqueueWriteImage(keyImg, CL_TRUE, origin, region, imageRowPitch, 0, keyHost.data());
    CHECK_CL(res, "write key image");
    res = queue.enqueueWriteImage(valueImg, CL_TRUE, origin, region, imageRowPitch, 0, valueHost.data());
    CHECK_CL(res, "write value image");

    auto kernel = runtime->buildKernelFromSource(kRawHybridProbeSource, kernelNameForCase(c), {}, precision);
    if (kernel == nullptr) {
        MNN_ERROR("failed to build raw hybrid probe kernel: %s\n", c.name);
        return false;
    }
    const uint64_t maxWorkGroupSize = runtime->getMaxWorkGroupSize(kernel);
    const std::vector<uint32_t> maxItems32 = runtime->getMaxWorkItemSizes();
    const uint64_t waveSize = runtime->GetKernelWaveSize(kernel);
    const std::vector<std::array<size_t, 2>> candidates = {
        {8, 4}, {16, 4}, {32, 4}, {8, 8}, {16, 8}, {32, 8}, {64, 4}
    };

    cl::NDRange gws(static_cast<size_t>(channelBlocks), static_cast<size_t>(c.qRows));
    double bestMs = std::numeric_limits<double>::max();
    std::array<size_t, 2> bestLws = {0, 0};
    bool found = false;
    for (const auto& lws : candidates) {
        if (!validLws(lws, maxItems32, maxWorkGroupSize)) {
            continue;
        }
        uint32_t idx = 0;
        res = CL_SUCCESS;
        res |= kernel->get().setArg(idx++, queryBuf);
        res |= kernel->get().setArg(idx++, slotBuf);
        if (c.keyImage && c.valueImage) {
            res |= kernel->get().setArg(idx++, keyImg);
            res |= kernel->get().setArg(idx++, valueImg);
        } else if (c.keyImage) {
            res |= kernel->get().setArg(idx++, keyImg);
            res |= kernel->get().setArg(idx++, valueBuf);
        } else if (c.valueImage) {
            res |= kernel->get().setArg(idx++, keyBuf);
            res |= kernel->get().setArg(idx++, valueImg);
        } else {
            res |= kernel->get().setArg(idx++, keyBuf);
            res |= kernel->get().setArg(idx++, valueBuf);
        }
        res |= kernel->get().setArg(idx++, outputBuf);
        res |= kernel->get().setArg(idx++, channelBlocks);
        res |= kernel->get().setArg(idx++, c.qRows);
        res |= kernel->get().setArg(idx++, c.kvRows);
        res |= kernel->get().setArg(idx++, c.span);
        res |= kernel->get().setArg(idx++, c.scramble ? 1 : 0);
        CHECK_CL(res, "setArg raw hybrid");

        cl::NDRange lwsRange(lws[0], lws[1]);
        for (int i = 0; i < warmup; ++i) {
            res = queue.enqueueNDRangeKernel(kernel->get(), cl::NullRange, gws, lwsRange);
            CHECK_CL(res, "warmup raw hybrid");
        }
        queue.finish();

        double totalMs = 0.0;
        for (int i = 0; i < repeat; ++i) {
            const auto start = std::chrono::steady_clock::now();
            res = queue.enqueueNDRangeKernel(kernel->get(), cl::NullRange, gws, lwsRange);
            CHECK_CL(res, "run raw hybrid");
            queue.finish();
            const auto end = std::chrono::steady_clock::now();
            totalMs += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
        }
        const double avgMs = totalMs / static_cast<double>(repeat);
        MNN_PRINT("[bench_ops/opencl/perf/RawHybridProbe] %-38s kv=%s access=%s q=%d kv_rows=%d c=%d span=%d "
                  "wave=%llu lws=%zux%zu avg=%.4f ms\n",
                  c.name, storageName(c), c.scramble ? "scramble" : "span",
                  c.qRows, c.kvRows, c.channels, c.span, static_cast<unsigned long long>(waveSize),
                  lws[0], lws[1], avgMs);
        ::fflush(stdout);
        if (avgMs < bestMs) {
            bestMs = avgMs;
            bestLws = lws;
            found = true;
        }
    }

    if (!found) {
        MNN_ERROR("no valid lws for %s\n", c.name);
        return false;
    }
    res = queue.enqueueReadBuffer(outputBuf, CL_TRUE, 0, qBytes, outputHost.data());
    CHECK_CL(res, "read output buffer");
    const float checksum = outputHost.empty() ? 0.0f : outputHost[(static_cast<size_t>(c.qRows) * 13u) % outputHost.size()];
    MNN_PRINT("[bench_ops/opencl/perf/RawHybridProbeBest] %-34s best_lws=%zux%zu best=%.4f ms checksum=%.6f\n",
              c.name, bestLws[0], bestLws[1], bestMs, checksum);
    ::fflush(stdout);
    return true;
}

} // namespace

class OpenCLRawHybridProbePerf : public MNNTestCase {
public:
    bool run(int precision) override {
        const int resolvedPrecision = envInt("MNN_BENCH_OPENCL_PRECISION", precision);
        const int warmup = std::max(1, envInt("MNN_BENCH_OPENCL_RAW_HYBRID_WARMUP", 3));
        const int repeat = std::max(1, envInt("MNN_BENCH_OPENCL_RAW_HYBRID_REPEAT", 20));
        bool ok = true;
        for (const auto& c : allCases()) {
            if (!enabledByFilter("MNN_BENCH_OPENCL_RAW_HYBRID_CASE", c.name)) {
                continue;
            }
            ok = runCase(c, resolvedPrecision, warmup, repeat) && ok;
        }
        return ok;
    }
};

MNNTestSuiteRegister(OpenCLRawHybridProbePerf, "bench_ops/opencl/perf/RawHybridProbe");

#endif
