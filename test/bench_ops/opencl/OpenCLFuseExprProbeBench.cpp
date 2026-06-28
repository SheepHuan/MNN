#if defined(MNN_SUPPORT_TRANSFORMER_FUSE) && !defined(MNN_OPENCL_BUFFER_CLOSED)

#include "MNNTestSuite.h"
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "MNN_generated.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace MNN;
using namespace MNN::Express;

namespace {

struct ExprProbeCase {
    const char* name;
    const char* kernelName;
    bool useImage;
    int inputCount;
    int rows;
    int channels;
};

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

static int envInt(const char* name, int fallback) {
    const char* value = ::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::atoi(value);
}

static bool enabledByFilter(const char* envName, const char* name) {
    const char* filter = ::getenv(envName);
    if (filter == nullptr || filter[0] == '\0') {
        return true;
    }
    return std::string(name).find(filter) != std::string::npos;
}

static const char* kAdrenoExprProbeSource = R"CLC(
#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void AdrenoExprBuildOnly(__read_only image2d_t input0,
                                  __write_only image2d_t output0,
                                  __private const int global_size_dim0,
                                  __private const int global_size_dim1,
                                  __private const int global_size_dim2) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(x, y, z);
}

__kernel void AdrenoExprImageCopy(__read_only image2d_t input0,
                                  __write_only image2d_t output0,
                                  __private const int global_size_dim0,
                                  __private const int global_size_dim1,
                                  __private const int global_size_dim2) {
    const int channel_block_idx = get_global_id(0);
    const int w = get_global_id(1);
    const int hb = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(channel_block_idx, w, hb);
    const int width = global_size_dim1;
    const int pos = mad24(channel_block_idx, width, w);
    WI_F(output0, (int2)(pos, hb), RI_F(input0, SAMPLER, (int2)(pos, hb)));
}

__kernel void AdrenoExprDualReadR8(__read_only image2d_t input0,
                                   __read_only image2d_t input1,
                                   __write_only image2d_t output0,
                                   __private const int global_size_dim0,
                                   __private const int global_size_dim1,
                                   __private const int global_size_dim2) {
    const int channel_block_idx = get_global_id(0);
    const int w = get_global_id(1);
    const int hb = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(channel_block_idx, w, hb);
    const int width = global_size_dim1;
    const int pos = mad24(channel_block_idx, width, w);
    const int2 coord = (int2)(pos, hb);
    FLOAT4 acc = (FLOAT4)0;
    for (int i = 0; i < 8; ++i) {
        acc += RI_F(input0, SAMPLER, coord);
        acc += RI_F(input1, SAMPLER, coord);
    }
    WI_F(output0, coord, acc);
}

__kernel void AdrenoExprScrambleDualReadR8(__read_only image2d_t input0,
                                           __read_only image2d_t input1,
                                           __write_only image2d_t output0,
                                           __private const int global_size_dim0,
                                           __private const int global_size_dim1,
                                           __private const int global_size_dim2) {
    const int channel_block_idx = get_global_id(0);
    const int w = get_global_id(1);
    const int hb = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(channel_block_idx, w, hb);
    const int src_cb = (channel_block_idx * 17 + hb * 3 + 5) % global_size_dim0;
    const int src_hb = (hb * 13 + channel_block_idx * 5 + 7) % global_size_dim2;
    const int src_w = (w + src_cb) % global_size_dim1;
    const int width = global_size_dim1;
    const int src_pos = mad24(src_cb, width, src_w);
    const int dst_pos = mad24(channel_block_idx, width, w);
    const int2 src_coord = (int2)(src_pos, src_hb);
    const int2 dst_coord = (int2)(dst_pos, hb);
    FLOAT4 acc = (FLOAT4)0;
    for (int i = 0; i < 8; ++i) {
        acc += RI_F(input0, SAMPLER, src_coord);
        acc += RI_F(input1, SAMPLER, src_coord);
    }
    WI_F(output0, dst_coord, acc);
}

__kernel void AdrenoBufBuildOnly(__global const FLOAT* input0,
                                 __global FLOAT* output0,
                                 __private const int global_size_dim0,
                                 __private const int global_size_dim1,
                                 __private const int global_size_dim2) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(x, y, z);
}

__kernel void AdrenoBufCopy(__global const FLOAT* input0,
                            __global FLOAT* output0,
                            __private const int global_size_dim0,
                            __private const int global_size_dim1,
                            __private const int global_size_dim2) {
    const int channel_block_idx = get_global_id(0);
    const int w = get_global_id(1);
    const int hb = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(channel_block_idx, w, hb);
    const int offset = (((channel_block_idx * global_size_dim2) + hb) * global_size_dim1 + w) << 2;
    const FLOAT4 value = vload4(0, input0 + offset);
    vstore4(value, 0, output0 + offset);
}

__kernel void AdrenoBufDualReadR8(__global const FLOAT* input0,
                                  __global const FLOAT* input1,
                                  __global FLOAT* output0,
                                  __private const int global_size_dim0,
                                  __private const int global_size_dim1,
                                  __private const int global_size_dim2) {
    const int channel_block_idx = get_global_id(0);
    const int w = get_global_id(1);
    const int hb = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(channel_block_idx, w, hb);
    const int offset = (((channel_block_idx * global_size_dim2) + hb) * global_size_dim1 + w) << 2;
    FLOAT4 acc = (FLOAT4)0;
    for (int i = 0; i < 8; ++i) {
        acc += vload4(0, input0 + offset);
        acc += vload4(0, input1 + offset);
    }
    vstore4(acc, 0, output0 + offset);
}

__kernel void AdrenoBufScrambleDualReadR8(__global const FLOAT* input0,
                                          __global const FLOAT* input1,
                                          __global FLOAT* output0,
                                          __private const int global_size_dim0,
                                          __private const int global_size_dim1,
                                          __private const int global_size_dim2) {
    const int channel_block_idx = get_global_id(0);
    const int w = get_global_id(1);
    const int hb = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(channel_block_idx, w, hb);
    const int src_cb = (channel_block_idx * 17 + hb * 3 + 5) % global_size_dim0;
    const int src_hb = (hb * 13 + channel_block_idx * 5 + 7) % global_size_dim2;
    const int src_w = (w + src_cb) % global_size_dim1;
    const int src_offset = (((src_cb * global_size_dim2) + src_hb) * global_size_dim1 + src_w) << 2;
    const int dst_offset = (((channel_block_idx * global_size_dim2) + hb) * global_size_dim1 + w) << 2;
    FLOAT4 acc = (FLOAT4)0;
    for (int i = 0; i < 8; ++i) {
        acc += vload4(0, input0 + src_offset);
        acc += vload4(0, input1 + src_offset);
    }
    vstore4(acc, 0, output0 + dst_offset);
}
)CLC";

static std::shared_ptr<Module> createExprProbeModule(const ExprProbeCase& c) {
    std::vector<VARP> inputs;
    for (int i = 0; i < c.inputCount; ++i) {
        auto input = _Input({1, c.rows, 1, c.channels}, NHWC, halide_type_of<float>());
        input->setName("input" + std::to_string(i));
        inputs.emplace_back(input);
    }

    std::unique_ptr<OpT> op(new OpT);
    op->name = "opencl_expr_probe";
    op->type = OpType_Extra;
    op->defaultDimentionFormat = MNN_DATA_FORMAT_NHWC;
    op->main.type = OpParameter_Extra;
    op->main.value = new ExtraT;
    auto* extra = op->main.AsExtra();
    extra->type = c.kernelName;
    extra->engine = "MNN";
    const size_t sourceLen = ::strlen(kAdrenoExprProbeSource) + 1;
    extra->info.resize(sourceLen);
    ::memcpy(extra->info.data(), kAdrenoExprProbeSource, sourceLen);

    std::vector<std::string> inputNames;
    for (int i = 0; i < c.inputCount; ++i) {
        inputNames.emplace_back("input" + std::to_string(i));
    }

    auto output = Variable::create(Expr::create(op.get(), std::move(inputs)));
    output->setName("output");

    std::unique_ptr<NetT> net(new NetT);
    std::vector<VARP> outputs{output};
    Variable::save(outputs, net.get());
    flatbuffers::FlatBufferBuilder builder;
    auto len = Net::Pack(builder, net.get());
    builder.Finish(len);
    return std::shared_ptr<Module>(Module::load(inputNames, {"output"}, builder.GetBufferPointer(), builder.GetSize()),
                                   Module::destroy);
}

static std::vector<ExprProbeCase> allCases() {
    return {
        {"img_build_only_rows269_c2048", "AdrenoExprBuildOnly", true, 1, 269, 2048},
        {"img_copy_rows269_c2048", "AdrenoExprImageCopy", true, 1, 269, 2048},
        {"img_dual_read_r8_rows269_c2048", "AdrenoExprDualReadR8", true, 2, 269, 2048},
        {"img_scramble_dual_read_r8_rows269_c2048", "AdrenoExprScrambleDualReadR8", true, 2, 269, 2048},
        {"img_copy_rows519_c2048", "AdrenoExprImageCopy", true, 1, 519, 2048},
        {"img_dual_read_r8_rows519_c2048", "AdrenoExprDualReadR8", true, 2, 519, 2048},
        {"img_scramble_dual_read_r8_rows519_c2048", "AdrenoExprScrambleDualReadR8", true, 2, 519, 2048},
        {"img_copy_rows269_c1536", "AdrenoExprImageCopy", true, 1, 269, 1536},
        {"img_dual_read_r8_rows269_c1536", "AdrenoExprDualReadR8", true, 2, 269, 1536},
        {"buf_build_only_rows269_c2048", "AdrenoBufBuildOnly", false, 1, 269, 2048},
        {"buf_copy_rows269_c2048", "AdrenoBufCopy", false, 1, 269, 2048},
        {"buf_dual_read_r8_rows269_c2048", "AdrenoBufDualReadR8", false, 2, 269, 2048},
        {"buf_scramble_dual_read_r8_rows269_c2048", "AdrenoBufScrambleDualReadR8", false, 2, 269, 2048},
        {"buf_copy_rows519_c2048", "AdrenoBufCopy", false, 1, 519, 2048},
        {"buf_dual_read_r8_rows519_c2048", "AdrenoBufDualReadR8", false, 2, 519, 2048},
        {"buf_scramble_dual_read_r8_rows519_c2048", "AdrenoBufScrambleDualReadR8", false, 2, 519, 2048},
        {"buf_copy_rows269_c1536", "AdrenoBufCopy", false, 1, 269, 1536},
        {"buf_dual_read_r8_rows269_c1536", "AdrenoBufDualReadR8", false, 2, 269, 1536},
    };
}

static bool runProbeCase(const ExprProbeCase& c, int precision, int warmup, int repeat) {
    BackendConfig config;
    config.precision = static_cast<BackendConfig::PrecisionMode>(precision);
    config.memory = BackendConfig::Memory_Normal;
    int gpuMode = (c.useImage ? MNN_GPU_MEMORY_IMAGE : MNN_GPU_MEMORY_BUFFER) | MNN_GPU_TUNING_NONE;
    gpuMode = applyOpenCLTuneLevelOverride(gpuMode);
    std::shared_ptr<Executor> executor(Executor::newExecutor(MNN_FORWARD_OPENCL, config, gpuMode));
    if (!executor) {
        MNN_ERROR("failed to create OpenCL executor for expr probe\n");
        return false;
    }
    ExecutorScope scope(executor);
    executor->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, gpuMode);

    auto module = createExprProbeModule(c);
    if (!module) {
        MNN_ERROR("failed to create expr probe module\n");
        return false;
    }

    std::vector<VARP> inputs;
    for (int inputIndex = 0; inputIndex < c.inputCount; ++inputIndex) {
        auto input = _Input({1, c.rows, 1, c.channels}, NHWC, halide_type_of<float>());
        auto ptr = input->writeMap<float>();
        const int count = c.rows * c.channels;
        for (int i = 0; i < count; ++i) {
            ptr[i] = static_cast<float>(((i + inputIndex * 7) % 29) - 14) * 0.03125f;
        }
        input->unMap();
        inputs.emplace_back(input);
    }

    for (int i = 0; i < warmup; ++i) {
        auto outputs = module->onForward(inputs);
        outputs[0]->readMap<float>();
    }
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
        auto outputs = module->onForward(inputs);
        outputs[0]->readMap<float>();
    }
    auto end = std::chrono::steady_clock::now();
    const double totalMs =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    const double avgMs = totalMs / static_cast<double>(repeat);
    MNN_PRINT("[bench_ops/opencl/perf/FuseExprProbe] %-28s kernel=%-20s rows=%d c=%d inputs=%d precision=%d gpu_mode=0x%x avg=%.4f ms\n",
              c.name, c.kernelName, c.rows, c.channels, c.inputCount, precision, gpuMode, avgMs);
    ::fflush(stdout);
    return true;
}

} // namespace

class OpenCLFuseExprProbePerf : public MNNTestCase {
public:
    bool run(int precision) override {
        const int resolvedPrecision =
            envInt("MNN_BENCH_OPENCL_PRECISION", precision);
        const int warmup = std::max(1, envInt("MNN_BENCH_OPENCL_FUSE_EXPR_WARMUP", 3));
        const int repeat = std::max(1, envInt("MNN_BENCH_OPENCL_FUSE_EXPR_REPEAT", 10));
        bool ok = true;
        for (const auto& c : allCases()) {
            if (!enabledByFilter("MNN_BENCH_OPENCL_FUSE_EXPR_CASE", c.name)) {
                continue;
            }
            ok = runProbeCase(c, resolvedPrecision, warmup, repeat) && ok;
        }
        return ok;
    }
};

MNNTestSuiteRegister(OpenCLFuseExprProbePerf, "bench_ops/opencl/perf/FuseExprProbe");

#endif
