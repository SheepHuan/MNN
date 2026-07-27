#include "ReductionOp.hpp"
#include <cmath>
#include <cstring>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

bool OpenCLReductionSumKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "reduct_buf";
    const int batch = spec.batch(), height = spec.h(), width = spec.w();
    const int count = batch * width * 4;

    std::vector<float> input(batch * height * width * 4);
    for (int i = 0; i < static_cast<int>(input.size()); ++i) input[i] = 0.1f * (i % 7);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf;
    outBuf.sizeBytes = count * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    ac.validatorInputA = input;
    ac.m = batch; ac.h = height; ac.w = width;
    ac.elementCount = count;

    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(batch));
    ac.args.push_back(AdaptedArg::scalarInt(height));
    ac.args.push_back(AdaptedArg::scalarInt(width));
    ac.globalSize[0] = batch;
    ac.globalSize[1] = width;
    ac.dims = 2;
    return true;
}

bool OpenCLReductionSumKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    // NC4HW4 layout: input[((b*h+h_idx)*w+w_idx)*4], output[(b*w+w_idx)*4]
    for (int b = 0; b < ac.m; ++b) {
        for (int w_idx = 0; w_idx < ac.w; ++w_idx) {
            float sum = 0.0f;
            for (int h_idx = 0; h_idx < ac.h; ++h_idx) {
                const int off = ((b * ac.h + h_idx) * ac.w + w_idx) * 4;
                if (off < static_cast<int>(ac.validatorInputA.size())) sum += ac.validatorInputA[off];
            }
            const int out_off = (b * ac.w + w_idx) * 4;
            if (out_off < static_cast<int>(output.size())) {
                if (std::fabs(output[out_off] - sum) > 1e-2f) return false;
            }
        }
    }
    return true;
}

bool VulkanReductionSumKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    // Scalar layout: input[outside][axis][inside], output[outside][inside]
    const int outside = ac.m, axis = ac.h, inside = ac.w;
    if (static_cast<int>(output.size()) < outside * inside) return false;
    for (int o = 0; o < outside; ++o) {
        for (int i = 0; i < inside; ++i) {
            float sum = 0.0f;
            for (int a = 0; a < axis; ++a) {
                const int off = (o * axis + a) * inside + i;
                if (off < static_cast<int>(ac.validatorInputA.size())) sum += ac.validatorInputA[off];
            }
            const int out_off = o * inside + i;
            if (std::fabs(output[out_off] - sum) > 1e-2f) return false;
        }
    }
    return true;
}

bool VulkanReductionSumKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    // MNN Vulkan reduce.comp (SUM baked into SPIR-V).
    // layout(binding=0) output FLOAT[], binding=1 input FLOAT[],
    // binding=2 uniform { int w; int h; int c; float k; int reduceAxis; }
    ac.entry = "vulkan_reduce_buf_sum_fp32";
    const int batch = spec.batch(), height = spec.h(), width = spec.w();
    const int inside = width;
    const int axis = height;
    const int outside = batch;
    const int reduceAxis = 1;
    const int inputCount = outside * axis * inside;
    const int outputCount = outside * inside;

    std::vector<float> input(inputCount);
    for (int i = 0; i < inputCount; ++i) input[i] = 0.1f * (i % 7);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf;
    outBuf.sizeBytes = outputCount * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    ac.vulkanBindings = {1, 0};  // input->binding=1, output->binding=0
    ac.validatorInputA = input;
    ac.m = batch; ac.h = height; ac.w = width;
    ac.elementCount = outputCount;

    struct PushParam { int32_t w; int32_t h; int32_t c; float k; int32_t reduceAxis; };
    PushParam pp;
    pp.w = inside; pp.h = axis; pp.c = outside; pp.k = 1.0f; pp.reduceAxis = reduceAxis;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    const uint32_t localX = 256;
    const uint32_t total = static_cast<uint32_t>(outside * inside * reduceAxis);
    ac.globalSize[0] = ((total + localX - 1) / localX) * localX;
    ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    ac.localSize[0] = localX; ac.localSize[1] = 1; ac.localSize[2] = 1;
    ac.dims = 1;
    return true;
}

void registerReductionOp() {
    static struct Reg {
        Reg() {
            OpAdapterRegistry::instance().registerAdapter(
                std::unique_ptr<OpAdapter>(new OpenCLReductionSumKernel()));
            OpAdapterRegistry::instance().registerAdapter(
                std::unique_ptr<OpAdapter>(new VulkanReductionSumKernel()));
        }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
