#include "ReductionOp.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

bool ReductionOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "reduct_buf";
    const int batch = spec.m > 0 ? spec.m : 1;
    const int height = spec.h > 0 ? spec.h : 8;
    const int width = spec.w > 0 ? spec.w : 16;
    const int count = batch * width * 4;  // output: batch * width * FLOAT4

    std::vector<float> input(batch * height * width * 4);
    for (int i = 0; i < static_cast<int>(input.size()); ++i) input[i] = 0.1f * (i % 7);

    AdaptedBuffer inBuf;
    inBuf.setFp32(input);
    inBuf.isOutput = false;
    AdaptedBuffer outBuf;
    outBuf.sizeBytes = count * sizeof(float);
    outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    ac.validatorInputA = input;
    ac.m = batch; ac.h = height; ac.w = width;

    // args: dim0=batch, dim1=width, input, output, batch, height, width
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

void registerReductionOp() {
    static struct Reg {
        Reg() { OpAdapterRegistry::instance().registerAdapter(std::unique_ptr<OpAdapter>(new ReductionOp())); }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
