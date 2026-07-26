#include "PoolingOp.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

bool PoolingOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "pooling";
    const int ih = spec.h(), iw = spec.w(), channel = spec.c();
    const int channel_block = (channel + 3) / 4;
    const int kh = spec.kernelSize(), kw = spec.kernelSize();
    const int stride = spec.stride();
    const int oh = (ih - kh) / stride + 1;
    const int ow = (iw - kw) / stride + 1;
    const int batch = 1;

    const int inputCount = batch * channel_block * ih * iw * 4;
    const int outputCount = batch * channel_block * oh * ow * 4;

    std::vector<float> input(inputCount);
    for (int i = 0; i < inputCount; ++i) input[i] = 0.1f * (i % 13);

    AdaptedBuffer inBuf;
    inBuf.setFp32(input);
    inBuf.isOutput = false;
    AdaptedBuffer outBuf;
    outBuf.sizeBytes = outputCount * sizeof(float);
    outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    ac.validatorInputA = input;
    ac.h = ih; ac.w = iw; ac.c = channel; ac.k = kh;

    // args: dim0=ow, dim1=batch*oh, dim2=channel_block,
    //   input, int2(ih,iw), int2(oh,ow), int2(0,0), int2(stride,stride),
    //   int2(kh,kw), output, channel_block
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::sizeConst(2));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::int2(ih, iw));
    ac.args.push_back(AdaptedArg::int2(oh, ow));
    ac.args.push_back(AdaptedArg::int2(0, 0));
    ac.args.push_back(AdaptedArg::int2(stride, stride));
    ac.args.push_back(AdaptedArg::int2(kh, kw));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(channel_block));
    ac.globalSize[0] = ow;
    ac.globalSize[1] = batch * oh;
    ac.globalSize[2] = channel_block;
    ac.dims = 3;
    return true;
}

void registerPoolingOp() {
    static struct Reg {
        Reg() { OpAdapterRegistry::instance().registerAdapter(std::unique_ptr<OpAdapter>(new PoolingOp())); }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
