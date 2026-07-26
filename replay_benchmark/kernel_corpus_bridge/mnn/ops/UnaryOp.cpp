#include "UnaryOp.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

bool UnaryBufExpOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "unary_buf";
    const int count = spec.intParam("size", 1024);
    std::vector<float> input(count);
    for (int i = 0; i < count; ++i) input[i] = 0.1f * (i % 10);
    AdaptedBuffer inBuf;
    inBuf.setFp32(input);
    inBuf.isOutput = false;
    AdaptedBuffer outBuf;
    outBuf.sizeBytes = count * sizeof(float);
    outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    ac.validatorInputA = input;

    const int cb = (count + 3) / 4;
    if (spec.tag == "1.2.0") {
        // args: dim0=cb, dim1=1, dim2=1, input, output, height=1
        ac.args.push_back(AdaptedArg::sizeConst(0));
        ac.args.push_back(AdaptedArg::sizeConst(1));
        ac.args.push_back(AdaptedArg::sizeConst(2));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.args.push_back(AdaptedArg::scalarInt(1));
        ac.globalSize[0] = cb; ac.globalSize[1] = 1; ac.globalSize[2] = 1;
        ac.dims = 3;
    } else {
        // 3.6.0: dim0=cb, dim1=1, input, output, size=count
        ac.args.push_back(AdaptedArg::sizeConst(0));
        ac.args.push_back(AdaptedArg::sizeConst(1));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.args.push_back(AdaptedArg::scalarInt(count));
        ac.globalSize[0] = cb; ac.globalSize[1] = 1;
        ac.dims = 2;
    }
    return true;
}

void registerUnaryOp() {
    static struct Reg {
        Reg() { OpAdapterRegistry::instance().registerAdapter(std::unique_ptr<OpAdapter>(new UnaryBufExpOp())); }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
