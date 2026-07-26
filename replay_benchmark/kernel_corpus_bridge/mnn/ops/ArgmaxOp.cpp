#include "ArgmaxOp.hpp"
#include <cstring>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// argmax_buf 3.6.0: (dim0, dim1, dim2, FLOAT* in, int* out, int inside, int outside, int dim)
bool ArgmaxBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "argmax_buf";
    const int inside = spec.intParam("inside", 4), outside = spec.intParam("outside", 4), dim = spec.intParam("dim", 16);
    const int inN = inside * outside * dim;
    const int outN = inside * outside;
    std::vector<float> in;
    in.resize(inN);
    for (int i = 0; i < inN; ++i) in[i] = 0.1f * (i % 13);
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    // output is int* but we use float buffer for simplicity
    AdaptedBuffer outBuf; outBuf.sizeBytes = outN * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::sizeConst(0));  // dim0
    ac.args.push_back(AdaptedArg::sizeConst(1));  // dim1
    ac.args.push_back(AdaptedArg::sizeConst(2));  // dim2
    ac.args.push_back(AdaptedArg::buffer(0));     // in
    ac.args.push_back(AdaptedArg::buffer(1));     // out
    ac.args.push_back(AdaptedArg::scalarInt(inside));
    ac.args.push_back(AdaptedArg::scalarInt(outside));
    ac.args.push_back(AdaptedArg::scalarInt(dim));
    ac.globalSize[0] = 1; ac.globalSize[1] = outside; ac.globalSize[2] = inside; ac.dims = 3;
    ac.validatorInputA = in;
    return true;
}

void registerArgmaxOp() {
    static struct Reg {
        Reg() { OpAdapterRegistry::instance().registerAdapter(std::unique_ptr<OpAdapter>(new ArgmaxBufOp())); }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
