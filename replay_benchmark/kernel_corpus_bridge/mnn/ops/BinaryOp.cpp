#include "BinaryOp.hpp"
#include <cstring>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

bool BinaryBufOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "binary_buf";
    const int elemCount = spec.intParam("size", 256);
    const int cb = (elemCount + 3) / 4;  // channel blocks

    // Input data
    std::vector<float> in0(elemCount), in1(elemCount);
    for (int i = 0; i < elemCount; ++i) {
        in0[i] = 0.1f * (i % 7);
        in1[i] = 0.1f * (i % 5);
    }

    AdaptedBuffer buf0;
    buf0.setFp32(in0);
    buf0.isOutput = false;
    AdaptedBuffer buf1;
    buf1.setFp32(in1);
    buf1.isOutput = false;
    AdaptedBuffer outBuf;
    outBuf.sizeBytes = elemCount * sizeof(float);
    outBuf.isOutput = true;
    ac.buffers.push_back(buf0);
    ac.buffers.push_back(buf1);
    ac.buffers.push_back(outBuf);
    ac.validatorInputA = in0;

    ac.compileMacros.push_back("-DOPERATOR=(in0+in1)");

    if (spec.tag == "1.2.0") {
        // (int dim0, int dim1, FLOAT* in0, FLOAT* in1, FLOAT* out, int4 shape, int2 isFull)
        ac.args.push_back(AdaptedArg::sizeConst(0));  // dim0 = cb
        ac.args.push_back(AdaptedArg::sizeConst(1));  // dim1 = 1
        ac.args.push_back(AdaptedArg::buffer(0));     // in0
        ac.args.push_back(AdaptedArg::buffer(1));     // in1
        ac.args.push_back(AdaptedArg::buffer(2));     // out
        ac.args.push_back(AdaptedArg::int4(1, 1, 1, cb));  // shape [N,H,W,C4]
        ac.args.push_back(AdaptedArg::int2(1, 1));    // isFull [x,y]
        ac.globalSize[0] = cb;
        ac.globalSize[1] = 1;
        ac.dims = 2;
    } else {
        // 3.6.0: (int dim0, int dim1, INPUT* in0, INPUT* in1, OUTPUT* out, int size, int actType)
        ac.args.push_back(AdaptedArg::sizeConst(0));  // dim0 = cb
        ac.args.push_back(AdaptedArg::sizeConst(1));  // dim1 = 1
        ac.args.push_back(AdaptedArg::buffer(0));     // in0
        ac.args.push_back(AdaptedArg::buffer(1));     // in1
        ac.args.push_back(AdaptedArg::buffer(2));     // out
        ac.args.push_back(AdaptedArg::scalarInt(elemCount));  // size
        ac.args.push_back(AdaptedArg::scalarInt(0));  // activationType = 0 (none)
        ac.globalSize[0] = cb;
        ac.globalSize[1] = 1;
        ac.dims = 2;
    }
    return true;
}

void registerBinaryOp() {
    static struct Reg {
        Reg() { OpAdapterRegistry::instance().registerAdapter(std::unique_ptr<OpAdapter>(new BinaryBufOp())); }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
