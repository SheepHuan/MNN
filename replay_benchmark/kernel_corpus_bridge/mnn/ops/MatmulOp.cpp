#include "MatmulOp.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

bool MatmulOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "matmul_buf";
    const int M = spec.m(), N = spec.n(), K = spec.k();
    const int M4 = (M + 3) / 4;
    const int N4 = (N + 3) / 4;
    const int K4 = (K + 3) / 4;
    const int aCount = M4 * K4 * 4;
    const int bCount = K4 * N4 * 4;
    const int outCount = M4 * N4 * 4;

    std::vector<float> a(aCount), b(bCount);
    for (int i = 0; i < aCount; ++i) a[i] = 0.01f * (i % 7);
    for (int i = 0; i < bCount; ++i) b[i] = 0.01f * (i % 5);

    AdaptedBuffer aBuf, bBuf, outBuf;
    aBuf.setFp32(a); aBuf.isOutput = false;
    bBuf.setFp32(b); bBuf.isOutput = false;
    outBuf.sizeBytes = outCount * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(aBuf);
    ac.buffers.push_back(bBuf);
    ac.buffers.push_back(outBuf);
    ac.validatorInputA = a;
    ac.validatorInputB = b;

    // args: dim0=N4, dim1=M4, a, b, out, K, K4, N4
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::scalarInt(K));
    ac.args.push_back(AdaptedArg::scalarInt(K4));
    ac.args.push_back(AdaptedArg::scalarInt(N4));
    ac.globalSize[0] = N4;
    ac.globalSize[1] = M4;
    ac.dims = 2;
    return true;
}

void registerMatmulOp() {
    static struct Reg {
        Reg() { OpAdapterRegistry::instance().registerAdapter(std::unique_ptr<OpAdapter>(new MatmulOp())); }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
