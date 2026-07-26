#include "ElementwiseOp.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace NcnnOps {

void adaptElementwise(const CaseSpec& spec, AdaptedCase& ac) {
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int cstep = w * h;
    const int total = c * cstep;

    std::vector<float> input(total);
    for (int i = 0; i < total; ++i) input[i] = 0.1f * (i % 13);

    AdaptedBuffer buf;
    buf.setFp32(input);  // in-place: initial data goes into the single buffer
    buf.isOutput = true;
    ac.buffers.push_back(buf);
    ac.vulkanBindings = {0};
    ac.validatorInputA = input;

    // push constant: int dims, w, h, c, cstep (20 bytes)
    struct PushParam { int dims, w, h, c, cstep; };
    PushParam pp;
    pp.dims = (c == 1 && h == 1) ? 1 : (h == 1 ? 2 : 3);
    pp.w = w; pp.h = h; pp.c = c; pp.cstep = cstep;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    // dispatch: global = (ceil(w/64)*64, 1, 1)
    const uint32_t localX = 64;
    ac.globalSize[0] = ((w + localX - 1) / localX) * localX;
    ac.globalSize[1] = 1;
    ac.globalSize[2] = 1;
    ac.localSize[0] = localX;
}

bool PermuteOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int cstep = w * h;
    const int total = c * cstep;

    std::vector<float> input(total);
    for (int i = 0; i < total; ++i) input[i] = 0.1f * (i % 13);

    AdaptedBuffer inBuf, outBuf;
    inBuf.setFp32(input); inBuf.isOutput = false;
    outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = input;

    struct PushParam {
        int dims, w, h, c, cstep;
        int outdims, outw, outh, outc, outcstep;
    };
    PushParam pp;
    pp.dims = 3;
    pp.w = w; pp.h = h; pp.c = c; pp.cstep = cstep;
    pp.outdims = 3; pp.outw = w; pp.outh = h; pp.outc = c; pp.outcstep = cstep;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    ac.specConstants.push_back({0, static_cast<uint32_t>(spec.orderType())});
    ac.globalSize[0] = w;
    ac.globalSize[1] = h;
    ac.globalSize[2] = c;
    return true;
}

void registerElementwiseOp() {
    static struct Reg {
        Reg() {
            OpAdapterRegistry::instance().registerAdapter(std::unique_ptr<OpAdapter>(new SigmoidOp()));
            OpAdapterRegistry::instance().registerAdapter(std::unique_ptr<OpAdapter>(new TanhOp()));
        }
    } r;
    (void)r;
}

void registerPermuteOp() {
    static struct Reg {
        Reg() {
            OpAdapterRegistry::instance().registerAdapter(std::unique_ptr<OpAdapter>(new PermuteOp()));
        }
    } r;
    (void)r;
}

} // namespace NcnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
