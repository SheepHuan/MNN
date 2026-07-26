#include "Pack4Op.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace NcnnOps {

static bool adaptPack4Elementwise(const CaseSpec& spec, AdaptedCase& ac) {
    if (spec.tag != "20260526") return false;
    const int elemCount = spec.intParam("size", 64);
    // pack4: elements stored as vec4, so n = elemCount / 4 (rounded up)
    const int n = (elemCount + 3) / 4;
    const int totalFloats = n * 4;

    std::vector<float> input(totalFloats);
    for (int i = 0; i < totalFloats; ++i) input[i] = 0.1f * (i % 13);

    AdaptedBuffer buf;
    buf.setFp32(input);  // in-place
    buf.isOutput = true;
    ac.buffers.push_back(buf);
    ac.vulkanBindings = {0};
    ac.validatorInputA = input;
    ac.elementCount = totalFloats;

    // push constant: uint n (4 bytes)
    struct PushParam { uint32_t n; };
    PushParam pp;
    pp.n = static_cast<uint32_t>(n);
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    // spec constant: constant_id=0 = n
    ac.specConstants.push_back({0, static_cast<uint32_t>(n)});

    // dispatch: global = (n, 1, 1), local 64
    const uint32_t localX = 64;
    ac.globalSize[0] = ((n + localX - 1) / localX) * localX;
    ac.globalSize[1] = 1;
    ac.globalSize[2] = 1;
    ac.localSize[0] = localX;
    return true;
}

#define DEFINE_PACK4_OP(name, vname) \
    bool name##Pack4Op::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
        return adaptPack4Elementwise(spec, ac); \
    }
PACK4_ELEMENTWISE_OPS(DEFINE_PACK4_OP)
#undef DEFINE_PACK4_OP

bool ConcatOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "20260526") return false;
    const int w = spec.w(), h = spec.h(), c = spec.c();
    const int cstep = w * h;
    const int total = c * cstep;

    std::vector<float> input(total);
    for (int i = 0; i < total; ++i) input[i] = 0.1f * (i % 13);

    AdaptedBuffer inBuf;
    inBuf.setFp32(input);
    inBuf.isOutput = false;
    AdaptedBuffer outBuf;
    outBuf.sizeBytes = total * sizeof(float);
    outBuf.isOutput = true;
    ac.buffers.push_back(inBuf);
    ac.buffers.push_back(outBuf);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = input;
    ac.w = w; ac.h = h; ac.c = c;

    // push constant: 13 ints
    struct PushParam {
        int dims, w, h, d, c, cstep;
        int outdims, outw, outh, outd, outc, outcstep;
        int offset;
    };
    PushParam pp;
    pp.dims = 3; pp.w = w; pp.h = h; pp.d = 1; pp.c = c; pp.cstep = cstep;
    pp.outdims = 3; pp.outw = w; pp.outh = h; pp.outd = 1; pp.outc = c; pp.outcstep = cstep;
    pp.offset = 0;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    // spec constants: axis=0, then shape constants (id 1..12)
    ac.specConstants.push_back({0, 0});  // axis
    ac.specConstants.push_back({1, 3});  // dims
    ac.specConstants.push_back({2, static_cast<uint32_t>(w)});
    ac.specConstants.push_back({3, static_cast<uint32_t>(h)});
    ac.specConstants.push_back({4, 1});  // d
    ac.specConstants.push_back({5, static_cast<uint32_t>(c)});
    ac.specConstants.push_back({6, static_cast<uint32_t>(cstep)});
    // output shape (same as input for identity concat)
    ac.specConstants.push_back({7, 3});  // outdims
    ac.specConstants.push_back({8, static_cast<uint32_t>(w)});
    ac.specConstants.push_back({9, static_cast<uint32_t>(h)});
    ac.specConstants.push_back({10, 1}); // outd
    ac.specConstants.push_back({11, static_cast<uint32_t>(c)});
    ac.specConstants.push_back({12, static_cast<uint32_t>(cstep)});

    ac.globalSize[0] = w;
    ac.globalSize[1] = h;
    ac.globalSize[2] = c;
    return true;
}

void registerPack4Ops() {
    static struct Reg {
        Reg() {
#define REG_PACK4_OP(name, vname) \
            OpAdapterRegistry::instance().registerAdapter( \
                std::unique_ptr<OpAdapter>(new name##Pack4Op()));
            PACK4_ELEMENTWISE_OPS(REG_PACK4_OP)
#undef REG_PACK4_OP
        }
    } r;
    (void)r;
}

void registerConcatOp() {
    static struct Reg {
        Reg() {
            OpAdapterRegistry::instance().registerAdapter(std::unique_ptr<OpAdapter>(new ConcatOp()));
        }
    } r;
    (void)r;
}

} // namespace NcnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
