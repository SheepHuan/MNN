#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_GENERIC_OPS_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_GENERIC_OPS_HPP

#include "../../OpAdapter.hpp"
#include <set>
#include <cstring>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// ---- Generic buffer adapter base ----
// Subclasses declare which variants they support and how many buffers/scalars
// to create. The adapt() fills deterministic test data.

class GenericBufAdapter : public OpAdapter {
public:
    struct TagSpec {
        std::string tag;
        std::string entry;
        int numInputBuffers;
        int numOutputBuffers;
        int numScalarInts;
        int numScalarFloats;
        int numSizeConsts;
        int numInt2s;
        int numInt4s;  // number of int4 params
        int globalDim;
        std::string validator;
    };
    struct Spec {
        const char* op;
        const char* variant;
        std::vector<TagSpec> tags;
    };

private:
    Spec mSpec;

public:
    GenericBufAdapter(const Spec& spec) : mSpec(spec) {}

    const char* opType() const override { return mSpec.op; }
    const char* variant() const override { return mSpec.variant; }

    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override {
        // Find matching tag
        const TagSpec* ts = nullptr;
        for (const auto& t : mSpec.tags) {
            if (t.tag == spec.tag) { ts = &t; break; }
        }
        if (ts == nullptr) return false;

        ac.entry = ts->entry;
        const int elemCount = spec.elementCount > 0 ? spec.elementCount : 64;

        // Operator-specific macros that cannot be in bake preamble.
        // OPERATOR/OPERATE are operator-specific and must be set per-variant.
        const std::string op = mSpec.op;
        const std::string var = mSpec.variant;
        if (ac.source.find("OPERATOR") != std::string::npos) {
            if (op == "binary" || op == "buffer_convert_buf") {
                ac.compileMacros.push_back("-DOPERATOR=(in0+in1)");
            } else if (op == "strassen") {
                ac.compileMacros.push_back("-DOPERATOR=(in0+in1)");
            } else {
                ac.compileMacros.push_back("-DOPERATOR=in");
            }
        }
        if (ac.source.find("OPERATE") != std::string::npos && ac.source.find("OPERATOR") == std::string::npos) {
            ac.compileMacros.push_back("-DOPERATE=(num+in)");
        }

        // Kernel-specific macros that MNN would pass at runtime via build options.
        // These are constants for the smoke test configuration.
        const std::string& src = ac.source;
        if (src.find("IN_C_BLOCK") != std::string::npos) ac.compileMacros.push_back("-DIN_C_BLOCK=4");
        if (src.find("LOCAL_SIZE") != std::string::npos) ac.compileMacros.push_back("-DLOCAL_SIZE=64");
        if (src.find("STRIDE_X") != std::string::npos) ac.compileMacros.push_back("-DSTRIDE_X=1");
        if (src.find("STRIDE_Y") != std::string::npos) ac.compileMacros.push_back("-DSTRIDE_Y=1");
        if (src.find("VEC_H") != std::string::npos) ac.compileMacros.push_back("-DVEC_H=4");
        if (src.find("OPWN") != std::string::npos) ac.compileMacros.push_back("-DOPWN=1");
        if (src.find("OPWM") != std::string::npos) ac.compileMacros.push_back("-DOPWM=1");
        if (src.find("K_SIZE") != std::string::npos) ac.compileMacros.push_back("-DK_SIZE=4");
        if (src.find("CPWK") != std::string::npos) ac.compileMacros.push_back("-DCPWK=1");
        if (src.find("KERNEL_X") != std::string::npos) ac.compileMacros.push_back("-DKERNEL_X=1");
        if (src.find("KERNEL_Y") != std::string::npos) ac.compileMacros.push_back("-DKERNEL_Y=1");
        if (src.find("OPTM") != std::string::npos) ac.compileMacros.push_back("-DOPTM=1");
        if (src.find("OPTN") != std::string::npos) ac.compileMacros.push_back("-DOPTN=1");
        if (src.find("INPUT_LINE_SIZE") != std::string::npos) ac.compileMacros.push_back("-DINPUT_LINE_SIZE=4");

        int argIdx = 0;
        for (int i = 0; i < ts->numSizeConsts; ++i) {
            ac.args.push_back(AdaptedArg::sizeConst(i));
            argIdx++;
        }
        for (int i = 0; i < ts->numInputBuffers; ++i) {
            std::vector<float> data(elemCount);
            for (int j = 0; j < elemCount; ++j) data[j] = 0.1f * (j % 13);
            AdaptedBuffer buf;
            buf.setFp32(data);
            buf.isOutput = false;
            ac.buffers.push_back(buf);
            ac.args.push_back(AdaptedArg::buffer(i));
            argIdx++;
        }
        for (int i = 0; i < ts->numOutputBuffers; ++i) {
            AdaptedBuffer buf;
            buf.sizeBytes = elemCount * sizeof(float);
            buf.isOutput = true;
            ac.buffers.push_back(buf);
            ac.args.push_back(AdaptedArg::buffer(ts->numInputBuffers + i));
            argIdx++;
        }
        for (int i = 0; i < ts->numScalarInts; ++i) {
            ac.args.push_back(AdaptedArg::scalarInt(elemCount));
            argIdx++;
        }
        for (int i = 0; i < ts->numScalarFloats; ++i) {
            ac.args.push_back(AdaptedArg::scalarFloat(0.5f));
            argIdx++;
        }
        for (int i = 0; i < ts->numInt2s; ++i) {
            ac.args.push_back(AdaptedArg::int2(8, 8));
            argIdx++;
        }
        for (int i = 0; i < ts->numInt4s; ++i) {
            ac.args.push_back(AdaptedArg::int4(1, 8, 8, 4));
            argIdx++;
        }
        if (ts->numInputBuffers > 0) {
            std::vector<float> data(elemCount);
            for (int j = 0; j < elemCount; ++j) data[j] = 0.1f * (j % 13);
            ac.validatorInputA = data;
        }
        ac.validator = "";  // No validator for generic adapters — semantics unknown
        ac.globalSize[0] = (elemCount + 3) / 4;
        ac.globalSize[1] = 1;
        ac.globalSize[2] = 1;
        ac.dims = ts->globalDim;
        return true;
    }
};

void registerGenericOps();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_GENERIC_OPS_HPP
