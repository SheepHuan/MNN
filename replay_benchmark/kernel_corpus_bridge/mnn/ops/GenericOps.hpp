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

        // OPERATOR macro: needed by unary/binary kernels. Use identity as default.
        if (ac.source.find("OPERATOR") != std::string::npos) {
            ac.compileMacros.push_back("-DOPERATOR=in");
        }
        // OPERATE macro: used by reduction_buf (sum). Default: num+in.
        if (ac.source.find("OPERATE") != std::string::npos && ac.source.find("OPERATOR") == std::string::npos) {
            ac.compileMacros.push_back("-DOPERATE=(num+in)");
        }

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
        if (ts->numInputBuffers > 0) {
            std::vector<float> data(elemCount);
            for (int j = 0; j < elemCount; ++j) data[j] = 0.1f * (j % 13);
            ac.validatorInputA = data;
        }
        ac.validator = ts->validator;
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
