#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_GENERIC_OPS_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_GENERIC_OPS_HPP

#include "../../OpAdapter.hpp"
#include <cstring>
#include <regex>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

class GenericBufAdapter : public OpAdapter {
public:
    struct TagSpec {
        std::string tag;
        std::string entry;
        int numInputBuffers;
        int numOutputBuffers;
        int globalDim;
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
        const TagSpec* ts = nullptr;
        for (const auto& t : mSpec.tags) {
            if (t.tag == spec.tag) { ts = &t; break; }
        }
        if (ts == nullptr) return false;

        ac.entry = ts->entry;
        const int elemCount = spec.elementCount > 0 ? spec.elementCount : 64;

        // Auto-detect compile macros from source
        const std::string& src = ac.source;
        const std::string op = mSpec.op;
        if (src.find("OPERATOR") != std::string::npos) {
            if (op == "binary" || op == "buffer_convert_buf" || op == "strassen") {
                ac.compileMacros.push_back("-DOPERATOR=(in0+in1)");
            } else {
                ac.compileMacros.push_back("-DOPERATOR=in");
            }
        }
        if (src.find("OPERATE") != std::string::npos && src.find("OPERATOR") == std::string::npos) {
            ac.compileMacros.push_back("-DOPERATE=(num+in)");
        }
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

        // Parse kernel signature from source to get exact parameter order.
        std::regex sig_re("__kernel\\s+(?:__attribute__\\s*\\([^)]*\\)\\s+)?void\\s+\\w+\\s*\\(([^)]*\\))");
        std::smatch sig_match;
        bool sig_found = std::regex_search(ac.source, sig_match, sig_re);

        int bufIdx = 0;
        int totalBufs = ts->numInputBuffers + ts->numOutputBuffers;

        if (sig_found) {
            std::string params = sig_match[1].str();
            std::vector<std::string> paramList;
            int depth = 0; std::string cur;
            for (char c : params) {
                if (c == '(' || c == '<') depth++;
                else if (c == ')' || c == '>') depth--;
                if (c == ',' && depth == 0) { paramList.push_back(cur); cur = ""; }
                else cur += c;
            }
            if (!cur.empty()) paramList.push_back(cur);

            for (const auto& p_raw : paramList) {
                std::string p = std::regex_replace(p_raw, std::regex("__private\\s+const\\s+"), "");
                p = std::regex_replace(p, std::regex("__private\\s+"), "");
                p = std::regex_replace(p, std::regex("const\\s+"), "");
                p = std::regex_replace(p, std::regex("^\\s+|\\s+$"), "");

                if (std::regex_search(p, std::regex("global_size_dim\\d|global_dim\\d"))) {
                    std::smatch dm;
                    std::regex_search(p, dm, std::regex("(\\d)"));
                    ac.args.push_back(AdaptedArg::sizeConst(std::stoi(dm[1].str())));
                } else if (p.find("__global") != std::string::npos || p.find("__read_only") != std::string::npos || p.find("__write_only") != std::string::npos) {
                    bool isOutput = (bufIdx >= ts->numInputBuffers);
                    if (!isOutput) {
                        std::vector<float> data(elemCount);
                        for (int j = 0; j < elemCount; ++j) data[j] = 0.1f * (j % 13);
                        AdaptedBuffer buf; buf.setFp32(data); buf.isOutput = false;
                        ac.buffers.push_back(buf);
                    } else {
                        AdaptedBuffer buf; buf.sizeBytes = elemCount * sizeof(float); buf.isOutput = true;
                        ac.buffers.push_back(buf);
                    }
                    ac.args.push_back(AdaptedArg::buffer(bufIdx));
                    bufIdx++;
                } else if (p.find("int4") != std::string::npos) {
                    ac.args.push_back(AdaptedArg::int4(1, 8, 8, 4));
                } else if (p.find("int2") != std::string::npos) {
                    ac.args.push_back(AdaptedArg::int2(8, 8));
                } else if (p.find("float") != std::string::npos) {
                    ac.args.push_back(AdaptedArg::scalarFloat(0.5f));
                } else if (p.find("int") != std::string::npos) {
                    ac.args.push_back(AdaptedArg::scalarInt(elemCount));
                } else {
                    ac.args.push_back(AdaptedArg::scalarInt(elemCount));
                }
            }
        }

        if (bufIdx > 0) {
            std::vector<float> data(elemCount);
            for (int j = 0; j < elemCount; ++j) data[j] = 0.1f * (j % 13);
            ac.validatorInputA = data;
        }
        ac.validator = "";
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
