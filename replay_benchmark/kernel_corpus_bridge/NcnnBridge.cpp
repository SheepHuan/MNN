#include "Bridge.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

namespace MNN {
namespace Replay {
namespace KernelCorpus {

namespace {

class NcnnBridge : public Bridge {
public:
    bool supports(const std::string& framework, const std::string& tag) const override {
        if (framework != "ncnn") return false;
        return tag == "20190611" || tag == "20260526";
    }

    AdaptedCase adapt(const CaseSpec& spec,
                      const std::string& sourceText,
                      const std::string& sourceFile,
                      const std::string& corpusRoot) const override {
        AdaptedCase ac;
        ac.framework = spec.framework;
        ac.tag = spec.tag;
        ac.backend = spec.backend;
        ac.opType = spec.opType;
        ac.variant = spec.variant;
        ac.caseName = spec.name;
        ac.validator = spec.validator;
        ac.warmupRuns = spec.warmupRuns;
        ac.workloadRuns = spec.workloadRuns;
        ac.elementCount = spec.elementCount;
        ac.w = spec.w; ac.h = spec.h; ac.c = spec.c;
        ac.orderType = spec.orderType;
        ac.source = sourceText;
        ac.entry = "main";

        // Load pre-compiled SPIR-V: replace .comp with .spv in the source path.
        if (!sourceFile.empty()) {
            std::string spvPath = sourceFile;
            const size_t len = spvPath.size();
            if (len >= 5 && spvPath.compare(len - 5, 5, ".comp") == 0) {
                spvPath.replace(len - 5, 5, ".spv");
                std::string full = corpusRoot;
                if (!full.empty() && full.back() != '/') full.push_back('/');
                full += spvPath;
                std::ifstream spv(full, std::ios::binary);
                if (spv) {
                    std::ostringstream ss;
                    ss << spv.rdbuf();
                    std::string data = ss.str();
                    ac.spirv.resize(data.size() / sizeof(uint32_t));
                    std::memcpy(ac.spirv.data(), data.data(), data.size());
                }
            }
        }

        if (spec.opType == "sigmoid" || spec.opType == "tanh") {
            adaptElementwise(spec, ac);
        } else if (spec.opType == "permute") {
            adaptPermute(spec, ac);
        } else {
            ac.entry = "";
            ac.validator.clear();
        }
        return ac;
    }

private:
    // sigmoid/tanh: layout(binding=0) buffer bottom_top_blob (in-place)
    // push_constant: {dims, w, h, c, cstep}
    // global = (w, 1, 1) for 1D; (w, h, c) for 3D
    void adaptElementwise(const CaseSpec& spec, AdaptedCase& ac) const {
        const int w = spec.w > 0 ? spec.w : 64;
        const int h = spec.h > 0 ? spec.h : 1;
        const int c = spec.c > 0 ? spec.c : 1;
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
        struct PushParam {
            int dims, w, h, c, cstep;
        };
        PushParam pp;
        pp.dims = (c == 1 && h == 1) ? 1 : (h == 1 ? 2 : 3);
        pp.w = w; pp.h = h; pp.c = c; pp.cstep = cstep;
        ac.pushConstants.resize(sizeof(pp));
        std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

        // dispatch: global = (w, 1, 1) with local_size_x_id=233 (we use 64)
        const uint32_t localX = 64;
        ac.globalSize[0] = ((w + localX - 1) / localX) * localX;
        ac.globalSize[1] = 1;
        ac.globalSize[2] = 1;
        ac.localSize[0] = localX;
    }

    // permute: layout(binding=0) readonly bottom_blob, layout(binding=1) writeonly top_blob
    // push_constant: {dims, w, h, c, cstep, outdims, outw, outh, outc, outcstep} (40 bytes)
    // spec constant: constant_id=0 order_type
    void adaptPermute(const CaseSpec& spec, AdaptedCase& ac) const {
        const int w = spec.w > 0 ? spec.w : 16;
        const int h = spec.h > 0 ? spec.h : 16;
        const int c = spec.c > 0 ? spec.c : 4;
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

        ac.specConstants.push_back({0, static_cast<uint32_t>(spec.orderType)});

        ac.globalSize[0] = w;
        ac.globalSize[1] = h;
        ac.globalSize[2] = c;
    }
};

void ensureRegistered() {
    static struct Registrar {
        Registrar() {
            registerBridge(std::unique_ptr<Bridge>(new NcnnBridge()));
        }
    } r;
    (void)r;
}

} // namespace

void registerNcnnBridge() {
    ensureRegistered();
}

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
