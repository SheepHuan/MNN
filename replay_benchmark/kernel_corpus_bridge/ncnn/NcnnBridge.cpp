#include "NcnnBridge.hpp"
#include "../OpAdapter.hpp"
#include "ops/ElementwiseOp.hpp"
#include "ops/Pack4Op.hpp"
#include "ops/NcnnElementwiseOps.hpp"
#include "ops/NcnnShapeOps.hpp"
#include "ops/NcnnNormOps.hpp"

#include <cstring>
#include <fstream>
#include <sstream>

namespace MNN {
namespace Replay {
namespace KernelCorpus {

bool NcnnBridge::supports(const std::string& framework, const std::string& tag) const {
    if (framework != "ncnn") return false;
    return tag == "20190611" || tag == "20260526";
}

AdaptedCase NcnnBridge::adapt(const CaseSpec& spec,
                              const std::string& sourceText,
                              const std::string& sourceFile,
                              const std::string& corpusRoot) const {
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
    ac.elementCount = spec.intParam("size", 0);
    ac.w = spec.w(); ac.h = spec.h(); ac.c = spec.c();
    ac.orderType = spec.orderType();
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

    const OpAdapter* adapter = findAdapter(spec, ac);
    if (adapter == nullptr) {
        ac.entry.clear();
        ac.validator.clear();
    }
    return ac;
}

void registerNcnnBridge() {
    static struct Reg {
        Reg() {
            NcnnOps::registerElementwiseOp();
            NcnnOps::registerPermuteOp();
            NcnnOps::registerPack4Ops();
            NcnnOps::registerConcatOp();
            NcnnOps::registerNcnnElementwiseOps();
            NcnnOps::registerNcnnShapeOps();
            NcnnOps::registerNcnnNormOps();
            registerFallbackAdapter();
            registerBridge(std::unique_ptr<Bridge>(new NcnnBridge()));
        }
    } r;
    (void)r;
}

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
