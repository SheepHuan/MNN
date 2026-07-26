#include "NcnnBridge.hpp"
#include "../OpAdapter.hpp"
#include "ops/ElementwiseOp.hpp"

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

    const OpAdapter* adapter = OpAdapterRegistry::instance().find(spec.opType);
    if (adapter == nullptr || !adapter->adapt(spec, ac)) {
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
            registerBridge(std::unique_ptr<Bridge>(new NcnnBridge()));
        }
    } r;
    (void)r;
}

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
