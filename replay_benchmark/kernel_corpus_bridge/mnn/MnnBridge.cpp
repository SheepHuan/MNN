#include "MnnBridge.hpp"
#include "../OpAdapter.hpp"
#include "ops/RasterOp.hpp"
#include "ops/UnaryOp.hpp"
#include "ops/MatmulOp.hpp"
#include "ops/ReductionOp.hpp"
#include "ops/PoolingOp.hpp"
#include "ops/BinaryOp.hpp"
#include "ops/ElementwiseOps.hpp"
#include "ops/NormConvOps.hpp"
#include "ops/ConvOps.hpp"
#include "ops/ArgmaxOp.hpp"
#include "ops/ComplexOps.hpp"

#include <cstring>
#include <fstream>
#include <sstream>

namespace MNN {
namespace Replay {
namespace KernelCorpus {

bool MnnBridge::supports(const std::string& framework, const std::string& tag) const {
    if (framework != "mnn") return false;
    // CUDA corpus spans 1.2.0 through 3.6.0; intermediate tags are added as
    // their kernel variants are implemented in the adapters.
    static const char* kSupported[] = {
        "1.2.0", "1.2.7", "2.0.4", "2.2.2", "2.2.3", "2.4.2",
        "2.5.1", "2.7.1", "2.8.0", "2.8.4", "3.6.0"
    };
    for (auto t : kSupported) if (tag == t) return true;
    return false;
}

AdaptedCase MnnBridge::adapt(const CaseSpec& spec,
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
    ac.m = spec.m(); ac.n = spec.n(); ac.k = spec.k();
    ac.w = spec.w(); ac.h = spec.h(); ac.c = spec.c();
    ac.orderType = spec.orderType();
    ac.stride = spec.stride();
    ac.source = sourceText;

    // For Vulkan MNN kernels, load pre-compiled SPIR-V (.spv) alongside the
    // .comp source so the device-side runner does not need glslangValidator.
    // Mirrors the NcnnBridge .spv loading path.
    if (spec.backend == "vulkan" && !sourceFile.empty()) {
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
    } else {
        ac.adapter = adapter;
    }
    return ac;
}

void registerMnnBridge() {
    static struct Reg {
        Reg() {
            MnnOps::registerRasterOp();
            MnnOps::registerUnaryOp();
            MnnOps::registerMatmulOp();
            MnnOps::registerReductionOp();
            MnnOps::registerPoolingOp();
            MnnOps::registerBinaryOp();
            MnnOps::registerElementwiseOps();
            MnnOps::registerNormConvOps();
            MnnOps::registerConvOps();
            MnnOps::registerArgmaxOp();
            MnnOps::registerComplexOps();
            registerFallbackAdapter();
            registerBridge(std::unique_ptr<Bridge>(new MnnBridge()));
        }
    } r;
    (void)r;
}

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
