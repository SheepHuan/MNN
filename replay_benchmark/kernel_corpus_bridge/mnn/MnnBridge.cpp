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

namespace MNN {
namespace Replay {
namespace KernelCorpus {

bool MnnBridge::supports(const std::string& framework, const std::string& tag) const {
    if (framework != "mnn") return false;
    return tag == "1.2.0" || tag == "3.6.0";
}

AdaptedCase MnnBridge::adapt(const CaseSpec& spec,
                              const std::string& sourceText,
                              const std::string& sourceFile,
                              const std::string& corpusRoot) const {
    (void)sourceFile;
    (void)corpusRoot;
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
    ac.source = sourceText;

    const OpAdapter* adapter = findAdapter(spec, ac);
    if (adapter == nullptr) {
        ac.entry.clear();
        ac.validator.clear();
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
