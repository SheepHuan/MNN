#include "RasterOp.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

bool RasterOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "buffer_set_zero";
    const int count = spec.elementCount > 0 ? spec.elementCount : 1024;
    AdaptedBuffer outBuf;
    outBuf.sizeBytes = count * sizeof(float);
    outBuf.isOutput = true;
    ac.buffers.push_back(outBuf);
    // args: dim0=count, dim1=1, output
    ac.args.push_back(AdaptedArg::sizeConst(0));
    ac.args.push_back(AdaptedArg::sizeConst(1));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.globalSize[0] = count;
    ac.globalSize[1] = 1;
    ac.dims = 2;
    return true;
}

void registerRasterOp() {
    static struct Reg {
        Reg() { OpAdapterRegistry::instance().registerAdapter(std::unique_ptr<OpAdapter>(new RasterOp())); }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
