#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_RASTER_OP_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_RASTER_OP_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// Raster adapter: handles buffer_set_zero across MNN tags.
// buffer_set_zero(GLOBAL_SIZE_2_DIMS, output) — layout is stable across
// 1.2.0 and 3.6.0 (the only difference is FLOAT vs OUTPUT_TYPE typedef,
// which is already baked into the source file).
class RasterOp : public OpAdapter {
public:
    const char* opType() const override { return "raster"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

void registerRasterOp();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_RASTER_OP_HPP
