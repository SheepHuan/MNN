#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_POOLING_OP_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_POOLING_OP_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// Pooling adapter: handles pooling (max variant, 2x2 stride 2 no pad).
// pooling(GLOBAL_SIZE_3_DIMS, input, int2 input_shape, int2 output_shape,
//        int2 pad_shape, int2 stride_shape, int2 kernel_shape, output, channel_block)
class PoolingOp : public OpAdapter {
public:
    const char* opType() const override { return "pooling"; }
    const char* variant() const override { return "pooling_max_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

void registerPoolingOp();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_POOLING_OP_HPP
