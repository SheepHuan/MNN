#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_MATMUL_OP_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_MATMUL_OP_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// Matmul adapter: handles matmul_buf variants.
// 1.2.0: matmul_buf(GLOBAL_SIZE_2_DIMS, a, b, out, channels=K, channel_blocks=K4, width_blocks=N4)
class MatmulOp : public OpAdapter {
public:
    const char* opType() const override { return "conv"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

void registerMatmulOp();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_MATMUL_OP_HPP
