#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_REDUCTION_OP_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_REDUCTION_OP_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// Reduction adapter: handles reduct_buf (sum variant).
// reduct_buf(GLOBAL_SIZE_2_DIMS, input, output, batch, height, width)
class ReductionOp : public OpAdapter {
public:
    const char* opType() const override { return "reduction"; }
    const char* variant() const override { return "reduct_buf_sum_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

void registerReductionOp();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_REDUCTION_OP_HPP
