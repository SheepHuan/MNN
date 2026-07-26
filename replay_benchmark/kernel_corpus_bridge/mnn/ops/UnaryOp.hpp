#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_UNARY_OP_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_UNARY_OP_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// Unary adapter: handles unary_buf_exp variant.
// 1.2.0: unary_buf(GLOBAL_SIZE_3_DIMS, input, output, height)
// 3.6.0: unary_buf(GLOBAL_SIZE_2_DIMS, input, output, size)
class UnaryBufExpOp : public OpAdapter {
public:
    const char* opType() const override { return "unary"; }
    const char* variant() const override { return "unary_buf_exp_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

void registerUnaryOp();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_UNARY_OP_HPP
