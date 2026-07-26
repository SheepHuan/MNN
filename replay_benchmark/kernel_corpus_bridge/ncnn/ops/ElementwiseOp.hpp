#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_OPS_ELEMENTWISE_OP_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_OPS_ELEMENTWISE_OP_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace NcnnOps {

// Shared elementwise adaptation for in-place unary shaders (sigmoid, tanh).
// push_constant: {dims, w, h, c, cstep}
void adaptElementwise(const CaseSpec& spec, AdaptedCase& ac);

class SigmoidOp : public OpAdapter {
public:
    const char* opType() const override { return "sigmoid"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override { adaptElementwise(spec, ac); return true; }
};

class TanhOp : public OpAdapter {
public:
    const char* opType() const override { return "tanh"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override { adaptElementwise(spec, ac); return true; }
};

// Permute adapter: handles permute with order_type specialization constant.
class PermuteOp : public OpAdapter {
public:
    const char* opType() const override { return "permute"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

void registerElementwiseOp();
void registerPermuteOp();

} // namespace NcnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_OPS_ELEMENTWISE_OP_HPP
