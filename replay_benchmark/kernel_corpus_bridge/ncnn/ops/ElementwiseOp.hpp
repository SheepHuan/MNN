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

// Sigmoid adapter for ncnn 20190611 (scalar sfp, in-place).
class SigmoidOp : public OpAdapter {
public:
    const char* opType() const override { return "sigmoid"; }
    const char* variant() const override { return "sigmoid_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override {
        if (spec.tag != "20190611") return false;
        adaptElementwise(spec, ac); return true;
    }
};

// Tanh adapter for ncnn 20190611 (scalar sfp, in-place).
class TanhOp : public OpAdapter {
public:
    const char* opType() const override { return "tanh"; }
    const char* variant() const override { return "tanh_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override {
        if (spec.tag != "20190611") return false;
        adaptElementwise(spec, ac); return true;
    }
};

// Permute adapter: handles permute with order_type specialization constant.
class PermuteOp : public OpAdapter {
public:
    const char* opType() const override { return "permute"; }
    const char* variant() const override { return "permute_order0_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

void registerElementwiseOp();
void registerPermuteOp();

} // namespace NcnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_OPS_ELEMENTWISE_OP_HPP
