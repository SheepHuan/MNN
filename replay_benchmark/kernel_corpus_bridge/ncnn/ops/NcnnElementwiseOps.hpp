#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_OPS_NCNN_ELEMENTWISE_OPS_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_OPS_NCNN_ELEMENTWISE_OPS_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace NcnnOps {

// ---- 20190611 scalar (sfp) elementwise — in-place, push_constant {dims,w,h,c,cstep} ----
// absval, relu, sigmoid, tanh already have dedicated adapters.
// These handle: clip, prelu, dropout, eltwise, unaryop

class VulkanClipKernel : public OpAdapter {
public:
    const char* opType() const override { return "clip"; }
    const char* variant() const override { return "clip"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanClipPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "clip"; }
    const char* variant() const override { return "clip_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanPreluKernel : public OpAdapter {
public:
    const char* opType() const override { return "prelu"; }
    const char* variant() const override { return "prelu"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanPreluPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "prelu"; }
    const char* variant() const override { return "prelu_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanDropoutKernel : public OpAdapter {
public:
    const char* opType() const override { return "dropout"; }
    const char* variant() const override { return "dropout"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanDropoutPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "dropout"; }
    const char* variant() const override { return "dropout_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanEltwiseKernel : public OpAdapter {
public:
    const char* opType() const override { return "eltwise"; }
    const char* variant() const override { return "eltwise"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanEltwisePack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "eltwise"; }
    const char* variant() const override { return "eltwise_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanUnaryopKernel : public OpAdapter {
public:
    const char* opType() const override { return "unaryop"; }
    const char* variant() const override { return "unaryop"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanUnaryopPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "unaryop"; }
    const char* variant() const override { return "unaryop_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

// ---- 20260526 pack4 elementwise — in-place, push_constant {uint n}, spec_const n ----
// absval/sigmoid/tanh/relu already have dedicated adapters.
// These handle: clip, celu, elu, erf, gelu, hardsigmoid, hardswish, mish, selu, shrink, softplus, swish

class VulkanClip2026Kernel : public OpAdapter {
public:
    const char* opType() const override { return "clip"; }
    const char* variant() const override { return "clip"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanCeluKernel : public OpAdapter {
public:
    const char* opType() const override { return "celu"; }
    const char* variant() const override { return "celu"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanEluKernel : public OpAdapter {
public:
    const char* opType() const override { return "elu"; }
    const char* variant() const override { return "elu"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanErfKernel : public OpAdapter {
public:
    const char* opType() const override { return "erf"; }
    const char* variant() const override { return "erf"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanGeluKernel : public OpAdapter {
public:
    const char* opType() const override { return "gelu"; }
    const char* variant() const override { return "gelu"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanHardsigmoidKernel : public OpAdapter {
public:
    const char* opType() const override { return "hardsigmoid"; }
    const char* variant() const override { return "hardsigmoid"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanHardswishKernel : public OpAdapter {
public:
    const char* opType() const override { return "hardswish"; }
    const char* variant() const override { return "hardswish"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanMishKernel : public OpAdapter {
public:
    const char* opType() const override { return "mish"; }
    const char* variant() const override { return "mish"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanSeluKernel : public OpAdapter {
public:
    const char* opType() const override { return "selu"; }
    const char* variant() const override { return "selu"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanShrinkKernel : public OpAdapter {
public:
    const char* opType() const override { return "shrink"; }
    const char* variant() const override { return "shrink"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanSoftplusKernel : public OpAdapter {
public:
    const char* opType() const override { return "softplus"; }
    const char* variant() const override { return "softplus"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanSwishKernel : public OpAdapter {
public:
    const char* opType() const override { return "swish"; }
    const char* variant() const override { return "swish"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

// ---- 20260526 scalar elementwise (no pack4) ----
class VulkanSigmoid2019Kernel : public OpAdapter {
public:
    const char* opType() const override { return "sigmoid"; }
    const char* variant() const override { return "sigmoid"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanSigmoidPack4Kernel2 : public OpAdapter {
public:
    const char* opType() const override { return "sigmoid"; }
    const char* variant() const override { return "sigmoid_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanTanh2019Kernel : public OpAdapter {
public:
    const char* opType() const override { return "tanh"; }
    const char* variant() const override { return "tanh"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanTanhPack4Kernel2 : public OpAdapter {
public:
    const char* opType() const override { return "tanh"; }
    const char* variant() const override { return "tanh_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanAbsval2019Kernel : public OpAdapter {
public:
    const char* opType() const override { return "absval"; }
    const char* variant() const override { return "absval"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanAbsvalPack4Kernel2 : public OpAdapter {
public:
    const char* opType() const override { return "absval"; }
    const char* variant() const override { return "absval_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanRelu2019Kernel : public OpAdapter {
public:
    const char* opType() const override { return "relu"; }
    const char* variant() const override { return "relu"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanReluPack4Kernel2 : public OpAdapter {
public:
    const char* opType() const override { return "relu"; }
    const char* variant() const override { return "relu_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

// ---- binaryop (20190611 + 20260526) ----
class VulkanBinaryopKernel : public OpAdapter {
public:
    const char* opType() const override { return "binaryop"; }
    const char* variant() const override { return "binaryop"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanBinaryopPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "binaryop"; }
    const char* variant() const override { return "binaryop_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanDropout2026Kernel : public OpAdapter {
public:
    const char* opType() const override { return "dropout"; }
    const char* variant() const override { return "dropout"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanEltwise2026Kernel : public OpAdapter {
public:
    const char* opType() const override { return "eltwise"; }
    const char* variant() const override { return "eltwise"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanUnaryop2026Kernel : public OpAdapter {
public:
    const char* opType() const override { return "unaryop"; }
    const char* variant() const override { return "unaryop"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

// ---- Deepcopy ----
class VulkanDeepcopyKernel : public OpAdapter {
public:
    const char* opType() const override { return "deepcopy"; }
    const char* variant() const override { return "deepcopy"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

class VulkanDeepcopyPack4Kernel : public OpAdapter {
public:
    const char* opType() const override { return "deepcopy"; }
    const char* variant() const override { return "deepcopy_pack4"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

void registerNcnnElementwiseOps();

} // namespace NcnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif
