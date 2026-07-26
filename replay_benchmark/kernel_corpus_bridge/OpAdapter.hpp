#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_OP_ADAPTER_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_OP_ADAPTER_HPP

#include "Bridge.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {

// OpAdapter: per-operator adapter that knows the operator's parameter layout,
// data preparation and dispatch geometry. Version-specific differences are
// expressed through the `tag` field of CaseSpec and handled inside adapt().
//
// An OpAdapter is stateless and registered per-framework. The framework
// bridge dispatches to the matching OpAdapter by opType.
class OpAdapter {
public:
    virtual ~OpAdapter() = default;
    // Returns the opType this adapter handles (e.g. "raster", "unary", "sigmoid").
    virtual const char* opType() const = 0;
    // Returns the exact variant name this adapter handles (e.g.
    // "buffer_set_zero_fp32", "unary_buf_exp_fp32"). An adapter claims exactly
    // one variant so multiple adapters can share an opType without conflict.
    virtual const char* variant() const = 0;
    // Fills ac.entry, ac.args, ac.buffers, ac.globalSize, ac.validator inputs,
    // and (for Vulkan) ac.vulkanBindings / ac.pushConstants / ac.specConstants.
    // ac.source / ac.framework / ac.tag / ac.backend / metadata are already set
    // by the caller. Returns false if the tag is not supported by this adapter.
    virtual bool adapt(const CaseSpec& spec, AdaptedCase& ac) const = 0;
};

// Registry for OpAdapters within one framework. Lookup by opType.
class OpAdapterRegistry {
public:
    static OpAdapterRegistry& instance();
    void registerAdapter(std::unique_ptr<OpAdapter> adapter);
    // Returns all adapters matching opType (multiple may exist for different tags).
    const std::vector<std::unique_ptr<OpAdapter>>& adapters() const { return mAdapters; }
private:
    OpAdapterRegistry() = default;
    std::vector<std::unique_ptr<OpAdapter>> mAdapters;
};

// Find the adapter matching (opType, variant) whose adapt() succeeds.
// Falls back to FallbackAdapter if no dedicated adapter matches.
inline const OpAdapter* findAdapter(const CaseSpec& spec, AdaptedCase& ac) {
    for (const auto& a : OpAdapterRegistry::instance().adapters()) {
        if (spec.opType == a->opType() && spec.variant == a->variant() && a->adapt(spec, ac)) {
            return a.get();
        }
    }
    // Fallback: try the generic adapter
    for (const auto& a : OpAdapterRegistry::instance().adapters()) {
        if (std::string(a->opType()) == "__fallback__" && a->adapt(spec, ac)) {
            return a.get();
        }
    }
    return nullptr;
}

// Fallback adapter: tries to compile the kernel with minimal parameters
// (single in/out buffer, 1D dispatch). Used for operators without a
// dedicated adapter — at least validates compilation.
class FallbackAdapter : public OpAdapter {
public:
    const char* opType() const override { return "__fallback__"; }
    const char* variant() const override { return "__fallback__"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

void registerFallbackAdapter();

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_OP_ADAPTER_HPP
