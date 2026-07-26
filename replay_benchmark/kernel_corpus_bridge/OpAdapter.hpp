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
    // Fills ac.entry, ac.args, ac.buffers, ac.globalSize, ac.validator inputs,
    // and (for Vulkan) ac.vulkanBindings / ac.pushConstants / ac.specConstants.
    // ac.source / ac.framework / ac.tag / ac.backend / metadata are already set
    // by the caller. Returns false if the (tag, variant) combination is not
    // supported by this adapter.
    virtual bool adapt(const CaseSpec& spec, AdaptedCase& ac) const = 0;
};

// Registry for OpAdapters within one framework. Lookup by opType.
class OpAdapterRegistry {
public:
    static OpAdapterRegistry& instance();
    void registerAdapter(std::unique_ptr<OpAdapter> adapter);
    const OpAdapter* find(const std::string& opType) const;
private:
    OpAdapterRegistry() = default;
    std::vector<std::unique_ptr<OpAdapter>> mAdapters;
};

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_OP_ADAPTER_HPP
