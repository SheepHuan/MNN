#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_MNN_BRIDGE_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_MNN_BRIDGE_HPP

#include "../Bridge.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {

// MnnBridge: framework-level bridge for MNN. Supports tags 1.2.0 and 3.6.0.
// Dispatches to per-operator OpAdapter registered in OpAdapterRegistry.
// The tag-specific parameter-layout differences live inside each OpAdapter.
class MnnBridge : public Bridge {
public:
    bool supports(const std::string& framework, const std::string& tag) const override;
    AdaptedCase adapt(const CaseSpec& spec,
                      const std::string& sourceText,
                      const std::string& sourceFile,
                      const std::string& corpusRoot) const override;
};

// Force registration of MNN OpAdapters and the bridge. Idempotent.
void registerMnnBridge();

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_MNN_BRIDGE_HPP
