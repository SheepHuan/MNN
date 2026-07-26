#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_NCNN_BRIDGE_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_NCNN_BRIDGE_HPP

#include "../Bridge.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {

// NcnnBridge: framework-level bridge for ncnn. Supports tags 20190611 and 20260526.
// Dispatches to per-operator OpAdapter. Loads pre-compiled SPIR-V (.spv) for
// each .comp source so device runs don't need glslangValidator.
class NcnnBridge : public Bridge {
public:
    bool supports(const std::string& framework, const std::string& tag) const override;
    AdaptedCase adapt(const CaseSpec& spec,
                      const std::string& sourceText,
                      const std::string& sourceFile,
                      const std::string& corpusRoot) const override;
};

void registerNcnnBridge();

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_NCNN_NCNN_BRIDGE_HPP
