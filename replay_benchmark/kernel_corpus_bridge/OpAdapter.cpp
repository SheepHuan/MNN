#include "OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {

OpAdapterRegistry& OpAdapterRegistry::instance() {
    static OpAdapterRegistry reg;
    return reg;
}

void OpAdapterRegistry::registerAdapter(std::unique_ptr<OpAdapter> adapter) {
    mAdapters.push_back(std::move(adapter));
}

const OpAdapter* OpAdapterRegistry::find(const std::string& opType) const {
    for (const auto& a : mAdapters) {
        if (opType == a->opType()) return a.get();
    }
    return nullptr;
}

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
