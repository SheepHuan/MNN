#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_CUDA_OP_ADAPTER_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_CUDA_OP_ADAPTER_HPP

#include <cuda_runtime.h>
#include <vector>
#include "../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {

// Context passed to CudaOpAdapter::launch(). The runner allocates and fills
// device buffers (devBufs, indexed by AdaptedArg.bufferIndex), collects the
// scalar args from AdaptedCase.args (intArgs/floatArgs, host-side, in the
// order they appear in ac.args), and sets grid/block/stream. The adapter
// packs these into the shim's C signature and calls it.
struct CudaLaunchCtx {
    std::vector<void*> devBufs;
    int grid = 1;
    int block = 128;
    cudaStream_t stream = nullptr;
    std::vector<int> intArgs;
    std::vector<float> floatArgs;
};

// CUDA-specific adapter interface. Adds launch() which the CUDA runner calls
// after allocating device buffers and before output readback. The adapter
// knows the shim signature and performs the <<<>>> launch internally.
class CudaOpAdapter : public OpAdapter {
public:
    bool isCuda() const override { return true; }
    void* asCuda() override { return this; }
    const void* asCuda() const override { return this; }
    // Launch the kernel. Returns cudaSuccess on success.
    virtual cudaError_t launch(const AdaptedCase& ac, const CudaLaunchCtx& ctx) const = 0;
};

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_CUDA_OP_ADAPTER_HPP
