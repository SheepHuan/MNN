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
// order they appear in ac.args), and sets grid/block/stream FROM
// ac.globalSize[0]/ac.localSize[0] (set by the adapter's adapt()). The adapter
// packs these into the shim's C signature and calls it.
//
// grid/block have NO defaults — every adapter's adapt() MUST set
// ac.globalSize[0] and ac.localSize[0] explicitly, branching by spec.tag to
// match real MNN's per-version launch policy:
//   - 1.2.0: adaptive block ladder via mnnBlock120() (maxThreadsPerBlock-based)
//   - 1.2.1+ and 3.6.0: fixed block = 128 (kBlock)
//   - single-block kernels (TopKV2): compute their own geometry in adapt()
// See CudaOps.hpp for the mnnBlock120()/mnnGridFor() helpers.
struct CudaLaunchCtx {
    std::vector<void*> devBufs;
    int grid = 0;     // set by runner from ac.globalSize[0] (no default)
    int block = 0;    // set by runner from ac.localSize[0] (no default)
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
