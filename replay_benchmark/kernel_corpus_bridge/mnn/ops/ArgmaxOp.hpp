#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_ARGMAX_OP_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_ARGMAX_OP_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// argmax_buf 3.6.0: (dim0, dim1, FLOAT* in, FLOAT* out, int inside, int outside, int dim)
class OpenCLArgmaxKernel : public OpAdapter {
public:
    const char* opType() const override { return "argmax"; }
    const char* variant() const override { return "argmax_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;

    // Smoke test: kernel compiled + dispatched. Precise validation not yet implemented.
    bool validate(const AdaptedCase& ac, const std::vector<float>& output) const override { (void)ac; (void)output; return true; }
};

void registerArgmaxOp();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif
