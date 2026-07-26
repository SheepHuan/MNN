#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_BINARY_OP_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_MNN_OPS_BINARY_OP_HPP

#include "../../OpAdapter.hpp"

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// binary_buf 1.2.0: (int dim0, int dim1, FLOAT* in0, FLOAT* in1, FLOAT* out, int4 shape, int2 isFull)
// binary_buf 3.6.0: (int dim0, int dim1, INPUT* in0, INPUT* in1, OUTPUT* out, int size, int actType)
class BinaryBufOp : public OpAdapter {
public:
    const char* opType() const override { return "binary"; }
    const char* variant() const override { return "binary_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

void registerBinaryOp();

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif
