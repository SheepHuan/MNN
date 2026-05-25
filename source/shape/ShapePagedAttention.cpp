//
//  ShapePagedAttention.cpp
//  MNN
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

class PagedAttentionSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto input = inputs[0], output = outputs[0];
        MNN_ASSERT(input->buffer().dimensions == 4);
        output->buffer().dim[0].extent = input->buffer().dim[0].extent;
        output->buffer().dim[1].extent = input->buffer().dim[1].extent;
        output->buffer().dim[2].extent = input->buffer().dim[2].extent * input->buffer().dim[3].extent;
        output->buffer().dimensions = 3;
        output->buffer().type = input->buffer().type;
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;
        return true;
    }
    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        auto seqLen = static_cast<float>(outputs[0]->length(1));
        auto headDim = static_cast<float>(outputs[0]->length(2));
        float flops = 2 * seqLen * headDim * seqLen + seqLen * seqLen;
        return flops / FLOPS_M;
    }
};

REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(PagedAttentionSizeComputer, OpType_PagedAttention);
#endif

} // namespace MNN
