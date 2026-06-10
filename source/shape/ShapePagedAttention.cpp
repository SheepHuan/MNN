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
        int queryLen = input->buffer().dim[1].extent;
        if (inputs.size() > 4 && inputs[4] != nullptr && inputs[4]->elementSize() > 0) {
            auto budget = inputs[4]->host<int32_t>();
            if (budget != nullptr && budget[0] > 0) {
                queryLen = budget[0];
            }
        }
        output->buffer().dim[0].extent = input->buffer().dim[0].extent;
        output->buffer().dim[1].extent = queryLen;
        output->buffer().dim[2].extent = input->buffer().dim[2].extent * input->buffer().dim[3].extent;
        output->buffer().dimensions = 3;
        output->buffer().type = input->buffer().type;
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;
        if (outputs.size() > 1 && outputs[1] != nullptr) {
            auto indexOutput = outputs[1];
            indexOutput->buffer().dimensions = 1;
            indexOutput->buffer().dim[0].extent = queryLen;
            indexOutput->buffer().type = halide_type_of<int32_t>();
            TensorUtils::getDescribe(indexOutput)->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;
        }
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

void ___PagedAttentionSizeComputer__OpType_PagedAttention__() {
    auto computer = new PagedAttentionSizeComputer;
    computer->setInputIndex(std::vector<int>{4});
    SizeComputerSuite::get()->insert(computer, OpType_PagedAttention);
}
void ___PagedAttentionSizeComputer__OpType_PicScoreAttention__() {
    auto computer = new PagedAttentionSizeComputer;
    computer->setInputIndex(std::vector<int>{4});
    SizeComputerSuite::get()->insert(computer, OpType_PicScoreAttention);
}
void ___PagedAttentionSizeComputer__OpType_PicSparseAttention__() {
    auto computer = new PagedAttentionSizeComputer;
    computer->setInputIndex(std::vector<int>{4});
    SizeComputerSuite::get()->insert(computer, OpType_PicSparseAttention);
}
#endif

} // namespace MNN
