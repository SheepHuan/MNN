//
//  ShapeBinaryOp.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include <cstdlib>
#include <string>
#include <vector>
namespace MNN {
class BinaryOpComputer : public SizeComputer {
public:
    static bool outputBool(int operation) {
        if (operation == BinaryOpOperation_GREATER_EQUAL) {
            return true;
        }
        if (operation == BinaryOpOperation_GREATER) {
            return true;
        }
        if (operation == BinaryOpOperation_LESS) {
            return true;
        }
        if (operation == BinaryOpOperation_LESS_EQUAL) {
            return true;
        }
        if (operation == BinaryOpOperation_EQUAL) {
            return true;
        }
        if (operation == BinaryOpOperation_NOTEQUAL) {
            return true;
        }
        return false;
    }
    virtual bool onComputeSize(const Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        // set output type & format
        auto input0 = inputs[0], input1 = inputs[1], output = outputs[0];
        auto &buffer = output->buffer();
        const auto opType = op->main_as_BinaryOp()->opType();
        if (outputBool(opType)) {
            buffer.type = halide_type_of<int32_t>();
        } else {
            buffer.type = input0->getType();
        }

        if (input0->getType().code != input1->getType().code) {
#ifdef DEBUG
            MNN_PRINT("Error for binary op: input0's type != input1's type, %d != %d, optype:%d, ", input0->getType().code, input1->getType().code, opType);
            if (nullptr != op->name()) {
                MNN_PRINT("op name: %s", op->name()->c_str());
            }
            MNN_PRINT("\n");
#endif
            return false;
        }

        if (input0->dimensions() < input1->dimensions()) {
            auto temp = input0;
            input0 = input1;
            input1 = temp;
        }
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input0)->dimensionFormat;
        bool res = SizeComputer::computeBroadCastDims(inputs, outputs);
        const char* opName = op->name() != nullptr ? op->name()->c_str() : "";
        if (std::getenv("MNN_PIC_GRAPH_PROFILE") != nullptr &&
            (std::string(opName).find("/layers.0/Add") != std::string::npos ||
             std::string(opName).find("/blocks.0/Add") != std::string::npos)) {
            std::fprintf(stderr,
                         "PicBinary shape name=%s res=%d in0_dims=%d in0=%d,%d,%d,%d in1_dims=%d in1=%d,%d,%d,%d out_dims=%d out=%d,%d,%d,%d format=%d\n",
                         opName, res ? 1 : 0,
                         inputs[0]->dimensions(),
                         inputs[0]->dimensions() > 0 ? inputs[0]->length(0) : -1,
                         inputs[0]->dimensions() > 1 ? inputs[0]->length(1) : -1,
                         inputs[0]->dimensions() > 2 ? inputs[0]->length(2) : -1,
                         inputs[0]->dimensions() > 3 ? inputs[0]->length(3) : -1,
                         inputs[1]->dimensions(),
                         inputs[1]->dimensions() > 0 ? inputs[1]->length(0) : -1,
                         inputs[1]->dimensions() > 1 ? inputs[1]->length(1) : -1,
                         inputs[1]->dimensions() > 2 ? inputs[1]->length(2) : -1,
                         inputs[1]->dimensions() > 3 ? inputs[1]->length(3) : -1,
                         output->dimensions(),
                         output->dimensions() > 0 ? output->length(0) : -1,
                         output->dimensions() > 1 ? output->length(1) : -1,
                         output->dimensions() > 2 ? output->length(2) : -1,
                         output->dimensions() > 3 ? output->length(3) : -1,
                         static_cast<int>(TensorUtils::getDescribe(output)->dimensionFormat));
            std::fflush(stderr);
        }
        return res;
    }
};

REGISTER_SHAPE(BinaryOpComputer, OpType_BinaryOp);
} // namespace MNN
