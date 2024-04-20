#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(RfftOnnx);

MNN::OpType RfftOnnx::opType()
{
    return MNN::OpType_Rfft;
}
MNN::OpParameter RfftOnnx::type()
{
    // No parameter
    return MNN::OpParameter_NONE;
}
void RfftOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, OnnxScope *scope)
{
    return;
}

REGISTER_CONVERTER(RfftOnnx, Rfft);