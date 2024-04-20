#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN
{
    class RfftComputer : public SizeComputer
    {
    public:
        virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                                   const std::vector<Tensor *> &outputs) const override
        {
            int *slPtr = inputs[1]->host<int>();
            int *tdPtr = inputs[2]->host<int>();

            outputs[0]->buffer().dimensions = inputs[0]->buffer().dimensions;
            outputs[1]->buffer().dimensions = inputs[0]->buffer().dimensions;

            for (int i = 0; i < inputs[0]->buffer().dimensions; i++)
            {
                if (i != tdPtr[0])
                {
                    outputs[0]->buffer().dim[i].extent = inputs[0]->buffer().dim[i].extent;
                    outputs[1]->buffer().dim[i].extent = inputs[0]->buffer().dim[i].extent;
                }
                else
                {
                    outputs[0]->buffer().dim[i].extent = inputs[0]->buffer().dim[i].extent / 2 + 1;
                    outputs[1]->buffer().dim[i].extent = inputs[0]->buffer().dim[i].extent / 2 + 1;
                }
            }
            TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
            TensorUtils::getDescribe(outputs[1])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
            outputs[0]->buffer().type = inputs[0]->getType();
            outputs[1]->buffer().type = inputs[0]->getType();
            return true;
        }
        virtual float onComputeFlops(const MNN::Op *op,
                                     const std::vector<Tensor *> &inputs,
                                     const std::vector<Tensor *> &outputs) const
        {
            return 0.0;
        }
    };
    REGISTER_SHAPE(RfftComputer, OpType_Rfft);
}
