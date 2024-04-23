/*
 * @Author: Huan Yang
 * @Date: 2024-04-22 01:26:16
 * @LastEditors: Huan Yang
 * @LastEditTime: 2024-04-23 05:47:06
 * @FilePath: /MNN/source/shape/ShapeRfft.cpp
 * @Description:
 *
 * Copyright (c) 2024 by Huan Yang, All Rights Reserved.
 */
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

            // 输出n维度，输出n+1维度，多出的维度分别存实部和虚部。
            outputs[0]->buffer().dimensions = inputs[0]->buffer().dimensions + 1;

            for (int i = 0; i < inputs[0]->buffer().dimensions; i++)
            {
                if (i != tdPtr[0])
                {
                    outputs[0]->buffer().dim[i].extent = inputs[0]->buffer().dim[i].extent;
                }
                else
                {
                    outputs[0]->buffer().dim[i].extent = inputs[0]->buffer().dim[i].extent / 2 + 1;
                }
            }
            outputs[0]->buffer().dim[inputs[0]->buffer().dimensions].extent = 2;
            TensorUtils::getDescribe(outputs[0])
                ->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
            outputs[0]->buffer().type = inputs[0]->getType();

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
