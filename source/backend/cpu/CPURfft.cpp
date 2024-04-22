/*
 * @Author: Huan Yang
 * @Date: 2024-04-22 01:26:16
 * @LastEditors: Huan Yang
 * @LastEditTime: 2024-04-22 09:34:43
 * @FilePath: /MNN/source/backend/cpu/CPURfft.cpp
 * @Description:
 *
 * Copyright (c) 2024 by Huan Yang, All Rights Reserved.
 */
#include "backend/cpu/CPURfft.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include <complex>
#include <cmath>

namespace MNN
{
    void make_euler_weights(int outSize, int signSize, float *realWeights, float *imagWeights)
    {
        return;
    }

    CPURfft::CPURfft(Backend *b) : Execution(b)
    {
    }
    ErrorCode CPURfft::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)
    {
        // 确定参数
        int *slPtr = inputs[1]->host<int>();
        int *tdPtr = inputs[2]->host<int>();

        this->_signalLength = slPtr[0];
        this->_computeDim = tdPtr[0];
        this->_computeDimSize = inputs[0]->buffer().dim[this->_computeDim].extent;
        this->_resultDimSize = int(this->_computeDimSize / 2) + 1;

        int numberThread = mSupportMultiThread ? ((CPUBackend *)backend())->threadNumber() : 1;

        return NO_ERROR;
    }

    ErrorCode CPURfft::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)
    {
        // NOTE: 这个实现只支持3D输入[n,f,s]
        if (this->implementMode == 0)
        {
            if (inputs[0]->buffer().dimensions == 3)
            {
                int batch = inputs[0]->buffer().dim[0].extent;
                int frameLen = inputs[0]->buffer().dim[1].extent;
                int signalLen = inputs[0]->buffer().dim[2].extent;
                float *realPtr = outputs[0]->host<float>();
                float *imagPtr = outputs[1]->host<float>();
                float *inputPtr = inputs[0]->host<float>();
                if (this->_resultDimSize == -1)
                {
                    int *slPtr = inputs[1]->host<int>();
                    int *tdPtr = inputs[2]->host<int>();

                    this->_signalLength = slPtr[0];
                    this->_computeDim = tdPtr[0];
                    this->_computeDimSize = inputs[0]->buffer().dim[this->_computeDim].extent;
                    this->_resultDimSize = int(this->_computeDimSize / 2) + 1;
                }
                executeRfft3d(0, this->_resultDimSize, batch, frameLen, this->_resultDimSize, signalLen, inputPtr, realPtr, imagPtr);
            }
            else
            {
                return ErrorCode::INPUT_DATA_ERROR;
            }
        }
        else
        {
            return ErrorCode::NOT_SUPPORT;
        }
        return NO_ERROR;
    }

    void CPURfft::executeRfft3d(int startIdx, int endIdx, int batch, int frameLen, int resSize, int signalSize, float *inputPtr, float *realPtr, float *imagPtr)
    {
        for (int i = 0; i < batch; i++)
        {
            for (int j = 0; j < frameLen; j++)
            {

                for (int k = startIdx; k < endIdx; k++)
                {
                    int index = i * frameLen * resSize + j * resSize + k;
                    realPtr[index] = 0.0;
                    imagPtr[index] = 0.0;
                    for (int n = 0; n < signalSize; n++)
                    {
                        int inputIdx = i * frameLen * signalSize + j * signalSize + n;
                        double angle = -2.0 * M_PI * k * n / signalSize;
                        double realWeight = cos(angle);
                        double imagWeight = sin(angle);
                        // outputs?
                        realPtr[index] += inputPtr[inputIdx] * realWeight;
                        imagPtr[index] += inputPtr[inputIdx] * imagWeight;
                    }
                }
            }
        }
    }

    class CPURfftCreator : public CPUBackend::Creator
    {
    public:
        virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                    const MNN::Op *op, Backend *backend) const override
        {

            return new CPURfft(backend);
        }
    };
    REGISTER_CPU_OP_CREATOR(CPURfftCreator, OpType_Rfft);
} // namespace MNN
