/*
 * @Author: Huan Yang
 * @Date: 2024-04-22 01:26:16
 * @LastEditors: Huan Yang
 * @LastEditTime: 2024-04-23 05:59:16
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

    CPURfft::CPURfft(Backend *b) : Execution(b)
    {
    }

    CPURfft::~CPURfft()
    {
        // if (this->cache_out1d != nullptr)
        // {
        //     fftwf_free(this->cache_out1d);
        // }
    }

    ErrorCode
    CPURfft::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)
    {
        // 确定参数
        int *slPtr = inputs[1]->host<int>();
        int *tdPtr = inputs[2]->host<int>();

        this->_signalLength = slPtr[0];
        this->_computeDim = tdPtr[0];
        this->_computeDimSize = inputs[0]->buffer().dim[this->_computeDim].extent;
        this->_resultDimSize = int(this->_computeDimSize / 2) + 1;

        int numberThread = mSupportMultiThread ? ((CPUBackend *)backend())->threadNumber() : 1;

        // 3D处理
        int batch = inputs[0]->buffer().dim[0].extent;
        int frameLen = inputs[0]->buffer().dim[1].extent;
        int signalLen = inputs[0]->buffer().dim[2].extent;
        int flag =
            0 | FFTW_PRESERVE_INPUT | FFTW_ESTIMATE;
        _cache_in = fftwf_alloc_real(inputs[0]->size());
        _cache_out = fftwf_alloc_complex(inputs[0]->size() / signalLen * (signalLen / 2 + 1));

        _fft_plan = fftwf_plan_dft_r2c_3d(batch, frameLen, signalLen, _cache_in, _cache_out, flag);
        // const std::vector<int> shape = {batch,
        //                                 frameLen,
        //                                 int(signalLen / 2 + 1),
        //                                 2};
        // _cache_out_tensor = Tensor::create(shape, inputs[0]->getType(), (void *)_cache_out, inputs[0]->getDimensionType());
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
                float *outputPtr = outputs[0]->host<float>();
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
                executeRfft3d(0, this->_resultDimSize, batch, frameLen, this->_resultDimSize, signalLen, inputPtr, outputPtr);
            }
            else
            {
                return ErrorCode::INPUT_DATA_ERROR;
            }
        }
        else if (this->implementMode == 1)
        {
            if (inputs[0]->buffer().dimensions == 3)
            {
                // int batch = inputs[0]->buffer().dim[0].extent;
                // int frameLen = inputs[0]->buffer().dim[1].extent;
                // int signalLen = inputs[0]->buffer().dim[2].extent;
                // float *outputPtr = outputs[0]->host<float>();
                // float *inputPtr = inputs[0]->host<float>();
                // int max_iter = batch * frameLen;

                // // 拷贝
                // // memcpy(_cache_in, inputs[0]->host<float>(), inputs[0]->size());
                // for (int i = 100; i < 105; i++)
                // {
                //     printf("%f\n", _cache_in[i]);
                // }
                // // fftwf_execute(_fft_plan);
                // fftwf_execute_dft_r2c(_fft_plan, inputPtr, (fftw_complex *));
                // // 张量合并 3个257x2的合并成
                // for (int i = 100; i < 105; i++)
                // {
                //     printf("%f\n", _cache_out[i]);
                // }
            }
        }
        else
        {
            return ErrorCode::NOT_SUPPORT;
        }
        return NO_ERROR;
    }

    void CPURfft::executeRfft3d(int startIdx, int endIdx, int batch, int frameLen, int resSize, int signalSize, float *inputPtr, float *outputPtr)
    {
        for (int i = 0; i < batch; i++)
        {
            for (int j = 0; j < frameLen; j++)
            {

                for (int k = startIdx; k < endIdx; k++)
                {
                    int index = i * frameLen * resSize * 2 + j * resSize * 2 + k * 2;
                    outputPtr[index] = 0.0;
                    outputPtr[index + 1] = 0.0;
                    for (int n = 0; n < signalSize; n++)
                    {
                        int inputIdx = i * frameLen * signalSize + j * signalSize + n;
                        double angle = -2.0 * M_PI * k * n / signalSize;
                        double realWeight = cos(angle);
                        double imagWeight = sin(angle);
                        // outputs?
                        outputPtr[index] += inputPtr[inputIdx] * realWeight;
                        outputPtr[index + 1] += inputPtr[inputIdx] * imagWeight;
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
