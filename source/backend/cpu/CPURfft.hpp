/*
 * @Author: Huan Yang
 * @Date: 2024-04-22 01:26:16
 * @LastEditors: Huan Yang
 * @LastEditTime: 2024-04-23 05:54:11
 * @FilePath: /MNN/source/backend/cpu/CPURfft.hpp
 * @Description:
 *
 * Copyright (c) 2024 by Huan Yang, All Rights Reserved.
 */
#ifndef CPURfft_hpp
#define CPURfft_hpp

#include "MNN_generated.h"
#include "core/Execution.hpp"

#include "fftw3.h"

namespace MNN
{
    uint64_t reverseBits(uint64_t n, uint64_t bits);
    void arraryReorder3D(float *input, int n, int c, int hw);

    class CPURfft : public Execution
    {
    public:
        CPURfft(Backend *backend);
        virtual ~CPURfft();
        virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
        virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
        void executeRfft3d(int startIdx, int endIdx, int batch, int frameLen, int resDim, int inputDim, float *input, float *output);

    protected:
        bool mSupportMultiThread = false;
        int implementMode = 0;

    private:
        std::vector<std::pair<std::function<void(int, int, const float *, const float *, const float *)>, int>> mPreFunctions;

        int _signalLength = -1;
        int _computeDim = -1;     // 要处理输入张量的哪一个维度
        int _computeDimSize = -1; // 处理输入张量的计算维度的大小
        int _resultDimSize = -1;  // 输出张量计算维度的大小

        float *_cache_in;
        fftwf_complex *_cache_out;
        fftwf_plan _fft_plan;
        Tensor *_cache_out_tensor;
    };
}
#endif