/*
 * @Author: Huan Yang
 * @Date: 2024-04-22 01:26:16
 * @LastEditors: Huan Yang
 * @LastEditTime: 2024-04-22 02:47:29
 * @FilePath: /MNN/source/backend/cpu/CPURfft.hpp
 * @Description:
 *
 * Copyright (c) 2024 by Huan Yang, All Rights Reserved.
 */
#ifndef CPURfft_hpp
#define CPURfft_hpp

#include "MNN_generated.h"
#include "core/Execution.hpp"

namespace MNN
{
    void make_euler_weights(int outSize, int signSize, float *realWeights, float *imagWeights);

    class CPURfft : public Execution
    {
    public:
        CPURfft(Backend *backend);
        virtual ~CPURfft() = default;
        virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
        virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    protected:
        bool mSupportMultiThread = false;
        int implementMode = 0;

    private:
        int _signalLength = -1;
        int _computeDim = -1;     // 要处理输入张量的哪一个维度
        int _computeDimSize = -1; // 处理输入张量的计算维度的大小
        int _resultDimSize = -1;  // 输出张量计算维度的大小
    };
}
#endif