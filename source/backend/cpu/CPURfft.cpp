#include "backend/cpu/CPURfft.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"

namespace MNN
{
    CPURfft::CPURfft(Backend *b) : Execution(b)
    {
    }
    ErrorCode CPURfft::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)
    {
        // 要根据输入的形状，在这里预先计算欧拉权重？

        // 确定rfft的计算维度、输入大小、输出大小
        int *tdPtr = inputs[2]->host<int>();
        int targetDim = tdPtr[0];
        int inDimSize = inputs[0]->buffer().dim[targetDim].extent;
        int outDimSize = inDimSize / 2 + 1;



        return NO_ERROR;
    }

    ErrorCode CPURfft::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)
    {

        return NO_ERROR;
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
