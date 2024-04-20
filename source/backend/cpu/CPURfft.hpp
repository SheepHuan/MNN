#ifndef CPURfft_hpp
#define CPURfft_hpp

#include "MNN_generated.h"
#include "core/Execution.hpp"

namespace MNN
{
    class CPURfft : public Execution
    {
    public:
        CPURfft(Backend *backend);
        virtual ~CPURfft() = default;
        virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
        virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    };
}
#endif