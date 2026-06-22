//
//  FuseExecution.cpp
//  MNN
//
//  Created by MNN on 2022/11/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/FuseExecution.hpp"
#include "backend/opencl/execution/image/ConvExecution.hpp"
#include "backend/opencl/execution/image/UnaryExecution.hpp"
#include "backend/opencl/execution/image/EltwiseExecution.hpp"
#include "core/Macro.h"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

static int picSiluMulFeatureDim(const Tensor* output) {
    if (output == nullptr || output->dimensions() <= 0) {
        return 0;
    }
    if (output->dimensions() == 3) {
        return output->length(2);
    }
    return output->channel();
}

static int picSiluMulActiveRows(const Tensor* output) {
    const int channel = picSiluMulFeatureDim(output);
    const int count = output->elementSize();
    if (channel > 0 && count % channel == 0) {
        return count / channel;
    }
    return count;
}

static bool usePicSiluMulFusedKernel(const Tensor* output) {
    constexpr int kMaxFusedRows = 16;
    constexpr int kMinChannel = 128;
    if (output == nullptr || output->dimensions() <= 0 || output->getType().code != halide_type_float) {
        return false;
    }
    return picSiluMulFeatureDim(output) >= kMinChannel && picSiluMulActiveRows(output) <= kMaxFusedRows;
}

class PicSiluMulImageExecution : public Execution {
public:
    PicSiluMulImageExecution(const MNN::Op* op, Backend* backend) : Execution(backend), mOp(op) {
    }
    virtual ~PicSiluMulImageExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return INPUT_DATA_ERROR;
        }
        mUseFusedKernel = usePicSiluMulFusedKernel(outputs[0]);
        if (mUseFusedKernel) {
            mFallbackTemp.reset();
            mFallbackSilu.reset();
            mFallbackMul.reset();
            mFused.reset(new EltwiseExecution(
                inputs, outputs, "in1*(in0*native_recip((float4)1+native_exp(-in0)))", mOp, backend()));
            if (!mFused->valid()) {
                return NOT_SUPPORT;
            }
            return mFused->onResize(inputs, outputs);
        }

        auto openCLBackend = static_cast<OpenCLBackend*>(backend());
        mFused.reset();
        mFallbackTemp.reset(Tensor::createDevice(outputs[0]->shape(), outputs[0]->getType(), outputs[0]->getDimensionType()));
        if (mFallbackTemp == nullptr || !openCLBackend->onAcquireBuffer(mFallbackTemp.get(), Backend::DYNAMIC)) {
            return OUT_OF_MEMORY;
        }
        mFallbackSilu.reset(new UnaryExecution(
            "(convert_float4(in)*native_recip((float4)1+native_exp(convert_float4(-in))))", mOp, backend()));
        if (!mFallbackSilu->valid()) {
            openCLBackend->onReleaseBuffer(mFallbackTemp.get(), Backend::DYNAMIC);
            return NOT_SUPPORT;
        }
        auto code = mFallbackSilu->onResize({inputs[0]}, {mFallbackTemp.get()});
        if (code != NO_ERROR) {
            openCLBackend->onReleaseBuffer(mFallbackTemp.get(), Backend::DYNAMIC);
            return code;
        }
        std::vector<Tensor*> mulInputs = {mFallbackTemp.get(), inputs[1]};
        mFallbackMul.reset(new EltwiseExecution(mulInputs, outputs, "in0*in1", mOp, backend()));
        if (!mFallbackMul->valid()) {
            openCLBackend->onReleaseBuffer(mFallbackTemp.get(), Backend::DYNAMIC);
            return NOT_SUPPORT;
        }
        code = mFallbackMul->onResize(mulInputs, outputs);
        openCLBackend->onReleaseBuffer(mFallbackTemp.get(), Backend::DYNAMIC);
        return code;
    }

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (mUseFusedKernel) {
            return mFused->onExecute(inputs, outputs);
        }
        auto code = mFallbackSilu->onExecute({inputs[0]}, {mFallbackTemp.get()});
        if (code != NO_ERROR) {
            return code;
        }
        return mFallbackMul->onExecute({mFallbackTemp.get(), inputs[1]}, outputs);
    }

private:
    const MNN::Op* mOp;
    bool mUseFusedKernel = true;
    std::shared_ptr<Tensor> mFallbackTemp;
    std::shared_ptr<Execution> mFused;
    std::shared_ptr<Execution> mFallbackSilu;
    std::shared_ptr<Execution> mFallbackMul;
};

FuseExecution::FuseExecution(const std::vector<Tensor *> &inputs, Backend *backend, const Op* op)
    : CommonExecution(backend, op) {
    mUnits.resize(1);
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    std::set<std::string> buildOptions;
    std::string kernelName;
    auto extra = op->main_as_Extra();
    auto source = reinterpret_cast<const char*>(extra->info()->data());
    auto name = extra->type()->c_str();
    mKernelName = extra->type()->str();
    mUnits[0].kernel = runtime->buildKernelFromSource(source, name, buildOptions, mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL_CTOR(mUnits[0].kernel);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mUnits[0].kernel));
}

ErrorCode FuseExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    auto &unit  = mUnits[0];

    const int outputBatch    = outputShape.at(0);
    const int outputHeight   = outputShape.at(1);
    const int outputWidth    = outputShape.at(2);
    const int outputChannels = outputShape.at(3);

    const int channelBlocks  = UP_DIV(outputChannels, 4);
    const int remainChannels = channelBlocks * 4 - outputChannels;
    mGlobalWorkSize = {
        static_cast<uint32_t>(channelBlocks),
        static_cast<uint32_t>(outputWidth),
        static_cast<uint32_t>(outputHeight * outputBatch)
    };
    
    uint32_t idx    = 0;
    cl_int ret = CL_SUCCESS;
    for (auto input : inputs) {
        ret |= unit.kernel->get().setArg(idx++, openCLImage(input));
    }
    for (auto output : outputs) {
        ret |= unit.kernel->get().setArg(idx++, openCLImage(output));
    }
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);
    MNN_CHECK_CL_SUCCESS(ret, "setArg FuseExecution");

    mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), mKernelName, unit.kernel, mOpenCLBackend->getCLTuneLevel(), "Fuse").first;
    mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
    return NO_ERROR;
}

class FuseCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto param = op->main_as_Extra();
        if (param == nullptr || param->type() == nullptr) {
            return nullptr;
        }
        if(param->type()->str() == "ExtraConvolution2DPrelu")OPENCL_CREATOR_CHECK(new ConvExecution(inputs, outputs, op, backend, true));
        if(param->type()->str() == "PicSiluMul") {
            OPENCL_CREATOR_CHECK(new PicSiluMulImageExecution(op, backend));
        }
        OPENCL_CREATOR_CHECK(new FuseExecution(inputs, backend, op));
    }
};
REGISTER_OPENCL_OP_CREATOR(FuseCreator, OpType_Extra, IMAGE);

} // namespace OpenCL
} // namespace MNN
