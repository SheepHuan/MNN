//
//  FuseExecution.cpp
//  MNN
//
//  Created by MNN on 2025/07/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/FuseBufExecution.hpp"
#include "backend/opencl/execution/buffer/ConvBufExecution.hpp"
#include "backend/opencl/execution/buffer/ConvBufLowMemoryExecution.hpp"
#include "backend/opencl/execution/buffer/UnaryBufExecution.hpp"
#include "backend/opencl/execution/buffer/BinaryBufExecution.hpp"
#include "core/Macro.h"
#include "core/FileLoader.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include <sstream>

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

class PicSiluMulBufExecution : public Execution {
public:
    PicSiluMulBufExecution(const MNN::Op* op, Backend* backend) : Execution(backend), mOp(op) {
    }
    virtual ~PicSiluMulBufExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return INPUT_DATA_ERROR;
        }
        mUseFusedKernel = usePicSiluMulFusedKernel(outputs[0]);
        if (mUseFusedKernel) {
            mFallbackTemp.reset();
            mFallbackSilu.reset();
            mFallbackMul.reset();
            mFused.reset(new BinaryBufExecution(
                inputs, "in1*(in0*native_recip((float4)1+native_exp(-in0)))", mOp, backend()));
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
        mFallbackSilu.reset(new UnaryBufExecution(
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
        mFallbackMul.reset(new BinaryBufExecution(mulInputs, "in0*in1", mOp, backend()));
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

static const char* kPicPackedSiluMulBufSource = R"(
#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__kernel void pic_packed_silu_mul_buf(__private const int global_dim0,
                                      __private const int global_dim1,
                                      __global const FLOAT* input,
                                      __global FLOAT* output,
                                      __private const int channels,
                                      __private const int input_channel_align,
                                      __private const int output_channel_stride) {
    const int c4 = get_global_id(0);
    const int row = get_global_id(1);
    if (c4 >= global_dim0 || row >= global_dim1) {
        return;
    }
    const int c = c4 << 2;
    const int row_input_offset = row * input_channel_align;
    const int gate_offset = row_input_offset + c;
    const int up_offset = row_input_offset + channels + c;
    const int out_offset = row * output_channel_stride + c;
    if (c + 3 < channels) {
        float4 gate = convert_float4(vload4(0, input + gate_offset));
        float4 up = convert_float4(vload4(0, input + up_offset));
        float4 silu = gate * native_recip((float4)1.0f + native_exp(-gate));
        vstore4(CONVERT_FLOAT4(silu * up), 0, output + out_offset);
    } else {
        float4 gate = (float4)0.0f;
        float4 up = (float4)0.0f;
        float* gate_ptr = (float*)&gate;
        float* up_ptr = (float*)&up;
        for (int i = 0; i < 4 && c + i < channels; ++i) {
            gate_ptr[i] = (float)input[gate_offset + i];
            up_ptr[i] = (float)input[up_offset + i];
        }
        float4 silu = gate * native_recip((float4)1.0f + native_exp(-gate));
        float4 out = silu * up;
        float* out_ptr = (float*)&out;
        for (int i = 0; i < 4 && c + i < channels; ++i) {
            output[out_offset + i] = (FLOAT)out_ptr[i];
        }
    }
}
)";

class PicPackedSiluMulBufExecution : public CommonExecution {
public:
    PicPackedSiluMulBufExecution(const MNN::Op* op, Backend* backend) : CommonExecution(backend, op) {
        mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
        auto runtime = mOpenCLBackend->getOpenCLRuntime();
        mUnits.resize(1);
        std::set<std::string> buildOptions;
        mUnits[0].kernel = runtime->buildKernelFromSource(kPicPackedSiluMulBufSource, "pic_packed_silu_mul_buf",
                                                           buildOptions, mOpenCLBackend->getPrecision());
        OPENCL_CHECK_KERNEL_CTOR(mUnits[0].kernel);
        mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mUnits[0].kernel));
    }
    virtual ~PicPackedSiluMulBufExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INPUT_DATA_ERROR;
        }
        auto input = inputs[0];
        auto output = outputs[0];
        mRows = picSiluMulActiveRows(output);
        mChannels = picSiluMulFeatureDim(output);
        if (mRows <= 0 || mChannels <= 0 || output->getType().code != halide_type_float) {
            return NOT_SUPPORT;
        }
        auto inputShape = tensorShapeFormat(input);
        const int inputChannels = inputShape.at(3);
        if (inputChannels < mChannels * 2) {
            return NOT_SUPPORT;
        }
        const int inputChannelAlign = ROUND_UP(inputChannels, 4);
        const int outputChannelStride = mChannels;

        auto runtime = mOpenCLBackend->getOpenCLRuntime();
        auto& unit = mUnits[0];
        if (unit.kernel == nullptr) {
            return NOT_SUPPORT;
        }
        mGlobalWorkSize = {
            static_cast<uint32_t>(UP_DIV(mChannels, 4)),
            static_cast<uint32_t>(mRows),
        };
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
        ret |= unit.kernel->get().setArg(idx++, mChannels);
        ret |= unit.kernel->get().setArg(idx++, inputChannelAlign);
        ret |= unit.kernel->get().setArg(idx++, outputChannelStride);
        MNN_CHECK_CL_SUCCESS(ret, "setArg PicPackedSiluMulBufExecution");

        const std::string tuneKey = "pic_packed_silu_mul_buf_c" + std::to_string(mChannels) +
                                    "_r" + std::to_string(mRows);
        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runtime, tuneKey, unit.kernel,
                                          mOpenCLBackend->getCLTuneLevel(), "PicPackedSiluMulBuf")
                             .first;
        mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
        unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
        return NO_ERROR;
    }

private:
    OpenCLBackend* mOpenCLBackend = nullptr;
    uint32_t mMaxWorkGroupSize = 1;
    int mRows = 0;
    int mChannels = 0;
    std::vector<uint32_t> mGlobalWorkSize{1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1};
};

#ifdef MNN_LOW_MEMORY
struct PicGateUpConvSpec {
    std::string name;
    std::vector<int64_t> external;
    int ic = 0;
    int oc = 0;
    int quantBit = 4;
    int quantBlock = 64;
    int aMin = 1;
    int readType = 0;
    bool shapeInt32 = false;
    bool hasBias = false;
};

static const Attribute* findPicGateUpAttr(const Extra* extra, const char* key) {
    if (extra == nullptr || extra->attr() == nullptr || key == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr != nullptr && attr->key() != nullptr && attr->key()->str() == key) {
            return attr;
        }
    }
    return nullptr;
}

static std::string picGateUpAttrString(const Extra* extra, const std::string& key) {
    auto attr = findPicGateUpAttr(extra, key.c_str());
    return (attr != nullptr && attr->s() != nullptr) ? attr->s()->str() : "";
}

static int picGateUpAttrInt(const Extra* extra, const std::string& key, int fallback = 0) {
    auto attr = findPicGateUpAttr(extra, key.c_str());
    return attr != nullptr ? attr->i() : fallback;
}

static std::vector<int64_t> parsePicGateUpExternal(const std::string& value) {
    std::vector<int64_t> result;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            result.emplace_back(static_cast<int64_t>(std::stoll(item)));
        }
    }
    while (result.size() < 5) {
        result.emplace_back(0);
    }
    return result;
}

static PicGateUpConvSpec parsePicGateUpConvSpec(const Extra* extra, const std::string& prefix) {
    PicGateUpConvSpec spec;
    spec.name = picGateUpAttrString(extra, prefix + "_name");
    spec.external = parsePicGateUpExternal(picGateUpAttrString(extra, prefix + "_external"));
    spec.ic = picGateUpAttrInt(extra, prefix + "_in_features");
    spec.oc = picGateUpAttrInt(extra, prefix + "_out_features");
    spec.quantBit = picGateUpAttrInt(extra, prefix + "_quant_bit", 4);
    spec.quantBlock = picGateUpAttrInt(extra, prefix + "_quant_block", 64);
    spec.aMin = picGateUpAttrInt(extra, prefix + "_a_min", 1);
    spec.readType = picGateUpAttrInt(extra, prefix + "_read_type", 0);
    spec.shapeInt32 = picGateUpAttrInt(extra, prefix + "_shape_int32", 0) != 0;
    spec.hasBias = picGateUpAttrInt(extra, prefix + "_has_bias", 0) != 0;
    return spec;
}

static PicGateUpConvSpec parsePicLinearConvSpec(const Extra* extra) {
    PicGateUpConvSpec spec;
    spec.name = picGateUpAttrString(extra, "linear_name");
    spec.external = parsePicGateUpExternal(picGateUpAttrString(extra, "external"));
    spec.ic = picGateUpAttrInt(extra, "in_features");
    spec.oc = picGateUpAttrInt(extra, "out_features");
    spec.quantBit = picGateUpAttrInt(extra, "quant_bit", 4);
    spec.quantBlock = picGateUpAttrInt(extra, "quant_block", 64);
    spec.aMin = picGateUpAttrInt(extra, "a_min", 1);
    spec.readType = picGateUpAttrInt(extra, "read_type", 0);
    spec.shapeInt32 = picGateUpAttrInt(extra, "shape_int32", 0) != 0;
    spec.hasBias = picGateUpAttrInt(extra, "has_bias", 0) != 0;
    return spec;
}

static std::vector<uint8_t> buildPicGateUpChildConvOp(const PicGateUpConvSpec& spec, const char* externalPath) {
    flatbuffers::FlatBufferBuilder builder;
    Convolution2DT conv;
    conv.common.reset(new Convolution2DCommonT);
    conv.common->padX = 0;
    conv.common->padY = 0;
    conv.common->kernelX = 1;
    conv.common->kernelY = 1;
    conv.common->strideX = 1;
    conv.common->strideY = 1;
    conv.common->dilateX = 1;
    conv.common->dilateY = 1;
    conv.common->padMode = PadMode_CAFFE;
    conv.common->group = 1;
    conv.common->outputCount = spec.oc;
    conv.common->inputCount = spec.ic;
    conv.common->relu = false;
    conv.common->relu6 = false;
    conv.common->hasOutputShape = false;
    conv.quanParameter.reset(new IDSTQuanT);
    conv.quanParameter->type = spec.quantBit == 16 ? 3 : 1;
    conv.quanParameter->useInt32 = false;
    conv.quanParameter->quantScale = 1.0f;
    conv.quanParameter->scaleIn = 0.0f;
    conv.quanParameter->scaleOut = 0.0f;
    conv.quanParameter->aMaxOrBits = spec.quantBit;
    conv.quanParameter->aMin = spec.aMin;
    conv.quanParameter->readType = spec.readType;
    conv.quanParameter->has_scaleInt = false;
    conv.quanParameter->shapeInt32 = spec.shapeInt32;
    conv.quanParameter->weightSize = 0;
    conv.external = spec.external;
    conv.bias.resize(std::max(0, spec.oc), 0.0f);
    if (externalPath != nullptr && spec.external.size() > 3 && spec.external[3] > 0) {
        FileLoader loader(externalPath);
        if (loader.valid()) {
            loader.offset(spec.external[0] + spec.external[1] + spec.external[2]);
            loader.read(reinterpret_cast<char*>(conv.bias.data()), spec.external[3]);
        }
    }
    auto convOffset = Convolution2D::Pack(builder, &conv);
    auto nameOffset = builder.CreateString(spec.name);
    flatbuffers::Offset<flatbuffers::String> externalPathOffset = 0;
    if (externalPath != nullptr) {
        externalPathOffset = builder.CreateString(externalPath);
    }
    OpBuilder opBuilder(builder);
    opBuilder.add_name(nameOffset);
    opBuilder.add_main(convOffset.Union());
    opBuilder.add_main_type(OpParameter_Convolution2D);
    opBuilder.add_type(OpType_Convolution);
    opBuilder.add_defaultDimentionFormat(MNN_DATA_FORMAT_NHWC);
    opBuilder.add_externalPath(externalPathOffset);
    builder.Finish(opBuilder.Finish());
    const uint8_t* ptr = builder.GetBufferPointer();
    return std::vector<uint8_t>(ptr, ptr + builder.GetSize());
}

static const char* kPicGateUpWeightOnlyBufSource = R"(
#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#define GLOBAL_SIZE_DIM_3 __private int global_size_dim0, __private int global_size_dim1, __private int global_size_dim2,
#define UCHAR4_TO_CHAR8_LOCAL(b, scale, offset, wei) \
    wei.s0 = (FLOAT)((b.s0 >> 4) - 8); \
    wei.s1 = (FLOAT)((b.s0 & 15) - 8); \
    wei.s2 = (FLOAT)((b.s1 >> 4) - 8); \
    wei.s3 = (FLOAT)((b.s1 & 15) - 8); \
    wei.s4 = (FLOAT)((b.s2 >> 4) - 8); \
    wei.s5 = (FLOAT)((b.s2 & 15) - 8); \
    wei.s6 = (FLOAT)((b.s3 >> 4) - 8); \
    wei.s7 = (FLOAT)((b.s3 & 15) - 8); \
    wei = wei * scale + offset;

inline FLOAT4 pic_gateup_silu_mul4(FLOAT4 gate, FLOAT4 up) {
    float4 gate_f = convert_float4(gate);
    float4 up_f = convert_float4(up);
    float4 fused = gate_f * native_recip((float4)1.0f + native_exp(-gate_f)) * up_f;
#ifdef MNN_SUPPORT_FP16
    return convert_half4(fused);
#else
    return fused;
#endif
}

__kernel void pic_gateup_int4_batch_gemv_buf(GLOBAL_SIZE_DIM_3
                        __global const FLOAT* input,
                        __read_only image2d_t gate_weight,
                        __global const FLOAT *gate_scale_offset,
                        __global const FLOAT *gate_bias,
#ifdef FUSE_SILU_OUTPUT
                        __read_only image2d_t up_weight,
                        __global const FLOAT *up_scale_offset,
                        __global const FLOAT *up_bias,
                        __global FLOAT* fused_output,
#else
                        __global FLOAT* gate_output,
                        __read_only image2d_t up_weight,
                        __global const FLOAT *up_scale_offset,
                        __global const FLOAT *up_bias,
                        __global FLOAT* up_output,
#endif
                        __private const int rows,
                        __private const int dst_channel_align,
                        __private const int dst_channel_c4,
                        __private const int src_channel_align,
                        __private const int src_channel_c4,
                        __private const int src_channel,
                        __private const int block_dim,
                        __private const float gate_coef,
                        __private const float up_coef) {
    const int lid = get_local_id(0);
    const int oc = get_global_id(1);
    const int b4 = get_global_id(2);
    if (lid >= global_size_dim0 || oc >= global_size_dim1 || b4 >= global_size_dim2) {
        return;
    }
    const int row = b4 << 2;
    const int oc8 = oc << 3;
    const int loop = (src_channel + 4 - 1) / 4;
    const int bhw4 = rows << 2;
    FLOAT8 gate0 = 0;
    FLOAT8 gate1 = 0;
    FLOAT8 gate2 = 0;
    FLOAT8 gate3 = 0;
    FLOAT8 up0 = 0;
    FLOAT8 up1 = 0;
    FLOAT8 up2 = 0;
    FLOAT8 up3 = 0;
    __local FLOAT8 gate_sum0[WGS];
    __local FLOAT8 gate_sum1[WGS];
    __local FLOAT8 gate_sum2[WGS];
    __local FLOAT8 gate_sum3[WGS];
    __local FLOAT8 up_sum0[WGS];
    __local FLOAT8 up_sum1[WGS];
    __local FLOAT8 up_sum2[WGS];
    __local FLOAT8 up_sum3[WGS];
    for (int j = lid; j < loop; j += WGS) {
        const int k4 = j << 2;
#ifdef ASYMMETRIC
        FLOAT8 gate_scale;
        FLOAT8 gate_offset;
        FLOAT8 up_scale;
        FLOAT8 up_offset;
        {
            FLOAT16 so = vload16(0, gate_scale_offset + oc8 * 2 + (k4 / block_dim) * dst_channel_c4 * 8) / (FLOAT16)gate_coef;
            gate_scale = so.s02468ace;
            gate_offset = so.s13579bdf;
            so = vload16(0, up_scale_offset + oc8 * 2 + (k4 / block_dim) * dst_channel_c4 * 8) / (FLOAT16)up_coef;
            up_scale = so.s02468ace;
            up_offset = so.s13579bdf;
        }
#else
        FLOAT8 gate_scale = vload8(0, gate_scale_offset + oc8 + (k4 / block_dim) * dst_channel_c4 * 4) / (FLOAT8)gate_coef;
        FLOAT8 up_scale = vload8(0, up_scale_offset + oc8 + (k4 / block_dim) * dst_channel_c4 * 4) / (FLOAT8)up_coef;
        FLOAT8 gate_offset = 0;
        FLOAT8 up_offset = 0;
#endif
        FLOAT4 in0 = vload4(0, input + row * src_channel_align + k4);
        FLOAT4 in1 = row + 1 < rows ? vload4(0, input + (row + 1) * src_channel_align + k4) : (FLOAT4)0;
        FLOAT4 in2 = row + 2 < rows ? vload4(0, input + (row + 2) * src_channel_align + k4) : (FLOAT4)0;
        FLOAT4 in3 = row + 3 < rows ? vload4(0, input + (row + 3) * src_channel_align + k4) : (FLOAT4)0;
        uchar16 gate_packed = as_uchar16(read_imagei(gate_weight, SAMPLER, (int2)(j, oc)));
        uchar16 up_packed = as_uchar16(read_imagei(up_weight, SAMPLER, (int2)(j, oc)));
        FLOAT8 gate_wei;
        FLOAT8 up_wei;
        UCHAR4_TO_CHAR8_LOCAL(gate_packed.s0123, gate_scale, gate_offset, gate_wei);
        UCHAR4_TO_CHAR8_LOCAL(up_packed.s0123, up_scale, up_offset, up_wei);
        gate0 = mad((FLOAT8)in0.s0, gate_wei, gate0); gate1 = mad((FLOAT8)in1.s0, gate_wei, gate1);
        gate2 = mad((FLOAT8)in2.s0, gate_wei, gate2); gate3 = mad((FLOAT8)in3.s0, gate_wei, gate3);
        up0 = mad((FLOAT8)in0.s0, up_wei, up0); up1 = mad((FLOAT8)in1.s0, up_wei, up1);
        up2 = mad((FLOAT8)in2.s0, up_wei, up2); up3 = mad((FLOAT8)in3.s0, up_wei, up3);
        UCHAR4_TO_CHAR8_LOCAL(gate_packed.s4567, gate_scale, gate_offset, gate_wei);
        UCHAR4_TO_CHAR8_LOCAL(up_packed.s4567, up_scale, up_offset, up_wei);
        gate0 = mad((FLOAT8)in0.s1, gate_wei, gate0); gate1 = mad((FLOAT8)in1.s1, gate_wei, gate1);
        gate2 = mad((FLOAT8)in2.s1, gate_wei, gate2); gate3 = mad((FLOAT8)in3.s1, gate_wei, gate3);
        up0 = mad((FLOAT8)in0.s1, up_wei, up0); up1 = mad((FLOAT8)in1.s1, up_wei, up1);
        up2 = mad((FLOAT8)in2.s1, up_wei, up2); up3 = mad((FLOAT8)in3.s1, up_wei, up3);
        UCHAR4_TO_CHAR8_LOCAL(gate_packed.s89ab, gate_scale, gate_offset, gate_wei);
        UCHAR4_TO_CHAR8_LOCAL(up_packed.s89ab, up_scale, up_offset, up_wei);
        gate0 = mad((FLOAT8)in0.s2, gate_wei, gate0); gate1 = mad((FLOAT8)in1.s2, gate_wei, gate1);
        gate2 = mad((FLOAT8)in2.s2, gate_wei, gate2); gate3 = mad((FLOAT8)in3.s2, gate_wei, gate3);
        up0 = mad((FLOAT8)in0.s2, up_wei, up0); up1 = mad((FLOAT8)in1.s2, up_wei, up1);
        up2 = mad((FLOAT8)in2.s2, up_wei, up2); up3 = mad((FLOAT8)in3.s2, up_wei, up3);
        UCHAR4_TO_CHAR8_LOCAL(gate_packed.scdef, gate_scale, gate_offset, gate_wei);
        UCHAR4_TO_CHAR8_LOCAL(up_packed.scdef, up_scale, up_offset, up_wei);
        gate0 = mad((FLOAT8)in0.s3, gate_wei, gate0); gate1 = mad((FLOAT8)in1.s3, gate_wei, gate1);
        gate2 = mad((FLOAT8)in2.s3, gate_wei, gate2); gate3 = mad((FLOAT8)in3.s3, gate_wei, gate3);
        up0 = mad((FLOAT8)in0.s3, up_wei, up0); up1 = mad((FLOAT8)in1.s3, up_wei, up1);
        up2 = mad((FLOAT8)in2.s3, up_wei, up2); up3 = mad((FLOAT8)in3.s3, up_wei, up3);
    }
    gate_sum0[lid] = gate0; gate_sum1[lid] = gate1; gate_sum2[lid] = gate2; gate_sum3[lid] = gate3;
    up_sum0[lid] = up0; up_sum1[lid] = up1; up_sum2[lid] = up2; up_sum3[lid] = up3;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = WGS / 2; i > 0; i >>= 1) {
        if (lid < i) {
            gate_sum0[lid] += gate_sum0[lid + i]; gate_sum1[lid] += gate_sum1[lid + i];
            gate_sum2[lid] += gate_sum2[lid + i]; gate_sum3[lid] += gate_sum3[lid + i];
            up_sum0[lid] += up_sum0[lid + i]; up_sum1[lid] += up_sum1[lid + i];
            up_sum2[lid] += up_sum2[lid + i]; up_sum3[lid] += up_sum3[lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        FLOAT8 gate_bias_v = vload8(0, gate_bias + oc8);
        FLOAT8 up_bias_v = vload8(0, up_bias + oc8);
        gate0 = gate_sum0[0] + gate_bias_v; gate1 = gate_sum1[0] + gate_bias_v;
        gate2 = gate_sum2[0] + gate_bias_v; gate3 = gate_sum3[0] + gate_bias_v;
        up0 = up_sum0[0] + up_bias_v; up1 = up_sum1[0] + up_bias_v;
        up2 = up_sum2[0] + up_bias_v; up3 = up_sum3[0] + up_bias_v;
#ifdef FUSE_SILU_OUTPUT
        FLOAT8 fused0 = (FLOAT8)(pic_gateup_silu_mul4(gate0.s0123, up0.s0123),
                                  pic_gateup_silu_mul4(gate0.s4567, up0.s4567));
        FLOAT8 fused1 = (FLOAT8)(pic_gateup_silu_mul4(gate1.s0123, up1.s0123),
                                  pic_gateup_silu_mul4(gate1.s4567, up1.s4567));
        FLOAT8 fused2 = (FLOAT8)(pic_gateup_silu_mul4(gate2.s0123, up2.s0123),
                                  pic_gateup_silu_mul4(gate2.s4567, up2.s4567));
        FLOAT8 fused3 = (FLOAT8)(pic_gateup_silu_mul4(gate3.s0123, up3.s0123),
                                  pic_gateup_silu_mul4(gate3.s4567, up3.s4567));
#endif
        const int out_c = oc << 1;
        const int out_offset = out_c * bhw4 + row * 4;
        if (row + 3 < rows) {
#ifdef FUSE_SILU_OUTPUT
            vstore16((FLOAT16)(fused0.s0123, fused1.s0123, fused2.s0123, fused3.s0123), 0, fused_output + out_offset);
            if (oc8 + 4 < dst_channel_align) {
                const int out_offset_hi = out_offset + bhw4;
                vstore16((FLOAT16)(fused0.s4567, fused1.s4567, fused2.s4567, fused3.s4567), 0, fused_output + out_offset_hi);
            }
#else
            vstore16((FLOAT16)(gate0.s0123, gate1.s0123, gate2.s0123, gate3.s0123), 0, gate_output + out_offset);
            vstore16((FLOAT16)(up0.s0123, up1.s0123, up2.s0123, up3.s0123), 0, up_output + out_offset);
            if (oc8 + 4 < dst_channel_align) {
                const int out_offset_hi = out_offset + bhw4;
                vstore16((FLOAT16)(gate0.s4567, gate1.s4567, gate2.s4567, gate3.s4567), 0, gate_output + out_offset_hi);
                vstore16((FLOAT16)(up0.s4567, up1.s4567, up2.s4567, up3.s4567), 0, up_output + out_offset_hi);
            }
#endif
        } else {
#ifdef FUSE_SILU_OUTPUT
            vstore4(fused0.s0123, 0, fused_output + out_offset);
            if (row + 1 < rows) {
                vstore4(fused1.s0123, 0, fused_output + out_offset + 4);
            }
            if (row + 2 < rows) {
                vstore4(fused2.s0123, 0, fused_output + out_offset + 8);
            }
            if (oc8 + 4 < dst_channel_align) {
                const int out_offset_hi = out_offset + bhw4;
                vstore4(fused0.s4567, 0, fused_output + out_offset_hi);
                if (row + 1 < rows) {
                    vstore4(fused1.s4567, 0, fused_output + out_offset_hi + 4);
                }
                if (row + 2 < rows) {
                    vstore4(fused2.s4567, 0, fused_output + out_offset_hi + 8);
                }
            }
#else
            vstore4(gate0.s0123, 0, gate_output + out_offset);
            vstore4(up0.s0123, 0, up_output + out_offset);
            if (row + 1 < rows) {
                vstore4(gate1.s0123, 0, gate_output + out_offset + 4);
                vstore4(up1.s0123, 0, up_output + out_offset + 4);
            }
            if (row + 2 < rows) {
                vstore4(gate2.s0123, 0, gate_output + out_offset + 8);
                vstore4(up2.s0123, 0, up_output + out_offset + 8);
            }
            if (oc8 + 4 < dst_channel_align) {
                const int out_offset_hi = out_offset + bhw4;
                vstore4(gate0.s4567, 0, gate_output + out_offset_hi);
                vstore4(up0.s4567, 0, up_output + out_offset_hi);
                if (row + 1 < rows) {
                    vstore4(gate1.s4567, 0, gate_output + out_offset_hi + 4);
                    vstore4(up1.s4567, 0, up_output + out_offset_hi + 4);
                }
                if (row + 2 < rows) {
                    vstore4(gate2.s4567, 0, gate_output + out_offset_hi + 8);
                    vstore4(up2.s4567, 0, up_output + out_offset_hi + 8);
                }
            }
#endif
        }
    }
}
)";

class PicGateUpWeightOnlyBufExecution : public CommonExecution {
public:
    PicGateUpWeightOnlyBufExecution(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const Op* op, Backend* backend, bool fuseSiluOutput = false)
        : CommonExecution(backend, op) {
        mFuseSiluOutput = fuseSiluOutput;
        mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
        auto extra = op->main_as_Extra();
        mGateSpec = parsePicGateUpConvSpec(extra, "gate");
        mUpSpec = parsePicGateUpConvSpec(extra, "up");
        const char* externalPath = op->externalPath() != nullptr ? op->externalPath()->c_str() : nullptr;
        if (mGateSpec.ic <= 0 || mGateSpec.oc <= 0 || mUpSpec.ic != mGateSpec.ic || mUpSpec.oc != mGateSpec.oc) {
            mValid = false;
            return;
        }
        mGateOpBuffer = buildPicGateUpChildConvOp(mGateSpec, externalPath);
        mUpOpBuffer = buildPicGateUpChildConvOp(mUpSpec, externalPath);
        mGateOp = flatbuffers::GetRoot<Op>(mGateOpBuffer.data());
        mUpOp = flatbuffers::GetRoot<Op>(mUpOpBuffer.data());
        mGateConv.reset(new ConvBufLowMemoryExecution(inputs, outputs, mGateOp, backend));
        mUpConv.reset(new ConvBufLowMemoryExecution(inputs, outputs, mUpOp, backend));
        if (mGateConv == nullptr || mUpConv == nullptr) {
            mValid = false;
            return;
        }
        mGateResource = mGateConv->resource();
        mUpResource = mUpConv->resource();
        if (mGateResource != nullptr && mUpResource != nullptr) {
            auto runtime = mOpenCLBackend->getOpenCLRuntime();
            std::set<std::string> buildOptions = mGateResource->mBuildOptions;
            buildOptions.emplace("-DWGS=" + std::to_string(kFusedWgs));
            buildOptions.emplace("-DQUANT_BIT=4");
            buildOptions.emplace("-DUSE_IMAGE");
            if (mFuseSiluOutput) {
                buildOptions.emplace("-DFUSE_SILU_OUTPUT");
            }
            mFusedKernel = runtime->buildKernelFromSource(kPicGateUpWeightOnlyBufSource,
                                                          "pic_gateup_int4_batch_gemv_buf",
                                                          buildOptions, mOpenCLBackend->getPrecision());
            if (mFusedKernel != nullptr) {
                mFusedMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mFusedKernel));
            }
        }
    }

    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (!mValid || inputs.size() != 1 || (mFuseSiluOutput ? outputs.size() != 1 : outputs.size() != 2)) {
            return INPUT_DATA_ERROR;
        }
        mUseFused = mFuseSiluOutput ? canUseFused(inputs[0], outputs[0], outputs[0])
                                    : canUseFused(inputs[0], outputs[0], outputs[1]);
        if (!mUseFused) {
            if (mFuseSiluOutput) {
                mFallbackGate.reset(Tensor::createDevice(outputs[0]->shape(), outputs[0]->getType(),
                                                          outputs[0]->getDimensionType()));
                mFallbackUp.reset(Tensor::createDevice(outputs[0]->shape(), outputs[0]->getType(),
                                                        outputs[0]->getDimensionType()));
                if (mFallbackGate == nullptr || mFallbackUp == nullptr ||
                    !mOpenCLBackend->onAcquireBuffer(mFallbackGate.get(), Backend::DYNAMIC) ||
                    !mOpenCLBackend->onAcquireBuffer(mFallbackUp.get(), Backend::DYNAMIC)) {
                    return OUT_OF_MEMORY;
                }
                auto code = mGateConv->onResize(inputs, {mFallbackGate.get()});
                if (code == NO_ERROR) {
                    code = mUpConv->onResize(inputs, {mFallbackUp.get()});
                }
                if (code == NO_ERROR) {
                    mFallbackSilu.reset(new PicSiluMulBufExecution(mOp, backend()));
                    code = mFallbackSilu->onResize({mFallbackGate.get(), mFallbackUp.get()}, outputs);
                }
                mOpenCLBackend->onReleaseBuffer(mFallbackGate.get(), Backend::DYNAMIC);
                mOpenCLBackend->onReleaseBuffer(mFallbackUp.get(), Backend::DYNAMIC);
                return code;
            }
            auto code = mGateConv->onResize(inputs, {outputs[0]});
            if (code != NO_ERROR) {
                return code;
            }
            return mUpConv->onResize(inputs, {outputs[1]});
        }
        mRows = outputs[0]->length(0);
        mIc = mGateSpec.ic;
        mOc = mGateSpec.oc;
        mInputChannelAlign = ROUND_UP(mIc, 4);
        mOutputChannelAlign = ROUND_UP(mOc, 4);
        mInputNhwc.reset(Tensor::createDevice<float>({mInputChannelAlign * ROUND_UP(mRows, 4)}));
        if (mInputNhwc == nullptr || !mOpenCLBackend->onAcquireBuffer(mInputNhwc.get(), Backend::DYNAMIC)) {
            return OUT_OF_MEMORY;
        }
        auto runtime = mOpenCLBackend->getOpenCLRuntime();
        mUnits.resize(2);
        {
            auto& unit = mUnits[0];
            unit.kernel = runtime->buildKernel("gemm_conv1x1_buf", "gemm_c4nhw4_to_nhwc", {}, mOpenCLBackend->getPrecision());
            OPENCL_CHECK_KERNEL(unit.kernel);
            const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
            mPreGws = {static_cast<uint32_t>(UP_DIV(mRows, 4)), static_cast<uint32_t>(UP_DIV(mIc, 4))};
            uint32_t idx = 0;
            cl_int ret = CL_SUCCESS;
            ret |= unit.kernel->get().setArg(idx++, mPreGws[0]);
            ret |= unit.kernel->get().setArg(idx++, mPreGws[1]);
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(inputs[0]));
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mInputNhwc.get()));
            ret |= unit.kernel->get().setArg(idx++, mRows);
            ret |= unit.kernel->get().setArg(idx++, mIc);
            ret |= unit.kernel->get().setArg(idx++, mInputChannelAlign);
            MNN_CHECK_CL_SUCCESS(ret, "setArg PicGateUpWeightOnly c4nhw4_to_nhwc");
            mPreLws = localWS2DDefault(mPreGws, maxWorkGroupSize, runtime, "pic_gateup_c4nhw4_to_nhwc",
                                       unit.kernel, mOpenCLBackend->getCLTuneLevel(), "PicGateUpWeightOnly").first;
            mOpenCLBackend->recordKernel2d(unit.kernel, mPreGws, mPreLws);
            unit.globalWorkSize = {mPreGws[0], mPreGws[1]};
            unit.localWorkSize = {mPreLws[0], mPreLws[1]};
        }
        {
            auto& unit = mUnits[1];
            unit.kernel = mFusedKernel;
            OPENCL_CHECK_KERNEL(unit.kernel);
            mFusedGws = {
                static_cast<uint32_t>(kFusedWgs),
                static_cast<uint32_t>(UP_DIV(mOc, 8)),
                static_cast<uint32_t>(UP_DIV(mRows, 4)),
            };
            if (mFusedMaxWorkGroupSize < static_cast<uint32_t>(kFusedWgs)) {
                return NOT_SUPPORT;
            }
            mFusedLws = {kFusedWgs, 1, 1};
            uint32_t idx = 0;
            cl_int ret = CL_SUCCESS;
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mFusedGws[0]));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mFusedGws[1]));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mFusedGws[2]));
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mInputNhwc.get()));
            ret |= unit.kernel->get().setArg(idx++, *mGateResource->mKernelImage.get());
            ret |= unit.kernel->get().setArg(idx++, *mGateResource->mDequantScaleOffsetBuffer.get());
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mGateResource->mBias.get()));
            if (mFuseSiluOutput) {
                ret |= unit.kernel->get().setArg(idx++, *mUpResource->mKernelImage.get());
                ret |= unit.kernel->get().setArg(idx++, *mUpResource->mDequantScaleOffsetBuffer.get());
                ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mUpResource->mBias.get()));
                ret |= unit.kernel->get().setArg(idx++, openCLBuffer(outputs[0]));
            } else {
                ret |= unit.kernel->get().setArg(idx++, openCLBuffer(outputs[0]));
                ret |= unit.kernel->get().setArg(idx++, *mUpResource->mKernelImage.get());
                ret |= unit.kernel->get().setArg(idx++, *mUpResource->mDequantScaleOffsetBuffer.get());
                ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mUpResource->mBias.get()));
                ret |= unit.kernel->get().setArg(idx++, openCLBuffer(outputs[1]));
            }
            ret |= unit.kernel->get().setArg(idx++, mRows);
            ret |= unit.kernel->get().setArg(idx++, mOutputChannelAlign);
            ret |= unit.kernel->get().setArg(idx++, UP_DIV(mOc, 4));
            ret |= unit.kernel->get().setArg(idx++, mInputChannelAlign);
            ret |= unit.kernel->get().setArg(idx++, UP_DIV(mIc, 4));
            ret |= unit.kernel->get().setArg(idx++, mIc);
            ret |= unit.kernel->get().setArg(idx++, mGateResource->mInputChannel / mGateResource->mBlockSize);
            ret |= unit.kernel->get().setArg(idx++, mGateResource->mCoef);
            ret |= unit.kernel->get().setArg(idx++, mUpResource->mCoef);
            MNN_CHECK_CL_SUCCESS(ret, "setArg PicGateUpWeightOnly fused");
            mOpenCLBackend->recordKernel3d(unit.kernel, mFusedGws, mFusedLws);
            unit.globalWorkSize = {mFusedGws[0], mFusedGws[1], mFusedGws[2]};
            unit.localWorkSize = {mFusedLws[0], mFusedLws[1], mFusedLws[2]};
        }
        mOpenCLBackend->onReleaseBuffer(mInputNhwc.get(), Backend::DYNAMIC);
        return NO_ERROR;
    }

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (!mUseFused) {
            if (mFuseSiluOutput) {
                auto code = mGateConv->onExecute(inputs, {mFallbackGate.get()});
                if (code != NO_ERROR) {
                    return code;
                }
                code = mUpConv->onExecute(inputs, {mFallbackUp.get()});
                if (code != NO_ERROR) {
                    return code;
                }
                return mFallbackSilu->onExecute({mFallbackGate.get(), mFallbackUp.get()}, outputs);
            }
            auto code = mGateConv->onExecute(inputs, {outputs[0]});
            if (code != NO_ERROR) {
                return code;
            }
            return mUpConv->onExecute(inputs, {outputs[1]});
        }
        return CommonExecution::onExecute(inputs, outputs);
    }

private:
    bool canUseFused(const Tensor* input, const Tensor* gateOutput, const Tensor* upOutput) const {
        if (input == nullptr || gateOutput == nullptr || upOutput == nullptr ||
            mGateResource == nullptr || mUpResource == nullptr || mFusedKernel == nullptr) {
            return false;
        }
        const int rows = gateOutput->length(0);
        return rows >= 1 && rows <= 8 &&
               mGateResource->mNumQuantBit == 4 && mUpResource->mNumQuantBit == 4 &&
               mGateResource->mUseImage && mUpResource->mUseImage &&
               mGateSpec.ic == mUpSpec.ic && mGateSpec.oc == mUpSpec.oc &&
               mGateSpec.oc == gateOutput->length(1) && mUpSpec.oc == upOutput->length(1);
    }

    bool mValid = true;
    bool mUseFused = false;
    bool mFuseSiluOutput = false;
    OpenCLBackend* mOpenCLBackend = nullptr;
    PicGateUpConvSpec mGateSpec;
    PicGateUpConvSpec mUpSpec;
    std::vector<uint8_t> mGateOpBuffer;
    std::vector<uint8_t> mUpOpBuffer;
    const Op* mGateOp = nullptr;
    const Op* mUpOp = nullptr;
    std::shared_ptr<ConvBufLowMemoryExecution> mGateConv;
    std::shared_ptr<ConvBufLowMemoryExecution> mUpConv;
    std::shared_ptr<ConvBufResource> mGateResource;
    std::shared_ptr<ConvBufResource> mUpResource;
    std::shared_ptr<Tensor> mFallbackGate;
    std::shared_ptr<Tensor> mFallbackUp;
    std::shared_ptr<Execution> mFallbackSilu;
    std::shared_ptr<KernelWrap> mFusedKernel;
    uint32_t mFusedMaxWorkGroupSize = 1;
    std::shared_ptr<Tensor> mInputNhwc;
    static constexpr int kFusedWgs = 64;
    int mRows = 0;
    int mIc = 0;
    int mOc = 0;
    int mInputChannelAlign = 0;
    int mOutputChannelAlign = 0;
    std::vector<uint32_t> mPreGws{1, 1};
    std::vector<uint32_t> mPreLws{1, 1};
    std::vector<uint32_t> mFusedGws{1, 1, 1};
    std::vector<uint32_t> mFusedLws{1, 1, 1};
};

class PicLinearNhwcWeightOnlyBufExecution : public CommonExecution {
public:
    PicLinearNhwcWeightOnlyBufExecution(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                        const Op* op, Backend* backend)
        : CommonExecution(backend, op) {
        mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
        auto extra = op->main_as_Extra();
        mSpec = parsePicLinearConvSpec(extra);
        const char* externalPath = op->externalPath() != nullptr ? op->externalPath()->c_str() : nullptr;
        if (mSpec.name.empty() || mSpec.ic <= 0 || mSpec.oc <= 0 || mSpec.quantBit != 4) {
            mValid = false;
            return;
        }
        mOpBuffer = buildPicGateUpChildConvOp(mSpec, externalPath);
        mChildOp = flatbuffers::GetRoot<Op>(mOpBuffer.data());
        mChildConv.reset(new ConvBufLowMemoryExecution(inputs, outputs, mChildOp, backend));
        if (mChildConv == nullptr) {
            mValid = false;
            return;
        }
        mResource = mChildConv->resource();
    }

    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (!mValid || inputs.size() != 1 || outputs.size() != 1 || mResource == nullptr) {
            return INPUT_DATA_ERROR;
        }
        int rows = 0;
        int inputChannels = 0;
        int outChannel = 0;
        auto code = resolveLinearRowsAndChannels(inputs, outputs, &rows, &inputChannels, &outChannel);
        if (code != NO_ERROR) {
            return code;
        }
        auto input = inputs[0];
        auto output = outputs[0];

        auto runtime = mOpenCLBackend->getOpenCLRuntime();
        mUnits.resize(1);
        auto& unit = mUnits[0];
        std::set<std::string> buildOption = mResource->mBuildOptions;
        if (mResource->mUseImage) {
            buildOption.emplace("-DUSE_IMAGE");
        }
        if (rows > 1) {
            buildOption.emplace("-DCOMPUTE_BATCH");
            buildOption.emplace("-DOUTPUT_BHW=" + std::to_string(rows));
        }
        buildOption.emplace("-DINPUT_CHANNEL_LEAVES_NUM=0");
        buildOption.emplace("-DWGS=" + std::to_string(kLocalSize));
        unit.kernel = runtime->buildKernel("gemv_conv1x1_buf", "gemv_conv_c8_buf",
                                           buildOption, mOpenCLBackend->getPrecision());
        OPENCL_CHECK_KERNEL(unit.kernel);
        const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        if (maxWorkGroupSize < static_cast<uint32_t>(kLocalSize)) {
            return NOT_SUPPORT;
        }

        const int inputChannelAlign = ROUND_UP(inputChannels, 4);
        const int outputChannelAlign8 = ROUND_UP(outChannel, 8);
        const int inputChannelBlocks = UP_DIV(inputChannels, 4);
        const int outputChannelBlocks = UP_DIV(outChannel, 4);
        const int blockNum = mResource->mBlockSize;
        const int blockDim = mResource->mInputChannel / mResource->mBlockSize;
        mGlobalWorkSize = {
            static_cast<uint32_t>(kLocalSize),
            static_cast<uint32_t>(UP_DIV(outChannel, 8)),
            static_cast<uint32_t>(UP_DIV(rows, 4)),
        };
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[0]));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[1]));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[2]));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
        if (mResource->mUseImage) {
            ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelImage.get());
        } else {
            ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
        }
        ret |= unit.kernel->get().setArg(idx++, *mResource->mDequantScaleOffsetBuffer.get());
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
        ret |= unit.kernel->get().setArg(idx++, outputChannelAlign8);
        ret |= unit.kernel->get().setArg(idx++, inputChannelAlign);
        ret |= unit.kernel->get().setArg(idx++, outputChannelBlocks);
        ret |= unit.kernel->get().setArg(idx++, inputChannelBlocks);
        ret |= unit.kernel->get().setArg(idx++, inputChannels);
        ret |= unit.kernel->get().setArg(idx++, blockNum);
        ret |= unit.kernel->get().setArg(idx++, blockDim);
        ret |= unit.kernel->get().setArg(idx++, static_cast<float>(mResource->mCoef));
        MNN_CHECK_CL_SUCCESS(ret, "setArg PicLinearNhwcWeightOnly");

        mLocalWorkSize = {static_cast<uint32_t>(kLocalSize), 1, 1};
        mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
        return NO_ERROR;
    }

private:
    ErrorCode resolveLinearRowsAndChannels(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                           int* rows, int* inputChannels, int* outputChannels) const {
        if (inputs.size() != 1 || outputs.size() != 1 || rows == nullptr) {
            return INPUT_DATA_ERROR;
        }
        auto input = inputs[0];
        auto output = outputs[0];
        int inputRows = 0;
        int localInputChannels = 0;
        int outputRows = 0;
        int localOutputChannels = 0;
        if (!resolveRowsAndChannels(input, mSpec.ic, &inputRows, &localInputChannels) ||
            !resolveRowsAndChannels(output, mSpec.oc, &outputRows, &localOutputChannels)) {
            return NOT_SUPPORT;
        }
        if (outputRows <= 0 || localInputChannels != mSpec.ic || localOutputChannels != mSpec.oc ||
            inputRows != outputRows ||
            localInputChannels % 4 != 0 || localOutputChannels % 8 != 0 ||
            input->getType().code != halide_type_float || output->getType().code != halide_type_float) {
            return NOT_SUPPORT;
        }
        *rows = outputRows;
        if (inputChannels != nullptr) {
            *inputChannels = localInputChannels;
        }
        if (outputChannels != nullptr) {
            *outputChannels = localOutputChannels;
        }
        return NO_ERROR;
    }

    static bool resolveRowsAndChannels(const Tensor* tensor, int expectedChannels, int* rows, int* channels) {
        if (tensor == nullptr || rows == nullptr || channels == nullptr) {
            return false;
        }
        if (tensor->dimensions() == 4 && tensor->length(1) == 1 && tensor->length(2) == 1 &&
            tensor->length(3) == expectedChannels) {
            *rows = tensor->length(0);
            *channels = tensor->length(3);
            return true;
        }
        auto shape = tensorShapeFormat(tensor);
        *rows = shape.at(0) * shape.at(1) * shape.at(2);
        *channels = shape.at(3);
        return true;
    }

    bool mValid = true;
    OpenCLBackend* mOpenCLBackend = nullptr;
    PicGateUpConvSpec mSpec;
    std::vector<uint8_t> mOpBuffer;
    const Op* mChildOp = nullptr;
    std::shared_ptr<ConvBufLowMemoryExecution> mChildConv;
    std::shared_ptr<ConvBufResource> mResource;
    static constexpr int kLocalSize = 64;
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1};
};
#endif

class FuseBufExecution : public CommonExecution {
public:
    FuseBufExecution(const std::vector<Tensor*>& inputs, Backend* backend, const Op* op)
        : CommonExecution(backend, op) {
        mUnits.resize(1);
        mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
        auto runtime = mOpenCLBackend->getOpenCLRuntime();
        std::set<std::string> buildOptions;
        auto extra = op->main_as_Extra();
        auto source = reinterpret_cast<const char*>(extra->info()->data());
        mKernelName = extra->type()->str();
        mUnits[0].kernel = runtime->buildKernelFromSource(source, extra->type()->c_str(), buildOptions,
                                                          mOpenCLBackend->getPrecision());
        OPENCL_CHECK_KERNEL_CTOR(mUnits[0].kernel);
        mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mUnits[0].kernel));
    }
    virtual ~FuseBufExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto output = outputs[0];
        auto outputShape = tensorShapeFormat(output);
        auto& unit = mUnits[0];
        const int outputBatch = outputShape.at(0);
        const int outputHeight = outputShape.at(1);
        const int outputWidth = outputShape.at(2);
        const int outputChannels = outputShape.at(3);
        const int channelBlocks = UP_DIV(outputChannels, 4);
        mGlobalWorkSize = {
            static_cast<uint32_t>(channelBlocks),
            static_cast<uint32_t>(outputWidth),
            static_cast<uint32_t>(outputHeight * outputBatch),
        };

        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        for (auto input : inputs) {
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
        }
        for (auto out : outputs) {
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(out));
        }
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);
        MNN_CHECK_CL_SUCCESS(ret, "setArg FuseBufExecution");

        mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(),
                                          mKernelName, unit.kernel, mOpenCLBackend->getCLTuneLevel(), "FuseBuf")
                             .first;
        mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        unit.globalWorkSize = cl::NDRange(mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]);
        unit.localWorkSize = cl::NDRange(mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]);
        return NO_ERROR;
    }

private:
    std::string mKernelName;
    uint32_t mMaxWorkGroupSize = 1;
    OpenCLBackend* mOpenCLBackend = nullptr;
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
};

class FuseBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto param = op->main_as_Extra();
        if (param == nullptr || param->type() == nullptr) {
            return nullptr;
        }
        if(param->type()->str() == "ExtraConvolution2DPrelu")OPENCL_CREATOR_CHECK(new ConvBufExecution(inputs, outputs, op, backend, true));
        if(param->type()->str() == "PicSiluMul") {
            OPENCL_CREATOR_CHECK(new PicSiluMulBufExecution(op, backend));
        }
        if(param->type()->str() == "PicPackedSiluMul") {
            OPENCL_CREATOR_CHECK(new PicPackedSiluMulBufExecution(op, backend));
        }
#ifdef MNN_LOW_MEMORY
        if(param->type()->str() == "PicGateUpWeightOnly") {
            OPENCL_CREATOR_CHECK(new PicGateUpWeightOnlyBufExecution(inputs, outputs, op, backend));
        }
        if(param->type()->str() == "PicGateUpSiluWeightOnly") {
            OPENCL_CREATOR_CHECK(new PicGateUpWeightOnlyBufExecution(inputs, outputs, op, backend, true));
        }
        if(param->type()->str() == "PicLinearNhwcWeightOnly") {
            OPENCL_CREATOR_CHECK(new PicLinearNhwcWeightOnlyBufExecution(inputs, outputs, op, backend));
        }
#endif
        if (param->info() != nullptr && param->info()->size() > 0) {
            OPENCL_CREATOR_CHECK(new FuseBufExecution(inputs, backend, op));
        }
        return nullptr;
    }
};
REGISTER_OPENCL_OP_CREATOR(FuseBufCreator, OpType_Extra, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
