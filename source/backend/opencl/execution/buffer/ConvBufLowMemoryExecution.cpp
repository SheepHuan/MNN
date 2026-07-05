//  ConvBufLowMemoryExecution.cpp
//
//  Created by MNN on 2023/10/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_LOW_MEMORY
#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "ConvBufLowMemoryExecution.hpp"
#include "backend/opencl/execution/buffer/ConvBufAdrenoUtils.hpp"
#include <chrono>
// #define LOG_VERBOSE
namespace MNN {
namespace OpenCL {
#define PACK_COUT 8
#define PACK_CIN 4

static bool _convProfileSelectionEnabled() {
    const char* attentionProfile = ::getenv("MNN_PAGED_ATTENTION_PROFILE");
    if (attentionProfile != nullptr && attentionProfile[0] != '\0' && attentionProfile[0] != '0') {
        return true;
    }
    const char* graphProfile = ::getenv("MNN_PIC_GRAPH_PROFILE");
    return graphProfile != nullptr && graphProfile[0] != '\0' && graphProfile[0] != '0';
}

static bool _convRequestProfileEnabled() {
    const char* value = ::getenv("MNN_PIC_REQUEST_PROFILE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool _convRequestProfileSync() {
    const char* value = ::getenv("MNN_PIC_REQUEST_PROFILE_SYNC");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static uint64_t _convProfileNowUs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

static std::string _convKernelFunctionName(const std::shared_ptr<KernelWrap>& kernel) {
    if (kernel == nullptr) {
        return "null_kernel";
    }
    std::string name;
    auto error = kernel->get().getInfo(CL_KERNEL_FUNCTION_NAME, &name);
    if (error != CL_SUCCESS || name.empty()) {
        return "unknown_kernel";
    }
    return name;
}

static size_t _convRangeDim(const cl::NDRange& range, size_t index) {
    return index < range.dimensions() ? range.get()[index] : 0;
}

enum CompactKernelMode : uint32_t {
    kCompactKernelGeneric = 0,
    kCompactKernelPic = 1,
    kCompactKernelPicB2 = 2,
    kCompactKernelPicC4 = 3,
    kCompactKernelPicWG64 = 4,
    kCompactKernelPicWG128 = 5,
    kCompactKernelAdrenoBatchGemv = 6,
    kCompactKernelPicWG4X32 = 7,
    kCompactKernelPicWG8X16 = 8,
    kCompactKernelPicInputCache32 = 9,
    kCompactKernelPicInputCache64 = 10,
    kCompactKernelAdrenoDirectC4Gemv = 11,
    kCompactKernelAdrenoBatchGemvC4Out = 12,
};

enum CompactDenseFamily : uint32_t {
    kCompactDenseFamilyGenericQuant = 0,
    kCompactDenseFamilyPicQuant = 1,
    kCompactDenseFamilyFPWeight = 2,
    kCompactDenseFamilyPicQuantB2 = 3,
    kCompactDenseFamilyPicQuantC4 = 4,
    kCompactDenseFamilyPicQuantWG64 = 5,
    kCompactDenseFamilyPicQuantWG128 = 6,
    kCompactDenseFamilyAdrenoBatchGemv = 7,
    kCompactDenseFamilyPicQuantWG4X32 = 8,
    kCompactDenseFamilyPicQuantWG8X16 = 9,
    kCompactDenseFamilyPicQuantInputCache32 = 10,
    kCompactDenseFamilyPicQuantInputCache64 = 11,
    kCompactDenseFamilyAdrenoDirectC4Gemv = 12,
    kCompactDenseFamilyAdrenoBatchGemvC4Out = 13,
};

static const char* _compactDenseFamilyName(uint32_t family) {
    switch (family) {
        case kCompactDenseFamilyGenericQuant:
            return "generic_quant";
        case kCompactDenseFamilyPicQuant:
            return "pic_quant";
        case kCompactDenseFamilyFPWeight:
            return "fp_weight";
        case kCompactDenseFamilyPicQuantB2:
            return "pic_quant_b2";
        case kCompactDenseFamilyPicQuantC4:
            return "pic_quant_c4";
        case kCompactDenseFamilyPicQuantWG64:
            return "pic_quant_wg64";
        case kCompactDenseFamilyPicQuantWG128:
            return "pic_quant_wg128";
        case kCompactDenseFamilyAdrenoBatchGemv:
            return "adreno_batch_gemv";
        case kCompactDenseFamilyPicQuantWG4X32:
            return "pic_quant_wg4x32";
        case kCompactDenseFamilyPicQuantWG8X16:
            return "pic_quant_wg8x16";
        case kCompactDenseFamilyPicQuantInputCache32:
            return "pic_quant_incache32";
        case kCompactDenseFamilyPicQuantInputCache64:
            return "pic_quant_incache64";
        case kCompactDenseFamilyAdrenoDirectC4Gemv:
            return "adreno_direct_c4_gemv";
        case kCompactDenseFamilyAdrenoBatchGemvC4Out:
            return "adreno_batch_gemv_c4out";
        default:
            return "unknown";
    }
}

static bool _parseCompactDenseFamilyOverride(const char* value, uint32_t* family) {
    if (value == nullptr || value[0] == '\0' || family == nullptr) {
        return false;
    }
    const std::pair<const char*, uint32_t> families[] = {
        {"generic_quant", kCompactDenseFamilyGenericQuant},
        {"pic_quant", kCompactDenseFamilyPicQuant},
        {"fp_weight", kCompactDenseFamilyFPWeight},
        {"pic_quant_b2", kCompactDenseFamilyPicQuantB2},
        {"pic_quant_c4", kCompactDenseFamilyPicQuantC4},
        {"pic_quant_wg64", kCompactDenseFamilyPicQuantWG64},
        {"pic_quant_wg128", kCompactDenseFamilyPicQuantWG128},
        {"adreno_batch_gemv", kCompactDenseFamilyAdrenoBatchGemv},
        {"pic_quant_wg4x32", kCompactDenseFamilyPicQuantWG4X32},
        {"pic_quant_wg8x16", kCompactDenseFamilyPicQuantWG8X16},
        {"pic_quant_incache32", kCompactDenseFamilyPicQuantInputCache32},
        {"pic_quant_incache64", kCompactDenseFamilyPicQuantInputCache64},
        {"adreno_direct_c4_gemv", kCompactDenseFamilyAdrenoDirectC4Gemv},
        {"adreno_batch_gemv_c4out", kCompactDenseFamilyAdrenoBatchGemvC4Out},
    };
    for (const auto& item : families) {
        if (::strcmp(value, item.first) == 0) {
            *family = item.second;
            return true;
        }
    }
    return false;
}

static bool _getCompactDenseFamilyOverride(uint32_t* family) {
    return _parseCompactDenseFamilyOverride(::getenv("MNN_BENCH_OPENCL_COMPACT_DENSE_FORCE_FAMILY"), family);
}

static bool _isAdrenoTinyDenseFamilyShape(OpenCLRuntime* runtime, int quantBit, int rows, int inputChannels,
                                          int outputChannels) {
    return runtime != nullptr &&
           runtime->getGpuType() == ADRENO &&
           quantBit == 4 &&
           rows > 1 && rows <= 16 &&
           inputChannels >= 1024 && outputChannels >= 256;
}

static bool _isMaliDecodeTinyDirectC4GemvShape(OpenCLRuntime* runtime, int quantBit, int rows, int inputChannels,
                                               int outputChannels) {
    return runtime != nullptr &&
           runtime->getGpuType() == MALI &&
           quantBit == 4 &&
           rows > 1 && rows <= 8 &&
           inputChannels >= 1024 && outputChannels >= 256;
}

static bool _pickAdrenoTinyDenseFamilyHeuristic(OpenCLRuntime* runtime, int quantBit, int rows, int inputChannels,
                                                int outputChannels, uint32_t* family) {
    if (family == nullptr ||
        !_isAdrenoTinyDenseFamilyShape(runtime, quantBit, rows, inputChannels, outputChannels)) {
        return false;
    }
    if (rows <= 2) {
        *family = kCompactDenseFamilyPicQuantC4;
    } else if (rows <= 4) {
        *family = kCompactDenseFamilyPicQuantInputCache64;
    } else {
        *family = kCompactDenseFamilyAdrenoBatchGemvC4Out;
    }
    return true;
}

static std::string _adrenoTinyDenseFamilyTuneKey(int quantBit, int rows, int inputChannels, int outputChannels) {
    return "adreno_tiny_dense_family_v1_q" + std::to_string(quantBit) +
           "_m" + std::to_string(rows) +
           "_ic" + std::to_string(inputChannels) +
           "_oc" + std::to_string(outputChannels);
}

static std::vector<uint32_t> _adrenoTinyDenseFamilyCandidates(int rows, int outputChannels) {
    std::vector<uint32_t> families;
    if (rows <= 2) {
        families = {
            kCompactDenseFamilyPicQuantB2,
            kCompactDenseFamilyPicQuantC4,
            kCompactDenseFamilyAdrenoDirectC4Gemv,
            kCompactDenseFamilyAdrenoBatchGemvC4Out,
            kCompactDenseFamilyPicQuantInputCache32,
            kCompactDenseFamilyPicQuantInputCache64,
            kCompactDenseFamilyPicQuant,
            kCompactDenseFamilyGenericQuant,
        };
    } else if (rows <= 4) {
        families = {
            kCompactDenseFamilyPicQuantC4,
            kCompactDenseFamilyPicQuantB2,
            kCompactDenseFamilyAdrenoDirectC4Gemv,
            kCompactDenseFamilyAdrenoBatchGemvC4Out,
            kCompactDenseFamilyPicQuantInputCache64,
            kCompactDenseFamilyPicQuantInputCache32,
            kCompactDenseFamilyPicQuant,
            kCompactDenseFamilyGenericQuant,
        };
    } else {
        families = {
            kCompactDenseFamilyAdrenoBatchGemvC4Out,
            kCompactDenseFamilyAdrenoDirectC4Gemv,
            kCompactDenseFamilyAdrenoBatchGemv,
            kCompactDenseFamilyPicQuantInputCache64,
            kCompactDenseFamilyPicQuantInputCache32,
            kCompactDenseFamilyPicQuantC4,
            kCompactDenseFamilyPicQuant,
            kCompactDenseFamilyGenericQuant,
        };
    }
    // K/V projections have only 256 output channels on MiniCPM5-1B. Keep the
    // narrower C4-output candidate early because it avoids launching twice as
    // many tiny output-channel work items when C8 occupancy is poor.
    if (outputChannels <= 256 && rows <= 4) {
        families.insert(families.begin(), kCompactDenseFamilyPicQuantC4);
    }
    std::vector<uint32_t> uniqueFamilies;
    uniqueFamilies.reserve(families.size());
    for (uint32_t family : families) {
        if (std::find(uniqueFamilies.begin(), uniqueFamilies.end(), family) == uniqueFamilies.end()) {
            uniqueFamilies.emplace_back(family);
        }
    }
    return uniqueFamilies;
}

static int _benchWeightStorageOverride() {
    const char* value = ::getenv("MNN_BENCH_OPENCL_WEIGHT_ONLY_FORCE_STORAGE");
    if (value == nullptr || value[0] == '\0') {
        return -1;
    }
    std::string mode(value);
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) { return static_cast<char>(::tolower(c)); });
    if (mode == "image" || mode == "img" || mode == "1") {
        return 1;
    }
    if (mode == "buffer" || mode == "buf" || mode == "0") {
        return 0;
    }
    return -1;
}

static void _applyBenchWeightStorageOverride(bool* useImage) {
    if (useImage == nullptr) {
        return;
    }
    const int overrideMode = _benchWeightStorageOverride();
    if (overrideMode >= 0) {
        *useImage = overrideMode == 1;
    }
}

static const char* _compactKernelModeName(uint32_t mode) {
    switch (mode) {
        case kCompactKernelGeneric:
            return "generic_b4c8";
        case kCompactKernelPic:
            return "pic_b4c8";
        case kCompactKernelPicB2:
            return "pic_b2c8";
        case kCompactKernelPicC4:
            return "pic_b4c4";
        case kCompactKernelPicWG64:
            return "pic_b4c8_wg64";
        case kCompactKernelPicWG128:
            return "pic_b4c8_wg128";
        case kCompactKernelAdrenoBatchGemv:
            return "adreno_batch_gemv";
        case kCompactKernelPicWG4X32:
            return "pic_b4c8_wg4x32";
        case kCompactKernelPicWG8X16:
            return "pic_b4c8_wg8x16";
        case kCompactKernelPicInputCache32:
            return "pic_b4c8_incache32";
        case kCompactKernelPicInputCache64:
            return "pic_b4c8_incache64";
        case kCompactKernelAdrenoDirectC4Gemv:
            return "adreno_direct_c4_gemv";
        case kCompactKernelAdrenoBatchGemvC4Out:
            return "adreno_batch_gemv_c4out";
        default:
            return "unknown";
    }
}

static int _compactKernelModeForFamily(uint32_t family) {
    switch (family) {
        case kCompactDenseFamilyPicQuant:
            return static_cast<int>(kCompactKernelPic);
        case kCompactDenseFamilyPicQuantB2:
            return static_cast<int>(kCompactKernelPicB2);
        case kCompactDenseFamilyPicQuantC4:
            return static_cast<int>(kCompactKernelPicC4);
        case kCompactDenseFamilyPicQuantWG64:
            return static_cast<int>(kCompactKernelPicWG64);
        case kCompactDenseFamilyPicQuantWG128:
            return static_cast<int>(kCompactKernelPicWG128);
        case kCompactDenseFamilyPicQuantWG4X32:
            return static_cast<int>(kCompactKernelPicWG4X32);
        case kCompactDenseFamilyPicQuantWG8X16:
            return static_cast<int>(kCompactKernelPicWG8X16);
        case kCompactDenseFamilyAdrenoBatchGemv:
            return static_cast<int>(kCompactKernelAdrenoBatchGemv);
        case kCompactDenseFamilyPicQuantInputCache32:
            return static_cast<int>(kCompactKernelPicInputCache32);
        case kCompactDenseFamilyPicQuantInputCache64:
            return static_cast<int>(kCompactKernelPicInputCache64);
        case kCompactDenseFamilyAdrenoDirectC4Gemv:
            return static_cast<int>(kCompactKernelAdrenoDirectC4Gemv);
        case kCompactDenseFamilyAdrenoBatchGemvC4Out:
            return static_cast<int>(kCompactKernelAdrenoBatchGemvC4Out);
        case kCompactDenseFamilyGenericQuant:
        default:
            return static_cast<int>(kCompactKernelGeneric);
    }
}

static uint32_t _compactDenseFamilyForSelection(bool useFPWeight, int compactKernelMode) {
    if (useFPWeight) {
        return kCompactDenseFamilyFPWeight;
    }
    switch (static_cast<uint32_t>(compactKernelMode)) {
        case kCompactKernelPic:
            return kCompactDenseFamilyPicQuant;
        case kCompactKernelPicB2:
            return kCompactDenseFamilyPicQuantB2;
        case kCompactKernelPicC4:
            return kCompactDenseFamilyPicQuantC4;
        case kCompactKernelPicWG64:
            return kCompactDenseFamilyPicQuantWG64;
        case kCompactKernelPicWG128:
            return kCompactDenseFamilyPicQuantWG128;
        case kCompactKernelPicWG4X32:
            return kCompactDenseFamilyPicQuantWG4X32;
        case kCompactKernelPicWG8X16:
            return kCompactDenseFamilyPicQuantWG8X16;
        case kCompactKernelAdrenoBatchGemv:
            return kCompactDenseFamilyAdrenoBatchGemv;
        case kCompactKernelPicInputCache32:
            return kCompactDenseFamilyPicQuantInputCache32;
        case kCompactKernelPicInputCache64:
            return kCompactDenseFamilyPicQuantInputCache64;
        case kCompactKernelAdrenoDirectC4Gemv:
            return kCompactDenseFamilyAdrenoDirectC4Gemv;
        case kCompactKernelAdrenoBatchGemvC4Out:
            return kCompactDenseFamilyAdrenoBatchGemvC4Out;
        case kCompactKernelGeneric:
        default:
            return kCompactDenseFamilyGenericQuant;
    }
}

// Discrete kernel-family decisions are not like LWS tuning: replaying the
// nearest cached shape can be actively wrong for a different M/N/K point.
// Use exact cache hits only, otherwise let heavy/wide tune re-measure.
static bool _getExactTunedInfo(const std::string& kernelName,
                               const std::vector<uint32_t>& gws,
                               std::pair<std::vector<uint32_t>, uint32_t>& tuneInfo,
                               OpenCLRuntime* runtime) {
    if (runtime == nullptr) {
        return false;
    }
    auto& tunedLws = runtime->tunedLwsMap();
    const auto exactKey = std::make_pair(kernelName, gws);
    auto tunedIter = tunedLws.find(exactKey);
    if (tunedIter != tunedLws.end()) {
        tuneInfo = std::make_pair(tunedIter->second.localSize, tunedIter->second.timeCost);
        return true;
    }
    auto& tuneLws = runtime->getTuneLwsMap();
    auto cacheIter = tuneLws.find(kernelName);
    if (cacheIter == tuneLws.end()) {
        return false;
    }
    for (const auto& info : cacheIter->second) {
        if (info.globalSize == gws) {
            tuneInfo = std::make_pair(info.localSize, info.timeCost);
            return true;
        }
    }
    return false;
}

// set mDequantScale mDequantOffset mNumQuantBit mFilterDataPtr from mConv2dParams
void ConvBufLowMemoryExecution::getInfoFromOpLowMemory(void *weight_ptr) {
    auto quanCommon = ConvolutionCommon::load(mOp, this->backend(), false, true, weight_ptr);
    if(quanCommon == nullptr){
        mValid = false;
        auto staticMapAlloc = mOpenCLBackend->getStaticAllocatorMMap();
        if(mOpenCLBackend->getRuntime()->hint().useCachedMmap && staticMapAlloc != nullptr){
            staticMapAlloc->setRemove(true);
        }
        return;
    }
    // set mResource->mNumQuantBit
    if(quanCommon->canUseInt4){
        mResource->mNumQuantBit = 4;
    }else{
        mResource->mNumQuantBit = 8;
    }
    if (mOp->main_as_Convolution2D()->common()->inputCount() > 0) {
        mResource->mInputChannel = mOp->main_as_Convolution2D()->common()->inputCount();
    } else {
        mResource->mInputChannel = quanCommon->weight.size() / (mResource->mKernelWidth * mResource->mKernelHeight * mResource->mOutputChannel);
    }
    // src of alpha in CPU
    float * dequantAlpha = quanCommon->alpha.get();
    int totalCount = quanCommon->alphaSize;
    int soSize = 1;
    if (quanCommon->asymmetric) {
        soSize = 2;
        totalCount /= 2;
        mResource->mBuildOptions.emplace("-DASYMMETRIC");
    }
    int numAlpha = mResource->mOutputChannel;
    mResource->mBlockSize = totalCount / numAlpha;
    // set mDequantScale mDequantOffset
    int numAlphaPack = ROUND_UP(numAlpha, 4);
    int fpBytes = mOpenCLBackend->fpBytes();
    int buffer_size = mResource->mBlockSize * numAlphaPack * fpBytes * soSize + sizeof(float);
    
    auto staticMapAlloc = mOpenCLBackend->getStaticAllocatorMMap();
    if(mOpenCLBackend->getRuntime()->hint().useCachedMmap && staticMapAlloc != nullptr){
        mResource->mDequantScaleOffsetBuffer = staticMapAlloc.get()->allocBuffer(buffer_size);
    }else{
        mResource->mDequantScaleOffsetBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
    }
    // transfer data from src in cpu to dst in gpu
    cl_int resBias, resScaleOffset;
    float coef = 1.0;
    
    void * dequantScaleOffsetBufferMap = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*mResource->mDequantScaleOffsetBuffer.get(), true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &resScaleOffset);
    if(mOpenCLBackend->getRuntime()->hint().useCachedMmap > 1){
        if(fpBytes == 2){
            float* coefMapPtr = (float*)(((half_float::half*)dequantScaleOffsetBufferMap) + (numAlphaPack * mResource->mBlockSize * soSize));
            coef = coefMapPtr[0];
        }else{
            coef = ((float *)dequantScaleOffsetBufferMap)[(numAlphaPack * mResource->mBlockSize * soSize)];
        }
    }else{
        if(fpBytes == 2) {
            float max_data = 0.0f;
            if (quanCommon->asymmetric){
                for (int i = 0; i < numAlpha; ++i) {
                    auto srcZ = dequantAlpha + i * mResource->mBlockSize * 2;
                    for(int j = 0; j < mResource->mBlockSize; ++j){
                        float s = fabsf(srcZ[2*j+0]);
                        float b = fabsf(srcZ[2*j+1]);
                        float temp = ALIMAX(s, b);
                        if(temp > max_data) {
                            max_data = temp;
                        }
                    }
                }
            }else{
                for (int i = 0; i < numAlpha; ++i) {
                    auto srcZ = dequantAlpha + i * mResource->mBlockSize;
                    for(int j = 0; j < mResource->mBlockSize; ++j){
                        float s = fabsf(srcZ[j]);
                        if(s > max_data) {
                            max_data = s;
                        }
                    }
                }
            }
            if(abs(max_data) >= 0.000001f){
                coef = 1000.0f / max_data;
            }
            if (dequantScaleOffsetBufferMap != nullptr && resScaleOffset == CL_SUCCESS) {
                if (quanCommon->asymmetric) {
                    for (int i = 0; i < numAlpha; ++i) {
                        auto srcZ = dequantAlpha + i * mResource->mBlockSize * 2;
                        for(int j = 0; j < mResource->mBlockSize; ++j){
                            float o = srcZ[2*j+0];
                            float s = srcZ[2*j+1];
                            ((half_float::half*)dequantScaleOffsetBufferMap)[(j * numAlphaPack + i) * 2] = (half_float::half)(s * coef);
                            ((half_float::half*)dequantScaleOffsetBufferMap)[(j * numAlphaPack + i) * 2 + 1] = (half_float::half)(o * coef);
                        }
                    }
                } else {
                    for (int i = 0; i < numAlpha; ++i) {
                        auto srcZ = dequantAlpha + i * mResource->mBlockSize;
                        for(int j = 0; j < mResource->mBlockSize; ++j){
                            ((half_float::half*)dequantScaleOffsetBufferMap)[(j * numAlphaPack + i)] = (half_float::half)(srcZ[j] * coef);
                        }
                    }
                }
                float* coefMapPtr = (float*)(((half_float::half*)dequantScaleOffsetBufferMap) + (numAlphaPack * mResource->mBlockSize * soSize));
                coefMapPtr[0] = coef;
            } else {
                MNN_ERROR("Map error dequantBufferMap == nullptr \n");
                MNN_ASSERT(false);
            }
        } else{
            if (dequantScaleOffsetBufferMap != nullptr && resScaleOffset == CL_SUCCESS) {
                if (quanCommon->asymmetric) {
                    for (int i = 0; i < numAlpha; ++i) {
                        auto srcZ = dequantAlpha + i * mResource->mBlockSize * 2;
                        for(int j = 0; j < mResource->mBlockSize; ++j){
                            float o = srcZ[2*j+0];
                            float s = srcZ[2*j+1];
                            ((float *)dequantScaleOffsetBufferMap)[(j * numAlphaPack + i) * 2] = s * coef;
                            ((float *)dequantScaleOffsetBufferMap)[(j * numAlphaPack + i) * 2 + 1] = o * coef;
                        }
                    }
                } else {
                    for (int i = 0; i < numAlpha; ++i) {
                        auto srcZ = dequantAlpha + i * mResource->mBlockSize;
                        for(int j = 0; j < mResource->mBlockSize; ++j){
                            ((float *)dequantScaleOffsetBufferMap)[(j * numAlphaPack + i)] = srcZ[j] * coef;
                        }
                    }
                }
                ((float *)dequantScaleOffsetBufferMap)[(numAlphaPack * mResource->mBlockSize * soSize)] = coef;
            } else {
                MNN_ERROR("Map error dequantBufferMap == nullptr \n");
                MNN_ASSERT(false);
            }
        }
    }
    mResource->mCoef = coef;
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mResource->mDequantScaleOffsetBuffer.get(), dequantScaleOffsetBufferMap);
    // set mFilterDataPtr
    mFilterDataPtr = (void *)quanCommon->weight.get();
}

bool ConvBufLowMemoryExecution::convertToQuantWeight1x1Buffer(cl::Buffer input) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start convertToQuantWeight1x1Buffer !\n");
#endif
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    std::string kernelName = "conv2d_1x1_weight_quant_buffer";
    if(mResource->mUseImage){
        kernelName = "conv2d_1x1_weight_quant_image";
    }
    std::set<std::string> buildOptions;
    if (mResource->mNumQuantBit == 8) {
        buildOptions.emplace("-DUSE_LOW_BIT_WEIGHT_INT8");
    } else if (mResource->mNumQuantBit == 4){
        // int4 case
        buildOptions.emplace("-DUSE_LOW_BIT_WEIGHT_INT4");
    } else {/* More types to be supported. */}

    mBufferToConv1x1Kernel = runtime->buildKernelWithCache("buffer_convert_quant", kernelName, buildOptions, mOpenCLBackend->getPrecision());
    if (mBufferToConv1x1Kernel == nullptr) {
        return false;
    }
    auto kernel = mBufferToConv1x1Kernel->get();
    uint32_t gws[2] = {static_cast<uint32_t>(UP_DIV(mResource->mInputChannel, PACK_CIN)), static_cast<uint32_t>(UP_DIV(mResource->mOutputChannel, PACK_COUT))};

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= kernel.setArg(idx++, gws[0]);
    ret |= kernel.setArg(idx++, gws[1]);
    ret |= kernel.setArg(idx++, input);
    if(mResource->mUseImage){
        ret |= kernel.setArg(idx++, *mResource->mKernelImage.get());
    }else{
        ret |= kernel.setArg(idx++, *mResource->mKernelBuffer.get());
    }
    ret |= kernel.setArg(idx++, mResource->mInputChannel);
    ret |= kernel.setArg(idx++, mResource->mOutputChannel);
    MNN_CHECK_CL_SUCCESS(ret, "setArg convertToQuantWeight1x1Buffer");

    const uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mBufferToConv1x1Kernel));
    const std::vector<uint32_t> lws = {16, std::max((uint32_t)1, maxWorkGroupSize / 16)};

    cl::Event event;
    cl_int res;

    std::vector<uint32_t> roundUpGroupWorkSize(lws.size());
    for (size_t i = 0; i < lws.size(); ++i) {
        roundUpGroupWorkSize[i] = ROUND_UP(gws[i], lws[i]);
    }

    res = runtime->commandQueue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                                         cl::NDRange(roundUpGroupWorkSize[0], roundUpGroupWorkSize[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);

    event.wait();
    MNN_CHECK_CL_SUCCESS(res, "convertToQuantWeight1x1Buffer");

#ifdef LOG_VERBOSE
    MNN_PRINT("end convertToQuantWeight1x1Buffer !\n");
#endif
    return true;
}

// set mKernelBuffer for the 1x1 kernels
void ConvBufLowMemoryExecution::set1x1WeightLowMemory() {
    bool preAllocGpuMem = mResource->mInputChannel != 0 && mResource->mConv2dParams->quanParameter();
    if(preAllocGpuMem){
        mResource->mNumQuantBit = mResource->mConv2dParams->quanParameter()->aMaxOrBits();
        if(mResource->mNumQuantBit == 0){
            // support old model for external weight file with int4/int8 quant
            mResource->mNumQuantBit = ConvolutionCommon::getQuantBitFromExternalFile(mOp);
        }
    } else{
        getInfoFromOpLowMemory(nullptr);
        if(mValid == false){
            return;
        }
    }
    cl_int res = CL_SUCCESS;
    std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({ROUND_UP(mResource->mOutputChannel, PACK_COUT), ROUND_UP(mResource->mInputChannel, PACK_CIN), 1, 1}));
    size_t buffer_size = filterBuffer->usize() / sizeof(float);
    size_t cpy_size = mResource->mOutputChannel * mResource->mInputChannel;
    int actual_packCin = PACK_CIN;
    // shared part for all cases
    if (mResource->mNumQuantBit == 4){
        // int4 case
        buffer_size /= 2;
        cpy_size = UP_DIV(cpy_size, 2);
    } else if(mResource->mNumQuantBit == 8){
        actual_packCin /= 2;
    } else {/* More types to be supported. */}
    if(mOpenCLBackend->getRuntime()->hint().useCachedMmap <= 1){
        cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
        void *mapPtr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
        if(mapPtr != nullptr && res == CL_SUCCESS){
            if(preAllocGpuMem){
                getInfoFromOpLowMemory(mapPtr);
                if(mValid == false){
                    return;
                }
            } else{
                ::memcpy(mapPtr, mFilterDataPtr, cpy_size);
            }
        } else {
            MNN_ERROR("set1x1WeightLowMemory: Map error ptrCL == nullptr \n");
            MNN_ASSERT(false);
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, mapPtr);
        // Use Image load weights
        if(UP_DIV(mResource->mInputChannel, actual_packCin) <= 16384 && ROUND_UP(mResource->mOutputChannel, PACK_COUT) <= 16384){
            mResource->mUseImage = true;
        }
        if (ConvAdreno::preferCompactDenseWeightBuffer(mOpenCLBackend->getOpenCLRuntime(), mResource->mNumQuantBit,
                                                       mResource->mInputChannel, mResource->mOutputChannel)) {
            mResource->mUseImage = false;
        }
        _applyBenchWeightStorageOverride(&mResource->mUseImage);
        auto staticMapAlloc = mOpenCLBackend->getStaticAllocatorMMap();
        if(mResource->mUseImage){
            size_t w = UP_DIV(mResource->mInputChannel, actual_packCin);
            size_t h = UP_DIV(mResource->mOutputChannel, PACK_COUT);
            if(mOpenCLBackend->getRuntime()->hint().useCachedMmap && staticMapAlloc != nullptr){
                mResource->mKernelImage = staticMapAlloc.get()->allocImage(w, h, CL_SIGNED_INT32);
            }else{
                mResource->mKernelImage.reset(new cl::Image2D(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_SIGNED_INT32), w, h, 0, nullptr, &res));
            }
            if (nullptr == mResource->mKernelImage.get() || res != CL_SUCCESS) {
                MNN_ERROR("Alloc Image %d x %d error, code:%d \n", (int)w, (int)h, (int)res);
            }
        }else{
            if(mOpenCLBackend->getRuntime()->hint().useCachedMmap && staticMapAlloc != nullptr){
                mResource->mKernelBuffer = staticMapAlloc.get()->allocBuffer(buffer_size);
            }else{
                mResource->mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
            }
        }
        convertToQuantWeight1x1Buffer(filterBufferCL);
    }else {
        if(preAllocGpuMem){
            getInfoFromOpLowMemory(nullptr);
            if(mValid == false){
                return;
            }
        }
        // Use Image load weights
        if(UP_DIV(mResource->mInputChannel, actual_packCin) <= 16384 && ROUND_UP(mResource->mOutputChannel, PACK_COUT) <= 16384){
            mResource->mUseImage = true;
        }
        if (ConvAdreno::preferCompactDenseWeightBuffer(mOpenCLBackend->getOpenCLRuntime(), mResource->mNumQuantBit,
                                                       mResource->mInputChannel, mResource->mOutputChannel)) {
            mResource->mUseImage = false;
        }
        _applyBenchWeightStorageOverride(&mResource->mUseImage);
        auto staticMapAlloc = mOpenCLBackend->getStaticAllocatorMMap();
        if(mResource->mUseImage){
            size_t w = UP_DIV(mResource->mInputChannel, actual_packCin);
            size_t h = UP_DIV(mResource->mOutputChannel, PACK_COUT);
            if(mOpenCLBackend->getRuntime()->hint().useCachedMmap && staticMapAlloc != nullptr){
                mResource->mKernelImage = staticMapAlloc.get()->allocImage(w, h, CL_SIGNED_INT32);
            }else{
                mResource->mKernelImage.reset(new cl::Image2D(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_SIGNED_INT32), w, h, 0, nullptr, &res));
            }
            if (nullptr == mResource->mKernelImage.get() || res != CL_SUCCESS) {
                MNN_ERROR("Alloc Image %d x %d error, code:%d \n", (int)w, (int)h, (int)res);
            }
        }else{
            if(mOpenCLBackend->getRuntime()->hint().useCachedMmap && staticMapAlloc != nullptr){
                mResource->mKernelBuffer = staticMapAlloc.get()->allocBuffer(buffer_size);
            }else{
                mResource->mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
            }
        }
    }
}
// set mFilter for the general kernels
void ConvBufLowMemoryExecution::setGeneralWeightLowMemory() {
    bool preAllocGpuMem = mResource->mInputChannel != 0 && mResource->mConv2dParams->quanParameter();
    if(preAllocGpuMem){
        mResource->mNumQuantBit = mResource->mConv2dParams->quanParameter()->aMaxOrBits();
        if(mResource->mNumQuantBit == 0){
            // support old model for external weight file with int4/int8 quant
            mResource->mNumQuantBit = ConvolutionCommon::getQuantBitFromExternalFile(mOp);
        }
    } else{
        getInfoFromOpLowMemory(nullptr);
        if(mValid == false){
            return;
        }
    }
    
    if(mOpenCLBackend->getRuntime()->hint().useCachedMmap <= 1){
        std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({ROUND_UP(mResource->mOutputChannel, 4), mResource->mInputChannel, mResource->mKernelWidth, mResource->mKernelHeight}));
        size_t buffer_size = filterBuffer->usize() / sizeof(float);
        size_t cpy_size = mResource->mOutputChannel * mResource->mInputChannel * mResource->mKernelWidth * mResource->mKernelHeight;
        if (mResource->mNumQuantBit == 4){
            buffer_size /= 2;
            cpy_size = UP_DIV(cpy_size, 2);
        }
        cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
        filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);
        // map and pack data from filterDataPtr
        cl_int res;
        auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &res);
        if(ptrCL != nullptr && res == CL_SUCCESS) {
            if(preAllocGpuMem){
                getInfoFromOpLowMemory(ptrCL);
                if(mValid == false){
                    return;
                }
            } else{
                ::memcpy(ptrCL, mFilterDataPtr, cpy_size);
            }
        } else {
            MNN_ERROR("setGeneralWeightLowMemory: Map error ptrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);
        if (mResource->mNumQuantBit == 8) {
            // ROUND_UP(IC, 4), UP_DIV(OC, 4) * mKernelWidth * mKernelHeight
            mResource->mFilter.reset(Tensor::createDevice<int8_t>({1, UP_DIV(mResource->mOutputChannel, 4) * mResource->mKernelWidth * mResource->mKernelHeight, 1, 4 * ROUND_UP(mResource->mInputChannel, 4)}));
            if (!(mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC))) {
                mValid = false;
                return;
            }
        } else if (mResource->mNumQuantBit == 4){
            // ROUND_UP(IC, 4), UP_DIV(OC, 4) * mKernelWidth * mKernelHeight
            // For int4 case, data stored in mFilter should be uint8_t,
            // while "Tensor::createDevice<uint8_t>" occupies more memory than "Tensor::createDevice<int8_t>".
            // Therefore, we use "Tensor::createDevice<int8_t>" currently, leaving "Tensor::createDevice<uint8_t>" to be supported.
            mResource->mFilter.reset(Tensor::createDevice<int8_t>({1, UP_DIV(mResource->mOutputChannel, 4) * mResource->mKernelWidth * mResource->mKernelHeight, 1, 2 * ROUND_UP(mResource->mInputChannel, 4)}));
            if (!(mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC))) {
                mValid = false;
                return;
            }
        }
        // convert to NC4HW4
        MNN::OpenCL::BufferConvertor bufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
        bufferConvertor.convertToNC4HW4Buffer(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mResource->mFilter.get(), mOpenCLBackend->getPrecision(), false, true, true, mResource->mNumQuantBit);
    }else{
        if(preAllocGpuMem){
            getInfoFromOpLowMemory(nullptr);
            if(mValid == false){
                return;
            }
        }
        if (mResource->mNumQuantBit == 8) {
            // ROUND_UP(IC, 4), UP_DIV(OC, 4) * mKernelWidth * mKernelHeight
            mResource->mFilter.reset(Tensor::createDevice<int8_t>({1, UP_DIV(mResource->mOutputChannel, 4) * mResource->mKernelWidth * mResource->mKernelHeight, 1, 4 * ROUND_UP(mResource->mInputChannel, 4)}));
            if (!(mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC))) {
                mValid = false;
                return;
            }
        } else if (mResource->mNumQuantBit == 4){
            // ROUND_UP(IC, 4), UP_DIV(OC, 4) * mKernelWidth * mKernelHeight
            // For int4 case, data stored in mFilter should be uint8_t,
            // while "Tensor::createDevice<uint8_t>" occupies more memory than "Tensor::createDevice<int8_t>".
            // Therefore, we use "Tensor::createDevice<int8_t>" currently, leaving "Tensor::createDevice<uint8_t>" to be supported.
            mResource->mFilter.reset(Tensor::createDevice<int8_t>({1, UP_DIV(mResource->mOutputChannel, 4) * mResource->mKernelWidth * mResource->mKernelHeight, 1, 2 * ROUND_UP(mResource->mInputChannel, 4)}));
            if (!(mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC))) {
                mValid = false;
                return;
            }
        }
    }

}
// select the fastest kernel for the general cases by tuning
void ConvBufLowMemoryExecution::tuneGeneralCaseLowMemory(Tensor * input, Tensor * output) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int batch              = outputShape.at(0);
    const int height             = outputShape.at(1);
    const int width              = outputShape.at(2);
    const int outChannel         = outputShape.at(3);
    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);
    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    const int blockDim = mResource->mInputChannel / mResource->mBlockSize;
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(outChannel) + "_" + std::to_string(mResource->mKernelHeight) + "_" + std::to_string(mResource->mKernelWidth) + "_" + std::to_string(mResource->mStrides[0]) + "_" + std::to_string(mResource->mStrides[1]) + "_" + std::to_string(mResource->mDilations[0]) + "_" + std::to_string(mResource->mDilations[1]);
    int inputImageShape[2]  = {inputHeight, inputWidth};
    int outputImageShape[2] = {height, width};
    int kernelShape[2]      = {mResource->mKernelHeight, mResource->mKernelWidth};
    int strideShape[2]      = {mResource->mStrides[0], mResource->mStrides[1]};
    int paddingShape[2]     = {mPaddings[0], mPaddings[1]};
    int dilationShape[2]    = {mResource->mDilations[0], mResource->mDilations[1]};
    // {"conv_2d_c4h1w2", "conv_2d_c4h1w1", "conv_2d_c8h1w1", "conv_2d_c4h1w4", "conv_2d_c8h2w1", "conv_2d_c4h4w1"};
    const int total_kernel = 4;
    std::string kernelName[total_kernel] = {"conv_2d_int_c4h1w1", "conv_2d_int_c4h1w2", "conv_2d_int_c4h1w4", "conv_2d_int_c8h1w4"};
    int itemC[total_kernel] = {4, 4, 4, 8};
    int itemH[total_kernel] = {1, 1, 1, 1};
    int itemW[total_kernel] = {1, 2, 4, 4};
    int actual_kernel = total_kernel;
    std::shared_ptr<KernelWrap> kernel[total_kernel];
    std::vector<uint32_t> globalWorkSize[total_kernel];
    std::vector<uint32_t> localWorkSize[total_kernel];
    std::pair<int, int> min_cost(INT_MAX, 0);//(min_time, min_index)
    // MNN_PRINT("Checking kernel %d.\n", knlCheck);
    for (int knl_idx = 0; knl_idx < actual_kernel; knl_idx++) {
        std::set<std::string> buildOption = mResource->mBuildOptions;
        if(itemC[knl_idx] == 8 && outputShape.at(3) % itemC[knl_idx] > 0 && outputShape.at(3) % itemC[knl_idx] <= 4){
            buildOption.emplace("-DCHANNEL_BOUNDARY_PROTECT");
        }
        if((outputShape.at(2) % itemW[knl_idx]) != 0 || (outputShape.at(1) % itemH[knl_idx]) != 0){
            buildOption.emplace("-DBLOCK_LEAVE");
        }
        if(inputChannels % 4 != 0){
            buildOption.emplace("-DINPUT_CHANNEL_BOUNDARY_PROTECT");
        }
        kernel[knl_idx]        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_int_buf", kernelName[knl_idx], buildOption, mOpenCLBackend->getPrecision());
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel[knl_idx]));

        globalWorkSize[knl_idx] = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), itemC[knl_idx]) * UP_DIV(outputShape.at(2), itemW[knl_idx])), static_cast<uint32_t>(outputShape.at(0) * UP_DIV(outputShape.at(1), itemH[knl_idx]))};
        uint32_t idx            = 0;
        cl_int ret = CL_SUCCESS;
        ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][0]);
        ret |= kernel[knl_idx]->get().setArg(idx++, globalWorkSize[knl_idx][1]);
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(input));
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
        ret |= kernel[knl_idx]->get().setArg(idx++, *mResource->mDequantScaleOffsetBuffer.get());
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
        ret |= kernel[knl_idx]->get().setArg(idx++, openCLBuffer(output));
        ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
        ret |= kernel[knl_idx]->get().setArg(idx++, inputChannels);
        ret |= kernel[knl_idx]->get().setArg(idx++, inputChannelBlocks);
        ret |= kernel[knl_idx]->get().setArg(idx++, batch);
        ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
        ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(kernelShape), kernelShape);
        ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(strideShape), strideShape);
        ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(paddingShape), paddingShape);
        ret |= kernel[knl_idx]->get().setArg(idx++, sizeof(dilationShape), dilationShape);
        ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(width, itemW[knl_idx]));
        ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(outChannel, 4));
        ret |= kernel[knl_idx]->get().setArg(idx++, UP_DIV(height, itemH[knl_idx]));
        ret |= kernel[knl_idx]->get().setArg(idx++, blockDim);
        ret |= kernel[knl_idx]->get().setArg(idx++, static_cast<float>(mResource->mCoef));
        MNN_CHECK_CL_SUCCESS(ret, "setArg ConvBufLowMemory Kernel Select");
        std::pair<std::vector<uint32_t>, int> retTune;
        retTune = localWS2DDefault(globalWorkSize[knl_idx], maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName[knl_idx] + info, kernel[knl_idx], mOpenCLBackend->getCLTuneLevel(), "conv_2d_int_buf");
        if(min_cost.first > retTune.second) {
            min_cost.first = retTune.second;
            min_cost.second = knl_idx;
            mLocalWorkSize = {retTune.first[0], retTune.first[1]};
        }
    }
    int min_index  = min_cost.second;
    mGlobalWorkSize = {globalWorkSize[min_index][0], globalWorkSize[min_index][1]};

    std::set<std::string> buildOption = mResource->mBuildOptions;
    if(itemC[min_index] == 8 && outputShape.at(3) % itemC[min_index] > 0 && outputShape.at(3) % itemC[min_index] <= 4){
        buildOption.emplace("-DCHANNEL_BOUNDARY_PROTECT");
    }
    if((outputShape.at(2) % itemW[min_index]) != 0 || (outputShape.at(1) % itemH[min_index]) != 0){
        buildOption.emplace("-DBLOCK_LEAVE");
    }
    if(inputChannels % 4 != 0){
        buildOption.emplace("-DINPUT_CHANNEL_BOUNDARY_PROTECT");
    }
    unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_int_buf", kernelName[min_index], buildOption, mOpenCLBackend->getPrecision());

    uint32_t idx            = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
    ret |= unit.kernel->get().setArg(idx++, *mResource->mDequantScaleOffsetBuffer.get());
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
    ret |= unit.kernel->get().setArg(idx++, inputChannels);
    ret |= unit.kernel->get().setArg(idx++, inputChannelBlocks);
    ret |= unit.kernel->get().setArg(idx++, batch);
    ret |= unit.kernel->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
    ret |= unit.kernel->get().setArg(idx++, sizeof(kernelShape), kernelShape);
    ret |= unit.kernel->get().setArg(idx++, sizeof(strideShape), strideShape);
    ret |= unit.kernel->get().setArg(idx++, sizeof(paddingShape), paddingShape);
    ret |= unit.kernel->get().setArg(idx++, sizeof(dilationShape), dilationShape);
    ret |= unit.kernel->get().setArg(idx++, UP_DIV(width, itemW[min_index]));
    ret |= unit.kernel->get().setArg(idx++, UP_DIV(outChannel, 4));
    ret |= unit.kernel->get().setArg(idx++, UP_DIV(height, itemH[min_index]));
    ret |= unit.kernel->get().setArg(idx++, blockDim);
    ret |= unit.kernel->get().setArg(idx++, static_cast<float>(mResource->mCoef));
    MNN_CHECK_CL_SUCCESS(ret, "setArg ConvBufLowMemory");
    mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    return;
}

// weight inverse quantization, use xgemm opt
void ConvBufLowMemoryExecution::useFPWeightGemmLowMemory(Tensor * input, Tensor * output) {
    mUnits.resize(3);
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    int channelPack = 2;
    if(mResource->mNumQuantBit == 4){
        channelPack = 4;
    }
    int area = inputShape.at(1) * inputShape.at(2);
    int M = outputShape.at(0) * area;
    int N = mResource->mOutputChannel;
    int K = mResource->mInputChannel;
    int mAlignK = 4;
    int mAlignN = 16;
    int mAlignM = 64;
    
    // set M Align and N Align
    if(mResource->mOutputChannel > 1024) {
        mAlignN = 128;
    } else if(mResource->mOutputChannel > 512) {
        mAlignN = 64;
    } else if(mResource->mOutputChannel > 96) {
        mAlignN = 32;
    }
    float ratio = 1.0 * M / 1024.0 * N / 1024.0 * K / 1024.0;
    if(M > 1024 && ratio >= 1.0) {
        mAlignM = 128;
    } else if(M > 512 && ratio >= 0.1) {
        mAlignM = 64;
    } else if(M > 192 && K >= 1024 && N >= 1024 && ratio >= 0.35) {
        // OrangePi Mali has a weak compact-row Xgemm bucket when high-arithmetic
        // PIC MLP shapes stay in the 32-row align regime, for example
        // M~=216 -> alignM=224 and M~=471 -> alignM=480. Promote these
        // large-channel mid-M shapes into the 64-row bucket so they round to
        // 256/320/512 and reuse the stronger Xgemm parameter space.
        mAlignM = 64;
    } else if(M > 96){
        mAlignM = 32;
    } else {
        mAlignM = 16;
    }
    int alignM = ROUND_UP(M, mAlignM);
    int alignN = ROUND_UP(N, mAlignN);
    int alignK = ROUND_UP(K, mAlignK);
    int blockDim = mResource->mInputChannel / mResource->mBlockSize;
    
    // alloc temp bufer
    mConvGemmWeightTensor.reset(Tensor::createDevice<float>({ROUND_UP(mResource->mOutputChannel, mAlignN) * ROUND_UP(mResource->mInputChannel, std::max(mAlignK, channelPack))}));
    mConvGemmInpTensor.reset(Tensor::createDevice<float>({alignK * alignM}));
    mConvGemmOutTensor.reset(Tensor::createDevice<float>({alignN * alignM}));
    mOpenCLBackend->onAcquireBuffer(mConvGemmWeightTensor.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mConvGemmOutTensor.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mConvGemmInpTensor.get(), Backend::DYNAMIC);
    
    //weight inverse quantization and rearrange
    {
        auto &unit = mUnits[0];
        int outputChannelAlign = ROUND_UP(mResource->mOutputChannel, alignN);
        int outputChannel4Align = ROUND_UP(mResource->mOutputChannel, 4);
        int inputChannel4Align = ROUND_UP(mResource->mInputChannel, 4);
        std::set<std::string> buildOption = mResource->mBuildOptions;
        if(mResource->mUseImage){
            buildOption.emplace("-DUSE_IMAGE");
        }
        mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(mResource->mInputChannel, channelPack)), static_cast<uint32_t>(UP_DIV(mResource->mOutputChannel, 8))};
        unit.kernel = runtime->buildKernel("gemm_conv1x1_buf", "inverse_quant_weight", buildOption, mOpenCLBackend->getPrecision());
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
        ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
        if(mResource->mUseImage){
            ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelImage.get());
        }else{
            ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
        }
        ret |= unit.kernel->get().setArg(idx++, *mResource->mDequantScaleOffsetBuffer.get());
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvGemmWeightTensor.get()));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mResource->mInputChannel));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannel4Align));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelAlign));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannel4Align));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockDim));
        ret |= unit.kernel->get().setArg(idx++, static_cast<float>(mResource->mCoef));
        MNN_CHECK_CL_SUCCESS(ret, "setArg inverse_quant_weight");
        
        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, runtime, "inverse_quant_weight", unit.kernel, mOpenCLBackend->getCLTuneLevel(), "gemm_conv1x1_buf").first;
        mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
        unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    }
    
    // rearange input
    {
        auto &unit = mUnits[1];
        std::set<std::string> buildOptions = mResource->mBuildOptions;
        
        int m_pack = 4;
        mGlobalWorkSize = {static_cast<uint32_t>(alignM/m_pack), static_cast<uint32_t>(alignK/4)};
        unit.kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_buf", "transpose_pad", buildOptions, mOpenCLBackend->getPrecision());
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));

        int offset = 0;
        int idx            = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[0]));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[1]));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(alignM));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(alignK));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(M));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(K));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(area));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvGemmInpTensor.get()));
        MNN_CHECK_CL_SUCCESS(ret, "setArg transpose_pad");
        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, runtime, "transpose_pad", unit.kernel, mOpenCLBackend->getCLTuneLevel(), "gemm_buf").first;

        mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
        unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    }
    
    // call gemm strassen
    {
        mStrassenComputor.reset(new StrassenMatrixComputor(backend(), 3));
        mStrassenComputor->onEncode(alignM, alignK, alignN, alignM, alignN, alignN, openCLBuffer(mConvGemmInpTensor.get()), openCLBuffer(mConvGemmWeightTensor.get()), openCLBuffer(mConvGemmOutTensor.get()), false, openCLBuffer(mResource->mBias.get()));
    }
        
    // call output transpose
    {
        auto &unit = mUnits[2];
        std::set<std::string> buildOptions = mResource->mBuildOptions;
        int pack_m = 1;
        if (M >= 256) {
            pack_m = 8;
        } else if (M >= 96) {
            // Compact PIC rows like 115 / 317 / 418 / 519 were previously
            // forced onto the scalar transpose path only because M had tail
            // rows. The kernel can safely guard the tail, so keep vectorized
            // transpose for these dense-heavy shapes.
            pack_m = 4;
        } else if (M >= 32) {
            pack_m = 2;
        }
        buildOptions.emplace("-DM_VEC=" + std::to_string(pack_m));
        // Generate cache for every option used by the runtime heuristic.
        std::vector<int> pack_m_vec = {1, 2, 4, 8};
        for (auto p : pack_m_vec) {
            auto option = mResource->mBuildOptions;
            option.emplace("-DM_VEC=" + std::to_string(p));
            auto kernel = runtime->buildKernel("gemm_buf", "transpose_bias", option, mOpenCLBackend->getPrecision());
        }
        unit.kernel = runtime->buildKernel("gemm_buf", "transpose_bias", buildOptions, mOpenCLBackend->getPrecision());
        uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

        mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(M, pack_m)), static_cast<uint32_t>(UP_DIV(N, 4))};

        int offset = 0;
        int idx            = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[0]));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[1]));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(alignM));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(alignN));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(M));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(N));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(area));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvGemmOutTensor.get()));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
        ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));

        MNN_CHECK_CL_SUCCESS(ret, "setArg transpose_bias");
        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, runtime, "transpose_bias", unit.kernel, mOpenCLBackend->getCLTuneLevel(), "gemm_buf").first;
        mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
        unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    }
    mOpenCLBackend->onReleaseBuffer(mConvGemmWeightTensor.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mConvGemmInpTensor.get(), Backend::DYNAMIC);
    mOpenCLBackend->onReleaseBuffer(mConvGemmOutTensor.get(), Backend::DYNAMIC);
    
    return;
}
void ConvBufLowMemoryExecution::tuneGemvLowMemory(Tensor * input, Tensor * output) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int outChannel = outputShape.at(3);
    const int inputChannels = inputShape.at(3);
    const int batch = outputShape.at(0);
    const int height = outputShape.at(1);
    const int width = outputShape.at(2);
    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    const int outputChannelBlocks = UP_DIV(outChannel, 4);
    const int blockNum = mResource->mBlockSize;
    const int blockDim = mResource->mInputChannel / mResource->mBlockSize;
    bool useLocalMem = inputChannels >= 32;
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(outChannel);
    std::set<std::string> buildOption = mResource->mBuildOptions;
    int inputChannelLeaves = 0;
    if(mResource->mNumQuantBit == 4){
        inputChannelLeaves = useLocalMem ? (inputChannels % 4) : (blockDim % 4);
    } else {
        inputChannelLeaves = useLocalMem ? (inputChannels % 2) : (blockDim % 2);
    }
    if(outChannel % 8 != 0){
        buildOption.emplace("-DOUTPUT_CHANNEL_LEAVES");
    }
    buildOption.emplace("-DINPUT_CHANNEL_LEAVES_NUM=" + std::to_string(inputChannelLeaves));
    if(mResource->mUseImage){
        buildOption.emplace("-DUSE_IMAGE");
    }
    
    int local_size = useLocalMem ? 128 : 1;
    if(useLocalMem && mOpenCLBackend->getCLTuneLevel() != None && mOpenCLBackend->getCLTuneLevel() != Fast){
        int min_time = INT_MAX;
        for (int ksize = 8; ksize <= 256; ksize*=2) {
            auto option = buildOption;
            option.emplace("-DWGS=" + std::to_string(ksize));
            auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemv_conv1x1_buf", "gemv_conv_c8_buf", option, mOpenCLBackend->getPrecision());
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
            std::vector<uint32_t> gws = {static_cast<uint32_t>(ksize), static_cast<uint32_t>(UP_DIV(outChannel, 8))};
            std::vector<uint32_t> lws = {static_cast<uint32_t>(ksize), 1};
            uint32_t idx = 0;
            cl_int ret = CL_SUCCESS;
            ret |= kernel->get().setArg(idx++, static_cast<int>(gws[0]));
            ret |= kernel->get().setArg(idx++, static_cast<int>(gws[1]));
            ret |= kernel->get().setArg(idx++, static_cast<int>(gws[1]));
            ret |= kernel->get().setArg(idx++, openCLBuffer(input));
            if(mResource->mUseImage){
                ret |= kernel->get().setArg(idx++, *mResource->mKernelImage.get());
            }else{
                ret |= kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
            }
            ret |= kernel->get().setArg(idx++, *mResource->mDequantScaleOffsetBuffer.get());
            ret |= kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
            ret |= kernel->get().setArg(idx++, openCLBuffer(output));
            ret |= kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
            ret |= kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
            ret |= kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
            ret |= kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
            ret |= kernel->get().setArg(idx++, inputChannels);
            ret |= kernel->get().setArg(idx++, static_cast<int>(blockNum));
            ret |= kernel->get().setArg(idx++, static_cast<int>(blockDim));
            ret |= kernel->get().setArg(idx++, static_cast<float>(mResource->mCoef));
            MNN_CHECK_CL_SUCCESS(ret, "setArg gemv_conv_c8_buf Kernel Select");
            std::pair<std::vector<uint32_t>, int> retTune;
            int cost_time = get2DUseLocalMemTime(gws, lws, mOpenCLBackend->getOpenCLRuntime(), "gemv_conv_c8_buf" + info, kernel, "gemv_conv1x1_buf");
            if(min_time > cost_time) {
                local_size = ksize;
                min_time = cost_time;
            }
        }
    }
    
    buildOption.emplace("-DWGS=" + std::to_string(local_size));
    mGlobalWorkSize = {static_cast<uint32_t>(local_size), static_cast<uint32_t>(UP_DIV(outChannel, 8))};
    unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemv_conv1x1_buf", "gemv_conv_c8_buf", buildOption, mOpenCLBackend->getPrecision());
    uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[0]));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[1]));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[1]));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
    if(mResource->mUseImage){
        ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelImage.get());
    }else{
        ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
    }
    ret |= unit.kernel->get().setArg(idx++, *mResource->mDequantScaleOffsetBuffer.get());
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannels));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockNum));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockDim));
    ret |= unit.kernel->get().setArg(idx++, static_cast<float>(mResource->mCoef));
    MNN_CHECK_CL_SUCCESS(ret, "setArg gemv_conv_c8_buf");
    if(useLocalMem){
        mLocalWorkSize = {static_cast<uint32_t>(local_size), 1};
    }else{
        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "gemv_conv_c8_buf" + info, unit.kernel, mOpenCLBackend->getCLTuneLevel(), "gemv_conv1x1_buf").first;
    }
    mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    return;
}

bool ConvBufLowMemoryExecution::usePicCompactGemmLowMemory(Tensor * input, Tensor * output) const {
    if (!mResource->mConv1x1Opt || mResource->mPrelu) {
        return false;
    }
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int inputChannels = inputShape.at(3);
    const int outChannel = outputShape.at(3);
    const int globalY = outputShape.at(0) * outputShape.at(1) * outputShape.at(2);
    const int minChannel = std::min(inputChannels, outChannel);

    // PIC sparse recompute produces compact rows after the score layer. These
    // shapes are large Linear ops with small-M / large-N,K and need their own
    // OpenCL tune namespace instead of reusing generic conv1x1 LWS history.
    //
    // Keep this ratio-aware. Hard row cutoffs alone make 623-row active shapes
    // fall back to the generic dense path even when they are still well below
    // the channel dimensions and behave like compact-row matmuls.
    return globalY > 16 && globalY <= 768 && inputChannels >= 1024 &&
           outChannel >= 1024 && globalY <= minChannel;
}

bool ConvBufLowMemoryExecution::tuneGemmLowMemory(Tensor * input, Tensor * output, int compactKernelMode) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int outChannel = outputShape.at(3);
    const int inputChannels = inputShape.at(3);
    const int batch = outputShape.at(0);
    const int width_height = outputShape.at(1) * outputShape.at(2);
    const int inputChannelAlign = ROUND_UP(inputChannels, 4);
    const int outputChannelAlign = ROUND_UP(outChannel, 4);
    const int blockNum = mResource->mBlockSize;
    const int blockDim = mResource->mInputChannel / mResource->mBlockSize;
    
    int global_y = batch * width_height;
    const bool defaultPicCompactKernel = usePicCompactGemmLowMemory(input, output);
    const uint32_t resolvedCompactKernelMode = compactKernelMode < 0
        ? (defaultPicCompactKernel ? static_cast<uint32_t>(kCompactKernelPic)
                                   : static_cast<uint32_t>(kCompactKernelGeneric))
        : static_cast<uint32_t>(compactKernelMode);
    const bool usePicCompactKernel = resolvedCompactKernelMode != kCompactKernelGeneric;
    const bool useDirectC4GemvCompactPath =
        resolvedCompactKernelMode == kCompactKernelAdrenoDirectC4Gemv;
    const bool useBatchGemvC4OutputPath =
        resolvedCompactKernelMode == kCompactKernelAdrenoBatchGemvC4Out;
    const bool useBatchGemvCompactPath =
        !useDirectC4GemvCompactPath &&
        (global_y <= 16 || resolvedCompactKernelMode == kCompactKernelAdrenoBatchGemv ||
         useBatchGemvC4OutputPath);
    const int compactBatchTile = resolvedCompactKernelMode == kCompactKernelPicB2 ? 2 : 4;
    const int compactChannelTile = resolvedCompactKernelMode == kCompactKernelPicC4 ? 4 : 8;
    std::string kernelName;
    switch (resolvedCompactKernelMode) {
        case kCompactKernelPic:
            kernelName = "pic_gemm_b4_c8";
            break;
        case kCompactKernelPicB2:
            kernelName = "pic_gemm_b2_c8";
            break;
        case kCompactKernelPicC4:
            kernelName = "pic_gemm_b4_c4";
            break;
        case kCompactKernelPicWG64:
            kernelName = "pic_gemm_b4_c8_wg64";
            break;
        case kCompactKernelPicWG128:
            kernelName = "pic_gemm_b4_c8_wg128";
            break;
        case kCompactKernelPicWG4X32:
            kernelName = "pic_gemm_b4_c8_wg4x32";
            break;
        case kCompactKernelPicWG8X16:
            kernelName = "pic_gemm_b4_c8_wg8x16";
            break;
        case kCompactKernelPicInputCache32:
            kernelName = "pic_gemm_b4_c8_incache32";
            break;
        case kCompactKernelPicInputCache64:
            kernelName = "pic_gemm_b4_c8_incache64";
            break;
        case kCompactKernelAdrenoBatchGemv:
            kernelName.clear();
            break;
        case kCompactKernelGeneric:
        default:
            kernelName = "gemm_b4_c8";
            break;
    }
    std::set<std::string> buildOption = mResource->mBuildOptions;
    int inputChannelLeaves = 0;
    int inputBatchLeaves = global_y % compactBatchTile;
    if(mResource->mNumQuantBit == 4){
        inputChannelLeaves = blockDim % 4;
        if (!kernelName.empty()) {
            kernelName += "_int4_buf";
        }
    } else {
        inputChannelLeaves = blockDim % 4;
        if (!kernelName.empty()) {
            kernelName += "_int8_buf";
        }
    }
    buildOption.emplace("-DINPUT_CHANNEL_LEAVES_NUM=" + std::to_string(inputChannelLeaves));
    buildOption.emplace("-DINPUT_BATCH_LEAVES_NUM=" + std::to_string(inputBatchLeaves));
    if(mResource->mUseImage){
        buildOption.emplace("-DUSE_IMAGE");
    }
    if (!useBatchGemvCompactPath) {
        // generate cache for every option
        for (int i = 0; i < compactBatchTile; i++) {
            std::set<std::string> option = mResource->mBuildOptions;
            if(mResource->mUseImage){
                option.emplace("-DUSE_IMAGE");
            }
            option.emplace("-DINPUT_CHANNEL_LEAVES_NUM=" + std::to_string(inputChannelLeaves));
            option.emplace("-DINPUT_BATCH_LEAVES_NUM=" + std::to_string(i));
            auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_conv1x1_buf", kernelName, option, mOpenCLBackend->getPrecision());
            if (nullptr == kernel.get()) {
                MNN_ERROR("OpenCLConvBufLowMemory preload kernel unavailable: %s ic=%d oc=%d rows=%d tail=%d\n",
                          kernelName.c_str(), inputChannels, outChannel, global_y, i);
            }
        }
    }
    std::string info = std::to_string(inputChannels) + "_" + std::to_string(outChannel);
    if (usePicCompactKernel) {
        if (resolvedCompactKernelMode == kCompactKernelPicB2) {
            info += "_picb2_m" + std::to_string(global_y);
        } else if (resolvedCompactKernelMode == kCompactKernelPicC4) {
            info += "_picc4_m" + std::to_string(global_y);
        } else if (resolvedCompactKernelMode == kCompactKernelAdrenoBatchGemv) {
            info += "_adbgv_m" + std::to_string(global_y);
        } else if (resolvedCompactKernelMode == kCompactKernelPicInputCache32) {
            info += "_picic32_m" + std::to_string(global_y);
        } else if (resolvedCompactKernelMode == kCompactKernelPicInputCache64) {
            info += "_picic64_m" + std::to_string(global_y);
        } else if (resolvedCompactKernelMode == kCompactKernelAdrenoDirectC4Gemv) {
            info += "_adc4gemv_m" + std::to_string(global_y);
        } else if (resolvedCompactKernelMode == kCompactKernelAdrenoBatchGemvC4Out) {
            info += "_adbgvc4out_m" + std::to_string(global_y);
        } else {
            info += "_pic_m" + std::to_string(global_y);
        }
    }
    if (useDirectC4GemvCompactPath) {
        if (mResource->mNumQuantBit != 4 || global_y <= 1 || global_y > 16) {
            return false;
        }
        mUnits.resize(1);
        const int outputChannelBlocks = UP_DIV(outChannel, 4);
        const int inputChannelBlocks = UP_DIV(inputChannels, 4);
        std::set<std::string> directBuildOption = mResource->mBuildOptions;
        if (mResource->mUseImage) {
            directBuildOption.emplace("-DUSE_IMAGE");
        }
        directBuildOption.emplace("-DINPUT_CHANNEL_LEAVES_NUM=" + std::to_string(inputChannelLeaves));
        directBuildOption.emplace("-DINPUT_BATCH_LEAVES_NUM=" + std::to_string(inputBatchLeaves));

        int local_size = 64;
        if (mOpenCLBackend->getCLTuneLevel() != None && mOpenCLBackend->getCLTuneLevel() != Fast) {
            int min_time = INT_MAX;
            for (int ksize = 16; ksize <= 256; ksize *= 2) {
                auto option = directBuildOption;
                option.emplace("-DWGS=" + std::to_string(ksize));
                auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel(
                    "gemv_conv1x1_buf", "gemv_conv_c8_c4nhw4_buf", option, mOpenCLBackend->getPrecision());
                uint32_t maxWorkGroupSize =
                    static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
                if (maxWorkGroupSize < static_cast<uint32_t>(ksize)) {
                    continue;
                }
                std::vector<uint32_t> gws = {
                    static_cast<uint32_t>(ksize),
                    static_cast<uint32_t>(UP_DIV(outChannel, 8)),
                    static_cast<uint32_t>(UP_DIV(global_y, 4)),
                };
                std::vector<uint32_t> lws = {static_cast<uint32_t>(ksize), 1, 1};
                uint32_t idx = 0;
                cl_int ret = CL_SUCCESS;
                ret |= kernel->get().setArg(idx++, static_cast<int>(gws[0]));
                ret |= kernel->get().setArg(idx++, static_cast<int>(gws[1]));
                ret |= kernel->get().setArg(idx++, static_cast<int>(gws[2]));
                ret |= kernel->get().setArg(idx++, openCLBuffer(input));
                if (mResource->mUseImage) {
                    ret |= kernel->get().setArg(idx++, *mResource->mKernelImage.get());
                } else {
                    ret |= kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
                }
                ret |= kernel->get().setArg(idx++, *mResource->mDequantScaleOffsetBuffer.get());
                ret |= kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
                ret |= kernel->get().setArg(idx++, openCLBuffer(output));
                ret |= kernel->get().setArg(idx++, static_cast<int>(global_y));
                ret |= kernel->get().setArg(idx++, static_cast<int>(outputChannelAlign));
                ret |= kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
                ret |= kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
                ret |= kernel->get().setArg(idx++, static_cast<int>(inputChannels));
                ret |= kernel->get().setArg(idx++, static_cast<int>(blockNum));
                ret |= kernel->get().setArg(idx++, static_cast<int>(blockDim));
                ret |= kernel->get().setArg(idx++, static_cast<float>(mResource->mCoef));
                MNN_CHECK_CL_SUCCESS(ret, "setArg gemv_conv_c8_c4nhw4_buf Kernel Select");
                int cost_time = get2DUseLocalMemTime(
                    gws, lws, mOpenCLBackend->getOpenCLRuntime(),
                    "gemv_conv_c8_c4nhw4_buf" + info, kernel, "gemv_conv1x1_buf");
                if (min_time > cost_time) {
                    local_size = ksize;
                    min_time = cost_time;
                }
            }
        }

        directBuildOption.emplace("-DWGS=" + std::to_string(local_size));
        auto &unit = mUnits[0];
        mGlobalWorkSize = {
            static_cast<uint32_t>(local_size),
            static_cast<uint32_t>(UP_DIV(outChannel, 8)),
            static_cast<uint32_t>(UP_DIV(global_y, 4)),
        };
        unit.kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel(
            "gemv_conv1x1_buf", "gemv_conv_c8_c4nhw4_buf", directBuildOption, mOpenCLBackend->getPrecision());
        if (nullptr == unit.kernel.get()) {
            return false;
        }
        uint32_t maxWorkGroupSize =
            static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));
        if (maxWorkGroupSize < static_cast<uint32_t>(local_size)) {
            return false;
        }
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
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(global_y));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelAlign));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannels));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockNum));
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockDim));
        ret |= unit.kernel->get().setArg(idx++, static_cast<float>(mResource->mCoef));
        MNN_CHECK_CL_SUCCESS(ret, "setArg gemv_conv_c8_c4nhw4_buf");
        mLocalWorkSize = {static_cast<uint32_t>(local_size), 1, 1};
        mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
        unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
        unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
        return true;
    }
    if(useBatchGemvCompactPath) {
        mUnits.resize(useBatchGemvC4OutputPath ? 2 : 3);
        int outputChannelAlign8 = ROUND_UP(outChannel, 8);
        mConvGemmInpTensor.reset(Tensor::createDevice<float>({inputChannelAlign * ROUND_UP(global_y, 4)}));
        mOpenCLBackend->onAcquireBuffer(mConvGemmInpTensor.get(), Backend::DYNAMIC);
        if (!useBatchGemvC4OutputPath) {
            mConvGemmOutTensor.reset(Tensor::createDevice<float>({outputChannelAlign8 * ROUND_UP(global_y, 4)}));
            mOpenCLBackend->onAcquireBuffer(mConvGemmOutTensor.get(), Backend::DYNAMIC);
        }
        mOpenCLBackend->onReleaseBuffer(mConvGemmInpTensor.get(), Backend::DYNAMIC);
        if (!useBatchGemvC4OutputPath) {
            mOpenCLBackend->onReleaseBuffer(mConvGemmOutTensor.get(), Backend::DYNAMIC);
        }
        
        {
            //c4nhw4 -> nhwc
            auto &unit = mUnits[0];
            unit.kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_conv1x1_buf", "gemm_c4nhw4_to_nhwc", buildOption, mOpenCLBackend->getPrecision());
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));
            
            mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(global_y, 4)), static_cast<uint32_t>(UP_DIV(inputChannels, 4))};
            uint32_t idx = 0;
            cl_int ret = CL_SUCCESS;
            ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
            ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvGemmInpTensor.get()));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(global_y));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannels));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelAlign));
            MNN_CHECK_CL_SUCCESS(ret, "setArg gemm_c4nhw4_to_nhwc");
            mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "gemm_c4nhw4_to_nhwc", unit.kernel, mOpenCLBackend->getCLTuneLevel(), "gemm_conv1x1_buf").first;
            mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
            unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
            unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
        }
        {
            const int inputChannelBlocks = UP_DIV(inputChannels, 4);
            const int outputChannelBlocks = UP_DIV(outChannel, 4);
            auto &unit = mUnits[1];
            std::set<std::string> buildOption = mResource->mBuildOptions;
            if(mResource->mUseImage){
                buildOption.emplace("-DUSE_IMAGE");
            }
            buildOption.emplace("-DCOMPUTE_BATCH");
            if (useBatchGemvC4OutputPath) {
                buildOption.emplace("-DOUTPUT_C4NHW4");
                buildOption.emplace("-DOUTPUT_BHW=" + std::to_string(global_y));
            }
            
            int local_size = 64;
            if(mOpenCLBackend->getCLTuneLevel() != None && mOpenCLBackend->getCLTuneLevel() != Fast){
                int min_time = INT_MAX;
                for (int ksize = 16; ksize <= 256; ksize*=2) {
                    auto option = buildOption;
                    option.emplace("-DWGS=" + std::to_string(ksize));
                    auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemv_conv1x1_buf", "gemv_conv_c8_buf", option, mOpenCLBackend->getPrecision());
                    uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
                    std::vector<uint32_t> gws = {static_cast<uint32_t>(ksize), static_cast<uint32_t>(UP_DIV(outChannel, 8)), static_cast<uint32_t>(UP_DIV(global_y, 4))};
                    std::vector<uint32_t> lws = {static_cast<uint32_t>(ksize), 1, 1};
                    uint32_t idx = 0;
                    cl_int ret = CL_SUCCESS;
                    ret |= kernel->get().setArg(idx++, static_cast<int>(gws[0]));
                    ret |= kernel->get().setArg(idx++, static_cast<int>(gws[1]));
                    ret |= kernel->get().setArg(idx++, static_cast<int>(gws[2]));
                    ret |= kernel->get().setArg(idx++, openCLBuffer(mConvGemmInpTensor.get()));
                    if(mResource->mUseImage){
                        ret |= kernel->get().setArg(idx++, *mResource->mKernelImage.get());
                    }else{
                        ret |= kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
                    }
                    ret |= kernel->get().setArg(idx++, *mResource->mDequantScaleOffsetBuffer.get());
                    ret |= kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
                    ret |= kernel->get().setArg(
                        idx++, useBatchGemvC4OutputPath ? openCLBuffer(output)
                                                         : openCLBuffer(mConvGemmOutTensor.get()));
                    ret |= kernel->get().setArg(idx++, static_cast<int>(outputChannelAlign8));
                    ret |= kernel->get().setArg(idx++, static_cast<int>(inputChannelAlign));
                    ret |= kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
                    ret |= kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
                    ret |= kernel->get().setArg(idx++, inputChannels);
                    ret |= kernel->get().setArg(idx++, static_cast<int>(blockNum));
                    ret |= kernel->get().setArg(idx++, static_cast<int>(blockDim));
                    ret |= kernel->get().setArg(idx++, static_cast<float>(mResource->mCoef));
                    MNN_CHECK_CL_SUCCESS(ret, "setArg gemv_conv_c8_buf Kernel Select");
                    std::pair<std::vector<uint32_t>, int> retTune;
                    int cost_time = get2DUseLocalMemTime(gws, lws, mOpenCLBackend->getOpenCLRuntime(), "gemv_conv_c8_buf" + info + "_batch", kernel, "gemv_conv1x1_buf");
                    if(min_time > cost_time) {
                        local_size = ksize;
                        min_time = cost_time;
                    }
                }
            }
            buildOption.emplace("-DWGS=" + std::to_string(local_size));
            mGlobalWorkSize = {static_cast<uint32_t>(local_size), static_cast<uint32_t>(UP_DIV(outChannel, 8)), static_cast<uint32_t>(UP_DIV(global_y, 4))};
            unit.kernel        = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemv_conv1x1_buf", "gemv_conv_c8_buf", buildOption, mOpenCLBackend->getPrecision());
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));
            uint32_t idx = 0;
            cl_int ret = CL_SUCCESS;
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[0]));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[1]));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mGlobalWorkSize[2]));
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvGemmInpTensor.get()));
            if(mResource->mUseImage){
                ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelImage.get());
            }else{
                ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
            }
            ret |= unit.kernel->get().setArg(idx++, *mResource->mDequantScaleOffsetBuffer.get());
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
            ret |= unit.kernel->get().setArg(
                idx++, useBatchGemvC4OutputPath ? openCLBuffer(output)
                                                 : openCLBuffer(mConvGemmOutTensor.get()));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelAlign8));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelAlign));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelBlocks));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelBlocks));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannels));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockNum));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockDim));
            ret |= unit.kernel->get().setArg(idx++, static_cast<float>(mResource->mCoef));
            MNN_CHECK_CL_SUCCESS(ret, "setArg gemv_conv_c8_buf");
            mLocalWorkSize = {static_cast<uint32_t>(local_size), 1, 1};
            mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
            unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
            unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};
        }
        if (useBatchGemvC4OutputPath) {
            return true;
        }
        {
            auto &unit = mUnits[2];
            unit.kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_conv1x1_buf", "gemm_nhwc_to_c4nhw4", buildOption, mOpenCLBackend->getPrecision());
            uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));
            mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(global_y, 4)), static_cast<uint32_t>(UP_DIV(outChannel, 4))};
            uint32_t idx = 0;
            cl_int ret = CL_SUCCESS;
            ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
            ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mConvGemmOutTensor.get()));
            ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(global_y));
            ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelAlign8));
            MNN_CHECK_CL_SUCCESS(ret, "setArg gemm_nhwc_to_c4nhw4");
            mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), "gemm_nhwc_to_c4nhw4", unit.kernel, mOpenCLBackend->getCLTuneLevel(), "gemm_conv1x1_buf").first;
            mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
            unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
            unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
        }
        return true;
    }
    unit.kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("gemm_conv1x1_buf", kernelName, buildOption, mOpenCLBackend->getPrecision());
    if (nullptr == unit.kernel.get()) {
        MNN_ERROR("OpenCLConvBufLowMemory kernel unavailable: %s ic=%d oc=%d rows=%d use_image=%d mode=%s\n",
                  kernelName.c_str(), inputChannels, outChannel, global_y, mResource->mUseImage ? 1 : 0,
                  _compactKernelModeName(resolvedCompactKernelMode));
        return false;
    }
    uint32_t maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));
    const uint32_t fixedWorkGroupY =
        resolvedCompactKernelMode == kCompactKernelPicWG64
            ? 64u
            : (resolvedCompactKernelMode == kCompactKernelPicWG128
                   ? 128u
                   : (resolvedCompactKernelMode == kCompactKernelPicWG4X32
                          ? 32u
                          : (resolvedCompactKernelMode == kCompactKernelPicWG8X16
                                 ? 16u
                                 : (resolvedCompactKernelMode == kCompactKernelPicInputCache32
                                        ? 32u
                                        : (resolvedCompactKernelMode == kCompactKernelPicInputCache64 ? 64u : 0u)))));
    const uint32_t fixedWorkGroupX =
        resolvedCompactKernelMode == kCompactKernelPicWG4X32
            ? 4u
            : (resolvedCompactKernelMode == kCompactKernelPicWG8X16 ? 8u : 1u);
    
    mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(global_y, compactBatchTile)),
                       static_cast<uint32_t>(UP_DIV(outChannel, compactChannelTile))};
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
    if(mResource->mUseImage){
        ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelImage.get());
    }else{
        ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
    }
    ret |= unit.kernel->get().setArg(idx++, *mResource->mDequantScaleOffsetBuffer.get());
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(global_y));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(outputChannelAlign));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(inputChannelAlign));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockNum));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(blockDim));
    ret |= unit.kernel->get().setArg(idx++, mResource->mCoef);
    MNN_CHECK_CL_SUCCESS(ret, "setArg gemm_conv1x1_buf");
    if (fixedWorkGroupY != 0u) {
        const auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
        if (maxWorkGroupSize < fixedWorkGroupX * fixedWorkGroupY ||
            maxWorkItemSizes.size() < 2 ||
            maxWorkItemSizes[0] < fixedWorkGroupX ||
            maxWorkItemSizes[1] < fixedWorkGroupY) {
            MNN_ERROR("OpenCLConvBufLowMemory fixed workgroup unavailable: %s ic=%d oc=%d rows=%d fixed_x=%u fixed_y=%u max_wg=%u max_item_x=%zu max_item_y=%zu\n",
                      kernelName.c_str(), inputChannels, outChannel, global_y, fixedWorkGroupX, fixedWorkGroupY,
                      maxWorkGroupSize,
                      maxWorkItemSizes.size() >= 1 ? maxWorkItemSizes[0] : 0,
                      maxWorkItemSizes.size() >= 2 ? maxWorkItemSizes[1] : 0);
            return false;
        }
        mLocalWorkSize = {fixedWorkGroupX, fixedWorkGroupY};
    } else {
        mLocalWorkSize = localWS2DDefault(mGlobalWorkSize, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName + info, unit.kernel, mOpenCLBackend->getCLTuneLevel(), "gemm_conv1x1_buf").first;
    }
    mOpenCLBackend->recordKernel2d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1]};
    return true;
}
ConvBufLowMemoryExecution::ConvBufLowMemoryExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend)
    : ConvBufCommonExecution(op->main_as_Convolution2D(), backend), CommonExecution(backend, op) {
    if (!mConvComValid) {
        mValid = false;
        return;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvBufLowMemoryExecution init !\n");
#endif
    mOpenCLBackend                 = static_cast<OpenCLBackend *>(backend);
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dParams       = conv2dParams;
    mResource->mConv2dCommonParams  = conv2dCommonParams;
    mResource->mStrides                       = {conv2dCommonParams->strideY(), conv2dCommonParams->strideX()};
    mResource->mDilations                     = {conv2dCommonParams->dilateY(), conv2dCommonParams->dilateX()};
    auto padding = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], conv2dCommonParams);
    mPaddings[0] = padding.second;//padY
    mPaddings[1] = padding.first;//padX

    mResource->mKernelWidth   = conv2dCommonParams->kernelX();
    mResource->mKernelHeight  = conv2dCommonParams->kernelY();
    mResource->mInputChannel = conv2dCommonParams->inputCount();
    mResource->mOutputChannel = conv2dCommonParams->outputCount();
        
    //select opt conv method
    if (mResource->mKernelHeight == mResource->mKernelWidth && mResource->mKernelHeight == 1 && mResource->mStrides[0] == 1 && mResource->mStrides[1] == 1 && conv2dCommonParams->padX() == 0 && conv2dCommonParams->padY() == 0 && conv2dCommonParams->dilateX() == 1 && conv2dCommonParams->dilateY() == 1) {
        set1x1WeightLowMemory();
        mResource->mConv1x1Opt = true;
    }else {
        // set mFilter for not 1x1 case
        setGeneralWeightLowMemory();
    }
    // Create Kernel
    if (conv2dCommonParams->relu()) {
        mResource->mBuildOptions.emplace("-DRELU");
    } else if (conv2dCommonParams->relu6()) {
        mResource->mBuildOptions.emplace("-DRELU6");
    }
    mResource->mBuildOptions.emplace("-DQUANT_BIT=" + std::to_string(mResource->mNumQuantBit));
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvBufLowMemoryExecution init !\n");
#endif
}

ConvBufLowMemoryExecution::ConvBufLowMemoryExecution(std::shared_ptr<ConvBufResource> resource, const MNN::Op* op, Backend *backend)
    : ConvBufCommonExecution(backend), CommonExecution(backend, op) {
    mResource = resource;
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dParams       = conv2dParams;
    mResource->mConv2dCommonParams  = conv2dCommonParams;
}

ConvBufLowMemoryExecution::~ConvBufLowMemoryExecution() {
    // Do nothing
}

bool ConvBufLowMemoryExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new ConvBufLowMemoryExecution(mResource, op, bn);
    return true;
}

ErrorCode ConvBufLowMemoryExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvBufLowMemoryExecution onResize !\n");
#endif
    auto runTime = mOpenCLBackend->getOpenCLRuntime();
    mOpenCLBackend->startRecord(mRecording);
    mUnits.resize(1);
    auto input  = inputs[0];
    auto output = outputs[0];
    auto padding = ConvolutionCommon::convolutionPad(input, output, mResource->mConv2dCommonParams);
    mPaddings[0] = padding.second;//padY
    mPaddings[1] = padding.first;//padX
    // onclone default use conv1x1Opt, need reset
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int batch = outputShape.at(0) * outputShape.at(1) * outputShape.at(2);
    std::vector<int> inputShape = tensorShapeFormat(input);
    const int minChannel = std::min(inputShape.at(3), outputShape.at(3));
    const int maxChannel = std::max(inputShape.at(3), outputShape.at(3));
    const bool usePicCompactKernel = usePicCompactGemmLowMemory(input, output);
    int compactKernelMode = usePicCompactKernel ? static_cast<int>(kCompactKernelPic)
                                                : static_cast<int>(kCompactKernelGeneric);
    const char* compactDecisionSource = "default";
    bool compactDecisionFallback = false;
    mUseFPWeight = false;
    if (mResource->mConv1x1Opt) {
        if(batch == 1){
            tuneGemvLowMemory(input, output);
        } else {
            if (batch <= 16) {
                bool familyResolved = false;
                uint32_t forcedFamily = 0;
                const bool hasForcedFamily = _getCompactDenseFamilyOverride(&forcedFamily);
                if (hasForcedFamily) {
                    compactDecisionSource = "env_force_family";
                    mUseFPWeight = forcedFamily == kCompactDenseFamilyFPWeight;
                    if (!mUseFPWeight) {
                        compactKernelMode = _compactKernelModeForFamily(forcedFamily);
                    }
                    familyResolved = true;
                }
                if (!familyResolved && _isMaliDecodeTinyDirectC4GemvShape(
                                           runTime, mResource->mNumQuantBit, batch,
                                           mResource->mInputChannel, mResource->mOutputChannel)) {
                    mUseFPWeight = false;
                    compactKernelMode = static_cast<int>(kCompactKernelAdrenoDirectC4Gemv);
                    compactDecisionSource = "mali_decode_tiny_direct_c4_gemv";
                    familyResolved = true;
                }
                const bool adrenoTinyFamilyShape = _isAdrenoTinyDenseFamilyShape(
                    runTime, mResource->mNumQuantBit, batch,
                    mResource->mInputChannel, mResource->mOutputChannel);
                if (!familyResolved && adrenoTinyFamilyShape) {
                    const std::string familyInfo = _adrenoTinyDenseFamilyTuneKey(
                        mResource->mNumQuantBit, batch, mResource->mInputChannel, mResource->mOutputChannel);
                    const std::vector<uint32_t> familyShape = {
                        static_cast<uint32_t>(batch),
                        static_cast<uint32_t>(mResource->mOutputChannel),
                        static_cast<uint32_t>(mResource->mInputChannel),
                    };
                    std::pair<std::vector<uint32_t>, uint32_t> tuneInfo;
                    if (_getExactTunedInfo(familyInfo, familyShape, tuneInfo, runTime) &&
                        !tuneInfo.first.empty()) {
                        const uint32_t family = tuneInfo.first[0];
                        mUseFPWeight = false;
                        compactKernelMode = _compactKernelModeForFamily(family);
                        compactDecisionSource = "adreno_tiny_family_cache";
                        familyResolved = true;
                        if (_convProfileSelectionEnabled()) {
                            MNN_PRINT("OpenCLConvBufLowMemory tiny-family-cache batch=%d ic=%d oc=%d "
                                      "family=%s time=%u exact=1\n",
                                      batch, mResource->mInputChannel, mResource->mOutputChannel,
                                      _compactDenseFamilyName(family), tuneInfo.second);
                        }
                    } else if (mOpenCLBackend->getCLTuneLevel() == Heavy ||
                               mOpenCLBackend->getCLTuneLevel() == Wide) {
                        bool tuned = false;
                        int bestTime = std::numeric_limits<int>::max();
                        uint32_t bestFamily = kCompactDenseFamilyPicQuantC4;
                        setRecordClose closeRecord(mOpenCLBackend);
                        for (uint32_t family : _adrenoTinyDenseFamilyCandidates(batch, mResource->mOutputChannel)) {
                            mUseFPWeight = false;
                            const int mode = _compactKernelModeForFamily(family);
                            if (!tuneGemmLowMemory(input, output, mode)) {
                                if (_convProfileSelectionEnabled()) {
                                    MNN_PRINT("OpenCLConvBufLowMemory tiny-family-candidate unavailable "
                                              "batch=%d ic=%d oc=%d family=%s\n",
                                              batch, mResource->mInputChannel, mResource->mOutputChannel,
                                              _compactDenseFamilyName(family));
                                }
                                continue;
                            }
                            const int candidateTime = getExecuteTime();
                            if (_convProfileSelectionEnabled()) {
                                MNN_PRINT("OpenCLConvBufLowMemory tiny-family-candidate batch=%d ic=%d oc=%d "
                                          "family=%s time=%d\n",
                                          batch, mResource->mInputChannel, mResource->mOutputChannel,
                                          _compactDenseFamilyName(family), candidateTime);
                            }
                            if (!tuned || candidateTime < bestTime) {
                                tuned = true;
                                bestTime = candidateTime;
                                bestFamily = family;
                            }
                        }
                        if (tuned) {
                            mUseFPWeight = false;
                            compactKernelMode = _compactKernelModeForFamily(bestFamily);
                            std::pair<std::vector<uint32_t>, uint32_t> bestInfo =
                                std::make_pair(std::vector<uint32_t>{bestFamily},
                                               static_cast<uint32_t>(std::max(0, bestTime)));
                            setTunedInfo(familyInfo, familyShape, bestInfo, runTime, "gemm_conv1x1_buf");
                            compactDecisionSource = "adreno_tiny_family_online_tuned";
                            familyResolved = true;
                        }
                    }
                }
                if (!familyResolved && adrenoTinyFamilyShape) {
                    uint32_t heuristicFamily = kCompactDenseFamilyGenericQuant;
                    if (_pickAdrenoTinyDenseFamilyHeuristic(
                            runTime, mResource->mNumQuantBit, batch,
                            mResource->mInputChannel, mResource->mOutputChannel,
                            &heuristicFamily)) {
                        mUseFPWeight = false;
                        compactKernelMode = _compactKernelModeForFamily(heuristicFamily);
                        compactDecisionSource = "adreno_tiny_family_row_heuristic";
                        familyResolved = true;
                    }
                }
            }
            if(batch > 16){
                const bool tuneAdrenoCompactFamily = ConvAdreno::shouldTuneCompactDenseFamily(
                    runTime, mResource->mNumQuantBit, batch,
                    mResource->mOutputChannel, mResource->mInputChannel);
                const bool compactTinyArithmeticRows =
                    usePicCompactKernel &&
                    mResource->mNumQuantBit == 4 &&
                    1LL * batch * batch <= 5LL * maxChannel;
                const bool compactLargeChannelSpillRows =
                    mResource->mNumQuantBit == 4 &&
                    batch > 512 && batch <= 576 &&
                    mResource->mInputChannel >= 1024 &&
                    mResource->mOutputChannel >= 1024;
                const bool compactMidChannelRetuneRows =
                    usePicCompactKernel &&
                    mResource->mNumQuantBit == 4 &&
                    batch >= 96 && batch < 192 &&
                    mResource->mInputChannel >= 1024 &&
                    mResource->mOutputChannel >= 1024;
                bool familyResolved = false;
                uint32_t forcedFamily = 0;
                const bool hasForcedFamily = _getCompactDenseFamilyOverride(&forcedFamily);
                if (hasForcedFamily) {
                    compactDecisionSource = "env_force_family";
                    mUseFPWeight = forcedFamily == kCompactDenseFamilyFPWeight;
                    if (!mUseFPWeight) {
                        compactKernelMode = _compactKernelModeForFamily(forcedFamily);
                    }
                    familyResolved = true;
                }
                if (!familyResolved && tuneAdrenoCompactFamily) {
                    const std::string familyInfo = ConvAdreno::compactDenseFamilyTuneKey(
                        mResource->mInputChannel, mResource->mOutputChannel, mResource->mNumQuantBit, batch);
                    const std::vector<uint32_t> familyShape = {
                        static_cast<uint32_t>(batch),
                        static_cast<uint32_t>(mResource->mOutputChannel),
                        static_cast<uint32_t>(mResource->mInputChannel),
                    };
                    std::pair<std::vector<uint32_t>, uint32_t> tuneInfo;
                    if (_getExactTunedInfo(familyInfo, familyShape, tuneInfo, runTime) &&
                        !tuneInfo.first.empty()) {
                        const uint32_t family = tuneInfo.first[0];
                        compactDecisionSource = "adreno_family_cache";
                        if (_convProfileSelectionEnabled()) {
                            MNN_PRINT("OpenCLConvBufLowMemory family-cache batch=%d ic=%d oc=%d family=%s "
                                      "time=%u exact=1\n",
                                      batch, mResource->mInputChannel, mResource->mOutputChannel,
                                      _compactDenseFamilyName(family),
                                      tuneInfo.second);
                        }
                        if (family == kCompactDenseFamilyFPWeight) {
                            mUseFPWeight = true;
                        } else {
                            compactKernelMode = _compactKernelModeForFamily(family);
                        }
                        familyResolved = true;
                    } else if ((mOpenCLBackend->getCLTuneLevel() == Heavy ||
                                mOpenCLBackend->getCLTuneLevel() == Wide)) {
                        bool tuned = false;
                        int bestTime = std::numeric_limits<int>::max();
                        uint32_t bestFamily = kCompactDenseFamilyPicQuant;
                        setRecordClose closeRecord(mOpenCLBackend);
                        for (uint32_t family : {kCompactDenseFamilyGenericQuant,
                                                kCompactDenseFamilyPicQuant,
                                                kCompactDenseFamilyPicQuantWG64,
                                                kCompactDenseFamilyPicQuantWG128,
                                                kCompactDenseFamilyPicQuantWG4X32,
                                                kCompactDenseFamilyPicQuantWG8X16,
                                                kCompactDenseFamilyPicQuantInputCache32,
                                                kCompactDenseFamilyPicQuantInputCache64,
                                                kCompactDenseFamilyPicQuantC4,
                                                kCompactDenseFamilyPicQuantB2,
                                                kCompactDenseFamilyAdrenoBatchGemv,
                                                kCompactDenseFamilyFPWeight}) {
                            mUseFPWeight = false;
                            if (family == kCompactDenseFamilyFPWeight) {
                                mUseFPWeight = true;
                                useFPWeightGemmLowMemory(input, output);
                            } else {
                                const int mode = _compactKernelModeForFamily(family);
                                if (!tuneGemmLowMemory(input, output, mode)) {
                                    if (_convProfileSelectionEnabled()) {
                                        MNN_PRINT("OpenCLConvBufLowMemory family-candidate unavailable batch=%d ic=%d oc=%d family=%s\n",
                                                  batch, mResource->mInputChannel, mResource->mOutputChannel,
                                                  _compactDenseFamilyName(family));
                                    }
                                    continue;
                                }
                            }
                            const int candidateTime = getExecuteTime();
                            if (_convProfileSelectionEnabled()) {
                                MNN_PRINT("OpenCLConvBufLowMemory family-candidate batch=%d ic=%d oc=%d family=%s "
                                          "time=%d\n",
                                          batch, mResource->mInputChannel, mResource->mOutputChannel,
                                          _compactDenseFamilyName(family),
                                          candidateTime);
                            }
                            if (!tuned || candidateTime < bestTime) {
                                tuned = true;
                                bestTime = candidateTime;
                                bestFamily = family;
                            }
                        }
                        if (tuned) {
                            mUseFPWeight = bestFamily == kCompactDenseFamilyFPWeight;
                            if (!mUseFPWeight) {
                                compactKernelMode = _compactKernelModeForFamily(bestFamily);
                            }
                            std::pair<std::vector<uint32_t>, uint32_t> bestInfo =
                                std::make_pair(std::vector<uint32_t>{bestFamily},
                                               static_cast<uint32_t>(std::max(0, bestTime)));
                            setTunedInfo(familyInfo, familyShape, bestInfo, runTime, "gemm_conv1x1_buf");
                            familyResolved = true;
                            compactDecisionSource = "adreno_family_online_tuned";
                        }
                    }
                }
                if (!familyResolved) {
                    std::pair<std::vector<uint32_t>, uint32_t> tuneInfo;
                    std::string info = "convBufLowMemory_" + std::to_string(mResource->mInputChannel) + "_" + std::to_string(mResource->mOutputChannel);
                    if (compactTinyArithmeticRows) {
                        // Very small compact-row PIC shapes, e.g. 115x1536->4608
                        // or 115x3072->8192, are dominated by FP-weight
                        // transpose/Strassen overhead. Use an M^2-vs-channel-width
                        // heuristic instead of a fixed-row cutoff so 115-row
                        // shapes stay on the generic quant path while mid-row
                        // shapes such as batch~=216 can still retune normally.
                        info += "_picsmall_v1";
                    } else if (compactLargeChannelSpillRows) {
                        // Give the 512<M<=576 PIC compact-row band its own tune key.
                        // Earlier OrangePi experiments clamped this band to the
                        // generic quant path, so reusing the old namespace would keep
                        // replaying that stale decision instead of re-benchmarking the
                        // current quant-vs-fp-weight choice.
                        info += "_picspill_v2";
                    } else if (compactMidChannelRetuneRows) {
                        // Mid compact-row PIC shapes such as batch~=115 have a very
                        // different arithmetic/transpose balance from the old
                        // generic low-batch bucket. Keep a dedicated tune namespace
                        // so heavy tune can re-compare generic quant vs fp-weight
                        // instead of replaying stale decisions from older kernels.
                        info += "_picmid_v1";
                    }
                    if(_getExactTunedInfo(info, {static_cast<unsigned int>(batch)}, tuneInfo, runTime)){
                        mUseFPWeight = tuneInfo.first[0];
                        compactDecisionSource = "legacy_binary_cache";
                    } else{
                        if((mOpenCLBackend->getCLTuneLevel() == Heavy || mOpenCLBackend->getCLTuneLevel() == Wide)){
                            if (!compactTinyArithmeticRows) {
                                setRecordClose closeRecord(mOpenCLBackend);
                                const bool shortBatchAvailable = tuneGemmLowMemory(input, output, compactKernelMode);
                                auto shortBatchTime = shortBatchAvailable ? getExecuteTime()
                                                                          : std::numeric_limits<int>::max();
                                mUseFPWeight = true;
                                useFPWeightGemmLowMemory(input, output);
                                auto longBatchTime = getExecuteTime();
                                mUseFPWeight = false;
                                if(!shortBatchAvailable || longBatchTime < shortBatchTime){
                                    mUseFPWeight = true;
                                }
                            }
                            std::pair<std::vector<uint32_t>, uint32_t> tuneInfoTmp = std::make_pair<std::vector<uint32_t>, uint32_t>({mUseFPWeight}, 0);
                            setTunedInfo(info, {static_cast<unsigned int>(batch)}, tuneInfoTmp, runTime, "gemm_conv1x1_buf");
                            compactDecisionSource = "legacy_binary_online_tuned";
                        } else{
                            // OrangePi headDim=128 PIC requests commonly land just
                            // above the 512-row compact GEMM cap, e.g. about 519
                            // active rows at ctx1024/cacheblend50. When heavy tune
                            // is unavailable, keep the generic quant GEMM as the
                            // conservative default in this narrow spill band;
                            // larger M keeps the existing FP-weight bias.
                            if(batch > 512 && !compactLargeChannelSpillRows){
                                mUseFPWeight = true;
                                compactDecisionSource = "legacy_heuristic_fp_weight";
                            } else {
                                compactDecisionSource = "legacy_default";
                            }
                        }
                    }
                }
            }
            if(mUseFPWeight){
                useFPWeightGemmLowMemory(input, output);
            }else{
                if (!tuneGemmLowMemory(input, output, compactKernelMode)) {
                    mUseFPWeight = true;
                    useFPWeightGemmLowMemory(input, output);
                    compactDecisionFallback = true;
                }
            }
        }
    } else {
        tuneGeneralCaseLowMemory(input, output);
    }
    const bool profileCompactRows =
        batch >= 96 && mResource->mInputChannel >= 1024 && mResource->mOutputChannel >= 1024;
    const bool profileTinyRows =
        batch > 1 && batch <= 16 && mResource->mInputChannel >= 1024 && mResource->mOutputChannel >= 256;
    if (_convProfileSelectionEnabled() && mResource->mConv1x1Opt && (profileCompactRows || profileTinyRows)) {
        const uint32_t family = _compactDenseFamilyForSelection(mUseFPWeight, compactKernelMode);
        MNN_PRINT("OpenCLConvBufLowMemory profile batch=%d ic=%d oc=%d int4=%d use_fp_weight=%d "
                  "weight_image=%d selected_pic_kernel=%d compact_mode=%s auto_pic_compact=%d family=%s "
                  "decision_source=%s execute_fallback=%d gws=%u,%u lws=%u,%u tune_level=%d\n",
                  batch, mResource->mInputChannel, mResource->mOutputChannel,
                  mResource->mNumQuantBit == 4 ? 1 : 0, mUseFPWeight ? 1 : 0,
                  mResource->mUseImage ? 1 : 0,
                  (!mUseFPWeight && compactKernelMode != kCompactKernelGeneric) ? 1 : 0,
                  _compactKernelModeName(static_cast<uint32_t>(compactKernelMode)),
                  usePicCompactKernel ? 1 : 0,
                  _compactDenseFamilyName(family),
                  compactDecisionSource,
                  compactDecisionFallback ? 1 : 0,
                  mGlobalWorkSize.size() > 0 ? static_cast<unsigned>(mGlobalWorkSize[0]) : 0u,
                  mGlobalWorkSize.size() > 1 ? static_cast<unsigned>(mGlobalWorkSize[1]) : 0u,
                  mLocalWorkSize.size() > 0 ? static_cast<unsigned>(mLocalWorkSize[0]) : 0u,
                  mLocalWorkSize.size() > 1 ? static_cast<unsigned>(mLocalWorkSize[1]) : 0u,
                  static_cast<int>(mOpenCLBackend->getCLTuneLevel()));
    }
    for (auto &unit : mUnits) {
        bool lws_null = true;
        for (size_t i = 0; i < unit.globalWorkSize.dimensions(); ++i) {
            unit.globalWorkSize.get()[i] = ROUND_UP(unit.globalWorkSize.get()[i], std::max((size_t)1, unit.localWorkSize.get()[i]));
            if(unit.localWorkSize.get()[i] != 0) {
                lws_null = false;
            }
        }
        if(lws_null){
            unit.localWorkSize = cl::NullRange;
        }
    }
    mOpenCLBackend->endRecord(mRecording);
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvBufLowMemoryExecution onResize !\n");
#endif
    return NO_ERROR;
}

int ConvBufLowMemoryExecution::getExecuteTime(){
    for (auto &unit : mUnits) {
        bool lws_null = true;
        for (size_t i = 0; i < unit.globalWorkSize.dimensions(); ++i) {
            unit.globalWorkSize.get()[i] = ROUND_UP(unit.globalWorkSize.get()[i], std::max((size_t)1, unit.localWorkSize.get()[i]));
            if(unit.localWorkSize.get()[i] != 0) {
                lws_null = false;
            }
        }
        if(lws_null){
            unit.localWorkSize = cl::NullRange;
        }
    }
    int executeTime = 0;
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto res = CL_SUCCESS;
    if(mUseFPWeight){
        // arrange input and weight
        int i = 0;
        for (; i < 2; ++i){
            auto unit = mUnits[i];
            cl::Event event;
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                   cl::NullRange,
                                                   unit.globalWorkSize,
                                                   unit.localWorkSize,
                                                   nullptr,
                                                   &event);
            executeTime += runtime->getEventTime(event);
        }
        // call gemm execute
        executeTime += mStrassenComputor->getExecuteTime();
        
        // rearrange output
        for (; i < mUnits.size(); ++i){
            auto unit = mUnits[i];
            cl::Event event;
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                   cl::NullRange,
                                                   unit.globalWorkSize,
                                                   unit.localWorkSize,
                                                   nullptr,
                                                   &event);
            executeTime += runtime->getEventTime(event);
        }
    }else{
        for (auto &unit : mUnits) {
            cl::Event event;
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                               cl::NullRange,
                                                               unit.globalWorkSize,
                                                               unit.localWorkSize,
                                                               nullptr,
                                                               &event);
            executeTime += runtime->getEventTime(event);
        }
    }
    return executeTime;
}

ErrorCode ConvBufLowMemoryExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvBufLowMemoryExecution onExecute !\n");
#endif
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const bool requestProfile = _convRequestProfileEnabled();
    const bool requestProfileSync = requestProfile && _convRequestProfileSync();
    const uint64_t requestProfileStartUs = requestProfile ? _convProfileNowUs() : 0;
    if (requestProfile) {
        std::vector<int> outputShape = tensorShapeFormat(outputs[0]);
        std::vector<int> inputShape = tensorShapeFormat(inputs[0]);
        const int batch = outputShape.at(0) * outputShape.at(1) * outputShape.at(2);
        MNN_PRINT("MNN_PIC_REQUEST_PROFILE stage=conv_lowmem_begin cost_ms=0.000 "
                  "batch=%d ic=%d oc=%d input=%d,%d,%d,%d output=%d,%d,%d,%d use_fp_weight=%d units=%zu conv1x1=%d\n",
                  batch,
                  inputShape.at(3),
                  outputShape.at(3),
                  inputShape.at(0), inputShape.at(1), inputShape.at(2), inputShape.at(3),
                  outputShape.at(0), outputShape.at(1), outputShape.at(2), outputShape.at(3),
                  mUseFPWeight ? 1 : 0,
                  mUnits.size(),
                  mResource->mConv1x1Opt ? 1 : 0);
    }
#ifdef ENABLE_OPENCL_TIME_PROFILER
    int idx = 0;
#else
    if(mOpenCLBackend->isUseRecordQueue()){
        mOpenCLBackend->addRecord(mRecording, mOpRecordUpdateInfo);
        return NO_ERROR;
    }
#endif
    auto res = CL_SUCCESS;
    auto logUnitProfile = [&](int unitIndex, const std::shared_ptr<KernelWrap>& kernel,
                              const cl::NDRange& gws, const cl::NDRange& lws, uint64_t unitStartUs) {
        runtime->commandQueue().finish();
        MNN_PRINT("MNN_PIC_REQUEST_PROFILE stage=conv_lowmem_unit cost_ms=%.3f "
                  "unit=%d kernel=%s gws=%zu,%zu,%zu lws=%zu,%zu,%zu use_fp_weight=%d\n",
                  (_convProfileNowUs() - unitStartUs) / 1000.0,
                  unitIndex,
                  _convKernelFunctionName(kernel).c_str(),
                  _convRangeDim(gws, 0), _convRangeDim(gws, 1), _convRangeDim(gws, 2),
                  _convRangeDim(lws, 0), _convRangeDim(lws, 1), _convRangeDim(lws, 2),
                  mUseFPWeight ? 1 : 0);
    };
    if(mUseFPWeight){
        // arrange input and weight
        int i = 0;
        for (; i < 2; ++i){
            auto unit = mUnits[i];
            const uint64_t unitStartUs = requestProfileSync ? _convProfileNowUs() : 0;
            #ifdef ENABLE_OPENCL_TIME_PROFILER
            cl::Event event;
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                   cl::NullRange,
                                                   unit.globalWorkSize,
                                                   unit.localWorkSize,
                                                   nullptr,
                                                   &event);
            runtime->pushEvent({EnumNameOpType(mOpType) + std::to_string(idx++), event});
            #else
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                   cl::NullRange,
                                                   unit.globalWorkSize,
                                                   unit.localWorkSize);
            #endif
            MNN_CHECK_CL_SUCCESS(res, EnumNameOpType(mOp->type()));
            if (requestProfileSync) {
                logUnitProfile(i, unit.kernel, unit.globalWorkSize, unit.localWorkSize, unitStartUs);
            }
        }
        // call gemm execute
        const uint64_t gemmStartUs = requestProfileSync ? _convProfileNowUs() : 0;
        mStrassenComputor->onExecute();
        if (requestProfileSync) {
            runtime->commandQueue().finish();
            MNN_PRINT("MNN_PIC_REQUEST_PROFILE stage=conv_lowmem_unit cost_ms=%.3f "
                      "unit=%d kernel=%s gws=0,0,0 lws=0,0,0 use_fp_weight=%d\n",
                      (_convProfileNowUs() - gemmStartUs) / 1000.0,
                      i,
                      "strassen_gemm",
                      mUseFPWeight ? 1 : 0);
        }
        ++i;
        
        // rearrange output
        for (; i < mUnits.size(); ++i){
            auto unit = mUnits[i];
            const uint64_t unitStartUs = requestProfileSync ? _convProfileNowUs() : 0;
            #ifdef ENABLE_OPENCL_TIME_PROFILER
            cl::Event event;
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                   cl::NullRange,
                                                   unit.globalWorkSize,
                                                   unit.localWorkSize,
                                                   nullptr,
                                                   &event);
            runtime->pushEvent({EnumNameOpType(mOpType) + std::to_string(idx++), event});
            #else
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                   cl::NullRange,
                                                   unit.globalWorkSize,
                                                   unit.localWorkSize);
            #endif
            MNN_CHECK_CL_SUCCESS(res, EnumNameOpType(mOp->type()));
            if (requestProfileSync) {
                logUnitProfile(i, unit.kernel, unit.globalWorkSize, unit.localWorkSize, unitStartUs);
            }
        }
    }else{
        int unitIndex = 0;
        for (auto &unit : mUnits) {
            const uint64_t unitStartUs = requestProfileSync ? _convProfileNowUs() : 0;
            #ifdef ENABLE_OPENCL_TIME_PROFILER
            cl::Event event;
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                               cl::NullRange,
                                                               unit.globalWorkSize,
                                                               unit.localWorkSize,
                                                               nullptr,
                                                               &event);
            runtime->pushEvent({EnumNameOpType(mOpType) + std::to_string(idx++), event});
            #else
            res = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel->get(),
                                                               cl::NullRange,
                                                               unit.globalWorkSize,
                                                               unit.localWorkSize);
            #endif
            MNN_CHECK_CL_SUCCESS(res, EnumNameOpType(mOp->type()));
            if (requestProfileSync) {
                logUnitProfile(unitIndex, unit.kernel, unit.globalWorkSize, unit.localWorkSize, unitStartUs);
            }
            ++unitIndex;
        }
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvBufLowMemoryExecution onExecute !\n");
#endif
    if (requestProfileSync) {
        runtime->commandQueue().finish();
    }
    if (requestProfile) {
        std::vector<int> outputShape = tensorShapeFormat(outputs[0]);
        std::vector<int> inputShape = tensorShapeFormat(inputs[0]);
        const int batch = outputShape.at(0) * outputShape.at(1) * outputShape.at(2);
        MNN_PRINT("MNN_PIC_REQUEST_PROFILE stage=conv_lowmem_end cost_ms=%.3f "
                  "batch=%d ic=%d oc=%d use_fp_weight=%d sync=%d\n",
                  (_convProfileNowUs() - requestProfileStartUs) / 1000.0,
                  batch,
                  inputShape.at(3),
                  outputShape.at(3),
                  mUseFPWeight ? 1 : 0,
                  requestProfileSync ? 1 : 0);
    }
    return NO_ERROR;
}

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
#endif /* MNN_LOW_MEMORY */
