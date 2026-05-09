#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "backend/opencl/execution/buffer/AttentionBufExecution.hpp"
#include "core/MNNFileUtils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <vector>

namespace MNN {
namespace OpenCL {

namespace {

constexpr int kCpuFlashAttentionBlockSize = 64;

static inline int roundUpInt(int value, int unit) {
    return ((value + unit - 1) / unit) * unit;
}

struct DiskPackedKVLayout {
    int hP = 4;
    int lP = 1;
    int bytes = 4;
};

static bool readBinaryFile(const std::string& path, std::vector<uint8_t>& data) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        MNN_ERROR("[Error]: failed to open OpenCL prefix cache file: %s\n", path.c_str());
        return false;
    }
    input.seekg(0, std::ios::end);
    auto size = input.tellg();
    if (size <= 0) {
        MNN_ERROR("[Error]: empty OpenCL prefix cache file: %s\n", path.c_str());
        return false;
    }
    input.seekg(0, std::ios::beg);
    data.resize(static_cast<size_t>(size));
    input.read(reinterpret_cast<char*>(data.data()), size);
    if (!input) {
        MNN_ERROR("[Error]: failed to read OpenCL prefix cache file: %s\n", path.c_str());
        return false;
    }
    return true;
}

static bool validPackedFileSizes(size_t keySize, size_t valueSize, int kvHeads, int headDim,
                                 int hP, int lP, int bytes, int tokenCount) {
    if (kvHeads <= 0 || headDim <= 0 || bytes <= 0 || tokenCount <= 0) {
        return false;
    }
    auto keyUnit = static_cast<size_t>(kvHeads) * roundUpInt(headDim, lP) * bytes;
    auto valueUnit = static_cast<size_t>(kvHeads) * roundUpInt(headDim, hP) *
                     roundUpInt(kCpuFlashAttentionBlockSize, lP) * bytes;
    if (keyUnit == 0 || valueUnit == 0 || keySize % keyUnit != 0 || valueSize % valueUnit != 0) {
        return false;
    }
    size_t keyCapacity = keySize / keyUnit;
    size_t valueCapacity = (valueSize / valueUnit) * kCpuFlashAttentionBlockSize;
    return static_cast<size_t>(tokenCount) <= keyCapacity && static_cast<size_t>(tokenCount) <= valueCapacity;
}

static bool resolveDiskPackedKVLayout(size_t keySize, size_t valueSize, int kvHeads, int headDim,
                                      int tokenCount, DiskPackedKVLayout& layout) {
    const DiskPackedKVLayout candidates[] = {
        {4, 1, 4},
        {4, 1, 2},
        {8, 1, 4},
        {8, 1, 2},
        {4, 8, 2},
        {4, 8, 4},
    };
    for (const auto& candidate : candidates) {
        if (validPackedFileSizes(keySize, valueSize, kvHeads, headDim,
                                 candidate.hP, candidate.lP, candidate.bytes, tokenCount)) {
            layout = candidate;
            return true;
        }
    }
    return false;
}

} // namespace

class OpenCLPrefixAttentionBufExecution : public AttentionBufExecution {
public:
    OpenCLPrefixAttentionBufExecution(const MNN::Op* op, Backend* backend, bool kvCache, int layerIndex)
        : AttentionBufExecution(op, backend, kvCache), mLayerIndex(layerIndex) {
    }

    OpenCLPrefixAttentionBufExecution(std::shared_ptr<KVCacheCLManager> manager, const MNN::Op* op,
                                      Backend* backend, int layerIndex)
        : AttentionBufExecution(manager, op, backend), mLayerIndex(layerIndex) {
    }

protected:
    virtual int onResizePrefixKVLength(const std::vector<Tensor*>& inputs, int seqlen) override {
        (void)inputs;
        (void)seqlen;
        if (mMeta == nullptr || mMeta->file_flag != KVMeta::PendingReadSegments ||
            mMeta->prefix_segments.empty() || mMeta->previous != mMeta->remove) {
            return 0;
        }
        int segmentTotalTokens = mMeta->segment_total_tokens;
        if (segmentTotalTokens <= 0) {
            for (const auto& segment : mMeta->prefix_segments) {
                segmentTotalTokens += segment.token_count;
            }
        }
        return segmentTotalTokens;
    }

    virtual ErrorCode onPrepareKVCacheBeforeAppend(const std::vector<Tensor*>& inputs, bool& prepared,
                                                   int& appendKvSeqLen) override {
        prepared = false;
        appendKvSeqLen = inputs[1]->shape()[1];
        if (mMeta == nullptr || mMeta->file_flag != KVMeta::PendingReadSegments ||
            mMeta->prefix_segments.empty()) {
            return NO_ERROR;
        }

        auto query = inputs[0];
        auto key = inputs[1];
        int batch = query->shape()[0];
        int promptLen = key->shape()[1];
        int kvHeads = key->shape()[2];
        int headDim = key->shape()[3];
        if (batch != 1) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention direct_segments only supports batch=1 now\n");
            return NOT_SUPPORT;
        }
        if (mMeta->previous != mMeta->remove) {
            return NO_ERROR;
        }
        if (mMeta->prefix_cache_dir.empty()) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention missing prefix_cache_dir in KVMeta\n");
            return INVALID_VALUE;
        }

        int segmentTotalTokens = mMeta->segment_total_tokens;
        if (segmentTotalTokens <= 0) {
            for (const auto& segment : mMeta->prefix_segments) {
                segmentTotalTokens += segment.token_count;
            }
            mMeta->segment_total_tokens = segmentTotalTokens;
        }
        if (segmentTotalTokens <= 0) {
            return INVALID_VALUE;
        }

        int requiredTotal = segmentTotalTokens + promptLen;
        if (!mKVCacheCLManager->ensureCapacity(requiredTotal)) {
            return OUT_OF_MEMORY;
        }
        int runtimeBytes = mKVCacheCLManager->byte();
        if (runtimeBytes != 2 && runtimeBytes != 4) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention only supports fp16/fp32 runtime KV cache\n");
            return NOT_SUPPORT;
        }

        int layerIndex = mLayerIndex;
        if (layerIndex < 0) {
            layerIndex = mMeta->layer_index;
            if (mMeta->layer_nums > 0) {
                mMeta->layer_index = (mMeta->layer_index + 1) % mMeta->layer_nums;
            } else {
                mMeta->layer_index++;
            }
        }

        auto runtime = mOpenCLBackend->getOpenCLRuntime();
        auto prefixKernel = runtime->buildKernel("attention_buf", "prefix_copy_rope", {}, mOpenCLBackend->getPrecision());
        OPENCL_CHECK_KERNEL(prefixKernel);
        int maxLen = ROUND_UP(mKVCacheCLManager->maxLength(), 4);

        int dstStart = 0;
        for (const auto& segment : mMeta->prefix_segments) {
            if (segment.token_count <= 0) {
                continue;
            }
            if (segment.key_rope_state != KVMeta::KeyRopeCanonicalNoRope ||
                segment.rope_pairing != KVMeta::RopePairingHalf) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention requires canonical_no_rope half-split prefix cache: %s\n",
                          segment.cache_name.c_str());
                return NOT_SUPPORT;
            }

            auto keyPath = MNNFilePathConcat(mMeta->prefix_cache_dir, segment.cache_name) + "_" +
                           std::to_string(layerIndex) + ".k";
            auto valuePath = MNNFilePathConcat(mMeta->prefix_cache_dir, segment.cache_name) + "_" +
                             std::to_string(layerIndex) + ".v";
            std::vector<uint8_t> keyFile;
            std::vector<uint8_t> valueFile;
            if (!readBinaryFile(keyPath, keyFile) || !readBinaryFile(valuePath, valueFile)) {
                return FILE_OPEN_FAILED;
            }

            DiskPackedKVLayout diskLayout;
            if (!resolveDiskPackedKVLayout(keyFile.size(), valueFile.size(), kvHeads, headDim,
                                           segment.token_count, diskLayout)) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention unsupported floating prefix KV file layout: %s / %s\n",
                          keyPath.c_str(), valuePath.c_str());
                return INVALID_VALUE;
            }

            size_t keySizePerHead = keyFile.size() / kvHeads;
            size_t valueSizePerHead = valueFile.size() / kvHeads;
            if (keySizePerHead > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                valueSizePerHead > static_cast<size_t>(std::numeric_limits<int>::max())) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention prefix KV file is too large for kernel int args\n");
                return INVALID_VALUE;
            }
            int ropeDim = segment.rope_dim > 0 ? segment.rope_dim : headDim;
            ropeDim = std::min(ropeDim, headDim);
            if (ropeDim <= 0 || (ropeDim % 2) != 0 || segment.rope_theta <= 0.0f) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention invalid RoPE metadata for cache %s\n",
                          segment.cache_name.c_str());
                return INVALID_VALUE;
            }

            cl_int res = CL_SUCCESS;
            cl::Buffer keyBuffer(runtime->context(), CL_MEM_READ_ONLY, keyFile.size(), nullptr, &res);
            if (res != CL_SUCCESS) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention failed to allocate key staging buffer\n");
                return OUT_OF_MEMORY;
            }
            cl::Buffer valueBuffer(runtime->context(), CL_MEM_READ_ONLY, valueFile.size(), nullptr, &res);
            if (res != CL_SUCCESS) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention failed to allocate value staging buffer\n");
                return OUT_OF_MEMORY;
            }
            res = runtime->commandQueue().enqueueWriteBuffer(keyBuffer, CL_TRUE, 0, keyFile.size(), keyFile.data());
            res |= runtime->commandQueue().enqueueWriteBuffer(valueBuffer, CL_TRUE, 0, valueFile.size(), valueFile.data());
            if (res != CL_SUCCESS) {
                MNN_ERROR("[Error]: OpenCL PrefixAttention failed to upload prefix KV staging buffers\n");
                return INVALID_VALUE;
            }

            std::vector<uint32_t> validGws = {
                static_cast<uint32_t>(headDim),
                static_cast<uint32_t>(segment.token_count),
                static_cast<uint32_t>(kvHeads)
            };
            std::vector<uint32_t> lws = {16, 4, 1};
            std::vector<uint32_t> runGws = {
                ROUND_UP(validGws[0], lws[0]),
                ROUND_UP(validGws[1], lws[1]),
                ROUND_UP(validGws[2], lws[2])
            };

            uint32_t index = 0;
            res = CL_SUCCESS;
            res |= prefixKernel->get().setArg(index++, validGws[0]);
            res |= prefixKernel->get().setArg(index++, validGws[1]);
            res |= prefixKernel->get().setArg(index++, validGws[2]);
            res |= prefixKernel->get().setArg(index++, keyBuffer);
            res |= prefixKernel->get().setArg(index++, valueBuffer);
            res |= prefixKernel->get().setArg(index++, *mKVCacheCLManager->mutableKey());
            res |= prefixKernel->get().setArg(index++, *mKVCacheCLManager->mutableValue());
            res |= prefixKernel->get().setArg(index++, segment.token_count);
            res |= prefixKernel->get().setArg(index++, dstStart);
            res |= prefixKernel->get().setArg(index++, maxLen);
            res |= prefixKernel->get().setArg(index++, headDim);
            res |= prefixKernel->get().setArg(index++, diskLayout.hP);
            res |= prefixKernel->get().setArg(index++, diskLayout.lP);
            res |= prefixKernel->get().setArg(index++, diskLayout.bytes);
            res |= prefixKernel->get().setArg(index++, static_cast<int>(keySizePerHead));
            res |= prefixKernel->get().setArg(index++, static_cast<int>(valueSizePerHead));
            res |= prefixKernel->get().setArg(index++, ropeDim);
            res |= prefixKernel->get().setArg(index++, segment.rope_theta);
            MNN_CHECK_CL_SUCCESS(res, "setArg prefix_copy_rope");
            run3DKernelDefault(prefixKernel, runGws, lws, runtime);
            dstStart += segment.token_count;
        }

        runtime->commandQueue().finish();
        if (dstStart != segmentTotalTokens) {
            MNN_ERROR("[Error]: OpenCL PrefixAttention copied prefix length %d but expected %d\n",
                      dstStart, segmentTotalTokens);
            return INVALID_VALUE;
        }

        mKVCacheCLManager->setPastKvLength(segmentTotalTokens);
        prepared = true;
        appendKvSeqLen = promptLen;
        return NO_ERROR;
    }

    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (dst == nullptr) {
            return true;
        }
        if (bn->getMetaPtr() == backend()->getMetaPtr()) {
            *dst = new OpenCLPrefixAttentionBufExecution(mKVCacheCLManager, op, bn, mLayerIndex);
        } else {
            *dst = new OpenCLPrefixAttentionBufExecution(op, bn, true, mLayerIndex);
        }
        return true;
    }

private:
    int mLayerIndex = -1;
};

class OpenCLPrefixAttentionBufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        bool kvCache = true;
        int layerIndex = -1;
        auto param = op->main_as_AttentionParam();
        if (param != nullptr) {
            kvCache = param->kv_cache();
            layerIndex = param->layer_index();
        }
        OPENCL_CREATOR_CHECK(new OpenCLPrefixAttentionBufExecution(op, backend, kvCache, layerIndex));
    }
};

REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(OpenCLPrefixAttentionBufCreator, OpType_PrefixAttention, BUFFER);

} // namespace OpenCL
} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
