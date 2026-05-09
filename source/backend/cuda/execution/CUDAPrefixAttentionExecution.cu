#include "AttentionExecution.hpp"
#include "core/MNNFileUtils.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

namespace MNN {
namespace CUDA {

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace {

constexpr int kCpuFlashAttentionBlockSize = 64;

template <typename T = void>
static inline T* getTensorDevicePtr(const Tensor* tensor) {
    if (!tensor || tensor->deviceId() == 0) {
        return nullptr;
    }
    return reinterpret_cast<T*>(tensor->deviceId());
}

__host__ __device__ static int roundUpInt(int value, int unit) {
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
        MNN_ERROR("[Error]: failed to open CUDA prefix cache file: %s\n", path.c_str());
        return false;
    }
    input.seekg(0, std::ios::end);
    auto size = input.tellg();
    if (size <= 0) {
        MNN_ERROR("[Error]: empty CUDA prefix cache file: %s\n", path.c_str());
        return false;
    }
    input.seekg(0, std::ios::beg);
    data.resize(static_cast<size_t>(size));
    input.read(reinterpret_cast<char*>(data.data()), size);
    if (!input) {
        MNN_ERROR("[Error]: failed to read CUDA prefix cache file: %s\n", path.c_str());
        return false;
    }
    return true;
}

__host__ __device__ static size_t packedKeyIndex(int seq, int dim, int headDim, int hP, int lP) {
    return static_cast<size_t>(seq / hP) * roundUpInt(headDim, lP) * hP +
           static_cast<size_t>(dim / lP) * hP * lP +
           static_cast<size_t>(seq % hP) * lP +
           static_cast<size_t>(dim % lP);
}

__host__ __device__ static size_t packedFlashValueIndex(int seq, int dim, int headDim, int hP, int lP) {
    constexpr int blockKv = kCpuFlashAttentionBlockSize;
    auto weightStride2 = lP * hP;
    auto weightStride1 = ((blockKv + lP - 1) / lP) * weightStride2;
    auto weightStride0 = weightStride1 * ((headDim + hP - 1) / hP);
    auto idxInner = (seq / blockKv) * weightStride0 +
                    ((seq % blockKv) / lP) * weightStride2 +
                    (seq % blockKv) % lP;
    auto idxBase = (dim / hP) * weightStride1 + (dim % hP) * lP;
    return static_cast<size_t>(idxBase + idxInner);
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
    // Current kvshare CPU prefix caches on x86 use float pack hP=4/lP=1.
    // Keep CUDA PrefixAttention independent from CPUBackend symbols by resolving
    // the small set of supported floating layouts from file sizes.
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

__device__ static float readDiskScalar(const uint8_t* ptr, int bytes) {
    if (bytes == 2) {
        return __half2float(*reinterpret_cast<const __half*>(ptr));
    }
    return *reinterpret_cast<const float*>(ptr);
}

template <typename T>
__device__ static T castPrefixScalar(float value) {
    return static_cast<T>(value);
}

template <>
__device__ __half castPrefixScalar<__half>(float value) {
    return __float2half(value);
}

template <typename T>
__global__ void materializePrefixKVKernel(const uint8_t* keyFile, const uint8_t* valueFile,
                                          size_t keySizePerHead, size_t valueSizePerHead,
                                          T* keyCache, T* valueCache, int kvHeads, int headDim,
                                          int srcLen, int dstStart, int maxLength,
                                          int hP, int lP, int diskBytes,
                                          int ropeDim, float ropeTheta) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = kvHeads * srcLen * headDim;
    if (index >= total) {
        return;
    }

    int dim = index % headDim;
    int token = (index / headDim) % srcLen;
    int head = index / (headDim * srcLen);
    int dstSeq = dstStart + token;

    const uint8_t* keyHead = keyFile + static_cast<size_t>(head) * keySizePerHead;
    const uint8_t* valueHead = valueFile + static_cast<size_t>(head) * valueSizePerHead;
    auto keyIndex = packedKeyIndex(token, dim, headDim, hP, lP);
    auto valueIndex = packedFlashValueIndex(token, dim, headDim, hP, lP);
    float keyValue = readDiskScalar(keyHead + keyIndex * diskBytes, diskBytes);
    float value = readDiskScalar(valueHead + valueIndex * diskBytes, diskBytes);

    if (dim < ropeDim) {
        int half = ropeDim / 2;
        int pairDim = dim < half ? dim + half : dim - half;
        int freqDim = dim < half ? dim : dim - half;
        auto pairIndex = packedKeyIndex(token, pairDim, headDim, hP, lP);
        float pairValue = readDiskScalar(keyHead + pairIndex * diskBytes, diskBytes);
        float left = dim < half ? keyValue : pairValue;
        float right = dim < half ? pairValue : keyValue;
        float invFreq = 1.0f / powf(ropeTheta, static_cast<float>(2 * freqDim) / static_cast<float>(ropeDim));
        float angle = static_cast<float>(dstSeq) * invFreq;
        float cosValue = cosf(angle);
        float sinValue = sinf(angle);
        keyValue = dim < half ? left * cosValue - right * sinValue : right * cosValue + left * sinValue;
    }

    keyCache[(static_cast<size_t>(dstSeq) * kvHeads + head) * headDim + dim] = castPrefixScalar<T>(keyValue);
    valueCache[(static_cast<size_t>(head) * maxLength + dstSeq) * headDim + dim] = castPrefixScalar<T>(value);
}

} // namespace

class CUDAPrefixAttentionExecution : public AttentionExecution {
public:
    CUDAPrefixAttentionExecution(Backend* backend, bool kvCache, int layerIndex)
        : AttentionExecution(backend, kvCache), mLayerIndex(layerIndex) {
    }

protected:
    virtual ErrorCode onPrepareKVCacheBeforeAppend(const std::vector<Tensor*>& inputs, cudaStream_t stream,
                                                   bool& prepared, int& appendKvSeqLen) override {
        prepared = false;
        appendKvSeqLen = mNewKvSeqLen;
        if (!mIsKVCacheEnabled || mMeta == nullptr ||
            mMeta->file_flag != KVMeta::PendingReadSegments || mMeta->prefix_segments.empty()) {
            return NO_ERROR;
        }
        if (mBatch != 1) {
            MNN_ERROR("[Error]: CUDA PrefixAttention direct_segments only supports batch=1 now\n");
            return NOT_SUPPORT;
        }
        if (mMeta->previous != mMeta->remove) {
            return NO_ERROR;
        }
        if (mMeta->prefix_cache_dir.empty()) {
            MNN_ERROR("[Error]: CUDA PrefixAttention missing prefix_cache_dir in KVMeta\n");
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

        appendKvSeqLen = static_cast<int>(mMeta->add);
        if (appendKvSeqLen <= 0) {
            appendKvSeqLen = mNewKvSeqLen;
        }
        const int requiredTotal = segmentTotalTokens + appendKvSeqLen;

        mCache->mPastLength = 0;
        auto err = reallocKVCache_gpu(requiredTotal, mBatch, mKvNumHead, mHeadDim, stream);
        if (err != NO_ERROR) {
            return err;
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

        int dstStart = 0;
        for (const auto& segment : mMeta->prefix_segments) {
            if (segment.token_count <= 0) {
                continue;
            }
            if (segment.key_rope_state != KVMeta::KeyRopeCanonicalNoRope ||
                segment.rope_pairing != KVMeta::RopePairingHalf) {
                MNN_ERROR("[Error]: CUDA PrefixAttention requires canonical_no_rope half-split prefix cache: %s\n",
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
            if (!resolveDiskPackedKVLayout(keyFile.size(), valueFile.size(), mKvNumHead, mHeadDim,
                                           segment.token_count, diskLayout)) {
                MNN_ERROR("[Error]: CUDA PrefixAttention unsupported floating prefix KV file layout: %s / %s\n",
                          keyPath.c_str(), valuePath.c_str());
                return INVALID_VALUE;
            }

            size_t keySizePerHead = keyFile.size() / mKvNumHead;
            size_t valueSizePerHead = valueFile.size() / mKvNumHead;
            int ropeDim = segment.rope_dim > 0 ? segment.rope_dim : mHeadDim;
            ropeDim = std::min(ropeDim, mHeadDim);
            if (ropeDim <= 0 || (ropeDim % 2) != 0 || segment.rope_theta <= 0.0f) {
                MNN_ERROR("[Error]: CUDA PrefixAttention invalid RoPE metadata for cache %s\n",
                          segment.cache_name.c_str());
                return INVALID_VALUE;
            }

            uint8_t* keyDevice = nullptr;
            uint8_t* valueDevice = nullptr;
            auto keyAlloc = cudaMalloc(reinterpret_cast<void**>(&keyDevice), keyFile.size());
            auto valueAlloc = cudaMalloc(reinterpret_cast<void**>(&valueDevice), valueFile.size());
            if (keyAlloc != cudaSuccess || valueAlloc != cudaSuccess) {
                if (keyDevice != nullptr) {
                    cudaFree(keyDevice);
                }
                if (valueDevice != nullptr) {
                    cudaFree(valueDevice);
                }
                MNN_ERROR("[Error]: CUDA PrefixAttention failed to allocate prefix KV staging buffers\n");
                return OUT_OF_MEMORY;
            }
            auto keyCopy = cudaMemcpyAsync(keyDevice, keyFile.data(), keyFile.size(), cudaMemcpyHostToDevice, stream);
            auto valueCopy = cudaMemcpyAsync(valueDevice, valueFile.data(), valueFile.size(), cudaMemcpyHostToDevice, stream);
            if (keyCopy != cudaSuccess || valueCopy != cudaSuccess) {
                MNN_ERROR("[Error]: CUDA PrefixAttention failed to copy prefix KV staging buffers: %s / %s\n",
                          cudaGetErrorString(keyCopy), cudaGetErrorString(valueCopy));
                cudaFree(keyDevice);
                cudaFree(valueDevice);
                return INVALID_VALUE;
            }

            int totalElements = mKvNumHead * segment.token_count * mHeadDim;
            int threads = 256;
            int blocks = UP_DIV(totalElements, threads);
            if (mPrecision == 4) {
                materializePrefixKVKernel<float><<<blocks, threads, 0, stream>>>(
                    keyDevice, valueDevice, keySizePerHead, valueSizePerHead,
                    getTensorDevicePtr<float>(mCache->mPastKey.get()),
                    getTensorDevicePtr<float>(mCache->mPastValue.get()),
                    mKvNumHead, mHeadDim, segment.token_count, dstStart, mCache->mMaxLength,
                    diskLayout.hP, diskLayout.lP, diskLayout.bytes, ropeDim, segment.rope_theta);
            } else if (mPrecision == 2) {
                materializePrefixKVKernel<__half><<<blocks, threads, 0, stream>>>(
                    keyDevice, valueDevice, keySizePerHead, valueSizePerHead,
                    getTensorDevicePtr<__half>(mCache->mPastKey.get()),
                    getTensorDevicePtr<__half>(mCache->mPastValue.get()),
                    mKvNumHead, mHeadDim, segment.token_count, dstStart, mCache->mMaxLength,
                    diskLayout.hP, diskLayout.lP, diskLayout.bytes, ropeDim, segment.rope_theta);
            } else {
                cudaFree(keyDevice);
                cudaFree(valueDevice);
                return NOT_SUPPORT;
            }
            checkKernelErrors;
            auto syncStatus = cudaStreamSynchronize(stream);
            if (syncStatus != cudaSuccess) {
                MNN_ERROR("[Error]: CUDA PrefixAttention prefix materialize kernel failed: %s\n",
                          cudaGetErrorString(syncStatus));
                cudaFree(keyDevice);
                cudaFree(valueDevice);
                return INVALID_VALUE;
            }
            cudaFree(keyDevice);
            cudaFree(valueDevice);
            dstStart += segment.token_count;
        }

        if (dstStart != segmentTotalTokens) {
            MNN_ERROR("[Error]: CUDA PrefixAttention copied prefix length %d but expected %d\n",
                      dstStart, segmentTotalTokens);
            return INVALID_VALUE;
        }

        mCache->mPastLength = segmentTotalTokens;
        prepared = true;
        return NO_ERROR;
    }

    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (dst == nullptr) {
            return true;
        }
        auto exe = new CUDAPrefixAttentionExecution(bn, mIsKVCacheEnabled, mLayerIndex);
        exe->mCache = mCache;
        exe->mMeta = mMeta;
        *dst = exe;
        return true;
    }

private:
    int mLayerIndex = -1;
};

class CUDAPrefixAttentionCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        bool kvCache = true;
        int layerIndex = -1;
        if (op->main_type() == OpParameter_AttentionParam) {
            auto param = op->main_as_AttentionParam();
            if (param != nullptr) {
                kvCache = param->kv_cache();
                layerIndex = param->layer_index();
            }
        }
        return new CUDAPrefixAttentionExecution(backend, kvCache, layerIndex);
    }
};

static CUDACreatorRegister<CUDAPrefixAttentionCreator> __init_prefix_attention(OpType_PrefixAttention);

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

} // namespace CUDA
} // namespace MNN
