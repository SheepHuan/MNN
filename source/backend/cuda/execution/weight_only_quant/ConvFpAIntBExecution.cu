//
//  ConvFpAIntBExecution.cpp
//  MNN
//
//  Created by MNN on 2024/03/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

// precision = low/normal 时使用运行时离线反量化，precision = high 时使用在线反量化
#include "ConvFpAIntBExecution.hpp"
#include "../Raster.cuh"
#include "../ConvBaseKernel.cuh"
#include "core/KVMeta.hpp"
#include <cuda_fp16.h>
#include <float.h>
#include <cublas_v2.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

//#define DEBUG

namespace MNN {
namespace CUDA {

const int TILE_DIM = 16; // GEMM_FpAIntB
const int GEMV_TILE = 64; // GEMV_FpAIntB (legacy, kept for INT8)
const int GEMV_TILE_V2 = 256; // P1: Optimized GEMV tile size
const int GEMV_OC_PER_BLOCK = 4; // P1: Output channels per block for GEMV
const int SUB_GEMM_TILE_DIM = 16; // GEMM_Int
const int BLOCK_SIZE = 256; // Quant / SumBq / Bias
const int DEQUANT_TILE_DIM = 32; // Dequant
const int WARP_SIZE = 32;
static std::atomic<size_t> gPicStaticDequantBytes{0};
static std::mutex gPicStaticDequantCacheMutex;
static std::unordered_map<std::string, std::weak_ptr<Tensor>> gPicStaticDequantCache;

static bool profilePicWeightOnlyConv() {
    const char* paged = ::getenv("MNN_PAGED_ATTENTION_PROFILE");
    if (paged != nullptr && paged[0] != '\0' && paged[0] != '0') {
        return true;
    }
    const char* graph = ::getenv("MNN_PIC_GRAPH_PROFILE");
    return graph != nullptr && graph[0] != '\0' && graph[0] != '0';
}

static int picRows45CublasPolicy() {
    static const int policy = []() {
        const char* value = ::getenv("MNN_CUDA_PIC_INT4_ROWS45_CUBLAS");
        if (value == nullptr || value[0] == '\0') {
            return 3;
        }
        if (value[0] == '0') {
            return 0;
        }
        if (::strcmp(value, "down") == 0) {
            return 1;
        }
        if (::strcmp(value, "mlp") == 0) {
            return 2;
        }
        if (::strcmp(value, "all") == 0) {
            return 3;
        }
        return std::atoi(value);
    }();
    return policy;
}

static bool picRows45CublasMatches(int policy, int batch, int ic, int oc) {
    if (policy <= 0 || batch < 4 || batch > 8) {
        return false;
    }
    const int minDim = std::min(ic, oc);
    const int maxDim = std::max(ic, oc);
    const bool isMlpLike = minDim >= 1024 && maxDim >= 2 * minDim && maxDim <= 8 * minDim;
    const bool isMlpDown = isMlpLike && ic > oc;
    const bool isMlpGateOrUp = isMlpLike && oc > ic;
    const bool isAttentionSquare = ic == oc && minDim >= 1024 && maxDim <= 4096;
    const bool isAttentionKV = ic > oc && ic <= 4096 && oc >= 512 && ic <= 4 * oc;
    const bool isAttentionProjection = isAttentionSquare || isAttentionKV;
    if (policy == 1) {
        return isMlpDown;
    }
    if (policy == 2) {
        return isMlpDown || isMlpGateOrUp;
    }
    return isMlpDown || isMlpGateOrUp || isAttentionProjection;
}

static int picRows45CublasMathPolicy() {
    static const int policy = []() {
        const char* value = ::getenv("MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_MATH");
        if (value == nullptr || value[0] == '\0') {
            return 1;
        }
        if (value[0] == '0' || ::strcmp(value, "keep") == 0 || ::strcmp(value, "none") == 0) {
            return 0;
        }
        if (::strcmp(value, "default") == 0) {
            return 2;
        }
        if (::strcmp(value, "tensor") == 0) {
            return 1;
        }
        return std::atoi(value);
    }();
    return policy;
}

static int picRows45CublasComputePolicy() {
    static const int policy = []() {
        const char* value = ::getenv("MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_COMPUTE");
        if (value == nullptr || value[0] == '\0' || ::strcmp(value, "32f") == 0) {
            return 0;
        }
        if (::strcmp(value, "32f_fast16") == 0 || ::strcmp(value, "fast16") == 0) {
            return 1;
        }
        if (::strcmp(value, "16f") == 0) {
            return 2;
        }
        return std::atoi(value);
    }();
    return policy;
}

static int picRows45CublasAlgoPolicy() {
    static const int policy = []() {
        const char* value = ::getenv("MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_ALGO");
        if (value == nullptr || value[0] == '\0' || ::strcmp(value, "tensor") == 0) {
#if CUDART_VERSION >= 9000
            return static_cast<int>(CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#else
            return static_cast<int>(CUBLAS_GEMM_DEFAULT);
#endif
        }
        if (::strcmp(value, "default") == 0) {
            return static_cast<int>(CUBLAS_GEMM_DEFAULT);
        }
        return std::atoi(value);
    }();
    return policy;
}

static int picRows48CublasLtPolicy() {
    const char* value = ::getenv("MNN_CUDA_PIC_INT4_ROWS48_CUBLASLT");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    if (value[0] == '0' || ::strcmp(value, "off") == 0 || ::strcmp(value, "disable") == 0) {
        return 0;
    }
    return std::atoi(value);
}

static bool picRows48CublasLtMatches(int batch, int ic, int oc) {
    if (picRows48CublasLtPolicy() <= 0) {
        return false;
    }
    if (!(batch == 4 || batch == 6 || batch == 8)) {
        return false;
    }
    return (ic == 3072 && oc == 8192) || (ic == 8192 && oc == 3072);
}

// V15 rows4/6/8 INT4-native small-M weight-only route (env-gated, default off).
// opt-in only; existing rows45/rows48/generic paths remain the production fallback.
static int picV15SmallMPolicy() {
    static const int policy = []() {
        const char* value = ::getenv("MNN_CUDA_PIC_INT4_SMALLM_V15");
        if (value == nullptr || value[0] == '\0' || value[0] == '0' ||
            ::strcmp(value, "off") == 0 || ::strcmp(value, "disable") == 0) {
            return 0;
        }
        if (::strcmp(value, "all") == 0) {
            return 3;
        }
        if (::strcmp(value, "mlp") == 0) {
            return 2;
        }
        if (::strcmp(value, "down") == 0) {
            return 1;
        }
        return std::atoi(value);
    }();
    return policy;
}

// V15 matches batch in {4,6,8} with the same MLP/attention shape families as
// rows45 cuBLAS. Policy selects which families: down-only / mlp / all(+attn).
static bool picV15SmallMMatches(int policy, int batch, int ic, int oc) {
    if (policy <= 0 || !(batch == 4 || batch == 6 || batch == 8)) {
        return false;
    }
    const int minDim = std::min(ic, oc);
    const int maxDim = std::max(ic, oc);
    const bool isMlpLike = minDim >= 1024 && maxDim >= 2 * minDim && maxDim <= 8 * minDim;
    const bool isMlpDown = isMlpLike && ic > oc;
    const bool isMlpGateOrUp = isMlpLike && oc > ic;
    const bool isAttentionSquare = ic == oc && minDim >= 1024 && maxDim <= 4096;
    const bool isAttentionKV = ic > oc && ic <= 4096 && oc >= 512 && ic <= 4 * oc;
    const bool isAttentionProjection = isAttentionSquare || isAttentionKV;
    if (policy == 1) {
        return isMlpDown;
    }
    if (policy == 2) {
        return isMlpDown || isMlpGateOrUp;
    }
    return isMlpDown || isMlpGateOrUp || isAttentionProjection;
}

// V16 INT8 __dp4a small-M route (env-gated, default off). Independent of V15;
// reuses mGemvParams + a precomputed mSumNibble table. Tests whether dp4a
// arithmetic can beat cuBLAS where int4 factored-GEMV is instruction-bound.
static int picV16Dp4aPolicy() {
    static const int policy = []() {
        const char* value = ::getenv("MNN_CUDA_PIC_INT4_SMALLM_DP4A");
        if (value == nullptr || value[0] == '\0' || value[0] == '0' ||
            ::strcmp(value, "off") == 0 || ::strcmp(value, "disable") == 0) {
            return 0;
        }
        if (::strcmp(value, "all") == 0) {
            return 3;
        }
        if (::strcmp(value, "mlp") == 0) {
            return 2;
        }
        if (::strcmp(value, "down") == 0) {
            return 1;
        }
        return std::atoi(value);
    }();
    return policy;
}

static uint64_t convProfileNowUs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

static size_t picStaticDequantBaseLimitBytes(const cudaDeviceProp& prop) {
    constexpr size_t minLimit = 256ull * 1024ull * 1024ull;
    constexpr size_t maxLimit = 2048ull * 1024ull * 1024ull;
    const size_t byMem = prop.totalGlobalMem / 16;
    return std::max(minLimit, std::min(maxLimit, byMem));
}

static size_t picStaticDequantDecodeHotLimitBytes(const cudaDeviceProp& prop) {
    constexpr size_t defaultMaxLimit = 4096ull * 1024ull * 1024ull;
    constexpr size_t highMemMaxLimit = 6144ull * 1024ull * 1024ull;
    constexpr size_t highMemThreshold = 30ull * 1024ull * 1024ull * 1024ull;
    const bool highMemDevice = prop.totalGlobalMem >= highMemThreshold;
    const size_t maxLimit = highMemDevice ? highMemMaxLimit : defaultMaxLimit;
    const size_t byMem = highMemDevice ? prop.totalGlobalMem / 5 : prop.totalGlobalMem / 8;
    return std::max(picStaticDequantBaseLimitBytes(prop), std::min(maxLimit, byMem));
}

static bool reservePicStaticDequantBytes(size_t bytes, size_t limit) {
    if (bytes == 0 || bytes > limit) {
        return false;
    }
    size_t current = gPicStaticDequantBytes.load(std::memory_order_relaxed);
    while (current <= limit - bytes) {
        if (gPicStaticDequantBytes.compare_exchange_weak(
                current, current + bytes, std::memory_order_relaxed, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

static std::string picStaticDequantCacheKey(Backend* backend, const Op* op, int hp, int icp, size_t bytes) {
    if (backend == nullptr || op == nullptr || op->main_type() != OpParameter_Convolution2D) {
        return "";
    }
    const auto conv = op->main_as_Convolution2D();
    if (conv == nullptr || conv->external() == nullptr || op->externalPath() == nullptr) {
        return "";
    }
    const auto external = conv->external();
    if (external->size() < 3) {
        return "";
    }
    std::string key;
    key.reserve(160);
    auto cudaBackend = static_cast<CUDABackend*>(backend);
    key.append(std::to_string(reinterpret_cast<uintptr_t>(cudaBackend->getCUDARuntime())));
    key.push_back('|');
    key.append(op->externalPath()->str());
    key.push_back('|');
    for (flatbuffers::uoffset_t i = 0; i < external->size(); ++i) {
        if (i > 0) {
            key.push_back(',');
        }
        key.append(std::to_string(external->Get(i)));
    }
    key.push_back('|');
    key.append(std::to_string(hp));
    key.push_back('x');
    key.append(std::to_string(icp));
    key.push_back('|');
    key.append(std::to_string(bytes));
    return key;
}

static std::shared_ptr<Tensor> findPicStaticDequantCache(const std::string& key) {
    if (key.empty()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> guard(gPicStaticDequantCacheMutex);
    auto iter = gPicStaticDequantCache.find(key);
    if (iter == gPicStaticDequantCache.end()) {
        return nullptr;
    }
    auto tensor = iter->second.lock();
    if (tensor == nullptr) {
        gPicStaticDequantCache.erase(iter);
    }
    return tensor;
}

static void insertPicStaticDequantCache(const std::string& key, const std::shared_ptr<Tensor>& tensor) {
    if (key.empty() || tensor == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> guard(gPicStaticDequantCacheMutex);
    gPicStaticDequantCache[key] = tensor;
}

static bool runRows45CublasFp16(cublasHandle_t handle, const void* input, const void* weight, void* output,
                                int batch, int ic, int icp, int oc, int ocp,
                                int* mathPolicyOut, int* computePolicyOut, int* algoPolicyOut) {
    if (handle == nullptr || input == nullptr || weight == nullptr || output == nullptr) {
        return false;
    }
    const int mathPolicy = picRows45CublasMathPolicy();
    const int computePolicy = picRows45CublasComputePolicy();
    const int algoPolicy = picRows45CublasAlgoPolicy();
    if (mathPolicyOut != nullptr) {
        *mathPolicyOut = mathPolicy;
    }
    if (computePolicyOut != nullptr) {
        *computePolicyOut = computePolicy;
    }
    if (algoPolicyOut != nullptr) {
        *algoPolicyOut = algoPolicy;
    }
#if CUDART_VERSION >= 9000
    if (mathPolicy == 1) {
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    } else if (mathPolicy == 2) {
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }
#endif
    const float alpha32 = 1.0f;
    const float beta32 = 0.0f;
    const __half alpha16 = __float2half(1.0f);
    const __half beta16 = __float2half(0.0f);
    const void* alpha = &alpha32;
    const void* beta = &beta32;
#if CUDART_VERSION >= 11000
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    if (computePolicy == 1) {
        computeType = CUBLAS_COMPUTE_32F_FAST_16F;
    } else if (computePolicy == 2) {
        computeType = CUBLAS_COMPUTE_16F;
        alpha = &alpha16;
        beta = &beta16;
    }
#else
    const auto computeType = CUDA_R_32F;
#endif
#if CUDART_VERSION >= 9000
    const auto algo = static_cast<cublasGemmAlgo_t>(algoPolicy);
#else
    const auto algo = CUBLAS_GEMM_DEFAULT;
#endif
    const cublasStatus_t status = cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        oc, batch, ic,
        alpha,
        weight, CUDA_R_16F, icp,
        input, CUDA_R_16F, icp,
        beta,
        output, CUDA_R_16F, ocp,
        computeType, algo);
    if (status != CUBLAS_STATUS_SUCCESS) {
        MNN_ERROR("rows45 cublasGemmEx failed: %d\n", static_cast<int>(status));
        return false;
    }
    return true;
}

#if MNN_CUDA_HAS_CUBLASLT
struct PicRowsCublasLtPlan {
    ~PicRowsCublasLtPlan() {
        if (preference != nullptr) {
            cublasLtMatmulPreferenceDestroy(preference);
        }
        if (cDesc != nullptr) {
            cublasLtMatrixLayoutDestroy(cDesc);
        }
        if (bDesc != nullptr) {
            cublasLtMatrixLayoutDestroy(bDesc);
        }
        if (aDesc != nullptr) {
            cublasLtMatrixLayoutDestroy(aDesc);
        }
        if (opDesc != nullptr) {
            cublasLtMatmulDescDestroy(opDesc);
        }
    }

    bool matches(int b, int i, int ip, int o, int op) const {
        return ready && batch == b && ic == i && icp == ip && oc == o && ocp == op;
    }

    bool init(cublasLtHandle_t handle, int b, int i, int ip, int o, int op, size_t workspaceBytes) {
        if (handle == nullptr) {
            return false;
        }
        batch = b;
        ic = i;
        icp = ip;
        oc = o;
        ocp = op;
        cublasOperation_t transA = CUBLAS_OP_T;
        cublasOperation_t transB = CUBLAS_OP_N;
        if (cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_16F, ic, oc, icp) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_16F, ic, batch, icp) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_16F, oc, batch, ocp) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                &workspaceBytes, sizeof(workspaceBytes)) != CUBLAS_STATUS_SUCCESS) {
            return false;
        }
        int returned = 0;
        if (cublasLtMatmulAlgoGetHeuristic(handle, opDesc, aDesc, bDesc, cDesc, cDesc,
                                           preference, 1, &heuristic, &returned) != CUBLAS_STATUS_SUCCESS ||
            returned <= 0) {
            return false;
        }
        ready = true;
        return true;
    }

    bool run(cublasLtHandle_t handle, const void* input, const void* weight, void* output,
             void* workspace, size_t workspaceBytes) const {
        if (!ready || handle == nullptr) {
            return false;
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        return cublasLtMatmul(handle, opDesc,
                              &alpha,
                              weight, aDesc,
                              input, bDesc,
                              &beta,
                              output, cDesc,
                              output, cDesc,
                              &heuristic.algo,
                              workspace, workspaceBytes,
                              0) == CUBLAS_STATUS_SUCCESS;
    }

    int batch = 0;
    int ic = 0;
    int icp = 0;
    int oc = 0;
    int ocp = 0;
    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t aDesc = nullptr;
    cublasLtMatrixLayout_t bDesc = nullptr;
    cublasLtMatrixLayout_t cDesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic{};
    bool ready = false;
};

static void destroyPicRowsCublasLtPlan(void*& planPtr) {
    if (planPtr != nullptr) {
        delete reinterpret_cast<PicRowsCublasLtPlan*>(planPtr);
        planPtr = nullptr;
    }
}

static PicRowsCublasLtPlan* ensurePicRowsCublasLtPlan(void*& planPtr, cublasLtHandle_t handle,
                                                       int batch, int ic, int icp, int oc, int ocp,
                                                       size_t workspaceBytes) {
    auto plan = reinterpret_cast<PicRowsCublasLtPlan*>(planPtr);
    if (plan != nullptr && plan->matches(batch, ic, icp, oc, ocp)) {
        return plan;
    }
    destroyPicRowsCublasLtPlan(planPtr);
    plan = new PicRowsCublasLtPlan;
    if (!plan->init(handle, batch, ic, icp, oc, ocp, workspaceBytes)) {
        delete plan;
        return nullptr;
    }
    planPtr = plan;
    return plan;
}
#else
static void destroyPicRowsCublasLtPlan(void*& planPtr) {
    planPtr = nullptr;
}
#endif

template<typename T, typename dT>
__global__ void DequantizeInt8Weight(
    const int8_t* quantized_kernel,
    dT* dequantized_kernel,
    const T* scale,
    const T* offset,
    const int oc, const int ic, const int ic_p, const int quanC
) {
    // 每个线程负责反量化一个权重值
    const int col = blockIdx.x * blockDim.x + threadIdx.x; // input channel
    const int row = blockIdx.y * blockDim.y + threadIdx.y; // output channel

    if (row < oc && col < ic) {
        // 计算量化参数的索引
        const int num_quan_groups_per_channel = (quanC > 0) ? (quanC / oc) : 1;
        const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;
        const int group_idx = col / ic_per_group;
        const int quan_param_index = row * num_quan_groups_per_channel + group_idx;

        const float x_scale  = (float)scale[quan_param_index];
        const float x_offset = (float)offset[quan_param_index];

        // 读取量化的权重
        const float quantized_val = quantized_kernel[row * ic_p + col];

        // 反量化并写入目标 Tensor
        dequantized_kernel[row * ic_p + col] = (dT)(quantized_val * x_scale + x_offset);
    }
}

// Optimized INT4 weight dequantization: each thread unpacks 8 bytes → 16 FP16 values
// Uses uint2 vectorized load for weights and float4 (8 half) vectorized writes
template<typename T, typename dT>
__global__ void DequantizeInt4Weight(
    const uint8_t* __restrict__ packed_weight,
    dT* __restrict__ output,
    const T* __restrict__ scale,
    const T* __restrict__ offset,
    const int oc, const int ic, const int ic_p, const int quanC
) {
    // Each thread handles 8 packed bytes = 16 int4 values
    const int tid = blockIdx.y * blockDim.x + threadIdx.x;
    const int row = blockIdx.x;  // OC index

    if (row >= oc) return;

    const int half_ic = ic_p / 2;  // bytes per row
    const int col_byte = tid * 8;  // starting byte within row
    if (col_byte >= half_ic) return;

    const int num_qg = (quanC > 0) ? (quanC / oc) : 1;
    const int ic_per_group = max(1, (num_qg > 0) ? (ic / num_qg) : ic);
    const int qparam_base = row * num_qg;

    // Vectorized 8-byte weight load (two uint32s)
    const uint32_t* src = reinterpret_cast<const uint32_t*>(packed_weight + row * half_ic + col_byte);
    const uint32_t packed0 = __ldg(src);
    const uint32_t packed1 = (col_byte + 4 < half_ic) ? __ldg(src + 1) : 0;

    const int col = col_byte * 2;  // starting ic index
    const int out_base = row * ic_p + col;

    // Process 8 bytes = 16 int4 values, write as half2 pairs
    half result[16];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const int kk = col + j * 2;
        if (kk >= ic) break;
        const int gidx = min(kk / ic_per_group, num_qg - 1);
        const float s = (float)__ldg(&scale[qparam_base + gidx]);
        const float o = (float)__ldg(&offset[qparam_base + gidx]);
        const uint8_t byte_val = (packed0 >> (j * 8)) & 0xFF;
        result[j * 2] = (half)(((float)(byte_val >> 4) - 8.0f) * s + o);
        result[j * 2 + 1] = (half)(((float)(byte_val & 0x0F) - 8.0f) * s + o);
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const int kk = col + 8 + j * 2;
        if (kk >= ic) break;
        const int gidx = min(kk / ic_per_group, num_qg - 1);
        const float s = (float)__ldg(&scale[qparam_base + gidx]);
        const float o = (float)__ldg(&offset[qparam_base + gidx]);
        const uint8_t byte_val = (packed1 >> (j * 8)) & 0xFF;
        result[8 + j * 2] = (half)(((float)(byte_val >> 4) - 8.0f) * s + o);
        result[8 + j * 2 + 1] = (half)(((float)(byte_val & 0x0F) - 8.0f) * s + o);
    }

    // Vectorized write: 16 halves = 32 bytes = float4 + float4
    if (col + 8 <= ic) {
        *reinterpret_cast<float4*>(output + out_base) = *reinterpret_cast<float4*>(result);
    } else {
        for (int i = 0; i < 8 && col + i < ic; i++) output[out_base + i] = (dT)result[i];
    }
    if (col + 16 <= ic) {
        *reinterpret_cast<float4*>(output + out_base + 8) = *reinterpret_cast<float4*>(result + 8);
    } else if (col + 8 < ic) {
        for (int i = 8; i < 16 && col + i < ic; i++) output[out_base + i] = (dT)result[i];
    }
}

// R6: Pre-compute GEMV params: scale and adj_offset as float2 array
// Eliminates scattered offset loads and runtime computation in GEMV
// Grid: (ceil(total/256)), Block: 256
template<typename T>
__global__ void PrecomputeGemvParams(
    const T* __restrict__ scale,
    const T* __restrict__ offset,
    float2* __restrict__ params,   // [oc * num_qg], .x=scale, .y=adj_offset
    const int total                // oc * num_qg
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const float s = (float)scale[idx];
    const float o = (float)offset[idx];
    params[idx] = make_float2(s, o - 8.0f * s);  // adj_offset = offset - 8*scale
}

// 动态量化 A、计算 sum_A
template<typename T>
__global__ void QuantA(
    const T* A_sub_fp, int8_t* A_sub_q,
    T* scale_A_out, T* offset_A_out, int32_t* sum_A_q_out,
    const int M, const int K_i, const int lda
) {
    // 使用 union 复用共享内存，减少总占用量
    __shared__ union {
        float min_max_vals[2][BLOCK_SIZE / WARP_SIZE]; // 用于 min/max 的 warp 间规约
        int32_t sum_vals[BLOCK_SIZE / WARP_SIZE];      // 用于 sum 的 warp 间规约
        float scale_offset[2];                        // 用于向块内所有线程广播 scale 和 offset
    } smem;

    const int m = blockIdx.x;
    if (m >= M) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    float my_min = FLT_MAX;
    float my_max = -FLT_MAX;

    for (int k = tid; k < K_i; k += BLOCK_SIZE) {
        float val = (float)A_sub_fp[m * lda + k];
        my_min = min(my_min, val);
        my_max = max(my_max, val);
    }

    // Warp 内规约：使用 __shfl_down_sync 在 warp 内无锁计算 min/max
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        my_min = min(my_min, __shfl_down_sync(0xFFFFFFFF, my_min, offset));
        my_max = max(my_max, __shfl_down_sync(0xFFFFFFFF, my_max, offset));
    }

    // Warp 间规约：每个 warp 的 0 号线程将 warp 结果写入共享内存
    if (lane_id == 0) {
        smem.min_max_vals[0][warp_id] = my_min;
        smem.min_max_vals[1][warp_id] = my_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        // 从共享内存加载各 warp 的结果
        if (lane_id < (BLOCK_SIZE / WARP_SIZE)) {
            my_min = smem.min_max_vals[0][lane_id];
            my_max = smem.min_max_vals[1][lane_id];
        } else {
            my_min = FLT_MAX;
            my_max = -FLT_MAX;
        }

        // 在第一个 warp 内部完成最终规约
        for (int offset = (BLOCK_SIZE / WARP_SIZE) / 2; offset > 0; offset >>= 1) {
            my_min = min(my_min, __shfl_down_sync(0xFFFFFFFF, my_min, offset));
            my_max = max(my_max, __shfl_down_sync(0xFFFFFFFF, my_max, offset));
        }
    }
    // 线程 0 计算 scale/offset 并写入共享内存进行广播
    if (tid == 0) {
        float scale = (my_max - my_min) / 255.0f;
        float offset;
        if (abs(scale) > 1e-5) {
            offset = my_max - scale * 127.0f; // my_max 映射到 127, my_min 映射到 -128
        } else {
            scale = 1.0f;
            offset = my_max;
        }

        scale_A_out[m] = (T)scale;
        offset_A_out[m] = (T)offset;
        
        smem.scale_offset[0] = scale;
        smem.scale_offset[1] = offset;
    }
    __syncthreads();

    // 所有线程从共享内存读取 scale/offset, 并进行量化和求和
    const float s_scale_A = smem.scale_offset[0];
    const float s_offset_A = smem.scale_offset[1];

    int32_t my_sum_q = 0;
    for (int k = tid; k < K_i; k += BLOCK_SIZE) {
        float val_fp = (float)A_sub_fp[m * lda + k];
        int32_t val_q = roundf((val_fp - s_offset_A) / s_scale_A);
        int8_t a_q = (int8_t)max(-128, min(127, val_q));
        A_sub_q[m * K_i + k] = a_q;
        my_sum_q += a_q;
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        my_sum_q += __shfl_down_sync(0xFFFFFFFF, my_sum_q, offset);
    }

    if (lane_id == 0) smem.sum_vals[warp_id] = my_sum_q;
    __syncthreads();

    if (warp_id == 0) {
        my_sum_q = (lane_id < (BLOCK_SIZE / WARP_SIZE)) ? smem.sum_vals[lane_id] : 0;
        for (int offset = (BLOCK_SIZE / WARP_SIZE) / 2; offset > 0; offset >>= 1) {
             my_sum_q += __shfl_down_sync(0xFFFFFFFF, my_sum_q, offset);
        }
    }

    // 线程 0 将最终的和写入全局内存
    if (tid == 0) sum_A_q_out[m] = my_sum_q;
}

__global__ void GEMM_Int8(
    const int8_t* A_q, const int8_t* B_q, int32_t* C_q,
    const int M, const int N, const int K_i,
    const int lda_q, const int ldb, const int ldc
) {
    __shared__ int8_t A_tile_s8[SUB_GEMM_TILE_DIM][SUB_GEMM_TILE_DIM];
    __shared__ int8_t B_tile_s8[SUB_GEMM_TILE_DIM][SUB_GEMM_TILE_DIM];

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int row = block_row * SUB_GEMM_TILE_DIM + ty;
    const int col = block_col * SUB_GEMM_TILE_DIM + tx;

    int32_t acc = 0;
    const int num_k_tiles = UP_DIV(K_i, SUB_GEMM_TILE_DIM);

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_tile_base = k_tile * SUB_GEMM_TILE_DIM;

        A_tile_s8[ty][tx] = (row < M && (k_tile_base + tx) < K_i) ? A_q[row * lda_q + k_tile_base + tx] : 0;
        B_tile_s8[ty][tx] = (col < N && (k_tile_base + ty) < K_i) ? B_q[col * ldb + k_tile_base + ty] : 0;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < SUB_GEMM_TILE_DIM; ++k) {
            acc += (int32_t)A_tile_s8[ty][k] * (int32_t)B_tile_s8[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) C_q[row * ldc + col] = acc;
}

template<typename T>
__global__ void DequantAndAcc(
    const int32_t* C_q, T* C_fp_final,
    const T* scale_A_in, const T* offset_A_in,
    const T* base_scale_B, const T* base_offset_B,
    const int32_t* base_sum_B_q,
    const int group_idx, const int num_oc_groups,
    const int32_t* sum_A_q_in,
    const int M, const int N, const int K_i, const int ldc
) {
    // 使用共享内存缓存行和列的量化参数
    __shared__ float smem_scale_A[DEQUANT_TILE_DIM];
    __shared__ float smem_offset_A[DEQUANT_TILE_DIM];
    __shared__ int32_t smem_sum_A_q[DEQUANT_TILE_DIM];

    __shared__ float smem_scale_B[DEQUANT_TILE_DIM];
    __shared__ float smem_offset_B[DEQUANT_TILE_DIM];
    __shared__ int32_t smem_sum_B_q[DEQUANT_TILE_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int block_row_start = blockIdx.y * DEQUANT_TILE_DIM;
    const int block_col_start = blockIdx.x * DEQUANT_TILE_DIM;

    // 每个线程加载一个 A 的参数
    int m_load_idx = block_row_start + ty;
    if (tx == 0 && m_load_idx < M) {
        smem_scale_A[ty]   = (float)scale_A_in[m_load_idx];
        smem_offset_A[ty]  = (float)offset_A_in[m_load_idx];
        smem_sum_A_q[ty]   = sum_A_q_in[m_load_idx];
    }

    // 每个线程加载一个 B 的参数
    int n_load_idx = block_col_start + tx;
    if (ty == 0 && n_load_idx < N) {
        const size_t b_param_idx = n_load_idx * num_oc_groups + group_idx;
        smem_scale_B[tx]  = (float)base_scale_B[b_param_idx];
        smem_offset_B[tx] = (float)base_offset_B[b_param_idx];
        smem_sum_B_q[tx]  = base_sum_B_q[b_param_idx];
    }

    __syncthreads();

    // 计算每个线程负责的输出点
    const int m = block_row_start + ty;
    const int n = block_col_start + tx;

    if (m < M && n < N) {
        // 从共享内存中读取参数
        const float scale_A = smem_scale_A[ty];
        const float offset_A = smem_offset_A[ty];
        const float sum_A_q = (float)smem_sum_A_q[ty];

        const float scale_B = smem_scale_B[tx];
        const float offset_B = smem_offset_B[tx];
        const float sum_B_q = (float)smem_sum_B_q[tx];
        
        const float c_q_val = (float)C_q[m * ldc + n];

        float term1 = scale_A * (c_q_val * scale_B + sum_A_q * offset_B);
        float term2 = offset_A * (sum_B_q * scale_B + K_i * offset_B);
        float final_val = term1 + term2;

        #if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700))) || defined(_NVHPC_CUDA)
            if constexpr (std::is_same_v<T, float>) {
                atomicAdd(&C_fp_final[m * ldc + n], final_val);
            } else if constexpr (std::is_same_v<T, half>) {
                // printf("[final_val]%.4f %.4f\n", __half2float(C_fp_final[m * ldc + n]), final_val);
                atomicAdd(&C_fp_final[m * ldc + n], __float2half(final_val));
                // printf("[C_fp_final]%.4f\n", __half2float(C_fp_final[m * ldc + n]));
            }
        #else
            C_fp_final[m * ldc + n] += final_val;
        #endif
    }
}

// 预计算权重 B 的修正项 (sum_B_q)
__global__ void Precompute_SumBq(
    const int8_t* B_q, int32_t* sum_B_q_out,
    const int num_groups, const int ic_per_group,
    const int oc, const int ic_p
) {
    // 每个线程处理一个 (output_channel, group) 的修正项
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)oc * num_groups; index += blockDim.x * gridDim.x) {
        
        const int i = index % num_groups; // group_idx
        const int n = index / num_groups; // output_channel
        
        const int k_start = i * ic_per_group;
        
        int32_t sum = 0;
        for (int k_offset = 0; k_offset < ic_per_group; ++k_offset) {
            // 访问 B 矩阵的布局是 [oc][ic_p]
            sum += (int32_t)B_q[n * ic_p + k_start + k_offset];
        }
        sum_B_q_out[index] = sum;
    }
}

template<typename T>
__global__ void BiasAndActivation(
    T* data, const T* bias,
    const float minV, const float maxV,
    const int M, const int N, const int ldc
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)M * N; index += blockDim.x * gridDim.x) {
        const int m = index / N;
        const int n = index % N;

        const size_t buffer_idx = m * ldc + n;
        float val = (float)data[buffer_idx];
        val += (float)bias[n];
        val = max(val, minV);
        val = min(val, maxV);
        data[buffer_idx] = (T)val;
    }
}

template <typename T>
__global__ void print_tensor_kernel(const T* d_tensor, int rows, int cols) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("----------- Tensor in Kernel -----------\n");
        printf("Rows: %d, Cols: %d\n", rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                T value = d_tensor[i * cols + j];

                if constexpr (std::is_same_v<T, half>) {
                    printf("%.4f ", __half2float(value));
                } else {
                    printf("%.4f ", value);
                }
            }
            printf("\n");
        }
        printf("---------------------------------------\n");
    }
}

template<typename T>
__global__ void GEMM_FpAInt8B(
    const T* input,
    const int8_t* kernel,
    const T* scale, const T* offset, const T* bias,
    T* output,
    const float maxV, const float minV,
    const int ic, const int ic_p, const int oc, const int oc_p,
    const int batch, const int quanC
) {
    __shared__ T A_tile[TILE_DIM][TILE_DIM]; // [batch, ic]
    __shared__ int8_t B_tile_s8[TILE_DIM][TILE_DIM]; // [ic, oc] kernel 本身是 B^T [oc, ic]
    __shared__ T B_tile_fp[TILE_DIM][TILE_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int block_row = blockIdx.y; // batch
    const int block_col = blockIdx.x; // output channel

    // 每个线程负责计算输出Tile中的一个元素
    const int out_row = block_row * TILE_DIM + ty; // M
    const int out_col = block_col * TILE_DIM + tx; // N

    float acc = 0.0f;
    // 沿K维度（输入通道ic）分块循环
    const int num_k_tiles = UP_DIV(ic, TILE_DIM);

    const int num_quan_groups_per_channel = (quanC > 0) ? (quanC / oc) : 1;
    const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_tile_base = k_tile * TILE_DIM;

        // 合并访问加载 A_tile (input)，线程 (ty, tx) 加载 A_tile[ty][tx]
        int a_col_idx = k_tile_base + tx;
        A_tile[ty][tx] = (out_row < batch && a_col_idx < ic) ? input[out_row * ic_p + a_col_idx] : (T)0.0f;
        
        // 合并访问加载 B_tile (kernel)，线程 (ty, tx) 加载 B_tile[ty][tx]
        // kernel 布局为 [oc, ic]，需要 B(k,n)，即 kernel(n,k)
        int b_load_row = block_col * TILE_DIM + ty;
        int b_col_idx = k_tile_base + tx;
        B_tile_s8[ty][tx] = (b_load_row < oc && b_col_idx < ic) ? kernel[b_load_row * ic_p + b_col_idx] : 0;

        __syncthreads();

        // 反量化 + 转置
        const int K = ty;
        const int N = tx;

        const int global_k = k_tile_base + K;
        const int global_n = block_col * TILE_DIM + N;

        if (global_n < oc && global_k < ic) {
            const int group_idx = global_k / ic_per_group;
            const int quan_param_index = global_n * num_quan_groups_per_channel + group_idx;

            const float x_scale  = (float)scale[quan_param_index];
            const float x_offset = (float)offset[quan_param_index];
            
            // B_tile_s8(n,k), thread(n,k) -> (tx, ty)
            // So we need to read from B_tile_s8[n_dim_in_tile][k_dim_in_tile] -> B_tile_s8[tx][ty]
            const float b_quant = (float)B_tile_s8[N][K];

            // B_tile_fp[k][n] -> B_tile_fp[ty][tx]
            B_tile_fp[K][N] = (T)(b_quant * x_scale + x_offset);
        }
        
        __syncthreads();

        // 在 SMEM 中进行子矩阵乘法
        if (out_col < oc) {
            #pragma unroll
            for (int k = 0; k < TILE_DIM; ++k) {
                acc += (float)A_tile[ty][k] * (float)B_tile_fp[k][tx];
            }
        }

        __syncthreads();
    }

    // 写回全局内存
    if (out_row < batch && out_col < oc) {
        acc += (float)bias[out_col];
        acc = max(acc, minV);
        acc = min(acc, maxV);
        output[out_row * oc_p + out_col] = (T)acc;
    }
}

template<typename T>
__global__ void GEMV_FpAInt8B(
    const T* input,
    const int8_t* kernel,
    const T* scale, const T* offset, const T* bias,
    T* output,
    const float maxV, const float minV,
    const int batch, const int ic, const int ic_p,
    const int oc, const int oc_p, const int quanC
) {
    __shared__ float partial_sums[GEMV_TILE]; // 每个线程块计算一个输出通道

    const int oz = blockIdx.x;
    const int b  = blockIdx.y;

    // if (oz >= oc || b >= batch) return;

    const int tid = threadIdx.x; // 块内线程ID
    
    // 准备量化参数
    const int num_quan_groups_per_channel = (quanC > 0) ? (quanC / oc) : 1;
    const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;
    const int quan_param_index_base = oz * num_quan_groups_per_channel;

    // 每个线程计算自己的部分和
    float my_sum = 0.0f;
    for (int k = tid; k < ic; k += blockDim.x) {
        const float inp0 = input[b * ic_p + k];
        const int8_t ker0 = kernel[oz * ic_p + k];
        
        const int group_idx = k / ic_per_group;
        const int quan_param_index = quan_param_index_base + group_idx;
        const float x_scale  = scale[quan_param_index];
        const float x_offset = offset[quan_param_index];
        
        my_sum += inp0 * ((float)ker0 * x_scale + x_offset);
    }

    partial_sums[tid] = my_sum;
    __syncthreads();

    // 在 SMEM 中执行并行规约（蝶式交换）
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial_sums[tid] += partial_sums[tid + s];
        __syncthreads();
    }

    if (tid == 0) { // 写回全局内存
        float final_val = partial_sums[0] + (float)bias[oz];
        final_val = max(final_val, minV);
        final_val = min(final_val, maxV);
        output[b * oc_p + oz] = (T)final_val;
    }
}

template<typename T>
__global__ void GEMM_FpAInt4B(
    const T* input,
    const uint8_t* kernel, // kernel 是打包的 int4
    const T* scale, const T* offset, const T* bias,
    T* output,
    const float maxV, const float minV,
    const int ic, const int ic_p, const int oc, const int oc_p,
    const int batch, const int quanC
) {
    __shared__ T A_tile[TILE_DIM][TILE_DIM]; // [batch, ic]
    __shared__ uint8_t B_tile_u8[TILE_DIM][TILE_DIM / 2]; // [ic, oc] kernel 本身是 B^T [oc, ic]
    __shared__ T B_tile_fp[TILE_DIM][TILE_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int block_row = blockIdx.y; // batch
    const int block_col = blockIdx.x; // output channel

    const int out_row = block_row * TILE_DIM + ty; // M
    const int out_col = block_col * TILE_DIM + tx; // N

    float acc = 0.0f;
    const int num_k_tiles = UP_DIV(ic, TILE_DIM);
    
    const int num_quan_groups_per_channel = (quanC > 0) ? (quanC / oc) : 1;
    const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_tile_base = k_tile * TILE_DIM;

        // 合并访问加载 A_tile (input)
        int a_col_idx = k_tile_base + tx;
        A_tile[ty][tx] = (out_row < batch && a_col_idx < ic) ? input[out_row * ic_p + a_col_idx] : (T)0.0f;
        
        // 合并访问加载 B_tile (kernel)，每个线程加载一个 byte，但 tile 宽度减半
        if (tx < TILE_DIM / 2) {
            int b_load_row = block_col * TILE_DIM + ty;
            // kernel 布局为 [oc, ic/2]，所以列索引要除以2
            int b_col_idx = (k_tile_base / 2) + tx;
            B_tile_u8[ty][tx] = (b_load_row < oc && (k_tile_base + tx*2) < ic) ? kernel[b_load_row * (ic_p / 2) + b_col_idx] : 0;
        }

        __syncthreads();

        // 反量化 + 转置
        const int K = ty;
        const int N = tx;

        const int global_k = k_tile_base + K;
        const int global_n = block_col * TILE_DIM + N;

        if (global_n < oc && global_k < ic) {
            const int group_idx = global_k / ic_per_group;
            const int quan_param_index = global_n * num_quan_groups_per_channel + group_idx;

            const float x_scale  = (float)scale[quan_param_index];
            const float x_offset = (float)offset[quan_param_index];
            
            // B_tile_u8(n,k) -> B_tile_u8[tx][ty/2] (转置后)
            const uint8_t b_packed = B_tile_u8[N][K / 2];
            // 根据 K 的奇偶性解包 int4
            const int8_t b_quant_s8 = (K % 2 == 0) ? ((b_packed >> 4) - 8) : ((b_packed & 0x0F) - 8);
            const float b_quant = (float)b_quant_s8;
            
            // B_tile_fp[k][n] -> B_tile_fp[ty][tx]
            B_tile_fp[K][N] = (T)(b_quant * x_scale + x_offset);
        }
        
        __syncthreads();

        // 在 SMEM 中进行子矩阵乘法
        if (out_col < oc) {
            #pragma unroll
            for (int k = 0; k < TILE_DIM; ++k) {
                acc += (float)A_tile[ty][k] * (float)B_tile_fp[k][tx];
            }
        }

        __syncthreads();
    }

    // 写回全局内存
    if (out_row < batch && out_col < oc) {
        acc += (float)bias[out_col];
        acc = max(acc, minV);
        acc = min(acc, maxV);
        output[out_row * oc_p + out_col] = (T)acc;
    }
}

template<typename T>
__global__ void GEMV_FpAInt4B(
    const T* input,
    const uint8_t* kernel, // kernel 是打包的 int4
    const T* scale, const T* offset, const T* bias,
    T* output,
    const float maxV, const float minV,
    const int batch, const int ic, const int ic_p,
    const int oc, const int oc_p, const int quanC
) {
    extern __shared__ uint8_t smem_buffer[];
    T* smem_input = reinterpret_cast<T*>(smem_buffer);
    float* partial_sums = reinterpret_cast<float*>(smem_buffer + ic_p * sizeof(T));
    
    const int oz = blockIdx.x; // 当前线程块负责计算的输出通道索引
    const int b  = blockIdx.y;

    const int tid = threadIdx.x;
    
    // if (oz >= oc || b >= batch) return;
    
    // 加载 input 到共享内存
    for (int i = tid; i < ic; i += blockDim.x) smem_input[i] = input[b * ic_p + i];
    for (int i = ic + tid; i < ic_p; i += blockDim.x) smem_input[i] = (T)0.0f;
    __syncthreads();

    const int num_quan_groups_per_channel = (quanC > 0) ? (quanC / oc) : 1;
    const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;
    const int quan_param_index_base = oz * num_quan_groups_per_channel;

    float my_sum = 0.0f;
    for (int k = tid * 2; k < ic; k += blockDim.x * 2) {
        // 加载打包的权重并解包
        const uint8_t ker_packed = kernel[oz * (ic_p / 2) + k / 2];
        const int8_t ker0_s8 = (ker_packed >> 4) - 8;
        const int8_t ker1_s8 = (ker_packed & 0x0F) - 8;

        const int group_idx0 = k / ic_per_group;
        const int quan_param_index0 = quan_param_index_base + group_idx0;
        const float x_scale0  = scale[quan_param_index0];
        const float x_offset0 = offset[quan_param_index0];
        my_sum += (float)smem_input[k] * ((float)ker0_s8 * x_scale0 + x_offset0);
        my_sum += (float)smem_input[k + 1] * ((float)ker1_s8 * x_scale0 + x_offset0);
        
        // if (k + 1 < ic) {
        //     const float inp1 = (float)smem_input[k + 1];
        //     const int group_idx1 = (k + 1) / ic_per_group;

        //     if (group_idx0 == group_idx1) {
        //         my_sum += inp1 * ((float)ker1_s8 * x_scale0 + x_offset0);
        //     } else {
        //         const int quan_param_index1 = quan_param_index_base + group_idx1;
        //         const float x_scale1  = (float)scale[quan_param_index1];
        //         const float x_offset1 = (float)offset[quan_param_index1];
        //         my_sum += inp1 * ((float)ker1_s8 * x_scale1 + x_offset1);
        //     }
        // }
    }

    partial_sums[tid] = my_sum;
    __syncthreads();

    // 并行规约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial_sums[tid] += partial_sums[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        float final_val = partial_sums[0] + (float)bias[oz];
        final_val = max(final_val, minV);
        final_val = min(final_val, maxV);
        output[b * oc_p + oz] = (T)final_val;
    }
}


// =====================================================================
// P1-V5: 128-thread GEMV — all 128 threads cooperate on 1 OC
// - Grid: (oc, batch), Block: 128
// =====================================================================
template<typename T>
__global__ void GEMV_FpAInt4B_V5(
    const T* __restrict__ input,
    const uint8_t* __restrict__ kernel,  // packed int4 weights [oc, ic_p/2]
    const T* __restrict__ scale, const T* __restrict__ offset, const T* __restrict__ bias,
    T* __restrict__ output,
    const float maxV, const float minV,
    const int batch, const int ic, const int ic_p,
    const int oc, const int oc_p, const int quanC
) {
    const int oz = blockIdx.x;  // 1 OC per block
    const int b = blockIdx.y;
    if (oz >= oc || b >= batch) return;

    const int tid = threadIdx.x;  // 0..127
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int num_qg = (quanC > 0) ? (quanC / oc) : 1;
    const int ic_per_group = (num_qg > 0) ? (ic / num_qg) : ic;
    const int qparam_base = oz * num_qg;

    const int row_bytes = ic_p / 2;
    const uint8_t* w_row = kernel + oz * row_bytes;
    const T* in_ptr = input + b * ic_p;

    float acc = 0.0f;

    // 128 threads × 8 elements = 1024 per iteration
    for (int k = tid * 8; k < ic; k += 1024) {
        uint32_t w32 = __ldg(reinterpret_cast<const uint32_t*>(w_row + k / 2));

        const int gidx = k / ic_per_group;
        const float s = (float)__ldg(&scale[qparam_base + gidx]);
        const float o = (float)__ldg(&offset[qparam_base + gidx]);
        const float adj_off = o - 8.0f * s;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int kk = k + j * 2;
            if (kk >= ic) break;
            const uint8_t packed = (w32 >> (j * 8)) & 0xFF;
            const float w0 = (float)(packed >> 4) * s + adj_off;
            const float w1 = (float)(packed & 0x0F) * s + adj_off;
            acc += (float)in_ptr[kk] * w0;
            if (kk + 1 < ic) acc += (float)in_ptr[kk + 1] * w1;
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    }

    // Cross-warp reduction via shared memory (4 warps → 1 result)
    __shared__ float smem[4];
    if (lane_id == 0) smem[warp_id] = acc;
    __syncthreads();

    if (tid == 0) {
        float result = smem[0] + smem[1] + smem[2] + smem[3];
        result += (float)bias[oz];
        result = fmaxf(result, minV);
        result = fminf(result, maxV);
        output[b * oc_p + oz] = (T)result;
    }
}

// =====================================================================
// V9: Multi-OC GEMV — each block processes OC_PER_BLK output channels
// 128 threads cooperatively reduce each OC in sequence (fewer blocks, better latency hiding)
// Grid: (ceil(oc/OC_PER_BLK), batch), Block: 128
// =====================================================================
template<typename T, int OC_PER_BLK>
__global__ void GEMV_FpAInt4B_V9(
    const T* __restrict__ input,
    const uint8_t* __restrict__ kernel,  // packed int4 weights [oc, ic_p/2]
    const T* __restrict__ scale, const T* __restrict__ offset, const T* __restrict__ bias,
    T* __restrict__ output,
    const float maxV, const float minV,
    const int batch, const int ic, const int ic_p,
    const int oc, const int oc_p, const int quanC
) {
    const int oc_base = blockIdx.x * OC_PER_BLK;
    const int b = blockIdx.y;
    if (oc_base >= oc || b >= batch) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int num_qg = (quanC > 0) ? (quanC / oc) : 1;
    const int ic_per_group = (num_qg > 0) ? (ic / num_qg) : ic;

    const int row_bytes = ic_p / 2;
    const T* in_ptr = input + b * ic_p;

    // blockDim.x threads, (blockDim.x/32) warps
    const int nwarps = blockDim.x / 32;
    __shared__ float smem[8];  // up to 8 warps (256 threads)

    #pragma unroll
    for (int oc_off = 0; oc_off < OC_PER_BLK; oc_off++) {
        const int oz = oc_base + oc_off;
        if (oz >= oc) break;

        const int qparam_base = oz * num_qg;
        const uint8_t* w_row = kernel + oz * row_bytes;

        float acc = 0.0f;

        // Each thread handles 8 IC elements per iteration, stride = nthreads * 8
        const int stride = blockDim.x * 8;
        for (int k = tid * 8; k < ic; k += stride) {
            uint32_t w32 = __ldg(reinterpret_cast<const uint32_t*>(w_row + k / 2));

            const int gidx = k / ic_per_group;
            const float s = (float)__ldg(&scale[qparam_base + gidx]);
            const float o = (float)__ldg(&offset[qparam_base + gidx]);
            const float adj_off = o - 8.0f * s;

            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int kk = k + j * 2;
                if (kk >= ic) break;
                const uint8_t packed = (w32 >> (j * 8)) & 0xFF;
                const float w0 = (float)(packed >> 4) * s + adj_off;
                const float w1 = (float)(packed & 0x0F) * s + adj_off;
                acc += (float)in_ptr[kk] * w0;
                if (kk + 1 < ic) acc += (float)in_ptr[kk + 1] * w1;
            }
        }

        // Warp-level reduction
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_xor_sync(0xffffffff, acc, off);

        if (lane_id == 0) smem[warp_id] = acc;
        __syncthreads();

        if (tid == 0) {
            float result = 0.0f;
            for (int w = 0; w < nwarps; w++) result += smem[w];
            result += (float)bias[oz];
            result = fmaxf(result, minV);
            result = fminf(result, maxV);
            output[b * oc_p + oz] = (T)result;
        }
        __syncthreads();
    }
}




// =====================================================================
// V14: Factored-dequant multi-OC GEMV — reduces ALU per weight byte.
// Key insight: acc += in * (s * nibble + adj) = s * sum(in*nibble) + adj * sum(in)
// By factoring out scale/adj, we save 6 FMA per OC per 8-IC iteration.
// Also: explicit load-compute separation for better instruction scheduling.
// Grid: (ceil(oc/OC_PER_BLK), batch), Block: 128
// =====================================================================
template<typename T, int OC_PER_BLK>
__global__ void __launch_bounds__(128, 12) GEMV_FpAInt4B_V14(
    const T* __restrict__ input,
    const uint8_t* __restrict__ kernel,  // packed int4 weights [oc, ic_p/2]
    const float2* __restrict__ gemv_params,  // [oc * num_qg], .x=scale, .y=adj_offset
    const T* __restrict__ bias,
    T* __restrict__ output,
    const float maxV, const float minV,
    const int batch, const int ic, const int ic_p,
    const int oc, const int oc_p, const int num_qg
) {
    // Grid layout: (oc/OC_PER_BLK, batch)
    const int oc_base = blockIdx.x * OC_PER_BLK;
    const int b = blockIdx.y;
    if (oc_base >= oc || b >= batch) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int nwarps = blockDim.x / 32;

    const int ic_per_group = ic / num_qg;
    const int row_bytes = ic_p / 2;
    const T* in_ptr = input + b * ic_p;

    // Compute per-OC base pointers (on-the-fly, no pointer arrays to save registers)
    float accs[OC_PER_BLK];
    #pragma unroll
    for (int i = 0; i < OC_PER_BLK; i++)
        accs[i] = 0.0f;

    // Main IC loop — all OCs processed simultaneously
    const int stride = blockDim.x * 8;
    for (int k = tid * 8; k < ic; k += stride) {
        // Phase 1: Load ALL input values
        float in_vals[8];
        #pragma unroll
        for (int j = 0; j < 8; j++)
            in_vals[j] = (float)in_ptr[k + j];

        // Pre-compute input sum for factored adj_offset term (shared across all OCs)
        float input_sum = in_vals[0] + in_vals[1] + in_vals[2] + in_vals[3]
                        + in_vals[4] + in_vals[5] + in_vals[6] + in_vals[7];

        const int gidx = k / ic_per_group;

        // Phase 2: Load ALL weights and params before any compute
        uint32_t w_data[OC_PER_BLK];
        float s_data[OC_PER_BLK], adj_data[OC_PER_BLK];
        #pragma unroll
        for (int oi = 0; oi < OC_PER_BLK; oi++) {
            const int oc_idx = oc_base + oi;
            if (oc_idx < oc) {
                w_data[oi] = __ldg(reinterpret_cast<const uint32_t*>(
                    kernel + oc_idx * row_bytes + k / 2));
                const float2 p = __ldg(gemv_params + oc_idx * num_qg + gidx);
                s_data[oi] = p.x;
                adj_data[oi] = p.y;
            }
        }

        // Phase 3: Compute raw dot products (no scale/adj in the hot loop)
        #pragma unroll
        for (int oi = 0; oi < OC_PER_BLK; oi++) {
            if (oc_base + oi >= oc) break;
            float raw_dot = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t packed = (w_data[oi] >> (j * 8)) & 0xFF;
                raw_dot += in_vals[j * 2] * (float)(packed >> 4);
                raw_dot += in_vals[j * 2 + 1] * (float)(packed & 0x0F);
            }
            // Apply scale and adj_offset: acc += s * raw_dot + adj * input_sum
            accs[oi] = fmaf(adj_data[oi], input_sum, fmaf(s_data[oi], raw_dot, accs[oi]));
        }
    }

    // Warp-level reduction for all accumulators
    #pragma unroll
    for (int oi = 0; oi < OC_PER_BLK; oi++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            accs[oi] += __shfl_xor_sync(0xffffffff, accs[oi], off);
    }

    // Single-sync parallel cross-warp reduction for ALL OCs at once
    __shared__ float smem[OC_PER_BLK][8];  // [oc_idx][warp_idx]

    if (lane_id == 0) {
        #pragma unroll
        for (int oi = 0; oi < OC_PER_BLK; oi++)
            smem[oi][warp_id] = accs[oi];
    }
    __syncthreads();

    // First OC_PER_BLK threads each reduce one OC
    if (tid < OC_PER_BLK && oc_base + tid < oc) {
        float result = 0.0f;
        for (int w = 0; w < nwarps; w++) result += smem[tid][w];
        result += (float)__ldg(&bias[oc_base + tid]);
        result = fmaxf(result, minV);
        result = fminf(result, maxV);
        output[b * oc_p + oc_base + tid] = (T)result;
    }
}

// V14_MB: Multi-batch variant — one block processes ALL batch elements for the same OC group.
// Weights are loaded ONCE and reused across batch elements, reducing memory bandwidth by ~batch×.
template<typename T, int OC_PER_BLK, int MAX_BATCH>
__global__ void __launch_bounds__(128, 8) GEMV_FpAInt4B_V14_MB(
    const T* __restrict__ input,
    const uint8_t* __restrict__ kernel,
    const float2* __restrict__ gemv_params,
    const T* __restrict__ bias,
    T* __restrict__ output,
    const float maxV, const float minV,
    const int batch, const int ic, const int ic_p,
    const int oc, const int oc_p, const int num_qg
) {
    const int oc_base = blockIdx.x * OC_PER_BLK;
    if (oc_base >= oc) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int nwarps = blockDim.x / 32;

    const int ic_per_group = ic / num_qg;
    const int row_bytes = ic_p / 2;

    float accs[MAX_BATCH * OC_PER_BLK];
    #pragma unroll
    for (int i = 0; i < MAX_BATCH * OC_PER_BLK; i++)
        accs[i] = 0.0f;

    const int stride = blockDim.x * 8;
    for (int k = tid * 8; k < ic; k += stride) {
        const int gidx = k / ic_per_group;

        // Load weights ONCE for all batch elements
        uint32_t w_data[OC_PER_BLK];
        float s_data[OC_PER_BLK], adj_data[OC_PER_BLK];
        #pragma unroll
        for (int oi = 0; oi < OC_PER_BLK; oi++) {
            const int oc_idx = oc_base + oi;
            if (oc_idx < oc) {
                w_data[oi] = __ldg(reinterpret_cast<const uint32_t*>(
                    kernel + oc_idx * row_bytes + k / 2));
                const float2 p = __ldg(gemv_params + oc_idx * num_qg + gidx);
                s_data[oi] = p.x;
                adj_data[oi] = p.y;
            }
        }

        // Process each batch element with the cached weights
        #pragma unroll
        for (int b = 0; b < MAX_BATCH; b++) {
            if (b >= batch) break;
            const T* in_ptr = input + b * ic_p;
            float in_vals[8];
            #pragma unroll
            for (int j = 0; j < 8; j++)
                in_vals[j] = (float)in_ptr[k + j];

            float input_sum = in_vals[0] + in_vals[1] + in_vals[2] + in_vals[3]
                            + in_vals[4] + in_vals[5] + in_vals[6] + in_vals[7];

            #pragma unroll
            for (int oi = 0; oi < OC_PER_BLK; oi++) {
                if (oc_base + oi >= oc) break;
                float raw_dot = 0.0f;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t packed = (w_data[oi] >> (j * 8)) & 0xFF;
                    raw_dot += in_vals[j * 2] * (float)(packed >> 4);
                    raw_dot += in_vals[j * 2 + 1] * (float)(packed & 0x0F);
                }
                accs[b * OC_PER_BLK + oi] = fmaf(adj_data[oi], input_sum, fmaf(s_data[oi], raw_dot, accs[b * OC_PER_BLK + oi]));
            }
        }
    }

    // Warp-level reduction for all accumulators
    #pragma unroll
    for (int i = 0; i < MAX_BATCH * OC_PER_BLK; i++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            accs[i] += __shfl_xor_sync(0xffffffff, accs[i], off);
    }

    // Cross-warp reduction using shared memory
    __shared__ float smem[MAX_BATCH * OC_PER_BLK][4]; // [batch*oc][warp_id], nwarps<=4

    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < MAX_BATCH * OC_PER_BLK; i++)
            smem[i][warp_id] = accs[i];
    }
    __syncthreads();

    // First batch*OC_PER_BLK threads each reduce one accumulator
    const int reduce_count = batch * OC_PER_BLK;
    if (tid < reduce_count) {
        const int b = tid / OC_PER_BLK;
        const int oi = tid % OC_PER_BLK;
        if (oc_base + oi < oc) {
            float result = 0.0f;
            for (int w = 0; w < nwarps; w++) result += smem[tid][w];
            result += (float)__ldg(&bias[oc_base + oi]);
            result = fmaxf(result, minV);
            result = fminf(result, maxV);
            output[b * oc_p + oc_base + oi] = (T)result;
        }
    }
}










// =====================================================================
// V15: rows4/6/8 INT4-native weight-only GEMV (small-M).
//
// Motivation: under pic_decode_repair_sparse, int4GemvBatchLimit drops to 3,
// so x=3 (batch=4) leaves V14_MB and enters the cuBLAS/CUTLASS dense path that
// consumes dequantized FP16 weights — a 4x weight-bandwidth blowup on Jetson's
// unified memory. V15 keeps weights as packed int4 and reuses them across batch
// rows (like V14_MB), but moves the batch dimension out of registers into a
// shared-memory weight tile + micro-passes. This decouples OC_PER_BLK from the
// batch count so OC_PER_BLK=8 is reachable without register spills, and it
// avoids the FP16 dequant buffer entirely (fixes Qwen3-4B runtime_dequant).
//
// Math is identical to V14 factored dequant:
//   w_deq = scale*nibble + adj_offset  (adj_offset = offset - 8*scale, precomputed)
//   acc   = scale * sum(nibble*in) + adj_offset * sum(in)
//
// Grid: (ceil(oc/OC_PER_BLK)), Block: 128
// Batch coverage: batch in {4,6,8} via ceil(batch/B_TILE) micro-passes, B_TILE=2.
// batch 1/2/3 stay on V14/V14_MB; V15 is only dispatched for batch>=4.
// =====================================================================
template<typename T, int OC_PER_BLK>
__global__ void __launch_bounds__(128, 8) GEMV_FpAInt4B_V15_SmallM(
    const T* __restrict__ input,
    const uint8_t* __restrict__ kernel,     // packed int4 [oc, ic_p/2]
    const float2* __restrict__ gemv_params, // [oc * num_qg], .x=scale, .y=adj_offset
    const T* __restrict__ bias,
    T* __restrict__ output,
    const float maxV, const float minV,
    const int batch, const int ic, const int ic_p,
    const int oc, const int oc_p, const int num_qg
) {
    // V15 = V14_MB single-load model: weights are loaded ONCE per IC stride and
    // reused across ALL batch rows in the same iteration (batch loop is INNER).
    // This makes weight traffic independent of batch (read once, not num_batch
    // times). Register pressure = batch*OC_PER_BLK acc + OC_PER_BLK weight words.
    // OC_PER_BLK is kept at 4 so batch=8 -> 32 acc, comfortably under the spill
    // threshold that defeated naive V14_MB MAX_BATCH=8 retunes at OC_PER_BLK>=4.
    // Grid: (ceil(oc/OC_PER_BLK)), Block: 128. Dispatched only for batch in {4,6,8}.
    const int oc_base = blockIdx.x * OC_PER_BLK;
    if (oc_base >= oc) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int nwarps = blockDim.x / 32;  // 4

    const int ic_per_group = ic / num_qg;
    const int row_bytes = ic_p / 2;

    // Accumulators for ALL batch rows; weights loaded once per IC stride feed
    // every batch row's dot product in the inner loop.
    float accs[8 * OC_PER_BLK];  // max batch = 8
    #pragma unroll
    for (int b = 0; b < 8; b++) {
        #pragma unroll
        for (int oi = 0; oi < OC_PER_BLK; oi++)
            accs[b * OC_PER_BLK + oi] = 0.0f;
    }

    const int stride = blockDim.x * 8;  // 128 * 8 = 1024 nibbles per IC iteration

    for (int k = tid * 8; k < ic; k += stride) {
        const int gidx = k / ic_per_group;

        // Load OC_PER_BLK weight words (one uint32 = 8 nibbles per OC) into
        // registers ONCE — every thread loads its own k-offset, all 128 threads
        // issue __ldg in parallel (same as V14_MB).
        uint32_t w_data[OC_PER_BLK];
        float s_data[OC_PER_BLK], adj_data[OC_PER_BLK];
        #pragma unroll
        for (int oi = 0; oi < OC_PER_BLK; oi++) {
            const int oc_idx = oc_base + oi;
            if (oc_idx < oc) {
                w_data[oi] = __ldg(reinterpret_cast<const uint32_t*>(
                    kernel + oc_idx * row_bytes + k / 2));
                const float2 p = __ldg(gemv_params + oc_idx * num_qg + gidx);
                s_data[oi] = p.x;
                adj_data[oi] = p.y;
            }
        }

        // Inner batch loop: reuse the cached weights for every batch row.
        for (int b = 0; b < batch; b++) {
            const T* in_ptr = input + b * ic_p;
            float in_vals[8];
            #pragma unroll
            for (int j = 0; j < 8; j++)
                in_vals[j] = (k + j < ic) ? (float)in_ptr[k + j] : 0.0f;

            float input_sum = in_vals[0] + in_vals[1] + in_vals[2] + in_vals[3]
                            + in_vals[4] + in_vals[5] + in_vals[6] + in_vals[7];

            #pragma unroll
            for (int oi = 0; oi < OC_PER_BLK; oi++) {
                if (oc_base + oi >= oc) break;
                float raw_dot = 0.0f;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t packed = (w_data[oi] >> (j * 8)) & 0xFF;
                    raw_dot += in_vals[j * 2]     * (float)(packed >> 4);
                    raw_dot += in_vals[j * 2 + 1] * (float)(packed & 0x0F);
                }
                accs[b * OC_PER_BLK + oi] = fmaf(adj_data[oi], input_sum,
                    fmaf(s_data[oi], raw_dot, accs[b * OC_PER_BLK + oi]));
            }
        }
    }

    // Warp-level shuffle reduction for all (batch, OC) accumulators.
    const int acc_count = batch * OC_PER_BLK;
    #pragma unroll
    for (int b = 0; b < 8; b++) {
        if (b >= batch) break;
        #pragma unroll
        for (int oi = 0; oi < OC_PER_BLK; oi++) {
            float v = accs[b * OC_PER_BLK + oi];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffff, v, off);
            accs[b * OC_PER_BLK + oi] = v;
        }
    }

    // Cross-warp reduction via shared memory.
    __shared__ float s_red[8 * OC_PER_BLK][4];  // [b*oc][warp_id], nwarps<=4
    if (lane_id == 0) {
        for (int i = 0; i < acc_count; i++)
            s_red[i][warp_id] = accs[i];
    }
    __syncthreads();

    // First acc_count threads each finalize one (batch, OC) output.
    if (tid < acc_count) {
        const int b = tid / OC_PER_BLK;
        const int oi = tid % OC_PER_BLK;
        if (b < batch && oc_base + oi < oc) {
            float result = 0.0f;
            for (int w = 0; w < nwarps; w++) result += s_red[tid][w];
            result += (float)__ldg(&bias[oc_base + oi]);
            result = fmaxf(result, minV);
            result = fminf(result, maxV);
            output[b * oc_p + oc_base + oi] = (T)result;
        }
    }
}

// =====================================================================
// V16: INT8 __dp4a small-M weight-only route (env-gated, opt-in).
//
// Hypothesis under test: on sm_72 (Volta), int4 factored-GEMV (V14/V15) is
// instruction-bound (per-nibble shift/mask/int2float/FMA), losing to static-
// dequant cuBLAS at batch>=4. __dp4a does 4 int8*int8->int32 MACs in one instr,
// which may move the batch>=4 small-M op off the nibble-ALU bottleneck.
//
// Factored-dp4a math (NO weight re-quant loss; nibble stays exact 0..15):
//   w_deq = s_w * nibble + adj_w        (adj_w = offset - 8*s_w, from mGemvParams)
//   in_fp = s_a * a_q + o_a            (a_q int8, per-row asymmetric runtime quant)
//   acc   = sum(w_deq * in_fp)
//         = s_w*s_a * dp4a(nibble,a_q)
//         + s_w*o_a * sum_nibble[oc,grp]      (precomputed static)
//         + adj_w*s_a * sum_aq[row]           (per-row, from runtime A-quant)
//         + adj_w*o_a * K_grp
// Combine per group: partial = s_a*(s_w*dp4a + adj_w*sum_aq) + o_a*(s_w*sum_nibble + adj_w*K)
// Grid: (ceil(oc/OC_PER_BLK), num_qg) — one block per (oc-chunk, quant-group).
// =====================================================================

// Precompute sum_nibble[oc * num_qg] = sum of nibble values per (oc, group).
__global__ void PrecomputeSumNibble(
    const uint8_t* __restrict__ packed_weight,  // [oc, ic_p/2]
    float* __restrict__ sum_nibble,             // [oc * num_qg]
    const int oc, const int ic, const int ic_p, const int num_qg)
{
    const int ic_per_group = ic / num_qg;
    const int row_bytes = ic_p / 2;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < (size_t)oc * num_qg; index += (size_t)blockDim.x * gridDim.x) {
        const int g = index % num_qg;
        const int o = index / num_qg;
        const int kstart = g * ic_per_group;
        int s = 0;
        const uint8_t* wrow = packed_weight + (size_t)o * row_bytes;
        for (int k = kstart; k < kstart + ic_per_group; k++) {
            uint8_t b = wrow[k / 2];
            s += (k & 1) ? (b & 0xF) : ((b >> 4) & 0xF);
        }
        sum_nibble[index] = (float)s;
    }
}

// Runtime per-row A quantization: fp16 [batch, ic] -> int8 [batch, ic] + s_a/o_a/sum_aq.
// One block per row, 128 threads.
template<typename T>
__global__ void QuantARowInt8(
    const T* __restrict__ A_fp, int8_t* __restrict__ A_q,
    float* __restrict__ s_a, float* __restrict__ o_a, int32_t* __restrict__ sum_aq,
    const int batch, const int ic)
{
    const int row = blockIdx.x;
    if (row >= batch) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const T* rowptr = A_fp + (size_t)row * ic;
    int8_t* qptr = A_q + (size_t)row * ic;

    float mymin = FLT_MAX, mymax = -FLT_MAX;
    for (int k = tid; k < ic; k += 128) {
        float v = (float)rowptr[k];
        mymin = fminf(mymin, v);
        mymax = fmaxf(mymax, v);
    }
    for (int off = 16; off > 0; off >>= 1) {
        mymin = fminf(mymin, __shfl_xor_sync(0xffffffff, mymin, off));
        mymax = fmaxf(mymax, __shfl_xor_sync(0xffffffff, mymax, off));
    }
    __shared__ float smm[4];
    if (lane == 0) smm[warp] = mymin;
    __syncthreads();
    if (lane == 0) {
        float mn = smm[0], mx = smm[0];
        #pragma unroll
        for (int i = 1; i < 4; i++) { mn = fminf(mn, smm[i]); mx = fmaxf(mx, smm[i]); }
        smm[0] = mn; smm[1] = mx;
    }
    __syncthreads();
    float mn = smm[0], mx = smm[1];
    float scale = (mx - mn) / 255.0f;
    float offset = (fabsf(scale) > 1e-5f) ? (mx - scale * 127.0f) : (scale = 1.0f, mx);
    if (tid == 0) { s_a[row] = scale; o_a[row] = offset; }

    int32_t mysum = 0;
    for (int k = tid; k < ic; k += 128) {
        float v = (float)rowptr[k];
        int32_t q = (int32_t)rintf((v - offset) / scale);
        q = max(-128, min(127, q));
        qptr[k] = (int8_t)q;
        mysum += q;
    }
    for (int off = 16; off > 0; off >>= 1)
        mysum += __shfl_xor_sync(0xffffffff, mysum, off);
    __shared__ int32_t ssum[4];
    if (lane == 0) ssum[warp] = mysum;
    __syncthreads();
    if (tid == 0) {
        int32_t s = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) s += ssum[i];
        sum_aq[row] = s;
    }
}

// dp4a small-M GEMV. One block per OC_PER_BLK chunk; loops over ALL quant groups
// IN-BLOCK (no group grid dim) to avoid block-count explosion. Per (oc, group) it
// dp4a-accumulates into a small int32 temp, applies that group's s_w/adj_w combine,
// and adds into a running float total[OC_PER_BLK * batch]. Single store at the end
// (no atomicAdd). Grid: (ceil(oc/OC_PER_BLK)), Block: 128.
template<int OC_PER_BLK>
__global__ void __launch_bounds__(128, 8) GEMV_FpAInt4B_V16_Dp4a(
    const int8_t* __restrict__ A_q,            // [batch, ic]
    const uint8_t* __restrict__ kernel,        // packed int4 [oc, ic_p/2]
    const float2* __restrict__ gemv_params,    // [oc * num_qg] {s_w, adj_w}
    const float* __restrict__ sum_nibble,      // [oc * num_qg]
    const float* __restrict__ s_a,             // [batch]
    const float* __restrict__ o_a,             // [batch]
    const int32_t* __restrict__ sum_aq,        // [batch]
    half* __restrict__ output,                 // [batch, oc_p]
    const half* __restrict__ bias,             // [oc]
    const int batch, const int ic, const int ic_p,
    const int oc, const int oc_p, const int num_qg,
    const int ic_per_group)
{
    const int oc_base = blockIdx.x * OC_PER_BLK;
    if (oc_base >= oc) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = 4;

    const int row_bytes = ic_p / 2;

    // Running float totals (one per batch row x OC).
    float totals[8 * OC_PER_BLK];
    #pragma unroll
    for (int i = 0; i < 8 * OC_PER_BLK; i++) totals[i] = 0.0f;

    // Load per-row A quant params once.
    float sa[8], oa[8], saq[8];
    #pragma unroll
    for (int b = 0; b < 8; b++) {
        if (b < batch) {
            sa[b] = s_a[b]; oa[b] = o_a[b]; saq[b] = (float)sum_aq[b];
        }
    }

    const int stride = blockDim.x * 8;
    // Loop over quant groups in-block.
    for (int g = 0; g < num_qg; g++) {
        const int k_base = g * ic_per_group;
        int32_t accs[8 * OC_PER_BLK];
        #pragma unroll
        for (int i = 0; i < 8 * OC_PER_BLK; i++) accs[i] = 0;

        for (int k = k_base + tid * 8; k < k_base + ic_per_group; k += stride) {
            char4 w0[OC_PER_BLK], w1[OC_PER_BLK];
            #pragma unroll
            for (int oi = 0; oi < OC_PER_BLK; oi++) {
                const int oc_idx = oc_base + oi;
                if (oc_idx < oc) {
                    const uint32_t* src = reinterpret_cast<const uint32_t*>(
                        kernel + (size_t)oc_idx * row_bytes + k / 2);
                    uint32_t pk = __ldg(src);
                    w0[oi].x = (int8_t)((pk >> 4) & 0xF);
                    w0[oi].y = (int8_t)(pk & 0xF);
                    w0[oi].z = (int8_t)((pk >> 12) & 0xF);
                    w0[oi].w = (int8_t)((pk >> 8) & 0xF);
                    w1[oi].x = (int8_t)((pk >> 20) & 0xF);
                    w1[oi].y = (int8_t)((pk >> 16) & 0xF);
                    w1[oi].z = (int8_t)((pk >> 28) & 0xF);
                    w1[oi].w = (int8_t)((pk >> 24) & 0xF);
                }
            }
            for (int b = 0; b < batch; b++) {
                const uint32_t* asrc = reinterpret_cast<const uint32_t*>(A_q + (size_t)b * ic + k);
                uint32_t apk = __ldg(asrc);
                char4 a0;
                a0.x = (int8_t)(apk & 0xFF);
                a0.y = (int8_t)((apk >> 8) & 0xFF);
                a0.z = (int8_t)((apk >> 16) & 0xFF);
                a0.w = (int8_t)((apk >> 24) & 0xFF);
                uint32_t apk2 = __ldg(asrc + 1);
                char4 a1;
                a1.x = (int8_t)(apk2 & 0xFF);
                a1.y = (int8_t)((apk2 >> 8) & 0xFF);
                a1.z = (int8_t)((apk2 >> 16) & 0xFF);
                a1.w = (int8_t)((apk2 >> 24) & 0xFF);
                #pragma unroll
                for (int oi = 0; oi < OC_PER_BLK; oi++) {
                    if (oc_base + oi >= oc) break;
                    int32_t d = 0;
                    d = __dp4a(w0[oi], a0, d);
                    d = __dp4a(w1[oi], a1, d);
                    accs[b * OC_PER_BLK + oi] += d;
                }
            }
        }

        // Reduce this group's dp4a accs, combine with per-group s_w/adj_w/sum_nibble,
        // add into running totals.
        const int acc_count = batch * OC_PER_BLK;
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            if (b >= batch) break;
            #pragma unroll
            for (int oi = 0; oi < OC_PER_BLK; oi++) {
                int32_t v = accs[b * OC_PER_BLK + oi];
                for (int off = 16; off > 0; off >>= 1)
                    v += __shfl_xor_sync(0xffffffff, v, off);
                const int oc_idx = oc_base + oi;
                const int pidx = oc_idx * num_qg + g;
                const float sw = gemv_params[pidx].x;
                const float aw = gemv_params[pidx].y;
                const float sn = sum_nibble[pidx];
                float partial = sa[b] * (sw * (float)v + aw * saq[b])
                              + oa[b] * (sw * sn + aw * (float)ic_per_group);
                totals[b * OC_PER_BLK + oi] += partial;
            }
        }
    }

    // Cross-warp reduce totals (each lane has its own partial; shuffle-sum).
    #pragma unroll
    for (int i = 0; i < 8 * OC_PER_BLK; i++) {
        if (i < batch * OC_PER_BLK) {
            float v = totals[i];
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffff, v, off);
            totals[i] = v;
        }
    }
    __shared__ float s_red[8 * OC_PER_BLK][4];
    if (lane == 0) {
        for (int i = 0; i < batch * OC_PER_BLK; i++) s_red[i][warp] = totals[i];
    }
    __syncthreads();

    const int acc_count = batch * OC_PER_BLK;
    if (tid < acc_count) {
        const int b = tid / OC_PER_BLK;
        const int oi = tid % OC_PER_BLK;
        if (b < batch && oc_base + oi < oc) {
            float result = 0.0f;
            for (int w = 0; w < nwarps; w++) result += s_red[tid][w];
            result += (float)__ldg(&bias[oc_base + oi]);
            output[(size_t)b * oc_p + (size_t)oc_base + oi] = (half)result;
        }
    }
}


// P1: Optimized INT8 GEMV with warp shuffle reduction
template<typename T>
__global__ void GEMV_FpAInt8B_V2(
    const T* __restrict__ input,
    const int8_t* __restrict__ kernel,
    const T* __restrict__ scale, const T* __restrict__ offset, const T* __restrict__ bias,
    T* __restrict__ output,
    const float maxV, const float minV,
    const int batch, const int ic, const int ic_p,
    const int oc, const int oc_p, const int quanC
) {
    const int b = blockIdx.y;
    const int oz_base = blockIdx.x * GEMV_OC_PER_BLOCK;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int oz = oz_base + warp_id;

    if (oz >= oc || b >= batch) return;

    const int num_quan_groups_per_channel = (quanC > 0) ? (quanC / oc) : 1;
    const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;
    const int quan_param_index_base = oz * num_quan_groups_per_channel;

    float my_sum = 0.0f;
    for (int k = lane_id; k < ic; k += WARP_SIZE) {
        const float inp0 = (float)input[b * ic_p + k];
        const int8_t ker0 = __ldg(&kernel[oz * ic_p + k]);

        const int group_idx = k / ic_per_group;
        const int quan_param_index = quan_param_index_base + group_idx;
        const float x_scale = (float)__ldg(&scale[quan_param_index]);
        const float x_offset = (float)__ldg(&offset[quan_param_index]);

        my_sum += inp0 * ((float)ker0 * x_scale + x_offset);
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int offset_val = WARP_SIZE / 2; offset_val > 0; offset_val >>= 1) {
        my_sum += __shfl_xor_sync(0xffffffff, my_sum, offset_val);
    }

    if (lane_id == 0) {
        float final_val = my_sum + (float)bias[oz];
        final_val = max(final_val, minV);
        final_val = min(final_val, maxV);
        output[b * oc_p + oz] = (T)final_val;
    }
}

template<typename T>
__global__ void CONV_FpAInt8B(const T* input,
    const int8_t* kernel,
    const T* scale, const T* offset, const T* bias,
    T *output,
    const float maxV, const float minV,
    const int ic,  const int ic_p, const int iw, const int ih,
    const int c, const int c_p, const int ow, const int oh,
    const int kw, const int kh, const int dw, const int dh,
    const int sw, const int sh, const int pw, const int ph,
    const int total, const int quanC,
    DivModFast d_oc, DivModFast d_ow, DivModFast d_oh
) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_2);
        d_ow.divmod(tmp1, tmp2, ox);
        d_oh.divmod(tmp2, ob, oy);

        int oz = oz_2;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        float color0 = bias[oz];
        
        const int num_quan_groups_per_channel = (c > 0 && quanC > 0) ? (quanC / c) : 1;
        const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;
        const int quan_param_index_base = oz * num_quan_groups_per_channel;

        int fxSta = max(0, (UP_DIV(-ix, dw)));
        int fySta = max(0, (UP_DIV(-iy, dh)));
        int fxEnd = min(kw, UP_DIV(iw - ix, dw));
        int fyEnd = min(kh, UP_DIV(ih - iy, dh));
        int fx, fy, fz;
        for (fy=fySta; fy<fyEnd; ++fy) {
            int sy = fy*dh + iy;
            for (fx=fxSta; fx<fxEnd; ++fx) {
                int sx = fx*dw + ix;
                for (int group_idx = 0; group_idx < num_quan_groups_per_channel; ++group_idx) {
                    const int quan_param_index = quan_param_index_base + group_idx;
                    const float x_scale = scale[quan_param_index];
                    const float x_offset = offset[quan_param_index];

                    const int sz_start = group_idx * ic_per_group;
                    const int sz_end = sz_start + ic_per_group;

                    for (int sz = sz_start; sz < sz_end && sz < ic_p; ++ sz) {
                        int src_offset = ((ob * ih + sy) * iw + sx) * ic_p + sz;
                        float inp0 = input[src_offset];
                        //[Cop, KhKw, Cip]
                        int8_t ker0 = kernel[((oz * kh + fy) * kw + fx) * ic_p + sz];

                        color0 = color0 + inp0 * ((float)ker0 * x_scale + x_offset);
                    }
                }
            }
        }
        color0 = max(color0, minV);
        color0 = min(color0, maxV);

        int dst_offset = ((ob * oh + oy) * ow + ox) * c_p + oz;

        output[dst_offset] = color0;
    }
}

template<typename T>
__global__ void CONV_FpAInt4B(const T* input,
    const uint8_t* kernel,
    const T* scale, const T* offset, const T* bias,
    T *output,
    const float maxV, const float minV,
    const int ic, const int ic_p, const int iw, const int ih,
    const int c, const int c_p, const int ow, const int oh,
    const int kw, const int kh, const int dw, const int dh,
    const int sw, const int sh, const int pw, const int ph,
    const int total, const int quanC,
    DivModFast d_oc, DivModFast d_ow, DivModFast d_oh
) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_2);
        d_ow.divmod(tmp1, tmp2, ox);
        d_oh.divmod(tmp2, ob, oy);

        int oz = oz_2;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        float color0 = bias[oz];

        const int num_quan_groups_per_channel = (c > 0 && quanC > 0) ? (quanC / c) : 1;
        const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;
        const int quan_param_index_base = oz * num_quan_groups_per_channel;

        int fxSta = max(0, (UP_DIV(-ix, dw)));
        int fySta = max(0, (UP_DIV(-iy, dh)));
        int fxEnd = min(kw, UP_DIV(iw - ix, dw));
        int fyEnd = min(kh, UP_DIV(ih - iy, dh));
        int fx, fy, fz;
        for (fy=fySta; fy<fyEnd; ++fy) {
            int sy = fy*dh + iy;
            for (fx=fxSta; fx<fxEnd; ++fx) {
                int sx = fx*dw + ix;
                for (int group_idx = 0; group_idx < num_quan_groups_per_channel; ++group_idx) {
                    const int quan_param_index = quan_param_index_base + group_idx;
                    const float x_scale = scale[quan_param_index];
                    const float x_offset = offset[quan_param_index];

                    const int sz_start = group_idx * ic_per_group / 2;
                    const int sz_end = sz_start + ic_per_group / 2;

                    for (int sz = sz_start; sz < sz_end && sz * 2 < ic_p; ++ sz) {
                        int src_offset = ((ob * ih + sy) * iw + sx) * ic_p + 2 * sz;
                        float inp0 = input[src_offset];
                        float inp1 = input[src_offset+1];

                        //[Cop, KhKw, Cip]
                        uint8_t ker = kernel[((oz * kh + fy) * kw + fx) * ic_p / 2 + sz];
                        int8_t ker0 = (ker >> 4) - 8;
                        int8_t ker1 = (ker & 15) - 8;
                        color0 = color0 + inp0 * ((float)ker0 * x_scale + x_offset);
                        color0 = color0 + inp1 * ((float)ker1 * x_scale + x_offset);
                    }
                }
            }
        }
        color0 = max(color0, minV);
        color0 = min(color0, maxV);

        int dst_offset = ((ob * oh + oy) * ow + ox) * c_p + oz;

        output[dst_offset] = color0;
    }
}

__global__ void Rearrange_Packed_Weight_Int4(const uint8_t* param,
    uint8_t* output,
    const int khw,
    const size_t maxCount,
    const int oc,
    const int ic,
    DivModFast d_khw,
    DivModFast d_icp2
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int icp2Index, temp, ocpIndex, khwIndex;
        d_icp2.divmod(index, temp, icp2Index);
        d_khw.divmod(temp, ocpIndex, khwIndex);
        if(ocpIndex >= oc || 2 * icp2Index >= ic) {
            output[index] = 0;
            continue;
        }

        int src_index = (ocpIndex * UP_DIV(ic, 2) + icp2Index) * khw + khwIndex;
        output[index] = param[src_index];
    }
}

__global__ void Rearrange_Weight_Int4(const int8_t* param,
    uint8_t* output,
    const int khw,
    const size_t maxCount,
    const int oc,
    const int ic,
    DivModFast d_khw,
    DivModFast d_icp2
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int icp2Index, temp, ocpIndex, khwIndex;
        d_icp2.divmod(index, temp, icp2Index);
        d_khw.divmod(temp, ocpIndex, khwIndex);
        if(2*icp2Index >= ic || ocpIndex >= oc) {
            output[index] = 0;
            continue;
        }
        // [Co, Ci, KhKw] -> [Cop, KhKw, Cip/2], Ci available for vectorize
        output[index] = ((uint8_t)(param[(ocpIndex * ic + 2*icp2Index+0) * khw + khwIndex] + 8 )) * 16;
        if(2*icp2Index+1 < ic) {
            output[index] += ((uint8_t)(param[(ocpIndex * ic + 2*icp2Index+1) * khw + khwIndex] + 8 ));
        }
    }
}

__global__ void Rearrange_Weight_Int8(const int8_t* param,
    int8_t* output,
    const int khw,
    const size_t maxCount,
    const int oc,
    const int ic,
    DivModFast d_khw,
    DivModFast d_icp
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int icpIndex, temp, ocpIndex, khwIndex;
        d_icp.divmod(index, temp, icpIndex);
        d_khw.divmod(temp, ocpIndex, khwIndex);
        if(icpIndex >= ic || ocpIndex >= oc) {
            output[index] = 0;
            continue;
        }
        // [Co, Ci, KhKw] -> [Cop, KhKw, Cip], Ci available for vectorize
        output[index] = param[(ocpIndex * ic + icpIndex) * khw + khwIndex];
    }
}

bool ConvFpAIntBExecution::isValid(const Convolution2D* conv, Backend* backend) {
    return true;
}


ConvFpAIntBExecution::Resource::Resource(Backend* bn, const MNN::Op* op) {
    mBackend = bn;
    auto runtime = static_cast<CUDABackend*>(bn)->getCUDARuntime();

    auto conv       = op->main_as_Convolution2D();
    auto common     = conv->common();

    //weight host->device
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon = ConvolutionCommon::load(op, mBackend, false, true);

    auto oc = common->outputCount();
    if (quanCommon->asymmetric) {
        mQuanC = quanCommon->alpha.size() / 2;
    } else {
        mQuanC = quanCommon->alpha.size();
    }
    auto weightSize = quanCommon->weight.size();
    int l = weightSize / oc;
    int h = oc;
    int ic = common->inputCount();
    if(ic == 0) {
        ic = l / common->kernelX() / common->kernelY();
    }

    int lp = UP_DIV(l, 8) * 8;
    int hp = UP_DIV(h, 8) * 8;
    int quanHp = UP_DIV(mQuanC, 8) * 8;

    // set dequant scale/offset
    {
        float * dequantAlpha = quanCommon->alpha.get();
        std::vector<float> dequantScale(quanHp, 0.0);
        std::vector<float> dequantOffset(quanHp, 0.0);

        for (int o = 0; o < mQuanC; o++) {
            float min = 0.0f;
            float alpha = 0.0f;
            if (quanCommon->asymmetric) {
                min = dequantAlpha[2*o];
                alpha = dequantAlpha[2*o+1];
            } else {
                alpha = dequantAlpha[o];
            }
            dequantScale[o] = alpha;
            dequantOffset[o] = min;
        }

        if(static_cast<CUDABackend*>(bn)->useFp16()) {
            auto tempScaleStorage = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(quanHp*sizeof(float));
            auto scaleTemp = (float*)((uint8_t*)tempScaleStorage.first + tempScaleStorage.second);
            cuda_check(cudaMemcpy(scaleTemp, dequantScale.data(), quanHp*sizeof(float), cudaMemcpyHostToDevice));

            scaleTensor.reset(Tensor::createDevice<int16_t>({quanHp}));
            bn->onAcquireBuffer(scaleTensor.get(), Backend::STATIC);
            mScale = (void *)scaleTensor.get()->buffer().device;
            callFloat2Half((const void*)scaleTemp, (void*)mScale, quanHp, runtime);

            // Reuse scaleTemp buffer
            cuda_check(cudaMemcpy(scaleTemp, dequantOffset.data(), quanHp*sizeof(float), cudaMemcpyHostToDevice));

            offsetTensor.reset(Tensor::createDevice<int16_t>({quanHp}));
            bn->onAcquireBuffer(offsetTensor.get(), Backend::STATIC);
            mOffset = (void *)offsetTensor.get()->buffer().device;
            callFloat2Half((const void*)scaleTemp, (void*)mOffset, quanHp, runtime);

            static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempScaleStorage);
        } else {
            scaleTensor.reset(Tensor::createDevice<int32_t>({quanHp}));
            bn->onAcquireBuffer(scaleTensor.get(), Backend::STATIC);
            mScale = (void *)scaleTensor.get()->buffer().device;
            cuda_check(cudaMemcpy(mScale, dequantScale.data(), quanHp*sizeof(float), cudaMemcpyHostToDevice));
            
            offsetTensor.reset(Tensor::createDevice<int32_t>({quanHp}));
            bn->onAcquireBuffer(offsetTensor.get(), Backend::STATIC);
            mOffset = (void *)offsetTensor.get()->buffer().device;
            cuda_check(cudaMemcpy(mOffset, dequantOffset.data(), quanHp*sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    // Reorder weight
    {
        int khw = common->kernelX() * common->kernelY();
        int icp = UP_DIV(ic, 8) * 8;
        DivModFast khwD(khw);
        DivModFast icp2D(icp/2);
        DivModFast icpD(icp);

        if(quanCommon->canUseInt4) {
            mIsWeightInt4 = true;

            auto tempCacheBuffer = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(weightSize * sizeof(int8_t));
            float* cacheWeight = (float*)((uint8_t*)tempCacheBuffer.first + tempCacheBuffer.second);
            runtime->memcpy(cacheWeight, quanCommon->weight.get(), weightSize * sizeof(int8_t), MNNMemcpyHostToDevice);

            weightTensor.reset(Tensor::createDevice<uint8_t>({khw * icp/2 * hp}));
            bn->onAcquireBuffer(weightTensor.get(), Backend::STATIC);
            mFilter = (void *)weightTensor.get()->buffer().device;

            //[Co, Ci, KhKw] -> [Cop, KhKw, Cip/2]
            int block_num = runtime->blocks_num(khw*icp/2*hp);
            int block_size = runtime->threads_num();
            Rearrange_Packed_Weight_Int4<<<block_num, block_size>>>((const uint8_t*)cacheWeight, (uint8_t*)mFilter, khw, khw*icp/2*hp, oc, ic, khwD, icp2D);
            checkKernelErrors;

            static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempCacheBuffer);

            // R6: Pre-compute GEMV params (scale, adj_offset) as float2 array for 1x1 conv
            if (khw == 1) {
                mNumQg = (mQuanC > 0) ? (mQuanC / oc) : 1;
                const size_t param_count = (size_t)oc * mNumQg;
                cudaError_t err = cudaMalloc(&mGemvParams, param_count * sizeof(float2));
                if (err == cudaSuccess) {
                    dim3 pg_block(256);
                    dim3 pg_grid((param_count + 255) / 256);
                    if (static_cast<CUDABackend*>(bn)->useFp16()) {
                        PrecomputeGemvParams<half><<<pg_grid, pg_block>>>(
                            (const half*)mScale, (const half*)mOffset,
                            mGemvParams, (int)param_count);
                    } else {
                        PrecomputeGemvParams<float><<<pg_grid, pg_block>>>(
                            (const float*)mScale, (const float*)mOffset,
                            mGemvParams, (int)param_count);
                    }
                    checkKernelErrors;
                }
                // V16: precompute sum_nibble[oc * num_qg] for the INT8 dp4a route.
                // Cheap static table; only consumed when MNN_CUDA_PIC_INT4_SMALLM_DP4A is set.
                if (mGemvParams != nullptr) {
                    cudaError_t err2 = cudaMalloc(&mSumNibble, param_count * sizeof(float));
                    if (err2 == cudaSuccess) {
                        dim3 sn_block(256);
                        dim3 sn_grid((param_count + 255) / 256);
                        PrecomputeSumNibble<<<sn_grid, sn_block>>>(
                            (const uint8_t*)mFilter, mSumNibble,
                            oc, ic, icp, mNumQg);
                        checkKernelErrors;
                    }
                }
            }
            const int backendPrecision = static_cast<CUDABackend*>(bn)->getPrecision();
            const bool cutlassLowPrecision = backendPrecision == 2 || backendPrecision == 0;
            if (khw == 1 && cutlassLowPrecision) {
                const size_t dequantBytes = static_cast<size_t>(hp) * static_cast<size_t>(icp) * sizeof(uint16_t);
                const int minLinearDim = std::max(1, std::min(hp, icp));
                const int maxLinearDim = std::max(hp, icp);
                const bool mlpLikeLargeLinear = dequantBytes >= 16ull * 1024ull * 1024ull &&
                    maxLinearDim >= 2 * minLinearDim && maxLinearDim <= 8 * minLinearDim;
                const bool attentionProjectionLikeLinear = dequantBytes >= 6ull * 1024ull * 1024ull &&
                    dequantBytes <= 64ull * 1024ull * 1024ull &&
                    minLinearDim >= 512 && maxLinearDim <= 4096;
                const bool decodeHotLinear = mlpLikeLargeLinear || attentionProjectionLikeLinear;
                const size_t cacheLimit = decodeHotLinear
                    ? picStaticDequantDecodeHotLimitBytes(runtime->prop())
                    : picStaticDequantBaseLimitBytes(runtime->prop());
                const std::string staticDequantCacheKey =
                    picStaticDequantCacheKey(bn, op, hp, icp, dequantBytes);
                auto cachedStaticDequantTensor = findPicStaticDequantCache(staticDequantCacheKey);
                if (cachedStaticDequantTensor) {
                    staticDequantWeightTensor = cachedStaticDequantTensor;
                    mStaticDequantFilter = reinterpret_cast<void*>(staticDequantWeightTensor->buffer().device);
                    mStaticDequantBytes = dequantBytes;
                    mOwnsStaticDequantBytes = false;
                } else if (reservePicStaticDequantBytes(dequantBytes, cacheLimit)) {
                    staticDequantWeightTensor.reset(Tensor::createDevice<int16_t>({hp, icp}));
                    if (staticDequantWeightTensor &&
                        bn->onAcquireBuffer(staticDequantWeightTensor.get(), Backend::STATIC)) {
                        mStaticDequantFilter = reinterpret_cast<void*>(staticDequantWeightTensor->buffer().device);
                        mStaticDequantBytes = dequantBytes;
                        mOwnsStaticDequantBytes = true;
                        int threads_per_row = UP_DIV(icp, 16);
                        dim3 dq_block(std::min(threads_per_row, 256));
                        dim3 dq_grid(oc, UP_DIV(threads_per_row, static_cast<int>(dq_block.x)));
                        if (backendPrecision == 2) {
                            DequantizeInt4Weight<half, half><<<dq_grid, dq_block>>>(
                                reinterpret_cast<const uint8_t*>(mFilter),
                                reinterpret_cast<half*>(mStaticDequantFilter),
                                reinterpret_cast<const half*>(mScale),
                                reinterpret_cast<const half*>(mOffset),
                                oc, ic, icp, mQuanC);
                        } else {
                            DequantizeInt4Weight<float, half><<<dq_grid, dq_block>>>(
                                reinterpret_cast<const uint8_t*>(mFilter),
                                reinterpret_cast<half*>(mStaticDequantFilter),
                                reinterpret_cast<const float*>(mScale),
                                reinterpret_cast<const float*>(mOffset),
                                oc, ic, icp, mQuanC);
                        }
                        checkKernelErrors;
                        insertPicStaticDequantCache(staticDequantCacheKey, staticDequantWeightTensor);
                    } else {
                        staticDequantWeightTensor.reset();
                        gPicStaticDequantBytes.fetch_sub(dequantBytes, std::memory_order_relaxed);
                        mOwnsStaticDequantBytes = false;
                    }
                }
            }
        } else {
            auto tempCacheBuffer = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(weightSize * sizeof(int8_t));
            float* cacheWeight = (float*)((uint8_t*)tempCacheBuffer.first + tempCacheBuffer.second);
            runtime->memcpy(cacheWeight, quanCommon->weight.get(), weightSize * sizeof(int8_t), MNNMemcpyHostToDevice);

            weightTensor.reset(Tensor::createDevice<int8_t>({khw * icp * hp}));
            bn->onAcquireBuffer(weightTensor.get(), Backend::STATIC);
            mFilter = (void *)weightTensor.get()->buffer().device;

            //[Co, Ci, KhKw] -> [Cop, KhKw, Cip]
            int block_num = runtime->blocks_num(khw*icp*hp);
            int block_size = runtime->threads_num();
            Rearrange_Weight_Int8<<<block_num, block_size>>>((const int8_t*)cacheWeight, (int8_t*)mFilter, khw, khw*icp*hp, oc, ic, khwD, icpD);
            checkKernelErrors;
            static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempCacheBuffer);
        }
    }

    // Copy Bias
    {
        if(static_cast<CUDABackend*>(bn)->useFp16()) {
            int biasSize = conv->bias()->size();
            int hp = UP_DIV(biasSize, 8) * 8;

            auto tempBiasStorage = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(hp*sizeof(float));
            auto biasTemp = (float*)((uint8_t*)tempBiasStorage.first + tempBiasStorage.second);
            runtime->memset(biasTemp, 0, hp * sizeof(int32_t));
            cuda_check(cudaMemcpy(biasTemp, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));

            biasTensor.reset(Tensor::createDevice<int16_t>({hp}));
            bn->onAcquireBuffer(biasTensor.get(), Backend::STATIC);
            mBias = (void *)biasTensor.get()->buffer().device;
            callFloat2Half((const void*)biasTemp, (void*)mBias, hp, runtime);

            static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempBiasStorage);
        } else {
            int biasSize = conv->bias()->size();
            int hp = UP_DIV(biasSize, 8) * 8;
            biasTensor.reset(Tensor::createDevice<int32_t>({hp}));
            bn->onAcquireBuffer(biasTensor.get(), Backend::STATIC);
            mBias = (void *)biasTensor.get()->buffer().device;
            runtime->memset(mBias, 0, hp * sizeof(int32_t));
            cuda_check(cudaMemcpy(mBias, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    // High + Int8：预计算 B 的修正项 sum_B_q
    // if(!quanCommon->canUseInt4) {
    //     const int icp = UP_DIV(ic, 8) * 8;
    //     const int num_groups = (mQuanC > 0) ? (mQuanC / oc) : 1;
    //     const int ic_per_group = (num_groups > 0) ? (ic / num_groups) : ic;

    //     const int param_count = oc * num_groups;
    //     mSumBQTensor.reset(Tensor::createDevice<int32_t>({param_count}));
    //     bn->onAcquireBuffer(mSumBQTensor.get(), Backend::STATIC);
    //     mSumBQ = (void*)mSumBQTensor.get()->buffer().device;

    //     const int block_size = BLOCK_SIZE;
    //     const int num_blocks = (param_count + block_size - 1) / block_size;

    //     Precompute_SumBq<<<num_blocks, block_size>>>((const int8_t*)mFilter, (int32_t*)mSumBQ, num_groups, ic_per_group, oc, icp);
    //     checkKernelErrors;
    // }
}

ConvFpAIntBExecution::Resource::~Resource() {
    if (mGemvParams) cudaFree(mGemvParams);
    if (mSumNibble) cudaFree(mSumNibble);
    if (mOwnsStaticDequantBytes && mStaticDequantBytes > 0) {
        gPicStaticDequantBytes.fetch_sub(mStaticDequantBytes, std::memory_order_relaxed);
        mStaticDequantBytes = 0;
        mOwnsStaticDequantBytes = false;
    }
}
ConvFpAIntBExecution::ConvFpAIntBExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res) : CutlassConvCommonExecution(backend) {
    mOp = op;
    mResource = res;
    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime();
    mPrecisonLevel = static_cast<CUDABackend*>(backend)->getPrecision();
    mFp16Infer = (mPrecisonLevel == 2);
    mFp32Infer = (mPrecisonLevel == 1);
    mFp16Fp32MixInfer = (mPrecisonLevel == 0);
    mBf16Infer = (mPrecisonLevel == 3);
}

ConvFpAIntBExecution::~ConvFpAIntBExecution() {
    // DYNAMIC dequant buffers are managed by the memory planner (alloc+release in onResize)
    // Only STATIC buffers need explicit release in destructor
    if (mDequantFilterTensor && mDequantIsStatic) {
        backend()->onReleaseBuffer(mDequantFilterTensor.get(), Backend::STATIC);
    }
    destroyPicRowsCublasLtPlan(mPicRowsCublasLtPlan);
}
bool ConvFpAIntBExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new ConvFpAIntBExecution(bn, op, mResource);
    *dst = dstExe;
    return true;
}


ErrorCode ConvFpAIntBExecution::onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto input = inputs[0], output = outputs[0];
    const int UNIT = PACK_NUMBER;
    auto convCommon = mOp->main_as_Convolution2D()->common();
    auto pads = ConvolutionCommon::convolutionPadFull(input, output, mOp->main_as_Convolution2D()->common());
    int ic = input->channel();
    const int icp = UP_DIV(ic, 8) * 8;
    auto icDiv = UP_DIV(ic, UNIT);

    mIm2ColParamter.dilateX         = convCommon->dilateX();
    mIm2ColParamter.dilateY         = convCommon->dilateY();
    mIm2ColParamter.strideX         = convCommon->strideX();
    mIm2ColParamter.strideY         = convCommon->strideY();
    mIm2ColParamter.icDiv4          = icDiv;
    mIm2ColParamter.ic              = ic;
    mIm2ColParamter.kernelX         = convCommon->kernelX();
    mIm2ColParamter.kernelY         = convCommon->kernelY();
    mIm2ColParamter.padX = std::get<0>(pads);
    mIm2ColParamter.padY = std::get<1>(pads);

    mIm2ColParamter.ih = input->height();
    mIm2ColParamter.iw = input->width();
    mIm2ColParamter.oh = output->height();
    mIm2ColParamter.ow = output->width();
    mIm2ColParamter.srcZStep = input->height() * input->width() * UNIT * input->batch();
    mIm2ColParamter.srcYStep = input->width() * UNIT;
    mIm2ColParamter.packCUnit = UNIT;

    mActivationType = convCommon->relu() ? 1 : convCommon->relu6() ? 2 : 0;

    //MNN_PRINT("conv size:%d-%d, %d-%d-%d, %d-%d-%d\n", mIm2ColParamter.kernelX, mIm2ColParamter.strideX, input->height(), input->width(), input->channel(), output->height(), output->width(), output->channel());
    int e = output->height() * output->width() * output->batch();
    int l = ic * mIm2ColParamter.kernelX * mIm2ColParamter.kernelY;
    int h = output->channel();
    mGemmInfo.elh[0] = e;
    mGemmInfo.elh[1] = l;
    mGemmInfo.elh[2] = h;
    mGemmInfo.elhPad[0] = UP_DIV(e, 8) * 8;
    mGemmInfo.elhPad[1] = UP_DIV(l, 8) * 8;
    mGemmInfo.elhPad[2] = UP_DIV(h, 8) * 8;
    const int oc = mGemmInfo.elh[2];
    const int ocp = mGemmInfo.elhPad[2];

    mIsConv1x1S1D1P0 = (mIm2ColParamter.kernelX == 1 && mIm2ColParamter.kernelY == 1 &&
        mIm2ColParamter.strideX == 1 && mIm2ColParamter.strideY == 1 &&
        mIm2ColParamter.dilateX == 1 && mIm2ColParamter.dilateY == 1 &&
        mIm2ColParamter.padX == 0 && mIm2ColParamter.padY == 0);

    mNeedIm2Col = !(mIsConv1x1S1D1P0 && (mFp16Infer || mFp32Infer));

    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
    if(mNeedIm2Col) {
        size_t im2colBytes = 2;
        // Only when fp32 Im2Col convert to fp32, Fp16Fp32Mix Im2Col convert to fp16
        if(mFp32Infer) {
            im2colBytes = 4;
        }
        auto buffer = pool->alloc(im2colBytes * (size_t)mGemmInfo.elh[0] * (size_t)mGemmInfo.elhPad[1]);
        mIm2ColBuffer = (void*)((uint8_t*)buffer.first + buffer.second);
        pool->free(buffer);
    }

    if (mDequantFilterTensor) {
        if (mDequantIsStatic) {
            backend()->onReleaseBuffer(mDequantFilterTensor.get(), Backend::STATIC);
        }
        // DYNAMIC buffers were already released in previous onResize, no action needed
        mDequantFilterTensor = nullptr;
    }
    mDequantFilter = nullptr;
    mNeedRuntimeDequant = false;
    mDequantIsStatic = false;

    // Weight dequantization: use DYNAMIC buffer with runtime dequant in onExecute.
    // DYNAMIC buffers are shared across all Conv layers (only one active at a time),
    // saving massive GPU memory vs STATIC (which would need ~30GB for 27B models).
    if (!mFp32Infer) {
        if (mIsConv1x1S1D1P0) {
            std::vector<int> dequantShape = {mGemmInfo.elhPad[2], mGemmInfo.elhPad[1]}; // Shape: [oc_p, ic_p]
            const size_t dequantBytes = static_cast<size_t>(mGemmInfo.elhPad[2]) *
                static_cast<size_t>(mGemmInfo.elhPad[1]) * sizeof(uint16_t);
            const bool useStaticDequant = (mFp16Infer || mFp16Fp32MixInfer) &&
                mResource->mStaticDequantFilter != nullptr &&
                mResource->mStaticDequantBytes >= dequantBytes;
            if (useStaticDequant) {
                mDequantFilter = mResource->mStaticDequantFilter;
                mNeedRuntimeDequant = false;
                mDequantIsStatic = false;
            } else {
                if (mFp16Infer || mFp16Fp32MixInfer) {
                    mDequantFilterTensor.reset(Tensor::createDevice<int16_t>(dequantShape));
                } else {
                    mDequantFilterTensor.reset(Tensor::createDevice<float>(dequantShape));
                }
                backend()->onAcquireBuffer(mDequantFilterTensor.get(), Backend::DYNAMIC);
                backend()->onReleaseBuffer(mDequantFilterTensor.get(), Backend::DYNAMIC);
                mNeedRuntimeDequant = true;
                mDequantIsStatic = false;
                mDequantFilter = (void*)mDequantFilterTensor->buffer().device;
            }

            mBiasAddr   = mResource->mBias;
            mBackendPtr = mResource->mBackend;
            mFilterAddr = mDequantFilter;

            if (mFp32Infer) {
                return callCutlassGemmCudaCoreFloat32(inputs, outputs);
            }
            mGpuComputeCap = static_cast<CUDABackend*>(backend())->getCUDARuntime()->compute_capability();
            if (mGpuComputeCap < 70) {
                return callCutlassGemmCudaCoreFloat16(inputs, outputs);
            } else if (mGpuComputeCap < 75) {
                const bool usePicCompactSm70Linear = mResource->mIsWeightInt4 && mFp16Infer && mActivationType == 0 &&
                    mIsConv1x1S1D1P0 && useStaticDequant && mGemmInfo.elh[0] >= 384 &&
                    mGemmInfo.elh[0] <= 768 && mGemmInfo.elhPad[1] >= 1024;
                if (usePicCompactSm70Linear) {
                    return callCutlassGemmTensorCore884PicCompact(inputs, outputs);
                }
                return callCutlassGemmTensorCore884(inputs, outputs);
            }
            return callCutlassGemmTensorCore(inputs, outputs);
        }
    }

    return NO_ERROR;
}

ErrorCode ConvFpAIntBExecution::onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];

    //printf("convcutlass:%p %p\n", input->deviceId(), output->deviceId());
    //MNN_PRINT("cutlass hw:%d-%d\n", input->height(), input->width());
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    const void *input_addr = (const void*)inputs[0]->deviceId();
    const void *filter_addr = mResource->mFilter;
    const void *bias_addr = mResource->mBias;
    auto bn = backend();
    void *output_addr = (void*)outputs[0]->deviceId();

    const int sw = mIm2ColParamter.strideX;
    const int sh = mIm2ColParamter.strideY;
    const int kw = mIm2ColParamter.kernelX;
    const int kh = mIm2ColParamter.kernelY;
    const int dw = mIm2ColParamter.dilateX;
    const int dh = mIm2ColParamter.dilateY;
    const int pw = mIm2ColParamter.padX;
    const int ph = mIm2ColParamter.padY;
    const int ic = mIm2ColParamter.ic;
    const int icp = UP_DIV(ic, 8) * 8;
    const int iw = mIm2ColParamter.iw;
    const int ih = mIm2ColParamter.ih;

    const int oc = mGemmInfo.elh[2];
    const int ocp = mGemmInfo.elhPad[2];
    const int ow = mIm2ColParamter.ow;
    const int oh = mIm2ColParamter.oh;

    float maxV = FLT_MAX;
    float minV = -FLT_MAX;
    if (mActivationType == 1) {
        minV = 0.0f;
    }
    if (mActivationType == 2) {
        minV = 0.0f;
        maxV = 6.0f;
    }

    auto total = outputs[0]->batch() * oh * ow * ocp;
    auto& prop = runtime->prop();
    int limitThreads = UP_DIV(total, prop.multiProcessorCount);
    int threadNum = ALIMIN(prop.maxThreadsPerBlock/2, limitThreads);
    int blockNum = prop.multiProcessorCount;

    DivModFast d_oc(ocp);
    DivModFast d_ow(ow);
    DivModFast d_oh(oh);

    const int batch = inputs[0]->batch();
    auto meta = static_cast<KVMeta*>(static_cast<CUDABackend*>(backend())->getMetaPtr());
    const bool picDecodeRepairSparse = meta != nullptr && meta->pic_decode_repair_sparse_active;
    int int4GemvBatchLimit = picDecodeRepairSparse ? 3 : 6;

    const bool profileConv = profilePicWeightOnlyConv() && mIsConv1x1S1D1P0 &&
        (picDecodeRepairSparse || batch > 4) &&
        (mFp16Infer || mFp16Fp32MixInfer);
    const bool staticDequant = mResource->mStaticDequantFilter != nullptr &&
        mDequantFilter == mResource->mStaticDequantFilter && !mNeedRuntimeDequant;

    if(mResource->mIsWeightInt4) {
        if (mIsConv1x1S1D1P0) {
            if (mFp32Infer) { // High + Int4
                if (batch <= 16) {
                    // P1-V5: 128-thread GEMV, 1 OC per block
                    dim3 v5_grid(oc, batch);
                    dim3 v5_block(128);
                    GEMV_FpAInt4B_V5<<<v5_grid, v5_block>>>(
                        (const float*)input_addr, (const uint8_t*)mResource->mFilter,
                        (const float*)mResource->mScale, (const float*)mResource->mOffset,
                        (const float*)bias_addr, (float*)output_addr,
                        maxV, minV, batch, ic, icp, oc, ocp, mResource->mQuanC);
                } else {
                    dim3 threads(TILE_DIM, TILE_DIM);
                    dim3 blocks(UP_DIV(ocp, TILE_DIM), UP_DIV(batch, TILE_DIM));

                    GEMM_FpAInt4B<<<blocks, threads>>>(
                        (const float*)input_addr, (const uint8_t*)mResource->mFilter,
                        (const float*)mResource->mScale,  (const float*)mResource->mOffset,
                        (const float*)bias_addr, (float*)output_addr,
                        maxV, minV, ic, icp, oc, ocp, batch, mResource->mQuanC
                    );
                }
            } else {
                if (batch <= int4GemvBatchLimit) {
                    uint64_t gemvStartUs = 0;
                    if (profileConv) {
                        cudaDeviceSynchronize();
                        gemvStartUs = convProfileNowUs();
                    }
                    const int num_qg = (mResource->mQuanC > 0) ? (mResource->mQuanC / oc) : 1;
                    if (mResource->mGemvParams) {
                        constexpr int V14_OC = 4;
                        if (batch > 1) {
                            // V14_MB: One block processes all tiny-M rows for the same OC group.
                            // PIC decode repair rows >=4 are faster on the existing CUTLASS/static
                            // dequant path, so only non-decode tiny-M keeps the batch<=6 GEMV policy.
                            constexpr int V14_TINY6_OC = 3;
                            const int ocPerBlock = batch <= 4 ? V14_OC :
                                V14_TINY6_OC;
                            dim3 mb_grid((oc + ocPerBlock - 1) / ocPerBlock);
                            dim3 mb_block(128);
                            if (batch <= 2 && mFp16Infer) {
                                GEMV_FpAInt4B_V14_MB<half, V14_OC, 2><<<mb_grid, mb_block>>>(
                                    (const half*)input_addr, (const uint8_t*)mResource->mFilter,
                                    mResource->mGemvParams,
                                    (const half*)bias_addr, (half*)output_addr,
                                    maxV, minV, batch, ic, icp, oc, ocp, num_qg);
                            } else if (batch <= 2 && mFp16Fp32MixInfer) {
                                GEMV_FpAInt4B_V14_MB<float, V14_OC, 2><<<mb_grid, mb_block>>>(
                                    (const float*)input_addr, (const uint8_t*)mResource->mFilter,
                                    mResource->mGemvParams,
                                    (const float*)bias_addr, (float*)output_addr,
                                    maxV, minV, batch, ic, icp, oc, ocp, num_qg);
                            } else if (batch <= 3 && mFp16Infer) {
                                GEMV_FpAInt4B_V14_MB<half, V14_OC, 3><<<mb_grid, mb_block>>>(
                                    (const half*)input_addr, (const uint8_t*)mResource->mFilter,
                                    mResource->mGemvParams,
                                    (const half*)bias_addr, (half*)output_addr,
                                    maxV, minV, batch, ic, icp, oc, ocp, num_qg);
                            } else if (batch <= 3 && mFp16Fp32MixInfer) {
                                GEMV_FpAInt4B_V14_MB<float, V14_OC, 3><<<mb_grid, mb_block>>>(
                                    (const float*)input_addr, (const uint8_t*)mResource->mFilter,
                                    mResource->mGemvParams,
                                    (const float*)bias_addr, (float*)output_addr,
                                    maxV, minV, batch, ic, icp, oc, ocp, num_qg);
                            } else if (batch <= 4 && mFp16Infer) {
                                GEMV_FpAInt4B_V14_MB<half, V14_OC, 4><<<mb_grid, mb_block>>>(
                                    (const half*)input_addr, (const uint8_t*)mResource->mFilter,
                                    mResource->mGemvParams,
                                    (const half*)bias_addr, (half*)output_addr,
                                    maxV, minV, batch, ic, icp, oc, ocp, num_qg);
                            } else if (batch <= 4 && mFp16Fp32MixInfer) {
                                GEMV_FpAInt4B_V14_MB<float, V14_OC, 4><<<mb_grid, mb_block>>>(
                                    (const float*)input_addr, (const uint8_t*)mResource->mFilter,
                                    mResource->mGemvParams,
                                    (const float*)bias_addr, (float*)output_addr,
                                    maxV, minV, batch, ic, icp, oc, ocp, num_qg);
                            } else if (mFp16Infer) {
                                GEMV_FpAInt4B_V14_MB<half, V14_TINY6_OC, 6><<<mb_grid, mb_block>>>(
                                    (const half*)input_addr, (const uint8_t*)mResource->mFilter,
                                    mResource->mGemvParams,
                                    (const half*)bias_addr, (half*)output_addr,
                                    maxV, minV, batch, ic, icp, oc, ocp, num_qg);
                            } else if (mFp16Fp32MixInfer) {
                                GEMV_FpAInt4B_V14_MB<float, V14_TINY6_OC, 6><<<mb_grid, mb_block>>>(
                                    (const float*)input_addr, (const uint8_t*)mResource->mFilter,
                                    mResource->mGemvParams,
                                    (const float*)bias_addr, (float*)output_addr,
                                    maxV, minV, batch, ic, icp, oc, ocp, num_qg);
                            }
                        } else {
                            // batch=1: original V14
                            dim3 v14_block(128);
                            if (batch == 1 && oc >= 32768 && mFp16Infer) {
                                constexpr int V14_VOCAB_OC = 2;
                                dim3 v14_grid((oc + V14_VOCAB_OC - 1) / V14_VOCAB_OC, batch);
                                GEMV_FpAInt4B_V14<half, V14_VOCAB_OC><<<v14_grid, v14_block>>>(
                                    (const half*)input_addr, (const uint8_t*)mResource->mFilter,
                                    mResource->mGemvParams,
                                    (const half*)bias_addr, (half*)output_addr,
                                    maxV, minV, batch, ic, icp, oc, ocp, num_qg);
                            } else if (batch == 1 && oc >= 32768 && mFp16Fp32MixInfer) {
                                constexpr int V14_VOCAB_OC = 2;
                                dim3 v14_grid((oc + V14_VOCAB_OC - 1) / V14_VOCAB_OC, batch);
                                GEMV_FpAInt4B_V14<float, V14_VOCAB_OC><<<v14_grid, v14_block>>>(
                                    (const float*)input_addr, (const uint8_t*)mResource->mFilter,
                                    mResource->mGemvParams,
                                    (const float*)bias_addr, (float*)output_addr,
                                    maxV, minV, batch, ic, icp, oc, ocp, num_qg);
                            } else if (mFp16Infer) {
                                dim3 v14_grid((oc + V14_OC - 1) / V14_OC, batch);
                                GEMV_FpAInt4B_V14<half, V14_OC><<<v14_grid, v14_block>>>(
                                    (const half*)input_addr, (const uint8_t*)mResource->mFilter,
                                    mResource->mGemvParams,
                                    (const half*)bias_addr, (half*)output_addr,
                                    maxV, minV, batch, ic, icp, oc, ocp, num_qg);
                            } else if (mFp16Fp32MixInfer) {
                                dim3 v14_grid((oc + V14_OC - 1) / V14_OC, batch);
                                GEMV_FpAInt4B_V14<float, V14_OC><<<v14_grid, v14_block>>>(
                                    (const float*)input_addr, (const uint8_t*)mResource->mFilter,
                                    mResource->mGemvParams,
                                    (const float*)bias_addr, (float*)output_addr,
                                    maxV, minV, batch, ic, icp, oc, ocp, num_qg);
                            }
                        }
                    } else {
                        // Fallback to V9/V5
                        dim3 v5_grid(oc, batch);
                        dim3 v5_block(128);
                        if (mFp16Infer) {
                            constexpr int OC_PER_BLK = 2;
                            dim3 v9_grid((oc + OC_PER_BLK - 1) / OC_PER_BLK, batch);
                            GEMV_FpAInt4B_V9<half, OC_PER_BLK><<<v9_grid, v5_block>>>(
                                (const half*)input_addr, (const uint8_t*)mResource->mFilter,
                                (const half*)mResource->mScale, (const half*)mResource->mOffset,
                                (const half*)bias_addr, (half*)output_addr,
                                maxV, minV, batch, ic, icp, oc, ocp, mResource->mQuanC);
                        } else if (mFp16Fp32MixInfer) {
                            GEMV_FpAInt4B_V5<<<v5_grid, v5_block>>>(
                                (const float*)input_addr, (const uint8_t*)mResource->mFilter,
                                (const float*)mResource->mScale, (const float*)mResource->mOffset,
                                (const float*)bias_addr, (float*)output_addr,
                                maxV, minV, batch, ic, icp, oc, ocp, mResource->mQuanC);
                        }
                    }
                    if (profileConv) {
                        cudaDeviceSynchronize();
                        MNN_PRINT("CUDAWeightOnlyConv profile op=conv_fpa_intb_1x1_tiny_gemv batch=%d "
                                  "ic=%d oc=%d icp=%d ocp=%d int4=%d gemv_params=%d "
                                  "pic_decode_repair=%d fp16=%d mix=%d gemv_us=%llu\n",
                                  batch, ic, oc, icp, ocp, mResource->mIsWeightInt4 ? 1 : 0,
                                  mResource->mGemvParams ? 1 : 0, picDecodeRepairSparse ? 1 : 0,
                                  mFp16Infer ? 1 : 0, mFp16Fp32MixInfer ? 1 : 0,
                                  static_cast<unsigned long long>(convProfileNowUs() - gemvStartUs));
                    }
                } else {
                    // === V16 INT8 __dp4a small-M route (env-gated, opt-in) ===
                    // Independent of V15. Reuses mGemvParams + precomputed mSumNibble.
                    // Quantizes A to int8 per row, dp4a over int4 nibbles (kept exact),
                    // factored asymmetric combine. Miss -> falls through to V15 / cuBLAS.
                    const int v16Dp4aPolicy = picV16Dp4aPolicy();
                    const bool useV16Dp4a = v16Dp4aPolicy > 0 && picDecodeRepairSparse &&
                        mFp16Infer &&
                        mResource->mGemvParams != nullptr && mResource->mSumNibble != nullptr &&
                        mActivationType == 0 &&
                        picV15SmallMMatches(v16Dp4aPolicy, batch, ic, oc) &&  // same shape families
                        (ic % 8 == 0);  // dp4a kernel assumes ic aligned to 8
                    if (useV16Dp4a) {
                        const int num_qg = mResource->mNumQg;
                        const int ic_per_group = (num_qg > 0) ? (ic / num_qg) : ic;
                        const int ocPerBlock = (oc >= 2048) ? 8 : 4;
                        // Runtime A-quant scratch (small: batch x ic int8 + 3*batch float/int32).
                        auto exePool = static_cast<CUDABackend*>(backend())->getBufferPool();
                        auto aqBuf = exePool->alloc((size_t)batch * ic * sizeof(int8_t));
                        auto saBuf = exePool->alloc((size_t)batch * sizeof(float));
                        auto oaBuf = exePool->alloc((size_t)batch * sizeof(float));
                        auto sumaqBuf = exePool->alloc((size_t)batch * sizeof(int32_t));
                        int8_t* aq = (int8_t*)((uint8_t*)aqBuf.first + aqBuf.second);
                        float* sa = (float*)((uint8_t*)saBuf.first + saBuf.second);
                        float* oa = (float*)((uint8_t*)oaBuf.first + oaBuf.second);
                        int32_t* sumaq = (int32_t*)((uint8_t*)sumaqBuf.first + sumaqBuf.second);
                        exePool->free(aqBuf);
                        exePool->free(saBuf);
                        exePool->free(oaBuf);
                        exePool->free(sumaqBuf);

                        uint64_t v16StartUs = 0;
                        if (profileConv) {
                            cudaDeviceSynchronize();
                            v16StartUs = convProfileNowUs();
                        }
                        // Per-row A quantization.
                        QuantARowInt8<half><<<batch, 128>>>(
                            (const half*)input_addr, aq, sa, oa, sumaq, batch, ic);
                        // dp4a: one block per OC chunk, groups looped in-block, single store.
                        dim3 v16_grid((oc + ocPerBlock - 1) / ocPerBlock);
                        if (ocPerBlock == 8) {
                            GEMV_FpAInt4B_V16_Dp4a<8><<<v16_grid, 128>>>(
                                aq, (const uint8_t*)mResource->mFilter, mResource->mGemvParams,
                                mResource->mSumNibble, sa, oa, sumaq,
                                (__half*)output_addr, (const half*)bias_addr,
                                batch, ic, icp, oc, ocp, num_qg, ic_per_group);
                        } else {
                            GEMV_FpAInt4B_V16_Dp4a<4><<<v16_grid, 128>>>(
                                aq, (const uint8_t*)mResource->mFilter, mResource->mGemvParams,
                                mResource->mSumNibble, sa, oa, sumaq,
                                (__half*)output_addr, (const half*)bias_addr,
                                batch, ic, icp, oc, ocp, num_qg, ic_per_group);
                        }
                        checkKernelErrors;
                        if (profileConv) {
                            cudaDeviceSynchronize();
                            MNN_PRINT("CUDAWeightOnlyConv profile op=conv_fpa_intb_1x1_smallm_v16_dp4a "
                                      "batch=%d ic=%d oc=%d icp=%d ocp=%d int4=%d oc_per_blk=%d "
                                      "num_qg=%d v16_us=%llu\n",
                                      batch, ic, oc, icp, ocp, mResource->mIsWeightInt4 ? 1 : 0,
                                      ocPerBlock, num_qg,
                                      static_cast<unsigned long long>(convProfileNowUs() - v16StartUs));
                        }
                    } else {
                    // === V15 rows4/6/8 INT4-native small-M route (env-gated, opt-in) ===
                    // Reads packed int4 weights + precomputed gemv_params directly; does
                    // NOT consume the FP16 dequant buffer, so it bypasses runtime_dequant
                    // (fixes Qwen3-4B batch-4). Miss -> falls through to existing routes.
                    const int v15SmallMPolicy = picV15SmallMPolicy();
                    const bool useV15SmallM = v15SmallMPolicy > 0 && picDecodeRepairSparse &&
                        (mFp16Infer || mFp16Fp32MixInfer) &&
                        mResource->mGemvParams != nullptr &&
                        mActivationType == 0 &&
                        picV15SmallMMatches(v15SmallMPolicy, batch, ic, oc);
                    if (useV15SmallM) {
                        const int num_qg = (mResource->mQuanC > 0) ? (mResource->mQuanC / oc) : 1;
                        // OC_PER_BLK by output-channel size: 8 for large oc, 4 for small.
                        const int ocPerBlock = (oc >= 2048) ? 8 : 4;
                        uint64_t v15StartUs = 0;
                        if (profileConv) {
                            cudaDeviceSynchronize();
                            v15StartUs = convProfileNowUs();
                        }
                        if (mFp16Infer) {
                            if (ocPerBlock == 8) {
                                dim3 grid((oc + 7) / 8);
                                GEMV_FpAInt4B_V15_SmallM<half, 8><<<grid, 128>>>(
                                    (const half*)input_addr, (const uint8_t*)mResource->mFilter,
                                    mResource->mGemvParams, (const half*)bias_addr, (half*)output_addr,
                                    maxV, minV, batch, ic, icp, oc, ocp, num_qg);
                            } else {
                                dim3 grid((oc + 3) / 4);
                                GEMV_FpAInt4B_V15_SmallM<half, 4><<<grid, 128>>>(
                                    (const half*)input_addr, (const uint8_t*)mResource->mFilter,
                                    mResource->mGemvParams, (const half*)bias_addr, (half*)output_addr,
                                    maxV, minV, batch, ic, icp, oc, ocp, num_qg);
                            }
                        } else if (mFp16Fp32MixInfer) {
                            if (ocPerBlock == 8) {
                                dim3 grid((oc + 7) / 8);
                                GEMV_FpAInt4B_V15_SmallM<float, 8><<<grid, 128>>>(
                                    (const float*)input_addr, (const uint8_t*)mResource->mFilter,
                                    mResource->mGemvParams, (const float*)bias_addr, (float*)output_addr,
                                    maxV, minV, batch, ic, icp, oc, ocp, num_qg);
                            } else {
                                dim3 grid((oc + 3) / 4);
                                GEMV_FpAInt4B_V15_SmallM<float, 4><<<grid, 128>>>(
                                    (const float*)input_addr, (const uint8_t*)mResource->mFilter,
                                    mResource->mGemvParams, (const float*)bias_addr, (float*)output_addr,
                                    maxV, minV, batch, ic, icp, oc, ocp, num_qg);
                            }
                        }
                        checkKernelErrors;
                        if (profileConv) {
                            cudaDeviceSynchronize();
                            MNN_PRINT("CUDAWeightOnlyConv profile op=conv_fpa_intb_1x1_smallm_v15 batch=%d "
                                      "ic=%d oc=%d icp=%d ocp=%d int4=%d gemv_params=%d "
                                      "oc_per_blk=%d pic_decode_repair=%d fp16=%d mix=%d v15_us=%llu\n",
                                      batch, ic, oc, icp, ocp, mResource->mIsWeightInt4 ? 1 : 0,
                                      mResource->mGemvParams ? 1 : 0, ocPerBlock,
                                      picDecodeRepairSparse ? 1 : 0, mFp16Infer ? 1 : 0,
                                      mFp16Fp32MixInfer ? 1 : 0,
                                      static_cast<unsigned long long>(convProfileNowUs() - v15StartUs));
                        }
                    } else {
                    const int rows45CublasPolicy = picRows45CublasPolicy();
                    const bool useRows45Cublas = picDecodeRepairSparse && mFp16Infer &&
                        staticDequant && !mNeedRuntimeDequant &&
                        picRows45CublasMatches(rows45CublasPolicy, batch, ic, oc);
#if MNN_CUDA_HAS_CUBLASLT
                    const bool useRows48CublasLt = picDecodeRepairSparse && mFp16Infer &&
                        staticDequant && !mNeedRuntimeDequant &&
                        picRows48CublasLtMatches(batch, ic, oc) &&
                        runtime->cublasLtHandle() != nullptr;
                    bool rows48CublasLtDone = false;
                    if (useRows48CublasLt) {
                        constexpr size_t kRows48CublasLtWorkspaceBytes = 4ull * 1024ull * 1024ull;
                        uint64_t cublasLtStartUs = 0;
                        if (profileConv) {
                            cudaDeviceSynchronize();
                            cublasLtStartUs = convProfileNowUs();
                        }
                        void* workspace = runtime->cublasLtWorkspace(kRows48CublasLtWorkspaceBytes);
                        auto plan = ensurePicRowsCublasLtPlan(mPicRowsCublasLtPlan, runtime->cublasLtHandle(),
                                                              batch, ic, icp, oc, ocp,
                                                              kRows48CublasLtWorkspaceBytes);
                        rows48CublasLtDone = workspace != nullptr && plan != nullptr &&
                            plan->run(runtime->cublasLtHandle(), input_addr, mDequantFilter, output_addr,
                                      workspace, kRows48CublasLtWorkspaceBytes);
                        if (!rows48CublasLtDone) {
                            destroyPicRowsCublasLtPlan(mPicRowsCublasLtPlan);
                        }
                        if (profileConv) {
                            cudaDeviceSynchronize();
                            MNN_PRINT("CUDAWeightOnlyConv profile op=conv_fpa_intb_1x1_rows48_cublaslt "
                                      "batch=%d ic=%d oc=%d icp=%d ocp=%d policy=%d "
                                      "static_cache_bytes=%zu static_cache_total=%zu cublaslt_us=%llu\n",
                                      batch, ic, oc, icp, ocp, picRows48CublasLtPolicy(),
                                      mResource->mStaticDequantBytes,
                                      gPicStaticDequantBytes.load(std::memory_order_relaxed),
                                      static_cast<unsigned long long>(convProfileNowUs() - cublasLtStartUs));
                        }
                    }
                    if (!rows48CublasLtDone) {
#endif
                    if (useRows45Cublas) {
                        uint64_t cublasStartUs = 0;
                        if (profileConv) {
                            cudaDeviceSynchronize();
                            cublasStartUs = convProfileNowUs();
                        }
                        auto handle = runtime->cublasHandle();
                        int rows45CublasMathPolicy = 0;
                        int rows45CublasComputePolicy = 0;
                        int rows45CublasAlgoPolicy = 0;
                        const bool cublasOk = runRows45CublasFp16(
                            handle, input_addr, mDequantFilter, output_addr,
                            batch, ic, icp, oc, ocp,
                            &rows45CublasMathPolicy,
                            &rows45CublasComputePolicy,
                            &rows45CublasAlgoPolicy);
                        if (!cublasOk) {
                            return INVALID_VALUE;
                        }
                        if (profileConv) {
                            cudaDeviceSynchronize();
                            MNN_PRINT("CUDAWeightOnlyConv profile op=conv_fpa_intb_1x1_rows45_cublas "
                                      "batch=%d ic=%d oc=%d icp=%d ocp=%d policy=%d math=%d compute=%d algo=%d "
                                      "static_cache_bytes=%zu static_cache_total=%zu cublas_us=%llu\n",
                                      batch, ic, oc, icp, ocp, rows45CublasPolicy,
                                      rows45CublasMathPolicy, rows45CublasComputePolicy, rows45CublasAlgoPolicy,
                                      mResource->mStaticDequantBytes,
                                      gPicStaticDequantBytes.load(std::memory_order_relaxed),
                                      static_cast<unsigned long long>(convProfileNowUs() - cublasStartUs));
                        }
                    } else {
                        // Prefill/batched GEMM path: CUTLASS with dequantized FP16 weights
                        uint64_t totalStartUs = 0;
                        uint64_t dequantUs = 0;
                        uint64_t convertUs = 0;
                        uint64_t gemmUs = 0;
                        if (profileConv) {
                            cudaDeviceSynchronize();
                            totalStartUs = convProfileNowUs();
                        }
                        if (mNeedRuntimeDequant) {
                            // Runtime dequantization: dequant INT4->FP16 into DYNAMIC buffer before CUTLASS
                            uint64_t dequantStartUs = 0;
                            if (profileConv) {
                                dequantStartUs = convProfileNowUs();
                            }
                            if (mResource->mIsWeightInt4) {
                                int threads_per_row = UP_DIV(icp, 16);
                                dim3 dq_block(min(threads_per_row, 256));
                                dim3 dq_grid(oc, UP_DIV(threads_per_row, (int)dq_block.x));
                                if (mFp16Infer) {
                                    DequantizeInt4Weight<half, half><<<dq_grid, dq_block>>>(
                                        (const uint8_t*)mResource->mFilter, (half*)mDequantFilter,
                                        (const half*)mResource->mScale, (const half*)mResource->mOffset,
                                        oc, ic, icp, mResource->mQuanC);
                                } else if (mFp16Fp32MixInfer) {
                                    DequantizeInt4Weight<float, half><<<dq_grid, dq_block>>>(
                                        (const uint8_t*)mResource->mFilter, (half*)mDequantFilter,
                                        (const float*)mResource->mScale, (const float*)mResource->mOffset,
                                        oc, ic, icp, mResource->mQuanC);
                                }
                            } else {
                                dim3 threads(32, 32);
                                dim3 blocks(UP_DIV(ic, threads.x), UP_DIV(oc, threads.y));
                                if (mFp16Infer) {
                                    DequantizeInt8Weight<half, half><<<blocks, threads>>>(
                                        (const int8_t*)mResource->mFilter, (half*)mDequantFilter,
                                        (const half*)mResource->mScale, (const half*)mResource->mOffset,
                                        oc, ic, icp, mResource->mQuanC);
                                } else if (mFp16Fp32MixInfer) {
                                    DequantizeInt8Weight<float, half><<<blocks, threads>>>(
                                        (const int8_t*)mResource->mFilter, (half*)mDequantFilter,
                                        (const float*)mResource->mScale, (const float*)mResource->mOffset,
                                        oc, ic, icp, mResource->mQuanC);
                                }
                            }
                            if (profileConv) {
                                cudaDeviceSynchronize();
                                dequantUs = convProfileNowUs() - dequantStartUs;
                            }
                        }
                        if (mFp16Fp32MixInfer) {
                            size_t maxCount = mGemmInfo.elh[0] * mGemmInfo.elhPad[1];
                            uint64_t convertStartUs = 0;
                            if (profileConv) {
                                convertStartUs = convProfileNowUs();
                            }
                            callFloat2Half(input_addr, mIm2ColBuffer, maxCount, runtime);
                            if (profileConv) {
                                cudaDeviceSynchronize();
                                convertUs = convProfileNowUs() - convertStartUs;
                            }
                        }
                        uint64_t gemmStartUs = 0;
                        if (profileConv) {
                            gemmStartUs = convProfileNowUs();
                        }
                        runCutlassGemmFunc();
                        if (profileConv) {
                            cudaDeviceSynchronize();
                            gemmUs = convProfileNowUs() - gemmStartUs;
                            MNN_PRINT("CUDAWeightOnlyConv profile op=conv_fpa_intb_1x1 batch=%d ic=%d oc=%d "
                                      "icp=%d ocp=%d int4=%d runtime_dequant=%d fp16=%d mix=%d "
                                      "static_dequant=%d pic_compact_sm70=%d pic_compact_tile=%d "
                                      "static_cache_bytes=%zu static_cache_total=%zu "
                                      "dequant_us=%llu convert_us=%llu cutlass_us=%llu total_us=%llu\n",
                                      batch, ic, oc, icp, ocp, mResource->mIsWeightInt4 ? 1 : 0,
                                      mNeedRuntimeDequant ? 1 : 0, mFp16Infer ? 1 : 0,
                                      mFp16Fp32MixInfer ? 1 : 0, staticDequant ? 1 : 0,
                                      mUsePicCompactSm70Linear ? 1 : 0, mPicCompactSm70Tile,
                                      mResource->mStaticDequantBytes,
                                      gPicStaticDequantBytes.load(std::memory_order_relaxed),
                                      static_cast<unsigned long long>(dequantUs),
                                      static_cast<unsigned long long>(convertUs),
                                      static_cast<unsigned long long>(gemmUs),
                                      static_cast<unsigned long long>(convProfileNowUs() - totalStartUs));
                        }
                    }
#if MNN_CUDA_HAS_CUBLASLT
                    }
#endif
                    }  // closes V15 `else` (fallthrough to existing rows48/rows45/generic)
                    }  // closes V16 `else` (fallthrough to V15 / existing routes)
                }
            }
        } else {
            if(mFp16Infer) {
                CONV_FpAInt4B<<<blockNum, threadNum>>>((const half*)input_addr, (const uint8_t*)mResource->mFilter,
                    (const half*)mResource->mScale,  (const half*)mResource->mOffset, (const half*)bias_addr, (half*)output_addr,
                    maxV, minV, ic, icp, iw, ih, oc, ocp, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total, mResource -> mQuanC,
                    d_oc, d_ow, d_oh);
            } else {
                CONV_FpAInt4B<<<blockNum, threadNum>>>((const float*)input_addr, (const uint8_t*)mResource->mFilter,
                    (const float*)mResource->mScale,  (const float*)mResource->mOffset, (const float*)bias_addr, (float*)output_addr,
                    maxV, minV, ic, icp, iw, ih, oc, ocp, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total, mResource -> mQuanC,
                    d_oc, d_ow, d_oh);
            }
        }
    } else { // int8
        if (mIsConv1x1S1D1P0) {
            if (mFp32Infer) { // High + Int8
                if (batch <= 16) {
                    // P1: Use optimized GEMV with warp shuffle reduction
                    dim3 threads(WARP_SIZE, GEMV_OC_PER_BLOCK);
                    dim3 blocks(UP_DIV(oc, GEMV_OC_PER_BLOCK), batch);

                    if (mFp16Infer) {
                        GEMV_FpAInt8B_V2<<<blocks, threads>>>(
                            (const half*)input_addr, (const int8_t*)mResource->mFilter,
                            (const half*)mResource->mScale,  (const half*)mResource->mOffset,
                            (const half*)bias_addr, (half*)output_addr,
                            maxV, minV, batch, ic, icp, oc, ocp, mResource->mQuanC
                        );
                    } else {
                        GEMV_FpAInt8B_V2<<<blocks, threads>>>(
                            (const float*)input_addr, (const int8_t*)mResource->mFilter,
                            (const float*)mResource->mScale,  (const float*)mResource->mOffset,
                            (const float*)bias_addr, (float*)output_addr,
                            maxV, minV, batch, ic, icp, oc, ocp, mResource->mQuanC
                        );
                    }
                } else {
                    dim3 threads(TILE_DIM, TILE_DIM);
                    dim3 blocks(UP_DIV(ocp, TILE_DIM), UP_DIV(batch, TILE_DIM));

                    GEMM_FpAInt8B<<<blocks, threads>>>(
                        (const float*)input_addr, (const int8_t*)mResource->mFilter,
                        (const float*)mResource->mScale,  (const float*)mResource->mOffset,
                        (const float*)bias_addr, (float*)output_addr,
                        maxV, minV, ic, icp, oc, ocp, batch, mResource->mQuanC
                    );
                }
            } else {
                if (batch <= 16) {
                    // P1: FP16 direct GEMV for INT8 small batch
                    dim3 threads(WARP_SIZE, GEMV_OC_PER_BLOCK);
                    dim3 blocks(UP_DIV(oc, GEMV_OC_PER_BLOCK), batch);
                    if (mFp16Infer) {
                        GEMV_FpAInt8B_V2<<<blocks, threads>>>(
                            (const half*)input_addr, (const int8_t*)mResource->mFilter,
                            (const half*)mResource->mScale, (const half*)mResource->mOffset,
                            (const half*)bias_addr, (half*)output_addr,
                            maxV, minV, batch, ic, icp, oc, ocp, mResource->mQuanC
                        );
                    } else if (mFp16Fp32MixInfer) {
                        GEMV_FpAInt8B_V2<<<blocks, threads>>>(
                            (const float*)input_addr, (const int8_t*)mResource->mFilter,
                            (const float*)mResource->mScale, (const float*)mResource->mOffset,
                            (const float*)bias_addr, (float*)output_addr,
                            maxV, minV, batch, ic, icp, oc, ocp, mResource->mQuanC
                        );
                    }
                } else {
                    // Prefill path: CUTLASS GEMM with dequantized FP16 weights
                    if (mFp16Fp32MixInfer) {
                        size_t maxCount = mGemmInfo.elh[0] * mGemmInfo.elhPad[1];
                        callFloat2Half(input_addr, mIm2ColBuffer, maxCount, runtime);
                    }
                    runCutlassGemmFunc();
                }
            }
        } else {
            if(mFp16Infer) {
                CONV_FpAInt8B<<<blockNum, threadNum>>>((const half*)input_addr, (const int8_t*)mResource->mFilter,
                    (const half*)mResource->mScale,  (const half*)mResource->mOffset, (const half*)bias_addr, (half*)output_addr,
                    maxV, minV, ic, icp, iw, ih, oc, ocp, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total, mResource -> mQuanC,
                    d_oc, d_ow, d_oh);
            } else {
                CONV_FpAInt8B<<<blockNum, threadNum>>>((const float*)input_addr, (const int8_t*)mResource->mFilter,
                        (const float*)mResource->mScale,  (const float*)mResource->mOffset, (const float*)bias_addr, (float*)output_addr,
                    maxV, minV, ic, icp, iw, ih, oc, ocp, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total, mResource -> mQuanC,
                    d_oc, d_ow, d_oh);
            }
        }
    }
    
    checkKernelErrors;
    return NO_ERROR;
}


}// namespace CUDA
}// namespace MNN
