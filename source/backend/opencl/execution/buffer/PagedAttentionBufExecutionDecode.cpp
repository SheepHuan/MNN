//
//  PagedAttentionBufExecutionDecode.cpp
//  MNN
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/PagedAttentionBufExecution.hpp"
#include "backend/opencl/execution/buffer/PagedAttentionMaliUtils.hpp"
#include "backend/opencl/core/OpenCLPmcProfiler.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace MNN {
namespace OpenCL {
namespace {

static bool _envFlagEnabled(const char* name, bool defaultValue) {
    const char* value = ::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    return value[0] != '0';
}

static bool _profilePagedAttention() {
    return _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE", false);
}

static bool _picOpenCLDebug() {
    return _envFlagEnabled("MNN_PIC_DECODE_DEBUG", false);
}

static uint64_t _nowUs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

static OpenCLPmcScopeMeta _makeOpenCLPmcMeta(const char* op, const char* phase, int layer, int query,
                                             int inputQuery, int kvLen, int baseLogical, int lane, int qTile,
                                             int heads, int kvHeads, int headDim,
                                             const std::vector<uint32_t>& gws,
                                             const std::vector<uint32_t>& lws,
                                             uint64_t denseKvWork = 0, uint64_t causalKvWork = 0,
                                             int appendCount = 0, int prepareLen = 0,
                                             bool decodePrepareInsideDecode = false) {
    OpenCLPmcScopeMeta meta;
    meta.op = op;
    meta.phase = phase;
    meta.layer = layer;
    meta.query = query;
    meta.inputQuery = inputQuery;
    meta.kvLen = kvLen;
    meta.baseLogical = baseLogical;
    meta.lane = lane;
    meta.qTile = qTile;
    meta.heads = heads;
    meta.kvHeads = kvHeads;
    meta.headDim = headDim;
    meta.gws = gws;
    meta.lws = lws;
    meta.denseKvWork = denseKvWork;
    meta.causalKvWork = causalKvWork;
    meta.appendCount = appendCount;
    meta.prepareLen = prepareLen;
    meta.decodePrepareInsideDecode = decodePrepareInsideDecode;
    return meta;
}

template <typename Runner>
static void _runOpenCLPmcScope(OpenCLRuntime* runtime, const OpenCLPmcScopeMeta& meta, const Runner& runner) {
    if (runtime == nullptr) {
        runner(nullptr);
        return;
    }
    auto& pmc = OpenCLPmcProfiler::get();
    auto& queue = runtime->commandQueue();
    uint64_t token = 0;
    if (pmc.enabledFor(meta)) {
        token = pmc.begin(runtime, queue, meta);
    }
    cl::Event event;
    runner(token != 0 ? &event : nullptr);
    if (token != 0) {
        pmc.end(token, runtime, queue, meta, &event);
    }
}

static void _run3DKernelDefaultPmc(const std::shared_ptr<KernelWrap>& kernel,
                                   const std::vector<uint32_t>& gws,
                                   const std::vector<uint32_t>& lws,
                                   OpenCLRuntime* runtime,
                                   const OpenCLPmcScopeMeta& meta) {
    _runOpenCLPmcScope(runtime, meta, [&](cl::Event* eventPtr) {
        run3DKernelDefault(kernel, gws, lws, runtime, eventPtr);
    });
}

static bool _legacyB863976OpenCL() {
    return _envFlagEnabled("MNN_PAGED_ATTENTION_OPENCL_LEGACY_B863976", false);
}

static bool _decodeGqaFusedKVEnabled() {
    static const bool enabled = _envFlagEnabled("MNN_PAGED_ATTENTION_DECODE_GQA_FUSED", false);
    return enabled;
}

static bool _decodeIdentityAttentionOnlyBenchEnabled() {
    static const bool enabled = _envFlagEnabled("MNN_PAGED_ATTENTION_DECODE_IDENTITY_ATTENTION_ONLY_BENCH", false);
    return enabled;
}

static bool _decodeQ1SplitProfileEnabled() {
    static const bool enabled = _envFlagEnabled("MNN_PAGED_ATTENTION_DECODE_Q1_SPLIT_PROFILE", false);
    return enabled;
}

static bool _decodeRepairSlotIdentitySplitEnabled() {
    static const bool enabled =
        _envFlagEnabled("MNN_PAGED_ATTENTION_DECODE_REPAIR_SLOT_IDENTITY_SPLIT", false);
    return enabled;
}

static bool _decodeRepairFusedAppendEnabled() {
    // Disabled after OrangePi Qwen3-4B ctx1024 A/B on 2026-07-05:
    // even with a single-kernel record path, fused append regressed x0/x1 TPOT.
    return false;
}

static bool _decodeRepairQ1RowEnabled() {
    static const bool enabled = _envFlagEnabled("MNN_PAGED_ATTENTION_DECODE_REPAIR_Q1_ROW", true);
    return enabled;
}

static bool _decodeRepairRecordQueueEnabled() {
    static const bool enabled = _envFlagEnabled("MNN_PAGED_ATTENTION_DECODE_REPAIR_RECORD_QUEUE", true);
    return enabled;
}

static uint32_t _decodeHD128LaneWidth(uint64_t causalWorkPerRow) {
    const char* forced = ::getenv("MNN_PAGED_ATTENTION_DECODE_HD128_FORCE_LANE");
    if (forced != nullptr && forced[0] != '\0') {
        const int lane = ::atoi(forced);
        if (lane == 32 || lane == 64 || lane == 128) {
            return static_cast<uint32_t>(lane);
        }
    }
    return causalWorkPerRow >= 512u ? 128u : (causalWorkPerRow >= 256u ? 64u : 32u);
}

static uint32_t _decodeHD128SparseLaneWidth(OpenCLRuntime* runtime, uint64_t causalWorkPerRow, int attnLen) {
    const char* forced = ::getenv("MNN_PAGED_ATTENTION_DECODE_HD128_FORCE_LANE");
    if (forced != nullptr && forced[0] != '\0') {
        const int lane = ::atoi(forced);
        if (lane == 32 || lane == 64 || lane == 128) {
            return static_cast<uint32_t>(lane);
        }
    }
    return PagedAttentionMali::selectDecodeHD128SparseLaneWidth(
        runtime, causalWorkPerRow, attnLen, _decodeHD128LaneWidth(causalWorkPerRow));
}

static int _decodeHD128QTile(OpenCLRuntime* runtime, int attnLen) {
    const char* forced = ::getenv("MNN_PAGED_ATTENTION_DECODE_QTILE_FORCE");
    if (forced != nullptr && forced[0] != '\0') {
        const int qTile = ::atoi(forced);
        if (qTile == 1 || qTile == 2 || qTile == 4 || qTile == 8) {
            return qTile;
        }
    }
    const int maliQTile = PagedAttentionMali::selectDecodeHD128QTile(runtime, attnLen);
    if (maliQTile > 0) {
        return maliQTile;
    }
    return attnLen <= 2 ? 2 : (attnLen <= 4 ? 4 : 8);
}

static uint32_t _sparseFlashLaneWidth(const PagedKVMeta* meta, int activeLen, int headDim) {
    if (headDim >= 128) {
        return 32u;
    }
    if (activeLen < 384) {
        return 64u;
    }
    if (meta != nullptr && meta->cacheblend_score_ready && meta->cacheblend_score_pic_token_count > 0) {
        const int selected = meta->cacheblend_score_selected_local_indices.empty()
            ? meta->cacheblend_score_top_k
            : static_cast<int>(meta->cacheblend_score_selected_local_indices.size());
        if (selected * 100 >= meta->cacheblend_score_pic_token_count * 50) {
            return 64u;
        }
    }
    return 32u;
}

static uint64_t _sparseLogicalWork(const PagedKVMeta* meta, int activeLen, int kvLen) {
    if (meta == nullptr || activeLen <= 0 || kvLen <= 0 ||
        static_cast<int>(meta->sparse_query_logical_indices.size()) < activeLen) {
        return 0;
    }
    uint64_t work = 0;
    for (int i = 0; i < activeLen; ++i) {
        const int logical = meta->sparse_query_logical_indices[static_cast<size_t>(i)];
        work += static_cast<uint64_t>(std::max(0, std::min(kvLen, logical + 1)));
    }
    return work;
}

} // namespace

ErrorCode PagedAttentionBufExecution::syncDecodeAttentionHeadIds() {
    if (mMeta == nullptr || mMeta->pic_decode_attention_head_ids.empty()) {
        mDecodeAttentionHeadIdsHost.clear();
        return NO_ERROR;
    }
    const auto& heads = mMeta->pic_decode_attention_head_ids;
    if (mDecodeAttentionHeadIds == nullptr ||
        mDecodeAttentionHeadIdCapacity < static_cast<int>(heads.size())) {
        mDecodeAttentionHeadIds.reset(Tensor::createDevice<int>({static_cast<int>(heads.size())}));
        if (!mDecodeAttentionHeadIds) {
            return OUT_OF_MEMORY;
        }
        OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mDecodeAttentionHeadIds.get(), Backend::STATIC));
        mDecodeAttentionHeadIdCapacity = static_cast<int>(heads.size());
        mDecodeAttentionHeadIdsHost.clear();
    }
    if (mDecodeAttentionHeadIdsHost != heads) {
        auto ret = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueWriteBuffer(
            openCLBuffer(mDecodeAttentionHeadIds.get()), CL_TRUE, 0, heads.size() * sizeof(int), heads.data());
        if (ret != CL_SUCCESS) {
            mDecodeAttentionHeadIdsHost.clear();
            return INVALID_VALUE;
        }
        mDecodeAttentionHeadIdsHost = heads;
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureDecodeCausalKernel() {
    if (mHeadDim != 64 || mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    const int groupSize = mNumHead / mKvNumHead;
    if (mDecodeCausalKernel32 && mDecodeCausalKernel64 && mDecodeCausalKernelGroupSize == groupSize) {
        return NO_ERROR;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    mDecodeCausalKernel32 = runtime->buildKernel("attention_buf", "decode_causal_attention_row32",
                                                 {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                                                 mOpenCLBackend->getPrecision());
    mDecodeCausalKernel64 = runtime->buildKernel("attention_buf", "decode_causal_attention_row64",
                                                 {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                                                 mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL(mDecodeCausalKernel32);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernel64);
    mDecodeCausalKernelGroupSize = groupSize;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureDecodeCausalKernelHD128Identity() {
    if (mHeadDim != 128 || mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    const int groupSize = mNumHead / mKvNumHead;
    const bool needGqaFusedKV = groupSize > 1 && _decodeGqaFusedKVEnabled();
    if (mDecodeCausalKernelHD128IdentityRow32 && mDecodeCausalKernelHD128IdentityRow64 &&
        mDecodeCausalKernelHD128IdentityRow128 &&
        mDecodeCausalKernelHD128IdentityFusedKVRow32 && mDecodeCausalKernelHD128IdentityFusedKVRow64 &&
        mDecodeCausalKernelHD128IdentityFusedKVRow128 &&
        mDecodeQKIdentityKernel &&
        (!needGqaFusedKV ||
         (mDecodeCausalKernelHD128IdentityFusedKVGQARow32 &&
          mDecodeCausalKernelHD128IdentityFusedKVGQARow64 &&
          mDecodeCausalKernelHD128IdentityFusedKVGQARow128)) &&
        mDecodeCausalHD128IdentityKernelGroupSize == groupSize) {
        return NO_ERROR;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    mDecodeCausalKernelHD128IdentityRow32 =
        runtime->buildKernel("paged_decode_attention_buf", "decode_causal_attention_hd128_identity_row32",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128IdentityRow64 =
        runtime->buildKernel("paged_decode_attention_buf", "decode_causal_attention_hd128_identity_row64",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128IdentityRow128 =
        runtime->buildKernel("paged_decode_attention_buf", "decode_causal_attention_hd128_identity_row128",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128IdentityFusedKVRow32 =
        runtime->buildKernel("paged_decode_attention_buf", "decode_causal_attention_hd128_identity_fused_kv_row32",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128IdentityFusedKVRow64 =
        runtime->buildKernel("paged_decode_attention_buf", "decode_causal_attention_hd128_identity_fused_kv_row64",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128IdentityFusedKVRow128 =
        runtime->buildKernel("paged_decode_attention_buf", "decode_causal_attention_hd128_identity_fused_kv_row128",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    mDecodeQKIdentityKernel =
        runtime->buildKernel("paged_decode_attention_buf", "matmul_qk_decode_paged_identity_hd128",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    if (needGqaFusedKV) {
        mDecodeCausalKernelHD128IdentityFusedKVGQARow32 =
            runtime->buildKernel("paged_decode_attention_buf", "decode_causal_attention_hd128_identity_fused_kv_gqa_row32",
                                 {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                                 mOpenCLBackend->getPrecision());
        mDecodeCausalKernelHD128IdentityFusedKVGQARow64 =
            runtime->buildKernel("paged_decode_attention_buf", "decode_causal_attention_hd128_identity_fused_kv_gqa_row64",
                                 {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                                 mOpenCLBackend->getPrecision());
        mDecodeCausalKernelHD128IdentityFusedKVGQARow128 =
            runtime->buildKernel("paged_decode_attention_buf", "decode_causal_attention_hd128_identity_fused_kv_gqa_row128",
                                 {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                                 mOpenCLBackend->getPrecision());
    } else {
        mDecodeCausalKernelHD128IdentityFusedKVGQARow32.reset();
        mDecodeCausalKernelHD128IdentityFusedKVGQARow64.reset();
        mDecodeCausalKernelHD128IdentityFusedKVGQARow128.reset();
    }
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128IdentityRow32);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128IdentityRow64);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128IdentityRow128);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128IdentityFusedKVRow32);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128IdentityFusedKVRow64);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128IdentityFusedKVRow128);
    OPENCL_CHECK_KERNEL(mDecodeQKIdentityKernel);
    if (needGqaFusedKV) {
        OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128IdentityFusedKVGQARow32);
        OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128IdentityFusedKVGQARow64);
        OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128IdentityFusedKVGQARow128);
    }
    mDecodeCausalHD128IdentityKernelGroupSize = groupSize;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureDecodeTransposedKKernel() {
    if (mHeadDim != 128 || mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    const int groupSize = mNumHead / mKvNumHead;
    const bool buildFusedAppend = _decodeRepairFusedAppendEnabled();
    // Negative A/B kernels are intentionally excluded from readiness/build:
    // keycache qtile v2, q1 identity fused-KV, identity qtile, and sparse GQA all
    // failed to remove the PIC x0 gap or regressed larger models.
    if (mDecodeKeyTransposeKernel && mDecodeKeyAppendKernel && mDecodeKeyAppendSparseKernel &&
        mDecodeQKTransposedKernel &&
        mDecodeQKVTransposedKernel && mDecodeCausalKernelHD128TransposedKFusedKVRow32 &&
        mDecodeCausalKernelHD128TransposedKFusedKVRow64 &&
        mDecodeCausalKernelHD128TransposedKFusedKVRow128 &&
        mDecodeCausalKernelHD128TransposedKReadonlyRow32 &&
        mDecodeCausalKernelHD128TransposedKReadonlyRow64 &&
        mDecodeCausalKernelHD128TransposedKReadonlyRow128 &&
        mDecodeCausalKernelHD128TransposedKSparseRow32 &&
        mDecodeCausalKernelHD128TransposedKSparseRow64 &&
        mDecodeCausalKernelHD128TransposedKSparseRow128 &&
        mDecodeCausalKernelHD128TransposedKQTile1Row32 &&
        mDecodeCausalKernelHD128TransposedKQTile1Row64 &&
        mDecodeCausalKernelHD128TransposedKQTile1Row128 &&
        mDecodeCausalKernelHD128TransposedKQTile2Row32 &&
        mDecodeCausalKernelHD128TransposedKQTile2Row64 &&
        mDecodeCausalKernelHD128TransposedKQTile2Row128 &&
        mDecodeCausalKernelHD128TransposedKQTile4Row32 &&
        mDecodeCausalKernelHD128TransposedKQTile4Row64 &&
        mDecodeCausalKernelHD128TransposedKQTile4Row128 &&
        mDecodeCausalKernelHD128TransposedKQTile8Row32 &&
        mDecodeCausalKernelHD128TransposedKQTile8Row64 &&
        mDecodeCausalKernelHD128TransposedKQTile8Row128 &&
        (!buildFusedAppend ||
         (mDecodeCausalKernelHD128TransposedKQTileFused1Row32 &&
          mDecodeCausalKernelHD128TransposedKQTileFused1Row64 &&
          mDecodeCausalKernelHD128TransposedKQTileFused1Row128 &&
          mDecodeCausalKernelHD128TransposedKQTileFused2Row32 &&
          mDecodeCausalKernelHD128TransposedKQTileFused2Row64 &&
          mDecodeCausalKernelHD128TransposedKQTileFused2Row128 &&
          mDecodeCausalKernelHD128TransposedKQTileFused4Row32 &&
          mDecodeCausalKernelHD128TransposedKQTileFused4Row64 &&
          mDecodeCausalKernelHD128TransposedKQTileFused4Row128 &&
          mDecodeCausalKernelHD128TransposedKQTileFused8Row32 &&
          mDecodeCausalKernelHD128TransposedKQTileFused8Row64 &&
          mDecodeCausalKernelHD128TransposedKQTileFused8Row128)) &&
        (groupSize <= 1 ||
         (mDecodeCausalKernelHD128TransposedKFusedKVGQARow32 &&
          mDecodeCausalKernelHD128TransposedKFusedKVGQARow64 &&
          mDecodeCausalKernelHD128TransposedKFusedKVGQARow128)) &&
        mDecodeTransposedKKernelGroupSize == groupSize) {
        return NO_ERROR;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const std::set<std::string> options = {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)};
    mDecodeKeyTransposeKernel =
        runtime->buildKernel("paged_decode_attention_buf", "transpose_paged_key_to_decode_key", options,
                             mOpenCLBackend->getPrecision());
    mDecodeKeyAppendKernel =
        runtime->buildKernel("paged_decode_attention_buf", "append_decode_key_value_hd128", options,
                             mOpenCLBackend->getPrecision());
    mDecodeKeyAppendSparseKernel =
        runtime->buildKernel("paged_decode_attention_buf", "append_sparse_decode_key_value_hd128", options,
                             mOpenCLBackend->getPrecision());
    mDecodeQKTransposedKernel =
        runtime->buildKernel("paged_decode_attention_buf", "matmul_qk_decode_transposed_hd128", options,
                             mOpenCLBackend->getPrecision());
    mDecodeQKVTransposedKernel =
        runtime->buildKernel("paged_decode_attention_buf", "matmul_qkv_decode_value_hd128_b8", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKFusedKVRow32 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_fused_kv_row32", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKFusedKVRow64 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_fused_kv_row64", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKFusedKVRow128 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_fused_kv_row128", options,
                             mOpenCLBackend->getPrecision());
    if (groupSize > 1) {
        mDecodeCausalKernelHD128TransposedKFusedKVGQARow32 =
            runtime->buildKernel("paged_decode_attention_buf",
                                 "decode_causal_attention_hd128_transposed_k_fused_kv_gqa_row32", options,
                                 mOpenCLBackend->getPrecision());
        mDecodeCausalKernelHD128TransposedKFusedKVGQARow64 =
            runtime->buildKernel("paged_decode_attention_buf",
                                 "decode_causal_attention_hd128_transposed_k_fused_kv_gqa_row64", options,
                                 mOpenCLBackend->getPrecision());
        mDecodeCausalKernelHD128TransposedKFusedKVGQARow128 =
            runtime->buildKernel("paged_decode_attention_buf",
                                 "decode_causal_attention_hd128_transposed_k_fused_kv_gqa_row128", options,
                                 mOpenCLBackend->getPrecision());
    } else {
        mDecodeCausalKernelHD128TransposedKFusedKVGQARow32 = nullptr;
        mDecodeCausalKernelHD128TransposedKFusedKVGQARow64 = nullptr;
        mDecodeCausalKernelHD128TransposedKFusedKVGQARow128 = nullptr;
    }
    mDecodeCausalKernelHD128TransposedKReadonlyRow32 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_readonly_row32", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKReadonlyRow64 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_readonly_row64", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKReadonlyRow128 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_readonly_row128", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKSparseRow32 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_sparse_row32", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKSparseRow64 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_sparse_row64", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKSparseRow128 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_sparse_row128", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKSparseDcontigRow128 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_sparse_dcontig_row128", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile1Row32 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q1_row32", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile1Row64 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q1_row64", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile1Row128 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q1_row128", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile2Row32 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q2_row32", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile2Row64 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q2_row64", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile2Row128 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q2_row128", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile4Row32 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q4_row32", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile4Row64 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q4_row64", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile4Row128 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q4_row128", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile8Row32 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q8_row32", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile8Row64 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q8_row64", options,
                             mOpenCLBackend->getPrecision());
    mDecodeCausalKernelHD128TransposedKQTile8Row128 =
        runtime->buildKernel("paged_decode_attention_buf",
                             "decode_causal_attention_hd128_transposed_k_qtile_q8_row128", options,
                             mOpenCLBackend->getPrecision());
    if (buildFusedAppend) {
        mDecodeCausalKernelHD128TransposedKQTileFused1Row32 =
            runtime->buildKernel("paged_decode_attention_buf",
                                 "decode_causal_attention_hd128_transposed_k_qtile_fused_q1_row32", options,
                                 mOpenCLBackend->getPrecision());
        mDecodeCausalKernelHD128TransposedKQTileFused1Row64 =
            runtime->buildKernel("paged_decode_attention_buf",
                                 "decode_causal_attention_hd128_transposed_k_qtile_fused_q1_row64", options,
                                 mOpenCLBackend->getPrecision());
        mDecodeCausalKernelHD128TransposedKQTileFused1Row128 =
            runtime->buildKernel("paged_decode_attention_buf",
                                 "decode_causal_attention_hd128_transposed_k_qtile_fused_q1_row128", options,
                                 mOpenCLBackend->getPrecision());
        mDecodeCausalKernelHD128TransposedKQTileFused2Row32 =
            runtime->buildKernel("paged_decode_attention_buf",
                                 "decode_causal_attention_hd128_transposed_k_qtile_fused_q2_row32", options,
                                 mOpenCLBackend->getPrecision());
        mDecodeCausalKernelHD128TransposedKQTileFused2Row64 =
            runtime->buildKernel("paged_decode_attention_buf",
                                 "decode_causal_attention_hd128_transposed_k_qtile_fused_q2_row64", options,
                                 mOpenCLBackend->getPrecision());
        mDecodeCausalKernelHD128TransposedKQTileFused2Row128 =
            runtime->buildKernel("paged_decode_attention_buf",
                                 "decode_causal_attention_hd128_transposed_k_qtile_fused_q2_row128", options,
                                 mOpenCLBackend->getPrecision());
        mDecodeCausalKernelHD128TransposedKQTileFused4Row32 =
            runtime->buildKernel("paged_decode_attention_buf",
                                 "decode_causal_attention_hd128_transposed_k_qtile_fused_q4_row32", options,
                                 mOpenCLBackend->getPrecision());
        mDecodeCausalKernelHD128TransposedKQTileFused4Row64 =
            runtime->buildKernel("paged_decode_attention_buf",
                                 "decode_causal_attention_hd128_transposed_k_qtile_fused_q4_row64", options,
                                 mOpenCLBackend->getPrecision());
        mDecodeCausalKernelHD128TransposedKQTileFused4Row128 =
            runtime->buildKernel("paged_decode_attention_buf",
                                 "decode_causal_attention_hd128_transposed_k_qtile_fused_q4_row128", options,
                                 mOpenCLBackend->getPrecision());
        mDecodeCausalKernelHD128TransposedKQTileFused8Row32 =
            runtime->buildKernel("paged_decode_attention_buf",
                                 "decode_causal_attention_hd128_transposed_k_qtile_fused_q8_row32", options,
                                 mOpenCLBackend->getPrecision());
        mDecodeCausalKernelHD128TransposedKQTileFused8Row64 =
            runtime->buildKernel("paged_decode_attention_buf",
                                 "decode_causal_attention_hd128_transposed_k_qtile_fused_q8_row64", options,
                                 mOpenCLBackend->getPrecision());
        mDecodeCausalKernelHD128TransposedKQTileFused8Row128 =
            runtime->buildKernel("paged_decode_attention_buf",
                                 "decode_causal_attention_hd128_transposed_k_qtile_fused_q8_row128", options,
                                 mOpenCLBackend->getPrecision());
    } else {
        mDecodeCausalKernelHD128TransposedKQTileFused1Row32 = nullptr;
        mDecodeCausalKernelHD128TransposedKQTileFused1Row64 = nullptr;
        mDecodeCausalKernelHD128TransposedKQTileFused1Row128 = nullptr;
        mDecodeCausalKernelHD128TransposedKQTileFused2Row32 = nullptr;
        mDecodeCausalKernelHD128TransposedKQTileFused2Row64 = nullptr;
        mDecodeCausalKernelHD128TransposedKQTileFused2Row128 = nullptr;
        mDecodeCausalKernelHD128TransposedKQTileFused4Row32 = nullptr;
        mDecodeCausalKernelHD128TransposedKQTileFused4Row64 = nullptr;
        mDecodeCausalKernelHD128TransposedKQTileFused4Row128 = nullptr;
        mDecodeCausalKernelHD128TransposedKQTileFused8Row32 = nullptr;
        mDecodeCausalKernelHD128TransposedKQTileFused8Row64 = nullptr;
        mDecodeCausalKernelHD128TransposedKQTileFused8Row128 = nullptr;
    }
    OPENCL_CHECK_KERNEL(mDecodeKeyTransposeKernel);
    OPENCL_CHECK_KERNEL(mDecodeKeyAppendKernel);
    OPENCL_CHECK_KERNEL(mDecodeKeyAppendSparseKernel);
    OPENCL_CHECK_KERNEL(mDecodeQKTransposedKernel);
    OPENCL_CHECK_KERNEL(mDecodeQKVTransposedKernel);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKFusedKVRow32);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKFusedKVRow64);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKFusedKVRow128);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKReadonlyRow32);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKReadonlyRow64);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKReadonlyRow128);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKSparseRow32);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKSparseRow64);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKSparseRow128);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile1Row32);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile1Row64);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile1Row128);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile2Row32);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile2Row64);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile2Row128);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile4Row32);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile4Row64);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile4Row128);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile8Row32);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile8Row64);
    OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTile8Row128);
    if (buildFusedAppend) {
        OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTileFused1Row32);
        OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTileFused1Row64);
        OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTileFused1Row128);
        OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTileFused2Row32);
        OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTileFused2Row64);
        OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTileFused2Row128);
        OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTileFused4Row32);
        OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTileFused4Row64);
        OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTileFused4Row128);
        OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTileFused8Row32);
        OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTileFused8Row64);
        OPENCL_CHECK_KERNEL(mDecodeCausalKernelHD128TransposedKQTileFused8Row128);
    }
    mDecodeTransposedKKernelGroupSize = groupSize;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureDecodeTransposedTemps(int kvLen, int rows) {
    if (kvLen <= 0 || rows <= 0 || mBatch <= 0 || mNumHead <= 0) {
        return INVALID_VALUE;
    }
    const int kvPack = ROUND_UP(kvLen, 8);
    if (!(mTempQK && mTempSoftmax && mDecodeTransposedTempKvLen >= kvPack &&
          mDecodeTransposedTempRows >= rows)) {
        const int elements = kvPack * rows * mNumHead * mBatch;
        mTempQK.reset(Tensor::createDevice<float>({elements}));
        mTempSoftmax.reset(Tensor::createDevice<float>({elements}));
        if (!mTempQK || !mTempSoftmax) {
            return OUT_OF_MEMORY;
        }
        mDecodeTransposedTempKvLen = kvPack;
        mDecodeTransposedTempRows = rows;
    }
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempQK.get(), Backend::DYNAMIC_IN_EXECUTION));
    OPENCL_CHECK_ALLOC(mOpenCLBackend->onAcquireBuffer(mTempSoftmax.get(), Backend::DYNAMIC_IN_EXECUTION));
    mOpenCLBackend->onReleaseBuffer(mTempQK.get(), Backend::DYNAMIC_IN_EXECUTION);
    mOpenCLBackend->onReleaseBuffer(mTempSoftmax.get(), Backend::DYNAMIC_IN_EXECUTION);
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureDecodeKeyReady(int requiredLen, bool insideDecode) {
    mLastDecodeKeyRequiredLen = requiredLen;
    mLastDecodeKeyPrepareUs = 0;
    mLastDecodeKeyReadyHit = false;
    mLastDecodeKeyPrepared = false;
    mLastDecodeKeyPrepareInsideDecode = false;
    mLastDecodeKeySlotIdentity = false;
    mLastDecodeKeyPrefixStable = false;
    if (requiredLen <= 0) {
        mLastDecodeKeyReadyHit = true;
        mLastDecodeKeyPrefixStable = true;
        return NO_ERROR;
    }
    if (!mCache || !mCache->key || !mCache->decodeKey ||
        requiredLen > mCache->maxSlots) {
        return OUT_OF_MEMORY;
    }
    mLastDecodeKeySlotIdentity = true;
    mLastDecodeKeyPrefixStable = true;
    if (mCache->decodeKeyReadyLength >= requiredLen) {
        mLastDecodeKeyReadyHit = true;
        if (_picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA decode_key ready hit layer=%d cache=%p required=%d ready=%d\n",
                      mLayerIndex, static_cast<void*>(mCache.get()), requiredLen,
                      mCache->decodeKeyReadyLength);
        }
        return NO_ERROR;
    }
    if (_picOpenCLDebug()) {
        MNN_PRINT("PIC OpenCL PA decode_key prepare layer=%d cache=%p required=%d ready=%d max_slots=%d\n",
                  mLayerIndex, static_cast<void*>(mCache.get()), requiredLen, mCache->decodeKeyReadyLength,
                  mCache->maxSlots);
    }
    if (insideDecode) {
        mLastDecodeKeyPrepareInsideDecode = true;
        if (_picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA decode_key prepare blocked inside decode layer=%d required=%d ready=%d "
                      "prefix_stable=1\n",
                      mLayerIndex, requiredLen, mCache->decodeKeyReadyLength);
        }
        return INVALID_VALUE;
    }
    auto err = ensureDecodeTransposedKKernel();
    if (err != NO_ERROR) {
        return err;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto& queue = runtime->commandQueue();
    const bool profile = _profilePagedAttention();
    const uint64_t startUs = profile ? _nowUs() : 0;
    std::vector<uint32_t> gws = {
        static_cast<uint32_t>(requiredLen),
        static_cast<uint32_t>(mBatch * mKvNumHead * mHeadDim),
    };
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mDecodeKeyTransposeKernel->get().setArg(idx++, gws[0]);
    ret |= mDecodeKeyTransposeKernel->get().setArg(idx++, gws[1]);
    ret |= mDecodeKeyTransposeKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= mDecodeKeyTransposeKernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
    ret |= mDecodeKeyTransposeKernel->get().setArg(idx++, mBatch);
    ret |= mDecodeKeyTransposeKernel->get().setArg(idx++, mKvNumHead);
    ret |= mDecodeKeyTransposeKernel->get().setArg(idx++, mHeadDim);
    ret |= mDecodeKeyTransposeKernel->get().setArg(idx++, requiredLen);
    ret |= mDecodeKeyTransposeKernel->get().setArg(idx++, mCache->maxSlots);
    MNN_CHECK_CL_SUCCESS(ret, "setArg transpose_paged_key_to_decode_key");
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const int layerIndex = mLayerIndex >= 0 ? mLayerIndex : (mMeta != nullptr ? mMeta->layer_index : -1);
    cl_int enqueueRet = CL_SUCCESS;
    const auto pmcMeta = _makeOpenCLPmcMeta("decode_prepare_transpose_k", "prepare", layerIndex,
                                           requiredLen, mQuerySeqLen, requiredLen, -1, 0, 0,
                                           mNumHead, mKvNumHead, mHeadDim, gws, {0u, 0u},
                                           0, 0, 0, requiredLen, insideDecode);
    _runOpenCLPmcScope(runtime, pmcMeta, [&](cl::Event* eventPtr) {
        enqueueRet = queue.enqueueNDRangeKernel(mDecodeKeyTransposeKernel->get(), cl::NullRange,
                                                cl::NDRange(gws[0], gws[1]), cl::NullRange,
                                                nullptr, eventPtr);
    });
    MNN_CHECK_CL_SUCCESS(enqueueRet, "enqueue transpose_paged_key_to_decode_key");
    if (enqueueRet != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    mCache->decodeKeyReadyLength = requiredLen;
    mLastDecodeKeyPrefixStable = true;
    mLastDecodeKeyPrepared = true;
    if (profile) {
        queue.finish();
        mLastDecodeKeyPrepareUs = _nowUs() - startUs;
        MNN_PRINT("OpenCLPagedAttention profile op=decode_prepare_transpose_k layer=%d kv_len=%d "
                  "inside_decode=%d slot_identity=%d prefix_stable=%d us=%llu prepare_us=%llu\n",
                  layerIndex, requiredLen, insideDecode ? 1 : 0, 1,
                  mLastDecodeKeyPrefixStable ? 1 : 0,
                  static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                  static_cast<unsigned long long>(mLastDecodeKeyPrepareUs));
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::ensureDecodeAttentionRankKernel() {
    if (mHeadDim != 128 || mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    const int groupSize = mNumHead / mKvNumHead;
    if (mDecodeAttentionRankScoreKernelHD128 &&
        mDecodeAttentionRankKernelGroupSize == groupSize) {
        return NO_ERROR;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    mDecodeAttentionRankScoreKernelHD128 =
        runtime->buildKernel("attention_buf", "decode_attention_pic_rank_score_hd128",
                             {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)},
                             mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL(mDecodeAttentionRankScoreKernelHD128);
    mDecodeAttentionRankKernelGroupSize = groupSize;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttention(const std::vector<Tensor*>& inputs,
                                                               const std::vector<Tensor*>& outputs, int kvLen,
                                                               int attnLen, int baseLogical, bool sparseQuery,
                                                               bool queryRowsAreFull) {
    if (attnLen <= 0 || kvLen <= 0 || mHeadDim != 64) {
        return INVALID_VALUE;
    }
    if (sparseQuery &&
        (mMeta == nullptr || static_cast<int>(mMeta->sparse_query_logical_indices.size()) < attnLen)) {
        return INVALID_VALUE;
    }
    auto err = ensureDecodeCausalKernel();
    if (err != NO_ERROR) {
        return err;
    }
    auto query = inputs[0];
    auto output = outputs[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t rearrangeUs = 0;
    uint64_t packUs = 0;
    uint64_t maskUs = 0;
    uint64_t qkUs = 0;
    uint64_t softmaxUs = 0;
    uint64_t qkvUs = 0;
    const uint32_t lanes = _sparseFlashLaneWidth(sparseQuery ? mMeta : nullptr, attnLen, mHeadDim);
    auto kernel = lanes == 32u ? mDecodeCausalKernel32 : mDecodeCausalKernel64;
    if (!kernel) {
        return INVALID_VALUE;
    }
    std::vector<uint32_t> gws = {
        lanes,
        static_cast<uint32_t>(attnLen),
        static_cast<uint32_t>(mNumHead * mBatch),
    };
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    ret |= kernel->get().setArg(idx++, gws[0]);
    ret |= kernel->get().setArg(idx++, gws[1]);
    ret |= kernel->get().setArg(idx++, gws[2]);
    ret |= kernel->get().setArg(idx++, openCLBuffer(query));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->sparseQuery.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= kernel->get().setArg(idx++, mScale);
    ret |= kernel->get().setArg(idx++, mBatch);
    ret |= kernel->get().setArg(idx++, mQuerySeqLen);
    ret |= kernel->get().setArg(idx++, attnLen);
    ret |= kernel->get().setArg(idx++, baseLogical);
    ret |= kernel->get().setArg(idx++, sparseQuery ? 1 : 0);
    ret |= kernel->get().setArg(idx++, queryRowsAreFull ? 1 : 0);
    ret |= kernel->get().setArg(idx++, kvLen);
    ret |= kernel->get().setArg(idx++, mCache->maxSlots);
    ret |= kernel->get().setArg(idx++, mNumHead);
    ret |= kernel->get().setArg(idx++, mKvNumHead);
    ret |= kernel->get().setArg(idx++, mHeadDim);
    MNN_CHECK_CL_SUCCESS(ret, lanes == 32u ? "setArg decode_causal_attention_row32"
                                           : "setArg decode_causal_attention_row64");
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const uint64_t denseWork = static_cast<uint64_t>(attnLen) * static_cast<uint64_t>(kvLen);
    uint64_t causalWork = 0;
    if (sparseQuery) {
        causalWork = _sparseLogicalWork(mMeta, attnLen, kvLen);
    } else {
        for (int i = 0; i < attnLen; ++i) {
            causalWork += static_cast<uint64_t>(std::max(0, std::min(kvLen, baseLogical + i + 1)));
        }
    }
    const int layerIndex = mLayerIndex >= 0 ? mLayerIndex : (mMeta != nullptr ? mMeta->layer_index : -1);
    const std::vector<uint32_t> lws = {lanes, 1u, 1u};
    const auto pmcMeta = _makeOpenCLPmcMeta("decode_causal_attention", "attention", layerIndex,
                                           attnLen, mQuerySeqLen, kvLen, baseLogical,
                                           static_cast<int>(lanes), 0, mNumHead, mKvNumHead, mHeadDim,
                                           gws, lws, denseWork, causalWork);
    _run3DKernelDefaultPmc(kernel, gws, lws, runtime, pmcMeta);
    if (profile) {
        runtime->commandQueue().finish();
        MNN_PRINT("OpenCLPagedAttention profile op=decode_causal_attention layer=%d "
                  "query=%d input_query=%d sparse=%d full_q=%d kv_len=%d lane=%u "
                  "dense_kv_work=%llu causal_kv_work=%llu us=%llu\n",
                  layerIndex, attnLen, mQuerySeqLen, sparseQuery ? 1 : 0, queryRowsAreFull ? 1 : 0,
                  kvLen, lanes,
                  static_cast<unsigned long long>(denseWork),
                  static_cast<unsigned long long>(causalWork),
                  static_cast<unsigned long long>(_nowUs() - startUs));
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128Identity(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int baseLogical, int layerIndex) {
    if (attnLen <= 0 || kvLen <= 0 || mHeadDim != 128 || mCache == nullptr ||
        !mCache->key || !mCache->value) {
        return INVALID_VALUE;
    }
    auto err = ensureDecodeCausalKernelHD128Identity();
    if (err != NO_ERROR) {
        return err;
    }
    auto query = inputs[0];
    auto output = outputs[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t attentionUs = 0;
    uint64_t rankUs = 0;
    uint64_t causalWork = 0;
    for (int i = 0; i < attnLen; ++i) {
        causalWork += static_cast<uint64_t>(std::max(0, std::min(kvLen, baseLogical + i + 1)));
    }
    const uint64_t causalWorkPerRow = attnLen > 0 ? causalWork / static_cast<uint64_t>(attnLen) : 0;
    const uint32_t lanes = _decodeHD128LaneWidth(causalWorkPerRow);
    auto kernel = lanes == 128u ? mDecodeCausalKernelHD128IdentityRow128 :
        (lanes == 64u ? mDecodeCausalKernelHD128IdentityRow64 : mDecodeCausalKernelHD128IdentityRow32);
    if (!kernel) {
        return INVALID_VALUE;
    }
    if (_decodeIdentityAttentionOnlyBenchEnabled() && _decodeQ1SplitProfileEnabled() && profile) {
        auto splitErr = runDecodeCausalAttentionHD128SplitProfile(inputs, outputs, kvLen, attnLen,
                                                                  baseLogical, layerIndex, false);
        if (splitErr == NO_ERROR) {
            return NO_ERROR;
        }
        if (_picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA decode identity split profile fallback layer=%d err=%d\n",
                      layerIndex, static_cast<int>(splitErr));
        }
    }
    if (mOpenCLBackend->isUseRecordQueue() && !profile &&
        (mMeta == nullptr || !mMeta->needsPicDecodeAttentionRankCapture(layerIndex))) {
        auto recordErr = runDecodeCausalAttentionHD128IdentityRecord(
            inputs, outputs, kvLen, attnLen, baseLogical, layerIndex, lanes, kernel);
        if (recordErr == NO_ERROR) {
            return NO_ERROR;
        }
    }
    std::vector<uint32_t> gws = {
        lanes,
        static_cast<uint32_t>(attnLen),
        static_cast<uint32_t>(mNumHead * mBatch),
    };
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    ret |= kernel->get().setArg(idx++, gws[0]);
    ret |= kernel->get().setArg(idx++, gws[1]);
    ret |= kernel->get().setArg(idx++, gws[2]);
    ret |= kernel->get().setArg(idx++, openCLBuffer(query));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= kernel->get().setArg(idx++, mScale);
    ret |= kernel->get().setArg(idx++, mBatch);
    ret |= kernel->get().setArg(idx++, mQuerySeqLen);
    ret |= kernel->get().setArg(idx++, attnLen);
    ret |= kernel->get().setArg(idx++, baseLogical);
    ret |= kernel->get().setArg(idx++, kvLen);
    ret |= kernel->get().setArg(idx++, mCache->maxSlots);
    ret |= kernel->get().setArg(idx++, mNumHead);
    ret |= kernel->get().setArg(idx++, mKvNumHead);
    ret |= kernel->get().setArg(idx++, mHeadDim);
    MNN_CHECK_CL_SUCCESS(ret, lanes == 128u ? "setArg decode_causal_attention_hd128_identity_row128" :
        (lanes == 64u ? "setArg decode_causal_attention_hd128_identity_row64"
                      : "setArg decode_causal_attention_hd128_identity_row32"));
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const uint64_t attentionStartUs = profileDetail ? _nowUs() : 0;
    const uint64_t denseWork = static_cast<uint64_t>(attnLen) * static_cast<uint64_t>(kvLen);
    const std::vector<uint32_t> lws = {lanes, 1u, 1u};
    const auto pmcMeta = _makeOpenCLPmcMeta("decode_causal_attention_hd128_identity", "attention", layerIndex,
                                           attnLen, mQuerySeqLen, kvLen, baseLogical,
                                           static_cast<int>(lanes), 0, mNumHead, mKvNumHead, mHeadDim,
                                           gws, lws, denseWork, causalWork);
    _run3DKernelDefaultPmc(kernel, gws, lws, runtime, pmcMeta);
    if (profileDetail) {
        runtime->commandQueue().finish();
        attentionUs = _nowUs() - attentionStartUs;
    }
    const uint64_t rankStartUs = profileDetail ? _nowUs() : 0;
    auto rankErr = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, attnLen, false);
    if (rankErr != NO_ERROR) {
        return rankErr;
    }
    if (profileDetail) {
        runtime->commandQueue().finish();
        rankUs = _nowUs() - rankStartUs;
    }
    if (profile) {
        runtime->commandQueue().finish();
        if (profileDetail) {
            MNN_PRINT("OpenCLPagedAttention profile op=decode_causal_attention_hd128_identity layer=%d "
                      "query=%d input_query=%d full_q=0 kv_len=%d lane=%u identity_slot=1 slot_identity=1 "
                      "dense_kv_work=%llu causal_kv_work=%llu us=%llu append_us=0 attention_us=%llu "
                      "rank_us=%llu attention_only=%d record_queue=0 "
                      "decode_key_ready_hit=0 decode_key_prepared=0 decode_prepare_inside_decode=0 "
                      "decode_prepare_us=0 prepare_us=0 decode_key_required=0 prefix_stable=1\n",
                      layerIndex, attnLen, mQuerySeqLen, kvLen, lanes,
                      static_cast<unsigned long long>(denseWork),
                      static_cast<unsigned long long>(causalWork),
                      static_cast<unsigned long long>(_nowUs() - startUs),
                      static_cast<unsigned long long>(attentionUs),
                      static_cast<unsigned long long>(rankUs),
                      _decodeIdentityAttentionOnlyBenchEnabled() ? 1 : 0);
        } else {
            MNN_PRINT("OpenCLPagedAttention profile op=decode_causal_attention_hd128_identity layer=%d "
                      "query=%d input_query=%d full_q=0 kv_len=%d lane=%u identity_slot=1 slot_identity=1 "
                      "dense_kv_work=%llu causal_kv_work=%llu us=%llu append_us=0 attention_us=%llu "
                      "rank_us=0 attention_only=%d record_queue=0 "
                      "decode_key_ready_hit=0 decode_key_prepared=0 decode_prepare_inside_decode=0 "
                      "decode_prepare_us=0 prepare_us=0 decode_key_required=0 prefix_stable=1\n",
                      layerIndex, attnLen, mQuerySeqLen, kvLen, lanes,
                      static_cast<unsigned long long>(denseWork),
                      static_cast<unsigned long long>(causalWork),
                      static_cast<unsigned long long>(_nowUs() - startUs),
                      static_cast<unsigned long long>(_nowUs() - startUs),
                      _decodeIdentityAttentionOnlyBenchEnabled() ? 1 : 0);
        }
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128IdentityFusedKV(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int baseLogical, int layerIndex) {
    if (attnLen <= 0 || kvLen <= 0 || mHeadDim != 128 || mCache == nullptr ||
        !mCache->key || !mCache->value || inputs.size() < 3) {
        return INVALID_VALUE;
    }
    auto err = ensureDecodeCausalKernelHD128Identity();
    if (err != NO_ERROR) {
        return err;
    }
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t attentionUs = 0;
    uint64_t rankUs = 0;
    uint64_t causalWork = 0;
    for (int i = 0; i < attnLen; ++i) {
        causalWork += static_cast<uint64_t>(std::max(0, std::min(kvLen, baseLogical + i + 1)));
    }
    const uint64_t causalWorkPerRow = attnLen > 0 ? causalWork / static_cast<uint64_t>(attnLen) : 0;
    const uint32_t lanes = _decodeHD128LaneWidth(causalWorkPerRow);
    auto kernel = lanes == 128u ? mDecodeCausalKernelHD128IdentityFusedKVRow128 :
        (lanes == 64u ? mDecodeCausalKernelHD128IdentityFusedKVRow64 : mDecodeCausalKernelHD128IdentityFusedKVRow32);
    if (!kernel) {
        return INVALID_VALUE;
    }
    std::vector<uint32_t> gws = {
        lanes,
        static_cast<uint32_t>(attnLen),
        static_cast<uint32_t>(mNumHead * mBatch),
    };
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    ret |= kernel->get().setArg(idx++, gws[0]);
    ret |= kernel->get().setArg(idx++, gws[1]);
    ret |= kernel->get().setArg(idx++, gws[2]);
    ret |= kernel->get().setArg(idx++, openCLBuffer(query));
    ret |= kernel->get().setArg(idx++, openCLBuffer(key));
    ret |= kernel->get().setArg(idx++, openCLBuffer(value));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= kernel->get().setArg(idx++, mScale);
    ret |= kernel->get().setArg(idx++, mBatch);
    ret |= kernel->get().setArg(idx++, mQuerySeqLen);
    ret |= kernel->get().setArg(idx++, attnLen);
    ret |= kernel->get().setArg(idx++, baseLogical);
    ret |= kernel->get().setArg(idx++, kvLen);
    ret |= kernel->get().setArg(idx++, mCache->maxSlots);
    ret |= kernel->get().setArg(idx++, mNumHead);
    ret |= kernel->get().setArg(idx++, mKvNumHead);
    ret |= kernel->get().setArg(idx++, mHeadDim);
    MNN_CHECK_CL_SUCCESS(ret, lanes == 128u ? "setArg decode_causal_attention_hd128_identity_fused_kv_row128" :
        (lanes == 64u ? "setArg decode_causal_attention_hd128_identity_fused_kv_row64"
                      : "setArg decode_causal_attention_hd128_identity_fused_kv_row32"));
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const uint64_t attentionStartUs = profileDetail ? _nowUs() : 0;
    const uint64_t denseWork = static_cast<uint64_t>(attnLen) * static_cast<uint64_t>(kvLen);
    const std::vector<uint32_t> lws = {lanes, 1u, 1u};
    const auto pmcMeta = _makeOpenCLPmcMeta("decode_causal_attention_hd128_identity_fused_kv", "attention",
                                           layerIndex, attnLen, mQuerySeqLen, kvLen, baseLogical,
                                           static_cast<int>(lanes), 0, mNumHead, mKvNumHead, mHeadDim,
                                           gws, lws, denseWork, causalWork);
    _run3DKernelDefaultPmc(kernel, gws, lws, runtime, pmcMeta);
    if (profileDetail) {
        runtime->commandQueue().finish();
        attentionUs = _nowUs() - attentionStartUs;
    }
    const uint64_t rankStartUs = profileDetail ? _nowUs() : 0;
    auto rankErr = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, attnLen, false);
    if (rankErr != NO_ERROR) {
        return rankErr;
    }
    if (profileDetail) {
        runtime->commandQueue().finish();
        rankUs = _nowUs() - rankStartUs;
    }
    if (profile) {
        runtime->commandQueue().finish();
        const uint64_t totalUs = _nowUs() - startUs;
        MNN_PRINT("OpenCLPagedAttention profile op=decode_causal_attention_hd128_identity_fused_kv layer=%d "
                  "query=%d input_query=%d full_q=0 kv_len=%d lane=%u identity_slot=1 slot_identity=1 "
                  "dense_kv_work=%llu causal_kv_work=%llu us=%llu append_us=0 attention_us=%llu "
                  "rank_us=%llu attention_only=0 record_queue=0 "
                  "decode_key_ready_hit=0 decode_key_prepared=0 decode_prepare_inside_decode=0 "
                  "decode_prepare_us=0 prepare_us=0 decode_key_required=0 prefix_stable=1\n",
                  layerIndex, attnLen, mQuerySeqLen, kvLen, lanes,
                  static_cast<unsigned long long>(denseWork),
                  static_cast<unsigned long long>(causalWork),
                  static_cast<unsigned long long>(totalUs),
                  static_cast<unsigned long long>(profileDetail ? attentionUs : totalUs),
                  static_cast<unsigned long long>(profileDetail ? rankUs : 0));
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128IdentityFusedKVGQA(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int baseLogical, int layerIndex) {
    if (attnLen != 1 || kvLen <= 0 || mHeadDim != 128 || mCache == nullptr ||
        !mCache->key || !mCache->value || inputs.size() < 3 || mKvNumHead <= 0 ||
        mNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    const int groupSize = mNumHead / mKvNumHead;
    // Keep this first grouped decode path on the small GQA groups used by the target models.
    if (groupSize <= 1 || groupSize > 8) {
        return INVALID_VALUE;
    }
    auto err = ensureDecodeCausalKernelHD128Identity();
    if (err != NO_ERROR) {
        return err;
    }
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t attentionUs = 0;
    uint64_t rankUs = 0;
    uint64_t causalWork = 0;
    for (int i = 0; i < attnLen; ++i) {
        causalWork += static_cast<uint64_t>(std::max(0, std::min(kvLen, baseLogical + i + 1)));
    }
    const uint64_t causalWorkPerRow = attnLen > 0 ? causalWork / static_cast<uint64_t>(attnLen) : 0;
    const uint32_t lanes = _decodeHD128LaneWidth(causalWorkPerRow);
    auto kernel = lanes == 128u ? mDecodeCausalKernelHD128IdentityFusedKVGQARow128 :
        (lanes == 64u ? mDecodeCausalKernelHD128IdentityFusedKVGQARow64
                      : mDecodeCausalKernelHD128IdentityFusedKVGQARow32);
    if (!kernel) {
        return INVALID_VALUE;
    }
    std::vector<uint32_t> gws = {
        lanes,
        static_cast<uint32_t>(attnLen),
        static_cast<uint32_t>(mKvNumHead * mBatch),
    };
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    ret |= kernel->get().setArg(idx++, gws[0]);
    ret |= kernel->get().setArg(idx++, gws[1]);
    ret |= kernel->get().setArg(idx++, gws[2]);
    ret |= kernel->get().setArg(idx++, openCLBuffer(query));
    ret |= kernel->get().setArg(idx++, openCLBuffer(key));
    ret |= kernel->get().setArg(idx++, openCLBuffer(value));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= kernel->get().setArg(idx++, mScale);
    ret |= kernel->get().setArg(idx++, mBatch);
    ret |= kernel->get().setArg(idx++, mQuerySeqLen);
    ret |= kernel->get().setArg(idx++, attnLen);
    ret |= kernel->get().setArg(idx++, baseLogical);
    ret |= kernel->get().setArg(idx++, kvLen);
    ret |= kernel->get().setArg(idx++, mCache->maxSlots);
    ret |= kernel->get().setArg(idx++, mNumHead);
    ret |= kernel->get().setArg(idx++, mKvNumHead);
    ret |= kernel->get().setArg(idx++, mHeadDim);
    MNN_CHECK_CL_SUCCESS(ret, lanes == 128u ? "setArg decode_causal_attention_hd128_identity_fused_kv_gqa_row128" :
        (lanes == 64u ? "setArg decode_causal_attention_hd128_identity_fused_kv_gqa_row64"
                      : "setArg decode_causal_attention_hd128_identity_fused_kv_gqa_row32"));
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const uint64_t attentionStartUs = profileDetail ? _nowUs() : 0;
    const uint64_t denseWork = static_cast<uint64_t>(attnLen) * static_cast<uint64_t>(kvLen);
    const std::vector<uint32_t> lws = {lanes, 1u, 1u};
    const auto pmcMeta = _makeOpenCLPmcMeta("decode_causal_attention_hd128_identity_fused_kv_gqa", "attention",
                                           layerIndex, attnLen, mQuerySeqLen, kvLen, baseLogical,
                                           static_cast<int>(lanes), 0, mNumHead, mKvNumHead, mHeadDim,
                                           gws, lws, denseWork, causalWork);
    _run3DKernelDefaultPmc(kernel, gws, lws, runtime, pmcMeta);
    if (profileDetail) {
        runtime->commandQueue().finish();
        attentionUs = _nowUs() - attentionStartUs;
    }
    const uint64_t rankStartUs = profileDetail ? _nowUs() : 0;
    auto rankErr = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, attnLen, false);
    if (rankErr != NO_ERROR) {
        return rankErr;
    }
    if (profileDetail) {
        runtime->commandQueue().finish();
        rankUs = _nowUs() - rankStartUs;
    }
    if (profile) {
        runtime->commandQueue().finish();
        const uint64_t totalUs = _nowUs() - startUs;
        MNN_PRINT("OpenCLPagedAttention profile op=decode_causal_attention_hd128_identity_fused_kv_gqa layer=%d "
                  "query=%d input_query=%d full_q=0 kv_len=%d lane=%u identity_slot=1 slot_identity=1 "
                  "group_size=%d kv_heads=%d dense_kv_work=%llu causal_kv_work=%llu us=%llu "
                  "append_us=0 attention_us=%llu rank_us=%llu attention_only=0 record_queue=0 "
                  "decode_key_ready_hit=0 decode_key_prepared=0 decode_prepare_inside_decode=0 "
                  "decode_prepare_us=0 prepare_us=0 decode_key_required=0 prefix_stable=1\n",
                  layerIndex, attnLen, mQuerySeqLen, kvLen, lanes, groupSize, mKvNumHead,
                  static_cast<unsigned long long>(denseWork),
                  static_cast<unsigned long long>(causalWork),
                  static_cast<unsigned long long>(totalUs),
                  static_cast<unsigned long long>(profileDetail ? attentionUs : totalUs),
                  static_cast<unsigned long long>(profileDetail ? rankUs : 0));
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128TransposedKFusedKVGQA(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int baseLogical, int layerIndex) {
    if (attnLen != 1 || kvLen <= 0 || baseLogical < 0 || mHeadDim != 128 || mCache == nullptr ||
        !mCache->key || !mCache->decodeKey || !mCache->value ||
        inputs.size() < 3 || mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    const int groupSize = mNumHead / mKvNumHead;
    if (groupSize <= 1 || groupSize > 8) {
        return INVALID_VALUE;
    }
    auto err = ensureDecodeTransposedKKernel();
    if (err != NO_ERROR) {
        return err;
    }
    err = ensureDecodeKeyReady(baseLogical, true);
    if (err != NO_ERROR) {
        return err;
    }

    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t attentionUs = 0;
    uint64_t rankUs = 0;
    uint64_t causalWork = 0;
    for (int i = 0; i < attnLen; ++i) {
        causalWork += static_cast<uint64_t>(std::max(0, std::min(kvLen, baseLogical + i + 1)));
    }
    const uint64_t causalWorkPerRow = attnLen > 0 ? causalWork / static_cast<uint64_t>(attnLen) : 0;
    const uint32_t lanes = _decodeHD128LaneWidth(causalWorkPerRow);
    auto kernel = lanes == 128u ? mDecodeCausalKernelHD128TransposedKFusedKVGQARow128 :
        (lanes == 64u ? mDecodeCausalKernelHD128TransposedKFusedKVGQARow64
                      : mDecodeCausalKernelHD128TransposedKFusedKVGQARow32);
    if (!kernel) {
        return INVALID_VALUE;
    }
    std::vector<uint32_t> gws = {
        lanes,
        static_cast<uint32_t>(attnLen),
        static_cast<uint32_t>(mKvNumHead * mBatch),
    };
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    ret |= kernel->get().setArg(idx++, gws[0]);
    ret |= kernel->get().setArg(idx++, gws[1]);
    ret |= kernel->get().setArg(idx++, gws[2]);
    ret |= kernel->get().setArg(idx++, openCLBuffer(query));
    ret |= kernel->get().setArg(idx++, openCLBuffer(key));
    ret |= kernel->get().setArg(idx++, openCLBuffer(value));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= kernel->get().setArg(idx++, mScale);
    ret |= kernel->get().setArg(idx++, mBatch);
    ret |= kernel->get().setArg(idx++, mQuerySeqLen);
    ret |= kernel->get().setArg(idx++, attnLen);
    ret |= kernel->get().setArg(idx++, baseLogical);
    ret |= kernel->get().setArg(idx++, kvLen);
    ret |= kernel->get().setArg(idx++, mCache->maxSlots);
    ret |= kernel->get().setArg(idx++, mNumHead);
    ret |= kernel->get().setArg(idx++, mKvNumHead);
    ret |= kernel->get().setArg(idx++, mHeadDim);
    MNN_CHECK_CL_SUCCESS(ret, lanes == 128u ?
        "setArg decode_causal_attention_hd128_transposed_k_fused_kv_gqa_row128" :
        (lanes == 64u ? "setArg decode_causal_attention_hd128_transposed_k_fused_kv_gqa_row64"
                      : "setArg decode_causal_attention_hd128_transposed_k_fused_kv_gqa_row32"));
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const uint64_t attentionStartUs = profileDetail ? _nowUs() : 0;
    const uint64_t denseWork = static_cast<uint64_t>(attnLen) * static_cast<uint64_t>(kvLen);
    const std::vector<uint32_t> lws = {lanes, 1u, 1u};
    const auto pmcMeta = _makeOpenCLPmcMeta("decode_causal_attention_hd128_transposed_k_fused_kv_gqa",
                                           "attention", layerIndex, attnLen, mQuerySeqLen, kvLen,
                                           baseLogical, static_cast<int>(lanes), 0, mNumHead,
                                           mKvNumHead, mHeadDim, gws, lws, denseWork, causalWork);
    _run3DKernelDefaultPmc(kernel, gws, lws, runtime, pmcMeta);
    if (profileDetail) {
        runtime->commandQueue().finish();
        attentionUs = _nowUs() - attentionStartUs;
    }
    mCache->decodeKeyReadyLength = std::max(mCache->decodeKeyReadyLength, baseLogical + 1);
    const uint64_t rankStartUs = profileDetail ? _nowUs() : 0;
    auto rankErr = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, attnLen, false);
    if (rankErr != NO_ERROR) {
        return rankErr;
    }
    if (profileDetail) {
        runtime->commandQueue().finish();
        rankUs = _nowUs() - rankStartUs;
    }
    if (profile) {
        runtime->commandQueue().finish();
        const uint64_t totalUs = _nowUs() - startUs;
        MNN_PRINT("OpenCLPagedAttention profile op=decode_causal_attention_hd128_transposed_k_fused_kv_gqa layer=%d "
                  "query=%d input_query=%d kv_len=%d lane=%u group_size=%d kv_heads=%d "
                  "dense_kv_work=%llu causal_kv_work=%llu us=%llu append_us=0 attention_us=%llu "
                  "rank_us=%llu decode_key_ready_hit=%d decode_key_prepared=%d "
                  "decode_prepare_inside_decode=%d decode_prepare_us=%llu prepare_us=%llu "
                  "decode_key_required=%d slot_identity=%d prefix_stable=%d record_queue=0\n",
                  layerIndex, attnLen, mQuerySeqLen, kvLen, lanes, groupSize, mKvNumHead,
                  static_cast<unsigned long long>(denseWork),
                  static_cast<unsigned long long>(causalWork),
                  static_cast<unsigned long long>(totalUs),
                  static_cast<unsigned long long>(profileDetail ? attentionUs : totalUs),
                  static_cast<unsigned long long>(profileDetail ? rankUs : 0),
                  mLastDecodeKeyReadyHit ? 1 : 0,
                  mLastDecodeKeyPrepared ? 1 : 0,
                  mLastDecodeKeyPrepareInsideDecode ? 1 : 0,
                  static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                  static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                  mLastDecodeKeyRequiredLen,
                  mLastDecodeKeySlotIdentity ? 1 : 0,
                  mLastDecodeKeyPrefixStable ? 1 : 0);
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128TransposedKFusedKV(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int baseLogical, int layerIndex) {
    if (attnLen != 1 || kvLen <= 0 || baseLogical < 0 || mHeadDim != 128 || mCache == nullptr ||
        !mCache->key || !mCache->decodeKey || !mCache->value || inputs.size() < 3 ||
        mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    auto err = ensureDecodeTransposedKKernel();
    if (err != NO_ERROR) {
        return err;
    }
    err = ensureDecodeKeyReady(baseLogical, true);
    if (err != NO_ERROR) {
        return err;
    }

    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const bool profile = _profilePagedAttention();
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t causalWork = 0;
    for (int i = 0; i < attnLen; ++i) {
        causalWork += static_cast<uint64_t>(std::max(0, std::min(kvLen, baseLogical + i + 1)));
    }
    const uint64_t causalWorkPerRow = attnLen > 0 ? causalWork / static_cast<uint64_t>(attnLen) : 0;
    const uint32_t lanes = _decodeHD128LaneWidth(causalWorkPerRow);
    auto kernel = lanes == 128u ? mDecodeCausalKernelHD128TransposedKFusedKVRow128 :
        (lanes == 64u ? mDecodeCausalKernelHD128TransposedKFusedKVRow64
                      : mDecodeCausalKernelHD128TransposedKFusedKVRow32);
    if (!kernel) {
        return INVALID_VALUE;
    }
    if (mOpenCLBackend->isUseRecordQueue() && !profile &&
        (mMeta == nullptr || !mMeta->needsPicDecodeAttentionRankCapture(layerIndex))) {
        auto recordErr = runDecodeCausalAttentionHD128TransposedKFusedKVRecord(
            inputs, outputs, kvLen, attnLen, baseLogical, layerIndex, lanes, kernel);
        if (recordErr == NO_ERROR) {
            return NO_ERROR;
        }
    }
    std::vector<uint32_t> gws = {
        lanes,
        static_cast<uint32_t>(attnLen),
        static_cast<uint32_t>(mNumHead * mBatch),
    };
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    ret |= kernel->get().setArg(idx++, gws[0]);
    ret |= kernel->get().setArg(idx++, gws[1]);
    ret |= kernel->get().setArg(idx++, gws[2]);
    ret |= kernel->get().setArg(idx++, openCLBuffer(query));
    ret |= kernel->get().setArg(idx++, openCLBuffer(key));
    ret |= kernel->get().setArg(idx++, openCLBuffer(value));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= kernel->get().setArg(idx++, mScale);
    ret |= kernel->get().setArg(idx++, mBatch);
    ret |= kernel->get().setArg(idx++, mQuerySeqLen);
    ret |= kernel->get().setArg(idx++, attnLen);
    ret |= kernel->get().setArg(idx++, baseLogical);
    ret |= kernel->get().setArg(idx++, kvLen);
    ret |= kernel->get().setArg(idx++, mCache->maxSlots);
    ret |= kernel->get().setArg(idx++, mNumHead);
    ret |= kernel->get().setArg(idx++, mKvNumHead);
    ret |= kernel->get().setArg(idx++, mHeadDim);
    MNN_CHECK_CL_SUCCESS(ret, lanes == 128u ?
        "setArg decode_causal_attention_hd128_transposed_k_fused_kv_row128" :
        (lanes == 64u ? "setArg decode_causal_attention_hd128_transposed_k_fused_kv_row64"
                      : "setArg decode_causal_attention_hd128_transposed_k_fused_kv_row32"));
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    uint64_t attentionUs = 0;
    uint64_t rankUs = 0;
    const uint64_t attentionStartUs = profileDetail ? _nowUs() : 0;
    const uint64_t denseWork = static_cast<uint64_t>(attnLen) * static_cast<uint64_t>(kvLen);
    const std::vector<uint32_t> lws = {lanes, 1u, 1u};
    const auto pmcMeta = _makeOpenCLPmcMeta("decode_causal_attention_hd128_transposed_k_fused_kv",
                                           "attention", layerIndex, attnLen, mQuerySeqLen, kvLen,
                                           baseLogical, static_cast<int>(lanes), 0, mNumHead,
                                           mKvNumHead, mHeadDim, gws, lws, denseWork, causalWork);
    _run3DKernelDefaultPmc(kernel, gws, lws, runtime, pmcMeta);
    if (profileDetail) {
        runtime->commandQueue().finish();
        attentionUs = _nowUs() - attentionStartUs;
    }
    mCache->decodeKeyReadyLength = std::max(mCache->decodeKeyReadyLength, baseLogical + 1);
    const uint64_t rankStartUs = profileDetail ? _nowUs() : 0;
    auto rankErr = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, attnLen, false);
    if (rankErr != NO_ERROR) {
        return rankErr;
    }
    if (profileDetail) {
        runtime->commandQueue().finish();
        rankUs = _nowUs() - rankStartUs;
    }

    if (profile) {
        runtime->commandQueue().finish();
        const uint64_t totalUs = _nowUs() - startUs;
        if (!profileDetail) {
            attentionUs = totalUs;
        }
        MNN_PRINT("OpenCLPagedAttention profile op=decode_causal_attention_hd128_transposed_k_fused_kv layer=%d "
                  "query=%d input_query=%d kv_len=%d lane=%u identity_slot=1 "
                  "dense_kv_work=%llu causal_kv_work=%llu us=%llu append_us=0 attention_us=%llu rank_us=%llu "
                  "decode_key_ready_hit=%d decode_key_prepared=%d "
                  "decode_prepare_inside_decode=%d decode_prepare_us=%llu prepare_us=%llu "
                  "decode_key_required=%d slot_identity=%d prefix_stable=%d record_queue=0 append_fused=1\n",
                  layerIndex, attnLen, mQuerySeqLen, kvLen, lanes,
                  static_cast<unsigned long long>(denseWork),
                  static_cast<unsigned long long>(causalWork),
                  static_cast<unsigned long long>(totalUs),
                  static_cast<unsigned long long>(attentionUs),
                  static_cast<unsigned long long>(rankUs),
                  mLastDecodeKeyReadyHit ? 1 : 0,
                  mLastDecodeKeyPrepared ? 1 : 0,
                  mLastDecodeKeyPrepareInsideDecode ? 1 : 0,
                  static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                  static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                  mLastDecodeKeyRequiredLen,
                  mLastDecodeKeySlotIdentity ? 1 : 0,
                  mLastDecodeKeyPrefixStable ? 1 : 0);
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128TransposedKFusedKVRecord(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int baseLogical, int layerIndex, uint32_t lanes, std::shared_ptr<KernelWrap> kernel) {
    if (!mOpenCLBackend->isUseRecordQueue() || _profilePagedAttention() || !kernel ||
        attnLen != 1 || kvLen <= 0 || mCache == nullptr || !mCache->key || !mCache->decodeKey ||
        !mCache->value || inputs.size() < 3 || outputs.empty() ||
        mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0 ||
        (mMeta != nullptr && mMeta->needsPicDecodeAttentionRankCapture(layerIndex))) {
        return INVALID_VALUE;
    }
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    const uint32_t heads = static_cast<uint32_t>(mNumHead * mBatch);
    const int groupSize = mNumHead / mKvNumHead;
    const bool needRecord = !mDecodeTransposedFusedRecordValid ||
        mDecodeTransposedFusedRecordLanes != lanes ||
        mDecodeTransposedFusedRecordAttnLen != attnLen ||
        mDecodeTransposedFusedRecordHeads != heads ||
        mDecodeTransposedFusedRecordGroupSize != groupSize;

    auto updateRecordArgs = [&]() {
        auto& args = mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args;
        if (args.size() < 12) {
            return false;
        }
        mDecodeTransposedFusedRecordQuerySeqLen = mQuerySeqLen;
        mDecodeTransposedFusedRecordBaseLogical = baseLogical;
        mDecodeTransposedFusedRecordKvLen = kvLen;
        mDecodeTransposedFusedRecordMaxSlots = mCache->maxSlots;
        mDecodeTransposedFusedRecordScale = mScale;
        args[0].arg_value = &openCLBuffer(query)();
        args[1].arg_value = &openCLBuffer(key)();
        args[2].arg_value = &openCLBuffer(value)();
        args[3].arg_value = &openCLBuffer(mCache->key.get())();
        args[4].arg_value = &openCLBuffer(mCache->value.get())();
        args[5].arg_value = &openCLBuffer(mCache->decodeKey.get())();
        args[6].arg_value = &openCLBuffer(output)();
        args[7].arg_value = &mDecodeTransposedFusedRecordScale;
        args[8].arg_value = &mDecodeTransposedFusedRecordQuerySeqLen;
        args[9].arg_value = &mDecodeTransposedFusedRecordBaseLogical;
        args[10].arg_value = &mDecodeTransposedFusedRecordKvLen;
        args[11].arg_value = &mDecodeTransposedFusedRecordMaxSlots;
        return true;
    };

    if (needRecord) {
        mDecodeTransposedFusedRecordGws0 = lanes;
        mDecodeTransposedFusedRecordGws1 = static_cast<uint32_t>(attnLen);
        mDecodeTransposedFusedRecordGws2 = heads;
        mDecodeTransposedFusedRecordQuerySeqLen = mQuerySeqLen;
        mDecodeTransposedFusedRecordBaseLogical = baseLogical;
        mDecodeTransposedFusedRecordKvLen = kvLen;
        mDecodeTransposedFusedRecordMaxSlots = mCache->maxSlots;
        mDecodeTransposedFusedRecordScale = mScale;

        std::vector<uint32_t> gws = {mDecodeTransposedFusedRecordGws0,
                                     mDecodeTransposedFusedRecordGws1,
                                     mDecodeTransposedFusedRecordGws2};
        std::vector<uint32_t> lws = {lanes, 1u, 1u};
        cl_int ret = CL_SUCCESS;
        uint32_t idx = 0;
        ret |= kernel->get().setArg(idx++, gws[0]);
        ret |= kernel->get().setArg(idx++, gws[1]);
        ret |= kernel->get().setArg(idx++, gws[2]);
        ret |= kernel->get().setArg(idx++, openCLBuffer(query));
        ret |= kernel->get().setArg(idx++, openCLBuffer(key));
        ret |= kernel->get().setArg(idx++, openCLBuffer(value));
        ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
        ret |= kernel->get().setArg(idx++, openCLBuffer(output));
        ret |= kernel->get().setArg(idx++, mDecodeTransposedFusedRecordScale);
        ret |= kernel->get().setArg(idx++, mBatch);
        ret |= kernel->get().setArg(idx++, mDecodeTransposedFusedRecordQuerySeqLen);
        ret |= kernel->get().setArg(idx++, attnLen);
        ret |= kernel->get().setArg(idx++, mDecodeTransposedFusedRecordBaseLogical);
        ret |= kernel->get().setArg(idx++, mDecodeTransposedFusedRecordKvLen);
        ret |= kernel->get().setArg(idx++, mDecodeTransposedFusedRecordMaxSlots);
        ret |= kernel->get().setArg(idx++, mNumHead);
        ret |= kernel->get().setArg(idx++, mKvNumHead);
        ret |= kernel->get().setArg(idx++, mHeadDim);
        MNN_CHECK_CL_SUCCESS(ret, lanes == 128u ?
            "record setArg decode_causal_attention_hd128_transposed_k_fused_kv_row128" :
            (lanes == 64u ? "record setArg decode_causal_attention_hd128_transposed_k_fused_kv_row64"
                          : "record setArg decode_causal_attention_hd128_transposed_k_fused_kv_row32"));
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }

        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.clear();
        mDecodeTransposedFusedRecordUpdateInfo.update_global_size.clear();
        mDecodeTransposedFusedRecordUpdateInfo.update_local_size.clear();
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 3, sizeof(cl_mem), &openCLBuffer(query)()});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 4, sizeof(cl_mem), &openCLBuffer(key)()});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 5, sizeof(cl_mem), &openCLBuffer(value)()});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 6, sizeof(cl_mem), &openCLBuffer(mCache->key.get())()});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 7, sizeof(cl_mem), &openCLBuffer(mCache->value.get())()});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 8, sizeof(cl_mem), &openCLBuffer(mCache->decodeKey.get())()});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 9, sizeof(cl_mem), &openCLBuffer(output)()});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 10, sizeof(mDecodeTransposedFusedRecordScale), &mDecodeTransposedFusedRecordScale});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 12, sizeof(mDecodeTransposedFusedRecordQuerySeqLen),
             &mDecodeTransposedFusedRecordQuerySeqLen});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 14, sizeof(mDecodeTransposedFusedRecordBaseLogical),
             &mDecodeTransposedFusedRecordBaseLogical});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 15, sizeof(mDecodeTransposedFusedRecordKvLen), &mDecodeTransposedFusedRecordKvLen});
        mDecodeTransposedFusedRecordUpdateInfo.update_kernel_args.push_back(
            {0, 16, sizeof(mDecodeTransposedFusedRecordMaxSlots), &mDecodeTransposedFusedRecordMaxSlots});
        mDecodeTransposedFusedRecordUpdateInfos.clear();
        mDecodeTransposedFusedRecordUpdateInfos.emplace_back(&mDecodeTransposedFusedRecordUpdateInfo);

        mOpenCLBackend->startRecord(mRecording);
        mOpenCLBackend->recordKernel3d(kernel, gws, lws, &mDecodeTransposedFusedRecordUpdateInfo);
        mOpenCLBackend->endRecord(mRecording);
        mDecodeTransposedFusedRecordValid = true;
        mDecodeTransposedFusedRecordLanes = lanes;
        mDecodeTransposedFusedRecordAttnLen = attnLen;
        mDecodeTransposedFusedRecordHeads = heads;
        mDecodeTransposedFusedRecordGroupSize = groupSize;
    }
    if (!updateRecordArgs()) {
        return INVALID_VALUE;
    }
    mOpenCLBackend->addRecord(mRecording, mDecodeTransposedFusedRecordUpdateInfos);
    mCache->decodeKeyReadyLength = std::max(mCache->decodeKeyReadyLength, baseLogical + 1);
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::appendDecodeKeyValueHD128(const std::vector<Tensor*>& inputs,
                                                                int baseLogical, bool profileDetail,
                                                                uint64_t* appendUs) {
    if (baseLogical < 0 || mHeadDim != 128 || mCache == nullptr || !mCache->key || !mCache->decodeKey ||
        !mCache->value || !mDecodeKeyAppendKernel || inputs.size() < 3 ||
        mKvNumHead <= 0) {
        return INVALID_VALUE;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (runtime == nullptr) {
        return INVALID_VALUE;
    }
    auto key = inputs[1];
    auto value = inputs[2];
    auto& queue = runtime->commandQueue();
    const uint64_t appendStartUs = profileDetail ? _nowUs() : 0;
    const uint32_t appendGws0 = 128u;
    const uint32_t appendGws1 = static_cast<uint32_t>(mBatch * mKvNumHead);
    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, appendGws0);
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, appendGws1);
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, openCLBuffer(key));
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, openCLBuffer(value));
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, mBatch);
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, mKvNumHead);
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, mHeadDim);
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, 0);
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, baseLogical);
    ret |= mDecodeKeyAppendKernel->get().setArg(idx++, mCache->maxSlots);
    MNN_CHECK_CL_SUCCESS(ret, "setArg append_decode_key_value_hd128");
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const std::vector<uint32_t> appendGws = {appendGws0, appendGws1};
    const std::vector<uint32_t> appendLws = {32u, 1u};
    const int layerIndex = mLayerIndex >= 0 ? mLayerIndex : (mMeta != nullptr ? mMeta->layer_index : -1);
    const auto pmcMeta = _makeOpenCLPmcMeta("append_decode_key_value_hd128", "append", layerIndex,
                                           1, mQuerySeqLen, baseLogical + 1, baseLogical, 32, 0,
                                           mNumHead, mKvNumHead, mHeadDim, appendGws, appendLws,
                                           0, 0, 1, 0, false);
    _runOpenCLPmcScope(runtime, pmcMeta, [&](cl::Event* eventPtr) {
        ret = queue.enqueueNDRangeKernel(mDecodeKeyAppendKernel->get(), cl::NullRange,
                                         cl::NDRange(appendGws0, appendGws1), cl::NDRange(32, 1),
                                         nullptr, eventPtr);
    });
    MNN_CHECK_CL_SUCCESS(ret, "enqueue append_decode_key_value_hd128");
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    mCache->decodeKeyReadyLength = std::max(mCache->decodeKeyReadyLength, baseLogical + 1);
    if (profileDetail) {
        queue.finish();
        if (appendUs != nullptr) {
            *appendUs = _nowUs() - appendStartUs;
        }
    } else if (appendUs != nullptr) {
        *appendUs = 0;
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128SplitProfile(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int baseLogical, int layerIndex, bool transposedK) {
    if (!_profilePagedAttention() || !_decodeQ1SplitProfileEnabled() || attnLen != 1 || kvLen <= 0 ||
        baseLogical + 1 != kvLen || mHeadDim != 128 || mCache == nullptr || !mCache->key ||
        !mCache->value || inputs.empty() || outputs.empty() || mKvNumHead <= 0 || mNumHead <= 0 ||
        mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    auto err = ensureDecodeTransposedKKernel();
    if (err != NO_ERROR) {
        return err;
    }
    if (!transposedK) {
        err = ensureDecodeCausalKernelHD128Identity();
        if (err != NO_ERROR) {
            return err;
        }
    } else if (!mCache->decodeKey || mCache->decodeKeyReadyLength < kvLen) {
        return INVALID_VALUE;
    }
    err = ensureDecodeTransposedTemps(kvLen);
    if (err != NO_ERROR) {
        return err;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (runtime == nullptr) {
        return INVALID_VALUE;
    }
    auto query = inputs[0];
    auto output = outputs[0];
    auto qkKernel = transposedK ? mDecodeQKTransposedKernel : mDecodeQKIdentityKernel;
    if (!query || !output || !qkKernel || !mDecodeQKVTransposedKernel || !mDecodeSoftmaxKernel) {
        return INVALID_VALUE;
    }
    auto& qkBuffer = openCLDeferBuffer(mTempQK.get());
    auto& softmaxBuffer = openCLDeferBuffer(mTempSoftmax.get());

    auto& queue = runtime->commandQueue();
    const uint64_t startUs = _nowUs();
    uint64_t qkUs = 0;
    uint64_t softmaxUs = 0;
    uint64_t qkvUs = 0;
    uint64_t rankUs = 0;
    const uint32_t outside = static_cast<uint32_t>(mBatch * mNumHead);
    const uint64_t denseWork = static_cast<uint64_t>(attnLen) * static_cast<uint64_t>(kvLen);
    const uint64_t causalWork = static_cast<uint64_t>(kvLen);

    {
        std::vector<uint32_t> gws = {static_cast<uint32_t>((kvLen + 3) / 4), outside};
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= qkKernel->get().setArg(idx++, gws[0]);
        ret |= qkKernel->get().setArg(idx++, gws[1]);
        ret |= qkKernel->get().setArg(idx++, openCLBuffer(query));
        ret |= qkKernel->get().setArg(
            idx++, transposedK ? openCLBuffer(mCache->decodeKey.get()) : openCLBuffer(mCache->key.get()));
        ret |= qkKernel->get().setArg(idx++, qkBuffer);
        ret |= qkKernel->get().setArg(idx++, mScale);
        ret |= qkKernel->get().setArg(idx++, mBatch);
        ret |= qkKernel->get().setArg(idx++, mQuerySeqLen);
        ret |= qkKernel->get().setArg(idx++, kvLen);
        ret |= qkKernel->get().setArg(idx++, mCache->maxSlots);
        ret |= qkKernel->get().setArg(idx++, mNumHead);
        ret |= qkKernel->get().setArg(idx++, mKvNumHead);
        ret |= qkKernel->get().setArg(idx++, mHeadDim);
        MNN_CHECK_CL_SUCCESS(ret, transposedK ? "setArg matmul_qk_decode_transposed_hd128"
                                              : "setArg matmul_qk_decode_paged_identity_hd128");
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        const uint64_t opStartUs = _nowUs();
        ret = queue.enqueueNDRangeKernel(qkKernel->get(), cl::NullRange, cl::NDRange(gws[0], gws[1]),
                                         cl::NullRange);
        MNN_CHECK_CL_SUCCESS(ret, transposedK ? "enqueue matmul_qk_decode_transposed_hd128"
                                              : "enqueue matmul_qk_decode_paged_identity_hd128");
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        queue.finish();
        qkUs = _nowUs() - opStartUs;
    }

    {
        std::vector<uint32_t> gws = {64u, 1u, outside};
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mDecodeSoftmaxKernel->get().setArg(idx++, gws[0]);
        ret |= mDecodeSoftmaxKernel->get().setArg(idx++, gws[1]);
        ret |= mDecodeSoftmaxKernel->get().setArg(idx++, gws[2]);
        ret |= mDecodeSoftmaxKernel->get().setArg(idx++, qkBuffer);
        ret |= mDecodeSoftmaxKernel->get().setArg(idx++, softmaxBuffer);
        ret |= mDecodeSoftmaxKernel->get().setArg(idx++, 1);
        ret |= mDecodeSoftmaxKernel->get().setArg(idx++, static_cast<int>(outside));
        ret |= mDecodeSoftmaxKernel->get().setArg(idx++, kvLen);
        MNN_CHECK_CL_SUCCESS(ret, "setArg decode split softmax_in1_buf");
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        const uint64_t opStartUs = _nowUs();
        run3DKernelDefault(mDecodeSoftmaxKernel, gws, {64u, 1u, 1u}, runtime);
        queue.finish();
        softmaxUs = _nowUs() - opStartUs;
    }

    {
        std::vector<uint32_t> gws = {static_cast<uint32_t>((mHeadDim + 7) / 8), outside};
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, gws[0]);
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, gws[1]);
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, softmaxBuffer);
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, openCLBuffer(output));
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, kvLen);
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, mCache->maxSlots);
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, mBatch);
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, mNumHead);
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, mKvNumHead);
        ret |= mDecodeQKVTransposedKernel->get().setArg(idx++, mHeadDim);
        MNN_CHECK_CL_SUCCESS(ret, "setArg matmul_qkv_decode_value_hd128_b8");
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        const uint64_t opStartUs = _nowUs();
        ret = queue.enqueueNDRangeKernel(mDecodeQKVTransposedKernel->get(), cl::NullRange,
                                         cl::NDRange(gws[0], gws[1]), cl::NullRange);
        MNN_CHECK_CL_SUCCESS(ret, "enqueue matmul_qkv_decode_value_hd128_b8");
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        queue.finish();
        qkvUs = _nowUs() - opStartUs;
    }

    const uint64_t rankStartUs = _nowUs();
    auto rankErr = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, attnLen, false);
    if (rankErr != NO_ERROR) {
        return rankErr;
    }
    queue.finish();
    rankUs = _nowUs() - rankStartUs;

    const uint64_t attentionUs = qkUs + softmaxUs + qkvUs;
    const uint64_t totalUs = _nowUs() - startUs;
    MNN_PRINT("OpenCLPagedAttention profile op=decode_causal_attention_hd128_%s_split layer=%d "
              "query=%d input_query=%d full_q=0 kv_len=%d lane=0 identity_slot=1 slot_identity=1 "
              "dense_kv_work=%llu causal_kv_work=%llu us=%llu append_us=0 attention_us=%llu "
              "qk_us=%llu softmax_us=%llu qkv_us=%llu qk_only_us=%llu qkv_only_us=%llu rank_us=%llu "
              "attention_only=1 record_queue=0 decode_key_ready_hit=%d decode_key_prepared=%d "
              "decode_prepare_inside_decode=%d decode_prepare_us=%llu prepare_us=%llu "
              "decode_key_required=%d prefix_stable=%d split_profile=1\n",
              transposedK ? "transposed_k" : "identity", layerIndex, attnLen, mQuerySeqLen, kvLen,
              static_cast<unsigned long long>(denseWork), static_cast<unsigned long long>(causalWork),
              static_cast<unsigned long long>(totalUs), static_cast<unsigned long long>(attentionUs),
              static_cast<unsigned long long>(qkUs), static_cast<unsigned long long>(softmaxUs),
              static_cast<unsigned long long>(qkvUs), static_cast<unsigned long long>(qkUs),
              static_cast<unsigned long long>(qkvUs), static_cast<unsigned long long>(rankUs),
              transposedK && mLastDecodeKeyReadyHit ? 1 : 0,
              transposedK && mLastDecodeKeyPrepared ? 1 : 0,
              transposedK && mLastDecodeKeyPrepareInsideDecode ? 1 : 0,
              static_cast<unsigned long long>(transposedK ? mLastDecodeKeyPrepareUs : 0),
              static_cast<unsigned long long>(transposedK ? mLastDecodeKeyPrepareUs : 0),
              transposedK ? mLastDecodeKeyRequiredLen : 0,
              (!transposedK || mLastDecodeKeyPrefixStable) ? 1 : 0);
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128TransposedKAppendReadonly(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int baseLogical, int layerIndex, bool appendCurrent) {
    if (attnLen != 1 || kvLen <= 0 || baseLogical < 0 || mHeadDim != 128 || mCache == nullptr ||
        !mCache->key || !mCache->decodeKey || !mCache->value || inputs.size() < 3 ||
        mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    auto err = ensureDecodeTransposedKKernel();
    if (err != NO_ERROR) {
        return err;
    }
    const int requiredDecodeKeyLen = appendCurrent ? baseLogical : kvLen;
    err = ensureDecodeKeyReady(requiredDecodeKeyLen, true);
    if (err != NO_ERROR) {
        return err;
    }

    auto query = inputs[0];
    auto output = outputs[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (runtime == nullptr || output == nullptr) {
        return INVALID_VALUE;
    }

    auto& queue = runtime->commandQueue();
    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t appendUs = 0;
    uint64_t attentionUs = 0;
    uint64_t rankUs = 0;

    uint64_t causalWork = 0;
    for (int i = 0; i < attnLen; ++i) {
        causalWork += static_cast<uint64_t>(std::max(0, std::min(kvLen, baseLogical + i + 1)));
    }
    const uint64_t causalWorkPerRow = attnLen > 0 ? causalWork / static_cast<uint64_t>(attnLen) : 0;
    const uint32_t lanes = _decodeHD128LaneWidth(causalWorkPerRow);
    auto kernel = lanes == 128u ? mDecodeCausalKernelHD128TransposedKReadonlyRow128 :
        (lanes == 64u ? mDecodeCausalKernelHD128TransposedKReadonlyRow64
                      : mDecodeCausalKernelHD128TransposedKReadonlyRow32);
    if (!kernel) {
        return INVALID_VALUE;
    }
    if (appendCurrent && mOpenCLBackend->isUseRecordQueue() && !profile &&
        (mMeta == nullptr || !mMeta->needsPicDecodeAttentionRankCapture(layerIndex))) {
        auto recordErr = runDecodeCausalAttentionHD128TransposedKAppendReadonlyRecord(
            inputs, outputs, kvLen, attnLen, baseLogical, layerIndex, lanes, kernel);
        if (recordErr == NO_ERROR) {
            return NO_ERROR;
        }
    }

    if (appendCurrent) {
        auto appendErr = appendDecodeKeyValueHD128(inputs, baseLogical, profileDetail, &appendUs);
        if (appendErr != NO_ERROR) {
            return appendErr;
        }
    }

    if (_decodeQ1SplitProfileEnabled() && profile) {
        auto splitErr = runDecodeCausalAttentionHD128SplitProfile(inputs, outputs, kvLen, attnLen,
                                                                  baseLogical, layerIndex, true);
        if (splitErr == NO_ERROR) {
            return NO_ERROR;
        }
        if (_picOpenCLDebug()) {
            MNN_PRINT("PIC OpenCL PA decode transposed_k split profile fallback layer=%d err=%d\n",
                      layerIndex, static_cast<int>(splitErr));
        }
    }

    const uint64_t attentionStartUs = profileDetail ? _nowUs() : 0;
    std::vector<uint32_t> gws = {
        lanes,
        static_cast<uint32_t>(attnLen),
        static_cast<uint32_t>(mNumHead * mBatch),
    };
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    ret |= kernel->get().setArg(idx++, gws[0]);
    ret |= kernel->get().setArg(idx++, gws[1]);
    ret |= kernel->get().setArg(idx++, gws[2]);
    ret |= kernel->get().setArg(idx++, openCLBuffer(query));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= kernel->get().setArg(idx++, mScale);
    ret |= kernel->get().setArg(idx++, mBatch);
    ret |= kernel->get().setArg(idx++, mQuerySeqLen);
    ret |= kernel->get().setArg(idx++, attnLen);
    ret |= kernel->get().setArg(idx++, baseLogical);
    ret |= kernel->get().setArg(idx++, kvLen);
    ret |= kernel->get().setArg(idx++, mCache->maxSlots);
    ret |= kernel->get().setArg(idx++, mNumHead);
    ret |= kernel->get().setArg(idx++, mKvNumHead);
    ret |= kernel->get().setArg(idx++, mHeadDim);
    MNN_CHECK_CL_SUCCESS(ret, lanes == 128u ?
        "setArg decode_causal_attention_hd128_transposed_k_readonly_row128" :
        (lanes == 64u ? "setArg decode_causal_attention_hd128_transposed_k_readonly_row64"
                      : "setArg decode_causal_attention_hd128_transposed_k_readonly_row32"));
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const uint64_t denseWork = static_cast<uint64_t>(attnLen) * static_cast<uint64_t>(kvLen);
    const std::vector<uint32_t> lws = {lanes, 1u, 1u};
    const auto pmcMeta = _makeOpenCLPmcMeta("decode_causal_attention_hd128_transposed_k_readonly",
                                           "attention", layerIndex, attnLen, mQuerySeqLen, kvLen,
                                           baseLogical, static_cast<int>(lanes), 0, mNumHead,
                                           mKvNumHead, mHeadDim, gws, lws, denseWork, causalWork);
    _run3DKernelDefaultPmc(kernel, gws, lws, runtime, pmcMeta);
    if (profileDetail) {
        queue.finish();
        attentionUs = _nowUs() - attentionStartUs;
    }

    const uint64_t rankStartUs = profileDetail ? _nowUs() : 0;
    auto rankErr = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, attnLen, false);
    if (rankErr != NO_ERROR) {
        return rankErr;
    }
    if (profileDetail) {
        queue.finish();
        rankUs = _nowUs() - rankStartUs;
    }

    if (profile) {
        queue.finish();
        const uint64_t totalUs = _nowUs() - startUs;
        MNN_PRINT("OpenCLPagedAttention profile op=decode_causal_attention_hd128_transposed_k_readonly layer=%d "
                  "query=%d input_query=%d kv_len=%d lane=%u append_count=%d prepare_len=%d identity_slot=1 "
                  "dense_kv_work=%llu causal_kv_work=%llu us=%llu append_us=%llu attention_us=%llu "
                  "rank_us=%llu attention_only=1 append_fused=0 skip_append=%d "
                  "decode_key_ready_hit=%d decode_key_prepared=%d "
                  "decode_prepare_inside_decode=%d decode_prepare_us=%llu prepare_us=%llu "
                  "decode_key_required=%d slot_identity=%d prefix_stable=%d record_queue=0\n",
                  layerIndex, attnLen, mQuerySeqLen, kvLen, lanes, appendCurrent ? 1 : 0,
                  requiredDecodeKeyLen,
                  static_cast<unsigned long long>(denseWork),
                  static_cast<unsigned long long>(causalWork),
                  static_cast<unsigned long long>(totalUs),
                  static_cast<unsigned long long>(appendUs),
                  static_cast<unsigned long long>(attentionUs),
                  static_cast<unsigned long long>(rankUs),
                  appendCurrent ? 0 : 1,
                  mLastDecodeKeyReadyHit ? 1 : 0,
                  mLastDecodeKeyPrepared ? 1 : 0,
                  mLastDecodeKeyPrepareInsideDecode ? 1 : 0,
                  static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                  static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                  mLastDecodeKeyRequiredLen,
                  mLastDecodeKeySlotIdentity ? 1 : 0,
                  mLastDecodeKeyPrefixStable ? 1 : 0);
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128TransposedKAppendReadonlyRecord(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int baseLogical, int layerIndex, uint32_t lanes, std::shared_ptr<KernelWrap> attentionKernel) {
    if (!mOpenCLBackend->isUseRecordQueue() || _profilePagedAttention() || !attentionKernel ||
        attnLen != 1 || kvLen <= 0 || mCache == nullptr || !mCache->key || !mCache->decodeKey ||
        !mCache->value || !mDecodeKeyAppendKernel || inputs.size() < 3 ||
        outputs.empty() || mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0 ||
        (mMeta != nullptr && mMeta->needsPicDecodeAttentionRankCapture(layerIndex))) {
        return INVALID_VALUE;
    }
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    const uint32_t appendGws0 = 128u;
    const uint32_t appendGws1 = static_cast<uint32_t>(mBatch * mKvNumHead);
    const uint32_t heads = static_cast<uint32_t>(mNumHead * mBatch);
    const int groupSize = mNumHead / mKvNumHead;
    const bool needRecord = !mDecodeTransposedAppendReadonlyRecordValid ||
        mDecodeTransposedAppendReadonlyRecordLanes != lanes ||
        mDecodeTransposedAppendReadonlyRecordAttnLen != attnLen ||
        mDecodeTransposedAppendReadonlyRecordHeads != heads ||
        mDecodeTransposedAppendReadonlyRecordGroupSize != groupSize ||
        mDecodeTransposedAppendReadonlyRecordAppendGws1 != appendGws1;

    auto updateRecordArgs = [&]() {
        auto& appendArgs = mDecodeTransposedAppendRecordUpdateInfo.update_kernel_args;
        auto& attentionArgs = mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args;
        if (appendArgs.size() < 7 || attentionArgs.size() < 9) {
            return false;
        }
        mDecodeTransposedAppendReadonlyRecordQuerySeqLen = mQuerySeqLen;
        mDecodeTransposedAppendReadonlyRecordBaseLogical = baseLogical;
        mDecodeTransposedAppendReadonlyRecordKvLen = kvLen;
        mDecodeTransposedAppendReadonlyRecordMaxSlots = mCache->maxSlots;
        mDecodeTransposedAppendReadonlyRecordScale = mScale;
        appendArgs[0].arg_value = &openCLBuffer(key)();
        appendArgs[1].arg_value = &openCLBuffer(value)();
        appendArgs[2].arg_value = &openCLBuffer(mCache->key.get())();
        appendArgs[3].arg_value = &openCLBuffer(mCache->value.get())();
        appendArgs[4].arg_value = &openCLBuffer(mCache->decodeKey.get())();
        appendArgs[5].arg_value = &mDecodeTransposedAppendReadonlyRecordBaseLogical;
        appendArgs[6].arg_value = &mDecodeTransposedAppendReadonlyRecordMaxSlots;
        attentionArgs[0].arg_value = &openCLBuffer(query)();
        attentionArgs[1].arg_value = &openCLBuffer(mCache->value.get())();
        attentionArgs[2].arg_value = &openCLBuffer(mCache->decodeKey.get())();
        attentionArgs[3].arg_value = &openCLBuffer(output)();
        attentionArgs[4].arg_value = &mDecodeTransposedAppendReadonlyRecordScale;
        attentionArgs[5].arg_value = &mDecodeTransposedAppendReadonlyRecordQuerySeqLen;
        attentionArgs[6].arg_value = &mDecodeTransposedAppendReadonlyRecordBaseLogical;
        attentionArgs[7].arg_value = &mDecodeTransposedAppendReadonlyRecordKvLen;
        attentionArgs[8].arg_value = &mDecodeTransposedAppendReadonlyRecordMaxSlots;
        return true;
    };

    if (needRecord) {
        mDecodeTransposedAppendReadonlyRecordAppendGws0 = appendGws0;
        mDecodeTransposedAppendReadonlyRecordAppendGws1 = appendGws1;
        mDecodeTransposedAppendReadonlyRecordGws0 = lanes;
        mDecodeTransposedAppendReadonlyRecordGws1 = static_cast<uint32_t>(attnLen);
        mDecodeTransposedAppendReadonlyRecordGws2 = heads;
        mDecodeTransposedAppendReadonlyRecordQuerySeqLen = mQuerySeqLen;
        mDecodeTransposedAppendReadonlyRecordBaseLogical = baseLogical;
        mDecodeTransposedAppendReadonlyRecordKvLen = kvLen;
        mDecodeTransposedAppendReadonlyRecordMaxSlots = mCache->maxSlots;
        mDecodeTransposedAppendReadonlyRecordScale = mScale;

        std::vector<uint32_t> appendGws = {mDecodeTransposedAppendReadonlyRecordAppendGws0,
                                           mDecodeTransposedAppendReadonlyRecordAppendGws1};
        std::vector<uint32_t> appendLws = {32u, 1u};
        std::vector<uint32_t> attentionGws = {mDecodeTransposedAppendReadonlyRecordGws0,
                                              mDecodeTransposedAppendReadonlyRecordGws1,
                                              mDecodeTransposedAppendReadonlyRecordGws2};
        std::vector<uint32_t> attentionLws = {lanes, 1u, 1u};
        cl_int ret = CL_SUCCESS;
        uint32_t idx = 0;
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, appendGws[0]);
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, appendGws[1]);
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, openCLBuffer(key));
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, openCLBuffer(value));
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, mBatch);
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, mKvNumHead);
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, mHeadDim);
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, 0);
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, mDecodeTransposedAppendReadonlyRecordBaseLogical);
        ret |= mDecodeKeyAppendKernel->get().setArg(idx++, mDecodeTransposedAppendReadonlyRecordMaxSlots);
        MNN_CHECK_CL_SUCCESS(ret, "record setArg append_decode_key_value_hd128");
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        idx = 0;
        ret = CL_SUCCESS;
        ret |= attentionKernel->get().setArg(idx++, attentionGws[0]);
        ret |= attentionKernel->get().setArg(idx++, attentionGws[1]);
        ret |= attentionKernel->get().setArg(idx++, attentionGws[2]);
        ret |= attentionKernel->get().setArg(idx++, openCLBuffer(query));
        ret |= attentionKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= attentionKernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
        ret |= attentionKernel->get().setArg(idx++, openCLBuffer(output));
        ret |= attentionKernel->get().setArg(idx++, mDecodeTransposedAppendReadonlyRecordScale);
        ret |= attentionKernel->get().setArg(idx++, mBatch);
        ret |= attentionKernel->get().setArg(idx++, mDecodeTransposedAppendReadonlyRecordQuerySeqLen);
        ret |= attentionKernel->get().setArg(idx++, attnLen);
        ret |= attentionKernel->get().setArg(idx++, mDecodeTransposedAppendReadonlyRecordBaseLogical);
        ret |= attentionKernel->get().setArg(idx++, mDecodeTransposedAppendReadonlyRecordKvLen);
        ret |= attentionKernel->get().setArg(idx++, mDecodeTransposedAppendReadonlyRecordMaxSlots);
        ret |= attentionKernel->get().setArg(idx++, mNumHead);
        ret |= attentionKernel->get().setArg(idx++, mKvNumHead);
        ret |= attentionKernel->get().setArg(idx++, mHeadDim);
        MNN_CHECK_CL_SUCCESS(ret, lanes == 128u ?
            "record setArg decode_causal_attention_hd128_transposed_k_readonly_row128" :
            (lanes == 64u ? "record setArg decode_causal_attention_hd128_transposed_k_readonly_row64"
                          : "record setArg decode_causal_attention_hd128_transposed_k_readonly_row32"));
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }

        mDecodeTransposedAppendRecordUpdateInfo.update_kernel_args.clear();
        mDecodeTransposedAppendRecordUpdateInfo.update_global_size.clear();
        mDecodeTransposedAppendRecordUpdateInfo.update_local_size.clear();
        mDecodeTransposedAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 2, sizeof(cl_mem), &openCLBuffer(key)()});
        mDecodeTransposedAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 3, sizeof(cl_mem), &openCLBuffer(value)()});
        mDecodeTransposedAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 4, sizeof(cl_mem), &openCLBuffer(mCache->key.get())()});
        mDecodeTransposedAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 5, sizeof(cl_mem), &openCLBuffer(mCache->value.get())()});
        mDecodeTransposedAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 6, sizeof(cl_mem), &openCLBuffer(mCache->decodeKey.get())()});
        mDecodeTransposedAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 11, sizeof(mDecodeTransposedAppendReadonlyRecordBaseLogical),
             &mDecodeTransposedAppendReadonlyRecordBaseLogical});
        mDecodeTransposedAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 12, sizeof(mDecodeTransposedAppendReadonlyRecordMaxSlots),
             &mDecodeTransposedAppendReadonlyRecordMaxSlots});

        mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args.clear();
        mDecodeTransposedReadonlyRecordUpdateInfo.update_global_size.clear();
        mDecodeTransposedReadonlyRecordUpdateInfo.update_local_size.clear();
        mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args.push_back(
            {1, 3, sizeof(cl_mem), &openCLBuffer(query)()});
        mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args.push_back(
            {1, 4, sizeof(cl_mem), &openCLBuffer(mCache->value.get())()});
        mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args.push_back(
            {1, 5, sizeof(cl_mem), &openCLBuffer(mCache->decodeKey.get())()});
        mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args.push_back(
            {1, 6, sizeof(cl_mem), &openCLBuffer(output)()});
        mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args.push_back(
            {1, 7, sizeof(mDecodeTransposedAppendReadonlyRecordScale),
             &mDecodeTransposedAppendReadonlyRecordScale});
        mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args.push_back(
            {1, 9, sizeof(mDecodeTransposedAppendReadonlyRecordQuerySeqLen),
             &mDecodeTransposedAppendReadonlyRecordQuerySeqLen});
        mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args.push_back(
            {1, 11, sizeof(mDecodeTransposedAppendReadonlyRecordBaseLogical),
             &mDecodeTransposedAppendReadonlyRecordBaseLogical});
        mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args.push_back(
            {1, 12, sizeof(mDecodeTransposedAppendReadonlyRecordKvLen),
             &mDecodeTransposedAppendReadonlyRecordKvLen});
        mDecodeTransposedReadonlyRecordUpdateInfo.update_kernel_args.push_back(
            {1, 13, sizeof(mDecodeTransposedAppendReadonlyRecordMaxSlots),
             &mDecodeTransposedAppendReadonlyRecordMaxSlots});
        mDecodeTransposedAppendReadonlyRecordUpdateInfos.clear();
        mDecodeTransposedAppendReadonlyRecordUpdateInfos.emplace_back(&mDecodeTransposedAppendRecordUpdateInfo);
        mDecodeTransposedAppendReadonlyRecordUpdateInfos.emplace_back(&mDecodeTransposedReadonlyRecordUpdateInfo);

        mOpenCLBackend->startRecord(mRecording);
        mOpenCLBackend->recordKernel2d(mDecodeKeyAppendKernel, appendGws, appendLws,
                                       &mDecodeTransposedAppendRecordUpdateInfo);
        mOpenCLBackend->recordKernel3d(attentionKernel, attentionGws, attentionLws,
                                       &mDecodeTransposedReadonlyRecordUpdateInfo);
        mOpenCLBackend->endRecord(mRecording);
        mDecodeTransposedAppendReadonlyRecordValid = true;
        mDecodeTransposedAppendReadonlyRecordLanes = lanes;
        mDecodeTransposedAppendReadonlyRecordAttnLen = attnLen;
        mDecodeTransposedAppendReadonlyRecordHeads = heads;
        mDecodeTransposedAppendReadonlyRecordGroupSize = groupSize;
    }
    if (!updateRecordArgs()) {
        return INVALID_VALUE;
    }
    mOpenCLBackend->addRecord(mRecording, mDecodeTransposedAppendReadonlyRecordUpdateInfos);
    mCache->decodeKeyReadyLength = std::max(mCache->decodeKeyReadyLength, baseLogical + 1);
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128TransposedKSparse(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int layerIndex) {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const bool slotIdentitySplit = _decodeRepairSlotIdentitySplitEnabled();
    if (attnLen <= 0 || attnLen > 8 || kvLen <= 0 || mHeadDim != 128 || mCache == nullptr ||
        !mCache->key || (!slotIdentitySplit && !mCache->decodeKey) || !mCache->value ||
        !mCache->sparseQuery || inputs.size() < 3 || mMeta == nullptr ||
        static_cast<int>(mMeta->sparse_query_logical_indices.size()) < attnLen ||
        mNewKvSeqLen < attnLen || mKvNumHead <= 0 || mNumHead <= 0 ||
        mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    if (slotIdentitySplit) {
        return runDecodeCausalAttentionHD128SlotIdentitySplit(inputs, outputs, kvLen, attnLen, layerIndex);
    }
    auto err = ensureDecodeTransposedKKernel();
    if (err != NO_ERROR) {
        return err;
    }
    const int appendCount = std::max(0, std::min(mMeta->pic_decode_recompute_append_count, attnLen));
    const int prepareLen = std::max(0, kvLen - appendCount);
    err = ensureDecodeKeyReady(prepareLen, true);
    if (err != NO_ERROR) {
        return err;
    }

    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    if (runtime == nullptr || output == nullptr) {
        return INVALID_VALUE;
    }

    auto& queue = runtime->commandQueue();
    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t appendUs = 0;
    uint64_t attentionUs = 0;
    uint64_t rankUs = 0;

    uint64_t causalWork = _sparseLogicalWork(mMeta, attnLen, kvLen);
    const uint64_t causalWorkPerRow = attnLen > 0 ? causalWork / static_cast<uint64_t>(attnLen) : 0;
    const uint32_t lanes = _decodeHD128SparseLaneWidth(runtime, causalWorkPerRow, attnLen);
    const bool fusedAppend = _decodeRepairFusedAppendEnabled();
    const bool q1Row = !fusedAppend && attnLen == 1 && _decodeRepairQ1RowEnabled();
    const int qTile = q1Row ? 1 : _decodeHD128QTile(runtime, attnLen);
    // Removed failed A/B dispatch: keycache qtile v2, identity qtile, q1 GQA,
    // and q1 identity fused-KV did not remove the x0 gap and regressed at least
    // one larger model. Keep x=0/1/3/5/7 in the transposed-K qtile family.

    std::shared_ptr<KernelWrap> kernel;
    const char* kernelName = nullptr;
    const char* profileOpName = "decode_causal_attention_hd128_transposed_k_sparse_qtile";
    // PoC (x0 only): d-continuous K read from key_cache via vload4 instead of
    // transposed decode_key gathers. Self-gated: only overrides the q1Row 128-lane
    // path when MNN_PIC_DECODE_SPARSE_DCONTIG_K is set; otherwise falls through to
    // the original sparse kernel. The dcontig kernel's arg layout is a superset
    // (extra key_cache arg after decode_key), so setArg below branches on dcontigK.
    const bool dcontigK = q1Row && lanes == 128u &&
        _envFlagEnabled("MNN_PIC_DECODE_SPARSE_DCONTIG_K", false);
    if (dcontigK) {
        kernel = mDecodeCausalKernelHD128TransposedKSparseDcontigRow128;
        kernelName = "setArg decode_causal_attention_hd128_transposed_k_sparse_dcontig_row128";
        profileOpName = "decode_causal_attention_hd128_transposed_k_sparse_dcontig";
    } else if (q1Row) {
        kernel = lanes == 128u ? mDecodeCausalKernelHD128TransposedKSparseRow128 :
            (lanes == 64u ? mDecodeCausalKernelHD128TransposedKSparseRow64
                          : mDecodeCausalKernelHD128TransposedKSparseRow32);
        kernelName = lanes == 128u ? "setArg decode_causal_attention_hd128_transposed_k_sparse_row128" :
            (lanes == 64u ? "setArg decode_causal_attention_hd128_transposed_k_sparse_row64"
                          : "setArg decode_causal_attention_hd128_transposed_k_sparse_row32");
    } else if (qTile == 1) {
        kernel = fusedAppend ?
            (lanes == 128u ? mDecodeCausalKernelHD128TransposedKQTileFused1Row128 :
                (lanes == 64u ? mDecodeCausalKernelHD128TransposedKQTileFused1Row64
                              : mDecodeCausalKernelHD128TransposedKQTileFused1Row32)) :
            (lanes == 128u ? mDecodeCausalKernelHD128TransposedKQTile1Row128 :
                (lanes == 64u ? mDecodeCausalKernelHD128TransposedKQTile1Row64
                              : mDecodeCausalKernelHD128TransposedKQTile1Row32));
        kernelName = fusedAppend ?
            (lanes == 128u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_fused_q1_row128" :
                (lanes == 64u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_fused_q1_row64"
                              : "setArg decode_causal_attention_hd128_transposed_k_qtile_fused_q1_row32")) :
            (lanes == 128u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_q1_row128" :
                (lanes == 64u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_q1_row64"
                              : "setArg decode_causal_attention_hd128_transposed_k_qtile_q1_row32"));
    } else if (qTile == 2) {
        kernel = fusedAppend ?
            (lanes == 128u ? mDecodeCausalKernelHD128TransposedKQTileFused2Row128 :
                (lanes == 64u ? mDecodeCausalKernelHD128TransposedKQTileFused2Row64
                              : mDecodeCausalKernelHD128TransposedKQTileFused2Row32)) :
            (lanes == 128u ? mDecodeCausalKernelHD128TransposedKQTile2Row128 :
                (lanes == 64u ? mDecodeCausalKernelHD128TransposedKQTile2Row64
                              : mDecodeCausalKernelHD128TransposedKQTile2Row32));
        kernelName = fusedAppend ?
            (lanes == 128u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_fused_q2_row128" :
                (lanes == 64u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_fused_q2_row64"
                              : "setArg decode_causal_attention_hd128_transposed_k_qtile_fused_q2_row32")) :
            (lanes == 128u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_q2_row128" :
                (lanes == 64u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_q2_row64"
                              : "setArg decode_causal_attention_hd128_transposed_k_qtile_q2_row32"));
    } else if (qTile == 4) {
        kernel = fusedAppend ?
            (lanes == 128u ? mDecodeCausalKernelHD128TransposedKQTileFused4Row128 :
                (lanes == 64u ? mDecodeCausalKernelHD128TransposedKQTileFused4Row64
                              : mDecodeCausalKernelHD128TransposedKQTileFused4Row32)) :
            (lanes == 128u ? mDecodeCausalKernelHD128TransposedKQTile4Row128 :
                (lanes == 64u ? mDecodeCausalKernelHD128TransposedKQTile4Row64
                              : mDecodeCausalKernelHD128TransposedKQTile4Row32));
        kernelName = fusedAppend ?
            (lanes == 128u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_fused_q4_row128" :
                (lanes == 64u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_fused_q4_row64"
                              : "setArg decode_causal_attention_hd128_transposed_k_qtile_fused_q4_row32")) :
            (lanes == 128u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_q4_row128" :
                (lanes == 64u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_q4_row64"
                              : "setArg decode_causal_attention_hd128_transposed_k_qtile_q4_row32"));
    } else {
        kernel = fusedAppend ?
            (lanes == 128u ? mDecodeCausalKernelHD128TransposedKQTileFused8Row128 :
                (lanes == 64u ? mDecodeCausalKernelHD128TransposedKQTileFused8Row64
                              : mDecodeCausalKernelHD128TransposedKQTileFused8Row32)) :
            (lanes == 128u ? mDecodeCausalKernelHD128TransposedKQTile8Row128 :
                (lanes == 64u ? mDecodeCausalKernelHD128TransposedKQTile8Row64
                              : mDecodeCausalKernelHD128TransposedKQTile8Row32));
        kernelName = fusedAppend ?
            (lanes == 128u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_fused_q8_row128" :
                (lanes == 64u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_fused_q8_row64"
                              : "setArg decode_causal_attention_hd128_transposed_k_qtile_fused_q8_row32")) :
            (lanes == 128u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_q8_row128" :
                (lanes == 64u ? "setArg decode_causal_attention_hd128_transposed_k_qtile_q8_row64"
                              : "setArg decode_causal_attention_hd128_transposed_k_qtile_q8_row32"));
    }
    if (!kernel) {
        return INVALID_VALUE;
    }
    // dcontig PoC bypasses the record queue (its arg layout differs from the
    // recorded sparse kernel); PoC validates the cache direction first, record
    // compatibility can follow. Falls through to the plain setArg path below.
    if (!fusedAppend && !dcontigK) {
        auto recordErr = runDecodeCausalAttentionHD128TransposedKSparseRecord(
            inputs, outputs, kvLen, attnLen, layerIndex, lanes, qTile, kernel);
        if (recordErr == NO_ERROR) {
            return NO_ERROR;
        }
        const uint64_t appendStartUs = profileDetail ? _nowUs() : 0;
        auto appendKernel = mDecodeKeyAppendSparseKernel;
        if (!appendKernel) {
            return INVALID_VALUE;
        }
        std::vector<uint32_t> appendGws = {
            128u,
            static_cast<uint32_t>(attnLen),
            static_cast<uint32_t>(mBatch * mKvNumHead),
        };
        uint32_t appendIdx = 0;
        cl_int appendRet = CL_SUCCESS;
        appendRet |= appendKernel->get().setArg(appendIdx++, appendGws[0]);
        appendRet |= appendKernel->get().setArg(appendIdx++, appendGws[1]);
        appendRet |= appendKernel->get().setArg(appendIdx++, appendGws[2]);
        appendRet |= appendKernel->get().setArg(appendIdx++, openCLBuffer(key));
        appendRet |= appendKernel->get().setArg(appendIdx++, openCLBuffer(value));
        appendRet |= appendKernel->get().setArg(appendIdx++, openCLBuffer(mCache->key.get()));
        appendRet |= appendKernel->get().setArg(appendIdx++, openCLBuffer(mCache->value.get()));
        appendRet |= appendKernel->get().setArg(appendIdx++, openCLBuffer(mCache->decodeKey.get()));
        appendRet |= appendKernel->get().setArg(appendIdx++, openCLBuffer(mCache->sparseQuery.get()));
        appendRet |= appendKernel->get().setArg(appendIdx++, mBatch);
        appendRet |= appendKernel->get().setArg(appendIdx++, mNewKvSeqLen);
        appendRet |= appendKernel->get().setArg(appendIdx++, attnLen);
        appendRet |= appendKernel->get().setArg(appendIdx++, mKvNumHead);
        appendRet |= appendKernel->get().setArg(appendIdx++, mHeadDim);
        appendRet |= appendKernel->get().setArg(appendIdx++, mCache->maxSlots);
        MNN_CHECK_CL_SUCCESS(appendRet, "setArg append_sparse_decode_key_value_hd128");
        if (appendRet != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        const std::vector<uint32_t> appendLws = {32u, 1u, 1u};
        const auto appendPmcMeta = _makeOpenCLPmcMeta("append_sparse_decode_key_value_hd128",
                                                      "append", layerIndex,
                                                      attnLen, mQuerySeqLen, kvLen, -1, 32, 0,
                                                      mNumHead, mKvNumHead, mHeadDim, appendGws, appendLws,
                                                      0, 0, appendCount, prepareLen, false);
        _run3DKernelDefaultPmc(appendKernel, appendGws, appendLws, runtime, appendPmcMeta);
        mCache->decodeKeyReadyLength = std::max(mCache->decodeKeyReadyLength, kvLen);
        if (profileDetail) {
            queue.finish();
            appendUs = _nowUs() - appendStartUs;
        }
    }
    const uint64_t attentionStartUs = profileDetail ? _nowUs() : 0;
    const uint32_t attentionHeads = static_cast<uint32_t>(mNumHead * mBatch);
    std::vector<uint32_t> gws = {
        lanes,
        static_cast<uint32_t>((attnLen + qTile - 1) / qTile),
        attentionHeads,
    };
    cl_int ret = CL_SUCCESS;
    uint32_t idx = 0;
    ret |= kernel->get().setArg(idx++, gws[0]);
    ret |= kernel->get().setArg(idx++, gws[1]);
    ret |= kernel->get().setArg(idx++, gws[2]);
    ret |= kernel->get().setArg(idx++, openCLBuffer(query));
    if (fusedAppend) {
        ret |= kernel->get().setArg(idx++, openCLBuffer(key));
        ret |= kernel->get().setArg(idx++, openCLBuffer(value));
        ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    }
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
    if (dcontigK) {
        // dcontig kernel takes key_cache (d-continuous K source) after decode_key.
        ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    }
    ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->sparseQuery.get()));
    ret |= kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= kernel->get().setArg(idx++, mScale);
    ret |= kernel->get().setArg(idx++, mBatch);
    ret |= kernel->get().setArg(idx++, mQuerySeqLen);
    if (fusedAppend) {
        ret |= kernel->get().setArg(idx++, mNewKvSeqLen);
    }
    ret |= kernel->get().setArg(idx++, attnLen);
    ret |= kernel->get().setArg(idx++, kvLen);
    ret |= kernel->get().setArg(idx++, mCache->maxSlots);
    ret |= kernel->get().setArg(idx++, mNumHead);
    ret |= kernel->get().setArg(idx++, mKvNumHead);
    ret |= kernel->get().setArg(idx++, mHeadDim);
    MNN_CHECK_CL_SUCCESS(ret, kernelName);
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const uint64_t denseWork = static_cast<uint64_t>(attnLen) * static_cast<uint64_t>(kvLen);
    const std::vector<uint32_t> lws = {lanes, 1u, 1u};
    const auto pmcMeta = _makeOpenCLPmcMeta(profileOpName,
                                           "attention", layerIndex, attnLen, mQuerySeqLen, kvLen,
                                           -1, static_cast<int>(lanes), qTile, mNumHead, mKvNumHead,
                                           mHeadDim, gws, lws, denseWork, causalWork,
                                           appendCount, prepareLen, false);
    _run3DKernelDefaultPmc(kernel, gws, lws, runtime, pmcMeta);
    if (fusedAppend) {
        mCache->decodeKeyReadyLength = std::max(mCache->decodeKeyReadyLength, kvLen);
    }
    if (profileDetail) {
        queue.finish();
        attentionUs = _nowUs() - attentionStartUs;
    }

    const uint64_t rankStartUs = profileDetail ? _nowUs() : 0;
    auto rankErr = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, attnLen, false);
    if (rankErr != NO_ERROR) {
        return rankErr;
    }
    if (profileDetail) {
        queue.finish();
        rankUs = _nowUs() - rankStartUs;
    }

    if (profile) {
        queue.finish();
        const uint64_t totalUs = _nowUs() - startUs;
        if (profileDetail) {
            MNN_PRINT("OpenCLPagedAttention profile op=%s layer=%d "
                      "query=%d input_query=%d kv_len=%d lane=%u q_tile=%d append_count=%d prepare_len=%d "
                      "dense_kv_work=%llu causal_kv_work=%llu fused_append=%d q1_row=%d "
                      "us=%llu append_us=%llu attention_us=%llu "
                      "rank_us=%llu decode_key_ready_hit=%d decode_key_prepared=%d "
                      "decode_prepare_inside_decode=%d decode_prepare_us=%llu prepare_us=%llu "
                      "decode_key_required=%d slot_identity=%d prefix_stable=%d identity_v=%d record_queue=0\n",
                      profileOpName, layerIndex, attnLen, mQuerySeqLen, kvLen, lanes, qTile, appendCount, prepareLen,
                      static_cast<unsigned long long>(denseWork),
                      static_cast<unsigned long long>(causalWork),
                      fusedAppend ? 1 : 0,
                      q1Row ? 1 : 0,
                      static_cast<unsigned long long>(totalUs),
                      static_cast<unsigned long long>(appendUs),
                      static_cast<unsigned long long>(attentionUs),
                      static_cast<unsigned long long>(rankUs),
                      mLastDecodeKeyReadyHit ? 1 : 0,
                      mLastDecodeKeyPrepared ? 1 : 0,
                      mLastDecodeKeyPrepareInsideDecode ? 1 : 0,
                      static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                      static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                      mLastDecodeKeyRequiredLen,
                      mLastDecodeKeySlotIdentity ? 1 : 0,
                      mLastDecodeKeyPrefixStable ? 1 : 0,
                      1);
        } else {
            MNN_PRINT("OpenCLPagedAttention profile op=%s layer=%d "
                      "query=%d input_query=%d kv_len=%d lane=%u q_tile=%d append_count=%d prepare_len=%d "
                      "dense_kv_work=%llu causal_kv_work=%llu fused_append=%d q1_row=%d "
                      "us=%llu append_us=0 attention_us=%llu rank_us=0 "
                      "decode_key_ready_hit=%d decode_key_prepared=%d "
                      "decode_prepare_inside_decode=%d decode_prepare_us=%llu prepare_us=%llu "
                      "decode_key_required=%d slot_identity=%d prefix_stable=%d identity_v=%d record_queue=0\n",
                      profileOpName, layerIndex, attnLen, mQuerySeqLen, kvLen, lanes, qTile, appendCount, prepareLen,
                      static_cast<unsigned long long>(denseWork),
                      static_cast<unsigned long long>(causalWork),
                      fusedAppend ? 1 : 0,
                      q1Row ? 1 : 0,
                      static_cast<unsigned long long>(totalUs),
                      static_cast<unsigned long long>(totalUs),
                      mLastDecodeKeyReadyHit ? 1 : 0,
                      mLastDecodeKeyPrepared ? 1 : 0,
                      mLastDecodeKeyPrepareInsideDecode ? 1 : 0,
                      static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                      static_cast<unsigned long long>(mLastDecodeKeyPrepareUs),
                      mLastDecodeKeyRequiredLen,
                      mLastDecodeKeySlotIdentity ? 1 : 0,
                      mLastDecodeKeyPrefixStable ? 1 : 0,
                      1);
        }
    }
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128TransposedKSparseRecord(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen,
    int attnLen, int layerIndex, uint32_t lanes, int qTile, std::shared_ptr<KernelWrap> kernel) {
    // The record path deliberately records only append_sparse_decode_key_value_hd128
    // plus transposed-K qtile attention; variant-specific key sources/head counts
    // were removed after negative A/B runs.
    auto appendKernel = mDecodeKeyAppendSparseKernel;
    if (!_decodeRepairRecordQueueEnabled() || !mOpenCLBackend->isUseRecordQueue() ||
        _profilePagedAttention() || !kernel || !appendKernel ||
        attnLen <= 0 || kvLen <= 0 || qTile <= 0 || mCache == nullptr ||
        !mCache->key || !mCache->decodeKey || !mCache->value || !mCache->sparseQuery ||
        inputs.size() < 3 || outputs.empty() ||
        (mMeta != nullptr && mMeta->needsPicDecodeAttentionRankCapture(layerIndex))) {
        return INVALID_VALUE;
    }
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    const uint32_t heads = static_cast<uint32_t>(mNumHead * mBatch);
    const bool needRecord = !mDecodeTransposedSparseRecordValid ||
        mDecodeTransposedSparseRecordLanes != lanes ||
        mDecodeTransposedSparseRecordAttnLen != attnLen ||
        mDecodeTransposedSparseRecordHeads != heads ||
        mDecodeTransposedSparseRecordQTile != qTile;

    auto updateRecordArgs = [&]() {
        auto& appendArgs = mDecodeTransposedSparseAppendRecordUpdateInfo.update_kernel_args;
        auto& attentionArgs = mDecodeTransposedSparseAttentionRecordUpdateInfo.update_kernel_args;
        if (appendArgs.size() < 9 || attentionArgs.size() < 10) {
            return false;
        }
        mDecodeTransposedSparseRecordQuerySeqLen = mQuerySeqLen;
        mDecodeTransposedSparseRecordNewKvSeqLen = mNewKvSeqLen;
        mDecodeTransposedSparseRecordKvLen = kvLen;
        mDecodeTransposedSparseRecordMaxSlots = mCache->maxSlots;
        mDecodeTransposedSparseRecordScale = mScale;

        appendArgs[0].arg_value = &openCLBuffer(key)();
        appendArgs[1].arg_value = &openCLBuffer(value)();
        appendArgs[2].arg_value = &openCLBuffer(mCache->key.get())();
        appendArgs[3].arg_value = &openCLBuffer(mCache->value.get())();
        appendArgs[4].arg_value = &openCLBuffer(mCache->decodeKey.get())();
        appendArgs[5].arg_value = &openCLBuffer(mCache->sparseQuery.get())();
        appendArgs[6].arg_value = &mDecodeTransposedSparseRecordNewKvSeqLen;
        appendArgs[7].arg_value = &mDecodeTransposedSparseRecordAttnLen;
        appendArgs[8].arg_value = &mDecodeTransposedSparseRecordMaxSlots;

        attentionArgs[0].arg_value = &openCLBuffer(query)();
        attentionArgs[1].arg_value = &openCLBuffer(mCache->value.get())();
        attentionArgs[2].arg_value = &openCLBuffer(mCache->decodeKey.get())();
        attentionArgs[3].arg_value = &openCLBuffer(mCache->sparseQuery.get())();
        attentionArgs[4].arg_value = &openCLBuffer(output)();
        attentionArgs[5].arg_value = &mDecodeTransposedSparseRecordScale;
        attentionArgs[6].arg_value = &mDecodeTransposedSparseRecordQuerySeqLen;
        attentionArgs[7].arg_value = &mDecodeTransposedSparseRecordAttnLen;
        attentionArgs[8].arg_value = &mDecodeTransposedSparseRecordKvLen;
        attentionArgs[9].arg_value = &mDecodeTransposedSparseRecordMaxSlots;
        return true;
    };

    if (needRecord) {
        mDecodeTransposedSparseRecordAppendGws0 = 128u;
        mDecodeTransposedSparseRecordAppendGws1 = static_cast<uint32_t>(attnLen);
        mDecodeTransposedSparseRecordAppendGws2 = static_cast<uint32_t>(mBatch * mKvNumHead);
        mDecodeTransposedSparseRecordGws0 = lanes;
        mDecodeTransposedSparseRecordGws1 = static_cast<uint32_t>((attnLen + qTile - 1) / qTile);
        mDecodeTransposedSparseRecordGws2 = heads;
        mDecodeTransposedSparseRecordQuerySeqLen = mQuerySeqLen;
        mDecodeTransposedSparseRecordNewKvSeqLen = mNewKvSeqLen;
        mDecodeTransposedSparseRecordKvLen = kvLen;
        mDecodeTransposedSparseRecordMaxSlots = mCache->maxSlots;
        mDecodeTransposedSparseRecordScale = mScale;

        std::vector<uint32_t> appendGws = {
            mDecodeTransposedSparseRecordAppendGws0,
            mDecodeTransposedSparseRecordAppendGws1,
            mDecodeTransposedSparseRecordAppendGws2,
        };
        std::vector<uint32_t> appendLws = {32u, 1u, 1u};
        std::vector<uint32_t> attentionGws = {
            mDecodeTransposedSparseRecordGws0,
            mDecodeTransposedSparseRecordGws1,
            mDecodeTransposedSparseRecordGws2,
        };
        std::vector<uint32_t> attentionLws = {lanes, 1u, 1u};
        cl_int ret = CL_SUCCESS;
        uint32_t idx = 0;
        ret |= appendKernel->get().setArg(idx++, appendGws[0]);
        ret |= appendKernel->get().setArg(idx++, appendGws[1]);
        ret |= appendKernel->get().setArg(idx++, appendGws[2]);
        ret |= appendKernel->get().setArg(idx++, openCLBuffer(key));
        ret |= appendKernel->get().setArg(idx++, openCLBuffer(value));
        ret |= appendKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= appendKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= appendKernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
        ret |= appendKernel->get().setArg(idx++, openCLBuffer(mCache->sparseQuery.get()));
        ret |= appendKernel->get().setArg(idx++, mBatch);
        ret |= appendKernel->get().setArg(idx++, mDecodeTransposedSparseRecordNewKvSeqLen);
        ret |= appendKernel->get().setArg(idx++, attnLen);
        ret |= appendKernel->get().setArg(idx++, mKvNumHead);
        ret |= appendKernel->get().setArg(idx++, mHeadDim);
        ret |= appendKernel->get().setArg(idx++, mDecodeTransposedSparseRecordMaxSlots);
        MNN_CHECK_CL_SUCCESS(ret, "record setArg append_sparse_decode_key_value_hd128");
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }

        idx = 0;
        ret = CL_SUCCESS;
        ret |= kernel->get().setArg(idx++, attentionGws[0]);
        ret |= kernel->get().setArg(idx++, attentionGws[1]);
        ret |= kernel->get().setArg(idx++, attentionGws[2]);
        ret |= kernel->get().setArg(idx++, openCLBuffer(query));
        ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->decodeKey.get()));
        ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->sparseQuery.get()));
        ret |= kernel->get().setArg(idx++, openCLBuffer(output));
        ret |= kernel->get().setArg(idx++, mDecodeTransposedSparseRecordScale);
        ret |= kernel->get().setArg(idx++, mBatch);
        ret |= kernel->get().setArg(idx++, mDecodeTransposedSparseRecordQuerySeqLen);
        ret |= kernel->get().setArg(idx++, attnLen);
        ret |= kernel->get().setArg(idx++, mDecodeTransposedSparseRecordKvLen);
        ret |= kernel->get().setArg(idx++, mDecodeTransposedSparseRecordMaxSlots);
        ret |= kernel->get().setArg(idx++, mNumHead);
        ret |= kernel->get().setArg(idx++, mKvNumHead);
        ret |= kernel->get().setArg(idx++, mHeadDim);
        MNN_CHECK_CL_SUCCESS(ret, "record setArg decode_causal_attention_hd128_transposed_k_sparse_qtile");
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }

        mDecodeTransposedSparseAppendRecordUpdateInfo.update_kernel_args.clear();
        mDecodeTransposedSparseAppendRecordUpdateInfo.update_global_size.clear();
        mDecodeTransposedSparseAppendRecordUpdateInfo.update_local_size.clear();
        mDecodeTransposedSparseAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 3, sizeof(cl_mem), &openCLBuffer(key)()});
        mDecodeTransposedSparseAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 4, sizeof(cl_mem), &openCLBuffer(value)()});
        mDecodeTransposedSparseAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 5, sizeof(cl_mem), &openCLBuffer(mCache->key.get())()});
        mDecodeTransposedSparseAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 6, sizeof(cl_mem), &openCLBuffer(mCache->value.get())()});
        mDecodeTransposedSparseAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 7, sizeof(cl_mem), &openCLBuffer(mCache->decodeKey.get())()});
        mDecodeTransposedSparseAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 8, sizeof(cl_mem), &openCLBuffer(mCache->sparseQuery.get())()});
        mDecodeTransposedSparseAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 10, sizeof(mDecodeTransposedSparseRecordNewKvSeqLen),
             &mDecodeTransposedSparseRecordNewKvSeqLen});
        mDecodeTransposedSparseAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 11, sizeof(mDecodeTransposedSparseRecordAttnLen),
             &mDecodeTransposedSparseRecordAttnLen});
        mDecodeTransposedSparseAppendRecordUpdateInfo.update_kernel_args.push_back(
            {0, 14, sizeof(mDecodeTransposedSparseRecordMaxSlots),
             &mDecodeTransposedSparseRecordMaxSlots});

        mDecodeTransposedSparseAttentionRecordUpdateInfo.update_kernel_args.clear();
        mDecodeTransposedSparseAttentionRecordUpdateInfo.update_global_size.clear();
        mDecodeTransposedSparseAttentionRecordUpdateInfo.update_local_size.clear();
        mDecodeTransposedSparseAttentionRecordUpdateInfo.update_kernel_args.push_back(
            {1, 3, sizeof(cl_mem), &openCLBuffer(query)()});
        mDecodeTransposedSparseAttentionRecordUpdateInfo.update_kernel_args.push_back(
            {1, 4, sizeof(cl_mem), &openCLBuffer(mCache->value.get())()});
        mDecodeTransposedSparseAttentionRecordUpdateInfo.update_kernel_args.push_back(
            {1, 5, sizeof(cl_mem), &openCLBuffer(mCache->decodeKey.get())()});
        mDecodeTransposedSparseAttentionRecordUpdateInfo.update_kernel_args.push_back(
            {1, 6, sizeof(cl_mem), &openCLBuffer(mCache->sparseQuery.get())()});
        mDecodeTransposedSparseAttentionRecordUpdateInfo.update_kernel_args.push_back(
            {1, 7, sizeof(cl_mem), &openCLBuffer(output)()});
        mDecodeTransposedSparseAttentionRecordUpdateInfo.update_kernel_args.push_back(
            {1, 8, sizeof(mDecodeTransposedSparseRecordScale),
             &mDecodeTransposedSparseRecordScale});
        mDecodeTransposedSparseAttentionRecordUpdateInfo.update_kernel_args.push_back(
            {1, 10, sizeof(mDecodeTransposedSparseRecordQuerySeqLen),
             &mDecodeTransposedSparseRecordQuerySeqLen});
        mDecodeTransposedSparseAttentionRecordUpdateInfo.update_kernel_args.push_back(
            {1, 11, sizeof(mDecodeTransposedSparseRecordAttnLen),
             &mDecodeTransposedSparseRecordAttnLen});
        mDecodeTransposedSparseAttentionRecordUpdateInfo.update_kernel_args.push_back(
            {1, 12, sizeof(mDecodeTransposedSparseRecordKvLen),
             &mDecodeTransposedSparseRecordKvLen});
        mDecodeTransposedSparseAttentionRecordUpdateInfo.update_kernel_args.push_back(
            {1, 13, sizeof(mDecodeTransposedSparseRecordMaxSlots),
             &mDecodeTransposedSparseRecordMaxSlots});
        mDecodeTransposedSparseRecordUpdateInfos.clear();
        mDecodeTransposedSparseRecordUpdateInfos.emplace_back(&mDecodeTransposedSparseAppendRecordUpdateInfo);
        mDecodeTransposedSparseRecordUpdateInfos.emplace_back(&mDecodeTransposedSparseAttentionRecordUpdateInfo);

        mOpenCLBackend->startRecord(mRecording);
        mOpenCLBackend->recordKernel3d(appendKernel, appendGws, appendLws,
                                       &mDecodeTransposedSparseAppendRecordUpdateInfo);
        mOpenCLBackend->recordKernel3d(kernel, attentionGws, attentionLws,
                                       &mDecodeTransposedSparseAttentionRecordUpdateInfo);
        mOpenCLBackend->endRecord(mRecording);
        mDecodeTransposedSparseRecordValid = true;
        mDecodeTransposedSparseRecordLanes = lanes;
        mDecodeTransposedSparseRecordAttnLen = attnLen;
        mDecodeTransposedSparseRecordHeads = heads;
        mDecodeTransposedSparseRecordQTile = qTile;
    }
    if (!updateRecordArgs()) {
        return INVALID_VALUE;
    }
    mOpenCLBackend->addRecord(mRecording, mDecodeTransposedSparseRecordUpdateInfos);
    mCache->decodeKeyReadyLength = std::max(mCache->decodeKeyReadyLength, kvLen);
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128IdentityRecord(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int baseLogical, int layerIndex, uint32_t lanes, std::shared_ptr<KernelWrap> kernel) {
    if (!mOpenCLBackend->isUseRecordQueue() || _profilePagedAttention() || !kernel ||
        attnLen <= 0 || kvLen <= 0 || mCache == nullptr || !mCache->key || !mCache->value) {
        return INVALID_VALUE;
    }
    auto query = inputs[0];
    auto output = outputs[0];
    const uint32_t heads = static_cast<uint32_t>(mNumHead * mBatch);
    const bool needRecord = !mDecodeIdentityRecordValid ||
        mDecodeIdentityRecordLanes != lanes ||
        mDecodeIdentityRecordAttnLen != attnLen ||
        mDecodeIdentityRecordHeads != heads;

    auto updateRecordArgs = [&]() {
        auto& args = mDecodeIdentityRecordUpdateInfo.update_kernel_args;
        if (args.size() < 8) {
            return false;
        }
        mDecodeIdentityRecordQuerySeqLen = mQuerySeqLen;
        mDecodeIdentityRecordBaseLogical = baseLogical;
        mDecodeIdentityRecordKvLen = kvLen;
        mDecodeIdentityRecordMaxSlots = mCache->maxSlots;
        args[0].arg_value = &openCLBuffer(query)();
        args[1].arg_value = &openCLBuffer(mCache->key.get())();
        args[2].arg_value = &openCLBuffer(mCache->value.get())();
        args[3].arg_value = &openCLBuffer(output)();
        args[4].arg_value = &mDecodeIdentityRecordQuerySeqLen;
        args[5].arg_value = &mDecodeIdentityRecordBaseLogical;
        args[6].arg_value = &mDecodeIdentityRecordKvLen;
        args[7].arg_value = &mDecodeIdentityRecordMaxSlots;
        return true;
    };

    if (needRecord) {
        mDecodeIdentityRecordGws0 = lanes;
        mDecodeIdentityRecordGws1 = static_cast<uint32_t>(attnLen);
        mDecodeIdentityRecordGws2 = heads;
        mDecodeIdentityRecordQuerySeqLen = mQuerySeqLen;
        mDecodeIdentityRecordBaseLogical = baseLogical;
        mDecodeIdentityRecordKvLen = kvLen;
        mDecodeIdentityRecordMaxSlots = mCache->maxSlots;

        std::vector<uint32_t> gws = {mDecodeIdentityRecordGws0, mDecodeIdentityRecordGws1,
                                     mDecodeIdentityRecordGws2};
        std::vector<uint32_t> lws = {lanes, 1u, 1u};
        cl_int ret = CL_SUCCESS;
        uint32_t idx = 0;
        ret |= kernel->get().setArg(idx++, gws[0]);
        ret |= kernel->get().setArg(idx++, gws[1]);
        ret |= kernel->get().setArg(idx++, gws[2]);
        ret |= kernel->get().setArg(idx++, openCLBuffer(query));
        ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= kernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= kernel->get().setArg(idx++, openCLBuffer(output));
        ret |= kernel->get().setArg(idx++, mScale);
        ret |= kernel->get().setArg(idx++, mBatch);
        ret |= kernel->get().setArg(idx++, mDecodeIdentityRecordQuerySeqLen);
        ret |= kernel->get().setArg(idx++, attnLen);
        ret |= kernel->get().setArg(idx++, mDecodeIdentityRecordBaseLogical);
        ret |= kernel->get().setArg(idx++, mDecodeIdentityRecordKvLen);
        ret |= kernel->get().setArg(idx++, mDecodeIdentityRecordMaxSlots);
        ret |= kernel->get().setArg(idx++, mNumHead);
        ret |= kernel->get().setArg(idx++, mKvNumHead);
        ret |= kernel->get().setArg(idx++, mHeadDim);
        MNN_CHECK_CL_SUCCESS(ret, lanes == 128u ? "record setArg decode_causal_attention_hd128_identity_row128" :
            (lanes == 64u ? "record setArg decode_causal_attention_hd128_identity_row64"
                          : "record setArg decode_causal_attention_hd128_identity_row32"));
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }

        mDecodeIdentityRecordUpdateInfo.update_kernel_args.clear();
        mDecodeIdentityRecordUpdateInfo.update_global_size.clear();
        mDecodeIdentityRecordUpdateInfo.update_local_size.clear();
        mDecodeIdentityRecordUpdateInfo.update_kernel_args.push_back(
            {0, 3, sizeof(cl_mem), &openCLBuffer(query)()});
        mDecodeIdentityRecordUpdateInfo.update_kernel_args.push_back(
            {0, 4, sizeof(cl_mem), &openCLBuffer(mCache->key.get())()});
        mDecodeIdentityRecordUpdateInfo.update_kernel_args.push_back(
            {0, 5, sizeof(cl_mem), &openCLBuffer(mCache->value.get())()});
        mDecodeIdentityRecordUpdateInfo.update_kernel_args.push_back(
            {0, 6, sizeof(cl_mem), &openCLBuffer(output)()});
        mDecodeIdentityRecordUpdateInfo.update_kernel_args.push_back(
            {0, 9, sizeof(mDecodeIdentityRecordQuerySeqLen), &mDecodeIdentityRecordQuerySeqLen});
        mDecodeIdentityRecordUpdateInfo.update_kernel_args.push_back(
            {0, 11, sizeof(mDecodeIdentityRecordBaseLogical), &mDecodeIdentityRecordBaseLogical});
        mDecodeIdentityRecordUpdateInfo.update_kernel_args.push_back(
            {0, 12, sizeof(mDecodeIdentityRecordKvLen), &mDecodeIdentityRecordKvLen});
        mDecodeIdentityRecordUpdateInfo.update_kernel_args.push_back(
            {0, 13, sizeof(mDecodeIdentityRecordMaxSlots), &mDecodeIdentityRecordMaxSlots});
        mDecodeIdentityRecordUpdateInfos.clear();
        mDecodeIdentityRecordUpdateInfos.emplace_back(&mDecodeIdentityRecordUpdateInfo);

        mOpenCLBackend->startRecord(mRecording);
        mOpenCLBackend->recordKernel3d(kernel, gws, lws, &mDecodeIdentityRecordUpdateInfo);
        mOpenCLBackend->endRecord(mRecording);
        mDecodeIdentityRecordValid = true;
        mDecodeIdentityRecordLanes = lanes;
        mDecodeIdentityRecordAttnLen = attnLen;
        mDecodeIdentityRecordHeads = heads;
    }
    if (!updateRecordArgs()) {
        return INVALID_VALUE;
    }
    mOpenCLBackend->addRecord(mRecording, mDecodeIdentityRecordUpdateInfos);
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeAttentionRankCaptureOpenCL(const Tensor* query, int layerIndex,
                                                                          int kvLen, int attnLen,
                                                                          bool queryRowsAreFull) {
    if (mMeta == nullptr || !mMeta->needsPicDecodeAttentionRankCapture(layerIndex)) {
        return NO_ERROR;
    }
    if (query == nullptr || attnLen <= 0 || kvLen <= 0) {
        return NO_ERROR;
    }
    if (mHeadDim != 128 || mCache == nullptr || !mCache->key || !mCache->sparseQuery) {
        return NO_ERROR;
    }
    if (static_cast<int>(mMeta->sparse_query_logical_indices.size()) < attnLen) {
        return NO_ERROR;
    }
    const int picStart = mMeta->pic_decode_attention_pic_start;
    const int picTokenCount = mMeta->pic_decode_attention_pic_token_count;
    const int topM = std::min(mMeta->pic_decode_attention_top_m, picTokenCount);
    if (picStart < 0 || picTokenCount <= 0 || topM <= 0 || picStart + picTokenCount > kvLen ||
        mBatch <= 0 || mNumHead <= 0 || mKvNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return NO_ERROR;
    }
    const int qIndex = attnLen - 1;
    const int qLogical = mMeta->sparse_query_logical_indices[static_cast<size_t>(qIndex)];
    const int qRow = queryRowsAreFull ? qLogical : qIndex;
    if (qLogical < 0 || qLogical >= kvLen || qRow < 0 || qRow >= mQuerySeqLen ||
        picStart + picTokenCount > qLogical + 1) {
        return NO_ERROR;
    }
    if (!mMeta->pic_decode_attention_head_ids.empty()) {
        bool hasValidHead = false;
        for (int head : mMeta->pic_decode_attention_head_ids) {
            if (head >= 0 && head < mNumHead) {
                hasValidHead = true;
                break;
            }
        }
        if (!hasValidHead) {
            return NO_ERROR;
        }
    }
    auto err = ensureDecodeAttentionRankKernel();
    if (err != NO_ERROR) {
        return err;
    }
    err = ensureCacheBlendScoreTemps(picTokenCount, topM, 0);
    if (err != NO_ERROR) {
        return err;
    }
    err = syncDecodeAttentionHeadIds();
    if (err != NO_ERROR) {
        return err;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto& queue = runtime->commandQueue();
    const bool profile = _profilePagedAttention();
    const bool profileDetail = profile && _envFlagEnabled("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const uint64_t startUs = profile ? _nowUs() : 0;
    uint64_t scoreUs = 0;
    uint64_t topKUs = 0;
    uint64_t readbackUs = 0;

    cl::Buffer& headIdsBuffer = (mDecodeAttentionHeadIds && !mMeta->pic_decode_attention_head_ids.empty())
        ? openCLBuffer(mDecodeAttentionHeadIds.get())
        : openCLBuffer(mCache->sparseQuery.get());
    const int headIdCount = mMeta->pic_decode_attention_head_ids.empty()
        ? 0
        : static_cast<int>(mMeta->pic_decode_attention_head_ids.size());

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, openCLBuffer(query));
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, openCLBuffer(mCache->key.get()));
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, headIdsBuffer);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, openCLBuffer(mCacheBlendScores.get()));
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, mBatch);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, mQuerySeqLen);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, mNumHead);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, mKvNumHead);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, kvLen);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, mCache->maxSlots);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, picStart);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, picTokenCount);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, qRow);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, qLogical);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, headIdCount);
    ret |= mDecodeAttentionRankScoreKernelHD128->get().setArg(idx++, mScale);
    MNN_CHECK_CL_SUCCESS(ret, "setArg decode_attention_pic_rank_score_hd128");
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const uint64_t scoreStartUs = profileDetail ? _nowUs() : 0;
    const size_t localSize = 128;
    const size_t globalSize =
        ((static_cast<size_t>(picTokenCount) + localSize - 1) / localSize) * localSize;
    {
        const std::vector<uint32_t> gws = {static_cast<uint32_t>(globalSize)};
        const std::vector<uint32_t> lws = {static_cast<uint32_t>(localSize)};
        const auto pmcMeta = _makeOpenCLPmcMeta("decode_attention_pic_rank_score_hd128", "rank_score",
                                               layerIndex, attnLen, mQuerySeqLen, kvLen, qLogical, 128, 0,
                                               headIdCount > 0 ? headIdCount : mNumHead, mKvNumHead, mHeadDim,
                                               gws, lws);
        _runOpenCLPmcScope(runtime, pmcMeta, [&](cl::Event* eventPtr) {
            ret = queue.enqueueNDRangeKernel(mDecodeAttentionRankScoreKernelHD128->get(), cl::NullRange,
                                             cl::NDRange(globalSize), cl::NDRange(localSize),
                                             nullptr, eventPtr);
        });
    }
    MNN_CHECK_CL_SUCCESS(ret, "enqueue decode_attention_pic_rank_score_hd128");
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    if (profileDetail) {
        queue.finish();
        scoreUs = _nowUs() - scoreStartUs;
    }

    idx = 0;
    ret = CL_SUCCESS;
    ret |= mCacheBlendTopKKernel->get().setArg(idx++, openCLBuffer(mCacheBlendScores.get()));
    ret |= mCacheBlendTopKKernel->get().setArg(idx++, openCLBuffer(mCacheBlendIndices.get()));
    ret |= mCacheBlendTopKKernel->get().setArg(idx++, picTokenCount);
    ret |= mCacheBlendTopKKernel->get().setArg(idx++, topM);
    MNN_CHECK_CL_SUCCESS(ret, "setArg decode_attention_rank_topk");
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    const uint64_t topKStartUs = profileDetail ? _nowUs() : 0;
    {
        const std::vector<uint32_t> gws = {256u};
        const std::vector<uint32_t> lws = {256u};
        const auto pmcMeta = _makeOpenCLPmcMeta("decode_attention_rank_topk", "rank_topk",
                                               layerIndex, attnLen, mQuerySeqLen, kvLen, qLogical, 256, 0,
                                               headIdCount > 0 ? headIdCount : mNumHead, mKvNumHead, mHeadDim,
                                               gws, lws);
        _runOpenCLPmcScope(runtime, pmcMeta, [&](cl::Event* eventPtr) {
            ret = queue.enqueueNDRangeKernel(mCacheBlendTopKKernel->get(), cl::NullRange,
                                             cl::NDRange(256), cl::NDRange(256), nullptr, eventPtr);
        });
    }
    MNN_CHECK_CL_SUCCESS(ret, "enqueue decode_attention_rank_topk");
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    if (profileDetail) {
        queue.finish();
        topKUs = _nowUs() - topKStartUs;
    }

    std::vector<int> selected(static_cast<size_t>(topM), -1);
    const uint64_t readStartUs = profileDetail ? _nowUs() : 0;
    {
        const std::vector<uint32_t> gws = {static_cast<uint32_t>(topM)};
        const auto pmcMeta = _makeOpenCLPmcMeta("decode_attention_rank_readback", "readback",
                                               layerIndex, attnLen, mQuerySeqLen, kvLen, qLogical, 0, 0,
                                               headIdCount > 0 ? headIdCount : mNumHead, mKvNumHead, mHeadDim,
                                               gws, {0u});
        _runOpenCLPmcScope(runtime, pmcMeta, [&](cl::Event* eventPtr) {
            ret = queue.enqueueReadBuffer(openCLBuffer(mCacheBlendIndices.get()), CL_TRUE, 0,
                                          static_cast<size_t>(topM) * sizeof(int), selected.data(),
                                          nullptr, eventPtr);
        });
    }
    MNN_CHECK_CL_SUCCESS(ret, "read decode_attention_rank_topk indices");
    if (ret != CL_SUCCESS) {
        return INVALID_VALUE;
    }
    if (profileDetail) {
        readbackUs = _nowUs() - readStartUs;
    }
    mMeta->setPicDecodeAttentionRankResult(selected, mMeta->pic_decode_attention_step_idx);
    if (profile) {
        queue.finish();
        MNN_PRINT("OpenCLPagedAttention profile op=decode_attention_rank layer=%d pic_tokens=%d top_m=%d "
                  "heads=%d q_logical=%d query=%d input_query=%d kv_len=%d lane=128 us=%llu rank_us=%llu "
                  "score_us=%llu topk_us=%llu readback_us=%llu append_us=0 attention_us=0 record_queue=0 "
                  "decode_key_ready_hit=0 decode_key_prepared=0 decode_prepare_inside_decode=0 "
                  "decode_prepare_us=0 prepare_us=0 slot_identity=%d prefix_stable=1\n",
                  layerIndex, picTokenCount, topM,
                  headIdCount > 0 ? headIdCount : mNumHead, qLogical, attnLen, mQuerySeqLen, kvLen,
                  static_cast<unsigned long long>(_nowUs() - startUs),
                  static_cast<unsigned long long>(_nowUs() - startUs),
                  static_cast<unsigned long long>(scoreUs),
                  static_cast<unsigned long long>(topKUs),
                  static_cast<unsigned long long>(readbackUs),
                  1);
    }
    return NO_ERROR;
}

} // namespace OpenCL
} // namespace MNN

#endif // MNN_OPENCL_BUFFER_CLOSED
#endif // MNN_SUPPORT_TRANSFORMER_FUSE
