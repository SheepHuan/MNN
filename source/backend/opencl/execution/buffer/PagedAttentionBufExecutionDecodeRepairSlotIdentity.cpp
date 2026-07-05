//
//  PagedAttentionBufExecutionDecodeRepairSlotIdentity.cpp
//  MNN
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/PagedAttentionBufExecution.hpp"
#include "backend/opencl/core/OpenCLPmcProfiler.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace MNN {
namespace OpenCL {
namespace {

static bool _envFlagEnabledLocal(const char* name, bool defaultValue) {
    const char* value = ::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    return value[0] != '0';
}

static bool _decodeRepairSlotIdentitySplitEnabledLocal() {
    static const bool enabled =
        _envFlagEnabledLocal("MNN_PAGED_ATTENTION_DECODE_REPAIR_SLOT_IDENTITY_SPLIT", false);
    return enabled;
}

static bool _profilePagedAttentionLocal() {
    return _envFlagEnabledLocal("MNN_PAGED_ATTENTION_PROFILE", false);
}

static bool _picOpenCLDebugLocal() {
    return _envFlagEnabledLocal("MNN_PIC_DECODE_DEBUG", false);
}

static uint64_t _nowUsLocal() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

static uint64_t _sparseLogicalWorkLocal(const PagedKVMeta* meta, int attnLen, int kvLen) {
    if (meta == nullptr || attnLen <= 0 || kvLen <= 0 ||
        static_cast<int>(meta->sparse_query_logical_indices.size()) < attnLen) {
        return static_cast<uint64_t>(std::max(0, attnLen)) * static_cast<uint64_t>(std::max(0, kvLen));
    }
    uint64_t work = 0;
    for (int i = 0; i < attnLen; ++i) {
        const int logical = meta->sparse_query_logical_indices[static_cast<size_t>(i)];
        work += static_cast<uint64_t>(std::max(0, std::min(kvLen, logical + 1)));
    }
    return work;
}

static OpenCLPmcScopeMeta _makeOpenCLPmcMetaLocal(const char* op, const char* phase, int layer,
                                                  int query, int inputQuery, int kvLen,
                                                  int baseLogical, int lane, int qTile,
                                                  int heads, int kvHeads, int headDim,
                                                  const std::vector<uint32_t>& gws,
                                                  const std::vector<uint32_t>& lws,
                                                  uint64_t denseKvWork = 0,
                                                  uint64_t causalKvWork = 0,
                                                  int appendCount = 0,
                                                  int prepareLen = 0) {
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
    return meta;
}

template <typename Runner>
static void _runOpenCLPmcScopeLocal(OpenCLRuntime* runtime, const OpenCLPmcScopeMeta& meta,
                                    const Runner& runner) {
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

static void _run3DKernelDefaultPmcLocal(const std::shared_ptr<KernelWrap>& kernel,
                                        const std::vector<uint32_t>& gws,
                                        const std::vector<uint32_t>& lws,
                                        OpenCLRuntime* runtime,
                                        const OpenCLPmcScopeMeta& meta) {
    _runOpenCLPmcScopeLocal(runtime, meta, [&](cl::Event* eventPtr) {
        run3DKernelDefault(kernel, gws, lws, runtime, eventPtr);
    });
}

} // namespace

ErrorCode PagedAttentionBufExecution::ensureDecodeRepairSlotIdentitySplitKernel() {
    if (mHeadDim != 128 || mKvNumHead <= 0 || mNumHead <= 0 || mNumHead % mKvNumHead != 0) {
        return INVALID_VALUE;
    }
    const int groupSize = mNumHead / mKvNumHead;
    if (mDecodeKeyAppendSparseSlotIdentityKernel && mDecodeQKRepairSlotIdentityKernel &&
        mDecodeQKVRepairSlotIdentityKernel && mDecodeRepairSlotIdentitySplitKernelGroupSize == groupSize) {
        return NO_ERROR;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (runtime == nullptr) {
        return INVALID_VALUE;
    }
    const std::set<std::string> options = {"-DNUMHEAD_GROUP_SIZE=" + std::to_string(groupSize)};
    mDecodeKeyAppendSparseSlotIdentityKernel =
        runtime->buildKernel("paged_decode_attention_buf", "append_sparse_decode_key_value_hd128_slot_identity",
                             options, mOpenCLBackend->getPrecision());
    mDecodeQKRepairSlotIdentityKernel =
        runtime->buildKernel("paged_decode_attention_buf", "matmul_qk_decode_repair_slot_identity_hd128",
                             options, mOpenCLBackend->getPrecision());
    mDecodeQKVRepairSlotIdentityKernel =
        runtime->buildKernel("paged_decode_attention_buf", "matmul_qkv_decode_repair_slot_identity_hd128_b8",
                             options, mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL(mDecodeKeyAppendSparseSlotIdentityKernel);
    OPENCL_CHECK_KERNEL(mDecodeQKRepairSlotIdentityKernel);
    OPENCL_CHECK_KERNEL(mDecodeQKVRepairSlotIdentityKernel);
    mDecodeRepairSlotIdentitySplitKernelGroupSize = groupSize;
    return NO_ERROR;
}

ErrorCode PagedAttentionBufExecution::runDecodeCausalAttentionHD128SlotIdentitySplit(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, int kvLen, int attnLen,
    int layerIndex) {
    if (!_decodeRepairSlotIdentitySplitEnabledLocal() || attnLen <= 0 || attnLen > 8 || kvLen <= 0 ||
        mHeadDim != 128 || mCache == nullptr || !mCache->key || !mCache->value ||
        !mCache->sparseQuery || inputs.size() < 3 || outputs.empty() || mMeta == nullptr ||
        static_cast<int>(mMeta->sparse_query_logical_indices.size()) < attnLen ||
        mNewKvSeqLen < attnLen || mKvNumHead <= 0 || mNumHead <= 0 ||
        mNumHead % mKvNumHead != 0 || kvLen > mCache->maxSlots) {
        return INVALID_VALUE;
    }
    for (int i = 0; i < attnLen; ++i) {
        const int logical = mMeta->sparse_query_logical_indices[static_cast<size_t>(i)];
        if (logical < 0 || logical >= kvLen || logical >= mCache->maxSlots) {
            if (_picOpenCLDebugLocal()) {
                MNN_PRINT("PIC OpenCL PA slot_identity split unsupported layer=%d row=%d logical=%d "
                          "kv_len=%d max_slots=%d\n",
                          layerIndex, i, logical, kvLen, mCache->maxSlots);
            }
            return INVALID_VALUE;
        }
    }
    auto err = ensureDecodeRepairSlotIdentitySplitKernel();
    if (err != NO_ERROR) {
        return err;
    }
    err = ensureDecodeTransposedTemps(kvLen, attnLen);
    if (err != NO_ERROR) {
        return err;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    if (runtime == nullptr || !mDecodeKeyAppendSparseSlotIdentityKernel ||
        !mDecodeQKRepairSlotIdentityKernel || !mDecodeQKVRepairSlotIdentityKernel ||
        !mDecodeSoftmaxKernel) {
        return INVALID_VALUE;
    }

    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output = outputs[0];
    if (!query || !key || !value || !output) {
        return INVALID_VALUE;
    }

    auto& queue = runtime->commandQueue();
    auto& qkBuffer = openCLDeferBuffer(mTempQK.get());
    auto& softmaxBuffer = openCLDeferBuffer(mTempSoftmax.get());
    const bool profile = _profilePagedAttentionLocal();
    const bool profileDetail = profile && _envFlagEnabledLocal("MNN_PAGED_ATTENTION_PROFILE_DETAIL", false);
    const uint64_t startUs = profile ? _nowUsLocal() : 0;
    uint64_t appendUs = 0;
    uint64_t qkUs = 0;
    uint64_t softmaxUs = 0;
    uint64_t qkvUs = 0;
    uint64_t rankUs = 0;
    const int appendCount = std::max(0, std::min(mMeta->pic_decode_recompute_append_count, attnLen));
    const int prepareLen = std::max(0, kvLen - appendCount);
    const uint64_t denseWork = static_cast<uint64_t>(attnLen) * static_cast<uint64_t>(kvLen);
    const uint64_t causalWork = _sparseLogicalWorkLocal(mMeta, attnLen, kvLen);

    mLastDecodeKeyRequiredLen = 0;
    mLastDecodeKeyPrepareUs = 0;
    mLastDecodeKeyReadyHit = true;
    mLastDecodeKeyPrepared = false;
    mLastDecodeKeyPrepareInsideDecode = false;
    mLastDecodeKeySlotIdentity = true;
    mLastDecodeKeyPrefixStable = true;

    {
        std::vector<uint32_t> gws = {
            128u,
            static_cast<uint32_t>(attnLen),
            static_cast<uint32_t>(mBatch * mKvNumHead),
        };
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mDecodeKeyAppendSparseSlotIdentityKernel->get().setArg(idx++, gws[0]);
        ret |= mDecodeKeyAppendSparseSlotIdentityKernel->get().setArg(idx++, gws[1]);
        ret |= mDecodeKeyAppendSparseSlotIdentityKernel->get().setArg(idx++, gws[2]);
        ret |= mDecodeKeyAppendSparseSlotIdentityKernel->get().setArg(idx++, openCLBuffer(key));
        ret |= mDecodeKeyAppendSparseSlotIdentityKernel->get().setArg(idx++, openCLBuffer(value));
        ret |= mDecodeKeyAppendSparseSlotIdentityKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= mDecodeKeyAppendSparseSlotIdentityKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= mDecodeKeyAppendSparseSlotIdentityKernel->get().setArg(idx++, openCLBuffer(mCache->sparseQuery.get()));
        ret |= mDecodeKeyAppendSparseSlotIdentityKernel->get().setArg(idx++, mBatch);
        ret |= mDecodeKeyAppendSparseSlotIdentityKernel->get().setArg(idx++, mNewKvSeqLen);
        ret |= mDecodeKeyAppendSparseSlotIdentityKernel->get().setArg(idx++, attnLen);
        ret |= mDecodeKeyAppendSparseSlotIdentityKernel->get().setArg(idx++, mKvNumHead);
        ret |= mDecodeKeyAppendSparseSlotIdentityKernel->get().setArg(idx++, mHeadDim);
        ret |= mDecodeKeyAppendSparseSlotIdentityKernel->get().setArg(idx++, mCache->maxSlots);
        MNN_CHECK_CL_SUCCESS(ret, "setArg append_sparse_decode_key_value_hd128_slot_identity");
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        const std::vector<uint32_t> lws = {32u, 1u, 1u};
        const uint64_t opStartUs = profileDetail ? _nowUsLocal() : 0;
        const auto pmcMeta = _makeOpenCLPmcMetaLocal("append_sparse_decode_key_value_hd128_slot_identity",
                                                     "append", layerIndex, attnLen, mQuerySeqLen, kvLen,
                                                     -1, 32, 0, mNumHead, mKvNumHead, mHeadDim, gws, lws,
                                                     0, 0, appendCount, prepareLen);
        _run3DKernelDefaultPmcLocal(mDecodeKeyAppendSparseSlotIdentityKernel, gws, lws, runtime, pmcMeta);
        if (profileDetail) {
            queue.finish();
            appendUs = _nowUsLocal() - opStartUs;
        }
    }

    {
        std::vector<uint32_t> gws = {
            static_cast<uint32_t>((kvLen + 3) / 4),
            static_cast<uint32_t>(attnLen),
            static_cast<uint32_t>(mBatch * mNumHead),
        };
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mDecodeQKRepairSlotIdentityKernel->get().setArg(idx++, gws[0]);
        ret |= mDecodeQKRepairSlotIdentityKernel->get().setArg(idx++, gws[1]);
        ret |= mDecodeQKRepairSlotIdentityKernel->get().setArg(idx++, gws[2]);
        ret |= mDecodeQKRepairSlotIdentityKernel->get().setArg(idx++, openCLBuffer(query));
        ret |= mDecodeQKRepairSlotIdentityKernel->get().setArg(idx++, openCLBuffer(mCache->key.get()));
        ret |= mDecodeQKRepairSlotIdentityKernel->get().setArg(idx++, openCLBuffer(mCache->sparseQuery.get()));
        ret |= mDecodeQKRepairSlotIdentityKernel->get().setArg(idx++, qkBuffer);
        ret |= mDecodeQKRepairSlotIdentityKernel->get().setArg(idx++, mScale);
        ret |= mDecodeQKRepairSlotIdentityKernel->get().setArg(idx++, mBatch);
        ret |= mDecodeQKRepairSlotIdentityKernel->get().setArg(idx++, mQuerySeqLen);
        ret |= mDecodeQKRepairSlotIdentityKernel->get().setArg(idx++, attnLen);
        ret |= mDecodeQKRepairSlotIdentityKernel->get().setArg(idx++, kvLen);
        ret |= mDecodeQKRepairSlotIdentityKernel->get().setArg(idx++, mCache->maxSlots);
        ret |= mDecodeQKRepairSlotIdentityKernel->get().setArg(idx++, mNumHead);
        ret |= mDecodeQKRepairSlotIdentityKernel->get().setArg(idx++, mKvNumHead);
        ret |= mDecodeQKRepairSlotIdentityKernel->get().setArg(idx++, mHeadDim);
        MNN_CHECK_CL_SUCCESS(ret, "setArg matmul_qk_decode_repair_slot_identity_hd128");
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        const std::vector<uint32_t> lws = {1u, 1u, 1u};
        const uint64_t opStartUs = profileDetail ? _nowUsLocal() : 0;
        const auto pmcMeta = _makeOpenCLPmcMetaLocal("matmul_qk_decode_repair_slot_identity_hd128",
                                                     "attention", layerIndex, attnLen, mQuerySeqLen, kvLen,
                                                     -1, 0, 0, mNumHead, mKvNumHead, mHeadDim, gws, lws,
                                                     denseWork, causalWork, appendCount, prepareLen);
        _run3DKernelDefaultPmcLocal(mDecodeQKRepairSlotIdentityKernel, gws, lws, runtime, pmcMeta);
        if (profileDetail) {
            queue.finish();
            qkUs = _nowUsLocal() - opStartUs;
        }
    }

    {
        const uint32_t outside = static_cast<uint32_t>(mBatch * mNumHead * attnLen);
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
        MNN_CHECK_CL_SUCCESS(ret, "setArg slot_identity split softmax_in1_buf");
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        const uint64_t opStartUs = profileDetail ? _nowUsLocal() : 0;
        run3DKernelDefault(mDecodeSoftmaxKernel, gws, {64u, 1u, 1u}, runtime);
        if (profileDetail) {
            queue.finish();
            softmaxUs = _nowUsLocal() - opStartUs;
        }
    }

    {
        std::vector<uint32_t> gws = {
            static_cast<uint32_t>((mHeadDim + 7) / 8),
            static_cast<uint32_t>(attnLen),
            static_cast<uint32_t>(mBatch * mNumHead),
        };
        uint32_t idx = 0;
        cl_int ret = CL_SUCCESS;
        ret |= mDecodeQKVRepairSlotIdentityKernel->get().setArg(idx++, gws[0]);
        ret |= mDecodeQKVRepairSlotIdentityKernel->get().setArg(idx++, gws[1]);
        ret |= mDecodeQKVRepairSlotIdentityKernel->get().setArg(idx++, gws[2]);
        ret |= mDecodeQKVRepairSlotIdentityKernel->get().setArg(idx++, softmaxBuffer);
        ret |= mDecodeQKVRepairSlotIdentityKernel->get().setArg(idx++, openCLBuffer(mCache->value.get()));
        ret |= mDecodeQKVRepairSlotIdentityKernel->get().setArg(idx++, openCLBuffer(output));
        ret |= mDecodeQKVRepairSlotIdentityKernel->get().setArg(idx++, kvLen);
        ret |= mDecodeQKVRepairSlotIdentityKernel->get().setArg(idx++, attnLen);
        ret |= mDecodeQKVRepairSlotIdentityKernel->get().setArg(idx++, mCache->maxSlots);
        ret |= mDecodeQKVRepairSlotIdentityKernel->get().setArg(idx++, mBatch);
        ret |= mDecodeQKVRepairSlotIdentityKernel->get().setArg(idx++, mNumHead);
        ret |= mDecodeQKVRepairSlotIdentityKernel->get().setArg(idx++, mKvNumHead);
        ret |= mDecodeQKVRepairSlotIdentityKernel->get().setArg(idx++, mHeadDim);
        MNN_CHECK_CL_SUCCESS(ret, "setArg matmul_qkv_decode_repair_slot_identity_hd128_b8");
        if (ret != CL_SUCCESS) {
            return INVALID_VALUE;
        }
        const std::vector<uint32_t> lws = {1u, 1u, 1u};
        const uint64_t opStartUs = profileDetail ? _nowUsLocal() : 0;
        const auto pmcMeta = _makeOpenCLPmcMetaLocal("matmul_qkv_decode_repair_slot_identity_hd128_b8",
                                                     "attention", layerIndex, attnLen, mQuerySeqLen, kvLen,
                                                     -1, 0, 0, mNumHead, mKvNumHead, mHeadDim, gws, lws,
                                                     denseWork, causalWork, appendCount, prepareLen);
        _run3DKernelDefaultPmcLocal(mDecodeQKVRepairSlotIdentityKernel, gws, lws, runtime, pmcMeta);
        if (profileDetail) {
            queue.finish();
            qkvUs = _nowUsLocal() - opStartUs;
        }
    }

    const uint64_t rankStartUs = profileDetail ? _nowUsLocal() : 0;
    auto rankErr = runDecodeAttentionRankCaptureOpenCL(query, layerIndex, kvLen, attnLen, false);
    if (rankErr != NO_ERROR) {
        return rankErr;
    }
    if (profileDetail) {
        queue.finish();
        rankUs = _nowUsLocal() - rankStartUs;
    }

    if (profile) {
        queue.finish();
        const uint64_t attentionUs = profileDetail ? (qkUs + softmaxUs + qkvUs) : 0;
        const uint64_t totalUs = _nowUsLocal() - startUs;
        MNN_PRINT("OpenCLPagedAttention profile op=decode_repair_slot_identity_split layer=%d "
                  "query=%d input_query=%d kv_len=%d append_count=%d prepare_len=%d "
                  "dense_kv_work=%llu causal_kv_work=%llu us=%llu append_us=%llu attention_us=%llu "
                  "qk_us=%llu softmax_us=%llu qkv_us=%llu rank_us=%llu "
                  "decode_key_ready_hit=1 decode_key_prepared=0 decode_prepare_inside_decode=0 "
                  "decode_prepare_us=0 prepare_us=0 decode_key_required=0 slot_identity=1 "
                  "prefix_stable=1 identity_v=1 record_queue=0 split_profile=1\n",
                  layerIndex, attnLen, mQuerySeqLen, kvLen, appendCount, prepareLen,
                  static_cast<unsigned long long>(denseWork),
                  static_cast<unsigned long long>(causalWork),
                  static_cast<unsigned long long>(totalUs),
                  static_cast<unsigned long long>(appendUs),
                  static_cast<unsigned long long>(attentionUs),
                  static_cast<unsigned long long>(qkUs),
                  static_cast<unsigned long long>(softmaxUs),
                  static_cast<unsigned long long>(qkvUs),
                  static_cast<unsigned long long>(rankUs));
    }
    return NO_ERROR;
}

} // namespace OpenCL
} // namespace MNN

#endif /* MNN_OPENCL_BUFFER_CLOSED */
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
