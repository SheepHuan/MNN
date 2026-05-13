//
//  AttentionBufExecution.hpp
//  MNN
//
//  Created by MNN on 2024/04/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef AttentionBufExecution_hpp
#define AttentionBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"
#include "core/OpCommonUtils.hpp"

#include <cstdint>
#include <mutex>
#include <vector>

namespace MNN {
namespace OpenCL {

std::recursive_mutex& openCLPrefixQueueMutex();

class KVCacheCLManager {
public:
    KVCacheCLManager(Backend *backend, bool kv_cache);

    ~KVCacheCLManager() = default;
    void allocKVCache(const KVMeta* meta, int seqlen);
    bool reallocKVCache(const KVMeta* meta, int seqlen, bool isExecute = true);
    void setArgs(int numHead, int kvNumHead, int headDim){
        mNumHead = numHead;
        mKvNumHead = kvNumHead;
        mHeadDim = headDim;
    }
    int pastKvLength() {
        return mPastLength;
    }
    void setPastKvLength(int length) {
        mPastLength = length;
        if (mPagedActive) {
            mPagedLogicalLength = length;
        }
    }
    void addKvLength(int seq_len){
        mPastLength += seq_len;
        if (mPagedActive) {
            mPagedLogicalLength = mPastLength;
        }
    }
    int maxLength() {
        return mMaxLength;
    }
    int numHead() {
        return mNumHead;
    }
    const cl::Buffer * key() {
        return mPastKey.get();
    }
    const cl::Buffer * value() {
        return mPastValue.get();
    }
    const cl::Buffer * tokenTable() {
        return mPagedTokenTable.get();
    }
    const cl::Buffer * ropeTable() {
        return mPagedRopeTable.get();
    }
    cl::Buffer * mutableKey() {
        return mPastKey.get();
    }
    cl::Buffer * mutableValue() {
        return mPastValue.get();
    }
    uint8_t * mutableKeyHostPtr() {
        return mPastKeyHostStorage.empty() ? nullptr : mPastKeyHostStorage.data();
    }
    uint8_t * mutableValueHostPtr() {
        return mPastValueHostStorage.empty() ? nullptr : mPastValueHostStorage.data();
    }
    bool hostPtrPagedEnabled() const {
        return mUseHostPtrPages;
    }
    void setHostPtrPagedEnabled(bool enabled) {
        mUseHostPtrPages = enabled;
    }
    cl::Buffer * mutableTokenTable() {
        return mPagedTokenTable.get();
    }
    cl::Buffer * mutableRopeTable() {
        return mPagedRopeTable.get();
    }
    const std::vector<int>& pagedTokenTableHost() const {
        return mPagedTokenTableHost;
    }
    const std::vector<int>& pagedRopeTableHost() const {
        return mPagedRopeTableHost;
    }
    bool updatePagedTableHost(int logicalStart, const std::vector<int>& tokenTable,
                              const std::vector<int>& ropeTable);
    bool uploadPagedTableRange(int logicalStart, int tokenCount);
    int byte() const {
        return mByte;
    }
    bool ensureCapacity(int requiredTotal, bool isExecute = true);
    bool ensurePagedCapacity(int requiredPhysicalTotal, int logicalTableLength);
    void discardPagedTablesForRewrite();
    bool pagedActive() const {
        return mPagedActive;
    }
    void setPagedActive(bool active) {
        mPagedActive = active;
    }
    void setPagedState(int logicalLength, int physicalLength, int pageSize, int ropeDim, float ropeTheta);
    bool ensurePagedStateForExistingHistory(int logicalLength, int physicalLength, int pageSize,
                                            int ropeDim, float ropeTheta);
    int pagedPhysicalLength() const {
        return mPagedPhysicalLength;
    }
    int pagedPageSize() const {
        return mPagedPageSize;
    }
    int pagedRopeDim() const {
        return mPagedRopeDim;
    }
    float pagedRopeTheta() const {
        return mPagedRopeTheta;
    }
    bool ensurePagedAppendTable(int logicalStart, int tokenCount, bool isolateSourceStart);
    bool ensurePagedDecodeAppendTable(int logicalStart, int tokenCount);

private:
    bool mKVCache;
    const int mExpandChunk = 64;
    std::shared_ptr<cl::Buffer> mPastKey, mPastValue;
    std::shared_ptr<cl::Buffer> mPagedTokenTable, mPagedRopeTable;
    std::vector<uint8_t> mPastKeyHostStorage, mPastValueHostStorage;
    std::vector<int> mPagedTokenTableHost, mPagedRopeTableHost;
    int mPastLength = 0, mMaxLength = 0, mNumHead = 0, mKvNumHead = 0, mHeadDim = 0;
    OpenCLBackend *mOpenCLBackend;
    int mByte = 4;
    bool mUseHostPtrPages = false;
    bool mPagedActive = false;
    bool mPagedDecodeStarted = false;
    int mPagedPageSize = 64;
    int mPagedLogicalLength = 0;
    int mPagedPhysicalLength = 0;
    int mPagedTableCapacity = 0;
    int mPagedRopeDim = 0;
    float mPagedRopeTheta = 10000.0f;
};

class AttentionBufExecution : public CommonExecution {
public:
    AttentionBufExecution(const MNN::Op *op, Backend *backend, bool kv_cache);
    AttentionBufExecution(std::shared_ptr<KVCacheCLManager> manager, const MNN::Op *op, Backend *backend);
    ErrorCode longPrefillResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode prefillResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode decodeResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);

    ErrorCode UpdateArgs(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode init();
    int getExecuteTime();
    virtual ~AttentionBufExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

protected:
    virtual int onResizePrefixKVLength(const std::vector<Tensor *> &inputs, int seqlen) {
        return 0;
    }
    virtual ErrorCode onPrepareKVCacheBeforeAppend(const std::vector<Tensor *> &inputs, bool& prepared,
                                                   int& appendKvSeqLen) {
        prepared = false;
        auto key = inputs[1];
        appendKvSeqLen = key->shape()[1];
        if (mMeta != nullptr && mMeta->file_flag == KVMeta::PendingReadSegments &&
            !mMeta->prefix_segments.empty()) {
            MNN_ERROR("[Error]: direct segment prefix cache requires PrefixAttention on OpenCL\n");
            return NOT_SUPPORT;
        }
        return NO_ERROR;
    }
    virtual void onBeforeAttentionComputeEnqueue() {
    }
    virtual bool onShouldProfileAttentionKernelEvents() const {
        return false;
    }
    virtual void onAttentionKernelEvent(const std::string& name, const cl::Event& event) {
        (void)name;
        (void)event;
    }
    virtual void onAfterAttentionComputeEnqueue() {
    }

    KVMeta* mMeta;
    OpenCLBackend *mOpenCLBackend;
    std::shared_ptr<KVCacheCLManager> mKVCacheCLManager;
    int mPastKvSeqlen = 0;
    int mKvSeqlen = 0;
    int mKeyValueMaxlen = 0;
    int mDecodeTmpMaxlen = 0;

private:
    int getLocalSize(int size, int maxGroupSize);
    ErrorCode saveOpenCLNativePrefixCache(const std::vector<Tensor *> &inputs, int totalKvLen);
    bool mIsDecode = false;
    void handleKVCache(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);


    uint32_t mMaxWorkGroupSize;
    RecordUpdateInfo mRgUpdateInfo;
    RecordUpdateInfo mRgQUpdateInfo;
    RecordUpdateInfo mRgMUpdateInfo;
    RecordUpdateInfo mQkUpdateInfo;
    RecordUpdateInfo mSoftMaxUpdateInfo;
    RecordUpdateInfo mRgVUpdateInfo;
    RecordUpdateInfo mQkvUpdateInfo;
    int mGlobalWorkSizeQk0 = 0;
    size_t mQkGlobal_size[2];
    size_t mQkPrefillGlobal_size[3];
    std::vector<RecordUpdateInfo*> mOpRecordUpdateInfo;
    std::shared_ptr<Tensor> mTempQK, mTempSoftMax;
private:
    int mAlignQ, mAlignKV, mAlignHDK, mAlignHDN;
    bool mLongPrefill = false;
    int mQseqSplitNum = 1;
    std::shared_ptr<Tensor> mTempQ, mTempK, mTempV, mTempMask, mTempQKV;
    bool mIsAddMask = false;
    bool mNeedKvCache = true;
    bool mHasMask = false;
    bool mKernelUsePagedKV = false;
    bool mKernelIsDecode = false;
private:
    std::vector<std::shared_ptr<KernelWrap>> mKernel_rearrange_vec;
    std::vector<std::shared_ptr<KernelWrap>> mKernel_mask_vec;
    std::vector<std::shared_ptr<KernelWrap>> mKernel_trans_vec;
    std::vector<std::shared_ptr<KernelWrap>> mKernel_clip_vec;
    std::vector<std::shared_ptr<KernelWrap>> mKernel_qk_vec;
    std::vector<std::shared_ptr<KernelWrap>> mKernel_softmax_vec;
    std::vector<std::shared_ptr<KernelWrap>> mKernel_qkv_vec;
    
    std::vector<std::vector<uint32_t>> mGwsQkVec;
    std::vector<std::vector<uint32_t>> mLwsQkVec;
    std::vector<std::vector<uint32_t>> mGwsSoftMaxVec;
    std::vector<std::vector<uint32_t>> mLwsSoftMaxVec;
    std::vector<std::vector<uint32_t>> mGwsQkvVec;
    std::vector<std::vector<uint32_t>> mLwsQkvVec;
    std::vector<std::vector<uint32_t>> mGwsRearrgVec;
    std::vector<std::vector<uint32_t>> mLwsRearrgVec;
    std::vector<std::vector<uint32_t>> mGwsMaskVec;
    std::vector<std::vector<uint32_t>> mLwsMaskVec;
    std::vector<std::vector<uint32_t>> mGwsTransVec;
    std::vector<std::vector<uint32_t>> mLwsTransVec;
    std::vector<std::vector<uint32_t>> mGwsClipVec;
    std::vector<std::vector<uint32_t>> mLwsClipVec;
private:
    std::shared_ptr<KernelWrap> mKernel_rearrangeQ;
    std::shared_ptr<KernelWrap> mKernel_rearrangeV;
    std::shared_ptr<KernelWrap> mKernel_rearrangeMask;
    std::shared_ptr<KernelWrap> mKernel_rearrange;
    std::shared_ptr<KernelWrap> mKernel_qk;
    std::shared_ptr<KernelWrap> mKernel_softmax;
    std::shared_ptr<KernelWrap> mKernel_qkv;
    
    std::vector<uint32_t> mGlobalWorkSizeQk;
    std::vector<uint32_t> mLocalWorkSizeQk;
    std::vector<uint32_t> mGlobalWorkSizeSoftMax;
    std::vector<uint32_t> mLocalWorkSizeSoftMax;
    std::vector<uint32_t> mGlobalWorkSizeQkv;
    std::vector<uint32_t> mLocalWorkSizeQkv;
    std::vector<uint32_t> mGlobalWorkSizeRearrgQ;
    std::vector<uint32_t> mLocalWorkSizeRearrgQ;
    std::vector<uint32_t> mGlobalWorkSizeRearrgV;
    std::vector<uint32_t> mLocalWorkSizeRearrgV;
    std::vector<uint32_t> mGlobalWorkSizeRearrg;
    std::vector<uint32_t> mLocalWorkSizeRearrg;
    std::vector<uint32_t> mGlobalWorkSizeRearrgM;
    std::vector<uint32_t> mLocalWorkSizeRearrgM;

};
} // namespace OpenCL
} // namespace MNN
#endif /* AttentionBufExecution_hpp */
#endif/* MNN_SUPPORT_TRANSFORMER_FUSE */
