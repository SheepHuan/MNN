//
//  PagedAttentionJetsonPolicy.cpp
//  MNN
//

#include "backend/cuda/execution/PagedAttentionJetsonPolicy.hpp"

namespace MNN {
namespace CUDA {

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
namespace PagedAttentionJetsonPolicy {

const char* sparseQTileVariantName(CudaSparseQTileVariant variant) {
    switch (variant) {
        case kCudaSparseQTileHD64Q4K16:
            return "hd64_q4k16";
        case kCudaSparseQTileHD128Q4K16:
            return "hd128_q4k16";
        case kCudaSparseQTileHD128Q4K8:
            return "hd128_q4k8";
        case kCudaSparseQTileHD128Q8K16:
            return "hd128_q8k16";
        case kCudaSparseQTileHD128Q8K8:
            return "hd128_q8k8";
        case kCudaSparseQTileHD128Q2K16:
            return "hd128_q2k16";
        case kCudaSparseQTileHD128Q2K8:
            return "hd128_q2k8";
        case kCudaSparseQTileHD128Q16K16:
            return "hd128_q16k16";
        case kCudaSparseQTileHD128Q16K8:
            return "hd128_q16k8";
        case kCudaSparseQTileNone:
        default:
            return "none";
    }
}

CudaSparseQTileVariant selectSparseQTileVariant(int headDim, int attnLen, bool cacheBlendSparse,
                                                bool fixedPlanSparse) {
    // Very small active-row batches are launch/occupancy bound on Xavier; keep
    // the existing non-qtile sparse attention implementation.
    const bool tinyHD128SparseRows = headDim == 128 && attnLen < 64;
    if (tinyHD128SparseRows || headDim != 128) {
        return kCudaSparseQTileNone;
    }
    // CacheBlend selected rows are often scattered across the full PIC span, so
    // keep its accepted q4/k16 default. Fixed-plan sparse rows are more regular
    // and can use wider-Q variants without adding a separate implementation.
    if (fixedPlanSparse && !cacheBlendSparse) {
        if (attnLen >= 384) {
            return kCudaSparseQTileHD128Q16K16;
        }
        if (attnLen >= 256) {
            return kCudaSparseQTileHD128Q8K16;
        }
        if (attnLen >= 128) {
            return kCudaSparseQTileHD128Q8K8;
        }
        return kCudaSparseQTileHD128Q4K8;
    }
    return kCudaSparseQTileHD128Q4K16;
}

CudaSparseQTileVariant selectDecodeRepairQTileVariant(int headDim, int attnLen) {
    // Decode repair uses one implementation family for x=0/1/3/5/7: x=0 is the
    // active-row=1 degenerate case, and x>0 only increases the compact row count.
    if (headDim == 128 && attnLen >= 1) {
        return kCudaSparseQTileHD128Q8K16;
    }
    if (headDim == 64 && attnLen >= 16) {
        return kCudaSparseQTileHD64Q4K16;
    }
    return kCudaSparseQTileNone;
}

bool decodeRepairQTileEnabled(int headDim, int attnLen) {
    return selectDecodeRepairQTileVariant(headDim, attnLen) != kCudaSparseQTileNone;
}

bool useCacheBlendLargeSparseTile(bool sparseQuery, bool cacheblendScoreActive, int attnLen) {
    return sparseQuery && cacheblendScoreActive && attnLen > 256;
}

bool useFixedPlanSparseTile(bool sparseQuery, bool activePlanReady) {
    return sparseQuery && activePlanReady;
}

bool useWideSparseTile(int attnLen) {
    return attnLen >= 384;
}

int prefillQSplitNum(int attnLen) {
    // Each PagedAttention execution owns QK and softmax workspaces sized as
    // max_q_piece * kv_len. Keeping a 1024-row piece at 3072 context would
    // replicate roughly 768 MiB per layer on Xavier. Bound the piece at 256
    // rows so all layers can coexist without exhausting Xavier unified memory.
    return attnLen > 256 ? (attnLen + 255) / 256 : 1;
}

} // namespace PagedAttentionJetsonPolicy
#endif // MNN_SUPPORT_TRANSFORMER_FUSE

} // namespace CUDA
} // namespace MNN
