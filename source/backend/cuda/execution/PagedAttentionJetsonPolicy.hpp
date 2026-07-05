//
//  PagedAttentionJetsonPolicy.hpp
//  MNN
//

#pragma once

#include <cstdint>

namespace MNN {
namespace CUDA {

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

enum CudaSparseQTileVariant : int {
    kCudaSparseQTileNone = 0,
    kCudaSparseQTileHD64Q4K16 = 2,
    kCudaSparseQTileHD128Q4K16 = 3,
    kCudaSparseQTileHD128Q4K8 = 4,
    kCudaSparseQTileHD128Q8K16 = 5,
    kCudaSparseQTileHD128Q8K8 = 6,
    kCudaSparseQTileHD128Q2K16 = 7,
    kCudaSparseQTileHD128Q2K8 = 8,
    kCudaSparseQTileHD128Q16K16 = 9,
    kCudaSparseQTileHD128Q16K8 = 10,
};

namespace PagedAttentionJetsonPolicy {

const char* sparseQTileVariantName(CudaSparseQTileVariant variant);
CudaSparseQTileVariant selectSparseQTileVariant(int headDim, int attnLen, bool cacheBlendSparse,
                                                bool fixedPlanSparse);
CudaSparseQTileVariant selectDecodeRepairQTileVariant(int headDim, int attnLen);
bool decodeRepairQTileEnabled(int headDim, int attnLen);
bool useCacheBlendLargeSparseTile(bool sparseQuery, bool cacheblendScoreActive, int attnLen);
bool useFixedPlanSparseTile(bool sparseQuery, bool activePlanReady);
bool useWideSparseTile(int attnLen);
int prefillQSplitNum(int attnLen);

} // namespace PagedAttentionJetsonPolicy

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

} // namespace CUDA
} // namespace MNN
