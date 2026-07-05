//
//  PagedAttentionAdrenoUtils.hpp
//  MNN
//

#pragma once

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#ifndef MNN_OPENCL_BUFFER_CLOSED

#include <cstdint>
#include <vector>

#include "backend/opencl/core/runtime/OpenCLRuntime.hpp"

namespace MNN {
namespace OpenCL {
namespace PagedAttentionAdreno {

bool useAdrenoGemmFullPrefill(OpenCLRuntime* runtime, bool legacyOpenCL, int seqLen, int kvLen, int batch,
                              int numHeads, int kvHeads, int headDim, int maskKeyLen);
bool useAdrenoSourceSlotValueHydrate(OpenCLRuntime* runtime, bool legacyOpenCL);
bool useAdrenoCacheBlendValueImage(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                   int tokenCount, int headDim);
bool computeAdrenoCacheBlendValueImageShape(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                            int tokenCount, int headDim, int* imageWidth, int* imageHeight);
bool supportsAdrenoSparseFlashKVImage(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                      int headDim, int kvLen);
bool supportsAdrenoSparseFlashKImage(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                     int headDim, int kvLen);
bool computeAdrenoSparseFlashKImageShape(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                         int headDim, int kvLen, int* imageWidth, int* imageHeight);
bool computeAdrenoSparseFlashKVImageShapes(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                           int headDim, int kvLen,
                                           int* keyImageWidth, int* keyImageHeight,
                                           int* valueImageWidth, int* valueImageHeight);
bool preferAdrenoSparseFlashKImage(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                   int headDim, int kvLen);
bool allowAdrenoSparseFlashKVImageFallback(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                           int headDim, int kvLen);
bool preferAdrenoScoreSparseFlash(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                  int headDim, int activeLen, int kvLen);
bool preferAdrenoLaterSparseQ4K8(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                 int headDim, int activeLen, int kvLen);
bool useAdrenoGemmTransNullLws(OpenCLRuntime* runtime, bool legacyOpenCL, int seqLen, int kvLen,
                               int batchHeads, std::vector<uint32_t>* lws);
bool useAdrenoGemmClipNullLws(OpenCLRuntime* runtime, bool legacyOpenCL, int seqLen, int numHeads,
                              int headDim, std::vector<uint32_t>* lws);
bool rejectSparseFlashVariantFromCache(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                       int headDim, int activeLen, int kvLen, bool queryRowsAreFull,
                                       uint32_t variant, bool variantIsKVImage, uint32_t laterQ4K8Variant);
bool rejectSparseFlashScheduleFromCache(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                        int headDim, int activeLen, int kvLen, bool queryRowsAreFull,
                                        uint32_t schedule, uint32_t rangeQ128Schedule);
bool disableStaticSparseFlashWorkspace(OpenCLRuntime* runtime, bool legacyOpenCL, int numHeads, int kvHeads,
                                       int headDim, int activeLen, int kvLen, bool queryRowsAreFull);

} // namespace PagedAttentionAdreno
} // namespace OpenCL
} // namespace MNN

#endif
#endif
