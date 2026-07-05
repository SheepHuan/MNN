//
//  PagedAttentionAdrenoUtils.cpp
//  MNN
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/PagedAttentionAdrenoUtils.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace MNN {
namespace OpenCL {
namespace PagedAttentionAdreno {
namespace {

static bool isAdrenoRuntime(OpenCLRuntime* runtime, bool legacyOpenCL) {
    return runtime != nullptr && runtime->getGpuType() == GpuType::ADRENO && !legacyOpenCL;
}

static bool fitsImage2D(OpenCLRuntime* runtime, int imageWidth, int imageHeight) {
    if (runtime == nullptr || imageWidth <= 0 || imageHeight <= 0) {
        return false;
    }
    const auto maxImage2D = runtime->getMaxImage2DSize();
    return maxImage2D.size() >= 2 && maxImage2D[0] > 0 && maxImage2D[1] > 0 &&
           imageWidth <= static_cast<int>(maxImage2D[0]) &&
           imageHeight <= static_cast<int>(maxImage2D[1]);
}

static bool isHeadDim128SparseHotShape(int batch, int kvHeads, int headDim, int activeLen, int kvLen) {
    if (batch <= 0 || kvHeads <= 0 || headDim != 128 || activeLen <= 0 || kvLen <= 0) {
        return false;
    }
    return kvLen >= 512 && activeLen >= 128;
}

} // namespace

bool useAdrenoGemmFullPrefill(OpenCLRuntime* runtime, bool legacyOpenCL, int seqLen, int kvLen, int batch,
                              int numHeads, int kvHeads, int headDim, int maskKeyLen) {
    if (!isAdrenoRuntime(runtime, legacyOpenCL)) {
        return false;
    }
    if (seqLen <= 1 || kvLen < seqLen || batch <= 0 || numHeads <= 0 || kvHeads <= 0 ||
        numHeads % kvHeads != 0 || headDim <= 0) {
        return false;
    }
    if (maskKeyLen != kvLen) {
        return false;
    }
    return (headDim % 32) == 0;
}

bool useAdrenoSourceSlotValueHydrate(OpenCLRuntime* runtime, bool legacyOpenCL) {
    return isAdrenoRuntime(runtime, legacyOpenCL);
}

bool computeAdrenoCacheBlendValueImageShape(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                            int tokenCount, int headDim, int* imageWidth, int* imageHeight) {
    if (!useAdrenoSourceSlotValueHydrate(runtime, legacyOpenCL) || imageWidth == nullptr || imageHeight == nullptr ||
        batch <= 0 || kvHeads <= 0 || tokenCount < 128 || headDim <= 0 || (headDim % 4) != 0) {
        return false;
    }
    const int width = tokenCount * UP_DIV(headDim, 4);
    const int height = batch * kvHeads;
    if (!fitsImage2D(runtime, width, height)) {
        return false;
    }
    *imageWidth = width;
    *imageHeight = height;
    return true;
}

bool useAdrenoCacheBlendValueImage(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                   int tokenCount, int headDim) {
    int imageWidth = 0;
    int imageHeight = 0;
    return computeAdrenoCacheBlendValueImageShape(runtime, legacyOpenCL, batch, kvHeads, tokenCount, headDim,
                                                  &imageWidth, &imageHeight);
}

bool computeAdrenoSparseFlashKImageShape(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                         int headDim, int kvLen, int* imageWidth, int* imageHeight) {
    if (!useAdrenoSourceSlotValueHydrate(runtime, legacyOpenCL) || imageWidth == nullptr || imageHeight == nullptr ||
        batch <= 0 || kvHeads <= 0 || headDim != 128 || kvLen <= 0) {
        return false;
    }
    const int kvPack = ROUND_UP(kvLen, 32);
    const int width = kvPack / 4;
    const int height = batch * kvHeads * headDim;
    if (!fitsImage2D(runtime, width, height)) {
        return false;
    }
    *imageWidth = width;
    *imageHeight = height;
    return true;
}

bool computeAdrenoSparseFlashKVImageShapes(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                           int headDim, int kvLen,
                                           int* keyImageWidth, int* keyImageHeight,
                                           int* valueImageWidth, int* valueImageHeight) {
    if (!computeAdrenoSparseFlashKImageShape(runtime, legacyOpenCL, batch, kvHeads, headDim, kvLen,
                                             keyImageWidth, keyImageHeight) ||
        valueImageWidth == nullptr || valueImageHeight == nullptr) {
        return false;
    }
    const int kvPack = ROUND_UP(kvLen, 32);
    const int width = headDim / 4;
    const int height = batch * kvHeads * kvPack;
    if (!fitsImage2D(runtime, width, height)) {
        return false;
    }
    *valueImageWidth = width;
    *valueImageHeight = height;
    return true;
}

bool supportsAdrenoSparseFlashKVImage(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                      int headDim, int kvLen) {
    int keyImageWidth = 0;
    int keyImageHeight = 0;
    int valueImageWidth = 0;
    int valueImageHeight = 0;
    return computeAdrenoSparseFlashKVImageShapes(runtime, legacyOpenCL, batch, kvHeads, headDim, kvLen,
                                                 &keyImageWidth, &keyImageHeight,
                                                 &valueImageWidth, &valueImageHeight);
}

bool supportsAdrenoSparseFlashKImage(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                     int headDim, int kvLen) {
    int imageWidth = 0;
    int imageHeight = 0;
    return computeAdrenoSparseFlashKImageShape(runtime, legacyOpenCL, batch, kvHeads, headDim, kvLen,
                                               &imageWidth, &imageHeight);
}

bool preferAdrenoSparseFlashKImage(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads, int headDim,
                                   int kvLen) {
    return supportsAdrenoSparseFlashKImage(runtime, legacyOpenCL, batch, kvHeads, headDim, kvLen);
}

bool allowAdrenoSparseFlashKVImageFallback(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                           int headDim, int kvLen) {
    return !preferAdrenoSparseFlashKImage(runtime, legacyOpenCL, batch, kvHeads, headDim, kvLen) &&
           supportsAdrenoSparseFlashKVImage(runtime, legacyOpenCL, batch, kvHeads, headDim, kvLen);
}

bool preferAdrenoScoreSparseFlash(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                  int headDim, int activeLen, int kvLen) {
    if (!isAdrenoRuntime(runtime, legacyOpenCL) ||
        !isHeadDim128SparseHotShape(batch, kvHeads, headDim, activeLen, kvLen)) {
        return false;
    }
    return activeLen >= 256;
}

bool preferAdrenoLaterSparseQ4K8(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                 int headDim, int activeLen, int kvLen) {
    if (!isAdrenoRuntime(runtime, legacyOpenCL) ||
        !isHeadDim128SparseHotShape(batch, kvHeads, headDim, activeLen, kvLen)) {
        return false;
    }
    return static_cast<int64_t>(activeLen) * 100 >= static_cast<int64_t>(kvLen) * 30;
}

bool useAdrenoGemmTransNullLws(OpenCLRuntime* runtime, bool legacyOpenCL, int seqLen, int kvLen,
                               int batchHeads, std::vector<uint32_t>* lws) {
    if (!isAdrenoRuntime(runtime, legacyOpenCL) || lws == nullptr) {
        return false;
    }
    if (seqLen <= 0 || kvLen <= 0 || batchHeads <= 0) {
        return false;
    }
    *lws = {0u, 0u, 0u};
    return true;
}

bool useAdrenoGemmClipNullLws(OpenCLRuntime* runtime, bool legacyOpenCL, int seqLen, int numHeads,
                              int headDim, std::vector<uint32_t>* lws) {
    if (!isAdrenoRuntime(runtime, legacyOpenCL) || lws == nullptr) {
        return false;
    }
    if (seqLen <= 0 || numHeads <= 0 || headDim <= 0 || (headDim % 32) != 0) {
        return false;
    }
    // Adreno full-prefill gemm currently sees repeated localWS3DDefault failures
    // on qkv_transpose_output. Using driver-selected local size avoids the
    // per-request tuning storm and has been more stable on Rhino.
    *lws = {0u, 0u, 0u};
    return true;
}

bool rejectSparseFlashVariantFromCache(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                       int headDim, int activeLen, int kvLen, bool queryRowsAreFull,
                                       uint32_t variant, bool variantIsKVImage, uint32_t laterQ4K8Variant) {
    if (!variantIsKVImage) {
        if (!queryRowsAreFull &&
            preferAdrenoLaterSparseQ4K8(runtime, legacyOpenCL, batch, kvHeads, headDim, activeLen, kvLen)) {
            return variant != laterQ4K8Variant;
        }
        return false;
    }
    return preferAdrenoSparseFlashKImage(runtime, legacyOpenCL, batch, kvHeads, headDim, kvLen);
}

bool rejectSparseFlashScheduleFromCache(OpenCLRuntime* runtime, bool legacyOpenCL, int batch, int kvHeads,
                                        int headDim, int activeLen, int kvLen, bool queryRowsAreFull,
                                        uint32_t schedule, uint32_t rangeQ128Schedule) {
    if (queryRowsAreFull || schedule != rangeQ128Schedule) {
        return false;
    }
    return preferAdrenoLaterSparseQ4K8(runtime, legacyOpenCL, batch, kvHeads, headDim, activeLen, kvLen);
}

bool disableStaticSparseFlashWorkspace(OpenCLRuntime* runtime, bool legacyOpenCL, int numHeads, int kvHeads,
                                       int headDim, int activeLen, int kvLen, bool queryRowsAreFull) {
    if (queryRowsAreFull || !isAdrenoRuntime(runtime, legacyOpenCL)) {
        return false;
    }
    if (headDim != 128 || numHeads < 32 || kvHeads <= 0 || activeLen < 768 || kvLen < 2048) {
        return false;
    }
    return true;
}

} // namespace PagedAttentionAdreno
} // namespace OpenCL
} // namespace MNN

#endif
#endif
