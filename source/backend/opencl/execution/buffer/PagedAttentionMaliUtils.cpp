//
//  PagedAttentionMaliUtils.cpp
//  MNN
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/PagedAttentionMaliUtils.hpp"

namespace MNN {
namespace OpenCL {
namespace PagedAttentionMali {

bool isMaliRuntime(OpenCLRuntime* runtime) {
    return runtime != nullptr && runtime->getGpuType() == GpuType::MALI;
}

bool decodeRepairSparseQTileDefaultEnabled(OpenCLRuntime* runtime, bool legacyOpenCL, int attnLen) {
    if (legacyOpenCL || !isMaliRuntime(runtime) || attnLen <= 0 || attnLen > 8) {
        return false;
    }
    return true;
}

uint32_t selectDecodeHD128SparseLaneWidth(OpenCLRuntime* runtime, uint64_t causalWorkPerRow,
                                          int attnLen, uint32_t fallbackLane) {
    (void)causalWorkPerRow;
    if (isMaliRuntime(runtime) && attnLen > 1) {
        return 64u;
    }
    return fallbackLane;
}

int selectDecodeHD128QTile(OpenCLRuntime* runtime, int attnLen) {
    if (!isMaliRuntime(runtime)) {
        return 0;
    }
    if (attnLen >= 8) {
        return 2;
    }
    if (attnLen >= 5) {
        return 4;
    }
    return 0;
}

bool preferLaterSparseQ8K16(OpenCLRuntime* runtime, int headDim, int kvLen, bool queryRowsAreFull) {
    return isMaliRuntime(runtime) && headDim == 128 && !queryRowsAreFull && kvLen > 0 && kvLen <= 2048;
}

bool headDim128SparseVariantSupported(OpenCLRuntime* runtime, int headDim, int kvLen,
                                      uint32_t variant, uint32_t row32Variant, uint32_t q8k16Variant) {
    if (!isMaliRuntime(runtime) || headDim != 128) {
        return false;
    }
    if (kvLen > 0 && kvLen <= 2048 && variant == q8k16Variant) {
        return true;
    }
    return variant == row32Variant;
}

bool rejectSparseFlashVariantFromCache(OpenCLRuntime* runtime, int headDim, int kvLen, bool queryRowsAreFull,
                                       uint32_t variant, uint32_t row32Variant, uint32_t q8k16Variant) {
    if (!isMaliRuntime(runtime) || headDim != 128) {
        return false;
    }
    if (preferLaterSparseQ8K16(runtime, headDim, kvLen, queryRowsAreFull)) {
        return variant != q8k16Variant;
    }
    return variant != row32Variant;
}

} // namespace PagedAttentionMali
} // namespace OpenCL
} // namespace MNN

#endif
#endif
