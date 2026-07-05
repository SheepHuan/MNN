//
//  PagedAttentionMaliUtils.hpp
//  MNN
//

#pragma once

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#ifndef MNN_OPENCL_BUFFER_CLOSED

#include <cstdint>

#include "backend/opencl/core/runtime/OpenCLRuntime.hpp"

namespace MNN {
namespace OpenCL {
namespace PagedAttentionMali {

bool isMaliRuntime(OpenCLRuntime* runtime);
bool decodeRepairSparseQTileDefaultEnabled(OpenCLRuntime* runtime, bool legacyOpenCL, int attnLen);
uint32_t selectDecodeHD128SparseLaneWidth(OpenCLRuntime* runtime, uint64_t causalWorkPerRow,
                                          int attnLen, uint32_t fallbackLane);
int selectDecodeHD128QTile(OpenCLRuntime* runtime, int attnLen);
bool preferLaterSparseQ8K16(OpenCLRuntime* runtime, int headDim, int kvLen, bool queryRowsAreFull);
bool headDim128SparseVariantSupported(OpenCLRuntime* runtime, int headDim, int kvLen,
                                      uint32_t variant, uint32_t row32Variant, uint32_t q8k16Variant);
bool rejectSparseFlashVariantFromCache(OpenCLRuntime* runtime, int headDim, int kvLen, bool queryRowsAreFull,
                                       uint32_t variant, uint32_t row32Variant, uint32_t q8k16Variant);

} // namespace PagedAttentionMali
} // namespace OpenCL
} // namespace MNN

#endif
#endif
