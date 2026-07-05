//
//  PagedAttentionBufExecutionExternal.hpp
//  MNN
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifndef PagedAttentionBufExecutionExternal_hpp
#define PagedAttentionBufExecutionExternal_hpp

#include "backend/opencl/execution/buffer/PagedAttentionBufExecution.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace MNN {
namespace OpenCL {

int _alignPicCacheSourceSlots(int slots);
int _picCacheSourceSlotBase(const PagedKVMeta* meta, int kvLen);
size_t _picCacheSourceSlotCount(const PagedKVMeta* meta);

void _registerExternalLayerMappedTarget(const PagedKVMeta* meta, int layerIndex, int batch, int kvHeads,
                                        int headDim, int bytes, int maxSlots,
                                        const std::shared_ptr<Tensor>& key,
                                        const std::shared_ptr<Tensor>& value,
                                        cl::CommandQueue& queue, bool valueSourceHydrate);

void _scheduleExternalLayerReadsFrom(const PagedKVMeta* meta, int startLayer, int batch, int kvHeads,
                                     int headDim, int bytes, int kvLen);

bool _readExternalValueSegmentToSourceCLBuffer(const std::string& path, cl::Buffer& buffer,
                                               size_t expectedBytes, cl::CommandQueue& queue,
                                               int batch, int kvHeads, size_t sourceTokenCount,
                                               size_t sourceTokenOffset, size_t tokenCount,
                                               int headDim, int bytes);

bool _readExternalValueSegmentToPagedCacheOpenCL(const std::string& path, cl::Buffer& valueCache,
                                                 cl::CommandQueue& queue, int batch, int kvHeads,
                                                 int maxSlots, size_t logicalStart,
                                                 size_t sourceTokenCount, size_t sourceTokenOffset,
                                                 size_t tokenCount, int headDim, int bytes);

} // namespace OpenCL
} // namespace MNN

#endif // PagedAttentionBufExecutionExternal_hpp
#endif // MNN_OPENCL_BUFFER_CLOSED
#endif // MNN_SUPPORT_TRANSFORMER_FUSE
