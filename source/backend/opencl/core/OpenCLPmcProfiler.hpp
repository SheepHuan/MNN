//
//  OpenCLPmcProfiler.hpp
//  MNN
//

#ifndef OpenCLPmcProfiler_hpp
#define OpenCLPmcProfiler_hpp

#include "backend/opencl/core/runtime/OpenCLRuntime.hpp"
#include <cstdint>
#include <vector>

namespace MNN {
namespace OpenCL {

struct OpenCLPmcScopeMeta {
    const char* op = nullptr;
    const char* phase = nullptr;
    int layer = -1;
    int query = -1;
    int inputQuery = -1;
    int kvLen = -1;
    int baseLogical = -1;
    int lane = 0;
    int qTile = 0;
    int heads = 0;
    int kvHeads = 0;
    int headDim = 0;
    std::vector<uint32_t> gws;
    std::vector<uint32_t> lws;
    uint64_t denseKvWork = 0;
    uint64_t causalKvWork = 0;
    int appendCount = 0;
    int prepareLen = 0;
    bool decodePrepareInsideDecode = false;
};

class OpenCLPmcProfiler {
public:
    static OpenCLPmcProfiler& get();

    bool enabledFor(const OpenCLPmcScopeMeta& meta) const;
    uint64_t begin(OpenCLRuntime* runtime, cl::CommandQueue& queue, const OpenCLPmcScopeMeta& meta);
    void end(uint64_t token, OpenCLRuntime* runtime, cl::CommandQueue& queue,
             const OpenCLPmcScopeMeta& meta, cl::Event* event = nullptr);
};

} // namespace OpenCL
} // namespace MNN

#endif /* OpenCLPmcProfiler_hpp */
