#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_HPP

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace MNN {
namespace Replay {
namespace KernelCorpus {

// A single kernel argument slot. The runner sets args[i] via setArg(i, value)
// in array order. Kind determines how the value is derived.
struct AdaptedArg {
    enum Kind { SizeConst, Buffer, Scalar, Int2, Int4 } kind = SizeConst;
    // SizeConst: which component of globalSize to use as the const int value.
    int whichDim = 0;
    // Buffer: index into AdaptedCase.buffers.
    int bufferIndex = 0;
    // Scalar: typed value set directly.
    enum ScalarType { Int, Float } scalarType = Int;
    int intVal = 0;
    float floatVal = 0.0f;
    // Int2: two ints packed (for OpenCL int2 params like int2 shape).
    int int2Val[2] = {0, 0};
    // Int4: four ints packed (for OpenCL int4 params).
    int int4Val[4] = {0, 0, 0, 0};

    static AdaptedArg sizeConst(int dim) {
        AdaptedArg a;
        a.kind = SizeConst;
        a.whichDim = dim;
        return a;
    }
    static AdaptedArg buffer(int idx) {
        AdaptedArg a;
        a.kind = Buffer;
        a.bufferIndex = idx;
        return a;
    }
    static AdaptedArg scalarInt(int v) {
        AdaptedArg a;
        a.kind = Scalar;
        a.scalarType = Int;
        a.intVal = v;
        return a;
    }
    static AdaptedArg scalarFloat(float v) {
        AdaptedArg a;
        a.kind = Scalar;
        a.scalarType = Float;
        a.floatVal = v;
        return a;
    }
    static AdaptedArg int2(int x, int y) {
        AdaptedArg a;
        a.kind = Int2;
        a.int2Val[0] = x;
        a.int2Val[1] = y;
        return a;
    }
    static AdaptedArg int4(int x, int y, int z, int w) {
        AdaptedArg a;
        a.kind = Int4;
        a.int4Val[0] = x;
        a.int4Val[1] = y;
        a.int4Val[2] = z;
        a.int4Val[3] = w;
        return a;
    }
};

// A host-side buffer with its initial content and readback flag.
struct AdaptedBuffer {
    std::vector<uint8_t> initialData;  // empty = leave uninitialized (or zero-fill)
    size_t sizeBytes = 0;
    bool isOutput = false;             // runner reads this back after dispatch

    // Convenience helpers for FP32 data.
    void setFp32(const std::vector<float>& data) {
        sizeBytes = data.size() * sizeof(float);
        initialData.resize(sizeBytes);
        std::memcpy(initialData.data(), data.data(), sizeBytes);
    }
    std::vector<float> getFp32() const {
        std::vector<float> out(sizeBytes / sizeof(float));
        if (!initialData.empty()) {
            std::memcpy(out.data(), initialData.data(), sizeBytes);
        }
        return out;
    }
};

// A fully-prepared, version-adapted case. The runner consumes this without
// knowing any operator/version semantics.
struct AdaptedCase {
    // Compile
    std::string source;                 // baked kernel source text
    std::string entry;                  // __kernel void X / "main"
    std::vector<std::string> compileMacros;
    // Pre-compiled SPIR-V (optional, Vulkan only). When non-empty, the runner
    // skips glslangValidator and uses these bytes directly.
    std::vector<uint32_t> spirv;

    // Arguments (OpenCL setArg order; Vulkan uses vulkanBindings for buffers)
    std::vector<AdaptedArg> args;
    std::vector<AdaptedBuffer> buffers;

    // Vulkan-specific
    std::vector<int> vulkanBindings;    // buffers[i] -> binding index (same size as buffers)
    std::vector<uint8_t> pushConstants; // raw bytes, empty = none
    std::vector<std::pair<int, uint32_t>> specConstants;  // {constant_id, value}

    // Dispatch
    uint32_t globalSize[3] = {1, 1, 1};
    uint32_t localSize[3] = {0, 0, 0};  // 0 = runtime default
    int dims = 2;                       // OpenCL NDRange dimension count

    // Validation
    std::string validator;              // "exp_fp32" / "sigmoid_fp32" / ...

    // Metadata
    std::string framework, tag, backend, opType, variant, caseName;
    int warmupRuns = 2;
    int workloadRuns = 5;

    // For validators that need original input/output shape
    int elementCount = 0;
    int m = 0, n = 0, k = 0;
    int w = 0, h = 0, c = 0;
    int orderType = 0;
    // Host-side copies of input data for validation (buffers[0] input for unary/matmul)
    std::vector<float> validatorInputA;
    std::vector<float> validatorInputB;
};

struct CaseSpec {
    std::string name;
    std::string backend;
    std::string opType;
    std::string framework;
    std::string tag;
    std::string variant;
    std::string dtype;
    std::string validator;
    int warmupRuns = 2;
    int workloadRuns = 5;
    int elementCount = 0;
    int m = 0, n = 0, k = 0;
    int w = 0, h = 0, c = 0;
    int orderType = 0;
};

class Bridge {
public:
    virtual ~Bridge() = default;
    virtual bool supports(const std::string& framework, const std::string& tag) const = 0;
    // Returns a fully-prepared AdaptedCase; sourceFile is the path to the baked
    // kernel file relative to corpusRoot (the Bridge may read it or receive text).
    virtual AdaptedCase adapt(const CaseSpec& spec,
                              const std::string& sourceText,
                              const std::string& sourceFile,
                              const std::string& corpusRoot) const = 0;
};

// Registry. findBridge returns nullptr if no bridge covers (framework, tag).
void registerBridge(std::unique_ptr<Bridge> bridge);
const Bridge* findBridge(const std::string& framework, const std::string& tag);

// Validators (shared by OpenCL and Vulkan runners).
bool validateAllZeroFp32(const std::vector<float>& output);
bool validateExpFp32(const std::vector<float>& input, const std::vector<float>& output);
bool validateMatmulFp32(int M, int N, int K,
                        const std::vector<float>& a,
                        const std::vector<float>& b,
                        const std::vector<float>& out);
bool validateSigmoidFp32(const std::vector<float>& input, const std::vector<float>& output);
bool validateTanhFp32(const std::vector<float>& input, const std::vector<float>& output);
bool validatePermuteIdentityFp32(int w, int h, int c,
                                const std::vector<float>& input,
                                const std::vector<float>& output);
bool validateReductionSumFp32(int batch, int height, int width,
                               const std::vector<float>& input,
                               const std::vector<float>& output);
bool validatePoolingMaxFp32(int ih, int iw, int channel, int kh, int kw, int stride,
                            const std::vector<float>& input,
                            const std::vector<float>& output);
bool validateAbsvalFp32(const std::vector<float>& input, const std::vector<float>& output);
bool validateReluFp32(const std::vector<float>& input, const std::vector<float>& output);
bool validateConcatIdentityFp32(const std::vector<float>& input, const std::vector<float>& output);
bool validateIdentityFp32(const std::vector<float>& input, const std::vector<float>& output);

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_HPP
