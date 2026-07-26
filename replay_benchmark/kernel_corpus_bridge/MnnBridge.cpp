#include "Bridge.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

namespace MNN {
namespace Replay {
namespace KernelCorpus {

namespace {

// Reads a file relative to corpusRoot; returns empty string on failure.
std::string readFile(const std::string& corpusRoot, const std::string& relative) {
    std::string path = corpusRoot;
    if (!path.empty() && path.back() != '/') path.push_back('/');
    path += relative;
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

class MnnBridge : public Bridge {
public:
    bool supports(const std::string& framework, const std::string& tag) const override {
        if (framework != "mnn") return false;
        return tag == "1.2.0" || tag == "3.6.0";
    }

    AdaptedCase adapt(const CaseSpec& spec,
                      const std::string& sourceText,
                      const std::string& sourceFile,
                      const std::string& corpusRoot) const override {
        AdaptedCase ac;
        ac.framework = spec.framework;
        ac.tag = spec.tag;
        ac.backend = spec.backend;
        ac.opType = spec.opType;
        ac.variant = spec.variant;
        ac.caseName = spec.name;
        ac.validator = spec.validator;
        ac.warmupRuns = spec.warmupRuns;
        ac.workloadRuns = spec.workloadRuns;
        ac.elementCount = spec.elementCount;
        ac.m = spec.m; ac.n = spec.n; ac.k = spec.k;
        ac.source = sourceText;

        if (spec.opType == "raster" && spec.variant.find("buffer_set_zero") != std::string::npos) {
            adaptBufferSetZero(spec, ac);
        } else if (spec.opType == "unary" && spec.variant.find("unary_buf_exp") != std::string::npos) {
            adaptUnaryBufExp(spec, ac);
        } else if (spec.opType == "conv" && spec.variant.find("matmul_buf") != std::string::npos) {
            adaptMatmulBuf(spec, ac);
        } else {
            ac.entry = "";
            ac.validator.clear();
        }
        return ac;
    }

private:
    // buffer_set_zero(GLOBAL_SIZE_2_DIMS, output)
    // GLOBAL_SIZE_2_DIMS expands to 2 __private const int params (dim0, dim1).
    void adaptBufferSetZero(const CaseSpec& spec, AdaptedCase& ac) const {
        ac.entry = "buffer_set_zero";
        const int count = spec.elementCount > 0 ? spec.elementCount : 1024;
        AdaptedBuffer outBuf;
        outBuf.sizeBytes = count * sizeof(float);
        outBuf.isOutput = true;
        ac.buffers.push_back(outBuf);
        ac.args.push_back(AdaptedArg::sizeConst(0));  // dim0 = count
        ac.args.push_back(AdaptedArg::sizeConst(1));  // dim1 = 1
        ac.args.push_back(AdaptedArg::buffer(0));     // output
        ac.globalSize[0] = count;
        ac.globalSize[1] = 1;
        ac.dims = 2;
    }

    // 1.2.0: unary_buf(GLOBAL_SIZE_3_DIMS, input, output, height)
    // 3.6.0: unary_buf(GLOBAL_SIZE_2_DIMS, input, output, size)
    void adaptUnaryBufExp(const CaseSpec& spec, AdaptedCase& ac) const {
        ac.entry = "unary_buf";
        const int count = spec.elementCount > 0 ? spec.elementCount : 1024;
        std::vector<float> input(count);
        for (int i = 0; i < count; ++i) input[i] = 0.1f * (i % 10);
        AdaptedBuffer inBuf;
        inBuf.setFp32(input);
        inBuf.isOutput = false;
        AdaptedBuffer outBuf;
        outBuf.sizeBytes = count * sizeof(float);
        outBuf.isOutput = true;
        ac.buffers.push_back(inBuf);
        ac.buffers.push_back(outBuf);
        ac.validatorInputA = input;  // for validator

        const int cb = (count + 3) / 4;
        if (spec.tag == "1.2.0") {
            // args: dim0=cb, dim1=1, dim2=1, input, output, height=1
            ac.args.push_back(AdaptedArg::sizeConst(0));
            ac.args.push_back(AdaptedArg::sizeConst(1));
            ac.args.push_back(AdaptedArg::sizeConst(2));
            ac.args.push_back(AdaptedArg::buffer(0));
            ac.args.push_back(AdaptedArg::buffer(1));
            ac.args.push_back(AdaptedArg::scalarInt(1));
            ac.globalSize[0] = cb; ac.globalSize[1] = 1; ac.globalSize[2] = 1;
            ac.dims = 3;
        } else {
            // 3.6.0: dim0=cb, dim1=1, input, output, size=count
            ac.args.push_back(AdaptedArg::sizeConst(0));
            ac.args.push_back(AdaptedArg::sizeConst(1));
            ac.args.push_back(AdaptedArg::buffer(0));
            ac.args.push_back(AdaptedArg::buffer(1));
            ac.args.push_back(AdaptedArg::scalarInt(count));
            ac.globalSize[0] = cb; ac.globalSize[1] = 1;
            ac.dims = 2;
        }
    }

    // 1.2.0: matmul_buf(GLOBAL_SIZE_2_DIMS, a, b, out, channels=K, channel_blocks=K4, width_blocks=N4)
    void adaptMatmulBuf(const CaseSpec& spec, AdaptedCase& ac) const {
        ac.entry = "matmul_buf";
        const int M = spec.m > 0 ? spec.m : 16;
        const int N = spec.n > 0 ? spec.n : 16;
        const int K = spec.k > 0 ? spec.k : 16;
        const int M4 = (M + 3) / 4;
        const int N4 = (N + 3) / 4;
        const int K4 = (K + 3) / 4;
        const int aCount = M4 * K4 * 4;
        const int bCount = K4 * N4 * 4;
        const int outCount = M4 * N4 * 4;

        std::vector<float> a(aCount), b(bCount);
        for (int i = 0; i < aCount; ++i) a[i] = 0.01f * (i % 7);
        for (int i = 0; i < bCount; ++i) b[i] = 0.01f * (i % 5);

        AdaptedBuffer aBuf;
        aBuf.setFp32(a);
        aBuf.isOutput = false;
        AdaptedBuffer bBuf;
        bBuf.setFp32(b);
        bBuf.isOutput = false;
        AdaptedBuffer outBuf;
        outBuf.sizeBytes = outCount * sizeof(float);
        outBuf.isOutput = true;
        ac.buffers.push_back(aBuf);
        ac.buffers.push_back(bBuf);
        ac.buffers.push_back(outBuf);
        ac.validatorInputA = a;
        ac.validatorInputB = b;

        // args: dim0=N4, dim1=M4, a, b, out, K, K4, N4
        ac.args.push_back(AdaptedArg::sizeConst(0));
        ac.args.push_back(AdaptedArg::sizeConst(1));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.args.push_back(AdaptedArg::buffer(2));
        ac.args.push_back(AdaptedArg::scalarInt(K));
        ac.args.push_back(AdaptedArg::scalarInt(K4));
        ac.args.push_back(AdaptedArg::scalarInt(N4));
        ac.globalSize[0] = N4;
        ac.globalSize[1] = M4;
        ac.dims = 2;
    }
};

// Self-registering initializer. Runs at static init time; the guard in
// Bridge.cpp ensures registration happens before first findBridge() use.
struct MnnBridgeRegistrar {
    MnnBridgeRegistrar() {
        registerBridge(std::unique_ptr<Bridge>(new MnnBridge()));
    }
};

// A function-local static ensures single registration without namespace-scope
// dynamic-init objects (project rule: no global non-POD with dynamic init).
void ensureRegistered() {
    static MnnBridgeRegistrar r;
    (void)r;
}

} // namespace

// External function called by benchmark main to force registration.
void registerMnnBridge() {
    ensureRegistered();
}

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
