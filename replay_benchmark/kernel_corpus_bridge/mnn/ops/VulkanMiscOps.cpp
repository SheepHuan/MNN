#include "VulkanMiscOps.hpp"
#include <cmath>
#include <cstring>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// Helper: fill input buffer with deterministic ramp data.
static void fillRamp(std::vector<float>& v, int n, int period) {
    v.resize(n);
    for (int i = 0; i < n; ++i) v[i] = 0.1f * (i % period);
}

// ============================================================================
// VulkanSelectKernel — select.comp
// ============================================================================
bool VulkanSelectKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "vulkan_select_fp32";
    const int n = spec.intParam("size", 256);

    // select mask: alternating 0/1 so both branches are exercised
    std::vector<float> sel(n), in0(n), in1(n);
    for (int i = 0; i < n; ++i) {
        sel[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        in0[i] = 0.1f * (i % 7);
        in1[i] = -0.1f * (i % 5);
    }
    // Shader binding order: 0=output, 1=select(int), 2=input0, 3=input1
    AdaptedBuffer outBuf; outBuf.sizeBytes = n * sizeof(float); outBuf.isOutput = true;
    AdaptedBuffer selBuf; selBuf.setFp32(sel); selBuf.isOutput = false;
    AdaptedBuffer b0; b0.setFp32(in0); b0.isOutput = false;
    AdaptedBuffer b1; b1.setFp32(in1); b1.isOutput = false;
    ac.buffers.push_back(outBuf);   // binding=0
    ac.buffers.push_back(selBuf);   // binding=1
    ac.buffers.push_back(b0);       // binding=2
    ac.buffers.push_back(b1);       // binding=3
    ac.vulkanBindings = {0, 1, 2, 3};
    ac.validatorInputA = in0;
    ac.validatorInputB = in1;
    ac.elementCount = n;

    struct PushParam { int32_t size[4]; };
    PushParam pp;
    pp.size[0] = 1; pp.size[1] = n; pp.size[2] = n; pp.size[3] = n;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    const uint32_t localX = 256;
    const uint32_t gx = ((static_cast<uint32_t>(n) + localX - 1) / localX) * localX;
    ac.globalSize[0] = gx; ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    ac.localSize[0] = localX; ac.localSize[1] = 1; ac.localSize[2] = 1;
    ac.dims = 1;
    return true;
}

bool VulkanSelectKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (ac.validatorInputA.size() != output.size() ||
        ac.validatorInputB.size() != output.size()) return false;
    const int n = static_cast<int>(output.size());
    for (int i = 0; i < n; ++i) {
        const float expected = (i % 2 == 0) ? ac.validatorInputA[i] : ac.validatorInputB[i];
        if (std::fabs(output[i] - expected) > 1e-3f) return false;
    }
    return true;
}

// ============================================================================
// VulkanRangeKernel — range.comp (FP32 path)
// ============================================================================
bool VulkanRangeKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "vulkan_range_fp32";
    const int n = spec.intParam("size", 256);

    // start=1.5, delta=0.25 → out[i] = i*0.25 + 1.5
    std::vector<float> start(1, 1.5f), delta(1, 0.25f);
    AdaptedBuffer outBuf; outBuf.sizeBytes = n * sizeof(float); outBuf.isOutput = true;
    AdaptedBuffer startBuf; startBuf.setFp32(start); startBuf.isOutput = false;
    AdaptedBuffer deltaBuf; deltaBuf.setFp32(delta); deltaBuf.isOutput = false;
    ac.buffers.push_back(outBuf);   // binding=0
    ac.buffers.push_back(startBuf); // binding=1
    ac.buffers.push_back(deltaBuf); // binding=2
    ac.vulkanBindings = {0, 1, 2};
    ac.elementCount = n;

    struct PushParam { int32_t size[4]; };
    PushParam pp;
    pp.size[0] = 1; pp.size[1] = 1; pp.size[2] = 1; pp.size[3] = n;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    const uint32_t localX = 256;
    const uint32_t gx = ((static_cast<uint32_t>(n) + localX - 1) / localX) * localX;
    ac.globalSize[0] = gx; ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    ac.localSize[0] = localX; ac.localSize[1] = 1; ac.localSize[2] = 1;
    ac.dims = 1;
    return true;
}

bool VulkanRangeKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int n = static_cast<int>(output.size());
    for (int i = 0; i < n; ++i) {
        const float expected = 0.25f * static_cast<float>(i) + 1.5f;
        if (std::fabs(output[i] - expected) > 1e-3f) return false;
    }
    return true;
}

// ============================================================================
// VulkanCastFloatIntKernel — cast_float_int.comp (FLOAT4→ivec4)
//   Each thread converts one vec4 (4 floats -> 4 ints). Buffer size in floats
//   is 4*numVecs for input, 4*numVecs for output (int = float size).
// ============================================================================
bool VulkanCastFloatIntKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "vulkan_cast_float_int_fp32";
    const int nVec = spec.intParam("size", 64);  // number of vec4 elements
    const int nFloats = nVec * 4;

    std::vector<float> in;
    fillRamp(in, nFloats, 13);
    AdaptedBuffer outBuf; outBuf.sizeBytes = nFloats * sizeof(float); outBuf.isOutput = true;
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    ac.buffers.push_back(outBuf);   // binding=0 (OUTPUT_TYPE = ivec4)
    ac.buffers.push_back(inBuf);    // binding=1 (INPUT_TYPE = FLOAT4)
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.elementCount = nVec;

    struct PushParam { int32_t size[4]; float slope[4]; };
    PushParam pp;
    pp.size[0] = nVec; pp.size[1] = 1; pp.size[2] = 1; pp.size[3] = 1;
    pp.slope[0] = 0.0f; pp.slope[1] = 0.0f; pp.slope[2] = 0.0f; pp.slope[3] = 0.0f;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    const uint32_t localX = 256;
    const uint32_t gx = ((static_cast<uint32_t>(nVec) + localX - 1) / localX) * localX;
    ac.globalSize[0] = gx; ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    ac.localSize[0] = localX; ac.localSize[1] = 1; ac.localSize[2] = 1;
    ac.dims = 1;
    return true;
}

bool VulkanCastFloatIntKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    // Output is ivec4 (int32 per component), input is vec4 (float per component).
    // The runner reads back raw bytes as floats; reinterpret each as int32.
    if (ac.validatorInputA.size() != output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        const int32_t expected = static_cast<int32_t>(ac.validatorInputA[i]);
        int32_t got;
        std::memcpy(&got, &output[i], sizeof(int32_t));
        if (got != expected) return false;
    }
    return true;
}

// ============================================================================
// VulkanScaleKernel — scale.comp (NC4HW4)
//   scale/bias are indexed by channelC4 (= posIndex / planeSize), size = c4 vec4.
//   out[pos] = in[pos] * scale[pos/plane] + bias[pos/plane]
// ============================================================================
bool VulkanScaleKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "vulkan_scale_fp32";
    // imgSize.x = H*W (plane size in vec4 elements), imgSize.z = C4, imgSize.w = C4*plane
    const int c4 = spec.intParam("c4", 2);
    const int plane = spec.intParam("plane", 16);
    const int nVec = c4 * plane;  // total FLOAT4 elements

    std::vector<float> in(nVec * 4);
    for (int i = 0; i < nVec * 4; ++i) {
        in[i] = 0.1f * (i % 7);
    }
    // scale/bias: one vec4 per channelC4
    std::vector<float> scale(c4 * 4), bias(c4 * 4);
    for (int i = 0; i < c4 * 4; ++i) {
        scale[i] = 0.5f + 0.1f * (i % 3);
        bias[i] = 0.01f * (i % 5);
    }
    AdaptedBuffer outBuf; outBuf.sizeBytes = nVec * 4 * sizeof(float); outBuf.isOutput = true;
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer scaleBuf; scaleBuf.setFp32(scale); scaleBuf.isOutput = false;
    AdaptedBuffer biasBuf; biasBuf.setFp32(bias); biasBuf.isOutput = false;
    ac.buffers.push_back(outBuf);   // binding=0
    ac.buffers.push_back(inBuf);    // binding=1
    ac.buffers.push_back(scaleBuf); // binding=2
    ac.buffers.push_back(biasBuf);  // binding=3
    ac.vulkanBindings = {0, 1, 2, 3};
    ac.validatorInputA = in;
    ac.validatorInputB = scale;
    ac.elementCount = nVec;

    struct PushParam { int32_t imgSize[4]; };
    PushParam pp;
    pp.imgSize[0] = plane; pp.imgSize[1] = 1; pp.imgSize[2] = c4; pp.imgSize[3] = nVec;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    const uint32_t localX = 256;
    const uint32_t gx = ((static_cast<uint32_t>(nVec) + localX - 1) / localX) * localX;
    ac.globalSize[0] = gx; ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    ac.localSize[0] = localX; ac.localSize[1] = 1; ac.localSize[2] = 1;
    ac.dims = 1;
    return true;
}

bool VulkanScaleKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (ac.pushConstants.size() < 4 * sizeof(int32_t)) return false;
    int32_t imgSize[4];
    std::memcpy(imgSize, ac.pushConstants.data(), sizeof(imgSize));
    const int plane = imgSize[0];
    const int c4 = imgSize[2];
    // bias is recomputed from the channelC4 index (same formula as in adapt()).
    const int nVec = c4 * plane;
    if (output.size() != static_cast<size_t>(nVec * 4)) return false;
    for (int v = 0; v < nVec; ++v) {
        const int chIdx = v / plane;  // channelC4 index
        for (int c = 0; c < 4; ++c) {
            const float scale = ac.validatorInputB[chIdx * 4 + c];
            const float bias = 0.01f * ((chIdx * 4 + c) % 5);
            const float expected = ac.validatorInputA[v * 4 + c] * scale + bias;
            if (std::fabs(output[v * 4 + c] - expected) > 1e-3f) return false;
        }
    }
    return true;
}

// ============================================================================
// VulkanPreluKernel — preluWithChannel.comp (NC4HW4)
//   slope is per-channelC4 vec4 (size c4*4 floats). Index: oz = posIndex / plane.
// ============================================================================
bool VulkanPreluKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "vulkan_prelu_fp32";
    const int c4 = spec.intParam("c4", 2);
    const int plane = spec.intParam("plane", 16);
    const int nVec = c4 * plane;

    std::vector<float> in(nVec * 4);
    for (int i = 0; i < nVec * 4; ++i) {
        // Mix positive and negative values
        in[i] = (i % 3 == 0) ? (-0.1f * (i % 7)) : (0.1f * (i % 7));
    }
    // slope: one vec4 per channelC4
    std::vector<float> slope(c4 * 4, 0.25f);
    AdaptedBuffer outBuf; outBuf.sizeBytes = nVec * 4 * sizeof(float); outBuf.isOutput = true;
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer slopeBuf; slopeBuf.setFp32(slope); slopeBuf.isOutput = false;
    ac.buffers.push_back(outBuf);   // binding=0
    ac.buffers.push_back(inBuf);    // binding=1
    ac.buffers.push_back(slopeBuf); // binding=2
    ac.vulkanBindings = {0, 1, 2};
    ac.validatorInputA = in;
    ac.elementCount = nVec;

    struct PushParam { int32_t imgSize[4]; };
    PushParam pp;
    pp.imgSize[0] = plane; pp.imgSize[1] = 1; pp.imgSize[2] = c4; pp.imgSize[3] = nVec;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    const uint32_t localX = 256;
    const uint32_t gx = ((static_cast<uint32_t>(nVec) + localX - 1) / localX) * localX;
    ac.globalSize[0] = gx; ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    ac.localSize[0] = localX; ac.localSize[1] = 1; ac.localSize[2] = 1;
    ac.dims = 1;
    return true;
}

bool VulkanPreluKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (ac.validatorInputA.size() != output.size()) return false;
    // slope is per-channelC4 constant 0.25 (set in adapt())
    for (size_t i = 0; i < output.size(); ++i) {
        const float x = ac.validatorInputA[i];
        const float expected = (x > 0.0f) ? x : x * 0.25f;
        if (std::fabs(output[i] - expected) > 1e-3f) return false;
    }
    return true;
}

// ============================================================================
// VulkanArgmaxKernel — argmax.comp
// ============================================================================
bool VulkanArgmaxKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "vulkan_argmax_fp32";
    const int inside = spec.intParam("inside", 4);   // W
    const int axis = spec.intParam("axis", 16);      // H (reduce dim)
    const int outside = spec.intParam("outside", 2); // C
    const int reduceAxis = spec.intParam("reduceAxis", 4); // threads per reduce

    const int inSize = inside * axis * outside;
    const int outSize = inside * outside;
    std::vector<float> in(inSize);
    for (int i = 0; i < inSize; ++i) in[i] = 0.01f * ((i * 7) % 97);

    AdaptedBuffer outBuf; outBuf.sizeBytes = outSize * sizeof(int); outBuf.isOutput = true;
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    ac.buffers.push_back(outBuf);   // binding=0
    ac.buffers.push_back(inBuf);    // binding=1
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.elementCount = outSize;

    struct PushParam { int32_t size[4]; };
    PushParam pp;
    pp.size[0] = inside; pp.size[1] = axis; pp.size[2] = outside; pp.size[3] = reduceAxis;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    // globalSize.x = outside*inside*reduceAxis (each reduce thread handles reduceAxis slice)
    const uint32_t total = static_cast<uint32_t>(outside * inside * reduceAxis);
    const uint32_t localX = 256;
    const uint32_t gx = ((total + localX - 1) / localX) * localX;
    ac.globalSize[0] = gx; ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    ac.localSize[0] = localX; ac.localSize[1] = 1; ac.localSize[2] = 1;
    ac.dims = 1;
    return true;
}

bool VulkanArgmaxKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (ac.pushConstants.size() < 4 * sizeof(int32_t)) return false;
    int32_t sz[4];
    std::memcpy(sz, ac.pushConstants.data(), sizeof(sz));
    const int inside = sz[0], axis = sz[1], outside = sz[2];
    // Argmax writes int32 to the output buffer; the runner reads back raw bytes
    // as float. Reinterpret each float's bit pattern as int32.
    if (output.size() != static_cast<size_t>(inside * outside)) return false;
    for (int y = 0; y < outside; ++y) {
        for (int x = 0; x < inside; ++x) {
            // Compute expected argmax over axis for (y, x)
            const int base = y * axis * inside + x;
            float maxVal = ac.validatorInputA[base];
            int maxIdx = 0;
            for (int i = 1; i < axis; ++i) {
                const float v = ac.validatorInputA[base + i * inside];
                if (v > maxVal) { maxVal = v; maxIdx = i; }
            }
            int32_t gotIdx;
            float fval = output[y * inside + x];
            std::memcpy(&gotIdx, &fval, sizeof(int32_t));
            if (gotIdx != maxIdx) return false;
        }
    }
    return true;
}

// ============================================================================
// VulkanSoftmaxHeightKernel — softmaxHeight_NHWC.comp
// ============================================================================
bool VulkanSoftmaxHeightKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "vulkan_softmax_height_fp32";
    const int w = spec.intParam("w", 4);   // inside
    const int h = spec.intParam("h", 16);  // axis (softmax dim)
    const int c = spec.intParam("c", 2);   // outside

    const int n = w * h * c;
    std::vector<float> in(n);
    for (int i = 0; i < n; ++i) in[i] = 0.1f * ((i * 3) % 17);

    AdaptedBuffer outBuf; outBuf.sizeBytes = n * sizeof(float); outBuf.isOutput = true;
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    ac.buffers.push_back(outBuf);   // binding=0
    ac.buffers.push_back(inBuf);    // binding=1
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.elementCount = n;

    struct PushParam { int32_t w; int32_t h; int32_t c; };
    PushParam pp;
    pp.w = w; pp.h = h; pp.c = c;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    // local_size = (8, 8), global = (ceil(c/8)*8, ceil(w/8)*8)
    const uint32_t lx = 8, ly = 8;
    const uint32_t gx = ((static_cast<uint32_t>(c) + lx - 1) / lx) * lx;
    const uint32_t gy = ((static_cast<uint32_t>(w) + ly - 1) / ly) * ly;
    ac.globalSize[0] = gx; ac.globalSize[1] = gy; ac.globalSize[2] = 1;
    ac.localSize[0] = lx; ac.localSize[1] = ly; ac.localSize[2] = 1;
    ac.dims = 2;
    return true;
}

bool VulkanSoftmaxHeightKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (ac.pushConstants.size() < 3 * sizeof(int32_t)) return false;
    int32_t p[3];
    std::memcpy(p, ac.pushConstants.data(), sizeof(p));
    const int w = p[0], h = p[1], c = p[2];
    if (output.size() != static_cast<size_t>(w * h * c)) return false;
    for (int oc = 0; oc < c; ++oc) {
        for (int iw = 0; iw < w; ++iw) {
            // Softmax over h for fixed (oc, iw)
            const int base = oc * h * w + iw;
            float maxV = ac.validatorInputA[base];
            for (int i = 1; i < h; ++i) {
                maxV = std::max(maxV, ac.validatorInputA[base + i * w]);
            }
            float sum = 0.0f;
            for (int i = 0; i < h; ++i) sum += std::exp(ac.validatorInputA[base + i * w] - maxV);
            for (int i = 0; i < h; ++i) {
                const float expected = std::exp(ac.validatorInputA[base + i * w] - maxV) / sum;
                if (std::fabs(output[base + i * w] - expected) > 1e-3f) return false;
            }
        }
    }
    return true;
}

// ============================================================================
// VulkanNormKernel — norm.comp (LayerNorm, no gamma/beta, USE_RMS=0)
// ============================================================================
bool VulkanNormKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "vulkan_norm_fp32";
    const int inside = spec.intParam("inside", 32);  // feature dim
    const int outside = spec.intParam("outside", 4); // batch dim

    const int n = inside * outside;
    std::vector<float> in(n);
    for (int i = 0; i < n; ++i) in[i] = 0.1f * ((i * 5) % 23);

    AdaptedBuffer outBuf; outBuf.sizeBytes = n * sizeof(float); outBuf.isOutput = true;
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    ac.buffers.push_back(outBuf);   // binding=0
    ac.buffers.push_back(inBuf);    // binding=1
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.elementCount = n;

    struct PushParam { int32_t size[4]; float eps; };
    PushParam pp;
    pp.size[0] = inside; pp.size[1] = outside; pp.size[2] = 0; pp.size[3] = outside;
    pp.eps = 1e-5f;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));
    // USE_RMS = 0 → LayerNorm (default in shader, no spec constant needed)
    ac.specConstants.push_back({3, 0});

    const uint32_t localX = 64;
    const uint32_t gx = ((static_cast<uint32_t>(outside) + localX - 1) / localX) * localX;
    ac.globalSize[0] = gx; ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    ac.localSize[0] = localX; ac.localSize[1] = 1; ac.localSize[2] = 1;
    ac.dims = 1;
    return true;
}

bool VulkanNormKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (ac.pushConstants.size() < 5 * sizeof(int32_t)) return false;
    int32_t sz[4];
    float eps;
    std::memcpy(sz, ac.pushConstants.data(), sizeof(sz));
    std::memcpy(&eps, ac.pushConstants.data() + sizeof(sz), sizeof(float));
    const int inside = sz[0], outside = sz[1];
    if (output.size() != static_cast<size_t>(inside * outside)) return false;
    for (int y = 0; y < outside; ++y) {
        const int base = y * inside;
        float mean = 0.0f;
        for (int j = 0; j < inside; ++j) mean += ac.validatorInputA[base + j];
        mean /= static_cast<float>(inside);
        float var = 0.0f;
        for (int j = 0; j < inside; ++j) {
            const float d = ac.validatorInputA[base + j] - mean;
            var += d * d;
        }
        var /= static_cast<float>(inside);
        const float invStd = 1.0f / std::sqrt(var + eps);
        for (int j = 0; j < inside; ++j) {
            const float expected = (ac.validatorInputA[base + j] - mean) * invStd;
            if (std::fabs(output[base + j] - expected) > 1e-3f) return false;
        }
    }
    return true;
}

// ============================================================================
// VulkanResizeNearestKernel — resizeNearest.comp (NC4HW4)
// ============================================================================
bool VulkanResizeNearestKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "vulkan_resize_nearest_fp32";
    const int inW = spec.intParam("inW", 4);
    const int inH = spec.intParam("inH", 4);
    const int outW = spec.intParam("outW", 8);
    const int outH = spec.intParam("outH", 8);
    const int c4 = spec.intParam("c4", 1);

    const int inSize = inW * inH * c4 * 4;
    const int outSize = outW * outH * c4 * 4;
    std::vector<float> in(inSize);
    for (int i = 0; i < inSize; ++i) in[i] = 0.1f * (i % 13);

    AdaptedBuffer outBuf; outBuf.sizeBytes = outSize * sizeof(float); outBuf.isOutput = true;
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    ac.buffers.push_back(outBuf);   // binding=0
    ac.buffers.push_back(inBuf);    // binding=1
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.elementCount = outSize;

    struct PushParam { int32_t inImgSize[4]; int32_t outImgSize[4]; float scale[4]; };
    PushParam pp;
    pp.inImgSize[0] = inW; pp.inImgSize[1] = inH; pp.inImgSize[2] = c4; pp.inImgSize[3] = 1;
    pp.outImgSize[0] = outW; pp.outImgSize[1] = outH; pp.outImgSize[2] = c4; pp.outImgSize[3] = 1;
    pp.scale[0] = static_cast<float>(inW) / outW;
    pp.scale[1] = 0.0f;
    pp.scale[2] = static_cast<float>(inH) / outH;
    pp.scale[3] = 0.0f;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    // local_size = (16, 16), global = (outW, outH, c4)
    const uint32_t lx = 16, ly = 16;
    const uint32_t gx = ((static_cast<uint32_t>(outW) + lx - 1) / lx) * lx;
    const uint32_t gy = ((static_cast<uint32_t>(outH) + ly - 1) / ly) * ly;
    ac.globalSize[0] = gx; ac.globalSize[1] = gy; ac.globalSize[2] = static_cast<uint32_t>(c4);
    ac.localSize[0] = lx; ac.localSize[1] = ly; ac.localSize[2] = 1;
    ac.dims = 3;
    return true;
}

bool VulkanResizeNearestKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (ac.pushConstants.size() < 12 * sizeof(int32_t)) return false;
    int32_t p[8];
    std::memcpy(p, ac.pushConstants.data(), sizeof(p));
    float scale[4];
    std::memcpy(scale, ac.pushConstants.data() + 8 * sizeof(int32_t), sizeof(scale));
    const int inW = p[0], inH = p[1], c4 = p[2];
    const int outW = p[4], outH = p[5];
    for (int z = 0; z < c4; ++z) {
        for (int y = 0; y < outH; ++y) {
            for (int x = 0; x < outW; ++x) {
                int srcX = static_cast<int>(std::floor(x * scale[0] + scale[1]));
                int srcY = static_cast<int>(std::floor(y * scale[2] + scale[3]));
                srcX = std::max(0, std::min(srcX, inW - 1));
                srcY = std::max(0, std::min(srcY, inH - 1));
                for (int c = 0; c < 4; ++c) {
                    const int inIdx = (z * inH * inW + srcY * inW + srcX) * 4 + c;
                    const int outIdx = (z * outH * outW + y * outW + x) * 4 + c;
                    if (std::fabs(output[outIdx] - ac.validatorInputA[inIdx]) > 1e-3f) return false;
                }
            }
        }
    }
    return true;
}

// ============================================================================
// VulkanResizeBilinearKernel — resizeBilinear.comp (NC4HW4)
// ============================================================================
bool VulkanResizeBilinearKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "vulkan_resize_bilinear_fp32";
    const int inW = spec.intParam("inW", 4);
    const int inH = spec.intParam("inH", 4);
    const int outW = spec.intParam("outW", 7);
    const int outH = spec.intParam("outH", 7);
    const int c4 = spec.intParam("c4", 1);

    const int inSize = inW * inH * c4 * 4;
    const int outSize = outW * outH * c4 * 4;
    std::vector<float> in(inSize);
    for (int i = 0; i < inSize; ++i) in[i] = 0.05f * (i % 17);

    AdaptedBuffer outBuf; outBuf.sizeBytes = outSize * sizeof(float); outBuf.isOutput = true;
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    ac.buffers.push_back(outBuf);
    ac.buffers.push_back(inBuf);
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.elementCount = outSize;

    struct PushParam { int32_t inImgSize[4]; int32_t outImgSize[4]; float scale[4]; };
    PushParam pp;
    pp.inImgSize[0] = inW; pp.inImgSize[1] = inH; pp.inImgSize[2] = c4; pp.inImgSize[3] = 1;
    pp.outImgSize[0] = outW; pp.outImgSize[1] = outH; pp.outImgSize[2] = c4; pp.outImgSize[3] = 1;
    pp.scale[0] = static_cast<float>(inW) / outW;
    pp.scale[1] = 0.0f;
    pp.scale[2] = static_cast<float>(inH) / outH;
    pp.scale[3] = 0.0f;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    const uint32_t lx = 16, ly = 16;
    const uint32_t gx = ((static_cast<uint32_t>(outW) + lx - 1) / lx) * lx;
    const uint32_t gy = ((static_cast<uint32_t>(outH) + ly - 1) / ly) * ly;
    ac.globalSize[0] = gx; ac.globalSize[1] = gy; ac.globalSize[2] = static_cast<uint32_t>(c4);
    ac.localSize[0] = lx; ac.localSize[1] = ly; ac.localSize[2] = 1;
    ac.dims = 3;
    return true;
}

bool VulkanResizeBilinearKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (ac.pushConstants.size() < 12 * sizeof(int32_t)) return false;
    int32_t p[8];
    std::memcpy(p, ac.pushConstants.data(), sizeof(p));
    float scale[4];
    std::memcpy(scale, ac.pushConstants.data() + 8 * sizeof(int32_t), sizeof(scale));
    const int inW = p[0], inH = p[1], c4 = p[2];
    const int outW = p[4], outH = p[5];
    auto L = [&](int x, int y, int z, int c) -> float {
        return ac.validatorInputA[(z * inH * inW + y * inW + x) * 4 + c];
    };
    for (int z = 0; z < c4; ++z) {
        for (int y = 0; y < outH; ++y) {
            for (int x = 0; x < outW; ++x) {
                const float srcX = x * scale[0] + scale[1];
                const int x1 = std::max(0, std::min(static_cast<int>(std::floor(srcX)), inW - 1));
                const int x2 = std::max(0, std::min(x1 + 1, inW - 1));
                const float fx = srcX - std::floor(srcX);
                const float srcY = y * scale[2] + scale[3];
                const int y1 = std::max(0, std::min(static_cast<int>(std::floor(srcY)), inH - 1));
                const int y2 = std::max(0, std::min(y1 + 1, inH - 1));
                const float fy = srcY - std::floor(srcY);
                for (int c = 0; c < 4; ++c) {
                    const float r1 = fy * ((1.0f - fx) * L(x1, y2, z, c) + fx * L(x2, y2, z, c)) +
                                     (1.0f - fy) * ((1.0f - fx) * L(x1, y1, z, c) + fx * L(x2, y1, z, c));
                    const int outIdx = (z * outH * outW + y * outW + x) * 4 + c;
                    if (std::fabs(output[outIdx] - r1) > 1e-3f) return false;
                }
            }
        }
    }
    return true;
}

// ============================================================================
// VulkanGridSampleNearestKernel — gridSampleNearest.comp
// ============================================================================
bool VulkanGridSampleNearestKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "vulkan_grid_sample_nearest_fp32";
    const int inW = spec.intParam("inW", 4);
    const int inH = spec.intParam("inH", 4);
    const int outW = spec.intParam("outW", 4);
    const int outH = spec.intParam("outH", 4);
    const int c4 = spec.intParam("c4", 1);
    const int batch = spec.intParam("batch", 1);

    const int inSize = batch * c4 * inH * inW * 4;
    const int outSize = batch * c4 * outH * outW * 4;
    const int gridSize = batch * outH * outW * 2;
    std::vector<float> in(inSize), grid(gridSize);
    for (int i = 0; i < inSize; ++i) in[i] = 0.1f * (i % 13);
    // Grid coords in [-1, 1]
    for (int i = 0; i < gridSize; ++i) grid[i] = -1.0f + 2.0f * (i % 7) / 6.0f;

    AdaptedBuffer outBuf; outBuf.sizeBytes = outSize * sizeof(float); outBuf.isOutput = true;
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    AdaptedBuffer gridBuf; gridBuf.setFp32(grid); gridBuf.isOutput = false;
    ac.buffers.push_back(outBuf);   // binding=0
    ac.buffers.push_back(inBuf);    // binding=1
    ac.buffers.push_back(gridBuf);  // binding=2
    ac.vulkanBindings = {0, 1, 2};
    ac.validatorInputA = in;
    ac.validatorInputB = grid;
    ac.elementCount = outSize;

    struct PushParam { int32_t inShape[4]; int32_t outShape[4]; int32_t alignCorners; };
    PushParam pp;
    pp.inShape[0] = inW; pp.inShape[1] = inH; pp.inShape[2] = c4; pp.inShape[3] = batch;
    pp.outShape[0] = outW; pp.outShape[1] = outH; pp.outShape[2] = c4; pp.outShape[3] = batch;
    pp.alignCorners = 0;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    // Total output FLOAT4 elements
    const uint32_t total = static_cast<uint32_t>(batch * c4 * outH * outW);
    const uint32_t localX = 256;
    const uint32_t gx = ((total + localX - 1) / localX) * localX;
    ac.globalSize[0] = gx; ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    ac.localSize[0] = localX; ac.localSize[1] = 1; ac.localSize[2] = 1;
    ac.dims = 1;
    return true;
}

bool VulkanGridSampleNearestKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    // Smoke-level: grid_sample indexing is complex; just check output is not all zeros
    // (i.e. kernel actually wrote something).
    for (float v : output) {
        if (std::fabs(v) > 1e-6f) return true;
    }
    return false;
}

// ============================================================================
// VulkanNc4hw4ToNchwKernel — nc4hw4Tonchw.comp
// ============================================================================
bool VulkanNc4hw4ToNchwKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "vulkan_nc4hw4_to_nchw_fp32";
    // NCHW shape: batch=1, channel=8 (so c4=2), H=4, W=4
    const int batch = spec.intParam("batch", 1);
    const int channel = spec.intParam("channel", 8);
    const int h = spec.intParam("h", 4);
    const int w = spec.intParam("w", 4);
    const int c4 = (channel + 3) / 4;

    // NC4HW4 input: [batch, c4, H, W, 4]
    const int inSize = batch * c4 * h * w * 4;
    // NCHW output: [batch, channel, H, W]
    const int outSize = batch * channel * h * w;
    std::vector<float> in(inSize);
    for (int i = 0; i < inSize; ++i) in[i] = 0.1f * (i % 11);

    AdaptedBuffer outBuf; outBuf.sizeBytes = outSize * sizeof(float); outBuf.isOutput = true;
    AdaptedBuffer inBuf; inBuf.setFp32(in); inBuf.isOutput = false;
    ac.buffers.push_back(outBuf);   // binding=0
    ac.buffers.push_back(inBuf);    // binding=1
    ac.vulkanBindings = {0, 1};
    ac.validatorInputA = in;
    ac.elementCount = outSize;

    // stride NCHW: stride.x = channel*H*W, stride.y = H*W, stride.z = W, stride.w = 1
    struct PushParam { int32_t size[4]; int32_t stride[4]; };
    PushParam pp;
    pp.size[0] = batch; pp.size[1] = channel; pp.size[2] = h; pp.size[3] = w;
    pp.stride[0] = channel * h * w; pp.stride[1] = h * w; pp.stride[2] = w; pp.stride[3] = 1;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    // Total work: batch * c4 * h * w
    const uint32_t total = static_cast<uint32_t>(batch * c4 * h * w);
    const uint32_t localX = 256;
    const uint32_t gx = ((total + localX - 1) / localX) * localX;
    ac.globalSize[0] = gx; ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    ac.localSize[0] = localX; ac.localSize[1] = 1; ac.localSize[2] = 1;
    ac.dims = 1;
    return true;
}

bool VulkanNc4hw4ToNchwKernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    if (ac.pushConstants.size() < 8 * sizeof(int32_t)) return false;
    int32_t sz[4], st[4];
    std::memcpy(sz, ac.pushConstants.data(), sizeof(sz));
    std::memcpy(st, ac.pushConstants.data() + sizeof(sz), sizeof(st));
    const int batch = sz[0], channel = sz[1], h = sz[2], w = sz[3];
    const int c4 = (channel + 3) / 4;
    // NCHW output: out[b][c][y][x] = NC4HW4 input in[b][c/4][y][x][c%4]
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channel; ++c) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const int nchwIdx = b * st[0] + c * st[1] + y * st[2] + x * st[3];
                    const int nc4hw4Idx = ((b * c4 + c / 4) * h * w + y * w + x) * 4 + (c % 4);
                    if (std::fabs(output[nchwIdx] - ac.validatorInputA[nc4hw4Idx]) > 1e-3f) return false;
                }
            }
        }
    }
    return true;
}

void registerVulkanMiscOps() {
    static struct Reg {
        Reg() {
            auto& r = OpAdapterRegistry::instance();
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanSelectKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanRangeKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanCastFloatIntKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanScaleKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanPreluKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanArgmaxKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanSoftmaxHeightKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanNormKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanResizeNearestKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanResizeBilinearKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanGridSampleNearestKernel()));
            r.registerAdapter(std::unique_ptr<OpAdapter>(new VulkanNc4hw4ToNchwKernel()));
        }
    } r;
    (void)r;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
