#ifndef MNN_REPLAY_KERNEL_CORPUS_BRIDGE_CPU_REFERENCE_HPP
#define MNN_REPLAY_KERNEL_CORPUS_BRIDGE_CPU_REFERENCE_HPP

// CpuReference: backend-agnostic CPU reference implementations for kernel
// validation. Each op's reference takes scalar float input/output (NCHW or
// linear layout) and recomputes the expected result. Backend-specific
// adapters (CUDA/OpenCL/Vulkan) call these after converting their GPU output
// to scalar float, so the CPU math logic is shared across all backends.
//
// Design principle: the CPU reference computes in a simple, canonical layout
// (e.g. depthwise conv in [batch][oh][ow][c_p] NCHW). The backend validate()
// is responsible for:
//   1. Reading GPU output in the backend's native layout (NC4HW4, half2, etc.)
//   2. Converting to the canonical float layout
//   3. Calling CpuReference to get expected values
//   4. Comparing with appropriate tolerance per dtype
//
// This avoids duplicating the core math (conv reduction, softmax, etc.)
// across CUDA/OpenCL/Vulkan adapters.

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace MNN {
namespace Replay {
namespace KernelCorpus {

// ============================================================================
// Depthwise Convolution (CPU reference)
// Used by: CUDA (CudaConvDwFp32Kernel, convdw_extra, convdw_bf16),
//          OpenCL (future), Vulkan (future)
// Layout: input [batch][ih][iw][c_p], weight [kh][kw][c_p], bias [c_p]
// Output: [batch][oh][ow][c_p] = clamp(bias[c] + Σ input[sy][sx][c]*weight[fy][fx][c], minV, maxV)
// ============================================================================
struct DepthwiseConvSpec {
    int batch = 1;
    int iw = 0, ih = 0;
    int c_p = 0;       // packed channel count
    int kw = 0, kh = 0;
    int sw = 1, sh = 1;
    int dw = 1, dh = 1;  // dilation
    int pw = 0, ph = 0;  // padding
    float maxV = 1e30f, minV = -1e30f;  // clamp range
};

inline int depthwiseConvOW(const DepthwiseConvSpec& s) {
    return (s.iw + 2 * s.pw - s.dw * (s.kw - 1) - 1) / s.sw + 1;
}
inline int depthwiseConvOH(const DepthwiseConvSpec& s) {
    return (s.ih + 2 * s.ph - s.dh * (s.kh - 1) - 1) / s.sh + 1;
}

inline std::vector<float> cpuDepthwiseConv(
    const DepthwiseConvSpec& spec,
    const std::vector<float>& input,   // [batch * ih * iw * c_p]
    const std::vector<float>& weight,  // [kh * kw * c_p]
    const std::vector<float>& bias)    // [c_p]
{
    const int ow = depthwiseConvOW(spec);
    const int oh = depthwiseConvOH(spec);
    std::vector<float> out((size_t)spec.batch * oh * ow * spec.c_p, 0.0f);
    for (int ob = 0; ob < spec.batch; ++ob)
        for (int oy = 0; oy < oh; ++oy)
            for (int ox = 0; ox < ow; ++ox)
                for (int oz = 0; oz < spec.c_p; ++oz) {
                    const int ix = ox * spec.sw - spec.pw;
                    const int iy = oy * spec.sh - spec.ph;
                    float color = (oz < (int)bias.size()) ? bias[oz] : 0.0f;
                    int fxSta = std::max(0, (-ix + spec.dw - 1) / spec.dw);
                    int fySta = std::max(0, (-iy + spec.dh - 1) / spec.dh);
                    int fxEnd = std::min(spec.kw, (spec.iw - ix + spec.dw - 1) / spec.dw);
                    int fyEnd = std::min(spec.kh, (spec.ih - iy + spec.dh - 1) / spec.dh);
                    for (int fy = fySta; fy < fyEnd; ++fy) {
                        int sy = fy * spec.dh + iy;
                        for (int fx = fxSta; fx < fxEnd; ++fx) {
                            int sx = fx * spec.dw + ix;
                            int src = ((ob * spec.ih + sy) * spec.iw + sx) * spec.c_p + oz;
                            float k = weight[(fy * spec.kw + fx) * spec.c_p + oz];
                            if (src >= 0 && src < (int)input.size())
                                color += input[src] * k;
                        }
                    }
                    color = std::max(color, spec.minV);
                    color = std::min(color, spec.maxV);
                    int dst = ((ob * oh + oy) * ow + ox) * spec.c_p + oz;
                    if (dst < (int)out.size()) out[dst] = color;
                }
    return out;
}

// ============================================================================
// Reduction (CPU reference): SUM/MEAN/MAX/MIN/PROD over axis
// Used by: CUDA (reduction adapters), OpenCL/Vulkan (future)
// Layout: input [outside][axis][inside], output [outside][inside]
// ============================================================================
enum class ReduceOp { Sum, Mean, Max, Min, Prod };

inline std::vector<float> cpuReduce(
    ReduceOp op, const std::vector<float>& input,
    int outside, int axis, int inside)
{
    std::vector<float> out((size_t)outside * inside, 0.0f);
    for (int o = 0; o < outside; ++o)
        for (int x = 0; x < inside; ++x) {
            const float* src = input.data() + o * axis * inside + x;
            float result;
            switch (op) {
                case ReduceOp::Sum:
                    result = 0.0f;
                    for (int v = 0; v < axis; ++v) result += src[v * inside];
                    break;
                case ReduceOp::Mean:
                    result = 0.0f;
                    for (int v = 0; v < axis; ++v) result += src[v * inside];
                    result /= axis;
                    break;
                case ReduceOp::Max:
                    result = src[0];
                    for (int v = 1; v < axis; ++v) result = std::max(result, src[v * inside]);
                    break;
                case ReduceOp::Min:
                    result = src[0];
                    for (int v = 1; v < axis; ++v) result = std::min(result, src[v * inside]);
                    break;
                case ReduceOp::Prod:
                    result = 1.0f;
                    for (int v = 0; v < axis; ++v) result *= src[v * inside];
                    break;
            }
            out[o * inside + x] = result;
        }
    return out;
}

// ============================================================================
// Matmul / GEMV (CPU reference)
// Used by: CUDA (GENERAL_BATCH_MATMUL, gemv), OpenCL/Vulkan (future)
// Layout: C[b][n] = clamp(bias[n] + Σ_k A[b][k]*B[n][k], minV, maxV)
// B layout: [n][k] (row-major)
// ============================================================================
struct MatmulSpec {
    int batch = 1;
    int m = 0;  // batch dim (for GEMV) or M (for GEMM)
    int k = 0;  // input channel (reduction dim)
    int n = 0;  // output channel
    int k_p = 0; // k padded
    int n_p = 0; // n padded
    float maxV = 1e30f, minV = -1e30f;
};

inline std::vector<float> cpuMatmul(
    const MatmulSpec& spec,
    const std::vector<float>& input,   // [batch][k_p]
    const std::vector<float>& weight,  // [n][k_p]
    const std::vector<float>& bias)    // [n] (not n_p)
{
    std::vector<float> out((size_t)spec.batch * spec.n_p, 0.0f);
    for (int b = 0; b < spec.batch; ++b)
        for (int n = 0; n < spec.n; ++n) {
            float acc = (n < (int)bias.size()) ? bias[n] : 0.0f;
            for (int k = 0; k < spec.k; ++k)
                acc += input[b * spec.k_p + k] * weight[n * spec.k_p + k];
            acc = std::max(acc, spec.minV);
            acc = std::min(acc, spec.maxV);
            out[b * spec.n_p + n] = acc;
        }
    return out;
}

// ============================================================================
// Cast / Quantize (CPU reference)
// Used by: CUDA (CASTMIDFLOAT, FLOAT_2_INT8, INT8_2_FLOAT), OpenCL/Vulkan (future)
// ============================================================================
enum class CastOp {
    FloatToInt32,    // truncating cast: (int32_t)input
    FloatToInt8,     // quantize: clamp(round(input*scale)+zp, min, max)
    Int8ToFloat,    // dequantize: (input-zp)*scale
    FloatToBFloat16, // bf16 conversion
    BFloat16ToFloat,
};

// Single-element helpers (for validators with packed/index-mapped layouts)
inline int8_t cpuFloatToInt8(float val, float scale, int zeroPoint,
                             int clampMax = 127, int clampMin = -128) {
    int res = (int)roundf(val * scale) + zeroPoint;
    res = std::min(res, clampMax);
    res = std::max(res, clampMin);
    return (int8_t)res;
}
inline float cpuInt8ToFloat(int8_t val, float scale, int zeroPoint) {
    return ((float)val - zeroPoint) * scale;
}
inline int32_t cpuFloatToInt32(float val) {
    return (int32_t)val;
}
inline float cpuInt32ToFloat(int32_t val) {
    return (float)val;
}
inline uint8_t cpuInt32ToUint8(int32_t val) {
    return (uint8_t)val;
}

inline std::vector<float> cpuCast(
    CastOp op, const std::vector<float>& input,
    float scale = 1.0f, int zeroPoint = 0,
    int clampMax = 127, int clampMin = -128)
{
    std::vector<float> out(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        switch (op) {
            case CastOp::FloatToInt32:
                out[i] = (float)(int32_t)input[i];  // store as float (reinterpret later)
                break;
            case CastOp::FloatToInt8: {
                int res = (int)roundf(input[i] * scale) + zeroPoint;
                res = std::min(res, clampMax);
                res = std::max(res, clampMin);
                out[i] = (float)(int8_t)res;  // store as float (reinterpret as int8_t)
                break;
            }
            case CastOp::Int8ToFloat:
                out[i] = ((int8_t)(int)input[i] - zeroPoint) * scale;
                break;
            case CastOp::FloatToBFloat16:
            case CastOp::BFloat16ToFloat:
                // These require bf16 bit manipulation — handled by backend-specific helpers
                out[i] = input[i];  // placeholder
                break;
        }
    }
    return out;
}

// ============================================================================
// Pooling (CPU reference) — shared by OpenCL/Vulkan already, extended to CUDA
// Layout: NC4HW4 [cb][ih][iw][4], output [cb][oh][ow][4]
// ============================================================================
struct PoolingSpec {
    int ih = 0, iw = 0;
    int channel = 0;  // actual channels (not packed)
    int kh = 0, kw = 0;
    int stride = 1;
    int pad = 0;
};

inline std::vector<float> cpuMaxPool(
    const PoolingSpec& spec,
    const std::vector<float>& input)  // NC4HW4: [cb][ih][iw][4]
{
    const int channel_block = (spec.channel + 3) / 4;
    const int oh = (spec.ih + 2 * spec.pad - spec.kh) / spec.stride + 1;
    const int ow = (spec.iw + 2 * spec.pad - spec.kw) / spec.stride + 1;
    std::vector<float> out((size_t)channel_block * oh * ow * 4, 0.0f);
    for (int cb = 0; cb < channel_block; ++cb)
        for (int oy = 0; oy < oh; ++oy)
            for (int ox = 0; ox < ow; ++ox) {
                float expected[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
                for (int ky = 0; ky < spec.kh; ++ky)
                    for (int kx = 0; kx < spec.kw; ++kx) {
                        int iy = oy * spec.stride - spec.pad + ky;
                        int ix = ox * spec.stride - spec.pad + kx;
                        if (iy < 0 || iy >= spec.ih || ix < 0 || ix >= spec.iw) continue;
                        int in_off = ((cb * spec.ih + iy) * spec.iw + ix) * 4;
                        for (int j = 0; j < 4; ++j)
                            if (in_off + j < (int)input.size())
                                expected[j] = std::max(expected[j], input[in_off + j]);
                    }
                int out_off = ((cb * oh + oy) * ow + ox) * 4;
                for (int j = 0; j < 4; ++j)
                    if (out_off + j < (int)out.size()) out[out_off + j] = expected[j];
            }
    return out;
}

// NCHW pooling (1.2.0 layout): input [bc][ih][iw], output [bc][oh][ow]
inline std::vector<float> cpuMaxPoolNCHW(
    const PoolingSpec& spec,
    const std::vector<float>& input, int batch, int channels)
{
    const int bc = batch * channels;
    const int oh = (spec.ih + 2 * spec.pad - spec.kh) / spec.stride + 1;
    const int ow = (spec.iw + 2 * spec.pad - spec.kw) / spec.stride + 1;
    std::vector<float> out((size_t)bc * oh * ow, 0.0f);
    for (int z = 0; z < bc; ++z)
        for (int oy = 0; oy < oh; ++oy)
            for (int ox = 0; ox < ow; ++ox) {
                float mx = -65504.0f;
                for (int ky = 0; ky < spec.kh; ++ky)
                    for (int kx = 0; kx < spec.kw; ++kx) {
                        int iy = oy * spec.stride - spec.pad + ky;
                        int ix = ox * spec.stride - spec.pad + kx;
                        if (iy < 0 || iy >= spec.ih || ix < 0 || ix >= spec.iw) continue;
                        int off = z * spec.ih * spec.iw + iy * spec.iw + ix;
                        if (off < (int)input.size()) mx = std::max(mx, input[off]);
                    }
                out[z * oh * ow + oy * ow + ox] = mx;
            }
    return out;
}

inline std::vector<float> cpuAvgPool(
    const PoolingSpec& spec,
    const std::vector<float>& input)
{
    const int channel_block = (spec.channel + 3) / 4;
    const int oh = (spec.ih + 2 * spec.pad - spec.kh) / spec.stride + 1;
    const int ow = (spec.iw + 2 * spec.pad - spec.kw) / spec.stride + 1;
    std::vector<float> out((size_t)channel_block * oh * ow * 4, 0.0f);
    for (int cb = 0; cb < channel_block; ++cb)
        for (int oy = 0; oy < oh; ++oy)
            for (int ox = 0; ox < ow; ++ox) {
                float sum[4] = {0, 0, 0, 0};
                int count = 0;
                for (int ky = 0; ky < spec.kh; ++ky)
                    for (int kx = 0; kx < spec.kw; ++kx) {
                        int iy = oy * spec.stride - spec.pad + ky;
                        int ix = ox * spec.stride - spec.pad + kx;
                        if (iy < 0 || iy >= spec.ih || ix < 0 || ix >= spec.iw) continue;
                        int in_off = ((cb * spec.ih + iy) * spec.iw + ix) * 4;
                        for (int j = 0; j < 4; ++j)
                            if (in_off + j < (int)input.size()) sum[j] += input[in_off + j];
                        count++;
                    }
                int out_off = ((cb * oh + oy) * ow + ox) * 4;
                for (int j = 0; j < 4; ++j)
                    if (out_off + j < (int)out.size())
                        out[out_off + j] = (count > 0) ? sum[j] / count : 0.0f;
            }
    return out;
}

// NCHW avgpool (1.2.0 layout): input [bc][ih][iw], output [bc][oh][ow]
inline std::vector<float> cpuAvgPoolNCHW(
    const PoolingSpec& spec,
    const std::vector<float>& input, int batch, int channels)
{
    const int bc = batch * channels;
    const int oh = (spec.ih + 2 * spec.pad - spec.kh) / spec.stride + 1;
    const int ow = (spec.iw + 2 * spec.pad - spec.kw) / spec.stride + 1;
    std::vector<float> out((size_t)bc * oh * ow, 0.0f);
    for (int z = 0; z < bc; ++z)
        for (int oy = 0; oy < oh; ++oy)
            for (int ox = 0; ox < ow; ++ox) {
                float sum = 0.0f; int cnt = 0;
                for (int ky = 0; ky < spec.kh; ++ky)
                    for (int kx = 0; kx < spec.kw; ++kx) {
                        int iy = oy * spec.stride - spec.pad + ky;
                        int ix = ox * spec.stride - spec.pad + kx;
                        if (iy < 0 || iy >= spec.ih || ix < 0 || ix >= spec.iw) continue;
                        int off = z * spec.ih * spec.iw + iy * spec.iw + ix;
                        if (off < (int)input.size()) { sum += input[off]; cnt++; }
                    }
                out[z * oh * ow + oy * ow + ox] = (cnt > 0) ? sum / cnt : 0.0f;
            }
    return out;
}

// ============================================================================
// Activation functions (CPU reference)
// ============================================================================
inline float cpuSigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
inline float cpuRelu(float x, float slope = 0.0f) { return x > 0 ? x : x * slope; }
// PReLU single-element helper (for validators with version-specific c_idx mapping)
inline float cpuPreluElem(float x, float slope) { return x > 0.0f ? x : x * slope; }
inline float cpuTanh(float x) { return tanhf(x); }

// ============================================================================
// Softmax (CPU reference)
// Used by: CUDA (softmax adapters), OpenCL/Vulkan (future)
// Layout: input [outside][axis][inside], output [outside][axis][inside]
// Formula: output[o][a][i] = exp(input[o][a][i] - max(o,*,i)) / sum(o,*,i)
// ============================================================================
inline std::vector<float> cpuSoftmax(
    const std::vector<float>& input,
    int outside, int axis, int inside,
    bool expCutoff = false)  // true: clamp t >= -87 to avoid exp underflow
{
    std::vector<float> out(input.size());
    for (int o = 0; o < outside; ++o)
        for (int i = 0; i < inside; ++i) {
            const float* src = input.data() + o * axis * inside + i;
            float* dst = out.data() + o * axis * inside + i;
            float mx = src[0];
            for (int a = 1; a < axis; ++a) mx = std::max(mx, src[a * inside]);
            float sum = 0.0f;
            for (int a = 0; a < axis; ++a) {
                float t = src[a * inside] - mx;
                if (expCutoff && t < -87.0f) t = -87.0f;
                dst[a * inside] = expf(t);
                sum += dst[a * inside];
            }
            for (int a = 0; a < axis; ++a) dst[a * inside] /= sum;
        }
    return out;
}

// ============================================================================
// Comparison helper: compare GPU output (as float) against expected with tol
// ============================================================================
inline bool compareWithTolerance(
    const std::vector<float>& output,
    const std::vector<float>& expected,
    float tol = 1e-3f)
{
    if (output.size() != expected.size()) return false;
    for (size_t i = 0; i < output.size(); ++i)
        if (std::fabs(output[i] - expected[i]) > tol) return false;
    return true;
}
// Bit-exact comparison (for int32/int8 output reinterpreted as float)
inline bool compareExact(
    const std::vector<float>& output,
    const std::vector<float>& expected)
{
    if (output.size() != expected.size()) return false;
    return std::memcmp(output.data(), expected.data(), output.size() * sizeof(float)) == 0;
}

// ============================================================================
// Element-wise ops (CPU reference)
// ============================================================================
inline std::vector<float> cpuRange(float start, float step, int count) {
    std::vector<float> out(count);
    for (int i = 0; i < count; ++i) out[i] = start + i * step;
    return out;
}
inline std::vector<float> cpuSelect(const std::vector<float>& a, const std::vector<float>& b, int count) {
    std::vector<float> out(count);
    for (int i = 0; i < count; ++i) out[i] = (i % 2 > 0) ? a[i] : b[i];
    return out;
}
inline std::vector<float> cpuBinaryAdd(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
    return out;
}
inline std::vector<float> cpuAddScalar(const std::vector<float>& a, float scalar) {
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] + scalar;
    return out;
}
inline std::vector<float> cpuBinaryMul(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] * b[i];
    return out;
}
inline std::vector<float> cpuBinaryAtan2(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = atan2f(a[i], b[i]);
    return out;
}
inline std::vector<float> cpuBinaryMod(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        float bv = b[i] != 0.0f ? b[i] : 1.0f;  // avoid div-by-zero
        out[i] = a[i] - a[i] / bv;  // matches MNN kernel: x - x/y
    }
    return out;
}
inline std::vector<float> cpuBinaryLogicalOr(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = (a[i] != 0.0f || b[i] != 0.0f) ? 1.0f : 0.0f;
    return out;
}
inline std::vector<float> cpuCastIdentity(const std::vector<float>& input) {
    return input;
}
// Blit2Half: even indices = input, odd indices = 0 (half2 packing)
inline std::vector<float> cpuBlitEvenOddZero(const std::vector<float>& input) {
    std::vector<float> out(input.size(), 0.0f);
    for (size_t i = 0; i < input.size(); i += 2) out[i] = input[i];
    return out;
}

// ============================================================================
// Layernorm (CPU reference)
// Layout: input[count], output[count], gamma[count], beta[count]
// Formula: output[i] = gamma[i] * (input[i]-mean)/sqrt(var+eps) + beta[i]
// Optional: apply Swish activation: output *= sigmoid(output)
// ============================================================================
inline std::vector<float> cpuLayernorm(
    const std::vector<float>& input,
    const std::vector<float>& gamma, const std::vector<float>& beta,
    int count, float eps, bool useSwish = false)
{
    float mean = 0.0f;
    for (int i = 0; i < count; ++i) mean += input[i];
    mean /= count;
    float var = 0.0f;
    for (int i = 0; i < count; ++i) { float d = input[i] - mean; var += d * d; }
    var /= count;
    float invStd = 1.0f / sqrtf(var + eps);
    std::vector<float> out(count);
    for (int i = 0; i < count; ++i) {
        float v = (input[i] - mean) * invStd;
        v = (i < (int)gamma.size() ? gamma[i] : 1.0f) * v + (i < (int)beta.size() ? beta[i] : 0.0f);
        if (useSwish) v = v * cpuSigmoid(v);
        out[i] = v;
    }
    return out;
}

// ============================================================================
// PReLU (CPU reference)
// Layout: input[count], output[count], slope[channels]
// Formula: output[i] = (input[i] > 0) ? input[i] : input[i] * slope[c]
// ============================================================================
inline std::vector<float> cpuPrelu(
    const std::vector<float>& input, const std::vector<float>& slope,
    int count, int channels)
{
    std::vector<float> out(count);
    for (int i = 0; i < count; ++i) {
        int c = channels > 0 ? (i % channels) : 0;
        float s = (c < (int)slope.size()) ? slope[c] : 0.0f;
        out[i] = (input[i] > 0) ? input[i] : input[i] * s;
    }
    return out;
}

// ============================================================================
// Scale (CPU reference)
// Layout: input[count], output[count], scaleData[channels], biasData[channels]
// Formula: output[i] = input[i] * scale[c] + bias[c]
// ============================================================================
inline std::vector<float> cpuScale(
    const std::vector<float>& input,
    const std::vector<float>& scaleData, const std::vector<float>& biasData,
    int count, int channels)
{
    std::vector<float> out(count);
    for (int i = 0; i < count; ++i) {
        int c = channels > 0 ? (i % channels) : 0;
        float s = (c < (int)scaleData.size()) ? scaleData[c] : 1.0f;
        float b = (c < (int)biasData.size()) ? biasData[c] : 0.0f;
        out[i] = input[i] * s + b;
    }
    return out;
}
// Scale single-element (for validators with version-specific c mapping)
inline float cpuScaleElem(float x, float scale, float bias) { return x * scale + bias; }

// ============================================================================
// GridSample (CPU reference) — nearest/bilinear 2D
// Layout: input[batch][ih][iw][channels], grid[batch][oh][ow][2]
// Output: [batch][oh][ow][channels]
// ============================================================================
struct GridSampleSpec {
    int batch = 1, ih = 0, iw = 0, oh = 0, ow = 0, channels = 0;
    bool alignCorners = false;
};
inline std::vector<float> cpuGridSampleNearest(
    const std::vector<float>& input, const std::vector<float>& grid,
    const GridSampleSpec& spec)
{
    std::vector<float> out((size_t)spec.batch * spec.oh * spec.ow * spec.channels, 0.0f);
    for (int b = 0; b < spec.batch; ++b)
        for (int oy = 0; oy < spec.oh; ++oy)
            for (int ox = 0; ox < spec.ow; ++ox) {
                int gidx = (b * spec.oh + oy) * spec.ow + ox;
                float gx = grid[gidx * 2 + 0];
                float gy = grid[gidx * 2 + 1];
                int ix = (int)roundf((gx + 1.0f) * (spec.iw - 1) / 2.0f);
                int iy = (int)roundf((gy + 1.0f) * (spec.ih - 1) / 2.0f);
                ix = std::max(0, std::min(spec.iw - 1, ix));
                iy = std::max(0, std::min(spec.ih - 1, iy));
                for (int c = 0; c < spec.channels; ++c) {
                    int src = ((b * spec.ih + iy) * spec.iw + ix) * spec.channels + c;
                    int dst = ((b * spec.oh + oy) * spec.ow + ox) * spec.channels + c;
                    out[dst] = input[src];
                }
            }
    return out;
}

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_KERNEL_CORPUS_BRIDGE_CPU_REFERENCE_HPP
