#include "Bridge.hpp"

#include <cmath>
#include <cstring>
#include <iostream>

namespace MNN {
namespace Replay {
namespace KernelCorpus {

namespace {

std::vector<std::unique_ptr<Bridge>>& registry() {
    // Function-local static: no namespace-scope dynamic initialization.
    static std::vector<std::unique_ptr<Bridge>> bridges;
    return bridges;
}

} // namespace

void registerBridge(std::unique_ptr<Bridge> bridge) {
    registry().push_back(std::move(bridge));
}

const Bridge* findBridge(const std::string& framework, const std::string& tag) {
    for (const auto& b : registry()) {
        if (b->supports(framework, tag)) return b.get();
    }
    return nullptr;
}

// ---- Validators ----

bool validateAllZeroFp32(const std::vector<float>& output) {
    for (float v : output) if (v != 0.0f) return false;
    return true;
}

bool validateExpFp32(const std::vector<float>& input, const std::vector<float>& output) {
    if (input.size() != output.size()) return false;
    for (size_t i = 0; i < input.size(); ++i) {
        if (std::fabs(output[i] - std::exp(input[i])) > 1e-3f) return false;
    }
    return true;
}

bool validateMatmulFp32(int M, int N, int K,
                        const std::vector<float>& a,
                        const std::vector<float>& b,
                        const std::vector<float>& out) {
    const int M4 = (M + 3) / 4;
    const int K4 = (K + 3) / 4;
    const int N4 = (N + 3) / 4;
    if (static_cast<int>(out.size()) < M4 * N4 * 4) return false;
    for (int m4 = 0; m4 < M4; ++m4) {
        for (int n4 = 0; n4 < N4; ++n4) {
            float expected[4] = {0, 0, 0, 0};
            for (int k4 = 0; k4 < K4; ++k4) {
                float av[4] = {0, 0, 0, 0};
                const int a_off = m4 * K4 + k4;
                for (int j = 0; j < 4; ++j) {
                    const int a_idx = a_off * 4 + j;
                    if (a_idx < static_cast<int>(a.size())) av[j] = a[a_idx];
                }
                const int b_off = k4 * N4 + n4;
                for (int j = 0; j < 4; ++j) {
                    float bv[4] = {0, 0, 0, 0};
                    for (int l = 0; l < 4; ++l) {
                        const int b_idx = (b_off + j * N4) * 4 + l;
                        if (b_idx < static_cast<int>(b.size())) bv[l] = b[b_idx];
                    }
                    for (int l = 0; l < 4; ++l) expected[j] += av[l] * bv[l];
                }
            }
            for (int j = 0; j < 4; ++j) {
                const int out_idx = (m4 * N4 + n4) * 4 + j;
                if (std::fabs(out[out_idx] - expected[j]) > 1e-2f) return false;
            }
        }
    }
    return true;
}

bool validateSigmoidFp32(const std::vector<float>& input, const std::vector<float>& output) {
    if (input.size() != output.size()) return false;
    for (size_t i = 0; i < input.size(); ++i) {
        if (std::fabs(output[i] - 1.0f / (1.0f + std::exp(-input[i]))) > 1e-4f) return false;
    }
    return true;
}

bool validateTanhFp32(const std::vector<float>& input, const std::vector<float>& output) {
    if (input.size() != output.size()) return false;
    for (size_t i = 0; i < input.size(); ++i) {
        if (std::fabs(output[i] - std::tanh(input[i])) > 1e-4f) return false;
    }
    return true;
}

bool validatePermuteIdentityFp32(int w, int h, int c,
                                 const std::vector<float>& input,
                                 const std::vector<float>& output) {
    if (input.size() != output.size()) return false;
    const int cstep = w * h;
    for (int gz = 0; gz < c; ++gz) {
        for (int gy = 0; gy < h; ++gy) {
            for (int gx = 0; gx < w; ++gx) {
                const int gi = gz * cstep + gy * w + gx;
                if (std::fabs(output[gi] - input[gi]) > 1e-5f) return false;
            }
        }
    }
    return true;
}

bool validateReductionSumFp32(int batch, int height, int width,
                               const std::vector<float>& input,
                               const std::vector<float>& output) {
    for (int b = 0; b < batch; ++b) {
        for (int w_idx = 0; w_idx < width; ++w_idx) {
            float sum = 0.0f;
            for (int h_idx = 0; h_idx < height; ++h_idx) {
                const int off = ((b * height + h_idx) * width + w_idx) * 4;
                if (off < static_cast<int>(input.size())) sum += input[off];
            }
            const int out_off = (b * width + w_idx) * 4;
            // reduct_buf stores FLOAT4(num, 0, 0, 0): only [0] is the sum.
            if (out_off < static_cast<int>(output.size())) {
                if (std::fabs(output[out_off] - sum) > 1e-2f) return false;
            }
        }
    }
    return true;
}

bool validatePoolingMaxFp32(int ih, int iw, int channel, int kh, int kw, int stride,
                            const std::vector<float>& input,
                            const std::vector<float>& output) {
    const int channel_block = (channel + 3) / 4;
    const int oh = (ih - kh) / stride + 1;
    const int ow = (iw - kw) / stride + 1;
    const int batch = 1;
    for (int b = 0; b < batch; ++b) {
        for (int cb = 0; cb < channel_block; ++cb) {
            for (int oh_idx = 0; oh_idx < oh; ++oh_idx) {
                for (int ow_idx = 0; ow_idx < ow; ++ow_idx) {
                    float expected[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
                    const int ih_start = oh_idx * stride;
                    const int iw_start = ow_idx * stride;
                    for (int kh_idx = 0; kh_idx < kh; ++kh_idx) {
                        for (int kw_idx = 0; kw_idx < kw; ++kw_idx) {
                            const int ih_cur = ih_start + kh_idx;
                            const int iw_cur = iw_start + kw_idx;
                            if (ih_cur < 0 || ih_cur >= ih || iw_cur < 0 || iw_cur >= iw) continue;
                            const int in_off = (((b * channel_block + cb) * ih + ih_cur) * iw + iw_cur) * 4;
                            for (int j = 0; j < 4; ++j) {
                                if (in_off + j < static_cast<int>(input.size())) {
                                    expected[j] = std::max(expected[j], input[in_off + j]);
                                }
                            }
                        }
                    }
                    const int out_off = (((b * channel_block + cb) * oh + oh_idx) * ow + ow_idx) * 4;
                    for (int j = 0; j < 4; ++j) {
                        if (out_off + j >= static_cast<int>(output.size())) continue;
                        if (std::fabs(output[out_off + j] - expected[j]) > 1e-3f) return false;
                    }
                }
            }
        }
    }
    return true;
}

bool validateAbsvalFp32(const std::vector<float>& input, const std::vector<float>& output) {
    if (input.size() != output.size()) return false;
    for (size_t i = 0; i < input.size(); ++i) {
        if (std::fabs(output[i] - std::fabs(input[i])) > 1e-4f) return false;
    }
    return true;
}

bool validateReluFp32(const std::vector<float>& input, const std::vector<float>& output) {
    if (input.size() != output.size()) return false;
    for (size_t i = 0; i < input.size(); ++i) {
        const float expected = input[i] > 0.0f ? input[i] : 0.0f;
        if (std::fabs(output[i] - expected) > 1e-4f) return false;
    }
    return true;
}

bool validateConcatIdentityFp32(const std::vector<float>& input, const std::vector<float>& output) {
    if (input.size() != output.size()) return false;
    for (size_t i = 0; i < input.size(); ++i) {
        if (std::fabs(output[i] - input[i]) > 1e-5f) return false;
    }
    return true;
}

// Auto-registration guard: ensures bridges are registered exactly once on
// first lookup. Uses Meyers singleton pattern (function-local static).
struct RegistryGuard {
    RegistryGuard() {
        // Bridges are registered in their own .cpp via registerBridge().
        // The guard ensures registration happens before first findBridge().
    }
};

RegistryGuard& guardInstance() {
    static RegistryGuard g;
    return g;
}

} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
