//
//  ConvBufAdrenoUtils.cpp
//  MNN
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/ConvBufAdrenoUtils.hpp"

#include <algorithm>

namespace MNN {
namespace OpenCL {
namespace ConvAdreno {
namespace {

constexpr int kCompactDenseFamilyTuneKeyVersion = 7;

} // namespace

int compactDenseMBucket(int M, int N, int K) {
    const int minNK = std::max(1, std::min(N, K));
    if (8LL * M <= minNK) {
        return 0;
    }
    if (4LL * M <= minNK) {
        return 1;
    }
    if (2LL * M <= minNK) {
        return 2;
    }
    if (M <= minNK) {
        return 3;
    }
    if (2LL * M <= 3LL * minNK) {
        return 4;
    }
    return 5;
}

int compactDenseNKBucket(int N, int K) {
    const int minNK = std::max(1, std::min(N, K));
    const int maxNK = std::max(N, K);
    if (4LL * minNK >= 3LL * maxNK) {
        return 0;
    }
    if (2LL * minNK >= maxNK) {
        return 1;
    }
    if (4LL * minNK >= maxNK) {
        return 2;
    }
    return 3;
}

int compactDenseMaxRows(int N, int K) {
    const int minNK = std::max(1, std::min(N, K));
    return std::min(1536, minNK * 2);
}

bool shouldTuneCompactDenseFamily(OpenCLRuntime* runtime, int quantBit, int M, int N, int K) {
    return runtime != nullptr &&
           runtime->getGpuType() == ADRENO &&
           quantBit == 4 &&
           M > 16 && M <= compactDenseMaxRows(N, K) &&
           N >= 1024 && K >= 1024;
}

bool preferCompactDenseWeightBuffer(OpenCLRuntime* runtime, int quantBit, int inputChannels, int outputChannels) {
    if (runtime == nullptr || runtime->getGpuType() != ADRENO || quantBit != 4) {
        return false;
    }
    (void)inputChannels;
    (void)outputChannels;
    // Rhino low-memory compact dense benches consistently show that the actual
    // weight-only 1x1 path benefits from image-backed weights. The previous
    // Adreno-wide "prefer buffer" override was derived from a different
    // generic-image path and incorrectly pushed PIC compact dense onto the
    // slower storage choice.
    return false;
}

std::string compactDenseFamilyTuneKey(int inputChannels, int outputChannels, int quantBit, int rows) {
    return "convBufLowMemory_adreno_family_exact_v" + std::to_string(kCompactDenseFamilyTuneKeyVersion) + "_" +
           std::to_string(inputChannels) + "_" + std::to_string(outputChannels) +
           "_q" + std::to_string(quantBit) +
           "_mb" + std::to_string(compactDenseMBucket(rows, outputChannels, inputChannels)) +
           "_nkb" + std::to_string(compactDenseNKBucket(outputChannels, inputChannels));
}

} // namespace ConvAdreno
} // namespace OpenCL
} // namespace MNN

#endif
