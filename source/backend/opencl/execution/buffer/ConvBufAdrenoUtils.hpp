//
//  ConvBufAdrenoUtils.hpp
//  MNN
//

#pragma once

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include <string>

#include "backend/opencl/core/runtime/OpenCLRuntime.hpp"

namespace MNN {
namespace OpenCL {
namespace ConvAdreno {

int compactDenseMBucket(int M, int N, int K);
int compactDenseNKBucket(int N, int K);
int compactDenseMaxRows(int N, int K);
bool shouldTuneCompactDenseFamily(OpenCLRuntime* runtime, int quantBit, int M, int N, int K);
bool preferCompactDenseWeightBuffer(OpenCLRuntime* runtime, int quantBit, int inputChannels, int outputChannels);
std::string compactDenseFamilyTuneKey(int inputChannels, int outputChannels, int quantBit, int rows);

} // namespace ConvAdreno
} // namespace OpenCL
} // namespace MNN

#endif
