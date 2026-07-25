#ifndef MNN_REPLAY_OPENCL_PMU_COMPUTE_KERNELS_HPP
#define MNN_REPLAY_OPENCL_PMU_COMPUTE_KERNELS_HPP

namespace MNN {
namespace Replay {

static const char kOpenCLPmuComputeKernels[] =
    "__kernel void pmu_int_compute(__global const uint* src, __global uint* dst, uint count, uint repeats) {\n"
    "  uint gid = (uint)get_global_id(0); uint v = src[gid % count];\n"
    "  for (uint r = 0; r < repeats; ++r) v = mad24(v, 1664525u, 1013904223u);\n"
    "  if (gid < count) dst[gid] = v;\n"
    "}\n"
    "__kernel void pmu_fp32_compute(__global const float* src, __global float* dst, uint count, uint repeats) {\n"
    "  uint gid = (uint)get_global_id(0); float v = src[gid % count];\n"
    "  for (uint r = 0; r < repeats; ++r) v = fma(v, 1.0001f, 0.0001f);\n"
    "  if (gid < count) dst[gid] = v;\n"
    "}\n"
    "__kernel void pmu_fp32_throughput(__global const float* src, __global float* dst, uint count, uint repeats) {\n"
    "  uint gid = (uint)get_global_id(0); float base = src[gid % count];\n"
    "  float a = base; float b = base + 1.0f; float c = base + 2.0f; float d = base + 3.0f;\n"
    "  #pragma unroll 4\n"
    "  for (uint r = 0; r < repeats; ++r) {\n"
    "    a = fma(a, 1.0001f, 0.0001f); b = fma(b, 1.0002f, 0.0002f);\n"
    "    c = fma(c, 1.0003f, 0.0003f); d = fma(d, 1.0004f, 0.0004f);\n"
    "  }\n"
    "  if (gid < count) dst[gid] = a + b + c + d;\n"
    "}\n"
    "__kernel void pmu_constant_compute(__global float* dst, __constant float* weights, uint count, uint repeats) {\n"
    "  uint gid = (uint)get_global_id(0); float v = 0.0f;\n"
    "  for (uint r = 0; r < repeats; ++r) v = fma(v, weights[r & 15u], 1.0f);\n"
    "  dst[gid % count] = v;\n"
    "}\n"
    "__kernel void pmu_constant_bandwidth(__global float* dst, __constant float* weights, uint count, uint tableMask, uint repeats) {\n"
    "  uint gid = (uint)get_global_id(0); float v = 0.0f;\n"
    "  for (uint r = 0; r < repeats; ++r) v = fma(v, weights[(gid + r * 17u) & tableMask], 1.0f);\n"
    "  if (gid < count) dst[gid] = v;\n"
    "}\n"
    "__kernel void pmu_local_memory(__global float* dst, uint count, uint repeats, __local float* scratch) {\n"
    "  uint lid = (uint)get_local_id(0); uint gid = (uint)get_global_id(0);\n"
    "  scratch[lid] = (float)(gid & 31u); barrier(CLK_LOCAL_MEM_FENCE); float v = scratch[lid];\n"
    "  for (uint r = 0; r < repeats; ++r) { v += scratch[(lid + r) % get_local_size(0)]; barrier(CLK_LOCAL_MEM_FENCE); }\n"
    "  if (gid < count) dst[gid] = v;\n"
    "}\n"
    "__kernel void pmu_local_bandwidth(__global float* dst, uint count, uint repeats, __local float* scratch) {\n"
    "  uint lid = (uint)get_local_id(0); uint gid = (uint)get_global_id(0); uint localMask = (uint)get_local_size(0) - 1u;\n"
    "  scratch[lid] = (float)(gid & 31u); barrier(CLK_LOCAL_MEM_FENCE); float v = scratch[lid];\n"
    "  for (uint r = 0; r < repeats; ++r) v += scratch[(lid + r * 3u) & localMask];\n"
    "  if (gid < count) dst[gid] = v;\n"
    "}\n"
    "__kernel void pmu_local_barrier(__global float* dst, uint count, uint repeats, __local float* scratch) {\n"
    "  uint lid = (uint)get_local_id(0); uint gid = (uint)get_global_id(0); scratch[lid] = (float)lid;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE); float v = (float)(gid & 7u);\n"
    "  for (uint r = 0; r < repeats; ++r) barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (gid < count) dst[gid] = v + scratch[lid];\n"
    "}\n"
    "__kernel void pmu_atomic(__global volatile uint* dst, uint count, uint repeats) {\n"
    "  uint gid = (uint)get_global_id(0); for (uint r = 0; r < repeats; ++r) atomic_add(dst + (gid % count), 1u);\n"
    "}\n"
    "__kernel void pmu_atomic_contended(__global volatile uint* dst, uint repeats) {\n"
    "  for (uint r = 0; r < repeats; ++r) atomic_add(dst, 1u);\n"
    "}\n"
    "__kernel void pmu_atomic_distributed(__global volatile uint* dst, uint counterCount, uint repeats) {\n"
    "  uint gid = (uint)get_global_id(0); for (uint r = 0; r < repeats; ++r) atomic_add(dst + (gid % counterCount), 1u);\n"
    "}\n";

static const char kOpenCLPmuFp16Kernels[] =
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "__kernel void pmu_fp16_compute(__global const half* src, __global half* dst, uint count, uint repeats) {\n"
    "  uint gid = (uint)get_global_id(0); half v = src[gid % count];\n"
    "  for (uint r = 0; r < repeats; ++r) v = mad(v, (half)1.0001h, (half)0.0001h);\n"
    "  if (gid < count) dst[gid] = v;\n"
    "}\n";

static const char kOpenCLPmuFp16ThroughputKernels[] =
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "__kernel void pmu_fp16_throughput(__global const half* src, __global half* dst, uint count, uint repeats) {\n"
    "  uint gid = (uint)get_global_id(0); half base = src[gid % count];\n"
    "  half a = base; half b = base + (half)1.0h; half c = base + (half)2.0h; half d = base + (half)3.0h;\n"
    "  #pragma unroll 4\n"
    "  for (uint r = 0; r < repeats; ++r) {\n"
    "    a = mad(a, (half)1.0001h, (half)0.0001h); b = mad(b, (half)1.0002h, (half)0.0002h);\n"
    "    c = mad(c, (half)1.0003h, (half)0.0003h); d = mad(d, (half)1.0004h, (half)0.0004h);\n"
    "  }\n"
    "  if (gid < count) dst[gid] = a + b + c + d;\n"
    "}\n";

} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_OPENCL_PMU_COMPUTE_KERNELS_HPP
