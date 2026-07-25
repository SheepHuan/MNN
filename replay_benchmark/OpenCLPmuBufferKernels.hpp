#ifndef MNN_REPLAY_OPENCL_PMU_BUFFER_KERNELS_HPP
#define MNN_REPLAY_OPENCL_PMU_BUFFER_KERNELS_HPP

namespace MNN {
namespace Replay {

static const char kOpenCLPmuBufferKernels[] =
    "__kernel void pmu_empty(__global uint* sink) {\n"
    "  if (get_global_id(0) == 0) sink[0] += 1u;\n"
    "}\n"
    "__kernel void pmu_control(__global uint* sink, uint repeats) {\n"
    "  uint gid = (uint)get_global_id(0); uint value = gid;\n"
    "  for (uint r = 0; r < repeats; ++r) value = mad24(value, 1664525u, 1013904223u);\n"
    "  if (gid == 0) sink[0] = value;\n"
    "}\n"
    "__kernel void pmu_buffer_reuse(__global const float* src, __global float* dst, uint count, uint active, uint repeats) {\n"
    "  uint gid = (uint)get_global_id(0); float v = 0.0f;\n"
    "  for (uint r = 0; r < repeats; ++r) { uint index = (gid * 17u + r * 13u) % active; v += src[index]; }\n"
    "  if (gid < count) dst[gid] = v;\n"
    "}\n"
    "__kernel void pmu_buffer_stride(__global const float* src, __global float* dst, uint count, uint activeMask, uint stride, uint repeats) {\n"
    "  uint gid = (uint)get_global_id(0); float v = 0.0f;\n"
    "  for (uint r = 0; r < repeats; ++r) { uint index = (gid * stride + r * stride) & activeMask; v += src[index]; }\n"
    "  if (gid < count) dst[gid] = v;\n"
    "}\n"
    "__kernel void pmu_buffer_pchase(__global const uint* next, __global uint* dst, uint count, uint repeats) {\n"
    "  uint gid = (uint)get_global_id(0); uint index = gid % count; uint value = index;\n"
    "  for (uint r = 0; r < repeats; ++r) { index = next[index]; value += index; }\n"
    "  if (gid < count) dst[gid] = value;\n"
    "}\n"
    "__kernel void pmu_buffer_vec4(__global const float4* src, __global float4* dst, uint count, uint repeats) {\n"
    "  uint gid = (uint)get_global_id(0); float4 v = src[gid % count];\n"
    "  for (uint r = 0; r < repeats; ++r) v = fma(v, (float4)(1.0001f), (float4)(0.0001f));\n"
    "  if (gid < count) dst[gid] = v;\n"
    "}\n"
    "__kernel void pmu_buffer_stream(__global const float* src, __global float* dst, uint count, uint repeats) {\n"
    "  uint gid = (uint)get_global_id(0); float v = 0.0f;\n"
    "  for (uint r = 0; r < repeats; ++r) { uint index = (gid + r * 8191u) % count; v += src[index]; }\n"
    "  if (gid < count) dst[gid] = v;\n"
    "}\n"
    "__kernel void pmu_buffer_write(__global float* dst, uint count, uint repeats) {\n"
    "  uint gid = (uint)get_global_id(0); float v = (float)(gid & 31u);\n"
    "  for (uint r = 0; r < repeats; ++r) v = v * 1.0001f + (float)r;\n"
    "  if (gid < count) dst[gid] = v;\n"
    "}\n";

} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_OPENCL_PMU_BUFFER_KERNELS_HPP
