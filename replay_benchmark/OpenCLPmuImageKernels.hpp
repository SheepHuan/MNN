#ifndef MNN_REPLAY_OPENCL_PMU_IMAGE_KERNELS_HPP
#define MNN_REPLAY_OPENCL_PMU_IMAGE_KERNELS_HPP

namespace MNN {
namespace Replay {

static const char kOpenCLPmuImageKernels[] =
    "__kernel void pmu_image_read(read_only image2d_t src, __global float4* dst, sampler_t sampler, uint width, uint height, uint repeats) {\n"
    "  uint x = (uint)get_global_id(0); uint y = (uint)get_global_id(1); uint gid = y * width + x; float4 value = (float4)(0.0f);\n"
    "  for (uint r = 0; r < repeats; ++r) value += read_imagef(src, sampler, (int2)((x + r) % width, y)); dst[gid] = value;\n"
    "}\n"
    "__kernel void pmu_texture_read_nearest(read_only image2d_t src, __global float4* dst, uint width, uint height, uint repeats) {\n"
    "  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
    "  uint x = (uint)get_global_id(0); uint y = (uint)get_global_id(1); uint gid = y * width + x; float4 value = (float4)(0.0f);\n"
    "  for (uint r = 0; r < repeats; ++r) value += read_imagef(src, sampler, (int2)(x, y)); dst[gid] = value;\n"
    "}\n"
    "__kernel void pmu_texture_read_linear(read_only image2d_t src, __global float4* dst, uint width, uint height, uint repeats) {\n"
    "  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;\n"
    "  uint x = (uint)get_global_id(0); uint y = (uint)get_global_id(1); uint gid = y * width + x; float2 coord = (float2)((float)x + 0.25f, (float)y + 0.25f); float4 value = (float4)(0.0f);\n"
    "  for (uint r = 0; r < repeats; ++r) value += read_imagef(src, sampler, coord); dst[gid] = value;\n"
    "}\n"
    "__kernel void pmu_image_write(write_only image2d_t dst, uint width, uint height, uint repeats) {\n"
    "  uint x = (uint)get_global_id(0); uint y = (uint)get_global_id(1); uint gid = y * width + x; float4 value = (float4)((float)(gid & 15u));\n"
    "  for (uint r = 0; r < repeats; ++r) write_imagef(dst, (int2)(x, y), value);\n"
    "}\n";

static const char kOpenCLPmuImageReadKernel[] =
    "__kernel void pmu_image_read(read_only image2d_t src, __global float4* dst, sampler_t sampler, uint width, uint height, uint repeats) {\n"
    "  uint x = (uint)get_global_id(0); uint y = (uint)get_global_id(1); uint gid = y * width + x; float4 value = (float4)(0.0f);\n"
    "  for (uint r = 0; r < repeats; ++r) value += read_imagef(src, sampler, (int2)((x + r) % width, y)); dst[gid] = value;\n"
    "}\n";

static const char kOpenCLPmuTextureNearestKernel[] =
    "__kernel void pmu_texture_read_nearest(read_only image2d_t src, __global float4* dst, uint width, uint height, uint repeats) {\n"
    "  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
    "  uint x = (uint)get_global_id(0); uint y = (uint)get_global_id(1); uint gid = y * width + x; float4 value = (float4)(0.0f);\n"
    "  for (uint r = 0; r < repeats; ++r) value += read_imagef(src, sampler, (int2)(x, y)); dst[gid] = value;\n"
    "}\n";

static const char kOpenCLPmuTextureLinearKernel[] =
    "__kernel void pmu_texture_read_linear(read_only image2d_t src, __global float4* dst, uint width, uint height, uint repeats) {\n"
    "  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;\n"
    "  uint x = (uint)get_global_id(0); uint y = (uint)get_global_id(1); uint gid = y * width + x; float2 coord = (float2)((float)x + 0.25f, (float)y + 0.25f); float4 value = (float4)(0.0f);\n"
    "  for (uint r = 0; r < repeats; ++r) value += read_imagef(src, sampler, coord); dst[gid] = value;\n"
    "}\n";

static const char kOpenCLPmuImageWriteKernel[] =
    "__kernel void pmu_image_write(write_only image2d_t dst, uint width, uint height, uint repeats) {\n"
    "  uint x = (uint)get_global_id(0); uint y = (uint)get_global_id(1); uint gid = y * width + x; float4 value = (float4)((float)(gid & 15u));\n"
    "  for (uint r = 0; r < repeats; ++r) write_imagef(dst, (int2)(x, y), value);\n"
    "}\n";

static const char kOpenCLPmuImageStrideXKernel[] =
    "__kernel void pmu_image_stride_x(read_only image2d_t src, __global float4* dst, sampler_t sampler, uint width, uint height, uint stride, uint repeats) {\n"
    "  uint x = (uint)get_global_id(0); uint y = (uint)get_global_id(1); uint gid = y * width + x; float4 value = (float4)(0.0f);\n"
    "  for (uint r = 0; r < repeats; ++r) value += read_imagef(src, sampler, (int2)((x + r * stride) % width, y));\n"
    "  dst[gid] = value;\n"
    "}\n";

static const char kOpenCLPmuImageStrideYKernel[] =
    "__kernel void pmu_image_stride_y(read_only image2d_t src, __global float4* dst, sampler_t sampler, uint width, uint height, uint stride, uint repeats) {\n"
    "  uint x = (uint)get_global_id(0); uint y = (uint)get_global_id(1); uint gid = y * width + x; float4 value = (float4)(0.0f);\n"
    "  for (uint r = 0; r < repeats; ++r) value += read_imagef(src, sampler, (int2)(x, (y + r * stride) % height));\n"
    "  dst[gid] = value;\n"
    "}\n";

} // namespace Replay
} // namespace MNN

#endif // MNN_REPLAY_OPENCL_PMU_IMAGE_KERNELS_HPP
