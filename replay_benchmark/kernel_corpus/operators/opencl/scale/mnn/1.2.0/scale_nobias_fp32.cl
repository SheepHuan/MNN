// Historical kernel: scale (no bias, FP32, image2d path)
// Origin: sources/mnn/1.2.0/source/backend/opencl/execution/cl/scale.cl
// Macro branches baked:
//   FLOAT=float, FLOAT4=float4
//   RI_F=read_imagef, WI_F=write_imagef
//   HAS_BIAS: not defined
//   MNN_SUPPORT_FP16: not defined
// Entry: scale
// NOTE: This kernel uses image2d_t. The OpenCL runner must support image
// buffers (clCreateImage2D) to execute this case.
#ifndef MNN_KERNEL_SCALE_NOBIAS_FP32_120
#define MNN_KERNEL_SCALE_NOBIAS_FP32_120

typedef float FLOAT;
typedef float4 FLOAT4;
#define RI_F(img, smp, coord) read_imagef(img, smp, coord)
#define WI_F(img, coord, val) write_imagef(img, coord, val)

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void scale(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __read_only image2d_t scale,
                    __write_only image2d_t output) {

    const int channel_block_idx = get_global_id(0);
    const int w                 = get_global_id(1);
    const int hb                = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(channel_block_idx, w, hb);
    const int width = global_size_dim1;

    const int pos = mad24(channel_block_idx, width, w);

    FLOAT4 in          = RI_F(input, SAMPLER, (int2)(pos, hb));
    FLOAT4 scale_value = RI_F(scale, SAMPLER, (int2)(channel_block_idx, 0));
    FLOAT4 out = in * scale_value;
    WI_F(output, (int2)(pos, hb), out);
}

#endif // MNN_KERNEL_SCALE_NOBIAS_FP32_120
