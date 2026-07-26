// Baked from MNN historical kernel. Copyright Alibaba Group Holding Limited.
// FP32 path: FLOAT=float, all macros expanded. See bake_mnn_kernels.py.
#ifndef MNN_BAKED_FP32_PREAMBLE
#define MNN_BAKED_FP32_PREAMBLE
typedef float FLOAT;
typedef float2 FLOAT2;
typedef float4 FLOAT4;
typedef float8 FLOAT8;
typedef float16 FLOAT16;
typedef float INPUT_TYPE;
typedef float OUTPUT_TYPE;
typedef float3 FLOAT3;
typedef float3 INPUT_TYPE3;
typedef float3 OUTPUT_TYPE3;
typedef float4 INPUT_TYPE4;
typedef float4 OUTPUT_TYPE4;
typedef float8 INPUT_TYPE8;
typedef float8 OUTPUT_TYPE8;
typedef float16 INPUT_TYPE16;
typedef float16 OUTPUT_TYPE16;
#define COMPUTE_FLOAT float
#define COMPUTE_FLOAT2 float2
#define COMPUTE_FLOAT3 float3
#define COMPUTE_FLOAT4 float4
#define COMPUTE_FLOAT8 float8
#define COMPUTE_FLOAT16 float16
#define CONVERT_FLOAT4(x) (convert_float4(x))
#define CONVERT_FLOAT(x) (convert_float(x))
#define CONVERT_FLOAT2(x) (convert_float2(x))
#define CONVERT_FLOAT3(x) (convert_float3(x))
#define CONVERT_FLOAT8(x) (convert_float8(x))
#define CONVERT_FLOAT16(x) (convert_float16(x))
#define CONVERT_COMPUTE_FLOAT2(x) (convert_float2(x))
#define CONVERT_COMPUTE_FLOAT3(x) (convert_float3(x))
#define CONVERT_COMPUTE_FLOAT4(x) (convert_float4(x))
#define CONVERT_COMPUTE_FLOAT8(x) (convert_float8(x))
#define CONVERT_COMPUTE_FLOAT16(x) (convert_float16(x))
#define CONVERT_OUTPUT4(x) (convert_float4(x))
#define CONVERT_INPUT4(x) (convert_float4(x))
#define CONVERT_OUTPUT8(x) (convert_float8(x))
#define CONVERT_INPUT8(x) (convert_float8(x))
#define CONVERT_OUTPUT3(x) (convert_float3(x))
#define CONVERT_INPUT3(x) (convert_float3(x))
#define CONVERT_OUTPUT16(x) (convert_float16(x))
#define CONVERT_INPUT16(x) (convert_float16(x))
#define AS_INPUT_DATA4(x) (convert_float4(x))
#define AS_INPUT_DATA8(x) (convert_float8(x))
#define AS_INPUT_DATA16(x) (convert_float16(x))
#define RI_F(img, smp, coord) read_imagef(img, smp, coord)
#define WI_F(img, coord, val) write_imagef(img, coord, val)
#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,
#define GLOBAL_SIZE_3_DIMS __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,
#define GLOBAL_SIZE_DIM2 __private const int global_size_dim0, __private const int global_size_dim1,
#define DEAL_NON_UNIFORM_DIM2(input1, input2) if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { return; }
#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3) if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { return; }
#endif
#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,


#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__kernel void buffer_set_zero(
                    GLOBAL_SIZE_2_DIMS
                    __global OUTPUT_TYPE *output
                    ) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    DEAL_NON_UNIFORM_DIM2(x, y);
    
    output[y*global_size_dim0 + x] = (OUTPUT_TYPE)(0.0f);
}

#define MNN_DATA_FORMAT_NCHW 0
#define MNN_DATA_FORMAT_NHWC 1
#define MNN_DATA_FORMAT_NC4HW4 2
__kernel void raster_direct_buffer(
                    GLOBAL_SIZE_3_DIMS
                    __private const int size_x,
                    __global INPUT_TYPE *input,
                    __private const int inputOffset,
                    __private const int combineSrcOffset,
                    __private const int inputStride0,
                    __private const int inputStride1,
                    __private const int inputStride2,
                    __private const int src_width,
                    __private const int src_height,
                    __private const int src_channel,
                    __private const int src_batch,
                    __global OUTPUT_TYPE *output,
                    __private const int outputOffset,
                    __private const int combineDstOffset,
                    __private const int outputStride0,
                    __private const int outputStride1,
                    __private const int outputStride2,
                    __private const int dst_width,
                    __private const int dst_height,
                    __private const int dst_channel,
                    __private const int dst_batch
                    ) {
    const int idx = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    
    DEAL_NON_UNIFORM_DIM3(idx, y, z);
    const int x = idx % size_x;
    const int id = idx / size_x;
    
    int inputIndex = inputOffset + id * combineSrcOffset + z * inputStride0 + y * inputStride1 + x * inputStride2;
    int outputIndex = outputOffset + id * combineDstOffset + z * outputStride0 + y * outputStride1 + x * outputStride2;
    int inputIndexReal = 0;
    int outputIndexReal = 0;
#if INPUT_FORMAT == MNN_DATA_FORMAT_NCHW
    inputIndexReal = inputIndex;
#elif INPUT_FORMAT == MNN_DATA_FORMAT_NHWC
    inputIndexReal = inputIndex;
#elif INPUT_FORMAT == MNN_DATA_FORMAT_NC4HW4
    int in_w = inputIndex % src_width; inputIndex /= src_width;
    int in_h = inputIndex % src_height; inputIndex /= src_height;
    int in_c = inputIndex % src_channel;
    int in_b = inputIndex / src_channel;
    inputIndexReal = (((in_b + (in_c / 4) * src_batch) * src_height + in_h) * src_width + in_w) * 4 + (in_c % 4);
#endif
    
#if OUTPUT_FORMAT == MNN_DATA_FORMAT_NCHW
    outputIndexReal = outputIndex;
#elif OUTPUT_FORMAT == MNN_DATA_FORMAT_NHWC
    outputIndexReal = outputIndex;
#elif OUTPUT_FORMAT == MNN_DATA_FORMAT_NC4HW4
    int out_w = outputIndex % dst_width; outputIndex /= dst_width;
    int out_h = outputIndex % dst_height; outputIndex /= dst_height;
    int out_c = outputIndex % dst_channel;
    int out_b = outputIndex / dst_channel;
    outputIndexReal = (((out_b + (out_c / 4) * dst_batch) * dst_height + out_h) * dst_width + out_w) * 4 + (out_c % 4);
#endif
    output[outputIndexReal] = (OUTPUT_TYPE)input[inputIndexReal];
}

__kernel void raster_nc4hw4_buffer(
                    GLOBAL_SIZE_3_DIMS
                    __global INPUT_TYPE *input,
                    __private const int inputOffset,
                    __private const int inputStride0,
                    __private const int inputStride1,
                    __private const int inputStride2,
                    __private const int inputHeight,
                    __private const int inputWidth,
                    __private const int inputChannel,
                    __global OUTPUT_TYPE *output,
                    __private const int outputOffset,
                    __private const int outputStride0,
                    __private const int outputStride1,
                    __private const int outputStride2,
                    __private const int outputHeight,
                    __private const int outputWidth,
                    __private const int outputChannel
                    ) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    int inputIndex = inputOffset + (z * inputStride0 + y * inputStride1 + x * inputStride2) * 4;
    int outputIndex = outputOffset + (z * outputStride0 + y * outputStride1 + x * outputStride2) * 4;
    
    OUTPUT_TYPE4 values = CONVERT_OUTPUT4(vload4(0, (__global INPUT_TYPE *)(input+inputIndex)));
    vstore4(values, 0, (__global OUTPUT_TYPE *)(output+outputIndex));
}
