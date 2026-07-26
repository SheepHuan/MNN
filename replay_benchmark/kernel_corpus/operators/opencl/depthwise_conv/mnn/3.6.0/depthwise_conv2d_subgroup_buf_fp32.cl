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
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void depthwise_conv_2d_buf_c16_c16(
    __global FLOAT* input,
    __global FLOAT* output,
    __global FLOAT* weights, 
    __global FLOAT* biases,
   __private const int inputHeight,
   __private const int inputWidth,
   __private const int Channel,
   __private const int Batch,
   __private const int input_pad_left,
   __private const int input_pad_right,
   __private const int outputHeight,
   __private const int outputWidth,
   __private const int output_pad_left,
   __private const int output_pad_right,
   __private const int pad_w,
   __private const int pad_h
) {
    const int x_blocks = (outputWidth + 7) / 8;
    const int sglid = get_sub_group_local_id();
    const int b = get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % x_blocks) * 8;
    const int y = (xy / x_blocks);

    const int c = get_group_id(1);


    const int input_x = x * STRIDE_WIDTH - pad_w;
    const int input_y = y * STRIDE_HEIGHT - pad_h;
    const int channel_pack = ((Channel + 15) / 16);

    const uint input_x_pitch = 16;
    const uint input_y_pitch = input_x_pitch * (inputWidth + input_pad_left + input_pad_right);
    const uint input_fs_pitch = input_y_pitch * (inputHeight);
    const uint input_b_pitch = input_fs_pitch * channel_pack;

    const uint input_offset = b * input_b_pitch +
                              c * input_fs_pitch + 
                              input_y * input_y_pitch +
                              (input_x + input_pad_left) * input_x_pitch;

    const uint output_x_pitch = 16;
    const uint output_y_pitch = output_x_pitch *  (outputWidth + output_pad_left + output_pad_right);
    const uint output_fs_pitch = output_y_pitch * outputHeight;
    const uint output_b_pitch = output_fs_pitch * channel_pack;

    const uint output_offset = b * output_b_pitch +
                               c * output_fs_pitch +
                               y * output_y_pitch +
                               (x + output_pad_left) * output_x_pitch;

    const uint filter_x_pitch = 16;
    const uint filter_y_pitch = filter_x_pitch * FILTER_WIDTH;
    const uint filter_is_pitch = filter_y_pitch * FILTER_HEIGHT;

    const uint filter_offset = c * filter_is_pitch;

    COMPUTE_FLOAT8 dst = (COMPUTE_FLOAT8)(as_float(intel_sub_group_block_read((__global uint*)(biases + c * 16))));

    for(int i = 0; i < FILTER_HEIGHT; ++i){
        if ((input_y + i * DILATION_HEIGHT) < 0 || (input_y + i * DILATION_HEIGHT) >= inputHeight)
            continue;
        for(int j = 0; j < FILTER_WIDTH; ++j){
            COMPUTE_FLOAT wei = as_float(intel_sub_group_block_read((__global ushort*)(weights + filter_offset + i * filter_y_pitch + j * filter_x_pitch)));
            for(int k = 0; k < 8; ++k){
                COMPUTE_FLOAT src = as_float(intel_sub_group_block_read((__global ushort*)(input + input_offset + i * DILATION_HEIGHT * input_y_pitch + (j * DILATION_WIDTH + k * STRIDE_WIDTH) * input_x_pitch)));
                dst[k] = mad(src, wei, dst[k]);
            }
        }
    }


#ifdef RELU
    dst = fmax(dst, (COMPUTE_FLOAT8)0);
#endif

#ifdef RELU6
    dst = clamp(dst, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
#endif
    
    for (int i = 0; i < 8 && (x + i) < outputWidth; i++) {
        intel_sub_group_block_write((__global uint*)(output + output_offset + i * output_x_pitch), as_uint((FLOAT)dst[i]));
    }
    if(x == 0){
        uint pad_offset = b * output_b_pitch + c * output_fs_pitch + y * output_y_pitch;
        for(int i = 0; i < output_pad_left; ++i){
            output[pad_offset + i * output_x_pitch + sglid] = 0;
        }
        pad_offset += (outputWidth + output_pad_left) * output_x_pitch;
        for(int i = 0; i < output_pad_right; ++i){
            output[pad_offset + i * output_x_pitch + sglid] = 0;
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void depthwise_conv_2d_buf_c16_c4(
    __global FLOAT* input,
    __global FLOAT* output,
    __global FLOAT* weights, 
    __global FLOAT* biases,
   __private const int inputHeight,
   __private const int inputWidth,
   __private const int Channel,
   __private const int Batch,
   __private const int input_pad_left,
   __private const int input_pad_right,
   __private const int outputHeight,
   __private const int outputWidth,
   __private const int output_pad_left,
   __private const int output_pad_right,
   __private const int pad_w,
   __private const int pad_h
) {
    const int x_blocks = (outputWidth + 7) / 8;
    const int sglid = get_sub_group_local_id();
    const int b = get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % x_blocks) * 8;
    const int y = (xy / x_blocks);

    const int c = get_group_id(1);


    const int input_x = x * STRIDE_WIDTH - pad_w;
    const int input_y = y * STRIDE_HEIGHT - pad_h;
    const int channel_pack = ((Channel + 15) / 16);

    const uint input_x_pitch = 16;
    const uint input_y_pitch = input_x_pitch * (inputWidth + input_pad_left + input_pad_right);
    const uint input_fs_pitch = input_y_pitch * (inputHeight);
    const uint input_b_pitch = input_fs_pitch * channel_pack;

    const uint input_offset = b * input_b_pitch +
                              c * input_fs_pitch + 
                              input_y * input_y_pitch +
                              (input_x + input_pad_left) * input_x_pitch;

    const uint output_x_pitch = 4;
    const uint output_y_pitch = output_x_pitch * outputWidth;
    const uint output_fs_pitch = output_y_pitch * outputHeight;
    const uint output_b_pitch = output_fs_pitch * Batch;

    const uint output_offset = (c << 2) * output_b_pitch +
                               b * output_fs_pitch +
                               y * output_y_pitch +
                               x * output_x_pitch;

    const uint filter_x_pitch = 16;
    const uint filter_y_pitch = filter_x_pitch * FILTER_WIDTH;
    const uint filter_is_pitch = filter_y_pitch * FILTER_HEIGHT;

    const uint filter_offset = c * filter_is_pitch;

    COMPUTE_FLOAT8 dst = (COMPUTE_FLOAT8)(as_float(intel_sub_group_block_read((__global uint*)(biases + c * 16))));

    for(int i = 0; i < FILTER_HEIGHT; ++i){
        if ((input_y + i * DILATION_HEIGHT) < 0 || (input_y + i * DILATION_HEIGHT) >= inputHeight)
            continue;
        for(int j = 0; j < FILTER_WIDTH; ++j){
            COMPUTE_FLOAT wei = as_float(intel_sub_group_block_read((__global ushort*)(weights + filter_offset + i * filter_y_pitch + j * filter_x_pitch)));
            for(int k = 0; k < 8; ++k){
                COMPUTE_FLOAT src = as_float(intel_sub_group_block_read((__global ushort*)(input + input_offset + i * DILATION_HEIGHT * input_y_pitch + (j * DILATION_WIDTH + k * STRIDE_WIDTH) * input_x_pitch)));
                dst[k] = mad(src, wei, dst[k]);
            }
        }
    }


#ifdef RELU
    dst = fmax(dst, (COMPUTE_FLOAT8)0);
#endif

#ifdef RELU6
    dst = clamp(dst, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
#endif

    const uint lid_x = sglid % 4;
    const uint lid_y = sglid / 4;
    for (int i = 0; i < 8 && (x + i) < outputWidth; i++) {
        output[output_offset + lid_y * output_b_pitch + i * output_x_pitch + lid_x] = dst[i];
    }
}
