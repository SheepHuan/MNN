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
#define CONVERT_FLOAT4(x) ((FLOAT4)(x))
#define CONVERT_FLOAT(x) ((FLOAT)(x))
#define CONVERT_COMPUTE_FLOAT4(x) ((float4)(x))
#define CONVERT_COMPUTE_FLOAT2(x) ((float2)(x))
#define CONVERT_COMPUTE_FLOAT3(x) ((float3)(x))
#define CONVERT_OUTPUT4(x) ((float4)(x))
#define CONVERT_INPUT4(x) ((float4)(x))
#define CONVERT_OUTPUT8(x) ((float8)(x))
#define CONVERT_INPUT8(x) ((float8)(x))
#define CONVERT_OUTPUT3(x) ((float3)(x))
#define CONVERT_INPUT3(x) ((float3)(x))
#define CONVERT_COMPUTE_FLOAT8(x) ((float8)(x))
#define CONVERT_COMPUTE_FLOAT16(x) ((float16)(x))
#define CONVERT_OUTPUT16(x) ((float16)(x))
#define CONVERT_INPUT16(x) ((float16)(x))
#define CONVERT_FLOAT3(x) ((float3)(x))
#define CONVERT_FLOAT2(x) ((float2)(x))
#define CONVERT_FLOAT8(x) ((float8)(x))
#define CONVERT_FLOAT16(x) ((float16)(x))
#define CONVERT_FLOAT(x) ((FLOAT)(x))
#define RI_F(img, smp, coord) read_imagef(img, smp, coord)
#define WI_F(img, coord, val) write_imagef(img, coord, val)
#define SAMPLER (CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST)
#define COMPUTE_FLOAT float
#define COMPUTE_FLOAT4 float4
#define CONVERT_COMPUTE_FLOAT4(x) ((float4)(x))
#define CONVERT_OUTPUT4(x) ((float4)(x))
#define CONVERT_INPUT4(x) ((float4)(x))
#define OUTPUT_TYPE4 float4
#define INPUT_TYPE4 float4
#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,
#define GLOBAL_SIZE_3_DIMS __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,
#define GLOBAL_SIZE_DIM2 __private const int global_size_dim0, __private const int global_size_dim1,
#define DEAL_NON_UNIFORM_DIM2(input1, input2) if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { return; }
#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3) if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { return; }
#endif
#define OPERATOR in
#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }
inline float4 gelu(float4 in){
    float4 value = 0.79788458f * (0.044715f * in * in * in + in);
    float4 x2 = value * value;
    float4 dst = value > (float4)5.0f ? (float4)1.0f : (value <= -(float4)5.0f ? -(float4)1.0f :
        (value * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)))) / (135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f))));
    return (1.0f + dst) * in * 0.5f;
}

__kernel void unary_buf_c4_c4(GLOBAL_SIZE_3_DIMS
                        __global const INPUT_TYPE *input,
                        __global OUTPUT_TYPE *output,
                        __private const int width,
                        __private const int height,
                        __private const int channel,
                        __private const int batch,
                        __private const int input_pad_left, __private const int input_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    const int channel_block_idx = get_global_id(0);
    const int w                 = get_global_id(1);
    const int hb                = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(channel_block_idx, w, hb);

    const int batch_idx = hb / height;
    const int height_idx = hb % height;

    const int offset = (((batch_idx+channel_block_idx*batch)*height+height_idx)*width+w) * 4;
    float4 in  = convert_float4(vload4(0, input+offset));
    float4 out = OPERATOR;
    vstore4(CONVERT_OUTPUT4(out), 0, output+offset);
}

__kernel void unary_buf_c4_c16(GLOBAL_SIZE_3_DIMS
                        __global const INPUT_TYPE *input,
                        __global OUTPUT_TYPE *output,
                        __private const int width,
                        __private const int height,
                        __private const int channel,
                        __private const int batch,
                        __private const int input_pad_left, __private const int input_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    const int channel_block_idx = get_global_id(0);
    const int w                 = get_global_id(1);
    const int hb                = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(channel_block_idx, w, hb);

    const int batch_idx = hb / height;
    const int height_idx = hb % height;
    const int dst_width = output_pad_left+width+output_pad_right;
    const int channel16 = (channel + 15) / 16;
    const int channe_out_idx = channel_block_idx >> 2;

    const int offset = (((batch_idx+channel_block_idx*batch)*height+height_idx)*width+w) * 4;
    const int dst_offset = (((batch_idx*channel16+channe_out_idx)*height+height_idx)*dst_width+w+output_pad_left) * 16 + (channel_block_idx % 4) * 4;
    float4 in  = convert_float4(vload4(0, input+offset));
    float4 out = OPERATOR;
    vstore4(CONVERT_OUTPUT4(out), 0, output+dst_offset);
    if(w == 0){
        int pad_offset = (((batch_idx*channel16+channe_out_idx)*height+height_idx)*dst_width) * 16 + (channel_block_idx % 4) * 4;
        for(int i = 0; i < output_pad_left; ++i){
            vstore4((OUTPUT_TYPE4)0, 0, output + pad_offset + i * 16);
        }
        pad_offset += (width + output_pad_left) * 16;
        for(int i = 0; i < output_pad_right; ++i){
            vstore4((OUTPUT_TYPE4)0, 0, output + pad_offset + i * 16);
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void unary_buf_c16_c16(GLOBAL_SIZE_3_DIMS
                        __global const INPUT_TYPE *input,
                        __global OUTPUT_TYPE *output,
                        __private const int width,
                        __private const int height,
                        __private const int channel,
                        __private const int batch,
                        __private const int input_pad_left, __private const int input_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    const int channel_idx = get_group_id(0);
    const int w                 = get_global_id(1) << 2;
    const int hb                = get_global_id(2);
    const int sglid = get_sub_group_local_id();

    const int batch_idx = hb / height;
    const int height_idx = hb % height;
    const int src_width = width + input_pad_left + input_pad_right;
    const int dst_width = width + output_pad_left + output_pad_right;
    const int channel16 = (channel + 15) / 16;


    const int src_offset = (((batch_idx*channel16+channel_idx)*height+height_idx)*src_width+w+input_pad_left) * 16;
    const int dst_offset = (((batch_idx*channel16+channel_idx)*height+height_idx)*dst_width+w+output_pad_left) * 16;
    
    float4 in = convert_float4(AS_INPUT_DATA4(INTEL_SUB_GROUP_READ4((__global INTEL_DATA*)(input + src_offset))));
    float4 out = OPERATOR;

    if (w + 4 > width) {
        for (int i = 0; i < width % 4; i++) {
            output[dst_offset + i * 16 + sglid] = (OUTPUT_TYPE)out[i];
        }
    } else{
        INTEL_SUB_GROUP_WRITE4((__global INTEL_DATA*)(output + dst_offset), AS_OUTPUT_DATA4(CONVERT_OUTPUT4(out)));
    }
    if(w == 0){
        int pad_offset = (((batch_idx*channel+channel_idx)*height+height_idx)*dst_width) * 16 + sglid;
        for(int i = 0; i < output_pad_left; ++i){
            output[pad_offset + i * 16] = (OUTPUT_TYPE)0;
        }
        pad_offset += (width + output_pad_left) * 16;
        for(int i = 0; i < output_pad_right; ++i){
            output[pad_offset + i * 16] = (OUTPUT_TYPE)0;
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void unary_buf_c16_c4(GLOBAL_SIZE_3_DIMS
                        __global const INPUT_TYPE *input,
                        __global OUTPUT_TYPE *output,
                        __private const int width,
                        __private const int height,
                        __private const int channel,
                        __private const int batch,
                        __private const int input_pad_left, __private const int input_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    const int channel_idx = get_group_id(0);
    const int w                 = get_global_id(1) << 2;
    const int hb                = get_global_id(2);
    const int sglid = get_sub_group_local_id();

    const int batch_idx = hb / height;
    const int height_idx = hb % height;
    const int src_width = width + input_pad_left + input_pad_right;
    const int channel16 = (channel + 15) / 16;


    const int src_offset = (((batch_idx*channel16+channel_idx)*height+height_idx)*src_width+w+input_pad_left) * 16;
    const int dst_offset = (((batch_idx+(channel_idx<<2)*batch)*height+height_idx)*width+w) * 4;
    const int height_width = height * width * 4;
    
    float4 in = convert_float4(AS_INPUT_DATA4(INTEL_SUB_GROUP_READ4((__global INTEL_DATA*)(input + src_offset))));
    float4 out = OPERATOR;

    const int lid_x = sglid % 4;
    const int lid_y = sglid / 4;
    int block_size = w + 4 > width ? (width % 4) : 4;
    for (int i = 0; i < block_size; i++) {
        output[dst_offset + i * 4 + lid_y * height_width + lid_x] = (OUTPUT_TYPE)out[i];
    }
}

