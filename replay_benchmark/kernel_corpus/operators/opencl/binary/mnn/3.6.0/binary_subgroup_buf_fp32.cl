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
#define PI 3.141592653589f
__kernel void binary_buf_c4_c4_c4(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global INPUT_TYPE* input0, __global INPUT_TYPE* input1, __global OUTPUT_TYPE* output,
                         __private const int4 shape,//[N,H,W,C]
                         __private const int2 isFull,
                         __private const int activationType,
                        __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int input1_pad_left, __private const int input1_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    if (get_global_id(0) >= global_dim0 || get_global_id(1) >= global_dim1 || get_global_id(2) >= global_dim2) 
        return;
    const int channel4 = (shape.w + 3) / 4;
    const int w_idx = get_global_id(0) % shape.z;
    const int h_idx = get_global_id(0) / shape.z;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_global_id(1);

    const int offset = (((batch_idx+channel_idx*shape.x)*shape.y+h_idx)*shape.z+w_idx) * 4;
    
    #ifdef INT_COMPUTE_MOD
    int4 in0 = convert_int4(vload4(0, input0 + offset*isFull.x));
    int4 in1 = convert_int4(vload4(0, input1 + offset*isFull.x));
    if(isFull.x == 0) {
        in0 = (int4)(in0.x, in0.x, in0.x, in0.x);
    }
    if(isFull.y == 0) {
        in1 = (int4)(in1.x, in1.x, in1.x, in1.x);
    }
    int4 out = in0 % in1;
    out = ((out < (int4)0 && in1 > (int4)0) || (out > (int4)0 && in1 < (int4)0)) ? out + in1 : out;
    if(activationType == 1) {
        out = out > 0 ? out : 0;
    }
    #else
    float4 in0 = convert_float4(vload4(0, input0 + offset*isFull.x));
    float4 in1 = convert_float4(vload4(0, input1 + offset*isFull.y));
    if(isFull.x == 0) {
        in0 = (float4)(in0.x, in0.x, in0.x, in0.x);
    }
    if(isFull.y == 0) {
        in1 = (float4)(in1.x, in1.x, in1.x, in1.x);
    }
    
    float4 out = OPERATOR;
    
    if(activationType == 1) {
        out = fmax(out, (float4)0);
    }
    #endif
    vstore4(CONVERT_OUTPUT4(out), 0, output + offset);
}

__kernel void binary_buf_c4_c4_c16(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global INPUT_TYPE* input0, __global INPUT_TYPE* input1, __global OUTPUT_TYPE* output,
                         __private const int4 shape,//[N,H,W,C]
                         __private const int2 isFull,
                         __private const int activationType,
                        __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int input1_pad_left, __private const int input1_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    if (get_global_id(0) >= global_dim0 || get_global_id(1) >= global_dim1 || get_global_id(2) >= global_dim2) 
        return;
    const int channel4 = (shape.w + 3) / 4;
    const int channel16 = (shape.w + 15) / 16;
    const int w_idx = get_global_id(0) % shape.z;
    const int h_idx = get_global_id(0) / shape.z;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_global_id(1);
    const int dst_width = shape.z + output_pad_left + output_pad_right;
    const int channe_out_idx = channel_idx >> 2;

    const int offset = (((batch_idx+channel_idx*shape.x)*shape.y+h_idx)*shape.z+w_idx) * 4;
    const int dst_offset =  (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*dst_width+w_idx+output_pad_left) * 16 + (channel_idx % 4) * 4;
    #ifdef INT_COMPUTE_MOD
    int4 in0 = convert_int4(vload4(0, input0 + offset*isFull.x));
    int4 in1 = convert_int4(vload4(0, input1 + offset*isFull.x));
    if(isFull.x == 0) {
        in0 = (int4)(in0.x, in0.x, in0.x, in0.x);
    }
    if(isFull.y == 0) {
        in1 = (int4)(in1.x, in1.x, in1.x, in1.x);
    }
    int4 out = in0 % in1;
    out = ((out < (int4)0 && in1 > (int4)0) || (out > (int4)0 && in1 < (int4)0)) ? out + in1 : out;
    if(activationType == 1) {
        out = out > 0 ? out : 0;
    }
    #else
    float4 in0 = convert_float4(vload4(0, input0 + offset*isFull.x));
    float4 in1 = convert_float4(vload4(0, input1 + offset*isFull.y));
    if(isFull.x == 0) {
        in0 = (float4)(in0.x, in0.x, in0.x, in0.x);
    }
    if(isFull.y == 0) {
        in1 = (float4)(in1.x, in1.x, in1.x, in1.x);
    }
    float4 out = OPERATOR;

    if(activationType == 1) {
        out = fmax(out, (float4)0);
    }
    #endif
    vstore4(CONVERT_OUTPUT4(out), 0, output + dst_offset);
    if(w_idx == 0){
        int pad_offset = (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*dst_width) * 16 + (channel_idx % 4) * 4;
        for(int i = 0; i < output_pad_left; ++i){
            vstore4((OUTPUT_TYPE4)0, 0, output + pad_offset + i * 16);
        }
        pad_offset += (shape.z + output_pad_left) * 16;
        for(int i = 0; i < output_pad_right; ++i){
            vstore4((OUTPUT_TYPE4)0, 0, output + pad_offset + i * 16);
        }
    }
}

__kernel void binary_buf_c4_c16_c4(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global INPUT_TYPE* input0, __global INPUT_TYPE* input1, __global OUTPUT_TYPE* output,
                         __private const int4 shape,//[N,H,W,C]
                         __private const int2 isFull,
                         __private const int activationType,
                        __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int input1_pad_left, __private const int input1_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    if (get_global_id(0) >= global_dim0 || get_global_id(1) >= global_dim1 || get_global_id(2) >= global_dim2) 
        return;
    const int channel4 = (shape.w + 3) / 4;
    const int channel16 = (shape.w + 15) / 16;
    const int w_idx = get_global_id(0) % shape.z;
    const int h_idx = get_global_id(0) / shape.z;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_global_id(1);
    const int src_width = shape.z + input1_pad_left + input1_pad_right;
    const int channe_out_idx = channel_idx >> 2;

    const int offset0 = (((batch_idx+channel_idx*shape.x)*shape.y+h_idx)*shape.z+w_idx) * 4;
    const int offset1 = (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*src_width+w_idx+input1_pad_left) * 16 + (channel_idx % 4) * 4;
    #ifdef INT_COMPUTE_MOD
    int4 in0 = convert_int4(vload4(0, input0 + offset0*isFull.x));
    int4 in1 = convert_int4(vload4(0, input1 + offset1*isFull.x));
    if(isFull.x == 0) {
        in0 = (int4)(in0.x, in0.x, in0.x, in0.x);
    }
    if(isFull.y == 0) {
        in1 = (int4)(in1.x, in1.x, in1.x, in1.x);
    }
    int4 out = in0 % in1;
    out = ((out < (int4)0 && in1 > (int4)0) || (out > (int4)0 && in1 < (int4)0)) ? out + in1 : out;
    if(activationType == 1) {
        out = out > 0 ? out : 0;
    }
    #else
    float4 in0 = convert_float4(vload4(0, input0 + offset0*isFull.x));
    float4 in1 = convert_float4(vload4(0, input1 + offset1*isFull.y));
    if(isFull.x == 0) {
        in0 = (float4)(in0.x, in0.x, in0.x, in0.x);
    }
    if(isFull.y == 0) {
        in1 = (float4)(in1.x, in1.x, in1.x, in1.x);
    }
    float4 out = OPERATOR;
    if(activationType == 1) {
        out = fmax(out, (float4)0);
    }
    #endif
    vstore4(CONVERT_OUTPUT4(out), 0, output + offset0);
}

__kernel void binary_buf_c16_c4_c4(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global INPUT_TYPE* input0, __global INPUT_TYPE* input1, __global OUTPUT_TYPE* output,
                         __private const int4 shape,//[N,H,W,C]
                         __private const int2 isFull,
                         __private const int activationType,
                        __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int input1_pad_left, __private const int input1_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    if (get_global_id(0) >= global_dim0 || get_global_id(1) >= global_dim1 || get_global_id(2) >= global_dim2) 
        return;
    const int channel4 = (shape.w + 3) / 4;
    const int channel16 = (shape.w + 15) / 16;
    const int w_idx = get_global_id(0) % shape.z;
    const int h_idx = get_global_id(0) / shape.z;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_global_id(1);
    const int src_width = shape.z + input0_pad_left + input0_pad_right;
    const int channe_out_idx = channel_idx >> 2;

    const int offset1 = (((batch_idx+channel_idx*shape.x)*shape.y+h_idx)*shape.z+w_idx) * 4;
    const int offset0 = (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*src_width+w_idx+input0_pad_left) * 16 + (channel_idx % 4) * 4;
    #ifdef INT_COMPUTE_MOD
    int4 in0 = convert_int4(vload4(0, input0 + offset0*isFull.x));
    int4 in1 = convert_int4(vload4(0, input1 + offset1*isFull.x));
    if(isFull.x == 0) {
        in0 = (int4)(in0.x, in0.x, in0.x, in0.x);
    }
    if(isFull.y == 0) {
        in1 = (int4)(in1.x, in1.x, in1.x, in1.x);
    }
    int4 out = in0 % in1;
    out = ((out < (int4)0 && in1 > (int4)0) || (out > (int4)0 && in1 < (int4)0)) ? out + in1 : out;
    if(activationType == 1) {
        out = out > 0 ? out : 0;
    }
    #else
    float4 in0 = convert_float4(vload4(0, input0 + offset0*isFull.x));
    float4 in1 = convert_float4(vload4(0, input1 + offset1*isFull.y));
    if(isFull.x == 0) {
        in0 = (float4)(in0.x, in0.x, in0.x, in0.x);
    }
    if(isFull.y == 0) {
        in1 = (float4)(in1.x, in1.x, in1.x, in1.x);
    }
    float4 out = OPERATOR;
    
    if(activationType == 1) {
        out = fmax(out, (float4)0);
    }
    #endif
    vstore4(CONVERT_OUTPUT4(out), 0, output + offset1);
}

__kernel void binary_buf_c4_c16_c16(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global INPUT_TYPE* input0, __global INPUT_TYPE* input1, __global OUTPUT_TYPE* output,
                         __private const int4 shape,//[N,H,W,C]
                         __private const int2 isFull,
                         __private const int activationType,
                        __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int input1_pad_left, __private const int input1_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    if (get_global_id(0) >= global_dim0 || get_global_id(1) >= global_dim1 || get_global_id(2) >= global_dim2) 
        return;
    const int channel4 = (shape.w + 3) / 4;
    const int channel16 = (shape.w + 15) / 16;
    const int w_idx = get_global_id(0) % shape.z;
    const int h_idx = get_global_id(0) / shape.z;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_global_id(1);
    const int src_width = shape.z + input1_pad_left + input1_pad_right;
    const int dst_width = shape.z + output_pad_left + output_pad_right;
    const int channe_out_idx = channel_idx >> 2;

    const int offset0 = (((batch_idx+channel_idx*shape.x)*shape.y+h_idx)*shape.z+w_idx) * 4;
    const int offset1 = (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*src_width+w_idx+input1_pad_left) * 16 + (channel_idx % 4) * 4;
    const int dst_offset =  (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*dst_width+w_idx+output_pad_left) * 16 + (channel_idx % 4) * 4;
    #ifdef INT_COMPUTE_MOD
    int4 in0 = convert_int4(vload4(0, input0 + offset0*isFull.x));
    int4 in1 = convert_int4(vload4(0, input1 + offset1*isFull.x));
    if(isFull.x == 0) {
        in0 = (int4)(in0.x, in0.x, in0.x, in0.x);
    }
    if(isFull.y == 0) {
        in1 = (int4)(in1.x, in1.x, in1.x, in1.x);
    }
    int4 out = in0 % in1;
    out = ((out < (int4)0 && in1 > (int4)0) || (out > (int4)0 && in1 < (int4)0)) ? out + in1 : out;
    if(activationType == 1) {
        out = out > 0 ? out : 0;
    }
    #else
    float4 in0 = convert_float4(vload4(0, input0 + offset0*isFull.x));
    float4 in1 = convert_float4(vload4(0, input1 + offset1*isFull.y));
    if(isFull.x == 0) {
        in0 = (float4)(in0.x, in0.x, in0.x, in0.x);
    }
    if(isFull.y == 0) {
        in1 = (float4)(in1.x, in1.x, in1.x, in1.x);
    }
    float4 out = OPERATOR;
    
    if(activationType == 1) {
        out = fmax(out, (float4)0);
    }
    #endif
    vstore4(CONVERT_OUTPUT4(out), 0, output + dst_offset);
    if(w_idx == 0){
        int pad_offset = (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*dst_width) * 16 + (channel_idx % 4) * 4;
        for(int i = 0; i < output_pad_left; ++i){
            vstore4((OUTPUT_TYPE4)0, 0, output + pad_offset + i * 16);
        }
        pad_offset += (shape.z + output_pad_left) * 16;
        for(int i = 0; i < output_pad_right; ++i){
            vstore4((OUTPUT_TYPE4)0, 0, output + pad_offset + i * 16);
        }
    }
}

__kernel void binary_buf_c16_c4_c16(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global INPUT_TYPE* input0, __global INPUT_TYPE* input1, __global OUTPUT_TYPE* output,
                         __private const int4 shape,//[N,H,W,C]
                         __private const int2 isFull,
                         __private const int activationType,
                        __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int input1_pad_left, __private const int input1_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    if (get_global_id(0) >= global_dim0 || get_global_id(1) >= global_dim1 || get_global_id(2) >= global_dim2) 
        return;
    const int channel4 = (shape.w + 3) / 4;
    const int channel16 = (shape.w + 15) / 16;
    const int w_idx = get_global_id(0) % shape.z;
    const int h_idx = get_global_id(0) / shape.z;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_global_id(1);
    const int src_width = shape.z + input0_pad_left + input0_pad_right;
    const int dst_width = shape.z + output_pad_left + output_pad_right;
    const int channe_out_idx = channel_idx >> 2;

    const int offset1 = (((batch_idx+channel_idx*shape.x)*shape.y+h_idx)*shape.z+w_idx) * 4;
    const int offset0 = (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*src_width+w_idx+input0_pad_left) * 16 + (channel_idx % 4) * 4;
    const int dst_offset =  (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*dst_width+w_idx+output_pad_left) * 16 + (channel_idx % 4) * 4;
    #ifdef INT_COMPUTE_MOD
    int4 in0 = convert_int4(vload4(0, input0 + offset0*isFull.x));
    int4 in1 = convert_int4(vload4(0, input1 + offset1*isFull.x));
    if(isFull.x == 0) {
        in0 = (int4)(in0.x, in0.x, in0.x, in0.x);
    }
    if(isFull.y == 0) {
        in1 = (int4)(in1.x, in1.x, in1.x, in1.x);
    }
    int4 out = in0 % in1;
    out = ((out < (int4)0 && in1 > (int4)0) || (out > (int4)0 && in1 < (int4)0)) ? out + in1 : out;
    if(activationType == 1) {
        out = out > 0 ? out : 0;
    }
    #else
    float4 in0 = convert_float4(vload4(0, input0 + offset0*isFull.x));
    float4 in1 = convert_float4(vload4(0, input1 + offset1*isFull.y));
    if(isFull.x == 0) {
        in0 = (float4)(in0.x, in0.x, in0.x, in0.x);
    }
    if(isFull.y == 0) {
        in1 = (float4)(in1.x, in1.x, in1.x, in1.x);
    }
    float4 out = OPERATOR;
    
    if(activationType == 1) {
        out = fmax(out, (float4)0);
    }
    #endif
    vstore4(CONVERT_OUTPUT4(out), 0, output + dst_offset);
    if(w_idx == 0){
        int pad_offset = (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*dst_width) * 16 + (channel_idx % 4) * 4;
        for(int i = 0; i < output_pad_left; ++i){
            vstore4((OUTPUT_TYPE4)0, 0, output + pad_offset + i * 16);
        }
        pad_offset += (shape.z + output_pad_left) * 16;
        for(int i = 0; i < output_pad_right; ++i){
            vstore4((OUTPUT_TYPE4)0, 0, output + pad_offset + i * 16);
        }
    }
}



__kernel void prelu_buf_c4_c4(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global INPUT_TYPE* input0, __global INPUT_TYPE* input1, __global OUTPUT_TYPE* output,
                         __private const int4 shape,//[N,H,W,C]
                         __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right
                         ) {
    if (get_global_id(0) >= global_dim0 || get_global_id(1) >= global_dim1 || get_global_id(2) >= global_dim2) 
        return;
    const int channel4 = (shape.w + 3) / 4;
    const int w_idx = get_global_id(0) % shape.z;
    const int h_idx = get_global_id(0) / shape.z;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_global_id(1);
    
    const int offset0 = (((batch_idx+channel_idx*shape.x)*shape.y+h_idx)*shape.z+w_idx) * 4;
    const int offset1 = channel_idx * 4;
    #ifdef INT_COMPUTE_MOD
    int4 in0 = convert_int4(vload4(0, input0 + offset0));
    int4 in1 = convert_int4(vload4(0, input1 + offset1));
    int4 out = in0 % in1;
    out = ((out < (int4)0 && in1 > (int4)0) || (out > (int4)0 && in1 < (int4)0)) ? out + in1 : out;
    #else
    float4 in0 = convert_float4(vload4(0, input0 + offset0));
    float4 in1 = convert_float4(vload4(0, input1 + offset1));
    float4 out = OPERATOR;
    #endif
    vstore4(CONVERT_OUTPUT4(out), 0, output + offset0);
}

__kernel void prelu_buf_c4_c16(__private int global_dim0, __private int global_dim1,__private int global_dim2,
                         __global INPUT_TYPE* input0, __global INPUT_TYPE* input1, __global OUTPUT_TYPE* output,
                         __private const int4 shape,//[N,H,W,C]
                         __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right
                         ) {
    if (get_global_id(0) >= global_dim0 || get_global_id(1) >= global_dim1 || get_global_id(2) >= global_dim2) 
        return;
    const int channel4 = (shape.w + 3) / 4;
    const int channel16 = (shape.w + 15) / 16;
    const int w_idx = get_global_id(0) % shape.z;
    const int h_idx = get_global_id(0) / shape.z;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_global_id(1);
    const int dst_width = shape.z + output_pad_left + output_pad_right;
    const int channe_out_idx = channel_idx >> 2;
    
    const int offset0 = (((batch_idx+channel_idx*shape.x)*shape.y+h_idx)*shape.z+w_idx) * 4;
    const int offset1 = channel_idx * 4;
    const int offset =  (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*dst_width+w_idx+output_pad_left) * 16 + (channel_idx % 4) * 4;
    #ifdef INT_COMPUTE_MOD
    int4 in0 = convert_int4(vload4(0, input0 + offset0));
    int4 in1 = convert_int4(vload4(0, input1 + offset1));
    int4 out = in0 % in1;
    out = ((out < (int4)0 && in1 > (int4)0) || (out > (int4)0 && in1 < (int4)0)) ? out + in1 : out;
    #else
    float4 in0 = convert_float4(vload4(0, input0 + offset0));
    float4 in1 = convert_float4(vload4(0, input1 + offset1));
    float4 out = OPERATOR;
    #endif
    vstore4(CONVERT_OUTPUT4(out), 0, output + offset);
    if(w_idx == 0){
        int pad_offset = (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*dst_width) * 16 + (channel_idx % 4) * 4;
        for(int i = 0; i < output_pad_left; ++i){
            vstore4((OUTPUT_TYPE4)0, 0, output + pad_offset + i * 16);
        }
        pad_offset += (shape.z + output_pad_left) * 16;
        for(int i = 0; i < output_pad_right; ++i){
            vstore4((OUTPUT_TYPE4)0, 0, output + pad_offset + i * 16);
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prelu_buf_c16_c16(__private int global_dim0, __private int global_dim1,__private int global_dim2,
                         __global INPUT_TYPE* input0, __global INPUT_TYPE* input1, __global OUTPUT_TYPE* output,
                         __private const int4 shape,//[N,H,W,C]
                        __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    const int channel16 = (shape.w + 15) / 16;
    const int width_pack = (shape.z + 3) / 4;
    const int w_idx = (get_global_id(0) % width_pack) << 2;
    const int h_idx = get_global_id(0) / width_pack;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_group_id(1);
    const int sglid = get_sub_group_local_id();
    const int src_width = shape.z + input0_pad_left + input0_pad_right;
    const int dst_width = shape.z + output_pad_left + output_pad_right;

    const int offset0 = (((batch_idx*channel16+channel_idx)*shape.y+h_idx)*src_width+w_idx+input0_pad_left) * 16;
    const int offset1 = channel_idx * 16;
    const int offset =  (((batch_idx*channel16+channel_idx)*shape.y+h_idx)*dst_width+w_idx+output_pad_left) * 16;
    #ifdef INT_COMPUTE_MOD
    int4 in0 = convert_int4(AS_INPUT_DATA4(INTEL_SUB_GROUP_READ4((__global INTEL_DATA*)(input0 + offset0))));
    int4 in1 = (int4)(AS_INPUT_DATA(INTEL_SUB_GROUP_READ((__global INTEL_DATA*)(input1 + offset1))));
    int4 out = in0 % in1;
    out = ((out < (int4)0 && in1 > (int4)0) || (out > (int4)0 && in1 < (int4)0)) ? out + in1 : out;
    #else
    float4 in0 = convert_float4(AS_INPUT_DATA4(INTEL_SUB_GROUP_READ4((__global INTEL_DATA*)(input0 + offset0))));
    float4 in1 = (float4)(AS_INPUT_DATA(INTEL_SUB_GROUP_READ((__global INTEL_DATA*)(input1 + offset1))));
    
    float4 out = OPERATOR;
    #endif
    {
        if (w_idx + 4 > shape.z) {
            for (int i = 0; i < shape.z % 4; i++) {
                output[offset + i * 16 + sglid] = (OUTPUT_TYPE)out[i];
            }
        }else{
            INTEL_SUB_GROUP_WRITE4((__global INTEL_DATA*)(output + offset), AS_OUTPUT_DATA4(CONVERT_OUTPUT4(out)));
        }
    }
    if(w_idx == 0){
        int pad_offset = (((batch_idx*channel16+channel_idx)*shape.y+h_idx)*dst_width) * 16 + sglid;
        for(int i = 0; i < output_pad_left; ++i){
            output[pad_offset + i * 16] = (OUTPUT_TYPE)0;
        }
        pad_offset += (shape.z + output_pad_left) * 16;
        for(int i = 0; i < output_pad_right; ++i){
            output[pad_offset + i * 16] = (OUTPUT_TYPE)0;
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prelu_buf_c16_c4(__private int global_dim0, __private int global_dim1,__private int global_dim2,
                         __global INPUT_TYPE* input0, __global INPUT_TYPE* input1, __global OUTPUT_TYPE* output,
                         __private const int4 shape,//[N,H,W,C]
                        __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    const int channel4 = (shape.w + 3) / 4;
    const int channel16 = (shape.w + 15) / 16;
    const int width_pack = (shape.z + 3) / 4;
    const int w_idx = (get_global_id(0) % width_pack) << 2;
    const int h_idx = get_global_id(0) / width_pack;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_group_id(1);
    const int sglid = get_sub_group_local_id();
    const int src_width = shape.z + input0_pad_left + input0_pad_right;
    const int batch_width_height = shape.x * shape.z * shape.y * 4;

    const int offset0 = (((batch_idx*channel16+channel_idx)*shape.y+h_idx)*src_width+w_idx+input0_pad_left) * 16;
    const int offset1 = channel_idx * 16;
    const int offset =  (((batch_idx+(channel_idx<<2)*shape.x)*shape.y+h_idx)*shape.z+w_idx) * 4;
    #ifdef INT_COMPUTE_MOD
    int4 in0 = convert_int4(AS_INPUT_DATA4(INTEL_SUB_GROUP_READ4((__global INTEL_DATA*)(input0 + offset0))));
    int4 in1 = (int4)(AS_INPUT_DATA(INTEL_SUB_GROUP_READ((__global INTEL_DATA*)(input1 + offset1))));
    int4 out = in0 % in1;
    out = ((out < (int4)0 && in1 > (int4)0) || (out > (int4)0 && in1 < (int4)0)) ? out + in1 : out;
    #else
    float4 in0 = convert_float4(AS_INPUT_DATA4(INTEL_SUB_GROUP_READ4((__global INTEL_DATA*)(input0 + offset0))));
    float4 in1 = (float4)(AS_INPUT_DATA(INTEL_SUB_GROUP_READ((__global INTEL_DATA*)(input1 + offset1))));
    
    float4 out = OPERATOR;
    #endif

    const int lid_x = sglid % 4;
    const int lid_y = sglid / 4;
    int block_size = w_idx + 4 > shape.z ? (shape.z % 4) : 4;
    for (int i = 0; i < block_size; i++) {
        output[offset + i * 4 + lid_y * batch_width_height + lid_x] = (OUTPUT_TYPE)out[i];
    }
}



__attribute__((intel_reqd_sub_group_size(16)))
__kernel void binary_buf_c16_c16_c16(__private int global_dim0, __private int global_dim1,__private int global_dim2,
                         __global INPUT_TYPE* input0, __global INPUT_TYPE* input1, __global OUTPUT_TYPE* output,
                         __private const int4 shape,//[N,H,W,C4]
                         __private const int2 isFull,
                         __private const int activationType,
                        __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int input1_pad_left, __private const int input1_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    const int channel16 = (shape.w + 15) / 16;
    const int width_pack = (shape.z + 3) / 4;
    const int w_idx = (get_global_id(0) % width_pack) << 2;
    const int h_idx = get_global_id(0) / width_pack;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_group_id(1);
    const int sglid = get_sub_group_local_id();
    const int src0_width = shape.z + input0_pad_left + input0_pad_right;
    const int src1_width = shape.z + input1_pad_left + input1_pad_right;
    const int dst_width = shape.z + output_pad_left + output_pad_right;

    const int offset0 = (((batch_idx*channel16+channel_idx)*shape.y+h_idx)*src0_width+w_idx+input0_pad_left) * 16;
    const int offset1 = (((batch_idx*channel16+channel_idx)*shape.y+h_idx)*src1_width+w_idx+input1_pad_left) * 16;
    const int offset =  (((batch_idx*channel16+channel_idx)*shape.y+h_idx)*dst_width+w_idx+output_pad_left) * 16;
    #ifdef INT_COMPUTE_MOD
    int4 in0 = isFull.x ? convert_int4(AS_INPUT_DATA4(INTEL_SUB_GROUP_READ4((__global INTEL_DATA*)(input0 + offset0)))) : (int4)(input0[0]);
    int4 in1 = isFull.y ? convert_int4(AS_INPUT_DATA4(INTEL_SUB_GROUP_READ4((__global INTEL_DATA*)(input1 + offset1)))) : (int4)(input1[0]);
    int4 out = in0 % in1;
    out = ((out < (int4)0 && in1 > (int4)0) || (out > (int4)0 && in1 < (int4)0)) ? out + in1 : out;
    #else
    float4 in0 = isFull.x ? convert_float4(AS_INPUT_DATA4(INTEL_SUB_GROUP_READ4((__global INTEL_DATA*)(input0 + offset0)))) : (float4)(input0[0]);
    float4 in1 = isFull.y ? convert_float4(AS_INPUT_DATA4(INTEL_SUB_GROUP_READ4((__global INTEL_DATA*)(input1 + offset1)))) : (float4)(input1[0]);
    
    float4 out = OPERATOR;
    #endif
    if(activationType == 1) {
        out = fmax(out, (float4)0);
    }

    {
        if (w_idx + 4 > shape.z) {
            for (int i = 0; i < shape.z % 4; i++) {
                output[offset + i * 16 + sglid] = (OUTPUT_TYPE)out[i];
            }
        }else{
            INTEL_SUB_GROUP_WRITE4((__global INTEL_DATA*)(output + offset), AS_OUTPUT_DATA4(CONVERT_OUTPUT4(out)));
        }
    }
    if(w_idx == 0){
        int pad_offset = (((batch_idx*channel16+channel_idx)*shape.y+h_idx)*dst_width) * 16 + sglid;
        for(int i = 0; i < output_pad_left; ++i){
            output[pad_offset + i * 16] = (OUTPUT_TYPE)0;
        }
        pad_offset += (shape.z + output_pad_left) * 16;
        for(int i = 0; i < output_pad_right; ++i){
            output[pad_offset + i * 16] = (OUTPUT_TYPE)0;
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void binary_buf_c16_c16_c4(__private int global_dim0, __private int global_dim1,__private int global_dim2,
                         __global INPUT_TYPE* input0, __global INPUT_TYPE* input1, __global OUTPUT_TYPE* output,
                         __private const int4 shape,//[N,H,W,C4]
                         __private const int2 isFull,
                         __private const int activationType,
                        __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int input1_pad_left, __private const int input1_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    const int channel16 = (shape.w + 15) / 16;
    const int channel4 = (shape.w + 3) / 4;
    const int width_pack = (shape.z + 3) / 4;
    const int w_idx = (get_global_id(0) % width_pack) << 2;
    const int h_idx = get_global_id(0) / width_pack;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_group_id(1);
    const int sglid = get_sub_group_local_id();
    const int src0_width = shape.z + input0_pad_left + input0_pad_right;
    const int src1_width = shape.z + input1_pad_left + input1_pad_right;
    const int batch_width_height = shape.x * shape.z * shape.y * 4;

    const int offset0 = (((batch_idx*channel16+channel_idx)*shape.y+h_idx)*src0_width+w_idx+input0_pad_left) * 16;
    const int offset1 = (((batch_idx*channel16+channel_idx)*shape.y+h_idx)*src1_width+w_idx+input1_pad_left) * 16;
    const int offset =  (((batch_idx+(channel_idx << 2)*shape.x)*shape.y+h_idx)*shape.z+w_idx) * 4;
    #ifdef INT_COMPUTE_MOD
    int4 in0 = isFull.x ? convert_int4(AS_INPUT_DATA4(INTEL_SUB_GROUP_READ4((__global INTEL_DATA*)(input0 + offset0)))) : (int4)(input0[0]);
    int4 in1 = isFull.y ? convert_int4(AS_INPUT_DATA4(INTEL_SUB_GROUP_READ4((__global INTEL_DATA*)(input1 + offset1)))) : (int4)(input1[0]);
    int4 out = in0 % in1;
    out = ((out < (int4)0 && in1 > (int4)0) || (out > (int4)0 && in1 < (int4)0)) ? out + in1 : out;
    #else
    float4 in0 = isFull.x ? convert_float4(AS_INPUT_DATA4(INTEL_SUB_GROUP_READ4((__global INTEL_DATA*)(input0 + offset0)))) : (float4)(input0[0]);
    float4 in1 = isFull.y ? convert_float4(AS_INPUT_DATA4(INTEL_SUB_GROUP_READ4((__global INTEL_DATA*)(input1 + offset1)))) : (float4)(input1[0]);
    
    float4 out = OPERATOR;
    #endif
    if(activationType == 1) {
        out = fmax(out, (float4)0);
    }

    const int lid_x = sglid % 4;
    const int lid_y = sglid / 4;
    int block_size = w_idx + 4 > shape.z ? (shape.z % 4) : 4;
    for (int i = 0; i < block_size; i++) {
        output[offset + i * 4 + lid_y * batch_width_height + lid_x] = (OUTPUT_TYPE)out[i];
    }
}
