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
#define PI 3.141592653589f

__kernel void binary_buf(__private int global_dim0, __private int global_dim1,
                         __global INPUT_TYPE* input0, __global INPUT_TYPE* input1, __global OUTPUT_TYPE* output,
                         __private const int size,
                         __private const int activationType) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));//NCHW, 1
    
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        int offset = pos.x << 2;
#ifdef PACK_LEAVE
        if(offset + 3 >= size){
            int remain = size - offset;
            #ifdef INT_COMPUTE_MOD
            int4 in0, in1;
            int* in0_ptr = (int*)&in0;
            int* in1_ptr = (int*)&in1;
            
            for(int i = 0; i < remain; ++i){
                #ifdef A_SINGLE
                in0_ptr[i] = (int)input0[0];
                #else
                in0_ptr[i] = (int)input0[offset + i];
                #endif
        
                #ifdef B_SINGLE
                in1_ptr[i] = (int)input1[0];
                #else
                in1_ptr[i] = (int)input1[offset + i];
                #endif
            }
            int4 out = in0 % in1;
            out = ((out < (int4)0 && in1 > (int4)0) || (out > (int4)0 && in1 < (int4)0)) ? out + in1 : out;
            if(activationType == 1) {
                out = out > 0 ? out : 0;
            }
            int* out_ptr = (int*)&out;
            for(int i = 0; i < remain; ++i){
                output[offset + i] = (OUTPUT_TYPE)out_ptr[i];
            }
            #else
            float4 in0, in1;
            float* in0_ptr = (float*)&in0;
            float* in1_ptr = (float*)&in1;
            
            for(int i = 0; i < remain; ++i){
                #ifdef A_SINGLE
                in0_ptr[i] = (float)input0[0];
                #else
                in0_ptr[i] = (float)input0[offset + i];
                #endif
        
                #ifdef B_SINGLE
                in1_ptr[i] = (float)input1[0];
                #else
                in1_ptr[i] = (float)input1[offset + i];
                #endif
            }
            float4 out = OPERATOR;
            if(activationType == 1) {
                out = fmax(out, (float4)0);
            }
            float* out_ptr = (float*)&out;
            for(int i = 0; i < remain; ++i){
                output[offset + i] = (OUTPUT_TYPE)out_ptr[i];
            }
            #endif
        }else {
#endif
        #ifdef INT_COMPUTE_MOD
            #ifdef A_SINGLE
            int data0 = input0[0];
            int4 in0 = (int4)(data0, data0, data0, data0);
            #else
            int4 in0 = convert_int4(vload4(0, input0 + offset));
            #endif
        
            #ifdef B_SINGLE
            int data1 = input1[0];
            int4 in1 = (int4)(data1, data1, data1, data1);
            #else
            int4 in1 = convert_int4(vload4(0, input1 + offset));
            #endif
            
            int4 out = in0 % in1;
            out = ((out < (int4)0 && in1 > (int4)0) || (out > (int4)0 && in1 < (int4)0)) ? out + in1 : out;
        
            if(activationType == 1) {
                out = out > 0 ? out : 1;
            }
            vstore4(CONVERT_OUTPUT4(out), 0, output + offset);
        #else
            #ifdef A_SINGLE
            float data0 = input0[0];
            float4 in0 = (float4)(data0, data0, data0, data0);
            #else
            float4 in0 = convert_float4(vload4(0, input0 + offset));
            #endif
        
            #ifdef B_SINGLE
            float data1 = input1[0];
            float4 in1 = (float4)(data1, data1, data1, data1);
            #else
            float4 in1 = convert_float4(vload4(0, input1 + offset));
            #endif
            
            float4 out = OPERATOR;
        
            if(activationType == 1) {
                out = fmax(out, (float4)0);
            }
            vstore4(CONVERT_OUTPUT4(out), 0, output + offset);
        #endif
#ifdef PACK_LEAVE
        }
#endif
    }
}


__kernel void prelu_buf(__private int global_dim0, __private int global_dim1,
                         __global INPUT_TYPE* input0, __global INPUT_TYPE* input1, __global OUTPUT_TYPE* output,
                         __private const int4 shape
                         ) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));//NC4, HW
                                 
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        int b = pos.x / shape.w;
        int c = pos.x % shape.w;
        int offset = (b + c * shape.x) * (shape.y*shape.z) + pos.y;
        float4 in0 = convert_float4(vload4(offset, input0));
        float4 in1 = convert_float4(vload4(pos.x % shape.w, input1));
        float4 out = OPERATOR;
        vstore4(CONVERT_OUTPUT4(out), offset, output);
    }
}
