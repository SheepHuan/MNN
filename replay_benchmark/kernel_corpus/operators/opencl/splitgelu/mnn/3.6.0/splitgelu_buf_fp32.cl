// Baked from MNN historical kernel. Copyright Alibaba Group Holding Limited.
// FP32 path: FLOAT=float, all macros expanded. See bake_mnn_kernels.py.
#ifndef MNN_BAKED_FP32_PREAMBLE
#define MNN_BAKED_FP32_PREAMBLE
typedef float FLOAT;
typedef float2 FLOAT2;
typedef float4 FLOAT4;
typedef float8 FLOAT8;
typedef float16 FLOAT16;
#define CONVERT_FLOAT4(x) ((FLOAT4)(x))
#define CONVERT_FLOAT(x) ((FLOAT)(x))
#define COMPUTE_FLOAT float
#define COMPUTE_FLOAT4 float4
#define CONVERT_COMPUTE_FLOAT4(x) ((float4)(x))
#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,
#define GLOBAL_SIZE_3_DIMS __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,
#define GLOBAL_SIZE_DIM2 __private const int global_size_dim0, __private const int global_size_dim1,
#define DEAL_NON_UNIFORM_DIM2(input1, input2) if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { return; }
#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3) if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { return; }
#endif
__kernel void splitgelu_buf(__private int global_dim0, __private int global_dim1,
                        __global const FLOAT * input,
                        #ifdef DOUBLE_INPUTS
                        __global const FLOAT * input1,
                        #endif
                        __global FLOAT * output,
                        __private const int4 shape
){
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        const int h   = pos.x;
        const int bc  = pos.y;

// The product of W and H is a multiple of 16
#ifdef WH_16
    const int in_offset = bc * shape.z * 2 + h * 16;
    const int out_offset = bc * shape.z + h * 16;

    float16 valueL = convert_float16(vload16(0, input + in_offset));
    float16 valueR = convert_float16(vload16(0, input + in_offset + shape.z));

    #ifdef DOUBLE_INPUTS
    float16 valueConstL = convert_float16(vload16(h, input1));
    float16 valueConstR = convert_float16(vload16(h, input1 + shape.z));
    valueL += valueConstL;
    valueR += valueConstR;
    #endif
    float16 out = (erf(valueR * (float16)0.7071067932881648) + (float16)1.0) * valueR * (float16)0.5;
    out *= valueL;
    vstore16(CONVERT_FLOAT16(out), 0, output + out_offset);

// The product of W and H is a multiple of 4
#elif defined (WH_4)

    const int in_offset = bc * shape.z * 2 + h * 4;
    const int out_offset = bc * shape.z + h * 4;

    float4 valueL = convert_float4(vload4(0, input + in_offset));
    float4 valueR = convert_float4(vload4(0, input + in_offset + shape.z));

    #ifdef DOUBLE_INPUTS
    float4 valueConstL = convert_float4(vload4(h, input1));
    float4 valueConstR = convert_float4(vload4(h, input1 + shape.z));
    valueL += valueConstL;
    valueR += valueConstR;
    #endif
    float4 out = (erf(valueR * (float4)0.7071067932881648) + (float4)1.0) * valueR * (float4)0.5;
    out *= valueL;
    vstore4(CONVERT_FLOAT4(out), 0, output + out_offset);
#else
    const int in_offset = bc * shape.z * 2 + h;
    const int out_offset = bc * shape.z + h;
    
    float valueL = (float)input[in_offset];
    float valueR = (float)input[in_offset + shape.z];

    #ifdef DOUBLE_INPUTS
    float valueConstL = input1[h];
    float valueConstR = input1[shape.z+h];
    valueL += valueConstL;
    valueR += valueConstR;
    #endif
    float out = (erf(valueR * 0.7071067932881648) + 1.0) * valueR * 0.5;
    out *= valueL;
    output[out_offset] = out;
#endif
    }
}
