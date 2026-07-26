// Historical kernel: unary_buf (exp variant, FP32)
// Origin: sources/mnn/3.6.0/source/backend/opencl/execution/cl/unary_buf.cl
// Macro branches baked:
//   INPUT_TYPE=float, OUTPUT_TYPE=float, CONVERT_OUTPUT4(x)=(float4)(x)
//   OPERATOR=exp(in)  (unary exp)
//   PACK_LEAVE: not defined (full vec4 path)
//   MNN_SUPPORT_FP16: not defined (FP32 path, CLAMP identity)
// Entry: unary_buf
#ifndef MNN_KERNEL_UNARY_BUF_EXP_FP32_36
#define MNN_KERNEL_UNARY_BUF_EXP_FP32_36

typedef float INPUT_TYPE;
typedef float OUTPUT_TYPE;
typedef float4 OUTPUT_TYPE4;
#define CONVERT_OUTPUT4(x) ((OUTPUT_TYPE4)(x))
#define OPERATOR exp(in)
#define CLAMP(a) a

#define GLOBAL_SIZE_2_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                                                   \
    }

__kernel void unary_buf(GLOBAL_SIZE_2_DIMS
                        __global const INPUT_TYPE *input,
                        __global OUTPUT_TYPE *output,
                        __private const int size) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(x, y);
    const int offset = x << 2;
    float4 in = convert_float4(vload4(0, input + offset));
    float4 out = CLAMP(OPERATOR);
    vstore4(CONVERT_OUTPUT4(out), 0, output + offset);
}

#endif // MNN_KERNEL_UNARY_BUF_EXP_FP32_36
