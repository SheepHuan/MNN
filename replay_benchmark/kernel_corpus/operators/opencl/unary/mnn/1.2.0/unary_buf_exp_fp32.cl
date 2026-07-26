// Historical kernel: unary_buf (exp variant)
// Origin: sources/mnn/1.2.0/source/backend/opencl/execution/cl/unary_buf.cl
// Macro branches baked:
//   FLOAT=float, FLOAT4=float4, CONVERT_FLOAT4(x)=(float4)(x)
//   OPERATOR=exp(in)  (unary exp)
// Entry: unary_buf
#ifndef MNN_KERNEL_UNARY_BUF_EXP_120
#define MNN_KERNEL_UNARY_BUF_EXP_120

typedef float FLOAT;
typedef float4 FLOAT4;
#define CONVERT_FLOAT4(x) (FLOAT4)(x)
#define OPERATOR exp(in)

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__kernel void unary_buf(GLOBAL_SIZE_3_DIMS
                        __global const FLOAT *input,
                        __global FLOAT *output,
                        __private const int height) {
    const int channel_block_idx = get_global_id(0);
    const int w                 = get_global_id(1);
    const int hb                = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(channel_block_idx, w, hb);

    const int batch_idx = hb / height;
    const int height_idx = hb % height;

    const int offset = (((batch_idx * global_size_dim0 + channel_block_idx) * height + height_idx) * global_size_dim1 + w) * 4;
    FLOAT4 in  = vload4(0, input + offset);
    FLOAT4 out = CONVERT_FLOAT4(OPERATOR);
    vstore4(out, 0, output + offset);
}

#endif // MNN_KERNEL_UNARY_BUF_EXP_120
