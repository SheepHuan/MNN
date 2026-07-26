// Historical kernel: matmul_buf (no bias, no transpose, FP32)
// Origin: sources/mnn/1.2.0/source/backend/opencl/execution/cl/matmul_buf.cl
// Macro branches baked:
//   FLOAT=float
//   BIAS: not defined (no bias input)
// Entry: matmul_buf
#ifndef MNN_KERNEL_MATMUL_BUF_NOBIAS_FP32_120
#define MNN_KERNEL_MATMUL_BUF_NOBIAS_FP32_120

typedef float FLOAT;

#define GLOBAL_SIZE_2_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                             \
if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
return;                                                                                   \
}

__kernel void matmul_buf(GLOBAL_SIZE_2_DIMS __global const FLOAT* input_a,
                     __global const FLOAT* input_b,
                     __global FLOAT* output_c,
                     __private const int channels,
                     __private const int channel_blocks,
                     __private const int width_blocks) {
    const int width_blocks_idx = get_global_id(0);// output W
    const int height_idx       = get_global_id(1);// output H

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_idx);
    FLOAT4 a;
    FLOAT4 b0 = 0, b1 = 0, b2 = 0, b3 = 0;

    FLOAT result0 = 0;
    FLOAT result1 = 0;
    FLOAT result2 = 0;
    FLOAT result3 = 0;

    for (short pos = 0; pos < channel_blocks; pos += 1) {
        const int inpa_offset = height_idx * channel_blocks + pos;
        a = vload4(inpa_offset, input_a);

        short remain = (pos + 1) * 4 - channels;
        const int inpb_offset = (pos*4) * width_blocks + width_blocks_idx;

        b0 = vload4(inpb_offset, input_b);
        b1 = vload4(inpb_offset + width_blocks, input_b);
        b2 = vload4(inpb_offset + width_blocks*2, input_b);
        b3 = vload4(inpb_offset + width_blocks*3, input_b);
        if (remain == 3) {
            b1 = 0;
            b2 = 0;
            b3 = 0;
        } else if (remain == 2) {
            b2 = 0;
            b3 = 0;
        } else if (remain == 1) {
            b3 = 0;
        }

        FLOAT4 btmp0 = (FLOAT4)(b0.s0, b1.s0, b2.s0, b3.s0);
        FLOAT4 btmp1 = (FLOAT4)(b0.s1, b1.s1, b2.s1, b3.s1);
        FLOAT4 btmp2 = (FLOAT4)(b0.s2, b1.s2, b2.s2, b3.s2);
        FLOAT4 btmp3 = (FLOAT4)(b0.s3, b1.s3, b2.s3, b3.s3);

        result0 += dot(a, btmp0);
        result1 += dot(a, btmp1);
        result2 += dot(a, btmp2);
        result3 += dot(a, btmp3);
    }

    const int out_offset = height_idx * width_blocks + width_blocks_idx;
    vstore4((FLOAT4)(result0, result1, result2, result3), out_offset, output_c);
}

#endif // MNN_KERNEL_MATMUL_BUF_NOBIAS_FP32_120
