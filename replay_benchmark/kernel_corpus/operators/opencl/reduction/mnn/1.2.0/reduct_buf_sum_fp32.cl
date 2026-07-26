// Historical kernel: reduct_buf (sum variant, FP32)
// Origin: sources/mnn/1.2.0/source/backend/opencl/execution/cl/reduction_buf.cl
// Macro branches baked:
//   FLOAT=float, FLOAT4=float4
//   OPERATE=(num+in)  (sum reduction)
//   GET_AVG: not defined (sum path)
//   MNN_SUPPORT_FP16: not defined
// Entry: reduct_buf
#ifndef MNN_KERNEL_REDUCT_BUF_SUM_FP32_120
#define MNN_KERNEL_REDUCT_BUF_SUM_FP32_120

typedef float FLOAT;
typedef float4 FLOAT4;
#define OPERATE (num + in)

#define GLOBAL_SIZE_2_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1,

__kernel void reduct_buf(GLOBAL_SIZE_2_DIMS
                            __global const FLOAT* input,
                            __global FLOAT* output,
                            __private const int batch,
                            __private const int height,
                            __private const int width
                            ) {
    const int batch_idx = get_global_id(0);
    const int width_idx = get_global_id(1);

    const int inp_offset = ((batch_idx * height + 0) * width + width_idx)*4;
    FLOAT num = input[inp_offset];
    for (int h = 1; h < height; h++) {
        FLOAT in = input[inp_offset + h*width*4];
        num = OPERATE;
    }

    const int out_offset = batch_idx * width + width_idx;
    vstore4((FLOAT4)(num, 0.0, 0.0, 0.0), out_offset, output);
}

#endif // MNN_KERNEL_REDUCT_BUF_SUM_FP32_120
