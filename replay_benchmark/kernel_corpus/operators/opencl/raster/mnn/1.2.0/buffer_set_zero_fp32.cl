// Historical kernel: buffer_set_zero
// Origin: sources/mnn/1.2.0/source/backend/opencl/execution/cl/raster_buf.cl
// Macro branches baked: FLOAT=float (FP32 path).
// Entry: buffer_set_zero
#ifndef MNN_KERNEL_BUFFER_SET_ZERO_FP32
#define MNN_KERNEL_BUFFER_SET_ZERO_FP32

typedef float FLOAT;

#define GLOBAL_SIZE_2_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

__kernel void buffer_set_zero(
    GLOBAL_SIZE_2_DIMS
    __global FLOAT *output) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(x, y);

    output[y * global_size_dim0 + x] = (FLOAT)(0.0f);
}

#endif // MNN_KERNEL_BUFFER_SET_ZERO_FP32
