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
#define AS_INPUT_DATA4(x) ((float4)(x))
#define AS_INPUT_DATA8(x) ((float8)(x))
#define AS_INPUT_DATA16(x) ((float16)(x))
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
#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,
#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

// convert data from buffer(nhwc) to buffer(nc4hw4)
__kernel void nhwc_buffer_to_nc4hw4_buffer(GLOBAL_SIZE_2_DIMS
                                   #ifdef BUFFER_FORMAT_INP_TRANS
                                   __global const float *input_ptr,
                                   #else
                                   __global const FLOAT *input_ptr,
                                   #endif
                                   __private const int height,
                                   __private const int width, __private const int channels,
                                   __global FLOAT *output) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx     = image_height_idx / height;
    const int height_idx    = image_height_idx % height;
    const int width_idx     = image_width_idx % width;
    const int channel_4_idx = (image_width_idx / width) << 2;
    const int buffer_offset = ((batch_idx * height + height_idx) * width + width_idx) * channels + channel_4_idx;

    const int remain_channel                = channels - channel_4_idx;
    FLOAT4 values                           = 0;

    #ifdef BUFFER_FORMAT_INP_TRANS
    __global const float *input_current_ptr = input_ptr + buffer_offset;
    values                                  = CONVERT_FLOAT4(vload4(0, input_current_ptr));
    #else
    __global const FLOAT *input_current_ptr = input_ptr + buffer_offset;
    values                                  = vload4(0, input_current_ptr);
    #endif

    if (remain_channel == 3) {
        values.w = 0;
    } else if (remain_channel == 2) {
        values.z = 0;
        values.w = 0;
    } else if (remain_channel == 1) {
        values.y = 0;
        values.z = 0;
        values.w = 0;
    }
    const int out_offset = (((batch_idx * ((channels+3)/4) + channel_4_idx/4) * height + height_idx) * width + width_idx)*4;
    vstore4(values, 0, output+out_offset);
}

// convert data from buffer(nchw) to buffer(nc4hw4)
__kernel void nchw_buffer_to_nc4hw4_buffer(GLOBAL_SIZE_2_DIMS
                                   #ifdef BUFFER_FORMAT_INP_TRANS
                                   __global const float *input_ptr,
                                   #else
                                   __global const FLOAT *input_ptr,
                                   #endif
                                   __private const int height, __private const int width, __private const int channels,
                                   __global FLOAT *output) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);
    
    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx     = image_height_idx / height;
    const int height_idx    = image_height_idx % height;
    const int width_idx     = image_width_idx % width;
    const int channel_4_idx = image_width_idx / width << 2;
    const int buffer_offset = ((batch_idx * channels + channel_4_idx) * height + height_idx) * width + width_idx;

    const int remain_channel    = channels - channel_4_idx;
    const int height_width_size = height * width;
    FLOAT4 output_values    = 0;

    if (remain_channel >= 4) {
        int offset      = buffer_offset;
        output_values.x = (FLOAT)*(input_ptr + offset);
        offset += height_width_size;
        output_values.y = (FLOAT)*(input_ptr + offset);
        offset += height_width_size;
        output_values.z = (FLOAT)*(input_ptr + offset);
        offset += height_width_size;
        output_values.w = (FLOAT)*(input_ptr + offset);
    } else if (remain_channel == 3) {
        int offset      = buffer_offset;
        output_values.x = (FLOAT)*(input_ptr + offset);
        offset += height_width_size;
        output_values.y = (FLOAT)*(input_ptr + offset);
        offset += height_width_size;
        output_values.z = (FLOAT)*(input_ptr + offset);
    } else if (remain_channel == 2) {
        int offset      = buffer_offset;
        output_values.x = (FLOAT)*(input_ptr + offset);
        offset += height_width_size;
        output_values.y = (FLOAT)*(input_ptr + offset);
    } else if (remain_channel == 1) {
        int offset      = buffer_offset;
        output_values.x = (FLOAT)*(input_ptr + offset);
    }

    const int out_offset = (((batch_idx * ((channels+3)/4) + channel_4_idx/4) * height + height_idx) * width + width_idx)*4;
    vstore4(output_values, 0, output+out_offset);
}

// convert data from image(b h, ic/4 w ic4) to buffer(nhwc)
__kernel void nc4hw4_buffer_to_nhwc_buffer(GLOBAL_SIZE_2_DIMS
                                    #ifdef BUFFER_FORMAT_OUT_TRANS
                                    __global float *output,
                                    #else
                                    __global FLOAT *output,
                                    #endif
                                    __private const int height, __private const int width,
                                    __private const int channels,
                                    __global FLOAT *input_ptr) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx     = image_height_idx / height;
    const int height_idx    = image_height_idx % height;
    const int width_idx     = image_width_idx % width;
    const int channel_4_idx = (image_width_idx / width) << 2;
    const int buffer_offset = ((batch_idx * height + height_idx) * width + width_idx) * channels + channel_4_idx;

    const int in_offset = (((batch_idx * ((channels+3)/4) + channel_4_idx/4) * height + height_idx) * width + width_idx)*4;
    
    #ifdef BUFFER_FORMAT_OUT_TRANS
    float4 values        = convert_float4(vload4(0, input_ptr+in_offset));
    #else
    FLOAT4 values        = vload4(0, input_ptr+in_offset);
    #endif
    const int remain_channel = channels - channel_4_idx;
    if (remain_channel >= 4) {
        vstore4(values, 0, output + buffer_offset);
    } else if (remain_channel == 3) {
        int offset     = buffer_offset;
        output[offset] = values.x;
        offset++;
        output[offset] = values.y;
        offset++;
        output[offset] = values.z;
    } else if (remain_channel == 2) {
        int offset     = buffer_offset;
        output[offset] = values.x;
        offset++;
        output[offset] = values.y;
    } else if (remain_channel == 1) {
        int offset     = buffer_offset;
        output[offset] = values.x;
    }
}

// convert data from buffer(nc4hw4) to buffer(nchw)
__kernel void nc4hw4_buffer_to_nchw_buffer(GLOBAL_SIZE_2_DIMS
                                    #ifdef BUFFER_FORMAT_OUT_TRANS
                                    __global float *output,
                                    #else
                                    __global FLOAT *output,
                                    #endif
                                    __private const int height, __private const int width,
                                    __private const int channels,
                                    __global FLOAT *input_ptr) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);
    
    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx  = image_height_idx / height;
    const int height_idx = image_height_idx % height;
    const int width_idx  = image_width_idx % width;
    int channel_4_idx    = (image_width_idx / width) * 4;
    int buffer_offset    = ((batch_idx * channels + channel_4_idx) * height + height_idx) * width + width_idx;
    
    const int in_offset = (((batch_idx * ((channels+3)/4) + channel_4_idx/4) * height + height_idx) * width + width_idx)*4;
    #ifdef BUFFER_FORMAT_OUT_TRANS
    float4 values    = convert_float4(vload4(0, input_ptr+in_offset));
    #else
    FLOAT4 values    = vload4(0, input_ptr+in_offset);
    #endif

    const int height_width_size = height * width;

    const int remain_channel = channels - channel_4_idx;

    if (remain_channel >= 4) {
        int offset     = buffer_offset;
        output[offset] = values.x;
        offset += height_width_size;
        output[offset] = values.y;
        offset += height_width_size;
        output[offset] = values.z;
        offset += height_width_size;
        output[offset] = values.w;
    } else if (remain_channel == 3) {
        int offset     = buffer_offset;
        output[offset] = values.x;
        offset += height_width_size;
        output[offset] = values.y;
        offset += height_width_size;
        output[offset] = values.z;
    } else if (remain_channel == 2) {
        int offset     = buffer_offset;
        output[offset] = values.x;
        offset += height_width_size;
        output[offset] = values.y;
    } else if (remain_channel == 1) {
        int offset     = buffer_offset;
        output[offset] = values.x;
    }
}

__kernel void nc4hw4_buffer_to_nc4hw4_buffer(GLOBAL_SIZE_2_DIMS
                                    #ifdef BUFFER_FORMAT_INP_TRANS
                                    __global const float *input_ptr,
                                    #else
                                    __global const FLOAT *input_ptr,
                                    #endif
                                    __private const int2 output_shape,
                                    __private const int channel_4,
                                    #ifdef BUFFER_FORMAT_OUT_TRANS
                                    __global float *output
                                    #else
                                    __global FLOAT *output
                                    #endif
) {

    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx         = image_height_idx / output_shape.x;
    const int height_idx        = image_height_idx % output_shape.x;
    const int width_idx         = image_width_idx % output_shape.y;
    const int channel_block_idx = image_width_idx / output_shape.y;
    int buffer_offset =
        (((batch_idx * channel_4 + channel_block_idx) * output_shape.x + height_idx) * output_shape.y + width_idx) * 4;
    #ifdef BUFFER_FORMAT_INP_TRANS
    FLOAT4 values = CONVERT_FLOAT4(vload4(0, input_ptr + buffer_offset));
    #else
    FLOAT4 values = vload4(0, input_ptr + buffer_offset);
    #endif
    
    #ifdef BUFFER_FORMAT_OUT_TRANS
    vstore4(convert_float4(values), 0, output+buffer_offset);
    #else
    vstore4(values, 0, output+buffer_offset);
    #endif
}

// convert kernel : from buffer(oihw) to image(oc/4 h w , ic oc4)
__kernel void conv2d_filter_buffer_to_nc4hw4_buffer(GLOBAL_SIZE_2_DIMS
                                            #ifdef BUFFER_FORMAT_INP_TRANS
                                            __global const float *input_ptr,
                                            #else
                                            __global const FLOAT *input_ptr,
                                            #endif
                                            __private const int output_channel,
                                            __private const int2 kernel_shape,
                                            __private const int ic_h_w_size,
                                            __private const int height_width_size,
                                            __global FLOAT *output) {
    int image_width_idx  = get_global_id(0); // ic
    int image_height_idx = get_global_id(1); // oc/4 h w

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int input_channel_4_idx  = image_width_idx;
    const int output_channel_4_idx = (image_height_idx / height_width_size) * 4;
    const int height_width_idx     = image_height_idx % height_width_size;
    const int buffer_height_idx    = height_width_idx / kernel_shape.y;
    const int buffer_width_idx     = height_width_idx % kernel_shape.y;

    const int buffer_offset = output_channel_4_idx * ic_h_w_size + input_channel_4_idx * height_width_size +
                              buffer_height_idx * kernel_shape.y + buffer_width_idx;

    FLOAT4 output_values = 0;
    if (output_channel_4_idx < output_channel) {
        const int remain_channel = output_channel - output_channel_4_idx;
        if (remain_channel >= 4) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (FLOAT)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.z = (FLOAT)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.w = (FLOAT)(*(input_ptr + offset));
        } else if (remain_channel == 3) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (FLOAT)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.z = (FLOAT)(*(input_ptr + offset));

        } else if (remain_channel == 2) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (FLOAT)(*(input_ptr + offset));
        } else if (remain_channel == 1) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
        }
    }
    const int out_offset = (image_width_idx*height_width_size*((output_channel+3)/4)+image_height_idx)*4;
    vstore4(output_values, 0, output+out_offset);
}

// convert kernel from buffer(mihw) to image(ic/4, ic4 h w m)
// but now dw only support m == 1
__kernel void dw_filter_buffer_to_nc4hw4_buffer(GLOBAL_SIZE_2_DIMS
                                        #ifdef BUFFER_FORMAT_INP_TRANS
                                        __global const float *input_ptr,
                                        #else
                                        __global const FLOAT *input_ptr,
                                        #endif
                                        __private const int4 kernel_shape,//[1, Cout, fh, fw]
                                        __private const int height_width_size,
                                        __global FLOAT *output) {
    const int image_width_idx  = get_global_id(0);//fh*fw
    const int image_height_idx = get_global_id(1);//UP_DIV(Cout, 4)

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    FLOAT4 output_values = 0;
    if (kernel_shape.x == 1) {
        const int input_channel_4_idx = image_height_idx * 4;
        const int buffer_height_idx   = image_width_idx / kernel_shape.w;
        const int buffer_width_idx    = image_width_idx % kernel_shape.w;

        const int buffer_offset =
            mad24(mad24(input_channel_4_idx, kernel_shape.z, buffer_height_idx), kernel_shape.w, buffer_width_idx);

        //input [1, Cout,                fh,                 fw]
        //index:[0, input_channel_4_idx, buffer_height_idx,  buffer_width_idx]
        const int remain_channel = kernel_shape.y - input_channel_4_idx;
        if (input_channel_4_idx < kernel_shape.y) {
            if (remain_channel >= 4) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.y = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.z = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.w = (FLOAT)(*(input_ptr + offset));
            } else if (remain_channel == 3) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.y = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.z = (FLOAT)(*(input_ptr + offset));

            } else if (remain_channel == 2) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.y = (FLOAT)(*(input_ptr + offset));
            } else if (remain_channel == 1) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
            }
        }
    }

    //output NC4HW4 [1, fw*fh,            1, Cout/4]x oc4
    //index:        [0, image_width_idx,  0, image_height_idx]
    const int out_offset = (image_width_idx*((kernel_shape.y+3)/4)+image_height_idx)*4;
    vstore4(output_values, 0, output+out_offset);
}
