#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define EXP exp
#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }


__kernel void softmax_in1_buf(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *input,
                              __global FLOAT *output,
                              __private const int inside,
                              __private const int outside,
                              __private const int dim) {

    const int x = get_global_id(0);
    const int y = get_global_id(1); // inside = 1
    const int z = get_global_id(2); // outside
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    const int offset = z * dim + y;
    const int dim4 = (dim + 3) / 4;
    const int loop_end = max(0, dim4 - 1);
#if SOFTMAX_LOCAL_SIZE >= 4
    int lid = get_local_id(0);
    COMPUTE_FLOAT local sum_mnn[SOFTMAX_LOCAL_SIZE];
    COMPUTE_FLOAT local max_mnn[SOFTMAX_LOCAL_SIZE];

    // compute maxvalue
    COMPUTE_FLOAT4 maxValue = (COMPUTE_FLOAT4)-FLT_MAX;
    for (int i = lid; i < loop_end; i+=SOFTMAX_LOCAL_SIZE) {
        maxValue = fmax(maxValue, CONVERT_COMPUTE_FLOAT4(vload4(i, input+offset)));
    }

    max_mnn[lid] = fmax(fmax(fmax(maxValue.x, maxValue.y), maxValue.z), maxValue.w);
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            max_mnn[lid] = fmax(max_mnn[lid], max_mnn[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maxValue.x = max_mnn[0];
    for(int i = loop_end << 2; i < dim; ++i){
        maxValue.x = fmax(maxValue.x, (COMPUTE_FLOAT)(input[offset+i]));
    }

    // compute sumvalue
    COMPUTE_FLOAT4 sumValue = (COMPUTE_FLOAT4)0;
    for (int i = lid; i < loop_end; i+=SOFTMAX_LOCAL_SIZE) {
        sumValue += exp(CONVERT_COMPUTE_FLOAT4(vload4(i, input+offset)) - (COMPUTE_FLOAT4)maxValue.x);
    }
    sum_mnn[lid] = sumValue.x + sumValue.y + sumValue.z + sumValue.w;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum_mnn[lid] = sum_mnn[lid] + sum_mnn[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sumValue.x = sum_mnn[0];
    for(int i = loop_end << 2; i < dim; ++i){
        sumValue.x += exp((COMPUTE_FLOAT)(input[offset+i]) - maxValue.x);
    }
    
    // store result
    for(int i = lid; i < loop_end; i+=SOFTMAX_LOCAL_SIZE){
        vstore4(CONVERT_FLOAT4(exp(CONVERT_COMPUTE_FLOAT4(vload4(i, input+offset)) - (COMPUTE_FLOAT4)maxValue.x) / (COMPUTE_FLOAT4)sumValue.x), 0, output + offset + i * 4);
    }
    for(int i = loop_end << 2; i < dim; ++i){
        output[offset + i] = (FLOAT)exp((COMPUTE_FLOAT)(input[offset + i]) - maxValue.x) / sumValue.x;
    }
#else
    // compute maxvalue
    COMPUTE_FLOAT4 maxValue = (COMPUTE_FLOAT4)-FLT_MAX;
    for (int i = 0; i < loop_end; i++) {
        maxValue = fmax(maxValue, CONVERT_COMPUTE_FLOAT4(vload4(i, input+offset)));
    }
    maxValue.x = fmax(fmax(fmax(maxValue.x, maxValue.y), maxValue.z), maxValue.w);
    for(int i = loop_end << 2; i < dim; ++i){
        maxValue.x = fmax(maxValue.x, (COMPUTE_FLOAT)(input[offset+i]));
    }
    
    // compute sumvalue
    COMPUTE_FLOAT4 sumValue = (COMPUTE_FLOAT4)0;
    for (int i = 0; i < loop_end; i++) {
        sumValue += exp(CONVERT_COMPUTE_FLOAT4(vload4(i, input+offset)) - (COMPUTE_FLOAT4)maxValue.x);
    }
    sumValue.x = sumValue.x + sumValue.y + sumValue.z + sumValue.w;
    for(int i = loop_end << 2; i < dim; ++i){
        sumValue.x += exp((COMPUTE_FLOAT)(input[offset+i]) - maxValue.x);
    }
    
    // store result
    for(int i = 0; i < loop_end; i++){
        vstore4(CONVERT_FLOAT4(exp(CONVERT_COMPUTE_FLOAT4(vload4(i, input+offset)) - (COMPUTE_FLOAT4)maxValue.x) / (COMPUTE_FLOAT4)sumValue.x), 0, output + offset + i * 4);
    }
    for(int i = loop_end << 2; i < dim; ++i){
        output[offset + i] = (FLOAT)exp((COMPUTE_FLOAT)(input[offset + i]) - maxValue.x) / sumValue.x;
    }
#endif
}

__kernel void softmax_buf(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *input,
                              __global FLOAT *output,
                              __private const int inside,
                              __private const int outside,
                              __private const int dim) {

    const int x = get_global_id(0);
    const int y = get_global_id(1); // inside
    const int z = get_global_id(2); // outside
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    const int offset = z * dim * inside + y;
#if SOFTMAX_LOCAL_SIZE >= 4
    int lid = get_local_id(0);
    COMPUTE_FLOAT local sum_mnn[SOFTMAX_LOCAL_SIZE];
    COMPUTE_FLOAT local max_mnn[SOFTMAX_LOCAL_SIZE];

    COMPUTE_FLOAT maxValue = (COMPUTE_FLOAT)-FLT_MAX;
    for (int i = lid; i < dim; i+=SOFTMAX_LOCAL_SIZE) {
        maxValue = fmax(maxValue, (COMPUTE_FLOAT)(input[offset+i*inside]));
    }

    max_mnn[lid] = maxValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            max_mnn[lid] = fmax(max_mnn[lid], max_mnn[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maxValue = max_mnn[0];

    COMPUTE_FLOAT sumValue = (COMPUTE_FLOAT)0;
    for (int i = lid; i < dim; i+=SOFTMAX_LOCAL_SIZE) {
        sumValue += exp((COMPUTE_FLOAT)(input[offset+i*inside]) - maxValue);
    }
    sum_mnn[lid] = sumValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum_mnn[lid] = sum_mnn[lid] + sum_mnn[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sumValue = sum_mnn[0];
    for(int i = lid; i < dim; i+=SOFTMAX_LOCAL_SIZE){
        output[offset + i * inside] = (FLOAT)exp((COMPUTE_FLOAT)(input[offset + i * inside]) - maxValue) / sumValue;
    }
#else
    COMPUTE_FLOAT maxValue = (COMPUTE_FLOAT)-FLT_MAX;
    for (int i = 0; i < dim; i++) {
        maxValue = fmax(maxValue, (COMPUTE_FLOAT)(input[offset+i*inside]));
    }

    COMPUTE_FLOAT sumValue = (COMPUTE_FLOAT)0;
    for (int i = 0; i < dim; i++) {
        sumValue += exp((COMPUTE_FLOAT)(input[offset+i*inside]) - maxValue);
    }
    for(int i = 0; i < dim; i++){
        output[offset + i * inside] = (FLOAT)exp((COMPUTE_FLOAT)(input[offset+i*inside]) - maxValue) / sumValue;
    }
#endif
}

__kernel void softmax_v4_buf(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *input,
                              __global FLOAT *output,
                              __private const int inside,
                              __private const int outside,
                              __private const int dim) {

    const int x = get_global_id(0);
    const int y = get_global_id(1); // inside
    const int z = get_global_id(2); // outside
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    const int offset = z * dim * inside + (y << 2);
#if SOFTMAX_LOCAL_SIZE >= 4
    int lid = get_local_id(0);
    COMPUTE_FLOAT4 local sum_mnn[SOFTMAX_LOCAL_SIZE];
    COMPUTE_FLOAT4 local max_mnn[SOFTMAX_LOCAL_SIZE];

    COMPUTE_FLOAT4 maxValue = (COMPUTE_FLOAT4)-FLT_MAX;
    for (int i = lid; i < dim; i+=SOFTMAX_LOCAL_SIZE) {
        maxValue = fmax(maxValue, CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset+i*inside)));
    }

    max_mnn[lid] = maxValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            max_mnn[lid] = fmax(max_mnn[lid], max_mnn[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maxValue = max_mnn[0];

    COMPUTE_FLOAT4 sumValue = (COMPUTE_FLOAT4)0;
    for (int i = lid; i < dim; i+=SOFTMAX_LOCAL_SIZE) {
        sumValue += exp(CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset+i*inside)) - maxValue);
    }
    sum_mnn[lid] = sumValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum_mnn[lid] = sum_mnn[lid] + sum_mnn[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sumValue = sum_mnn[0];
    for(int i = lid; i < dim; i+=SOFTMAX_LOCAL_SIZE){
        vstore4(CONVERT_FLOAT4(exp(CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset+i*inside)) - maxValue) / sumValue), 0, output+offset+i*inside);
    }
#else
    COMPUTE_FLOAT4 maxValue = (COMPUTE_FLOAT4)-FLT_MAX;
    for (int i = 0; i < dim; i++) {
        maxValue = fmax(maxValue, CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset+i*inside)));
    }

    COMPUTE_FLOAT4 sumValue = (COMPUTE_FLOAT4)0;
    for (int i = 0; i < dim; i++) {
        sumValue += exp(CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset+i*inside)) - maxValue);
    }
    for(int i = 0; i < dim; i++){
        vstore4(CONVERT_FLOAT4(exp(CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset+i*inside)) - maxValue) / sumValue), 0, output+offset+i*inside);
    }
	#endif
}

__kernel void softmax_v4_sparse_buf(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *input,
                              __global FLOAT *output,
                              __global const int *sparse_query,
                              __private const int inside,
                              __private const int outside,
                              __private const int dim,
                              __private const int query_seq_len,
                              __private const int q_start,
                              __private const int q_piece_len) {

    const int x = get_global_id(0);
    const int y = get_global_id(1); // inside / 4
    const int z = get_global_id(2); // outside
    DEAL_NON_UNIFORM_DIM3(x, y, z);

    const int y4 = y << 2;
    const int q0 = q_start + y4;
    const int q1 = q0 + 1;
    const int q2 = q0 + 2;
    const int q3 = q0 + 3;
    const int ql0 = (y4 < q_piece_len && q0 < query_seq_len) ? sparse_query[q0] : -1;
    const int ql1 = (y4 + 1 < q_piece_len && q1 < query_seq_len) ? sparse_query[q1] : -1;
    const int ql2 = (y4 + 2 < q_piece_len && q2 < query_seq_len) ? sparse_query[q2] : -1;
    const int ql3 = (y4 + 3 < q_piece_len && q3 < query_seq_len) ? sparse_query[q3] : -1;
    const int max_ql = max(max(ql0, ql1), max(ql2, ql3));
    const int active_dim = min(max(max_ql + 1, 0), dim);
    if (active_dim <= 0) {
        return;
    }

    const int offset = z * dim * inside + y4;
#if SOFTMAX_LOCAL_SIZE >= 4
    int lid = get_local_id(0);
    COMPUTE_FLOAT4 local sum_mnn[SOFTMAX_LOCAL_SIZE];
    COMPUTE_FLOAT4 local max_mnn[SOFTMAX_LOCAL_SIZE];

    COMPUTE_FLOAT4 maxValue = (COMPUTE_FLOAT4)-FLT_MAX;
    for (int i = lid; i < active_dim; i+=SOFTMAX_LOCAL_SIZE) {
        maxValue = fmax(maxValue, CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset+i*inside)));
    }

    max_mnn[lid] = maxValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            max_mnn[lid] = fmax(max_mnn[lid], max_mnn[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maxValue = max_mnn[0];

    COMPUTE_FLOAT4 sumValue = (COMPUTE_FLOAT4)0;
    for (int i = lid; i < active_dim; i+=SOFTMAX_LOCAL_SIZE) {
        sumValue += exp(CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset+i*inside)) - maxValue);
    }
    sum_mnn[lid] = sumValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum_mnn[lid] = sum_mnn[lid] + sum_mnn[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sumValue = sum_mnn[0];
    for(int i = lid; i < active_dim; i+=SOFTMAX_LOCAL_SIZE){
        vstore4(CONVERT_FLOAT4(exp(CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset+i*inside)) - maxValue) / sumValue), 0, output+offset+i*inside);
    }
#else
    COMPUTE_FLOAT4 maxValue = (COMPUTE_FLOAT4)-FLT_MAX;
    for (int i = 0; i < active_dim; i++) {
        maxValue = fmax(maxValue, CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset+i*inside)));
    }

    COMPUTE_FLOAT4 sumValue = (COMPUTE_FLOAT4)0;
    for (int i = 0; i < active_dim; i++) {
        sumValue += exp(CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset+i*inside)) - maxValue);
    }
    for(int i = 0; i < active_dim; i++){
        vstore4(CONVERT_FLOAT4(exp(CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset+i*inside)) - maxValue) / sumValue), 0, output+offset+i*inside);
    }
#endif
}
