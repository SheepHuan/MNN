// gridsample.cu - GRID_SAMPLE_NEAREST / BILINEAR (3.6.0) + 3D + 2.7.2 kernels
//   + shims
//   source/backend/cuda/execution/GridSampleExecution.cu
#include "corpus_common.cuh"

namespace MNN {
namespace Corpus {

// ============================================================================
// GridSample: source/backend/cuda/execution/GridSampleExecution.cu
// Inline device helpers (getPosition/sample/CLAMP) are re-implemented here
// because they live in an anonymous namespace in the original .cu file.
// BorderMode is passed as int (0=ZEROS,1=BORDER,2=REFLECTION,3=CUBE).
// ============================================================================
inline __device__ float gsGetPosition(float x, int range, bool alignCorners) {
    float a = alignCorners ? 1.0f : 0.0f;
    float b = alignCorners ? 0.0f : 1.0f;
    return ((1.0f + x) * (range - a) - b) / 2.0f;
}
inline __device__ int gsClamp(int value, int minV, int maxV) {
    return min(max(value, minV), maxV);
}
inline __device__ int gsSample(int pos, int total, int paddingMode) {
    if (pos < 0 || pos >= total) {
        if (paddingMode == 0) return -1; // ZEROS
        pos = gsClamp(pos, 0, total - 1);
    }
    return pos;
}

template <typename T>
__global__ void GRID_SAMPLE_NEAREST(const int count, const T* input, const T* grid, T* output,
                                     const int input_height, const int input_width,
                                     const int output_height, const int output_width,
                                     const int channel, const int channel_pack,
                                     int paddingMode, bool alignCorners) {
    CUDA_KERNEL_LOOP(index, count) {
        int idx_cp = index % channel;
        int idx_nhw = index / channel;
        int idx_ow = idx_nhw % output_width;
        int idx_nh = idx_nhw / output_width;
        int idx_oh = idx_nh % output_height;
        int idx_ob = idx_nh / output_height;
        float pos_x = grid[idx_nhw * 2 + 0];
        float pos_y = grid[idx_nhw * 2 + 1];
        float in_grid_x = gsGetPosition(pos_x, input_width, alignCorners);
        float in_grid_y = gsGetPosition(pos_y, input_height, alignCorners);
        int in_pos_x = (int)floor(in_grid_x + 0.5f);
        int in_pos_y = (int)floor(in_grid_y + 0.5f);
        in_pos_x = gsSample(in_pos_x, input_width, paddingMode);
        in_pos_y = gsSample(in_pos_y, input_height, paddingMode);
        int dst_offset = ((idx_ob * output_height + idx_oh) * output_width + idx_ow) * channel_pack + idx_cp;
        if (in_pos_x == -1 || in_pos_y == -1) {
            output[dst_offset] = (T)0.0;
            continue;
        }
        output[dst_offset] = input[((idx_ob * input_height + in_pos_y) * input_width + in_pos_x) * channel_pack + idx_cp];
    }
}

// ============================================================================
// GridSample bilinear: source/backend/cuda/execution/GridSampleExecution.cu
// ============================================================================
template <typename T>
__global__ void GRID_SAMPLE_BILINEAR(const int count, const T* input, const T* grid, T* output,
                                     const int input_height, const int input_width,
                                     const int output_height, const int output_width,
                                     const int channel, const int channel_pack,
                                     int paddingMode, bool alignCorners) {
    CUDA_KERNEL_LOOP(index, count) {
        int idx_cp = index % channel;
        int idx_nhw = index / channel;
        int idx_ow = idx_nhw % output_width;
        int idx_nh = idx_nhw / output_width;
        int idx_oh = idx_nh % output_height;
        int idx_ob = idx_nh / output_height;
        float pos_x = grid[idx_nhw * 2 + 0];
        float pos_y = grid[idx_nhw * 2 + 1];
        float in_grid_x = gsGetPosition(pos_x, input_width, alignCorners);
        float in_grid_y = gsGetPosition(pos_y, input_height, alignCorners);
        int in_pos_x0 = (int)floor(in_grid_x);
        int in_pos_y0 = (int)floor(in_grid_y);
        int in_pos_x1 = (int)ceil(in_grid_x);
        int in_pos_y1 = (int)ceil(in_grid_y);
        float x_weight = in_pos_x1 - in_grid_x;
        float y_weight = in_pos_y1 - in_grid_y;
        in_pos_x0 = gsSample(in_pos_x0, input_width, paddingMode);
        in_pos_y0 = gsSample(in_pos_y0, input_height, paddingMode);
        in_pos_x1 = gsSample(in_pos_x1, input_width, paddingMode);
        in_pos_y1 = gsSample(in_pos_y1, input_height, paddingMode);
        float in00 = (in_pos_y0 == -1 || in_pos_x0 == -1) ? 0.0f : (float)input[((idx_ob * input_height + in_pos_y0) * input_width + in_pos_x0) * channel_pack + idx_cp];
        float in01 = (in_pos_y0 == -1 || in_pos_x1 == -1) ? 0.0f : (float)input[((idx_ob * input_height + in_pos_y0) * input_width + in_pos_x1) * channel_pack + idx_cp];
        float in10 = (in_pos_y1 == -1 || in_pos_x0 == -1) ? 0.0f : (float)input[((idx_ob * input_height + in_pos_y1) * input_width + in_pos_x0) * channel_pack + idx_cp];
        float in11 = (in_pos_y1 == -1 || in_pos_x1 == -1) ? 0.0f : (float)input[((idx_ob * input_height + in_pos_y1) * input_width + in_pos_x1) * channel_pack + idx_cp];
        int dst_offset = ((idx_ob * output_height + idx_oh) * output_width + idx_ow) * channel_pack + idx_cp;
        output[dst_offset] = (T)(in00 * x_weight * y_weight + in01 * (1.0f - x_weight) * y_weight
                                  + in10 * x_weight * (1.0f - y_weight) + in11 * (1.0f - x_weight) * (1.0f - y_weight));
    }
}

// ============================================================================
// GridSample 3D: NEAREST + BILINEAR
// ============================================================================
template <typename T>
__global__ void GRID_SAMPLE_NEAREST_3D(const int count, const T* input, const T* grid, T* output,
                                        int id, int ih, int iw, int od, int oh, int ow,
                                        int channel, int channel_pack, int paddingMode, bool alignCorners) {
    CUDA_KERNEL_LOOP(index, count) {
        int idx_cp = index % channel;
        int idx_nhw = index / channel;
        int idx_ow = idx_nhw % ow;
        int idx_nh = idx_nhw / ow;
        int idx_oh = idx_nh % oh;
        int idx_obd = idx_nh / oh;
        int idx_od = idx_obd % od;
        int idx_ob = idx_obd / od;
        float pos_x = grid[idx_nhw * 3 + 0];
        float pos_y = grid[idx_nhw * 3 + 1];
        float pos_z = grid[idx_nhw * 3 + 2];
        float igx = gsGetPosition(pos_x, iw, alignCorners);
        float igy = gsGetPosition(pos_y, ih, alignCorners);
        float igz = gsGetPosition(pos_z, id, alignCorners);
        int ipx = (int)floor(igx + 0.5f);
        int ipy = (int)floor(igy + 0.5f);
        int ipz = (int)floor(igz + 0.5f);
        ipx = gsSample(ipx, iw, paddingMode);
        ipy = gsSample(ipy, ih, paddingMode);
        ipz = gsSample(ipz, id, paddingMode);
        int dst = (((idx_ob * od + idx_od) * oh + idx_oh) * ow + idx_ow) * channel_pack + idx_cp;
        if (ipx == -1 || ipy == -1 || ipz == -1) { output[dst] = (T)0; continue; }
        output[dst] = input[(((idx_ob * id + ipz) * ih + ipy) * iw + ipx) * channel_pack + idx_cp];
    }
}
template <typename T>
__global__ void GRID_SAMPLE_BILINEAR_3D(const int count, const T* input, const T* grid, T* output,
                                        int id, int ih, int iw, int od, int oh, int ow,
                                        int channel, int channel_pack, int paddingMode, bool alignCorners) {
    CUDA_KERNEL_LOOP(index, count) {
        int idx_cp = index % channel;
        int idx_nhw = index / channel;
        int idx_ow = idx_nhw % ow;
        int idx_nh = idx_nhw / ow;
        int idx_oh = idx_nh % oh;
        int idx_obd = idx_nh / oh;
        int idx_od = idx_obd % od;
        int idx_ob = idx_obd / od;
        float pos_x = grid[idx_nhw * 3 + 0];
        float pos_y = grid[idx_nhw * 3 + 1];
        float pos_z = grid[idx_nhw * 3 + 2];
        float igx = gsGetPosition(pos_x, iw, alignCorners);
        float igy = gsGetPosition(pos_y, ih, alignCorners);
        float igz = gsGetPosition(pos_z, id, alignCorners);
        int ix0 = (int)floor(igx), ix1 = (int)ceil(igx);
        int iy0 = (int)floor(igy), iy1 = (int)ceil(igy);
        int iz0 = (int)floor(igz), iz1 = (int)ceil(igz);
        float xw = ix1 - igx, yw = iy1 - igy, zw = iz1 - igz;
        ix0 = gsSample(ix0, iw, paddingMode); iy0 = gsSample(iy0, ih, paddingMode); iz0 = gsSample(iz0, id, paddingMode);
        ix1 = gsSample(ix1, iw, paddingMode); iy1 = gsSample(iy1, ih, paddingMode); iz1 = gsSample(iz1, id, paddingMode);
        float v000 = (iz0==-1||iy0==-1||ix0==-1)?0:input[(((idx_ob*id+iz0)*ih+iy0)*iw+ix0)*channel_pack+idx_cp];
        float v001 = (iz0==-1||iy0==-1||ix1==-1)?0:input[(((idx_ob*id+iz0)*ih+iy0)*iw+ix1)*channel_pack+idx_cp];
        float v010 = (iz0==-1||iy1==-1||ix0==-1)?0:input[(((idx_ob*id+iz0)*ih+iy1)*iw+ix0)*channel_pack+idx_cp];
        float v011 = (iz0==-1||iy1==-1||ix1==-1)?0:input[(((idx_ob*id+iz0)*ih+iy1)*iw+ix1)*channel_pack+idx_cp];
        float v100 = (iz1==-1||iy0==-1||ix0==-1)?0:input[(((idx_ob*id+iz1)*ih+iy0)*iw+ix0)*channel_pack+idx_cp];
        float v101 = (iz1==-1||iy0==-1||ix1==-1)?0:input[(((idx_ob*id+iz1)*ih+iy0)*iw+ix1)*channel_pack+idx_cp];
        float v110 = (iz1==-1||iy1==-1||ix0==-1)?0:input[(((idx_ob*id+iz1)*ih+iy1)*iw+ix0)*channel_pack+idx_cp];
        float v111 = (iz1==-1||iy1==-1||ix1==-1)?0:input[(((idx_ob*id+iz1)*ih+iy1)*iw+ix1)*channel_pack+idx_cp];
        int dst = (((idx_ob*od+idx_od)*oh+idx_oh)*ow+idx_ow)*channel_pack+idx_cp;
        output[dst] = (T)(v000*xw*yw*zw + v001*(1-xw)*yw*zw + v010*xw*(1-yw)*zw + v011*(1-xw)*(1-yw)*zw
                           + v100*xw*yw*(1-zw) + v101*(1-xw)*yw*(1-zw) + v110*xw*(1-yw)*(1-zw) + v111*(1-xw)*(1-yw)*(1-zw));
    }
}

// ---- GRID_SAMPLE_NEAREST 2.7.2: idx_cp = index % channel_pack, output[index] ----
template <typename T>
__global__ void GRID_SAMPLE_NEAREST_272(const int count, const T* input, const T* grid, T* output,
                                         int ih, int iw, int oh, int ow, int ch, int ch_p,
                                         int padMode, bool align) {
    CUDA_KERNEL_LOOP(index, count) {
        int idx_cp = index % ch_p;
        int idx_nhw = index / ch_p;
        int idx_ow = idx_nhw % ow;
        int idx_nh = idx_nhw / ow;
        int idx_oh = idx_nh % oh;
        int idx_ob = idx_nh / oh;
        if (idx_cp >= ch) { output[index] = (T)0.0; continue; }
        float pos_x = grid[idx_nhw * 2 + 0];
        float pos_y = grid[idx_nhw * 2 + 1];
        // align corners
        float igx = align ? (pos_x * (iw - 1) + 1) * 0.5f : (pos_x + 1) * 0.5f * iw;
        float igy = align ? (pos_y * (ih - 1) + 1) * 0.5f : (pos_y + 1) * 0.5f * ih;
        int ix = floor(igx + 0.5f), iy = floor(igy + 0.5f);
        // border
        if (padMode == 0) { ix = max(0, min(ix, iw-1)); iy = max(0, min(iy, ih-1)); }
        else { if (ix < 0 || ix >= iw || iy < 0 || iy >= ih) { output[index] = (T)0.0; continue; } }
        output[index] = input[((idx_ob * ih + iy) * iw + ix) * ch_p + idx_cp];
    }
}

// ---- GRID_SAMPLE_BILINEAR 2.7.2 ----
template <typename T>
__global__ void GRID_SAMPLE_BILINEAR_272(const int count, const T* input, const T* grid, T* output,
                                          int ih, int iw, int oh, int ow, int ch, int ch_p,
                                          int padMode, bool align) {
    CUDA_KERNEL_LOOP(index, count) {
        int idx_cp = index % ch_p;
        int idx_nhw = index / ch_p;
        int idx_ow = idx_nhw % ow;
        int idx_nh = idx_nhw / ow;
        int idx_oh = idx_nh % oh;
        int idx_ob = idx_nh / oh;
        if (idx_cp >= ch) { output[index] = (T)0.0; continue; }
        float pos_x = grid[idx_nhw * 2 + 0];
        float pos_y = grid[idx_nhw * 2 + 1];
        float igx = align ? (pos_x * (iw - 1) + 1) * 0.5f : (pos_x + 1) * 0.5f * iw;
        float igy = align ? (pos_y * (ih - 1) + 1) * 0.5f : (pos_y + 1) * 0.5f * ih;
        int ix0 = floor(igx), iy0 = floor(igy);
        int ix1 = ceil(igx), iy1 = ceil(igy);
        float xw = ix1 - igx, yw = iy1 - igy;
        auto samp = [&](int v, int lim)->int { return padMode==0 ? max(0,min(v,lim-1)) : ((v<0||v>=lim)?-1:v); };
        ix0 = samp(ix0, iw); iy0 = samp(iy0, ih);
        ix1 = samp(ix1, iw); iy1 = samp(iy1, ih);
        auto get = [&](int iy, int ix)->float { return (iy==-1||ix==-1)?0.0f:(float)input[((idx_ob*ih+iy)*iw+ix)*ch_p+idx_cp]; };
        output[index] = (T)(get(iy0,ix0)*xw*yw + get(iy0,ix1)*(1.0f-xw)*yw + get(iy1,ix0)*xw*(1.0f-yw) + get(iy1,ix1)*(1.0f-xw)*(1.0f-yw));
    }
}

} // namespace Corpus
} // namespace MNN

extern "C" {

// ---- GridSample nearest ----
void mnn_corpus_grid_sample_nearest_fp32(const int count, const float* input, const float* grid, float* output,
                                          int ih, int iw, int oh, int ow, int ch, int ch_p, int padMode, int align,
                                          int grid_, int block, cudaStream_t stream) {
    MNN::Corpus::GRID_SAMPLE_NEAREST<float><<<grid_, block, 0, stream>>>(
        count, input, grid, output, ih, iw, oh, ow, ch, ch_p, padMode, align != 0);
}

// ---- GridSample bilinear ----
void mnn_corpus_grid_sample_bilinear_fp32(const int count, const float* input, const float* grid, float* output,
                                           int ih, int iw, int oh, int ow, int ch, int ch_p, int padMode, int align,
                                           int grid_, int block, cudaStream_t stream) {
    MNN::Corpus::GRID_SAMPLE_BILINEAR<float><<<grid_, block, 0, stream>>>(
        count, input, grid, output, ih, iw, oh, ow, ch, ch_p, padMode, align != 0);
}

// ---- GridSample 3D ----
void mnn_corpus_grid_sample_nearest_3d_fp32(const int count, const float* input, const float* grid, float* output,
                                            int id, int ih, int iw, int od, int oh, int ow,
                                            int ch, int ch_p, int padMode, int align,
                                            int grid_, int block, cudaStream_t stream) {
    MNN::Corpus::GRID_SAMPLE_NEAREST_3D<float><<<grid_, block, 0, stream>>>(
        count, input, grid, output, id, ih, iw, od, oh, ow, ch, ch_p, padMode, align != 0);
}
void mnn_corpus_grid_sample_bilinear_3d_fp32(const int count, const float* input, const float* grid, float* output,
                                              int id, int ih, int iw, int od, int oh, int ow,
                                              int ch, int ch_p, int padMode, int align,
                                              int grid_, int block, cudaStream_t stream) {
    MNN::Corpus::GRID_SAMPLE_BILINEAR_3D<float><<<grid_, block, 0, stream>>>(
        count, input, grid, output, id, ih, iw, od, oh, ow, ch, ch_p, padMode, align != 0);
}

// ---- GRID_SAMPLE_NEAREST 2.7.2: output[index], idx_cp = index % channel_pack ----
void mnn_corpus_grid_sample_nearest_272_fp32(const int count, const float* input, const float* grid, float* output,
                                               int ih, int iw, int oh, int ow, int ch, int ch_p,
                                               int padMode, int align, int grid_, int block, cudaStream_t stream) {
    MNN::Corpus::GRID_SAMPLE_NEAREST_272<float><<<grid_, block, 0, stream>>>(
        count, input, grid, output, ih, iw, oh, ow, ch, ch_p, padMode, align != 0);
}

// ---- GRID_SAMPLE_BILINEAR 2.7.2 ----
void mnn_corpus_grid_sample_bilinear_272_fp32(const int count, const float* input, const float* grid, float* output,
                                                int ih, int iw, int oh, int ow, int ch, int ch_p,
                                                int padMode, int align, int grid_, int block, cudaStream_t stream) {
    MNN::Corpus::GRID_SAMPLE_BILINEAR_272<float><<<grid_, block, 0, stream>>>(
        count, input, grid, output, ih, iw, oh, ow, ch, ch_p, padMode, align != 0);
}

} // extern "C"
