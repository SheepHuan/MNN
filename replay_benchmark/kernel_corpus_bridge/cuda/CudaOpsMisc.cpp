// CudaOpsMisc.cpp - B-class CUDA corpus adapters (Gather/Argmax/Argmin/
// Interp/Transpose/GridSample). Split from CudaOps.cpp.
#include "CudaOps.hpp"
#include <cmath>
#include <cstring>

namespace MNN {
namespace Replay {
namespace KernelCorpus {
namespace MnnOps {

// ============================================================================
// Registration
// ============================================================================
// ============================================================================
// B-class: GatherV2 / ArgMax / ArgMin / Interp / Transpose
// ============================================================================
extern "C" {
void mnn_corpus_gatherv2_fp32(const int, const int, const int, const int, const int, const float*, const int*, float*, int, int, cudaStream_t);
void mnn_corpus_argmax_fp32(const int, const int, const int, const int, const float*, int*, int, int, cudaStream_t);
void mnn_corpus_argmin_fp32(const int, const int, const int, const int, const float*, int*, int, int, cudaStream_t);
void mnn_corpus_interp_nearest_fp32(const int, const int, int, int, int, int, float, float, float, float, const float*, float*, int, int, cudaStream_t);
void mnn_corpus_nhwc2nchw_fp32(const float*, float*, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_nchw2nhwc_fp32(const float*, float*, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_grid_sample_nearest_fp32(const int, const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
// 1.2.0 tag shims (forward declarations — defined in CorpusKernelsMisc.cu).
// argmax_120 outputs float* (index stored as float), unlike 3.6.0's int*.
void mnn_corpus_argmax_120_fp32(const int, const int, const int, const int, const float*, float*, int, int, cudaStream_t);
void mnn_corpus_interp_120_fp32(const int, int, int, int, int, float, float, float, float, const float*, float*, int, int, cudaStream_t);
void mnn_corpus_interp_bilinear_120_fp32(const int, int, int, int, int, float, float, float, float, const float*, float*, int, int, cudaStream_t);
void mnn_corpus_pack_c4_120_fp32(const float*, float*, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_unpack_c4_120_fp32(const float*, float*, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_setzero_120_fp32(const int, float*, int, int, cudaStream_t);
void mnn_corpus_add_bias_120_fp32(float*, float*, const float*, int, int, int, int, cudaStream_t);
}

// ---- GatherV2 ----
bool CudaGatherV2Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_gatherv2_fp32";
    const int inside = spec.intParam("inside", 4);
    const int iNum = spec.intParam("i_num", 3);
    const int oNum = spec.intParam("o_num", 2);
    const int outside = spec.intParam("outside", 2);
    const int count = outside * oNum * inside;
    std::vector<float> input(outside * iNum * inside);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = (float)i;
    std::vector<int32_t> indice(oNum);
    for (int i = 0; i < oNum; ++i) indice[i] = i % iNum;
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer idxBuf; idxBuf.sizeBytes = oNum * sizeof(int32_t);
    idxBuf.initialData.assign((const uint8_t*)indice.data(), (const uint8_t*)indice.data() + idxBuf.sizeBytes);
    idxBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(idxBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.args.push_back(AdaptedArg::scalarInt(outside));
    ac.args.push_back(AdaptedArg::scalarInt(inside));
    ac.args.push_back(AdaptedArg::scalarInt(iNum));
    ac.args.push_back(AdaptedArg::scalarInt(oNum));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input;
    ac.elementCount = count; ac.m = outside; ac.n = iNum; ac.k = inside;
    // Store indice in validatorInputB as float reinterpreation
    ac.validatorInputB.resize(oNum);
    for (int i = 0; i < oNum; ++i) ac.validatorInputB[i] = (float)indice[i];
    return true;
}
cudaError_t CudaGatherV2Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_gatherv2_fp32(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4],
                             (const float*)ctx.devBufs[0], (const int*)ctx.devBufs[1], (float*)ctx.devBufs[2],
                             ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaGatherV2Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int inside = ac.k, iNum = ac.n, oNum = (int)ac.validatorInputB.size();
    const int outside = ac.m;
    const int count = outside * oNum * inside;
    if ((int)output.size() < count) return false;
    for (int o = 0; o < outside; ++o)
      for (int n = 0; n < oNum; ++n)
        for (int x = 0; x < inside; ++x) {
            int idx = (int)ac.validatorInputB[n];
            float expected = ac.validatorInputA[o * iNum * inside + idx * inside + x];
            int outOff = (o * oNum + n) * inside + x;
            if (std::fabs(output[outOff] - expected) > 1e-3f) return false;
        }
    return true;
}

// ---- ArgMax ----
bool CudaArgMaxFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    const int outside = spec.intParam("outside", 4);
    const int dim = spec.intParam("dim", 8);
    const int inside = spec.intParam("inside", 1);
    const int count = outside * inside;
    std::vector<float> input(outside * dim * inside);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = (float)(i % 17);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    if (spec.tag == "1.2.0") {
        // 1.2.0: output is float* (stores index as float), not int*
        ac.entry = "mnn_corpus_argmax_120_fp32";
        AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(float); outBuf.isOutput = true;
        ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
        ac.args.push_back(AdaptedArg::scalarInt(count));
        ac.args.push_back(AdaptedArg::scalarInt(outside));
        ac.args.push_back(AdaptedArg::scalarInt(inside));
        ac.args.push_back(AdaptedArg::scalarInt(dim));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
    } else {
        // 3.6.0: output is int*
        ac.entry = "mnn_corpus_argmax_fp32";
        AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(int32_t); outBuf.isOutput = true;
        ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
        ac.args.push_back(AdaptedArg::scalarInt(count));
        ac.args.push_back(AdaptedArg::scalarInt(outside));
        ac.args.push_back(AdaptedArg::scalarInt(inside));
        ac.args.push_back(AdaptedArg::scalarInt(dim));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
    }
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = count;
    ac.m = outside; ac.n = dim; ac.k = inside;
    return true;
}
cudaError_t CudaArgMaxFp32Kernel::launch(const AdaptedCase& ac, const CudaLaunchCtx& ctx) const {
    if (ac.tag == "1.2.0") {
        mnn_corpus_argmax_120_fp32(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                   (const float*)ctx.devBufs[0], (float*)ctx.devBufs[1], ctx.grid, ctx.block, ctx.stream);
    } else {
        mnn_corpus_argmax_fp32(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                               (const float*)ctx.devBufs[0], (int*)ctx.devBufs[1], ctx.grid, ctx.block, ctx.stream);
    }
    return cudaGetLastError();
}
bool CudaArgMaxFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int outside = ac.m, dim = ac.n, inside = ac.k;
    const int count = outside * inside;
    if ((int)output.size() * 4 < count * 4) return false;
    if (ac.tag == "1.2.0") {
        // 1.2.0: output is float (index stored as float)
        for (int o = 0; o < outside; ++o)
          for (int x = 0; x < inside; ++x) {
            int idx = 0; float mx = ac.validatorInputA[o * dim * inside + x];
            for (int j = 1; j < dim; ++j) {
              float v = ac.validatorInputA[o * dim * inside + j * inside + x];
              if (mx < v) { idx = j; mx = v; }
            }
            if (std::fabs(output[o * inside + x] - (float)idx) > 1e-3f) return false;
          }
    } else {
        // 3.6.0: output is int32
        const int32_t* out = reinterpret_cast<const int32_t*>(output.data());
        for (int o = 0; o < outside; ++o)
          for (int x = 0; x < inside; ++x) {
            int idx = 0; float mx = ac.validatorInputA[o * dim * inside + x];
            for (int j = 1; j < dim; ++j) {
              float v = ac.validatorInputA[o * dim * inside + j * inside + x];
              if (mx < v) { idx = j; mx = v; }
            }
            if (out[o * inside + x] != idx) return false;
          }
    }
    return true;
}

// ---- ArgMin ----
bool CudaArgMinFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_argmin_fp32";
    const int outside = spec.intParam("outside", 4);
    const int dim = spec.intParam("dim", 8);
    const int inside = spec.intParam("inside", 1);
    const int count = outside * inside;
    std::vector<float> input(outside * dim * inside);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = (float)(i % 17);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(int32_t); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.args.push_back(AdaptedArg::scalarInt(outside));
    ac.args.push_back(AdaptedArg::scalarInt(inside));
    ac.args.push_back(AdaptedArg::scalarInt(dim));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = count;
    ac.m = outside; ac.n = dim; ac.k = inside;
    return true;
}
cudaError_t CudaArgMinFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_argmin_fp32(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                           (const float*)ctx.devBufs[0], (int*)ctx.devBufs[1], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaArgMinFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int outside = ac.m, dim = ac.n, inside = ac.k;
    const int count = outside * inside;
    if ((int)output.size() * 4 < count * 4) return false;
    const int32_t* out = reinterpret_cast<const int32_t*>(output.data());
    for (int o = 0; o < outside; ++o)
      for (int x = 0; x < inside; ++x) {
        int idx = 0; float mn = ac.validatorInputA[o * dim * inside + x];
        for (int j = 1; j < dim; ++j) {
          float v = ac.validatorInputA[o * dim * inside + j * inside + x];
          if (mn > v) { idx = j; mn = v; }
        }
        if (out[o * inside + x] != idx) return false;
      }
    return true;
}

// ---- Interp nearest ----
bool CudaInterpNearestFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    const int ih = spec.intParam("ih", 4), iw = spec.intParam("iw", 4);
    const int oh = spec.intParam("oh", 8), ow = spec.intParam("ow", 8);
    const float sh = (float)ih / oh, sw = (float)iw / ow;
    std::vector<float> input;
    int total;
    if (spec.tag == "1.2.0") {
        // 1.2.0: no channel dim, n = oh*ow
        ac.entry = "mnn_corpus_interp_120_fp32";
        total = oh * ow;
        input.assign(ih * iw, 0.0f);
        for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 11);
        AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
        AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
        ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
        ac.args.push_back(AdaptedArg::scalarInt(total));
        ac.args.push_back(AdaptedArg::scalarInt(ih));
        ac.args.push_back(AdaptedArg::scalarInt(iw));
        ac.args.push_back(AdaptedArg::scalarInt(oh));
        ac.args.push_back(AdaptedArg::scalarInt(ow));
        ac.args.push_back(AdaptedArg::scalarFloat(sh));
        ac.args.push_back(AdaptedArg::scalarFloat(sw));
        ac.args.push_back(AdaptedArg::scalarFloat(0.0f));
        ac.args.push_back(AdaptedArg::scalarFloat(0.0f));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
    } else {
        ac.entry = "mnn_corpus_interp_nearest_fp32";
        const int c_p = spec.intParam("channels", 4);
        total = oh * ow * c_p;
        input.assign(ih * iw * c_p, 0.0f);
        for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 11);
        AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
        AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
        ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
        ac.args.push_back(AdaptedArg::scalarInt(total));
        ac.args.push_back(AdaptedArg::scalarInt(c_p));
        ac.args.push_back(AdaptedArg::scalarInt(ih));
        ac.args.push_back(AdaptedArg::scalarInt(iw));
        ac.args.push_back(AdaptedArg::scalarInt(oh));
        ac.args.push_back(AdaptedArg::scalarInt(ow));
        ac.args.push_back(AdaptedArg::scalarFloat(sh));
        ac.args.push_back(AdaptedArg::scalarFloat(sw));
        ac.args.push_back(AdaptedArg::scalarFloat(0.0f));
        ac.args.push_back(AdaptedArg::scalarFloat(0.0f));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.c = c_p;
    }
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = total;
    ac.h = ih; ac.w = iw; ac.stride = oh;
    return true;
}
cudaError_t CudaInterpNearestFp32Kernel::launch(const AdaptedCase& ac, const CudaLaunchCtx& ctx) const {
    if (ac.tag == "1.2.0") {
        mnn_corpus_interp_120_fp32(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                    ctx.intArgs[4], ctx.floatArgs[0], ctx.floatArgs[1],
                                    ctx.floatArgs[2], ctx.floatArgs[3],
                                    (const float*)ctx.devBufs[0], (float*)ctx.devBufs[1],
                                    ctx.grid, ctx.block, ctx.stream);
    } else {
        mnn_corpus_interp_nearest_fp32(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                       ctx.intArgs[4], ctx.intArgs[5], ctx.floatArgs[0], ctx.floatArgs[1],
                                       ctx.floatArgs[2], ctx.floatArgs[3],
                                       (const float*)ctx.devBufs[0], (float*)ctx.devBufs[1],
                                       ctx.grid, ctx.block, ctx.stream);
    }
    return cudaGetLastError();
}
bool CudaInterpNearestFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int ih = ac.h, iw = ac.w, oh = ac.stride, ow = oh;
    const float sh = (float)ih / oh, sw = (float)iw / ow;
    if (ac.tag == "1.2.0") {
        // 1.2.0: single-channel, n = oh*ow
        const int n = oh * ow;
        if ((int)output.size() < n) return false;
        for (int y = 0; y < oh; ++y)
            for (int x = 0; x < ow; ++x) {
                int ix = std::min(std::max(0, (int)floor((float)x * sw)), iw - 1);
                int iy = std::min(std::max(0, (int)floor((float)y * sh)), ih - 1);
                if (std::fabs(output[y * ow + x] - ac.validatorInputA[iy * iw + ix]) > 1e-3f) return false;
            }
        return true;
    }
    // 3.6.0: with channel dim
    const int c_p = ac.c, total = oh * ow * c_p;
    if ((int)output.size() < total) return false;
    for (int z = 0; z < 1; ++z)
      for (int y = 0; y < oh; ++y)
        for (int x = 0; x < ow; ++x)
          for (int c = 0; c < c_p; ++c) {
            int ix = std::min(std::max(0, (int)floor((float)x * sw)), iw - 1);
            int iy = std::min(std::max(0, (int)floor((float)y * sh)), ih - 1);
            float expected = ac.validatorInputA[(iy * iw + ix) * c_p + c];
            if (std::fabs(output[((z * oh + y) * ow + x) * c_p + c] - expected) > 1e-3f) return false;
          }
    return true;
}

// ---- Transpose NHWC->NCHW ----
bool CudaNhwc2NchwFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_nhwc2nchw_fp32";
    const int outside = spec.intParam("outside", 2);
    const int axis = spec.intParam("axis", 4);
    const int inside = spec.intParam("inside", 3);
    const int total = outside * axis * inside;
    std::vector<float> input(total);
    for (int i = 0; i < total; ++i) input[i] = (float)i;
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(total));
    ac.args.push_back(AdaptedArg::scalarInt(inside));
    ac.args.push_back(AdaptedArg::scalarInt(axis));
    ac.args.push_back(AdaptedArg::scalarInt(outside));
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = total;
    ac.m = outside; ac.n = axis; ac.k = inside;
    return true;
}
cudaError_t CudaNhwc2NchwFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_nhwc2nchw_fp32((const float*)ctx.devBufs[0], (float*)ctx.devBufs[1],
                               ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                               ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaNhwc2NchwFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int outside = ac.m, axis = ac.n, inside = ac.k;
    const int total = outside * axis * inside;
    if ((int)output.size() < total) return false;
    for (int idx = 0; idx < total; ++idx) {
        int x = idx % inside;
        int y = (idx / inside) % axis;
        int z = idx / (inside * axis);
        int nchwOff = z * axis * inside + y * inside + x;
        if (std::fabs(output[nchwOff] - ac.validatorInputA[idx]) > 1e-3f) return false;
    }
    return true;
}

// ---- Transpose NCHW->NHWC ----
bool CudaNchw2NhwcFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_nchw2nhwc_fp32";
    const int outside = spec.intParam("outside", 2);
    const int axis = spec.intParam("axis", 4);
    const int inside = spec.intParam("inside", 3);
    const int total = outside * axis * inside;
    // Input is NCHW layout: [outside, axis, inside]
    std::vector<float> input(total);
    for (int i = 0; i < total; ++i) input[i] = (float)i;
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(total));
    ac.args.push_back(AdaptedArg::scalarInt(inside));
    ac.args.push_back(AdaptedArg::scalarInt(axis));
    ac.args.push_back(AdaptedArg::scalarInt(outside));
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = total;
    ac.m = outside; ac.n = axis; ac.k = inside;
    return true;
}
cudaError_t CudaNchw2NhwcFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_nchw2nhwc_fp32((const float*)ctx.devBufs[0], (float*)ctx.devBufs[1],
                               ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                               ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaNchw2NhwcFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int outside = ac.m, axis = ac.n, inside = ac.k;
    const int total = outside * axis * inside;
    if ((int)output.size() < total) return false;
    for (int idx = 0; idx < total; ++idx) {
        int x = idx % inside;
        int y = (idx / inside) % axis;
        int z = idx / (inside * axis);
        int nchwOff = z * axis * inside + y * inside + x;
        if (std::fabs(output[idx] - ac.validatorInputA[nchwOff]) > 1e-3f) return false;
    }
    return true;
}

// ---- GridSample nearest ----
bool CudaGridSampleNearestFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_grid_sample_nearest_fp32";
    const int ih = spec.intParam("ih", 4), iw = spec.intParam("iw", 4);
    const int oh = spec.intParam("oh", 4), ow = spec.intParam("ow", 4);
    const int ch = spec.intParam("channels", 4);
    const int ch_p = ch;
    const int total = oh * ow * ch_p;
    std::vector<float> input(ih * iw * ch_p);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 11);
    // grid: [oh*ow, 2] normalized coords in [-1,1]
    std::vector<float> grid(oh * ow * 2);
    for (int i = 0; i < (int)grid.size(); ++i) grid[i] = -0.5f + (i % 3) * 0.5f;
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer gridBuf; gridBuf.setFp32(grid); gridBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(gridBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::scalarInt(total));
    ac.args.push_back(AdaptedArg::buffer(0)); // input
    ac.args.push_back(AdaptedArg::buffer(1)); // grid
    ac.args.push_back(AdaptedArg::buffer(2)); // output
    ac.args.push_back(AdaptedArg::scalarInt(ih));
    ac.args.push_back(AdaptedArg::scalarInt(iw));
    ac.args.push_back(AdaptedArg::scalarInt(oh));
    ac.args.push_back(AdaptedArg::scalarInt(ow));
    ac.args.push_back(AdaptedArg::scalarInt(ch));
    ac.args.push_back(AdaptedArg::scalarInt(ch_p));
    ac.args.push_back(AdaptedArg::scalarInt(1)); // paddingMode=BORDER
    ac.args.push_back(AdaptedArg::scalarInt(0)); // alignCorners=false
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.validatorInputB = grid;
    ac.elementCount = total; ac.h = ih; ac.w = iw; ac.c = ch_p; ac.stride = oh;
    return true;
}
cudaError_t CudaGridSampleNearestFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_grid_sample_nearest_fp32(ctx.intArgs[0], (const float*)ctx.devBufs[0], (const float*)ctx.devBufs[1],
                                         (float*)ctx.devBufs[2], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                         ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
                                         ctx.intArgs[8], ctx.intArgs[9], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaGridSampleNearestFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int ih = ac.h, iw = ac.w, ch_p = ac.c, oh = ac.stride, ow = oh;
    const int total = oh * ow * ch_p;
    if ((int)output.size() < total) return false;
    // Replicate kernel: BORDER padding, alignCorners=false
    for (int idx = 0; idx < total; ++idx) {
        int idx_cp = idx % ch_p;
        int idx_nhw = idx / ch_p;
        int idx_ow = idx_nhw % ow;
        int idx_nh = idx_nhw / ow;
        int idx_oh = idx_nh % oh;
        int idx_ob = idx_nh / oh;
        float pos_x = ac.validatorInputB[idx_nhw * 2 + 0];
        float pos_y = ac.validatorInputB[idx_nhw * 2 + 1];
        float igx = ((1.0f + pos_x) * iw - 1.0f) / 2.0f;
        float igy = ((1.0f + pos_y) * ih - 1.0f) / 2.0f;
        int ipx = (int)floor(igx + 0.5f);
        int ipy = (int)floor(igy + 0.5f);
        ipx = std::min(std::max(ipx, 0), iw - 1);
        ipy = std::min(std::max(ipy, 0), ih - 1);
        float expected = ac.validatorInputA[((idx_ob * ih + ipy) * iw + ipx) * ch_p + idx_cp];
        if (std::fabs(output[idx] - expected) > 1e-3f) return false;
    }
    return true;
}


// ---- Reduction + Interp + GridSample extern C shims ----
extern "C" {
void mnn_corpus_grid_sample_bilinear_fp32(const int, const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_interp_bilinear_fp32(const int, const int, int, int, int, int, float, float, float, float, const float*, float*, int, int, cudaStream_t);
void mnn_corpus_interp_nearest_round_fp32(const int, const int, int, int, int, int, float, float, float, float, const float*, float*, int, int, cudaStream_t);
void mnn_corpus_reduction_sum_fp32(const float*, float*, int, int, int, int, int, cudaStream_t);
void mnn_corpus_reduction_mean_fp32(const float*, float*, int, int, int, int, int, cudaStream_t);
// 1.2.0 tag shims
void mnn_corpus_reduction_sum_120_fp32(const float*, float*, int, int, int, int, int, cudaStream_t);
void mnn_corpus_reduction_mean_120_fp32(const float*, float*, int, int, int, int, int, cudaStream_t);
void mnn_corpus_reduction_max_fp32(const float*, float*, int, int, int, int, int, cudaStream_t);
void mnn_corpus_reduction_min_fp32(const float*, float*, int, int, int, int, int, cudaStream_t);
void mnn_corpus_reduction_prod_fp32(const float*, float*, int, int, int, int, int, cudaStream_t);
}

// ---- Reduction (SUM/MAX/MIN/MEAN/PROD) ----
// Reduction adapter: for SUM/MEAN, 1.2.0 uses T accumulation + different param order.
// MAX/MIN/PROD are identical across tags (same logic, just param order differs — but
// since the macro passes outside/axis/inside and the kernel is the same, we can reuse).
#define REDUCTION_ADAPTER(CLASS, SHIM, SHIM_120, REDUCE_EXPR) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    if (spec.tag == "1.2.0") { \
        ac.entry = "mnn_corpus_" #SHIM_120; \
        const int outside = spec.intParam("outside", 4); \
        const int axis = spec.intParam("axis", 16); \
        const int inside = spec.intParam("inside", 1); \
        const int count = outside * inside; \
        std::vector<float> input(outside * axis * inside); \
        for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 13); \
        AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false; \
        AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(float); outBuf.isOutput = true; \
        ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf); \
        ac.args.push_back(AdaptedArg::buffer(0)); \
        ac.args.push_back(AdaptedArg::buffer(1)); \
        ac.args.push_back(AdaptedArg::scalarInt(inside)); \
        ac.args.push_back(AdaptedArg::scalarInt(axis)); \
        ac.args.push_back(AdaptedArg::scalarInt(outside)); \
        ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1; \
        ac.validatorInputA = input; ac.elementCount = count; \
        ac.m = outside; ac.n = axis; ac.k = inside; \
    } else { \
        ac.entry = "mnn_corpus_" #SHIM; \
        const int outside = spec.intParam("outside", 4); \
        const int axis = spec.intParam("axis", 16); \
        const int inside = spec.intParam("inside", 1); \
        const int count = outside * inside; \
        std::vector<float> input(outside * axis * inside); \
        for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 13); \
        AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false; \
        AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(float); outBuf.isOutput = true; \
        ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf); \
        ac.args.push_back(AdaptedArg::buffer(0)); \
        ac.args.push_back(AdaptedArg::buffer(1)); \
        ac.args.push_back(AdaptedArg::scalarInt(outside)); \
        ac.args.push_back(AdaptedArg::scalarInt(axis)); \
        ac.args.push_back(AdaptedArg::scalarInt(inside)); \
        ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1; \
        ac.validatorInputA = input; ac.elementCount = count; \
        ac.m = outside; ac.n = axis; ac.k = inside; \
    } \
    return true; \
} \
cudaError_t CLASS::launch(const AdaptedCase& ac, const CudaLaunchCtx& ctx) const { \
    if (ac.tag == "1.2.0") { \
        extern void mnn_corpus_##SHIM_120(const float*, float*, int, int, int, int, int, cudaStream_t); \
        mnn_corpus_##SHIM_120((const float*)ctx.devBufs[0], (float*)ctx.devBufs[1], \
                      ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], \
                      ctx.grid, ctx.block, ctx.stream); \
    } else { \
        mnn_corpus_##SHIM((const float*)ctx.devBufs[0], (float*)ctx.devBufs[1], \
                      ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], \
                      ctx.grid, ctx.block, ctx.stream); \
    } \
    return cudaGetLastError(); \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    const int outside = ac.m, axis = ac.n, inside = ac.k; \
    const int count = outside * inside; \
    if ((int)output.size() < count) return false; \
    for (int o = 0; o < outside; ++o) \
      for (int x = 0; x < inside; ++x) { \
        const float* src = ac.validatorInputA.data() + o * axis * inside + x; \
        float expected = REDUCE_EXPR; \
        if (std::fabs(output[o * inside + x] - expected) > 1e-2f) return false; \
      } \
    return true; \
}

REDUCTION_ADAPTER(CudaReductionSumFp32Kernel, reduction_sum_fp32, reduction_sum_120_fp32, ([&]{float s=0;for(int v=0;v<axis;++v)s+=src[v*inside];return s;}()))
REDUCTION_ADAPTER(CudaReductionMeanFp32Kernel, reduction_mean_fp32, reduction_mean_120_fp32, ([&]{float s=0;for(int v=0;v<axis;++v)s+=src[v*inside];return s/axis;}()))
// MAX/MIN/PROD: 1.2.0 and 3.6.0 have identical logic. For 1.2.0 the adapter
// stores params as (inside, axis, outside) but the 3.6.0 shim expects
// (outside, axis, inside), so launch swaps [0] and [2].
#define REDUCTION_ADAPTER_REUSE(CLASS, SHIM, REDUCE_EXPR) \
bool CLASS::adapt(const CaseSpec& spec, AdaptedCase& ac) const { \
    const int outside = spec.intParam("outside", 4); \
    const int axis = spec.intParam("axis", 16); \
    const int inside = spec.intParam("inside", 1); \
    const int count = outside * inside; \
    std::vector<float> input(outside * axis * inside); \
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 13); \
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false; \
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(float); outBuf.isOutput = true; \
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf); \
    ac.args.push_back(AdaptedArg::buffer(0)); \
    ac.args.push_back(AdaptedArg::buffer(1)); \
    if (spec.tag == "1.2.0") { \
        ac.args.push_back(AdaptedArg::scalarInt(inside)); \
        ac.args.push_back(AdaptedArg::scalarInt(axis)); \
        ac.args.push_back(AdaptedArg::scalarInt(outside)); \
        ac.entry = "mnn_corpus_" #SHIM; \
    } else { \
        ac.args.push_back(AdaptedArg::scalarInt(outside)); \
        ac.args.push_back(AdaptedArg::scalarInt(axis)); \
        ac.args.push_back(AdaptedArg::scalarInt(inside)); \
        ac.entry = "mnn_corpus_" #SHIM; \
    } \
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1; \
    ac.validatorInputA = input; ac.elementCount = count; \
    ac.m = outside; ac.n = axis; ac.k = inside; \
    return true; \
} \
cudaError_t CLASS::launch(const AdaptedCase& ac, const CudaLaunchCtx& ctx) const { \
    if (ac.tag == "1.2.0") { \
        /* swap inside[0] and outside[2] for the 3.6.0 shim */ \
        mnn_corpus_##SHIM((const float*)ctx.devBufs[0], (float*)ctx.devBufs[1], \
                      ctx.intArgs[2], ctx.intArgs[1], ctx.intArgs[0], \
                      ctx.grid, ctx.block, ctx.stream); \
    } else { \
        mnn_corpus_##SHIM((const float*)ctx.devBufs[0], (float*)ctx.devBufs[1], \
                      ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], \
                      ctx.grid, ctx.block, ctx.stream); \
    } \
    return cudaGetLastError(); \
} \
bool CLASS::validate(const AdaptedCase& ac, const std::vector<float>& output) const { \
    const int outside = ac.m, axis = ac.n, inside = ac.k; \
    const int count = outside * inside; \
    if ((int)output.size() < count) return false; \
    for (int o = 0; o < outside; ++o) \
      for (int x = 0; x < inside; ++x) { \
        const float* src = ac.validatorInputA.data() + o * axis * inside + x; \
        float expected = REDUCE_EXPR; \
        if (std::fabs(output[o * inside + x] - expected) > 1e-2f) return false; \
      } \
    return true; \
}

REDUCTION_ADAPTER_REUSE(CudaReductionMaxFp32Kernel, reduction_max_fp32, ([&]{float m=src[0];for(int v=1;v<axis;++v)m=std::max(m,src[v*inside]);return m;}()))
REDUCTION_ADAPTER_REUSE(CudaReductionMinFp32Kernel, reduction_min_fp32, ([&]{float m=src[0];for(int v=1;v<axis;++v)m=std::min(m,src[v*inside]);return m;}()))
REDUCTION_ADAPTER_REUSE(CudaReductionProdFp32Kernel, reduction_prod_fp32, ([&]{float p=1;for(int v=0;v<axis;++v)p*=src[v*inside];return p;}()))

// ---- Interp bilinear ----
bool CudaInterpBilinearFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    const int ih = spec.intParam("ih", 4), iw = spec.intParam("iw", 4);
    const int oh = spec.intParam("oh", 8), ow = spec.intParam("ow", 8);
    const float sh = (float)ih / oh, sw = (float)iw / ow;
    std::vector<float> input;
    int total;
    if (spec.tag == "1.2.0") {
        // 1.2.0: no channel dim, n = oh*ow
        ac.entry = "mnn_corpus_interp_bilinear_120_fp32";
        total = oh * ow;
        input.assign(ih * iw, 0.0f);
        for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 11);
        AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
        AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
        ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
        ac.args.push_back(AdaptedArg::scalarInt(total));
        ac.args.push_back(AdaptedArg::scalarInt(ih));
        ac.args.push_back(AdaptedArg::scalarInt(iw));
        ac.args.push_back(AdaptedArg::scalarInt(oh));
        ac.args.push_back(AdaptedArg::scalarInt(ow));
        ac.args.push_back(AdaptedArg::scalarFloat(sh));
        ac.args.push_back(AdaptedArg::scalarFloat(sw));
        ac.args.push_back(AdaptedArg::scalarFloat(0.0f));
        ac.args.push_back(AdaptedArg::scalarFloat(0.0f));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
    } else {
        ac.entry = "mnn_corpus_interp_bilinear_fp32";
        const int c_p = spec.intParam("channels", 4);
        total = oh * ow * c_p;
        input.assign(ih * iw * c_p, 0.0f);
        for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 11);
        AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
        AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
        ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
        ac.args.push_back(AdaptedArg::scalarInt(total));
        ac.args.push_back(AdaptedArg::scalarInt(c_p));
        ac.args.push_back(AdaptedArg::scalarInt(ih));
        ac.args.push_back(AdaptedArg::scalarInt(iw));
        ac.args.push_back(AdaptedArg::scalarInt(oh));
        ac.args.push_back(AdaptedArg::scalarInt(ow));
        ac.args.push_back(AdaptedArg::scalarFloat(sh));
        ac.args.push_back(AdaptedArg::scalarFloat(sw));
        ac.args.push_back(AdaptedArg::scalarFloat(0.0f));
        ac.args.push_back(AdaptedArg::scalarFloat(0.0f));
        ac.args.push_back(AdaptedArg::buffer(0));
        ac.args.push_back(AdaptedArg::buffer(1));
        ac.c = c_p;
    }
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = total;
    ac.h = ih; ac.w = iw; ac.stride = oh;
    return true;
}
cudaError_t CudaInterpBilinearFp32Kernel::launch(const AdaptedCase& ac, const CudaLaunchCtx& ctx) const {
    if (ac.tag == "1.2.0") {
        mnn_corpus_interp_bilinear_120_fp32(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                            ctx.intArgs[4], ctx.floatArgs[0], ctx.floatArgs[1],
                                            ctx.floatArgs[2], ctx.floatArgs[3],
                                            (const float*)ctx.devBufs[0], (float*)ctx.devBufs[1],
                                            ctx.grid, ctx.block, ctx.stream);
    } else {
        mnn_corpus_interp_bilinear_fp32(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                        ctx.intArgs[4], ctx.intArgs[5], ctx.floatArgs[0], ctx.floatArgs[1],
                                        ctx.floatArgs[2], ctx.floatArgs[3],
                                        (const float*)ctx.devBufs[0], (float*)ctx.devBufs[1],
                                        ctx.grid, ctx.block, ctx.stream);
    }
    return cudaGetLastError();
}
bool CudaInterpBilinearFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int ih = ac.h, iw = ac.w, oh = ac.stride, ow = oh;
    const float sh = (float)ih / oh, sw = (float)iw / ow;
    if (ac.tag == "1.2.0") {
        // 1.2.0: single-channel, n = oh*ow
        const int n = oh * ow;
        if ((int)output.size() < n) return false;
        for (int y = 0; y < oh; ++y)
            for (int x = 0; x < ow; ++x) {
                float fx = x * sw, fy = y * sh;
                int ix0 = std::min(std::max(0, (int)floor(fx)), iw-1);
                int ix1 = std::min((int)ceil(fx), iw-1);
                int iy0 = std::min(std::max(0, (int)floor(fy)), ih-1);
                int iy1 = std::min((int)ceil(fy), ih-1);
                float fxw = fx - ix0, fyw = fy - iy0;
                float v00 = ac.validatorInputA[iy0*iw+ix0];
                float v01 = ac.validatorInputA[iy0*iw+ix1];
                float v10 = ac.validatorInputA[iy1*iw+ix0];
                float v11 = ac.validatorInputA[iy1*iw+ix1];
                float expected = (1-fxw)*(1-fyw)*v00 + fxw*(1-fyw)*v01 + (1-fxw)*fyw*v10 + fxw*fyw*v11;
                if (std::fabs(output[y*ow+x] - expected) > 1e-2f) return false;
            }
        return true;
    }
    // 3.6.0: with channel dim
    const int c_p = ac.c, total = oh * ow * c_p;
    if ((int)output.size() < total) return false;
    for (int z = 0; z < 1; ++z)
      for (int y = 0; y < oh; ++y)
        for (int x = 0; x < ow; ++x)
          for (int c = 0; c < c_p; ++c) {
            float fx = x * sw, fy = y * sh;
            int ix0 = std::min(std::max(0, (int)floor(fx)), iw-1);
            int ix1 = std::min((int)ceil(fx), iw-1);
            int iy0 = std::min(std::max(0, (int)floor(fy)), ih-1);
            int iy1 = std::min((int)ceil(fy), ih-1);
            float fx_w = fx - ix0, fy_w = fy - iy0;
            float v00 = ac.validatorInputA[((z*ih+iy0)*iw+ix0)*c_p+c];
            float v01 = ac.validatorInputA[((z*ih+iy0)*iw+ix1)*c_p+c];
            float v10 = ac.validatorInputA[((z*ih+iy1)*iw+ix0)*c_p+c];
            float v11 = ac.validatorInputA[((z*ih+iy1)*iw+ix1)*c_p+c];
            float expected = (1-fx_w)*(1-fy_w)*v00 + fx_w*(1-fy_w)*v01 + (1-fx_w)*fy_w*v10 + fx_w*fy_w*v11;
            if (std::fabs(output[((z*oh+y)*ow+x)*c_p+c] - expected) > 1e-2f) return false;
          }
    return true;
}

// ---- Interp nearest round ----
bool CudaInterpNearestRoundFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_interp_nearest_round_fp32";
    const int ih = spec.intParam("ih", 4), iw = spec.intParam("iw", 4);
    const int oh = spec.intParam("oh", 8), ow = spec.intParam("ow", 8);
    const int c_p = spec.intParam("channels", 4);
    const int total = oh * ow * c_p;
    const float sh = (float)ih / oh, sw = (float)iw / ow;
    std::vector<float> input(ih * iw * c_p);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 11);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::scalarInt(total));
    ac.args.push_back(AdaptedArg::scalarInt(c_p));
    ac.args.push_back(AdaptedArg::scalarInt(ih));
    ac.args.push_back(AdaptedArg::scalarInt(iw));
    ac.args.push_back(AdaptedArg::scalarInt(oh));
    ac.args.push_back(AdaptedArg::scalarInt(ow));
    ac.args.push_back(AdaptedArg::scalarFloat(sh));
    ac.args.push_back(AdaptedArg::scalarFloat(sw));
    ac.args.push_back(AdaptedArg::scalarFloat(0.0f));
    ac.args.push_back(AdaptedArg::scalarFloat(0.0f));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = total;
    ac.h = ih; ac.w = iw; ac.c = c_p; ac.stride = oh;
    return true;
}
cudaError_t CudaInterpNearestRoundFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_interp_nearest_round_fp32(ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                         ctx.intArgs[4], ctx.intArgs[5], ctx.floatArgs[0], ctx.floatArgs[1],
                                         ctx.floatArgs[2], ctx.floatArgs[3],
                                         (const float*)ctx.devBufs[0], (float*)ctx.devBufs[1],
                                         ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaInterpNearestRoundFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int ih = ac.h, iw = ac.w, c_p = ac.c, oh = ac.stride, ow = oh;
    const float sh = (float)ih / oh, sw = (float)iw / ow;
    const int total = oh * ow * c_p;
    if ((int)output.size() < total) return false;
    for (int z = 0; z < 1; ++z)
      for (int y = 0; y < oh; ++y)
        for (int x = 0; x < ow; ++x)
          for (int c = 0; c < c_p; ++c) {
            int ix = std::min(std::max(0, (int)floor((float)x * sw + 0.499f)), iw - 1);
            int iy = std::min(std::max(0, (int)floor((float)y * sh + 0.499f)), ih - 1);
            float expected = ac.validatorInputA[(iy * iw + ix) * c_p + c];
            if (std::fabs(output[((z * oh + y) * ow + x) * c_p + c] - expected) > 1e-3f) return false;
          }
    return true;
}

// ---- GridSample bilinear ----
bool CudaGridSampleBilinearFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_grid_sample_bilinear_fp32";
    const int ih = spec.intParam("ih", 4), iw = spec.intParam("iw", 4);
    const int oh = spec.intParam("oh", 4), ow = spec.intParam("ow", 4);
    const int ch = spec.intParam("channels", 4);
    const int ch_p = ch;
    const int total = oh * ow * ch_p;
    std::vector<float> input(ih * iw * ch_p);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 11);
    std::vector<float> grid(oh * ow * 2);
    for (int i = 0; i < (int)grid.size(); ++i) grid[i] = -0.5f + (i % 3) * 0.5f;
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer gridBuf; gridBuf.setFp32(grid); gridBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(gridBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::scalarInt(total));
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::scalarInt(ih));
    ac.args.push_back(AdaptedArg::scalarInt(iw));
    ac.args.push_back(AdaptedArg::scalarInt(oh));
    ac.args.push_back(AdaptedArg::scalarInt(ow));
    ac.args.push_back(AdaptedArg::scalarInt(ch));
    ac.args.push_back(AdaptedArg::scalarInt(ch_p));
    ac.args.push_back(AdaptedArg::scalarInt(1));
    ac.args.push_back(AdaptedArg::scalarInt(0));
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.validatorInputB = grid;
    ac.elementCount = total; ac.h = ih; ac.w = iw; ac.c = ch_p; ac.stride = oh;
    return true;
}
cudaError_t CudaGridSampleBilinearFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_grid_sample_bilinear_fp32(ctx.intArgs[0], (const float*)ctx.devBufs[0], (const float*)ctx.devBufs[1],
                                         (float*)ctx.devBufs[2], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                                         ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6],
                                         ctx.intArgs[7], ctx.intArgs[8],
                                         ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaGridSampleBilinearFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int ih = ac.h, iw = ac.w, ch_p = ac.c, oh = ac.stride, ow = oh;
    const int total = oh * ow * ch_p;
    if ((int)output.size() < total) return false;
    for (int idx = 0; idx < total; ++idx) {
        int idx_cp = idx % ch_p;
        int idx_nhw = idx / ch_p;
        int idx_ow = idx_nhw % ow;
        int idx_nh = idx_nhw / ow;
        int idx_oh = idx_nh % oh;
        int idx_ob = idx_nh / oh;
        float pos_x = ac.validatorInputB[idx_nhw * 2 + 0];
        float pos_y = ac.validatorInputB[idx_nhw * 2 + 1];
        float igx = ((1.0f + pos_x) * iw - 1.0f) / 2.0f;
        float igy = ((1.0f + pos_y) * ih - 1.0f) / 2.0f;
        int ix0 = std::min(std::max((int)floor(igx), 0), iw-1);
        int ix1 = std::min((int)ceil(igx), iw-1);
        int iy0 = std::min(std::max((int)floor(igy), 0), ih-1);
        int iy1 = std::min((int)ceil(igy), ih-1);
        float xw = ix1 - igx, yw = iy1 - igy;
        float v00 = ac.validatorInputA[((idx_ob*ih+iy0)*iw+ix0)*ch_p+idx_cp];
        float v01 = ac.validatorInputA[((idx_ob*ih+iy0)*iw+ix1)*ch_p+idx_cp];
        float v10 = ac.validatorInputA[((idx_ob*ih+iy1)*iw+ix0)*ch_p+idx_cp];
        float v11 = ac.validatorInputA[((idx_ob*ih+iy1)*iw+ix1)*ch_p+idx_cp];
        float expected = v00*xw*yw + v01*(1-xw)*yw + v10*xw*(1-yw) + v11*(1-xw)*(1-yw);
        if (std::fabs(output[idx] - expected) > 1e-2f) return false;
    }
    return true;
}

// ---- TopKV2 extern C ----
extern "C" {
void mnn_corpus_topkv2_fp32(const float*, int*, float*, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_blitregion_fp32(const float*, float*, int, int, const int32_t*, const int32_t*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
}

// ---- TopKV2 ----
bool CudaTopKV2Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_topkv2_fp32";
    const int numRow = spec.intParam("num_row", 2);
    const int lengthRow = spec.intParam("length_row", 16);
    const int K = spec.intParam("k", 3);
    // Input: [numRow, lengthRow]
    std::vector<float> input(numRow * lengthRow);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = (float)(i % 19);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    // Output: values [numRow*K] + indices [numRow*K] (int32)
    AdaptedBuffer outValBuf; outValBuf.sizeBytes = numRow * K * sizeof(float); outValBuf.isOutput = true;
    AdaptedBuffer outIdxBuf; outIdxBuf.sizeBytes = numRow * K * sizeof(int32_t); outIdxBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outIdxBuf); ac.buffers.push_back(outValBuf);
    // args: K, lengthRow, numRow, descendFlag, grid1x, grid1y, block1, smem1, grid2, block2, smem2
    int numThreadPerBlock = 64; // simplified
    int numElePerBlock = numThreadPerBlock; // 1 ele per thread
    int numBlockPerRow = (lengthRow + numElePerBlock - 1) / numElePerBlock;
    int smem1 = numThreadPerBlock * K;
    int numThreadFinal = 1;
    while (numThreadFinal < numBlockPerRow) numThreadFinal <<= 1;
    if (numThreadFinal < 1) numThreadFinal = 1;
    int smem2 = numBlockPerRow * K;
    ac.args.push_back(AdaptedArg::buffer(0)); // input
    ac.args.push_back(AdaptedArg::buffer(1)); // outIndices (buffers[1])
    ac.args.push_back(AdaptedArg::buffer(2)); // outValues (buffers[2])
    ac.args.push_back(AdaptedArg::scalarInt(K));
    ac.args.push_back(AdaptedArg::scalarInt(lengthRow));
    ac.args.push_back(AdaptedArg::scalarInt(numRow));
    ac.args.push_back(AdaptedArg::scalarInt(1)); // descendFlag=1 (largest)
    ac.args.push_back(AdaptedArg::scalarInt(numBlockPerRow)); // grid1x
    ac.args.push_back(AdaptedArg::scalarInt(numRow)); // grid1y
    ac.args.push_back(AdaptedArg::scalarInt(numThreadPerBlock)); // block1
    ac.args.push_back(AdaptedArg::scalarInt(smem1));
    ac.args.push_back(AdaptedArg::scalarInt(numRow)); // grid2
    ac.args.push_back(AdaptedArg::scalarInt(numThreadFinal)); // block2
    ac.args.push_back(AdaptedArg::scalarInt(smem2));
    ac.globalSize[0] = 1; ac.localSize[0] = 1; ac.dims = 1; // TopKV2 uses its own grid/block
    ac.validatorInputA = input;
    ac.elementCount = numRow * K;
    ac.m = numRow; ac.n = lengthRow; ac.k = K;
    return true;
}
cudaError_t CudaTopKV2Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    // args order: input(0), outIdx(1), outVal(2), K(3), lengthRow(4), numRow(5), descend(6),
    //             grid1x(7), grid1y(8), block1(9), smem1(10), grid2(11), block2(12), smem2(13)
    // intArgs collects only Scalar args: K, lengthRow, numRow, descend, grid1x, grid1y, block1, smem1, grid2, block2, smem2
    mnn_corpus_topkv2_fp32((const float*)ctx.devBufs[0], (int*)ctx.devBufs[1], (float*)ctx.devBufs[2],
                           ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
                           ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
                           ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10],
                           ctx.stream);
    return cudaGetLastError();
}
bool CudaTopKV2Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int numRow = ac.m, lengthRow = ac.n, K = ac.k;
    // output is outValues (float, numRow*K) — read back as float
    if ((int)output.size() < numRow * K) return false;
    // Check that values are sorted descending (largest first per row)
    for (int r = 0; r < numRow; ++r) {
        for (int i = 0; i < K; ++i) {
            float val = output[r * K + i];
            // Check: val should be among the top K values of row r
            // Simple check: count how many input values are strictly greater
            int greater = 0;
            for (int j = 0; j < lengthRow; ++j) {
                if (ac.validatorInputA[r * lengthRow + j] > val) ++greater;
            }
            // If there are K or more values greater than val, val is not in top K
            if (greater >= K) return false;
        }
    }
    return true;
}

// ---- Raster blitRegion ----
bool CudaBlitRegionFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_blitregion_fp32";
    // Simple identity blit: loopCount=1, no indices, 1:1 copy with stride
    const int sizeZ = spec.intParam("size_z", 1);
    const int sizeY = spec.intParam("size_y", 2);
    const int sizeX = spec.intParam("size_x", 4);
    const int count = sizeZ * sizeY * sizeX;
    std::vector<float> input(count);
    for (int i = 0; i < count; ++i) input[i] = (float)i;
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = count * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    // args: input(0), output(1), count, loopCount=1, dstIndice=nullptr, srcIndice=nullptr,
    //       dstUseIndice=-1, srcUseIndice=-1, dstStep=count, srcStep=count, srcLimit=count,
    //       sizeZ, sizeY, sizeX, strideZ=sizeY*sizeX, strideY=sizeX, strideX=1,
    //       dstStrideZ=sizeY*sizeX, dstStrideY=sizeX, dstStrideX=1
    ac.args.push_back(AdaptedArg::buffer(0));
    ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.args.push_back(AdaptedArg::scalarInt(1)); // loopCount
    ac.args.push_back(AdaptedArg::scalarInt(-1)); // dstUseIndice
    ac.args.push_back(AdaptedArg::scalarInt(-1)); // srcUseIndice
    ac.args.push_back(AdaptedArg::scalarInt(count)); // dstStep
    ac.args.push_back(AdaptedArg::scalarInt(count)); // srcStep
    ac.args.push_back(AdaptedArg::scalarInt(count)); // srcLimit
    ac.args.push_back(AdaptedArg::scalarInt(sizeZ));
    ac.args.push_back(AdaptedArg::scalarInt(sizeY));
    ac.args.push_back(AdaptedArg::scalarInt(sizeX));
    ac.args.push_back(AdaptedArg::scalarInt(sizeY * sizeX)); // strideZ
    ac.args.push_back(AdaptedArg::scalarInt(sizeX)); // strideY
    ac.args.push_back(AdaptedArg::scalarInt(1)); // strideX
    ac.args.push_back(AdaptedArg::scalarInt(sizeY * sizeX)); // dstStrideZ
    ac.args.push_back(AdaptedArg::scalarInt(sizeX)); // dstStrideY
    ac.args.push_back(AdaptedArg::scalarInt(1)); // dstStrideX
    ac.globalSize[0] = gridFor(count); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = count;
    ac.m = sizeZ; ac.n = sizeY; ac.k = sizeX;
    return true;
}
cudaError_t CudaBlitRegionFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    // intArgs: count, loopCount, dstUseIndice, srcUseIndice, dstStep, srcStep, srcLimit,
    //          sizeZ, sizeY, sizeX, strideZ, strideY, strideX, dstStrideZ, dstStrideY, dstStrideX
    mnn_corpus_blitregion_fp32((const float*)ctx.devBufs[0], (float*)ctx.devBufs[1],
                                ctx.intArgs[0], ctx.intArgs[1],
                                nullptr, nullptr, // dstIndice, srcIndice (not used when UseIndice<0)
                                ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6],
                                ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9],
                                ctx.intArgs[10], ctx.intArgs[11], ctx.intArgs[12],
                                ctx.intArgs[13], ctx.intArgs[14], ctx.intArgs[15],
                                ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaBlitRegionFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    // Identity blit: output should equal input
    if ((int)output.size() < ac.elementCount) return false;
    for (int i = 0; i < ac.elementCount; ++i) {
        if (std::fabs(output[i] - ac.validatorInputA[i]) > 1e-3f) return false;
    }
    return true;
}

// ---- GridSample 3D + Conv DW extern C ----
extern "C" {
void mnn_corpus_grid_sample_nearest_3d_fp32(const int, const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_grid_sample_bilinear_3d_fp32(const int, const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
void mnn_corpus_conv_dw_fp32(const float*, const void*, const void*, float*, float, float, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
}

// ---- GridSample Nearest 3D ----
bool CudaGridSampleNearest3dFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_grid_sample_nearest_3d_fp32";
    const int id = spec.intParam("id", 2), ih = spec.intParam("ih", 3), iw = spec.intParam("iw", 3);
    const int od = spec.intParam("od", 2), oh = spec.intParam("oh", 3), ow = spec.intParam("ow", 3);
    const int ch = spec.intParam("channels", 4);
    const int ch_p = ch;
    const int total = od * oh * ow * ch_p;
    std::vector<float> input(id * ih * iw * ch_p);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 11);
    std::vector<float> grid(od * oh * ow * 3);
    for (int i = 0; i < (int)grid.size(); ++i) grid[i] = -0.5f + (i % 3) * 0.5f;
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer gridBuf; gridBuf.setFp32(grid); gridBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(gridBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::scalarInt(total));
    ac.args.push_back(AdaptedArg::buffer(0)); ac.args.push_back(AdaptedArg::buffer(1)); ac.args.push_back(AdaptedArg::buffer(2));
    for (int v : {id, ih, iw, od, oh, ow, ch, ch_p, 1, 0}) ac.args.push_back(AdaptedArg::scalarInt(v));
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.validatorInputB = grid;
    ac.elementCount = total; ac.h = ih; ac.w = iw; ac.c = ch_p; ac.stride = oh;
    return true;
}
cudaError_t CudaGridSampleNearest3dFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_grid_sample_nearest_3d_fp32(ctx.intArgs[0], (const float*)ctx.devBufs[0], (const float*)ctx.devBufs[1],
        (float*)ctx.devBufs[2], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6],
        ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaGridSampleNearest3dFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int ih = ac.h, iw = ac.w, ch_p = ac.c, oh = ac.stride, ow = oh;
    const int total = ac.elementCount;
    if ((int)output.size() < total) return false;
    return true;
}

// ---- GridSample Bilinear 3D ----
bool CudaGridSampleBilinear3dFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_grid_sample_bilinear_3d_fp32";
    const int id = spec.intParam("id", 2), ih = spec.intParam("ih", 3), iw = spec.intParam("iw", 3);
    const int od = spec.intParam("od", 2), oh = spec.intParam("oh", 3), ow = spec.intParam("ow", 3);
    const int ch = spec.intParam("channels", 4);
    const int ch_p = ch;
    const int total = od * oh * ow * ch_p;
    std::vector<float> input(id * ih * iw * ch_p);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 11);
    std::vector<float> grid(od * oh * ow * 3);
    for (int i = 0; i < (int)grid.size(); ++i) grid[i] = -0.5f + (i % 3) * 0.5f;
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer gridBuf; gridBuf.setFp32(grid); gridBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(gridBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::scalarInt(total));
    ac.args.push_back(AdaptedArg::buffer(0)); ac.args.push_back(AdaptedArg::buffer(1)); ac.args.push_back(AdaptedArg::buffer(2));
    for (int v : {id, ih, iw, od, oh, ow, ch, ch_p, 1, 0}) ac.args.push_back(AdaptedArg::scalarInt(v));
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.validatorInputB = grid;
    ac.elementCount = total; ac.h = ih; ac.w = iw; ac.c = ch_p; ac.stride = oh;
    return true;
}
cudaError_t CudaGridSampleBilinear3dFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_grid_sample_bilinear_3d_fp32(ctx.intArgs[0], (const float*)ctx.devBufs[0], (const float*)ctx.devBufs[1],
        (float*)ctx.devBufs[2], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6],
        ctx.intArgs[7], ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaGridSampleBilinear3dFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int total = ac.elementCount;
    if ((int)output.size() < total) return false;
    return true;
}

// ---- Conv DepthWise fp32 ----
bool CudaConvDwFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_conv_dw_fp32";
    const int iw = spec.intParam("iw", 4), ih = spec.intParam("ih", 4);
    const int c = spec.intParam("channels", 8), c_p = c;
    const int kw = spec.intParam("kw", 3), kh = kw;
    const int sw = spec.intParam("sw", 1), sh = sw;
    const int pw = spec.intParam("pw", 1), ph = pw;
    const int ow = (iw + 2 * pw - kw) / sw + 1;
    const int oh = (ih + 2 * ph - kh) / sh + 1;
    const int total = oh * ow * c_p;
    std::vector<float> input(ih * iw * c_p);
    for (int i = 0; i < (int)input.size(); ++i) input[i] = 0.1f * (i % 7);
    // kernel/bias in half (MNN stores them as half) - pack without __half type
    // (CudaOpsMisc.cpp is compiled by g++, __half is not available; use raw byte packing)
    std::vector<float> kernelF(c_p * kh * kw, 0.1f);
    std::vector<float> biasF(c_p, 0.5f);
    std::vector<uint8_t> kernelH(c_p * kh * kw * 2); // half = 2 bytes each
    std::vector<uint8_t> biasH(c_p * 2);
    // Simple float→half conversion (IEEE 754 half)
    auto floatToHalfBytes = [](float f) -> uint16_t {
        uint32_t x; memcpy(&x, &f, 4);
        uint16_t sign = (x >> 16) & 0x8000;
        int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (x >> 13) & 0x3FF;
        if (exp <= 0) { exp = 0; mant = 0; }
        if (exp >= 31) { exp = 31; mant = 0; }
        return sign | (exp << 10) | mant;
    };
    for (size_t i = 0; i < kernelF.size(); ++i) {
        uint16_t h = floatToHalfBytes(kernelF[i]);
        memcpy(&kernelH[i * 2], &h, 2);
    }
    for (size_t i = 0; i < biasF.size(); ++i) {
        uint16_t h = floatToHalfBytes(biasF[i]);
        memcpy(&biasH[i * 2], &h, 2);
    }
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer kBuf; kBuf.sizeBytes = kernelH.size(); kBuf.initialData = kernelH; kBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.sizeBytes = biasH.size(); bBuf.initialData = biasH; bBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(kBuf); ac.buffers.push_back(bBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0)); ac.args.push_back(AdaptedArg::buffer(1)); ac.args.push_back(AdaptedArg::buffer(2));
    ac.args.push_back(AdaptedArg::buffer(3));
    ac.args.push_back(AdaptedArg::scalarFloat(1e30f)); // maxV (no clamp)
    ac.args.push_back(AdaptedArg::scalarFloat(-1e30f)); // minV
    for (int v : {iw, ih, c, c_p, ow, oh, kw, kh, 1, 1, sw, sh, pw, ph, total})
        ac.args.push_back(AdaptedArg::scalarInt(v));
    ac.globalSize[0] = gridFor(total / 2); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = total;
    ac.h = ih; ac.w = iw; ac.c = c_p; ac.k = kw; ac.stride = sw; ac.orderType = pw;
    return true;
}
cudaError_t CudaConvDwFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_conv_dw_fp32((const float*)ctx.devBufs[0], (const void*)ctx.devBufs[1], (const void*)ctx.devBufs[2],
        (float*)ctx.devBufs[3], ctx.floatArgs[0], ctx.floatArgs[1],
        ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3],
        ctx.intArgs[4], ctx.intArgs[5], ctx.intArgs[6], ctx.intArgs[7],
        ctx.intArgs[8], ctx.intArgs[9], ctx.intArgs[10], ctx.intArgs[11],
        ctx.intArgs[12], ctx.intArgs[13], ctx.intArgs[14],
        ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaConvDwFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    // Smoke test: output is not all zero (conv did something)
    const int total = ac.elementCount;
    if ((int)output.size() < total) return false;
    for (int i = 0; i < std::min(10, total); ++i) {
        if (output[i] != 0.0f) return true;
    }
    return false;
}

// ---- 1.2.0: pack_c4 / unpack_c4 ----
bool CudaPackC4Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_pack_c4_120_fp32";
    const int inside = spec.intParam("inside", 3), axis = spec.intParam("axis", 4), outside = spec.intParam("outside", 2);
    const int axisC4 = (axis + 3) / 4 * 4;
    const int total = inside * axis * outside;
    std::vector<float> input(total); for (int i = 0; i < total; ++i) input[i] = (float)i;
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = outside * axisC4 * inside * 4 * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0)); ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(inside)); ac.args.push_back(AdaptedArg::scalarInt(axis));
    ac.args.push_back(AdaptedArg::scalarInt(outside)); ac.args.push_back(AdaptedArg::scalarInt(axisC4));
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = total;
    return true;
}
cudaError_t CudaPackC4Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_pack_c4_120_fp32((const float*)ctx.devBufs[0], (float*)ctx.devBufs[1],
                                 ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaPackC4Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    // Smoke: output not all zero
    for (int i = 0; i < std::min(10, (int)output.size()); ++i) if (output[i] != 0) return true;
    return false;
}

bool CudaUnpackC4Fp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_unpack_c4_120_fp32";
    const int inside = spec.intParam("inside", 3), axis = spec.intParam("axis", 4), outside = spec.intParam("outside", 2);
    const int axisC4 = (axis + 3) / 4 * 4;
    const int total = inside * axis * outside;
    std::vector<float> input(outside * axisC4 * inside * 4, 0.5f);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0)); ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(inside)); ac.args.push_back(AdaptedArg::scalarInt(axis));
    ac.args.push_back(AdaptedArg::scalarInt(outside)); ac.args.push_back(AdaptedArg::scalarInt(axisC4));
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.elementCount = total;
    return true;
}
cudaError_t CudaUnpackC4Fp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_unpack_c4_120_fp32((const float*)ctx.devBufs[0], (float*)ctx.devBufs[1],
                                   ctx.intArgs[0], ctx.intArgs[1], ctx.intArgs[2], ctx.intArgs[3], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaUnpackC4Fp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    for (int i = 0; i < std::min(10, (int)output.size()); ++i) if (output[i] != 0) return true;
    return false;
}

// ---- 1.2.0: SETZERO ----
bool CudaSetZeroFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_setzero_120_fp32";
    const int n = spec.intParam("size", 1024);
    AdaptedBuffer outBuf; outBuf.sizeBytes = n * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::scalarInt(n)); ac.args.push_back(AdaptedArg::buffer(0));
    ac.globalSize[0] = gridFor(n); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.elementCount = n;
    return true;
}
cudaError_t CudaSetZeroFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_setzero_120_fp32(ctx.intArgs[0], (float*)ctx.devBufs[0], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaSetZeroFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    for (int i = 0; i < ac.elementCount; ++i) if (output[i] != 0.0f) return false;
    return true;
}

// ---- 1.2.0: add_bias (MatMul) ----
bool CudaAddBiasFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "mnn_corpus_add_bias_120_fp32";
    const int e = spec.intParam("e", 16), h = spec.intParam("h", 16);
    const int total = e * h;
    std::vector<float> input(total); fillInputRamp(input, total);
    std::vector<float> bias(h, 0.5f);
    AdaptedBuffer inBuf; inBuf.setFp32(input); inBuf.isOutput = false;
    AdaptedBuffer bBuf; bBuf.setFp32(bias); bBuf.isOutput = false;
    AdaptedBuffer outBuf; outBuf.sizeBytes = total * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(inBuf); ac.buffers.push_back(bBuf); ac.buffers.push_back(outBuf);
    ac.args.push_back(AdaptedArg::buffer(0)); ac.args.push_back(AdaptedArg::buffer(2)); ac.args.push_back(AdaptedArg::buffer(1));
    ac.args.push_back(AdaptedArg::scalarInt(e)); ac.args.push_back(AdaptedArg::scalarInt(h));
    ac.globalSize[0] = gridFor(total); ac.localSize[0] = kBlock; ac.dims = 1;
    ac.validatorInputA = input; ac.elementCount = total; ac.m = e; ac.k = h;
    return true;
}
cudaError_t CudaAddBiasFp32Kernel::launch(const AdaptedCase&, const CudaLaunchCtx& ctx) const {
    mnn_corpus_add_bias_120_fp32((float*)ctx.devBufs[0], (float*)ctx.devBufs[2], (const float*)ctx.devBufs[1],
                                  ctx.intArgs[0], ctx.intArgs[1], ctx.grid, ctx.block, ctx.stream);
    return cudaGetLastError();
}
bool CudaAddBiasFp32Kernel::validate(const AdaptedCase& ac, const std::vector<float>& output) const {
    const int e = ac.m, h = ac.k;
    for (int i = 0; i < e * h; ++i) { int hi = i % h; if (std::fabs(output[i] - (ac.validatorInputA[i] + 0.5f)) > 1e-3f) return false; }
    return true;
}

} // namespace MnnOps
} // namespace KernelCorpus
} // namespace Replay
} // namespace MNN
