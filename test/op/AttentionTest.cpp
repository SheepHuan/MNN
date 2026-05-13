//
//  AttentionTest.cpp
//  MNNTests
//
//  Created by MNN on 2024/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "core/OpCommonUtils.hpp"
#include "core/MNNFileUtils.h"
#include "core/PagedKVCachePlan.hpp"
#include "core/PrefixCachePath.hpp"
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include <stdlib.h>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>
#include <MNN/AutoTime.hpp>
#include <half.hpp>

using namespace MNN::Express;
using MNN::KVMeta;

int NumHead   = 16;
int KvNumHead = 2;
int HeadDim   = 128;
const float diff_threshold = 0.001;
const float diff_percent_threshold = 0.1;
const int pastLength = 101;
#define GENERATE_TOKENS 128

static KVMeta gMeta;
static std::shared_ptr<Module> _makeAttentionModuleEx(MNN::OpType opType,
                                                      int attentionMode = 8,
                                                      int layerIndex = -1,
                                                      const std::string& prefixCacheDir = "") {
    auto Q = _Input();
    auto K = _Input();
    auto V = _Input();
    auto mask = _Input();
    std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
    attention->type = opType;
    attention->main.type = MNN::OpParameter_AttentionParam;
    attention->main.value = new MNN::AttentionParamT;
    attention->main.AsAttentionParam()->kv_cache = true;
    attention->main.AsAttentionParam()->layer_index = layerIndex;
    auto o = Variable::create(Expr::create(attention.get(), {Q, K, V, mask}));
    auto buffer = Variable::save({o});
    MNN::ScheduleConfig config;
    auto status = MNNTestSuite::get()->pStaus;
    config.type = (MNNForwardType)status.forwardType;
    MNN::BackendConfig bnConfig;
    bnConfig.memory = (MNN::BackendConfig::MemoryMode)status.memory;
    bnConfig.precision = (MNN::BackendConfig::PrecisionMode)status.precision;
    bnConfig.power = (MNN::BackendConfig::PowerMode)status.power;
    config.backendConfig = &bnConfig;
    config.numThread = status.forwardType == MNN_FORWARD_OPENCL ?
        (MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_FAST) : 1;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setHintPtr(MNN::Interpreter::KVCACHE_INFO, &gMeta);
    rtmgr->setHint(MNN::Interpreter::ATTENTION_OPTION, attentionMode);
    if (!prefixCacheDir.empty()) {
        rtmgr->setExternalPath(prefixCacheDir, MNN::Interpreter::EXTERNAL_PATH_PREFIXCACHE_DIR);
        auto runtimeKVDir = MNNFilePathConcat(prefixCacheDir, "runtime");
        MNNCreateDir(prefixCacheDir.c_str());
        MNNCreateDir(runtimeKVDir.c_str());
        rtmgr->setExternalPath(runtimeKVDir, MNN::Interpreter::EXTERNAL_PATH_KVCACHE_DIR);
    }
    std::shared_ptr<Module> m(Module::load({}, {}, (uint8_t*)buffer.data(), buffer.size(), rtmgr));
    return m;
}

static std::shared_ptr<Module> _makeAttentionModule(int attentionMode = 8) {
    return _makeAttentionModuleEx(MNN::OpType_Attention, attentionMode);
}

struct KVCache {
    VARP pastK;
    VARP pastV;
    VARP pastMask;
    int current = 0;
    KVCache() {
        pastK = _Input({1, KvNumHead, 1, pastLength, HeadDim}, NCHW);
        pastV = _Input({1, KvNumHead, 1, pastLength, HeadDim}, NCHW);
        pastMask = _Input({pastLength}, NCHW);
        ::memset(pastK->writeMap<float>(), 0, pastK->getInfo()->size * sizeof(float));
        ::memset(pastV->writeMap<float>(), 0, pastK->getInfo()->size * sizeof(float));
        for (int v=0; v<pastLength; ++v) {
            pastMask->writeMap<float>()[v] = std::numeric_limits<float>::lowest();
        }
    }
};

static VARP _computeAttentionExpr(VARP Q, VARP K, VARP V, VARP mask, KVCache cache) {
    auto qinfo = Q->getInfo();
    auto kinfo = K->getInfo();
    auto vinfo = V->getInfo();
    auto seqLength = qinfo->dim[1];
    auto numHead = qinfo->dim[2];
    auto headDim = qinfo->dim[3];
    auto kvNumHead = kinfo->dim[2];
    auto batch = qinfo->dim[0];
    auto group = numHead / kvNumHead;
    if (mask->getInfo()->type.code == halide_type_int) {
        mask = (_Scalar<float>(1.0) - _Cast<float>(mask)) * _Scalar<float>(std::numeric_limits<float>::lowest());
    }

    Q = _Reshape(Q, {batch, seqLength, kvNumHead,group, headDim});
    Q = _Transpose(Q, {0, 2, 3, 1, 4});
    K = _Reshape(K, {batch, seqLength, kvNumHead, 1, headDim});
    K = _Transpose(K, {0, 2, 3, 1, 4});

    auto scale = 1.0f / sqrtf(headDim);
    K = K * _Scalar<float>(scale);
    K.fix(VARP::CONSTANT);
    auto QK = _MatMul(Q, K, false, true); // [batch, kvNumHead, group , seq_len, seq_len]
    QK = QK + mask;
    auto QKPast = _MatMul(Q, cache.pastK, false, true);
    QKPast = QKPast + cache.pastMask;
    QK = _Concat({QKPast, QK}, -1);
    QK = _Softmax(QK, -1);
    V = _Reshape(V, {batch, seqLength, kvNumHead, 1, headDim});
    V = _Transpose(V, {0, 2, 3, 1, 4});
    V.fix(VARP::CONSTANT);
    auto totalV = _Concat({cache.pastV, V}, 3);
    auto QKV = _MatMul(QK, totalV, false, false);
    auto info = QKV->getInfo();
    auto O = _Transpose(QKV, {0, 3, 1, 2, 4});
    O = _Reshape(O, {batch, seqLength, -1});
    O.fix(VARP::CONSTANT);
    // Update KVCache
    for (int y=0; y<kvNumHead; ++y) {
        ::memcpy(cache.pastK->writeMap<float>() + y * pastLength * headDim + cache.current * headDim, K->readMap<float>() + y * seqLength * headDim, seqLength * headDim * sizeof(float));
        ::memcpy(cache.pastV->writeMap<float>() + y * pastLength * headDim + cache.current * headDim, V->readMap<float>() + y * seqLength * headDim, seqLength * headDim * sizeof(float));
    }
    for (int i=0; i<seqLength; ++i) {
        cache.pastMask->writeMap<float>()[i+cache.current] = 0.0f;
    }
    cache.current += seqLength;
    return O;
}

static std::vector< std::vector< std::vector<float> > > generateRandTensor(int C, int H, int W, int precision) {
    std::vector< std::vector< std::vector<float> > > a;
    a.resize(C);
    for (int i = 0; i < C; i++) {
        a[i].resize(H);
        for (int j = 0; j < H; j++) {
            a[i][j].resize(W);
            for (int k = 0; k < W; k++) {
                if (precision == 2) {
                    a[i][j][k] = ((i + j + k) % 10) * 0.002;
                } else {
                    a[i][j][k] = ((i + j + k) % 10) * 0.16 - 5.6;
                }
            }
        }
    }
    return a;
}

VARP vector_to_var(std::vector< std::vector< std::vector<float> > > & a) {
    int C = a.size();
    int H = a[0].size();
    int W = a[0][0].size();
    VARP var = _Input({1, C, H, W}, NCHW, halide_type_of<float>());
    float * ptr = var->writeMap<float>();
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < H; j++) {
            for (int k = 0; k < W; k++) {
                ptr[i * H * W + j * W + k] = a[i][j][k];
            }
        }
    }
    var->unMap();
    return var;
}

VARP vector_to_var(std::vector< std::vector<int> > & a) {
    int H = a.size();
    int W = a[0].size();
    VARP var = _Input({1, 1, H, W}, NCHW, halide_type_of<int>());
    int * ptr = var->writeMap<int>();
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            ptr[i * W + j] = a[i][j];
        }
    }
    var->unMap();
    return var;
}

static std::vector< std::vector< std::vector<float> > >
computeAttention (
    std::vector< std::vector< std::vector<float> > > & query,
    std::vector< std::vector< std::vector<float> > > & key,
    std::vector< std::vector< std::vector<float> > > & value,
    std::vector< std::vector<int> > & mask,
    int seq_len, int kv_seq_len )
{
    int group_size = NumHead / KvNumHead;
    std::vector< std::vector< std::vector<float> > > output(seq_len);
    for (int i = 0; i < seq_len; i++) {
        output[i].resize(NumHead);
        for (int j = 0; j < NumHead; j++) {
            output[i][j].resize(HeadDim);
        }
    }
    for (int h = 0; h < NumHead; h++) {
        int kv_h = h / group_size;
        /*---- Q * K ----*/
        std::vector< std::vector<float> > qk(seq_len, std::vector<float>(kv_seq_len, 0.0f));
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < kv_seq_len; j++) {
                qk[i][j] = 0.0f;
                for (int k = 0; k < HeadDim; k++) {
                    qk[i][j] += query[i][h][k] * key[j][kv_h][k];
                }
            }
        }
        /*---- Mask QK ----*/
        if(mask.size() > 0) {
            float scale = 1.0 / sqrt(HeadDim);
            if (mask[0].size() == seq_len) {
                auto diff = kv_seq_len - seq_len;
                for (int i = 0; i < seq_len; i++) {
                    for (int j = 0; j < seq_len; j++) {
                        qk[i][j+diff] = qk[i][j+diff] * scale + (1.f - mask[i][j]) * std::numeric_limits<float>::lowest();
                    }
                }
            } else {
                for (int i = 0; i < seq_len; i++) {
                    for (int j = 0; j < kv_seq_len; j++) {
                        qk[i][j] = qk[i][j] * scale + (1.f - mask[i][j]) * std::numeric_limits<float>::lowest();
                    }
                }
            }
        } else {
            float scale = 1.0 / sqrt(HeadDim);
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < kv_seq_len; j++) {
                    qk[i][j] *= scale;
                }
            }
        }
        /*---- Softmax QK ----*/
        for (int i = 0; i < seq_len; i++) {
            float maxValue = qk[i][0];
            for (int j = 1; j < kv_seq_len; j++) {
                maxValue = ALIMAX(maxValue, qk[i][j]);
            }
            for (int j = 0; j < kv_seq_len; j++) {
                qk[i][j] -= maxValue;
            }
            float sum = 0.0f;
            for (int j = 0; j < kv_seq_len; j++) {
                sum += exp(qk[i][j]);
            }
            for (int j = 0; j < kv_seq_len; j++) {
                qk[i][j] = exp(qk[i][j]) / sum;
            }
        }
        /*---- QK * V ----*/
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < HeadDim; j++) {
                output[i][h][j] = 0.0f;
                for (int k = 0; k < kv_seq_len; k++) {
                    output[i][h][j] += qk[i][k] * value[k][kv_h][j];
                }
            }
        }
    }
    return output;
}

class NaiveAttention {
    private:
        std::vector< std::vector< std::vector<float> > >  mPastKey, mPastValue;
        int mPastLen;
    public:
        NaiveAttention() : mPastLen(0) {}
        ~NaiveAttention() = default;
        std::vector< std::vector< std::vector<float> > > onExecute (
            std::vector< std::vector< std::vector<float> > > & query,
            std::vector< std::vector< std::vector<float> > > & key,
            std::vector< std::vector< std::vector<float> > > & value,
            std::vector< std::vector<int> > & mask,
            int seq_len )
        {
            for (int i = 0; i < seq_len; i++) {
                mPastKey.push_back(key[i]);
                mPastValue.push_back(value[i]);
            }
            mPastLen += seq_len;
            return computeAttention(query, mPastKey, mPastValue, mask, seq_len, mPastLen);
        }
};

class AttentionTest : public MNNTestCase {
protected:
    std::vector< std::vector< std::vector<float> > > query;
    std::vector< std::vector< std::vector<float> > > key;
    std::vector< std::vector< std::vector<float> > > value;
    std::vector< std::vector<int> > mask;
    std::vector< std::vector< std::vector<float> > > expected_result;
    VARP Query, Key, Value, Mask, Output;
    VARP Query1, Key1, Value1, Mask1;
public:
    AttentionTest() = default;
    virtual ~AttentionTest() = default;
    void generateInput(int seq_len, int precision, bool genDecodeInput = false) {
        query = generateRandTensor(seq_len, NumHead, HeadDim, precision);
        key   = generateRandTensor(seq_len, KvNumHead, HeadDim, precision);
        value = generateRandTensor(seq_len, KvNumHead, HeadDim, precision);
        Query = vector_to_var(query);
        Key   = vector_to_var(key);
        Value = vector_to_var(value);
        if (genDecodeInput) {
            auto vecquery = generateRandTensor(1, NumHead, HeadDim, precision);
            auto veckey   = generateRandTensor(1, KvNumHead, HeadDim, precision);
            auto vecvalue = generateRandTensor(1, KvNumHead, HeadDim, precision);
            Query1 = vector_to_var(vecquery);
            Key1   = vector_to_var(veckey);
            Value1 = vector_to_var(vecvalue);
        }
    }
    void generateChunkMask(int seq_len, int kv_seq_len, int chunk_size, bool genDecodeInput = false) {
        // 防止除以0
        if (chunk_size <= 0) chunk_size = 1;

        mask.resize(seq_len);

        // 计算历史长度 (Gap)，用于处理 KV 长度大于 Seq 长度的情况 (Right Alignment)
        // j < gap 的部分通常被视为 History，默认可见
        int gap = kv_seq_len - seq_len;

        for (int i = 0; i < seq_len; i++) {
            mask[i].resize(kv_seq_len);

            // --- 核心逻辑对应 ---
            // MNN Expr: auto N = _Divide(i, rankVar) * rankVar + rankVar;
            // i 是当前行 (Query)，计算当前块的右边界 (不包含)
            // 比如 rank=2, i=0, block_end_rel=2; i=2, block_end_rel=4
            int block_end_rel = (i / chunk_size) * chunk_size + chunk_size;

            for (int j = 0; j < kv_seq_len; j++) {
                // 将 j 转换为相对于当前 seq_len 的坐标
                int j_rel = j - gap;

                if (j_rel < 0) {
                    // 情况 1: j 在 Gap 区域 (历史 KV Cache)
                    // 通常历史数据对当前所有 Token 都是可见的
                    mask[i][j] = 1;
                } else {
                    // 情况 2: j 在当前处理的序列范围内
                    // 对应 MNN Expr: _Less(j, N)
                    if (j_rel < block_end_rel) {
                        mask[i][j] = 1;
                    } else {
                        mask[i][j] = 0;
                    }
                }
            }
        }

        // 转为 VARP 并处理成 -inf / 0.0 格式
        Mask = vector_to_var(mask);
        Mask = (_Scalar<float>(1.0) - _Cast<float>(Mask)) * _Scalar<float>(std::numeric_limits<float>::lowest());

        // Decode Input 部分通常保持全 1 (即看清所有历史)，或者根据需求修改
        if (genDecodeInput) {
            std::vector<std::vector<int>> vecmask;
            vecmask.resize(1);
            vecmask[0].resize(gMeta.previous + 1);
            for (int i = 0; i < gMeta.previous + 1; ++i) {
                vecmask[0][i] = 1;
            }
            Mask1 = vector_to_var(vecmask);
            Mask1 = (_Scalar<float>(1.0) - _Cast<float>(Mask1)) * _Scalar<float>(std::numeric_limits<float>::lowest());
        }
    }

    void generateMask(int seq_len, int kv_seq_len, bool genDecodeInput = false) {
        mask.resize(seq_len);
        for (int i = 0; i < seq_len; i++) {
            mask[i].resize(kv_seq_len);
            for (int j = 0; j < kv_seq_len; j++) {
                if (j - i <= kv_seq_len - seq_len) {
                    mask[i][j] = 1;
                } else {
                    mask[i][j] = 0;
                }
            }
        }
        Mask = _Input({}, NCHW, halide_type_of<float>());
        Mask1 = _Input({}, NCHW, halide_type_of<float>());
        Mask->writeMap<float>()[0] = 0.0f;
        Mask1->writeMap<float>()[0] = 0.0f;
    }

    bool compareResult(int seq_len) {
        const float * resultPtr = Output->readMap<float>();
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < NumHead; j++) {
                for (int k = 0; k < HeadDim; k++) {
                    float diff = fabs(resultPtr[i * NumHead * HeadDim + j * HeadDim + k] - expected_result[i][j][k]);
                    float diff_percent = fabs(diff / expected_result[i][j][k]);
                    if (diff > diff_threshold && diff_percent > diff_percent_threshold) {
                        printf("Result Mismatch: expected %lf but got %lf in CPU Attention Test\n", expected_result[i][j][k], resultPtr[i * NumHead * HeadDim + j * HeadDim + k]);
                        printf("Error Position: Output[%d][%d][%d]\n", i, j, k);
                        return false;
                    }
                }
            }
        }
        Output->unMap();
        return true;
    }

    virtual bool run(int precision) {
        srand(2024);
        // unit test 1
        {
            std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
            std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
            attention->type = MNN::OpType_Attention;
            attention->main.type = MNN::OpParameter_AttentionParam;
            attention->main.value = new MNN::AttentionParamT;
            attention->main.AsAttentionParam()->kv_cache = true;
            int seq_len = 10;
            generateInput(seq_len, precision);
            generateMask(seq_len, seq_len);
            expected_result = naiveAttention->onExecute(query, key, value, mask, seq_len);
            auto attn = _makeAttentionModule();
            gMeta.add = seq_len;
            Output = attn->onForward({Query, Key, Value, Mask})[0];
            gMeta.sync();
            KVCache kvCache;
            bool pass = compareResult(seq_len);
            if (!pass) {
                printf("Error: LowerTriangular Attention with kv_cache unit test failed!\n");
                return false;
            }

            /* generate mask expr */
            /* generate mask expr */
            auto MaskExpr = vector_to_var(mask);
            MaskExpr = (_Scalar<float>(1.0) - _Cast<float>(MaskExpr)) * _Scalar<float>(std::numeric_limits<float>::lowest());
            Output = _computeAttentionExpr(Query, Key, Value, MaskExpr, kvCache);
            pass = compareResult(seq_len);
            if (!pass) {
                FUNC_PRINT(1);
                return false;
            }
            // naiveAttention with history is error, use expr to test
            Output = _computeAttentionExpr(Query, Key, Value, MaskExpr, kvCache);
            gMeta.add = seq_len;
            auto output2 = attn->onForward({Query, Key, Value, Mask})[0];
            gMeta.sync();
            auto diff = _ReduceMax(output2 - Output)->readMap<float>()[0];
            if (diff >= 0.01f) {                 FUNC_PRINT_ALL(diff, f);
                return false;
            }
        }
        // test2
        {
            std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
            std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
            attention->type = MNN::OpType_Attention;
            attention->main.type = MNN::OpParameter_AttentionParam;
            attention->main.value = new MNN::AttentionParamT;
            attention->main.AsAttentionParam()->kv_cache = true;
            int seq_len = 10;
            generateInput(seq_len, precision);
            generateChunkMask(seq_len, seq_len, 2);
            expected_result = naiveAttention->onExecute(query, key, value, mask, seq_len);
            auto attn = _makeAttentionModule();
            gMeta.previous = 0;
            gMeta.add = seq_len;
            Output = attn->onForward({Query, Key, Value, Mask})[0];
            gMeta.sync();
            KVCache kvCache;
            bool pass = compareResult(seq_len);
            if (!pass) {
                printf("Error: Not LowerTriangular Attention with kv_cache unit test failed!\n");
                return false;
            }
            Output = _computeAttentionExpr(Query, Key, Value, Mask, kvCache);
            pass = compareResult(seq_len);
            if (!pass) {
                FUNC_PRINT(1);
                return false;
            }
            // naiveAttention with history is error, use expr to test
            Output = _computeAttentionExpr(Query, Key, Value, Mask, kvCache);
            gMeta.add = seq_len;
            auto output2 = attn->onForward({Query, Key, Value, Mask})[0];
            gMeta.sync();
            auto diff = _ReduceMax(output2 - Output)->readMap<float>()[0];
            if (diff >= 0.01f) {
                FUNC_PRINT_ALL(diff, f);
                return false;
            }
        }
        // unit test 3
        {
            auto rtInfo = ExecutorScope::Current()->getRuntime().first;
            bool cpuInfer = true;
            for(auto &rt : rtInfo) {
                if(rt.first != MNN_FORWARD_CPU) {
                    cpuInfer = false;
                    break;
                }
            }
            if(cpuInfer) {
                // TODO: CPU support kv_cache == false
                return true;
            }
            std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
            std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
            attention->type = MNN::OpType_Attention;
            attention->main.type = MNN::OpParameter_AttentionParam;
            attention->main.value = new MNN::AttentionParamT;
            attention->main.AsAttentionParam()->kv_cache = false;
            int seq_len = 128;
            generateInput(seq_len, precision);
            mask.clear();
            expected_result = naiveAttention->onExecute(query, key, value, mask, seq_len);
            Output = Variable::create(Expr::create(attention.get(), {Query, Key, Value}));
            bool pass = compareResult(seq_len);
            if (!pass) {
                printf("Error: Attention without kv_cacheunit test failed!\n");
                return false;
            }
        }
        return true;
    }
};

class SpeedAttentionTest : public AttentionTest {
    protected:
        std::vector< std::vector< std::vector<float> > > query;
        std::vector< std::vector< std::vector<float> > > key;
        std::vector< std::vector< std::vector<float> > > value;
        std::vector< std::vector<int> > mask;
        std::vector< std::vector< std::vector<float> > > expected_result;

public:
SpeedAttentionTest() = default;
    virtual ~SpeedAttentionTest() = default;

    virtual bool run(int precision) {
        std::vector<int> seqs = {4096};
        std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
        std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
        attention->type = MNN::OpType_Attention;
        attention->main.type = MNN::OpParameter_AttentionParam;
        attention->main.value = new MNN::AttentionParamT;
        attention->main.AsAttentionParam()->kv_cache = true;
        /* 3 attention module */
        std::vector<int> quantQKV = {8, 9, 10};
        std::vector<std::string> testNames = {"float qkv", "quant qk", "quant qkv"};
        for (int n = 0; n < seqs.size(); ++n) {
            int seq_len = seqs[n];
            MNN_PRINT(">>> seq_len=%d, decode_len=%d\n", seq_len, GENERATE_TOKENS);
            generateInput(seqs[n], precision, true);
            generateMask(seqs[n], seq_len, true);
            for (int m = 0; m < testNames.size(); ++m) {
                gMeta.previous = 0;
                gMeta.add = seq_len;
                auto _module = _makeAttentionModule(quantQKV[m]);
                MNN::Timer t1;
                for (int x = 0; x < 5; ++x) {
                    Output = _module->onForward({Query, Key, Value, Mask})[0];
                }
                auto time = (float)t1.durationInUs() / 1000.0f / 5.f;
                MNN_PRINT("%s: prefill cost = %.2f\n", testNames[m].c_str(), time);
                gMeta.sync();
                MNN::Timer t2;
                for (int x = 0; x < GENERATE_TOKENS; ++x) {
                    gMeta.add = 1;
                    auto output2 = _module->onForward({Query1, Key1, Value1, Mask1})[0];
                    gMeta.sync();
                }
                time = (float)t2.durationInUs() / 1000.0f;
                MNN_PRINT("%s: decode cost = %f\n", testNames[m].c_str(), time);
            }
        }
        return true;
    }
};

namespace {

static constexpr int kPrefixBlockPageSize = 64;

static void resetKVMetaForAttentionTest() {
    gMeta = KVMeta();
}

static float prefixTestValue(int token, int head, int dim, int salt) {
    int v = (token + 3) * 131 + (head + 5) * 17 + (dim + 7) * 3 + salt * 29;
    return static_cast<float>((v % 97) - 48) * 0.0035f;
}

static void fillPrefixTestTensor(std::vector<float>& dst, int tokens, int heads, int headDim, int salt) {
    dst.resize(tokens * heads * headDim);
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            for (int d = 0; d < headDim; ++d) {
                dst[(t * heads + h) * headDim + d] = prefixTestValue(t, h, d, salt);
            }
        }
    }
}

static VARP prefixTensorToVar(const std::vector<float>& data, int tokens, int heads, int headDim) {
    VARP var = _Input({1, tokens, heads, headDim}, NCHW, halide_type_of<float>());
    ::memcpy(var->writeMap<float>(), data.data(), data.size() * sizeof(float));
    var->unMap();
    return var;
}

static void applyForwardRope(std::vector<float>& key, int tokens, int heads, int headDim,
                             int logicalStart, int ropeDim, float ropeTheta) {
    if (ropeDim <= 0) {
        ropeDim = headDim;
    }
    ropeDim = ALIMIN(ropeDim, headDim);
    int half = ropeDim / 2;
    for (int t = 0; t < tokens; ++t) {
        int position = logicalStart + t;
        for (int h = 0; h < heads; ++h) {
            for (int d = 0; d < half; ++d) {
                double invFreq = 1.0 / std::pow(static_cast<double>(ropeTheta),
                                                static_cast<double>(2 * d) / ropeDim);
                float angle = static_cast<float>(static_cast<double>(position) * invFreq);
                float cosValue = std::cos(angle);
                float sinValue = std::sin(angle);
                size_t leftIndex = (t * heads + h) * headDim + d;
                size_t rightIndex = (t * heads + h) * headDim + d + half;
                float left = key[leftIndex];
                float right = key[rightIndex];
                key[leftIndex] = left * cosValue - right * sinValue;
                key[rightIndex] = right * cosValue + left * sinValue;
            }
        }
    }
}

static std::vector<float> computePrefixReference(const std::vector<float>& query,
                                                 const std::vector<float>& docAKey,
                                                 const std::vector<float>& docAValue,
                                                 const std::vector<float>& docBKey,
                                                 const std::vector<float>& docBValue,
                                                 const std::vector<float>& promptKey,
                                                 const std::vector<float>& promptValue,
                                                 int docATokens, int docBTokens, int promptTokens,
                                                 int numHeads, int kvHeads, int headDim,
                                                 int ropeDim, float ropeTheta) {
    std::vector<float> docAKeyEncoded = docAKey;
    std::vector<float> docBKeyEncoded = docBKey;
    applyForwardRope(docAKeyEncoded, docATokens, kvHeads, headDim, 0, ropeDim, ropeTheta);
    applyForwardRope(docBKeyEncoded, docBTokens, kvHeads, headDim, docATokens, ropeDim, ropeTheta);

    const int kvTokens = docATokens + docBTokens + promptTokens;
    std::vector<float> key(kvTokens * kvHeads * headDim);
    std::vector<float> value(kvTokens * kvHeads * headDim);
    auto copyRange = [&](const std::vector<float>& src, std::vector<float>& dst, int dstTokenStart, int tokens) {
        ::memcpy(dst.data() + dstTokenStart * kvHeads * headDim,
                 src.data(), static_cast<size_t>(tokens) * kvHeads * headDim * sizeof(float));
    };
    copyRange(docAKeyEncoded, key, 0, docATokens);
    copyRange(docAValue, value, 0, docATokens);
    copyRange(docBKeyEncoded, key, docATokens, docBTokens);
    copyRange(docBValue, value, docATokens, docBTokens);
    copyRange(promptKey, key, docATokens + docBTokens, promptTokens);
    copyRange(promptValue, value, docATokens + docBTokens, promptTokens);

    const int group = numHeads / kvHeads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
    std::vector<float> output(promptTokens * numHeads * headDim, 0.0f);
    std::vector<float> qk(kvTokens);
    std::vector<float> softmax(kvTokens);
    for (int t = 0; t < promptTokens; ++t) {
        int visibleTokens = docATokens + docBTokens + t + 1;
        for (int h = 0; h < numHeads; ++h) {
            int kvh = h / group;
            float maxValue = -std::numeric_limits<float>::infinity();
            for (int k = 0; k < visibleTokens; ++k) {
                float dot = 0.0f;
                for (int d = 0; d < headDim; ++d) {
                    dot += query[(t * numHeads + h) * headDim + d] *
                           key[(k * kvHeads + kvh) * headDim + d];
                }
                qk[k] = dot * scale;
                maxValue = ALIMAX(maxValue, qk[k]);
            }
            float sum = 0.0f;
            for (int k = 0; k < visibleTokens; ++k) {
                softmax[k] = std::exp(qk[k] - maxValue);
                sum += softmax[k];
            }
            for (int k = 0; k < visibleTokens; ++k) {
                softmax[k] /= sum;
            }
            for (int d = 0; d < headDim; ++d) {
                float acc = 0.0f;
                for (int k = 0; k < visibleTokens; ++k) {
                    acc += softmax[k] * value[(k * kvHeads + kvh) * headDim + d];
                }
                output[(t * numHeads + h) * headDim + d] = acc;
            }
        }
    }
    return output;
}

template <typename T>
static T castPrefixFileScalar(float value) {
    return static_cast<T>(value);
}

template <>
half_float::half castPrefixFileScalar<half_float::half>(float value) {
    return half_float::half(value);
}

template <typename T>
static bool writeCudaPagedPrefixKVFile(const std::string& path, const std::vector<float>& data,
                                       int tokens, int heads, int headDim, bool valueLayout) {
    int blocks = UP_DIV(tokens, kPrefixBlockPageSize);
    std::vector<T> packed(static_cast<size_t>(blocks) * kPrefixBlockPageSize * heads * headDim, T(0));
    for (int t = 0; t < tokens; ++t) {
        int block = t / kPrefixBlockPageSize;
        int offset = t - block * kPrefixBlockPageSize;
        for (int h = 0; h < heads; ++h) {
            for (int d = 0; d < headDim; ++d) {
                size_t dstIndex = valueLayout ?
                    (((static_cast<size_t>(block) * heads + h) * kPrefixBlockPageSize + offset) * headDim + d) :
                    ((static_cast<size_t>(block) * kPrefixBlockPageSize + offset) * heads + h) * headDim + d;
                packed[dstIndex] = castPrefixFileScalar<T>(data[(t * heads + h) * headDim + d]);
            }
        }
    }
    std::ofstream out(path.c_str(), std::ios::binary);
    if (!out.good()) {
        return false;
    }
    out.write(reinterpret_cast<const char*>(packed.data()), packed.size() * sizeof(T));
    return out.good();
}

static bool writeCudaPrefixSegment(const std::string& root, const std::string& cacheName,
                                   const std::vector<float>& key, const std::vector<float>& value,
                                   int tokens, int kvHeads, int headDim, int bytes) {
    if (!MNN::ensurePrefixCacheObjectDirs(root, "cuda", cacheName)) {
        return false;
    }
    auto base = MNN::prefixCacheLayerBase(root, "cuda", cacheName, 0);
    if (bytes == 2) {
        return writeCudaPagedPrefixKVFile<half_float::half>(base + ".k", key, tokens, kvHeads, headDim, false) &&
               writeCudaPagedPrefixKVFile<half_float::half>(base + ".v", value, tokens, kvHeads, headDim, true);
    }
    return writeCudaPagedPrefixKVFile<float>(base + ".k", key, tokens, kvHeads, headDim, false) &&
           writeCudaPagedPrefixKVFile<float>(base + ".v", value, tokens, kvHeads, headDim, true);
}

template <typename T>
static bool writeOpenCLNativePrefixKVFile(const std::string& path, const std::vector<float>& data,
                                          int tokens, int heads, int headDim, bool valueLayout) {
    int alignedTokens = ROUND_UP(tokens, 4);
    std::vector<T> packed(static_cast<size_t>(alignedTokens) * heads * headDim, T(0));
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            for (int d = 0; d < headDim; ++d) {
                size_t dstIndex = valueLayout ?
                    ((static_cast<size_t>(h) * alignedTokens + t) * headDim + d) :
                    ((static_cast<size_t>(h) * headDim + d) * alignedTokens + t);
                packed[dstIndex] = castPrefixFileScalar<T>(data[(t * heads + h) * headDim + d]);
            }
        }
    }
    std::ofstream out(path.c_str(), std::ios::binary);
    if (!out.good()) {
        return false;
    }
    out.write(reinterpret_cast<const char*>(packed.data()), packed.size() * sizeof(T));
    return out.good();
}

static bool writeOpenCLPrefixSegment(const std::string& root, const std::string& cacheName,
                                     const std::vector<float>& key, const std::vector<float>& value,
                                     int tokens, int kvHeads, int headDim, int bytes) {
    if (!MNN::ensurePrefixCacheObjectDirs(root, "opencl", cacheName)) {
        return false;
    }
    auto base = MNN::prefixCacheLayerBase(root, "opencl", cacheName, 0);
    if (bytes == 2) {
        return writeOpenCLNativePrefixKVFile<half_float::half>(base + ".k", key, tokens, kvHeads, headDim, false) &&
               writeOpenCLNativePrefixKVFile<half_float::half>(base + ".v", value, tokens, kvHeads, headDim, true);
    }
    return writeOpenCLNativePrefixKVFile<float>(base + ".k", key, tokens, kvHeads, headDim, false) &&
           writeOpenCLNativePrefixKVFile<float>(base + ".v", value, tokens, kvHeads, headDim, true);
}

static bool buildCpuPrefixSegmentWithAttention(const std::string& root, const std::string& cacheName,
                                               const std::vector<float>& key,
                                               const std::vector<float>& value,
                                               int tokens, int kvHeads, int headDim) {
    std::vector<float> query;
    fillPrefixTestTensor(query, tokens, NumHead, headDim, 31);
    auto Query = prefixTensorToVar(query, tokens, NumHead, headDim);
    auto Key = prefixTensorToVar(key, tokens, kvHeads, headDim);
    auto Value = prefixTensorToVar(value, tokens, kvHeads, headDim);
    VARP Mask = _Input({}, NCHW, halide_type_of<float>());
    Mask->writeMap<float>()[0] = 0.0f;
    resetKVMetaForAttentionTest();
    gMeta.file_flag = KVMeta::PendingWrite;
    gMeta.file_name = cacheName;
    gMeta.layer_nums = 1;
    gMeta.layer_index = 0;
    gMeta.previous = 0;
    gMeta.add = tokens;
    // The synthetic file already stores raw canonical keys. Keep PendingWrite
    // from applying inverse RoPE while the later segment metadata marks it as canonical.
    gMeta.key_rope_state = KVMeta::KeyRopePositionEncoded;
    {
        auto writer = _makeAttentionModuleEx(MNN::OpType_Attention, 8, -1, root);
        auto outputs = writer->onForward({Query, Key, Value, Mask});
        (void)outputs;
        gMeta.sync();
    }
    auto base = MNN::prefixCacheLayerBase(root, "cpu", cacheName, 0);
    std::ifstream keyFile((base + ".k").c_str(), std::ios::binary);
    std::ifstream valueFile((base + ".v").c_str(), std::ios::binary);
    return keyFile.good() && valueFile.good();
}

static void setupPrefixReadMeta(const std::string& root, const std::string& backend,
                                const std::string& layout, int pageSize,
                                int docATokens, int docBTokens, int promptTokens,
                                int ropeDim, float ropeTheta) {
    resetKVMetaForAttentionTest();
    gMeta.file_flag = KVMeta::PendingReadSegments;
    gMeta.prefix_cache_dir = root;
    gMeta.prefix_segments.resize(2);
    gMeta.prefix_segments[0].cache_name = "doc_a";
    gMeta.prefix_segments[0].token_count = docATokens;
    gMeta.prefix_segments[1].cache_name = "doc_b";
    gMeta.prefix_segments[1].token_count = docBTokens;
    for (auto& segment : gMeta.prefix_segments) {
        segment.key_rope_state = KVMeta::KeyRopeCanonicalNoRope;
        segment.rope_dim = ropeDim;
        segment.rope_theta = ropeTheta;
        segment.rope_pairing = KVMeta::RopePairingHalf;
        segment.backend = backend;
        segment.layout = layout;
        segment.dtype = "float";
        segment.page_size = pageSize;
    }
    gMeta.segment_total_tokens = docATokens + docBTokens;
    gMeta.layer_nums = 1;
    gMeta.layer_index = 0;
    gMeta.previous = 0;
    gMeta.remove = 0;
    gMeta.add = promptTokens;
    gMeta.prefix_prompt_token_count = promptTokens;
    gMeta.prefix_request_id = 1;
}

} // namespace

class PrefixAttentionBlockTableTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        auto status = MNNTestSuite::get()->pStaus;
        const bool isCuda = status.forwardType == MNN_FORWARD_CUDA;
        const bool isCpu = status.forwardType == MNN_FORWARD_CPU;
        const bool isOpenCL = status.forwardType == MNN_FORWARD_OPENCL;
        if (!isCuda && !isCpu && !isOpenCL) {
            MNN_PRINT("PrefixAttention block-table correctness currently targets CPU/CUDA/OpenCL; skip backend=%d\n",
                      status.forwardType);
            return true;
        }

        const int docATokens = 17;
        const int docBTokens = 19;
        const int promptTokens = 1;
        const int ropeDim = HeadDim;
        const float ropeTheta = 10000.0f;
        const std::string root = ".cache/mnn_prefix_attention_block_table_test";
        std::vector<float> docAKey, docAValue, docBKey, docBValue, promptQuery, promptKey, promptValue;
        fillPrefixTestTensor(docAKey, docATokens, KvNumHead, HeadDim, 1);
        fillPrefixTestTensor(docAValue, docATokens, KvNumHead, HeadDim, 2);
        fillPrefixTestTensor(docBKey, docBTokens, KvNumHead, HeadDim, 3);
        fillPrefixTestTensor(docBValue, docBTokens, KvNumHead, HeadDim, 4);
        fillPrefixTestTensor(promptQuery, promptTokens, NumHead, HeadDim, 5);
        fillPrefixTestTensor(promptKey, promptTokens, KvNumHead, HeadDim, 6);
        fillPrefixTestTensor(promptValue, promptTokens, KvNumHead, HeadDim, 7);

        RuntimeKVBlockTablePlan plan;

        if (isCuda) {
            const int bytes = precision == MNN::BackendConfig::Precision_High ? 4 : 2;
            if (!writeCudaPrefixSegment(root, "doc_a", docAKey, docAValue, docATokens, KvNumHead, HeadDim, bytes) ||
                !writeCudaPrefixSegment(root, "doc_b", docBKey, docBValue, docBTokens, KvNumHead, HeadDim, bytes)) {
                MNN_ERROR("Failed to write CUDA paged prefix KV test files under %s\n", root.c_str());
                return false;
            }
            setupPrefixReadMeta(root, "cuda", "cuda-paged-canonical-no-rope-v1", kPrefixBlockPageSize,
                                docATokens, docBTokens, promptTokens, ropeDim, ropeTheta);
        } else if (isOpenCL) {
            const int bytes = precision == MNN::BackendConfig::Precision_High ? 4 : 2;
            if (!writeOpenCLPrefixSegment(root, "doc_a", docAKey, docAValue, docATokens, KvNumHead, HeadDim, bytes) ||
                !writeOpenCLPrefixSegment(root, "doc_b", docBKey, docBValue, docBTokens, KvNumHead, HeadDim, bytes)) {
                MNN_ERROR("Failed to write OpenCL native prefix KV test files under %s\n", root.c_str());
                return false;
            }
            setupPrefixReadMeta(root, "opencl", "opencl-buffer-canonical-no-rope-v1", kPrefixBlockPageSize,
                                docATokens, docBTokens, promptTokens, ropeDim, ropeTheta);
        } else {
            if (!buildCpuPrefixSegmentWithAttention(root, "doc_a", docAKey, docAValue, docATokens, KvNumHead, HeadDim) ||
                !buildCpuPrefixSegmentWithAttention(root, "doc_b", docBKey, docBValue, docBTokens, KvNumHead, HeadDim)) {
                MNN_ERROR("Failed to write CPU prefix KV test files under %s\n", root.c_str());
                return false;
            }
            setupPrefixReadMeta(root, "cpu", "cpu-flash-packed-canonical-no-rope-v1", 0,
                                docATokens, docBTokens, promptTokens, ropeDim, ropeTheta);
        }

        if (!buildPrefixRuntimeKVBlockTablePlan(plan, &gMeta, promptTokens, kPrefixBlockPageSize)) {
            MNN_ERROR("Failed to build PrefixAttention block-table plan\n");
            return false;
        }
        if (plan.refs().size() != 3 ||
            plan.refs()[0].physicalTokenStart != 0 ||
            plan.refs()[1].physicalTokenStart != kPrefixBlockPageSize ||
            plan.refs()[2].physicalTokenStart != 2 * kPrefixBlockPageSize) {
            MNN_ERROR("PrefixAttention block isolation mismatch: refs=%zu docA_physical=%d docB_physical=%d prompt_physical=%d\n",
                      plan.refs().size(),
                      plan.refs().size() > 0 ? plan.refs()[0].physicalTokenStart : -1,
                      plan.refs().size() > 1 ? plan.refs()[1].physicalTokenStart : -1,
                      plan.refs().size() > 2 ? plan.refs()[2].physicalTokenStart : -1);
            return false;
        }

        auto expected = computePrefixReference(promptQuery, docAKey, docAValue, docBKey, docBValue,
                                               promptKey, promptValue,
                                               docATokens, docBTokens, promptTokens,
                                               NumHead, KvNumHead, HeadDim, ropeDim, ropeTheta);
        auto Query = prefixTensorToVar(promptQuery, promptTokens, NumHead, HeadDim);
        auto Key = prefixTensorToVar(promptKey, promptTokens, KvNumHead, HeadDim);
        auto Value = prefixTensorToVar(promptValue, promptTokens, KvNumHead, HeadDim);
        VARP Mask = _Input({}, NCHW, halide_type_of<float>());
        Mask->writeMap<float>()[0] = 0.0f;

        auto ordinaryAttention = _makeAttentionModuleEx(MNN::OpType_Attention, 8, -1, root);
        auto rejectedOutputs = ordinaryAttention->onForward({Query, Key, Value, Mask});
        if (!rejectedOutputs.empty()) {
            MNN_ERROR("Ordinary Attention unexpectedly accepted PendingReadSegments on backend=%d\n",
                      status.forwardType);
            return false;
        }

        auto prefixAttention = _makeAttentionModuleEx(MNN::OpType_PrefixAttention, 8, 0, root);
        auto outputs = prefixAttention->onForward({Query, Key, Value, Mask});
        if (outputs.empty() || outputs[0] == nullptr) {
            MNN_ERROR("PrefixAttention block-table test produced no output\n");
            return false;
        }
        auto output = outputs[0];
        const float* resultPtr = output->readMap<float>();
        if (resultPtr == nullptr) {
            MNN_ERROR("PrefixAttention block-table test cannot map output\n");
            return false;
        }
        float maxDiff = 0.0f;
        int maxIndex = -1;
        for (int i = 0; i < promptTokens * NumHead * HeadDim; ++i) {
            float diff = std::fabs(resultPtr[i] - expected[i]);
            if (diff > maxDiff) {
                maxDiff = diff;
                maxIndex = i;
            }
        }
        output->unMap();
        const float tolerance = precision == MNN::BackendConfig::Precision_High ? 0.01f : 0.08f;
        if (maxDiff > tolerance) {
            MNN_ERROR("PrefixAttention block-table dense reference mismatch: backend=%d max_diff=%f max_index=%d "
                      "docA_logical=0 docA_physical=0 docB_logical=%d docB_physical=%d prompt_logical=%d prompt_physical=%d page_size=%d\n",
                      status.forwardType, maxDiff, maxIndex,
                      docATokens, kPrefixBlockPageSize,
                      docATokens + docBTokens, 2 * kPrefixBlockPageSize,
                      kPrefixBlockPageSize);
            return false;
        }
        MNN_PRINT("PrefixAttention block-table correctness passed: backend=%d max_diff=%f page_size=%d "
                  "docA_tokens=%d docB_tokens=%d prompt_tokens=%d physical_total=%d\n",
                  status.forwardType, maxDiff, kPrefixBlockPageSize,
                  docATokens, docBTokens, promptTokens, plan.physicalLength());
        resetKVMetaForAttentionTest();
        return true;
    }
};

MNNTestSuiteRegister(AttentionTest, "op/attention");
MNNTestSuiteRegister(PrefixAttentionBlockTableTest, "op/prefix_attention_block_table");
MNNTestSuiteRegister(SpeedAttentionTest, "speed/attention");
#endif
