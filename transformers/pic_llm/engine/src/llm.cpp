//
//  llm.cpp
//
//  Created by MNN on 2023/08/25.
//  ZhaodeWang
//
// #define MNN_OPEN_TIME_TRACE 1

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>

#include "prompt_cache_utils.hpp"
#include <MNN/AutoTime.hpp>
#include "core/PagedKVMeta.hpp"
#include "core/TensorUtils.hpp"
#include "cpp/ExprDebug.hpp"
#include "MNN_generated.h"
#include "llm/llm.hpp"
#include "kvmeta.hpp"
#include "llmconfig.hpp"
#include "tokenizer/tokenizer.hpp"
#include "diskembedding.hpp"
#include "sampler.hpp"
#include "omni.hpp"
#include "speculative_decoding/generate.hpp"
#include "core/MNNFileUtils.h"

// 0: no debug, 1: test op time, 2: print tensor info, 3: print tensor in output
#define DEBUG_MODE 0
//#define DEBUG_IMAGE

namespace MNN {
using namespace Express;
namespace Transformer {

static bool picRequestProfileEnabled() {
    const char* value = std::getenv("MNN_PIC_REQUEST_PROFILE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static int64_t picRequestMonotonicUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static void picRequestProfileLog(const char* stage, int64_t elapsedUs, const std::string& detail = "") {
    if (!picRequestProfileEnabled()) {
        return;
    }
    std::fprintf(stderr, "MNN_PIC_REQUEST_PROFILE stage=%s cost_ms=%.3f", stage, elapsedUs / 1000.0);
    if (!detail.empty()) {
        std::fprintf(stderr, " %s", detail.c_str());
    }
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

static bool picDecodeRepairProfileEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("MNN_PIC_DECODE_REPAIR_PROFILE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

static void picDecodeRepairProfileLog(int step, const char* selector, int normalLogical, int repairRows,
                                      int sparseRows, int nextRepairRows, int attentionRankCount,
                                      int attentionRankSourceStep, int64_t buildRowsUs,
                                      int64_t beginRecomputeUs, int64_t embeddingUs,
                                      int64_t maskPosUs, int64_t forwardRawUs,
                                      int64_t validateUs, int64_t rankResultUs,
                                      int64_t finishSparseUs, int64_t bookkeepingUs,
                                      int64_t totalUs) {
    if (!picDecodeRepairProfileEnabled()) {
        return;
    }
    std::fprintf(stderr,
                 "MNN_PIC_DECODE_REPAIR_PROFILE step=%d selector=%s normal_logical=%d repair_rows=%d "
                 "sparse_rows=%d next_repair_rows=%d attention_rank_count=%d attention_rank_source_step=%d "
                 "build_rows_ms=%.3f begin_recompute_ms=%.3f embedding_ms=%.3f mask_pos_ms=%.3f "
                 "forward_raw_ms=%.3f validate_ms=%.3f rank_result_ms=%.3f finish_sparse_ms=%.3f "
                 "bookkeeping_ms=%.3f total_ms=%.3f\n",
                 step, selector != nullptr ? selector : "", normalLogical, repairRows, sparseRows,
                 nextRepairRows, attentionRankCount, attentionRankSourceStep,
                 buildRowsUs / 1000.0, beginRecomputeUs / 1000.0, embeddingUs / 1000.0,
                 maskPosUs / 1000.0, forwardRawUs / 1000.0, validateUs / 1000.0,
                 rankResultUs / 1000.0, finishSparseUs / 1000.0, bookkeepingUs / 1000.0,
                 totalUs / 1000.0);
    std::fflush(stderr);
}

static void clearGenerateForwardState(const std::shared_ptr<GenerationParams>& params) {
    if (params == nullptr) {
        return;
    }
    params->input_embeds = nullptr;
    params->outputs.clear();
    params->validLogitSize = 0;
    params->validLogitStart = 0;
}

static MNNForwardType backend_type_convert(const std::string& type_str) {
    if (type_str == "cpu")
        return MNN_FORWARD_CPU;
    if (type_str == "metal")
        return MNN_FORWARD_METAL;
    if (type_str == "cuda")
        return MNN_FORWARD_CUDA;
    if (type_str == "opencl")
        return MNN_FORWARD_OPENCL;
    if (type_str == "opengl")
        return MNN_FORWARD_OPENGL;
    if (type_str == "vulkan")
        return MNN_FORWARD_VULKAN;
    if (type_str == "npu")
        return MNN_FORWARD_NN;
    return MNN_FORWARD_AUTO;
}

static int apply_opencl_tune_level_override(int numThread, MNNForwardType backendType) {
    if (backendType != MNN_FORWARD_OPENCL) {
        return numThread;
    }
    const char* envValue = std::getenv("MNN_OPENCL_TUNE_LEVEL");
    if (envValue == nullptr || envValue[0] == '\0') {
        return numThread;
    }
    std::string level(envValue);
    std::transform(level.begin(), level.end(), level.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const int tuneMask = MNN_GPU_TUNING_NONE | MNN_GPU_TUNING_FAST | MNN_GPU_TUNING_NORMAL |
        MNN_GPU_TUNING_HEAVY | MNN_GPU_TUNING_WIDE;
    int tuneFlag = 0;
    if (level == "none") {
        tuneFlag = MNN_GPU_TUNING_NONE;
    } else if (level == "fast") {
        tuneFlag = MNN_GPU_TUNING_FAST;
    } else if (level == "normal") {
        tuneFlag = MNN_GPU_TUNING_NORMAL;
    } else if (level == "heavy") {
        tuneFlag = MNN_GPU_TUNING_HEAVY;
    } else if (level == "wide") {
        tuneFlag = MNN_GPU_TUNING_WIDE;
    } else {
        return numThread;
    }
    return (numThread & ~tuneMask) | tuneFlag;
}

template <typename T>
static inline VARP _var(std::vector<T> vec, const std::vector<int> &dims) {
    return _Const(vec.data(), dims, NHWC, halide_type_of<T>());
}

static inline int _picRecomputeBudgetValue(PagedKVMeta* paged, int seqLen) {
    int budget = std::max(0, seqLen);
    if (paged != nullptr && paged->sparse_query_active && !paged->sparse_query_logical_indices.empty()) {
        budget = static_cast<int>(paged->sparse_query_logical_indices.size());
    } else if (paged != nullptr) {
        budget = paged->graphActiveBudget(seqLen);
    }
    return budget;
}

static inline VARP _picRecomputeBudgetVar(PagedKVMeta* paged, int seqLen, bool enabled) {
    if (!enabled) {
        return nullptr;
    }
    const int budget = _picRecomputeBudgetValue(paged, seqLen);
    return _var<int>(std::vector<int>{budget}, {1});
}

Llm* Llm::createLLM(const std::string& config_path) {
    std::shared_ptr<LlmConfig> config(new LlmConfig(config_path));
    Llm* llm = nullptr;
    if (config->is_visual() || config->is_audio() || config->has_talker()) {
        llm = new Omni(config);
    } else {
        llm = new Llm(config);
    }
    return llm;
}
void Llm::destroy(Llm* llm) {
    delete llm;
}

std::string Llm::dump_config() {
    return mConfig->config_.dump();
}

void Llm::setChatTemplate() {
    if (!mTokenizer || !mConfig->config_.contains("jinja")) return;
    auto jinja = mConfig->config_["jinja"];
    if (jinja.contains("chat_template")) {
        std::string context;
        if (jinja.contains("context")) {
            context = jinja["context"].dump();
        }
        mTokenizer->set_chat_template(jinja["chat_template"].get<std::string>(), jinja.value("eos", ""), context);
    }
}

bool Llm::set_config(const std::string& content) {
    mConfig->config_.merge(ujson::json::parse(content));
    setChatTemplate();
    mAsync = mConfig->config_.value("async", true);
    mGenerateParam->timeout_ms = mConfig->timeout_ms();
    if (mConfig->paged_attention()) {
        auto paged = static_cast<PagedKVMeta*>(mMeta.get());
        if (paged != nullptr && !paged->request_active) {
            paged->max_tokens = mConfig->paged_kv_max_tokens();
            paged->full_causal_attention_mask =
                mConfig->attention_mask() == "float" && mConfig->attention_type() == "full";
        }
    }
    if (mSampler != nullptr) {
        mSampler.reset(Sampler::createSampler(mContext, mConfig));
    }
    mValidBlockSize.clear();
    mBlockSize = mConfig->config_.value("chunk", 0);
    if (mConfig->config_.contains("chunk_limits")) {
        auto size_limit = mConfig->config_["chunk_limits"];
        do {
            if (!size_limit.is_array()) {
                MNN_ERROR("size_limit must be array, eg: [128, 1]\n");
                break;
            }
            for (size_t i = 0; i < size_limit.size(); i++) {
                mValidBlockSize.emplace_back(size_limit[i].get<int>());
            }
            if (mValidBlockSize.size() < 2) {
                MNN_ERROR("size_limit must be array larger than 1, eg: [128, 1]\n");
                mValidBlockSize.clear();
                break;
            }
            std::sort(mValidBlockSize.begin(), mValidBlockSize.end());
            mBlockSize = mValidBlockSize[mValidBlockSize.size()-1];
        } while (false);
    }
    return true;
}

void Llm::setDebugCallback(MNN::TensorCallBackWithInfo&& before, MNN::TensorCallBackWithInfo&& after) {
    mExecutor->setCallBack(std::move(before), std::move(after));
}

void Llm::setRuntimeHint(std::shared_ptr<Express::Executor::RuntimeManager> &rtg) {
    rtg->setHint(MNN::Interpreter::INIT_THREAD_NUMBER, 4);

    rtg->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);

    /* 'quant_qkv' is deprecated, use 'attention_mode '*/
    int legacyAttentionMode = mConfig->config_.value("quant_qkv", 8); // compatibility
    int attentionMode = mConfig->config_.value("attention_mode", legacyAttentionMode); // try to read 'attention_mode'

    // 3. 设置 Hint
    rtg->setHint(MNN::Interpreter::ATTENTION_OPTION, attentionMode);
    if (mConfig->reuse_kv() && attentionMode == 10) {
        rtg->setHint(MNN::Interpreter::ATTENTION_OPTION, 9);
    }
    if (mConfig->use_cached_mmap()) {
        rtg->setHint(MNN::Interpreter::USE_CACHED_MMAP, 1);
    }
    std::string tmpPath = mConfig->tmp_path();
    if (mConfig->kvcache_mmap()) {
        rtg->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_PATH_KVCACHE_DIR);
    }
    auto cachePath = mConfig->prefix_cache_path();
    rtg->setExternalPath(cachePath, MNN::Interpreter::EXTERNAL_PATH_PREFIXCACHE_DIR);
    if (mConfig->use_mmap()) {
        rtg->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_WEIGHT_DIR);
    }
    // set npu model dir
    rtg->setExternalPath(mConfig->npu_model_dir(), MNN::Interpreter::EXTERNAL_NPU_FILE_DIR);
    rtg->setHint(MNN::Interpreter::DYNAMIC_QUANT_OPTIONS, mConfig->config_.value("dynamic_option", 0));

    rtg->setHintPtr(Interpreter::KVCACHE_INFO, mMeta.get());
    if (backend_type_convert(mConfig->backend_type()) != 0) { // not cpu
        std::string cacheFilePath = tmpPath.length() != 0 ? tmpPath : ".";
        rtg->setCache(cacheFilePath + "/mnn_cachefile.bin");
    }
    rtg->setHint(MNN::Interpreter::CPU_SME2_NEON_DIVISION_RATIO, mConfig->config_.value("cpu_sme2_neon_division_ratio", 41));
    rtg->setHint(MNN::Interpreter::CPU_SME_CORES, mConfig->config_.value("cpu_sme_core_num", 2));
    rtg->setHint(MNN::Interpreter::MMAP_FILE_SIZE, mConfig->mmap_size());
}

std::shared_ptr<Express::Executor::RuntimeManager> Llm::createRuntimeManagerForCurrentConfig() {
    ScheduleConfig config;
    BackendConfig cpuBackendConfig;
    config.type      = backend_type_convert(mConfig->backend_type());
    config.numThread = mConfig->thread_num();
    if(config.type == 3){
        // opencl need set numThread = 64(buffer mode)
        config.numThread |= 64;
    }
    config.numThread = apply_opencl_tune_level_override(config.numThread, config.type);
    if (mConfig->power() == "high") {
        cpuBackendConfig.power = BackendConfig::Power_High;
    } else if (mConfig->power() == "low") {
        cpuBackendConfig.power = BackendConfig::Power_Low;
    }
    if (mConfig->memory() == "high") {
        cpuBackendConfig.memory = BackendConfig::Memory_High;
    } else if (mConfig->memory() == "low") {
        cpuBackendConfig.memory = BackendConfig::Memory_Low;
    }
    if (mConfig->precision() == "high") {
        cpuBackendConfig.precision = BackendConfig::Precision_High;
    } else if (mConfig->precision() == "low") {
        cpuBackendConfig.precision = BackendConfig::Precision_Low;
    }
    config.backendConfig = &cpuBackendConfig;

    std::shared_ptr<Express::Executor::RuntimeManager> runtimeManager(
        Executor::RuntimeManager::createRuntimeManager(config));
    if (runtimeManager == nullptr) {
        return nullptr;
    }
    setRuntimeHint(runtimeManager);

#if DEBUG_MODE == 1
    runtimeManager->setMode(MNN::Interpreter::Session_Debug);
    _initTimeTrace();
#endif
#if DEBUG_MODE == 2
    runtimeManager->setMode(MNN::Interpreter::Session_Debug);
    _initTensorStatic();
#endif
#if DEBUG_MODE == 3
    runtimeManager->setMode(MNN::Interpreter::Session_Debug);
    _initDebug();
#endif
    // get linear input thresholds and max values
    if (mConfig->config_.value("enable_debug", false)) {
        runtimeManager->setMode(MNN::Interpreter::Session_Debug);
    }
    return runtimeManager;
}

void Llm::initRuntime() {
    mRuntimeManager = createRuntimeManagerForCurrentConfig();
}

static bool canSpecDecode(std::shared_ptr<Express::Module> module) {
    bool canSpec = false;
    auto info = module->getInfo();
    // check from mnn model
    for (int i=0; i<info->inputNames.size(); ++i) {
        auto& varInfo = info->inputs[i];
        if(info->inputNames[i] == "logits_index") {
            if (varInfo.dim.size() > 0) {
                canSpec = true;
            }
        }
    }
    return canSpec;
}
void Llm::setSpeculativeConfig() {
    auto specultive_type = mConfig->speculative_type();
    if(!specultive_type.empty()) {
        if(!canSpecDecode(mModule)) {
            mInSpec = false;
            return;
        }
        mDraftLength = mConfig->draft_predict_length();
        mInSpec = true;
    }
}

static bool checkFile(const std::string& path, const char* name) {
    if (!MNNFileExist(path.c_str())) {
        MNN_ERROR("[Error]: %s not found: %s\n", name, path.c_str());
        return false;
    }
    std::ifstream f(path);
    if (!f.is_open()) {
        MNN_ERROR("[Error]: Failed to open %s (permission denied?): %s\n", name, path.c_str());
        return false;
    }
    return true;
}

static std::string findPagedAttentionOutputName(const std::string& modelPath, int layerIndex) {
    if (layerIndex < 0) {
        return "";
    }
    std::ifstream input(modelPath, std::ios::binary | std::ios::ate);
    if (!input.is_open()) {
        MNN_ERROR("Failed to open MNN model for cacheblend score-layer scan: %s\n", modelPath.c_str());
        return "";
    }
    auto fileSize = input.tellg();
    if (fileSize <= 0) {
        return "";
    }
    std::vector<char> buffer(static_cast<size_t>(fileSize));
    input.seekg(0, std::ios::beg);
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    if (!input.good()) {
        return "";
    }
    auto net = flatbuffers::GetRoot<MNN::Net>(buffer.data());
    if (net == nullptr || net->oplists() == nullptr || net->tensorName() == nullptr) {
        return "";
    }
    auto ops = net->oplists();
    auto tensorNames = net->tensorName();
    for (int i = 0; i < ops->size(); ++i) {
        auto op = ops->GetAs<MNN::Op>(i);
        if (op == nullptr ||
            (op->type() != MNN::OpType_PagedAttention && op->type() != MNN::OpType_PicScoreAttention)) {
            continue;
        }
        auto param = op->main_as_AttentionParam();
        if (param == nullptr || param->layer_index() != layerIndex) {
            continue;
        }
        auto outputs = op->outputIndexes();
        if (outputs == nullptr || outputs->size() <= 0) {
            return "";
        }
        int outputIndex = outputs->Get(0);
        if (outputIndex < 0 || outputIndex >= tensorNames->size()) {
            return "";
        }
        auto name = tensorNames->GetAsString(outputIndex);
        return name == nullptr ? "" : name->str();
    }
    return "";
}

bool Llm::load() {
    // check required files before loading
    std::string tokenizer_path = mConfig->tokenizer_file();
    std::string model_path = mConfig->llm_model();
    std::string weight_path = mConfig->llm_weight();
    std::string decode_model_path = mConfig->has_llm_decode_model() ? mConfig->llm_decode_model() : "";
    std::string decode_weight_path = mConfig->has_llm_decode_model() ? mConfig->llm_decode_weight() : "";
    if (!checkFile(tokenizer_path, "tokenizer file") ||
        !checkFile(model_path, "LLM model file") ||
        !checkFile(weight_path, "LLM weight file")) {
        return false;
    }
    if (mConfig->has_llm_decode_model() &&
        (!checkFile(decode_model_path, "LLM decode model file") ||
         !checkFile(decode_weight_path, "LLM decode weight file"))) {
        return false;
    }
    MNN::Express::ExecutorScope s(mExecutor);
    Timer _t;
    initRuntime();
    if (mRuntimeManager == nullptr) {
        MNN_ERROR("[Error]: Failed to initialize runtime manager.\n");
        return false;
    }
    // init module status
    // 1. load vocab
    mTokenizer.reset(Tokenizer::createTokenizer(tokenizer_path));
    if (mTokenizer == nullptr) {
        MNN_ERROR("[Error]: Failed to load tokenizer from: %s\n", tokenizer_path.c_str());
        return false;
    }
    // 2. load context
    {
        std::ifstream contextFile(mConfig->context_file());
        if (contextFile.is_open()) {
            std::ostringstream contextStream;
            contextStream << contextFile.rdbuf();
            auto contextStr = contextStream.str();
            // check valid json
            auto contextJson = ujson::json::parse(contextStr);
            if (!contextJson.is_null()) {
                std::string config_json = R"({
                    "jinja": {
                        "context": )" + contextStr + R"(
                    }
                })";
                mConfig->config_.merge(ujson::json::parse(config_json));
            }
        }
    }
    mDiskEmbedding.reset(new DiskEmbedding(mConfig));
    // PLE (Per-Layer Embeddings) for gemma4
    if (mConfig->has_ple()) {
        mPleEmbedding.reset(new DiskEmbedding(mConfig, mConfig->ple_embed_file(), mConfig->ple_embed_dim(), mConfig->ple_quant()));
    }
    setChatTemplate();
    mSampler.reset(Sampler::createSampler(mContext, mConfig));
    // 3. load model
    Module::Config module_config;
    if (mConfig->backend_type() == "opencl" || mConfig->backend_type() == "vulkan" || mConfig->backend_type() == "npu") {
        module_config.shapeMutable = false;
    } else {
        module_config.shapeMutable = true;
    }
    module_config.rearrange    = true;
    // using base module for lora module
    if (mBaseModule != nullptr) {
        module_config.base = mBaseModule;
    }
    // load single model
    std::vector<std::string> inputNames {"input_ids", "attention_mask", "position_ids", "logits_index"};
    if (mConfig->has_pic_recompute_budget()) {
        inputNames.emplace_back("pic_recompute_budget");
    }
    std::vector<std::string> outputNames {"logits"};
    if (mConfig->has_talker()) {
        outputNames.emplace_back("talker_embeds");
    }
    bool needHiddenState = mConfig->config_.value("hidden_states", false);
    if(mConfig->speculative_type() == "mtp") {
        needHiddenState = true;
    }
    if (needHiddenState) {
        outputNames.emplace_back("hidden_states");
    }
    mRuntimeManager->setExternalFile(weight_path);
    if (mConfig->has_deepstack()) {
        inputNames.emplace_back("deepstack_embeds");
    }
    if (mConfig->has_ple()) {
        inputNames.emplace_back("ple_embeddings");
    }
    mModule.reset(Module::load(inputNames, outputNames, model_path.c_str(), mRuntimeManager, &module_config));
    mRuntimeManager->setExternalFile("");
    if(nullptr == mModule) {
        MNN_ERROR("[Error]: Load module failed, please check model.\n");
        if(outputNames.size() > 1) {
            MNN_ERROR("[Warning]: Set module multi outputs, please double check.\n");
        }
        return false;
    }
    if (mConfig->has_llm_decode_model()) {
        std::vector<std::string> decodeInputNames {"input_ids", "attention_mask", "position_ids", "logits_index"};
        if (mConfig->has_deepstack()) {
            decodeInputNames.emplace_back("deepstack_embeds");
        }
        if (mConfig->has_ple()) {
            decodeInputNames.emplace_back("ple_embeddings");
        }
        mRuntimeManager->setExternalFile(decode_weight_path);
        mDecodeModule.reset(Module::load(decodeInputNames, outputNames, decode_model_path.c_str(),
                                         mRuntimeManager, &module_config));
        mRuntimeManager->setExternalFile("");
        if(nullptr == mDecodeModule) {
            MNN_ERROR("[Error]: Load decode module failed, please check model: %s\n", decode_model_path.c_str());
            return false;
        }
        if (std::getenv("MNN_PIC_DECODE_DEBUG") != nullptr) {
            std::fprintf(stderr,
                         "PIC decode graph loaded model=%s weight=%s shared=%d inputs=%d outputs=%d\n",
                         decode_model_path.c_str(), decode_weight_path.c_str(),
                         mConfig->llm_decode_shared_weight() ? 1 : 0,
                         static_cast<int>(decodeInputNames.size()),
                         static_cast<int>(outputNames.size()));
            std::fflush(stderr);
        }
    }
    // set speculative decoding params
    setSpeculativeConfig();
    // create generation strategy
    mGenerationStrategy = GenerationStrategyFactory::create(this, mContext, mConfig, mInSpec);

    int decode_type_num = 1;
    int verify_length = 1;
    if(mInSpec) {
        // decode one token or mDraftLength token
        decode_type_num = 2;
        verify_length = mDraftLength + 1;
        // speculative decode module
        mModulePool[std::make_pair(verify_length, true)] = cloneModuleWithRuntime(mModule.get());
    }

    // autoregressive decode module
    mModulePool[std::make_pair(1, false)] = cloneModuleWithRuntime(mModule.get());
    if (mDecodeModule != nullptr) {
        mDecodeModulePool[std::make_pair(1, false)] = cloneModuleWithRuntime(mDecodeModule.get());
    }
    // prefill module
    mModulePool[std::make_pair(mPrefillKey, mConfig->all_logits())] = mModule;

    // module input varp setting
    logitsLastIdx = _var<int>({-1}, {1});
    logitsAllIdx = _var<int>({0}, {1});
    // index match with seq_len
    mAttentionMaskVarVec.resize(decode_type_num);
    mPositionIdsVarVec.resize(decode_type_num);
    for(int i = 0; i < decode_type_num; i++) {
        int index = 1;
        if(i > 0) {
            index = verify_length;
        }
        // attentiion mask var
        {
            // Mask: lower triangular
           if (mConfig->backend_type() == "cpu" && mValidBlockSize.empty() && !mConfig->has_pic_recompute_budget()) {
               mAttentionMaskVarVec[i] = _Input({}, NCHW, halide_type_of<float>());
               auto ptr = mAttentionMaskVarVec[i]->writeMap<float>();
               ptr[0] = 0;
           } else {
                mAttentionMaskVarVec[i] = _Input({1, 1, index, index}, NCHW, halide_type_of<float>());
                auto ptr = mAttentionMaskVarVec[i]->writeMap<float>();
                for (int i = 0; i < index; i++) {
                    for (int j = 0; j < index; j++) {
                        ptr[index * i + j] = (j > i) * std::numeric_limits<float>::lowest();
                    }
                }
           }
        }

        if (mConfig->is_mrope()) {
            mPositionIdsVarVec[i] = _Input({3, index}, NCHW, halide_type_of<int>());
        } else {
            mPositionIdsVarVec[i] = _Input({1, index}, NCHW, halide_type_of<int>());
        }
    }

    // MTP model load
    mGenerationStrategy->load(module_config);
    mContext->load_us += _t.durationInUs();
    mContext->status = LlmStatus::RUNNING;  // Set status to RUNNING after successful load
    return true;
}

Llm* Llm::create_lora(const std::string& lora_path) {
    auto llm = new Llm(std::make_shared<LlmConfig>(*mConfig));
    llm->set_config("{\"llm_model\": \"" + lora_path + "\", \"use_mmap\": false, \"use_cached_mmap\": false}");
    llm->mBaseModule = mModule.get();
    auto res = llm->load();
    if (!res) {
        MNN_ERROR("[MNN:LLM] Load Lora error\n");
        delete llm;
        return nullptr;
    }
    return llm;
}

void Llm::tuning(TuneType type, std::vector<int> candidates) {
    CHECK_LLM_RUNNING(mContext);
    MNN::Express::ExecutorScope s(mExecutor);
    if (type != OP_ENCODER_NUMBER) {
        MNN_ERROR("tuning type not supported\n");
        return;
    }
    // FIXME: Currently OpenCL Don't support KVMeta
    if (mConfig->backend_type() == "opencl") {
        return;
    }
    auto finishTuning = [&]() {
        setKVCacheInfo(0, getCurrentHistory());
        reset();
    };
    beginPagedRequestIfNeeded();
    int decode_seq = 1;
    // Set to decode mode
    mContext->gen_seq_len = 1;
    if(mInSpec) {
        // start autoregressive decoding
        std::vector<int> input_ids = {0};
        auto logits = forwardVec(input_ids);
        if(logits.empty()) {
            finishTuning();
            return;
        }
        int verify_length = mDraftLength + 1;
        decode_seq = verify_length;
    }
    int64_t min_time     = INT64_MAX;
    int prefer_candidate = 10;
    for (auto& candidate : candidates) {
        mRuntimeManager->setHint(MNN::Interpreter::OP_ENCODER_NUMBER_FOR_COMMIT, candidate);
        Timer _t;
        std::vector<int> input_ids(decode_seq, 0);
        auto outputs = forwardVec(input_ids);
        if(outputs.empty()) {
            finishTuning();
            return;
        }
        auto logits = outputs[0];
        if (nullptr == logits.get()) {
            finishTuning();
            return;
        }
        if (logits->getInfo()->size == 0) {
            finishTuning();
            return;
        }
        auto token   = sample(logits);
        auto time = _t.durationInUs();
        if (time < min_time) {
            prefer_candidate = candidate;
            min_time         = time;
            // MNN_PRINT("op encode number:%d, decode time: %lld us\n", candidate, time);
        }
    }
    mRuntimeManager->setHint(MNN::Interpreter::OP_ENCODER_NUMBER_FOR_COMMIT, prefer_candidate);
    // clear dirty tuning kv history
    finishTuning();
}

void Llm::updateRuntimeCache() {
    if (mRuntimeManager != nullptr) {
        mRuntimeManager->updateCache();
    }
    if (mCacheBlendScoreRuntimeManager != nullptr) {
        mCacheBlendScoreRuntimeManager->updateCache();
    }
}

void Llm::switchMode(Llm::Stage stage) {
    // do nothing, only reserve api
    return;
}

void Llm::setKVCacheInfo(size_t add, size_t remove, int* reserve, int n_reserve) {
    if (remove > mMeta->previous) {
        remove = mMeta->previous;
    }

    mMeta->remove = remove;
    mMeta->reserve = reserve;
    mMeta->n_reserve = n_reserve;
    mMeta->add = add;
}

bool Llm::beginExternalPagedKVRequest() {
    return beginPagedRequestIfNeeded() || (mConfig->paged_attention() && static_cast<PagedKVMeta*>(mMeta.get())->request_active);
}

bool Llm::reserveExternalPagedKVSourceSlots(size_t token_count) {
    if (token_count == 0) {
        return true;
    }
    if (!mConfig->paged_attention()) {
        MNN_ERROR("Persistent PIC cache source slot reservation requires paged_attention=true\n");
        return false;
    }
    auto paged = static_cast<PagedKVMeta*>(mMeta.get());
    if (paged == nullptr) {
        return false;
    }
    if (!paged->request_active) {
        paged->beginRequest(mConfig->paged_kv_max_tokens());
    }
    return paged->reserveExternalSourceSlots(token_count);
}

bool Llm::appendExternalPagedKV(const std::vector<int>& token_ids,
                                const std::vector<MNN::PagedKVExternalSegment>& segments) {
    if (!mConfig->paged_attention()) {
        MNN_ERROR("Persistent PIC cache source binding requires paged_attention=true\n");
        return false;
    }
    auto paged = static_cast<PagedKVMeta*>(mMeta.get());
    if (paged == nullptr) {
        return false;
    }
    if (!paged->request_active) {
        paged->beginRequest(mConfig->paged_kv_max_tokens());
    }
    size_t segmentTokens = 0;
    for (const auto& segment : segments) {
        segmentTokens += segment.tokenCount;
    }
    if (segmentTokens != token_ids.size()) {
        MNN_ERROR("Persistent PIC cache source token mismatch, token_ids=%d segment_tokens=%d\n",
                  static_cast<int>(token_ids.size()), static_cast<int>(segmentTokens));
        return false;
    }
    if (!paged->appendExternalSegments(segments, token_ids.size())) {
        MNN_ERROR("Persistent PIC cache source exceeds request PagedCache capacity, current=%d add=%d capacity=%d\n",
                  paged->logical_length, static_cast<int>(token_ids.size()), paged->request_capacity);
        return false;
    }
    mContext->all_seq_len += static_cast<int>(token_ids.size());
    mContext->history_tokens.insert(mContext->history_tokens.end(), token_ids.begin(), token_ids.end());
    return true;
}

bool Llm::recomputeExternalPagedKV(const std::vector<int>& logical_indices, const std::vector<int>& token_ids,
                                   int sparse_start_layer_idx) {
    if (!mConfig->paged_attention()) {
        MNN_ERROR("Sparse PIC recompute requires paged_attention=true\n");
        return false;
    }
    if (logical_indices.empty()) {
        return true;
    }
    if (logical_indices.size() != token_ids.size()) {
        MNN_ERROR("Sparse PIC recompute mismatch, logical_indices=%d token_ids=%d\n",
                  static_cast<int>(logical_indices.size()), static_cast<int>(token_ids.size()));
        return false;
    }
    auto paged = static_cast<PagedKVMeta*>(mMeta.get());
    if (paged == nullptr || !paged->beginSparseQuery(logical_indices, sparse_start_layer_idx)) {
        MNN_ERROR("Sparse PIC recompute failed to bind logical indices\n");
        return false;
    }
    auto oldStatus = mContext->status;
    auto outputs = forwardVec(token_ids);
    paged->finishSparseQuery();
    if (outputs.empty()) {
        return false;
    }
    if (oldStatus == LlmStatus::RUNNING && mContext->status != LlmStatus::INTERNAL_ERROR &&
        mContext->status != LlmStatus::TIMEOUT && mContext->status != LlmStatus::USER_CANCEL) {
        mContext->status = oldStatus;
    }
    return true;
}

std::shared_ptr<Module> Llm::getCacheBlendScoreModule(int scoreLayerIdx) {
    auto iter = mCacheBlendScoreModulePool.find(scoreLayerIdx);
    if (iter != mCacheBlendScoreModulePool.end()) {
        return iter->second;
    }
    auto scoreOutputName = findPagedAttentionOutputName(mConfig->llm_model(), scoreLayerIdx);
    if (scoreOutputName.empty()) {
        MNN_ERROR("Failed to find PagedAttention output for cacheblend score layer %d\n", scoreLayerIdx);
        return nullptr;
    }
    Module::Config moduleConfig;
    if (mConfig->backend_type() == "opencl" || mConfig->backend_type() == "vulkan" ||
        mConfig->backend_type() == "npu") {
        moduleConfig.shapeMutable = false;
    } else {
        moduleConfig.shapeMutable = true;
    }
    moduleConfig.rearrange = true;
    if (mBaseModule != nullptr) {
        moduleConfig.base = mBaseModule;
    }
    std::vector<std::string> inputNames {"input_ids", "attention_mask", "position_ids", "logits_index"};
    if (mConfig->has_pic_recompute_budget()) {
        inputNames.emplace_back("pic_recompute_budget");
    }
    if (mConfig->has_deepstack()) {
        inputNames.emplace_back("deepstack_embeds");
    }
    if (mConfig->has_ple()) {
        inputNames.emplace_back("ple_embeddings");
    }
    auto runtimeManager = mRuntimeManager;
    if (mConfig->backend_type() == "opencl") {
        if (mCacheBlendScoreRuntimeManager == nullptr) {
            mCacheBlendScoreRuntimeManager = createRuntimeManagerForCurrentConfig();
            if (mCacheBlendScoreRuntimeManager == nullptr) {
                MNN_ERROR("Failed to create isolated OpenCL runtime for cacheblend score module\n");
                return nullptr;
            }
            MNN_PRINT("OpenCL cacheblend score-layer module uses an isolated runtime manager\n");
        }
        runtimeManager = mCacheBlendScoreRuntimeManager;
        runtimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mMeta.get());
    }
    runtimeManager->setExternalFile(mConfig->llm_weight());
    std::shared_ptr<Module> module(Module::load(inputNames, {scoreOutputName}, mConfig->llm_model().c_str(),
                                                runtimeManager, &moduleConfig));
    runtimeManager->setExternalFile("");
    if (module == nullptr) {
        MNN_ERROR("Failed to load cacheblend score-layer module for layer %d output %s\n",
                  scoreLayerIdx, scoreOutputName.c_str());
        return nullptr;
    }
    MNN_PRINT("Loaded cacheblend score-layer module layer=%d output=%s\n", scoreLayerIdx, scoreOutputName.c_str());
    mCacheBlendScoreModulePool[scoreLayerIdx] = module;
    return module;
}

bool Llm::runCacheBlendScorePrefill(const std::vector<int>& fullPromptTokenIds, int scoreLayerIdx) {
    if (fullPromptTokenIds.empty()) {
        return false;
    }
    MNN::Express::ExecutorScope s(mExecutor);
    auto scoreModule = getCacheBlendScoreModule(scoreLayerIdx);
    if (scoreModule == nullptr) {
        return false;
    }
    scoreModule->clearCache();
    auto hiddenStates = embedding(fullPromptTokenIds);
    if (hiddenStates == nullptr) {
        return false;
    }
    int seqLen = hiddenStates->getInfo()->dim[mSeqLenIndex];
    if (seqLen <= 0) {
        return false;
    }
    mMeta->add = seqLen;
    auto attentionMask = gen_attention_mask(seqLen);
    auto positionIds = gen_position_ids(seqLen);
    mGenerateParam->input_embeds = nullptr;
    mGenerateParam->outputs.clear();
    mGenerateParam->validLogitSize = 0;
    mGenerateParam->validLogitStart = 0;
    Express::VARPS extraArgs;
    if (mPleInput.get()) {
        extraArgs.push_back(mPleInput);
    }
    std::vector<Express::VARP> inputs {hiddenStates, attentionMask, positionIds, logitsLastIdx};
    auto picBudget = _picRecomputeBudgetVar(mConfig->paged_attention() ? static_cast<PagedKVMeta*>(mMeta.get()) : nullptr, seqLen,
                                            mConfig->has_pic_recompute_budget());
    if (picBudget.get() != nullptr) {
        inputs.emplace_back(picBudget);
    }
    inputs.insert(inputs.end(), extraArgs.begin(), extraArgs.end());
    auto outputs = scoreModule->onForward(inputs);
    if (outputs.empty() || outputs[0] == nullptr) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return false;
    }
    // Force the truncated graph to execute so the target PagedAttention layer can
    // run its backend-side delta scoring and top-k selection.
    if (outputs[0]->readMap<float>() == nullptr) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return false;
    }
    if (mConfig->paged_attention()) {
        static_cast<PagedKVMeta*>(mMeta.get())->syncPaged();
    } else {
        mMeta->sync();
    }
    updateContext(seqLen, 0);
    if (mConfig->backend_type() == "opencl") {
        outputs.clear();
        hiddenStates = nullptr;
        attentionMask = nullptr;
        positionIds = nullptr;
        this->inputsEmbeds = nullptr;
        this->attentionMask = nullptr;
        this->positionIds = nullptr;
        this->mPicRecomputeBudget = nullptr;
        this->mPleInput = nullptr;
        this->mTextEmbedsForPle = nullptr;
        MNN::Express::ExecutorScope::Current()->gc(Executor::PART);
    }
    return true;
}

bool Llm::selectCacheBlendExternalPagedKV(const std::vector<int>& full_prompt_token_ids,
                                          const std::vector<MNN::PagedKVExternalSegment>& segments,
                                          int pic_start, int pic_token_count, int score_layer_idx,
                                          double recompute_ratio, std::vector<int>& selected_local_indices) {
    const int64_t totalStartUs = picRequestProfileEnabled() ? picRequestMonotonicUs() : 0;
    selected_local_indices.clear();
    if (!mConfig->paged_attention()) {
        MNN_ERROR("Native cacheblend scoring requires paged_attention=true\n");
        return false;
    }
    if (pic_start < 0 || pic_token_count < 0 || score_layer_idx < 0 ||
        pic_start + pic_token_count > static_cast<int>(full_prompt_token_ids.size())) {
        return false;
    }
    int topK = 0;
    if (pic_token_count > 0 && recompute_ratio > 0.0) {
        topK = std::min(pic_token_count,
                        std::max(1, static_cast<int>(std::ceil(pic_token_count * recompute_ratio))));
    }
    auto paged = static_cast<PagedKVMeta*>(mMeta.get());
    if (paged == nullptr) {
        return false;
    }
    bool startedPagedRequest = beginPagedRequestIfNeeded();
    if (!paged->beginCacheBlendScoring(pic_start, pic_token_count, score_layer_idx, topK, segments)) {
        if (startedPagedRequest) {
            finishPagedRequestIfNeeded();
        }
        return false;
    }
    auto oldStatus = mContext->status;
    const int64_t scorePrefillStartUs = picRequestProfileEnabled() ? picRequestMonotonicUs() : 0;
    if (!runCacheBlendScorePrefill(full_prompt_token_ids, score_layer_idx)) {
        paged->finishCacheBlendScoring();
        if (startedPagedRequest) {
            finishPagedRequestIfNeeded();
        }
        if (oldStatus == LlmStatus::RUNNING && mContext->status != LlmStatus::TIMEOUT &&
            mContext->status != LlmStatus::USER_CANCEL) {
            mContext->status = oldStatus;
        }
        return false;
    }
    if (picRequestProfileEnabled()) {
        std::ostringstream os;
        os << "score_layer=" << score_layer_idx
           << " pic_start=" << pic_start
           << " pic_tokens=" << pic_token_count
           << " topk=" << topK;
        picRequestProfileLog("select_cacheblend_score_prefill", picRequestMonotonicUs() - scorePrefillStartUs, os.str());
    }
    const bool ready = paged->cacheblend_score_ready;
    if (ready) {
        selected_local_indices = paged->cacheblend_score_selected_local_indices;
    }
    paged->finishCacheBlendScoring();
    if (startedPagedRequest) {
        finishPagedRequestIfNeeded();
    }
    if (oldStatus == LlmStatus::RUNNING && mContext->status != LlmStatus::INTERNAL_ERROR &&
        mContext->status != LlmStatus::TIMEOUT && mContext->status != LlmStatus::USER_CANCEL) {
        mContext->status = oldStatus;
    }
    if (picRequestProfileEnabled()) {
        std::ostringstream os;
        os << "score_layer=" << score_layer_idx
           << " selected_count=" << selected_local_indices.size()
           << " topk=" << topK;
        picRequestProfileLog("select_cacheblend_external_pagedkv_total", picRequestMonotonicUs() - totalStartUs,
                             os.str());
    }
    return ready && static_cast<int>(selected_local_indices.size()) == topK;
}

bool Llm::prefillCacheBlendGraphExternalPagedKV(const std::vector<int>& full_prompt_token_ids,
                                                const std::vector<MNN::PagedKVExternalSegment>& segments,
                                                int pic_start, int pic_token_count, int score_layer_idx,
                                                double recompute_ratio,
                                                std::vector<int>& selected_local_indices) {
    const int64_t totalStartUs = picRequestProfileEnabled() ? picRequestMonotonicUs() : 0;
    selected_local_indices.clear();
    if (!mConfig->paged_attention() || full_prompt_token_ids.empty()) {
        return false;
    }
    if (!mConfig->has_pic_recompute_budget()) {
        MNN_ERROR("Graph-level cacheblend prefill requires pic_recompute_budget input\n");
        return false;
    }
    if (pic_start < 0 || pic_token_count < 0 || score_layer_idx < 0 ||
        pic_start + pic_token_count > static_cast<int>(full_prompt_token_ids.size())) {
        return false;
    }
    int topK = 0;
    if (pic_token_count > 0 && recompute_ratio > 0.0) {
        topK = std::min(pic_token_count,
                        std::max(1, static_cast<int>(std::ceil(pic_token_count * recompute_ratio))));
    }
    auto paged = static_cast<PagedKVMeta*>(mMeta.get());
    if (paged == nullptr) {
        return false;
    }
    beginPagedRequestIfNeeded();
    std::vector<PagedKVExternalSegment> boundSegments;
    boundSegments.reserve(segments.size());
    size_t cursor = 0;
    for (auto segment : segments) {
        segment.logicalStart = static_cast<size_t>(pic_start) + cursor;
        cursor += segment.tokenCount;
        boundSegments.emplace_back(std::move(segment));
    }
    if (cursor != static_cast<size_t>(pic_token_count)) {
        MNN_ERROR("Graph-level cacheblend segment token mismatch, segments=%d pic_tokens=%d\n",
                  static_cast<int>(cursor), pic_token_count);
        return false;
    }
    int64_t stageStartUs = picRequestProfileEnabled() ? picRequestMonotonicUs() : 0;
    if (!paged->bindExternalSegments(boundSegments, static_cast<int>(full_prompt_token_ids.size()))) {
        MNN_ERROR("Graph-level cacheblend failed to bind persistent PIC cache source segments\n");
        return false;
    }
    if (picRequestProfileEnabled()) {
        std::ostringstream os;
        os << "pic_start=" << pic_start
           << " pic_tokens=" << pic_token_count
           << " full_prompt_tokens=" << full_prompt_token_ids.size()
           << " segments=" << boundSegments.size();
        picRequestProfileLog("graph_cacheblend_bind_external_segments", picRequestMonotonicUs() - stageStartUs,
                             os.str());
    }
    paged->external_hydrate_start_layer_idx = score_layer_idx + 1;
    stageStartUs = picRequestProfileEnabled() ? picRequestMonotonicUs() : 0;
    if (!paged->beginCacheBlendScoring(pic_start, pic_token_count, score_layer_idx, topK, segments)) {
        MNN_ERROR("Graph-level cacheblend failed to start score-layer scoring\n");
        return false;
    }
    if (picRequestProfileEnabled()) {
        std::ostringstream os;
        os << "score_layer=" << score_layer_idx
           << " topk=" << topK
           << " pic_tokens=" << pic_token_count;
        picRequestProfileLog("graph_cacheblend_begin_scoring", picRequestMonotonicUs() - stageStartUs, os.str());
    }
    stageStartUs = picRequestProfileEnabled() ? picRequestMonotonicUs() : 0;
    const bool ok = prefill(full_prompt_token_ids);
    if (picRequestProfileEnabled()) {
        std::ostringstream os;
        os << "score_layer=" << score_layer_idx
           << " full_prompt_tokens=" << full_prompt_token_ids.size();
        picRequestProfileLog("graph_cacheblend_prefill_full_prompt", picRequestMonotonicUs() - stageStartUs, os.str());
    }
    const bool ready = paged->cacheblend_score_ready;
    if (ready) {
        selected_local_indices = paged->cacheblend_score_selected_local_indices;
    }
    paged->finishCacheBlendScoring();
    paged->finishSparseQuery();
    if (!ok || !ready || static_cast<int>(selected_local_indices.size()) != topK) {
        return false;
    }
    if (picRequestProfileEnabled()) {
        std::ostringstream os;
        os << "score_layer=" << score_layer_idx
           << " selected_count=" << selected_local_indices.size()
           << " topk=" << topK;
        picRequestProfileLog("prefill_cacheblend_graph_external_pagedkv_total",
                             picRequestMonotonicUs() - totalStartUs, os.str());
    }
    return true;
}

bool Llm::prefillFixedGraphExternalPagedKV(const std::vector<int>& full_prompt_token_ids,
                                           const std::vector<MNN::PagedKVExternalSegment>& segments,
                                           int pic_start, int pic_token_count, int score_layer_idx,
                                           const std::vector<int>& selected_local_indices) {
    if (!mConfig->paged_attention() || full_prompt_token_ids.empty()) {
        return false;
    }
    if (!mConfig->has_pic_recompute_budget()) {
        MNN_ERROR("Fixed graph-level PIC prefill requires pic_recompute_budget input\n");
        return false;
    }
    if (pic_start < 0 || pic_token_count < 0 || score_layer_idx < 0 ||
        pic_start + pic_token_count > static_cast<int>(full_prompt_token_ids.size())) {
        return false;
    }
    auto paged = static_cast<PagedKVMeta*>(mMeta.get());
    if (paged == nullptr) {
        return false;
    }
    beginPagedRequestIfNeeded();
    std::vector<PagedKVExternalSegment> boundSegments;
    boundSegments.reserve(segments.size());
    size_t cursor = 0;
    for (auto segment : segments) {
        segment.logicalStart = static_cast<size_t>(pic_start) + cursor;
        cursor += segment.tokenCount;
        boundSegments.emplace_back(std::move(segment));
    }
    if (cursor != static_cast<size_t>(pic_token_count)) {
        MNN_ERROR("Fixed graph-level PIC segment token mismatch, segments=%d pic_tokens=%d\n",
                  static_cast<int>(cursor), pic_token_count);
        return false;
    }
    if (!paged->bindExternalSegments(boundSegments, static_cast<int>(full_prompt_token_ids.size()))) {
        MNN_ERROR("Fixed graph-level PIC failed to bind persistent PIC cache source segments\n");
        return false;
    }
    paged->external_hydrate_start_layer_idx = score_layer_idx + 1;
    if (!paged->beginPicGraphActivePlan(pic_start, pic_token_count, score_layer_idx, selected_local_indices)) {
        MNN_ERROR("Fixed graph-level PIC failed to bind active rows\n");
        return false;
    }
    const bool ok = prefill(full_prompt_token_ids);
    paged->finishPicGraphActivePlan();
    paged->finishSparseQuery();
    return ok;
}

bool Llm::preparePicDecodeRepair(int pic_start, const std::vector<int>& pic_token_ids,
                                 const std::vector<int>& ranked_pic_local_indices,
                                 const std::vector<int>& seed_selected_pic_local_indices,
                                 int tokens_per_decode_step, const std::string& selector,
                                 int attention_layer_idx,
                                 const std::vector<int>& attention_head_ids,
                                 int attention_candidate_pool_size) {
    clearPicDecodeRepair();
    const int picTokenCount = static_cast<int>(pic_token_ids.size());
    if (!mConfig->paged_attention()) {
        MNN_ERROR("PIC decode repair requires paged_attention in exported model\n");
        return false;
    }
    if (mConfig->has_deepstack() || mConfig->has_ple()) {
        MNN_ERROR("PIC decode repair token-id sparse decode currently does not support deepstack or PLE models\n");
        return false;
    }
    if (mConfig->attention_mask() != "float" || mConfig->attention_type() != "full") {
        MNN_ERROR("PIC decode repair token-id sparse decode currently supports only float full-attention masks\n");
        return false;
    }
    if (pic_start < 0 || picTokenCount <= 0 || tokens_per_decode_step < 0) {
        return false;
    }
    std::vector<uint8_t> seen(static_cast<size_t>(picTokenCount), 0);
    std::vector<int> ranked;
    ranked.reserve(static_cast<size_t>(picTokenCount));
    auto appendRank = [&](int local) {
        if (local < 0 || local >= picTokenCount || seen[static_cast<size_t>(local)] != 0) {
            return;
        }
        seen[static_cast<size_t>(local)] = 1;
        ranked.emplace_back(pic_start + local);
    };
    for (int local : ranked_pic_local_indices) {
        appendRank(local);
    }
    for (int local = 0; local < picTokenCount; ++local) {
        appendRank(local);
    }
    mPicDecodeRepair.enabled = true;
    mPicDecodeRepair.picStart = pic_start;
    mPicDecodeRepair.picTokenCount = picTokenCount;
    mPicDecodeRepair.tokensPerDecodeStep = tokens_per_decode_step;
    mPicDecodeRepair.selector = selector == "lagged_attention_hkvd" ? selector : "top_hkvd";
    mPicDecodeRepair.attentionLayerIdx = attention_layer_idx;
    mPicDecodeRepair.attentionCandidatePoolSize = tokens_per_decode_step > 0
        ? std::max(tokens_per_decode_step, attention_candidate_pool_size)
        : 0;
    mPicDecodeRepair.cursor = 0;
    mPicDecodeRepair.stepIdx = 0;
    mPicDecodeRepair.picTokenIds = pic_token_ids;
    mPicDecodeRepair.rankedLogicalIndices = std::move(ranked);
    mPicDecodeRepair.attentionHeadIds.clear();
    mPicDecodeRepair.attentionHeadIds.reserve(attention_head_ids.size());
    for (int head : attention_head_ids) {
        if (head >= 0) {
            mPicDecodeRepair.attentionHeadIds.emplace_back(head);
        }
    }
    std::sort(mPicDecodeRepair.attentionHeadIds.begin(), mPicDecodeRepair.attentionHeadIds.end());
    mPicDecodeRepair.attentionHeadIds.erase(
        std::unique(mPicDecodeRepair.attentionHeadIds.begin(), mPicDecodeRepair.attentionHeadIds.end()),
        mPicDecodeRepair.attentionHeadIds.end());
    mPicDecodeRepair.lastAttentionRankedPicLocalIndices.clear();
    mPicDecodeRepair.lastAttentionSourceStepIdx = -2;
    mPicDecodeRepair.repairedPicLocal.assign(static_cast<size_t>(picTokenCount), 0);
    mPicDecodeRepair.picTokenEmbeddings.clear();
    mPicDecodeRepair.picTokenEmbeddingHiddenSize = 0;
    const size_t decodeRepairRows = static_cast<size_t>(tokens_per_decode_step + 1);
    mPicDecodeRepair.scratchLogicalIndices.reserve(decodeRepairRows);
    mPicDecodeRepair.scratchSparseTokenIds.reserve(decodeRepairRows);
    const int hiddenSize = mConfig->hidden_size();
    if (tokens_per_decode_step > 0 && hiddenSize > 0 && !mPicDecodeRepair.picTokenIds.empty()) {
        mPicDecodeRepair.picTokenEmbeddings.resize(
            static_cast<size_t>(mPicDecodeRepair.picTokenIds.size()) * static_cast<size_t>(hiddenSize));
        mDiskEmbedding->embedding(mPicDecodeRepair.picTokenIds, mPicDecodeRepair.picTokenEmbeddings.data());
        mPicDecodeRepair.picTokenEmbeddingHiddenSize = hiddenSize;
    }
    for (int local : seed_selected_pic_local_indices) {
        if (local >= 0 && local < picTokenCount) {
            mPicDecodeRepair.repairedPicLocal[static_cast<size_t>(local)] = 1;
        }
    }
    mPicDecodeRepair.pendingRepairLogicalIndices = selectPicDecodeRepairLogicalIndices();
    if (mConfig->paged_attention()) {
        auto paged = static_cast<PagedKVMeta*>(mMeta.get());
        paged->pic_decode_repair_tokens_per_step = tokens_per_decode_step;
    }
    MNN_PRINT("Prepared PIC decode repair token-id sparse decode selector=%s pic_start=%d pic_tokens=%d "
              "budget_per_step=%d attention_layer=%d attention_pool=%d ranked_tokens=%d first_scheduled=%d\n",
              mPicDecodeRepair.selector.c_str(), pic_start, picTokenCount, tokens_per_decode_step,
              mPicDecodeRepair.attentionLayerIdx, mPicDecodeRepair.attentionCandidatePoolSize,
              static_cast<int>(mPicDecodeRepair.rankedLogicalIndices.size()),
              static_cast<int>(mPicDecodeRepair.pendingRepairLogicalIndices.size()));
    return true;
}

void Llm::clearPicDecodeRepair() {
    mPicDecodeRepair = PicDecodeRepairRuntimeState();
    if (mConfig->paged_attention()) {
        auto paged = static_cast<PagedKVMeta*>(mMeta.get());
        paged->pic_decode_repair_tokens_per_step = 0;
    }
}

std::vector<int> Llm::selectPicDecodeRepairLogicalIndices() {
    std::vector<int> selected;
    if (!mPicDecodeRepair.enabled || mPicDecodeRepair.tokensPerDecodeStep <= 0) {
        return selected;
    }
    selected.reserve(static_cast<size_t>(mPicDecodeRepair.tokensPerDecodeStep));
    std::vector<uint8_t> selectedLocal(static_cast<size_t>(std::max(0, mPicDecodeRepair.picTokenCount)), 0);
    auto tryAppendLogical = [&](int logical) {
        const int local = logical - mPicDecodeRepair.picStart;
        if (local < 0 || local >= mPicDecodeRepair.picTokenCount) {
            return false;
        }
        if (mPicDecodeRepair.repairedPicLocal[static_cast<size_t>(local)] != 0) {
            return false;
        }
        if (!selectedLocal.empty() && selectedLocal[static_cast<size_t>(local)] != 0) {
            return false;
        }
        selected.emplace_back(logical);
        if (!selectedLocal.empty()) {
            selectedLocal[static_cast<size_t>(local)] = 1;
        }
        return true;
    };
    auto appendTopHkvdFill = [&]() {
        while (mPicDecodeRepair.cursor < static_cast<int>(mPicDecodeRepair.rankedLogicalIndices.size()) &&
               static_cast<int>(selected.size()) < mPicDecodeRepair.tokensPerDecodeStep) {
            const int logical = mPicDecodeRepair.rankedLogicalIndices[mPicDecodeRepair.cursor++];
            tryAppendLogical(logical);
        }
    };
    if (mPicDecodeRepair.selector == "lagged_attention_hkvd" &&
        !mPicDecodeRepair.lastAttentionRankedPicLocalIndices.empty()) {
        std::vector<uint8_t> inAttentionPool(
            static_cast<size_t>(std::max(0, mPicDecodeRepair.picTokenCount)), 0);
        int poolCount = 0;
        const int poolLimit = std::max(0, mPicDecodeRepair.attentionCandidatePoolSize);
        for (int local : mPicDecodeRepair.lastAttentionRankedPicLocalIndices) {
            if (local < 0 || local >= mPicDecodeRepair.picTokenCount) {
                continue;
            }
            if (!inAttentionPool.empty() && inAttentionPool[static_cast<size_t>(local)] != 0) {
                continue;
            }
            if (!inAttentionPool.empty()) {
                inAttentionPool[static_cast<size_t>(local)] = 1;
            }
            ++poolCount;
            if (poolLimit > 0 && poolCount >= poolLimit) {
                break;
            }
        }
        if (poolCount > 0) {
            for (int logical : mPicDecodeRepair.rankedLogicalIndices) {
                if (static_cast<int>(selected.size()) >= mPicDecodeRepair.tokensPerDecodeStep) {
                    break;
                }
                const int local = logical - mPicDecodeRepair.picStart;
                if (local < 0 || local >= mPicDecodeRepair.picTokenCount) {
                    continue;
                }
                if (inAttentionPool.empty() || inAttentionPool[static_cast<size_t>(local)] == 0) {
                    continue;
                }
                tryAppendLogical(logical);
            }
        }
    }
    appendTopHkvdFill();
    std::sort(selected.begin(), selected.end());
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    return selected;
}

int Llm::picDecodeRepairCandidateCount() const {
    return static_cast<int>(mPicDecodeRepair.rankedLogicalIndices.size());
}

std::vector<int> Llm::picDecodeRepairRepairedLogicalIndices() const {
    std::vector<int> repaired;
    if (!mPicDecodeRepair.enabled || mPicDecodeRepair.picTokenCount <= 0 ||
        mPicDecodeRepair.repairedPicLocal.empty()) {
        return repaired;
    }
    for (int local = 0; local < mPicDecodeRepair.picTokenCount; ++local) {
        if (mPicDecodeRepair.repairedPicLocal[static_cast<size_t>(local)] != 0) {
            repaired.emplace_back(mPicDecodeRepair.picStart + local);
        }
    }
    return repaired;
}

std::vector<std::vector<int>> Llm::picDecodeRepairStepLogicalIndices() const {
    return mPicDecodeRepair.stepLogicalIndices;
}

VARP Llm::embeddingForPicDecodeRepair(const std::vector<int>& inputIds) {
    if (inputIds.empty()) {
        return nullptr;
    }
    MNN::Express::ExecutorScope s(mExecutor);
    AUTOTIME;
    const int hiddenSize = mConfig->hidden_size();
    const int seqLen = static_cast<int>(inputIds.size());
    auto& cached = mPicDecodeRepair.embeddingByLen[seqLen];
    if (cached == nullptr) {
        cached = _Input({seqLen, 1, hiddenSize}, NCHW);
    }
    auto* dst = cached->writeMap<float>();
    const int repairRows = std::max(0, seqLen - 1);
    const bool canUsePicEmbeddingCache =
        mPicDecodeRepair.picTokenEmbeddingHiddenSize == hiddenSize &&
        !mPicDecodeRepair.picTokenEmbeddings.empty() &&
        static_cast<int>(mPicDecodeRepair.scratchLogicalIndices.size()) == seqLen;
    if (canUsePicEmbeddingCache && repairRows > 0) {
        for (int i = 0; i < repairRows; ++i) {
            const int logical = mPicDecodeRepair.scratchLogicalIndices[static_cast<size_t>(i)];
            const int local = logical - mPicDecodeRepair.picStart;
            if (local < 0 || local >= static_cast<int>(mPicDecodeRepair.picTokenIds.size())) {
                mDiskEmbedding->embedding(inputIds, dst);
                dst = nullptr;
                break;
            }
            const size_t srcOffset = static_cast<size_t>(local) * static_cast<size_t>(hiddenSize);
            std::memcpy(dst + static_cast<size_t>(i) * static_cast<size_t>(hiddenSize),
                        mPicDecodeRepair.picTokenEmbeddings.data() + srcOffset,
                        static_cast<size_t>(hiddenSize) * sizeof(float));
        }
        if (dst != nullptr) {
            const std::vector<int> decodeToken{inputIds.back()};
            mDiskEmbedding->embedding(decodeToken,
                                      dst + static_cast<size_t>(repairRows) * static_cast<size_t>(hiddenSize));
        }
    } else {
        mDiskEmbedding->embedding(inputIds, dst);
    }

    if (mPleEmbedding && (!mPleInput.get() || seqLen == 1)) {
        const int pleDim = mConfig->ple_embed_dim();
        const float pleScale = mConfig->ple_embed_scale();
        mPleInput = _Input({1, seqLen, pleDim}, NCHW);
        mPleEmbedding->embedding(inputIds, mPleInput->writeMap<float>());
        if (pleScale != 1.0f) {
            mPleInput = mPleInput * _Scalar<float>(pleScale);
        }
    }
    return cached;
}

VARP Llm::genDecodeRepairAttentionMask(const std::vector<int>& logicalIndices) {
    int queryLen = static_cast<int>(logicalIndices.size());
    int kvLen = 0;
    bool fullCausalDecodeRepair = false;
    if (mConfig->paged_attention()) {
        auto paged = static_cast<PagedKVMeta*>(mMeta.get());
        if (paged != nullptr && paged->pic_decode_recompute_active) {
            queryLen = std::max(1, paged->pic_active_count);
            kvLen = std::max(1, paged->logical_length);
            fullCausalDecodeRepair = paged->full_causal_attention_mask;
        }
    }
    if (fullCausalDecodeRepair) {
        auto& cached = mPicDecodeRepair.causalMaskByLen[queryLen];
        if (cached == nullptr) {
            cached = _Input({1, 1, queryLen, 1}, NCHW, halide_type_of<float>());
            auto ptr = cached->writeMap<float>();
            std::fill(ptr, ptr + queryLen, 0.0f);
        }
        attentionMask = cached;
        return attentionMask;
    }
    if (kvLen <= 0) {
        kvLen = std::max(1, mContext->all_seq_len + 1);
    }
    attentionMask = _Input({1, 1, queryLen, kvLen}, NCHW, halide_type_of<float>());
    auto ptr = attentionMask->writeMap<float>();
    for (int q = 0; q < queryLen; ++q) {
        const int logicalQ = q < static_cast<int>(logicalIndices.size()) ? logicalIndices[q] : q;
        for (int k = 0; k < kvLen; ++k) {
            ptr[q * kvLen + k] = k > logicalQ ? std::numeric_limits<float>::lowest() : 0.0f;
        }
    }
    return attentionMask;
}

VARP Llm::genDecodeRepairPositionIds(const std::vector<int>& logicalIndices) {
    const int seqLen = static_cast<int>(logicalIndices.size());
    auto& cached = mPicDecodeRepair.positionIdsByLen[seqLen];
    if (mConfig->is_mrope()) {
        if (cached == nullptr) {
            cached = _Input({3, seqLen}, NCHW, halide_type_of<int>());
        }
        positionIds = cached;
        auto ptr = positionIds->writeMap<int>();
        for (int i = 0; i < seqLen; ++i) {
            ptr[0 * seqLen + i] = logicalIndices[i];
            ptr[1 * seqLen + i] = logicalIndices[i];
            ptr[2 * seqLen + i] = logicalIndices[i];
        }
        return positionIds;
    }
    if (cached == nullptr) {
        cached = _Input({1, seqLen}, NCHW, halide_type_of<int>());
    }
    positionIds = cached;
    auto ptr = positionIds->writeMap<int>();
    for (int i = 0; i < seqLen; ++i) {
        ptr[i] = logicalIndices[i];
    }
    return positionIds;
}

std::vector<VARP> Llm::forwardVecWithPicDecodeRepair(const std::vector<int>& inputIds) {
    if (!mPicDecodeRepair.enabled || inputIds.size() != 1 || !mConfig->paged_attention()) {
        return forwardVec(inputIds);
    }
    auto paged = static_cast<PagedKVMeta*>(mMeta.get());
    if (paged == nullptr || !paged->request_active) {
        return forwardVec(inputIds);
    }
    const bool profileDecodeRepair = picDecodeRepairProfileEnabled();
    const int64_t totalStartUs = profileDecodeRepair ? picRequestMonotonicUs() : 0;
    int64_t stageStartUs = totalStartUs;
    auto finishProfileStage = [&]() -> int64_t {
        if (!profileDecodeRepair) {
            return 0;
        }
        const int64_t nowUs = picRequestMonotonicUs();
        const int64_t elapsedUs = nowUs - stageStartUs;
        stageStartUs = nowUs;
        return elapsedUs;
    };
    const int decodeLogical = mContext->all_seq_len;
    const auto& repairLogical = mPicDecodeRepair.pendingRepairLogicalIndices;
    auto& logicalIndices = mPicDecodeRepair.scratchLogicalIndices;
    auto& sparseTokenIds = mPicDecodeRepair.scratchSparseTokenIds;
    logicalIndices.clear();
    sparseTokenIds.clear();
    for (int logical : repairLogical) {
        const int local = logical - mPicDecodeRepair.picStart;
        if (local < 0 || local >= mPicDecodeRepair.picTokenCount ||
            local >= static_cast<int>(mPicDecodeRepair.picTokenIds.size())) {
            continue;
        }
        logicalIndices.emplace_back(logical);
        sparseTokenIds.emplace_back(mPicDecodeRepair.picTokenIds[static_cast<size_t>(local)]);
    }
    logicalIndices.emplace_back(decodeLogical);
    sparseTokenIds.emplace_back(inputIds[0]);
    const int actualRepairRows = std::max(0, static_cast<int>(logicalIndices.size()) - 1);
    const int sparseRows = static_cast<int>(logicalIndices.size());
    const int64_t buildRowsUs = finishProfileStage();

    if (!paged->beginPicDecodeRecomputeRows(logicalIndices, 0, 1)) {
        MNN_ERROR("PIC decode repair failed to bind token-id sparse decode rows\n");
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    if (mPicDecodeRepair.selector == "lagged_attention_hkvd" &&
        mPicDecodeRepair.attentionLayerIdx >= 0 &&
        mPicDecodeRepair.attentionCandidatePoolSize > 0) {
        paged->beginPicDecodeAttentionRankCapture(
            mPicDecodeRepair.attentionLayerIdx, mPicDecodeRepair.picStart,
            mPicDecodeRepair.picTokenCount, mPicDecodeRepair.attentionCandidatePoolSize,
            mPicDecodeRepair.stepIdx, mPicDecodeRepair.attentionHeadIds);
    } else {
        paged->finishPicDecodeAttentionRankCapture();
        paged->clearPicDecodeAttentionRankResult();
    }
    const int64_t beginRecomputeUs = finishProfileStage();
    auto inputEmbeds = embeddingForPicDecodeRepair(sparseTokenIds);
    const int64_t embeddingUs = finishProfileStage();
    if (inputEmbeds == nullptr) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        paged->finishSparseQuery();
        return {};
    }
    mMeta->add = logicalIndices.size();
    auto repairMask = genDecodeRepairAttentionMask(logicalIndices);
    auto repairPos = genDecodeRepairPositionIds(logicalIndices);
    const int64_t maskPosUs = finishProfileStage();
    auto outputs = forwardRaw(inputEmbeds, repairMask, repairPos);
    const int64_t forwardRawUs = finishProfileStage();
    if (outputs.empty()) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        paged->finishSparseQuery();
        return outputs;
    }
    for (auto output : outputs) {
        if (output == nullptr || output->getInfo() == nullptr) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
            paged->finishSparseQuery();
            return outputs;
        }
    }
    const int64_t validateUs = finishProfileStage();
    const int consumedStepIdx = mPicDecodeRepair.stepIdx;
    const bool consumedAttentionRank = mPicDecodeRepair.selector == "lagged_attention_hkvd" &&
        paged->pic_decode_attention_rank_source_step_idx == consumedStepIdx;
    if (consumedAttentionRank) {
        mPicDecodeRepair.lastAttentionRankedPicLocalIndices =
            paged->pic_decode_attention_ranked_local_indices;
        mPicDecodeRepair.lastAttentionSourceStepIdx =
            paged->pic_decode_attention_rank_source_step_idx;
    }
    const int attentionRankCount =
        consumedAttentionRank ? static_cast<int>(mPicDecodeRepair.lastAttentionRankedPicLocalIndices.size()) : 0;
    const int attentionRankSourceStep = consumedAttentionRank ? mPicDecodeRepair.lastAttentionSourceStepIdx : -1;
    const int64_t rankResultUs = finishProfileStage();
    paged->finishSparseQuery();
    const int64_t finishSparseUs = finishProfileStage();
    mPicDecodeRepair.stepLogicalIndices.emplace_back(repairLogical);
    for (int logical : repairLogical) {
        const int local = logical - mPicDecodeRepair.picStart;
        if (local >= 0 && local < mPicDecodeRepair.picTokenCount) {
            mPicDecodeRepair.repairedPicLocal[static_cast<size_t>(local)] = 1;
        }
    }
    ++mPicDecodeRepair.stepIdx;
    mPicDecodeRepair.pendingRepairLogicalIndices = selectPicDecodeRepairLogicalIndices();
    mGenerateParam->input_embeds = inputEmbeds;
    mGenerateParam->outputs = outputs;
    mGenerateParam->validLogitSize = 0;
    mGenerateParam->validLogitStart = 0;
    const int64_t bookkeepingUs = finishProfileStage();
    if (profileDecodeRepair) {
        const int64_t totalUs = picRequestMonotonicUs() - totalStartUs;
        picDecodeRepairProfileLog(consumedStepIdx, mPicDecodeRepair.selector.c_str(), decodeLogical,
                                  actualRepairRows, sparseRows,
                                  static_cast<int>(mPicDecodeRepair.pendingRepairLogicalIndices.size()),
                                  attentionRankCount, attentionRankSourceStep, buildRowsUs,
                                  beginRecomputeUs, embeddingUs, maskPosUs, forwardRawUs, validateUs,
                                  rankResultUs, finishSparseUs, bookkeepingUs, totalUs);
    }
    if (std::getenv("MNN_PIC_DECODE_DEBUG") != nullptr) {
        std::fprintf(stderr,
                     "PIC decode repair token-id forward step=%d normal_logical=%d sparse_rows=%d repair_rows=%d "
                     "next_scheduled=%d\n",
                     consumedStepIdx, decodeLogical, static_cast<int>(logicalIndices.size()),
                     static_cast<int>(repairLogical.size()),
                     static_cast<int>(mPicDecodeRepair.pendingRepairLogicalIndices.size()));
        std::fflush(stderr);
    }
    return outputs;
}

void Llm::finishExternalPagedKVRequest() {
    clearGenerateForwardState(mGenerateParam);
    clearPicDecodeRepair();
    clearModuleForwardCaches();
    finishPagedRequestIfNeeded();
}

bool Llm::preparePagedDecode(int maxNewTokens) {
    if (!mConfig->paged_attention()) {
        return true;
    }
    if (maxNewTokens < 0) {
        maxNewTokens = mConfig->max_new_tokens();
    }
    if (maxNewTokens <= 0) {
        return true;
    }
    auto paged = static_cast<PagedKVMeta*>(mMeta.get());
    if (paged == nullptr) {
        return true;
    }
    beginPagedRequestIfNeeded(0, maxNewTokens);
    if (paged->logical_length <= 0 && mContext != nullptr) {
        paged->logical_length = std::max(paged->logical_length, mContext->all_seq_len);
    }
    if (paged->logical_length <= 0) {
        return true;
    }
    if (mConfig->backend_type() != "opencl") {
        return true;
    }
    const int64_t startUs = picRequestProfileEnabled() ? picRequestMonotonicUs() : 0;
    if (paged->decode_prepare_callback == nullptr) {
        std::ostringstream err;
        err << "OpenCL PagedAttention decode prepare callback is unavailable"
            << " logical_length=" << paged->logical_length
            << " request_capacity=" << paged->request_capacity;
        mLastError = err.str();
        MNN_ERROR("%s\n", mLastError.c_str());
        return false;
    }
    const bool ok = paged->decode_prepare_callback(paged, maxNewTokens);
    if (picRequestProfileEnabled()) {
        std::ostringstream os;
        os << "tokens=" << maxNewTokens
           << " logical_length=" << paged->logical_length
           << " request_capacity=" << paged->request_capacity
           << " ok=" << (ok ? 1 : 0);
        picRequestProfileLog("prepare_paged_decode", picRequestMonotonicUs() - startUs, os.str());
    }
    if (!ok) {
        std::ostringstream err;
        err << "OpenCL PagedAttention decode prepare failed"
            << " logical_length=" << paged->logical_length
            << " request_capacity=" << paged->request_capacity
            << " max_new_tokens=" << maxNewTokens;
        mLastError = err.str();
        MNN_ERROR("%s\n", mLastError.c_str());
    }
    return ok;
}

int Llm::pagedRequestCapacity(int pendingInputTokens, int maxNewTokens) const {
    int capacity = mConfig->paged_kv_max_tokens();
    int currentTokens = mContext != nullptr ? std::max(0, mContext->all_seq_len) : 0;
    int pendingTokens = std::max(0, pendingInputTokens);
    int decodeBudget = maxNewTokens < 0 ? mConfig->max_new_tokens() : maxNewTokens;
    decodeBudget = std::max(0, decodeBudget);
    return std::max(capacity, currentTokens + pendingTokens + decodeBudget);
}

bool Llm::beginPagedRequestIfNeeded(int pendingInputTokens, int maxNewTokens) {
    if (!mConfig->paged_attention()) {
        return false;
    }
    auto paged = static_cast<PagedKVMeta*>(mMeta.get());
    if (paged == nullptr) {
        return false;
    }
    int capacity = pagedRequestCapacity(pendingInputTokens, maxNewTokens);
    if (paged->request_active) {
        if (!paged->reserveRequestCapacity(capacity)) {
            MNN_ERROR("PagedAttention request failed to reserve capacity=%d current=%d\n",
                      capacity, paged->request_capacity);
        }
        return false;
    }
    paged->beginRequest(capacity);
    return true;
}

void Llm::finishPagedRequestIfNeeded() {
    if (!mConfig->paged_attention()) {
        return;
    }
    auto paged = static_cast<PagedKVMeta*>(mMeta.get());
    if (paged != nullptr && paged->request_active) {
        paged->finishRequest();
    }
}

void Llm::clearModuleForwardCaches() {
    if (mModule != nullptr) {
        mModule->clearCache();
    }
    if (mDecodeModule != nullptr) {
        mDecodeModule->clearCache();
    }
    for (auto& item : mModulePool) {
        if (item.second != nullptr && item.second != mModule) {
            item.second->clearCache();
        }
    }
    for (auto& item : mDecodeModulePool) {
        if (item.second != nullptr && item.second != mDecodeModule) {
            item.second->clearCache();
        }
    }
    for (auto& item : mCacheBlendScoreModulePool) {
        if (item.second != nullptr) {
            item.second->clearCache();
        }
    }
    for (auto iter = mModulePool.begin(); iter != mModulePool.end();) {
        if (iter->first.first == mPrefillKey) {
            iter = mModulePool.erase(iter);
        } else {
            ++iter;
        }
    }
}

void Llm::collectRuntimeGarbage() {
    if (mExecutor != nullptr) {
        MNN::Express::ExecutorScope s(mExecutor);
        MNN::Express::ExecutorScope::Current()->gc(Executor::FULL);
    }
}

size_t Llm::releaseForwardModuleClones() {
    MNN::Express::ExecutorScope s(mExecutor);
    clearModuleForwardCaches();
    inputsEmbeds = nullptr;
    attentionMask = nullptr;
    positionIds = nullptr;
    mPicRecomputeBudget = nullptr;
    mPleInput = nullptr;
    mTextEmbedsForPle = nullptr;

    size_t released = 0;
    for (auto iter = mModulePool.begin(); iter != mModulePool.end();) {
        if (iter->second != nullptr && iter->second != mModule) {
            iter = mModulePool.erase(iter);
            ++released;
        } else if (iter->first.first == mPrefillKey) {
            iter = mModulePool.erase(iter);
        } else {
            ++iter;
        }
    }
    released += mDecodeModulePool.size();
    mDecodeModulePool.clear();
    released += mCacheBlendScoreModulePool.size();
    mCacheBlendScoreModulePool.clear();
    mCacheBlendScoreRuntimeManager.reset();
    collectRuntimeGarbage();
    return released;
}

std::shared_ptr<Module> Llm::cloneModuleWithRuntime(const Module* module) {
    if (module == nullptr || mRuntimeManager == nullptr) {
        return nullptr;
    }
    mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mMeta.get());
    mRuntimeManager->setExternalPath(mConfig->prefix_cache_path(), MNN::Interpreter::EXTERNAL_PATH_PREFIXCACHE_DIR);
    if (mConfig->kvcache_mmap()) {
        mRuntimeManager->setExternalPath(mConfig->tmp_path(), MNN::Interpreter::EXTERNAL_PATH_KVCACHE_DIR);
    }
    if (mConfig->use_mmap()) {
        mRuntimeManager->setExternalPath(mConfig->tmp_path(), MNN::Interpreter::EXTERNAL_WEIGHT_DIR);
    }
    mRuntimeManager->setExternalPath(mConfig->npu_model_dir(), MNN::Interpreter::EXTERNAL_NPU_FILE_DIR);
    Module::CloneContext cloneContext;
    cloneContext.pRuntimeManager = mRuntimeManager;
    return std::shared_ptr<Module>(module->clone(&cloneContext));
}

std::vector<Express::VARP> Llm::forwardRaw(Express::VARP hiddenState, Express::VARP mask, Express::VARP inputPos, Express::VARPS extraArgs) {
    CHECK_LLM_RUNNING_RET(mContext, std::vector<Express::VARP>());
    MNN::Express::ExecutorScope s(mExecutor);
    
    Express::VARP logitsIndex;
    bool inDecode = mDecodeForwardActive || mContext->gen_seq_len > 0;
    bool isAllLogists = mConfig->all_logits() ? true : (inDecode ? mInSpec : false);
    auto seqLen = hiddenState->getInfo()->dim[mSeqLenIndex];
    int seqLenKey = inDecode ? hiddenState->getInfo()->dim[mSeqLenIndex] : mPrefillKey;
    isAllLogists = seqLenKey == 1 ? false : isAllLogists;
    const bool useDecodeGraph = mDecodeModule != nullptr &&
                                inDecode &&
                                seqLen == 1 &&
                                seqLenKey == 1 &&
                                !isAllLogists &&
                                !mPicDecodeRepair.enabled;
    auto moduleKey = std::make_pair(seqLenKey, isAllLogists);
    std::shared_ptr<Module> selectModule = mModule;
    if (useDecodeGraph) {
        auto iter = mDecodeModulePool.find(moduleKey);
        if(iter == mDecodeModulePool.end()) {
            MNN_PRINT("Warning: module need new clone, cloning now.\n");
            mDecodeModulePool[moduleKey] = cloneModuleWithRuntime(mDecodeModule.get());
            iter = mDecodeModulePool.find(moduleKey);
        }
        if (iter != mDecodeModulePool.end()) {
            selectModule = iter->second;
        }
    } else if (mValidBlockSize.empty()) {
        if(mModulePool.find(moduleKey) == mModulePool.end()) {
            MNN_PRINT("Warning: module need new clone, cloning now.\n");
            mModulePool[moduleKey] = cloneModuleWithRuntime(mModule.get());
        }
        selectModule = mModulePool[moduleKey];
    }
    if (selectModule == nullptr) {
        mLastError = "LLM module clone failed";
        MNN_ERROR("%s\n", mLastError.c_str());
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }

    if (isAllLogists) {
        logitsIndex = logitsAllIdx;
    } else {
        logitsIndex = logitsLastIdx;
    }
    if (mMeta->add != seqLen) {
        // Has Pad, need all logits
        logitsIndex = logitsAllIdx;
    }

    mGenerateParam->input_embeds = nullptr;
    mGenerateParam->outputs.clear();
    mGenerateParam->validLogitSize = 0;
    mGenerateParam->validLogitStart = 0;
    std::vector<Express::VARP> inputs {hiddenState, mask, inputPos, logitsIndex};
    auto picBudget = _picRecomputeBudgetVar(mConfig->paged_attention() ? static_cast<PagedKVMeta*>(mMeta.get()) : nullptr, seqLen,
                                            mConfig->has_pic_recompute_budget() && !useDecodeGraph);
    if (picBudget.get() != nullptr) {
        inputs.emplace_back(picBudget);
        mPicRecomputeBudget = picBudget;
    }
    inputs.insert(inputs.end(), extraArgs.begin(), extraArgs.end());
    if (std::getenv("MNN_PIC_DECODE_DEBUG") != nullptr && useDecodeGraph) {
        std::fprintf(stderr,
                     "PIC decode graph route seq_len=%d add=%d all_seq=%d gen_seq=%d inputs=%d\n",
                     seqLen, static_cast<int>(mMeta->add), mContext->all_seq_len, mContext->gen_seq_len,
                     static_cast<int>(inputs.size()));
        std::fflush(stderr);
    }
    std::vector<Express::VARP> outputs = selectModule->onForward(inputs);

    if (outputs.empty()) {
        std::ostringstream err;
        err << "forwardRaw outputs empty seq_len=" << seqLen
            << " add=" << static_cast<int>(mMeta->add)
            << " all_seq=" << mContext->all_seq_len
            << " gen_seq=" << mContext->gen_seq_len;
        if (mConfig->paged_attention()) {
            auto paged = static_cast<PagedKVMeta*>(mMeta.get());
            err << " paged_max=" << paged->max_tokens
                << " request_capacity=" << paged->request_capacity
                << " logical_length=" << paged->logical_length
                << " previous=" << static_cast<int>(paged->previous)
                << " remove=" << static_cast<int>(paged->remove);
        }
        mLastError = err.str();
        MNN_ERROR("PIC forward failed: outputs empty, seq_len=%d add=%d all_seq=%d gen_seq=%d paged_max=%d "
                  "request_capacity=%d logical_length=%d\n",
                  seqLen, static_cast<int>(mMeta->add), mContext->all_seq_len, mContext->gen_seq_len,
                  mConfig->paged_attention() ? static_cast<PagedKVMeta*>(mMeta.get())->max_tokens : 0,
                  mConfig->paged_attention() ? static_cast<PagedKVMeta*>(mMeta.get())->request_capacity : 0,
                  mConfig->paged_attention() ? static_cast<PagedKVMeta*>(mMeta.get())->logical_length : 0);
        if (std::getenv("MNN_PIC_DECODE_DEBUG") != nullptr) {
            std::fprintf(stderr, "PIC forward debug outputs empty seq_len=%d add=%d all_seq=%d gen_seq=%d\n",
                         seqLen, static_cast<int>(mMeta->add), mContext->all_seq_len, mContext->gen_seq_len);
            std::fflush(stderr);
        }
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return outputs;
    }
    // Validate output VARP. Only logits need to be materialized here; auxiliary
    // hidden outputs are consumed lazily by speculative/decode-repair paths.
    for (size_t outputIndex = 0; outputIndex < outputs.size(); ++outputIndex) {
        auto o = outputs[outputIndex];
        if(nullptr == o || nullptr == o->getInfo()) {
            std::ostringstream err;
            err << "forwardRaw invalid output index=" << outputIndex
                << " null=" << (o == nullptr ? 1 : 0)
                << " seq_len=" << seqLen
                << " add=" << static_cast<int>(mMeta->add)
                << " all_seq=" << mContext->all_seq_len
                << " gen_seq=" << mContext->gen_seq_len;
            mLastError = err.str();
            MNN_ERROR("PIC forward failed: invalid output index=%d null=%d seq_len=%d add=%d all_seq=%d "
                      "gen_seq=%d\n",
                      static_cast<int>(outputIndex), o == nullptr ? 1 : 0, seqLen, static_cast<int>(mMeta->add),
                      mContext->all_seq_len, mContext->gen_seq_len);
            if (std::getenv("MNN_PIC_DECODE_DEBUG") != nullptr) {
                auto info = nullptr != o ? o->getInfo() : nullptr;
                std::fprintf(stderr,
                             "PIC forward debug invalid output index=%d null=%d info=%p seq_len=%d add=%d "
                             "all_seq=%d gen_seq=%d\n",
                             static_cast<int>(outputIndex), o == nullptr ? 1 : 0, info, seqLen,
                             static_cast<int>(mMeta->add), mContext->all_seq_len, mContext->gen_seq_len);
                std::fflush(stderr);
            }
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return outputs;
        }
    }
    if (outputs[0]->readMap<float>() == nullptr) {
        std::ostringstream err;
        err << "forwardRaw logits materialize failed seq_len=" << seqLen
            << " add=" << static_cast<int>(mMeta->add)
            << " all_seq=" << mContext->all_seq_len
            << " gen_seq=" << mContext->gen_seq_len;
        if (mConfig->paged_attention()) {
            auto paged = static_cast<PagedKVMeta*>(mMeta.get());
            err << " paged_max=" << paged->max_tokens
                << " request_capacity=" << paged->request_capacity
                << " logical_length=" << paged->logical_length
                << " previous=" << static_cast<int>(paged->previous)
                << " remove=" << static_cast<int>(paged->remove);
        }
        mLastError = err.str();
        MNN_ERROR("PIC forward failed: logits materialize failed, seq_len=%d add=%d all_seq=%d gen_seq=%d "
                  "paged_max=%d request_capacity=%d logical_length=%d\n",
                  seqLen, static_cast<int>(mMeta->add), mContext->all_seq_len, mContext->gen_seq_len,
                  mConfig->paged_attention() ? static_cast<PagedKVMeta*>(mMeta.get())->max_tokens : 0,
                  mConfig->paged_attention() ? static_cast<PagedKVMeta*>(mMeta.get())->request_capacity : 0,
                  mConfig->paged_attention() ? static_cast<PagedKVMeta*>(mMeta.get())->logical_length : 0);
        if (std::getenv("MNN_PIC_DECODE_DEBUG") != nullptr) {
            std::fprintf(stderr,
                         "PIC forward debug logits materialize failed seq_len=%d add=%d all_seq=%d gen_seq=%d\n",
                         seqLen, static_cast<int>(mMeta->add), mContext->all_seq_len, mContext->gen_seq_len);
            std::fflush(stderr);
        }
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return outputs;
    }
    if (!mAsync) {
        ((MNN::Tensor*)(outputs[0]->getTensor()))->wait(Tensor::MAP_TENSOR_READ, true);
    }
    mGenerateParam->input_embeds = hiddenState;
    mGenerateParam->outputs = outputs;

#if DEBUG_MODE == 3
    VARP logits = outputs[0];
    if(logits->getInfo()->dim[1] < 10 && logits->getInfo()->dim[1] >= 1) {
        for (int j = 0; j < logits->getInfo()->dim[1]; j++) {
            {
                int length = hiddenState->getInfo()->dim[2];
                float total = 0.0;
                float max_ = std::numeric_limits<float>::lowest();
                float min_ = std::numeric_limits<float>::max();
                for (int i = 0; i < length; i++) {
                    int index = j * length + i;
                    float temp = hiddenState->readMap<float>()[index];
                    total += temp;
                    max_ = fmax(max_, temp);
                    min_ = fmin(min_, temp);
                }
                MNN_PRINT("\nhiddenState statistic value:%6f, %6f, %6f\n", total, max_, min_);
            }

            {
                int length = mask->getInfo()->dim[3];
                float total = 0.0;
                float max_ = std::numeric_limits<float>::lowest();
                float min_ = std::numeric_limits<float>::max();
                for (int i = 0; i < length; i++) {
                    int index = j * length + i;
                    float temp = mask->readMap<float>()[index];
                    total += (temp / length);
                    max_ = fmax(max_, temp);
                    min_ = fmin(min_, temp);
                }
                MNN_PRINT("mask statistic value:%6f, %6f, %6f\n", total, max_, min_);
            }
            MNN_PRINT("position statistic value:%d\n", inputPos->readMap<int>()[j]);
            {
                int length = logits->getInfo()->dim[2];
                float total = 0.0;
                float max_ = std::numeric_limits<float>::lowest();
                float min_ = std::numeric_limits<float>::max();
                for (int i = 0; i < length; i++) {
                    int index = j * length + i;
                    float temp = logits->readMap<float>()[index];
                    total += temp;
                    max_ = fmax(max_, temp);
                    min_ = fmin(min_, temp);
                }
                auto ptr = logits->readMap<float>() + j * logits->getInfo()->dim[2];
                //            MNN_PRINT("\noutput data value:%6f %6f %6f %6f %6f\n", ptr[0], ptr[length/5], ptr[length/10], ptr[length/20], ptr[length/100]);
                MNN_PRINT("output statistic value:%6f, %6f, %6f\n", total, max_, min_);
            }
        }
    }
#endif
    if (mConfig->paged_attention()) {
        static_cast<PagedKVMeta*>(mMeta.get())->syncPaged();
    } else {
        mMeta->sync();
    }
    return outputs;
}

VARP Llm::forward(const std::vector<int>& input_ids, bool is_prefill) {
    MNN::Express::ExecutorScope s(mExecutor);
    auto hidden_states = embedding(input_ids);
    if(hidden_states == nullptr) {
        return nullptr;
    }
    return forward(hidden_states);
}
VARP Llm::forward(MNN::Express::VARP input_embeds) {
    MNN::Express::ExecutorScope s(mExecutor);
    int seq_len         = input_embeds->getInfo()->dim[mSeqLenIndex];
    auto out = forwardVec(input_embeds);
    if (out.empty()) {
        return nullptr;
    }
    auto logits = out[0];
    updateContext(seq_len, 1);
    return logits;
}

std::vector<VARP> Llm::forwardVec(const std::vector<int>& input_ids) {
    MNN::Express::ExecutorScope s(mExecutor);
    auto input_embeds = embedding(input_ids);
    auto outputs = forwardVec(input_embeds);
    return outputs;
}

std::vector<VARP> Llm::forwardVec(MNN::Express::VARP input_embeds) {
    CHECK_LLM_RUNNING_RET(mContext, std::vector<VARP>());
    MNN::Express::ExecutorScope s(mExecutor);
    
    int seq_len         = input_embeds->getInfo()->dim[mSeqLenIndex];
    // Prepare PLE extra arg
    Express::VARPS extraArgs;
    if (mPleInput.get()) {
        extraArgs.push_back(mPleInput);
    }
    if (0 == mBlockSize) {
        mMeta->add = seq_len;
        auto attention_mask = gen_attention_mask(seq_len);
        auto position_ids = gen_position_ids(seq_len);
        if (picRequestProfileEnabled()) {
            std::ostringstream os;
            os << "seq_len=" << seq_len
               << " add=" << static_cast<int>(mMeta->add)
               << " has_pic_budget=" << (mConfig->has_pic_recompute_budget() ? 1 : 0)
               << " paged_attention=" << (mConfig->paged_attention() ? 1 : 0);
            picRequestProfileLog("forward_raw_begin", 0, os.str());
        }
        const int64_t forwardRawStartUs = picRequestProfileEnabled() ? picRequestMonotonicUs() : 0;
        auto res = forwardRaw(input_embeds, attention_mask, position_ids, extraArgs);
        if (picRequestProfileEnabled()) {
            std::ostringstream os;
            os << "seq_len=" << seq_len
               << " outputs=" << res.size();
            picRequestProfileLog("forward_raw_end", picRequestMonotonicUs() - forwardRawStartUs, os.str());
        }
        return res;
    }
    // For decode can't support seq_len <= mBlockSize
    MNN_ASSERT(mContext->gen_seq_len <= 0 || seq_len <= mBlockSize);
    auto blockNumber = seq_len / mBlockSize;
    auto blockRemain = seq_len % mBlockSize;
    std::vector<VARP> logits;
    std::vector<VARP> embeddings;
    auto blockSize = mBlockSize;
    INTS sizeSplits;
    if (0 < blockNumber) {
        sizeSplits.resize(blockNumber);
        for (int i=0; i<blockNumber; ++i) {
            sizeSplits[i] = blockSize;
        }
        if (blockRemain > 0) {
            sizeSplits.emplace_back(blockRemain);
        }
    }
    if (sizeSplits.size() > 1) {
        embeddings = MNN::Express::_Split(input_embeds, sizeSplits);
    } else {
        embeddings = {input_embeds};
    }
    int addSize = blockSize;
    // Split PLE if needed
    std::vector<VARP> pleChunks;
    if (mPleInput.get() && sizeSplits.size() > 1) {
        pleChunks = MNN::Express::_Split(mPleInput, sizeSplits, 1); // split on seq_len dim (dim=1)
    } else if (mPleInput.get()) {
        pleChunks = {mPleInput};
    }
    for (int i=0; i<blockNumber; ++i) {
        logits.clear();
        mMeta->add = blockSize;
        auto embed = embeddings[i];
        auto attention_mask = gen_attention_mask(blockSize);
        auto position_ids = gen_position_ids(blockSize);
        Express::VARPS blockExtraArgs;
        if (!pleChunks.empty()) blockExtraArgs.push_back(pleChunks[i]);
        logits = forwardRaw(embed, attention_mask, position_ids, blockExtraArgs);
        if(logits.empty()) {
            return logits;
        }
        updateContext(blockSize, 0);
    }
    bool hasPad = false;
    if (blockRemain != 0) {
        logits.clear();
        mMeta->add = blockRemain;
        addSize = blockRemain;
        int forwardSize = blockRemain;
        input_embeds = embeddings[embeddings.size()-1];
        if (!mValidBlockSize.empty()) {
            forwardSize = mValidBlockSize[mValidBlockSize.size()-1];
            for (int j=mValidBlockSize.size()-2; j>=0; --j) {
                if (mValidBlockSize[j] < blockRemain) {
                    break;
                }
                forwardSize = mValidBlockSize[j];
            }
            if (blockRemain < forwardSize) {
                // Pad
                hasPad = true;
                auto dim = input_embeds->getInfo()->dim;
                dim[mSeqLenIndex] = forwardSize;
                auto newEmbed = _Input(dim, NCHW);
                ::memcpy(newEmbed->writeMap<void>(), input_embeds->readMap<void>(), input_embeds->getInfo()->size * sizeof(float));
                ::memset(newEmbed->writeMap<float>() + input_embeds->getInfo()->size, 0, (newEmbed->getInfo()->size - input_embeds->getInfo()->size) * sizeof(float));
                input_embeds = newEmbed;
            }
        }
        auto attention_mask = gen_attention_mask(forwardSize);
        auto position_ids = gen_position_ids(forwardSize);
        Express::VARPS remainExtraArgs;
        if (!pleChunks.empty()) remainExtraArgs.push_back(pleChunks.back());
        logits = forwardRaw(input_embeds, attention_mask, position_ids, remainExtraArgs);
        if(logits.empty()) {
            return logits;
        }
    }
    updateContext(-blockSize * blockNumber, 0);
    if (hasPad) {
        auto logitSize = logits[0]->getInfo()->dim[2];
        // encode
        mGenerateParam->validLogitStart = ((int)addSize - 1) * logitSize;
        mGenerateParam->validLogitSize = logitSize;
    }

    return logits;
}

void Llm::updateContext(int seq_len, int gen_len) {
    mContext->all_seq_len += seq_len;
    mContext->gen_seq_len += gen_len;
}

int Llm::sample(VARP logits, int offset, int size) {
    MNN::Express::ExecutorScope s(mExecutor);
    auto logitsShape = logits->getInfo()->dim;
    if (offset && size) {
        MNN_ASSERT(logits->getInfo()->size >= offset + size);
        logits = _Const(logits->readMap<float>() + offset, {size}, NHWC, halide_type_of<float>());
    }
    auto token_id = mSampler->sample(logits);
    return token_id;
}

void Llm::reset() {
    clearGenerateForwardState(mGenerateParam);
    clearModuleForwardCaches();
    mContext->output_tokens.clear();
    mContext->history_tokens.clear();
    mContext->all_seq_len = 0;
    mContext->gen_seq_len = 0;
    mContext->vision_us = 0;
    mContext->pixels_mp = 0.0f;
    mContext->audio_us = 0;
    mContext->audio_input_s = 0.0f;
    mMeta->remove = mMeta->previous;
    finishPagedRequestIfNeeded();
    mPicRecomputeBudget = nullptr;
    mDecodeForwardActive = false;
    clearPicDecodeRepair();
    mCachedPromptText.clear();
}

void Llm::generate_init(std::ostream* os, const char* end_with) {
    clearGenerateForwardState(mGenerateParam);
    // init status
    mLastError.clear();
    mContext->os = os;
    if (nullptr != end_with) {
        mContext->end_with = end_with;
    }
    if (!mContext->generate_str.empty()) {
        mContext->generate_str.clear();
    }
    mContext->gen_seq_len = 0;
    mContext->prefill_us  = 0;
    mContext->decode_us   = 0;
    mContext->current_token = -1;
    mContext->sample_us = 0;
    if (!mConfig->reuse_kv()) {
        mContext->all_seq_len = 0;
        mContext->history_tokens.clear();
        mMeta->remove = mMeta->previous;
    }
    mContext->output_tokens.clear();
    if(mContext->status != LlmStatus::NOT_LOADED) {
        mContext->status = LlmStatus::RUNNING;
    }
    beginPagedRequestIfNeeded();
}

size_t Llm::getCurrentHistory() const {
    return mMeta->previous;
}
void Llm::eraseHistory(size_t begin, size_t end) {
    if (0 == end) {
        end = mMeta->previous;
    }
    if (end > mMeta->previous || begin >= end) {
        MNN_ERROR("Invalid erase range history larger than current\n");
        return;
    }
    if (mMeta->remove != 0) {
        MNN_ERROR("MNN-LLM: erase history hasn't been executed by response, override erase info\n");
    }
    mMeta->remove = mMeta->previous - begin;
    int revertNumber = 0;
    if (end != mMeta->previous) {
        mMeta->reserveHost.resize(2);
        mMeta->reserve = mMeta->reserveHost.data();
        mMeta->n_reserve = 1;
        mMeta->reserve[0] = end - begin;
        mMeta->reserve[1] = mMeta->previous - end;
        revertNumber = mMeta->reserve[1];
    }
    mContext->all_seq_len = mMeta->previous - mMeta->remove + revertNumber;
    // FIXME: support history_tokens erease the tokens with correct position
    if(revertNumber == 0 && mMeta->remove <  mContext->history_tokens.size()){
        mContext->history_tokens.resize(mContext->history_tokens.size() - mMeta->remove);
    }
}

bool Llm::stoped() {
    return is_stop(mContext->current_token);
}

void Llm::generate(int max_token) {
    CHECK_LLM_RUNNING(mContext);
    MNN::Express::ExecutorScope s(mExecutor);
    if (max_token < 0) {
        max_token = mConfig->max_new_tokens();
    }
    beginPagedRequestIfNeeded(0, max_token);
    if (is_stop(mContext->current_token)) {
        finishPagedRequestIfNeeded();
        return;
    }
    mGenerateParam->max_new_tokens = max_token;
    const bool callerSkipNextLogitsOnLimit = mGenerateParam->skipNextLogitsOnLimit;
    mGenerationStrategy->generate(*mGenerateParam);
    mGenerateParam->skipNextLogitsOnLimit = callerSkipNextLogitsOnLimit;
    // MAX_TOKENS_FINISHED only means this chunk ended; generate(1) callers may continue the same request.
    const bool interrupted = mContext->status == LlmStatus::INTERNAL_ERROR ||
                             mContext->status == LlmStatus::TIMEOUT ||
                             mContext->status == LlmStatus::USER_CANCEL;
    if (max_token != 1 || stoped() || interrupted) {
        finishPagedRequestIfNeeded();
    }
}

bool Llm::prefill(const std::vector<int>& input_ids) {
    CHECK_LLM_RUNNING_RET(mContext, false);
    MNN::Express::ExecutorScope s(mExecutor);
    beginPagedRequestIfNeeded();

    bool passExecute = false;
    if(mPrefixCacheMode) {
        mCallIndex++;

        if(mCallIndex == 1) {
            passExecute = mIsPrefixFileExist;

            if(!mIsPrefixFileExist) {
                mMeta->file_name = mPrefixCacheFileName;
                mMeta->file_flag = KVMeta::PendingWrite;
            }
            mPrefixLength = input_ids.size();
        } else if(mCallIndex == 2) {
            if(mIsPrefixFileExist) {
                mMeta->file_name = mPrefixCacheFileName;
                mMeta->file_flag = KVMeta::PendingRead;
                mMeta->seqlen_in_disk = mPrefixLength;
            }
        }
    }

    mContext->history_tokens.insert(mContext->history_tokens.end(), input_ids.begin(), input_ids.end());
    if(!passExecute) {
        if (0 == mBlockSize || input_ids.size() <= mBlockSize) {
            const int64_t embeddingStartUs = picRequestProfileEnabled() ? picRequestMonotonicUs() : 0;
            auto hidden_states = embedding(input_ids);
            if(hidden_states == nullptr) {
                std::ostringstream err;
                err << "prefill embedding null tokens=" << input_ids.size()
                    << " status=" << static_cast<int>(mContext->status)
                    << " all_seq=" << mContext->all_seq_len;
                mLastError = err.str();
                MNN_ERROR("PIC prefill failed: embedding returned null, tokens=%d status=%d all_seq=%d\n",
                          static_cast<int>(input_ids.size()), static_cast<int>(mContext->status),
                          mContext->all_seq_len);
                return false;
            }
            if (picRequestProfileEnabled()) {
                std::ostringstream os;
                os << "tokens=" << input_ids.size();
                picRequestProfileLog("prefill_embedding", picRequestMonotonicUs() - embeddingStartUs, os.str());
            }
            const int64_t generateStartUs = picRequestProfileEnabled() ? picRequestMonotonicUs() : 0;
            generate(hidden_states, 0);
            if (picRequestProfileEnabled()) {
                std::ostringstream os;
                os << "tokens=" << input_ids.size()
                   << " status=" << static_cast<int>(mContext->status)
                   << " all_seq=" << mContext->all_seq_len;
                picRequestProfileLog("prefill_generate", picRequestMonotonicUs() - generateStartUs, os.str());
            }
            if (mContext->status == LlmStatus::INTERNAL_ERROR ||
                mContext->status == LlmStatus::TIMEOUT ||
                mContext->status == LlmStatus::USER_CANCEL) {
                if (mLastError.empty()) {
                    std::ostringstream err;
                    err << "prefill forward failed tokens=" << input_ids.size()
                        << " status=" << static_cast<int>(mContext->status)
                        << " all_seq=" << mContext->all_seq_len
                        << " prompt=" << mContext->prompt_len
                        << " prefill_us=" << static_cast<long long>(mContext->prefill_us);
                    mLastError = err.str();
                }
                MNN_ERROR("PIC prefill failed after forward, tokens=%d status=%d all_seq=%d prompt=%d prefill_us=%lld\n",
                          static_cast<int>(input_ids.size()), static_cast<int>(mContext->status),
                          mContext->all_seq_len, mContext->prompt_len,
                          static_cast<long long>(mContext->prefill_us));
                return false;
            }
            completePrefixWrite();
            mContext->prompt_len = static_cast<int>(input_ids.size());
            return true;
        }
        int total_size = static_cast<int>(input_ids.size());
        int loop_size = UP_DIV(total_size, mBlockSize);
        for (int i = 0; i < loop_size; i++) {
            auto start = i * mBlockSize;
            auto end = (i + 1) * mBlockSize;
            if (end >= total_size) {
                end = total_size;
            }
            std::vector<int> chunk_ids(input_ids.begin() + start, input_ids.begin() + end);
            auto input_embeds = embedding(chunk_ids);
            if(input_embeds == nullptr) {
                std::ostringstream err;
                err << "prefill chunk embedding null chunk=" << (i + 1) << "/" << loop_size
                    << " tokens=" << chunk_ids.size()
                    << " status=" << static_cast<int>(mContext->status)
                    << " all_seq=" << mContext->all_seq_len;
                mLastError = err.str();
                MNN_ERROR("PIC prefill failed: chunk embedding returned null, chunk=%d/%d tokens=%d status=%d "
                          "all_seq=%d\n",
                          i + 1, loop_size, static_cast<int>(chunk_ids.size()), static_cast<int>(mContext->status),
                          mContext->all_seq_len);
                return false;
            }
            generate(input_embeds, 0);
            if (mContext->status == LlmStatus::INTERNAL_ERROR ||
                mContext->status == LlmStatus::TIMEOUT ||
                mContext->status == LlmStatus::USER_CANCEL) {
                if (mLastError.empty()) {
                    std::ostringstream err;
                    err << "prefill chunk forward failed chunk=" << (i + 1) << "/" << loop_size
                        << " tokens=" << chunk_ids.size()
                        << " status=" << static_cast<int>(mContext->status)
                        << " all_seq=" << mContext->all_seq_len
                        << " prompt=" << mContext->prompt_len
                        << " prefill_us=" << static_cast<long long>(mContext->prefill_us);
                    mLastError = err.str();
                }
                MNN_ERROR("PIC prefill failed after chunk forward, chunk=%d/%d tokens=%d status=%d all_seq=%d "
                          "prompt=%d prefill_us=%lld\n",
                          i + 1, loop_size, static_cast<int>(chunk_ids.size()), static_cast<int>(mContext->status),
                          mContext->all_seq_len, mContext->prompt_len,
                          static_cast<long long>(mContext->prefill_us));
                return false;
            }
        }
        completePrefixWrite();
    } else {
        updateContext(static_cast<int>(input_ids.size()), 0);
    }

    mContext->prompt_len = static_cast<int>(input_ids.size());
    return true;
}

std::vector<int> Llm::decode(int max_tokens) {
    CHECK_LLM_RUNNING_RET(mContext, std::vector<int>());
    if (max_tokens < 0) {
        max_tokens = mConfig->max_new_tokens();
    }
    const bool decodeDebug = std::getenv("MNN_PIC_DECODE_DEBUG") != nullptr;
    if (decodeDebug) {
        std::fprintf(stderr,
                     "PIC decode debug before max=%d status=%d current=%d output_tokens=%d gen_seq=%d all_seq=%d "
                     "param_outputs=%d valid_start=%d valid_size=%d\n",
                     max_tokens, static_cast<int>(mContext->status), mContext->current_token,
                     static_cast<int>(mContext->output_tokens.size()), mContext->gen_seq_len,
                     mContext->all_seq_len, static_cast<int>(mGenerateParam->outputs.size()),
                     mGenerateParam->validLogitStart, mGenerateParam->validLogitSize);
        std::fflush(stderr);
    }
    if (mConfig->paged_attention()) {
        auto paged = static_cast<PagedKVMeta*>(mMeta.get());
        if (paged != nullptr) {
            paged->finishSparseQuery();
            paged->finishCacheBlendScoring();
            paged->finishPicGraphActivePlan();
            if (paged->detachHydratedExternalSegmentsForDecode(mConfig->layer_nums()) && decodeDebug) {
                std::fprintf(stderr,
                             "PIC decode debug detached hydrated external source segments before decode, "
                             "logical=%d previous=%d\n",
                             paged->logical_length, static_cast<int>(paged->previous));
                std::fflush(stderr);
            }
        }
    }
    if (max_tokens > 0 && mContext->output_tokens.empty() && mContext->current_token >= 0 &&
        mTokenizer->is_stop(mContext->current_token)) {
        mContext->current_token = -1;
    }
    if (max_tokens > 0) {
        struct ScopedDecodeForwardFlag {
            bool& flag;
            bool old;
            explicit ScopedDecodeForwardFlag(bool& value) : flag(value), old(value) {
                flag = true;
            }
            ~ScopedDecodeForwardFlag() {
                flag = old;
            }
        } decodeForwardGuard(mDecodeForwardActive);
        const bool oldSkipNextLogitsOnLimit = mGenerateParam->skipNextLogitsOnLimit;
        mGenerateParam->skipNextLogitsOnLimit = true;
        generate(max_tokens);
        mGenerateParam->skipNextLogitsOnLimit = oldSkipNextLogitsOnLimit;
        finishPagedRequestIfNeeded();
    }
    if (decodeDebug) {
        std::fprintf(stderr, "PIC decode debug after status=%d current=%d output_tokens=%d gen_seq=%d all_seq=%d\n",
                     static_cast<int>(mContext->status), mContext->current_token,
                     static_cast<int>(mContext->output_tokens.size()), mContext->gen_seq_len,
                     mContext->all_seq_len);
        std::fflush(stderr);
    }
    return mContext->output_tokens;
}

std::vector<int> Llm::generate(const std::vector<int>& input_ids, int max_tokens) {
    CHECK_LLM_RUNNING_RET(mContext, std::vector<int>());
    MNN::Express::ExecutorScope s(mExecutor);
    if (max_tokens < 0) {
        max_tokens = mConfig->max_new_tokens();
    }
    bool startedPagedRequest = beginPagedRequestIfNeeded(static_cast<int>(input_ids.size()), max_tokens);

    bool passExecute = false;
    if(mPrefixCacheMode) {
        mCallIndex++;

        // first time execute generate function
        if(mCallIndex == 1) {
            passExecute = mIsPrefixFileExist;

            if(!mIsPrefixFileExist) {
                // save prefix kvcache file
                mMeta->file_name = mPrefixCacheFileName;
                mMeta->file_flag = KVMeta::PendingWrite; // write
            } else {
                // first time and cachefile exist, pass this time
            }
            mPrefixLength = input_ids.size();
        }
        // second time execute generate function
        else if(mCallIndex == 2) {
            // second time and cachefile exist, load prefix file
            if(mIsPrefixFileExist) {
                mMeta->file_name = mPrefixCacheFileName;
                mMeta->file_flag = KVMeta::PendingRead; // read
                mMeta->seqlen_in_disk = mPrefixLength; // set_length
            }
        }
    }

    mContext->history_tokens.insert(mContext->history_tokens.end(), input_ids.begin(), input_ids.end()); // push to history_ids_
    if(!passExecute) {
        if (0 == mBlockSize || input_ids.size() <= mBlockSize) {
            auto hidden_states = embedding(input_ids);
            if(hidden_states == nullptr) {
                return {};
            }
            auto result = generate(hidden_states, max_tokens);
            completePrefixWrite();
            if (startedPagedRequest && max_tokens != 0) {
                finishPagedRequestIfNeeded();
            }
            return result;
        }
        int total_size = (int)input_ids.size();
        int loop_size = UP_DIV(total_size, mBlockSize);
        for (int i = 0; i < loop_size; i++) {
            auto start = i * mBlockSize;
            auto end = (i+1) * mBlockSize;
            if (end >= total_size) {
                end = total_size;
            }
            std::vector<int> chunk_ids(input_ids.begin() + start, input_ids.begin() + end);
            auto input_embeds = embedding(chunk_ids);
            if(input_embeds == nullptr) {
                if (startedPagedRequest) {
                    finishPagedRequestIfNeeded();
                }
                return {};
            }
            generate(input_embeds, 0);
        }
        completePrefixWrite();
    } else {
        // update states
        updateContext((int)input_ids.size(), 0);
    }

    generate(max_tokens);
    mContext->prompt_len = static_cast<int>(input_ids.size());
    if (startedPagedRequest && max_tokens != 0) {
        finishPagedRequestIfNeeded();
    }
    return mContext->output_tokens;
}

std::string Llm::apply_chat_template(const std::string& user_content) const {
    return mTokenizer->apply_chat_template(user_content, mConfig->system_prompt());
}

std::string Llm::apply_chat_template(const ChatMessages& chat_prompts) const {
    return mTokenizer->apply_chat_template(chat_prompts, true);
}

std::string Llm::apply_chat_template(const ChatMessages& chat_prompts, bool add_generation_prompt) const {
    return mTokenizer->apply_chat_template(chat_prompts, add_generation_prompt);
}

std::vector<int> Llm::tokenizer_encode(const std::string& user_content) {
    return mTokenizer->encode(user_content);
}

std::vector<int> Llm::tokenizer_encode(const MultimodalPrompt& multimodal_input) {
    return mTokenizer->encode(multimodal_input.prompt_template);
}

void Llm::response(const MultimodalPrompt& multimodal_input,
                   std::ostream* os, const char* end_with, int max_new_tokens) {
    CHECK_LLM_RUNNING(mContext);
    MNN::Express::ExecutorScope s(mExecutor);
    auto multimodal_input_copy = multimodal_input;
    if (mConfig->use_template()) {
        multimodal_input_copy.prompt_template = apply_chat_template(multimodal_input_copy.prompt_template);
    }
    std::vector<int> input_ids = tokenizer_encode(multimodal_input_copy);
    response(input_ids, os, end_with, max_new_tokens);
}

std::vector<int> Llm::generate(MNN::Express::VARP input_embeds, int max_tokens) {
    CHECK_LLM_RUNNING_RET(mContext, std::vector<int>());
    MNN::Express::ExecutorScope s(mExecutor);
    if (max_tokens < 0) {
        max_tokens = mConfig->max_new_tokens();
    }
    int seqLen = input_embeds->getInfo()->dim[mSeqLenIndex];
    bool startedPagedRequest = beginPagedRequestIfNeeded(seqLen, max_tokens);
    mContext->prompt_len = seqLen;

    Timer _t;
    auto outputs = forwardVec(input_embeds);
    if(mGenerateParam->outputs.size() < 1) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        if (startedPagedRequest) {
            finishPagedRequestIfNeeded();
        }
        return {};
    }
    updateContext(seqLen, 0);
    mContext->prefill_us += _t.durationInUs();
    MNN::Express::ExecutorScope::Current()->gc(); // after prefill

    // prefix cache mode and response second time
    if(mPrefixCacheMode && mCallIndex == 2) {
        if(mIsPrefixFileExist) {
            // when cachefile exist, after second time prefill, updata previous length
            mMeta->previous += mMeta->seqlen_in_disk;
        }
        // recover meta status
        mMeta->seqlen_in_disk = 0;
        mMeta->file_name = "";
        mMeta->file_flag = KVMeta::NoChange;
        mMeta->layer_index = 0;
        // recover normal mode
        mPrefixCacheMode = false;
    }


#if DEBUG_MODE == 3
    {
        std::ofstream outFile("input_embeds.txt");
        auto temp = input_embeds->readMap<float>();
        for (size_t i = 0; i < input_embeds->getInfo()->size; ++i) {
            outFile << temp[i] << " "; // 每个数字后加空格
        }
        outFile.close();
    }
    {
        std::ofstream outFile("logits.txt");
        auto temp = mGenerateParam->outputs[0]->readMap<float>();
        for (size_t i = 0; i < mGenerateParam->outputs[0]->getInfo()->size; ++i) {
            outFile << temp[i] << " "; // 每个数字后加空格
        }
        outFile.close();
    }
#endif

    // check timeout after prefill
    if (mGenerateParam->timeout_ms > 0 && mContext->prefill_us / 1000 >= mGenerateParam->timeout_ms) {
        mContext->status = LlmStatus::TIMEOUT;
        if (startedPagedRequest) {
            finishPagedRequestIfNeeded();
        }
        return mContext->output_tokens;
    }
    // call generation function
    if (0 < max_tokens) {
        mGenerateParam->max_new_tokens = max_tokens;
        mGenerateParam->skipNextLogitsOnLimit = true;
        mGenerationStrategy->generate(*mGenerateParam);
        mGenerateParam->skipNextLogitsOnLimit = false;
    }
    if (startedPagedRequest && max_tokens != 0) {
        finishPagedRequestIfNeeded();
    }
    return mContext->output_tokens;
}

void Llm::response(const std::vector<int>& input_ids, std::ostream* os, const char* end_with, int max_new_tokens) {
    MNN::Express::ExecutorScope s(mExecutor);
    if (!end_with) { end_with = "\n"; }
    generate_init(os, end_with);
    CHECK_LLM_RUNNING(mContext);
    generate(input_ids, max_new_tokens);
    if (max_new_tokens != 0 || stoped() || mContext->status != LlmStatus::RUNNING) {
        finishPagedRequestIfNeeded();
    }
}

void Llm::response(MNN::Express::VARP input_embeds, std::ostream* os, const char* end_with, int max_new_tokens) {
    MNN::Express::ExecutorScope s(mExecutor);
    if (!end_with) { end_with = "\n"; }
    generate_init(os, end_with);
    CHECK_LLM_RUNNING(mContext);
    generate(input_embeds, max_new_tokens);
    if (max_new_tokens != 0 || stoped() || mContext->status != LlmStatus::RUNNING) {
        finishPagedRequestIfNeeded();
    }
}

void Llm::response(const std::string& user_content, std::ostream* os, const char* end_with, int max_new_tokens) {
    CHECK_LLM_RUNNING(mContext);
    MNN::Express::ExecutorScope s(mExecutor);
    auto prompt = user_content;
    if (mConfig->use_template()) {
        prompt = apply_chat_template(user_content);
        if (prompt.empty()) {
            prompt = user_content;
        }
    }
    std::vector<int> input_ids = tokenizer_encode(prompt);
    response(input_ids, os, end_with, max_new_tokens);
}

void Llm::response(const ChatMessages& chat_prompts, std::ostream* os, const char* end_with, int max_new_tokens) {
    CHECK_LLM_RUNNING(mContext);
    MNN::Express::ExecutorScope s(mExecutor);
    if (chat_prompts.empty()) {
        return;
    }
    auto prompt = apply_chat_template(chat_prompts);

    // Prompt cache: compare current prompt text against the previous turn's to
    // find the common prefix, then only tokenize and prefill the new suffix.
    if (mConfig->prompt_cache() && !mCachedPromptText.empty()) {
        // Use add_generation_prompt=false for comparison to avoid enable_thinking
        // asymmetry: the template adds <think> to the LAST assistant message only
        // when true. Using false renders all messages consistently.
        auto prompt_for_compare = mTokenizer->apply_chat_template(chat_prompts, false);
        size_t text_common = 0;
        size_t text_max = std::min(mCachedPromptText.size(), prompt_for_compare.size());
        while (text_common < text_max && mCachedPromptText[text_common] == prompt_for_compare[text_common])
            text_common++;

        // If history was trimmed (text_common < cached), fall back to full
        // re-prefill rather than conditioning on stale KV entries.
        if (text_common < mCachedPromptText.size()) {
            // History was trimmed — clear all stale state and do full re-prefill.
            mCachedPromptText.clear();
            mContext->all_seq_len = 0;
            mContext->history_tokens.clear();
            mMeta->remove = mMeta->previous;
            std::vector<int> input_ids = tokenizer_encode(prompt);
            size_t history_before = input_ids.size(); // generate() pushes these first
            response(input_ids, os, end_with, max_new_tokens);
            if (mConfig->prompt_cache()) {
                updateCachedPromptText(chat_prompts, history_before);
            }
            return;
        }

        // Tokenize the full prompt and split at a token boundary rather than
        // tokenizing a text substring (which can land mid-token for BPE/UTF-8).
        std::vector<int> full_tokens = tokenizer_encode(prompt);
        // Detect prefix token count (BOS, etc.) by encoding an empty string.
        // tokenizer_encode("") returns only the prefix tokens that are
        // automatically prepended to every input (e.g., <|begin_of_text|>).
        // This count is used to skip prefix tokens when mapping text
        // positions to token indices, since they don't correspond to any
        // prompt text bytes. Assumption: prefix tokens are constant across
        // inputs for a given tokenizer configuration.
        size_t prefix_count = tokenizer_encode("").size();
        size_t char_pos = 0;
        size_t token_split = prefix_count; // start after prefix tokens
        for (size_t i = prefix_count; i < full_tokens.size(); i++) {
            std::string piece = tokenizer_decode(full_tokens[i]);
            if (char_pos + piece.size() > text_common)
                break;
            char_pos += piece.size();
            token_split++;
        }
        std::vector<int> delta(full_tokens.begin() + token_split, full_tokens.end());
        MNN_PRINT("[prompt_cache] cached=%d, compare=%d, common=%d, token_split=%d, delta=%d\n",
                  (int)mCachedPromptText.size(), (int)prompt_for_compare.size(), (int)text_common, (int)token_split,
                  (int)delta.size());

        // Save/restore KV state across generate_init — only the delta path
        // should preserve KV; other response() overloads clear as normal.
        // Also preserve mMeta->remove so that a pending eraseHistory() (from
        // a prior cancelled decode) is not silently cleared before sync().
        int saved_all_seq_len = mContext->all_seq_len;
        auto saved_history = std::move(mContext->history_tokens);
        size_t saved_previous = mMeta->previous;
        size_t saved_remove = mMeta->remove;
        generate_init(os, end_with);
        mContext->all_seq_len = saved_all_seq_len;
        mContext->history_tokens = std::move(saved_history);
        mMeta->previous = saved_previous;
        mMeta->remove = saved_remove;
        CHECK_LLM_RUNNING(mContext);
        // history_before must account for the delta tokens that generate()
        // pushes into history_tokens before decode starts, so that
        // updateCachedPromptText only sees the assistant response tokens.
        size_t history_before = mContext->history_tokens.size() + delta.size();
        generate(delta, max_new_tokens);

        // Update cache after generation so non-Android callers (llm_demo,
        // mls) don't need to call syncPromptCache() externally.
        updateCachedPromptText(chat_prompts, history_before);
    } else {
        // First turn or prompt_cache disabled: full tokenization, full prefill.
        // Clear any existing KV state before a full re-prefill. generate_init()
        // only clears KV when reuse_kv=false, so handle reuse_kv=true here.
        if (mContext->all_seq_len > 0) {
            mContext->all_seq_len = 0;
            mContext->history_tokens.clear();
            mMeta->remove = mMeta->previous;
        }
        std::vector<int> input_ids = tokenizer_encode(prompt);
        // history_before accounts for input_ids that generate() pushes first
        size_t history_before = input_ids.size();
        response(input_ids, os, end_with, max_new_tokens);
        if (mConfig->prompt_cache()) {
            updateCachedPromptText(chat_prompts, history_before);
        } else {
            // Cache text must not survive while caching is disabled —
            // if re-enabled later, stale text would mismatch the KV state
            // that was rebuilt from scratch on each turn while disabled.
            mCachedPromptText.clear();
        }
    }
    if (max_new_tokens != 0 || stoped() || mContext->status != LlmStatus::RUNNING) {
        finishPagedRequestIfNeeded();
    }
}

void Llm::updateCachedPromptText(const ChatMessages& chat_prompts, size_t history_before) {
    // Re-render the template with the assistant response included.
    // Response tokens = history_tokens[history_before:] — the entries added
    // during generate(). Using history_before (snapshot before generate) rather
    // than gen_seq_len because Eagle's gen_seq_len can be stale on the final
    // accepted batch (updateDraft is not called on the terminal iteration).
    // history_tokens is correct for ALL decode paths:
    // - ArGeneration, Lookahead, MTP, Eagle all write to it
    // - Eagle's step counter is only in output_tokens, not history_tokens
    // - mTokenizer->is_stop() avoids Llm::is_stop() TIMEOUT/CANCEL guard
    ChatMessages msgs_with_response(chat_prompts.begin(), chat_prompts.end());
    std::string response_text;
    for (size_t i = history_before; i < mContext->history_tokens.size(); i++) {
        int tok = mContext->history_tokens[i];
        if (mTokenizer->is_stop(tok))
            continue;
        response_text += tokenizer_decode(tok);
    }
    if (!response_text.empty()) {
        msgs_with_response.emplace_back("assistant", response_text);
    }
    mCachedPromptText = mTokenizer->apply_chat_template(msgs_with_response, false);
    MNN::Transformer::stripThinkBlocks(mCachedPromptText);
}

Llm::Llm(std::shared_ptr<LlmConfig> config) : mConfig(config) {
    MNN::BackendConfig backendConfig;
    mExecutor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
    mContext.reset(new LlmContext);
    if (mConfig->paged_attention()) {
        auto paged = std::make_shared<PagedKVMeta>();
        paged->max_tokens = mConfig->paged_kv_max_tokens();
        paged->full_causal_attention_mask =
            mConfig->attention_mask() == "float" && mConfig->attention_type() == "full";
        mMeta = paged;
    } else {
        mMeta.reset(new KVMeta);
    }
    mMeta->layer_nums = mConfig->layer_nums();
    mMeta->attn_scale = mConfig->attn_scale();
    mMeta->rope_theta = mConfig->rope_theta();
    mMeta->rope_dim = mConfig->rope_dim();
    mMeta->rope_type = mConfig->rope_type();
    mMeta->rope_scaling_factor = mConfig->rope_scaling_value("factor", 1.0f);
    mMeta->rope_scaling_low_freq_factor = mConfig->rope_scaling_value("low_freq_factor", 1.0f);
    mMeta->rope_scaling_high_freq_factor = mConfig->rope_scaling_value("high_freq_factor", 4.0f);
    mMeta->rope_scaling_original_max_position_embeddings =
        mConfig->rope_scaling_int_value("original_max_position_embeddings", 0);
    mMeta->max_position_embeddings = mConfig->max_position_embeddings();
    mGenerateParam.reset(new GenerationParams);
    mGenerateParam->timeout_ms = mConfig->timeout_ms();
}

Llm::~Llm() {
#if DEBUG_MODE == 1
    if (nullptr != gTimeTraceInfo) {
        gTimeTraceInfo->dump();
    }
#endif
    mGenerateParam.reset();
    mModule.reset();
    mDecodeModule.reset();
    mDecodeModulePool.clear();
    mCacheBlendScoreRuntimeManager.reset();
    mRuntimeManager.reset();
    mProcessorRuntimeManager.reset();
    mExecutor.reset();
}
int Llm::getOutputIndex(const std::string& name) const {
    if (mModulePool.empty()) {
        return -1;
    }
    auto info = mModulePool.begin()->second->getInfo();
    for (int i=0; i<info->outputNames.size(); ++i) {
        if (info->outputNames[i] == name) {
            return i;
        }
    }
    return -1;
}
std::vector<Express::VARP> Llm::getOutputs() const {
    return mGenerateParam->outputs;
}

bool Llm::setPrefixCacheFile(const std::string& filename, int flag) {
    mPrefixCacheFileName = filename;
    mCallIndex = 0;
    mPrefixCacheMode = true;

    const int configuredLayerNums = mConfig->layer_nums();
    int layersToCheck = configuredLayerNums;
    if (layersToCheck <= 0) {
        layersToCheck = 0;
        for (int i = 0;; ++i) {
            auto base = MNNFilePathConcat(mConfig->prefix_cache_path(), mPrefixCacheFileName) + "_" +
                        std::to_string(i);
            if (!MNNFileExist((base + "_sync.k").c_str()) || !MNNFileExist((base + "_sync.v").c_str())) {
                break;
            }
            ++layersToCheck;
        }
    }

    mIsPrefixFileExist = layersToCheck > 0;
    // check kvcache, validate file existence
    for(int i = 0; i < layersToCheck; i++) {
        auto k_file = MNNFilePathConcat(mConfig->prefix_cache_path(), mPrefixCacheFileName) + "_" + std::to_string(i) + "_sync.k";
        if(!MNNFileExist(k_file.c_str())) {
            mIsPrefixFileExist = false;
            break;
        }
        auto v_file = MNNFilePathConcat(mConfig->prefix_cache_path(), mPrefixCacheFileName) + "_" + std::to_string(i) + "_sync.v";
        if(!MNNFileExist(v_file.c_str())) {
            mIsPrefixFileExist = false;
            break;
        }
    }

    // Further check: the real .k/.v data files must exist, be non-empty, and have consistent
    // sizes across layers. This guards against cases where _sync markers remain but the
    // actual data files were deleted / truncated / partially written, which would otherwise
    // lead to a crash in CPUKVCacheManager::onAlloc (e.g. memset on a null mmap address).
    if (mIsPrefixFileExist) {
        size_t refKeySize = 0;
        size_t refValueSize = 0;
        for (int i = 0; i < layersToCheck; i++) {
            auto base = MNNFilePathConcat(mConfig->prefix_cache_path(), mPrefixCacheFileName) + "_" + std::to_string(i);
            auto k_data = base + ".k";
            auto v_data = base + ".v";
            if (!MNNFileExist(k_data.c_str()) || !MNNFileExist(v_data.c_str())) {
                MNN_PRINT("Prefix cache data file missing: %s or %s\n", k_data.c_str(), v_data.c_str());
                mIsPrefixFileExist = false;
                break;
            }
            auto kfd = MNNOpenFile(k_data.c_str(), MNN_FILE_READ);
            auto vfd = MNNOpenFile(v_data.c_str(), MNN_FILE_READ);
            size_t kSize = (kfd == INVALID_FILE) ? INVALID_SIZE : MNNGetFileSize(kfd);
            size_t vSize = (vfd == INVALID_FILE) ? INVALID_SIZE : MNNGetFileSize(vfd);
            if (kfd != INVALID_FILE)
                MNNCloseFile(kfd);
            if (vfd != INVALID_FILE)
                MNNCloseFile(vfd);
            if (kSize == INVALID_SIZE || kSize == 0 || vSize == INVALID_SIZE || vSize == 0) {
                MNN_PRINT("Prefix cache data file has invalid size: %s(%zu) %s(%zu)\n", k_data.c_str(), kSize,
                          v_data.c_str(), vSize);
                mIsPrefixFileExist = false;
                break;
            }
            if (i == 0) {
                refKeySize = kSize;
                refValueSize = vSize;
            } else if (kSize != refKeySize || vSize != refValueSize) {
                // All layers share the same model shape, so their .k (and .v) files must
                // have identical sizes. Any mismatch means the cache directory has been
                // corrupted / partially overwritten and must be rebuilt.
                MNN_PRINT("Prefix cache size mismatch at layer %d: k=%zu(ref=%zu) v=%zu(ref=%zu)\n", i, kSize,
                          refKeySize, vSize, refValueSize);
                mIsPrefixFileExist = false;
                break;
            }
        }
    }

    // If the cache is deemed unusable (missing / truncated / size-mismatched /
    // partially written), wipe all related files under prefix_cache_path for this
    // filename so that the subsequent PendingWrite path starts from a clean slate.
    // This is especially important for the _sync.k / _sync.v markers, which would
    // otherwise remain and mislead the next run into thinking the cache is valid.
    // MNNRemoveFile is a no-op on non-existent files, so this is safe to run
    // unconditionally whenever mIsPrefixFileExist is false.
    if (!mIsPrefixFileExist) {
        int layersToClean = configuredLayerNums > 0 ? configuredLayerNums : layersToCheck;
        for (int i = 0; i < layersToClean; i++) {
            auto base = MNNFilePathConcat(mConfig->prefix_cache_path(), mPrefixCacheFileName) + "_" + std::to_string(i);
            MNNRemoveFile((base + ".k").c_str());
            MNNRemoveFile((base + ".v").c_str());
            MNNRemoveFile((base + "_sync.k").c_str());
            MNNRemoveFile((base + "_sync.v").c_str());
        }
        MNN_PRINT("Prefix cache unusable, cleaned up stale files under %s for %s\n",
                  mConfig->prefix_cache_path().c_str(), mPrefixCacheFileName.c_str());
    }

    return mIsPrefixFileExist;
}

void Llm::clearPrefixCacheFile() {
    mPrefixCacheMode = false;
    mPrefixCacheFileName.clear();
    mCallIndex = 0;
    mPrefixLength = 0;
    mIsPrefixFileExist = false;
    mMeta->file_flag = KVMeta::NoChange;
    mMeta->file_name = "";
    mMeta->seqlen_in_disk = 0;
    mMeta->layer_index = 0;
}

bool Llm::beginTextCacheExport(const std::string& filename) {
    if (filename.empty()) {
        return false;
    }
    clearPrefixCacheFile();
    mPrefixCacheMode = false;
    mPrefixCacheFileName.clear();
    mCallIndex = 0;
    mPrefixLength = 0;
    mIsPrefixFileExist = false;
    mMeta->file_name = filename;
    mMeta->file_flag = KVMeta::PendingWrite;
    mMeta->seqlen_in_disk = 0;
    mMeta->layer_index = 0;
    return true;
}

void Llm::clearTextCacheExport() {
    if (mMeta->file_flag == KVMeta::PendingWrite) {
        mMeta->file_flag = KVMeta::NoChange;
        mMeta->file_name = "";
        mMeta->seqlen_in_disk = 0;
        mMeta->layer_index = 0;
    }
}

void Llm::completePrefixWrite() {
    if (!mPrefixCacheMode || mCallIndex != 1 || mIsPrefixFileExist) {
        return;
    }
    mMeta->file_flag = KVMeta::NoChange;
    mMeta->file_name = "";
    mMeta->layer_index = 0;
    // Create sync files to mark prefix cache as valid
    auto prefixDir = mConfig->prefix_cache_path();
    int layersToMark = mConfig->layer_nums();
    if (layersToMark <= 0) {
        layersToMark = 0;
        for (int i = 0;; ++i) {
            auto base = MNNFilePathConcat(prefixDir, mPrefixCacheFileName) + "_" + std::to_string(i);
            if (!MNNFileExist((base + ".k").c_str()) || !MNNFileExist((base + ".v").c_str())) {
                break;
            }
            ++layersToMark;
        }
    }
    for (int i = 0; i < layersToMark; i++) {
        auto base = MNNFilePathConcat(prefixDir, mPrefixCacheFileName) + "_" + std::to_string(i);
        auto k_file = base + ".k";
        if (MNNFileExist(k_file.c_str())) {
            auto k_sync = base + "_sync.k";
            auto fd = MNNCreateFile(k_sync.c_str());
            if (fd != INVALID_FILE) { MNNCloseFile(fd); }
        }
        auto v_file = base + ".v";
        if (MNNFileExist(v_file.c_str())) {
            auto v_sync = base + "_sync.v";
            auto fd = MNNCreateFile(v_sync.c_str());
            if (fd != INVALID_FILE) { MNNCloseFile(fd); }
        }
    }
}

bool Llm::reuse_kv() { return mConfig->reuse_kv(); }

void Llm::syncPromptCache(const ChatMessages& chat_prompts) {
    if (!mConfig->prompt_cache() || chat_prompts.empty())
        return;
    // Override cached text with caller-provided messages (e.g. after
    // deleteThinkPart processing in Android). Uses the same rendering
    // and <think> stripping as updateCachedPromptText.
    mCachedPromptText = mTokenizer->apply_chat_template(chat_prompts, false);
    MNN::Transformer::stripThinkBlocks(mCachedPromptText);
}

static inline bool needNewVar(VARP var, int axis, int seq_len, int kv_seq_len = 0) {
    if (var == nullptr) {
        return true;
    }
    if (var->getInfo()->dim[axis] != seq_len) {
        return true;
    }
    if (kv_seq_len != 0 && var->getInfo()->dim[axis + 1] != kv_seq_len) {
        return true;
    }
    return false;
}

VARP Llm::embedding(const std::vector<int>& input_ids) {
    MNN::Express::ExecutorScope s(mExecutor);
    AUTOTIME;
    int hidden_size = mConfig->hidden_size();
    int seq_len = static_cast<int>(input_ids.size());

    VARP res = _Input({seq_len, 1, hidden_size}, NCHW);
    // disk embedding to save memory
    mDiskEmbedding->embedding(input_ids, res->writeMap<float>());
    // scale_emb is in ONNX model, not here. C++ passes raw embedding.

    // PLE embedding lookup (gemma4)
    // mPleInput may be pre-set by Omni::embedding for multimodal prefill;
    // update only if null or if seq_len matches (decode single token)
    if (mPleEmbedding && (!mPleInput.get() || seq_len == 1)) {
        int ple_dim = mConfig->ple_embed_dim();
        float ple_scale = mConfig->ple_embed_scale();
        mPleInput = _Input({1, seq_len, ple_dim}, NCHW);
        mPleEmbedding->embedding(input_ids, mPleInput->writeMap<float>());
        if (ple_scale != 1.0f) {
            mPleInput = mPleInput * _Scalar<float>(ple_scale);
        }
    }
    return res;
}

std::string Llm::tokenizer_decode(int id) {
    std::string word = mTokenizer->decode(id);
    // Fix utf-8 garbled characters
    if (word.length() == 6 && word[0] == '<' && word[word.length() - 1] == '>' && word[1] == '0' && word[2] == 'x') {
        int num = std::stoi(word.substr(3, 2), nullptr, 16);
        word    = static_cast<char>(num);
    }
    return word;
}

VARP Llm::gen_attention_mask(int seq_len) {
    MNN::Express::ExecutorScope s(mExecutor);
    if (mConfig->paged_attention()) {
        auto paged = static_cast<PagedKVMeta*>(mMeta.get());
        if (paged != nullptr && paged->sparse_query_active && !mConfig->has_pic_recompute_budget()) {
            attentionMask = _Input({}, NCHW, halide_type_of<float>());
            auto ptr = attentionMask->writeMap<float>();
            ptr[0] = 0.0f;
            return attentionMask;
        }
    }
    int kv_seq_len = mContext->all_seq_len + seq_len;
    if (mConfig->attention_mask() == "float") {
        // full and sliding mix, using normal mask
        if (mConfig->attention_type() == "mix") {
            const int sliding_window = mConfig->sliding_window();
            // mix attention mask
            attentionMask = _Input({2, 1, 1, seq_len, kv_seq_len}, NCHW, halide_type_of<float>());
            auto full_attn_ptr = attentionMask->writeMap<float>();
            // full attn mask
            for (int i = 0; i < seq_len; i++) {
                const int query_pos = i + (kv_seq_len - seq_len);
                for (int j = 0; j < kv_seq_len; j++) {
                    if (j > query_pos) {
                        full_attn_ptr[kv_seq_len * i + j] = std::numeric_limits<float>::lowest();
                    } else {
                        full_attn_ptr[kv_seq_len * i + j] = 0.0f;
                    }
                }
            }
            // sliding attn mask
            auto sliding_attn_ptr = full_attn_ptr + seq_len * kv_seq_len;
            const int query_pos_offset = kv_seq_len - seq_len;
            for (int i = 0; i < seq_len; i++) {
                const int query_pos = i + query_pos_offset;
                for (int j = 0; j < kv_seq_len; j++) {
                    const int key_pos = j;
                    bool is_allowed = (key_pos <= query_pos) && (key_pos > query_pos - sliding_window);
                    if (is_allowed) {
                        sliding_attn_ptr[kv_seq_len * i + j] = 0.0f;
                    } else {
                        sliding_attn_ptr[kv_seq_len * i + j] = std::numeric_limits<float>::lowest();
                    }
                }
            }
            return attentionMask;
        }
        // Use square mask just for new generation token, save memory of attention mask
        bool useFullPagedExternalMask = false;
        if (mConfig->paged_attention()) {
            auto paged = static_cast<PagedKVMeta*>(mMeta.get());
            useFullPagedExternalMask = paged != nullptr && paged->request_active &&
                !paged->external_segments.empty() && !paged->sparse_query_active;
        }
        if (!useFullPagedExternalMask) {
            kv_seq_len = seq_len;
        }
        if (mAttentionMaskVarVec.size() > 0) {
            if(seq_len == 1 && !useFullPagedExternalMask) {
                return mAttentionMaskVarVec[0];
            }
            if (mAttentionMaskVarVec.size() > 1 && seq_len == mDraftLength + 1 && !useFullPagedExternalMask) {
                return mAttentionMaskVarVec[1];
            }
        }

        // Mask: lower triangular
       if (mConfig->backend_type() == "cpu" && mValidBlockSize.empty() && !mConfig->has_pic_recompute_budget()) { // Now only cpu supports using lower triangular to opt the attention performance
           attentionMask = _Input({}, NCHW, halide_type_of<float>());
           auto ptr = attentionMask->writeMap<float>();
           ptr[0] = 0;
       } else {
            attentionMask = _Input({1, 1, seq_len, kv_seq_len}, NCHW, halide_type_of<float>());
            auto ptr = attentionMask->writeMap<float>();
            const int queryOffset = kv_seq_len - seq_len;
            for (int i = 0; i < seq_len; i++) {
                const int queryPos = queryOffset + i;
                for (int j = 0; j < kv_seq_len; j++) {
                    ptr[kv_seq_len * i + j] = (j > queryPos) * std::numeric_limits<float>::lowest();
                }
            }
       }
        return attentionMask;
    } else {
        if (needNewVar(attentionMask, 2, seq_len, kv_seq_len)) {
            attentionMask = _Input({1, 1, seq_len, kv_seq_len}, NCHW, halide_type_of<int>());
        } else {
            return attentionMask;
        }
        auto ptr = attentionMask->writeMap<int>();
        if (mConfig->attention_mask() == "glm") {
            // chatglm
            for (int i = 0; i < seq_len * kv_seq_len; i++) {
                ptr[i] = 0;
            }
            if (seq_len > 1) {
                for (int i = 1; i < seq_len; i++) {
                    ptr[seq_len * i - 1] = 1;
                }
            }
        } else {
            bool is_glm2 = mConfig->attention_mask() == "glm2";
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < kv_seq_len; j++) {
                    int row              = i + mContext->all_seq_len;
                    ptr[seq_len * i + j] = is_glm2 ? j > row : j <= row;
                }
            }
        }
        return attentionMask;
    }
}

VARP Llm::gen_position_ids(int seq_len) {
    MNN::Express::ExecutorScope s(mExecutor);
    if (mConfig->attention_mask() == "glm") {
        // chatglm
        if (needNewVar(positionIds, 2, seq_len)) {
            positionIds = _Input({1, 2, seq_len}, NCHW, halide_type_of<int>());
        }
        auto ptr = positionIds->writeMap<int>();
        if (seq_len == 1) {
            ptr[0] = mContext->all_seq_len - mContext->gen_seq_len - 2;
            ptr[1] = mContext->gen_seq_len + 1;
        } else {
            for (int i = 0; i < seq_len - 1; i++) {
                ptr[i]           = i;
                ptr[seq_len + i] = 0;
            }
            ptr[seq_len - 1]     = seq_len - 2;
            ptr[2 * seq_len - 1] = 1;
        }
        return positionIds;
    } else {
        bool is_glm2 = mConfig->attention_mask() == "glm2";
        if (mConfig->paged_attention()) {
            auto paged = static_cast<PagedKVMeta*>(mMeta.get());
            if (paged != nullptr && paged->sparse_query_active && !mConfig->has_pic_recompute_budget()) {
                if (mConfig->is_mrope()) {
                    positionIds = _Input({3, seq_len}, NCHW, halide_type_of<int>());
                    auto ptr = positionIds->writeMap<int>();
                    for (int i = 0; i < seq_len; i++) {
                        int pos = paged->sparseLogicalIndex(i);
                        ptr[0 * seq_len + i] = pos;
                        ptr[1 * seq_len + i] = pos;
                        ptr[2 * seq_len + i] = pos;
                    }
                    return positionIds;
                }
                positionIds = _Input({1, seq_len}, NCHW, halide_type_of<int>());
                auto ptr = positionIds->writeMap<int>();
                for (int i = 0; i < seq_len; i++) {
                    ptr[i] = paged->sparseLogicalIndex(i);
                }
                return positionIds;
            }
        }
        if (seq_len == 1) {
            auto ptr = mPositionIdsVarVec[0]->writeMap<int>();
            ptr[0] = is_glm2 ? mContext->gen_seq_len : mContext->all_seq_len;
            if (mConfig->is_mrope()) {
                ptr[1] = ptr[0];
                ptr[2] = ptr[0];
            }
            return mPositionIdsVarVec[0];
        }
        if (mPositionIdsVarVec.size() > 1 && seq_len == mDraftLength + 1) {
            auto ptr = mPositionIdsVarVec[1]->writeMap<int>();
            for (int i = 0; i < seq_len; i++) {
                ptr[i] = i + mContext->all_seq_len;
            }
            return mPositionIdsVarVec[1];
        }

        if (mConfig->is_mrope()) {
            positionIds = _Input({3, seq_len}, NCHW, halide_type_of<int>());
            auto ptr = positionIds->writeMap<int>();
            for (int i = 0; i < seq_len; i++) {
                ptr[0 * seq_len + i] = i + mContext->all_seq_len;
                ptr[1 * seq_len + i] = i + mContext->all_seq_len;
                ptr[2 * seq_len + i] = i + mContext->all_seq_len;
            }
            return positionIds;
        }

        positionIds = _Input({1, seq_len}, NCHW, halide_type_of<int>());
        auto ptr = positionIds->writeMap<int>();
        if (seq_len == 1) {
            ptr[0] = is_glm2 ? mContext->gen_seq_len : mContext->all_seq_len;
        } else {
            for (int i = 0; i < seq_len; i++) {
                ptr[i] = i + mContext->all_seq_len;
            }
        }
        return positionIds;
    }
}

bool Llm::is_stop(int token_id) {
    CHECK_LLM_RUNNING_RET(mContext, true);
    bool stop = mTokenizer->is_stop(token_id);
    if (stop) {
        mContext->status = LlmStatus::NORMAL_FINISHED;
    }
    return stop;
}
} // namespace Transformer
} // namespace MNN
