//
//  ShapePicExtra.cpp
//  MNN
//

#include "shape/SizeComputer.hpp"
#include "core/TensorUtils.hpp"
#include <cstdlib>
#include <cstdio>
#include <string>

namespace MNN {

static int extraAttrInt(const Extra* extra, const char* key, int fallback = 0) {
    if (extra == nullptr || extra->attr() == nullptr || key == nullptr) {
        return fallback;
    }
    for (int i = 0; i < extra->attr()->size(); ++i) {
        auto attr = extra->attr()->GetAs<Attribute>(i);
        if (attr != nullptr && attr->key() != nullptr && attr->key()->str() == key) {
            return attr->i();
        }
    }
    return fallback;
}

static int positiveExtent(const Tensor* tensor, int axis) {
    if (tensor == nullptr || axis < 0 || axis >= tensor->buffer().dimensions) {
        return 1;
    }
    const int extent = tensor->buffer().dim[axis].extent;
    return extent > 0 ? extent : 1;
}

static int formattedTensorRows(const Tensor* tensor) {
    if (tensor == nullptr || tensor->buffer().dimensions <= 0) {
        return 1;
    }
    if (tensor->buffer().dimensions == 2) {
        return positiveExtent(tensor, 0);
    }
    if (tensor->buffer().dimensions == 1) {
        return 1;
    }
    const auto format = TensorUtils::getDescribe(tensor)->dimensionFormat;
    const int n = positiveExtent(tensor, 0);
    if (format == MNN_DATA_FORMAT_NHWC) {
        return n * positiveExtent(tensor, 1) * positiveExtent(tensor, 2);
    }
    int rows = n * positiveExtent(tensor, 2) * positiveExtent(tensor, 3);
    for (int i = 4; i < tensor->buffer().dimensions; ++i) {
        rows *= positiveExtent(tensor, i);
    }
    return rows;
}

static void logPicExtraOutput(const std::string& type, const Tensor* output) {
    if (std::getenv("MNN_PIC_GRAPH_PROFILE") == nullptr || output == nullptr) {
        return;
    }
    std::fprintf(stderr, "PicExtra shape output type=%s dims=%d shape=%d,%d,%d,%d format=%d\n",
                 type.c_str(), output->dimensions(),
                 output->dimensions() > 0 ? output->length(0) : -1,
                 output->dimensions() > 1 ? output->length(1) : -1,
                 output->dimensions() > 2 ? output->length(2) : -1,
                 output->dimensions() > 3 ? output->length(3) : -1,
                 static_cast<int>(TensorUtils::getDescribe(output)->dimensionFormat));
    std::fflush(stderr);
}

class PicExtraSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        if (op == nullptr || op->main_type() != OpParameter_Extra || inputs.empty() || outputs.empty()) {
            return false;
        }
        auto extra = op->main_as_Extra();
        const std::string type = (extra != nullptr && extra->type() != nullptr) ? extra->type()->str() : "";
        if (std::getenv("MNN_PIC_GRAPH_PROFILE") != nullptr) {
            std::fprintf(stderr, "PicExtra shape enter type=%s inputs=%zu outputs=%zu\n",
                         type.c_str(), inputs.size(), outputs.size());
            std::fflush(stderr);
        }
        if (type == "PicGateUpWeightOnly" || type == "PicGateUpSiluWeightOnly" ||
            type == "PicAdrenoGateUpSiluWeightOnly") {
            if (std::getenv("MNN_PIC_GRAPH_PROFILE") != nullptr) {
                MNN_PRINT("PicExtra shape type=%s input_dims=%d output_size=%zu input0=%d input1=%d input2=%d input3=%d\n",
                          type.c_str(), inputs[0]->dimensions(), outputs.size(),
                          inputs[0]->dimensions() > 0 ? inputs[0]->length(0) : -1,
                          inputs[0]->dimensions() > 1 ? inputs[0]->length(1) : -1,
                          inputs[0]->dimensions() > 2 ? inputs[0]->length(2) : -1,
                          inputs[0]->dimensions() > 3 ? inputs[0]->length(3) : -1);
            }
            const bool fusedSiluOutput = type == "PicGateUpSiluWeightOnly" ||
                                         type == "PicAdrenoGateUpSiluWeightOnly";
            if (outputs.size() != (fusedSiluOutput ? 1 : 2)) {
                return false;
            }
            const int oc = extraAttrInt(extra, "out_features", 0);
            if (oc <= 0) {
                return false;
            }
            for (auto output : outputs) {
                output->buffer().dimensions = 4;
                output->setLength(0, inputs[0]->dimensions() > 0 ? inputs[0]->length(0) : 1);
                output->setLength(1, oc);
                output->setLength(2, 1);
                output->setLength(3, 1);
                output->buffer().type = inputs[0]->getType();
                TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            }
            return true;
        }
        if (type == "PicAdrenoTinyMlpWeightOnly") {
            if (outputs.size() != 1) {
                return false;
            }
            const int oc = extraAttrInt(extra, "out_features", 0);
            if (oc <= 0) {
                return false;
            }
            outputs[0]->buffer().dimensions = 4;
            outputs[0]->setLength(0, inputs[0]->dimensions() > 0 ? inputs[0]->length(0) : 1);
            outputs[0]->setLength(1, oc);
            outputs[0]->setLength(2, 1);
            outputs[0]->setLength(3, 1);
            outputs[0]->buffer().type = inputs[0]->getType();
            TensorUtils::getDescribe(outputs[0])->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            logPicExtraOutput(type, outputs[0]);
            return true;
        }
        if (type == "PicPackedSiluMul" || type == "PicAdrenoPackedSiluMul") {
            if (outputs.size() != 1) {
                return false;
            }
            int oc = extraAttrInt(extra, "out_features", 0);
            if (oc <= 0 && inputs[0]->dimensions() > 0) {
                const int lastDim = inputs[0]->length(inputs[0]->dimensions() - 1);
                oc = lastDim > 0 ? lastDim / 2 : 0;
            }
            if (oc <= 0) {
                return false;
            }
            const bool packedNhwc = extraAttrInt(extra, "packed_input_nhwc", 0) != 0;
            if (inputs[0]->dimensions() == 4 && packedNhwc) {
                const int rows = inputs[0]->length(0) * inputs[0]->length(1) * inputs[0]->length(2);
                outputs[0]->buffer().dimensions = 3;
                outputs[0]->setLength(0, 1);
                outputs[0]->setLength(1, rows);
                outputs[0]->setLength(2, oc);
                TensorUtils::getDescribe(outputs[0])->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            } else if (inputs[0]->dimensions() == 3 && packedNhwc) {
                outputs[0]->buffer().dimensions = 3;
                outputs[0]->setLength(0, inputs[0]->length(0));
                outputs[0]->setLength(1, inputs[0]->length(1));
                outputs[0]->setLength(2, oc);
                TensorUtils::getDescribe(outputs[0])->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            } else if (inputs[0]->dimensions() == 4 && extraAttrInt(extra, "packed_input_nc4", 0) != 0) {
                outputs[0]->buffer().dimensions = 3;
                outputs[0]->setLength(0, 1);
                outputs[0]->setLength(1, inputs[0]->length(0));
                outputs[0]->setLength(2, oc);
                TensorUtils::getDescribe(outputs[0])->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            } else if (inputs[0]->dimensions() == 4) {
                outputs[0]->buffer().dimensions = 4;
                outputs[0]->setLength(0, inputs[0]->length(0));
                outputs[0]->setLength(1, oc);
                outputs[0]->setLength(2, inputs[0]->length(2));
                outputs[0]->setLength(3, inputs[0]->length(3));
                TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
            } else {
                TensorUtils::copyShape(inputs[0], outputs[0], true);
                outputs[0]->setLength(outputs[0]->dimensions() - 1, oc);
                TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
            }
            outputs[0]->buffer().type = inputs[0]->getType();
            logPicExtraOutput(type, outputs[0]);
            return true;
        }
        if (type == "PicSiluMulNhwc" || type == "PicAdrenoSiluMulNhwc") {
            if (std::getenv("MNN_PIC_GRAPH_PROFILE") != nullptr) {
                MNN_PRINT("PicExtra shape type=%s input_dims=%d output_size=%zu input0=%d input1=%d input2=%d input3=%d\n",
                          type.c_str(), inputs[0]->dimensions(), outputs.size(),
                          inputs[0]->dimensions() > 0 ? inputs[0]->length(0) : -1,
                          inputs[0]->dimensions() > 1 ? inputs[0]->length(1) : -1,
                          inputs[0]->dimensions() > 2 ? inputs[0]->length(2) : -1,
                          inputs[0]->dimensions() > 3 ? inputs[0]->length(3) : -1);
                std::fprintf(stderr, "PicExtra shape detail type=%s input0=%d,%d,%d,%d\n",
                             type.c_str(),
                             inputs[0]->dimensions() > 0 ? inputs[0]->length(0) : -1,
                             inputs[0]->dimensions() > 1 ? inputs[0]->length(1) : -1,
                             inputs[0]->dimensions() > 2 ? inputs[0]->length(2) : -1,
                             inputs[0]->dimensions() > 3 ? inputs[0]->length(3) : -1);
                std::fflush(stderr);
            }
            if (inputs.size() < 2 || outputs.size() != 1) {
                return false;
            }
            int oc = extraAttrInt(extra, "out_features", 0);
            if (oc <= 0) {
                if (inputs[0]->dimensions() == 4) {
                    oc = inputs[0]->length(1);
                } else if (inputs[0]->dimensions() > 0) {
                    oc = inputs[0]->length(inputs[0]->dimensions() - 1);
                }
            }
            if (oc <= 0) {
                return false;
            }
            if (type == "PicAdrenoSiluMulNhwc") {
                outputs[0]->buffer().dimensions = 4;
                outputs[0]->setLength(0, inputs[0]->dimensions() > 0 ? inputs[0]->length(0) : 1);
                outputs[0]->setLength(1, 1);
                outputs[0]->setLength(2, 1);
                outputs[0]->setLength(3, oc);
                TensorUtils::getDescribe(outputs[0])->dimensionFormat = MNN_DATA_FORMAT_NHWC;
            } else {
                outputs[0]->buffer().dimensions = 3;
                outputs[0]->setLength(0, 1);
                outputs[0]->setLength(1, inputs[0]->dimensions() > 0 ? inputs[0]->length(0) : 1);
                outputs[0]->setLength(2, oc);
                TensorUtils::getDescribe(outputs[0])->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            }
            outputs[0]->buffer().type = inputs[0]->getType();
            logPicExtraOutput(type, outputs[0]);
            return true;
        }
        if (type == "PicLinearNhwcWeightOnly" || type == "PicAdrenoLinearNhwcWeightOnly") {
            if (std::getenv("MNN_PIC_GRAPH_PROFILE") != nullptr) {
                MNN_PRINT("PicExtra shape type=%s input_dims=%d output_size=%zu input0=%d input1=%d input2=%d input3=%d\n",
                          type.c_str(), inputs[0]->dimensions(), outputs.size(),
                          inputs[0]->dimensions() > 0 ? inputs[0]->length(0) : -1,
                          inputs[0]->dimensions() > 1 ? inputs[0]->length(1) : -1,
                          inputs[0]->dimensions() > 2 ? inputs[0]->length(2) : -1,
                          inputs[0]->dimensions() > 3 ? inputs[0]->length(3) : -1);
                std::fprintf(stderr, "PicExtra shape detail type=%s input0=%d,%d,%d,%d\n",
                             type.c_str(),
                             inputs[0]->dimensions() > 0 ? inputs[0]->length(0) : -1,
                             inputs[0]->dimensions() > 1 ? inputs[0]->length(1) : -1,
                             inputs[0]->dimensions() > 2 ? inputs[0]->length(2) : -1,
                             inputs[0]->dimensions() > 3 ? inputs[0]->length(3) : -1);
                std::fflush(stderr);
            }
            if (outputs.size() != 1) {
                return false;
            }
            const int oc = extraAttrInt(extra, "out_features", 0);
            if (oc <= 0 || inputs[0]->dimensions() <= 0) {
                return false;
            }
            if (type == "PicLinearNhwcWeightOnly") {
                TensorUtils::copyShape(inputs[0], outputs[0], true);
                outputs[0]->setLength(outputs[0]->dimensions() - 1, oc);
                TensorUtils::getDescribe(outputs[0])->dimensionFormat = MNN_DATA_FORMAT_NHWC;
                outputs[0]->buffer().type = inputs[0]->getType();
                logPicExtraOutput(type, outputs[0]);
                return true;
            }
            int rows = 1;
            if (type == "PicAdrenoLinearNhwcWeightOnly" &&
                inputs[0]->dimensions() == 4 && inputs[0]->length(1) == 1 && inputs[0]->length(2) == 1) {
                rows = inputs[0]->length(0);
            } else if (inputs[0]->dimensions() == 3) {
                rows = inputs[0]->length(1);
            } else if (type == "PicAdrenoLinearNhwcWeightOnly" && inputs[0]->dimensions() == 2) {
                rows = inputs[0]->length(0);
            } else {
                rows = formattedTensorRows(inputs[0]);
            }
            outputs[0]->buffer().dimensions = 3;
            outputs[0]->setLength(0, 1);
            outputs[0]->setLength(1, rows);
            outputs[0]->setLength(2, oc);
            TensorUtils::getDescribe(outputs[0])->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            outputs[0]->buffer().type = inputs[0]->getType();
            logPicExtraOutput(type, outputs[0]);
            return true;
        }
        if (outputs.size() == 1) {
            TensorUtils::copyShape(inputs[0], outputs[0], true);
            return true;
        }
        return false;
    }
};

REGISTER_SHAPE(PicExtraSizeComputer, OpType_Extra);

} // namespace MNN
