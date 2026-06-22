//
//  ShapePicExtra.cpp
//  MNN
//

#include "shape/SizeComputer.hpp"
#include "core/TensorUtils.hpp"
#include <cstdlib>

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

class PicExtraSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        if (op == nullptr || op->main_type() != OpParameter_Extra || inputs.empty() || outputs.empty()) {
            return false;
        }
        auto extra = op->main_as_Extra();
        const std::string type = (extra != nullptr && extra->type() != nullptr) ? extra->type()->str() : "";
        if (type == "PicGateUpWeightOnly") {
            if (std::getenv("MNN_PIC_GRAPH_PROFILE") != nullptr) {
                MNN_PRINT("PicExtra shape type=%s input_dims=%d output_size=%zu input0=%d input1=%d input2=%d input3=%d\n",
                          type.c_str(), inputs[0]->dimensions(), outputs.size(),
                          inputs[0]->dimensions() > 0 ? inputs[0]->length(0) : -1,
                          inputs[0]->dimensions() > 1 ? inputs[0]->length(1) : -1,
                          inputs[0]->dimensions() > 2 ? inputs[0]->length(2) : -1,
                          inputs[0]->dimensions() > 3 ? inputs[0]->length(3) : -1);
            }
            if (outputs.size() != 2) {
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
        if (type == "PicPackedSiluMul") {
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
            if (inputs[0]->dimensions() == 4 && extraAttrInt(extra, "packed_input_nc4", 0) != 0) {
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
