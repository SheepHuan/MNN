//
//  ExecutionClassLogger.hpp
//  MNN
//

#ifndef ExecutionClassLogger_hpp
#define ExecutionClassLogger_hpp

#include "MNN_generated.h"
#include <MNN/MNNDefine.h>

namespace MNN {

class Execution;

MNN_PUBLIC void printExecutionClass(const char* backend, const Op* op, OpType dispatchType, const Execution* execution);

} // namespace MNN

#endif
