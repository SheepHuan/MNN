//
//  ExecutionClassLogger.cpp
//  MNN
//

#include "core/ExecutionClassLogger.hpp"

#include "core/Execution.hpp"
#include "core/Macro.h"

#include <string>
#if defined(__GXX_RTTI) || defined(__cpp_rtti) || defined(_CPPRTTI)
#include <typeinfo>
#if defined(__GNUG__) && !defined(MNN_BUILD_FOR_ANDROID)
#include <cxxabi.h>
#include <cstdlib>
#endif
#endif

namespace MNN {

static std::string executionClassName(const Execution* execution) {
    if (nullptr == execution) {
        return "nullptr";
    }
#if defined(__GXX_RTTI) || defined(__cpp_rtti) || defined(_CPPRTTI)
    const char* name = typeid(*execution).name();
#if defined(__GNUG__) && !defined(MNN_BUILD_FOR_ANDROID)
    int status = 0;
    char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    if (nullptr != demangled && 0 == status) {
        std::string result(demangled);
        std::free(demangled);
        return result;
    }
    if (nullptr != demangled) {
        std::free(demangled);
    }
#endif
    return name;
#else
    return "RTTI disabled";
#endif
}

void printExecutionClass(const char* backend, const Op* op, OpType dispatchType, const Execution* execution) {
    auto opName = nullptr != op && nullptr != op->name() ? op->name()->c_str() : "";
    auto opType = nullptr != op ? op->type() : OpType_Extra;
    auto className = executionClassName(execution);
    MNN_PRINT("[%sExecution] op=\"%s\", type=%s, dispatch=%s, execution=%s\n", backend, opName,
              EnumNameOpType(opType), EnumNameOpType(dispatchType), className.c_str());
}

} // namespace MNN
