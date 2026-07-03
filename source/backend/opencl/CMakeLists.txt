#add_custom_command(OUTPUT "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc"
#    COMMAND ${PYTHON_EXECUTABLE}
#    "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_codegen.py"
#    "${CMAKE_CURRENT_LIST_DIR}/execution/cl/"
#    "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc"
#    COMMENT "OpenCL Code Generation"
#)
#add_custom_target (MNN_CLCodeGen DEPENDS "${CMAKE_CURRENT_LIST_DIR}/execution/cl/")
file(GLOB_RECURSE MNN_OpenCL_SRC ${CMAKE_CURRENT_LIST_DIR}/*)

option(MNN_OPENCL_SIZE_CUT "Disable MNN OpenCL Buffer Opt" OFF)
option(MNN_OPENCL_PMC_PROFILE "Enable ARM libGPUCounters PMC profiling for OpenCL PIC kernels" OFF)
set(LIBGPUCOUNTERS_ROOT "" CACHE PATH "Install or build root for ARM libGPUCounters")
set(LIBGPUCOUNTERS_BUILD_ROOT "" CACHE PATH "Optional separate build root for ARM libGPUCounters libraries")

if (MNN_OPENCL_PMC_PROFILE)
    set(LIBGPUCOUNTERS_HINT_ROOTS
        ${LIBGPUCOUNTERS_ROOT}
        ${LIBGPUCOUNTERS_BUILD_ROOT}
        $ENV{LIBGPUCOUNTERS_ROOT}
        $ENV{LIBGPUCOUNTERS_BUILD_ROOT})
    find_path(LIBGPUCOUNTERS_HWCPIPE_INCLUDE_DIR
        NAMES hwcpipe/sampler.hpp
        HINTS ${LIBGPUCOUNTERS_HINT_ROOTS}
        PATH_SUFFIXES include hwcpipe/include)
    find_path(LIBGPUCOUNTERS_DEVICE_INCLUDE_DIR
        NAMES device/product_id.hpp
        HINTS ${LIBGPUCOUNTERS_HINT_ROOTS}
        PATH_SUFFIXES include backend/device/include)
    find_library(LIBGPUCOUNTERS_HWCPIPE_LIBRARY
        NAMES hwcpipe libhwcpipe
        HINTS ${LIBGPUCOUNTERS_HINT_ROOTS}
        PATH_SUFFIXES lib lib64 build build/hwcpipe hwcpipe)
    find_library(LIBGPUCOUNTERS_DEVICE_LIBRARY
        NAMES device libdevice
        HINTS ${LIBGPUCOUNTERS_HINT_ROOTS}
        PATH_SUFFIXES lib lib64 build build/backend/device backend/device)
    if (NOT LIBGPUCOUNTERS_HWCPIPE_INCLUDE_DIR OR NOT LIBGPUCOUNTERS_DEVICE_INCLUDE_DIR OR
        NOT LIBGPUCOUNTERS_HWCPIPE_LIBRARY OR NOT LIBGPUCOUNTERS_DEVICE_LIBRARY)
        message(FATAL_ERROR
            "MNN_OPENCL_PMC_PROFILE=ON requires ARM libGPUCounters. "
            "Set LIBGPUCOUNTERS_ROOT to the source/install root and, if needed, "
            "LIBGPUCOUNTERS_BUILD_ROOT to the build root containing libhwcpipe/libdevice.")
    endif()
    set(MNN_OPENCL_PMC_INCLUDE_DIRS
        ${LIBGPUCOUNTERS_HWCPIPE_INCLUDE_DIR}
        ${LIBGPUCOUNTERS_DEVICE_INCLUDE_DIR})
    set(MNN_OPENCL_PMC_LIBS
        ${LIBGPUCOUNTERS_HWCPIPE_LIBRARY}
        ${LIBGPUCOUNTERS_DEVICE_LIBRARY})
endif()

IF (MNN_GPU_TIME_PROFILE)
    add_definitions(-DENABLE_OPENCL_TIME_PROFILER)
ENDIF()

IF (MNN_OPENCL_SIZE_CUT)
    add_definitions(-DMNN_OPENCL_BUFFER_CLOSED)
ENDIF()


if (${CMAKE_SYSTEM_NAME} MATCHES "Android")
    add_definitions(-DMNN_USE_LIB_WRAPPER)
    add_definitions(-DMNN_OPENCL_SVM_ENABLE)
    add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=110)
else()
    if(${CMAKE_SYSTEM_NAME} MATCHES "Windows" OR ${CMAKE_SYSTEM_NAME} MATCHES "Linux" OR MNN_SUPPORT_INTEL_SUBGROUP)
        add_definitions(-DMNN_SUPPORT_INTEL_SUBGROUP)
    endif()
    if (MNN_USE_SYSTEM_LIB)
        find_package(OpenCL REQUIRED)
    else()
        add_definitions(-DMNN_USE_LIB_WRAPPER)
        add_definitions(-DMNN_OPENCL_SVM_ENABLE)
        add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=110)
    endif()
endif()
IF(MNN_SEP_BUILD)
    add_definitions(-DMNN_OPENCL_SEP_BUILD)
    if (MNN_BUILD_FOR_ANDROID_COMMAND)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../../../")
    endif()
    add_library(MNN_CL SHARED ${MNN_OpenCL_SRC} "${CMAKE_CURRENT_LIST_DIR}/execution/cl/")
    target_link_libraries(MNN_CL MNN ${OpenCL_LIBRARIES} ${MNN_OPENCL_PMC_LIBS})
ELSE()
    add_library(MNN_CL OBJECT ${MNN_OpenCL_SRC} "${CMAKE_CURRENT_LIST_DIR}/execution/cl/")
    set(MNN_OCL_LIBS ${OpenCL_LIBRARIES} ${MNN_OPENCL_PMC_LIBS} PARENT_SCOPE)
ENDIF()

target_include_directories(MNN_CL PRIVATE
    ${CMAKE_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_LIST_DIR}/schema/current
    ${CMAKE_SOURCE_DIR}/3rd_party/half
    ${MNN_OPENCL_PMC_INCLUDE_DIRS}
)

if (MNN_OPENCL_PMC_PROFILE)
    target_compile_definitions(MNN_CL PRIVATE MNN_OPENCL_PMC_PROFILE)
    target_compile_features(MNN_CL PRIVATE cxx_std_14)
endif()
