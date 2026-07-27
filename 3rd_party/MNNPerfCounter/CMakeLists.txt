cmake_minimum_required(VERSION 3.6)

add_library(MNNPerfCounter STATIC
    src/MNNPerfCounter.cpp
    src/A7xxPerfCounters.cpp
    upstream/hardware_perfcounter/lib/gpu/adreno/a5xx.c
    upstream/hardware_perfcounter/lib/gpu/adreno/a6xx.c
    upstream/hardware_perfcounter/lib/gpu/adreno/common.c
    upstream/hardware_perfcounter/lib/gpu/adreno/context.c
    upstream/hardware_perfcounter/lib/gpu/adreno/driver_ioctl.c
    upstream/hardware_perfcounter/lib/gpu/mali/bifrost.c
    upstream/hardware_perfcounter/lib/gpu/mali/common.c
    upstream/hardware_perfcounter/lib/gpu/mali/context.c
    upstream/hardware_perfcounter/lib/gpu/mali/driver_ioctl.c
    upstream/hardware_perfcounter/lib/gpu/mali/valhall.c
    upstream/libgpucounters/device/src/device/handle.cpp
    upstream/libgpucounters/device/src/device/hwcnt/backend_type.cpp
    upstream/libgpucounters/device/src/device/hwcnt/reader.cpp
    upstream/libgpucounters/device/src/device/hwcnt/sampler/detail/backend.cpp
    upstream/libgpucounters/device/src/device/hwcnt/sampler/kinstr_prfcnt/block_index_remap.cpp
    upstream/libgpucounters/device/src/device/hwcnt/sampler/kinstr_prfcnt/enum_info_parser.cpp
    upstream/libgpucounters/device/src/device/hwcnt/sampler/kinstr_prfcnt/metadata_parser.cpp
    upstream/libgpucounters/device/src/device/instance.cpp
    upstream/libgpucounters/device/src/device/num_exec_engines.cpp
    upstream/libgpucounters/device/src/device/product_id.cpp
    upstream/libgpucounters/hwcpipe/src/error.cpp
    upstream/libgpucounters/hwcpipe/src/hwcpipe/detail/counter_database.cpp
    upstream/libgpucounters/hwcpipe/src/hwcpipe/all_gpu_counters.cpp
    upstream/libgpucounters/hwcpipe/src/hwcpipe/counter_metadata.cpp
    upstream/libgpucounters/hwcpipe/src/hwcpipe/derived_functions.cpp)

# Optional NVIDIA CUPTI Range Profiler backend. Enabled when the consuming
# project passes MNN_PERFCOUNTER_HAS_CUDA=ON and provides CUDA_INCLUDE_DIRS
# (pointing at a CUDA toolkit). The NvRangeProfiler.cpp TU must be added
# explicitly here so the static lib carries the NVPA/CUPTI symbols.
option(MNN_PERFCOUNTER_HAS_CUDA "Enable NVIDIA CUPTI Range Profiler backend" OFF)
if(MNN_PERFCOUNTER_HAS_CUDA)
    target_sources(MNNPerfCounter PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/NvRangeProfiler.cpp)
    target_compile_definitions(MNNPerfCounter PUBLIC MNN_PERFCOUNTER_HAS_CUDA)
    target_include_directories(MNNPerfCounter PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(MNNPerfCounter PUBLIC ${CUDA_LIBRARIES})
    find_library(NV_CUDA_DRIVER_LIB cuda HINTS ${CUDA_INCLUDE_DIRS}/../lib64 ${CUDA_INCLUDE_DIRS}/../lib/stubs)
    if(NV_CUDA_DRIVER_LIB)
        target_link_libraries(MNNPerfCounter PUBLIC ${NV_CUDA_DRIVER_LIB})
    endif()
    # libcupti + nvperf_* ship in <cuda>/lib64. nvperf_host/cuda_host/target
    # provide NVPW_* symbols used by the Range Profiler backend.
    find_library(CUPTI_LIBRARY cupti HINTS ${CUDA_INCLUDE_DIRS}/../lib64 ${CUDA_INCLUDE_DIRS}/../lib)
    find_library(NVPERF_HOST_LIBRARY nvperf_host HINTS ${CUDA_INCLUDE_DIRS}/../lib64 ${CUDA_INCLUDE_DIRS}/../lib)
    find_library(NVPERF_CUDA_HOST_LIBRARY nvperf_cuda_host HINTS ${CUDA_INCLUDE_DIRS}/../lib64 ${CUDA_INCLUDE_DIRS}/../lib)
    find_library(NVPERF_TARGET_LIBRARY nvperf_target HINTS ${CUDA_INCLUDE_DIRS}/../lib64 ${CUDA_INCLUDE_DIRS}/../lib)
    if(CUPTI_LIBRARY)
        target_link_libraries(MNNPerfCounter PUBLIC ${CUPTI_LIBRARY})
    else()
        target_link_libraries(MNNPerfCounter PUBLIC -lcupti)
    endif()
    if(NVPERF_HOST_LIBRARY)
        target_link_libraries(MNNPerfCounter PUBLIC ${NVPERF_HOST_LIBRARY})
    endif()
    if(NVPERF_CUDA_HOST_LIBRARY)
        target_link_libraries(MNNPerfCounter PUBLIC ${NVPERF_CUDA_HOST_LIBRARY})
    endif()
    if(NVPERF_TARGET_LIBRARY)
        target_link_libraries(MNNPerfCounter PUBLIC ${NVPERF_TARGET_LIBRARY})
    endif()
endif()

target_include_directories(MNNPerfCounter
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/upstream/hardware_perfcounter/include
        ${CMAKE_CURRENT_LIST_DIR}/upstream/hardware_perfcounter/lib/gpu/adreno
        ${CMAKE_CURRENT_LIST_DIR}/upstream/hardware_perfcounter/lib/gpu/mali
        ${CMAKE_CURRENT_LIST_DIR}/upstream/hardware_perfcounter/third_party
        ${CMAKE_CURRENT_LIST_DIR}/upstream/hardware_perfcounter/third_party/hwcpipe
        ${CMAKE_CURRENT_LIST_DIR}/upstream/libgpucounters/device/include
        ${CMAKE_CURRENT_LIST_DIR}/upstream/libgpucounters/device/src
        ${CMAKE_CURRENT_LIST_DIR}/upstream/libgpucounters/hwcpipe/include
        ${CMAKE_CURRENT_LIST_DIR}/upstream/libgpucounters/hwcpipe/src)

set_target_properties(MNNPerfCounter PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

option(MNN_PERFCOUNTER_BUILD_TESTS "Build MNNPerfCounter host tests" OFF)
if(MNN_PERFCOUNTER_BUILD_TESTS)
    add_executable(mnn_perfcounter_test test/MNNPerfCounterTest.cpp)
    target_link_libraries(mnn_perfcounter_test PRIVATE MNNPerfCounter)
    enable_testing()
    add_test(NAME mnn_perfcounter_test COMMAND mnn_perfcounter_test)
endif()
