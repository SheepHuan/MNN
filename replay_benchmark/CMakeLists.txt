add_executable(replay_benchmark.out
    ${CMAKE_CURRENT_LIST_DIR}/replay_benchmark.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ReplayRecord.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ReplayRunner.cpp
    ${CMAKE_CURRENT_LIST_DIR}/PerfCounterReport.cpp
    ${CMAKE_CURRENT_LIST_DIR}/OpenCLPmuBenchmark.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ModelPmuBenchmark.cpp
    ${CMAKE_CURRENT_LIST_DIR}/KernelCorpusBenchmark.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/Bridge.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/OpAdapter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/mnn/MnnBridge.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/mnn/ops/RasterOp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/mnn/ops/UnaryOp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/mnn/ops/MatmulOp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/mnn/ops/ReductionOp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/mnn/ops/PoolingOp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/mnn/ops/BinaryOp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/mnn/ops/ElementwiseOps.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/mnn/ops/NormConvOps.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/mnn/ops/ArgmaxOp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/mnn/ops/ConvOps.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/mnn/ops/ArgmaxOp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/ncnn/NcnnBridge.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/ncnn/ops/ElementwiseOp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/ncnn/ops/Pack4Op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cpp/revertMNNModel.cpp)
target_include_directories(replay_benchmark.out PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cpp/
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/
    ${CMAKE_CURRENT_SOURCE_DIR}/replay_benchmark/)
target_link_libraries(replay_benchmark.out ${MNN_DEPS})

if(MNN_REPLAY_ENABLE_PERFCOUNTER)
  target_link_libraries(replay_benchmark.out MNNPerfCounter)
  target_compile_definitions(replay_benchmark.out PRIVATE MNN_REPLAY_HAS_PERFCOUNTER)
endif()

if(MNN_BUILD_LLM)
  target_compile_definitions(replay_benchmark.out PRIVATE MNN_REPLAY_HAS_LLM)
  target_include_directories(replay_benchmark.out PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/transformers/llm/engine/include)
  if(MNN_SEP_BUILD)
    target_link_libraries(replay_benchmark.out llm)
  endif()
endif()

if(MNN_OPENCL)
  target_link_libraries(replay_benchmark.out MNN_CL)
  target_compile_definitions(replay_benchmark.out PRIVATE MNN_REPLAY_HAS_OPENCL)
  if(MNN_USE_SYSTEM_LIB)
    find_package(OpenCL REQUIRED)
    target_link_libraries(replay_benchmark.out ${OpenCL_LIBRARIES})
  else()
    # OpenCLWrapper.cpp is owned by MNN_CL. Link the executable to that
    # exported wrapper instead of compiling a second copy with another
    # OpenCLSymbols singleton.
    target_compile_definitions(replay_benchmark.out PRIVATE MNN_USE_LIB_WRAPPER)
  endif()
endif()

if(MNN_VULKAN)
  target_link_libraries(replay_benchmark.out MNN_Vulkan)
  target_compile_definitions(replay_benchmark.out PRIVATE MNN_REPLAY_HAS_VULKAN)
  target_include_directories(replay_benchmark.out PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/source/backend/vulkan)
  if(MNN_USE_SYSTEM_LIB)
    find_package(Vulkan REQUIRED)
    target_link_libraries(replay_benchmark.out ${Vulkan_LIBRARIES})
  else()
    target_link_libraries(replay_benchmark.out ${CMAKE_DL_LIBS})
  endif()
endif()

if(MSVC AND NOT MNN_BUILD_SHARED_LIBS)
  foreach(DEPEND ${MNN_DEPS})
    target_link_options(replay_benchmark.out PRIVATE /WHOLEARCHIVE:$<TARGET_FILE:${DEPEND}>)
  endforeach()
endif()

option(MNN_OPENCL_PMU_BENCH_BUILD_TESTS "Build OpenCL PMU benchmark metadata tests" OFF)
if(MNN_OPENCL_PMU_BENCH_BUILD_TESTS)
  add_executable(opencl_pmu_benchmark_test
      ${CMAKE_CURRENT_LIST_DIR}/OpenCLPmuBenchmarkTest.cpp
      ${CMAKE_CURRENT_LIST_DIR}/OpenCLPmuBenchmark.cpp
      ${CMAKE_CURRENT_LIST_DIR}/ModelPmuBenchmark.cpp
      ${CMAKE_CURRENT_LIST_DIR}/ReplayRecord.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/cpp/revertMNNModel.cpp)
  target_include_directories(opencl_pmu_benchmark_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/cpp/
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/
      ${CMAKE_CURRENT_SOURCE_DIR}/replay_benchmark/)
  target_link_libraries(opencl_pmu_benchmark_test ${MNN_DEPS})
  if(MNN_REPLAY_ENABLE_PERFCOUNTER)
    target_link_libraries(opencl_pmu_benchmark_test MNNPerfCounter)
    target_compile_definitions(opencl_pmu_benchmark_test PRIVATE MNN_REPLAY_HAS_PERFCOUNTER)
  endif()
  if(MNN_BUILD_LLM)
    target_compile_definitions(opencl_pmu_benchmark_test PRIVATE MNN_REPLAY_HAS_LLM)
    target_include_directories(opencl_pmu_benchmark_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/transformers/llm/engine/include)
    if(MNN_SEP_BUILD)
      target_link_libraries(opencl_pmu_benchmark_test llm)
    endif()
  endif()
  if(MNN_OPENCL)
    target_link_libraries(opencl_pmu_benchmark_test MNN_CL)
    target_compile_definitions(opencl_pmu_benchmark_test PRIVATE MNN_REPLAY_HAS_OPENCL)
    if(MNN_USE_SYSTEM_LIB)
      find_package(OpenCL REQUIRED)
      target_link_libraries(opencl_pmu_benchmark_test ${OpenCL_LIBRARIES})
    else()
      target_compile_definitions(opencl_pmu_benchmark_test PRIVATE MNN_USE_LIB_WRAPPER)
    endif()
  endif()
endif()
