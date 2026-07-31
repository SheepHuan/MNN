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
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/mnn/ops/ComplexOps.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/mnn/ops/VulkanMiscOps.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/mnn/ops/VulkanTag120Ops.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/ncnn/NcnnBridge.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/ncnn/ops/ElementwiseOp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/ncnn/ops/Pack4Op.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/ncnn/ops/NcnnElementwiseOps.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/ncnn/ops/NcnnShapeOps.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/ncnn/ops/NcnnNormOps.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cpp/revertMNNModel.cpp)
target_include_directories(replay_benchmark.out PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cpp/
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/
    ${CMAKE_CURRENT_SOURCE_DIR}/replay_benchmark/
    ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/spdlog/include)
# spdlog requires -fexceptions; MNN global uses -fno-exceptions but
# replay_benchmark is a standalone tool and can use exceptions.
target_compile_options(replay_benchmark.out PRIVATE -fexceptions)
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

if(MNN_CUDA)
  # The corpus launch shims (forward declarations of MNN __global__ kernels +
  # extern "C" wrappers) are compiled by nvcc here and linked against
  # MNN_Cuda_Main, which contains the kernel definitions. Device-link time
  # resolves the forward-declared __global__ symbols. The MNN backend
  # directory itself is untouched.
  find_package(CUDA 8.0)
  if(CUDA_FOUND)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES -Wno-deprecated-gpu-targets -w --std=c++17 -O3")
    # MNN CUDA backend adds -fexceptions (source/backend/cuda/CMakeLists.txt:110)
    # so that cub::BlockScan / thrust can compile. The corpus mirrors the same
    # override here: plugins.cu groupNormNHWCSumKernel uses cub::BlockScan for
    # segmented inclusive scan, which requires -fexceptions on the host side
    # (thrust's system_error.inl has a catch(...) clause). Without this flag
    # nvcc fails with "error: exception handling disabled, use -fexceptions".
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fexceptions")
    # CUDA 12.0 nvcc rejects GCC >= 13. Force GCC 12 host compiler if available.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "13")
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -allow-unsupported-compiler")
      find_program(MNN_NVCC_HOST_COMPILER NAMES g++-12 g++-11)
      if(MNN_NVCC_HOST_COMPILER)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -ccbin ${MNN_NVCC_HOST_COMPILER}")
      endif()
    endif()
    include(${CMAKE_CURRENT_SOURCE_DIR}/source/backend/cuda/SelectCudaComputeArch.cmake)
    CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${CUDA_ARCHS})
    # Propagate the auto-detected gencode flags so device-side intrinsics like
    # __hfma2 (sm_53+) are enabled. Without this the default target is sm_30
    # which lacks half2 arithmetic.
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH_FLAGS}")
    cuda_add_library(replay_cuda_corpus STATIC
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/CudaOpsFp16.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/range.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/select.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/binary.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/unary_cast.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/layernorm.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/prelu.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/scale.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/pool.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/softmax.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/gatherv2_argmax.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/interp.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/transpose.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/gridsample.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/reduction_naive.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/topkv2.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/raster.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/convdw.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/rope.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/matmul.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/conv_base.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/deconv.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/attention.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/winograd.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/convdw_extra.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/transpose_half.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/plugins.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/int8.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/raster_fuse.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/weight_only_quant.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/kernels/convdw_bf16.cu)
    # CudaOps.cpp / CudaOpsMisc.cpp / CudaOpsWoq.cpp (fp32 adapters) reference
    # mnn_corpus_* shims compiled by nvcc. Only include them when CUDA is
    # enabled so host builds without nvcc still link cleanly.
    target_sources(replay_benchmark.out PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/CudaOps.cpp
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/CudaOpsMisc.cpp
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge/cuda/CudaOpsWoq.cpp)
    # CudaOpsFp16.cu includes CudaOps.hpp -> CudaOpAdapter.hpp -> OpAdapter.hpp
    # which live under replay_benchmark/. Expose that path so nvcc can find them.
    target_include_directories(replay_cuda_corpus PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge
        ${CUDA_INCLUDE_DIRS})
    target_link_libraries(replay_benchmark.out replay_cuda_corpus MNN_Cuda_Main ${CUDA_LIBRARIES})
    # CudaOps.cpp (fp32 adapters) needs CudaOps.hpp/CudaOpAdapter.hpp too.
    target_include_directories(replay_benchmark.out PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/kernel_corpus_bridge)
    # libcuda (driver API) + nvperf_* (NVPW_* symbols) needed by the
    # NvRangeProfiler backend compiled into MNNPerfCounter. These must be
    # linked when MNN_REPLAY_ENABLE_PERFCOUNTER is on so the static archive's
    # NvRangeProfiler.cpp.o gets pulled in by the linker.
    find_library(CUDA_DRIVER_LIB cuda HINTS ${CUDA_INCLUDE_DIRS}/../lib64 ${CUDA_INCLUDE_DIRS}/../lib/stubs)
    find_library(RB_NVPERF_HOST nvperf_host HINTS ${CUDA_INCLUDE_DIRS}/../lib64 ${CUDA_INCLUDE_DIRS}/../lib)
    find_library(RB_NVPERF_CUDA_HOST nvperf_cuda_host HINTS ${CUDA_INCLUDE_DIRS}/../lib64 ${CUDA_INCLUDE_DIRS}/../lib)
    find_library(RB_NVPERF_TARGET nvperf_target HINTS ${CUDA_INCLUDE_DIRS}/../lib64 ${CUDA_INCLUDE_DIRS}/../lib)
    if(CUDA_DRIVER_LIB)
      target_link_libraries(replay_benchmark.out ${CUDA_DRIVER_LIB})
    endif()
    if(RB_NVPERF_HOST)
      target_link_libraries(replay_benchmark.out ${RB_NVPERF_HOST})
    endif()
    if(RB_NVPERF_CUDA_HOST)
      target_link_libraries(replay_benchmark.out ${RB_NVPERF_CUDA_HOST})
    endif()
    if(RB_NVPERF_TARGET)
      target_link_libraries(replay_benchmark.out ${RB_NVPERF_TARGET})
    endif()
    target_compile_definitions(replay_benchmark.out PRIVATE MNN_REPLAY_HAS_CUDA)
    target_include_directories(replay_benchmark.out PRIVATE ${CUDA_INCLUDE_DIRS})

    # Standalone NVIDIA metric enumerator. Prints one metric name per line.
    # Build target: list_cuda_metrics. Not built unless MNN_CUDA is ON.
    cuda_add_executable(list_cuda_metrics
        ${CMAKE_CURRENT_LIST_DIR}/list_cuda_metrics.cu)
    target_include_directories(list_cuda_metrics PRIVATE
        ${CUDA_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/source/backend/cuda)
    find_library(LCM_CUPTI_LIBRARY cupti HINTS ${CUDA_INCLUDE_DIRS}/../lib64 ${CUDA_INCLUDE_DIRS}/../lib)
    find_library(LCM_NVPERF_HOST nvperf_host HINTS ${CUDA_INCLUDE_DIRS}/../lib64 ${CUDA_INCLUDE_DIRS}/../lib)
    find_library(LCM_NVPERF_TARGET nvperf_target HINTS ${CUDA_INCLUDE_DIRS}/../lib64 ${CUDA_INCLUDE_DIRS}/../lib)
    find_library(LCM_CUDA_DRIVER cuda HINTS ${CUDA_INCLUDE_DIRS}/../lib64 ${CUDA_INCLUDE_DIRS}/../lib/stubs)
    if(LCM_CUPTI_LIBRARY)
      target_link_libraries(list_cuda_metrics ${LCM_CUPTI_LIBRARY} ${LCM_NVPERF_HOST} ${LCM_NVPERF_TARGET} ${LCM_CUDA_DRIVER} ${CUDA_LIBRARIES})
    else()
      target_link_libraries(list_cuda_metrics -lcupti ${LCM_NVPERF_HOST} ${LCM_NVPERF_TARGET} ${LCM_CUDA_DRIVER} ${CUDA_LIBRARIES})
    endif()

    # PMU multi-pass test: verifies that collecting N metrics in one session
    # (which may need multiple passes) produces the same values as collecting
    # each metric individually (single-pass). Run with sudo on NVIDIA GPUs.
    # Build: cmake -DMNN_CUDA=ON .. && make replay_pmu_test
    # Run:   sudo ./replay_pmu_test
    # Filter: ./replay_pmu_test --filter PMU.MultiPass
    add_executable(replay_pmu_test
        ${CMAKE_CURRENT_LIST_DIR}/tests/test_pmu_multipass.cpp)
    target_include_directories(replay_pmu_test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/spdlog/include)

    # PMU availability test: discovers all CUDA PMU metrics via list_cuda_metrics,
    # collects each individually, verifies it returns a valid value. Reports
    # per-domain summary. Run before multi-pass or accuracy tests.
    # Build: cmake -DMNN_CUDA=ON .. && make replay_pmu_availability
    # Run:   sudo ./replay_pmu_availability
    # Filter: ./replay_pmu_availability --filter CudaMetric.SmDomain
    add_executable(replay_pmu_availability
        ${CMAKE_CURRENT_LIST_DIR}/tests/test_pmu_availability.cpp)
    target_include_directories(replay_pmu_availability PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/spdlog/include)

    # CUDA latency test: runs every CUDA corpus case with PMU explicitly
    # disabled and writes one latency_us value per case.
    add_executable(replay_kernel_latency
        ${CMAKE_CURRENT_LIST_DIR}/tests/test_kernel_latency.cpp)
    target_include_directories(replay_kernel_latency PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/spdlog/include)
  else()
    message(WARNING "MNN_CUDA enabled but CUDA toolkit not found; CUDA corpus backend disabled")
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
