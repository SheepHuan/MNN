add_executable(replay_benchmark.out
    ${CMAKE_CURRENT_LIST_DIR}/replay_benchmark.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ReplayRecord.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ReplayRunner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cpp/revertMNNModel.cpp)
target_include_directories(replay_benchmark.out PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/cpp/
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/
    ${CMAKE_CURRENT_SOURCE_DIR}/replay_benchmark/)
target_link_libraries(replay_benchmark.out ${MNN_DEPS})

if(MNN_OPENCL)
  target_compile_definitions(replay_benchmark.out PRIVATE MNN_REPLAY_HAS_OPENCL)
  if(MNN_USE_SYSTEM_LIB)
    find_package(OpenCL REQUIRED)
    target_link_libraries(replay_benchmark.out ${OpenCL_LIBRARIES})
  else()
    # replay_benchmark includes OpenCL C++ wrapper types from the OpenCL
    # backend headers. Keep the executable on the same dynamically-loaded
    # OpenCL path as MNN_CL instead of requiring a platform OpenCL link
    # library.
    target_sources(replay_benchmark.out PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/source/backend/opencl/core/runtime/OpenCLWrapper.cpp)
    target_compile_definitions(replay_benchmark.out PRIVATE MNN_USE_LIB_WRAPPER)
  endif()
endif()

if(MSVC AND NOT MNN_BUILD_SHARED_LIBS)
  foreach(DEPEND ${MNN_DEPS})
    target_link_options(replay_benchmark.out PRIVATE /WHOLEARCHIVE:$<TARGET_FILE:${DEPEND}>)
  endforeach()
endif()
