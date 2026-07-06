option(BUILD_MLS "Build PC Commandline." OFF)
option(MNN_LLM_BUILD_DEMO "Build LLM demo" ON)
option(MNN_PIC_BUILD_SERVER "Build standalone PIC server." ON)
option(LLM_SUPPORT_HTTP_RESOURCE "Support HTTP resource download" ON)

set(LLM_DEPS ${MNN_DEPS})
if (MNN_BUILD_OPENCV)
    add_definitions(-DLLM_SUPPORT_VISION)
    if (MNN_SEP_BUILD)
        list(APPEND LLM_DEPS MNNOpenCV)
    endif()
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../../../tools/cv/include/)
endif()

if(MNN_IMGCODECS)
    add_definitions(-DMNN_IMGCODECS)
endif()

if (MNN_BUILD_AUDIO)
    if (MNN_SEP_BUILD)
        list(APPEND LLM_DEPS MNNAudio)
    endif()
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../../../tools/audio/include/)
endif()

# include dir
include_directories(BEFORE ${CMAKE_CURRENT_LIST_DIR}/include/)

# source files
FILE(GLOB_RECURSE SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*)

if (MNN_SEP_BUILD)
    if (MNN_BUILD_SHARED_LIBS)
        # compile dynamic so, support Linux/Mac
        add_library(pic_llm SHARED ${SRCS})
        set_target_properties(pic_llm PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
        target_compile_options(pic_llm PRIVATE -fvisibility=default)
        target_link_libraries(pic_llm ${LLM_DEPS})
    else()
        add_library(pic_llm STATIC ${SRCS})
    endif()
    list(APPEND LLM_DEPS pic_llm)
else()
    add_library(pic_llm OBJECT ${SRCS})
endif()
# jinja.cpp template engine (always enabled, header-only)
target_compile_definitions(pic_llm PRIVATE LLM_USE_JINJA)
set_property(TARGET pic_llm PROPERTY CXX_STANDARD 17)
set_property(TARGET pic_llm PROPERTY CXX_STANDARD_REQUIRED ON)

# apply_template demo (uses jinja)
add_executable(pic_apply_template ${CMAKE_CURRENT_LIST_DIR}/demo/apply_template.cpp)
target_link_libraries(pic_apply_template ${LLM_DEPS})

if (MNN_BUILD_OPENCV)
    target_compile_definitions(pic_llm PRIVATE LLM_SUPPORT_VISION)
endif()
if (MNN_BUILD_AUDIO)
    add_definitions(-DLLM_SUPPORT_AUDIO)
endif()

if (LLM_SUPPORT_HTTP_RESOURCE)
    target_compile_definitions(pic_llm PRIVATE LLM_SUPPORT_HTTP_RESOURCE)
    # Disable exceptions in httplib since MNN is compiled with -fno-exceptions
    target_compile_definitions(pic_llm PRIVATE CPPHTTPLIB_NO_EXCEPTIONS)
endif()

IF(CMAKE_SYSTEM_NAME MATCHES "^Android" AND NOT MNN_BUILD_FOR_ANDROID_COMMAND)
IF(NOT NATIVE_INCLUDE_OUTPUT)
  set(NATIVE_INCLUDE_OUTPUT ".")
ENDIF()
add_custom_command(
  TARGET pic_llm
  POST_BUILD
  COMMAND ${CMAKE_COMMAND}
  ARGS -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/include ${NATIVE_INCLUDE_OUTPUT}
)
ELSE()
INSTALL(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/ DESTINATION include FILES_MATCHING PATTERN *.hpp)
ENDIF()


if(MNN_LLM_BUILD_DEMO)
    add_executable(pic_llm_demo ${CMAKE_CURRENT_LIST_DIR}/demo/llm_demo.cpp)
    add_executable(pic_embedding_demo ${CMAKE_CURRENT_LIST_DIR}/demo/embedding_demo.cpp)
    add_executable(pic_reranker_demo ${CMAKE_CURRENT_LIST_DIR}/demo/reranker_demo.cpp)
    add_executable(pic_rollback_demo ${CMAKE_CURRENT_LIST_DIR}/demo/rollback_demo.cpp)
    FILE(GLOB TOKENIZER_SRCS ${CMAKE_CURRENT_LIST_DIR}/src/tokenizer/*.cpp)
    add_executable(pic_tokenizer_demo ${CMAKE_CURRENT_LIST_DIR}/demo/tokenizer_demo.cpp ${TOKENIZER_SRCS})
    target_include_directories(pic_tokenizer_demo PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/tokenizer/ ${CMAKE_CURRENT_LIST_DIR}/src/)

    include(${CMAKE_CURRENT_LIST_DIR}/tools/CMakeLists.txt)
    target_link_libraries(pic_llm_demo ${LLM_DEPS})
    target_link_libraries(pic_embedding_demo ${LLM_DEPS})
    target_link_libraries(pic_reranker_demo ${LLM_DEPS})
    target_link_libraries(pic_rollback_demo ${LLM_DEPS})
    target_link_libraries(pic_tokenizer_demo ${LLM_DEPS})
endif()

option(MNN_LLM_BUILD_TEST "Build LLM unit tests" OFF)
if(MNN_LLM_BUILD_TEST)
    add_executable(pic_test_tokenizer ${CMAKE_CURRENT_LIST_DIR}/test/test_tokenizer.cpp)
    target_include_directories(pic_test_tokenizer PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/)
    target_link_libraries(pic_test_tokenizer pic_llm ${LLM_DEPS})
endif()

if (MNN_PIC_BUILD_SERVER)
    add_executable(pic_server
        ${CMAKE_CURRENT_LIST_DIR}/app/pic_server_main.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/pic_server.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/pic_server_prefill.cpp
    )
    target_include_directories(pic_server PRIVATE
        ${MNN_INCLUDES}
        ${CMAKE_CURRENT_LIST_DIR}/src/
        ${CMAKE_CURRENT_LIST_DIR}/app/
    )
    set_property(TARGET pic_server PROPERTY CXX_STANDARD 17)
    set_property(TARGET pic_server PROPERTY CXX_STANDARD_REQUIRED ON)
    target_link_libraries(pic_server PRIVATE ${LLM_DEPS})
    target_compile_definitions(pic_server PRIVATE CPPHTTPLIB_NO_EXCEPTIONS)
endif()

if (BUILD_MLS)
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version" FORCE)

add_executable(pic_mls
        ${CMAKE_CURRENT_LIST_DIR}/app/mls.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/remote_model_downloader.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/file_utils.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/llm_benchmark.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/llm_benchmark_common.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/hf_api_client.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/mls_server.cpp
)
# set(OPENSSL_ROOT_DIR "C:/Program Files/OpenSSL-Win64")
# set(OPENSSL_INCLUDE_DIR "C:/Program Files/OpenSSL-Win64/include")
# set(OPENSSL_CRYPTO_LIBRARY "C:/Program Files/OpenSSL-Win64/lib/VC/x64/MD/libcrypto.lib")
# set(OPENSSL_LIBRARIES "C:/Program Files/OpenSSL-Win64/lib//VC/x64/MD/libssl.lib;C:/Program Files/OpenSSL-Win64/lib/VC/x64/MD/libcrypto.lib")
# Set OpenSSL_FOUND based on the presence of include directory and libraries
set(OPENSSL_USE_STATIC_LIBS TRUE)
message(STATUS "OPENSSL_INCLUDE_DIR: ${OPENSSL_INCLUDE_DIR} OPENSSL_LIBRARIES:${OPENSSL_LIBRARIES}")
if(OPENSSL_LIBRARIES AND OPENSSL_INCLUDE_DIR)
    set(OpenSSL_FOUND TRUE)
else()
    set(OpenSSL_FOUND FALSE)
endif()
# Check if OPENSSL_LIBRARIES is not set to find OpenSSL
if(NOT OpenSSL_FOUND)
    find_package(OpenSSL REQUIRED)
endif()

# Check if OpenSSL is found
if(OpenSSL_FOUND)
    message(STATUS "OpenSSL found: ${OPENSSL_INCLUDE_DIR} ${OPENSSL_LIBRARIES}")

    # Use the OpenSSL include directory
    target_include_directories(pic_mls PRIVATE ${OPENSSL_INCLUDE_DIR})

    # Link to OpenSSL libraries
    target_link_libraries(pic_mls PRIVATE ${OPENSSL_LIBRARIES})

    # If on Apple platform, link with Security framework
    if(APPLE)
        find_library(SECURITY_FRAMEWORK Security)
        if(SECURITY_FRAMEWORK)
            target_link_libraries(pic_mls PRIVATE ${SECURITY_FRAMEWORK})
        else()
            message(FATAL_ERROR "Could not find Security framework")
        endif()
    endif()
else()
    message(FATAL_ERROR "OpenSSL not found. Please install OpenSSL and set paths.")
endif()

target_include_directories(pic_mls PRIVATE ${MNN_INCLUDES} ${CMAKE_CURRENT_LIST_DIR}/src/)
set_property(TARGET pic_mls PROPERTY CXX_STANDARD 17)
set_property(TARGET pic_mls PROPERTY CXX_STANDARD_REQUIRED ON)
# target_compile_options(pic_mls PRIVATE -std=c++17)
target_link_libraries(pic_mls PRIVATE ${LLM_DEPS})
target_compile_definitions(pic_mls PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()
