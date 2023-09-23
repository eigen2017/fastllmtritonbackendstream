cmake_minimum_required(VERSION 3.5)

project(glmstreambackendcmakeproject LANGUAGES C CXX)

option(TRITON_ENABLE_GPU "enable gpu support in backend" OFF)
option(TRITON_ENABLE_STATS "include statistics collections in backend" ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)

FetchContent_Declare(
    repo-common
    PREFIX repo-common
    SOURCE_DIR ../../common-main
)
FetchContent_Declare(
    repo-core
    PREFIX repo-core
    SOURCE_DIR ../../core-main
)
FetchContent_Declare(
    repo-backend
    PREFIX repo-backend
    SOURCE_DIR ../../backend-main
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

configure_file(src/libtriton_glmstreambackend.ldscript libtriton_glmstreambackend.ldscript COPYONLY)

add_library(
    triton_glmstreambackend SHARED
    src/glmstreambackend.cc
)

enable_language(CUDA)
add_compile_definitions(USE_CUDA)
set(CMAKE_CUDA_ARCHITECTURES "native")

target_include_directories(
    triton_glmstreambackend
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    /models/glm/fastllm-master/include
    /models/glm/fastllm-master/include/utils
    /models/glm/fastllm-master/include/models
    /models/glm/fastllm-master/include/devices/cuda
)

target_compile_features(triton_glmstreambackend PRIVATE cXX_std_11)
target_compile_options(
    triton_glmstreambackend PRIVATE
    -pthread --std=c++17 -march-native -g
)

target_link_libraries(
    triton_glmstreambackend
    PRIVATE
    triton-core-serverapi
    triton-core-backendapi
    triton-core-serverstub
    triton-backend-utils
    /models/glm/fastllm-master/build/libfastllm.so
)

set_target_properties (
    triton_glmstreambackend PROPERTIES
    PROITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_glmstreambackend
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_glmstreambackend.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_glmstreambackend.ldscript"
)


