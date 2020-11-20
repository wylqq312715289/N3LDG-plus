cmake_minimum_required(VERSION 2.8)
project(N3LDG)

option(USE_GPU "use gpu" OFF)

set(BASE_FLAGS "-std=c++11 -Wno-unused-result -iquote src")
set(CMAKE_BUILD_TYPE "release")
set(CMAKE_CXX_FLAGS_DEBUG "${BASE_FLAGS} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${BASE_FLAGS} -w -O3")
set(CMAKE_EXE_LINKER_FLAGS["debug"] "-g")
set(BOOST_ROOT=/opt/boost_1_68_0)
add_definitions(-Wignored-attributes)


find_package(Boost)
if (Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif ()


if (USE_GPU)
  add_definitions(-DUSE_GPU)
  find_package(CUDA REQUIRED)
  find_cuda_helper_libs(curand)
  set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_ROOT})
  message("CUDA_INCLUDE: " ${CUDA_INCLUDE_DIRS})
  add_subdirectory(cuda)
  include_directories(cuda)
endif ()

include_directories(/usr/local/cuda/samples/common/inc)
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
message("CUDA_LIBRARIES: ${CUDA_LIBRARIES}")

set(LIBS ${LIBS} ${CUDA_LIBRARIES})
include_directories(cnmem/include include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../eigen)

add_subdirectory(cnmem)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRCS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/include SRCS)
add_library(N3LDG ${SRCS})

set(LIBS ${LIBS} n3ldg_cuda)
set(N3LDG_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    CACHE INTERNAL "Directory where headers are located")