cmake_minimum_required(VERSION ${CMAKE_VERSION})

set(ENV{CUDA_HOME} "/usr/local/cuda-11.6")
set(ENV{LD_LIBRARY_PATH} $ENV{LD_LIBRARY_PATH}:/usr/local/lib)
set(ENV{LD_LIBRARY_PATH} $ENV{LD_LIBRARY_PATH}:$ENV{CUDA_HOME}/lib64)
set(ENV{LD_LIBRARY_PATH} $ENV{LD_LIBRARY_PATH}:$ENV{CUDA_HOME}/extras/CUPTI/lib64)
set(ENV{LD_LIBRARY_PATH} $ENV{LD_LIBRARY_PATH}:/usr/lib/x86_64-linux-gnu)
set(ENV{PATH} $ENV{PATH}:$ENV{CUDA_HOME}/bin)
message("PATH: $ENV{PATH}")
message("LD_LIBRARY_PATH: $ENV{LD_LIBRARY_PATH}")
set(CMAKE_SOLUTION_NAME "openholo")
set(CMAKE_CXX_COMPILER "g++")
set(LIBRARY_OUTPUT_PATH lib)

project(${CMAKE_SOLUTION_NAME} CXX CUDA)
find_package(OpenMP REQUIRED) 
find_package(CUDA REQUIRED)
find_package(FFTW3 REQUIRED)


add_subdirectory(Openholo)
add_subdirectory(ophgen)
add_subdirectory(ophrec)
