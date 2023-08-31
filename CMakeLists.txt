cmake_minimum_required(VERSION 3.17)

set(ENV{LD_LIBRARY_PATH} $ENV{LD_LIBRARY_PATH}:/usr/local/lib)
set(ENV{LD_LIBRARY_PATH} $ENV{LD_LIBRARY_PATH}:/usr/lib/x86_64-linux-gnu)
set(ENV{PATH} $ENV{PATH}:$ENV{CUDA_HOME}/bin)
set(ENV{TEST} $ENV{TEST}:1234)
set(CMAKE_SOLUTION_NAME "Openholo")
set(CMAKE_CXX_COMPILER "g++")
set(LIBRARY_OUTPUT_PATH lib)

# 바이너리 디렉토리 설정
set(CMAKE_PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR})
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

project(${CMAKE_SOLUTION_NAME} LANGUAGES CXX CUDA)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)	
if(CUDAToolkit_FOUND)
	message("Found CUDAToolkit")
endif()
find_package(FFTW3 REQUIRED)

message("*************** ENV LIST ***************")
message("CMAKE_VERSION			= ${CMAKE_VERSION}")
message("CMAKE_SOLUTION_NAME		= ${CMAKE_SOLUTION_NAME}")
message("CMAKE_HOST_SYSTEM_NAME		= ${CMAKE_HOST_SYSTEM_NAME}")
message("CMAKE_BUILD_TYPE		= ${CMAKE_BUILD_TYPE}")
message("CMAKE_SOURCE_DIR		= ${CMAKE_SOURCE_DIR}")
message("CMAKE_BINARY_DIR		= ${CMAKE_BINARY_DIR}")
message("CMAKE_CURRENT_SOURCE_DIR	= ${CMAKE_CURRENT_SOURCE_DIR}")
message("CMAKE_CURRENT_BINARY_DIR	= ${CMAKE_CURRENT_BINARY_DIR}")
message("CMAKE_CXX_COMPILER_ID		= ${CMAKE_CXX_COMPILER_ID}")
message("CMAKE_CXX_COMPILER_VERSION	= ${CMAKE_CXX_COMPILER_VERSION}")
message("CMAKE_CXX_COMPILER		= ${CMAKE_CXX_COMPILER}")
message("CMAKE_INSTALL_PREFIX		= ${CMAKE_INSTALL_PREFIX}")
message("CUDAToolkit_VERSION		= ${CUDAToolkit_VERSION}")
message("CUDAToolkit_BIN_DIR		= ${CUDAToolkit_BIN_DIR}")
message("CUDAToolkit_INCLUDE_DIR		= ${CUDAToolkit_INCLUDE_DIR}")
message("CUDAToolkit_LIBRARY_DIR		= ${CUDAToolkit_LIBRARY_DIR}")
message("CUDAToolkit_TARGET_DIR		= ${CUDAToolkit_TARGET_DIR}")
message("CUDAToolkit_NVCC_EXECUTABLE	= ${CUDAToolkit_NVCC_EXECUTABLE}")
message("FFTW3_LIBRARIES			= ${FFTW3_LIBRARIES}")
message("FFTW3_INCLUDE_DIRS		= ${FFTW3_INCLUDE_DIRS}")
message("FFTW3_LIBRARY_DIRS		= ${FFTW3_LIBRARY_DIRS}")
message("PATH 				= $ENV{PATH}")
message("LD_LIBRARY_PATH		= $ENV{LD_LIBRARY_PATH}")
message("****************************************")

file(WRITE ./Colab/build.env "export CUDA_HOME=${CUDAToolkit_TARGET_DIR}\n")
file(APPEND ./Colab/build.env "export FFTW3_LIB_DIR=${FFTW3_LIBRARY_DIRS}")


if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
	add_definitions(-D_LINUX)
	include_directories(/usr/include)
# Position Independent Code 컴파일러가 생성하는 코드가 다른 위치에 링크될 때도 정상동작하도록 작동
	add_compile_options(-fPIC)
elseif (CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
	add_definitions(-D_WIN32)
	add_definitions(-D_WIN64)
# check compiler
if(MSVC)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
	message("Add defined CMAKE_BUILD_TYPE=Release") 
endif()

# check build type
if(${CMAKE_BUILD_TYPE} MATCHES Debug)
	message("Configuring for Debug build")
	add_definitions(-D_DEBUG)
	add_compile_options(-Wall)
elseif(${CMAKE_BUILD_TYPE} MATCHES Release)
	set(CMAKE_CUDA_FLAGS_RELEASE "")
	message("Configuring for Release build")
	add_definitions(-D_NDEBUG)
	add_compile_options(-w)
endif()

add_subdirectory(Openholo)
add_subdirectory(ophgen)
add_subdirectory(ophrec)
add_subdirectory(ophsig)


#add_custom_target(clean-all
#	COMMAND ${CMAKE_BUILD_TOOL} clean
#	COMMAND ${CMAKE_COMMAND} -E remove ./build/CMakeCache.txt
#	COMMAND ${CMAKE_COMMAND} -E remove ./build/Makefile
#	COMMAND ${CMAKE_COMMAND} -E remove ./build/cmake_install.cmake
#	COMMAND ${CMAKE_COMMAND} -E -rf remove_drectory ./build/CMakeFiles
#)


#message("File Copied: ${CXX_HEADER_FORMAT} to ${CMAKE_SOURCE_DIR}/Reference/include")
#message("File Copied: ${CUDA_HEADER_FORMAT} to ${CMAKE_SOURCE_DIR}/Reference/include")
#add_custom_command(
#	TARGET ${CMAKE_SOLUTION_NAME}_static
#	POST_BUILD
#	COMMAND ${CMAKE_COMMAND} -E rename 
#	${CMAKE_BINARY_DIR}/lib${CMAKE_SOLUTION_NAME}_static.a 
#	${CMAKE_BINARY_DIR}/lib${CMAKE_SOLUTION_NAME}.a
#)

