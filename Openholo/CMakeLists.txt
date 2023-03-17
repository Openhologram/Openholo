cmake_minimum_required(VERSION ${CMAKE_VERSION})

message("CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message("LIBRARY_OUTPUT_PATH: ${LIBRARY_OUTPUT_PATH}")

set(CMAKE_PROJECT_NAME "openholo")

message("*** ${CMAKE_PROJECT_NAME} begin ***")


#add_custom_target(clean-all
#	COMMAND ${CMAKE_BUILD_TOOL} clean
#	COMMAND ${CMAKE_COMMAND} -E remove ./build/CMakeCache.txt
#	COMMAND ${CMAKE_COMMAND} -E remove ./build/Makefile
#	COMMAND ${CMAKE_COMMAND} -E remove ./build/cmake_install.cmake
#	COMMAND ${CMAKE_COMMAND} -E -rf remove_drectory ./build/CMakeFiles
#)

# 빌드 타입 설정 *추가시 링크 오류*
#if(NOT CMAKE_BUILD_TYPE)
#	set(CMAKE_BUILD_TYPE Debug)
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLGAS} -g")
#endif()


# Set CUDA architectures
set(CUDA_ARCH "52")

# Set CUDA flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_${CUDA_ARCH}, code=sm_${CUDA_ARCH}")



# 바이너리 디렉토리 설정
set(CMAKE_PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR})
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})


if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
	add_definitions(-D_LINUX)
	include_directories(usr/include)
elseif (CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
	add_definitions(-D_WIN32)
	add_definitions(-D_WIN64)
endif()

add_definitions(-DOPH_EXPORT)
add_definitions(-DFFTW3_THREADS)


include_directories(${CMAKE_SOURCE_DIR}../Reference/include)
include_directories(${CUDA_TOOLKIT_INCLUDE})

link_directories(${CMAKE_SOURCE_DIR}../Reference/lib)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
link_directories(usr/local/lib/fftw3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
set(CUDA_SEPARABLE_COMPILATION ON)

if (CMAKE_COMPILER_IS_GNUCC)
	message("GNUCC COMPILER")
	# remove warning
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()
if (MSVC)
	message("MSVC COMPILER")
endif()

set(HEADER_FILES
    src/AngularC_data.h
    src/AngularC_types.h
    src/Base.h
    src/coder_array.h
    src/comment.h
    src/complex.h
    src/define.h
    src/epsilon.h
    src/FFTImplementationCallback.h
    src/fftw3.h
    src/function.h
    src/ImgCodecDefine.h
    src/ImgCodecOhc.h
    src/ImgControl.h
    src/include.h
    src/ivec.h
    src/mat.h
    src/Openholo.h
    src/ophKernel.cuh
    src/PLYparser.h
    src/rtGetInf.h
    src/rtGetNaN.h
    src/rtwtypes.h
    src/rt_nonfinite.h
    src/struct.h
    src/sys.h
    src/tmwtypes.h
    src/typedef.h
    src/vec.h
    src/ophKernel.cuh
)

set(SRC_FILES
    ${HEADER_FILES}
    src/AngularC_data.cpp
    src/epsilon.cpp
    src/FFTImplementationCallback.cpp
    src/ImgCodecOhc.cpp
    src/ImgControl.cpp
    src/Openholo.cpp
    src/PLYparser.cpp
    src/rtGetInf.cpp
    src/rtGetNaN.cpp
    src/rt_nonfinite.cpp
    src/sys.cpp
    src/vec.cpp
)

add_library(${CMAKE_SOLUTION_NAME}_static STATIC ${SRC_FILES})
add_library(${CMAKE_SOLUTION_NAME} SHARED ${SRC_FILES})
set_target_properties(
	${CMAKE_SOLUTION_NAME}_static
	PROPERTIES
	OUTPUT_NAME
	${CMAKE_SOLUTION_NAME}
)


add_compile_options(/Od)
target_compile_options(${CMAKE_SOLUTION_NAME}_static PRIVATE -Wno-include-guard)
target_compile_options(${CMAKE_SOLUTION_NAME} PRIVATE -Wno-include-guard)

# 라이브러리 링크
target_link_libraries(${CMAKE_SOLUTION_NAME} cudart)
target_link_libraries(${CMAKE_SOLUTION_NAME} cufft)
target_link_libraries(${CMAKE_SOLUTION_NAME} ${FFTW3_LIBRARIES}_threads)


message("*************** ENV LIST ***************")
message("CMAKE_PROJECT_NAME			= ${CMAKE_PROJECT_NAME}")
message("CMAKE_HOST_SYSTEM_NAME			= ${CMAKE_HOST_SYSTEM_NAME}")
message("CMAKE_BUILD_TYPE			= ${CMAKE_BUILD_TYPE}")
message("CMAKE_SOURCE_DIR			= ${CMAKE_SOURCE_DIR}")
message("CMAKE_BINARY_DIR			= ${CMAKE_BINARY_DIR}")
message("CMAKE_CURRENT_SOURCE_DIR		= ${CMAKE_CURRENT_SOURCE_DIR}")
message("CMAKE_CURRENT_BINARY_DIR		= ${CMAKE_CURRENT_BINARY_DIR}")
message("CMAKE_CXX_COMPILER_ID			= ${CMAKE_CXX_COMPILER_ID}")
message("CMAKE_CXX_COMPILER_VERSION		= ${CMAKE_CXX_COMPILER_VERSION}")
message("CMAKE_CXX_COMPILER			= ${CMAKE_CXX_COMPILER}")
message("CMAKE_CXX_FLAGS				= ${CMAKE_CXX_FLAGS}")
message("CMAKE_CUDA_HOST_COMPILER_ID		= ${CMAKE_CUDA_HOST_COMPILER_ID}")
message("CMAKE_CUDA_HOST_COMPILER_VERSION	= ${CMAKE_CUDA_HOST_COMPILER_VERSION}")
message("CMAKE_CUDA_HOST_COMPILER		= ${CMAKE_CUDA_HOST_COMPILER}")
message("CMAKE_INSTALL_PREFIX			= ${CMAKE_INSTALL_PREFIX}")
message("CUDA_TOOLKIT_INCLUDE			= ${CUDA_TOOLKIT_INCLUDE}")
message("CUDA_TOOLKIT_ROOT_DIR			= ${CUDA_TOOLKIT_ROOT_DIR}")
message("CUDA_INCLUDE_DIRS			= ${CUDA_INCLUDE_DIRS}")
message("CUDA_LIBRARY_DIRS			= ${CUDA_LIBRARY_DIRS}")
message("FFTW3_ROOT				= ${FFTW3_ROOT}")
message("FFTW3_INCLUDE_DIRS			= ${FFTW3_INCLUDE_DIRS}")
message("FFTW3_LIBRARIES				= ${FFTW3_LIBRARIES}")
message("****************************************")

# 인스톨 명령
install(TARGETS ${CMAKE_SOLUTION_NAME} DESTINATION ${CMAKE_PROJECT_ROOT_DIR}/Reference/lib/)

# 파일 복사
file(GLOB CXX_HEADER_FORMAT "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
file(GLOB CUDA_HEADER_FORMAT "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh")
file(COPY ${CXX_HEADER_FORMAT} DESTINATION ${CMAKE_SOURCE_DIR}/Reference/include)
file(COPY ${CUDA_HEADER_FORMAT} DESTINATION ${CMAKE_SOURCE_DIR}/Reference/include)
message("File Copied: ${CXX_HEADER_FORMAT} to ${CMAKE_SOURCE_DIR}/Reference/include")
message("File Copied: ${CUDA_HEADER_FORMAT} to ${CMAKE_SOURCE_DIR}/Reference/include")

#add_custom_command(
#	TARGET ${CMAKE_SOLUTION_NAME}_static
#	POST_BUILD
#	COMMAND ${CMAKE_COMMAND} -E rename 
#	${CMAKE_BINARY_DIR}/lib${CMAKE_SOLUTION_NAME}_static.a 
#	${CMAKE_BINARY_DIR}/lib${CMAKE_SOLUTION_NAME}.a
#)


message("*** ${CMAKE_PROJECT_NAME} end ***")

