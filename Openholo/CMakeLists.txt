cmake_minimum_required(VERSION ${CMAKE_VERSION})
set(CMAKE_PROJECT_NAME "openholo")

message("\n*** ${CMAKE_PROJECT_NAME} begin ***\n")

# build file list
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

add_library(${CMAKE_PROJECT_NAME} OBJECT ${SRC_FILES})

# check compiler
if(MSVC)
	target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
	message("MSVC Compiler")
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES GNU)
	message("GNU Compiler")
endif()

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE OPH_EXPORT)
# check build type
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
	target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE _DEBUG)
	target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -g -Wall)
elseif(${CMAKE_BUILD_TYPE} STREQUAL "Release")
	target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE _NDEBUG)
	# 최적화 옵션 -O0: 최적화x / -O1: 적은 최적화 수준 / -O2: 일반적인 최적화 수준 / -O3: 최대한 최적화 수준
	target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -w -O3)
endif()
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -fPIC)

# 전역 변수의 중복 정의를 허용
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcommon")
# C++ 언어 규칙을 완전히 준수하지 않는 코드일때, 오류를 경고로 출력
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
# OpenMP를 사용하도록 함. 사용 시 _OPENMP 가 정의됨
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
# Position Independent Code 컴파일러가 생성하는 코드가 다른 위치에 링크될 때도 정상동작하도록 작동
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -w)

# 라이브러리 링크
target_link_libraries(${CMAKE_PROJECT_NAME} cudart)
target_link_libraries(${CMAKE_PROJECT_NAME} cufft)
target_link_libraries(${CMAKE_PROJECT_NAME} ${FFTW3_LIBRARIES}_threads)
add_definitions(-DFFTW3_THREADS)
include_directories(${CMAKE_SOURCE_DIR}../Reference/include)
include_directories(${CUDA_TOOLKIT_INCLUDE})
link_directories(${CMAKE_SOURCE_DIR}../Reference/lib)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
link_directories(usr/local/lib/fftw3)



add_library(${CMAKE_PROJECT_NAME}_static STATIC $<TARGET_OBJECTS:${CMAKE_PROJECT_NAME}>)
add_library(${CMAKE_PROJECT_NAME}_shared SHARED $<TARGET_OBJECTS:${CMAKE_PROJECT_NAME}>)


#add_custom_target(clean-all
#	COMMAND ${CMAKE_BUILD_TOOL} clean
#	COMMAND ${CMAKE_COMMAND} -E remove ./build/CMakeCache.txt
#	COMMAND ${CMAKE_COMMAND} -E remove ./build/Makefile
#	COMMAND ${CMAKE_COMMAND} -E remove ./build/cmake_install.cmake
#	COMMAND ${CMAKE_COMMAND} -E -rf remove_drectory ./build/CMakeFiles
#)

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
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")



# set output name
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
	set_target_properties(${CMAKE_PROJECT_NAME}_shared PROPERTIES OUTPUT_NAME ${CMAKE_PROJECT_NAME}_d)
	set_target_properties(${CMAKE_PROJECT_NAME}_static PROPERTIES OUTPUT_NAME ${CMAKE_PROJECT_NAME}_d)
elseif(${CMAKE_BUILD_TYPE} STREQUAL "Release")
	set_target_properties(${CMAKE_PROJECT_NAME}_shared PROPERTIES OUTPUT_NAME ${CMAKE_PROJECT_NAME})
	set_target_properties(${CMAKE_PROJECT_NAME}_static PROPERTIES OUTPUT_NAME ${CMAKE_PROJECT_NAME})
endif()


if(OPH_EXPORT)
	message("OPH_EXPORT is definied")
else()
	message("OPH_EXPORT is not defined")
endif()

# 인스톨 명령
install(TARGETS ${CMAKE_PROJECT_NAME}_static DESTINATION ${CMAKE_PROJECT_ROOT_DIR}/Reference/lib/)
install(TARGETS ${CMAKE_PROJECT_NAME}_shared DESTINATION ${CMAKE_PROJECT_ROOT_DIR}/Reference/lib/)

# 파일 복사
file(GLOB CXX_HEADER_FORMAT "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
file(GLOB CUDA_HEADER_FORMAT "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh")
file(COPY ${CXX_HEADER_FORMAT} DESTINATION ${CMAKE_SOURCE_DIR}/Reference/include)
file(COPY ${CUDA_HEADER_FORMAT} DESTINATION ${CMAKE_SOURCE_DIR}/Reference/include)
#message("File Copied: ${CXX_HEADER_FORMAT} to ${CMAKE_SOURCE_DIR}/Reference/include")
#message("File Copied: ${CUDA_HEADER_FORMAT} to ${CMAKE_SOURCE_DIR}/Reference/include")
#add_custom_command(
#	TARGET ${CMAKE_SOLUTION_NAME}_static
#	POST_BUILD
#	COMMAND ${CMAKE_COMMAND} -E rename 
#	${CMAKE_BINARY_DIR}/lib${CMAKE_SOLUTION_NAME}_static.a 
#	${CMAKE_BINARY_DIR}/lib${CMAKE_SOLUTION_NAME}.a
#)


message("*** ${CMAKE_PROJECT_NAME} end ***")

