cmake_minimum_required(VERSION ${CMAKE_VERSION})
set(CMAKE_PROJECT_NAME "ophgen")

message("\n*** ${CMAKE_PROJECT_NAME} begin ***\n")

# build file list
set(SRC_FILES
    src/CUDA.h
#    src/OpenCL.h
    src/ophACPAS.h
    src/ophAS.h
    src/ophAS_GPU.h
    src/ophDepthMap.h
    src/ophDepthMap_GPU.h
    src/ophGen.h
    src/ophIFTA.h
    src/ophLightField.h
    src/ophLightField_GPU.h
    src/ophNonHogelLF.h
    src/ophPAS.h
    src/ophPAS_GPU.h
    src/ophPointCloud.h
    src/ophPointCloud_GPU.h
    src/ophTriMesh.h
    src/ophTriMesh_GPU.h
    src/ophWRP.h
    src/ophWRP_GPU.h
    src/tinyxml2.h
    src/CUDA.cpp
#    src/OpenCL.cpp
    src/ophACPAS.cpp
    src/ophAS.cpp
    src/ophAS_GPU.cpp
    src/ophDepthMap.cpp
    src/ophDepthMap_GPU.cpp
    src/ophGen.cpp
    src/ophIFTA.cpp
    src/ophLightField.cpp
    src/ophLightField_GPU.cpp
    src/ophNonHogelLF.cpp
    src/ophPAS.cpp
    src/ophPAS_GPU.cpp
    src/ophPointCloud.cpp
    src/ophPointCloud_GPU.cpp
    src/ophTriMesh.cpp
    src/ophTriMesh_GPU.cpp
    src/ophWRP.cpp
    src/ophWRP_GPU.cpp
    src/tinyxml2.cpp
    src/ophPCKernel.cu
    src/ophDMKernel.cu
    src/ophLFKernel.cu
    src/ophWRPKernel.cu
    src/ophTriMeshKernel.cu
    src/ophASKernel.cu
    src/ophPASKernel.cu
)

add_library(${CMAKE_PROJECT_NAME} OBJECT ${SRC_FILES})

# check compiler
if(MSVC)
	target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
	message("MSVC Compiler")
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES GNU)
	message("GNU Compiler")
endif()

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE GEN_EXPORT)

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

# 라이브러리 링크
add_dependencies(${CMAKE_PROJECT_NAME} openholo_static)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${CMAKE_PROJECT_NAME} ${FFTW3_LIBRARIES}_threads)
target_link_libraries(${CMAKE_PROJECT_NAME} openholo)
target_link_libraries(${CMAKE_PROJECT_NAME} cudart)
target_link_libraries(${CMAKE_PROJECT_NAME} cufft)
target_link_libraries(${CMAKE_PROJECT_NAME} cuda)
add_definitions(-DFFTW3_THREADS)
message("CMAKE_SOURCE_DIR : ${CMAKE_SOURCE_DIR}")
include_directories(${CMAKE_SOURCE_DIR}/Reference/include)
include_directories(${CUDA_TOOLKIT_INCLUDE})
link_directories(${CMAKE_SOURCE_DIR}/Reference/lib)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
link_directories(usr/local/lib/fftw3)


add_library(${CMAKE_PROJECT_NAME}_static STATIC $<TARGET_OBJECTS:${CMAKE_PROJECT_NAME}>)
add_library(${CMAKE_PROJECT_NAME}_shared SHARED $<TARGET_OBJECTS:${CMAKE_PROJECT_NAME}>)

# 바이너리 디렉토리 설정
set(CMAKE_PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR})
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# no warning
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_52 -Xcompiler")

# set output name
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
	set_target_properties(${CMAKE_PROJECT_NAME}_shared PROPERTIES OUTPUT_NAME ${CMAKE_PROJECT_NAME}_d)
	set_target_properties(${CMAKE_PROJECT_NAME}_static PROPERTIES OUTPUT_NAME ${CMAKE_PROJECT_NAME}_d)
elseif(${CMAKE_BUILD_TYPE} STREQUAL "Release")
	set_target_properties(${CMAKE_PROJECT_NAME}_shared PROPERTIES OUTPUT_NAME ${CMAKE_PROJECT_NAME})
	set_target_properties(${CMAKE_PROJECT_NAME}_static PROPERTIES OUTPUT_NAME ${CMAKE_PROJECT_NAME})
endif()


if(GEN_EXPORT)
	message("GEN_EXPORT is definied")
else()
	message("GEN_EXPORT is not defined")
endif()

# 인스톨 명령
install(TARGETS ${CMAKE_PROJECT_NAME}_static DESTINATION ${CMAKE_PROJECT_ROOT_DIR}/Reference/lib/)
install(TARGETS ${CMAKE_PROJECT_NAME}_shared DESTINATION ${CMAKE_PROJECT_ROOT_DIR}/Reference/lib/)


# 소스 파일 복사
file(GLOB CXX_HEADER_FORMAT "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
file(COPY ${CXX_HEADER_FORMAT} DESTINATION ${CMAKE_SOURCE_DIR}/Reference/include)
message("File Copied")
message("*** ${CMAKE_PROJECT_NAME} end ***")

