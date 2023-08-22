cmake_minimum_required(VERSION ${CMAKE_VERSION})
set(CMAKE_PROJECT_NAME "ophgen")

message("\n*** ${CMAKE_PROJECT_NAME} begin ***\n")
set(CMAKE_VERBOSE_MAKEFILE ON)


# build file list
set(SRC_FILES
    src/CUDA.h
#    src/OpenCL.h
    src/ophACPAS.h
#    src/ophAS.h
#    src/ophAS_GPU.h
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
#    src/ophAS.cpp
#    src/ophAS_GPU.cpp
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
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)

# 추가 포함 디렉터리
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/Reference/include)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIR})

# 추가 정의
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE "-DGEN_EXPORT")


add_library(${CMAKE_PROJECT_NAME}_static STATIC $<TARGET_OBJECTS:${CMAKE_PROJECT_NAME}>)
add_library(${CMAKE_PROJECT_NAME}_shared SHARED $<TARGET_OBJECTS:${CMAKE_PROJECT_NAME}>)

# 추가 링크 디렉터
link_directories(${FFTW3_LIBRARY_DIRS})
link_directories(${CMAKE_SOURCE_DIR}/Reference/lib)
link_directories(${CUDAToolkit_LIBRARY_DIR})

find_library(FFTW3_LIB NAMES ${FFTW3_LIBRARIES}_threads PATHS ${FFTW3_LIBRARY_DIRS})
if(FFTW3_LIB)
	message("${FFTW3_LIB} library found.")
	target_link_libraries(${CMAKE_PROJECT_NAME}_static PRIVATE ${FFTW3_LIBRARIES}_threads)
	target_link_libraries(${CMAKE_PROJECT_NAME}_shared PRIVATE ${FFTW3_LIBRARIES}_threads)
else()
	message(FATAL_ERROR "${FFTW3_LIB} library not found.")
endif()

find_library(OPENHOLO_LIB NAMES openholo PATHS ${CMAKE_SOURCE_DIR}/Reference/lib)
if(OPENHOLO_LIB)
	message("${OPENHOLO_LIB} library found.")
	target_link_libraries(${CMAKE_PROJECT_NAME}_static PRIVATE openholo)
	target_link_libraries(${CMAKE_PROJECT_NAME}_shared PRIVATE openholo)
else()
	message(FATAL_ERROR "${OPENHOLO_LIB} library not found.")
endif()

find_library(CUDART_LIB NAMES cudart PATHS ${CUDAToolkit_LIBRARY_DIR})
if(CUDART_LIB)
	message("${CUDART_LIB} library found.")
	target_link_libraries(${CMAKE_PROJECT_NAME}_static PRIVATE cudart)
	target_link_libraries(${CMAKE_PROJECT_NAME}_shared PRIVATE cudart)
else()
	message(FATAL_ERROR "${CUDART_LIB} library not found.")
endif()

find_library(CUFFT_LIB NAMES cufft PATHS ${CUDAToolkit_LIBRARY_DIR})
if(CUFFT_LIB)
	message("${CUFFT_LIB} library found.")
	target_link_libraries(${CMAKE_PROJECT_NAME}_static PRIVATE cufft)
	target_link_libraries(${CMAKE_PROJECT_NAME}_shared PRIVATE cufft)
else()
	message(FATAL_ERROR "${CUFFT_LIB} library not found.")
endif()

find_library(CUDA_LIB NAMES cuda)
if(CUDA_LIB)
	message("${CUDA_LIB} library found.")
	target_link_libraries(${CMAKE_PROJECT_NAME}_static PRIVATE cuda)
	target_link_libraries(${CMAKE_PROJECT_NAME}_shared PRIVATE cuda)
else()
	message(FATAL_ERROR "${CUDA_LIB} library not found.")
endif()

#target_link_libraries(${CMAKE_PROJECT_NAME} ${FFTW3_LIBRARIES}_threads)
#target_link_libraries(${CMAKE_PROJECT_NAME} openholo)
#target_link_libraries(${CMAKE_PROJECT_NAME} cudart)
#target_link_libraries(${CMAKE_PROJECT_NAME} cufft)
#target_link_libraries(${CMAKE_PROJECT_NAME} cuda)

add_dependencies(${CMAKE_PROJECT_NAME}_static openholo_static)
add_dependencies(${CMAKE_PROJECT_NAME}_shared openholo_shared)
add_dependencies(${CMAKE_PROJECT_NAME} openholo)

# 바이너리 디렉토리 설정
set(CMAKE_PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR})
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# set output name
if(${CMAKE_BUILD_TYPE} MATCHES Debug)
	set_target_properties(${CMAKE_PROJECT_NAME}_shared PROPERTIES CUDA_ARCHITECTURES OFF OUTPUT_NAME ${CMAKE_PROJECT_NAME}_d)
	set_target_properties(${CMAKE_PROJECT_NAME}_static PROPERTIES CUDA_ARCHITECTURES OFF OUTPUT_NAME ${CMAKE_PROJECT_NAME}_d)
elseif(${CMAKE_BUILD_TYPE} MATCHES Release)
	set_target_properties(${CMAKE_PROJECT_NAME}_shared PROPERTIES CUDA_ARCHITECTURES OFF OUTPUT_NAME ${CMAKE_PROJECT_NAME})
	set_target_properties(${CMAKE_PROJECT_NAME}_static PROPERTIES CUDA_ARCHITECTURES OFF OUTPUT_NAME ${CMAKE_PROJECT_NAME})
endif()


# 인스톨 명령
install(TARGETS ${CMAKE_PROJECT_NAME}_static DESTINATION ${CMAKE_PROJECT_ROOT_DIR}/Reference/lib/)
install(TARGETS ${CMAKE_PROJECT_NAME}_shared DESTINATION ${CMAKE_PROJECT_ROOT_DIR}/Reference/lib/)
install(TARGETS ${CMAKE_PROJECT_NAME}_static DESTINATION ${CMAKE_PROJECT_ROOT_DIR}/../Openholo_Tile_CGH/Reference/lib/)
install(TARGETS ${CMAKE_PROJECT_NAME}_shared DESTINATION ${CMAKE_PROJECT_ROOT_DIR}/../Openholo_Tile_CGH/Reference/lib/)



# 소스 파일 복사
file(GLOB CXX_HEADER_FORMAT "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
file(COPY ${CXX_HEADER_FORMAT} DESTINATION ${CMAKE_SOURCE_DIR}/Reference/include)
message("File Copied")
message("*** ${CMAKE_PROJECT_NAME} end ***")

