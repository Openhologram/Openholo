cmake_minimum_required(VERSION ${CMAKE_VERSION})
set(CMAKE_PROJECT_NAME "ophrec")

message("\n*** ${CMAKE_PROJECT_NAME} begin ***\n")
set(CMAKE_VERBOSE_MAKEFILE ON)


# 빌드 대상 소스
set(SRC_FILES
	src/ophRec.h
	src/ophRec_GPU.h
	src/ophCascadedPropagation.h
	src/ophWaveAberration.h
	src/tinyxml2.h
    	src/ophRec.cpp
	src/ophRec_GPU.cpp
	src/ophCascadedPropagation.cpp
	src/ophWaveAberration.cpp
	src/tinyxml2.cpp
	src/ophRecKernel.cu	
)

# *** 컴파일 관련 ***
add_library(${CMAKE_PROJECT_NAME} OBJECT ${SRC_FILES})
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)

# CUDA 컴파일러 옵션
if(${CMAKE_BUILD_TYPE} MATCHES Debug)
	target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -G>)
endif()
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -lineinfo>)
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -use_fast_math>)
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -maxrregcount=0>)
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --machine 64>)
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --compile -cudart static>)

# 추가 포함 디렉터리
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/Reference/include)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# 추가 정의
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE "-DREC_EXPORT")

# *** 링크 관련 ***
add_library(${CMAKE_PROJECT_NAME}_static STATIC $<TARGET_OBJECTS:${CMAKE_PROJECT_NAME}>)
add_library(${CMAKE_PROJECT_NAME}_shared SHARED $<TARGET_OBJECTS:${CMAKE_PROJECT_NAME}>)

# 추가 링크 디렉터
target_link_directories(${CMAKE_PROJECT_NAME}_static PRIVATE ${FFTW3_LIBRARY_DIRS})
target_link_directories(${CMAKE_PROJECT_NAME}_shared PRIVATE ${FFTW3_LIBRARY_DIRS})
target_link_directories(${CMAKE_PROJECT_NAME}_static PRIVATE ${LIBRARY_OUTPUT_PATH})
target_link_directories(${CMAKE_PROJECT_NAME}_shared PRIVATE ${LIBRARY_OUTPUT_PATH})
target_link_directories(${CMAKE_PROJECT_NAME}_static PRIVATE ${CUDAToolkit_LIBRARY_DIR})
target_link_directories(${CMAKE_PROJECT_NAME}_shared PRIVATE ${CUDAToolkit_LIBRARY_DIR})

message("${LIBRARY_OUTPUT_PATH}")
find_library(OPENHOLO_LIB NAMES openholo PATHS ${LIBRARY_OUTPUT_PATH})
#if(OPENHOLO_LIB)
#	message("${OPENHOLO_LIB} library found.")
	target_link_libraries(${CMAKE_PROJECT_NAME}_static PRIVATE openholo)
	target_link_libraries(${CMAKE_PROJECT_NAME}_shared PRIVATE openholo)
#else()
#	message(FATAL_ERROR "${OPENHOLO_LIB} library not found.")
#endif()

find_library(FFTW3_LIB NAMES ${FFTW3_LIBRARIES}_threads PATHS ${FFTW3_LIBRARY_DIRS})
if(FFTW3_LIB)
	message("${FFTW3_LIB} library found.")
	target_link_libraries(${CMAKE_PROJECT_NAME}_static PRIVATE ${FFTW3_LIBRARIES}_threads)
	target_link_libraries(${CMAKE_PROJECT_NAME}_shared PRIVATE ${FFTW3_LIBRARIES}_threads)
else()
	message(FATAL_ERROR "${FFTW3_LIB} library not found.")
endif()

find_library(CUDART_LIB NAMES cudart PATHS ${CUDA_LIBRARY_DIRS})
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

target_link_libraries(${CMAKE_PROJECT_NAME}_static PRIVATE cuda)
target_link_libraries(${CMAKE_PROJECT_NAME}_shared PRIVATE cuda)

find_library(NPPS_LIB NAMES npps PATHS ${CUDAToolkit_LIBRARY_DIR})
if(NPPS_LIB)
	message("${NPPS_LIB} library found.")
	target_link_libraries(${CMAKE_PROJECT_NAME}_static PRIVATE npps)
	target_link_libraries(${CMAKE_PROJECT_NAME}_shared PRIVATE npps)
else()
	message(FATAL_ERROR "${NPPS_LIB} library not found.")
endif()

add_dependencies(${CMAKE_PROJECT_NAME}_static openholo_static)
add_dependencies(${CMAKE_PROJECT_NAME}_shared openholo_shared)



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


# 소스 파일 복사
file(GLOB CXX_HEADER_FORMAT "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
file(COPY ${CXX_HEADER_FORMAT} DESTINATION ${CMAKE_SOURCE_DIR}/Reference/include)
message("File Copied")
message("*** ${CMAKE_PROJECT_NAME} end ***")

