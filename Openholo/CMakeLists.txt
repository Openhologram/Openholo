cmake_minimum_required(VERSION ${CMAKE_VERSION})
set(CMAKE_PROJECT_NAME "openholo")

message("*** ${CMAKE_PROJECT_NAME} begin ***")
set(CMAKE_VERBOSE_MAKEFILE OFF)
# build file list
set(SRC_FILES
    src/Base.h
    src/comment.h
    src/complex.h
    src/define.h
    src/epsilon.h
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
    src/struct.h
    src/sys.h
    src/typedef.h
    src/vec.h
    src/ophKernel.cuh
    src/epsilon.cpp
    src/ImgCodecOhc.cpp
    src/ImgControl.cpp
    src/Openholo.cpp
    src/PLYparser.cpp
    src/sys.cpp
    src/vec.cpp
)
# 컴파일
add_library(${CMAKE_PROJECT_NAME} OBJECT ${SRC_FILES})

# 추가 포함 디렉터리
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/Reference/include)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIR})

# 추가 정의
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE "-DOPH_EXPORT")

add_library(${CMAKE_PROJECT_NAME}_static STATIC $<TARGET_OBJECTS:${CMAKE_PROJECT_NAME}>)
add_library(${CMAKE_PROJECT_NAME}_shared SHARED $<TARGET_OBJECTS:${CMAKE_PROJECT_NAME}>)


# 전역 변수의 중복 정의를 허용
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcommon")
# C++ 언어 규칙을 완전히 준수하지 않는 코드일때, 오류를 경고로 출력
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
# OpenMP를 사용하도록 함. 사용 시 _OPENMP 가 정의됨
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")



# 추가 링크 디렉터리
target_link_directories(${CMAKE_PROJECT_NAME}_static PRIVATE ${FFTW3_LIBRARY_DIRS})
target_link_directories(${CMAKE_PROJECT_NAME}_static PRIVATE ${CUDAToolkit_LIBRARY_DIR})
target_link_directories(${CMAKE_PROJECT_NAME}_shared PRIVATE ${FFTW3_LIBRARY_DIRS})
target_link_directories(${CMAKE_PROJECT_NAME}_shared PRIVATE ${CUDAToolkit_LIBRARY_DIR})

find_library(FFTW3_LIB NAMES ${FFTW3_LIBRARIES}_threads PATHS ${FFTW3_LIBRARY_DIRS})
if(FFTW3_LIB)
	message("${FFTW3_LIB} library found.")
	target_link_libraries(${CMAKE_PROJECT_NAME}_static PRIVATE ${FFTW3_LIBRARIES}_threads)
	target_link_libraries(${CMAKE_PROJECT_NAME}_shared PRIVATE ${FFTW3_LIBRARIES}_threads)
else()
	message(FATAL_ERROR "${FFTW3_LIB} library not found.")
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

# 파일 복사
file(GLOB CXX_HEADER_FORMAT "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
file(GLOB CUDA_HEADER_FORMAT "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh")
file(COPY ${CXX_HEADER_FORMAT} DESTINATION ${CMAKE_SOURCE_DIR}/Reference/include)
file(COPY ${CUDA_HEADER_FORMAT} DESTINATION ${CMAKE_SOURCE_DIR}/Reference/include)

message("*** ${CMAKE_PROJECT_NAME} end ***")

