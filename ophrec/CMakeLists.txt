cmake_minimum_required(VERSION ${CMAKE_VERSION})
set(CMAKE_PROJECT_NAME "ophrec")

message("*** ${CMAKE_PROJECT_NAME} begin ***")

# 빌드 타입 설정 *추가시 링크 오류*
#if(NOT CMAKE_BUILD_TYPE)
#	set(CMAKE_BUILD_TYPE Debug)
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLGAS} -g")
#endif()


add_definitions(-DOPH_EXPORT)

include_directories(${CMAKE_SOURCE_DIR}/Reference/include)
include_directories(${CUDA_INCLUDE_DIRS})

link_directories(${CMAKE_SOURCE_DIR}/../Reference/lib)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
link_directories(usr/local/lib/fftw3)

# 바이너리 디렉토리 설정
set(CMAKE_PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR})
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# 전역 변수의 중복 정의를 허용
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcommon")
# C++ 언어 규칙을 완전히 준수하지 않는 코드일때, 오류를 경고로 출력
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
# OpenMP를 사용하도록 함. 사용 시 _OPENMP 가 정의됨
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
# Position Independent Code 컴파일러가 생성하는 코드가 다른 위치에 링크될 때도 정상동작하도록 작동
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
# 반복문을 펼쳐서 반복 횟수를 줄여 성능 개선
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLGAS} -funroll-loops")
# 함수 호출을 줄여 성능 개선
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLGAS} -finline-functions")
# 현재 시스템 아키텍처에 최적화된 코드를 생성하여 성능 개선
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLGAS} -march=native")
# 프레임 포인터를 생략하여 함수 호출 시간을 줄여 성능 개선
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
# 최적화 옵션 -O0: 최적화x / -O1: 적은 최적화 수준 / -O2: 일반적인 최적화 수준 / -O3: 최대한 최적화 수준
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# signed unsigned 비교
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wsign-compare")
# 모든 경고
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
# 초기화 관련 경고
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-field-initializers")
# 오버로드 가상함수 관련
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Woverloaded-virtual")
# 미사용 파라미터 관련
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
# 미사용 변수 련
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")


# no warning
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_52 -Xcompiler")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")


if (CMAKE_COMPILER_IS_GNUCC)
	message("GNUCC COMPILER")
	# remove warning
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()
if (MSVC)
	message("MSVC COMPILER")
endif()

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


add_library(${CMAKE_PROJECT_NAME} SHARED ${SRC_FILES})
add_library(${CMAKE_PROJECT_NAME}_static STATIC ${SRC_FILES})
set_target_properties(
	${CMAKE_PROJECT_NAME}_static
	PROPERTIES
	OUTPUT_NAME
	${CMAKE_PROJECT_NAME}
)


add_dependencies(${CMAKE_PROJECT_NAME} ${CMAKE_SOLUTION_NAME}_static)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Visual Studio 전용 최적화 비활성화
#add_compile_options(/Od)

target_link_libraries(${CMAKE_PROJECT_NAME} ${FFTW3_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} openholo)
target_link_libraries(${CMAKE_PROJECT_NAME} cudart)
target_link_libraries(${CMAKE_PROJECT_NAME} cufft)
target_link_libraries(${CMAKE_PROJECT_NAME} cuda)


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
install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION ${CMAKE_PROJECT_ROOT_DIR}/Reference/lib/)


# 소스 파일 복사
file(GLOB CXX_HEADER_FORMAT "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
file(COPY ${CXX_HEADER_FORMAT} DESTINATION ${CMAKE_SOURCE_DIR}/Reference/include)
message("File Copied")
message("*** ${CMAKE_PROJECT_NAME} end ***")

