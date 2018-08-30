#include "ophPointCloud.h"
#include "ophPointCloud_GPU.h"

#include <sys.h> //for LOG() macro


void ophPointCloud::genCghPointCloudGPU(uint diff_flag)
{
	int devID;
	HANDLE_ERROR(cudaGetDevice(&devID));
	cudaDeviceProp devProp;
	HANDLE_ERROR(cudaGetDeviceProperties(&devProp, devID));

	//그래픽 카드 정보 받아서 메모리 용량 및 스트림 분할하기 / n_streams 분할 구간 결정 추가
	//그래픽 카드 정보 받아서 float / double 여부 결정
	//GpuConst의 cu 파일 내 상수화 : GPU로 전송 방법 결정

#ifdef __DEBUG_LOG_GPU_SPEC_
	std::cout << "GPU Spec : " << devProp.name << std::endl;
	std::cout << "	- Global Memory : " << devProp.totalGlobalMem << std::endl;
	std::cout << "	- Const Memory : " << devProp.totalConstMem << std::endl;
	std::cout << "	- Shared Memory / SM : " << devProp.sharedMemPerMultiprocessor << std::endl;
	std::cout << "	- Shared Memory / Block : " << devProp.sharedMemPerBlock << std::endl;
	std::cout << "	- SM Counter : " << devProp.multiProcessorCount << std::endl;
	std::cout << "	- Maximum Threads / SM : " << devProp.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "	- Maximum Threads / Block : " << devProp.maxThreadsPerBlock << std::endl;
	std::cout << "	- Maximum Threads of each Dimension of a Block, X : " << devProp.maxThreadsDim[0] << ", Y : " << devProp.maxThreadsDim[1] << ", Z : " << devProp.maxThreadsDim[2] << std::endl;
	std::cout << "	- Maximum Blocks of each Dimension of a Grid, X : " << devProp.maxGridSize[0] << ", Y : " << devProp.maxGridSize[1] << ", Z : " << devProp.maxGridSize[2] << std::endl;
	std::cout << "	- Device supports allocating Managed Memory on this system : " << devProp.managedMemory << std::endl;
	std::cout << std::endl;
#endif

	bool bSupportDouble = false;

	const ulonglong n_pixels = this->context_.pixel_number[_X] * this->context_.pixel_number[_Y];
	const int blockSize = 512; //n_threads
	const ulonglong gridSize = (n_pixels + blockSize - 1) / blockSize; //n_blocks

	std::cout << ">>> All " << blockSize * gridSize << " threads in CUDA" << std::endl;
	std::cout << ">>> " << blockSize << " threads/block, " << gridSize << " blocks/grid" << std::endl;

	const int n_streams = 3;

	//threads number
	const ulonglong bufferSize = n_pixels * sizeof(Real);

	//Host Memory Location
	const int n_colors = this->pc_data_.n_colors;
	Real* host_pc_data = this->pc_data_.vertex;
	Real* host_amp_data = this->pc_data_.color;
	Real* host_dst = nullptr;
	if ((diff_flag == PC_DIFF_RS_ENCODED) || (diff_flag == PC_DIFF_FRESNEL_ENCODED)) {
		host_dst = new Real[n_pixels];
		std::memset(host_dst, 0., bufferSize);
	}
	else if ((diff_flag == PC_DIFF_RS_NOT_ENCODED) || (diff_flag == PC_DIFF_FRESNEL_NOT_ENCODED)) {
		host_dst = new Real[n_pixels * 2];
		std::memset(host_dst, 0., bufferSize * 2);
	}

	GpuConst* host_config = new GpuConst(
		this->n_points, n_colors,
		this->pc_config_.scale, this->pc_config_.offset_depth,
		this->context_.pixel_number,
		this->context_.pixel_pitch,
		this->context_.ss,
		this->context_.k
	);

	//Device(GPU) Memory Location
	Real* device_pc_data;
	HANDLE_ERROR(cudaMalloc((void**)&device_pc_data, this->n_points * 3 * sizeof(Real)));

	Real* device_amp_data;
	HANDLE_ERROR(cudaMalloc((void**)&device_amp_data, this->n_points * n_colors * sizeof(Real)));

	Real* device_dst = nullptr;
	if ((diff_flag == PC_DIFF_RS_ENCODED) || (diff_flag == PC_DIFF_FRESNEL_ENCODED)) {
		HANDLE_ERROR(cudaMalloc((void**)&device_dst, bufferSize));
		HANDLE_ERROR(cudaMemset(device_dst, 0., bufferSize));
	}
	else if ((diff_flag == PC_DIFF_RS_NOT_ENCODED) || (diff_flag == PC_DIFF_FRESNEL_NOT_ENCODED)) {
		HANDLE_ERROR(cudaMalloc((void**)&device_dst, bufferSize * 2));
		HANDLE_ERROR(cudaMemset(device_dst, 0., bufferSize * 2));
	}

	GpuConst* device_config = nullptr;
	switch (diff_flag) {
	case PC_DIFF_RS_ENCODED: {
		host_config = new GpuConstERS(*host_config, this->pc_config_.tilt_angle);
		HANDLE_ERROR(cudaMalloc((void**)&device_config, sizeof(GpuConstERS)));
		HANDLE_ERROR(cudaMemcpy(device_config, host_config, sizeof(GpuConstERS), cudaMemcpyHostToDevice));
		break;
	}
	case PC_DIFF_FRESNEL_ENCODED: {
		break;
	}
	case PC_DIFF_RS_NOT_ENCODED: {
		host_config = new GpuConstNERS(*host_config, this->context_.lambda);
		HANDLE_ERROR(cudaMalloc((void**)&device_config, sizeof(GpuConstNERS)));
		HANDLE_ERROR(cudaMemcpy(device_config, host_config, sizeof(GpuConstNERS), cudaMemcpyHostToDevice));
		break;
	}
	case PC_DIFF_FRESNEL_NOT_ENCODED: {
		host_config = new GpuConstNEFR(*host_config, this->context_.lambda);
		HANDLE_ERROR(cudaMalloc((void**)&device_config, sizeof(GpuConstNEFR)));
		HANDLE_ERROR(cudaMemcpy(device_config, host_config, sizeof(GpuConstNEFR), cudaMemcpyHostToDevice));
		break;
	}
	}

	int stream_points = this->n_points / n_streams;
	int offset = 0;
	for (int i = 0; i < n_streams; ++i) {
		offset = i * stream_points;

		HANDLE_ERROR(cudaMemcpy(device_pc_data + 3 * offset, host_pc_data + 3 * offset, stream_points * 3 * sizeof(Real), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(device_amp_data + n_colors * offset, host_amp_data + n_colors * offset, stream_points * sizeof(Real), cudaMemcpyHostToDevice));

		switch (diff_flag) {
		case PC_DIFF_RS_ENCODED: {
			cudaGenCghPointCloud_EncodedRS(gridSize, blockSize, stream_points, device_pc_data + 3 * offset, device_amp_data + n_colors * offset, device_dst, (GpuConstERS*)device_config);

			HANDLE_ERROR(cudaMemcpy(host_dst, device_dst, bufferSize, cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemset(device_dst, 0., bufferSize));
			for (ulonglong n = 0; n < n_pixels; ++n) {
				this->holo_encoded[n] += host_dst[n];
			}
			break;
		}
		case PC_DIFF_FRESNEL_ENCODED: {
			break;
		}
		case PC_DIFF_RS_NOT_ENCODED: {
			cudaGenCghPointCloud_NotEncodedRS(gridSize, blockSize, stream_points, device_pc_data + 3 * offset, device_amp_data + n_colors * offset, device_dst, device_dst + n_pixels, (GpuConstNERS*)device_config);

			HANDLE_ERROR(cudaMemcpy(host_dst, device_dst, bufferSize * 2, cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemset(device_dst, 0., bufferSize * 2));
			for (ulonglong n = 0; n < n_pixels; ++n) {
				this->holo_gen[n][_RE] += host_dst[n];
				this->holo_gen[n][_IM] += host_dst[n + n_pixels];
			}
			break;
		}
		case PC_DIFF_FRESNEL_NOT_ENCODED: {
			cudaGenCghPointCloud_NotEncodedFrsn(gridSize, blockSize, stream_points, device_pc_data + 3 * offset, device_amp_data + n_colors * offset, device_dst, device_dst + n_pixels, (GpuConstNEFR*)device_config);

			HANDLE_ERROR(cudaMemcpy(host_dst, device_dst, bufferSize * 2, cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemset(device_dst, 0., bufferSize * 2));
			for (ulonglong n = 0; n < n_pixels; ++n) {
				this->holo_gen[n][_RE] += host_dst[n];
				this->holo_gen[n][_IM] += host_dst[n + n_pixels];
			}
			break;
		}
		}
	}

	//free memory
	HANDLE_ERROR(cudaFree(device_pc_data));
	HANDLE_ERROR(cudaFree(device_amp_data));
	HANDLE_ERROR(cudaFree(device_dst));
	HANDLE_ERROR(cudaFree(device_config));
	delete[] host_dst;
	delete host_config;
}