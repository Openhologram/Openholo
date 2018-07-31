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

	bool bSupportDouble = false;
	
	const ulonglong n_pixels = this->context_.pixel_number[_X] * this->context_.pixel_number[_Y];
	const int blockSize = 512; //n_threads
	const ulonglong gridSize = (n_pixels + blockSize - 1) / blockSize; //n_blocks

	std::cout << "CUDA Threads : " << blockSize << ", Blocks : " << gridSize << std::endl;
	
	const int n_streams = 3;

	//threads number
	const ulonglong bufferSize = n_pixels * sizeof(Real);
	
	//Host Memory Location
	Real* host_pc_data = this->pc_data_.vertex;
	Real* host_amp_data = this->pc_data_.color;
	
	Real* host_dst = new Real[n_pixels];
	std::memset(host_dst, 0., bufferSize);
	
	GpuConst* host_config = new GpuConst(this->n_points, this->pc_config_.scale, this->pc_config_.offset_depth,
		this->context_.pixel_number, this->pc_config_.tilt_angle,
		this->context_.k, this->context_.pixel_pitch, this->context_.ss);

	//Device(GPU) Memory Location
	Real* device_pc_data;
	HANDLE_ERROR(cudaMalloc((void**)&device_pc_data, this->n_points * 3 * sizeof(Real)));

	Real* device_amp_data;
	HANDLE_ERROR(cudaMalloc((void**)&device_amp_data, this->n_points * sizeof(Real)));

	Real* device_dst;
	HANDLE_ERROR(cudaMalloc((void**)&device_dst, bufferSize));
	HANDLE_ERROR(cudaMemset(device_dst, 0., bufferSize));

	GpuConst* device_config;
	HANDLE_ERROR(cudaMalloc((void**)&device_config, sizeof(GpuConst)));
	HANDLE_ERROR(cudaMemcpy(device_config, host_config, sizeof(GpuConst), cudaMemcpyHostToDevice));

	int stream_points = this->n_points / n_streams;
	int offset = 0;
	for (int i = 0; i < n_streams; ++i) {
		offset = i * stream_points;
		
		HANDLE_ERROR(cudaMemcpy(device_pc_data + 3 * offset, host_pc_data + 3 * offset, stream_points * 3 * sizeof(Real), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(device_amp_data + offset, host_amp_data + offset, stream_points * sizeof(Real), cudaMemcpyHostToDevice));

		cudaGenCghPointCloud(gridSize, blockSize, stream_points, device_pc_data + 3 * offset, device_amp_data + offset, device_dst, device_config);

		HANDLE_ERROR(cudaMemcpy(host_dst, device_dst, bufferSize, cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemset(device_dst, 0., bufferSize));
		for (ulonglong n = 0; n < n_pixels; ++n) {
			holo_encoded[n] += host_dst[n];
		}
	}

	//free memory
	HANDLE_ERROR(cudaFree(device_pc_data));
	HANDLE_ERROR(cudaFree(device_amp_data));
	HANDLE_ERROR(cudaFree(device_dst));
	HANDLE_ERROR(cudaFree(device_config));
	delete[] host_dst;
	delete host_config;


	/*
	//Create Fringe Pattern
	switch (diff_flag)
	{
	case PC_DIFF_RS_ENCODED:
		for (int j = 0; j < n_points; ++j) { //Create Fringe Pattern
			uint idx = 3 * j;
			uint color_idx = pc_data_.n_colors * j;
			Real pcx = pc_data_.vertex[idx + _X] * pc_config_.scale[_X];
			Real pcy = pc_data_.vertex[idx + _Y] * pc_config_.scale[_Y];
			Real pcz = pc_data_.vertex[idx + _Z] * pc_config_.scale[_Z] + pc_config_.offset_depth;
			Real amplitude = pc_data_.color[color_idx];

			diffractEncodedRS_CPU(pn, pp, ss, vec3(pcx, pcy, pcz), k, amplitude, vec2(thetaX, thetaY));
		}
		break;
	case PC_DIFF_RS_NOT_ENCODED:
		Complex<Real> lambda(1, context_.lambda);

		for (int j = 0; j < n_points; ++j) { //Create Fringe Pattern
			uint idx = 3 * j;
			uint color_idx = pc_data_.n_colors * j;
			Real pcx = pc_data_.vertex[idx + _X] * pc_config_.scale[_X];
			Real pcy = pc_data_.vertex[idx + _Y] * pc_config_.scale[_Y];
			Real pcz = pc_data_.vertex[idx + _Z] * pc_config_.scale[_Z] + pc_config_.offset_depth;
			Real amplitude = pc_data_.color[color_idx];

			diffractNotEncodedRS_CPU(pn, pp, ss, vec3(pcx, pcy, pcz), k, amplitude, lambda);
		}		
		break;
	case PC_DIFF_FRESNEL_ENCODED:
		for (int j = 0; j < n_points; ++j) { //Create Fringe Pattern
			uint idx = 3 * j;
			uint color_idx = pc_data_.n_colors * j;
			Real pcx = pc_data_.vertex[idx + _X] * pc_config_.scale[_X];
			Real pcy = pc_data_.vertex[idx + _Y] * pc_config_.scale[_Y];
			Real pcz = pc_data_.vertex[idx + _Z] * pc_config_.scale[_Z] + pc_config_.offset_depth;
			Real amplitude = pc_data_.color[color_idx];

			diffractEncodedFrsn_CPU();
		}
		break;
	case PC_DIFF_FRESNEL_NOT_ENCODED:
		for (int j = 0; j < n_points; ++j) { //Create Fringe Pattern
			uint idx = 3 * j;
			uint color_idx = pc_data_.n_colors * j;
			Real pcx = pc_data_.vertex[idx + _X] * pc_config_.scale[_X];
			Real pcy = pc_data_.vertex[idx + _Y] * pc_config_.scale[_Y];
			Real pcz = pc_data_.vertex[idx + _Z] * pc_config_.scale[_Z] + pc_config_.offset_depth;
			Real amplitude = pc_data_.color[color_idx];

			diffractNotEncodedFrsn_CPU();
		}		
		break;
	}
	*/
}