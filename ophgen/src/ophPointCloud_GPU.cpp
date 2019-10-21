/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install, copy or use the software.
//
//
//                           License Agreement
//                For Open Source Digital Holographic Library
//
// Openholo library is free software;
// you can redistribute it and/or modify it under the terms of the BSD 2-Clause license.
//
// Copyright (C) 2017-2024, Korea Electronics Technology Institute. All rights reserved.
// E-mail : contact.openholo@gmail.com
// Web : http://www.openholo.org
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//  1. Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holder or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
// This software contains opensource software released under GNU Generic Public License,
// NVDIA Software License Agreement, or CUDA supplement to Software License Agreement.
// Check whether software you use contains licensed software.
//
//M*/

#include "ophPointCloud.h"
#include "ophPointCloud_GPU.h"

#include <sys.h> //for LOG() macro

//#define USE_ASYNC
void ophPointCloud::genCghPointCloudGPU(uint diff_flag)
{
	int devID;
	HANDLE_ERROR(cudaGetDevice(&devID));
	cudaDeviceProp devProp;
	HANDLE_ERROR(cudaGetDeviceProperties(&devProp, devID));

#ifdef __DEBUG_LOG_GPU_SPEC_
	cout << "GPU Spec : " << devProp.name << endl;
	cout << " - Global Memory : " << devProp.totalGlobalMem << endl;
	cout << " - Const Memory : " << devProp.totalConstMem << endl;	
	cout << "  - MP(Multiprocessor) Count : " << devProp.multiProcessorCount << endl;
	cout << "  - Maximum Threads per MP : " << devProp.maxThreadsPerMultiProcessor << endl;
	cout << "  - Shared Memory per MP : " << devProp.sharedMemPerMultiprocessor << endl;
	cout << "   - Block per MP : " << devProp.maxThreadsPerMultiProcessor/devProp.maxThreadsPerBlock << endl;
	
	cout << "   - Shared Memory per Block : " << devProp.sharedMemPerBlock << endl;
	cout << "   - Maximum Threads per Block : " << devProp.maxThreadsPerBlock << endl;
	printf("   - Maximum Threads of each Dimension of a Block (X: %d / Y: %d / Z: %d)\n", 
		devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
	printf("   - Maximum Blocks of each Dimension of a Grid, (X: %d / Y: %d / Z: %d)\n", 
		devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
	cout << "   - Device supports allocating Managed Memory on this system : " << devProp.managedMemory << endl;
	cout << endl;
#endif

	bool bSupportDouble = false;

	const ulonglong n_pixels = this->context_.pixel_number[_X] * this->context_.pixel_number[_Y];
	const int blockSize = 512; //n_threads // blockSize < devProp.maxThreadsPerBlock
	const ulonglong gridSize = (n_pixels + blockSize - 1) / blockSize; //n_blocks

	cout << ">>> All " << blockSize * gridSize << " threads in CUDA" << endl;
	cout << ">>> " << blockSize << " threads/block, " << gridSize << " blocks/grid" << endl;

	//const int n_streams = OPH_CUDA_N_STREAM;
	int n_streams;
	if (pc_config_.n_streams == 0)
		n_streams = pc_data_.n_points / 300 + 1;
	else if (pc_config_.n_streams < 0)
	{
		LOG("Invalid value : NumOfStream");
		return;
	}
	else
		n_streams = pc_config_.n_streams;

	LOG(">>> Number Of Stream : %d\n", n_streams);

	//threads number
	const ulonglong bufferSize = n_pixels * sizeof(Real);

	//Host Memory Location
	const int n_colors = this->pc_data_.n_colors;
	Real* host_pc_data = nullptr;
	Real* host_amp_data = this->pc_data_.color;
	Real* host_dst = nullptr;

	// Keep original buffer
	if (is_ViewingWindow) {
		host_pc_data = new Real[n_points * 3];
		transVW(n_points * 3, host_pc_data, this->pc_data_.vertex);
	}
	else {
		host_pc_data = this->pc_data_.vertex;
	}
	
	if ((diff_flag == PC_DIFF_RS) || (diff_flag == PC_DIFF_FRESNEL)) {
		host_dst = new Real[n_pixels * 2];
		std::memset(host_dst, 0., bufferSize * 2);
	}
#ifdef USE_3CHANNEL
	for (uint nColor = 0; nColor < context_.waveNum; nColor++)
	{
		std::memset(host_dst, 0., bufferSize * 2);
		context_.k = (2 * M_PI) / context_.wave_length[nColor];
#endif
		GpuConst* host_config = new GpuConst(
			n_points, n_colors, pc_config_.n_streams,
			pc_config_.scale, pc_config_.offset_depth,
			context_.pixel_number,
			context_.pixel_pitch,
			context_.ss,
			context_.k
		);

		//Device(GPU) Memory Location
		Real* device_pc_data;
		HANDLE_ERROR(cudaMalloc((void**)&device_pc_data, this->n_points * 3 * sizeof(Real)));
		
		Real* device_amp_data;
		HANDLE_ERROR(cudaMalloc((void**)&device_amp_data, this->n_points * n_colors * sizeof(Real)));

		Real* device_dst = nullptr;
		if ((diff_flag == PC_DIFF_RS) || (diff_flag == PC_DIFF_FRESNEL)) {
			HANDLE_ERROR(cudaMalloc((void**)&device_dst, bufferSize * 2));
#ifdef USE_ASYNC
			HANDLE_ERROR(cudaMemsetAsync(device_dst, 0., bufferSize * 2));
#else
			HANDLE_ERROR(cudaMemset(device_dst, 0., bufferSize * 2));
#endif
		}

		GpuConst* device_config = nullptr;
		switch (diff_flag) {
		case PC_DIFF_RS/*_NOT_ENCODED*/: {
#ifdef USE_3CHANNEL
			host_config = new GpuConstNERS(*host_config, this->context_.wave_length[nColor]);
#else
			host_config = new GpuConstNERS(*host_config, this->context_.wave_length[0]);
#endif
			HANDLE_ERROR(cudaMalloc((void**)&device_config, sizeof(GpuConstNERS)));
#ifdef USE_ASYNC
			HANDLE_ERROR(cudaMemcpyAsync(device_config, host_config, sizeof(GpuConstNERS), cudaMemcpyHostToDevice));
#else
			HANDLE_ERROR(cudaMemcpy(device_config, host_config, sizeof(GpuConstNERS), cudaMemcpyHostToDevice));
#endif
			break;
		}
		case PC_DIFF_FRESNEL/*_NOT_ENCODED*/: {
#ifdef USE_3CHANNEL
			host_config = new GpuConstNEFR(*host_config, this->context_.wave_length[nColor]);
#else
			host_config = new GpuConstNEFR(*host_config, this->context_.wave_length[0]);
#endif
			HANDLE_ERROR(cudaMalloc((void**)&device_config, sizeof(GpuConstNEFR)));
#ifdef USE_ASYNC
			HANDLE_ERROR(cudaMemcpyAsync(device_config, host_config, sizeof(GpuConstNEFR), cudaMemcpyHostToDevice));
#else
			HANDLE_ERROR(cudaMemcpy(device_config, host_config, sizeof(GpuConstNEFR), cudaMemcpyHostToDevice));
#endif
			break;
		}
		}

		int stream_points = this->n_points / n_streams;
		int remainder = this->n_points % n_streams;

		int offset = 0;
		for (int i = 0; i < n_streams; ++i) {
			offset = i * stream_points;
			if (i == n_streams - 1) { // 마지막 스트림 연산 시,
				stream_points += remainder;
			}

#ifdef USE_ASYNC
			HANDLE_ERROR(cudaMemcpyAsync(device_pc_data + 3 * offset, host_pc_data + 3 * offset, stream_points * 3 * sizeof(Real), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpyAsync(device_amp_data + n_colors * offset, host_amp_data + n_colors * offset, stream_points * sizeof(Real), cudaMemcpyHostToDevice));
#else
			HANDLE_ERROR(cudaMemcpy(device_pc_data + 3 * offset, host_pc_data + 3 * offset, stream_points * 3 * sizeof(Real), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(device_amp_data + n_colors * offset, host_amp_data + n_colors * offset, stream_points * sizeof(Real), cudaMemcpyHostToDevice));
#endif
			switch (diff_flag) {
			case PC_DIFF_RS/*_NOT_ENCODED*/: {

				cudaGenCghPointCloud_NotEncodedRS(gridSize, blockSize, stream_points, device_pc_data + 3 * offset, device_amp_data + n_colors * offset, device_dst, device_dst + n_pixels, (GpuConstNERS*)device_config);
#ifdef USE_ASYNC
				HANDLE_ERROR(cudaMemcpyAsync(host_dst, device_dst, bufferSize * 2, cudaMemcpyDeviceToHost));
				HANDLE_ERROR(cudaMemsetAsync(device_dst, 0., bufferSize * 2));
#else
				HANDLE_ERROR(cudaMemcpy(host_dst, device_dst, bufferSize * 2, cudaMemcpyDeviceToHost));
				HANDLE_ERROR(cudaMemset(device_dst, 0., bufferSize * 2));
#endif
				for (ulonglong n = 0; n < n_pixels; ++n) {
#ifdef USE_3CHANNEL
					complex_H[nColor][n][_RE] += host_dst[n];
					complex_H[nColor][n][_IM] += host_dst[n + n_pixels];
#else
					(*complex_H)[n][_RE] += host_dst[n];
					(*complex_H)[n][_IM] += host_dst[n + n_pixels];
#endif
					if (n == 0) {
						LOG("GPU > %.16f / %.16f\n", host_dst[n], host_dst[n + n_pixels]);
					}
				}
				break;
			}
			case PC_DIFF_FRESNEL/*_NOT_ENCODED*/: {
				cudaGenCghPointCloud_NotEncodedFrsn(gridSize, blockSize, stream_points, device_pc_data + 3 * offset, device_amp_data + n_colors * offset, device_dst, device_dst + n_pixels, (GpuConstNEFR*)device_config);
#ifdef USE_ASYNC
				HANDLE_ERROR(cudaMemcpyAsync(host_dst, device_dst, bufferSize * 2, cudaMemcpyDeviceToHost));
				HANDLE_ERROR(cudaMemsetAsync(device_dst, 0., bufferSize * 2));
#else
				HANDLE_ERROR(cudaMemcpy(host_dst, device_dst, bufferSize * 2, cudaMemcpyDeviceToHost));
				HANDLE_ERROR(cudaMemset(device_dst, 0., bufferSize * 2));
#endif
				for (ulonglong n = 0; n < n_pixels; ++n) {
#ifdef USE_3CHANNEL
					complex_H[nColor][n][_RE] += host_dst[n];
					complex_H[nColor][n][_IM] += host_dst[n + n_pixels];
#else
					(*complex_H)[n][_RE] += host_dst[n];
					(*complex_H)[n][_IM] += host_dst[n + n_pixels];
#endif
				}
				break;
			} // case
			} // switch
		} // for

		//free memory
		HANDLE_ERROR(cudaFree(device_pc_data));
		HANDLE_ERROR(cudaFree(device_amp_data));
		HANDLE_ERROR(cudaFree(device_dst));
		HANDLE_ERROR(cudaFree(device_config));
		
		delete host_config;
#ifdef USE_3CHANNEL
	}
#endif

	delete[] host_dst;
	if (is_ViewingWindow) {
		delete[] host_pc_data;
	}
}