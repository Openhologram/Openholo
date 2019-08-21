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
//M*/
#include "ophWRP.h"
#include "ophWRP_GPU.h"
#include "sys.h"

double ophWRP::calculateWRPGPU(void)
{
	//	auto time_start = CUR_TIME;

	if (p_wrp_) delete[] p_wrp_;
	p_wrp_ = new oph::Complex<Real>[context_.pixel_number[_X] * context_.pixel_number[_Y]];
	memset(p_wrp_, 0.0, sizeof(oph::Complex<Real>) * context_.pixel_number[_X] * context_.pixel_number[_Y]);

	prepareInputdataGPU();

	return 0;

}

void ophWRP::prepareInputdataGPU()
{
	// GPU information
	int devID;
	HANDLE_ERROR(cudaGetDevice(&devID));
	cudaDeviceProp devProp;
	HANDLE_ERROR(cudaGetDeviceProperties(&devProp, devID));

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

	const ulonglong n_points = this->obj_.n_points;

	const int blockSize = 512; //n_threads
							   //	const ulonglong gridSize = (n_pixels + blockSize - 1) / blockSize; //n_blocks
	const ulonglong gridSize = (n_points + blockSize - 1) / blockSize; //n_blocks


	std::cout << ">>> All " << blockSize * gridSize << " threads in CUDA" << std::endl;
	std::cout << ">>> " << blockSize << " threads/block, " << gridSize << " blocks/grid" << std::endl;


	//threads number
	//const ulonglong bufferSize = n_pixels * sizeof(Real);

	//Host Memory Location
	const int n_colors = this->obj_.n_colors;
	Real* host_pc_data = this->obj_.vertex;
	Real* host_amp_data = obj_.color;


	float wz = this->pc_config_.wrp_location - zmax_;
	float wm = round(fabs(wz*tan(this->context_.wave_length[0] / (2 * this->context_.pixel_pitch[_X])) / this->context_.pixel_pitch[_X]));


	//	Real* host_dst = new Real[n_pixels * 2];
	//	std::memset(host_dst, 0., bufferSize * 2);

	Real* pc_index = new Real[obj_.n_points * 3];
	memset(pc_index, 0.0, sizeof(Real) * obj_.n_points * 3);

	Real ppx = this->context_.pixel_pitch[_X];

	WRPGpuConst* host_config = new WRPGpuConst(
		this->obj_.n_points, n_colors, 1,
		this->context_.pixel_number,
		this->context_.pixel_pitch,
		this->pc_config_.wrp_location,
		this->pc_config_.propagation_distance, this->zmax_,
		this->context_.k, this->context_.wave_length[0]
	);

	//Device(GPU) Memory Location
	Real* device_pc_data;
	HANDLE_ERROR(cudaMalloc((void**)&device_pc_data, this->n_points * 3 * sizeof(Real)));

	Real* device_amp_data;
	HANDLE_ERROR(cudaMalloc((void**)&device_amp_data, this->n_points * n_colors * sizeof(Real)));

	Real* device_pc_xindex;
	HANDLE_ERROR(cudaMalloc((void**)&device_pc_xindex, this->n_points * 3 * sizeof(Real)));

	WRPGpuConst* device_config = nullptr;

	HANDLE_ERROR(cudaMalloc((void**)&device_config, sizeof(WRPGpuConst)));
	HANDLE_ERROR(cudaMemcpy(device_config, host_config, sizeof(WRPGpuConst), cudaMemcpyHostToDevice));


	HANDLE_ERROR(cudaMemcpy(device_pc_data, host_pc_data, n_points * 3 * sizeof(Real), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(device_amp_data, host_amp_data, n_points * n_colors * sizeof(Real), cudaMemcpyHostToDevice));

	cudaGenindexx(gridSize, blockSize, n_points, device_pc_data, device_pc_xindex, (WRPGpuConst*)device_config);

	HANDLE_ERROR(cudaMemcpy(pc_index, device_pc_xindex, sizeof(Real) * 3 * n_points, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(device_pc_data));


	// calculates WRP with CUDA

	//cuda obj dst
	const ulonglong n_pixels = this->context_.pixel_number[_X] * this->context_.pixel_number[_Y];
	const ulonglong bufferSize = n_pixels * sizeof(Real);


	Real *host_obj_dst = new Real[n_pixels];
	std::memset(host_obj_dst, 0., bufferSize);

	Real *host_amp_dst = new Real[n_pixels];
	std::memset(host_amp_dst, 0., bufferSize);

	Real *device_obj_dst;
	HANDLE_ERROR(cudaMalloc((void**)&device_obj_dst, n_pixels * sizeof(Real)));
	HANDLE_ERROR(cudaMemset(device_obj_dst, 0., bufferSize));

	Real *device_amp_dst;
	HANDLE_ERROR(cudaMalloc((void**)&device_amp_dst, n_pixels * sizeof(Real)));
	HANDLE_ERROR(cudaMemset(device_amp_dst, 0., bufferSize));


	cudaGetObjDst(gridSize, blockSize, n_points, device_pc_xindex, device_obj_dst, (WRPGpuConst*)device_config);
	cudaGetAmpDst(gridSize, blockSize, n_points, device_pc_xindex, device_amp_data, device_amp_dst, (WRPGpuConst*)device_config);

	HANDLE_ERROR(cudaMemcpy(host_obj_dst, device_obj_dst, bufferSize, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(host_amp_dst, device_amp_dst, bufferSize, cudaMemcpyDeviceToHost));

	const ulonglong gridSize2 = (n_pixels + blockSize - 1) / blockSize; //n_blocks

	Real* device_dst;
	HANDLE_ERROR(cudaMalloc((void**)&device_dst, n_pixels * 2 * sizeof(Real)));
	HANDLE_ERROR(cudaMemset(device_dst, 0., bufferSize * 2));

	Real* host_dst = new Real[n_pixels * 2];
	std::memset(host_dst, 0., bufferSize * 2);


	// cuda WRP
	cudaGenWRP(gridSize2, blockSize, n_points, device_obj_dst, device_amp_dst, device_dst, device_dst + n_pixels, (WRPGpuConst*)device_config);

	HANDLE_ERROR(cudaMemcpy(host_dst, device_dst, bufferSize * 2, cudaMemcpyDeviceToHost));

	for (ulonglong n = 0; n < n_pixels; ++n) {
		if (host_dst[n] != 0)
		{
			p_wrp_[n][_RE] = host_dst[n];
			p_wrp_[n][_IM] = host_dst[n + n_pixels];
		}
	}


	*(complex_H) = p_wrp_;

	//free memory
	HANDLE_ERROR(cudaFree(device_amp_data));
	HANDLE_ERROR(cudaFree(device_pc_xindex));
	HANDLE_ERROR(cudaFree(device_config));
	delete host_config;

}

