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
#ifdef CHECK_PROC_TIME
	auto begin = CUR_TIME;
#endif
	if (p_wrp_) delete[] p_wrp_;
	p_wrp_ = new oph::Complex<Real>[context_.pixel_number[_X] * context_.pixel_number[_Y]];
	memset(p_wrp_, 0.0, sizeof(oph::Complex<Real>) * context_.pixel_number[_X] * context_.pixel_number[_Y]);

	prepareInputdataGPU();
#ifdef CHECK_PROC_TIME
	auto end = CUR_TIME;
	LOG("\n%s : %lf(s)\n\n", __FUNCTION__, ((std::chrono::duration<Real>)(end - begin)).count());
#endif
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
	cout << "GPU Spec : " << devProp.name << endl;
	cout << "	- Global Memory : " << devProp.totalGlobalMem << endl;
	cout << "	- Const Memory : " << devProp.totalConstMem << endl;
	cout << "	- Shared Memory / SM : " << devProp.sharedMemPerMultiprocessor << endl;
	cout << "	- Shared Memory / Block : " << devProp.sharedMemPerBlock << endl;
	cout << "	- SM Counter : " << devProp.multiProcessorCount << endl;
	cout << "	- Maximum Threads / SM : " << devProp.maxThreadsPerMultiProcessor << endl;
	cout << "	- Maximum Threads / Block : " << devProp.maxThreadsPerBlock << endl;
	cout << "	- Maximum Threads of each Dimension of a Block, X : " << devProp.maxThreadsDim[0] << ", Y : " << devProp.maxThreadsDim[1] << ", Z : " << devProp.maxThreadsDim[2] << endl;
	cout << "	- Maximum Blocks of each Dimension of a Grid, X : " << devProp.maxGridSize[0] << ", Y : " << devProp.maxGridSize[1] << ", Z : " << devProp.maxGridSize[2] << endl;
	cout << "	- Device supports allocating Managed Memory on this system : " << devProp.managedMemory << endl;
	cout << endl;
#endif

	bool bSupportDouble = false;

	const ulonglong n_points = obj_.n_points;

	const int blockSize = 512; //n_threads


	const ulonglong gridSize = (n_points + blockSize - 1) / blockSize; //n_blocks


	cout << ">>> All " << blockSize * gridSize << " threads in CUDA" << endl;
	cout << ">>> " << blockSize << " threads/block, " << gridSize << " blocks/grid" << endl;

	//threads number

	//Host Memory Location
	const int n_colors = obj_.n_colors;
	Real* host_pc_data = scaledVertex;//obj_.vertex;
	Real* host_amp_data = obj_.color;
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];
	const Real distance = wrp_config_.propagation_distance;
	const uint nChannel = context_.waveNum;

	Real* pc_index = new Real[obj_.n_points * 3];
	memset(pc_index, 0.0, sizeof(Real) * obj_.n_points * 3);

	float wz = wrp_config_.wrp_location - zmax_;
	
	//Device(GPU) Memory Location
	Real* device_pc_data;
	HANDLE_ERROR(cudaMalloc((void**)&device_pc_data, n_points * 3 * sizeof(Real)));
	Real* device_amp_data;
	HANDLE_ERROR(cudaMalloc((void**)&device_amp_data, n_points * n_colors * sizeof(Real)));
	Real* device_pc_xindex;
	HANDLE_ERROR(cudaMalloc((void**)&device_pc_xindex, n_points * 3 * sizeof(Real)));

	WRPGpuConst* device_config = nullptr;
	HANDLE_ERROR(cudaMalloc((void**)&device_config, sizeof(WRPGpuConst)));

	//cuda obj dst
	const ulonglong bufferSize = pnXY * sizeof(Real);

	Real *host_obj_dst = new Real[pnXY];
	memset(host_obj_dst, 0., bufferSize);

	Real *host_amp_dst = new Real[pnXY];
	memset(host_amp_dst, 0., bufferSize);

	Real *device_obj_dst;
	HANDLE_ERROR(cudaMalloc((void**)&device_obj_dst, bufferSize));

	Real *device_amp_dst;
	HANDLE_ERROR(cudaMalloc((void**)&device_amp_dst, bufferSize));


	const ulonglong gridSize2 = (pnXY + blockSize - 1) / blockSize; //n_blocks

	Real* device_dst;
	HANDLE_ERROR(cudaMalloc((void**)&device_dst, bufferSize * 2));

	Real* host_dst = new Real[pnXY * 2];
	
	for (uint ch = 0; ch < nChannel; ch++) {

		Real lambda = context_.wave_length[ch];
		Real k = context_.k = (2 * M_PI / lambda);
		float wm = round(fabs(wz * tan(lambda / (2 * ppX)) / ppX));

		HANDLE_ERROR(cudaMemset(device_pc_xindex, 0., n_points * 3 * sizeof(Real)));

		WRPGpuConst* host_config = new WRPGpuConst(
			obj_.n_points, n_colors, 1,
			context_.pixel_number,
			context_.pixel_pitch,
			wrp_config_.wrp_location,
			wrp_config_.propagation_distance, zmax_,
			k, lambda
		);

		HANDLE_ERROR(cudaMemcpy(device_config, host_config, sizeof(WRPGpuConst), cudaMemcpyHostToDevice));
			   
		HANDLE_ERROR(cudaMemcpy(device_pc_data, host_pc_data, n_points * 3 * sizeof(Real), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(device_amp_data, host_amp_data, n_points * n_colors * sizeof(Real), cudaMemcpyHostToDevice));
		cudaGenindexx(gridSize, blockSize, n_points, device_pc_data, device_pc_xindex, (WRPGpuConst*)device_config);

		HANDLE_ERROR(cudaMemcpy(pc_index, device_pc_xindex, sizeof(Real) * 3 * n_points, cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemset(device_obj_dst, 0., bufferSize));
		HANDLE_ERROR(cudaMemset(device_amp_dst, 0., bufferSize));

		cudaGetObjDst(gridSize, blockSize, n_points, device_pc_xindex, device_obj_dst, (WRPGpuConst*)device_config);
		cudaGetAmpDst(gridSize, blockSize, n_points, device_pc_xindex, device_amp_data, device_amp_dst, (WRPGpuConst*)device_config);

		HANDLE_ERROR(cudaMemcpy(host_obj_dst, device_obj_dst, bufferSize, cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(host_amp_dst, device_amp_dst, bufferSize, cudaMemcpyDeviceToHost));

		HANDLE_ERROR(cudaMemset(device_dst, 0., bufferSize * 2));
		memset(host_dst, 0., bufferSize * 2);

		// cuda WRP
		cudaGenWRP(gridSize2, blockSize, n_points, device_obj_dst, device_amp_dst, device_dst, device_dst + pnXY, (WRPGpuConst*)device_config);

		HANDLE_ERROR(cudaMemcpy(host_dst, device_dst, bufferSize * 2, cudaMemcpyDeviceToHost));

		for (ulonglong n = 0; n < pnXY; ++n) {
			if (host_dst[n] != 0)
			{
				p_wrp_[n][_RE] = host_dst[n];
				p_wrp_[n][_IM] = host_dst[n + pnXY];
			}
		}

		fresnelPropagation(p_wrp_, complex_H[ch], distance, ch);

		delete host_config;
	}
	
	HANDLE_ERROR(cudaFree(device_pc_data));
	   	 
	//*(complex_H) = p_wrp_;

	//free memory
	HANDLE_ERROR(cudaFree(device_amp_data));
	HANDLE_ERROR(cudaFree(device_pc_xindex));
	HANDLE_ERROR(cudaFree(device_config));

}

