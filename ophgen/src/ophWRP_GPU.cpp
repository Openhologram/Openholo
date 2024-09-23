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
#include "CUDA.h"

void ophWRP::calculateWRPGPU()
{
	LOG("%s\n", __FUNCTION__);
	LOG("\tMemory Allocation : ");
	auto begin = CUR_TIME;
	auto step = CUR_TIME;

	CUDA *cuda = CUDA::getInstance();

	bool bSupportDouble = false;

	const ulonglong n_points = obj_.n_points;

	
	int blockSize = cuda->getMaxThreads(0); //n_threads

	ulonglong gridSize = (n_points + blockSize - 1) / blockSize; //n_blocks
	   
	//threads number

	//Host Memory Location
	const int n_colors = obj_.n_colors;
	//Real* host_pc_data = scaledVertex;//obj_.vertex;
	Vertex* host_pc_data = scaledVertex;//obj_.vertex;
	//Real* host_amp_data = obj_.color;
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const long long int pnXY = pnX * pnY;
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];
	const Real distance = wrp_config_.propagation_distance;
	const uint nChannel = context_.waveNum;

	//Device(GPU) Memory Location
	//Real* device_pc_data;
	//HANDLE_ERROR(cudaMalloc((void**)&device_pc_data, n_points * 3 * sizeof(Real)));
	//Real* device_amp_data;
	//HANDLE_ERROR(cudaMalloc((void**)&device_amp_data, n_points * n_colors * sizeof(Real)));
	Vertex* device_pc_data = nullptr;
	HANDLE_ERROR(cudaMalloc((void**)&device_pc_data, n_points * sizeof(Vertex)));
	WRPGpuConst* device_config = nullptr;
	HANDLE_ERROR(cudaMalloc((void**)&device_config, sizeof(WRPGpuConst)));

	//cuda obj dst
	const ulonglong bufferSize = pnXY * sizeof(Real);

	ulonglong gridSize2 = (pnXY + blockSize - 1) / blockSize; //n_blocks
	ulonglong gridSize3 = (pnXY * 4 + blockSize - 1) / blockSize;
	cuDoubleComplex* device_dst = nullptr;
	//Real* device_dst;
	HANDLE_ERROR(cudaMalloc((void**)&device_dst, pnXY * sizeof(cuDoubleComplex)));

	cuDoubleComplex* src;
	cufftDoubleComplex *fftsrc;
	cufftDoubleComplex *fftdst;
	HANDLE_ERROR(cudaMalloc((void**)&src, pnXY * 4 * sizeof(cuDoubleComplex)));
	HANDLE_ERROR(cudaMalloc((void**)&fftsrc, pnXY * 4 * sizeof(cufftDoubleComplex)));
	HANDLE_ERROR(cudaMalloc((void**)&fftdst, pnXY * 4 * sizeof(cufftDoubleComplex)));	
	//HANDLE_ERROR(cudaMemcpy(device_pc_data, host_pc_data, n_points * 3 * sizeof(Real), cudaMemcpyHostToDevice));
	//HANDLE_ERROR(cudaMemcpy(device_amp_data, host_amp_data, n_points * n_colors * sizeof(Real), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(device_pc_data, host_pc_data, n_points * sizeof(Vertex), cudaMemcpyHostToDevice));
	bool bRandomPhase = GetRandomPhase();

	LOG("%lf (s)\n", ELAPSED_TIME(step, CUR_TIME));


	//Real wz = wrp_config_.wrp_location - zmax_;
	for (uint ch = 0; ch < nChannel; ch++)
	{
		LOG("\tCUDA Gen WRP <<<%llu, %d>>> : ", gridSize, blockSize);
		HANDLE_ERROR(cudaMemset(src, 0, pnXY * 4 * sizeof(cuDoubleComplex)));
		HANDLE_ERROR(cudaMemset(fftsrc, 0, pnXY * 4 * sizeof(cufftDoubleComplex)));
		HANDLE_ERROR(cudaMemset(fftdst, 0, pnXY * 4 * sizeof(cufftDoubleComplex)));

		step = CUR_TIME;

		Real lambda = context_.wave_length[ch];
		Real k = context_.k = (2 * M_PI / lambda);
		int nAdd = ch;

		WRPGpuConst* host_config = new WRPGpuConst(
			obj_.n_points, n_colors, 1,
			context_.pixel_number,
			context_.pixel_pitch,
			wrp_config_.wrp_location,
			wrp_config_.propagation_distance, zmax_,
			k, lambda, bRandomPhase, nAdd
		);
		HANDLE_ERROR(cudaMemcpy(device_config, host_config, sizeof(WRPGpuConst), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemset(device_dst, 0., pnXY * sizeof(cuDoubleComplex)));

		// cuda WRP
		cudaGenWRP(gridSize, blockSize, n_points, device_pc_data, device_dst, (WRPGpuConst*)device_config);

		LOG("%lf (s)\n", ELAPSED_TIME(step, CUR_TIME));
		
		// 20200824_mwnam_
		cudaError error = cudaGetLastError();
		if (error != cudaSuccess) {
			LOG("cudaGetLastError(): %s\n", cudaGetErrorName(error));
			if (error == cudaErrorLaunchOutOfResources) {
				ch--;
				blockSize /= 2;
				gridSize = (n_points + blockSize - 1) / blockSize;
				gridSize2 = (pnXY + blockSize - 1) / blockSize;
				gridSize3 = (pnXY * 4 + blockSize - 1) / blockSize;
				cuda->setCurThreads(blockSize);
				delete host_config;
				continue;
			}
		}
		LOG("\tCUDA FresnelPropagation <<<%llu, %d>>> : ", gridSize2, blockSize);
		step = CUR_TIME;
		cudaFresnelPropagationWRP(gridSize2, gridSize3, blockSize, pnX, pnY, device_dst, src, fftsrc, fftdst, (WRPGpuConst*)device_config);
		HANDLE_ERROR(cudaMemcpy(complex_H[ch], device_dst, sizeof(cuDoubleComplex) * pnXY, cudaMemcpyDeviceToHost));

		LOG("%lf (s)\n", ELAPSED_TIME(step, CUR_TIME));
		// 20200824_mwnam_
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			LOG("cudaGetLastError(): %s\n", cudaGetErrorName(error));
		}
		delete host_config;
	}

	//free memory
	HANDLE_ERROR(cudaFree(src));
	HANDLE_ERROR(cudaFree(fftsrc));
	HANDLE_ERROR(cudaFree(fftdst));
	HANDLE_ERROR(cudaFree(device_dst));
	HANDLE_ERROR(cudaFree(device_pc_data));
	//HANDLE_ERROR(cudaFree(device_amp_data));
	HANDLE_ERROR(cudaFree(device_config));
	LOG("Total : %lf (s)\n", ELAPSED_TIME(begin, CUR_TIME));
}