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

#include "ophLightField_GPU.h"
#include "CUDA.h"
#include "sys.h"

void ophLF::convertLF2ComplexField_GPU()
{
	LOG("%s\n", __FUNCTION__);
	auto begin = CUR_TIME;

	CUDA *pCUDA = CUDA::getInstance();
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const int pnXY = pnX * pnY;
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];
	const int nX = num_image[_X];
	const int nY = num_image[_Y];
	const int N = nX * nY;
	const int rX = resolution_image[_X];
	const int rY = resolution_image[_Y];
	const int R = rX * rY;
	const Real distance = distanceRS2Holo;
	const int nWave = context_.waveNum;
	bool bRandomPhase = GetRandomPhase();

	// device
	uchar1** device_LF;
	uchar** device_LFData;
	cufftDoubleComplex* device_FFT_src;
	cufftDoubleComplex* device_FFT_dst;
	cufftDoubleComplex *device_dst;
	cufftDoubleComplex *device_FFT_tmp;
	cufftDoubleComplex *device_FFT_tmp2;
	cufftDoubleComplex *device_FFT_tmp3;
	LFGpuConst* device_config;

	Complex<Real>* host_FFT_tmp = new Complex<Real>[pnXY];
	auto step = CUR_TIME;
	LOG("\tMemory Allocation : ");

	HANDLE_ERROR(cudaMalloc(&device_LF, sizeof(uchar1*) * N));
	device_LFData = new uchar*[N];

	for (int i = 0; i < N; i++)
	{
		int size = m_vecImgSize[i];
		HANDLE_ERROR(cudaMalloc(&device_LFData[i], sizeof(uchar1) * size));
		HANDLE_ERROR(cudaMemcpy(device_LFData[i], m_vecImages[i], sizeof(uchar) * size, cudaMemcpyHostToDevice));
	}
	HANDLE_ERROR(cudaMemcpy(device_LF, device_LFData, sizeof(uchar*) * N, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void**)&device_config, sizeof(LFGpuConst)));
	HANDLE_ERROR(cudaMalloc((void**)&device_FFT_src, sizeof(cufftDoubleComplex) * N * R));
	HANDLE_ERROR(cudaMalloc((void**)&device_FFT_dst, sizeof(cufftDoubleComplex) * N * R));

	HANDLE_ERROR(cudaMalloc((void **)&device_dst, sizeof(cufftDoubleComplex) * N * R));
	HANDLE_ERROR(cudaMemset(device_FFT_src, 0, sizeof(cufftDoubleComplex) * N * R));// , streamLF));
	HANDLE_ERROR(cudaMemset(device_FFT_dst, 0, sizeof(cufftDoubleComplex) * N * R));//, streamLF));

	HANDLE_ERROR(cudaMalloc((void**)&device_FFT_tmp, sizeof(cufftDoubleComplex) * pnXY));
	HANDLE_ERROR(cudaMalloc((void**)&device_FFT_tmp2, sizeof(cufftDoubleComplex) * pnXY * 4));
	HANDLE_ERROR(cudaMalloc((void**)&device_FFT_tmp3, sizeof(cufftDoubleComplex) * pnXY * 4));

	LOG("%lf (s)\n", ELAPSED_TIME(step, CUR_TIME));

	int nThreads = pCUDA->getMaxThreads();
	int nBlocks = (R + nThreads - 1) / nThreads;
	int nBlocks2 = (N * R + nThreads - 1) / nThreads;
	int nBlocks3 = (N * R * 4 + nThreads - 1) / nThreads;
	int nBlocks4 = (N + nThreads - 1) / nThreads;

	Real pi2 = M_PI * 2;
	for (int ch = 0; ch < nWave; ch++)
	{
		LOG("\tMemory Initialize : ");
		step = CUR_TIME;
		HANDLE_ERROR(cudaMemset(device_dst, 0, sizeof(cuDoubleComplex) * N * R));//, streamLF));
		HANDLE_ERROR(cudaMemset(device_FFT_tmp, 0, sizeof(cuDoubleComplex) * pnXY));//, streamLF));
		HANDLE_ERROR(cudaMemset(device_FFT_tmp2, 0, sizeof(cuDoubleComplex) * pnXY * 4));//, streamLF));
		HANDLE_ERROR(cudaMemset(device_FFT_tmp3, 0, sizeof(cuDoubleComplex) * pnXY * 4));//, streamLF));
		LOG("%lf (s)\n", ELAPSED_TIME(step, CUR_TIME));

		Real lambda = context_.wave_length[ch];

		LFGpuConst* host_config = new LFGpuConst(
			nWave, nWave - 1 - ch, pnX, pnY, ppX, ppY, nX, nY, rX, rY, distance, pi2 / lambda, lambda, bRandomPhase
		);

		HANDLE_ERROR(cudaMemcpy(device_config, host_config, sizeof(LFGpuConst), cudaMemcpyHostToDevice));

		LOG("\tConvertLF2ComplexField <<<%d, %d>>> : ", nBlocks, nThreads);
		step = CUR_TIME;
		cudaConvertLF2ComplexField_Kernel(0, nBlocks, nThreads, device_config, device_LF, device_FFT_src);
		LOG("%lf (s)\n", ELAPSED_TIME(step, CUR_TIME));

		// 20200824_mwnam_
		cudaError error = cudaGetLastError();
		if (error != cudaSuccess) {
			LOG("cudaGetLastError(): %s\n", cudaGetErrorName(error));
			if (error == cudaErrorLaunchOutOfResources) {
				ch--;
				nThreads /= 2;
				nBlocks = (R + nThreads - 1) / nThreads;
				nBlocks2 = (N * R + nThreads - 1) / nThreads;
				nBlocks3 = (N * R * 4 + nThreads - 1) / nThreads;
				nBlocks4 = (N * 4 + nThreads - 1) / nThreads;
				delete host_config;
				continue;
			}
		}

		LOG("\tCUDA FFT <<<%d, %d>>> : ", nBlocks4, nThreads);
		step = CUR_TIME;

		cufftHandle plan;
		cufftResult result;
		// fft
		result = cufftPlan2d(&plan, nY, nX, CUFFT_Z2Z);
		if (result != CUFFT_SUCCESS)
		{
			LOG("\tcufftPlan2d : Failed (%d\n", result);
			return;
		};

		cufftDoubleComplex* in, *out;
		for (int r = 0; r < R; r++)
		{
			int offset = N * r;
			in = &device_FFT_src[offset];
			out = &device_FFT_dst[offset];
			cudaFFT_LF(&plan, 0, nBlocks4, nThreads, nX, nY, in, out, -1);
		}
		if (cudaDeviceSynchronize() != cudaSuccess)
			LOG("Sync Failed!\n");
		cufftDestroy(plan);
		LOG("%lf (s)\n", ELAPSED_TIME(step, CUR_TIME));

		LOG("\tMultiply Phase <<<%d, %d>>> : ", nBlocks, nThreads);
		step = CUR_TIME;
		procMultiplyPhase(0, nBlocks, nThreads, device_config, device_FFT_dst, device_FFT_tmp);
		LOG("%lf (s)\n", ELAPSED_TIME(step, CUR_TIME));

		LOG("\tCUDA Fresnel Propagation <<<%d, %d>>> : ", nBlocks2, nThreads);
		step = CUR_TIME;
		cudaFresnelPropagationLF(nBlocks2, nBlocks3, nThreads, pnX, pnY, device_FFT_tmp, device_FFT_tmp2, device_FFT_tmp3, device_dst, device_config);

		// 여기 문제
		HANDLE_ERROR(cudaMemcpy(complex_H[ch], device_dst, sizeof(cuDoubleComplex) * pnXY, cudaMemcpyDeviceToHost));
		LOG("%lf (s)\n", ELAPSED_TIME(step, CUR_TIME));

		delete host_config;
	}

	delete[] host_FFT_tmp;
	cudaFree(device_LF);
	for (int i = 0; i < N; i++)
		cudaFree(device_LFData[i]);
	delete[] device_LFData;
	
	cudaFree(device_config);
	cudaFree(device_FFT_src);
	cudaFree(device_FFT_dst);
	cudaFree(device_FFT_tmp);
	cudaFree(device_FFT_tmp2);
	cudaFree(device_FFT_tmp3);
	cudaFree(device_dst);
	LOG("Total: %lf (s)\n", ELAPSED_TIME(begin, CUR_TIME));
}
