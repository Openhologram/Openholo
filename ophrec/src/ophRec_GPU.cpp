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
#include "ophRec.h"
#include "ophRec_GPU.h"
#include "sys.h"
//#include "CUDA.h"
#include <npp.h>

void ophRec::ASM_Propagation_GPU()
{
	LOG("%s\n", __FUNCTION__);
	LOG("\tMemory Allocation : ");
	auto begin = CUR_TIME;
	auto step = CUR_TIME;

	//CUDA *cuda = CUDA::getInstance();

	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const int N = pnX * pnY;
	const int nWave = context_.waveNum;

	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];

	const Real simFrom = rec_config.SimulationFrom;
	const Real simTo = rec_config.SimulationTo;
	const int simStep = rec_config.SimulationStep;
	const Real simGap = (simStep > 1) ? (simTo - simFrom) / (simStep - 1) : 0;

	const Real tx = 1 / ppX;
	const Real ty = 1 / ppY;
	const Real dx = tx / pnX;
	const Real dy = ty / pnY;

	const Real htx = tx / 2;
	const Real hty = ty / 2;
	const Real hdx = dx / 2;
	const Real hdy = dy / 2;
	const Real baseX = -htx + hdx;
	const Real baseY = -hty + hdy;

	const uint nChannel = context_.waveNum;

	RecGpuConst* device_config = nullptr;
	HANDLE_ERROR(cudaMalloc((void**)&device_config, sizeof(RecGpuConst)));

	cuDoubleComplex *device_src;
	cuDoubleComplex *device_dst;
	Real *device_encode;
	HANDLE_ERROR(cudaMalloc((void**)&device_src, N * sizeof(cuDoubleComplex)));
	HANDLE_ERROR(cudaMalloc((void**)&device_dst, N * sizeof(cuDoubleComplex)));
	HANDLE_ERROR(cudaMalloc((void**)&device_encode, N * sizeof(Real)));

	bool bRandomPhase = true;
	int nThread = 512;// cuda->getMaxThreads();
	int nBlock = (N + nThread - 1) / nThread;

	Real* max_device;
	Real* min_device;
	HANDLE_ERROR(cudaMalloc(&max_device, sizeof(Real)));
	HANDLE_ERROR(cudaMalloc(&min_device, sizeof(Real)));

	byte* nppMaxBuffer;
	int nBuffer;
	nppsSumGetBufferSize_64f(N, &nBuffer);
	HANDLE_ERROR(cudaMalloc(&nppMaxBuffer, nBuffer));

	LOG("%lf (s)\n", ELAPSED_TIME(step, CUR_TIME));

	for (int step = 0; step < simStep; step++)
	{
		Real min = MAX_DOUBLE;
		Real max = MIN_DOUBLE;
		for (uint ch = 0; ch < nChannel; ch++)
		{
			const Real lambda = context_.wave_length[ch];
			const Real k = 2 * M_PI / lambda;

			RecGpuConst* host_config = new RecGpuConst(
				nChannel, 1, pnX, pnY, ppX, ppY,
				simFrom + (step * simGap), k, lambda, bRandomPhase
			);

			HANDLE_ERROR(cudaMemcpy(device_config, host_config, sizeof(RecGpuConst), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(device_src, complex_H[ch], sizeof(Complex<Real>) * N, cudaMemcpyHostToDevice));

			cudaASMPropagation(nBlock, nThread, pnX, pnY, device_src, device_dst, device_encode, device_config);

			Real *encode = new Real[N];
			uchar *normal = new uchar[N];
			HANDLE_ERROR(cudaMemcpy(encode, device_encode, sizeof(Real) * N, cudaMemcpyDeviceToHost));


			Real locmin, locmax;
			nppsMax_64f(device_encode, N, max_device, nppMaxBuffer);
			nppsMin_64f(device_encode, N, min_device, nppMaxBuffer);

			cudaMemcpy(&locmax, max_device, sizeof(Real), cudaMemcpyDeviceToHost);
			cudaMemcpy(&locmin, min_device, sizeof(Real), cudaMemcpyDeviceToHost);

			if (min > locmin) min = locmin;
			if (max < locmax) max = locmax;

			m_vecEncoded.push_back(encode);
			m_vecNormalized.push_back(normal);

			delete host_config;
		}

		LOG("step: %d => max: %e / min: %e\n", step, max, min);
		if (nWave == 3)
		{
			for (int ch = 0; ch < nWave; ch++)
			{
				int idx = step * nWave + ch;
				normalize(m_vecEncoded[idx], m_vecNormalized[idx], pnX, pnY, max, min);
			}
		}
		else
			normalize(m_vecEncoded[step], m_vecNormalized[step], pnX, pnY);
	}

	HANDLE_ERROR(cudaFree(nppMaxBuffer));
	HANDLE_ERROR(cudaFree(max_device));
	HANDLE_ERROR(cudaFree(min_device));
	HANDLE_ERROR(cudaFree(device_src));
	HANDLE_ERROR(cudaFree(device_dst));
	HANDLE_ERROR(cudaFree(device_encode));
	HANDLE_ERROR(cudaFree(device_config));
	LOG("Total : %lf (s)\n", ELAPSED_TIME(begin, CUR_TIME));
}