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

#ifndef ophRecKernel_cu__
#define ophRecKernel_cu__

#include "ophKernel.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand_uniform.h>
#include <device_launch_parameters.h>
#include "ophRec_GPU.h"

__global__ void cudaKernel_Encode(cuDoubleComplex *src, double *dst, const RecGpuConst* config)
{
	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int pnX;
	__shared__ int pnY;
	__shared__ int pnXY;

	if (threadIdx.x == 0)
	{
		pnX = config->pnX;
		pnY = config->pnY;
		pnXY = pnX * pnY;
	}
	__syncthreads();

	if (tid < pnXY)
	{
		double re = src[tid].x;
		double im = src[tid].y;
		dst[tid] = sqrt(re * re + im * im);
	}
}

__global__ void cudaKernel_GetKernel(cuDoubleComplex *dst, const RecGpuConst* config)
{
	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int pnX;
	__shared__ int pnY;
	__shared__ int pnXY;
	__shared__ double lambda;
	__shared__ double dx;
	__shared__ double dy;

	__shared__ double baseX;
	__shared__ double baseY;

	__shared__ double k;
	__shared__ double z;

	if (threadIdx.x == 0)
	{
		pnX = config->pnX;
		pnY = config->pnY;
		pnXY = pnX * pnY;
		lambda = config->lambda;
		dx = config->dx;
		dy = config->dy;
		baseX = config->baseX;
		baseY = config->baseY;
		k = config->k;
		z = config->distance;
	}
	__syncthreads();

	if (tid < pnXY)
	{
		int x = tid % pnX;
		int y = tid / pnX;

		double curX = baseX + (x * dx);
		double curY = baseY + (y * dy);
		double xx = curX * lambda;
		double yy = curY * lambda;

		cuDoubleComplex kernel;
		kernel.x = 0;
		kernel.y = sqrt(1 - xx * xx - yy * yy) * k * z;
		exponent_complex(&kernel);

		double re = dst[tid].x;
		double im = dst[tid].y;

		dst[tid].x = re * kernel.x - im * kernel.y;
		dst[tid].y = re * kernel.y + im * kernel.x;
	}
}


extern "C"
{
	void cudaASMPropagation(
		const int &nBlocks, const int &nThreads, const int &nx, const int &ny,
		cuDoubleComplex *src, cuDoubleComplex *dst, Real *encode, const RecGpuConst* cuda_config)
	{
		cudaFFT(nullptr, nx, ny, src, dst, CUFFT_FORWARD, false);

		cudaKernel_GetKernel << <nBlocks, nThreads >> > (dst, cuda_config);

		if (cudaDeviceSynchronize() != cudaSuccess) {
			LOG("Cuda error: Failed to synchronize\n");
			return;
		}

		cudaFFT(nullptr, nx, ny, dst, src, CUFFT_INVERSE, true);

		cudaKernel_Encode << <nBlocks, nThreads >> > (src, encode, cuda_config);


	}
}

#endif // !OphRecKernel_cu__