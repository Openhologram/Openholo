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

#ifndef ophLFKernel_cu__
#define ophLFKernel_cu__

#include "ophKernel.cuh"
#include "ophLightField_GPU.h"
#include <curand_kernel.h>

__global__ void cudaKernel_CalcData(cufftDoubleComplex *src, const LFGpuConst* config)
{
	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ double s_ppX;
	__shared__ double s_ppY;
	__shared__ int s_pnX;
	__shared__ int s_pnY;
	__shared__ int s_pnXY;
	__shared__ double s_ssX;
	__shared__ double s_ssY;
	__shared__ double s_z;
	__shared__ double s_v;
	__shared__ double s_lambda;
	__shared__ double s_distance;
	__shared__ double s_pi2;

	if (threadIdx.x == 0)
	{
		s_ppX = config->ppX;
		s_ppY = config->ppY;
		s_pnX = config->pnX;
		s_pnY = config->pnY;
		s_pnXY = s_pnX * s_pnY;
		s_ssX = s_pnX * s_ppX * 2;
		s_ssY = s_pnY * s_ppY * 2;
		s_lambda = config->lambda;
		s_distance = config->distance;
		s_pi2 = config->pi2;
		s_z = s_distance * s_pi2;
		s_v = 1 / (s_lambda * s_lambda);
	}
	__syncthreads();

	if (tid < s_pnXY * 4)
	{
		int pnX2 = s_pnX * 2;

		int w = tid % pnX2;
		int h = tid / pnX2;

		double fy = (-s_pnY + h) / s_ssY;
		double fyy = fy * fy;
		double fx = (-s_pnX + w) / s_ssX;
		double fxx = fx * fx;
		double sqrtpart = sqrt(s_v - fxx - fyy);

		cuDoubleComplex prop;
		prop.x = 0;
		prop.y = s_z * sqrtpart;

		exponent_complex(&prop);

		cuDoubleComplex val;
		val.x = src[tid].x;
		val.y = src[tid].y;

		cuDoubleComplex val2 = cuCmul(val, prop);

		src[tid].x = val2.x;
		src[tid].y = val2.y;
	}
}

__global__ void cudaKernel_MoveDataPost(cufftDoubleComplex *src, cuDoubleComplex *dst, const LFGpuConst* config)
{
	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int pnX;
	__shared__ int pnY;
	__shared__ ulonglong pnXY;

	if (threadIdx.x == 0)
	{
		pnX = config->pnX;
		pnY = config->pnY;
		pnXY = pnX * pnY;
	}
	__syncthreads();

	if (tid < pnXY)
	{
		int w = tid % pnX;
		int h = tid / pnX;
		ulonglong iSrc = pnX * 2 * (pnY / 2 + h) + pnX / 2;

		dst[tid] = src[iSrc + w];
	}
}

__global__ void cudaKernel_MoveDataPre(cuDoubleComplex *src, cufftDoubleComplex *dst, const LFGpuConst* config)
{
	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int pnX;
	__shared__ int pnY;
	__shared__ ulonglong pnXY;

	if (threadIdx.x == 0)
	{
		pnX = config->pnX;
		pnY = config->pnY;
		pnXY = pnX * pnY;
	}
	__syncthreads();

	if (tid < pnXY)
	{
		int w = tid % pnX;
		int h = tid / pnX;
		ulonglong iDst = pnX * 2 * (pnY / 2 + h) + pnX / 2;
		dst[iDst + w] = src[tid];
	}
}

#if false // use constant
__global__ void cudaKernel_convertLF2ComplexField(/*const LFGpuConst *config, */uchar1** LF, cufftDoubleComplex* output)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < img_resolution[0])
	{
		int c = tid % img_resolution[1];
		int r = tid / img_resolution[1];
		int iWidth = c * channel_info[0] + channel_info[1];
		int cWidth = (img_resolution[1] * channel_info[0] + 3) & ~3;

		int src = r * cWidth + iWidth;
		int dst = (r * img_resolution[1] + c) * img_number[0];
		for (int k = 0; k < img_number[0]; k++)
		{
			output[dst + k] = make_cuDoubleComplex((double)LF[k][src].x, 0);
		}
	}
#endif

__global__ void cudaKernel_convertLF2ComplexField(const LFGpuConst *config, uchar1** LF, cufftDoubleComplex* output)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int rX = config->rX;
	int rY = config->rY;
	int nX = config->nX;
	int nY = config->nY;
	int N = nX * nY;
	int R = rX * rY;
	int nChannel = config->nChannel;
	int iAmplitude = config->iAmp;

	if (tid < R)
	{
		int c = tid % rX;
		int r = tid / rX;
		int iWidth = c * nChannel + iAmplitude;
		int cWidth = (rX * nChannel + 3) & ~3;

		int src = r * cWidth + iWidth;
		int dst = (r * rX + c) * N;
		for (int k = 0; k < N; k++)
		{
			output[dst + k] = make_cuDoubleComplex((double)LF[k][src].x, 0);
		}
	}
}

__global__ void cudaKernel_MultiplyPhase(const LFGpuConst *config, cufftDoubleComplex* in, cufftDoubleComplex* output)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ double s_pi2;
	__shared__ int s_R;
	__shared__ int s_N;
	__shared__ int s_rX;
	__shared__ int s_rY;
	__shared__ int s_nX;
	__shared__ int s_nY;
	__shared__ bool s_bRandomPhase;
	__shared__ int s_iAmp;

	if (threadIdx.x == 0)
	{
		s_pi2 = config->pi2;
		s_rX = config->rX;
		s_rY = config->rY;
		s_nX = config->nX;
		s_nY = config->nY;
		s_N = s_nX * s_nY;
		s_R = s_rX * s_rY;
		s_bRandomPhase = config->randomPhase;
		s_iAmp = config->iAmp;
	}


	__syncthreads();
	
	if (tid < s_R) {

		int c = tid % s_rX;
		int r = tid / s_rX;
		curandState state;
		if (s_bRandomPhase)
		{
			curand_init(s_N * s_R * (s_iAmp + 1), 0, 0, &state);
		}

		int src = (r * s_rX + c) * s_N;
		int dst = c * s_nX + r * s_rX * s_N;

		for (int n = 0; n < s_N; n++)
		{
			double randomData = s_bRandomPhase ? curand_uniform_double(&state) : 1.0;

			if (n == 15 && tid >= 0 && tid <= 10)
			{
				printf("bid(%d)tid(%d) : %lf\n", blockIdx.x, threadIdx.x, randomData);
			}

			cufftDoubleComplex phase = make_cuDoubleComplex(0, randomData * s_pi2);
			exponent_complex(&phase);

			cufftDoubleComplex val = in[src + n];
			int cc = n % s_nX; // 0 ~ 9
			int rr = n / s_nX; // 0 ~ 9
			output[dst + cc + rr * s_nX * s_rX] = cuCmul(val, phase);
		}
	}
}


extern "C"
{
	void cudaConvertLF2ComplexField_Kernel(CUstream_st* stream, const int &nBlocks, const int &nThreads, const LFGpuConst *config, uchar1** LF, cufftDoubleComplex* output)
	{
		//cudaKernel_convertLF2ComplexField << <nBlocks, nThreads, 0, stream >> > (config, LF, output);
		cudaKernel_convertLF2ComplexField << < nBlocks, nThreads >> > (config, LF, output);

		if (cudaDeviceSynchronize() != cudaSuccess)
			return;
	}

	void cudaFFT_LF(cufftHandle *plan, CUstream_st* stream, const int &nBlocks, const int &nThreads, const int &nx, const int &ny, cufftDoubleComplex* in_field, cufftDoubleComplex* output_field, const int &direction)
	{
		//cudaFFT(nullptr, nx, ny, in_field, output_field, CUFFT_FORWARD, false);
		int N = nx * ny;

		//fftShift << <nBlocks, nThreads, 0, stream >> > (N, nx, ny, in_field, output_field, false);
		fftShift << < nBlocks, nThreads >> > (N, nx, ny, in_field, output_field, false);

		cufftResult result;
		if (direction == -1)
			result = cufftExecZ2Z(*plan, output_field, in_field, CUFFT_FORWARD);
		else
			result = cufftExecZ2Z(*plan, output_field, in_field, CUFFT_INVERSE);

		if (result != CUFFT_SUCCESS)
			return;

		if (cudaDeviceSynchronize() != cudaSuccess)
			return;

		//fftShift << < nBlocks, nThreads, 0, stream >> > (N, nx, ny, in_field, output_field, false);
		fftShift << < nBlocks, nThreads >> > (N, nx, ny, in_field, output_field, false);
	}

	void procMultiplyPhase(CUstream_st* stream, const int &nBlocks, const int &nThreads, const LFGpuConst *config, cufftDoubleComplex* in, cufftDoubleComplex* out)
	{
		//cudaKernel_MultiplyPhase << <nBlocks, nThreads, 0, stream >> > (config, in, output);
		cudaKernel_MultiplyPhase << <nBlocks, nThreads >> > (config, in, out);

		if (cudaDeviceSynchronize() != cudaSuccess)
			return;
	}

	void cudaFresnelPropagationLF(
		const int &nBlocks, const int&nBlocks2, const int &nThreads, const int &nx, const int &ny,
		cufftDoubleComplex *src, cufftDoubleComplex *tmp, cufftDoubleComplex *tmp2, cufftDoubleComplex *dst, 
		const LFGpuConst* cuda_config)
	{
		cudaError_t error;
		cudaKernel_MoveDataPre << <nBlocks, nThreads >> > (src, tmp, cuda_config);
		error = cudaDeviceSynchronize();
		if (error != cudaSuccess)
		{
			LOG("cudaDeviceSynchronize(%d) : Failed\n", __LINE__);
		}
		cudaFFT(nullptr, nx * 2, ny * 2, tmp, tmp2, CUFFT_FORWARD, false);

		cudaKernel_CalcData << <nBlocks2, nThreads >> > (tmp2, cuda_config);
		error = cudaDeviceSynchronize();
		if (error != cudaSuccess)
		{
			LOG("cudaDeviceSynchronize(%d) : Failed\n", __LINE__);
		}
		cudaFFT(nullptr, nx * 2, ny * 2, tmp2, tmp, CUFFT_INVERSE, true);

		cudaKernel_MoveDataPost << <nBlocks, nThreads >> > (tmp, dst, cuda_config);
		error = cudaDeviceSynchronize();
		if (error != cudaSuccess)
		{
			LOG("cudaDeviceSynchronize(%d) : Failed\n", __LINE__);
		}
	}
}

#endif // !ophLFKernel_cu__