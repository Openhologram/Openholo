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

#ifndef __ophKernel_cuh__
#define __ophKernel_cuh__

#include <cuComplex.h>
#include <cufft.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime.h>

static const int kBlockThreads = 512;


__global__ void fftShift(int N, int nx, int ny, cufftDoubleComplex* input, cufftDoubleComplex* output, bool bNormalized)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	double normalF = 1.0;
	if (bNormalized == true)
		normalF = nx * ny;

	while (tid < N)
	{
		int i = tid % nx;
		int j = tid / nx;

		int ti = i - nx / 2; if (ti < 0) ti += nx;
		int tj = j - ny / 2; if (tj < 0) tj += ny;

		int oindex = tj * nx + ti;


		output[tid].x = input[oindex].x / normalF;
		output[tid].y = input[oindex].y / normalF;

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void fftShiftf(int N, int nx, int ny, cuFloatComplex* input, cuFloatComplex* output, bool bNormalized)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	float normalF = 1.0;
	if (bNormalized == true)
		normalF = nx * ny;

	while (tid < N)
	{
		int i = tid % nx;
		int j = tid / nx;

		int ti = i - nx / 2; if (ti < 0) ti += nx;
		int tj = j - ny / 2; if (tj < 0) tj += ny;

		int oindex = tj * nx + ti;


		output[tid].x = input[oindex].x / normalF;
		output[tid].y = input[oindex].y / normalF;

		tid += blockDim.x * gridDim.x;
	}
}

__device__  void exponent_complex(cuDoubleComplex* val)
{
	double exp_val = exp(val->x);
	double re = val->x;
	double im = val->y;

	val->x = exp_val * cos(im);
	val->y = exp_val * sin(im);
}

extern "C"

void cudaFFT(CUstream_st* stream, int nx, int ny, cufftDoubleComplex* in_field, cufftDoubleComplex* output_field, int direction, bool bNormalized)
{
	unsigned int nblocks = (nx*ny + kBlockThreads - 1) / kBlockThreads;
	int N = nx * ny;
	fftShift << <nblocks, kBlockThreads, 0, stream >> > (N, nx, ny, in_field, output_field, false);

	cufftHandle plan;
	cufftResult result;
	// fft
	result = cufftPlan2d(&plan, ny, nx, CUFFT_Z2Z);
	if (result != CUFFT_SUCCESS)
	{
		LOG("cufftPlan2d : Failed (%d)\n", result);
		return;
	};

	if (direction == -1)
		result = cufftExecZ2Z(plan, output_field, in_field, CUFFT_FORWARD);
	else
		result = cufftExecZ2Z(plan, output_field, in_field, CUFFT_INVERSE);

	if (result != CUFFT_SUCCESS)
	{
		LOG("cufftExecZ2Z : Failed (%d)\n", result);
		return;
	}

	if (cudaDeviceSynchronize() != cudaSuccess) {
		LOG("cudaDeviceSynchronize() : Failed\n");
		return;
	}

	fftShift << < nblocks, kBlockThreads, 0, stream >> > (N, nx, ny, in_field, output_field, bNormalized);

	cufftDestroy(plan);
}

#endif // !__ophKernel_cuh__