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


#ifndef __ophSigKernel_cu
#define __ophSigKernel_cu

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <npp.h>
#include "typedef.h"
#include "ophSig.h"

static const int kBlockThreads = 1024;


__global__ void cudaKernel_FFT(int nx, int ny, cufftDoubleComplex* input, cufftDoubleComplex* output, bool bNormailzed)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	double normalF = nx * ny;

	output[tid].x = input[tid].x / normalF;
	output[tid].y = input[tid].y / normalF;
}

__global__ void cudaKernel_CreateSphtialCarrier() {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
}

//__global__ void cudaKernel_sigCvtOFF(cuDoubleComplex *src_data, Real *dst_data, ophSigConfig *device_config, int nx, int ny, Real wl, cuDoubleComplex *F, Real *angle) {
__global__ void cudaKernel_sigCvtOFF(cuDoubleComplex *src_data, Real *dst_data, ophSigConfig *device_config, int nx, int ny, Real wl, cuDoubleComplex*F, Real *angle) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int r = tid % ny;
	int c = tid / ny;
	Real x, y;
	if (tid < nx*ny)
	{
		x = (device_config->height / (ny - 1)*r - device_config->height / 2);
		y = (device_config->width / (nx - 1)*c - device_config->width / 2);
		F[tid].x = cos(((2 * M_PI) / wl)*(x * sin(angle[_X]) + y * sin(angle[_Y])));
		F[tid].y = sin(((2 * M_PI) / wl)*(x * sin(angle[_X]) + y * sin(angle[_Y])));
		dst_data[tid] = src_data[tid].x * F[tid].x - src_data[tid].y * F[tid].y;
	}
}
//__global__ void cudaKernel_sigCvtHPO(ophSigConfig *device_config, cuDoubleComplex *F, int nx, int ny, Real Rephase, Real Imphase) {
__global__ void cudaKernel_sigCvtHPO(ophSigConfig *device_config, cuDoubleComplex *F, int nx, int ny, Real Rephase, Real Imphase) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int r = tid % ny;
	int c = tid / ny;
	int xshift = nx / 2;
	int yshift = ny / 2;
	Real y;

	if (tid < nx*ny)
	{
		int ii = (r + yshift) % ny;
		int jj = (c + xshift) % nx;
		y = (2 * M_PI * (c) / device_config->width - M_PI * (nx - 1) / device_config->width);
		F[ny*jj + ii].x = exp(Rephase*y * y)*cos(Imphase*y * y);
		F[ny*jj + ii].y = exp(Rephase*y * y)*sin(Imphase*y * y);
	}
}

__global__ void cudaKernel_sigCvtCAC(cuDoubleComplex*FFZP, ophSigConfig *device_config, int nx, int ny, Real sigmaf, Real radius) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int r = tid % ny;
	int c = tid / ny;
	int xshift = nx >> 1;
	int yshift = ny >> 1;
	Real x, y;
	if (tid < nx*ny)
	{
		int ii = (r + yshift) % ny;
		int jj = (c + xshift) % nx;
		y = (2 * M_PI * c) / radius - (M_PI*(nx - 1)) / radius;
		x = (2 * M_PI * r) / radius - (M_PI*(ny - 1)) / radius;

		FFZP[ny*jj + ii].x = cos(sigmaf * (x*x + y * y));
		FFZP[ny*jj + ii].y = -sin(sigmaf * (x*x + y * y));

	}

}

//__global__ void cudaKernel_multiply(cufftDoubleComplex *src_data, cufftDoubleComplex *dst_data, cuDoubleComplex *F, int nx, int ny) {
__global__ void cudaKernel_multiply(cufftDoubleComplex *src_data, cufftDoubleComplex *dst_data, cuDoubleComplex *F, int nx, int ny)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < nx*ny)
	{
		dst_data[tid].x = src_data[tid].x * F[tid].x - src_data[tid].y * F[tid].y;
		dst_data[tid].y = src_data[tid].y * F[tid].x + src_data[tid].x * F[tid].y;
	}
}


//__global__ void cudaKernel_Realmultiply(cufftDoubleComplex *src_data, Real *dst_data, cuDoubleComplex *F, int nx, int ny) {
__global__ void cudaKernel_Realmultiply(cufftDoubleComplex *src_data, Real *dst_data, cuDoubleComplex *F, int nx, int ny)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < nx*ny)
	{
		dst_data[tid] = src_data[tid].x * F[tid].x - src_data[tid].y * F[tid].y;
	}
}


__global__ void cudaKernel_Propagation(cuDoubleComplex*FH, ophSigConfig *device_config, int nx, int ny, Real sigmaf) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int r = tid % ny;
	int c = tid / ny;
	int xshift = nx >> 1;
	int yshift = ny >> 1;
	int ii = (r + yshift) % ny;
	int jj = (c + xshift) % nx;
	Real x, y;

	x = (2 * M_PI * (c)) / device_config->height - (M_PI*(nx - 1)) / (device_config->height);
	y = (2 * M_PI * (r)) / device_config->width - (M_PI*(ny - 1)) / (device_config->width);
	FH[ny*jj + ii].x = cos(sigmaf * (x * x + y * y));
	FH[ny*jj + ii].y = sin(sigmaf * (x * x + y * y));
}

__global__ void cudaKernel_GetParamAT1(cuDoubleComplex*src_data, cuDoubleComplex*Flr, cuDoubleComplex*Fli, cuDoubleComplex*G, ophSigConfig *device_config, int nx, int ny, Real_t NA_g, Real wl)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int r = tid % ny;
	int c = tid / ny;
	Real x, y;
	if (tid < nx*ny)
	{
		x = 2 * M_PI * (c) / device_config->height - M_PI * (nx - 1) / device_config->height;
		y = 2 * M_PI * (r) / device_config->width - M_PI * (ny - 1) / device_config->width;

		G[tid].x = exp(-M_PI * pow((wl) / (2 * M_PI * NA_g), 2) * (x * x + y * y));

		Flr[tid].x = src_data[tid].x;
		Fli[tid].x = src_data[tid].y;
		Flr[tid].y = 0;
		Fli[tid].y = 0;
	}
}

__global__ void cudaKernel_GetParamAT2(cuDoubleComplex*Flr, cuDoubleComplex*Fli, cuDoubleComplex*G, cuDoubleComplex *temp_data, int nx, int ny)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int r = tid % ny;
	int c = tid / ny;
	int xshift = nx >> 1;
	int yshift = ny >> 1;
	int ii = (c + xshift) % nx;
	int jj = (r + yshift) % ny;

	if (tid < nx*ny)
	{
		temp_data[ny*ii + jj].x = 
			(
				((Flr[tid].x * G[tid].x * Flr[tid].x * G[tid].x) - 
					(Fli[tid].x * G[tid].x * Fli[tid].x * G[tid].x)) /
				((Flr[tid].x * G[tid].x * Flr[tid].x * G[tid].x) + 
					(Fli[tid].x * G[tid].x * Fli[tid].x * G[tid].x) + 
					pow(10., -300)
					)
				);
		temp_data[ny*ii + jj].y = 
			(2 * Flr[tid].x * G[tid].x * (Fli[tid].x * G[tid].x)) / 
			((Flr[tid].x * G[tid].x * Flr[tid].x * G[tid].x) + 
				(Fli[tid].x * G[tid].x * Fli[tid].x * G[tid].x)	+ 
				pow(10., -300)
				);
	}
}

__global__ void cudaKernel_sigGetParamSF(cufftDoubleComplex *src_data, Real *f, int nx, int ny, float th)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int r = tid % ny;
	int c = tid / ny;
	Real ret1 = 0, ret2 = 0;
	if (tid < (nx)*(ny))
	{
		f[tid] = 0;
		if (r != ny - 2 && r != ny - 1)
		{
			if (c != nx - 2 && c != nx - 1)
			{
				ret1 = abs(src_data[r + (c + 2)*ny].x - src_data[tid].x);
				ret2 = abs(src_data[(r + 2) + c * ny].x - src_data[tid].x);
			}
		}
		if (ret1 >= th) { f[tid] = ret1 * ret1; }
		else if (ret2 >= th) { f[tid] = ret2 * ret2; }
	}
}

__global__ void cudaKernel_sub(Real *data, int nx, int ny, Real *Min)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	data[tid] = data[tid] - *Min;
}

__global__ void cudaKernel_div(Real *src_data, cuDoubleComplex *dst_data, int nx, int ny, Real *Max)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	dst_data[tid].x = src_data[tid] / *Max;
	dst_data[tid].y = 0;
}
__global__ void fftShift(int N, int nx, int ny, cufftDoubleComplex* input, cufftDoubleComplex* output, bool bNormailzed)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	/*double normalF = 1.0;
	if (bNormailzed == true)
		normalF = nx * ny;


	int r = tid % ny;
	int c = tid / ny;
	int xshift = nx / 2;
	int yshift = ny / 2;
	int ii = (c + xshift) % nx;
	int jj = (r + yshift) % ny;

	output[ny*jj + ii].x = input[tid].x;
	output[ny*jj + ii].y = input[tid].y;*/
	double normalF = 1.0;
	if (bNormailzed == true)
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

extern "C"
{
	void cudaFFT(int nx, int ny, cufftDoubleComplex* in_field, cufftDoubleComplex* output_field, int direction, bool bNormalized)
	{
		unsigned int nblocks = (nx*ny + kBlockThreads - 1) / kBlockThreads;
		int N = nx * ny;
		cufftHandle plan;
		cufftResult result = cufftPlan2d(&plan, ny, nx, CUFFT_Z2Z);

		if (direction == -1)
			result = cufftExecZ2Z(plan, output_field, in_field, CUFFT_FORWARD);
		else
			result = cufftExecZ2Z(plan, output_field, in_field, CUFFT_INVERSE);

		if (result != CUFFT_SUCCESS)
		{
			LOG("------------------FAIL: execute cufft, code=%s", result);
			return;
		}

		if (cudaDeviceSynchronize() != cudaSuccess) {
			LOG("Cuda error: Failed to synchronize\n");
			return;
		}


		cufftDestroy(plan);
	}

	void cudaCuFFT(cufftHandle* plan, cufftDoubleComplex *src_data, cufftDoubleComplex *dst_data, int nx, int ny, int direction) {
		unsigned int nBlocks = (nx*ny + kBlockThreads - 1) / kBlockThreads;


		cufftResult result;
		result = cufftExecZ2Z(*plan, src_data, dst_data, CUFFT_FORWARD);

		if (result != CUFFT_SUCCESS)
			return;

		if (cudaDeviceSynchronize() != cudaSuccess)
			return;

	}


	void cudaCuIFFT(cufftHandle* plan, cufftDoubleComplex *src_data, cufftDoubleComplex *dst_data, int nx, int ny, int direction) {
		unsigned int nBlocks = (nx*ny + kBlockThreads - 1) / kBlockThreads;
		cufftResult result;
		//
		result = cufftExecZ2Z(*plan, src_data, dst_data, CUFFT_INVERSE);
		cudaKernel_FFT << < nBlocks, kBlockThreads >> > (nx, ny, dst_data, src_data, direction);

		if (result != CUFFT_SUCCESS)
			return;

		if (cudaDeviceSynchronize() != cudaSuccess)
			return;

	}

	void cudaCvtOFF(cuDoubleComplex *src_data, Real *dst_data, ophSigConfig *device_config, int nx, int ny, Real wl, cuDoubleComplex*F, Real *angle)
	{
		unsigned int nBlocks = (nx * ny + kBlockThreads - 1) / kBlockThreads;


		cudaKernel_sigCvtOFF << < nBlocks, kBlockThreads >> > (src_data, dst_data, device_config, nx, ny, wl, F, angle);
		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		uchar * pDeviceBuffer;
		Real pM = 0;
		int nBufferSize;
		nppsSumGetBufferSize_64f(nx*ny, &nBufferSize);
		cudaMalloc((void**)(&pDeviceBuffer), nBufferSize);
		nppsMin_64f(dst_data, nx*ny, &pM, pDeviceBuffer);
		cudaKernel_sub << < nBlocks, kBlockThreads >> > (dst_data, nx, ny, &pM);
		nppsMax_64f(dst_data, nx*ny, &pM, pDeviceBuffer);
		cudaKernel_div << < nBlocks, kBlockThreads >> > (dst_data, src_data, nx, ny, &pM);
		cudaFree(pDeviceBuffer);

	}

	void cudaCvtHPO(CUstream_st* stream, cufftDoubleComplex *src_data, cufftDoubleComplex *dst_data, ophSigConfig *device_config, cuDoubleComplex*F, int nx, int ny, Real Rephase, Real Imphase) {
		unsigned int nBlocks = (nx * ny + kBlockThreads - 1) / kBlockThreads;
		cudaKernel_sigCvtHPO << < nBlocks, kBlockThreads, 0, stream >> > (device_config, F, nx, ny, Rephase, Imphase);
		cudaKernel_multiply << < nBlocks, kBlockThreads, 0, stream >> > (src_data, dst_data, F, nx, ny);

	}
	void cudaCvtCAC(cufftDoubleComplex *src_data, cufftDoubleComplex *dst_data, cuDoubleComplex *FFZP, ophSigConfig *device_config, int nx, int ny, Real sigmaf, Real radius) {
		unsigned int nBlocks = (nx * ny + kBlockThreads - 1) / kBlockThreads;
		cudaKernel_sigCvtCAC << < nBlocks, kBlockThreads >> > (FFZP, device_config, nx, ny, sigmaf, radius);
		cudaKernel_multiply << < nBlocks, kBlockThreads >> > (src_data, dst_data, FFZP, nx, ny);
	}

	void cudaPropagation(cufftDoubleComplex *src_data, cufftDoubleComplex *dst_data, cuDoubleComplex *FH, ophSigConfig *device_config, int nx, int ny, Real sigmaf) {
		unsigned int nBlocks = (nx * ny + kBlockThreads - 1) / kBlockThreads;
		cudaKernel_Propagation << < nBlocks, kBlockThreads >> > (FH, device_config, nx, ny, sigmaf);
		cudaKernel_multiply << < nBlocks, kBlockThreads >> > (src_data, dst_data, FH, nx, ny);
		//__syncthreads();
	}
	void cudaGetParamAT1(cuDoubleComplex *src_data, cuDoubleComplex *Flr, cuDoubleComplex *Fli, cuDoubleComplex *G, ophSigConfig *device_config, int nx, int ny, Real_t NA_g, Real wl) {
		unsigned int nBlocks = (nx * ny + kBlockThreads - 1) / kBlockThreads;

		cudaKernel_GetParamAT1 << < nBlocks, kBlockThreads >> > (src_data, Flr, Fli, G, device_config, nx, ny, NA_g, wl);

		//__syncthreads();
	}

	void cudaGetParamAT2(cuDoubleComplex *Flr, cuDoubleComplex *Fli, cuDoubleComplex *G, cuDoubleComplex *temp_data, int nx, int ny) {
		unsigned int nBlocks = (nx * ny + kBlockThreads - 1) / kBlockThreads;

		cudaKernel_GetParamAT2 << < nBlocks, kBlockThreads >> > (Flr, Fli, G, temp_data, nx, ny);

		//__syncthreads();
	}


	double cudaGetParamSF(cufftHandle *fftplan, cufftDoubleComplex *src_data, cufftDoubleComplex *temp_data, cufftDoubleComplex *dst_data, Real *f, cuDoubleComplex *FH, ophSigConfig *device_config, int nx, int ny, float zMax, float zMin, int sampN, float th, Real wl) {
		unsigned int nBlocks = (nx * ny + kBlockThreads - 1) / kBlockThreads;

		int nBufferSize;
		Real max = MIN_DOUBLE;
		Real depth = 0;
		Real nSumHost = 0;
		uchar * pDeviceBuffer;
		Real* pSum;

		cudaMalloc((void **)(&pSum), sizeof(Real));
		nppsSumGetBufferSize_64f(nx*ny, &nBufferSize);
		cudaMalloc((void**)(&pDeviceBuffer), nBufferSize);
		Real dz = (zMax - zMin) / sampN;
		for (int n = 0; n < sampN + 1; n++)
		{

			Real z = ((n)* dz + zMin);
			Real_t sigmaf = (z*wl) / (4 * M_PI);
			//propagation
			cudaPropagation(src_data, dst_data, FH, device_config, nx, ny, sigmaf);
			cudaCuIFFT(fftplan, dst_data, temp_data, nx, ny, CUFFT_INVERSE);
			//SF
			cudaKernel_sigGetParamSF << < nBlocks, kBlockThreads >> > (dst_data, f, nx, ny, th);
			////	//////sum matrix
			////
			nppsSum_64f(f, nx*ny, pSum, pDeviceBuffer);
			cudaMemcpy(&nSumHost, pSum, sizeof(Real), cudaMemcpyDeviceToHost);
			cout << (float)n / sampN * 100 << " %" << endl;
			////	//////find max
			if (nSumHost > max) {
				max = nSumHost;
				depth = z;
			}
		}
		cudaFree(pDeviceBuffer);
		cudaFree(pSum);
		return depth;
		//__syncthreads();
	}


};


#endif
