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

//#include <cuda_runtime.h>
//#include <cuComplex.h>
//#include <cuda.h>
//#include <device_launch_parameters.h>
//#include <device_functions.h>
//#include <cufft.h>
#include <curand_kernel.h>

__global__ void cudaKernel_convertLF2ComplexField(int nx, int ny, int rx, int ry, uchar1** LF, cufftDoubleComplex* output)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int c = tid % rx;
	int r = tid / rx;

	if (tid < rx*ry) {

		for (int k = 0; k < nx*ny; k++)
		{
			double val = (double)LF[k][r*rx + c].x;
			output[(r*rx + c)*nx*ny + k] = make_cuDoubleComplex(val, 0);
		}
	}
}

__global__ void cudaKernel_MultiplyPhase(int nx, int ny, int rx, int ry, cufftDoubleComplex* in, cufftDoubleComplex* output, double PI)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int c = tid % rx;
	int r = tid / rx;

	if (tid < rx*ry) {

		curandState state;
		curand_init(c*r, 0, 0, &state);

		double randomData;

		for (int k = 0; k < nx*ny; k++)
		{
			randomData = curand_uniform_double(&state);
			cufftDoubleComplex phase = make_cuDoubleComplex(0, randomData * PI * 2);
			exponent_complex(&phase);

			cufftDoubleComplex val = in[(r*rx + c)*nx*ny + k];
			int cc = k % nx;
			int rr = k / nx;
			output[c*nx + r * rx*nx*ny + cc + rr * nx*rx] = cuCmul(val, phase);
		}
	}

}

__global__ void cudaKernel_MoveToin2x(int Nx, int Ny, cufftDoubleComplex* in, cufftDoubleComplex* out)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < Nx*Ny) {

		int c = tid % Nx;
		int r = tid / Nx;
		int offsetX = Nx / 2;
		int offsetY = Ny / 2;

		out[(c + offsetX) + (r + offsetY) * Nx * 2] = in[c + r * Nx];

	}

}

__global__ void cudaKernel_ProcMultiplyProp(int Nx, int Ny, cufftDoubleComplex* inout, double PI, double dist, double wavelength, double ppx, double ppy)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < Nx*Ny) {

		int c = tid % Nx;
		int r = tid / Nx;

		double fx = (-Nx / 2.0 + (double)c) / (Nx*ppx);
		double fy = (-Ny / 2.0 + (double)r) / (Ny*ppy);

		cufftDoubleComplex prop = make_cuDoubleComplex(0, dist * PI * 2);
		double sqrtPart = sqrt(1.0 / (wavelength*wavelength) - fx * fx - fy * fy);
		prop.y *= sqrtPart;
		exponent_complex(&prop);

		inout[tid] = cuCmul(inout[tid], prop);

	}
}

__global__ void cudaKernel_CopyToOut(int Nx, int Ny, cufftDoubleComplex* in, cufftDoubleComplex* out)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < Nx*Ny) {

		int c = tid % Nx;
		int r = tid / Nx;
		int offsetX = Nx / 2;
		int offsetY = Ny / 2;

		out[c + r * Nx] = in[(c + offsetX) + (r + offsetY) * Nx * 2];

	}
}

extern "C"
{
	void cudaConvertLF2ComplexField_Kernel(CUstream_st* stream, int nx, int ny,
		int rx, int ry, uchar1** LF, cufftDoubleComplex* output)
	{
		dim3 grid((rx*ry + kBlockThreads - 1) / kBlockThreads, 1, 1);
		cudaKernel_convertLF2ComplexField << <grid, kBlockThreads, 0, stream >> > (nx, ny, rx, ry, LF, output);
	}

	void cudaFFT_LF(cufftHandle* plan, CUstream_st* stream, int nx, int ny, cufftDoubleComplex* in_field, cufftDoubleComplex* output_field, int direction)
	{
		unsigned int nblocks = (nx*ny + kBlockThreads - 1) / kBlockThreads;
		int N = nx * ny;
		//fftShift_LF<< <nblocks, kBlockThreads, 0, stream >> > (N, nx, ny, in_field, output_field, false);
		fftShift << <nblocks, kBlockThreads, 0, stream >> > (N, nx, ny, in_field, output_field, false);

		cufftResult result;
		if (direction == -1)
			result = cufftExecZ2Z(*plan, output_field, in_field, CUFFT_FORWARD);
		else
			result = cufftExecZ2Z(*plan, output_field, in_field, CUFFT_INVERSE);

		if (result != CUFFT_SUCCESS)
			return;

		if (cudaDeviceSynchronize() != cudaSuccess)
			return;

		//fftShift_LF << < nblocks, kBlockThreads, 0, stream >> > (N, nx, ny, in_field, output_field, false);
		fftShift << < nblocks, kBlockThreads, 0, stream >> > (N, nx, ny, in_field, output_field, false);

	}

	void procMultiplyPhase(CUstream_st* stream, int nx, int ny, int rx, int ry, cufftDoubleComplex* input, cufftDoubleComplex* output, double PI)
	{
		dim3 grid((rx*ry + kBlockThreads - 1) / kBlockThreads, 1, 1);
		cudaKernel_MultiplyPhase << <grid, kBlockThreads, 0, stream >> > (nx, ny, rx, ry, input, output, PI);
	}

	void procMoveToin2x(CUstream_st* stream, int Nx, int Ny, cufftDoubleComplex* in, cufftDoubleComplex* out)
	{
		dim3 grid((Nx*Ny + kBlockThreads - 1) / kBlockThreads, 1, 1);
		cudaKernel_MoveToin2x << <grid, kBlockThreads, 0, stream >> > (Nx, Ny, in, out);

	}

	void procMultiplyProp(CUstream_st* stream, int Nx, int Ny, cufftDoubleComplex* inout, double PI, double dist, double wavelength, double ppx, double ppy)
	{
		dim3 grid((Nx*Ny + kBlockThreads - 1) / kBlockThreads, 1, 1);
		cudaKernel_ProcMultiplyProp << <grid, kBlockThreads, 0, stream >> > (Nx, Ny, inout, PI, dist, wavelength, ppx, ppy);

	}

	void procCopyToOut(CUstream_st* stream, int Nx, int Ny, cufftDoubleComplex* in, cufftDoubleComplex* out)
	{
		dim3 grid((Nx*Ny + kBlockThreads - 1) / kBlockThreads, 1, 1);
		cudaKernel_CopyToOut << <grid, kBlockThreads, 0, stream >> > (Nx, Ny, in, out);

	}
}

#endif // !ophLFKernel_cu__