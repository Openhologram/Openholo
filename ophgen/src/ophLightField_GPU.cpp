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

#include "sys.h"

extern "C"
void cudaFFT(CUstream_st* stream, int nx, int ny, cufftDoubleComplex* in_filed, cufftDoubleComplex* output_field, int direction, bool bNormalized);

void ophLF::prepareInputdataGPU()
{
	int nx = num_image[_X];
	int ny = num_image[_Y];
	int rx = resolution_image[_X];
	int ry = resolution_image[_Y];

	if (!streamLF)
		cudaStreamCreate(&streamLF);

	if (LF_gpu) cudaFree(LF_gpu);
	if (LFData_gpu) {
		for (int i = 0; i < nx*ny; i++)
			cudaFree(LFData_gpu[i]);
		free(LFData_gpu);
	}

	HANDLE_ERROR(cudaMalloc(&LF_gpu, sizeof(uchar1*)*nx*ny));
	LFData_gpu = (uchar**)malloc(sizeof(uchar*)*nx*ny);

	for (int i = 0; i < nx*ny; i++) {
		HANDLE_ERROR(cudaMalloc(&LFData_gpu[i], sizeof(uchar1)*rx*ry));
		HANDLE_ERROR(cudaMemset(LFData_gpu[i], 0, sizeof(uchar1) * rx * ry));
		HANDLE_ERROR(cudaMemcpyAsync(LFData_gpu[i], LF[i], sizeof(uchar)*rx*ry, cudaMemcpyHostToDevice), streamLF);
	}

	HANDLE_ERROR(cudaMemcpy(LF_gpu, LFData_gpu, sizeof(uchar*)*nx*ny, cudaMemcpyHostToDevice));

	if (RSplane_complex_field_gpu)
		cudaFree(RSplane_complex_field_gpu);
	HANDLE_ERROR(cudaMalloc((void**)&RSplane_complex_field_gpu, sizeof(cufftDoubleComplex)*rx*ry*nx*ny));
}

void ophLF::convertLF2ComplexField_GPU()
{
	auto start = CUR_TIME;
	int nx = num_image[_X];
	int ny = num_image[_Y];
	int rx = resolution_image[_X];
	int ry = resolution_image[_Y];

	cufftDoubleComplex *complexLF_gpu;
	cufftDoubleComplex *FFTLF_temp_gpu;

	HANDLE_ERROR(cudaMalloc((void**)&complexLF_gpu, sizeof(cufftDoubleComplex)*rx*ry*nx*ny));
	HANDLE_ERROR(cudaMalloc((void**)&FFTLF_temp_gpu, sizeof(cufftDoubleComplex)*rx*ry*nx*ny));
	HANDLE_ERROR(cudaMemsetAsync(complexLF_gpu, 0, sizeof(cufftDoubleComplex)*rx*ry*nx*ny, streamLF));
	HANDLE_ERROR(cudaMemsetAsync(FFTLF_temp_gpu, 0, sizeof(cufftDoubleComplex)*rx*ry*nx*ny, streamLF));
	HANDLE_ERROR(cudaMemsetAsync(RSplane_complex_field_gpu, 0, sizeof(cufftDoubleComplex)*rx*ry*nx*ny, streamLF));

	cudaConvertLF2ComplexField_Kernel(streamLF, nx, ny, rx, ry, LF_gpu, complexLF_gpu);
	LOG("\tcudaConvertLF2ComplexField_Kernel() ... %.5lfsec\n", ((std::chrono::duration<Real>)(CUR_TIME - start)).count());
	cufftHandle fftplan;
	if (cufftPlan2d(&fftplan, ny, nx, CUFFT_Z2Z) != CUFFT_SUCCESS)
	{
		LOG("FAIL in creating cufft plan");
		return;
	};
	LOG("\tcufftPlan2d() ... %.5lfsec\n", ((std::chrono::duration<Real>)(CUR_TIME - start)).count());

	cufftDoubleComplex* in, *out;
#if 0
	for (int k = 0; k < nx*ny; k++)
	{
		int offset = rx * ry * k;
		in = complexLF_gpu + offset;
		out = FFTLF_temp_gpu + offset;
		cudaFFT_LF(&fftplan, streamLF, rx, ry, in, out, -1);
	}
#else
	for (int k = 0; k < rx*ry; k++)
	{
		int offset = nx * ny*k;
		in = complexLF_gpu + offset;
		out = FFTLF_temp_gpu + offset;
		cudaFFT_LF(&fftplan, streamLF, nx, ny, in, out, -1);
	}
#endif
	LOG("\tcudaFFT_LF() ... %.5lfsec\n", ((std::chrono::duration<Real>)(CUR_TIME - start)).count());
	cufftDestroy(fftplan);

	procMultiplyPhase(streamLF, nx, ny, rx, ry, FFTLF_temp_gpu, RSplane_complex_field_gpu, CUDART_PI);

	LOG("\tprocMultiplyPhase() ... %.5lfsec\n", ((std::chrono::duration<Real>)(CUR_TIME - start)).count());
	cudaFree(complexLF_gpu);
	cudaFree(FFTLF_temp_gpu);
}

void ophLF::fresnelPropagation_GPU()
{
	int Nx = context_.pixel_number[_X];
	int Ny = context_.pixel_number[_Y];

	cufftDoubleComplex *in2x;
	cufftDoubleComplex *temp;

	HANDLE_ERROR(cudaMalloc((void**)&in2x, sizeof(cufftDoubleComplex)*Nx*Ny * 4));
	HANDLE_ERROR(cudaMalloc((void**)&temp, sizeof(cufftDoubleComplex)*Nx*Ny * 4));
	HANDLE_ERROR(cudaMemsetAsync(in2x, 0, sizeof(cufftDoubleComplex)*Nx*Ny * 4, streamLF));
	HANDLE_ERROR(cudaMemsetAsync(temp, 0, sizeof(cufftDoubleComplex)*Nx*Ny * 4, streamLF));

	procMoveToin2x(streamLF, Nx, Ny, RSplane_complex_field_gpu, in2x);

	cudaFFT(streamLF, Nx * 2, Ny * 2, in2x, temp, -1, false);

	Real wavelength = context_.wave_length[0];
	vec2 pp = context_.pixel_pitch;

	procMultiplyProp(streamLF, Nx * 2, Ny * 2, temp, CUDART_PI, distanceRS2Holo, wavelength, pp.v[0], pp.v[1]);

	HANDLE_ERROR(cudaMemsetAsync(in2x, 0, sizeof(cufftDoubleComplex)*Nx*Ny * 4, streamLF));
	cudaFFT(streamLF, Nx * 2, Ny * 2, temp, in2x, 1, false);

	HANDLE_ERROR(cudaMemsetAsync(RSplane_complex_field_gpu, 0, sizeof(cufftDoubleComplex)*Nx*Ny, streamLF));
	procCopyToOut(streamLF, Nx, Ny, in2x, RSplane_complex_field_gpu);

	cufftDoubleComplex* output = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*Nx*Ny);
	memset(output, 0.0, sizeof(cufftDoubleComplex)*Nx*Ny);
	HANDLE_ERROR(cudaMemcpyAsync(output, RSplane_complex_field_gpu, sizeof(cufftDoubleComplex)*Nx*Ny, cudaMemcpyDeviceToHost), streamLF);
	for (int i = 0; i < Nx*Ny; ++i)
	{
		// 1-channel로 코딩. 추후 변경
		(*complex_H)[i][_RE] = output[i].x;
		(*complex_H)[i][_IM] = output[i].y;
	}

	cudaFree(in2x);
	cudaFree(temp);
}