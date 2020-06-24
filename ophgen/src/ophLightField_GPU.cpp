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
	auto begin = CUR_TIME;

	const int nX = num_image[_X];
	const int nY = num_image[_Y];
	const int nXY = nX * nY;
	const int rX = resolution_image[_X];
	const int rY = resolution_image[_Y];
	const int rXY = rX * rY;

	if (!streamLF)
		cudaStreamCreate(&streamLF);

	if (LF_gpu) cudaFree(LF_gpu);
	if (LFData_gpu) {
		for (int i = 0; i < nXY; i++)
			cudaFree(LFData_gpu[i]);
		free(LFData_gpu);
	}

	HANDLE_ERROR(cudaMalloc(&LF_gpu, sizeof(uchar1*) * nXY));
	LFData_gpu = (uchar**)malloc(sizeof(uchar*) * nXY);

	for (int i = 0; i < nXY; i++) {
		HANDLE_ERROR(cudaMalloc(&LFData_gpu[i], sizeof(uchar1) * rXY));
		HANDLE_ERROR(cudaMemset(LFData_gpu[i], 0, sizeof(uchar1) * rXY));
		HANDLE_ERROR(cudaMemcpyAsync(LFData_gpu[i], LF[i], sizeof(uchar) * rXY, cudaMemcpyHostToDevice), streamLF);
	}

	HANDLE_ERROR(cudaMemcpy(LF_gpu, LFData_gpu, sizeof(uchar*) * nXY, cudaMemcpyHostToDevice));

	if (RSplane_complex_field_gpu)
		cudaFree(RSplane_complex_field_gpu);
	HANDLE_ERROR(cudaMalloc((void**)&RSplane_complex_field_gpu, sizeof(cufftDoubleComplex) * nXY * rXY));

	auto end = CUR_TIME;
	LOG("\n%s : %lf(s)\n\n", __FUNCTION__, ((chrono::duration<Real>)(end - begin)).count());
}

void ophLF::convertLF2ComplexField_GPU()
{
	auto begin = CUR_TIME;

	const int nX = num_image[_X];
	const int nY = num_image[_Y];
	const int nXY = nX * nY;
	const int rX = resolution_image[_X];
	const int rY = resolution_image[_Y];
	const int rXY = rX * rY;

	cufftDoubleComplex *complexLF_gpu;
	cufftDoubleComplex *FFTLF_temp_gpu;

	HANDLE_ERROR(cudaMalloc((void**)&complexLF_gpu, sizeof(cufftDoubleComplex) * nXY * rXY));
	HANDLE_ERROR(cudaMalloc((void**)&FFTLF_temp_gpu, sizeof(cufftDoubleComplex) * nXY * rXY));
	HANDLE_ERROR(cudaMemsetAsync(complexLF_gpu, 0, sizeof(cufftDoubleComplex) * nXY * rXY, streamLF));
	HANDLE_ERROR(cudaMemsetAsync(FFTLF_temp_gpu, 0, sizeof(cufftDoubleComplex) * nXY * rXY, streamLF));
	HANDLE_ERROR(cudaMemsetAsync(RSplane_complex_field_gpu, 0, sizeof(cufftDoubleComplex) * nXY * rXY, streamLF));

	cudaConvertLF2ComplexField_Kernel(streamLF, nX, nY, rX, rY, LF_gpu, complexLF_gpu);
	cufftHandle fftplan;
	if (cufftPlan2d(&fftplan, nY, nX, CUFFT_Z2Z) != CUFFT_SUCCESS)
	{
		LOG("FAIL in creating cufft plan");
		return;
	};
	cufftDoubleComplex* in, *out;
#if 0
	for (int k = 0; k < nXY; k++)
	{
		int offset = rX * rY * k;
		in = complexLF_gpu + offset;
		out = FFTLF_temp_gpu + offset;
		cudaFFT_LF(&fftplan, streamLF, rX, rY, in, out, -1);
	}
#else
	for (int k = 0; k < rXY; k++)
	{
		int offset = nX * nY * k;
		in = complexLF_gpu + offset;
		out = FFTLF_temp_gpu + offset;
		cudaFFT_LF(&fftplan, streamLF, nX, nY, in, out, -1);
	}
#endif
	cufftDestroy(fftplan);

	procMultiplyPhase(streamLF, nX, nY, rX, rY, FFTLF_temp_gpu, RSplane_complex_field_gpu, CUDART_PI);
	cudaFree(complexLF_gpu);
	cudaFree(FFTLF_temp_gpu);

	auto end = CUR_TIME;
	LOG("\n%s : %lf(s)\n\n", __FUNCTION__, ((std::chrono::duration<Real>)(end - begin)).count());
}

void ophLF::fresnelPropagation_GPU()
{
	auto begin = CUR_TIME;

	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const int pnXY = pnX * pnY;
	const uint nChannel = context_.waveNum;
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];

	cufftDoubleComplex *in2x;
	cufftDoubleComplex *temp;

	HANDLE_ERROR(cudaMalloc((void**)&in2x, sizeof(cufftDoubleComplex) * pnXY * 4));
	HANDLE_ERROR(cudaMalloc((void**)&temp, sizeof(cufftDoubleComplex) * pnXY * 4));
	HANDLE_ERROR(cudaMemsetAsync(in2x, 0, sizeof(cufftDoubleComplex) * pnXY * 4, streamLF));
	HANDLE_ERROR(cudaMemsetAsync(temp, 0, sizeof(cufftDoubleComplex) * pnXY * 4, streamLF));

	procMoveToin2x(streamLF, pnX, pnY, RSplane_complex_field_gpu, in2x);

	cudaFFT(streamLF, pnX * 2, pnY * 2, in2x, temp, -1, false);
	

	for (uint ch = 0; ch < nChannel; ch++) {
		Real wavelength = context_.wave_length[ch];

		procMultiplyProp(streamLF, pnX * 2, pnY * 2, temp, CUDART_PI, distanceRS2Holo, wavelength, ppX, ppY);

		HANDLE_ERROR(cudaMemsetAsync(in2x, 0, sizeof(cufftDoubleComplex) * pnXY * 4, streamLF));
		cudaFFT(streamLF, pnX * 2, pnY * 2, temp, in2x, 1, false);

		HANDLE_ERROR(cudaMemsetAsync(RSplane_complex_field_gpu, 0, sizeof(cufftDoubleComplex) * pnXY, streamLF));
		procCopyToOut(streamLF, pnX, pnY, in2x, RSplane_complex_field_gpu);

		cufftDoubleComplex* output = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex) * pnXY);
		memset(output, 0.0, sizeof(cufftDoubleComplex) * pnXY);
		HANDLE_ERROR(cudaMemcpyAsync(output, RSplane_complex_field_gpu, sizeof(cufftDoubleComplex) * pnXY, cudaMemcpyDeviceToHost), streamLF);
		for (int i = 0; i < pnXY; ++i)
		{
			complex_H[ch][i][_RE] = output[i].x;
			complex_H[ch][i][_IM] = output[i].y;
		}
		free(output);
	}
	cudaFree(in2x);
	cudaFree(temp);

	auto end = CUR_TIME;
	LOG("\n%s : %lf(s)\n\n", __FUNCTION__, ((std::chrono::duration<Real>)(end - begin)).count());
}