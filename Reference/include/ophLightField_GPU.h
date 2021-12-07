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

#ifndef __ophLightField_GPU_h
#define __ophLightField_GPU_h

#include "ophLightField.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <math_constants.h>


static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		return;
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}} 

typedef struct KernelConst {
	int pnX;
	int pnY;
	Real ppX;
	Real ppY;
	Real lambda;
	Real pi2;
	Real k;
	Real distance;
	bool randomPhase;
	int nX;
	int nY;
	int rX;
	int rY;
	int nChannel;
	int iAmp;

	KernelConst(
		const int &channel,
		const int &iAmp,
		const int &pnX,
		const int &pnY,
		const Real &ppX,
		const Real &ppY,
		const int &nX,
		const int &nY,
		const int &rX,
		const int &rY,
		const Real &distance,
		const Real &k,
		const Real &lambda,
		const bool &random_phase
	)
	{
		this->nChannel = channel;
		this->iAmp = iAmp;
		this->pnX = pnX;
		this->pnY = pnY;
		this->ppX = ppX;
		this->ppY = ppY;
		this->nX = nX;
		this->nY = nY;
		this->rX = rX;
		this->rY = rY;
		this->lambda = lambda;
		this->pi2 = M_PI * 2;
		this->k = pi2 / lambda;
		this->distance = distance;
		this->randomPhase = random_phase;
	}
} LFGpuConst;
#if 0
#define IMG_R "img_resolution"
#define IMG_N "img_number"
#define CHANNEL_I "channel_info"
#endif
extern "C"
{
#if 0
	__constant__ int channel_info[2];
	__constant__ int img_resolution[3];
	__constant__ int img_number[3];
#endif
	void cudaConvertLF2ComplexField_Kernel(CUstream_st* stream, const int &nBlocks, const int &nThreads, const LFGpuConst *config, uchar1** LF, cufftDoubleComplex* output);
	void cudaFFT_LF(cufftHandle *plan, CUstream_st* stream, const int &nBlocks, const int &nThreads, const int &nx, const int &ny, cufftDoubleComplex* in_field, cufftDoubleComplex* output_field, const int &direction);
	//void cudaFFT_LF(CUstream_st* stream, const int &nBlocks, const int &nThreads, const int &nx, const int &ny, const LFGpuConst *config, cufftDoubleComplex* in_field, cufftDoubleComplex* output_field, int direction);

	void procMultiplyPhase(CUstream_st* stream, const int &nBlocks, const int &nThreads, const LFGpuConst *config, cufftDoubleComplex* in, cufftDoubleComplex* output);

	void cudaFresnelPropagationLF(
		const int &nBlocks, const int &nBlocks2, const int &nThreads, const int &nx, const int &ny,
		cufftDoubleComplex *src, cufftDoubleComplex *tmp, cufftDoubleComplex *tmp2, cufftDoubleComplex *dst,
		const LFGpuConst* cuda_config);
}


#endif