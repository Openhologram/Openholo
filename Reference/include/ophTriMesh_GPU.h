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

#ifndef __ophTriMesh_GPU_h
#define __ophTriMesh_GPU_h

#include "ophTriMesh.h"

#include    "sys.h"
#include	<stdio.h>
#include	<cuda_runtime_api.h>
#include	<cufft.h>
#include	<curand.h>
#include	<math_constants.h>


static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
     printf( "Host memory failed in %s at line %d\n", __FILE__, __LINE__ ); \
     exit( EXIT_FAILURE );}} 

#define CUDA_CALL(x) { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit( EXIT_FAILURE ); }}
#define CURAND_CALL(x) { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
	exit( EXIT_FAILURE ); }}


typedef struct MeshKernelConfig {
	int pn_X;		/// Number of pixel of SLM in x direction
	int pn_Y;		/// Number of pixel of SLM in y direction
	double pp_X; /// Pixel pitch of SLM in x direction
	double pp_Y; /// Pixel pitch of SLM in y direction
	unsigned int shading_flag; // flat or continuous
	double dfx;
	double dfy;
	double lambda;
	double min_double;
	double tolerence;
	double pi;
	double pi2;
	double square_pi2;
	double cube_pi2;

	MeshKernelConfig(
		const ivec2& pixel_number,	/// Number of pixel of SLM in x, y direction
		const vec2& pixel_pitch,	/// Pixel pitch of SLM in x, y direction
		const Real& lambda,			/// Wave length
		const uint& shading_flag
	)
	{
		// Output Image Size
		this->pn_X = pixel_number[_X];
		this->pn_Y = pixel_number[_Y];

		// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
		this->pp_X = pixel_pitch[_X];
		this->pp_Y = pixel_pitch[_Y];

		this->lambda = lambda;

		this->dfx = (1.0 / this->pp_X) / this->pn_X;
		this->dfy = (1.0 / this->pp_Y) / this->pn_Y;

		this->shading_flag = shading_flag;

		min_double = (double)2.2250738585072014e-308;
		tolerence = 1e-12;
		pi = M_PI;
		pi2 = pi * 2;
		square_pi2 = pi2 * pi2;
		cube_pi2 = square_pi2 * pi2;
	}
} MeshKernelConfig;


extern "C"
{
	void cudaMesh_Flat(
		const int& nBlocks, const int& nThreads, cufftDoubleComplex* output,
		const MeshKernelConfig* config, double shading_factor, const geometric* geom,
		double carrierWaveX, double carrierWaveY, double carrierWaveZ, CUstream_st* stream
	);

	void cudaMesh_Continuous(
		const int& nBlocks, const int& nThreads, cufftDoubleComplex* output,
		const MeshKernelConfig* config, const geometric* geom, double av0, double av1, double av2,
		double carrierWaveX, double carrierWaveY, double carrierWaveZ, CUstream_st* stream
	);

	void call_fftGPU(int nx, int ny, cufftDoubleComplex* input, cufftDoubleComplex* output, CUstream_st* streamTriMesh);
	void call_fftGPUf(int nx, int ny, cuFloatComplex* input, cuFloatComplex* output, CUstream_st* streamTriMesh);

}

#endif
