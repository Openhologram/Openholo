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
#include	<cuda_runtime.h>
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

cufftDoubleComplex *angularSpectrum_GPU;
cufftDoubleComplex *ffttemp;

cudaStream_t	streamTriMesh;

extern "C"
{
	void call_cudaKernel_refAS(cufftDoubleComplex* output, int nx, int ny, double px, double py, unsigned int SHADING_FLAG, int idx, 
		double waveLength, double pi, double shadingFactor, double av0, double av1, double av2,
		double glRot0, double glRot1, double glRot2, double glRot3, double glRot4, double glRot5, double glRot6, double glRot7, double glRot8,
		double loRot0, double loRot1, double loRot2, double loRot3, double glShiftX, double glShiftY, double glShiftZ,
		double carrierWaveX, double carrierWaveY, double carrierWaveZ, double min_double, double tolerence, CUstream_st* streamTriMesh);
	
	void call_cudaKernel_refASf(cuFloatComplex* output, int nx, int ny, float px, float py, unsigned int sflag, int idx, float waveLength,
		float pi, float shadingFactor, float av0, float av1, float av2,
		float glRot0, float glRot1, float glRot2, float glRot3, float glRot4, float glRot5, float glRot6, float glRot7, float glRot8,
		float loRot0, float loRot1, float loRot2, float loRot3, float glShiftX, float glShiftY, float glShiftZ,
		float carrierWaveX, float carrierWaveY, float carrierWaveZ, float min_double, float tolerence, CUstream_st* streamTriMesh);

	void call_fftGPU(int nx, int ny, cufftDoubleComplex* input, cufftDoubleComplex* output, CUstream_st* streamTriMesh);
	void call_fftGPUf(int nx, int ny, cuFloatComplex* input, cuFloatComplex* output, CUstream_st* streamTriMesh);

}

#endif
