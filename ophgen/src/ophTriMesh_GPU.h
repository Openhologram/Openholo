#pragma once
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

//#ifndef __ophTriMesh_GPU_h
//#define __ophTriMesh_GPU_h

#include "ophTriMesh.h"

#define __DEBUG_LOG_GPU_SPEC_

#include <cuda_runtime.h>
#include <cufft.h>

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
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}} 


cufftDoubleComplex * k_input_d;
cufftDoubleComplex * k_output_d;
cufftDoubleComplex * k_temp_d;

cudaStream_t	stream_;
cudaEvent_t		start, stop;

extern float kCGHGenerationTime;

extern "C"
{
	void cudaGetFringeFromGPUKernel(CUstream_st* stream, int N, double* save_a_d_, double* save_b_d_, int nx, int ny, cufftDoubleComplex* in_filed, cufftDoubleComplex* output_field, bool isCrop, int SignalLoc1, int SignalLoc2, int direction);

	void cudaPolygonKernel(CUstream_st* stream, int N, double* real_part_hologram, double* imagery_part_hologram, double* intensities, cufftDoubleComplex* temp_term,
		int vertex_idx, int nx, int ny, double px, double py, double ss1, double ss2, double lambda, double pi, double tolerence,
		double del_fxx, double del_fyy, double f_cx, double f_cy, double f_cz, bool is_multiple_carrier_wave, double cw_amp,
		double t_Coff00, double t_Coff01, double t_Coff02, double t_Coff10, double t_Coff11, double t_Coff12,
		double detAff, double R_31, double R_32, double R_33, double T1, double T2, double T3);

	void cudaTranslationMatrixKernel(CUstream_st* stream_, int N, cufftDoubleComplex* temp_term, double* save_a_d_, double* save_b_d_, int nx, int ny, double px, double py, double ss1, double ss2, double lambda,
		int disp_x, int disp_y, double cw_amp, double R_31, double R_32, double R_33);

	void cudaFFT(CUstream_st* stream, int N, int nx, int ny, cufftDoubleComplex* in_filed, cufftDoubleComplex* output_field, int direction);
}