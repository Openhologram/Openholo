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
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}} 


cufftDoubleComplex *RSplane_complex_field_gpu;

uchar1**		LF_gpu;
uchar**			LFData_gpu;
cudaStream_t	streamLF;


extern "C"
{
	void cudaConvertLF2ComplexField_Kernel(CUstream_st* stream, int nx, int ny, int rx, int ry, uchar1** LF, cufftDoubleComplex* output);

	void cudaFFT_LF(cufftHandle* plan, CUstream_st* stream, int nx, int ny, cufftDoubleComplex* in_field, cufftDoubleComplex* output_field, int direction);

	void procMultiplyPhase(CUstream_st* stream, int nx, int ny, int rx, int ry, cufftDoubleComplex* input, cufftDoubleComplex* output, double PI);

	void procMoveToin2x(CUstream_st* streamLF, int Nx, int Ny, cufftDoubleComplex* in, cufftDoubleComplex* out);

	void procMultiplyProp(CUstream_st* stream, int Nx, int Ny, cufftDoubleComplex* inout, double PI, double dist, double wavelength, double ppx, double ppy);

	void procCopyToOut(CUstream_st* stream, int Nx, int Ny, cufftDoubleComplex* in, cufftDoubleComplex* out);
}


#endif