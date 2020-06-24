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

#ifndef __ophSig_GPU_H_
#define __ophSig_GPU_H_

#include	<cufft.h>
#include <npp.h>
//#include	"ophSig.h"
#include <cuda_profiler_api.h>

#include <cuda_runtime.h>

cufftDoubleComplex* complex_holog_gpu;

cudaStream_t streamLF;

extern "C"
{
	void cudaCvtFieldToCuFFT(Complex<Real> *src_data, cufftDoubleComplex *dst_data, int nx, int ny);
	void cudaCvtCuFFTToField(cufftDoubleComplex *src_data, Complex<Real> *dst_data, int nx, int ny);
	void cudaCuFFT(cufftHandle* plan, cufftDoubleComplex *src_data, cufftDoubleComplex *dst_data, int nx, int ny, int direction);
	
	void cudaCuIFFT(cufftHandle* plan, cufftDoubleComplex *src_data, cufftDoubleComplex *dst_data, int nx, int ny, int direction);
	
	
	void cudaCvtOFF( Complex<Real> *src_data, Real *dst_data, ophSigConfig *device_config,  int nx, int ny, Real wl,Complex<Real> *F, Real *angle);
	void cudaCvtHPO(CUstream_st* stream, cufftDoubleComplex *src_data, cufftDoubleComplex *dst_data, ophSigConfig *device_config, Complex<Real> *F,int nx, int ny, Real Rephase, Real Imphase);
	void cudaCvtCAC(cufftDoubleComplex *src_data, cufftDoubleComplex *dst_data,  Complex<Real> *FFZP, ophSigConfig *device_config, int nx, int ny,Real sigmaf, Real radius);

	void cudaPropagation(cufftDoubleComplex *src_data, cufftDoubleComplex *dst_data, Complex<Real> *FH, ophSigConfig *device_config, int nx, int ny, Real sigmaf);
	double cudaGetParamSF(cufftHandle *fftplan, cufftDoubleComplex *src_data,  cufftDoubleComplex *temp_data, cufftDoubleComplex *dst_data, Real *f, Complex<Real> *FH, ophSigConfig *device_config, int nx, int ny, float zMax, float zMin, int sampN, float th, Real wl);
	void cudaGetParamAT1(Complex<Real> *src_data, Complex<Real> *Flr, Complex<Real> *Fli, Complex<Real> *G, ophSigConfig *device_config, int nx, int ny, Real_t NA_g, Real wl);
	void cudaGetParamAT2(Complex<Real> *Flr, Complex<Real> *Fli, Complex<Real> *G, Complex<Real> *temp_data, int nx, int ny);

}
#endif