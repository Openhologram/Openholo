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

//#ifndef __ophRec_GPU_h
//#define __ophRec_GPU_h

#include "ophRec.h"
#include <cuda_runtime.h>
#include <cufft.h>

#define __CUDA_INTERNAL_COMPILATION__ //for CUDA Math Module
#include <math_constants.h>
#include <math_functions.h> //Single Precision Floating
#include <math_functions_dbl_ptx3.h> //Double Precision Floating
#include <vector_functions.h> //Vector Processing Function
#undef __CUDA_INTERNAL_COMPILATION__

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

typedef struct KernelConst {
	int channel;	/// number of colors per point cloud
	int n_streams;	/// number of streams

	Real distance;

	int pnX;		/// Number of pixel of SLM in x direction
	int pnY;		/// Number of pixel of SLM in y direction

	double ppX; /// Pixel pitch of SLM in x direction
	double ppY; /// Pixel pitch of SLM in y direction

	double k;		  /// Wave Number = (2 * PI) / lambda;
	double lambda;    /// wave length = lambda;

	bool bRandomPhase;	// use random phase
	double pi2;

	double tx;
	double ty;
	double dx;
	double dy;
	double htx;
	double hty;
	double hdx;
	double hdy;

	double baseX;
	double baseY;

	KernelConst(
		const int &channel,		/// number of colors per point cloud
		const int &n_streams,
		const int &pnX,	/// Number of pixel of SLM in x, y direction
		const int &pnY,	/// Number of pixel of SLM in x, y direction
		const Real &ppX,	/// Pixel pitch of SLM in x, y direction
		const Real &ppY,	/// Pixel pitch of SLM in x, y direction
		const Real &propagation_distance, /// propagation distance
		const Real &k,				/// Wave Number = (2 * PI) / lambda
		const Real &lambda,        /// wave length
		const bool &random_phase
	)
	{
		this->channel = channel;
		this->n_streams = n_streams;
		this->lambda = lambda;

		// Output Image Size
		this->pnX = pnX;
		this->pnY = pnY;

		// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
		this->ppX = ppX;
		this->ppY = ppY;

		// WRP 
		this->distance = propagation_distance;

		// Wave Number
		this->k = k;

		this->lambda = lambda;

		// Random Phase
		this->bRandomPhase = random_phase;
		
		this->pi2 = M_PI * 2;

		tx = 1 / ppX;
		ty = 1 / ppY;
		dx = tx / pnX;
		dy = ty / pnY;

		htx = tx / 2;
		hty = ty / 2;
		hdx = dx / 2;
		hdy = dy / 2;
		baseX = -htx + hdx;
		baseY = -hty + hdy;

	}
} RecGpuConst;


extern "C"
{
	void cudaASMPropagation(
		const int &nBlocks, const int &nThreads, const int &nx, const int &ny,
		cuDoubleComplex *src, cuDoubleComplex *dst, Real *encode, const RecGpuConst* cuda_config);
}