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

//#ifndef __ophWRP_GPU_h
//#define __ophWRP_GPU_h

#include "ophWRP.h"

#define __DEBUG_LOG_GPU_SPEC_

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

// for PointCloud only GPU
typedef struct KernelConst {
	int n_points;	/// number of point cloud
	int n_colors;	/// number of colors per point cloud
	int n_streams;	/// number of streams

	Real wrp_d;	/// wrp location
	Real propa_d;  /// propagation distance

	int pn_X;		/// Number of pixel of SLM in x direction
	int pn_Y;		/// Number of pixel of SLM in y direction

	double pp_X; /// Pixel pitch of SLM in x direction
	double pp_Y; /// Pixel pitch of SLM in y direction

	double k;		  /// Wave Number = (2 * PI) / lambda;
	double lambda;    /// wave length = lambda;

	Real zmax;
	bool bRandomPhase;	// use random phase
	int iAmplitude;

	double pi2;
	double tx;
	double ty;
	double det_tx;
	double det_ty;

	KernelConst(
		const int &n_points,		/// number of point cloud
		const int &n_colors,		/// number of colors per point cloud
		const int &n_streams,
		const ivec2 &pixel_number,	/// Number of pixel of SLM in x, y direction
		const vec2 &pixel_pitch,	/// Pixel pitch of SLM in x, y direction
		const Real wrp_dis,    /// WRP location
		const Real propagation_distance, /// propagation distance
		const Real depth_max,
		const Real &k,				/// Wave Number = (2 * PI) / lambda
		const Real &lambda,        /// wave length
		const bool &random_phase,
		const int &index_amplitude
	)
	{
		this->lambda = lambda;

		this->n_points = n_points;
		this->n_colors = n_colors;
		this->n_streams = n_streams;

		// Output Image Size
		this->pn_X = pixel_number[_X];
		this->pn_Y = pixel_number[_Y];

		// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
		this->pp_X = pixel_pitch[_X];
		this->pp_Y = pixel_pitch[_Y];

		// WRP 
		this->wrp_d = wrp_dis;
		this->propa_d = propagation_distance;
		this->zmax = depth_max;

		// Wave Number
		this->k = k;

		this->lambda = lambda;

		// Random Phase
		this->bRandomPhase = random_phase;

		// Amplitude index
		this->iAmplitude = index_amplitude;

		this->pi2 = M_PI * 2;

		this->tx = lambda / (2 * pp_X);
		this->ty = lambda / (2 * pp_Y);
		this->det_tx = tx / sqrt(1 - tx * tx);
		this->det_ty = ty / sqrt(1 - ty * ty);

	}
} WRPGpuConst;

//cufftDoubleComplex *p_wrp_gpu_;


extern "C"
{
	void cudaFresnelPropagationWRP(
		const int &nBlocks, const int &nBlocks2, const int &nThreads, const int &nx, const int &ny,
		cuDoubleComplex *src, cuDoubleComplex *dst, cufftDoubleComplex *fftsrc, cufftDoubleComplex *fftdst,
		const WRPGpuConst* cuda_config);

	void cudaGenWRP(
		const int &nBlocks, const int &nThreads, const int &n_pts_per_stream,
		Real* cuda_pc_data, Real* cuda_amp_data,
		cuDoubleComplex* cuda_dst, const WRPGpuConst* cuda_config);
}