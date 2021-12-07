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

/**
* @file		ophPointCloud_GPU.h
* @brief	Openholo Point Cloud based CGH generation with CUDA GPGPU
* @author	Hyeong-Hak Ahn
* @date		2018/09
*/

#ifndef __ophPointCloud_GPU_h
#define __ophPointCloud_GPU_h

#include "ophPointCloud.h"

#define __DEBUG_LOG_GPU_SPEC_

/* CUDA Library Include */
#include <cuda_runtime.h>

#define __CUDA_INTERNAL_COMPILATION__ //for CUDA Math Module
#include <math_constants.h>
#include <math_functions.h> //Single Precision Floating
#include <math_functions_dbl_ptx3.h> //Double Precision Floating
#include <vector_functions.h> //Vector Processing Function
#undef __CUDA_INTERNAL_COMPILATION__

#define OPH_CUDA_N_STREAM 100
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

	double scale_X;		/// Scaling factor of x coordinate of point cloud
	double scale_Y;		/// Scaling factor of y coordinate of point cloud
	double scale_Z;		/// Scaling factor of z coordinate of point cloud

	double offset_depth;	/// Offset value of point cloud in z direction

	int pn_X;		/// Number of pixel of SLM in x direction
	int pn_Y;		/// Number of pixel of SLM in y direction

	double pp_X; /// Pixel pitch of SLM in x direction
	double pp_Y; /// Pixel pitch of SLM in y direction

	double half_ss_X; /// (pixel_x * nx) / 2
	double half_ss_Y; /// (pixel_y * ny) / 2

	double k;		  /// Wave Number = (2 * PI) / lambda;
	double lambda;

	KernelConst(
		const int &n_points,		/// number of point cloud
		const int &n_colors,		/// number of colors per point cloud
		const int &n_streams,		/// number of streams
		const vec3 &scale_factor,	/// Scaling factor of x, y, z coordinate of point cloud
		const Real &offset_depth,	/// Offset value of point cloud in z direction
		const ivec2 &pixel_number,	/// Number of pixel of SLM in x, y direction
		const vec2 &pixel_pitch,	/// Pixel pitch of SLM in x, y direction
		const vec2 &ss,				/// (pixel_x * nx), (pixel_y * ny)
		const Real &k,				/// Wave Number = (2 * PI) / lambda
		const Real &lambda,			/// Wave length
		const Real &ratio			/// Wave ratio
	)
	{
		this->n_points = n_points;
		this->n_colors = n_colors;
		this->n_streams = n_streams;
		this->scale_X = scale_factor[_X] * ratio;
		this->scale_Y = scale_factor[_Y] * ratio;
		this->scale_Z = scale_factor[_Z];
		this->offset_depth = offset_depth;

		// Output Image Size
		this->pn_X = pixel_number[_X];
		this->pn_Y = pixel_number[_Y];

		// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
		this->pp_X = pixel_pitch[_X];
		this->pp_Y = pixel_pitch[_Y];

		// Length (Width) of complex field at eyepiece plane (by simple magnification)
		this->half_ss_X = ss[_X] / 2;
		this->half_ss_Y = ss[_Y] / 2;

		// Wave Number
		this->k = k;

		this->lambda = lambda;
	}
} GpuConst;

cudaStream_t *streams;

typedef struct KernelConst_NotEncodedRS : public KernelConst {
	double det_tx;  /// tx / sqrt(1 - tx^2), tx = lambda / (2 * pp_X)
	double det_ty;  /// ty / sqrt(1 - ty^2), ty = lambda / (2 * pp_Y)

	KernelConst_NotEncodedRS(
		const int &n_points,		/// number of point cloud
		const int &n_colors,		/// number of colors per point cloud
		const int &n_streams,
		const vec3 &scale_factor,	/// Scaling factor of x, y, z coordinate of point cloud
		const Real &offset_depth,	/// Offset value of point cloud in z direction
		const ivec2 &pixel_number,	/// Number of pixel of SLM in x, y direction
		const vec2 &pixel_pitch,	/// Pixel pitch of SLM in x, y direction
		const vec2 &ss,				/// (pixel_x * nx), (pixel_y * ny)
		const Real &k,				/// Wave Number = (2 * PI) / lambda
		const Real &lambda,			/// Wave length = lambda
		const Real &ratio			/// Wave ratio
	)
		: KernelConst(n_points, n_colors, n_streams, scale_factor, offset_depth, pixel_number, pixel_pitch, ss, k, lambda, ratio)
	{
		double tx = lambda / (2 * pixel_pitch[_X]);
		double ty = lambda / (2 * pixel_pitch[_Y]);

		this->det_tx = tx / sqrt(1 - tx * tx);
		this->det_ty = ty / sqrt(1 - ty * ty);
	}

	KernelConst_NotEncodedRS(GpuConst &cuda_config)
		: KernelConst(cuda_config)
	{
		double tx = lambda / (2 * cuda_config.pp_X);
		double ty = lambda / (2 * cuda_config.pp_Y);

		this->det_tx = tx / sqrt(1 - tx * tx);
		this->det_ty = ty / sqrt(1 - ty * ty);
	}
} GpuConstNERS;


typedef struct KernelConst_NotEncodedFrsn : public KernelConst {

	double tx;	/// tx = lambda / (2 * pp_X)
	double ty;	/// ty = lambda / (2 * pp_Y)

	KernelConst_NotEncodedFrsn(
		const int &n_points,		/// number of point cloud
		const int &n_colors,		/// number of colors per point cloud
		const int &n_streams,
		const vec3 &scale_factor,	/// Scaling factor of x, y, z coordinate of point cloud
		const Real &offset_depth,	/// Offset value of point cloud in z direction
		const ivec2 &pixel_number,	/// Number of pixel of SLM in x, y direction
		const vec2 &pixel_pitch,	/// Pixel pitch of SLM in x, y direction
		const vec2 &ss,				/// (pixel_x * nx), (pixel_y * ny)
		const Real &k,				/// Wave Number = (2 * PI) / lambda
		const Real &lambda,			/// Wave length = lambda
		const Real &ratio			/// Wave ratio
	)
		: KernelConst(n_points, n_colors, n_streams, scale_factor, offset_depth, pixel_number, pixel_pitch, ss, k, lambda, ratio)
	{
		this->tx = lambda / (2 * pixel_pitch[_X]);
		this->ty = lambda / (2 * pixel_pitch[_Y]);
	}

	KernelConst_NotEncodedFrsn(GpuConst &cuda_config)
		: KernelConst(cuda_config)
	{
		this->tx = lambda / (2 * cuda_config.pp_X);
		this->ty = lambda / (2 * cuda_config.pp_Y);
	}
} GpuConstNEFR;


extern "C"
{	
	void cudaGenCghPointCloud_NotEncodedRS(
		const int &nBlocks, const int &nThreads, const int &n_pts_per_stream,
		double* cuda_pc_data, double* cuda_amp_data,
		double* cuda_dst_double, double* cuda_dst_imag,
		const GpuConstNERS* cuda_config, const uint &iChannel, const uint &mode);

	void cudaGenCghPointCloud_NotEncodedFrsn(
		const int &nBlocks, const int &nThreads, const int &n_pts_per_stream,
		double* cuda_pc_data, double* cuda_amp_data,
		double* cuda_dst_double, double* cuda_dst_imag,
		const GpuConstNEFR* cuda_config, const uint &iChannel);
}

#endif