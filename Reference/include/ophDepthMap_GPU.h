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
* @file		ophDepthMap_GPU.h
* @brief	Openholo Point Cloud based CGH generation with CUDA GPGPU
* @author	Hyeong-Hak Ahn
* @date		2018/09
*/

#ifndef __ophDepthMap_GPU_h
#define __ophDepthMap_GPU_h

#include "ophDepthMap.h"

#define __DEBUG_LOG_GPU_SPEC_

/* CUDA Library Include */
#include	<cuda_runtime.h>
#include	<cufft.h>
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
cufftDoubleComplex *u_o_gpu_;
cufftDoubleComplex *u_complex_gpu_;
cufftDoubleComplex *k_temp_d_;

cudaStream_t	stream_;

extern "C"
{
	/**
	* @brief Convert data from the spatial domain to the frequency domain using 2D FFT on GPU.
	* @details call CUDA Kernel - fftShift and CUFFT Library.
	* @param stream : CUDA Stream
	* @param nx : the number of column of the input data
	* @param ny : the number of row of the input data
	* @param in_field : input complex data variable
	* @param output_field : output complex data variable
	* @param direction : If direction == -1, forward FFT, if type == 1, inverse FFT.
	* @param bNomarlized : If bNomarlized == true, normalize the result after FFT.
	* @see propagation_AngularSpectrum_GPU, encoding_GPU
	*/
	void cudaFFT(CUstream_st* stream, int nx, int ny, cufftDoubleComplex* in_filed, cufftDoubleComplex* output_field, int direction);


	//void cudaCropFringe(CUstream_st* stream, int nx, int ny, cufftDoubleComplex* in_field, cufftDoubleComplex* out_field, int cropx1, int cropx2, int cropy1, int cropy2);

	/**
	* @brief Find each depth plane of the input image and apply carrier phase delay to it on GPU.
	* @details call CUDA Kernel - depth_sources_kernel.
	* @param stream : CUDA Stream
	* @param nx : the number of column of the input data
	* @param ny : the number of row of the input data
	* @param u_o_gpu_ : output variable
	* @param img_src_gpu : input image data
	* @param dimg_src_gpu : input depth map data
	* @param depth_index_gpu : input quantized depth map data
	* @param dtr : current working depth level
	* @param rand_phase_val_a : the Real part of the random phase value
	* @param rand_phase_val_b : the imaginary part of the random phase value
	* @param carrier_phase_delay_a : the Real part of the carrier phase delay
	* @param carrier_phase_delay_b : the imaginary part of the carrier phase delay
	* @param flag_change_depth_quan : if true, change the depth quantization from the default value.
	* @param default_depth_quan : default value of the depth quantization - 256
	* @see calc_Holo_GPU
	*/
	void cudaDepthHoloKernel(CUstream_st* stream, int pnx, int pny, cufftDoubleComplex* u_o_gpu_, unsigned char* img_src_gpu, unsigned char* dimg_src_gpu, Real* depth_index_gpu,
		int dtr, Real rand_phase_val_a, Real rand_phase_val_b, Real carrier_phase_delay_a, Real carrier_phase_delay_b, int flag_change_depth_quan, unsigned int default_depth_quan);

	/**
	* @brief Angular spectrum propagation method for GPU implementation.
	* @details The propagation results of all depth levels are accumulated in the variable 'u_complex_gpu_'.
	* @param stream : CUDA Stream
	* @param pnx : the number of column of the input data
	* @param pny : the number of row of the input data
	* @param input_d : input data
	* @param u_complex : output data
	* @param ppx : pixel pitch of x-axis
	* @param ppy : pixel pitch of y-axis
	* @param ssx : pnx * ppx
	* @param ssy : pny * ppy
	* @param lambda : wavelength
	* @param params_k :  2 * PI / lambda
	* @param propagation_dist : the distance from the object to the hologram plane
	* @see propagation_AngularSpectrum_GPU
	*/
	void cudaPropagation_AngularSpKernel(CUstream_st* stream_, int pnx, int pny, cufftDoubleComplex* input_d, cufftDoubleComplex* u_complex,
		Real ppx, Real ppy, Real ssx, Real ssy, Real lambda, Real params_k, Real propagation_dist);

	//void cudaGetFringe(CUstream_st* stream, int pnx, int pny, cufftDoubleComplex* in_field, cufftDoubleComplex* out_field, int sig_locationx, int sig_locationy,
	//	Real ssx, Real ssy, Real ppx, Real ppy, Real PI);

	/**
	* @brief Quantize depth map on the GPU, only when the number of depth quantization is not the default value (i.e. FLAG_CHANGE_DEPTH_QUANTIZATION == 1 ).
	* @details Calculate the value of 'depth_index_gpu'.
	* @param stream : CUDA Stream
	* @param pnx : the number of column of the input data
	* @param pny : the number of row of the input data
	* @param depth_index_gpu : output variable
	* @param dimg_src_gpu : input depth map data
	* @param dtr : the current working depth level
	* @param d1 : the starting physical point of each depth level
	* @param d2 : the ending physical point of each depth level
	* @param params_num_of_depth : the number of depth level
	* @param params_far_depthmap : NEAR_OF_DEPTH_MAP at config file
	* @param params_near_depthmap : FAR_OF_DEPTH_MAP at config file
	* @see change_depth_quan_GPU
	*/
	void cudaChangeDepthQuanKernel(CUstream_st* stream_, int pnx, int pny, Real* depth_index_gpu, unsigned char* dimg_src_gpu,
		int dtr, Real d1, Real d2, Real params_num_of_depth, Real params_far_depthmap, Real params_near_depthmap);

	/**@}*/

}



#endif