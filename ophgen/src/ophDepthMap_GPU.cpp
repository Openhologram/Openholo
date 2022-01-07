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

#include	"ophDepthMap.h"
#include	"ophDepthMap_GPU.h"
#include	<sys.h> //for LOG() macro

void empty(int a, int b, int& sum)
{
	int* arr;
	int idx = 0;
	arr = new int[b - a + 1];

	idx == b - a ? sum += arr[idx] : idx++;
}

/**
* @brief Initialize variables for the GPU implementation.
* @details Memory allocation for the GPU variables.
* @see initialize
*/
void ophDepthMap::initGPU()
{
	const int nx = context_.pixel_number[0];
	const int ny = context_.pixel_number[1];
	const int N = nx * ny;

	if (!stream_)
		cudaStreamCreate(&stream_);

	if (img_src_gpu)	cudaFree(img_src_gpu);
	HANDLE_ERROR(cudaMalloc((void**)&img_src_gpu, sizeof(uchar1)*N));

	if (dimg_src_gpu)	cudaFree(dimg_src_gpu);
	HANDLE_ERROR(cudaMalloc((void**)&dimg_src_gpu, sizeof(uchar1)*N));

	if (depth_index_gpu) cudaFree(depth_index_gpu);
	if (dm_config_.change_depth_quantization == 1)
		HANDLE_ERROR(cudaMalloc((void**)&depth_index_gpu, sizeof(Real)*N));

	if (u_o_gpu_)	cudaFree(u_o_gpu_);
	if (u_complex_gpu_)	cudaFree(u_complex_gpu_);

	HANDLE_ERROR(cudaMalloc((void**)&u_o_gpu_, sizeof(cufftDoubleComplex)*N));
	HANDLE_ERROR(cudaMalloc((void**)&u_complex_gpu_, sizeof(cufftDoubleComplex)*N));

	if (k_temp_d_)	cudaFree(k_temp_d_);
	HANDLE_ERROR(cudaMalloc((void**)&k_temp_d_, sizeof(cufftDoubleComplex)*N));
}

/**
* @brief Copy input image & depth map data into a GPU.
* @return true if input data are sucessfully copied on GPU, flase otherwise.
* @see readImageDepth
*/
bool ophDepthMap::prepareInputdataGPU()
{
	LOG("%s : ", __FUNCTION__);
	auto begin = CUR_TIME;
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint N = pnX * pnY;

	HANDLE_ERROR(cudaMemcpyAsync(dimg_src_gpu, depth_img, sizeof(uchar1) * N, cudaMemcpyHostToDevice, stream_));
	LOG("%lf (s)\n", ELAPSED_TIME(begin, CUR_TIME));
	return true;
}

/**
* @brief Quantize depth map on the GPU, when the number of depth quantization is not the default value (i.e. change_depth_quantization == 1 ).
* @details Calculate the value of 'depth_index_gpu'.
* @see getDepthValues
*/
void ophDepthMap::changeDepthQuanGPU()
{
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const int N = pnX * pnY;

	HANDLE_ERROR(cudaMemsetAsync(depth_index_gpu, 0, sizeof(Real) * N, stream_));

	for (uint dtr = 0; dtr < dm_config_.num_of_depth; dtr++)
	{
		Real temp_depth = dlevel[dtr];
		Real d1 = temp_depth - dstep / 2.0;
		Real d2 = temp_depth + dstep / 2.0;

		cudaChangeDepthQuanKernel(stream_, pnX, pnY, depth_index_gpu, dimg_src_gpu,
			dtr, d1, d2, dm_config_.num_of_depth, dm_config_.far_depthmap, dm_config_.near_depthmap);
	}
}

/**
* @brief Main method for generating a hologram on the GPU.
* @details For each depth level,
*   1. find each depth plane of the input image.
*   2. apply carrier phase delay.
*   3. propagate it to the hologram plan.
*   4. accumulate the result of each propagation.
* .
* It uses CUDA kernels, cudaDepthHoloKernel & cudaPropagation_AngularSpKernel.<br>
* The final result is accumulated in the variable 'u_complex_gpu_'.
* @param frame : the frame number of the image.
* @see calc_Holo_by_Depth
*/
void ophDepthMap::calcHoloGPU()
{
	LOG("%s : ", __FUNCTION__);
	auto begin = CUR_TIME;

	if (!stream_)
		cudaStreamCreate(&stream_);

	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];
	const Real ssX = context_.ss[_X] = pnX * ppX;
	const Real ssY = context_.ss[_Y] = pnY * ppY;
	const int N = pnX * pnY;
	const int nChannel = context_.waveNum;

	size_t depth_sz = dm_config_.render_depth.size();

	const bool bRandomPhase = GetRandomPhase();

	for (int ch = 0; ch < nChannel; ch++)
	{
		HANDLE_ERROR(cudaMemsetAsync(u_complex_gpu_, 0, sizeof(cufftDoubleComplex) * N, stream_));
		HANDLE_ERROR(cudaMemcpyAsync(img_src_gpu, m_vecRGB[ch], sizeof(uchar1) * N, cudaMemcpyHostToDevice, stream_));
		Real lambda = context_.wave_length[ch];
		Real k = context_.k = (2 * M_PI / lambda);

		for (int p = 0; p < depth_sz; ++p)
		{
			Complex<Real> rand_phase_val;
			GetRandomPhaseValue(rand_phase_val, bRandomPhase);

			int dtr = dm_config_.render_depth[p];
			Real temp_depth = (is_ViewingWindow) ? dlevel_transform[dtr - 1] : dlevel[dtr - 1];
			Complex<Real> carrier_phase_delay(0, k * -temp_depth);
			carrier_phase_delay.exp();

			HANDLE_ERROR(cudaMemsetAsync(u_o_gpu_, 0, sizeof(cufftDoubleComplex) * N, stream_));

			cudaDepthHoloKernel(stream_, pnX, pnY, u_o_gpu_, img_src_gpu, dimg_src_gpu, depth_index_gpu,
				dtr, rand_phase_val[_RE], rand_phase_val[_IM], carrier_phase_delay[_RE], carrier_phase_delay[_IM], dm_config_.change_depth_quantization, dm_config_.default_depth_quantization);

			HANDLE_ERROR(cudaMemsetAsync(k_temp_d_, 0, sizeof(cufftDoubleComplex) * N, stream_));

			cudaFFT(stream_, pnX, pnY, u_o_gpu_, k_temp_d_, -1);


			cudaPropagation_AngularSpKernel(stream_, pnX, pnY, k_temp_d_, u_complex_gpu_,
				ppX, ppY, ssX, ssY, lambda, context_.k, temp_depth);

			//propagationAngularSpectrumGPU(ch, u_o_gpu_, temp_depth);

			m_nProgress = (int)((Real)(ch * depth_sz + p + 1) * 100 / ((Real)depth_sz * nChannel));
		}

		cudaMemcpy(complex_H[ch], u_complex_gpu_, sizeof(cufftDoubleComplex) * N, cudaMemcpyDeviceToHost);
	}
	LOG("%lf (s)\n", ELAPSED_TIME(begin, CUR_TIME));
}

void ophDepthMap::free_gpu()
{
	if (u_o_gpu_)		cudaFree(u_o_gpu_);
	if (u_complex_gpu_)	cudaFree(u_complex_gpu_);
	if (k_temp_d_)		cudaFree(k_temp_d_);
}