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
* @file		ophPCKernel.cu
* @brief	Openholo Point Cloud based CGH generation with CUDA GPGPU
* @author	Hyeong-Hak Ahn
* @date		2018/09
*/

#ifndef OphPCKernel_cu__
#define OphPCKernel_cu__

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "typedef.h"
#include "ophPointCloud_GPU.h"


__global__ void cudaKernel_diffractEncodedRS(Real* pc_data, Real* amp_data, const GpuConstERS* config, const int n_points_stream, Real* dst) {
	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	ulonglong tid_offset = blockDim.x * gridDim.x;
	ulonglong n_pixels = config->pn_X * config->pn_Y;

	for (tid; tid < n_pixels; tid += tid_offset) {
		int xxtr = tid % config->pn_X;
		int yytr = tid / config->pn_X;
		ulonglong idx = xxtr + yytr * config->pn_X;

		Real xxx = ((Real)xxtr + 0.5) * config->pp_X - config->half_ss_X;
		Real yyy = config->half_ss_Y - ((Real)yytr + 0.5) * config->pp_Y;
		Real interWav = xxx * config->sin_thetaX + yyy * config->sin_thetaY;

		for (int j = 0; j < n_points_stream; ++j) { //Create Fringe Pattern
			Real pcx = pc_data[3 * j + _X] * config->scale_X;
			Real pcy = pc_data[3 * j + _Y] * config->scale_Y;
			Real pcz = pc_data[3 * j + _Z] * config->scale_Z + config->offset_depth;

			Real r = sqrt((xxx - pcx) * (xxx - pcx) + (yyy - pcy) * (yyy - pcy) + (pcz * pcz));
			Real p = config->k * (r - interWav);
			Real res = amp_data[config->n_colors * j] * cos(p);

			*(dst + idx) += res;
		}
	}
	__syncthreads();
}


__global__ void cudaKernel_diffractNotEncodedRS(Real* pc_data, Real* amp_data, const GpuConstNERS* config, const int n_points_stream, Real* dst_real, Real* dst_imag) {
	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	ulonglong tid_offset = blockDim.x * gridDim.x;
	ulonglong n_pixels = config->pn_X * config->pn_Y;

	for (tid; tid < n_pixels; tid += tid_offset) {
		int xxtr = tid % config->pn_X;
		int yytr = tid / config->pn_X;
		ulonglong idx = xxtr + yytr * config->pn_X;

		Real xxx = -config->half_ss_X + (xxtr - 1) * config->pp_X;
		Real yyy = -config->half_ss_Y + (config->pn_Y - yytr) * config->pp_Y;

		for (int j = 0; j < n_points_stream; ++j) { //Create Fringe Pattern
			Real pcx = pc_data[3 * j + _X] * config->scale_X;
			Real pcy = pc_data[3 * j + _Y] * config->scale_Y;
			Real pcz = pc_data[3 * j + _Z] * config->scale_Z + config->offset_depth;
			Real amplitude = amp_data[config->n_colors * j];

			//boundary test
			Real abs_det_txy_pcz = abs(config->det_tx * pcz);
			Real _xbound[2] = {
				pcx + abs_det_txy_pcz,
				pcx - abs_det_txy_pcz
			};

			abs_det_txy_pcz = abs(config->det_ty * pcz);
			Real _ybound[2] = {
				pcy + abs_det_txy_pcz,
				pcy - abs_det_txy_pcz
			};

			Real Xbound[2] = {
				floor((_xbound[0] + config->half_ss_X) / config->pp_X) + 1,
				floor((_xbound[1] + config->half_ss_X) / config->pp_X) + 1
			};

			Real Ybound[2] = {
				config->pn_Y - floor((_ybound[1] + config->half_ss_Y) / config->pp_Y),
				config->pn_Y - floor((_ybound[0] + config->half_ss_Y) / config->pp_Y)
			};

			if (Xbound[0] > config->pn_X)	Xbound[0] = config->pn_X;
			if (Xbound[1] < 0)				Xbound[1] = 0;
			if (Ybound[0] > config->pn_Y)	Ybound[0] = config->pn_Y;
			if (Ybound[1] < 0)				Ybound[1] = 0;
			//

			if (((xxtr >= Xbound[1]) && (xxtr < Xbound[0])) && ((yytr >= Ybound[1]) && (yytr < Ybound[0]))) {
				Real xxx_pcx_sq = (xxx - pcx) * (xxx - pcx);
				Real yyy_pcy_sq = (yyy - pcy) * (yyy - pcy);
				Real pcz_sq = pcz * pcz;

				//range test
				Real abs_det_txy_sqrt = abs(config->det_tx * sqrt(yyy_pcy_sq + pcz_sq));
				Real range_x[2] = {
					pcx + abs_det_txy_sqrt,
					pcx - abs_det_txy_sqrt
				};

				abs_det_txy_sqrt = abs(config->det_ty * sqrt(xxx_pcx_sq + pcz_sq));
				Real range_y[2] = {
					pcy + abs_det_txy_sqrt,
					pcy - abs_det_txy_sqrt
				};
				//

				if (((xxx < range_x[0]) && (xxx > range_x[1])) && ((yyy < range_y[0]) && (yyy > range_y[1]))) {
					Real r = sqrt(xxx_pcx_sq + yyy_pcy_sq + pcz_sq);
					Real p = config->k * r;
					Real a = (amplitude * pcz) / (config->lambda * r * r);;
					Real res_real = sin(p) * a;
					Real res_imag = -cos(p) * a;

					*(dst_real + idx) += res_real;
					*(dst_imag + idx) += res_imag;
				}
			}
		}
	}
	__syncthreads();
}


__global__ void cudaKernel_diffractNotEncodedFrsn(Real* pc_data, Real* amp_data, const GpuConstNEFR* config, const int n_points_stream, Real* dst_real, Real* dst_imag) {
	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	ulonglong tid_offset = blockDim.x * gridDim.x;
	ulonglong n_pixels = config->pn_X * config->pn_Y;

	for (tid; tid < n_pixels; tid += tid_offset) {
		int xxtr = tid % config->pn_X;
		int yytr = tid / config->pn_X;
		ulonglong idx = xxtr + yytr * config->pn_X;

		Real xxx = -config->half_ss_X + (xxtr - 1) * config->pp_X;
		Real yyy = -config->half_ss_Y + (config->pn_Y - yytr) * config->pp_Y;

		for (int j = 0; j < n_points_stream; ++j) { //Create Fringe Pattern
			Real pcx = pc_data[3 * j + _X] * config->scale_X;
			Real pcy = pc_data[3 * j + _Y] * config->scale_Y;
			Real pcz = pc_data[3 * j + _Z] * config->scale_Z + config->offset_depth;
			Real amplitude = amp_data[config->n_colors * j];

			//boundary test
			Real abs_txy_pcz = abs(config->tx * pcz);
			Real _xbound[2] = {
				pcx + abs_txy_pcz,
				pcx - abs_txy_pcz
			};

			abs_txy_pcz = abs(config->ty * pcz);
			Real _ybound[2] = {
				pcy + abs_txy_pcz,
				pcy - abs_txy_pcz
			};

			Real Xbound[2] = {
				floor((_xbound[0] + config->half_ss_X) / config->pp_X) + 1,
				floor((_xbound[1] + config->half_ss_X) / config->pp_X) + 1
			};

			Real Ybound[2] = {
				config->pn_Y - floor((_ybound[1] + config->half_ss_Y) / config->pp_Y),
				config->pn_Y - floor((_ybound[0] + config->half_ss_Y) / config->pp_Y)
			};

			if (Xbound[0] > config->pn_X)	Xbound[0] = config->pn_X;
			if (Xbound[1] < 0)				Xbound[1] = 0;
			if (Ybound[0] > config->pn_Y)	Ybound[0] = config->pn_Y;
			if (Ybound[1] < 0)				Ybound[1] = 0;
			//

			if (((xxtr >= Xbound[1]) && (xxtr < Xbound[0])) && ((yytr >= Ybound[1]) && (yytr < Ybound[0]))) {
				Real p = config->k * ((xxx - pcx) * (xxx - pcx) + (yyy - pcy) * (yyy - pcy) + (2 * pcz * pcz)) / (2 * pcz);
				Real a = amplitude / (config->lambda * pcz);
				Real res_real = sin(p) * a;
				Real res_imag = -cos(p) * a;

				*(dst_real + idx) += res_real;
				*(dst_imag + idx) += res_imag;
			}
		}
	}
	__syncthreads();
}


extern "C"
{
	void cudaGenCghPointCloud_EncodedRS(
		const int &nBlocks, const int &nThreads, const int &n_pts_per_stream,
		Real* cuda_pc_data, Real* cuda_amp_data,
		Real* cuda_dst,
		const GpuConstERS* cuda_config)
	{
		cudaKernel_diffractEncodedRS << < nBlocks, nThreads >> > (cuda_pc_data, cuda_amp_data, cuda_config, n_pts_per_stream, cuda_dst);
	}

	void cudaGenCghPointCloud_NotEncodedRS(
		const int &nBlocks, const int &nThreads, const int &n_pts_per_stream,
		Real* cuda_pc_data, Real* cuda_amp_data,
		Real* cuda_dst_real, Real* cuda_dst_imag,
		const GpuConstNERS* cuda_config)
	{
		cudaKernel_diffractNotEncodedRS << < nBlocks, nThreads >> > (cuda_pc_data, cuda_amp_data, cuda_config, n_pts_per_stream, cuda_dst_real, cuda_dst_imag);
	}

	void cudaGenCghPointCloud_NotEncodedFrsn(
		const int &nBlocks, const int &nThreads, const int &n_pts_per_stream,
		Real* cuda_pc_data, Real* cuda_amp_data,
		Real* cuda_dst_real, Real* cuda_dst_imag,
		const GpuConstNEFR* cuda_config)
	{
		cudaKernel_diffractNotEncodedFrsn << < nBlocks, nThreads >> > (cuda_pc_data, cuda_amp_data, cuda_config, n_pts_per_stream, cuda_dst_real, cuda_dst_imag);
	}
}

#endif // !OphPCKernel_cu__