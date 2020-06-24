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

/*
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
*/

__global__ void cudaKernel_diffractNotEncodedRS(Real* pc_data, Real* amp_data, const GpuConstNERS* config, const int n_points_stream, Real* dst_real, Real* dst_imag, uint iChannel)
{

	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;

#ifdef FAST_MATH_WITH_FLOAT
	//extern __shared__ Real vertex[];
	//extern __shared__ Real amplitude[];
	__shared__ int pnX;
	__shared__ int pnY;
	__shared__ int nColor;
	__shared__ int loop;
	__shared__ float ppX;
	__shared__ float ppY;
	__shared__ float scaleX;
	__shared__ float scaleY;
	__shared__ float scaleZ;
	__shared__ float half_ssX;
	__shared__ float half_ssY;
	__shared__ float offsetDepth;
	__shared__ float det_tx;
	__shared__ float det_ty;
	__shared__ float k;
	__shared__ float lambda;


	if (threadIdx.x == 0) { // 64byte
		pnX = config->pn_X;
		pnY = config->pn_Y;
		ppX = config->pp_X;
		ppY = config->pp_Y;
		scaleX = config->scale_X;
		scaleY = config->scale_Y;
		scaleZ = config->scale_Z;
		half_ssX = config->half_ss_X;
		half_ssY = config->half_ss_Y;
		offsetDepth = config->offset_depth;
		det_tx = config->det_tx;
		det_ty = config->det_ty;
		nColor = config->n_colors;
		k = config->k;
		lambda = config->lambda;
		loop = n_points_stream;
	}
	__syncthreads();


	int xxtr = tid % pnX;
	int yytr = tid / pnX;
	ulonglong idx = xxtr + yytr * pnX;

	float xxx = __fadd_rz(-half_ssX, __fmul_rz(xxtr - 1, ppX)); // -half_ssX + (xxtr - 1) * ppX;
	float yyy = __fadd_rz(-half_ssY, __fmul_rz(pnY - yytr, ppY)); // -half_ssY + (pnY - yytr) * ppY;

	for (int j = 0; j < loop; ++j) { // Create Fringe Pattern
		int offset = 3 * j;

		float pcx = __fmul_rz(pc_data[offset + _X], scaleX);
		float pcy = __fmul_rz(pc_data[offset + _Y], scaleY);
		float pcz = __fmul_rz(pc_data[offset + _Z], scaleZ);
		pcz = __fadd_rz(pcz, offsetDepth);

		float amp = amp_data[nColor * j + iChannel];

		float abs_det_txy_pcz = abs(__fmul_rz(det_tx, pcz));

		float _xbound[2] = {
			pcx + abs_det_txy_pcz,
			pcx - abs_det_txy_pcz
		};

		abs_det_txy_pcz = abs(__fmul_rz(det_ty, pcz));

		float _ybound[2] = {
			pcy + abs_det_txy_pcz,
			pcy - abs_det_txy_pcz
		};

		float Xbound[2] = {
			floor(__fdividef((_xbound[_X] + half_ssX), ppX)) + 1,
			floor(__fdividef((_xbound[_Y] + half_ssX), ppX)) + 1
		};

		float Ybound[2] = {
			pnY - floor(__fdividef((_ybound[_Y] + half_ssY), ppY)),
			pnY - floor(__fdividef((_ybound[_X] + half_ssY), ppY))
		};

		if (Xbound[_X] > pnX)	Xbound[_X] = pnX;
		if (Xbound[_Y] < 0)		Xbound[_Y] = 0;
		if (Ybound[_X] > pnY)	Ybound[_X] = pnY;
		if (Ybound[_Y] < 0)		Ybound[_Y] = 0;

		if (((xxtr >= Xbound[_Y]) && (xxtr < Xbound[_X])) &&
			((yytr >= Ybound[_Y]) && (yytr < Ybound[_X]))) {

			float xxx_pcx_sq = __fmul_rz(xxx - pcx, xxx - pcx);
			float yyy_pcy_sq = __fmul_rz(yyy - pcy, yyy - pcy);
			float pcz_sq = __fmul_rz(pcz, pcz);

			// abs(det_tx * sqrt(yyy_pcy_sq + pcz_sq));
			float abs_det_txy_sqrt = abs(__fmul_rz(det_tx, __fsqrt_rz(__fadd_rz(yyy_pcy_sq, pcz_sq))));

			float range_x[2] = {
				pcx + abs_det_txy_sqrt,
				pcx - abs_det_txy_sqrt
			};

			abs_det_txy_sqrt = abs(__fmul_rz(det_ty, __fsqrt_rz(__fadd_rz(xxx_pcx_sq, pcz_sq))));

			float range_y[2] = {
				pcy + abs_det_txy_sqrt,
				pcy - abs_det_txy_sqrt
			};

			if (amp != 0.0f && 
				((xxx < range_x[_X]) && (xxx > range_x[_Y])) && 
				((yyy < range_y[_X]) && (yyy > range_y[_Y]))) {
#if 0
				float orir = __fadd_rz(__fadd_rz(xxx_pcx_sq, yyy_pcy_sq), pcz_sq);
				float r = __fsqrt_rz(orir);
				float p = __fmul_rz(k, r);
				float a = __fdividef(__fmul_rz(amp, pcz), __fmul_rz(lambda, orir));
#else
				float r = __fsqrt_rz(__fadd_rz(__fadd_rz(xxx_pcx_sq, yyy_pcy_sq), pcz_sq));
				float p = __fmul_rz(k, r);
				float a = __fdividef(__fmul_rz(amp, pcz), __fmul_rz(lambda, __fmul_rz(r, r)));
#endif
				float res_real = __fmul_rz(__sinf(p), a);
				float res_imag = __fmul_rz(-__cosf(p), a);

				*(dst_real + idx) += res_real;
				*(dst_imag + idx) += res_imag;
			}
		}
	}

#else

	int xxtr = tid % config->pn_X;
	int yytr = tid / config->pn_X;
	ulonglong idx = xxtr + yytr * config->pn_X;

	Real xxx = -config->half_ss_X + (xxtr - 1) * config->pp_X;
	Real yyy = -config->half_ss_Y + (config->pn_Y - yytr) * config->pp_Y;

	for (int j = 0; j < n_points_stream; ++j) { // Create Fringe Pattern
		int k = 3 * j;
		Real pcx = pc_data[k + _X] * config->scale_X;
		Real pcy = pc_data[k + _Y] * config->scale_Y;
		Real pcz = pc_data[k + _Z] * config->scale_Z;
		pcz += config->offset_depth;

		Real amplitude = amp_data[config->n_colors * j + iChannel];			//boundary test
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
			floor((_xbound[_X] + config->half_ss_X) / config->pp_X) + 1,
			floor((_xbound[_Y] + config->half_ss_X) / config->pp_X) + 1
		};

		Real Ybound[2] = {
			config->pn_Y - floor((_ybound[_Y] + config->half_ss_Y) / config->pp_Y),
			config->pn_Y - floor((_ybound[_X] + config->half_ss_Y) / config->pp_Y)
		};

		if (Xbound[_X] > config->pn_X)	Xbound[_X] = config->pn_X;
		if (Xbound[_Y] < 0)				Xbound[_Y] = 0;
		if (Ybound[_X] > config->pn_Y)	Ybound[_X] = config->pn_Y;
		if (Ybound[_Y] < 0)				Ybound[_Y] = 0;

		if (((xxtr >= Xbound[_Y]) && (xxtr < Xbound[_X])) && ((yytr >= Ybound[_Y]) && (yytr < Ybound[_X]))) {
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

			if (amplitude != 0.0 && ((xxx < range_x[_X]) && (xxx > range_x[_Y])) && ((yyy < range_y[_X]) && (yyy > range_y[_Y]))) {
				Real r = sqrt(xxx_pcx_sq + yyy_pcy_sq + pcz_sq);
				Real p = config->k * r;
				Real a = (amplitude * pcz) / (config->lambda * r * r);

				Real res_real = sin(p) * a;
				Real res_imag = -cos(p) * a;
				*(dst_real + idx) += res_real;
				*(dst_imag + idx) += res_imag;
			}
		}
	}
#endif
}


__global__ void cudaKernel_diffractNotEncodedFrsn(Real* pc_data, Real* amp_data, const GpuConstNEFR* config, const int n_points_stream, Real* dst_real, Real* dst_imag, uint iChannel)
{
	__shared__ int pnX;
	__shared__ int pnY;
	__shared__ int nColor;
	__shared__ int loop;
	__shared__ float ppX;
	__shared__ float ppY;
	__shared__ float scaleX;
	__shared__ float scaleY;
	__shared__ float scaleZ;
	__shared__ float half_ssX;
	__shared__ float half_ssY;
	__shared__ float offsetDepth;
	__shared__ float tx;
	__shared__ float ty;
	__shared__ float k;
	__shared__ float lambda;

	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIdx.x == 0) {
		pnX = config->pn_X;
		pnY = config->pn_Y;
		ppX = config->pp_X;
		ppY = config->pp_Y;
		scaleX = config->scale_X;
		scaleY = config->scale_Y;
		scaleZ = config->scale_Z;
		half_ssX = config->half_ss_X;
		half_ssY = config->half_ss_Y;
		offsetDepth = config->offset_depth;
		tx = config->tx;
		ty = config->ty;
		nColor = config->n_colors;
		k = config->k;
		lambda = config->lambda;
		loop = n_points_stream;

	}
	__syncthreads();

	int xxtr = tid % pnX;
	int yytr = tid / pnX;
	ulonglong idx = xxtr + yytr * pnX;
#ifdef FAST_MATH_WITH_FLOAT 
	float xxx = __fadd_rz(-half_ssX, __fmul_rz((xxtr - 1), ppX));
	float yyy = __fadd_rz(-half_ssY, __fmul_rz((pnY - yytr), ppY));

	for (int j = 0; j < loop; ++j) { //Create Fringe Pattern
		int offset = 3 * j;
		float pcx = __fmul_rz(pc_data[offset + _X], scaleX);
		float pcy = __fmul_rz(pc_data[offset + _Y], scaleY);
		float pcz = __fmul_rz(pc_data[offset + _Z], scaleZ);
		pcz = __fadd_rz(pcz, offsetDepth);
		float amplitude = amp_data[nColor * j + iChannel];

		//boundary test
		float abs_txy_pcz = abs(__fmul_rz(tx, pcz));
		float _xbound[2] = {
			pcx + abs_txy_pcz,
			pcx - abs_txy_pcz
		};

		abs_txy_pcz = abs(__fmul_rz(ty, pcz));
		float _ybound[2] = {
			pcy + abs_txy_pcz,
			pcy - abs_txy_pcz
		};

		float Xbound[2] = {
			floor(__fdividef(__fadd_rz(_xbound[_X], half_ssX), ppX)) + 1,
			floor(__fdividef(__fadd_rz(_xbound[_Y], half_ssX), ppX)) + 1
		};

		float Ybound[2] = {
			pnY - floor(__fdividef(__fadd_rz(_ybound[_Y], half_ssY), ppY)),
			pnY - floor(__fdividef(__fadd_rz(_ybound[_X], half_ssY), ppY))
		};

		if (Xbound[_X] > pnX)	Xbound[_X] = pnX;
		if (Xbound[_Y] < 0)				Xbound[_Y] = 0;
		if (Ybound[_X] > pnY)	Ybound[_X] = pnY;
		if (Ybound[_Y] < 0)				Ybound[_Y] = 0;
		//

		if (((xxtr >= Xbound[_Y]) && (xxtr < Xbound[_X])) && ((yytr >= Ybound[_Y]) && (yytr < Ybound[_X]))) {
			float p = __fdividef(__fmul_rz(k, __fadd_rz(__fadd_rz(__fmul_rz((xxx - pcx), (xxx - pcx)), __fmul_rz((yyy - pcy), (yyy - pcy))),
				__fmul_rz(2, __fmul_rz(pcz, pcz)))), __fmul_rz(2, pcz));
			float a = __fdividef(amplitude, __fmul_rz(lambda, pcz));
			float res_real = __fmul_rz(__sinf(p), a);
			float res_imag = __fmul_rz(-__cosf(p), a);

			*(dst_real + idx) += res_real;
			*(dst_imag + idx) += res_imag;
		}
	}
#else
	Real xxx = -config->half_ss_X + (xxtr - 1) * config->pp_X;
	Real yyy = -config->half_ss_Y + (config->pn_Y - yytr) * config->pp_Y;

	for (int j = 0; j < n_points_stream; ++j) { //Create Fringe Pattern
		Real pcx = pc_data[3 * j + _X] * config->scale_X;
		Real pcy = pc_data[3 * j + _Y] * config->scale_Y;
		Real pcz = pc_data[3 * j + _Z] * config->scale_Z + config->offset_depth;
		Real amplitude = amp_data[config->n_colors * j + iChannel];

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
			floor((_xbound[_X] + config->half_ss_X) / config->pp_X) + 1,
			floor((_xbound[_Y] + config->half_ss_X) / config->pp_X) + 1
		};

		Real Ybound[2] = {
			config->pn_Y - floor((_ybound[_Y] + config->half_ss_Y) / config->pp_Y),
			config->pn_Y - floor((_ybound[_X] + config->half_ss_Y) / config->pp_Y)
		};

		if (Xbound[_X] > config->pn_X)	Xbound[_X] = config->pn_X;
		if (Xbound[_Y] < 0)				Xbound[_Y] = 0;
		if (Ybound[_X] > config->pn_Y)	Ybound[_X] = config->pn_Y;
		if (Ybound[_Y] < 0)				Ybound[_Y] = 0;
		//

		if (((xxtr >= Xbound[_Y]) && (xxtr < Xbound[_X])) && ((yytr >= Ybound[_Y]) && (yytr < Ybound[_X]))) {
			Real p = config->k * ((xxx - pcx) * (xxx - pcx) + (yyy - pcy) * (yyy - pcy) + (2 * pcz * pcz)) / (2 * pcz);
			Real a = amplitude / (config->lambda * pcz);
			Real res_real = sin(p) * a;
			Real res_imag = -cos(p) * a;

			*(dst_real + idx) += res_real;
			*(dst_imag + idx) += res_imag;
		}
	}
#endif
}


extern "C"
{
	/*
	void cudaGenCghPointCloud_EncodedRS(
		const int &nBlocks, const int &nThreads, const int &n_pts_per_stream,
		Real* cuda_pc_data, Real* cuda_amp_data,
		Real* cuda_dst,
		const GpuConstERS* cuda_config)
	{
		cudaKernel_diffractEncodedRS << < nBlocks, nThreads >> > (cuda_pc_data, cuda_amp_data, cuda_config, n_pts_per_stream, cuda_dst);
	}
	*/
	void cudaGenCghPointCloud_NotEncodedRS(
		const int &nBlocks, const int &nThreads, const int &n_pts_per_stream,
		Real* cuda_pc_data, Real* cuda_amp_data,
		Real* cuda_dst_real, Real* cuda_dst_imag,
		const GpuConstNERS* cuda_config, const uint &iChannel)
	{
		//cudaMemcpyToSymbolAsync(&nPoints, &n_pts_per_stream, sizeof(int));
		//int size = sizeof(Real) * n_pts_per_stream * 3;
		//size += sizeof(Real) * n_pts_per_stream * 3;
		//size += sizeof(int) * 4;
		//size += sizeof(float) * 12;		
		cudaKernel_diffractNotEncodedRS << < nBlocks, nThreads/*, size*/ >> > (cuda_pc_data, cuda_amp_data, cuda_config, n_pts_per_stream, cuda_dst_real, cuda_dst_imag, iChannel);
	}

	void cudaGenCghPointCloud_NotEncodedFrsn(
		const int &nBlocks, const int &nThreads, const int &n_pts_per_stream,
		Real* cuda_pc_data, Real* cuda_amp_data,
		Real* cuda_dst_real, Real* cuda_dst_imag,
		const GpuConstNEFR* cuda_config, const uint &iChannel)
	{
		cudaKernel_diffractNotEncodedFrsn << < nBlocks, nThreads >> > (cuda_pc_data, cuda_amp_data, cuda_config, n_pts_per_stream, cuda_dst_real, cuda_dst_imag, iChannel);
	}
}

#endif // !OphPCKernel_cu__