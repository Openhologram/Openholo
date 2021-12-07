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

__global__ void cudaKernel_diffractNotEncodedRS(double* pc_data, double* amp_data, const GpuConstNERS* config, const int n_points_stream, double* dst_real, double* dst_imag, uint iChannel, uint mode)
{
	__shared__ int pnX;
	__shared__ int pnY;
	__shared__ int nColor;
	__shared__ int loop;

	if (mode & MODE_FLOAT)
	{
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
		//	__shared__ float 

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
			det_tx = config->det_tx;
			det_ty = config->det_ty;
			nColor = config->n_colors;
			k = config->k;
			lambda = config->lambda;
			loop = n_points_stream;
		}
		__syncthreads();


		ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
		int xxtr = tid % pnX;
		int yytr = tid / pnX;
		ulonglong idx = xxtr + yytr * pnX;

		if (mode & MODE_FASTMATH)
		{
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

					if (((xxx < range_x[_X]) && (xxx > range_x[_Y])) &&
						((yyy < range_y[_X]) && (yyy > range_y[_Y]))) {

						float r = __fsqrt_rz(__fadd_rz(__fadd_rz(xxx_pcx_sq, yyy_pcy_sq), pcz_sq));
						float p = __fmul_rz(k, r);
						float a = __fdividef(__fmul_rz(amp, pcz), __fmul_rz(lambda, __fmul_rz(r, r)));
						float res_real = __fmul_rz(__sinf(p), a);
						float res_imag = __fmul_rz(-__cosf(p), a);

						dst_real[idx] += res_real;
						dst_imag[idx] += res_imag;
					}
				}
			}
		}
		else
		{
			float xxx = -half_ssX + (xxtr - 1) * ppX;
			float yyy = -half_ssY + (pnY - yytr) * ppY;

			for (int j = 0; j < loop; ++j) { // Create Fringe Pattern
				int offset = 3 * j;

				float pcx = pc_data[offset + _X] * scaleX;
				float pcy = pc_data[offset + _Y] * scaleY;
				float pcz = pc_data[offset + _Z] * scaleZ;

				pcz = pcz + offsetDepth;

				float amp = amp_data[nColor * j + iChannel];

				float abs_det_txy_pcz = abs(det_tx * pcz);

				float _xbound[2] = {
					pcx + abs_det_txy_pcz,
					pcx - abs_det_txy_pcz
				};

				abs_det_txy_pcz = abs(det_ty * pcz);

				float _ybound[2] = {
					pcy + abs_det_txy_pcz,
					pcy - abs_det_txy_pcz
				};

				float Xbound[2] = {
					floor((_xbound[_X] + half_ssX) / ppX) + 1,
					floor((_xbound[_Y] + half_ssX) / ppX) + 1
				};

				float Ybound[2] = {
					pnY - floor((_ybound[_Y] + half_ssY) / ppY),
					pnY - floor((_ybound[_X] + half_ssY) / ppY)
				};

				if (Xbound[_X] > pnX)	Xbound[_X] = pnX;
				if (Xbound[_Y] < 0)		Xbound[_Y] = 0;
				if (Ybound[_X] > pnY)	Ybound[_X] = pnY;
				if (Ybound[_Y] < 0)		Ybound[_Y] = 0;

				if (((xxtr >= Xbound[_Y]) && (xxtr < Xbound[_X])) &&
					((yytr >= Ybound[_Y]) && (yytr < Ybound[_X]))) {

					float xxx_pcx_sq = (xxx - pcx) * (xxx - pcx);
					float yyy_pcy_sq = (yyy - pcy) * (yyy - pcy);
					float pcz_sq = pcz * pcz;

					float abs_det_txy_sqrt = abs(det_tx * sqrt(yyy_pcy_sq + pcz_sq));

					float range_x[2] = {
						pcx + abs_det_txy_sqrt,
						pcx - abs_det_txy_sqrt
					};

					abs_det_txy_sqrt = abs(det_ty * sqrt(xxx_pcx_sq + pcz_sq));

					float range_y[2] = {
						pcy + abs_det_txy_sqrt,
						pcy - abs_det_txy_sqrt
					};

					if (((xxx < range_x[_X]) && (xxx > range_x[_Y])) &&
						((yyy < range_y[_X]) && (yyy > range_y[_Y]))) {

						float r = sqrt(xxx_pcx_sq + yyy_pcy_sq + pcz_sq);
						float p = k * r;
						float a = (amp * pcz) / (lambda * (r * r));
						float res_real = sinf(p) * a;
						float res_imag = -cosf(p) * a;

						dst_real[idx] += res_real;
						dst_imag[idx] += res_imag;
					}
				}
			}
		}
	}
	else
	{
		__shared__ double ppX;
		__shared__ double ppY;
		__shared__ double scaleX;
		__shared__ double scaleY;
		__shared__ double scaleZ;
		__shared__ double half_ssX;
		__shared__ double half_ssY;
		__shared__ double offsetDepth;
		__shared__ double det_tx;
		__shared__ double det_ty;
		__shared__ double k;
		__shared__ double lambda;

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
			det_tx = config->det_tx;
			det_ty = config->det_ty;
			nColor = config->n_colors;
			k = config->k;
			lambda = config->lambda;
			loop = n_points_stream;
		}
		__syncthreads();
		

		ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
		int xxtr = tid % pnX;
		int yytr = tid / pnX;
		ulonglong idx = xxtr + yytr * pnX;

		if (mode & MODE_FASTMATH)
		{
			double xxx = __dadd_rz(-half_ssX, __dmul_rz(xxtr - 1, ppX)); // -half_ssX + (xxtr - 1) * ppX;
			double yyy = __dadd_rz(-half_ssY, __dmul_rz(pnY - yytr, ppY)); // -half_ssY + (pnY - yytr) * ppY;

			for (int j = 0; j < loop; ++j) { // Create Fringe Pattern
				int offset = 3 * j;

				double pcx = __dmul_rz(pc_data[offset + _X], scaleX);
				double pcy = __dmul_rz(pc_data[offset + _Y], scaleY);
				double pcz = __dmul_rz(pc_data[offset + _Z], scaleZ);

				pcz = __dadd_rz(pcz, offsetDepth);

				double amp = amp_data[nColor * j + iChannel];

				double abs_det_txy_pcz = abs(__dmul_rz(det_tx, pcz));

				double _xbound[2] = {
					pcx + abs_det_txy_pcz,
					pcx - abs_det_txy_pcz
				};

				abs_det_txy_pcz = abs(__dmul_rz(det_ty, pcz));

				double _ybound[2] = {
					pcy + abs_det_txy_pcz,
					pcy - abs_det_txy_pcz
				};

				double Xbound[2] = {
					floor(__ddiv_rz((_xbound[_X] + half_ssX), ppX)) + 1,
					floor(__ddiv_rz((_xbound[_Y] + half_ssX), ppX)) + 1
				};

				double Ybound[2] = {
					pnY - floor(__ddiv_rz((_ybound[_Y] + half_ssY), ppY)),
					pnY - floor(__ddiv_rz((_ybound[_X] + half_ssY), ppY))
				};

				if (Xbound[_X] > pnX)	Xbound[_X] = pnX;
				if (Xbound[_Y] < 0)		Xbound[_Y] = 0;
				if (Ybound[_X] > pnY)	Ybound[_X] = pnY;
				if (Ybound[_Y] < 0)		Ybound[_Y] = 0;

				if (((xxtr >= Xbound[_Y]) && (xxtr < Xbound[_X])) &&
					((yytr >= Ybound[_Y]) && (yytr < Ybound[_X]))) {

					double xxx_pcx_sq = __dmul_rz(xxx - pcx, xxx - pcx);
					double yyy_pcy_sq = __dmul_rz(yyy - pcy, yyy - pcy);
					double pcz_sq = __dmul_rz(pcz, pcz);

					// abs(det_tx * sqrt(yyy_pcy_sq + pcz_sq));
					double abs_det_txy_sqrt = abs(__dmul_rz(det_tx, __dsqrt_rz(__dadd_rz(yyy_pcy_sq, pcz_sq))));

					double range_x[2] = {
						pcx + abs_det_txy_sqrt,
						pcx - abs_det_txy_sqrt
					};

					abs_det_txy_sqrt = abs(__dmul_rz(det_ty, __dsqrt_rz(__dadd_rz(xxx_pcx_sq, pcz_sq))));

					double range_y[2] = {
						pcy + abs_det_txy_sqrt,
						pcy - abs_det_txy_sqrt
					};

					if (((xxx < range_x[_X]) && (xxx > range_x[_Y])) &&
						((yyy < range_y[_X]) && (yyy > range_y[_Y]))) {

						double r = __dsqrt_rz(__dadd_rz(__dadd_rz(xxx_pcx_sq, yyy_pcy_sq), pcz_sq));
						double p = __dmul_rz(k, r);
						double a = __ddiv_rz(__dmul_rz(amp, pcz), __dmul_rz(lambda, __dmul_rz(r, r)));
						double res_real = __dmul_rz(sin(p), a);
						double res_imag = __dmul_rz(-cos(p), a);

						*(dst_real + idx) += res_real;
						*(dst_imag + idx) += res_imag;
					}
				}
			}
		}
		else
		{
			double xxx = -half_ssX + (xxtr - 1) * ppX;
			double yyy = -half_ssY + (pnY - yytr) * ppY;

			for (int j = 0; j < loop; ++j) { // Create Fringe Pattern
				int offset = 3 * j;

				double pcx = pc_data[offset + _X] * scaleX;
				double pcy = pc_data[offset + _Y] * scaleY;
				double pcz = pc_data[offset + _Z] * scaleZ;

				pcz = pcz + offsetDepth;

				double amp = amp_data[nColor * j + iChannel];

				double abs_det_txy_pcz = abs(det_tx * pcz);

				double _xbound[2] = {
					pcx + abs_det_txy_pcz,
					pcx - abs_det_txy_pcz
				};

				abs_det_txy_pcz = abs(det_ty * pcz);

				double _ybound[2] = {
					pcy + abs_det_txy_pcz,
					pcy - abs_det_txy_pcz
				};

				double Xbound[2] = {
					floor((_xbound[_X] + half_ssX) / ppX) + 1,
					floor((_xbound[_Y] + half_ssX) / ppX) + 1
				};

				double Ybound[2] = {
					pnY - floor((_ybound[_Y] + half_ssY) / ppY),
					pnY - floor((_ybound[_X] + half_ssY) / ppY)
				};

				if (Xbound[_X] > pnX)	Xbound[_X] = pnX;
				if (Xbound[_Y] < 0)		Xbound[_Y] = 0;
				if (Ybound[_X] > pnY)	Ybound[_X] = pnY;
				if (Ybound[_Y] < 0)		Ybound[_Y] = 0;

				if (((xxtr >= Xbound[_Y]) && (xxtr < Xbound[_X])) &&
					((yytr >= Ybound[_Y]) && (yytr < Ybound[_X]))) {

					double xxx_pcx_sq = (xxx - pcx) * (xxx - pcx);
					double yyy_pcy_sq = (yyy - pcy) * (yyy - pcy);
					double pcz_sq = pcz * pcz;

					// abs(det_tx * sqrt(yyy_pcy_sq + pcz_sq));
					double abs_det_txy_sqrt = abs(det_tx * sqrt(yyy_pcy_sq + pcz_sq));

					double range_x[2] = {
						pcx + abs_det_txy_sqrt,
						pcx - abs_det_txy_sqrt
					};

					abs_det_txy_sqrt = abs(det_ty * sqrt(xxx_pcx_sq + pcz_sq));

					double range_y[2] = {
						pcy + abs_det_txy_sqrt,
						pcy - abs_det_txy_sqrt
					};

					if (((xxx < range_x[_X]) && (xxx > range_x[_Y])) &&
						((yyy < range_y[_X]) && (yyy > range_y[_Y]))) {

						double r = sqrt(xxx_pcx_sq + yyy_pcy_sq + pcz_sq);
						double p = k * r;
						double a = (amp * pcz) / (lambda * (r * r));
						double res_real = sin(p) * a;
						double res_imag = -cos(p) * a;

						dst_real[idx] += res_real;
						dst_imag[idx] += res_imag;
					}
				}
			}
		}
	}
}

/*
__global__ void cudaKernel_diffractNotEncodedFrsn(double* pc_data, double* amp_data, const GpuConstNEFR* config, const int n_points_stream, double* dst_real, double* dst_imag, uint iChannel)
{
	double xxx = -config->half_ss_X + (xxtr - 1) * config->pp_X;
	double yyy = -config->half_ss_Y + (config->pn_Y - yytr) * config->pp_Y;

	for (int j = 0; j < n_points_stream; ++j) { //Create Fringe Pattern
		double pcx = pc_data[3 * j + _X] * config->scale_X;
		double pcy = pc_data[3 * j + _Y] * config->scale_Y;
		double pcz = pc_data[3 * j + _Z] * config->scale_Z + config->offset_depth;
		double amplitude = amp_data[config->n_colors * j + iChannel];

		//boundary test
		double abs_txy_pcz = abs(config->tx * pcz);
		double _xbound[2] = {
			pcx + abs_txy_pcz,
			pcx - abs_txy_pcz
		};

		abs_txy_pcz = abs(config->ty * pcz);
		double _ybound[2] = {
			pcy + abs_txy_pcz,
			pcy - abs_txy_pcz
		};

		double Xbound[2] = {
			floor((_xbound[_X] + config->half_ss_X) / config->pp_X) + 1,
			floor((_xbound[_Y] + config->half_ss_X) / config->pp_X) + 1
		};

		double Ybound[2] = {
			config->pn_Y - floor((_ybound[_Y] + config->half_ss_Y) / config->pp_Y),
			config->pn_Y - floor((_ybound[_X] + config->half_ss_Y) / config->pp_Y)
		};

		if (Xbound[_X] > config->pn_X)	Xbound[_X] = config->pn_X;
		if (Xbound[_Y] < 0)				Xbound[_Y] = 0;
		if (Ybound[_X] > config->pn_Y)	Ybound[_X] = config->pn_Y;
		if (Ybound[_Y] < 0)				Ybound[_Y] = 0;
		//

		if (((xxtr >= Xbound[_Y]) && (xxtr < Xbound[_X])) && ((yytr >= Ybound[_Y]) && (yytr < Ybound[_X]))) {
			double p = config->k * ((xxx - pcx) * (xxx - pcx) + (yyy - pcy) * (yyy - pcy) + (2 * pcz * pcz)) / (2 * pcz);
			double a = amplitude / (config->lambda * pcz);
			double res_real = sin(p) * a;
			double res_imag = -cos(p) * a;

			*(dst_real + idx) += res_real;
			*(dst_imag + idx) += res_imag;
		}
	}
}
*/

__global__ void cudaKernel_diffractNotEncodedFrsn(double* pc_data, double* amp_data, const GpuConstNEFR* config, const int n_points_stream, double* dst_real, double* dst_imag, uint iChannel)
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

	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	int xxtr = tid % pnX;
	int yytr = tid / pnX;
	ulonglong idx = xxtr + yytr * pnX;

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
}


extern "C"
{
	void cudaGenCghPointCloud_NotEncodedRS(
		const int &nBlocks, const int &nThreads, const int &n_pts_per_stream,
		double* cuda_pc_data, double* cuda_amp_data,
		double* cuda_dst_real, double* cuda_dst_imag,
		const GpuConstNERS* cuda_config, const uint &iChannel, const uint &mode)
	{
		cudaKernel_diffractNotEncodedRS << < nBlocks, nThreads >> > (cuda_pc_data, cuda_amp_data, cuda_config, n_pts_per_stream, cuda_dst_real, cuda_dst_imag, iChannel, mode);
	}

	void cudaGenCghPointCloud_NotEncodedFrsn(
		const int &nBlocks, const int &nThreads, const int &n_pts_per_stream,
		double* cuda_pc_data, double* cuda_amp_data,
		double* cuda_dst_real, double* cuda_dst_imag,
		const GpuConstNEFR* cuda_config, const uint &iChannel)
	{
		cudaKernel_diffractNotEncodedFrsn << < nBlocks, nThreads >> > (cuda_pc_data, cuda_amp_data, cuda_config, n_pts_per_stream, cuda_dst_real, cuda_dst_imag, iChannel);
	}
}

#endif // !OphPCKernel_cu__