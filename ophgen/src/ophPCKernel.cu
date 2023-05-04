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
* @author	Hyeong-Hak Ahn, Minwoo Nam
* @date		2018/09
*/

#ifndef OphPCKernel_cu__
#define OphPCKernel_cu__

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <device_launch_parameters.h>
#include <device_functions.h>
#include <math_functions.h>
#include "typedef.h"
#include "ophPointCloud_GPU.h"

__global__
void cudaKernel_double_RS_Diffraction(uint channel, Vertex* vertex_data, const GpuConstNERS* config, const int N, cuDoubleComplex* dst)
{
	__shared__ int pnX, pnY, loop, schannel, offsetX, offsetY;
	__shared__ double ppX, ppY, scaleX, scaleY, scaleZ, half_ssX, half_ssY, distance, det_tx, det_ty, k, lambda;

	if (threadIdx.x == 0) {
		schannel = channel;
		pnX = config->pn_X;
		pnY = config->pn_Y;
		ppX = config->pp_X;
		ppY = config->pp_Y;
		offsetX = config->offset_X;
		offsetY = config->offset_Y;
		scaleX = config->scale_X;
		scaleY = config->scale_Y;
		scaleZ = config->scale_Z;
		half_ssX = config->half_ss_X;
		half_ssY = config->half_ss_Y;
		distance = config->offset_depth;
		det_tx = config->det_tx;
		det_ty = config->det_ty;
		k = config->k;
		lambda = config->lambda;
		loop = N;
	}
	__syncthreads();

	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	int xxtr = tid % pnX;
	int yytr = tid / pnX;
	ulonglong idx = xxtr + yytr * pnX;

	double xxx = -half_ssX + (xxtr - 1 + offsetX) * ppX;
	double yyy = -half_ssY + (pnY - yytr + offsetY) * ppY;

	for (int j = 0; j < loop; ++j)
	{ 
		// Create Fringe Pattern
		double pcx = vertex_data[j].point.pos[_X] * scaleX;
		double pcy = vertex_data[j].point.pos[_Y] * scaleY;
		double pcz = vertex_data[j].point.pos[_Z] * scaleZ;

		pcz = pcz + distance;

		double amp = vertex_data[j].color.color[schannel];

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
				dst[idx].x += res_real;
				dst[idx].y += res_imag;
			}
		}
	}
}

__global__
void cudaKernel_double_FastMath_RS_Diffraction(uint channel, Vertex* vertex_data, const GpuConstNERS* config, const int N, cuDoubleComplex* dst)
{
	__shared__ int pnX, pnY, loop, schannel, offsetX, offsetY;
	__shared__ double ppX, ppY, scaleX, scaleY, scaleZ, half_ssX, half_ssY, distance, det_tx, det_ty, k, lambda;

	if (threadIdx.x == 0) {
		schannel = channel;
		pnX = config->pn_X;
		pnY = config->pn_Y;
		ppX = config->pp_X;
		ppY = config->pp_Y;
		offsetX = config->offset_X;
		offsetY = config->offset_Y;
		scaleX = config->scale_X;
		scaleY = config->scale_Y;
		scaleZ = config->scale_Z;
		half_ssX = config->half_ss_X;
		half_ssY = config->half_ss_Y;
		distance = config->offset_depth;
		det_tx = config->det_tx;
		det_ty = config->det_ty;
		k = config->k;
		lambda = config->lambda;
		loop = N;
	}
	__syncthreads();

	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	int xxtr = tid % pnX;
	int yytr = tid / pnX;
	ulonglong idx = xxtr + yytr * pnX;

	double xxx = __dadd_rz(-half_ssX, __dmul_rz(__dadd_rz(__dsub_rz(xxtr, 1), offsetX), ppX));
	double yyy = __dadd_rz(-half_ssY, __dmul_rz(__dadd_rz(__dsub_rz(pnY, yytr), offsetY), ppY));

	for (int j = 0; j < loop; ++j)
	{ // Create Fringe Pattern
		double pcx = __dmul_rz(vertex_data[j].point.pos[_X], scaleX);
		double pcy = __dmul_rz(vertex_data[j].point.pos[_Y], scaleY);
		double pcz = __dmul_rz(vertex_data[j].point.pos[_Z], scaleZ);

		pcz = __dadd_rz(pcz, distance);

		double amp = vertex_data[j].color.color[schannel];

		double abs_det_txy_pcz = abs(__dmul_rz(det_tx, pcz));

		double _xbound[2] = {
			__dadd_rz(pcx, abs_det_txy_pcz),
			__dsub_rz(pcx, abs_det_txy_pcz)
		};

		abs_det_txy_pcz = abs(__dmul_rz(det_ty, pcz));

		double _ybound[2] = {
			__dadd_rz(pcy, abs_det_txy_pcz),
			__dsub_rz(pcy, abs_det_txy_pcz)
		};

		double Xbound[2] = {
			__dadd_rz(floor(__ddiv_rz(__dadd_rz(_xbound[_X], half_ssX), ppX)), 1),
			__dadd_rz(floor(__ddiv_rz(__dadd_rz(_xbound[_Y], half_ssX), ppX)), 1)
		};

		double Ybound[2] = {
			__dsub_rz(pnY, floor(__ddiv_rz(__dadd_rz(_ybound[_Y], half_ssY), ppY))),
			__dsub_rz(pnY, floor(__ddiv_rz(__dadd_rz(_ybound[_X], half_ssY), ppY)))
		};

		if (Xbound[_X] > pnX)	Xbound[_X] = pnX;
		if (Xbound[_Y] < 0)		Xbound[_Y] = 0;
		if (Ybound[_X] > pnY)	Ybound[_X] = pnY;
		if (Ybound[_Y] < 0)		Ybound[_Y] = 0;

		if (((xxtr >= Xbound[_Y]) && (xxtr < Xbound[_X])) &&
			((yytr >= Ybound[_Y]) && (yytr < Ybound[_X]))) {
			double xx = __dsub_rz(xxx, pcx);
			double yy = __dsub_rz(yyy, pcy);

			double xxx_pcx_sq = __dmul_rz(xx, xx);
			double yyy_pcy_sq = __dmul_rz(yy, yy);
			double pcz_sq = __dmul_rz(pcz, pcz);

			double abs_det_txy_sqrt = abs(__dmul_rz(det_tx, __dsqrt_rz(__dadd_rz(yyy_pcy_sq, pcz_sq))));

			double range_x[2] = {
				__dadd_rz(pcx, abs_det_txy_sqrt),
				__dsub_rz(pcx, abs_det_txy_sqrt)
			};

			abs_det_txy_sqrt = abs(__dmul_rz(det_ty, __dsqrt_rz(__dadd_rz(xxx_pcx_sq, pcz_sq))));

			double range_y[2] = {
				__dadd_rz(pcy, abs_det_txy_sqrt),
				__dsub_rz(pcy, abs_det_txy_sqrt)
			};

			if (((xxx < range_x[_X]) && (xxx > range_x[_Y])) &&
				((yyy < range_y[_X]) && (yyy > range_y[_Y]))) {

				double r = __dsqrt_rz(__dadd_rz(__dadd_rz(xxx_pcx_sq, yyy_pcy_sq), pcz_sq));
				double p = __dmul_rz(k, r);
				double a = __ddiv_rz(__dmul_rz(amp, pcz), __dmul_rz(lambda, __dmul_rz(r, r)));
				double res_real = __dmul_rz(sin(p), a);
				double res_imag = __dmul_rz(-cos(p), a);
				dst[idx].x = __dadd_rz(dst[idx].x, res_real);
				dst[idx].y = __dadd_rz(dst[idx].y, res_imag);
			}
		}
	}
}

__global__
void cudaKernel_single_RS_Diffraction(uint channel, Vertex* vertex_data, const GpuConstNERS* config, const int N, cuDoubleComplex* dst)
{
	__shared__ int pnX, pnY, loop, schannel, offsetX, offsetY;
	__shared__ float ppX, ppY, scaleX, scaleY, scaleZ, half_ssX, half_ssY, distance, det_tx, det_ty, k, lambda;

	if (threadIdx.x == 0) {
		schannel = channel;
		pnX = config->pn_X;
		pnY = config->pn_Y;
		ppX = config->pp_X;
		ppY = config->pp_Y;
		offsetX = config->offset_X;
		offsetY = config->offset_Y;
		scaleX = config->scale_X;
		scaleY = config->scale_Y;
		scaleZ = config->scale_Z;
		half_ssX = config->half_ss_X;
		half_ssY = config->half_ss_Y;
		distance = config->offset_depth;
		det_tx = config->det_tx;
		det_ty = config->det_ty;
		k = config->k;
		lambda = config->lambda;
		loop = N;
	}
	__syncthreads();

	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	int xxtr = tid % pnX;
	int yytr = tid / pnX;
	ulonglong idx = xxtr + yytr * pnX;

	float xxx = -half_ssX + (xxtr - 1 + offsetX) * ppX;
	float yyy = -half_ssY + (pnY - yytr + offsetY) * ppY;

	for (int j = 0; j < loop; ++j)
	{ // Create Fringe Pattern
		float pcx = vertex_data[j].point.pos[_X] * scaleX;
		float pcy = vertex_data[j].point.pos[_Y] * scaleY;
		float pcz = vertex_data[j].point.pos[_Z] * scaleZ;

		pcz = pcz + distance;

		float amp = vertex_data[j].color.color[schannel];

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

			// abs(det_tx * sqrt(yyy_pcy_sq + pcz_sq));
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
				dst[idx].x += res_real;
				dst[idx].y += res_imag;
			}
		}
	}
}

__global__
void cudaKernel_single_FastMath_RS_Diffraction(uint channel, Vertex* vertex_data, const GpuConstNERS* config, const int N, cuDoubleComplex* dst)
{
	__shared__ int pnX, pnY, loop, schannel, offsetX, offsetY;
	__shared__ float ppX, ppY, scaleX, scaleY, scaleZ, half_ssX, half_ssY, distance, det_tx, det_ty, k, lambda;

	if (threadIdx.x == 0) {
		schannel = channel;
		pnX = config->pn_X;
		pnY = config->pn_Y;
		ppX = config->pp_X;
		ppY = config->pp_Y;
		offsetX = config->offset_X;
		offsetY = config->offset_Y;
		scaleX = config->scale_X;
		scaleY = config->scale_Y;
		scaleZ = config->scale_Z;
		half_ssX = config->half_ss_X;
		half_ssY = config->half_ss_Y;
		distance = config->offset_depth;
		det_tx = config->det_tx;
		det_ty = config->det_ty;
		k = config->k;
		lambda = config->lambda;
		loop = N;
	}
	__syncthreads();

	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	int xxtr = tid % pnX;
	int yytr = tid / pnX;
	ulonglong idx = xxtr + yytr * pnX;

	float xxx = __fadd_rz(-half_ssX, __fmul_rz(__fadd_rz(__fsub_rz(xxtr, 1), offsetX), ppX));
	float yyy = __fadd_rz(-half_ssY, __fmul_rz(__fadd_rz(__fsub_rz(pnY, yytr), offsetY), ppY));

	for (int j = 0; j < loop; ++j)
	{ // Create Fringe Pattern
		float pcx = __fmul_rz(vertex_data[j].point.pos[_X], scaleX);
		float pcy = __fmul_rz(vertex_data[j].point.pos[_Y], scaleY);
		float pcz = __fmul_rz(vertex_data[j].point.pos[_Z], scaleZ);

		pcz = __fadd_rz(pcz, distance);

		float amp = vertex_data[j].color.color[schannel];

		float abs_det_txy_pcz = abs(__fmul_rz(det_tx, pcz));

		float _xbound[2] = {
			__fadd_rz(pcx, abs_det_txy_pcz),
			__fsub_rz(pcx, abs_det_txy_pcz)
		};

		abs_det_txy_pcz = abs(__fmul_rz(det_ty, pcz));

		float _ybound[2] = {
			__fadd_rz(pcy, abs_det_txy_pcz),
			__fsub_rz(pcy, abs_det_txy_pcz)
		};

		float Xbound[2] = {
			__fadd_rz(floor(__fdiv_rz(__fadd_rz(_xbound[_X], half_ssX), ppX)), 1),
			__fadd_rz(floor(__fdiv_rz(__fadd_rz(_xbound[_Y], half_ssX), ppX)), 1)
		};

		float Ybound[2] = {
			__fsub_rz(pnY, floor(__fdiv_rz(__fadd_rz(_ybound[_Y], half_ssY), ppY))),
			__fsub_rz(pnY, floor(__fdiv_rz(__fadd_rz(_ybound[_X], half_ssY), ppY)))
		};

		if (Xbound[_X] > pnX)	Xbound[_X] = pnX;
		if (Xbound[_Y] < 0)		Xbound[_Y] = 0;
		if (Ybound[_X] > pnY)	Ybound[_X] = pnY;
		if (Ybound[_Y] < 0)		Ybound[_Y] = 0;

		if (((xxtr >= Xbound[_Y]) && (xxtr < Xbound[_X])) &&
			((yytr >= Ybound[_Y]) && (yytr < Ybound[_X]))) {
			float xx = __fsub_rz(xxx, pcx);
			float yy = __fsub_rz(yyy, pcy);

			float xxx_pcx_sq = __fmul_rz(xx, xx);
			float yyy_pcy_sq = __fmul_rz(yy, yy);
			float pcz_sq = __fmul_rz(pcz, pcz);

			float abs_det_txy_sqrt = abs(__fmul_rz(det_tx, __fsqrt_rz(__fadd_rz(yyy_pcy_sq, pcz_sq))));

			float range_x[2] = {
				__fadd_rz(pcx, abs_det_txy_sqrt),
				__fsub_rz(pcx, abs_det_txy_sqrt)
			};

			abs_det_txy_sqrt = abs(__fmul_rz(det_ty, __fsqrt_rz(__fadd_rz(xxx_pcx_sq, pcz_sq))));

			float range_y[2] = {
				__fadd_rz(pcy, abs_det_txy_sqrt),
				__fsub_rz(pcy, abs_det_txy_sqrt)
			};

			if (((xxx < range_x[_X]) && (xxx > range_x[_Y])) &&
				((yyy < range_y[_X]) && (yyy > range_y[_Y]))) {

				float r = __fsqrt_rz(__fadd_rz(__fadd_rz(xxx_pcx_sq, yyy_pcy_sq), pcz_sq));
				float p = __fmul_rz(k, r);
				float a = __fdiv_rz(__fmul_rz(amp, pcz), __fmul_rz(lambda, __fmul_rz(r, r)));
				float res_real = __fmul_rz(sin(p), a);
				float res_imag = __fmul_rz(-cos(p), a);
				dst[idx].x = __dadd_rz(dst[idx].x, res_real);
				dst[idx].y = __dadd_rz(dst[idx].y, res_imag);
			}
		}
	}
}

__global__
void cudaKernel_double_Fresnel_Diffraction(uint channel, Vertex* vertex_data, const GpuConstNEFR* config, const int N, cuDoubleComplex* dst)
{
	__shared__ int pnX, pnY, loop, schannel, offsetX, offsetY;
	__shared__ double ppX, ppY, scaleX, scaleY, scaleZ, half_ssX, half_ssY, distance, k, tx, ty, lambda;

	if (threadIdx.x == 0) {
		schannel = channel;
		pnX = config->pn_X;
		pnY = config->pn_Y;
		ppX = config->pp_X;
		ppY = config->pp_Y;
		offsetX = config->offset_X;
		offsetY = config->offset_Y;
		scaleX = config->scale_X;
		scaleY = config->scale_Y;
		scaleZ = config->scale_Z;
		half_ssX = config->half_ss_X;
		half_ssY = config->half_ss_Y;
		distance = config->offset_depth;
		k = config->k;
		tx = config->tx;
		ty = config->ty;
		lambda = config->lambda;
		loop = N;
	}
	__syncthreads();

	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	int xxtr = tid % pnX;
	int yytr = tid / pnX;
	ulonglong idx = xxtr + yytr * pnX;

	double xxx = -half_ssX + (xxtr - 1 + offsetX) * ppX;
	double yyy = -half_ssY + (pnY - yytr + offsetY) * ppY;

	for (int j = 0; j < loop; ++j)
	{ //Create Fringe Pattern

		double pcx = vertex_data[j].point.pos[_X] * scaleX;
		double pcy = vertex_data[j].point.pos[_Y] * scaleY;
		double pcz = vertex_data[j].point.pos[_Z] * scaleZ;
		pcz = pcz + distance;

		double amp = vertex_data[j].color.color[schannel];

		//boundary test
		double abs_txy_pcz = abs(tx * pcz);
		double _xbound[2] = {
			pcx + abs_txy_pcz,
			pcx - abs_txy_pcz
		};

		abs_txy_pcz = abs(ty * pcz);
		double _ybound[2] = {
			pcy + abs_txy_pcz,
			pcy - abs_txy_pcz
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
		//

		if (((xxtr >= Xbound[_Y]) && (xxtr < Xbound[_X])) && ((yytr >= Ybound[_Y]) && (yytr < Ybound[_X]))) {
			double p = k * ((xxx - pcx) * (xxx - pcx) + (yyy - pcy) * (yyy - pcy) + (2 * pcz * pcz)) / (2 * pcz);
			double a = amp / (lambda * pcz);
			double res_real = sin(p) * a;
			double res_imag = -cos(p) * a;

			dst[idx].x += res_real;
			dst[idx].y += res_imag;
		}
	}
}

__global__
void cudaKernel_double_FastMath_Fresnel_Diffraction(uint channel, Vertex* vertex_data, const GpuConstNEFR* config, const int N, cuDoubleComplex* dst)
{
	__shared__ int pnX, pnY, loop, schannel, offsetX, offsetY;
	__shared__ double ppX, ppY, scaleX, scaleY, scaleZ, half_ssX, half_ssY, distance, k, tx, ty, lambda;

	if (threadIdx.x == 0) {
		schannel = channel;
		pnX = config->pn_X;
		pnY = config->pn_Y;
		ppX = config->pp_X;
		ppY = config->pp_Y;
		offsetX = config->offset_X;
		offsetY = config->offset_Y;
		scaleX = config->scale_X;
		scaleY = config->scale_Y;
		scaleZ = config->scale_Z;
		half_ssX = config->half_ss_X;
		half_ssY = config->half_ss_Y;
		distance = config->offset_depth;
		k = config->k;
		tx = config->tx;
		ty = config->ty;
		lambda = config->lambda;
		loop = N;
	}
	__syncthreads();

	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	int xxtr = tid % pnX;
	int yytr = tid / pnX;
	ulonglong idx = xxtr + yytr * pnX;

	double xxx = __dadd_rz(-half_ssX, __dmul_rz(__dadd_rz(__dsub_rz(xxtr, 1), offsetX), ppX));
	double yyy = __dadd_rz(-half_ssY, __dmul_rz(__dadd_rz(__dsub_rz(pnY, yytr), offsetY), ppY));

	for (int j = 0; j < loop; ++j)
	{ //Create Fringe Pattern

		double pcx = __dmul_rz(vertex_data[j].point.pos[_X], scaleX);
		double pcy = __dmul_rz(vertex_data[j].point.pos[_Y], scaleY);
		double pcz = __dmul_rz(vertex_data[j].point.pos[_Z], scaleZ);
		pcz = __dadd_rz(pcz, distance);

		double amp = vertex_data[j].color.color[schannel];

		//boundary test
		double abs_txy_pcz = abs(__dmul_rz(tx, pcz));
		double _xbound[2] = {
			__dadd_rz(pcx, abs_txy_pcz),
			__dsub_rz(pcx, abs_txy_pcz)
		};

		abs_txy_pcz = abs(__dmul_rz(ty, pcz));
		double _ybound[2] = {
			__dadd_rz(pcy, abs_txy_pcz),
			__dsub_rz(pcy, abs_txy_pcz)
		};

		double Xbound[2] = {
			__dadd_rz(floor(__ddiv_rz(__dadd_rz(_xbound[_X], half_ssX), ppX)), 1),
			__dadd_rz(floor(__ddiv_rz(__dadd_rz(_xbound[_Y], half_ssX), ppX)), 1)
		};

		double Ybound[2] = {
			__dsub_rz(pnY, floor(__ddiv_rz(__dadd_rz(_ybound[_Y], half_ssY), ppY))),
			__dsub_rz(pnY, floor(__ddiv_rz(__dadd_rz(_ybound[_X], half_ssY), ppY)))
		};

		if (Xbound[_X] > pnX)	Xbound[_X] = pnX;
		if (Xbound[_Y] < 0)		Xbound[_Y] = 0;
		if (Ybound[_X] > pnY)	Ybound[_X] = pnY;
		if (Ybound[_Y] < 0)		Ybound[_Y] = 0;
		//

		if (((xxtr >= Xbound[_Y]) && (xxtr < Xbound[_X])) && ((yytr >= Ybound[_Y]) && (yytr < Ybound[_X]))) {
			double xx = __dsub_rz(xxx, pcx);
			double yy = __dsub_rz(yyy, pcy);
			double z2 = __dmul_rz(2, pcz);

			double p = __ddiv_rz(__dmul_rz(k, __dadd_rz(__dadd_rz(__dmul_rz(xx, xx), __dmul_rz(yy, yy)), __dmul_rz(z2, pcz))), z2);

			double a = __ddiv_rz(amp, __dmul_rz(lambda, pcz));
			double res_real = __dmul_rz(sin(p), a);
			double res_imag = __dmul_rz(-cos(p), a);

			dst[idx].x = __dadd_rz(dst[idx].x, res_real);
			dst[idx].y = __dadd_rz(dst[idx].y, res_imag);
		}
	}
}

__global__
void cudaKernel_single_Fresnel_Diffraction(uint channel, Vertex* vertex_data, const GpuConstNEFR* config, const int N, cuDoubleComplex* dst)
{
	__shared__ int pnX, pnY, loop, schannel, offsetX, offsetY;
	__shared__ float ppX, ppY, scaleX, scaleY, scaleZ, half_ssX, half_ssY, distance, k, tx, ty, lambda;

	if (threadIdx.x == 0) {
		schannel = channel;
		pnX = config->pn_X;
		pnY = config->pn_Y;
		ppX = config->pp_X;
		ppY = config->pp_Y;
		offsetX = config->offset_X;
		offsetY = config->offset_Y;
		scaleX = config->scale_X;
		scaleY = config->scale_Y;
		scaleZ = config->scale_Z;
		half_ssX = config->half_ss_X;
		half_ssY = config->half_ss_Y;
		distance = config->offset_depth;
		k = config->k;
		tx = config->tx;
		ty = config->ty;
		lambda = config->lambda;
		loop = N;
	}
	__syncthreads();

	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	int xxtr = tid % pnX;
	int yytr = tid / pnX;
	ulonglong idx = xxtr + yytr * pnX;

	float xxx = -half_ssX + (xxtr - 1 + offsetX) * ppX;
	float yyy = -half_ssY + (pnY - yytr + offsetY) * ppY;

	for (int j = 0; j < loop; ++j)
	{ //Create Fringe Pattern

		float pcx = vertex_data[j].point.pos[_X] * scaleX;
		float pcy = vertex_data[j].point.pos[_Y] * scaleY;
		float pcz = vertex_data[j].point.pos[_Z] * scaleZ;
		pcz = pcz + distance;

		float amp = vertex_data[j].color.color[schannel];

		//boundary test
		float abs_txy_pcz = abs(tx * pcz);
		float _xbound[2] = {
			pcx + abs_txy_pcz,
			pcx - abs_txy_pcz
		};

		abs_txy_pcz = abs(ty * pcz);
		float _ybound[2] = {
			pcy + abs_txy_pcz,
			pcy - abs_txy_pcz
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
		//

		if (((xxtr >= Xbound[_Y]) && (xxtr < Xbound[_X])) && ((yytr >= Ybound[_Y]) && (yytr < Ybound[_X]))) {
			float p = k * ((xxx - pcx) * (xxx - pcx) + (yyy - pcy) * (yyy - pcy) + (2 * pcz * pcz)) / (2 * pcz);
			float a = amp / (lambda * pcz);
			float res_real = sinf(p) * a;
			float res_imag = -cosf(p) * a;

			dst[idx].x += res_real;
			dst[idx].y += res_imag;
		}
	}
}

__global__
void cudaKernel_single_FastMath_Fresnel_Diffraction(uint channel, Vertex* vertex_data, const GpuConstNEFR* config, const int N, cuDoubleComplex* dst)
{
	__shared__ int pnX, pnY, loop, schannel, offsetX, offsetY;
	__shared__ float ppX, ppY, scaleX, scaleY, scaleZ, half_ssX, half_ssY, distance, k, tx, ty, lambda;

	if (threadIdx.x == 0) {
		schannel = channel;
		pnX = config->pn_X;
		pnY = config->pn_Y;
		ppX = config->pp_X;
		ppY = config->pp_Y;
		offsetX = config->offset_X;
		offsetY = config->offset_Y;
		scaleX = config->scale_X;
		scaleY = config->scale_Y;
		scaleZ = config->scale_Z;
		half_ssX = config->half_ss_X;
		half_ssY = config->half_ss_Y;
		distance = config->offset_depth;
		k = config->k;
		tx = config->tx;
		ty = config->ty;
		lambda = config->lambda;
		loop = N;
	}
	__syncthreads();

	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	int xxtr = tid % pnX;
	int yytr = tid / pnX;
	ulonglong idx = xxtr + yytr * pnX;

	float xxx = __fadd_rz(-half_ssX, __fmul_rz(__fadd_rz(__fsub_rz(xxtr, 1), offsetX), ppX));
	float yyy = __fadd_rz(-half_ssY, __fmul_rz(__fadd_rz(__fsub_rz(pnY, yytr), offsetY), ppY));

	for (int j = 0; j < loop; ++j)
	{ //Create Fringe Pattern

		float pcx = __fmul_rz(vertex_data[j].point.pos[_X], scaleX);
		float pcy = __fmul_rz(vertex_data[j].point.pos[_Y], scaleY);
		float pcz = __fmul_rz(vertex_data[j].point.pos[_Z], scaleZ);
		pcz = __fadd_rz(pcz, distance);

		float amp = vertex_data[j].color.color[schannel];

		//boundary test
		float abs_txy_pcz = abs(__fmul_rz(tx, pcz));
		float _xbound[2] = {
			__fadd_rz(pcx, abs_txy_pcz),
			__fsub_rz(pcx, abs_txy_pcz)
		};

		abs_txy_pcz = abs(__fmul_rz(ty, pcz));
		float _ybound[2] = {
			__fadd_rz(pcy, abs_txy_pcz),
			__fsub_rz(pcy, abs_txy_pcz)
		};

		float Xbound[2] = {
			__fadd_rz(floor(__fdiv_rz(__fadd_rz(_xbound[_X], half_ssX), ppX)), 1),
			__fadd_rz(floor(__fdiv_rz(__fadd_rz(_xbound[_Y], half_ssX), ppX)), 1)
		};

		float Ybound[2] = {
			__fsub_rz(pnY, floor(__fdiv_rz(__fadd_rz(_ybound[_Y], half_ssY), ppY))),
			__fsub_rz(pnY, floor(__fdiv_rz(__fadd_rz(_ybound[_X], half_ssY), ppY)))
		};

		if (Xbound[_X] > pnX)	Xbound[_X] = pnX;
		if (Xbound[_Y] < 0)		Xbound[_Y] = 0;
		if (Ybound[_X] > pnY)	Ybound[_X] = pnY;
		if (Ybound[_Y] < 0)		Ybound[_Y] = 0;
		//

		if (((xxtr >= Xbound[_Y]) && (xxtr < Xbound[_X])) && ((yytr >= Ybound[_Y]) && (yytr < Ybound[_X]))) {
			float xx = __fsub_rz(xxx, pcx);
			float yy = __fsub_rz(yyy, pcy);
			float z2 = __fmul_rz(2, pcz);

			float p = __fdiv_rz(__fmul_rz(k, __fadd_rz(__fadd_rz(__fmul_rz(xx, xx), __fmul_rz(yy, yy)), __fmul_rz(z2, pcz))), z2);

			float a = __fdiv_rz(amp, __fmul_rz(lambda, pcz));
			float res_real = __fmul_rz(sin(p), a);
			float res_imag = __fmul_rz(-cos(p), a);

			dst[idx].x = __fadd_rz(dst[idx].x, res_real);
			dst[idx].y = __fadd_rz(dst[idx].y, res_imag);
		}
	}
}


extern "C"
{
	void cudaPointCloud_RS(
		const int& nBlocks, const int& nThreads, const int& n_pts_per_stream,
		Vertex* cuda_vertex_data,
		cuDoubleComplex* cuda_dst,
		const GpuConstNERS* cuda_config, const uint& iChannel, const uint& mode
	)
	{
		if (mode & MODE_FLOAT)
		{
			if (mode & MODE_FASTMATH)
				cudaKernel_single_FastMath_RS_Diffraction << < nBlocks, nThreads >> > (iChannel, cuda_vertex_data, cuda_config, n_pts_per_stream, cuda_dst);
			else
				cudaKernel_single_RS_Diffraction << < nBlocks, nThreads >> > (iChannel, cuda_vertex_data, cuda_config, n_pts_per_stream, cuda_dst);
		}
		else
		{
			if (mode & MODE_FASTMATH)
				cudaKernel_double_FastMath_RS_Diffraction << < nBlocks, nThreads >> > (iChannel, cuda_vertex_data, cuda_config, n_pts_per_stream, cuda_dst);
			else
				cudaKernel_double_RS_Diffraction << < nBlocks, nThreads >> > (iChannel, cuda_vertex_data, cuda_config, n_pts_per_stream, cuda_dst);
		}
	}

	void cudaPointCloud_Fresnel(
		const int& nBlocks, const int& nThreads, const int& n_pts_per_stream,
		Vertex* cuda_vertex_data,
		cuDoubleComplex* cuda_dst,
		const GpuConstNEFR* cuda_config, const uint& iChannel, const uint& mode
	)
	{
		if (mode & MODE_FLOAT)
		{
			if (mode & MODE_FASTMATH)
				cudaKernel_single_FastMath_Fresnel_Diffraction << < nBlocks, nThreads >> > (iChannel, cuda_vertex_data, cuda_config, n_pts_per_stream, cuda_dst);
			else
				cudaKernel_single_Fresnel_Diffraction << < nBlocks, nThreads >> > (iChannel, cuda_vertex_data, cuda_config, n_pts_per_stream, cuda_dst);
		}
		else
		{
			if (mode & MODE_FASTMATH)
				cudaKernel_double_FastMath_Fresnel_Diffraction << < nBlocks, nThreads >> > (iChannel, cuda_vertex_data, cuda_config, n_pts_per_stream, cuda_dst);
			else
				cudaKernel_double_Fresnel_Diffraction << < nBlocks, nThreads >> > (iChannel, cuda_vertex_data, cuda_config, n_pts_per_stream, cuda_dst);
		}
	}
}

#endif // !OphPCKernel_cu__