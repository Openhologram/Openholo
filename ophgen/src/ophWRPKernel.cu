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

#ifndef ophWRPKernel_cu__
#define ophWRPKernel_cu__

#include "ophKernel.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand_uniform.h>
#include <device_launch_parameters.h>
#include "ophWRP_GPU.h"

__global__ void cudaKernel_CalcData(cufftDoubleComplex *src, const WRPGpuConst* config)
{
	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ double ppX;
	__shared__ double ppY;
	__shared__ int pnX;
	__shared__ int pnY;
	__shared__ int pnXY;
	__shared__ double ssX;
	__shared__ double ssY;
	__shared__ double z;
	__shared__ double v;
	__shared__ double lambda;
	__shared__ double distance;
	__shared__ double pi2;

	if (threadIdx.x == 0)
	{
		ppX = config->pp_X;
		ppY = config->pp_Y;
		pnX = config->pn_X;
		pnY = config->pn_Y;
		pnXY = pnX * pnY;
		ssX = pnX * ppX * 2;
		ssY = pnY * ppY * 2;
		lambda = config->lambda;
		distance = config->propa_d;
		pi2 = config->pi2;
		z = distance * pi2;
		v = 1 / (lambda * lambda);
	}
	__syncthreads();

	if (tid < pnXY * 4)
	{
		int pnX2 = pnX * 2;

		int w = tid % pnX2;
		int h = tid / pnX2;

		double fy = (-pnY + h) / ssY;
		double fyy = fy * fy;
		double fx = (-pnX + w) / ssX;
		double fxx = fx * fx;
		double sqrtpart = sqrt(v - fxx - fyy);

		cuDoubleComplex prop;
		prop.x = 0;
		prop.y = z * sqrtpart;

		exponent_complex(&prop);

		cuDoubleComplex val;
		val.x = src[tid].x;
		val.y = src[tid].y;

		cuDoubleComplex val2 = cuCmul(val, prop);
		src[tid].x = val2.x;
		src[tid].y = val2.y;
	}
}

__global__ void cudaKernel_MoveDataPost(cuDoubleComplex *src, cuDoubleComplex *dst, const WRPGpuConst* config)
{
	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int pnX;
	__shared__ int pnY;
	__shared__ ulonglong pnXY;

	if (threadIdx.x == 0)
	{
		pnX = config->pn_X;
		pnY = config->pn_Y;
		pnXY = pnX * pnY;
	}
	__syncthreads();

	if (tid < pnXY)
	{
		int w = tid % pnX;
		int h = tid / pnX;
		ulonglong iSrc = pnX * 2 * (pnY / 2 + h) + pnX / 2;

		dst[tid] = src[iSrc + w];
	}
}

__global__ void cudaKernel_MoveDataPre(cuDoubleComplex *src, cuDoubleComplex *dst, const WRPGpuConst* config)
{
	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int pnX;
	__shared__ int pnY;
	__shared__ ulonglong pnXY;

	if (threadIdx.x == 0)
	{
		pnX = config->pn_X;
		pnY = config->pn_Y;
		pnXY = pnX * pnY;
	}
	__syncthreads();

	if (tid < pnXY)
	{
		int w = tid % pnX;
		int h = tid / pnX;
		ulonglong iDst = pnX * 2 * (pnY / 2 + h) + pnX / 2;
		dst[iDst + w] = src[tid];
	}
}

#if 0
__global__ void cudaKernel_GenWRP(Real* pc_dst, Real* amp_dst, const WRPGpuConst* config, const int n_points_stream, cuDoubleComplex* dst)
{
	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;

	curandState state;

	int idx = tid * 3;

	__shared__ double ppX;
	__shared__ double ppY;
	__shared__ int pnX;
	__shared__ int pnY;
	__shared__ int pnXY;
	__shared__ int hpnX;
	__shared__ int hpnY;
	__shared__ double dz;
	__shared__ double dzz;
	__shared__ double pi2;
	__shared__ double k;
	__shared__ double lambda;
	__shared__ bool random_phase;
	__shared__ bool sign;
	__shared__ int iAmp;
	__shared__ double det_tx;
	__shared__ double det_ty;
	__shared__ double half_ssX;
	__shared__ double half_ssY;

	if (threadIdx.x == 0) {
		ppX = config->pp_X;
		ppY = config->pp_Y;
		pnX = config->pn_X;
		pnY = config->pn_Y;
		pnXY = pnX * pnY;
		hpnX = pnX / 2;
		hpnY = pnY / 2;
		dz = config->wrp_d - config->zmax;
		dzz = dz * dz;
		k = config->k;
		lambda = config->lambda;
		pi2 = config->pi2;
		random_phase = config->bRandomPhase;
		sign = dz > 0.0 ? true : false;
		iAmp = config->iAmplitude;
		det_tx = config->det_tx;
		det_ty = config->det_ty;
		half_ssX = pnX * ppX / 2;
		half_ssY = pnY * ppY / 2;
	}
	__syncthreads();

	if (tid < pnXY)
	{
		int xxtr = tid % pnX;
		int yytr = tid / pnX;
		
		double xxx = -half_ssX + (xxtr - 1) * ppX;
		double yyy = -half_ssY + (pnY - yytr) * ppY;

		for (int i = 0; i < n_points_stream; i++)
		{
			int idx = 3 * i;
			double x = pc_dst[idx + _X];
			double y = pc_dst[idx + _Y];
			double z = pc_dst[idx + _Z];
			double amp = amp_dst[idx + iAmp];

			double abs_det_txy_pcz = abs(det_tx * dz);

			double _xbound[2] = {
				x + abs_det_txy_pcz,
				x - abs_det_txy_pcz
			};

			abs_det_txy_pcz = abs(det_ty * dz);

			double _ybound[2] = {
				y + abs_det_txy_pcz,
				y - abs_det_txy_pcz
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

				double xxx_pcx_sq = (xxx - x) * (xxx - x);
				double yyy_pcy_sq = (yyy - y) * (yyy - y);
				double pcz_sq = dz * dz;

				// abs(det_tx * sqrt(yyy_pcy_sq + pcz_sq));
				double abs_det_txy_sqrt = abs(det_tx * sqrt(yyy_pcy_sq + pcz_sq));

				double range_x[2] = {
					x + abs_det_txy_sqrt,
					x - abs_det_txy_sqrt
				};

				abs_det_txy_sqrt = abs(det_ty * sqrt(xxx_pcx_sq + pcz_sq));

				double range_y[2] = {
					y + abs_det_txy_sqrt,
					y - abs_det_txy_sqrt
				};

				if (((xxx < range_x[_X]) && (xxx > range_x[_Y])) &&
					((yyy < range_y[_X]) && (yyy > range_y[_Y]))) {

					double r = sqrt(xxx_pcx_sq + yyy_pcy_sq + pcz_sq);
					double p = k * r;
					double a = (amp * dz) / (lambda * (r * r));
					double res_real = sin(p) * a;
					double res_imag = -cos(p) * a;
					
					dst[tid].x += res_real;
					dst[tid].y += res_imag;
				}
			}
		}
	}
}

#else
__global__ void cudaKernel_GenWRP(Real* pc_dst, Real* amp_dst, const WRPGpuConst* config, const int n_points_stream, cuDoubleComplex* dst)
{
	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < n_points_stream)
	{
		__shared__ double ppX;
		__shared__ double ppY;
		__shared__ int pnX;
		__shared__ int pnY;
		__shared__ double dz;
		__shared__ double dzz;
		__shared__ double pi2;
		__shared__ double k;
		__shared__ double lambda;
		__shared__ bool random_phase;
		__shared__ bool sign;

		if (threadIdx.x == 0) {
			ppX = config->pp_X;
			ppY = config->pp_Y;
			pnX = config->pn_X;
			pnY = config->pn_Y;
			dz = config->wrp_d - config->zmax;
			dzz = dz * dz;
			k = config->k;
			lambda = config->lambda;
			pi2 = config->pi2;
			random_phase = config->bRandomPhase;
			sign = dz > 0.0 ? true : false;
		}
		__syncthreads();

		int idx = tid * 3;
		double x = pc_dst[idx + _X];
		double y = pc_dst[idx + _Y];
		double z = pc_dst[idx + _Z];
		double amp = amp_dst[idx + config->iAmplitude];

		int hpnX = pnX / 2;
		int hpnY = pnY / 2;
		double ppXX = ppX * ppX * 2;
		//double dz = config->wrp_d - config->zmax;
		double tw = fabs(lambda * dz / ppXX) * 2;

		int w = (int)tw;
		int tx = (int)(x / ppX) + hpnX;
		int ty = (int)(y / ppY) + hpnY;

		curandState state;
		if (random_phase)
		{
			curand_init(4 * w * w, 0, 0, &state);
		}

		for (int wy = -w; wy < w; wy++)
		{
			double dy = wy * ppY;
			double dyy = dy * dy;
			int tmpY = wy + ty;
			int baseY = tmpY * pnX;

			for (int wx = -w; wx < w; wx++) //WRP coordinate
			{
				int tmpX = wx + tx;

				if (tmpX >= 0 && tmpX < pnX && tmpY >= 0 && tmpY < pnY) {
					int iDst = tmpX + baseY;

					double dx = wx * ppX;

					double r = sign ? sqrt(dx * dx + dyy + dzz) : -sqrt(dx * dx + dyy + dzz);
					double randomData = random_phase ? curand_uniform_double(&state) : 1.0;
					double randVal = randomData * pi2;

					cuDoubleComplex tmp;
					tmp.x = (amp * cos(k*r) * cos(randVal)) / r;
					tmp.y = (-amp * sin(k*r) * sin(randVal)) / r;

#if defined(__cplusplus) && defined(__CUDACC__)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
					dst[iDst].x = atomicAdd(&dst[iDst].x, tmp.x);
					dst[iDst].y = atomicAdd(&dst[iDst].y, tmp.y);

#else
					dst[iDst].x += tmp.x; // <-- sync problem
					dst[iDst].y += tmp.y; // <-- sync problem
#endif					
#else
					dst[iDst].x += tmp.x; // <-- sync problem
					dst[iDst].y += tmp.y; // <-- sync problem
#endif
				}
			}
		}
	}
}
#endif

extern "C"
{
	void cudaFresnelPropagationWRP(
		const int &nBlocks, const int&nBlocks2, const int &nThreads, const int &nx, const int &ny,
		cuDoubleComplex *src, cuDoubleComplex *dst, cufftDoubleComplex *fftsrc, cufftDoubleComplex *fftdst,
		const WRPGpuConst* cuda_config)
	{
		cudaKernel_MoveDataPre << <nBlocks, nThreads >> > (src, dst, cuda_config);

		cudaFFT(nullptr, nx * 2, ny * 2, dst, fftsrc, CUFFT_FORWARD, false);

		cudaKernel_CalcData << <nBlocks2, nThreads >> > (fftsrc, cuda_config);

		cudaFFT(nullptr, nx * 2, ny * 2, fftsrc, fftdst, CUFFT_INVERSE, true);

		cudaKernel_MoveDataPost << <nBlocks, nThreads >> > (fftdst, src, cuda_config);
	}

	void cudaGenWRP(
		const int &nBlocks, const int &nThreads, const int &n_pts_per_stream,
		Real* cuda_pc_data, Real* cuda_amp_data,
		cuDoubleComplex* cuda_dst, const WRPGpuConst* cuda_config)
	{
		cudaKernel_GenWRP << <nBlocks, nThreads >> > (cuda_pc_data, cuda_amp_data, cuda_config, n_pts_per_stream, cuda_dst);
	}
}

#endif // !OphWRPKernel_cu__