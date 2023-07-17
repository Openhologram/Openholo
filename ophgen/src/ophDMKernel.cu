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
#ifndef ophDMKernel_cu__
#define ophDMKernel_cu__

#include "ophDepthMap_GPU.h"
#include "ophKernel.cuh"
#include "typedef.h"
#include <define.h>


__global__
void cudaKernel_double_get_kernel(cufftDoubleComplex* u_o_gpu, unsigned char* img_src_gpu, unsigned char* dimg_src_gpu, double* depth_index_gpu,
	int dtr, cuDoubleComplex rand_phase, cuDoubleComplex carrier_phase_delay, int pnx, int pny,
	int change_depth_quantization, unsigned int default_depth_quantization)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < pnx*pny) {

		double img = ((double)img_src_gpu[tid]) / 255.0;
		double depth_idx;
		if (change_depth_quantization == 1)
			depth_idx = depth_index_gpu[tid];
		else
			depth_idx = (double)default_depth_quantization - (double)dimg_src_gpu[tid];

		double alpha_map = ((double)img_src_gpu[tid] > 0.0 ? 1.0 : 0.0);

		u_o_gpu[tid].x = img * alpha_map * (depth_idx == (double)dtr ? 1.0 : 0.0);

		cuDoubleComplex tmp1 = cuCmul(rand_phase, carrier_phase_delay);
		u_o_gpu[tid] = cuCmul(u_o_gpu[tid], tmp1);
	}
}


__global__
void cudaKernel_single_get_kernel(cufftDoubleComplex* u_o_gpu, unsigned char* img_src_gpu, unsigned char* dimg_src_gpu, double* depth_index_gpu,
	int dtr, cuComplex rand_phase, cuComplex carrier_phase_delay, int pnx, int pny,
	int change_depth_quantization, unsigned int default_depth_quantization)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < pnx * pny) {

		float img = ((float)img_src_gpu[tid]) / 255.0f;
		float depth_idx;
		if (change_depth_quantization == 1)
			depth_idx = depth_index_gpu[tid];
		else
			depth_idx = (float)default_depth_quantization - (float)dimg_src_gpu[tid];

		float alpha_map = ((float)img_src_gpu[tid] > 0.0f ? 1.0f : 0.0f);

		u_o_gpu[tid].x = img * alpha_map * (depth_idx == (float)dtr ? 1.0f : 0.0f);

		cuComplex tmp1 = cuCmulf(rand_phase, carrier_phase_delay);

		u_o_gpu[tid].x = (u_o_gpu[tid].x * tmp1.x) - (u_o_gpu[tid].y * tmp1.y);
		u_o_gpu[tid].y = (u_o_gpu[tid].x * tmp1.y) + (u_o_gpu[tid].y * tmp1.x);
	}
}


__global__
void propagation_angularsp_kernel(cufftDoubleComplex* input_d, cufftDoubleComplex* u_complex, const DMKernelConfig* config, double propagation_dist)
{
#if 0
	__shared__ int s_pnX, s_pnY;
	__shared__ double s_k, s_ssX, s_ssY, s_ppX, s_ppY, s_lambda, s_distance;

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (threadIdx.x == 0)
	{
		s_pnX = config->pn_X;
		s_pnY = config->pn_Y;
		s_ppX = -1.0 / (config->pp_X * 2.0);
		s_ppY = 1.0 / (config->pp_Y * 2.0);
		s_ssX = 1.0 / config->ss_X;
		s_ssY = 1.0 / config->ss_Y;
		s_lambda = config->lambda;
		s_k = config->k * config->k;
		s_distance = propagation_dist;
	}
	__syncthreads();

	if (tid < s_pnX * s_pnY) {

		int x = tid % s_pnX;
		int y = tid / s_pnY;

		double fxx = s_ppX + s_ssX * (double)x;
		double fyy = s_ppY - s_ssY - s_ssY * (double)y;

		double sval = sqrt(1 - (s_lambda * fxx) * (s_lambda * fxx) - (s_lambda * fyy) * (s_lambda * fyy));
		sval *= s_k * s_distance;

		cuDoubleComplex kernel = make_cuDoubleComplex(0, sval);
		exponent_complex(&kernel);

		int prop_mask = ((fxx * fxx + fyy * fyy) < s_k) ? 1 : 0;

		cuDoubleComplex u_frequency = make_cuDoubleComplex(0, 0);
		if (prop_mask == 1)
			u_frequency = cuCmul(kernel, input_d[tid]);

		u_complex[tid] = cuCadd(u_complex[tid], u_frequency);
	}
#else
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < config->pn_X * config->pn_Y)
	{
		int x = tid % config->pn_X;
		int y = tid / config->pn_X;

		double fxx = (-1.0 / (2.0 * config->pp_X)) + (1.0 / config->ss_X) * (double)x;
		double fyy = (1.0 / (2.0 * config->pp_Y)) - (1.0 / config->ss_Y) - (1.0 / config->ss_Y) * (double)y;


		double sval = sqrt(1 - (config->lambda * fxx) * (config->lambda * fxx) -
			(config->lambda * fyy) * (config->lambda * fyy));
		sval *= config->k * propagation_dist;

		int prop_mask = ((fxx * fxx + fyy * fyy) < (config->k * config->k)) ? 1 : 0;

		cuDoubleComplex kernel = make_cuDoubleComplex(0, sval);
		exponent_complex(&kernel);

		cuDoubleComplex u_frequency = make_cuDoubleComplex(0, 0);
		if (prop_mask == 1)
			u_frequency = cuCmul(kernel, input_d[tid]);

		u_complex[tid] = cuCadd(u_complex[tid], u_frequency);

	}

#endif
}


__global__ 
void cropFringe(int nx, int ny, cufftDoubleComplex* in_filed, cufftDoubleComplex* out_filed, int cropx1, int cropx2, int cropy1, int cropy2)
{
	__shared__ int s_pnX, s_pnY, s_cropx1, s_cropx2, s_cropy1, s_cropy2;

	if (threadIdx.x == 0)
	{
		s_pnX = nx;
		s_pnY = ny;
		s_cropx1 = cropx1;
		s_cropx2 = cropx2;
		s_cropy1 = cropy1;
		s_cropy2 = cropy2;
	}
	__syncthreads();

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < s_pnX * s_pnY)
	{
		int x = tid % s_pnX;
		int y = tid / s_pnX;

		if (x >= s_cropx1 && x <= s_cropx2 && y >= s_cropy1 && y <= s_cropy2)
			out_filed[tid] = in_filed[tid];
	}
}


__global__ 
void getFringe(int nx, int ny, cufftDoubleComplex* in_filed, cufftDoubleComplex* out_filed, int sig_locationx, int sig_locationy,
	double ssx, double ssy, double ppx, double ppy, double pi)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < nx*ny)
	{
		cuDoubleComplex shift_phase = make_cuDoubleComplex(1, 0);

		if (sig_locationy != 0)
		{
			int r = tid / nx;
			double yy = (ssy / 2.0) - (ppy)*(double)r - ppy;

			cuDoubleComplex val = make_cuDoubleComplex(0, 0);
			if (sig_locationy == 1)
				val.y = 2.0 * pi * (yy / (4.0 * ppy));
			else
				val.y = 2.0 * pi * (-yy / (4.0 * ppy));

			exponent_complex(&val);

			shift_phase = cuCmul(shift_phase, val);
		}

		if (sig_locationx != 0)
		{
			int c = tid % nx;
			double xx = (-ssx / 2.0) - (ppx)*(double)c - ppx;

			cuDoubleComplex val = make_cuDoubleComplex(0, 0);
			if (sig_locationx == -1)
				val.y = 2.0 * pi * (-xx / (4.0 * ppx));
			else
				val.y = 2.0 * pi * (xx / (4.0 * ppx));

			exponent_complex(&val);
			shift_phase = cuCmul(shift_phase, val);
		}

		out_filed[tid] = cuCmul(in_filed[tid], shift_phase);
	}

}

__global__
void change_depth_quan_kernel(double* depth_index_gpu, unsigned char* dimg_src_gpu, int pnx, int pny,
	int dtr, double d1, double d2, double num_depth, double far_depth, double near_depth)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < pnx * pny) {

		int tdepth;
		double dmap_src = double(dimg_src_gpu[tid]) / 255.0;
		double dmap = (1.0 - dmap_src)*(far_depth - near_depth) + near_depth;
		
		if (dtr < num_depth - 1)
			tdepth = (dmap >= d1 ? 1 : 0) * (dmap < d2 ? 1 : 0);
		else
			tdepth = (dmap >= d1 ? 1 : 0) * (dmap <= d2 ? 1 : 0);

		depth_index_gpu[tid] = depth_index_gpu[tid] + (double)(tdepth * (dtr + 1));
	}
}

extern "C"
{
	void cudaDepthHoloKernel(CUstream_st* stream, int pnx, int pny, cufftDoubleComplex* u_o_gpu, unsigned char* img_src_gpu, unsigned char* dimg_src_gpu, double* depth_index_gpu,
		int dtr, cuDoubleComplex rand_phase_val, cuDoubleComplex carrier_phase_delay, int flag_change_depth_quan, unsigned int default_depth_quan, const unsigned int& mode)
	{
		dim3 grid((pnx * pny + kBlockThreads - 1) / kBlockThreads, 1, 1);

		if (mode & MODE_FLOAT)
		{
			//if (mode & MODE_FASTMATH)
			//	//cudaKernel_single_FastMath_RS_Diffraction << < nBlocks, nThreads >> > (iChannel, cuda_vertex_data, cuda_config, n_pts_per_stream, cuda_dst);
			//else
				cudaKernel_single_get_kernel << <grid, kBlockThreads, 0, stream >> > (u_o_gpu, img_src_gpu, dimg_src_gpu, depth_index_gpu,
					dtr, make_cuComplex((float)rand_phase_val.x, (float)rand_phase_val.y),
					make_cuComplex((float)carrier_phase_delay.x, (float)carrier_phase_delay.y), pnx, pny, flag_change_depth_quan, default_depth_quan);
		}
		else
		{
			//if (mode & MODE_FASTMATH)
			//	//cudaKernel_double_FastMath_RS_Diffraction << < nBlocks, nThreads >> > (iChannel, cuda_vertex_data, cuda_config, n_pts_per_stream, cuda_dst);
			//else
				cudaKernel_double_get_kernel << <grid, kBlockThreads, 0, stream >> > (u_o_gpu, img_src_gpu, dimg_src_gpu, depth_index_gpu,
					dtr, rand_phase_val, carrier_phase_delay, pnx, pny, flag_change_depth_quan, default_depth_quan);
		}

	}


	void cudaPropagation_AngularSpKernel(
		const int& nBlocks, const int& nThreads,
		CUstream_st* stream, cufftDoubleComplex* input_d, cufftDoubleComplex* u_complex, 
		const DMKernelConfig*cuda_config, double propagation_dist)
	{
		propagation_angularsp_kernel << <nBlocks, nThreads >> > (input_d, u_complex, cuda_config, propagation_dist);
	}

	void cudaCropFringe(CUstream_st* stream, int nx, int ny, cufftDoubleComplex* in_field, cufftDoubleComplex* out_field, int cropx1, int cropx2, int cropy1, int cropy2)
	{
		unsigned int nblocks = (nx * ny + kBlockThreads - 1) / kBlockThreads;

		cropFringe << < nblocks, kBlockThreads, 0, stream >> > (nx, ny, in_field, out_field, cropx1, cropx2, cropy1, cropy2);
	}

	void cudaGetFringe(CUstream_st* stream, int pnx, int pny, cufftDoubleComplex* in_field, cufftDoubleComplex* out_field, int sig_locationx, int sig_locationy,
		double ssx, double ssy, double ppx, double ppy, double PI)
	{
		unsigned int nblocks = (pnx * pny + kBlockThreads - 1) / kBlockThreads;

		getFringe << < nblocks, kBlockThreads, 0, stream >> > (pnx, pny, in_field, out_field, sig_locationx, sig_locationy, ssx, ssy, ppx, ppy, PI);
	}

	void cudaChangeDepthQuanKernel(CUstream_st* stream, int pnx, int pny, double* depth_index_gpu, unsigned char* dimg_src_gpu,
		int dtr, double d1, double d2, double params_num_of_depth, double params_far_depthmap, double params_near_depthmap)
	{
		dim3 grid((pnx * pny + kBlockThreads - 1) / kBlockThreads, 1, 1);
		change_depth_quan_kernel << <grid, kBlockThreads, 0, stream >> > (depth_index_gpu, dimg_src_gpu, pnx, pny,
			dtr, d1, d2, params_num_of_depth, params_far_depthmap, params_near_depthmap);
	}
}

#endif // !ophDMKernel_cu__