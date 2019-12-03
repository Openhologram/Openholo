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

//#ifndef ophWRPKernel_cu__
//#define ophWRPKernel_cu__

//#include "ophKernel.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <cufft.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <math.h>

#include <stdio.h>
#include "typedef.h"
#include "ophWRP_GPU.h"

__global__ void cudaKernel_GenWRP(Real* pc_dst, Real* amp_dst, const WRPGpuConst* config, const int n_points_stream, Real* dst_re, Real* dst_im)
{

	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	ulonglong tid_offset = blockDim.x * gridDim.x;
	ulonglong n_pixels = config->pn_X * config->pn_Y;

	if (tid < n_pixels)
	{
		int xxtr = tid % config->pn_X;
		int yytr = tid / config->pn_X;
		ulonglong idx = xxtr + yytr * config->pn_X;

		float wz = config->wrp_d - config->zmax;
		float wm = round(fabs(wz*tan(config->lambda / (2 * config->pp_X)) / config->pp_X));

		for (int wy = -wm; wy < wm; wy++) {
			for (int wx = -wm; wx < wm; wx++) {//WRP coordinate

				int xidx = xxtr + wx;
				int yidx = yytr + wy;
				ulonglong oidx = xidx + yidx*config->pn_X;

				if (oidx>0 && oidx<n_pixels) {

					if (pc_dst[oidx] != 0) {

						Real dz = config->wrp_d - pc_dst[oidx];
						float tw = round(fabs(dz*tan(config->lambda / (2 * config->pp_X)) / config->pp_X));

						if (abs(wx) <= tw)
						{
							Real dx = wx*config->pp_X;
							Real dy = wy*config->pp_Y;

							double sign = (dz > 0.0) ? (1.0) : (-1.0);
							double r = sign * sqrt(dx * dx + dy * dy + dz * dz);

							//	randomData = curand_uniform_double(&state);

							double tmp_re, tmp_im;
							//	cufftDoubleComplex tmp;
							tmp_re = amp_dst[oidx] * (cosf(config->k*r) * cosf(config->k*config->lambda)) / r;
							tmp_im = amp_dst[oidx] * (sinf(config->k*r) * sinf(config->k*config->lambda)) / r;  //*randomData

							dst_re[idx] += tmp_re;
							dst_im[idx] += tmp_im;
						}
					}
				}
			}
		}
	}
}

__global__ void cudaKernel_genindexx(Real* pc_data, Real* pc_index, const WRPGpuConst* config, const int n_points_stream)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid<n_points_stream)
	{
		pc_index[3 * tid] = round(pc_data[3 * tid] / config->pp_X + (config->pn_X - 1) / 2);
		pc_index[3 * tid + 1] = round(pc_data[3 * tid + 1] / config->pp_Y + (config->pn_Y - 1) / 2);
		pc_index[3 * tid + 2] = pc_data[3 * tid + 2];
	}
}

__global__ void cudaKernel_GetObjDst(Real* pc_index, Real* obj_dst, const WRPGpuConst* config, const int n_points_stream)
{
	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < n_points_stream)
	{
		ulonglong ox, oy, o_index;

		ox = pc_index[3 * tid];
		oy = pc_index[3 * tid + 1];
		o_index = ox + oy * config->pn_X;
		obj_dst[o_index] = pc_index[3 * tid + 2];

	}
}
__global__ void cudaKernel_GetAmpDst(Real* pc_index, Real* pc_amp, Real* amp_dst, const WRPGpuConst* config, const int n_points_stream)
{
	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	ulonglong tid_offset = blockDim.x * gridDim.x;

	ulonglong n_pixels = config->pn_X * config->pn_Y;
	const int n_colors = config->n_colors;

	if (tid<n_points_stream)
	{
		ulonglong ox, oy, o_index;

		ox = pc_index[3 * tid];
		oy = pc_index[3 * tid + 1];
		o_index = ox + oy * config->pn_X;
		amp_dst[o_index] = pc_amp[3 * tid];
	}
}

extern "C"
{
	void cudaGenWRP(
		const int &nBlocks, const int &nThreads, const int &n_pts_per_stream,
		Real* cuda_pc_data, Real* cuda_amp_data,
		Real* cuda_dst_re, Real* cuda_dst_im,
		const WRPGpuConst* cuda_config)
	{
		cudaKernel_GenWRP << <nBlocks, nThreads >> >(cuda_pc_data, cuda_amp_data, cuda_config, n_pts_per_stream, cuda_dst_re, cuda_dst_im);
	}

	void cudaGenindexx(
		const int &nBlocks, const int &nThreads, const int &n_pts_per_stream,
		Real* cuda_pc_data, Real* cuda_pc_indexx,
		const WRPGpuConst* cuda_config)
	{
		cudaKernel_genindexx << <nBlocks, nThreads >> > (cuda_pc_data, cuda_pc_indexx, cuda_config, n_pts_per_stream);
	}

	void cudaGetObjDst(
		const int &nBlocks, const int &nThreads, const int &n_pts_per_stream,
		Real* cuda_pc_index, Real* cuda_pc_obj_dst,
		const WRPGpuConst* cuda_config)
	{
		cudaKernel_GetObjDst << <nBlocks, nThreads >> > (cuda_pc_index, cuda_pc_obj_dst, cuda_config, n_pts_per_stream);
	}

	void cudaGetAmpDst(
		const int &nBlocks, const int &nThreads, const int &n_pts_per_stream,
		Real* cuda_pc_index, Real* cuda_pc_amp, Real* cuda_amp_dst,
		const WRPGpuConst* cuda_config)
	{
		cudaKernel_GetAmpDst << <nBlocks, nThreads >> > (cuda_pc_index, cuda_pc_amp, cuda_amp_dst, cuda_config, n_pts_per_stream);
	}


}

//#endif // !OphWRPKernel_cu__



