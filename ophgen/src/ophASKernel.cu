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
* @file		ophASKernel.cu
* @brief	Openholo Angluar Spectrum with CUDA GPGPU
* @author	Minwoo Nam
* @date		2023/03
*/

#ifndef OphASKernel_cu__
#define OphASKernel_cu__

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "typedef.h"
#include "ophAS_GPU.h"

__global__ void cudaKernel_Transfer(constValue val, creal_T* a, creal_T* b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < val.w && j < val.w)
	{
		double eta_id = val.wavelength * (((double(i) + 1.0) - (val.w / 2.0 + 1.0)) *
			val.minfrequency_eta);
		double xi_id = val.wavelength * (((double(j) + 1.0) - (val.w / 2.0 + 1.0)) *
			val.minfrequency_xi);
		double y_im = (val.knumber * val.depth) * sqrt((1.0 - eta_id * eta_id) - xi_id * xi_id);
		double y_re = cos(y_im);
		y_im = sin(y_im);
		b[i + val.w * j].re = a[i + val.w * j].re * y_re -
			a[i + val.w * j].im * y_im;
		b[i + val.w * j].im = a[i + val.w * j].re * y_im +
			a[i + val.w * j].im * y_re;
	}
}

__global__ void cudaKernel_Tilting(constValue val, creal_T* a, creal_T* b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < val.w && j < val.w)
	{
		double f_eta = (((double(i) + 1.0) - 1.0) - val.w / 2.0) *
			val.eta_interval;
		double f_xi = val.knumber * ((((double(j) + 1.0) - 1.0) - val.w / 2.0) *
			val.xi_interval * 0.0 + f_eta * 0.0);

		double y_re, y_im;

		if (!f_xi)
		{
			y_re = 1.0;
			y_im = 0.0;
		}
		else
		{
			y_re = nan("");
			y_im = nan("");
		}
		b[i + val.w * j].re = a[i + val.w * j].re * y_re - a[i + val.w * j].im * y_im;
		b[i + val.w * j].im = a[i + val.w * j].re * y_im + a[i + val.w * j].im * y_re;
	}
}


extern "C"
{
	void cuda_Wrapper_Transfer(const int& w, const int& h, constValue val, creal_T* a, creal_T* b)
	{
		dim3 blockSize = dim3(32, 32);
		dim3 gridSize = dim3((w + 32 - 1) / 32, (h + 32 - 1) / 32);
		cudaKernel_Transfer << <gridSize, blockSize >> > (val, a, b);
	}

	void cuda_Wrapper_Tilting(const int& w, const int& h, constValue val, creal_T* a, creal_T* b)
	{
		dim3 blockSize = dim3(32, 32);
		dim3 gridSize = dim3((w + 32 - 1) / 32, (h + 32 - 1) / 32);
		cudaKernel_Tilting << <gridSize, blockSize >> > (val, a, b);
	}
}

#endif // !OphASKernel_cu__