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
* @file		ophPASKernel.cu
* @brief	Openholo Phase Added Stereogram with CUDA GPGPU
* @author	Minwoo Nam
* @date		2023/03
*/

#ifndef OphPASKernel_cu__
#define OphPASKernel_cu__

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "typedef.h"
#include "ophPAS_GPU.h"


/**
@fn __global__ void phaseCalc(float* inRe, float* inIm, constValue val)
@brief CalcCompensatedPhase의 GPU버전 함수
@return void
@param inRe
@param inIm
*/
__global__
void cudaKernel_phaseCalc(float* inRe, float* inIm, constValue val, int c_x, int c_y, int c_z, int amplitude, 
							int sex, int sey, int sen)
{
	int segy = blockIdx.x * blockDim.x + threadIdx.x;
	int segx = blockIdx.y * blockDim.y + threadIdx.y;// coordinate in a Segment 
	int segX = sex;
	int segY = sey;

	if ((segy < sey) && (segx < sex))
	{
		int		segxx, segyy;
		float	theta_s, theta_c;
		int		dtheta_s, dtheta_c;
		int		idx_c, idx_s;
		float	theta;
		int segNo = sen;
		int tbl = 1024;
		float amp = amplitude;
		float pi = 3.14159265358979323846f;
		float m2_pi = (float)(pi * 2.0);
		float rWaveNum = 9926043.13930423;// _CGHE->rWaveNumber;
		float R;
		int cf_cx = val.cf_cx[segx];
		int cf_cy = val.cf_cy[segy];
		float xc = val.xc[segx];
		float yc = val.yc[segy];
		segyy = segy * segX + segx;
		segxx = cf_cy * segNo + cf_cx;
		R = (float)(sqrt((xc - c_x) * (xc - c_x) + (yc - c_y) * (yc - c_y) + c_z * c_z));
		theta = rWaveNum * R;
		theta_c = theta;
		theta_s = theta + pi;
		dtheta_c = ((int)(theta_c * tbl / (pi * 2.0)));
		dtheta_s = ((int)(theta_s * tbl / (pi * 2.0)));
		idx_c = (dtheta_c) & (tbl - 1);
		idx_s = (dtheta_s) & (tbl - 1);
		float costbl = val.costbl[idx_c];
		float sintbl = val.sintbl[idx_s];
		atomicAdd(&inRe[segyy * segNo * segNo + segxx], (float)(amplitude * costbl));
		atomicAdd(&inIm[segyy * segNo * segNo + segxx], (float)(amplitude * sintbl));

		/*
		inRe[segyy*sen*sen + segxx]+= (float)(amplitude * costbl);
		inIm[segyy*sen*sen + segxx]+= (float)(amplitude * sintbl);
		*/
	}
}

extern "C"
{
	void cuda_Wrapper_phaseCalc(float* inRe, float* inIm, constValue val, float& cx, float&cy,
		float&cz, float& amp, ivec3& seg)
	{
		dim3 blockSize(seg[_Y] / 32 + 1, seg[_X] / 32 + 1);
		dim3 gridSize(32, 32);

		cudaKernel_phaseCalc << <gridSize, blockSize >> > (inRe, inIm, val, cx, cy, cz, amp, seg[_X], seg[_Y], seg[_Z]);
	}
}

#endif // !OphPASKernel_cu__