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
#pragma once
#ifndef ophTriMeshKernel_cu__
#define ophTriMeshKernel_cu__

#include "ophKernel.cuh"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cufft.h>
#include <vector>


__device__  void exponent_complex_mesh(cuDoubleComplex* val)
{
	double exp_val = exp(val->x);
	double cos_v;
	double sin_v;
	sincos(val->y, &sin_v, &cos_v);

	val->x = exp_val * cos_v;
	val->y = exp_val * sin_v;
}

void cudaFFT_Mesh(CUstream_st* stream, int nx, int ny, cufftDoubleComplex* in_field, cufftDoubleComplex* output_field, int direction)
{
	unsigned int nblocks = (nx*ny + kBlockThreads - 1) / kBlockThreads;
	int N = nx * ny;
	fftShift << <nblocks, kBlockThreads, 0, stream >> > (N, nx, ny, in_field, output_field, false);

	cufftHandle plan;

	// fft
	if (cufftPlan2d(&plan, ny, nx, CUFFT_Z2Z) != CUFFT_SUCCESS)
	{
		//LOG("FAIL in creating cufft plan");
		return;
	};

	cufftResult result;

	if (direction == -1)
		result = cufftExecZ2Z(plan, output_field, in_field, CUFFT_FORWARD);
	else
		result = cufftExecZ2Z(plan, output_field, in_field, CUFFT_INVERSE);

	if (result != CUFFT_SUCCESS)
	{
		//LOG("------------------FAIL: execute cufft, code=%s", result);
		return;
	}

	if (cudaDeviceSynchronize() != cudaSuccess) {
		//LOG("Cuda error: Failed to synchronize\n");
		return;
	}

	fftShift << < nblocks, kBlockThreads, 0, stream >> > (N, nx, ny, in_field, output_field, false);

	cufftDestroy(plan);
}


__global__ void cudaKernel_refAS(cufftDoubleComplex* output, int nx, int ny, double px, double py, unsigned int sflag, int idx, double waveLength, 
	double pi, double shadingFactor, double av0, double av1, double av2,
	double glRot0, double glRot1, double glRot2, double glRot3, double glRot4, double glRot5, double glRot6, double glRot7, double glRot8,
	double loRot0, double loRot1, double loRot2, double loRot3, double glShiftX, double glShiftY, double glShiftZ,
	double carrierWaveX, double carrierWaveY, double carrierWaveZ, double min_double, double tolerence)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	   	
	if (tid < nx*ny) {

		int col = tid % nx;
		int row = tid / nx;
		
		double flx, fly, flz, fx, fy, fz, flxShifted, flyShifted, freqTermX, freqTermY;
		double dfx = (1.0 / px) / (double)nx;
		double dfy = (1.0 / py) / (double)ny;

		double det = loRot0 * loRot3 - loRot1 * loRot2;
		if (det == 0)
			return;

		
		double invLoRot0, invLoRot1, invLoRot2, invLoRot3;
		invLoRot0 = (1 / det)*loRot3;
		invLoRot1 = -(1 / det)*loRot2;
		invLoRot2 = -(1 / det)*loRot1;
		invLoRot3 = (1 / det)*loRot0;

		cuDoubleComplex refTerm1 = make_cuDoubleComplex(0, 0);
		cuDoubleComplex refTerm2 = make_cuDoubleComplex(0, 0);
		cuDoubleComplex refTerm3 = make_cuDoubleComplex(0, 0);
		cuDoubleComplex refAS = make_cuDoubleComplex(0, 0);
		cuDoubleComplex term1 = make_cuDoubleComplex(0, 0);
		cuDoubleComplex term2 = make_cuDoubleComplex(0, 0);

		term1.y = -2 * pi / waveLength * (
			carrierWaveX * (glRot0 * glShiftX + glRot3 * glShiftY + glRot6 * glShiftZ)
			+ carrierWaveY * (glRot1 * glShiftX + glRot4 * glShiftY + glRot7 * glShiftZ)
			+ carrierWaveZ * (glRot2 * glShiftX + glRot5 * glShiftY + glRot8 * glShiftZ));

		
		// calculate frequency term =======================================================================
		int idxFx = -nx / 2 + col; 
		int idxFy = nx / 2 - row;

		fx = (double)idxFx * dfx;
		fy = (double)idxFy * dfy;
		fz = sqrt((1.0 / waveLength)*(1.0 / waveLength) - fx * fx - fy * fy);

		flx = glRot0 * fx + glRot1 * fy + glRot2 * fz;
		fly = glRot3 * fx + glRot4 * fy + glRot5 * fz;
		flz = sqrt((1.0 / waveLength)*(1.0 / waveLength) - flx * flx - fly * fly);

		flxShifted = flx - (1.0 / waveLength)*(glRot0 * carrierWaveX + glRot1 * carrierWaveY + glRot2 * carrierWaveZ);
		flyShifted = fly - (1.0 / waveLength)*(glRot3 * carrierWaveX + glRot4 * carrierWaveY + glRot5 * carrierWaveZ);

		freqTermX = invLoRot0 * flxShifted + invLoRot1 * flyShifted;
		freqTermY = invLoRot2 * flxShifted + invLoRot3 * flyShifted;
		
		
		//==============================================================================================
		if (sflag == 0) // SHADING_FLAT
		{
			
			//if (freqTermX == -freqTermY && freqTermY != 0) {
			if (abs(freqTermX-freqTermY) <= tolerence && abs(freqTermY) > tolerence) {
				refTerm1.y = 2 * pi *freqTermY;
				refTerm2.y = 1;

				//refAS = shadingFactor * (((Complex<Real>)1 - exp(refTerm1)) / (4 * pi*pi*freqTermY * freqTermY) + refTerm2 / (2 * pi*freqTermY));
				exponent_complex_mesh(&refTerm1);
				cuDoubleComplex value1 = make_cuDoubleComplex(1, 0);
				cuDoubleComplex value2 = cuCsub(value1, refTerm1);
				double value3 = 4 * pi*pi*freqTermY * freqTermY;
				cuDoubleComplex value4 = cuCdiv(value2, make_cuDoubleComplex(value3, 0));
				cuDoubleComplex value5 = cuCdiv(refTerm2, make_cuDoubleComplex(2 * pi*freqTermY, 0));
				cuDoubleComplex value6 = cuCadd(value4, value5);
				refAS = cuCmul(value6, make_cuDoubleComplex(shadingFactor, 0));
			
			//}else if (freqTermX == freqTermY && freqTermX == 0) {
			} else if (abs(freqTermX-freqTermY) <= tolerence && abs(freqTermX) <= tolerence) {

				//refAS = shadingFactor * 1 / 2;
				refAS = make_cuDoubleComplex(shadingFactor*0.5, 0);
			
			//} else if (freqTermX != 0 && freqTermY == 0) {
			} else if (abs(freqTermX) > tolerence && abs(freqTermY) <= tolerence) {
								
				refTerm1.y = -2 * pi*freqTermX;
				refTerm2.y = 1;

				//refAS = shadingFactor * ((exp(refTerm1) - (Complex<Real>)1) / (2 * M_PI*freqTermX * 2 * M_PI*freqTermX) + (refTerm2 * exp(refTerm1)) / (2 * M_PI*freqTermX));
				exponent_complex_mesh(&refTerm1);
				cuDoubleComplex value1 = make_cuDoubleComplex(1, 0);
				cuDoubleComplex value2 = cuCsub(refTerm1, value1);
				double value3 = 2 * pi*freqTermX * 2 * pi*freqTermX;
				cuDoubleComplex value4 = cuCdiv(value2, make_cuDoubleComplex(value3, 0));

				cuDoubleComplex value5 = cuCmul(refTerm2, refTerm1);
				cuDoubleComplex value6 = cuCdiv(value5, make_cuDoubleComplex(2 * pi*freqTermX, 0));

				cuDoubleComplex value7 = cuCadd(value4, value6);
				refAS = cuCmul(value7, make_cuDoubleComplex(shadingFactor, 0));

			//} else if (freqTermX == 0 && freqTermY != 0) {
			} else if (abs(freqTermX) <= tolerence && abs(freqTermY) > tolerence) {

				refTerm1.y = 2 * pi*freqTermY;
				refTerm2.y = 1;
				
				//refAS = shadingFactor * (((Complex<Real>)1 - exp(refTerm1)) / (4 * M_PI*M_PI*freqTermY * freqTermY) - refTerm2 / (2 * M_PI*freqTermY));
				exponent_complex_mesh(&refTerm1);
				cuDoubleComplex value1 = make_cuDoubleComplex(1, 0);
				cuDoubleComplex value2 = cuCsub(value1, refTerm1);
				double value3 = 4 * pi*pi*freqTermY * freqTermY;
				cuDoubleComplex value4 = cuCdiv(value2, make_cuDoubleComplex(value3, 0));
				cuDoubleComplex value5 = cuCdiv(refTerm2, make_cuDoubleComplex(2 * pi*freqTermY, 0));
				cuDoubleComplex value6 = cuCsub(value4, value5);
				refAS = cuCmul(value6, make_cuDoubleComplex(shadingFactor, 0));
		
			} else {

				refTerm1.y = -2 * pi*freqTermX;
				refTerm2.y = -2 * pi*(freqTermX + freqTermY);

				//refAS = shadingFactor * ((exp(refTerm1) - (Complex<Real>)1) / (4 * M_PI*M_PI*freqTermX * freqTermY) + ((Complex<Real>)1 - exp(refTerm2)) / (4 * M_PI*M_PI*freqTermY * (freqTermX + freqTermY)));
				exponent_complex_mesh(&refTerm1);
				cuDoubleComplex value1 = make_cuDoubleComplex(1, 0);
				cuDoubleComplex value2 = cuCsub(refTerm1, value1);
				double value3 = 4 * pi*pi*freqTermX * freqTermY;
				cuDoubleComplex value4 = cuCdiv(value2, make_cuDoubleComplex(value3, 0));

				exponent_complex_mesh(&refTerm2);
				cuDoubleComplex value5 = cuCsub(make_cuDoubleComplex(1, 0), refTerm2);
				double value6 = 4 * pi*pi*freqTermY * (freqTermX + freqTermY);
				cuDoubleComplex value7 = cuCdiv(value5, make_cuDoubleComplex(value6, 0));

				cuDoubleComplex value8 = cuCadd(value4, value7);
				refAS = cuCmul(value8, make_cuDoubleComplex(shadingFactor, 0));


			}
			

		} else if (sflag == 1) {  // SHADING_CONTINUOUS
			
			cuDoubleComplex D1 = make_cuDoubleComplex(0, 0);
			cuDoubleComplex D2 = make_cuDoubleComplex(0, 0);
			cuDoubleComplex D3 = make_cuDoubleComplex(0, 0);
			

			//if (freqTermX == 0.0 && freqTermY == 0.0) {
			if (abs(freqTermX) <= tolerence && abs(freqTermY) <= tolerence) {

				D1.x = (double)1.0 / (double)3.0;
				D2.x = (double)1.0 / (double)5.0;
				D3.x = (double)1.0 / (double)2.0;
			
			//}else if (freqTermX == 0.0 && freqTermY != 0.0) {
			}else if (abs(freqTermX) <= tolerence && abs(freqTermY) > tolerence) {

				refTerm1.y = -2 * pi*freqTermY;
				refTerm2.y = 1;

				//D1 = (refTerm1 - (Real)1)*refTerm1.exp() / (8 * M_PI*M_PI*M_PI*freqTermY * freqTermY * freqTermY)
				//	- refTerm1 / (4 * M_PI*M_PI*M_PI*freqTermY * freqTermY * freqTermY);
				
				cuDoubleComplex refTerm1_exp = make_cuDoubleComplex(refTerm1.x, refTerm1.y);
				exponent_complex_mesh(&refTerm1_exp);
				cuDoubleComplex value1 = make_cuDoubleComplex(1, 0);
				cuDoubleComplex value2 = cuCsub(refTerm1, value1);
				cuDoubleComplex value3 = cuCmul(value2, refTerm1_exp);
				cuDoubleComplex value4 = cuCdiv(value3, make_cuDoubleComplex(8 * pi*pi*pi*freqTermY * freqTermY * freqTermY, 0));
				cuDoubleComplex value5 = cuCdiv(refTerm1, make_cuDoubleComplex(4 * pi*pi*pi*freqTermY * freqTermY * freqTermY, 0));

				D1 = cuCsub(value4, value5);
							   
				//D2 = -(M_PI*freqTermY + refTerm2) / (4 * M_PI*M_PI*M_PI*freqTermY * freqTermY * freqTermY)*exp(refTerm1)
				//	+ refTerm1 / (8 * M_PI*M_PI*M_PI*freqTermY * freqTermY * freqTermY);
				cuDoubleComplex value6 = cuCadd(make_cuDoubleComplex(pi*freqTermY, 0), refTerm2);
				cuDoubleComplex value7 = cuCmul(make_cuDoubleComplex(-1, 0), value6);
				cuDoubleComplex value8 = cuCdiv(value7, make_cuDoubleComplex(4 * pi*pi*pi*freqTermY * freqTermY * freqTermY, 0));
				cuDoubleComplex value9 = cuCmul(value8, refTerm1_exp);
				cuDoubleComplex value10 = cuCdiv(refTerm1, make_cuDoubleComplex(8 * pi*pi*pi*freqTermY * freqTermY * freqTermY, 0));
				D2 = cuCadd(value9, value10);

				//D3 = exp(refTerm1) / (2 * M_PI*freqTermY) + ((Real)1 - refTerm2) / (2 * M_PI*freqTermY);
				cuDoubleComplex value11 = cuCdiv(refTerm1_exp, make_cuDoubleComplex(2 * pi*freqTermY, 0));
				cuDoubleComplex value12 = cuCsub(make_cuDoubleComplex(1, 0), refTerm2);
				cuDoubleComplex value13 = cuCdiv(value12, make_cuDoubleComplex(2 * pi*freqTermY,0));

				D3 = cuCadd(value11, value13);
								
			//} else if (freqTermX != 0.0 && freqTermY == 0.0) {
			} else if (abs(freqTermX) > tolerence && abs(freqTermY) <= tolerence) {

				refTerm1.y = 4 * pi*pi*freqTermX * freqTermX;
				refTerm2.y = 1;
				refTerm3.y = 2 * pi*freqTermX;

				//D1 = (refTerm1 + 4 * M_PI*freqTermX - (Real)2 * refTerm2) / (8 * M_PI*M_PI*M_PI*freqTermY * freqTermY * freqTermY)*exp(-refTerm3)
				//	+ refTerm2 / (4 * M_PI*M_PI*M_PI*freqTermX * freqTermX * freqTermX);

				cuDoubleComplex refTerm3_exp = make_cuDoubleComplex(refTerm3.x, refTerm3.y);
				exponent_complex_mesh(&refTerm3_exp);

				cuDoubleComplex value1 = cuCadd(refTerm1, make_cuDoubleComplex(4 * pi*freqTermX, 0));
				cuDoubleComplex value2 = cuCmul(make_cuDoubleComplex(2, 0), refTerm2);
				cuDoubleComplex value3 = cuCsub(value1, value2);
				cuDoubleComplex value4 = cuCdiv(value3, make_cuDoubleComplex(8 * pi*pi*pi*freqTermY * freqTermY * freqTermY, 0));
				cuDoubleComplex value5 = cuCmul(value4, refTerm3_exp);
				cuDoubleComplex value6 = cuCdiv(refTerm2, make_cuDoubleComplex(4 * pi*pi*pi*freqTermX * freqTermX * freqTermX, 0));

				D1 = cuCadd(value5, value6);

				//D2 = (Real)1 / (Real)2 * D1;
				D2 = cuCmul(make_cuDoubleComplex(1.0 / 2.0, 0), D1);

				//D3 = ((refTerm3 + (Real)1)*exp(-refTerm3) - (Real)1) / (4 * M_PI*M_PI*freqTermX * freqTermX);
				cuDoubleComplex value7 = cuCadd(refTerm3, make_cuDoubleComplex(1.0, 0));
				cuDoubleComplex value8 = cuCmul(refTerm3, make_cuDoubleComplex(-1.0, 0));
				exponent_complex_mesh(&value8);
				cuDoubleComplex value9 = cuCmul(value7, value8);
				cuDoubleComplex value10 = cuCsub(value9, make_cuDoubleComplex(1.0, 0));
				D3 = cuCdiv(value10, make_cuDoubleComplex(4 * pi*pi*freqTermX * freqTermX, 0));

			//} else if (freqTermX == -freqTermY) {
			} else if (abs(freqTermX+freqTermY) <= tolerence ) {

				refTerm1.y = 1;
				refTerm2.y = 2 * pi*freqTermX;
				refTerm3.y = 2 * pi*pi*freqTermX * freqTermX;

				//D1 = (-2 * M_PI*freqTermX + refTerm1) / (8 * M_PI*M_PI*M_PI*freqTermX * freqTermX * freqTermX)*exp(-refTerm2)
				//	- (refTerm3 + refTerm1) / (8 * M_PI*M_PI*M_PI*freqTermX * freqTermX * freqTermX);
				
				cuDoubleComplex value1 = cuCadd(make_cuDoubleComplex(-2 * pi*freqTermX, 0), refTerm1);
				cuDoubleComplex value2 = cuCdiv(value1, make_cuDoubleComplex(8 * pi*pi*pi*freqTermX * freqTermX * freqTermX, 0));
				cuDoubleComplex value3 = cuCmul(refTerm2, make_cuDoubleComplex(-1.0, 0));
				exponent_complex_mesh(&value3);
				cuDoubleComplex value4 = cuCmul(value2, value3);

				cuDoubleComplex value5 = cuCadd(refTerm3, refTerm1);
				cuDoubleComplex value6 = cuCdiv(value5, make_cuDoubleComplex(8 * pi*pi*pi*freqTermX * freqTermX * freqTermX,0));

				D1 = cuCsub(value4, value6);

				//D2 = (-refTerm1) / (8 * M_PI*M_PI*M_PI*freqTermX * freqTermX * freqTermX)*exp(-refTerm2)
				//	+ (-refTerm3 + refTerm1 + 2 * M_PI*freqTermX) / (8 * M_PI*M_PI*M_PI*freqTermX * freqTermX * freqTermX);

				cuDoubleComplex value7 = cuCmul(refTerm1, make_cuDoubleComplex(-1.0, 0));
				cuDoubleComplex value8 = cuCdiv(value7, make_cuDoubleComplex(8 * pi*pi*pi*freqTermX * freqTermX * freqTermX,0));
				cuDoubleComplex value9 = cuCmul(value8, value3);

				cuDoubleComplex value10 = cuCmul(refTerm3, make_cuDoubleComplex(-1.0, 0));
				cuDoubleComplex value11 = cuCadd(value10, refTerm1);
				cuDoubleComplex value12 = cuCadd(value11, make_cuDoubleComplex(2 * pi*freqTermX, 0));
				cuDoubleComplex value13 = cuCdiv(value12, make_cuDoubleComplex(8 * pi*pi*pi*freqTermX * freqTermX * freqTermX,0));

				D2 = cuCadd(value9, value13);

				//D3 = (-refTerm1) / (4 * M_PI*M_PI*freqTermX * freqTermX)*exp(-refTerm2)
				//	+ (-refTerm2 + (Real)1) / (4 * M_PI*M_PI*freqTermX * freqTermX);

				cuDoubleComplex value14 = cuCdiv(value7, make_cuDoubleComplex(4 * pi*pi*freqTermX * freqTermX, 0));
				cuDoubleComplex value15 = cuCmul(value14, value3);

				cuDoubleComplex value16 = cuCmul(refTerm2, make_cuDoubleComplex(-1.0, 0));
				cuDoubleComplex value17 = cuCadd(value16, make_cuDoubleComplex(1.0, 0));
				cuDoubleComplex value18 = cuCdiv(value17, make_cuDoubleComplex(4 * pi*pi*freqTermX * freqTermX, 0));

				D3 = cuCadd(value15, value18);

			} else {

				refTerm1.y = -2.0 * pi*(freqTermX + freqTermY);
				refTerm2.y = 1.0;
				refTerm3.y = -2.0 * pi*freqTermX;

				//D1 = exp(refTerm1)*(refTerm2 - 2 * M_PI*(freqTermX + freqTermY)) / (8 * M_PI*M_PI*M_PI*freqTermY * (freqTermX + freqTermY)*(freqTermX + freqTermY))
				//	+ exp(refTerm3)*(2 * M_PI*freqTermX - refTerm2) / (8 * M_PI*M_PI*M_PI*freqTermX * freqTermX * freqTermY)
				//	+ ((2 * freqTermX + freqTermY)*refTerm2) / (8 * M_PI*M_PI*M_PI*freqTermX * freqTermX * (freqTermX + freqTermY)*(freqTermX + freqTermY));

				cuDoubleComplex refTerm1_exp = make_cuDoubleComplex(refTerm1.x, refTerm1.y);
				exponent_complex_mesh(&refTerm1_exp);

				double val1 = 2.0 * pi*(freqTermX + freqTermY);
				cuDoubleComplex value1 = cuCsub(refTerm2, make_cuDoubleComplex(val1, 0));
				cuDoubleComplex value2 = cuCmul(refTerm1_exp, value1);
				double val2 = 8.0 * pi*pi*pi*freqTermY * (freqTermX + freqTermY)*(freqTermX + freqTermY);
				cuDoubleComplex value3 = cuCdiv(value2, make_cuDoubleComplex(val2,0));

				cuDoubleComplex refTerm3_exp = make_cuDoubleComplex(refTerm3.x, refTerm3.y);
				exponent_complex_mesh(&refTerm3_exp);

				double val3 = 2.0 * pi*freqTermX;
				cuDoubleComplex value4 = cuCsub(make_cuDoubleComplex(val3, 0), refTerm2);
				cuDoubleComplex value5 = cuCmul(refTerm3_exp, value4);
				double val4 = 8.0 * pi*pi*pi*freqTermX * freqTermX * freqTermY;
				cuDoubleComplex value6 = cuCdiv(value5, make_cuDoubleComplex(val4,0));

				double val5 = 2.0 * freqTermX + freqTermY;
				cuDoubleComplex value7 = cuCmul(make_cuDoubleComplex(val5,0), refTerm2);
				double val6 = 8.0 * pi*pi*pi*freqTermX * freqTermX * (freqTermX + freqTermY)*(freqTermX + freqTermY);
				cuDoubleComplex value8 = cuCdiv(value7, make_cuDoubleComplex(val6,0));

				cuDoubleComplex value9 = cuCadd(value3, value6);
				D1 = cuCadd(value9, value8);

				//D2 = exp(refTerm1)*(refTerm2*(freqTermX + 2 * freqTermY) - 2 * M_PI*freqTermY * (freqTermX + freqTermY)) / (8 * M_PI*M_PI*M_PI*freqTermY * freqTermY * (freqTermX + freqTermY)*(freqTermX + freqTermY))
				//	+ exp(refTerm3)*(-refTerm2) / (8 * M_PI*M_PI*M_PI*freqTermX * freqTermY * freqTermY)
				//	+ refTerm2 / (8 * M_PI*M_PI*M_PI*freqTermX * (freqTermX + freqTermY)* (freqTermX + freqTermY));
							   
				double val7 = freqTermX + 2.0 * freqTermY;
				cuDoubleComplex value10 = cuCmul(refTerm2, make_cuDoubleComplex(val7,0));
				double val8 = 2.0 * pi*freqTermY * (freqTermX + freqTermY);
				cuDoubleComplex value11 = cuCsub(value10, make_cuDoubleComplex(val8,0));
				cuDoubleComplex value12 = cuCmul(refTerm1_exp, value11);
				double val9 = 8.0 * pi*pi*pi*freqTermY * freqTermY * (freqTermX + freqTermY)*(freqTermX + freqTermY);
				cuDoubleComplex value13 = cuCdiv(value12, make_cuDoubleComplex(val9,0));
				
				cuDoubleComplex value14 = cuCmul(refTerm2, make_cuDoubleComplex(-1.0, 0));
				cuDoubleComplex value15 = cuCmul(refTerm3_exp, value14);
				double val10 = 8.0 * pi*pi*pi*freqTermX * freqTermY * freqTermY;
				cuDoubleComplex value16 = cuCdiv(value15, make_cuDoubleComplex(val10,0));

				double val11 = 8.0 * pi*pi*pi*freqTermX * (freqTermX + freqTermY)* (freqTermX + freqTermY);
				cuDoubleComplex value17 = cuCdiv(refTerm2, make_cuDoubleComplex(val11,0));

				cuDoubleComplex value18 = cuCadd(value13, value16);
				D2 = cuCadd(value18, value17);
				
				//D3 = -exp(refTerm1) / (4 * M_PI*M_PI*freqTermY * (freqTermX + freqTermY))
				//	+ exp(refTerm3) / (4 * M_PI*M_PI*freqTermX * freqTermY)
				//	- (Real)1 / (4 * M_PI*M_PI*freqTermX * (freqTermX + freqTermY));

				cuDoubleComplex value19 = cuCmul(refTerm1_exp, make_cuDoubleComplex(-1.0, 0));
				double val12 = 4.0 * pi*pi*freqTermY * (freqTermX + freqTermY);
				cuDoubleComplex value20 = cuCdiv(value19, make_cuDoubleComplex(val12,0));

				double val13 = 4.0 * pi*pi*freqTermX * freqTermY;
				cuDoubleComplex value21 = cuCdiv(refTerm3_exp, make_cuDoubleComplex(val13,0));

				double val14 = 1.0 / (4.0 * pi*pi*freqTermX * (freqTermX + freqTermY));
				cuDoubleComplex value22 = make_cuDoubleComplex(val14,0);

				cuDoubleComplex value23 = cuCadd(value20, value21);
				D3 = cuCsub(value23, value22);

			}

			//refAS = (av1 - av0)*D1 + (av2 - av1)*D2 + av0 * D3;
	
			double t1 = av1 - av0;
			double t2 = av2 - av1;
			cuDoubleComplex value_temp1 = cuCmul(make_cuDoubleComplex(t1, 0), D1);
			cuDoubleComplex value_temp2 = cuCmul(make_cuDoubleComplex(t2, 0), D2);
			cuDoubleComplex value_temp3 = cuCmul(make_cuDoubleComplex(av0, 0), D3);

			cuDoubleComplex valeF = cuCadd(value_temp1, value_temp2);
			refAS = cuCadd(valeF, value_temp3);
	
		}
			
		cuDoubleComplex temp;
		if (abs(fz) <= tolerence)
			temp = make_cuDoubleComplex(0, 0);
		else {
			term2.y = 2.0 * pi*(flx * glShiftX + fly * glShiftY + flz * glShiftZ);

			//temp = refAS / det * exp(term1)* flz / fz * exp(term2);

			exponent_complex_mesh(&term1);
			exponent_complex_mesh(&term2);

			cuDoubleComplex tmp1 = cuCdiv(refAS, make_cuDoubleComplex(det,0));
			cuDoubleComplex tmp2 = cuCmul(tmp1, term1);
			cuDoubleComplex tmp3 = cuCmul(tmp2, make_cuDoubleComplex(flz, 0));
			cuDoubleComplex tmp4 = cuCdiv(tmp3, make_cuDoubleComplex(fz, 0));
			temp = cuCmul(tmp4, term2);
			
		}

		double absval = sqrt((temp.x*temp.x) + (temp.y*temp.y));
		if (absval > min_double)
		{
		} else { 
			temp = make_cuDoubleComplex(0, 0); 
		}

		//cuDoubleComplex addtmp = output[col + row * nx];
		//output[col+row*nx] = cuCadd(addtmp,temp);

		output[col + row * nx].x = output[col + row * nx].x + temp.x;
		output[col + row * nx].y = output[col + row * nx].y + temp.y;
		
	}

}

extern "C"
void call_cudaKernel_refAS(cufftDoubleComplex* output, int nx, int ny, double px, double py, unsigned int sflag, int idx, double waveLength, 
	double pi, double shadingFactor, double av0, double av1, double av2,
	double glRot0, double glRot1, double glRot2, double glRot3, double glRot4, double glRot5, double glRot6, double glRot7, double glRot8,
	double loRot0, double loRot1, double loRot2, double loRot3, double glShiftX, double glShiftY, double glShiftZ,
	double carrierWaveX, double carrierWaveY, double carrierWaveZ, double min_double, double tolerence, CUstream_st* streamTriMesh)
{
	dim3 grid((nx*ny + kBlockThreads - 1) / kBlockThreads, 1, 1);
	cudaKernel_refAS << <grid, kBlockThreads, 0, streamTriMesh >> > (output, nx, ny, px, py, sflag, idx, waveLength, pi, shadingFactor, av0, av1, av2,
		glRot0, glRot1, glRot2, glRot3, glRot4, glRot5, glRot6, glRot7, glRot8,
		loRot0, loRot1, loRot2, loRot3, glShiftX, glShiftY, glShiftZ,
		carrierWaveX, carrierWaveY, carrierWaveZ, min_double, tolerence);

	   	 
}

extern "C"
void call_fftGPU(int nx, int ny, cufftDoubleComplex* input, cufftDoubleComplex* output, CUstream_st* streamTriMesh)
{
	
	cudaFFT_Mesh(streamTriMesh, nx, ny, input, output, 1);
	   
}

#endif // !ophTriMeshKernel_cu__