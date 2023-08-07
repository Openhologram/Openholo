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
#include "ophTriMesh_GPU.h"
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
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

__device__  void exponent_complex_meshf(cuFloatComplex* val)
{
	float exp_val = expf(val->x);
	float cos_v;
	float sin_v;
	sincosf(val->y, &sin_v, &cos_v);

	val->x = exp_val * cos_v;
	val->y = exp_val * sin_v;
}


void cudaFFT_Mesh(CUstream_st* stream, int nx, int ny, cufftDoubleComplex* in_field, cufftDoubleComplex* output_field, int direction)
{
	unsigned int nblocks = (nx * ny + kBlockThreads - 1) / kBlockThreads;
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

void cudaFFT_Meshf(CUstream_st* stream, int nx, int ny, cufftComplex* in_field, cufftComplex* output_field, int direction)
{
	unsigned int nblocks = (nx * ny + kBlockThreads - 1) / kBlockThreads;
	int N = nx * ny;
	fftShiftf << <nblocks, kBlockThreads, 0, stream >> > (N, nx, ny, in_field, output_field, false);

	cufftHandle plan;

	// fft
	if (cufftPlan2d(&plan, ny, nx, CUFFT_C2C) != CUFFT_SUCCESS)
	{
		//LOG("FAIL in creating cufft plan");
		return;
	};

	cufftResult result;

	if (direction == -1)
		result = cufftExecC2C(plan, output_field, in_field, CUFFT_FORWARD);
	else
		result = cufftExecC2C(plan, output_field, in_field, CUFFT_INVERSE);

	if (result != CUFFT_SUCCESS)
	{
		//LOG("------------------FAIL: execute cufft, code=%s", result);
		return;
	}

	if (cudaDeviceSynchronize() != cudaSuccess) {
		//LOG("Cuda error: Failed to synchronize\n");
		return;
	}

	fftShiftf << < nblocks, kBlockThreads, 0, stream >> > (N, nx, ny, in_field, output_field, false);

	cufftDestroy(plan);
}

__global__
void cudaKernel_double_RefAS_flat(cufftDoubleComplex* output, const MeshKernelConfig* config,
	double shadingFactor, const geometric* geom, double carrierWaveX, double carrierWaveY, double carrierWaveZ)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < config->pn_X * config->pn_Y) {

		int col = tid % config->pn_X;
		int row = tid / config->pn_X;

		double flx, fly, flz, fx, fy, fz, flxShifted, flyShifted, freqTermX, freqTermY;

		double det = geom->loRot[0] * geom->loRot[3] - geom->loRot[1] * geom->loRot[2];
		if (det == 0)
			return;

		double a = 1 / det;
		double invLoRot[4];
		invLoRot[0] = a * geom->loRot[3];
		invLoRot[1] = -a * geom->loRot[2];
		invLoRot[2] = -a * geom->loRot[1];
		invLoRot[3] = a * geom->loRot[0];

		cuDoubleComplex refTerm1 = make_cuDoubleComplex(0, 0);
		cuDoubleComplex refTerm2 = make_cuDoubleComplex(0, 0);
		cuDoubleComplex refTerm3 = make_cuDoubleComplex(0, 0);
		cuDoubleComplex refAS = make_cuDoubleComplex(0, 0);
		cuDoubleComplex term1 = make_cuDoubleComplex(0, 0);
		cuDoubleComplex term2 = make_cuDoubleComplex(0, 0);

		term1.y = -config->pi2 / config->lambda * (
			carrierWaveX * (geom->glRot[0] * geom->glShift[0] + geom->glRot[3] * geom->glShift[1] + geom->glRot[6] * geom->glShift[2])
			+ carrierWaveY * (geom->glRot[1] * geom->glShift[0] + geom->glRot[4] * geom->glShift[1] + geom->glRot[7] * geom->glShift[2])
			+ carrierWaveZ * (geom->glRot[2] * geom->glShift[0] + geom->glRot[5] * geom->glShift[1] + geom->glRot[8] * geom->glShift[2]));


		// calculate frequency term =======================================================================
		int idxFx = -config->pn_X / 2 + col;
		int idxFy = config->pn_X / 2 - row;
		double w = 1.0 / config->lambda;

		fx = (double)idxFx * config->dfx;
		fy = (double)idxFy * config->dfy;
		fz = sqrt(w * w - fx * fx - fy * fy);

		flx = geom->glRot[0] * fx + geom->glRot[1] * fy + geom->glRot[2] * fz;
		fly = geom->glRot[3] * fx + geom->glRot[4] * fy + geom->glRot[5] * fz;
		flz = sqrt(w * w - flx * flx - fly * fly);


		flxShifted = flx - w * (geom->glRot[0] * carrierWaveX + geom->glRot[1] * carrierWaveY + geom->glRot[2] * carrierWaveZ);
		flyShifted = fly - w * (geom->glRot[3] * carrierWaveX + geom->glRot[4] * carrierWaveY + geom->glRot[5] * carrierWaveZ);
		freqTermX = invLoRot[0] * flxShifted + invLoRot[1] * flyShifted;
		freqTermY = invLoRot[2] * flxShifted + invLoRot[3] * flyShifted;

		double sqFreqTermX = freqTermX * freqTermX;
		double cuFreqTermX = sqFreqTermX * freqTermX;
		double sqFreqTermY = freqTermY * freqTermY;
		double cuFreqTermY = sqFreqTermY * freqTermY;

		//if (freqTermX == -freqTermY && freqTermY != 0) {
		if (abs(freqTermX - freqTermY) <= config->tolerence && abs(freqTermY) > config->tolerence) {
			refTerm1.y = config->pi2 * freqTermY;
			refTerm2.y = 1;

			//refAS = shadingFactor * (((Complex<Real>)1 - exp(refTerm1)) / (4 * pi*pi*freqTermY * freqTermY) + refTerm2 / (2 * pi*freqTermY));
			exponent_complex_mesh(&refTerm1);
			cuDoubleComplex value1 = make_cuDoubleComplex(1, 0);
			cuDoubleComplex value2 = cuCsub(value1, refTerm1);
			double value3 = config->square_pi2 * freqTermY * freqTermY;
			cuDoubleComplex value4 = cuCdiv(value2, make_cuDoubleComplex(value3, 0));
			cuDoubleComplex value5 = cuCdiv(refTerm2, make_cuDoubleComplex(config->pi2 * freqTermY, 0));
			cuDoubleComplex value6 = cuCadd(value4, value5);
			refAS = cuCmul(value6, make_cuDoubleComplex(shadingFactor, 0));

			//}else if (freqTermX == freqTermY && freqTermX == 0) {
		}
		else if (abs(freqTermX - freqTermY) <= config->tolerence && abs(freqTermX) <= config->tolerence) {

			//refAS = shadingFactor * 1 / 2;
			refAS = make_cuDoubleComplex(shadingFactor * 0.5, 0);

			//} else if (freqTermX != 0 && freqTermY == 0) {
		}
		else if (abs(freqTermX) > config->tolerence && abs(freqTermY) <= config->tolerence) {

			refTerm1.y = -config->pi2 * freqTermX;
			refTerm2.y = 1;

			//refAS = shadingFactor * ((exp(refTerm1) - (Complex<Real>)1) / (2 * M_PI*freqTermX * 2 * M_PI*freqTermX) + (refTerm2 * exp(refTerm1)) / (2 * M_PI*freqTermX));
			exponent_complex_mesh(&refTerm1);
			cuDoubleComplex value1 = make_cuDoubleComplex(1, 0);
			cuDoubleComplex value2 = cuCsub(refTerm1, value1);
			double value3 = config->square_pi2 * sqFreqTermX;
			cuDoubleComplex value4 = cuCdiv(value2, make_cuDoubleComplex(value3, 0));

			cuDoubleComplex value5 = cuCmul(refTerm2, refTerm1);
			cuDoubleComplex value6 = cuCdiv(value5, make_cuDoubleComplex(config->pi2 * freqTermX, 0));

			cuDoubleComplex value7 = cuCadd(value4, value6);
			refAS = cuCmul(value7, make_cuDoubleComplex(shadingFactor, 0));

			//} else if (freqTermX == 0 && freqTermY != 0) {
		}
		else if (abs(freqTermX) <= config->tolerence && abs(freqTermY) > config->tolerence) {

			refTerm1.y = config->pi2 * freqTermY;
			refTerm2.y = 1;

			//refAS = shadingFactor * (((Complex<Real>)1 - exp(refTerm1)) / (4 * M_PI*M_PI*freqTermY * freqTermY) - refTerm2 / (2 * M_PI*freqTermY));
			exponent_complex_mesh(&refTerm1);
			cuDoubleComplex value1 = make_cuDoubleComplex(1, 0);
			cuDoubleComplex value2 = cuCsub(value1, refTerm1);
			double value3 = config->square_pi2 * sqFreqTermY;
			cuDoubleComplex value4 = cuCdiv(value2, make_cuDoubleComplex(value3, 0));
			cuDoubleComplex value5 = cuCdiv(refTerm2, make_cuDoubleComplex(config->pi2 * freqTermY, 0));
			cuDoubleComplex value6 = cuCsub(value4, value5);
			refAS = cuCmul(value6, make_cuDoubleComplex(shadingFactor, 0));

		}
		else {

			refTerm1.y = -config->pi2 * freqTermX;
			refTerm2.y = -config->pi2 * (freqTermX + freqTermY);

			//refAS = shadingFactor * ((exp(refTerm1) - (Complex<Real>)1) / (4 * M_PI*M_PI*freqTermX * freqTermY) + ((Complex<Real>)1 - exp(refTerm2)) / (4 * M_PI*M_PI*freqTermY * (freqTermX + freqTermY)));
			exponent_complex_mesh(&refTerm1);
			cuDoubleComplex value1 = make_cuDoubleComplex(1, 0);
			cuDoubleComplex value2 = cuCsub(refTerm1, value1);
			double value3 = config->square_pi2 * freqTermX * freqTermY;
			cuDoubleComplex value4 = cuCdiv(value2, make_cuDoubleComplex(value3, 0));

			exponent_complex_mesh(&refTerm2);
			cuDoubleComplex value5 = cuCsub(make_cuDoubleComplex(1, 0), refTerm2);
			double value6 = config->square_pi2 * freqTermY * (freqTermX + freqTermY);
			cuDoubleComplex value7 = cuCdiv(value5, make_cuDoubleComplex(value6, 0));

			cuDoubleComplex value8 = cuCadd(value4, value7);
			refAS = cuCmul(value8, make_cuDoubleComplex(shadingFactor, 0));
		}

		cuDoubleComplex temp;
		if (abs(fz) <= config->tolerence)
			temp = make_cuDoubleComplex(0, 0);
		else {
			term2.y = config->pi2 * (flx * geom->glShift[0] + fly * geom->glShift[1] + flz * geom->glShift[2]);

			//temp = refAS / det * exp(term1)* flz / fz * exp(term2);

			exponent_complex_mesh(&term1);
			exponent_complex_mesh(&term2);

			cuDoubleComplex tmp1 = cuCdiv(refAS, make_cuDoubleComplex(det, 0));
			cuDoubleComplex tmp2 = cuCmul(tmp1, term1);
			cuDoubleComplex tmp3 = cuCmul(tmp2, make_cuDoubleComplex(flz, 0));
			cuDoubleComplex tmp4 = cuCdiv(tmp3, make_cuDoubleComplex(fz, 0));
			temp = cuCmul(tmp4, term2);

		}

		double absval = sqrt((temp.x * temp.x) + (temp.y * temp.y));
		if (absval > config->min_double)
		{
		}
		else {
			temp = make_cuDoubleComplex(0, 0);
		}

		//cuDoubleComplex addtmp = output[col + row * config->pn_X];
		//output[col+row*config->pn_X] = cuCadd(addtmp,temp);

		output[tid].x += temp.x;
		output[tid].y += temp.y;
	}
}

__global__
void cudaKernel_double_RefAS_continuous(cufftDoubleComplex* output, const MeshKernelConfig* config,
	const geometric* geom, double av0, double av1, double av2, double carrierWaveX, double carrierWaveY, double carrierWaveZ)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < config->pn_X * config->pn_Y) {

		int col = tid % config->pn_X;
		int row = tid / config->pn_X;

		double flx, fly, flz, fx, fy, fz, flxShifted, flyShifted, freqTermX, freqTermY;

		double det = geom->loRot[0] * geom->loRot[3] - geom->loRot[1] * geom->loRot[2];
		if (det == 0)
			return;

		double a = 1 / det;
		double invLoRot[4];
		invLoRot[0] = a * geom->loRot[3];
		invLoRot[1] = -a * geom->loRot[2];
		invLoRot[2] = -a * geom->loRot[1];
		invLoRot[3] = a * geom->loRot[0];

		cuDoubleComplex refTerm1 = make_cuDoubleComplex(0, 0);
		cuDoubleComplex refTerm2 = make_cuDoubleComplex(0, 0);
		cuDoubleComplex refTerm3 = make_cuDoubleComplex(0, 0);
		cuDoubleComplex refAS = make_cuDoubleComplex(0, 0);
		cuDoubleComplex term1 = make_cuDoubleComplex(0, 0);
		cuDoubleComplex term2 = make_cuDoubleComplex(0, 0);

		term1.y = -config->pi2 / config->lambda * (
			carrierWaveX * (geom->glRot[0] * geom->glShift[0] + geom->glRot[3] * geom->glShift[1] + geom->glRot[6] * geom->glShift[2])
			+ carrierWaveY * (geom->glRot[1] * geom->glShift[0] + geom->glRot[4] * geom->glShift[1] + geom->glRot[7] * geom->glShift[2])
			+ carrierWaveZ * (geom->glRot[2] * geom->glShift[0] + geom->glRot[5] * geom->glShift[1] + geom->glRot[8] * geom->glShift[2]));


		// calculate frequency term =======================================================================
		int idxFx = -config->pn_X / 2 + col;
		int idxFy = config->pn_X / 2 - row;
		double w = 1.0 / config->lambda;

		fx = (double)idxFx * config->dfx;
		fy = (double)idxFy * config->dfy;
		fz = sqrt(w * w - fx * fx - fy * fy);

		flx = geom->glRot[0] * fx + geom->glRot[1] * fy + geom->glRot[2] * fz;
		fly = geom->glRot[3] * fx + geom->glRot[4] * fy + geom->glRot[5] * fz;
		flz = sqrt(w * w - flx * flx - fly * fly);


		flxShifted = flx - w * (geom->glRot[0] * carrierWaveX + geom->glRot[1] * carrierWaveY + geom->glRot[2] * carrierWaveZ);
		flyShifted = fly - w * (geom->glRot[3] * carrierWaveX + geom->glRot[4] * carrierWaveY + geom->glRot[5] * carrierWaveZ);
		freqTermX = invLoRot[0] * flxShifted + invLoRot[1] * flyShifted;
		freqTermY = invLoRot[2] * flxShifted + invLoRot[3] * flyShifted;

		double sqFreqTermX = freqTermX * freqTermX;
		double cuFreqTermX = sqFreqTermX * freqTermX;
		double sqFreqTermY = freqTermY * freqTermY;
		double cuFreqTermY = sqFreqTermY * freqTermY;

		cuDoubleComplex D1 = make_cuDoubleComplex(0, 0);
		cuDoubleComplex D2 = make_cuDoubleComplex(0, 0);
		cuDoubleComplex D3 = make_cuDoubleComplex(0, 0);

		//if (freqTermX == 0.0 && freqTermY == 0.0) {
		if (abs(freqTermX) <= config->tolerence && abs(freqTermY) <= config->tolerence) {

			D1.x = (double)1.0 / (double)3.0;
			D2.x = (double)1.0 / (double)5.0;
			D3.x = (double)1.0 / (double)2.0;

			//}else if (freqTermX == 0.0 && freqTermY != 0.0) {
		}
		else if (abs(freqTermX) <= config->tolerence && abs(freqTermY) > config->tolerence) {

			refTerm1.y = -config->pi2 * freqTermY;
			refTerm2.y = 1;

			//D1 = (refTerm1 - (Real)1)*refTerm1.exp() / (8 * M_PI*M_PI*M_PI*freqTermY * freqTermY * freqTermY)
			//	- refTerm1 / (4 * M_PI*M_PI*M_PI*freqTermY * freqTermY * freqTermY);

			cuDoubleComplex refTerm1_exp = make_cuDoubleComplex(refTerm1.x, refTerm1.y);
			exponent_complex_mesh(&refTerm1_exp);
			cuDoubleComplex value1 = make_cuDoubleComplex(1, 0);
			cuDoubleComplex value2 = cuCsub(refTerm1, value1);
			cuDoubleComplex value3 = cuCmul(value2, refTerm1_exp);
			cuDoubleComplex value4 = cuCdiv(value3, make_cuDoubleComplex(config->cube_pi2 * cuFreqTermY, 0));
			cuDoubleComplex value5 = cuCdiv(refTerm1, make_cuDoubleComplex(config->square_pi2 * config->pi * cuFreqTermY, 0));

			D1 = cuCsub(value4, value5);

			//D2 = -(M_PI*freqTermY + refTerm2) / (4 * M_PI*M_PI*M_PI*freqTermY * freqTermY * freqTermY)*exp(refTerm1)
			//	+ refTerm1 / (8 * M_PI*M_PI*M_PI*freqTermY * freqTermY * freqTermY);
			cuDoubleComplex value6 = cuCadd(make_cuDoubleComplex(config->pi * freqTermY, 0), refTerm2);
			cuDoubleComplex value7 = cuCmul(make_cuDoubleComplex(-1, 0), value6);
			cuDoubleComplex value8 = cuCdiv(value7, make_cuDoubleComplex(config->square_pi2 * config->pi * cuFreqTermY, 0));
			cuDoubleComplex value9 = cuCmul(value8, refTerm1_exp);
			cuDoubleComplex value10 = cuCdiv(refTerm1, make_cuDoubleComplex(config->cube_pi2 * cuFreqTermY, 0));
			D2 = cuCadd(value9, value10);

			//D3 = exp(refTerm1) / (2 * M_PI*freqTermY) + ((Real)1 - refTerm2) / (2 * M_PI*freqTermY);
			cuDoubleComplex value11 = cuCdiv(refTerm1_exp, make_cuDoubleComplex(config->pi2 * freqTermY, 0));
			cuDoubleComplex value12 = cuCsub(make_cuDoubleComplex(1, 0), refTerm2);
			cuDoubleComplex value13 = cuCdiv(value12, make_cuDoubleComplex(config->pi2 * freqTermY, 0));

			D3 = cuCadd(value11, value13);

			//} else if (freqTermX != 0.0 && freqTermY == 0.0) {
		}
		else if (abs(freqTermX) > config->tolerence && abs(freqTermY) <= config->tolerence) {

			refTerm1.y = config->square_pi2 * freqTermX * freqTermX;
			refTerm2.y = 1;
			refTerm3.y = config->pi2 * freqTermX;

			//D1 = (refTerm1 + 4 * M_PI*freqTermX - (Real)2 * refTerm2) / (8 * M_PI*M_PI*M_PI*freqTermY * freqTermY * freqTermY)*exp(-refTerm3)
			//	+ refTerm2 / (4 * M_PI*M_PI*M_PI*freqTermX * freqTermX * freqTermX);

			cuDoubleComplex refTerm3_exp = make_cuDoubleComplex(refTerm3.x, refTerm3.y);
			exponent_complex_mesh(&refTerm3_exp);

			cuDoubleComplex value1 = cuCadd(refTerm1, make_cuDoubleComplex(4 * config->pi * freqTermX, 0));
			cuDoubleComplex value2 = cuCmul(make_cuDoubleComplex(2, 0), refTerm2);
			cuDoubleComplex value3 = cuCsub(value1, value2);
			cuDoubleComplex value4 = cuCdiv(value3, make_cuDoubleComplex(config->cube_pi2 * cuFreqTermY, 0));
			cuDoubleComplex value5 = cuCmul(value4, refTerm3_exp);
			cuDoubleComplex value6 = cuCdiv(refTerm2, make_cuDoubleComplex(config->square_pi2 * config->pi * cuFreqTermX, 0));

			D1 = cuCadd(value5, value6);

			//D2 = (Real)1 / (Real)2 * D1;
			D2 = cuCmul(make_cuDoubleComplex(1.0 / 2.0, 0), D1);

			//D3 = ((refTerm3 + (Real)1)*exp(-refTerm3) - (Real)1) / (4 * M_PI*M_PI*freqTermX * freqTermX);
			cuDoubleComplex value7 = cuCadd(refTerm3, make_cuDoubleComplex(1.0, 0));
			cuDoubleComplex value8 = cuCmul(refTerm3, make_cuDoubleComplex(-1.0, 0));
			exponent_complex_mesh(&value8);
			cuDoubleComplex value9 = cuCmul(value7, value8);
			cuDoubleComplex value10 = cuCsub(value9, make_cuDoubleComplex(1.0, 0));
			D3 = cuCdiv(value10, make_cuDoubleComplex(config->square_pi2 * sqFreqTermX, 0));

			//} else if (freqTermX == -freqTermY) {
		}
		else if (abs(freqTermX + freqTermY) <= config->tolerence) {

			refTerm1.y = 1;
			refTerm2.y = config->pi2 * freqTermX;
			refTerm3.y = config->pi2 * config->pi * freqTermX * freqTermX;

			//D1 = (-2 * M_PI*freqTermX + refTerm1) / (8 * M_PI*M_PI*M_PI*freqTermX * freqTermX * freqTermX)*exp(-refTerm2)
			//	- (refTerm3 + refTerm1) / (8 * M_PI*M_PI*M_PI*freqTermX * freqTermX * freqTermX);

			cuDoubleComplex value1 = cuCadd(make_cuDoubleComplex(-config->pi2 * freqTermX, 0), refTerm1);
			cuDoubleComplex value2 = cuCdiv(value1, make_cuDoubleComplex(config->cube_pi2 * cuFreqTermX, 0));
			cuDoubleComplex value3 = cuCmul(refTerm2, make_cuDoubleComplex(-1.0, 0));
			exponent_complex_mesh(&value3);
			cuDoubleComplex value4 = cuCmul(value2, value3);

			cuDoubleComplex value5 = cuCadd(refTerm3, refTerm1);
			cuDoubleComplex value6 = cuCdiv(value5, make_cuDoubleComplex(config->cube_pi2 * cuFreqTermX, 0));

			D1 = cuCsub(value4, value6);

			//D2 = (-refTerm1) / (8 * M_PI*M_PI*M_PI*freqTermX * freqTermX * freqTermX)*exp(-refTerm2)
			//	+ (-refTerm3 + refTerm1 + 2 * M_PI*freqTermX) / (8 * M_PI*M_PI*M_PI*freqTermX * freqTermX * freqTermX);

			cuDoubleComplex value7 = cuCmul(refTerm1, make_cuDoubleComplex(-1.0, 0));
			cuDoubleComplex value8 = cuCdiv(value7, make_cuDoubleComplex(config->cube_pi2 * cuFreqTermX, 0));
			cuDoubleComplex value9 = cuCmul(value8, value3);

			cuDoubleComplex value10 = cuCmul(refTerm3, make_cuDoubleComplex(-1.0, 0));
			cuDoubleComplex value11 = cuCadd(value10, refTerm1);
			cuDoubleComplex value12 = cuCadd(value11, make_cuDoubleComplex(config->pi2 * freqTermX, 0));
			cuDoubleComplex value13 = cuCdiv(value12, make_cuDoubleComplex(config->cube_pi2 * cuFreqTermX, 0));

			D2 = cuCadd(value9, value13);

			//D3 = (-refTerm1) / (4 * M_PI*M_PI*freqTermX * freqTermX)*exp(-refTerm2)
			//	+ (-refTerm2 + (Real)1) / (4 * M_PI*M_PI*freqTermX * freqTermX);

			cuDoubleComplex value14 = cuCdiv(value7, make_cuDoubleComplex(config->square_pi2 * sqFreqTermX, 0));
			cuDoubleComplex value15 = cuCmul(value14, value3);

			cuDoubleComplex value16 = cuCmul(refTerm2, make_cuDoubleComplex(-1.0, 0));
			cuDoubleComplex value17 = cuCadd(value16, make_cuDoubleComplex(1.0, 0));
			cuDoubleComplex value18 = cuCdiv(value17, make_cuDoubleComplex(config->square_pi2 * sqFreqTermX, 0));

			D3 = cuCadd(value15, value18);

		}
		else {

			refTerm1.y = -config->pi2 * (freqTermX + freqTermY);
			refTerm2.y = 1.0;
			refTerm3.y = -config->pi2 * freqTermX;

			//D1 = exp(refTerm1)*(refTerm2 - 2 * M_PI*(freqTermX + freqTermY)) / (8 * M_PI*M_PI*M_PI*freqTermY * (freqTermX + freqTermY)*(freqTermX + freqTermY))
			//	+ exp(refTerm3)*(2 * M_PI*freqTermX - refTerm2) / (8 * M_PI*M_PI*M_PI*freqTermX * freqTermX * freqTermY)
			//	+ ((2 * freqTermX + freqTermY)*refTerm2) / (8 * M_PI*M_PI*M_PI*freqTermX * freqTermX * (freqTermX + freqTermY)*(freqTermX + freqTermY));

			cuDoubleComplex refTerm1_exp = make_cuDoubleComplex(refTerm1.x, refTerm1.y);
			exponent_complex_mesh(&refTerm1_exp);

			double val1 = config->pi2 * (freqTermX + freqTermY);
			cuDoubleComplex value1 = cuCsub(refTerm2, make_cuDoubleComplex(val1, 0));
			cuDoubleComplex value2 = cuCmul(refTerm1_exp, value1);

			double val2 = config->cube_pi2 * freqTermY * (freqTermX + freqTermY) * (freqTermX + freqTermY);
			cuDoubleComplex value3 = cuCdiv(value2, make_cuDoubleComplex(val2, 0));

			cuDoubleComplex refTerm3_exp = make_cuDoubleComplex(refTerm3.x, refTerm3.y);
			exponent_complex_mesh(&refTerm3_exp);

			double val3 = config->pi2 * freqTermX;
			cuDoubleComplex value4 = cuCsub(make_cuDoubleComplex(val3, 0), refTerm2);
			cuDoubleComplex value5 = cuCmul(refTerm3_exp, value4);
			double val4 = config->cube_pi2 * sqFreqTermX * freqTermY;
			cuDoubleComplex value6 = cuCdiv(value5, make_cuDoubleComplex(val4, 0));

			double val5 = 2.0 * freqTermX + freqTermY;
			cuDoubleComplex value7 = cuCmul(make_cuDoubleComplex(val5, 0), refTerm2);
			double val6 = config->cube_pi2 * sqFreqTermX * (freqTermX + freqTermY) * (freqTermX + freqTermY);
			cuDoubleComplex value8 = cuCdiv(value7, make_cuDoubleComplex(val6, 0));

			cuDoubleComplex value9 = cuCadd(value3, value6);
			D1 = cuCadd(value9, value8);

			//D2 = exp(refTerm1)*(refTerm2*(freqTermX + 2 * freqTermY) - 2 * M_PI*freqTermY * (freqTermX + freqTermY)) / (8 * M_PI*M_PI*M_PI*freqTermY * freqTermY * (freqTermX + freqTermY)*(freqTermX + freqTermY))
			//	+ exp(refTerm3)*(-refTerm2) / (8 * M_PI*M_PI*M_PI*freqTermX * freqTermY * freqTermY)
			//	+ refTerm2 / (8 * M_PI*M_PI*M_PI*freqTermX * (freqTermX + freqTermY)* (freqTermX + freqTermY));

			double val7 = freqTermX + 2.0 * freqTermY;
			cuDoubleComplex value10 = cuCmul(refTerm2, make_cuDoubleComplex(val7, 0));
			double val8 = config->pi2 * freqTermY * (freqTermX + freqTermY);
			cuDoubleComplex value11 = cuCsub(value10, make_cuDoubleComplex(val8, 0));
			cuDoubleComplex value12 = cuCmul(refTerm1_exp, value11);
			double val9 = config->cube_pi2 * sqFreqTermY * (freqTermX + freqTermY) * (freqTermX + freqTermY);
			cuDoubleComplex value13 = cuCdiv(value12, make_cuDoubleComplex(val9, 0));

			cuDoubleComplex value14 = cuCmul(refTerm2, make_cuDoubleComplex(-1.0, 0));
			cuDoubleComplex value15 = cuCmul(refTerm3_exp, value14);
			double val10 = config->cube_pi2 * freqTermX * sqFreqTermY;
			cuDoubleComplex value16 = cuCdiv(value15, make_cuDoubleComplex(val10, 0));

			double val11 = config->cube_pi2 * freqTermX * (freqTermX + freqTermY) * (freqTermX + freqTermY);
			cuDoubleComplex value17 = cuCdiv(refTerm2, make_cuDoubleComplex(val11, 0));

			cuDoubleComplex value18 = cuCadd(value13, value16);
			D2 = cuCadd(value18, value17);

			//D3 = -exp(refTerm1) / (4 * M_PI*M_PI*freqTermY * (freqTermX + freqTermY))
			//	+ exp(refTerm3) / (4 * M_PI*M_PI*freqTermX * freqTermY)
			//	- (Real)1 / (4 * M_PI*M_PI*freqTermX * (freqTermX + freqTermY));

			cuDoubleComplex value19 = cuCmul(refTerm1_exp, make_cuDoubleComplex(-1.0, 0));
			double val12 = config->square_pi2 * freqTermY * (freqTermX + freqTermY);
			cuDoubleComplex value20 = cuCdiv(value19, make_cuDoubleComplex(val12, 0));

			double val13 = config->square_pi2 * freqTermX * freqTermY;
			cuDoubleComplex value21 = cuCdiv(refTerm3_exp, make_cuDoubleComplex(val13, 0));

			double val14 = 1.0 / (config->square_pi2 * freqTermX * (freqTermX + freqTermY));
			cuDoubleComplex value22 = make_cuDoubleComplex(val14, 0);

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

		cuDoubleComplex temp;
		if (abs(fz) <= config->tolerence)
			temp = make_cuDoubleComplex(0, 0);
		else {
			term2.y = config->pi2 * (flx * geom->glShift[0] + fly * geom->glShift[1] + flz * geom->glShift[2]);

			//temp = refAS / det * exp(term1)* flz / fz * exp(term2);

			exponent_complex_mesh(&term1);
			exponent_complex_mesh(&term2);

			cuDoubleComplex tmp1 = cuCdiv(refAS, make_cuDoubleComplex(det, 0));
			cuDoubleComplex tmp2 = cuCmul(tmp1, term1);
			cuDoubleComplex tmp3 = cuCmul(tmp2, make_cuDoubleComplex(flz, 0));
			cuDoubleComplex tmp4 = cuCdiv(tmp3, make_cuDoubleComplex(fz, 0));
			temp = cuCmul(tmp4, term2);

		}

		double absval = sqrt((temp.x * temp.x) + (temp.y * temp.y));
		if (absval > config->min_double)
		{
		}
		else {
			temp = make_cuDoubleComplex(0, 0);
		}

		//cuDoubleComplex addtmp = output[col + row * config->pn_X];
		//output[col+row*config->pn_X] = cuCadd(addtmp,temp);

		output[tid].x += temp.x;
		output[tid].y += temp.y;
	}
}

extern "C"
{
	void cudaMesh_Flat(
		const int& nBlocks, const int& nThreads, cufftDoubleComplex* output,
		const MeshKernelConfig* config, double shading_factor, const geometric* geom,
		double carrierWaveX, double carrierWaveY, double carrierWaveZ, CUstream_st* stream)
	{
		cudaKernel_double_RefAS_flat << <nBlocks, nThreads, 0, stream >> > (output, config, shading_factor,
			geom, carrierWaveX, carrierWaveY, carrierWaveZ);
	}

	void cudaMesh_Continuous(
		const int& nBlocks, const int& nThreads, cufftDoubleComplex* output,
		const MeshKernelConfig* config, const geometric* geom, double av0, double av1, double av2,
		double carrierWaveX, double carrierWaveY, double carrierWaveZ, CUstream_st* stream)
	{
		cudaKernel_double_RefAS_continuous << <nBlocks, nThreads, 0, stream >> > (output, config,
			geom, av0, av1, av2, carrierWaveX, carrierWaveY, carrierWaveZ);
	}


	void call_fftGPU(int nx, int ny, cufftDoubleComplex* input, cufftDoubleComplex* output, CUstream_st* streamTriMesh)
	{
		cudaFFT_Mesh(streamTriMesh, nx, ny, input, output, 1);
	}

	void call_fftGPUf(int nx, int ny, cuFloatComplex* input, cuFloatComplex* output, CUstream_st* streamTriMesh)
	{
		cudaFFT_Meshf(streamTriMesh, nx, ny, input, output, 1);
	}
}

#endif // !ophTriMeshKernel_cu__