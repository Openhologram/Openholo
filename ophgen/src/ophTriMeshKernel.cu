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

#ifndef OphTriMeshKernel_cu__
#define OphTriMeshKernel_cu__
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda.h>
#include <vector>
#include <device_launch_parameters.h>
#include <cufft.h>

static const int kBlockThreads = 512;

//% Possible values : [1 0], [-1 0], [0 1], [0 - 1], [1 1], [-1 1], [-1 - 1], [1 - 1]

__global__ void convertToCufftComplex(int N, int nx, int ny, double* save_a_d_, double* save_b_d_, cufftDoubleComplex* in_filed, bool isCrop, int SignalLoc1, int SignalLoc2)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < N)
	{
		int i = tid % nx;
		int j = tid / nx;

		double real_v = save_a_d_[i + j * nx];
		double img_v = save_b_d_[i + j * nx];

		if (isCrop == 0)
		{
			in_filed[tid].x = real_v;
			in_filed[tid].y = img_v;

			return;
		}

		int start_x, end_x, start_y, end_y, xmove, ymove;

		if (SignalLoc2 == 0)
		{
			start_y = 0;
			end_y = ny - 1;
			ymove = 0;

		}
		else {

			start_y = ny / 4 - 1;
			end_y = ny / 2 + ny / 4 - 2;

			if (SignalLoc2 == 1)
				ymove = -(ny / 4 - 1);
			else
				ymove = (ny / 4 + 1);
		}

		if (SignalLoc1 == 0)
		{
			start_x = 0;
			end_x = nx - 1;
			xmove = 0;

		}
		else {

			start_x = nx / 4 - 1;
			end_x = nx / 2 + nx / 4 - 1;

			if (SignalLoc1 == -1)
				xmove = -(nx / 4 - 1);
			else
				xmove = nx / 4 + 1;

		}

		if (i >= start_x && i <= end_x && j >= start_y && j <= end_y)
		{
			int idx = (i + xmove) + (j + ymove)*nx;

			in_filed[idx].x = real_v;
			in_filed[idx].y = img_v;

		}
	}
}

__global__ void cropFringe(int N, int nx, int ny, cufftDoubleComplex* in_filed, cufftDoubleComplex* out_filed, int SignalLoc1, int SignalLoc2)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < N)
	{
		int i = tid % nx;
		int j = tid / nx;

		cufftDoubleComplex src_v = in_filed[i + j * nx];

		int start_x, end_x, start_y, end_y, xmove, ymove;

		if (SignalLoc2 == 0)
		{
			start_y = 0;
			end_y = ny - 1;
			ymove = 0;

		}
		else {

			start_y = ny / 4 - 1;
			end_y = ny / 2 + ny / 4 - 2;

			if (SignalLoc2 == 1)
				ymove = -(ny / 4 - 1);
			else
				ymove = (ny / 4 + 1);
		}

		if (SignalLoc1 == 0)
		{
			start_x = 0;
			end_x = nx - 1;
			xmove = 0;

		}
		else {

			start_x = nx / 4 - 1;
			end_x = nx / 2 + nx / 4 - 1;

			if (SignalLoc1 == -1)
				xmove = -(nx / 4 - 1);
			else
				xmove = nx / 4 + 1;

		}

		if (i >= start_x + xmove && i <= end_x + xmove && j >= start_y + ymove && j <= end_y + ymove)
		{
			int idx = (i - xmove) + (j - ymove)*nx;

			if ((i + xmove) >= nx / 2 - 1 - 2 && (i + xmove) <= nx / 2 - 1 + 2 && (j + ymove) >= ny / 2 - 1 - 2 && (j + ymove) <= ny / 2 - 1 + 2)
			{
				out_filed[idx].x = 0.0;
				out_filed[idx].y = 0.0;

			}
			else
				out_filed[idx] = src_v;

		}
	}
}

__global__ void fftShift(int N, int nx, int ny, cufftDoubleComplex* input, cufftDoubleComplex* output)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	while (tid < N)
	{
		int i = tid % nx;
		int j = tid / nx;

		int ti = i - nx / 2; if (ti < 0) ti += nx;
		int tj = j - ny / 2; if (tj < 0) tj += ny;

		int oindex = tj * nx + ti;

		output[tid].x = input[oindex].x;
		output[tid].y = input[oindex].y;

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void fftShiftDouble(int N, int nx, int ny, cufftDoubleReal* input, cufftDoubleComplex* output)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	while (tid < N)
	{
		int i = tid % nx;
		int j = tid / nx;

		int ti = i - nx / 2; if (ti < 0) ti += nx;
		int tj = j - ny / 2; if (tj < 0) tj += ny;

		int oindex = tj * nx + ti;

		cuDoubleComplex value = make_cuDoubleComplex(input[oindex], 0);

		output[tid] = value;

		tid += blockDim.x * gridDim.x;
	}
}

__device__  void exponent_complex(cuDoubleComplex* val)
{
	double exp_val = exp(val->x);
	double cos_v;
	double sin_v;
	sincos(val->y, &sin_v, &cos_v);

	val->x = exp_val * cos_v;
	val->y = exp_val * sin_v;

}

__device__  double sinc_ft(double v, double pi)
{
	if (v == 0.0)
		return 1.0;

	return (sin(v*pi) / (v*pi));
}


__global__ void polygon_reconstruct_kernel(cufftDoubleComplex* input_term, cufftDoubleComplex* output_term, double temp_z, int nx, int ny, double px, double py, double ss1, double ss2, double lambda, double pi)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	double f_xx, f_yy, f_zz;
	if (tid < nx*ny) {

		int col = tid % nx;
		int row = tid / ny;

		f_xx = ((-1.0 / px) / 2.0) + ((double)col*(1.0 / ss1));
		f_yy = (((1.0 / py) / 2.0) - (1.0 / ss2)) - ((double)row * (1.0 / ss2));
		f_zz = sqrt(1.0 / (lambda*lambda) - f_xx * f_xx - f_yy * f_yy);

		cuDoubleComplex value2 = make_cuDoubleComplex(0, 2 * pi*f_zz*temp_z);
		exponent_complex(&value2);

		cuDoubleComplex value1 = input_term[tid];

		output_term[tid] = cuCmul(value1, value2);

	}
}


__global__ void polygon_sources_kernel(double* real_part_hologram, double* imagery_part_hologram, double* intensities, cufftDoubleComplex* temp_term,
	int vertex_idx, int nx, int ny, double px, double py, double ss1, double ss2, double lambda, double pi, double tolerence,
	double del_fxx, double del_fyy, double f_cx, double f_cy, double f_cz, bool is_multiple_carrier_wave, double cw_amp,
	double t_Coff00, double t_Coff01, double t_Coff02, double t_Coff10, double t_Coff11, double t_Coff12,
	double detAff, double R_31, double R_32, double R_33, double t1, double t2, double t3)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	//double r = 1.0, g = 0.0, b = 0.0;

	double f_xx, f_yy, f_zz, f_ref_x, f_ref_y;
	if (tid < nx*ny) {

		int col = tid % nx;
		int row = tid / ny;

		//if (col != 0 || row != 0)
		//	return;


		double Bx = 1 / px;  double By = 1 / py;
		double wx = px * nx;
		double wy = py * ny;

		f_xx = ((-1.0 / px) / 2.0) + ((double)col*(1.0 / ss1));
		f_yy = (((1.0 / py) / 2.0) - (1.0 / ss2)) - ((double)row * (1.0 / ss2));
		f_zz = sqrt(1.0 / (lambda*lambda) - f_xx * f_xx - f_yy * f_yy);

		//if (f_zz > 1508030.0 && f_zz <= 1508030.9201776853)
		//	g = 1.0;
		//else
		//	r = 1.0;



		f_ref_x = t_Coff00 * (f_xx - f_cx) + t_Coff01 * (f_yy - f_cy) + t_Coff02 * (f_zz - f_cz);
		f_ref_y = t_Coff10 * (f_xx - f_cx) + t_Coff11 * (f_yy - f_cy) + t_Coff12 * (f_zz - f_cz);

		cuDoubleComplex D_1 = make_cuDoubleComplex(0, 0);
		cuDoubleComplex D_2 = make_cuDoubleComplex(0, 0);
		cuDoubleComplex D_3 = make_cuDoubleComplex(0, 0);

		int casenum = 0;
		if (abs(f_ref_x) <= tolerence && abs(f_ref_y) <= tolerence)
		{
			casenum = 1;
			D_1 = make_cuDoubleComplex(1.0 / 3.0, 0);
			D_2 = make_cuDoubleComplex(1.0 / 6.0, 0);
			D_3 = make_cuDoubleComplex(1.0 / 2.0, 0);

		}
		else if (abs(f_ref_x) > tolerence && abs(f_ref_y) <= tolerence)
		{
			casenum = 2;

			cuDoubleComplex value1 = make_cuDoubleComplex(0, -2.0 * pi*f_ref_x);
			exponent_complex(&value1);

			cuDoubleComplex value2 = make_cuDoubleComplex(0, 2 * (pi*pi)*(f_ref_x*f_ref_x));
			cuDoubleComplex value3 = make_cuDoubleComplex(2 * pi*f_ref_x, 0);
			cuDoubleComplex value4 = make_cuDoubleComplex(0, 1);
			cuDoubleComplex value5 = cuCadd(value2, value3);
			cuDoubleComplex value6 = cuCsub(value5, value4);

			double value7 = 4 * (pi * pi * pi)*(f_ref_x * f_ref_x * f_ref_x);

			cuDoubleComplex value8 = make_cuDoubleComplex(4 * (pi *pi*pi)*(f_ref_x * f_ref_x* f_ref_x), 0);
			cuDoubleComplex value9 = cuCdiv(value4, value8);

			cuDoubleComplex value10 = cuCmul(value1, value6);
			value10 = cuCdiv(value10, make_cuDoubleComplex(value7, 0));

			D_1 = cuCadd(value10, value9);

			//exp(-j * 2 * pi*f_ref_x(case_2))  .*		// value1
			//(j * 2 * (pi ^ 2)*(f_ref_x(case_2). ^ 2)   +   2 * pi*f_ref_x(case_2) - j )  . /   // value 6
			//(4 * (pi ^ 3)*(f_ref_x(case_2). ^ 3))  +	// value 7						===> value10
			//j. / (4 * (pi ^ 3).*(f_ref_x(case_2). ^ 3));		// value 9


			value7 = 8 * (pi * pi * pi)*(f_ref_x * f_ref_x * f_ref_x);

			value8 = make_cuDoubleComplex(8 * (pi *pi*pi)*(f_ref_x * f_ref_x* f_ref_x), 0);
			value9 = cuCdiv(value4, value8);

			value10 = cuCmul(value1, value6);
			value10 = cuCdiv(value10, make_cuDoubleComplex(value7, 0));

			D_2 = cuCadd(value10, value9);

			//exp(-j * 2 * pi*f_ref_x(case_2)) .*		// value1
			//(j * 2 * (pi ^ 2)*(f_ref_x(case_2). ^ 2) + 2 * pi*f_ref_x(case_2) - j) . /	// value6
			//(8 * (pi ^ 3)*(f_ref_x(case_2). ^ 3)) +		// value7          ====> value10
			//j. / (8 * (pi ^ 3).*(f_ref_x(case_2). ^ 3));	// value9

			value6 = make_cuDoubleComplex(0, 2 * pi*f_ref_x);
			value6 = cuCadd(make_cuDoubleComplex(1, 0), value6);

			value7 = 4 * (pi *pi)*(f_ref_x* f_ref_x);

			double value11 = 4 * (pi * pi)*(f_ref_x * f_ref_x);
			value11 = -1.0 / value11;

			value10 = cuCmul(value1, value6);
			value10 = cuCdiv(value10, make_cuDoubleComplex(value7, 0));

			D_3 = cuCadd(value10, make_cuDoubleComplex(value11, 0));

			//exp(-j * 2 * pi*f_ref_x(case_2)) .*				// value1
			//(1 + j * 2 * pi*f_ref_x(case_2)) . /			// value6
			//(4 * (pi ^ 2)*(f_ref_x(case_2). ^ 2)) +			// value7    ==> value10
			//(-1). / (4 * (pi ^ 2).*(f_ref_x(case_2). ^ 2));	// value11

		}
		else if (abs(f_ref_x) <= tolerence && abs(f_ref_y) > tolerence)
		{
			casenum = 3;

			cuDoubleComplex value1 = make_cuDoubleComplex(0, -2.0 * pi*f_ref_y);
			exponent_complex(&value1);

			double value2 = -2 * pi*f_ref_y;
			cuDoubleComplex value3 = make_cuDoubleComplex(0, 1);
			value3 = cuCadd(make_cuDoubleComplex(value2, 0), value3);

			double value4 = 8 * (pi * pi * pi)*(f_ref_y* f_ref_y* f_ref_y);

			cuDoubleComplex value5 = make_cuDoubleComplex(0, 2 * (pi*pi)*(f_ref_y* f_ref_y));
			cuDoubleComplex value6 = make_cuDoubleComplex(0, -1);
			cuDoubleComplex value7 = cuCsub(value6, value5);

			double value8 = 8 * (pi*pi*pi)*(f_ref_y*f_ref_y*f_ref_y);

			value7 = cuCdiv(value7, make_cuDoubleComplex(value8, 0));

			cuDoubleComplex value10 = cuCmul(value1, value3);
			value10 = cuCdiv(value10, make_cuDoubleComplex(value4, 0));

			D_1 = cuCadd(value10, value7);

			//exp(-j * 2 * pi*f_ref_y(case_3)) .*				// value1
			//(-2 * pi*f_ref_y(case_3) + j) . /				// value3
			//(8 * (pi ^ 3).*(f_ref_y(case_3). ^ 3))  +		// value4    ===> value10
			//(-j - j * 2 * (pi ^ 2).*(f_ref_y(case_3). ^ 2)) . /		// value7
			//(8 * (pi ^ 3).*(f_ref_y(case_3). ^ 3));			// value8        ===> value7


			value3 = make_cuDoubleComplex(0, 2);
			value3 = cuCadd(make_cuDoubleComplex(value2, 0), value3);

			value10 = cuCmul(value1, value3);
			value10 = cuCdiv(value10, make_cuDoubleComplex(value4, 0));

			value5 = make_cuDoubleComplex(2 * pi*f_ref_y, 0);
			value6 = make_cuDoubleComplex(0, -2);
			value7 = cuCsub(value6, value5);

			value7 = cuCdiv(value7, make_cuDoubleComplex(value8, 0));

			D_2 = cuCadd(value10, value7);

			//exp(-j * 2 * pi*f_ref_y(case_3)) .*				// value1
			//(-2 * pi*f_ref_y(case_3) + 2 * j) . /			// value3
			//(8 * (pi ^ 3).*(f_ref_y(case_3). ^ 3)) +		// value4		===> value10
			//(-2 * j - 2 * pi*f_ref_y(case_3)) . /			// value7
			//(8 * (pi ^ 3).*(f_ref_y(case_3). ^ 3));			// value8   ====> value7


			value3 = make_cuDoubleComplex(-1, 0);
			value4 = 4 * (pi * pi)*(f_ref_y * f_ref_y);
			value10 = cuCmul(value1, value3);
			value10 = cuCdiv(value10, make_cuDoubleComplex(value4, 0));

			value5 = make_cuDoubleComplex(0, 2 * pi*f_ref_y);
			value5 = cuCsub(make_cuDoubleComplex(1, 0), value5);

			value8 = 4 * (pi *pi)*(f_ref_y* f_ref_y);

			value5 = cuCdiv(value5, make_cuDoubleComplex(value8, 0));

			D_3 = cuCadd(value10, value5);

			//exp(-j * 2 * pi*f_ref_y(case_3)) .*		// value1
			//(-1) . /									// value3
			//(4 * (pi ^ 2).*(f_ref_y(case_3). ^ 2)) +	// value4		===> value10
			//(1 - j * 2 * pi*f_ref_y(case_3)) . /		// value5
			//(4 * (pi ^ 2).*(f_ref_y(case_3). ^ 2));		// value8   ===> value5


		}
		else if (abs(f_ref_x) > tolerence && abs(f_ref_y) > tolerence && abs(f_ref_x + f_ref_y) <= tolerence)
		{
			casenum = 4;

			cuDoubleComplex value1 = make_cuDoubleComplex(0, -2.0 * pi*f_ref_x);
			exponent_complex(&value1);

			double value2 = 2 * pi*f_ref_x;
			cuDoubleComplex value3 = make_cuDoubleComplex(0, 1);
			value3 = cuCsub(make_cuDoubleComplex(value2, 0), value3);

			double value4 = 8 * (pi*pi*pi)*(f_ref_x*f_ref_x)*f_ref_y;

			cuDoubleComplex value10 = cuCmul(value1, value3);
			value10 = cuCdiv(value10, make_cuDoubleComplex(value4, 0));

			cuDoubleComplex value5 = make_cuDoubleComplex(0, 2 * (pi*pi)*(f_ref_x*f_ref_x));
			cuDoubleComplex value6 = make_cuDoubleComplex(0, 1);
			cuDoubleComplex value7 = cuCadd(value5, value6);

			double value8 = 8 * (pi*pi*pi)*(f_ref_x*f_ref_x)*f_ref_y;
			value7 = cuCdiv(value7, make_cuDoubleComplex(value8, 0));

			D_1 = cuCadd(value10, value7);

			//exp(-j * 2 * pi*f_ref_x(case_4)) .*						// value1
			//(2 * pi*f_ref_x(case_4) - j) . /							// value3
			//(8 * (pi ^ 3)*(f_ref_x(case_4). ^ 2).*f_ref_y(case_4)) +	// value4    ===> value10
			//(j * 2 * (pi ^ 2).*(f_ref_x(case_4). ^ 2) + j) . /		// value7
			//(8 * (pi ^ 3).*(f_ref_x(case_4). ^ 2).*f_ref_y(case_4));	// value8    ===> value7

			value3 = make_cuDoubleComplex(0, -1);
			value2 = 8 * (pi*pi*pi)*f_ref_x*(f_ref_y*f_ref_y);

			value10 = cuCmul(value1, value3);
			value10 = cuCdiv(value10, make_cuDoubleComplex(value2, 0));

			value5 = make_cuDoubleComplex(0, 2 * (pi*pi)*f_ref_x*f_ref_y);
			value6 = make_cuDoubleComplex(0, 1);
			value4 = 2 * pi*f_ref_x;
			value8 = 8 * (pi*pi*pi)*f_ref_x*(f_ref_y*f_ref_y);

			value5 = cuCadd(value5, make_cuDoubleComplex(value4, 0));
			value7 = cuCadd(value5, value6);
			value7 = cuCdiv(value7, make_cuDoubleComplex(value8, 0));

			D_2 = cuCadd(value10, value7);

			//exp(-j * 2 * pi*f_ref_x(case_4)) .*						// value1
			//(-j) . /													// value3
			//(8 * (pi ^ 3).*f_ref_x(case_4).*(f_ref_y(case_4). ^ 2)) +	// value2		===> value10
			//( j * 2 * (pi ^ 2)*f_ref_x(case_4) .* f_ref_y(case_4) +	// value5
			//  2 * pi*f_ref_x(case_4) +								// value4      
			//  j ) . /													// value6       ===> value7
			//(8 * (pi ^ 3)*f_ref_x(case_4).*(f_ref_y(case_4). ^ 2));	// value8       ===> value7


			value2 = 4 * (pi*pi)*f_ref_x*f_ref_y;

			value1 = cuCdiv(value1, make_cuDoubleComplex(value2, 0));

			value3 = make_cuDoubleComplex(0, 2 * pi*f_ref_x);
			value3 = cuCsub(value3, make_cuDoubleComplex(1, 0));

			value4 = 4 * (pi*pi)*f_ref_x*f_ref_y;

			value3 = cuCdiv(value3, make_cuDoubleComplex(value4, 0));

			D_3 = cuCadd(value1, value3);

			//exp(-j * 2 * pi*f_ref_x(case_4)) . /						// value1
			//(4 * (pi ^ 2)*f_ref_x(case_4).*f_ref_y(case_4)) +			// value2		===> value1
			//(j * 2 * pi*f_ref_x(case_4) - 1) . /						// value3
			//(4 * (pi ^ 2)*f_ref_x(case_4).*f_ref_y(case_4));			// value4       ===> value3

		}
		else if (abs(f_ref_x) > tolerence  && abs(f_ref_y) > tolerence && abs(f_ref_x + f_ref_y) > tolerence)
		{
			casenum = 5;

			cuDoubleComplex value1 = make_cuDoubleComplex(0, -2 * pi*(f_ref_x + f_ref_y));
			exponent_complex(&value1);

			double value2 = 2 * pi*(f_ref_x + f_ref_y);
			cuDoubleComplex value3 = make_cuDoubleComplex(0, 1);
			value3 = cuCsub(value3, make_cuDoubleComplex(value2, 0));

			double value4 = 8 * (pi*pi*pi)*f_ref_y*((f_ref_x + f_ref_y)*(f_ref_x + f_ref_y));

			cuDoubleComplex value5 = make_cuDoubleComplex(0, -2 * pi*f_ref_x);
			exponent_complex(&value5);

			double value6 = 2 * pi*f_ref_x;
			cuDoubleComplex value7 = make_cuDoubleComplex(0, 1);
			value7 = cuCsub(make_cuDoubleComplex(value6, 0), value7);

			double value8 = 8 * (pi*pi*pi)*(f_ref_x * f_ref_x)*f_ref_y;

			cuDoubleComplex value9 = make_cuDoubleComplex(0, 2 * f_ref_x + f_ref_y);

			double value10 = 8 * (pi*pi*pi)*(f_ref_x * f_ref_x);
			double value11 = (f_ref_x + f_ref_y)* (f_ref_x + f_ref_y);

			cuDoubleComplex value12 = cuCmul(value1, value3);
			value12 = cuCdiv(value12, make_cuDoubleComplex(value4, 0));

			cuDoubleComplex value13 = cuCmul(value5, value7);
			value13 = cuCdiv(value13, make_cuDoubleComplex(value8, 0));

			value9 = cuCdiv(value9, make_cuDoubleComplex(value10 *value11, 0));

			D_1 = cuCadd(value12, value13);
			D_1 = cuCadd(D_1, value9);

			//exp(-j * 2 * pi*(f_ref_x(case_5) + f_ref_y(case_5))) .*			// value1
			//(j - 2 * pi*(f_ref_x(case_5) + f_ref_y(case_5))) . /				// value3
			//(8 * (pi ^ 3)*f_ref_y(case_5).*((f_ref_x(case_5) + f_ref_y(case_5)). ^ 2)) +	// value4
			//exp(-j * 2 * pi*f_ref_x(case_5)) .*								// value5
			//(2 * pi*f_ref_x(case_5) - j) . /									// value7
			//(8 * (pi ^ 3)*(f_ref_x(case_5). ^ 2).*f_ref_y(case_5)) +			// value8
			//j*(2 * f_ref_x(case_5) + f_ref_y(case_5)). /						// value9		
			//(  8 * (pi ^ 3)*(f_ref_x(case_5). ^ 2) .*							// value10
			//   ((f_ref_x(case_5) + f_ref_y(case_5)). ^ 2)  );					// value11

			value3 = make_cuDoubleComplex(0, f_ref_x + 2 * f_ref_y);
			value4 = 2 * pi*f_ref_y*(f_ref_x + f_ref_y);
			value3 = cuCsub(value3, make_cuDoubleComplex(value4, 0));

			value6 = 8 * (pi*pi*pi)*(f_ref_y*f_ref_y)*((f_ref_x + f_ref_y)*(f_ref_x + f_ref_y));

			value5 = make_cuDoubleComplex(0, -2 * pi*f_ref_x);
			exponent_complex(&value5);

			value7 = make_cuDoubleComplex(0, -1);

			value8 = 8 * (pi*pi*pi)*f_ref_x*(f_ref_y* f_ref_y);

			value9 = make_cuDoubleComplex(0, 1);
			value10 = 8 * (pi*pi*pi)*f_ref_x*((f_ref_x + f_ref_y)*(f_ref_x + f_ref_y));
			value9 = cuCdiv(value9, make_cuDoubleComplex(value10, 0));

			value12 = cuCmul(value1, value3);
			value12 = cuCdiv(value12, make_cuDoubleComplex(value6, 0));

			value13 = cuCmul(value5, value7);
			value13 = cuCdiv(value13, make_cuDoubleComplex(value8, 0));

			D_2 = cuCadd(value12, value13);
			D_2 = cuCadd(D_2, value9);

			//exp(-j * 2 * pi*(f_ref_x(case_5) + f_ref_y(case_5))) .*					// value1
			//(  j*(f_ref_x(case_5) + 2 * f_ref_y(case_5)) -							// value3 
			//   2 * pi*f_ref_y(case_5).*(f_ref_x(case_5) + f_ref_y(case_5))  ) . /	// value4    ==> value3
			//(8 * (pi ^ 3)*(f_ref_y(case_5). ^ 2).*((f_ref_x(case_5) + f_ref_y(case_5)). ^ 2)) +		// value6   ==> value12
			//exp(-j * 2 * pi*f_ref_x(case_5)) .*								// value5
			//(-j) . /														// value7   
			//(8 * (pi ^ 3).*f_ref_x(case_5).*(f_ref_y(case_5). ^ 2)) +		// value8   ==> value13
			//j. /															// value9
			//(8 * (pi ^ 3).*f_ref_x(case_5).*((f_ref_x(case_5) + f_ref_y(case_5)). ^ 2));	// value10    ==> value9


			value1 = cuCmul(value1, make_cuDoubleComplex(-1, 0));

			value2 = 4 * (pi*pi)*f_ref_y*(f_ref_x + f_ref_y);

			value3 = make_cuDoubleComplex(0, -2 * pi*f_ref_x);
			exponent_complex(&value3);

			value4 = 4 * (pi*pi)*f_ref_x*f_ref_y;

			value6 = 4 * (pi*pi)*f_ref_x*(f_ref_x + f_ref_y);
			value6 = 1 / value6;

			value1 = cuCdiv(value1, make_cuDoubleComplex(value2, 0));

			value3 = cuCdiv(value3, make_cuDoubleComplex(value4, 0));

			D_3 = cuCadd(value1, value3);
			D_3 = cuCsub(D_3, make_cuDoubleComplex(value6, 0));

			//exp(-j * 2 * pi*(f_ref_x(case_5) + f_ref_y(case_5))).*					// value1
			//(-1) . /                                                                              ===> value1
			//(4 * (pi ^ 2).*f_ref_y(case_5).*(f_ref_x(case_5) + f_ref_y(case_5))) +	// value2   ===> value1
			//exp(-j * 2 * pi*f_ref_x(case_5)). /										// value3
			//(4 * (pi ^ 2).*f_ref_x(case_5).*f_ref_y(case_5)) -						// value4   ===> value3
			//1 . / (4 * (pi ^ 2).*f_ref_x(case_5).*(f_ref_x(case_5) + f_ref_y(case_5))); // value6

		}

		cuDoubleComplex temp_U = make_cuDoubleComplex(0, 0);
		cuDoubleComplex sec_term = make_cuDoubleComplex(0, 0);
		cuDoubleComplex third_term = make_cuDoubleComplex(0, 0);
		double first_term = 0.0;

		if (casenum != 0)
		{
			double value1 = (f_xx - f_cx)*t1 + (f_yy - f_cy)*t2 + (f_zz - f_cz)*t3;

			sec_term.y = -2 * pi* value1;
			exponent_complex(&sec_term);
			sec_term = cuCdiv(sec_term, make_cuDoubleComplex(detAff, 0));

			//exp(-j * 2 * pi* ( (f_xx(all_cases) - f_cx)*T(1, 1) + 
			//	               (f_yy(all_cases) - f_cy)*T(2, 1) + 
			//				   (f_zz(all_cases) - f_cz)*T(3, 1) )
			//) / det(Aff);

			double a_v1 = intensities[vertex_idx];
			double a_v2 = intensities[vertex_idx + 1];
			double a_v3 = intensities[vertex_idx + 2];

			cuDoubleComplex tmp1 = cuCmul(make_cuDoubleComplex(a_v2 - a_v1, 0), D_1);
			cuDoubleComplex tmp2 = cuCmul(make_cuDoubleComplex(a_v3 - a_v2, 0), D_2);
			cuDoubleComplex tmp3 = cuCmul(make_cuDoubleComplex(a_v1, 0), D_3);
			third_term = cuCadd(tmp1, tmp2);
			third_term = cuCadd(third_term, tmp3);

			/*
			cuDoubleComplex tmp1 = cuCmul(make_cuDoubleComplex(a_v2 - a_v1, 0), D_1);
			cuDoubleComplex tmp2 = cuCmul(make_cuDoubleComplex(a_v3 - a_v2, 0), D_2);
			cuDoubleComplex tmp3 = cuCmul(make_cuDoubleComplex(a_v1, 0), D_3);
			cuDoubleComplex third_term = cuCadd(tmp1, tmp2);
			third_term = cuCadd(third_term, tmp3);
			*/

		}

		if (is_multiple_carrier_wave == 0)
		{
			if (casenum != 0)
			{
				//(a_v2 - a_v1)*D_1(all_cases) + (a_v3 - a_v2)*D_2(all_cases) + a_v1*D_3(all_cases);

				double f_l_zz = R_31 * f_xx + R_32 * f_yy + R_33 * f_zz;

				//R_(3, 1)*f_xx(all_cases) + R_(3, 2)*f_yy(all_cases) + R_(3, 3)*f_zz(all_cases);

				first_term = f_l_zz / f_zz;

				//f_l_zz(all_cases). / f_zz(all_cases);

				temp_U = make_cuDoubleComplex(cw_amp*first_term, 0);
				temp_U = cuCmul(temp_U, sec_term);

				temp_U = cuCmul(temp_U, third_term);

				//cw_amp*FIRST_TERM(all_cases).*SECOND_TERM(all_cases).*THIRD_TERM(all_cases);
			}

			int index = col + row * nx;


			real_part_hologram[index] = real_part_hologram[index] + temp_U.x;
			imagery_part_hologram[index] = imagery_part_hologram[index] + temp_U.y;

			return;


		}
		else {

			int index = col + row * nx;

			temp_term[index] = cuCmul(sec_term, third_term);

			return;
		}
	}
}

__global__ void translation_sources_kernel(cufftDoubleComplex* temp_term, double* real_part_hologram, double* imagery_part_hologram,
	int nx, int ny, double px, double py, double ss1, double ss2, double lambda,
	int disp_x, int disp_y, double cw_amp, double R_31, double R_32, double R_33)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	double f_xx, f_yy, f_zz;
	if (tid < nx*ny) {

		int col = tid % nx;
		int row = tid / ny;
		int index = col + row * nx;

		int paste_x_start = (1 + disp_x < 1 ? 1 : 1 + disp_x);
		int paste_x_end = (nx + disp_x > nx ? nx : nx + disp_x);
		int paste_y_start = (1 - disp_y < 1 ? 1 : 1 - disp_y);
		int paste_y_end = (ny - disp_y > ny ? ny : ny - disp_y);

		if ((paste_x_end - paste_x_start < 0) || (paste_y_end - paste_y_start < 0))
			return;

		paste_x_start -= 1;
		paste_x_end -= 1;
		paste_y_start -= 1;
		paste_y_end -= 1;

		if (row < paste_y_start || row > paste_y_end || col < paste_x_start || col > paste_x_end)
			return;

		int crop_row = (row + disp_y) < 0 ? 0 : (row + disp_y) >= ny ? ny - 1 : row + disp_y;
		int crop_col = (col - disp_x) < 0 ? 0 : (col - disp_x) >= nx ? nx - 1 : col - disp_x;

		int crop_index = crop_col + crop_row * nx;

		f_xx = ((-1.0 / px) / 2.0) + (col*(1.0 / ss1));
		f_yy = (((1.0 / py) / 2.0) - (1.0 / ss2)) - ((double)row * (1.0 / ss2));
		f_zz = sqrt(1.0 / (lambda*lambda) - f_xx * f_xx - f_yy * f_yy);

		double f_l_zz = R_32 * f_xx + R_32 * f_yy + R_33 * f_zz;
		double first_term = f_l_zz / f_zz;

		//cuDoubleComplex temp_U_r = cuCmul(make_cuDoubleComplex(cw_amp*first_term, 0), temp_term[crop_index]);
		//cuDoubleComplex temp_U_g = cuCmul(make_cuDoubleComplex(cw_amp*first_term, 0), temp_term[crop_index + 1]);
		//cuDoubleComplex temp_U_b = cuCmul(make_cuDoubleComplex(cw_amp*first_term, 0), temp_term[crop_index + 2]);

		cuDoubleComplex temp_U = cuCmul(make_cuDoubleComplex(cw_amp*first_term, 0), temp_term[crop_index]);

		real_part_hologram[index] = real_part_hologram[index] + temp_U.x;

		imagery_part_hologram[index] = imagery_part_hologram[index] + temp_U.y;

	}
}




extern "C"
void cudaFFT(CUstream_st* stream, int N, int nx, int ny, cufftDoubleComplex* in_field, cufftDoubleComplex* output_field, int direction)
{
	cufftHandle plan;

	unsigned int nblocks = (N + kBlockThreads - 1) / kBlockThreads;

	fftShift << <nblocks, kBlockThreads, 0, stream >> > (N, nx, ny, in_field, output_field);

	// fft
	if (cufftPlan2d(&plan, nx, ny, CUFFT_Z2Z) != CUFFT_SUCCESS)
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

	fftShift << < nblocks, kBlockThreads, 0, stream >> > (N, nx, ny, in_field, output_field);

	cufftDestroy(plan);

}

extern "C"
void cudaGetFringeFromGPUKernel(CUstream_st* stream, int N, double* save_a_d_, double* save_b_d_, int nx, int ny, cufftDoubleComplex* in_field, cufftDoubleComplex* output_field, bool isCrop, int SignalLoc1, int SignalLoc2, int direction)
{
	unsigned int nblocks = (N + kBlockThreads - 1) / kBlockThreads;

	convertToCufftComplex << < nblocks, kBlockThreads, 0, stream >> > (N, nx, ny, save_a_d_, save_b_d_, in_field, isCrop, SignalLoc1, SignalLoc2);

	cudaFFT(stream, N, nx, ny, in_field, output_field, direction);
}

extern "C"
void cudaCropFringe(CUstream_st* stream, int N, int nx, int ny, cufftDoubleComplex* in_field, cufftDoubleComplex* out_field, int SignalLoc1, int SignalLoc2)
{
	unsigned int nblocks = (N + kBlockThreads - 1) / kBlockThreads;

	cropFringe << < nblocks, kBlockThreads, 0, stream >> > (N, nx, ny, in_field, out_field, SignalLoc1, SignalLoc2);

}

extern "C"
void cudaPolygonKernel(CUstream_st* stream, int N, double* real_part_hologram, double* imagery_part_hologram, double* intensities, cufftDoubleComplex* temp_term,
	int vertex_idx, int nx, int ny, double px, double py, double ss1, double ss2, double lambda, double pi, double tolerence,
	double del_fxx, double del_fyy, double f_cx, double f_cy, double f_cz, bool is_multiple_carrier_wave, double cw_amp,
	double t_Coff00, double t_Coff01, double t_Coff02, double t_Coff10, double t_Coff11, double t_Coff12,
	double detAff, double R_31, double R_32, double R_33, double T1, double T2, double T3)
{
	dim3 grid((N + kBlockThreads - 1) / kBlockThreads, 1, 1);
	polygon_sources_kernel << <grid, kBlockThreads, 0, stream >> > (real_part_hologram, imagery_part_hologram, intensities, temp_term, vertex_idx, nx, ny, px, py, ss1, ss2, lambda, pi, tolerence,
		del_fxx, del_fyy, f_cx, f_cy, f_cz, is_multiple_carrier_wave, cw_amp, t_Coff00, t_Coff01, t_Coff02, t_Coff10, t_Coff11, t_Coff12,
		detAff, R_31, R_32, R_33, T1, T2, T3);

}

extern "C"
void cudaTranslationMatrixKernel(CUstream_st* stream, int N, cufftDoubleComplex* temp_term, double* real_part_hologram, double* imagery_part_hologram, int nx, int ny, double px, double py, double ss1, double ss2, double lambda,
	int disp_x, int disp_y, double cw_amp, double R_31, double R_32, double R_33)
{

	dim3 grid((N + kBlockThreads - 1) / kBlockThreads, 1, 1);
	translation_sources_kernel << <grid, kBlockThreads, 0, stream >> > (temp_term, real_part_hologram, imagery_part_hologram, nx, ny, px, py, ss1, ss2, lambda,
		disp_x, disp_y, cw_amp, R_31, R_32, R_33);
	   
}



#endif // !OphTriMeshKernel_cu__