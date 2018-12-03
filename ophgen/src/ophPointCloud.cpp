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

#include "ophPointCloud.h"
#include "include.h"

#include <sys.h>
#include <cufft.h>

ophPointCloud::ophPointCloud(void)
	: ophGen()
{
	setMode(false);
	n_points = -1;
}

ophPointCloud::ophPointCloud(const char* pc_file, const char* cfg_file)
	: ophGen()
{
	setMode(false);
	n_points = loadPointCloud(pc_file);
	if (n_points == -1) std::cerr << "OpenHolo Error : Failed to load Point Cloud Data File(*.dat)" << std::endl;

	bool b_read = readConfig(cfg_file);
	if (!b_read) std::cerr << "OpenHolo Error : Failed to load Config Specification Data File(*.config)" << std::endl;
}

ophPointCloud::~ophPointCloud(void)
{
}

void ophPointCloud::setMode(bool is_CPU)
{
	this->is_CPU = is_CPU;
}

int ophPointCloud::loadPointCloud(const char* pc_file)
{
	n_points = ophGen::loadPointCloud(pc_file, &pc_data_);

	return n_points;
}

bool ophPointCloud::readConfig(const char* cfg_file)
{
	if (!ophGen::readConfig(cfg_file, pc_config_))
		return false;

	initialize();

	return true;
}

Real ophPointCloud::generateHologram(uint diff_flag)
{
	auto start_time = CUR_TIME;

	// Create CGH Fringe Pattern by 3D Point Cloud
	if (is_CPU == true) { //Run CPU
#ifdef _OPENMP
		std::cout << "Generate Hologram with Multi Core CPU" << std::endl;
#else
		std::cout << "Generate Hologram with Single Core CPU" << std::endl;
#endif
		genCghPointCloudCPU(diff_flag); /// 홀로그램 데이터 Complex data로 변경 시 (*complex_H)으로
	}
	else { //Run GPU
		std::cout << "Generate Hologram with GPU" << std::endl;

		genCghPointCloudGPU(diff_flag);
	}

	auto end_time = CUR_TIME;

	auto during_time = ((std::chrono::duration<Real>)(end_time - start_time)).count();

	LOG("Implement time : %.5lf sec\n", during_time);

	return during_time;
}

void ophPointCloud::encodeHologram(const vec2 band_limit, const vec2 spectrum_shift)
{
	if ((*complex_H) == nullptr) {
		LOG("Not found diffracted data.");
		return;
	}

	LOG("Single Side Band Encoding..");

	ivec2 pn = context_.pixel_number;
	encode_size = pn;
	vec2 pp = context_.pixel_pitch;
	vec2 ss = context_.ss;

	Real cropx = floor(pn[_X] * band_limit[_X]);
	Real cropx1 = cropx - floor(cropx / 2);
	Real cropx2 = cropx1 + cropx - 1;

	Real cropy = floor(pn[_Y] * band_limit[_Y]);
	Real cropy1 = cropy - floor(cropy / 2);
	Real cropy2 = cropy1 + cropy - 1;

	Real* x_o = new Real[pn[_X]];
	Real* y_o = new Real[pn[_Y]];

	for (int i = 0; i < pn[_X]; i++)
		x_o[i] = (-ss[_X] / 2) + (pp[_X] * i) + (pp[_X] / 2);

	for (int i = 0; i < pn[_Y]; i++)
		y_o[i] = (ss[_Y] - pp[_Y]) - (pp[_Y] * i);

	Real* xx_o = new Real[pn[_X] * pn[_Y]];
	Real* yy_o = new Real[pn[_X] * pn[_Y]];

	for (int i = 0; i < pn[_X] * pn[_Y]; i++)
		xx_o[i] = x_o[i % pn[_X]];


	for (int i = 0; i < pn[_X]; i++)
		for (int j = 0; j < pn[_Y]; j++)
			yy_o[i + j * pn[_X]] = y_o[j];

	Complex<Real>* h = new Complex<Real>[pn[_X] * pn[_Y]];

	fftwShift((*complex_H), h, pn[_X], pn[_Y], OPH_FORWARD);
	fft2(pn, h, OPH_FORWARD);
	fftExecute(h);
	fftwShift(h, h, pn[_X], pn[_Y], OPH_BACKWARD);

	fftwShift(h, h, pn[_X], pn[_Y], OPH_FORWARD);
	fft2(pn, h, OPH_BACKWARD);
	fftExecute(h);
	fftwShift(h, h, pn[_X], pn[_Y], OPH_BACKWARD);

	for (int i = 0; i < pn[_X] * pn[_Y]; i++) {
		Complex<Real> shift_phase(1.0, 0.0);
		int r = i / pn[_X];
		int c = i % pn[_X];

		Real X = (M_PI * xx_o[i] * spectrum_shift[_X]) / pp[_X];
		Real Y = (M_PI * yy_o[i] * spectrum_shift[_Y]) / pp[_Y];

		shift_phase._Val[_RE] = shift_phase._Val[_RE] * (cos(X) * cos(Y) - sin(X) * sin(Y));

		holo_encoded[i] = (h[i] * shift_phase).real();
	}

	delete[] h;
	delete[] x_o;
	delete[] xx_o;
	delete[] y_o;
	delete[] yy_o;

	LOG("Done.\n");
}

void ophPointCloud::encoding(unsigned int ENCODE_FLAG, unsigned int SSB_PASSBAND)
{
	if (ENCODE_FLAG == ENCODE_SSB)  encodeHologram();
	else  ophGen::encoding(ENCODE_FLAG, SSB_PASSBAND);
}

void ophPointCloud::genCghPointCloudCPU(uint diff_flag)
{
	// Output Image Size
	ivec2 pn;
	pn[_X] = context_.pixel_number[_X];
	pn[_Y] = context_.pixel_number[_Y];

	// Tilt Angle
	Real thetaX = RADIAN(pc_config_.tilt_angle[_X]);
	Real thetaY = RADIAN(pc_config_.tilt_angle[_Y]);

	// Wave Number (2 * PI / lambda(wavelength))
	Real k = context_.k;

	// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
	vec2 pp;
	pp[_X] = context_.pixel_pitch[_X];
	pp[_Y] = context_.pixel_pitch[_Y];

	// Length (Width) of complex field at eyepiece plane (by simple magnification)
	vec2 ss;
	ss[_X] = context_.ss[_X];
	ss[_Y] = context_.ss[_Y];

	int j; // private variable for Multi Threading
#ifdef _OPENMP
	int num_threads = 0;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads(); // get number of Multi Threading
#pragma omp for private(j)
#endif
		for (j = 0; j < n_points; ++j) { //Create Fringe Pattern
			uint idx = 3 * j;
			uint color_idx = pc_data_.n_colors * j;
			Real pcx = pc_data_.vertex[idx + _X] * pc_config_.scale[_X];
			Real pcy = pc_data_.vertex[idx + _Y] * pc_config_.scale[_Y];
			Real pcz = pc_data_.vertex[idx + _Z] * pc_config_.scale[_Z] + pc_config_.offset_depth;
			Real amplitude = pc_data_.color[color_idx];

			switch (diff_flag)
			{
			//case PC_DIFF_RS_ENCODED:
			//	diffractEncodedRS(pn, pp, ss, vec3(pcx, pcy, pcz), k, amplitude, vec2(thetaX, thetaY));
			//	break;
			case PC_DIFF_RS/*_NOT_ENCODED*/:
				diffractNotEncodedRS(pn, pp, ss, vec3(pcx, pcy, pcz), k, amplitude, context_.wave_length[0], vec2(thetaX, thetaY));
				break;
			//case PC_DIFF_FRESNEL_ENCODED:
			//	diffractEncodedFrsn();
			//	break;
			case PC_DIFF_FRESNEL/*_NOT_ENCODED*/:
				diffractNotEncodedFrsn(pn, pp, vec3(pcx, pcy, pcz), amplitude, context_.wave_length[0], vec2(thetaX, thetaY));
				break;
			}
		}
#ifdef _OPENMP
	}
	std::cout << ">>> All " << num_threads << " threads" << std::endl;
#endif
}

void ophPointCloud::diffractEncodedRS(ivec2 pn, vec2 pp, vec2 ss, vec3 pc, Real k, Real amplitude, vec2 theta)
{
	for (int yytr = 0; yytr < pn[_Y]; ++yytr)
	{
		for (int xxtr = 0; xxtr < pn[_X]; ++xxtr)
		{
			Real xxx = ((Real)xxtr + 0.5) * pp[_X] - (ss[_X] / 2);
			Real yyy = (ss[_Y] / 2) - ((Real)yytr + 0.5) * pp[_Y];

			Real r = sqrt((xxx - pc[_X]) * (xxx - pc[_X]) + (yyy - pc[_Y]) * (yyy - pc[_Y]) + (pc[_Z] * pc[_Z]));
			Real p = k * (r - xxx * sin(theta[_X]) - yyy * sin(theta[_Y]));
			Real res = amplitude * cos(p);

			holo_encoded[xxtr + yytr * pn[_X]] += res;

			//LOG("(%3d, %3d) [%7d] : ", xxtr, yytr, xxtr + yytr * pn[_X]);
			//LOG("holo=(%15.5lf)\n", holo_encoded[xxtr + yytr * pn[_X]]);
		}
	}
}

void ophPointCloud::diffractNotEncodedRS(ivec2 pn, vec2 pp, vec2 ss, vec3 pc, Real k, Real amplitude, Real lambda, vec2 theta)
{
	Real tx = context_.wave_length[0] / (2 * pp[_X]);
	Real ty = context_.wave_length[0] / (2 * pp[_Y]);

	Real _xbound[2] = {
		pc[_X] + abs(tx / sqrt(1 - (tx * tx)) * pc[_Z]),
		pc[_X] - abs(tx / sqrt(1 - (tx * tx)) * pc[_Z])
	};

	Real _ybound[2] = {
		pc[_Y] + abs(ty / sqrt(1 - (ty * ty)) * pc[_Z]),
		pc[_Y] - abs(ty / sqrt(1 - (ty * ty)) * pc[_Z])
	};

	Real Xbound[2] = {
		floor((_xbound[0] + ss[_X] / 2) / pp[_X]) + 1,
		floor((_xbound[1] + ss[_X] / 2) / pp[_X]) + 1
	};

	Real Ybound[2] = {
		pn[_Y] - floor((_ybound[1] + ss[_Y] / 2) / pp[_Y]),
		pn[_Y] - floor((_ybound[0] + ss[_Y] / 2) / pp[_Y])
	};

	if (Xbound[0] > pn[_X])	Xbound[0] = pn[_X];
	if (Xbound[1] < 0)		Xbound[1] = 0;
	if (Ybound[0] > pn[_Y]) Ybound[0] = pn[_Y];
	if (Ybound[1] < 0)		Ybound[1] = 0;

	for (int xxtr = Xbound[1]; xxtr < Xbound[0]; xxtr++)
	{
		for (int yytr = Ybound[1]; yytr < Ybound[0]; yytr++)
		{
			Real xxx = (-ss[_X] / 2) + ((xxtr - 1) * pp[_X]);
			Real yyy = (-ss[_Y] / 2) + ((pn[_Y] - yytr) * pp[_Y]);

			Real r = sqrt((xxx - pc[_X]) * (xxx - pc[_X]) + (yyy - pc[_Y]) * (yyy - pc[_Y]) + (pc[_Z] * pc[_Z]));

			Real range_x[2] = {
				pc[_X] + abs(tx / sqrt(1 - (tx * tx)) * sqrt((yyy - pc[_Y]) * (yyy - pc[_Y]) + (pc[_Z] * pc[_Z]))),
				pc[_X] - abs(tx / sqrt(1 - (tx * tx)) * sqrt((yyy - pc[_Y]) * (yyy - pc[_Y]) + (pc[_Z] * pc[_Z])))
			};

			Real range_y[2] = {
				pc[_Y] + abs(ty / sqrt(1 - (ty * ty)) * sqrt((xxx - pc[_X]) * (xxx - pc[_X]) + (pc[_Z] * pc[_Z]))),
				pc[_Y] - abs(ty / sqrt(1 - (ty * ty)) * sqrt((xxx - pc[_X]) * (xxx - pc[_X]) + (pc[_Z] * pc[_Z])))
			};

			if (((xxx < range_x[0]) && (xxx > range_x[1])) && ((yyy < range_y[0]) && (yyy > range_y[1]))) {
				Real kr = k * r;

				Real res_real = (amplitude * pc[_Z] * sin(kr)) / (lambda * r * r);
				Real res_imag = (-amplitude * pc[_Z] * cos(kr)) / (lambda * r * r);

				(*complex_H)[xxtr + yytr * pn[_X]][_RE] += res_real;
				(*complex_H)[xxtr + yytr * pn[_X]][_IM] += res_imag;

				//LOG("(%3d, %3d) [%7d] : ", xxtr, yytr, xxtr + yytr * pn[_X]);
				//LOG("holo=(%15.5lf + %20.10lf * i )\n", (*complex_H)[xxtr + yytr * pn[_X]][_RE], (*complex_H)[xxtr + yytr * pn[_X]][_IM]);
			}
		}
	}
}

void ophPointCloud::diffractEncodedFrsn(void)
{
}

void ophPointCloud::diffractNotEncodedFrsn(ivec2 pn, vec2 pp, vec3 pc, Real amplitude, Real lambda, vec2 theta)
{
	Real k = context_.k;
	vec2 ss = context_.ss;

	Real _xbound[2] = {
		pc[_X] + abs(lambda * pc[_Z] / (2 * pp[_X])),
		pc[_X] - abs(lambda * pc[_Z] / (2 * pp[_X]))
	};

	Real _ybound[2] = {
		pc[_Y] + abs(lambda * pc[_Z] / (2 * pp[_Y])),
		pc[_Y] - abs(lambda * pc[_Z] / (2 * pp[_Y]))
	};

	Real Xbound[2] = {
		floor((_xbound[0] + ss[_X] / 2) / pp[_X]) + 1,
		floor((_xbound[1] + ss[_X] / 2) / pp[_X]) + 1
	};

	Real Ybound[2] = {
		pn[_Y] - floor((_ybound[1] + ss[_Y] / 2) / pp[_Y]),
		pn[_Y] - floor((_ybound[0] + ss[_Y] / 2) / pp[_Y])
	};

	if (Xbound[0] > pn[_X])	Xbound[0] = pn[_X];
	if (Xbound[1] < 0)		Xbound[1] = 0;
	if (Ybound[0] > pn[_Y]) Ybound[0] = pn[_Y];
	if (Ybound[1] < 0)		Ybound[1] = 0;

	for (int yytr = Ybound[1]; yytr < Ybound[0]; yytr++)
	{
		for (int xxtr = Xbound[1]; xxtr < Xbound[0]; xxtr++)
		{
			Real xxx = ((-ss[_X]) / 2 + (xxtr - 1) * pp[_X]) - pc[_X];
			Real yyy = ((-ss[_Y]) / 2 + (pn[_Y] - yytr) * pp[_Y]) - pc[_Y];
			Real p = k * (xxx * xxx + yyy * yyy + 2 * pc[_Z] * pc[_Z]) / (2 * pc[_Z]);

			Real res_real = amplitude * sin(p) / (lambda * pc[_Z]);
			Real res_imag = amplitude * (-cos(p)) / (lambda * pc[_Z]);

			(*complex_H)[xxtr + yytr * pn[_X]][_RE] += res_real;
			(*complex_H)[xxtr + yytr * pn[_X]][_IM] += res_imag;

			//LOG("(%3d, %3d) [%7d] : ", xxtr, yytr, xxtr + yytr * pn[_X]);
			//LOG("holo=(%15.5lf + %20.10lf * i )\n", (*complex_H)[xxtr + yytr * pn[_X]][_RE], (*complex_H)[xxtr + yytr * pn[_X]][_IM]);
		}
	}
}

void ophPointCloud::ophFree(void)
{
	delete[] pc_data_.vertex;
	delete[] pc_data_.color;
	delete[] pc_data_.phase;
}