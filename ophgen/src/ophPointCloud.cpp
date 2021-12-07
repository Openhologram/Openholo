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
#include "tinyxml2.h"
#include <sys.h>
#include <cufft.h>

ophPointCloud::ophPointCloud(void)
	: ophGen()
	, is_CPU(true)
	, is_ViewingWindow(false)
	, m_nProgress(0)
	, n_points(-1)
	, bSinglePrecision(false)
{
	LOG("*** POINT CLOUD : BUILD DATE: %s %s ***\n\n", __DATE__, __TIME__);
}

ophPointCloud::ophPointCloud(const char* pc_file, const char* cfg_file)
	: ophGen()
	, is_CPU(true)
	, is_ViewingWindow(false)
	, m_nProgress(0)
{
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

void ophPointCloud::setViewingWindow(bool is_ViewingWindow)
{
	this->is_ViewingWindow = is_ViewingWindow;
}

int ophPointCloud::loadPointCloud(const char* pc_file)
{
	n_points = ophGen::loadPointCloud(pc_file, &pc_data_);

	return n_points;
}

bool ophPointCloud::readConfig(const char* fname)
{
	if (!ophGen::readConfig(fname))
		return false;

	LOG("Reading....%s...", fname);

	auto start = CUR_TIME;

	using namespace tinyxml2;
	/*XML parsing*/
	tinyxml2::XMLDocument xml_doc;
	XMLNode *xml_node;

	if (!checkExtension(fname, ".xml"))
	{
		LOG("file's extension is not 'xml'\n");
		return false;
	}
	if (xml_doc.LoadFile(fname) != XML_SUCCESS)
	{
		LOG("Failed to load file \"%s\"\n", fname);
		return false;
	}

	xml_node = xml_doc.FirstChild();
	// about point
	auto next = xml_node->FirstChildElement("ScaleX");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&pc_config_.scale[_X]))
		return false;
	next = xml_node->FirstChildElement("ScaleY");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&pc_config_.scale[_Y]))
		return false;
	next = xml_node->FirstChildElement("ScaleZ");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&pc_config_.scale[_Z]))
		return false;
	next = xml_node->FirstChildElement("Distance");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&pc_config_.distance))
		return false;

	auto end = CUR_TIME;
	auto during = ((chrono::duration<Real>)(end - start)).count();
	LOG("%lf (s)..done\n", during);

	initialize();
	return true;
}

Real ophPointCloud::generateHologram(uint diff_flag)
{
	if (diff_flag < PC_DIFF_RS || diff_flag > PC_DIFF_FRESNEL) {
		LOG("Wrong Diffraction Method.\n");
		return 0.0;
	}

	resetBuffer();
	auto begin = CUR_TIME;
	LOG("1) Algorithm Method : Point Cloud\n");
	LOG("2) Generate Hologram with %s\n", m_mode & MODE_GPU ?
		"GPU" :
#ifdef _OPENMP
		"Multi Core CPU"
#else
		"Single Core CPU"
#endif
		);
	//LOG("3) Transform Viewing Window : %s\n", is_ViewingWindow ? "ON" : "OFF");
	LOG("3) Diffraction Method : %s\n", diff_flag == PC_DIFF_RS ? "R-S" : "Fresnel");
	LOG("4) Number of Point Cloud : %d\n", n_points);
	LOG("5) Precision Level : %s\n", m_mode & MODE_FLOAT ? "Single" : "Double");
	if(m_mode & MODE_GPU)
		LOG("6) Use FastMath : %s\n", m_mode & MODE_FASTMATH ? "Y" : "N");

	// Create CGH Fringe Pattern by 3D Point Cloud
	if (m_mode & MODE_GPU) { //Run GPU
		genCghPointCloudGPU(diff_flag);
	}
	else { //Run CPU
		genCghPointCloudCPU(diff_flag);
	}

	m_nProgress = 0;
	auto end = CUR_TIME;
	m_elapsedTime = ((std::chrono::duration<Real>)(end - begin)).count();
	LOG("Total Elapsed Time: %lf (s)\n", m_elapsedTime);
	return m_elapsedTime;
}

void ophPointCloud::encodeHologram(const vec2 band_limit, const vec2 spectrum_shift)
{
	if (complex_H == nullptr) {
		LOG("Not found diffracted data.");
		return;
	}

	LOG("Single Side Band Encoding..");
	const uint nChannel = context_.waveNum;
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];
	const uint pnXY = pnX * pnY;

	m_vecEncodeSize = ivec2(pnX, pnY);
	context_.ss[_X] = pnX * ppX;
	context_.ss[_Y] = pnY * ppY;
	vec2 ss = context_.ss;

	Real cropx = floor(pnX * band_limit[_X]);
	Real cropx1 = cropx - floor(cropx / 2);
	Real cropx2 = cropx1 + cropx - 1;

	Real cropy = floor(pnY * band_limit[_Y]);
	Real cropy1 = cropy - floor(cropy / 2);
	Real cropy2 = cropy1 + cropy - 1;

	Real* x_o = new Real[pnX];
	Real* y_o = new Real[pnY];

	for (int i = 0; i < pnX; i++)
		x_o[i] = (-ss[_X] / 2) + (ppX * i) + (ppX / 2);

	for (int i = 0; i < pnY; i++)
		y_o[i] = (ss[_Y] - ppY) - (ppY * i);

	Real* xx_o = new Real[pnXY];
	Real* yy_o = new Real[pnXY];

	for (int i = 0; i < pnXY; i++)
		xx_o[i] = x_o[i % pnX];


	for (int i = 0; i < pnX; i++)
		for (int j = 0; j < pnY; j++)
			yy_o[i + j * pnX] = y_o[j];

	Complex<Real>* h = new Complex<Real>[pnXY];

	for (uint ch = 0; ch < nChannel; ch++) {
#if 1
		fft2(ivec2(pnX, pnY), complex_H[ch], OPH_FORWARD);
		fft2(complex_H[ch], h, pnX, pnY, OPH_FORWARD);
		fft2(ivec2(pnX, pnY), h, OPH_BACKWARD);
		fft2(h, h, pnX, pnY, OPH_BACKWARD);
#else
		fft2(complex_H[ch], h, pnX, pnY, OPH_FORWARD);
		fft2(ivec2(pnX, pnY), h, OPH_FORWARD);
		fftExecute(h);
		fft2(h, h, pnX, pnY, OPH_BACKWARD);

		fft2(h, h, pnX, pnY, OPH_FORWARD);
		fft2(ivec2(pnX, pnY), h, OPH_BACKWARD);
		fftExecute(h);
		fft2(h, h, pnX, pnY, OPH_BACKWARD);
#endif
		for (int i = 0; i < pnXY; i++) {
			Complex<Real> shift_phase(1.0, 0.0);
			int r = i / pnX;
			int c = i % pnX;

			Real X = (M_PI * xx_o[i] * spectrum_shift[_X]) / ppX;
			Real Y = (M_PI * yy_o[i] * spectrum_shift[_Y]) / ppY;

			shift_phase[_RE] = shift_phase[_RE] * (cos(X) * cos(Y) - sin(X) * sin(Y));

			m_lpEncoded[ch][i] = (h[i] * shift_phase).real();
		}
	}
	delete[] h;
	delete[] x_o;
	delete[] xx_o;
	delete[] y_o;
	delete[] yy_o;

	LOG("Done.\n");
}

void ophPointCloud::encoding(unsigned int ENCODE_FLAG)
{
	ophGen::encoding(ENCODE_FLAG);
}

void ophPointCloud::encoding(unsigned int ENCODE_FLAG, unsigned int SSB_PASSBAND)
{
	if (ENCODE_FLAG == ENCODE_SSB)  encodeHologram();
	else if (ENCODE_FLAG == ENCODE_OFFSSB) ophGen::encoding(ENCODE_FLAG, SSB_PASSBAND);
}


Real ophPointCloud::genCghPointCloudCPU(uint diff_flag)
{
	auto begin = CUR_TIME;

	// Output Image Size
	ivec2 pn;
	pn[_X] = context_.pixel_number[_X];
	pn[_Y] = context_.pixel_number[_Y];

	// Tilt Angle
	Real thetaX = RADIAN(pc_config_.tilt_angle[_X]);
	Real thetaY = RADIAN(pc_config_.tilt_angle[_Y]);

	// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
	vec2 pp;
	pp[_X] = context_.pixel_pitch[_X];
	pp[_Y] = context_.pixel_pitch[_Y];

	// Length (Width) of complex field at eyepiece plane (by simple magnification)
	vec2 ss;
	ss[_X] = context_.ss[_X] = pn[_X] * pp[_X];
	ss[_Y] = context_.ss[_Y] = pn[_Y] * pp[_Y];

	uint nChannel = context_.waveNum;

	bool bIsGrayScale = pc_data_.n_colors == 1 ? true : false;

	int i; // private variable for Multi Threading
	int num_threads = 1;
	int sum = 0;
	m_nProgress = 0;

	Real *pVertex = nullptr;
	if (is_ViewingWindow) {
		pVertex = new Real[n_points * 3];
		transVW(n_points * 3, pVertex, pc_data_.vertex);
	}
	else {
		pVertex = pc_data_.vertex;
	}
	
	for (uint ch = 0; ch < nChannel; ++ch) {
		// Wave Number (2 * PI / lambda(wavelength))
		Real lambda = context_.wave_length[ch];
		Real k = context_.k = (2 * M_PI / lambda);

		Real ratio = context_.wave_length[nChannel - 1] / context_.wave_length[ch];

		uint nAdd = bIsGrayScale ? 0 : ch;
#ifdef _OPENMP
#pragma omp parallel for private(i) firstprivate(nAdd, lambda)
#endif
		for (i = 0; i < n_points; ++i) { //Create Fringe Pattern
			uint iVertex = 3 * i; // x, y, z
			uint iColor = pc_data_.n_colors * i + nAdd; // rgb or gray-scale
			Real pcx, pcy, pcz;

			pcx = pVertex[iVertex + _X];
			pcy = pVertex[iVertex + _Y];
			pcz = pVertex[iVertex + _Z];
			pcx *= pc_config_.scale[_X];
			pcy *= pc_config_.scale[_Y];
			pcz *= pc_config_.scale[_Z];
			//pcx *= ratio;
			//pcy *= ratio;
#if 0
			pcz += pc_config_.distance;

			Real amplitude = pc_data_.color[iColor];

			switch (diff_flag)
			{
			case PC_DIFF_RS:
				diffractNotEncodedRS(ch, pn, pp, ss, vec3(pcx, pcy, pcz), k, amplitude, lambda);
#else
			Real amplitude = pc_data_.color[iColor];
			switch (diff_flag)
			{
			case PC_DIFF_RS:
				RS_Diffraction(vec3(pcx, pcy, pcz), complex_H[ch], lambda, pc_config_.distance, amplitude);
#endif
				break;
			case PC_DIFF_FRESNEL:
				diffractNotEncodedFrsn(ch, pn, pp, ss, vec3(pcx, pcy, pcz), k, amplitude, lambda);
				break;
			}
#ifdef _OPENMP
#pragma omp atomic
#endif
			sum++;

			m_nProgress = (int)((Real)sum * 100 / ((Real)n_points * nChannel));
		}
	}
	if (is_ViewingWindow) {
		delete[] pVertex;
	}
	auto end = CUR_TIME;
	Real elapsed_time = ((chrono::duration<Real>)(end - begin)).count();
	LOG("\n%s : %lf(s) <%d threads>\n\n",
		__FUNCTION__,
		elapsed_time,
		num_threads);

	return elapsed_time;
}

void ophPointCloud::diffractEncodedRS(uint channel, ivec2 pn, vec2 pp, vec2 ss, vec3 pc, Real k, Real amplitude, vec2 theta)
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

			m_lpEncoded[channel][xxtr + yytr * pn[_X]] += res;
		}
	}
}

void ophPointCloud::diffractNotEncodedRS(uint channel, ivec2 pn, vec2 pp, vec2 ss, vec3 pc, Real k, Real amplitude, Real lambda)
{
	// for performance
	Real tx = lambda / (2 * pp[_X]);
	Real ty = lambda / (2 * pp[_Y]);
	Real sqrtX = sqrt(1 - (tx * tx));
	Real sqrtY = sqrt(1 - (ty * ty));
	Real x = -ss[_X] / 2;
	Real y = -ss[_Y] / 2;
	Real zz = pc[_Z] * pc[_Z];
	Real ampZ = amplitude * pc[_Z];

	Real _xbound[2] = {
		pc[_X] + abs(tx / sqrtX * pc[_Z]),
		pc[_X] - abs(tx / sqrtX * pc[_Z])
	};

	Real _ybound[2] = {
		pc[_Y] + abs(ty / sqrtY * pc[_Z]),
		pc[_Y] - abs(ty / sqrtY * pc[_Z])
	};

	Real Xbound[2] = {
		floor((_xbound[_X] - x) / pp[_X]) + 1,
		floor((_xbound[_Y] - x) / pp[_X]) + 1
	};

	Real Ybound[2] = {
		pn[_Y] - floor((_ybound[_Y] - y) / pp[_Y]),
		pn[_Y] - floor((_ybound[_X] - y) / pp[_Y])
	};

	if (Xbound[_X] > pn[_X])	Xbound[_X] = pn[_X];
	if (Xbound[_Y] < 0)		Xbound[_Y] = 0;
	if (Ybound[_X] > pn[_Y]) Ybound[_X] = pn[_Y];
	if (Ybound[_Y] < 0)		Ybound[_Y] = 0;


	for (int yytr = Ybound[_Y]; yytr < Ybound[_X]; ++yytr)
	{
		int offset = yytr * pn[_X];
		Real yyy = y + ((pn[_Y] - yytr) * pp[_Y]);

		Real range_x[2] = {
				pc[_X] + abs(tx / sqrtX * sqrt((yyy - pc[_Y]) * (yyy - pc[_Y]) + zz)),
				pc[_X] - abs(tx / sqrtX * sqrt((yyy - pc[_Y]) * (yyy - pc[_Y]) + zz))
		};

		for (int xxtr = Xbound[_Y]; xxtr < Xbound[_X]; ++xxtr)
		{
			Real xxx = x + ((xxtr - 1) * pp[_X]);
			Real r = sqrt((xxx - pc[_X]) * (xxx - pc[_X]) + (yyy - pc[_Y]) * (yyy - pc[_Y]) + zz);
			Real range_y[2] = {
				pc[_Y] + abs(ty / sqrtY * sqrt((xxx - pc[_X]) * (xxx - pc[_X]) + zz)),
				pc[_Y] - abs(ty / sqrtY * sqrt((xxx - pc[_X]) * (xxx - pc[_X]) + zz))
			};

			if (((xxx < range_x[_X]) && (xxx > range_x[_Y])) && ((yyy < range_y[_X]) && (yyy > range_y[_Y]))) {
				//	int idx = (flag) ? pn[_X] * i + j : pn[_Y] * j + i;
				Real kr = k * r;
				Real operand = lambda * r * r;
				Real res_real = (ampZ * sin(kr)) / operand;
				Real res_imag = (-ampZ * cos(kr)) / operand;
#ifdef _OPENMP 
#pragma omp atomic
				complex_H[channel][offset + xxtr][_RE] += res_real;
#pragma omp atomic
				complex_H[channel][offset + xxtr][_IM] += res_imag;
#else

				complex_H[channel][offset + xxtr][_RE] += res_real;
				complex_H[channel][offset + xxtr][_IM] += res_imag;
#endif
			}
		}
	}
}

void ophPointCloud::diffractEncodedFrsn(void)
{
}

void ophPointCloud::diffractNotEncodedFrsn(uint channel, ivec2 pn, vec2 pp, vec2 ss, vec3 pc, Real k, Real amplitude, Real lambda)
{
	// for performance
	Real x = -ss[_X] / 2;
	Real y = -ss[_Y] / 2;
	Real operand = lambda * pc[_Z];

	Real _xbound[2] = {
		pc[_X] + abs(operand / (2 * pp[_X])),
		pc[_X] - abs(operand / (2 * pp[_X]))
	};

	Real _ybound[2] = {
		pc[_Y] + abs(operand / (2 * pp[_Y])),
		pc[_Y] - abs(operand / (2 * pp[_Y]))
	};

	Real Xbound[2] = {
		floor((_xbound[_X] - x) / pp[_X]) + 1,
		floor((_xbound[_Y] - x) / pp[_X]) + 1
	};

	Real Ybound[2] = {
		pn[_Y] - floor((_ybound[_Y] - y) / pp[_Y]),
		pn[_Y] - floor((_ybound[_X] - y) / pp[_Y])
	};

	if (Xbound[_X] > pn[_X])	Xbound[_X] = pn[_X];
	if (Xbound[_Y] < 0)		Xbound[_Y] = 0;
	if (Ybound[_X] > pn[_Y]) Ybound[_X] = pn[_Y];
	if (Ybound[_Y] < 0)		Ybound[_Y] = 0;

	for (int yytr = Ybound[_Y]; yytr < Ybound[_X]; ++yytr)
	{
		Real yyy = (y + (pn[_Y] - yytr) * pp[_Y]) - pc[_Y];
		int offset = yytr * pn[_X];
		for (int xxtr = Xbound[_Y]; xxtr < Xbound[_X]; ++xxtr)
		{
			Real xxx = (x + (xxtr - 1) * pp[_X]) - pc[_X];
			Real p = k * (xxx * xxx + yyy * yyy + 2 * pc[_Z] * pc[_Z]) / (2 * pc[_Z]);

			Real res_real = amplitude * sin(p) / operand;
			Real res_imag = amplitude * (-cos(p)) / operand;

#ifdef _OPENMP
#pragma omp atomic
			complex_H[channel][offset + xxtr][_RE] += res_real;
#pragma omp atomic
			complex_H[channel][offset + xxtr][_IM] += res_imag;
#else
			complex_H[channel][offset + xxtr][_RE] += res_real;
			complex_H[channel][offset + xxtr][_IM] += res_imag;
#endif
		}
	}
}

void ophPointCloud::ophFree(void)
{
	if (pc_data_.vertex) {
		delete[] pc_data_.vertex;
		pc_data_.vertex = nullptr;
	}
	if (pc_data_.color) {
		delete[] pc_data_.color;
		pc_data_.color = nullptr;
	}
	if (pc_data_.phase) {
		delete[] pc_data_.phase;
		pc_data_.phase = nullptr;
	}
}