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

#include "ophwrp.h"
#include "sys.h"
#include "tinyxml2.h"

ophWRP::ophWRP(void)
	: ophGen()
	, scaledVertex(nullptr)
	, bSinglePrecision(false)
{
	n_points = -1;
	p_wrp_ = nullptr;
	is_CPU = true;
	is_ViewingWindow = false;
	LOG("*** WRP : BUILD DATE: %s %s ***\n\n", __DATE__, __TIME__);
}

ophWRP::~ophWRP(void)
{
}

void ophWRP::setViewingWindow(bool is_ViewingWindow)
{
	this->is_ViewingWindow = is_ViewingWindow;
}

void ophWRP::autoScaling()
{
	LOG("%s : ", __FUNCTION__);
	auto begin = CUR_TIME;

	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];

	Real size = pnY * ppY * 0.8 / 2.0;

	OphPointCloudData pc = obj_;
	if (scaledVertex) {
		delete[] scaledVertex;
		scaledVertex = nullptr;
	}
	scaledVertex = new Real[n_points * 3];
	memcpy(scaledVertex, pc.vertex, sizeof(Real) * n_points * 3);


	Real* x = new Real[n_points];
	Real* y = new Real[n_points];
	Real* z = new Real[n_points];

	int i;
#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
	for (i = 0; i < n_points; i++) {
		uint idx = 3 * i;
		x[i] = scaledVertex[idx + _X];
		y[i] = scaledVertex[idx + _Y];
		z[i] = scaledVertex[idx + _Z];
	}

	Real xmax = maxOfArr(x, n_points);
	Real ymax = maxOfArr(y, n_points);
	Real zmax = maxOfArr(z, n_points);

	Real maxXY = (xmax > ymax) ? xmax : ymax;

	int j;
#ifdef _OPENMP
#pragma omp parallel for private(j) firstprivate(maxXY, zmax, size)
#endif
	for (j = 0; j < n_points; ++j) { //Create Fringe Pattern
		uint idx = 3 * j;
		scaledVertex[idx + _X] = scaledVertex[idx + _X] / maxXY * size;
		scaledVertex[idx + _Y] = scaledVertex[idx + _Y] / maxXY * size;
		scaledVertex[idx + _Z] = scaledVertex[idx + _Z] / zmax * size;
		z[j] = scaledVertex[idx + _Z];
	}

	zmax_ = maxOfArr(z, n_points);

	delete[] x;
	delete[] y;
	delete[] z;

	LOG("%lf (s)\n", ELAPSED_TIME(begin, CUR_TIME));
}

int ophWRP::loadPointCloud(const char* pc_file)
{
	n_points = ophGen::loadPointCloud(pc_file, &obj_);

	return n_points;

}

bool ophWRP::readConfig(const char* fname)
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
	auto ret = xml_doc.LoadFile(fname);
	if (ret != XML_SUCCESS)
	{
		LOG("Failed to load file \"%s\"\n", fname);
		return false;
	}

	xml_node = xml_doc.FirstChild();

	// about viewing window
	auto next = xml_node->FirstChildElement("FieldLength");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&wrp_config_.fieldLength))
		return false;

	// about point
	next = xml_node->FirstChildElement("ScaleX");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&wrp_config_.scale[_X]))
		return false;
	next = xml_node->FirstChildElement("ScaleY");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&wrp_config_.scale[_Y]))
		return false;
	next = xml_node->FirstChildElement("ScaleZ");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&wrp_config_.scale[_Z]))
		return false;
	next = xml_node->FirstChildElement("Distance");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&wrp_config_.propagation_distance))
		return false;
	next = xml_node->FirstChildElement("LocationOfWRP");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&wrp_config_.wrp_location))
		return false;
	next = xml_node->FirstChildElement("NumOfWRP");
	if (!next || XML_SUCCESS != next->QueryIntText(&wrp_config_.num_wrp))
		return false;

	auto end = CUR_TIME;
	auto during = ((chrono::duration<Real>)(end - start)).count();
	LOG("%lf (s)..done\n", during);

	initialize();
	return true;
}

void ophWRP::addPixel2WRP(int x, int y, Complex<Real> temp)
{
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];

	if (x >= 0 && x < pnX && y >= 0 && y < pnY) {
		uint adr = x + y * pnX;
		if (adr == 0) 
			std::cout << ".0";
		p_wrp_[adr] = p_wrp_[adr] + temp;
	}

}

void ophWRP::addPixel2WRP(int x, int y, oph::Complex<Real> temp, oph::Complex<Real>* wrp)
{
	long long int Nx = context_.pixel_number.v[0];
	long long int Ny = context_.pixel_number.v[1];

	if (x >= 0 && x<Nx && y >= 0 && y< Ny) {
		long long int adr = x + y*Nx;
		wrp[adr] += temp[adr];
	}
}

oph::Complex<Real>* ophWRP::calSubWRP(double wrp_d, Complex<Real>* wrp, OphPointCloudData* pc)
{

	Real wave_num = context_.k;   // wave_number
	Real wave_len = context_.wave_length[0];  //wave_length

	int Nx = context_.pixel_number.v[0]; //slm_pixelNumberX
	int Ny = context_.pixel_number.v[1]; //slm_pixelNumberY

	Real wpx = context_.pixel_pitch.v[0];//wrp pitch
	Real wpy = context_.pixel_pitch.v[1];


	int Nx_h = Nx >> 1;
	int Ny_h = Ny >> 1;

	int num = n_points;


#ifdef _OPENMP
	omp_set_num_threads(omp_get_num_threads());
#pragma omp parallel for
#endif

	for (int k = 0; k < num; k++) {

		uint idx = 3 * k;
		uint color_idx = pc->n_colors * k;

		Real x = pc->vertex[idx + _X];
		Real y = pc->vertex[idx + _Y];
		Real z = pc->vertex[idx + _Z];
		Real amplitude = pc->color[color_idx];

		float dz = wrp_d - z;
		//	float tw = (int)fabs(wave_len*dz / wpx / wpx / 2 + 0.5) * 2 - 1;
		float tw = fabs(dz)*wave_len / wpx / wpx / 2;

		int w = (int)tw;

		int tx = (int)(x / wpx) + Nx_h;
		int ty = (int)(y / wpy) + Ny_h;

		printf("num=%d, tx=%d, ty=%d, w=%d\n", k, tx, ty, w);

		for (int wy = -w; wy < w; wy++) {
			for (int wx = -w; wx<w; wx++) {//WRP coordinate

				double dx = wx*wpx;
				double dy = wy*wpy;
				double dz = wrp_d - z;

				double sign = (dz>0.0) ? (1.0) : (-1.0);
				double r = sign*sqrt(dx*dx + dy*dy + dz*dz);

				//double tmp_re,tmp_im;
				Complex<Real> tmp;

				tmp._Val[_RE] = (amplitude*cosf(wave_num*r)*cosf(wave_num*wave_len*rand(0, 1))) / (r + 0.05);
				tmp._Val[_IM] = (-amplitude*sinf(wave_num*r)*sinf(wave_num*wave_len*rand(0, 1))) / (r + 0.05);

				if (tx + wx >= 0 && tx + wx < Nx && ty + wy >= 0 && ty + wy < Ny)
				{
					addPixel2WRP(wx + tx, wy + ty, tmp, wrp);
				}
			}
		}
	}

	return wrp;
}

void ophWRP::setMode(bool isCPU)
{
	is_CPU = isCPU;
}

double ophWRP::calculateWRPCPU(void)
{
	LOG("%s\n", __FUNCTION__);
	auto begin = CUR_TIME;

	const uint pnX = context_.pixel_number[_X]; //slm_pixelNumberX
	const uint pnY = context_.pixel_number[_Y]; //slm_pixelNumberY
	const Real ppX = context_.pixel_pitch[_X]; //wrp pitch
	const Real ppY = context_.pixel_pitch[_Y];
	const uint N = pnX * pnY;
	const uint nChannel = context_.waveNum;
	const Real distance = wrp_config_.propagation_distance;
	int hpnX = pnX >> 1;
	int hpnY = pnY >> 1;

	OphPointCloudData pc = obj_;
	Real wrp_d = wrp_config_.wrp_location;

	// Memory Location for Result Image

	if (p_wrp_) {
		delete[] p_wrp_;
		p_wrp_ = nullptr;
	}
	p_wrp_ = new Complex<Real>[N];
	memset(p_wrp_, 0.0, sizeof(Complex<Real>) * N);
	
	int sum = 0;
	m_nProgress = 0;
	Real ppXX = ppX * ppX * 2;
	Real pi2 = 2 * M_PI;
	Real dz = wrp_d - zmax_;
	Real dzz = dz * dz;

	bool bRandomPhase = GetRandomPhase();

	for (uint ch = 0; ch < nChannel; ch++)
	{
		Real lambda = context_.wave_length[ch];  //wave_length
		Real k = context_.k = pi2 / lambda;
		int iColor = ch;
		int i;
		int sum = 0;
#ifdef _OPENMP
#pragma omp parallel for private(i) firstprivate(iColor, ppXX, pnX, pnY, ppX, ppY, hpnX, hpnY, wrp_d, k, pi2)
#endif
		for (i = 0; i < n_points; ++i)
		{
			uint idx = 3 * i;
			uint color_idx = pc.n_colors * i;

			Real x = scaledVertex[idx + _X];
			Real y = scaledVertex[idx + _Y];
			Real z = scaledVertex[idx + _Z];
			Real amplitude = pc.color[color_idx + iColor];

			//Real dz = wrp_d - z;
			//Real dzz = dz * dz;
			bool sign = (dz > 0.0) ? true : false;
			//Real tw = fabs(lambda * dz / ppXX + 0.5) * 2 - 1;
			Real tw = fabs(lambda * dz / ppXX) * 2;

			int w = (int)tw;

			int tx = (int)(x / ppX) + hpnX;
			int ty = (int)(y / ppY) + hpnY;

			for (int wy = -w; wy < w; wy++)
			{
				Real dy = wy * ppY;
				Real dyy = dy * dy;
				int tmpY = wy + ty;
				int baseY = tmpY * pnX;

				for (int wx = -w; wx < w; wx++) //WRP coordinate
				{
					int tmpX = wx + tx;
					if (tmpX >= 0 && tmpX < pnX && tmpY >= 0 && tmpY < pnY) {
						uint adr = tmpX + baseY;

						Real dx = wx * ppX;
						Real r = sign ? sqrt(dx * dx + dyy + dzz) : -sqrt(dx * dx + dyy + dzz);
						Real kr = k * r;
						Real randVal = bRandomPhase ? rand(0.0, 1.0) : 1.0;
						Real randpi2 = pi2 * randVal;
						Complex<Real> tmp;
						tmp[_RE] = (amplitude * cos(kr) * cos(randpi2)) / r;
						tmp[_IM] = (-amplitude * sin(kr) * sin(randpi2)) / r;

						if (adr == 0)
							std::cout << ".0";
#ifdef _OPENMP
#pragma omp atomic
						p_wrp_[adr][_RE] += tmp[_RE];
#pragma omp atomic
						p_wrp_[adr][_IM] += tmp[_IM];
#else
						p_wrp_[adr] += tmp;
#endif						
					}
				}
			}
		}
		LOG("\t");

		fresnelPropagation(p_wrp_, complex_H[ch], distance, ch);
		memset(p_wrp_, 0.0, sizeof(Complex<Real>) * N);
	}
	delete[] p_wrp_;
	delete[] scaledVertex;
	p_wrp_ = nullptr;
	scaledVertex = nullptr;

	LOG("Total : %lf (s)\n", ELAPSED_TIME(begin, CUR_TIME));

	return 0.;
}

void ophWRP::generateHologram(void)
{
	resetBuffer();

	auto begin = CUR_TIME;
	LOG("1) Algorithm Method : WRP\n");
	LOG("2) Generate Hologram with %s\n", m_mode & MODE_GPU ?
		"GPU" :
#ifdef _OPENMP
		"Multi Core CPU"
#else
		"Single Core CPU"
#endif
	);
	LOG("3) Random Phase Use : %s\n", GetRandomPhase() ? "Y" : "N");
	
	autoScaling();
	m_mode & MODE_GPU ? calculateWRPGPU() : calculateWRPCPU();

	fftFree();
	LOG("Total Elapsed Time: %lf (s)\n", ELAPSED_TIME(begin, CUR_TIME));
}

Complex<Real>** ophWRP::calculateMWRP(void)
{
	int wrp_num = wrp_config_.num_wrp;

	if (wrp_num < 1)
		return nullptr;

	Complex<Real>** wrp_list = nullptr;

	Real k = context_.k;   // wave_number
	Real lambda = context_.wave_length[0];  //wave_length

	const uint pnX = context_.pixel_number[_X]; //slm_pixelNumberX
	const uint pnY = context_.pixel_number[_Y]; //slm_pixelNumberY
	const uint pnXY = pnX * pnY;
	const Real ppX = context_.pixel_pitch[_X];//wrp pitch
	const Real ppY = context_.pixel_pitch[_Y];

	Complex<Real>* wrp = new Complex<Real>[pnXY];

	OphPointCloudData pc = obj_;

	for (int i = 0; i<wrp_num; i++)
	{
//		wrp = calSubWRP(wrp_d, wrp, &pc);
		wrp_list[i] = wrp;
	}

	return wrp_list;
}

void ophWRP::ophFree(void)
{
	if (obj_.vertex) {
		delete[] obj_.vertex;
		obj_.vertex = nullptr;
	}
	if (obj_.color) {
		delete[] obj_.color;
		obj_.color = nullptr;
	}
}

void ophWRP::transVW(Real* dst, Real* src, int size)
{
	Real fieldLens = getFieldLens();
	for (int i = 0; i < size; i++) {
		dst[i] = (-fieldLens * src[i]) / (src[i] - fieldLens);
	}
}