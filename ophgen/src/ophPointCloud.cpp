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

ophPointCloud::ophPointCloud(void)
	: ophGen()
	, is_ViewingWindow(false)
	, m_nProgress(0)
{
	LOG("*** POINT CLOUD : BUILD DATE: %s %s ***\n\n", __DATE__, __TIME__);
}

ophPointCloud::ophPointCloud(const char* pc_file, const char* cfg_file)
	: ophGen()
	, is_ViewingWindow(false)
	, m_nProgress(0)
{
	LOG("*** POINT CLOUD : BUILD DATE: %s %s ***\n\n", __DATE__, __TIME__);
	if (loadPointCloud(pc_file) == -1) LOG("<FAILED> Load point cloud data file(\'%s\')", pc_file);
	if (!readConfig(cfg_file)) LOG("<FAILED> Load config specification data file(\'%s\')", cfg_file);
}

ophPointCloud::~ophPointCloud(void)
{
}

int ophPointCloud::loadPointCloud(const char* pc_file)
{
	return ophGen::loadPointCloud(pc_file, &pc_data_);
}

void ophPointCloud::setViewingWindow(bool is_ViewingWindow)
{
	this->is_ViewingWindow = is_ViewingWindow;
}

bool ophPointCloud::readConfig(const char* fname)
{
	if (!ophGen::readConfig(fname))
		return false;

	bool bRet = true;

	using namespace tinyxml2;
	/*XML parsing*/
	tinyxml2::XMLDocument xml_doc;
	XMLNode *xml_node;
	XMLError error;

	if (!checkExtension(fname, ".xml"))
	{
		LOG("<FAILED> Wrong file ext.\n");
		return false;
	}
	if ((error = xml_doc.LoadFile(fname)) != XML_SUCCESS)
	{
		LOG("<FAILED> Loading file (%d)\n", error);
		return false;
	}
	
	xml_node = xml_doc.FirstChild();
	
	char szNodeName[32] = { 0, };
	sprintf(szNodeName, "ScaleX");
	// about point
	auto next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&pc_config_.scale[_X]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "ScaleY");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&pc_config_.scale[_Y]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "ScaleZ");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&pc_config_.scale[_Z]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "Distance");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&pc_config_.distance))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}

	initialize();

	LOG("**************************************************\n");
	LOG("              Read Config (Point Cloud)           \n");
	LOG("1) Focal Length : %.5lf\n", pc_config_.distance);
	LOG("2) Object Scale : %.5lf / %.5lf / %.5lf\n", pc_config_.scale[_X], pc_config_.scale[_Y], pc_config_.scale[_Z]);
	LOG("**************************************************\n");
	
	return bRet;
}

Real ophPointCloud::generateHologram(uint diff_flag)
{
	auto begin = CUR_TIME;
	if (diff_flag != PC_DIFF_RS && diff_flag != PC_DIFF_FRESNEL) {
		LOG("<FAILED> Wrong parameters.");
		return 0.0;
	}

	resetBuffer();
	LOG("**************************************************\n");
	LOG("                Generate Hologram                 \n");
	LOG("1) Algorithm Method : Point Cloud\n");
	LOG("2) Generate Hologram with %s\n", m_mode & MODE_GPU ?
		"GPU" :
#ifdef _OPENMP
		"Multi Core CPU"
#else
		"Single Core CPU"
#endif
	);
	LOG("3) Diffraction Method : %s\n", diff_flag == PC_DIFF_RS ? "R-S" : "Fresnel");
	LOG("4) Number of Point Cloud : %llu\n", pc_data_.n_points);
	LOG("5) Precision Level : %s\n", m_mode & MODE_FLOAT ? "Single" : "Double");
	if(m_mode & MODE_GPU)
		LOG("6) Use FastMath : %s\n", m_mode & MODE_FASTMATH ? "Y" : "N");
	LOG("**************************************************\n");

	// Create CGH Fringe Pattern by 3D Point Cloud
	if (m_mode & MODE_GPU) { //Run GPU
		genCghPointCloudGPU(diff_flag);
	}
	else { //Run CPU
		genCghPointCloudCPU(diff_flag);
	}
	

	Real elapsed_time = ELAPSED_TIME(begin, CUR_TIME);
	LOG("Total Elapsed Time: %.5lf (s)\n", elapsed_time);
	m_nProgress = 0;
	return elapsed_time;
}

void ophPointCloud::encodeHologram(const vec2 band_limit, const vec2 spectrum_shift)
{
	if (complex_H == nullptr) {
		LOG("<FAILED> Not found diffracted data.");
		return;
	}

	const uint nChannel = context_.waveNum;
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];
	const long long int pnXY = pnX * pnY;

	m_vecEncodeSize = ivec2(pnX, pnY);
	context_.ss[_X] = pnX * ppX;
	context_.ss[_Y] = pnY * ppY;
	vec2 ss = context_.ss;

	Real halfppX = ppX / 2;
	Real halfssX = ss[_X] / 2;

	Complex<Real>* tmp = new Complex<Real>[pnXY];

	for (uint ch = 0; ch < nChannel; ch++) {

		fft2(ivec2(pnX, pnY), complex_H[ch], OPH_FORWARD);
		fft2(complex_H[ch], tmp, pnX, pnY, OPH_FORWARD);
		fft2(ivec2(pnX, pnY), tmp, OPH_BACKWARD);
		fft2(tmp, tmp, pnX, pnY, OPH_BACKWARD);

		for (int i = 0; i < pnXY; i++)
		{
			Complex<Real> shift_phase(1.0, 0.0);

			int w = i % pnX;
			int h = i / pnX;
			Real y = (ss[_Y] - ppY) - (ppY * h);
			Real Y = (M_PI * y * spectrum_shift[_Y]) / ppY;
			Real x = -halfssX + ppX * w + halfppX;
			Real X = (M_PI * x * spectrum_shift[_X]) / ppX;

			shift_phase[_RE] = shift_phase[_RE] * (cos(X) * cos(Y) - sin(X) * sin(Y));
			m_lpEncoded[ch][i] = (tmp[i] * shift_phase).real();
		}
	}
	delete[] tmp;
}

void ophPointCloud::encoding(unsigned int ENCODE_FLAG)
{
	ophGen::encoding(ENCODE_FLAG);
}

void ophPointCloud::encoding(unsigned int ENCODE_FLAG, unsigned int SSB_PASSBAND)
{
	if (ENCODE_FLAG == ENCODE_SSB) encodeHologram();
	else if (ENCODE_FLAG == ENCODE_OFFSSB) ophGen::encoding(ENCODE_FLAG, SSB_PASSBAND);
}

void ophPointCloud::genCghPointCloudCPU(uint diff_flag)
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

	int sum = 0;
	m_nProgress = 0;
	int n_points = pc_data_.n_points;

	Vertex *pVertex = nullptr;
	if (is_ViewingWindow) {
		pVertex = new Vertex[pc_data_.n_points];
		std::memcpy(pVertex, pc_data_.vertices, sizeof(Vertex) * pc_data_.n_points);
		transVW(pc_data_.n_points, pVertex, pVertex);
	}
	else {
		pVertex = pc_data_.vertices;
	}
	
	for (uint ch = 0; ch < nChannel; ++ch) {
		Real lambda = context_.wave_length[ch];
		Real k = context_.k = (2 * M_PI / lambda);

#ifdef _OPENMP
#pragma omp parallel for firstprivate(lambda)
#endif
		for (int i = 0; i < n_points; ++i) { //Create Fringe Pattern

			Point pc = pVertex[i].point;
			Real amplitude = pVertex[i].color.color[ch];
			
			pc.pos[_X] *= pc_config_.scale[_X];
			pc.pos[_Y] *= pc_config_.scale[_Y];
			pc.pos[_Z] *= pc_config_.scale[_Z];

			
			switch (diff_flag)
			{
			case PC_DIFF_RS:
				RS_Diffraction(pc, complex_H[ch], lambda, pc_config_.distance, amplitude);
				break;
			case PC_DIFF_FRESNEL:
				Fresnel_Diffraction(pc, complex_H[ch], lambda, pc_config_.distance, amplitude);
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
	LOG("%s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
}

void ophPointCloud::ophFree(void)
{
	ophGen::ophFree();
	if (pc_data_.vertices) {
		delete[] pc_data_.vertices;
		pc_data_.vertices = nullptr;
	}
}