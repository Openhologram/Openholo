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

#include "ophGen.h"
#include "sys.h"
#include "function.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <omp.h>
#include "tinyxml2.h"
#include "PLYparser.h"

extern "C"
{
	/**
	* @brief Convert data from the spatial domain to the frequency domain using 2D FFT on GPU.
	* @details call CUDA Kernel - fftShift and CUFFT Library.
	* @param stream : CUDA Stream
	* @param nx : the number of column of the input data
	* @param ny : the number of row of the input data
	* @param in_field : input complex data variable
	* @param output_field : output complex data variable
	* @param direction : If direction == -1, forward FFT, if type == 1, inverse FFT.
	* @param bNomarlized : If bNomarlized == true, normalize the result after FFT.
	* @see propagation_AngularSpectrum_GPU, encoding_GPU
	*/
	void cudaFFT(CUstream_st* stream, int nx, int ny, cufftDoubleComplex* in_filed, cufftDoubleComplex* output_field, int direction, bool bNormailized = false);

	/**
	* @brief Crop input data according to x, y coordinates on GPU.
	* @details call CUDA Kernel - cropFringe.
	* @param stream : CUDA Stream
	* @param nx : the number of column of the input data
	* @param ny : the number of row of the input data
	* @param in_field : input complex data variable
	* @param output_field : output complex data variable
	* @param cropx1 : the start x-coordinate to crop.
	* @param cropx2 : the end x-coordinate to crop.
	* @param cropy1 : the start y-coordinate to crop.
	* @param cropy2 : the end y-coordinate to crop.
	* @see encoding_GPU
	*/
	void cudaCropFringe(CUstream_st* stream, int nx, int ny, cufftDoubleComplex* in_field, cufftDoubleComplex* out_field, int cropx1, int cropx2, int cropy1, int cropy2);

	/**
	* @brief Encode the CGH according to a signal location parameter on the GPU.
	* @details The variable, ((Real*)p_hologram) has the final result.
	* @param stream : CUDA Stream
	* @param pnx : the number of column of the input data
	* @param pny : the number of row of the input data
	* @param in_field : input data
	* @param out_field : output data
	* @param sig_locationx : signal location of x-axis, left or right half
	* @param sig_locationy : signal location of y-axis, upper or lower half
	* @param ssx : pnx * ppx
	* @param ssy : pny * ppy
	* @param ppx : pixel pitch of x-axis
	* @param ppy : pixel pitch of y-axis
	* @param PI : Pi
	* @see encoding_GPU
	*/
	void cudaGetFringe(CUstream_st* stream, int pnx, int pny, cufftDoubleComplex* in_field, cufftDoubleComplex* out_field, int sig_locationx, int sig_locationy,
		Real ssx, Real ssy, Real ppx, Real ppy, Real PI);
}

ophGen::ophGen(void)
	: Openholo()
	, m_lpEncoded(nullptr)
	, m_lpNormalized(nullptr)
	, m_nOldChannel(0)
	, m_elapsedTime(0.0)
	, m_dFieldLength(0.0)
	, m_nStream(1)
	, m_mode(0)
	, m_precision(PRECISION::DOUBLE)
{

}

ophGen::~ophGen(void)
{

}

void ophGen::initialize(void)
{
	auto begin = CUR_TIME;
	// Output Image Size
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;
	const int nChannel = context_.waveNum;

	// Memory Location for Result Image
	if (complex_H != nullptr) {
		for (uint i = 0; i < m_nOldChannel; i++) {
			if (complex_H[i] != nullptr) {
				delete[] complex_H[i];
				complex_H[i] = nullptr;
			}
		}
		delete[] complex_H;
		complex_H = nullptr;
	}

	complex_H = new Complex<Real>*[nChannel];
	for (uint i = 0; i < nChannel; i++) {
		complex_H[i] = new Complex<Real>[pnXY];
		memset(complex_H[i], 0, sizeof(Complex<Real>) * pnXY);
	}

	if (m_lpEncoded != nullptr) {
		for (uint i = 0; i < m_nOldChannel; i++) {
			if (m_lpEncoded[i] != nullptr) {
				delete[] m_lpEncoded[i];
				m_lpEncoded[i] = nullptr;
			}
		}
		delete[] m_lpEncoded;
		m_lpEncoded = nullptr;
	}
	m_lpEncoded = new Real*[nChannel];
	for (uint i = 0; i < nChannel; i++) {
		m_lpEncoded[i] = new Real[pnXY];
		memset(m_lpEncoded[i], 0, sizeof(Real) * pnXY);
	}

	if (m_lpNormalized != nullptr) {
		for (uint i = 0; i < m_nOldChannel; i++) {
			if (m_lpNormalized[i] != nullptr) {
				delete[] m_lpNormalized[i];
				m_lpNormalized[i] = nullptr;
			}
		}
		delete[] m_lpNormalized;
		m_lpNormalized = nullptr;
	}
	m_lpNormalized = new uchar*[nChannel];
	for (uint i = 0; i < nChannel; i++) {
		m_lpNormalized[i] = new uchar[pnXY];
		memset(m_lpNormalized[i], 0, sizeof(uchar) * pnXY);
	}

	m_nOldChannel = nChannel;
	m_vecEncodeSize[_X] = pnX;
	m_vecEncodeSize[_Y] = pnY;
	LOG("%s => %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
}

int ophGen::loadPointCloud(const char* pc_file, OphPointCloudData *pc_data_)
{
	int n_points = 0;
	auto begin = CUR_TIME;

	PLYparser plyIO;
	if (!plyIO.loadPLY(pc_file, pc_data_->n_points, pc_data_->n_colors, &pc_data_->vertex, &pc_data_->color, &pc_data_->phase, pc_data_->isPhaseParse))
		n_points = -1;
	else
		n_points = pc_data_->n_points;
	LOG("%s => %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
	return n_points;
}

bool ophGen::readConfig(const char* fname)
{
	bool bRet = true;
	using namespace tinyxml2;

	auto begin = CUR_TIME;
	tinyxml2::XMLDocument xml_doc;
	XMLNode *xml_node = nullptr;

	if (!checkExtension(fname, ".xml"))
	{
		LOG("<FAILED> Wrong file ext.\n");
		return false;
	}
	auto ret = xml_doc.LoadFile(fname);
	if (ret != XML_SUCCESS)
	{
		LOG("<FAILED> Loading file.\n");
		return false;
	}
	xml_node = xml_doc.FirstChild();

	int nWave = 1;
	char szNodeName[32] = { 0, };

	wsprintfA(szNodeName, "SLM_WaveNum");
	auto next = xml_node->FirstChildElement(szNodeName); // OffsetInDepth
	if (!next || XML_SUCCESS != next->QueryIntText(&nWave))
	{
		LOG("<FAILED> Not found node : \'%s\' (Integer) \n", szNodeName);
		bRet = false;
	}
	context_.waveNum = nWave;
	if (context_.wave_length) delete[] context_.wave_length;
	context_.wave_length = new Real[nWave];

	for (int i = 1; i <= nWave; i++) {
		wsprintfA(szNodeName, "SLM_WaveLength_%d", i);
		next = xml_node->FirstChildElement(szNodeName);
		if (!next || XML_SUCCESS != next->QueryDoubleText(&context_.wave_length[i - 1]))
		{
			LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
			bRet = false;
		}
	}

	wsprintfA(szNodeName, "SLM_PixelNumX");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryIntText(&context_.pixel_number[_X]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Integer) \n", szNodeName);
		bRet = false;
	}

	wsprintfA(szNodeName, "SLM_PixelNumY");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryIntText(&context_.pixel_number[_Y]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Integer) \n", szNodeName);
		bRet = false;
	}

	wsprintfA(szNodeName, "SLM_PixelPitchX");
	next = xml_node->FirstChildElement("SLM_PixelPitchX");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&context_.pixel_pitch[_X]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}

	wsprintfA(szNodeName, "SLM_PixelPitchY");
	next = xml_node->FirstChildElement("SLM_PixelPitchY");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&context_.pixel_pitch[_Y]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}

	// option
	next = xml_node->FirstChildElement("IMG_Rotation");
	if (!next || XML_SUCCESS != next->QueryBoolText(&imgCfg.rotate))
		imgCfg.rotate = false;
	next = xml_node->FirstChildElement("IMG_Merge");
	if (!next || XML_SUCCESS != next->QueryBoolText(&imgCfg.merge))
		imgCfg.merge = false;
	next = xml_node->FirstChildElement("IMG_Flip");
	if (!next || XML_SUCCESS != next->QueryIntText(&imgCfg.flip))
		imgCfg.flip = 0;
	next = xml_node->FirstChildElement("DoublePrecision");
	if (!next || XML_SUCCESS != next->QueryBoolText(&context_.bUseDP))
		context_.bUseDP = true;
	next = xml_node->FirstChildElement("ShiftX");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&context_.shift[_X]))
		context_.shift[_X] = 0.0;
	next = xml_node->FirstChildElement("ShiftY");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&context_.shift[_Y]))
		context_.shift[_Y] = 0.0;
	next = xml_node->FirstChildElement("ShiftZ");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&context_.shift[_Z]))
		context_.shift[_Z] = 0.0;
	next = xml_node->FirstChildElement("FieldLength");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&m_dFieldLength))
		m_dFieldLength = 0.0;
	next = xml_node->FirstChildElement("RandomPhase");
	if (!next || XML_SUCCESS != next->QueryBoolText(&m_bRandomPhase))
		m_bRandomPhase = true;
	next = xml_node->FirstChildElement("NumOfStream");
	if (!next || XML_SUCCESS != next->QueryIntText(&m_nStream))
		m_nStream = 1;

	context_.ss[_X] = context_.pixel_number[_X] * context_.pixel_pitch[_X];
	context_.ss[_Y] = context_.pixel_number[_Y] * context_.pixel_pitch[_Y];

	Openholo::setPixelNumberOHC(context_.pixel_number);
	Openholo::setPixelPitchOHC(context_.pixel_pitch);

	OHC_encoder->clearWavelength();
	for (int i = 0; i < nWave; i++)
		Openholo::setWavelengthOHC(context_.wave_length[i], LenUnit::m);

	LOG("%s => %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));

	return bRet;
}

void ophGen::RS_Diffraction(vec3 src, Complex<Real> *dst, Real lambda, Real distance, Real amplitude)
{
	OphConfig *pConfig = &context_;
	const int pnX = pConfig->pixel_number[_X];
	const int pnY = pConfig->pixel_number[_Y];
	const int pnXY = pnX * pnY;
	const Real ppX = pConfig->pixel_pitch[_X];
	const Real ppY = pConfig->pixel_pitch[_Y];
	const Real ssX = pConfig->ss[_X] = pnX * ppX;
	const Real ssY = pConfig->ss[_Y] = pnY * ppY;

	const Real tx = lambda / (2 * ppX);
	const Real ty = lambda / (2 * ppY);
	const Real sqrtX = sqrt(1 - (tx * tx));
	const Real sqrtY = sqrt(1 - (ty * ty));
	const Real x = -ssX / 2;
	const Real y = -ssY / 2;
	const Real k = (2 * M_PI) / lambda;
	Real z = src[_Z] + distance;
	Real zz = z * z;
	Real ampZ = amplitude * z;

	Real _xbound[2] = {
		src[_X] + abs(tx / sqrtX * z),
		src[_X] - abs(tx / sqrtX * z)
	};

	Real _ybound[2] = {
		src[_Y] + abs(ty / sqrtY * z),
		src[_Y] - abs(ty / sqrtY * z)
	};

	Real Xbound[2] = {
		floor((_xbound[_X] - x) / ppX) + 1,
		floor((_xbound[_Y] - x) / ppX) + 1
	};

	Real Ybound[2] = {
		pnY - floor((_ybound[_Y] - y) / ppY),
		pnY - floor((_ybound[_X] - y) / ppY)
	};

	if (Xbound[_X] > pnX)	Xbound[_X] = pnX;
	if (Xbound[_Y] < 0)		Xbound[_Y] = 0;
	if (Ybound[_X] > pnY) Ybound[_X] = pnY;
	if (Ybound[_Y] < 0)		Ybound[_Y] = 0;


	for (int yytr = Ybound[_Y]; yytr < Ybound[_X]; ++yytr)
	{
		int offset = yytr * pnX;
		Real yyy = y + ((pnY - yytr) * ppY);

		Real range_x[2] = {
			src[_X] + abs(tx / sqrtX * sqrt((yyy - src[_Y]) * (yyy - src[_Y]) + zz)),
			src[_X] - abs(tx / sqrtX * sqrt((yyy - src[_Y]) * (yyy - src[_Y]) + zz))
		};

		for (int xxtr = Xbound[_Y]; xxtr < Xbound[_X]; ++xxtr)
		{
			Real xxx = x + ((xxtr - 1) * ppX);
			Real r = sqrt((xxx - src[_X]) * (xxx - src[_X]) + (yyy - src[_Y]) * (yyy - src[_Y]) + zz);
			Real range_y[2] = {
				src[_Y] + abs(ty / sqrtY * sqrt((xxx - src[_X]) * (xxx - src[_X]) + zz)),
				src[_Y] - abs(ty / sqrtY * sqrt((xxx - src[_X]) * (xxx - src[_X]) + zz))
			};

			if (((xxx < range_x[_X]) && (xxx > range_x[_Y])) && ((yyy < range_y[_X]) && (yyy > range_y[_Y]))) {
				Real kr = k * r;
				Real operand = lambda * r * r;
				Real res_real = (ampZ * sin(kr)) / operand;
				Real res_imag = (-ampZ * cos(kr)) / operand;
#ifdef _OPENMP 
#pragma omp atomic
#endif
				dst[offset + xxtr][_RE] += res_real;
#ifdef _OPENMP 
#pragma omp atomic
#endif
				dst[offset + xxtr][_IM] += res_imag;
			}
		}
	}
}

void ophGen::Fresnel_Diffraction(vec3 src, Complex<Real> *dst, Real lambda, Real distance, Real amplitude)
{
	OphConfig *pConfig = &context_;
	const int pnX = pConfig->pixel_number[_X];
	const int pnY = pConfig->pixel_number[_Y];
	const int pnXY = pnX * pnY;
	const Real ppX = pConfig->pixel_pitch[_X];
	const Real ppY = pConfig->pixel_pitch[_Y];
	const Real ssX = pConfig->ss[_X] = pnX * ppX;
	const Real ssY = pConfig->ss[_Y] = pnY * ppY;
	const Real k = (2 * M_PI) / lambda;

	// for performance
	Real x = -ssX / 2;
	Real y = -ssY / 2;
	Real z = src[_Z] + distance;
	Real zz = z * z;
	Real operand = lambda * z;

	Real _xbound[2] = {
		src[_X] + abs(operand / (2 * ppX)),
		src[_X] - abs(operand / (2 * ppX))
	};

	Real _ybound[2] = {
		src[_Y] + abs(operand / (2 * ppY)),
		src[_Y] - abs(operand / (2 * ppY))
	};

	Real Xbound[2] = {
		floor((_xbound[_X] - x) / ppX) + 1,
		floor((_xbound[_Y] - x) / ppX) + 1
	};

	Real Ybound[2] = {
		pnY - floor((_ybound[_Y] - y) / ppY),
		pnY - floor((_ybound[_X] - y) / ppY)
	};

	if (Xbound[_X] > pnX)	Xbound[_X] = pnX;
	if (Xbound[_Y] < 0)		Xbound[_Y] = 0;
	if (Ybound[_X] > pnY) Ybound[_X] = pnY;
	if (Ybound[_Y] < 0)		Ybound[_Y] = 0;

	for (int yytr = Ybound[_Y]; yytr < Ybound[_X]; ++yytr)
	{
		Real yyy = (y + (pnY - yytr) * ppY) - src[_Y];
		int offset = yytr * pnX;
		for (int xxtr = Xbound[_Y]; xxtr < Xbound[_X]; ++xxtr)
		{
			Real xxx = (x + (xxtr - 1) * ppX) - src[_X];
			Real p = k * (xxx * xxx + yyy * yyy + 2 * zz) / (2 * z);

			Real res_real = amplitude * sin(p) / operand;
			Real res_imag = amplitude * (-cos(p)) / operand;

#ifdef _OPENMP 
#pragma omp atomic
#endif
			dst[offset + xxtr][_RE] += res_real;
#ifdef _OPENMP 
#pragma omp atomic
#endif
			dst[offset + xxtr][_IM] += res_imag;
		}
	}
}

void ophGen::Fresnel_FFT(Complex<Real> *src, Complex<Real> *dst, Real lambda, Real waveRatio, Real distance)
{
	OphConfig *pConfig = &context_;
	const int pnX = pConfig->pixel_number[_X];
	const int pnY = pConfig->pixel_number[_Y];
	const int pnXY = pnX * pnY;
	const Real ppX = pConfig->pixel_pitch[_X];
	const Real ppY = pConfig->pixel_pitch[_Y];
	const Real ssX = pConfig->ss[_X] = pnX * ppX;
	const Real ssY = pConfig->ss[_Y] = pnY * ppY;
	const Real ssX2 = ssX * 2;
	const Real ssY2 = ssY * 2;
	const Real k = (2 * M_PI) / lambda;

	int newSize = pnXY * 4;// *waveRatio;
	Complex<Real>* in2x = new Complex<Real>[newSize];
	memset(in2x, 0, sizeof(Complex<Real>) * newSize);

	uint idxIn = 0;
	int half_pnX = pnX >> 1;
	int half_pnY = pnY >> 1;
	int pnX2 = pnX * 2;
	int pnY2 = pnY * 2;

	for (int i = 0; i < pnY; i++) {
		for (int j = 0; j < pnX; j++) {
			in2x[(i + half_pnY) * pnX2 + (j + half_pnX)] = src[idxIn++];
		}
	}

	Complex<Real>* temp1 = new Complex<Real>[newSize];

	fft2({ pnX2, pnY2 }, in2x, OPH_FORWARD, OPH_ESTIMATE); // fft spatial domain 
	fft2(in2x, temp1, pnX2, pnY2, OPH_FORWARD, false); // 

	Real* fx = new Real[newSize];
	Real* fy = new Real[newSize];

	uint i = 0;

	for (int idxFy = -pnY; idxFy < pnY; idxFy++) {
		for (int idxFx = -pnX; idxFx < pnX; idxFx++) {
			fx[i] = idxFx / ssX2;
			fy[i] = idxFy / ssY2;
			i++;
		}
	}

	Complex<Real>* prop = new Complex<Real>[newSize];
	memset(prop, 0, sizeof(Complex<Real>) * newSize);

	Real sqrtPart;

	Complex<Real>* temp2 = new Complex<Real>[newSize];
	Real lambda_square = lambda * lambda;
	Real tmp = 2 * M_PI * distance;

	for (int i = 0; i < newSize; i++) {
		sqrtPart = sqrt(1 / lambda_square - fx[i] * fx[i] - fy[i] * fy[i]);
		prop[i][_IM] = tmp;
		prop[i][_IM] *= sqrtPart;
		temp2[i] = temp1[i] * exp(prop[i]);
	}

	Complex<Real>* temp3 = new Complex<Real>[newSize];
	fft2({ pnX2, pnY2 }, temp2, OPH_BACKWARD, OPH_ESTIMATE);
	fft2(temp2, temp3, pnX2, pnY2, OPH_BACKWARD, false);

	uint idxOut = 0;
	for (int i = 0; i < pnY; i++) {
		for (int j = 0; j < pnX; j++) {
			dst[idxOut++] = temp3[(i + half_pnY) * pnX2 + (j + half_pnX)];
		}
	}

	delete[] in2x;
	delete[] temp1;
	delete[] fx;
	delete[] fy;
	delete[] prop;
	delete[] temp2;
	delete[] temp3;
}

void ophGen::AngularSpectrumMethod(Complex<Real> *src, Complex<Real> *dst, Real lambda, Real distance)
{
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const int N = pnX * pnY;
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];
	const Real ssX = context_.ss[_X] = pnX * ppX;
	const Real ssY = context_.ss[_Y] = pnY * ppY;

	Real dfx = 1 / ssX;
	Real dfy = 1 / ssY;

	Real k = context_.k = (2 * M_PI / lambda);
	Real kk = k * k;
	Real kd = k * distance;
	Real fx = -1 / (ppX * 2);
	Real fy = 1 / (ppY * 2);

	int i;
#ifdef _OPENMP
#pragma omp parallel for private(i) firstprivate(pnX, dfx, dfy, lambda, kd, kk)
#endif
	for (i = 0; i < N; i++)
	{
		Real x = i % pnX;
		Real y = i / pnX;

		Real fxx = fx + dfx * x;
		Real fyy = fy - dfy - dfy * y;

		Real fxxx = lambda * fxx;
		Real fyyy = lambda * fyy;

		Real sval = sqrt(1 - (fxxx * fxxx) - (fyyy * fyyy));
		sval = sval * kd;
		Complex<Real> kernel(0, sval);
		kernel.exp();

		bool prop_mask = ((fxx * fxx + fyy * fyy) < kk) ? true : false;

		Complex<Real> u_frequency;
		if (prop_mask) {
			u_frequency = kernel * src[i];
			dst[i][_RE] += u_frequency[_RE];
			dst[i][_IM] += u_frequency[_IM];
		}
	}
}

void ophGen::conv_fft2(Complex<Real>* src1, Complex<Real>* src2, Complex<Real>* dst, ivec2 size)
{
	src1FT = new Complex<Real>[size[_X] * size[_Y]];
	src2FT = new Complex<Real>[size[_X] * size[_Y]];
	dstFT = new Complex<Real>[size[_X] * size[_Y]];


	//fft2(size, src1, OPH_FORWARD, OPH_ESTIMATE);
	fft2(src1, src1FT, size[_X], size[_Y], OPH_FORWARD, (bool)OPH_ESTIMATE);

	//fft2(size, src2, OPH_FORWARD, OPH_ESTIMATE);
	fft2(src2, src2FT, size[_X], size[_Y], OPH_FORWARD, (bool)OPH_ESTIMATE);


	for (int i = 0; i < size[_X] * size[_Y]; i++)
		dstFT[i] = src1FT[i] * src2FT[i];

	//fft2(size, dstFT, OPH_BACKWARD, OPH_ESTIMATE);
	fft2(dstFT, dst, size[_X], size[_Y], OPH_BACKWARD, (bool)OPH_ESTIMATE);

	//for (int i = 0; i < size[_X] * size[_Y]; i++) {
	//	if (src2[i][_RE]!=0)
	//		cout << i << ": " << src2[i] << endl;
	//}
	//fftFree();
	delete[] src1FT, src2FT, dstFT;
}

void ophGen::normalize(int ch)
{
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	oph::normalize((Real *)m_lpEncoded[ch], m_lpNormalized[ch], pnX, pnY);
}

void ophGen::normalize(void)
{
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const int nWave = context_.waveNum;
	const int N = pnX * pnY;
	
	Real min = MAX_DOUBLE, max = MIN_DOUBLE, gap = 0;
	for (int ch = 0; ch < nWave; ch++)
	{
		Real minTmp = minOfArr(m_lpEncoded[ch], N);
		Real maxTmp = maxOfArr(m_lpEncoded[ch], N);

		if (min > minTmp)
			min = minTmp;
		if (max < maxTmp)
			max = maxTmp;
	}

	gap = max - min;
	int j;
	for (int ch = 0; ch < nWave; ch++)
	{
#ifdef _OPENMP
#pragma omp parallel for private(j) firstprivate(min, gap, pnX)
#endif
		for (j = 0; j < pnY; j++) {
			for (int i = 0; i < pnX; i++) {
				int idx = j * pnX + i;
				m_lpNormalized[ch][idx] = (((m_lpEncoded[ch][idx] - min) / gap) * 255 + 0.5);
			}
		}
	}
}

bool ophGen::save(const char * fname, uint8_t bitsperpixel, uchar* src, uint px, uint py)
{
	if (fname == nullptr) return false;

	uchar* source = src;
	bool bAlloc = false;
	const uint nChannel = context_.waveNum;

	ivec2 p(px, py);
	if (px == 0 && py == 0)
		p = ivec2(context_.pixel_number[_X], context_.pixel_number[_Y]);

	char path[_MAX_PATH] = { 0, };
	char drive[_MAX_DRIVE] = { 0, };
	char dir[_MAX_DIR] = { 0, };
	char file[_MAX_FNAME] = { 0, };
	char ext[_MAX_EXT] = { 0, };
	_splitpath_s(fname, drive, dir, file, ext);

	sprintf_s(path, "%s", fname);

	if (!strlen(ext)) {
		sprintf_s(path, "%s.bmp", path);
		sprintf_s(ext, ".bmp");
	}
	if (!strlen(drive)) { // Relative path to Absolute path
		char curDir[MAX_PATH] = { 0, };
		GetCurrentDirectory(MAX_PATH, curDir);
		sprintf_s(path, "%s\\%s", curDir, fname);
		for (int i = 0; i < strlen(path); i++) {
			char ch = path[i];
			if (ch == '/')
				path[i] = '\\';
		}
	}


	if (src == nullptr) {
		if (nChannel == 1) {
			source = m_lpNormalized[0];
			saveAsImg(path, bitsperpixel, source, p[_X], p[_Y]);
		}
		else if (nChannel == 3) {
			if (imgCfg.merge) {
				uint nSize = (((p[_X] * bitsperpixel / 8) + 3) & ~3) * p[_Y];
				source = new uchar[nSize];
				bAlloc = true;
				for (int i = 0; i < nChannel; i++) {
					mergeColor(i, p[_X], p[_Y], m_lpNormalized[i], source);
				}
				saveAsImg(path, bitsperpixel, source, p[_X], p[_Y]);
				if (bAlloc) delete[] source;
			}
			else {
				for (int i = 0; i < nChannel; i++) {
					sprintf_s(path, "%s%s%s_%d%s", drive, dir, file, i, ext);
					source = m_lpNormalized[i];
					saveAsImg(path, bitsperpixel / nChannel, source, p[_X], p[_Y]);
				}
			}
		}
		else return false;
	}
	else
		saveAsImg(path, bitsperpixel, source, p[_X], p[_Y]);

	return true;
}

bool ophGen::save(const char * fname, uint8_t bitsperpixel, uint px, uint py, uint fnum, uchar* args ...)
{
	std::string file = fname;
	std::string name;
	std::string ext;

	size_t ex = file.rfind(".");
	if (ex == -1) ex = file.length();

	name = file.substr(0, ex);
	ext = file.substr(ex, file.length() - 1);

	va_list ap;
	__crt_va_start(ap, args);

	for (uint i = 0; i < fnum; i++) {
		name.append(std::to_string(i)).append(ext);
		if (i == 0) {
			save(name.c_str(), bitsperpixel, args, px, py);
			continue;
		}
		uchar* data = __crt_va_arg(ap, uchar*);
		save(name.c_str(), bitsperpixel, data, px, py);
	}

	__crt_va_end(ap);

	return true;
}

void* ophGen::load(const char * fname)
{
	if (checkExtension(fname, ".bmp")) {
		return Openholo::loadAsImg(fname);
	}
	else {			// when extension is not .bmp
		return nullptr;
	}

	return nullptr;
}

bool ophGen::loadAsOhc(const char * fname)
{
	if (!Openholo::loadAsOhc(fname)) return false;

	const uint nChannel = context_.waveNum;
	const uint pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];

	m_lpEncoded = new Real*[nChannel];
	m_lpNormalized = new uchar*[nChannel];
	for (uint ch = 0; ch < nChannel; ch++) {
		m_lpEncoded[ch] = new Real[pnXY];
		memset(m_lpEncoded[ch], 0, sizeof(Real) * pnXY);
		m_lpNormalized[ch] = new uchar[pnXY];
		memset(m_lpNormalized[ch], 0, sizeof(uchar) * pnXY);
	}
	return true;
}

void ophGen::resetBuffer()
{
	int N = context_.pixel_number[_X] * context_.pixel_number[_Y];
	int N2 = m_vecEncodeSize[_X] * m_vecEncodeSize[_Y];

	for (int ch = 0; ch < context_.waveNum; ch++) {
		if (complex_H[ch])
			memset(complex_H[ch], 0., sizeof(Complex<Real>) * N);
		if (m_lpEncoded[ch])
			memset(m_lpEncoded[ch], 0., sizeof(Real) * N2);
		if (m_lpNormalized[ch])
			memset(m_lpNormalized[ch], 0, sizeof(uchar) * N2);
	}
}

void ophGen::encoding(unsigned int ENCODE_FLAG)
{
	auto begin = CUR_TIME;

	// func pointer
	void (ophGen::*encodeFunc) (Complex<Real>*, Real*, const int) = nullptr;

	Complex<Real> *holo = nullptr;
	Real *encoded = nullptr;
	const uint pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];
	m_vecEncodeSize = context_.pixel_number;

	switch (ENCODE_FLAG)
	{
	case ENCODE_PHASE: encodeFunc = &ophGen::Phase; LOG("ENCODE_PHASE\n"); break;
	case ENCODE_AMPLITUDE: encodeFunc = &ophGen::Amplitude; LOG("ENCODE_AMPLITUDE\n"); break;
	case ENCODE_REAL: encodeFunc = &ophGen::RealPart; LOG("ENCODE_REAL\n"); break;
	case ENCODE_IMAGINARY: encodeFunc = &ophGen::ImaginaryPart; LOG("ENCODE_IMAGINARY\n"); break;
	case ENCODE_SIMPLENI: encodeFunc = &ophGen::SimpleNI; LOG("ENCODE_SIMPLENI\n"); break;
	case ENCODE_BURCKHARDT: encodeFunc = &ophGen::Burckhardt; LOG("ENCODE_BURCKHARDT\n"); break;
	case ENCODE_TWOPHASE: encodeFunc = &ophGen::TwoPhase; LOG("ENCODE_TWOPHASE\n"); break;
	default: 
		LOG("<FAILED> WRONG PARAMETERS.\n");
		LOG("%s => %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
		return;
	}

	for (int ch = 0; ch < context_.waveNum; ch++) {
		holo = complex_H[ch];
		encoded = m_lpEncoded[ch];
		(this->*encodeFunc)(holo, encoded, m_vecEncodeSize[_X] * m_vecEncodeSize[_Y]);
	}
	LOG("%s => %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
}

//template <typename T>
//void ophGen::encoding(unsigned int ENCODE_FLAG, Complex<T>* holo, T* encoded)
void ophGen::encoding(unsigned int ENCODE_FLAG, Complex<Real>* holo, Real* encoded)
{
	auto begin = CUR_TIME;

	// func pointer
	void (ophGen::*encodeFunc) (Complex<Real>*, Real*, const int) = nullptr;

	const uint pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];
	m_vecEncodeSize = context_.pixel_number;

	switch (ENCODE_FLAG)
	{
	case ENCODE_PHASE: encodeFunc = &ophGen::Phase; LOG("ENCODE_PHASE\n"); break;
	case ENCODE_AMPLITUDE: encodeFunc = &ophGen::Amplitude; LOG("ENCODE_AMPLITUDE\n"); break;
	case ENCODE_REAL: encodeFunc = &ophGen::RealPart; LOG("ENCODE_REAL\n"); break;
	case ENCODE_IMAGINARY: encodeFunc = &ophGen::ImaginaryPart; LOG("ENCODE_IMAGINARY\n"); break;
	case ENCODE_SIMPLENI: encodeFunc = &ophGen::SimpleNI; LOG("ENCODE_SIMPLENI\n"); break;
	case ENCODE_BURCKHARDT: encodeFunc = &ophGen::Burckhardt; LOG("ENCODE_BURCKHARDT\n"); break;
	case ENCODE_TWOPHASE: encodeFunc = &ophGen::TwoPhase; LOG("ENCODE_TWOPHASE\n"); break;
	default: LOG("<FAILED> WRONG PARAMETERS.\n");  return;
	}
	
	if (holo == nullptr) holo = complex_H[0];
	if (encoded == nullptr) encoded = m_lpEncoded[0];

	(this->*encodeFunc)(holo, encoded, m_vecEncodeSize[_X] * m_vecEncodeSize[_Y]);

	LOG("%s => %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
}

void ophGen::encoding(unsigned int ENCODE_FLAG, unsigned int passband, Complex<Real>* holo, Real* encoded)
{
	holo == nullptr ? holo = *complex_H : holo;
	encoded == nullptr ? encoded = *m_lpEncoded : encoded;

	const uint pnX = m_vecEncodeSize[_X] = context_.pixel_number[_X];
	const uint pnY = m_vecEncodeSize[_Y] = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;
	const uint nChannel = context_.waveNum;

	for (uint ch = 0; ch < nChannel; ch++) {
		/*	initialize	*/
		int m_vecEncodeSize = pnXY;
		if (m_lpEncoded[ch] != nullptr) delete[] m_lpEncoded[ch];
		m_lpEncoded[ch] = new Real[m_vecEncodeSize];
		memset(m_lpEncoded[ch], 0, sizeof(Real) * m_vecEncodeSize);

		if (m_lpNormalized[ch] != nullptr) delete[] m_lpNormalized[ch];
		m_lpNormalized[ch] = new uchar[m_vecEncodeSize];
		memset(m_lpNormalized[ch], 0, sizeof(uchar) * m_vecEncodeSize);

		switch (ENCODE_FLAG)
		{
		case ENCODE_SSB:
			LOG("ENCODE_SSB");
			singleSideBand((holo), m_lpEncoded[ch], context_.pixel_number, passband);
			break;
		case ENCODE_OFFSSB:
		{
			LOG("ENCODE_OFFSSB");
			Complex<Real> *tmp = new Complex<Real>[pnXY];
			memcpy(tmp, holo, sizeof(Complex<Real>) * pnXY);
			freqShift(tmp, tmp, context_.pixel_number, 0, 100);
			singleSideBand(tmp, m_lpEncoded[ch], context_.pixel_number, passband);
			delete[] tmp;
			break;
		}
		default:
			LOG("<FAILED> WRONG PARAMETERS.\n");
			return;
		}
	}
}

void ophGen::encoding(unsigned int BIN_ENCODE_FLAG, unsigned int ENCODE_FLAG, Real threshold, Complex<Real>* holo, Real* encoded)
{
	auto begin = CUR_TIME;
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint nChannel = context_.waveNum;
	const uint pnXY = pnX * pnY;

	switch (BIN_ENCODE_FLAG) {
	case ENCODE_SIMPLEBINARY:
		LOG("ENCODE_SIMPLEBINARY\n");
		if (holo == nullptr || encoded == nullptr)
			for (int ch = 0; ch < nChannel; ch++) {
				binarization(complex_H[ch], m_lpEncoded[ch], pnXY, ENCODE_FLAG, threshold);
			}
		else
			binarization(holo, encoded, pnXY, ENCODE_FLAG, threshold);
		break;
	case ENCODE_EDBINARY:
		LOG("ENCODE_EDBINARY\n");
		if (ENCODE_FLAG != ENCODE_REAL) {
			LOG("<FAILED> WRONG PARAMETERS : %d\n", ENCODE_FLAG);
			return;
		}
		if (holo == nullptr || encoded == nullptr)
			for (int ch = 0; ch < nChannel; ch++) {
				binaryErrorDiffusion(complex_H[ch], m_lpEncoded[ch], context_.pixel_number, FLOYD_STEINBERG, threshold);
			}
		else
			binaryErrorDiffusion(holo, encoded, pnXY, FLOYD_STEINBERG, threshold);

	default:
		LOG("<FAILED> WRONG PARAMETERS.\n");
		return;
	}

	LOG("%s => %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
}

void ophGen::encoding()
{
	const uint pnX = m_vecEncodeSize[_X] = context_.pixel_number[_X];
	const uint pnY = m_vecEncodeSize[_Y] = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;
	const uint nChannel = context_.waveNum;

	if (ENCODE_METHOD == ENCODE_BURCKHARDT) m_vecEncodeSize[_X] *= 3;
	else if (ENCODE_METHOD == ENCODE_TWOPHASE)  m_vecEncodeSize[_X] *= 2;

	for (uint ch = 0; ch < nChannel; ch++) {
		/*	initialize	*/
		if (m_lpEncoded[ch] != nullptr) delete[] m_lpEncoded[ch];
		m_lpEncoded[ch] = new Real[m_vecEncodeSize[_X] * m_vecEncodeSize[_Y]];
		memset(m_lpEncoded[ch], 0, sizeof(Real) * m_vecEncodeSize[_X] * m_vecEncodeSize[_Y]);

		if (m_lpNormalized[ch] != nullptr) delete[] m_lpNormalized[ch];
		m_lpNormalized[ch] = new uchar[m_vecEncodeSize[_X] * m_vecEncodeSize[_Y]];
		memset(m_lpNormalized[ch], 0, sizeof(uchar) * m_vecEncodeSize[_X] * m_vecEncodeSize[_Y]);


		switch (ENCODE_METHOD)
		{
		case ENCODE_SIMPLENI:
			LOG("ENCODE_SIMPLENI\n");
			SimpleNI(complex_H[ch], m_lpEncoded[ch], pnXY);
			break;
		case ENCODE_REAL:
			LOG("ENCODE_REAL\n");
			realPart<Real>(complex_H[ch], m_lpEncoded[ch], pnXY);
			break;
		case ENCODE_BURCKHARDT:
			LOG("ENCODE_BURCKHARDT\n");
			Burckhardt(complex_H[ch], m_lpEncoded[ch], pnXY);
			break;
		case ENCODE_TWOPHASE:
			LOG("ENCODE_TWOPHASE\n");
			TwoPhase(complex_H[ch], m_lpEncoded[ch], pnXY);
			break;
		case ENCODE_PHASE:
			LOG("ENCODE_PHASE\n");
			Phase(complex_H[ch], m_lpEncoded[ch], pnXY);
			break;
		case ENCODE_AMPLITUDE:
			LOG("ENCODE_AMPLITUDE\n");
			getAmplitude(complex_H[ch], m_lpEncoded[ch], pnXY);
			break;
		case ENCODE_SSB:
			LOG("ENCODE_SSB\n");
			singleSideBand(complex_H[ch], m_lpEncoded[ch], context_.pixel_number, SSB_PASSBAND);
			break;
		case ENCODE_OFFSSB:
			LOG("ENCODE_OFFSSB\n");
			freqShift(complex_H[ch], complex_H[ch], context_.pixel_number, 0, 100);
			singleSideBand(complex_H[ch], m_lpEncoded[ch], context_.pixel_number, SSB_PASSBAND);
			break;
		default:
			LOG("<FAILED> WRONG PARAMETERS.\n");
			return;
		}
	}
}

void ophGen::singleSideBand(Complex<Real>* src, Real* dst, const ivec2 holosize, int SSB_PASSBAND)
{
	const int nX = holosize[_X];
	const int nY = holosize[_Y];
	const int half_nX = nX >> 1;
	const int half_nY = nY >> 1;

	int N = nX * nY;
	const int half_N = N >> 1;

	Complex<Real>* AS = new Complex<Real>[N];
	//fft2(holosize, holo, OPH_FORWARD, OPH_ESTIMATE);
	fft2(src, AS, nX, nY, OPH_FORWARD, false);
	//fftExecute(temp);


	switch (SSB_PASSBAND)
	{
	case SSB_LEFT:
		for (int i = 0; i < nY; i++)
		{
			int k = i * nX;
			for (int j = half_nX; j < nX; j++)
			{
				AS[k + j] = 0;
			}
		}
		break;
	case SSB_RIGHT:
		for (int i = 0; i < nY; i++)
		{
			int k = i * nX;
			for (int j = 0; j < half_nX; j++)
			{
				AS[k + j] = 0;
			}
		}
		break;
	case SSB_TOP:
		memset(&AS[half_N], 0, sizeof(Complex<Real>) * half_N);
		break;

	case SSB_BOTTOM:
		memset(&AS[0], 0, sizeof(Complex<Real>) * half_N);
		break;
	}

	Complex<Real>* filtered = new Complex<Real>[N];
	//fft2(holosize, AS, OPH_BACKWARD, OPH_ESTIMATE);
	fft2(AS, filtered, nX, nY, OPH_BACKWARD, false);

	//fftExecute(filtered);


	Real* realFiltered = new Real[N];
	oph::realPart<Real>(filtered, realFiltered, N);

	oph::normalize(realFiltered, dst, N);

	delete[] AS, filtered, realFiltered;
}

void ophGen::freqShift(Complex<Real>* src, Complex<Real>* dst, const ivec2 holosize, int shift_x, int shift_y)
{
	int N = holosize[_X] * holosize[_Y];

	Complex<Real>* AS = new Complex<Real>[N];
	//fft2(holosize, src, OPH_FORWARD, OPH_ESTIMATE);
	fft2(src, AS, holosize[_X], holosize[_Y], OPH_FORWARD);
	//fftExecute(AS);

	Complex<Real>* shifted = new Complex<Real>[N];
	circShift<Complex<Real>>(AS, shifted, shift_x, shift_y, holosize.v[_X], holosize.v[_Y]);

	//fft2(holosize, shifted, OPH_BACKWARD, OPH_ESTIMATE);
	fft2(shifted, dst, holosize[_X], holosize[_Y], OPH_BACKWARD);
	//fftExecute(dst);

	delete[] AS;
	delete[] shifted;
}


bool ophGen::saveRefImages(char* fnameW, char* fnameWC, char* fnameAS, char* fnameSSB, char* fnameHP, char* fnameFreq, char* fnameReal, char* fnameBin, char* fnameReconBin, char* fnameReconErr, char* fnameReconNo)
{
	ivec2 holosize = context_.pixel_number;
	int nx = holosize[_X], ny = holosize[_Y];
	int ss = nx*ny;

	Real* temp1 = new Real[ss];
	uchar* temp2 = new uchar[ss];

	oph::normalize(weight, temp2, nx, ny);
	saveAsImg(fnameW, 8, temp2, nx, ny);
	cout << "W saved" << endl;

	oph::absCplxArr<Real>(weightC, temp1, ss);
	oph::normalize(temp1, temp2, nx, ny);
	saveAsImg(fnameWC, 8, temp2, nx, ny);
	cout << "WC saved" << endl;

	oph::absCplxArr<Real>(AS, temp1, ss);
	oph::normalize(temp1, temp2, nx, ny);
	saveAsImg(fnameAS, 8, temp2, nx, ny);
	cout << "AS saved" << endl;

	oph::normalize(maskSSB, temp2, nx, ny);
	saveAsImg(fnameSSB, 8, temp2, nx, ny);
	cout << "SSB saved" << endl;

	oph::normalize(maskHP, temp2, nx, ny);
	saveAsImg(fnameHP, 8, temp2, nx, ny);
	cout << "HP saved" << endl;

	oph::absCplxArr<Real>(freqW, temp1, ss);
	oph::normalize(temp1, temp2, nx, ny);
	saveAsImg(fnameFreq, 8, temp2, nx, ny);
	cout << "Freq saved" << endl;

	oph::normalize(realEnc, temp2, nx, ny);
	saveAsImg(fnameReal, 8, temp2, nx, ny);
	cout << "Real saved" << endl;

	oph::normalize(binary, temp2, nx, ny);
	saveAsImg(fnameBin, 8, temp2, nx, ny);
	cout << "Bin saved" << endl;


	Complex<Real>* temp = new Complex<Real>[ss];
	for (int i = 0; i < ss; i++) {
		temp[i][_RE] = binary[i];
		temp[i][_IM] = 0;
	}
	fft2(ivec2(nx, ny), temp, OPH_FORWARD);
	fft2(temp, temp, nx, ny, OPH_FORWARD);
	for (int i = 0; i < ss; i++) {
		temp[i][_RE] *= maskSSB[i];
		temp[i][_IM] *= maskSSB[i];
	}
	fft2(ivec2(nx, ny), temp, OPH_BACKWARD);
	fft2(temp, temp, nx, ny, OPH_BACKWARD);

	Complex<Real>* reconBin = new Complex<Real>[ss];
	memsetArr<Complex<Real>>(reconBin, (0, 0), 0, ss - 1);
	fresnelPropagation(temp, reconBin, 0.001, 0);

	oph::absCplxArr<Real>(reconBin, temp1, ss);
	for (int i = 0; i < ss; i++) {
		temp1[i] = temp1[i] * temp1[i];
	}
	oph::normalize(temp1, temp2, nx, ny);
	saveAsImg(fnameReconBin, 8, temp2, nx, ny);
	cout << "recon bin saved" << endl;


	temp = new Complex<Real>[ss];
	for (int i = 0; i < ss; i++) {
		temp[i][_RE] = m_lpEncoded[0][i];
		temp[i][_IM] = 0;
	}
	fft2(ivec2(nx, ny), temp, OPH_FORWARD);
	fft2(temp, temp, holosize[_X], holosize[_Y], OPH_FORWARD);
	for (int i = 0; i < ss; i++) {
		temp[i][_RE] *= maskHP[i];
		temp[i][_IM] *= maskHP[i];
	}
	fft2(ivec2(nx, ny), temp, OPH_BACKWARD);
	fft2(temp, temp, nx, ny, OPH_BACKWARD);

	reconBin = new Complex<Real>[ss];
	fresnelPropagation(temp, reconBin, 0.001, 0);

	oph::absCplxArr<Real>(reconBin, temp1, ss);
	for (int i = 0; i < ss; i++) {
		temp1[i] = temp1[i] * temp1[i];
	}
	oph::normalize(temp1, temp2, nx, ny);
	saveAsImg(fnameReconErr, 8, temp2, nx, ny);
	cout << "recon error saved" << endl;




	temp = new Complex<Real>[ss];
	for (int i = 0; i < ss; i++) {
		temp[i][_RE] = normalized[i][_RE];
		temp[i][_IM] = normalized[i][_IM];
	}
	fft2(ivec2(nx, ny), temp, OPH_FORWARD);
	fft2(temp, temp, holosize[_X], holosize[_Y], OPH_FORWARD);
	for (int i = 0; i < ss; i++) {
		temp[i][_RE] *= maskSSB[i];
		temp[i][_IM] *= maskSSB[i];
	}
	fft2(ivec2(nx, ny), temp, OPH_BACKWARD);
	fft2(temp, temp, nx, ny, OPH_BACKWARD);

	reconBin = new Complex<Real>[ss];
	fresnelPropagation(temp, reconBin, 0.001, 0);

	oph::absCplxArr<Real>(reconBin, temp1, ss);
	for (int i = 0; i < ss; i++) {
		temp1[i] = temp1[i] * temp1[i];
	}
	oph::normalize(temp1, temp2, nx, ny);
	saveAsImg(fnameReconNo, 8, temp2, nx, ny);


	return true;
}

bool ophGen::binaryErrorDiffusion(Complex<Real>* holo, Real* encoded, const ivec2 holosize, const int type, Real threshold)
{

	//cout << "\nin?" << endl;
	int ss = holosize[_X] * holosize[_Y];
	weight = new Real[ss];
	//cout << "?" << endl;
	weightC = new Complex<Real>[ss];
	//cout << "??" << endl;
	ivec2 Nw;
	memsetArr<Real>(weight, 0.0, 0, ss - 1);
	//cout << "???" << endl;
	if (!getWeightED(holosize, type, &Nw))
		return false;
	//cout << "1?" << endl;
	AS = new Complex<Real>[ss];
	fft2(ivec2(holosize[_X], holosize[_Y]), holo, OPH_FORWARD);
	fft2(holo, AS, holosize[_X], holosize[_Y], OPH_FORWARD);
	//cout << "2?" << endl;
	// SSB mask generation
	maskSSB = new Real[ss];
	for (int i = 0; i < ss; i++) {
		if (((Real)i / (Real)holosize[_X]) < ((Real)holosize[_Y] / 2.0))
			maskSSB[i] = 1;
		else
			maskSSB[i] = 0;
		AS[i] *= maskSSB[i];
	}

	//cout << "3?" << endl;
	Complex<Real>* filtered = new Complex<Real>[ss];
	fft2(ivec2(holosize[_X], holosize[_Y]), AS, OPH_BACKWARD);
	fft2(AS, filtered, holosize[_X], holosize[_Y], OPH_BACKWARD);
	//cout << "4?" << endl;
	normalized = new Complex<Real>[ss];
	oph::normalize(filtered, normalized, ss);
	LOG("normalize finishied..\n");
	if (encoded == nullptr)
		encoded = new Real[ss];

	shiftW(holosize);
	LOG("shiftW finishied..\n");

	// HP mask generation
	maskHP = new Real[ss];
	Real absFW;
	for (int i = 0; i < ss; i++) {
		oph::absCplx<Real>(freqW[i], absFW);
		if (((Real)i / (Real)holosize[_X]) < ((Real)holosize[_Y] / 2.0) && absFW < 0.6)
			maskHP[i] = 1;
		else
			maskHP[i] = 0;
	}
	//cout << "5?" << endl;

	// For checking
	binary = new Real[ss];
	realEnc = new Real[ss];
	for (int i = 0; i < ss; i++) {
		realEnc[i] = normalized[i][_RE];
		if (normalized[i][_RE] > threshold)
			binary[i] = 1;
		else
			binary[i] = 0;
	}

	Complex<Real>* toBeBin = new Complex<Real>[ss];
	for (int i = 0; i < ss; i++) {
		toBeBin[i] = normalized[i];
	}
	int ii, iii, jjj;
	int cx = (holosize[_X] + 1) / 2, cy = (holosize[_Y] + 1) / 2;
	Real error;
	for (int iy = 0; iy < holosize[_Y] - Nw[_Y]; iy++) {
		for (int ix = Nw[_X]; ix < holosize[_X] - Nw[_X]; ix++) {

			ii = ix + iy*holosize[_X];
			if (ix >= Nw[_X] && ix < (holosize[_X] - Nw[_X]) && iy < (holosize[_Y] - Nw[_Y])) {

				if (toBeBin[ii][_RE] > threshold)
					encoded[ii] = 1;
				else
					encoded[ii] = 0;

				error = toBeBin[ii][_RE] - encoded[ii];

				for (int iwy = 0; iwy < Nw[_Y] + 1; iwy++) {
					for (int iwx = -Nw[_X]; iwx < Nw[_X] + 1; iwx++) {
						iii = (ix + iwx) + (iy + iwy)*holosize[_X];
						jjj = (cx + iwx) + (cy + iwy)*holosize[_X];

						toBeBin[iii] += weightC[jjj] * error;
					}
				}
			}
			else {
				encoded[ii] = 0;
			}
		}
	}
	LOG("binary finishied..\n");

	return true;
}


bool ophGen::getWeightED(const ivec2 holosize, const int type, ivec2* pNw)
{

	int cx = (holosize[_X] + 1) / 2;
	int cy = (holosize[_Y] + 1) / 2;

	ivec2 Nw;

	switch (type) {
	case FLOYD_STEINBERG:
		LOG("ERROR DIFFUSION : FLOYD_STEINBERG\n");
		weight[(cx + 1) + cy*holosize[_X]] = 7.0 / 16.0;
		weight[(cx - 1) + (cy + 1)*holosize[_X]] = 3.0 / 16.0;
		weight[(cx)+(cy + 1)*holosize[_X]] = 5.0 / 16.0;
		weight[(cx + 1) + (cy + 1)*holosize[_X]] = 1.0 / 16.0;
		Nw[_X] = 1;  Nw[_Y] = 1;
		break;
	case SINGLE_RIGHT:
		LOG("ERROR DIFFUSION : SINGLE_RIGHT\n");
		weight[(cx + 1) + cy*holosize[_X]] = 1.0;
		Nw[_X] = 1;  Nw[_Y] = 0;
		break;
	case SINGLE_DOWN:
		LOG("ERROR DIFFUSION : SINGLE_DOWN\n");
		weight[cx + (cy + 1)*holosize[_X]] = 1.0;
		Nw[_X] = 0;  Nw[_Y] = 1;
		break;
	default:
		LOG("<FAILED> WRONG PARAMETERS.\n");
		return false;
	}

	*pNw = Nw;
	return true;

}

bool ophGen::shiftW(ivec2 holosize) {

	int ss = holosize[_X] * holosize[_Y];

	Complex<Real> term(0, 0);
	Complex<Real> temp(0, 0);
	int x, y;
	for (uint i = 0; i < ss; i++) {

		x = i%holosize[_X] - holosize[_X] / 2;  y = i / holosize[_X] - holosize[_Y];
		term[_IM] = 2.0 * M_PI*((1.0 / 4.0)*(Real)x + (0.0)*(Real)y);
		temp[_RE] = weight[i];
		weightC[i] = temp *exp(term);
	}

	freqW = new Complex<Real>[ss];

	fft2(ivec2(holosize[_X], holosize[_Y]), weightC, OPH_FORWARD);
	fft2(weightC, freqW, holosize[_X], holosize[_Y], OPH_FORWARD);
	for (int i = 0; i < ss; i++) {
		freqW[i][_RE] -= 1.0;
	}
	return true;

}

void ophGen::binarization(Complex<Real>* src, Real* dst, const int size, int ENCODE_FLAG, Real threshold)
{
	oph::normalize(src, src, size);
	encoding(ENCODE_FLAG);

	int i;
#ifdef _OPENMP
#pragma omp parallel for private(i) firstprivate(threshold)
#endif
	for (i = 0; i < size; i++) {
		if (src[i][_RE] > threshold)
			dst[i] = 1;
		else
			dst[i] = 0;
	}
}

void ophGen::fresnelPropagation(OphConfig context, Complex<Real>* in, Complex<Real>* out, Real distance)
{
	const int pnX = context.pixel_number[_X];
	const int pnY = context.pixel_number[_Y];
	const uint pnXY = pnX * pnY;

	Complex<Real>* in2x = new Complex<Real>[pnXY * 4];
	Complex<Real> zero(0, 0);
	memset(in2x, 0, sizeof(Complex<Real>) * pnXY * 4);

	uint idxIn = 0;
	int beginY = pnY >> 1;
	int beginX = pnX >> 1;
	int endY = pnY + beginY;
	int endX = pnX + beginX;
	
	for (int idxnY = beginY; idxnY < endY; idxnY++) {
		for (int idxnX = beginX; idxnX < endX; idxnX++) {
			in2x[idxnY * pnX * 2 + idxnX] = in[idxIn++];
		}
	}


	Complex<Real>* temp1 = new Complex<Real>[pnXY * 4];

	fft2({ pnX * 2, pnY * 2 }, in2x, OPH_FORWARD, OPH_ESTIMATE);
	fft2(in2x, temp1, pnX * 2, pnY * 2, OPH_FORWARD);
	//fftExecute(temp1);
	Real* fx = new Real[pnXY * 4];
	Real* fy = new Real[pnXY * 4];

	uint i = 0;
	for (int idxFy = -pnY; idxFy < pnY; idxFy++) {
		for (int idxFx = -pnX; idxFx < pnX; idxFx++) {
			fx[i] = idxFx / (2 * pnX * context.pixel_pitch[_X]);
			fy[i] = idxFy / (2 * pnY * context.pixel_pitch[_Y]);
			i++;
		}
	}

	Complex<Real>* prop = new Complex<Real>[pnXY * 4];
	memsetArr<Complex<Real>>(prop, zero, 0, pnXY * 4 - 1);

	Real sqrtPart;

	Complex<Real>* temp2 = new Complex<Real>[pnXY * 4];

	for (int i = 0; i < pnXY * 4; i++) {
		sqrtPart = sqrt(1 / (context.wave_length[0] * context.wave_length[0]) - fx[i] * fx[i] - fy[i] * fy[i]);
		prop[i][_IM] = 2 * M_PI * distance;
		prop[i][_IM] *= sqrtPart;
		temp2[i] = temp1[i] * exp(prop[i]);
	}

	Complex<Real>* temp3 = new Complex<Real>[pnXY * 4];
	fft2({ pnX * 2, pnY * 2 }, temp2, OPH_BACKWARD, OPH_ESTIMATE);
	fft2(temp2, temp3, pnX * 2, pnY * 2, OPH_BACKWARD);
	//fftExecute(temp3);

	uint idxOut = 0;

	for (int idxNy = pnY / 2; idxNy < pnY + (pnY / 2); idxNy++) {
		for (int idxNx = pnX / 2; idxNx < pnX + (pnX / 2); idxNx++) {
			out[idxOut] = temp3[idxNy * pnX * 2 + idxNx];
			idxOut++;
		}
	}

	delete[] in2x;
	delete[] temp1;
	delete[] fx;
	delete[] fy;
	delete[] prop;
	delete[] temp2;
	delete[] temp3;
}

void ophGen::fresnelPropagation(Complex<Real>* in, Complex<Real>* out, Real distance, uint channel)
{
	auto begin = CUR_TIME;

	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];
	const int pnXY = pnX * pnY;
	const Real lambda = context_.wave_length[channel];
	const Real ssX = pnX * ppX * 2;
	const Real ssY = pnY * ppY * 2;
	const Real z = 2 * M_PI * distance;
	const Real v = 1 / (lambda * lambda);
	const int hpnX = pnX / 2;
	const int hpnY = pnY / 2;
	const int pnX2 = pnX * 2;
	const int pnY2 = pnY * 2;

	Complex<Real>* temp = new Complex<Real>[pnXY * 4];
	memset(temp, 0, sizeof(Complex<Real>) * pnXY * 4);

	int i;
#ifdef _OPENMP
#pragma omp parallel for private(i) firstprivate(pnX, pnX2, hpnX, hpnY)
#endif
	for (i = 0; i < pnY; i++)
	{
		int src = pnX * i;
		int dst = pnX2 * (i + hpnY) + hpnX;
		memcpy(&temp[dst], &in[src], sizeof(Complex<Real>) * pnX);
	}
	
	fft2(temp, temp, pnX2, pnY2, OPH_FORWARD, false);
	
	int j;
#ifdef _OPENMP
#pragma omp parallel for private(j) firstprivate(ssX, ssY, z, v)
#endif
	for (j = 0; j < pnY2; j++)
	{
		Real fy = (-pnY + j) / ssY;
		Real fyy = fy * fy;
		int iWidth = j * pnX2;
		for (int i = 0; i < pnX2; i++)
		{
			Real fx = (-pnX + i) / ssX;
			Real fxx = fx * fx;

			Real sqrtPart = sqrt(v - fxx - fyy);
			Complex<Real> prop(0, z * sqrtPart);
			temp[iWidth + i] *= prop.exp();
		}
	}

	fft2(temp, temp, pnX2, pnY2, OPH_BACKWARD, true);

#ifdef _OPENMP
#pragma omp parallel for private(i) firstprivate(pnX, pnX2, hpnX, hpnY)
#endif
	for (i = 0; i < pnY; i++)
	{
		int src = pnX2 * (i + hpnY) + hpnX;
		int dst = pnX * i;
		memcpy(&out[dst], &temp[src], sizeof(Complex<Real>) * pnX);
	}
	delete[] temp;

}

bool ophGen::Shift(Real x, Real y)
{
	if (x == 0.0 && y == 0.0) return false;
	
	bool bAxisX = (x == 0.0) ? false : true;
	bool bAxisY = (y == 0.0) ? false : true;
	const int nChannel = context_.waveNum;
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];
	const uint pnXY = pnX * pnY;
	const vec2 ss = context_.ss;
	Real ppY2 = ppY * 2;
	Real ppX2 = ppX * 2;
	Real ssX = -ss[_X] / 2;
	Real ssY = -ss[_Y] / 2;

	Real *waveRatio = new Real[nChannel];

	Complex<Real> pi2(0.0, -2 * M_PI);
	int num_threads = 1;

	for (int i = 0; i < nChannel; i++) {
		waveRatio[i] = context_.wave_length[nChannel - 1] / context_.wave_length[i];

		Real ratioX = x * waveRatio[i];
		Real ratioY = y * waveRatio[i];
		int y;

#ifdef _OPENMP
#pragma omp parallel for private(y) firstprivate(ppX, ppY, ppX2, ppY2, ssX, ssY, pi2)
#endif
		for (y = 0; y < pnY; y++) {
			Complex<Real> yy(0, 0);
			if (bAxisY) {
				Real startY = ssY + (ppY * y);
				Real shiftY = startY / ppY2 * ratioY;
				yy = (pi2 * shiftY).exp();
			}
			int offset = y * pnX;

			for (int x = 0; x < pnX; x++) {
				if (bAxisY) {
					complex_H[i][offset + x] = complex_H[i][offset + x] * yy;
				}
				if (bAxisX) {
					Real startX = ssX + (ppX * x);
					Real shiftX = startX / ppX2 * ratioX;
					Complex<Real> xx = (pi2 * shiftX).exp();
					complex_H[i][offset + x] = complex_H[i][offset + x] * xx;
				}
			}
		}
	}
	return true;
}

void ophGen::waveCarry(Real carryingAngleX, Real carryingAngleY, Real distance)
{
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];
	const int pnXY = pnX * pnY;
	const int nChannel = context_.waveNum;
	Real dfx = 1 / ppX / pnX;
	Real dfy = 1 / ppY / pnY;
	Real* fx = new Real[pnXY];
	Real* fy = new Real[pnXY];
	Real* fz = new Real[pnXY];
	int i = 0;
	for (int ch = 0; ch < nChannel; ch++) {
		Real lambda = context_.wave_length[ch];

		for (int idxFy = pnY / 2; idxFy > -pnY / 2; idxFy--) {
			for (int idxFx = -pnX / 2; idxFx < pnX / 2; idxFx++) {
				fx[i] = idxFx * dfx;
				fy[i] = idxFy * dfy;
				fz[i] = sqrt((1 / lambda)*(1 / lambda) - fx[i] * fx[i] - fy[i] * fy[i]);

				i++;
			}
		}

		Complex<Real> carrier;
		for (int i = 0; i < pnXY; i++) {
			carrier[_RE] = 0;
			carrier[_IM] = distance * tan(carryingAngleX)*fx[i] + distance * tan(carryingAngleY)*fy[i];
			complex_H[ch][i] = complex_H[ch][i] * exp(carrier);
		}
	}
	delete[] fx;
	delete[] fy;
	delete[] fz;
}

void ophGen::waveCarry(Complex<Real>* src, Complex<Real>* dst, Real wavelength, int carryIdxX, int carryIdxY)
{
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const int pnXY = pnX * pnY;
	const int nChannel = context_.waveNum;
	Real dfx = 1 / context_.pixel_pitch[_X] / pnX;
	Real dfy = 1 / context_.pixel_pitch[_Y] / pnY;
	Real* fx = new Real[pnXY];
	Real* fy = new Real[pnXY];
	Real* fz = new Real[pnXY];
	uint i = 0;

	for (int idxFy = pnY / 2; idxFy > -pnY / 2; idxFy--) {
		for (int idxFx = -pnX / 2; idxFx < pnX / 2; idxFx++) {
			fx[i] = idxFx * dfx;
			fy[i] = idxFy * dfy;
			fz[i] = sqrt((1 / wavelength)*(1 / wavelength) - fx[i] * fx[i] - fy[i] * fy[i]);

			i++;
		}
	}

	Complex<Real>* carrier = new Complex<Real>[pnXY];

	for (int i = 0; i < pnXY; i++) {
		carrier[i][_RE] = 0;
		carrier[i][_IM] = 2 * M_PI*(carryIdxX*context_.pixel_pitch[_X] * fx[i] + carryIdxY*context_.pixel_pitch[_Y] * fy[i]);
		dst[i] = src[i] * exp(carrier[i]);
	}

	delete[] carrier;
	delete[] fx;
	delete[] fy;
	delete[] fz;
}

void ophGen::encodeSideBand(bool bCPU, ivec2 sig_location)
{
	if (complex_H == nullptr) {
		LOG("Not found diffracted data.");
		return;
	}

	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];

	int cropx1, cropx2, cropx, cropy1, cropy2, cropy;
	if (sig_location[1] == 0) { //Left or right half
		cropy1 = 1;
		cropy2 = pnY;
	}
	else {
		cropy = (int)floor(((Real)pnY) / 2);
		cropy1 = cropy - (int)floor(((Real)cropy) / 2);
		cropy2 = cropy1 + cropy - 1;
	}

	if (sig_location[0] == 0) { // Upper or lower half
		cropx1 = 1;
		cropx2 = pnX;
	}
	else {
		cropx = (int)floor(((Real)pnX) / 2);
		cropx1 = cropx - (int)floor(((Real)cropx) / 2);
		cropx2 = cropx1 + cropx - 1;
	}

	cropx1 -= 1;
	cropx2 -= 1;
	cropy1 -= 1;
	cropy2 -= 1;

	if (bCPU)
		encodeSideBand_CPU(cropx1, cropx2, cropy1, cropy2, sig_location);
	else
		encodeSideBand_GPU(cropx1, cropx2, cropy1, cropy2, sig_location);
}

void ophGen::encodeSideBand_CPU(int cropx1, int cropx2, int cropy1, int cropy2, oph::ivec2 sig_location)
{
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;
	const uint nChannel = context_.waveNum;

	Complex<Real>* h_crop = new Complex<Real>[pnXY];

	for (uint ch = 0; ch < nChannel; ch++) {

		memset(h_crop, 0.0, sizeof(Complex<Real>) * pnXY);

		int p = 0;
#pragma omp parallel for private(p)
		for (p = 0; p < pnXY; p++)
		{
			int x = p % pnX;
			int y = p / pnX;
			if (x >= cropx1 && x <= cropx2 && y >= cropy1 && y <= cropy2)
				h_crop[p] = complex_H[ch][p];
		}

		Complex<Real> *in = nullptr;

		fft2(ivec2(pnX, pnY), in, OPH_BACKWARD);
		fft2(h_crop, h_crop, pnX, pnY, OPH_BACKWARD, true);

		memset(m_lpEncoded[ch], 0.0, sizeof(Real) * pnXY);
		int i = 0;
#pragma omp parallel for private(i)	
		for (i = 0; i < pnXY; i++) {
			Complex<Real> shift_phase(1, 0);
			getShiftPhaseValue(shift_phase, i, sig_location);

			m_lpEncoded[ch][i] = (h_crop[i] * shift_phase).real();
		}
	}
	delete[] h_crop;
}

void ophGen::encodeSideBand_GPU(int cropx1, int cropx2, int cropy1, int cropy2, oph::ivec2 sig_location)
{
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const int pnXY = pnX * pnY;
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];
	const Real ssX = context_.ss[_X] = pnX * ppX;
	const Real ssY = context_.ss[_Y] = pnY * ppY;
	const uint nChannel = context_.waveNum;

	cufftDoubleComplex *k_temp_d_, *u_complex_gpu_;
	cudaStream_t stream_;
	cudaStreamCreate(&stream_);

	cudaMalloc((void**)&u_complex_gpu_, sizeof(cufftDoubleComplex) * pnXY);
	cudaMalloc((void**)&k_temp_d_, sizeof(cufftDoubleComplex) * pnXY);

	for (uint ch = 0; ch < nChannel; ch++) {
		cudaMemcpy(u_complex_gpu_, complex_H[ch], sizeof(cufftDoubleComplex) * pnXY, cudaMemcpyHostToDevice);

		cudaMemsetAsync(k_temp_d_, 0, sizeof(cufftDoubleComplex) * pnXY, stream_);
		cudaCropFringe(stream_, pnX, pnY, u_complex_gpu_, k_temp_d_, cropx1, cropx2, cropy1, cropy2);

		cudaMemsetAsync(u_complex_gpu_, 0, sizeof(cufftDoubleComplex) * pnXY, stream_);
		cudaFFT(stream_, pnX, pnY, k_temp_d_, u_complex_gpu_, 1, true);

		cudaMemsetAsync(k_temp_d_, 0, sizeof(cufftDoubleComplex) * pnXY, stream_);
		cudaGetFringe(stream_, pnX, pnY, u_complex_gpu_, k_temp_d_, sig_location[0], sig_location[1], ssX, ssY, ppX, ppY, M_PI);

		cufftDoubleComplex* sample_fd = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex) * pnXY);
		memset(sample_fd, 0.0, sizeof(cufftDoubleComplex) * pnXY);

		cudaMemcpyAsync(sample_fd, k_temp_d_, sizeof(cufftDoubleComplex) * pnXY, cudaMemcpyDeviceToHost), stream_;
		memset(m_lpEncoded[ch], 0.0, sizeof(Real) * pnXY);

		for (int i = 0; i < pnX * pnY; i++)
			m_lpEncoded[ch][i] = sample_fd[i].x;

		cudaFree(sample_fd);
	}
	cudaStreamDestroy(stream_);
}

void ophGen::getShiftPhaseValue(oph::Complex<Real>& shift_phase_val, int idx, oph::ivec2 sig_location)
{
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];
	const Real ssX = context_.ss[_X] = pnX * ppX;
	const Real ssY = context_.ss[_Y] = pnY * ppY;

	if (sig_location[1] != 0)
	{
		int r = idx / pnX;
		int c = idx % pnX;
		Real yy = (ssY / 2.0) - (ppY)*r - ppY;

		oph::Complex<Real> val;
		if (sig_location[1] == 1)
			val[_IM] = 2 * M_PI * (yy / (4 * ppY));
		else
			val[_IM] = 2 * M_PI * (-yy / (4 * ppY));

		val.exp();
		shift_phase_val *= val;
	}

	if (sig_location[0] != 0)
	{
		int r = idx / pnX;
		int c = idx % pnX;
		Real xx = (-ssX / 2.0) - (ppX)*c - ppX;

		oph::Complex<Real> val;
		if (sig_location[0] == -1)
			val[_IM] = 2 * M_PI * (-xx / (4 * ppX));
		else
			val[_IM] = 2 * M_PI * (xx / (4 * ppX));

		val.exp();
		shift_phase_val *= val;
	}
}

void ophGen::GetRandomPhaseValue(Complex<Real>& rand_phase_val, bool rand_phase)
{
	if (rand_phase)
	{
		rand_phase_val[_RE] = 0.0;
		Real min, max;
#if REAL_IS_DOUBLE & true
		min = 0.0;
		max = 1.0;
#else
		min = 0.f;
		max = 1.f;
#endif
		rand_phase_val[_IM] = 2 * M_PI * oph::rand(min, max);
		rand_phase_val.exp();

	}
	else {
		rand_phase_val[_RE] = 1.0;
		rand_phase_val[_IM] = 0.0;
	}
}

void ophGen::setResolution(ivec2 resolution)
{
	// 기존 해상도와 다르면 버퍼를 다시 생성.
	if (context_.pixel_number != resolution) {
		setPixelNumber(resolution);
		Openholo::setPixelNumberOHC(resolution);
		initialize();
	}
}

template <typename T>
void ophGen::RealPart(Complex<T> *holo, T *encoded, const int size)
{
	int i;
#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
	for (i = 0; i < size; i++) {
		encoded[i] = real(holo[i]);
	}
}

template <typename T>
void ophGen::ImaginaryPart(Complex<T> *holo, T *encoded, const int size)
{
	int i;
#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
	for (i = 0; i < size; i++) {
		encoded[i] = imag(holo[i]);
	}
}

template <typename T>
void ophGen::Phase(Complex<T> *holo, T *encoded, const int size)
{
	int i;
	double pi2 = M_PI * 2;
#ifdef _OPENMP
#pragma omp parallel for private(i) firstprivate(pi2)
#endif
	for (i = 0; i < size; i++) {
		encoded[i] = (holo[i].angle() + M_PI) / pi2; // 0 ~ 1
	}
}

template <typename T>
void ophGen::Amplitude(Complex<T> *holo, T *encoded, const int size)
{
	int i;
#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
	for (i = 0; i < size; i++) {
		encoded[i] = holo[i].mag();
	}
}

template <typename T>
void ophGen::TwoPhase(Complex<T>* holo, T* encoded, const int size)
{
	int resize = size >> 1;
	int i;
	Complex<T>* normCplx = new Complex<T>[resize];

#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
	for (i = 0; i < resize; i++) {
		normCplx[i] = holo[i * 2];
	}

	oph::normalize<T>(normCplx, normCplx, resize);

	T* ampl = new T[resize];
	Amplitude(normCplx, ampl, resize);

	T* phase = new T[resize];
	Phase(normCplx, phase, resize);

#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
	for (i = 0; i < resize; i++) {
		T delPhase = acos(ampl[i]);
		encoded[i * 2] = (phase[i] + M_PI) + delPhase;
		encoded[i * 2 + 1] = (phase[i] + M_PI) - delPhase;
	}

	delete[] normCplx;
	delete[] ampl;
	delete[] phase;
}

template <typename T>
void ophGen::Burckhardt(Complex<T>* holo, T* encoded, const int size)
{
	int resize = size / 3;
	int i;
	Complex<T>* norm = new Complex<T>[resize];
#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
	for (i = 0; i < resize; i++) {
		norm[i] = holo[i * 3];
	}

	oph::normalize(norm, norm, resize);

	T* phase = new T[resize];
	Phase(norm, phase, resize);

	T* ampl = new T[resize];
	Amplitude(norm, ampl, resize);

	T sqrt3 = sqrt(3);
	T pi2 = 2 * M_PI;
	T pi4 = 4 * M_PI;

#ifdef _OPENMP
#pragma omp parallel for private(i) firstprivate(pi2, pi4, sqrt3)
#endif
	for (i = 0; i < resize; i++) {
		int idx = 3 * i;
		if (phase[i] >= 0 && phase[i] < (pi2 / 3))
		{
			encoded[idx] = ampl[i] * (cos(phase[i]) + sin(phase[i]) / sqrt3);
			encoded[idx + 1] = 2 * sin(phase[i]) / sqrt3;
		}
		else if (phase[i] >= (pi2 / 3) && phase[i] < (pi4 / 3))
		{
			encoded[idx + 1] = ampl[i] * (cos(phase[i] - (pi2 / 3)) + sin(phase[i] - (pi2 / 3)) / sqrt3);
			encoded[idx + 2] = 2 * sin(phase[i] - (pi2 / 3)) / sqrt3;
		}
		else if (phase[i] >= (pi4 / 3) && phase[i] < (pi2))
		{
			encoded[idx + 2] = ampl[i] * (cos(phase[i] - (pi4 / 3)) + sin(phase[i] - (pi4 / 3)) / sqrt3);
			encoded[idx] = 2 * sin(phase[i] - (pi4 / 3)) / sqrt3;
		}
	}

	delete[] ampl;
	delete[] phase;
	delete[] norm;
}


template <typename T>
void ophGen::SimpleNI(Complex<T>* holo, T* encoded, const int size)
{
	T* tmp1 = new T[size];
	int i;
#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
	for (i = 0; i < size; i++) {
		tmp1[i] = holo[i].mag();
	}

	T max = maxOfArr(tmp1, size);
	delete[] tmp1;

#ifdef _OPENMP
#pragma omp parallel for private(i) firstprivate(max)
#endif
	for (i = 0; i < size; i++) {
		T tmp = (holo[i] + max).mag();
		encoded[i] = tmp * tmp;
	}
}

void ophGen::transVW(int nSize, Real *dst, Real *src)
{
	Real fieldLens = m_dFieldLength;
	for (int i = 0; i < nSize; i++) {
		*(dst + i) = -fieldLens * src[i] / (src[i] - fieldLens);
	}
}

void ophGen::ScaleChange(Real *src, Real *dst, int nSize, Real scaleX, Real scaleY, Real scaleZ)
{
	Real x = scaleX;
	Real y = scaleY;
	Real z = scaleZ;
	int i;
#ifdef _OPENMP
#pragma omp parallel for private(i) firstprivate(x, y, z)
#endif
	for (i = 0; i < nSize; i++) {
		dst[i + 0] = src[i + 0] * x;
		dst[i + 1] = src[i + 1] * y;
		dst[i + 2] = src[i + 2] * z;
	}
}

void ophGen::GetMaxMin(Real *src, int len, Real& max, Real& min)
{
	Real maxTmp = MIN_DOUBLE;
	Real minTmp = MAX_DOUBLE;

	for (int i = 0; i < len; i++) {
		if (src[i] > maxTmp)
			maxTmp = src[i];
		if (src[i] < minTmp)
			minTmp = src[i];
	}
	max = maxTmp;
	min = minTmp;
}


void ophGen::CorrectionChromaticAberration(uchar* src, uchar* dst, int width, int height, int ch)
{
	if (ch < 2)
	{
		const int nWave = context_.waveNum;
		const int pnXY = width * height;
		const Real lambda = context_.wave_length[ch];
		const Real waveRatio = nWave > 1 ? context_.wave_length[nWave - 1] / lambda : 1.0;

		int scaleX = round(width * 4 * waveRatio);
		int scaleY = round(height * 4 * waveRatio);

		int ww = width * 4;
		int hh = height * 4;
		int nSize = ww * hh;
		int nScaleSize = scaleX * scaleY;
		uchar *img_tmp = new uchar[nSize];
		uchar *imgScaled = new uchar[nScaleSize];
		imgScaleBilinear(src, imgScaled, width, height, scaleX, scaleY);

		memset(img_tmp, 0, sizeof(uchar) * nSize);

		int h1 = round((hh - scaleY) / 2);
		int w1 = round((ww - scaleX) / 2);
		int y;
#ifdef _OPENMP
#pragma omp parallel for private(y) firstprivate(w1, h1, scaleX, ww)
#endif
		for (y = 0; y < scaleY; y++) {
			for (int x = 0; x < scaleX; x++) {
				img_tmp[(y + h1)*ww + x + w1] = imgScaled[y*scaleX + x];
			}
		}
		imgScaleBilinear(img_tmp, dst, ww, hh, width, height);

		delete[] img_tmp;
		delete[] imgScaled;
	}
}


void ophGen::ophFree(void)
{
	Openholo::ophFree();

	const int nChannel = context_.waveNum;

	if (m_lpEncoded != nullptr) {
		for (uint i = 0; i < nChannel; i++) {
			if (m_lpEncoded[i] != nullptr) {
				delete[] m_lpEncoded[i];
				m_lpEncoded[i] = nullptr;
			}
		}
		delete[] m_lpEncoded;
		m_lpEncoded = nullptr;
	}

	if (m_lpNormalized != nullptr) {
		for (uint i = 0; i < nChannel; i++) {
			if (m_lpNormalized[i] != nullptr) {
				delete[] m_lpNormalized[i];
				m_lpNormalized[i] = nullptr;
			}
		}
		delete[] m_lpNormalized;
		m_lpNormalized = nullptr;
	}
}
