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
#include <windows.h>
#include "sys.h"
#include "function.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <omp.h>
#include "tinyxml2.h"
#include "PLYparser.h"
//#include "OpenCL.h"
//#include "CUDA.h"


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
	, holo_encoded(nullptr)
	, holo_normalized(nullptr)
	, nOldChannel(0)
	, elapsedTime(0.0)
	, m_nFieldLength(0.0)
	, m_nStream(1)
{
	//OpenCL::getInstance();
	//CUDA::getInstance();
}

ophGen::~ophGen(void)
{
	//OpenCL::releaseInstance();
	//CUDA::releaseInstance();
}

void ophGen::initialize(void)
{
	LOG("[%s]\n", __FUNCTION__);
	// Output Image Size
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;
	const int nChannel = context_.waveNum;

	// Memory Location for Result Image
	if (complex_H != nullptr) {
		for (uint i = 0; i < nOldChannel; i++) {
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

	if (holo_encoded != nullptr) {
		for (uint i = 0; i < nOldChannel; i++) {
			if (holo_encoded[i] != nullptr) {
				delete[] holo_encoded[i];
				holo_encoded[i] = nullptr;
			}
		}
		delete[] holo_encoded;
		holo_encoded = nullptr;
	}
	holo_encoded = new Real*[nChannel];
	for (uint i = 0; i < nChannel; i++) {
		holo_encoded[i] = new Real[pnXY];
		memset(holo_encoded[i], 0, sizeof(Real) * pnXY);
	}

	if (holo_normalized != nullptr) {
		for (uint i = 0; i < nOldChannel; i++) {
			if (holo_normalized[i] != nullptr) {
				delete[] holo_normalized[i];
				holo_normalized[i] = nullptr;
			}
		}
		delete[] holo_normalized;
		holo_normalized = nullptr;
	}
	holo_normalized = new uchar*[nChannel];
	for (uint i = 0; i < nChannel; i++) {
		holo_normalized[i] = new uchar[pnXY];
		memset(holo_normalized[i], 0, sizeof(uchar) * pnXY);
	}

	nOldChannel = nChannel;
	encode_size[_X] = pnX;
	encode_size[_Y] = pnY;
}

int ophGen::loadPointCloud(const char* pc_file, OphPointCloudData *pc_data_)
{
	LOG("[%s] %s\n", __FUNCTION__, pc_file);
	auto begin = CUR_TIME;

	PLYparser plyIO;
	if (!plyIO.loadPLY(pc_file, pc_data_->n_points, pc_data_->n_colors, &pc_data_->vertex, &pc_data_->color, &pc_data_->phase, pc_data_->isPhaseParse))
		return -1;

	auto end = CUR_TIME;
	LOG("%.5lfsec...done\n", ELAPSED_TIME(begin, end));
	return pc_data_->n_points;
}

bool ophGen::readConfig(const char* fname)
{
	LOG("[%s] %s\n", __FUNCTION__, fname);
	using namespace tinyxml2;

	auto begin = CUR_TIME;
	tinyxml2::XMLDocument xml_doc;
	XMLNode *xml_node = nullptr;

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

	int nWave = 1;
	auto next = xml_node->FirstChildElement("SLM_WaveNum"); // OffsetInDepth
	if (!next || XML_SUCCESS != next->QueryIntText(&nWave))
		return false;

	context_.waveNum = nWave;
	if (context_.wave_length) delete[] context_.wave_length;
	context_.wave_length = new Real[nWave];

	char szNodeName[32] = { 0, };
	for (int i = 1; i <= nWave; i++) {
		wsprintfA(szNodeName, "SLM_WaveLength_%d", i);
		next = xml_node->FirstChildElement(szNodeName);
		if (!next || XML_SUCCESS != next->QueryDoubleText(&context_.wave_length[i - 1]))
			return false;
	}
	next = xml_node->FirstChildElement("SLM_PixelNumX");
	if (!next || XML_SUCCESS != next->QueryIntText(&context_.pixel_number[_X]))
		return false;
	next = xml_node->FirstChildElement("SLM_PixelNumY");
	if (!next || XML_SUCCESS != next->QueryIntText(&context_.pixel_number[_Y]))
		return false;
	next = xml_node->FirstChildElement("SLM_PixelPitchX");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&context_.pixel_pitch[_X]))
		return false;
	next = xml_node->FirstChildElement("SLM_PixelPitchY");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&context_.pixel_pitch[_Y]))
		return false;
	next = xml_node->FirstChildElement("IMG_Rotation");
	if (!next || XML_SUCCESS != next->QueryBoolText(&context_.bRotation))
		context_.bRotation = false;
	next = xml_node->FirstChildElement("IMG_Merge");
	if (!next || XML_SUCCESS != next->QueryBoolText(&context_.bMergeImg))
		context_.bMergeImg = true;
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
	if (!next || XML_SUCCESS != next->QueryDoubleText(&m_nFieldLength))
		m_nFieldLength = 0.0;
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

	auto end = CUR_TIME;
	LOG("%.5lfsec...done\n", ELAPSED_TIME(begin, end));

	return true;
}

void ophGen::propagationAngularSpectrum(int ch, Complex<Real>* input_u, Real propagation_dist, Real k, Real lambda)
{
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];
	const Real ssX = context_.ss[_X] = pnX * ppX;
	const Real ssY = context_.ss[_Y] = pnY * ppY;
	int i;

#if 0
#ifdef _OPENMP
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
#pragma omp for private(i)
#endif
		for (i = 0; i < pnY; i++) {
			int idx = i * pnX;
			for (int j = 0; j < pnX; j++) {
				//Real x = i % pnX;
				//Real y = i / pnX;

				Real fxx = (-1.0 / (2.0*ppX)) + (1.0 / ssX) * j;
				Real fyy = (1.0 / (2.0*ppY)) - (1.0 / ssY) - (1.0 / ssY) * i;

				Real fxxx = lambda * fxx;
				Real fyyy = lambda * fyy;

				Real sval = sqrt(1 - (fxxx * fxxx) - (fyyy * fyyy));
				sval *= k * propagation_dist;
				Complex<Real> kernel(0, sval);
				kernel.exp();

				int prop_mask = ((fxx * fxx + fyy * fyy) < (k * k)) ? 1 : 0;

				Complex<Real> u_frequency;

				if (prop_mask == 1) {
					u_frequency = kernel * input_u[idx + j];
#ifdef _OPENMP
#pragma omp atomic
#endif
					complex_H[ch][idx + j][_RE] += u_frequency[_RE];
#ifdef _OPENMP
#pragma omp atomic
#endif
					complex_H[ch][idx + j][_IM] += u_frequency[_IM];
				}
			}
		}
#ifdef _OPENMP
	}
#endif

#else

#ifdef _OPENMP
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
#pragma omp for private(i)
#endif

		for (i = 0; i < pnX * pnY; i++) {
			Real x = i % pnX;
			Real y = i / pnX;

			Real fxx = (-1.0 / (2.0*ppX)) + (1.0 / ssX) * x;
			Real fyy = (1.0 / (2.0*ppY)) - (1.0 / ssY) - (1.0 / ssY) * y;

			Real fxxx = lambda * fxx;
			Real fyyy = lambda * fyy;

			Real sval = sqrt(1 - (fxxx * fxxx) - (fyyy * fyyy));
			sval *= k * propagation_dist;
			Complex<Real> kernel(0, sval);
			kernel.exp();

			int prop_mask = ((fxx * fxx + fyy * fyy) < (k * k)) ? 1 : 0;

			Complex<Real> u_frequency;
			if (prop_mask == 1) {
				u_frequency = kernel * input_u[i];

#ifdef _OPENMP
#pragma omp atomic
#endif
				complex_H[ch][i][_RE] += u_frequency[_RE];
#ifdef _OPENMP
#pragma omp atomic
#endif
				complex_H[ch][i][_IM] += u_frequency[_IM];
			}
		}
#ifdef _OPENMP
	}
#endif

#endif
}

bool ophGen::mergeColor(int idx, int width, int height, uchar *src, uchar *dst)
{
	if (idx < 0 || idx > 2) return false;

	int a = 2 - idx;
#ifdef _OPENMP
	int i;
#pragma omp for private(i)
	for (i = 0; i < width*height; i++) {
#else
	for (int i = 0; i < width*height; i++) {
#endif
		dst[i * 3 + a] = src[i];
	}

	return true;
}

bool ophGen::separateColor(int idx, int width, int height, uchar *src, uchar *dst)
{
	if (idx < 0 || idx > 2) return false;

	int a = 2 - idx;
#ifdef _OPENMP
	int i;
#pragma omp for private(i)
	for (i = 0; i < width*height; i++) {
#else
	for (int i = 0; i < width*height; i++) {
#endif
		dst[i] = src[i * 3 + a];
	}

	return true;
}

void ophGen::normalize(void)
{
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	for (uint ch = 0; ch < context_.waveNum; ch++)
		oph::normalize((Real*)holo_encoded[ch], holo_normalized[ch], pnX, pnY);
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
			source = holo_normalized[0];
			saveAsImg(path, bitsperpixel, source, p[_X], p[_Y]);
		}
		else if (nChannel == 3) {
			if (context_.bMergeImg) {
				uint nSize = (((p[_X] * bitsperpixel / 8) + 3) & ~3) * p[_Y];
				source = new uchar[nSize];
				bAlloc = true;
				for (int i = 0; i < nChannel; i++) {
					mergeColor(i, p[_X], p[_Y], holo_normalized[i], source);
				}
				saveAsImg(path, bitsperpixel, source, p[_X], p[_Y]);
				if (bAlloc) delete[] source;
			}
			else {
				for (int i = 0; i < nChannel; i++) {
					sprintf_s(path, "%s%s%s_%d%s", drive, dir, file, i, ext);
					source = holo_normalized[i];
					saveAsImg(path, bitsperpixel/nChannel, source, p[_X], p[_Y]);
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

	holo_encoded = new Real*[nChannel];
	holo_normalized = new uchar*[nChannel];
	for (uint ch = 0; ch < nChannel; ch++) {
		holo_encoded[ch] = new Real[pnXY];
		memset(holo_encoded[ch], 0, sizeof(Real) * pnXY);
		holo_normalized[ch] = new uchar[pnXY];
		memset(holo_normalized[ch], 0, sizeof(uchar) * pnXY);
	}
	return true;
}

void ophGen::resetBuffer()
{
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;

	for (uint ch = 0; ch < context_.waveNum; ch++) {
		if (complex_H[ch])
			memset(complex_H[ch], 0., sizeof(Complex<Real>) * pnXY);
		if (holo_encoded[ch])
			memset(holo_encoded[ch], 0., sizeof(Real) * encode_size[_X] * encode_size[_Y]);
		if (holo_normalized[ch])
			memset(holo_normalized[ch], 0, sizeof(uchar) * encode_size[_X] * encode_size[_Y]);
	}

}

void ophGen::encoding(unsigned int ENCODE_FLAG, Complex<Real>* holo)
{
	LOG("\n[Encoding] ");
	auto begin = CUR_TIME;
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint nChannel = context_.waveNum;
	const uint pnXY = pnX * pnY;
	void (ophGen::*encodeFunc) (Complex<Real>*, Real*, const int) = nullptr;

	switch (ENCODE_FLAG)
	{
	case ENCODE_PHASE: encodeFunc = &ophGen::Phase; LOG("Phase\n"); break;
	case ENCODE_AMPLITUDE: encodeFunc = &ophGen::Amplitude; LOG("Amplitude\n"); break;
	case ENCODE_REAL: encodeFunc = &ophGen::RealPart; LOG("Real\n"); break;
	case ENCODE_SIMPLENI: encodeFunc = &ophGen::SimpleNI; LOG("SimpleNI\n"); break;
	case ENCODE_BURCKHARDT: encodeFunc = &ophGen::Burckhardt; LOG("Burckhardt\n"); break;
	case ENCODE_TWOPHASE: encodeFunc = &ophGen::TwoPhase; LOG("Two-Phase\n"); break;
	//		ENCODE_SSB,
	//		ENCODE_OFFSSB,
	//		ENCODE_SYMMETRIZATION
	default: LOG("Wrong encode flag.\n");  return;
	}
	
	// initialzed zero
	for (int ch = 0; ch < nChannel; ch++) {
		memset(holo_encoded[ch], 0, sizeof(Real) * pnXY);
		memset(holo_normalized[ch], 0, sizeof(uchar) * pnXY);
		holo = complex_H[ch];
		(this->*encodeFunc)(holo, holo_encoded[ch], pnXY);
	}

	//encodeSymmetrization((holo), holo_encoded[ch], ivec2(0, 1));
	auto end = CUR_TIME;
	LOG("[Done] %lf(s)\n", ELAPSED_TIME(begin, end));
}

void ophGen::encoding(unsigned int ENCODE_FLAG, unsigned int passband, Complex<Real>* holo)
{
	holo == nullptr ? holo = *complex_H : holo;

	const uint pnX = encode_size[_X] = context_.pixel_number[_X];
	const uint pnY = encode_size[_Y] = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;
	const uint nChannel = context_.waveNum;

	for (uint ch = 0; ch < nChannel; ch++) {
		/*	initialize	*/
		int encode_size = pnXY;
		if (holo_encoded[ch] != nullptr) delete[] holo_encoded[ch];
		holo_encoded[ch] = new Real[encode_size];
		memset(holo_encoded[ch], 0, sizeof(Real) * encode_size);

		if (holo_normalized[ch] != nullptr) delete[] holo_normalized[ch];
		holo_normalized[ch] = new uchar[encode_size];
		memset(holo_normalized[ch], 0, sizeof(uchar) * encode_size);

		switch (ENCODE_FLAG)
		{
		case ENCODE_SSB:
			LOG("Single Side Band Encoding..");
			singleSideBand((holo), holo_encoded[ch], context_.pixel_number, passband);
			LOG("Done.");
			break;
		case ENCODE_OFFSSB:
			LOG("Off-axis Single Side Band Encoding..");
			freqShift(complex_H[ch], complex_H[ch], context_.pixel_number, 0, 100);
			singleSideBand((holo), holo_encoded[ch], context_.pixel_number, passband);
			LOG("Done.\n");
			break;
		default:
			LOG("error: WRONG ENCODE_FLAG\n");
			cin.get();
			return;
		}
	}
}

void ophGen::encoding()
{
	const uint pnX = encode_size[_X] = context_.pixel_number[_X];
	const uint pnY = encode_size[_Y] = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;
	const uint nChannel = context_.waveNum;
	
	if (ENCODE_METHOD == ENCODE_BURCKHARDT) encode_size[_X] *= 3;
	else if (ENCODE_METHOD == ENCODE_TWOPHASE)  encode_size[_X] *= 2;

	for (uint ch = 0; ch < nChannel; ch++) {
		/*	initialize	*/
		if (holo_encoded[ch] != nullptr) delete[] holo_encoded[ch];
		holo_encoded[ch] = new Real[encode_size[_X] * encode_size[_Y]];
		memset(holo_encoded[ch], 0, sizeof(Real) * encode_size[_X] * encode_size[_Y]);

		if (holo_normalized[ch] != nullptr) delete[] holo_normalized[ch];
		holo_normalized[ch] = new uchar[encode_size[_X] * encode_size[_Y]];
		memset(holo_normalized[ch], 0, sizeof(uchar) * encode_size[_X] * encode_size[_Y]);


		switch (ENCODE_METHOD)
		{
		case ENCODE_SIMPLENI:
			cout << "Simple Numerical Interference Encoding.." << endl;
			SimpleNI(complex_H[ch], holo_encoded[ch], pnXY);
			break;
		case ENCODE_REAL:
			cout << "Real Part Encoding.." << endl;
			realPart<Real>(complex_H[ch], holo_encoded[ch], pnXY);
			break;
		case ENCODE_BURCKHARDT:
			cout << "Burckhardt Encoding.." << endl;
			Burckhardt(complex_H[ch], holo_encoded[ch], pnXY);
			break;
		case ENCODE_TWOPHASE:
			cout << "Two Phase Encoding.." << endl;
			TwoPhase(complex_H[ch], holo_encoded[ch], pnXY);
			break;
		case ENCODE_PHASE:
		{
			auto begin = CUR_TIME;
			cout << "Phase Encoding.." << endl;
			Phase(complex_H[ch], holo_encoded[ch], pnXY);
			auto end = CUR_TIME;
			Real elapsed_time = ((chrono::duration<Real>)(end - begin)).count();
			LOG("\n%s : %lf(sec)\n\n",
				__FUNCTION__,
				elapsed_time);
		}
		break;
		case ENCODE_AMPLITUDE:
			cout << "Amplitude Encoding.." << endl;
			getAmplitude(complex_H[ch], holo_encoded[ch], pnXY);
			break;
		case ENCODE_SSB:
			cout << "Single Side Band Encoding.." << endl;
			singleSideBand(complex_H[ch], holo_encoded[ch], context_.pixel_number, SSB_PASSBAND);
			break;
		case ENCODE_OFFSSB:
			cout << "Off-axis Single Side Band Encoding.." << endl;
			freqShift(complex_H[ch], complex_H[ch], context_.pixel_number, 0, 100);
			singleSideBand(complex_H[ch], holo_encoded[ch], context_.pixel_number, SSB_PASSBAND);
			break;
		case ENCODE_SYMMETRIZATION:
			cout << "Symmetrization Encoding.." << endl;
			encodeSymmetrization(complex_H[ch], holo_encoded[ch], ivec2(0, 1));
			break;
		default:
			cout << "error: WRONG ENCODE_FLAG" << endl;
			cin.get();
			return;
		}
	}
}

void ophGen::singleSideBand(oph::Complex<Real>* holo, Real* encoded, const ivec2 holosize, int SSB_PASSBAND)
{
	int size = holosize[_X] * holosize[_Y];

	oph::Complex<Real>* AS = new oph::Complex<Real>[size];
	fft2(holosize, holo, OPH_FORWARD, OPH_ESTIMATE);
	fftwShift(holo, AS, holosize[_X], holosize[_Y], OPH_FORWARD, false);
	//fftExecute(temp);

	switch (SSB_PASSBAND)
	{
	case SSB_LEFT:
		for (int i = 0; i < holosize[_Y]; i++)
		{
			for (int j = holosize[_X] / 2; j < holosize[_X]; j++)
			{
				AS[i*holosize[_X] + j] = 0;
			}
		}
		break;
	case SSB_RIGHT:
		for (int i = 0; i < holosize[_Y]; i++)
		{
			for (int j = 0; j < holosize[_X] / 2; j++)
			{
				AS[i*holosize[_X] + j] = 0;
			}
		}
		break;
	case SSB_TOP:
		for (int i = size / 2; i < size; i++)
		{
			AS[i] = 0;
		}
		break;

	case SSB_BOTTOM:
		for (int i = 0; i < size / 2; i++)
		{
			AS[i] = 0;
		}
		break;
	}

	oph::Complex<Real>* filtered = new oph::Complex<Real>[size];
	fft2(holosize, AS, OPH_BACKWARD, OPH_ESTIMATE);
	fftwShift(AS, filtered, holosize[_X], holosize[_Y], OPH_BACKWARD, false);

	//fftExecute(filtered);


	Real* realFiltered = new Real[size];
	oph::realPart<Real>(filtered, realFiltered, size);

	oph::normalize(realFiltered, encoded, size);

	delete[] AS, filtered, realFiltered;
}

void ophGen::freqShift(oph::Complex<Real>* src, Complex<Real>* dst, const ivec2 holosize, int shift_x, int shift_y)
{
	int size = holosize[_X] * holosize[_Y];

	Complex<Real>* AS = new oph::Complex<Real>[size];
	fft2(holosize, src, OPH_FORWARD, OPH_ESTIMATE);
	fftwShift(src, AS, holosize[_X], holosize[_Y], OPH_FORWARD);
	//fftExecute(AS);

	Complex<Real>* shifted = new oph::Complex<Real>[size];
	circShift<Complex<Real>>(AS, shifted, shift_x, shift_y, holosize.v[_X], holosize.v[_Y]);

	fft2(holosize, shifted, OPH_BACKWARD, OPH_ESTIMATE);
	fftwShift(shifted, dst, holosize[_X], holosize[_Y], OPH_BACKWARD);
	//fftExecute(dst);
}

void ophGen::fresnelPropagation(OphConfig context, Complex<Real>* in, Complex<Real>* out, Real distance)
{
	const int pnX = context.pixel_number[_X];
	const int pnY = context.pixel_number[_Y];
	const uint pnXY = pnX * pnY;

	Complex<Real>* in2x = new Complex<Real>[pnXY * 4];
	Complex<Real> zero(0, 0);
	memsetArr<Complex<Real>>(in2x, zero, 0, pnXY * 4 - 1);

	uint idxIn = 0;

	for (int idxNy = pnY / 2; idxNy < pnY + (pnY / 2); idxNy++) {
		for (int idxNx = pnX / 2; idxNx < pnX + (pnX / 2); idxNx++) {
			in2x[idxNy * pnX * 2 + idxNx] = in[idxIn];
			idxIn++;
		}
	}

	Complex<Real>* temp1 = new Complex<Real>[pnXY * 4];

	fft2({ pnX * 2, pnY * 2 }, in2x, OPH_FORWARD, OPH_ESTIMATE);
	fftwShift(in2x, temp1, pnX, pnY, OPH_FORWARD);
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
	fftwShift(temp2, temp3, pnX * 2, pnY * 2, OPH_BACKWARD);
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
	const uint pnXY = pnX * pnY;
	const Real lambda = context_.wave_length[channel];

	Complex<Real>* in2x = new Complex<Real>[pnXY * 4];
	Complex<Real> zero(0, 0);
	memsetArr<Complex<Real>>(in2x, zero, 0, pnXY * 4 - 1);

	uint idxIn = 0;
	int idxnY = pnY / 2;

#ifdef _OPENMP
#pragma omp parallel
	{
#pragma omp parallel for private(idxnY) reduction(+:idxIn)
#endif
		for (idxnY = pnY / 2; idxnY < pnY + (pnY / 2); idxnY++) {
			for (int idxnX = pnX / 2; idxnX < pnX + (pnX / 2); idxnX++) {
				in2x[idxnY * pnX * 2 + idxnX] = in[idxIn++];
			}
		}
#ifdef _OPENMP
	}
#endif

	Complex<Real>* temp1 = new Complex<Real>[pnXY * 4];

	fft2({ pnX * 2, pnY * 2 }, in2x, OPH_FORWARD, OPH_ESTIMATE);
	fftwShift(in2x, temp1, pnX * 2, pnY * 2, OPH_FORWARD, false);

	Real* fx = new Real[pnXY * 4];
	Real* fy = new Real[pnXY * 4];

	uint i = 0;
	for (int idxFy = -pnY; idxFy < pnY; idxFy++) {
		for (int idxFx = -pnX; idxFx < pnX; idxFx++) {
			fx[i] = idxFx / (2 * pnX * ppX);
			fy[i] = idxFy / (2 * pnY * ppY);
			i++;
		}
	}

	Complex<Real>* prop = new Complex<Real>[pnXY * 4];
	memsetArr<Complex<Real>>(prop, zero, 0, pnXY * 4 - 1);

	Real sqrtPart;

	Complex<Real>* temp2 = new Complex<Real>[pnXY * 4];

	for (int i = 0; i < pnXY * 4; i++) {
		sqrtPart = sqrt(1 / (lambda * lambda) - fx[i] * fx[i] - fy[i] * fy[i]);
		prop[i][_IM] = 2 * M_PI * distance;
		prop[i][_IM] *= sqrtPart;
		temp2[i] = temp1[i] * exp(prop[i]);
	}

	Complex<Real>* temp3 = new Complex<Real>[pnXY * 4];
	fft2({ pnX * 2, pnY * 2 }, temp2, OPH_BACKWARD, OPH_ESTIMATE);
	fftwShift(temp2, temp3, pnX * 2, pnY * 2, OPH_BACKWARD, false);

	uint idxOut = 0;
	// 540 ~ 1620
	// 960 ~ 2880
	// 540 * 1920 * 2 + 960
	for (int idxnY = pnY / 2; idxnY < pnY + (pnY / 2); idxnY++) {
		for (int idxnX = pnX / 2; idxnX < pnX + (pnX / 2); idxnX++) {
			out[idxOut++] = temp3[idxnY * pnX * 2 + idxnX];
		}
	}
	//delete[] in;
	delete[] in2x;
	delete[] temp1;
	delete[] fx;
	delete[] fy;
	delete[] prop;
	delete[] temp2;
	delete[] temp3;

	auto end = CUR_TIME;
	LOG("\n%s : %lf(s)\n\n",
		__FUNCTION__,
		((chrono::duration<Real>)(end - begin)).count()
	);
}

bool ophGen::Shift(Real x, Real y)
{
	if (x == 0.0 && y == 0.0) return false;

	auto begin = CUR_TIME;

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
		waveRatio[i] = context_.wave_length[i] / context_.wave_length[nChannel - 1];

		Real ratioX = x * waveRatio[i];
		Real ratioY = y * waveRatio[i];
		int y;

#ifdef _OPENMP
#pragma omp parallel
		{
			num_threads = omp_get_num_threads();
#pragma omp for private(y)
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
#ifdef _OPENMP
		}
#endif
	}

	auto end = CUR_TIME;
	LOG("Complex Field Shift (%d threads): %lf(s)\n",
		num_threads,
		ELAPSED_TIME(begin, end));
	return true;
}

void ophGen::waveCarry(Real carryingAngleX, Real carryingAngleY, Real distance)
{
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];
	const uint pnXY = pnX * pnY;
	const uint nChannel = context_.waveNum;
	Real dfx = 1 / ppX / pnX;
	Real dfy = 1 / ppY / pnY;
	Real* fx = new Real[pnXY];
	Real* fy = new Real[pnXY];
	Real* fz = new Real[pnXY];
	uint i = 0;
	for (uint ch = 0; ch < nChannel; ch++) {
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
		fftwShift(h_crop, h_crop, pnX, pnY, OPH_BACKWARD, true);

		memset(holo_encoded[ch], 0.0, sizeof(Real) * pnXY);
		int i = 0;
#pragma omp parallel for private(i)	
		for (i = 0; i < pnXY; i++) {
			Complex<Real> shift_phase(1, 0);
			getShiftPhaseValue(shift_phase, i, sig_location);

			holo_encoded[ch][i] = (h_crop[i] * shift_phase).real();
		}
	}
	delete[] h_crop;
}

void ophGen::encodeSymmetrization(Complex<Real>* holo, Real* encoded, const ivec2 sig_loc)
{
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const int pnXY = pnX * pnY;

	int cropx1, cropx2, cropx, cropy1, cropy2, cropy;
	if (sig_loc[1] == 0) //Left or right half
	{
		cropy1 = 1;
		cropy2 = pnY;

	}
	else {

		cropy = floor(pnY / 2);
		cropy1 = cropy - floor(cropy / 2);
		cropy2 = cropy1 + cropy - 1;
	}

	if (sig_loc[0] == 0) // Upper or lower half
	{
		cropx1 = 1;
		cropx2 = pnX;

	}
	else {

		cropx = floor(pnX / 2);
		cropx1 = cropx - floor(cropx / 2);
		cropx2 = cropx1 + cropx - 1;
	}

	cropx1 -= 1;
	cropx2 -= 1;
	cropy1 -= 1;
	cropy2 -= 1;

	Complex<Real>* h_crop = new Complex<Real >[pnXY];
	memset(h_crop, 0.0, sizeof(Complex<Real>) * pnXY);
	int i;
#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
	for (i = 0; i < pnXY; i++) {
		int x = i % pnX;
		int y = i / pnX;
		if (x >= cropx1 && x <= cropx2 && y >= cropy1 && y <= cropy2)
			h_crop[i] = holo[i];
	}
	fftw_complex *in = nullptr, *out = nullptr;
	fftw_plan plan = fftw_plan_dft_2d(pnX, pnY, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftwShift(h_crop, h_crop, pnX, pnY, -1, true);
	fftw_destroy_plan(plan);
	fftw_cleanup();

#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
	for (i = 0; i < pnXY; i++) {
		Complex<Real> shift_phase(1, 0);
		getShiftPhaseValue(shift_phase, i, sig_loc);
		encoded[i] = (h_crop[i] * shift_phase)._Val[_RE];
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
		memset(holo_encoded[ch], 0.0, sizeof(Real) * pnXY);

		for (int i = 0; i < pnX * pnY; i++)
			holo_encoded[ch][i] = sample_fd[i].x;

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

void ophGen::getRandPhaseValue(Complex<Real>& rand_phase_val, bool rand_phase)
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

void ophGen::RealPart(Complex<Real> *holo, Real *encoded, const int size)
{
	int num_threads = 1;
	int i;
#ifdef _OPENMP
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
#pragma omp for private(i)
#endif
		for (i = 0; i < size; i++) {
			encoded[i] = real(holo[i]);
		}
#ifdef _OPENMP
	}
#endif
}

void ophGen::Phase(Complex<Real> *holo, Real *encoded, const int size)
{
	int num_threads = 1;
	int i;
#ifdef _OPENMP
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
#pragma omp for private(i)
#endif
		for (i = 0; i < size; i++) {
			encoded[i] = holo[i].angle() + M_PI;
		}
#ifdef _OPENMP
	}
#endif
}

void ophGen::Amplitude(Complex<Real> *holo, Real *encoded, const int size)
{
	int num_threads = 1;
	int i;
#ifdef _OPENMP
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
#pragma omp for private(i)
#endif
		for (i = 0; i < size; i++) {
			encoded[i] = holo[i].mag();
		}
#ifdef _OPENMP
	}
#endif
}

void ophGen::TwoPhase(Complex<Real>* holo, Real* encoded, const int size)
{
	int resize = size / 2;
	int num_threads = 1;
	int i;
	Complex<Real>* normCplx = new Complex<Real>[resize];

#ifdef _OPENMP
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
#pragma omp for private(i)
#endif
		for (i = 0; i < resize; i++) {
			normCplx[i] = holo[i * 2];
		}
#ifdef _OPENMP
	}
#endif

	oph::normalize<Real>(normCplx, normCplx, resize);

	Real* ampl = new Real[resize];
	Amplitude(normCplx, ampl, resize);

	Real* phase = new Real[resize];
	Phase(normCplx, phase, resize);

#ifdef _OPENMP
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
#pragma omp for private(i)
#endif
		for (i = 0; i < resize; i++) {
			Real delPhase = acos(ampl[i]);
			encoded[i * 2] = (phase[i] + M_PI) + delPhase;
			encoded[i * 2 + 1] = (phase[i] + M_PI) - delPhase;
		}
#ifdef _OPENMP
	}
#endif
	delete[] normCplx;
	delete[] ampl;
	delete[] phase;
}

void ophGen::Burckhardt(Complex<Real>* holo, Real* encoded, const int size)
{
	int resize = size / 3;
	int num_threads = 1;
	int i;
	Complex<Real>* norm = new Complex<Real>[resize];
#ifdef _OPENMP
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
#pragma omp for private(i)
#endif
		for (i = 0; i < resize; i++) {
			norm[i] = holo[i * 3];
		}
#ifdef _OPENMP
	}
#endif

	oph::normalize(norm, norm, resize);

	Real* phase = new Real[resize];
	Phase(norm, phase, resize);

	Real* ampl = new Real[resize];
	Amplitude(norm, ampl, resize);

	Real sqrt3 = sqrt(3);
	Real pi2 = 2 * M_PI;
	Real pi4 = 4 * M_PI;

#ifdef _OPENMP
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
#pragma omp for private(i)
#endif
		for( i = 0; i < resize; i++) {
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
#ifdef _OPENMP
	}
#endif
	delete[] ampl;
	delete[] phase;
	delete[] norm;
}

void ophGen::SimpleNI(Complex<Real>* holo, Real* encoded, const int size)
{
	int num_threads = 1;
	Real* tmp1 = new Real[size];
	int i;
#ifdef _OPENMP
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
#pragma omp for private(i)
#endif
		for (i = 0; i < size; i++) {
			tmp1[i] = holo[i].mag();
		}
#ifdef _OPENMP
	}
#endif

	Real max = maxOfArr(tmp1, size);
	delete[] tmp1;

#ifdef _OPENMP
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
#pragma omp for private(i)
#endif
		for (i = 0; i < size; i++) {
			Real tmp = (holo[i] + max).mag();
			encoded[i] = tmp * tmp;
		}
#ifdef _OPENMP
	}
#endif
}

void ophGen::transVW(int nSize, Real *dst, Real *src)
{
	Real fieldLens = m_nFieldLength;
	for (int i = 0; i < nSize; i++) {
		*(dst + i) = -fieldLens * src[i] / (src[i] - fieldLens);
	}
}

void ophGen::ophFree(void)
{
	Openholo::ophFree();
	if (holo_encoded) {
		delete[] holo_encoded;
		holo_encoded = nullptr;
	}
	if (holo_normalized) {
		delete[] holo_normalized;
		holo_normalized = nullptr;
	}
}
