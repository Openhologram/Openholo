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

#include "tinyxml2.h"
#include "PLYparser.h"

ophGen::ophGen(void)
	: Openholo()
	, holo_encoded(nullptr)
	, holo_normalized(nullptr)
	, bCarried(false)
	, nOldChannel(0)
	, elapsedTime(0.0)
{
}

ophGen::~ophGen(void)
{
}

void ophGen::initialize(void)
{
	LOG("%s...\n", __FUNCTION__);
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
	LOG("Reading....%s...\n", pc_file);

	auto start = CUR_TIME;

	PLYparser plyIO;
	if (!plyIO.loadPLY(pc_file, pc_data_->n_points, pc_data_->n_colors, &pc_data_->vertex, &pc_data_->color, &pc_data_->phase, pc_data_->isPhaseParse))
		return -1;

	auto end = CUR_TIME;
	auto during = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("%.5lfsec...done\n", during);
	return pc_data_->n_points;
}

bool ophGen::readConfig(const char* fname)
{
	LOG("Reading....%s...", fname);
	using namespace tinyxml2;

	auto start = CUR_TIME;
	tinyxml2::XMLDocument xml_doc;
	XMLNode *xml_node = nullptr;
	
	if (checkExtension(fname, ".xml") == 0)
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

	context_.ss[_X] = context_.pixel_number[_X] * context_.pixel_pitch[_X];
	context_.ss[_Y] = context_.pixel_number[_Y] * context_.pixel_pitch[_Y];

	Openholo::setPixelNumberOHC(context_.pixel_number);
	Openholo::setPixelPitchOHC(context_.pixel_pitch);

	OHC_encoder->clearWavelength();
	for (int i = 0; i < nWave; i++)
		Openholo::setWavelengthOHC(context_.wave_length[i], LenUnit::m);

	auto end = CUR_TIME;
	auto during = ((chrono::duration<Real>)(end - start)).count();
	LOG("%lf (s)...done\n", during);

	return true;
}


/**
* @brief Angular spectrum propagation method
* @details The propagation results of all depth levels are accumulated in the variable 'U_complex_'.
* @param input_u : each depth plane data.
* @param propagation_dist : the distance from the object to the hologram plane.
* @see Calc_Holo_by_Depth, Calc_Holo_CPU, fftwShift
*/
void ophGen::propagationAngularSpectrum(int ch, Complex<Real>* input_u, Real propagation_dist, Real k, Real lambda)
{
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];
	const Real ssX = context_.ss[_X] = pnX * ppX;
	const Real ssY = context_.ss[_Y] = pnY * ppY;

	for (int i = 0; i < pnX * pnY; i++) {
		Real x = i % pnX;
		Real y = i / pnX;

		Real fxx = (-1.0 / (2.0*ppX)) + (1.0 / ssX) * x;
		Real fyy = (1.0 / (2.0*ppY)) - (1.0 / ssY) - (1.0 / ssY) * y;

		Real sval = sqrt(1 - (lambda*fxx)*(lambda*fxx) - (lambda*fyy)*(lambda*fyy));
		sval *= k * propagation_dist;
		Complex<Real> kernel(0, sval);
		kernel.exp();

		int prop_mask = ((fxx * fxx + fyy * fyy) < (k * k)) ? 1 : 0;

		Complex<Real> u_frequency;
		if (prop_mask == 1)
			u_frequency = kernel * input_u[i];

#pragma omp atomic
		complex_H[ch][i][_RE] += u_frequency[_RE];
#pragma omp atomic
		complex_H[ch][i][_IM] += u_frequency[_IM];
	}
}

void ophGen::normalize(void)
{
	for (uint ch = 0; ch < context_.waveNum; ch++)
		oph::normalize((Real*)holo_encoded[ch], holo_normalized[ch], context_.pixel_number[_X], context_.pixel_number[_Y]);
}

void ophGen::normalizeEncoded(void)
{
	for (uint ch = 0; ch < context_.waveNum; ch++)
		oph::normalize((Real*)holo_encoded[ch], holo_normalized[ch], context_.pixel_number[_X], context_.pixel_number[_Y]);
}

int ophGen::save(const char * fname, uint8_t bitsperpixel, uchar* src, uint px, uint py)
{
	if (fname == nullptr) return -1;

	uchar* source = src;
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
	
	for (uint ch = 0; ch < nChannel; ch++) {
		if (src == nullptr)
			source = holo_normalized[ch];
		if (nChannel > 1) {
			sprintf_s(path, "%s%s%s_%d%s", drive, dir, file, ch + 1, ext);
		}

		if (checkExtension(path, ".bmp")) 	// when the extension is bmp
			Openholo::saveAsImg(path, bitsperpixel, source, p[_X], p[_Y]);
		else if (
			checkExtension(path, ".jpg") ||
			checkExtension(path, ".gif") ||
			checkExtension(path, ".png")) {
			Openholo::saveAsImg(path, bitsperpixel, source, p[_X], p[_Y]);
		}
		else {									// when extension is not .ohf, .bmp - force bmp
			char buf[256];
			memset(buf, 0x00, sizeof(char) * 256);
			sprintf_s(buf, "%s.bmp", path);

			Openholo::saveAsImg(buf, bitsperpixel, source, p[_X], p[_Y]);
		}
	}
}

int ophGen::save(const char * fname, uint8_t bitsperpixel, uint px, uint py, uint fnum, uchar* args ...)
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

	return 0;
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

int ophGen::loadAsOhc(const char * fname)
{
	if (Openholo::loadAsOhc(fname) == -1) return -1;

	const uint nChannel = context_.waveNum;
	const uint pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];

	holo_encoded = new Real*[nChannel];
	holo_normalized = new uchar*[nChannel];
	for (uint ch = 0; ch < nChannel; ch++) {
		holo_encoded[ch] = new Real[pnXY];
		memset(holo_encoded, 0, sizeof(Real) * pnXY);
		holo_normalized[ch] = new uchar[pnXY];
		memset(holo_normalized, 0, sizeof(uchar) * pnXY);
	}
	return 0;
}

void ophGen::resetBuffer()
{
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;

	for (uint ch = 0; ch < context_.waveNum; ch++) {
		if(complex_H[ch])
			memset(complex_H[ch], 0., sizeof(Complex<Real>) * pnXY);
		if (holo_encoded[ch])
			memset(holo_encoded[ch], 0., sizeof(Real) * encode_size[_X] * encode_size[_Y]);
		if (holo_normalized[ch])
			memset(holo_normalized[ch], 0, sizeof(uchar) * encode_size[_X] * encode_size[_Y]);
	}

}

#define for_i(itr, oper) for(int i=0; i<itr; i++){ oper }

void ophGen::loadComplex(char* real_file, char* imag_file, const uint pnX, const uint pnY)
{
	context_.pixel_number[_X] = pnX;
	context_.pixel_number[_Y] = pnY;

	ifstream freal, fimag;
	freal.open(real_file);
	fimag.open(imag_file);
	if (!freal) {
		cout << "open failed - real" << endl;
		cin.get();
		return;
	}
	if (!fimag) {
		cout << "open failed - imag" << endl;
		cin.get();
		return;
	}

	for (uint i = 0; i < context_.waveNum; i++) {
		if (complex_H[i] != nullptr) delete[] complex_H[i];
		complex_H[i] = new oph::Complex<Real>[pnX * pnY];
		memset(complex_H[i], 0.0, sizeof(Complex<Real>) * pnX * pnY);

		Real realVal, imagVal;

		for (int j = 0; j < pnX * pnY; j++) {
			freal >> realVal;
			fimag >> imagVal;

			Complex<Real> compVal;
			compVal(realVal, imagVal);
			complex_H[i][j] = compVal;
			if (realVal == EOF || imagVal == EOF)
				break;
		}
	}
}

void ophGen::encoding(unsigned int ENCODE_FLAG, Complex<Real>* holo, bool bShift)
{
	bool bChangeSize = false;
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint nChannel = context_.waveNum;
	const uint pnXY = pnX * pnY;
	Complex<Real>* dst = nullptr;
	if (bShift) {
		dst = new Complex<Real>[pnXY];
	}

	for (int ch = 0; ch < nChannel; ch++) {
		if (bShift) {
			memset(dst, 0.0, sizeof(Complex<Real>) * pnXY);
			fft2(context_.pixel_number, complex_H[ch], OPH_BACKWARD);
			fftwShift(complex_H[ch], dst, pnX, pnY, OPH_BACKWARD);			
			holo = complex_H[ch];
		}
		else
			holo = complex_H[ch];

		if (ENCODE_FLAG == ENCODE_BURCKHARDT) {
			encode_size[_X] = pnX * 3;
			encode_size[_Y] = pnY;
			bChangeSize = true;
		}
		else if (ENCODE_FLAG == ENCODE_TWOPHASE) {
			encode_size[_X] = pnX * 2;
			encode_size[_Y] = pnY;
			bChangeSize = true;
		}
		else {
			encode_size[_X] = pnX;
			encode_size[_Y] = pnY;
		}

		/*	initialize	*/
		if (bChangeSize) {
			if (holo_encoded != nullptr) delete[] holo_encoded;
			if (holo_normalized != nullptr) delete[] holo_normalized;

			holo_encoded[ch] = new Real[encode_size[_X] * encode_size[_Y]];
			holo_normalized[ch] = new uchar[encode_size[_X] * encode_size[_Y]];
		}

		memset(holo_encoded[ch], 0, sizeof(Real) * encode_size[_X] * encode_size[_Y]);
		memset(holo_normalized[ch], 0, sizeof(uchar) * encode_size[_X] * encode_size[_Y]);

		switch (ENCODE_FLAG)
		{
		case ENCODE_SIMPLENI:
			LOG("Simple Numerical Interference Encoding..");
			numericalInterference((holo), holo_encoded[ch], pnXY);
			LOG("Done.\n.");
			break;
		case ENCODE_REAL:
			LOG("Real Part Encoding..");
			realPart<Real>((holo), holo_encoded[ch], pnXY);
			LOG("Done.\n.");
			break;
		case ENCODE_BURCKHARDT:
			LOG("Burckhardt Encoding..");
			burckhardt((holo), holo_encoded[ch], pnXY);
			LOG("Done.\n.");
			break;
		case ENCODE_TWOPHASE:
			LOG("Two Phase Encoding..");
			twoPhaseEncoding((holo), holo_encoded[ch], pnXY);
			LOG("Done.\n.");
			break;
		case ENCODE_PHASE:
			LOG("Phase Encoding..");
			getPhase((holo), holo_encoded[ch], pnXY);
			LOG("Done.\n.");
			break;
		case ENCODE_AMPLITUDE:
			LOG("Amplitude Encoding..");
			getAmplitude((holo), holo_encoded[ch], pnXY);
			LOG("Done.\n.");
			break;
		case ENCODE_SSB:
		case ENCODE_OFFSSB:
			LOG("error: PUT PASSBAND\n");
			cin.get();
			return;
		case ENCODE_SYMMETRIZATION:
			LOG("Symmetrization Encoding..");
			encodeSymmetrization((holo), holo_encoded[ch], ivec2(0, 1));
			LOG("Done.\n.");
			break;
		default:
			LOG("error: WRONG ENCODE_FLAG\n");
			cin.get();
			return;
		}
	}
	if(dst)
		delete[] dst;
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
			numericalInterference(complex_H[ch], holo_encoded[ch], pnXY);
			break;
		case ENCODE_REAL:
			cout << "Real Part Encoding.." << endl;
			realPart<Real>(complex_H[ch], holo_encoded[ch], pnXY);
			break;
		case ENCODE_BURCKHARDT:
			cout << "Burckhardt Encoding.." << endl;
			burckhardt(complex_H[ch], holo_encoded[ch], pnXY);
			break;
		case ENCODE_TWOPHASE:
			cout << "Two Phase Encoding.." << endl;
			twoPhaseEncoding(complex_H[ch], holo_encoded[ch], pnXY);
			break;
		case ENCODE_PHASE:
			cout << "Phase Encoding.." << endl;
			getPhase(complex_H[ch], holo_encoded[ch], pnXY);
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

void ophGen::numericalInterference(oph::Complex<Real>* holo, Real* encoded, const int size)
{
	Real* temp1 = new Real[size];
	absCplxArr<Real>(holo, temp1, size);

	Real* ref = new Real;
	*ref = maxOfArr(temp1, size);

	oph::Complex<Real>* temp2 = new oph::Complex<Real>[size];
	for_i(size,
		temp2[i] = holo[i] + *ref;
	);

	Real* temp3 = new Real[size];
	absCplxArr<Real>(temp2, temp3, size);

	for_i(size,
		encoded[i] = temp3[i] * temp3[i];
	);

	delete[] temp1;
	delete[] temp2;
	delete[] temp3;
	delete ref;
}

void ophGen::twoPhaseEncoding(oph::Complex<Real>* holo, Real* encoded, const int size)
{
	Complex<Real>* normCplx = new Complex<Real>[size];
	oph::normalize<Real>(holo, normCplx, size);

	Real* amp = new Real[size];
	oph::getAmplitude(normCplx, encoded, size);

	Real* pha = new Real[size];
	oph::getPhase(normCplx, pha, size);

	for_i(size, *(pha + i) += M_PI;);

	Real* delPhase = new Real[size];
	for_i(size, *(delPhase + i) = acos(*(amp + i)););

	for_i(size,
		*(encoded + i * 2) = *(pha + i) + *(delPhase + i);
	*(encoded + i * 2 + 1) = *(pha + i) - *(delPhase + i);
	);

	delete[] normCplx; 
	delete[] amp;
	delete[] pha;
	delete[] delPhase;
}

void ophGen::burckhardt(oph::Complex<Real>* holo, Real* encoded, const int size)
{
	Complex<Real>* norm = new Complex<Real>[size];
	oph::normalize(holo, norm, size);

	Real* phase = new Real[size];
	oph::getPhase(norm, phase, size);

	Real* ampl = new Real[size];
	oph::getAmplitude(norm, ampl, size);

	Real* A1 = new Real[size];
	memsetArr<Real>(A1, 0, 0, size - 1);
	Real* A2 = new Real[size];
	memsetArr<Real>(A2, 0, 0, size - 1);
	Real* A3 = new Real[size];
	memsetArr<Real>(A3, 0, 0, size - 1);

	for_i(size,
		if (*(phase + i) >= 0 && *(phase + i) < (2 * M_PI / 3))
		{
			*(A1 + i) = *(ampl + i)*(cos(*(phase + i)) + sin(*(phase + i)) / sqrt(3));
			*(A2 + i) = 2 * sin(*(phase + i)) / sqrt(3);
		}
		else if (*(phase + i) >= (2 * M_PI / 3) && *(phase + i) < (4 * M_PI / 3))
		{
			*(A2 + i) = *(ampl + i)*(cos(*(phase + i) - (2 * M_PI / 3)) + sin(*(phase + i) - (2 * M_PI / 3)) / sqrt(3));
			*(A3 + i) = 2 * sin(*(phase + i) - (2 * M_PI / 3)) / sqrt(3);
		}
		else if (*(phase + i) >= (4 * M_PI / 3) && *(phase + i) < (2 * M_PI))
		{
			*(A3 + i) = *(ampl + i)*(cos(*(phase + i) - (4 * M_PI / 3)) + sin(*(phase + i) - (4 * M_PI / 3)) / sqrt(3));
			*(A1 + i) = 2 * sin(*(phase + i) - (4 * M_PI / 3)) / sqrt(3);
		}
	);

	for_i(size,
		*(encoded + (3 * i)) = *(A1 + i);
	*(encoded + (3 * i + 1)) = *(A2 + i);
	*(encoded + (3 * i + 2)) = *(A3 + i);
	);

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

	delete[] AS, filtered , realFiltered;
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
#ifdef CHECK_PROC_TIME
	auto begin = CUR_TIME;
#endif
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
	fftwShift(in2x, temp1, pnX*2, pnY*2, OPH_FORWARD, false);

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
#ifdef CHECK_PROC_TIME
	auto end = CUR_TIME;
	LOG("\n%s : %lf(s)\n\n",
		__FUNCTION__,
		((chrono::duration<Real>)(end - begin)).count()
		);
#endif
}

void ophGen::waveCarry(Real carryingAngleX, Real carryingAngleY, Real distance)
{
	if (bCarried == FALSE) bCarried = TRUE;
	else return;

	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;
	const uint nChannel = context_.waveNum;
	Real dfx = 1 / context_.pixel_pitch[_X] / pnX;
	Real dfy = 1 / context_.pixel_pitch[_Y] / pnY;
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

		Complex<Real>* carrier = new Complex<Real>[pnXY];

		for (int i = 0; i < pnXY; i++) {
			carrier[i][_RE] = 0;
			carrier[i][_IM] = distance * tan(carryingAngleX)*fx[i] + distance * tan(carryingAngleY)*fy[i];
			complex_H[ch][i] = complex_H[ch][i] * exp(carrier[i]);
		}

		delete[] carrier;
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

void ophGen::getRandPhaseValue(oph::Complex<Real>& rand_phase_val, bool rand_phase)
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

void ophGen::ophFree(void)
{
	if (holo_encoded) delete[] holo_encoded;
	if (holo_normalized) delete[] holo_normalized;
}