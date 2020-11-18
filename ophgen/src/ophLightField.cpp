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

#include "ophLightField.h"
#include "include.h"
#include "sys.h"
#include "tinyxml2.h"

#define for_i(itr, oper) for(int i=0; i<itr; i++){ oper }

ophLF::ophLF(void)
	: num_image(ivec2(0, 0))
	, resolution_image(ivec2(0, 0))
	, distanceRS2Holo(0.0)
	, is_CPU(true)
	, is_ViewingWindow(false)
	, LF(nullptr)
	, complex_field(nullptr)
	, bSinglePrecision(false)
{
	LOG("*** LIGHT FIELD : BUILD DATE: %s %s ***\n\n", __DATE__, __TIME__);
}

void ophLF::setMode(bool isCPU)
{
	is_CPU = isCPU;
}

void ophLF::setViewingWindow(bool is_ViewingWindow)
{
	this->is_ViewingWindow = is_ViewingWindow;
}

bool ophLF::readConfig(const char* fname)
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

	// about viewing window
	auto next = xml_node->FirstChildElement("FieldLength");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&fieldLens))

		// about image
		next = xml_node->FirstChildElement("Image_NumOfX");
	if (!next || XML_SUCCESS != next->QueryIntText(&num_image[_X]))
		return false;
	next = xml_node->FirstChildElement("Image_NumOfY");
	if (!next || XML_SUCCESS != next->QueryIntText(&num_image[_Y]))
		return false;
	next = xml_node->FirstChildElement("Image_Width");
	if (!next || XML_SUCCESS != next->QueryIntText(&resolution_image[_X]))
		return false;
	next = xml_node->FirstChildElement("Image_Height");
	if (!next || XML_SUCCESS != next->QueryIntText(&resolution_image[_Y]))
		return false;
	next = xml_node->FirstChildElement("Image_PitchX");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&image_pitch[_X]))
		return false;
	next = xml_node->FirstChildElement("Image_PitchY");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&image_pitch[_Y]))
		return false;
	next = xml_node->FirstChildElement("Distance");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&distanceRS2Holo))
		return false;
	next = xml_node->FirstChildElement("Random_Phase");
	if (!next || XML_SUCCESS != next->QueryBoolText(&randPhase))
		return false;

	auto end = CUR_TIME;
	auto during = ((chrono::duration<Real>)(end - start)).count();
	LOG("%lf (s)..done\n", during);

	initialize();
	return true;
}

int ophLF::loadLF(const char* directory, const char* exten)
{
	LF_directory = directory;
	ext = exten;

	initializeLF();

	_finddata_t data;

	string sdir = std::string(LF_directory).append("\\").append("*.").append(ext);
	intptr_t ff = _findfirst(sdir.c_str(), &data);
	if (ff != -1)
	{
		int num = 0;
		uchar* rgbOut;
		ivec2 sizeOut;
		int bytesperpixel;

		while (1)
		{
			string imgfullname = std::string(LF_directory).append("\\").append(data.name);

			getImgSize(sizeOut[_X], sizeOut[_Y], bytesperpixel, imgfullname.c_str());

			rgbOut = loadAsImg(imgfullname.c_str());

			if (rgbOut == 0) {
				cout << "LF load was failed." << endl;
				return -1;
			}

			convertToFormatGray8(rgbOut, *(LF + num), sizeOut[_X], sizeOut[_Y], bytesperpixel);
			delete[] rgbOut; // solved memory leak.
			num++;

			int out = _findnext(ff, &data);
			if (out == -1)
				break;
		}
		_findclose(ff);
		cout << "LF load was successed." << endl;

		if (num_image[_X] * num_image[_Y] != num) {
			cout << "num_image is not matched." << endl;
		}
		return 1;
	}
	else
	{
		cout << "LF load was failed." << endl;
		return -1;
	}
}

int ophLF::loadLF()
{
	initializeLF();

	_finddata_t data;

	string sdir = std::string("./").append(LF_directory).append("/").append("*.").append(ext);
	intptr_t ff = _findfirst(sdir.c_str(), &data);
	if (ff != -1)
	{
		int num = 0;
		uchar* rgbOut;
		ivec2 sizeOut;
		int bytesperpixel;

		while (1)
		{
			string imgfullname = std::string(LF_directory).append("/").append(data.name);

			getImgSize(sizeOut[_X], sizeOut[_Y], bytesperpixel, imgfullname.c_str());
			rgbOut = loadAsImg(imgfullname.c_str());

			if (rgbOut == 0) {
				cout << "LF load was failed." << endl;
				cin.get();
				return -1;
			}

			convertToFormatGray8(rgbOut, *(LF + num), sizeOut[_X], sizeOut[_Y], bytesperpixel);

			num++;

			int out = _findnext(ff, &data);
			if (out == -1)
				break;
		}
		_findclose(ff);
		cout << "LF load was successed." << endl;

		if (num_image[_X] * num_image[_Y] != num) {
			cout << "num_image is not matched." << endl;
			cin.get();
		}
		return 1;
	}
	else
	{
		cout << "LF load was failed." << endl;
		cin.get();
		return -1;
	}
}

void ophLF::generateHologram()
{
	resetBuffer();

	LOG("1) Algorithm Method : Light Field\n");
	LOG("2) Generate Hologram with %s\n", is_CPU ?
#ifdef _OPENMP
		"Multi Core CPU" :
#else
		"Single Core CPU" :
#endif
		"GPU");
	LOG("3) Transform Viewing Window : %s\n", is_ViewingWindow ? "ON" : "OFF");

	auto begin = CUR_TIME;
	if (is_CPU)
	{
		convertLF2ComplexField();
		for (uint ch = 0; ch < context_.waveNum; ch++)
			fresnelPropagation(complex_field, complex_H[ch], distanceRS2Holo, ch);
	}
	else
	{
		prepareInputdataGPU();
		convertLF2ComplexField_GPU();
		fresnelPropagation_GPU();
	}

	auto end = CUR_TIME;
	m_elapsedTime = ((std::chrono::duration<Real>)(end - begin)).count();
	LOG("Total Elapsed Time: %lf (sec)\n", m_elapsedTime);
}

//int ophLF::saveAsOhc(const char * fname)
//{
//	setPixelNumberOHC(getEncodeSize());
//
//	Openholo::saveAsOhc(fname);
//
//	return 0;
//}


void ophLF::initializeLF()
{
	if (LF) {
		for (int i = 0; i < nImages; i++) {
			if (LF[i]) {
				delete[] LF[i];
				LF[i] = nullptr;
			}
		}
		delete[] LF;
		LF = nullptr;
	}

	LF = new uchar*[num_image[_X] * num_image[_Y]];
	for (int i = 0; i < num_image[_X] * num_image[_Y]; i++) {
		LF[i] = new uchar[resolution_image[_X] * resolution_image[_Y]];
		memset(LF[i], 0, resolution_image[_X] * resolution_image[_Y]);
	}
	nImages = num_image[_X] * num_image[_Y];
	cout << "The Number of the Images : " << num_image[_X] * num_image[_Y] << endl;
}


void ophLF::convertLF2ComplexField()
{
	auto begin = CUR_TIME;

	const uint nX = num_image[_X];
	const uint nY = num_image[_Y];
	const uint nXY = nX * nY;
	const uint rX = resolution_image[_X];
	const uint rY = resolution_image[_Y];
	const uint rXY = rX * rY;

	switch (method) {
	case LF_HOGEL:
		hogelLFCGH();
	case LF_NONHOGEL:
		nonHogelLFCGH();
	default:
		LOG("error: WRONG LF_METHOD\n");
		cin.get();
	}

#ifdef CHECK_PROC_TIME
	auto end = CUR_TIME;
	LOG("\n%s : %lf(s)\n\n", __FUNCTION__, ((std::chrono::duration<Real>)(end - begin)).count());
#endif
}

void ophLF::hogelLFCGH() {

	const uint nx = num_image[_X];
	const uint ny = num_image[_Y];
	const uint nXY = nx * ny;
	const uint rx = resolution_image[_X];
	const uint ry = resolution_image[_Y];
	const uint rXY = rx * ry;

	if (complex_field) {
		delete[] complex_field;
		complex_field = nullptr;
	}
	complex_field = new Complex<Real>[nXY * rXY];
	memset(complex_field, 0.0, sizeof(Complex<Real>) * nXY * rXY);

	Complex<Real>* complexLF = new Complex<Real>[nXY];
	Complex<Real>* FFTLF = new Complex<Real>[nXY];

	Real randVal;
	Complex<Real> phase(0.0, 0.0);

	int idxrX, idxrY, idxnX, idxnY;
	int idxImg = 0;

	for (idxrX = 0; idxrX < rx; idxrX++) { // 192
		for (idxrY = 0; idxrY < ry; idxrY++) { // 108
			memset(complexLF, 0.0, sizeof(Complex<Real>) * nXY);
			memset(FFTLF, 0.0, sizeof(Complex<Real>) * nXY);

			for (idxnY = 0; idxnY < ny; idxnY++) { // 10
				for (idxnX = 0; idxnX < nx; idxnX++) { // 10
													   // LF[img idx][pixel idx]
					complexLF[idxnX + nx * idxnY] = (Real)(LF[idxnX + nx * idxnY][idxrX + rx * idxrY]);
				}
			}

			fft2(num_image, complexLF, OPH_FORWARD, OPH_ESTIMATE);
			fftwShift(complexLF, FFTLF, nx, ny, OPH_FORWARD);

			for (int idxnX = 0; idxnX < nx; idxnX++) { // 10
				for (int idxnY = 0; idxnY < ny; idxnY++) { // 10
					randVal = rand((Real)0, (Real)1, idxrX * idxrY);
					phase(0, 2 * M_PI * randVal); // random phase

												  //*(complex_field + nXY * rx*idxrY + nx * rx*idxnY + nx * idxrX + idxnX) = *(FFTLF + (idxnX + nx * idxnY))*exp(phase);
												  // 100 * 192 * 107 + 10 * 192 * 107 + 10 * 191 + 9
					complex_field[nXY * rx*idxrY + nx * rx*idxnY + nx * idxrX + idxnX] =
						FFTLF[idxnX + nx * idxnY] * exp(phase);
					//
					// (20/5) (x:5/y:8) => 165
				}
			}
			idxImg++;
		}
	}
	delete[] complexLF, FFTLF;
	fftFree();
}

void ophLF::nonHogelLFCGH() {

	const uint nx = num_image[_X];
	const uint ny = num_image[_Y];
	const uint nXY = nx * ny;
	const uint rx = resolution_image[_X];
	const uint ry = resolution_image[_Y];
	const uint rXY = rx * ry;

	Complex<Real>** complexLF = new Complex<Real>*[rXY];
	Complex<Real>* FFTLF = new Complex<Real>[nXY];

	uint rx2 = 2 * rx + (nx / 2 + 1) * 2;
	uint ry2 = 2 * ry + (ny / 2 + 1) * 2;

	Complex<Real> phase(0.0, 0.0);
	Complex<Real>* phaseTerm = new Complex<Real>[rx2*ry2];

	int idxrX, idxrY, idxnX, idxnY, idxrx2, idxry2;

	// phase term
	for (idxrX = 0; idxrX < rx2; idxrX++) {
		for (idxrY = 0; idxrY < ry2; idxrY++) {
			if (randPhase) {
				if (idxrX >= (int)(-rx / 2 + rx2 / 2) && idxrX <= (int)(rx / 2 + rx2 / 2) && idxrY >= (int)(-ry / 2 + ry2 / 2) && idxrY <= (int)(ry / 2 + ry2 / 2)) {
					phase[_IM] = 2 * M_PI*rand((Real)0, (Real)1);
					phaseTerm[idxrX + rx*idxrY] = rand((Real)0, (Real)1)*exp(phase);
				}
				else {
					phaseTerm[idxrX + rx*idxrY] = Complex<Real>(0.0, 0.0);
				}
			}
			else {
				phaseTerm[idxrX + rx*idxrY] = 1;
			}
		}
	}
	if (randPhase) {
		fftwShift(phaseTerm, phaseTerm, rx2, ry2, OPH_BACKWARD);
		fft2((rx2, ry2), phaseTerm, OPH_BACKWARD, OPH_ESTIMATE);
		fftwShift(phaseTerm, phaseTerm, rx2, ry2, OPH_FORWARD);
	}

	Complex<Real>* holo2x = new Complex<Real>[rx2*ry2];
	memset(holo2x, 0.0, sizeof(Complex<Real>) * rx2*ry2);

	// complex field
	for (idxrX = 0; idxrX < rx; idxrX++) { // 192
		for (idxrY = 0; idxrY < ry; idxrY++) { // 108

			complexLF[idxrX + rx*idxrY] = new Complex<Real>[nXY];

			for (idxnY = 0; idxnY < ny; idxnY++) { // 10
				for (idxnX = 0; idxnX < nx; idxnX++) { // 10
					complexLF[idxrX + rx*idxrY][idxnX + nx*idxnY] = LF[idxnX + nx*idxnY][idxrX + rx*idxrY];
				}
			}

			fftwShift(complexLF[idxrX + rx*idxrY], FFTLF, nx, ny, OPH_BACKWARD);
			fft2(num_image, FFTLF, OPH_FORWARD, OPH_ESTIMATE);
			fftwShift(FFTLF, FFTLF, nx, ny, OPH_FORWARD);

			for (idxnY = 0; idxnY < ny; idxnY++) { // 10
				for (idxnX = 0; idxnX < nx; idxnX++) { // 10

					holo2x[(idxrX * 2 + idxnX) + rx2*(idxrY * 2 + idxnY)] += phaseTerm[(idxrX * 2 + nx - idxnX - 1) + rx2*(idxrY * 2 + ny - idxnY - 1)] * FFTLF[idxnX + nx*idxnY];
					holo2x[(idxrX * 2 + idxnX + 1) + rx2*(idxrY * 2 + idxnY)] += phaseTerm[(idxrX * 2 + nx - idxnX - 1 - 1) + rx2*(idxrY * 2 + ny - idxnY - 1)] * FFTLF[idxnX + nx*idxnY];
					holo2x[(idxrX * 2 + idxnX) + rx2*(idxrY * 2 + idxnY + 1)] += phaseTerm[(idxrX * 2 + nx - idxnX - 1) + rx2*(idxrY * 2 + ny - idxnY - 1 - 1)] * FFTLF[idxnX + nx*idxnY];
					holo2x[(idxrX * 2 + idxnX + 1) + rx2*(idxrY * 2 + idxnY + 1)] += phaseTerm[(idxrX * 2 + nx - idxnX - 1 - 1) + rx2*(idxrY * 2 + ny - idxnY - 1 - 1)] * FFTLF[idxnX + nx*idxnY];
				}
			}
		}
	}

	// downsizing
	if (complex_field) {
		delete[] complex_field;
		complex_field = nullptr;
	}
	complex_field = new Complex<Real>[rx2 * ry2 / 4];
	memset(complex_field, 0.0, sizeof(Complex<Real>) * rx2*ry2 / 4);

	for (idxrx2 = 0; idxrx2 < rx2 / 2; idxrx2++) {
		for (idxry2 = 0; idxry2 < ry2 / 2; idxry2++) {
			complex_field[idxrx2 + (rx2 / 2)*idxry2] = (holo2x[(2 * idxrx2 - 1) + rx2*(2 * idxry2 - 1)] +
				holo2x[(2 * idxrx2 - 1) + rx2*(2 * idxry2)] +
				holo2x[(2 * idxrx2) + rx2*(2 * idxry2 - 1)] +
				holo2x[(2 * idxrx2) + rx2*(2 * idxry2)]) / (Complex<Real>)4.0;
		}
	}


}

void ophLF::writeIntensity_gray8_bmp(const char* fileName, int nx, int ny, Complex<Real>* complexvalue, int k)
{
	const int n = nx * ny;

	double* intensity = (double*)malloc(sizeof(double)*n);
	for (int i = 0; i < n; i++)
		intensity[i] = complexvalue[i].real();
	//intensity[i] = complexvalue[i].mag2();

	double min_val, max_val;
	min_val = intensity[0];
	max_val = intensity[0];

	for (int i = 0; i < n; ++i)
	{
		if (min_val > intensity[i])
			min_val = intensity[i];
		else if (max_val < intensity[i])
			max_val = intensity[i];
	}

	char fname[100];
	strcpy_s(fname, fileName);
	if (k != -1)
	{
		char num[30];
		sprintf_s(num, "_%d", k);
		strcat_s(fname, num);
	}
	strcat_s(fname, ".bmp");

	//LOG("minval %e, max val %e\n", min_val, max_val);

	unsigned char* cgh = (unsigned char*)malloc(sizeof(unsigned char)*n);

	for (int i = 0; i < n; ++i) {
		double val = (intensity[i] - min_val) / (max_val - min_val);
		//val = pow(val, 1.0 / 1.5);
		val = val * 255.0;
		unsigned char v = (uchar)val;

		cgh[i] = v;
	}

	int ret = Openholo::saveAsImg(fname, 8, cgh, nx, ny);

	free(intensity);
	free(cgh);
}
