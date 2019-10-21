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
{
}

void ophLF::setMode(bool isCPU)
{
	is_CPU = isCPU;
}

void ophLF::setViewingWindow(bool is_ViewingWindow)
{
	this->is_ViewingWindow = is_ViewingWindow;
}

int ophLF::readLFConfig(const char* LF_config) {
	LOG("Reading....%s...\n", LF_config);

	auto start = CUR_TIME;

	/*XML parsing*/
	tinyxml2::XMLDocument xml_doc;
	tinyxml2::XMLNode *xml_node;

	if (checkExtension(LF_config, ".xml") == 0)
	{
		LOG("file's extension is not 'xml'\n");
		return false;
	}
	auto ret = xml_doc.LoadFile(LF_config);
	if (ret != tinyxml2::XML_SUCCESS)
	{
		LOG("Failed to load file \"%s\"\n", LF_config);
		return false;
	}

	xml_node = xml_doc.FirstChild();

	//LF_directory = (xml_node->FirstChildElement("LightFieldImageDirectory"))->GetText();
	//ext = (xml_node->FirstChildElement("LightFieldImageExtention"))->GetText();
#if REAL_IS_DOUBLE & true
	auto next = xml_node->FirstChildElement("FieldLens");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&fieldLens))
		return false;		
	next = xml_node->FirstChildElement("DistanceRS2Holo");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&distanceRS2Holo))
		return false;
	next = xml_node->FirstChildElement("SLMPixelPitchX");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&context_.pixel_pitch[_X]))
		return false;
	next = xml_node->FirstChildElement("SLMPixelPitchY");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&context_.pixel_pitch[_Y]))
		return false;
	next = xml_node->FirstChildElement("WavelengthofLaser");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&context_.wave_length[0]))
		return false;
	//(xml_node->FirstChildElement("DistanceRS2Holo"))->QueryDoubleText(&distanceRS2Holo);
	//(xml_node->FirstChildElement("SLMPixelPitchX"))->QueryDoubleText(&context_.pixel_pitch[_X]);
	//(xml_node->FirstChildElement("SLMPixelPitchY"))->QueryDoubleText(&context_.pixel_pitch[_Y]);
	//(xml_node->FirstChildElement("WavelengthofLaser"))->QueryDoubleText(&context_.wave_length[0]);
#else
	auto next = xml_node->FirstChildElement("DistanceRS2Holo");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryFloatText(&distanceRS2Holo))
		return false;
	next = xml_node->FirstChildElement("SLMPixelPitchX");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryFloatText(&context_.pixel_pitch[_X]))
		return false;
	next = xml_node->FirstChildElement("SLMPixelPitchY");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryFloatText(&context_.pixel_pitch[_Y]))
		return false;
	next = xml_node->FirstChildElement("WavelengthofLaser");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryFloatText(&context_.wave_length[0]))
		return false;
	//(xml_node->FirstChildElement("DistanceRS2Holo"))->QueryFloatText(&distanceRS2Holo);
	//(xml_node->FirstChildElement("SLMPixelPitchX"))->QueryFloatText(&context_.pixel_pitch[_X]);
	//(xml_node->FirstChildElement("SLMPixelPitchY"))->QueryFloatText(&context_.pixel_pitch[_Y]);
	//(xml_node->FirstChildElement("WavelengthofLaser"))->QueryFloatText(&context_.wave_length[0]);
#endif
	next = xml_node->FirstChildElement("NumberofImagesXofLF");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryIntText(&num_image[_X]))
		return false;
	next = xml_node->FirstChildElement("NumberofImagesYofLF");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryIntText(&num_image[_Y]))
		return false;
	next = xml_node->FirstChildElement("NumberofPixelXofLF");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryIntText(&resolution_image[_X]))
		return false;
	next = xml_node->FirstChildElement("NumberofPixelYofLF");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryIntText(&resolution_image[_Y]))
		return false;
	//(xml_node->FirstChildElement("NumberofImagesXofLF"))->QueryIntText(&num_image[_X]);
	//(xml_node->FirstChildElement("NumberofImagesYofLF"))->QueryIntText(&num_image[_Y]);
	//(xml_node->FirstChildElement("NumberofPixelXofLF"))->QueryIntText(&resolution_image[_X]);
	//(xml_node->FirstChildElement("NumberofPixelYofLF"))->QueryIntText(&resolution_image[_Y]);
	//(xml_node->FirstChildElement("EncodingMethod"))->QueryIntText(&ENCODE_METHOD);
	//(xml_node->FirstChildElement("SingleSideBandPassBand"))->QueryIntText(&SSB_PASSBAND);

	context_.pixel_number[_X] = num_image[_X] * resolution_image[_X];
	context_.pixel_number[_Y] = num_image[_Y] * resolution_image[_Y];

	context_.k = (2 * M_PI) / context_.wave_length[0];
	context_.ss[_X] = context_.pixel_number[_X] * context_.pixel_pitch[_X];
	context_.ss[_Y] = context_.pixel_number[_Y] * context_.pixel_pitch[_Y];

	cout << endl;
	cout << "SLM pixel pitch: " << context_.pixel_pitch[_X] << ", " << context_.pixel_pitch[_Y] << endl;
	cout << "Wavelength of LASER: " << context_.wave_length[0] << endl;
	cout << "Distance RS plane to Hologram plane: " << distanceRS2Holo << endl;
	cout << "# of images: " << num_image[_X] << ", " << num_image[_Y] << endl;
	cout << "Resolution of the images: " << resolution_image[_X] << ", " << resolution_image[_Y] << endl;
	cout << "Resolution of hologram: " << context_.pixel_number[_X] << ", " << context_.pixel_number[_Y] << endl;
	cout << endl;

	setPixelNumberOHC(context_.pixel_number);
	setPixelPitchOHC(context_.pixel_pitch);
	addWaveLengthOHC(context_.wave_length[0]);

	auto end = CUR_TIME;

	auto during = ((std::chrono::duration<Real>)(end - start)).count();
	initialize();
	LOG("%.5lfsec...done\n", during);
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
	MEMORYSTATUS memStatus;
	GlobalMemoryStatus(&memStatus);
	LOG("\n*Available Memory: %u (byte)\n", memStatus.dwAvailVirtual);
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
		fresnelPropagation(RSplane_complex_field, (*complex_H), distanceRS2Holo);
	}
	else
	{
		prepareInputdataGPU();
		convertLF2ComplexField_GPU();
		fresnelPropagation_GPU();
	}

	auto end = CUR_TIME;
	elapsedTime = ((std::chrono::duration<Real>)(end - begin)).count();
	LOG("Total Elapsed Time: %lf (sec)\n", elapsedTime);
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
			if (LF[i]) delete[] LF[i];
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
#ifdef CHECK_PROC_TIME
	auto begin = CUR_TIME;
#endif
	const int nX = num_image[_X];
	const int nY = num_image[_Y];
	const int nXY = nX * nY;
	const int rX = resolution_image[_X];
	const int rY = resolution_image[_Y];
	const int rXY = rX * rY;

	if (RSplane_complex_field) {
		delete[] RSplane_complex_field;
		RSplane_complex_field = nullptr;
	}
	RSplane_complex_field = new Complex<Real>[nXY * rXY];

	Complex<Real>* complexLF = new Complex<Real>[rXY];

	Complex<Real>* FFTLF = new Complex<Real>[rXY];

	Real randVal;
	Complex<Real> phase(0.0, 0.0);

	for (int idxrX = 0; idxrX < rX; idxrX++) { // 192
		for (int idxrY = 0; idxrY < rY; idxrY++) { // 108
#if 0
			complexLF[idxnX + nX*idxnY]
#else
			for (int idxnY = 0; idxnY < nY; idxnY++) { // 10
				for (int idxnX = 0; idxnX < nX; idxnX++) { // 10
					(*(complexLF + (idxnX + nX*idxnY))) = (Real)*(*(LF + (idxnX + nX*idxnY)) + (idxrX + rX*idxrY));
				}
			}
#endif
			fft2(num_image, complexLF, OPH_FORWARD, OPH_ESTIMATE);
#if 1
			fftwShift(complexLF, FFTLF, nX, nY, OPH_FORWARD);
#else
			fftwShift(complexLF, FFTLF, rX, rY, OPH_FORWARD);
#endif
			//fftExecute(FFTLF);

			for (int idxnX = 0; idxnX < nX; idxnX++) { // 10
				for (int idxnY = 0; idxnY < nY; idxnY++) { // 10

					randVal = rand((Real)0, (Real)1, idxrX*idxrY);
					phase(0, 2 * M_PI*randVal);

					*(RSplane_complex_field + nXY*rX*idxrY + nX*rX*idxnY + nX*idxrX + idxnX) = *(FFTLF + (idxnX + nX*idxnY))*exp(phase);

				}
			}		
		}
	}
	delete[] complexLF, FFTLF;
	fftFree();
#ifdef CHECK_PROC_TIME
	auto end = CUR_TIME;
	LOG("\n%s : %lf(s)\n\n", __FUNCTION__, ((std::chrono::duration<Real>)(end - begin)).count());
#endif
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