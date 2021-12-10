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

ophLF::ophLF(void)
	: num_image(ivec2(0, 0))
	, resolution_image(ivec2(0, 0))
	, distanceRS2Holo(0.0)
	, is_ViewingWindow(false)
	, bSinglePrecision(false)
{
	LOG("*** LIGHT FIELD : BUILD DATE: %s %s ***\n\n", __DATE__, __TIME__);
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
		return false;

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
	next = xml_node->FirstChildElement("Distance");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&distanceRS2Holo))
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
	int nWave = context_.waveNum;

	initializeLF();

	_finddata_t data;

	string sdir = std::string(LF_directory).append("\\").append("*.").append(ext);
	intptr_t ff = _findfirst(sdir.c_str(), &data);
	if (ff != -1)
	{
		int num = 0;
		ivec2 sizeOut;
		int bytesperpixel;

		while (true)
		{
			string imgfullname = std::string(LF_directory).append("/").append(data.name);

			getImgSize(sizeOut[_X], sizeOut[_Y], bytesperpixel, imgfullname.c_str());

			int size = (((sizeOut[_X] * bytesperpixel) + 3) & ~3) * sizeOut[_Y];

			if (nWave == 1)
			{
				size = ((sizeOut[_X] + 3) & ~3) * sizeOut[_Y];
				uchar *img = loadAsImg(imgfullname.c_str());
				m_vecImages[num] = new uchar[size];
				convertToFormatGray8(img, m_vecImages[num], sizeOut[_X], sizeOut[_Y], bytesperpixel);
				m_vecImgSize[num] = size;
				if (img == nullptr) {
					cout << "LF load was failed." << endl;
					return -1;
				}
			}
			else
			{
				m_vecImages[num] = loadAsImg(imgfullname.c_str());
				m_vecImgSize[num] = size;
				if (m_vecImages[num] == nullptr) {
					cout << "LF load was failed." << endl;
					return -1;
				}
			}
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
	int nWave = context_.waveNum;
	initializeLF();

	_finddata_t data;

	string sdir = std::string("./").append(LF_directory).append("/").append("*.").append(ext);
	intptr_t ff = _findfirst(sdir.c_str(), &data);
	if (ff != -1)
	{
		int num = 0;
		ivec2 sizeOut;
		int bytesperpixel;

		while (true)
		{
			string imgfullname = std::string(LF_directory).append("/").append(data.name);

			getImgSize(sizeOut[_X], sizeOut[_Y], bytesperpixel, imgfullname.c_str());

			int size = (((sizeOut[_X] * bytesperpixel) + 3) & ~3) * sizeOut[_Y];

			if (nWave == 1)
			{
				size = ((sizeOut[_X] + 3) & ~3) * sizeOut[_Y];
				uchar *img = loadAsImg(imgfullname.c_str());
				m_vecImages[num] = new uchar[size];
				convertToFormatGray8(img, m_vecImages[num], sizeOut[_X], sizeOut[_Y], bytesperpixel);
				m_vecImgSize[num] = size;
				if (img == nullptr) {
					cout << "LF load was failed." << endl;
					return -1;
				}
			}
			else
			{
				m_vecImages[num] = loadAsImg(imgfullname.c_str());
				m_vecImgSize[num] = size;
				if (m_vecImages[num] == nullptr) {
					cout << "LF load was failed." << endl;
					return -1;
				}
			}
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
	LOG("2) Generate Hologram with %s\n", m_mode & MODE_GPU ?
		"GPU" :
#ifdef _OPENMP
		"Multi Core CPU"
#else
		"Single Core CPU"
#endif
	);
	LOG("3) Random Phase Use : %s\n", GetRandomPhase() ? "Y" : "N");
	//LOG("3) Transform Viewing Window : %s\n", is_ViewingWindow ? "ON" : "OFF");

	auto begin = CUR_TIME;

	if (m_mode & MODE_GPU)
	{
		convertLF2ComplexField_GPU();
	}
	else
	{
		convertLF2ComplexField();

		for (int ch = 0; ch < context_.waveNum; ch++)
		{
			fresnelPropagation(m_vecRSplane[ch], complex_H[ch], distanceRS2Holo, ch);
		}
	}
	fftFree();
	LOG("Total Elapsed Time: %lf (s)\n", ELAPSED_TIME(begin, CUR_TIME));
}

void ophLF::initializeLF()
{
	for (vector<uchar *>::iterator it = m_vecImages.begin(); it != m_vecImages.end(); it++) delete[](*it);
	m_vecImages.clear();
	m_vecImgSize.clear();

	const int nX = num_image[_X];
	const int nY = num_image[_Y];
	const int N = nX * nY;

	m_vecImages.resize(N);
	m_vecImgSize.resize(N);
	nImages = N;
	cout << "The Number of the Images : " << N << endl;
}

void ophLF::convertLF2ComplexField()
{
	auto begin = CUR_TIME;

	const uint nX = num_image[_X];
	const uint nY = num_image[_Y];
	const uint N = nX * nY; // Image count

	const uint rX = resolution_image[_X];
	const uint rY = resolution_image[_Y];
	const uint R = rX * rY; // LF Image resolution
	const uint nWave = context_.waveNum;
	const bool bRandomPhase = GetRandomPhase();

	// initialize
	for (vector<Complex<Real>*>::iterator it = m_vecRSplane.begin(); it != m_vecRSplane.end(); it++) delete[](*it);
	m_vecRSplane.clear();
	m_vecRSplane.resize(nWave);

	for (int i = 0; i < nWave; i++)
	{
		m_vecRSplane[i] = new Complex<Real>[N * R];
		//memset(m_vecRSplane[i], 0.0, sizeof(Complex<Real>) * N * R);
	}

	Complex<Real> *tmp = new Complex<Real>[N];
	Real pi2 = M_PI * 2;
	for (int ch = 0; ch < nWave; ch++)
	{
		int iColor = nWave - ch - 1;
		for (int r = 0; r < R; r++) // pixel num
		{
			int w = r % rX;
			int h = r / rX;
			int iWidth = r * nWave;

			for (int n = 0; n < N; n++) // image num
			{
				tmp[n][_RE] = (Real)(m_vecImages[n][iWidth + iColor]);
				tmp[n][_IM] = 0.0;
			}

			fft2(tmp, tmp, nX, nY, OPH_FORWARD);

			int base1 = N * rX * h;
			int base2 = w * nX;
			for (int n = 0; n < N; n++)
			{
				int j = n % nX;
				int i = n / nX;

				Real randVal = bRandomPhase ? rand(0.0, 1.0) : 1.0;
				Complex<Real> phase(0, pi2 * randVal);
				m_vecRSplane[ch][base1 + base2 + ((n - j) * rX) + j] = tmp[n] * phase.exp();
			}
		}	
	}


	delete[] tmp;
	fftFree();
	LOG("\n%s : %lf(s)\n\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
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

void ophLF::ophFree()
{
	ophGen::ophFree();
	
	for (vector<uchar *>::iterator it = m_vecImages.begin(); it != m_vecImages.end(); it++) delete[](*it);
	for (vector<Complex<Real> *>::iterator it = m_vecRSplane.begin(); it != m_vecRSplane.end(); it++) delete[](*it);
	m_vecImages.clear();
	m_vecImgSize.clear();
	m_vecRSplane.clear();

}