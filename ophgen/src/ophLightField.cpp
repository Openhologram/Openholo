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
#include <fstream>
#ifdef _WIN64
#include <io.h>
#include <direct.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#endif

ophLF::ophLF(void)
	: num_image(ivec2(0, 0))
	, resolution_image(ivec2(0, 0))
	, distanceRS2Holo(0.0)
	, fieldLens(0.0)
	, is_ViewingWindow(false)
	, nImages(-1)
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

	bool bRet = true;

	using namespace tinyxml2;
	/*XML parsing*/
	tinyxml2::XMLDocument xml_doc;
	XMLNode *xml_node;

	if (!checkExtension(fname, ".xml"))
	{
		LOG("<FAILED> Wrong file ext.\n");
		return false;
	}
	if (xml_doc.LoadFile(fname) != XML_SUCCESS)
	{
		LOG("<FAILED> Loading file.\n");
		return false;
	}

	xml_node = xml_doc.FirstChild();

	char szNodeName[32] = { 0, };
	sprintf(szNodeName, "FieldLength");
	// about viewing window
	auto next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&fieldLens))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}

	// about image
	sprintf(szNodeName, "Image_NumOfX");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryIntText(&num_image[_X]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Integer) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "Image_NumOfY");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryIntText(&num_image[_Y]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Integer) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "Image_Width");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryIntText(&resolution_image[_X]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Integer) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "Image_Height");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryIntText(&resolution_image[_Y]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Integer) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "Distance");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&distanceRS2Holo))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}

	initialize();

	LOG("**************************************************\n");
	LOG("             Read Config (Light Field)            \n");
	LOG("1) Focal Length : %.5lf\n", distanceRS2Holo);
	LOG("2) Number of Images : %d x %d\n", num_image[_X], num_image[_Y]);
	LOG("3) Resolution of Each Image : %d x %d\n", resolution_image[_X], resolution_image[_Y]);
	LOG("4) Field Length (Unused) : %.5lf\n", fieldLens);
	LOG("**************************************************\n");

	return bRet;
}

int ophLF::loadLF(const char* directory, const char* exten)
{
	int nWave = context_.waveNum;

	initializeLF();

#ifdef _WIN64
	_finddata_t data;

	string sdir = std::string(directory).append("\\").append("*.").append(exten);
	intptr_t ff = _findfirst(sdir.c_str(), &data);

	if (ff != -1)
	{
		int num = 0;
		ivec2 sizeOut;
		int bytesperpixel;

		while (true)
		{
			string imgfullname = std::string(directory).append("/").append(data.name);
			getImgSize(sizeOut[_X], sizeOut[_Y], bytesperpixel, imgfullname.c_str());

			int size = (((sizeOut[_X] * bytesperpixel) + 3) & ~3) * sizeOut[_Y];

			if (nWave == 1)
			{
				size = ((sizeOut[_X] + 3) & ~3) * sizeOut[_Y];
				uchar* img = loadAsImg(imgfullname.c_str());
				m_vecImages[num] = new uchar[size];
				convertToFormatGray8(img, m_vecImages[num], sizeOut[_X], sizeOut[_Y], bytesperpixel);
				m_vecImgSize[num] = size;
				if (img == nullptr) {
					LOG("<FAILED> Load image.\n");
					return -1;
				}
			}
			else
			{
				m_vecImages[num] = loadAsImg(imgfullname.c_str());
				m_vecImgSize[num] = size;
				if (m_vecImages[num] == nullptr) {
					LOG("<FAILED> Load image.\n");
					return -1;
				}
			}
			num++;

			int out = _findnext(ff, &data);
			if (out == -1)
				break;
		}
		_findclose(ff);

		if (num_image[_X] * num_image[_Y] != num) {
			LOG("<FAILED> Not matching image.\n");
		}
		return 1;
}
	else
	{
		LOG("<FAILED> Load image.\n");
		return -1;
	}

#else

	string sdir;
	DIR* dir = nullptr;
	if (directory[0] != '/') {
		char buf[PATH_MAX] = { 0, };
		if (getcwd(buf, sizeof(buf)) != nullptr) {
			sdir = sdir.append(buf).append("/").append(directory);
		}
	}
	else
		sdir = string(directory);
	string ext = string(exten);

	if ((dir = opendir(sdir.c_str())) != nullptr) {

		int num = 0;
		ivec2 sizeOut;
		int bytesperpixel;
		struct dirent* ent;

		// Add file
		int cnt = 0;
		vector<string> fileList;
		while ((ent = readdir(dir)) != nullptr) {
			string filePath;
			filePath = filePath.append(sdir.c_str()).append("/").append(ent->d_name);
			if (filePath != "." && filePath != "..") {
				struct stat fileInfo;
				if (stat(filePath.c_str(), &fileInfo) == 0 && S_ISREG(fileInfo.st_mode)) {
					if (filePath.substr(filePath.find_last_of(".") + 1) == ext) {
						fileList.push_back(filePath);
						cnt++;
					}
				}
			}
		}
		closedir(dir);
		std::sort(fileList.begin(), fileList.end());

		for (size_t i = 0; i < fileList.size(); i++)
		{
			// to do
			getImgSize(sizeOut[_X], sizeOut[_Y], bytesperpixel, fileList[i].c_str());
			int size = (((sizeOut[_X] * bytesperpixel) + 3) & ~3) * sizeOut[_Y];

			if (nWave == 1)
			{
				size = ((sizeOut[_X] + 3) & ~3) * sizeOut[_Y];
				uchar* img = loadAsImg(fileList[i].c_str());
				m_vecImages[i] = new uchar[size];
				convertToFormatGray8(img, m_vecImages[i], sizeOut[_X], sizeOut[_Y], bytesperpixel);
				m_vecImgSize[i] = size;
				if (img == nullptr) {
					LOG("<FAILED> Load image.\n");
					return -1;
				}
			}
			else
			{
				m_vecImages[i] = loadAsImg(fileList[i].c_str());
				m_vecImgSize[i] = size;
				if (m_vecImages[i] == nullptr) {
					LOG("<FAILED> Load image.\n");
					return -1;
				}
			}
		}
		if (num_image[_X] * num_image[_Y] != (int)fileList.size()) {
			LOG("<FAILED> Not matching image.\n");
		}
		return 1;

	}
	else
	{
		LOG("<FAILED> Load image : %s\n", sdir.c_str());
		return -1;
	}
#endif
}


void ophLF::generateHologram()
{
	resetBuffer();
	LOG("**************************************************\n");
	LOG("                Generate Hologram                 \n");
	LOG("1) Algorithm Method : Light Field\n");
	LOG("2) Generate Hologram with %s\n", m_mode & MODE_GPU ?
		"GPU" :
#ifdef _OPENMP
		"Multi Core CPU"
#else
		"Single Core CPU"
#endif
	);
	LOG("3) Use Random Phase : %s\n", GetRandomPhase() ? "Y" : "N");
	LOG("**************************************************\n");

	auto begin = CUR_TIME;

	if (m_mode & MODE_GPU)
	{
		convertLF2ComplexField_GPU();
	}
	else
	{
		convertLF2ComplexField();

		for (uint ch = 0; ch < context_.waveNum; ch++)
		{
			fresnelPropagation(m_vecRSplane[ch], complex_H[ch], distanceRS2Holo, ch);
		}
	}
	fftFree();
	LOG("Total Elapsed Time: %.5lf (sec)\n", ELAPSED_TIME(begin, CUR_TIME));
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
}

void ophLF::convertLF2ComplexField()
{
	auto begin = CUR_TIME;

	const uint nX = num_image[_X];
	const uint nY = num_image[_Y];
	const long long int N = nX * nY; // Image count

	const uint rX = resolution_image[_X];
	const uint rY = resolution_image[_Y];
	const uint R = rX * rY; // LF Image resolution
	const uint nWave = context_.waveNum;
	const bool bRandomPhase = GetRandomPhase();

	// initialize
	for (vector<Complex<Real>*>::iterator it = m_vecRSplane.begin(); it != m_vecRSplane.end(); it++) delete[](*it);
	m_vecRSplane.clear();
	m_vecRSplane.resize(nWave);

	for (uint i = 0; i < nWave; i++)
	{
		m_vecRSplane[i] = new Complex<Real>[N * R];
		//memset(m_vecRSplane[i], 0.0, sizeof(Complex<Real>) * N * R);
	}

	Complex<Real> *tmp = new Complex<Real>[N];
	Real pi2 = M_PI * 2;
	for (uint ch = 0; ch < nWave; ch++)
	{
		int iColor = nWave - ch - 1;
		for (uint r = 0; r < R; r++) // pixel num
		{
			int w = r % rX;
			int h = r / rX;
			int iWidth = r * nWave;

			for (uint n = 0; n < N; n++) // image num
			{
				tmp[n][_RE] = (Real)(m_vecImages[n][iWidth + iColor]);
				tmp[n][_IM] = 0.0;
			}

			fft2(tmp, tmp, nX, nY, OPH_FORWARD);

			int base1 = N * rX * h;
			int base2 = w * nX;
			for (uint n = 0; n < N; n++)
			{
				uint j = n % nX;
				uint i = n / nX;

				Real randVal = bRandomPhase ? rand(0.0, 1.0) : 1.0;
				Complex<Real> phase(0, pi2 * randVal);
				m_vecRSplane[ch][base1 + base2 + ((n - j) * rX) + j] = tmp[n] * phase.exp();
			}
		}	
	}


	delete[] tmp;
	fftFree();
	LOG("%s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
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
	strcpy(fname, fileName);
	if (k != -1)
	{
		char num[30];
		sprintf(num, "_%d", k);
		strcat(fname, num);
	}
	strcat(fname, ".bmp");

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