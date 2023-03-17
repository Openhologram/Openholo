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

#include	"ophDepthMap.h"
#include	<random>
#ifdef _WIN64
#include	<io.h>
#include	<direct.h>
#else
#include	<dirent.h>
#endif
#include    "sys.h"
#include	"tinyxml2.h"
#include	"include.h"

ophDepthMap::ophDepthMap()
	: ophGen()
	, m_nProgress(0)
{
	// GPU Variables
	img_src_gpu = nullptr;
	dimg_src_gpu = nullptr;
	depth_index_gpu = nullptr;

	depth_img = nullptr;
	m_vecRGB.clear();

	// CPU Variables
	dmap_src = nullptr;
	depth_index = nullptr;
	dmap = 0;
	dstep = 0;
	dlevel.clear();
	setViewingWindow(false);
	LOG("*** DEPTH MAP : BUILD DATE: %s %s ***\n\n", __DATE__, __TIME__);
}

ophDepthMap::~ophDepthMap()
{
}

void ophDepthMap::setViewingWindow(bool is_ViewingWindow)
{
	this->is_ViewingWindow = is_ViewingWindow;
}

bool ophDepthMap::readConfig(const char * fname)
{
	if (!ophGen::readConfig(fname))
		return false;

	bool bRet = true;
	/*XML parsing*/

	using namespace tinyxml2;
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

	char szNodeName[32] = { 0, };
	sprintf(szNodeName, "FlagChangeDepthQuantization");
	auto next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryBoolText(&dm_config_.change_depth_quantization))
	{
		LOG("<FAILED> Not found node : \'%s\' (Boolean) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "DefaultDepthQuantization");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryUnsignedText(&dm_config_.default_depth_quantization))
	{
		LOG("<FAILED> Not found node : \'%s\' (Unsinged Integer) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "NumberOfDepthQuantization");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryUnsignedText(&dm_config_.num_of_depth_quantization))
	{
		LOG("<FAILED> Not found node : \'%s\' (Unsinged Integer) \n", szNodeName);
		bRet = false;
	}

	if (dm_config_.change_depth_quantization == 0)
		dm_config_.num_of_depth = dm_config_.default_depth_quantization;
	else
		dm_config_.num_of_depth = dm_config_.num_of_depth_quantization;

	string render_depth;
	sprintf(szNodeName, "RenderDepth");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryBoolText(&dm_config_.change_depth_quantization))
	{
		LOG("<FAILED> Not found node : \'%s\' (Boolean) \n", szNodeName);
		bRet = false;
	}
	else
		render_depth = (xml_node->FirstChildElement(szNodeName))->GetText();

	size_t found = render_depth.find(':');
	if (found != string::npos)
	{
		string s = render_depth.substr(0, found);
		string e = render_depth.substr(found + 1);
		int start = stoi(s);
		int end = stoi(e);
		dm_config_.render_depth.clear();
		for (int k = start; k <= end; k++)
			dm_config_.render_depth.push_back(k);
	}
	else
	{
		stringstream ss(render_depth);
		int render;

		while (ss >> render)
			dm_config_.render_depth.push_back(render);
	}

	if (dm_config_.render_depth.empty()) {
		LOG("<FAILED> Not found node : \'%s\' (String) \n", szNodeName);
		bRet = false;
	}

	sprintf(szNodeName, "RandomPhase");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryBoolText(&dm_config_.random_phase))
	{
		LOG("<FAILED> Not found node : \'%s\' (Boolean) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "FieldLength");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&dm_config_.fieldLength))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "NearOfDepth");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&dm_config_.near_depthmap))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "FarOfDepth");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&dm_config_.far_depthmap))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}

	initialize();

	LOG("**************************************************\n");
	LOG("              Read Config (Depth Map)             \n");
	LOG("1) Focal Length : %.5lf ~ %.5lf\n", dm_config_.near_depthmap, dm_config_.far_depthmap);
	LOG("2) Render Depth : %d:%d\n", dm_config_.render_depth[0], dm_config_.render_depth[dm_config_.render_depth.size() - 1]);
	LOG("3) Number of Depth Quantization : %d\n", dm_config_.num_of_depth_quantization);
	LOG("**************************************************\n");

	return bRet;
}

bool ophDepthMap::readImage(const char* fname, IMAGE_TYPE type)
{
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const int N = pnX * pnY;
	const int ch = context_.waveNum;

	int w, h, bytesperpixel;
	bool ret = getImgSize(w, h, bytesperpixel, fname);

	uchar *pSrc = loadAsImg(fname);

	if (pSrc == nullptr) {
		LOG("<FAILED> Load image: %s\n", fname);
		return false;
	}

	if (type == RGB)
	{
		for (vector<uchar *>::iterator it = m_vecRGB.begin(); it != m_vecRGB.end(); it++) delete[](*it);
		m_vecRGB.clear();

		uchar* pBuf = new uchar[w * h];

		for (int i = 0; i < ch; i++)
		{
			// step 1. color relocation
			if (ch == 1) // rgb to greyscale
			{
				if (bytesperpixel != 1)
					convertToFormatGray8(pSrc, pBuf, w, h, bytesperpixel);
				else
					memcpy(pBuf, pSrc, w * h);
			}
			else if (ch == bytesperpixel)
			{
				// [B0,G0,R0,B1,G1,R1...] -> [R0,R1...] [G0,G1...] [B0,B1...]
				if (separateColor(i, w, h, pSrc, pBuf))
				{

				}
			}
			else
			{
				memcpy(pBuf, pSrc, w * h);
			}

			uchar* pDst = new uchar[N];

			if (w != pnX || h != pnY)
			{
				imgScaleBilinear(pBuf, pDst, w, h, pnX, pnY, ch);
			}
			else
			{
				memcpy(pDst, pBuf, N);
			}
			m_vecRGB.push_back(pDst);
		}
		delete[] pSrc;
		delete[] pBuf;

		// 2019-10-14 mwnam
		m_vecRGBImg[_X] = pnX;
		m_vecRGBImg[_Y] = pnY;
	}
	else if (type == DEPTH)
	{
		uchar* pBuf = new uchar[w * h];
		if (depth_img) delete[] depth_img;

		depth_img = new uchar[N];
		memset(depth_img, 0, sizeof(char) * N);

		if (w != pnX || h != pnY)
			imgScaleBilinear(pSrc, depth_img, w, h, pnX, pnY);
		else
			memcpy(depth_img, pSrc, sizeof(char) * N);

		// 2019-10-14 mwnam
		m_vecDepthImg[_X] = pnX;
		m_vecDepthImg[_Y] = pnY;
	}
	else
	{
		LOG("<FAILED> Unknown image type: %s\n", fname);
		return false;
	}
	LOG(" <SUCCEEDED> Load image: %s\n", fname);
	return true;
}

bool ophDepthMap::readImageDepth(const char* source_folder, const char* img_prefix, const char* depth_img_prefix)
{
	auto begin = CUR_TIME;
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const int N = pnX * pnY;

	for (size_t i = 0; i < m_vecRGB.size(); i++)
	{
		delete[] m_vecRGB[i];
	} 
	m_vecRGB.clear();

	// RGB Image

#ifdef _WIN64
	std::string sdir = source_folder;
	sdir = sdir.append("\\").append(img_prefix).append("*.bmp");
	_finddatai64_t fd;
	intptr_t handle;
	handle = _findfirst64(sdir.c_str(), &fd);
	if (handle == -1)
	{
		LOG("<FAILED> Source image does not exist: %s.\n", sdir.c_str());
		LOG("%.5lf (sec)\n.", ELAPSED_TIME(begin, CUR_TIME));
		return false;
	}
	std::string imgfullname;
	imgfullname = std::string(source_folder).append("\\").append(fd.name);
#else
	std::string sdir = source_folder;
	std::string file = std::string(img_prefix) + ".bmp";

	DIR* dir = nullptr;
	struct dirent* ent;
	if ((dir = opendir(sdir.c_str())) != nullptr) {
		while ((ent = readdir(dir)) != nullptr) {
			if (!strcmp(file.c_str(), ent->d_name))	break;
		}
		closedir(dir);
	}
	else
	{
		LOG("<FAILED> Source image does not exist: %s.\n", sdir.c_str());
		LOG("%.5lf (sec)\n.", ELAPSED_TIME(begin, CUR_TIME));
		return false;
	}

	std::string imgfullname;
	imgfullname = std::string(source_folder).append("/").append(file);

#endif

	int w, h, bytesperpixel;
	bool ret = getImgSize(w, h, bytesperpixel, imgfullname.c_str());

	// RGB Image
	oph::uchar* buf = new uchar[w * h * bytesperpixel]; // 1-Dimension left top
	ret = loadAsImgUpSideDown(imgfullname.c_str(), buf);
	if (!ret) {
		LOG("<FAILED> Image Load: %s\n", imgfullname.c_str());
		LOG("%s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
		return false;
	}
	LOG(" <SUCCEEDED> Image Load: %s\n", imgfullname.c_str());

	int ch = context_.waveNum;
	uchar* img = new uchar[w * h];

	for (int i = 0; i < ch; i++)
	{
		if (ch == 1) // rgb img to grayscale
			convertToFormatGray8(buf, img, w, h, bytesperpixel);
		else if (ch == bytesperpixel) // rgb img to rgb
		{
			separateColor(i, w, h, buf, img);
		}
		else // grayscale img to rgb
		{
			memcpy(img, buf, sizeof(char) * w * h);
		}


		//resized image
		uchar *rgb_img = new uchar[N];
		memset(rgb_img, 0, sizeof(char) * N);

		if (w != pnX || h != pnY)
			imgScaleBilinear(img, rgb_img, w, h, pnX, pnY, ch);
		else
			memcpy(rgb_img, img, sizeof(char) * N);

		m_vecRGB.push_back(rgb_img);
	}
	delete[] buf;

	// 2019-10-14 mwnam
	m_vecRGBImg[_X] = pnX;
	m_vecRGBImg[_Y] = pnY;

	// Depth Image
	//=================================================================================
#ifdef _WIN64
	std::string sddir = std::string(source_folder).append("\\").append(depth_img_prefix).append("*.bmp");
	handle = _findfirst64(sddir.c_str(), &fd);
	if (handle == -1)
	{
		LOG("<FAILED> Source depthmap does not exist: %s.\n", sddir.c_str());
		LOG("%s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
		return false;
	}

	std::string dimgfullname = std::string(source_folder).append("\\").append(fd.name);
#else
	std::string sddir = std::string(source_folder);
	std::string file2 = std::string(depth_img_prefix) + ".bmp";

	if ((dir = opendir(sddir.c_str())) != nullptr) {
		while ((ent = readdir(dir)) != nullptr) {
			if (!strcmp(file.c_str(), ent->d_name)) break;
		}
		closedir(dir);
	}
	else
	{
		LOG("<FAILED> Source depthmap does not exist: %s.\n", sddir.c_str());
		LOG("%.5lf (sec)\n.", ELAPSED_TIME(begin, CUR_TIME));
		return false;
	}
	std::string dimgfullname = std::string(source_folder).append("/").append(file2);

#endif
	int dw, dh, dbytesperpixel;
	ret = getImgSize(dw, dh, dbytesperpixel, dimgfullname.c_str());
	
	// Depth Image
	uchar* dbuf = new uchar[dw * dh * dbytesperpixel];
	ret = loadAsImgUpSideDown(dimgfullname.c_str(), dbuf);
	if (!ret) {
		LOG("<FAILED> Image Load: %s\n", dimgfullname.c_str());
		LOG("%s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
		return false;
	}
	LOG(" <SUCCEEDED> Image Load: %s\n", dimgfullname.c_str());

	// 2019-10-14 mwnam
	m_vecDepthImg[_X] = dw;
	m_vecDepthImg[_Y] = dh;

	uchar* dimg = new uchar[dw * dh];
	convertToFormatGray8(dbuf, dimg, dw, dh, dbytesperpixel);

	delete[] dbuf;

	if (depth_img) delete[] depth_img;

	depth_img = new uchar[N];
	memset(depth_img, 0, sizeof(char) * N);

	if (dw != pnX || dh != pnY)
		imgScaleBilinear(dimg, depth_img, dw, dh, pnX, pnY);
	else
		memcpy(depth_img, dimg, sizeof(char) * N);

	// 2019-10-14 mwnam
	m_vecDepthImg[_X] = pnX;
	m_vecDepthImg[_Y] = pnY;

	delete[] img;
	delete[] dimg;

	LOG("%s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
	return true;
}

Real ophDepthMap::generateHologram()
{
	auto begin = CUR_TIME;
	LOG("**************************************************\n");
	LOG("                Generate Hologram                 \n");
	LOG("1) Algorithm Method : Depth Map\n");
	LOG("2) Generate Hologram with %s\n", m_mode & MODE_GPU ?
		"GPU" :
#ifdef _OPENMP
		"Multi Core CPU"
#else
		"Single Core CPU"
#endif
		);
	LOG("**************************************************\n");

	resetBuffer();
	m_vecEncodeSize = context_.pixel_number;
	if (m_mode & MODE_GPU)
	{
		prepareInputdataGPU();
		getDepthValues();
		//if (is_ViewingWindow)
		//	transVW();
		calcHoloGPU();
	}
	else
	{
		prepareInputdataCPU();
		getDepthValues();
		//if (is_ViewingWindow)
		//	transVW();
		calcHoloCPU();
	}
	
	Real elapsed_time = ELAPSED_TIME(begin, CUR_TIME);
	LOG("Total Elapsed Time: %lf (s)\n", elapsed_time);
	m_nProgress = 0;
	return elapsed_time;
}

void ophDepthMap::encoding(unsigned int ENCODE_FLAG)
{
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint nChannel = context_.waveNum;
	Complex<Real>** dst = new Complex<Real>*[nChannel];
	for (uint ch = 0; ch < nChannel; ch++) {
		dst[ch] = new Complex<Real>[pnX * pnY];
		//fft2(context_.pixel_number, nullptr, OPH_BACKWARD);
		fft2(complex_H[ch], dst[ch], pnX, pnY, OPH_BACKWARD, true);
		ophGen::encoding(ENCODE_FLAG, dst[ch], m_lpEncoded[ch]);
	}

	for (uint ch = 0; ch < nChannel; ch++)
		delete[] dst[ch];
	delete[] dst;
}

void ophDepthMap::encoding(unsigned int ENCODE_FLAG, unsigned int SSB_PASSBAND)
{
	auto begin = CUR_TIME;
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint nChannel = context_.waveNum;

	bool is_CPU = (m_mode & MODE_GPU) ? false : true;

	for (uint ch = 0; ch < nChannel; ch++) {

		if (ENCODE_FLAG == ophGen::ENCODE_SSB) {
			ivec2 location = ivec2(0, 0);
			switch (SSB_PASSBAND) {
			case SSB_TOP:
				location = ivec2(0, 1);
				break;
			case SSB_BOTTOM:
				location = ivec2(0, -1);
				break;
			case SSB_LEFT:
				location = ivec2(-1, 0);
				break;
			case SSB_RIGHT:
				location = ivec2(1, 0);
				break;
			}

			encodeSideBand(is_CPU, location);
		}
		else
		{
			Complex<Real>* dst = new Complex<Real>[pnX * pnY];
			fft2(context_.pixel_number, complex_H[ch], OPH_BACKWARD);
			fft2(complex_H[ch], dst, pnX, pnY, OPH_BACKWARD);
			ophGen::encoding(ENCODE_FLAG, SSB_PASSBAND, dst);
			delete[] dst;
		}
	}
	auto end = CUR_TIME;
	LOG("Elapsed Time: %lf(s)\n", ELAPSED_TIME(begin, end));
}

void ophDepthMap::initialize()
{
	dstep = 0;
	dlevel.clear();

	ophGen::initialize();

	initCPU();
	initGPU();
}

void ophDepthMap::getDepthValues()
{
	auto begin = CUR_TIME;
	if (dm_config_.num_of_depth > 1)
	{
		dstep = (dm_config_.far_depthmap - dm_config_.near_depthmap) / (dm_config_.num_of_depth - 1);
		Real val = dm_config_.near_depthmap;
		while (val <= dm_config_.far_depthmap)
		{
			dlevel.push_back(val);
			val += dstep;
		}
	} 
	else {

		dstep = (dm_config_.far_depthmap + dm_config_.near_depthmap) / 2;
		dlevel.push_back(dm_config_.near_depthmap);
	}
	

	if (dm_config_.change_depth_quantization == 1)
	{
		bool is_CPU = m_mode & MODE_GPU ? false : true;
		if (is_CPU)
			changeDepthQuanCPU();
		else
			changeDepthQuanGPU();
	}
	LOG("%s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
}

void ophDepthMap::transVW()
{
	Real val;
	dlevel_transform.clear();
	for (size_t p = 0; p < dlevel.size(); p++)
	{
		val = -dm_config_.fieldLength * dlevel[p] / (dlevel[p] - dm_config_.fieldLength);
		dlevel_transform.push_back(val);
	}
}

void ophDepthMap::initCPU()
{
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint N = pnX * pnY;
	const uint nChannel = context_.waveNum;

	for (vector<Real *>::iterator it = m_vecImgSrc.begin(); it != m_vecImgSrc.end(); it++)
	{
		delete[](*it);
	}
	m_vecImgSrc.clear();
	for (vector<int *>::iterator it = m_vecAlphaMap.begin(); it != m_vecAlphaMap.end(); it++)
	{
		delete[](*it);
	}
	m_vecAlphaMap.clear();

	for (uint ch = 0; ch < nChannel; ch++)
	{
		Real *img_src = new Real[N];
		int *alpha_map = new int[N];

		m_vecImgSrc.push_back(img_src);
		m_vecAlphaMap.push_back(alpha_map);
	}

	if (dmap_src) delete[] dmap_src;
	dmap_src = new Real[N];

	if (depth_index) delete[] depth_index;
	depth_index = new uint[N];

	if (dmap) delete[] dmap;
	dmap = new Real[N];

	fftw_cleanup();
}

bool ophDepthMap::prepareInputdataCPU()
{
	auto begin = CUR_TIME;
	const long long int N = context_.pixel_number[_X] * context_.pixel_number[_Y];
	const uint nChannel = context_.waveNum;

	memset(depth_index, 0, sizeof(uint) * N);

	if (depth_img == nullptr) // not used depth
	{
		depth_img = new uchar[N];
		memset(depth_img, 0, N);
	}

	Real gapDepth = dm_config_.far_depthmap - dm_config_.near_depthmap;
	Real nearDepth = dm_config_.near_depthmap;

	for (uint ch = 0; ch < nChannel; ch++)
	{
#ifdef _OPENMP
#pragma omp parallel for firstprivate(gapDepth, nearDepth)
#endif
		for (long long int k = 0; k < N; k++)
		{
			m_vecImgSrc[ch][k] = Real(m_vecRGB[ch][k]) / 255.0; // RGB IMG
			m_vecAlphaMap[ch][k] = (m_vecRGB[ch][k] > 0 ? 1 : 0); // RGB IMG

			// once
			if (ch == 0)
			{
				dmap_src[k] = Real(depth_img[k]) / 255.0; // DEPTH IMG
				dmap[k] = (1 - dmap_src[k]) * gapDepth + nearDepth;
				if (dm_config_.change_depth_quantization == 0) {
					depth_index[k] = dm_config_.default_depth_quantization - depth_img[k];
				}
			}
		}
	}

	LOG("%s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
	return true;
}

void ophDepthMap::changeDepthQuanCPU()
{
	auto begin = CUR_TIME;
	const long long int N = context_.pixel_number[_X] * context_.pixel_number[_Y];

	uint num_depth = dm_config_.num_of_depth;
	Real near_depth = dm_config_.near_depthmap;
	Real far_depth = dm_config_.far_depthmap;
	
	double nearv = dlevel[0];
	double half_step = dstep / 2.0;
	depth_fill.clear();
	depth_fill.resize(dm_config_.render_depth.size() + 1, 0);
	Real locDstep = dstep;

#ifdef _OPENMP
#pragma omp parallel for firstprivate(nearv, half_step, locDstep)
#endif
	for (long long int i = 0; i < N; i++)
	{
		int idx = int(((dmap[i] - nearv) + half_step) / locDstep);
		depth_index[i] = idx + 1;
		depth_fill[idx + 1] = 1;
	}

	LOG("%s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
}

void ophDepthMap::calcHoloCPU()
{
	auto begin = CUR_TIME;

	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const long long int N = pnX * pnY;
	const uint nChannel = context_.waveNum;

	size_t depth_sz = dm_config_.render_depth.size();

	const bool bRandomPhase = GetRandomPhase();
	Complex<Real> *input = new Complex<Real>[N];

	fftInit2D(context_.pixel_number, OPH_FORWARD, OPH_ESTIMATE);

	for (uint ch = 0; ch < nChannel; ch++)
	{
		Real lambda = context_.wave_length[ch];
		Real k = context_.k = (2 * M_PI / lambda);
		Real *img_src = m_vecImgSrc[ch];
		int *alpha_map = m_vecAlphaMap[ch];

		for (size_t i = 0; i < depth_sz; i++)
		{
			int dtr = dm_config_.render_depth[i];
			if (depth_fill[dtr])
			{
				memset(input, 0, sizeof(Complex<Real>) * N);
				Real temp_depth = (is_ViewingWindow) ? dlevel_transform[dtr - 1] : dlevel[dtr - 1];

				Complex<Real> rand_phase_val;
				GetRandomPhaseValue(rand_phase_val, bRandomPhase);

				Complex<Real> carrier_phase_delay(0, k * -temp_depth);
				carrier_phase_delay.exp();				
#ifdef _OPENMP
#pragma omp parallel for firstprivate(dtr, rand_phase_val, carrier_phase_delay)
#endif
				for (long long int j = 0; j < N; j++)
				{
					input[j][_RE] = img_src[j] * alpha_map[j] * ((int)depth_index[j] == dtr ? 1.0 : 0.0);
					input[j] *= rand_phase_val * carrier_phase_delay;
				}

				fft2(input, input, pnX, pnY, OPH_FORWARD, false);
				AngularSpectrumMethod(input, complex_H[ch], lambda, temp_depth);

			}
			m_nProgress = (int)((Real)(ch * depth_sz + i) * 100 / (depth_sz * nChannel));
		}
	}
	delete[] input;
	fftFree();
	LOG("%s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
}

void ophDepthMap::ophFree(void)
{
	ophGen::ophFree();
	if (depth_img) {
		delete[] depth_img;
		depth_img = nullptr;
	}


	for (vector<uchar *>::iterator it = m_vecRGB.begin(); it != m_vecRGB.end(); it++)
	{
		delete[](*it);
	}
	m_vecRGB.clear();
}

void ophDepthMap::setResolution(ivec2 resolution)
{	
	if (context_.pixel_number != resolution) {
		ophGen::setResolution(resolution);
		initCPU();
		initGPU();
	}
}

void ophDepthMap::normalize()
{
	ophGen::normalize();
}