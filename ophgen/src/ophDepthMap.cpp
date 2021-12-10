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
#include	<iomanip>
#include	<io.h>
#include	<direct.h>
#include    "sys.h"
#include	"tinyxml2.h"
#include	"include.h"

/** 
* @brief Constructor
* @details Initialize variables.
*/
ophDepthMap::ophDepthMap()
	: ophGen()
	, m_nProgress(0)
	, bSinglePrecision(false)
{
	// GPU Variables
	img_src_gpu = nullptr;
	dimg_src_gpu = nullptr;
	depth_index_gpu = nullptr;

	depth_img = nullptr;
	m_vecRGB.clear();

	// CPU Variables
	img_src = nullptr;
	dmap_src = nullptr;
	alpha_map = nullptr;
	depth_index = nullptr;
	dmap = 0;
	dstep = 0;
	dlevel.clear();
	setViewingWindow(FALSE);
	LOG("*** DEPTH MAP : BUILD DATE: %s %s ***\n\n", __DATE__, __TIME__);
}

/**
* @brief Destructor 
*/
ophDepthMap::~ophDepthMap()
{
}

void ophDepthMap::setViewingWindow(bool is_ViewingWindow)
{
	this->is_ViewingWindow = is_ViewingWindow;
}
/**
* @brief Read parameters from a config file(config_openholo.txt).
* @return true if config infomation are sucessfully read, flase otherwise.
*/
bool ophDepthMap::readConfig(const char * fname)
{
	if (!ophGen::readConfig(fname))
		return false;

	LOG("Reading....%s...", fname);
	auto start = CUR_TIME;
	/*XML parsing*/

	using namespace tinyxml2;
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
	auto next = xml_node->FirstChildElement("FlagChangeDepthQuantization");
	if (!next || XML_SUCCESS != next->QueryBoolText(&dm_config_.FLAG_CHANGE_DEPTH_QUANTIZATION))
		return false;
	next = xml_node->FirstChildElement("DefaultDepthQuantization");
	if (!next || XML_SUCCESS != next->QueryUnsignedText(&dm_config_.DEFAULT_DEPTH_QUANTIZATION))
		return false;
	next = xml_node->FirstChildElement("NumberOfDepthQuantization");
	if (!next || XML_SUCCESS != next->QueryUnsignedText(&dm_config_.NUMBER_OF_DEPTH_QUANTIZATION))
		return false;
	if (dm_config_.FLAG_CHANGE_DEPTH_QUANTIZATION == 0)
		dm_config_.num_of_depth = dm_config_.DEFAULT_DEPTH_QUANTIZATION;
	else
		dm_config_.num_of_depth = dm_config_.NUMBER_OF_DEPTH_QUANTIZATION;

	string render_depth;
	next = xml_node->FirstChildElement("RenderDepth");
	if (!next || XML_SUCCESS != next->QueryBoolText(&dm_config_.FLAG_CHANGE_DEPTH_QUANTIZATION))
		return false;
	else render_depth = (xml_node->FirstChildElement("RenderDepth"))->GetText();

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
		LOG("not found Render Depth Parameter\n");
		return false;
	}

	next = xml_node->FirstChildElement("RandomPhase");
	if (!next || XML_SUCCESS != next->QueryBoolText(&dm_config_.RANDOM_PHASE))
		return false;
	next = xml_node->FirstChildElement("FieldLength");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&dm_config_.fieldLength))
		return false;
	next = xml_node->FirstChildElement("NearOfDepth");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&dm_config_.near_depthmap))
		return false;
	next = xml_node->FirstChildElement("FarOfDepth");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&dm_config_.far_depthmap))
		return false;

	auto end = CUR_TIME;
	auto during = ((chrono::duration<Real>)(end - start)).count();
	LOG("%lf (s)...done\n", during);

	initialize();
	return true;
}

/**
* @brief Read image and depth map.
* @details Read input files and load image & depth map data.
*  If the input image size is different with the dislay resolution, resize the image size.
* @param ftr : the frame number of the image.
* @return true if image data are sucessfully read, flase otherwise.
* @see prepare_inputdata_CPU, prepare_inputdata_GPU
*/
bool ophDepthMap::readImageDepth(const char* source_folder, const char* img_prefix, const char* depth_img_prefix)
{
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const int N = pnX * pnY;

	for (int i = 0; i < m_vecRGB.size(); i++)
	{
		delete[] m_vecRGB[i];
	} 
	m_vecRGB.clear();

	// RGB Image
	std::string sdir = source_folder;
	sdir = sdir.append("\\").append(img_prefix).append("*.bmp");

	_finddatai64_t fd;
	intptr_t handle;
	handle = _findfirst64(sdir.c_str(), &fd);
	if (handle == -1)
	{
		LOG("Error: Source image does not exist: %s.\n", sdir.c_str());
		return false;
	}

	std::string imgfullname;
	imgfullname = std::string(source_folder).append("\\").append(fd.name);

	int w, h, bytesperpixel;
	bool ret = getImgSize(w, h, bytesperpixel, imgfullname.c_str());

	// RGB Image
	oph::uchar* buf = new uchar[w * h * bytesperpixel]; // 1-Dimension left top
	ret = loadAsImgUpSideDown(imgfullname.c_str(), buf);
	if (!ret) {
		LOG("Failed::Image Load: %s\n", imgfullname.c_str());
		return false;
	}
	LOG("Succeed::Image Load: %s\n", imgfullname.c_str());

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
	std::string sddir = std::string(source_folder).append("\\").append(depth_img_prefix).append("*.bmp");
	handle = _findfirst64(sddir.c_str(), &fd);
	if (handle == -1)
	{
		LOG("Error: Source depthmap does not exist: %s.\n", sddir);
		return false;
	}

	std::string dimgfullname = std::string(source_folder).append("\\").append(fd.name);

	int dw, dh, dbytesperpixel;
	ret = getImgSize(dw, dh, dbytesperpixel, dimgfullname.c_str());
	
	// Depth Image
	uchar* dbuf = new uchar[dw * dh * dbytesperpixel];
	ret = loadAsImgUpSideDown(dimgfullname.c_str(), dbuf);
	if (!ret) {
		LOG("Failed::Depth Image Load: %s\n", dimgfullname.c_str());
		return false;
	}
	LOG("Succeed::Depth Image Load: %s\n", dimgfullname.c_str());

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

	return true;
}

Real ophDepthMap::generateHologram(void)
{
	resetBuffer();

	m_vecEncodeSize = context_.pixel_number;
	auto begin = CUR_TIME;

	LOG("1) Algorithm Method : Depth Map\n");
	LOG("2) Generate Hologram with %s\n", m_mode & MODE_GPU ?
		"GPU" :
#ifdef _OPENMP
		"Multi Core CPU"
#else
		"Single Core CPU"
#endif
		);

	int nChannel = context_.waveNum;
	if (m_mode & MODE_GPU)
	{
		for (int ch = 0; ch < nChannel; ch++)
		{
			prepareInputdataGPU(m_vecRGB[ch], depth_img);
			getDepthValues();
			//if (is_ViewingWindow)
			//	transVW();
			calcHoloGPU(ch);
		}
	}
	else
	{
		for (int ch = 0; ch < nChannel; ch++)
		{
			prepareInputdataCPU(m_vecRGB[ch], depth_img);
			getDepthValues();
			if (is_ViewingWindow)
				transVW();
			calcHoloCPU(ch);
		}
	}
	
	LOG("Total Elapsed Time: %lf (s)\n", ELAPSED_TIME(begin, CUR_TIME));
	m_nProgress = 0;
	return 0;
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
			ivec2 location;
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


/**
* @brief Initialize variables for CPU and GPU implementation.
* @see initCPU, initGPU
*/
void ophDepthMap::initialize()
{
	dstep = 0;
	dlevel.clear();

	ophGen::initialize();

	initCPU();
	initGPU();
}

/**
* @brief Calculate the physical distances of depth map layers
* @details Initialize 'dstep_' & 'dlevel_' variables.
*  If FLAG_CHANGE_DEPTH_QUANTIZATION == 1, recalculate  'depth_index_' variable.
* @see changeDepthQuanCPU, changeDepthQuanGPU
*/
void ophDepthMap::getDepthValues()
{
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
		dlevel.push_back(dm_config_.far_depthmap - dm_config_.near_depthmap);

	}
	
	bool is_CPU = m_mode & MODE_GPU ? false : true;

	if (dm_config_.FLAG_CHANGE_DEPTH_QUANTIZATION == 1)
	{
		if (is_CPU)
			changeDepthQuanCPU();
		else
			changeDepthQuanGPU();
	}
}

/**
* @brief Transform target object to reflect the system configuration of holographic display.
* @details Calculate 'dlevel_transform_' variable by using 'fieldLength' & 'dlevel_'.
*/
void ophDepthMap::transVW()
{
	Real val;
	dlevel_transform.clear();
	for (int p = 0; p < dlevel.size(); p++)
	{
		val = -dm_config_.fieldLength * dlevel[p] / (dlevel[p] - dm_config_.fieldLength);
		dlevel_transform.push_back(val);
	}
}


/**
* @brief Initialize variables for the CPU implementation.
* @details Memory allocation for the CPU variables.
* @see initialize
*/
void ophDepthMap::initCPU()
{
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint N = pnX * pnY;

	if (img_src)	delete[] img_src;
	img_src = new Real[N];

	if (dmap_src) delete[] dmap_src;
	dmap_src = new Real[N];

	if (alpha_map) delete[] alpha_map;
	alpha_map = new int[N];

	if (depth_index) delete[] depth_index;
	depth_index = new short[N];

	if (dmap) delete[] dmap;
	dmap = new Real[N];

	fftw_cleanup();
}

/**
* @brief Preprocess input image & depth map data for the CPU implementation.
* @details Prepare variables, img_src_, dmap_src_, alpha_map_, depth_index_.
* @param imgptr : input image data pointer
* @param dimgptr : input depth map data pointer
* @return true if input data are sucessfully prepared, flase otherwise.
* @see ReadImageDepth
*/
bool ophDepthMap::prepareInputdataCPU(uchar* imgptr, uchar* dimgptr)
{
	auto begin = CUR_TIME;
	const uint N = context_.pixel_number[_X] * context_.pixel_number[_Y];

	memset(img_src, 0, sizeof(Real) * N);
	memset(dmap_src, 0, sizeof(Real) * N);
	memset(alpha_map, 0, sizeof(int) * N);
	memset(depth_index, 0, sizeof(short) * N);
	memset(dmap, 0, sizeof(Real) * N);

	Real gapDepth = dm_config_.far_depthmap - dm_config_.near_depthmap;
	Real nearDepth = dm_config_.near_depthmap;
	
	int k = 0;
#ifdef _OPENMP
#pragma omp parallel for private(k) firstprivate(gapDepth, nearDepth)
#endif
	for (k = 0; k < N; k++) {
		img_src[k] = Real(imgptr[k]) / 255.0; // RGB IMG
		dmap_src[k] = Real(dimgptr[k]) / 255.0; // DEPTH IMG
		alpha_map[k] = (imgptr[k] > 0 ? 1 : 0); // RGB IMG
		dmap[k] = (1 - dmap_src[k]) * gapDepth + nearDepth;

		if (dm_config_.FLAG_CHANGE_DEPTH_QUANTIZATION == 0) {
			depth_index[k] = dm_config_.DEFAULT_DEPTH_QUANTIZATION - short(dimgptr[k]);
		}
	}

	LOG("\n%s : %lf(s)\n\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
	return true;
}

/**
* @brief Quantize depth map on the CPU, when the number of depth quantization is not the default value (i.e. FLAG_CHANGE_DEPTH_QUANTIZATION == 1 ).
* @details Calculate the value of 'depth_index_'.
* @see GetDepthValues
*/
void ophDepthMap::changeDepthQuanCPU()
{
	auto begin = CUR_TIME;

	const uint N = context_.pixel_number[_X] * context_.pixel_number[_Y];

	uint num_depth = dm_config_.num_of_depth;
	Real near_depth = dm_config_.near_depthmap;
	Real far_depth = dm_config_.far_depthmap;
	
	double nearv = dlevel[0];
	double half_step = dstep / 2.0;
	depth_fill.clear();
	depth_fill.resize(dm_config_.render_depth.size() + 1, 0);
	Real locDstep = dstep;

	int i;
#ifdef _OPENMP
#pragma omp parallel for private(i) firstprivate(nearv, half_step, locDstep)
#endif
	for (i = 0; i < N; i++)
	{
		int idx = int(((dmap[i] - nearv) + half_step) / locDstep);
		depth_index[i] = idx + 1;
		depth_fill[idx + 1] = 1;
	}
	LOG("\n%s : %lf(s)\n\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
}

/**
* @brief Main method for generating a hologram on the CPU.
* @details For each depth level,
*   1. find each depth plane of the input image.
*   2. apply carrier phase delay.
*   3. propagate it to the hologram plan.
*   4. accumulate the result of each propagation.
* .
* The final result is accumulated in the variable 'U_complex_'.
* @param frame : the frame number of the image.
* @see Calc_Holo_by_Depth, Propagation_AngularSpectrum_CPU
*/
void ophDepthMap::calcHoloCPU(int ch)
{
	auto begin = CUR_TIME;

	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint N = pnX * pnY;
	const int nChannel = context_.waveNum;

	size_t depth_sz = dm_config_.render_depth.size();

	OphDepthMapConfig cfg = dm_config_;
	Complex<Real> *in = nullptr, *out = nullptr;
	fft2(ivec2(pnX, pnY), in, OPH_FORWARD, OPH_ESTIMATE);

	int frame = 0;

	//for (int ch = 0; ch < nChannel; ch++) {
		Real lambda = context_.wave_length[ch];
		Real k = context_.k = (2 * M_PI / lambda);

		int i;
#ifdef _OPENMP
//#pragma omp parallel for private(i) firstprivate(k, lambda, N, cfg, ch, depth_sz)
#endif
		for (i = 0; i < depth_sz; i++) {
			int dtr = dm_config_.render_depth[i];
			if (!depth_fill[dtr]) continue;

			Real temp_depth = (is_ViewingWindow) ? dlevel_transform[dtr - 1] : dlevel[dtr - 1];

			Complex<Real> *input = new Complex<Real>[N];
			memset(input, 0.0, sizeof(Complex<Real>) * N);

			for (int j = 0; j < N; j++)
			{
				input[j][_RE] = img_src[j] * alpha_map[j] * (depth_index[j] == dtr ? 1.0 : 0.0);
			}

			LOG("Frame#: %d, Depth: %d of %d, z = %f mm\n", frame, dtr, dm_config_.num_of_depth, -temp_depth * 1000);

			Complex<Real> rand_phase_val;
			getRandPhaseValue(rand_phase_val, dm_config_.RANDOM_PHASE);

			Complex<Real> carrier_phase_delay(0, k * temp_depth);
			carrier_phase_delay.exp();

			for (int j = 0; j < N; j++)
			{
				input[j] *= rand_phase_val * carrier_phase_delay;
			}

			fft2(input, input, pnX, pnY, OPH_FORWARD, false);
			propagationAngularSpectrum(ch, input, -temp_depth, k, lambda);
			
			delete[] input;
			m_nProgress = (int)((Real)(ch * depth_sz + i) * 100 / (depth_sz * nChannel));
		}
	//}
	fftFree();
	LOG("\n%s : %lf(s)\n\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));

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
	// 해상도 변경이 있을 시, 버퍼를 재생성 한다.
	if (context_.pixel_number != resolution) {
		ophGen::setResolution(resolution);
		initCPU();
		initGPU();
	}
}

void ophDepthMap::normalize()
{
#if 1
	ophGen::normalize();
#else
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];

	for (uint ch = 0; ch < context_.waveNum; ch++)
		oph::normalize((Real*)m_lpEncoded[ch], m_lpNormalized[ch], pnX, pnY);
#endif
}