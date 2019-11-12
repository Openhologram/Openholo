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

#include	<windows.h>
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
{
	is_CPU = true;

	// GPU Variables
	img_src_gpu = 0;
	dimg_src_gpu = 0;
	depth_index_gpu = 0;

	depth_img = nullptr;
	rgb_img = nullptr;

	// CPU Variables
	img_src = 0;
	dmap_src = 0;
	alpha_map = 0;
	depth_index = 0;
	dmap = 0;
	dstep = 0;
	dlevel.clear();
	setViewingWindow(FALSE);
}

/**
* @brief Destructor 
*/
ophDepthMap::~ophDepthMap()
{
}

/**
* @brief Set the value of a variable isCPU_(true or false)
* @details <pre>
    if isCPU_ == true
	   CPU implementation
	else
	   GPU implementation </pre>
* @param isCPU : the value for specifying whether the hologram generation method is implemented on the CPU or GPU
*/
void ophDepthMap::setMode(bool isCPU) 
{ 
	is_CPU = isCPU;
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
	int ret = getImgSize(w, h, bytesperpixel, imgfullname.c_str());

	oph::uchar* imgload = new uchar[w*h*bytesperpixel];
	ret = loadAsImgUpSideDown(imgfullname.c_str(), imgload);
	if (!ret) {
		LOG("Failed::Image Load: %s\n", imgfullname.c_str());
		return false;
	}
	LOG("Succeed::Image Load: %s\n", imgfullname.c_str());

	oph::uchar* img = new uchar[w*h];
	convertToFormatGray8(imgload, img, w, h, bytesperpixel);

	delete[] imgload;


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
	
	uchar* dimgload = new uchar[dw*dh*dbytesperpixel];
	ret = loadAsImgUpSideDown(dimgfullname.c_str(), dimgload);
	if (!ret) {
		LOG("Failed::Depth Image Load: %s\n", dimgfullname.c_str());
		return false;
	}
	LOG("Succeed::Depth Image Load: %s\n", dimgfullname.c_str());

	// 2019-10-14 mwnam
	m_vecDepthImg[_X] = dw;
	m_vecDepthImg[_Y] = dh;

	uchar* dimg = new uchar[dw*dh];
	convertToFormatGray8(dimgload, dimg, dw, dh, dbytesperpixel);

	delete[] dimgload;

	//resize image
	int pnX = context_.pixel_number[_X];
	int pnY = context_.pixel_number[_Y];

	if (rgb_img) delete[] rgb_img;

	rgb_img = new uchar[pnX*pnY];
	memset(rgb_img, 0, sizeof(char)*pnX*pnY);

	if (w != pnX || h != pnY)
		imgScaleBilnear(img, rgb_img, w, h, pnX, pnY);
	else
		memcpy(rgb_img, img, sizeof(char)*pnX*pnY);

	// 2019-10-14 mwnam
	m_vecRGBImg[_X] = pnX;
	m_vecRGBImg[_Y] = pnY;

	//ret = creatBitmapFile(newimg, pnX, pnY, 8, "stest");
	if (depth_img) delete[] depth_img;

	depth_img = new uchar[pnX*pnY];
	memset(depth_img, 0, sizeof(char)*pnX*pnY);

	if (dw != pnX || dh != pnY)
		imgScaleBilnear(dimg, depth_img, dw, dh, pnX, pnY);
	else
		memcpy(depth_img, dimg, sizeof(char)*pnX*pnY);
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

	encode_size = context_.pixel_number;

	MEMORYSTATUS memStatus;
	GlobalMemoryStatus(&memStatus);
	LOG("\n*Available Memory: %u (byte)\n", memStatus.dwAvailVirtual);

	auto start_time = CUR_TIME;

	LOG("1) Algorithm Method : Depth Map\n");
	LOG("2) Generate Hologram with %s\n", is_CPU ?
#ifdef _OPENMP
		"Multi Core CPU" :
#else
		"Single Core CPU" :
#endif
		"GPU");
	LOG("3) Transform Viewing Window : %s\n", is_ViewingWindow ? "ON" : "OFF");

	if (is_CPU)
		prepareInputdataCPU(rgb_img, depth_img);
	else
		prepareInputdataGPU(rgb_img, depth_img);
	getDepthValues();
	if(is_ViewingWindow)
		transVW();
	calcHoloByDepth();

	auto end_time = CUR_TIME;

	elapsedTime = ((std::chrono::duration<Real>)(end_time - start_time)).count();

	LOG("Total Elapsed Time: %lf (s)\n", elapsedTime);

	return elapsedTime;
}

void ophDepthMap::encodeHologram(void)
{
	LOG("Single Side Band Encoding..");
	encodeSideBand(is_CPU, ivec2(0, 1));
	LOG("Done.\n.");
}

void ophDepthMap::encoding(unsigned int ENCODE_FLAG)
{
	ophGen::encoding(ENCODE_FLAG, nullptr, true);
}

void ophDepthMap::encoding(unsigned int ENCODE_FLAG, unsigned int SSB_PASSBAND)
{
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint nChannel = context_.waveNum;
	Complex<Real>* dst = new Complex<Real>[pnX * pnY];

	for (uint ch = 0; ch < nChannel; ch++) {
		fft2(context_.pixel_number, complex_H[ch], OPH_BACKWARD);
		fftwShift(complex_H[ch], dst, pnX, pnY, OPH_BACKWARD);

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
		else ophGen::encoding(ENCODE_FLAG, SSB_PASSBAND, dst);
	}
	delete[] dst;
}


/**
* @brief Initialize variables for CPU and GPU implementation.
* @see init_CPU, init_GPU
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
* @see change_depth_quan_CPU, change_depth_quan_GPU
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

	} else {

		dstep = (dm_config_.far_depthmap + dm_config_.near_depthmap) / 2;
		dlevel.push_back(dm_config_.far_depthmap - dm_config_.near_depthmap);

	}
	
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
* @brief Generate a hologram.
* @param frame : the frame number of the image.
* @see Calc_Holo_CPU, Calc_Holo_GPU
*/
void ophDepthMap::calcHoloByDepth()
{
	if (is_CPU)
		calcHoloCPU();
	else
		calcHoloGPU();
	
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
	const uint pnXY = pnX * pnY;

	if (img_src)	delete[] img_src;
	img_src = new Real[pnXY];

	if (dmap_src) delete[] dmap_src;
	dmap_src = new Real[pnXY];

	if (alpha_map) delete[] alpha_map;
	alpha_map = new int[pnXY];

	if (depth_index) delete[] depth_index;
	depth_index = new Real[pnXY];

	if (dmap) delete[] dmap;
	dmap = new Real[pnXY];

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
#ifdef CHECK_PROC_TIME
	auto begin = CUR_TIME;
#endif
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;

	memset(img_src, 0, sizeof(Real) * pnXY);
	memset(dmap_src, 0, sizeof(Real) * pnXY);
	memset(alpha_map, 0, sizeof(int) * pnXY);
	memset(depth_index, 0, sizeof(Real) * pnXY);
	memset(dmap, 0, sizeof(Real) * pnXY);

	Real nearDepth = dm_config_.near_depthmap;

	int k = 0;
#ifdef _OPENMP
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
#pragma omp for private(k)
#endif
		for (k = 0; k < pnXY; k++)	{
			Real rgbVal = Real(imgptr[k]) / 255.0;
			img_src[k] = rgbVal;
			Real dVal = Real(dimgptr[k]) / 255.0;
			dmap_src[k] = dVal;
			int alphaVal = (imgptr[k] > 0 ? 1 : 0);
			alpha_map[k] = alphaVal;			

			if (dm_config_.FLAG_CHANGE_DEPTH_QUANTIZATION == 0) {
				Real imgVal = Real(dimgptr[k]);
				depth_index[k] = dm_config_.DEFAULT_DEPTH_QUANTIZATION - imgVal;
			}
		}
#ifdef _OPENMP
	}
#endif
#ifdef CHECK_PROC_TIME
	auto end = CUR_TIME;
	LOG("\n%s : %lf(s)\n\n", __FUNCTION__, ((std::chrono::duration<Real>)(end - begin)).count());
#endif
	return true;
}

/**
* @brief Quantize depth map on the CPU, when the number of depth quantization is not the default value (i.e. FLAG_CHANGE_DEPTH_QUANTIZATION == 1 ).
* @details Calculate the value of 'depth_index_'.
* @see GetDepthValues
*/
void ophDepthMap::changeDepthQuanCPU()
{
#ifdef CHECK_PROC_TIME
	auto begin = CUR_TIME;
#endif
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;

	Real temp_depth, d1, d2;

	//int tdepth;

	for (uint dtr = 0; dtr < dm_config_.num_of_depth; ++dtr) {
		temp_depth = dlevel[dtr];
		d1 = temp_depth - dstep / 2.0;
		d2 = temp_depth + dstep / 2.0;

		int p;
#ifdef _OPENMP
#pragma omp	parallel
		{
			int tid = omp_get_thread_num();
#pragma omp for private(p)
#endif
			for (p = 0; p < pnXY; ++p) {
				Real dmap = (1.0 - dmap_src[p])*(dm_config_.far_depthmap - dm_config_.near_depthmap) + dm_config_.near_depthmap;

				int tdepth;
				if (dtr < dm_config_.num_of_depth - 1)
					tdepth = (dmap >= d1 ? 1 : 0) * (dmap < d2 ? 1 : 0);
				else
					tdepth = (dmap >= d1 ? 1 : 0) * (dmap <= d2 ? 1 : 0);
				depth_index[p] += tdepth * (dtr + 1);
			}
#ifdef _OPENMP
		}
#endif
	}
#ifdef CHECK_PROC_TIME
	auto end = CUR_TIME;
	LOG("\n%s : %lf(s)\n\n", __FUNCTION__, ((std::chrono::duration<Real>)(end - begin)).count());
#endif
	//writeIntensity_gray8_bmp("test.bmp", pnX, pnY, depth_index_);
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
void ophDepthMap::calcHoloCPU()
{
#ifdef CHECK_PROC_TIME
	auto begin = CUR_TIME;
#endif
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;
	const int nChannel = context_.waveNum;

	size_t depth_sz = dm_config_.render_depth.size();

	Complex<Real> *in = nullptr, *out = nullptr;
	fft2(ivec2(pnX, pnY), in, OPH_FORWARD, OPH_ESTIMATE);
	int p = 0;

	for (int ch = 0; ch < nChannel; ch++) {
		Real lambda = context_.wave_length[ch];
		Real k = context_.k = (2 * M_PI / lambda);

#ifdef _OPENMP
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
#pragma omp for private(p)
#endif
			for (p = 0; p < depth_sz; ++p) {
				int dtr = dm_config_.render_depth[p];

				Real temp_depth = (is_ViewingWindow) ? dlevel_transform[dtr - 1] : dlevel[dtr - 1];

				Complex<Real> *input = new Complex<Real>[pnXY];
				memset(input, 0.0, sizeof(Complex<Real>) * pnXY);
				//Complex<Real> *output = new Complex<Real>[pnXY];
				//memset(output, 0.0, sizeof(Complex<Real>) * pnXY);

				Real sum = 0.0;
				for (int i = 0; i < pnXY; i++)
				{
					input[i][_RE] += img_src[i] * alpha_map[i] * (depth_index[i] == dtr ? 1.0 : 0.0);
					sum += input[i][_RE];
				}

				if (sum > 0.0)
				{
					//LOG("Depth: %d of %d, z = %f mm\n", dtr, dm_config_.num_of_depth, -temp_depth * 1000);
					Complex<Real> rand_phase_val;
					getRandPhaseValue(rand_phase_val, dm_config_.RANDOM_PHASE);

					Complex<Real> carrier_phase_delay(0, k * temp_depth);
					carrier_phase_delay.exp();

					for (int i = 0; i < pnXY; i++)
						input[i] = input[i] * rand_phase_val * carrier_phase_delay;
									
					Openholo::fftwShift(input, input, pnX, pnY, OPH_FORWARD, false);
					propagationAngularSpectrum(ch, input, -temp_depth, k, lambda);					
				}
				else {
					//LOG("Depth: %d of %d : Nothing here\n", dtr, dm_config_.num_of_depth);
				}
				delete[] input;
			}
#ifdef _OPENMP
		}
#endif

		LOG("\n%s (%d/%d) : %lf(s)\n\n", __FUNCTION__, ch+1, nChannel, ((std::chrono::duration<Real>)(CUR_TIME - begin)).count());
	}
#ifdef CHECK_PROC_TIME
	auto end = CUR_TIME;
	LOG("\n%s : %lf(s)\n\n", __FUNCTION__, ((std::chrono::duration<Real>)(end - begin)).count());
#endif
}

void ophDepthMap::ophFree(void)
{
	if(depth_img) delete[] depth_img;
	if(rgb_img) delete[] rgb_img;
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