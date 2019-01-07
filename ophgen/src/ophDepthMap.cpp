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

	// CPU Variables
	img_src = 0;
	dmap_src = 0;
	alpha_map = 0;
	depth_index = 0;
	dmap = 0;
	dstep = 0;
	dlevel.clear();
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

/**
* @brief Read parameters from a config file(config_openholo.txt).
* @return true if config infomation are sucessfully read, flase otherwise.
*/
bool ophDepthMap::readConfig(const char * fname)
{
	if (!ophGen::readConfig(fname, dm_config_))
		return false;

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

	uchar* dimg = new uchar[dw*dh];
	convertToFormatGray8(dimgload, dimg, dw, dh, dbytesperpixel);

	delete[] dimgload;

	//resize image
	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];

	rgb_img = new uchar[pnx*pny];
	memset(rgb_img, 0, sizeof(char)*pnx*pny);

	if (w != pnx || h != pny)
		imgScaleBilnear(img, rgb_img, w, h, pnx, pny);
	else
		memcpy(rgb_img, img, sizeof(char)*pnx*pny);

	//ret = creatBitmapFile(newimg, pnx, pny, 8, "stest");

	depth_img = new uchar[pnx*pny];
	memset(depth_img, 0, sizeof(char)*pnx*pny);

	if (dw != pnx || dh != pny)
		imgScaleBilnear(dimg, depth_img, dw, dh, pnx, pny);
	else
		memcpy(depth_img, dimg, sizeof(char)*pnx*pny);

	delete[] img;
	delete[] dimg;

	return true;
}

Real ophDepthMap::generateHologram(void)
{
	initialize();

	encode_size = context_.pixel_number;

	auto start_time = CUR_TIME;

	if (is_CPU)
		prepareInputdataCPU(rgb_img, depth_img);
	else
		prepareInputdataGPU(rgb_img, depth_img);

	getDepthValues();
	transformViewingWindow();
	calcHoloByDepth();

	auto end_time = CUR_TIME;

	auto during_time = ((std::chrono::duration<Real>)(end_time - start_time)).count();

	LOG("Implement time : %.5lf sec\n", during_time);

	return during_time;
}

void ophDepthMap::encodeHologram(void)
{
	LOG("Single Side Band Encoding..");
	encodeSideBand(is_CPU, ivec2(0, 1));
	LOG("Done.\n.");
}

void ophDepthMap::encoding(unsigned int ENCODE_FLAG)
{
	fft2(context_.pixel_number, *complex_H, OPH_BACKWARD);
	Complex<Real>* dst = new Complex<Real>[context_.pixel_number[_X] * context_.pixel_number[_Y]];
	fftwShift(*complex_H, dst, context_.pixel_number[_X], context_.pixel_number[_Y], OPH_BACKWARD);

	ophGen::encoding(ENCODE_FLAG, dst);

	delete[] dst;
}

void ophDepthMap::encoding(unsigned int ENCODE_FLAG, unsigned int SSB_PASSBAND)
{
	fft2(context_.pixel_number, *complex_H, OPH_BACKWARD);
	Complex<Real>* dst = new Complex<Real>[context_.pixel_number[_X] * context_.pixel_number[_Y]];
	fftwShift(*complex_H, dst, context_.pixel_number[_X], context_.pixel_number[_Y], OPH_BACKWARD);

	
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

	delete[] dst;
}

int ophDepthMap::save(const char * fname, uint8_t bitsperpixel)
{
	std::string resName = std::string(fname);
	if (resName.find(".bmp") == std::string::npos && resName.find(".BMP") == std::string::npos) resName.append(".bmp");

	int pnx = context_.pixel_number[_X];
	int pny = context_.pixel_number[_Y];
	int px = static_cast<int>(pnx / 3);
	int py = pny;

	ophGen::save(resName.c_str(), bitsperpixel, holo_normalized, px, py);

	return 1;
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

	if (is_CPU)
		initCPU();
	else
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
* @details Calculate 'dlevel_transform_' variable by using 'field_lens' & 'dlevel_'.
*/
void ophDepthMap::transformViewingWindow()
{
	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];

	Real val;
	dlevel_transform.clear();
	for (int p = 0; p < dlevel.size(); p++)
	{
		val = -dm_config_.field_lens * dlevel[p] / (dlevel[p] - dm_config_.field_lens);
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

void ophDepthMap::ophFree(void)
{

}