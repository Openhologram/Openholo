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
	if (!ophGen::readConfig(fname, dm_config_))
		return false;

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

	// 2019-10-14 mwnam
	m_vecRGBImg[_X] = w;
	m_vecRGBImg[_Y] = h;

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
	LOG("<1> getDepthValues %lf / %lf\n", (*complex_H)[0][_RE], (*complex_H)[0][_IM]);
	if(is_ViewingWindow)
		transVW();
	calcHoloByDepth();
	LOG("<2> calcHoloByDepth %lf / %lf\n", (*complex_H)[0][_RE], (*complex_H)[0][_IM]);

	auto end_time = CUR_TIME;

	auto during_time = ((std::chrono::duration<Real>)(end_time - start_time)).count();

	LOG("Implement time : %.5lf sec\n", during_time);
#ifdef TEST_MODE
	HWND hwndNotepad = NULL;
	hwndNotepad = ::FindWindow(NULL, "test.txt - ¸Þ¸ðÀå");
	if (hwndNotepad) {
		hwndNotepad = FindWindowEx(hwndNotepad, NULL, "edit", NULL);

		char *pBuf = NULL;
		int nLen = SendMessage(hwndNotepad, WM_GETTEXTLENGTH, 0, 0);
		pBuf = new char[nLen + 100];
		
		SendMessage(hwndNotepad, WM_GETTEXT, nLen+1, (LPARAM)pBuf);
		//sprintf(pBuf, "%s%.5lf\r\n", pBuf, during_time);
		sprintf(pBuf, "%s : RE: %.5lf / IM: %.5lf\r\n",
			is_CPU ? "CPU" : "GPU", (*complex_H)[0][0], (*complex_H)[0][1]);

		SendMessage(hwndNotepad, WM_SETTEXT, 0, (LPARAM)pBuf);
		delete[] pBuf;
	}
#endif

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

		encodeSideBand(is_CPU, ivec2(0, 1));
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
	int px = pnx;//static_cast<int>(pnx / 3);
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
	uint pX = context_.pixel_number[_X];
	uint pY = context_.pixel_number[_Y];

	if (img_src)	delete[] img_src;
	img_src = new Real[pX * pY];

	if (dmap_src) delete[] dmap_src;
	dmap_src = new Real[pX * pY];

	if (alpha_map) delete[] alpha_map;
	alpha_map = new int[pX * pY];

	if (depth_index) delete[] depth_index;
	depth_index = new Real[pX * pY];

	if (dmap) delete[] dmap;
	dmap = new Real[pX * pY];

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
	int pnx = context_.pixel_number[_X];
	int pny = context_.pixel_number[_Y];

	memset(img_src, 0, sizeof(Real)*pnx * pny);
	memset(dmap_src, 0, sizeof(Real)*pnx * pny);
	memset(alpha_map, 0, sizeof(int)*pnx * pny);
	memset(depth_index, 0, sizeof(Real)*pnx * pny);
	memset(dmap, 0, sizeof(Real)*pnx * pny);

	int k = 0;
#pragma omp parallel for private(k)
	for (k = 0; k < pnx*pny; k++)
	{
		img_src[k] = Real(imgptr[k]) / 255.0;
		dmap_src[k] = Real(dimgptr[k]) / 255.0;
		alpha_map[k] = (imgptr[k] > 0 ? 1 : 0);
		dmap[k] = (1 - dmap_src[k])*(dm_config_.far_depthmap - dm_config_.near_depthmap) + dm_config_.near_depthmap;

		if (dm_config_.FLAG_CHANGE_DEPTH_QUANTIZATION == 0)
			depth_index[k] = dm_config_.DEFAULT_DEPTH_QUANTIZATION - Real(dimgptr[k]);
	}
}

/**
* @brief Quantize depth map on the CPU, when the number of depth quantization is not the default value (i.e. FLAG_CHANGE_DEPTH_QUANTIZATION == 1 ).
* @details Calculate the value of 'depth_index_'.
* @see GetDepthValues
*/
void ophDepthMap::changeDepthQuanCPU()
{
	int pnx = context_.pixel_number[_X];
	int pny = context_.pixel_number[_Y];

	Real temp_depth, d1, d2;
	//int dtr;
	int tdepth;


	for (uint dtr = 0; dtr < dm_config_.num_of_depth; ++dtr)
	{
		temp_depth = dlevel[dtr];
		d1 = temp_depth - dstep / 2.0;
		d2 = temp_depth + dstep / 2.0;

		int p;
#pragma omp	parallel for private(p)
		for (p = 0; p < pnx * pny; ++p)
		{
#if 1
			if (dtr < dm_config_.num_of_depth - 1)
				tdepth = (dmap[p] >= d1 ? 1 : 0) * (dmap[p] < d2 ? 1 : 0);
			else
				tdepth = (dmap[p] >= d1 ? 1 : 0) * (dmap[p] <= d2 ? 1 : 0);
#else
			int nVal = (dtr < dm_config_.num_of_depth - 1) ? 1 : 0;
			int tdepth = (dmap[p] >= d1 ? 1 : 0) * (dmap[p] <= d2 - nVal ? 1 : 0);
#endif
			depth_index[p] += tdepth * (dtr + 1);
		}
	}
	//writeIntensity_gray8_bmp("test.bmp", pnx, pny, depth_index_);
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
	int pnx = context_.pixel_number[_X];
	int pny = context_.pixel_number[_Y];

	//memset((*complex_H), 0.0, sizeof(Complex<Real>)*pnx*pny);
	size_t depth_sz = dm_config_.render_depth.size();

	Complex<Real> *in = nullptr, *out = nullptr;
	fft2(ivec2(pnx, pny), in, OPH_FORWARD, OPH_ESTIMATE);

	int p = 0;
#ifdef _OPENMP
#pragma omp parallel 
	{
		int tid = omp_get_thread_num();
#pragma omp for private(p)
#endif
		for (p = 0; p < depth_sz; ++p)
		{
			int dtr = dm_config_.render_depth[p];
			Real temp_depth = (is_ViewingWindow) ? dlevel_transform[dtr - 1] : dlevel[dtr - 1];

			Complex<Real>* u_o = (Complex<Real>*)malloc(sizeof(Complex<Real>)*pnx*pny);
			memset(u_o, 0.0, sizeof(Complex<Real>)*pnx*pny);

			Real sum = 0.0;
			for (int i = 0; i < pnx * pny; i++)
			{
				u_o[i]._Val[_RE] = img_src[i] * alpha_map[i] * (depth_index[i] == dtr ? 1.0 : 0.0);
				sum += u_o[i]._Val[_RE];
			}

			if (sum > 0.0)
			{
				//LOG("Depth: %d of %d, z = %f mm\n", dtr, dm_config_.num_of_depth, -temp_depth * 1000);

				Complex<Real> rand_phase_val;
				getRandPhaseValue(rand_phase_val, dm_config_.RANDOM_PHASE);

				Complex<Real> carrier_phase_delay(0, context_.k* temp_depth);
				carrier_phase_delay.exp();

				for (int i = 0; i < pnx * pny; i++)
					u_o[i] = u_o[i] * rand_phase_val * carrier_phase_delay;

				//if (dm_params_.Propagation_Method_ == 0) {
				Openholo::fftwShift(u_o, u_o, pnx, pny, OPH_FORWARD, false);
				propagationAngularSpectrum(u_o, -temp_depth);
				//}
			}
			else {
				//LOG("Depth: %d of %d : Nothing here\n", dtr, dm_config_.num_of_depth);
			}

			free(u_o);
		}
	}
}

void ophDepthMap::ophFree(void)
{

}