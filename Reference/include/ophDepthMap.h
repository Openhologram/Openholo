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

#ifndef __ophDepthMap_h
#define __ophDepthMap_h

#include "ophGen.h"
#include <cufft.h>
#include "include.h"

//Build Option : Multi Core Processing (OpenMP)
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace oph;


/**
* @addtogroup depthmap
//@{
* @details

* @section Introduction

This module is related methods which generates CGH based on depth map. It is supported single core
processing, multi-core processing(with OpenMP) and GPGPU parallel processing(with CUDA).

I. Depth Map Hologram Generation

-   Implement the hologram generation method using depth map data.
-   Improve the performance of the hologram generation method.
-   Implemented on CPU and GPU.
-   The original algorithm is modified in the way that can be easily implemented in parallel.

![](pics/ophgen/depthmap/gen_depthmap01.png)

![](pics/ophgen/depthmap/depth_slice_image01.png)

II. Algorithm

-   Propagate from the previous depth plane to the current depth plane.
-   At the last plane, back propagate to the hologram plane.

![](pics/ophgen/depthmap/gen_depthmap_flowchart02.png)

![](pics/ophgen/depthmap/depth_slice_image02.png)

III. Modified Algorithm

-   Back propagate each depth plane to the hologram plane.
-   Accumulate the results of each propagation.

![](pics/ophgen/depthmap/gen_depthmap_flowchart03.png)

![](pics/ophgen/depthmap/depth_slice_image03.png)


*/
//! @} depthmap



/**
* @ingroup depthmap
* @brief This class generates CGH based on depth map.
* @author
*/
class GEN_DLL ophDepthMap : public ophGen {

public:
	explicit ophDepthMap();

protected:
	virtual ~ophDepthMap();

public:
	/**
	* @brief Set the value of a variable is_CPU(true or false)
	* @details <pre>
	if is_CPU == true
	CPU implementation
	else
	GPU implementation </pre>
	* @param[in] is_CPU the value for specifying whether the hologram generation method is implemented on the CPU or GPU
	*/
	void setMode(bool is_CPU);

	/**
	* @brief Function for setting precision
	* @param[in] precision level.
	*/
	void setPrecision(bool bPrecision) { bSinglePrecision = bPrecision; }
	bool getPrecision() { return bSinglePrecision; }

	bool readConfig(const char* fname);
	bool readImageDepth(const char* source_folder, const char* img_prefix, const char* depth_img_prefix);
	//bool readImageDepth(const char* rgb, const char* depth);
	
	/**
	* @brief Generate a hologram, main funtion. When the calculation is finished, the angular spectrum is performed.
	* @return implement time (sec)
	*/
	Real generateHologram(void);

	virtual void encoding(unsigned int ENCODE_FLAG);
	virtual void encoding(unsigned int ENCODE_FLAG, unsigned int SSB_PASSBAND);
	
	/**
	* @brief Set the value of a variable is_ViewingWindow(true or false)
	* @details <pre>
	if is_ViewingWindow == true
	Transform viewing window
	else
	Hologram </pre>
	* @param is_ViewingWindow : the value for specifying whether the hologram generation method is implemented on the viewing window
	*/
	void setViewingWindow(bool is_ViewingWindow);

	ivec2 getRGBImgSize() { return m_vecRGBImg; };
	ivec2 getDepthImgSize() { return m_vecDepthImg; };

	void setConfig(OphDepthMapConfig config) {
		dm_config_ = config;
	};
	void setResolution(ivec2 resolution);
	uint* getProgress() { return &m_nProgress; }


	void normalize();

public:
	inline void setFieldLens(Real fieldlens) { dm_config_.fieldLength = fieldlens; }
	inline void setNearDepth(Real neardepth) { dm_config_.near_depthmap = neardepth; }
	inline void setFarDepth(Real fardetph) { dm_config_.far_depthmap = fardetph; }
	inline void setNumOfDepth(uint numofdepth) { dm_config_.num_of_depth = numofdepth; }

	inline Real getFieldLens(void) { return dm_config_.fieldLength; }
	inline Real getNearDepth(void) { return dm_config_.near_depthmap; }
	inline Real getFarDepth(void) { return dm_config_.far_depthmap; }
	inline uint getNumOfDepth(void) { return dm_config_.num_of_depth; }
	inline void getRenderDepth(std::vector<int>& renderdepth) { renderdepth = dm_config_.render_depth; }

	inline const OphDepthMapConfig& getConfig() { return dm_config_; }
	
private:

	void initialize();
	void initCPU();
	void initGPU();

	bool prepareInputdataCPU(uchar* img, uchar* dimg);
	bool prepareInputdataGPU(uchar* img, uchar* dimg);

	void getDepthValues();
	void changeDepthQuanCPU();
	void changeDepthQuanGPU();

	void transVW();

	void calcHoloCPU(int ch = 0);
	void calcHoloGPU(int ch = 0);
	void propagationAngularSpectrumGPU(uint channel, cufftDoubleComplex* input_u, Real propagation_dist);

protected:
	void free_gpu(void);

	void ophFree(void);

private:
	bool					is_CPU;								///< if true, it is implemented on the CPU, otherwise on the GPU.
	bool					is_ViewingWindow;
	bool					bSinglePrecision;
	unsigned char*			depth_img;
	vector<uchar*>			m_vecRGB;
	ivec2					m_vecRGBImg;
	ivec2					m_vecDepthImg;
	unsigned char*			img_src_gpu;						///< GPU variable - image source data, values are from 0 to 255.
	unsigned char*			dimg_src_gpu;						///< GPU variable - depth map data, values are from 0 to 255.
	Real*					depth_index_gpu;					///< GPU variable - quantized depth map data.

	Real*					img_src;							///< CPU variable - image source data, values are from 0 to 1.
	Real*					dmap_src;							///< CPU variable - depth map data, values are from 0 to 1.
	short*					depth_index;						///< CPU variable - quantized depth map data.
	int*					alpha_map;							///< CPU variable - calculated alpha map data, values are 0 or 1.
	vector<short>			depth_fill;
	Real*					dmap;								///< CPU variable - physical distances of depth map.

	Real					dstep;								///< the physical increment of each depth map layer.
	vector<Real>			dlevel;								///< the physical value of all depth map layer.
	vector<Real>			dlevel_transform;					///< transfomed dlevel variable

	OphDepthMapConfig		dm_config_;							///< structure variable for depthmap hologram configuration.


	uint m_nProgress;
};

#endif //>__ophDepthMap_h