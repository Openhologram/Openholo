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

using namespace oph;


/**
* @addtogroup depthmap
//@{
* @detail

* @section Introduction

This module is related methods which generates CGH based on depth map. It is supported single core
processing, multi-core processing(with OpenMP) and GPGPU parallel processing(with CUDA).

I. Depth Map Hologram Generation

-   Implement the hologram generation method using depth map data.
-   Improve the performance of the hologram generation method.
-   Implemented on CPU and GPU.
-   The original algorithm is modified in the way that can be easily implemented in parallel.

![](@ref pics/ophgen/depthmap/gen_depthmap01.png)

![](@ref pics/ophgen/depthmap/depth_slice_image01.png)

II. Algorithm

-   Propagate from the previous depth plane to the current depth plane.
-   At the last plane, back propagate to the hologram plane.

![](@ref pics/ophgen/depthmap/gen_depthmap_flowchart02.png)

![](@ref pics/ophgen/depthmap/depth_slice_image02.png)

III. Modified Algorithm

-   Back propagate each depth plane to the hologram plane.
-   Accumulate the results of each propagation.

![](@ref pics/ophgen/depthmap/gen_depthmap_flowchart03.png)

![](@ref pics/ophgen/depthmap/depth_slice_image03.png)


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

	void setMode(bool is_CPU);
	bool readConfig(const char* fname);
	bool readImageDepth(const char* source_folder, const char* img_prefix, const char* depth_img_prefix);

	Real generateHologram(void);

	void encodeHologram(void);

	virtual int save(const char* fname, uint8_t bitsperpixel = 24);

public:
	inline void setFieldLens(Real fieldlens) { dm_config_.field_lens = fieldlens; }
	inline void setNearDepth(Real neardepth) { dm_config_.near_depthmap = neardepth; }
	inline void setFarDepth(Real fardetph) { dm_config_.far_depthmap = fardetph; }
	inline void setNumOfDepth(uint numofdepth) { dm_config_.num_of_depth = numofdepth; }

	inline Real getFieldLens(void) { return dm_config_.field_lens; }
	inline Real getNearDepth(void) { return dm_config_.near_depthmap; }
	inline Real getFarDepth(void) { return dm_config_.far_depthmap; }
	inline uint getNumOfDepth(void) { return dm_config_.num_of_depth; }
	inline void getRenderDepth(std::vector<int>& renderdepth) { renderdepth = dm_config_.render_depth; }
	inline const OphDepthMapConfig& getConfig(void) { return dm_config_; }
	
private:

	void initialize();
	void initCPU();   
	void initGPU();

	bool prepareInputdataCPU(uchar* img, uchar* dimg);
	bool prepareInputdataGPU(uchar* img, uchar* dimg);

	void getDepthValues();
	void changeDepthQuanCPU();
	void changeDepthQuanGPU();

	void transformViewingWindow();

	void calcHoloByDepth(void);
	void calcHoloCPU(void);
	void calcHoloGPU(void);
	void propagationAngularSpectrumGPU(cufftDoubleComplex* input_u, Real propagation_dist);

protected:
	void free_gpu(void);

	void ophFree(void);

private:
	bool					is_CPU;								///< if true, it is implemented on the CPU, otherwise on the GPU.

	unsigned char*			depth_img;
	unsigned char*			rgb_img;

	unsigned char*			img_src_gpu;						///< GPU variable - image source data, values are from 0 to 255.
	unsigned char*			dimg_src_gpu;						///< GPU variable - depth map data, values are from 0 to 255.
	Real*					depth_index_gpu;					///< GPU variable - quantized depth map data.

	Real*					img_src;							///< CPU variable - image source data, values are from 0 to 1.
	Real*					dmap_src;							///< CPU variable - depth map data, values are from 0 to 1.
	Real*					depth_index;						///< CPU variable - quantized depth map data.
	int*					alpha_map;							///< CPU variable - calculated alpha map data, values are 0 or 1.

	Real*					dmap;								///< CPU variable - physical distances of depth map.

	Real					dstep;								///< the physical increment of each depth map layer.
	std::vector<Real>		dlevel;								///< the physical value of all depth map layer.
	std::vector<Real>		dlevel_transform;					///< transfomed dlevel variable

	OphDepthMapConfig		dm_config_;							///< structure variable for depthmap hologram configuration.
};

#endif //>__ophDepthMap_h