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
	enum IMAGE_TYPE {
		COLOR = 0,
		DEPTH = 1
	};

	/**
	* @brief Constructor
	* @details Initialize variables.
	*/
	explicit ophDepthMap();

protected:
	/**
	* @brief Destructor 
	*/
	virtual ~ophDepthMap();

public:
	/**
	* @brief Read parameters from a config file. (*.xml)
	* @return true if config infomation are sucessfully read, flase otherwise.
	*/
	bool readConfig(const char* fname);

	/**
	* @brief Read image and depth map.
	* @details Read input files and load image & depth map data.
	*  If the input image size is different with the dislay resolution, resize the image size.
	* @param fname : image path.
	* @param type : rgb image or depth image
	* @return true if image data are sucessfully read, flase otherwise.
	* @see loadAsImg, convertToFormatGray8, imgScaleBilinear
	*/
	bool readImage(const char* fname, IMAGE_TYPE type = COLOR);

	/**
	* @brief Read image and depth map.
	* @details Read input files and load image & depth map data.
	*  If the input image size is different with the dislay resolution, resize the image size.
	*  Invert the Image to the y-axis.
	* @param source_folder : directory path
	* @param img_prefix : rgb image prefix
	* @param depth_img_prefix : depth image prefix
	* @return true if image data are sucessfully read, flase otherwise.
	* @see convertToFormatGray8, imgScaleBilinear, loadAsImgUpSideDown
	*/
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

	void setRGBImgSize(ivec2 size) {
		m_vecRGBImg[_X] = size[_X];
		m_vecRGBImg[_Y] = size[_Y];
	};

	void setDepthImgSize(ivec2 size) {
		m_vecDepthImg[_X] = size[_X];
		m_vecDepthImg[_Y] = size[_Y];
	};

	void setConfig(OphDepthMapConfig config) {
		dm_config_ = config;
	};
	void setResolution(ivec2 resolution);
	uint* getProgress() { return &m_nProgress; }


	void normalize();

	inline void setFieldLens(Real fieldlens) { dm_config_.fieldLength = fieldlens; }
	inline void setNearDepth(Real neardepth) { dm_config_.near_depthmap = neardepth; }
	inline void setFarDepth(Real fardetph) { dm_config_.far_depthmap = fardetph; }
	inline void setNumOfDepth(uint numofdepth) { 
		dm_config_.default_depth_quantization = numofdepth;
		dm_config_.num_of_depth_quantization = numofdepth;
		dm_config_.num_of_depth = numofdepth;
		dm_config_.render_depth.clear();
		dm_config_.change_depth_quantization = true;
		for(int i = 1; i <= numofdepth; i++)
			dm_config_.render_depth.push_back(i);
	}
	inline void setRGBImageBuffer(int idx, unsigned char* buffer, unsigned long long size)
	{
		if (idx < 0 || idx > 2) return;
		if (m_vecRGB.size() > idx)
		{
			if (m_vecRGB[idx] != nullptr)
			{
				delete[] m_vecRGB[idx];
				m_vecRGB[idx] = new uchar[size];
				memcpy(m_vecRGB[idx], buffer, size);
			}
		}
		else
		{
			uchar* pImg = new uchar[size];
			memcpy(pImg, buffer, size);
			m_vecRGB.push_back(pImg);
		}
	}

	inline void setDepthImageBuffer(unsigned char* buffer, unsigned long long size)
	{
		if (depth_img != nullptr)
		{
			delete[] depth_img;
			depth_img = nullptr;
		}
		depth_img = new unsigned char[size];
		memcpy(depth_img, buffer, size);
	}

	inline Real getFieldLens(void) { return dm_config_.fieldLength; }
	inline Real getNearDepth(void) { return dm_config_.near_depthmap; }
	inline Real getFarDepth(void) { return dm_config_.far_depthmap; }
	inline uint getNumOfDepth(void) { return dm_config_.num_of_depth; }
	inline void getRenderDepth(std::vector<int>& renderdepth) { renderdepth = dm_config_.render_depth; }
	inline unsigned char* getRGBImageBuffer(int idx) { return m_vecRGB[idx]; }
	inline unsigned char* getDepthImageBuffer() { return depth_img; }

	inline const OphDepthMapConfig& getConfig() { return dm_config_; }
	
private:
	/**
	* @brief Initialize variables for the CPU implementation.
	* @details Memory allocation for the CPU variables.
	* @see initialize
	*/
	void initCPU();

	/**
	* @brief Initialize variables for the GPU implementation.
	* @details Memory allocation for the GPU variables.
	* @see initialize
	*/
	void initGPU();

	/**
	* @brief Preprocess input image & depth map data for the CPU implementation.
	* @details Prepare variables, m_vecImgSrc, dmap_src_, m_vecAlphaMap, depth_index_.
	* @return true if input data are sucessfully prepared, flase otherwise.
	*/
	bool prepareInputdataCPU();
	
	/**
	* @brief Copy input image & depth map data into a GPU.
	* @return true if input data are sucessfully copied on GPU, flase otherwise.
	* @see readImageDepth
	*/
	bool prepareInputdataGPU();

	/**
	* @brief Calculate the physical distances of depth map layers
	* @details Initialize 'dstep_' & 'dlevel_' variables.
	*  If change_depth_quantization == 1, recalculate  'depth_index_' variable.
	* @see changeDepthQuanCPU, changeDepthQuanGPU
	*/
	void getDepthValues();

	/**
	* @brief Quantize depth map on the CPU, when the number of depth quantization is not the default value (i.e. change_depth_quantization == 1 ).
	* @details Calculate the value of 'depth_index_'.
	*/
	void changeDepthQuanCPU();

	/**
	* @brief Quantize depth map on the GPU, when the number of depth quantization is not the default value (i.e. change_depth_quantization == 1 ).
	* @details Calculate the value of 'depth_index_gpu'.
	* @see getDepthValues
	*/
	void changeDepthQuanGPU();

	/**
	* @brief Transform target object to reflect the system configuration of holographic display.
	* @details Calculate 'dlevel_transform_' variable by using 'fieldLength' & 'dlevel_'.
	*/
	void transVW();
	
	/**
	* @brief Main method for generating a hologram on the CPU.
	* @details For each depth level,
	*   1. find each depth plane of the input image.
	*   2. apply carrier phase delay.
	*   3. propagate it to the hologram plan.
	*   4. accumulate the result of each propagation.
	* .
	* The final result is accumulated in the variable 'complex_H'.
	* @see fftInit2D, GetRandomPhase, GetRandomPhaseValue, fft2, AngularSpectrumMethod, fftFree
	*/
	void calcHoloCPU();
	
	/**
	* @brief Main method for generating a hologram on the GPU.
	* @details For each depth level,
	*   1. find each depth plane of the input image.
	*   2. apply carrier phase delay.
	*   3. propagate it to the hologram plan.
	*   4. accumulate the result of each propagation.
	* .
	* It uses CUDA kernels, cudaDepthHoloKernel & cudaPropagation_AngularSpKernel.<br>
	* The final result is accumulated in the variable 'u_complex_gpu_'.
	* @param frame : the frame number of the image.
	* @see calc_Holo_by_Depth
	*/
	void calcHoloGPU();

	/**
	* @brief Method for checking input images.
	*/
	bool convertImage();

protected:
	void free_gpu(void);

	void ophFree(void);

private:
	bool					is_ViewingWindow;
	unsigned char*			depth_img;
	vector<uchar*>			m_vecRGB;
	ivec2					m_vecRGBImg;
	ivec2					m_vecDepthImg;
	unsigned char*			img_src_gpu;						///< GPU variable - image source data, values are from 0 to 255.
	unsigned char*			dimg_src_gpu;						///< GPU variable - depth map data, values are from 0 to 255.
	Real*					depth_index_gpu;					///< GPU variable - quantized depth map data.
	
	vector<Real *>			m_vecImgSrc;
	vector<int *>			m_vecAlphaMap;
	Real*					dmap_src;							///< CPU variable - depth map data, values are from 0 to 1.
	uint*					depth_index;						///< CPU variable - quantized depth map data.
	vector<short>			depth_fill;
	Real*					dmap;								///< CPU variable - physical distances of depth map.

	Real					dstep;								///< the physical increment of each depth map layer.
	vector<Real>			dlevel;								///< the physical value of all depth map layer.
	vector<Real>			dlevel_transform;					///< transfomed dlevel variable

	OphDepthMapConfig		dm_config_;							///< structure variable for depthmap hologram configuration.

	cufftDoubleComplex* u_o_gpu_;
	cufftDoubleComplex* u_complex_gpu_;
	cufftDoubleComplex* k_temp_d_;

	cudaStream_t	stream_;

	uint m_nProgress;
};

#endif //>__ophDepthMap_h