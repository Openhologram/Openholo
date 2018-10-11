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

#ifndef __ophPointCloud_h
#define __ophPointCloud_h

#define _USE_MATH_DEFINES

#include "ophGen.h"

//Build Option : Multi Core Processing (OpenMP)
#ifdef _OPENMP
#include <omp.h>
#endif

/* Bitmap File Definition*/
#define OPH_Bitsperpixel 8 //24 // 3byte=24 
#define OPH_Planes 1
#define OPH_Compression 0
#define OPH_Xpixelpermeter 0x130B //2835 , 72 DPI
#define OPH_Ypixelpermeter 0x130B //2835 , 72 DPI
#define OPH_Pixel 0xFF

using namespace oph;

/**
* @addtogroup pointcloud
//@{
* @detail

*/
//! @} pointcloud

/**
* @ingroup pointcloud
* @brief Openholo Point Cloud based CGH generation
* @author
*/
class GEN_DLL ophPointCloud : public ophGen
{
public:
	/**
	* @brief Constructor
	* @details Initialize variables.
	*/
	explicit ophPointCloud(void);
	/**
	* @overload
	*/
	explicit ophPointCloud(const char*, const char* cfg_file);
protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophPointCloud(void);

public:
	inline void setScale(Real sx, Real sy, Real sz) { pc_config_.scale.v[0] = sx; pc_config_.scale.v[1] = sy; pc_config_.scale.v[2] = sz; }
	inline void setOffsetDepth(Real offset_depth) { pc_config_.offset_depth = offset_depth; }
	inline void setFilterShapeFlag(int8_t* fsf) { pc_config_.filter_shape_flag = fsf; }
	inline void setFilterWidth(Real wx, Real wy) { pc_config_.filter_width.v[0] = wx; pc_config_.filter_width.v[1] = wy; }
	inline void setFocalLength(Real lens_in, Real lens_out, Real lens_eye_piece) { pc_config_.focal_length_lens_in = lens_in; pc_config_.focal_length_lens_out = lens_out; pc_config_.focal_length_lens_eye_piece = lens_eye_piece; }
	inline void setTiltAngle(Real ax, Real ay) { pc_config_.tilt_angle.v[0] = ax; pc_config_.tilt_angle.v[1] = ay; }

	inline void getScale(vec3& scale) { scale = pc_config_.scale; }
	inline Real getOffsetDepth(void) { return pc_config_.offset_depth; }
	inline int8_t* getFilterShapeFlag(void) { return pc_config_.filter_shape_flag; }
	inline void getFilterWidth(vec2& filterwidth) { filterwidth = pc_config_.filter_width; }
	inline void getFocalLength(Real* lens_in, Real* lens_out, Real* lens_eye_piece) {
		if (lens_in != nullptr) *lens_in = pc_config_.focal_length_lens_in;
		if (lens_out != nullptr) *lens_out = pc_config_.focal_length_lens_out;
		if (lens_eye_piece != nullptr) *lens_eye_piece = pc_config_.focal_length_lens_eye_piece;
	}
	inline void getTiltAngle(vec2& tiltangle) { tiltangle = pc_config_.tilt_angle; }
	inline Real** getVertex(void) { return &pc_data_.vertex; }
	inline Real** getColorPC(void) { return &pc_data_.color; }
	inline Real** getPhasePC(void) { return &pc_data_.phase; }
	inline void setPointCloudModel(Real* vertex, Real* color, Real *phase) {
		pc_data_.vertex = vertex;
		pc_data_.color = color;
		pc_data_.phase = phase;
	}
	inline void getPointCloudModel(Real *vertex, Real *color, Real *phase) {
		getModelLocation(vertex);
		getModelColor(color);
		getModelPhase(phase);
	}

	/**
	* @brief Directly Set Basic Data
	*/
	/**
	* @param Location 3D Point Cloud Geometry Data
	* @param Color 3D Point Cloud Color Data
	* @param Amplitude 3D Point Cloud Model Amplitude Data of Point-Based Light Wave
	* @param Phase 3D Point Cloud Model Phase Data of Point-Based Light Wave
	*/
	inline void getModelLocation(Real *vertex) { vertex = pc_data_.vertex; }
	inline void getModelColor(Real *color) { color = pc_data_.color; }
	inline void getModelPhase(Real *phase) { phase = pc_data_.phase; }
	inline int getNumberOfPoints() { return n_points; }

public:
	/**
	* @brief Set the value of a variable is_CPU(true or false)
	* @details <pre>
	if is_CPU == true
	CPU implementation
	else
	GPU implementation </pre>
	* @param is_CPU : the value for specifying whether the hologram generation method is implemented on the CPU or GPU
	*/
	void setMode(bool is_CPU);

	/**
	* @brief override
	* @{
	* @brief Import Point Cloud Data Base File : *.dat file.
	* This Function is included memory location of Input Point Clouds.
	*/
	/**
	* @brief override
	* @param InputModelFile PointCloud(*.dat) input file path
	* @return number of Pointcloud (if it failed loading, it returned -1)
	*/
	int loadPointCloud(const char* pc_file);

	/**
	* @brief
	* @{
	* @brief Import Specification Config File(*.config) file
	*/
	/**
	* @param InputConfigFile Specification Config(*.config) file path
	*/
	bool readConfig(const char* cfg_file);

	/**
	* @brief Generate a hologram, main funtion.
	* @return implement time (sec)
	*/
	Real generateHologram(uint diff_flag = PC_DIFF_RS);
	void encodeHologram(vec2 band_limit = vec2(0.8, 0.5), vec2 spectrum_shift = vec2(0.0, 0.5));

private:
	/**
	* @brief Calculate Integral Fringe Pattern of 3D Point Cloud based Computer Generated Holography
	* @param VertexArray Input 3D PointCloud Model Coordinate Array Data
	* @param AmplitudeArray Input 3D PointCloud Model Amplitude Array Data
	* @param dst Output Fringe Pattern
	* @return implement time (sec)
	*/
	void genCghPointCloudCPU(uint diff_flag);
	void diffractEncodedRS(ivec2 pn, vec2 pp, vec2 ss, vec3 pc, Real k, Real amplitude, vec2 theta);
	void diffractNotEncodedRS(ivec2 pn, vec2 pp, vec2 ss, vec3 pc, Real k, Real amplitude, Real lambda, vec2 theta);
	void diffractEncodedFrsn(void);
	void diffractNotEncodedFrsn(ivec2 pn, vec2 pp, vec3 pc, Real amplitude, Real lambda, vec2 theta);

	/**
	* @overload
	* @param Model Input 3D PointCloud Model Data
	* @param dst Output Fringe Pattern
	* @return implement time (sec)
	*/
	//double genCghPointCloud(const std::vector<PointCloud> &Model, float *dst);
	/** @}	*/

	/**
	* @brief GPGPU Accelation of genCghPointCloud() using nVidia CUDA
	* @param VertexArray Input 3D PointCloud Model Coordinate Array Data
	* @param AmplitudeArray Input 3D PointCloud Model Amplitude Array Data
	* @param dst Output Fringe Pattern
	* @return implement time (sec)
	*/
	void genCghPointCloudGPU(uint diff_flag);

	/** @}	*/

	/**
	* @brief normalize calculated fringe pattern to 8bit grayscale value.
	* @param src: Input float type pointer
	* @param dst: Output char tpye pointer
	* @param nx: The number of pixels in X
	* @param ny: The number of pixels in Y
	*/
	virtual void ophFree(void);

	bool is_CPU;
	int n_points;

	OphPointCloudConfig pc_config_;
	OphPointCloudData	pc_data_;
};

#endif // !__ophPointCloud_h