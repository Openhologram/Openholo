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

#ifndef __ophWRP_h
#define __ophWRP_h

#define _USE_MATH_DEFINES

#include "ophGen.h"

#ifdef RECON_EXPORT
#define RECON_DLL __declspec(dllexport)
#else
#define RECON_DLL __declspec(dllimport)
#endif

//Build Option : Multi Core Processing (OpenMP)
#ifdef _OPENMP
#include <omp.h>
#endif


#define THREAD_X 32
#define THREAD_Y 16

/* Bitmap File Definition*/
#define OPH_Bitsperpixel 8 //24 // 3byte=24 
#define OPH_Planes 1
#define OPH_Compression 0
#define OPH_Xpixelpermeter 0x130B //2835 , 72 DPI
#define OPH_Ypixelpermeter 0x130B //2835 , 72 DPI
#define OPH_Pixel 0xFF

using namespace oph;



/**
* @addtogroup wrp
//@{
* @details

* @section Introduction

CGH generation with WRP methods is supported in this module, including single WRP method,
multiple WRP method.

I. WRP based hologram generation

-   Implement the hologram generation with point cloud and RGB-Depth data
-   Reduce the hologram generation time

II. Single WRP method

-   Wavefront Recording Plane(WRP) is a visual plane between the object plane and the hologram plane
-   calculate each active area which is the wavefront from object to hologram plane passing through a small area on WRP
-   And then propagate to the hologram by Fast Fourier Transform(FFT)

![](pics/ophgen/wrp/wrp01.jpg)

III. Multiple WRP method

-   In this Multiple WRP method, each WRPs are supporting to each corresponding depth layer
	which means the WRPs have uniformed quantizing distance between the object points and WRP
-   The location of each WRPs is calculated by a constant dimension of the active area
-   and then calculate each corresponding WRP of depth layer
-   finally, the diffraction of every WRP to hologram plane by Fresnel propagation

![](pics/ophgen/wrp/wrp02.jpg)

*/
//! @} wrp


/**
* @ingroup wrp
* @brief
* @author
*/
class GEN_DLL ophWRP : public ophGen
{

public:
	/**
	* @brief Constructor
	* @details Initialize variables.
	*/
	explicit ophWRP(void);

protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophWRP(void);

public:
	const vec3& getScale() { return wrp_config_.scale; }
	const Real& getLocation() { return wrp_config_.wrp_location; }
	const Real& getDistance() { return wrp_config_.propagation_distance; }
	const int& getNumOfWRP() { return wrp_config_.num_wrp; }
	const Real& getFieldLens() { return wrp_config_.fieldLength; }
	void setScale(vec3 scale) { wrp_config_.scale = scale; }
	void setLocation(Real location) { wrp_config_.wrp_location = location; }
	void setDistance(Real distance) { wrp_config_.propagation_distance; }
	void autoScaling();
	int getNumOfPoints() { return n_points; }

	/**
	* @brief load to point cloud data.
	* @param[in] pc_file Point cloud data file name
	* @return Type: <B>int</B>\n
	*				If the function succeeds, the return value is <B>Positive integer</B>.\n
	*				If the function fails, the return value is <B>Negative interger</B>.
	*/
	virtual int loadPointCloud(const char* pc_file);
	
	/**
	* @brief load to configuration file.
	* @param[in] fname config file name
	* @return Type: <B>bool</B>\n
	*				If the function succeeds, the return value is <B>true</B>.\n
	*				If the function fails, the return value is <B>false</B>.
	*/
	virtual bool readConfig(const char* fname);

	/**
	* @brief Function for setting precision
	* @param[in] precision level.
	*/
	void setPrecision(bool bPrecision) { bSinglePrecision = bPrecision; }
	bool getPrecision() { return bSinglePrecision; }


	double calculateWRPCPU(void);
	void calculateWRPGPU(void);

//	virtual void fresnelPropagation(Complex<Real>* in, Complex<Real>* out, Real distance);

	/**
	* @brief Generate a hologram, main funtion.
	* @return implement time (sec)
	*/
	void generateHologram(void);
	/**
	* @brief Generate multiple wavefront recording planes, main funtion.
	* @return multiple WRP (sec)
	*/
	Complex<Real>** calculateMWRP(void);

	inline Complex<Real>* getWRPBuff(void) { return p_wrp_; };

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
	uint* getProgress() { return &m_nProgress; }
	
protected:


private:

	Complex<Real>* ophWRP::calSubWRP(double d, oph::Complex<Real>* wrp, OphPointCloudData* sobj);

	void addPixel2WRP(int x, int y, oph::Complex<Real> temp);
	void addPixel2WRP(int x, int y, oph::Complex<Real> temp, oph::Complex<Real>* wrp);

	virtual void ophFree(void);
	inline Real transVW(Real pt) {
		Real fieldLens = this->getFieldLens();
		return -fieldLens * pt / (pt - fieldLens);
	}

	void transVW(Real* dst, Real *src, int size);
protected:

	int n_points;                 ///< numbers of points

	Complex<Real>* p_wrp_;   ///< wrp buffer - complex type

	OphPointCloudData obj_;       ///< Input Pointcloud Data
	Real *scaledVertex;
	OphWRPConfig wrp_config_;      ///< structure variable for WRP hologram configuration

private:
	bool is_ViewingWindow;
	bool bSinglePrecision;
	Real zmax_;
	uint m_nProgress;

};
#endif
