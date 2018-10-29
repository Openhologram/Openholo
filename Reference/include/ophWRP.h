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
* @detail

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
	const vec3& getScale() { return pc_config_.scale; }
	const Real& getLocation() { return pc_config_.wrp_location; }
	const Real& getDistance() { return pc_config_.propagation_distance; }
	const int& getNumOfWRP() { return pc_config_.num_wrp; }
	void setScale(vec3 scale) { pc_config_.scale = scale; }
	void setLocation(Real location) { pc_config_.wrp_location = location; }
	void setDistance(Real distance) { pc_config_.propagation_distance; }

	/**
	* @brief override
	* @{
	* @brief Import Point Cloud Data Base File : *.PYL file.
	* This Function is included memory location of Input Point Clouds.
	*/
	/**
	* @brief override
	* @param InputModelFile PointCloud(*.PYL) input file path
	* @return number of Pointcloud (if it failed loading, it returned -1)
	*/
	virtual int loadPointCloud(const char* pc_file);
	virtual bool readConfig(const char* cfg_file);
	virtual void normalize(void);

	void encodeHologram(void);

	double calculateWRP(void);

	virtual void fresnelPropagation(Complex<Real>* in, Complex<Real>* out, Real distance);

	void generateHologram(void);

	oph::Complex<Real>** calculateMWRP(void);

	inline oph::Complex<Real>* getWRPBuff(void) { return p_wrp_; };


private:

	Complex<Real>* ophWRP::calSubWRP(double d, oph::Complex<Real>* wrp, OphPointCloudData* sobj);

	void addPixel2WRP(int x, int y, oph::Complex<Real> temp);
	void addPixel2WRP(int x, int y, oph::Complex<Real> temp, oph::Complex<Real>* wrp);

	virtual void ophFree(void);

protected:

	int n_points;   //number of points


	oph::Complex<Real>* p_wrp_;   //wrp buffer

	OphPointCloudData obj_;

	OphWRPConfig pc_config_;

};
#endif
