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

/**
* @file		ophLightField.h
* @brief	Openholo Light Field based CGH generation
* @author	Yeon-Gyeong Ju, Jae-Hyeung Park
* @data		2018-08
* @version	0.0.1
*/
#ifndef __ophLightField_h
#define __ophLightField_h

#include "ophGen.h"
#include <fstream>
#include <io.h>

using namespace oph;

/**
* @brief	Openholo Light Field based CGH generation class
*/
class GEN_DLL ophLF : public ophGen
{
public:
	/**
	* @brief Constructor
	*/
	explicit ophLF(void) {}

protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophLF(void) {}

private:

	uchar** LF;										/// Light Field array / 4-D array
	Complex<Real>* RSplane_complex_field;			/// Complex field in Ray Sampling plane

private:
	
	// Light Field save parameters

	const char* LF_directory;
	const char* ext;

public:
	/** \ingroup getter/setter */
	inline void setNumImage(int nx, int ny) { num_image[_X] = nx; num_image[_Y] = ny; }
	/** \ingroup getter/setter */
	inline void setNumImage(ivec2 num) { num_image = num; }
	/** \ingroup getter/setter */
	inline void setResolImage(int nx, int ny) { resolution_image[_X] = nx; resolution_image[_Y] = ny; }
	/** \ingroup getter/setter */
	inline void setResolImage(ivec2 num) { resolution_image = num; }
	/** \ingroup getter/setter */
	inline void setDistRS2Holo(Real dist) { distanceRS2Holo = dist; }
	/** \ingroup getter/setter */
	inline ivec2 getNumImage() { return num_image; }
	/** \ingroup getter/setter */
	inline ivec2 getResolImage() { return resolution_image; }
	/** \ingroup getter/setter */
	inline Real getDistRS2Holo() { return distanceRS2Holo; }
	/** \ingroup getter/setter */
	inline uchar** getLF() { return LF; }
	/** \ingroup getter/setter */
	inline oph::Complex<Real>* getRSPlane() { return RSplane_complex_field; }
public:
	/**
	* @brief	Light Field based CGH configuration file load
	* @details	xml configuration file load
	* @return	distanceRS2Holo
	* @return	num_image
	* @return	resolution_image
	* @return	context_.pixel_pitch
	* @return	context_.pixel_number
	* @return	context_.lambda
	*/
	int readLFConfig(const char* LF_config);

	/**
	* @brief	Light Field images load
	* @param	directory		Directory which has the Light Field source image files
	* @param	exten			Light Field images extension
	* @return	LF
	* @overload
	*/
	int loadLF(const char* directory, const char* exten);
	int loadLF();
	//void readPNG(const string filename, uchar* data);

	/**
	* @brief	Hologram generation
	* @return	holo_gen
	*/
	void generateHologram();

protected:
	
	// Inner functions

	void initializeLF();
	void convertLF2ComplexField();

public:
	/**
	* @brief	Wave carry
	* @param	Real	carryingAngleX		Wave carrying angle in horizontal direction
	* @param	Real	carryingAngleY		Wave carrying angle in vertical direction
	*/
	void waveCarry(Real carryingAngleX, Real carryingAngleY);

private:
	
	ivec2 num_image;						/// The number of LF source images {numX, numY}
	ivec2 resolution_image;					/// Resolution of LF source images {resolutionX, resolutionY}
	Real distanceRS2Holo;					/// Distance from Ray Sampling plane to Hologram plane
};


#endif