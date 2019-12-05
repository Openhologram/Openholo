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



#pragma once
#ifndef __OphWaveAberration_h
#define __OphWaveAberration_h

#include "ophRec.h"
#include <iostream>
#include <fstream>
#include <complex>
#include <cmath>
#include <memory>
#include <algorithm>
#include <vector>
#include "tinyxml2.h"
#include "sys.h"

using namespace std;


/**
* @addtogroup waveaberr
//@{
* @details

* @section Introduction

The Class generates the complex field of wavefront aberration of optical systems.

The wave aberrations can be mathematically represented using Zernike polynomials.
 ![](@ref pics/ophdis/waveaberration/zernike_poly.bmp)

 The Zernike polynomial is calculated according to Zernike polynomial coefficient using the Zernike polynomial equation to be described in reference.
 The each of the calculated Zernike polynomials is accumulated into one data set, which is transformed into a complex field of the wave aberration

* @section Reference

Minsik Park, Hyun-Eui Kim, Hyon-Gon Choo, Jinwoong Kim, and Cheong Hee Park,"Distortion Compensation ofReconstructed Hologram Image in Digital Holographic Display Based on Viewing Window", ETRI Journal, Volume 29, Number 4, pp. 480-492, 2017

*/
//! @} waveaberr


/**
* @ingroup waveaberr
* @brief Wave Aberration module
* @author Minsik Park
*/
class RECON_DLL ophWaveAberration : public ophRec
{
private :
	/**
	* @param wave length of illumination light
	*/
	Real waveLength;
	
	/**
	* @param sampling interval in x axis of the exit pupil
	*/
	Real pixelPitchX;
	/**
	* @param sampling interval in y axis of the exit pupil
	*/
	Real pixelPitchY;
		/**
	* @param order of the radial term of Zernike polynomial
	*/
	int nOrder; 
	/**
	* @param frequency of the sinusoidal component of Zernike polynomial 
	*/
	int mFrequency; 
	/**
	* @param Zernike coeffient
	*/
	Real zernikeCoefficent[45];

public:

	/**
	* @param resolution in x axis of the exit pupil
	*/
	oph::uint resolutionX;
	/**
	* @param resolution in y axis of the exit pupil
	*/
	oph::uint resolutionY;
	/**
	* @brief double pointer of the 2D data array of a wave aberration
	*/
	oph::Complex<Real> ** complex_W;


	/**
	* @brief Constructor
	*/
	ophWaveAberration();
	/**
	* @brief Destructor
	*/
	~ophWaveAberration();

	/**
	* @brief read configuration from a configration file
	* @param fname: a path name of a configration file
	*/
	bool readConfig(const char* fname);

	
	/**
	* @brief Factorial using recursion
	* @param x: a number of factorial 
	*/
	Real factorial(double x);
	/**
	* @brief Resizes 2D data using bilinear interpolation
	* @param X: 2D source image
	* @param Nx: resolution in x axis of source image
	* @param Ny: resolution in y axis of source image
	* @param nx: resolution in x axis target image
	* @param ny: resolution in y axis target image
	* @param ny: 2D target image
	*/
	void imresize(double **X, int Nx, int Ny, int nx, int ny, double **Y); 
	/**
	* @brief Calculates Zernike polynomial
	* @param n: order of the radial term of Zernike polynomial
	* @param m: frequency of the sinusoidal component of Zernike polynomial 
	* @param x: resolution in y axis of the exit pupil
	* @param y: resolution in y axis of the exit pupil
	* @param d: diameter of aperture of the exit pupil
	*/
	double ** calculateZernikePolynomial(double n, double m, vector<double> x, vector<double> y, double d);
	/**
	* @brief Sums up the calculated Zernike polynomials
	*/
	void accumulateZernikePolynomial();
	/**
	* @brief deletes 2D memory array using double pointer 
	*/
	void Free2D(oph::Complex<Real> ** doublePtr);
	/**
	* @brief saves the 2D data array of a wave aberration into a file
	* @param fname: a path name of a file to save a wave aberration 
	*/
	void saveAberration(const char* fname);
	/**
	* @brief reads the 2D data array of a wave aberration from a file
	* @param fname: a path name of a file to save a wave aberration
	*/
	void readAberration(const char* fname);
	virtual bool loadAsOhc(const char* fname);

	void ophFree(void);
};

#endif