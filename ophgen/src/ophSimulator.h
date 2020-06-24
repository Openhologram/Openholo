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

#ifndef __ophSimulator_h
#define __ophSimulator_h

#define _USE_MATH_DEFINES

#include "ophGen.h"
#include "ophPointCloud.h"
#include <vector>

//Build Option : Multi Core Processing (OpenMP)
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace oph;

class GEN_DLL ophSimulator : public ophGen
{
public:
	/**
	* @brief Constructor
	* @details Initialize variables.
	*/
	explicit ophSimulator(void);
	/**
	* @overload
	*/
protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophSimulator(void);

public:
	int AddPoint(vec3 point, Real amplitude = 0.5);
	int AddPoint(Real x, Real y, Real z, Real amplitude = 0.5);
	int AddPlane(Real theta, Real phi);
	bool SetResolution(ivec2 resolution);
	bool SetResolution(int width, int height);
	bool SetPixelPitch(vec2 size);
	bool SetPixelPitch(Real width, Real height);
	bool SetWaveLength(Real waveLength);
	bool SetDistance(Real distance);
	bool SetWaveNum(int nNum);
	bool Save(char *path);
	bool Encode(int option);
	void Init();
	bool GenerateHologram();
	template<typename T>
	void Normalize(T *src, uchar *dst, int width, int height);

private:
	ophPointCloud *m_pPointCloud;
	uchar *m_pNormalize;
	int m_nPoints;
	int m_nPlanes;
	Real m_distance;
	vector<vec3> m_vecPoints;
	vector<vec2> m_vecPlanes;
	vector<Real> m_vecAmplitude;
	bool m_bHasPoint;
	bool m_bHasPlane;
};

#endif // !__ophSimulator_h