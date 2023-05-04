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

#include "ophSimulator.h"
#include "include.h"
#include "tinyxml2.h"
#include <sys.h>
#include <cufft.h>
#include "Openholo.h"
#include "function.h"

ophSimulator::ophSimulator(void)
	: ophGen()
{
	m_nPoints = 0;
	m_nPlanes = 0;
	m_vecPoints.clear();
	m_vecPlanes.clear();
	m_vecAmplitude.clear();
	m_bHasPoint = false;
	m_bHasPlane = false;
	m_pPointCloud = nullptr;
	m_distance = 0;
	m_pNormalize = nullptr;
}

ophSimulator::~ophSimulator(void)
{
	delete[] m_pNormalize;
	m_pPointCloud->release();
}

void ophSimulator::Init()
{
	m_vecPoints.clear();
	m_vecPlanes.clear();
	m_vecAmplitude.clear();
	m_nPoints = 0;
	m_bHasPoint = false;
	m_bHasPlane = false;
	m_distance = 0;
	m_pNormalize = nullptr;

	if (m_pPointCloud != nullptr) {
		m_pPointCloud->release();
	}
	m_pPointCloud = new ophPointCloud();
}

int ophSimulator::AddPlane(Real theta, Real phi)
{
	m_bHasPlane = true;
	m_vecPlanes.push_back(vec2(theta, phi));
	return m_nPlanes = (int)m_vecPlanes.size();
}

int ophSimulator::AddPoint(vec3 point, Real amplitude)
{
	m_bHasPoint = true;
	m_vecPoints.push_back(point);
	m_vecAmplitude.push_back(amplitude);
	return m_nPoints = (int)m_vecPoints.size();
}

int ophSimulator::AddPoint(Real x, Real y, Real z, Real amplitude)
{
	return AddPoint(vec3(x, y, z), amplitude);
}

bool ophSimulator::SetResolution(ivec2 resolution)
{
	if (m_pPointCloud == nullptr) return false;
	m_pPointCloud->setPixelNumber(resolution);
	return true;
}

bool ophSimulator::SetResolution(int width, int height)
{
	return SetResolution(ivec2(width, height));
}

bool ophSimulator::SetPixelPitch(vec2 size)
{
	if (m_pPointCloud == nullptr) return false;
	m_pPointCloud->setPixelPitch(size);
	return true;
}

bool ophSimulator::SetPixelPitch(Real width, Real height)
{
	return SetPixelPitch(vec2(width, height));
}

bool  ophSimulator::SetWaveLength(Real waveLength)
{
	if (m_pPointCloud == nullptr) return false;
	m_pPointCloud->setWaveLength(waveLength);
	return true;
}

bool ophSimulator::SetDistance(Real distance)
{
	if (m_pPointCloud == nullptr) return false;
	m_distance = distance;
	m_pPointCloud->setDistance(distance);
	return true;
}

bool ophSimulator::SetWaveNum(int nNum)
{
	if (m_pPointCloud == nullptr) return false;
	m_pPointCloud->setWaveNum(nNum);
	return true;
}

bool ophSimulator::Save(char *path)
{
	if (m_pPointCloud == nullptr) return false;
	return m_pPointCloud->save(path, 8, (m_bHasPlane) ? m_pNormalize : nullptr, 0, 0);
}

bool ophSimulator::Encode(int option)
{
	if (m_pPointCloud == nullptr) return false;
	m_pPointCloud->encoding(option);
	m_pPointCloud->normalize();
	return true;
}

template<typename T>
void ophSimulator::Normalize(T *src, uchar *dst, int width, int height)
{
	T minVal, maxVal;
	for (int i = 0; i < width * height; i++) {
		T *tmpVal = src + i;
		if (i == 0) {
			minVal = maxVal = *(tmpVal);
		}
		else {
			if ((*tmpVal) < minVal) minVal = (*tmpVal);
			if ((*tmpVal) > maxVal) maxVal = (*tmpVal);
		}
	}

	T gap = maxVal - minVal;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			T *tmpVal = src + x + (y * width);
			uchar *resVal = dst + x + (height - y - 1) * width;
			*(resVal) = oph::force_cast<oph::uchar>(((*(tmpVal)-minVal) / gap) * 255 + 0.5);
		}
	}
}

bool ophSimulator::GenerateHologram()
{
	if (m_pPointCloud == nullptr) return false;

	m_pPointCloud->SetMode(MODE_CPU);
	m_pPointCloud->initialize();
	if (m_bHasPoint)
	{
		m_pPointCloud->setScale(1.0, 1.0, 1.0);

		Real *pVertex = new Real[m_nPoints * 3];
		Real *pColor = new Real[m_nPoints];
		//m_pPointCloud->setPointCloudModel(pVertex, pColor, nullptr);

		int nSize = (int)m_vecPoints.size();
		int i = 0;
		while (i < nSize) {
			pVertex[3 * i + 0] = m_vecPoints[i].v[0];
			pVertex[3 * i + 1] = m_vecPoints[i].v[1];
			pVertex[3 * i + 2] = m_vecPoints[i].v[2];
			pColor[i] = m_vecAmplitude[i];
			i++;
		}

		m_pPointCloud->setNumberOfPoints(m_nPoints);
		m_pPointCloud->generateHologram(0);
		m_pPointCloud->encoding(ENCODE_PHASE);
		m_pPointCloud->normalize();
	}
	if (m_bHasPlane)
	{
		//Real k = 2 * M_PI /
		OphConfig cfg = m_pPointCloud->getContext();
		Real lambda = cfg.wave_length[0];
		Real k = 2 * M_PI / lambda;
		const uint pnX = cfg.pixel_number[_X];
		const uint pnY = cfg.pixel_number[_Y];
		const Real ppX = cfg.pixel_pitch[_X];
		const Real ppY = cfg.pixel_pitch[_Y];
		const Real ssX = pnX * ppX;
		const Real ssY = pnY * ppY;
		const Real startX = -ssX / 2 + (ppX / 2);
		const Real startY = -ssY / 2 + (ppY / 2);
		const Real distance = m_distance;
		Complex<Real> *result = new Complex<Real>[pnX * pnY];
		Real *encode = new Real[pnX * pnY];
		if (m_pNormalize) {
			delete[] m_pNormalize;
			m_pNormalize = nullptr;
		}
		m_pNormalize = new uchar[pnX * pnY];
		memset(m_pNormalize, 0, pnX * pnY);
		int nSize = (int)m_vecPlanes.size();
		int i = 0;
		while (i < nSize) {
			Real theta = m_vecPlanes[i].v[0] * M_PI / 180;
			Real phi = m_vecPlanes[i].v[1] * M_PI / 180;
			const Real kx = k * cos(phi) * sin(theta);
			const Real ky = k * sin(phi) * sin(theta);
			const Real kz = k * cos(theta);
			const Real kzd = kz * distance;

			for (int y = 0; y < pnY; y++) {
				Real curY = startY + (y * ppY);
				int offset = y * pnX;
				for (int x = 0; x < pnX; x++) {
					Real curX = startX + (x * ppX);
					Complex<Real> tmp(0.0, (kx*curX + ky * curY + kzd));
					result[offset + x] += tmp.exp();
				}
			}
			i++;
		}

		realPart(result, encode, pnX * pnY);
		Normalize(encode, m_pNormalize, pnX, pnY);
		if (m_bHasPoint) {
			uchar **pNormal = m_pPointCloud->getNormalizedBuffer();
			int N = pnX * pnY;
			int i;
#ifdef _OPENMP
#pragma omp for private(i)
			for (i = 0; i < N; i++) {
#endif
				m_pNormalize[i] = ((int)m_pNormalize[i] + (int)pNormal[0][i]) / 2;
			}
		}
		delete[] encode;
		delete[] result;
	}


	return true;
}