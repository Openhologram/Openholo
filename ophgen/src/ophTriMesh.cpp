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

#include "ophTriMesh.h"
#include "tinyxml2.h"
#include "PLYparser.h"
#include "sys.h"

#define _X1 0
#define _Y1 1
#define _Z1 2
#define _X2 3
#define _Y2 4
#define _Z2 5
#define _X3 6
#define _Y3 7
#define _Z3 8

ophTri::ophTri(void)
	: ophGen()
	, scaledMeshData(nullptr)
	, no(nullptr)
	, na(nullptr)
	, nv(nullptr)
	, angularSpectrum(nullptr)
	, refAS(nullptr)
	, rearAS(nullptr)
	, convol(nullptr)
	, phaseTerm(nullptr)
	, is_ViewingWindow(false)
	, meshData(nullptr)
	, tempFreqTermX(nullptr)
	, tempFreqTermY(nullptr)
	, streamTriMesh(nullptr)
	, angularSpectrum_GPU(nullptr)
	, ffttemp(nullptr)
{
	LOG("*** TRI MESH : BUILD DATE: %s %s ***\n\n", __DATE__, __TIME__);
}

void ophTri::setViewingWindow(bool is_ViewingWindow)
{
	this->is_ViewingWindow = is_ViewingWindow;
}

bool ophTri::loadMeshData(const char* fileName, const char* ext)
{
	auto begin = CUR_TIME;

	if (meshData != nullptr)
	{
		delete meshData;
	}

	meshData = new OphMeshData;

	if (!strcmp(ext, "ply")) {
		PLYparser meshPLY;
		if (meshPLY.loadPLY(fileName, meshData->n_faces, &meshData->faces))
			cout << "Mesh Data Load Finished.." << endl;
		else
		{
			LOG("<FAILED> Loading ply file.\n");
			return false;
		}
	}
	else {
		LOG("<FAILED> Wrong file ext.\n");
		return false;
	}

	triMeshArray = meshData->faces;

	LOG("%s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
	return true;
}

bool ophTri::readConfig(const char* fname)
{
	if (!ophGen::readConfig(fname))
		return false;

	bool bRet = true;
	using namespace tinyxml2;
	/*XML parsing*/
	tinyxml2::XMLDocument xml_doc;
	XMLNode *xml_node;

	if (!checkExtension(fname, ".xml"))
	{
		LOG("<FAILED> Wrong file ext.\n");
		return false;
	}
	if (xml_doc.LoadFile(fname) != XML_SUCCESS)
	{
		LOG("<FAILED> Loading file.\n");
		return false;
	}
	xml_node = xml_doc.FirstChild();

	char szNodeName[32] = { 0, };
	sprintf(szNodeName, "ScaleX");
	// about object
	auto next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&objSize[_X]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "ScaleY");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&objSize[_Y]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "ScaleZ");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&objSize[_Z]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}

	sprintf(szNodeName, "LampDirectionX");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&illumination[_X]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "LampDirectionY");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&illumination[_Y]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "LampDirectionZ");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryDoubleText(&illumination[_Z]))
	{
		LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
		bRet = false;
	}

	sprintf(szNodeName, "Random_Phase");
	// about extra functions
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryBoolText(&randPhase))
	{
		LOG("<FAILED> Not found node : \'%s\' (Boolean) \n", szNodeName);
		bRet = false;
	}

	sprintf(szNodeName, "Occlusion");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryBoolText(&occlusion))
	{
		LOG("<FAILED> Not found node : \'%s\' (Boolean) \n", szNodeName);
		bRet = false;
	}
	sprintf(szNodeName, "Texture");
	next = xml_node->FirstChildElement(szNodeName);
	if (!next || XML_SUCCESS != next->QueryBoolText(&textureMapping))
	{
		LOG("<FAILED> Not found node : \'%s\' (Boolean) \n", szNodeName);
		bRet = false;
	}
	if (textureMapping == true)
	{
		sprintf(szNodeName, "TextureSizeX");
		next = xml_node->FirstChildElement(szNodeName);
		if (!next || XML_SUCCESS != next->QueryIntText(&texture.dim[_X]))
		{
			LOG("<FAILED> Not found node : \'%s\' (Integer) \n", szNodeName);
			bRet = false;
		}
		sprintf(szNodeName, "TextureSizeY");
		next = xml_node->FirstChildElement(szNodeName);
		if (!next || XML_SUCCESS != next->QueryIntText(&texture.dim[_Y]))
		{
			LOG("<FAILED> Not found node : \'%s\' (Integer) \n", szNodeName);
			bRet = false;
		}
		sprintf(szNodeName, "TexturePeriod");
		next = xml_node->FirstChildElement(szNodeName);
		if (!next || XML_SUCCESS != next->QueryDoubleText(&texture.period))
		{
			LOG("<FAILED> Not found node : \'%s\' (Double) \n", szNodeName);
			bRet = false;
		}
	}
	initialize();

	LOG("**************************************************\n");
	LOG("              Read Config (Tri Mesh)              \n");
	LOG("1) Illumination Direction : %.5lf / %.5lf / %.5lf\n", illumination[_X], illumination[_Y], illumination[_Z]);
	LOG("2) Object Scale : %.5lf / %.5lf / %.5lf\n", objSize[_X], objSize[_Y], objSize[_Z]);
	LOG("**************************************************\n");
	return bRet;
}

void ophTri::loadTexturePattern(const char* fileName, const char* ext)
{
	int N = context_.pixel_number[_X] * context_.pixel_number[_Y];

	if (tempFreqTermX != nullptr) delete[] tempFreqTermX;
	if (tempFreqTermY != nullptr) delete[] tempFreqTermY;
	if (texture.pattern != nullptr) delete[] texture.pattern;
	if (textFFT != nullptr) delete[] textFFT;

	// not used.
	//uchar* image;
	//image = loadAsImg(fileName);

	int bytesperpixel;
	int size[2] = { 0,0 };
	getImgSize(texture.dim[_X], texture.dim[_Y], bytesperpixel, fileName);
	
	texture.freq = 1 / texture.period;

	texture.pattern = new Complex<Real>[texture.dim[_X] * texture.dim[_Y]];
	textFFT = new Complex<Real>[texture.dim[_X] * texture.dim[_Y]];
	fft2(texture.dim, texture.pattern, OPH_FORWARD, OPH_ESTIMATE);
	fftExecute(texture.pattern);
	fft2(texture.pattern, textFFT, texture.dim[_X], texture.dim[_Y], OPH_FORWARD);


	tempFreqTermX = new Real[N];
	tempFreqTermY = new Real[N];
}

void ophTri::initializeAS()
{
	const long long int pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];
	const int N = meshData->n_faces;

	if (scaledMeshData != nullptr) {
		delete[] scaledMeshData;
	}

	scaledMeshData = new Face[N];
	memset(scaledMeshData, 0, sizeof(Face) * N);

	if (angularSpectrum != nullptr) {
		delete[] angularSpectrum;
	}
	angularSpectrum = new Complex<Real>[pnXY];
	memset(angularSpectrum, 0, sizeof(Complex<Real>) * pnXY);

	if (rearAS != nullptr) {
		delete[] rearAS;
	}
	rearAS = new Complex<Real>[pnXY];
	memset(rearAS, 0, sizeof(Complex<Real>) * pnXY);

	if (refAS != nullptr) {
		delete[] refAS;
	}
	refAS = new Complex<Real>[pnXY];
	memset(refAS, 0, sizeof(Complex<Real>) * pnXY);

	if (phaseTerm != nullptr) {
		delete[] phaseTerm;
	}
	phaseTerm = new Complex<Real>[pnXY];
	memset(phaseTerm, 0, sizeof(Complex<Real>) * pnXY);


	if (convol != nullptr) {
		delete[] convol;
	}
	convol = new Complex<Real>[pnXY];
	memset(convol, 0, sizeof(Complex<Real>) * pnXY);


	if (no != nullptr) {
		delete[] no;
	}
	no = new vec3[N];
	memset(no, 0, sizeof(vec3) * N);


	if (na != nullptr) {
		delete[] na;
	}
	na = new vec3[N];
	memset(na, 0, sizeof(vec3) * N);


	if (nv != nullptr) {
		delete[] nv;
	}
	nv = new vec3[N * 3];
	memset(nv, 0, sizeof(vec3) * N * 3);
}

void ophTri::objSort(bool isAscending)
{
	auto begin = CUR_TIME;
	int N = meshData->n_faces;
	Real* centerZ = new Real[N];

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < N; i++) {
		centerZ[i] = (
			scaledMeshData[i].vertices[_FIRST].point.pos[_Z] + 
			scaledMeshData[i].vertices[_SECOND].point.pos[_Z] + 
			scaledMeshData[i].vertices[_THIRD].point.pos[_Z]) / 3;
	}
	
	bool bSwap = false;

	Face tmp;
	if (isAscending)
	{
		while (true)
		{
			for (int i = 0; i < N - 1; i++)
			{
				bSwap = false;
				int j = i + 1;
				if (centerZ[i] > centerZ[j]) {
					Real tmpZ = centerZ[i];
					centerZ[i] = centerZ[j];
					centerZ[j] = tmpZ;

					tmp = scaledMeshData[i];
					scaledMeshData[i] = scaledMeshData[j];
					scaledMeshData[j] = tmp;
					bSwap = true;
				}
			}
			if (!bSwap)
				break;
		}
	}
	else
	{
		while (true)
		{
			for (int i = 0; i < N - 1; i++)
			{
				bSwap = false;
				int j = i + 1;
				if (centerZ[i] < centerZ[j]) {
					Real tmpZ = centerZ[i];
					centerZ[i] = centerZ[j];
					centerZ[j] = tmpZ;

					tmp = scaledMeshData[i];
					scaledMeshData[i] = scaledMeshData[j];
					scaledMeshData[j] = tmp;
					bSwap = true;
				}
			}
			if (!bSwap)
				break;
		}
	}
	delete[] centerZ;
	LOG("<END> %s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
	
}

vec3 vecCross(const vec3& a, const vec3& b)
{
	vec3 c;

	c(0) = a(0 + 1) * b(0 + 2) - a(0 + 2) * b(0 + 1);

	c(1) = a(1 + 1) * b(1 + 2) - a(1 + 2) * b(1 + 1);

	c(2) = a(2 + 1) * b(2 + 2) - a(2 + 2) * b(2 + 1);


	return c;
}

void ophTri::triTimeMultiplexing(char* dirName, uint ENCODE_METHOD, Real cenFx, Real cenFy, Real rangeFx, Real rangeFy, Real stepFx, Real stepFy)
{
	TM = true;
	char strFxFy[30];
	Complex<Real>* AS = new Complex<Real>[context_.pixel_number[_X] * context_.pixel_number[_Y]];

	int nFx = floor(rangeFx / stepFx);
	int nFy = floor(rangeFy / stepFy);
	Real tFx, tFy, tFz;
	for (int iFy = 0; iFy <= nFy; iFy++) {
		for (int iFx = 0; iFx <= nFx; iFx++) {

			tFx = cenFx - rangeFx / 2 + iFx*stepFx;
			tFy = cenFy - rangeFy / 2 + iFy*stepFy;
			tFz = sqrt(1.0 / context_.wave_length[0] / context_.wave_length[0] - tFx*tFx - tFy*tFy);

			carrierWave[_X] = tFx*context_.wave_length[0];
			carrierWave[_Y] = tFy*context_.wave_length[0];
			carrierWave[_Z] = tFz*context_.wave_length[0];

			generateHologram(SHADING_FLAT);

			setEncodeMethod(ENCODE_METHOD);
			ophGen::encoding();
			normalize();
			sprintf(strFxFy, "%s/holo_%d,%d.bmp", dirName, (int)tFx, (int)tFy);
			save(strFxFy, 8, nullptr, m_vecEncodeSize[_X], m_vecEncodeSize[_Y]);

			fft2(context_.pixel_number, complex_H[0], OPH_FORWARD);
			fft2(complex_H[0], complex_H[0], context_.pixel_number[_X], context_.pixel_number[_Y], OPH_FORWARD);

			setEncodeMethod(ENCODE_AMPLITUDE);
			ophGen::encoding();
			normalize();
			sprintf(strFxFy, "%s/AS_%d,%d.bmp", dirName, (int)tFx, (int)tFy);
			save(strFxFy, 8, nullptr, m_vecEncodeSize[_X], m_vecEncodeSize[_Y]);
		}
	}
}

Real ophTri::generateHologram(uint SHADING_FLAG)
{
	if (SHADING_FLAG != SHADING_FLAT && SHADING_FLAG != SHADING_CONTINUOUS) {
		LOG("<FAILED> WRONG SHADING_FLAG\n");
		return 0.0;
	}

	resetBuffer();

	auto begin = CUR_TIME;
	LOG("**************************************************\n");
	LOG("                Generate Hologram                 \n");
	LOG("1) Algorithm Method : Tri Mesh\n");
	LOG("2) Generate Hologram with %s\n", m_mode & MODE_GPU ?
		"GPU" :
#ifdef _OPENMP
		"Multi Core CPU"
#else
		"Single Core CPU"
#endif
	);
	LOG("3) Random Phase Use : %s\n", GetRandomPhase() ? "Y" : "N");
	LOG("4) Shading Flag : %s\n", SHADING_FLAG == SHADING_FLAG::SHADING_FLAT ? "Flat" : "Continuous");
	LOG("5) Number of Mesh : %llu\n", meshData->n_faces);
	LOG("**************************************************\n");

	if (m_mode & MODE_GPU)
	{
		initialize_GPU();
		prepareMeshData();
		objSort(false);
		generateAS_GPU(SHADING_FLAG);
	}
	else
	{
		initializeAS();
		prepareMeshData();
		objSort(false);
		generateAS(SHADING_FLAG);
	}

	/*

	if (!(m_mode & MODE_GPU)) {
		fft2(context_.pixel_number, angularSpectrum, OPH_BACKWARD, OPH_ESTIMATE);
		fft2(angularSpectrum, *(complex_H), context_.pixel_number[_X], context_.pixel_number[_Y], OPH_BACKWARD);
		//fft2(context_.pixel_number, *(complex_H), OPH_FORWARD, OPH_ESTIMATE);
		//fft2(*(complex_H), *(complex_H), context_.pixel_number[_X], context_.pixel_number[_Y], OPH_FORWARD);
		//fftExecute((*complex_H));
		//complex_H[0]= angularSpectrum;

		fftFree();
	}
	*/
	//fresnelPropagation(*(complex_H), *(complex_H), objShift[_Z]);
	Real elapsed_time = ELAPSED_TIME(begin, CUR_TIME);
	LOG("Total Elapsed Time: %.5lf (s)\n", elapsed_time);
	m_nProgress = 0;
	return elapsed_time;
}

bool ophTri::generateAS(uint SHADING_FLAG)
{
	auto begin = CUR_TIME;

	const long long int pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];
	int N = meshData->n_faces;

	Point* freq = new Point[pnXY];
	Point* fl = new Point[pnXY];
	Real *freqTermX = new Real[pnXY];
	Real *freqTermY = new Real[pnXY];

	findNormals(SHADING_FLAG);

	for (uint ch = 0; ch < context_.waveNum; ch++)
	{
		calGlobalFrequency(freq, context_.wave_length[ch]);		

		for (int j = 0; j < N; j++)
		{
			Face mesh = scaledMeshData[j];

			geometric geom;// = { 0, };
			//memcpy((void *)&mesh, &scaledMeshData[9 * j], sizeof(Real) * 9);
			
			if (!checkValidity(no[j])) // don't care
				continue;
			if (!findGeometricalRelations(mesh, no[j], geom))
				continue;
			if (!calFrequencyTerm(freq, fl, freqTermX, freqTermY, geom, context_.wave_length[ch]))
				continue;

			switch (SHADING_FLAG)
			{
			case SHADING_FLAT:
				refAS_Flat(no[j], freq, mesh, freqTermX, freqTermY, geom, context_.wave_length[ch]);
				//refAS_Flat(no[j], freq, mesh, freqTermX, freqTermY, geom);
				break;
			case SHADING_CONTINUOUS:
				refAS_Continuous(j, freqTermX, freqTermY);
				break;
			default:
				LOG("<FAILED> WRONG SHADING_FLAG\n");
				return false;
			}
			if (!refToGlobal(complex_H[ch], freq, fl, geom))
				continue;

			m_nProgress = ((Real)(ch * N + j) / (Real)(N * context_.waveNum)) * 100;

			//m_nProgress = (int)((Real)j * 100 / ((Real)N));
		}
	}
#if 1
	delete[] freq;
	delete[] fl;
	delete[] scaledMeshData;
	delete[] freqTermX;
	delete[] freqTermY;
	delete[] refAS;
	delete[] phaseTerm;
	delete[] convol;
	delete[] no;
	delete[] na;
	delete[] nv;
	scaledMeshData = nullptr;
	refAS = nullptr;
	phaseTerm = nullptr;
	convol = nullptr;
#else
	for (int i = 0; i < 3; i++) {
		delete[] freq[i];
		delete[] fl[i];
	}

	delete[] freq, fl, scaledMeshData, freqTermX, freqTermY, refAS, phaseTerm, convol;
	scaledMeshData = nullptr;
	refAS = nullptr;
	phaseTerm = nullptr;
	convol = nullptr;
#endif

	LOG("<END> %s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));

	return true;
}

void ophTri::calGlobalFrequency(Point* frequency, Real lambda)
{
	auto begin = CUR_TIME;
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];

	Real dfx = 1 / (ppX * pnX);
	Real dfy = 1 / (ppY * pnY);

	int startX = pnX >> 1;
	int startY = pnY >> 1;

	Real dfl = 1 / lambda;
	Real sqdfl = dfl * dfl;

#ifdef _OPENMP
#pragma omp parallel for firstprivate(pnX, startX, dfy, dfx, sqdfl)
#endif
	for (int i = startY; i > -startY; i--) {
		Real y = i * dfy;
		Real yy = y * y;

		int base = (startY - i) * pnX; // for parallel

		for (int j = -startX; j < startX; j++) {
			int idx = base + (j + startX); // for parallel
			Real x = j * dfx;
			Real xx = x * x;
			frequency[idx].pos[_X] = x;
			frequency[idx].pos[_Y] = y;
			frequency[idx].pos[_Z] = sqrt(sqdfl - xx - yy);
		}
	}
	LOG("<END> %s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
}

bool ophTri::findNormals(uint SHADING_FLAG)
{
	auto begin = CUR_TIME;

	int N = meshData->n_faces;

	Real normNo = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:normNo)
#endif
	for (int i = 0; i < N; i++) {
		no[i] = vecCross(
			{
				scaledMeshData[i].vertices[_FIRST].point.pos[_X] - scaledMeshData[i].vertices[_SECOND].point.pos[_X],
				scaledMeshData[i].vertices[_FIRST].point.pos[_Y] - scaledMeshData[i].vertices[_SECOND].point.pos[_Y],
				scaledMeshData[i].vertices[_FIRST].point.pos[_Z] - scaledMeshData[i].vertices[_SECOND].point.pos[_Z]
			},
			{
				scaledMeshData[i].vertices[_THIRD].point.pos[_X] - scaledMeshData[i].vertices[_SECOND].point.pos[_X],
				scaledMeshData[i].vertices[_THIRD].point.pos[_Y] - scaledMeshData[i].vertices[_SECOND].point.pos[_Y],
				scaledMeshData[i].vertices[_THIRD].point.pos[_Z] - scaledMeshData[i].vertices[_SECOND].point.pos[_Z]
			}
			);
		Real tmp = norm(no[i]);
		normNo += tmp * tmp;
	}
	normNo = sqrt(normNo);

#ifdef _OPENMP
#pragma omp parallel for firstprivate(normNo)
#endif
	for (int i = 0; i < N; i++) {
		na[i] = no[i] / normNo;
	}

	if (SHADING_FLAG == SHADING_CONTINUOUS) {
		vec3* vertices = new vec3[N * 3];
		vec3 zeros(0, 0, 0);
		uint N3 = N * 3;
		for (uint i = 0; i < N3; i++)
		{
			int idxFace = i / 3;
			int idxVertex = i % 3;
			std::memcpy(&vertices[i], &scaledMeshData[idxFace].vertices[idxVertex].point, sizeof(Point));
		}

		//for (uint idx = 0; idx < N * 3; idx++) {
		//	memcpy(&vertices[idx], &scaledMeshData[idx * 3], sizeof(vec3));
		//}
		for (uint idx1 = 0; idx1 < N3; idx1++) {
			if (vertices[idx1] == zeros)
				continue;
			vec3 sum = na[idx1 / 3];
			uint count = 1;
			uint* idxes = new uint[N3];
			idxes[0] = idx1;
			for (uint idx2 = idx1 + 1; idx2 < N3; idx2++) {
				if (vertices[idx2] == zeros)
					continue;
				if ((vertices[idx1][0] == vertices[idx2][0])
					&& (vertices[idx1][1] == vertices[idx2][1])
					&& (vertices[idx1][2] == vertices[idx2][2])) {

					sum += na[idx2 / 3];
					vertices[idx2] = zeros;
					idxes[count++] = idx2;
				}
			}
			vertices[idx1] = zeros;

			sum = sum / count;
			sum = sum / norm(sum);
			for (uint i = 0; i < count; i++)
				nv[idxes[i]] = sum;

			delete[] idxes;
		}

		delete[] vertices;
	}

	LOG("<END> %s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));

	return true;
}

bool ophTri::checkValidity(vec3 no)
{
	if (no[_Z] < 0 || (no[_X] == 0 && no[_Y] == 0 && no[_Z] == 0))
		return false;
	if (no[_Z] >= 0)
		return true;

	return false;
}

bool ophTri::findGeometricalRelations(Face mesh, vec3 no, geometric& geom)
{
	vec3 n = no / norm(no);
	Face mesh_local;

	Real th, ph;
	if (n[_X] == 0 && n[_Z] == 0)
		th = 0;
	else
		th = atan(n[_X] / n[_Z]);

	Real temp = n[_Y] / sqrt(n[_X] * n[_X] + n[_Z] * n[_Z]);
	ph = atan(temp);

	Real costh = cos(th);
	Real cosph = cos(ph);
	Real sinth = sin(th);
	Real sinph = sin(ph);

	geom.glRot[0] = costh;			geom.glRot[1] = 0;		geom.glRot[2] = -sinth;
	geom.glRot[3] = -sinph * sinth;	geom.glRot[4] = cosph;	geom.glRot[5] = -sinph * costh;
	geom.glRot[6] = cosph * sinth;	geom.glRot[7] = sinph;	geom.glRot[8] = cosph * costh;

	// get distance 1st pt(x, y, z) between (0, 0, 0)
	for (int i = 0; i < 3; i++)
	{
		int idx = i * 3;
		geom.glShift[i] = -(
			geom.glRot[idx + 0] * mesh.vertices[_FIRST].point.pos[_X] +
			geom.glRot[idx + 1] * mesh.vertices[_FIRST].point.pos[_Y] +
			geom.glRot[idx + 2] * mesh.vertices[_FIRST].point.pos[_Z]);
	}

	mesh_local.vertices[_FIRST].point.pos[_X] = (geom.glRot[0] * mesh.vertices[_FIRST].point.pos[_X] + geom.glRot[1] * mesh.vertices[_FIRST].point.pos[_Y] + geom.glRot[2] * mesh.vertices[_FIRST].point.pos[_Z]) + geom.glShift[_X];
	mesh_local.vertices[_FIRST].point.pos[_Y] = (geom.glRot[3] * mesh.vertices[_FIRST].point.pos[_X] + geom.glRot[4] * mesh.vertices[_FIRST].point.pos[_Y] + geom.glRot[5] * mesh.vertices[_FIRST].point.pos[_Z]) + geom.glShift[_Y];
	mesh_local.vertices[_FIRST].point.pos[_Z] = (geom.glRot[6] * mesh.vertices[_FIRST].point.pos[_X] + geom.glRot[7] * mesh.vertices[_FIRST].point.pos[_Y] + geom.glRot[8] * mesh.vertices[_FIRST].point.pos[_Z]) + geom.glShift[_Z];

	mesh_local.vertices[_SECOND].point.pos[_X] = (geom.glRot[0] * mesh.vertices[_SECOND].point.pos[_X] + geom.glRot[1] * mesh.vertices[_SECOND].point.pos[_Y] + geom.glRot[2] * mesh.vertices[_SECOND].point.pos[_Z]) + geom.glShift[_X];
	mesh_local.vertices[_SECOND].point.pos[_Y] = (geom.glRot[3] * mesh.vertices[_SECOND].point.pos[_X] + geom.glRot[4] * mesh.vertices[_SECOND].point.pos[_Y] + geom.glRot[5] * mesh.vertices[_SECOND].point.pos[_Z]) + geom.glShift[_Y];
	mesh_local.vertices[_SECOND].point.pos[_Z] = (geom.glRot[6] * mesh.vertices[_SECOND].point.pos[_X] + geom.glRot[7] * mesh.vertices[_SECOND].point.pos[_Y] + geom.glRot[8] * mesh.vertices[_SECOND].point.pos[_Z]) + geom.glShift[_Z];

	mesh_local.vertices[_THIRD].point.pos[_X] = (geom.glRot[0] * mesh.vertices[_THIRD].point.pos[_X] + geom.glRot[1] * mesh.vertices[_THIRD].point.pos[_Y] + geom.glRot[2] * mesh.vertices[_THIRD].point.pos[_Z]) + geom.glShift[_X];
	mesh_local.vertices[_THIRD].point.pos[_Y] = (geom.glRot[3] * mesh.vertices[_THIRD].point.pos[_X] + geom.glRot[4] * mesh.vertices[_THIRD].point.pos[_Y] + geom.glRot[5] * mesh.vertices[_THIRD].point.pos[_Z]) + geom.glShift[_Y];
	mesh_local.vertices[_THIRD].point.pos[_Z] = (geom.glRot[6] * mesh.vertices[_THIRD].point.pos[_X] + geom.glRot[7] * mesh.vertices[_THIRD].point.pos[_Y] + geom.glRot[8] * mesh.vertices[_THIRD].point.pos[_Z]) + geom.glShift[_Z];

	Real mul1 = mesh_local.vertices[_THIRD].point.pos[_X] * mesh_local.vertices[_SECOND].point.pos[_Y];
	Real mul2 = mesh_local.vertices[_THIRD].point.pos[_Y] * mesh_local.vertices[_SECOND].point.pos[_X];

	if (mul1 == mul2)
		return false;

	Real refTri[9] = { 0,0,0,1,1,0,1,0,0 };

	geom.loRot[0] = (refTri[_X3] * mesh_local.vertices[_SECOND].point.pos[_Y] - refTri[_X2] * mesh_local.vertices[_THIRD].point.pos[_Y]) / (mul1 - mul2);
	geom.loRot[1] = (refTri[_X3] * mesh_local.vertices[_SECOND].point.pos[_X] - refTri[_X2] * mesh_local.vertices[_THIRD].point.pos[_X]) / (-mul1 + mul2);
	geom.loRot[2] = (refTri[_Y3] * mesh_local.vertices[_SECOND].point.pos[_Y] - refTri[_Y2] * mesh_local.vertices[_THIRD].point.pos[_Y]) / (mul1 - mul2);
	geom.loRot[3] = (refTri[_Y3] * mesh_local.vertices[_SECOND].point.pos[_X] - refTri[_Y2] * mesh_local.vertices[_THIRD].point.pos[_X]) / (-mul1 + mul2);

	if ((geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2]) == 0)
		return false;
	return true;

}

bool ophTri::calFrequencyTerm(Point* frequency, Point* fl, Real *freqTermX, Real *freqTermY, geometric& geom, Real lambda)
{
	// p.s. only 1 channel
	const long long int pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];

	Real w = 1 / lambda;
	Real ww = w * w;

	Real det = geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2];
	Real divDet = 1 / det;

	Real invLoRot[4];
	invLoRot[0] = divDet * geom.loRot[3];
	invLoRot[1] = -divDet * geom.loRot[2];
	invLoRot[2] = -divDet * geom.loRot[1];
	invLoRot[3] = divDet * geom.loRot[0];

	Real carrierWaveLoc[3];
	Real glRot[9];
	memcpy(glRot, geom.glRot, sizeof(glRot));
	memcpy(carrierWaveLoc, carrierWave, sizeof(carrierWaveLoc));

#ifdef _OPENMP
#pragma omp parallel for firstprivate(w, ww, glRot, carrierWaveLoc, invLoRot)
#endif
	for (int i = 0; i < pnXY; i++)
	{
		fl[i].pos[_X] = glRot[0] * frequency[i].pos[_X] + glRot[1] * frequency[i].pos[_Y] + glRot[2] * frequency[i].pos[_Z];
		fl[i].pos[_Y] = glRot[3] * frequency[i].pos[_X] + glRot[4] * frequency[i].pos[_Y] + glRot[5] * frequency[i].pos[_Z];
		fl[i].pos[_Z] = sqrt(ww - fl[i].pos[_X] * fl[i].pos[_X] - fl[i].pos[_Y] * fl[i].pos[_Y]);

		Real flxShifted = fl[i].pos[_X] - w * (glRot[0] * carrierWaveLoc[_X] + glRot[1] * carrierWaveLoc[_Y] + glRot[2] * carrierWaveLoc[_Z]);
		Real flyShifted = fl[i].pos[_Y] - w * (glRot[3] * carrierWaveLoc[_X] + glRot[4] * carrierWaveLoc[_Y] + glRot[5] * carrierWaveLoc[_Z]);

		freqTermX[i] = invLoRot[0] * flxShifted + invLoRot[1] * flyShifted;
		freqTermY[i] = invLoRot[2] * flxShifted + invLoRot[3] * flyShifted;
	}
	return true;
}

bool ophTri::calFrequencyTerm(Real** frequency, Real** fl, Real *freqTermX, Real *freqTermY, geometric& geom)
{
	// p.s. only 1 channel
	const long long int pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];

	Real waveLength = context_.wave_length[0];
	Real w = 1 / waveLength;
	Real ww = w * w;

	Real det = geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2];
	Real divDet = 1 / det;

	Real invLoRot[4];
	invLoRot[0] = divDet * geom.loRot[3];
	invLoRot[1] = -divDet * geom.loRot[2];
	invLoRot[2] = -divDet * geom.loRot[1];
	invLoRot[3] = divDet * geom.loRot[0];

	Real carrierWaveLoc[3];
	Real glRot[9];
	memcpy(glRot, geom.glRot, sizeof(glRot));
	memcpy(carrierWaveLoc, carrierWave, sizeof(carrierWaveLoc));

#ifdef _OPENMP
#pragma omp parallel for firstprivate(w, ww, glRot, carrierWaveLoc, invLoRot)
#endif
	for (int i = 0; i < pnXY; i++) {
		Real flxShifted;
		Real flyShifted;

		fl[_X][i] = glRot[0] * frequency[_X][i] + glRot[1] * frequency[_Y][i] + glRot[2] * frequency[_Z][i];
		fl[_Y][i] = glRot[3] * frequency[_X][i] + glRot[4] * frequency[_Y][i] + glRot[5] * frequency[_Z][i];
		fl[_Z][i] = sqrt(ww - fl[_X][i] * fl[_X][i] - fl[_Y][i] * fl[_Y][i]);
		flxShifted = fl[_X][i] - w * (glRot[0] * carrierWaveLoc[_X] + glRot[1] * carrierWaveLoc[_Y] + glRot[2] * carrierWaveLoc[_Z]);
		flyShifted = fl[_Y][i] - w * (glRot[3] * carrierWaveLoc[_X] + glRot[4] * carrierWaveLoc[_Y] + glRot[5] * carrierWaveLoc[_Z]);

		freqTermX[i] = invLoRot[0] * flxShifted + invLoRot[1] * flyShifted;
		freqTermY[i] = invLoRot[2] * flxShifted + invLoRot[3] * flyShifted;
	}
	return true;
}

void ophTri::refAS_Flat(vec3 no, Point* frequency, Face mesh, Real* freqTermX, Real* freqTermY, geometric& geom, Real lambda)
{
	const Real ssX = context_.ss[_X] = context_.pixel_number[_X] * context_.pixel_pitch[_X];
	const Real ssY = context_.ss[_Y] = context_.pixel_number[_Y] * context_.pixel_pitch[_Y];
	const long long int pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];

	vec3 n = no / norm(no);

	Complex<Real> shadingFactor;
	Real PI2 = M_PI * 2;
	Real sqPI2 = PI2 * PI2;

	Complex<Real> term1(0, 0);
	term1[_IM] = -PI2 / lambda * (
		carrierWave[_X] * (geom.glRot[0] * geom.glShift[_X] + geom.glRot[3] * geom.glShift[_Y] + geom.glRot[6] * geom.glShift[_Z])
		+ carrierWave[_Y] * (geom.glRot[1] * geom.glShift[_X] + geom.glRot[4] * geom.glShift[_Y] + geom.glRot[7] * geom.glShift[_Z])
		+ carrierWave[_Z] * (geom.glRot[2] * geom.glShift[_X] + geom.glRot[5] * geom.glShift[_Y] + geom.glRot[8] * geom.glShift[_Z])
		);

	if (illumination[_X] == 0 && illumination[_Y] == 0 && illumination[_Z] == 0) {
		shadingFactor = exp(term1);
	}
	else {
		vec3 normIllu = illumination / norm(illumination);
		shadingFactor = (2 * (n[_X] * normIllu[_X] + n[_Y] * normIllu[_Y] + n[_Z] * normIllu[_Z]) + 0.3) * exp(term1);
		if (shadingFactor[_RE] * shadingFactor[_RE] + shadingFactor[_IM] * shadingFactor[_IM] < 0)
			shadingFactor = 0;
	}
	Real dfx = 1 / ssX;
	Real dfy = 1 / ssY;

	if (occlusion) {
		Complex<Real> term1(0, 0);
		Real dfxy = dfx * dfy;

#ifdef _OPENMP
#pragma omp parallel for firstprivate(PI2, dfxy, mesh, term1)
#endif
		for (int i = 0; i < pnXY; i++) {
			term1[_IM] = PI2 * (
				frequency[i].pos[_X] * mesh.vertices[_FIRST].point.pos[_X] + 
				frequency[i].pos[_Y] * mesh.vertices[_FIRST].point.pos[_Y] + 
				frequency[i].pos[_Z] * mesh.vertices[_FIRST].point.pos[_Z]);
			rearAS[i] = angularSpectrum[i] * exp(term1) * dfxy;
		}

		refASInner_flat(freqTermX, freqTermY);			// refAS main function including texture mapping 

		if (randPhase) {
#ifdef _OPENMP
#pragma omp parallel for firstprivate(PI2, shadingFactor)
#endif
			for (int i = 0; i < pnXY; i++) {
				Complex<Real> phase(0, 0);
				phase[_IM] = PI2 * rand(0.0, 1.0, i);
				convol[i] = shadingFactor * exp(phase) - rearAS[i];
			}
			conv_fft2_scale(refAS, convol, refAS, context_.pixel_number);
		}
		else {
			conv_fft2_scale(rearAS, refAS, convol, context_.pixel_number);
#ifdef _OPENMP
#pragma omp parallel for firstprivate(shadingFactor)
#endif
			for (int i = 0; i < pnXY; i++) {
				refAS[i] = refAS[i] * shadingFactor - convol[i];
			}
		}
	}
	else {
		refASInner_flat(freqTermX, freqTermY);			// refAS main function including texture mapping

		if (randPhase == true) {
			Complex<Real> phase(0, 0);
#ifdef _OPENMP
#pragma omp parallel for firstprivate(PI2, shadingFactor)
#endif
			for (int i = 0; i < pnXY; i++) {
				Real randVal = rand(0.0, 1.0, i);
				phase[_IM] = PI2 * randVal;
				phaseTerm[i] = shadingFactor * exp(phase);
			}
			conv_fft2_scale(refAS, phaseTerm, refAS, context_.pixel_number);
		}
		else {
#ifdef _OPENMP
#pragma omp parallel for firstprivate(shadingFactor)
#endif
			for (int i = 0; i < pnXY; i++) {
				refAS[i] *= shadingFactor;
			}
		}
	}
}

void ophTri::refASInner_flat(Real* freqTermX, Real* freqTermY)
{
	const long long int pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];
	Real PI2 = M_PI * 2.0;

//#ifndef _OPENMP
//#pragma omp parallel for private(i) 
//#endif
	for (int i = 0; i < pnXY; i++) {
		if (textureMapping == true) {
			refAS[i] = 0;
			for (int idxFy = -texture.dim[_Y] / 2; idxFy < texture.dim[_Y] / 2; idxFy++) {
				for (int idxFx = -texture.dim[_X] / 2; idxFx < texture.dim[_X] / 2; idxFy++) {
					textFreqX = idxFx * texture.freq;
					textFreqY = idxFy * texture.freq;

					tempFreqTermX[i] = freqTermX[i] - textFreqX;
					tempFreqTermY[i] = freqTermY[i] - textFreqY;

					if (tempFreqTermX[i] == -tempFreqTermY[i] && tempFreqTermY[i] != 0.0) {
						refTerm1[_IM] = PI2 * tempFreqTermY[i];
						refTerm2[_IM] = 1.0;
						refTemp = ((Complex<Real>)1.0 - exp(refTerm1)) / (4.0 * M_PI*M_PI*tempFreqTermY[i] * tempFreqTermY[i]) + refTerm2 / (PI2*tempFreqTermY[i]);
						refAS[i] = refAS[i] + textFFT[idxFx + texture.dim[_X] / 2 + (idxFy + texture.dim[_Y] / 2)*texture.dim[_X]] * refTemp;

					}
					else if (tempFreqTermX[i] == tempFreqTermY[i] && tempFreqTermX[i] == 0.0) {
						refTemp = (Real)(1.0 / 2.0);
						refAS[i] = refAS[i] + textFFT[idxFx + texture.dim[_X] / 2 + (idxFy + texture.dim[_Y] / 2)*texture.dim[_X]] * refTemp;
					}
					else if (tempFreqTermX[i] != 0.0 && tempFreqTermY[i] == 0.0) {
						refTerm1[_IM] = -PI2 * tempFreqTermX[i];
						refTerm2[_IM] = 1.0;
						refTemp = (exp(refTerm1) - (Complex<Real>)1.0) / (PI2*tempFreqTermX[i] * PI2*tempFreqTermX[i]) + (refTerm2 * exp(refTerm1)) / (PI2*tempFreqTermX[i]);
						refAS[i] = refAS[i] + textFFT[idxFx + texture.dim[_X] / 2 + (idxFy + texture.dim[_Y] / 2)*texture.dim[_X]] * refTemp;
					}
					else if (tempFreqTermX[i] == 0.0 && tempFreqTermY[i] != 0.0) {
						refTerm1[_IM] = PI2 * tempFreqTermY[i];
						refTerm2[_IM] = 1.0;
						refTemp = ((Complex<Real>)1.0 - exp(refTerm1)) / (4.0 * M_PI*M_PI*tempFreqTermY[i] * tempFreqTermY[i]) - refTerm2 / (PI2*tempFreqTermY[i]);
						refAS[i] = refAS[i] + textFFT[idxFx + texture.dim[_X] / 2 + (idxFy + texture.dim[_Y] / 2)*texture.dim[_X]] * refTemp;
					}
					else {
						refTerm1[_IM] = -PI2 * tempFreqTermX[i];
						refTerm2[_IM] = -PI2 * (tempFreqTermX[i] + tempFreqTermY[i]);
						refTemp = (exp(refTerm1) - (Complex<Real>)1.0) / (4.0 * M_PI*M_PI*tempFreqTermX[i] * tempFreqTermY[i]) + ((Complex<Real>)1.0 - exp(refTerm2)) / (4.0 * M_PI*M_PI*tempFreqTermY[i] * (tempFreqTermX[i] + tempFreqTermY[i]));
						refAS[i] = refAS[i] + textFFT[idxFx + texture.dim[_X] / 2 + (idxFy + texture.dim[_Y] / 2)*texture.dim[_X]] * refTemp;
					}
				}
			}
		}
		else {
			if (freqTermX[i] == -freqTermY[i] && freqTermY[i] != 0.0) {
				refTerm1[_IM] = PI2 * freqTermY[i];
				refTerm2[_IM] = 1.0;
				refAS[i] = ((Complex<Real>)1.0 - exp(refTerm1)) / (4.0 * M_PI*M_PI*freqTermY[i] * freqTermY[i]) + refTerm2 / (PI2*freqTermY[i]);
			}
			else if (freqTermX[i] == freqTermY[i] && freqTermX[i] == 0.0) {
				refAS[i] = (Real)(1.0 / 2.0);
			}
			else if (freqTermX[i] != 0 && freqTermY[i] == 0.0) {
				refTerm1[_IM] = -PI2 * freqTermX[i];
				refTerm2[_IM] = 1.0;
				refAS[i] = (exp(refTerm1) - (Complex<Real>)1.0) / (PI2 * freqTermX[i] * PI2 * freqTermX[i]) + (refTerm2 * exp(refTerm1)) / (PI2*freqTermX[i]);
			}
			else if (freqTermX[i] == 0 && freqTermY[i] != 0.0) {
				refTerm1[_IM] = PI2 * freqTermY[i];
				refTerm2[_IM] = 1.0;
				refAS[i] = ((Complex<Real>)1.0 - exp(refTerm1)) / (PI2 * PI2 * freqTermY[i] * freqTermY[i]) - refTerm2 / (PI2*freqTermY[i]);
			}
			else {
				refTerm1[_IM] = -PI2 * freqTermX[i];
				refTerm2[_IM] = -PI2 * (freqTermX[i] + freqTermY[i]);
				refAS[i] = (exp(refTerm1) - (Complex<Real>)1.0) / (PI2 * PI2 * freqTermX[i] * freqTermY[i]) + ((Complex<Real>)1.0 - exp(refTerm2)) / (4.0 * M_PI*M_PI*freqTermY[i] * (freqTermX[i] + freqTermY[i]));
			}
		}
	}
}

bool ophTri::refAS_Continuous(uint n, Real* freqTermX, Real* freqTermY)
{
	const int pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];

	av = vec3(0.0, 0.0, 0.0);
	av[0] = nv[3 * n + 0][0] * illumination[0] + nv[3 * n + 0][1] * illumination[1] + nv[3 * n + 0][2] * illumination[2] + 0.1;
	av[2] = nv[3 * n + 1][0] * illumination[0] + nv[3 * n + 1][1] * illumination[1] + nv[3 * n + 1][2] * illumination[2] + 0.1;
	av[1] = nv[3 * n + 2][0] * illumination[0] + nv[3 * n + 2][1] * illumination[1] + nv[3 * n + 2][2] * illumination[2] + 0.1;

	Complex<Real> refTerm1(0, 0);
	Complex<Real> refTerm2(0, 0);
	Complex<Real> refTerm3(0, 0);
	Complex<Real> D1, D2, D3;

	for (int i = 0; i < pnXY; i++) {
		if (freqTermX[i] == 0.0 && freqTermY[i] == 0.0) {
			D1(1.0 / 3.0, 0);
			D2(1.0 / 5.0, 0);
			D3(1.0 / 2.0, 0);
		}

		else if (freqTermX[i] == 0.0 && freqTermY[i] != 0.0) {
			refTerm1[_IM] = -2 * M_PI*freqTermY[i];
			refTerm2[_IM] = 1;

			D1 = (refTerm1 - 1.0)*refTerm1.exp() / (8.0 * M_PI*M_PI*M_PI*freqTermY[i] * freqTermY[i] * freqTermY[i])
				- refTerm1 / (4.0 * M_PI*M_PI*M_PI*freqTermY[i] * freqTermY[i] * freqTermY[i]);
			D2 = -(M_PI*freqTermY[i] + refTerm2) / (4.0 * M_PI*M_PI*M_PI*freqTermY[i] * freqTermY[i] * freqTermY[i])*exp(refTerm1)
				+ refTerm1 / (8.0 * M_PI*M_PI*M_PI*freqTermY[i] * freqTermY[i] * freqTermY[i]);
			D3 = exp(refTerm1) / (2.0 * M_PI*freqTermY[i]) + (1.0 - refTerm2) / (2.0 * M_PI*freqTermY[i]);
		}
		else if (freqTermX[i] != 0.0 && freqTermY[i] == 0.0) {
			refTerm1[_IM] = 4.0 * M_PI*M_PI*freqTermX[i] * freqTermX[i];
			refTerm2[_IM] = 1.0;
			refTerm3[_IM] = 2.0 * M_PI*freqTermX[i];

			D1 = (refTerm1 + 4.0 * M_PI*freqTermX[i] - 2.0 * refTerm2) / (8.0 * M_PI*M_PI*M_PI*freqTermY[i] * freqTermY[i] * freqTermY[i])*exp(-refTerm3)
				+ refTerm2 / (4.0 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * freqTermX[i]);
			D2 = 1.0 / 2.0 * D1;
			D3 = ((refTerm3 + 1.0)*exp(-refTerm3) - 1.0) / (4.0 * M_PI*M_PI*freqTermX[i] * freqTermX[i]);
		}
		else if (freqTermX[i] == -freqTermY[i]) {
			refTerm1[_IM] = 1.0;
			refTerm2[_IM] = 2.0 * M_PI*freqTermX[i];
			refTerm3[_IM] = 2.0 * M_PI*M_PI*freqTermX[i] * freqTermX[i];

			D1 = (-2.0 * M_PI*freqTermX[i] + refTerm1) / (8.0 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * freqTermX[i])*exp(-refTerm2)
				- (refTerm3 + refTerm1) / (8.0 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * freqTermX[i]);
			D2 = (-refTerm1) / (8.0 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * freqTermX[i])*exp(-refTerm2)
				+ (-refTerm3 + refTerm1 + 2.0 * M_PI*freqTermX[i]) / (8.0 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * freqTermX[i]);
			D3 = (-refTerm1) / (4.0 * M_PI*M_PI*freqTermX[i] * freqTermX[i])*exp(-refTerm2)
				+ (-refTerm2 + 1.0) / (4.0 * M_PI*M_PI*freqTermX[i] * freqTermX[i]);
		}
		else {
			refTerm1[_IM] = -2.0 * M_PI*(freqTermX[i] + freqTermY[i]);
			refTerm2[_IM] = 1.0;
			refTerm3[_IM] = -2.0 * M_PI*freqTermX[i];

			D1 = exp(refTerm1)*(refTerm2 - 2.0 * M_PI*(freqTermX[i] + freqTermY[i])) / (8 * M_PI*M_PI*M_PI*freqTermY[i] * (freqTermX[i] + freqTermY[i])*(freqTermX[i] + freqTermY[i]))
				+ exp(refTerm3)*(2.0 * M_PI*freqTermX[i] - refTerm2) / (8.0 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * freqTermY[i])
				+ ((2.0 * freqTermX[i] + freqTermY[i])*refTerm2) / (8.0 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * (freqTermX[i] + freqTermY[i])*(freqTermX[i] + freqTermY[i]));
			D2 = exp(refTerm1)*(refTerm2*(freqTermX[i] + 2.0 * freqTermY[i]) - 2.0 * M_PI*freqTermY[i] * (freqTermX[i] + freqTermY[i])) / (8.0 * M_PI*M_PI*M_PI*freqTermY[i] * freqTermY[i] * (freqTermX[i] + freqTermY[i])*(freqTermX[i] + freqTermY[i]))
				+ exp(refTerm3)*(-refTerm2) / (8.0 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermY[i] * freqTermY[i])
				+ refTerm2 / (8.0 * M_PI*M_PI*M_PI*freqTermX[i] * (freqTermX[i] + freqTermY[i])* (freqTermX[i] + freqTermY[i]));
			D3 = -exp(refTerm1) / (4.0 * M_PI*M_PI*freqTermY[i] * (freqTermX[i] + freqTermY[i]))
				+ exp(refTerm3) / (4.0 * M_PI*M_PI*freqTermX[i] * freqTermY[i])
				- 1.0 / (4.0 * M_PI*M_PI*freqTermX[i] * (freqTermX[i] + freqTermY[i]));
		}
		refAS[i] = (av[1] - av[0])*D1 + (av[2] - av[1])*D2 + av[0] * D3;
	}
	if (randPhase == true) {
		Complex<Real> phase(0, 0);
		Real PI2 = M_PI * 2.0;
		for (int i = 0; i < pnXY; i++) {
			Real randVal = rand(0.0, 1.0, i);
			phase[_IM] = PI2 * randVal;
			phaseTerm[i] = exp(phase);
		}

		conv_fft2_scale(refAS, phaseTerm, convol, context_.pixel_number);
	}

	return true;
}

bool ophTri::refToGlobal(Complex<Real> *dst, Point* frequency, Point* fl, geometric& geom)
{
	const long long int pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];

	Real PI2 = M_PI * 2;
	Real det = geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2];

	if (det == 0)
		return false;
	if (det < 0)
		det = -det;


	geometric g;
	memcpy(&g, &geom, sizeof(geometric));

#ifdef _OPENMP
#pragma omp parallel for firstprivate(det, g)
#endif
	for (int i = 0; i < pnXY; i++)
	{
		Complex<Real> term1(0, 0);
		Complex<Real> term2(0, 0);

		if (frequency[i].pos[_Z] == 0)
			term2 = 0;
		else
		{
			term1[_IM] = PI2 * (fl[i].pos[_X] * g.glShift[_X] + fl[i].pos[_Y] * g.glShift[_Y] + fl[i].pos[_Z] * g.glShift[_Z]);
			term2 = refAS[i] / det * fl[i].pos[_Z] / frequency[i].pos[_Z] * exp(term1);
		}
		if (abs(term2) > MIN_DOUBLE)
			; // do nothing
		else
			term2 = 0;

		dst[i] += term2;

		//complex_H[0][i] += term2;
	}

	return true;
}

bool ophTri::refToGlobal(Real** frequency, Real** fl, geometric& geom)
{
	const int pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];

	Real PI2 = M_PI * 2;
	Real det = geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2];

	if (det == 0)
		return false;
	if (det < 0)
		det = -det;
	
	geometric g;
	memcpy(&g, &geom, sizeof(geometric));
	Complex<Real> term1(0, 0);
	Complex<Real> term2(0, 0);

#ifdef _OPENMP
#pragma omp parallel for firstprivate(det, g, term1, term2)
#endif
	for (int i = 0; i < pnXY; i++) {
		if (frequency[_Z][i] == 0)
			term2 = 0;
		else {
			term1[_IM] = PI2 * (fl[_X][i] * g.glShift[_X] + fl[_Y][i] * g.glShift[_Y] + fl[_Z][i] * g.glShift[_Z]);
			term2 = refAS[i] / det * fl[_Z][i] / frequency[_Z][i] * exp(term1);// *phaseTerm[i];
		}
		if (abs(term2) > MIN_DOUBLE) {}
		else { term2 = 0; }
		//angularSpectrum[i] += term2;
		complex_H[0][i] += term2;
	}

	return true;
}

void ophTri::reconTest(const char* fname)
{
	const int pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];

	Complex<Real>* recon = new Complex<Real>[pnXY];
	Fresnel_FFT(complex_H[0], recon, 0, context_.shift[_Z]);
	ophGen::encoding(ENCODE_AMPLITUDE, recon, nullptr);

	normalize();
	save(fname, 8, nullptr, m_vecEncodeSize[_X], m_vecEncodeSize[_Y]);

	delete[] recon;
}
// correct the output scale of the  ophGen::conv_fft2 
void ophTri::conv_fft2_scale(Complex<Real>* src1, Complex<Real>* src2, Complex<Real>* dst, ivec2 size)
{
	const int N = size[_X] * size[_Y];
	if (N <= 0) return;


	Complex<Real>* src1FT = new Complex<Real>[N];
	Complex<Real>* src2FT = new Complex<Real>[N];
	Complex<Real>* dstFT = new Complex<Real>[N];


	fft2(src1, src1FT, size[_X], size[_Y], OPH_FORWARD, (bool)OPH_ESTIMATE);

	fft2(src2, src2FT, size[_X], size[_Y], OPH_FORWARD, (bool)OPH_ESTIMATE);

	Real scale = (Real)N * (Real)N;
	for (int i = 0; i < N; i++)
		dstFT[i] = src1FT[i] * src2FT[i] * scale;

	fft2(dstFT, dst, size[_X], size[_Y], OPH_BACKWARD, (bool)OPH_ESTIMATE);

	delete[] src1FT;
	delete[] src2FT;
	delete[] dstFT;
}

void ophTri::prepareMeshData()
{
	auto begin = CUR_TIME;
	const int N = meshData->n_faces;
	const int N3 = N * 3;


	Real x_max, x_min, y_max, y_min, z_max, z_min;
	x_max = y_max = z_max = MIN_DOUBLE;
	x_min = y_min = z_min = MAX_DOUBLE;

	for (int i = 0; i < N3; i++)
	{
		int idxFace = i / 3;
		int idxPoint = i % 3;

		if (x_max < triMeshArray[idxFace].vertices[idxPoint].point.pos[_X]) x_max = triMeshArray[idxFace].vertices[idxPoint].point.pos[_X];
		if (x_min > triMeshArray[idxFace].vertices[idxPoint].point.pos[_X]) x_min = triMeshArray[idxFace].vertices[idxPoint].point.pos[_X];
		if (y_max < triMeshArray[idxFace].vertices[idxPoint].point.pos[_Y]) y_max = triMeshArray[idxFace].vertices[idxPoint].point.pos[_Y];
		if (y_min > triMeshArray[idxFace].vertices[idxPoint].point.pos[_Y]) y_min = triMeshArray[idxFace].vertices[idxPoint].point.pos[_Y];
		if (z_max < triMeshArray[idxFace].vertices[idxPoint].point.pos[_Z]) z_max = triMeshArray[idxFace].vertices[idxPoint].point.pos[_Z];
		if (z_min > triMeshArray[idxFace].vertices[idxPoint].point.pos[_Z]) z_min = triMeshArray[idxFace].vertices[idxPoint].point.pos[_Z];
	}

	Real x_cen = (x_max + x_min) / 2;
	Real y_cen = (y_max + y_min) / 2;
	Real z_cen = (z_max + z_min) / 2;
	vec3 cen(x_cen, y_cen, z_cen);

	Real x_del = x_max - x_min;
	Real y_del = y_max - y_min;
	Real z_del = z_max - z_min;

	Real del = maxOfArr({ x_del, y_del, z_del });

	vec3 shift = getContext().shift;
	vec3 locObjSize = objSize;
#ifdef _OPENMP
#pragma omp parallel for firstprivate(cen, del, locObjSize, shift)
#endif
	for (int i = 0; i < N3; i++)
	{
		int idxFace = i / 3;
		int idxPoint = i % 3;
		scaledMeshData[idxFace].vertices[idxPoint].point.pos[_X] = (triMeshArray[idxFace].vertices[idxPoint].point.pos[_X] - cen[_X]) / del * locObjSize[_X] + shift[_X];
		scaledMeshData[idxFace].vertices[idxPoint].point.pos[_Y] = (triMeshArray[idxFace].vertices[idxPoint].point.pos[_Y] - cen[_Y]) / del * locObjSize[_Y] + shift[_Y];
		scaledMeshData[idxFace].vertices[idxPoint].point.pos[_Z] = (triMeshArray[idxFace].vertices[idxPoint].point.pos[_Z] - cen[_Z]) / del * locObjSize[_Z] + shift[_Z];
	}

	LOG("<END> %s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
}


void ophTri::encoding(unsigned int ENCODE_FLAG)
{
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint nChannel = context_.waveNum;
	Complex<Real>** dst = new Complex<Real>*[nChannel];
	for (uint ch = 0; ch < nChannel; ch++) {
		dst[ch] = new Complex<Real>[pnX * pnY];
		//fft2(context_.pixel_number, nullptr, OPH_BACKWARD);
		fft2(complex_H[ch], dst[ch], pnX, pnY, OPH_BACKWARD, true);
		ophGen::encoding(ENCODE_FLAG, dst[ch], m_lpEncoded[ch]);
	}

	for (uint ch = 0; ch < nChannel; ch++)
		delete[] dst[ch];
	delete[] dst;
}