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
	, is_CPU(true)
	, is_ViewingWindow(false)
	, scaledMeshData(nullptr)
	, normalizedMeshData(nullptr)
	, angularSpectrum(nullptr)
	, bSinglePrecision(false)
	, refAS(nullptr)
	, ASTerm(nullptr)
	, randTerm(nullptr)
	, phaseTerm(nullptr)
	, convol(nullptr)
	, no(nullptr)
	, na(nullptr)
	, nv(nullptr)
{
	LOG("*** MESH : BUILD DATE: %s %s ***\n\n", __DATE__, __TIME__);
}

void ophTri::setMode(bool isCPU)
{
	is_CPU = isCPU;
}

void ophTri::setViewingWindow(bool is_ViewingWindow)
{
	this->is_ViewingWindow = is_ViewingWindow;
}

uint ophTri::loadMeshText(const char* fileName)
{

	cout << "Mesh Text File Load..." << endl;

	ifstream file;
	file.open(fileName);

	if (!file) {
		cout << "Open failed - no such file" << endl;
		cin.get();
		return 0;
	}

	triMeshArray = new Real[9 * 10000];

	Real data;
	uint num_data;

	num_data = 0;
	do {
		file >> data;
		triMeshArray[num_data] = data;
		num_data++;
	} while (file.get() != EOF);

	meshData->n_faces = num_data / 9;
	triMeshArray[meshData->n_faces * 9] = EOF;

	return 1;
}

bool ophTri::loadMeshData(const char* fileName, const char* ext)
{
	meshData = new OphMeshData;
	cout << "ext = " << ext << endl;

	if (!strcmp(ext, "txt")) {
		cout << "Text File.." << endl;
		if (loadMeshText(fileName))
			cout << "Mesh Data Load Finished.." << endl;
		else
			cout << "Mesh Data Load Failed.." << endl;
	}
	else if (!strcmp(ext, "ply")) {
		cout << "PLY File.." << endl;
		PLYparser meshPLY;
		if (meshPLY.loadPLY(fileName, meshData->n_faces, meshData->color_channels, &meshData->face_idx, &meshData->vertex, &meshData->color))
			cout << "Mesh Data Load Finished.." << endl;
		else
		{
			cout << "Mesh Data Load Failed.." << endl;
			return false;
		}
	}
	else {
		cout << "Error: Mesh data must be .txt or .ply" << endl;
		return false;
	}
	meshData->n_faces /= 3;
	triMeshArray = meshData->vertex;

	return true;
}

bool ophTri::readConfig(const char* fname)
{
	if (!ophGen::readConfig(fname))
		return false;

	LOG("Reading....%s...", fname);

	auto start = CUR_TIME;

	using namespace tinyxml2;
	/*XML parsing*/
	tinyxml2::XMLDocument xml_doc;
	XMLNode *xml_node;

	if (!checkExtension(fname, ".xml"))
	{
		LOG("file's extension is not 'xml'\n");
		return false;
	}
	if (xml_doc.LoadFile(fname) != XML_SUCCESS)
	{
		LOG("Failed to load file \"%s\"\n", fname);
		return false;
	}

	xml_node = xml_doc.FirstChild();

	// about object
	auto next = xml_node->FirstChildElement("ScaleX");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&objSize[_X]))
		return false;
	next = xml_node->FirstChildElement("ScaleY");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&objSize[_Y]))
		return false;
	next = xml_node->FirstChildElement("ScaleZ");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&objSize[_Z]))
		return false;

	objShift = context_.shift;
	/*
	next = xml_node->FirstChildElement("ObjectShiftX");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&objShift[_X]))
		return false;
	next = xml_node->FirstChildElement("ObjectShiftY");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&objShift[_Y]))
		return false;
	next = xml_node->FirstChildElement("ObjectShiftZ");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&objShift[_Z]))
		return false;
	*/
	next = xml_node->FirstChildElement("LampDirectionX");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&illumination[_X]))
		return false;
	next = xml_node->FirstChildElement("LampDirectionY");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&illumination[_Y]))
		return false;
	next = xml_node->FirstChildElement("LampDirectionZ");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&illumination[_Z]))
		return false;

	auto end = CUR_TIME;
	auto during = ((chrono::duration<Real>)(end - start)).count();
	LOG("%lf (s)..done\n", during);

	initialize();
	return true;
}


//int ophTri::saveAsOhc(const char * fname)
//{
//	setPixelNumberOHC(getEncodeSize());
//
//	Openholo::saveAsOhc(fname);
//	
//	return 0;
//}

void ophTri::initializeAS()
{
	const uint pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];
	const int N = meshData->n_faces;

	if (normalizedMeshData) {
		delete[] normalizedMeshData;
		normalizedMeshData = nullptr;
	}
	normalizedMeshData = new Real[N * 9];
	memset(normalizedMeshData, 0, sizeof(Real) * N * 9);

	if (scaledMeshData) {
		delete[] scaledMeshData;
		scaledMeshData = nullptr;
	}
	scaledMeshData = new Real[N * 9];
	memset(scaledMeshData, 0, sizeof(Real) * N * 9);

	if (angularSpectrum) {
		delete[] angularSpectrum;
		angularSpectrum = nullptr;
	}
	angularSpectrum = new Complex<Real>[pnXY];
	memset(angularSpectrum, 0, sizeof(Complex<Real>) * pnXY);

	if (refAS) {
		delete[] refAS;
		refAS = nullptr;
	}
	refAS = new Complex<Real>[pnXY];
	memset(refAS, 0, sizeof(Complex<Real>) * pnXY);

	if (ASTerm) {
		delete[] ASTerm;
		ASTerm = nullptr;
	}
	ASTerm = new Complex<Real>[pnXY];
	memset(ASTerm, 0, sizeof(Complex<Real>) * pnXY);

	if (randTerm) {
		delete[] randTerm;
		randTerm = nullptr;
	}
	randTerm = new Complex<Real>[pnXY];
	memset(randTerm, 0, sizeof(Complex<Real>) * pnXY);


	if (phaseTerm) {
		delete[] phaseTerm;
		phaseTerm = nullptr;
	}
	phaseTerm = new Complex<Real>[pnXY];
	memset(phaseTerm, 0, sizeof(Complex<Real>) * pnXY);


	if (convol) {
		delete[] convol;
		convol = nullptr;
	}
	convol = new Complex<Real>[pnXY];
	memset(convol, 0, sizeof(Complex<Real>) * pnXY);


	if (no) {
		delete[] no;
		no = nullptr;
	}
	no = new vec3[N];
	memset(no, 0, sizeof(vec3) * N);


	if (na) {
		delete[] na;
		na = nullptr;
	}
	na = new vec3[N];
	memset(na, 0, sizeof(vec3) * N);


	if (nv) {
		delete[] nv;
		nv = nullptr;
	}
	nv = new vec3[N * 3];
	memset(nv, 0, sizeof(vec3) * N * 3);
}


void ophTri::objNormCenter()
{
	int N = meshData->n_faces;
	int N3 = N * 3;

	Real* x_point = new Real[N3];
	Real* y_point = new Real[N3];
	Real* z_point = new Real[N3];

	int i;
#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
	for (i = 0; i < N3; i++) {
		int idx = i * 3;
		x_point[i] = triMeshArray[idx + _X];
		y_point[i] = triMeshArray[idx + _Y];
		z_point[i] = triMeshArray[idx + _Z];
	}
	
	Real x_max = maxOfArr(x_point, N3);
	Real x_min = minOfArr(x_point, N3);
	Real y_max = maxOfArr(y_point, N3);
	Real y_min = minOfArr(y_point, N3);
	Real z_max = maxOfArr(z_point, N3);
	Real z_min = minOfArr(z_point, N3);

	Real x_cen = (x_max + x_min) / 2;
	Real y_cen = (y_max + y_min) / 2;
	Real z_cen = (z_max + z_min) / 2;

	Real x_del = x_max - x_min;
	Real y_del = y_max - y_min;
	Real z_del = z_max - z_min;

	Real del = maxOfArr({ x_del, y_del, z_del });

	cout << "center: " << x_cen << ", " << y_cen << ", " << z_cen << endl;

	Real* centered = new Real[N * 9];

#ifdef _OPENMP
#pragma omp parallel for private(i) firstprivate(x_cen) firstprivate(y_cen) firstprivate(z_cen) firstprivate(del)
#endif
	for (i = 0; i < N3; i++) {
		int idx = i * 3;
		normalizedMeshData[idx + _X] = (x_point[i] - x_cen) / del;
		normalizedMeshData[idx + _Y] = (y_point[i] - y_cen) / del;
		normalizedMeshData[idx + _Z] = (z_point[i] - z_cen) / del;
	}
	cout << "normalizedMeshData: " << normalizedMeshData[_X] << ", " << normalizedMeshData[_Y] << ", " << normalizedMeshData[_Z] << endl;

	delete[] x_point, y_point, z_point;
}


void ophTri::objScaleShift()
{
	int N = meshData->n_faces;
	objNormCenter();
	
	vec3 shift = getContext().shift;

	int i;
#ifdef _OPENMP
	int num_threads;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads(); // get number of Multi Threading
#pragma omp for private(i)
#endif
		for (i = 0; i < N * 3; i++) {
			int idx = i * 3;
			scaledMeshData[idx + _X] = normalizedMeshData[idx + _X] * objSize[_X] + shift[_X];
			scaledMeshData[idx + _Y] = normalizedMeshData[idx + _Y] * objSize[_Y] + shift[_Y];
			scaledMeshData[idx + _Z] = normalizedMeshData[idx + _Z] * objSize[_Z] + shift[_Z];
		}
#ifdef _OPENMP
	}
#endif

	delete[] normalizedMeshData;

	cout << "Object Scaling and Shifting Finishied.." << endl;

#ifdef _OPENMP
	if(is_CPU)
		cout << ">>> All " << num_threads << " threads" << endl;
#endif
}

void ophTri::objScaleShift(vec3 objSize_, vector<Real> objShift_)
{
	setObjSize(objSize_);
	setObjShift(objShift_);

	scaledMeshData = new Real[meshData->n_faces * 9];

	objNormCenter();
	
	vec3 shift = getContext().shift;
	int i;

#ifdef _OPENMP
	int num_threads;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads(); // get number of Multi Threading
#pragma omp for private(i)
#endif
		for (i = 0; i < meshData->n_faces * 3; i++) {
			int idx = i * 3;
			scaledMeshData[idx + _X] = normalizedMeshData[idx + _X] * objSize[_X] + shift[_X];
			scaledMeshData[idx + _Y] = normalizedMeshData[idx + _Y] * objSize[_Y] + shift[_Y];
			scaledMeshData[idx + _Z] = normalizedMeshData[idx + _Z] * objSize[_Z] + shift[_Z];
		}
#ifdef _OPENMP
	}
#endif

	delete[] normalizedMeshData;
	cout << "Object Scaling and Shifting Finishied.." << endl;

#ifdef _OPENMP
	cout << ">>> All " << num_threads << " threads" << endl;
#endif
}

void ophTri::objScaleShift(vec3 objSize_, vec3 objShift_)
{
	setObjSize(objSize_);
	setObjShift(objShift_);

	scaledMeshData = new Real[meshData->n_faces * 9];

	objNormCenter();

	vec3 shift = getContext().shift;
	int i;

#ifdef _OPENMP
	int num_threads;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads(); // get number of Multi Threading
#pragma omp for private(i)
#endif
		for (i = 0; i < meshData->n_faces * 3; i++) {
			int idx = i * 3;
			scaledMeshData[idx + _X] = normalizedMeshData[idx + _X] * objSize[_X] + shift[_X];
			scaledMeshData[idx + _Y] = normalizedMeshData[idx + _Y] * objSize[_Y] + shift[_Y];
			scaledMeshData[idx + _Z] = normalizedMeshData[idx + _Z] * objSize[_Z] + shift[_Z];
		}
#ifdef _OPENMP
	}
	cout << ">>> All " << num_threads << " threads" << endl;
#endif

	delete[] normalizedMeshData;
	cout << "Object Scaling and Shifting Finishied.." << endl;
}

vec3 vecCross(const vec3& a, const vec3& b)
{
	vec3 c;

	c(0) = a(0 + 1) * b(0 + 2) - a(0 + 2) * b(0 + 1);

	c(1) = a(1 + 1) * b(1 + 2) - a(1 + 2) * b(1 + 1);

	c(2) = a(2 + 1) * b(2 + 2) - a(2 + 2) * b(2 + 1);


	return c;
}


void ophTri::generateHologram(uint SHADING_FLAG)
{
	resetBuffer();

	auto start_time = CUR_TIME;
	LOG("1) Algorithm Method : Tri Mesh\n");
	LOG("2) Generate Hologram with %s\n", is_CPU ?
#ifdef _OPENMP
		"Multi Core CPU" :
#else
		"Single Core CPU" :
#endif
		"GPU");
	//LOG("3) Transform Viewing Window : %s\n", is_ViewingWindow ? "ON" : "OFF");

	auto start = CUR_TIME;
	(is_CPU) ? initializeAS() : initialize_GPU();
	prepareMeshData();
	(is_CPU) ? generateAS(SHADING_FLAG) : generateAS_GPU(SHADING_FLAG);

	if (is_CPU) {
		fft2(context_.pixel_number, angularSpectrum, OPH_BACKWARD, OPH_ESTIMATE);
		fftwShift(angularSpectrum, (*complex_H), context_.pixel_number[_X], context_.pixel_number[_Y], OPH_BACKWARD);
		/*fftExecute((*complex_H));*/
	}
	//fresnelPropagation(*(complex_H), *(complex_H), objShift[_Z]);
	m_nProgress = 0;
	auto end = CUR_TIME;
	m_elapsedTime = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("Total Elapsed Time: %lf (s)\n", m_elapsedTime);
}


void ophTri::generateAS(uint SHADING_FLAG)
{
	const uint pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];
	int N = meshData->n_faces;

	Real** freq = new Real*[3];
	Real** fl = new Real*[3];
	for (int i = 0; i < 3; i++) {
		freq[i] = new Real[pnXY];
		fl[i] = new Real[pnXY];
	}
	calGlobalFrequency(freq);

	Real *freqTermX = new Real[pnXY];
	Real *freqTermY = new Real[pnXY];

	findNormals(SHADING_FLAG);

	for (int j = 0; j < N; j++) {
		Real mesh[9] = { 0.0, };
		memcpy(mesh, &scaledMeshData[9 * j], sizeof(Real) * 9);

		if (!checkValidity(no[j]))
			continue;
		if (!findGeometricalRelations(mesh, no[j]))
			continue;
		if (!calFrequencyTerm(freq, fl, freqTermX, freqTermY))
			continue;

		switch (SHADING_FLAG)
		{
		case SHADING_FLAT:
			refAS_Flat(no[j], freqTermX, freqTermY);
			break;
		case SHADING_CONTINUOUS:
			refAS_Continuous(j, freqTermX, freqTermY);
			break;
		default:
			LOG("error: WRONG SHADING_FLAG\n");
			cin.get();
		}
		if (!refToGlobal(freq, fl))
			continue;

		m_nProgress = (int)((Real)j * 100 / ((Real)N));
	}

	LOG("Angular Spectrum Generated...\n");

	for (int i = 0; i < 3; i++) {
		delete[] freq[i];
		delete[] fl[i];
	}
	delete[] freq, fl, scaledMeshData, freqTermX, freqTermY, refAS, ASTerm, randTerm, phaseTerm, convol;
	scaledMeshData = nullptr;
	refAS = nullptr;
	ASTerm = nullptr;
	randTerm = nullptr;
	phaseTerm = nullptr;
	convol = nullptr;
}


uint ophTri::findNormals(uint SHADING_FLAG)
{
	int N = meshData->n_faces;

	for (int i = 0; i < N; i++)
	{
		int idx = i * 9;
		no[i] = vecCross(
			{
				scaledMeshData[idx + _X1] - scaledMeshData[idx + _X2],
				scaledMeshData[idx + _Y1] - scaledMeshData[idx + _Y2],
				scaledMeshData[idx + _Z1] - scaledMeshData[idx + _Z2]
			},
			{
				scaledMeshData[idx + _X3] - scaledMeshData[idx + _X2],
				scaledMeshData[idx + _Y3] - scaledMeshData[idx + _Y2],
				scaledMeshData[idx + _Z3] - scaledMeshData[idx + _Z2]
			}
			);
	}
	Real normNo = 0;

	int i;
#if 0 // 반복 사이즈가 클수록 openMP의 효율이 좋기때문에 적정값을 찾을 필요가 있음.
#ifdef _OPENMP
#pragma omp parallel for private(i) reduction(+:normNo)
#endif
#endif
	for (i = 0; i < N; i++) {
		Real tmp = norm(no[i]);
		normNo += tmp * tmp;
	}

	normNo = sqrt(normNo);

#if 0
#ifdef _OPENMP
#pragma omp parallel for firstprivate(normNo)
#endif
#endif
	for (int i = 0; i < N; i++) {
		na[i] = no[i] / normNo;
	}



	if (SHADING_FLAG == SHADING_CONTINUOUS) {
		vec3* vertices = new vec3[N * 3];
		vec3 zeros(0, 0, 0);

		for (uint idx = 0; idx < N * 3; idx++) {
			memcpy(&vertices[idx], &scaledMeshData[idx * 3], sizeof(vec3));
		}
		for (uint idx1 = 0; idx1 < N * 3; idx1++) {
			if (vertices[idx1] == zeros)
				continue;
			vec3 sum = na[idx1 / 3];
			uint count = 1;
			uint* idxes = new uint[N * 3];
			idxes[0] = idx1;
			for (uint idx2 = idx1 + 1; idx2 < N * 3; idx2++) {
				if (vertices[idx2] == zeros)
					continue;
				if ((vertices[idx1][0] == vertices[idx2][0])
					& (vertices[idx1][1] == vertices[idx2][1])
					& (vertices[idx1][2] == vertices[idx2][2])) {

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

	return 1;
}

bool ophTri::checkValidity(vec3 no)
{
	if (no[_Z] < 0 || (no[_X] == 0 && no[_Y] == 0 && no[_Z] == 0))
		return false;
	else
		return true;
}

bool ophTri::findGeometricalRelations(Real* mesh, vec3 no)
{
	vec3 n = no / norm(no);
	Real mesh_local[9] = { 0.0 };
	Real th, ph;
	if (n[_X] == 0 && n[_Z] == 0)
		th = 0;
	else
		th = atan(n[_X] / n[_Z]);

	Real temp = n[_Y] / sqrt(n[_X] * n[_X] + n[_Z] * n[_Z]);
	ph = atan(temp);
	geom.glRot[0] = cos(th);			geom.glRot[1] = 0;			geom.glRot[2] = -sin(th);
	geom.glRot[3] = -sin(ph)*sin(th);	geom.glRot[4] = cos(ph);	geom.glRot[5] = -sin(ph)*cos(th);
	geom.glRot[6] = cos(ph)*sin(th);	geom.glRot[7] = sin(ph);	geom.glRot[8] = cos(ph)*cos(th);

	for (int i = 0; i < 3; i++) {
		int idx = 3 * i;
		Real x = mesh[idx];
		Real y = mesh[idx + 1];
		Real z = mesh[idx + 2];

		mesh_local[idx] = geom.glRot[0] * x + geom.glRot[1] * y + geom.glRot[2] * z;
		mesh_local[idx + 1] = geom.glRot[3] * x + geom.glRot[4] * y + geom.glRot[5] * z;
		mesh_local[idx + 2] = geom.glRot[6] * x + geom.glRot[7] * y + geom.glRot[8] * z;
	}

	geom.glShift[_X] = -mesh_local[_X1];
	geom.glShift[_Y] = -mesh_local[_Y1];
	geom.glShift[_Z] = -mesh_local[_Z1];

	for (int i = 0; i < 3; i++) {
		int idx = 3 * i;
		mesh_local[idx] += geom.glShift[_X];
		mesh_local[idx + 1] += geom.glShift[_Y];
		mesh_local[idx + 2] += geom.glShift[_Z];
	}

	if (mesh_local[_X3] * mesh_local[_Y2] == mesh_local[_Y3] * mesh_local[_X2])
		return false;

	Real refTri[9] = { 0,0,0,1,1,0,1,0,0 };

	geom.loRot[0] = (refTri[_X3] * mesh_local[_Y2] - refTri[_X2] * mesh_local[_Y3]) / (mesh_local[_X3] * mesh_local[_Y2] - mesh_local[_Y3] * mesh_local[_X2]);
	geom.loRot[1] = (refTri[_X3] * mesh_local[_X2] - refTri[_X2] * mesh_local[_X3]) / (-mesh_local[_X3] * mesh_local[_Y2] + mesh_local[_Y3] * mesh_local[_X2]);
	geom.loRot[2] = (refTri[_Y3] * mesh_local[_Y2] - refTri[_Y2] * mesh_local[_Y3]) / (mesh_local[_X3] * mesh_local[_Y2] - mesh_local[_Y3] * mesh_local[_X2]);
	geom.loRot[3] = (refTri[_Y3] * mesh_local[_X2] - refTri[_Y2] * mesh_local[_X3]) / (-mesh_local[_X3] * mesh_local[_Y2] + mesh_local[_Y3] * mesh_local[_X2]);

	if ((geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2]) == 0)
		return false;


	return true;
}

void ophTri::calGlobalFrequency(Real** frequency)
{
	LOG("Calculate Global Frequency\n");
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];
	const uint nChannel = context_.waveNum;

	Real dfx = 1 / ppX / pnX;
	Real dfy = 1 / ppY / pnY;

	int startX = pnX / 2;
	int startY = pnY / 2;
#if 0
	for (uint ch = 0; ch < nChannel; ch++) {
		Real lambda = context_.wave_length[ch];
#else
	Real lambda = context_.wave_length[0];
#endif
		Real dfl = 1 / lambda;
		Real sqdfl = dfl * dfl;

		int i = 0;
#ifdef _OPENMP
#pragma omp parallel for private(i) firstprivate(dfy) firstprivate(dfx) firstprivate(sqdfl)
#endif
		for (i = startY; i > -startY; i--) {
			Real y = i * dfy;
			Real yy = y * y;

			int base = (-i + startY) * pnX;

			for (int j = -startX; j < startX; j++) { // -960 ~ 959
				int idx = base + (j + startX);
				frequency[_X][idx] = j * dfx;
				frequency[_Y][idx] = y;
				frequency[_Z][idx] = sqrt(sqdfl - (frequency[_X][idx] * frequency[_X][idx]) - yy);
				//k++;
			}
		}
#if 0
	}
#endif

LOG("Done\n");
}

bool ophTri::calFrequencyTerm(Real** frequency, Real** fl, Real *freqTermX, Real *freqTermY)
{
	// p.s. only 1 channel
	const uint pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];

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

	int i;
#ifdef _OPENMP
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
#pragma omp for private(i) firstprivate(w) firstprivate(ww) firstprivate(glRot) firstprivate(carrierWaveLoc) firstprivate(invLoRot)
#endif
		for (i = 0; i < pnXY; i++) {
			Real flxShifted;
			Real flyShifted;

			fl[_X][i] = geom.glRot[0] * frequency[_X][i] + geom.glRot[1] * frequency[_Y][i] + geom.glRot[2] * frequency[_Z][i];
			fl[_Y][i] = geom.glRot[3] * frequency[_X][i] + geom.glRot[4] * frequency[_Y][i] + geom.glRot[5] * frequency[_Z][i];
			fl[_Z][i] = sqrt(ww - fl[_X][i] * fl[_X][i] - fl[_Y][i] * fl[_Y][i]);
			flxShifted = fl[_X][i] - w * (geom.glRot[0] * carrierWaveLoc[_X] + geom.glRot[1] * carrierWaveLoc[_Y] + geom.glRot[2] + carrierWaveLoc[_Z]);
			flyShifted = fl[_Y][i] - w * (geom.glRot[3] * carrierWaveLoc[_X] + geom.glRot[4] * carrierWaveLoc[_Y] + geom.glRot[5] + carrierWaveLoc[_Z]);
			
			freqTermX[i] = invLoRot[0] * flxShifted + invLoRot[1] * flyShifted;
			freqTermY[i] = invLoRot[2] * flxShifted + invLoRot[3] * flyShifted;
		}
#ifdef _OPENMP
	}
#endif
	return true;
}

uint ophTri::refAS_Flat(vec3 no, Real* freqTermX, Real* freqTermY)
{
	const uint pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];

	n = no / norm(no);
	
	Real shadingFactor;

	if (illumination[_X] == 0 && illumination[_Y] == 0 && illumination[_Z] == 0) {
		shadingFactor = 1;
	}
	else {
		vec3 normIllu = illumination / norm(illumination);
		shadingFactor = 2 * (n[_X] * normIllu[_X] + n[_Y] * normIllu[_Y] + n[_Z] * normIllu[_Z]) + 0.3;
		if (shadingFactor < 0)
			shadingFactor = 0;
	}

	Real PI2 = M_PI * 2;
	Real sqPI2 = PI2 * PI2;

	int i;
#ifdef _OPENMP
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
#pragma omp for private(i) firstprivate(PI2) firstprivate(sqPI2) firstprivate(shadingFactor)
#endif
		for (i = 0; i < pnXY; i++) {
			Complex<Real> refTerm1(0, 0);
			Complex<Real> refTerm2(0, 0);

			if (freqTermX[i] == -freqTermY[i] && freqTermY[i] != 0) {
				refTerm1[_IM] = PI2 * freqTermY[i];
				refTerm2[_IM] = 1;
				refAS[i] = shadingFactor * (((Complex<Real>)1 - exp(refTerm1)) / (sqPI2 * freqTermY[i] * freqTermY[i]) + refTerm2 / (PI2 * freqTermY[i]));
			}
			else if (freqTermX[i] == freqTermY[i] && freqTermX[i] == 0) {
				refAS[i] = shadingFactor * 1 / 2;
			}
			else if (freqTermX[i] != 0 && freqTermY[i] == 0) {
				refTerm1[_IM] = -PI2 * freqTermX[i];
				refTerm2[_IM] = 1;
				refAS[i] = shadingFactor * ((exp(refTerm1) - (Complex<Real>)1) / (sqPI2 * freqTermX[i] * freqTermX[i]) + (refTerm2 * exp(refTerm1)) / (PI2 * freqTermX[i]));
			}
			else if (freqTermX[i] == 0 && freqTermY[i] != 0) {
				refTerm1[_IM] = PI2 * freqTermY[i];
				refTerm2[_IM] = 1;
				refAS[i] = shadingFactor * (((Complex<Real>)1 - exp(refTerm1)) / (sqPI2 * freqTermY[i] * freqTermY[i]) - refTerm2 / (PI2 * freqTermY[i]));
			}
			else {
				refTerm1[_IM] = -PI2 * freqTermX[i];
				refTerm2[_IM] = -PI2 * (freqTermX[i] + freqTermY[i]);
				refAS[i] = shadingFactor * ((exp(refTerm1) - (Complex<Real>)1) / (sqPI2 * freqTermX[i] * freqTermY[i]) + ((Complex<Real>)1 - exp(refTerm2)) / (sqPI2 * freqTermY[i] * (freqTermX[i] + freqTermY[i])));
			}
		}
#ifdef _OPENMP
	}
#endif
	//randPhaseDist(refAS);

	return 1;
}

uint ophTri::refAS_Continuous(uint n, Real* freqTermX, Real* freqTermY)
{
	const uint pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];

	vec3 av(0, 0, 0);
	av[0] = nv[3 * n + 0][0] * illumination[0] + nv[3 * n + 0][1] * illumination[1] + nv[3 * n + 0][2] * illumination[2] + 0.1;
	av[2] = nv[3 * n + 1][0] * illumination[0] + nv[3 * n + 1][1] * illumination[1] + nv[3 * n + 1][2] * illumination[2] + 0.1;
	av[1] = nv[3 * n + 2][0] * illumination[0] + nv[3 * n + 2][1] * illumination[1] + nv[3 * n + 2][2] * illumination[2] + 0.1;


	Real PI2 = M_PI * 2;
	Real sqPI2 = PI2 * PI2;
	Real cuPI2 = PI2 * PI2 * PI2;

	int i;
#ifdef _OPENMP
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
#pragma omp for private(i) firstprivate(PI2) firstprivate(sqPI2) firstprivate(cuPI2)
#endif
		for (i = 0; i < pnXY; i++) {
			Complex<Real> refTerm1(0, 0);
			Complex<Real> refTerm2(0, 0);
			Complex<Real> refTerm3(0, 0);
			Complex<Real> D1(0, 0);
			Complex<Real> D2(0, 0);
			Complex<Real> D3(0, 0);

			if (freqTermX[i] == 0 && freqTermY[i] == 0) {
				D1((Real)1 / (Real)3, 0);
				D2((Real)1 / (Real)5, 0);
				D3((Real)1 / (Real)2, 0);
			}
			else if (freqTermX[i] == 0 && freqTermY[i] != 0) {
				refTerm1[_IM] = -PI2 * freqTermY[i];
				refTerm2[_IM] = 1;
				std::complex<Real> expRefTerm1 = exp(refTerm1);

				D1 = (refTerm1 - (Real)1)*refTerm1.exp() / (cuPI2 * freqTermY[i] * freqTermY[i] * freqTermY[i])
					- refTerm1 / (sqPI2 * M_PI * freqTermY[i] * freqTermY[i] * freqTermY[i]);
				D2 = -(M_PI * freqTermY[i] + refTerm2) / (sqPI2 * M_PI * freqTermY[i] * freqTermY[i] * freqTermY[i]) * expRefTerm1
					+ refTerm1 / (cuPI2 * freqTermY[i] * freqTermY[i] * freqTermY[i]);
				D3 = expRefTerm1 / (PI2 * freqTermY[i]) + ((Real)1 - refTerm2) / (PI2 * freqTermY[i]);
			}
			else if (freqTermX[i] != 0 && freqTermY[i] == 0) {
				refTerm1[_IM] = sqPI2 * freqTermX[i] * freqTermX[i];
				refTerm2[_IM] = 1;
				refTerm3[_IM] = PI2 * freqTermX[i];
				std::complex<Real> expRefTerm3 = exp(-refTerm3);

				D1 = (refTerm1 + 2 * PI2 * freqTermX[i] - (Real)2 * refTerm2) / (cuPI2 * freqTermY[i] * freqTermY[i] * freqTermY[i]) * expRefTerm3
					+ refTerm2 / (sqPI2 * M_PI * freqTermX[i] * freqTermX[i] * freqTermX[i]);
				D2 = (Real)1 / (Real)2 * D1;
				D3 = ((refTerm3 + (Real)1) * expRefTerm3 - (Real)1) / (sqPI2 * freqTermX[i] * freqTermX[i]);
			}
			else if (freqTermX[i] == -freqTermY[i]) {
				refTerm1[_IM] = 1;
				refTerm2[_IM] = PI2 * freqTermX[i];
				refTerm3[_IM] = PI2 * M_PI * freqTermX[i] * freqTermX[i];
				std::complex<Real> expRefTerm2 = exp(-refTerm2);

				D1 = (-PI2 * freqTermX[i] + refTerm1) / (cuPI2 * freqTermX[i] * freqTermX[i] * freqTermX[i]) * expRefTerm2
					- (refTerm3 + refTerm1) / (cuPI2 * freqTermX[i] * freqTermX[i] * freqTermX[i]);
				D2 = (-refTerm1) / (cuPI2 * freqTermX[i] * freqTermX[i] * freqTermX[i]) * expRefTerm2;
				+(-refTerm3 + refTerm1 + PI2 * freqTermX[i]) / (cuPI2 * freqTermX[i] * freqTermX[i] * freqTermX[i]);
				D3 = (-refTerm1) / (sqPI2 * freqTermX[i] * freqTermX[i]) * expRefTerm2
					+ (-refTerm2 + (Real)1) / (sqPI2 * freqTermX[i] * freqTermX[i]);
			}
			else {
				refTerm1[_IM] = -PI2 * (freqTermX[i] + freqTermY[i]);
				refTerm2[_IM] = 1;
				refTerm3[_IM] = -PI2 * freqTermX[i];
				std::complex<Real> expRefTerm1 = exp(refTerm1);
				std::complex<Real> expRefTerm3 = exp(refTerm3);

				D1 = expRefTerm1 * (refTerm2 - PI2 * (freqTermX[i] + freqTermY[i])) / (cuPI2 * freqTermY[i] * (freqTermX[i] + freqTermY[i]) * (freqTermX[i] + freqTermY[i]))
					+ expRefTerm3 * (PI2 * freqTermX[i] - refTerm2) / (cuPI2 * freqTermX[i] * freqTermX[i] * freqTermY[i])
					+ ((2 * freqTermX[i] + freqTermY[i])*refTerm2) / (cuPI2 * freqTermX[i] * freqTermX[i] * (freqTermX[i] + freqTermY[i])*(freqTermX[i] + freqTermY[i]));
				D2 = expRefTerm1 * (refTerm2 * (freqTermX[i] + 2 * freqTermY[i]) - PI2 * freqTermY[i] * (freqTermX[i] + freqTermY[i])) / (cuPI2 * freqTermY[i] * freqTermY[i] * (freqTermX[i] + freqTermY[i]) * (freqTermX[i] + freqTermY[i]))
					+ expRefTerm3 * (-refTerm2) / (cuPI2 * freqTermX[i] * freqTermY[i] * freqTermY[i])
					+ refTerm2 / (cuPI2 * freqTermX[i] * (freqTermX[i] + freqTermY[i])* (freqTermX[i] + freqTermY[i]));
				D3 = -expRefTerm1 / (sqPI2 * freqTermY[i] * (freqTermX[i] + freqTermY[i]))
					+ expRefTerm3 / (sqPI2 * freqTermX[i] * freqTermY[i])
					- (Real)1 / (sqPI2 * freqTermX[i] * (freqTermX[i] + freqTermY[i]));
			}
			refAS[i] = (av[1] - av[0]) * D1 + (av[2] - av[1]) * D2 + av[0] * D3;
		}
#ifdef _OPENMP
	}
#endif


	//randPhaseDist(refAS);

	return 1;
}

void ophTri::randPhaseDist(Complex<Real>* AS)
{
	ivec2 px = context_.pixel_number;

	fft2(px, AS, OPH_FORWARD, OPH_ESTIMATE);
	fftwShift(AS, ASTerm, px[_X], px[_Y], OPH_FORWARD, (bool)OPH_ESTIMATE);
	//fftExecute(ASTerm);

	Real randVal;
	Complex<Real> phase;
	int n = px[_X] * px[_Y];

	for (int i = 0; i < n; i++) {
		randVal = rand((Real)0, (Real)1, px[_X] * px[_Y]);
		phase[_RE] = 0;
		phase[_IM] = 2 * M_PI*randVal;
		phaseTerm[i] = exp(phase);
	}

	fft2(px, phaseTerm, OPH_FORWARD, OPH_ESTIMATE);
	fftwShift(phaseTerm, randTerm, px[_X], px[_Y], OPH_FORWARD, (bool)OPH_ESTIMATE);
	//fftExecute(randTerm);

	for (int i = 0; i < n; i++) {
		convol[i] = ASTerm[i] * randTerm[i];
	}

	fft2(px, convol, OPH_BACKWARD, OPH_ESTIMATE);
	fftwShift(convol, AS, px[_X], px[_Y], OPH_BACKWARD, (bool)OPH_ESTIMATE);
	//fftExecute(AS);

}

bool ophTri::refToGlobal(Real** frequency, Real** fl)
{
	const uint pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];

	Complex<Real> term1(0, 0);

	Real det = geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2];

	Real PI2 = M_PI * 2;

	if (det == 0)
		return false;

	term1[_IM] = -PI2 / context_.wave_length[0] * (
		carrierWave[_X] * (geom.glRot[0] * geom.glShift[_X] + geom.glRot[3] * geom.glShift[_Y] + geom.glRot[6] * geom.glShift[_Z])
		+ carrierWave[_Y] * (geom.glRot[1] * geom.glShift[_X] + geom.glRot[4] * geom.glShift[_Y] + geom.glRot[7] * geom.glShift[_Z])
		+ carrierWave[_Z] * (geom.glRot[2] * geom.glShift[_X] + geom.glRot[5] * geom.glShift[_Y] + geom.glRot[8] * geom.glShift[_Z]));


	std::complex<Real> expTerm = exp(term1);


	Real glShift[3];
	memcpy(glShift, geom.glShift, sizeof(glShift));

	int i;
#ifdef _OPENMP
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		Complex<Real> temp(0, 0);
		Complex<Real> term2(0, 0);
#pragma omp for private(i) firstprivate(PI2) firstprivate(glShift) firstprivate(expTerm) firstprivate(det)
#endif
		for (i = 0; i < pnXY; i++) {
			if (frequency[_Z][i] == 0)
				temp = 0;
			else {
				term2[_IM] = PI2 * (fl[_X][i] * glShift[_X] + fl[_Y][i] * glShift[_Y] + fl[_Z][i] * glShift[_Z]);
				temp = refAS[i] / det * expTerm * fl[_Z][i] / frequency[_Z][i] * exp(term2);
			}

			if (abs(temp) > MIN_DOUBLE) {}
			else { temp = 0; }

			angularSpectrum[i] += temp;
		}
#ifdef _OPENMP
	}
#endif
	return true;
}

void ophTri::prepareMeshData()
{
	int N = meshData->n_faces;

	int N3 = N * 3;

	Real* x_point = new Real[N3];
	Real* y_point = new Real[N3];
	Real* z_point = new Real[N3];

	int i;
#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
	for (i = 0; i < N3; i++) {
		int idx = i * 3;
		x_point[i] = triMeshArray[idx + _X];
		y_point[i] = triMeshArray[idx + _Y];
		z_point[i] = triMeshArray[idx + _Z];
	}

	Real x_max = maxOfArr(x_point, N3);
	Real x_min = minOfArr(x_point, N3);
	Real y_max = maxOfArr(y_point, N3);
	Real y_min = minOfArr(y_point, N3);
	Real z_max = maxOfArr(z_point, N3);
	Real z_min = minOfArr(z_point, N3);

	Real x_cen = (x_max + x_min) / 2;
	Real y_cen = (y_max + y_min) / 2;
	Real z_cen = (z_max + z_min) / 2;

	Real x_del = x_max - x_min;
	Real y_del = y_max - y_min;
	Real z_del = z_max - z_min;

	Real del = maxOfArr({ x_del, y_del, z_del });

	cout << "center: " << x_cen << ", " << y_cen << ", " << z_cen << endl;

	Real* centered = new Real[N * 9];
	vec3 shift = getContext().shift;

#ifdef _OPENMP
	int num_threads;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads(); // get number of Multi Threading
#pragma omp for private(i) firstprivate(x_cen) firstprivate(y_cen) firstprivate(z_cen) firstprivate(del)
#endif
		for (i = 0; i < N3; i++) {
			int idx = i * 3;
			scaledMeshData[idx + _X] = (x_point[i] - x_cen) / del * objSize[_X] + shift[_X];
			scaledMeshData[idx + _Y] = (y_point[i] - y_cen) / del * objSize[_Y] + shift[_Y];
			scaledMeshData[idx + _Z] = (z_point[i] - z_cen) / del * objSize[_Z] + shift[_Z];
		}
#ifdef _OPENMP
	}
#endif

	delete[] x_point, y_point, z_point;

	cout << "scaledMeshData: " << scaledMeshData[_X] << ", " << scaledMeshData[_Y] << ", " << scaledMeshData[_Z] << endl;

#ifdef _OPENMP
	cout << ">>> All " << num_threads << " threads" << endl;
#endif

}
