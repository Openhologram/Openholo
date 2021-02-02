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

void ophTri::setMode(bool isCPU)
{
	is_CPU = isCPU;
}

void ophTri::setViewingWindow(bool is_ViewingWindow)
{
	this->is_ViewingWindow = is_ViewingWindow;
}

uint ophTri::loadMeshText(const char* fileName) {

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

bool ophTri::loadMeshData(const char* fileName, const char* ext) {
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
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;

	if (angularSpectrum) {
		delete[] angularSpectrum;
		angularSpectrum = nullptr;
	}
	angularSpectrum = new Complex<Real>[pnXY];
	memset(angularSpectrum, 0, sizeof(Complex<Real>) * pnXY);
}


void ophTri::objNormCenter()
{
	if (normalizedMeshData) {
		delete[] normalizedMeshData;
		normalizedMeshData = nullptr;
	}

	int nFace = meshData->n_faces;

	normalizedMeshData = new Real[nFace * 9];

	Real* x_point = new Real[nFace * 3];
	Real* y_point = new Real[nFace * 3];
	Real* z_point = new Real[nFace * 3];

	int i;
#ifdef _OPENMP
#pragma omp for private(i)
#endif
	for (i = 0; i < nFace * 3; i++) {
		int idx = i * 3;
		x_point[i] = triMeshArray[idx + _X];
		y_point[i] = triMeshArray[idx + _Y];
		z_point[i] = triMeshArray[idx + _Z];
	}
	Real x_cen = (maxOfArr(x_point, nFace * 3) + minOfArr(x_point, nFace * 3)) / 2;
	Real y_cen = (maxOfArr(y_point, nFace * 3) + minOfArr(y_point, nFace * 3)) / 2;
	Real z_cen = (maxOfArr(z_point, nFace * 3) + minOfArr(z_point, nFace * 3)) / 2;

	Real* centered = new Real[nFace * 9];

#ifdef _OPENMP
#pragma omp for private(i)
#endif
	for (i = 0; i < nFace * 3; i++) {
		int idx = i * 3;
		centered[idx + _X] = x_point[i] - x_cen;
		centered[idx + _Y] = y_point[i] - y_cen;
		centered[idx + _Z] = z_point[i] - z_cen;
	}
	//
	Real x_cen1 = (maxOfArr(x_point, nFace * 3) + minOfArr(x_point, nFace * 3)) / 2;
	Real y_cen1 = (maxOfArr(y_point, nFace * 3) + minOfArr(y_point, nFace * 3)) / 2;
	Real z_cen1 = (maxOfArr(z_point, nFace * 3) + minOfArr(z_point, nFace * 3)) / 2;

	cout << "center: "<< x_cen1 << ", " << y_cen1 << ", " << z_cen1 << endl;

	//
	Real x_del = (maxOfArr(x_point, nFace * 3) - minOfArr(x_point, nFace * 3));
	Real y_del = (maxOfArr(y_point, nFace * 3) - minOfArr(y_point, nFace * 3));
	Real z_del = (maxOfArr(z_point, nFace * 3) - minOfArr(z_point, nFace * 3));
	Real del = maxOfArr({ x_del, y_del, z_del });

#ifdef _OPENMP
#pragma omp for private(i)
#endif
	for (i = 0; i < nFace * 9; i++) {
		normalizedMeshData[i] = centered[i] / del;
	}
	delete[] centered, x_point, y_point, z_point;
}


void ophTri::objScaleShift()
{
	if (scaledMeshData) {
		delete[] scaledMeshData;
		scaledMeshData = nullptr;
	}
	scaledMeshData = new Real[meshData->n_faces * 9];
	
	objNormCenter();

	Real *pMesh = nullptr;

	if (is_ViewingWindow) {
		pMesh = new Real[meshData->n_faces * 9];
		transVW(meshData->n_faces * 9, pMesh, normalizedMeshData);
	}
	else {
		pMesh = normalizedMeshData;
	}

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
			Real pcx = pMesh[idx + _X];
			Real pcy = pMesh[idx + _Y];
			Real pcz = pMesh[idx + _Z];

			scaledMeshData[idx + _X] = pcx * objSize[_X] + shift[_X];
			scaledMeshData[idx + _Y] = pcy * objSize[_Y] + shift[_Y];
			scaledMeshData[idx + _Z] = pcz * objSize[_Z] + shift[_Z];
		}
	}

	if (is_ViewingWindow) {
		delete[] pMesh;
	}
	delete[] normalizedMeshData;

	cout << "Object Scaling and Shifting Finishied.." << endl;

#ifdef _OPENMP
	cout << ">>> All " << num_threads << " threads" << endl;
#endif
}

void ophTri::objScaleShift(vec3 objSize_, vector<Real> objShift_) 
{
	setObjSize(objSize_);
	setObjShift(objShift_);

	scaledMeshData = new Real[meshData->n_faces * 9];

	objNormCenter();
	Real *pMesh = nullptr;

	if (is_ViewingWindow) {
		pMesh = new Real[meshData->n_faces * 9];
		transVW(meshData->n_faces * 9, pMesh, normalizedMeshData);
	}
	else {
		pMesh = normalizedMeshData;
	}

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
			Real pcx = *(pMesh + 3 * i + _X);
			Real pcy = *(pMesh + 3 * i + _Y);
			Real pcz = *(pMesh + 3 * i + _Z);

			*(scaledMeshData + 3 * i + _X) = pcx * objSize[_X] + shift[_X];
			*(scaledMeshData + 3 * i + _Y) = pcy * objSize[_Y] + shift[_Y];
			*(scaledMeshData + 3 * i + _Z) = pcz * objSize[_Z] + shift[_Z];
		}
	}

	if (is_ViewingWindow) {
		delete[] pMesh;
	}
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
	Real *pMesh = nullptr;

	if (is_ViewingWindow) {
		pMesh = new Real[meshData->n_faces * 9];
		transVW(meshData->n_faces * 9, pMesh, normalizedMeshData);
	}
	else {
		pMesh = normalizedMeshData;
	}
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
			Real pcx = *(pMesh + 3 * i + _X);
			Real pcy = *(pMesh + 3 * i + _Y);
			Real pcz = *(pMesh + 3 * i + _Z);

			*(scaledMeshData + 3 * i + _X) = pcx * objSize[_X] + shift[_X];
			*(scaledMeshData + 3 * i + _Y) = pcy * objSize[_Y] + shift[_Y];
			*(scaledMeshData + 3 * i + _Z) = pcz * objSize[_Z] + shift[_Z];
		}
#ifdef _OPENMP
	}
	cout << ">>> All " << num_threads << " threads" << endl;
#endif
	if (is_ViewingWindow) {
		delete[] pMesh;
	}
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
	LOG("3) Transform Viewing Window : %s\n", is_ViewingWindow ? "ON" : "OFF");

	auto start = CUR_TIME;
	objScaleShift();
	(is_CPU) ? initializeAS() : initialize_GPU();
	(is_CPU) ? generateAS(SHADING_FLAG) : generateAS_GPU(SHADING_FLAG);

	if (is_CPU) {
		fft2(context_.pixel_number, angularSpectrum, OPH_BACKWARD, OPH_ESTIMATE);
		fftwShift(angularSpectrum, (*complex_H), context_.pixel_number[_X], context_.pixel_number[_Y], OPH_BACKWARD);
		/*fftExecute((*complex_H));*/
	}
	//fresnelPropagation(*(complex_H), *(complex_H), objShift[_Z]);

	auto end = CUR_TIME;
	m_elapsedTime = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("Total Elapsed Time: %lf (s)\n", m_elapsedTime);
}

void ophTri::generateMeshHologram() {
	cout << "Hologram Generation ..." << endl;
	auto start = CUR_TIME;
	resetBuffer();
	initializeAS();
	generateAS(SHADING_TYPE);

	fft2(context_.pixel_number, angularSpectrum, OPH_BACKWARD, OPH_ESTIMATE);
	fftwShift(angularSpectrum, (*complex_H), context_.pixel_number[_X], context_.pixel_number[_Y], OPH_BACKWARD);
	//fftExecute((*complex_H));

	auto end = CUR_TIME;
	auto during = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("Total Elapsed Time: %lf (sec)\n", during);
}


void ophTri::generateAS(uint SHADING_FLAG)
{
	calGlobalFrequency();

	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;

	flx = new Real[pnXY];
	fly = new Real[pnXY];
	flz = new Real[pnXY];

	freqTermX = new Real[pnXY];
	freqTermY = new Real[pnXY];

	refAS = new Complex<Real>[pnXY];

	ASTerm = new Complex<Real>[pnXY];
	randTerm = new Complex<Real>[pnXY];
	phaseTerm = new Complex<Real>[pnXY];
	convol = new Complex<Real>[pnXY];

	findNormals(SHADING_FLAG);
	
	//int j; // private variable for Multi Threading
	for (int j = 0; j < meshData->n_faces; j++) {
		Real mesh[9] = { 0.0, };
		memcpy(mesh, &scaledMeshData[9 * j], sizeof(Real) * 9);

		if (checkValidity(mesh, no[j]) != 1)
			continue;

		if (findGeometricalRelations(mesh, no[j]) != 1)
			continue;

		if (calFrequencyTerm() != 1)
			continue;

		switch (SHADING_FLAG)
		{
		case SHADING_FLAT:
			refAS_Flat(no[j]);
			break;
		case SHADING_CONTINUOUS:
			refAS_Continuous(j);
			break;
		default:
			LOG("error: WRONG SHADING_FLAG\n");
			cin.get();
		}
		if (refToGlobal() != 1)
			continue;

		char szLog[MAX_PATH];
		sprintf_s(szLog, "%d / %llu\n", j + 1, meshData->n_faces);
		LOG(szLog);
	}
	LOG("Angular Spectrum Generated...\n");

	delete[]/* mesh, mesh_local,*/scaledMeshData, fx, fy, fz, flx, fly, flz, freqTermX, freqTermY, refAS, ASTerm, randTerm, phaseTerm, convol;
}


uint ophTri::findNormals(uint SHADING_FLAG)
{
	no = new vec3[meshData->n_faces];
	na = new vec3[meshData->n_faces];
	nv = new vec3[meshData->n_faces * 3];

	int num;
//#ifdef _OPENMP
//#pragma omp for private(num)
//#endif
	for (num = 0; num < meshData->n_faces; num++)
	{
		*(no + num) = vecCross({ scaledMeshData[num * 9 + _X1] - scaledMeshData[num * 9 + _X2],
			scaledMeshData[num * 9 + _Y1] - scaledMeshData[num * 9 + _Y2],
			scaledMeshData[num * 9 + _Z1] - scaledMeshData[num * 9 + _Z2] },
			{ scaledMeshData[num * 9 + _X3] - scaledMeshData[num * 9 + _X2],
			scaledMeshData[num * 9 + _Y3] - scaledMeshData[num * 9 + _Y2],
			scaledMeshData[num * 9 + _Z3] - scaledMeshData[num * 9 + _Z2] });
	}
	Real normNo = 0;
//#ifdef _OPENMP
//#pragma omp for private(num) reduction(+:num)
//#endif
	for (num = 0; num < meshData->n_faces; num++) {
		normNo += norm(no[num])*norm(no[num]);
	}
	LOG("normNo: %lf\n", normNo);

	normNo = sqrt(normNo);

	for (uint num = 0; num < meshData->n_faces; num++) {
		*(na + num) = no[num] / normNo;
	}

	if (SHADING_FLAG == SHADING_CONTINUOUS) {
		vec3* vertices = new vec3[meshData->n_faces * 3];
		vec3 zeros(0, 0, 0);

		for (uint idx = 0; idx < meshData->n_faces * 3; idx++) {
			memcpy(&vertices[idx], &scaledMeshData[idx * 3], sizeof(vec3));
		}
		for (uint idx1 = 0; idx1 < meshData->n_faces * 3; idx1++) {
			if (*(vertices + idx1) == zeros)
				continue;
			vec3 sum = *(na + idx1 / 3);
			uint count = 1;
			uint* idxes = new uint[meshData->n_faces * 3];
			*(idxes) = idx1;
			for (uint idx2 = idx1 + 1; idx2 < meshData->n_faces * 3; idx2++) {
				if (*(vertices + idx2) == zeros)
					continue;
				if ((vertices[idx1][0] == vertices[idx2][0])
					& (vertices[idx1][1] == vertices[idx2][1])
					& (vertices[idx1][2] == vertices[idx2][2])) {
					sum += *(na + idx2 / 3);
					*(vertices + idx2) = zeros;
					*(idxes + count) = idx2;
					count++;
				}
			}
			*(vertices + idx1) = zeros;

			sum = sum / count;
			sum = sum / norm(sum);
			for (uint i = 0; i < count; i++)
				*(nv + *(idxes + i)) = sum;

			delete[] idxes;
		}

		delete[] vertices;
	}

	return 1;
}

uint ophTri::checkValidity(Real* mesh, vec3 no) {
	
	if (no[_Z] < 0 || (no[_X] == 0 && no[_Y] == 0 && no[_Z] == 0)) {
		return -1;
	}
	if (no[_Z] >= 0)
		return 1;

	return 0;
}

uint ophTri::findGeometricalRelations(Real* mesh, vec3 no)
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
		mesh_local[3 * i] = geom.glRot[0] * mesh[3 * i] + geom.glRot[1] * mesh[3 * i + 1] + geom.glRot[2] * mesh[3 * i + 2];
		mesh_local[3 * i + 1] = geom.glRot[3] * mesh[3 * i] + geom.glRot[4] * mesh[3 * i + 1] + geom.glRot[5] * mesh[3 * i + 2];
		mesh_local[3 * i + 2] = geom.glRot[6] * mesh[3 * i] + geom.glRot[7] * mesh[3 * i + 1] + geom.glRot[8] * mesh[3 * i + 2];
	}

	geom.glShift[_X] = -mesh_local[_X1];
	geom.glShift[_Y] = -mesh_local[_Y1];
	geom.glShift[_Z] = -mesh_local[_Z1];

	for (int i = 0; i < 3; i++) {
		mesh_local[3 * i] += geom.glShift[_X];
		mesh_local[3 * i + 1] += geom.glShift[_Y];
		mesh_local[3 * i + 2] += geom.glShift[_Z];
	}

	if (mesh_local[_X3] * mesh_local[_Y2] == mesh_local[_Y3] * mesh_local[_X2])
		return -1;

	geom.loRot[0] = (refTri[_X3] * mesh_local[_Y2] - refTri[_X2] * mesh_local[_Y3]) / (mesh_local[_X3] * mesh_local[_Y2] - mesh_local[_Y3] * mesh_local[_X2]);
	geom.loRot[1] = (refTri[_X3] * mesh_local[_X2] - refTri[_X2] * mesh_local[_X3]) / (-mesh_local[_X3] * mesh_local[_Y2] + mesh_local[_Y3] * mesh_local[_X2]);
	geom.loRot[2] = (refTri[_Y3] * mesh_local[_Y2] - refTri[_Y2] * mesh_local[_Y3]) / (mesh_local[_X3] * mesh_local[_Y2] - mesh_local[_Y3] * mesh_local[_X2]);
	geom.loRot[3] = (refTri[_Y3] * mesh_local[_X2] - refTri[_Y2] * mesh_local[_X3]) / (-mesh_local[_X3] * mesh_local[_Y2] + mesh_local[_Y3] * mesh_local[_X2]);

	if ((geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2]) == 0)
		return -1;


	return 1;
}

void ophTri::calGlobalFrequency()
{
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const int pnXY = pnX * pnY;
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];
	const Real ssX = context_.ss[_X] = pnX * ppX;
	const Real ssY = context_.ss[_Y] = pnY * ppY;
	const uint nChannel = context_.waveNum;

	Real dfx = 1 / ssX;
	Real dfy = 1 / ssY;
	fx = new Real[pnXY];
	fy = new Real[pnXY];
	fz = new Real[pnXY];
	uint i = 0;
	
	for (uint ch = 0; ch < nChannel; ch++) {
		Real lambda = context_.wave_length[ch];
		Real dfl = 1 / lambda;
		for (uint idxFy = pnY / 2; idxFy > -pnY / 2; idxFy--) {
			for (uint idxFx = -pnX / 2; idxFx < pnX / 2; idxFx++) {

				Real x;
				fx[i] = x = idxFx * dfx;
				Real y;
				fy[i] = y = idxFy * dfy;
				Real z;
				fz[i] = z = sqrt((dfl*dfl) - (x * x) - (y * y));
				i++;
			}
		}
	}
}

uint ophTri::calFrequencyTerm()
{	
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;

	Real* flxShifted = new Real[pnXY];
	Real* flyShifted = new Real[pnXY];
	Real waveLength = context_.wave_length[0];
	Real w = 1 / waveLength;
	Real ww = w * w;

#if 1
	int i;
#ifdef _OPENMP
#pragma omp for private(i)
#endif
	for (i = 0; i < pnXY; i++) {
		flx[i] = geom.glRot[0] * fx[i] + geom.glRot[1] * fy[i] + geom.glRot[2] * fz[i];
		fly[i] = geom.glRot[3] * fx[i] + geom.glRot[4] * fy[i] + geom.glRot[5] * fz[i];
		flz[i] = sqrt(ww - flx[i] * flx[i] - fly[i] * fly[i]);

		flxShifted[i] = flx[i] - w * (geom.glRot[0] * carrierWave[_X] + geom.glRot[1] * carrierWave[_Y] + geom.glRot[2] + carrierWave[_Z]);
		flyShifted[i] = fly[i] - w * (geom.glRot[3] * carrierWave[_X] + geom.glRot[4] * carrierWave[_Y] + geom.glRot[5] + carrierWave[_Z]);
	}
#else
	for_i(pnXY,
		flx[i] = geom.glRot[0] * fx[i] + geom.glRot[1] * fy[i] + geom.glRot[2] * fz[i];
		fly[i] = geom.glRot[3] * fx[i] + geom.glRot[4] * fy[i] + geom.glRot[5] * fz[i];
		flz[i] = sqrt((1 / context_.wave_length[0])*(1 / context_.wave_length[0]) - flx[i] * flx[i] - fly[i] * fly[i]);
		
		flxShifted[i] = flx[i] - (1 / context_.wave_length[0])*(geom.glRot[0] * carrierWave[_X] + geom.glRot[1] * carrierWave[_Y] + geom.glRot[2] * carrierWave[_Z]);
		flyShifted[i] = fly[i] - (1 / context_.wave_length[0])*(geom.glRot[3] * carrierWave[_X] + geom.glRot[4] * carrierWave[_Y] + geom.glRot[5] * carrierWave[_Z]);
		);
#endif
	Real det = geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2];

	Real* invLoRot = new Real[4];
	invLoRot[0] = (1 / det)*geom.loRot[3];
	invLoRot[1] = -(1 / det)*geom.loRot[2];
	invLoRot[2] = -(1 / det)*geom.loRot[1];
	invLoRot[3] = (1 / det)*geom.loRot[0];

#ifdef _OPENMP
#pragma omp for private(i)
#endif
	for (i = 0; i < pnXY; i++) {
		freqTermX[i] = invLoRot[0] * flxShifted[i] + invLoRot[1] * flyShifted[i];
		freqTermY[i] = invLoRot[2] * flxShifted[i] + invLoRot[3] * flyShifted[i];
	}
	
	delete[] flxShifted;
	delete[] flyShifted;
	delete[] invLoRot;
	return 1;
}

uint ophTri::refAS_Flat(vec3 no)
{
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;

	n = no / norm(no);
		
	refTerm1(0,0);
	refTerm2(0,0);

	if (illumination[_X] == 0 && illumination[_Y] == 0 && illumination[_Z] == 0) {
		shadingFactor = 1;
	}
	else {
		vec3 normIllu = illumination / norm(illumination);
		shadingFactor = 2 * (n[_X] * normIllu[_X] + n[_Y] * normIllu[_Y] + n[_Z] * normIllu[_Z]) + 0.3;
		if (shadingFactor < 0)
			shadingFactor = 0;		
	}
	for (int i = 0; i < pnXY; i++) {
		if (freqTermX[i] == -freqTermY[i] && freqTermY[i] != 0) {
			refTerm1[_IM] = 2 * M_PI*freqTermY[i];
			refTerm2[_IM] = 1;
			refAS[i] = shadingFactor*(((Complex<Real>)1 - exp(refTerm1)) / (4 * M_PI*M_PI*freqTermY[i] * freqTermY[i]) + refTerm2 / (2 * M_PI*freqTermY[i]));
		}
		else if (freqTermX[i] == freqTermY[i] && freqTermX[i] == 0) {
			refAS[i] = shadingFactor * 1 / 2;
		}
		else if (freqTermX[i] != 0 && freqTermY[i] == 0) {
			refTerm1[_IM] = -2 * M_PI*freqTermX[i];
			refTerm2[_IM] = 1;
			refAS[i] = shadingFactor*((exp(refTerm1) - (Complex<Real>)1) / (2 * M_PI*freqTermX[i] * 2 * M_PI*freqTermX[i]) + (refTerm2 * exp(refTerm1)) / (2 * M_PI*freqTermX[i]));
		}
		else if (freqTermX[i] == 0 && freqTermY[i] != 0) {
			refTerm1[_IM] = 2 * M_PI*freqTermY[i];
			refTerm2[_IM] = 1;
			refAS[i] = shadingFactor*(((Complex<Real>)1 - exp(refTerm1)) / (4 * M_PI*M_PI*freqTermY[i] * freqTermY[i]) - refTerm2 / (2 * M_PI*freqTermY[i]));
		}
		else {
			refTerm1[_IM] = -2 * M_PI*freqTermX[i];
			refTerm2[_IM] = -2 * M_PI*(freqTermX[i] + freqTermY[i]);
			refAS[i] = shadingFactor*((exp(refTerm1) - (Complex<Real>)1) / (4 * M_PI*M_PI*freqTermX[i] * freqTermY[i]) + ((Complex<Real>)1 - exp(refTerm2)) / (4 * M_PI*M_PI*freqTermY[i] * (freqTermX[i] + freqTermY[i])));
		}
	}

	//randPhaseDist(refAS);

	return 1;
}

uint ophTri::refAS_Continuous(uint n)
{
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;

	vec3 av(0, 0, 0);
	av[0] = nv[3 * n + 0][0] * illumination[0] + nv[3 * n + 0][1] * illumination[1] + nv[3 * n + 0][2] * illumination[2] + 0.1;
	av[2] = nv[3 * n + 1][0] * illumination[0] + nv[3 * n + 1][1] * illumination[1] + nv[3 * n + 1][2] * illumination[2] + 0.1;
	av[1] = nv[3 * n + 2][0] * illumination[0] + nv[3 * n + 2][1] * illumination[1] + nv[3 * n + 2][2] * illumination[2] + 0.1;

	D1(0, 0);
	D2(0, 0);
	D3(0, 0);

	refTerm1(0, 0);
	refTerm2(0, 0);
	refTerm3(0, 0);
	
	for (int i = 0; i < pnXY; i++) {
		if (freqTermX[i] == 0 && freqTermY[i] == 0) {
			D1((Real)1 / (Real)3, 0);
			D2((Real)1 / (Real)5, 0);
			D3((Real)1 / (Real)2, 0);
		}
		else if (freqTermX[i] == 0 && freqTermY[i] != 0) {
			refTerm1[_IM] = -2 * M_PI*freqTermY[i];
			refTerm2[_IM] = 1;
			
			D1 = (refTerm1 - (Real)1)*refTerm1.exp() / (8 * M_PI*M_PI*M_PI*freqTermY[i] * freqTermY[i] * freqTermY[i]) 
				- refTerm1 / (4 * M_PI*M_PI*M_PI*freqTermY[i] * freqTermY[i] * freqTermY[i]);
			D2 = -(M_PI*freqTermY[i] + refTerm2) / (4 * M_PI*M_PI*M_PI*freqTermY[i] * freqTermY[i] * freqTermY[i])*exp(refTerm1) 
				+ refTerm1 / (8 * M_PI*M_PI*M_PI*freqTermY[i] * freqTermY[i] * freqTermY[i]);
			D3 = exp(refTerm1) / (2 * M_PI*freqTermY[i]) + ((Real)1 - refTerm2) / (2 * M_PI*freqTermY[i]);
		}
		else if (freqTermX[i] != 0 && freqTermY[i] == 0) {
			refTerm1[_IM] = 4 * M_PI*M_PI*freqTermX[i] * freqTermX[i];
			refTerm2[_IM] = 1;
			refTerm3[_IM] = 2 * M_PI*freqTermX[i];

			D1 = (refTerm1 + 4 * M_PI*freqTermX[i] - (Real)2 * refTerm2) / (8 * M_PI*M_PI*M_PI*freqTermY[i] * freqTermY[i] * freqTermY[i])*exp(-refTerm3) 
				+ refTerm2 / (4 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * freqTermX[i]);
			D2 = (Real)1 / (Real)2 * D1;
			D3 = ((refTerm3 + (Real)1)*exp(-refTerm3) - (Real)1) / (4 * M_PI*M_PI*freqTermX[i] * freqTermX[i]);
		}
		else if (freqTermX[i] == -freqTermY[i]) {
			refTerm1[_IM] = 1;
			refTerm2[_IM] = 2 * M_PI*freqTermX[i];
			refTerm3[_IM] = 2 * M_PI*M_PI*freqTermX[i] * freqTermX[i];

			D1 = (-2 * M_PI*freqTermX[i] + refTerm1) / (8 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * freqTermX[i])*exp(-refTerm2) 
				- (refTerm3 + refTerm1) / (8 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * freqTermX[i]);
			D2 = (-refTerm1) / (8 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * freqTermX[i])*exp(-refTerm2) 
				+ (-refTerm3 + refTerm1 + 2 * M_PI*freqTermX[i]) / (8 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * freqTermX[i]);
			D3 = (-refTerm1) / (4 * M_PI*M_PI*freqTermX[i] * freqTermX[i])*exp(-refTerm2) 
				+ (-refTerm2 + (Real)1) / (4 * M_PI*M_PI*freqTermX[i] * freqTermX[i]);
		}
		else {
			refTerm1[_IM] = -2 * M_PI*(freqTermX[i] + freqTermY[i]);
			refTerm2[_IM] = 1;
			refTerm3[_IM] = -2 * M_PI*freqTermX[i];

			D1 = exp(refTerm1)*(refTerm2 - 2 * M_PI*(freqTermX[i] + freqTermY[i])) / (8 * M_PI*M_PI*M_PI*freqTermY[i] * (freqTermX[i] + freqTermY[i])*(freqTermX[i] + freqTermY[i]))
				+ exp(refTerm3)*(2 * M_PI*freqTermX[i] - refTerm2) / (8 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * freqTermY[i])
				+ ((2 * freqTermX[i] + freqTermY[i])*refTerm2) / (8 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * (freqTermX[i] + freqTermY[i])*(freqTermX[i] + freqTermY[i]));
			D2 = exp(refTerm1)*(refTerm2*(freqTermX[i] + 2 * freqTermY[i]) - 2 * M_PI*freqTermY[i] * (freqTermX[i] + freqTermY[i])) / (8 * M_PI*M_PI*M_PI*freqTermY[i] * freqTermY[i] * (freqTermX[i] + freqTermY[i])*(freqTermX[i] + freqTermY[i]))
				+ exp(refTerm3)*(-refTerm2) / (8 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermY[i] * freqTermY[i])
				+ refTerm2 / (8 * M_PI*M_PI*M_PI*freqTermX[i] * (freqTermX[i] + freqTermY[i])* (freqTermX[i] + freqTermY[i]));
			D3 = -exp(refTerm1) / (4 * M_PI*M_PI*freqTermY[i] * (freqTermX[i] + freqTermY[i]))
				+ exp(refTerm3) / (4 * M_PI*M_PI*freqTermX[i] * freqTermY[i])
				- (Real)1 / (4 * M_PI*M_PI*freqTermX[i] * (freqTermX[i] + freqTermY[i]));
		}
		refAS[i] = (av[1] - av[0])*D1 + (av[2] - av[1])*D2 + av[0] * D3;
	}


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

uint ophTri::refToGlobal() 
{
	int Nx = context_.pixel_number[_X];
	int Ny = context_.pixel_number[_Y];

	Complex<Real> term1(0,0);
	Complex<Real> term2(0,0);

	Real det = geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2];
	
	if (det == 0)
		return -1;

	term1[_IM] = -2 * M_PI / context_.wave_length[0]*(
		carrierWave[_X] * (geom.glRot[0] * geom.glShift[_X] + geom.glRot[3] * geom.glShift[_Y] + geom.glRot[6] * geom.glShift[_Z])
		+ carrierWave[_Y] * (geom.glRot[1] * geom.glShift[_X] + geom.glRot[4] * geom.glShift[_Y] + geom.glRot[7] * geom.glShift[_Z])
		+ carrierWave[_Z] * (geom.glRot[2] * geom.glShift[_X] + geom.glRot[5] * geom.glShift[_Y] + geom.glRot[8] * geom.glShift[_Z]));
	Complex<Real> temp(0,0);

	for (int i = 0; i < Nx*Ny; i++) {
		if (fz[i] == 0)
			temp = 0;
		else {
			term2[_IM] = 2 * M_PI*(flx[i] * geom.glShift[_X] + fly[i] * geom.glShift[_Y] + flz[i] * geom.glShift[_Z]);
			temp = refAS[i] / det * exp(term1)* flz[i] / fz[i] * exp(term2);
		}
		if (abs(temp) > MIN_DOUBLE) {}
		else { temp = 0; }
		angularSpectrum[i] += temp;	
	}

	return 1;
}