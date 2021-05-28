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

#include "ophTriMesh_GPU.h"


void ophTri::initialize_GPU()
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

	if (!streamTriMesh)
		cudaStreamCreate(&streamTriMesh);


	if (angularSpectrum_GPU)   cudaFree(angularSpectrum_GPU);
	HANDLE_ERROR(cudaMalloc((void**)&angularSpectrum_GPU, sizeof(cufftDoubleComplex) * pnXY));
	
	if (ffttemp)   cudaFree(ffttemp);
	HANDLE_ERROR(cudaMalloc((void**)&ffttemp, sizeof(cufftDoubleComplex) * pnXY));

}
void ophTri::generateAS_GPU(uint SHADING_FLAG)
{
	if (SHADING_FLAG != SHADING_FLAT && SHADING_FLAG != SHADING_CONTINUOUS) {
		LOG("error: WRONG SHADING_FLAG\n");
		exit(0);
	}

	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const uint pnXY = pnX * pnY;
	int N = meshData->n_faces;

	Real* mesh = new Real[9];

	uint nChannel = context_.waveNum;

	cufftDoubleComplex* output = new cufftDoubleComplex[pnXY];

	for (uint ch = 0; ch < nChannel; ch++) {
		
		findNormals(SHADING_FLAG);

		HANDLE_ERROR(cudaMemsetAsync(angularSpectrum_GPU, 0, sizeof(cufftDoubleComplex) * pnXY, streamTriMesh));

		for (int j = 0; j < N; j++) {
			memcpy(mesh, &scaledMeshData[9 * j], sizeof(Real) * 9);

			if (!checkValidity(mesh,no[j])) // Ignore Invalid
				continue;

			if (!findGeometricalRelations(mesh, no[j]))
				continue;

			refAS_GPU(j, ch, SHADING_FLAG);

			m_nProgress = (int)((Real)(ch * N + j + 1) * 50 / ((Real)N * nChannel));
		}

		HANDLE_ERROR(cudaMemsetAsync(ffttemp, 0, sizeof(cufftDoubleComplex) * pnXY, streamTriMesh));
		call_fftGPU(pnX, pnY, angularSpectrum_GPU, ffttemp, streamTriMesh);


		// GPU Memory -> CPU Memory
		HANDLE_ERROR(cudaMemcpyAsync(output, ffttemp, sizeof(cufftDoubleComplex) * pnXY, cudaMemcpyDeviceToHost, streamTriMesh));

		for (int i = 0; i < pnXY; ++i)
		{
			complex_H[ch][i][_RE] = output[i].x;
			complex_H[ch][i][_IM] = output[i].y;
		}
	}

	m_nProgress = 100;
	delete[] output, mesh, scaledMeshData, no, na, nv;
}


void ophTri::refAS_GPU(int idx, int ch, uint SHADING_FLAG)
{/*
 int nx = context_.pixel_number[_X];
 int ny = context_.pixel_number[_Y];
 double px = context_.pixel_pitch[_X];
 double py = context_.pixel_pitch[_Y];
 double waveLength = context_.wave_length[ch];

 shadingFactor = 0;
 vec3 av(0, 0, 0);

 if (SHADING_FLAG == SHADING_FLAT) {
 vec3 no_ = no[idx];
 n = no_ / norm(no_);
 if (illumination[_X] == 0 && illumination[_Y] == 0 && illumination[_Z] == 0) {
 shadingFactor = 1;
 }
 else {
 vec3 normIllu = illumination / norm(illumination);
 shadingFactor = 2 * (n[_X] * normIllu[_X] + n[_Y] * normIllu[_Y] + n[_Z] * normIllu[_Z]) + 0.3;
 if (shadingFactor < 0)
 shadingFactor = 0;
 }
 }
 else if (SHADING_FLAG == SHADING_CONTINUOUS) {

 av[0] = nv[3 * idx + 0][0] * illumination[0] + nv[3 * idx + 0][1] * illumination[1] + nv[3 * idx + 0][2] * illumination[2] + 0.1;
 av[2] = nv[3 * idx + 1][0] * illumination[0] + nv[3 * idx + 1][1] * illumination[1] + nv[3 * idx + 1][2] * illumination[2] + 0.1;
 av[1] = nv[3 * idx + 2][0] * illumination[0] + nv[3 * idx + 2][1] * illumination[1] + nv[3 * idx + 2][2] * illumination[2] + 0.1;


 }

 double min_double = (double)2.2250738585072014e-308;
 double tolerence = 1e-12;

 call_cudaKernel_refAS(angularSpectrum_GPU, nx, ny, px, py, SHADING_FLAG, idx, waveLength, M_PI, shadingFactor, av[0], av[1], av[2],
 geom.glRot[0], geom.glRot[1], geom.glRot[2], geom.glRot[3], geom.glRot[4], geom.glRot[5], geom.glRot[6], geom.glRot[7], geom.glRot[8],
 geom.loRot[0], geom.loRot[1], geom.loRot[2], geom.loRot[3], geom.glShift[_X], geom.glShift[_Y], geom.glShift[_Z],
 carrierWave[_X], carrierWave[_Y], carrierWave[_Z], min_double, tolerence, streamTriMesh);
 */
}
