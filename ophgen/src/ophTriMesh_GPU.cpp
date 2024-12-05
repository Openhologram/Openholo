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
#include "cudaWrapper.h"

using namespace oph;

void ophTri::initialize_GPU()
{
	const long long int pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];
	const int N = meshData->n_faces;

	if (scaledMeshData) {
		delete[] scaledMeshData;
		scaledMeshData = nullptr;
	}
	
	//scaledMeshData = new Real[N * 9];
	//memset(scaledMeshData, 0, sizeof(Real) * N * 9);
	scaledMeshData = new Face[N];
	memset(scaledMeshData, 0, sizeof(Face) * N);

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
	const uint pnX = context_.pixel_number[_X];
	const uint pnY = context_.pixel_number[_Y];
	const long long int pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];
	int N = meshData->n_faces;
	uint nChannel = context_.waveNum;

	cufftDoubleComplex* output = new cufftDoubleComplex[pnXY];

	findNormals(SHADING_FLAG);

	MeshKernelConfig* device_config = nullptr;
	cudaMalloc((void**)&device_config, sizeof(MeshKernelConfig));
	geometric* device_geom = nullptr;
	cudaMalloc((void**)&device_geom, sizeof(geometric));

	int nBlockThreads = cudaWrapper::getInstance()->getMaxThreads(0) >> 1;
	int nBlocks = (pnX * pnY + nBlockThreads - 1) / nBlockThreads;

	for (uint ch = 0; ch < nChannel; ch++) {	

		HANDLE_ERROR(cudaMemsetAsync(angularSpectrum_GPU, 0, sizeof(cufftDoubleComplex) * pnXY, streamTriMesh));
		geometric geom;

		MeshKernelConfig* host_config = new MeshKernelConfig(
			context_.pixel_number,
			context_.pixel_pitch,
			context_.wave_length[ch],
			SHADING_FLAG
		);

		cudaMemcpy(device_config, host_config, sizeof(MeshKernelConfig), cudaMemcpyHostToDevice);

		for (int j = 0; j < N; j++)
		{
			if (!checkValidity(no[j])) // Ignore Invalid
				continue;

			if (!findGeometricalRelations(scaledMeshData[j], no[j], geom))
				continue;

			cudaMemcpy(device_geom, (void*)&geom, sizeof(geometric), cudaMemcpyHostToDevice);

			Real shadingFactor = 0;
			vec3 av(0, 0, 0);

			if (SHADING_FLAG == SHADING_FLAT)
			{
				vec3 no_ = no[j];
				vec3 n = no_ / norm(no_);
				if (illumination[_X] == 0 && illumination[_Y] == 0 && illumination[_Z] == 0) {
					shadingFactor = 1;
				}
				else {
					vec3 normIllu = illumination / norm(illumination);
					shadingFactor = 2 * (n[_X] * normIllu[_X] + n[_Y] * normIllu[_Y] + n[_Z] * normIllu[_Z]) + 0.3;
					if (shadingFactor < 0) shadingFactor = 0;
				}

				cudaMesh_Flat(nBlocks, nBlockThreads, angularSpectrum_GPU, device_config, shadingFactor,
					device_geom, carrierWave[_X], carrierWave[_Y], carrierWave[_Z], streamTriMesh);
			}
			else if (SHADING_FLAG == SHADING_CONTINUOUS)
			{
				av[0] = nv[3 * j + 0][0] * illumination[0] + nv[3 * j + 0][1] * illumination[1] + nv[3 * j + 0][2] * illumination[2] + 0.1;
				av[2] = nv[3 * j + 1][0] * illumination[0] + nv[3 * j + 1][1] * illumination[1] + nv[3 * j + 1][2] * illumination[2] + 0.1;
				av[1] = nv[3 * j + 2][0] * illumination[0] + nv[3 * j + 2][1] * illumination[1] + nv[3 * j + 2][2] * illumination[2] + 0.1;

				cudaMesh_Continuous(nBlocks, nBlockThreads, angularSpectrum_GPU, device_config,
					device_geom, av[0], av[1], av[2], carrierWave[_X], carrierWave[_Y], carrierWave[_Z], streamTriMesh);
			}
			m_nProgress = (int)((Real)(ch * N + j + 1) * 50 / ((Real)N * nChannel));
		}

		HANDLE_ERROR(cudaMemcpyAsync(output, angularSpectrum_GPU, sizeof(cufftDoubleComplex) * pnXY, cudaMemcpyDeviceToHost, streamTriMesh));

		for (int i = 0; i < pnXY; ++i)
		{
			complex_H[ch][i][_RE] = output[i].x;
			complex_H[ch][i][_IM] = output[i].y;
		}

		delete host_config;
	}

	cudaFree(device_geom);
	cudaFree(device_config);
	cudaFree(angularSpectrum_GPU);
	cudaFree(fftTemp);

	cudaStreamDestroy(streamTriMesh);
	streamTriMesh = nullptr;

	m_nProgress = 100;
	if (output != nullptr)
	{
		delete[] output;
		output = nullptr;
	}
	if (scaledMeshData != nullptr)
	{
		delete[] scaledMeshData;
		scaledMeshData = nullptr;
	}
	if (no != nullptr)
	{
		delete[] no;
		no = nullptr;
	}
	if (na != nullptr)
	{
		delete[] na;
		na = nullptr;
	}
	if (nv != nullptr)
	{
		delete[] nv;
		nv = nullptr;
	}
}