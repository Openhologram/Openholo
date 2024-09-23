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

#include "ophPointCloud.h"
#include "ophPointCloud_GPU.h"
#include "CUDA.h"

#ifdef _USE_OPENCL
#include "OpenCL.h"

void ophPointCloud::genCghPointCloudGPU(uint diff_flag)
{
	int nErr;
	auto begin = CUR_TIME;
	OpenCL* cl = OpenCL::getInstance();

	cl_context context = cl->getContext();
	cl_command_queue commands = cl->getCommand();
	cl_mem d_pc_data;
	cl_mem d_amp_data;
	cl_mem d_result;
	cl_mem d_config;

	//threads number
	const ulonglong pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];
	const ulonglong bufferSize = pnXY * sizeof(Real);

	//Host Memory Location
	const int n_colors = pc_data_.n_colors;
	Real* h_pc_data = nullptr;
	Real* h_amp_data = pc_data_.color;
	Real* h_dst = nullptr;

	// Keep original buffer
	if (is_ViewingWindow) {
		h_pc_data = new Real[n_points * 3];
		transVW(n_points * 3, h_pc_data, pc_data_.vertex);
	}
	else {
		h_pc_data = pc_data_.vertex;
	}

	uint nChannel = context_.waveNum;
	bool bIsGrayScale = n_colors == 1 ? true : false;

	cl->LoadKernel();

	cl_kernel* kernel = cl->getKernel();

	cl_kernel* current_kernel = nullptr;
	if ((diff_flag == PC_DIFF_RS) || (diff_flag == PC_DIFF_FRESNEL)) {
		h_dst = new Real[pnXY * 2];
		memset(h_dst, 0., bufferSize * 2);

		current_kernel = diff_flag == PC_DIFF_RS ? &kernel[0] : &kernel[1];

		d_pc_data = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(Real) * n_points * 3, nullptr, &nErr);
		d_amp_data = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(Real) * n_points * n_colors, nullptr, &nErr);
		nErr = clEnqueueWriteBuffer(commands, d_pc_data, CL_TRUE, 0, sizeof(Real) * n_points * 3, h_pc_data, 0, nullptr, nullptr);
		nErr = clEnqueueWriteBuffer(commands, d_amp_data, CL_TRUE, 0, sizeof(Real) * n_points * n_colors, h_amp_data, 0, nullptr, nullptr);

		d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(Real) * pnXY * 2, nullptr, &nErr);


		size_t global[2] = { context_.pixel_number[_X], context_.pixel_number[_Y] };
		size_t local[2] = { 32, 32 };

		clSetKernelArg(*current_kernel, 1, sizeof(cl_mem), &d_pc_data);
		clSetKernelArg(*current_kernel, 2, sizeof(cl_mem), &d_amp_data);
		clSetKernelArg(*current_kernel, 4, sizeof(uint), &n_points);
		for (uint ch = 0; ch < nChannel; ch++)
		{
			uint nAdd = bIsGrayScale ? 0 : ch;
			context_.k = (2 * M_PI) / context_.wave_length[ch];
			Real ratio = 1.0; //context_.wave_length[nChannel - 1] / context_.wave_length[ch];

			GpuConst* h_config = new GpuConst(
				n_points, n_colors, 1,
				pc_config_.scale, pc_config_.distance,
				context_.pixel_number,
				context_.pixel_pitch,
				context_.ss,
				context_.k,
				context_.wave_length[ch],
				ratio
			);

			if (diff_flag == PC_DIFF_RS)
			{
				h_config = new GpuConstNERS(*h_config);
				d_config = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(GpuConstNERS), nullptr, &nErr);

				nErr = clEnqueueWriteBuffer(commands, d_result, CL_TRUE, 0, sizeof(Real) * pnXY * 2, h_dst, 0, nullptr, nullptr);
				nErr = clEnqueueWriteBuffer(commands, d_config, CL_TRUE, 0, sizeof(GpuConstNERS), h_config, 0, nullptr, nullptr);
			}
			else if (diff_flag == PC_DIFF_FRESNEL)
			{
				h_config = new GpuConstNEFR(*h_config);
				d_config = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(GpuConstNEFR), nullptr, &nErr);

				nErr = clEnqueueWriteBuffer(commands, d_result, CL_TRUE, 0, sizeof(Real) * pnXY * 2, h_dst, 0, nullptr, nullptr);
				nErr = clEnqueueWriteBuffer(commands, d_config, CL_TRUE, 0, sizeof(GpuConstNEFR), h_config, 0, nullptr, nullptr);
			}

			clSetKernelArg(*current_kernel, 0, sizeof(cl_mem), &d_result);
			clSetKernelArg(*current_kernel, 3, sizeof(cl_mem), &d_config);
			clSetKernelArg(*current_kernel, 5, sizeof(uint), &ch);

			nErr = clEnqueueNDRangeKernel(commands, *current_kernel, 2, nullptr, global, nullptr/*local*/, 0, nullptr, nullptr);


			//nErr = clFlush(commands);
			nErr = clFinish(commands);

			if (nErr != CL_SUCCESS) cl->errorCheck(nErr, "Check", __FILE__, __LINE__);

			nErr = clEnqueueReadBuffer(commands, d_result, CL_TRUE, 0, sizeof(Real) * pnXY * 2, complex_H[ch], 0, nullptr, nullptr);

			delete h_config;

			m_nProgress = (ch + 1) * 100 / nChannel;
		}

		clReleaseMemObject(d_result);
		clReleaseMemObject(d_amp_data);
		clReleaseMemObject(d_pc_data);
		if (h_dst)	delete[] h_dst;
		if (is_ViewingWindow && h_pc_data)	delete[] h_pc_data;
	}

	LOG("%s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
}
#endif

using namespace oph;
void ophPointCloud::genCghPointCloudGPU(uint diff_flag)
{
	if ((diff_flag != PC_DIFF_RS) && (diff_flag != PC_DIFF_FRESNEL))
	{
		LOG("<FAILED> Wrong parameters.");
		return;
	}
	CUDA* pCuda = CUDA::getInstance();

	auto begin = CUR_TIME;
	const ulonglong pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];
	int blockSize = pCuda->getMaxThreads(0); //n_threads // blockSize < devProp.maxThreadsPerBlock
	ulonglong gridSize = (pnXY + blockSize - 1) / blockSize; //n_blocks

	cout << ">>> All " << blockSize * gridSize << " threads in CUDA" << endl;
	cout << ">>> " << blockSize << " threads/block, " << gridSize << " blocks/grid" << endl;


	//Host Memory Location
	Vertex* h_vertex_data = nullptr;
	if (!is_ViewingWindow)
		h_vertex_data = pc_data_.vertices;
	else
	{
		h_vertex_data = new Vertex[pc_data_.n_points];
		std::memcpy(h_vertex_data, pc_data_.vertices, sizeof(Vertex) * pc_data_.n_points);
		transVW(pc_data_.n_points, h_vertex_data, h_vertex_data);
	}

	//threads number
	const ulonglong bufferSize = pnXY * sizeof(cuDoubleComplex);
	int gpu_num = pCuda->getActiveGPUs();

	uint nChannel = context_.waveNum;
	// cacl workload
	pCuda->setWorkload(pc_data_.n_points);


	// default
	if (gpu_num == 1) {
		cuDoubleComplex* d_dst = nullptr;
		HANDLE_ERROR(cudaMalloc((void**)&d_dst, bufferSize));
		HANDLE_ERROR(cudaMemset(d_dst, 0., bufferSize));

		Vertex* d_vertex_data = nullptr;
		pCuda->printMemoryInfo(0);
		HANDLE_ERROR(cudaMalloc((void**)&d_vertex_data, pc_data_.n_points * sizeof(Vertex)));
		CudaPointCloudConfig *base_config = new CudaPointCloudConfig(
			pc_data_.n_points,
			pc_config_.scale,
			pc_config_.distance,
			context_.pixel_number,
			context_.offset,
			context_.pixel_pitch,
			context_.ss,
			context_.k,
			context_.wave_length[0]
		);

		HANDLE_ERROR(cudaMemcpy(d_vertex_data, h_vertex_data, pc_data_.n_points * sizeof(Vertex), cudaMemcpyHostToDevice));

		for (uint ch = 0; ch < nChannel; ch++)
		{
			base_config->k = context_.k = (2 * M_PI) / context_.wave_length[ch];
			base_config->lambda = context_.wave_length[ch];
			CudaPointCloudConfig* h_config = nullptr;
			CudaPointCloudConfig* d_config = nullptr;
			switch (diff_flag) {
			case PC_DIFF_RS: {
				h_config = new CudaPointCloudConfigRS(*base_config);
				HANDLE_ERROR(cudaMalloc((void**)&d_config, sizeof(CudaPointCloudConfigRS)));
				HANDLE_ERROR(cudaMemcpy(d_config, h_config, sizeof(CudaPointCloudConfigRS), cudaMemcpyHostToDevice));
				cudaPointCloud_RS(gridSize, blockSize, d_vertex_data, d_dst, (CudaPointCloudConfigRS*)d_config, ch, m_mode);
				break;
			}
			case PC_DIFF_FRESNEL: {
				h_config = new CudaPointCloudConfigFresnel(*base_config);
				HANDLE_ERROR(cudaMalloc((void**)&d_config, sizeof(CudaPointCloudConfigFresnel)));
				HANDLE_ERROR(cudaMemcpy(d_config, h_config, sizeof(CudaPointCloudConfigFresnel), cudaMemcpyHostToDevice));
				cudaPointCloud_Fresnel(gridSize, blockSize, d_vertex_data, d_dst, (CudaPointCloudConfigFresnel*)d_config, ch, m_mode);
				break;
			}
			}
			cudaError error = cudaGetLastError();
			if (error != cudaSuccess) {
				LOG("cudaGetLastError(): %s\n", cudaGetErrorName(error));
				if (error == cudaErrorLaunchOutOfResources) {
					ch--;
					blockSize /= 2;
					gridSize *= 2;
					continue;
				}
			}
			HANDLE_ERROR(cudaMemcpy(complex_H[ch], d_dst, bufferSize, cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemset(d_dst, 0., bufferSize));
			HANDLE_ERROR(cudaFree(d_config));
			delete h_config;

			m_nProgress = (ch + 1) * 100 / nChannel;
		}
		HANDLE_ERROR(cudaFree(d_vertex_data));
		HANDLE_ERROR(cudaFree(d_dst));
		delete base_config;

		if (is_ViewingWindow) {
			delete[] h_vertex_data;
		}
	}
	// multi gpu
	else {
		int current_idx = 0;
		cuDoubleComplex* d_dst = nullptr;
		//Complex<Real>* h_dst[MAX_GPU];
		cuDoubleComplex* d_tmp[MAX_GPU];
		Vertex* d_vertex_data[MAX_GPU];
		CudaPointCloudConfig* d_config[MAX_GPU];// = nullptr;

		HANDLE_ERROR(cudaSetDevice(0));
		HANDLE_ERROR(cudaMalloc((void**)&d_dst, bufferSize));
		HANDLE_ERROR(cudaMemset(d_dst, 0., bufferSize));

		CudaPointCloudConfig* base_config = new CudaPointCloudConfig(
			pc_data_.n_points,
			pc_config_.scale,
			pc_config_.distance,
			context_.pixel_number,
			context_.offset,
			context_.pixel_pitch,
			context_.ss,
			context_.k,
			context_.wave_length[0]
		);

		for (int devID = 0; devID < gpu_num && devID < MAX_GPU; devID++)
		{
			pCuda->printMemoryInfo(devID);
			HANDLE_ERROR(cudaSetDevice(devID));
			int n_point = pCuda->getWorkload(devID);
			int n_size = n_point * sizeof(Vertex);

			HANDLE_ERROR(cudaMalloc((void**)&d_vertex_data[devID], n_size));
			HANDLE_ERROR(cudaMemcpy((void*)d_vertex_data[devID], (const void *)&h_vertex_data[current_idx], n_size, cudaMemcpyHostToDevice));

			HANDLE_ERROR(cudaMalloc((void**)&d_tmp[devID], bufferSize));
			HANDLE_ERROR(cudaMemset(d_tmp[devID], 0., bufferSize));

			switch (diff_flag) {
			case PC_DIFF_RS: {
				HANDLE_ERROR(cudaMalloc((void**)&d_config[devID], sizeof(CudaPointCloudConfigRS)));
				break;
			}
			case PC_DIFF_FRESNEL: {
				HANDLE_ERROR(cudaMalloc((void**)&d_config[devID], sizeof(CudaPointCloudConfigFresnel)));
				break;
			}
			}

			current_idx += n_point;
		}

		for (uint ch = 0; ch < nChannel; ch++)
		{
			base_config->k = context_.k = (2 * M_PI) / context_.wave_length[ch];
			base_config->lambda = context_.wave_length[ch];
			current_idx = 0;
			for (int devID = 0; devID < gpu_num && devID < MAX_GPU; devID++)
			{
				int n_point = pCuda->getWorkload(devID);
				HANDLE_ERROR(cudaSetDevice(devID));
				base_config->n_points = n_point;

				HANDLE_ERROR(cudaMemset(d_tmp[devID], 0., bufferSize));
				CudaPointCloudConfig* h_config = nullptr;

				switch (diff_flag) {
				case PC_DIFF_RS: {
					h_config = new CudaPointCloudConfigRS(*base_config);
					HANDLE_ERROR(cudaMemcpy(d_config[devID], h_config, sizeof(CudaPointCloudConfigRS), cudaMemcpyHostToDevice));
					cudaPointCloud_RS(gridSize, blockSize, d_vertex_data[devID], d_tmp[devID], (CudaPointCloudConfigRS*)d_config[devID], ch, m_mode);
					break;
				}
				case PC_DIFF_FRESNEL: {
					h_config = new CudaPointCloudConfigFresnel(*base_config);
					HANDLE_ERROR(cudaMemcpy(d_config[devID], h_config, sizeof(CudaPointCloudConfigFresnel), cudaMemcpyHostToDevice));
					cudaPointCloud_Fresnel(gridSize, blockSize, d_vertex_data[devID], d_tmp[devID], (CudaPointCloudConfigFresnel*)d_config[devID], ch, m_mode);
					break;
				}
				}
				delete h_config;
			}

			// wait for cudaPointCloud_* kernel
			for (int devID = 0; devID < gpu_num && devID < MAX_GPU; devID++)
			{
				HANDLE_ERROR(cudaSetDevice(devID));
				cudaDeviceSynchronize();
			}

			HANDLE_ERROR(cudaSetDevice(0));
			for (int devID = 0; devID < gpu_num && devID < MAX_GPU; devID++)
			{
				if (devID != 0)
				{
					// GPU N to GPU 0
					cudaMemcpyPeer(d_tmp[0], 0, d_tmp[devID], devID, bufferSize);
				}

				sum_Kernel(gridSize, blockSize, d_dst, d_tmp[0], pnXY);
				cudaDeviceSynchronize();
			}
			HANDLE_ERROR(cudaMemcpy(complex_H[ch], d_dst, bufferSize, cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemset(d_dst, 0, bufferSize));

			m_nProgress = (ch + 1) * 100 / nChannel;
		}

		for (int devID = 0; devID < gpu_num && devID < MAX_GPU; devID++)
		{
			//delete[] h_dst[devID];
			cudaSetDevice(devID);
			HANDLE_ERROR(cudaFree(d_config[devID]));
			HANDLE_ERROR(cudaFree(d_vertex_data[devID]));
			HANDLE_ERROR(cudaFree(d_tmp[devID]));
		}

		HANDLE_ERROR(cudaSetDevice(0));
		HANDLE_ERROR(cudaFree(d_dst));
		delete base_config;
		if (is_ViewingWindow) {
			delete[] h_vertex_data;
		}
	}

	LOG("%s : %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
}