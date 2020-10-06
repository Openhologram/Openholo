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
//#include "ophPCKernel.cl"
#include <sys.h> //for LOG() macro
#if 0
#include "OpenCL.h"


Real ophPointCloud::genCghPointCloudGPU(uint diff_flag)
{
	int nErr;
	auto begin = CUR_TIME;
	OpenCL *cl = OpenCL::getInstance();
	cl_context context = cl->getContext();
	cl_command_queue commands = cl->getCommand();
	cl_kernel *kernel = cl->getKernel();
	cl_mem device_pc_data;
	cl_mem device_amp_data;
	cl_mem result;
	cl_mem device_config;

	//threads number
	const ulonglong pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];
	const ulonglong bufferSize = pnXY * sizeof(Real);

	//Host Memory Location
	const int n_colors = pc_data_.n_colors;
	Real* host_pc_data = nullptr;
	Real* host_amp_data = pc_data_.color;
	Real* host_dst = nullptr;

	// Keep original buffer
	if (is_ViewingWindow) {
		host_pc_data = new Real[n_points * 3];
		transVW(n_points * 3, host_pc_data, pc_data_.vertex);
	}
	else {
		host_pc_data = pc_data_.vertex;
	}

	if ((diff_flag == PC_DIFF_RS) || (diff_flag == PC_DIFF_FRESNEL)) {
		host_dst = new Real[pnXY * 2];
		memset(host_dst, 0., bufferSize * 2);
	}
	context_.k = (2 * M_PI) / context_.wave_length[0];
	GpuConst* host_config = new GpuConst(
		n_points, n_colors, pc_config_.n_streams,
		pc_config_.scale, pc_config_.offset_depth,
		context_.pixel_number,
		context_.pixel_pitch,
		context_.ss,
		context_.k
	);

	host_config = new GpuConstNERS(*host_config, context_.wave_length[0]);
	device_config = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(GpuConstNERS), nullptr, &nErr);
	device_pc_data = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(Real) * n_points * 3, nullptr, &nErr);
	device_amp_data = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(Real) * n_points * n_colors, nullptr, &nErr);
	result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(Real) * pnXY * 2, nullptr, &nErr);
	Real *host_result = new Real[pnXY * 2];
	memset(host_result, 0, sizeof(Real) * pnXY * 2);
	nErr = clEnqueueWriteBuffer(commands, device_config, CL_TRUE, 0, sizeof(GpuConstNERS), host_config, 0, nullptr, nullptr);
	nErr = clEnqueueWriteBuffer(commands, device_pc_data, CL_TRUE, 0, sizeof(Real) * n_points * 3, host_pc_data, 0, nullptr, nullptr);
	nErr = clEnqueueWriteBuffer(commands, device_amp_data, CL_TRUE, 0, sizeof(Real) * n_points * n_colors, host_amp_data, 0, nullptr, nullptr);
	clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &result);
	clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &device_pc_data);
	clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &device_amp_data);
	clSetKernelArg(kernel[0], 3, sizeof(cl_mem), &device_config);
	clSetKernelArg(kernel[0], 4, sizeof(uint), &pnXY);

	size_t global[2] = { 1024, 1024 };
	size_t local[2] = { 32, 32 };
	LOG("Group Size: <Global:%d> <Local:%d>\n", global, local);
	nErr = clEnqueueNDRangeKernel(commands, kernel[0], 2, nullptr, global, local, 0, nullptr, nullptr);

	nErr = clFlush(commands);
	nErr = clFinish(commands);

	if (nErr != CL_SUCCESS) cl->errorCheck(nErr, "Check", __FILE__, __LINE__);

	nErr = clEnqueueReadBuffer(commands, result, CL_TRUE, 0, sizeof(Real) * pnXY * 2, host_result, 0, nullptr, nullptr);
	memcpy(complex_H[0], host_result, sizeof(Real) * pnXY * 2);
	/*
	

	cl_mem d_a;
	cl_mem d_b;
	cl_mem d_result;
	int cnt = 1024;
	int *a = new int[cnt];
	int *b = new int[cnt];
	int *result = new int[cnt];
	ZeroMemory(a, cnt);
	ZeroMemory(b, cnt);
	ZeroMemory(result, cnt);

	for (int i = 0; i < cnt; i++) {
		a[i] = i % cnt;
		b[i] = i / 2;
	}

	d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * cnt, nullptr, &nErr);
	d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * cnt, nullptr, &nErr);
	d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * cnt, nullptr, &nErr);
	nErr = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(int) * cnt, a, 0, nullptr, nullptr);
	nErr = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(int) * cnt, b, 0, nullptr, nullptr);

	nErr = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &d_a);
	nErr != clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &d_b);
	nErr != clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &d_result);
	nErr != clSetKernelArg(kernel[0], 3, sizeof(unsigned int), &cnt);
	// 1 CU : 1024x1024x64 (28 CU)
	size_t global[2] = { 1920, 1080 };
	auto begin = CUR_TIME;
	nErr = clEnqueueNDRangeKernel(commands, kernel[0], 2, nullptr, global, nullptr, 0, nullptr, nullptr);
	nErr = clFinish(commands);
	auto end = CUR_TIME;
	LOG("%lf\n", ((std::chrono::duration<Real>)(end - begin)).count());

	nErr = clEnqueueReadBuffer(commands, d_result, CL_TRUE, 0, sizeof(int) * cnt, result, 0, nullptr, nullptr);

	//for (int i = 0; i < cnt; i++) {
	//	LOG("%d\n", result[i]);
	//}

	clReleaseMemObject(d_a);
	clReleaseMemObject(d_b);
	clReleaseMemObject(d_result);
	delete[] a;
	delete[] b;
	delete[] result;
	// do something
	
	*/

	clReleaseMemObject(result);
	clReleaseMemObject(device_amp_data);
	clReleaseMemObject(device_pc_data);
	delete host_config;
	if (host_dst)	delete[] host_dst;
	if (is_ViewingWindow && host_pc_data)	delete[] host_pc_data;
	if (host_result) delete[] host_result;
	auto end = CUR_TIME;
	Real elapsed_time = ((chrono::duration<Real>)(end - begin)).count();
	LOG("\n%s : %lf(s) \n\n",
		__FUNCTION__,
		elapsed_time);
	return elapsed_time;
}
#else


Real ophPointCloud::genCghPointCloudGPU(uint diff_flag)
{
	auto begin = CUR_TIME;
	const ulonglong pnXY = context_.pixel_number[_X] * context_.pixel_number[_Y];
	int blockSize = CUDA::getInstance()->getMaxThreads(); //n_threads // blockSize < devProp.maxThreadsPerBlock
	ulonglong gridSize = (pnXY + blockSize - 1) / blockSize; //n_blocks

	cout << ">>> All " << blockSize * gridSize << " threads in CUDA" << endl;
	cout << ">>> " << blockSize << " threads/block, " << gridSize << " blocks/grid" << endl;

	//const int n_streams = OPH_CUDA_N_STREAM;
	int n_streams;
	if (getStream() == 0)
		n_streams = pc_data_.n_points / 300 + 1;
	else if (getStream() < 0)
	{
		LOG("Invalid value : NumOfStream");
		return 0.0;
	}
	else
		n_streams = getStream();

	LOG(">>> Number Of Stream : %d\n", n_streams);

	//threads number
	const ulonglong bufferSize = pnXY * sizeof(Real);

	//Host Memory Location
	const int n_colors = pc_data_.n_colors;
	Real* host_pc_data = nullptr;
	Real* host_amp_data = pc_data_.color;
	Real* host_dst = nullptr;

	// Keep original buffer
	if (is_ViewingWindow) {
		host_pc_data = new Real[n_points * 3];
		transVW(n_points * 3, host_pc_data, pc_data_.vertex);
	}
	else {
		host_pc_data = pc_data_.vertex;
	}
	
	if ((diff_flag == PC_DIFF_RS) || (diff_flag == PC_DIFF_FRESNEL)) {
		host_dst = new Real[pnXY * 2];
	}

	uint nChannel = context_.waveNum;
	bool bIsGrayScale = n_colors == 1 ? true : false;

	for (uint ch = 0; ch < nChannel; ch++)
	{
		uint nAdd = bIsGrayScale ? 0 : ch;
		memset(host_dst, 0., bufferSize * 2);
		context_.k = (2 * M_PI) / context_.wave_length[ch];
		Real ratio = context_.wave_length[nChannel - 1] / context_.wave_length[ch];

		GpuConst* host_config = new GpuConst(
			n_points, n_colors, n_streams,
			pc_config_.scale, pc_config_.distance,
			context_.pixel_number,
			context_.pixel_pitch,
			context_.ss,
			context_.k,
			context_.wave_length[ch],
			ratio
		);

		//Device(GPU) Memory Location
		Real* device_pc_data;
		HANDLE_ERROR(cudaMalloc((void**)&device_pc_data, n_points * 3 * sizeof(Real)));
		
		Real* device_amp_data;
		HANDLE_ERROR(cudaMalloc((void**)&device_amp_data, n_points * n_colors * sizeof(Real)));

		Real* device_dst = nullptr;
		if ((diff_flag == PC_DIFF_RS) || (diff_flag == PC_DIFF_FRESNEL)) {
			HANDLE_ERROR(cudaMalloc((void**)&device_dst, bufferSize * 2));
#ifndef _DEBUG
			HANDLE_ERROR(cudaMemsetAsync(device_dst, 0., bufferSize * 2));
#else
			HANDLE_ERROR(cudaMemset(device_dst, 0., bufferSize * 2));
#endif
		}

		GpuConst* device_config = nullptr;
		switch (diff_flag) {
		case PC_DIFF_RS: {
			host_config = new GpuConstNERS(*host_config);
			HANDLE_ERROR(cudaMalloc((void**)&device_config, sizeof(GpuConstNERS)));
#ifndef _DEBUG
			HANDLE_ERROR(cudaMemcpyAsync(device_config, host_config, sizeof(GpuConstNERS), cudaMemcpyHostToDevice));
#else
			HANDLE_ERROR(cudaMemcpy(device_config, host_config, sizeof(GpuConstNERS), cudaMemcpyHostToDevice));
#endif
			break;
		}
		case PC_DIFF_FRESNEL: {
			host_config = new GpuConstNEFR(*host_config);
			HANDLE_ERROR(cudaMalloc((void**)&device_config, sizeof(GpuConstNEFR)));
#ifndef _DEBUG
			HANDLE_ERROR(cudaMemcpyAsync(device_config, host_config, sizeof(GpuConstNEFR), cudaMemcpyHostToDevice));
#else
			HANDLE_ERROR(cudaMemcpy(device_config, host_config, sizeof(GpuConstNEFR), cudaMemcpyHostToDevice));
#endif
			break;
		}
		}

		int stream_points = n_points / n_streams;
		int remainder = n_points % n_streams;

		int offset = 0;
		for (int i = 0; i < n_streams; ++i) {
			offset = i * stream_points;
			if (i == n_streams - 1) { // 마지막 스트림 연산 시,
				stream_points += remainder;
			}
#ifndef _DEBUG
			HANDLE_ERROR(cudaMemcpyAsync(device_pc_data + 3 * offset, host_pc_data + 3 * offset, stream_points * 3 * sizeof(Real), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpyAsync(device_amp_data + n_colors * offset, host_amp_data + n_colors * offset, stream_points * sizeof(Real) * n_colors, cudaMemcpyHostToDevice));
#else
			HANDLE_ERROR(cudaMemcpy(device_pc_data + 3 * offset, host_pc_data + 3 * offset, stream_points * 3 * sizeof(Real), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(device_amp_data + n_colors * offset, host_amp_data + n_colors * offset, stream_points * sizeof(Real) * n_colors, cudaMemcpyHostToDevice));
#endif
			switch (diff_flag) {
			case PC_DIFF_RS: {
				cudaGenCghPointCloud_NotEncodedRS(gridSize, blockSize, stream_points, device_pc_data + 3 * offset,
					device_amp_data + n_colors * offset, device_dst, device_dst + pnXY, (GpuConstNERS*)device_config, nAdd);

				// 20200824_mwnam_
				cudaError error = cudaGetLastError();
				if (error != cudaSuccess) {
					LOG("cudaGetLastError(): %s\n", cudaGetErrorName(error));
					if (error == cudaErrorLaunchOutOfResources) {
						i = 0;
						blockSize /= 2;
						gridSize *= 2;
						continue;
					}
				}
#ifndef _DEBUG
				HANDLE_ERROR(cudaMemcpyAsync(host_dst, device_dst, bufferSize * 2, cudaMemcpyDeviceToHost));
				HANDLE_ERROR(cudaMemsetAsync(device_dst, 0., bufferSize * 2));
#else
				HANDLE_ERROR(cudaMemcpy(host_dst, device_dst, bufferSize * 2, cudaMemcpyDeviceToHost));
				HANDLE_ERROR(cudaMemset(device_dst, 0., bufferSize * 2));
#endif
				for (ulonglong n = 0; n < pnXY; ++n) {
					complex_H[ch][n][_RE] += host_dst[n];
					complex_H[ch][n][_IM] += host_dst[n + pnXY];
				}
				break;
			}
			case PC_DIFF_FRESNEL: {
				cudaGenCghPointCloud_NotEncodedFrsn(gridSize, blockSize, stream_points, device_pc_data + 3 * offset,
					device_amp_data + n_colors * offset, device_dst, device_dst + pnXY, (GpuConstNEFR*)device_config, nAdd);
#ifndef _DEBUG
				HANDLE_ERROR(cudaMemcpyAsync(host_dst, device_dst, bufferSize * 2, cudaMemcpyDeviceToHost));
				HANDLE_ERROR(cudaMemsetAsync(device_dst, 0., bufferSize * 2));
#else
				HANDLE_ERROR(cudaMemcpy(host_dst, device_dst, bufferSize * 2, cudaMemcpyDeviceToHost));
				HANDLE_ERROR(cudaMemset(device_dst, 0., bufferSize * 2));
#endif
				// 20200824_mwnam_
				cudaError error = cudaGetLastError();
				if (error != cudaSuccess) {
					LOG("cudaGetLastError(): %s\n", cudaGetErrorName(error));
					if (error == cudaErrorLaunchOutOfResources) {
						i--;
						blockSize /= 2;
						gridSize *= 2;
						continue;
					}
				}
				for (ulonglong n = 0; n < pnXY; ++n) {
					complex_H[ch][n][_RE] += host_dst[n];
					complex_H[ch][n][_IM] += host_dst[n + pnXY];
				}
				break;
			} // case
			} // switch


			m_nProgress = (int)((Real)(ch*n_streams + i + 1) * 100 / ((Real)n_streams * nChannel));
			LOG("GPU(%d/%d) > %.16f / %.16f\n", i+1, n_streams,
				complex_H[ch][0][_RE], complex_H[ch][0][_IM]);

		} // for

		//free memory
		HANDLE_ERROR(cudaFree(device_pc_data));
		HANDLE_ERROR(cudaFree(device_amp_data));
		HANDLE_ERROR(cudaFree(device_dst));
		HANDLE_ERROR(cudaFree(device_config));
		
		delete host_config;
	}

	delete[] host_dst;
	if (is_ViewingWindow) {
		delete[] host_pc_data;
	}

	auto end = CUR_TIME;
	Real elapsed_time = ((chrono::duration<Real>)(end - begin)).count();
	LOG("\n%s : %lf(s) \n\n",
		__FUNCTION__,
		elapsed_time);
	
	return elapsed_time;
}
#endif