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
#include "ophTriMesh_GPU.h"

#include <sys.h> //for LOG() macro

void ophTri::initDev()
{
	const int nx = context_.pixel_number.v[0];
	const int ny = context_.pixel_number.v[1];
	const int N = nx * ny;


	if (k_input_d)	cudaFree(k_input_d);
	if (k_output_d)	cudaFree(k_output_d);

	HANDLE_ERROR(cudaMalloc((void**)&k_input_d, sizeof(cufftDoubleComplex)*N));
	HANDLE_ERROR(cudaMalloc((void**)&k_output_d, sizeof(cufftDoubleComplex)*N));

	if (k_temp_d)	cudaFree(k_temp_d);

	HANDLE_ERROR(cudaMalloc((void**)&k_temp_d, sizeof(cufftDoubleComplex)*N));


}

bool ophTri::PreProcessingforVertex(int kth, double& t_Coff00, double& t_Coff01, double& t_Coff02, double& t_Coff10, double& t_Coff11, double& t_Coff12,
	double& detAff, double& R_31, double& R_32, double& R_33, double& T1, double& T2, double& T3)
{
	vec3 t_v1, t_v2, t_v3;
	t_v1 = cghObjPosition_[kth];
	t_v2 = cghObjPosition_[kth + 1];
	t_v3 = cghObjPosition_[kth + 2];

	T1 = t_v1.v[0];
	T2 = t_v1.v[1];
	T3 = t_v1.v[2];

	// Normal vector of triangular mesh plane
	matrix3x3 t_vertex_, t_vertex;
	t_vertex.set_col(0, t_v1);
	t_vertex.set_col(1, t_v2);
	t_vertex.set_col(2, t_v3);
	t_vertex_ = t_vertex.get_transpose();

	vec3 t_norm;
	if (t_v1.v[2] == t_v2.v[2] && t_v2.v[2] == t_v3.v[2] && t_v3.v[2] == t_v1.v[2])
		t_norm = vec3(0.0, 0.0, 1.0);
	else {
		if (apx_equal(fabs(t_vertex_.determinant()), 0.0, hologram_param_.precision_tolerance))
			return false;
		t_norm = t_vertex_.inverse() * vec3(1.0, 1.0, 1.0);
	}


	if (t_norm.v[2] < 0.0)
		t_norm = t_norm * (-1);


	// Estimate R(rotation) and T(translation) from local to global coordinate system
	double temp_x = t_norm.v[0];
	double temp_y = t_norm.v[1];
	double temp_z = t_norm.v[2];
	double temp_rxy = sqrt(temp_x* temp_x + temp_y * temp_y);
	double temp_r = sqrt(temp_x * temp_x + temp_y * temp_y + temp_z * temp_z);

	matrix3x3 R;
	if (temp_rxy != 0)
	{
		double CP = temp_x / temp_rxy;
		double SP = temp_y / temp_rxy;
		double CT = temp_z / temp_r;
		double ST = temp_rxy / temp_r;
		R.set(CP*CT, -SP, CP*ST, SP*CT, CP, SP*ST, -ST, 0, CT);
	}
	else {
		R.set(1, 0, 0, 0, 1, 0, 0, 0, 1);
	}

	matrix3x3 R_ = R.get_transpose();
	t_vertex.set_col(0, t_v1 - t_v1);
	t_vertex.set_col(1, t_v2 - t_v1);
	t_vertex.set_col(2, t_v3 - t_v1);

	matrix3x3 t_vertex_l = R_ * t_vertex;

	matrix t_vertex_lXY(2, 3);
	t_vertex_lXY(0, 0) = t_vertex_l.a00;
	t_vertex_lXY(0, 1) = t_vertex_l.a01;
	t_vertex_lXY(0, 2) = t_vertex_l.a02;
	t_vertex_lXY(1, 0) = t_vertex_l.a10;
	t_vertex_lXY(1, 1) = t_vertex_l.a11;
	t_vertex_lXY(1, 2) = t_vertex_l.a12;

	//Estimate affine transform from local to reference coordinate system
	t_vertex_lXY = t_vertex_lXY.transpose();
	graphics::vector<real> val2;
	val2.add(1.0); val2.add(1.0); val2.add(1.0);
	t_vertex_lXY.add_col(val2);
	matrix t_vertex_lXY2 = t_vertex_lXY.inverse();
	if (t_vertex_lXY2.rows() == 0 || t_vertex_lXY2.cols() == 0)
		return false;

	matrix tmpA(3, 2);
	tmpA(0, 0) = 0.0;
	tmpA(0, 1) = 0.0;
	tmpA(1, 0) = 1.0;
	tmpA(1, 1) = 0.0;
	tmpA(2, 1) = 1.0;
	tmpA(2, 2) = 1.0;

	matrix est_affine = t_vertex_lXY2 ^ tmpA;
	matrix Aff(2, 2);
	Aff(0, 0) = est_affine[0];
	Aff(0, 1) = est_affine[1];
	Aff(1, 0) = est_affine[2];
	Aff(1, 1) = est_affine[3];
	Aff = Aff.transpose();
	vec2 bff = vec2(est_affine[4], est_affine[5]);

	matrix A_T_ = Aff.inverse().transpose();

	matrix R_tmp(2, 3);
	R_tmp(0, 0) = R_.a00;
	R_tmp(0, 1) = R_.a01;
	R_tmp(0, 2) = R_.a02;
	R_tmp(1, 0) = R_.a10;
	R_tmp(1, 1) = R_.a11;
	R_tmp(1, 2) = R_.a12;

	matrix t_CoffMat = A_T_ ^ R_tmp;

	t_Coff00 = t_CoffMat[0];
	t_Coff01 = t_CoffMat[1];
	t_Coff02 = t_CoffMat[2];
	t_Coff10 = t_CoffMat[3];
	t_Coff11 = t_CoffMat[4];
	t_Coff12 = t_CoffMat[5];

	detAff = Aff.determ();

	R_31 = R_(2, 0);
	R_32 = R_(2, 1);
	R_33 = R_(2, 2);

	return true;

}

void ophTri::genCghTriMeshGPU()
{
	initDev();
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if (!stream_)
		cudaStreamCreate(&stream_);

	cudaEventRecord(start, stream_);

	//---------------------------------------------------------------------
	// copy memory from host to device
	//---------------------------------------------------------------------
	const int nx = context_.pixel_number.v[0];
	const int ny = context_.pixel_number.v[1];
	const int N = nx * ny;

	HANDLE_ERROR(cudaMemsetAsync(save_a_d_, 0, sizeof(double)*N, stream_));
	HANDLE_ERROR(cudaMemsetAsync(save_b_d_, 0, sizeof(double)*N, stream_));
	HANDLE_ERROR(cudaMemsetAsync(intensities_d_, 0, sizeof(double)*num_vertex_, stream_));

	ulonglong num_vertex_ = meshData->n_faces;
	Real* tmp = (double*)malloc(sizeof(Real) * num_vertex_);
	for (int i = 0; i < cghObjIntensity_.size(); i++) {
		tmp[i] = cghObjIntensity_[i];
	}

	HANDLE_ERROR(cudaMemcpyAsync(intensities_d_, tmp, sizeof(Real)*num_vertex_, cudaMemcpyHostToDevice), stream_);

	unsigned int nblocks;
	LOG("object number %d\n", num_vertex_);

	double lambda = *context_.wave_length;
	double k = 2.0 * M_PI * lambda;
	ivec2 pn = context_.pixel_number;
	vec2 pp = context_.pixel_pitch;
	vec2 ss = vec2(pn.v[0] * pp.v[0], pn.v[1] * pp.v[1]);

	double cw_amp = context_.cw_amplitude;
	vec3 cw_dir = context_.cw_direction;

	cw_dir.unit();
	vec3 f_c = cw_dir / lambda;
	double del_fxx = 1 / ss.v[0];
	double del_fyy = 1 / ss.v[1];
	double f_cx = f_c.v[0];
	double f_cy = f_c.v[1];
	double f_cz = f_c.v[2];

	double t_Coff00, t_Coff01, t_Coff02, t_Coff10, t_Coff11, t_Coff12;
	double detAff;
	double a_v1, a_v2, a_v3;
	double R_31, R_32, R_33;
	double T1, T2, T3;

	int err = 0;
	for (int k = 0; k < num_vertex_; k += 3)
	{
		if (!PreProcessingforVertex(k, t_Coff00, t_Coff01, t_Coff02, t_Coff10, t_Coff11, t_Coff12, detAff, R_31, R_32, R_33, T1, T2, T3))
		{
			err++;
			continue;
		}

		HANDLE_ERROR(cudaMemsetAsync(k_temp_d, 0, sizeof(double)*N, stream_));

		cudaPolygonKernel(stream_, N, save_a_d_, save_b_d_, intensities_d_, k_temp_d, k, nx, ny, pp[0], pp[1], ss[0], ss[1], lambda, M_PI,
			context_.precision_tolerance, del_fxx, del_fyy, f_cx, f_cy, f_cz, hologram_param_.is_multiple_carrier_wave, cw_amp,
			t_Coff00, t_Coff01, t_Coff02, t_Coff10, t_Coff11, t_Coff12, detAff, R_31, R_32, R_33, T1, T2, T3);

		if (hologram_param_.is_multiple_carrier_wave == 1)
		{
			for (int m = 0; m < hologram_param_.cw_displacement.size(); m++)
			{
				vec2 disp = hologram_param_.cw_displacement[m];
				int disp_x = int(round(disp[0] / del_fxx));
				int disp_y = int(round(disp[1] / del_fyy));
				double cw_amp = hologram_param_.cw_multiamp[m];

				cudaTranslationMatrixKernel(stream_, N, k_temp_d, save_a_d_, save_b_d_, nx, ny, pp[0], pp[1], ss[0], ss[1], lambda,
					disp_x, disp_y, cw_amp, R_31, R_32, R_33);

			}

		}

	}

	LOG("ERR : %d \n", err);

	//HANDLE_ERROR(cudaMemcpyAsync(cghSpectrumReal_, save_a_d_, N * sizeof(double), cudaMemcpyDeviceToHost, stream_));
	//writeIntensity_gray8_bmp("test", nx, ny, cghSpectrumReal_);


	cudaEventRecord(stop, stream_);
	cudaEventSynchronize(stop);

	float elapsedTime = 0.0f;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	LOG("Time= %f ms. \n", elapsedTime);

}