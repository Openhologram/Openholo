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

#include "ophSig.h"
#include "ophSig_GPU.h"



static void HandleError(cudaError_t err,
	const char* file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



void ophSig::cvtOffaxis_GPU(Real angleX, Real angleY) {
	int nx = context_.pixel_number[_X];
	int ny = context_.pixel_number[_Y];
	Real wl = *context_.wave_length;
	Complex<Real> *host_data;
	Complex<Real> *temp_data = new Complex<Real>[nx*ny];
	
	cField2Buffer(*ComplexH, &host_data,nx,ny);

	cudaStreamCreate(&streamLF);

	Complex<Real> *src_data;
	Real *device_angle = new Real[2];
	Real *temp_angle = new Real[2];
	temp_angle[0] = angleX;
	temp_angle[1] = angleY;
	
	//
	Real *dst_data;
	Complex<Real> *F;
	ophSigConfig *device_config = nullptr;
	//Malloc 
	
	cudaMalloc(&src_data, sizeof(Complex<Real>)*nx*ny);
	cudaMalloc(&dst_data, sizeof(Real)*nx*ny);
	cudaMalloc(&F, sizeof(Complex<Real>)*ny*nx);
	cudaMalloc(&device_config, sizeof(ophSigConfig));
	cudaMalloc(&device_angle, sizeof(Real) * 2);

	//memcpy
	cudaMemcpy(src_data, host_data, sizeof(Complex<Real>)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(dst_data, 0, sizeof(Real)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(F, 0, sizeof(Complex<Real>)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(device_config, &_cfgSig, sizeof(ophSigConfig), cudaMemcpyHostToDevice);
	cudaMemcpy(device_angle, temp_angle, sizeof(Real)*2, cudaMemcpyHostToDevice);
	
	//start
	
	cudaCvtOFF(src_data, dst_data, device_config, nx, ny,wl, F, device_angle);
	//end
	cudaMemcpy(temp_data, src_data, nx*ny * sizeof(Complex<Real>), cudaMemcpyDeviceToHost);
	ivec2 size(nx, ny);
	Buffer2Field(temp_data, *ComplexH, size);

	cudaFree(src_data);
	cudaFree(dst_data);
	cudaFree(F);
	cudaFree(device_config);
	cudaFree(device_angle);

	delete[] temp_data;
	delete[] host_data;
}

bool ophSig::sigConvertHPO_GPU(Real depth, Real_t redRate) {

	
	int nx = context_.pixel_number[_X];
	int ny = context_.pixel_number[_Y];
	Complex<Real> *host_data,*temp_data,*F;
	cufftDoubleComplex *fft_temp_data,*out_data;
	cufftHandle fftplan;
	ophSigConfig *device_config = nullptr;
	if (cufftPlan2d(&fftplan, nx, ny, CUFFT_Z2Z) != CUFFT_SUCCESS)
	{
		LOG("FAIL in creating cufft plan");
		return false;
	};
	if (!streamLF)
		cudaStreamCreate(&streamLF);

	cField2Buffer(*ComplexH, &host_data, nx, ny);
	
	cudaMalloc(&temp_data, sizeof(Complex<Real>)*nx*ny);
	cudaMalloc(&fft_temp_data, sizeof(cufftDoubleComplex)*nx*ny);
	cudaMalloc(&out_data, sizeof(cufftDoubleComplex)*nx*ny);
	cudaMalloc(&F, sizeof(Complex<Real>)*ny*nx);
	cudaMalloc(&device_config, sizeof(ophSigConfig));

	cudaMemcpy(temp_data, host_data, sizeof(Complex<Real>)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(fft_temp_data, 0, sizeof(cufftDoubleComplex)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(out_data, 0, sizeof(cufftDoubleComplex)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(F, 0, sizeof(Complex<Real>)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(device_config, &_cfgSig, sizeof(ophSigConfig), cudaMemcpyHostToDevice);

	cudaCvtFieldToCuFFT(temp_data, fft_temp_data, nx, ny);
	cudaCuFFT(&fftplan, fft_temp_data, out_data, nx, ny, CUFFT_FORWARD);

	// 데이터 계산
	Real wl = *context_.wave_length;
	Real NA = _cfgSig.NA;
	Real_t NA_g = NA * redRate;
	Real Rephase = -(1 / (4 * M_PI)*pow((wl / NA_g), 2));
	Real Imphase = ((1 / (4 * M_PI))*depth*wl);

	cudaCvtHPO(streamLF,out_data,fft_temp_data,device_config,F,nx, ny,Rephase,Imphase);

	cudaCuIFFT(&fftplan, fft_temp_data, out_data, nx, ny, CUFFT_INVERSE);
	

	cudaCvtCuFFTToField(fft_temp_data, temp_data, nx, ny);
	cudaMemcpy(host_data, temp_data, nx*ny * sizeof(Complex<Real>), cudaMemcpyDeviceToHost);
	ivec2 size(nx, ny);
	Buffer2Field(host_data, *ComplexH, size);
	//
	cudaFree(F);
	cudaFree(device_config);
	cudaFree(temp_data);
	cudaFree(out_data);
	cudaFree(fft_temp_data);
	cufftDestroy(fftplan);

	delete[] host_data;

	
	return true;
}




bool ophSig::sigConvertCAC_GPU(double red, double green, double blue) {
	int nx = context_.pixel_number[_X];
	int ny = context_.pixel_number[_Y];

	Complex<Real> *host_data, *temp_data, *F;
	cufftDoubleComplex  *fft_temp_data, *out_data;
	cufftHandle fftplan;
	ophSigConfig *device_config = nullptr;
	Real radius = _radius;

	if (cufftPlan2d(&fftplan, nx, ny, CUFFT_Z2Z) != CUFFT_SUCCESS)
	{
		LOG("FAIL in creating cufft plan");
		return false;
	};
	
	ColorField2Buffer(ComplexH[0], &host_data, nx, ny);
	cudaMalloc(&temp_data, sizeof(Complex<Real>)*nx*ny);
	cudaMalloc(&fft_temp_data, sizeof(cufftDoubleComplex)*nx*ny);
	cudaMalloc(&out_data, sizeof(cufftDoubleComplex)*nx*ny);
	cudaMalloc(&F, sizeof(Complex<Real>)*ny*nx);
	cudaMalloc(&device_config, sizeof(ophSigConfig));

	cudaMemcpy(F, 0, sizeof(Complex<Real>)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(device_config, &_cfgSig, sizeof(ophSigConfig), cudaMemcpyHostToDevice);
	cudaMemcpy(fft_temp_data, 0, sizeof(cufftDoubleComplex)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(out_data, 0, sizeof(cufftDoubleComplex)*nx*ny, cudaMemcpyHostToDevice);

	//blue

	cudaMemcpy(temp_data, host_data, sizeof(Complex<Real>)*nx*ny, cudaMemcpyHostToDevice);
	cudaCvtFieldToCuFFT(temp_data, out_data, nx, ny);
	cudaCuFFT(&fftplan,  out_data, fft_temp_data, nx, ny, CUFFT_FORWARD);
	
	double sigmaf = ((_foc[2] - _foc[0]) * blue) / (4 * M_PI);
	cudaCvtCAC(fft_temp_data, out_data,F,device_config,nx, ny,sigmaf,radius);

	cudaCuIFFT(&fftplan, out_data, fft_temp_data, nx, ny, CUFFT_INVERSE);

	cudaCvtCuFFTToField(out_data, temp_data, nx, ny);
	cudaMemcpy(host_data, temp_data, nx*ny * sizeof(Complex<Real>), cudaMemcpyDeviceToHost);
	ivec2 size(nx, ny);
	Buffer2Field(host_data, ComplexH[0], size);

	// green
	ColorField2Buffer(ComplexH[1], &host_data, nx, ny);
	cudaMemcpy(temp_data, host_data, sizeof(Complex<Real>)*nx*ny, cudaMemcpyHostToDevice);
	cudaCvtFieldToCuFFT(temp_data, out_data, nx, ny);
	cudaCuFFT(&fftplan,  out_data, fft_temp_data, nx, ny, CUFFT_FORWARD);

	sigmaf = ((_foc[2] - _foc[1]) * green) / (4 * M_PI);
	cudaCvtCAC(fft_temp_data, out_data, F, device_config, nx, ny, sigmaf, radius);

	cudaCuIFFT(&fftplan, out_data, fft_temp_data, nx, ny, CUFFT_INVERSE);

	cudaCvtCuFFTToField(out_data, temp_data, nx, ny);
	cudaMemcpy(host_data, temp_data, nx*ny * sizeof(Complex<Real>), cudaMemcpyDeviceToHost);
	Buffer2Field(host_data, ComplexH[1], size);

	//free
	cudaFree(F);
	cudaFree(device_config);
	cudaFree(temp_data);
	cudaFree(out_data);
	cudaFree(fft_temp_data);
	cufftDestroy(fftplan);

	delete[] host_data;

	return true;
}

bool ophSig::propagationHolo_GPU(float depth) {
	int nx = context_.pixel_number[_X];
	int ny = context_.pixel_number[_Y];
	Complex<Real> *host_data, *temp_data, *F;
	cufftDoubleComplex *fft_temp_data, *out_data;
	cufftHandle fftplan;

	ophSigConfig *device_config = nullptr;

	if (cufftPlan2d(&fftplan, nx, ny, CUFFT_Z2Z) != CUFFT_SUCCESS)
	{
		LOG("FAIL in creating cufft plan");
		return false;
	};

	cField2Buffer(*ComplexH, &host_data, nx, ny);

	cudaMalloc(&temp_data, sizeof(Complex<Real>)*nx*ny);
	cudaMalloc(&fft_temp_data, sizeof(cufftDoubleComplex)*nx*ny);
	cudaMalloc(&out_data, sizeof(cufftDoubleComplex)*nx*ny);
	cudaMalloc(&F, sizeof(Complex<Real>)*ny*nx);
	cudaMalloc(&device_config, sizeof(ophSigConfig));

	cudaMemcpy(temp_data, host_data, sizeof(Complex<Real>)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(fft_temp_data, 0, sizeof(cufftDoubleComplex)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(out_data, 0, sizeof(cufftDoubleComplex)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(F, 0, sizeof(cufftDoubleComplex)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(device_config, &_cfgSig, sizeof(ophSigConfig), cudaMemcpyHostToDevice);

	cudaCvtFieldToCuFFT(temp_data, fft_temp_data, nx, ny);
	cudaCuFFT(&fftplan, fft_temp_data, out_data, nx, ny, CUFFT_FORWARD);
	
	Real wl = *context_.wave_length;
	Real_t sigmaf = (depth*wl) / (4 * M_PI);

	cudaPropagation(out_data, fft_temp_data, F, device_config,  nx, ny, sigmaf);

	cudaCuIFFT(&fftplan, fft_temp_data, out_data, nx, ny, CUFFT_INVERSE);
	

	cudaCvtCuFFTToField(fft_temp_data, temp_data, nx, ny);
	cudaMemcpy(host_data, temp_data, nx*ny * sizeof(Complex<Real>), cudaMemcpyDeviceToHost);
	ivec2 size(nx, ny);
	Buffer2Field(host_data, *ComplexH, size);
	//
	cudaFree(F);
	cudaFree(device_config);
	cudaFree(temp_data);
	cudaFree(out_data);
	cudaFree(fft_temp_data);
	cufftDestroy(fftplan);

	delete[] host_data;

	return true;
}

double ophSig::sigGetParamSF_GPU(float zMax, float zMin, int sampN, float th) {
	int nx = context_.pixel_number[_X];
	int ny = context_.pixel_number[_Y];

	Complex<Real> *host_data, *temp_data, *FH;
	cufftDoubleComplex *fft_temp_data, *out_data, *Ftemp_data;
	cufftHandle fftplan;
	cufftResult a;
	Real wl = *context_.wave_length;
	Real depth;
	Real *f;
	ophSigConfig *device_config = nullptr;

	cudaMalloc(&f, sizeof(Real)*nx*ny);
	cudaMalloc(&FH, sizeof(Complex<Real>)*ny*nx);
	cudaMalloc(&device_config, sizeof(ophSigConfig));
	cudaMalloc(&temp_data, sizeof(Complex<Real>)*nx*ny);
	cudaMalloc(&fft_temp_data, sizeof(cufftDoubleComplex)*nx*ny);
	cudaMalloc(&Ftemp_data, sizeof(cufftDoubleComplex)*nx*ny);
	cudaMalloc(&out_data, sizeof(cufftDoubleComplex)*nx*ny);

	if (cufftPlan2d(&fftplan, nx, ny, CUFFT_Z2Z) != CUFFT_SUCCESS)
	{
		LOG("FAIL in creating cufft plan");
		return false;
	};
		
	cField2Buffer(*ComplexH, &host_data, nx, ny);

	cudaMemcpy(temp_data, host_data, sizeof(Complex<Real>)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(fft_temp_data, 0, sizeof(cufftDoubleComplex)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(out_data, 0, sizeof(cufftDoubleComplex)*nx*ny, cudaMemcpyHostToDevice);


	cudaCvtFieldToCuFFT(temp_data, fft_temp_data, nx, ny);
	cudaCuFFT(&fftplan, fft_temp_data, out_data, nx, ny, CUFFT_FORWARD);

	cudaMemcpy(FH, 0, sizeof(Complex<Real>)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(f, 0, sizeof(Complex<Real>)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(device_config, &_cfgSig, sizeof(ophSigConfig), cudaMemcpyHostToDevice);
	
	depth = cudaGetParamSF(&fftplan, out_data, Ftemp_data, fft_temp_data, f, FH, device_config, nx, ny, zMax, zMin, sampN, th, wl);

	
	/*cudaCvtCuFFTToField(out_data, temp_data, nx, ny);
	cudaMemcpy(host_data, temp_data, nx*ny * sizeof(Complex<Real>), cudaMemcpyDeviceToHost);
	ivec2 size(nx, ny);*/
	//Buffer2Field(host_data, *ComplexH, size);
	//
	cudaFree(FH);
	cudaFree(device_config);
	cudaFree(temp_data);
	cudaFree(Ftemp_data);
	cudaFree(out_data);
	cudaFree(fft_temp_data);
	cudaFree(f);
	cufftDestroy(fftplan);

	delete[] host_data;

	return depth;
}

double ophSig::sigGetParamAT_GPU() {

	Real index;
	int nx = context_.pixel_number[_X];
	int ny = context_.pixel_number[_Y];
	int tid = 0;
	ivec2 size(nx, ny);
	Real_t NA_g = (Real_t)0.025;
	Real wl = *context_.wave_length;
	Real max = 0;
	Complex<Real> *host_data, *Flr, *Fli, *G, *temp_data;
	cufftDoubleComplex *fft_data, *out_data;

	OphComplexField Fo_temp(nx, ny);
	OphComplexField Fon, yn, Ab_yn;
	OphRealField Ab_yn_half;
	ophSigConfig *device_config = nullptr;
	cufftHandle fftplan;
	vector<Real> t, tn;

	cufftResult a;
	//a = cufftPlan2d(&fftplan, nx, ny, CUFFT_Z2Z);
	//cout << a << endl;
	if (cufftPlan2d(&fftplan, nx, ny, CUFFT_Z2Z) != CUFFT_SUCCESS)
	{
		LOG("FAIL in creating cufft plan");
		return 0;
	};

	cField2Buffer(*ComplexH, &host_data, nx, ny);

	cudaMalloc(&temp_data, sizeof(Complex<Real>)*nx*ny);
	cudaMalloc(&Flr, sizeof(Complex<Real>)*nx*ny);
	cudaMalloc(&Fli, sizeof(Complex<Real>)*nx*ny);
	cudaMalloc(&G, sizeof(Complex<Real>)*nx*ny);
	cudaMalloc(&device_config, sizeof(ophSigConfig));

	cudaMalloc(&fft_data, sizeof(cufftDoubleComplex)*nx*ny);
	cudaMalloc(&out_data, sizeof(cufftDoubleComplex)*nx*ny);


	cudaMemcpy(temp_data, host_data, sizeof(Complex<Real>)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(Flr, 0, sizeof(Complex<Real>)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(Fli, 0, sizeof(Complex<Real>)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(G, 0, sizeof(Complex<Real>)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(device_config, &_cfgSig, sizeof(ophSigConfig), cudaMemcpyHostToDevice);

	cudaMemcpy(fft_data, 0, sizeof(cufftDoubleComplex)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(out_data, 0, sizeof(cufftDoubleComplex)*nx*ny, cudaMemcpyHostToDevice);

	cudaGetParamAT1(temp_data, Flr, Fli, G, device_config, nx, ny, NA_g, wl);

	cudaCvtFieldToCuFFT(Flr, fft_data, nx, ny);
	cudaCuFFT(&fftplan, fft_data, out_data, nx, ny, CUFFT_FORWARD);
	cudaCvtCuFFTToField(out_data, Flr, nx, ny);

	cudaCvtFieldToCuFFT(Fli, fft_data, nx, ny);
	cudaCuFFT(&fftplan, fft_data, out_data, nx, ny, CUFFT_FORWARD);
	cudaCvtCuFFTToField(out_data, Fli, nx, ny);


	cudaGetParamAT2(Flr, Fli, G, temp_data, nx, ny);

	cudaMemcpy(host_data, temp_data, nx*ny * sizeof(Complex<Real>), cudaMemcpyDeviceToHost);
	Buffer2Field(host_data, Fo_temp, size);

	cudaFree(temp_data);
	cudaFree(Flr);
	cudaFree(Fli);
	cudaFree(device_config);
	cudaFree(G);
	cudaFree(out_data);
	cudaFree(fft_data);
	cufftDestroy(fftplan);

	delete[] host_data;

	t = linspace(0., 1., nx / 2 + 1);
	tn.resize(t.size());
	Fon.resize(1, t.size());

	for (int i = 0; i < tn.size(); i++)
	{
		tn.at(i) = pow(t.at(i), 0.5);
		Fon(0, i)._Val[_RE] = Fo_temp(nx / 2 - 1, nx / 2 - 1 + i)._Val[_RE];
		Fon(0, i)._Val[_IM] = 0;
	}

	yn.resize(1, tn.size());
	linInterp(t, Fon, tn, yn);
	fft1(yn, yn);
	Ab_yn.resize(yn.size[_X], yn.size[_Y]);
	absMat(yn, Ab_yn);
	Ab_yn_half.resize(1, nx / 4 + 1);

	for (int i = 0; i < nx / 4 + 1; i++)
	{
		Ab_yn_half(0, i) = Ab_yn(0, nx / 4 + i - 1)._Val[_RE];
		if (i == 0) max = Ab_yn_half(0, 0);
		else
		{
			if (Ab_yn_half(0, i) > max)
			{
				max = Ab_yn_half(0, i);
				index = i;
			}
		}
	}

	index = -(((index + 1) - 120) / 10) / 140 + 0.1;

	return index;


}

