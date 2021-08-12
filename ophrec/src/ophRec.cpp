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

#include "ophRec.h"
#include "sys.h"
#include "function.h"
#include "tinyxml2.h"
#include "ImgControl.h"
#include "fftw3.h"
#include <omp.h>

ophRec::ophRec(void)
	: Openholo()
{
}

ophRec::~ophRec(void)
{
}
#define IMAGE_VAL(x,y,c,w,n,mem) (mem[x*n + y*w*n + c])
vec3 image_sample(float xx, float yy, int c, size_t w, size_t h, double* in);
void circshift(Real* in, Real* out, int shift_x, int shift_y, int nx, int ny);
void circshift(Complex<Real>* in, Complex<Real>* out, int shift_x, int shift_y, int nx, int ny);
void ScaleBilnear(double* src, double* dst, int w, int h, int neww, int newh, double multiplyval = 1.0);
void reArrangeChannel(std::vector<double*>& src, double* dst, int pnx, int pny, int chnum);
void rotateCCW180(double* src, double* dst, int pnx, int pny, double mulival = 1.0);
bool ophRec::readConfig(const char* fname)
{
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
	// about point
	auto ret = xml_doc.LoadFile(fname);
	if (ret != XML_SUCCESS)
	{
		LOG("Failed to load file \"%s\"\n", fname);
		return false;
	}
	xml_node = xml_doc.FirstChild();

	int nWave = 1;
	auto next = xml_node->FirstChildElement("SLM_WaveNum"); // OffsetInDepth
	if (!next || XML_SUCCESS != next->QueryIntText(&nWave))
		return false;

	context_.waveNum = nWave;
	if (context_.wave_length) delete[] context_.wave_length;
	context_.wave_length = new Real[nWave];
	
	char szNodeName[32] = { 0, };
	for (int i = 1; i <= nWave; i++) {
		wsprintfA(szNodeName, "SLM_WaveLength_%d", i);
		next = xml_node->FirstChildElement(szNodeName);
		if (!next || XML_SUCCESS != next->QueryDoubleText(&context_.wave_length[i - 1]))
			return false;
	}
	next = xml_node->FirstChildElement("SLM_PixelNumX");
	if (!next || XML_SUCCESS != next->QueryIntText(&context_.pixel_number[_X]))
		return false;
	next = xml_node->FirstChildElement("SLM_PixelNumY");
	if (!next || XML_SUCCESS != next->QueryIntText(&context_.pixel_number[_Y]))
		return false;
	next = xml_node->FirstChildElement("SLM_PixelPitchX");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&context_.pixel_pitch[_X]))
		return false;
	next = xml_node->FirstChildElement("SLM_PixelPitchY");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&context_.pixel_pitch[_Y]))
		return false;
	next = xml_node->FirstChildElement("EyeLength");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&rec_config.EyeLength))
		return false;
	next = xml_node->FirstChildElement("EyePupilDiameter");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&rec_config.EyePupilDiaMeter))
		return false;
	next = xml_node->FirstChildElement("EyeBoxSizeScaleFactor");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&rec_config.EyeBoxSizeScale))
		return false;
	next = xml_node->FirstChildElement("EyeBoxSizeX");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&rec_config.EyeBoxSize[_X]))
		return false;
	next = xml_node->FirstChildElement("EyeBoxSizeY");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&rec_config.EyeBoxSize[_Y]))
		return false;
	next = xml_node->FirstChildElement("EyeBoxUnit");
	if (!next || XML_SUCCESS != next->QueryIntText(&rec_config.EyeBoxUnit))
		return false;
	next = xml_node->FirstChildElement("EyeCenterX");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&rec_config.EyeCenter[_X]))
		return false;
	next = xml_node->FirstChildElement("EyeCenterY");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&rec_config.EyeCenter[_Y]))
		return false;
	next = xml_node->FirstChildElement("EyeCenterZ");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&rec_config.EyeCenter[_Z]))
		return false;
	next = xml_node->FirstChildElement("EyeFocusDistance");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&rec_config.EyeFocusDistance))
		return false;
	next = xml_node->FirstChildElement("ResultSizeScale");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&rec_config.ResultSizeScale))
		return false;
	next = xml_node->FirstChildElement("SimulationTo");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&rec_config.SimulationTo))
		return false;
	next = xml_node->FirstChildElement("SimulationFrom");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&rec_config.SimulationFrom))
		return false;
	next = xml_node->FirstChildElement("SimulationStep");
	if (!next || XML_SUCCESS != next->QueryIntText(&rec_config.SimulationStep))
		return false;
	next = xml_node->FirstChildElement("SimulationMode");
	if (!next || XML_SUCCESS != next->QueryIntText(&rec_config.SimulationMode))
		return false;
	next = xml_node->FirstChildElement("RatioAtRetina");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&rec_config.RatioAtRetina))
		return false;
	next = xml_node->FirstChildElement("RatioAtPupil");
	if (!next || XML_SUCCESS != next->QueryDoubleText(&rec_config.RatioAtPupil))
		return false;
	next = xml_node->FirstChildElement("CreatePupilFieldImg");
	if (!next || XML_SUCCESS != next->QueryBoolText(&rec_config.CreatePupilFieldImg))
		return false;
	next = xml_node->FirstChildElement("CenteringRetinalImg");
	if (!next || XML_SUCCESS != next->QueryBoolText(&rec_config.CenteringRetinaImg))
		return false;

	rec_config.EyeBoxSize = rec_config.EyeBoxSizeScale * rec_config.EyePupilDiaMeter * rec_config.EyeBoxSize;
	Initialize();
	auto end = CUR_TIME;
	auto during = ((chrono::duration<Real>)(end - start)).count();
	LOG("%lf (s)..done\n", during);

	return true;
}

bool ophRec::readImage(const char* path)
{
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const int N = pnX * pnY;

	int w, h, bytesperpixel;
	bool ret = getImgSize(w, h, bytesperpixel, path);
	uchar* imgload = new uchar[w * h * bytesperpixel];

	ret = loadAsImgUpSideDown(path, imgload);
	if (!ret) {
		LOG("Failed::Image Load: %s\n", path);
		return false;
	}
	LOG("Succeed::Image Load: %s\n", path);

	uchar *tmp = new uchar[N * bytesperpixel];
	memset(tmp, 0, sizeof(char) * N);

	if (w != pnX || h != pnY)
		imgScaleBilinear(imgload, tmp, w, h, pnX, pnY);
	else
		memcpy(tmp, imgload, sizeof(char) * N);

	delete[] imgload;

	for (int ch = 0; ch < bytesperpixel; ch++)
	{
		//memset(complex_H[ch], 0.0, sizeof(Complex<Real>) * N);

		for (int i = 0; i < N; i++)
		{
			complex_H[ch][i][_RE] = Real(tmp[i]);
			complex_H[ch][i][_IM] = 0.0;
		}
	}
	delete[] tmp;
	return true;
}

bool ophRec::readImagePNA(const char *phase, const char *amplitude)
{
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const int N = pnX * pnY;

	int w, h, bytesperpixel;
	bool ret = getImgSize(w, h, bytesperpixel, phase);
	uchar* imgload = new uchar[w * h * bytesperpixel];

	ret = loadAsImgUpSideDown(phase, imgload);
	if (!ret) {
		LOG("Failed::Image Load: %s\n", phase);
		return false;
	}
	LOG("Succeed::Image Load: %s\n", phase);
	
	uchar *phaseTmp = new uchar[N * bytesperpixel];
	memset(phaseTmp, 0, sizeof(char) * N);

	if (w != pnX || h != pnY)
		imgScaleBilinear(imgload, phaseTmp, w, h, pnX, pnY);
	else
		memcpy(phaseTmp, imgload, sizeof(char) * N);

	delete[] imgload;


	ret = getImgSize(w, h, bytesperpixel, amplitude);
	imgload = new uchar[w * h * bytesperpixel];

	ret = loadAsImgUpSideDown(amplitude, imgload);
	if (!ret) {
		LOG("Failed::Image Load: %s\n", amplitude);
		return false;
	}
	LOG("Succeed::Image Load: %s\n", amplitude);

	uchar *ampTmp = new uchar[N * bytesperpixel];
	memset(ampTmp, 0, sizeof(char) * N);

	if (w != pnX || h != pnY)
		imgScaleBilinear(imgload, ampTmp, w, h, pnX, pnY);
	else
		memcpy(ampTmp, imgload, sizeof(char) * N);

	delete[] imgload;
	   
	Real PI2 = M_PI * 2;
	for (int ch = 0; ch < bytesperpixel; ch++)
	{
		//memset(complex_H[ch], 0.0, sizeof(Complex<Real>) * N);

		for (int i = 0; i < N; i++)
		{
			Real p = Real(phaseTmp[i]) * PI2 / 255.0;
			Real a = Real(ampTmp[i]) / 255.0;
			Complex<Real> tmp(0, p);
			tmp.exp();

			complex_H[ch][i] = a * tmp;
		}
	}
	delete[] phaseTmp;
	delete[] ampTmp;
	return true;
}

bool ophRec::readImageRNI(const char *real, const char *imag)
{
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const int N = pnX * pnY;

	int w, h, bytesperpixel;
	bool ret = getImgSize(w, h, bytesperpixel, real);
	uchar* imgload = new uchar[w * h * bytesperpixel];

	ret = loadAsImgUpSideDown(real, imgload);
	if (!ret) {
		LOG("Failed::Image Load: %s\n", real);
		return false;
	}
	LOG("Succeed::Image Load: %s\n", real);

	uchar *realTmp = new uchar[N * bytesperpixel];
	memset(realTmp, 0, sizeof(char) * N);

	if (w != pnX || h != pnY)
		imgScaleBilinear(imgload, realTmp, w, h, pnX, pnY);
	else
		memcpy(realTmp, imgload, sizeof(char) * N);

	delete[] imgload;


	ret = getImgSize(w, h, bytesperpixel, imag);
	imgload = new uchar[w * h * bytesperpixel];

	ret = loadAsImgUpSideDown(imag, imgload);
	if (!ret) {
		LOG("Failed::Image Load: %s\n", imag);
		return false;
	}
	LOG("Succeed::Image Load: %s\n", imag);

	uchar *imagTmp = new uchar[N * bytesperpixel];
	memset(imagTmp, 0, sizeof(char) * N);

	if (w != pnX || h != pnY)
		imgScaleBilinear(imgload, imagTmp, w, h, pnX, pnY);
	else
		memcpy(imagTmp, imgload, sizeof(char) * N);

	delete[] imgload;

	for (int ch = 0; ch < bytesperpixel; ch++)
	{
		//memset(complex_H[ch], 0.0, sizeof(Complex<Real>) * N);

		for (int i = 0; i < N; i++)
		{
			complex_H[ch][i][_RE] = (Real)realTmp[i];
			complex_H[ch][i][_IM] = (Real)imagTmp[i];
		}
	}
	delete[] realTmp;
	delete[] imagTmp;
	return true;
}

void ophRec::GetPupilFieldFromHologram()
{
	LOG("Propagation to observer plane\n");
	
	const int channel = context_.waveNum;
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	field_set_.resize(channel);
	pp_set_.resize(channel);
	fftw_complex *in = nullptr;
	fftw_complex *out = nullptr;
	fftw_plan fftw_plan_fwd = fftw_plan_dft_2d(pnY, pnX, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

	for (int ch = 0; ch < channel; ch++)
	{
		Propagation_Fresnel_FFT(ch);
	}

	fftw_destroy_plan(fftw_plan_fwd);
}

void ophRec::GetPupilFieldFromVWHologram()
{
	LOG("Propagation to observer plane\n");

	const int nChannel = context_.waveNum;
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const int N = pnX * pnY;
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];

	field_set_.resize(nChannel);
	pn_set_.resize(nChannel);
	pp_set_.resize(nChannel);

	fftw_complex *in = nullptr, *out = nullptr;
	fftw_plan fft_plan_fwd = fftw_plan_dft_2d(pnY, pnX, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

	double prop_z = rec_config.EyeCenter[_Z];
	double f_field = rec_config.EyeCenter[_Z];


	Real ssX = pnX * ppX;
	Real ssY = pnY * ppY;


	for (int ctr = 0; ctr < nChannel; ctr++)
	{
		LOG("Color Number: %d\n", ctr + 1);

		Complex<Real>* u = complex_H[ctr];
		field_set_[ctr] = new Complex<Real>[N];
		memset(field_set_[ctr], 0.0, sizeof(Complex<Real>) * N);

		Real lambda = context_.wave_length[ctr];

		Real k = 2 * M_PI / lambda;
		Real kk = k / (prop_z * 2);
		Real kpropz = k * prop_z;
		Real lambdapropz = lambda * prop_z;
		Real ss_res_x = fabs(lambdapropz / ppX);
		Real ss_res_y = fabs(lambdapropz / ppY);
		Real hss_res_x = ss_res_x / 2.0;
		Real hss_res_y = ss_res_y / 2.0;
		Real pp_res_x = ss_res_x / Real(pnX);
		Real pp_res_y = ss_res_y / Real(pnY);
		Real absppX = fabs(lambdapropz / (4 * ppX));
		Real absppY = fabs(lambdapropz / (4 * ppY));

		int loopi;
#pragma omp parallel for private(loopi)	
		for (loopi = 0; loopi < N; loopi++)
		{
			int x = loopi % pnX;
			int y = loopi / pnX;

			double xx_src = (-ssX / 2.0) + (ppX * double(x));
			double yy_src = (ssY / 2.0) - ppY - (ppY * double(y));

			if (f_field != prop_z)
			{
				double effective_f = f_field * prop_z / (f_field - prop_z);
				double sval = (xx_src*xx_src) + (yy_src*yy_src);
				sval *= M_PI / lambda / effective_f;
				Complex<Real> kernel(0, sval);
				kernel.exp();
				//exponent_complex(&kernel);

				double anti_aliasing_mask = fabs(xx_src) < fabs(lambda*effective_f / (4 * ppX)) ? 1 : 0;
				anti_aliasing_mask *= fabs(yy_src) < fabs(lambda*effective_f / (4 * ppY)) ? 1 : 0;

				field_set_[ctr][x + y * pnX] = u[x + y * pnX] * kernel * anti_aliasing_mask;


			}
			else {

				field_set_[ctr][x + y * pnX] = u[x + y * pnX];

			}
		}
		//delete[] fringe_[ctr];
		//free(fringe_[ctr]);

		fftwShift(field_set_[ctr], field_set_[ctr], pnX, pnY, FFTW_FORWARD, false);



#pragma omp parallel for private(loopi)	
		for (loopi = 0; loopi < N; loopi++)
		{
			int x = loopi % pnX;
			int y = loopi / pnX;

			double xx_res = (-ss_res_x / 2.0) + (pp_res_x * double(x));
			double yy_res = (ss_res_y / 2.0) - pp_res_y - (pp_res_y * double(y));

			Complex<Real> tmp1(0, k*prop_z);
			tmp1.exp();
			//exponent_complex(&tmp1);

			Complex<Real> tmp2(0, lambda*prop_z);

			double ssval = (xx_res*xx_res) + (yy_res*yy_res);
			ssval *= k / (2 * prop_z);
			Complex<Real> tmp3(0, ssval);
			//exponent_complex(&tmp3);
			tmp3.exp();

			Complex<Real> coeff = tmp1 / tmp2 * tmp3;

			field_set_[ctr][x + y * pnX] *= coeff;

		}
		pp_set_[ctr][0] = pp_res_x;
		pp_set_[ctr][1] = pp_res_y;

	}

	fftw_destroy_plan(fft_plan_fwd);
}

void ophRec::Propagation_Fresnel_FFT(int chnum)
{
	LOG("Color Number: %d\n", chnum + 1);

	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const int N = pnX * pnY;
	Real ppX = context_.pixel_pitch[_X];
	Real ppY = context_.pixel_pitch[_Y];
	Real ppX4 = ppX * 4;
	Real ppY4 = ppY * 4;
	Real lambda = context_.wave_length[chnum];
	Real ssX = pnX * ppX;
	Real ssY = pnY * ppY;
	Real hssX = ssX / 2.0;
	Real hssY = ssY / 2.0;
	Real prop_z = rec_config.EyeCenter[_Z];
	vec2 cxy = vec2(0);

	Real k = 2 * M_PI / lambda;
	Real kk = k / (prop_z * 2);
	Real kpropz = k * prop_z;
	Real lambdapropz = lambda * prop_z;
	Real ss_res_x = fabs(lambdapropz / ppX);
	Real ss_res_y = fabs(lambdapropz / ppY);
	Real hss_res_x = ss_res_x / 2.0;
	Real hss_res_y = ss_res_y / 2.0;
	Real pp_res_x = ss_res_x / Real(pnX);
	Real pp_res_y = ss_res_y / Real(pnY);

	Complex<Real>* tmp = complex_H[chnum];
	field_set_[chnum] =  new Complex<Real>[N];
	memset(field_set_[chnum], 0.0, sizeof(Complex<Real>) * N);

	//Real xx_src, yy_src, xx_res, yy_res;
	//int x, y;

	Real absppX = fabs(lambdapropz / (4 * ppX));
	Real absppY = fabs(lambdapropz / (4 * ppY));

	int i;
#pragma omp parallel for private(i)	
	for (i = 0; i < N; i++)
	{
		int x = i % pnX;
		int y = i / pnX;

		Real xx_src = -hssX + (ppX * Real(x));
		Real yy_src = hssY - ppY - (ppY * Real(y));
		Real xxx = xx_src - cxy[_X];
		Real yyy = yy_src - cxy[_Y];

		Real sval = xxx * xxx + yyy * yyy;
		sval *= kk;
		Complex<Real> kernel(0, sval);
		kernel.exp();
		//exponent_complex(&kernel);

		Real anti_aliasing_mask = fabs(xxx) < absppX ? 1 : 0;
		anti_aliasing_mask *= fabs(yyy) < absppY ? 1 : 0;

		field_set_[chnum][x + y * pnX] = tmp[x + y * pnX] * kernel * anti_aliasing_mask;

	}
	//free(fringe_[chnum]);

	//fftw_complex *in = nullptr, *out = nullptr;
	fftwShift(field_set_[chnum], field_set_[chnum], pnX, pnY, FFTW_FORWARD, false);

#pragma omp parallel for private(i)	
	for (i = 0; i < N; i++)
	{
		int x = i % pnX;
		int y = i / pnX;

		Real xx_res = -hss_res_x + (pp_res_x * Real(x));
		Real yy_res = hss_res_y - pp_res_y - (pp_res_y * Real(y));
		Real xxx = xx_res - cxy[_X];
		Real yyy = yy_res - cxy[_Y];

		Complex<Real> tmp1(0, kpropz);
		tmp1.exp();
		//exponent_complex(&tmp1);

		Complex<Real> tmp2(0, lambdapropz);

		Real ssval = xxx * xxx + yyy * yyy;
		ssval *= kk;
		Complex<Real> tmp3(0, ssval);
		tmp3.exp();
		//exponent_complex(&tmp3);

		Complex<Real> coeff = tmp1 / tmp2 * tmp3;

		field_set_[chnum][x + y * pnX] *= coeff;

	}

	//pn_set_[chnum] = PIXEL_NUMBER;
	pp_set_[chnum][_X] = pp_res_x;
	pp_set_[chnum][_Y] = pp_res_y;
}

void ophRec::Perform_Simulation()
{
	LOG("Simulation start\n");
	FILE *fp;
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const Real ppX = context_.pixel_pitch[_X];
	const Real ppY = context_.pixel_pitch[_Y];
	const int nChannel = context_.waveNum;
	const int simStep = rec_config.SimulationStep;
	const Real simFrom = rec_config.SimulationFrom;
	const Real simTo = rec_config.SimulationTo;
	const int simMode = rec_config.SimulationMode;
	vec2 boxSize = rec_config.EyeBoxSize;
	vec3 eyeCenter = rec_config.EyeCenter;
	Real eyeLen = rec_config.EyeLength;
	Real eyeFocusDistance = rec_config.EyeFocusDistance;
	bool bCreatePupilImg = rec_config.CreatePupilFieldImg;
	bool bCenteringRetinaImg = rec_config.CenteringRetinaImg;
	Real resultSizeScale = rec_config.ResultSizeScale;
	Real eyePupil = rec_config.EyePupilDiaMeter;
	bool bSimPos[3];
	Real ratioAtRetina = rec_config.RatioAtRetina;

	memcpy(&bSimPos[0], rec_config.SimulationPos, sizeof(bSimPos));
	//char temp[100];
	//sprintf(temp, "%.2f", SATURATION_RATIO_AT_RETINA);
	//SAT_VAL_PREFIX.clear();
	//SAT_VAL_PREFIX.append(temp);
	//SAT_VAL_PREFIX.replace(SAT_VAL_PREFIX.find('.'), 1, "_");

	vec3 var_step;
	std::vector<vec3> var_vals;
	var_vals.resize(simStep);

	if (simStep > 1)
	{
		var_step = (simTo - simFrom) / (simStep - 1);
		for (int i = 0; i < simStep; i++)
			var_vals[i] = simFrom + var_step * i;

	}
	else
		var_vals[0] = (simTo + simFrom) / 2.0;

	Real lambda, k;
	int pn_e_x, pn_e_y;
	Real pp_e_x, pp_e_y, ss_e_x, ss_e_y;
	int pn_p_x, pn_p_y;
	Real pp_p_x, pp_p_y, ss_p_x, ss_p_y;
	vec2 eye_box_range_x, eye_box_range_y;
	ivec2 eye_box_range_idx, eye_box_range_idy;
	ivec2 eye_shift_by_pn;

	field_ret_set_.resize(nChannel);
	pn_ret_set_.resize(nChannel);
	pp_ret_set_.resize(nChannel);
	ss_ret_set_.resize(nChannel);
	recon_set.resize(simStep * nChannel);
	img_set.resize(simStep * nChannel);
	img_size.resize(simStep * nChannel);
	focus_recon_set.resize(simStep);
	focus_img_set.resize(simStep);
	focus_img_size.resize(simStep);

	std::string varname2;

	for (int vtr = 0; vtr < simStep; vtr++)
	{
		if (simMode == 0) {
			eyeFocusDistance = var_vals[vtr][_X];
		}
		else {
			if (bSimPos[_X]) eyeCenter[_X] = var_vals[vtr][_X];
			if (bSimPos[_Y]) eyeCenter[_Y] = var_vals[vtr][_X];
			if (bSimPos[_Z]) eyeCenter[_Z] = var_vals[vtr][_X];
		}

		for (int ctr = 0; ctr < nChannel; ctr++)
		{
			lambda = context_.wave_length[ctr];
			k = 2 * M_PI / lambda;

			pn_e_x = pnX;
			pn_e_y = pnY;
			pp_e_x = pp_set_[ctr][_X];
			pp_e_y = pp_set_[ctr][_Y];
			ss_e_x = Real(pn_e_x) * pp_e_x;
			ss_e_y = Real(pn_e_y) * pp_e_y;

			eye_shift_by_pn[0] = round(eyeCenter[_X] / pp_e_x);
			eye_shift_by_pn[1] = round(eyeCenter[_Y] / pp_e_y);

			Complex<Real>* hh_e_shift = new Complex<Real>[pn_e_x * pn_e_y];
			memset(hh_e_shift, 0.0, sizeof(Complex<Real>) * pn_e_x * pn_e_y);
			circshift(field_set_[ctr], hh_e_shift, -eye_shift_by_pn[0], eye_shift_by_pn[1], pn_e_x, pn_e_y);


			if (rec_config.EyeBoxUnit == 0)
			{
				eye_box_range_x[0] = -boxSize[_X] / 2.0;
				eye_box_range_x[1] = boxSize[_X] / 2.0;
				eye_box_range_y[0] = -boxSize[_Y] / 2.0;
				eye_box_range_y[1] = boxSize[_Y] / 2.0;
				eye_box_range_idx[0] = floor((eye_box_range_x[0] + ss_e_x / 2.0) / pp_e_x);
				eye_box_range_idx[1] = floor((eye_box_range_x[1] + ss_e_x / 2.0) / pp_e_x);
				eye_box_range_idy[0] = pn_e_y - floor((eye_box_range_y[1] + ss_e_y / 2.0) / pp_e_y);
				eye_box_range_idy[1] = pn_e_y - floor((eye_box_range_y[0] + ss_e_y / 2.0) / pp_e_y);

			}
			else {

				int temp = floor((pn_e_x - boxSize[_X]) / 2.0);
				eye_box_range_idx[0] = temp;
				eye_box_range_idx[1] = temp + boxSize[_X] - 1;
				temp = floor((pn_e_y - boxSize[_Y]) / 2.0);
				eye_box_range_idy[0] = temp;
				eye_box_range_idy[1] = temp + boxSize[_Y] - 1;
			}

			pn_p_x = eye_box_range_idx[1] - eye_box_range_idx[0] + 1;
			pn_p_y = eye_box_range_idy[1] - eye_box_range_idy[0] + 1;
			pp_p_x = pp_e_x;
			pp_p_y = pp_e_y;
			ss_p_x = pn_p_x * pp_p_x;
			ss_p_y = pn_p_y * pp_p_y;

			int N = pn_p_x * pn_p_y;
			int N2 = pn_e_x * pn_e_y;
			Complex<Real>* hh_p = new Complex<Real>[N];
			memset(hh_p, 0.0, sizeof(Complex<Real>) * N);

			int cropidx = 0;

			for (int p = 0; p < N2; p++)
			{
				int x = p % pn_e_x;
				int y = p / pn_e_x;

				if (x >= eye_box_range_idx[0] - 1 && x <= eye_box_range_idx[1] - 1 && y >= eye_box_range_idy[0] - 1 && y <= eye_box_range_idy[1] - 1)
				{
					int xx = cropidx % pn_p_x;
					int yy = cropidx / pn_p_x;
					hh_p[yy*pn_p_x + xx] = hh_e_shift[p];
					cropidx++;
				}
			}
			delete[] hh_e_shift;

			Real f_eye = eyeLen * (eyeCenter[_Z] - eyeFocusDistance) / (eyeLen + (eyeCenter[_Z] - eyeFocusDistance));
			Real effective_f = f_eye * eyeLen / (f_eye - eyeLen);
			
			Complex<Real>* hh_e_ = new Complex<Real>[N];
			memset(hh_e_, 0.0, sizeof(Complex<Real>) * N);

			int loopp;
//#pragma omp parallel for private(loopp)	
			for (loopp = 0; loopp < N; loopp++)
			{
				int x = loopp % pn_p_x;
				int y = loopp / pn_p_x;

				Real XE = -ss_p_x / 2.0 + (pp_p_x *x);
				Real YE = ss_p_y / 2.0 - pp_p_y - (pp_p_y * y);

				Real sval = (XE*XE) + (YE*YE);
				sval *= M_PI / lambda / effective_f;
				Complex<Real> eye_propagation_kernel(0, sval);
				//exponent_complex(&eye_propagation_kernel);
				eye_propagation_kernel.exp();

				Real eye_lens_anti_aliasing_mask = fabs(XE) < fabs(lambda*effective_f / (4 * pp_e_x)) ? 1.0 : 0.0;
				eye_lens_anti_aliasing_mask *= fabs(YE) < fabs(lambda*effective_f / (4 * pp_e_y)) ? 1.0 : 0.0;

				Real eye_pupil_mask = sqrt(XE*XE + YE * YE) < (eyePupil / 2.0) ? 1.0 : 0.0;

				hh_e_[x + y * pn_p_x] = hh_p[x + y * pn_p_x] * eye_propagation_kernel * eye_lens_anti_aliasing_mask * eye_pupil_mask;
				//hh_e_[x + y * pn_p_x] = hh_e_shift[x + y * pn_p_x] * eye_propagation_kernel * eye_lens_anti_aliasing_mask * eye_pupil_mask;

			}



			delete[] hh_p;
			//free(hh_p);

			//fftw_complex *in = nullptr, *out = nullptr;
			//fftw_plan fft_plan_fwd_ = fftw_plan_dft_2d(pn_p_y, pn_p_x, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

			fftwShift(hh_e_, hh_e_, pn_p_x, pn_p_y, FFTW_FORWARD, false);

			if (fp = fopen("d:\\a\\hh_e_.dat", "wb"))
			{
				int size = sizeof(Complex<Real>);
				int cnt = N;
				for (int i = 0; i < cnt; i++)
				{
					fwrite(&hh_e_[i], size, 1, fp);
				}
				fclose(fp);
			}

			//fftw_destroy_plan(fft_plan_fwd_);

			Real pp_ret_x, pp_ret_y;
			int pn_ret_x, pn_ret_y;
			vec2 ret_size_xy;

			pp_ret_x = lambda * eyeLen / ss_p_x;
			pp_ret_y = lambda * eyeLen / ss_p_y;
			pn_ret_x = pn_p_x;
			pn_ret_y = pn_p_y;
			ret_size_xy[0] = pp_ret_x * pn_ret_x;
			ret_size_xy[1] = pp_ret_y * pn_ret_y;

			field_ret_set_[ctr] = new Real[pn_p_x * pn_p_y];
			memset(field_ret_set_[ctr], 0.0, sizeof(Real)*pn_p_x*pn_p_y);

//#pragma omp parallel for private(loopp)	
			for (loopp = 0; loopp < pn_p_x*pn_p_y; loopp++)
			{
				int x = loopp % pn_p_x;
				int y = loopp / pn_p_x;

				Real XR = ret_size_xy[0] / 2.0 + (pp_ret_x * x);
				Real YR = ret_size_xy[1] / 2.0 - pp_ret_y - (pp_ret_y * y);

				Real sval = (XR*XR) + (YR*YR);
				sval *= k / (2 * eyeLen);
				Complex<Real> val1(0, sval);
				//exponent_complex(&val1);
				val1.exp();

				Complex<Real> val2(0, k * eyeLen);
				//exponent_complex(&val2);
				val2.exp();
				Complex<Real> val3(0, lambda * eyeLen);

				field_ret_set_[ctr][x + pn_p_x * y] = (hh_e_[x + pn_p_x * y] * (val1 * val2 / val3)).mag();

			}
			delete[] hh_e_;

			pp_ret_set_[ctr] = vec2(pp_ret_x, pp_ret_y);
			pn_ret_set_[ctr] = ivec2(pn_ret_x, pn_ret_y);
			ss_ret_set_[ctr] = pp_ret_set_[ctr] * pn_ret_set_[ctr];

			if (bCreatePupilImg)
			{
				Real pp_min = (pp_e_x > pp_e_y) ? pp_e_y : pp_e_x;
				Real ss_max = (ss_e_x > ss_e_y) ? ss_e_x : ss_e_y;
				Real pn_tar = ceil(ss_max / pp_min);
				pp_min = ss_max / pn_tar;
				Real pn_e_tar_x = round(ss_e_x / pp_min);
				Real pn_e_tar_y = round(ss_e_y / pp_min);

				Real resize_scale_x = pn_e_tar_x * resultSizeScale;
				Real resize_scale_y = pn_e_tar_y * resultSizeScale;

				int N = resize_scale_x * resize_scale_y;

				recon_set[vtr * nChannel + ctr] = new Real[N];
				img_set[vtr * nChannel + ctr] = new uchar[N];

				//Real* pf_image = new Real[N];
				//uchar* img = new uchar[N];
				memset(recon_set[vtr * nChannel + ctr], 0.0, sizeof(Real) * N);
				memset(img_set[vtr * nChannel + ctr], 0, sizeof(uchar) * N);

				GetPupilFieldImage(field_set_[ctr], recon_set[vtr * nChannel + ctr]
					, pn_e_x, pn_e_y, pp_e_x, pp_e_y, resize_scale_x, resize_scale_y);

				std::string fname = std::string("./").append("").append("/FIELD/");
				fname = fname.append("FIELD_COLOR_").append(std::to_string(ctr + 1)).append("_").append("").append("_SAT_").append("").append("_").append(std::to_string(vtr + 1)).append(".bmp");
				normalize(recon_set[vtr * nChannel + ctr], img_set[vtr * nChannel + ctr], (int)resize_scale_x, (int)resize_scale_y);
				img_size[vtr * nChannel + ctr][_X] = (int)resize_scale_x;
				img_size[vtr * nChannel + ctr][_Y] = (int)resize_scale_y;
				
				//saveAsImg(fname.c_str(), 8, img_set[vtr * nChannel + ctr], (int)resize_scale_x, (int)resize_scale_y);
				//writeImage(fname, pf_image, resize_scale_x, resize_scale_y, 8);

				//delete[]pf_image;
				//delete[] img;
			}

		}	// end of ctr

		Real pnx_max = 0.0, pny_max = 0.0;
		for (int i = 0; i < nChannel; i++)
		{
			pnx_max = (pn_ret_set_[i][0] > pnx_max ? pn_ret_set_[i][0] : pnx_max);
			pny_max = (pn_ret_set_[i][1] > pny_max ? pn_ret_set_[i][1] : pny_max);
		}

		Real retinal_image_shift_x = eyeCenter[_X] * eyeLen / eyeCenter[_Z];
		Real retinal_image_shift_y = eyeCenter[_Y] * eyeLen / eyeCenter[_Z];

		res_set_.resize(nChannel);
		res_set_norm_255_.resize(nChannel);

		int loopi;
//#pragma omp parallel for private(loopi)	
		for (loopi = 0; loopi < nChannel; loopi++)
		{
			Real* hh_ret_ = new Real[pn_ret_set_[loopi][0] * pn_ret_set_[loopi][1]];
			memset(hh_ret_, 0.0, sizeof(Real)*pn_ret_set_[loopi][0] * pn_ret_set_[loopi][1]);

			if (bCenteringRetinaImg)
			{
				Real retinal_image_shift_by_pn_x = round(retinal_image_shift_x / pp_ret_set_[loopi][0]);
				Real retinal_image_shift_by_pn_y = round(retinal_image_shift_y / pp_ret_set_[loopi][1]);

				circshift(field_ret_set_[loopi], hh_ret_, -retinal_image_shift_by_pn_x, retinal_image_shift_by_pn_y, pn_ret_set_[loopi][0], pn_ret_set_[loopi][1]);

			}
			else
				hh_ret_ = field_ret_set_[loopi];

			delete[] field_ret_set_[loopi];

			res_set_[loopi] = new Real[pnx_max * pny_max];
			memset(res_set_[loopi], 0.0, sizeof(Real)*pnx_max*pny_max);

			//ScaleBilnear(hh_ret_, res_set_[i], pn_ret_set_[i][0], pn_ret_set_[i][1], pnx_max, pny_max);
			//writeImage("test3.bmp", res_set_[i], pnx_max, pny_max, 8);
	
			ScaleBilnear(hh_ret_, res_set_[loopi], pn_ret_set_[loopi][0], pn_ret_set_[loopi][1], pnx_max, pny_max, lambda*lambda);

		}

		if (fp = fopen("d:\\a\\res_set_.dat", "wb"))
		{
			int size = sizeof(Real);
			int cnt = pnx_max * pny_max;
			for (int i = 0; i < cnt; i++)
			{
				fwrite(&res_set_[0][i], size, 1, fp);
			}
			fclose(fp);
		}


		Real maxvalue = res_set_[0][0];
		Real minvalue = res_set_[0][0];
		for (int i = 0; i < nChannel; i++)
		{
			for (int j = 0; j < pnx_max*pny_max; j++)
			{
				maxvalue = res_set_[i][j] > maxvalue ? res_set_[i][j] : maxvalue;
				minvalue = res_set_[i][j] < minvalue ? res_set_[i][j] : minvalue;
			}
		}

		for (int j = 0; j < pnx_max*pny_max; j++)
		{
			res_set_[0][j] = (res_set_[0][j] - minvalue) / (maxvalue - minvalue) * 255;
		}

//#pragma omp parallel for private(loopi)	
		for (loopi = 0; loopi < nChannel; loopi++)
		{
			for (int j = 0; j < pnx_max*pny_max; j++)
			{
				if (res_set_[loopi][j] > ratioAtRetina * maxvalue)
					res_set_[loopi][j] = ratioAtRetina * maxvalue;

				res_set_[loopi][j] = (res_set_[loopi][j] - minvalue) / (ratioAtRetina*maxvalue - minvalue);

			}
		}

		Real ret_size_x = pnx_max * resultSizeScale;
		Real ret_size_y = pny_max * resultSizeScale;

//#pragma omp parallel for private(loopi)			
		for (loopi = 0; loopi < nChannel; loopi++)
		{
			Real* res_set_norm = new Real[ret_size_x * ret_size_y];
			memset(res_set_norm, 0.0, sizeof(Real)*ret_size_x * ret_size_y);

			ScaleBilnear(res_set_[loopi], res_set_norm, pnx_max, pny_max, ret_size_x, ret_size_y);
			//writeImage("test.bmp", res_set_norm, ret_size_x, ret_size_y, 8);

			res_set_norm_255_[loopi] = new Real[ret_size_x * ret_size_y];
			memset(res_set_norm_255_[loopi], 0.0, sizeof(Real)*ret_size_x * ret_size_y);

			rotateCCW180(res_set_norm, res_set_norm_255_[loopi], ret_size_x, ret_size_y, 255.0);
			//writeImage("test2.bmp", res_set_norm_255_[loopi], ret_size_x, ret_size_y, 8);

			delete[] res_set_norm;
		}

		if (fp = fopen("d:\\a\\res_set_norm_255_.dat", "wb"))
		{
			int size = sizeof(Real);
			int cnt = ret_size_x * ret_size_y;
			for (int i = 0; i < cnt; i++)
			{
				fwrite(&res_set_norm_255_[0][i], size, 1, fp);
			}
			fclose(fp);
		}



		for (int i = 0; i < nChannel; i++)
			delete[] res_set_[i];

		int N = ret_size_x * ret_size_y;

		//Real* finalimg = new Real[N * nChannel];
		focus_recon_set[vtr] = new Real[N * nChannel];
		//uchar* img = new uchar[N * nChannel];
		focus_img_set[vtr] = new uchar[N * nChannel];
		memset(focus_recon_set[vtr], 0.0, sizeof(Real) * N * nChannel);
		memset(focus_img_set[vtr], 0, sizeof(uchar) * N * nChannel);

		if (nChannel == 1)
			memcpy(focus_recon_set[vtr], res_set_norm_255_[0], sizeof(Real)*ret_size_x*ret_size_y);
		else
			reArrangeChannel(res_set_norm_255_, focus_recon_set[vtr], ret_size_x, ret_size_y, nChannel);

		std::string fname = std::string("./").append("").append("/").append("").append("_SAT_").append("").append("_").append(varname2).append("_").append(std::to_string(vtr + 1)).append(".bmp");
		
		normalize(focus_recon_set[vtr], focus_img_set[vtr], ret_size_x, ret_size_y);
		
		char tmp[MAX_PATH] = { 0, };
		sprintf(tmp, "d:\\a\\focus_recon_set[%d].dat", vtr);
		if (fp = fopen(tmp, "wb"))
		{
			int size = sizeof(Real);
			int cnt = N * nChannel;
			for (int i = 0; i < cnt; i++)
			{
				fwrite(&focus_recon_set[vtr][i], size, 1, fp);
			}
			fclose(fp);
		}

		
		focus_img_size[vtr][_X] = (int)ret_size_x;
		focus_img_size[vtr][_Y] = (int)ret_size_y;
		//saveAsImg(fname.c_str(), 8 * nChannel, focus_img_set[vtr], ret_size_x, ret_size_y);
		
		//writeImage(fname, finalimg, ret_size_x, ret_size_y, 8 * nChannel);

		for (int i = 0; i < nChannel; i++)
			delete[] res_set_norm_255_[i];
		//delete[] img;
		//delete[] finalimg;
	} // end of vtr
}

void ophRec::SaveImage(const char *path, const char *ext)
{
	int nSimStep = rec_config.SimulationStep;
	int nChannel = context_.waveNum;
	bool bCreatePupilImg = rec_config.CreatePupilFieldImg;
	Real simTo = rec_config.SimulationTo;
	Real simFrom = rec_config.SimulationFrom;
	Real step;

	if (nSimStep > 1)
	{
		step = (simTo - simFrom) / (nSimStep - 1);
	}

	char tmpPath[MAX_PATH] = { 0, };

	for (int i = 0; i < nSimStep; i++)
	{
		if (bCreatePupilImg)
		{
			for (int j = 0; j < nChannel; j++)
			{			
				if (nSimStep > 1)
					sprintf(tmpPath, "%s\\PUPIL_%.2f.%s", path, simFrom + step * i, ext);
				else
					sprintf(tmpPath, "%s\\PUPIL_%.2f.%s", path, (simFrom + simTo) / 2, ext);

				int idx = i * nChannel + j;
				saveAsImg(tmpPath, 8, img_set[idx], img_size[idx][_X], img_size[idx][_Y]);
			}
		}
		if (nSimStep > 1)
			sprintf(tmpPath, "%s\\FOCUS_%.2f.%s", path, simFrom + step * i, ext);
		else
			sprintf(tmpPath, "%s\\FOCUS_%.2f.%s", path, (simFrom + simTo) / 2, ext);
		saveAsImg(tmpPath, 8 * nChannel, focus_img_set[i], focus_img_size[i][_X], focus_img_size[i][_Y]);
	}
}

void ophRec::GetPupilFieldImage(Complex<Real>* src, Real* dst, int pnx, int pny, Real ppx, Real ppy, Real scaleX, Real scaleY)
{
	const int N = pnx * pny;
	Real ratioAtPupil = rec_config.RatioAtPupil;
	Real eyePupil = rec_config.EyePupilDiaMeter;
	vec3 eyeCenter = rec_config.EyeCenter;


	Real* dimg = new Real[N];
	memset(dimg, 0.0, sizeof(Real) * N);

	Real maxvalue = src[0].mag();
	for (int k = 0; k < N; k++)
	{
		Real val = src[k].mag();
		maxvalue = val > maxvalue ? val : maxvalue;
		dimg[k] = val;
	}

	Real SAT_VAL = maxvalue * ratioAtPupil;
	Real hss_x = (pnx * ppx) / 2.0;
	Real hss_y = (pny * ppy) / 2.0;
	Real halfPupil = eyePupil / 2.0;

	Real maxv = dimg[0], minv = dimg[0];
	for (int k = 0; k < N; k++)
	{
		if (dimg[k] > SAT_VAL)
			dimg[k] = SAT_VAL;

		maxv = (dimg[k] > maxv) ? dimg[k] : maxv;
		minv = (dimg[k] < minv) ? dimg[k] : minv;
	}

	for (int k = 0; k < N; k++)
	{
		int x = k % pnx;
		int y = k / pnx;

		Real xx = -hss_x + (ppx * x);
		Real yy = -hss_y + (pny - 1) * ppy - (ppy * y);
		Real xxx = xx - eyeCenter[_X];
		Real yyy = yy - eyeCenter[_Y];
		bool eye_pupil_mask = sqrt(xxx * xxx + yyy * yyy) < halfPupil ? 1.0 : 0.0;

		Real val = dimg[k];
		Real field_out = (val - minv) / (maxv - minv) * 255.0;
		Real field_in = (val - minv) / (maxv - minv) * 127.0 + 128.0;
		dimg[k] = field_in * eye_pupil_mask + field_out * (1 - eye_pupil_mask);
		//dimg[k] = field_in * eye_pupil_mask;

	}
	ScaleBilnear(dimg, dst, pnx, pny, scaleX, scaleY);

	delete[] dimg;
}

void ophRec::getVarname(int vtr, vec3& var_vals, std::string& varname2)
{
	bool bSimPos[3];
	memcpy(&bSimPos[0], rec_config.SimulationPos, sizeof(bSimPos));

	std::string varname;
	varname.clear();
	varname2.clear();
	if (rec_config.SimulationMode == 0) {

		varname.append("FOCUS");

		LOG("Step # %d %s = %.8f \n", vtr + 1, varname.c_str(), var_vals[0]);
		char temp[100];
		sprintf(temp, "%.5f", var_vals[0]);
		varname2.clear();
		varname2.append(varname).append("_").append(temp);
		varname2.replace(varname2.find('.'), 1, "_");

	}
	else {

		varname.append("POSITION");

		if (bSimPos[_X]) varname.append("_X");
		if (bSimPos[_Y]) varname.append("_Y");
		if (bSimPos[_Z]) varname.append("_Z");

		LOG("Step # %d %s = ", vtr + 1, varname.c_str());
		varname2.clear();
		varname2.append(varname);

		if (bSimPos[_X]) {
			LOG("%.8f ", var_vals[0]);
			char temp[100];
			sprintf(temp, "%.5f", var_vals[0]);
			varname2.append("_").append(temp);
			varname2.replace(varname2.find('.'), 1, "_");
		}
		if (bSimPos[_Y]) {
			LOG("%.8f ", var_vals[1]);
			char temp[100];
			sprintf(temp, "%.5f", var_vals[1]);
			varname2.append("_").append(temp);
			varname2.replace(varname2.find('.'), 1, "_");
		}
		if (bSimPos[_Z]) {
			LOG("%.8f ", var_vals[2]);
			char temp[100];
			sprintf(temp, "%.5f", var_vals[2]);
			varname2.append("_").append(temp);
			varname2.replace(varname2.find('.'), 1, "_");
		}
		LOG("\n");
	}
}

bool ophRec::ReconstructImage()
{
	if (!rec_config.ViewingWindow)
	{
		GetPupilFieldFromHologram();
	}
	else
	{
		GetPupilFieldFromVWHologram();
	}
	Perform_Simulation();

	return true;
}

void ophRec::Initialize()
{
	const int nChannel = context_.waveNum;
	const int pnX = context_.pixel_number[_X];
	const int pnY = context_.pixel_number[_Y];
	const int N = pnX * pnY;
	// Memory Location for Result Image
	if (complex_H != nullptr) {
		for (uint i = 0; i < nChannel; i++) {
			if (complex_H[i] != nullptr) {
				delete[] complex_H[i];
				complex_H[i] = nullptr;
			}
		}
		delete[] complex_H;
		complex_H = nullptr;
	}
	complex_H = new Complex<Real>*[nChannel];
	for (uint i = 0; i < nChannel; i++) {
		complex_H[i] = new Complex<Real>[N];
		memset(complex_H[i], 0, sizeof(Complex<Real>) * N);
	}
}

void ophRec::ophFree(void)
{
	Openholo::ophFree();

}

void circshift(Real* in, Real* out, int shift_x, int shift_y, int nx, int ny)
{
	int ti, tj;
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			ti = (i + shift_x) % nx;
			if (ti < 0)
				ti = ti + nx;
			tj = (j + shift_y) % ny;
			if (tj < 0)
				tj = tj + ny;

			out[ti + tj * nx] = in[i + j * nx];
		}
	}
}

void circshift(Complex<Real>* in, Complex<Real>* out, int shift_x, int shift_y, int nx, int ny)
{
	int ti, tj;
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			ti = (i + shift_x) % nx;
			if (ti < 0)
				ti = ti + nx;
			tj = (j + shift_y) % ny;
			if (tj < 0)
				tj = tj + ny;

			out[ti + tj * nx] = in[i + j * nx];
		}
	}
}

vec3 image_sample(float xx, float yy, int c, size_t w, size_t h, double* in)
{
	int x1 = (int)floor(xx);
	int x2 = (int)ceil(xx);
	int y1 = (int)floor(yy);
	int y2 = (int)ceil(yy);

	if (x1 < 0 || x1 >= w || x2 < 0 || x2 >= w) return vec3(0);
	if (y1 < 0 || y1 >= h || y2 < 0 || y2 >= h) return vec3(0);

	vec3 ret(0);
	double v1, v2, v3, v4, tvx, tvy;

	tvx = xx - floor(xx);
	tvy = yy - floor(yy);
	int tc = min(c, 3);
	for (int i = 0; i < tc; i++) {
		v1 = IMAGE_VAL(x1, y1, i, w, c, in);
		v2 = IMAGE_VAL(x2, y1, i, w, c, in);
		v3 = IMAGE_VAL(x1, y2, i, w, c, in);
		v4 = IMAGE_VAL(x2, y2, i, w, c, in);
		v1 = (1.0 - tvx)*v1 + tvx * v2;
		v3 = (1.0 - tvx)*v3 + tvx * v4;
		v1 = (1.0 - tvy)*v1 + tvy * v3;

		ret[i] = v1;
	}

	return ret;
}

void ScaleBilnear(double* src, double* dst, int w, int h, int neww, int newh, double multiplyval)
{
	for (int y = 0; y < newh; y++)
	{
		for (int x = 0; x < neww; x++)
		{
			float gx = (x / (float)neww) * (w - 1);
			float gy = (y / (float)newh) * (h - 1);

			vec3 ret = multiplyval * image_sample(gx, gy, 1, w, h, src);
			dst[x + y * neww] = ret[0];

			/*
			int gxi = (int)gx;
			int gyi = (int)gy;

			double a00 = src[gxi + 0 + gyi * w];
			double a01 = src[gxi + 1 + gyi * w];
			double a10 = src[gxi + 0 + (gyi + 1)*w];
			double a11 = src[gxi + 1 + (gyi + 1)*w];

			float dx = gx - gxi;
			float dy = gy - gyi;

			dst[x + y * neww] = multiplyval * (a00 * (1 - dx)*(1 - dy) + a01 * dx*(1 - dy) + a10 * (1 - dx)*dy + a11 * dx*dy);
			*/

		}
	}
}



void rotateCCW180(double* src, double* dst, int pnx, int pny, double mulival)
{
	for (int i = 0; i < pnx*pny; i++)
	{
		int x = i % pnx;
		int y = i / pnx;

		int newx = pnx - x - 1;
		int newy = pny - y - 1;

		dst[newx + newy * pnx] = mulival * src[x + y * pnx];

	}

}

void reArrangeChannel(std::vector<double*>& src, double* dst, int pnx, int pny, int chnum)
{
	for (int k = 0; k < pnx*pny; k++)
	{
		for (int c = 0; c < chnum; c++)
		{
			if (c == 0)
				dst[k*chnum + 2] = src[c][k];
			else if (c == 2)
				dst[k*chnum + 0] = src[c][k];
			else
				dst[k*chnum + c] = src[c][k];
		}

	}
}

template<typename T>
void ophRec::normalize(T* src, uchar* dst, int x, int y)
{
	T maxVal = src[0];
	T minVal = src[0];
	int size = x * y;
	
	for (int i = 0; i < size; i++)
	{
		if (src[i] < minVal)
			minVal = src[i];
		if (src[i] > maxVal)
			maxVal = src[i];
	}
	T gap = maxVal - minVal;

	for (int i = 0; i < y; i++)
	{
		int base = i * x;
		//int base2 = (y - 1 - i) * x;

		for (int j = 0; j < x; j++)
		{
			dst[base + j] = (uchar)((src[base + j] - minVal) / gap * 255.0);
		}
	}
}