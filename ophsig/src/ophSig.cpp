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

ophSig::ophSig(void) 
	: _cfgSig()
	, _angleX(0)
	, _angleY(0)
	, _redRate(0)
	, _radius(0)
{
	memset(_foc, 0, sizeof(float) * 3);
}


template<typename T>
inline void ophSig::absMat(matrix<Complex<T>>& src, matrix<T>& dst) {
	if (src.size != dst.size) {
		dst.resize(src.size[_X], src.size[_Y]);
	}
	for (int i = 0; i < src.size[_X]; i++)
	{
		for (int j = 0; j < src.size[_Y]; j++)
		{
			dst.mat[i][j] = sqrt(src.mat[i][j]._Val[_RE] * src.mat[i][j]._Val[_RE] + src.mat[i][j]._Val[_IM] * src.mat[i][j]._Val[_IM]);
		}
	}
}


template<typename T>
inline void ophSig::absMat(matrix<T>& src, matrix<T>& dst) {
	if (src.size != dst.size) {
		dst.resize(src.size[_X], src.size[_Y]);
	}
	for (int i = 0; i < src.size[_X]; i++)
	{
		for (int j = 0; j < src.size[_Y]; j++)
		{
			dst.mat[i][j] = abs(src.mat[i][j]);
		}
	}
}

template<typename T>
inline void ophSig::angleMat(matrix<Complex<T>>& src, matrix<T>& dst) {
	if (src.size != dst.size) {
		dst.resize(src.size[_X], src.size[_Y]);
	}
	for (int i = 0; i < src.size[_X]; i++)
	{
		for (int j = 0; j < src.size[_Y]; j++)
		{
			angle(src(i, j), dst(i, j));
		}
	}
}

template<typename T>
inline void ophSig::conjMat(matrix<Complex<T>>& src, matrix<Complex<T>>& dst) {
	if (src.size != dst.size) {
		dst.resize(src.size[_X], src.size[_Y]);
	}
	for (int i = 0; i < src.size[_X]; i++)
	{
		for (int j = 0; j < src.size[_Y]; j++)
		{
			dst(i, j) = src(i, j).conj();

		}
	}
}

template<typename T>
inline void ophSig::expMat(matrix<Complex<T>>& src, matrix<Complex<T>>& dst) {
	if (src.size != dst.size) {
		dst.resize(src.size[_X], src.size[_Y]);
	}
	for (int i = 0; i < src.size[_X]; i++)
	{
		for (int j = 0; j < src.size[_Y]; j++)
		{
			dst.mat[i][j]._Val[_RE] = exp(src.mat[i][j]._Val[_RE]) * cos(src.mat[i][j]._Val[_IM]);
			dst.mat[i][j]._Val[_IM] = exp(src.mat[i][j]._Val[_RE]) * sin(src.mat[i][j]._Val[_IM]);
		}
	}
}

template<typename T>
inline void ophSig::expMat(matrix<T>& src, matrix<T>& dst) {
	if (src.size != dst.size) {
		dst.resize(src.size[_X], src.size[_Y]);
	}
	for (int i = 0; i < src.size[_X]; i++)
	{
		for (int j = 0; j < src.size[_Y]; j++)
		{
			dst.mat[i][j] = exp(src.mat[i][j]);
		}
	}
}

template<typename T>
inline Real ophSig::maxOfMat(matrix<T>& src) {
	Real max = MIN_DOUBLE;
	for (int i = 0; i < src.size[_X]; i++)
	{
		for (int j = 0; j < src.size[_Y]; j++)
		{
			if (src(i, j) > max) max = src(i, j);
		}
	}
	return max;
}

template<typename T>
void ophSig::meshgrid(vector<T>& src1, vector<T>& src2, matrix<T>& dst1, matrix<T>& dst2)
{
	/*int src1_total = src1.size();
	int src2_total = src2.size();*/
	int src1_total = static_cast<int>(src1.size());
	int src2_total = static_cast<int>(src2.size());

	dst1.resize(src2_total, src1_total);
	dst2.resize(src2_total, src1_total);
	for (int i = 0; i < src1_total; i++)
	{
		for (int j = 0; j < src2_total; j++)
		{
			dst1(j, i) = src1.at(i);
			dst2(j, i) = src2.at(j);
		}
	}
}

template<typename T>
inline Real ophSig::minOfMat(matrix<T>& src) {
	Real min = MAX_DOUBLE;
	for (int i = 0; i < src.size[_X]; i++)
	{
		for (int j = 0; j < src.size[_Y]; j++)
		{
			if (src(i, j) < min) min = src(i, j);
		}
	}
	return min;
}

template<typename T>
void ophSig::fft1(matrix<Complex<T>> &src, matrix<Complex<T>> &dst, int sign, uint flag)
{
	if (src.size != dst.size) {
		dst.resize(src.size[_X], src.size[_Y]);
	}
	fftw_complex *fft_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * src.size[_Y]);
	fftw_complex *fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * src.size[_Y]);

	for (int i = 0; i < src.size[_Y]; i++) {
		fft_in[i][_RE] = src(0, i).real();
		fft_in[i][_IM] = src(0, i).imag();
	}

	fftw_plan plan = fftw_plan_dft_1d(src.size[_Y], fft_in, fft_out, sign, flag);

	fftw_execute(plan);
	if (sign == OPH_FORWARD)
	{
		for (int i = 0; i < src.size[_Y]; i++) {
			dst(0, i)._Val[_RE] = fft_out[i][_RE];
			dst(0, i)._Val[_IM] = fft_out[i][_IM];
		}
	}
	else if (sign == OPH_BACKWARD)
	{
		for (int i = 0; i < src.size[_Y]; i++) {
			dst(0, i)._Val[_RE] = fft_out[i][_RE] / src.size[_Y];
			dst(0, i)._Val[_IM] = fft_out[i][_IM] / src.size[_Y];
		}
	}

	fftw_destroy_plan(plan);
	fftw_free(fft_in);
	fftw_free(fft_out);
}
template<typename T>
void ophSig::fft2(matrix<Complex<T>> &src, matrix<Complex<T>> &dst, int sign, uint flag)
{
	if (src.size != dst.size) {
		dst.resize(src.size[_X], src.size[_Y]);
	}

	fftw_complex *fft_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * src.size[_X] * src.size[_Y]);
	fftw_complex *fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * src.size[_X] * src.size[_Y]);

	for (int i = 0; i < src.size[_X]; i++) {
		for (int j = 0; j < src.size[_Y]; j++) {
			fft_in[src.size[_Y] * i + j][_RE] = src(i, j).real();
			fft_in[src.size[_Y] * i + j][_IM] = src(i, j).imag();
		}
	}

	fftw_plan plan = fftw_plan_dft_2d(src.size[_X], src.size[_Y], fft_in, fft_out, sign, flag);

	fftw_execute(plan);
	if (sign == OPH_FORWARD)
	{
		for (int i = 0; i < src.size[_X]; i++) {
			for (int j = 0; j < src.size[_Y]; j++) {
				dst(i, j)._Val[_RE] = fft_out[src.size[_Y] * i + j][_RE];
				dst(i, j)._Val[_IM] = fft_out[src.size[_Y] * i + j][_IM];
			}
		}
	}
	else if (sign == OPH_BACKWARD)
	{
		for (int i = 0; i < src.size[_X]; i++) {
			for (int j = 0; j < src.size[_Y]; j++) {
				dst(i, j)._Val[_RE] = fft_out[src.size[_Y] * i + j][_RE] / (src.size[_X] * src.size[_Y]);
				dst(i, j)._Val[_IM] = fft_out[src.size[_Y] * i + j][_IM] / (src.size[_X] * src.size[_Y]);

			}
		}
	}

	fftw_destroy_plan(plan);
	fftw_free(fft_in);
	fftw_free(fft_out);
}
template<typename T>
void ophSig::fftShift(matrix<Complex<T>> &src, matrix<Complex<T>> &dst)
{
	if (src.size != dst.size) {
		dst.resize(src.size[_X], src.size[_Y]);
	}
	int xshift = src.size[_X] / 2;
	int yshift = src.size[_Y] / 2;
	for (int i = 0; i < src.size[_X]; i++)
	{
		int ii = (i + xshift) % src.size[_X];
		for (int j = 0; j < src.size[_Y]; j++)
		{
			int jj = (j + yshift) % src.size[_Y];
			dst.mat[ii][jj]._Val[_RE] = src.mat[i][j].real();
			dst.mat[ii][jj]._Val[_IM] = src.mat[i][j].imag();

		}

	}
}

vector<Real> ophSig::linspace(double first, double last, int len) {
	vector<Real> result;
	for (int i = 0; i < len; i++)
	{
		result.push_back(0);
	}
	double step = (last - first) / (len - 1);
	for (int i = 0; i < len; i++) { result[i] = first + i*step; }
	return result;
}

template<typename T>
inline void ophSig::meanOfMat(matrix<T> &src, double &dst)
{
	dst = 0;
	for (int i = 0; i < src.size[_X]; i++)
	{
		for (int j = 0; j < src.size[_Y]; j++)
		{
			dst += src(i, j);
		}
	}
	dst = dst / (src.size[_X] * src.size[_Y]);
}

template<typename T>
void ophSig::linInterp(vector<T> &X, matrix<Complex<T>> &src, vector<T> &Xq, matrix<Complex<T>> &dst)
{
	if (src.size != dst.size) {
		dst.resize(src.size[_X], src.size[_Y]);
	}
	int size = src.size[_Y];

	for (int i = 0, j = 0; j < dst.size[_Y]; j++)
	{
		if ((Xq[j]) >= (X[size - 2]))
		{
			i = size - 2;
		}
		else
		{
			while ((Xq[j]) > (X[i + 1])) i++;
		}
		dst(0, j)._Val[_RE] = src(0, i).real() + (src(0, i + 1).real() - src(0, i).real()) / (X[i + 1] - X[i]) * (Xq[j] - X[i]);
		dst(0, j)._Val[_IM] = src(0, i).imag() + (src(0, i + 1).imag() - src(0, i).imag()) / (X[i + 1] - X[i]) * (Xq[j] - X[i]);
	}
}

int ophSig::loadAsOhc(const char *fname)
{
	std::string fullname = fname;
	if (checkExtension(fname, ".ohc") == 0) fullname.append(".ohc");
	OHC_decoder->setFileName(fullname.c_str());

	if (!OHC_decoder->load()) return -1;
	vector<Real> wavelengthArray;
	OHC_decoder->getWavelength(wavelengthArray);

	if (_cfgSig.colorType == 0)
	{
		
		_cfgSig.lambda = wavelengthArray[0];

		ivec2 pixel_number;
		pixel_number = OHC_decoder->getNumOfPixel();
		_cfgSig.rows = pixel_number[0];
		_cfgSig.cols = pixel_number[1];

		OHC_decoder->getComplexFieldData(&complex_H);
		Buffer2Field(complex_H[0], ComplexH[0], pixel_number);
	}
	else if (_cfgSig.colorType == 1)
	{
		for (int i = 0; i < wavelengthArray.size(); i++)
		{
			_cfgSig.wavelength[2 - i] = wavelengthArray[i];
		}
		ivec2 pixel_number;
		pixel_number = OHC_decoder->getNumOfPixel();
		_cfgSig.rows = pixel_number[0];
		_cfgSig.cols = pixel_number[1];

		OHC_decoder->getComplexFieldData(&complex_H);
		for (int i = 0; i < wavelengthArray.size(); i++)
		{
			Buffer2Field(complex_H[i], ComplexH[2 - i], pixel_number);
		}
	}
	return 0;

}

int ophSig::saveAsOhc(const char *fname)
{
	std::string fullname = fname;
	if (checkExtension(fname, ".ohc") == 0) fullname.append(".ohc");
	OHC_encoder->setFileName(fullname.c_str());

	OHC_encoder->setNumOfPixel(_cfgSig.rows, _cfgSig.cols);

	OHC_encoder->setFieldEncoding(FldStore::Directly, FldCodeType::RI);
	if (_cfgSig.colorType == 0)
	{
		OHC_encoder->setNumOfWavlen(1);
		OHC_encoder->setWavelength(_cfgSig.lambda, LenUnit::nm);
		*&complex_H = new Complex<Real>*[1];
		Field2Buffer(ComplexH[0], complex_H);
		OHC_encoder->addComplexFieldData(complex_H[0]);
	}
	else if (_cfgSig.colorType == 1)
	{
		OHC_encoder->setNumOfWavlen(3);
		OHC_encoder->setWavelength(_cfgSig.wavelength[0], LenUnit::nm);
		*&complex_H = new Complex<Real>*[3];
		for (int i = 0; i < 3; i++)
		{
			Field2Buffer(ComplexH[2 - i], &complex_H[i]);
			OHC_encoder->addComplexFieldData(complex_H[i]);
		}
	}

	if (!OHC_encoder->save()) return -1;

	return 1;
}

bool ophSig::load(const char *real, const char *imag)
{
	string realname = real;
	string imagname = imag;

	int nz = 1;
	char* RGB_name[3] = { "","","" };

	if (_cfgSig.colorType) {
		nz = 3;
		RGB_name[0] = "_B";
		RGB_name[1] = "_G";
		RGB_name[2] = "_R";
	}

	int checktype = static_cast<int>(realname.rfind("."));
	OphRealField realMat[3], imagMat[3];

	std::string realtype = realname.substr(checktype + 1, realname.size());
	std::string imgtype = imagname.substr(checktype + 1, realname.size());

	if (realtype != imgtype) {
		LOG("failed : The data type between real and imaginary is different!\n");
		return false;
	}
	if (realtype == "bmp")
	{
		realname = real;
		imagname = imag;
		//realname.insert(checktype, RGB_name[z]);
		//imagname.insert(checktype, RGB_name[z]);

		uchar* realdata = loadAsImg(realname.c_str());
		uchar* imagdata = loadAsImg(imagname.c_str());
		
		if (realdata == 0 && imagdata == 0) {
			cout << "failed : hologram data load was failed." << endl;
			return false;
		}

		for (int z = 0; z < nz; z++)
		{
			realMat[z].resize(_cfgSig.rows, _cfgSig.cols);
			imagMat[z].resize(_cfgSig.rows, _cfgSig.cols);
		
			for (int i = _cfgSig.rows - 1; i >= 0; i--)
			{
				for (int j = 0; j < _cfgSig.cols; j++)
				{
					realMat[z](_cfgSig.rows - i - 1, j) = (double)realdata[(i * _cfgSig.cols + j)*nz + z];
					imagMat[z](_cfgSig.rows - i - 1, j) = (double)imagdata[(i * _cfgSig.cols + j)*nz + z];
				}
			}
		}
		delete[] realdata;
		delete[] imagdata;

		LOG("file load complete!\n");
	}
	else if (realtype == "bin")
	{
		int total = _cfgSig.rows*_cfgSig.cols;

		double *realdata = new  double[total];
		double *imagdata = new  double[total];

		for (int z = 0; z < nz; z++)
		{
			realname = real;
			imagname = imag;

			realname.insert(checktype, RGB_name[z]);
			imagname.insert(checktype, RGB_name[z]);

			ifstream freal(realname.c_str(), ifstream::binary);
			ifstream fimag(imagname.c_str(), ifstream::binary);

			freal.read(reinterpret_cast<char*>(realdata), sizeof(double) * total);
			fimag.read(reinterpret_cast<char*>(imagdata), sizeof(double) * total);

			for (int col = 0; col < _cfgSig.cols; col++)
			{
				for (int row = 0; row < _cfgSig.rows; row++)
				{
					realMat[z].resize(_cfgSig.rows, _cfgSig.cols);
					imagMat[z].resize(_cfgSig.rows, _cfgSig.cols);

					realMat[z](row, col) = realdata[_cfgSig.rows*col + row];
					imagMat[z](row, col) = imagdata[_cfgSig.rows*col + row];
				}
			}
			freal.close();
			fimag.close();
		}
		delete[] realdata;
		delete[] imagdata;
	}	
	else
	{
		LOG("wrong type\n");
	}

	//nomalization
	double realout, imagout;
	for (int z = 0; z < nz; z++)
	{
		meanOfMat(realMat[z], realout); meanOfMat(imagMat[z], imagout);
		realMat[z] / realout; imagMat[z] / imagout;
		absMat(realMat[z], realMat[z]);
		absMat(imagMat[z], imagMat[z]);
		realout = maxOfMat(realMat[z]); imagout = maxOfMat(imagMat[z]);
		realMat[z] / realout; imagMat[z] / imagout;
		realout = minOfMat(realMat[z]); imagout = minOfMat(imagMat[z]);
		realMat[z] - realout; imagMat[z] - imagout;

		ComplexH[z].resize(_cfgSig.rows, _cfgSig.cols);

		for (int i = 0; i < _cfgSig.rows; i++)
		{
			for (int j = 0; j < _cfgSig.cols; j++)
			{
				ComplexH[z](i, j)._Val[_RE] = realMat[z](i, j);
				ComplexH[z](i, j)._Val[_IM] = imagMat[z](i, j);

			}
		}
	}
	LOG("data nomalization complete\n");

	return true;
}

bool ophSig::save(const char *real, const char *imag)
{
	string realname = real;
	string imagname = imag;


	int nz = 1;
	char* RGB_name[3] = { "","","" };

	if (_cfgSig.colorType) {
		nz = 3; 
		RGB_name[0] = "_B";
		RGB_name[1] = "_G";
		RGB_name[2] = "_R";
	}

	int checktype = static_cast<int>(realname.rfind("."));
	string type = realname.substr(checktype + 1, realname.size());
	if (type == "bin")
	{
		double *realdata = new  double[_cfgSig.rows * _cfgSig.cols];
		double *imagdata = new  double[_cfgSig.rows * _cfgSig.cols];

		for (int z = 0; z < nz; z++)
		{
			realname = real;
			imagname = imag;
			realname.insert(checktype, RGB_name[z]);
			imagname.insert(checktype, RGB_name[z]);
			std::ofstream cos(realname.c_str(), std::ios::binary);
			std::ofstream sin(imagname.c_str(), std::ios::binary);

			if (!cos.is_open()) {
				LOG("real file not found.\n");
				cos.close();
				delete[] realdata;
				delete[] imagdata;
				return FALSE;
			}

			if (!sin.is_open()) {
				LOG("imag file not found.\n");
				sin.close();
				delete[] realdata;
				delete[] imagdata;
				return FALSE;
			}

			for (int col = 0; col < _cfgSig.cols; col++)
			{
				for (int row = 0; row < _cfgSig.rows; row++)
				{
					realdata[_cfgSig.rows*col + row] = ComplexH[z].mat[row][col]._Val[_RE];
					imagdata[_cfgSig.rows*col + row] = ComplexH[z].mat[row][col]._Val[_IM];
				}
			}
			cos.write(reinterpret_cast<const char*>(realdata), sizeof(double) * _cfgSig.rows * _cfgSig.cols);
			sin.write(reinterpret_cast<const char*>(imagdata), sizeof(double) * _cfgSig.rows * _cfgSig.cols);
			
			cos.close();
			sin.close();
		}
		delete[]realdata;
		delete[]imagdata;
	
		std::cout << "file save binary complete" << endl;
	}
	else {
		LOG("failed : The Invalid data type! - %s\n", type);
	}
	return TRUE;
}

bool ophSig::save(const char *real)
{
	string realname = real;
	string fname;
	int nz = 1;
	char* RGB_name[3] = {"","",""};

	if (_cfgSig.colorType) {
		nz = 3;
		RGB_name[0] = "_B";
		RGB_name[1] = "_G";
		RGB_name[2] = "_R";
	}
	int checktype = static_cast<int>(realname.rfind("."));

	if (realname.substr(checktype + 1, realname.size()) == "bmp")
	{
		oph::uchar* realdata;
		realdata = (oph::uchar*)malloc(sizeof(oph::uchar) * _cfgSig.rows * _cfgSig.cols * nz);
		
		double gamma = 0.5;
		double maxIntensity = 0.0;
		double realVal = 0.0;
		double imagVal = 0.0;
		double intensityVal = 0.0;

		for (int z = 0; z < nz; z++)
		{
			for (int j = 0; j < ComplexH[z].size[_Y]; j++) {
				for (int i = 0; i < ComplexH[z].size[_X]; i++) {
					realVal = ComplexH[z](i, j)._Val[_RE];
					imagVal = ComplexH[z](i, j)._Val[_RE];
					intensityVal = realVal*realVal + imagVal*imagVal;
					if (intensityVal > maxIntensity) {
						maxIntensity = intensityVal;
					}
				}
			}
			for (int i = _cfgSig.rows - 1; i >= 0; i--)
			{
				for (int j = 0; j <  _cfgSig.cols; j++)
				{
					realVal = ComplexH[z](_cfgSig.rows - i - 1, j)._Val[_RE];
					imagVal = ComplexH[z](_cfgSig.rows - i - 1, j)._Val[_IM];
					intensityVal = realVal*realVal + imagVal*imagVal;
					realdata[(i*_cfgSig.cols + j)* nz + z] = (uchar)(pow(intensityVal / maxIntensity, gamma)*255.0);
				}
			}
			//sprintf(str, "_%.2u", z);
			//realname.insert(checktype, RGB_name[z]);
		}
		saveAsImg(realname.c_str(), nz * 8, realdata, _cfgSig.cols, _cfgSig.rows);

		delete[] realdata;
		std::cout << "file save bmp complete" << endl;
		return TRUE;
	}
	else if (realname.substr(checktype + 1, realname.size()) == "bin")
	{
		double *realdata = new  double[_cfgSig.rows * _cfgSig.cols];

		for (int z = 0; z < nz; z++)
		{
			realname = real;
			realname.insert(checktype, RGB_name[z]);
			std::ofstream cos(realname.c_str(), std::ios::binary);

			if (!cos.is_open()) {
				LOG("real file not found.\n");
				cos.close();
				delete[] realdata;
				return FALSE;
			}

			for (int col = 0; col < _cfgSig.cols; col++)
			{
				for (int row = 0; row < _cfgSig.rows; row++)
				{
					realdata[_cfgSig.rows*col + row] = ComplexH[z].mat[row][col]._Val[_RE];
				}
			}
			cos.write(reinterpret_cast<const char*>(realdata), sizeof(double) * _cfgSig.rows * _cfgSig.cols);

			cos.close();
		}
		delete[]realdata;
		std::cout << "file save binary complete" << endl;
	}
	
	return TRUE;
}

bool ophSig::sigConvertOffaxis() {
	OphRealField x, y, H1;
	vector<Real> X, Y;
	OphComplexField expSource, exp, offh;
	x.resize(_cfgSig.rows, _cfgSig.cols);
	y.resize(_cfgSig.rows, _cfgSig.cols);
	expSource.resize(_cfgSig.rows, _cfgSig.cols);
	exp.resize(_cfgSig.rows, _cfgSig.cols);
	offh.resize(_cfgSig.rows, _cfgSig.cols);
	H1.resize(_cfgSig.rows, _cfgSig.cols);
	vector<Real> r = linspace(1, _cfgSig.cols, _cfgSig.cols);
	vector<Real> c = linspace(1, _cfgSig.rows, _cfgSig.rows);

	for (int i = 0; i < _cfgSig.cols; i++)
	{
		X.push_back(_cfgSig.width / (_cfgSig.cols - 1)*(r[i] - 1) - _cfgSig.width / 2);
	}
	for (int i = 0; i < _cfgSig.rows; i++)
	{
		Y.push_back(_cfgSig.height / (_cfgSig.rows - 1)*(c[i] - 1) - _cfgSig.height / 2);
	}
	meshgrid(X, Y, x, y);

	for (int i = 0; i < _cfgSig.rows; i++)
	{
		for (int j = 0; j < _cfgSig.cols; j++)
		{
			expSource(i, j)._Val[_RE] = 0;
			expSource(i, j)._Val[_IM] = ((2 * M_PI) / _cfgSig.lambda)*((x(i, j) *sin(_angleX)) + (y(i, j) *sin(_angleY)));
		}
	}
	expMat(expSource, exp);
	offh = ComplexH[0].mulElem(exp);
	for (int i = 0; i < _cfgSig.rows; i++)
	{
		for (int j = 0; j < _cfgSig.cols; j++)
		{
			H1(i, j) = offh(i, j)._Val[_RE];
		}
	}

	double out = minOfMat(H1);

	H1 - out;
	out = maxOfMat(H1);
	for (int i = 0; i < ophSig::_cfgSig.rows; i++)
	{
		for (int j = 0; j < ophSig::_cfgSig.cols; j++)
		{
			ComplexH[0](i, j)._Val[_RE] = H1(i, j) / out;
			ComplexH[0](i, j)._Val[_IM] = 0;
		}
	}

	return true;
}

bool ophSig::sigConvertHPO() {
	OphRealField x, y; OphComplexField expSource, exp, F1, G1, OUT_H, HPO;
	vector<Real> X, Y;
	x.resize(_cfgSig.rows, _cfgSig.cols);
	y.resize(_cfgSig.rows, _cfgSig.cols);
	expSource.resize(_cfgSig.rows, _cfgSig.cols);
	exp.resize(_cfgSig.rows, _cfgSig.cols);
	F1.resize(_cfgSig.rows, _cfgSig.cols);
	G1.resize(_cfgSig.rows, _cfgSig.cols);
	OUT_H.resize(_cfgSig.rows, _cfgSig.cols);
	HPO.resize(_cfgSig.rows, _cfgSig.cols);

	float NA = _cfgSig.width / (2 * _cfgSig.z);
	float NA_g = NA*_redRate;

	vector<Real> r = linspace(1, _cfgSig.cols, _cfgSig.cols);
	vector<Real> c = linspace(1, _cfgSig.rows, _cfgSig.rows);
	for (int i = 0; i < _cfgSig.cols; i++)
	{
		X.push_back(2 * M_PI*(r[i] - 1) / _cfgSig.width - M_PI*(_cfgSig.cols - 1) / _cfgSig.width);
	}
	for (int i = 0; i < _cfgSig.rows; i++)
	{
		Y.push_back(2 * M_PI*(c[i] - 1) / _cfgSig.height - M_PI*(_cfgSig.rows - 1) / _cfgSig.height);
	}

	meshgrid(X, Y, x, y);
	double sigmaf = (_cfgSig.z*_cfgSig.lambda) / (4 * M_PI);
	for (int i = 0; i < _cfgSig.rows; i++)
	{
		for (int j = 0; j < _cfgSig.cols; j++)
		{
			expSource(i, j)._Val[_RE] = 0;
			expSource(i, j)._Val[_IM] = sigmaf*(y(i, j)*y(i, j));
		}
	}
	expMat(expSource, exp);
	fftShift(exp, F1);
	for (int i = 0; i < _cfgSig.rows; i++)
	{
		for (int j = 0; j < _cfgSig.cols; j++)
		{
			expSource(i, j)._Val[_RE] = ((-M_PI*((_cfgSig.lambda / (2 * M_PI*NA_g))*(_cfgSig.lambda / (2 * M_PI*NA_g))))*((y(i, j)*y(i, j))));
			expSource(i, j)._Val[_IM] = 0;
		}
	}
	expMat(expSource, exp);
	fftShift(exp, G1);
	fft2(ComplexH[0], OUT_H, OPH_FORWARD);
	HPO = G1.mulElem(F1.mulElem(OUT_H));
	fft2(HPO, ComplexH[0], OPH_BACKWARD);
	return true;
}

bool ophSig::sigConvertCAC(double red, double green, double blue) {
	OphRealField x, y;
	OphComplexField FFZP, exp, FH, conj, FH_CAC;
	vector<Real> X, Y;
	_cfgSig.wavelength[0] = blue;
	_cfgSig.wavelength[1] = green;
	_cfgSig.wavelength[2] = red;
	x.resize(_cfgSig.rows, _cfgSig.cols);
	y.resize(_cfgSig.rows, _cfgSig.cols);
	FFZP.resize(_cfgSig.rows, _cfgSig.cols);
	exp.resize(_cfgSig.rows, _cfgSig.cols);
	FH.resize(_cfgSig.rows, _cfgSig.cols);
	conj.resize(_cfgSig.rows, _cfgSig.cols);
	FH_CAC.resize(_cfgSig.rows, _cfgSig.cols);
	for (int z = 0; z < 3; z++)
	{
		double sigmaf = ((_foc[2] - _foc[z])*_cfgSig.wavelength[z]) / (4 * M_PI);
		vector<Real> r = linspace(1, _cfgSig.cols, _cfgSig.cols);
		vector<Real> c = linspace(1, _cfgSig.rows, _cfgSig.rows);
		for (int i = 0; i < ophSig::_cfgSig.cols; i++)
		{
			X.push_back(2 * M_PI*(r[i] - 1) / _radius - M_PI*(ophSig::_cfgSig.cols - 1) / _radius);
		}
		for (int i = 0; i < ophSig::_cfgSig.rows; i++)
		{
			Y.push_back(2 * M_PI*(c[i] - 1) / _radius - M_PI*(ophSig::_cfgSig.rows - 1) / _radius);
		}
		meshgrid(X, Y, x, y);
		for (int i = 0; i < ophSig::_cfgSig.rows; i++)
		{
			for (int j = 0; j < ophSig::_cfgSig.cols; j++)
			{
				FFZP(i, j)._Val[_RE] = 0;
				FFZP(i, j)._Val[_IM] = sigmaf*((x(i, j)*x(i, j)) + (y(i, j)*y(i, j)));
			}
		}
		expMat(FFZP, exp);
		fftShift(exp, FFZP);
		fft2(ComplexH[z], FH, OPH_FORWARD);
		conjMat(FFZP, conj);
		FH_CAC = FH.mulElem(conj);
		fft2(FH_CAC, ComplexH[z], OPH_BACKWARD);
	}
	return true;
}

bool ophSig::readConfig(const char* fname)
{
	LOG("Reading....%s...\n", fname);

	/*XML parsing*/
	tinyxml2::XMLDocument xml_doc;
	tinyxml2::XMLNode *xml_node;

	if (checkExtension(fname, ".xml") == 0)
	{
		LOG("file's extension is not 'xml'\n");
		return false;
	}
	auto ret = xml_doc.LoadFile(fname);
	if (ret != tinyxml2::XML_SUCCESS)
	{
		LOG("Failed to load file \"%s\"\n", fname);
		return false;
	}

	xml_node = xml_doc.FirstChild();

	(xml_node->FirstChildElement("rows"))->QueryIntText(&_cfgSig.rows);
	(xml_node->FirstChildElement("cols"))->QueryIntText(&_cfgSig.cols);
	(xml_node->FirstChildElement("width"))->QueryFloatText(&_cfgSig.width);
	(xml_node->FirstChildElement("height"))->QueryFloatText(&_cfgSig.height);
	(xml_node->FirstChildElement("wavelength"))->QueryDoubleText(&_cfgSig.lambda);
	(xml_node->FirstChildElement("colorType"))->QueryIntText(&_cfgSig.colorType);
	(xml_node->FirstChildElement("NA"))->QueryFloatText(&_cfgSig.NA);
	(xml_node->FirstChildElement("z"))->QueryFloatText(&_cfgSig.z);
	(xml_node->FirstChildElement("angle_X"))->QueryFloatText(&_angleX);
	(xml_node->FirstChildElement("angle_Y"))->QueryFloatText(&_angleY);
	(xml_node->FirstChildElement("reduction_rate"))->QueryFloatText(&_redRate);
	(xml_node->FirstChildElement("radius_of_lens"))->QueryFloatText(&_radius);
	(xml_node->FirstChildElement("focal_length_R"))->QueryFloatText(&_foc[2]);
	(xml_node->FirstChildElement("focal_length_G"))->QueryFloatText(&_foc[1]);
	(xml_node->FirstChildElement("focal_length_B"))->QueryFloatText(&_foc[0]);

	return true;
}


bool ophSig::propagationHolo(float depth) {
	int i, j, nx = 0, ny = 0, nz = 1;
	Real x, y, sigmaf;

	OphComplexField FH;
	OphComplexField FFZP;

	if (_cfgSig.colorType)		nz = 3;	
	else 	_cfgSig.wavelength[0] = _cfgSig.lambda;	

	for (int z = 0; z < nz; z++) {
		nx = ComplexH[z].size[_X];
		ny = ComplexH[z].size[_Y];

		sigmaf = (depth * _cfgSig.wavelength[z]) / (4 * M_PI);
		
		FH.resize(nx, ny);
		FFZP.resize(nx, ny);

		fft2(ComplexH[z], FH);

		for (i = 0; i < nx; i++)
		{
			int ii = (i + ny / 2) % ny;
			for (j = 0; j < ny; j++)
			{
				x = (2 * M_PI * (i)) / _cfgSig.width - M_PI*(nx - 1) / (_cfgSig.width);
				y = (2 * M_PI * (j)) / _cfgSig.height - M_PI*(ny - 1) / (_cfgSig.height);
				int jj = (j + nx / 2) % nx;
				double temp = sigmaf * ((x * x + y * y));
				FFZP(ii, jj)._Val[_RE] = cos(temp) * FH(ii, jj)._Val[_RE] - sin(temp) * FH(ii, jj)._Val[_IM];
				FFZP(ii, jj)._Val[_IM] = sin(temp) * FH(ii, jj)._Val[_RE] + cos(temp) * FH(ii, jj)._Val[_IM];
			}
		}	

		this->fft2(FFZP, ComplexH[z], OPH_BACKWARD);
	}
	return true;
}

OphComplexField ophSig::propagationHolo(OphComplexField complexH, float depth) {
	int i, j, nx = 0, ny = 0;
	Real x, y, sigmaf;

	OphComplexField FH;
	OphComplexField FFZP;
	
	nx = complexH.size[_X];
	ny = complexH.size[_Y];

	sigmaf = (depth * _cfgSig.lambda) / (4 * M_PI);

	FH.resize(nx, ny);
	FFZP.resize(nx, ny);
	fft2(complexH, FH);

	for (i = 0; i < nx; i++)
	{
		int ii = (i + ny / 2) % ny;
		for (j = 0; j < ny; j++)
		{
			x = (2 * M_PI * (i)) / _cfgSig.width - M_PI*(nx - 1) / (_cfgSig.width);
			y = (2 * M_PI * (j)) / _cfgSig.height - M_PI*(ny - 1) / (_cfgSig.height);
			int jj = (j + nx / 2) % nx;
			double temp = sigmaf * ((x * x + y * y));
			FFZP(ii, jj)._Val[_RE] = cos(temp) * FH(ii, jj)._Val[_RE] - sin(temp) * FH(ii, jj)._Val[_IM];
			FFZP(ii, jj)._Val[_IM] = sin(temp) * FH(ii, jj)._Val[_RE] + cos(temp) * FH(ii, jj)._Val[_IM];
		}
	}
	fft2(FFZP, complexH, OPH_BACKWARD);
	
	return complexH;
}

double ophSig::sigGetParamAT() {

	Real max = 0;	double index = 0;
	OphComplexField Flr(ComplexH[0].size[_X], ComplexH[0].size[_Y]);
	OphComplexField Fli(ComplexH[0].size[_X], ComplexH[0].size[_Y]);
	OphComplexField Hsyn(ComplexH[0].size[_X], ComplexH[0].size[_Y]);
	OphComplexField Hsyn_copy1(ComplexH[0].size[_X], ComplexH[0].size[_Y]);
	OphComplexField Hsyn_copy2(ComplexH[0].size[_X], ComplexH[0].size[_Y]);
	OphRealField Hsyn_copy3(ComplexH[0].size[_X], ComplexH[0].size[_Y]);

	OphComplexField Fo(ComplexH[0].size[_X], ComplexH[0].size[_Y]);
	OphComplexField Fo1(ComplexH[0].size[_X], ComplexH[0].size[_Y]);
	OphComplexField Fon, yn, Ab_yn;

	OphRealField Ab_yn_half, kx, ky, temp, G; 
	vector<Real> r, c;
	vector<Real> t, tn;

	r = ophSig::linspace(1, _cfgSig.rows, _cfgSig.rows);
	c = ophSig::linspace(1, _cfgSig.cols, _cfgSig.cols);

	for (int i = 0; i < r.size(); i++)
	{
		r.at(i) = (2 * M_PI*(r.at(i) - 1) / _cfgSig.height - M_PI*(_cfgSig.rows - 1) / _cfgSig.height);
	}

	for (int i = 0; i < c.size(); i++)
	{
		c.at(i) = (2 * M_PI*(c.at(i) - 1) / _cfgSig.width - M_PI*(_cfgSig.cols - 1) / _cfgSig.width);
	}
	meshgrid(c, r, kx, ky);

	float NA_g = (float)0.025;

	temp.resize(kx.size[_X], kx.size[_Y]);
	G.resize(temp.size[_X], temp.size[_Y]);

	for (int i = 0; i < temp.size[_X]; i++)
	{
		for (int j = 0; j < temp.size[_Y]; j++)
		{
			temp(i, j) = -M_PI * (_cfgSig.lambda / (2 * M_PI * NA_g)) * (_cfgSig.lambda / (2 * M_PI * NA_g)) * (kx(i, j) * kx(i, j) + ky(i, j) * ky(i, j));
		}
	}

	expMat(temp, G);

	for (int i = 0; i < ComplexH[0].size[_X]; i++)
	{
		for (int j = 0; j < ComplexH[0].size[_Y]; j++)
		{
			Flr(i, j)._Val[0] = ComplexH[0](i, j)._Val[0];
			Fli(i, j)._Val[0] = ComplexH[0](i, j)._Val[1];
			Flr(i, j)._Val[1] = 0;
			Fli(i, j)._Val[1] = 0;
		}
	}

	fft2(Flr, Flr);
	fft2(Fli, Fli);

	for (int i = 0; i < Hsyn.size[_X]; i++)
	{
		for (int j = 0; j < Hsyn.size[_Y]; j++)
		{
			Hsyn(i, j)._Val[_RE] = Flr(i, j)._Val[_RE] * G(i, j);
			Hsyn(i, j)._Val[_IM] = Fli(i, j)._Val[_RE] * G(i, j);
		}
	}

	for (int i = 0; i < Hsyn.size[_X]; i++)
	{
		for (int j = 0; j < Hsyn.size[_Y]; j++)
		{
			Hsyn_copy1(i, j) = Hsyn(i, j);
			Hsyn_copy2(i, j) = Hsyn_copy1(i, j) * Hsyn(i, j);
		}
	}

	absMat(Hsyn, Hsyn_copy3);
	Hsyn_copy3 = Hsyn_copy3.mulElem(Hsyn_copy3) + pow(10, -300);




	for (int i = 0; i < Hsyn_copy2.size[_X]; i++)
	{
		for (int j = 0; j < Hsyn_copy2.size[_Y]; j++)
		{
			Fo(i, j)._Val[0] = Hsyn_copy2(i, j)._Val[0] / Hsyn_copy3(i, j);
			Fo(i, j)._Val[1] = Hsyn_copy2(i, j)._Val[1] / Hsyn_copy3(i, j);
		}
	}


	fftShift(Fo, Fo1);

	t = linspace(0, 1, _cfgSig.rows / 2 + 1);

	tn.resize(t.size());

	for (int i = 0; i < tn.size(); i++)
	{
		tn.at(i) = pow(t.at(i), 0.5);
	}

	Fon.resize(1, Fo.size[_X] / 2 + 1);

	for (int i = 0; i < Fo.size[_X] / 2 + 1; i++)
	{
		Fon(0, i)._Val[0] = Fo1(_cfgSig.rows / 2 - 1, _cfgSig.rows / 2 - 1 + i)._Val[0];
		Fon(0, i)._Val[1] = 0;
	}

	yn.resize(1, static_cast<int>(tn.size()));
	linInterp(t, Fon, tn, yn);
	fft1(yn, yn);
	Ab_yn.resize(yn.size[_X], yn.size[_Y]);
	absMat(yn, Ab_yn);
	Ab_yn_half.resize(1, _cfgSig.rows / 4 + 1);

	for (int i = 0; i < _cfgSig.rows / 4 + 1; i++)
	{
		Ab_yn_half(0, i) = Ab_yn(0, _cfgSig.rows / 4 + i - 1)._Val[_RE];
	}



	max = maxOfMat(Ab_yn_half);

	for (int i = 0; i < Ab_yn_half.size[1]; i++)
	{
		if (Ab_yn_half(0, i) == max)
		{
			index = i;
			break;
		}
	}

	index = -(((index + 1) - 120) / 10) / 140 + 0.1;

	return index;
}



double ophSig::sigGetParamSF(float zMax, float zMin, int sampN, float th) {
	
	int nx, ny;

	nx = ComplexH[0].size[_X];
	ny = ComplexH[0].size[_Y];

	OphComplexField I(nx, ny);
	vector<Real> F;
	Real dz = (zMax - zMin) / sampN;
	Real f;
	Real_t z = 0;
	Real depth = 0;
	Real max = MIN_DOUBLE;
	int i, j, n = 0;
	Real ret1;
	Real ret2;

	for (n = 0; n < sampN + 1; n++)
	{
		z = ((n)* dz + zMin);
		f = 0;
		I = propagationHolo(ComplexH[0], -z);

		for (i = 0; i < nx - 2; i++)
		{
			for (j = 0; j < ny - 2; j++)
			{
				ret1 = abs(I(i + 2, j)._Val[0] - I(i, j)._Val[0]);
				ret2 = abs(I(i, j + 2)._Val[0] - I(i, j)._Val[0]);
				if (ret1 >= th) { f += ret1 * ret1; }
				else if (ret2 >= th) { f += ret2 * ret2; }
			}
		}
		cout << (float)n / sampN * 100 << " %" << endl;

		if (f > max) {
			max = f;
			depth = z;
		}
	}

	return depth;
}

bool ophSig::getComplexHFromPSDH(const char * fname0, const char * fname90, const char * fname180, const char * fname270)
{
	string fname0str = fname0;
	string fname90str = fname90;
	string fname180str = fname180;
	string fname270str = fname270;
	int checktype = static_cast<int>(fname0str.rfind("."));
	OphRealField f0Mat[3], f90Mat[3], f180Mat[3], f270Mat[3];

	std::string f0type = fname0str.substr(checktype + 1, fname0str.size());

	uint8_t bitsperpixel;

	if (f0type == "bmp")
	{
		FILE *f0, *f90, *f180, *f270;
		fileheader hf;
		bitmapinfoheader hInfo;
		fopen_s(&f0, fname0str.c_str(), "rb"); fopen_s(&f90, fname90str.c_str(), "rb");
		fopen_s(&f180, fname180str.c_str(), "rb"); fopen_s(&f270, fname270str.c_str(), "rb");
		if (!f0)
		{
			LOG("bmp file open fail! (phase shift = 0)\n");
			return false;
		}
		if (!f90)
		{
			LOG("bmp file open fail! (phase shift = 90)\n");
			return false;
		}
		if (!f180)
		{
			LOG("bmp file open fail! (phase shift = 180)\n");
			return false;
		}
		if (!f270)
		{
			LOG("bmp file open fail! (phase shift = 270)\n");
			return false;
		}
		fread(&hf, sizeof(fileheader), 1, f0);
		fread(&hInfo, sizeof(bitmapinfoheader), 1, f0);

		if (hf.signature[0] != 'B' || hf.signature[1] != 'M') { LOG("Not BMP File!\n"); }
		if ((hInfo.height == 0) || (hInfo.width == 0))
		{
			LOG("bmp header is empty!\n");
			hInfo.height = _cfgSig.rows;
			hInfo.width = _cfgSig.cols;
			if (_cfgSig.rows == 0 || _cfgSig.cols == 0)
			{
				LOG("check your parameter file!\n");
				return false;
			}
		}
		if ((_cfgSig.rows != hInfo.height) || (_cfgSig.cols != hInfo.width)) {
			LOG("image size is different!\n");
			_cfgSig.rows = hInfo.height;
			_cfgSig.cols = hInfo.width;
			LOG("changed parameter of size %d x %d\n", _cfgSig.cols, _cfgSig.rows);
		}
		bitsperpixel = hInfo.bitsperpixel;
		if (hInfo.bitsperpixel == 8)
		{
			rgbquad palette[256];
			fread(palette, sizeof(rgbquad), 256, f0);
			fread(palette, sizeof(rgbquad), 256, f90);
			fread(palette, sizeof(rgbquad), 256, f180);
			fread(palette, sizeof(rgbquad), 256, f270);

			f0Mat[0].resize(hInfo.height, hInfo.width);
			f90Mat[0].resize(hInfo.height, hInfo.width);
			f180Mat[0].resize(hInfo.height, hInfo.width);
			f270Mat[0].resize(hInfo.height, hInfo.width);
			ComplexH[0].resize(hInfo.height, hInfo.width);
		}
		else
		{
			f0Mat[0].resize(hInfo.height, hInfo.width);
			f90Mat[0].resize(hInfo.height, hInfo.width);
			f180Mat[0].resize(hInfo.height, hInfo.width);
			f270Mat[0].resize(hInfo.height, hInfo.width);
			ComplexH[0].resize(hInfo.height, hInfo.width);

			f0Mat[1].resize(hInfo.height, hInfo.width);
			f90Mat[1].resize(hInfo.height, hInfo.width);
			f180Mat[1].resize(hInfo.height, hInfo.width);
			f270Mat[1].resize(hInfo.height, hInfo.width);
			ComplexH[1].resize(hInfo.height, hInfo.width);

			f0Mat[2].resize(hInfo.height, hInfo.width);
			f90Mat[2].resize(hInfo.height, hInfo.width);
			f180Mat[2].resize(hInfo.height, hInfo.width);
			f270Mat[2].resize(hInfo.height, hInfo.width);
			ComplexH[2].resize(hInfo.height, hInfo.width);
		}

		uchar* f0data = (uchar*)malloc(sizeof(uchar)*hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8));
		uchar* f90data = (uchar*)malloc(sizeof(uchar)*hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8));
		uchar* f180data = (uchar*)malloc(sizeof(uchar)*hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8));
		uchar* f270data = (uchar*)malloc(sizeof(uchar)*hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8));

		fread(f0data, sizeof(uchar), hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8), f0);
		fread(f90data, sizeof(uchar), hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8), f90);
		fread(f180data, sizeof(uchar), hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8), f180);
		fread(f270data, sizeof(uchar), hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8), f270);

		fclose(f0);
		fclose(f90);
		fclose(f180);
		fclose(f270);

		for (int i = hInfo.height - 1; i >= 0; i--)
		{
			for (int j = 0; j < static_cast<int>(hInfo.width); j++)
			{
				for (int z = 0; z < (hInfo.bitsperpixel / 8); z++)
				{
					f0Mat[z](hInfo.height - i - 1, j) = (double)f0data[i*hInfo.width*(hInfo.bitsperpixel / 8) + (hInfo.bitsperpixel / 8)*j + z];
					f90Mat[z](hInfo.height - i - 1, j) = (double)f90data[i*hInfo.width*(hInfo.bitsperpixel / 8) + (hInfo.bitsperpixel / 8)*j + z];
					f180Mat[z](hInfo.height - i - 1, j) = (double)f180data[i*hInfo.width*(hInfo.bitsperpixel / 8) + (hInfo.bitsperpixel / 8)*j + z];
					f270Mat[z](hInfo.height - i - 1, j) = (double)f270data[i*hInfo.width*(hInfo.bitsperpixel / 8) + (hInfo.bitsperpixel / 8)*j + z];
				}
			}
		}
		LOG("PSDH file load complete!\n");

		free(f0data);
		free(f90data);
		free(f180data);
		free(f270data);
		
	}
	else
	{
		LOG("wrong type (only BMP supported)\n");
	}

	// calculation complexH from 4 psdh and then normalize
	double normalizefactor = 1. / 256.;
	for (int z = 0; z < (bitsperpixel / 8); z++)
	{
		for (int i = 0; i < _cfgSig.rows; i++)
		{
			for (int j = 0; j < _cfgSig.cols; j++)
			{
				ComplexH[z](i, j)._Val[_RE] = (f0Mat[z](i, j) - f180Mat[z](i,j))*normalizefactor;
				ComplexH[z](i, j)._Val[_IM] = (f90Mat[z](i, j) - f270Mat[z](i, j))*normalizefactor;

			}
		}
	}
	LOG("complex field obtained from 4 psdh\n");
	return true;
}

void ophSig::ophFree(void) {

}