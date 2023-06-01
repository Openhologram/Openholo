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
#include "include.h"


ophSig::ophSig(void)
	//:_angleX(0)
	//, _angleY(0)
	//, _redRate(0)
	//, _radius(0)
{
	_foc = new Real_t[3];
}
void ophSig::cField2Buffer(matrix<Complex<Real>>& src, Complex<Real> **dst,int nx,int ny) {
	ivec2 bufferSize(nx, ny); //= src.getSize();

	*dst = new oph::Complex<Real>[nx * ny];

	int idx = 0;

	for (int x = 0; x < bufferSize[_X]; x++) {
		for (int y = 0; y < bufferSize[_Y]; y++) {
			(*dst)[idx] = src[x][y];
			idx++;
		}
	}
}

void ophSig::ColorField2Buffer(matrix<Complex<Real>>& src, Complex<Real> **dst, int nx, int ny) {
	ivec2 bufferSize(nx, ny); //= src.getSize();

	*dst = new oph::Complex<Real>[3*nx*ny];

	int idx = 0;
	for (int x = 0; x < bufferSize[_X]; x++) {
		for (int y = 0; y < bufferSize[_Y]; y++) {
			(*dst)[idx] = src[x][y];
			idx++;
		}
	}
}
void ophSig::setMode(bool is_CPU)
{
	this->is_CPU = is_CPU;
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
			while ((Xq[j]) >(X[i + 1])) i++;
		}
		dst(0, j)[_RE] = src(0, i).real() + (src(0, i + 1).real() - src(0, i).real()) / (X[i + 1] - X[i]) * (Xq[j] - X[i]);
		dst(0, j)[_IM] = src(0, i).imag() + (src(0, i + 1).imag() - src(0, i).imag()) / (X[i + 1] - X[i]) * (Xq[j] - X[i]);
	}
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
			dst(0, i)[_RE] = fft_out[i][_RE];
			dst(0, i)[_IM] = fft_out[i][_IM];
		}
	}
	else if (sign == OPH_BACKWARD)
	{
		for (int i = 0; i < src.size[_Y]; i++) {
			dst(0, i)[_RE] = fft_out[i][_RE] / src.size[_Y];
			dst(0, i)[_IM] = fft_out[i][_IM] / src.size[_Y];
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
				dst(i, j)[_RE] = fft_out[src.size[_Y] * i + j][_RE];
				dst(i, j)[_IM] = fft_out[src.size[_Y] * i + j][_IM];
			}
		}
	}
	else if (sign == OPH_BACKWARD)
	{
		for (int i = 0; i < src.size[_X]; i++) {
			for (int j = 0; j < src.size[_Y]; j++) {
				dst(i, j)[_RE] = fft_out[src.size[_Y] * i + j][_RE] / (src.size[_X] * src.size[_Y]);
				dst(i, j)[_IM] = fft_out[src.size[_Y] * i + j][_IM] / (src.size[_X] * src.size[_Y]);

			}
		}
	}

	fftw_destroy_plan(plan);
	fftw_free(fft_in);
	fftw_free(fft_out);
}






bool ophSig::loadAsOhc(const char *fname)
{
	std::string fullname = fname;
	if (!checkExtension(fname, ".ohc")) fullname.append(".ohc");
	OHC_decoder->setFileName(fullname.c_str());
	
	if (!OHC_decoder->load()) return false;
	vector<Real> wavelengthArray;
	OHC_decoder->getWavelength(wavelengthArray);
	_wavelength_num = OHC_decoder->getNumOfWavlen();
	int wavelength_num = OHC_decoder->getNumOfWavlen();
		
	context_.pixel_number = OHC_decoder->getNumOfPixel();

	context_.wave_length = new Real[_wavelength_num];

	ComplexH = new OphComplexField[_wavelength_num];

	for (int i = 0; i < _wavelength_num; i++)
	{
		context_.wave_length[i] = wavelengthArray[(_wavelength_num - 1) - i];

		ComplexH[i].resize(context_.pixel_number[_X], context_.pixel_number[_Y]);
		OHC_decoder->getComplexFieldData(ComplexH[i], (_wavelength_num - 1) - i);		
		//OHC_decoder->getComplexFieldData(&ComplexH);
	}
	return true;
}

bool ophSig::saveAsOhc(const char *fname)
{
	std::string fullname = fname;
	if (!checkExtension(fname, ".ohc")) fullname.append(".ohc");
	OHC_encoder->setFileName(fullname.c_str());

	ohcHeader header;

	OHC_encoder->getOHCheader(header);

	OHC_encoder->setNumOfPixel(context_.pixel_number[_X], context_.pixel_number[_Y]);

	OHC_encoder->setFieldEncoding(FldStore::Directly, FldCodeType::RI);
		
	OHC_encoder->setNumOfWavlen(_wavelength_num);
		
	for (int i = _wavelength_num - 1; i >= 0; i--)
	{
		//int wl = context_.wave_length[i] * 1000000000;
		OHC_encoder->setWavelength(context_.wave_length[i], LenUnit::nm);

		OHC_encoder->addComplexFieldData(ComplexH[i]);
	}

	if (!OHC_encoder->save()) return false;
	

	return true;
}

bool ophSig::load(const char *real, const char *imag)
{
	string realname = real;
	string imagname = imag;

	char* RGB_name[3] = { 0, };
	const char* postfix = "";

	if (_wavelength_num > 1) {
		postfix = "_B";
		strcpy(RGB_name[0], postfix);
		postfix = "_G";
		strcpy(RGB_name[1], postfix);
		postfix = "_R";
		strcpy(RGB_name[2], postfix);
	}
	else
		strcpy(RGB_name[0], postfix);

	int checktype = static_cast<int>(realname.rfind("."));

	OphRealField* realMat = new OphRealField[_wavelength_num];
	OphRealField* imagMat = new OphRealField[_wavelength_num];

	std::string realtype = realname.substr(checktype + 1, realname.size());
	std::string imgtype = imagname.substr(checktype + 1, realname.size());

	ComplexH = new OphComplexField[_wavelength_num];

	if (realtype != imgtype) {
		LOG("failed : The data type between real and imaginary is different!\n");
		return false;
	}
	if (realtype == "bmp")
	{
		realname = real;
		imagname = imag;		

		oph::uchar* realdata = loadAsImg(realname.c_str());
		oph::uchar* imagdata = loadAsImg(imagname.c_str());

		if (realdata == 0 && imagdata == 0) {
			cout << "failed : hologram data load was failed." << endl;
			return false;
		}

		for (int z = 0; z < _wavelength_num; z++)
		{
			realMat[z].resize(context_.pixel_number[_X], context_.pixel_number[_Y]);
			imagMat[z].resize(context_.pixel_number[_X], context_.pixel_number[_Y]);
			for (int i = context_.pixel_number[_X] - 1; i >= 0; i--)
			{
				for (int j = 0; j < context_.pixel_number[_Y]; j++)
				{
					realMat[z](context_.pixel_number[_X] - i - 1, j) = (double)realdata[(i * context_.pixel_number[_Y] + j)*_wavelength_num + z];
					imagMat[z](context_.pixel_number[_X] - i - 1, j) = (double)imagdata[(i * context_.pixel_number[_Y] + j)*_wavelength_num + z];
				}
			}
		}
		delete[] realdata;
		delete[] imagdata;
	}
	else if (realtype == "bin")
	{
		double *realdata = new  double[context_.pixel_number[_X] * context_.pixel_number[_Y]];
		double *imagdata = new  double[context_.pixel_number[_X] * context_.pixel_number[_Y]];

		for (int z = 0; z < _wavelength_num; z++)
		{
			realname = real;
			imagname = imag;

			realname.insert(checktype, RGB_name[z]);
			imagname.insert(checktype, RGB_name[z]);

			ifstream freal(realname.c_str(), ifstream::binary);
			ifstream fimag(imagname.c_str(), ifstream::binary);

			freal.read(reinterpret_cast<char*>(realdata), sizeof(double) * context_.pixel_number[_X] * context_.pixel_number[_Y]);
			fimag.read(reinterpret_cast<char*>(imagdata), sizeof(double) * context_.pixel_number[_X] * context_.pixel_number[_Y]);
			
			realMat[z].resize(context_.pixel_number[_X], context_.pixel_number[_Y]);
			imagMat[z].resize(context_.pixel_number[_X], context_.pixel_number[_Y]);

			for (int i = 0; i < context_.pixel_number[_X]; i++)
			{
				for (int j = 0; j < context_.pixel_number[_Y]; j++)
				{
					realMat[z](i, j) = realdata[i + j * context_.pixel_number[_X]];
					imagMat[z](i, j) = imagdata[i + j * context_.pixel_number[_X]];
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
		LOG("Error: wrong type\n");
	}
	//nomalization
	//double realout, imagout;
	for (int z = 0; z < _wavelength_num; z++)
	{
		/*meanOfMat(realMat[z], realout); meanOfMat(imagMat[z], imagout);
		
		realMat[z] / realout; imagMat[z] / imagout;
		absMat(realMat[z], realMat[z]);
		absMat(imagMat[z], imagMat[z]);
		realout = maxOfMat(realMat[z]); imagout = maxOfMat(imagMat[z]);
		realMat[z] / realout; imagMat[z] / imagout;
		realout = minOfMat(realMat[z]); imagout = minOfMat(imagMat[z]);
		realMat[z] - realout; imagMat[z] - imagout;*/

		ComplexH[z].resize(context_.pixel_number[_X], context_.pixel_number[_Y]);

		for (int i = 0; i < context_.pixel_number[_X]; i++)
		{
			for (int j = 0; j < context_.pixel_number[_Y]; j++)
			{
				ComplexH[z](i, j)[_RE] = realMat[z](i, j);
				ComplexH[z](i, j)[_IM] = imagMat[z](i, j);
			}
		}
	}
	LOG("Reading Openholo Complex Field File...%s, %s\n", realname.c_str(), imagname.c_str());

	return true;
}



bool ophSig::save(const char *real, const char *imag)
{
	string realname = real;
	string imagname = imag;

	char* RGB_name[3] = { 0, };
	const char* postfix = "";

	if (_wavelength_num > 1) {
		postfix = "_B";
		strcpy(RGB_name[0], postfix);
		postfix = "_G";
		strcpy(RGB_name[1], postfix);
		postfix = "_R";
		strcpy(RGB_name[2], postfix);
	}
	else
		strcpy(RGB_name[0], postfix);

	int checktype = static_cast<int>(realname.rfind("."));
	string type = realname.substr(checktype + 1, realname.size());
	if (type == "bin")
	{
		double *realdata = new  double[context_.pixel_number[_X] * context_.pixel_number[_Y]];
		double *imagdata = new  double[context_.pixel_number[_X] * context_.pixel_number[_Y]];

		for (int z = 0; z < _wavelength_num; z++)
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
				return false;
			}

			if (!sin.is_open()) {
				LOG("imag file not found.\n");
				sin.close();
				delete[] realdata;
				delete[] imagdata;
				return false;
			}

			for (int i = 0; i < context_.pixel_number[_X]; i++)
			{
				for (int j = 0; j < context_.pixel_number[_Y]; j++)
				{
					realdata[i + j * context_.pixel_number[_X]] = ComplexH[z](i, j)[_RE];
					imagdata[i + j * context_.pixel_number[_X]] = ComplexH[z](i, j)[_IM];
				}
			}
			cos.write(reinterpret_cast<const char*>(realdata), sizeof(double) * context_.pixel_number[_X] * context_.pixel_number[_Y]);
			sin.write(reinterpret_cast<const char*>(imagdata), sizeof(double) * context_.pixel_number[_X] * context_.pixel_number[_Y]);

			cos.close();
			sin.close();
		}
		delete[]realdata;
		delete[]imagdata;

		LOG("Writing Openholo Complex Field...%s, %s\n", realname.c_str(), imagname.c_str());
	}
	else if (type == "bmp")
	{
		oph::uchar* realdata;
		oph::uchar* imagdata;
		int _pixelbytesize = 0;
		int _width = context_.pixel_number[_Y], _height = context_.pixel_number[_X];
		int bitpixel = _wavelength_num * 8;

		if (bitpixel == 8)
		{
			_pixelbytesize = _height * _width;
		}
		else
		{
			_pixelbytesize = _height * _width * 3;
		}
		int _filesize = 0;


		FILE *freal, *fimag;
		freal = fopen(realname.c_str(), "wb");
		fimag = fopen(imagname.c_str(), "wb");

		if ((freal == nullptr) || (fimag == nullptr))
		{
			LOG("file not found\n");
			return false;
		}

		if (bitpixel == 8)
		{
			realdata = (oph::uchar*)malloc(sizeof(oph::uchar) * _width * _height);
			imagdata = (oph::uchar*)malloc(sizeof(oph::uchar) * _width * _height);
			_filesize = _pixelbytesize + sizeof(bitmap8bit);

			bitmap8bit *pbitmap = (bitmap8bit*)calloc(1, sizeof(bitmap8bit));
			memset(pbitmap, 0x00, sizeof(bitmap8bit));

			pbitmap->_fileheader.signature[0] = 'B';
			pbitmap->_fileheader.signature[1] = 'M';
			pbitmap->_fileheader.filesize = _filesize;
			pbitmap->_fileheader.fileoffset_to_pixelarray = sizeof(bitmap8bit);

			for (int i = 0; i < 256; i++) {
				pbitmap->_rgbquad[i].rgbBlue = i;
				pbitmap->_rgbquad[i].rgbGreen = i;
				pbitmap->_rgbquad[i].rgbRed = i;
			}


			//// denormalization
			for (int i = _height - 1; i >= 0; i--)
			{
				for (int j = 0; j < _width; j++)
				{
					if (ComplexH[0].mat[_height - i - 1][j][_RE] < 0)
					{
						ComplexH[0].mat[_height - i - 1][j][_RE] = 0;
					}

					if (ComplexH[0].mat[_height - i - 1][j][_IM] < 0)
					{
						ComplexH[0].mat[_height - i - 1][j][_IM] = 0;
					}
				}
			}

			double minVal, iminVal, maxVal, imaxVal;
			maxVal = minVal = ComplexH[0](0, 0)[_RE];
			imaxVal = iminVal = ComplexH[0](0, 0)[_IM];

			for (int j = 0; j < ComplexH[0].size[_Y]; j++) {
				for (int i = 0; i < ComplexH[0].size[_X]; i++) {
					if (ComplexH[0](i, j)[_RE] < minVal) minVal = ComplexH[0](i, j).real();
					if (ComplexH[0](i, j)[_RE] > maxVal) maxVal = ComplexH[0](i, j).real();
					if (ComplexH[0](i, j)[_IM] < iminVal) iminVal = ComplexH[0](i, j)[_IM];
					if (ComplexH[0](i, j)[_IM] > imaxVal) imaxVal = ComplexH[0](i, j)[_IM];
				}
			}
			for (int i = _height - 1; i >= 0; i--)
			{
				for (int j = 0; j < _width; j++)
				{
					realdata[i*_width + j] = (uchar)((ComplexH[0](_height - i - 1, j)[_RE] - minVal) / (maxVal - minVal) * 255 + 0.5);
					imagdata[i*_width + j] = (uchar)((ComplexH[0](_height - i - 1, j)[_IM] - iminVal) / (imaxVal - iminVal) * 255 + 0.5);
				}
			}

			pbitmap->_bitmapinfoheader.dibheadersize = sizeof(bitmapinfoheader);
			pbitmap->_bitmapinfoheader.width = _width;
			pbitmap->_bitmapinfoheader.height = _height;
			pbitmap->_bitmapinfoheader.planes = OPH_PLANES;
			pbitmap->_bitmapinfoheader.bitsperpixel = bitpixel;
			pbitmap->_bitmapinfoheader.compression = OPH_COMPRESSION;
			pbitmap->_bitmapinfoheader.imagesize = _pixelbytesize;
			pbitmap->_bitmapinfoheader.ypixelpermeter = 0;
			pbitmap->_bitmapinfoheader.xpixelpermeter = 0;
			pbitmap->_bitmapinfoheader.numcolorspallette = 256;

			fwrite(pbitmap, 1, sizeof(bitmap8bit), freal);
			fwrite(realdata, 1, _pixelbytesize, freal);

			fwrite(pbitmap, 1, sizeof(bitmap8bit), fimag);
			fwrite(imagdata, 1, _pixelbytesize, fimag);

			fclose(freal);
			fclose(fimag);
			free(pbitmap);
		}
		else
		{
			realdata = (oph::uchar*)malloc(sizeof(oph::uchar) * _width * _height * bitpixel / 3);
			imagdata = (oph::uchar*)malloc(sizeof(oph::uchar) * _width * _height * bitpixel / 3);
			_filesize = _pixelbytesize + sizeof(fileheader) + sizeof(bitmapinfoheader);

			fileheader *hf = (fileheader*)calloc(1, sizeof(fileheader));
			bitmapinfoheader *hInfo = (bitmapinfoheader*)calloc(1, sizeof(bitmapinfoheader));

			hf->signature[0] = 'B';
			hf->signature[1] = 'M';
			hf->filesize = _filesize;
			hf->fileoffset_to_pixelarray = sizeof(fileheader) + sizeof(bitmapinfoheader);

			double minVal, iminVal, maxVal, imaxVal;

			for (int z = 0; z < 3; z++)
			{
				maxVal = minVal = ComplexH[z](0, 0)[_RE];
				imaxVal = iminVal = ComplexH[z](0, 0)[_IM];

				for (int j = 0; j < ComplexH[0].size[_Y]; j++) {
					for (int i = 0; i < ComplexH[0].size[_X]; i++) {
						if (ComplexH[z](i, j)[_RE] < minVal)
							minVal = ComplexH[z](i, j)[_RE];
						if (ComplexH[z](i, j)[_RE] > maxVal)
							maxVal = ComplexH[z](i, j)[_RE];
						if (ComplexH[z](i, j)[_IM] < iminVal)
							iminVal = ComplexH[z](i, j)[_IM];
						if (ComplexH[z](i, j)[_IM] > imaxVal)
							imaxVal = ComplexH[z](i, j)[_IM];
					}
				}

				for (int i = _height - 1; i >= 0; i--)
				{
					for (int j = 0; j < _width; j++)
					{
						realdata[3 * j + 3 * i * _width + z] = (uchar)((ComplexH[z](_height - i - 1, j)[_RE] - minVal) / (maxVal - minVal) * 255);
						imagdata[3 * j + 3 * i * _width + z] = (uchar)((ComplexH[z](_height - i - 1, j)[_IM] - iminVal) / (imaxVal - iminVal) * 255);

					}
				}
			}
			hInfo->dibheadersize = sizeof(bitmapinfoheader);
			hInfo->width = _width;
			hInfo->height = _height;
			hInfo->planes = OPH_PLANES;
			hInfo->bitsperpixel = bitpixel;
			hInfo->compression = OPH_COMPRESSION;
			hInfo->imagesize = _pixelbytesize;
			hInfo->ypixelpermeter = 0;
			hInfo->xpixelpermeter = 0;

			fwrite(hf, 1, sizeof(fileheader), freal);
			fwrite(hInfo, 1, sizeof(bitmapinfoheader), freal);
			fwrite(realdata, 1, _pixelbytesize, freal);

			fwrite(hf, 1, sizeof(fileheader), fimag);
			fwrite(hInfo, 1, sizeof(bitmapinfoheader), fimag);
			fwrite(imagdata, 1, _pixelbytesize, fimag);

			fclose(freal);
			fclose(fimag);
			free(hf);
			free(hInfo);
		}

		free(realdata);
		free(imagdata);
		std::cout << "file save bmp complete\n" << endl;

	}
	else {
		LOG("failed : The Invalid data type! - %s\n", type.c_str());
	}
	return true;
}

bool ophSig::save(const char *fname)
{
	string fullname = fname;

	char* RGB_name[3] = { 0, };
	const char* postfix = "";

	if (_wavelength_num > 1) {
		postfix = "_B";
		strcpy(RGB_name[0], postfix);
		postfix = "_G";
		strcpy(RGB_name[1], postfix);
		postfix = "_R";
		strcpy(RGB_name[2], postfix);
	}
	else
		strcpy(RGB_name[0], postfix);

	int checktype = static_cast<int>(fullname.rfind("."));

	if (fullname.substr(checktype + 1, fullname.size()) == "bmp")
	{
		oph::uchar* realdata;
		realdata = (oph::uchar*)malloc(sizeof(oph::uchar) * context_.pixel_number[_X] * context_.pixel_number[_Y] * _wavelength_num);

		double gamma = 0.5;
		double maxIntensity = 0.0;
		double realVal = 0.0;
		double imagVal = 0.0;
		double intensityVal = 0.0;

		for (int z = 0; z < _wavelength_num; z++)
		{
			for (int j = 0; j < context_.pixel_number[_Y]; j++) {
				for (int i = 0; i < context_.pixel_number[_X]; i++) {
					realVal = ComplexH[z](i, j)[_RE];
					imagVal = ComplexH[z](i, j)[_RE];
					intensityVal = realVal*realVal + imagVal*imagVal;
					if (intensityVal > maxIntensity) {
						maxIntensity = intensityVal;
					}
				}
			}
			for (int i = context_.pixel_number[_X] - 1; i >= 0; i--)
			{
				for (int j = 0; j < context_.pixel_number[_Y]; j++)
				{
					realVal = ComplexH[z](context_.pixel_number[_X] - i - 1, j)[_RE];
					imagVal = ComplexH[z](context_.pixel_number[_X] - i - 1, j)[_IM];
					intensityVal = realVal*realVal + imagVal*imagVal;
					realdata[(i*context_.pixel_number[_Y] + j)* _wavelength_num + z] = (uchar)(pow(intensityVal / maxIntensity, gamma)*255.0);
				}
			}
			//sprintf(str, "_%.2u", z);
			//realname.insert(checktype, RGB_name[z]);
		}
		saveAsImg(fullname.c_str(), _wavelength_num * 8, realdata, context_.pixel_number[_X], context_.pixel_number[_Y]);

		delete[] realdata;
	}
	else if (fullname.substr(checktype + 1, fullname.size()) == "bin")
	{
		double *realdata = new  double[context_.pixel_number[_X] * context_.pixel_number[_Y]];

		for (int z = 0; z < _wavelength_num; z++)
		{
			fullname = fname;
			fullname.insert(checktype, RGB_name[z]);
			std::ofstream cos(fullname.c_str(), std::ios::binary);

			if (!cos.is_open()) {
				LOG("Error: file name not found.\n");
				cos.close();
				delete[] realdata;
				return false;
			}

			for (int i = 0; i < context_.pixel_number[_X]; i++)
			{
				for (int j = 0; j < context_.pixel_number[_Y]; j++)
				{
					realdata[context_.pixel_number[_Y] * i + j] = ComplexH[z](i, j)[_RE];
				}
			}
			cos.write(reinterpret_cast<const char*>(realdata), sizeof(double) * context_.pixel_number[_X] * context_.pixel_number[_Y]);

			cos.close();
		}
		delete[] realdata;
	}

	LOG("Writing Openholo Complex Field...%s\n", fullname.c_str());
	return true;
}

void ophSig::Data_output(uchar *data, int pos, int bitpixel)
{

	int _width = context_.pixel_number[_Y], _height = context_.pixel_number[_X];
	OphComplexField* abs_data;
	abs_data = new OphComplexField[1];
	abs_data->resize(context_.pixel_number[_Y], _height = context_.pixel_number[_X]);
	if(pos == 0)
	{ 
	if (bitpixel == 8)
	{
	
		absMat(ComplexH[0], *abs_data);
		for (int i = _height - 1; i >= 0; i--)
		{
			for (int j = 0; j < _width; j++)
			{
				if (abs_data[0].mat[_height - i - 1][j][_RE] < 0)
				{
					abs_data[0].mat[_height - i - 1][j][_RE] = 0;
				}
			}
		}

		double minVal, iminVal, maxVal, imaxVal;
		for (int j = 0; j < abs_data[0].size[_Y]; j++) {
			for (int i = 0; i < abs_data[0].size[_X]; i++) {
				if ((i == 0) && (j == 0))
				{
					minVal = abs_data[0](i, j)[_RE];
					maxVal = abs_data[0](i, j)[_RE];
				}
				else {
					if (abs_data[0](i, j)[_RE] < minVal)
					{
						minVal = abs_data[0](i, j).real();
					}
					if (abs_data[0](i, j)[_RE] > maxVal)
					{
						maxVal = abs_data[0](i, j).real();
					}
				}
			}
		}
		for (int i = _height - 1; i >= 0; i--)
		{
			for (int j = 0; j < _width; j++)
			{
				data[i*_width + j] = (uchar)((abs_data[0](_height - i - 1, j)[_RE] - minVal) / (maxVal - minVal) * 255 + 0.5);
			}
		}
	}

	else
	{
		double minVal, iminVal, maxVal, imaxVal;
		for (int z = 0; z < 3; z++)
		{
			absMat(ComplexH[z], abs_data[0]);
			for (int j = 0; j < abs_data[0].size[_Y]; j++) {
				for (int i = 0; i < abs_data[0].size[_X]; i++) {
					if ((i == 0) && (j == 0))
					{
						minVal = abs_data[0](i, j)[_RE];
						maxVal = abs_data[0](i, j)[_RE];
					}
					else {
						if (abs_data[0](i, j)[_RE] < minVal)
						{
							minVal = abs_data[0](i, j)[_RE];
						}
						if (abs_data[0](i, j)[_RE] > maxVal)
						{
							maxVal = abs_data[0](i, j)[_RE];
						}
					}
				}
			}

			for (int i = _height - 1; i >= 0; i--)
			{
				for (int j = 0; j < _width; j++)
				{
					data[3 * j + 3 * i * _width + z] = (uchar)((abs_data[0](_height - i - 1, j)[_RE] - minVal) / (maxVal - minVal) * 255);
				}
			}
		}
	}
	}

	else if (pos == 2)
	{
		if (bitpixel == 8)
		{
			for (int i = _height - 1; i >= 0; i--)
			{
				for (int j = 0; j < _width; j++)
				{
					if (ComplexH[0].mat[_height - i - 1][j][_RE] < 0)
					{
						ComplexH[0].mat[_height - i - 1][j][_RE] = 0;
					}
				}
			}

			double minVal, iminVal, maxVal, imaxVal;
			for (int j = 0; j < ComplexH[0].size[_Y]; j++) {
				for (int i = 0; i < ComplexH[0].size[_X]; i++) {
					if ((i == 0) && (j == 0))
					{
						minVal = ComplexH[0](i, j)[_RE];
						maxVal = ComplexH[0](i, j)[_RE];
					}
					else {
						if (ComplexH[0](i, j)[_RE] < minVal)
						{
							minVal = ComplexH[0](i, j).real();
						}
						if (ComplexH[0](i, j)[_RE] > maxVal)
						{
							maxVal = ComplexH[0](i, j).real();
						}
					}
				}
			}
			for (int i = _height - 1; i >= 0; i--)
			{
				for (int j = 0; j < _width; j++)
				{
					data[i*_width + j] = (uchar)((ComplexH[0](_height - i - 1, j)[_RE] - minVal) / (maxVal - minVal) * 255 + 0.5);
				}
			}
		}

		else
		{
			double minVal, iminVal, maxVal, imaxVal;
			for (int z = 0; z < 3; z++)
			{
				for (int j = 0; j < ComplexH[0].size[_Y]; j++) {
					for (int i = 0; i < ComplexH[0].size[_X]; i++) {
						if ((i == 0) && (j == 0))
						{
							minVal = ComplexH[z](i, j)[_RE];
							maxVal = ComplexH[z](i, j)[_RE];
						}
						else {
							if (ComplexH[z](i, j)[_RE] < minVal)
							{
								minVal = ComplexH[z](i, j)[_RE];
							}
							if (ComplexH[z](i, j)[_RE] > maxVal)
							{
								maxVal = ComplexH[z](i, j)[_RE];
							}
						}
					}
				}

				for (int i = _height - 1; i >= 0; i--)
				{
					for (int j = 0; j < _width; j++)
					{
						data[3 * j + 3 * i * _width + z] = (uchar)((ComplexH[z](_height - i - 1, j)[_RE] - minVal) / (maxVal - minVal) * 255);
					}
				}
			}
		}
	}

	else if (pos == 1)
	{
		if (bitpixel == 8)
		{
			for (int i = _height - 1; i >= 0; i--)
			{
				for (int j = 0; j < _width; j++)
				{
					if (ComplexH[0].mat[_height - i - 1][j][_IM] < 0)
					{
						ComplexH[0].mat[_height - i - 1][j][_IM] = 0;
					}
				}
			}

			double minVal, iminVal, maxVal, imaxVal;
			for (int j = 0; j < ComplexH[0].size[_Y]; j++) {
				for (int i = 0; i < ComplexH[0].size[_X]; i++) {
					if ((i == 0) && (j == 0))
					{
						minVal = ComplexH[0](i, j)[_IM];
						maxVal = ComplexH[0](i, j)[_IM];
					}
					else {
						if (ComplexH[0](i, j)[_IM] < minVal)
						{
							minVal = ComplexH[0](i, j).imag();
						}
						if (ComplexH[0](i, j)[_IM] > maxVal)
						{
							maxVal = ComplexH[0](i, j).imag();
						}
					}
				}
			}
			for (int i = _height - 1; i >= 0; i--)
			{
				for (int j = 0; j < _width; j++)
				{
					data[i*_width + j] = (uchar)((ComplexH[0](_height - i - 1, j)[_IM] - minVal) / (maxVal - minVal) * 255 + 0.5);
				}
			}
		}

		else
		{
			double minVal, iminVal, maxVal, imaxVal;
			for (int z = 0; z < 3; z++)
			{
				for (int j = 0; j < ComplexH[0].size[_Y]; j++) {
					for (int i = 0; i < ComplexH[0].size[_X]; i++) {
						if ((i == 0) && (j == 0))
						{
							minVal = ComplexH[z](i, j)[_IM];
							maxVal = ComplexH[z](i, j)[_IM];
						}
						else {
							if (ComplexH[z](i, j)[_IM] < minVal)
							{
								minVal = ComplexH[z](i, j)[_IM];
							}
							if (ComplexH[z](i, j)[_IM] > maxVal)
							{
								maxVal = ComplexH[z](i, j)[_IM];
							}
						}
					}
				}

				for (int i = _height - 1; i >= 0; i--)
				{
					for (int j = 0; j < _width; j++)
					{
						data[3 * j + 3 * i * _width + z] = (uchar)((ComplexH[z](_height - i - 1, j)[_IM] - minVal) / (maxVal - minVal) * 255);
					}
				}
			}
		}
	}
}


bool ophSig::sigConvertOffaxis(Real angleX, Real angleY) {
	auto start_time = CUR_TIME;
	
	if (is_CPU == true)
	{
		std::cout << "Start Single Core CPU" << endl;
		cvtOffaxis_CPU(angleX,angleY);
	}
	else {
		std::cout << "Start Multi Core GPU" << std::endl;
		cvtOffaxis_GPU(angleX, angleY);
	}
	auto end_time = CUR_TIME;
	auto during_time = ((std::chrono::duration<Real>)(end_time - start_time)).count();
	LOG("Implement time : %.5lf sec\n", during_time);
	return true;
}

bool ophSig::sigConvertHPO(Real depth, Real_t redRate) {
	auto start_time = CUR_TIME;
	if (is_CPU == true)
	{
		std::cout << "Start Single Core CPU" << endl;
		sigConvertHPO_CPU(depth,redRate);
	
	}
	else {
		std::cout << "Start Multi Core GPU" << std::endl;
		
		sigConvertHPO_GPU(depth, redRate);
		
	}
	
	auto end_time = CUR_TIME;
	auto during_time = ((std::chrono::duration<Real>)(end_time - start_time)).count();
	LOG("Implement time : %.5lf sec\n", during_time);
	return true;
}

bool ophSig::sigConvertCAC(double red, double green, double blue){
	auto start_time = CUR_TIME;
	if (is_CPU == true)
	{
		std::cout << "Start Single Core CPU" << endl;
		sigConvertCAC_CPU(red, green, blue);

	}
	else {
		std::cout << "Start Multi Core GPU" << std::endl;
		sigConvertCAC_GPU(red, green, blue);
	
	}
	auto end_time = CUR_TIME;
	auto during_time = ((std::chrono::duration<Real>)(end_time - start_time)).count();
	LOG("Implement time : %.5lf sec\n", during_time);
	return true;
}

bool ophSig::propagationHolo(float depth) {
	auto start_time = CUR_TIME;
	if (is_CPU == true)
	{
		std::cout << "Start Single Core CPU" << endl;
		propagationHolo_CPU(depth);

	}
	else {
		std::cout << "Start Multi Core GPU" << std::endl;
		if (_wavelength_num == 1)
		{
			propagationHolo_GPU(depth);
		}
		else if (_wavelength_num == 3)
		{
			Color_propagationHolo_GPU(depth);
		}

	}
	auto end_time = CUR_TIME;
	auto during_time = ((std::chrono::duration<Real>)(end_time - start_time)).count();
	LOG("Implement time : %.5lf sec\n", during_time);
	return true;
}

double ophSig::sigGetParamSF(float zMax, float zMin, int sampN, float th) {
	auto start_time = CUR_TIME;
	double out = 0;
	if (is_CPU == true)
	{
		std::cout << "Start Single Core CPU" << endl;
		out = sigGetParamSF_CPU(zMax,zMin,sampN,th);

	}
	else {
		std::cout << "Start Multi Core GPU" << std::endl;
		out = sigGetParamSF_GPU(zMax, zMin, sampN, th);

	}
	auto end_time = CUR_TIME;
	auto during_time = ((std::chrono::duration<Real>)(end_time - start_time)).count();
	LOG("Implement time : %.5lf sec\n", during_time);
	return out;
}


bool ophSig::cvtOffaxis_CPU(Real angleX, Real angleY) {
	
	int nx = context_.pixel_number[_X];
	int ny = context_.pixel_number[_Y];
	OphRealField H1(nx,ny);
	Real x, y;
	Complex<Real> F;
	H1.resize(nx, ny);

	for (int i = 0; i < nx; i++)
	{
		y = (_cfgSig.height / (nx - 1)*i - _cfgSig.height / 2);
		for (int j = 0; j < ny; j++)
		{
			x = (_cfgSig.width / (ny - 1)*j - _cfgSig.width / 2);

			//(*ComplexH)(i, j)[_RE] = cos(((2 * M_PI) / *context_.wave_length)*(x*sin(angle[_X]) + y *sin(angle[_Y])));
			//(*ComplexH)(i, j)[_IM] = sin(((2 * M_PI) / *context_.wave_length)*(x*sin(angle[_X]) + y *sin(angle[_Y])));
			F[_RE] = cos(((2 * M_PI) / *context_.wave_length)*(x*sin(angleX) + y *sin(angleY)));
			F[_IM] = sin(((2 * M_PI) / *context_.wave_length)*(x*sin(angleX) + y *sin(angleY)));
			H1(i, j) = ((*ComplexH)(i, j) * F)[_RE];
		}
	}
	double out = minOfMat(H1);
	H1 - out;
	out = maxOfMat(H1);
	H1 / out;
	//normalizeMat(H1, H1);


	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			(*ComplexH)(i, j)[_RE] = H1(i, j);
			(*ComplexH)(i, j)[_IM] = 0;
		}
	}

	

	return true;
}

bool ophSig::sigConvertHPO_CPU(Real depth, Real_t redRate) {

	
	int nx = context_.pixel_number[_X];
	int ny = context_.pixel_number[_Y];

	Real wl = *context_.wave_length;
	Real NA = _cfgSig.width/(2*depth);

	int xshift = nx / 2;
	int yshift = ny / 2;

	Real  y;

	Real_t NA_g = NA * redRate;

	OphComplexField F1(nx, ny);
	OphComplexField FH(nx, ny);


	Real Rephase = -(1 / (4 * M_PI)*pow((wl / NA_g), 2));
	Real Imphase = ((1 / (4 * M_PI))*depth*wl);

	for (int i = 0; i < ny; i++)
	{
		int ii = (i + yshift) % ny;
		
		for (int j = 0; j < nx; j++)
		{
			y = (2 * M_PI * (j) / _cfgSig.height - M_PI * (nx - 1) / _cfgSig.height);
			int jj = (j + xshift) % nx;
			F1(jj, ii)[_RE] = std::exp(Rephase*pow(y, 2))*cos(Imphase*pow(y, 2));
			F1(jj, ii)[_IM] = std::exp(Rephase*pow(y, 2))*sin(Imphase*pow(y, 2));
		}
	}
	fft2((*ComplexH), FH, OPH_FORWARD);
	F1.mulElem(FH);
	fft2(F1, (*ComplexH), OPH_BACKWARD);

	


	return true;

}

bool ophSig::sigConvertCAC_CPU(double red, double green, double blue) {
	
	Real x, y;
	//OphComplexField  exp, conj, FH_CAC;
	int nx = context_.pixel_number[_X];
	int ny = context_.pixel_number[_Y];

	if (_wavelength_num != 3) {
		_wavelength_num = 3;
		delete[] context_.wave_length;
		context_.wave_length = new Real[_wavelength_num];
	}

	context_.wave_length[0] = blue;
	context_.wave_length[1] = green;
	context_.wave_length[2] = red;
	
	OphComplexField FFZP(nx, ny);
	OphComplexField FH(nx, ny);
	
	for (int z = 0; z < _wavelength_num; z++)
	{
		double sigmaf = ((_foc[2] - _foc[z]) * context_.wave_length[z]) / (4 * M_PI);
		int xshift = nx / 2;
		int yshift = ny / 2;

		for (int i = 0; i < ny; i++)
		{
			int ii = (i + yshift) % ny;
			y = (2 * M_PI * i) / _radius - (M_PI*(ny - 1)) / _radius;
			for (int j = 0; j < nx; j++)
			{
				x = (2 * M_PI * j) / _radius - (M_PI*(nx - 1)) / _radius;
				
				int jj = (j + xshift) % nx;
			
				FFZP(jj, ii)[_RE] = cos(sigmaf * (pow(x, 2) + pow(y, 2)));
				FFZP(jj, ii)[_IM] = -sin(sigmaf * (pow(x, 2) + pow(y, 2))); //conjugate  때문에 -붙음여
			}
		}
		fft2(ComplexH[z], FH, OPH_FORWARD);
		FH.mulElem(FFZP);
		fft2(FH, ComplexH[z], OPH_BACKWARD);
	}

	return true;
}
void ophSig::Parameter_Set(int nx, int ny, double width, double height,  double NA)
{
	context_.pixel_number[_X] = nx;
	context_.pixel_number[_Y] = ny;
	_cfgSig.width = width;
	_cfgSig.height = height;
	_cfgSig.NA = NA;
}

void ophSig::wavelength_Set(double wavelength)
{
	*context_.wave_length = wavelength;
}

void ophSig::focal_length_Set(double red, double green, double blue,double rad)
{
	_foc[2] = red;
	_foc[1] = green;
	_foc[0] = blue;
	_radius = rad;
}

void ophSig::Wavenumber_output(int &wavenumber)
{
	wavenumber = _wavelength_num;
}

bool ophSig::readConfig(const char* fname)
{
	//LOG("Reading....%s...\n", fname);

	/*XML parsing*/
	tinyxml2::XMLDocument xml_doc;
	tinyxml2::XMLNode* xml_node;

	if (!checkExtension(fname, ".xml"))
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

	(xml_node->FirstChildElement("pixel_number_x"))->QueryIntText(&context_.pixel_number[_X]);
	(xml_node->FirstChildElement("pixel_number_y"))->QueryIntText(&context_.pixel_number[_Y]);
	(xml_node->FirstChildElement("width"))->QueryFloatText(&_cfgSig.width);
	(xml_node->FirstChildElement("height"))->QueryFloatText(&_cfgSig.height);
	(xml_node->FirstChildElement("wavelength_num"))->QueryIntText(&_wavelength_num);

	context_.wave_length = new Real[_wavelength_num];

	(xml_node->FirstChildElement("wavelength"))->QueryDoubleText(context_.wave_length);
	(xml_node->FirstChildElement("NA"))->QueryFloatText(&_cfgSig.NA);
	(xml_node->FirstChildElement("z"))->QueryFloatText(&_cfgSig.z);
	(xml_node->FirstChildElement("radius_of_lens"))->QueryFloatText(&_radius);
	(xml_node->FirstChildElement("focal_length_R"))->QueryFloatText(&_foc[2]);
	(xml_node->FirstChildElement("focal_length_G"))->QueryFloatText(&_foc[1]);
	(xml_node->FirstChildElement("focal_length_B"))->QueryFloatText(&_foc[0]);

	return true;
}



bool ophSig::propagationHolo_CPU(float depth) {
	int i, j;
	Real x, y, sigmaf;
	int nx = context_.pixel_number[_X];
	int ny = context_.pixel_number[_Y];

	OphComplexField FH(nx, ny);
	//OphComplexField FFZP(nx, ny);
	int xshift = nx / 2;
	int yshift = ny / 2;

	for (int z = 0; z < _wavelength_num; z++) {

		sigmaf = (depth * context_.wave_length[z]) / (4 * M_PI);

		/*FH.resize(nx, ny);
		FFZP.resize(nx, ny);*/

		fft2(ComplexH[z], FH, OPH_FORWARD);

		for (i = 0; i < ny; i++)
		{
			int ii = (i + yshift) % ny;
			y = (2 * M_PI * (i)) / _cfgSig.width - (M_PI*(ny - 1)) / (_cfgSig.width);
			
			for (j = 0; j < nx; j++)
			{
				x = (2 * M_PI * (j)) / _cfgSig.height - (M_PI*(nx - 1)) / (_cfgSig.height);
				int jj = (j + xshift) % nx;
				double temp = FH(jj, ii)[_RE];
				FH(jj, ii)[_RE] = cos(sigmaf * (pow(x, 2) + pow(y, 2))) * FH(jj, ii)[_RE] - sin(sigmaf * (pow(x, 2) + pow(y, 2))) * FH(jj, ii)[_IM];
				FH(jj, ii)[_IM] = sin(sigmaf * (pow(x, 2) + pow(y, 2))) * temp + cos(sigmaf * (pow(x, 2) + pow(y, 2))) * FH(jj, ii)[_IM];
			
			}
		}

		fft2(FH, ComplexH[z], OPH_BACKWARD);
	}
	return true;
}

OphComplexField ophSig::propagationHolo(OphComplexField complexH, float depth) {
	int i, j;
	Real x, y, sigmaf;
	int nx = context_.pixel_number[_X];
	int ny = context_.pixel_number[_Y];

	OphComplexField FH(nx, ny);
	int xshift = nx / 2;
	int yshift = ny / 2;


	sigmaf = (depth * (*context_.wave_length)) / (4 * M_PI);

	fft2(complexH, FH);

	for (i = 0; i < ny; i++)
	{
		int ii = (i + yshift) % ny;
		y = (2 * M_PI * (i)) / _cfgSig.width - (M_PI*(ny - 1)) / (_cfgSig.width);

		for (j = 0; j < nx; j++)
		{
			x = (2 * M_PI * (j)) / _cfgSig.height - (M_PI*(nx - 1)) / (_cfgSig.height);
			int jj = (j + xshift) % nx;
			double temp = FH(jj, ii)[_RE];
			FH(jj, ii)[_RE] = cos(sigmaf * (pow(x, 2) + pow(y, 2))) * FH(jj, ii)[_RE] - sin(sigmaf * (pow(x, 2) + pow(y, 2))) * FH(jj, ii)[_IM];
			FH(jj, ii)[_IM] = sin(sigmaf * (pow(x, 2) + pow(y, 2))) * temp + cos(sigmaf * (pow(x, 2) + pow(y, 2))) * FH(jj, ii)[_IM];

		}
	}
	fft2(FH, complexH, OPH_BACKWARD);

	return complexH;
}

double ophSig::sigGetParamAT()
{
	//auto start_time = CUR_TIME;

	Real index = 0;
	if (is_CPU == true)
	{
		//std::cout << "Start Single Core CPU" << endl;
		index = sigGetParamAT_CPU();

	}
	else {
		//std::cout << "Start Multi Core GPU" << std::endl;
		index = sigGetParamAT_GPU();
	
	}

	//auto end_time = CUR_TIME;

	//auto during_time = ((std::chrono::duration<Real>)(end_time - start_time)).count();

	//LOG("Implement time : %.5lf sec\n", during_time);
	return index;
}

double ophSig::sigGetParamAT_CPU() {
	
	int i = 0, j = 0;
	Real max = 0;	Real index = 0;
	Real_t NA_g = (Real_t)0.025;
	int nx = context_.pixel_number[_X];
	int ny = context_.pixel_number[_Y];

	OphComplexField Flr(nx, ny);
	OphComplexField Fli(nx, ny);
	OphComplexField Hsyn(nx, ny);

	OphComplexField Fo(nx, ny);
	OphComplexField Fon, yn, Ab_yn;

	OphRealField Ab_yn_half;
	OphRealField G(nx, ny);
	Real x = 1, y = 1;
	vector<Real> t, tn;

	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{

			x = (2 * M_PI*(i) / _cfgSig.height - M_PI*(nx - 1) / _cfgSig.height);
			y = (2 * M_PI*(j) / _cfgSig.width - M_PI*(ny - 1) / _cfgSig.width);
			G(i, j) = std::exp(-M_PI * pow((*context_.wave_length) / (2 * M_PI * NA_g), 2) * (pow(y, 2) + pow(x, 2)));
			Flr(i, j)[_RE] = (*ComplexH)(i, j)[_RE];
			Fli(i, j)[_RE] = (*ComplexH)(i, j)[_IM];
			Flr(i, j)[_IM] = 0;
			Fli(i, j)[_IM] = 0;
		}
	}

	fft2(Flr, Flr);
	fft2(Fli, Fli);

	int xshift = nx / 2;
	int yshift = ny / 2;

	for (i = 0; i < nx; i++)
	{
		int ii = (i + xshift) % nx;
		for (j = 0; j < ny; j++)
		{
			int jj = (j + yshift) % ny;
			Hsyn(i, j)[_RE] = Flr(i, j)[_RE] * G(i, j);
			Hsyn(i, j)[_IM] = Fli(i, j)[_RE] * G(i, j);
			/*Hsyn_copy1(i, j) = Hsyn(i, j);
			Hsyn_copy2(i, j) = Hsyn_copy1(i, j) * Hsyn(i, j);
			Hsyn_copy3(i, j) = pow(sqrt(Hsyn(i, j)[_RE] * Hsyn(i, j)[_RE] + Hsyn(i, j)[_IM] * Hsyn(i, j)[_IM]), 2) + pow(10, -300);
			Fo(ii, jj)[_RE] = Hsyn_copy2(i, j)[0] / Hsyn_copy3(i, j);
			Fo(ii, jj)[_IM] = Hsyn_copy2(i, j)[1] / Hsyn_copy3(i, j);*/
			Fo(ii, jj) = pow(Hsyn(i, j), 2) / (pow(abs(Hsyn(i, j)), 2) + pow(10, -300));

		}
	}

	t = linspace(0., 1., nx / 2 + 1);
	tn.resize(t.size());
	Fon.resize(1, t.size());

	int size = (int)tn.size();
	for (int i = 0; i < size; i++)
	{
		tn.at(i) = pow(t.at(i), 0.5);
		Fon(0, i)[_RE] = Fo(nx / 2 - 1, nx / 2 - 1 + i)[_RE];
		Fon(0, i)[_IM] = 0;
	}
	yn.resize(1, size);
	linInterp(t, Fon, tn, yn);
	fft1(yn, yn);
	Ab_yn.resize(yn.size[_X], yn.size[_Y]);
	absMat(yn, Ab_yn);
	Ab_yn_half.resize(1, nx / 4 + 1);

	for (int i = 0; i < nx / 4 + 1; i++)
	{
		Ab_yn_half(0, i) = Ab_yn(0, nx / 4 + i - 1)[_RE];
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

double ophSig::sigGetParamSF_CPU(float zMax, float zMin, int sampN, float th)
{	
	int nx = context_.pixel_number[_X];
	int ny = context_.pixel_number[_Y];

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
		OphComplexField field = *(ComplexH);
		z = ((n)* dz + zMin);
		f = 0;
		I = propagationHolo(field, z);

		for (i = 0; i < nx - 2; i++)
		{
			for (j = 0; j < ny - 2; j++)
			{
				ret1 = abs(I(i + 2, j)[_RE] - I(i, j)[_RE]);
				ret2 = abs(I(i, j + 2)[_RE] - I(i, j)[_RE]);
				if (ret1 >= th) { f += ret1 * ret1; }
				else if (ret2 >= th) { f += ret2 * ret2; }
			}
		}

		if (f > max) {
			max = f;
			depth = z;
		}
	}


	return depth;
}

bool ophSig::getComplexHFromPSDH(const char * fname0, const char * fname90, const char * fname180, const char * fname270)
{
	auto start_time = CUR_TIME;
	string fname0str = fname0;
	string fname90str = fname90;
	string fname180str = fname180;
	string fname270str = fname270;
	int checktype = static_cast<int>(fname0str.rfind("."));
	OphRealField f0Mat[3], f90Mat[3], f180Mat[3], f270Mat[3];

	std::string f0type = fname0str.substr(checktype + 1, fname0str.size());

	uint16_t bitsperpixel;
	fileheader hf;
	bitmapinfoheader hInfo;

	if (f0type == "bmp")
	{
		FILE *f0, *f90, *f180, *f270;
		f0 = fopen(fname0str.c_str(), "rb"); 
		f90 = fopen(fname90str.c_str(), "rb");
		f180 = fopen(fname180str.c_str(), "rb");
		f270 = fopen(fname270str.c_str(), "rb");
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
		size_t nRead = fread(&hf, sizeof(fileheader), 1, f0);
		nRead = fread(&hInfo, sizeof(bitmapinfoheader), 1, f0);

		if (hf.signature[0] != 'B' || hf.signature[1] != 'M') { LOG("Not BMP File!\n"); }
		if ((hInfo.height == 0) || (hInfo.width == 0))
		{
			LOG("bmp header is empty!\n");
			hInfo.height = context_.pixel_number[_X];
			hInfo.width = context_.pixel_number[_Y];
			if (hInfo.height == 0 || hInfo.width == 0)
			{
				LOG("check your parameter file!\n");
				return false;
			}
		}
		if ((context_.pixel_number[_Y] != (int)hInfo.height) || (context_.pixel_number[_X] != (int)hInfo.width)) {
			LOG("image size is different!\n");
			context_.pixel_number[_Y] = (int)hInfo.height;
			context_.pixel_number[_X] = (int)hInfo.width;
			LOG("changed parameter of size %d x %d\n", context_.pixel_number[_X], context_.pixel_number[_Y]);
		}
		bitsperpixel = hInfo.bitsperpixel;
		if (hInfo.bitsperpixel == 8)
		{
			_wavelength_num = 1;
			rgbquad palette[256];
			size_t nRead;
			nRead = fread(palette, sizeof(rgbquad), 256, f0);
			nRead = fread(palette, sizeof(rgbquad), 256, f90);
			nRead = fread(palette, sizeof(rgbquad), 256, f180);
			nRead = fread(palette, sizeof(rgbquad), 256, f270);

			f0Mat[0].resize(hInfo.height, hInfo.width);
			f90Mat[0].resize(hInfo.height, hInfo.width);
			f180Mat[0].resize(hInfo.height, hInfo.width);
			f270Mat[0].resize(hInfo.height, hInfo.width);
			ComplexH = new OphComplexField;
			ComplexH[0].resize(hInfo.height, hInfo.width);
		}
		else
		{
			_wavelength_num = 3;
			ComplexH = new OphComplexField[3];
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

		size_t size = hInfo.width * hInfo.height * (hInfo.bitsperpixel >> 3);
		uchar* f0data = (uchar*)malloc(sizeof(uchar) * size);
		uchar* f90data = (uchar*)malloc(sizeof(uchar) * size);
		uchar* f180data = (uchar*)malloc(sizeof(uchar) * size);
		uchar* f270data = (uchar*)malloc(sizeof(uchar) * size);

		nRead = fread(f0data, sizeof(uchar), size, f0);
		nRead = fread(f90data, sizeof(uchar), size, f90);
		nRead = fread(f180data, sizeof(uchar), size, f180);
		nRead = fread(f270data, sizeof(uchar), size, f270);

		fclose(f0);
		fclose(f90);
		fclose(f180);
		fclose(f270);

		uint16_t bpp = hInfo.bitsperpixel >> 3;
		uint32_t nLine = hInfo.width * bpp;
		for (int i = hInfo.height - 1; i >= 0; i--)
		{
			for (int j = 0; j < static_cast<int>(hInfo.width); j++)
			{
				for (int z = 0; z < (hInfo.bitsperpixel / 8); z++)
				{
					int idx = hInfo.height - i - 1;
					int idx2 = i * nLine + bpp * j + z;
					f0Mat[z](idx, j) = (double)f0data[idx2];
					f90Mat[z](idx, j) = (double)f90data[idx2];
					f180Mat[z](idx, j) = (double)f180data[idx2];
					f270Mat[z](idx, j) = (double)f270data[idx2];
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
	for (int z = 0; z < (hInfo.bitsperpixel >> 3); z++)
	{
		for (int i = 0; i < context_.pixel_number[_X]; i++)
		{
			for (int j = 0; j < context_.pixel_number[_Y]; j++)
			{
				ComplexH[z][j][i][_RE] = (f0Mat[z][j][i] - f180Mat[z][j][i])*normalizefactor;
				ComplexH[z][j][i][_IM] = (f90Mat[z][j][i] - f270Mat[z][j][i])*normalizefactor;

			}
		}
	}
	LOG("complex field obtained from 4 psdh\n");

	auto end_time = CUR_TIME;

	auto during_time = ((std::chrono::duration<Real>)(end_time - start_time)).count();

	LOG("Implement time : %.5lf sec\n", during_time);

	return true;
}

bool ophSig::getComplexHFrom3ArbStepPSDH(const char* fname0, const char* fname1, const char* fname2, const char* fnameOI, const char* fnameRI, int nIter)
{
	auto start_time = CUR_TIME;
	string fname0str = fname0;
	string fname1str = fname1;
	string fname2str = fname2;
	string fnameOIstr = fnameOI;
	string fnameRIstr = fnameRI;
	int checktype = static_cast<int>(fname0str.rfind("."));
	OphRealField f0Mat[3], f1Mat[3], f2Mat[3], fOIMat[3], fRIMat[3];

	std::string f0type = fname0str.substr(checktype + 1, fname0str.size());

	uint16_t bitsperpixel;
	fileheader hf;
	bitmapinfoheader hInfo;

	if (f0type == "bmp")
	{
		FILE *f0, *f1, *f2, *fOI, *fRI;
		f0 = fopen(fname0str.c_str(), "rb");
		f1 = fopen(fname1str.c_str(), "rb");
		f2 = fopen(fname2str.c_str(), "rb");
		fOI = fopen(fnameOIstr.c_str(), "rb");
		fRI = fopen(fnameRIstr.c_str(), "rb");
		if (!f0)
		{
			LOG("bmp file open fail! (first interference pattern)\n");
			return false;
		}
		if (!f1)
		{
			LOG("bmp file open fail! (second interference pattern)\n");
			return false;
		}
		if (!f2)
		{
			LOG("bmp file open fail! (third interference pattern)\n");
			return false;
		}
		if (!fOI)
		{
			LOG("bmp file open fail! (object wave intensity pattern)\n");
			return false;
		}
		if (!fRI)
		{
			LOG("bmp file open fail! (reference wave intensity pattern)\n");
			return false;
		}

		size_t nRead;
		nRead = fread(&hf, sizeof(fileheader), 1, f0);
		nRead = fread(&hInfo, sizeof(bitmapinfoheader), 1, f0);
		nRead = fread(&hf, sizeof(fileheader), 1, f1);
		nRead = fread(&hInfo, sizeof(bitmapinfoheader), 1, f1);
		nRead = fread(&hf, sizeof(fileheader), 1, f2);
		nRead = fread(&hInfo, sizeof(bitmapinfoheader), 1, f2);
		nRead = fread(&hf, sizeof(fileheader), 1, fOI);
		nRead = fread(&hInfo, sizeof(bitmapinfoheader), 1, fOI);
		nRead = fread(&hf, sizeof(fileheader), 1, fRI);
		nRead = fread(&hInfo, sizeof(bitmapinfoheader), 1, fRI);

		if (hf.signature[0] != 'B' || hf.signature[1] != 'M') { LOG("Not BMP File!\n"); }
		if ((hInfo.height == 0) || (hInfo.width == 0))
		{
			LOG("bmp header is empty!\n");
			hInfo.height = context_.pixel_number[_X];
			hInfo.width = context_.pixel_number[_Y];
			if (hInfo.height == 0 || hInfo.width == 0)
			{
				LOG("check your parameter file!\n");
				return false;
			}
		}
		if ((context_.pixel_number[_Y] != (int)hInfo.height) || (context_.pixel_number[_X] != (int)hInfo.width)) {
			LOG("image size is different!\n");
			context_.pixel_number[_Y] = (int)hInfo.height;
			context_.pixel_number[_X] = (int)hInfo.width;
			LOG("changed parameter of size %d x %d\n", context_.pixel_number[_X], context_.pixel_number[_Y]);
		}
		bitsperpixel = hInfo.bitsperpixel;
		if (hInfo.bitsperpixel == 8)
		{
			_wavelength_num = 1;
			rgbquad palette[256];

			size_t nRead;
			nRead = fread(palette, sizeof(rgbquad), 256, f0);
			nRead = fread(palette, sizeof(rgbquad), 256, f1);
			nRead = fread(palette, sizeof(rgbquad), 256, f2);
			nRead = fread(palette, sizeof(rgbquad), 256, fOI);
			nRead = fread(palette, sizeof(rgbquad), 256, fRI);

			f0Mat[0].resize(hInfo.height, hInfo.width);
			f1Mat[0].resize(hInfo.height, hInfo.width);
			f2Mat[0].resize(hInfo.height, hInfo.width);
			fOIMat[0].resize(hInfo.height, hInfo.width);
			fRIMat[0].resize(hInfo.height, hInfo.width);
			ComplexH = new OphComplexField;
			ComplexH[0].resize(hInfo.height, hInfo.width);
		}
		else
		{
			_wavelength_num = 3;
			ComplexH = new OphComplexField[3];
			f0Mat[0].resize(hInfo.height, hInfo.width);
			f1Mat[0].resize(hInfo.height, hInfo.width);
			f2Mat[0].resize(hInfo.height, hInfo.width);
			fOIMat[0].resize(hInfo.height, hInfo.width);
			fRIMat[0].resize(hInfo.height, hInfo.width);
			ComplexH[0].resize(hInfo.height, hInfo.width);

			f0Mat[1].resize(hInfo.height, hInfo.width);
			f1Mat[1].resize(hInfo.height, hInfo.width);
			f2Mat[1].resize(hInfo.height, hInfo.width);
			fOIMat[1].resize(hInfo.height, hInfo.width);
			fRIMat[1].resize(hInfo.height, hInfo.width);
			ComplexH[1].resize(hInfo.height, hInfo.width);

			f0Mat[2].resize(hInfo.height, hInfo.width);
			f1Mat[2].resize(hInfo.height, hInfo.width);
			f2Mat[2].resize(hInfo.height, hInfo.width);
			fOIMat[2].resize(hInfo.height, hInfo.width);
			fRIMat[2].resize(hInfo.height, hInfo.width);
			ComplexH[2].resize(hInfo.height, hInfo.width);
		}

		size_t size = hInfo.width * hInfo.height * (hInfo.bitsperpixel >> 3);

		uchar* f0data = (uchar*)malloc(sizeof(uchar) * size);
		uchar* f1data = (uchar*)malloc(sizeof(uchar) * size);
		uchar* f2data = (uchar*)malloc(sizeof(uchar) * size);
		uchar* fOIdata = (uchar*)malloc(sizeof(uchar) * size);
		uchar* fRIdata = (uchar*)malloc(sizeof(uchar) * size);

		nRead = fread(f0data, sizeof(uchar), size, f0);
		nRead = fread(f1data, sizeof(uchar), size, f1);
		nRead = fread(f2data, sizeof(uchar), size, f2);
		nRead = fread(fOIdata, sizeof(uchar), size, fOI);
		nRead = fread(fRIdata, sizeof(uchar), size, fRI);

		fclose(f0);
		fclose(f1);
		fclose(f2);
		fclose(fOI);
		fclose(fRI);

		uint16_t bpp = hInfo.bitsperpixel >> 3;

		for (int i = hInfo.height - 1; i >= 0; i--)
		{
			for (int j = 0; j < static_cast<int>(hInfo.width); j++)
			{
				for (int z = 0; z < (hInfo.bitsperpixel >> 3); z++)
				{
					int idx = hInfo.height - i - 1;
					size_t idx2 = i * size + bpp * j + z;
					f0Mat[z](idx, j) = (double)f0data[idx2];
					f1Mat[z](idx, j) = (double)f1data[idx2];
					f2Mat[z](idx, j) = (double)f2data[idx2];
					fOIMat[z](idx, j) = (double)fOIdata[idx2];
					fRIMat[z](idx, j) = (double)fRIdata[idx2];
				}
			}
		}

		LOG("PSDH_3ArbStep file load complete!\n");

		free(f0data);
		free(f1data);
		free(f2data);
		free(fOIdata);
		free(fRIdata);

	}
	else
	{
		LOG("wrong type (only BMP supported)\n");
	}

	// calculation complexH from 3 arbitrary step intereference patterns and the object wave intensity
	// prepare some variables
	double P[2] = { 0.0, }; // please see ref.
	double C[2] = { 2.0/M_PI, 2.0/M_PI };
	double alpha[2] = { 0.0, }; //phaseShift[j+1]-phaseShift[j]
	double ps[3] = { 0.0, };	// reference wave phase shift for each inteference pattern
	const int nX = context_.pixel_number[_X];
	const int nY = context_.pixel_number[_Y];
	const int nXY = nX * nY;
	

	// calculate difference between interference patterns
	OphRealField I01Mat, I02Mat, I12Mat, OAMat, RAMat;
	I01Mat.resize(nY, nX);
	I02Mat.resize(nY, nX);
	I12Mat.resize(nY, nX);
	OAMat.resize(nY, nX);
	RAMat.resize(nY, nX);
	
	double sin2m1h, sin2m0h, sin1m0h, sin0p1h, sin0p2h, cos0p1h, cos0p2h, sin1p2h, cos1p2h;
	double sinP, cosP;
	for (int z = 0; z < (hInfo.bitsperpixel / 8); z++)
	{
		// initialize
		P[0] = 0.0;
		P[1] = 0.0;
		C[0] = 2.0 / M_PI;
		C[1] = 2.0 / M_PI;

		// load current channel 
		for (int i = 0; i < nX; i++)
		{
			for (int j = 0; j < nY; j++)
			{
				I01Mat[j][i] = (f0Mat[z][j][i] - f1Mat[z][j][i]) / 255.;	// difference & normalize
				I02Mat[j][i] = (f0Mat[z][j][i] - f2Mat[z][j][i]) / 255.;  // difference & normalize
				I12Mat[j][i] = (f1Mat[z][j][i] - f2Mat[z][j][i]) / 255.;  // difference & normalize
				OAMat[j][i] = sqrt(fOIMat[z][j][i] / 255.);			// normalize & then calculate amplitude from intensity
				RAMat[j][i] = sqrt(fRIMat[z][j][i] / 255.);			// normalize & then calculate amplitude from intensity
			}
		}

		// calculate P
		for (int i = 0; i < nX; i++)
		{
			for (int j = 0; j < nY; j++)
			{
				P[0] += abs(I01Mat[j][i] / OAMat[j][i] / RAMat[j][i]);
				P[1] += abs(I12Mat[j][i] / OAMat[j][i] / RAMat[j][i]);
			}
		}
		P[0] = P[0] / (4.*((double) nXY));
		P[1] = P[1] / (4.*((double) nXY));
		LOG("P %f  %f\n", P[0], P[1]);
		
		// iterative search
		for (int iter = 0; iter < nIter; iter++)
		{
			LOG("C %d %f  %f\n", iter, C[0], C[1]);
			LOG("ps %d %f  %f  %f\n", iter, ps[0], ps[1], ps[2]);

			alpha[0] = 2.*asin(P[0] / C[0]);
			alpha[1] = 2.*asin(P[1] / C[1]);

			ps[0] = 0.0;
			ps[1] = ps[0] + alpha[0];
			ps[2] = ps[1] + alpha[1];

			sin2m1h = sin((ps[2] - ps[1]) / 2.);
			sin2m0h = sin((ps[2] - ps[0]) / 2.);
			sin1m0h = sin((ps[1] - ps[0]) / 2.);
			sin0p1h = sin((ps[0] + ps[1]) / 2.);
			sin0p2h = sin((ps[0] + ps[2]) / 2.);
			cos0p1h = cos((ps[0] + ps[1]) / 2.);
			cos0p2h = cos((ps[0] + ps[2]) / 2.);
			for (int i = 0; i < nX; i++)
			{
				for (int j = 0; j < nY; j++)
				{
					ComplexH[z][j][i][_RE] = (1. / (4.*RAMat[j][i]*sin2m1h))*((cos0p1h / sin2m0h)*I02Mat[j][i] - (cos0p2h / sin1m0h)*I01Mat[j][i]);
					ComplexH[z][j][i][_IM] = (1. / (4.*RAMat[j][i]*sin2m1h))*((sin0p1h / sin2m0h)*I02Mat[j][i] - (sin0p2h / sin1m0h)*I01Mat[j][i]);
				}
			}

			// update C
			C[0] = 0.0;
			C[1] = 0.0;
			sin1p2h = sin((ps[1] + ps[2]) / 2.);
			cos1p2h = cos((ps[1] + ps[2]) / 2.);
			for (int i = 0; i < nX; i++)
			{
				for (int j = 0; j < nY; j++)
				{
					sinP = ComplexH[z][j][i][_IM] / OAMat[j][i];
					cosP = ComplexH[z][j][i][_RE] / OAMat[j][i];
					C[0] += abs(sinP*cos0p1h - cosP*sin0p1h);
					C[1] += abs(sinP*cos1p2h - cosP*sin1p2h);
				}
			}
			LOG("C1 %d %f  %f\n", iter, C[0], C[1]);
			C[0] = C[0] / ((double)nXY);
			C[1] = C[1] / ((double)nXY);	
			LOG("C2 %d %f  %f\n", iter, C[0], C[1]);

			/// temporary. only because save function clamps negative values to zero.
			for (int i = 0; i < nX; i++)
			{
				for (int j = 0; j < nY; j++)
				{
					ComplexH[z][j][i][_RE] = ComplexH[z][j][i][_RE] + 0.5;
					ComplexH[z][j][i][_IM] = ComplexH[z][j][i][_IM] + 0.5;
				}
			}
		}
	}
	

	LOG("complex field obtained from 3 interference patterns\n");

	auto end_time = CUR_TIME;

	auto during_time = ((std::chrono::duration<Real>)(end_time - start_time)).count();

	LOG("Implement time : %.5lf sec\n", during_time);

	return true;
}


void ophSig::ophFree(void) {

}