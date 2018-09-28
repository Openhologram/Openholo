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

#include "Openholo.h"

#include <windows.h>
#include <fileapi.h>

#include "sys.h"

#include "ImgCodecOhc.h"

Openholo::Openholo(void)
	: Base()
	, plan_fwd(nullptr)
	, plan_bwd(nullptr)
	, fft_in(nullptr)
	, fft_out(nullptr)
	, pnx(1)
	, pny(1)
	, pnz(1)
	, fft_sign(OPH_FORWARD)
	, OHC_encoder(nullptr)
	, OHC_decoder(nullptr)
{
	OHC_encoder = new oph::ImgEncoderOhc;
	OHC_decoder = new oph::ImgDecoderOhc;
}

Openholo::~Openholo(void)
{
}

int Openholo::checkExtension(const char * fname, const char * ext)
{	
	//return	1	: the extension of "fname" and "ext" is the same
	//			0	: the extension of "fname" and "ext" is not the same

	std::string filename(fname);
	size_t pos = filename.find(ext);
	if (pos == std::string::npos)
		//when there is no search string
		return 0;
	else
		return 1;
}

int Openholo::saveAsImg(const char * fname, uint8_t bitsperpixel, uchar* src, int pic_width, int pic_height)
{
	LOG("Saving...%s...", fname);
	auto start = CUR_TIME;

	int _width = pic_width, _height = pic_height;

	int _pixelbytesize = _height * _width * bitsperpixel / 8;
	int _filesize = _pixelbytesize + sizeof(bitmap);

	FILE *fp;
	fopen_s(&fp, fname, "wb");
	if (fp == nullptr) return -1;

	bitmap *pbitmap = (bitmap*)calloc(1, sizeof(bitmap));
	memset(pbitmap, 0x00, sizeof(bitmap));

	pbitmap->fileheader.signature[0] = 'B';
	pbitmap->fileheader.signature[1] = 'M';
	pbitmap->fileheader.filesize = _filesize;
	pbitmap->fileheader.fileoffset_to_pixelarray = sizeof(bitmap);

	for (int i = 0; i < 256; i++) {
		pbitmap->rgbquad[i].rgbBlue = i;
		pbitmap->rgbquad[i].rgbGreen = i;
		pbitmap->rgbquad[i].rgbRed = i;
	}

	pbitmap->bitmapinfoheader.dibheadersize = sizeof(bitmapinfoheader);
	pbitmap->bitmapinfoheader.width = _width;
	pbitmap->bitmapinfoheader.height = _height;
	pbitmap->bitmapinfoheader.planes = OPH_PLANES;
	pbitmap->bitmapinfoheader.bitsperpixel = bitsperpixel;
	pbitmap->bitmapinfoheader.compression = OPH_COMPRESSION;
	pbitmap->bitmapinfoheader.imagesize = _pixelbytesize;
	pbitmap->bitmapinfoheader.ypixelpermeter = Y_PIXEL_PER_METER;
	pbitmap->bitmapinfoheader.xpixelpermeter = X_PIXEL_PER_METER;
	pbitmap->bitmapinfoheader.numcolorspallette = 256;
	fwrite(pbitmap, 1, sizeof(bitmap), fp);

	fwrite(src, 1, _pixelbytesize, fp);
	fclose(fp);
	free(pbitmap);

	auto end = CUR_TIME;

	auto during = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("%.5lfsec...done\n", during);

	return 1;
}

uchar * Openholo::loadAsImg(const char * fname)
{
	FILE *infile;
	fopen_s(&infile, fname, "rb");
	if (infile == nullptr) { LOG("No such file"); return 0; }

	// BMP Header Information
	fileheader hf;
	bitmapinfoheader hInfo;
	fread(&hf, sizeof(fileheader), 1, infile);
	if (hf.signature[0] != 'B' || hf.signature[1] != 'M') { LOG("Not BMP File");  return 0; }

	fread(&hInfo, sizeof(bitmapinfoheader), 1, infile);
	fseek(infile, hf.fileoffset_to_pixelarray, SEEK_SET);

	oph::uchar *img_tmp;
	if (hInfo.imagesize == 0) {
		img_tmp = new uchar[hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8)];
		fread(img_tmp, sizeof(oph::uchar), hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8), infile);
	}
	else {
		img_tmp = new uchar[hInfo.imagesize];
		fread(img_tmp, sizeof(oph::uchar), hInfo.imagesize, infile);
	}
	fclose(infile);

	return img_tmp;
}

int Openholo::saveAsOhc(const char * fname, Complex<Real> *src)
{
	std::string fullname = fname;
	if (checkExtension(fname, ".ohc") == 0) fullname.append(".ohc");
	OHC_encoder->setFileName(fullname.c_str());
	OHC_encoder->addComplexFieldData(src);

	if (!OHC_encoder->save()) return -1;

	return 1;
}

int Openholo::saveAsOhc(const char * fname, OphComplexField & src)
{
	std::string fullname = fname;
	if (checkExtension(fname, ".ohc") == 0) fullname.append(".ohc");
	OHC_encoder->setFileName(fullname.c_str());
	OHC_encoder->addComplexFieldData(src);	

	if (!OHC_encoder->save()) return -1;

	return 1;
}

int Openholo::loadAsOhc(const char * fname, Complex<Real>** dst, ivec2 &pixel_number, vec2 &pixel_pitch, Real &wave_length)
{
	std::string fullname = fname;
	if (checkExtension(fname, ".ohc") == 0) fullname.append(".ohc");
	OHC_decoder->setFileName(fullname.c_str());
	if (!OHC_decoder->load()) return -1;

	pixel_number = OHC_decoder->getNumOfPixel();
	pixel_pitch = OHC_decoder->getPixelPitch();

	vector<Real> wavelengthArray;
	OHC_decoder->getWavelength(wavelengthArray);
	wave_length = wavelengthArray[0];
	
	OphComplexField res(pixel_number[_X], pixel_number[_Y]);
	OHC_decoder->getComplexFieldData(res);

	Field2Buffer(res, dst);

	return 1;
}

int Openholo::loadAsOhc(const char * fname, OphComplexField & dst, ivec2 &pixel_number, vec2 &pixel_pitch, Real &wave_length)
{
	std::string fullname = fname;
	if (checkExtension(fname, ".ohc") == 0) fullname.append(".ohc");
	OHC_decoder->setFileName(fullname.c_str());
	if (!OHC_decoder->load()) return -1;

	pixel_number = OHC_decoder->getNumOfPixel();
	pixel_pitch = OHC_decoder->getPixelPitch();

	vector<Real> wavelengthArray;
	OHC_decoder->getWavelength(wavelengthArray);
	wave_length = wavelengthArray[0];

	dst.resize(pixel_number[_X], pixel_number[_Y]);
	dst.zeros();

	OHC_decoder->getComplexFieldData(dst);

	return 1;
}


int Openholo::loadAsImgUpSideDown(const char * fname, uchar* dst)
{
	FILE *infile;
	fopen_s(&infile, fname, "rb");
	if (infile == nullptr) { LOG("No such file"); return 0; }

	// BMP Header Information
	fileheader hf;
	bitmapinfoheader hInfo;
	fread(&hf, sizeof(fileheader), 1, infile);
	if (hf.signature[0] != 'B' || hf.signature[1] != 'M') { LOG("Not BMP File");  return 0; }

	fread(&hInfo, sizeof(bitmapinfoheader), 1, infile);
	fseek(infile, hf.fileoffset_to_pixelarray, SEEK_SET);

	oph::uchar* img_tmp;
	if (hInfo.imagesize == 0) {
		img_tmp = (oph::uchar*)malloc(sizeof(oph::uchar)*hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8));
		fread(img_tmp, sizeof(oph::uchar), hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8), infile);
	}
	else {
		img_tmp = (oph::uchar*)malloc(hInfo.imagesize);
		fread(img_tmp, sizeof(oph::uchar), hInfo.imagesize, infile);
	}
	fclose(infile);

	// data upside down
	int bytesperpixel = hInfo.bitsperpixel / 8;
	int rowsz = bytesperpixel * hInfo.width;

	for (oph::uint k = 0; k < hInfo.height*rowsz; k++) {
		int r = k / rowsz;
		int c = k % rowsz;
		((oph::uchar*)dst)[(hInfo.height - r - 1)*rowsz + c] = img_tmp[r*rowsz + c];
	}

	delete[] img_tmp;

	return 1;
}

int Openholo::getImgSize(int & w, int & h, int & bytesperpixel, const char * file_name)
{
	char bmpFile[256];
	sprintf_s(bmpFile, "%s", file_name);
	FILE *infile;
	fopen_s(&infile, bmpFile, "rb");
	if (infile == NULL) { LOG("No Image File"); return 0; }

	// BMP Header Information
	fileheader hf;
	bitmapinfoheader hInfo;
	fread(&hf, sizeof(fileheader), 1, infile);
	if (hf.signature[0] != 'B' || hf.signature[1] != 'M') return 0;
	fread(&hInfo, sizeof(bitmapinfoheader), 1, infile);
	//if (hInfo.bitsperpixel != 8) { printf("Bad File Format!!"); return 0; }

	w = hInfo.width;
	h = hInfo.height;
	bytesperpixel = hInfo.bitsperpixel / 8;

	fclose(infile);

	return 1;
}

void Openholo::imgScaleBilnear(unsigned char * src, unsigned char * dst, int w, int h, int neww, int newh)
{
	for (int y = 0; y < newh; y++)
	{
		for (int x = 0; x < neww; x++)
		{
			float gx = (x / (float)neww) * (w - 1);
			float gy = (y / (float)newh) * (h - 1);

			int gxi = (int)gx;
			int gyi = (int)gy;

			uint32_t a00 = src[gxi + 0 + gyi * w];
			uint32_t a01 = src[gxi + 1 + gyi * w];
			uint32_t a10 = src[gxi + 0 + (gyi + 1)*w];
			uint32_t a11 = src[gxi + 1 + (gyi + 1)*w];

			float dx = gx - gxi;
			float dy = gy - gyi;

			dst[x + y * neww] = int(a00 * (1 - dx)*(1 - dy) + a01 * dx*(1 - dy) + a10 * (1 - dx)*dy + a11 * dx*dy);

		}
	}
}

void Openholo::convertToFormatGray8(unsigned char * src, unsigned char * dst, int w, int h, int bytesperpixel)
{
	int idx = 0;
	unsigned int r = 0, g = 0, b = 0;
	for (int i = 0; i < w*h*bytesperpixel; i++)
	{
		unsigned int r = src[i + 0];
		unsigned int g = src[i + 1];
		unsigned int b = src[i + 2];
		dst[idx++] = (r + g + b) / 3;
		i += bytesperpixel - 1;
	}
}

void Openholo::fft1(int n, Complex<Real>* in, int sign, uint flag)
{
	pnx = n;

	fft_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);
	fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);

	if (!in)
		in = new Complex<Real>[pnx];

	for (int i = 0; i < n; i++) {
		fft_in[i][_RE] = in[i].real();
		fft_in[i][_IM] = in[i].imag();
	}

	fft_sign = sign;

	if (sign == OPH_FORWARD)
		plan_fwd = fftw_plan_dft_1d(n, fft_in, fft_out, sign, flag);
	else if (sign == OPH_BACKWARD)
		plan_bwd = fftw_plan_dft_1d(n, fft_in, fft_out, sign, flag);
	else {
		LOG("failed fftw : wrong sign");
		fftFree();
		return;
	}
}

void Openholo::fft2(oph::ivec2 n, Complex<Real>* in, int sign, uint flag)
{
	pnx = n[_X], pny = n[_Y];

	fft_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * pnx * pny);
	fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * pnx * pny);

	if (!in)
		in = new Complex<Real>[pnx * pny];

	for (int i = 0; i < pnx * pny; i++) {
		fft_in[i][_RE] = in[i].real();
		fft_in[i][_IM] = in[i].imag();
	}

	fft_sign = sign;

	if (sign == OPH_FORWARD)
		plan_fwd = fftw_plan_dft_2d(pny, pnx, fft_in, fft_out, sign, flag);
	else if (sign == OPH_BACKWARD)
		plan_bwd = fftw_plan_dft_2d(pny, pnx, fft_in, fft_out, sign, flag);
	else {
		LOG("failed fftw : wrong sign");
		fftFree();
		return;
	}
}

void Openholo::fft3(oph::ivec3 n, Complex<Real>* in, int sign, uint flag)
{
	pnx = n[_X], pny = n[_Y], pnz = n[_Z];

	fft_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * pnx * pny * pnz);
	fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * pnx * pny * pnz);

	if (!in)
		in = new Complex<Real>[pnx * pny * pnz];

	for (int i = 0; i < pnx * pny * pnz; i++) {
		fft_in[i][_RE] = in[i].real();
		fft_in[i][_IM] = in[i].imag();
	}

	fft_sign = sign;

	if (sign == OPH_FORWARD)
		plan_fwd = fftw_plan_dft_3d(pnz, pny, pnx, fft_in, fft_out, sign, flag);
	else if (sign == OPH_BACKWARD)
		plan_bwd = fftw_plan_dft_3d(pnz, pny, pnx, fft_in, fft_out, sign, flag);
	else {
		LOG("failed fftw : wrong sign");
		fftFree();
		return;
	}
}

void Openholo::fftExecute(Complex<Real>* out)
{
	if (fft_sign == OPH_FORWARD)
		fftw_execute(plan_fwd);
	else if (fft_sign == OPH_BACKWARD)
		fftw_execute(plan_bwd);
	else {
		LOG("failed fftw : wrong sign");
		out = nullptr;
		fftFree();
		return;
	}

	for (int i = 0; i < pnx * pny * pnz; i++) {
		out[i][_RE] = fft_out[i][_RE];
		out[i][_IM] = fft_out[i][_IM];
	}

	fftFree();
}

void Openholo::fftFree(void)
{
	if (plan_fwd) {
		fftw_destroy_plan(plan_fwd);
		plan_fwd = nullptr;
	}
	if (plan_bwd) {
		fftw_destroy_plan(plan_bwd);
		plan_fwd = nullptr;
	}
	fftw_free(fft_in);
	fftw_free(fft_out);

	plan_bwd = nullptr;
	fft_in = nullptr;
	fft_out = nullptr;

	pnx = 1;
	pny = 1;
	pnz = 1;
}

void Openholo::fftwShift(Complex<Real>* src, Complex<Real>* dst, int nx, int ny, int type, bool bNormalized)
{
	Complex<Real>* tmp = (Complex<Real>*)malloc(sizeof(Complex<Real>)*nx*ny);
	memset(tmp, 0, sizeof(Complex<Real>)*nx*ny);
	fftShift(nx, ny, src, tmp);

	fftw_complex *in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nx * ny);
	fftw_complex *out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nx * ny);

	for (int i = 0; i < nx*ny; i++) {
		in[i][_RE] = tmp[i][_RE];
		in[i][_IM] = tmp[i][_IM];
	}	

	fftw_plan plan = nullptr;
	if (!plan_fwd && !plan_bwd) {
		plan = fftw_plan_dft_2d(ny, nx, in, out, type, OPH_ESTIMATE);
		fftw_execute(plan);
	}
	else {
		if (type == OPH_FORWARD)
			fftw_execute_dft(plan_fwd, in, out);
		else if (type == OPH_BACKWARD)
			fftw_execute_dft(plan_bwd, in, out);
	}

	int normalF = 1;
	if (bNormalized) normalF = nx * ny;
	memset(tmp, 0, sizeof(Complex<Real>)*nx*ny);

	for (int k = 0; k < nx*ny; k++) {
		tmp[k][_RE] = out[k][_RE] / normalF;
		tmp[k][_IM] = out[k][_IM] / normalF;
	}

	fftw_free(in);
	fftw_free(out);
	if (plan)
		fftw_destroy_plan(plan);

	memset(dst, 0, sizeof(Complex<Real>)*nx*ny);
	fftShift(nx, ny, tmp, dst);
	delete[] tmp;
}

void Openholo::fftShift(int nx, int ny, Complex<Real>* input, Complex<Real>* output)
{
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			int ti = i - nx / 2; if (ti < 0) ti += nx;
			int tj = j - ny / 2; if (tj < 0) tj += ny;

			output[ti + tj * nx] = input[i + j * nx];
		}
	}
}

void Openholo::ophFree(void)
{
	delete OHC_encoder;
	delete OHC_decoder;
}