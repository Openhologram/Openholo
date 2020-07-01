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
#include <omp.h>
#include "sys.h"
#include "ImgCodecOhc.h"
#include "ImgControl.h"

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
	, complex_H(nullptr)
{
	context_ = { 0 };
	fftw_init_threads();
	fftw_plan_with_nthreads(omp_get_max_threads());
	OHC_encoder = new oph::ImgEncoderOhc;
	OHC_decoder = new oph::ImgDecoderOhc;
}

Openholo::~Openholo(void)
{
	if (OHC_encoder) {
		delete OHC_encoder;
		OHC_encoder = nullptr;
	}
	if (OHC_decoder) {
		delete OHC_decoder;
		OHC_decoder = nullptr;
	}
	fftw_cleanup_threads();
}

bool Openholo::checkExtension(const char * fname, const char * ext)
{
	string filename(fname);
	string fext(ext);
	size_t find = filename.find_last_of(".");
	size_t len = filename.length();

	if (find > len)
		return false;
	
	if (!filename.substr(find).compare(fext))
		return true;
	else
		return false;
}

bool Openholo::saveAsImg(const char * fname, uint8_t bitsperpixel, uchar* src, int width, int height)
{
	LOG("Saving...%s...\n", fname);
	bool bOK = true;
	auto start = CUR_TIME;
	int _width = width, _height = height;

	int _byteperline = ((_width * bitsperpixel / 8) + 3) & ~3;
	int _pixelbytesize = _height * _byteperline;
	int _filesize = _pixelbytesize;
	bool hasColorTable = (bitsperpixel <= 8) ? true : false;
	int _headersize = sizeof(bitmap);
	int _iColor = (hasColorTable) ? 256 : 0;


	rgbquad *table = nullptr;

	if (hasColorTable) {
		_headersize += _iColor * sizeof(rgbquad);
		table = new rgbquad[_iColor];
		memset(table, 0, sizeof(rgbquad) * _iColor);
		for (int i = 0; i < _iColor; i++) { // for gray-scale
			table[i].rgbBlue = i;
			table[i].rgbGreen = i;
			table[i].rgbRed = i;
		}		
	}

	_filesize += _headersize;

	bool bConvert = _stricmp(PathFindExtensionA(fname) + 1, "bmp") ? true : false;

	uchar *pBitmap = new uchar[_filesize];
	memset(pBitmap, 0x00, _filesize);

	bitmap bitmap;
	memset(&bitmap, 0, sizeof(bitmap));
	int iCur = 0;

	bitmap.fileheader.signature[0] = 'B';
	bitmap.fileheader.signature[1] = 'M';
	bitmap.fileheader.filesize = _filesize;
	bitmap.fileheader.fileoffset_to_pixelarray = _headersize;

	bitmap.bitmapinfoheader.dibheadersize = sizeof(bitmapinfoheader);
	bitmap.bitmapinfoheader.width = _width;
	bitmap.bitmapinfoheader.height = _height;
	bitmap.bitmapinfoheader.planes = OPH_PLANES;
	bitmap.bitmapinfoheader.bitsperpixel = bitsperpixel;
	bitmap.bitmapinfoheader.compression = OPH_COMPRESSION; //(=BI_RGB)
	bitmap.bitmapinfoheader.imagesize = _pixelbytesize;
	bitmap.bitmapinfoheader.ypixelpermeter = 0;// Y_PIXEL_PER_METER;
	bitmap.bitmapinfoheader.xpixelpermeter = 0;// X_PIXEL_PER_METER;
	bitmap.bitmapinfoheader.numcolorspallette = _iColor;
	
	memcpy(&pBitmap[iCur], &bitmap.fileheader, sizeof(fileheader));
	iCur += sizeof(fileheader);
	memcpy(&pBitmap[iCur], &bitmap.bitmapinfoheader, sizeof(bitmapinfoheader));
	iCur += sizeof(bitmapinfoheader);
	
	if (hasColorTable) {
		memcpy(&pBitmap[iCur], table, sizeof(rgbquad) * _iColor);
		iCur += sizeof(rgbquad) * _iColor;
	}

	if (context_.bRotation) {
		ImgControl *pControl = ImgControl::getInstance();
		uchar *pTmp = new uchar[_pixelbytesize];
		pControl->Rotate(180.0, src, pTmp, _width, _height, _width, _height, bitsperpixel / 8);
		memcpy(&pBitmap[iCur], pTmp, _pixelbytesize);
		delete[] pTmp;
	}
	else {
		memcpy(&pBitmap[iCur], src, _pixelbytesize);
	}

	iCur += _pixelbytesize;


	if (!bConvert) {
		if (iCur != _filesize)
			bOK = false;
		else {
			FILE *fp;
			fopen_s(&fp, fname, "wb");
			if (fp == nullptr)
				bOK = false;
			else {
				fwrite(pBitmap, 1, _filesize, fp);
				fclose(fp);
			}
		}
	}
	else {
		ImgControl *pControl = ImgControl::getInstance();
		pControl->Save(fname, pBitmap, _filesize);
	}

	if(hasColorTable && table) delete[] table;
	delete[] pBitmap;
	auto end = CUR_TIME;

	auto during = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("%.5lfsec...done\n", during);

	return bOK;
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
		int nSize = (((hInfo.width * hInfo.bitsperpixel / 8) + 3) & ~3) * hInfo.height;
		img_tmp = new uchar[nSize];
		fread(img_tmp, sizeof(oph::uchar), nSize, infile);
	}
	else {
		img_tmp = new uchar[hInfo.imagesize];
		fread(img_tmp, sizeof(oph::uchar), hInfo.imagesize, infile);
	}
	fclose(infile);

	return img_tmp;
}

bool Openholo::saveAsOhc(const char * fname)
{
	std::string fullname = fname;
	if (!checkExtension(fname, ".ohc")) fullname.append(".ohc");
	OHC_encoder->setFileName(fullname.c_str());
	
	// Clear vector
	OHC_encoder->releaseFldData();

	ohcHeader header;
	OHC_encoder->getOHCheader(header);
	auto wavelength_num = header.fieldInfo.wavlenNum;

	for (uint i = 0; i < wavelength_num; i++)
		OHC_encoder->addComplexFieldData(complex_H[i]);

	if (!OHC_encoder->save()) return false;

	return true;
}

bool Openholo::loadAsOhc(const char * fname)
{
	std::string fullname = fname;
	if (!checkExtension(fname, ".ohc")) fullname.append(".ohc");
	OHC_decoder->setFileName(fullname.c_str());
	if (!OHC_decoder->load()) return false;

	context_.waveNum = OHC_decoder->getNumOfWavlen();
	context_.pixel_number = OHC_decoder->getNumOfPixel();
	context_.pixel_pitch = OHC_decoder->getPixelPitch();

	vector<Real> wavelengthArray;
	OHC_decoder->getWavelength(wavelengthArray);
	context_.wave_length = new Real[wavelengthArray.size()];
	for (int i = 0; i < wavelengthArray.size(); i++)
		context_.wave_length[i] = wavelengthArray[i];
	
	OHC_decoder->getComplexFieldData(&complex_H);

	context_.k = (2 * M_PI) / context_.wave_length[0];
	context_.ss[_X] = context_.pixel_number[_X] * context_.pixel_pitch[_X];
	context_.ss[_Y] = context_.pixel_number[_Y] * context_.pixel_pitch[_Y];

	return true;
}

bool Openholo::loadAsImgUpSideDown(const char * fname, uchar* dst)
{
	FILE *infile;
	fopen_s(&infile, fname, "rb");
	if (infile == nullptr) { LOG("No such file"); return false; }

	// BMP Header Information
	fileheader hf;
	bitmapinfoheader hInfo;
	fread(&hf, sizeof(fileheader), 1, infile);
	if (hf.signature[0] != 'B' || hf.signature[1] != 'M') { LOG("Not BMP File");  return false; }

	fread(&hInfo, sizeof(bitmapinfoheader), 1, infile);
	fseek(infile, hf.fileoffset_to_pixelarray, SEEK_SET);

	oph::uchar* img_tmp;
	if (hInfo.imagesize == 0) {
		img_tmp = new oph::uchar[hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8)];
		fread(img_tmp, sizeof(oph::uchar), hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8), infile);
	}
	else {
		img_tmp = new oph::uchar[hInfo.imagesize];
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
	return true;
}

bool Openholo::getImgSize(int & w, int & h, int & bytesperpixel, const char * fname)
{

	char szExtension[_MAX_EXT] = { 0, };
	sprintf(szExtension, "%s", PathFindExtension(fname) + 1);

	if (_stricmp(szExtension, "bmp")) { // not bmp
		return false;
	}

	// BMP
	FILE *infile;
	fopen_s(&infile, fname, "rb");
	if (infile == NULL) { LOG("No Image File"); return false; }

	// BMP Header Information
	fileheader hf;
	bitmapinfoheader hInfo;
	fread(&hf, sizeof(fileheader), 1, infile);
	if (hf.signature[0] != 'B' || hf.signature[1] != 'M') return false;
	fread(&hInfo, sizeof(bitmapinfoheader), 1, infile);
	//if (hInfo.bitsperpixel != 8) { printf("Bad File Format!!"); return 0; }

	w = hInfo.width;
	h = hInfo.height;
	bytesperpixel = hInfo.bitsperpixel / 8;
	fclose(infile);

	return true;
}

void Openholo::ImageRotation(double rotate, uchar* src, uchar* dst, int w, int h, int channels)
{
	auto begin = CUR_TIME;
	int channel = channels;
	int nBytePerLine = ((w * channel) + 3) & ~3;

	int origX, origY;
	double radian = rotate * M_PI / 180.0;
	double cc = cos(radian);
	double ss = sin(-radian);
	double centerX = (double)w / 2.0;
	double centerY = (double)h / 2.0;

	uchar pixel;
	uchar R, G, B;

	int num_threads = 1;
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			origX = (int)(centerX + ((double)y - centerY)*ss + ((double)x - centerX)*cc);
			origY = (int)(centerY + ((double)y - centerY)*cc - ((double)x - centerX)*ss);

			pixel = 0;
			R = G = B = 0;
			if ((origY >= 0 && origY < h) && (origX >= 0 && origX < w)) {
				int offsetX = origX * channel;
				int offsetY = origY * nBytePerLine;
				B = src[offsetY + offsetX + 0];
				G = src[offsetY + offsetX + 1];
				R = src[offsetY + offsetX + 2];

			}
			if (channel == 1) {
				dst[y * nBytePerLine + x] = pixel;
			}
			else if (channel == 3) {
				int tmpX = x * 3;
				int tmpY = y * nBytePerLine;
				dst[tmpY + tmpX + 0] = B;
				dst[tmpY + tmpX + 1] = G;
				dst[tmpY + tmpX + 2] = R;
			}
		}
	}
	auto end = CUR_TIME;
	LOG("Image Rotated (%d threads): (%d/%d) (%lf degree) : %lf(s)\n",
		num_threads,
		w, h, rotate,
		((chrono::duration<Real>)(end - begin)).count());
}

void Openholo::imgScaleBilinear(uchar* src, uchar* dst, int w, int h, int neww, int newh, int channels)
{
	auto begin = CUR_TIME;
	int channel = channels;
	int nBytePerLine = ((w * channel) + 3) & ~3;
	int nNewBytePerLine = ((neww * channel) + 3) & ~3;
	int num_threads = 1;
#ifdef _OPENMP
	int y;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
#pragma omp for private(y)
		for (y = 0; y < newh; y++)
#else
		for (int y = 0; y < newh; y++)
#endif
		{
			int nbppY = y * nNewBytePerLine;
			for (int x = 0; x < neww; x++)
			{
				float gx = (x / (float)neww) * (w - 1);
				float gy = (y / (float)newh) * (h - 1);

				int gxi = (int)gx;
				int gyi = (int)gy;

				if (channel == 1) {
					uint32_t a00, a01, a10, a11;					

					a00 = src[gxi + 0 + gyi * nBytePerLine];
					a01 = src[gxi + 1 + gyi * nBytePerLine];
					a10 = src[gxi + 0 + (gyi + 1) * nBytePerLine];
					a11 = src[gxi + 1 + (gyi + 1) * nBytePerLine];

					float dx = gx - gxi;
					float dy = gy - gyi;

					float w1 = (1 - dx) * (1 - dy);
					float w2 = dx * (1 - dy);
					float w3 = (1 - dx) * dy;
					float w4 = dx * dy;

					dst[x + y * neww] = int(a00 * w1 + a01 * w2 + a10 * w3 + a11 * w4);
				}
				else if (channel == 3) {
					uint32_t b00[3], b01[3], b10[3], b11[3];
					int srcX = gxi * channel;
					int dstX = x * channel;

					b00[0] = src[srcX + 0 + gyi * nBytePerLine];
					b00[1] = src[srcX + 1 + gyi * nBytePerLine];
					b00[2] = src[srcX + 2 + gyi * nBytePerLine];
					
					b01[0] = src[srcX + 3 + gyi * nBytePerLine];
					b01[1] = src[srcX + 4 + gyi * nBytePerLine];
					b01[2] = src[srcX + 5 + gyi * nBytePerLine];

					b10[0] = src[srcX + 0 + (gyi + 1) * nBytePerLine];
					b10[1] = src[srcX + 1 + (gyi + 1) * nBytePerLine];
					b10[2] = src[srcX + 2 + (gyi + 1) * nBytePerLine];

					b11[0] = src[srcX + 3 + (gyi + 1) * nBytePerLine];
					b11[1] = src[srcX + 4 + (gyi + 1) * nBytePerLine];
					b11[2] = src[srcX + 5 + (gyi + 1) * nBytePerLine];
					
					float dx = gx - gxi;
					float dy = gy - gyi;

					float w1 = (1 - dx) * (1 - dy);
					float w2 = dx * (1 - dy);
					float w3 = (1 - dx) * dy;
					float w4 = dx * dy;

					dst[dstX + 0 + nbppY] = int(b00[0] * w1 + b01[0] * w2 + b10[0] * w3 + b11[0] * w4);
					dst[dstX + 1 + nbppY] = int(b00[1] * w1 + b01[1] * w2 + b10[1] * w3 + b11[1] * w4);
					dst[dstX + 2 + nbppY] = int(b00[2] * w1 + b01[2] * w2 + b10[2] * w3 + b11[2] * w4);
				}
			}
		}
#ifdef _OPENMP
	}
#endif
	auto end = CUR_TIME;
	LOG("Scaled img size (%d threads): (%d/%d) => (%d/%d) : %lf(s)\n",
		num_threads,
		w, h, neww, newh, 
		((chrono::duration<Real>)(end - begin)).count());
}

void Openholo::convertToFormatGray8(unsigned char * src, unsigned char * dst, int w, int h, int bytesperpixel)
{
	int idx = 0;
	unsigned int r = 0, g = 0, b = 0;
	for (int i = 0; i < w*h*bytesperpixel; i++)
	{
		unsigned int blue = src[i + 0];
		unsigned int green = src[i + 1];
		unsigned int red = src[i + 2];
		dst[idx++] = (blue + green + red) / 3;
		i += bytesperpixel - 1;
	}
}

void Openholo::fft1(int n, Complex<Real>* in, int sign, uint flag)
{
	pnx = n;
	bool bIn = true;

	if (fft_in == nullptr)
		fft_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);
	if (fft_out == nullptr)
		fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);

	if (!in){
		in = new Complex<Real>[pnx];
		bIn = false;
	}

	for (int i = 0; i < n; i++) {
		fft_in[i][_RE] = in[i].real();
		fft_in[i][_IM] = in[i].imag();
	}

	fft_sign = sign;

	if (!bIn) delete[] in;

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
	if (in == nullptr) return;

	pnx = n[_X], pny = n[_Y];

	if (fft_in == nullptr)
		fft_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * pnx * pny);
	if (fft_out == nullptr)
		fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * pnx * pny);

#if 0
	memcpy(fft_in, in, sizeof(fftw_complex) * pnx * pny);
#else
	int i;
	for (i = 0; i < pnx * pny; i++) {
		fft_in[i][_RE] = in[i][_RE];
		fft_in[i][_IM] = in[i][_IM];
	}
#endif

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
	bool bIn = true;
	if (fft_in == nullptr)
		fft_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * pnx * pny * pnz);
	if (fft_out == nullptr)
		fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * pnx * pny * pnz);

	if (!in) {
		in = new Complex<Real>[pnx * pny * pnz];
		bIn = false;
	}

	for (int i = 0; i < pnx * pny * pnz; i++) {
		fft_in[i][_RE] = in[i].real();
		fft_in[i][_IM] = in[i].imag();
	}

	fft_sign = sign;

	if (!bIn) delete[] in;

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

void Openholo::fftExecute(Complex<Real>* out, bool bReverse)
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

	if (!bReverse) {
		int i;
#ifdef _OPENMP
#pragma omp for private(i)
#endif
		for (i = 0; i < pnx * pny * pnz; i++) {
			out[i][_RE] = fft_out[i][_RE];
			out[i][_IM] = fft_out[i][_IM];
		}
	}
	else {
		int div = pnx * pny * pnz;
		int i;
#ifdef _OPENMP
#pragma omp for private(i)
#endif
		for (i = 0; i < pnx * pny * pnz; i++) {
			out[i][_RE] = fft_out[i][_RE] / div;
			out[i][_IM] = fft_out[i][_IM] / div;
		}

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
	Complex<Real>* tmp = new Complex<Real>[nx*ny];
	memset(tmp, 0., sizeof(Complex<Real>)*nx*ny);
	fftShift(nx, ny, src, tmp);

	fftw_complex *in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nx * ny);
	fftw_complex *out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nx * ny);
	
	memcpy(in, tmp, sizeof(Complex<Real>) * nx * ny);

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

	int k;
#pragma omp parallel for private(k)
	for (k = 0; k < nx*ny; k++) {
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
	int hnx = nx / 2;
	int hny = ny / 2;

	if (nx <= ny) {
		int i;
#ifdef _OPENMP
#pragma omp for private(i)
#endif
		for (i = 0; i < nx; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				int ti = i - hnx; if (ti < 0) ti += nx;
				int tj = j - hny; if (tj < 0) tj += ny;

				output[ti + tj * nx] = input[i + j * nx];
			}
		}
	}
	else {
		int j;
#ifdef _OPENMP
#pragma omp for private(j)
#endif
		for (j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				int ti = i - hnx; if (ti < 0) ti += nx;
				int tj = j - hny; if (tj < 0) tj += ny;

				output[ti + tj * nx] = input[i + j * nx];
			}
		}

	}
}

void Openholo::setWaveNum(int nNum)
{
	context_.waveNum = nNum;
	if (context_.wave_length != nullptr) {
		delete[] context_.wave_length;
		context_.wave_length = nullptr;
	}
	
	context_.wave_length = new Real[nNum];
}


void Openholo::ophFree(void)
{
	ohcHeader header;
	OHC_encoder->getOHCheader(header);
	auto wavelength_num = header.fieldInfo.wavlenNum;
	for (uint i = 0; i < wavelength_num; i++) {
		if (complex_H[i]) {
			delete[] complex_H[i];
			complex_H[i] = nullptr;
		}
	}
	if (complex_H) {
		delete[] complex_H;
		complex_H = nullptr;
	}
	if (context_.wave_length) {
		delete[] context_.wave_length;
		context_.wave_length = nullptr;
	}
	if (OHC_encoder) {
		delete OHC_encoder;
		OHC_encoder = nullptr;
	}
	if (OHC_decoder) {
		delete OHC_decoder;
		OHC_decoder = nullptr;
	}
}

