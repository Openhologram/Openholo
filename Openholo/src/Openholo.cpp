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
#include <omp.h>
#include <limits.h>
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


bool Openholo::mergeColor(int idx, int width, int height, uchar *src, uchar *dst)
{
	if (idx < 0 || idx > 2) return false;

	int N = width * height;
	int a = 2 - idx;
#ifdef _OPENMP
#pragma omp parallel for firstprivate(a)
#endif
	for (int i = 0; i < N; i++) {
		dst[i * 3 + a] = src[i];
	}

	return true;
}

bool Openholo::separateColor(int idx, int width, int height, uchar *src, uchar *dst)
{
	if (idx < 0 || idx > 2) return false;

	int N = width * height;
	int a = 2 - idx;
#ifdef _OPENMP
#pragma omp parallel for firstprivate(a)
#endif
	for (int i = 0; i < N; i++) {
		dst[i] = src[i * 3 + a];
	}

	return true;
}

bool Openholo::saveAsImg(const char * fname, uint8_t bitsperpixel, uchar* src, int width, int height)
{
	bool bOK = true;
	auto begin = CUR_TIME;

	int padding = 0;
	int _byteperline = ((width * bitsperpixel >> 3) + 3) & ~3;
	int _pixelbytesize = height * _byteperline;
	int _filesize = _pixelbytesize;
	bool hasColorTable = (bitsperpixel <= 8) ? true : false;
	int _headersize = sizeof(bitmap);
	int _iColor = (hasColorTable) ? 256 : 0;

	int mod = width % 4;
	if (mod != 0) {
		padding = 4 - mod;
	}

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

	uchar *pBitmap = new uchar[_filesize];
	memset(pBitmap, 0x00, _filesize);

	bitmap bitmap;
	memset(&bitmap, 0, sizeof(bitmap));
	int iCur = 0;

	bitmap._fileheader.signature[0] = 'B';
	bitmap._fileheader.signature[1] = 'M';
	bitmap._fileheader.filesize = _filesize;
	bitmap._fileheader.fileoffset_to_pixelarray = _headersize;

	bitmap._bitmapinfoheader.dibheadersize = sizeof(bitmapinfoheader);
	bitmap._bitmapinfoheader.width = width;
	bitmap._bitmapinfoheader.height = height;
	bitmap._bitmapinfoheader.planes = OPH_PLANES;
	bitmap._bitmapinfoheader.bitsperpixel = bitsperpixel;
	bitmap._bitmapinfoheader.compression = OPH_COMPRESSION; //(=BI_RGB)
	bitmap._bitmapinfoheader.imagesize = _pixelbytesize;
	bitmap._bitmapinfoheader.ypixelpermeter = 0;// Y_PIXEL_PER_METER;
	bitmap._bitmapinfoheader.xpixelpermeter = 0;// X_PIXEL_PER_METER;
	bitmap._bitmapinfoheader.numcolorspallette = _iColor;

	memcpy(&pBitmap[iCur], &bitmap._fileheader, sizeof(fileheader));
	iCur += sizeof(fileheader);
	memcpy(&pBitmap[iCur], &bitmap._bitmapinfoheader, sizeof(bitmapinfoheader));
	iCur += sizeof(bitmapinfoheader);

	if (hasColorTable) {
		memcpy(&pBitmap[iCur], table, sizeof(rgbquad) * _iColor);
		iCur += sizeof(rgbquad) * _iColor;
	}

	ImgControl *pControl = ImgControl::getInstance();
	uchar *pTmp = new uchar[_pixelbytesize];
	memcpy(pTmp, src, _pixelbytesize);

	if (imgCfg.flip)
	{
		pControl->Flip((oph::FLIP)imgCfg.flip, pTmp, pTmp, width + padding, height, bitsperpixel >> 3);
	}

	if (imgCfg.rotate) {
		pControl->Rotate(180.0, pTmp, pTmp, width + padding, height, width + padding, height, bitsperpixel >> 3);
	}

	if (padding != 0)
	{
		for (int i = 0; i < height; i++)
		{
			memcpy(&pBitmap[iCur], &pTmp[width * i], width);
			iCur += width;
			memset(&pBitmap[iCur], 0x00, padding);
			iCur += padding;
		}
	}
	else
	{
		memcpy(&pBitmap[iCur], pTmp, _pixelbytesize);
		iCur += _pixelbytesize;
	}
	delete[] pTmp;

	if (iCur != _filesize)
		bOK = false;
	else {
		FILE* fp = fopen(fname, "wb");
		if (fp == nullptr)
			bOK = false;
		else {
			fwrite(pBitmap, 1, _filesize, fp);
			fclose(fp);
		}
	}

	if (hasColorTable && table) delete[] table;
	delete[] pBitmap;

	LOG("%s \'%s\' => %.5lf (sec)\n", __FUNCTION__, fname, ELAPSED_TIME(begin, CUR_TIME));

	return bOK;
}


bool Openholo::saveAsOhc(const char * fname)
{
	bool bRet = true;
	auto begin = CUR_TIME;

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

	if (!OHC_encoder->save())
	{
		bRet = false;
		LOG("<FAILED> Saving ohc file: %s\n", fname);
	}
	LOG("%s => %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
	return bRet;
}

bool Openholo::loadAsOhc(const char * fname)
{
	auto begin = CUR_TIME;

	std::string fullname = fname;
	if (!checkExtension(fname, ".ohc")) fullname.append(".ohc");
	OHC_decoder->setFileName(fullname.c_str());
	if (!OHC_decoder->load())
	{
		LOG("<FAILED> Load ohc : %s\n", fname);
		LOG("%.5lf (sec)\n", ELAPSED_TIME(begin, CUR_TIME));
		return false;
	}
	context_.waveNum = OHC_decoder->getNumOfWavlen();
	context_.pixel_number = OHC_decoder->getNumOfPixel();
	context_.pixel_pitch = OHC_decoder->getPixelPitch();

	vector<Real> wavelengthArray;
	OHC_decoder->getWavelength(wavelengthArray);
	size_t nWave = wavelengthArray.size();
	if (nWave < 1)
	{
		LOG("<FAILED> Do not load wavelength size.\n");
		return false;
	}

	context_.wave_length = new Real[nWave];
	for (int i = 0; i < nWave; i++)
		context_.wave_length[i] = wavelengthArray[i];

	OHC_decoder->getComplexFieldData(&complex_H);

	context_.k = (2 * M_PI) / context_.wave_length[0];
	context_.ss[_X] = context_.pixel_number[_X] * context_.pixel_pitch[_X];
	context_.ss[_Y] = context_.pixel_number[_Y] * context_.pixel_pitch[_Y];

	LOG("%s => %.5lf (sec)\n", __FUNCTION__, ELAPSED_TIME(begin, CUR_TIME));
	return true;
}


uchar* Openholo::loadAsImg(const char * fname)
{
	FILE *infile = fopen(fname, "rb");
	if (infile == nullptr)
	{ 
		LOG("<FAILED> No such file.\n");
		return nullptr;
	}

	// BMP Header Information
	fileheader hf;
	bitmapinfoheader hInfo;
	size_t nRead = fread(&hf, sizeof(fileheader), 1, infile);
	if (hf.signature[0] != 'B' || hf.signature[1] != 'M')
	{
		LOG("<FAILED> Not BMP file.\n");
		fclose(infile);
		return nullptr;
	}

	nRead = fread(&hInfo, sizeof(bitmapinfoheader), 1, infile);
	fseek(infile, hf.fileoffset_to_pixelarray, SEEK_SET);

	uint size = hInfo.imagesize != 0 ? hInfo.imagesize : (((hInfo.width * hInfo.bitsperpixel >> 3) + 3) & ~3) * hInfo.height;

	oph::uchar* img_tmp = new uchar[size];
	nRead = fread(img_tmp, sizeof(uchar), size, infile);
	fclose(infile);

	return img_tmp;
}

bool Openholo::loadAsImgUpSideDown(const char * fname, uchar* dst)
{
	FILE *infile = fopen(fname, "rb");

	if (infile == nullptr)
	{
		LOG("<FAILED> No such file.\n");
		return false;
	}

	// BMP Header Information
	fileheader hf;
	bitmapinfoheader hInfo;
	size_t nRead = fread(&hf, sizeof(fileheader), 1, infile);
	if (hf.signature[0] != 'B' || hf.signature[1] != 'M')
	{
		LOG("<FAILED> Not BMP file.\n");
		return false; 
	}

	nRead = fread(&hInfo, sizeof(bitmapinfoheader), 1, infile);
	fseek(infile, hf.fileoffset_to_pixelarray, SEEK_SET);
	
	oph::uchar* img_tmp;
	if (hInfo.imagesize == 0) {
		img_tmp = new oph::uchar[hInfo.width*hInfo.height*(hInfo.bitsperpixel >> 3)];
		nRead = fread(img_tmp, sizeof(oph::uchar), hInfo.width*hInfo.height*(hInfo.bitsperpixel >> 3), infile);
	}
	else {
		img_tmp = new oph::uchar[hInfo.imagesize];
		nRead = fread(img_tmp, sizeof(oph::uchar), hInfo.imagesize, infile);
	}
	fclose(infile);

	// data upside down
	uint bytesperpixel = hInfo.bitsperpixel >> 3;
	uint cRow = ((hInfo.width * bytesperpixel) + 3) & ~3;
	uint cImg = hInfo.height * cRow;

	for (oph::uint k = 0; k < cImg; k++) {
		uint r = k / cRow;
		uint c = k % cRow;
		((oph::uchar*)dst)[(hInfo.height - r - 1) * cRow + c] = img_tmp[r * cRow + c];
	}

	delete[] img_tmp;
	return true;
}

bool Openholo::getImgSize(int & w, int & h, int & bytesperpixel, const char * fname)
{
	char szExtension[FILENAME_MAX] = { 0, };

	strcpy(szExtension, strrchr(fname, '.') + 1);
	
#ifdef _MSC_VER
	if (_stricmp(szExtension, "bmp")) { // not bmp
#elif __GNUC__
	if (strcasecmp(szExtension, "bmp")) {
#endif
		LOG("<FAILED> Not BMP file.\n");
		return false;
	}
	// BMP
	FILE *infile = fopen(fname, "rb");

	if (infile == nullptr) { LOG("<FAILED> Load image file.\n"); return false; }

	// BMP Header Information
	fileheader hf;
	bitmapinfoheader hInfo;
	size_t nRead = fread(&hf, sizeof(fileheader), 1, infile);
	if (hf.signature[0] != 'B' || hf.signature[1] != 'M') return false;
	nRead = fread(&hInfo, sizeof(bitmapinfoheader), 1, infile);
	//if (hInfo.bitsperpixel != 8) { printf("Bad File Format!!"); return 0; }

	w = hInfo.width;
	h = hInfo.height;
	bytesperpixel = hInfo.bitsperpixel >> 3;
	fclose(infile);

	return true;
}

void Openholo::imgScaleBilinear(uchar* src, uchar* dst, int w, int h, int neww, int newh, int channels)
{
	int channel = channels;
	int nBytePerLine = ((w * channel) + 3) & ~3;
	int nNewBytePerLine = ((neww * channel) + 3) & ~3;
#ifdef _OPENMP
#pragma omp parallel for firstprivate(nBytePerLine, nNewBytePerLine, w, h, neww, newh, channel)
#endif
	for (int y = 0; y < newh; y++)
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
}

void Openholo::convertToFormatGray8(unsigned char * src, unsigned char * dst, int w, int h, int bytesperpixel)
{
	int idx = 0;
	unsigned int r = 0, g = 0, b = 0;
	int N = (((w * bytesperpixel) + 3) & ~3) * h;

	for (int i = 0; i < N; i++)
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

	if (in == nullptr) {
		in = new Complex<Real>[pnx];
		fft_in = reinterpret_cast<fftw_complex*>(in);
		bIn = false;
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
	pnx = n[_X], pny = n[_Y];
	int N = pnx * pny;

	if (fft_in == nullptr)
		fft_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	if (fft_out == nullptr)
		fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

	if (in != nullptr)
	{
		fft_in = reinterpret_cast<fftw_complex*>(in);
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
	int size = pnx * pny * pnz;

	bool bIn = true;
	if (fft_in == nullptr)
		fft_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * size);
	if (fft_out == nullptr)
		fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * size);

	if (!in) {
		in = new Complex<Real>[size];
		bIn = false;
	}

	for (int i = 0; i < size; i++) {
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

	int size = pnx * pny * pnz;

	if (!bReverse) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < size; i++) {
			out[i][_RE] = fft_out[i][_RE];
			out[i][_IM] = fft_out[i][_IM];
		}
	}
	else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < size; i++) {
			out[i][_RE] = fft_out[i][_RE] / size;
			out[i][_IM] = fft_out[i][_IM] / size;
		}
	}

	fftFree();
}

void Openholo::fftInit2D(ivec2 size, int sign, unsigned int flag)
{
	int pnX = size[_X];
	int pnY = size[_Y];
	int N = pnX * pnY;

	if (fft_in == nullptr)
		fft_in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
	if (fft_out == nullptr)
		fft_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);

	if (plan_fwd == nullptr)
		plan_fwd = fftw_plan_dft_2d(pnY, pnX, fft_in, fft_out, sign, flag);
	if (plan_bwd == nullptr)
		plan_bwd = fftw_plan_dft_2d(pnY, pnX, fft_in, fft_out, sign, flag);
}

void Openholo::fftFree(void)
{
	if (plan_fwd) {
		fftw_destroy_plan(plan_fwd);
		plan_fwd = nullptr;
	}
	if (plan_bwd) {
		fftw_destroy_plan(plan_bwd);
		plan_bwd = nullptr;
	}
	fftw_free(fft_in);
	fftw_free(fft_out);

	fft_in = nullptr;
	fft_out = nullptr;

	pnx = 1;
	pny = 1;
	pnz = 1;
}

void Openholo::fft2(Complex<Real>* src, Complex<Real>* dst, int nx, int ny, int type, bool bNormalized, bool bShift)
{
	const int N = nx * ny;
	fftw_complex *in, *out;
	const bool bIn = fft_in == nullptr ? true : false;
	const bool bOut = fft_out == nullptr ? true : false;

	if (bIn)
		in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
	else
		in = fft_in;

	if (bOut)
		out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
	else
		out = fft_out;

	if (bShift)
		fftShift(nx, ny, src, reinterpret_cast<Complex<Real> *>(in));
	else
		memcpy(in, src, sizeof(Complex<Real>) * N);

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

	if (bNormalized)
	{
#pragma omp parallel for
		for (int k = 0; k < N; k++) {
			out[k][_RE] /= N;
			out[k][_IM] /= N;
		}
	}
	if (plan)
		fftw_destroy_plan(plan);

	if (bShift)
		fftShift(nx, ny, reinterpret_cast<Complex<Real> *>(out), dst);
	else
		memcpy(dst, out, sizeof(Complex<Real>) * N);

	if (bIn)
		fftw_free(in);
	if (bOut)
		fftw_free(out);
}


void Openholo::fftShift(int nx, int ny, Complex<Real>* input, Complex<Real>* output)
{
	int hnx = nx >> 1;
	int hny = ny >> 1;

#ifdef _OPENMP
#pragma omp parallel for firstprivate(hnx, hny)
#endif
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			int ti = i - hnx; 
			int tj = j - hny; 
			if (ti < 0) ti += nx;
			if (tj < 0) tj += ny;
			output[ti + tj * nx] = input[i + j * nx];
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

void Openholo::setMaxThreadNum(int num)
{
#ifdef _OPENMP
	if (num > omp_get_max_threads())
		omp_set_num_threads(omp_get_max_threads());
	else
		omp_set_num_threads(num);
#else
	LOG("Not used openMP\n");
#endif
}

int Openholo::getMaxThreadNum()
{
	int num_threads = 1;
#ifdef _OPENMP
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
#endif
	return num_threads;
}

void Openholo::ophFree(void)
{
	uint nWave = context_.waveNum;

	for (uint i = 0; i < nWave; i++) {
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

