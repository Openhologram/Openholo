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


#ifndef __Openholo_h
#define __Openholo_h

#include "Base.h"
#include "include.h"
#include "vec.h"
#include "ivec.h"
#include "fftw3.h"
#include "ImgCodecOhc.h"

using namespace oph;

struct OPH_DLL ResolutionConfig
{
	int double_pnX;
	int double_pnY;
	int pnX;
	int pnY;
	int half_pnX;
	int half_pnY;
	long long int size;
	long long int half_size;
	long long int double_size;

	ResolutionConfig() {
		this->pnX = 0;
		this->pnY = 0;
		this->half_pnX = 0;
		this->half_pnY = 0;
		this->double_pnX = 01;
		this->double_pnY = 01;
		this->size = 0;
		this->half_size = 0;
		this->double_size = 0;
	}

	ResolutionConfig(const ivec2& pixel_number) {
		this->pnX = pixel_number[_X];
		this->pnY = pixel_number[_Y];
		this->half_pnX = pnX >> 1;
		this->half_pnY = pnY >> 1;
		this->double_pnX = pnX << 1;
		this->double_pnY = pnY << 1;
		this->size = pnX * pnY;
		this->half_size = size >> 2;
		this->double_size = size << 2;
	}
};

struct OPH_DLL OphConfig
{
	bool			bUseDP;						// use double precision
	ivec2			pixel_number;				// SLM_PIXEL_NUMBER_X & SLM_PIXEL_NUMBER_Y
	ivec2			offset;						// start offset
	vec2			pixel_pitch;				// SLM_PIXEL_PITCH_X & SLM_PIXEL_PITCH_Y
	vec3			shift;						// shift
	Real			k;							// 2 * PI / lambda(wavelength)
	vec2			ss;							// pn * pp
	uint			waveNum;					// wave num
	Real*			wave_length;				// wave length

	//basic constructor
	OphConfig() {
		this->bUseDP = true;
		this->pixel_number = ivec2();
		this->offset = ivec2();
		this->pixel_pitch = vec2();
		this->shift = vec3();
		this->k = 0;
		this->ss = vec2();
		this->waveNum = 0;
		this->wave_length = 0;
	}
};

struct OPH_DLL ImageConfig
{
	bool		rotate;							// rotat
	bool		merge;							// merge
	int			flip;							// flip


	//basic constructor
	ImageConfig() {
		this->rotate = false;
		this->merge = false;
		this->flip = 0;
	}
};



/**
* @ingroup oph
* @brief Abstract class
* @details Top class of Openholo library. Common functions required by subclasses are implemented.
* @author Kim Ryeon-woo, Nam Min-woo
*/
class OPH_DLL Openholo : public Base{

public:
	/**
	* @brief Constructor
	*/
	explicit Openholo(void);

protected:
	/**
	* @brief Destructor
	* @details Pure virtual function for class abstraction
	*/
	virtual ~Openholo(void) = 0;

protected:
	/**
	* @brief Functions for extension checking
	* @param[in] fname File name
	* @param[in] ext File extension
	* @return Type: <B>bool</B>\n
	*				If fname contains ext, the return value is <B>true</B>.\n
	*				If fname not contains ext, the return value is <B>false</B>.
	*/
	bool checkExtension(const char* fname, const char* ext);

public:
	/**
	* @brief Function for creating image files
	* @param[in] fname Output file name
	* @param[in] bitsperpixel Bit per pixel
	* @param[in] src Source of Image file's data
	* @param[in] width Number of pixel - width
	* @param[in] height Number of pixel - height
	* @return Type: <B>bool</B>\n
	*				If the succeeds to save image file, the return value is <B>true</B>.\n
	*				If the fails to save image file, the return value is <B>false</B>.
	*/
	virtual bool saveAsImg(const char* fname, uint8_t bitsperpixel, uchar* src, int width, int height);

	/**
	* @brief Function for loading image files
	* @param[in] fname Input file name
	* @return Type: <B>uchar*</B>\n
	*				If the succeeds to load image file, the return value is <B>image data' pointer</B>.\n
	*				If the fails to load image file, the return value is <B>nullptr</B>.
	*/
	virtual uchar* loadAsImg(const char* fname);

	/**
	* @brief Function to write OHC file	
	* @param[in] fname File name
	* @return Type: <B>bool</B>\n
	*				If the succeeds to save OHC file, the return value is <B>true</B>.\n
	*				If the fails to save OHC file, the return value is <B>false</B>.
	*/
	virtual bool saveAsOhc(const char *fname);


	/**
	* @brief Function to read OHC file
	* @param[in] fname File name
	* @return Type: <B>bool</B>\n
	*				If the succeeds to load OHC file, the return value is <B>true</B>.\n
	*				If the fails to load OHC file, the return value is <B>false</B>.
	*/
	virtual bool loadAsOhc(const char *fname);

	
	/**
	* @brief Function for getting the complex field
	* @return Type: <B>Complex<Real>**</B>\n
	*				If the succeeds to get complex field, the return value is <B>complex field data's pointer</B>.\n
	*				If the fails to get complex field, the return value is <B>nullptr</B>.
	*/
	inline Complex<Real>** getComplexField(void) { return complex_H; }
	
	
	/**
	* @brief Function for getting the current context
	* @return Type: <B>OphConfig&</B>\n
	*				If the succeeds to get context, the return value is <B>cotext pointer</B>.\n
	*				If the fails to get context, the return value is <B>nullptr</B>.
	*/
	OphConfig& getContext(void) { return context_; }

	/**
	* @brief Function for getting the image config
	* @return Type: <B>ImageConfig</B>\n
	*				The return value is <B>Image config pointer</B>.\n
	*/
	ImageConfig& getImageConfig() { return imgCfg; }

	/**
	* @brief Function for setting the output resolution
	* @param[in] n resolution vector value.
	*/
	inline void setPixelNumber(ivec2 n) { 
		context_.pixel_number[_X] = n[_X]; context_.pixel_number[_Y] = n[_Y];
		context_.ss = context_.pixel_number * context_.pixel_pitch;
	}
	inline void setPixelNumber(int width, int height) { 
		context_.pixel_number[_X] = width; context_.pixel_number[_Y] = height;
		context_.ss = context_.pixel_number * context_.pixel_pitch;
	}

	/**
	* @brief Function for setting the output pixel pitch
	* @param[in] p pitch vector value.
	*/
	inline void setPixelPitch(vec2 p) { 
		context_.pixel_pitch[_X] = p[_X]; context_.pixel_pitch[_Y] = p[_Y];
		context_.ss = context_.pixel_number * context_.pixel_pitch;
	}
	inline void setPixelPitch(Real pitchX, Real pitchY) { 
		context_.pixel_pitch[_X] = pitchX; context_.pixel_pitch[_Y] = pitchY;
		context_.ss = context_.pixel_number * context_.pixel_pitch;
	}
	
	/**
	* @brief Function for setting the wave length
	* @param[in] w wave length.
	* @param[in] idx index of channel.
	*/
	inline void setWaveLength(Real w, const uint idx = 0) { context_.wave_length[idx] = w; }

	/**
	* @brief Function for setting the wave number
	* @param[in] num Number of wave/
	*/
	void setWaveNum(int num);

	/**
	* @brief Function for setting the offset
	* @param[in] offset spatial domain offset value.
	*/
	inline void setOffset(ivec2 offset) { context_.offset[_X] = offset[_X]; context_.offset[_Y] = offset[_Y]; }

	/**
	* @brief Function for setting the image merge(true or false)
	* @details <pre>
	if merge == true
	One RGB image
	else
	Each grayscale image </pre>
	* @param[in] merge : the value for specifying whether output image merge.
	*/
	void setImageMerge(bool merge) { imgCfg.merge = merge; }

	/**
	* @brief Function for setting the image rotate(true or false)
	* @details <pre>
	if rotate == true
	180 degree rotate.
	else
	not change. </pre>
	* @param[in] rotate : the value for specifying whether output image rotate.
	*/
	void setImageRotate(bool rotate) { imgCfg.rotate = rotate; }

	/**
	* @brief Function for setting the image flip
	* @details <pre>
	if flip == 0
	not change.
	else if flip == 1
	vertical change.
	else if flipp == 2
	horizon tal change.
	else B
	both.</pre>
	* @param[in] rotate : the value for specifying whether output image flip.
	*/
	void setImageFlip(int flip) { imgCfg.flip = flip; }

	/**
	* @brief Function for setting the max thread num
	* @param[in] num : number of max thread.
	*/
	void setMaxThreadNum(int num);

	/**
	* @brief Function for getting the max thread num
	* @return Type: <B>int</B>\n
					If the currently max thread.
	*/
	int getMaxThreadNum();

	/**
	* @brief Function for generate RGB image from each grayscale image
	* @param[in] idx : 0 is red, 1 is green, 2 is blue.
	* @param[in] width : Number of pixel - width
	* @param[in] height : Number of pixel - height
	* @param[in] src : Source of Image file's data - grayscale
	* @param[out] dst : Destination of Image file's data - RGB
	* @return Type: <B>bool</B>\n
	*				If the succeeds to merge image, the return value is <B>true</B>.\n
	*				If the fails to merge image, the return value is <B>false</B>.
	*/
	bool mergeColor(int idx, int width, int height, uchar *src, uchar *dst);

	/**
	* @brief Function for generate each grayscale image from RGB image
	* @param[in] idx : 0 is red, 1 is green, 2 is blue.
	* @param[in] width : Number of pixel - width
	* @param[in] height : Number of pixel - height
	* @param[in] src : Source of Image file's data - RGB
	* @param[out] dst : Destination of Image file's data - grayscale
	* @return Type: <B>bool</B>\n
	*				If the succeeds to separate image, the return value is <B>true</B>.\n
	*				If the fails to separate image, the return value is <B>false</B>.
	*/
	bool separateColor(int idx, int width, int height, uchar *src, uchar *dst);

protected:

	/**
	* @brief Function for loading image files | Output image data upside down
	* @param[in] Input file name.
	* @param[out] destination to load.
	* @return Type: <B>bool</B>\n
	*				If the function succeeds, the return value is <B>true</B>.\n
	*				If the function fails, the return value is <B>false</B>.
	*/
	bool loadAsImgUpSideDown(const char* fname, uchar* dst);

	/**
	* @brief Function for getting the image size
	* @param[out] w Image size - width.
	* @param[out] h Image size - Height.
	* @param[out] bytesperpixel Bytes per pixel.
	* @param[in] fname Input file name.
	* @return Type: <B>bool</B>\n
	*				If the function succeeds, the return value is <B>true</B>.\n
	*				If the function fails, the return value is <B>false</B>.
	*/
	bool getImgSize(int& w, int& h, int& bytesperpixel, const char* fname);

	/**
	* @brief Function for change image size
	* @param[in] src Source image data.
	* @param[out] dst Destination image data.
	* @param[in] w Original width.
	* @param[in] h Original height.
	* @param[in] neww Width to replace.
	* @param[in] newh Height to replace.
	*/
	void imgScaleBilinear(uchar* src, uchar* dst, int w, int h, int neww, int newh, int channels = 1);

	/**
	* @brief Function for convert image format to gray8
	* @param[in] src Source image data.
	* @param[out] dst Destination image data.
	* @param[in] w Image size, width.
	* @param[in] h Image size, Height.
	* @param[in] bytesperpixel Bytes per pixel.
	*/
	void convertToFormatGray8(uchar* src, uchar* dst, int w, int h, int bytesperpixel);
	
	/**
	* @brief Functions for performing fftw 1-dimension operations inside Openholo
	* @param[in] n Number of data.
	* @param[in] in Source of data.
	* @param[in] sign Sign of FFTW(FORWARD or BACKWARD)
	* @param[in] flag Flag of FFTW(MEASURE, DESTROY_INPUT, UNALIGNED, CONSERVE_MEMORY, EXHAUSTIVE, PRESERVE_INPUT, PATIENT, ESTIMATE, WISDOM_ONLY)
	*/
	void fft1(int n, Complex<Real>* in, int sign = OPH_FORWARD, uint flag = OPH_ESTIMATE);
	
	/**
	* @brief Functions for performing fftw 2-dimension operations inside Openholo
	* @param[in] n Number of data(int x, int y)
	* @param[in] in Source of data.
	* @param[in] sign Sign of FFTW(FORWARD or BACKWARD)
	* @param[in] flag Flag of FFTW(MEASURE, DESTROY_INPUT, UNALIGNED, CONSERVE_MEMORY, EXHAUSTIVE, PRESERVE_INPUT, PATIENT, ESTIMATE, WISDOM_ONLY)
	*/
	void fft2(ivec2 n, Complex<Real>* in, int sign = OPH_FORWARD, uint flag = OPH_ESTIMATE);
	/**
	* @brief Functions for performing fftw 3-dimension operations inside Openholo
	* @param[in] n Number of data(int x, int y, int z)
	* @param[in] in Source of data.
	* @param[in] sign Sign of FFTW(FORWARD or BACKWARD)
	* @param[in] flag Flag of FFTW(MEASURE, DESTROY_INPUT, UNALIGNED, CONSERVE_MEMORY, EXHAUSTIVE, PRESERVE_INPUT, PATIENT, ESTIMATE, WISDOM_ONLY)
	*/
	void fft3(ivec3 n, Complex<Real>* in, int sign = OPH_FORWARD, uint flag = OPH_ESTIMATE);

	/**
	* @brief Execution functions to be called after fft1, fft2, and fft3
	* @param[out] out Dest of data.
	*/
	void fftExecute(Complex<Real>* out, bool bReverse = false);

	/**
	* @brief Resource release method.
	*/
	void fftFree(void);

	/**
	* @brief initialize method for 2D FFT
	* @param[in] size Number of data (int x, int y)
	* @param[in] sign Sign of FFTW(FORWARD or BACKWARD)
	* @param[in] flag Flag of FFTW(MEASURE, DESTROY_INPUT, UNALIGNED, CONSERVE_MEMORY, EXHAUSTIVE, PRESERVE_INPUT, PATIENT, ESTIMATE, WISDOM_ONLY) 
	*/
	void fftInit2D(ivec2 size, int sign, unsigned int flag);

	/**
	* @brief Convert data from the spatial domain to the frequency domain using 2D FFT on CPU.
	* @param[in] src Input data variable.
	* @param[out] dst Output data variable.
	* @param[in] nx the number of column of the input data.
	* @param[in] ny the number of row of the input data.
	* @param[in] type If type == 1, forward FFT, if type == -1, backward FFT.
	* @param[in] bNormalized If bNomarlized == true, normalize the result after FFT.
	* @param[in] bShift If bShift == true, shifts the input and output values before and after the FFT.
	*/
	void fft2(Complex<Real>* src, Complex<Real>* dst, int nx, int ny, int type, bool bNormalized = false, bool bShift = true);

	/**
	* @brief Swap the top-left quadrant of data with the bottom-right , and the top-right quadrant with the bottom-left.
	* @param[in] nx the number of column of the input data.
	* @param[in] ny the number of row of the input data.
	* @param[in] input input data variable.
	* @param[out] output output data variable.
	*/
	void fftShift(int nx, int ny, Complex<Real>* input, Complex<Real>* output);


protected:
	/**
	* @brief Pure virtual function for override in child classes
	*/
	virtual void ophFree(void);

private:
	/**
	* @brief fftw-library variables for running fft inside Openholo
	*/
	fftw_plan plan_fwd, plan_bwd;
	fftw_complex *fft_in, *fft_out;
	int pnx, pny, pnz;
	int fft_sign;
protected:
	OphConfig context_;
	ResolutionConfig resCfg;
	ImageConfig imgCfg;

	Complex<Real>** complex_H;
protected:
	/**
	* @brief OHC file format Variables for read and write
	*/
	ImgEncoderOhc* OHC_encoder;
	ImgDecoderOhc* OHC_decoder;

protected:
	/**
	* @brief getter/setter for OHC file read and write
	*/
	inline void setPixelNumberOHC(const ivec2 pixel_number) 
		{ OHC_encoder->setNumOfPixel(pixel_number); }

	inline void setPixelPitchOHC(const vec2 pixel_pitch)
		{ OHC_encoder->setPixelPitch(pixel_pitch); }

	inline void setWavelengthOHC(const Real wavelength, const LenUnit wavelength_unit)
		{ OHC_encoder->setWavelength(wavelength, wavelength_unit); }

	inline void setWaveLengthNumOHC(const uint wavelength_num)
		{ OHC_encoder->setNumOfWavlen(wavelength_num); }

	inline void setColorTypeOHC(const ColorType color_type)
		{ OHC_encoder->setColorType(color_type); }

	inline void setColorArrangeOHC(const ColorArran color_arrange)
		{ OHC_encoder->setColorArrange(color_arrange);	}

	inline 	void setWaveLengthUnitOHC(const LenUnit length_unit)
		{ OHC_encoder->setUnitOfWavlen(length_unit);	}

	inline 	void setFieldEncodingOHC(const FldStore field_store, const FldCodeType field_code_type)
		{ OHC_encoder->setFieldEncoding(field_store, field_code_type); }

	inline 	void setPhaseEncodingOHC(const BPhaseCode phase_code, const vec2 phase_code_range)
		{ OHC_encoder->setPhaseEncoding(phase_code, phase_code_range); }
	/**
	* @brief Function to add ComplexField when adding wavelength data
	*/
	inline void addWaveLengthNComplexFieldDataOHC(const Real wavelength, const OphComplexField& complex_field)
		{ OHC_encoder->addWavelengthNComplexFieldData(wavelength, complex_field); }

	inline void addWaveLengthOHC(const Real wavelength)
		{ OHC_encoder->addWavelength(wavelength); }

	inline void addComplexFieldDataOHC(const OphComplexField& complex_field)
		{ OHC_encoder->addComplexFieldData(complex_field); }

	inline void getPixelNumberOHC(ivec2& pixel_number)
		{ pixel_number = OHC_decoder->getNumOfPixel(); }

	inline void getPixelPitchOHC(vec2& pixel_pitch)
		{ pixel_pitch = OHC_decoder->getPixelPitch(); }

	inline void getWavelengthOHC(vector<Real>& wavelength)
		{ OHC_decoder->getWavelength(wavelength); }

	inline void getWaveLengthNumOHC(uint& wavelength_num)
		{ wavelength_num = OHC_decoder->getNumOfWavlen(); }

	inline void getColorTypeOHC(ColorType& color_type)
		{ color_type = OHC_decoder->getColorType(); }

	inline void getColorArrangeOHC(ColorArran& color_arrange)
		{ color_arrange = OHC_decoder->getColorArrange(); }

	inline 	void getWaveLengthUnitOHC(LenUnit& length_unit)
		{ length_unit = OHC_decoder->getUnitOfWavlen(); }

	inline void getComplexFieldDataOHC(Complex<Real>** cmplx, uint wavelen_idx)
		{ OHC_decoder->getComplexFieldData(cmplx, wavelen_idx); }

	inline void getComplexFieldDataOHC(OphComplexField& cmplx, uint wavelen_idx)
		{ OHC_decoder->getComplexFieldData(cmplx, wavelen_idx); }	
};

#endif // !__Openholo_h