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

/**
* @mainpage Openholo library Documentation
* @section Introduction

OpenHolo is an open source library which contains algorithms and their software implementation
for generation of holograms to be applied in various fields. The goal behind the library development
is facilitating production of digital holographic contents and expanding the area of their application.
The developed by us open source library is a tool for computer generation of holograms, simulations and
signal processing at various formats of 3D input data and properties of the 3D displays. Based on this,
we want to lay the foundation for commercializing digital holographic service in various fields.


* @section Examples

Generation Hologram - Point Cloud Example

@code
	#include "ophPointCloud.h"

	ophPointCloud* Hologram = new ophPointCloud();

	Hologram->setMode(MODE_CPU); //Select CPU or GPU Processing
	Hologram->readConfig("config/TestSpecPointCloud.xml");
	Hologram->loadPointCloud("source/TestPointCloud_Saturn.ply");

	Hologram->generateHologram(PC_DIFF_RS_NOT_ENCODED);
	Hologram->encodeHologram();
	Hologram->normalize();
	Hologram->save("result/Result_PointCloudSample_Saturn");
	Hologram->release();
@endcode


Generation Hologram - Depth Map Example

@code
	#include "ophDepthMap.h"

	ophDepthMap* Hologram = new ophDepthMap();

	Hologram->setMode(MODE_CPU); //Select CPU or GPU Processing
	Hologram->readConfig("config/TestSpecDepthMap.xml");
	Hologram->readImageDepth("source", "RGB_D", "D_D");

	Hologram->generateHologram();
	Hologram->encodeHologram();
	Hologram->normalize();
	Hologram->save("result/Result_DepthmapSample.bmp");
	Hologram->release();
@endcode


Generation Hologram - Triangle Mesh Example

@code
	#include "ophTriMesh.h"

	ophTri* Hologram = new ophTri();

	Hologram->readMeshConfig("config/TestSpecMesh.xml");
	Hologram->loadMeshData("source/mesh_teapot.ply","ply");
	Hologram->objScaleShift();

	Hologram->generateMeshHologram(Hologram->SHADING_FLAT);
	Hologram->encoding(Hologram->ENCODE_AMPLITUDE);
	Hologram->normalizeEncoded();
	ivec2 encode_size = Hologram->getEncodeSize();
	Hologram->save("result/Mesh.bmp", 8, nullptr, encode_size[_X], encode_size[_Y]);
@endcode


Generation Hologram - Light Field Example

@code
	#include "ophLightField.h"

	ophLF* Hologram = new ophLF();

	Hologram->readLFConfig("config/TestSpecLF.xml");
	Hologram->loadLF("source/sample_orthographic_images","bmp");

	Hologram->generateHologram();
	Hologram->encoding(Hologram->ENCODE_AMPLITUDE);
	Hologram->normalizeEncoded();
	ivec2 encode_size = Hologram->getEncodeSize();
	Hologram->save("result/Light_Field.bmp", 8, nullptr, encode_size[_X], encode_size[_Y]);
@endcode


Generation Hologram - Wavefront Recording Plane(WRP) Example

@code
	#include "ophWRP.h"

	ophWRP* Hologram = new ophWRP();

	Hologram->readConfig("config/TestSpecWRP.xml");
	Hologram->loadPointCloud("source/TestPointCloud_WRP.ply");
	Hologram->calculateMWRP();

	Hologram->generateHologram();
	Hologram->encodeHologram();
	Hologram->normalize();
	Hologram->save("result/Result_WRP.bmp");
	Hologram->release();
@endcode


Encoding Example

@code
	#include "ophPointCloud.h"

	ophPointCloud* Hologram = new ophPointCloud();

	Hologram->loadComplex("source/teapot_real_1920,1080.txt", "source/teapot_imag_1920,1080.txt", 1920, 1080);
	Hologram->encoding(ophGen::ENCODE_AMPLITUDE);
	Hologram->normalizeEncoded();
	ivec2 encode_size = Hologram->getEncodeSize();
	Hologram->save("result/Encoding.bmp",8,nullptr,encode_size[_X], encode_size[_Y]);
@endcode


Wave Aberration Example

@code
	#include "ophWaveAberration.h"

	ophWaveAberration* wa = new ophWaveAberration;

	wa->readConfig("config/TestSpecAberration.xml"); // reads parameters from a configuration file
	wa->accumulateZernikePolynomial(); // generates 2D complex data array of wave aberration according to parameters
	wa->complex_W; // double pointer variable of 2D complex data array of wave aberration
	wa->resolutionX; // resolution in x axis of 2D complex data array of wave aberration
	wa->resolutionY; // resolution in y axis of 2D complex data array of wave aberration
	wa->saveAberration("result/aberration.bin"); // saves 2D complex data array of complex wave aberration into a file

	wa->readAberration("result/aberration.bin"); // reads 2D complex data array of complex wave aberration from a file
	wa->complex_W; // double pointer variable of 2D complex data array of wave aberration
	wa->resolutionX; // resolution in x axis of 2D complex data array of wave aberration
	wa->resolutionY; // resolution in y axis of 2D complex data array of wave aberration
	wa->release();
@endcode


Hologram core processing - HPO transform Example

@code
	#include "ophSig.h"

	ophSig *holo = new ophSig();

	if (!holo->readConfig("config/holoParam.xml")) {
		// no file
		return false;
	}

	if (!holo->load("source/3_point_re.bmp", "source/3_point_im.bmp", 8)) {
		// no file
		return false;
	}

	holo->sigConvertHPO();
	holo->save("result/HPO_re_C.bmp", "result/HPO_im_C.bmp", 8);
@endcode


Hologram core processing - CAC transform Example

@code
	#include "ophSig.h"

	ophSig *holo = new ophSig();

	if (!holo->readConfig("config/holoParam.xml")) {
		// no file
		return false;
	}

	if (!holo->load("source/ColorPoint_re.bmp", "source/ColorPoint_im.bmp",24)) {
		// no file
		return false;
	}

	holo->sigConvertCAC(0.000000633,0.000000532,0.000000473);
	holo->save("result/CAC_re_C.bin", "result/CAC_im_C.bin",24);
@endcode


Hologram core processing - Off-axis hologram transform Example

@code
	#include "ophSig.h"

	ophSig *holo = new ophSig();

	if (!holo->readConfig("config/holoParam.xml")) {
		// no file
		return false;
	}

	if (!holo->load("source/3_point_re.bmp", "source/3_point_im.bmp", 8)) {
		// no file
		return false;
	}

	holo->sigConvertOffaxis();
	holo->save("result/Off_axis.bmp",8);
@endcode


Hologram core processing - get parameter using axis transformation Example

@code
	#include "ophSig.h"

	ophSig* holo = new ophSig();

	float depth = 0;

	if (!holo->readConfig("config/holoParam.xml")) {
		// no file
		return false;
	}

	if (!holo->load("source/3_point_re.bmp", "source/3_point_im.bmp", 8)) {
		// no file
		return false;
	}

	depth = holo->sigGetParamSF(10, -10, 100, 0.3);
	holo->propagationHolo(depth); // backpropagation
	holo->save("result/SF_re.bmp", "result/SF_im.bmp", 8);
@endcode


@code
	#include "ophSig.h"

	ophSig* holo = new ophSig();

	float depth = 0;

	if (!holo->readConfig("config/holoParam.xml")) {
		// no file
		return false;
	}

	if (!holo->load("source/0.1point_re.bmp", "source/0.1point_im.bmp", 8)) {
		// no file
		return false;
	}

	depth = holo->sigGetParamAT();
	holo->propagationHolo(-depth); // backpropagation
	holo->save("result/AT_re.bmp", "result/AT_im.bmp", 8);
@endcode


Cascaded Propagation Example

@code
	#include "ophCascadedPropagation.h"

	ophCascadedPropagation* pCp = new ophCascadedPropagation(L"config/TestSpecCascadedPropagation.xml");

	if (pCp->propagate())
		pCp->saveIntensityAsImg(L"result/intensityRGB.bmp", pCp->getNumColors() * 8);

	pCp->release();
@endcode

*
*/


/**
* \defgroup const,dest Constructor & Destructor
* \defgroup oper Operator
* \defgroup get,set Parameters
* \defgroup init Initialize
* \defgroup calc Calculate
* \defgroup gen Generate Hologram
* \defgroup reconstruct Reconstruct Hologram
* \defgroup signal Signal Processing
* \defgroup encode Encoding
* \defgroup read Read Data
* \defgroup write Write Data
*/

#ifndef __Openholo_h
#define __Openholo_h

#include "Base.h"
#include "include.h"
#include "vec.h"
#include "ivec.h"
#include "fftw3.h"

#include "ImgCodecOhc.h"

using namespace oph;


struct OphConfig
{
	oph::ivec2		pixel_number;				//< SLM_PIXEL_NUMBER_X & SLM_PIXEL_NUMBER_Y
	oph::vec2		pixel_pitch;				//< SLM_PIXEL_PITCH_X & SLM_PIXEL_PITCH_Y

	Real			k;							//< 2 * PI / lambda(wavelength)
	vec2			ss;							//< pn * pp

	Real*			wave_length;				//< wave length
};

//namespace oph{
//	class ImgEncoderOhc;
//	class ImgDecoderOhc;
//	enum class LenUnit : uint8_t;
//	enum class ColorType : uint8_t;
//	enum class ColorArran : uint8_t;
//	enum class DataType : uint8_t;
//	enum class FldStore : uint8_t;
//	enum class FldCodeType : uint8_t;
//	enum class BPhaseCode : uint8_t;
//	enum class ImageFormat : uint8_t;
//}
/**
* @brief Abstract class
* @detail Top class of Openholo library. Common functions required by subclasses are implemented.
*/
class OPH_DLL Openholo : public Base{

public:
	/**
	* \ingroup const,dest
	* @brief Constructor
	*/
	explicit Openholo(void);

protected:
	/**
	* \ingroup const,dest
	* @brief Destructor
	* @detail Pure virtual function for class abstraction
	*/
	virtual ~Openholo(void) = 0;

protected:
	/**
	* @brief Functions for extension checking
	* @param const char* File name
	* @param const char* File extension
	* @return int return 0 : The extension of "fname" and "ext" is the same
	*			  return 1 : The extension of "fname" and "ext" is not the same
	*/
	int checkExtension(const char* fname, const char* ext);

public:
	/**
	* \ingroup write
	* @brief Function for creating image files
	* @param const char* Output file name
	* @param uint8_t Bit per pixel
	* @param unsigned char* Source of Image file's data
	* @param int Number of pixel - width
	* @param int Number of pixel - height
	* @return int  return -1 : Failed to save image file
	*			   return  1 : Success to save image file
	*/
	virtual int saveAsImg(const char* fname, uint8_t bitsperpixel, uchar* src, int pic_width, int pic_height);

	/**
	* \ingroup read
	* @brief Function for loading image files
	* @param const char* Input file name
	* @return unsigned char* Image file's data
	*/
	virtual uchar* loadAsImg(const char* fname);

	/**
	* \ingroup write
	* @brief Function to write OHC file
	*/
	virtual int saveAsOhc(const char *fname);

	/**
	* \ingroup read
	* @brief Function to read OHC file
	*/
	virtual int loadAsOhc(const char *fname);

	/**
	* \ingroup get,set
	*/
	inline oph::Complex<Real>** getComplexField(void) { return complex_H; }

	/**
	* \ingroup get,set
	*/
	inline void setPixelNumber(ivec2 n) { context_.pixel_number[_X] = n[_X]; context_.pixel_number[_Y] = n[_Y]; }

	/**
	* \ingroup get,set
	*/
	inline void setPixelPitch(vec2 p) { context_.pixel_pitch[_X] = p[_X]; context_.pixel_pitch[_Y] = p[_Y]; }

	/**
	* \ingroup get,set
	*/
	inline void setWaveLength(Real w, const uint idx) { context_.wave_length[idx] = w; }

	/**
	* \ingroup get,set
	*/
	OphConfig& getContext(void) { return context_; }
protected:
	/**
	* \ingroup read
	* @brief Function for loading image files | Output image data upside down
	* @param const char* Input file name
	* @return unsigned char* Image file's data
	*/
	int loadAsImgUpSideDown(const char* fname, uchar* dst);

	/**
	* \ingroup read
	* @brief Function for getting the image size
	* @param int& Image size - width
	* @param int& Image size - Height
	* @param int& Bytes per pixel
	* @param const char* Input file name
	*/
	int getImgSize(int& w, int& h, int& bytesperpixel, const char* file_name);

	/**
	* @brief Function for change image size
	* @param unsigned char* Source image data
	* @param unsigned char* Dest image data
	* @param int Original width
	* @param int Original height
	* @param int Width to replace
	* @param int Height to replace
	*/
	void imgScaleBilnear(unsigned char* src, unsigned char* dst, int w, int h, int neww, int newh);

	/**
	* @brief Function for convert image format to gray8
	* @param unsigned char* Source image data
	* @param unsigned char* Dest image data
	* @param int Image size, width
	* @param int Image size, Height
	* @param int Bytes per pixel
	*/
	void convertToFormatGray8(unsigned char* src, unsigned char* dst, int w, int h, int bytesperpixel);


	/**
	* \ingroup calc
	* @brief Functions for performing fftw 1-dimension operations inside Openholo
	* @param int Number of data
	* @param Complex<Real>* Source of data
	* @param int Sign of FFTW(FORWARD or BACKWARD)
	* @param unsigned int Flag of FFTW(MEASURE, DESTROY_INPUT, UNALIGNED, CONSERVE_MEMORY, EXHAUSTIVE, PRESERVE_INPUT, PATIENT, ESTIMATE, WISDOM_ONLY)
	*/
	void fft1(int n, Complex<Real>* in, int sign = OPH_FORWARD, uint flag = OPH_ESTIMATE);
	/**
	* \ingroup calc
	* @brief Functions for performing fftw 2-dimension operations inside Openholo
	* @param oph::ivec2 Number of data(int x, int y)
	* @param Complex<Real>* Source of data
	* @param int Sign of FFTW(FORWARD or BACKWARD)
	* @param unsigned int Flag of FFTW(MEASURE, DESTROY_INPUT, UNALIGNED, CONSERVE_MEMORY, EXHAUSTIVE, PRESERVE_INPUT, PATIENT, ESTIMATE, WISDOM_ONLY)
	*/
	void fft2(oph::ivec2 n, Complex<Real>* in, int sign = OPH_FORWARD, uint flag = OPH_ESTIMATE);
	/**
	* \ingroup calc
	* @brief Functions for performing fftw 3-dimension operations inside Openholo
	* @param oph::ivec3 Number of data(int x, int y, int z)
	* @param Complex<Real>* Source of data
	* @param int Sign of FFTW(FORWARD or BACKWARD)
	* @param unsigned int Flag of FFTW(MEASURE, DESTROY_INPUT, UNALIGNED, CONSERVE_MEMORY, EXHAUSTIVE, PRESERVE_INPUT, PATIENT, ESTIMATE, WISDOM_ONLY)
	*/
	void fft3(oph::ivec3 n, Complex<Real>* in, int sign = OPH_FORWARD, uint flag = OPH_ESTIMATE);

	/**
	* \ingroup calc
	* @brief Execution functions to be called after fft1, fft2, and fft3
	* @param Complex<Real>* Dest of data
	*/
	void fftExecute(Complex<Real>* out);
	void fftFree(void);
	/**
	* \ingroup calc
	* @brief Convert data from the spatial domain to the frequency domain using 2D FFT on CPU.
	* @param Complex<Real>* Input data variable
	* @param Complex<Real>* Output data variable
	* @param int the number of column of the input data
	* @param int the number of row of the input data
	* @param int If type == 1, forward FFT, if type == -1, backward FFT.
	* @param bool If bNomarlized == true, normalize the result after FFT.
	*/
	void fftwShift(Complex<Real>* src, Complex<Real>* dst, int nx, int ny, int type, bool bNormalized = false);

	/**
	* \ingroup calc
	* @brief Swap the top-left quadrant of data with the bottom-right , and the top-right quadrant with the bottom-left.
	* @param int the number of column of the input data
	* @param int the number of row of the input data
	* @param Complex<Real>* input data variable
	* @param Complex<Real>* output data variable
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

	//inline void setCompressedFormatType(const CompresType compress_type)
	//	{ OHC_encoder->setCompressedFormatType(compress_type); }

	/**
	* @brief Function to add ComplexField when adding wavelength data
	*/
	inline void addWaveLengthNComplexFieldDataOHC(const Real wavelength, const OphComplexField& complex_field)
		{ OHC_encoder->addWavelengthNComplexFieldData(wavelength, complex_field); }

	inline void addWaveLengthOHC(const Real wavelength)
		{ OHC_encoder->addWavelength(wavelength); }

	inline void addComplexFieldDataOHC(const OphComplexField& complex_field)
		{ OHC_encoder->addComplexFieldData(complex_field); }

	/**

	*/
	//inline void addLinkFilePath(const std::string& path)
	//	{ OHC_encoder->addLinkFilePath(path); }

	/**

	*/
	//inline void getLinkFilePath(std::vector<std::string> &linkFilePath_array)
	//	{ OHC_decoder->getLinkFilePath(linkFilePath_array); }

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

	//inline void getCompressedFormatType(const CompresType compress_type)
	//	{ OHC_encoder->setCompressedFormatType(compress_type); }


};

#endif // !__Openholo_h