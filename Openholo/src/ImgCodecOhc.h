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

#ifndef __ImgCodecOhc_h
#define __ImgCodecOhc_h

#include <fstream>
#include "include.h"
#include "mat.h"
#include "vec.h"

#include "ImgCodecDefine.h"

#ifdef OPH_EXPORT
#define OPH_DLL __declspec(dllexport)
#else
#define OPH_DLL __declspec(dllimport)
#endif

namespace oph
{
	class OPH_DLL ImgCodecOhc {
	public: /* method */
		ImgCodecOhc();
		ImgCodecOhc(const std::string &_fname);
		ImgCodecOhc(const std::string &_fname, const ohcHeader &_Header);
		virtual ~ImgCodecOhc() = 0;
		virtual void initOHCheader();
		virtual void releaseFldData();
		void releaseOHCheader();
		void releaseCodeBuffer();

		bool setFileName(const std::string &_fname);
		bool setOHCheader(const ohcHeader &_Header);

		void getOHCheader(ohcHeader &_Header);
		void getFieldInfo(ohcFieldInfoHeader &_FieldInfo, std::vector<double_t> &_wavlenTable);

		void getComplexFieldData(OphComplexField& cmplx_field, uint wavelen_idx) { cmplx_field = field_cmplx[wavelen_idx]; }
		void getComplexFieldData(Complex<Real>** cmplx_field, uint wavelen_idx);

		void getComplexFieldData(OphComplexField** cmplx_field);
		void getComplexFieldData(Complex<Real>*** cmplx_field);
		
	protected: /* field */
		std::string fname;
		//void* buf = nullptr; //coded data
		float*	buf_f32 = nullptr; //coded data
		double* buf_f64 = nullptr; //coded data		
		std::vector<OphComplexField> field_cmplx; //Real & Imagine data
		std::vector<std::string> linkFilePath;

		ohcHeader* Header = nullptr;
	};


	/* Load *.ohc file format to Complex field data */
	class OPH_DLL ImgDecoderOhc : public ImgCodecOhc {
	public:
		ImgDecoderOhc();
		ImgDecoderOhc(const std::string &_fname);
		ImgDecoderOhc(const std::string &_fname, const ohcHeader &_Header);
		virtual ~ImgDecoderOhc();
		virtual void releaseFldData();

		//Get field Info parameters functions
		ivec2		getNumOfPixel();
		vec2		getPixelPitch();
		LenUnit		getPixelPitchUnit();
		uint		getNumOfWavlen();
		ColorType	getColorType();
		ColorArran	getColorArrange();
		LenUnit		getUnitOfWavlen();
		CompresType	getCompressedFormatType();
		void getWavelength(std::vector<double_t> &wavlen_array);
		void getLinkFilePath(std::vector<std::string> &linkFilePath_array);

		bool load();

	protected:
		bool bLoadFile = false;
		//template<typename T> bool decodeFieldData();
		//template<typename T> Real decodePhase(const T phase, const Real min_p, const Real max_p, const double min_T, const double max_T);
		bool decodeFieldData();

		//Only Amplitude Encoding or Only Phase Encoding or Amplitude & Phase data
		std::vector<OphRealField> field_ampli;
		std::vector<OphRealField> field_phase;
		std::ifstream File;
	};


	/* Save Complex field data to *.ohc file format */
	class OPH_DLL ImgEncoderOhc : public ImgCodecOhc {
	public:
		ImgEncoderOhc();
		ImgEncoderOhc(const std::string &_fname);
		ImgEncoderOhc(const std::string &_fname, const ohcHeader &_Header);
		virtual ~ImgEncoderOhc();
		void initOHCheader();

		//Set field Info parameters functions
		void setNumOfPixel(const uint _pxNumX, const uint _pxNumY);
		void setNumOfPixel(const ivec2 _pxNum);
		void setPixelPitch(const double _pxPitchX, const double _pxPitchY, const LenUnit unit = LenUnit::m);
		void setPixelPitch(const vec2 _pxPitch, const LenUnit unit = LenUnit::m);
		void setNumOfWavlen(const uint n_wavlens);
		void setWavelength(const Real _wavlen, const LenUnit _unit = LenUnit::m);
		void setColorType(const ColorType _clrType);
		void setColorArrange(const ColorArran _clrArrange);
		void setUnitOfWavlen(const LenUnit unit);
		void setFieldEncoding(const FldStore _fldStore, const FldCodeType _fldCodeType); //const DataType _cmplxFldType = DataType::Float64);
		void setPhaseEncoding(const BPhaseCode _bPhaseCode, const double _phaseCodeMin, const double _phaseCodeMax);
		void setPhaseEncoding(const BPhaseCode _bPhaseCode, const vec2 _phaseCodeRange);
		//void setCompressedFormatType(const CompresType _comprsType);

		void addWavelengthNComplexFieldData(const Real wavlen, const OphComplexField &data);
		void addComplexFieldData(const OphComplexField &data);
		void addComplexFieldData(const Complex<Real> *data);
		void addWavelength(const Real wavlen);
		//void addLinkFilePath(const std::string &path);

		bool save();

	protected:
		//template<typename T> uint64_t encodeFieldData();
		//template<typename T> T encodePhase(const Real phase_angle, const Real min_p, const Real max_p, const double min_T, const double max_T);
		uint64_t encodeFieldData();

		std::ofstream File;
	};
}

#endif // !__ImgCodecOhc_h