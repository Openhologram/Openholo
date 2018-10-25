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

#ifndef __DefineImgCodec_h
#define __DefineImgCodec_h


namespace oph
{

#define FMT_SIGN_OHC "OH" // File Format Signature : 0x484F
#define LINK_IMG_PATH_SIZE 4*1024*sizeof(BYTE) // 4KB

	/************************ Enumerator Class for OHC *****************************/
	
	/* Unit of Length */
	enum class LenUnit : uint8_t {
		Null= 0,	/* Null Data */
		m	= 1,	/* Meter */
		cm	= 2,	/* Centi Meter */
		mm	= 3,	/* Milli Meter */
		um	= 4,	/* Micro Meter */
		nm	= 5,	/* Nano Meter */
	};

	/* Color Channel Type */
	enum class ColorType : uint8_t {
		Null= 0,	/* Null Data */
		RGB	= 1,	/* RGB 3-channel */
		MLT	= 2,	/* Multiple Colors : Grayscale color is the one case of MLT. */
		//GRY = 3,	/* Grayscale 1-channel */
	};

	/* Color Arrangement */
	enum class ColorArran : uint8_t {
		Null	  = 0,	/* Null Data */
		SeqtChanl = 1,	/* Sequential RGB Color channels */
		EachChanl = 2,	/* Each Color channels */
	};

	/* Complex Field Data Type */
	enum class DataType : uint8_t {
		Null	= 0,	/* Null Data */
		Int8	= 1,	/* char */
		Int16	= 2,	/* short */
		Int32	= 3,	/* long */
		Int64	= 4,	/* longlong */
		Uint8	= 5,	/* uchar */
		Uint16	= 6,	/* ushort */
		Uint32	= 7,	/* ulong */
		Uint64	= 8,	/* ulonglong */
		Float32 = 9,	/* Single precision floating */
		Float64 = 10,	/* Double precision floating */
		CmprFmt = 11,	/* Compressed Image File */
	};

	/* Field Store Type */
	enum class FldStore : uint8_t {
		Null	 = 0,	/* Null Data */
		Directly = 1,	/* Field data is directly stored at the 'Field Data' region. */
		LinkFile = 2,	/* Field data is stored at separate files and they are referred by path. 'Field Data' region stores those file paths. */
	};

	/* Encoding Type of Field Data Domain */
	enum class FldCodeType : uint8_t {
		Null = 0,	/* Null Data */
		AP	 = 1,	/* Amplitude & Phase */
		RI	 = 2,	/* Real & Imaginary */
		AE	 = 3,	/* Amplitude-only Encoded */
		PE	 = 4,	/* Phase-only Encoded */
	};

	/* Phase Encoded Type : Boolean */
	enum class BPhaseCode : uint8_t {
		Null = 0,
		NotEncoded = 0,
		Encoded = 1,
	};

	/* Compressed Image Type File Format */
	enum class CompresType : uint8_t {
		Null = 0,	/* No Image Format, Directly store raw data. */
		BMP  = 1,	/* Bitmap (bmp, dib) */
		JPG  = 2,	/* JPEG (jpg, jpeg, jpe) */
		J2K  = 3,	/* JPEG-2000 (jpf, jpx, jp2, j2c, j2k, jpc) */
		PNG  = 4,	/* PNG (png, pns) */
		GIF  = 5,	/* GIF (gif) */
		TIF  = 6,	/* TIFF (tif, tiff) */
	};


	/************************ File Header Struct for OHC *****************************/

	/* Openholo Complex Field File Format(*.ohc) Definition */
	struct ohcFileHeader {
		int8_t		fileSignature[2];	/* File Type(2 Byte) : 'OH' 0x484F */
		uint64_t	fileSize;			/* Entire file size(in byte) */
		uint8_t		fileVersionMajor;	/* Major version of file format */
		uint8_t		fileVersionMinor;	/* Minor version of file format */
		uint32_t	fileReserved1;		/* For potential use. Currently zero. */
		uint32_t	fileReserved2;		/* For potential use. Currently zero. */
		uint32_t	fileOffBytes;		/* Address of complex field data */

		//basic constructor
		ohcFileHeader() {
			this->fileSignature[0] = FMT_SIGN_OHC[0];
			this->fileSignature[1] = FMT_SIGN_OHC[1];
			this->fileSize = 0;
			this->fileVersionMajor = _OPH_LIB_VERSION_MAJOR_;
			this->fileVersionMinor = _OPH_LIB_VERSION_MINOR_;
			this->fileReserved1 = 0;
			this->fileReserved2 = 0;
			this->fileOffBytes = -1;
		}
	};

	struct ohcFieldInfoHeader {
		uint32_t	headerSize;		/* Size of Field Info Header(in byte) : InfoHeader + WaveLengthTable */
		uint32_t	pxNumX;			/* Number of pixels of field data in x-direction */
		uint32_t	pxNumY;			/* Number of pixels of field data in y-direction */
		double_t	pxPitchX;		/* Pixel pitch of field data in x-direction */
		double_t	pxPitchY;		/* Pixel pitch of field data in y-direction */
		LenUnit		pitchUnit;		/* Unit of pixel pitch */
		uint32_t	wavlenNum;		/* Number of Wavelengths */
		ColorType	clrType;		/* Color Type */
		ColorArran	clrArrange;		/* Color arrangement */
		LenUnit		wavlenUnit;		/* Unit of Wavelength (in Wavelength Table). */
		DataType	cmplxFldType;	/* Complex Field Data Type.  */
		FldStore	fldStore;		/* Field Store Type.  */
		FldCodeType	fldCodeType;	/* Field Encoding Type.  */
		BPhaseCode	bPhaseCode;		/* Phase Encoded Type(Boolean). 0: Not Encoded, 1: Encoded */
		double_t	phaseCodeMin;	/* Phase Encoded Min. */
		double_t	phaseCodeMax;	/* Phase Encoded Max. */
		uint64_t	fldSize;		/* Entire Field data size */
		CompresType	comprsType;		/* Image file format of complex data : for 'cmplxFldType == ImgFmt' */

		//basic constructor
		ohcFieldInfoHeader() {
			this->headerSize = 0;
			this->pxNumX = (uint32_t)-1;
			this->pxNumY = (uint32_t)-1;
			this->pxPitchX = (double_t)-1;
			this->pxPitchY = (double_t)-1;
			this->pitchUnit = LenUnit::Null;
			this->wavlenNum = 0;
			this->clrType = ColorType::Null;
			this->clrArrange = ColorArran::Null;
			this->wavlenUnit = LenUnit::Null;
			this->cmplxFldType = DataType::Null;
			this->fldStore = FldStore::Null;
			this->fldCodeType = FldCodeType::Null;
			this->bPhaseCode = BPhaseCode::Null;
			this->phaseCodeMin = -1.0;
			this->phaseCodeMax = 1.0;
			this->fldSize = 0;
			this->comprsType = CompresType::Null;
		}
	};

	struct ohcHeader {
		ohcFileHeader			fileHeader;
		ohcFieldInfoHeader		fieldInfo;
		std::vector<double_t>	wavlenTable; /* Wavelength : Scalable Data Size(8/24/8n). When 'clrType' is RGB, wavelengths of red, green, and blue are stored sequentially; When 'clrType' is MLT, size of this field is 8*n bytes, where 'n' is the 'wavlenNum'. */
	};
}

#endif