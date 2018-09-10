#ifndef __DefineImgCodec_h
#define __DefineImgCodec_h


namespace oph
{
	/************************ Enumerator Class for OHC *****************************/

	/* Unit of Length */
	enum class LenUnit : uint8_t {
		m = 0,	/* Meter */
		cm = 1,	/* Centi Meter */
		mm = 2,	/* Milli Meter */
		um = 3,	/* Micro Meter */
		nm = 4,	/* Nano Meter */
	};

	/* Color Channel Type */
	enum class ColorType : uint8_t {
		RGB = 0,	/* RGB 3-channel */
		MLT = 1,	/* Multiple Colors : Grayscale color is the one case of MLT. */
		//GRY = 2,	/* Grayscale 1-channel */
	};

	/* Color Arrangement */
	enum class ColorArran : uint8_t {
		SequentialRGB = 0,
		EachChannel = 1,
	};

	/* Complex Field Data Type */
	enum class DataType : uint8_t {
		Int8 = 0,	/* char : 8S */
		Int16 = 1,	/* short : 16S */
		Int32 = 2,	/* long : 32S */
		Int64 = 3,	/* longlong : 64S */
		Uint8 = 4,	/* uchar : 8U */
		Uint16 = 5,	/* ushort : 16U */
		Uint32 = 6,	/* ulong : 32U */
		Uint64 = 7,	/* ulonglong : 64U */
		Float32 = 8,	/* Single precision floating : 32F */
		Float64 = 9,	/* Double precision floating : 64F */
		ImgFmt = 10,	/* Compressed Image File : IMG */
	};

	/* Field Store Type */
	enum class FldStore : uint8_t {
		Directly = 0,	/* Field data is directly stored at the 'Field Data' region. */
		LinkFile = 1,	/* Field data is stored at separate files and they are referred by path. 'Field Data' region stores those file paths. */
	};

	/* Encoding Type of Field Data Domain */
	enum class FldCodeType : uint8_t {
		AP = 0,		/* Amplitude & Phase */
		RI = 1,		/* Real & Imaginary */
		AE = 2,		/* Amplitude-only Encoded */
		PE = 3,		/* Phase-only Encoded */
	};

	/* Phase Encoded Type : Boolean */
	enum class BPhaseCode : uint8_t {
		NotEncoded = 0,
		Encoded = 1,
	};

	/* Compressed Image File Format */
	enum class ImageFormat : uint8_t {
		RAW = 0,	/* No Image Format, Directly store raw data. */
		BMP = 1,	/* Bitmap (bmp, dib) */
		JPG = 2,	/* JPEG (jpg, jpeg, jpe) */
		J2K = 3,	/* JPEG-2000 (jpf, jpx, jp2, j2c, j2k, jpc) */
		PNG = 4,	/* PNG (png, pns) */
		GIF = 5,	/* GIF (gif) */
		TIF = 6,	/* TIFF (tif, tiff) */
	};


	/************************ File Header Struct for OHC *****************************/

	/* Openholo Complex Field File Format(*.ohc) Definition */
	typedef struct ohcFileHeader {
		int8_t		fileSignature[2];	/* File Type(2 Byte) : 'OH' 0x484F */
		uint64_t	fileSize;			/* Entire file size(in byte) */
		uint8_t		fileVersionMajor;	/* Major version of file format */
		uint8_t		fileVersionMinor;	/* Minor version of file format */
		uint32_t	fileReserved1;		/* For potential use. Currently zero. */
		uint32_t	fileReserved2;		/* For potential use. Currently zero. */
		uint32_t	fileOffBytes;		/* Address of complex field data */
	} OHCFILEHEADER;

	typedef struct ohcFieldInfoHeader {
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
		ImageFormat	imgFmt;			/* Image file format of complex data : for 'cmplxFldType == ImgFmt' */
	} OHCFIELDINFOHEADER;

	typedef struct ophComplexFile {
		OHCFILEHEADER			fileHeader;
		OHCFIELDINFOHEADER		fieldInfo;
		std::vector<double_t>	wavlenTable; /* Wavelength : Scalable Data Size(8/24/8n). When 'clrType' is RGB, wavelengths of red, green, and blue are stored sequentially; When 'clrType' is MLT, size of this field is 8*n bytes, where 'n' is the 'wavlenNum'. */
	} OHCheader;
}

#endif