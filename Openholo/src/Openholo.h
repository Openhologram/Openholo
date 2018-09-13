
/**
* \defgroup const/dest Constructor & Destructor
* \defgroup oper Operator
* \defgroup get/set Parameters
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
	* \ingroup const/dest
	* @brief Constructor
	*/
	explicit Openholo(void);

protected:
	/**
	* \ingroup const/dest
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
	virtual int saveAsOhc(const char* fname);

	/**
	* \ingroup read
	* @brief Function to read OHC file
	*/
	virtual int loadAsOhc(const char* fname);

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
	/**
	* @brief OHC file format Variables for read and write
	*/
	ImgEncoderOhc* OHC_encoder;
	ImgDecoderOhc* OHC_decoder;

protected:
	/**
	* @brief getter/setter for OHC file read and write
	*/
	inline void setPixelNumber(const ivec2 pixel_number) 
		{ OHC_encoder->setNumOfPixel(pixel_number); }

	inline void setPixelPitch(const vec2 pixel_pitch)
		{ OHC_encoder->setPixelPitch(pixel_pitch); }

	inline void setWavelength(const Real wavelength, const LenUnit wavelength_unit)
		{ OHC_encoder->setWavelength(wavelength, wavelength_unit); }

	inline void setWaveLengthNum(const uint wavelength_num)
		{ OHC_encoder->setNumOfWavlen(wavelength_num); }

	inline void setColorType(const ColorType color_type)
		{ OHC_encoder->setColorType(color_type); }

	inline void setColorArrange(const ColorArran color_arrange)
		{ OHC_encoder->setColorArrange(color_arrange);	}

	inline 	void setWaveLengthUnit(const LenUnit length_unit)
		{ OHC_encoder->setUnitOfWavlen(length_unit);	}

	inline 	void setFieldEncoding(const FldStore field_store, const FldCodeType field_code_type, const DataType cplxfield_type)
		{ OHC_encoder->setFieldEncoding(field_store, field_code_type, cplxfield_type); }

	inline 	void setPhaseEncoding(const BPhaseCode phase_code, const vec2 phase_code_range)
		{ OHC_encoder->setPhaseEncoding(phase_code, phase_code_range); }

	inline void setImageFormat(const ImageFormat image_format)
		{ OHC_encoder->setImageFormat(image_format); }

	/**
	* @brief Function to add ComplexField when adding wavelength data
	*/
	inline void addWaveLengthNComplexFieldData(const Real wavelength, const OphComplexField& complex_field)
		{ OHC_encoder->addWavelengthNComplexFieldData(wavelength, complex_field); }

	inline void addWaveLength(const Real wavelength)
		{ OHC_encoder->addWavelength(wavelength); }

	inline void addComplexFieldData(const OphComplexField& complex_field)
		{ OHC_encoder->addComplexFieldData(complex_field); }

	/**

	*/
	inline void addLinkFilePath(const std::string& path)
		{ OHC_encoder->addLinkFilePath(path); }

	/**

	*/
	inline void getLinkFilePath(const int idx, std::string& path)
		{ OHC_encoder->getLinkFilePath(idx, path); }


};

#endif // !__Openholo_h