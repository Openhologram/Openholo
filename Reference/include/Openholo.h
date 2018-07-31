/**
* @mainpage Openholo 
* @brief Abstract class
*/

#ifndef __Openholo_h
#define __Openholo_h

#include "Base.h"
#include "include.h"
#include "vec.h"
#include "ivec.h"
#include "fftw3.h"

using namespace oph;

class OPH_DLL Openholo : public Base{

public:
	/**
	* @brief Constructor
	*/
	explicit Openholo(void);

protected:
	/**
	* @brief Destructor
	*/
	virtual ~Openholo(void) = 0;

public:
	/**
	* @breif save data is Openholo::p_hologram if src is nullptr.
	*/
	//int save(const char* fname, uint8_t bitsperpixel = 8, void* src = nullptr, int pic_width = 0, int pic_height = 0);

	/**
	* @breif load data from .oph or .bmp
	* @breif loaded data is stored in the Openholo::p_hologram if dst is nullptr.
	*/
	//int load(const char* fname, void* dst = nullptr);

protected:
	int checkExtension(const char* fname, const char* ext);

protected:
	int saveAsImg(const char* fname, uint8_t bitsperpixel, uchar* src, int pic_width, int pic_height);
	uchar* loadAsImg(const char* fname);

	/**
	*/
	int loadAsImgUpSideDown(const char* fname, uchar* dst);

	/**
	* @param output parameter. image size, width
	* @param output parameter. image size, Height
	* @param output parameter. bytes per pixel
	* @param input parameter. file name
	*/
	int getImgSize(int& w, int& h, int& bytesperpixel, const char* file_name);

	/**
	* @brief Function for change image size
	* @param input parameter. source image data
	* @param output parameter. dest image data
	* @param input parameter. original width
	* @param input parameter. original height
	* @param input parameter. width to replace
	* @param input parameter. height to replace
	*/
	void imgScaleBilnear(unsigned char* src, unsigned char* dst, int w, int h, int neww, int newh);

	/**
	* @brief Function for convert image format to gray8
	* @param input parameter. source image data
	* @param output parameter. dest image data
	* @param input parameter. image size, width
	* @param input parameter. image size, Height
	* @param input parameter. bytes per pixel
	*/
	void convertToFormatGray8(unsigned char* src, unsigned char* dst, int w, int h, int bytesperpixel);


	/**

	*/
	void fft1(int n, Complex<Real>* in, int sign = OPH_FORWARD, uint flag = OPH_ESTIMATE);
	void fft2(oph::ivec2 n, Complex<Real>* in, int sign = OPH_FORWARD, uint flag = OPH_ESTIMATE);
	void fft3(oph::ivec3 n, Complex<Real>* in, int sign = OPH_FORWARD, uint flag = OPH_ESTIMATE);

	void fftExecute(Complex<Real>* out);
	void fftFree(void);
	/**
	* @brief Convert data from the spatial domain to the frequency domain using 2D FFT on CPU.
	* @details It is equivalent to Matlab code, dst = ifftshift(fft2(fftshift(src))).
	* @param src : input data variable
	* @param dst : output data variable
	* @param in : input data pointer connected with FFTW plan
	* @param out : ouput data pointer connected with FFTW plan
	* @param nx : the number of column of the input data
	* @param ny : the number of row of the input data
	* @param type : If type == 1, forward FFT, if type == -1, backward FFT.
	* @param bNomarlized : If bNomarlized == true, normalize the result after FFT.
	* @see propagation_AngularSpectrum_CPU, encoding_CPU
	*/
	void fftwShift(Complex<Real>* src, Complex<Real>* dst, int nx, int ny, int type, bool bNormalized = false);

	/**
	* @brief Swap the top-left quadrant of data with the bottom-right , and the top-right quadrant with the bottom-left.
	* @param nx : the number of column of the input data
	* @param ny : the number of row of the input data
	* @param input : input data variable
	* @param output : output data variable
	* @see fftwShift
	*/
	void fftShift(int nx, int ny, Complex<Real>* input, Complex<Real>* output);

protected:
	/**
	* @brief Pure virtual function for override in child classes
	*/
	virtual void ophFree(void);

private:
	fftw_plan plan_fwd, plan_bwd;
	fftw_complex *fft_in, *fft_out;
	int pnx, pny, pnz;
	int fft_sign;
};

#endif // !__Openholo_h