/** @mainpage
@tableofcontents
\section sec Introduction
Diffraction calculation, such as the angular spectrum method and Fresnel diffraction, 
are used in wide-ranging optics fields, ultrasonic, X-ray and electron beam to calculate scalar wave propagation.
For example, in optics field, these calculations are useful for computer-generated holograms (CGHs), digital holography, 
the design of diffractive optical elements, microscopy, image encryption and decryption, 
three-dimensional analysis for optical devices and so on.@n
We develop a C++ class library, which is referred to as Computational Wave Optics library 
for C++ (CWO++), for diffraction and useful auxiliary functions.
The diffraction caluclations are Fresnel diffraction, angular spectrum method, 
scaled diffractions, tilted diffractions and so forth.
This library can run on both CPUs and NVIDIA GPUs.@n

@n
This library is designed for easily programming wave optics phenomena.  
For example, the following sample code shows the calculation of a diffracted field using the angular spectrum method on a CPU.
@n

@code
CWO a;
a.Load("sample.bmp"); //load image file 
a.Diffract(0.1, CWO_ANGULAR); //calculate diffraction from the image using the angular spectrum method
a.SaveAsImage("diffract.bmp",CWO_SAVE_AS_INTENSITY); //save the intensity of diffracted field as bitmap file
a.SaveAsImage("diffract.bmp",CWO_SAVE_AS_AMP); //save the amplitude of diffracted field as bitmap file
a.SaveAsImage("diffract.bmp",CWO_SAVE_AS_ARG); //save the argument (phase) of diffracted field  as bitmap file
a.SaveAsImage("diffract.bmp",CWO_SAVE_AS_RE); //save the real part of diffracted field as bitmap file
a.SaveAsImage("diffract.bmp",CWO_SAVE_AS_IM); //save the imaginary part of fiffracted field as bitmap file
@endcode

If you want to execute the above source code on a GPU, you only rewrite class CWO with class GWO as the following source code.
Other code rewritings are not necessary. This feature improves portability from a CPU source code to GPU one.
The data communication between a host computer and GPU is implicitly done by class GWO.
@code
GWO a;
a.Load("sample.bmp"); //load image file
a.Diffract(0.1, CWO_ANGULAR); //calculate diffraction from the image using the angular spectrum method
a.SaveAsImage("diffract.bmp",CWO_SAVE_AS_INTENSITY); //save the intensity of diffracted field as bitmap file
a.SaveAsImage("diffract.bmp",CWO_SAVE_AS_AMP); //save the amplitude of diffracted field as bitmap file
a.SaveAsImage("diffract.bmp",CWO_SAVE_AS_ARG); //save the argument (phase) of diffracted field  as bitmap file
a.SaveAsImage("diffract.bmp",CWO_SAVE_AS_RE); //save the real part of diffracted field as bitmap file
a.SaveAsImage("diffract.bmp",CWO_SAVE_AS_IM); //save the imaginary part of fiffracted field as bitmap file
@endcode

You can also use the following sample code if you need explicitly data communication.
@n
@code
CWO a;
GWO g; //class for GPU
a.Load("sample.bmp"); //load image file 
g.Send(a); //send the image data to the GPU
g.Diffract(0.1, CWO_ANGULAR); //calculate diffraction on the GPU
g.Intensity(); //calculate the light intensity on the GPU
g.Scale(255); // convert to 256 gray scale data on the GPU
g.Recv(a); //host computer receives the diffraction image
a.Save("diffract.bmp"); //save the intensity as bitmap file
@endcode

The left and right images are the original image and diffracted image, respectively.
@image html lena512x512_diffract.jpg
For more sample codes, please see example page.

@section gallary Gallary
The following CGHs and the reconstructed images were obtained by CWO++ library.@n
@image html white_asian_dragon.jpg
Reconstructed images from CGHs. (original data : The Stanford 3D Scanning Repository)

@image html color_buddah.jpg
Reconstructed images from CGHs.
(original data : S.Wanner, S. Meister, and B. Goldluecke, "Datasets and Benchmarks for Densely Sampled 4D Light Fields," Vision, Modeling & Visualization, 225–226 (2013))


@section download Downloads
You can download this library from 
<a href="https://sourceforge.net/projects/cwolibrary/?source=directory">here</a>.


@section env Environment
- Windows XP or higher
- CPU version is available in 32 and 64bit version OSs
- GPU version is only available in 64bit version OS
- Linux (Cent OS) version is comming soon
- If using NVIDIA GPUs, CWO++ requires CUDA 7.5 (64bit version is required)

@section inst Installation for Windows
- Please download latest CWO library from <a href="https://sourceforge.net/projects/cwolibrary/?source=directory">here</a>.
- The library file "cwo.lib" and dll file "cwo.dll" is placed in arbitrary directory.
	Please set to the PATH in your system.
	Or, please place these files on your C++ project directory.
- The directory "src" includes C++ source codes for CWO library. 
	Please set the PATH in your system.
- In your C++ source code, please include the header file "cwo.h"
- If you want to use jpeg format etc., except for bitmap format, you need to 
	install <a href="http://www.imagemagick.org/download/binaries/ImageMagick-6.8.9-2-Q8-x64-static.exe">ImageMagick</a>.

@section inst_linux Installation for Linux 
Coming soon... 

@section update How to update new CWO++ library to existing code using the old version
If you have already used CWO++ library and want to update new version of CWO++ library, 
please overwrite the existing directory "src" and files "cwo.dll" "cwo.lib"  "gwo.dll" 
and "gwo.lib" by the new version of them.

@section problems If you have problems...
- Runtime error of "cannot find VCOMP10.dll" @n
VCOMP10.dll is Microsoft C/C++ OpenMP Runtime. You will solve this problem by installing 
"Microsoft Visual C++ 2010 SP1 Redistributable Package (x64)".
http://www.microsoft.com/en-us/download/details.aspx?id=13523

- Cannot open include file: 'omp.h' @n
"omp.h" is the header file of OpenMP. If you use Visual Studio C++ express version, 
you may find the error of "cannot find omp.h" because the VISUAL C++ express version 
did not use OpenMP. Please check that "Properties -> C/C++ -> Language -> OpenMP Support" is "no".

- Cannot load and save as jpg, png, tif etc. file @n
CWO++ library can load and save as jpg, png, tif etc using ImageMagik.
For example, please install "ImageMagick-6.8.9-7-Q8-x64-dll.exe". @n
http://www.imagemagick.org/script/binary-releases.php#windows

@section ref Reference
- T. Shimobaba, J. Weng, T. Sakurai, N. Okada, T. Nishitsuji, N. Takada, A. Shiraki, 
N. Masuda and T. Ito, "Computational wave optics library for C++: CWO++ library", 
Computer Physics Communications, 183, 1124-1138 (2012) @n

@section academic Academic achievements using CWO library
Please contact me when you can obtain academic / commercial achievements using CWO library.

- T. Shimobaba, M. Makowski, Y. Nagahama, Y. Endo, R. Hirayama, D. Hiyama, S. Hasegawa, M. Sano, T. Kakue, M. Oikawa, T. Sugie, N. Takada, T. Ito, "Computer-generated hologram using random phase-free method and color space conversion", Applied Optics, 55, 4159-4165 (2016)
- Y. Nagahama, T. Shimobaba, T. Kawashima, T. Kakue, T. Ito,"Holographic multi-projection using random phase-free method", Applied Optics, 55, 1118-1123 (2016)
- T. Shimobaba, T. Kakue, Y. Endo, R. Hirayama, D. Hiyama, S. Hasegawa, Y. Nagahama, M. Sano, M. Oikawa, T. Sugie, T. Ito, "Improvement of the image quality of random phase-free holography using an iterative method", Optics Communications, 355, 596-601 (2015)
- T. Shimobaba, T. Kakue, Y. Endo, R. Hirayama, D. Hiyama, S. Hasegawa, Y. Nagahama, M. Sano, M. Oikawa, T. Sugie, T. Ito,"Random phase-free kinoform for large objects", Optics Express, 23, 17269-17274 (Jun. 2015)
- T. Oshima, Y. Matsudo, T. Kakue，D. Arai, T. Shimobaba and T. Ito, "Twin-image reduction method for in-line digital holography using periphery and random reference phase-shifting techniques", Optics Comminications, 350, 270–275 (Sep. 2015)
- T. Shimobaba, T. Ito, "Random phase-free computer-generated hologram", Optics Express, 23, 9549-9554 (2015)
- I. Ducin, T. Shimobaba, M. Makowski, K. Kakarenko, A. Kowalczyk, J. Suszek, M. Bieda, A. Kolodziejczyk, M. Sypek, "Holographic projection of images with step-less zoom and noise suppression by pixel separation", Optics Communications, 340, 131–135 (2015)
- D. Hiyama, T. Shimobaba, T. Kakue and T. Ito, "Acceleration of color computer-generated hologram from RGB-D images using color space conversion", Optics Communications, 340, 121–125 (2015)
- D.V. Pantelic, D.Z. Grujic, D.M. Vasiljevic, "Single-beam, dual-view digital holographic interferometry for biomechanical strain measurements of biological objects", J. Biomed. Opt. 19, 127005 (2014)
- T. Shimobaba, M. Makowski, T. Kakue, N. Okada, Y. Endo, R. Hirayama, D. Hiyama, S. Hasegawa, Y. Nagahama, T. Ito, "Numerical investigation of lensless zoomable holographic projection to multiple tilted planes", Optics Communications, 333, 274-280 (2014)
- T. Shimobaba, T. Kakue, N. Okada, Y. Endo, R. Hirayama, D. Hiyama, and T. Ito, "Ptychography by changing the area of probe light and scaled ptychography", Optics Communications, 331, 189–193 (2014)
- T. Shimobaba, T. Kakue, M. Oikawa, N. Takada, N. Okada, Y. Endo, R. Hirayama, T. Ito, "Calculation reduction method for color computer-generated hologram using color space conversion", Optical Engineering (accepted)
- T. Shimobaba, T. Kakue, M. Oikawa, N. Okada, Y. Endo, R. Hirayama, N. Masuda, T. Ito, "Non-uniform sampled scalar diffraction calculation using non-uniform Fast Fourier transform", Optics Letters, 38, 5130-5133 (2013)
- T. Shimobaba, M. Makowski, T. Kakue, M. Oikawa, N. Okada, Y. Endo, R. Hirayama, T. Ito, "Lensless zoomable holographic projection using scaled Fresnel diffraction", Optics Express, 21, 25285-25290 (2013)
- T. Shimobaba, H. Yamanashi, T. Kakue, M. Oikawa, N. Okada, Y. Endo, R. Hirayama, N. Masuda, T. Ito, "Inline digital holographic microscopy using a consumer scanner", Scientific Reports, 3, 2664 (2013) 
- T. Shimobaba, T. Kakue, N. Okada, M. Oikawa, Y. Yamaguchi, T. Ito, "Aliasing-reduced Fresnel diffraction with scale and shift operations", Journal of Optics, 15, 075302(5pp) (2013)
- N. Okada, T. Shimobaba, Y. Ichihashi, R. Oi, K. Yamamoto, M. Oikawa, T. Kakue, N. Masuda, T. Ito, "Band-limited double-step Fresnel diffraction and its application to computer-generated holograms," Opt. Express 21, 9192-9197 (2013) 
- T. Shimobaba, T. Kakue, Nobuyuki Masuda and Tomoyoshi Ito, "Numerical investigation of zoomable holographic projection without a zoom lens", Journal of the Society for Information Display, 20, 9, 533-538 (2012)
- T. Shimobaba, K. Matsushima, T. Kakue, N. Masuda, T. Ito, "Scaled angular spectrum method", Optics Letters, 37, 4128-4130 (2012.09)
- T. Shimobaba, N. Masuda and T. Ito, "Arbitrary shape surface Fresnel diffraction", Optics Express 20, 9335-9340 (2012.04) 
- T. Shimobaba, T. Takahashi, N. Masuda, T. Ito, "Numerical study of color holographic projection using space-division method", Optics Express 19, 10287-10292 (2011)
- T. Shimobaba, H. Nakayama, N. Masuda, T. Ito, "Rapid calculation of Fresnel computer-generated-hologram using look-up table and wavefront-recording plane methods for three-dimensional display", Optics Express, 18, 19, 19504-19509 (2010)
- T. Shimobaba, N. Masuda, Y. Ichihashi, T. Ito, "Real-time digital holographic microscopy observable in multi-view and multi-resolution", Journal of Optics, 12, 065402 (4pp) (2010)
- T. Shimobaba, N. Masuda, T. Ito, "Simple and fast calclulation algorithm for computer-generated hologram with wavefront recording plane", Optics Letters, 34, 20, 3133-3135 (2009)
- H. T. Dai, X. W. Sun, D. Luo, and Y. J. Liu, "Airy beams generated by a binary phase element made of polymer-dispersed liquid crystals," Opt. Express 17, 19365-19370 (2009) 
- D. Luo, H. T.Dai, X. W. Sun, H. V. Demir, "Electrically switchable finite energy Airy beams generated by a liquid crystal cell with patterned electrode", Optics Communications, 283(20), 3846-3849 (2010).

@section copyright Copyright 
(C) Tomoyoshi Shimobaba 2011-2016
*/



#ifndef _CWO_H
#define _CWO_H


//###########################################
/** @defgroup macros Macros
*/
//@{
/**
* @def CWO_FLD_COMPLEX
* indicates complex amplitude.
*/
#define CWO_FLD_COMPLEX		(0)
/**
* @def CWO_FLD_INTENSITY
* indicates the light intensity of complex amplitude.
*/
#define CWO_FLD_INTENSITY	(1)
/**
* @def CWO_FLD_PHASE
* indicates the argument of complex amplitude.
*/
#define CWO_FLD_PHASE		(2)
/**
* @def CWO_FLD_CHAR
* This field-type is now testing. Do not use it. 
*/
#define CWO_FLD_CHAR		(3)


//color
/**
* @def CWO_RED 
* indicate red color. 
*/
#define CWO_RED		(0)
/**
* @def CWO_GREEN
* indicate green color. 
*/
#define CWO_GREEN	(1)
/**
* @def CWO_BLUE
*indicate blue color. 
*/
#define CWO_BLUE	(2)
/**
* @def CWO_GREY
* indicate glay scale. 
*/
#define CWO_GREY	(3)


/**
* @def CWO_INTER_NEAREST
* indicate nearrest neighbor interpolation. 
*/
#define CWO_INTER_NEAREST	(0)
/**
* @def CWO_INTER_LINEAR
* indicate bi-linear interpolation. 
*/
#define CWO_INTER_LINEAR	(1)
/**
* @def CWO_INTER_CUBIC
* indicate bi-cubic interpolation.
*/
#define CWO_INTER_CUBIC		(2)
/**
* @def CWO_INTER_LANCZOS
* indicate bi-cubic interpolation.
*/
#define CWO_INTER_LANCZOS	(3)



/**
* @def CWO_ED_FS
* indicate Floyd & Steinberg's error diffusion method
*/
#define CWO_ED_FS	(0)

/**
* @def CWO_ED_MFS
* indicate modified Floyd & Steinberg's error diffusion method
*/
#define CWO_ED_MFS	(1)

/**
* @def CWO_ED_JJN
* indicate Jarvis, Judice & Ninke's error diffusion method
*/
#define CWO_ED_JJN	(2)

/**
* @def CWO_ED_ST
* indicate Stucki's error diffusion method
*/
#define CWO_ED_ST	(3)

/**
* @def CWO_ED_BUR
* indicate Burkes 's error diffusion method
*/
#define CWO_ED_BUR	(4)


/**
* @def CWO_ED_ATK
* indicate Atkinson  's error diffusion method
*/
#define CWO_ED_ATK	(5)



/**@def CWO_SAVE_AS_INTENSITY
save intensity of complex amplitude as 256 gray scale image*/
#define CWO_SAVE_AS_INTENSITY	(0x01)

#define CWO_SAVE_AS_PHASE		(0x02)

/**@def CWO_SAVE_AS_AMP
save amplitude of complex amplitude as 256 gray scale image*/
#define CWO_SAVE_AS_AMP			(0x03)

/**@def CWO_SAVE_AS_RE
save real part of complex amplitude as 256 gray scale image*/
#define CWO_SAVE_AS_RE			(0x04)

/**@def CWO_SAVE_AS_IM
save imaginary part of complex amplitude as 256 gray scale image*/
#define CWO_SAVE_AS_IM			(0x05)

/**@def CWO_SAVE_AS_ARG
save argument part of complex amplitude as 256 gray scale image*/
#define CWO_SAVE_AS_ARG			(0x06)


#define CWO_SAVE_AS_LOG			(0x100)


#define CWO_PROP_CENTER			(0)
#define CWO_PROP_FFT_SHIFT		(1)
#define CWO_PROP_MUL_CENTER		(2)
#define CWO_PROP_MUL_FFT_SHIFT	(3)

#define CWO_FILL_INSIDE		(0)
#define CWO_FILL_OUTSIDE	(1)

#define CWO_EXPAND_ZEROPAD	(0)
#define CWO_EXPAND_MIRROR	(1)
#define CWO_EXPAND_WITHOUT	(2)

#define CWO_BUFFER_FIELD				(0)
#define CWO_BUFFER_DIFF_A				(1)
#define CWO_BUFFER_DIFF_B				(2)
#define CWO_BUFFER_D1					(3)
#define CWO_BUFFER_D2					(4)
#define CWO_BUFFER_X1					(5)
#define CWO_BUFFER_X2					(6)
#define CWO_BUFFER_PLS					(7)

#define CWO_SUCCESS	(0)
#define CWO_FAIL	(-1)
//@}

#include "cwo_lib.h"
#include <math.h>
#include <stdarg.h>


//!  Top class. 
/*!
This class is top class of CWO++ library
*/
class CWO {
public:
	cwoCtx ctx;//current context
	cwoCtx prev_ctx; //previous context

	long long int rnd_seed;
private:
	

public:
	//****************
	//For Diffraction
	//****************
	void *p_field;//complex amplitude field
//	cwoComplex *p_diff_a;//temporary buffer for diffraction (convolution type)
//	cwoComplex *p_diff_b;//temporary buffer for diffraction (convolution type)


	float *fre_s_tbl;
	float *fre_c_tbl;
	
	//****************
	//For PLS
	//****************
	//int Npnt; // number of object point
	cwoObjPoint *p_pnt; //buffer for object point
	//unsigned char *p_tex; //buffer for texture
	//unsigned char *p_dep; //buffer for depth

	//****************
	//other
	//****************
	int field_type; //field type
	void InitParams(); /// Intialize parameters 
	void InitBuffers(); /// Initialze buffers
	void FreeBuffers(); /// Free buffers


public:
	virtual void* __Malloc(size_t size); 
	virtual void __Free(void **a);
	virtual void __Memcpy(void *dst, void *src, size_t size);
	virtual void __Memset(void *p, int c, size_t size);
	//virtual void __Memset(void *p, cwoComplex c, unsigned int size);

	virtual void __Expand(
		void *src, int sx, int sy, int srcNx, int srcNy, 
		void *dst, int dx, int dy, int dstNx, int dstNy,
		int type);

	virtual void __ExpandMirror(
		cwoComplex* src, int srcNx, int srcNy, 
		cwoComplex* dst, int dstNx, int dstNy,
		int type);
	

	virtual void __ShiftedFresnelAperture(cwoComplex *a);
	virtual void __ShiftedFresnelProp(cwoComplex *a);
	virtual void __ShiftedFresnelCoeff(cwoComplex *a);

	virtual void __ARSSFresnelAperture(cwoComplex *a);
	virtual void __ARSSFresnelProp(cwoComplex *a);
	virtual void __ARSSFresnelCoeff(cwoComplex *a);

	virtual void __FresnelConvProp(cwoComplex *a);
	virtual void __FresnelConvCoeff(cwoComplex *a, float const_val=1.0f);

	virtual void __FresnelAnalysisTransfer(cwoComplex *a, cwoComplex *b);

	virtual void __AngularProp(cwoComplex *a, int flag);
	void __AngularLim(float *fx_c, float *fx_w, float *fy_c, float *fy_w);
//	virtual void __ShiftedAngularProp(cwoComplex *a);

	virtual void __HuygensProp(cwoComplex *a);

	virtual void __FresnelFourierProp(cwoComplex *a);
	virtual void __FresnelFourierCoeff(cwoComplex *a);

	virtual void __FresnelDblAperture(cwoComplex *a, float z1);
	virtual void __FresnelDblFourierDomain(cwoComplex *a, float z1, float z2, cwoInt4 *zp);
	virtual void __FresnelDblCoeff(cwoComplex *a, float z1, float z2);

	virtual void __FFT(void *src, void *dst, int type);
	virtual void __IFFT(void *src, void *dst);
	virtual void __FFTShift(void *src);

	void __Add(cwoComplex *a, float b, cwoComplex *c);//c=a+b
	virtual void __Add( cwoComplex *a,  cwoComplex b, cwoComplex *c);//c=a+b
	virtual void __Add( cwoComplex *a,  cwoComplex *b, cwoComplex *c);//c=a+b

	void __Sub( cwoComplex *a,  float b, cwoComplex *c);//c=a-b
	virtual void __Sub( cwoComplex *a,  cwoComplex b, cwoComplex *c);//c=a-b
	virtual void __Sub( cwoComplex *a,  cwoComplex *b, cwoComplex *c);//c=a-b

	void __Mul( cwoComplex *a,  float b, cwoComplex *c);//c=a*b
	virtual void __Mul( cwoComplex *a,  cwoComplex b, cwoComplex *c);//c=a*b
	virtual void __Mul( cwoComplex *a,  cwoComplex *b, cwoComplex *c);//c=a*b
	
	void __Div( cwoComplex *a,  float b, cwoComplex *c);//c=a/b
	virtual void __Div( cwoComplex *a,  cwoComplex b, cwoComplex *c);//c=a/b
	virtual void __Div( cwoComplex *a,  cwoComplex *b, cwoComplex *c);//c=a/b

	
	virtual void __Re(cwoComplex *a , cwoComplex *b);
	virtual void __Im(cwoComplex *a , cwoComplex *b);
	virtual void __Conj(cwoComplex *a);

	
	virtual void __Intensity(cwoComplex *a, cwoComplex *b); //OK
	virtual void __Amp(cwoComplex *a , cwoComplex *b);//OK
	virtual void __Phase(cwoComplex *a , cwoComplex *b, float offset);//OK
	virtual void __Arg(cwoComplex *a , cwoComplex *b, float scale, float offset);//OK
	virtual void __Real2Complex(float *src, cwoComplex *dst);
	virtual void __Phase2Complex(float *src, cwoComplex *dst);
	virtual void __Arg2Cplx(cwoComplex *src, cwoComplex *dst, float scale, float offset);
	virtual void __Polar(float *amp, float *ph, cwoComplex *c);
	virtual void __ReIm(cwoComplex *re, cwoComplex *im, cwoComplex *c);
	virtual void __RectFillInside(cwoComplex *p, int m, int n, int Sx, int Sy, cwoComplex a);
	virtual void __RectFillOutside(cwoComplex *p, int m, int n, int Sx, int Sy, cwoComplex a);

	//virtual void __CopyFloat(float *src, int x1, int y1, int sNx, int sNy, float *dst, int x2, int y2, int dNx, int dNy, int Sx, int Sy);
	//virtual void __CopyComplex(cwoComplex *src, int x1, int y1, int sNx, int sNy, cwoComplex *dst, int x2, int y2, int dNx, int dNy, int Sx, int Sy);
	
	virtual void __FloatToChar(char *dst, float *src, int N);
	virtual void __CharToFloat(float *dst, char *src, int N);

	void SetFieldType(int type); 
	void SetSize(int Nx, int Ny, int Nz=1);
	void SetPropDist(float z);
	float GetPropDist();

	//! check whether calculation size is changed or not 
	/*!
	@return if true, the calculation size was changed, otherwise false
	*/
	bool IsSizeChanged();

public:
	
	//###########################################
	/** @defgroup const Constructors & Destructors
	*/
	//@{
	CWO();//!< constructor

	//! Constructor with size
    /*!
      \sa CWO() 
      \param Nx x size
      \param Ny y size
    */
	
	virtual ~CWO();//!< destructor

	CWO(CWO &tmp);//!< copy constructor
	CWO(int Nx, int Ny, int Nz=1);//!< constructor
	//CWO(char *fname);

	//@}

	//###########################################
	/** @defgroup operator Operators
	*/
	//@{
	CWO& operator=(CWO &tmp);	
	CWO& operator=(float a);
	CWO& operator=(cwoComplex a);
	
	CWO operator+(CWO &a);
	CWO operator+(float a);
	CWO operator+(cwoComplex a);
	CWO operator-(CWO &a);
	CWO operator-(float a);
	CWO operator-(cwoComplex a);
	CWO operator*(CWO &a);
	CWO operator*(float a);
	CWO operator*(cwoComplex a);
	CWO operator/(CWO &a);
	CWO operator/(float a);
	CWO operator/(cwoComplex a);
	
	CWO& operator+=(CWO &a);	
	CWO& operator+=(float a);
	CWO& operator+=(cwoComplex a);
	CWO& operator-=(CWO &a);
	CWO& operator-=(float a);
	CWO& operator-=(cwoComplex a);
	CWO& operator*=(CWO &a);
	CWO& operator*=(float a);
	CWO& operator*=(cwoComplex a);
	CWO& operator/=(CWO &a);
	CWO& operator/=(float a);
	CWO& operator/=(cwoComplex a);

	//CWO operator+(CWO &a);
	//CWO operator+(float a);
	//CWO operator-(CWO &a);
	//CWO operator-(float a);
	//CWO operator*(CWO &a);
	//CWO operator*(float a);
	//CWO operator/(CWO &a);
	//CWO operator/(float a);
	
	//@}

	//###########################################
	/** @defgroup commun Communication
	*/
	//@{


	//! Send data to GPU
    /*!
      \param a send the data of "CWO a" to GPU
    */
	virtual void Send(CWO &a);
	//! Receive data from GPU
    /*!
      \param a "CWO a" receives from the data on GPU
    */
	
	virtual void Recv(CWO &a);
	
	//@}

	//!  Create complex amplitude with Nx x Ny size (in pixel unit)
    /*!
	@param Nx : x size (in pixel unit)
	@param Ny : y size (in pixel unit)
    */
	int Create(int Nx, int Ny=1, int Nz=1); 

	//!  Release complex amplitude with Nx x Ny size (in pixel unit) from memory
    /*!
    */
	virtual void Destroy();


	void Attach(CWO &a);

	//###########################################
	/** @defgroup time Time mesurement
	*/
	//@{

	//! Start time mesurement
    /*!
    */
	void SetTimer(); 

	//! End time mesurement
    /*!
	@return Elapsed time (in millisecond) between SetTimer() and EndTimer()
    */
	double EndTimer(); 

	//@}



	//###########################################
	/** @defgroup params Parameters
	*/
	//@{

	//! Set context which manages the parameters of the wavelength, pixel pitch, etc.
    /*!
	@brief Set context which manages the parameters of the wavelength, pixel pitch, etc.
	@param c Context
	*/	
	void SetCtx(cwoCtx &c){ctx=c;};
	
	//! Get context which manages the parameters of the wavelength, pixel pitch, etc.
    /*!
	@brief Get context which manages the parameters of the wavelength, pixel pitch, etc.
	@return Current context
	*/	
	cwoCtx GetCtx(){return ctx;};

	//! Set diffraction type. Do not use it.
    /*!
      \param type diffraction type (e.g. CWO_ANGULAR)
    */
	void SetCalcType(int type);
	
	//! Set the sampling rate on source and destination planes
    /*!
      \param px horizon sampling rate (in meter) 
	  \param py vertical sampling rate (in meter)
    */
	void SetPitch(float px, float py, float pz=10.0e-6f); 
	
	void SetPitch(float p); 
	
	//! Set the sampling rate on source plane only
    /*!
      \param px horizon sampling rate (in meter) 
	  \param py vertical sampling rate (in meter)
    */
	void SetSrcPitch(float px, float py, float pz=10.0e-6f); //!< Set sampling rate on source plane
	
	void SetSrcPitch(float p); 

	//! Set the sampling rate on destination plane only
    /*!
      \param px horizon sampling rate (in meter) 
	  \param py vertical sampling rate (in meter)
    */
	void SetDstPitch(float px, float py, float pz=10.0e-6f); //!< Set sampling rate on destination plane
	
	void SetDstPitch(float p);

	//! Set the wavelength of light
    /*!
      \param w Wavelength of light (in meter) 
	*/
	void SetWaveLength(float w); //!< Set wavelength 

	//! Set the offset of source plane away from the origin. 
    /*!
	@param x Horizonal offset (in meter) 
	@param y Vertical offset (in meter) 
	*/
	void SetOffset(float x, float y, float z=0.0f);
	//! Set the offset of source plane away from the origin.
	/*!
	@brief The offset affects off-axis diffraction calculations (e.g. Shifted Fresnel diffraction).
	@param x Horizonal offset (in meter) 
	@param y Vertical offset (in meter) 
	*/
	void SetSrcOffset(float x, float y, float z=0.0f); 
	//! Set the offset of destination plane away from the origin.
    /*!
	@brief The offset affects off-axis diffraction calculations (e.g. Shifted Fresnel diffraction).
	@param x Horizonal offset (in meter) 
	@param y Vertical offset (in meter) 
	*/	
	void SetDstOffset(float x, float y, float z=0.0f); 

	//! set the number of threads
    /*!
	@param N : number of threads 
    */
	//void SetThreads(int Nx, int Ny=1);
	void SetThreads(int Nx, int Ny=1);


	//! get the number of threads
    /*!
	@return number of threads
    */
	int GetThreads();

	//! get the number of threads along to x-axis
    /*!
	@return number of threads along to x-axis
    */
//	int GetThreadsX();
	//! get the number of threads along to y-axis
    /*!
	@return number of threads along to y-axis
    */
//	int GetThreadsY();

	//! get x size of the field
    /*!
	@return x size (in meter unit)
    */
	size_t GetNx();
	//! get y size of the field
    /*!
	@return y size (in meter unit)
    */
	size_t GetNy(); 
	size_t GetNz();

	//! get wavelength
    /*!
	@return wavelength (in meter unit)
    */
	float GetWaveLength();

	//! get wave number
    /*!
	@return wave number (in meter unit)
    */
	float GetWaveNum();
	//! get propagation distance
    /*!
	@return propagation distance (in meter unit)
    */
	float GetDistance(); 
	//! get sampling rate along to x-axis on source plane
    /*!
	@return sampling rate along to x-axis on source plane (in meter unit)
    */
	float GetPx();
	//! get sampling rate along to y-axis on source plane
    /*!
	@return sampling rate along to y-axis on source plane (in meter unit)
    */
	float GetPy();
	float GetPz();
	//! get sampling rate along to x-axis on source plane
    /*!
	@return sampling rate along to x-axis on source plane (in meter unit)
    */
	float GetSrcPx();
	//! get sampling rate along to y-axis on source plane
    /*!
	@return sampling rate along to y-axis on source plane (in meter unit)
    */
	float GetSrcPy();
	float GetSrcPz();
	//! get sampling rate along to x-axis on destination plane
    /*!
	@return sampling rate along to x-axis on destination plane (in meter unit)
    */
	float GetDstPx();
	//! get sampling rate along to y-axis on destination plane
    /*!
	@return sampling rate along to y-axis on destination plane (in meter unit)
    */
	float GetDstPy();
	float GetDstPz();

	//! get offset along to x-axis on source plane
    /*!
	@return offset along to x-axis on source plane (in meter unit)
    */
	float GetOx();
	//! get offset along to y-axis on source plane
    /*!
	@return offset along to y-axis on source plane (in meter unit)
    */
	float GetOy();
	float GetOz();
		
	//! get offset along to x-axis on source plane
    /*!
	@return offset along to x-axis on source plane (in meter unit)
    */
	float GetSrcOx();
	//! get offset along to y-axis on source plane
    /*!
	@return offset along to y-axis on source plane (in meter unit)
    */
	float GetSrcOy();
	float GetSrcOz();

	//! get offset along to x-axis on destination plane
    /*!
	@return offset along to x-axis on destination plane (in meter unit)
    */
	float GetDstOx();
	//! get offset along to y-axis on destination plane
    /*!
	@return offset along to y-axis on destination plane (in meter unit)
    */
	float GetDstOy();
	float GetDstOz();

	//! get physical length of source along to x-axis
    /*!
	@return length of source along to x-axis (in meter unit)
    */
	float GetLx();

	//! get physical length of source along to y-axis
    /*!
	@return length of source along to y-axis (in meter unit)
    */
	float GetLy();

	//! get physical length of source along to x-axis
    /*!
	@return length of source along to x-axis (in meter unit)
    */
	float GetSrcLx();

	//! get physical length of source along to y-axis
    /*!
	@return length of source along to y-axis (in meter unit)
    */
	float GetSrcLy();

	//! get physical length of destication along to x-axis
    /*!
	@return length of destication along to x-axis (in meter unit)
    */
	float GetDstLx();

	//! get physical length of destication along to y-axis
    /*!
	@return length of destication along to y-axis (in meter unit)
    */
	float GetDstLy();

	//! get aliasing-free radius of approximated spherical wave
	/*!
	@return aliasing-free radius
	*/
	int GetAliasingFreeApproxSphWave(float z, float p, float wl);



	//! Get current field-type.
    /*!
    @return Current field-type. (e.g. CWO_FLD_COMPLEX)
    */
	int GetFieldType();
	//! Get current field-type as string.
    /*!
    @return Current field-type as string. (e.g. string "CWO_FLD_COMPLEX")
    */
	char *GetFieldName();



	//@}
	
	//###########################################
	/** @defgroup file File
	*/
	//@{
	int CheckExt(const char* fname, const char* ext);
	
	//! Load image file or cwo file. 
	/*! 	
	@param fname Filename
	@param c select color to be read the color image file. The default value is CWO_GREY.
	@return When the function success to load file, the return value is CWO_SUCCESS, otherwise CWO_ERROR.
	@note  This function can load bitmap format file. If you use jpeg, tiff, png image formats, you need to install ImageMagik. 
	*/
	int Load(char* fname, int c=CWO_GREY);

	//! Load image file or cwo file. 
	/*! 	
	@param fname_amp filename as amplitude
	@param fname_pha filename as argument
	@param c select color to be read the color image file. The default value is CWO_GREY.
	@return When the function success to load file, the return value is CWO_SUCCESS, otherwise CWO_ERROR.
	@note  This function can load bitmap format file. If you use jpeg, tiff, png image formats, you need to install ImageMagik. 
	*/
	int Load(char* fname_amp, char *fname_pha, int c=CWO_GREY);

	//! Save image file or cwo file. 
	/*! 	
	@param fname filename
	@param r 
	@param g 
	@param b 
	@return When the function success to load file, the return value is CWO_SUCCESS, otherwise CWO_ERROR.
	@note  This function can load bitmap format file. If you use jpeg, tiff, png image formats, you need to install ImageMagik. 
	*/	
	int Save(char* fname, int bmp_8_24=24);
	int Save(char* fname, CWO *r, CWO *g=NULL, CWO *b=NULL);
	
	int SaveMonosToColor(char* fname, char *r_name, char *g_name, char *b_name);
	int SaveAsImage(char* fname, float i1, float i2, float o1, float o2, int flag=CWO_SAVE_AS_RE, int bmp_8_24=24);
	int SaveAsImage(char* fname, int flag=CWO_SAVE_AS_RE, int bmp_8_24=24);
	virtual int SaveAsImage(cwoComplex *p, int Nx, int Ny, char* fname, int flag=CWO_SAVE_AS_RE, int bmp_8_24=24);


	int SaveLineAsText(char* fname, int flag);
	int SaveLineAsText(char* fname, int flag, int x1, int y1, int x2, int y2);

	//@}


	//###########################################
	/** @defgroup diffract Diffraction calculation
	*/
	//@{

	//! Get pointer to the buffer of the complex amplitude
	/*! 	
	@return pointer to the buffer of the complex amplitude
	*/	
	//void* GetBuffer(int flag=CWO_BUFFER_FIELD);
	cwoComplex* GetBuffer();


	int CmpCtx(cwoCtx &a, cwoCtx &b);
	
	//! Get required memory with Nx x Ny size in cwoComplex
	/*! 	
	@return required memory with Nx x Ny size in cwoComplex (bytes)
	*/	
	size_t GetMemSizeCplx();
	//! Get required memory with Nx x Ny size in float
	/*! 	
	@return required memory with Nx x Ny size in float (bytes)
	*/	
	size_t GetMemSizeFloat();
	//! Get required memory with Nx x Ny size in char
	/*! 	
	@return required memory with Nx x Ny size in char (bytes)
	*/	
	size_t GetMemSizeChar();

	//! Calculate diffraction
	/*! 	
	@param d propogation distance
	@param type type of diffraction. 
	*/	
	int GetDfrExpand();
	void SetDfrExpand(int flag=0);


	void Diffract(float d, int type=CWO_ANGULAR, cwoInt4 *zp=NULL, cwoComplex *knl_mask=NULL, CWO *numap=NULL);
	//void Diffract(float d, int type, CWO *tmp_buf1,CWO *tmp_buf2);
	void Diffract(float d, int type, CWO *tmp_buf);
	void Diffract(int type, CWO *kernel[]);

	void DiffractConv(float d, int type, cwoComplex *ape, cwoComplex *prop, cwoInt4 *zp=NULL, cwoComplex *knl_mask=NULL, CWO *numap=NULL, int prop_calc=1);
	//void DiffractConvImplicit(float d, int type, cwoComplex *ape, cwoComplex *prop, int prop_calc=1);
	void DiffractFourier(float d, int type, cwoInt4 *zp=NULL);
	void DiffractDirect(float d, CWO *snumap, CWO *dnumap);


	void Diffract3D(CWO &a, float d, int type);
	void Diffract(float d, int type, int Dx, int Dy, char *dir=NULL);

	void DiffractDepth(float z, float dz, int num, 
		const char* path, const char *name, const char *ext, int type=CWO_ANGULAR);


	void AngularProp(CWO *a);
	void ShiftedAngularProp(CWO *a);

	void FresnelConvProp(CWO *a);
	void FresnelConvCoeff(CWO *a);

	void ShiftedFresnelProp(CWO *a);
	void ShiftedFresnelAperture(CWO *a);
	void ShiftedFresnelCoeff(CWO *a);

	void ARSSFresnelProp(CWO *a);
	void ARSSFresnelAperture(CWO *a);
	void ARSSFresnelCoeff(CWO *a);

	void FresnelFourierProp(CWO *a);
	void FresnelFourierCoeff(CWO *a);

	void FresnelDblAperture(CWO *a,float z1);
	void FresnelDblFourierDomain(CWO *a, float z1, float z2, cwoInt4 *zp);
	void FresnelDblCoeff(CWO *a, float z1, float z2);

	void FresnelDupProp(CWO *a, float M);

	//CWO* PreDiffract(float z, int type, CWO *a);
	void PreDiffract(float z, int type, CWO *a[]);

	void ReleaseTmpBuffer();

	void FresnelCoeff(float z1, float z2, float wl1, float wl2);

	//void AngularProp(float z, int iNx, int iNy);

		
	float fresnel_c(float x);
	float fresnel_s(float x);
	void FresnelInt(
		float z, int x1, int y1, int x2, int y2);
	//@}

	//###########################################
	/** @defgroup waves Planar & Spherical waves
	@image html sphericalwave.jpg "Geometry for spherical wave"
	*/
	//@{

	virtual void __AddSphericalWave(cwoComplex *p, float x, float y, float z, float px, float py, float a);
	virtual void __MulSphericalWave(cwoComplex *p, float x, float y, float z, float px, float py, float a);
	
	


	//! Add spherical wave to the field. The equation is as follows:@n
	//! \f$ u(x,y)=a \exp(i k \sqrt{(x-x_0)^2+(y-y_0)^2+z^2}) \f$ where \f$ k \f$ is the wave number.
	/*! 
	@param x : center of the spherical wave 
	@param y : center of the spherical wave
	@param z : distance between the field and the spherical wave
	@param a : amplitude of the spherical wave
	
	*/
	void AddSphericalWave(float x, float y, float z, float a=1.0f);
	//! Mutily spherical wave to the field.
	/*! 
	@param x : center of the spherical wave 
	@param y : center of the spherical wave
	@param z : distance between the field and the spherical wave
	@param a : amplitude of the spherical wave
	*/
	void MulSphericalWave(float x, float y, float z, float a=1.0f);
	//! Add spherical wave to the field.
	/*! 
	@param x : center of the spherical wave 
	@param y : center of the spherical wave
	@param z : distance between the field and the spherical wave
	@param px : sampling rate along to x-axis on the field
	@param py : sampling rate along to y-axis on the field
	@param a : amplitude of the spherical wave
	*/
	void AddSphericalWave(float x, float y, float z, float px, float py, float a=1.0f);
	//! Multiply spherical wave to the field.
	/*! 
	@param x : center of the spherical wave 
	@param y : center of the spherical wave
	@param z : distance between the field and the spherical wave
	@param px : sampling rate along to x-axis on the field
	@param py : sampling rate along to y-axis on the field
	@param a : amplitude of the spherical wave
	*/
	void MulSphericalWave(float x, float y, float z, float px, float py, float a=1.0f);
	//! Add Fresnel-approximated spherical wave to the field.
	/*! 
	@param x : center of the spherical wave 
	@param y : center of the spherical wave
	@param z : distance between the field and the spherical wave
	@param px : sampling rate along to x-axis on the field
	@param py : sampling rate along to y-axis on the field
	@param a : amplitude of the spherical wave
	*/

	//! Add Fresnel-approximated spherical wave to the field.
	/*! 
	@param x : center of the spherical wave 
	@param y : center of the spherical wave
	@param phi : initial phase of the spherical wave
	@param px : sampling rate along to x-axis on the field
	@param py : sampling rate along to y-axis on the field
	@param a : amplitude of the spherical wave
	*/

	//virtual void __AddApproxSphWave(cwoComplex *p, float x, float y, float z, float px, float py, float a);
	virtual void __AddApproxSphWave(cwoComplex *p, float x, float y, float phi, float zx, float zy, float px, float py, float a);
	
	void AddApproxSphWave(float x, float y, float z, float a=1.0f);
	void AddApproxSphWave(float x, float y, float z, float px, float py, float a=1.0f);
	void AddApproxSphWave(float x, float y, float phi, float zx, float zy, float px, float py, float a=1.0f);
	
	//! Multiply Fresnel-approximated spherical wave to the field.
	/*! 
	@param x : center of the spherical wave 
	@param y : center of the spherical wave
	@param phi : initial phase of the spherical wave
	@param px : sampling rate along to x-axis on the field
	@param py : sampling rate along to y-axis on the field
	@param a : amplitude of the spherical wave
	*/
	//virtual void __MulApproxSphWave(cwoComplex *p, float x, float y, float phi, float px, float py, float a);
	virtual void __MulApproxSphWave(cwoComplex *p, float x, float y, float phi, float zx, float zy, float px, float py, float a);

	void MulApproxSphWave(float x, float y, float z, float a=1.0f);
	void MulApproxSphWave(float x, float y, float z, float px, float py, float a=1.0f);
	void MulApproxSphWave(float x, float y, float phi, float zx, float zy, float px, float py, float a=1.0f);


	//! Add planar wave to the field.
	/*! 
	@param kx : wave number along to x-axis on the field
	@param ky : wave number along to y-axis on the field
	@param px : sampling rate along to x-axis on the field
	@param py : sampling rate along to y-axis on the field
	@param a : amplitude of the planar wave
	*/
	void AddPlanarWave(float kx, float ky, float px, float py, float a=1.0f);
	//! Add planar wave to the field.
	/*! 
	@param kx : wave number along to x-axis on the field
	@param ky : wave number along to y-axis on the field
	@param a : amplitude of the planar wave
	*/
	void AddPlanarWave(float kx, float ky, float a=1.0f);

	//! Multiply planar wave to the field.
	/*! 
	@param kx : wave number along to x-axis on the field
	@param ky : wave number along to y-axis on the field
	@param kz : wave number along to z-axis on the field
	@param px : sampling rate along to x-axis on the field
	@param py : sampling rate along to y-axis on the field
	@param a : amplitude of the planar wave
	*/

	void MulPlanarWave(float kx, float ky, float px, float py, float a=1.0f);
	//! Multiply planar wave to the field.
	/*! 
	@param kx : wave number along to x-axis on the field
	@param ky : wave number along to y-axis on the field
	@param kz : wave number along to z-axis on the field
	@param a : amplitude of the planar wave
	*/
	void MulPlanarWave(float kx, float ky, float a=1.0f);

	//@}	

	//###########################################
	/** @defgroup cplx_ope Operations for complex amplitude
	*/
	//@{
	
	//! Taking only the real part of complex amplitude.
	/*! 	
	@note 
	\f$ Re[a] \leftarrow Re[a] \f$ @n
	\f$ Im[a] \leftarrow 0  \f$ 
	*/
	void Re(); 

	//! Taking only the imaginary part of complex amplitude.
	/*! 	
	@note 
	\f$ Re[a] \leftarrow Im[a] \f$ @n
	\f$ Im[a] \leftarrow 0  \f$ 
	*/
	void Im(); 
	
	//! Calculating the complex conjugation of complex amplitude
	/*! 	
	@note 
	\f$ Re[a] \leftarrow Re[a] \f$ @n
	\f$ Im[a] \leftarrow -Im[a]\f$ 
	*/
	void Conj();

	//! Calculating the absolute square (light intensity) of complex amplitude
	/*! 	
	@note 
	\f$ Re[a] \leftarrow Re[a]^2+Im[a]^2 \f$ @n
	\f$ Im[a] \leftarrow 0 \f$ 
	*/
	void Intensity();

	//! Calculating the amplitude of complex amplitude
	/*! 	
	@note 
	\f$ Re[a] \leftarrow \sqrt{Re[a]^2+Im[a]^2} \f$ @n
	\f$ Im[a] \leftarrow 0 \f$ 
	*/
	void Amp();

	//! Calculating complex amplitude with the constant amplitude .
	/*! 
	@param offset : adjust the phase distribution
	@note
	\f$ \theta=\tan^{-1}\frac{Im[a]}{Re[a]} \f$@n
	\f$ Re[a] \leftarrow \cos(\theta) \f$ @n
	\f$ Im[a] \leftarrow \sin(\theta) \f$ 
	*/
	void Phase(float offset=0.0f);


	//! Calculating the argument of complex amplitude with real value from \f$-\pi\f$ to \f$+\pi\f$.@n
	/*! 
	@param scale : adjust scale, ex. if setting scale=1.0, the phase range is \f-$\pi\f to +\f$\pi\f$. if setting scale=1/\f-2$\pi\f, the phase range is -0.5 to +0.5.
	@param offset : adjust the phase range, ex. if setting offset=CWO_PI, the phase range in the phase distribution is 0 to \f$2\pi\f$.
	@note 
	\f$ Re[a] \leftarrow tan^{-1}(\frac{Im[a]}{Re[a]})+offset  \f$ @n
	\f$ Im[a] \leftarrow 0  \f$ 

	\code
	//sample code of generating kinoform
	CWO a;
	a.Load("test.bmp");
	a.Diffract(0.1);
	a.Arg();//This kinoform has the argument range of -pi to +pi
	a.SaveAsImage("kinoform.bmp",CWO_SAVE_AS_RE);
	\endcode

	\code
	//sample code of generating kinoform.
	CWO a;
	a.Load("test.bmp");
	a.Diffract(0.1);
	a.Arg(CWO_PI);//This kinoform has the argument range of 0 to 2pi
	a.SaveAsImage("kinoform.bmp",CWO_SAVE_AS_RE);
	\endcode
	*/
	void Arg(float scale=1.0f, float offset=0.0f);

	//! I do not recommend to use this function. I will abolish the function in future CWO++ library.
//	void Cplx();  
	//! I do not recommend to use this function. I will abolish the function in future CWO++ library.
//	int Cplx(CWO &amp, CWO &ph);


	void Expi();


	
	//! Convering argument (real value) to complex amplitude.@n
	//! Concreately, the following calculation is done.
	/*! 
	@param offset : .
	@note 
	\f$ Re[a] \leftarrow tan^{-1}(\frac{Im[a]}{Re[a]})+offset  \f$ @n
	\f$ Im[a] \leftarrow 0  \f$ 

	\code
	//sample code of generating kinoform
	CWO a;
	a.Load("test.bmp");
	a.Diffract(0.1);
	a.Arg();//This kinoform has the argument range of -pi to +pi
	a.SaveAsImage("kinoform.bmp",CWO_SAVE_AS_RE);
	\endcode
	*/
	void Arg2Cplx(float scale=1.0, float offset=0.0);


	void Cart2Polar();
	void Polar2Cart();


	
	//! replace the real and imaginary parts.@n
	/*! 
	@param re : 
	@param im : 
	@note
	CWO a @n
	\f$ Re[a] \leftarrow Re[re]  \f$ @n
	\f$ Im[a] \leftarrow Re[im]  \f$ 
	
	\code
	//The real and imaginary parts of c[2] are replaced by the real parts of c[0] and c[1], respectively 
	CWO c[3];
	c[2].ReIm(c[0],c[1]);
	\endcode
	*/
	void ReIm(CWO &re, CWO &im);

	//@}	

void Char(); //conver to current data type to char
void Float(); //conver to current data type to float

	//! Take the square root of only the real part of the complex amplitude field
	/*! 
	*/
	virtual void SqrtReal();
	//! Take the square root of the amplitude of the complex amplitude field
	/*! 
	*/
	virtual void SqrtCplx();

	void Div(float a);

//	cwoComplex Mul(cwoComplex a, cwoComplex b);
	cwoComplex Conj(cwoComplex a);

	//
	cwoComplex Polar(float amp, float arg);


	//###########################################
	/** @defgroup pixel Pixel operations
	*/
	//@{

	void SetPixel(int x, int y, float a); 
	void SetPixel(int x, int y, float amp, float ph);
	void SetPixel(int x, int y, cwoComplex a);
	void SetPixel(int x, int y, int z, float a); 
	void SetPixel(int x, int y, int z, float amp, float ph);
	void SetPixel(int x, int y, int z, cwoComplex a);
	void SetPixel(int x, int y, CWO &a);
	
	void AddPixel(int x, int y, cwoComplex a);
	void AddPixel(int x, int y, CWO &a);

	void MulPixel(int x, int y, float a);
	void MulPixel(int x, int y, cwoComplex a);

	void GetPixel(int x, int y, float &a);
	void GetPixel(int x, int y, cwoComplex &a);
	void GetPixel(int x, int y, int z, float &a);
	void GetPixel(int x, int y, int z, cwoComplex &a);
	cwoComplex GetPixel(int m, int n);	
	void GetPixel(int x, int y, CWO &a);

	void GetLineH(int n, int N, CWO &a);
	void GetLineV(int m, int N, CWO &a);
	void GetLine(int m1, int n1, int m2, int n2, CWO &a);
	void SetLineH(int n, CWO &a);
	void SetLineV(int m, CWO &a);
	void SetLine(int m1, int n1, int m2, int n2, CWO &a);

	virtual void __Copy(
		cwoComplex *src, int x1, int y1, int sNx, int sNy,
		cwoComplex *dst, int x2, int y2, int dNx, int dNy, 
		int Sx, int Sy);

	void Copy(CWO &a, int x1, int y1, int x2,int y2, int Sx, int Sy);
	//CWO Extract(int x, int y, int Nx, int Ny);

	void Clear(int c=0);
	virtual void Fill(cwoComplex a);

	template <class T> void FlipH(T *a);
	template <class T> void FlipV(T *a);
	void Flip(int mode=0);

	void Crop(int x1, int y1, int Sx, int Sy);
	void Crop(int x1, int y1, int Sx, int Sy, CWO &a);

	void ShiftX(int s, int flag=0);
	void ShiftY(int s, int flag=0);
	
	void Transpose();
	virtual void __Transpose(cwoComplex *pi, cwoComplex *po);

	void JoinH(CWO &a);
	void JoinV(CWO &a);

	//! fill inside and outside rectangular area
	/*! 
	@param (m, n) : start position (in pixel unit)
	@param (Sx,Sy) : area size (in pixel unit)
	@param a : complex value
	@param flag : when CWO_FILL_INSIDE, the function fills inside the rectangular area with $sx \times sy$.
	*/
	void Rect(int m, int n, int Sx, int Sy, cwoComplex a, int flag=CWO_FILL_INSIDE);
	
	//! fill inside or outside circular area with radius r
	/*!
	@param m center position (in pixel unit)
	@param n center position (in pixel unit)
	@param r radius (in pixel unit)
	@param a complex value
	@param flag fill inside or outside (CWO_FILL_INSIDE, CWO_FILL_OUTSIDE)
	*/
	void Circ(int m, int n, int r, cwoComplex a, int flag=CWO_FILL_INSIDE);
	
	//! multiply complex value in circular area
	/*!
	@param m center position (in pixel unit)
	@param n center position (in pixel unit)
	@param r radius (in pixel unit)
	@param a complex value
	*/
	void MulCirc(int m, int n, int r, cwoComplex a);



	///////////////////
	//! convolve the field with filter kernel
	/*!
	@param a filtered field
	@param ker filter kernel
	@param Sx size of filter kernel
	@param Sy size of filter kernel
	*/
	void Convolve(CWO &a, float *ker, int Sx, int Sy);

	void ConvolveFFT(CWO *a, CWO *b, CWO *c);

	//
	virtual void Expand(int Nx, int Ny);
	void ExpandMirror();
	void ExpandTwice(CWO *src);
	void ExpandHalf(CWO *dst);


	

	//! scale current pixels (real part of complex amplitude) by specified maximum value
	/*! 
	@param lim maximum value
	@image html scale1.jpg
	@note
	this function convert current pixels by the following equation:
	@f$output~pixel = \frac{input~pixel - min}{max-min} \times lim @f$
	where @f$ max, min @f$ are the maximum and minimum current pixel,rerspectively.
		
	\code
	CWO a;
	a.Load("lena.bmp"); //load bitmap image with 256 gray value
	a.Scale(128); //the maximum value of the bitmap image is 128
	\endcode
	*/
	int ScaleReal(float lim=1.0);

	//! scale real part of complex amplitude by specified range
	/*! 
	@param i1 
	@param i2  
	@param o1
	@param o2
	@image html scale2.jpg
	@note 
	convert the current real part from the minimum value i1 to o1.
	similarly, convert the current real part from the maximum value i2 to o2.
	\code
	CWO a;
	a.Load("lena.bmp"); //load bitmap image with 256 gray value
	a.Scale(0,255,255,0); //each pixel value of the bitmap image is inverted 
	\endcode
	*/
	int ScaleReal(float i1, float i2, float o1, float o2);

	virtual int __ScaleReal(float i1, float i2, float o1, float o2);

	//! scale complex amplitude by specified maximum value
	/*! 
	@param lim maximum value
	]@image html scale1.jpg
	*/
	int ScaleCplx(float lim=1.0);
	
	//! scalecomplex amplitude by specified range
	/*! 
	@param i1 
	@param i2  
	@param o1
	@param o2
	@image html scale2.jpg
	@note 
	convert the current complexc amplitude from the minimum value i1 to o1.
	similarly, convert the current complexc amplitude from the maximum value i2 to o2.
	*/
	int ScaleCplx(float i1, float i2, float o1, float o2);

	virtual int __ScaleCplx(float i1, float i2, float o1, float o2);
	
	//@}	


	//###########################################
	/** @defgroup rand Random
	*/
	//@{
	//! set the seed of random sequence to generate other 
	/*! 
	@param s seed
	@code
	#include <time.h>
	CWO a(1024,1024);
	a.SetRandSeed((unsigned)time(NULL)); //set current time as seed
	@endcode
	*/
	virtual void SetRandSeed(long long int s);
	long long int GetRandSeed();

	//! get random value 
	/*! 
	@return uniform random value from 0.0 to 1.0 
	@note
	this random value is generated by Mersenne twister
	*/
	float GetRandVal();


	cwoComplex GetRandComplex();
	virtual void __RandPhase(cwoComplex *a, float max, float min);
	virtual void __MulRandPhase(cwoComplex *a, float max, float min);

	virtual void RandReal(float max = 1.0f, float min = 0.0f);
	void RandPhase(float max=CWO_PI, float min=-CWO_PI);
	void SetRandPhase(float max=CWO_PI, float min=-CWO_PI);
	void MulRandPhase(float max=CWO_PI, float min=-CWO_PI);
	//@}	
	



	void __Hanning(cwoComplex *a, int m, int n, int Wx, int Wy); //!< Hanning window
	void __Hamming(cwoComplex *a, int m, int n, int Wx, int Wy); //!< Hamming window
	void __WindowCos(cwoComplex *a, int m, int n, int W, int Wa);
	void __WindowTukey(cwoComplex *a, int m, int n, float r); //!<Tukey window
	void __WindowSigmoid(cwoComplex *p, int m, int n, float a, float Tmax, float Tmin); //!<Sigmoid window

	//cwoComplex inter_nearest(cwoComplex *old, double x, double y);
	//cwoComplex inter_linear(cwoComplex *old, float x, float y);
	//cwoComplex inter_cubic(cwoComplex *old, float x, float y);
	
	/*
	virtual void __InterNearest(
		cwoComplex *p_new, int newNx, int newNy,
		cwoComplex *p_old, int oldNx, int oldNy,
		float mx, float my,
		int xx, int yy);

	virtual void __InterLinear(
		cwoComplex *p_new, int newNx, int newNy,
		cwoComplex *p_old, int oldNx, int oldNy,
		float mx, float my,
		int xx, int yy);

	virtual void __InterCubic(
		cwoComplex *p_new, int newNx, int newNy,
		cwoComplex *p_old, int oldNx, int oldNy,
		float mx, float my,
		int xx, int yy);
*/
	virtual void __ResizeNearest(
		cwoComplex *p_new, int newNx, int newNy,
		cwoComplex *p_old, int oldNx, int oldNy);

	virtual void __ResizeLinear(
		cwoComplex *p_new, int newNx, int newNy,
		cwoComplex *p_old, int oldNx, int oldNy);

	virtual void __ResizeCubic(
		cwoComplex *p_new, int newNx, int newNy,
		cwoComplex *p_old, int oldNx, int oldNy);

	virtual void __ResizeLanczos(
		cwoComplex *p_new, int newNx, int newNy,
		cwoComplex *p_old, int oldNx, int oldNy);


	void AddNoiseWhite(float max=1.0);
	void MulNoiseWhite(float max=1.0);
	void AddNoiseGaussian(float mu, float sigma);
	void MulNoiseGaussian(float mu, float sigma);


	//###########################################
	/** @defgroup image Image Processing
	*/
	//@{

	
	//! resize an image @n
	/*! 
	@param dNx new size along to horizontal direction 
	@param dNy new size along to vertical direction 
	@param flag interpolaion method (CWO_INTER_NEAREST, CWO_INTER_LINEAR, CWO_INTER_CUBIC,CWO_INTER_LANCSOZ)
	@note
	GetNx() and GetNy() returns new image sizes.	

	\code
	CWO a;
	a.Load("test.bmp");
	a.Resize(a.GetNx()*1.5,a.GetNy()*1.2,CWO_INTER_LINEAR); // the image is resized by horizontally 1.5 times and vertically 1.2 times, respectively
	\endcode
	*/
	void Resize(int dNx, int dNy, int flag=CWO_INTER_LINEAR);
	void Rotate(float deg);

	void AffineAngularSpectrum(float *mat_affine, float px, float py, int flag=CWO_INTER_LINEAR);

	virtual void ErrorDiffusion(CWO *a=NULL, int flag=CWO_ED_FS);
	virtual void ErrorDiffusionSegmented(CWO *output, int flag = CWO_ED_FS);


	void RGB2YCbCr(CWO *rgb, CWO *ycbcr);
	void YCbCr2RGB(CWO *rgb, CWO *ycbcr);

	void MulMatrix(CWO *a, cwoComplex *b , CWO *c);
	//@}	

	//###########################################
	/** @defgroup signal Signal processing
	*/
	//@{
	void FourierShift(float m, float n);

	int FFT(int flag=0); //!< Fast Fourier Transform (FFT)
	virtual void FFTShift(); //!< FFT Shift 

	int FFT(int dx, int dy, int flag = 0); //!< Fast Fourier Transform (FFT) 
	virtual void FFTShift(int dx, int dy); //!< FFT Shift 
	
	virtual void __ScaledFFTCoeff(cwoComplex *p, float sx, float sy, cwoComplex alpha);
	virtual void __ScaledFFTKernel(cwoComplex *p, float sx, float sy, cwoComplex alpha);
	//void ScaledFFT(int flag=0); //!< Scaled FFT 

	virtual void __NUFFT_T1(cwoComplex *p_fld, cwoFloat2 *p_x, int R=2, int Msp=12);
	virtual void __NUFFT_T2(cwoComplex *p_fld, cwoFloat2 *p_x, int R=2, int Msp=12);
	
	//void NUFFT_T1(int R=2, int Msp=12);//!< Non-uniform FFT (Type1) 
	//void NUFFT_T2(int R=2, int Msp=12);//!< Non-uniform FFT (Type2) 

	void NUFFT1(CWO *map, int R=2, int Msp=12);//!< Non-uniform FFT (Type1) 
	void NUFFT2(CWO *map, int R=2, int Msp=12);//!< Non-uniform FFT (Type2) 

	void ConvertSamplingMap(int type); 
	void SamplingMapScaleOnly(int Nx, int Ny, float R, float sgn);
	//@}

	void Log(float base=10.0f);

	virtual void Gamma(float g);//OK
	virtual void Threshold(float max, float min=0.0);//OK
	virtual void Threshold(float max_th, float max_val, float min_th, float min_val);
	void ThresholdAmp(float max_amp, float min_amp);
	void ThresholdToZero(float th);
	void ThresholdAbsToZero(float th);
	void ThresholdHist(float var);
	void Binary(float th=0.0, float max=1.0, float min=0.0);//OK
	
	virtual void __PickupFloat(float *src, float *pix_p, float pix);
	virtual void __PickupCplx(cwoComplex *src, cwoComplex *pix_p, float pix);
	
	//! Pickup pixels
	/*!
	@brief if a(m,n)=pix , this(m,n)=this(m,n) otherwise this(m,n)=0 
	@param a : pointer to source image 
	@param pix : pixel value 
	*/
	void	Pickup(CWO *a, float pix);
	

	//! phase only correlation (POC)
	/*!
	@brief calculate phase only correlation, then the correlation result is maintained in this instance. 
	@param ref : reference image
	@return : if success, it returns CWO_SUCCESS, otherwise CWO_ERROR
	@note The image sizes of the target and reference images should be the same.
	the traget and ref is destroyed after POC.
	*/
	int POC(CWO &ref);


	//###########################################
	/** @defgroup stastics Stastics
	*/
	//@{

	unsigned int Count(float a);
	unsigned int CountNonZero();


	//! calculate the average of the real part
	/*! 
	@param m : coordinate of Region of Interest(ROI) 
	@param n : coordinate of ROI 
	@param Sx : ROI size
	@param Sy : ROI size
	@return average
	*/
	virtual float Average();

	float Average(int m, int n, int Sx, int Sy);
	
	//! calculate the variance of the real part
	/*! 
	@return variance
	*/
	virtual float Variance();

	float Variance(int m, int n, int Sx, int Sy);

	//! calculate the standard deviation
	/*! 
	@return standard deviation
	*/
	float StdDev(int m, int n, int Sx, int Sy);



	virtual void __MaxMin(cwoComplex *a, float *max, float *min, int *max_x=NULL, int *max_y=NULL,int *min_x=NULL, int *min_y=NULL);//OK

	//! find maximum of the real part
	/*! 
	@return maximum
	*/
	float Max(int *m=NULL, int *n=NULL);
	//! find minimum of the real part
	/*! 
	@return minimum
	*/
	float Min(int *m = NULL, int *n = NULL);

	//! find maxmum and minimum of the real part
	/*! 
	@param *max pointer to maxmim value
	@param *min pointer to minimum value
	@param *max_x pointer to maxmim value
	@param *max_y pointer to minimum value
	@param *min_x pointer to the location of maxmim value
	@param *min_y pointer to the location of minimum value
	@return reserved
	@code
	float max,min;
	CWO a;
	a.Load("test.bmp");
	a.MaxMin(&max,&min); //you can find max and min velues at the same time
	printf("max=%e min=%e\n",max,min); 
	@endcode
	*/
	int MaxMin(float *max, float *min, int *max_x=NULL, int *max_y=NULL,int *min_x=NULL, int *min_y=NULL);//OK

	//! generate variance map
	/*! 
	@param sx 
	@param sy 
	@note
	VAriance map is useful for detecting in-focus reconstructed planes from a hologram.@n
	Conor P. McElhinney, John B. McDonald, Albertina Castro, Yann Frauel, Bahram Javidi, and Thomas J. Naughton, "Depth-independent segmentation of macroscopic three-dimensional objects encoded in single perspectives of digital holograms," Opt. Lett. 32, 1229-1231 (2007) 
	*/
	void VarianceMap(int sx, int sy);

	//! generate histogram
	/*! 
	@param hist  pointer to frequency whose buffer is required to prepare by myself  
	@param N  bin number 
	@return  bin width. 

	\code
	//sample code of generating kinoform.
	int N=256, hist[256];
	CWO a;
	a.Load("test.bmp");
	float bin_width=a.Histogram(hist,N);
	\endcode
	*/
	float Histogram(int *hist, int N);

	//! Calculate total sum of real part
	/*! 
	@return  Total sum 
	*/
	virtual cwoComplex TotalSum();
	
	/*	 
	//
	float Variance(float ave);
	float SMDx();*/

	//@}




	//###########################################
	/** @defgroup error Error measurement
	*/
	//@{

	//! Measure MSE between target and reference images
	/**
	\code
	CWO ref,tar;
	ref.Load("reference_image.bmp");
	tar.Load("taeget_image.bmp");
	float mse=tar.MSE(ref);
	printf("MSE is %f\n",mse);
	\endcode
	**/
	virtual float MSE(CWO &ref);

	//! Measure SNR(Signal to Noise Ratio) between target and reference images
	/**
	\code
	CWO ref,tar;
	ref.Load("reference_image.bmp");
	tar.Load("taeget_image.bmp");
	float snr=tar.SNR(ref);
	printf("SNR is %f\n",snr);
	\endcode
	**/
	float SNR(CWO &ref);

	//! Measure PSNR between target and reference images
	/**
	\code
	CWO ref,tar;
	ref.Load("reference_image.bmp");
	tar.Load("taeget_image.bmp");
	float psnr=tar.PSNR(ref);
	printf("PSNR is %f\n",psnr);
	\endcode
	**/
	float PSNR(CWO &ref);

	//! Measure SSIM between target and reference images
	/**
	\code
	CWO ref,tar;
	ref.Load("reference_image.bmp");
	tar.Load("taeget_image.bmp");
	float ssim=tar.SSIM(ref);
	printf("SSIM is %f\n",ssim);
	\endcode
	**/
	float SSIM(CWO &ref);

	float SpeckleContrast(int m, int n, int Sx, int Sy);
	
	//@}

	//###########################################
	/** @defgroup phase_unwrap Phase Unwrapping
	*/
	//@{

	CWO LaplacianPhase();
	void PoisonSolverFFT();
	void PhaseUnwrap(int type=0);

	//@}



	virtual void test();
	void test2(CWO &a);

	//****************
	//For PLS
	//****************
	virtual cwoObjPoint* GetPointBuffer(){return NULL;};
	virtual int GetPointNum(){return 0;};
	virtual void SetPointNum(int num){};
	virtual void ScalePoint(float lim){};
	void PLS(int flag);

//	virtual void __PLS_Fresnel(){};
//	virtual void __PLS_CGH_Fresnel(){};

	virtual void __PLS_Huygens(float ph=0.0f){};
	virtual void __PLS_Fresnel(float ph=0.0f){};
	virtual void __PLS_CGH_Fresnel(float ph=0.0f){};


	
	
	

////////////////////////
//Test code
////////////////////////

//void SamplingMapX(cwoFloat2 *p, int Nx, int Ny, int quadrant);
//void SamplingMapY(cwoFloat2 *p, int Nx, int Ny, int quadrant);


//void NUDFT();

//void __ScaledAngularProp(cwoComplex *a, float px, float py);




virtual void __InvFx2Fy2(cwoComplex *a);

//###########################################
/** @defgroup	sdiff	Special diffractions
*/
//@{
float z0;
float pz;
float fresnel_c_tbl(float x, float z, float z0, float pz);
float fresnel_s_tbl(float x, float z, float z0, float pz);
void PrepareTblFresnelApproxInt(int nx, float z0, float pz, int nz);
void FresnelApproxIntTbl(
	float z, int x1, int y1, int sx1, int sy1);

void FresnelPolyAperture();	

void ParticleField(cwoFloat3 pos, float radius, float amp=1.0f, float init_ph=0.0f); //!< Complex amplitude from particle with radius
//@}


};


//inline cwoComplex cwoCplxPolar(float amp, float ph);
//inline cwoComplex cwoCplxCart(float r, float i);
inline cwoComplex cwoCplxPolar(float amp, float ph)
{
	cwoComplex tmp;
	CWO_RE(tmp)=amp*cos(ph);
	CWO_IM(tmp)=amp*sin(ph);
	return tmp;
}

inline cwoComplex cwoCplxCart(float r, float i)
{
	cwoComplex tmp;
	CWO_RE(tmp)=r;
	CWO_IM(tmp)=i;
	return tmp;
}


inline double cwoCplx2Arg(cwoComplex a)
{
	double arg = atan2(CWO_IM(a), CWO_RE(a));
	if (arg < 0.0f) arg += CWO_2PI;
	if (arg >= CWO_2PI) arg -= CWO_2PI;
	return arg;
}


char* cwoFname(char *arg, ...);
//!  Class for vector operations. 
/*!
This class serves vector operations.
*/
class cwoVect{
public:
    float x, y, z;
public:
	cwoVect(){};

	cwoVect(float tx, float ty, float tz) {
        x = tx; y = ty; z = tz;
    };

	void Set(float tx, float ty, float tz){
		 x = tx; y = ty; z = tz;
	}

	cwoVect operator=(cwoVect a){
        cwoVect c;
        x =a.x;
        y =a.y;
        z =a.z;
        return *this;
    };

	cwoVect operator+(cwoVect a){	//vector+vector
		cwoVect c;
		c.x = x+a.x;
		c.y = y+a.y;
		c.z = z+a.z;
		return c;
	}

    cwoVect operator-(cwoVect a){
        cwoVect c;
        c.x = x - a.x;
        c.y = y - a.y;
        c.z = z - a.z;
        return c;
    };

    cwoVect operator*(float a){
        cwoVect c;
        c.x = x * a;
        c.y = y * a;
        c.z = z * a;
        return c;
    };

	cwoVect operator/(float a){
        cwoVect c;
        if(a == 0.0f) return *this;
        c.x = x / a;
        c.y = y / a;
        c.z = z / a;
        return c;
    };

    cwoVect& operator+=(cwoVect a) {
        x += a.x;
        y += a.y;
        z += a.z;
        return *this;
    };

    cwoVect& operator-=(cwoVect a) {
		x -= a.x;
        y -= a.y;
        z -= a.z;
        return *this;
    };

    cwoVect& operator*=(float a){
        x *= a;
        y *= a;
        z *= a;
        return *this;
    };

    cwoVect& operator/=(float a){
        if(a == 0.0f) return *this;
        x /= a;
        y /= a;
        z /= a;
        return *this;
    };

	float Length(){
		double tx = (double)x;
		double ty = (double)y;
		double tz = (double)z;
		return (float)sqrt(tx*tx + ty*ty + tz*tz);
	}

	cwoVect Normalize(){
		cwoVect tmp;
		tmp=*this;

		float len=Length();
		len = 1.0f / len;
		tmp.x *= len;
		tmp.y *= len;
		tmp.z *= len;
		return tmp;
	}

	float Dot(cwoVect a){
		return (x*a.x+y*a.y+z*a.z);
	}

	cwoVect Cross(cwoVect a){
		cwoVect vec;
		double x1, y1, z1, x2, y2, z2;
        x1 = (double)x;
		y1 = (double)y;
		z1 = (double)z;
		x2 = (double)a.x;
		y2 = (double)a.y;
		z2 = (double)a.z;
 		vec.x = (float)(y1 * z2 - z1 * y2);
		vec.y = (float)(z1 * x2 - x1 * z2);
		vec.z = (float)(x1 * y2 - y1 * x2);
		return vec;
	}

};




//!  Class for look-up table
/*!
This class manages look-up tables.
*/
class cwoTbl{
	float *tbl_sin;
	float *tbl_cos;
	CWO *tbl_wrp;
	int __Nz;

	//! set the number of tables along to z
	/*! 
	@param Nz : the number of tables along to z
	@sa 
	*/
	void SetNz(int Nz);
public:
	
	cwoTbl();
	~cwoTbl();

	//! get the number of tables along to z
	/*! 
	@return the number of tables along to z
	*/
	int GetNz();

	//! Make sin table
	/*! 
	@param N : sampling number in one cycle of sin
	*/
	void MakeSin(int N);
	
	//! Make cos table
	/*! 
	@param N : sampling number in one cycle of cos
	*/
	void MakeCos(int N);

	//! Make WRP table
	/*! 
	@param z : distance from WRP
	@param Nz : number of planes along to z-axis
	@param pz : sampling rate between neighbor planes (in meter)
	@param wl : wavelength (in meter)
	@param px : sampling rate on WRP (in meter)
	@param py : sampling rate on WRP (in meter)
	*/
	void MakeWRPTbl(float z, int Nz, float pz, float wl, float px, float py);
	
	//! Clip WRP tables to the half size
	/*! 
	*/
	void ClipWRPTbl();
	

	//! Get WRP table as pointer to CWO 
	/*! 
	@param idx : index for plane
	@return pointer to CWO 
	*/
	CWO *GetWRPTbl(int idx);


};


template<class T>
class cwoMat{
private:	
	int dim; 
public:
	T *a;
};


#endif




/** @example  simple_diffraction.cpp
This code is to calculate diffraction using the angular spectrum method.
@image html lena512x512_diffract.jpg
*/

/** @example simple_diffraction_with_gpu.cpp
This code is to calculate diffraction using the angular spectrum method on a GPU.
@image html lena512x512_diffract.jpg
*/

/** @example "Amplitude CGH"
This code is to calculate an amplitude CGH.
@image html lena512x512_diffract.jpg
@code
CWO a;
a.Load("sample.bmp"); //Load image file 
a.Diffract(0.1, CWO_ANGULAR); //Calculate diffraction from the image using the angular spectrum method
a.Re(); //Taking the real part from the diffracted result
a.Scale(255); // Convert the intensity to 256 gray scale data
a.Save("diffract.bmp"); //Save the intensity as bitmap file
@endcode
*/

/** @example "Kinoform"
This code calculates a kinoform.
@image html lena512x512_diffract.jpg
@code
CWO a;
a.Load("sample.bmp"); //Load image file 
a.Diffract(0.1, CWO_ANGULAR); //Calculate diffraction from the image using the angular spectrum method
a.Arg(); //Taking the argument of the diffracted result
a.Scale(255); // Convert the argument to 256 gray scale data
a.Save("diffract.bmp"); //Save the argument as bitmap file
@endcode
*/

/** @example "Shifted-Fresnel diffraction"
This code calculates Shifted-Fresnel diffraction.
@code
CWO a;
a.Load("sample.bmp"); //Load image file
a.SetSrcPitch(10e-6,10e-6);
a.SetDstPitch(24e-6,24e-6);
a.Diffract(0.1, CWO_SHIFTED_FRESNEL); //Calculate diffraction from the image using the angular spectrum method
a.Intensity(); 
a.Scale(255); // Convert the intensity to 256 gray scale data
a.Save("diffract.bmp"); //Save the intensity as bitmap file
@endcode
*/

/** @example "Using CPU threads"
This code calculates Shifted-Fresnel diffraction with 8 CPU threads.
@code
CWO a;
a.Load("sample.bmp"); //Load image file
a.SetThreads(8);
a.SetSrcPitch(10e-6,10e-6);
a.SetDstPitch(24e-6,24e-6);
a.Diffract(0.1, CWO_SHIFTED_FRESNEL); //Calculate diffraction from the image using the angular spectrum method
a.Intensity(); 
a.Scale(255); // Convert the intensity to 256 gray scale data
a.Save("diffract.bmp"); //Save the intensity as bitmap file
@endcode

/** @example "Phase shifting digital holography"
This code calculates 4-steps phase shifting digital holography.
@code
	CWO c1,c2,c3,c4;

	c1.Load("lena1024.bmp","mandrill1024.bmp"); //Amp=lena, Arg=mandrill
	c1.Diffract(0.1);
	c1.ScaleCplx();

	c2=c1;
	c3=c1;
	c4=c1;
	
	//generating inline hologram 1
	c1+=c1.Polar(1.0f, 0.0f); //add reference light with no phase shift
	c1.Intensity(); 
	
	//generating inline hologram 2
	c2+=c2.Polar(1.0f, CWO_PI/2);//add reference light with phase shift of pi/2
	c2.Intensity();

	//generating inline hologram 3
	c3+=c3.Polar(1.0f, CWO_PI);//add reference light with phase shift of pi
	c3.Intensity();
	
	//generating inline hologram 4
	c4+=c4.Polar(1.0f, 3.0f/2.0f*CWO_PI);//add reference light with phase shift of 3pi/2
	c4.Intensity();

	//retrieve object wave on the inline holograms using 4-steps phase shifting digital holography
	c1-=c3;
	c2-=c4;
	c3.Clear();
	c3.ReIm(c1,c2);//The real and imaginary parts of c3 are replaced by the real parts of c1 and c2, respectively 

	//reconstruction
	c3.Diffract(-0.1);
	c3.SaveAsImage("phase_shift_lena.jpg",CWO_SAVE_AS_AMP);
	c3.SaveAsImage("phase_shift_mandrill.jpg",CWO_SAVE_AS_ARG);
@endcode

*/