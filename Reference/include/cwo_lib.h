// Copyright (C) Tomoyoshi Shimobaba 2011-


#ifndef _CWO_LIB_H
#define _CWO_LIB_H


#ifndef __linux__
	#ifdef CWODLL
		#define CWO_EXPORT   __declspec(dllexport)
	#else
		#define CWO_EXPORT   __declspec(dllimport)
	#endif
#else
	#define CWO_EXPORT
#endif

//if you want to generate DLL, omit the following comment out.
#ifndef __linux__
#define CWOAPI __stdcall
#else
#define CWOAPI 
#endif


#ifdef __cplusplus
extern "C" {
#endif


//###########################################
/** @defgroup conv Convolution
*/
//@{
/**
* @def CWO_CONV_EXPLICIT
* Convolution is caluculated by normal convolution (explicit convolution).
*/ 
#define CWO_CONV_EXPLICIT 0x00

/**
* @def CWO_CONV_IMPLICIT
* Convolution is caluculated by implicit convolution.
*/ 
#define CWO_CONV_IMPLICIT 0x80

//###########################################
/** @defgroup error Error
*/
//@{
/**
* @def CWO_ERR
*/ 
#define CWO_ERROR		(-1)

#define CWO_SUCCESS		(0)


//@}

//###########################################
/** @defgroup difftype Diffraction
*/
//@{

/**
* @def CWO_ANGULAR
* When selecting CWO_ANGULAR, the CWO library calculates the angular spectrum method.
* The Angular spectrum method is expressed as:

* @f{eqnarray*}
u(x,y) = \int \!\!\int_{-\infty}^{+\infty} & & 
A(f_x, f_y, 0) \exp(i z \sqrt{k^2 - 4 \pi^2 (f_x^2 + f_y^2)}) \\ 
& & \exp(i \ 2 \pi (f_x x + f_y y)) d f_x d f_y
* @f}

The CWO library calculates this diffraction using the FFT algorithm,

* @f{eqnarray*}
u(m_2,n_2) = FFT[ u(m_1, n_1) ] \exp(i z \sqrt{k^2 - 4 \pi^2 (f_x^2 + f_y^2)}) 
* @f}
*/

/**
* @def CWO_ANGULAR
* When selecting CWO_ANGULAR, the CWO library calculates the angular spectrum method.
* The Angular spectrum method is expressed as:

* @f{eqnarray*}
u(x,y) = \int \!\!\int_{-\infty}^{+\infty} & & 
A(f_x, f_y, 0) \exp(i z \sqrt{k^2 - 4 \pi^2 (f_x^2 + f_y^2)}) \\ 
& & \exp(i \ 2 \pi (f_x x + f_y y)) d f_x d f_y
* @f}

The CWO library calculates this diffraction using the FFT algorithm,

* @f{eqnarray*}
u(m_2,n_2) = FFT[ u(m_1, n_1) ] \exp(i z \sqrt{k^2 - 4 \pi^2 (f_x^2 + f_y^2)}) 
* @f}
*/
#define CWO_ANGULAR					(0x01)



/**
* @def CWO_HUYGENS
* When selecting CWO_HUYGENS, the CWO library calculates 
* the Huygens-Fresnel diffraction (convolution form).
*/
#define CWO_HUYGENS					(0x03)

/**
* @def CWO_FRESNEL_CONV
* When selecting CWO_FRESNEL_CONV, the CWO library calculates 
* the Fresnel diffraction (convolution form).
*/
#define CWO_FRESNEL_CONV			(0x04)

/**
* @def CWO_FRESNEL_FOURIER
* When selecting CWO_FRESNEL_ANALYSYS, the CWO library calculates 
* the Fresnel diffraction (fourier form).
*/
#define CWO_FRESNEL_FOURIER			(0x05)

/**
* @ingroup difftype
* @def CWO_FRESNEL_ANALYSYS
* When selecting CWO_FRESNEL_ANALYSYS, the CWO library calculates 
* the Fresnel diffraction (convolution form).
*/
#define CWO_FRESNEL_ANALYSIS		(0x06)

/**
* @ingroup difftype
* @def CWO_FRESNEL_DBL
* When selecting CWO_FRESNEL_DBL, the CWO library calculates 
* the double-step Fresnel diffraction.
*/
#define CWO_FRESNEL_DBL				(0x07)



/**
* @def CWO_FRAUNHOFER
* When selecting CWO_FRAUNHOFER, the CWO library calculates 
* the Fraunhofer diffraction.
*/
#define CWO_FRAUNHOFER				(0x08)

/**
* @def CWO_SHIFTED_ANGULAR
* When selecting CWO_SHIFTFRESNEL, the CWO library calculates 
* the Shfted-Fresnel diffraction.
*/
#define CWO_SHIFTED_ANGULAR			(0x09)
#define CWO_SCALED_ANGULAR			(0x0a)
#define CWO_SCALED_FRESNEL			(0x0b)


#define CWO_FRESNEL_ARBITRARY_DIRECT	(0x0c)
#define CWO_FRESNEL_ARBITRARY			(0x0d)
#define CWO_FRESNEL_ARBITRARY2			(0x0e)
#define CWO_FRAUNHOFER_ARBITRARY		(0x0f)

#define CWO_LAPLACIAN_WAVE_RETV			(0x10)

#define CWO_NU_ANGULAR1					(0x11)
#define CWO_NU_ANGULAR2					(0x12)

#define CWO_NU_FRESNEL1					(0x13)
#define CWO_NU_FRESNEL2					(0x14)

/**
* @def CWO_ARSS_FRESNEL
* When selecting CWO_ARSS_FRESNEL, the CWO library calculates 
* the aliasing-reduced scaled and shifted Fresnel diffraction.
*/
#define CWO_ARSS_FRESNEL			(0x14)

#define CWO_DIRECT_SOMMER			(0x15)


/**
* @def CWO_FFT
* When selecting CWO_FFT, the function Diffract executes FFT operation.
*/
#define CWO_FFT			(0x16)
/**
* @def CWO_IFFT
* When selecting CWO_IFFT, the function Diffract executes inverse FFT operation.
*/
#define CWO_IFFT			(0x17)

/**
* @def CWO_SCALED_FFT
* When selecting CWO_SCALED_FFT, the function Diffract executes scaled FFT operation.
*/
#define CWO_SCALED_FFT			(0x18)


/**
* @def CWO_SHIFTFRESNEL
* When selecting CWO_SHIFTED_FRESNEL, the CWO library calculates 
* the Shfted-Fresnel diffraction.
*/
#define CWO_SHIFTED_FRESNEL			(0x19)

/**
* @def CWO_SIMPLE_SASM
* When selecting CWO_SHIFTED_FRESNEL, the CWO library calculates 
* the Shfted-Fresnel diffraction.
*/
#define CWO_SIMPLE_SASM			(0x20)



//@}

#define CWO_DIFFRACT_WITHOUT_COFF		(0x8000)

#define CWO_CGH_BASIC				(0x20)
#define CWO_CGH_FRESNEL				(0x21)
#define CWO_CGH_RECURRENCE			(0x22)


#define CWO_PLS_FRESNEL				(0x30)
#define CWO_PLS_FRESNEL_CGH			(0x31)
#define CWO_PLS_HUYGENS				(0x32)
//#define CWO_PLS_HUYGENS_CGH			(0x33)



////////////////////////////////////////////////
//
////////////////////////////////////////////////
#define CWO_ON		1
#define CWO_OFF		0

////////////////////////////////////////////////
//
////////////////////////////////////////////////
//#define CWO_MEM_PAGELOCKED			(0x01)


////////////////////////////////////////////////
//define pi
////////////////////////////////////////////////
#define CWO_PI (3.1415926535897932384626433832795)
#define CWO_2PI (3.1415926535897932384626433832795f * 2.0f)
#define CWO_PI2 (3.1415926535897932384626433832795f / 2.0f)

////////////////////////////////////////////////
//type
////////////////////////////////////////////////

#define CWO_RE(a) ((a).cplx[0])
#define CWO_IM(a) ((a).cplx[1])

#define CWO_CONJ(a) (CWO_IM(a)*=-1.0) 
#define CWO_INTENSITY(a) (CWO_RE(a)*CWO_RE(a)+CWO_IM(a)*CWO_IM(a))
#define CWO_AMP(a) (sqrt(CWO_RE(a)*CWO_RE(a)+CWO_IM(a)*CWO_IM(a)))
#define CWO_PHASE(a) (atan2(CWO_IM(a),CWO_RE(a)))
#define CWO_ARG(a) (atan2(CWO_IM(a),CWO_RE(a)))


//convert degree to radian
#define CWO_RAD(deg)	(deg*CWO_PI/180.0)

//convert radian to degree
#define CWO_DEG(rad)	(rad*180.0/CWO_PI)


////////////////////////
#ifdef __CUDACC__ //for CUDA
	#define HOST_DEV __host__ __device__ 
#else //for host
	#define HOST_DEV 
#endif


//!  Structure for complex number
/*!
This structure serves complex number and its operations.
*/

typedef struct cwoComplex{
	float cplx[2];
	//cwoComplex(){};
	//cwoComplex(float r, float i){cplx[0]=r;cplx[1]=i;};

	void Re(const float a){cplx[0]=a;};
	void Im(const float a){cplx[1]=a;};

	/*void Polar(const float amp, const float ph){
		cplx[0]=amp*cos(ph);
		cplx[1]=amp*sin(ph);
	};*/

	HOST_DEV cwoComplex& operator=(const cwoComplex &a){
		cplx[0]=a.cplx[0]; cplx[1]=a.cplx[1];
		return *this;
	};

	HOST_DEV cwoComplex& operator=(const float a){
		cplx[0]=a; cplx[1]=0.0f;
		return *this;
	};

	HOST_DEV cwoComplex& operator+=(const cwoComplex &a){
		cplx[0]+=a.cplx[0]; cplx[1]+=a.cplx[1];
		return *this;
	};
	HOST_DEV cwoComplex& operator+=(const float a){
		cplx[0]+=a;
		return *this;
	};
	HOST_DEV cwoComplex& operator-=(const cwoComplex &a){
		cplx[0]-=a.cplx[0]; cplx[1]-=a.cplx[1];
		return *this;
	};
	HOST_DEV cwoComplex& operator-=(const float a){
		cplx[0]-=a;
		return *this;
	};
	HOST_DEV cwoComplex& operator*=(const cwoComplex &a){
		cwoComplex tmp;
		
		tmp.cplx[0]=cplx[0]*a.cplx[0]-cplx[1]*a.cplx[1];
		tmp.cplx[1]=cplx[0]*a.cplx[1]+cplx[1]*a.cplx[0];
		cplx[0]=tmp.cplx[0];
		cplx[1]=tmp.cplx[1];
		return *this;
	}
	HOST_DEV cwoComplex& operator*=(const float a){
		cplx[0]*=a; cplx[1]*=a; 
		return *this;
	};
	
	HOST_DEV cwoComplex& operator/=(const cwoComplex &a){
		cwoComplex tmp;
		float deno=a.cplx[0]*a.cplx[0]+a.cplx[1]*a.cplx[1];
		tmp.cplx[0]=(cplx[0]*a.cplx[0]+cplx[1]*a.cplx[1])/deno;
		tmp.cplx[1]=(cplx[1]*a.cplx[0]-cplx[0]*a.cplx[1])/deno;
		cplx[0]=tmp.cplx[0];
		cplx[1]=tmp.cplx[1];
		return *this;
	};

	HOST_DEV cwoComplex& operator/=(const float a){
		cplx[0]/=a; cplx[1]/=a;
		return *this;
	};
	
	HOST_DEV const cwoComplex operator+(const cwoComplex &a)const{
		cwoComplex tmp;
		tmp.cplx[0]=cplx[0]+a.cplx[0]; 
		tmp.cplx[1]=cplx[1]+a.cplx[1]; 
		return tmp;
	}
	HOST_DEV const cwoComplex operator+(const float a)const{
		cwoComplex tmp;
		tmp.cplx[0]=cplx[0]+a;
		tmp.cplx[1]=cplx[1];
		return tmp;
	}
	HOST_DEV const cwoComplex operator-(const cwoComplex &a)const{
		cwoComplex tmp;
		tmp.cplx[0]=cplx[0]-a.cplx[0]; 
		tmp.cplx[1]=cplx[1]-a.cplx[1]; 
		return tmp;
	}
	HOST_DEV const cwoComplex operator-(const float a)const{
		cwoComplex tmp;
		tmp.cplx[0]=cplx[0]-a; 
		return tmp;
	}
	HOST_DEV const cwoComplex operator*(const cwoComplex &a)const{
		cwoComplex tmp;
		tmp.cplx[0]=cplx[0]*a.cplx[0]-cplx[1]*a.cplx[1];
		tmp.cplx[1]=cplx[0]*a.cplx[1]+cplx[1]*a.cplx[0];
		return tmp;
	}
	HOST_DEV const cwoComplex operator*(const float a)const{
		cwoComplex tmp;
		tmp.cplx[0]=cplx[0]*a; 
		tmp.cplx[1]=cplx[1]*a; 
		return tmp;
	}
	HOST_DEV const cwoComplex operator/(const cwoComplex &a)const{
		cwoComplex tmp;
		float deno=a.cplx[0]*a.cplx[0]+a.cplx[1]*a.cplx[1];
		tmp.cplx[0]=(cplx[0]*a.cplx[0]+cplx[1]*a.cplx[1])/deno;
		tmp.cplx[1]=(cplx[1]*a.cplx[0]-cplx[0]*a.cplx[1])/deno;
		return tmp;
	}
	HOST_DEV const cwoComplex operator/(const float a)const{
		cwoComplex tmp;
		tmp.cplx[0]=cplx[0]/a; 
		tmp.cplx[1]=cplx[1]/a; 
		return tmp;
	}
/*	cwoComplex operator+(cwoComplex &a){
		cwoComplex tmp;
		tmp.cplx[0]=cplx[0]+a.cplx[0]; 
		tmp.cplx[1]=cplx[1]+a.cplx[1]; 
		return tmp;
	}
	 cwoComplex operator+( float a){
		cwoComplex tmp;
		tmp.cplx[0]=cplx[0]+a;
		tmp.cplx[1]=cplx[1];
		return tmp;
	}
	 cwoComplex operator-( cwoComplex &a){
		cwoComplex tmp;
		tmp.cplx[0]=cplx[0]-a.cplx[0]; 
		tmp.cplx[1]=cplx[1]-a.cplx[1]; 
		return tmp;
	}
	 cwoComplex operator-( float a){
		cwoComplex tmp;
		tmp.cplx[0]=cplx[0]-a; 
		return tmp;
	}
	 cwoComplex operator*( cwoComplex &a){
		cwoComplex tmp;
		tmp.cplx[0]=cplx[0]*a.cplx[0]-cplx[1]*a.cplx[1];
		tmp.cplx[1]=cplx[0]*a.cplx[1]+cplx[1]*a.cplx[0];
		return tmp;
	}
	 cwoComplex operator*( float a){
		cwoComplex tmp;
		tmp.cplx[0]=cplx[0]*a; 
		tmp.cplx[1]=cplx[1]*a; 
		return tmp;
	}
	 cwoComplex operator/( cwoComplex &a){
		cwoComplex tmp;
		float deno=a.cplx[0]*a.cplx[0]+a.cplx[1]*a.cplx[1];
		tmp.cplx[0]=(cplx[0]*a.cplx[0]+cplx[1]*a.cplx[1])/deno;
		tmp.cplx[1]=(cplx[1]*a.cplx[0]-cplx[0]*a.cplx[1])/deno;
		return tmp;
	}
	 cwoComplex operator/( float a){
		cwoComplex tmp;
		tmp.cplx[0]=cplx[0]/a; 
		tmp.cplx[1]=cplx[1]/a; 
		return tmp;
	}
*/


	HOST_DEV cwoComplex Conj(){
		cwoComplex tmp; 
		tmp.cplx[0]=cplx[0]; 
		tmp.cplx[1]=-cplx[1];
		return tmp;
	}
	HOST_DEV float Intensity(){
		return cplx[0]*cplx[0]+cplx[1]*cplx[1];
	}

}cwoComplex;




typedef struct cwoObjPoint{
	cwoComplex	a;
	float	x;
	float	y;
	float	z;
}cwoObjPoint;

typedef struct cwoFloat2{
	cwoFloat2(){};
	cwoFloat2(float tx, float ty){x=tx;y=ty;};
	float x;
	float y;	
}cwoFloat2;

typedef struct cwoFloat3{
	cwoFloat3(){};
	cwoFloat3(float tx, float ty, float tz){x=tx;y=ty;z=tz;};
	float x;
	float y;
	float z;
}cwoFloat3;

typedef struct cwoFloat4{
	cwoFloat4(){};
	cwoFloat4(float tx, float ty, float tz, float ta){x=tx;y=ty;z=tz;a=ta;};
	float x;
	float y;
	float z;
	float a;
}cwoFloat4;

typedef struct cwoInt2{
	cwoInt2(){};
	cwoInt2(int tx, int ty){x=tx;y=ty;};
	int x;
	int y;	
}cwoInt2;

typedef struct cwoInt3{
	cwoInt3(){};
	cwoInt3(int tx, int ty, int tz){x=tx;y=ty;z=tz;};
	int x;
	int y;
	int z;
}cwoInt3;

typedef struct cwoInt4{
	cwoInt4(){};
	cwoInt4(int tx1, int tx2, int tx3, int tx4){
		x1=tx1;
		x2=tx2;
		x3=tx3;
		x4=tx4;
	};
	
	int Range(int x,int y){
		//Check range
		// return 0 : out range
		// return 1 : in range
		return (x >= x1 && x <= x3 && y >= x2 && y <= x4);
	}
	
	int x1;
	int x2;
	int x3;
	int x4;
}cwoInt4;


#ifndef __FFTW_H_
#define __FFTW_H_
	#include "fftw3.h"
#endif

typedef fftwf_plan cwofftPlan; //CPU뾭

//typedef int cwoStream; //CPU궻뤾뜃궼stream궼럊귦궶궋궻궳???궴궢궲int귩럊궎

	
//!  Structure for CWO context
/*!
This structure maintains calculation parameters.
*/
typedef struct cwoCtx{

//	int nthread_x; //!< number of threads 
//	int nthread_y; //!< number of threads along to y (for GPU only)

	int Nx; //!< x size 
	int Ny; //!< y size 
	int Nz; //!< z size 

	int field_type; //!< field type
	int calc_type; //!< diffraction type

	float z;//!< propagation distance for diffraction
//	float z2;//propagation distance for diffraction ( for double-step algorithm )

	float wave_length; //!< wavelength (in meter unit)

	//cwoComplex coeff;

	float src_px, src_py, src_pz; //!< sampling rate on source plane (in meter unit)
	float dst_px, dst_py, dst_pz; //!< sampling rate on destination plane (in meter unit)

	float src_ox, src_oy, src_oz; //!< offset from the origin on source plane (in meter unit)
	float dst_ox, dst_oy, dst_oz; //!< offset from the origin on destination plane (in meter unit)

	float planer_a;  
	float planer_argx;
	float planer_argy;

//	unsigned long seed; //!< random number seed

	int expand; //!< expand method

	
//	void **cstream; //!< for CUDA stream

	//**************
	//For PLS
	//**************
	int PLS_num;


	int GetNx(){return Nx;};
	int GetNy(){return Ny;};
	float GetPropDist(){return z;};
	float GetWaveLength(){return wave_length;};
	float GetSrcPx(){return src_px;};
	float GetSrcPy(){return src_py;};
	float GetDstPx(){return dst_px;};
	float GetDstPy(){return dst_py;};
	float GetSrcOx(){return src_ox;};
	float GetSrcOy(){return src_oy;};
	float GetDstOx(){return dst_ox;};
	float GetDstOy(){return dst_oy;};


}cwoCtx;


////////////////////////////////////////////////
//Threads
////////////////////////////////////////////////
CWO_EXPORT void CWOAPI cwoSetThreads(int N);
CWO_EXPORT int CWOAPI cwoGetThreads();


////////////////////////////////////////////////
//timer
////////////////////////////////////////////////

CWO_EXPORT void CWOAPI cwoSetTimer();
CWO_EXPORT float CWOAPI cwoEndTimer();
/*
////////////////////////////////////////////////
//Image File I/O
////////////////////////////////////////////////

CWO_EXPORT void* CWOAPI cwoImgCreate();
CWO_EXPORT CWOAPI cwoImgDestroy(void *handle);

CWO_EXPORT void CWOAPI cwoImgLoadAsFloat(char *fname,float *c, int *Nx, int *Ny);
CWO_EXPORT void CWOAPI cwoImgLoadAsFloatRGB(char *fname,float *r, float *g, float *b, int *Nx, int *Ny);
CWO_EXPORT void CWOAPI cwoImgSaveAsFloat(char *fname,float *c, int Nx, int Ny);
CWO_EXPORT void CWOAPI cwoImgSaveAsFloatRGB(char *fname,float *r, float *g, float *b, int Nx, int Ny);
CWO_EXPORT void CWOAPI cwoImgFree();
*/

//////////////////////////////
//////////////////////////////



#define CWO_C2C	0
#define CWO_R2C	1
#define CWO_C2R	2


CWO_EXPORT int CWOAPI cwoInitFFT(); //!< Initialize FFTW library
//CWO_EXPORT void CWOAPI cwoSetThreads(int N); //!< Set the number of threads


CWO_EXPORT void CWOAPI cwoCGHRecurrence(cwoCtx *c, cwoObjPoint *obj, float *hol, int N);


CWO_EXPORT void CWOAPI cwoFFT(cwoCtx *c, void *src, void *dst, int mode);
CWO_EXPORT void CWOAPI cwoIFFT(cwoCtx *c, void *src, void *dst);
CWO_EXPORT void CWOAPI cwoFFTShift(cwoCtx *c, void *a);

CWO_EXPORT void CWOAPI cwoNUFFT_T1(cwoCtx *c, cwoComplex *p_fld, cwoFloat2 *p_x, int R, int Msp);
CWO_EXPORT void CWOAPI cwoNUFFT_T2(cwoCtx *c, cwoComplex *p_fld, cwoFloat2 *p_x, int R, int Msp);

CWO_EXPORT void CWOAPI cwoWaveletHaar(cwoCtx *c, cwoComplex *src, cwoComplex *dst, int mode, int scale);


//CWO_EXPORT void CWOAPI cwoAngularProp(cwoCtx *c, cwoComplex *buf, float px, float py);

CWO_EXPORT void CWOAPI cwoAngularProp(cwoCtx *c, cwoComplex *buf); //Create, Center  
CWO_EXPORT void CWOAPI cwoAngularPropFS(cwoCtx *c, cwoComplex *buf); //Create, FFTShift
CWO_EXPORT void CWOAPI cwoAngularPropMul(cwoCtx *c, cwoComplex *buf); //Multiply, Center
CWO_EXPORT void CWOAPI cwoAngularPropMulFS(cwoCtx *c, cwoComplex *buf); //Multiply, FFTShift

CWO_EXPORT void CWOAPI cwoAngularLim(cwoCtx *c, float *fx_c, float *fx_w, float *fy_c, float *fy_w);
//CWO_EXPORT void CWOAPI cwoShiftedAngularProp(cwoCtx *c, cwoComplex *a);

CWO_EXPORT void CWOAPI cwoHuygensProp(cwoCtx *c, cwoComplex *buf);

CWO_EXPORT void CWOAPI cwoFresnelConvProp(cwoCtx *c, cwoComplex *buf);
CWO_EXPORT void CWOAPI cwoFresnelConvCoeff(cwoCtx *c, cwoComplex *buf, float const_val);

CWO_EXPORT void CWOAPI cwoFresnelAnalysisTransfer(cwoCtx *c, cwoComplex *a, cwoComplex *b);

CWO_EXPORT void CWOAPI cwoShiftedFresnelAperture(cwoCtx *c, cwoComplex *buf);
CWO_EXPORT void CWOAPI cwoShiftedFresnelProp(cwoCtx *c, cwoComplex *buf);
CWO_EXPORT void CWOAPI cwoShiftedFresnelCoeff(cwoCtx *c, cwoComplex *buf);

CWO_EXPORT void CWOAPI cwoARSSFresnelAperture(cwoCtx *c, cwoComplex *buf);
CWO_EXPORT void CWOAPI cwoARSSFresnelProp(cwoCtx *c, cwoComplex *buf);
CWO_EXPORT void CWOAPI cwoARSSFresnelCoeff(cwoCtx *c, cwoComplex *buf);

CWO_EXPORT void CWOAPI cwoFresnelFourierProp(cwoCtx *c, cwoComplex *a);
CWO_EXPORT void CWOAPI cwoFresnelFourierCoeff(cwoCtx *c, cwoComplex *a);

CWO_EXPORT void CWOAPI cwoFresnelDblAperture(cwoCtx *c, cwoComplex *a, float z1);
CWO_EXPORT void CWOAPI cwoFresnelDblFourierDomain(cwoCtx *c, cwoComplex *a, float z1, float z2, cwoInt4 *zp);
CWO_EXPORT void CWOAPI cwoFresnelDblCoeff(cwoCtx *c, cwoComplex *a, float z1, float z2);


CWO_EXPORT void CWOAPI cwoPhase(cwoCtx *c, cwoComplex *a, cwoComplex *b, float offset);
CWO_EXPORT void CWOAPI cwoArg(cwoCtx *c, cwoComplex *a, cwoComplex *b, float scale, float offset);
CWO_EXPORT void CWOAPI cwoIntensity(cwoCtx *c, cwoComplex *a, cwoComplex *b);
CWO_EXPORT void CWOAPI cwoAmp(cwoCtx *c, cwoComplex *a, cwoComplex *b);

CWO_EXPORT void CWOAPI cwoExpi(cwoCtx *c, cwoComplex *a, cwoComplex *b);

CWO_EXPORT void CWOAPI cwoExpand(cwoCtx *ctx, 
								 void *src, int sx, int sy, int sNx, int sNy,
								 void *dst, int dx, int dy, int dNx, int dNy, int mode);

CWO_EXPORT void CWOAPI cwoScaleReal(cwoCtx *ctx, 
	cwoComplex *pi, cwoComplex *po, 
	float i1,float i2, float o1, float o2);

CWO_EXPORT void CWOAPI cwoScaleCplx(cwoCtx *ctx, 
	cwoComplex *pi, cwoComplex *po, 
	float i1,float i2, float o1, float o2);


CWO_EXPORT void CWOAPI cwoMultComplex(cwoCtx *c, void *A, void *B, void *C);
CWO_EXPORT void CWOAPI cwoDiv(cwoCtx *c, void* a, float b); // c=a/b
//CWO_EXPORT void CWOAPI cwoGamma(cwoCtx *c, float* buf, float gamma);

CWO_EXPORT void CWOAPI cwoSqrtReal(cwoCtx *ctx, 
	cwoComplex *pi, cwoComplex *po);

CWO_EXPORT void CWOAPI cwoSqrtCplx(cwoCtx *ctx, 
	cwoComplex *pi, cwoComplex *po);

CWO_EXPORT void CWOAPI cwoMaxMin(
	cwoCtx *c, cwoComplex* a, float *max, float *min, 
	int *max_x, int *max_y,int *min_x, int *min_y);


CWO_EXPORT float CWOAPI cwoMSE(cwoCtx *ctx, cwoComplex *tar, cwoComplex *ref);


CWO_EXPORT void CWOAPI cwoAddSphericalWave(
	cwoCtx *c,cwoComplex *p, float x, float y, float z, 
	float px, float py, float a);

CWO_EXPORT void CWOAPI cwoMulSphericalWave(
	cwoCtx *c,cwoComplex *p, float x, float y, float z, 
	float px, float py, float a);



CWO_EXPORT void CWOAPI cwoAddApproxSphWave(
	cwoCtx *c,cwoComplex *p, float x, float y, float phi, float zx, float zy,
	float px, float py, float a);
CWO_EXPORT void CWOAPI cwoMulApproxSphWave(
	cwoCtx *c,cwoComplex *p, float x, float y, float phi, float zx, float zy, 
	float px, float py, float a);


CWO_EXPORT void CWOAPI cwoSetRandSeed(unsigned long s);
CWO_EXPORT float CWOAPI cwoRandVal();
CWO_EXPORT void CWOAPI cwoSetRandReal(cwoCtx *c, cwoComplex *a, float max, float min);
CWO_EXPORT void CWOAPI cwoSetRandPhase(cwoCtx *c, cwoComplex *a, float max, float min);
CWO_EXPORT void CWOAPI cwoMulRandPhase(cwoCtx *c, cwoComplex *a, float max, float min);

CWO_EXPORT void CWOAPI cwoParticleField(cwoCtx *c, cwoComplex *p, cwoFloat3 pos, float radius, float amp, float init_ph);

CWO_EXPORT float CWOAPI cwoSumTotal(cwoCtx *c, cwoComplex *a);
CWO_EXPORT float CWOAPI cwoAverage(cwoCtx *c, cwoComplex *a);
CWO_EXPORT float CWOAPI cwoVariance(cwoCtx *c, cwoComplex *a, float ave);
CWO_EXPORT void CWOAPI cwoVarianceMap(cwoCtx *c, cwoComplex *src, cwoComplex *dst, int Sx, int Sy);
CWO_EXPORT float CWOAPI cwoSSIM(cwoComplex *data1, cwoComplex *data2, int Nx, int Ny);
CWO_EXPORT float CWOAPI cwoSMDx(cwoCtx *c, float *a);

//CWO_EXPORT void CWOAPI cwoSearchMaxMin(cwoCtx *c, void* buf, float *max, float *min, int Nx, int Ny);


CWO_EXPORT void CWOAPI cwoHanning(cwoCtx *c, cwoComplex *a, int m, int n, int Wx, int Wy);
CWO_EXPORT void CWOAPI cwoHamming(cwoCtx *c, cwoComplex *a, int m, int n, int Wx, int Wy);

cwoComplex CWOAPI cwoInterNearest(
	cwoComplex *p,
	float x, float y, int Nx, int Ny);

CWO_EXPORT void CWOAPI cwoResizeNearest(
	cwoCtx *ctx,
	cwoComplex *p_new, int dNx, int dNy,
	cwoComplex *p_old, int sNx, int sNy
);

cwoComplex CWOAPI cwoInterLinear(
	cwoComplex *p,
	float x, float y, int Nx, int Ny);

CWO_EXPORT void CWOAPI cwoResizeLinear(
	cwoCtx *ctx,
	cwoComplex *p_new, int dNx, int dNy,
	cwoComplex *p_old, int sNx, int sNy
);

cwoComplex CWOAPI cwoInterCubic(
	cwoComplex *p,
	float x, float y, int Nx, int Ny);

CWO_EXPORT void CWOAPI cwoResizeCubic(
	cwoCtx *ctx,
	cwoComplex *p_new, int dNx, int dNy,
	cwoComplex *p_old, int sNx, int sNy
);

cwoComplex CWOAPI cwoInterLanczos(
	cwoComplex *p,
	float x, float y, int Nx, int Ny);

CWO_EXPORT void CWOAPI cwoResizeLanczos(
	cwoCtx *ctx,
	cwoComplex *p_new, int dNx, int dNy,
	cwoComplex *p_old, int sNx, int sNy
);

CWO_EXPORT void CWOAPI cwoAffineAngularSpectrum(
	cwoCtx *ctx,
	cwoComplex *p_new, cwoComplex *p_old, int Nx, int Ny,
	float px, float py, float wl, float *mat_affine, int flag);



//CWO_EXPORT int CWOAPI cwoImgLoad(cwoCtx *ctx, char *fname, int c, cwoComplex *p);
CWO_EXPORT int CWOAPI cwoPrepareCimg(char *fname, int *Nx, int *Ny);
CWO_EXPORT int CWOAPI cwoLoadCimg(cwoCtx *ctx, char *fname, int c, cwoComplex *a);
CWO_EXPORT int CWOAPI cwoSaveCimgMono(cwoCtx *ctx, char *fname, cwoComplex *a);
CWO_EXPORT int CWOAPI cwoSaveCimgColor(cwoCtx *ctx, char *fname, cwoComplex *pr,cwoComplex *pg, cwoComplex *pb);
CWO_EXPORT int CWOAPI cwoSaveCimgMonosToColor(cwoCtx *ctx, char *fname, char *r_name, char *g_name, char *b_name);


CWO_EXPORT int CWOAPI cwoLoadBmp( 
	const char *file_name, 
	unsigned char ** bmp_data, int * bmp_width, int * bmp_height, int * bmp_bits, 
	int fixed_width);

CWO_EXPORT int CWOAPI cwoSaveBmp( 
	const char * file_name, 
	const unsigned char * bmp_data, int width, int height, int bmp_bits, 
	int fixed_width);


//Phase unwrapping
CWO_EXPORT float CWOAPI cwoWrap(float p1, float p2);
CWO_EXPORT void CWOAPI cwoLaplacianPhase(cwoCtx *ctx, cwoComplex *pi, cwoComplex *po, float *dxwts, float *dywts, int laptype);
CWO_EXPORT void CWOAPI cwoPoisonSolverFFT(cwoCtx *ctx, cwoComplex *p);



#ifdef __cplusplus
}
#endif


#endif



/////////////////////////////

