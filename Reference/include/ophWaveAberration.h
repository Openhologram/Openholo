/**
* @file ophWaveAberration.h
* @brief Wave Aberration module
* @author Minsik Park
* @date 2018/07/30
*/



#pragma once
#ifndef __OphWaveAberration_h
#define __OphWaveAberration_h

#include "ophDis.h"
#include <iostream>
#include <fstream>
#include <complex>
#include <cmath>
#include <memory>
#include <algorithm>
#include <vector>
#include "tinyxml2.h"
#include "sys.h"

using namespace std;


#ifdef DISP_EXPORT
#define DISP_DLL __declspec(dllexport)
#else
#define DISP_DLL __declspec(dllimport)
#endif

class DISP_DLL ophWaveAberration : public ophDis
{
private :
	/**
	* @param wave length of illumination light
	*/
	Real waveLength;
	
	/**
	* @param sampling interval in x axis of the exit pupil
	*/
	Real pixelPitchX;
	/**
	* @param sampling interval in y axis of the exit pupil
	*/
	Real pixelPitchY;
		/**
	* @param order of the radial term of Zernike polynomial
	*/
	int nOrder; 
	/**
	* @param frequency of the sinusoidal component of Zernike polynomial 
	*/
	int mFrequency; 
	/**
	* @param Zernike coeffient
	*/
	Real zernikeCoefficent[45];

public:

	/**
	* @param resolution in x axis of the exit pupil
	*/
	oph::uint resolutionX;
	/**
	* @param resolution in y axis of the exit pupil
	*/
	oph::uint resolutionY;
	/**
	* @brief double pointer of the 2D data array of a wave aberration
	*/
	oph::Complex<Real> ** complex_W;


	/**
	* @brief Constructor
	*/
	ophWaveAberration();
	/**
	* @brief Destructor
	*/
	~ophWaveAberration();

	/**
	* @brief read configuration from a configration file
	* @param fname: a path name of a configration file
	*/
	bool readConfig(const char* fname);

	
	/**
	* @brief Factorial using recursion
	* @param x: a number of factorial 
	*/
	Real factorial(double x);
	/**
	* @brief Resizes 2D data using bilinear interpolation
	* @param X: 2D source image
	* @param Nx: resolution in x axis of source image
	* @param Ny: resolution in y axis of source image
	* @param nx: resolution in x axis target image
	* @param ny: resolution in y axis target image
	* @param ny: 2D target image
	*/
	void imresize(double **X, int Nx, int Ny, int nx, int ny, double **Y); 
	/**
	* @brief Calculates Zernike polynomial
	* @param n: order of the radial term of Zernike polynomial
	* @param m: frequency of the sinusoidal component of Zernike polynomial 
	* @param x: resolution in y axis of the exit pupil
	* @param y: resolution in y axis of the exit pupil
	* @param d: diameter of aperture of the exit pupil
	*/
	double ** calculateZernikePolynomial(double n, double m, vector<double> x, vector<double> y, double d);
	/**
	* @brief Sums up the calculated Zernike polynomials
	*/
	void accumulateZernikePolynomial();
	/**
	* @brief deletes 2D memory array using double pointer 
	*/
	void Free2D(oph::Complex<Real> ** doublePtr);
	/**
	* @brief saves the 2D data array of a wave aberration into a file
	* @param fname: a path name of a file to save a wave aberration 
	*/
	void saveAberration(const char* fname);
	/**
	* @brief reads the 2D data array of a wave aberration from a file
	* @param fname: a path name of a file to save a wave aberration
	*/
	void readAberration(const char* fname);

	void ophFree(void);
};

#endif



