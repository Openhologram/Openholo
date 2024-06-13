#pragma once


#ifndef __ophSigCH_h
#define __ophSigCH_h

#include "ophSig.h"

/**
* @addtogroup CH
//@{
* @details
* @section Introduction

This module is related methods which perform compressed holography algorithm.
Compressed holography algorithm is based on a paper D.Brady et al., Opt. Express 18, pp. 13040 (2009).
The methods in this module (except twist) has been created based on the original code included in the supplement material of S.Lim, D. Marks, and D. Brady, "Sampling and processing for compressive holography," Applied Optics, vol. 50, no. 34, pp. H75-H86, 2011.

This module also uses a TwIST algorithm (J.M. Bioucas-Dias et al., IEEE Trans. Image Proc. 16, pp.2992 (2007)).
The 'twist' method in this module is based on the optimization algorithm (TwIST) code created and distributed by Jose Bioucas-Dias and Mario Figueiredo, October, 2007
J. Bioucas-Dias and M. Figueiredo, "A New TwIST: Two-Step Iterative Shrinkage/Thresholding Algorithms for Image Restoration",  IEEE Transactions on Image processing, 2007.
www.lx.it.pt/~bioucas/TwIST


![Compressed holography finds 3D object from complex field data by maximizing sparisity in 3D object space.](@ref pics/ophsig/ch/ch_concept.png)\n
\n

![Comparison with usual numerical reconstruction.](@ref pics/ophsig/ch/ch_result.png)

*/
//! @} CH

/**
* @ingroup CH
* @brief
* @author
*/
class SIG_DLL ophSigCH : public ophSig
{
public:
	/**
	* @brief Constructor
	*/
	explicit ophSigCH(void);
	
	bool setCHparam(vector<Real> &z, int maxIter, double tau, double tolA, int tvIter);
	bool runCH(int complexHidx);
	bool saveNumRec(const char *fname);
	bool readConfig(const char * fname);
	bool loadCHtemp(const char *real, const char *imag, uint8_t bitpixel);

	matrix<Complex<Real>> propagationHoloAS(matrix<Complex<Real>> complexH, float depth);


protected:

	virtual ~ophSigCH(void) = default;
	void tvdenoise(matrix<Real> &input, double lam, int iters, matrix<Real> &output);
	double tvnorm(matrix<Real> &input);
	void c2ri(matrix<Complex<Real>> &complexinput, matrix<Real> &realimagoutput);
	void ri2c(matrix<Real> &realimaginput, matrix<Complex<Real>> &complexoutput);
	void volume2plane(matrix<Real>& realimagvolumeinput, vector<Real> z, matrix<Real>& realimagplaneoutput);
	void plane2volume(matrix<Real>& realimagplaneinput, vector<Real> z, matrix<Real>& realimagplaneoutput);
	void convert3Dto2D(matrix<Complex<Real>> *complex3Dinput, int nz, matrix<Complex<Real>> &complex2Doutput);
	void convert2Dto3D(matrix<Complex<Real>> &complex2Dinput, int nz, matrix<Complex<Real>> *complex3Doutput);
	void twist(matrix<Real>& realimagplaneinput, matrix<Real>& realimagvolumeoutput);
	double matrixEleSquareSum(matrix<Real> &input);


	
public:


	
protected:
	int Nz;
	int MaxIter;
	double Tau;
	double TolA;
	int TvIter;
	matrix<Real> NumRecRealImag;
	vector<Real> Z;

};

#endif // !__ophSigCH_h