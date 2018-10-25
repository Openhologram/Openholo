#pragma once


#ifndef __ophSigCH_h
#define __ophSigCH_h

#include "ophSig.h"

/**
* @addtogroup CH
//@{
* @detail
* @section Introduction

This module is related methods which perform compressed holography algorithm.
Compressed holography algorithm is based on a paper D.Brady et al., Opt. Express 18, pp. 13040 (2009). 
The methods in this module (except twist) has been created based on the original code included in the supplement material of S.Lim, D. Marks, and D. Brady, ��Sampling and processing for compressive holography,�� Applied Optics, vol. 50, no. 34, pp. H75-H86, 2011.

This module also uses a TwIST algorithm (J.M. Bioucas-Dias et al., IEEE Trans. Image Proc. 16, pp.2992 (2007)).
The 'twist' method in this module is based on the optimization algorithm (TwIST) code created and distributed by Jose Bioucas-Dias and Mario Figueiredo, October, 2007
J. Bioucas-Dias and M. Figueiredo, "A New TwIST: Two-Step Iterative Shrinkage/Thresholding Algorithms for Image Restoration",  IEEE Transactions on Image processing, 2007.
www.lx.it.pt/~bioucas/TwIST

Compressed holography finds 3D object from complex field data by maximizing sparisity in 3D object space.
![](pics/ophsig/CH/CH_concept.png)

Comparison with usual numerical reconstruction.
![](pics/ophsig/CH/CH_result.png)

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
	
	/**
	* @brief Set Compressed Holography parameters
	* @param z : vector containing depth values for numerical reconstruction
	* @param maxIter : maximum number of iteration for TwIST optimization
	* @param tau : regularization parameter ('lambda in the picture above)
	* @param tolA : stopping threshold
	* @param tvIter : the number of iterations in denoising based on total variation
	*/
	bool setCHparam(vector<Real> &z, int maxIter, double tau, double tolA, int tvIter);

	/**
	* @brief Run compressed holography algorithm
	* @param complexHidx : index (0, 1, 2) to monochromatic complex field data in the member variable complexH[3] 
	*/
	bool runCH(int complexHidx);

	/**
	* @brief Save numerical reconstruction results for z vector after compressed holography
	* @param fname : image file name where the numerical reconstruction results are stored. Image index will be appened to fname for each depth in the vector z.
	*/
	bool saveNumRec(const char *fname);

	/**
	* @brief Read configure file (xml)
	* @param fname : configure file name
	*/
	bool readConfig(const char * fname);

	/**
	* @brief Load complex field from image files
	* @param real : image file name of real data,  imag: image file name of imaginary data,  bitpixel : pixel depth of image file
	*/
	bool loadCHtemp(const char *real, const char *imag, uint8_t bitpixel);

	/**
	* @brief Numerical propagation of complex field using angular spectrum method
	* @param complexH : complex field
	* @param depth: distance to be propagated (positive or negative)
	*/
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