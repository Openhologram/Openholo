#pragma once


#ifndef __ophSigCH_h
#define __ophSigCH_h

#include "ophSig.h"

/**
* @addtogroup CH
//@{
* @detail

*/
//! @} CH

/**
* @ingroup CH
* @brief Openholo Compressed Holography
* @author Jae-Hyeung Park
*/
class SIG_DLL ophSigCH : public ophSig
{
public:
	/**
	* @brief Constructor
	*/
	explicit ophSigCH(void);
	
	/**
	* @ingroup CH
	* @brief	Setting Compressed Holography Parameters
	* @detail
	* @param	z			Depth plane distances
	* @param	maxIter		Maximum number of iteration
	* @param	tau			
	* @param	tolA
	* @return	tvIter
	*/
	bool setCHparam(vector<Real> &z, int maxIter, double tau, double tolA, int tvIter);

	/**
	* @ingroup CH
	* @brief	Main funtcion to reconstruct 3D scene with compressed holography
	* @detail
	* @param	complexHidx
	* @return
	*/
	bool runCH(int complexHidx);

	/**
	* @ingroup CH
	* @brief	Function to save the image file of reconstruction by compressed holography
	* @detail
	* @param	fname	file name
	* @return
	*/
	bool saveNumRec(const char *fname);

	/**
	* @ingroup CH
	* @brief	CH configuration file load
	* @detail
	* @param	fname	Configuration file name(.xml)
	* @return
	*/
	bool readConfig(const char * fname);

	/**
	* @ingroup CH
	* @brief	Load complex field from the real and imaginary image files
	* @detail	
	* @param	real		complex field real part image file name	
	* @param	imag		complex field imaginary part image file name
	* @param	bitpixel	bit per pixel
	* @return
	*/
	bool loadCHtemp(const char *real, const char *imag, uint8_t bitpixel);

	bool loadCH(const char *fname);

	/**
	* @ingroup CH
	* @brief	Numerical propagation
	* @detail
	* @param
	* @param
	* @param
	* @param
	* @return
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