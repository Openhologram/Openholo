#pragma once


#ifndef __ophSigCH_h
#define __ophSigCH_h

#include "ophSig.h"



/**
* @ingroup CH
* @brief
* @detail
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