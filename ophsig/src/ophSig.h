/**
* @mainpage ophSig
* @brief Abstract class for core processing classes
*/

#ifndef __ophSig_h
#define __ophSig_h


#include "Openholo.h"

/**
* @brief openCV library link
*/
#pragma comment(lib,"opencv_world340.lib")
#include "opencv\cv.hpp"

/**
* @brief cwo++ library link
*/
#pragma comment(lib,"cwo.lib")
#include <cwo.h>

#ifdef SIG_EXPORT
#define SIG_DLL __declspec(dllexport)
#else
#define SIG_DLL __declspec(dllimport)
#endif

struct SIG_DLL ophSigConfig {
	int rows;
	int cols;
	double width;
	double height;
	double lambda[3];
	double NA;
	double z;
};

class SIG_DLL ophSig : public Openholo
{
public:
	/**
	* @brief Constructor
	*/
	ophSig(void);

protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophSig(void) = default;

protected:
	cv::Mat linspace(float first, float last, int len);
	void add(cv::Mat &A, cv::Mat &B, cv::Mat &out);
	void exp(cv::Mat &in, cv::Mat &out);
	void mul(cv::Mat &A, cv::Mat &B, cv::Mat &out);
	void min(cv::Mat &in, float &out);
	void max(cv::Mat &in, float &out);
	void fftshift(cv::Mat &in, cv::Mat &out);
	void fftshift2d(cv::Mat &in, cv::Mat &out);
	void fft1d(cv::Mat &in, cv::Mat &out);
	void fft2d(cv::Mat &in, cv::Mat &out);
	void ifft2d(cv::Mat &in, cv::Mat &out);
	void mean(cv::Mat &in, float &out);
	void conj(cv::Mat &in, cv::Mat &out);
	void meshgrid(const cv::Mat&x, const cv::Mat &y, cv::Mat &a, cv::Mat &b);
	void abs(cv::Mat &in, cv::Mat &out);
	void div(cv::Mat &A, cv::Mat &B, cv::Mat &out);
	void linInterp(cv::Mat &X, cv::Mat &in, cv::Mat &Xq, cv::Mat &out);

public:
	/**
	* @brief Import complex hologram data
	*/
	virtual bool loadHolo(std::string cosh, std::string sinh, std::string type, float flag);
	
	virtual bool saveHolo(std::string cosh, std::string sinh, std::string type, float flag);
	
	virtual bool loadParam(std::string cfg);

	virtual cv::Mat propagationHolo(float depth);

	virtual cv::Mat propagationHolo(cv::Mat complexH, float depth);
	
protected:
	/**
	* @brief Pure virtual function for override in child classes
	*/
	virtual void ophFree(void);

	ophSigConfig _cfgSig;
	cv::Mat complexH;
};

#endif // !__ophSig_h
