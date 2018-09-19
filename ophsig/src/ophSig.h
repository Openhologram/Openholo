/**
* @mainpage ophSig
* @brief Abstract class for core processing classes
*/

#ifndef __ophSig_h
#define __ophSig_h

#include "tinyxml2.h"
#include "Openholo.h"
#include "sys.h"



#ifdef SIG_EXPORT
#define SIG_DLL __declspec(dllexport)
#else
#define SIG_DLL __declspec(dllimport)
#endif

struct SIG_DLL ophSigConfig {
	int rows;
	int cols;
	float width;
	float height;
	double lambda;
	float NA;
	float z;
};

/**
* @file
* @author
* @brief
* @pram
*/

class SIG_DLL ophSig : public Openholo
{
public:
	/**
	* @brief Constructor
	*/
	explicit ophSig(void);
	/**
	* @brief          Load bmp or bin file
	* @param real     real data file name
	* @param imag     imag data file name
	* @param bitpixel bit per pixel
	* @return         if works well return 0  or error occurs return -1
	*/
	bool load(const char *real, const char *imag, uint8_t bitpixel);
	/**
	* @brief          Save data as bmp or bin file
	* @param real     real data file name
	* @param imag     imag data file name
	* @param bitpixel bit per pixel
	* @return         if works well return 0  or error occurs return -1
	*/
	bool save(const char *real, const char *imag, uint8_t bitpixel);
	bool save(const char *real, uint8_t bitpixel);

protected:

	virtual ~ophSig(void) = default;
	/**
	* @brief          Generate linearly spaced vector
	* @param first    first number of vector
	* @param last     last number of vector
	* @param len      vector with specified number of values
	* @return         result vector
	*/
	vector<Real> linspace(double first, double last, int len);
	template<typename T>
	/**
	* @brief        Function for Linear interpolation 1D
	* @param X      input signal coordinate
	* @param in     input signal
	* @param Xq     output signal coordinate
	* @param out    output signal
	*/
	void linInterp(vector<T> &X, matrix<Complex<T>> &in, vector<T> &Xq, matrix<Complex<T>> &out);
	/**
	* @brief         Function for extracts Complex magnitude value
	* @param src     input signal
	* @param dst     output signal
	*/
	template<typename T>
	inline void absMat(matrix<Complex<T>>& src, matrix<T>& dst);
	template<typename T>
	/**
	* @brief         Function for extracts real absolute value
	* @param src     input signal
	* @param dst     output signal
	*/
	inline void absMat(matrix<T>& src, matrix<T>& dst);
	template<typename T>
	/**
	* @brief         Function for extracts Complex phase value
	* @param src     input signal
	* @param dst     output signal
	*/
	inline void angleMat(matrix<Complex<T>>& src, matrix<T>& dst);
	template<typename T>
	/**
	* @brief         Function for extracts Complex conjugate value
	* @param src     input signal
	* @param dst     output signal
	*/
	inline void conjMat(matrix<Complex<T>>& src, matrix<Complex<T>>& dst);
	template<typename T>
	/**
	* @brief         Function for returns exponent ex
	* @param src     input signal
	* @param dst     output signal
	*/
	inline void expMat(matrix<Complex<T>>& src, matrix<Complex<T>>& dst);
	template<typename T>
	/**
	* @brief         Function for returns exponent e(x), where x is complex number
	* @param src     input signal
	* @param dst     output signal
	*/
	inline void expMat(matrix<T>& src, matrix<T>& dst);
	template<typename T>
	/**
	* @brief         Function for returns exponent e(x), where x is real number
	* @param src     input signal
	* @param dst     output signal
	*/
	inline void meanOfMat(matrix<T> &input, double &output);
	template<typename T>
	/**
	* @brief         Function for extracts max of matrix
	* @param src     input signal
	*/
	inline  Real maxOfMat(matrix<T>& src);
	template<typename T>
	/**
	* @brief         Function for extracts min of matrix
	* @param src     input signal
	*/
	inline Real minOfMat(matrix<T>& src);

	template<typename T>
	/**
	* @brief		Function for returns 2d matrix based on vector src1, src2
	* @param src1	input vector
	* @param src2	input vector
	* @param dst1	input signal
	* @param dst2	input signal
	*/
	void meshgrid(vector<T>& src1, vector<T>& src2, matrix<T>& dst1, matrix<T>& dst2);

	/**
	* @brief         Function for Fast Fourier transform 1D
	* @param src     input signal
	* @param dst     output signal
	* @param sign    sign = OPH_FORWARD is fft and sign= OPH_BACKWARD is inverse fft
	* @param flag    flag = OPH_ESTIMATE is fine best way to compute the transform but it is need some time, flag = OPH_ESTIMATE is probably sub-optimal
	*/
	template<typename T>
	void fft1(matrix<Complex<T>> &src, matrix<Complex<T>> &dst, int sign = OPH_FORWARD, uint flag = OPH_ESTIMATE);
	/**
	* @brief         Function for Fast Fourier transform 2D
	* @param src     input signal
	* @param dst     output signal
	* @param sign    sign = OPH_FORWARD is fft and sign= OPH_BACKWARD is inverse fft
	* @param flag    flag = OPH_ESTIMATE is fine best way to compute the transform but it is need some time, flag = OPH_ESTIMATE is probably sub-optimal
	*/
	template<typename T>
	void fft2(matrix<Complex<T>> &src, matrix<Complex<T>> &dst, int sign = OPH_FORWARD, uint flag = OPH_ESTIMATE);
	/**
	* @brief         Function for Shift zero-frequency component to center of spectrum
	* @param src     input signal
	* @param dst     output signal
	*/
	template<typename T>
	void fftShift(matrix<Complex<T>> &src, matrix<Complex<T>> &dst);
public:
	/**
	* @brief          Function for Read parameter
	* @param fname    file name
	* @return         if works well return 0  or error occurs return -1
	*/
	virtual bool readConfig(const char* fname);
	/**
	* @brief          Function for Convert complex hologram to off-axis hologram
	* @return         if works well return 0  or error occurs return -1
	*/
	bool sigConvertOffaxis();
	/**
	* @brief          Function for Convert complex hologram to horizontal parallax only hologram
	* @return         if works well return 0  or error occurs return -1
	*/
	bool sigConvertHPO();
	/**
	* @brief          Function for Chromatic aberration compensation filter
	* @return         if works well return 0  or error occurs return -1
	*/
	bool sigConvertCAC(double red, double green, double blue);
	/**
	* @brief          Function for Chromatic aberration compensation filter
	* @return         if works well return 0  or error occurs return -1
	*/
	bool propagationHolo(float depth);
	/**
	* @brief          Function for propagation hologram
	* @param depth    position from hologram plane to propagation hologram plane
	* @return         output signal
	*/
	matrix<Complex<Real>> propagationHolo(matrix<Complex<Real>> complexH, float depth);
	/**
	* @brief          Extraction of distance parameter using axis transfomation
	* @return         result distance
	*/
	double sigGetParamAT();
	/**
	* @brief          Extraction of distance parameter using sharpness functions
	* @param zMax     maximum value of distance on z axis
	* @param zMin     minimum value of distance on z axis
	* @param sampN    count of search step
	* @param th       threshold value
	* @return         result distance
	*/
	double sigGetParamSF(float zMax, float zMin, int sampN, float th);

	bool getComplexHFromPSDH(const char* fname0, const char* fname90, const char* fname180, const char* fname270);
	
protected:

	virtual void ophFree(void);

	ophSigConfig _cfgSig;
	matrix<Complex<Real>> ComplexH[3];
	float _angleX;
	float _angleY;
	float _redRate;
	float _radius;
	float _foc[3];



};

#endif // !__ophSig_h