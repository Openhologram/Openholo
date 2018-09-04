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

class SIG_DLL ophSig : public Openholo
{
public:
	/**
	* @brief Constructor
	*/
	explicit ophSig(void);
	bool load(const char *real, const char *imag, uint8_t bitpixel);
	bool save(const char *real, const char *imag, uint8_t bitpixel);
	
protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophSig(void) = default;

protected:
	vector<Real> linspace(double first, double last, int len);
	template<typename T>
	void linInterp(vector<T> &X, matrix<Complex<T>> &in, vector<T> &Xq, matrix<Complex<T>> &out);

	//virtual ~ophSig(void) = default;

	template<typename T>
	inline void absMat(matrix<Complex<T>>& src, matrix<T>& dst);
	template<typename T>
	inline void absMat(matrix<T>& src, matrix<T>& dst);
	template<typename T>
	inline void angleMat(matrix<Complex<T>>& src, matrix<T>& dst);
	template<typename T>
	inline void conjMat(matrix<Complex<T>>& src, matrix<Complex<T>>& dst);
	template<typename T>
	inline void expMat(matrix<Complex<T>>& src, matrix<Complex<T>>& dst);
	template<typename T>
	inline void expMat(matrix<T>& src, matrix<T>& dst);
	template<typename T>
	inline void meanOfMat(matrix<T> &input, double &output);
	template<typename T>
	inline  Real maxOfMat(matrix<T>& src);
	template<typename T>
	inline Real minOfMat(matrix<T>& src);

	template<typename T>
	void meshgrid(vector<T>& src1, vector<T>& src2, matrix<T>& dst1, matrix<T>& dst2);

	template<typename T>
	void fft1(matrix<Complex<T>> &src, matrix<Complex<T>> &dst, int sign = OPH_FORWARD, uint flag = OPH_ESTIMATE);
	template<typename T>
	void fft2(matrix<Complex<T>> &src, matrix<Complex<T>> &dst, int sign = OPH_FORWARD, uint flag = OPH_ESTIMATE);
	template<typename T>
	void fftShift(matrix<Complex<T>> &src, matrix<Complex<T>> &dst);




public:
	virtual bool readConfig(const char* fname);
	//virtual bool loadParam(string cfg);
	bool sigConvertOffaxis();
	bool sigConvertHPO();
	bool sigConvertCAC(double red , double green, double blue);
	bool propagationHolo(float depth);
	Mat propagationHolo(Mat complexH, float depth);
	double sigGetParamAT();
	double sigGetParamSF(float zMax, float zMin, int sampN, float th);
	
protected:
	/**
	* @brief Pure virtual function for override in child classes
	*/
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
