#ifndef __ophAS_h
#define __ophAS_h


#include "ophGen.h"
#include "complex.h"
#include "sys.h"
#include "AngularC_types.h"



using namespace oph;

const complex<double> I = sqrt(complex<double>(-1, 0));




class GEN_DLL ophAS : public ophGen 
{
private:
	coder::array<creal_T, 2U> im;
	unsigned char* res;
	double w;
	double h; 
	double wavelength;
	double knumber;
	double xi_interval;
	double eta_interval;
	double depth;
	double x;
	double y;
	double z;
	double amplitude;
	OphPointCloudConfig pc_config;
	OphPointCloudData pc_data;
	int n_points;
	bool is_CPU;
public:
	ophAS();
	virtual ~ophAS();
	bool readConfig(const char* fname);
	int loadPointCloud(const char* fname);
	void setmode(bool is_cpu);
	void ASCalculation(double w, double h, double wavelength, double knumber, double
		xi_interval, double eta_interval, double depth, coder::
		array<creal_T, 2U> &fringe, coder::array<creal_T, 2U>
		&b_AngularC);
	void RayleighSommerfield(double w, double h, double wavelength, double knumber, double
		xi_interval, double eta_interval, double depth, coder::
		array<creal_T, 2U> &fringe);
	void AngularSpectrum(double w, double h, double wavelength, double knumber, double
		xi_interval, double eta_interval, double depth, const coder::
		array<creal_T, 2U> &fringe, coder::array<creal_T, 2U>
		&b_AngularC);
	void generateHologram();
	void MemoryRelease();
	void ifft2(const coder::array<creal_T, 2U> &x, coder::array<creal_T, 2U> &y);
	void fft2_matlab(coder::array<creal_T, 2U> &x, coder::array<creal_T, 2U>
		&y);
	void eml_fftshift(coder::array<creal_T, 2U> &x, int dim);
	void ifftshift(coder::array<creal_T, 2U> &x);
	int save(const char * fname, uint8_t bitsperpixel, uchar* src, uint px, uint py);
	void save(const char* fname);
};

#endif