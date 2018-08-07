#ifndef __ophWRP_h
#define __ophWRP_h

#define _USE_MATH_DEFINES

#include "ophGen.h"

#ifdef RECON_EXPORT
#define RECON_DLL __declspec(dllexport)
#else
#define RECON_DLL __declspec(dllimport)
#endif

//Build Option : Multi Core Processing (OpenMP)
#ifdef _OPENMP
#include <omp.h>
#endif


#define THREAD_X 32
#define THREAD_Y 16

/* Bitmap File Definition*/
#define OPH_Bitsperpixel 8 //24 // 3byte=24 
#define OPH_Planes 1
#define OPH_Compression 0
#define OPH_Xpixelpermeter 0x130B //2835 , 72 DPI
#define OPH_Ypixelpermeter 0x130B //2835 , 72 DPI
#define OPH_Pixel 0xFF

using namespace oph;

class GEN_DLL ophWRP : public ophGen
{

public:
	/**
	* @brief Constructor
	* @details Initialize variables.
	*/
	explicit ophWRP(void);

protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophWRP(void);

public:

	/**
	\defgroup loadPointCloud
	* @brief override
	* @{
	* @brief Import Point Cloud Data Base File : *.PYL file.
	* This Function is included memory location of Input Point Clouds.
	*/
	/**
	* @brief override
	* @param InputModelFile PointCloud(*.PYL) input file path
	* @return number of Pointcloud (if it failed loading, it returned -1)
	*/
	virtual int loadPointCloud(const char* pc_file);
	virtual bool readConfig(const char* cfg_file);
	virtual void normalize(void);

	void encodeHologram(void);

	double calculateWRP(void);

	virtual void fresnelPropagation(Complex<Real>* in, Complex<Real>* out, Real distance);

	void generateHologram(void);

	oph::Complex<Real>** calculateMWRP(void);

	inline oph::Complex<Real>* getWRPBuff(void) { return p_wrp_; };


private:

	Complex<Real>* ophWRP::calSubWRP(double d, oph::Complex<Real>* wrp, OphPointCloudData* sobj);

	void addPixel2WRP(int x, int y, oph::Complex<Real> temp);
	void addPixel2WRP(int x, int y, oph::Complex<Real> temp, oph::Complex<Real>* wrp);

	virtual void ophFree(void);

protected:

	int n_points;   //number of points


	oph::Complex<Real>* p_wrp_;   //wrp buffer

	OphPointCloudData obj_;

	OphWRPConfig pc_config_;

};
#endif
