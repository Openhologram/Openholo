
#define _USE_MATH_DEFINES

#include "ophGen.h"

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

class ophWRP : public ophGen
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
	//	virtual ~ophWRP(void);

public:

	/**
	\defgroup PointCloud_Load
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

	void initialize();

	void encodefield(void);

	double calculateWRP(double wrp_d);

	oph::Complex<Real>** calculateMWRP(int n);

	inline oph::Complex<Real>* getWRPBuff(void) { return p_wrp_; };


private:

	Complex<Real>* ophWRP::calSubWRP(double d, oph::Complex<Real>* wrp, OphPointCloudData* sobj);

	void AddPixel2WRP(int x, int y, oph::Complex<Real> temp);

	void AddPixel2WRP(int x, int y, oph::Complex<Real> temp, oph::Complex<Real>* wrp);

	virtual void ophFree(void);

protected:

	int n_points;   //number of points


	oph::Complex<Real>* p_wrp_;   //wrp buffer

	OphPointCloudData obj_;

	OphPointCloudConfig pc_config_;

};

