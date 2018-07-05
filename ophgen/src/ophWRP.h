
#define _USE_MATH_DEFINES

#include "ophGen.h"
//#include "complex.h"

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
	//	virtual ~ophWRP(void);

public:

	int loadwPointCloud(const char* pc_file, bool colorinfo);
	virtual bool readConfig(const char* cfg_file);
	double calculateWRP(double wrp_d);
	oph::Complex<Real>** calculateWRP(int n);

private:
	inline oph::Complex<Real>* getWRPBuff(void) { return p_wrp_; };
	OphPointCloudData* vector2pointer(std::vector<OphPointCloudData> vec);
	Complex<Real>* ophWRP::subWRP_calcu(double d, oph::Complex<Real>* wrp, OphPointCloudData* sobj);
	int pobj2vecobj();
	void AddPixel2WRP(int x, int y, oph::Complex<Real> temp);
	void AddPixel2WRP(int x, int y, oph::Complex<Real> temp, oph::Complex<Real>* wrp);

protected:

	int n_points;   //number of points
	std::vector<Real> vertex_array_;
	std::vector<Real> amplitude_array_;
	std::vector<Real> phase_array_;

	oph::Complex<Real>* p_wrp_;   //wrp buffer
	OphPointCloudData* obj_;
	vector<OphPointCloudData> vec_obj;

	//	ophObjPoint* obj_;   

	OphPointCloudConfig pc_config_;

};

