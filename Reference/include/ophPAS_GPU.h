#ifndef __ophPAS_GPU_h
#define __ophPAS_GPU_h

#include "ophGen.h"


using namespace oph;

#define PI				(3.14159265358979323846f)
#define M2_PI			(PI*2.0)
#define RADIANS			(PI/180.0)			// Angle in radians	
// DEGREE*asin(mytheta)
#define DEGREE2			(180./PI)			// sin(RADIANS*DEGREE*asin(mytheta))

#define NUMTBL			1024
#define NUMTBL2			(NUMTBL-1)
#define MAX_STR_LEN 4000

#define FFT_SEGMENT_SIZE 64


struct constValue {
	int* cf_cx;
	int* cf_cy;
	float* xc;
	float* yc;
	
	float* costbl;
	float* sintbl;
	float cx, cy, cz, amplitude;
	int segx, segy, segn;

	
};


class GEN_DLL ophPAS_GPU : public ophGen
{
private:
	OphPointCloudConfig pc_config;
	OphPointCloudData pc_data;
	constValue val_d;
	
	int n_points;
public:
	explicit ophPAS_GPU();
protected:
	virtual ~ophPAS_GPU();

public:



	bool readConfig(const char* fname);
	int loadPoint(const char* _filename);

	//int saveAsImg(const char * fname, uint8_t bitsperpixel, void* src, int pic_width, int pic_height);	// 이미지 저장
	int save(const char * fname, uint8_t bitsperpixel, uchar* src, uint px, uint py);
	void save(const char* fname);

	//util 
	char* trim(char *s); // 문자열 좌우 공백 모두 삭제 함수
	char* ltrim(char *s); // 문자열 좌측 공백 제거 함수
	char* rtrim(char* s); // 문자열 우측 공백 제거 함수
	


												//void PASCalcuation(long voxnum, unsigned char *cghfringe, VoxelStruct* h_vox, CGHEnvironmentData* _CGHE);
	void PASCalculation(long voxnum, unsigned char * cghfringe, OphPointCloudData *data, OphPointCloudConfig& conf);
	//void PAS(long voxelnum, struct VoxelStruct *voxel, double *m_pHologram, CGHEnvironmentData* _CGHE);
	void PAS(long voxelnum, OphPointCloudData *data, double *m_pHologram, OphPointCloudConfig& conf);
	void DataInit(int segsize, int cghwidth, int cghheight, float xiinter, float etainter);
	void DataInit(OphPointCloudConfig& conf);
	void MemoryRelease(void);

	void generateHologram();
	void PASCalculation_GPU(long voxnum, unsigned char * cghfringe, OphPointCloudData *data, OphPointCloudConfig& conf);

	
	void CalcSpatialFrequency(float cx, float cy, float cz, float amp, int segnumx, int segnumy, int segsize, int hsegsize, float sf_base, float * xc, float * yc, float * sf_cx, float * sf_cy, int * pp_cx, int * pp_cy, int * cf_cx, int * cf_cy, float xiint, float etaint, OphPointCloudConfig& conf);
	
	void CalcCompensatedPhase(float cx, float cy, float cz, float amp, int segnumx, int segnumy, int segsize, int hsegsize, float sf_base, float *xc, float *yc, int *cf_cx, int *cf_cy, float *COStbl, float *SINtbl, float **inRe, float **inIm, OphPointCloudConfig& conf);
	
	void RunFFTW(int segnumx, int segnumy, int segsize, int hsegsize, float **inRe, float **inIm, fftw_complex *in, fftw_complex *out, fftw_plan *plan, double *pHologram, OphPointCloudConfig& conf);

	void encodeHologram(const vec2 band_limit, const vec2 spectrum_shift);

	void encoding(unsigned int ENCODE_FLAG);
	

	double *m_pHologram;

	float* m_COStbl;
	float* m_SINtbl;

	int m_segSize;
	int m_hsegSize;
	int m_dsegSize;
	int m_segNumx;
	int m_segNumy;
	int m_hsegNumx;
	int m_hsegNumy;

	float	*m_SFrequency_cx;
	float	*m_SFrequency_cy;
	int		*m_PickPoint_cx;
	int		*m_PickPoint_cy;
	int		*m_Coefficient_cx;
	int		*m_Coefficient_cy;
	float	*m_xc;
	float	*m_yc;
	unsigned char* cgh_fringe;

	float	m_sf_base;
	float  *m_inReHost;
	float  *m_inImHost;
	fftw_complex *m_in, *m_out;
	fftw_plan m_plan;

	float	**m_inRe;
	float	**m_inIm;
	float   *m_inRe_h;
	float	*m_inIm_h;

	float	m_cx;
	float	m_cy;
	float	m_cz;
	float	m_amp;
};


#endif // !__ophPAS_h