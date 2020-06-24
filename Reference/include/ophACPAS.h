#pragma once

#ifndef __ophACPAS_h
#define __ophACPAS_h

#include "ophGen.h"


#define PI				(3.14159265358979323846f)
#define M2_PI			(PI*2.0)
#define RADIANS			(PI/180.0)			// Angle in radians	
// DEGREE*asin(mytheta)
#define DEGREE2			(180./PI)			// sin(RADIANS*DEGREE*asin(mytheta))

#define NUMTBL			1024
#define NUMTBL2			(NUMTBL-1)
#define MAX_STR_LEN 4000

struct VoxelStruct;
struct CGHEnvironmentData;
struct Segment;

#define FFT_SEG_SIZE	64
#define SEG_SIZE		8

using namespace oph;

class GEN_DLL ophACPAS : public ophGen
{
public:
	explicit ophACPAS();
protected:
	virtual ~ophACPAS();

public:
	int init(const char* _filename, CGHEnvironmentData* _CGHE);		// 초기화

	bool loadConfig(const char* filename, CGHEnvironmentData* _CGHE);
	bool readConfig(const char* fname, OphPointCloudConfig& configdata);
	bool loadPoint(const char* _filename, VoxelStruct* h_vox);
	bool load_Num_Point(const char* _filename, long* num_point);
	//int saveAsImg(const char * fname, uint8_t bitsperpixel, void* src, int pic_width, int pic_height);	// 이미지 저장
	int save(const char * fname, uint8_t bitsperpixel, uchar* src, uint px, uint py);

	//util 
	char* trim(char *s); // 문자열 좌우 공백 모두 삭제 함수
	char* ltrim(char *s); // 문자열 좌측 공백 제거 함수
	char* rtrim(char* s); // 문자열 우측 공백 제거 함수

	void DataInit(CGHEnvironmentData* _CGHE);	//데이터 초기화
	void DataInit(OphPointCloudConfig& conf);
	int ACPASCalcuation(long voxnum, unsigned char *cghfringe, VoxelStruct* h_vox, CGHEnvironmentData* _CGHE);	// 패턴계산
	int ACPASCalcuation(long voxnum, unsigned char *cghfringe, OphPointCloudData *data, OphPointCloudConfig& conf);	// 패턴계산
	void ACPAS(long voxelnum, VoxelStruct* _h_vox, CGHEnvironmentData* _CGHE);
	void ACPAS(long voxelnum, OphPointCloudData *data, OphPointCloudConfig& conf);

	double *m_pHologram;

	float m_COStbl[NUMTBL];
	float m_SINtbl[NUMTBL];

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

	float	m_sf_base;

	fftw_complex *m_in, *m_out;
	fftw_plan m_plan;

	float	**m_inRe;
	float	**m_inIm;

	float	m_cx;
	float	m_cy;
	float	m_cz;
	float	m_amp;
};

struct GEN_DLL VoxelStruct							// voxel structure - data
{
	int num;								// voxel or point number
	float x;								// x axis coordinate
	float y;								// y axis coordinate
	float z;								// z axis coordinate
	float ph;								// phase
	float r;								// amplitude in red channel
	//float g;								// amplitude in green channel
	//float b;								// amplitude in blue channel
};

struct GEN_DLL  CGHEnvironmentData
{
	int		CghWidth;			// cgh width
	int		CghHeight;		// cgh height
	int		SegmentationSize;
	int		fftSegmentationSize;
	float	rWaveLength;		// red laser lambda
	float	rWaveNumber;		// red laser lambda
	float	ThetaX;
	float	ThetaY;
	float	DefaultDepth;
	float 	xInterval;
	float 	yInterval;
	float 	xiInterval;
	float 	etaInterval;
	float	CGHScale;
};

struct GEN_DLL Segment
{
	bool	WorkingFlag;
	long	SegmentIndex;
	int		SegSize_x;
	int		SegSize_y;
	int 	hSegSize_x;		// Half size
	int 	hSegSize_y;		// Half size
	double	CenterX;
	double	CenterY;
	double	FrequencySlope;
};
#endif // !__ophPAS_h