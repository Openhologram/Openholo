#pragma once

#ifndef __ophLUT_h
#define __ophLUT_h

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

using namespace oph;

class GEN_DLL ophLUT : public ophGen
{
public:
	explicit ophLUT();
protected:
	virtual ~ophLUT();

public:
	int init(const char* _filename, CGHEnvironmentData* _CGHE);		// 초기화

	bool loadConfig(const char* filename, CGHEnvironmentData* _CGHE);
	bool loadPoint(const char* _filename, VoxelStruct* h_vox);
	bool load_Num_Point(const char* _filename, long* num_point);

	void DataInit(CGHEnvironmentData* _CGHE);	//데이터 초기화
	int LUTCalcuation(long voxnum, unsigned char *cghfringe, VoxelStruct* h_vox, CGHEnvironmentData* _CGHE);	// 패턴계산
	void APAS(long voxelnum, VoxelStruct* _h_vox, CGHEnvironmentData* _CGHE);

	int saveAsImg(const char * fname, uint8_t bitsperpixel, void* src, int pic_width, int pic_height);	// 이미지 저장

	//util 
	char* trim(char *s); // 문자열 좌우 공백 모두 삭제 함수
	char* ltrim(char *s); // 문자열 좌측 공백 제거 함수
	char* rtrim(char* s); // 문자열 우측 공백 제거 함수

	double *m_pHologram;
	Segment *m_Segment;
	float m_COStbl[NUMTBL];
	float m_SINtbl[NUMTBL];

	//long num_point = 0;	// number of point cloud

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
#endif // !__ophLUT_h