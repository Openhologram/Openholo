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
#define FFT_SEG_SIZE	64
#define SEG_SIZE		8

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

using namespace oph;

class GEN_DLL ophACPAS : public ophGen
{
public:
	explicit ophACPAS();
protected:
	virtual ~ophACPAS();

public:
	bool readConfig(const char* fname);
	int loadPointCloud(const char* pc_file);
	int save(const char * fname, uint8_t bitsperpixel, uchar* src, uint px, uint py);

	void DataInit();
	int ACPASCalcuation(unsigned char *cghfringe);	// 패턴계산
	void ACPAS();

	double *m_pHologram;

	float m_COStbl[NUMTBL];
	float m_SINtbl[NUMTBL];
private:
	int n_points;
	OphPointCloudConfig pc_config_;
	OphPointCloudData	pc_data_;
	CGHEnvironmentData	env;
};



#endif // !__ophPAS_h