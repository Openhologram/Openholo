#pragma once

#ifndef __ophACPAS_h
#define __ophACPAS_h

#include "ophGen.h"

#define NUMTBL			1024
#define NUMTBL2			(NUMTBL-1)
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