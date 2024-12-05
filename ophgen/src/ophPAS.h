#pragma once

#ifndef __ophPAS_h
#define __ophPAS_h

#include "ophGen.h"
#define NUMTBL			1024
#define NUMTBL2			(NUMTBL-1)

#define FFT_SEGMENT_SIZE	64
#define SEG_SIZE			8

using namespace oph;

class GEN_DLL ophPAS : public ophGen
{
private:
	OphPointCloudConfig pc_config;
	OphPointCloudData pc_data;
public:
	explicit ophPAS();
	void Init();
	void InitGPU();
	void PAS();
	void CreateLookupTables();
	void CalcSpatialFrequency(Point* pt, Real lambda, bool accurate = false);
	void CalcCompensatedPhase(Point* pt, Real amplitude, Real phase, Real lambda, bool accurate = false);

protected:
	virtual ~ophPAS();

public:
	void setAccurate(bool accurate) { this->is_accurate = accurate; }
	bool getAccurate() { return this->is_accurate; }
	bool readConfig(const char* fname);
	int loadPoint(const char* _filename);

	void generateHologram();

	void encodeHologram(const vec2 band_limit, const vec2 spectrum_shift);
	void encoding(unsigned int ENCODE_FLAG);


private:
	Real LUTCos[NUMTBL] = {};
	Real LUTSin[NUMTBL] = {};

	bool is_accurate;
	int* coefficient_cx;
	int* coefficient_cy;
	Real* compensation_cx;
	Real* compensation_cy;
	Real* xc;
	Real* yc;
	Complex<Real>** input;
};

#endif // !__ophPAS_h