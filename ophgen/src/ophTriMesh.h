#ifndef __ophTriMesh_h
#define __ophTriMesh_h

#include "ophGen.h"

using namespace oph;

struct geometric {
	Real glRot[9];
	Real glShift[3];
	Real loRot[4];
};

class GEN_DLL ophTri : public ophGen
{
public:
	/**
	* @brief Constructor
	*/
	explicit ophTri(void) {}

protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophTri(void) {}

private:
	//OphTriangleObject object_config;
	//OphTriangleShading shading_config;

private:
	uint num_mesh = 0;
	Real* triMeshData;
	Complex<Real>* angularSpectrum;

private:
	Real objSize;
	Real objShift[3];

	Real carrierWave[3] = { 0,0,1 };

	vec3 illumination;

public:
	void setObjSize(Real in) { objSize = in; }
	void setObjShift(Real in[]) { objShift[_X] = in[_X]; objShift[_Y] = in[_Y]; objShift[_Z] = in[_Z]; }
	void setObjShift(vector<Real> in) { objShift[_X] = in[_X]; objShift[_Y] = in[_Y]; objShift[_Z] = in[_Z]; }
	Real getNumMesh() { return num_mesh; }
	Real* getMeshData() { return triMeshData; }
	Complex<Real>* getAngularSpectrum() { return angularSpectrum; }
	Real* getScaledMeshData() {	return scaledMeshData; }

public:
	/**
		@brief mesh text data load
	*/
	void loadMeshData(const char* fileName);
	void loadConfig(const string configFile);

	void objScaleShift();
	void objScaleShift(Real objSize_, vector<Real> objShift_);
	void objScaleShift(Real objSize_, Real objShift_[]);

	enum SHADING_FLAG { SHADING_FLAT, SHADING_CONTINUOUS };
	void generateAS(uint SHADING_FLAG);
	
private:
	void objNormCenter();
	uint checkValidity(Real* mesh, vec3 no);
	uint findGeometricalRelations(Real* mesh, vec3 no);
	void calGlobalFrequency();
	uint calFrequencyTerm();
	void refAS_Flat(vec3 no);
	void refAS_Continuous();
	uint findNormalForContinuous();
	void refToGlobal();


private:
	Real* normalizedMeshData;
	Real* scaledMeshData;

private:
	Real refTri[9] = { 0,0,0,1,1,0,1,0,0 };
	Real* fx;
	Real* fy;
	Real* fz;

private:
	geometric geom;
	Real* mesh_local;
	Real* flx;
	Real* fly;
	Real* flz;
	Real* freqTermX;
	Real* freqTermY;
	Complex<Real>* refAS;

};



#endif