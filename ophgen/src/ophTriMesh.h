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
	explicit ophTri(void) {
		
	}

protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophTri(void) {}

private:
	const char* meshDataFileName;

private:
	uint num_mesh = 0;
	Real* triMeshData;
	Complex<Real>* angularSpectrum;

private:
	Real objSize;
	Real objShift[3];

	Real carrierWave[3] = { 0,0,1 };

	int SHADING_TYPE;
	vec3 illumination;

public:
	void setObjSize(Real in) { objSize = in; }
	void setObjShift(Real in[]) { objShift[_X] = in[_X]; objShift[_Y] = in[_Y]; objShift[_Z] = in[_Z]; }
	void setObjShift(vector<Real> in) { objShift[_X] = in[_X]; objShift[_Y] = in[_Y]; objShift[_Z] = in[_Z]; }
	/** CarrierWave = {0, 0, 1} (default) */
	void setCarrierWave(Real in1, Real in2, Real in3) { carrierWave[_X] = in1; carrierWave[_Y] = in2; carrierWave[_Z] = in3; }
	/** No-Illumination Effect : Illumination = {0, 0, 0} */
	void setIllumination(vec3 in) { illumination = in; }
	void setIllumination(Real inx, Real iny, Real inz) { illumination = { inx, iny, inz }; }
	/** Shading Type : SHADING_FLAT, SHADING_CONTINUOUS */
	void setShadingType(int in) { SHADING_TYPE = in; }
	Real getNumMesh() { return num_mesh; }
	Real* getMeshData() { return triMeshData; }
	Complex<Real>* getAngularSpectrum() { return angularSpectrum; }
	Real* getScaledMeshData() {	return scaledMeshData; }

public:
	int readMeshConfig(const char* mesh_config);
	/**
	*@brief mesh data load
	* file extension : .txt
	* data structure : each row = [x1 y1 z1 x2 y2 z2 x3 y3 z3]
	*/
	void loadMeshData(const char* fileName);
	void loadMeshData();

	void objScaleShift();
	void objScaleShift(Real objSize_, vector<Real> objShift_);
	void objScaleShift(Real objSize_, Real objShift_[]);

	enum SHADING_FLAG { SHADING_FLAT, SHADING_CONTINUOUS };
	void generateAS(uint SHADING_FLAG);
	void generateMeshHologram(uint SHADING_FLAG);
	void generateMeshHologram();
	
private:
	void initializeAS();
	void objNormCenter();
	uint checkValidity(Real* mesh, vec3 no);
	uint findGeometricalRelations(Real* mesh, vec3 no);
	void calGlobalFrequency();
	uint calFrequencyTerm();
	uint refAS_Flat(vec3 na);
	uint refAS_Continuous(uint n);
	uint findNormals(uint SHADING_FLAG);
	void refToGlobal();


private:
	Real* normalizedMeshData;
	Real* scaledMeshData;

private:
	Real refTri[9] = { 0,0,0,1,1,0,1,0,0 };
	Real* fx;
	Real* fy;
	Real* fz;
	vec3* no;
	vec3* na;
	vec3* nv;

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