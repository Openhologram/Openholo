/**
* @file		ophTriMesh.h
* @brief	Openholo Triangular Mesh based CGH generation
* @author	Yeon-Gyeong Ju, Jae-Hyeung Park
* @data		2018-08
* @version	0.0.1
*/

#ifndef __ophTriMesh_h
#define __ophTriMesh_h

#include "ophGen.h"

using namespace oph;

/**
* @brief	geometrical relations
* @details	inner parameters
*/
struct geometric {
	Real glRot[9];
	Real glShift[3];
	Real loRot[4];
};

/**
* @brief	Openholo Triangular Mesh based CGH Generation Class
*/
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
	/**
	* @param	Real*				triMeshArray		Original triangular mesh array (N*9)
	* @param	Complex<Real>*		angularSpectrum		Angular spectrum of the hologram
	* @param	OphMeshData*		meshData			OphMeshData type data structure pointer
	*/

	Real* triMeshArray;
	Complex<Real>* angularSpectrum;
	OphMeshData* meshData;

private:
	/**
	* @param	Real	objSize				Object maximum of width and height / unit :[m]
	* @param	Real	objShift			Object shift value / Data structure - [shiftX, shiftY, shiftZ] / unit : [m]
	* @param	Real	carrierWave[3]		Carrier wave direction / default : {0, 0, 1}
	* @param	vec3	illumination		Position of the light source (for shading effect) / No-illumination : {0, 0, 0}
	* @param	int		SHADING_TYPE		SHADING_FLAT, SHADING_CONTINUOUS
	*/

	Real objSize;
	Real objShift[3];

	Real carrierWave[3] = { 0,0,1 };

	vec3 illumination;
	int SHADING_TYPE;

public:
	/** \ingroup */
	void setObjSize(Real in) { objSize = in; }
	/** \ingroup */
	void setObjShift(Real in[]) { objShift[_X] = in[_X]; objShift[_Y] = in[_Y]; objShift[_Z] = in[_Z]; }
	/** \ingroup */
	void setObjShift(vector<Real> in) { objShift[_X] = in[_X]; objShift[_Y] = in[_Y]; objShift[_Z] = in[_Z]; }
	/** \ingroup */
	void setCarrierWave(Real in1, Real in2, Real in3) { carrierWave[_X] = in1; carrierWave[_Y] = in2; carrierWave[_Z] = in3; }
	/** \ingroup */
	void setIllumination(vec3 in) { illumination = in; }
	/** \ingroup */
	void setIllumination(Real inx, Real iny, Real inz) { illumination = { inx, iny, inz }; }
	/** \ingroup */
	void setShadingType(int in) { SHADING_TYPE = in; }
	/** \ingroup */
	ulonglong getNumMesh() { return meshData->n_faces; }
	/** \ingroup */
	Real* getMeshData() { return triMeshArray; }
	/** \ingroup */
	Complex<Real>* getAngularSpectrum() { return angularSpectrum; }
	/** \ingroup */
	Real* getScaledMeshData() {	return scaledMeshData; }

public:
	/**
	* @brief	Triangular mesh basc CGH configuration file load
	* @details	xml configuration file load
	* @return	context_.pixel_pitch
	* @return	context_.pixel_number
	* @return	context_.lambda
	* @return	illumination
	* @return	objSize
	* @return	objShift
	*/
	int readMeshConfig(const char* mesh_config);
	/**
	* @brief	Mesh data load
	* @details	Text file data structure : N*9 / Each row = [x1 y1 z1 x2 y2 z2 x3 y3 z3]
	* @details	File extension : txt, ply
	* @param	ext				File extension
	* @return	triMeshArray
	*/
	void loadMeshData(const char* fileName, const char* ext);

	/**
	* @brief	Mesh object data scaling and shifting
	* @param	objSize_		Object maximum of width and height / unit : [m]
	* @param	objShift_		Object shift value / Data structure : [shiftX, shiftY, shiftZ] / unit : [m]
	* @return	scaledMeshData
	* @overload
	*/
	void objScaleShift();
	void objScaleShift(Real objSize_, vector<Real> objShift_);
	void objScaleShift(Real objSize_, Real objShift_[]);

	enum SHADING_FLAG { SHADING_FLAT, SHADING_CONTINUOUS };
	/**
	* @brief	Hologram generation
	* @param	SHADING_FLAG : SHADING_FLAT, SHADING_CONTINUOUS
	* @overload
	*/
	void generateMeshHologram(uint SHADING_FLAG);
	void generateMeshHologram();
	
private:
	/**
	* @brief	inner functions
	* @details	not used for the users
	*/

	void initializeAS();
	void objNormCenter();
	uint checkValidity(Real* mesh, vec3 no);
	uint findGeometricalRelations(Real* mesh, vec3 no);
	void calGlobalFrequency();
	uint calFrequencyTerm();
	uint refAS_Flat(vec3 na);
	uint refAS_Continuous(uint n);
	void generateAS(uint SHADING_FLAG);
	uint findNormals(uint SHADING_FLAG);
	uint refToGlobal();

	uint loadMeshText(const char* fileName);
private:
	/**
	* @param	Real*	normalizedMeshData	Normalized mesh array / Data structure : N*9
	* @param	Real*	scaledMeshData		Scaled and shifted mesh array / Data structure : N*9
	*/

	Real* normalizedMeshData;
	Real* scaledMeshData;

private:
	/**
	* @brief	inner global parameters
	* @details	not considered for the users
	*/

	Real refTri[9] = { 0,0,0,1,1,0,1,0,0 };
	Real* fx;
	Real* fy;
	Real* fz;
	vec3* no;
	vec3* na;
	vec3* nv;

private:
	/**
	* @brief	inner local parameters
	* @details	not considered for the users
	*/

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