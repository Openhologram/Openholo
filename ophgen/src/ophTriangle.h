#ifndef __ophTriangle_h
#define __ophTriangle_h

#include "ophGen.h"

using namespace oph;

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
	OphTriangleObject object_config;
	OphTriangleShading shading_config;

private:
	real* triMeshData;
	real* angularSpectrum;

public:
	void loadMeshData(const string fileName);
	void loadConfig(const string configFile);



	void refAS_Flat();
	void refAS_Continuous();

public:
	void objNormCenter();
	void objScaleShift();


};

#endif