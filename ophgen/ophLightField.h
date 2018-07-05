#ifndef __ophLightField_h
#define __ophLightField_h

#include "ophGen.h"
#include <fstream>
#include <io.h>

using namespace oph;

class GEN_DLL ophLF : public ophGen
{
public:
	/**
	* @brief Constructor
	*/
	explicit ophLF(void) {}

protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophLF(void) {}

private:
	OphLightFieldConfig LF_config;
private:
	real** LF;
	oph::Complex<real>* RSplane_complex_field;
protected:
	inline void setNumImage(int nx, int ny) { LF_config.num_image[_X] = nx; LF_config.num_image[_Y] = ny; }
	inline void setResolImage(int nx, int ny) { LF_config.resolution_image[_X] = nx; LF_config.resolution_image[_Y] = ny; }
	inline void setDistRS2Holo(real dist) { LF_config.distanceRS2Holo = dist; }
	
	inline ivec2 getNumImage() { return LF_config.num_image; }
	inline ivec2 getResolImage() { return LF_config.resolution_image; }
	inline real getDistRS2Holo() { return LF_config.distanceRS2Holo; }
	inline real** getLF() { return LF; }
	inline oph::Complex<real>* getRSPlane() { return RSplane_complex_field; }
public:
	int readConfig(const char* LF_config);
	void generateHologram();
protected:
	int loadLF(const char* LF_directory, const char* ext);
	void convertLF2ComplexField();
	void fresnelPropagation(); 

private:
	ivec2 num_image;
	ivec2 resolution_image;
	real distanceRS2Holo;

};


#endif