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
	uchar** LF;
	oph::Complex<Real>* RSplane_complex_field;
public:
	inline void setNumImage(int nx, int ny) { num_image[_X] = nx; num_image[_Y] = ny; }
	inline void setNumImage(ivec2 num) { num_image = num; }
	inline void setResolImage(int nx, int ny) { resolution_image[_X] = nx; resolution_image[_Y] = ny; }
	inline void setResolImage(ivec2 num) { resolution_image = num; }
	inline void setDistRS2Holo(Real dist) { distanceRS2Holo = dist; }
	
	inline ivec2 getNumImage() { return num_image; }
	inline ivec2 getResolImage() { return resolution_image; }
	inline Real getDistRS2Holo() { return distanceRS2Holo; }
	inline uchar** getLF() { return LF; }
	inline oph::Complex<Real>* getRSPlane() { return RSplane_complex_field; }
public:
	int readConfig(const char* LF_config);
	int loadLF(const char* LF_directory, const char* ext);
	void readPNG(const char* filename);

	void generateHologram();
protected:
	void initializeLF();
	void convertLF2ComplexField();
	void fresnelPropagation(); 

private:
	ivec2 num_image;
	ivec2 resolution_image;
	Real distanceRS2Holo;

};


#endif