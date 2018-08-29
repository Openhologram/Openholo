/**
* @file		ophLightField.h
* @brief	Openholo Light Field based CGH generation
* @author	Yeon-Gyeong Ju, Jae-Hyeung Park
* @data		2018-08
* @version	0.0.1
*/
#ifndef __ophLightField_h
#define __ophLightField_h

#include "ophGen.h"
#include <fstream>
#include <io.h>

using namespace oph;

/**
* @brief	Openholo Light Field based CGH generation class
*/
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
	/**
	* @param	uchar**			LF						Light Field array / 4-D array
	* @param	Complex<Real>*	RSplane_complex_field	Complex field in Ray Sampling plane
	*/
	uchar** LF;
	Complex<Real>* RSplane_complex_field;

private:
	/**
	* @brief	Light Field save parameters
	*/

	const char* LF_directory;
	const char* ext;
public:
	/** \ingroup */
	inline void setNumImage(int nx, int ny) { num_image[_X] = nx; num_image[_Y] = ny; }
	/** \ingroup */
	inline void setNumImage(ivec2 num) { num_image = num; }
	/** \ingroup */
	inline void setResolImage(int nx, int ny) { resolution_image[_X] = nx; resolution_image[_Y] = ny; }
	/** \ingroup */
	inline void setResolImage(ivec2 num) { resolution_image = num; }
	/** \ingroup */
	inline void setDistRS2Holo(Real dist) { distanceRS2Holo = dist; }
	/** \ingroup */
	inline ivec2 getNumImage() { return num_image; }
	/** \ingroup */
	inline ivec2 getResolImage() { return resolution_image; }
	/** \ingroup */
	inline Real getDistRS2Holo() { return distanceRS2Holo; }
	/** \ingroup */
	inline uchar** getLF() { return LF; }
	/** \ingroup */
	inline oph::Complex<Real>* getRSPlane() { return RSplane_complex_field; }
public:
	/**
	* @brief	Light Field based CGH configuration file load
	* @details	xml configuration file load
	* @return	distanceRS2Holo
	* @return	num_image
	* @return	resolution_image
	* @return	context_.pixel_pitch
	* @return	context_.pixel_number
	* @return	context_.lambda
	*/
	int readLFConfig(const char* LF_config);
	/**
	* @brief	Light Field images load
	* @param	directory		Directory which has the Light Field source image files
	* @param	exten			Light Field images extension
	* @return	LF
	* @overload
	*/
	int loadLF(const char* directory, const char* exten);
	int loadLF();
	//void readPNG(const string filename, uchar* data);

	/**
	* @brief	Hologram generation
	* @return	holo_gen
	*/
	void generateHologram();

protected:
	/**
	* @brief inner functions
	*/

	void initializeLF();
	void convertLF2ComplexField();

private:
	/**
	* @param	ivec2	num_image			The number of LF source images {numX, numY}
	* @param	ivec2	resolution_image	Resolution of LF source images {resolutionX, resolutionY}
	* @param	Real	distanceRS2Holo		Distance from Ray Sampling plane to Hologram plane
	*/

	ivec2 num_image;
	ivec2 resolution_image;
	Real distanceRS2Holo;
};


#endif