#ifndef __ophNonHogelLF_h
#define __ophNonHogelLF_h

#include "ophGen.h"
#include <fstream>
#include <io.h>

using namespace oph;


//Build Option : Multi Core Processing (OpenMP)
#ifdef _OPENMP
#include <omp.h>
#endif#pragma once


class GEN_DLL ophNonHogelLF : public ophGen
{
public:
	/**
	* @brief Constructor
	*/
	explicit ophNonHogelLF(void);

protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophNonHogelLF(void) {}

private:

	uchar** LF;										/// Light Field array / 4-D array
	Complex<Real>** FToverUV_LF;					/// 2D Fourier transform of light field
	Complex<Real>* WField;							/// Complex field determining the carrier wave
	Complex<Real>* Hologram;						/// Generated complex field

private:

	// Light Field save parameters

	const char* LF_directory;
	const char* ext;

public:
	inline void setNumImage(int nx, int ny) { num_image[_X] = nx; num_image[_Y] = ny; }
	inline void setNumImage(ivec2 num) { num_image = num; }
	inline void setResolImage(int nx, int ny) { resolution_image[_X] = nx; resolution_image[_Y] = ny; }
	inline void setResolImage(ivec2 num) { resolution_image = num; }
	inline void setDistRS2Holo(Real dist) { distanceRS2Holo = dist; }
	inline void setFieldLens(Real lens) { fieldLens = lens; }
	inline ivec2 getNumImage() { return num_image; }
	inline ivec2 getResolImage() { return resolution_image; }
	inline Real getDistRS2Holo() { return distanceRS2Holo; }
	inline Real getFieldLens() { return fieldLens; }
	inline uchar** getLF() { return LF; }
	
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
	bool readConfig(const char* fname);

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
	* @return	(*complex_H)
	*/
	void preprocessLF();
	void generateHologram();
	void generateHologram(double thetaX, double thetaY);

	/**
	* @brief Function for setting precision
	* @param[in] precision level.
	*/
	void setPrecision(bool bPrecision) { bSinglePrecision = bPrecision; }
	bool getPrecision() { return bSinglePrecision; }

	// for Testing 
	void writeIntensity_gray8_bmp(const char* fileName, int nx, int ny, Complex<Real>* complexvalue, int k = -1);
	/**
	* @brief Set the value of a variable is_ViewingWindow(true or false)
	* @details <pre>
	if is_ViewingWindow == true
	Transform viewing window
	else
	Hologram </pre>
	* @param is_ViewingWindow : the value for specifying whether the hologram generation method is implemented on the viewing window
	*/
	void setViewingWindow(bool is_ViewingWindow);
protected:

	// Inner functions

	void initializeLF();
	void convertLF2ComplexFieldUsingNonHogelMethod();
	void makeRandomWField();
	void makePlaneWaveWField(double thetaX, double thetaY);
	void fourierTransformOverUVLF();
	void setBuffer();

	// ==== GPU Methods ===============================================
	//void prepareInputdataGPU();
	//void convertLF2ComplexField_GPU();
	//void fresnelPropagation_GPU();

private:

	ivec2 num_image;						/// The number of LF source images {numX, numY}
	ivec2 resolution_image;					/// Resolution of LF source images {resolutionX, resolutionY}
	Real distanceRS2Holo;					/// Distance from Ray Sampling plane to Hologram plane
	Real fieldLens;
	bool is_ViewingWindow;
	bool bSinglePrecision;
	int nImages;
	int nBufferX;
	int nBufferY;
};


#endif