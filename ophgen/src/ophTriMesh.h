/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install, copy or use the software.
//
//
//                           License Agreement
//                For Open Source Digital Holographic Library
//
// Openholo library is free software;
// you can redistribute it and/or modify it under the terms of the BSD 2-Clause license.
//
// Copyright (C) 2017-2024, Korea Electronics Technology Institute. All rights reserved.
// E-mail : contact.openholo@gmail.com
// Web : http://www.openholo.org
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//  1. Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holder or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
// This software contains opensource software released under GNU Generic Public License,
// NVDIA Software License Agreement, or CUDA supplement to Software License Agreement.
// Check whether software you use contains licensed software.
//
//M*/

#ifndef __ophTriMesh_h
#define __ophTriMesh_h

#include "ophGen.h"
#include "sys.h"

//Build Option : Multi Core Processing (OpenMP)
#ifdef _OPENMP
#include <omp.h>
#endif
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
* @brief	texture mapping parameters
* @details
*/
struct TextMap {
	Complex<Real>* pattern;
	ivec2 dim;
	Real period;
	Real freq;
};


/**
* @addtogroup mesh
//@{
* @details

* @section Introduction

Triangular mesh based CGH generates the complex field of 3D objects represented as a collection of the triangular meshes.
The algorithm aggregates the angular spectrums of individual triangular meshes and then performs a Fourier transform to obtain the complex field for entire objects.

![](pics/ophgen/mesh/mesh_fig3.png)

The angular spectrum of the individual triangular mesh is obtained using the analytic formula of the Fourier transform of the reference triangular aperture, considering the geometrical relation between the hologram plane and the local mesh plane, and also between the local mesh and the reference triangular aperture.

![](pics/ophgen/mesh/mesh_fig1.png)
![](pics/ophgen/mesh/mesh_fig2.png)

The phase distribution on the mesh is determined by the carrier wave is assumed to be a plane wave of a specfic direction in the code.
The amplitude inside each mesh is determined by the surface shading model and it can be either linearly varying for the continuous shading or uniform for the flat shading.

![continuous shading](pics/ophgen/mesh/mesh_ex_continuous.png)
-Fig.continuous shading
![flat shading](pics/ophgen/mesh/mesh_ex_flat.png)
-Fig.flat shading

*/
//! @} mesh

/**
* @ingroup mesh
* @brief Openholo Triangular Mesh based CGH generation
* @author Yeon-Gyeong Ju, Jae-Hyeung Park
*/
class GEN_DLL ophTri : public ophGen
{
public:
	/**
	* @brief Constructor
	* @details Initialize variables.
	*/
	explicit ophTri(void);

protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophTri(void) {}

private:
	const char* meshDataFileName;

private:

	Real* triMeshArray;						/// Original triangular mesh array (N*9)
	Complex<Real>* angularSpectrum;			/// Angular spectrum of the hologram
	OphMeshData* meshData;					/// OphMeshData type data structure pointer

private:

	//Real fieldLength;
	vec3 objSize;							/// Object maximum of width and height / unit :[m]

	Real carrierWave[3] = { 0,0,1 };		/// Carrier wave direction / default : {0, 0, 1}
	vec3 illumination;						/// Position of the light source (for shading effect) / No-illumination : {0, 0, 0}
	int SHADING_TYPE;						/// SHADING_FLAT, SHADING_CONTINUOUS

	bool randPhase;
	bool occlusion;
	bool textureMapping;
	TextMap texture;

public:
	void setObjSize(vec3 in) { objSize = in; }
	void setObjShift(vec3 in) { context_.shift[_X] = in[_X]; context_.shift[_Y] = in[_Y]; context_.shift[_Z] = in[_Z]; }
	void setCarrierWave(Real in1, Real in2, Real in3) { carrierWave[_X] = in1; carrierWave[_Y] = in2; carrierWave[_Z] = in3; }
	void setIllumination(vec3 in) { illumination = in; }
	void setIllumination(Real inx, Real iny, Real inz) { illumination = { inx, iny, inz }; }
	void setShadingType(int in) { SHADING_TYPE = in; }

	void setRandPhase(bool in) { randPhase = in; }
	void setOcclusion(bool in) { occlusion = in; }
	void setTextureMapping(bool in) { textureMapping = in; }
	void setTextureImgDim(int dim1, int dim2) { texture.dim[0] = dim1; texture.dim[1] = dim2; }
	void setTexturePeriod(Real in) { texture.period = in; }

	ulonglong getNumMesh() { return meshData->n_faces; }
	Real* getMeshData() { return triMeshArray; }
	Complex<Real>* getAngularSpectrum() { return angularSpectrum; }
	Real* getScaledMeshData() { return scaledMeshData; }

	const vec3& getObjSize(void) { return objSize; }
	const vec3& getObjShift(void) { return context_.shift; }
	const vec3&	getIllumination(void) { return illumination; }
	//const Real& getFieldLens(void) { return fieldLength; }

	/**
	* @brief Function for setting precision
	* @param[in] precision level.
	*/
	void setPrecision(bool bPrecision) { bSinglePrecision = bPrecision; }
	bool getPrecision() { return bSinglePrecision; }
public:
	/**
	* @brief	Triangular mesh basc CGH configuration file load
	* @details	xml configuration file load
	* @return bool return false : Failed to load configure file
	*			   return true : Success to load configure file
	*/
	bool readConfig(const char* fname);

	/**
	* @brief	Mesh data load
	* @details	Text file data structure : N*9 / Each row = [x1 y1 z1 x2 y2 z2 x3 y3 z3]
	* @details	File extension : txt, ply
	* @param	ext				File extension
	* @return bool return false : Failed to load mesh data
	*			   return true : Success to load mesh data
	*/
	bool loadMeshData(const char* fileName, const char* ext);

	/**
	* @brief	Mesh object data scaling and shifting
	* @param	objSize_		Object maximum of width and height / unit : [m]
	* @param	objShift_		Object shift value / Data structure : [shiftX, shiftY, shiftZ] / unit : [m]
	* @return	scaledMeshData
	* @overload
	*/
	//void objScaleShift();
	//void objScaleShift(vec3 objSize_, vector<Real> objShift_);
	//void objScaleShift(vec3 objSize_, vec3 objShift_);

	enum SHADING_FLAG { SHADING_FLAT, SHADING_CONTINUOUS };

	/**
	*/
	void loadTexturePattern(const char* fileName, const char* ext);


	/**
	* @brief	Hologram generation
	* @param	SHADING_FLAG : SHADING_FLAT, SHADING_CONTINUOUS
	* @overload
	*/
	bool generateHologram(uint SHADING_FLAG);

	void reconTest(const char* fname);

	bool TM = false;
	int idxCarr, idxCarrFx, idxCarrFy;
	void triTimeMultiplexing(char* dirName, uint ENCODE_METHOD, Real cenFx, Real cenFy, Real rangeFx, Real rangeFy, Real stepFx, Real stepFy);

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

	uint* getProgress() { return &m_nProgress; }
private:

	// Inner functions
	/// not used for users

	void initializeAS();
	void prepareMeshData();
	void objSort(bool isAscending);
	bool checkValidity(vec3 no);
	bool findGeometricalRelations(Real* mesh, vec3 no, geometric& geom);
	void calGlobalFrequency(Real** frequency);
	bool calFrequencyTerm(Real** frequency, Real** fl, Real* freqTermX, Real* freqTermY, geometric& geom);
	uint refAS_Flat(vec3 na, Real** frequency, Real* mesh, Real* freqTermX, Real* freqTermY, geometric& geom);
	void refASInner_flat(Real* freqTermX, Real* freqTermY);
	bool refAS_Continuous(uint n, Real* freqTermX, Real* freqTermY);
	bool generateAS(uint SHADING_FLAG);
	bool findNormals(uint SHADING_FLAG);
	bool refToGlobal(Real** frequency, Real** fl, geometric& geom);
	
	bool loadMeshText(const char* fileName);

	void initialize_GPU();
	void generateAS_GPU(uint SHADING_FLAG);
	void refAS_GPU(int idx, int ch, uint SHADING_FLAG);

	// correct the output scale of the  ophGen::conv_fft2 
	void conv_fft2_scale(Complex<Real>* src1, Complex<Real>* src2, Complex<Real>* dst, ivec2 size);
private:

	Real* scaledMeshData;					/// Scaled and shifted mesh array / Data structure : N*9

private:

	//	Inner global parameters
	///	do not need to consider to users
	vec3* no;
	vec3* na;
	vec3* nv;

private:

	//	Inner local parameters
	///	do not need to consider to users

	geometric geom;
	Complex<Real>* refAS;

	// calGlobalFrequency()
	Real dfx, dfy;

	// findNormals()


	// findGeometricalRelations()
	Real mesh_local[9] = { 0.0 };
	Real th, ph;
	Real temp;


	// calFrequencyTerm()
	Real k, kk;
	Real* flxShifted;
	Real* flyShifted;
	Real det;
	Real* invLoRot;


	// refAS_Flat() , refAS_Continuous()

	Complex<Real> refTerm1;
	Complex<Real> refTerm2;
	Complex<Real> refTerm3;



	/// continuous shading
	vec3 av;

	/// occlusion
	Complex<Real>* rearAS;
	Complex<Real>* convol;

	/// random phase
	Complex<Real>* phaseTerm;

	bool is_ViewingWindow;

	/// texture mapping
	Complex<Real>* textFFT;
	Real textFreqX;
	Real textFreqY;
	Complex<Real> refTemp;
	Real* tempFreqTermX;
	Real* tempFreqTermY;
	bool bSinglePrecision;
	uint m_nProgress;

};



#endif
