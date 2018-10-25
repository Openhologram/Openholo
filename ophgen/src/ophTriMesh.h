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
* @addtogroup mesh
//@{
* @detail

* @section Introduction

Triangular mesh based CGH generates the complex field of 3D objects represented as a collection of the triangular meshes.
The algorithm aggregates the angular spectrums of individual triangular meshes and then performs a Fourier transform to obtain the complex field for entire objects.

![](@ref pics/ophgen/mesh/mesh_fig3.png)

The angular spectrum of the individual triangular mesh is obtained using the analytic formula of the Fourier transform of the reference triangular aperture, considering the geometrical relation between the hologram plane and the local mesh plane, and also between the local mesh and the reference triangular aperture.

![](@ref pics/ophgen/mesh/mesh_fig1.png)
![](@ref pics/ophgen/mesh/mesh_fig2.png)

The phase distribution on the mesh is determined by the carrier wave is assumed to be a plane wave of a specfic direction in the code.
The amplitude inside each mesh is determined by the surface shading model and it can be either linearly varying for the continuous shading or uniform for the flat shading.

![continuous shading](@ref pics/ophgen/mesh/mesh_ex_continuous.png)
![flat shading](@ref pics/ophgen/mesh/mesh_ex_flat.png)

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

	Real* triMeshArray;						/// Original triangular mesh array (N*9)
	Complex<Real>* angularSpectrum;			/// Angular spectrum of the hologram
	OphMeshData* meshData;					/// OphMeshData type data structure pointer

private:

	Real objSize;							/// Object maximum of width and height / unit :[m]
	vec3 objShift;							/// Object shift value / Data structure - [shiftX, shiftY, shiftZ] / unit : [m]

	Real carrierWave[3] = { 0,0,1 };		/// Carrier wave direction / default : {0, 0, 1}

	vec3 illumination;						/// Position of the light source (for shading effect) / No-illumination : {0, 0, 0}
	int SHADING_TYPE;						/// SHADING_FLAT, SHADING_CONTINUOUS

public:
	void setObjSize(Real in) { objSize = in; }
	void setObjShift(vec3 in) { objShift[_X] = in[_X]; objShift[_Y] = in[_Y]; objShift[_Z] = in[_Z]; }
	void setObjShift(Real in[]) { objShift[_X] = in[_X]; objShift[_Y] = in[_Y]; objShift[_Z] = in[_Z]; }
	void setObjShift(vector<Real> in) { objShift[_X] = in[_X]; objShift[_Y] = in[_Y]; objShift[_Z] = in[_Z]; }
	void setCarrierWave(Real in1, Real in2, Real in3) { carrierWave[_X] = in1; carrierWave[_Y] = in2; carrierWave[_Z] = in3; }
	void setIllumination(vec3 in) { illumination = in; }
	void setIllumination(Real inx, Real iny, Real inz) { illumination = { inx, iny, inz }; }
	void setShadingType(int in) { SHADING_TYPE = in; }
	ulonglong getNumMesh() { return meshData->n_faces; }
	Real* getMeshData() { return triMeshArray; }
	Complex<Real>* getAngularSpectrum() { return angularSpectrum; }
	Real* getScaledMeshData() {	return scaledMeshData; }

	const Real& getObjSize(void) { return objSize; }
	const vec3& getObjShift(void) { return objShift; }
	const vec3&	getIllumination(void) { return illumination; }


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
	bool loadMeshData(const char* fileName, const char* ext);

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
	
	/**
	* @brief	Wave carry
	* @param	Real	carryingAngleX		Wave carrying angle in horizontal direction
	* @param	Real	carryingAngleY		Wave carrying angle in vertical direction
	*/
	void waveCarry(Real carryingAngleX, Real carryingAngleY);

private:
	
	// Inner functions
	/// not used for users
	
	void initializeAS();
	void objNormCenter();
	uint checkValidity(Real* mesh, vec3 no);
	uint findGeometricalRelations(Real* mesh, vec3 no);
	void calGlobalFrequency();
	uint calFrequencyTerm();
	uint refAS_Flat(vec3 na);
	uint refAS_Continuous(uint n);
	void randPhaseDist(Complex<Real>* AS);
	void generateAS(uint SHADING_FLAG);
	uint findNormals(uint SHADING_FLAG);
	uint refToGlobal();

	uint loadMeshText(const char* fileName);
private:

	Real* normalizedMeshData;				/// Normalized mesh array / Data structure : N*9
	Real* scaledMeshData;					/// Scaled and shifted mesh array / Data structure : N*9

private:

	//	Inner global parameters
	///	do not need to consider to users

	Real refTri[9] = { 0,0,0,1,1,0,1,0,0 };
	Real* fx;
	Real* fy;
	Real* fz;
	vec3* no;
	vec3* na;
	vec3* nv;

private:

	//	Inner local parameters
	///	do not need to consider to users

	vec3 n;
	Real shadingFactor;
	geometric geom;
	Real* mesh_local;
	Real* flx;
	Real* fly;
	Real* flz;
	Real* freqTermX;
	Real* freqTermY;
	Complex<Real>* refAS;

	Complex<Real> refTerm1;
	Complex<Real> refTerm2;
	Complex<Real> refTerm3;

	Complex<Real> D1;
	Complex<Real> D2;
	Complex<Real> D3;

	Complex<Real>* ASTerm;
	Complex<Real>* randTerm;
	Complex<Real>* phaseTerm;
	Complex<Real>* convol;

};



#endif