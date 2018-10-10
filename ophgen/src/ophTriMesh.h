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

	Real* triMeshArray;						/// Original triangular mesh array (N*9)
	Complex<Real>* angularSpectrum;			/// Angular spectrum of the hologram
	OphMeshData* meshData;					/// OphMeshData type data structure pointer

private:

	Real objSize;							/// Object maximum of width and height / unit :[m]
	Real objShift[3];						/// Object shift value / Data structure - [shiftX, shiftY, shiftZ] / unit : [m]

	Real carrierWave[3] = { 0,0,1 };		/// Carrier wave direction / default : {0, 0, 1}

	vec3 illumination;						/// Position of the light source (for shading effect) / No-illumination : {0, 0, 0}
	int SHADING_TYPE;						/// SHADING_FLAT, SHADING_CONTINUOUS

public:
	/** \ingroup getter/setter */
	void setObjSize(Real in) { objSize = in; }
	/** \ingroup getter/setter */
	void setObjShift(Real in[]) { objShift[_X] = in[_X]; objShift[_Y] = in[_Y]; objShift[_Z] = in[_Z]; }
	/** \ingroup getter/setter */
	void setObjShift(vector<Real> in) { objShift[_X] = in[_X]; objShift[_Y] = in[_Y]; objShift[_Z] = in[_Z]; }
	/** \ingroup getter/setter */
	void setCarrierWave(Real in1, Real in2, Real in3) { carrierWave[_X] = in1; carrierWave[_Y] = in2; carrierWave[_Z] = in3; }
	/** \ingroup getter/setter */
	void setIllumination(vec3 in) { illumination = in; }
	/** \ingroup getter/setter */
	void setIllumination(Real inx, Real iny, Real inz) { illumination = { inx, iny, inz }; }
	/** \ingroup getter/setter */
	void setShadingType(int in) { SHADING_TYPE = in; }
	/** \ingroup getter/setter */
	ulonglong getNumMesh() { return meshData->n_faces; }
	/** \ingroup getter/setter */
	Real* getMeshData() { return triMeshArray; }
	/** \ingroup getter/setter */
	Complex<Real>* getAngularSpectrum() { return angularSpectrum; }
	/** \ingroup getter/setter */
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