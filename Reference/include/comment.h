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
* @mainpage Openholo library Documentation
* @section Introduction

OpenHolo is an open source library which contains algorithms and their software implementation
for generation of holograms to be applied in various fields. The goal behind the library development
is facilitating production of digital holographic contents and expanding the area of their application.
The developed by us open source library is a tool for computer generation of holograms, simulations and
signal processing at various formats of 3D input data and properties of the 3D displays. Based on this,
we want to lay the foundation for commercializing digital holographic service in various fields.


* @section Examples

Generation Hologram - Point Cloud Example

: Implementation of the hologram generation method using point cloud data.

@code
	#include "ophPointCloud.h"

	ophPointCloud* Hologram = new ophPointCloud();									// Create ophPointCloud instance

	Hologram->readConfig("config/TestSpecPointCloud.xml");							// Read Config Parameters for Point Cloud CGH
	Hologram->loadPointCloud("source/PointCloud/TestPointCloud_Plane.ply");			// Load Point Cloud Data(*.PLY)

	Hologram->setMode(MODE_GPU);													// Select CPU or GPU Processing

	Hologram->generateHologram(PC_DIFF_RS);											// Select R-S diffraction or Fresnel diffraction

	Hologram->encodeHologram();														// Encode Complex Field to Real Field
	Hologram->normalize();															// Normalize Real Field to unsigned char(0~255) for save to image(*.BMP)

	Hologram->save("result/PointCloud/Result_PointCloudSample_Plane.bmp");			// Save to bmp

	Hologram->release();															// Release memory used to Generate Point Cloud
@endcode

![PointCloud based CGH Example](@ref pics/ophgen/pointcloud/pointcloud_example01.png)	


Generation Hologram - Depth Map Example.

: Implementation of the hologram generation method using depth map data.

@code
	#include "ophDepthMap.h"

	ophDepthMap* Hologram = new ophDepthMap();										// Create ophDepthMap instance

	Hologram->readConfig("config/TestSpecDepthMap.xml");							// Read Config Parameters for Depth Map CGH
	Hologram->readImageDepth("source/DepthMap", "RGB_D", "D_D");					// Load Depth and RGB image

	Hologram->setMode(MODE_GPU); //Select CPU or GPU Processing						// Select CPU or GPU Processing

	Hologram->generateHologram();													// CGH by depth map

	Hologram->encodeHologram();														// Encode Complex Field to Real Field
	Hologram->normalize();															// Normalize Real Field to unsigned char(0~255) for save to image(*.BMP)

	Hologram->save("result/DepthMap/Result_DepthmapSample.bmp");					// Save to bmp

	Hologram->release();															// Release memory used to Generate DepthMap
@endcode

![DepthMap based CGH Example](@ref pics/ophgen/depthmap/depthmap_example01.png)


Generation Hologram - Triangle Mesh Example

@code
	#include "ophTriMesh.h"

		ophTri* Hologram = new ophTri();

		// Load
		Hologram->readMeshConfig("config/TestSpecMesh.xml");						// Read the Mesh hologram configuration file
		Hologram->loadMeshData("source/TriMesh/mesh_teapot.ply", "ply");			// Read the Meshed object data
		Hologram->objScaleShift();													// Object scaling and shifting

		// Generate
		Hologram->generateMeshHologram(Hologram->SHADING_FLAT);						// Generate the hologram
			/// Put the shading effect type

		// Save as Complex Field Data
		Hologram->saveAsOhc("result/TriMesh/Mesh_complexField.ohc");				// Save the hologram complex field data

		// Encode
		Hologram->encoding(Hologram->ENCODE_SIMPLENI);								// Encode the hologram

		// Save as Encoded Image
		Hologram->normalizeEncoded();												// Normalize the encoded hologram to generate image file
		ivec2 encode_size = Hologram->getEncodeSize();								// Get encoded hologram size
		Hologram->save("result/TriMesh/Mesh_0.1m_ni_-0.3deg.bmp",
			8, nullptr, encode_size[_X], encode_size[_Y]);							// Save the encoded hologram image

		Hologram->release();														// Release memory used to Generate Triangle Mesh
@endcode

![Triangle Mesh based CGH Example](@ref pics/ophgen/mesh/result_mesh_01.png)


Generation Hologram - Light Field Example

@code
	#include "ophLightField.h"

		ophLF* Hologram = new ophLF();

		// Load
		Hologram->readLFConfig("config/TestSpecLF.xml");							// Read the LF hologram configuration file
		Hologram->loadLF("source/LightField/sample_orthographic_images_5x5", "bmp");// Load the Light field source image files
			/// Put the directory which has the source files and Put the image file type

		// Generate
		Hologram->generateHologram();												// Generate the hologram

		// Save as Complex field data
		Hologram->saveAsOhc("result/LightField/LF_complexField.ohc");				// Save the hologram complex field data

		// Encode
		Hologram->encoding(Hologram->ENCODE_SIMPLENI);								// Encode the hologram

		// Save as Encoded Image
		Hologram->normalizeEncoded();												// Normalize the encoded hologram to generate image file
		ivec2 encode_size = Hologram->getEncodeSize();								// Get encoded hologram size
		Hologram->save("result/LightField/Light_Field_NI_carrier.bmp",
			8, nullptr, encode_size[_X], encode_size[_Y]);							// Save the encoded hologram image

		Hologram->release();														// Release memory used to Generate Light Field
@endcode

![LightField based CGH Example](@ref pics/ophgen/lightfield/result_lightfield_01.png)


Generation Hologram - Wavefront Recording Plane(WRP) Example

@code
	#include "ophWRP.h"

	ophWRP* Hologram = new ophWRP();

	Hologram->readConfig("config/TestSpecWRP.xml");
	Hologram->loadPointCloud("source/WRP/TestPointCloud_WRP.ply");

	Hologram->calculateWRP();
	Hologram->generateHologram();

	Hologram->encodeHologram();
	Hologram->normalize();

	Hologram->save("result/WRP/Result_WRP.bmp");

	Hologram->release();
@endcode


Encoding Example

@code
	#include "ophPointCloud.h"

	ophPointCloud* Hologram = new ophPointCloud();

	Hologram->loadComplex("source/Encoding/teapot_real_1920,1080.txt", "source/Encoding/teapot_imag_1920,1080.txt", 1920, 1080);

	Hologram->encoding(ophGen::ENCODE_AMPLITUDE);
	Hologram->normalizeEncoded();

	ivec2 encode_size = Hologram->getEncodeSize();
	Hologram->save("result/Encoding/Encoding.bmp", 8, nullptr, encode_size[_X], encode_size[_Y]);
@endcode


Wave Aberration Example

@code
	#include "ophWaveAberration.h"

	ophWaveAberration* wa = new ophWaveAberration;

	wa->readConfig("config/TestSpecAberration.xml");			// reads parameters from a configuration file
	wa->accumulateZernikePolynomial();							// generates 2D complex data array of wave aberration according to parameters
	wa->complex_W;												// double pointer variable of 2D complex data array of wave aberration
	wa->resolutionX;											// resolution in x axis of 2D complex data array of wave aberration
	wa->resolutionY;											// resolution in y axis of 2D complex data array of wave aberration
	wa->saveAberration("result/WaveAberration/aberration.bin"); // saves 2D complex data array of complex wave aberration into a file

	wa->readAberration("result/WaveAberration/aberration.bin"); // reads 2D complex data array of complex wave aberration from a file
	wa->complex_W;												// double pointer variable of 2D complex data array of wave aberration
	wa->resolutionX;											// resolution in x axis of 2D complex data array of wave aberration
	wa->resolutionY;											// resolution in y axis of 2D complex data array of wave aberration

	wa->release();
@endcode


Cascaded Propagation Example

@code
	#include "ophCascadedPropagation.h"

	ophCascadedPropagation* pCp = new ophCascadedPropagation(L"config/TestSpecCascadedPropagation.xml");
	if (pCp->propagate())
		pCp->saveIntensityAsImg(L"result/CascadedPropagation/intensityRGB.bmp", pCp->getNumColors() * 8);

	pCp->release();
@endcode


Hologram signal processing - Off-axis hologram transform Example

@code
	#include "ophSig.h"

	ophSig *holo = new ophSig();

	if (!holo->readConfig("config/holoParam.xml")) {
		// no file
		return false;
	}

	if (!holo->load("source/OffAxis/3_point_re.bmp", "source/OffAxis/3_point_im.bmp", 8)) {
		// no file
		return false;
	}

	holo->sigConvertOffaxis();

	holo->save("result/OffAxis/Off_axis.bmp", 8);

	holo->release();
@endcode


Hologram signal processing - CAC transform Example

@code
	#include "ophSig.h"

	ophSig *holo = new ophSig();

	if (!holo->readConfig("config/holoParam.xml")) {
		// no file
		return false;
	}

	if (!holo->load("source/CAC/ColorPoint_re.bmp", "source/CAC/ColorPoint_im.bmp", 24)) {
		// no file
		return false;
	}

	holo->sigConvertCAC(0.000000633, 0.000000532, 0.000000473);

	holo->save("result/CAC/CAC_re_C.bin", "result/CAC/CAC_im_C.bin", 24);

	holo->release();
@endcode


Hologram signal processing - HPO transform Example

@code
	#include "ophSig.h"

	ophSig *holo = new ophSig();

	if (!holo->readConfig("config/holoParam.xml")) {
		// no file
		return false;
	}

	if (!holo->load("source/HPO/3_point_re.bmp", "source/HPO/3_point_im.bmp", 8)) {
		// no file
		return false;
	}

	holo->sigConvertHPO();

	holo->save("result/HPO/HPO_re_C.bmp", "result/HPO/HPO_im_C.bmp", 8);

	holo->release();
@endcode


Hologram signal processing - get parameter using axis transformation Example

@code
	#include "ophSig.h"

	ophSig* holo = new ophSig();

	float depth = 0;

	if (!holo->readConfig("config/holoParam.xml")) {
		// no file
		return false;
	}

	if (!holo->load("source/AT/0.1point_re.bmp", "source/AT/0.1point_im.bmp", 8)) {
		// no file
		return false;
	}

	depth = holo->sigGetParamAT();
	std::cout << depth << endl;
	// backpropagation
	holo->propagationHolo(-depth);

	holo->save("result/AT/AT_re.bmp", "result/AT/AT_im.bmp", 8);

	holo->release();
@endcode


Hologram signal processing - get parameter using SF Example

@code
	#include "ophSig.h"

	ophSig* holo = new ophSig();

	float depth = 0;

	if (!holo->readConfig("config/holoParam.xml")) {
		// no file
		return false;
	}

	if (!holo->load("source/SF/3_point_re.bmp", "source/SF/3_point_im.bmp", 8)) {
		// no file
		return false;
	}

	depth = holo->sigGetParamSF(10, -10, 100, 0.3);
	std::cout << depth << endl;
	// backpropagation
	holo->propagationHolo(depth);

	holo->save("result/SF/SF_re.bmp", "result/SF/SF_im.bmp", 8);
	holo->release();
@endcode


Hologram signal processing - get parameter using Phase Shift Digital Hologram Example

@code
	#include "ophSig.h"

	ophSig *holo = new ophSig();

	const char *f0 = "source/PhaseShiftedHolograms/0930_005_gray.bmp";
	const char *f90 = "source/PhaseShiftedHolograms/0930_006_gray.bmp";
	const char *f180 = "source/PhaseShiftedHolograms/0930_007_gray.bmp";
	const char *f270 = "source/PhaseShiftedHolograms/0930_008_gray.bmp";

	holo->getComplexHFromPSDH(f0, f90, f180, f270);

	holo->save("result/PhaseShift/PSDH_re_C.bmp", "result/PhaseShift/PSDH_im_C.bmp", 8);

	holo->release();
@endcode


Hologram signal processing - get parameter using Phase Unwrapping Example

@code
	#include "ophSigPU.h"

	ophSigPU *holo = new ophSigPU;

	if (!holo->loadPhaseOriginal("source/PhaseUnwrapping/phase_unwrapping_example.bmp", 8)) {
		return false;
	}
	int maxBoxRadius = 4;
	holo->setPUparam(maxBoxRadius);

	holo->runPU();

	holo->savePhaseUnwrapped("result/PhaseUnwrapping/PU_Test.bmp");

	holo->release();
@endcode


Hologram signal processing - get parameter using Compressive Holography Example

@code
	#include "ophSigCH.h"

	ophSigCH *holo = new ophSigCH;

	if (!holo->readConfig("config/TestSpecCH.xml")) {
		return false;
	}

	if (!holo->loadCHtemp("source/CompressiveHolo/sampleComplexH_re.bmp", "source/CompressiveHolo/sampleComplexH_im.bmp", 8)) {
		return false;
	}

	holo->runCH(0);

	holo->saveNumRec("result/CompressiveHolo/CH_Test.bmp");

	holo->release();
@endcode



*
*/


/**
* @defgroup oph Openholo

* @defgroup gen Generation
* @ingroup oph

* @defgroup pointcloud Point Cloud
* @ingroup gen
* @defgroup depthmap Depth Map
* @ingroup gen
* @defgroup lightfield LightField
* @ingroup gen
* @defgroup mesh Triangle Mesh
* @ingroup gen
* @defgroup wrp WRP
* @ingroup gen

* @defgroup rec Reconstruction
* @ingroup oph
* @defgroup dis Display
* @ingroup rec

* @defgroup waveaberr Wave Aberration
* @ingroup dis
* @defgroup casprop Cacaded Propagation
* @ingroup dis
* @defgroup partial Partial Coherence
* @ingroup dis

* @defgroup sig Signal Processing
* @ingroup oph

* @defgroup offaxis Off-Axis
* @ingroup sig
* @defgroup convCAC Convert-CAC
* @ingroup sig
* @defgroup convHPO Convert-HPO
* @ingroup sig
* @defgroup getAT Get Parameter-AT
* @ingroup sig
* @defgroup getSF Get Parameter-SF
* @ingroup sig
* @defgroup PSDH Phase Shifting
* @ingroup sig
* @defgroup PU Phase Unwrapping
* @ingroup sig
* @defgroup CH Compressive Holography
* @ingroup sig
*/