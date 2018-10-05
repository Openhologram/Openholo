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
#pragma once
#ifndef _OphCascadedPropagation_h
#define _OphCascadedPropagation_h

#include "ophDis.h"

struct OphCascadedPropagationConfig {
	OphCascadedPropagationConfig()
		: num_colors(0),
		wavelengths{ 0.0, 0.0, 0.0 },
		dx(0.0),
		dy(0.0),
		nx(0),
		ny(0),
		field_lens_focal_length(0.0),
		dist_reconstruction_plane_to_pupil(0.0),
		dist_pupil_to_retina(0.0),
		pupil_diameter(0.0),
		nor(0.0)
		{}

	/**
	* @param number of colors
	*/
	oph::uint num_colors;

	/**
	* @param wavelengths in meter
	*/
	oph::vec3 wavelengths;

	/**
	* @param horizontal pixel pitch in meter
	*/
	Real dx;

	/**
	* @param vertical pixel pitch in meter
	*/
	Real dy;

	/**
	* @param horizontal resolution in meter
	*/
	oph::uint nx;

	/**
	* @param vertical resolution in meter
	*/
	oph::uint ny;

	/**
	* @param focal length of field lens in meter
	*/
	Real field_lens_focal_length;

	/**
	* @param distance from reconstruction plane to pupil plane in meter
	*/
	Real dist_reconstruction_plane_to_pupil;

	/**
	* @param distance from pupil plane to retina plane
	*/
	Real dist_pupil_to_retina;

	/**
	* @param pupil diameter
	*/
	Real pupil_diameter;

	/**
	* @param scaling term for output intensity
	*/
	Real nor;
};


#ifdef DISP_EXPORT
#define DISP_DLL __declspec(dllexport)
#else
#define DISP_DLL __declspec(dllimport)
#endif



/**
* @ingroup casprop
* @brief Cascaded propagation module
* @detail
* @author Seunghyup Shin
*/
class DISP_DLL ophCascadedPropagation : public ophDis {
	private:
		/**
		* @brief Constructor (later use)
		*/
		ophCascadedPropagation();

	public:
		/**
		* @brief Constructor
		* @param configfilepath: absolute or relative path of configuration file
		*/
		ophCascadedPropagation(const wchar_t* configfilepath);

		/**
		* @brief Destructor
		*/
		~ophCascadedPropagation();

		/**
		* @brief Do cascaded propagation
		* @return true if successful
		* @return false when failed
		*/
		bool propagate();

		/**
		* @brief Save wavefield at retina plane as Windows Bitmap file
		* @param pathname: absolute or relative path of output file
		* @param bitsperpixel: number of bits per pixel
		* @return true if successfully saved
		* @return false when failed
		*/
		bool save(const wchar_t* pathname, uint8_t bitsperpixel);


	private:
		/**
		* @param config_: configuration parameters for cascaded propagation
		*/
		OphCascadedPropagationConfig config_;

		/**
		* @param wavefield_SLM: wavefield data at SLM plane
		*/
		vector<oph::Complex<Real>*> wavefield_SLM;

		/**
		* @param wavefield_SLM: wavefield data at pupil plane
		*/
		vector<oph::Complex<Real>*> wavefield_pupil;

		/**
		* @param wavefield_SLM: wavefield data at retina plane
		*/
		vector<oph::Complex<Real>*> wavefield_retina;

		/**
		* @param ready_to_propagate: indicates if configurations and input wavefield are all loaded succesfully
		*/
		bool ready_to_propagate;

		/**
		* @param hologram_path: absolute or relative path of input wavefield file
		*/
		wstring hologram_path;

	private:
		/**
		* @brief Reads configurations from XML file
		* @return true if successful
		* @return false when failed
		*/
		bool readConfig(const wchar_t* fname);

		/**
		* @brief Calculates 1st propagation (from SLM plane to pupil plane)
		* @return true if successful
		* @return false when failed
		*/
		bool propagateSlmToPupil();

		/**
		* @brief Calculates 2nd propagation (from pupil plane to retina plane)
		* @return true if successful
		* @return false when failed
		*/
		bool propagatePupilToRetina();

		/**
		* @brief Allocates memory according to configuration setup
		* @return true if successful
		* @return false when failed
		*/
		bool allocateMem();

		/**
		* @brief Deallocates memory
		*/
		void deallocateMem();

		/**
		* @brief Loads wavefield data from input file
		* @return true if successful
		* @return false when failed
		*/
		bool loadInput();

		/**
		* @brief Generates intensity fields from complex wavefields
		* @details each output color channel is in 8-bits
		* @param wavefields: vector of monochromatic complex wavefields
		* @return pointer to color-interleaved intensity sequence
		* @return nullptr if failed
		*/
		oph::uchar* getIntensityfields(vector<oph::Complex<Real>*> wavefields);


	public:
		/**
		* @brief Returns number of colors
		*/
		oph::uint getNumColors() { return config_.num_colors; }

		/**
		* @brief Returns wavelengths in meter
		*/
		oph::vec3 getWavelengths() { return config_.wavelengths; }

		/**
		* @brief Returns horizontal pixel pitch in meter
		*/
		Real getPixelPitchX() { return config_.dx; }

		/**
		* @brief Returns vertical pixel pitch in meter
		*/
		Real getPixelPitchY() { return config_.dy; }

		/**
		* @brief Returns horizontal resolution
		*/
		oph::uint getResX() { return config_.nx; }

		/**
		* @brief Returns vertical resolution
		*/
		oph::uint getResY() { return config_.ny; }

		/**
		* @brief Returns focal length of field lens in meter
		*/
		Real getFieldLensFocalLength() { return config_.field_lens_focal_length; }

		/**
		* @brief Returns distance from reconstruction plane to pupil plane in meter
		*/
		Real getDistObjectToPupil() { return config_.dist_reconstruction_plane_to_pupil; }

		/**
		* @brief Returns distance from pupil plane to retina plane in meter
		*/
		Real getDistPupilToRetina() { return config_.dist_pupil_to_retina; }

		/**
		* @brief Returns diameter of pupil in meter
		*/
		Real getPupilDiameter() { return config_.pupil_diameter; }

		/**
		* @brief Returns \a Nor, which affects the range of output intensity
		* @details \a Nor is NOT intuitive at all and should be changed sometime
		*/
		Real getNor() { return config_.nor; }

		/**
		* @brief Return monochromatic wavefield at SLM plane
		*/
		oph::Complex<Real>* getSlmWavefield(oph::uint id);

		/**
		* @brief Return monochromatic wavefield at pupil plane
		*/
		oph::Complex<Real>* getPupilWavefield(oph::uint id);

		/**
		* @brief Return monochromatic wavefield at retina plane
		*/
		oph::Complex<Real>* getRetinaWavefield(oph::uint id);

		/**
		* @brief Return all wavefields at retina plane
		*/
		vector<oph::Complex<Real>*> getRetinaWavefieldAll();


		// setters
		//virtual bool SetSlmWavefield(Complex<Real>* srcHologram) = 0; // set input wavefield (for later use)
		//virtual bool SetSlmWavefield(ophGen& srcHologram) = 0; // set input wavefield (for later use)


	protected:
		/**
		* @brief Pure virtual function for override in child classes
		*/
		virtual void ophFree(void);
};



// utilities
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#define PRINT_ERROR(errorMsg)           { cout << "Error(" << __FILENAME__ << ":" << __LINE__ << "): " << ( errorMsg ) << endl; }

#endif