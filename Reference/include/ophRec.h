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

#ifndef __OphReconstruction_h
#define __OphReconstruction_h

#include "Openholo.h"

#ifdef RECON_EXPORT
#define RECON_DLL __declspec(dllexport)
#else
#define RECON_DLL __declspec(dllimport)
#endif


struct RECON_DLL OphRecConfig
{
	Real EyeLength;
	Real EyePupilDiaMeter;
	Real EyeBoxSizeScale;
	vec2 EyeBoxSize;
	int EyeBoxUnit;
	vec3 EyeCenter;
	Real EyeFocusDistance;
	Real ResultSizeScale;
	Real SimulationTo;
	Real SimulationFrom;
	int SimulationStep;
	int SimulationMode;
	Real RatioAtRetina;
	Real RatioAtPupil;
	bool CreatePupilFieldImg;
	bool CenteringRetinaImg;
	bool ViewingWindow;
	bool SimulationPos[3];
};

/**
* @ingroup rec
* @brief
* @author
*/
class RECON_DLL ophRec : public Openholo
{
public:
	/**
	* @brief Constructor
	*/
	explicit ophRec(void);

protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophRec(void);


private:
	vector<Real *>			m_vecEncoded;
	vector<uchar *>			m_vecNormalized;

	std::vector<Complex<Real>*>	field_set_;
	std::vector<double*>	field_ret_set_;

	std::vector<double*>	res_set_;
	std::vector<double*>	res_set_norm_255_;

	std::vector<ivec2>		pn_set_;						// Pixel number of output plane
	std::vector<vec2>		pp_set_;						// Pixel pitch of output plane
	std::vector<ivec2>		pn_ret_set_;
	std::vector<vec2>		pp_ret_set_;
	std::vector<vec2>		ss_ret_set_;

	std::vector<Real*>		recon_set;
	std::vector<uchar*>		img_set;
	std::vector<ivec2>		img_size;
	std::vector<Real*>		focus_recon_set;
	std::vector<uchar*>		focus_img_set;
	std::vector<ivec2>		focus_img_size;

	int						m_oldSimStep;


	OphRecConfig			rec_config;
	int						m_nOldChannel;
	int						m_idx;
	unsigned int			m_mode;

protected:
	/**
	* @brief Pure virtual function for override in child classes
	*/
	virtual void ophFree(void);
	void Clear();
	void GetPupilFieldFromHologram();
	void GetPupilFieldFromVWHologram();
	void Propagation_Fresnel_FFT(int chnum);
	void ASM_Propagation();
	void ASM_Propagation_GPU();
	void GetPupilFieldImage(Complex<Real>* src, double* dst, int pnx, int pny, double ppx, double ppy, double scaleX, double scaleY);
	void getVarname(int vtr, vec3& var_vals, std::string& varname2);
public:
	void SaveImage(const char* path, const char* ext = "bmp");
	void setConfig(OphRecConfig config) { rec_config = config; }
	void SetMode(unsigned int mode) { m_mode = mode; }
	OphRecConfig& getConfig() { return rec_config; }
	bool ReconstructImage();
	bool readConfig(const char* fname);
	bool readImage(const char* path);
	bool readImagePNA(const char* phase, const char* amplitude);
	bool readImageRNI(const char* real, const char* imaginary);
	void Perform_Simulation();
	void Initialize();
	bool save(const char * fname, uint8_t bitsperpixel, uchar* src, uint px, uint py);
	template<typename T>
	void normalize(T* src, uchar* dst, int x, int y);
	template<typename T>
	void normalize(T* src, uchar* dst, int x, int y, T max, T min);

};


#endif // !__OphReconstruction_h