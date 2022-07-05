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

#include <iostream>
#include <string>
#include <filesystem>
#include <ctime>

#include "ophWRP.h"

namespace fs = std::experimental::filesystem;

int main() {
	// Create ophWRP instance
	ophWRP* Hologram = new ophWRP();

	std::string proj_dir = fs::current_path().generic_string();
	std::string conf_file = proj_dir + "/../dataset/Dice/WRP/Dice_WRP.xml";

	// Read Config Parameters for Point Cloud CGH based WRP algorism
	if (!Hologram->readConfig(conf_file.c_str())) {
		std::cerr << "Fail to load Config file : " << conf_file << std::endl;
		exit(1);
	}

	// Load Point Cloud Data(*.PLY)
	std::string wrp_file = proj_dir + "/../dataset/Dice/WRP/dice_100000.ply";
	if (Hologram->loadPointCloud(wrp_file.c_str()) < 0) {
		std::cerr << "Fail to load WRP file : " << wrp_file << std::endl;
		exit(1);
	}

	// CGH from WRP
	Hologram->generateHologram();

	// Save to ohc(Openholo complex field file format)
	std::string res_path = "./result/";
	if (!fs::exists(res_path)) {
		fs::create_directories(res_path);
	}

	//
	time_t curr_time = time(nullptr);

	struct tm local;
#ifdef _MSC_VER
	localtime_s(&local, &curr_time);
#else
	localtime_r(&curr_time, plocal);
#endif

	char buff[32];
#ifdef _MSC_VER
	sprintf_s(buff, "%4d%02d%02d-%02d%02d%02d",
		local.tm_year + 1900, local.tm_mon + 1, local.tm_mday,
		local.tm_hour, local.tm_min, local.tm_sec);
#else
	sprintf(buff, "%4d%02d%02d-%02d%02d%02d",
		local.tm_year + 1900, local.tm_mon + 1, local.tm_mday,
		local.tm_hour, local.tm_min, local.tm_sec);
#endif

	std::string ohc_file = res_path + "wrp_sample_" + std::string(buff) + ".ohc";
	if (!Hologram->saveAsOhc(ohc_file.c_str())) {
		std::cerr << "Fail to save OHC file : " << ohc_file << std::endl;
		exit(1);
	}

	// Encode Complex Field to Real Field
	Hologram->encoding(Hologram->ENCODE_PHASE);

	// Normalize Real Field to unsigned char(0~255) for save to image(*.BMP)
	Hologram->normalize();

	// Save to bmp
	std::string bmp_file = res_path + "wrp_sample_" + std::string(buff) + ".bmp";
	if (!Hologram->save(bmp_file.c_str())) {
		std::cerr << "Fail to save BMP file : " << bmp_file << std::endl;
		exit(1);
	}

	// Release memory used to Generate WRP
	Hologram->release();

	return 0;
}
