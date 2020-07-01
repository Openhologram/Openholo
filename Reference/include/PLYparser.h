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

#ifndef __PLY_PARSER_H__
#define __PLY_PARSER_H__

#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
#include <typeinfo>
#include "include.h"


/* PLY File Header for Openholo Point Cloud Format
ply
format ascii 1.0
comment Openholo Point Cloud Format
element color 1
property int channel
element vertex n_vertices
property Real x
property Real y
property Real z
property uchar red		#diffuse_red
property uchar green	#diffuse_green
property uchar blue		#diffuse_blue
property Real phase
end_header
*/

/* PLY File Header for Openholo Triangle Mesh Format
ply
format ascii 1.0
comment Openholo Triangle Mesh Format
element color 1
property int channel
element vertex n_vertices
property uint face_idx
property Real x
property Real y
property Real z
property uchar red		#diffuse_red
property uchar green	#diffuse_green
property uchar blue		#diffuse_blue
end_header
*/


class __declspec(dllexport) PLYparser {
public:
	PLYparser();
	~PLYparser();

private:
	enum class Type {
		INVALID,
		INT8,
		UINT8,
		INT16,
		UINT16,
		INT32,
		UINT32,
		FLOAT32,
		FLOAT64
	};

	//pair<int : stride, string : str>
	std::map<Type, std::pair<int, std::string> > PropertyTable;

	struct PlyProperty {
		std::string name;
		Type propertyType;
		bool isList = false;
		Type listType = Type::INVALID;
		longlong listCount = 0;

		PlyProperty(std::istream &is);
		PlyProperty(const Type type, const std::string &_name);
		PlyProperty(const Type list_type, const Type prop_type, const std::string &_name, const ulonglong list_count);
	};

	struct PlyElement {
		std::string name;
		longlong size;
		std::vector<PLYparser::PlyProperty> properties;

		PlyElement(std::istream &is);
		PlyElement(const std::string &_name, const ulonglong count);
	};	

	static Type propertyTypeFromString(const std::string &t);
	
	bool findIdxOfPropertiesAndElement(
		const std::vector<PlyElement> &elements,
		const std::string &elementKey,
		const std::string &propertyKeys,
		longlong &elementIdx,
		int &propertyIdx);
	
public:
	bool loadPLY(					// for Point Cloud Data
		const std::string& fileName,
		ulonglong &n_points,
		int &color_channels, //1 or 3 (If it is 4, Alpha channel is not loaded.)
		Real** vertexArray,
		Real** colorArray,
		Real** phaseArray, //If isPhaseParse is false, PhaseArray is nullptr
		bool &isPhaseParse);

	bool savePLY(					
		const std::string& fileName,
		const ulonglong n_points,
		const int color_channels,
		Real* vertexArray,
		Real* colorArray,
		Real* phaseArray);

	bool loadPLY(					// for Triangle Mesh Data
		const char* fileName,
		ulonglong &n_vertices,
		int &color_channels,
		uint** face_idx,
		Real** vertexArray,
		Real** colorArray);

	bool savePLY(
		const char* fileName,
		const ulonglong n_vertices,
		const int color_channels,
		uint* face_idx,
		Real* vertexArray,
		Real* colorArray);
};


#endif