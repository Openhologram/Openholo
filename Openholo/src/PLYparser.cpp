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

#include "PLYparser.h"
#include "sys.h"
#include <typeinfo>

PLYparser::PLYparser()
{
	PropertyTable.insert(std::make_pair(Type::INT8, std::make_pair(1, "char")));
	PropertyTable.insert(std::make_pair(Type::UINT8, std::make_pair(1, "uchar")));
	PropertyTable.insert(std::make_pair(Type::INT16, std::make_pair(2, "short")));
	PropertyTable.insert(std::make_pair(Type::UINT16, std::make_pair(2, "ushort")));
	PropertyTable.insert(std::make_pair(Type::INT32, std::make_pair(4, "int")));
	PropertyTable.insert(std::make_pair(Type::UINT32, std::make_pair(4, "uint")));
	PropertyTable.insert(std::make_pair(Type::FLOAT32, std::make_pair(4, "float")));
	PropertyTable.insert(std::make_pair(Type::FLOAT64, std::make_pair(8, "double")));
	PropertyTable.insert(std::make_pair(Type::INVALID, std::make_pair(0, "INVALID")));
}

PLYparser::~PLYparser()
{
}

PLYparser::PlyProperty::PlyProperty(std::istream &is)
	: isList(false)
{
	std::string type;
	is >> type;
	if (type == "list") {
		std::string countType;
		is >> countType >> type;
		listType = PLYparser::propertyTypeFromString(countType);
		isList = true;
	}
	propertyType = PLYparser::propertyTypeFromString(type);
	is >> name;
}

PLYparser::PlyProperty::PlyProperty(const Type type, const std::string &_name)
	: propertyType(type), name(_name)
{
}

PLYparser::PlyProperty::PlyProperty(const Type list_type, const Type prop_type, const std::string &_name, const ulonglong list_count)
	: listType(list_type), propertyType(prop_type), name(_name), listCount(list_count), isList(true)
{
}

PLYparser::PlyElement::PlyElement(std::istream &is)
{
	is >> name >> size;
}

PLYparser::PlyElement::PlyElement(const std::string &_name, const ulonglong count)
	: name(_name), size(count)
{
}

PLYparser::Type PLYparser::propertyTypeFromString(const std::string &t)
{
	if (t == "int8" || t == "char")								return Type::INT8;
	else if (t == "uint8" || t == "uchar")						return Type::UINT8;
	else if (t == "int16" || t == "short")						return Type::INT16;
	else if (t == "uint16" || t == "ushort")					return Type::UINT16;
	else if (t == "int32" || t == "int")						return Type::INT32;
	else if (t == "uint32" || t == "uint")						return Type::UINT32;
	else if (t == "float32" || t == "float")					return Type::FLOAT32;
	else if (t == "float64" || t == "double" || t == "real")	return Type::FLOAT64;
	else														return Type::INVALID;
}

bool PLYparser::findIdxOfPropertiesAndElement(const std::vector<PlyElement> &elements, const std::string &elementKey, const std::string &propertyKeys, longlong &elementIdx, int &propertyIdx)
{
	elementIdx = -1;
	for (size_t i = 0; i < elements.size(); ++i) {
		if (elements[i].name == elementKey) {
			elementIdx = i;
			break;
		}
	}

	if (elementIdx >= 0) {
		const PlyElement &element = elements[elementIdx];

		propertyIdx = -1;
		for (int j = 0; j < element.properties.size(); ++j) {
			if (element.properties[j].name == propertyKeys) {
				propertyIdx = j;
				break;
			}
		}

		if (propertyIdx >= 0) return true;
		else return false;
	}
	else return false;
}

bool PLYparser::loadPLY(const std::string& fileName, ulonglong& n_points,  Vertex** vertices)
{
	std::string inputPath = fileName;
	if ((fileName.find(".ply") == std::string::npos) && (fileName.find(".PLY") == std::string::npos))
		inputPath.append(".ply");
	std::ifstream File(inputPath, std::ios::in | std::ios::binary);

	int color_channels;
	bool isBinary = false;
	bool isBigEndian = false;
	std::vector<PlyElement> elements;
	std::vector<std::string> comments;
	std::vector<std::string> objInfo;

	if (File.is_open()) {
		//parse header
		std::string line;
		std::getline(File, line);
		std::istringstream lineStr(line);
		std::string token;
		lineStr.clear();
		lineStr >> token;

		if ((token != "ply") && (token != "PLY")) {
			LOG("<FAILED> Wrong file ext: %s\n", token.c_str());
			File.close();
			return false;
		}
		else {
			//parse PLY header
			while (std::getline(File, line)) {
				//std::istringstream lineStr(line);
				lineStr.clear();
				lineStr.str(line);
				std::istream(lineStr.rdbuf()) >> token;

				if (token == "comment") {
					comments.push_back((8 > 0) ? line.erase(0, 8) : line);
				}
				else if (token == "format") {
					std::string str;
					lineStr.clear();
					lineStr >> str;
					if (str == "binary_little_endian") isBinary = true;
					else if (str == "binary_big_endian") isBinary = isBigEndian = true;
				}
				else if (token == "element") {
					elements.emplace_back(lineStr);
				}
				else if (token == "property") {
					if (!elements.size())
						LOG("<FAILED> No Elements defined, file is malformed.\n");
					elements.back().properties.emplace_back(lineStr);
				}
				else if (token == "obj_info") objInfo.push_back((9 > 0) ? line.erase(0, 9) : line);
				else if (token == "end_header") break;
			}

#ifdef _DEBUG
			//print comment list
			for (auto cmt : comments) {
				LOG("Comment : %s\n", cmt.c_str());
			}

			//print element and property list
			for (auto elmnt : elements) {
				LOG("Element - %s : ( %lld )\n", elmnt.name.c_str(), elmnt.size);
				for (auto Property : elmnt.properties) {
					auto tmp = PropertyTable[Property.propertyType].second;
					LOG("\tProperty : %s : ( %s )\n", Property.name.c_str(), PropertyTable[Property.propertyType].second.c_str());
				}
			}
#endif
			longlong idxE_color = -1;
			int idxP_channel = -1;
			bool found_channel = findIdxOfPropertiesAndElement(elements, "color", "channel", idxE_color, idxP_channel);

			longlong idxE_vertex = -1;
			int idxP_x = -1;
			int idxP_y = -1;
			int idxP_z = -1;
			bool found_vertex = findIdxOfPropertiesAndElement(elements, "vertex", "x", idxE_vertex, idxP_x);
			found_vertex = findIdxOfPropertiesAndElement(elements, "vertex", "y", idxE_vertex, idxP_y);
			found_vertex = findIdxOfPropertiesAndElement(elements, "vertex", "z", idxE_vertex, idxP_z);
			if (!found_vertex) {
				LOG("<FAILED> File is not having vertices data.\n");
				File.close();
				return false;
			}

			longlong idxE_face = -1;
			int idxP_list = -1;
			int idxP_red = -1;
			int idxP_green = -1;
			int idxP_blue = -1;
			int idxP_alpha = -1;
			bool found_face = findIdxOfPropertiesAndElement(elements, "face", "vertex_indices", idxE_face, idxP_list);
			bool found_alpha = findIdxOfPropertiesAndElement(elements, "face", "alpha", idxE_face, idxP_alpha);
			bool found_color = findIdxOfPropertiesAndElement(elements, "vertex", "red", idxE_vertex, idxP_red);
			found_color = findIdxOfPropertiesAndElement(elements, "vertex", "green", idxE_vertex, idxP_green);
			found_color = findIdxOfPropertiesAndElement(elements, "vertex", "blue", idxE_vertex, idxP_blue);

			if (!found_color) {
				if (found_vertex) {
					found_color = findIdxOfPropertiesAndElement(elements, "vertex", "diffuse_red", idxE_vertex, idxP_red);
					found_color = findIdxOfPropertiesAndElement(elements, "vertex", "diffuse_green", idxE_vertex, idxP_green);
					found_color = findIdxOfPropertiesAndElement(elements, "vertex", "diffuse_blue", idxE_vertex, idxP_blue);
				}
				if (!found_color && found_face) {
					found_color = findIdxOfPropertiesAndElement(elements, "face", "red", idxE_face, idxP_red);
					found_color = findIdxOfPropertiesAndElement(elements, "face", "green", idxE_face, idxP_green);
					found_color = findIdxOfPropertiesAndElement(elements, "face", "blue", idxE_face, idxP_blue);

				}
			}

			n_points = elements[idxE_vertex].size;


			// Memory allocation
			if (*vertices != nullptr)
			{
				delete[] *vertices;
				*vertices = nullptr;
			}

			*vertices = new Vertex[n_points];
			std::memset(*vertices, 0, sizeof(Vertex) * n_points);

			int idxP_phase = -1;
			bool isPhaseParse = findIdxOfPropertiesAndElement(elements, "vertex", "phase", idxE_vertex, idxP_phase);

			// BINARY
			if (isBinary) {
				//parse Point Cloud Data
				for (size_t idxE = 0; idxE < elements.size(); ++idxE) {
					// parse vertex
					for (longlong e = 0; e < elements[idxE].size; ++e) {
						//vertex data parsing
						if (idxE == idxE_vertex) {
							auto list = 0;
							auto x = 0.0f;
							auto y = 0.0f;
							auto z = 0.0f;
							auto red = 0;
							auto green = 0;
							auto blue = 0;
							auto alpha = 0;
							auto phase = 0.0f;
							void* tmp = nullptr;

							//line Processing
							for (int idxP = 0; idxP < elements[idxE].properties.size(); ++idxP) {

								if (idxP == idxP_x) tmp = &x;
								else if (idxP == idxP_y) tmp = &y;
								else if (idxP == idxP_z) tmp = &z;
								else if (idxP == idxP_red) tmp = &red;
								else if (idxP == idxP_green) tmp = &green;
								else if (idxP == idxP_blue) tmp = &blue;
								else if (idxP == idxP_phase && isPhaseParse) tmp = &phase;
								else tmp = nullptr;
								if (tmp != nullptr)
								{
									File.read((char*)tmp, PropertyTable[elements[idxE].properties[idxP].propertyType].first);
								}
							}

							(*vertices)[e].point.pos[_X] = x;
							(*vertices)[e].point.pos[_Y] = y;
							(*vertices)[e].point.pos[_Z] = z;

							if (!found_face)
							{
								(*vertices)[e].color.color[_R] = (Real)(red / 255.f);
								(*vertices)[e].color.color[_G] = (Real)(green / 255.f);
								(*vertices)[e].color.color[_B] = (Real)(blue / 255.f);
							}
							if (isPhaseParse) {
								(*vertices)[e].phase = (Real)phase;
							}
						}
						//face data parsing
						else if (idxE == idxE_face) {
							auto red = 0;
							auto green = 0;
							auto blue = 0;
							auto alpha = 0;
							void* tmp = nullptr;

							//line Processing
							for (int idxP = 0; idxP < elements[idxE].properties.size(); ++idxP) {
								// do not processing
								if (elements[idxE].properties[idxP].isList) { // list type

									auto nCnt = 0;
									File.read((char*)&nCnt, PropertyTable[elements[idxE].properties[idxP].listType].first);
									int* pTest = new int[nCnt];
									for (int i = 0; i < nCnt; ++i) {
										File.read((char*)&pTest[i], PropertyTable[elements[idxE].properties[idxP].propertyType].first);
									}
									delete[] pTest;
								}

								if (idxP == idxP_red) tmp = &red;
								else if (idxP == idxP_green) tmp = &green;
								else if (idxP == idxP_blue) tmp = &blue;
								else if (idxP == idxP_alpha) tmp = &alpha;
								else tmp = nullptr;
								if (tmp != nullptr)
								{
									File.read((char*)tmp, PropertyTable[elements[idxE].properties[idxP].propertyType].first);
								}

								(*vertices)[e].color.color[_R] = (Real)(red / 255.f);
								(*vertices)[e].color.color[_G] = (Real)(green / 255.f);
								(*vertices)[e].color.color[_B] = (Real)(blue / 255.f);
							}
						}
						//color channel parsing
						else if (found_channel && (idxE == idxE_color)) {
							for (int idxP = 0; idxP < elements[idxE].properties.size(); ++idxP) {
								File.read((char*)&color_channels, PropertyTable[elements[idxE].properties[idxP].propertyType].first);
							}
						}
					}
				}
			}
			// ASCII
			else {
				//parse Point Cloud Data
				for (size_t idxE = 0; idxE < elements.size(); ++idxE) {
					// parse vertex
					for (longlong e = 0; e < elements[idxE].size; ++e) {

						std::getline(File, line);
						lineStr.str(line);
						std::string val;

						//color channel parsing
						if (found_channel && (idxE == idxE_color)) {
							lineStr.clear();
							lineStr >> val;
							color_channels = std::stoi(val);
						}

						//vertex data parsing
						if (idxE == idxE_vertex) {
							Real x = 0.0;
							Real y = 0.0;
							Real z = 0.0;
							auto red = 0;
							auto green = 0;
							auto blue = 0;
							auto alpha = 0;
							auto phase = 0.f;

							//line Processing
							for (int idxP = 0; idxP < elements[idxE].properties.size(); ++idxP) {
								lineStr.clear();

								lineStr >> val;

								if (idxP == idxP_x) x = std::stod(val);
								else if (idxP == idxP_y) y = std::stod(val);
								else if (idxP == idxP_z) z = std::stod(val);
								else if (idxP == idxP_red) red = std::stoi(val);
								else if (idxP == idxP_green) green = std::stoi(val);
								else if (idxP == idxP_blue) blue = std::stoi(val);
								else if (idxP == idxP_alpha) alpha = std::stoi(val);
								else if ((idxP == idxP_phase) && isPhaseParse) phase = std::stod(val);
							}

							(*vertices)[e].point.pos[_X] = x;
							(*vertices)[e].point.pos[_Y] = y;
							(*vertices)[e].point.pos[_Z] = z;

							if (!found_face)
							{
								(*vertices)[e].color.color[_R] = (Real)(red / 255.f);
								(*vertices)[e].color.color[_G] = (Real)(green / 255.f);
								(*vertices)[e].color.color[_B] = (Real)(blue / 255.f);
							}
							if (isPhaseParse) {
								(*vertices)[e].phase = (Real)phase;
							}
						}
					}
				}
			}
			File.close();

			// post process
			if (!found_channel) {
				for (ulonglong i = 0; i < n_points; ++i) {
					(*vertices)[i].color.color[_R] = 0.5;
					(*vertices)[i].color.color[_G] = 0.5;
					(*vertices)[i].color.color[_B] = 0.5;
				}
			}
			return true;
		}
	}
	else {
		LOG("<FAILED> Loading ply file.\n");
		return false;
	}
}

bool PLYparser::savePLY(const std::string& fileName, const ulonglong n_points, Vertex* vertices, bool isBinary)
{
	if (vertices == nullptr) {
		LOG("<FAILED> There is not data for saving ply file.\n");
		return false;
	}

	std::string outputPath = fileName;
	if ((fileName.find(".ply") == std::string::npos) && (fileName.find(".PLY") == std::string::npos))
		outputPath.append(".ply");
	if (isBinary)
	{
		std::ofstream File(outputPath, std::ios::out | std::ios::trunc | std::ios::binary);

		if (File.is_open()) {
			File << "ply\n";
			File << "format ascii 1.0\n";
			File << "comment Point Cloud Data Format in OpenHolo Library v" << _OPH_LIB_VERSION_MAJOR_ << "." << _OPH_LIB_VERSION_MINOR_ << "\n";
			File << "element color 1\n";
			File << "property int channel\n";
			File << "element vertex " << n_points << std::endl;
			File << "property float x\n";
			File << "property float y\n";
			File << "property float z\n";
			File << "property uchar red\n";
			File << "property uchar green\n";
			File << "property uchar blue\n";
			File << "property Real phase\n";
			File << "end_header\n";
			int color_channels = 3;
			File.write(reinterpret_cast<char*>(&color_channels), sizeof(color_channels));

			for (ulonglong i = 0; i < n_points; ++i) {
				float x = vertices[i].point.pos[_X];
				float y = vertices[i].point.pos[_Y];
				float z = vertices[i].point.pos[_Z];
				char r = vertices[i].color.color[_R] * 255.f + 0.5f;
				char g = vertices[i].color.color[_G] * 255.f + 0.5f;
				char b = vertices[i].color.color[_B] * 255.f + 0.5f;
				// Vertex Geometry
				File.write(reinterpret_cast<char *>(&x), sizeof(x));
				File.write(reinterpret_cast<char *>(&y), sizeof(y));
				File.write(reinterpret_cast<char *>(&z), sizeof(z));
				File.write(reinterpret_cast<char *>(&r), sizeof(r));
				File.write(reinterpret_cast<char *>(&g), sizeof(g));
				File.write(reinterpret_cast<char *>(&b), sizeof(b));
			}
			File.close();
			return true;
		}
	}
	else
	{
		std::ofstream File(outputPath, std::ios::out | std::ios::trunc);

		if (File.is_open()) {
			File << "ply\n";
			File << "format ascii 1.0\n";
			File << "comment Point Cloud Data Format in OpenHolo Library v" << _OPH_LIB_VERSION_MAJOR_ << "." << _OPH_LIB_VERSION_MINOR_ << "\n";
			File << "element color 1\n";
			File << "property int channel\n";
			File << "element vertex " << n_points << std::endl;
			File << "property float x\n";
			File << "property float y\n";
			File << "property float z\n";
			File << "property uchar red\n";
			File << "property uchar green\n";
			File << "property uchar blue\n";
			File << "property Real phase\n";
			File << "end_header\n";
			int color_channels = 3;
			File << color_channels << std::endl;

			for (ulonglong i = 0; i < n_points; ++i) {
				// Vertex Geometry
				File << std::fixed << vertices[i].point.pos[_X] << " " << vertices[i].point.pos[_Y] << " " << vertices[i].point.pos[_Z] << " ";
				// Color
				File << vertices[i].color.color[_R] * 255.f + 0.5f << " " << vertices[i].color.color[_G] * 255.f + 0.5f << " " << vertices[i].color.color[_B] * 255.f + 0.5f << " ";
				// Phase
				File << std::fixed << vertices[i].phase << std::endl;
			}
			File.close();
			return true;
		}
		else {
			LOG("<FAILED> Saving ply file.\n");
			return false;
		}
	}
}

bool PLYparser::loadPLY(const std::string& fileName, ulonglong& n_vertices, Face** faces)
{
	std::string inputPath = fileName;
	if ((inputPath.find(".ply") == std::string::npos) && (inputPath.find(".PLY") == std::string::npos))
		inputPath.append(".ply");
	std::ifstream File(inputPath, std::ios::in | std::ios::binary);

	int color_channels;
	bool isBinary = false;
	bool isBigEndian = false;
	std::vector<PlyElement> elements;
	std::vector<std::string> comments;
	std::vector<std::string> objInfo;

	if (File.is_open()) {
		//parse header
		std::string line;
		std::getline(File, line);
		std::istringstream lineStr(line);
		std::string token;
		lineStr >> token;

		if ((token != "ply") && (token != "PLY")) {
			LOG("<FAILED> Wrong file ext: %s\n", token.c_str());
			File.close();
			return false;
		}
		else {
			//parse PLY header
			while (std::getline(File, line)) {
				lineStr.clear();
				lineStr.str(line);
				std::istream(lineStr.rdbuf()) >> token;

				if (token == "comment") comments.push_back((8 > 0) ? line.erase(0, 8) : line);
				else if (token == "format") {
					std::string str;
					lineStr >> str;
					if (str == "binary_little_endian") isBinary = true;
					else if (str == "binary_big_endian") isBinary = isBigEndian = true;
				}
				else if (token == "element") elements.emplace_back(lineStr);
				else if (token == "property") {
					if (!elements.size())
						LOG("<FAILED> No Elements defined, file is malformed.\n");
					elements.back().properties.emplace_back(lineStr);
				}
				else if (token == "obj_info") objInfo.push_back((9 > 0) ? line.erase(0, 9) : line);
				else if (token == "end_header") break;
			}
#ifdef _DEBUG
			//print comment list
			for (auto cmt : comments) {
				LOG("Comment : %s\n", cmt.c_str());
			}

			//print element and property list
			for (auto elmnt : elements) {
				LOG("Element - %s : ( %lld )\n", elmnt.name.c_str(), elmnt.size);
				for (auto Property : elmnt.properties) {
					LOG("\tProperty : %s : ( %s )\n", Property.name.c_str(), PropertyTable[Property.propertyType].second);
				}
			}
#endif
			longlong idxE_color = -1;
			int idxP_channel = -1;
			bool ok_channel = findIdxOfPropertiesAndElement(elements, "color", "channel", idxE_color, idxP_channel);

			longlong idxE_vertex = -1;
			int idxP_face_idx = -1;
			int idxP_x = -1;
			int idxP_y = -1;
			int idxP_z = -1;
			bool ok_vertex = findIdxOfPropertiesAndElement(elements, "vertex", "face_idx", idxE_vertex, idxP_face_idx);
			ok_vertex = findIdxOfPropertiesAndElement(elements, "vertex", "x", idxE_vertex, idxP_x);
			ok_vertex = findIdxOfPropertiesAndElement(elements, "vertex", "y", idxE_vertex, idxP_y);
			ok_vertex = findIdxOfPropertiesAndElement(elements, "vertex", "z", idxE_vertex, idxP_z);

			if (!ok_vertex) {
				LOG("<FAILED> File is not having vertices data.\n");
				File.close();
				return false;
			}

			int idxP_red = -1;
			int idxP_green = -1;
			int idxP_blue = -1;
			int idxP_alpha = -1;
			bool ok_color = findIdxOfPropertiesAndElement(elements, "vertex", "red", idxE_vertex, idxP_red);
			ok_color = findIdxOfPropertiesAndElement(elements, "vertex", "green", idxE_vertex, idxP_green);
			ok_color = findIdxOfPropertiesAndElement(elements, "vertex", "blue", idxE_vertex, idxP_blue);
			if (!ok_color) {
				ok_color = findIdxOfPropertiesAndElement(elements, "vertex", "diffuse_red", idxE_vertex, idxP_red);
				ok_color = findIdxOfPropertiesAndElement(elements, "vertex", "diffuse_green", idxE_vertex, idxP_green);
				ok_color = findIdxOfPropertiesAndElement(elements, "vertex", "diffuse_blue", idxE_vertex, idxP_blue);

				if (!ok_vertex) {
					LOG("<FAILED> File is not having color data.\n");
					File.close();
					return false;
				}
			}

			n_vertices = elements[idxE_vertex].size;

			// Memory allocation
			if (*faces != nullptr)
			{
				delete[] * faces;
				*faces = nullptr;
			}


			*faces = new Face[n_vertices];
			std::memset(*faces, 0, sizeof(Face) * n_vertices);

			// Binary Mode
			if (isBinary)
			{
				// Elements Size
				for (size_t idxE = 0; idxE < elements.size(); ++idxE) {
					// Property Size
					for (longlong e = 0; e < elements[idxE].size; ++e) {

						//color channel parsing
						if (ok_channel && (idxE == idxE_color)) {
							for (int idxP = 0; idxP < elements[idxE].properties.size(); ++idxP) {
								int nSize = PropertyTable[elements[idxE].properties[idxP].propertyType].first;
								void *tmp = &color_channels;
								if (tmp != nullptr) {
									File.read((char*)tmp, nSize);
								}
							}
						}

						//vertex data parsing
						if (idxE == idxE_vertex) {
							auto face = 0;
							auto x = 0.f;
							auto y = 0.f;
							auto z = 0.f; 
							auto red = 0.f;
							auto green = 0.f;
							auto blue = 0.f;
							void *tmp = nullptr;
							
							//line Processing
							for (int idxP = 0; idxP < elements[idxE].properties.size(); ++idxP) {

								if (idxP == idxP_face_idx) tmp = &face;
								else if (idxP == idxP_x) tmp = &x;
								else if (idxP == idxP_y) tmp = &y;
								else if (idxP == idxP_z) tmp = &z;
								else if (idxP == idxP_red) tmp = &red;
								else if (idxP == idxP_green) tmp = &green;
								else if (idxP == idxP_blue) tmp = &blue;

								if (tmp != nullptr)
								{
									File.read((char*)tmp, PropertyTable[elements[idxE].properties[idxP].propertyType].first);
								}
							}

							int div = e / 3;
							int mod = e % 3;

							(*faces)[div].idx = face;
							(*faces)[div].vertices[mod].point.pos[_X] = x;
							(*faces)[div].vertices[mod].point.pos[_Y] = y;
							(*faces)[div].vertices[mod].point.pos[_Z] = z;
							(*faces)[div].vertices[mod].color.color[_R] = (Real)(red / 255.f);
							(*faces)[div].vertices[mod].color.color[_G] = (Real)(green / 255.f);
							(*faces)[div].vertices[mod].color.color[_B] = (Real)(blue / 255.f);
						}
					}
				}

			}
			// Text Mode
			else
			{
				for (size_t idxE = 0; idxE < elements.size(); ++idxE) {

					// Parse element vertex
					for (longlong e = 0; e < elements[idxE].size; ++e) {

						std::getline(File, line);
						lineStr.clear();
						lineStr.str(line);
						std::string val;
						if (ok_channel && (idxE == idxE_color)) {
							lineStr >> val;
							color_channels = std::stoi(val);
						}

						//vertex data parsing
						if (idxE == idxE_vertex) {
							auto face = 0;
							auto x = 0.f;
							auto y = 0.f;
							auto z = 0.f;
							auto red = 0.f;
							auto green = 0.f;
							auto blue = 0.f;

							//line Processing
							for (int p = 0; p < elements[idxE].properties.size(); ++p) {
								lineStr.clear();
								lineStr >> val;
								if (p == idxP_face_idx) face = std::stoi(val);
								if (p == idxP_x) x = std::stof(val);
								else if (p == idxP_y) y = std::stof(val);
								else if (p == idxP_z) z = std::stof(val);
								else if (p == idxP_red) red = std::stoi(val);
								else if (p == idxP_green) green = std::stoi(val);
								else if (p == idxP_blue) blue = std::stoi(val);
							}

							int div = e / 3;
							int mod = e % 3;

							(*faces)[div].idx = div;
							(*faces)[div].vertices[mod].point.pos[_X] = x;
							(*faces)[div].vertices[mod].point.pos[_Y] = y;
							(*faces)[div].vertices[mod].point.pos[_Z] = z;
							(*faces)[div].vertices[mod].color.color[_R] = (Real)(red / 255.f);
							(*faces)[div].vertices[mod].color.color[_G] = (Real)(green / 255.f);
							(*faces)[div].vertices[mod].color.color[_B] = (Real)(blue / 255.f);
						}
					}
				}
			}

			File.close();
			
			return true;
		}
	}
	else {
		LOG("<FAILED> Loading ply file.\n");
		return false;
	}
}

bool PLYparser::savePLY(const std::string& fileName, const ulonglong n_vertices, Face *faces, bool isBinary)
{
	if (faces == nullptr) {
		LOG("<FAILED> There is not data for saving ply file.\n");
		return false;
	}

	std::string outputPath = fileName;
	if ((outputPath.find(".ply") == std::string::npos) && (outputPath.find(".PLY") == std::string::npos)) outputPath += ".ply";

	if (isBinary)
	{
		std::ofstream File(outputPath, std::ios::out | std::ios::trunc | std::ios::binary);

		if (File.is_open()) {
			File << "ply\n";
			File << "format ascii 1.0\n";
			File << "comment Triangle Mesh Data Format in OpenHolo Library v" << _OPH_LIB_VERSION_MAJOR_ << "." << _OPH_LIB_VERSION_MINOR_ << "\n";
			File << "element color 1\n";
			File << "property int channel\n";
			File << "element vertex " << n_vertices << std::endl;
			File << "property uint face_idx\n";
			File << "property float x\n";
			File << "property float y\n";
			File << "property float z\n";
			File << "property uchar red\n";
			File << "property uchar green\n";
			File << "property uchar blue\n";
			File << "end_header\n";
			int color_channels = 3;

			File.write(reinterpret_cast<char*>(&color_channels), sizeof(color_channels));

			for (ulonglong i = 0; i < n_vertices; ++i) {

				int div = i / 3;
				int mod = i % 3;

				uint face_idx = faces[div].idx;
				float x = faces[div].vertices[mod].point.pos[_X];
				float y = faces[div].vertices[mod].point.pos[_Y];
				float z = faces[div].vertices[mod].point.pos[_Z];
				char r = faces[div].vertices[mod].color.color[_R] * 255.f + 0.5f;
				char g = faces[div].vertices[mod].color.color[_G] * 255.f + 0.5f;
				char b = faces[div].vertices[mod].color.color[_B] * 255.f + 0.5f;

				// index
				File.write(reinterpret_cast<char*>(&face_idx), sizeof(face_idx));

				// Vertex Geometry
				File.write(reinterpret_cast<char*>(&x), sizeof(x));
				File.write(reinterpret_cast<char*>(&y), sizeof(y));
				File.write(reinterpret_cast<char*>(&z), sizeof(z));
				File.write(reinterpret_cast<char*>(&r), sizeof(r));
				File.write(reinterpret_cast<char*>(&g), sizeof(g));
				File.write(reinterpret_cast<char*>(&b), sizeof(b));
			}
			File.close();
			return true;
		}
		else {
			LOG("<FAILED> Saving ply file.\n");
			return false;
		}
	}
	else
	{
		std::ofstream File(outputPath, std::ios::out | std::ios::trunc);

		if (File.is_open()) {
			File << "ply\n";
			File << "format ascii 1.0\n";
			File << "comment Triangle Mesh Data Format in OpenHolo Library v" << _OPH_LIB_VERSION_MAJOR_ << "." << _OPH_LIB_VERSION_MINOR_ << "\n";
			File << "element color 1\n";
			File << "property int channel\n";
			File << "element vertex " << n_vertices << std::endl;
			File << "property uint face_idx\n";
			File << "property float x\n";
			File << "property float y\n";
			File << "property float z\n";
			File << "property uchar red\n";
			File << "property uchar green\n";
			File << "property uchar blue\n";
			File << "end_header\n";
			int color_channels = 3;
			File << color_channels << std::endl;

			for (ulonglong i = 0; i < n_vertices; ++i) {
				int div = i / 3;
				int mod = i % 3;

				//Vertex Face Index
				File << std::fixed << faces[div].idx << " ";

				//Vertex Geometry
				File << std::fixed << faces[div].vertices[mod].point.pos[_X] << " " << faces[div].vertices[mod].point.pos[_Y] << " " << faces[div].vertices[mod].point.pos[_Z] << " ";

				//Color Amplitude
				File << faces[div].vertices[mod].color.color[_R] * 255.f + 0.5f << " " << faces[div].vertices[mod].color.color[_G] * 255.f + 0.5f << " " << faces[div].vertices[mod].color.color[_B] * 255.f + 0.5f << " ";

			}
			File.close();
			return true;
		}
		else {
			LOG("<FAILED> Saving ply file.\n");
			return false;
		}
	}	
}