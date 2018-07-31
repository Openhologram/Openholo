#ifndef __PLY_PARSER_H__
#define __PLY_PARSER_H__

#include <iostream>
#include <sstream>
#include <fstream>
#include <map>

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


class __declspec(dllexport) PLYparser {

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
	std::map<Type, std::pair<int, std::string> > PropertyTable {
		{ Type::INT8,		{ 1, "char"		} },
		{ Type::UINT8,		{ 1, "uchar"	} },
		{ Type::INT16,		{ 2, "short"	} },
		{ Type::UINT16,		{ 2, "ushort"	} },
		{ Type::INT32,		{ 4, "int"		} },
		{ Type::UINT32,		{ 4, "uint"		} },
		{ Type::FLOAT32,	{ 4, "float"	} },
		{ Type::FLOAT64,	{ 8, "double"	} },
		{ Type::INVALID,	{ 0, "INVALID"	} }
	};

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
	bool loadPLY(
		const std::string fileName,
		ulonglong &n_points,
		int &color_channels, //1 or 3 (If it is 4, Alpha channel is not loaded.)
		Real** vertexArray,
		Real** colorArray,
		Real** phaseArray, //If isPhaseParse is false, PhaseArray is nullptr
		bool &isPhaseParse);

	bool savePLY(
		const std::string fileName,
		const ulonglong n_points,
		const int color_channels,
		Real* vertexArray, 
		Real* colorArray, 
		Real* phaseArray);

};


#endif