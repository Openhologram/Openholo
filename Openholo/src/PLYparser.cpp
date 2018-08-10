#include "PLYparser.h"

PLYparser::PLYparser()
{
	PropertyTable.insert(std::make_pair(Type::INT8,		std::make_pair(1, "char"	)));
	PropertyTable.insert(std::make_pair(Type::UINT8,	std::make_pair(1, "uchar"	)));
	PropertyTable.insert(std::make_pair(Type::INT16,	std::make_pair(2, "short"	)));
	PropertyTable.insert(std::make_pair(Type::UINT16,	std::make_pair(2, "ushort"	)));
	PropertyTable.insert(std::make_pair(Type::INT32,	std::make_pair(4, "int"		)));
	PropertyTable.insert(std::make_pair(Type::UINT32,	std::make_pair(4, "uint"	)));
	PropertyTable.insert(std::make_pair(Type::FLOAT32,	std::make_pair(4, "float"	)));
	PropertyTable.insert(std::make_pair(Type::FLOAT64,	std::make_pair(8, "double"	)));
	PropertyTable.insert(std::make_pair(Type::INVALID,	std::make_pair(0, "INVALID"	)));
}

PLYparser::~PLYparser()
{
}

PLYparser::PlyProperty::PlyProperty(std::istream &is)
	: isList(false) {
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
	: propertyType(type), name(_name) {
}


PLYparser::PlyProperty::PlyProperty(const Type list_type, const Type prop_type, const std::string &_name, const ulonglong list_count)
	: listType(list_type), propertyType(prop_type), name(_name), listCount(list_count), isList(true) {
}


PLYparser::PlyElement::PlyElement(std::istream &is) {
	is >> name >> size;
}


PLYparser::PlyElement::PlyElement(const std::string &_name, const ulonglong count)
	: name(_name), size(count) {
}

PLYparser::Type PLYparser::propertyTypeFromString(const std::string &t) {
	if (t == "int8" || t == "char")             return Type::INT8;
	else if (t == "uint8" || t == "uchar")      return Type::UINT8;
	else if (t == "int16" || t == "short")      return Type::INT16;
	else if (t == "uint16" || t == "ushort")    return Type::UINT16;
	else if (t == "int32" || t == "int")        return Type::INT32;
	else if (t == "uint32" || t == "uint")      return Type::UINT32;
	else if (t == "float32" || t == "float")    return Type::FLOAT32;
	else if (t == "float64" || t == "double")   return Type::FLOAT64;
	else return Type::INVALID;
}


bool PLYparser::findIdxOfPropertiesAndElement(const std::vector<PlyElement> &elements, const std::string &elementKey, const std::string &propertyKeys, longlong &elementIdx, int &propertyIdx) {
	elementIdx = -1;
	for (size_t i = 0; i < elements.size(); ++i) {
		if (elements[i].name == elementKey) elementIdx = i;
	}

	if (elementIdx >= 0) {
		const PlyElement &element = elements[elementIdx];

		propertyIdx = -1;
		for (int j = 0; j < element.properties.size(); ++j) {
			if (element.properties[j].name == propertyKeys) propertyIdx = j;
		}

		if (propertyIdx >= 0) return true;
		else return false;
	}
	else return false;
}


bool PLYparser::loadPLY(const std::string fileName, ulonglong &n_points, int &color_channels, Real** vertexArray, Real** colorArray, Real** phaseArray, bool &isPhaseParse) {
	std::string inputPath = fileName;
	if ((fileName.find(".ply") == std::string::npos) && (fileName.find(".PLY") == std::string::npos)) inputPath += ".ply";
	std::ifstream File(inputPath, std::ios::in | std::ios::binary);

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
			std::cerr << "Error : Failed loading ply file..." << std::endl;
			File.close();
			return false;
		}
		else {
			std::cout << "Parsing *.PLY file for OpenHolo Point Cloud Generation..." << std::endl;

			//parse PLY header
			while (std::getline(File, line)) {
				lineStr.str(line);
				lineStr >> token;

				if (token == "comment") comments.push_back((8 > 0) ? line.erase(0, 8) : line);
				else if (token == "format") {
					std::string str;
					lineStr >> str;
					if (str == "binary_little_endian") isBinary = true;
					else if (str == "binary_big_endian") isBinary = isBigEndian = true;
				}
				else if (token == "element") elements.emplace_back(lineStr);
				else if (token == "property") {
					if (!elements.size()) std::cerr << "No Elements defined, file is malformed" << std::endl;
					elements.back().properties.emplace_back(lineStr);
				}
				else if (token == "obj_info") objInfo.push_back((9 > 0) ? line.erase(0, 9) : line);
				else if (token == "end_header") break;
			}

			//print comment list
			for (auto cmt : comments) {
				std::cout << "Comment : " << cmt << std::endl;
			}

			//print element and property list
			for (auto elmnt : elements) {
				std::cout << "Element - " << elmnt.name << " : ( " << elmnt.size << " )" << std::endl;
				for (auto Property : elmnt.properties) {
					auto tmp = PropertyTable[Property.propertyType].second;
					std::cout << "\tProperty : " << Property.name << " : ( " << PropertyTable[Property.propertyType].second << " )" << std::endl;
				}
			}

			longlong idxE_color = -1;
			int idxP_channel = -1;
			bool ok_channel = findIdxOfPropertiesAndElement(elements, "color", "channel", idxE_color, idxP_channel);

			longlong idxE_vertex = -1;
			int idxP_x = -1;
			int idxP_y = -1;
			int idxP_z = -1;
			bool ok_vertex = findIdxOfPropertiesAndElement(elements, "vertex", "x", idxE_vertex, idxP_x);
			ok_vertex = findIdxOfPropertiesAndElement(elements, "vertex", "y", idxE_vertex, idxP_y);
			ok_vertex = findIdxOfPropertiesAndElement(elements, "vertex", "z", idxE_vertex, idxP_z);
			if (!ok_vertex) {
				std::cerr << "Error : file is not having vertices data..." << std::endl;
				return false;
			}

			int idxP_red = -1;
			int idxP_green = -1;
			int idxP_blue = -1;
			bool ok_color = findIdxOfPropertiesAndElement(elements, "vertex", "red", idxE_vertex, idxP_red);
			ok_color = findIdxOfPropertiesAndElement(elements, "vertex", "green", idxE_vertex, idxP_green);
			ok_color = findIdxOfPropertiesAndElement(elements, "vertex", "blue", idxE_vertex, idxP_blue);
			if (!ok_color) {
				ok_color = findIdxOfPropertiesAndElement(elements, "vertex", "diffuse_red", idxE_vertex, idxP_red);
				ok_color = findIdxOfPropertiesAndElement(elements, "vertex", "diffuse_green", idxE_vertex, idxP_green);
				ok_color = findIdxOfPropertiesAndElement(elements, "vertex", "diffuse_blue", idxE_vertex, idxP_blue);

				if (!ok_vertex) {
					std::cerr << "Error : file is not having vertices colour data..." << std::endl;
					return false;
				}
			}

			int idxP_phase = -1;
			isPhaseParse = findIdxOfPropertiesAndElement(elements, "vertex", "phase", idxE_vertex, idxP_phase);
			if (!isPhaseParse) *phaseArray = nullptr;

			n_points = elements[idxE_vertex].size;
			*vertexArray = new Real[3 * n_points];
			*colorArray = new Real[3 * n_points];
			std::memset(*vertexArray, NULL, sizeof(Real) * 3 * n_points);
			std::memset(*colorArray, NULL, sizeof(Real) * 3 * n_points);
			if (isPhaseParse) {
				*phaseArray = new Real[n_points];
				std::memset(*phaseArray, NULL, sizeof(Real) * n_points);
			}

			//parse Point Cloud Data
			for (size_t idxE = 0; idxE < elements.size(); ++idxE) {
				for (longlong e = 0; e < elements[idxE].size; ++e) {
					std::getline(File, line);
					lineStr.str(line);
					std::string val;

					//color channel parsing
					if (ok_channel && (idxE == idxE_color)) {
						lineStr >> val;
						color_channels = std::stoi(val);
					}

					//vertex data parsing
					if (idxE == idxE_vertex) {
						Real x = 0.f;
						Real y = 0.f;
						Real z = 0.f;
						uchar red = 0;
						uchar green = 0;
						uchar blue = 0;
						Real phase = 0.f;

						//line Processing
						for (int p = 0; p < elements[idxE].properties.size(); ++p) {
							lineStr >> val;
							if (p == idxP_x) x = std::stof(val);
							else if (p == idxP_y) y = std::stof(val);
							else if (p == idxP_z) z = std::stof(val);
							else if (p == idxP_red) red = std::stoi(val);
							else if (p == idxP_green) green = std::stoi(val);
							else if (p == idxP_blue) blue = std::stoi(val);
							else if ((p == idxP_phase) && isPhaseParse) phase = std::stof(val);
						}

						(*vertexArray)[3 * e + 0] = x;
						(*vertexArray)[3 * e + 1] = y;
						(*vertexArray)[3 * e + 2] = z;
						(*colorArray)[3 * e + 0] = (Real)(red / 255.f);
						(*colorArray)[3 * e + 1] = (Real)(green / 255.f);
						(*colorArray)[3 * e + 2] = (Real)(blue / 255.f);
						if (isPhaseParse) (*phaseArray)[e] = phase;
					}
				}
			}
			File.close();

			if (ok_channel && (color_channels == 1)) {
				Real* grayArray = new Real[n_points];
				for (ulonglong i = 0; i < n_points; ++i) {
					grayArray[i] = (*colorArray)[3 * i];
				}
				delete[](*colorArray);
				*colorArray = grayArray;
			}
			else if (!ok_channel) {
				bool check = false;
				for (ulonglong i = 0; i < n_points; ++i) {
					if (((*colorArray)[3 * i + 0] != (*colorArray)[3 * i + 1]) || ((*colorArray)[3 * i + 1] != (*colorArray)[3 * i + 2])) {
						check = true;
						break;
					}
				}

				if (check) color_channels = 3;
				else if (!check) {
					color_channels = 1;
					Real* grayArray = new Real[n_points];
					for (ulonglong i = 0; i < n_points; ++i) {
						grayArray[i] = (*colorArray)[3 * i];
					}
					delete[](*colorArray);
					*colorArray = grayArray;
				}
			}

			std::cout << "Success loading " << n_points << " Point Clouds, Color Channels : " << color_channels << std::endl;

			return true;
		}
	}
	else {
		std::cerr << "Error : Failed loading ply file..." << std::endl;
		return false;
	}
}


bool PLYparser::savePLY(const std::string fileName, const ulonglong n_points, const int color_channels, Real* vertexArray, Real* colorArray, Real* phaseArray) {
	if ((vertexArray == nullptr) || (colorArray == nullptr) || (phaseArray == nullptr)) {
		std::cerr << "Error : There is not data for saving ply file..." << std::endl;
		return false;
	}
	if ((color_channels != 1) && (color_channels != 3)) {
		std::cerr << "Error : Number of Color channels for saving ply file is false value..." << std::endl;
		return false;
	}

	std::string outputPath = fileName;
	if ((fileName.find(".ply") == std::string::npos) && (fileName.find(".PLY") == std::string::npos)) outputPath += ".ply";
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

		File << color_channels << std::endl;

		for (ulonglong i = 0; i < n_points; ++i) {
			//Vertex Geometry
			File << std::fixed << vertexArray[3 * i + 0] << " " << vertexArray[3 * i + 1] << " " << vertexArray[3 * i + 2] << " ";
			
			//Color Amplitude
			if (color_channels == 3)
				File << (int)(255.f*colorArray[3 * i + 0] + 0.5f) << " " << (int)(255.f*colorArray[3 * i + 1] + 0.5f) << " " << (int)(255.f*colorArray[3 * i + 2] + 0.5f) << " ";
			else if (color_channels == 1) {
				int indensity = (int)(255.f*colorArray[i] + 0.5f);
				File << indensity << " " << indensity << " " << indensity << " ";
			}

			//Phase
			File << std::fixed << phaseArray[i] << std::endl;
		}
		File.close();
		return true;
	}
	else {
		std::cerr << "Error : Failed saving ply file..." << std::endl;
		return false;
	}
}

bool PLYparser::loadPLY(const char* fileName, ulonglong & n_vertices, int & color_channels, uint ** face_idx, Real ** vertexArray, Real ** colorArray)
{
	std::string inputPath = fileName;
	if ((inputPath.find(".ply") == std::string::npos) && (inputPath.find(".PLY") == std::string::npos)) inputPath += ".ply";
	std::ifstream File(inputPath, std::ios::in | std::ios::binary);

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
			std::cerr << "Error : Failed loading ply file..." << std::endl;
			File.close();
			return false;
		}
		else {
			std::cout << "Parsing *.PLY file for OpenHolo Triangle Mesh Generation..." << std::endl;

			//parse PLY header
			while (std::getline(File, line)) {
				lineStr.str(line);
				lineStr >> token;

				if (token == "comment") comments.push_back((8 > 0) ? line.erase(0, 8) : line);
				else if (token == "format") {
					std::string str;
					lineStr >> str;
					if (str == "binary_little_endian") isBinary = true;
					else if (str == "binary_big_endian") isBinary = isBigEndian = true;
				}
				else if (token == "element") elements.emplace_back(lineStr);
				else if (token == "property") {
					if (!elements.size()) std::cerr << "No Elements defined, file is malformed" << std::endl;
					elements.back().properties.emplace_back(lineStr);
				}
				else if (token == "obj_info") objInfo.push_back((9 > 0) ? line.erase(0, 9) : line);
				else if (token == "end_header") break;
			}

			//print comment list
			for (auto cmt : comments) {
				std::cout << "Comment : " << cmt << std::endl;
			}

			//print element and property list
			for (auto elmnt : elements) {
				std::cout << "Element - " << elmnt.name << " : ( " << elmnt.size << " )" << std::endl;
				for (auto Property : elmnt.properties) {
					std::cout << "\tProperty : " << Property.name << " : ( " << PropertyTable[Property.propertyType].second << " )" << std::endl;
				}
			}

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
				std::cerr << "Error : file is not having vertices data..." << std::endl;
				return false;
			}

			int idxP_red = -1;
			int idxP_green = -1;
			int idxP_blue = -1;
			bool ok_color = findIdxOfPropertiesAndElement(elements, "vertex", "red", idxE_vertex, idxP_red);
			ok_color = findIdxOfPropertiesAndElement(elements, "vertex", "green", idxE_vertex, idxP_green);
			ok_color = findIdxOfPropertiesAndElement(elements, "vertex", "blue", idxE_vertex, idxP_blue);
			if (!ok_color) {
				ok_color = findIdxOfPropertiesAndElement(elements, "vertex", "diffuse_red", idxE_vertex, idxP_red);
				ok_color = findIdxOfPropertiesAndElement(elements, "vertex", "diffuse_green", idxE_vertex, idxP_green);
				ok_color = findIdxOfPropertiesAndElement(elements, "vertex", "diffuse_blue", idxE_vertex, idxP_blue);

				if (!ok_vertex) {
					std::cerr << "Error : file is not having vertices color data..." << std::endl;
					return false;
				}
			}

			n_vertices = elements[idxE_vertex].size;
			*face_idx = new uint[n_vertices];
			*vertexArray = new Real[3 * n_vertices];
			*colorArray = new Real[3 * n_vertices];
			std::memset(*face_idx, NULL, sizeof(uint) * n_vertices);
			std::memset(*vertexArray, NULL, sizeof(Real) * 3 * n_vertices);
			std::memset(*colorArray, NULL, sizeof(Real) * 3 * n_vertices);

			//parse Triangle Mesh Data
			for (size_t idxE = 0; idxE < elements.size(); ++idxE) {
				for (longlong e = 0; e < elements[idxE].size; ++e) {
					std::getline(File, line);
					lineStr.str(line);
					std::string val;

					//color channel parsing
					if (ok_channel && (idxE == idxE_color)) {
						lineStr >> val;
						color_channels = std::stoi(val);
					}

					//vertex data parsing
					if (idxE == idxE_vertex) {
						uint face = 0;
						Real x = 0.f;
						Real y = 0.f;
						Real z = 0.f;
						uchar red = 0;
						uchar green = 0;
						uchar blue = 0;

						//line Processing
						for (int p = 0; p < elements[idxE].properties.size(); ++p) {
							lineStr >> val;
							if (p == idxP_face_idx) face = std::stoul(val);
							if (p == idxP_x) x = std::stof(val);
							else if (p == idxP_y) y = std::stof(val);
							else if (p == idxP_z) z = std::stof(val);
							else if (p == idxP_red) red = std::stoi(val);
							else if (p == idxP_green) green = std::stoi(val);
							else if (p == idxP_blue) blue = std::stoi(val);
						}

						(*face_idx)[e] = face;
						(*vertexArray)[3 * e + 0] = x;
						(*vertexArray)[3 * e + 1] = y;
						(*vertexArray)[3 * e + 2] = z;
						(*colorArray)[3 * e + 0] = (Real)(red / 255.f);
						(*colorArray)[3 * e + 1] = (Real)(green / 255.f);
						(*colorArray)[3 * e + 2] = (Real)(blue / 255.f);
					}
				}
			}
			File.close();

			if (ok_channel && (color_channels == 1)) {
				Real* grayArray = new Real[n_vertices];
				for (ulonglong i = 0; i < n_vertices; ++i) {
					grayArray[i] = (*colorArray)[3 * i];
				}
				delete[](*colorArray);
				*colorArray = grayArray;
			}
			else if (!ok_channel) {
				bool check = false;
				for (ulonglong i = 0; i < n_vertices; ++i) {
					if (((*colorArray)[3 * i + 0] != (*colorArray)[3 * i + 1]) || ((*colorArray)[3 * i + 1] != (*colorArray)[3 * i + 2])) {
						check = true;
						break;
					}
				}

				if (check) color_channels = 3;
				else if (!check) {
					color_channels = 1;
					Real* grayArray = new Real[n_vertices];
					for (ulonglong i = 0; i < n_vertices; ++i) {
						grayArray[i] = (*colorArray)[3 * i];
					}
					delete[](*colorArray);
					*colorArray = grayArray;
				}
			}

			std::cout << "Success loading " << n_vertices/3 << " Triangle Mesh, Color Channels : " << color_channels << std::endl;

			return true;
		}
	}
	else {
		std::cerr << "Error : Failed loading ply file..." << std::endl;
		return false;
	}
}

bool PLYparser::savePLY(const char* fileName, const ulonglong n_vertices, const int color_channels, uint * face_idx, Real * vertexArray, Real * colorArray)
{
	if ((vertexArray == nullptr) || (colorArray == nullptr)) {
		std::cerr << "Error : There is not data for saving ply file..." << std::endl;
		return false;
	}
	if ((color_channels != 1) && (color_channels != 3)) {
		std::cerr << "Error : Number of Color channels for saving ply file is false value..." << std::endl;
		return false;
	}

	std::string outputPath = fileName;
	if ((outputPath.find(".ply") == std::string::npos) && (outputPath.find(".PLY") == std::string::npos)) outputPath += ".ply";
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

		File << color_channels << std::endl;

		for (ulonglong i = 0; i < n_vertices; ++i) {
			//Vertex Face Index
			File << std::fixed << face_idx[i] << " ";

			//Vertex Geometry
			File << std::fixed << vertexArray[3 * i + 0] << " " << vertexArray[3 * i + 1] << " " << vertexArray[3 * i + 2] << " ";

			//Color Amplitude
			if (color_channels == 3)
				File << (int)(255.f*colorArray[3 * i + 0] + 0.5f) << " " << (int)(255.f*colorArray[3 * i + 1] + 0.5f) << " " << (int)(255.f*colorArray[3 * i + 2] + 0.5f) << std::endl;
			else if (color_channels == 1) {
				int indensity = (int)(255.f*colorArray[i] + 0.5f);
				File << indensity << " " << indensity << " " << indensity << std::endl;
			}
		}
		File.close();
		return true;
	}
	else {
		std::cerr << "Error : Failed saving ply file..." << std::endl;
		return false;
	}
}
