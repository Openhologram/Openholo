#ifndef __function_h
#define __function_h

namespace oph
{
	//int LoadPointCloudData(const std::string InputModelFile, std::vector<float> &VertexArray, std::vector<float> &AmplitudeArray, std::vector<float> &PhaseArray)
	//{
	//	std::ifstream File(InputModelFile, std::ios::in);
	//	if (!File.is_open()) {
	//		File.close();
	//		return -1;
	//	}

	//	std::string Line;
	//	std::getline(File, Line);
	//	int n_pts = atoi(Line.c_str());

	//	// parse input point cloud file
	//	for (int i = 0; i < n_pts; ++i) {
	//		int idx;
	//		float pX, pY, pZ, phase, amplitude;
	//		std::getline(File, Line);
	//		sscanf_s(Line.c_str(), "%d %f %f %f %f %f\n", &idx, &pX, &pY, &pZ, &phase, &amplitude);

	//		if (idx == i) {
	//			VertexArray.push_back(pX);
	//			VertexArray.push_back(pY);
	//			VertexArray.push_back(pZ);
	//			PhaseArray.push_back(phase);
	//			AmplitudeArray.push_back(amplitude);
	//		}
	//		else {
	//			File.close();
	//			return -1;
	//		}
	//	}
	//	File.close();
	//	return n_pts;
	//}

	//bool ophReadConfigFile(const std::string InputConfigFile, oph::OphConfigParams &configParams)
	//{
	//	std::ifstream File(InputConfigFile, std::ios::in);
	//	if (!File.is_open()) {
	//		File.close();
	//		return false;
	//	}

	//	std::vector<std::string> Title;
	//	std::vector<std::string> Value;
	//	std::string Line;
	//	std::stringstream LineStream;

	//	int i = 0;
	//	while (std::getline(File, Line)) {
	//		std::string _Title;
	//		std::string _Value;
	//		std::string _Equal; // " = "
	//		LineStream << Line;
	//		LineStream >> _Title >> _Equal >> _Value;
	//		LineStream.clear();

	//		Title.push_back(_Title);
	//		Value.push_back(_Value);
	//		++i;
	//	}

	//	if (i != 17) {
	//		File.close();
	//		return false;
	//	}

	//	configParams.pointCloudScaleX = atof(Value[0].c_str());
	//	configParams.pointCloudScaleY = atof(Value[1].c_str());
	//	configParams.pointCloudScaleZ = atof(Value[2].c_str());
	//	configParams.offsetDepth = atof(Value[3].c_str());
	//	configParams.samplingPitchX = atof(Value[4].c_str());
	//	configParams.samplingPitchY = atof(Value[5].c_str());
	//	configParams.nx = atoi(Value[6].c_str());
	//	configParams.ny = atoi(Value[7].c_str());
	//	configParams.filterShapeFlag = (char*)Value[8].c_str();
	//	configParams.filterXwidth = atof(Value[9].c_str());
	//	configParams.filterYwidth = atof(Value[10].c_str());
	//	configParams.focalLengthLensIn = atof(Value[11].c_str());
	//	configParams.focalLengthLensOut = atof(Value[12].c_str());
	//	configParams.focalLengthLensEyePiece = atof(Value[13].c_str());
	//	configParams.lambda = atof(Value[14].c_str());
	//	configParams.tiltAngleX = atof(Value[15].c_str());
	//	configParams.tiltAngleY = atof(Value[16].c_str());
	//	File.close();
	//	return true;
	//}
}

#endif // !__function_h
