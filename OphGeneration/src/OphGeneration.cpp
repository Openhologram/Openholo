#include "OphGeneration.h"

OphGeneration::OphGeneration(void)
{
}

OphGeneration::~OphGeneration(void)
{
}

int OphGeneration::loadPointCloudData(const std::string InputModelFile, std::vector<float>* VertexArray, std::vector<float>* AmplitudeArray, std::vector<float>* PhaseArray)
{
	std::ifstream File(InputModelFile, std::ios::in);
	if (!File.is_open()) {
		File.close();
		return -1;
	}

	std::string Line;
	std::getline(File, Line);
	int n_pts = atoi(Line.c_str());

	// parse input point cloud file
	for (int i = 0; i < n_pts; ++i) {
		int idx;
		float pX, pY, pZ, phase, amplitude;
		std::getline(File, Line);
		sscanf_s(Line.c_str(), "%d %f %f %f %f %f\n", &idx, &pX, &pY, &pZ, &phase, &amplitude);

		if (idx == i) {
			if (VertexArray)
			{
				VertexArray->push_back(pX);
				VertexArray->push_back(pY);
				VertexArray->push_back(pZ);
			}

			if (PhaseArray) 
				PhaseArray->push_back(phase);

			if (AmplitudeArray) 
				AmplitudeArray->push_back(amplitude);
		}
		else {
			File.close();
			return -1;
		}
	}
	File.close();
	return n_pts;
}

bool OphGeneration::readConfigFile(const std::string InputConfigFile, oph::OphConfigParams & configParams)
{
	std::ifstream File(InputConfigFile, std::ios::in);
	if (!File.is_open()) {
		File.close();
		return false;
	}

	std::vector<std::string> Title;
	std::vector<std::string> Value;
	std::string Line;
	std::stringstream LineStream;

	int i = 0;
	while (std::getline(File, Line)) {
		std::string _Title;
		std::string _Value;
		std::string _Equal; // " = "
		LineStream << Line;
		LineStream >> _Title >> _Equal >> _Value;
		LineStream.clear();

		Title.push_back(_Title);
		Value.push_back(_Value);
		++i;
	}

	if (i != 17) {
		File.close();
		return false;
	}

	configParams.pointCloudScaleX = static_cast<float>(atof(Value[0].c_str()));
	configParams.pointCloudScaleY = static_cast<float>(atof(Value[1].c_str()));
	configParams.pointCloudScaleZ = static_cast<float>(atof(Value[2].c_str()));
	configParams.offsetDepth = static_cast<float>(atof(Value[3].c_str()));
	configParams.samplingPitchX = static_cast<float>(atof(Value[4].c_str()));
	configParams.samplingPitchY = static_cast<float>(atof(Value[5].c_str()));
	configParams.nx = atoi(Value[6].c_str());
	configParams.ny = atoi(Value[7].c_str());
	configParams.filterShapeFlag = (char*)Value[8].c_str();
	configParams.filterXwidth = static_cast<float>(atof(Value[9].c_str()));
	configParams.filterYwidth = static_cast<float>(atof(Value[10].c_str()));
	configParams.focalLengthLensIn = static_cast<float>(atof(Value[11].c_str()));
	configParams.focalLengthLensOut = static_cast<float>(atof(Value[12].c_str()));
	configParams.focalLengthLensEyePiece = static_cast<float>(atof(Value[13].c_str()));
	configParams.lambda = static_cast<float>(atof(Value[14].c_str()));
	configParams.tiltAngleX = static_cast<float>(atof(Value[15].c_str()));
	configParams.tiltAngleY = static_cast<float>(atof(Value[16].c_str()));
	File.close();
	return true;
}