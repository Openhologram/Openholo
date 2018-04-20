#include "ophGen.h"

ophGen::ophGen(void)
{
}

ophGen::~ophGen(void)
{
}

int ophGen::loadPointCloudData(const std::string InputModelFile, std::vector<float>* VertexArray, std::vector<float>* AmplitudeArray, std::vector<float>* PhaseArray)
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

bool ophGen::readConfigFile(const std::string InputConfigFile, oph::ConfigParams & configParams)
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

	configParams.pointCloudScaleX = stof(Value[0]);
	configParams.pointCloudScaleY = stof(Value[1]);
	configParams.pointCloudScaleZ = stof(Value[2]);
	configParams.offsetDepth = stof(Value[3]);
	configParams.samplingPitchX = stof(Value[4]);
	configParams.samplingPitchY = stof(Value[5]);
	configParams.nx = stoi(Value[6]);
	configParams.ny = stoi(Value[7]);
	configParams.filterShapeFlag = (char*)Value[8].c_str();
	configParams.filterXwidth = stof(Value[9]);
	configParams.filterYwidth = stof(Value[10]);
	configParams.focalLengthLensIn = stof(Value[11]);
	configParams.focalLengthLensOut = stof(Value[12]);
	configParams.focalLengthLensEyePiece = stof(Value[13]);
	configParams.lambda = stof(Value[14]);
	configParams.tiltAngleX = stof(Value[15]);
	configParams.tiltAngleY = stof(Value[16]);
	File.close();
	return true;
}