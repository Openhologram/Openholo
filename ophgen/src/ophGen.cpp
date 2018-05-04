#include "ophGen.h"
#include <windows.h>
#include "sys.h"

ophGen::ophGen(void)
{
}

ophGen::~ophGen(void)
{
}

int ophGen::loadPointCloud(const std::string pc_file, std::vector<real> *vertex_array, std::vector<real> *amplitude_array, std::vector<real> *phase_array)
{
	std::ifstream File(pc_file, std::ios::in);
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
		real pX, pY, pZ, phase, amplitude;
		std::getline(File, Line);
		sscanf_s(Line.c_str(), "%d %lf %lf %lf %lf %lf\n", &idx, &pX, &pY, &pZ, &phase, &amplitude);

		if (idx == i) {
			if (vertex_array)
			{
				vertex_array->push_back(pX);
				vertex_array->push_back(pY);
				vertex_array->push_back(pZ);
			}

			if (amplitude_array)
				amplitude_array->push_back(phase);

			if (phase_array)
				phase_array->push_back(amplitude);
		}
		else {
			File.close();
			return -1;
		}
	}
	File.close();
	return n_pts;
}

bool ophGen::readConfig(const std::string fname, OphPointCloudConfig& configdata)
{
	std::ifstream File(fname, std::ios::in);
	if (!File.is_open()) {
		File.close();
		return false;
	}

	std::vector<std::string> Title, Value;
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

	configdata.scale.v[0] = stod(Value[0]);
	configdata.scale.v[1] = stod(Value[1]);
	configdata.scale.v[2] = stod(Value[2]);

	configdata.offset_depth = stod(Value[3]);

	context_.pixel_pitch.v[0] = stod(Value[4]);
	context_.pixel_pitch.v[1] = stod(Value[5]);

	context_.pixel_number.v[0] = stod(Value[6]);
	context_.pixel_number.v[1] = stod(Value[7]);

	context_.ss[0] = context_.pixel_number.v[0] * context_.pixel_pitch.v[0];
	context_.ss[1] = context_.pixel_number.v[1] * context_.pixel_pitch.v[1];

	configdata.filter_shape_flag = (signed char*)Value[8].c_str();

	configdata.filter_width.v[0] = stod(Value[9]);
	configdata.filter_width.v[1] = stod(Value[10]);

	configdata.focal_length_lens_in = stod(Value[11]);
	configdata.focal_length_lens_out = stod(Value[12]);
	configdata.focal_length_lens_eye_piece = stod(Value[13]);

	context_.lambda = stod(Value[14]);
	context_.k = (2 * M_PI) / context_.lambda;

	configdata.tilt_angle.v[0] = stod(Value[15]);
	configdata.tilt_angle.v[1] = stod(Value[16]);

	File.close();
	return true;
}

bool ophGen::readConfig(const std::string fname, OphDepthMapConfig & config, OphDepthMapParams& params, OphDepthMapSimul& simuls)
{
	std::string inputFileName_ = "config_openholo.txt";

	LOG("Reading....%s\n", inputFileName_.c_str());

	std::ifstream inFile(inputFileName_.c_str());

	if (!inFile.is_open()) {
		LOG("file not found.\n");
		return false;
	}

	// skip 7 lines
	std::string temp;
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');
	getline(inFile, temp, '\n');

	inFile >> params.SOURCE_FOLDER;									getline(inFile, temp, '\n');
	inFile >> params.IMAGE_PREFIX;									getline(inFile, temp, '\n');
	inFile >> params.DEPTH_PREFIX;									getline(inFile, temp, '\n');
	inFile >> params.RESULT_FOLDER;									getline(inFile, temp, '\n');
	inFile >> params.RESULT_PREFIX;									getline(inFile, temp, '\n');
	inFile >> params.FLAG_STATIC_IMAGE;								getline(inFile, temp, '\n');
	inFile >> params.START_OF_FRAME_NUMBERING;						getline(inFile, temp, '\n');
	inFile >> params.NUMBER_OF_FRAME;								getline(inFile, temp, '\n');
	inFile >> params.NUMBER_OF_DIGIT_OF_FRAME_NUMBERING;			getline(inFile, temp, '\n');

	// skip 3 lines
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');

	inFile >> params.Transform_Method_;								getline(inFile, temp, '\n');
	inFile >> params.Propagation_Method_;							getline(inFile, temp, '\n');
	inFile >> params.Encoding_Method_;								getline(inFile, temp, '\n');

	// skip 3 lines
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');

	inFile >> config.field_lens;									getline(inFile, temp, '\n');
	inFile >> context_.lambda;									getline(inFile, temp, '\n');
	context_.k = 2 * M_PI / context_.lambda;

	inFile >> context_.pixel_number[0];								getline(inFile, temp, '\n');
	inFile >> context_.pixel_number[1];								getline(inFile, temp, '\n');

	inFile >> context_.pixel_pitch[0];								getline(inFile, temp, '\n');
	inFile >> context_.pixel_pitch[1];								getline(inFile, temp, '\n');

	context_.ss[0] = context_.pixel_pitch[0] * context_.pixel_number[0];
	context_.ss[1] = context_.pixel_pitch[1] * context_.pixel_number[1];

	// skip 3 lines
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');

	real NEAR_OF_DEPTH_MAP, FAR_OF_DEPTH_MAP;
	inFile >> NEAR_OF_DEPTH_MAP;									getline(inFile, temp, '\n');
	inFile >> FAR_OF_DEPTH_MAP;										getline(inFile, temp, '\n');

	config.near_depthmap = min(NEAR_OF_DEPTH_MAP, FAR_OF_DEPTH_MAP);
	config.far_depthmap = max(NEAR_OF_DEPTH_MAP, FAR_OF_DEPTH_MAP);

	inFile >> params.FLAG_CHANGE_DEPTH_QUANTIZATION;				getline(inFile, temp, '\n');
	inFile >> params.DEFAULT_DEPTH_QUANTIZATION;					getline(inFile, temp, '\n');
	inFile >> params.NUMBER_OF_DEPTH_QUANTIZATION;					getline(inFile, temp, '\n');

	if (params.FLAG_CHANGE_DEPTH_QUANTIZATION == 0)
		config.num_of_depth = params.DEFAULT_DEPTH_QUANTIZATION;
	else
		config.num_of_depth = params.NUMBER_OF_DEPTH_QUANTIZATION;

	inFile >> temp;
	std::size_t found = temp.find(':');
	if (found != std::string::npos)
	{
		std::string s = temp.substr(0, found);
		std::string e = temp.substr(found + 1);
		int start = std::stoi(s);
		int end = std::stoi(e);
		config.render_depth.clear();
		for (int k = start; k <= end; k++)
			config.render_depth.push_back(k);

	}
	else {

		config.render_depth.clear();
		config.render_depth.push_back(std::stoi(temp));
		inFile >> temp;

		while (temp.find('/') == std::string::npos)
		{
			config.render_depth.push_back(std::stoi(temp));
			inFile >> temp;
		}
	}
	if (config.render_depth.empty()) {
		LOG("Error: RENDER_DEPTH \n");
		return false;
	}

	getline(inFile, temp, '\n');
	inFile >> params.RANDOM_PHASE;									getline(inFile, temp, '\n');

	//==Simulation parameters ======================================================================
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');

	inFile >> simuls.Simulation_Result_File_Prefix_;				getline(inFile, temp, '\n');
	inFile >> simuls.test_pixel_number_scale_;						getline(inFile, temp, '\n');
	inFile >> simuls.eye_length_;									getline(inFile, temp, '\n');
	inFile >> simuls.eye_pupil_diameter_;							getline(inFile, temp, '\n');
	inFile >> simuls.eye_center_xy_[0];								getline(inFile, temp, '\n');
	inFile >> simuls.eye_center_xy_[1];								getline(inFile, temp, '\n');
	inFile >> simuls.focus_distance_;								getline(inFile, temp, '\n');

	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');

	inFile >> simuls.sim_type_;										getline(inFile, temp, '\n');
	inFile >> simuls.sim_from_;										getline(inFile, temp, '\n');
	inFile >> simuls.sim_to_;										getline(inFile, temp, '\n');
	inFile >> simuls.sim_step_num_;									getline(inFile, temp, '\n');

	//=====================================================================================
	inFile.close();

	LOG("done\n");

	return true;
}
