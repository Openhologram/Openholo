#include "ophGen.h"
#include <windows.h>
#include "sys.h"
#include "function.h"

ophGen::ophGen(void)
	: Openholo()
	, holo_gen(nullptr)
	, holo_encoded(nullptr)
	, holo_normalized(nullptr)
{
}

ophGen::~ophGen(void)
{
}

int ophGen::loadPointCloud(const char* pc_file, std::vector<real> *vertex_array, std::vector<real> *amplitude_array, std::vector<real> *phase_array)
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
				vertex_array->reserve(n_pts * 3);
				vertex_array->push_back(pX);
				vertex_array->push_back(pY);
				vertex_array->push_back(pZ);
			}

			if (amplitude_array)
			{
				amplitude_array->reserve(n_pts);
				amplitude_array->push_back(phase);
			}

			if (phase_array)
			{
				phase_array->reserve(n_pts);
				phase_array->push_back(amplitude);
			}
		}
		else {
			File.close();
			return -1;
		}
	}
	File.close();
	return n_pts;
}

bool ophGen::readConfig(const char* fname, OphPointCloudConfig& configdata)
{
	std::ifstream inFile(fname, std::ios::in);
	if (!inFile.is_open()) {
		LOG("file not found.\n");
		inFile.close();
		return false;
	}

	std::vector<std::string> Title, Value;
	std::string Line;
	std::stringstream LineStream;

	int i = 0;
	while (std::getline(inFile, Line)) {
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
		inFile.close();
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

	inFile.close();
	return true;
}

bool ophGen::readConfig(const char* fname, OphDepthMapConfig & config, OphDepthMapParams& params, OphDepthMapSimul& simuls)
{
	std::string inputFileName_ = fname;

	LOG("Reading....%s\n", fname);

	std::ifstream inFile(fname);

	if (!inFile.is_open()) {
		LOG("file not found.\n");
		inFile.close();
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

void ophGen::normalize(const int frame)
{
	oph::normalize((real*)holo_encoded, holo_normalized, context_.pixel_number[_X], context_.pixel_number[_Y], frame);
}

int ophGen::save(const char * fname, uint8_t bitsperpixel, uchar* src, uint px, uint py)
{
	if (fname == nullptr) return -1;

	uchar* source = src;
	ivec2 p(px, py);

	if (src == nullptr)
		source = holo_normalized;
	if (px == 0 && py == 0)
		p = ivec2(context_.pixel_number[_X], context_.pixel_number[_Y]);

	if (checkExtension(fname, ".ohf"))	// save as *.ohf
		return Openholo::saveAsOhf(fname, bitsperpixel, source, p[_X], p[_Y]);
	else {										// save as image file - (bmp)
		if (checkExtension(fname, ".bmp")) 	// when the extension is bmp
			return Openholo::saveAsImg(fname, bitsperpixel, source, p[_X], p[_Y]);
		else {									// when extension is not .ohf, .bmp - force bmp
			char buf[256];
			memset(buf, 0x00, sizeof(char) * 256);
			sprintf_s(buf, "%s.bmp", fname);

			return Openholo::saveAsImg(buf, bitsperpixel, source, p[_X], p[_Y]);
		}
	}
}

int ophGen::save(const char * fname, uint8_t bitsperpixel, uint px, uint py, uint fnum, uchar* args ...)
{
	std::string file = fname;
	std::string name;
	std::string ext;

	size_t ex = file.rfind(".");
	if (ex == -1) ex = file.length();
	 
	name = file.substr(0, ex);
	ext = file.substr(ex, file.length() - 1);

	va_list ap;
	__crt_va_start(ap, args);

	for (uint i = 0; i < fnum; i++) {
		name.append(std::to_string(i)).append(ext);
		if (i == 0) {
			save(name.c_str(), bitsperpixel, args, px, py);
			continue;
		}
		uchar* data = __crt_va_arg(ap, uchar*);
		save(name.c_str(), bitsperpixel, data, px, py);
	}

	__crt_va_end(ap);

	return 0;
}

int ophGen::save(const char * fname, uint8_t bitsperpixel, uint px, uint py, uint fnum, uchar* args ...)
{
	std::string file = fname;
	std::string name;
	std::string ext;

	size_t ex = file.rfind(".");
	if (ex == -1) ex = file.length();
	 
	name = file.substr(0, ex);
	ext = file.substr(ex, file.length() - 1);

	va_list ap;
	__crt_va_start(ap, args);

	for (uint i = 0; i < fnum; i++) {
		name.append(std::to_string(i)).append(ext);
		if (i == 0) {
			save(name.c_str(), bitsperpixel, args, px, py);
			continue;
		}
		uchar* data = __crt_va_arg(ap, uchar*);
		save(name.c_str(), bitsperpixel, data, px, py);
	}

	__crt_va_end(ap);

	return 0;
}


int ophGen::load(const char * fname, void * dst)
{
	if (holo_normalized != nullptr) {
		delete[] holo_normalized;
	}

	if (checkExtension(fname, ".ohf")) {
		if (dst != nullptr)
			return Openholo::loadAsOhf(fname, dst);
		else
			return Openholo::loadAsOhf(fname, holo_normalized);
	} 
	else {
		if (checkExtension(fname, ".bmp"))
		{
			if (dst != nullptr)
				return Openholo::loadAsImg(fname, dst);
			else
				return Openholo::loadAsImg(fname, holo_normalized);
		}
		else			// when extension is not .ohf, .bmp
		{
			// how to load another image file format?
		}
	}

	return 0;
}

#define for_i(itr, oper) for(int i=0; i<itr; i++){ oper }

void ophGen::calPhase(oph::Complex<real>* holo, real* encoded, const vec2 holosize)
{
	int size = (int)holosize.v[0] * holosize.v[1];
	for_i(size, 
		oph::angle<real>(*(holo + i), *(encoded + i));
	);
}
void ophGen::calAmplitude(oph::Complex<real>* holo, real* encoded, const vec2 holosize) {
	int size = (int)holosize.v[0] * holosize.v[1];
	oph::absCplxArr<real>(holo, encoded, size);
}

void ophGen::numericalInterference(oph::Complex<real>* holo, real* encoded, const vec2 holosize)
{
	int size = (int) holosize.v[0] * holosize.v[1];
	
	real* temp1 = new real[size];
	oph::absCplxArr<real>(holo, temp1, size);
	
	real* ref = new real;
	*ref = oph::maxOfArr<real>(temp1, size);

	oph::Complex<real>* temp2 = new oph::Complex<real>[size];
	temp2 = holo;
	for_i(size,
		temp2[i].re += *ref;
	);

	oph::absCplxArr<real>(temp2, encoded, size);

	delete[] temp1, temp2;
	delete ref;
}

void ophGen::numericalInterference(void)
{
	int size = (int)context_.pixel_number[0] * context_.pixel_number[1];

	real* temp1 = new real[size];
	oph::absCplxArr<real>(holo_gen, temp1, size);

	real* ref = new real;
	*ref = oph::maxOfArr<real>(temp1, size);

	oph::Complex<real>* temp2 = new oph::Complex<real>[size];
	temp2 = holo_gen;
	for_i(size, 
		temp2[i].re += *ref;
	);

	oph::absCplxArr<real>(temp2, holo_encoded, size);

	delete[] temp1, temp2;
	delete ref;
}
/*
void ophGen::singleSideBand(oph::Complex<real>* holo, real* encoded, const vec2 holosize, int passband)
{
	int size = (int)holosize.v[0] * holosize.v[1];
	
	oph::Complex<real>* AS = new oph::Complex<real>[size];
	fft2((int)holosize.v[0], (int)holosize.v[1], holo, AS, sign);

	switch (passband)
	{
	case left:
		for (int i = 0; i < (int)holosize.v[1]; i++)
		{
			for (int j = (int)holosize.v[0] / 2; j < (int)holosize.v[0]; j++)
			{ AS[i*(int)holosize.v[0] + j] = 0; }
		}
	case rig:
		for (int i = 0; i < (int)holosize.v[1]; i++)
		{
			for (int j = 0; j < (int)holosize.v[0] / 2; j++)
			{ AS[i*(int)holosize.v[0] + j] = 0; }
		}
	case top:
		for (int i = size / 2; i < size; i++)
		{
			AS[i] = 0;
		}
	case btm:
		for (int i = 0; i < size / 2; i++)
		{
			AS[i] = 0;
		}
	}

	oph::Complex<real>* filtered = new oph::Complex<real>[size];
	fft2((int)holosize.v[0], (int)holosize.v[1], AS, filtered, sign);

	real* realPart = new real[size];
	oph::realPart<real>(filtered, realPart, size);

	real *minReal = new real;
	*minReal = oph::minOfArr(realPart, size);

	real* realPos = new real[size];
	for_i(size, 
		*(realPos + i) = *(realPart + i) - *minReal;
	);

	real *maxReal = new real;
	*maxReal = oph::maxOfArr(realPos, size);

	for_i(size,
		*(encoded + i) = *(realPos + i) / *maxReal;
	);

	delete[] AS, filtered, realPart, realPos;
	delete maxReal, minReal;
}
*/
void ophGen::twoPhaseEncoding(oph::Complex<real>* holo, real* encoded, const vec2 holosize)
{
	int size = (int)holosize.v[0] * holosize.v[1];

	Complex<real>* normCplx = new Complex<real>[size];
	oph::normalize<real>(holo, normCplx, size);

	real* amplitude = new real[size];
	calAmplitude(normCplx, amplitude, holosize);
	real* phase = new real[size];
	calPhase(normCplx, phase, holosize);
	for_i(size, *(phase + i) += M_PI;);
	real* delPhase = new real[size];
	for_i(size, *(delPhase + i) = acos(*(amplitude + i)););

	for_i(size,
		*(encoded + i * 2) = *(phase + i) + *(delPhase + i);
		*(encoded + i * 2 + 1) = *(phase + i) - *(delPhase + i);
		);

	delete[] normCplx, amplitude, phase, delPhase;
}

void ophGen::burckhardt(oph::Complex<real>* holo, real* encoded, const vec2 holosize)
{
	int size = (int)holosize.v[0] * holosize.v[1];

	Complex<real>* norm = new Complex<real>[size];
	oph::normalize(holo, norm, size);
	real* phase = new real[size];
	calPhase(holo, phase, size);
	real* ampl = new real[size];
	calAmplitude(holo, ampl, size);

	real* A1 = new real[size];
	memsetArr<real>(A1, 0, 0, size - 1);
	real* A2 = new real[size];
	memsetArr<real>(A2, 0, 0, size - 1);
	real* A3 = new real[size];
	memsetArr<real>(A3, 0, 0, size - 1);

	for_i(size,
		if (*(phase + i) >= 0 && *(phase + i) < (2 * M_PI / 3))
		{
			*(A1 + i) = *(ampl + i)*(cos(*(phase + i)) + sin(*(phase + i)) / sqrt(3));
			*(A2 + i) = 2 * sin(*(phase + i)) / sqrt(3);
		}
		else if (*(phase + i) >= (2 * M_PI / 3) && *(phase + i) < (4 * M_PI / 3))
		{
			*(A2 + i) = *(ampl + i)*(cos(*(phase + i) - (2 * M_PI / 3)) + sin(*(phase + i) - (2 * M_PI / 3)) / sqrt(3));
			*(A3 + i) = 2 * sin(*(phase + i) - (2 * M_PI / 3)) / sqrt(3);
		}
		else if (*(phase + i) >= (4 * M_PI / 3) && *(phase + i) < (2 * M_PI))
		{
			*(A3 + i) = *(ampl + i)*(cos(*(phase + i) - (4 * M_PI / 3)) + sin(*(phase + i) - (4 * M_PI / 3)) / sqrt(3));
			*(A1 + i) = 2 * sin(*(phase + i) - (4 * M_PI / 3)) / sqrt(3);
		}
	);

	for_i(size,
		*(encoded + (3 * i)) = *(A1 + i);
		*(encoded + (3 * i + 1)) = *(A2 + i);
		*(encoded + (3 * i + 2)) = *(A3 + i);
	);
}
/*
void ophGen::freqShift(oph::Complex<real>* holo, Complex<real>* encoded, const vec2 holosize, int shift_x, int shift_y)
{
	int size = (int)holosize.v[0] * holosize.v[1];

	oph::Complex<real>* AS = new oph::Complex<real>[size];
	fft2((int)holosize.v[0], (int)holosize.v[1], holo, AS, sign);
	oph::Complex<real>* shifted = new oph::Complex<real>[size];
	circshift<Complex<real>>(AS, shifted, shift_x, shift_y, holosize.v[0], holosize.v[1]);
	fft2((int)holosize.v[0], (int)holosize.v[1], shifted, encoded, sign);
}
*/
void ophGen::fft2(int n0, int n1, const oph::Complex<real>* in, oph::Complex<real>* out, int sign, unsigned int flag)
{
	int pnx, pny;
	n0 == 0 ? pnx = context_.pixel_number[_X] : pnx = n0;
	n1 == 0 ? pny = context_.pixel_number[_Y] : pnx = n1;

	fftw_complex *fft_in, *fft_out;
	fftw_plan plan;

	fft_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * pnx * pny);
	fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * pnx * pny);

	for (int i = 0; i < pnx * pny; i++) {
		fft_in[i][0] = in[i].re;
		fft_in[i][1] = in[i].im;
	}

	plan = fftw_plan_dft_2d(pnx, pny, fft_in, fft_out, sign, flag);

	fftw_execute(plan);

	for (int i = 0; i < pnx * pny; i++) {
		out[i].re = fft_out[i][0];
		out[i].im = fft_out[i][1];
	}

	fftw_destroy_plan(plan);
	fftw_free(fft_in);
	fftw_free(fft_out);
}

void ophGen::ophFree(void)
{
	if (holo_gen) delete[] holo_gen;
	if (holo_encoded) delete[] holo_encoded;
	if (holo_normalized) delete[] holo_normalized;
}