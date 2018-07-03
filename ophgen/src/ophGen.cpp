#include "ophGen.h"
#include <windows.h>
#include "sys.h"
#include "function.h"
#include <cuda_runtime.h>
#include <cufft.h>

#include "encoding.h"

ophGen::ophGen(void)
	: Openholo()
	, holo_gen(nullptr)
	, holo_encoded(nullptr)
	, holo_normalized(nullptr)
{
	context_ = { 0 };
}

ophGen::~ophGen(void)
{
}

int ophGen::loadPointCloud(const char* pc_file, OphPointCloudData *pc_data_, uint flag)
{
	LOG("Reading....%s...", pc_file);

	auto start = _cur_time;

	std::ifstream File(pc_file, std::ios::in);
	if (!File.is_open()) {
		File.close();
		return -1;
	}

	int n_pts;

	File >> n_pts;

	pc_data_->location	= new vec3[n_pts];
	pc_data_->color		= new vec3[n_pts];
	pc_data_->amplitude	= new Real[n_pts];
	pc_data_->phase		= new Real[n_pts];

	memset(pc_data_->location, NULL, sizeof(vec3) * n_pts);
	memset(pc_data_->color, NULL, sizeof(vec3) * n_pts);
	memset(pc_data_->amplitude, NULL, sizeof(Real) * n_pts);
	memset(pc_data_->phase, NULL, sizeof(Real) * n_pts);

	// parse input point cloud file
	for (int i = 0; i < n_pts; ++i) {
		int idx;
		Real pX, pY, pZ, phase, amplitude;
		int pR, pG, pB;

		File >> idx;
		if (idx == i) {
			if (flag & PC_XYZ){
				File >> pX >> pY >> pZ;
				pc_data_->location[idx][_X] = pX;
				pc_data_->location[idx][_Y] = pY;
				pc_data_->location[idx][_Z] = pY;
			}
			if (flag & PC_RGB){
				File >> pR >> pG >> pB;
				pc_data_->color[idx][_X] = pR;
				pc_data_->color[idx][_Y] = pG;
				pc_data_->color[idx][_Z] = pB;
			}
			if (flag & PC_PHASE){
				File >> phase;
				pc_data_->phase[idx] = phase;
			}
			if (flag & PC_AMPLITUDE){
				File >> amplitude;
				pc_data_->amplitude[idx] = amplitude;
			}
		}
		else {
			File.close();
			return -1;
		}
	}
	File.close();

	auto end = _cur_time;

	auto during = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("%.5lfsec...done\n", during);
	return n_pts;
}

bool ophGen::readConfig(const char* fname, OphPointCloudConfig& configdata)
{
	LOG("Reading....%s...", fname);

	auto start = _cur_time;

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
	context_.ss[_X] = context_.pixel_number[_X] * context_.pixel_pitch[_X];
	context_.ss[_Y] = context_.pixel_number[_Y] * context_.pixel_pitch[_Y];

	configdata.tilt_angle.v[0] = stod(Value[15]);
	configdata.tilt_angle.v[1] = stod(Value[16]);

	inFile.close();

	auto end = _cur_time;

	auto during = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("%.5lfsec...done\n", during);
	return true;
}

bool ophGen::readConfig(const char* fname, OphDepthMapConfig & config, OphDepthMapParams& params)
{
	std::string inputFileName_ = fname;

	LOG("Reading....%s...", fname);

	auto start = _cur_time;

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
	inFile >> context_.lambda;										getline(inFile, temp, '\n');
	context_.k = 2 * M_PI / context_.lambda;

	inFile >> context_.pixel_number[0];								getline(inFile, temp, '\n');
	inFile >> context_.pixel_number[1];								getline(inFile, temp, '\n');

	inFile >> context_.pixel_pitch[0];								getline(inFile, temp, '\n');
	inFile >> context_.pixel_pitch[1];								getline(inFile, temp, '\n');

	context_.ss[0] = context_.pixel_pitch[0] * context_.pixel_number[0];
	context_.ss[1] = context_.pixel_pitch[1] * context_.pixel_number[1];

	// skip 3 lines
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');

	Real NEAR_OF_DEPTH_MAP, FAR_OF_DEPTH_MAP;
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

	//inFile >> simuls.Simulation_Result_File_Prefix_;				getline(inFile, temp, '\n');
	//inFile >> simuls.test_pixel_number_scale_;						getline(inFile, temp, '\n');
	//inFile >> simuls.eye_length_;									getline(inFile, temp, '\n');
	//inFile >> simuls.eye_pupil_diameter_;							getline(inFile, temp, '\n');
	//inFile >> simuls.eye_center_xy_[0];								getline(inFile, temp, '\n');
	//inFile >> simuls.eye_center_xy_[1];								getline(inFile, temp, '\n');
	//inFile >> simuls.focus_distance_;								getline(inFile, temp, '\n');

	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');

	//inFile >> simuls.sim_type_;										getline(inFile, temp, '\n');
	//inFile >> simuls.sim_from_;										getline(inFile, temp, '\n');
	//inFile >> simuls.sim_to_;										getline(inFile, temp, '\n');
	//inFile >> simuls.sim_step_num_;									getline(inFile, temp, '\n');

	//=====================================================================================
	inFile.close();

	auto end = _cur_time;

	auto during = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("%.5lfsec...done\n", during);

	return true;
}

void ophGen::normalize(void)
{
	oph::normalize((Real*)holo_encoded, holo_normalized, context_.pixel_number[_X], context_.pixel_number[_Y]);
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

	if (checkExtension(fname, ".bmp")) 	// when the extension is bmp
		return Openholo::saveAsImg(fname, bitsperpixel, source, p[_X], p[_Y]);
	else {									// when extension is not .ohf, .bmp - force bmp
		char buf[256];
		memset(buf, 0x00, sizeof(char) * 256);
		sprintf_s(buf, "%s.bmp", fname);

		return Openholo::saveAsImg(buf, bitsperpixel, source, p[_X], p[_Y]);
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

int ophGen::load(const char * fname, void * dst)
{
	if (holo_normalized != nullptr) {
		delete[] holo_normalized;
	}

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

	return 0;
}

void ophGen::encoding(int px, int py, unsigned int flag) {
	int size = px*py;

	switch (flag)
	{
	case ENCODING_NUMERICAL_INTERFERENCE:
		numericalInterference(holo_gen, holo_encoded, size);
	case ENCODING_TWO_PHASE:
		twoPhaseEncoding(holo_gen, holo_encoded, size);
	case ENCODING_BURCKHARDT:
		burckhardt(holo_gen, holo_encoded, size);
	/*case ENCODING_FREQ_SHIFT:
		freqShift(holo_gen, holo_encoded, size, shift_x, shift_y);*/
	}
}

//void ophGen::encoding(oph::Complex<Real>*holo, Real* encoded, const ivec2 holosize, unsigned int flag) {
//	int size = holosize.v[0] * holosize.v[1];
//
//	switch (flag)
//	{
//	case ENCODING_NUMERICAL_INTERFERENCE:
//		numericalInterference(holo, encoded, holosize);
//	case ENCODING_TWO_PHASE:
//		twoPhaseEncoding(holo, encoded, holosize);
//	}
//
//	
//}

void ophGen::calPhase(oph::Complex<Real>* holo, Real* encoded, const ivec2 holosize)
{
	int size = holosize.v[0] * holosize.v[1];
	for_i(size,
		oph::angle<Real>(*(holo + i), *(encoded + i));
	);
}
void ophGen::calAmplitude(oph::Complex<Real>* holo, Real* encoded, const ivec2 holosize) {
	int size = holosize.v[0] * holosize.v[1];
	oph::absCplxArr<Real>(holo, encoded, size);
}
void ophGen::numericalInterference(oph::Complex<Real>* holo, Real* encoded, const int size)
{
	Real* temp1 = new Real[size];
	oph::absCplxArr<Real>(holo, temp1, size);

	Real* ref = new Real;
	*ref = oph::maxOfArr<Real>(temp1, size);

	oph::Complex<Real>* temp2 = new oph::Complex<Real>[size];
	temp2 = holo;
	for_i(size,
		temp2[i][_RE] += *ref;
	);

	oph::absCplxArr<Real>(temp2, encoded, size);
	for (int i = size - 10; i < size; i++)	// for debugging
		cout << "result(" << i << "): " << *(encoded + i) << endl;

	delete[] temp1, temp2;
	delete ref;
}

void ophGen::twoPhaseEncoding(oph::Complex<Real>* holo, Real* encoded, const int size)
{
	Complex<Real>* normCplx = new Complex<Real>[size];
	oph::normalize<Real>(holo, normCplx, size);

	Real* amplitude = new Real[size];
	calAmplitude(normCplx, amplitude, size);

	Real* phase = new Real[size];
	calPhase(normCplx, phase, size);

	for_i(size, *(phase + i) += M_PI;);

	Real* delPhase = new Real[size];
	for_i(size, *(delPhase + i) = acos(*(amplitude + i)););

	for_i(size,
		*(encoded + i * 2) = *(phase + i) + *(delPhase + i);
	*(encoded + i * 2 + 1) = *(phase + i) - *(delPhase + i);
	);

	delete[] normCplx, amplitude, phase, delPhase;
}

void ophGen::burckhardt(oph::Complex<Real>* holo, Real* encoded, const int size)
{
	Complex<Real>* norm = new Complex<Real>[size];
	oph::normalize(holo, norm, size);
	for (int i = 0; i < size; i++) {				// for debugging
		cout << "norm(" << i << ": " << *(norm + i) << endl;
	}
	Real* phase = new Real[size];
	calPhase(holo, phase, size);

	for (int i = 0; i < size; i++) {				// for debugging
		cout << "phase(" << i << ": " << *(phase + i) << endl;
	}

	Real* ampl = new Real[size];
	calAmplitude(holo, ampl, size);

	Real* A1 = new Real[size];
	memsetArr<Real>(A1, 0, 0, size - 1);
	Real* A2 = new Real[size];
	memsetArr<Real>(A2, 0, 0, size - 1);
	Real* A3 = new Real[size];
	memsetArr<Real>(A3, 0, 0, size - 1);

	for_i(size,
		if (*(phase + i) >= 0 && *(phase + i) < (2 * M_PI / 3))
		{
			*(A1 + i) = *(ampl + i)*(cos(*(phase + i)) + sin(*(phase + i)) / sqrt(3));
			*(A2 + i) = 2 * sin(*(phase + i)) / sqrt(3);
			//cout << "A1,A2 : " << i << endl;
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
	for (int i = 0; i < size; i++) {				// for debugging
		cout << "A1(" << i << ": " << *(A1 + i) << endl;
		cout << "A2(" << i << ": " << *(A2 + i) << endl;
		cout << "A3(" << i << ": " << *(A3 + i) << endl;
	}
	for_i(size / 3,
		*(encoded + (3 * i)) = *(A1 + i);
	*(encoded + (3 * i + 1)) = *(A2 + i);
	*(encoded + (3 * i + 2)) = *(A3 + i);
	);
	for (int i = 0; i < size; i++) {				// for debugging
		cout << "encoded(" << i << ": " << *(encoded + i) << endl;
	}
}

void ophGen::freqShift(oph::Complex<Real>* holo, Complex<Real>* encoded, const ivec2 holosize, int shift_x, int shift_y)
{
	int size = holosize.v[0] * holosize.v[1];

	oph::Complex<Real>* AS = new oph::Complex<Real>[size];
	fft2(holosize, holo, AS, OPH_FORWARD);
	oph::Complex<Real>* shifted = new oph::Complex<Real>[size];
	circShift<Complex<Real>>(AS, shifted, shift_x, shift_y, holosize.v[0], holosize.v[1]);
	fft2(holosize, shifted, encoded, OPH_BACKWARD);
}


void ophGen::encodeSideBand(bool bCPU, ivec2 sig_location)
{
	if (holo_gen == nullptr) {
		LOG("Not found diffracted data.");
		return;
	}

	int pnx = context_.pixel_number[_X];
	int pny = context_.pixel_number[_Y];

	int cropx1, cropx2, cropx, cropy1, cropy2, cropy;
	if (sig_location[1] == 0) { //Left or right half
		cropy1 = 1;
		cropy2 = pny;
	}
	else {
		cropy = (int)floor(((Real)pny) / 2);
		cropy1 = cropy - (int)floor(((Real)cropy) / 2);
		cropy2 = cropy1 + cropy - 1;
	}

	if (sig_location[0] == 0) { // Upper or lower half
		cropx1 = 1;
		cropx2 = pnx;
	}
	else {
		cropx = (int)floor(((Real)pnx) / 2);
		cropx1 = cropx - (int)floor(((Real)cropx) / 2);
		cropx2 = cropx1 + cropx - 1;
	}

	cropx1 -= 1;
	cropx2 -= 1;
	cropy1 -= 1;
	cropy2 -= 1;

	if (bCPU)
		encodeSideBand_CPU(cropx1, cropx2, cropy1, cropy2, sig_location);
	else
		encodeSideBand_GPU(cropx1, cropx2, cropy1, cropy2, sig_location);
}

fftw_plan fft_plan_fwd;
fftw_plan fft_plan_bwd;
void ophGen::encodeSideBand_CPU(int cropx1, int cropx2, int cropy1, int cropy2, oph::ivec2 sig_location)
{
	int pnx = context_.pixel_number[_X];
	int pny = context_.pixel_number[_Y];

	oph::Complex<Real>* h_crop = new oph::Complex<Real>[pnx*pny];
	memset(h_crop, 0.0, sizeof(oph::Complex<Real>)*pnx*pny);

	int p = 0;
#pragma omp parallel for private(p)
	for (p = 0; p < pnx*pny; p++)
	{
		int x = p % pnx;
		int y = p / pnx;
		if (x >= cropx1 && x <= cropx2 && y >= cropy1 && y <= cropy2)
			h_crop[p] = holo_gen[p];
	}

	oph::Complex<Real> *in = nullptr, *out = nullptr;

	fft2(oph::ivec2(pnx, pny), in, out, OPH_BACKWARD);
	fftwShift(h_crop, h_crop, pnx, pny, OPH_FORWARD, true);

	memset(holo_encoded, 0.0, sizeof(Real)*pnx*pny);
	int i = 0;
#pragma omp parallel for private(i)	
	for (i = 0; i < pnx*pny; i++) {
		oph::Complex<Real> shift_phase(1, 0);
		getShiftPhaseValue(shift_phase, i, sig_location);

		holo_encoded[i] = (h_crop[i] * shift_phase).real();
	}

	delete[] h_crop;
}

extern "C"
{
	/**
	* \defgroup gpu_model GPU Modules
	* @{
	*/
	/**
	* @brief Convert data from the spatial domain to the frequency domain using 2D FFT on GPU.
	* @details call CUDA Kernel - fftShift and CUFFT Library.
	* @param stream : CUDA Stream
	* @param nx : the number of column of the input data
	* @param ny : the number of row of the input data
	* @param in_field : input complex data variable
	* @param output_field : output complex data variable
	* @param direction : If direction == -1, forward FFT, if type == 1, inverse FFT.
	* @param bNomarlized : If bNomarlized == true, normalize the result after FFT.
	* @see propagation_AngularSpectrum_GPU, encoding_GPU
	*/
	void cudaFFT(CUstream_st* stream, int nx, int ny, cufftDoubleComplex* in_filed, cufftDoubleComplex* output_field, int direction, bool bNormailized = false);	
	
	/**
	* @brief Crop input data according to x, y coordinates on GPU.
	* @details call CUDA Kernel - cropFringe. 
	* @param stream : CUDA Stream
	* @param nx : the number of column of the input data
	* @param ny : the number of row of the input data
	* @param in_field : input complex data variable
	* @param output_field : output complex data variable
	* @param cropx1 : the start x-coordinate to crop.
	* @param cropx2 : the end x-coordinate to crop.
	* @param cropy1 : the start y-coordinate to crop.
	* @param cropy2 : the end y-coordinate to crop.
	* @see encoding_GPU
	*/
	void cudaCropFringe(CUstream_st* stream, int nx, int ny, cufftDoubleComplex* in_field, cufftDoubleComplex* out_field, int cropx1, int cropx2, int cropy1, int cropy2);

	/**
	* @brief Encode the CGH according to a signal location parameter on the GPU.
	* @details The variable, ((Real*)p_hologram) has the final result.
	* @param stream : CUDA Stream
	* @param pnx : the number of column of the input data
	* @param pny : the number of row of the input data
	* @param in_field : input data
	* @param out_field : output data
	* @param sig_locationx : signal location of x-axis, left or right half
	* @param sig_locationy : signal location of y-axis, upper or lower half
	* @param ssx : pnx * ppx
	* @param ssy : pny * ppy
	* @param ppx : pixel pitch of x-axis
	* @param ppy : pixel pitch of y-axis
	* @param PI : Pi
	* @see encoding_GPU
	*/
	void cudaGetFringe(CUstream_st* stream, int pnx, int pny, cufftDoubleComplex* in_field, cufftDoubleComplex* out_field, int sig_locationx, int sig_locationy,
		Real ssx, Real ssy, Real ppx, Real ppy, Real PI);
}

void ophGen::encodeSideBand_GPU(int cropx1, int cropx2, int cropy1, int cropy2, oph::ivec2 sig_location)
{
	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];
	Real ppx = context_.pixel_pitch[0];
	Real ppy = context_.pixel_pitch[1];
	Real ssx = context_.ss[0];
	Real ssy = context_.ss[1];

	cufftDoubleComplex *k_temp_d_, *u_complex_gpu_;
	cudaStream_t stream_;
	cudaStreamCreate(&stream_);

	cudaMalloc((void**)&u_complex_gpu_, sizeof(cufftDoubleComplex) * pnx * pny);
	cudaMalloc((void**)&k_temp_d_, sizeof(cufftDoubleComplex) * pnx * pny);
	cudaMemcpy(u_complex_gpu_, holo_gen, sizeof(cufftDoubleComplex) * pnx * pny, cudaMemcpyHostToDevice);

	cudaMemsetAsync(k_temp_d_, 0, sizeof(cufftDoubleComplex)*pnx*pny, stream_);
	cudaCropFringe(stream_, pnx, pny, u_complex_gpu_, k_temp_d_, cropx1, cropx2, cropy1, cropy2);

	cudaMemsetAsync(u_complex_gpu_, 0, sizeof(cufftDoubleComplex)*pnx*pny, stream_);
	cudaFFT(stream_, pnx, pny, k_temp_d_, u_complex_gpu_, 1, true);

	cudaMemsetAsync(k_temp_d_, 0, sizeof(cufftDoubleComplex)*pnx*pny, stream_);
	cudaGetFringe(stream_, pnx, pny, u_complex_gpu_, k_temp_d_, sig_location[0], sig_location[1], ssx, ssy, ppx, ppy, M_PI);

	cufftDoubleComplex* sample_fd = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*pnx*pny);
	memset(sample_fd, 0.0, sizeof(cufftDoubleComplex)*pnx*pny);

	cudaMemcpyAsync(sample_fd, k_temp_d_, sizeof(cufftDoubleComplex)*pnx*pny, cudaMemcpyDeviceToHost), stream_;
	memset(holo_encoded, 0.0, sizeof(Real)*pnx*pny);

	for (int i = 0; i < pnx * pny; i++)
		holo_encoded[i] = sample_fd[i].x;

	delete[] sample_fd;
	cudaStreamDestroy(stream_);
}

void ophGen::getShiftPhaseValue(oph::Complex<Real>& shift_phase_val, int idx, oph::ivec2 sig_location)
{
	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];
	Real ppx = context_.pixel_pitch[0];
	Real ppy = context_.pixel_pitch[1];
	Real ssx = context_.ss[0];
	Real ssy = context_.ss[1];

	if (sig_location[1] != 0)
	{
		int r = idx / pnx;
		int c = idx % pnx;
		Real yy = (ssy / 2.0) - (ppy)*r - ppy;

		oph::Complex<Real> val;
		if (sig_location[1] == 1)
			val[_IM] = 2 * M_PI * (yy / (4 * ppy));
		else
			val[_IM] = 2 * M_PI * (-yy / (4 * ppy));

		val.exp();
		shift_phase_val *= val;
	}

	if (sig_location[0] != 0)
	{
		int r = idx / pnx;
		int c = idx % pnx;
		Real xx = (-ssx / 2.0) - (ppx)*c - ppx;

		oph::Complex<Real> val;
		if (sig_location[0] == -1)
			val[_IM] = 2 * M_PI * (-xx / (4 * ppx));
		else
			val[_IM] = 2 * M_PI * (xx / (4 * ppx));

		val.exp();
		shift_phase_val *= val;
	}
}

void ophGen::getRandPhaseValue(oph::Complex<Real>& rand_phase_val, bool rand_phase)
{
	if (rand_phase)
	{
		rand_phase_val[_RE] = 0.0;
		rand_phase_val[_IM] = 2 * M_PI * oph::rand(0.0, 1.0);
		rand_phase_val.exp();

	}
	else {
		rand_phase_val[_RE] = 1.0;
		rand_phase_val[_IM] = 0.0;
	}
}

void ophGen::ophFree(void)
{
	if (holo_gen) delete[] holo_gen;
	if (holo_encoded) delete[] holo_encoded;
	if (holo_normalized) delete[] holo_normalized;
}