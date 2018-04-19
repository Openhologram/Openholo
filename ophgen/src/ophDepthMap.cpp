#define OPH_DM_EXPORT 

#include	"ophDepthMap.h"
#include	<windows.h>
#include	<random>
#include	<iomanip>
#include	<io.h>
#include	<direct.h>
#include    "sys.h"

/** 
* @brief Constructor
* @details Initialize variables.
*/
ophDepthMap::ophDepthMap()
{
	isCPU_ = true;

	// GPU Variables
	img_src_gpu_ = 0;
	dimg_src_gpu_ = 0;
	depth_index_gpu_ = 0;

	// CPU Variables
	img_src_ = 0;
	dmap_src_ = 0;
	alpha_map_ = 0;
	depth_index_ = 0;
	dmap_ = 0;
	dstep_ = 0;
	dlevel_.clear();
	U_complex_ = 0;
	u255_fringe_ = 0;

	sim_final_ = 0;
	hh_complex_ = 0;

}

/**
* @brief Destructor 
*/
ophDepthMap::~ophDepthMap()
{
}

/**
* @brief Set the value of a variable isCPU_(true or false)
* @details <pre>
    if isCPU_ == true
	   CPU implementation
	else
	   GPU implementation </pre>
* @param isCPU : the value for specifying whether the hologram generation method is implemented on the CPU or GPU
*/
void ophDepthMap::setMode(bool isCPU) 
{ 
	isCPU_ = isCPU; 
}

/**
* @brief Read parameters from a config file(config_openholo.txt).
* @return true if config infomation are sucessfully read, flase otherwise.
*/
bool ophDepthMap::readConfig()
{
	std::string inputFileName_ = "config_openholo.txt";

	LOG("Reading....%s\n", inputFileName_.c_str());

	std::ifstream inFile(inputFileName_.c_str());

	if (!inFile.is_open()){
		LOG("file not found.\n");
		return false;
	}

	// skip 7 lines
	std::string temp;
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');
	getline(inFile, temp, '\n');

	inFile >> SOURCE_FOLDER;						getline(inFile, temp, '\n');
	inFile >> IMAGE_PREFIX;							getline(inFile, temp, '\n');
	inFile >> DEPTH_PREFIX;							getline(inFile, temp, '\n');
	inFile >> RESULT_FOLDER;						getline(inFile, temp, '\n');
	inFile >> RESULT_PREFIX;						getline(inFile, temp, '\n');
	inFile >> FLAG_STATIC_IMAGE;					getline(inFile, temp, '\n');
	inFile >> START_OF_FRAME_NUMBERING;				getline(inFile, temp, '\n');
	inFile >> NUMBER_OF_FRAME;						getline(inFile, temp, '\n');
	inFile >> NUMBER_OF_DIGIT_OF_FRAME_NUMBERING;	getline(inFile, temp, '\n');

	// skip 3 lines
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');

	inFile >> Transform_Method_;					getline(inFile, temp, '\n');
	inFile >> Propagation_Method_;					getline(inFile, temp, '\n');
	inFile >> Encoding_Method_;						getline(inFile, temp, '\n');
	
	// skip 3 lines
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');

	inFile >> params_.field_lens;					getline(inFile, temp, '\n');
	inFile >> WAVELENGTH;							getline(inFile, temp, '\n');
	params_.lambda = WAVELENGTH;
	params_.k = 2 * PI / WAVELENGTH;
	
	inFile >> params_.pn[0];
	getline(inFile, temp, '\n');
	
	inFile >> params_.pn[1];
	getline(inFile, temp, '\n');
	
	inFile >> params_.pp[0];
	getline(inFile, temp, '\n');

	inFile >> params_.pp[1];
	getline(inFile, temp, '\n');

	params_.ss[0] = params_.pp[0] * params_.pn[0];
	params_.ss[1] = params_.pp[1] * params_.pn[1];

	// skip 3 lines
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');

	double NEAR_OF_DEPTH_MAP, FAR_OF_DEPTH_MAP;
	inFile >> NEAR_OF_DEPTH_MAP;					getline(inFile, temp, '\n');
	inFile >> FAR_OF_DEPTH_MAP;						getline(inFile, temp, '\n');

	params_.near_depthmap = min(NEAR_OF_DEPTH_MAP, FAR_OF_DEPTH_MAP);
	params_.far_depthmap = max(NEAR_OF_DEPTH_MAP, FAR_OF_DEPTH_MAP);

	inFile >> FLAG_CHANGE_DEPTH_QUANTIZATION;		getline(inFile, temp, '\n');
	inFile >> DEFAULT_DEPTH_QUANTIZATION;			getline(inFile, temp, '\n');
	inFile >> NUMBER_OF_DEPTH_QUANTIZATION;			getline(inFile, temp, '\n');
		
	if (FLAG_CHANGE_DEPTH_QUANTIZATION == 0)
		params_.num_of_depth = DEFAULT_DEPTH_QUANTIZATION;
	else
		params_.num_of_depth = NUMBER_OF_DEPTH_QUANTIZATION;
	
	inFile >> temp;			
	std::size_t found = temp.find(':');
	if (found != std::string::npos)
	{
		std::string s = temp.substr(0, found);
		std::string e = temp.substr(found + 1);
		int start = std::stoi(s);
		int end = std::stoi(e);
		params_.render_depth.clear();
		for (int k = start; k <= end; k++)
			params_.render_depth.push_back(k);

	}else {

		params_.render_depth.clear();
		params_.render_depth.push_back(std::stoi(temp));
		inFile >> temp;

		while (temp.find('/') == std::string::npos)
		{
			params_.render_depth.push_back(std::stoi(temp));
			inFile >> temp;
		}
	}
	if (params_.render_depth.empty()){
		LOG("Error: RENDER_DEPTH \n");
		return false;
	}
		
	getline(inFile, temp, '\n');
	inFile >> RANDOM_PHASE;							getline(inFile, temp, '\n');
	
	//==Simulation parameters ======================================================================
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');

	inFile >> Simulation_Result_File_Prefix_;			getline(inFile, temp, '\n');
	inFile >> test_pixel_number_scale_;					getline(inFile, temp, '\n');
	inFile >> eye_length_;								getline(inFile, temp, '\n');
	inFile >> eye_pupil_diameter_;						getline(inFile, temp, '\n');
	inFile >> eye_center_xy_[0];						getline(inFile, temp, '\n');
	inFile >> eye_center_xy_[1];						getline(inFile, temp, '\n');
	inFile >> focus_distance_;							getline(inFile, temp, '\n');

	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');
	getline(inFile, temp, '\n');	getline(inFile, temp, '\n');	
	
	inFile >> sim_type_;								getline(inFile, temp, '\n');
	inFile >> sim_from_;								getline(inFile, temp, '\n');
	inFile >> sim_to_;									getline(inFile, temp, '\n');
	inFile >> sim_step_num_;							getline(inFile, temp, '\n');
	
	//=====================================================================================
	inFile.close();

	LOG("done\n");

	return true;

}

/**
* @brief Initialize variables for CPU and GPU implementation.
* @see init_CPU, init_GPU
*/
void ophDepthMap::initialize()
{
	dstep_ = 0;
	dlevel_.clear();

	if (u255_fringe_)		free(u255_fringe_);
	u255_fringe_ = (double*)malloc(sizeof(double) * params_.pn[0] * params_.pn[1]);
	
	if (isCPU_)
		init_CPU();
	else
		init_GPU();	
}

/**
* @brief Generate a hologram, main funtion.
* @details For each frame, 
*    1. Read image depth data.
*    2. Compute the physical distance of depth map.
*    3. Transform target object to reflect the system configuration of holographic display.
*    4. Generate a hologram.
*    5. Encode the generated hologram.
*    6. Write the hologram to a image.
* .
* @see readImageDepth, getDepthValues, transformViewingWindow, calc_Holo_by_Depth, encodingSymmetrization, writeResultimage
*/
void ophDepthMap::generateHologram()
{
	int num_of_frame;
	if (FLAG_STATIC_IMAGE == 0)
		num_of_frame = NUMBER_OF_FRAME;
	else
		num_of_frame = 1;

	for (int ftr = 0; ftr <= num_of_frame - 1; ftr++)
	{
		LOG("Calculating hologram of frame %d.\n", ftr);

		if (!readImageDepth(ftr)) {
			LOG("Error: Reading image of frame %d.\n", ftr);
			continue;
		}

		getDepthValues();

		if (Transform_Method_ == 0)
			transformViewingWindow();

		calc_Holo_by_Depth(ftr);
		
		if (Encoding_Method_ == 0)
			encodingSymmetrization(ivec2(0,1));

		writeResultimage(ftr);

		//writeMatFileDouble("u255_fringe", u255_fringe_);
		//writeMatFileComplex("U_complex", U_complex_);
	}

}

/**
* @brief Read image and depth map.
* @details Read input files and load image & depth map data.
*  If the input image size is different with the dislay resolution, resize the image size.
* @param ftr : the frame number of the image.
* @return true if image data are sucessfully read, flase otherwise.
* @see prepare_inputdata_CPU, prepare_inputdata_GPU
*/
int ophDepthMap::readImageDepth(int ftr)
{	
	std::string src_folder;
	if (FLAG_STATIC_IMAGE == 0)
	{
		if (NUMBER_OF_DIGIT_OF_FRAME_NUMBERING > 0) {
			//src_folder = std::string().append(SOURCE_FOLDER).append("/") + QString("%1").arg((uint)(ftr + START_OF_FRAME_NUMBERING), (int)NUMBER_OF_DIGIT_OF_FRAME_NUMBERING, 10, (QChar)'0');
			std::stringstream ss;
			ss << std::setw(NUMBER_OF_DIGIT_OF_FRAME_NUMBERING) << std::setfill('0') << ftr + START_OF_FRAME_NUMBERING;
			src_folder = ss.str();
		} else
			src_folder = std::string().append(SOURCE_FOLDER).append("/").append(std::to_string(ftr + START_OF_FRAME_NUMBERING));

	}else 
		src_folder = std::string().append(SOURCE_FOLDER);

	
	std::string sdir = std::string("./").append(src_folder).append("/").append(IMAGE_PREFIX).append("*.bmp");

	_finddatai64_t fd;
	intptr_t handle;
	handle = _findfirst64(sdir.c_str(), &fd); 
	if (handle == -1)
	{
		LOG("Error: Source image does not exist: %s.\n", sdir.c_str());
		return false;
	}

	std::string imgfullname = std::string("./").append(src_folder).append("/").append(fd.name);

	int w, h, bytesperpixel;
	int ret = getBitmapSize(w, h, bytesperpixel, imgfullname.c_str());

	unsigned char* imgload = (unsigned char*)malloc(sizeof(unsigned char)*w*h*bytesperpixel);
	ret = loadBitmapFile(imgload, imgfullname.c_str());
	if (!ret) {
		LOG("Failed::Image Load: %s\n", imgfullname.c_str());
		return false;
	}
	LOG("Succeed::Image Load: %s\n", imgfullname.c_str());

	unsigned char* img = (unsigned char*)malloc(sizeof(unsigned char)*w*h);
	convertToFormatGray8(img, imgload, w, h, bytesperpixel);

	//ret = creatBitmapFile(img, w, h, 8, "load_img.bmp");

	//QImage qtimg(img, w, h, QImage::Format::Format_Grayscale8);
	//qtimg.save("load_qimg.bmp");

	free(imgload);


	//=================================================================================
	std::string sddir = std::string("./").append(src_folder).append("/").append(DEPTH_PREFIX).append("*.bmp");
	handle = _findfirst64(sddir.c_str(), &fd);
	if (handle == -1)
	{
		LOG("Error: Source depthmap does not exist: %s.\n", sddir);
		return false;
	}

	std::string dimgfullname = std::string("./").append(src_folder).append("/").append(fd.name);

	int dw, dh, dbytesperpixel;
	ret = getBitmapSize(dw, dh, dbytesperpixel, dimgfullname.c_str());

	unsigned char* dimgload = (unsigned char*)malloc(sizeof(unsigned char)*dw*dh*dbytesperpixel);
	ret = loadBitmapFile(dimgload, dimgfullname.c_str());
	if (!ret) {
		LOG("Failed::Depth Image Load: %s\n", dimgfullname.c_str());
		return false;
	}
	LOG("Succeed::Depth Image Load: %s\n", dimgfullname.c_str());

	unsigned char* dimg = (unsigned char*)malloc(sizeof(unsigned char)*dw*dh);
	convertToFormatGray8(dimg, dimgload, dw, dh, dbytesperpixel);

	free(dimgload);

	//ret = creatBitmapFile(dimg, dw, dh, 8, "dtest");
	//=======================================================================
	//resize image
	int pnx = params_.pn[0];
	int pny = params_.pn[1];

	unsigned char* newimg = (unsigned char*)malloc(sizeof(char)*pnx*pny);
	memset(newimg, 0, sizeof(char)*pnx*pny);

	if (w != pnx || h != pny)
		imgScaleBilnear(img, newimg, w, h, pnx, pny);
	else
		memcpy(newimg, img, sizeof(char)*pnx*pny);

	//ret = creatBitmapFile(newimg, pnx, pny, 8, "stest");

	unsigned char* newdimg = (unsigned char*)malloc(sizeof(char)*pnx*pny);
	memset(newdimg, 0, sizeof(char)*pnx*pny);

	if (dw != pnx || dh != pny)
		imgScaleBilnear(dimg, newdimg, dw, dh, pnx, pny);
	else
		memcpy(newdimg, dimg, sizeof(char)*pnx*pny);

	if (isCPU_)
		ret = prepare_inputdata_CPU(newimg, newdimg);
	else
		ret = prepare_inputdata_GPU(newimg, newdimg);

	free(img);
	free(dimg);

	//writeIntensity_gray8_bmp("test.bmp", pnx, pny, dmap_src_);
	//writeIntensity_gray8_bmp("test2.bmp", pnx, pny, dmap_);
	//dimg.save("test_dmap.bmp");
	//img.save("test_img.bmp");

	return ret;
	
}

/**
* @brief Calculate the physical distances of depth map layers
* @details Initialize 'dstep_' & 'dlevel_' variables.
*  If FLAG_CHANGE_DEPTH_QUANTIZATION == 1, recalculate  'depth_index_' variable.
* @see change_depth_quan_CPU, change_depth_quan_GPU
*/
void ophDepthMap::getDepthValues()
{
	if (params_.num_of_depth > 1)
	{
		dstep_ = (params_.far_depthmap - params_.near_depthmap) / (params_.num_of_depth - 1);
		double val = params_.near_depthmap;
		while (val <= params_.far_depthmap)
		{
			dlevel_.push_back(val);
			val += dstep_;
		}

	} else {

		dstep_ = (params_.far_depthmap + params_.near_depthmap) / 2;
		dlevel_.push_back(params_.far_depthmap - params_.near_depthmap);

	}
	
	if (FLAG_CHANGE_DEPTH_QUANTIZATION == 1)
	{
		if (isCPU_)
			change_depth_quan_CPU();
		else
			change_depth_quan_GPU();
	}
}

/**
* @brief Transform target object to reflect the system configuration of holographic display.
* @details Calculate 'dlevel_transform_' variable by using 'field_lens' & 'dlevel_'.
*/
void ophDepthMap::transformViewingWindow()
{
	int pnx = params_.pn[0];
	int pny = params_.pn[1];

	double val;
	dlevel_transform_.clear();
	for (int p = 0; p < dlevel_.size(); p++)
	{
		val = -params_.field_lens * dlevel_[p] / (dlevel_[p] - params_.field_lens);
		dlevel_transform_.push_back(val);
	}
}

/**
* @brief Generate a hologram.
* @param frame : the frame number of the image.
* @see calc_Holo_CPU, calc_Holo_GPU
*/
void ophDepthMap::calc_Holo_by_Depth(int frame)
{
	if (isCPU_)
		calc_Holo_CPU(frame);
	else
		calc_Holo_GPU(frame);
	
}

/**
* @brief Assign random phase value if RANDOM_PHASE == 1
* @details If RANDOM_PHASE == 1, calculate a random phase value using random generator;
*  otherwise, random phase value is 1.
* @param rand_phase_val : Input & Ouput value.
*/
void ophDepthMap::get_rand_phase_value(Complex& rand_phase_val)
{
	if (RANDOM_PHASE)
	{
		std::default_random_engine generator;
		std::uniform_real_distribution<double> distribution(0.0, 1.0);

		rand_phase_val.a = 0.0;
		rand_phase_val.b = 2 * PI * distribution(generator);
		exponent_complex(&rand_phase_val);

	} else {
		rand_phase_val.a = 1.0;
		rand_phase_val.b = 0.0;
	}

}

/**
* @brief Write the result image.
* @param ftr : the frame number of the image.
*/

void ophDepthMap::writeResultimage(int ftr)
{	
	std::string outdir = std::string("./").append(RESULT_FOLDER);

	if (!CreateDirectory(outdir.c_str(), NULL) && ERROR_ALREADY_EXISTS != GetLastError())
	{
		LOG("Fail to make output directory\n");
		return;
	}
	
	std::string fname = std::string("./").append(RESULT_FOLDER).append("/").append(RESULT_PREFIX).append(std::to_string(ftr)).append(".bmp");

	int pnx = params_.pn[0];
	int pny = params_.pn[1];
	int px = static_cast<int>(pnx / 3);
	int py = pny;

	double min_val, max_val;
	min_val = u255_fringe_[0];
	max_val = u255_fringe_[0];
	for (int i = 0; i < pnx*pny; ++i)
	{
		if (min_val > u255_fringe_[i])
			min_val = u255_fringe_[i];
		else if (max_val < u255_fringe_[i])
			max_val = u255_fringe_[i];
	}

	uchar* data = (uchar*)malloc(sizeof(uchar)*pnx*pny);
	memset(data, 0, sizeof(uchar)*pnx*pny);

	int x = 0;
#pragma omp parallel for private(x)	
	for (x = 0; x < pnx*pny; ++x)
		data[x] = (uint)((u255_fringe_[x] - min_val) / (max_val - min_val) * 255);

	//QImage img(data, px, py, QImage::Format::Format_RGB888);
	//img.save("test_qimg.bmp");

	int ret = Openholo::createBitmapFile(data, px, py, 24, fname.c_str());

	free(data);
	
}

/**
* @brief It is a testing function used for the reconstruction.
*/

void ophDepthMap::writeSimulationImage(int num, double val)
{
	std::string outdir = std::string("./").append(RESULT_FOLDER);
	if (!CreateDirectory(outdir.c_str(), NULL) && ERROR_ALREADY_EXISTS != GetLastError())
	{
		LOG("Fail to make output directory\n");
		return;
	}

	std::string fname = std::string("./").append(RESULT_FOLDER).append("/").append(Simulation_Result_File_Prefix_).append("_");
	fname = fname.append(RESULT_PREFIX).append(std::to_string(num)).append("_").append(sim_type_ == 0 ? "FOCUS_" : "EYE_Y_");
	int v = (int)round(val * 1000);
	fname = fname.append(std::to_string(v));
	
	int pnx = params_.pn[0];
	int pny = params_.pn[1];
	int px = pnx / 3;
	int py = pny;

	double min_val, max_val;
	min_val = sim_final_[0];
	max_val = sim_final_[0];
	for (int i = 0; i < pnx*pny; ++i)
	{
		if (min_val > sim_final_[i])
			min_val = sim_final_[i];
		else if (max_val < sim_final_[i])
			max_val = sim_final_[i];
	}

	uchar* data = (uchar*)malloc(sizeof(uchar)*pnx*pny);
	memset(data, 0, sizeof(uchar)*pnx*pny);
		
	for (int k = 0; k < pnx*pny; k++)
		data[k] = (uint)((sim_final_[k] - min_val) / (max_val - min_val) * 255);

	//QImage img(data, px, py, QImage::Format::Format_RGB888);
	//img.save(QString(fname));

	int ret = Openholo::createBitmapFile(data, px, py, 24, fname.c_str());

	free(data);

}

void ophDepthMap::ophFree(void)
{
}



/*
void ophDepthMap::writeIntensity_gray8_bmp(const char* fileName, int nx, int ny, double* intensity)
{
	const int n = nx*ny;

	double min_val, max_val;
	min_val = intensity[0];
	max_val = intensity[0];

	for (int i = 0; i < n; ++i)
	{
		if (min_val > intensity[i])
			min_val = intensity[i];
		else if (max_val < intensity[i])
			max_val = intensity[i];
	}

	char fname[100];
	strcpy(fname, fileName);
	strcat(fname, ".bmp");

	//LOG("minval %f, max val %f\n", min_val, max_val);
	unsigned char* cgh = (unsigned char*)malloc(sizeof(unsigned char)*n);

	for (int i = 0; i < n; ++i){
		double val = 255 * ((intensity[i] - min_val) / (max_val - min_val));
		cgh[i] = val;
	}

	QImage img(cgh, nx, ny, QImage::Format::Format_Grayscale8);
	img.save(QString(fname));

	ophFree(cgh);
}

void ophDepthMap::writeIntensity_gray8_bmp(const char* fileName, int nx, int ny, Complex* complexvalue)
{
	const int n = nx*ny;

	double* intensity = (double*)malloc(sizeof(double)*n);
	for (int i = 0; i < n; i++)
		intensity[i] = complexvalue[i].a;
		//intensity[i] = complexvalue[i].mag2();

	double min_val, max_val;
	min_val = intensity[0];
	max_val = intensity[0];

	for (int i = 0; i < n; ++i)
	{
		if (min_val > intensity[i])
			min_val = intensity[i];
		else if (max_val < intensity[i])
			max_val = intensity[i];
	}

	char fname[100];
	strcpy(fname, fileName);
	strcat(fname, ".bmp");

	//LOG("minval %e, max val %e\n", min_val, max_val);

	unsigned char* cgh = (unsigned char*)malloc(sizeof(unsigned char)*n);

	for (int i = 0; i < n; ++i) {
		double val = (intensity[i] - min_val) / (max_val - min_val);
		//val = pow(val, 1.0 / 1.5);
		val = val * 255.0;
		unsigned char v = (uchar)val;

		cgh[i] = v;
	}

	QImage img(cgh, nx, ny, QImage::Format::Format_Grayscale8);
	img.save(QString(fname));


	ophFree(intensity);
	ophFree(cgh);
}

void ophDepthMap::writeIntensity_gray8_real_bmp(const char* fileName, int nx, int ny, Complex* complexvalue)
{
	const int n = nx*ny;

	double* intensity = (double*)malloc(sizeof(double)*n);
	for (int i = 0; i < n; i++)
		intensity[i] = complexvalue[i].a;

	double min_val, max_val;
	min_val = intensity[0];
	max_val = intensity[0];

	for (int i = 0; i < n; ++i)
	{
		if (min_val > intensity[i])
			min_val = intensity[i];
		else if (max_val < intensity[i])
			max_val = intensity[i];
	}

	char fname[100];
	strcpy(fname, fileName);
	strcat(fname, ".bmp");

	//LOG("minval %e, max val %e\n", min_val, max_val);

	unsigned char* cgh = (unsigned char*)malloc(sizeof(new unsigned char)*n);

	for (int i = 0; i < n; ++i) {
		double val = (intensity[i] - min_val) / (max_val - min_val);
		//val = pow(val, 1.0 / 1.5);
		val = val * 255.0;
		unsigned char v = (uchar)val;

		cgh[i] = v;
	}

	QImage img(cgh, nx, ny, QImage::Format::Format_Grayscale8);
	img.save(QString(fname));

	ophFree(intensity);
	ophFree(cgh);

}
*/

/*
bool ophDepthMap::readMatFileDouble(const char* fileName, double * val)
{
	MATFile *pmat;
	mxArray *parray;

	char fname[100];
	strcpy(fname, fileName);

	pmat = matOpen(fname, "r");
	if (pmat == NULL) {
		OG("Error creating file %s\n", fname);
		return false;
	}

	//===============================================================
	parray = matGetVariableInfo(pmat, "inputmat");

	if (parray == NULL) {
		printf("Error reading existing matrix \n");
		return false;
	}

	int m = mxGetM(parray);
	int n = mxGetN(parray);

	if (params_.pn[0] * params_.pn[1] != m*n)
	{
		printf("Error : different matrix dimension \n");
		return false;
	}

	double* dst_r;
	parray = matGetVariable(pmat, "inputmat");
	dst_r = val;

	double* CompRealPtr = mxGetPr(parray);

	for (int col = 0; col < n; col++)
	{
		for (int row = 0; row < m; row++)
		{
			dst_r[n*row + col] = *CompRealPtr++;
		}
	}

	// clean up
	mxDestroyArray(parray);

	if (matClose(pmat) != 0) {
		LOG("Error closing file %s\n", fname);
		return false;
	}

	LOG("Read Mat file %s\n", fname);
	return true;
}

void ophDepthMap::writeMatFileComplex(const char* fileName, Complex* val)
{
	MATFile *pmat;
	mxArray *pData;

	char fname[100];
	strcpy(fname, fileName);
	strcat(fname, ".mat");

	pmat = matOpen(fname, "w");
	if (pmat == NULL) {
		LOG("Error creating file %s\n", fname);
		return;
	}

	ivec2 pn = params_.pn;
	int w = pn[0];
	int h = pn[1];
	const int n = w * h;

	pData = mxCreateDoubleMatrix(h, w, mxCOMPLEX);
	if (pData == NULL) {
		LOG("Unable to create mxArray.\n");
		return;
	}

	double* CompRealPtr = mxGetPr(pData);
	double* CompImgPtr = mxGetPi(pData);

	for (int col = 0; col < w; col++)
	{
		//for (int row = h-1; row >= 0; row--)
		for (int row = 0; row < h; row++)
		{
			*CompRealPtr++ = val[w*row + col].a;
			*CompImgPtr++ = val[w*row + col].b;
		}
	}

	int status;
	status = matPutVariable(pmat, "data", pData);

	if (status != 0) {
		LOG("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
		return;
	}

	///* clean up
	mxDestroyArray(pData);

	if (matClose(pmat) != 0) {
		LOG("Error closing file %s\n", fname);
		return;
	}

	LOG("Write Mat file %s\n", fname);
	
}

void ophDepthMap::writeMatFileDouble(const char* fileName, double * val)
{
	MATFile *pmat;
	mxArray *pData;

	char fname[100];
	strcpy(fname, fileName);
	strcat(fname, ".mat");

	pmat = matOpen(fname, "w");
	if (pmat == NULL) {
		LOG("Error creating file %s\n", fname);
		return;
	}

	ivec2 pn = params_.pn;
	int w = pn[0];
	int h = pn[1];
	const int n = w * h;

	pData = mxCreateDoubleMatrix(h, w, mxREAL);
	if (pData == NULL) {
		LOG("Unable to create mxArray.\n");
		return;
	}

	double* CompRealPtr = mxGetPr(pData);
	for (int col = 0; col < w; col++)
	{
		//for (int row = h-1; row >= 0; row--)
		for (int row = 0; row < h; row++)
			*CompRealPtr++ = val[w*row + col];
	}

	int status;
	status = matPutVariable(pmat, "inputmat", pData);

	if (status != 0) {
		LOG("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
		return;
	}

	///* clean up
	mxDestroyArray(pData);

	if (matClose(pmat) != 0) {
		LOG("Error closing file %s\n", fname);
		return;
	}

	LOG("Write Mat file %s\n", fname);
}
*/
