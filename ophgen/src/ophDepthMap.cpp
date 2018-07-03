#define OPH_DM_EXPORT 

#include	"ophDepthMap.h"
#include	<windows.h>
#include	<random>
#include	<iomanip>
#include	<io.h>
#include	<direct.h>
#include    "sys.h"

#include	"include.h"

/** 
* @brief Constructor
* @details Initialize variables.
*/
ophDepthMap::ophDepthMap()
	: ophGen()
{
	is_CPU = true;

	// GPU Variables
	img_src_gpu = 0;
	dimg_src_gpu = 0;
	depth_index_gpu = 0;

	// CPU Variables
	img_src = 0;
	dmap_src = 0;
	alpha_map = 0;
	depth_index = 0;
	dmap = 0;
	dstep = 0;
	dlevel.clear();
}

/**
* @brief Destructor 
*/
ophDepthMap::~ophDepthMap()
{
}

/**
* @brief Set the value of a variable is_CPU(true or false)
* @details <pre>
    if is_CPU == true
	   CPU implementation
	else
	   GPU implementation </pre>
* @param is_CPU : the value for specifying whether the hologram generation method is implemented on the CPU or GPU
*/
void ophDepthMap::setMode(bool is_CPU) 
{ 
	this->is_CPU = is_CPU; 
}

/**
* @brief Read parameters from a config file(config_openholo.txt).
* @return true if config infomation are sucessfully read, flase otherwise.
*/
bool ophDepthMap::readConfig(const char* fname)
{
	bool b_ok = ophGen::readConfig(fname, dm_config_, dm_params_);

	return b_ok;
}

/**
* @brief Initialize variables for CPU and GPU implementation.
* @see init_CPU, init_GPU
*/
void ophDepthMap::initialize(void)
{
	dstep = 0;
	dlevel.clear();

	if (holo_gen) delete[] holo_gen;
	holo_gen = new oph::Complex<Real>[context_.pixel_number[_X] * context_.pixel_number[_Y]];
	memset(holo_gen, 0.0, sizeof(oph::Complex<Real>) * context_.pixel_number[_X] * context_.pixel_number[_Y]);

	if (holo_encoded) delete[] holo_encoded;
	holo_encoded = new Real[context_.pixel_number[_X] * context_.pixel_number[_Y]];
	memset(holo_encoded, 0.0, sizeof(Real) * context_.pixel_number[_X] * context_.pixel_number[_Y]);

	if (holo_normalized) delete[] holo_normalized;
	holo_normalized = new uchar[context_.pixel_number[_X] * context_.pixel_number[_Y]];
	memset(holo_normalized, 0.0, sizeof(uchar) * context_.pixel_number[_X] * context_.pixel_number[_Y]);
	
	if (is_CPU)
		initCPU();
	else
		initGPU();
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
double ophDepthMap::generateHologram()
{
	auto time_start = CUR_TIME;

	initialize();

	if (!readImageDepth())
		LOG("Error: Reading image.\n");

	getDepthValues();

	if (dm_params_.Transform_Method_ == 0)
		transformViewingWindow();

	calcHoloByDepth();

	auto time_end = CUR_TIME;

	return ((std::chrono::duration<Real>)(time_end - time_start)).count();
}

void ophDepthMap::encodeHologram(void)
{
	if (dm_params_.Encoding_Method_ == 0)
		encodeSideBand(is_CPU, ivec2(0, 1));
}

/**
* @brief Read image and depth map.
* @details Read input files and load image & depth map data.
*  If the input image size is different with the dislay resolution, resize the image size.
* @param ftr : the frame number of the image.
* @return true if image data are sucessfully read, flase otherwise.
* @see prepare_inputdata_CPU, prepare_inputdata_GPU
*/
int ophDepthMap::readImageDepth(void)
{	
	std::string sdir = std::string("./").append(dm_params_.SOURCE_FOLDER).append("/").append(dm_params_.IMAGE_PREFIX).append("*.bmp");

	_finddatai64_t fd;
	intptr_t handle;
	handle = _findfirst64(sdir.c_str(), &fd); 
	if (handle == -1)
	{
		LOG("Error: Source image does not exist: %s.\n", sdir.c_str());
		return false;
	}

	std::string imgfullname = std::string("./").append(dm_params_.SOURCE_FOLDER).append("/").append(fd.name);

	int w, h, bytesperpixel;
	int ret = getImgSize(w, h, bytesperpixel, imgfullname.c_str());

	oph::uchar* imgload = new uchar[w*h*bytesperpixel];
	ret = loadAsImg(imgfullname.c_str(), (void*)imgload);
	if (!ret) {
		LOG("Failed::Image Load: %s\n", imgfullname.c_str());
		return false;
	}
	LOG("Succeed::Image Load: %s\n", imgfullname.c_str());

	oph::uchar* img = new uchar[w*h];
	convertToFormatGray8(imgload, img, w, h, bytesperpixel);

	delete[] imgload;


	//=================================================================================
	std::string sddir = std::string("./").append(dm_params_.SOURCE_FOLDER).append("/").append(dm_params_.DEPTH_PREFIX).append("*.bmp");
	handle = _findfirst64(sddir.c_str(), &fd);
	if (handle == -1)
	{
		LOG("Error: Source depthmap does not exist: %s.\n", sddir);
		return false;
	}

	std::string dimgfullname = std::string("./").append(dm_params_.SOURCE_FOLDER).append("/").append(fd.name);

	int dw, dh, dbytesperpixel;
	ret = getImgSize(dw, dh, dbytesperpixel, dimgfullname.c_str());

	uchar* dimgload = new uchar[dw*dh*dbytesperpixel];
	ret = loadAsImg(dimgfullname.c_str(), (void*)dimgload);
	if (!ret) {
		LOG("Failed::Depth Image Load: %s\n", dimgfullname.c_str());
		return false;
	}
	LOG("Succeed::Depth Image Load: %s\n", dimgfullname.c_str());

	uchar* dimg = new uchar[dw*dh];
	convertToFormatGray8(dimgload, dimg, dw, dh, dbytesperpixel);

	delete[] dimgload;

	//resize image
	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];

	uchar* newimg = new uchar[pnx*pny];
	memset(newimg, 0, sizeof(char)*pnx*pny);

	if (w != pnx || h != pny)
		imgScaleBilnear(img, newimg, w, h, pnx, pny);
	else
		memcpy(newimg, img, sizeof(char)*pnx*pny);

	//ret = creatBitmapFile(newimg, pnx, pny, 8, "stest");

	uchar* newdimg = new uchar[pnx*pny];
	memset(newdimg, 0, sizeof(char)*pnx*pny);

	if (dw != pnx || dh != pny)
		imgScaleBilnear(dimg, newdimg, dw, dh, pnx, pny);
	else
		memcpy(newdimg, dimg, sizeof(char)*pnx*pny);

	if (is_CPU)
		ret = prepareInputdataCPU(newimg, newdimg);
	else
		ret = prepareInputdataGPU(newimg, newdimg);

	delete[] img;
	delete[] dimg;

	return ret;
}

/**
* @brief Calculate the physical distances of depth map layers
* @details Initialize 'dstep' & 'dlevel' variables.
*  If FLAG_CHANGE_DEPTH_QUANTIZATION == 1, recalculate  'depth_index' variable.
* @see change_depth_quan_CPU, change_depth_quan_GPU
*/
void ophDepthMap::getDepthValues()
{
	if (dm_config_.num_of_depth > 1)
	{
		dstep = (dm_config_.far_depthmap - dm_config_.near_depthmap) / (dm_config_.num_of_depth - 1);
		Real val = dm_config_.near_depthmap;
		while (val <= dm_config_.far_depthmap)
		{
			dlevel.push_back(val);
			val += dstep;
		}

	} else {

		dstep = (dm_config_.far_depthmap + dm_config_.near_depthmap) / 2;
		dlevel.push_back(dm_config_.far_depthmap - dm_config_.near_depthmap);

	}
	
	if (dm_params_.FLAG_CHANGE_DEPTH_QUANTIZATION == 1)
	{
		if (is_CPU)
			changeDepthQuanCPU();
		else
			changeDepthQuanGPU();
	}
}

/**
* @brief Transform target object to reflect the system configuration of holographic display.
* @details Calculate 'dlevel_transform' variable by using 'field_lens' & 'dlevel'.
*/
void ophDepthMap::transformViewingWindow()
{
	int pnx = context_.pixel_number[_X];
	int pny = context_.pixel_number[_Y];

	Real val;
	dlevel_transform.clear();
	for (int p = 0; p < dlevel.size(); p++)
	{
		val = -dm_config_.field_lens * dlevel[p] / (dlevel[p] - dm_config_.field_lens);
		dlevel_transform.push_back(val);
	}
}

/**
* @brief Generate a hologram.
* @param frame : the frame number of the image.
* @see calc_Holo_CPU, calc_Holo_GPU
*/
void ophDepthMap::calcHoloByDepth(void)
{
	if (is_CPU)
		calcHoloCPU();
	else
		calcHoloGPU();
}

/**
* @brief Write the result image.
* @param ftr : the frame number of the image.
*/

int ophDepthMap::save(const char* fname, uint8_t bitsperpixel)
{
	std::string outdir = std::string("./").append(dm_params_.RESULT_FOLDER);

	if (!CreateDirectory(outdir.c_str(), NULL) && ERROR_ALREADY_EXISTS != GetLastError())
	{
		LOG("Fail to make output directory\n");
		return 0;
	}
	
	std::string resName = std::string("./").append(dm_params_.RESULT_FOLDER).append("/").append(dm_params_.RESULT_PREFIX).append(".bmp");

	int pnx = context_.pixel_number[_X];
	int pny = context_.pixel_number[_Y];
	int px = static_cast<int>(pnx / 3);
	int py = pny;

	ophGen::save(resName.c_str(), bitsperpixel, holo_normalized, px, py);

	return 1;
}

/**
* @brief It is a testing function used for the reconstruction.
*/
//commit test

//void ophDepthMap::writeSimulationImage(int num, Real val)
//{
//	std::string outdir = std::string("./").append(dm_params_.RESULT_FOLDER);
//	if (!CreateDirectory(outdir.c_str(), NULL) && ERROR_ALREADY_EXISTS != GetLastError())
//	{
//		LOG("Fail to make output directory\n");
//		return;
//	}
//
//	std::string fname = std::string("./").append(dm_params_.RESULT_FOLDER).append("/").append(dm_simuls_.Simulation_Result_File_Prefix_).append("_");
//	fname = fname.append(dm_params_.RESULT_PREFIX).append(std::to_string(num)).append("_").append(dm_simuls_.sim_type_ == 0 ? "FOCUS_" : "EYE_Y_");
//	int v = (int)round(val * 1000);
//	fname = fname.append(std::to_string(v));
//	
//	int pnx = context_.pixel_number[0];
//	int pny = context_.pixel_number[1];
//	int px = pnx / 3;
//	int py = pny;
//
//	Real min_val, max_val;
//	min_val = dm_simuls_.sim_final_[0];
//	max_val = dm_simuls_.sim_final_[0];
//	for (int i = 0; i < pnx*pny; ++i)
//	{
//		if (min_val > dm_simuls_.sim_final_[i])
//			min_val = dm_simuls_.sim_final_[i];
//		else if (max_val < dm_simuls_.sim_final_[i])
//			max_val = dm_simuls_.sim_final_[i];
//	}
//
//	uchar* data = new uchar[pnx*pny];
//	memset(data, 0, sizeof(uchar)*pnx*pny);
//		
//	for (int k = 0; k < pnx*pny; k++)
//		data[k] = (uint)((dm_simuls_.sim_final_[k] - min_val) / (max_val - min_val) * 255);
//
//	int ret = ophGen::save(fname.c_str(), 24, data, px, py);
//
//	delete[] data;
//}

void ophDepthMap::ophFree(void)
{
	if (img_src)			delete[] img_src;
	if (dmap_src)			delete[] dmap_src;
	if (alpha_map)			delete[] alpha_map;
	if (depth_index)		delete[] depth_index;
	if (dmap)				delete[] dmap;

	free_gpu();
}