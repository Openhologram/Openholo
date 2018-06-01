/** @mainpage
@tableofcontents
@section intro Introduction
This library implements the hologram generation method using depth map data. <br>
It is implemented on the CPU and the GPU to improve the performance of the hologram generation method.
Thus, user can compare the performance between the CPU and GPU implementation. 
<br>
@image html doc_intro.png
@image latex doc_intro.png

@section algo Algorithm Reference
The original algorithm is modified in the way that can be easily implemented in parallel. <br>
Back propagate each depth plane to the hologram plane and accumulate the results of each propagation.
<br>
@image html doc_algo.png "Depth Map Hologram Generation Algorithm"
@image latex doc_algo.png "Depth Map Hologram Generation Algorithm"

@section swcom Software Components
The library consists a main hologram generation module(Hologram folder) and its sample program(HologramDepthmap folder).
<br>The following shows the list of files with the directory structure.
<br>
@image html doc_swfolders.png
@image latex doc_swfolders.png

@section proc Main Procedure
The main function of the library is a  \c \b generateHologram() of \c ophDepthMap class.
The following is the procedure of it and functions called form it..
<br><br>
@image html doc_proc.png "generateHologram Function Procedure"
@image latex doc_proc.png "generateHologram Function Procedure"

@section env Environment
 - Microsoft Visual Studio 2015 C++
 - Qt 5.6.2
 - CUDA 8.0
 - FFTW 3.3.5

@section build How to Build Source Codes
Before building an execution file, you need to install MS Visual Studio 2015 C++ and Qt, also CUDA for the GPU execution. 
 1. Download the source code from <a href="https://github.com/Openhologram/OpenHologram/tree/master/OpenHolo_DepthMap">here</a>.
 2. Go to the directory 'HologramDepthmap'.
 3. Open the Visual Studio soulution file, 'HologramDepthmap.sln'. 
 4. Check the configuation of the Qt & CUDA to work with the Visual Studio. 
 5. For Qt, you may need to set QTDIR environment variable -> System Properties->Advanced->Environment Variable.
 6. To use FFTW, copy 'libfftw3-3.dll' into the 'bin' directory and copy 'libfftw3-3.lib' into the 'lib' directory.
 7. Visual Studio Build Menu -> Configuration Menu, set "Release" for the Active solution configuration, "x64" for the Active solution platform.
 8. Set 'HologramDepthmap' as a StartUp Project.
 9. Build a Solution.
 10. After building, you can find the execution file, 'HologramDepthmap.exe' under the 'bin' directory.
 11. Execute 'HologramDepthmap.exe', then you can see the following GUI of the sample program. <br><br>
  @image html doc_exe.png "the Sample Program & its Execution"
  @image latex doc_exe.png "the Sample Program & its Execution"

  */

/*
  @section setup How to Install a sample program
  After installing, user can execute the sample program without building the sources and installing Qt & Visual Studio.
  1. Download 'Setup.zip' file from the directory, 'Setup' - setup.exe & Setup.msi
  2. Upzip the zip file.
  3. Execute 'setup.exe'.
  4. Then, the setup process installs a sample program.
  5. User can specify the position that the program is installed.
  6. After finishing the installation, user can find an execution file, 'HologramDepthmap.exe' under the installed directory.
  7. To reinstall the program, first remove the installed program using control panel.
*/

 /**
 * \defgroup init_module Initialize
 * \defgroup load_module Loading Data
 * \defgroup depth_module Computing Depth Value
 * \defgroup trans_module Transform 
 * \defgroup gen_module Generation Hologram
 * \defgroup encode_modulel Encoding
 * \defgroup write_module Writing Image
 * \defgroup recon_module reconstruction
 */

#ifndef __ophDepthMap_h
#define __ophDepthMap_h

#include "ophGen.h"
#include "vec.h"
#include "fftw3.h"

#include <cufft.h>


using namespace oph;

/**
* @brief Structure variable for hologram paramemters
* @details This structure has all necessary parameters for generating a hologram. 
      It is read from the configuration file, 'config_openholo.txt'(Input data).
*/

/** 
* @brief Main class for generating a hologram using depth map data.
* @details This is a main class for generating a digital hologram using depth map data. It is implemented on the CPU and GPU.
*  1. Read Config file. - to set all parameters needed for generating a hologram.
*  2. Initialize all variables. - memory allocation on the CPU and GPU.
*  3. Generate a digital hologram using depth map data.
*  4. For the testing purpose, reconstruct a image from the generated hologram.
*/

class GEN_DLL ophDepthMap : public ophGen {

public:
	explicit ophDepthMap();

protected:
	virtual ~ophDepthMap();

public:
	
	void setMode(bool isCPU);

	/** \ingroup init_module */
	bool readConfig(const char* fname);
	
	/** \ingroup gen_module */
	double generateHologram(void);

	/** \ingroup encode_module */
	void encodeHologram(void);

	/** \ingroup write_module */
	void normalize(void);

	/** \ingroup write_module */
	virtual int save(const char* fname = nullptr, uint8_t bitsperpixel = 24);

	/** \ingroup recon_module */
	void reconstructImage(void);

public:
	/** \ingroup getter/setter */
	inline void setFieldLens(real fieldlens)		{ dm_config_.field_lens		= fieldlens;	}
	/** \ingroup getter/setter */
	inline void setNearDepth(real neardepth)		{ dm_config_.near_depthmap	= neardepth;	}
	/** \ingroup getter/setter */
	inline void setFarDepth(real fardetph)			{ dm_config_.far_depthmap	= fardetph;		}
	/** \ingroup getter/setter */
	inline void setNumOfDepth(uint numofdepth)		{ dm_config_.num_of_depth	= numofdepth;	}

	/** \ingroup getter/setter */
	inline real getFieldLens(void)	{ return dm_config_.field_lens;		}
	/** \ingroup getter/setter */
	inline real getNearDepth(void)	{ return dm_config_.near_depthmap;	}
	/** \ingroup getter/setter */
	inline real getFarDepth(void)	{ return dm_config_.far_depthmap;	}
	/** \ingroup getter/setter */
	inline uint getNumOfDepth(void) { return dm_config_.num_of_depth;	}
	/** \ingroup getter/setter */
	inline void getRenderDepth(std::vector<int>& renderdepth) { renderdepth = dm_config_.render_depth; }

private:
	/** \ingroup init_module */
	void initialize(int numOfFrame);

	/** \ingroup init_module
	* @{ */
	void init_CPU();
	void init_GPU();
	/** @} */

	/** \ingroup load_module
	* @{ */
	int readImageDepth(int ftr);
	bool prepare_inputdata_CPU(uchar* img, uchar* dimg);
	bool prepare_inputdata_GPU(uchar* img, uchar* dimg);
	/** @} */

	/** \ingroup depth_module
	* @{ */
	void getDepthValues();
	void change_depth_quan_CPU();
	void change_depth_quan_GPU();
	/** @} */

	/** \ingroup trans_module
	* @{ */
	void transformViewingWindow();
	/** @} */

	/** \ingroup gen_module 
	* @{ */
	void calc_Holo_by_Depth(int frame);
	void calc_Holo_CPU(int frame);
	void calc_Holo_GPU(int frame);
	void propagation_AngularSpectrum_CPU(oph::Complex<real>* input_u, real propagation_dist);
	void propagation_AngularSpectrum_GPU(cufftDoubleComplex* input_u, real propagation_dist);
	/** @} */

	/** \ingroup encode_module
	* @{ */
	/**
	* @brief Encode the CGH according to a signal location parameter.
	* @param sig_location : ivec2 type,
	*  sig_location[0]: upper or lower half, sig_location[1]:left or right half.
	* @see encoding_CPU, encoding_GPU
	*/
	void encodingSymmetrization(ivec2 sig_location)
	{
		int pnx = context_.pixel_number[0];
		int pny = context_.pixel_number[1];

		int cropx1, cropx2, cropx, cropy1, cropy2, cropy;
		if (sig_location[1] == 0) { //Left or right half
			cropy1 = 1;
			cropy2 = pny;
		}
		else {
			cropy = (int)floor(((real)pny) / 2);
			cropy1 = cropy - (int)floor(((real)cropy) / 2);
			cropy2 = cropy1 + cropy - 1;
		}

		if (sig_location[0] == 0) { // Upper or lower half
			cropx1 = 1;
			cropx2 = pnx;
		}
		else {
			cropx = (int)floor(((real)pnx) / 2);
			cropx1 = cropx - (int)floor(((real)cropx) / 2);
			cropx2 = cropx1 + cropx - 1;
		}

		cropx1 -= 1;
		cropx2 -= 1;
		cropy1 -= 1;
		cropy2 -= 1;

		//if (isCPU_)
			encoding_CPU(cropx1, cropx2, cropy1, cropy2, sig_location);
		//else
		//	encoding_GPU(cropx1, cropx2, cropy1, cropy2, sig_location);


	}
	void encoding_CPU(int cropx1, int cropx2, int cropy1, int cropy2, oph::ivec2 sig_location);
	void encoding_GPU(int cropx1, int cropx2, int cropy1, int cropy2, oph::ivec2 sig_location);
	/** @} */

	void get_rand_phase_value(oph::Complex<real>& rand_phase_val);
	void get_shift_phase_value(oph::Complex<real>& shift_phase_val, int idx, oph::ivec2 sig_location);

	void fftwShift(oph::Complex<real>* src, oph::Complex<real>* dst, fftw_complex* in, fftw_complex* out, int nx, int ny, int type, bool bNomalized = false);
	void fftShift(int nx, int ny, oph::Complex<real>* input, oph::Complex<real>* output);

	//void writeIntensity_gray8_bmp(const char* fileName, int nx, int ny, real* intensity);
	//void writeIntensity_gray8_bmp(const char* fileName, int nx, int ny, Complex* complexvalue);
	//void writeIntensity_gray8_real_bmp(const char* fileName, int nx, int ny, Complex* complexvalue);
	//void writeImage_fromGPU(QString imgname, int pnx, int pny, cufftrealComplex* gpu_data);

	/** \ingroup recon_module
	* @{ */
	void reconstruction(fftw_complex* in, fftw_complex* out);
	void testPropagation2EyePupil(fftw_complex* in, fftw_complex* out);
	void writeSimulationImage(int num, real val);
	void circshift(oph::Complex<real>* in, oph::Complex<real>* out, int shift_x, int shift_y, int nx, int ny);
	/** @} */

	/**

	*/
	void free_gpu(void);

	virtual void ophFree(void);

private:

	bool					isCPU_;								///< if true, it is implemented on the CPU, otherwise on the GPU.

	unsigned char*			img_src_gpu_;						///< GPU variable - image source data, values are from 0 to 255.
	unsigned char*			dimg_src_gpu_;						///< GPU variable - depth map data, values are from 0 to 255.
	real*					depth_index_gpu_;					///< GPU variable - quantized depth map data.
	
	real*					img_src_;							///< CPU variable - image source data, values are from 0 to 1.
	real*					dmap_src_;							///< CPU variable - depth map data, values are from 0 to 1.
	real*					depth_index_;						///< CPU variable - quantized depth map data.
	int*					alpha_map_;							///< CPU variable - calculated alpha map data, values are 0 or 1.

	real*					dmap_;								///< CPU variable - physical distances of depth map.
	int						cur_frame_;
	
	real					dstep_;								///< the physical increment of each depth map layer.
	std::vector<real>		dlevel_;							///< the physical value of all depth map layer.
	std::vector<real>		dlevel_transform_;					///< transfomed dlevel_ variable
	
	OphDepthMapConfig		dm_config_;							///< structure variable for depthmap hologram configuration.
	OphDepthMapParams		dm_params_;							///< structure variable for depthmap hologram parameters.
	OphDepthMapSimul		dm_simuls_;							///< structure variable for depthmap simulation parameters.
};

#endif // !__ophDepthMap_h