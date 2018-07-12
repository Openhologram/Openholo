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
The main function of the library is a  \c \b GenerateHologram() of \c ophDepthMap2 class.
The following is the procedure of it and functions called form it..
<br><br>
@image html doc_proc.png "GenerateHologram Function Procedure"
@image latex doc_proc.png "GenerateHologram Function Procedure"

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
 * \defgroup recon_module Reconstruction
 */

#ifndef __Hologram_Generator_h
#define __Hologram_Generator_h

#include "ophGen.h"
#include <cufft.h>

#include "include.h"

/** 
* @brief Main class for generating a hologram using depth map data.
* @details This is a main class for generating a digital hologram using depth map data. It is implemented on the CPU and GPU.
*  1. Read Config file. - to set all parameters needed for generating a hologram.
*  2. Initialize all variables. - memory allocation on the CPU and GPU.
*  3. Generate a digital hologram using depth map data.
*  4. For the testing purpose, reconstruct a image from the generated hologram.
*/
struct GEN_DLL HologramParams {

double				field_lens;					///< FIELD_LENS at config file  
double				lambda;						///< WAVELENGTH  at config file
double				k;							///< 2 * PI / lambda
ivec2				pn;							///< SLM_PIXEL_NUMBER_X & SLM_PIXEL_NUMBER_Y
vec2				pp;							///< SLM_PIXEL_PITCH_X & SLM_PIXEL_PITCH_Y
vec2				ss;							///< pn * pp

double				near_depthmap;				///< NEAR_OF_DEPTH_MAP at config file
double				far_depthmap;				///< FAR_OF_DEPTH_MAP at config file

uint				num_of_depth;				///< the number of depth level.
												/**< <pre>
												if FLAG_CHANGE_DEPTH_QUANTIZATION == 0
												num_of_depth = DEFAULT_DEPTH_QUANTIZATION
												else
												num_of_depth = NUMBER_OF_DEPTH_QUANTIZATION  </pre> */

std::vector<int>	render_depth;				///< Used when only few specific depth levels are rendered, usually for test purpose
};

using namespace oph;

class GEN_DLL ophDepthMap2 : public ophGen {

public:
	explicit ophDepthMap2();

protected:
	virtual ~ophDepthMap2();

public:

	void setMode(bool is_CPU);

	/** \ingroup init_module */
	bool readConfig(const char* fname);

	/** \ingroup gen_module */
	double generateHologram(void);

	/** \ingroup encode_module */
	void encodeHologram(void);

	/** \ingroup write_module */
	virtual int save(const char* fname = nullptr, uint8_t bitsperpixel = 24);

public:
	/** \ingroup getter/setter */
	inline void setFieldLens(Real fieldlens) { dm_config_.field_lens = fieldlens; }
	/** \ingroup getter/setter */
	inline void setNearDepth(Real neardepth) { dm_config_.near_depthmap = neardepth; }
	/** \ingroup getter/setter */
	inline void setFarDepth(Real fardetph) { dm_config_.far_depthmap = fardetph; }
	/** \ingroup getter/setter */
	inline void setNumOfDepth(uint numofdepth) { dm_config_.num_of_depth = numofdepth; }

	/** \ingroup getter/setter */
	inline Real getFieldLens(void) { return dm_config_.field_lens; }
	/** \ingroup getter/setter */
	inline Real getNearDepth(void) { return dm_config_.near_depthmap; }
	/** \ingroup getter/setter */
	inline Real getFarDepth(void) { return dm_config_.far_depthmap; }
	/** \ingroup getter/setter */
	inline uint getNumOfDepth(void) { return dm_config_.num_of_depth; }
	/** \ingroup getter/setter */
	inline void getRenderDepth(std::vector<int>& renderdepth) { renderdepth = dm_config_.render_depth; }
	
public:
	//void setMode(bool isCPU);

	/** \ingroup init_module */
	//bool readConfig(bool isXML);

	/** \ingroup init_module */
	void initialize();
	
	/** \ingroup gen_module */
	void GenerateHologram();

	/** \ingroup recon_module */
	//void ReconstructImage();

	//void writeMatFileComplex(const char* fileName, Complex<Real>* val);							
	//void writeMatFileDouble(const char* fileName, double * val);
	//bool readMatFileDouble(const char* fileName, double * val);

private:

	/** \ingroup init_module
	* @{ */
	void initCPU();   
	//void initGPU();
	/** @} */

	/** \ingroup load_module
	* @{ */
	bool readImageDepth(void);
	bool prepareInputdataCPU(uchar* img, uchar* dimg);
	//bool prepareInputdataGPU(uchar* img, uchar* dimg);
	/** @} */

	/** \ingroup depth_module
	* @{ */
	void getDepthValues();
	void changeDepthQuanCPU();
	//void changeDepthQuanGPU();
	/** @} */

	/** \ingroup trans_module
	* @{ */
	void transformViewingWindow();
	/** @} */

	/** \ingroup gen_module 
	* @{ */
	void calcHoloByDepth(void);
	void calcHoloCPU(void);
	//void calcHoloGPU(int frame);
	void PropagationAngularSpectrumCPU(Complex<Real>* input_u, double propagation_dist);
	//void PropagationAngularSpectrumGPU(cufftDoubleComplex* input_u, double propagation_dist);
	void exponent_complex(Complex<Real>* val);


private:
	bool					is_CPU;								///< if true, it is implemented on the CPU, otherwise on the GPU.

	unsigned char*			img_src_gpu;						///< GPU variable - image source data, values are from 0 to 255.
	unsigned char*			dimg_src_gpu;						///< GPU variable - depth map data, values are from 0 to 255.
	Real*					depth_index_gpu;					///< GPU variable - quantized depth map data.

	Real*					img_src;							///< CPU variable - image source data, values are from 0 to 1.
	Real*					dmap_src;							///< CPU variable - depth map data, values are from 0 to 1.
	Real*					depth_index;						///< CPU variable - quantized depth map data.
	int*					alpha_map;							///< CPU variable - calculated alpha map data, values are 0 or 1.

	Real*					dmap;								///< CPU variable - physical distances of depth map.

	Real					dstep;								///< the physical increment of each depth map layer.
	std::vector<Real>		dlevel;							///< the physical value of all depth map layer.
	std::vector<Real>		dlevel_transform;					///< transfomed dlevel variable

	OphDepthMapConfig		dm_config_;							///< structure variable for depthmap hologram configuration.
	OphDepthMapParams		dm_params_;							///< structure variable for depthmap hologram parameters.
																//OphDepthMapSimul		dm_simuls_;							///< structure variable for depthmap simulation parameters.

};


#endif