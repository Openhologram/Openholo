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

#ifndef __OphDepthMap_h
#define __OphDepthMap_h

#include "ophGen.h"
#include "vec.h"
#include "complex.h"
#include "fftw3.h"

#include <cufft.h>


using namespace oph;

/**
* @brief Structure variable for hologram paramemters
* @details This structure has all necessary parameters for generating a hologram. 
      It is read from the configuration file, 'config_openholo.txt'.
*/
struct GEN_DLL HologramParams{
	
	double				field_lens;					///< FIELD_LENS at config file  
	double				lambda;						///< WAVELENGTH  at config file
	double				k;							///< 2 * PI / lambda
	oph::ivec2			pn;							///< SLM_PIXEL_NUMBER_X & SLM_PIXEL_NUMBER_Y
	oph::vec2			pp;							///< SLM_PIXEL_PITCH_X & SLM_PIXEL_PITCH_Y
	oph::vec2			ss;							///< pn * pp

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

	ophDepthMap();
	~ophDepthMap();
	
	void setMode(bool isCPU);

	/** \ingroup init_module */
	bool readConfig();

	/** \ingroup init_module */
	void initialize();
	
	/** \ingroup gen_module */
	void generateHologram();

	/** \ingroup recon_module */
	void reconstructImage();

	//void writeMatFileComplex(const char* fileName, Complex* val);							
	//void writeMatFileDouble(const char* fileName, double * val);
	//bool readMatFileDouble(const char* fileName, double * val);

private:

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
	void propagation_AngularSpectrum_CPU(oph::Complex* input_u, double propagation_dist);
	void propagation_AngularSpectrum_GPU(cufftDoubleComplex* input_u, double propagation_dist);
	/** @} */

	/** \ingroup encode_modulel
	* @{ */
	/**
	* @brief Encode the CGH according to a signal location parameter.
	* @param sig_location : ivec2 type,
	*  sig_location[0]: upper or lower half, sig_location[1]:left or right half.
	* @see encoding_CPU, encoding_GPU
	*/
	void encodingSymmetrization(ivec2 sig_location)
	{
		int pnx = params_.pn[0];
		int pny = params_.pn[1];

		int cropx1, cropx2, cropx, cropy1, cropy2, cropy;
		if (sig_location[1] == 0) //Left or right half
		{
			cropy1 = 1;
			cropy2 = pny;

		}
		else {

			cropy = floor(pny / 2);
			cropy1 = cropy - floor(cropy / 2);
			cropy2 = cropy1 + cropy - 1;
		}

		if (sig_location[0] == 0) // Upper or lower half
		{
			cropx1 = 1;
			cropx2 = pnx;

		}
		else {

			cropx = floor(pnx / 2);
			cropx1 = cropx - floor(cropx / 2);
			cropx2 = cropx1 + cropx - 1;
		}

		cropx1 -= 1;
		cropx2 -= 1;
		cropy1 -= 1;
		cropy2 -= 1;

		if (isCPU_)
			encoding_CPU(cropx1, cropx2, cropy1, cropy2, sig_location);
		else
			encoding_GPU(cropx1, cropx2, cropy1, cropy2, sig_location);


	}
	void encoding_CPU(int cropx1, int cropx2, int cropy1, int cropy2, oph::ivec2 sig_location);
	void encoding_GPU(int cropx1, int cropx2, int cropy1, int cropy2, oph::ivec2 sig_location);
	/** @} */

	/** \ingroup write_module
	* @{ */
	void writeResultimage(int ftr);
	/** @} */

	void get_rand_phase_value(oph::Complex& rand_phase_val);
	void get_shift_phase_value(oph::Complex& shift_phase_val, int idx, oph::ivec2 sig_location);

	void fftwShift(Complex* src, Complex* dst, fftw_complex* in, fftw_complex* out, int nx, int ny, int type, bool bNomalized = false);
	void exponent_complex(oph::Complex* val);
	void fftShift(int nx, int ny, oph::Complex* input, oph::Complex* output);

	//void writeIntensity_gray8_bmp(const char* fileName, int nx, int ny, double* intensity);
	//void writeIntensity_gray8_bmp(const char* fileName, int nx, int ny, Complex* complexvalue);
	//void writeIntensity_gray8_real_bmp(const char* fileName, int nx, int ny, Complex* complexvalue);
	//void writeImage_fromGPU(QString imgname, int pnx, int pny, cufftDoubleComplex* gpu_data);

	/** \ingroup recon_module
	* @{ */
	void reconstruction(fftw_complex* in, fftw_complex* out);
	void testPropagation2EyePupil(fftw_complex* in, fftw_complex* out);
	void writeSimulationImage(int num, double val);
	void circshift(oph::Complex* in, oph::Complex* out, int shift_x, int shift_y, int nx, int ny);
	/** @} */

	virtual void ophFree(void);

private:

	bool					isCPU_;						///< if true, it is implemented on the CPU, otherwise on the GPU.

	unsigned char*			img_src_gpu_;				///< GPU variable - image source data, values are from 0 to 255.
	unsigned char*			dimg_src_gpu_;				///< GPU variable - depth map data, values are from 0 to 255.
	double*					depth_index_gpu_;			///< GPU variable - quantized depth map data.
	
	double*					img_src_;					///< CPU variable - image source data, values are from 0 to 1.
	double*					dmap_src_;					///< CPU variable - depth map data, values are from 0 to 1.
	double*					depth_index_;				///< CPU variable - quantized depth map data.
	int*					alpha_map_;					///< CPU variable - calculated alpha map data, values are 0 or 1.

	double*					dmap_;						///< CPU variable - physical distances of depth map.
	
	double					dstep_;						///< the physical increment of each depth map layer.
	std::vector<double>		dlevel_;					///< the physical value of all depth map layer.
	std::vector<double>		dlevel_transform_;			///< transfomed dlevel_ variable
	
	oph::Complex*			U_complex_;					///< CPU variable - the generated hologram before encoding.
	double*					u255_fringe_;				///< the final hologram, used for writing the result image.

	HologramParams			params_;					///< structure variable for hologram parameters

	std::string				SOURCE_FOLDER;				///< input source folder - config file.
	std::string				IMAGE_PREFIX;				///< the prefix of the input image file - config file.
	std::string				DEPTH_PREFIX;				///< the prefix of the deptmap file - config file	
	std::string				RESULT_FOLDER;				///< the name of the result folder - config file
	std::string				RESULT_PREFIX;				///< the prefix of the result file - config file
	bool					FLAG_STATIC_IMAGE;			///< if true, the input image is static.
	uint					START_OF_FRAME_NUMBERING;	///< the start frame number.
	uint					NUMBER_OF_FRAME;			///< the total number of the frame.	
	uint					NUMBER_OF_DIGIT_OF_FRAME_NUMBERING; ///< the number of digit of frame number.

	int						Transform_Method_;			///< transform method 
	int						Propagation_Method_;		///< propagation method - currently AngularSpectrum
	int						Encoding_Method_;			///< encoding method - currently Symmetrization

	double					WAVELENGTH;					///< wave length

	bool					FLAG_CHANGE_DEPTH_QUANTIZATION;	///< if true, change the depth quantization from the default value.
	uint					DEFAULT_DEPTH_QUANTIZATION;		///< default value of the depth quantization - 256
	uint					NUMBER_OF_DEPTH_QUANTIZATION;   ///< depth level of input depthmap.
	bool					RANDOM_PHASE;					///< If true, random phase is imposed on each depth layer.

	// for Simulation (reconstruction)
	//===================================================
	std::string				Simulation_Result_File_Prefix_;	///< reconstruction variable for testing
	int						test_pixel_number_scale_;		///< reconstruction variable for testing
	oph::vec2				Pixel_pitch_xy_;				///< reconstruction variable for testing
	oph::ivec2				SLM_pixel_number_xy_;			///< reconstruction variable for testing
	double					f_field_;						///< reconstruction variable for testing
	double					eye_length_;					///< reconstruction variable for testing
	double					eye_pupil_diameter_;			///< reconstruction variable for testing
	oph::vec2				eye_center_xy_;					///< reconstruction variable for testing
	double					focus_distance_;				///< reconstruction variable for testing
	int						sim_type_;						///< reconstruction variable for testing
	double					sim_from_;						///< reconstruction variable for testing
	double					sim_to_;						///< reconstruction variable for testing
	int						sim_step_num_;					///< reconstruction variable for testing
	double*					sim_final_;						///< reconstruction variable for testing
	oph::Complex*			hh_complex_;					///< reconstruction variable for testing

};


#endif