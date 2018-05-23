/**
* @mainpage ophGen
* @brief Abstract class for generation classes
*/

#ifndef __ophGen_h
#define __ophGen_h

#include "Openholo.h"

#include "complex.h"

#ifdef GEN_EXPORT
#define GEN_DLL __declspec(dllexport)
#else
#define GEN_DLL __declspec(dllimport)
#endif

struct GEN_DLL OphPointCloudConfig {
	oph::vec3 scale;								///< Scaling factor of coordinate of point cloud
	real offset_depth;								///< Offset value of point cloud

	int8_t* filter_shape_flag;						///< Shape of spatial bandpass filter ("Circle" or "Rect" for now)
	oph::vec2 filter_width;							///< Width of spatial bandpass filter

	real focal_length_lens_in;						///< Focal length of input lens of Telecentric
	real focal_length_lens_out;						///< Focal length of output lens of Telecentric
	real focal_length_lens_eye_piece;				///< Focal length of eyepiece lens				

	oph::vec2 tilt_angle;							///< Tilt angle for spatial filtering
};

struct GEN_DLL OphDepthMapConfig {
	real				field_lens;					///< FIELD_LENS at config file

	real				near_depthmap;				///< NEAR_OF_DEPTH_MAP at config file
	real				far_depthmap;				///< FAR_OF_DEPTH_MAP at config file
	
	oph::uint			num_of_depth;				///< the number of depth level.
													/**< <pre>
													if FLAG_CHANGE_DEPTH_QUANTIZATION == 0
													num_of_depth = DEFAULT_DEPTH_QUANTIZATION
													else
													num_of_depth = NUMBER_OF_DEPTH_QUANTIZATION  </pre> */

	std::vector<int>	render_depth;				///< Used when only few specific depth levels are rendered, usually for test purpose

	OphDepthMapConfig():field_lens(0), near_depthmap(0), far_depthmap(0), num_of_depth(0) {}
};

struct OphDepthMapParams
{
	std::string				SOURCE_FOLDER;						///< input source folder - config file.
	std::string				IMAGE_PREFIX;						///< the prefix of the input image file - config file.
	std::string				DEPTH_PREFIX;						///< the prefix of the deptmap file - config file	
	std::string				RESULT_FOLDER;						///< the name of the result folder - config file
	std::string				RESULT_PREFIX;						///< the prefix of the result file - config file
	bool					FLAG_STATIC_IMAGE;					///< if true, the input image is static.
	oph::uint				START_OF_FRAME_NUMBERING;			///< the start frame number.
	oph::uint				NUMBER_OF_FRAME;					///< the total number of the frame.	
	oph::uint				NUMBER_OF_DIGIT_OF_FRAME_NUMBERING; ///< the number of digit of frame number.

	int						Transform_Method_;					///< transform method 
	int						Propagation_Method_;				///< propagation method - currently AngularSpectrum
	int						Encoding_Method_;					///< encoding method - currently Symmetrization

	bool					FLAG_CHANGE_DEPTH_QUANTIZATION;		///< if true, change the depth quantization from the default value.
	oph::uint				DEFAULT_DEPTH_QUANTIZATION;			///< default value of the depth quantization - 256
	oph::uint				NUMBER_OF_DEPTH_QUANTIZATION;		///< depth level of input depthmap.
	bool					RANDOM_PHASE;						///< If true, random phase is imposed on each depth layer.
};

struct OphDepthMapSimul
{
	// for Simulation (reconstruction)
	//===================================================
	std::string				Simulation_Result_File_Prefix_;		///< reconstruction variable for testing
	int						test_pixel_number_scale_;			///< reconstruction variable for testing
	oph::vec2				Pixel_pitch_xy_;					///< reconstruction variable for testing
	oph::ivec2				SLM_pixel_number_xy_;				///< reconstruction variable for testing
	real					f_field_;							///< reconstruction variable for testing
	real					eye_length_;						///< reconstruction variable for testing
	real					eye_pupil_diameter_;				///< reconstruction variable for testing
	oph::vec2				eye_center_xy_;						///< reconstruction variable for testing
	real					focus_distance_;					///< reconstruction variable for testing
	int						sim_type_;							///< reconstruction variable for testing
	real					sim_from_;							///< reconstruction variable for testing
	real					sim_to_;							///< reconstruction variable for testing
	int						sim_step_num_;						///< reconstruction variable for testing
	real*					sim_final_;							///< reconstruction variable for testing
	oph::Complex<real>*		hh_complex_;						///< reconstruction variable for testing
};

class GEN_DLL ophGen : public Openholo
{
public:
	/**
	* @brief Constructor
	*/
	ophGen(void);

protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophGen(void) = 0;

public:
	inline oph::Complex<real>* getGenBuffer(void) { return holo_gen; }
	inline real* getEncodedBuffer(void) { return holo_encoded; }
	inline uchar* getNormalizedBuffer(void) { return holo_normalized; }

	/**
	* @param input parameter. point cloud data file name
	* @param output parameter. point cloud data, vertices(x0, y0, z0, x1, y1, z1, ...) container's pointer
	* @param output parameter. point cloud data, amplitudes container's pointer
	* @param output parameter. point cloud data, phases container's pointer
	* @return positive integer is points number of point cloud, return a negative integer if the load fails
	*/
	virtual int loadPointCloud(const std::string pc_file, std::vector<real> *vertex_array, std::vector<real> *amplitude_array, std::vector<real> *phase_array);

	/**
	* @param input parameter. configuration data file name
	* @param output parameter. OphConfigParams struct variable can get configuration data
	*/
	virtual bool readConfig(const std::string fname, OphPointCloudConfig& config);
	virtual bool readConfig(const std::string fname, OphDepthMapConfig& config, OphDepthMapParams& params, OphDepthMapSimul& simuls);

	virtual void normalize(void);

	virtual int save(const char* fname, uint8_t bitsperpixel = 8);
	virtual int load(const char* fname, void* dst);

protected:
	oph::Complex<real>*		holo_gen;
	real*					holo_encoded;
	oph::uchar*				holo_normalized;

protected:
	/**
	* @brief Pure virtual function for override in child classes
	*/
	virtual void ophFree(void);
};

#endif // !__ophGen_h