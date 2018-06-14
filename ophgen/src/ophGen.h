/**
* @mainpage ophGen
* @brief Abstract class for generation classes
*/

#ifndef __ophGen_h
#define __ophGen_h

#include "Openholo.h"

#include "fftw3.h"

#ifdef GEN_EXPORT
#define GEN_DLL __declspec(dllexport)
#else
#define GEN_DLL __declspec(dllimport)
#endif

struct GEN_DLL OphContext {
	oph::ivec2		pixel_number;				///< SLM_PIXEL_NUMBER_X & SLM_PIXEL_NUMBER_Y
	oph::vec2		pixel_pitch;				///< SLM_PIXEL_PITCH_X & SLM_PIXEL_PITCH_Y

	real			k;							///< 2 * PI / lambda(wavelength)
	oph::vec2		ss;							///< pn * pp

	real			lambda;						///< wave length
};

struct OphPointCloudConfig;
struct OphPointCloudData;
struct OphDepthMapConfig;
struct OphDepthMapParams;
struct OphDepthMapSimul;

class GEN_DLL ophGen : public Openholo
{
protected:
	enum DIFF_FLAG {
		DIFF_RS,
		DIFF_FRESNEL,
	};

	enum ENCODE_FLAG {
		SingleSideBand,
		NumericalInterference,
		PhaseToAmplitude,
		Real,
		OffAxisSSB,
		Burckhardt,
		TwoPhase,
	};

	enum PC_FLAG {
		PC_XYZ			= 1,
		PC_RGB			= 2,
		PC_PHASE		= 4,
		PC_AMPLITUDE	= 8,
	};

public:
	/**
	* @brief Constructor
	*/
	explicit ophGen(void);

protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophGen(void) = 0;

public:
	inline oph::Complex<real>* getGenBuffer(void) { return holo_gen; }
	inline real* getEncodedBuffer(void) { return holo_encoded; }
	inline uchar* getNormalizedBuffer(void) { return holo_normalized; }

	inline void setPixelNumber(int nx, int ny) { context_.pixel_number.v[0] = nx; context_.pixel_number.v[1] = ny; }
	inline void setPixelPitch(real px, real py) { context_.pixel_pitch.v[0] = px; context_.pixel_pitch.v[1] = py; }
	inline void setWaveLength(real w) { context_.lambda = w; }

	OphContext& getContext(void) { return context_; }

	/**
	* @param input parameter. point cloud data file name
	* @param output parameter. point cloud data, vertices(x0, y0, z0, x1, y1, z1, ...) container's pointer
	* @param output parameter. point cloud data, amplitudes container's pointer
	* @param output parameter. point cloud data, phases container's pointer
	* @return positive integer is points number of point cloud, return a negative integer if the load fails
	*/
	int loadPointCloud(const char* pc_file, OphPointCloudData *pc_data_, uint flag);

	/**
	* @param input parameter. configuration data file name
	* @param output parameter. OphConfigParams struct variable can get configuration data
	*/
	virtual bool readConfig(const char* fname, OphPointCloudConfig& config);
	virtual bool readConfig(const char* fname, OphDepthMapConfig& config, OphDepthMapParams& params, OphDepthMapSimul& simuls);

	virtual void normalize(void);

	/** \ingroup write_module */
	virtual int save(const char* fname, uint8_t bitsperpixel = 8, uchar* src = nullptr, uint px = 0, uint py = 0);
	virtual int load(const char* fname, void* dst = nullptr);

	/**	*/
	void fft2(int n0, int n1, const oph::Complex<real>* in, oph::Complex<real>* out, int sign = FFTW_FORWARD, unsigned int flag = FFTW_ESTIMATE);

protected:
	/** 
	* @brief Called when saving multiple hologram data at a time
	*/
	virtual int save(const char* fname, uint8_t bitsperpixel, uint px, uint py, uint fnum, uchar* args ...);

protected:
	OphContext				context_;

	oph::Complex<real>*		holo_gen;
	real*					holo_encoded;
	oph::uchar*				holo_normalized;

public:
	/**
	* @brief Encoding Functions
	*/

	/** @brief Phase and Amplitude */
	void calPhase(oph::Complex<real>* holo, real* encoded, const vec2 holosize);
	void calAmplitude(oph::Complex<real>* holo, real* encoded, const vec2 holosize);

	/** @brief Single Side Band Encoding */
	enum passband {left, rig, top, btm};
	//void singleSideBand(oph::Complex<real>* holo, real* encoded, const vec2 holosize, int passband);
	
	/** @brief Numerical Interface */
	void numericalInterference(oph::Complex<real>* holo, real* encoded, const vec2 holosize);
	void numericalInterference(void);

	/** @brief Two Phase Encoding */
	/**
	* @param output parameter(encoded) : (sizeX*2, sizeY)
	*/
	void twoPhaseEncoding(oph::Complex<real>* holo, real* encoded, const vec2 holosize);
	
	/** @brief Burckhardt Encoding */
	/**
	* @param output parameter(encoded) : (sizeX*3, sizeY)
	*/
	void burckhardt(oph::Complex<real>* holo, real* encoded, const vec2 holosize);
	
	/** @brief Frequency Shift */
	//void freqShift(oph::Complex<real>* holo, Complex<real>* encoded, const vec2 holosize, int shift_x, int shift_y);

protected:
	/** \ingroup encode_module
	/**
	* @brief Encode the CGH according to a signal location parameter.
	* @param sig_location : ivec2 type,
	*  sig_location[0]: upper or lower half, sig_location[1]:left or right half.
	* @see encoding_CPU, encoding_GPU
	*/
	void encodeSideBand(bool bCPU, ivec2 sig_location);
	/**
	* @brief Encode the CGH according to a signal location parameter on the CPU.
	* @details The CPU variable, holo_gen on CPU has the final result.
	* @param cropx1 : the start x-coordinate to crop.
	* @param cropx2 : the end x-coordinate to crop.
	* @param cropy1 : the start y-coordinate to crop.
	* @param cropy2 : the end y-coordinate to crop.
	* @param sig_location : ivec2 type,
	*  sig_location[0]: upper or lower half, sig_location[1]:left or right half.
	* @see encodingSymmetrization, fftwShift
	*/
	void encodeSideBand_CPU(int cropx1, int cropx2, int cropy1, int cropy2, oph::ivec2 sig_location);
	void encodeSideBand_GPU(int cropx1, int cropx2, int cropy1, int cropy2, oph::ivec2 sig_location);

	/**
	* @brief Calculate the shift phase value.
	* @param shift_phase_val : output variable.
	* @param idx : the current pixel position.
	* @param sig_location :  signal location.
	* @see encodingSideBand_CPU
	*/
	void get_shift_phase_value(oph::Complex<real>& shift_phase_val, int idx, oph::ivec2 sig_location);

	/**
	* @brief Assign random phase value if RANDOM_PHASE == 1
	* @details If RANDOM_PHASE == 1, calculate a random phase value using random generator;
	*  otherwise, random phase value is 1.
	* @param rand_phase_val : Input & Ouput value.
	*/
	void get_rand_phase_value(oph::Complex<real>& rand_phase_val, bool rand_phase);

	/**
	* @brief Convert data from the spatial domain to the frequency domain using 2D FFT on CPU.
	* @details It is equivalent to Matlab code, dst = ifftshift(fft2(fftshift(src))).
	* @param src : input data variable
	* @param dst : output data variable
	* @param in : input data pointer connected with FFTW plan
	* @param out : ouput data pointer connected with FFTW plan
	* @param nx : the number of column of the input data
	* @param ny : the number of row of the input data
	* @param type : If type == 1, forward FFT, if type == -1, backward FFT.
	* @param bNomarlized : If bNomarlized == true, normalize the result after FFT.
	* @see propagation_AngularSpectrum_CPU, encoding_CPU
	*/
	void fftwShift(oph::Complex<real>* src, oph::Complex<real>* dst, fftw_complex* in, fftw_complex* out, int nx, int ny, int type, bool bNormalized = false);

	/**
	* @brief Swap the top-left quadrant of data with the bottom-right , and the top-right quadrant with the bottom-left.
	* @param nx : the number of column of the input data
	* @param ny : the number of row of the input data
	* @param input : input data variable
	* @param output : output data variable
	* @see fftwShift
	*/
	void fftShift(int nx, int ny, oph::Complex<real>* input, oph::Complex<real>* output);

protected:
	/**
	* @brief Pure virtual function for override in child classes
	*/
	virtual void ophFree(void);
};

#endif // !__ophGen_h

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
struct GEN_DLL OphPointCloudData {
	vec3* location;
	ivec3* color;
	real* amplitude;
	real* phase;

	OphPointCloudData() :location(nullptr), color(nullptr), amplitude(nullptr), phase(nullptr) {}
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

	OphDepthMapConfig() :field_lens(0), near_depthmap(0), far_depthmap(0), num_of_depth(0) {}
	//test commit
};
struct GEN_DLL OphDepthMapParams
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
struct GEN_DLL OphDepthMapSimul
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