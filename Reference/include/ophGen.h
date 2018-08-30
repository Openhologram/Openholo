#ifndef __ophGen_h
#define __ophGen_h

#include "Openholo.h"

#ifdef GEN_EXPORT
#define GEN_DLL __declspec(dllexport)
#else
#define GEN_DLL __declspec(dllimport)
#endif

#pragma comment(lib, "libfftw3-3.lib")

struct GEN_DLL OphContext {
	oph::ivec2		pixel_number;				///< SLM_PIXEL_NUMBER_X & SLM_PIXEL_NUMBER_Y
	oph::vec2		pixel_pitch;				///< SLM_PIXEL_PITCH_X & SLM_PIXEL_PITCH_Y

	Real			k;							///< 2 * PI / lambda(wavelength)
	vec2			ss;							///< pn * pp

	Real			lambda;						///< wave length
};

struct OphPointCloudConfig;
struct OphPointCloudData;
struct OphDepthMapConfig;
struct OphMeshData;
struct OphWRPConfig;

enum PC_DIFF_FLAG {
	PC_DIFF_RS_ENCODED,
	PC_DIFF_FRESNEL_ENCODED,
	PC_DIFF_RS_NOT_ENCODED,
	PC_DIFF_FRESNEL_NOT_ENCODED,
};

/**
* @brief Abstract class for generation classes
*/
class GEN_DLL ophGen : public Openholo
{
public:

	enum ENCODE_FLAG {
		ENCODE_PHASE,
		ENCODE_AMPLITUDE,
		ENCODE_REAL,
		ENCODE_SIMPLENI,
		ENCODE_BURCKHARDT,
		ENCODE_TWOPHASE,
		ENCODE_SSB,
		ENCODE_OFFSSB,
	};

public:
	/**
	* \ingroup const/dest
	* @brief Constructor
	*/
	explicit ophGen(void);

protected:
	/**
	* \ingroup const/dest
	* @brief Destructor
	*/
	virtual ~ophGen(void) = 0;

public:
	/**
	* \ingroup get/set
	*/
	inline oph::Complex<Real>* getHoloBuffer(void) { return holo_gen; }
	/**
	* \ingroup get/set
	*/
	inline Real* getEncodedBuffer(void) { return holo_encoded; }
	/**
	* \ingroup get/set
	*/
	inline uchar* getNormalizedBuffer(void) { return holo_normalized; }

	/**
	* \ingroup get/set
	*/
	inline void setPixelNumber(int nx, int ny) { context_.pixel_number[_X] = nx; context_.pixel_number[_Y] = ny; }

	/**
	* \ingroup get/set
	*/
	inline void setPixelPitch(Real px, Real py) { context_.pixel_pitch[_X] = px; context_.pixel_pitch[_Y] = py; }

	/**
	* \ingroup get/set
	*/
	inline void setWaveLength(Real w) { context_.lambda = w; }

	/**
	* \ingroup get/set
	*/
	OphContext& getContext(void) { return context_; }

	/**
	* \ingroup init
	* @brief Initialize variables for Hologram complex field, encoded data, normalized data
	*/
	void initialize(void);

	/**
	* \ingroup read
	* @param const char* Point cloud data file name
	* @param OphPointCloudData* Point cloud data - number of points, number of color, geometry of point cloud, color data, phase data
	* @return Positive integer is points number of point cloud, return a negative integer if the load fails
	*/
	int loadPointCloud(const char* pc_file, OphPointCloudData *pc_data_);

	/**
	* \ingroup read
	* @param const char* Input file name
	* @param OphPointCloudConfig& Config structures variable can get configuration data
	*/
	bool readConfig(const char* fname, OphPointCloudConfig& config);

	/**
	* \ingroup read
	* @param const char* Input file name
	* @param OphDepthMapConfig& Config structures variable can get configuration data
	*/
	bool readConfig(const char* fname, OphDepthMapConfig& config);
	bool readConfig(const char* fname, OphWRPConfig& config);

	/**
	* \ingroup calc
	* @brief Normalization function to save as image file after hologram creation
	*/
	void normalize(void);

	/**
	* \ingroup write
	* @brief Function for saving image files
	*/
	int save(const char* fname, uint8_t bitsperpixel = 8, uchar* src = nullptr, uint px = 0, uint py = 0);
	
	/** \ingroup read
	* @brief Function for loading image files
	*/
	void* load(const char* fname);

protected:
	/**
	* \ingroup write
	* @brief Called when saving multiple hologram data at a time
	*/
	int save(const char* fname, uint8_t bitsperpixel, uint px, uint py, uint fnum, uchar* args ...);

protected:
	OphContext				context_;

	oph::Complex<Real>*		holo_gen;
	Real*					holo_encoded;
	oph::uchar*				holo_normalized;

public:
	/**
	* @brief	Encoding Functions
	* @details
	*	ENCODE_PHASE		:	Phase
	*	ENCODE_AMPLITUDE	:	Amplitude
	*	ENCODE_REAL			:	Real Part
	*	ENCODE_SIMPLENI		:	Simple numerical interference
	*	ENCODE_BURCKHARDT	:	Burckhardt encoding
	*							@see C.B. Burckhardt, ¡°A simplification of Lee¡¯s method of generating holograms by computer,¡± Applied Optics, vol. 9, no. 8, pp. 1949-1949, 1970.
	*	ENCODE_TWOPHASE		:	Two Phase Encoding
	* @return	holo_encoded
	*	ENCODE_BURCKHARDT - (holosizeX*3, holosizeY)
	*	ENCODE_TWOPHASE - (holosizeX*2, holosizeY)
	*	else - (holosizeX, holosizeY)
	* @overload
	*/
	void encoding(unsigned int ENCODE_FLAG);
	/*
	* @brief	Encoding Functions
	* @details
	*	 ENCODE_SSB		:	Single Side Band Encoding
	*	 ENCODE_OFFSSB	:	Off-axis + Single Side Band Encoding
	* @param	SSB_PASSBAND : SSB_LEFT, SSB_RIGHT, SSB_TOP, SSB_BOTTOM
	* @overload
	*/
	void encoding(unsigned int ENCODE_FLAG, unsigned int SSB_PASSBAND);
	void encoding();
	enum SSB_PASSBAND { SSB_LEFT, SSB_RIGHT, SSB_TOP, SSB_BOTTOM };

protected:
	/**
	* @param	encode_size		Encoded hologram size, varied from encoding type
	* @param	ENCODE_METHOD	Encodinng method flag
	* @param	SSB_PASSBAND	Passband in single side band encoding
	*/

	ivec2 encode_size;
	int ENCODE_METHOD;
	int SSB_PASSBAND;
public:
	/** \ingroup */
	void setEncodeMethod(int in) { ENCODE_METHOD = in; }
	/** \ingroup */
	void setSSBPassBand(int in){ SSB_PASSBAND = in; }
	/** \ingroup */
	ivec2& getEncodeSize(void) { return encode_size; }

public:
	/**
	* @brief	Complex field file load
	* @details	Just used for the reference
	*/
	void loadComplex(char* real_file, char* imag_file, int n_x, int n_y);
	/**
	* @brief	Normalize the encoded hologram
	* @details	Considering the encoded hologram size
	*/
	void normalizeEncoded(void);

	void fourierTest() {
		fft2(context_.pixel_number, holo_gen, OPH_FORWARD, OPH_ESTIMATE);
		fftExecute(holo_gen);
		fft2(context_.pixel_number, holo_gen, OPH_BACKWARD, OPH_ESTIMATE);
		fftExecute(holo_gen);
	}
	void fresnelTest(Real dis) {
		context_.lambda = 532e-9;
		context_.pixel_pitch = { 8e-6, 8e-6 };
		fresnelPropagation(context_,holo_gen, holo_gen, dis);
	}

protected:
	/**
	* @brief	Encoding functions
	*/

	void numericalInterference(oph::Complex<Real>* holo, Real* encoded, const int size);
	void twoPhaseEncoding(oph::Complex<Real>* holo, Real* encoded, const int size);
	void burckhardt(oph::Complex<Real>* holo, Real* encoded, const int size);
	void singleSideBand(oph::Complex<Real>* holo, Real* encoded, const ivec2 holosize, int passband);

	/**
	* @brief	Frequency shift
	*/
	void freqShift(oph::Complex<Real>* src, Complex<Real>* dst, const ivec2 holosize, int shift_x, int shift_y);
public:
	/**
	* @brief	Fresnel propagation
	* @param	OphContext		context		OphContext structure
	* @param	Complex<Real>*	in			Input complex field
	* @param	Complex<Real>*	out			Output complex field
	* @param	Real			distance	Propagation distance
	* @return	out
	*/
	void fresnelPropagation(OphContext context, Complex<Real>* in, Complex<Real>* out, Real distance);
	void fresnelPropagation(Complex<Real>* in, Complex<Real>* out, Real distance);
protected:
	/** 
	* \ingroup encode
	* @brief Encode the CGH according to a signal location parameter.
	* @param bool Select whether to operate with CPU or GPU
	* @param oph::ivec2 sig_location[0]: upper or lower half, sig_location[1]:left or right half.
	* @see encodeSideBand_CPU, encodeSideBand_GPU
	*/
	void encodeSideBand(bool bCPU, ivec2 sig_location);
	/**
	* \ingroup encode
	* @brief Encode the CGH according to a signal location parameter on the CPU.
	* @details The CPU variable, holo_gen on CPU has the final result.
	* @param int the start x-coordinate to crop
	* @param int the end x-coordinate to crop
	* @param int the start y-coordinate to crop
	* @param int the end y-coordinate to crop
	* @param oph::ivec2 Signal location
	*  sig_location[0]: upper or lower half, sig_location[1]:left or right half.
	* @see encodingSymmetrization, fftwShift
	*/
	void encodeSideBand_CPU(int cropx1, int cropx2, int cropy1, int cropy2, oph::ivec2 sig_location);
	void encodeSideBand_GPU(int cropx1, int cropx2, int cropy1, int cropy2, oph::ivec2 sig_location);

	/**
	* \ingroup calc
	* @brief Calculate the shift phase value.
	* @param shift_phase_val : output variable.
	* @param idx : the current pixel position.
	* @param sig_location :  signal location.
	* @see encodingSideBand_CPU
	*/
	void getShiftPhaseValue(oph::Complex<Real>& shift_phase_val, int idx, oph::ivec2 sig_location);

	/**
	* \ingropu calc
	* @brief Assign random phase value if RANDOM_PHASE == 1
	* @details If RANDOM_PHASE == 1, calculate a random phase value using random generator;
	*  otherwise, random phase value is 1.
	* @param rand_phase_val : Input & Ouput value.
	*/
	void getRandPhaseValue(oph::Complex<Real>& rand_phase_val, bool rand_phase);

protected:
	/**
	* @brief Pure virtual function for override in child classes
	*/
	virtual void ophFree(void);
};


/**
* @param oph::vec3 Scaling factor of coordinate of point cloud
* @param Real Offset value of point cloud
* @param int8_t* Shape of spatial bandpass filter ("Circle" or "Rect" for now)
* @param oph::vec2 Width of spatial bandpass filter
* @param Real Focal length of input lens of Telecentric
* @param Real Focal length of output lens of Telecentric
* @param Real Focal length of eyepiece lens
* @param oph::vec2 Tilt angle for spatial filtering
*/
struct GEN_DLL OphPointCloudConfig {
	oph::vec3 scale;
	Real offset_depth;

	int8_t* filter_shape_flag;
	oph::vec2 filter_width;

	Real focal_length_lens_in;
	Real focal_length_lens_out;
	Real focal_length_lens_eye_piece;

	oph::vec2 tilt_angle;
};

/**
* @param ulonglong Number of points
* @param int Number of color chennel
* @param Real* Geometry of point clouds
* @param Real* Color data of point clouds
* @param Real* Phase value of point clouds
* @param bool Selects whether to parse the phase data
*/
struct GEN_DLL OphPointCloudData {
	ulonglong n_points;
	int n_colors;
	Real *vertex;
	Real *color;
	Real *phase;
	bool isPhaseParse;

	OphPointCloudData() :vertex(nullptr), color(nullptr), phase(nullptr) { n_points = 0; n_colors = 0; isPhaseParse = 0; }
};

/**
* @param Real FIELD_LENS at config file
* @param Real NEAR_OF_DEPTH_MAP at config file
* @param Real FAR_OF_DEPTH_MAP at config file
* @param oph::uint the number of depth level.
* <pre>
* if FLAG_CHANGE_DEPTH_QUANTIZATION == 0
* num_of_depth = DEFAULT_DEPTH_QUANTIZATION
* else
* num_of_depth = NUMBER_OF_DEPTH_QUANTIZATION
* </pre> 
* @param std::vector<int> Used when only few specific depth levels are rendered, usually for test purpose
* @param bool if true, change the depth quantization from the default value.
* @param unsigned int default value of the depth quantization - 256
* @param unsigned int depth level of input depthmap.
* @param bool If true, random phase is imposed on each depth layer.
*/
struct GEN_DLL OphDepthMapConfig {
	Real				field_lens;

	Real				near_depthmap;
	Real				far_depthmap;

	oph::uint			num_of_depth;
										

	std::vector<int>	render_depth;

	bool				FLAG_CHANGE_DEPTH_QUANTIZATION;
	oph::uint			DEFAULT_DEPTH_QUANTIZATION;
	oph::uint			NUMBER_OF_DEPTH_QUANTIZATION;
	bool				RANDOM_PHASE;

	OphDepthMapConfig() :field_lens(0), near_depthmap(0), far_depthmap(0), num_of_depth(0) {}
};

/**
* @brief	Triangular mesh data structure
* @param	ulonglong	n_faces			The number of faces in object
* @param	int			color_channels
* @param	uint*		face_idx		Face indexes
* @param	Real*		vertex			Vertex array
* @param	Real*		color			Color array
*/
struct GEN_DLL OphMeshData {
	ulonglong n_faces = 0;
	int color_channels;
	uint* face_idx;
	Real* vertex;
	Real* color;
};

struct GEN_DLL OphWRPConfig {
	oph::vec3 scale;								///< Scaling factor of coordinate of point cloud

	int num_wrp;                                    ///< Number of wavefront recording plane(WRP)  
	Real wrp_location;                              ///< location distance of WRP
	Real propagation_distance;                      ///< distance of Hologram plane

};

#endif // !__ophGen_h
