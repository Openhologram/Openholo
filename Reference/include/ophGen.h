/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install, copy or use the software.
//
//
//                           License Agreement
//                For Open Source Digital Holographic Library
//
// Openholo library is free software;
// you can redistribute it and/or modify it under the terms of the BSD 2-Clause license.
//
// Copyright (C) 2017-2024, Korea Electronics Technology Institute. All rights reserved.
// E-mail : contact.openholo@gmail.com
// Web : http://www.openholo.org
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//  1. Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holder or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
// This software contains opensource software released under GNU Generic Public License,
// NVDIA Software License Agreement, or CUDA supplement to Software License Agreement.
// Check whether software you use contains licensed software.
//
//M*/

#ifndef __ophGen_h
#define __ophGen_h

#include "Openholo.h"

#ifdef GEN_EXPORT
#define GEN_DLL __declspec(dllexport)
#else
#define GEN_DLL __declspec(dllimport)
#endif

struct OphPointCloudConfig;
struct OphPointCloudData;
struct OphDepthMapConfig;
struct OphMeshData;
struct OphWRPConfig;

/**
* @ingroup gen
* @brief
* @author
*/
class GEN_DLL ophGen : public Openholo
{
public:

	enum ENCODE_FLAG {
		ENCODE_PHASE,
		ENCODE_AMPLITUDE,
		ENCODE_REAL,
		ENCODE_IMAGINEARY,
		ENCODE_SIMPLENI,
		ENCODE_BURCKHARDT,
		ENCODE_TWOPHASE,
		ENCODE_SSB,
		ENCODE_OFFSSB,
		ENCODE_SIMPLEBINARY,
		ENCODE_EDBINARY
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
	/**
	* @brief Function for getting the encoded complex field buffer
	* @return Type: <B>Real**</B>\n
	*				If the function succeeds, the return value is <B>encoded complex field data's pointer</B>.\n
	*				If the function fails, the return value is <B>nullptr</B>.
	*/
	inline Real** getEncodedBuffer(void) { return m_lpEncoded; }
	/**
	* @brief Function for getting the normalized(0~255) complex field buffer
	* @return Type: <B>uchar**</B>\n
	*				If the function succeeds, the return value is <B>normalized complex field data's pointer</B>.\n
	*				If the function fails, the return value is <B>nullptr</B>.
	*/
	inline uchar** getNormalizedBuffer(void) { return m_lpNormalized; }

	/**
	* @brief Initialize variables for Hologram complex field, encoded data, normalized data
	*/
	void initialize(void);

	/**
	* @brief load to point cloud data.
	* @param[in] pc_file Point cloud data file name
	* @param[out] pc_data_ Point cloud data - number of points, number of color, geometry of point cloud, color data, phase data
	* @return Type: <B>int</B>\n
	*				If the function succeeds, the return value is <B>Positive integer</B>.\n
	*				If the function fails, the return value is <B>Negative interger</B>.
	*/
	int loadPointCloud(const char* pc_file, OphPointCloudData *pc_data_);

	/**
	* @brief load to configuration file.
	* @param[in] fname config file name
	* @return Type: <B>bool</B>\n
	*				If the function succeeds, the return value is <B>true</B>.\n
	*				If the function fails, the return value is <B>false</B>.
	*/
	bool readConfig(const char* fname);

	/**
	* @brief Angular spectrum propagation method.
	* @param[in] input Each depth plane data.
	* @param[out] output complex data.
	* @param[in] distance the distance from the object to the hologram plane.
	* @param[in] k const value.
	* @param[in] lambda wave length.
	* @see calcHoloCPU, fft2
	*/
	void AngularSpectrumMethod(Complex<Real>* input, Complex<Real>* output, Real distance, Real k, Real lambda);

	/**
	@brief Convolution between Complex arrays which have same size
	@param[in] src1 convolution matrix 1
	@param[in] src2 convolution matrix 2
	@param[in] dst convolution destination matrix
	@param[in] size matrix size
	*/
	void conv_fft2(Complex<Real>* src1, Complex<Real>* src2, Complex<Real>* dst, ivec2 size);


	/**
	* @brief Normalization function to save as image file after hologram creation
	*/
	void normalize(void);
	void normalize(int ch);

	/**
	* @brief Function for saving image files
	* @param[in] fname Input file name.
	* @param[in] bitsperpixel bits per pixel.
	* @param[in] src Source data.
	* @param[in] px image width.
	* @param[in] py image height.
	* @return Type: <B>bool</B>\n
	*				If the function succeeds, the return value is <B>true</B>.\n
	*				If the function fails, the return value is <B>false</B>.
	*/
	bool save(const char* fname, uint8_t bitsperpixel = 8, uchar* src = nullptr, uint px = 0, uint py = 0);

	/**
	* @brief Function for loading image files
	* @param[in] fname File name.
	*/
	void* load(const char* fname);

	/**
	* @brief Function to read OHC file
	* @param[in] fname File name.
	* @return Type: <B>bool</B>\n
	*				If the function succeeds, the return value is <B>true</B>.\n
	*				If the function fails, the return value is <B>false</B>.
	*/
	virtual bool loadAsOhc(const char *fname);

protected:
	/**
	* @brief Called when saving multiple hologram data at a time
	* @param[in] fname Input file name.
	* @param[in] bitsperpixel bits per pixel.
	* @param[in] px image width.
	* @param[in] py image height.
	* @return Type: <B>bool</B>\n
	*				If the function succeeds, the return value is <B>true</B>.\n
	*				If the function fails, the return value is <B>false</B>.
	*/
	bool save(const char* fname, uint8_t bitsperpixel, uint px, uint py, uint fnum, uchar* args ...);

	/**
	* @brief	reset buffer
	* @details	buffer memory set '0'
	*/
	void resetBuffer();


public:

	void setEncodeMethod(unsigned int ENCODE_FLAG) { ENCODE_METHOD = ENCODE_FLAG; }

	/**
	* @brief	Encoding Functions
	* @details
	*	ENCODE_PHASE		:	Phase@n
	*	ENCODE_AMPLITUDE	:	Amplitude@n
	*	ENCODE_REAL			:	Real Part@n
	*	ENCODE_SIMPLENI		:	Simple numerical interference@n
	*	ENCODE_BURCKHARDT	:	Burckhardt encoding@n
	*	ENCODE_TWOPHASE		:	Two Phase Encoding@n
	* @param[in] ENCODE_FLAG encoding method.
	* @param[in] holo buffer to encode.
	* @overload
	*/
	//void encoding(unsigned int ENCODE_FLAG, Complex<Real>* holo = nullptr);
	void encoding(unsigned int ENCODE_FLAG);
	void encoding(unsigned int ENCODE_FLAG, Complex<Real>* holo, Real* encoded);
	//template<typename T>
	//void encoding(unsigned int ENCODE_FLAG, Complex<T>* holo = nullptr, T* encoded = nullptr);

	void encoding();
	/*
	* @brief	Encoding Functions
	* @details
	*	 ENCODE_SSB		:	Single Side Band Encoding@n
	*	 ENCODE_OFFSSB	:	Off-axis + Single Side Band Encoding@n
	* @param[in] ENCODE_FLAG encoding method.
	* @param[in] SSB_PASSBAND shift direction.
	* @param[in] holo buffer to encode.
	* @overload
	*/
	virtual void encoding(unsigned int ENCODE_FLAG, unsigned int SSB_PASSBAND, Complex<Real>* holo = nullptr, Real* encoded = nullptr);
	enum SSB_PASSBAND { SSB_LEFT, SSB_RIGHT, SSB_TOP, SSB_BOTTOM };

	/**
	* @brief	Binary Encoding Functions
	* @details
	*	ENCODE_PHASE		:	Phase@n
	*	ENCODE_AMPLITUDE	:	Amplitude@n
	*	ENCODE_REAL			:	Real Part@n
	*	ENCODE_SIMPLENI		:	Simple numerical interference@n
	*	ENCODE_BURCKHARDT	:	Burckhardt encoding@n
	*	ENCODE_TWOPHASE		:	Two Phase Encoding@n
	*	ENCODE_SIMPLEBINARY	:	Simple binary encoding@n
	*	ENCODE_EDBINARY		:	Error diffusion binary encoding
	* @param[in] BIN_ENCODE_FLAG binarization method.
	* @param[in] ENCODE_FLAG encoding method for binarization.
	* @param[in] threshold threshold for binarization.
	* @param[in] holo buffer to encode.
	* @overload
	*/
	void encoding(unsigned int BIN_ENCODE_FLAG, unsigned int ENCODE_FLAG, Real threshold, Complex<Real>* holo = nullptr, Real* encoded = nullptr);

public:

	bool Shift(Real x, Real y);
	/**
	* @brief Wave carry
	* @param[in] carryingAngleX	Wave carrying angle in horizontal direction
	* @param[in] carryingAngleY	Wave carrying angle in vertical direction
	* @param[in] distance Distance between the display and the object
	*/
	void waveCarry(Real carryingAngleX, Real carryingAngleY, Real distance);

	void waveCarry(Complex<Real>* src, Complex<Real>* dst, Real wavelength, int carryIdxX, int carryIdxY);
protected:
	/// Encoded hologram size, varied from encoding type.
	ivec2					m_vecEncodeSize;
	/// Encoding method flag.
	int						ENCODE_METHOD;
	/// Passband in Single-side band encoding.
	int						SSB_PASSBAND;
	/// Elapsed time of generate hologram.
	Real					m_elapsedTime;
	/// buffer to encoded.
	Real**					m_lpEncoded;
	/// buffer to normalized.
	uchar**					m_lpNormalized;

private:
	/// previous number of channel.
	int						m_nOldChannel;

protected:
	Real					m_dFieldLength;
	int						m_nStream;

	/// buffer to conv_fft2
	Complex<Real>*			src1FT;
	Complex<Real>*			src2FT;
	Complex<Real>*			dstFT;
public:
	void transVW(int nSize, Real *dst, Real *src);
	int getStream() { return m_nStream; }
	Real getFieldLength() { return m_dFieldLength; }
	/**
	* @brief Function for getting encode size
	* @return Type: <B>ivec2&</B>\n
	*				If the function succeeds, the return value is <B>encode size</B>.\n
	*				If the function fails, the return value is <B>nullptr</B>.
	*/
	ivec2& getEncodeSize(void) { return m_vecEncodeSize; }

	/**
	* @brief Function for setting buffer size
	* @param[in] resolution buffer size.
	*/
	void setResolution(ivec2 resolution);

	/**
	* @brief Function for getting elapsed time.
	* @return Type: <B>Real</B>\n
	*				If the function succeeds, the return value is <B>elapsed time</B>.
	*/
	Real getElapsedTime() { return m_elapsedTime; };

protected:
	/**
	* @brief	Encoding method.
	* @param[in] holo Source data.
	* @param[out] encoded Destination data.
	* @param[in] size size of encode.
	*/
	template <typename T>
	void RealPart(Complex<T>* holo, T* encoded, const int size);
	template <typename T>
	void ImaginearyPart(Complex<T>* holo, T* encoded, const int size);
	template <typename T>
	void Phase(Complex<T>* holo, T* encoded, const int size);
	template <typename T>
	void Amplitude(Complex<T>* holo, T* encoded, const int size);
	template <typename T>
	void TwoPhase(Complex<T>* holo, T* encoded, const int size);
	template <typename T>
	void Burckhardt(Complex<T>* holo, T* encoded, const int size);
	template <typename T>
	void SimpleNI(Complex<T>* holo, T* encoded, const int size);

	/**
	* @brief	Encoding method.
	* @param[in] holo Source data.
	* @param[out] encoded Destination data.
	* @param[in] holosize size of encode.
	* @param[in] passband direction of passband.
	*/
	void singleSideBand(Complex<Real>* holo, Real* encoded, const ivec2 holosize, int passband);

	/**
	* @brief	Frequency shift
	* @param[in] src Source data.
	* @param[out] dst Destination data.
	* @param[in] holosize
	* @param[in] shift_x X pixel value to shift
	* @param[in] shift_y Y pixel value to shift
	*/
	void freqShift(Complex<Real>* src, Complex<Real>* dst, const ivec2 holosize, int shift_x, int shift_y);


public:
	enum ED_WType { FLOYD_STEINBERG, SINGLE_RIGHT, SINGLE_DOWN, ITERATIVE_DESIGN };
	bool saveRefImages(char* fnameW, char* fnameWC, char* fnameAS, char* fnameSSB, char* fnameHP, char* fnameFreq, char* fnameReal, char* fnameBin, char* fnameReconBin, char* fnameReconErr, char* fnameReconNo);

protected:
	/// Binary Encoding - Error diffusion
	int ss;
	Complex<Real>* AS;
	Complex<Real>* normalized;
	Complex<Real>* fftTemp;
	Real* weight;
	Complex<Real>* weightC;
	Complex<Real>* freqW;
	Real* realEnc;
	Real* binary;
	Real* maskSSB;
	Real* maskHP;
	unsigned int m_mode;
	bool m_bRandomPhase;

	bool binaryErrorDiffusion(Complex<Real>* holo, Real* encoded, const ivec2 holosize, const int type, Real threshold);
	bool getWeightED(const ivec2 holosize, const int type, ivec2* pNw);
	bool shiftW(ivec2 holosize);
	void binarization(Complex<Real>* src, Real* dst, const int size, int ENCODE_FLAG, Real threshold);
	void CorrectionChromaticAberration(uchar* src, uchar* dst, int width, int height, int ch);

	//public:
	//bool carrierWaveMultiplexingEncoding(char* dirName, uint ENCODE_METHOD, Real cenFxIdx, Real cenFyIdx, Real stepFx, Real stepFy, int nFx, int nFy);
	//bool carrierWaveMultiplexingEncoding(char* dirName, uint ENCODE_METHOD, uint PASSBAND, Real cenFxIdx, Real cenFyIdx, Real stepFx, Real stepFy, int nFx, int nFy);

	//protected:



public:
	/**
	* @brief Fresnel propagation
	* @param[in] context OphContext structure
	* @param[in] in Input complex field
	* @param[out] out Output complex field
	* @param[in] distance Propagation distance
	*/
	void fresnelPropagation(OphConfig context, Complex<Real>* in, Complex<Real>* out, Real distance);
	/**
	* @brief Fresnel propagation
	* @param[in] in Input complex field
	* @param[out] out Output complex field
	* @param[in] distance Propagation distance
	* @param[in] channel index of channel
	*/
	void fresnelPropagation(Complex<Real>* in, Complex<Real>* out, Real distance, uint channel);
protected:
	/**
	* @brief Encode the CGH according to a signal location parameter.
	* @param[in] bCPU Select whether to operate with CPU or GPU
	* @param[in] sig_location Signal location@n
	*			sig_location[0]: upper or lower half, sig_location[1]:left or right half.
	* @see encodeSideBand_CPU, encodeSideBand_GPU
	*/
	void encodeSideBand(bool bCPU, ivec2 sig_location);
	/**
	* @brief Encode the CGH according to a signal location parameter on the CPU.
	* @details The CPU variable, (*complex_H) on CPU has the final result.
	* @param[in] cropx1 the start x-coordinate to crop
	* @param[in] cropx2 the end x-coordinate to crop
	* @param[in] cropy1 the start y-coordinate to crop
	* @param[in] cropy2 the end y-coordinate to crop
	* @param[in] sig_location Signal location@n
	*			sig_location[0]: upper or lower half, sig_location[1]:left or right half.
	* @see fft2
	*/
	void encodeSideBand_CPU(int cropx1, int cropx2, int cropy1, int cropy2, ivec2 sig_location);

	/**
	* @brief Encode the CGH according to a signal location parameter on the GPU.
	* @details The GPU variable, (*complex_H) on GPU has the final result.
	* @param[in] cropx1 the start x-coordinate to crop
	* @param[in] cropx2 the end x-coordinate to crop
	* @param[in] cropy1 the start y-coordinate to crop
	* @param[in] cropy2 the end y-coordinate to crop
	* @param[in] sig_location Signal location@n
	*			sig_location[0]: upper or lower half, sig_location[1]:left or right half.
	* @see cudaCropFringe, cudaFFT, cudaGetFringe
	*/
	void encodeSideBand_GPU(int cropx1, int cropx2, int cropy1, int cropy2, ivec2 sig_location);

	/**
	* @brief Calculate the shift phase value.
	* @param[in] shift_phase_val output variable.
	* @param[in] idx the current pixel position.
	* @param[in] sig_location signal location.
	* @see encodingSideBand_CPU
	*/
	void getShiftPhaseValue(Complex<Real>& shift_phase_val, int idx, ivec2 sig_location);

	/**
	* @brief Assign random phase value if random_phase == 1
	* @details If random_phase == 1, calculate a random phase value using random generator@n
	*  otherwise, random phase value is 1.
	* @param[out] rand_phase_val Input & Ouput value.
	* @param[in] rand_phase random or not.
	*/
	void GetRandomPhaseValue(Complex<Real>& rand_phase_val, bool rand_phase);

	void ScaleChange(Real *src, Real *dst, int nSize, Real scaleX, Real scaleY, Real scaleZ);
	void GetMaxMin(Real *src, int len, Real& max, Real& min);

public:

	void AngularSpectrum(Complex<Real> *src, Complex<Real> *dst, Real lambda, Real distance);
	void RS_Diffraction(vec3 src, Complex<Real> *dst, Real lambda, Real distance, Real amplitude);
	void RS_Diffraction(uchar *src, Complex<Real> *dst, Real lambda, Real distance);
	void Fresnel_Convolution(vec3 src, Complex<Real> *dst, Real lambda, Real distance, Real amplitude);
	void Fresnel_FFT(Complex<Real> *src, Complex<Real> *dst, Real lambda, Real waveRatio, Real distance);
	bool readImage(const char* fname, bool bRGB);
	void SetMode(unsigned int mode) { m_mode = mode; }
	unsigned int GetMode() { return m_mode; }
	void SetRandomPhase(bool bRandomPhase) { m_bRandomPhase = bRandomPhase; }
	bool GetRandomPhase() { return m_bRandomPhase; }

	uchar* imgRGB;
	uchar* imgDepth;
	int m_width;
	int m_height;
	int m_bpp;
protected:
	/**
	* @brief Pure virtual function for override in child classes
	*/
	virtual void ophFree(void);
};

/**
* @struct OphPointCloudConfig
* @brief Configuration for Point Cloud
*/
struct GEN_DLL OphPointCloudConfig {
	/// Scaling factor of coordinate of point cloud
	vec3 scale;
	/// Offset value of point cloud
	Real distance;
	/// Shape of spatial bandpass filter ("Circle" or "Rect" for now)
	int8_t* filter_shape_flag;
	/// Width of spatial bandpass filter
	vec2 filter_width;
	/// Focal length of input lens of Telecentric
	Real focal_length_lens_in;
	/// Focal length of output lens of Telecentric
	Real focal_length_lens_out;
	/// Focal length of eyepiece lens
	Real focal_length_lens_eye_piece;
	/// Tilt angle for spatial filtering
	vec2 tilt_angle;

	OphPointCloudConfig()
		: scale(0, 0, 0), distance(0), filter_shape_flag(0), focal_length_lens_in(0), focal_length_lens_out(0), focal_length_lens_eye_piece(0), tilt_angle(0, 0)
	{}
};

/**
* @struct OphPointCloudData
* @brief Data for Point Cloud.
*/
struct GEN_DLL OphPointCloudData {
	/// Number of points
	ulonglong n_points;
	/// Number of color channel
	int n_colors;
	/// Geometry of point clouds
	Real *vertex;
	/// Color data of point clouds
	Real *color;
	/// Phase value of point clouds
	Real *phase;
	/// Selects wheter to parse the phase data
	bool isPhaseParse;

	OphPointCloudData() :vertex(nullptr), color(nullptr), phase(nullptr) { n_points = 0; n_colors = 0; isPhaseParse = 0; }
};

/**
* @struct OphDepthMapConfig
* @brief Configuration for Depth Map
* @param Real fieldLength at config file
* @param Real NEAR_OF_DEPTH_MAP at config file
* @param Real FAR_OF_DEPTH_MAP at config file
* @param oph::uint the number of depth level.
* <pre>
* if change_depth_quantization == 0
* num_of_depth = default_depth_quantization
* else
* num_of_depth = num_of_depth_quantization
* </pre>
* @param std::vector<int> Used when only few specific depth levels are rendered, usually for test purpose
* @param bool if true, change the depth quantization from the default value.
* @param unsigned int default value of the depth quantization - 256
* @param unsigned int depth level of input depthmap.
* @param bool If true, random phase is imposed on each depth layer.
*/
struct GEN_DLL OphDepthMapConfig {
	/// fieldLength variable for viewing window.
	Real				fieldLength;
	/// near value of depth in object
	Real				near_depthmap;
	/// far value of depth in object
	Real				far_depthmap;
	/// The number of depth level.@n
	/// if change_depth_quantization == 0@n
	/// num_of_depth = default_depth_quantization@n
	/// else@n
	/// num_of_depth = num_of_depth_quantization
	uint				num_of_depth;
	/// Used when only few specific depth levels are rendered, usually for test purpose
	vector<int>			render_depth;
	/// if true, change the depth quantization from the default value.
	bool				change_depth_quantization;
	/// default value of the depth quantization - 256
	uint				default_depth_quantization;
	/// depth level of input depthmap.
	uint				num_of_depth_quantization;
	/// If true, random phase is imposed on each depth layer.
	bool				random_phase;

	OphDepthMapConfig() :fieldLength(0), near_depthmap(0), far_depthmap(0), num_of_depth(0) {}
};

/**
* @struct OphMeshData
* @brief Data for triangular mesh
*/
struct GEN_DLL OphMeshData {
	/// The number of faces in object
	ulonglong n_faces = 0;
	/// The number of color
	int color_channels;
	/// Face indexes
	uint* face_idx;
	/// Vertex array
	Real* vertex;
	/// Color array
	Real* color;
};

/**
* @struct OphWRPConfig
* @brief Configuration for WRP
*/
struct GEN_DLL OphWRPConfig {
	/// fieldLength variable for viewing window.
	Real fieldLength;
	/// Scaling factor of coordinate of point cloud
	vec3 scale;
	/// Number of wavefront recording plane(WRP) 
	int num_wrp;
	/// Location distance of WRP
	Real wrp_location;
	/// Distance of Hologram plane
	Real propagation_distance;

};

/**
* @struct OphIFTAConfig
*/
struct GEN_DLL OphIFTAConfig {
	/// near value of depth in object
	Real				near_depthmap;
	/// far value of depth in object
	Real				far_depthmap;
	/// num_of_depth = num_of_depth_quantization
	int					num_of_depth;

	int					num_of_iteration;

	OphIFTAConfig() :near_depthmap(0), far_depthmap(0), num_of_depth(0), num_of_iteration(0) {}
};
#endif // !__ophGen_h

