/**
* @mainpage Openholo Generation Point Cloud : GPGPU Accelation using CUDA
* @brief
*/
#ifndef __ophPointCloud_h
#define __ophPointCloud_h

#define _USE_MATH_DEFINES

#include "ophGen.h"

//Build Option : Multi Core Processing (OpenMP)
#ifdef _OPENMP
#include <omp.h>
#endif

/* CUDA Library Include */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#define __CUDA_INTERNAL_COMPILATION__ //for CUDA Math Module
#include <math_constants.h>
#include <math_functions.h> //Single Precision Floating
//#include <math_functions_dbl_ptx3.h> //Double Precision Floating
#include <vector_functions.h> //Vector Processing Function

#undef __CUDA_INTERNAL_COMPILATION__

#define THREAD_X 32
#define THREAD_Y 16

/* Bitmap File Definition*/
#define OPH_Bitsperpixel 8 //24 // 3byte=24 
#define OPH_Planes 1
#define OPH_Compression 0
#define OPH_Xpixelpermeter 0x130B //2835 , 72 DPI
#define OPH_Ypixelpermeter 0x130B //2835 , 72 DPI
#define OPH_Pixel 0xFF

using namespace oph;

class GEN_DLL ophPointCloud : public ophGen
{
public:
	/**
	* @brief Constructor
	* @details Initialize variables.
	*/
	explicit ophPointCloud(void);
	/**
	* @overload
	*/
	explicit ophPointCloud(const char*, const char* cfg_file);
protected:
	/**
	* @brief Destructor
	*/
	virtual ~ophPointCloud(void);

public:
	/** \ingroup getter/setter */
	inline void setScale(Real sx, Real sy, Real sz) { pc_config_.scale.v[0] = sx; pc_config_.scale.v[1] = sy; pc_config_.scale.v[2] = sz; }
	/** \ingroup getter/setter */
	inline void setOffsetDepth(Real offset_depth) { pc_config_.offset_depth = offset_depth; }
	/** \ingroup getter/setter */
	inline void setFilterShapeFlag(int8_t* fsf) { pc_config_.filter_shape_flag = fsf; }
	/** \ingroup getter/setter */
	inline void setFilterWidth(Real wx, Real wy) { pc_config_.filter_width.v[0] = wx; pc_config_.filter_width.v[1] = wy; }
	/** \ingroup getter/setter */
	inline void setFocalLength(Real lens_in, Real lens_out, Real lens_eye_piece) { pc_config_.focal_length_lens_in = lens_in; pc_config_.focal_length_lens_out = lens_out; pc_config_.focal_length_lens_eye_piece = lens_eye_piece; }
	/** \ingroup getter/setter */
	inline void setTiltAngle(Real ax, Real ay) { pc_config_.tilt_angle.v[0] = ax; pc_config_.tilt_angle.v[1] = ay; }

	/** \ingroup getter/setter */
	inline void getScale(vec3& scale) { scale = pc_config_.scale; }
	/** \ingroup getter/setter */
	inline Real getOffsetDepth(void) { return pc_config_.offset_depth; }
	/** \ingroup getter/setter */
	inline int8_t* getFilterShapeFlag(void) { return pc_config_.filter_shape_flag; }
	/** \ingroup getter/setter */
	inline void getFilterWidth(vec2& filterwidth) { filterwidth = pc_config_.filter_width; }
	/** \ingroup getter/setter */
	inline void getFocalLength(Real* lens_in, Real* lens_out, Real* lens_eye_piece) {
		if (lens_in != nullptr) *lens_in = pc_config_.focal_length_lens_in;
		if (lens_out != nullptr) *lens_out = pc_config_.focal_length_lens_out;
		if (lens_eye_piece != nullptr) *lens_eye_piece = pc_config_.focal_length_lens_eye_piece;
	}
	/** \ingroup getter/setter */
	inline void getTiltAngle(vec2& tiltangle) { tiltangle = pc_config_.tilt_angle; }
	/** \ingroup getter/setter */
	inline vec3* getLocationPC(void) { return pc_data_.location; }
	/** \ingroup getter/setter */
	inline vec3* getColorPC(void) { return pc_data_.color; }
	/** \ingroup getter/setter */
	inline Real* getAmplitudePC(void) { return pc_data_.amplitude; }
	/** \ingroup getter/setter */
	inline Real* getPhasePC(void) { return pc_data_.phase; }
	/** \ingroup getter/setter */
	inline void setPointCloudModel(vec3* location, vec3 *color, Real *amplitude, Real *phase) {
		pc_data_.location = location;
		pc_data_.color = color;
		pc_data_.amplitude = amplitude;
		pc_data_.phase = phase;
	}
	/** \ingroup getter/setter */
	inline void getPointCloudModel(vec3 *location, vec3 *color, Real *amplitude, Real *phase) {
		getModelLocation(location);
		getModelColor(color);
		getModelAmplitude(amplitude);
		getModelPhase(phase);
	}

	/**
	\ingroup getter/setter
	* @{
	* @brief Directly Set Basic Data
	*/
	/**
	* @param Location 3D Point Cloud Geometry Data
	* @param Color 3D Point Cloud Color Data
	* @param Amplitude 3D Point Cloud Model Amplitude Data of Point-Based Light Wave
	* @param Phase 3D Point Cloud Model Phase Data of Point-Based Light Wave
	*/
	inline void getModelLocation(vec3 *location) { location = pc_data_.location; }
	inline void getModelColor(vec3 *color) { color = pc_data_.color; }
	inline void getModelAmplitude(Real *amplitude) { amplitude = pc_data_.amplitude; }
	inline void getModelPhase(Real *phase) { phase = pc_data_.phase; }
	inline int getNumberOfPoints() { return n_points; }

public:
	void initialize(void);
	/**
	* @brief Set the value of a variable isCPU_(true or false)
	* @details <pre>
	if isCPU_ == true
	CPU implementation
	else
	GPU implementation </pre>
	* @param isCPU : the value for specifying whether the hologram generation method is implemented on the CPU or GPU
	*/
	void setMode(bool IsCPU);

	/**
	\defgroup PointCloud_Load 
	* @brief override
	* @{
	* @brief Import Point Cloud Data Base File : *.dat file.
	* This Function is included memory location of Input Point Clouds.
	*/
	/**
	* @brief override
	* @param InputModelFile PointCloud(*.dat) input file path
	* @return number of Pointcloud (if it failed loading, it returned -1)
	*/
	virtual int loadPointCloud(const char* pc_file, uint flag = PC_XYZ | PC_PHASE | PC_AMPLITUDE);

	/**
	\defgroup Import_Configfile
	* @brief
	* @{
	* @brief Import Specification Config File(*.config) file
	*/
	/**
	* @param InputConfigFile Specification Config(*.config) file path
	*/
	virtual bool readConfig(const char* cfg_file);

	/**
	* @brief Generate a hologram, main funtion.
	* @return implement time (sec)
	*/
	double generateHologram();
	double diffract(void);
	void encode(void);

private:
	/**
	\defgroup PointCloud_Generation
	* @{
	* @brief Calculate Integral Fringe Pattern of 3D Point Cloud based Computer Generated Holography
	*/
	/**
	* @param VertexArray Input 3D PointCloud Model Coordinate Array Data
	* @param AmplitudeArray Input 3D PointCloud Model Amplitude Array Data
	* @param dst Output Fringe Pattern
	* @return implement time (sec)
	*/
	void genCghPointCloud(Real* dst);

	/**
	* @overload
	* @param Model Input 3D PointCloud Model Data
	* @param dst Output Fringe Pattern
	* @return implement time (sec)
	*/
	//double genCghPointCloud(const std::vector<PointCloud> &Model, float *dst);
	/** @}	*/

	/**
	\defgroup PointCloud_Generation
	* @{
	* @brief GPGPU Accelation of genCghPointCloud() using nVidia CUDA
	*/
	/**
	* @param VertexArray Input 3D PointCloud Model Coordinate Array Data
	* @param AmplitudeArray Input 3D PointCloud Model Amplitude Array Data
	* @param dst Output Fringe Pattern
	* @return implement time (sec)
	*/
	void genCghPointCloud_cuda(Real* dst);

	/** @}	*/

	/**
	* @brief normalize calculated fringe pattern to 8bit grayscale value.
	* @param src: Input float type pointer
	* @param dst: Output char tpye pointer
	* @param nx: The number of pixels in X
	* @param ny: The number of pixels in Y
	*/
	virtual void ophFree(void);

	bool IsCPU_;
	int n_points;

	OphPointCloudConfig pc_config_;
	OphPointCloudData	pc_data_;
};

extern "C"
{
	void cudaPointCloudKernel(const int block_x, const int block_y, const int thread_x, const int thread_y, float3 *PointCloud, Real *amplitude, const GpuConst *Config, Real *dst);
}

#endif // !__ophPointCloud_h