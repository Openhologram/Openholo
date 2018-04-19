/**
* @mainpage Openholo Generation Point Cloud : GPGPU Accelation using CUDA
* @brief
*/

#ifndef __OphPointCloud_h
#define __OphPointCloud_h

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
	ophPointCloud(void);
	/**
	* @overload
	*/
	ophPointCloud(const std::string InputModelFile, const std::string InputConfigFile);

private:
	/**
	* @brief Destructor
	*/
	virtual ~ophPointCloud(void);

public:
	/**
	* @brief Set the value of a variable isCPU_(true or false)
	* @details <pre>
	if isCPU_ == true
	CPU implementation
	else
	GPU implementation </pre>
	* @param isCPU : the value for specifying whether the hologram generation method is implemented on the CPU or GPU
	*/
	void setMode(bool isCPU);

	/**
	\defgroup PointCloud_Load
	* @{
	* @brief Import Point Cloud Data Base File : *.dat file.
	* This Function is included memory location of Input Point Clouds.
	*/
	/**
	* @param InputModelFile PointCloud(*.dat) input file path
	* @return number of Pointcloud (if it failed loading, it returned -1)
	*/
	int loadPointCloud(const std::string InputModelFile);

	/**
	\defgroup Import_Configfile
	* @{
	* @brief Import Specification Config File(*.config) file
	*/
	/**
	* @param InputConfigFile Specification Config(*.config) file path
	*/
	bool readConfig(const std::string InputConfigFile);

	/**
	\defgroup Set_Data
	* @{
	* @brief Directly Set Basic Data
	*/
	/**
	* @param VertexArray 3D Point Cloud Model Geometry Data (x0, y0, z0, x1, y1, z1 ...)
	* @param AmplitudeArray 3D Point Cloud Model Amplitude Data of Point-Based Light Wave
	* @param PhaseArray  3D Point Cloud Model Phase Data of Point-Based Light Wave
	*/
	void setPointCloudModel(const std::vector<float> &VertexArray, const std::vector<float> &AmplitudeArray, const std::vector<float> &PhaseArray);
	void getPointCloudModel(std::vector<float> &VertexArray, std::vector<float> &AmplitudeArray, std::vector<float> &PhaseArray);

	void getModelVertexArray(std::vector<float> &VertexArray);
	void getModelAmplitudeArray(std::vector<float> &AmplitudeArray);
	void getModelPhaseArray(std::vector<float> &PhaseArray);
	int getNumberOfPoints();

	uchar* getHologramBufferData();

	/**
	* @param Model 3D Point Cloud Model Data
	*/
	//void setPointCloudModel(const std::vector<PointCloud> &Model);

	//void getPointCloudModel(std::vector<PointCloud> &Model);

	/**
	* @param InputConfig Specification Config Data
	*/
	void setConfigParams(const oph::ConfigParams &InputConfig);

	oph::ConfigParams getConfigParams();
	/** @} */

	/**
	* @brief Generate a hologram, main funtion.
	* @return implement time (sec)
	*/
	double generateHologram();

	/**
	* @brief Save Pixel Buffer to Bitmap File format Image.
	* @param OutputFileName: filename to save
	*/
	void saveFileBmp(std::string OutputFileName);

	/**
	\defgroup Set_Data
	* @{
	* @brief Directly Set Config Specification Information without *.config file
	*/
	/**
	* @param scaleX Scaling factor of x coordinate of point cloud
	* @param scaleY Scaling factor of y coordinate of point cloud
	* @param scaleZ Scaling factor of z coordinate of point cloud
	*/
	void setScaleFactor(const float scaleX, const float scaleY, const float scaleZ);

	void getScaleFactor(float &scaleX, float &scaleY, float &scaleZ);

	/**
	* @param offsetDepth Offset value of point cloud in z direction
	*/
	void setOffsetDepth(const float offsetDepth);

	float getOffsetDepth();

	/**
	* @param pitchX Pixel pitch of SLM in x direction
	* @param pitchY Pixel pitch of SLM in y direction
	*/
	void setSamplingPitch(const float pitchX, const float pitchY);

	void getSamplingPitch(float &pitchX, float &pitchY);

	/**
	* @param n_x Number of pixel of SLM in x direction
	* @param n_y Number of pixel of SLM in y direction
	*/
	void setImageSize(const int n_x, const int n_y);

	void getImageSize(int &n_x, int &n_y);

	/**
	* @param lambda Wavelength of laser
	*/
	void setWaveLength(const float lambda);

	float getWaveLength();

	/**
	* @param tiltAngleX Tilt angle in x direction for spatial filtering
	* @param tiltAngleY Tilt angle in y direction for spatial filtering
	*/
	void setTiltAngle(const float tiltAngleX, const float tiltAngleY);

	void getTiltAngle(float &tiltAngleX, float &tiltAngleY);
	/** @}	*/

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
	double genCghPointCloud(const std::vector<float> &VertexArray, const std::vector<float> &AmplitudeArray, float *dst);

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
	double genCghPointCloud_cuda(const std::vector<float> &VertexArray, const std::vector<float> &AmplitudeArray, float *dst);

	/** @}	*/

	/**
	* @brief normalize calculated fringe pattern to 8bit grayscale value.
	* @param src: Input float type pointer
	* @param dst: Output char tpye pointer
	* @param nx: The number of pixels in X
	* @param ny: The number of pixels in Y
	*/
	void normalize(float *src, uchar *dst, const int nx, const int ny);
	virtual void ophFree(void);

	bool bIsCPU;
	int n_points;

	std::string InputSrcFile;
	std::string InputConfigFile;

	//std::vector<PointCloud> ModelData;
	std::vector<float> VertexArray;
	std::vector<float> AmplitudeArray;
	std::vector<float> PhaseArray;

	oph::ConfigParams ConfigParams;
	uchar *data_hologram;
};

extern "C"
{
	void cudaPointCloudKernel(const int block_x, const int block_y, const int thread_x, const int thread_y, float3 *PointCloud, float *amplitude, const GpuConst *Config, float *dst);
}

#endif // !__OphPointCloud_h