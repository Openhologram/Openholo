#ifndef __ophPointCloud_GPU_h
#define __ophPointCloud_GPU_h

#include "ophPointCloud.h"

/* CUDA Library Include */
#include <cuda_runtime.h>

#define __CUDA_INTERNAL_COMPILATION__ //for CUDA Math Module
#include <math_constants.h>
#include <math_functions.h> //Single Precision Floating
#include <math_functions_dbl_ptx3.h> //Double Precision Floating
#include <vector_functions.h> //Vector Processing Function
#undef __CUDA_INTERNAL_COMPILATION__


static void handleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR(err) (handleError(err, __FILE__, __LINE__))


#define HANDLE_NULL(a) { \
	if (a == NULL) { \
		printf("Host memory failed in %s at line %d\n", __FILE__, __LINE__); \
		exit(EXIT_FAILURE); \
	} \
}


// for PointCloud only GPU
typedef struct KernelConst {
	int n_points;	///number of point cloud

	double scale_X;		/// Scaling factor of x coordinate of point cloud
	double scale_Y;		/// Scaling factor of y coordinate of point cloud
	double scale_Z;		/// Scaling factor of z coordinate of point cloud

	double offset_depth;	/// Offset value of point cloud in z direction

	int pn_X;		/// Number of pixel of SLM in x direction
	int pn_Y;		/// Number of pixel of SLM in y direction

	double sin_thetaX; ///sin(tiltAngleX)
	double sin_thetaY; ///sin(tiltAngleY)
	double k;		  ///Wave Number = (2 * PI) / lambda;

	double pp_X; /// Pixel pitch of SLM in x direction
	double pp_Y; /// Pixel pitch of SLM in y direction
	double half_ss_X; /// (pixel_x * nx) / 2
	double half_ss_Y; /// (pixel_y * ny) / 2 

	KernelConst(
		const int &n_points,		/// number of point cloud
		const vec3 &scale_factor,	/// Scaling factor of x, y, z coordinate of point cloud
		const Real &offset_depth,	/// Offset value of point cloud in z direction
		const ivec2 &pixel_number,	/// Number of pixel of SLM in x, y direction
		const vec2 &tilt_angle,		/// tilt_Angle_X, tilt_Angle_Y
		const Real &k,				/// Wave Number = (2 * PI) / lambda;
		const vec2 &pixel_pitch,	/// Pixel pitch of SLM in x, y direction
		const vec2 &ss)				/// (pixel_x * nx), (pixel_y * ny);
	{
		this->n_points = n_points;
		this->scale_X = scale_factor[_X];
		this->scale_Y = scale_factor[_Y];
		this->scale_Z = scale_factor[_Z];
		this->offset_depth = offset_depth;

		// Output Image Size
		this->pn_X = pixel_number[_X];
		this->pn_Y = pixel_number[_Y];

		// Tilt Angle
		this->sin_thetaX = sin(RADIAN(tilt_angle[_X]));
		this->sin_thetaY = sin(RADIAN(tilt_angle[_Y]));

		// Wave Number
		this->k = k;

		// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
		this->pp_X = pixel_pitch[_X];
		this->pp_Y = pixel_pitch[_Y];

		// Length (Width) of complex field at eyepiece plane (by simple magnification)
		this->half_ss_X = ss[_X] / 2.0;
		this->half_ss_Y = ss[_Y] / 2.0;
	}
} GpuConst;


extern "C"
{
	void cudaGenCghPointCloud(
		const int &nBlocks,
		const int &nThreads,
		const int &n_pts_per_stream,
		Real* cuda_pc_data,
		Real* cuda_amp_data,
		Real* cuda_dst,
		const GpuConst* cuda_config);
}


#endif