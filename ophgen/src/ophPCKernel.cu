#ifndef OphPCKernel_cu__
#define OphPCKernel_cu__

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "typedef.h"
#include "ophPointCloud_GPU.h"


__global__ void kernelCghPointCloud_cuda(Real* cuda_pc_data, Real* cuda_amp_data, const GpuConst* cuda_config, const int n_points_stream, Real* dst) {
	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	ulonglong tid_offset = blockDim.x * gridDim.x;
	ulonglong n_pixels = cuda_config->pn_X * cuda_config->pn_Y;

	for (tid; tid < n_pixels; tid += tid_offset) {
		int col = tid % cuda_config->pn_X;
		int row = tid / cuda_config->pn_X;

		for (int j = 0; j < n_points_stream; ++j) { //Create Fringe Pattern
			Real pcx = cuda_pc_data[3 * j + _X] * cuda_config->scale_X;
			Real pcy = cuda_pc_data[3 * j + _Y] * cuda_config->scale_Y;
			Real pcz = cuda_pc_data[3 * j + _Z] * cuda_config->scale_Z + cuda_config->offset_depth;

			Real SLM_y = cuda_config->half_ss_Y - ((Real)row + 0.5) * cuda_config->pp_Y;
			Real SLM_x = ((Real)col + 0.5) * cuda_config->pp_X - cuda_config->half_ss_X;

			Real r = sqrt((SLM_x - pcx)*(SLM_x - pcx) + (SLM_y - pcy)*(SLM_y - pcy) + pcz * pcz);
			Real phi = cuda_config->k*r - cuda_config->k*SLM_x*cuda_config->sin_thetaX - cuda_config->k*SLM_y*cuda_config->sin_thetaY;
			Real result = cuda_amp_data[j] * cos(phi);

			*(dst + col + row * cuda_config->pn_X) += result; //R-S Integral
		}
	}
	__syncthreads();
}


extern "C"
{
	void cudaGenCghPointCloud(const int &nBlocks, const int &nThreads, const int &n_pts_per_stream, Real* cuda_pc_data, Real* cuda_amp_data, Real* cuda_dst, const GpuConst* cuda_config)
	{
		kernelCghPointCloud_cuda << < nBlocks, nThreads >> > (cuda_pc_data, cuda_amp_data, cuda_config, n_pts_per_stream, cuda_dst);
	}
}

#endif // !OphPCKernel_cu__