#ifndef OphPCKernel_cu__
#define OphPCKernel_cu__

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "typedef.h"
#include "ophPointCloud_GPU.h"


__global__ void cudaKernel_diffractEncodedRS(Real* pc_data, Real* amp_data, const GpuConstERS* config, const int n_points_stream, Real* dst) {
	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	ulonglong tid_offset = blockDim.x * gridDim.x;
	ulonglong n_pixels = config->pn_X * config->pn_Y;

	for (tid; tid < n_pixels; tid += tid_offset) {
		int xxtr = tid % config->pn_X;
		int yytr = tid / config->pn_X;

		for (int j = 0; j < n_points_stream; ++j) { //Create Fringe Pattern
			Real pcx = pc_data[3 * j + _X] * config->scale_X;
			Real pcy = pc_data[3 * j + _Y] * config->scale_Y;
			Real pcz = pc_data[3 * j + _Z] * config->scale_Z + config->offset_depth;

			Real xxx = ((Real)xxtr + 0.5) * config->pp_X - config->half_ss_X;
			Real yyy = config->half_ss_Y - ((Real)yytr + 0.5) * config->pp_Y;

			Real r = sqrt((xxx - pcx) * (xxx - pcx) + (yyy - pcy) * (yyy - pcy) + (pcz * pcz));
			Real p = config->k * (r - xxx * config->sin_thetaX - yyy * config->sin_thetaY);
			Real res = amp_data[config->n_colors * j] * cos(p);

			*(dst + xxtr + yytr * config->pn_X) += res;
		}
	}
	__syncthreads();
}


__global__ void cudaKernel_diffractNotEncodedRS(Real* pc_data, Real* amp_data, const GpuConstNERS* config, const int n_points_stream, Real* dst_real, Real* dst_imag) {
	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	ulonglong tid_offset = blockDim.x * gridDim.x;
	ulonglong n_pixels = config->pn_X * config->pn_Y;

	for (tid; tid < n_pixels; tid += tid_offset) {
		int xxtr = tid % config->pn_X;
		int yytr = tid / config->pn_X;

		for (int j = 0; j < n_points_stream; ++j) { //Create Fringe Pattern
			Real amplitude = amp_data[config->n_colors * j];
			Real pcx = pc_data[3 * j + _X] * config->scale_X;
			Real pcy = pc_data[3 * j + _Y] * config->scale_Y;
			Real pcz = pc_data[3 * j + _Z] * config->scale_Z + config->offset_depth;

			//boundary test
			Real _xbound[2] = {
				pcx + abs(config->tx / sqrt(1 - (config->tx * config->tx)) * pcz),
				pcx - abs(config->tx / sqrt(1 - (config->tx * config->tx)) * pcz)
			};

			Real _ybound[2] = {
				pcy + abs(config->ty / sqrt(1 - (config->ty * config->ty)) * pcz),
				pcy - abs(config->ty / sqrt(1 - (config->ty * config->ty)) * pcz)
			};

			Real Xbound[2] = {
				floor((_xbound[0] + config->half_ss_X) / config->pp_X) + 1,
				floor((_xbound[1] + config->half_ss_X) / config->pp_X) + 1
			};

			Real Ybound[2] = {
				config->pn_Y - floor((_ybound[1] + config->half_ss_Y) / config->pp_Y),
				config->pn_Y - floor((_ybound[0] + config->half_ss_Y) / config->pp_Y)
			};

			if (Xbound[0] > config->pn_X)	Xbound[0] = config->pn_X;
			if (Xbound[1] < 0)				Xbound[1] = 0;
			if (Ybound[0] > config->pn_Y)	Ybound[0] = config->pn_Y;
			if (Ybound[1] < 0)				Ybound[1] = 0;
			//

			if (((xxtr >= Xbound[1]) && (xxtr < Xbound[0])) && ((yytr >= Ybound[1]) && (yytr < Ybound[0]))) {
				Real xxx = -config->half_ss_X + (xxtr - 1) * config->pp_X;
				Real yyy = -config->half_ss_Y + (config->pn_Y - yytr) * config->pp_Y;

				Real r = sqrt((xxx - pcx) * (xxx - pcx) + (yyy - pcy) * (yyy - pcy) + (pcz * pcz));

				Real range_x[2] = {
					pcx + abs(config->tx / sqrt(1 - (config->tx * config->tx)) * sqrt((yyy - pcy) * (yyy - pcy) + (pcz * pcz))),
					pcx - abs(config->tx / sqrt(1 - (config->tx * config->tx)) * sqrt((yyy - pcy) * (yyy - pcy) + (pcz * pcz)))
				};

				Real range_y[2] = {
					pcy + abs(config->ty / sqrt(1 - (config->ty * config->ty)) * sqrt((xxx - pcx) * (xxx - pcx) + (pcz * pcz))),
					pcy - abs(config->ty / sqrt(1 - (config->ty * config->ty)) * sqrt((xxx - pcx) * (xxx - pcx) + (pcz * pcz)))
				};

				if (((xxx < range_x[0]) && (xxx > range_x[1])) && ((yyy < range_y[0]) && (yyy > range_y[1]))) {
					Real kr = config->k * r;
					Real res_real = (amplitude * pcz * sin(kr)) / (config->lambda * r * r);
					Real res_imag = (-amplitude * pcz * cos(kr)) / (config->lambda * r * r);

					*(dst_real + xxtr + yytr * config->pn_X) += res_real;
					*(dst_imag + xxtr + yytr * config->pn_X) += res_imag;
				}
			}
		}
	}
	__syncthreads();
}


__global__ void cudaKernel_diffractNotEncodedFrsn(Real* pc_data, Real* amp_data, const GpuConstNEFR* config, const int n_points_stream, Real* dst_real, Real* dst_imag) {
	ulonglong tid = blockIdx.x * blockDim.x + threadIdx.x;
	ulonglong tid_offset = blockDim.x * gridDim.x;
	ulonglong n_pixels = config->pn_X * config->pn_Y;

	for (tid; tid < n_pixels; tid += tid_offset) {
		int xxtr = tid % config->pn_X;
		int yytr = tid / config->pn_X;

		for (int j = 0; j < n_points_stream; ++j) { //Create Fringe Pattern
			Real amplitude = amp_data[config->n_colors * j];
			Real pcx = pc_data[3 * j + _X] * config->scale_X;
			Real pcy = pc_data[3 * j + _Y] * config->scale_Y;
			Real pcz = pc_data[3 * j + _Z] * config->scale_Z + config->offset_depth;

			//boundary test
			Real _xbound[2] = {
				pcx + abs(config->lambda * pcz / (2 * config->pp_X)),
				pcx - abs(config->lambda * pcz / (2 * config->pp_X))
			};

			Real _ybound[2] = {
				pcy + abs(config->lambda * pcz / (2 * config->pp_Y)),
				pcy - abs(config->lambda * pcz / (2 * config->pp_Y))
			};

			Real Xbound[2] = {
				floor((_xbound[0] + config->half_ss_X) / config->pp_X) + 1,
				floor((_xbound[1] + config->half_ss_X) / config->pp_X) + 1
			};

			Real Ybound[2] = {
				config->pn_Y - floor((_ybound[1] + config->half_ss_Y) / config->pp_Y),
				config->pn_Y - floor((_ybound[0] + config->half_ss_Y) / config->pp_Y)
			};

			if (Xbound[0] > config->pn_X)	Xbound[0] = config->pn_X;
			if (Xbound[1] < 0)				Xbound[1] = 0;
			if (Ybound[0] > config->pn_Y)	Ybound[0] = config->pn_Y;
			if (Ybound[1] < 0)				Ybound[1] = 0;
			//

			if (((xxtr >= Xbound[1]) && (xxtr < Xbound[0])) && ((yytr >= Ybound[1]) && (yytr < Ybound[0]))) {
				Real xxx = (-config->half_ss_X + (xxtr - 1) * config->pp_X) - pcx;
				Real yyy = (-config->half_ss_Y + (config->pn_Y - yytr) * config->pp_Y) - pcy;
				Real p = config->k * (xxx * xxx + yyy * yyy + 2 * pcz * pcz) / (2 * pcz);

				Real res_real = amplitude * sin(p) / (config->lambda * pcz);
				Real res_imag = amplitude * (-cos(p)) / (config->lambda * pcz);

				*(dst_real + xxtr + yytr * config->pn_X) += res_real;
				*(dst_imag + xxtr + yytr * config->pn_X) += res_imag;
			}
		}
	}
	__syncthreads();
}


extern "C"
{
	void cudaGenCghPointCloud_EncodedRS(
		const int &nBlocks, const int &nThreads, const int &n_pts_per_stream,
		Real* cuda_pc_data, Real* cuda_amp_data,
		Real* cuda_dst,
		const GpuConstERS* cuda_config)
	{
		cudaKernel_diffractEncodedRS << < nBlocks, nThreads >> > (cuda_pc_data, cuda_amp_data, cuda_config, n_pts_per_stream, cuda_dst);
	}

	void cudaGenCghPointCloud_NotEncodedRS(
		const int &nBlocks, const int &nThreads, const int &n_pts_per_stream,
		Real* cuda_pc_data, Real* cuda_amp_data,
		Real* cuda_dst_real, Real* cuda_dst_imag,
		const GpuConstNERS* cuda_config)
	{
		cudaKernel_diffractNotEncodedRS << < nBlocks, nThreads >> > (cuda_pc_data, cuda_amp_data, cuda_config, n_pts_per_stream, cuda_dst_real, cuda_dst_imag);
	}

	void cudaGenCghPointCloud_NotEncodedFrsn(
		const int &nBlocks, const int &nThreads, const int &n_pts_per_stream,
		Real* cuda_pc_data, Real* cuda_amp_data,
		Real* cuda_dst_real, Real* cuda_dst_imag,
		const GpuConstNEFR* cuda_config)
	{
		cudaKernel_diffractNotEncodedFrsn << < nBlocks, nThreads >> > (cuda_pc_data, cuda_amp_data, cuda_config, n_pts_per_stream, cuda_dst_real, cuda_dst_imag);
	}
}

#endif // !OphPCKernel_cu__