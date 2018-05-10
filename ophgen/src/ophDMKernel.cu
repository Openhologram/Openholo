#ifndef OphPCKernel_cu__
#define OphPCKernel_cu__

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cufft.h>

static const int kBlockThreads = 512;

__global__ void fftShift(int N, int nx, int ny, cufftDoubleComplex* input, cufftDoubleComplex* output, bool bNormailzed)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	double normalF = 1.0;
	if (bNormailzed == true)
		normalF = nx * ny;

	while (tid < N)
	{
		int i = tid % nx;
		int j = tid / nx;

		int ti = i - nx / 2; if (ti<0) ti += nx;
		int tj = j - ny / 2; if (tj<0) tj += ny;

		int oindex = tj * nx + ti;


		output[tid].x = input[oindex].x / normalF;
		output[tid].y = input[oindex].y / normalF;

		tid += blockDim.x * gridDim.x;
	}
}

__device__  void exponent_complex(cuDoubleComplex* val)
{
	double exp_val = exp(val->x);
	double cos_v;
	double sin_v;
	sincos(val->y, &sin_v, &cos_v);

	val->x = exp_val * cos_v;
	val->y = exp_val * sin_v;

}


__global__ void depth_sources_kernel(cufftDoubleComplex* u_o_gpu, unsigned char* img_src_gpu, unsigned char* dimg_src_gpu, double* depth_index_gpu,
	int dtr, double rand_phase_val_a, double rand_phase_val_b, double carrier_phase_delay_a, double carrier_phase_delay_b, int pnx, int pny,
	int FLAG_CHANGE_DEPTH_QUANTIZATION, unsigned int DEFAULT_DEPTH_QUANTIZATION)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < pnx*pny) {

		double img = ((double)img_src_gpu[tid]) / 255.0;
		double depth_idx;
		if (FLAG_CHANGE_DEPTH_QUANTIZATION == 1)
			depth_idx = depth_index_gpu[tid];
		else
			depth_idx = (double)DEFAULT_DEPTH_QUANTIZATION - (double)dimg_src_gpu[tid];

		double alpha_map = ((double)img_src_gpu[tid] > 0.0 ? 1.0 : 0.0);

		u_o_gpu[tid].x = img * alpha_map * (depth_idx == (double)dtr ? 1.0 : 0.0);

		cuDoubleComplex tmp1 = cuCmul(make_cuDoubleComplex(rand_phase_val_a, rand_phase_val_b), make_cuDoubleComplex(carrier_phase_delay_a, carrier_phase_delay_b));
		u_o_gpu[tid] = cuCmul(u_o_gpu[tid], tmp1);
	}
}


__global__ void propagation_angularsp_kernel(cufftDoubleComplex* input_d, cufftDoubleComplex* u_complex, int pnx, int pny,
	double ppx, double ppy, double ssx, double ssy, double lambda, double params_k, double propagation_dist)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < pnx*pny) {

		int x = tid % pnx;
		int y = tid / pnx;

		double fxx = (-1.0 / (2.0*ppx)) + (1.0 / ssx) * (double)x;
		double fyy = (1.0 / (2.0*ppy)) - (1.0 / ssy) - (1.0 / ssy) * (double)y;

		double sval = sqrt(1 - (lambda*fxx)*(lambda*fxx) - (lambda*fyy)*(lambda*fyy));
		sval *= params_k * propagation_dist;

		cuDoubleComplex kernel = make_cuDoubleComplex(0, sval);
		exponent_complex(&kernel);

		int prop_mask = ((fxx * fxx + fyy * fyy) < (params_k *params_k)) ? 1 : 0;

		cuDoubleComplex u_frequency = make_cuDoubleComplex(0, 0);
		if (prop_mask == 1)
			u_frequency = cuCmul(kernel, input_d[tid]);

		u_complex[tid] = cuCadd(u_complex[tid], u_frequency);
	}
}


__global__ void cropFringe(int nx, int ny, cufftDoubleComplex* in_filed, cufftDoubleComplex* out_filed, int cropx1, int cropx2, int cropy1, int cropy2)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < nx*ny)
	{
		int x = tid % nx;
		int y = tid / nx;

		if (x >= cropx1 && x <= cropx2 && y >= cropy1 && y <= cropy2)
			out_filed[tid] = in_filed[tid];
	}
}


__global__ void getFringe(int nx, int ny, cufftDoubleComplex* in_filed, cufftDoubleComplex* out_filed, int sig_locationx, int sig_locationy,
	double ssx, double ssy, double ppx, double ppy, double pi)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < nx*ny)
	{
		cuDoubleComplex shift_phase = make_cuDoubleComplex(1, 0);

		if (sig_locationy != 0)
		{
			int r = tid / nx;
			double yy = (ssy / 2.0) - (ppy)*(double)r - ppy;

			cuDoubleComplex val = make_cuDoubleComplex(0, 0);
			if (sig_locationy == 1)
				val.y = 2.0 * pi * (yy / (4.0 * ppy));
			else
				val.y = 2.0 * pi * (-yy / (4.0 * ppy));

			exponent_complex(&val);

			shift_phase = cuCmul(shift_phase, val);
		}

		if (sig_locationx != 0)
		{
			int c = tid % nx;
			double xx = (-ssx / 2.0) - (ppx)*(double)c - ppx;

			cuDoubleComplex val = make_cuDoubleComplex(0, 0);
			if (sig_locationx == -1)
				val.y = 2.0 * pi * (-xx / (4.0 * ppx));
			else
				val.y = 2.0 * pi * (xx / (4.0 * ppx));

			exponent_complex(&val);
			shift_phase = cuCmul(shift_phase, val);
		}

		out_filed[tid] = cuCmul(in_filed[tid], shift_phase);
	}

}

__global__ void change_depth_quan_kernel(double* depth_index_gpu, unsigned char* dimg_src_gpu, int pnx, int pny,
	int dtr, double d1, double d2, double num_depth, double far_depth, double near_depth)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < pnx*pny) {

		int tdepth;
		double dmap_src = double(dimg_src_gpu[tid]) / 255.0;
		double dmap = (1.0 - dmap_src)*(far_depth - near_depth) + near_depth;

		if (dtr < num_depth - 1)
			tdepth = (dmap >= d1 ? 1 : 0) * (dmap < d2 ? 1 : 0);
		else
			tdepth = (dmap >= d1 ? 1 : 0) * (dmap <= d2 ? 1 : 0);

		depth_index_gpu[tid] = depth_index_gpu[tid] + (double)(tdepth * (dtr + 1));
	}
}


extern "C"
void cudaFFT(CUstream_st* stream, int nx, int ny, cufftDoubleComplex* in_field, cufftDoubleComplex* output_field, int direction, bool bNormalized)
{
	unsigned int nblocks = (nx*ny + kBlockThreads - 1) / kBlockThreads;
	int N = nx * ny;
	fftShift << <nblocks, kBlockThreads, 0, stream >> >(N, nx, ny, in_field, output_field, false);

	cufftHandle plan;

	// fft
	if (cufftPlan2d(&plan, ny, nx, CUFFT_Z2Z) != CUFFT_SUCCESS)
	{
		//LOG("FAIL in creating cufft plan");
		return;
	};

	cufftResult result;

	if (direction == -1)
		result = cufftExecZ2Z(plan, output_field, in_field, CUFFT_FORWARD);
	else
		result = cufftExecZ2Z(plan, output_field, in_field, CUFFT_INVERSE);

	if (result != CUFFT_SUCCESS)
	{
		//LOG("------------------FAIL: execute cufft, code=%s", result);
		return;
	}

	if (cudaDeviceSynchronize() != cudaSuccess) {
		//LOG("Cuda error: Failed to synchronize\n");
		return;
	}

	fftShift << < nblocks, kBlockThreads, 0, stream >> >(N, nx, ny, in_field, output_field, bNormalized);

	cufftDestroy(plan);

}


extern "C"
void cudaDepthHoloKernel(CUstream_st* stream, int pnx, int pny, cufftDoubleComplex* u_o_gpu, unsigned char* img_src_gpu, unsigned char* dimg_src_gpu, double* depth_index_gpu,
	int dtr, double rand_phase_val_a, double rand_phase_val_b, double carrier_phase_delay_a, double carrier_phase_delay_b, int flag_change_depth_quan, unsigned int default_depth_quan)
{
	dim3 grid((pnx*pny + kBlockThreads - 1) / kBlockThreads, 1, 1);
	depth_sources_kernel << <grid, kBlockThreads, 0, stream >> >(u_o_gpu, img_src_gpu, dimg_src_gpu, depth_index_gpu,
		dtr, rand_phase_val_a, rand_phase_val_b, carrier_phase_delay_a, carrier_phase_delay_b, pnx, pny, flag_change_depth_quan, default_depth_quan);
}

extern "C"
void cudaPropagation_AngularSpKernel(CUstream_st* stream, int pnx, int pny, cufftDoubleComplex* input_d, cufftDoubleComplex* u_complex,
	double ppx, double ppy, double ssx, double ssy, double lambda, double params_k, double propagation_dist)
{
	dim3 grid((pnx*pny + kBlockThreads - 1) / kBlockThreads, 1, 1);
	propagation_angularsp_kernel << <grid, kBlockThreads, 0, stream >> >(input_d, u_complex, pnx, pny, ppx, ppy, ssx, ssy, lambda, params_k, propagation_dist);

}

extern "C"
void cudaCropFringe(CUstream_st* stream, int nx, int ny, cufftDoubleComplex* in_field, cufftDoubleComplex* out_field, int cropx1, int cropx2, int cropy1, int cropy2)
{
	unsigned int nblocks = (nx*ny + kBlockThreads - 1) / kBlockThreads;

	cropFringe << < nblocks, kBlockThreads, 0, stream >> > (nx, ny, in_field, out_field, cropx1, cropx2, cropy1, cropy2);

}

extern "C"
void cudaGetFringe(CUstream_st* stream, int pnx, int pny, cufftDoubleComplex* in_field, cufftDoubleComplex* out_field, int sig_locationx, int sig_locationy,
	double ssx, double ssy, double ppx, double ppy, double PI)
{
	unsigned int nblocks = (pnx*pny + kBlockThreads - 1) / kBlockThreads;

	getFringe << < nblocks, kBlockThreads, 0, stream >> > (pnx, pny, in_field, out_field, sig_locationx, sig_locationy, ssx, ssy, ppx, ppy, PI);

}

extern "C"
void cudaChangeDepthQuanKernel(CUstream_st* stream, int pnx, int pny, double* depth_index_gpu, unsigned char* dimg_src_gpu,
	int dtr, double d1, double d2, double params_num_of_depth, double params_far_depthmap, double params_near_depthmap)
{
	dim3 grid((pnx*pny + kBlockThreads - 1) / kBlockThreads, 1, 1);
	change_depth_quan_kernel << <grid, kBlockThreads, 0, stream >> > (depth_index_gpu, dimg_src_gpu, pnx, pny,
		dtr, d1, d2, params_num_of_depth, params_far_depthmap, params_near_depthmap);

}

#endif // !OphPCKernel_cu__