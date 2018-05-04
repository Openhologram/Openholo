
#include	"ophDepthMap.h"
#include    "sys.h"
#include	<cuda_runtime.h>
#include	<cufft.h>

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}} 

cufftDoubleComplex *u_o_gpu_;
cufftDoubleComplex *u_complex_gpu_;
cufftDoubleComplex *k_temp_d_;

cudaStream_t	stream_;
cudaEvent_t		start, stop;

extern "C"
{
	/**
	* \defgroup gpu_model GPU Modules
	* @{
	*/
	/**
	* @brief Convert data from the spatial domain to the frequency domain using 2D FFT on GPU.
	* @details call CUDA Kernel - fftShift and CUFFT Library.
	* @param stream : CUDA Stream
	* @param nx : the number of column of the input data
	* @param ny : the number of row of the input data
	* @param in_field : input complex data variable
	* @param output_field : output complex data variable
	* @param direction : If direction == -1, forward FFT, if type == 1, inverse FFT.
	* @param bNomarlized : If bNomarlized == true, normalize the result after FFT.
	* @see propagation_AngularSpectrum_GPU, encoding_GPU
	*/
	void cudaFFT(CUstream_st* stream, int nx, int ny, cufftDoubleComplex* in_filed, cufftDoubleComplex* output_field, int direction,  bool bNormailized = false);

	/**
	* @brief Crop input data according to x, y coordinates on GPU.
	* @details call CUDA Kernel - cropFringe. 
	* @param stream : CUDA Stream
	* @param nx : the number of column of the input data
	* @param ny : the number of row of the input data
	* @param in_field : input complex data variable
	* @param output_field : output complex data variable
	* @param cropx1 : the start x-coordinate to crop.
	* @param cropx2 : the end x-coordinate to crop.
	* @param cropy1 : the start y-coordinate to crop.
	* @param cropy2 : the end y-coordinate to crop.
	* @see encoding_GPU
	*/
	void cudaCropFringe(CUstream_st* stream, int nx, int ny, cufftDoubleComplex* in_field, cufftDoubleComplex* out_field, int cropx1, int cropx2, int cropy1, int cropy2);

	/**
	* @brief Find each depth plane of the input image and apply carrier phase delay to it on GPU.
	* @details call CUDA Kernel - depth_sources_kernel.
	* @param stream : CUDA Stream
	* @param nx : the number of column of the input data
	* @param ny : the number of row of the input data
	* @param u_o_gpu_ : output variable
	* @param img_src_gpu_ : input image data 
	* @param dimg_src_gpu_ : input depth map data
	* @param depth_index_gpu_ : input quantized depth map data
	* @param dtr : current working depth level
	* @param rand_phase_val_a : the real part of the random phase value
	* @param rand_phase_val_b : the imaginary part of the random phase value
	* @param carrier_phase_delay_a : the real part of the carrier phase delay
	* @param carrier_phase_delay_b : the imaginary part of the carrier phase delay
	* @param flag_change_depth_quan : if true, change the depth quantization from the default value.
	* @param default_depth_quan : default value of the depth quantization - 256
	* @see calc_Holo_GPU
	*/
	void cudaDepthHoloKernel(CUstream_st* stream, int pnx, int pny, cufftDoubleComplex* u_o_gpu_, unsigned char* img_src_gpu_, unsigned char* dimg_src_gpu_, real* depth_index_gpu_,
		int dtr, real rand_phase_val_a, real rand_phase_val_b, real carrier_phase_delay_a, real carrier_phase_delay_b, int flag_change_depth_quan, unsigned int default_depth_quan);

	/**
	* @brief Angular spectrum propagation method for GPU implementation.
	* @details The propagation results of all depth levels are accumulated in the variable 'u_complex_gpu_'.
	* @param stream : CUDA Stream
	* @param pnx : the number of column of the input data
	* @param pny : the number of row of the input data
	* @param input_d : input data
	* @param u_complex : output data
	* @param ppx : pixel pitch of x-axis
	* @param ppy : pixel pitch of y-axis
	* @param ssx : pnx * ppx
	* @param ssy : pny * ppy
	* @param lambda : wavelength
	* @param params_k :  2 * PI / lambda
	* @param propagation_dist : the distance from the object to the hologram plane
	* @see propagation_AngularSpectrum_GPU
	*/
	void cudaPropagation_AngularSpKernel(CUstream_st* stream_, int pnx, int pny, cufftDoubleComplex* input_d, cufftDoubleComplex* u_complex,
		real ppx, real ppy, real ssx, real ssy, real lambda, real params_k, real propagation_dist);

	/**
	* @brief Encode the CGH according to a signal location parameter on the GPU.
	* @details The variable, ((real*)p_hologram) has the final result.
	* @param stream : CUDA Stream
	* @param pnx : the number of column of the input data
	* @param pny : the number of row of the input data
	* @param in_field : input data
	* @param out_field : output data 
	* @param sig_locationx : signal location of x-axis, left or right half
	* @param sig_locationy : signal location of y-axis, upper or lower half
	* @param ssx : pnx * ppx
	* @param ssy : pny * ppy
	* @param ppx : pixel pitch of x-axis
	* @param ppy : pixel pitch of y-axis
	* @param PI : Pi
	* @see encoding_GPU
	*/
	void cudaGetFringe(CUstream_st* stream, int pnx, int pny, cufftDoubleComplex* in_field, cufftDoubleComplex* out_field, int sig_locationx, int sig_locationy,
		real ssx, real ssy, real ppx, real ppy, real PI);

	/**
	* @brief Quantize depth map on the GPU, only when the number of depth quantization is not the default value (i.e. FLAG_CHANGE_DEPTH_QUANTIZATION == 1 ).
	* @details Calculate the value of 'depth_index_gpu_'.
	* @param stream : CUDA Stream
	* @param pnx : the number of column of the input data
	* @param pny : the number of row of the input data
	* @param depth_index_gpu : output variable
	* @param dimg_src_gpu : input depth map data
	* @param dtr : the current working depth level
	* @param d1 : the starting physical point of each depth level
	* @param d2 : the ending physical point of each depth level
	* @param params_num_of_depth : the number of depth level
	* @param params_far_depthmap : NEAR_OF_DEPTH_MAP at config file
	* @param params_near_depthmap : FAR_OF_DEPTH_MAP at config file
	* @see change_depth_quan_GPU
	*/
	void cudaChangeDepthQuanKernel(CUstream_st* stream_, int pnx, int pny, real* depth_index_gpu, unsigned char* dimg_src_gpu,
		int dtr, real d1, real d2, real params_num_of_depth, real params_far_depthmap, real params_near_depthmap);

	/**@}*/

}

/**
* @brief Initialize variables for the GPU implementation.
* @details Memory allocation for the GPU variables.
* @see initialize
*/
void ophDepthMap::init_GPU()
{
	const int nx = context_.pixel_number[0];
	const int ny = context_.pixel_number[1];
	const int N = nx * ny;

	if (!stream_)
		cudaStreamCreate(&stream_);
	
	if (img_src_gpu_)	cudaFree(img_src_gpu_);
	HANDLE_ERROR(cudaMalloc((void**)&img_src_gpu_, sizeof(uchar1)*N));

	if (dimg_src_gpu_)	cudaFree(dimg_src_gpu_);
	HANDLE_ERROR(cudaMalloc((void**)&dimg_src_gpu_, sizeof(uchar1)*N));

	if (depth_index_gpu_) cudaFree(depth_index_gpu_);
	if (dm_params_.FLAG_CHANGE_DEPTH_QUANTIZATION == 1)
		HANDLE_ERROR(cudaMalloc((void**)&depth_index_gpu_, sizeof(real)*N));
	
	if (u_o_gpu_)	cudaFree(u_o_gpu_);
	if (u_complex_gpu_)	cudaFree(u_complex_gpu_);

	HANDLE_ERROR(cudaMalloc((void**)&u_o_gpu_, sizeof(cufftDoubleComplex)*N));
	HANDLE_ERROR(cudaMalloc((void**)&u_complex_gpu_, sizeof(cufftDoubleComplex)*N));

	if (k_temp_d_)	cudaFree(k_temp_d_);
	HANDLE_ERROR(cudaMalloc((void**)&k_temp_d_, sizeof(cufftDoubleComplex)*N));

}

/**
* @brief Copy input image & depth map data into a GPU.
* @param imgptr : input image data pointer
* @param dimgptr : input depth map data pointer
* @return true if input data are sucessfully copied on GPU, flase otherwise.
* @see readImageDepth
*/
bool ophDepthMap::prepare_inputdata_GPU(uchar* imgptr, uchar* dimgptr)
{
	const int nx = context_.pixel_number[0];
	const int ny = context_.pixel_number[1];
	const int N = nx * ny;
	
	HANDLE_ERROR(cudaMemcpyAsync(img_src_gpu_, imgptr, sizeof(uchar1)*N, cudaMemcpyHostToDevice), stream_);
	HANDLE_ERROR(cudaMemcpyAsync(dimg_src_gpu_, dimgptr, sizeof(uchar1)*N, cudaMemcpyHostToDevice), stream_);
	
	return true;
}

/**
* @brief Quantize depth map on the GPU, when the number of depth quantization is not the default value (i.e. FLAG_CHANGE_DEPTH_QUANTIZATION == 1 ).
* @details Calculate the value of 'depth_index_gpu_'.
* @see getDepthValues
*/
void ophDepthMap::change_depth_quan_GPU()
{
	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];

	real temp_depth, d1, d2;

	HANDLE_ERROR(cudaMemsetAsync(depth_index_gpu_, 0, sizeof(real)*pnx*pny, stream_));

	for (oph::uint dtr = 0; dtr < dm_config_.num_of_depth; dtr++)
	{
		temp_depth = dlevel_[dtr];
		d1 = temp_depth - dstep_ / 2.0;
		d2 = temp_depth + dstep_ / 2.0;

		cudaChangeDepthQuanKernel(stream_, pnx, pny, depth_index_gpu_, dimg_src_gpu_, 
			dtr, d1, d2, dm_config_.num_of_depth, dm_config_.far_depthmap, dm_config_.near_depthmap);
	}

}

/**
* @brief Main method for generating a hologram on the GPU.
* @details For each depth level,
*   1. find each depth plane of the input image.
*   2. apply carrier phase delay.
*   3. propagate it to the hologram plan.
*   4. accumulate the result of each propagation.
* .
* It uses CUDA kernels, cudaDepthHoloKernel & cudaPropagation_AngularSpKernel.<br>
* The final result is accumulated in the variable 'u_complex_gpu_'.
* @param frame : the frame number of the image.
* @see calc_Holo_by_Depth, propagation_AngularSpectrum_GPU
*/
void ophDepthMap::calc_Holo_GPU(int frame)
{
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if (!stream_)
		cudaStreamCreate(&stream_);

	cudaEventRecord(start, stream_);

	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];
	int N = pnx*pny;

	HANDLE_ERROR(cudaMemsetAsync(u_complex_gpu_, 0, sizeof(cufftDoubleComplex)*N, stream_));
	size_t depth_sz = dm_config_.render_depth.size();

	for (int p = 0; p < depth_sz; ++p)
	{
		oph::Complex<real> rand_phase_val;
		get_rand_phase_value(rand_phase_val);

		int dtr = dm_config_.render_depth[p];
		real temp_depth = dlevel_transform_[dtr - 1];
		oph::Complex<real> carrier_phase_delay(0, context_.k* temp_depth);
		exponent_complex(&carrier_phase_delay);

		HANDLE_ERROR(cudaMemsetAsync(u_o_gpu_, 0, sizeof(cufftDoubleComplex)*N, stream_));

		cudaDepthHoloKernel(stream_, pnx, pny, u_o_gpu_, img_src_gpu_, dimg_src_gpu_, depth_index_gpu_, 
			dtr, rand_phase_val.re, rand_phase_val.im, carrier_phase_delay.re, carrier_phase_delay.im, dm_params_.FLAG_CHANGE_DEPTH_QUANTIZATION, dm_params_.DEFAULT_DEPTH_QUANTIZATION);

		if (dm_params_.Propagation_Method_ == 0)
		{
			HANDLE_ERROR(cudaMemsetAsync(k_temp_d_, 0, sizeof(cufftDoubleComplex)*N, stream_));
			cudaFFT(stream_, pnx, pny, u_o_gpu_, k_temp_d_, -1);

			propagation_AngularSpectrum_GPU(u_o_gpu_, -temp_depth);
		}

		LOG("Frame#: %d, Depth: %d of %d, z = %f mm\n", frame, dtr, dm_config_.num_of_depth, -temp_depth * 1000);

	}

	cudaEventRecord(stop, stream_);
	cudaEventSynchronize(stop);

	float elapsedTime = 0.0f;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	LOG("GPU Time= %f ms. \n", elapsedTime);
}

/**
* @brief Angular spectrum propagation method for GPU implementation.
* @details The propagation results of all depth levels are accumulated in the variable 'u_complex_gpu_'.
* @param input_u : each depth plane data.
* @param propagation_dist : the distance from the object to the hologram plane.
* @see calc_Holo_by_Depth, calc_Holo_GPU, cudaFFT
*/
void ophDepthMap::propagation_AngularSpectrum_GPU(cufftDoubleComplex* input_u, real propagation_dist)
{
	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];
	int N = pnx* pny;
	real ppx = context_.pixel_pitch[0];
	real ppy = context_.pixel_pitch[1];
	real ssx = context_.ss[0];
	real ssy = context_.ss[1];
	real lambda = context_.lambda;

	cudaPropagation_AngularSpKernel(stream_, pnx, pny, k_temp_d_, u_complex_gpu_, 
		ppx, ppy, ssx, ssy, lambda, context_.k, propagation_dist);
		
}

/**
* @brief Encode the CGH according to a signal location parameter on GPU.
* @details The variable, ((real*)p_hologram) has the final result.
* @param cropx1 : the start x-coordinate to crop.
* @param cropx2 : the end x-coordinate to crop.
* @param cropy1 : the start y-coordinate to crop.
* @param cropy2 : the end y-coordinate to crop.
* @param sig_location : ivec2 type,
*  sig_location[0]: upper or lower half, sig_location[1]:left or right half.
* @see encodingSymmetrization, cudaCropFringe, cudaFFT, cudaGetFringe
*/
void ophDepthMap::encoding_GPU(int cropx1, int cropx2, int cropy1, int cropy2, ivec2 sig_location)
{
	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];
	real ppx = context_.pixel_pitch[0];
	real ppy = context_.pixel_pitch[1];
	real ssx = context_.ss[0];
	real ssy = context_.ss[1];

	HANDLE_ERROR(cudaMemsetAsync(k_temp_d_, 0, sizeof(cufftDoubleComplex)*pnx*pny, stream_));
	cudaCropFringe(stream_, pnx, pny, u_complex_gpu_, k_temp_d_, cropx1, cropx2, cropy1, cropy2);

	HANDLE_ERROR(cudaMemsetAsync(u_complex_gpu_, 0, sizeof(cufftDoubleComplex)*pnx*pny, stream_));
	cudaFFT(stream_, pnx, pny, k_temp_d_, u_complex_gpu_, 1, true);

	HANDLE_ERROR(cudaMemsetAsync(k_temp_d_, 0, sizeof(cufftDoubleComplex)*pnx*pny, stream_));
	cudaGetFringe(stream_, pnx, pny, u_complex_gpu_, k_temp_d_, sig_location[0], sig_location[1], ssx, ssy, ppx, ppy, PI);

	cufftDoubleComplex* sample_fd = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*pnx*pny);
	memset(sample_fd, 0.0, sizeof(cufftDoubleComplex)*pnx*pny);

	HANDLE_ERROR(cudaMemcpyAsync(sample_fd, k_temp_d_, sizeof(cufftDoubleComplex)*pnx*pny, cudaMemcpyDeviceToHost), stream_);
	memset(((real*)p_hologram), 0.0, sizeof(real)*pnx*pny);

	for (int i = 0; i < pnx*pny; ++i)
	{
		((real*)p_hologram)[i] = sample_fd[i].x;
	}

	free(sample_fd);
}