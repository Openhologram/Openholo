
#include	"ophDepthMap.h"
#include    "sys.h"

/**
* @brief Initialize variables for the CPU implementation.
* @details Memory allocation for the CPU variables.
* @see initialize
*/
void ophDepthMap::initCPU()
{
	if (img_src)	delete[] img_src;
	img_src = new Real[context_.pixel_number[_X] * context_.pixel_number[_Y]];

	if (dmap_src) delete[] dmap_src;
	dmap_src = new Real[context_.pixel_number[_X] * context_.pixel_number[_Y]];

	if (alpha_map) delete[] alpha_map;
	alpha_map = new int[context_.pixel_number[_X] * context_.pixel_number[_Y]];

	if (depth_index) delete[] depth_index;
	depth_index = new Real[context_.pixel_number[_X] * context_.pixel_number[_Y]];

	if (dmap) delete[] dmap;
	dmap = new Real[context_.pixel_number[_X] * context_.pixel_number[_Y]];

	fftw_cleanup();
}

/**
* @brief Preprocess input image & depth map data for the CPU implementation.
* @details Prepare variables, img_src_, dmap_src_, alpha_map_, depth_index_.
* @param imgptr : input image data pointer
* @param dimgptr : input depth map data pointer
* @return true if input data are sucessfully prepared, flase otherwise.
* @see ReadImageDepth
*/
bool ophDepthMap::prepareInputdataCPU(uchar* imgptr, uchar* dimgptr)
{
	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];

	memset(img_src, 0, sizeof(Real)*pnx * pny);
	memset(dmap_src, 0, sizeof(Real)*pnx * pny);
	memset(alpha_map, 0, sizeof(int)*pnx * pny);
	memset(depth_index, 0, sizeof(Real)*pnx * pny);
	memset(dmap, 0, sizeof(Real)*pnx * pny);

	int k = 0;
#pragma omp parallel for private(k)
	for (k = 0; k < pnx*pny; k++)
	{
		img_src[k] = Real(imgptr[k]) / 255.0;
		dmap_src[k] = Real(dimgptr[k]) / 255.0;
		alpha_map[k] = (imgptr[k] > 0 ? 1 : 0);
		dmap[k] = (1 - dmap_src[k])*(dm_config_.far_depthmap - dm_config_.near_depthmap) + dm_config_.near_depthmap;

		if (dm_config_.FLAG_CHANGE_DEPTH_QUANTIZATION == 0)
			depth_index[k] = dm_config_.DEFAULT_DEPTH_QUANTIZATION - Real(dimgptr[k]);
	}
}

/**
* @brief Quantize depth map on the CPU, when the number of depth quantization is not the default value (i.e. FLAG_CHANGE_DEPTH_QUANTIZATION == 1 ).
* @details Calculate the value of 'depth_index_'.
* @see GetDepthValues
*/
void ophDepthMap::changeDepthQuanCPU()
{
	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];

	Real temp_depth, d1, d2;
	int tdepth;

	for (uint dtr = 0; dtr < dm_config_.num_of_depth; dtr++)
	{
		temp_depth = dlevel[dtr];
		d1 = temp_depth - dstep / 2.0;
		d2 = temp_depth + dstep / 2.0;

		int p;
#pragma omp parallel for private(p)
		for (p = 0; p < pnx * pny; p++)
		{
			if (dtr < dm_config_.num_of_depth - 1)
				tdepth = (dmap[p] >= d1 ? 1 : 0) * (dmap[p] < d2 ? 1 : 0);
			else
				tdepth = (dmap[p] >= d1 ? 1 : 0) * (dmap[p] <= d2 ? 1 : 0);

			depth_index[p] += tdepth*(dtr + 1);
		}
	}

	//writeIntensity_gray8_bmp("test.bmp", pnx, pny, depth_index_);
}

/**
* @brief Main method for generating a hologram on the CPU.
* @details For each depth level, 
*   1. find each depth plane of the input image.
*   2. apply carrier phase delay.
*   3. propagate it to the hologram plan.
*   4. accumulate the result of each propagation.
* .
* The final result is accumulated in the variable 'U_complex_'.
* @param frame : the frame number of the image.
* @see Calc_Holo_by_Depth, Propagation_AngularSpectrum_CPU
*/
void ophDepthMap::calcHoloCPU()
{
	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];

	memset(holo_gen, 0.0, sizeof(Complex<Real>)*pnx*pny);
	size_t depth_sz = dm_config_.render_depth.size();

	Complex<Real> *in = nullptr, *out = nullptr;
	fft2(ivec2(pnx, pny), in, OPH_FORWARD, OPH_ESTIMATE);

	int p = 0;
#pragma omp parallel for private(p)
	for (p = 0; p < depth_sz; ++p)
	{
		int dtr = dm_config_.render_depth[p];
		Real temp_depth = dlevel_transform[dtr - 1];

		Complex<Real>* u_o = (Complex<Real>*)malloc(sizeof(Complex<Real>)*pnx*pny);
		memset(u_o, 0.0, sizeof(Complex<Real>)*pnx*pny);

		Real sum = 0.0;
		for (int i = 0; i < pnx * pny; i++)
		{
			u_o[i]._Val[_RE] = img_src[i] * alpha_map[i] * (depth_index[i] == dtr ? 1.0 : 0.0);
			sum += u_o[i]._Val[_RE];
		}

		if (sum > 0.0)
		{
			LOG("Depth: %d of %d, z = %f mm\n", dtr, dm_config_.num_of_depth, -temp_depth * 1000);

			Complex<Real> rand_phase_val;
			getRandPhaseValue(rand_phase_val, dm_config_.RANDOM_PHASE);

			Complex<Real> carrier_phase_delay(0, context_.k* temp_depth);
			carrier_phase_delay.exp();

			for (int i = 0; i < pnx * pny; i++)
				u_o[i] = u_o[i] * rand_phase_val * carrier_phase_delay;

			//if (dm_params_.Propagation_Method_ == 0) {
			Openholo::fftwShift(u_o, u_o, pnx, pny, OPH_FORWARD, false);
			propagationAngularSpectrum(u_o, -temp_depth);
			//}
		}
		else
			LOG("Depth: %d of %d : Nothing here\n", dtr, dm_config_.num_of_depth);

		free(u_o);
	}
}