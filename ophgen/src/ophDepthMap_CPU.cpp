
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
	img_src = new Real[context_.pixel_number[_X]*context_.pixel_number[_Y]];

	if (dmap_src) delete[] dmap_src;
	dmap_src = new Real[context_.pixel_number[_X]*context_.pixel_number[_Y]];

	if (alpha_map) delete[] alpha_map;
	alpha_map = new int[context_.pixel_number[_X] * context_.pixel_number[_Y]];

	if (depth_index) delete[] depth_index;
	depth_index = new Real[context_.pixel_number[_X] * context_.pixel_number[_Y]];

	if (dmap) delete[] dmap;
	dmap = new Real[context_.pixel_number[_X] * context_.pixel_number[_Y]];
}

/**
* @brief Preprocess input image & depth map data for the CPU implementation.
* @details Prepare variables, img_src, dmap_src, alpha_map, depth_index.
* @param imgptr : input image data pointer
* @param dimgptr : input depth map data pointer
* @return true if input data are sucessfully prepared, flase otherwise.
* @see readImageDepth
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

		if (dm_params_.FLAG_CHANGE_DEPTH_QUANTIZATION == 0)
			depth_index[k] = dm_params_.DEFAULT_DEPTH_QUANTIZATION - Real(dimgptr[k]);
	}

	return true;
}

/**
* @brief Quantize depth map on the CPU, when the number of depth quantization is not the default value (i.e. FLAG_CHANGE_DEPTH_QUANTIZATION == 1 ).
* @details Calculate the value of 'depth_index'.
* @see getDepthValues
*/
void ophDepthMap::changeDepthQuanCPU()
{
	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];

	Real temp_depth, d1, d2;

	for (uint dtr = 0; dtr < dm_config_.num_of_depth; dtr++)
	{
		temp_depth = dlevel[dtr];
		d1 = temp_depth - dstep / 2.0;
		d2 = temp_depth + dstep / 2.0;

		int p;
#pragma omp parallel for private(p)
		for (p = 0; p < pnx * pny; p++)
		{
			int tdepth;
			if (dtr < dm_config_.num_of_depth - 1)
				tdepth = (dmap[p] >= d1 ? 1 : 0) * (dmap[p] < d2 ? 1 : 0);
			else
				tdepth = (dmap[p] >= d1 ? 1 : 0) * (dmap[p] <= d2 ? 1 : 0);

			depth_index[p] += tdepth*(dtr + 1);
		}
	}
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
* @see calc_Holo_by_Depth, propagation_AngularSpectrum_CPU
*/
void ophDepthMap::calcHoloCPU(void)
{
	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];

	int depth_sz = static_cast<int>(dm_config_.render_depth.size());

	oph::Complex<Real> *in = NULL, *out = NULL;

	fft2(oph::ivec2(pnx, pny), in, out, OPH_FORWARD, OPH_ESTIMATE);

	int p = 0;
#pragma omp parallel for private(p)
	for (p = 0; p < depth_sz; ++p)
	{
		int dtr = dm_config_.render_depth[p];
		Real temp_depth = dlevel_transform[dtr - 1];

		oph::Complex<Real>* u_o = new oph::Complex<Real>[pnx*pny];
		memset(u_o, 0.0, sizeof(oph::Complex<Real>)*pnx*pny);

		Real sum = 0.0;
		for (int i = 0; i < pnx * pny; i++)
		{
			u_o[i][_RE] = img_src[i] * alpha_map[i] * (depth_index[i] == dtr ? 1.0 : 0.0);
			sum += u_o[i][_RE];
		}

		if (sum > 0.0)
		{
			LOG("Depth: %d of %d, z = %f mm\n", dtr, dm_config_.num_of_depth, -temp_depth * 1000);

			oph::Complex<Real> rand_phase_val;
			getRandPhaseValue(rand_phase_val, dm_params_.RANDOM_PHASE);

			oph::Complex<Real> carrier_phase_delay(0, context_.k* temp_depth);
			carrier_phase_delay.exp();

			for (int i = 0; i < pnx * pny; i++)
				u_o[i] = u_o[i] * rand_phase_val * carrier_phase_delay;

			if (dm_params_.Propagation_Method_ == 0) {
				fftwShift(u_o, u_o, pnx, pny, 1, false);
				propagationAngularSpectrumCPU(u_o, -temp_depth);
			}
		}
		else
			LOG("Depth: %d of %d : Nothing here\n", dtr, dm_config_.num_of_depth);

		delete[] u_o;
	}
}

/**
* @brief Angular spectrum propagation method for CPU implementation.
* @details The propagation results of all depth levels are accumulated in the variable 'U_complex_'.
* @param input_u : each depth plane data.
* @param propagation_dist : the distance from the object to the hologram plane.
* @see calc_Holo_by_Depth, calc_Holo_CPU, fftwShift
*/
void ophDepthMap::propagationAngularSpectrumCPU(oph::Complex<Real>* input_u, Real propagation_dist)
{
	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];
	Real ppx = context_.pixel_pitch[0];
	Real ppy = context_.pixel_pitch[1];
	Real ssx = context_.ss[0];
	Real ssy = context_.ss[1];
	Real lambda = context_.lambda;

	for (int i = 0; i < pnx * pny; i++)
	{
		Real x = i % pnx;
		Real y = i / pnx;

		Real fxx = (-1.0 / (2.0*ppx)) + (1.0 / ssx) * x;
		Real fyy = (1.0 / (2.0*ppy)) - (1.0 / ssy) - (1.0 / ssy) * y;

		Real sval = sqrt(1 - (lambda*fxx)*(lambda*fxx) - (lambda*fyy)*(lambda*fyy));
		sval *= context_.k * propagation_dist;
		oph::Complex<Real> kernel(0, sval);
		kernel.exp();

		int prop_mask = ((fxx * fxx + fyy * fyy) < (context_.k *context_.k)) ? 1 : 0;

		oph::Complex<Real> u_frequency;
		if (prop_mask == 1)
			u_frequency = kernel * input_u[i];
		
		for (uint frm = 0; frm < dm_params_.NUMBER_OF_FRAME; frm++)
			holo_gen[i] += u_frequency;
	}
}


//=====reconstruction =======================================================================
/**
* @brief It is a testing function used for the reconstruction.
*/
//void ophDepthMap::reconstructImage()
//{
//	dm_simuls_.Pixel_pitch_xy_[0] = context_.pixel_pitch[0] / dm_simuls_.test_pixel_number_scale_;
//	dm_simuls_.Pixel_pitch_xy_[1] = context_.pixel_pitch[1] / dm_simuls_.test_pixel_number_scale_;
//
//	dm_simuls_.SLM_pixel_number_xy_[0] = context_.pixel_number[0] / dm_simuls_.test_pixel_number_scale_;
//	dm_simuls_.SLM_pixel_number_xy_[1] = context_.pixel_number[1] / dm_simuls_.test_pixel_number_scale_;
//
//	dm_simuls_.f_field_ = dm_config_.field_lens;
//
//	if (dm_simuls_.sim_final_)		free(dm_simuls_.sim_final_);
//	dm_simuls_.sim_final_ = (Real*)malloc(sizeof(Real)*dm_simuls_.SLM_pixel_number_xy_[0] * dm_simuls_.SLM_pixel_number_xy_[1]);
//	memset(dm_simuls_.sim_final_, 0.0, sizeof(Real)*dm_simuls_.SLM_pixel_number_xy_[0] * dm_simuls_.SLM_pixel_number_xy_[1]);
//
//	Real vmax, vmin, vstep, vval;
//	if (dm_simuls_.sim_step_num_ > 1)
//	{
//		vmax = max(dm_simuls_.sim_to_, dm_simuls_.sim_from_);
//		vmin = min(dm_simuls_.sim_to_, dm_simuls_.sim_from_);
//		vstep = (dm_simuls_.sim_to_ - dm_simuls_.sim_from_) / (dm_simuls_.sim_step_num_ - 1);
//
//	}
//	else if (dm_simuls_.sim_step_num_ == 1) {
//		vval = (dm_simuls_.sim_to_ + dm_simuls_.sim_from_) / 2.0;
//	}
//
//	fftw_complex *in = NULL, *out = NULL;
//	fft_plan_fwd_ = fftw_plan_dft_2d(dm_simuls_.SLM_pixel_number_xy_[1], dm_simuls_.SLM_pixel_number_xy_[0], in, out, FFTW_FORWARD, FFTW_ESTIMATE);
//
//	if (dm_simuls_.hh_complex_)		free(dm_simuls_.hh_complex_);
//	dm_simuls_.hh_complex_ = (oph::Complex<Real>*)malloc(sizeof(oph::Complex<Real>) *dm_simuls_.SLM_pixel_number_xy_[0] * dm_simuls_.SLM_pixel_number_xy_[1]);
//
//	testPropagation2EyePupil(in, out);
//
//	if (dm_simuls_.sim_step_num_ > 0)
//	{
//		for (int vtr = 1; vtr <= dm_simuls_.sim_step_num_; vtr++)
//		{
//			LOG("Calculating Frame %d of %d \n", vtr, dm_simuls_.sim_step_num_);
//			if (dm_simuls_.sim_step_num_ > 1)
//				vval = vmin + (vtr - 1)*vstep;
//			if (dm_simuls_.sim_type_ == 0)
//				dm_simuls_.focus_distance_ = vval;
//			else
//				dm_simuls_.eye_center_xy_[1] = vval;
//
//			reconstruction(in, out);
//			writeSimulationImage(vtr, vval);
//		}
//
//	}
//	else {
//
//		reconstruction(in, out);
//		writeSimulationImage(0, 0);
//
//	}
//
//	fftw_destroy_plan(fft_plan_fwd_);
//	fftw_cleanup();
//
//	free(dm_simuls_.hh_complex_);
//	free(dm_simuls_.sim_final_);
//	dm_simuls_.sim_final_ = 0;
//	dm_simuls_.hh_complex_ = 0;
//
//
//}

/**
* @brief It is a testing function used for the reconstruction.
*/
//void ophDepthMap::testPropagation2EyePupil(fftw_complex* in, fftw_complex* out)
//{
//	int pnx = dm_simuls_.SLM_pixel_number_xy_[0];
//	int pny = dm_simuls_.SLM_pixel_number_xy_[1];
//	Real ppx = dm_simuls_.Pixel_pitch_xy_[0];
//	Real ppy = dm_simuls_.Pixel_pitch_xy_[1];
//	Real F_size_x = pnx*ppx;
//	Real F_size_y = pny*ppy;
//	Real lambda = context_.lambda;
//
//	oph::Complex<Real>* hh = (oph::Complex<Real>*)malloc(sizeof(oph::Complex<Real>) * pnx*pny);
//
//	for (int k = 0; k < pnx*pny; k++)
//	{
//		hh[k][_RE] = holo_encoded[k];
//		hh[k][_IM] = 0.0;
//	}
//
//	fftwShift(hh, hh, in, out, pnx, pny, 1, false);
//
//	Real pp_ex = lambda * dm_simuls_.f_field_ / F_size_x;
//	Real pp_ey = lambda * dm_simuls_.f_field_ / F_size_y;
//	Real E_size_x = pp_ex*pnx;
//	Real E_size_y = pp_ey*pny;
//
//	int p;
//#pragma omp parallel for private(p)
//	for (p = 0; p < pnx * pny; p++)
//	{
//		Real x = p % pnx;
//		Real y = p / pnx;
//
//		Real xe = (-E_size_x / 2.0) + (pp_ex * x);
//		Real ye = (E_size_y / 2.0 - pp_ey) - (pp_ey * y);
//
//		Real sval = M_PI / lambda / dm_simuls_.f_field_ * (xe*xe + ye*ye);
//		oph::Complex<Real> kernel(0, sval);
//		kernel.exp();
//
//		dm_simuls_.hh_complex_[p] = hh[p] * kernel;
//	}
//
//	free(hh);
//}

/**
* @brief It is a testing function used for the reconstruction.
*/
//void ophDepthMap::reconstruction(fftw_complex* in, fftw_complex* out)
//{
//	int pnx = dm_simuls_.SLM_pixel_number_xy_[0];
//	int pny = dm_simuls_.SLM_pixel_number_xy_[1];
//	Real ppx = dm_simuls_.Pixel_pitch_xy_[0];
//	Real ppy = dm_simuls_.Pixel_pitch_xy_[1];
//	Real F_size_x = pnx*ppx;
//	Real F_size_y = pny*ppy;
//	Real lambda = context_.lambda;
//	Real pp_ex = lambda * dm_simuls_.f_field_ / F_size_x;
//	Real pp_ey = lambda * dm_simuls_.f_field_ / F_size_y;
//	Real E_size_x = pp_ex*pnx;
//	Real E_size_y = pp_ey*pny;
//
//	oph::Complex<Real>* hh_e_shift = (oph::Complex<Real>*)malloc(sizeof(oph::Complex<Real>) * pnx*pny);
//	oph::Complex<Real>* hh_e_ = (oph::Complex<Real>*)malloc(sizeof(oph::Complex<Real>) * pnx*pny);
//
//	int eye_shift_by_pnx = (int)round(dm_simuls_.eye_center_xy_[0] / pp_ex);
//	int eye_shift_by_pny = (int)round(dm_simuls_.eye_center_xy_[1] / pp_ey);
//	oph::circshift(dm_simuls_.hh_complex_, hh_e_shift, -eye_shift_by_pnx, eye_shift_by_pny, pnx, pny);
//
//	Real f_eye = dm_simuls_.eye_length_*(dm_simuls_.f_field_ - dm_simuls_.focus_distance_) / (dm_simuls_.eye_length_ + (dm_simuls_.f_field_ - dm_simuls_.focus_distance_));
//	Real effective_f = f_eye*dm_simuls_.eye_length_ / (f_eye - dm_simuls_.eye_length_);
//
//	int p;
//#pragma omp parallel for private(p)
//	for (p = 0; p < pnx * pny; p++)
//	{
//		Real x = p % pnx;
//		Real y = p / pnx;
//
//		Real xe = (-E_size_x / 2.0) + (pp_ex * x);
//		Real ye = (E_size_y / 2.0 - pp_ey) - (pp_ey * y);
//
//		oph::Complex<Real> eye_propagation_kernel(0, M_PI / lambda / effective_f * (xe*xe + ye*ye));
//		eye_propagation_kernel.exp();
//		int eye_lens_anti_aliasing_mask = (sqrt(xe*xe + ye*ye) < abs(lambda*effective_f / (2.0 * max(pp_ex, pp_ey)))) ? 1 : 0;
//		int eye_pupil_mask = (sqrt(xe*xe + ye*ye) < (dm_simuls_.eye_pupil_diameter_ / 2.0)) ? 1 : 0;
//
//		hh_e_[p] = hh_e_shift[p] * eye_propagation_kernel * eye_lens_anti_aliasing_mask * eye_pupil_mask;
//
//	}
//
//	fftwShift(hh_e_, hh_e_, in, out, pnx, pny, 1, false);
//
//	Real pp_ret_x = lambda*dm_simuls_.eye_length_ / E_size_x;
//	Real pp_ret_y = lambda*dm_simuls_.eye_length_ / E_size_y;
//	Real Ret_size_x = pp_ret_x*pnx;
//	Real Ret_size_y = pp_ret_y*pny;
//
//#pragma omp parallel for private(p)
//	for (p = 0; p < pnx * pny; p++)
//	{
//		Real x = p % pnx;
//		Real y = p / pnx;
//
//		Real xr = (-Ret_size_x / 2.0) + (pp_ret_x * x);
//		Real yr = (Ret_size_y / 2.0 - pp_ret_y) - (pp_ret_y * y);
//
//		Real sval = M_PI / lambda / dm_simuls_.eye_length_*(xr*xr + yr*yr);
//		oph::Complex<Real> kernel(0, sval);
//		kernel.exp();
//
//		dm_simuls_.sim_final_[p] = (hh_e_[p] * kernel).mag();
//
//	}
//
//	free(hh_e_shift);
//	free(hh_e_);
//
//}

/**
* @brief It is a testing function used for the reconstruction.
*/
//void ophDepthMap::circShift(oph::Complex<Real>* in, oph::Complex<Real>* out, int shift_x, int shift_y, int nx, int ny)
//{
//	int ti, tj;
//	for (int i = 0; i < nx; i++)
//	{
//		for (int j = 0; j < ny; j++)
//		{
//			ti = (i + shift_x) % nx;
//			if (ti < 0)
//				ti = ti + nx;
//			tj = (j + shift_y) % ny;
//			if (tj < 0)
//				tj = tj + ny;
//
//			out[ti + tj * nx] = in[i + j * nx];
//		}
//	}
//}
