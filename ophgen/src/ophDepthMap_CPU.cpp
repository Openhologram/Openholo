
#include	"ophDepthMap.h"
#include    "sys.h"

fftw_plan fft_plan_fwd_;
fftw_plan fft_plan_bwd_;

/**
* @brief Initialize variables for the CPU implementation.
* @details Memory allocation for the CPU variables.
* @see initialize
*/
void ophDepthMap::init_CPU()
{
	if (img_src_)	delete[] img_src_;
	img_src_ = new real[context_.pixel_number[_X]*context_.pixel_number[_Y]];

	if (dmap_src_) delete[] dmap_src_;
	dmap_src_ = new real[context_.pixel_number[_X]*context_.pixel_number[_Y]];

	if (alpha_map_) delete[] alpha_map_;
	alpha_map_ = new int[context_.pixel_number[_X] * context_.pixel_number[_Y]];

	if (depth_index_) delete[] depth_index_;
	depth_index_ = new real[context_.pixel_number[_X] * context_.pixel_number[_Y]];

	if (dmap_) delete[] dmap_;
	dmap_ = new real[context_.pixel_number[_X] * context_.pixel_number[_Y]];

	fftw_cleanup();
}

/**
* @brief Preprocess input image & depth map data for the CPU implementation.
* @details Prepare variables, img_src_, dmap_src_, alpha_map_, depth_index_.
* @param imgptr : input image data pointer
* @param dimgptr : input depth map data pointer
* @return true if input data are sucessfully prepared, flase otherwise.
* @see readImageDepth
*/
bool ophDepthMap::prepare_inputdata_CPU(uchar* imgptr, uchar* dimgptr)
{
	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];

	memset(img_src_, 0, sizeof(real)*pnx * pny);
	memset(dmap_src_, 0, sizeof(real)*pnx * pny);
	memset(alpha_map_, 0, sizeof(int)*pnx * pny);
	memset(depth_index_, 0, sizeof(real)*pnx * pny);
	memset(dmap_, 0, sizeof(real)*pnx * pny);

	int k = 0;
#pragma omp parallel for private(k)
	for (k = 0; k < pnx*pny; k++)
	{
		img_src_[k] = real(imgptr[k]) / 255.0;
		dmap_src_[k] = real(dimgptr[k]) / 255.0;
		alpha_map_[k] = (imgptr[k] > 0 ? 1 : 0);
		dmap_[k] = (1 - dmap_src_[k])*(dm_config_.far_depthmap - dm_config_.near_depthmap) + dm_config_.near_depthmap;

		if (dm_params_.FLAG_CHANGE_DEPTH_QUANTIZATION == 0)
			depth_index_[k] = dm_params_.DEFAULT_DEPTH_QUANTIZATION - real(dimgptr[k]);
	}

	return true;
}

/**
* @brief Quantize depth map on the CPU, when the number of depth quantization is not the default value (i.e. FLAG_CHANGE_DEPTH_QUANTIZATION == 1 ).
* @details Calculate the value of 'depth_index_'.
* @see getDepthValues
*/
void ophDepthMap::change_depth_quan_CPU()
{
	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];

	real temp_depth, d1, d2;

	for (uint dtr = 0; dtr < dm_config_.num_of_depth; dtr++)
	{
		temp_depth = dlevel_[dtr];
		d1 = temp_depth - dstep_ / 2.0;
		d2 = temp_depth + dstep_ / 2.0;

		int p;
#pragma omp parallel for private(p)
		for (p = 0; p < pnx * pny; p++)
		{
			int tdepth;
			if (dtr < dm_config_.num_of_depth - 1)
				tdepth = (dmap_[p] >= d1 ? 1 : 0) * (dmap_[p] < d2 ? 1 : 0);
			else
				tdepth = (dmap_[p] >= d1 ? 1 : 0) * (dmap_[p] <= d2 ? 1 : 0);

			depth_index_[p] += tdepth*(dtr + 1);
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
void ophDepthMap::calc_Holo_CPU(void)
{
	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];

	int depth_sz = static_cast<int>(dm_config_.render_depth.size());

	fftw_complex *in = NULL, *out = NULL;
	fft_plan_fwd_ = fftw_plan_dft_2d(pny, pnx, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

	int p = 0;
#pragma omp parallel for private(p)
	for (p = 0; p < depth_sz; ++p)
	{
		int dtr = dm_config_.render_depth[p];
		real temp_depth = dlevel_transform_[dtr - 1];

		oph::Complex<real>* u_o = new oph::Complex<real>[pnx*pny];
		memset(u_o, 0.0, sizeof(oph::Complex<real>)*pnx*pny);

		real sum = 0.0;
		for (int i = 0; i < pnx * pny; i++)
		{
			u_o[i].re = img_src_[i] * alpha_map_[i] * (depth_index_[i] == dtr ? 1.0 : 0.0);
			sum += u_o[i].re;
		}

		if (sum > 0.0)
		{
			LOG("Depth: %d of %d, z = %f mm\n", dtr, dm_config_.num_of_depth, -temp_depth * 1000);

			oph::Complex<real> rand_phase_val;
			get_rand_phase_value(rand_phase_val, dm_params_.RANDOM_PHASE);

			oph::Complex<real> carrier_phase_delay(0, context_.k* temp_depth);
			carrier_phase_delay.exp();

			for (int i = 0; i < pnx * pny; i++)
				u_o[i] = u_o[i] * rand_phase_val * carrier_phase_delay;

			if (dm_params_.Propagation_Method_ == 0) {
				fftwShift(u_o, u_o, in, out, pnx, pny, 1, false);
				propagation_AngularSpectrum_CPU(u_o, -temp_depth);
			}
		}
		else
			LOG("Depth: %d of %d : Nothing here\n", dtr, dm_config_.num_of_depth);

		delete[] u_o;
	}

	fftw_destroy_plan(fft_plan_fwd_);	
	fftw_cleanup();
}

/**
* @brief Angular spectrum propagation method for CPU implementation.
* @details The propagation results of all depth levels are accumulated in the variable 'U_complex_'.
* @param input_u : each depth plane data.
* @param propagation_dist : the distance from the object to the hologram plane.
* @see calc_Holo_by_Depth, calc_Holo_CPU, fftwShift
*/
void ophDepthMap::propagation_AngularSpectrum_CPU(oph::Complex<real>* input_u, real propagation_dist)
{
	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];
	real ppx = context_.pixel_pitch[0];
	real ppy = context_.pixel_pitch[1];
	real ssx = context_.ss[0];
	real ssy = context_.ss[1];
	real lambda = context_.lambda;

	for (int i = 0; i < pnx * pny; i++)
	{
		real x = i % pnx;
		real y = i / pnx;

		real fxx = (-1.0 / (2.0*ppx)) + (1.0 / ssx) * x;
		real fyy = (1.0 / (2.0*ppy)) - (1.0 / ssy) - (1.0 / ssy) * y;

		real sval = sqrt(1 - (lambda*fxx)*(lambda*fxx) - (lambda*fyy)*(lambda*fyy));
		sval *= context_.k * propagation_dist;
		oph::Complex<real> kernel(0, sval);
		kernel.exp();

		int prop_mask = ((fxx * fxx + fyy * fyy) < (context_.k *context_.k)) ? 1 : 0;

		oph::Complex<real> u_frequency;
		if (prop_mask == 1)
			u_frequency = kernel * input_u[i];
		
		for (uint frm = 0; frm < dm_params_.NUMBER_OF_FRAME; frm++)
		{
			uint frame = pnx * pny * frm;
			holo_gen[i + frame] += u_frequency;
		}
	}
}


//=====reconstruction =======================================================================
/**
* @brief It is a testing function used for the reconstruction.
*/
void ophDepthMap::reconstructImage()
{
	dm_simuls_.Pixel_pitch_xy_[0] = context_.pixel_pitch[0] / dm_simuls_.test_pixel_number_scale_;
	dm_simuls_.Pixel_pitch_xy_[1] = context_.pixel_pitch[1] / dm_simuls_.test_pixel_number_scale_;

	dm_simuls_.SLM_pixel_number_xy_[0] = context_.pixel_number[0] / dm_simuls_.test_pixel_number_scale_;
	dm_simuls_.SLM_pixel_number_xy_[1] = context_.pixel_number[1] / dm_simuls_.test_pixel_number_scale_;

	dm_simuls_.f_field_ = dm_config_.field_lens;

	if (dm_simuls_.sim_final_)		free(dm_simuls_.sim_final_);
	dm_simuls_.sim_final_ = (real*)malloc(sizeof(real)*dm_simuls_.SLM_pixel_number_xy_[0] * dm_simuls_.SLM_pixel_number_xy_[1]);
	memset(dm_simuls_.sim_final_, 0.0, sizeof(real)*dm_simuls_.SLM_pixel_number_xy_[0] * dm_simuls_.SLM_pixel_number_xy_[1]);

	real vmax, vmin, vstep, vval;
	if (dm_simuls_.sim_step_num_ > 1)
	{
		vmax = max(dm_simuls_.sim_to_, dm_simuls_.sim_from_);
		vmin = min(dm_simuls_.sim_to_, dm_simuls_.sim_from_);
		vstep = (dm_simuls_.sim_to_ - dm_simuls_.sim_from_) / (dm_simuls_.sim_step_num_ - 1);

	}
	else if (dm_simuls_.sim_step_num_ == 1) {
		vval = (dm_simuls_.sim_to_ + dm_simuls_.sim_from_) / 2.0;
	}

	fftw_complex *in = NULL, *out = NULL;
	fft_plan_fwd_ = fftw_plan_dft_2d(dm_simuls_.SLM_pixel_number_xy_[1], dm_simuls_.SLM_pixel_number_xy_[0], in, out, FFTW_FORWARD, FFTW_ESTIMATE);

	if (dm_simuls_.hh_complex_)		free(dm_simuls_.hh_complex_);
	dm_simuls_.hh_complex_ = (oph::Complex<real>*)malloc(sizeof(oph::Complex<real>) *dm_simuls_.SLM_pixel_number_xy_[0] * dm_simuls_.SLM_pixel_number_xy_[1]);

	testPropagation2EyePupil(in, out);

	if (dm_simuls_.sim_step_num_ > 0)
	{
		for (int vtr = 1; vtr <= dm_simuls_.sim_step_num_; vtr++)
		{
			LOG("Calculating Frame %d of %d \n", vtr, dm_simuls_.sim_step_num_);
			if (dm_simuls_.sim_step_num_ > 1)
				vval = vmin + (vtr - 1)*vstep;
			if (dm_simuls_.sim_type_ == 0)
				dm_simuls_.focus_distance_ = vval;
			else
				dm_simuls_.eye_center_xy_[1] = vval;

			reconstruction(in, out);
			writeSimulationImage(vtr, vval);
		}

	}
	else {

		reconstruction(in, out);
		writeSimulationImage(0, 0);

	}

	fftw_destroy_plan(fft_plan_fwd_);
	fftw_cleanup();

	free(dm_simuls_.hh_complex_);
	free(dm_simuls_.sim_final_);
	dm_simuls_.sim_final_ = 0;
	dm_simuls_.hh_complex_ = 0;


}

/**
* @brief It is a testing function used for the reconstruction.
*/
void ophDepthMap::testPropagation2EyePupil(fftw_complex* in, fftw_complex* out)
{
	int pnx = dm_simuls_.SLM_pixel_number_xy_[0];
	int pny = dm_simuls_.SLM_pixel_number_xy_[1];
	real ppx = dm_simuls_.Pixel_pitch_xy_[0];
	real ppy = dm_simuls_.Pixel_pitch_xy_[1];
	real F_size_x = pnx*ppx;
	real F_size_y = pny*ppy;
	real lambda = context_.lambda;

	oph::Complex<real>* hh = (oph::Complex<real>*)malloc(sizeof(oph::Complex<real>) * pnx*pny);

	for (int k = 0; k < pnx*pny; k++)
	{
		hh[k].re = holo_encoded[k];
		hh[k].im = 0.0;
	}

	fftwShift(hh, hh, in, out, pnx, pny, 1, false);

	real pp_ex = lambda * dm_simuls_.f_field_ / F_size_x;
	real pp_ey = lambda * dm_simuls_.f_field_ / F_size_y;
	real E_size_x = pp_ex*pnx;
	real E_size_y = pp_ey*pny;

	int p;
#pragma omp parallel for private(p)
	for (p = 0; p < pnx * pny; p++)
	{
		real x = p % pnx;
		real y = p / pnx;

		real xe = (-E_size_x / 2.0) + (pp_ex * x);
		real ye = (E_size_y / 2.0 - pp_ey) - (pp_ey * y);

		real sval = M_PI / lambda / dm_simuls_.f_field_ * (xe*xe + ye*ye);
		oph::Complex<real> kernel(0, sval);
		kernel.exp();

		dm_simuls_.hh_complex_[p] = hh[p] * kernel;
	}

	free(hh);
}

/**
* @brief It is a testing function used for the reconstruction.
*/
void ophDepthMap::reconstruction(fftw_complex* in, fftw_complex* out)
{
	int pnx = dm_simuls_.SLM_pixel_number_xy_[0];
	int pny = dm_simuls_.SLM_pixel_number_xy_[1];
	real ppx = dm_simuls_.Pixel_pitch_xy_[0];
	real ppy = dm_simuls_.Pixel_pitch_xy_[1];
	real F_size_x = pnx*ppx;
	real F_size_y = pny*ppy;
	real lambda = context_.lambda;
	real pp_ex = lambda * dm_simuls_.f_field_ / F_size_x;
	real pp_ey = lambda * dm_simuls_.f_field_ / F_size_y;
	real E_size_x = pp_ex*pnx;
	real E_size_y = pp_ey*pny;

	oph::Complex<real>* hh_e_shift = (oph::Complex<real>*)malloc(sizeof(oph::Complex<real>) * pnx*pny);
	oph::Complex<real>* hh_e_ = (oph::Complex<real>*)malloc(sizeof(oph::Complex<real>) * pnx*pny);

	int eye_shift_by_pnx = (int)round(dm_simuls_.eye_center_xy_[0] / pp_ex);
	int eye_shift_by_pny = (int)round(dm_simuls_.eye_center_xy_[1] / pp_ey);
	oph::circshift(dm_simuls_.hh_complex_, hh_e_shift, -eye_shift_by_pnx, eye_shift_by_pny, pnx, pny);

	real f_eye = dm_simuls_.eye_length_*(dm_simuls_.f_field_ - dm_simuls_.focus_distance_) / (dm_simuls_.eye_length_ + (dm_simuls_.f_field_ - dm_simuls_.focus_distance_));
	real effective_f = f_eye*dm_simuls_.eye_length_ / (f_eye - dm_simuls_.eye_length_);

	int p;
#pragma omp parallel for private(p)
	for (p = 0; p < pnx * pny; p++)
	{
		real x = p % pnx;
		real y = p / pnx;

		real xe = (-E_size_x / 2.0) + (pp_ex * x);
		real ye = (E_size_y / 2.0 - pp_ey) - (pp_ey * y);

		oph::Complex<real> eye_propagation_kernel(0, M_PI / lambda / effective_f * (xe*xe + ye*ye));
		eye_propagation_kernel.exp();
		int eye_lens_anti_aliasing_mask = (sqrt(xe*xe + ye*ye) < abs(lambda*effective_f / (2.0 * max(pp_ex, pp_ey)))) ? 1 : 0;
		int eye_pupil_mask = (sqrt(xe*xe + ye*ye) < (dm_simuls_.eye_pupil_diameter_ / 2.0)) ? 1 : 0;

		hh_e_[p] = hh_e_shift[p] * eye_propagation_kernel * eye_lens_anti_aliasing_mask * eye_pupil_mask;

	}

	fftwShift(hh_e_, hh_e_, in, out, pnx, pny, 1, false);

	real pp_ret_x = lambda*dm_simuls_.eye_length_ / E_size_x;
	real pp_ret_y = lambda*dm_simuls_.eye_length_ / E_size_y;
	real Ret_size_x = pp_ret_x*pnx;
	real Ret_size_y = pp_ret_y*pny;

#pragma omp parallel for private(p)
	for (p = 0; p < pnx * pny; p++)
	{
		real x = p % pnx;
		real y = p / pnx;

		real xr = (-Ret_size_x / 2.0) + (pp_ret_x * x);
		real yr = (Ret_size_y / 2.0 - pp_ret_y) - (pp_ret_y * y);

		real sval = M_PI / lambda / dm_simuls_.eye_length_*(xr*xr + yr*yr);
		oph::Complex<real> kernel(0, sval);
		kernel.exp();

		dm_simuls_.sim_final_[p] = (hh_e_[p] * kernel).mag();

	}

	free(hh_e_shift);
	free(hh_e_);

}

/**
* @brief It is a testing function used for the reconstruction.
*/
void ophDepthMap::circshift(oph::Complex<real>* in, oph::Complex<real>* out, int shift_x, int shift_y, int nx, int ny)
{
	int ti, tj;
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			ti = (i + shift_x) % nx;
			if (ti < 0)
				ti = ti + nx;
			tj = (j + shift_y) % ny;
			if (tj < 0)
				tj = tj + ny;

			out[ti + tj * nx] = in[i + j * nx];
		}
	}
}
