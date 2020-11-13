#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "AngularC_types.h"


struct constValue {
	double wavelength;
	double knumber;
	double depth;
	double minfrequency_eta;
	double minfrequency_xi;
	double eta_interval;
	double xi_interval;
	double xi_tilting;
	double eta_tilting;
	int w, h;
};


extern "C" void Angular_Spectrum_GPU(double w, double h, double wavelength, double knumber, double xi_interval, double eta_interval, double depth, const coder::array<creal_T, 2U>& fringe, coder::array<creal_T, 2U>& b_AngularC);