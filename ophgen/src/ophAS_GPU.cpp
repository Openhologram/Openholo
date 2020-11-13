#include "ophAS_GPU.h"
#include "complex.h"
#include "sys.h"
#include "ophAS.h"

__global__ void transfer(constValue val, creal_T* a, creal_T* b)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < val.w && j < val.w)
	{
		double eta_id = val.wavelength * (((double(i) + 1.0) - (val.w / 2.0 + 1.0))*
			val.minfrequency_eta);
		double xi_id = val.wavelength * (((double(j) + 1.0) - (val.w / 2.0 + 1.0))*
			val.minfrequency_xi);
		double y_im = (val.knumber*val.depth)*sqrt((1.0 - eta_id*eta_id) - xi_id*xi_id);
		double y_re = cos(y_im);
		y_im = sin(y_im);
		b[i + val.w * j].re = a[i + val.w*j].re * y_re -
			a[i + val.w*j].im*y_im;
		b[i + val.w * j].im = a[i + val.w*j].re * y_im +
			a[i + val.w*j].im*y_re;
	}
}

__global__ void tilting(constValue val, creal_T* a, creal_T* b)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < val.w && j < val.w)
	{
		double f_eta = (((double(i) + 1.0) - 1.0) - val.w / 2.0)*
			val.eta_interval;
		double f_xi = val.knumber*((((double(j) + 1.0) - 1.0) - val.w / 2.0)*
			val.xi_interval*0.0 + f_eta*0.0);

		double y_re, y_im;

		if (!f_xi)
		{
			y_re = 1.0;
			y_im = 0.0;
		}
		else
		{
			y_re = nan("");
			y_im = nan("");
		}
		b[i + val.w*j].re = a[i + val.w*j].re*y_re - a[i + val.w*j].im*y_im;
		b[i + val.w*j].im = a[i + val.w*j].re*y_im + a[i + val.w*j].im*y_re;
	}
}

void Angular_Spectrum_GPU(double w, double h, double wavelength, double knumber, double xi_interval, double eta_interval, double depth, const coder::array<creal_T, 2U>& fringe, coder::array<creal_T, 2U>& b_AngularC)
{
	
	coder::array<creal_T, 2U> fringe_temp1;
	int loop_ub;
	int i;
	coder::array<creal_T, 2U> fringe_temp3;
	double eta;
	int xi_id;
	coder::array<creal_T, 2U> fringe_temp2;
	double y_im;
	double y_re;
	double minfrequency_eta;
	double minfrequency_xi;

	constValue val;
	dim3 blockSize = dim3(32, 32);
	dim3 gridSize = dim3((w + 32 - 1) / 32, (h + 32 - 1) / 32);


	val.wavelength = wavelength;
	val.depth = depth;
	val.knumber = knumber;

	val.w = w;
	val.h = h;
	val.xi_interval = xi_interval;
	val.eta_interval = eta_interval;
	fringe_temp1.set_size(fringe.size(0), fringe.size(1));
	loop_ub = fringe.size(0) * fringe.size(1);
	for (i = 0; i < loop_ub; i++) {
		fringe_temp1[i] = fringe[i];
	}

	fringe_temp3.set_size(fringe.size(0), fringe.size(1));
	loop_ub = fringe.size(0) * fringe.size(1);
	for (i = 0; i < loop_ub; i++) {
		fringe_temp3[i] = fringe[i];
	}

	{
		dim3 blockSize = dim3(32, 32);
		dim3 gridSize = dim3((w + 32 - 1) / 32, (h + 32 - 1) / 32);
		creal_T* fringe_temp1d;
		creal_T* fringe_d;
		cudaMalloc((void**)&fringe_temp1d, sizeof(creal_T)*w*h);
		cudaMalloc((void**)&fringe_d, sizeof(creal_T)*w*h);
		cudaMemcpy(fringe_d, fringe.data(), sizeof(creal_T)*w*h, cudaMemcpyHostToDevice);

		tilting << <gridSize, blockSize >> >(val, fringe_d, fringe_temp1d);
		cudaMemcpy(fringe_temp1.data(), fringe_temp1d, sizeof(creal_T)*w*h, cudaMemcpyDeviceToHost);

		cudaFree(fringe_temp1d);
		cudaFree(fringe_d);

	}

	/*i = static_cast<int>(w);
	for (loop_ub = 0; loop_ub < i; loop_ub++) {
	eta = (((static_cast<double>(loop_ub) + 1.0) - 1.0) - w / 2.0) *
	eta_interval;
	for (xi_id = 0; xi_id < i; xi_id++) {
	y_im = knumber * ((((static_cast<double>(xi_id) + 1.0) - 1.0) - w / 2.0) *
	xi_interval * 0.0 + eta * 0.0);
	if (-y_im == 0.0) {
	y_re = std::exp(y_im * 0.0);
	y_im = 0.0;
	}
	else {
	y_re = std::numeric_limits<double>::quiet_NaN();
	y_im = std::numeric_limits<double>::quiet_NaN();
	}

	fringe_temp1[loop_ub + fringe_temp1.size(0) * xi_id].re = fringe[loop_ub +
	fringe.size(0) * xi_id].re * y_re - fringe[loop_ub + fringe.size(0) *
	xi_id].im * y_im;
	fringe_temp1[loop_ub + fringe_temp1.size(0) * xi_id].im = fringe[loop_ub +
	fringe.size(0) * xi_id].re * y_im + fringe[loop_ub + fringe.size(0) *
	xi_id].im * y_re;
	}
	}*/

	ophAS* as;
	auto start = CUR_TIME;
	//  fourier transform of fringe pattern
	as->eml_fftshift(fringe_temp1, 1);
	as->eml_fftshift(fringe_temp1, 2);
	as->fft2_matlab(fringe_temp1, fringe_temp2);
	as->eml_fftshift(fringe_temp2, 1);
	as->eml_fftshift(fringe_temp2, 2);
	auto end = CUR_TIME;

	auto during = ((std::chrono::duration<Real>)(end - start)).count();

	{

		dim3 blockSize = dim3(32, 32);
		dim3 gridSize = dim3((w + 32 - 1) / 32, (h + 32 - 1) / 32);
		creal_T* fringe_temp2d;
		creal_T* fringe_temp3d;
		cudaMalloc((void**)&fringe_temp2d, sizeof(creal_T)*w*h);
		cudaMalloc((void**)&fringe_temp3d, sizeof(creal_T)*w*h);


		// spatial frequency distribution
		minfrequency_eta = 1.0 / (w * eta_interval);
		minfrequency_xi = 1.0 / (w * xi_interval);
		val.minfrequency_eta = minfrequency_eta;
		val.minfrequency_xi = minfrequency_xi;

		/*for (loop_ub = 0; loop_ub < i; loop_ub++) {
		double a;
		a = wavelength * (((static_cast<double>(loop_ub) + 1.0) - (w/2.0+1.0)) *
		minfrequency_eta);
		for (xi_id = 0; xi_id < i; xi_id++) {
		y_im = wavelength * (((static_cast<double>(xi_id) + 1.0) - (w / 2.0 + 1.0)) *
		minfrequency_xi);
		y_im = (knumber*depth) * std::sqrt((1.0 - a * a) - y_im * y_im);
		y_re = std::cos(y_im);
		y_im = std::sin(y_im);
		fringe_temp3[loop_ub + fringe_temp3.size(0) * xi_id].re =
		fringe_temp2[loop_ub + fringe_temp2.size(0) * xi_id].re * y_re -
		fringe_temp2[loop_ub + fringe_temp2.size(0) * xi_id].im * y_im;
		fringe_temp3[loop_ub + fringe_temp3.size(0) * xi_id].im =
		fringe_temp2[loop_ub + fringe_temp2.size(0) * xi_id].re * y_im +
		fringe_temp2[loop_ub + fringe_temp2.size(0) * xi_id].im * y_re;
		}
		}*/

		//  angular spectrum transfor function







		cudaMemcpy(fringe_temp2d, fringe_temp2.data(), sizeof(creal_T)*w*h, cudaMemcpyHostToDevice);

		transfer << <gridSize, blockSize >> > (val, fringe_temp2d, fringe_temp3d);




		cudaMemcpy(fringe_temp3.data(), fringe_temp3d, sizeof(creal_T)*w*h, cudaMemcpyDeviceToHost);
		cudaFree(fringe_temp2d);
		cudaFree(fringe_temp3d);
	}


	start = CUR_TIME;
	as->ifft2(fringe_temp3, b_AngularC);
	

	end = CUR_TIME;

	during += ((std::chrono::duration<Real>)(end - start)).count();
	LOG("%.5lfsec...done\n", during);
	
}
