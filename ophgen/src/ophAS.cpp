#define OPH_DM_EXPORT 

#include "ophAS.h"
#include <random>
#include "sys.h"
#include "tinyxml2.h"
#include "PLYparser.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ophAS_GPU.h"
#include "FFTImplementationCallback.h"



ophAS::ophAS()
{
	this->x = 0.0;
	this->y = 0.0;
	this->z = 0.0;
	this->amplitude = 1.0;
}

ophAS::~ophAS()
{
	
}

bool ophAS::readConfig(const char * fname)
{
	LOG("Reading....%s...", fname);

	auto start = CUR_TIME;

	/*XML parsing*/
	tinyxml2::XMLDocument xml_doc;
	tinyxml2::XMLNode *xml_node;


	if (!checkExtension(fname, ".xml"))
	{
		LOG("file's extension is not 'xml'\n");
		return false;
	}
	auto ret = xml_doc.LoadFile(fname);
	LOG("%d", ret);
	if (ret != tinyxml2::XML_SUCCESS)
	{
		LOG("Failed to load file \"%s\"\n", fname);
		return false;
	}

	xml_node = xml_doc.FirstChild();

	int nWave = 1;
	auto next = xml_node->FirstChildElement("ScaleX");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&pc_config.scale[_X]))
		return false;
	next = xml_node->FirstChildElement("ScaleY");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&pc_config.scale[_Y]))
		return false;
	next = xml_node->FirstChildElement("ScaleZ");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&pc_config.scale[_Z]))
		return false;
	next = xml_node->FirstChildElement("Distance");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&pc_config.distance))
		return false;
	depth = pc_config.distance;



	next = xml_node->FirstChildElement("SLM_WaveNum"); // OffsetInDepth
	if (!next || tinyxml2::XML_SUCCESS != next->QueryIntText(&nWave))
		return false;

	context_.waveNum = nWave;
	if (context_.wave_length) delete[] context_.wave_length;
	context_.wave_length = new Real[nWave];

	char szNodeName[32] = { 0, };
	for (int i = 1; i <= nWave; i++) {
		wsprintfA(szNodeName, "SLM_WaveLength_%d", i);
		
		next = xml_node->FirstChildElement(szNodeName);
		if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&context_.wave_length[i - 1]))
			return false;
	}
	wavelength = context_.wave_length[0];
	next = xml_node->FirstChildElement("SLM_PixelNumX");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryIntText(&context_.pixel_number[_X]))
		return false;
	next = xml_node->FirstChildElement("SLM_PixelNumY");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryIntText(&context_.pixel_number[_Y]))
		return false;
	this->w = context_.pixel_number[_X];
	this->h = context_.pixel_number[_Y];
	next = xml_node->FirstChildElement("SLM_PixelPitchX");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&context_.pixel_pitch[_X]))
		return false;
	next = xml_node->FirstChildElement("SLM_PixelPitchY");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&context_.pixel_pitch[_Y]))
		return false;
	xi_interval = context_.pixel_pitch[_X];
	eta_interval = context_.pixel_pitch[_Y];
	next = xml_node->FirstChildElement("IMG_Rotation");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryBoolText(&context_.bRotation))
		context_.bRotation = false;
	next = xml_node->FirstChildElement("IMG_Merge");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryBoolText(&context_.bMergeImg))
		context_.bMergeImg = true;
	next = xml_node->FirstChildElement("DoublePrecision");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryBoolText(&context_.bUseDP))
		context_.bUseDP = true;
	next = xml_node->FirstChildElement("ShiftX");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&context_.shift[_X]))
		context_.shift[_X] = 0.0;
	next = xml_node->FirstChildElement("ShiftY");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&context_.shift[_Y]))
		context_.shift[_Y] = 0.0;
	next = xml_node->FirstChildElement("ShiftZ");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&context_.shift[_Z]))
		context_.shift[_Z] = 0.0;
	next = xml_node->FirstChildElement("FieldLength");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&m_dFieldLength))
		m_dFieldLength = 0.0;
	next = xml_node->FirstChildElement("NumOfStream");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryIntText(&m_nStream))
		m_nStream = 1;

	context_.ss[_X] = context_.pixel_number[_X] * context_.pixel_pitch[_X];
	context_.ss[_Y] = context_.pixel_number[_Y] * context_.pixel_pitch[_Y];

	Openholo::setPixelNumberOHC(context_.pixel_number);
	Openholo::setPixelPitchOHC(context_.pixel_pitch);

	OHC_encoder->clearWavelength();
	for (int i = 0; i < nWave; i++)
		Openholo::setWavelengthOHC(context_.wave_length[i], LenUnit::m);

	auto end = CUR_TIME;

	auto during = ((std::chrono::duration<Real>)(end - start)).count();
	knumber = 2 * M_PI / wavelength;
	LOG("%.5lfsec...done\n", during);
	initialize();
	return true;
}

int ophAS::loadPoint(const char * fname)
{
	n_points = ophGen::loadPointCloud(fname, &pc_data);
	return n_points;
}

void ophAS::setmode(bool is_cpu)
{
	this->is_CPU = is_cpu;
}

void ophAS::ASCalculation(double w, double h, double wavelength, double knumber, double
	xi_interval, double eta_interval, double depth, coder::
	array<creal_T, 2U> &fringe, coder::array<creal_T, 2U>
	&b_AngularC)
{
	fringe.set_size(w, h);
	RayleighSommerfield(w, h, wavelength, knumber, xi_interval, eta_interval, depth, fringe);
	depth = -500e-3;
	if (is_CPU)
	{
		AngularSpectrum(w, h, wavelength, knumber, xi_interval, eta_interval, depth, fringe, b_AngularC);
	}
	else
	{
		Angular_Spectrum_GPU(w, h, wavelength, knumber, xi_interval, eta_interval, depth, fringe, b_AngularC);
	}
	
}

void ophAS::RayleighSommerfield(double w, double h, double wavelength, double knumber, double xi_interval, double eta_interval, double depth, coder::array<creal_T, 2U>& fringe)
{
	double pi = atan(1) * 4;
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> dis(0, 1);
	double init_phase = 1.0;
	double eta_tilting = 0 * pi / 180;
	double xi_tilting = 0 * pi / 180;
	int n = log10(w) / log10(2);
	for(int i=0;i<w; i++)
	{
		double eta = (i - w / 2)*eta_interval;
		for (int j = 0; j < h; j++)
		{
			double xi = (j - h / 2)*xi_interval;
			double R = sqrt(pow(eta - this->y, 2) + pow(xi - this->x, 2) + pow(depth + this->z, 2));
			double total_phase = knumber*R + knumber*xi*sin(xi_tilting) + knumber*eta*sin(eta_tilting) + init_phase * 2 * pi;
			complex<double> tmp = (this->amplitude / R)*exp(total_phase*I);
			fringe[i + (j<<n)].re = tmp.real();
			fringe[i + (j <<n)].im = tmp.imag();
		}
	}
}

void ophAS::AngularSpectrum(double w, double h, double wavelength, double knumber, double xi_interval, double eta_interval, double depth, const coder::array<creal_T, 2U>& fringe, coder::array<creal_T, 2U>& b_AngularC)
{
	auto start = CUR_TIME;
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



	i = static_cast<int>(w);
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
	}
	
	eml_fftshift(fringe_temp1, 1);
	eml_fftshift(fringe_temp1, 2);
	fft2_matlab(fringe_temp1, fringe_temp2);
	eml_fftshift(fringe_temp2, 1);
	eml_fftshift(fringe_temp2, 2);

	
	
	minfrequency_eta = 1.0 / (w * eta_interval);
	minfrequency_xi = 1.0 / (w * xi_interval);
		

	for (loop_ub = 0; loop_ub < i; loop_ub++) {
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
	}

	//  angular spectrum transfor function
	

	
	ifft2(fringe_temp3, b_AngularC);
	
	
	auto end = CUR_TIME;

	auto during = ((std::chrono::duration<Real>)(end - start)).count();
	LOG("%.5lfsec...done\n", during);
	
	
}

void ophAS::generateHologram()
{
	coder::array<creal_T, 2U> temp;
	res = new unsigned char[w*h]{ 0 };
	ASCalculation(w, h, wavelength, knumber, xi_interval, eta_interval, depth, temp, im);
	for (int i = 0; i < w*h; i++)
	{
		Complex<double> a(im[i].re, im[i].im);
		if (a.mag() * 255 > 255)
			res[i] = 255;
		else if (a.mag() * 255 < 0)
			res[i] = 0;
		else
			res[i] = a.mag() * 255;
		
	}

}

void ophAS::MemoryRelease()
{
	delete[] res;
}

void ophAS::ifft2(const coder::array<creal_T, 2U>& x, coder::array<creal_T, 2U>& y)
{
	coder::array<creal_T, 2U> xPerm;
	int loop_ub;
	int i;
	short lens[2];
	boolean_T guard1 = false;
	int nRows;
	coder::array<double, 2U> costab;
	coder::array<double, 2U> sintab;
	coder::array<double, 2U> sintabinv;
	coder::array<creal_T, 2U> acc;
	
	xPerm.set_size(x.size(0), x.size(1));
	loop_ub = x.size(0) * x.size(1);
	for (i = 0; i < loop_ub; i++) {
		xPerm[i] = x[i];
	}

	ifftshift(xPerm);
	lens[0] = static_cast<short>(xPerm.size(0));
	lens[1] = static_cast<short>(xPerm.size(1));
	guard1 = false;
	if ((xPerm.size(0) == 0) || (xPerm.size(1) == 0)) {
		guard1 = true;
	}
	else {
		boolean_T useRadix2;
		boolean_T exitg1;
		useRadix2 = false;
		loop_ub = 0;
		exitg1 = false;
		while ((!exitg1) && (loop_ub < 2)) {
			if (0 == lens[loop_ub]) {
				useRadix2 = true;
				exitg1 = true;
			}
			else {
				loop_ub++;
			}
		}

		if (useRadix2) {
			guard1 = true;
		}
		else {
			int i1;
			useRadix2 = ((lens[0] > 0) && ((lens[0] & (lens[0] - 1)) == 0));
			FFTImplementationCallback::get_algo_sizes((static_cast<int>(lens[0])),
				(useRadix2), (&loop_ub), (&nRows));
			FFTImplementationCallback::b_generate_twiddle_tables((nRows), (useRadix2),
				(costab), (sintab), (sintabinv));
			if (useRadix2) {
				FFTImplementationCallback::b_r2br_r2dit_trig((xPerm), (static_cast<int>
					(lens[0])), (costab), (sintab), (acc));
			}
			else {
				FFTImplementationCallback::b_dobluesteinfft((xPerm), (loop_ub), (
					static_cast<int>(lens[0])), (costab), (sintab), (sintabinv), (acc));
			}

			xPerm.set_size(acc.size(1), acc.size(0));
			loop_ub = acc.size(0);
			for (i = 0; i < loop_ub; i++) {
				nRows = acc.size(1);
				for (i1 = 0; i1 < nRows; i1++) {
					xPerm[i1 + xPerm.size(0) * i] = acc[i + acc.size(0) * i1];
				}
			}

			useRadix2 = ((lens[1] > 0) && ((lens[1] & (lens[1] - 1)) == 0));
			FFTImplementationCallback::get_algo_sizes((static_cast<int>(lens[1])),
				(useRadix2), (&loop_ub), (&nRows));
			FFTImplementationCallback::b_generate_twiddle_tables((nRows), (useRadix2),
				(costab), (sintab), (sintabinv));
			if (useRadix2) {
				FFTImplementationCallback::b_r2br_r2dit_trig((xPerm), (static_cast<int>
					(lens[1])), (costab), (sintab), (acc));
			}
			else {
				FFTImplementationCallback::b_dobluesteinfft((xPerm), (loop_ub), (
					static_cast<int>(lens[1])), (costab), (sintab), (sintabinv), (acc));
			}

			y.set_size(acc.size(1), acc.size(0));
			loop_ub = acc.size(0);
			for (i = 0; i < loop_ub; i++) {
				nRows = acc.size(1);
				for (i1 = 0; i1 < nRows; i1++) {
					y[i1 + y.size(0) * i] = acc[i + acc.size(0) * i1];
				}
			}
		}
	}

	if (guard1) {
		y.set_size((static_cast<int>(lens[0])), (static_cast<int>(lens[1])));
		loop_ub = lens[0] * lens[1];
		for (i = 0; i < loop_ub; i++) {
			y[i].re = 0.0;
			y[i].im = 0.0;
		}
	}

	ifftshift(y);
}

void ophAS::fft2_matlab(coder::array<creal_T, 2U>& x, coder::array<creal_T, 2U>& y)
{
	short lens[2];
	boolean_T guard1 = false;
	int k;
	int i;
	int nRows;
	coder::array<double, 2U> costab;
	coder::array<double, 2U> sintab;
	coder::array<double, 2U> sintabinv;
	coder::array<creal_T, 2U> acc;
	coder::array<creal_T, 2U> xPerm;
	lens[0] = static_cast<short>(x.size(0));
	lens[1] = static_cast<short>(x.size(1));
	guard1 = false;
	if ((x.size(0) == 0) || (x.size(1) == 0)) {
		guard1 = true;
	}
	else {
		boolean_T useRadix2;
		boolean_T exitg1;
		useRadix2 = false;
		k = 0;
		exitg1 = false;
		while ((!exitg1) && (k < 2)) {
			if (0 == lens[k]) {
				useRadix2 = true;
				exitg1 = true;
			}
			else {
				k++;
			}
		}

		if (useRadix2) {
			guard1 = true;
		}
		else {
			int i1;
			useRadix2 = ((lens[0] > 0) && ((lens[0] & (lens[0] - 1)) == 0));
			FFTImplementationCallback::get_algo_sizes((static_cast<int>(lens[0])),
				(useRadix2), (&k), (&nRows));
			FFTImplementationCallback::generate_twiddle_tables((nRows), (useRadix2),
				(costab), (sintab), (sintabinv));
			if (useRadix2) {
				FFTImplementationCallback::r2br_r2dit_trig((x), (static_cast<int>(lens[0])),
					(costab), (sintab), (acc));
			}
			else {
				FFTImplementationCallback::dobluesteinfft((x), (k), (static_cast<int>
					(lens[0])), (costab), (sintab), (sintabinv), (acc));
			}

			xPerm.set_size(acc.size(1), acc.size(0));
			k = acc.size(0);
			for (i = 0; i < k; i++) {
				nRows = acc.size(1);
				for (i1 = 0; i1 < nRows; i1++) {
					xPerm[i1 + xPerm.size(0) * i] = acc[i + acc.size(0) * i1];
				}
			}

			useRadix2 = ((lens[1] > 0) && ((lens[1] & (lens[1] - 1)) == 0));
			FFTImplementationCallback::get_algo_sizes((static_cast<int>(lens[1])),
				(useRadix2), (&k), (&nRows));
			FFTImplementationCallback::generate_twiddle_tables((nRows), (useRadix2),
				(costab), (sintab), (sintabinv));
			if (useRadix2) {
				FFTImplementationCallback::r2br_r2dit_trig((xPerm), (static_cast<int>
					(lens[1])), (costab), (sintab), (acc));
			}
			else {
				FFTImplementationCallback::dobluesteinfft((xPerm), (k), (static_cast<int>
					(lens[1])), (costab), (sintab), (sintabinv), (acc));
			}

			y.set_size(acc.size(1), acc.size(0));
			k = acc.size(0);
			for (i = 0; i < k; i++) {
				nRows = acc.size(1);
				for (i1 = 0; i1 < nRows; i1++) {
					y[i1 + y.size(0) * i] = acc[i + acc.size(0) * i1];
				}
			}
		}
	}

	if (guard1) {
		y.set_size((static_cast<int>(lens[0])), (static_cast<int>(lens[1])));
		k = lens[0] * lens[1];
		for (i = 0; i < k; i++) {
			y[i].re = 0.0;
			y[i].im = 0.0;
		}
	}
}

void ophAS::eml_fftshift(coder::array<creal_T, 2U>& x, int dim)
{
	int vstride;
	int npages;
	int i2;
	int ia;
	int ib;
	if (x.size((dim - 1)) > 1) {
		int vlend2;
		int k;
		int vspread;
		int midoffset;
		vlend2 = x.size((dim - 1)) / 2;
		vstride = 1;
		for (k = 0; k <= dim - 2; k++) {
			vstride *= x.size(0);
		}

		npages = 1;
		vspread = dim + 1;
		for (k = vspread; k < 3; k++) {
			npages *= x.size(1);
		}

		vspread = (x.size((dim - 1)) - 1) * vstride;
		midoffset = vlend2 * vstride - 1;
		if (vlend2 << 1 == x.size((dim - 1))) {
			i2 = 0;
			for (int i = 0; i < npages; i++) {
				int i1;
				i1 = i2;
				i2 += vspread;
				for (int j = 0; j < vstride; j++) {
					i1++;
					i2++;
					ia = i1 - 1;
					ib = i1 + midoffset;
					for (k = 0; k < vlend2; k++) {
						double xtmp_re;
						double xtmp_im;
						xtmp_re = x[ia].re;
						xtmp_im = x[ia].im;
						x[ia] = x[ib];
						x[ib].re = xtmp_re;
						x[ib].im = xtmp_im;
						ia += vstride;
						ib += vstride;
					}
				}
			}
		}
		else {
			i2 = 0;
			for (int i = 0; i < npages; i++) {
				int i1;
				i1 = i2;
				i2 += vspread;
				for (int j = 0; j < vstride; j++) {
					double xtmp_re;
					double xtmp_im;
					i1++;
					i2++;
					ia = i1 - 1;
					ib = i1 + midoffset;
					xtmp_re = x[ib].re;
					xtmp_im = x[ib].im;
					for (k = 0; k < vlend2; k++) {
						int ic;
						ic = ib + vstride;
						x[ib] = x[ia];
						x[ia] = x[ic];
						ia += vstride;
						ib = ic;
					}

					x[ib].re = xtmp_re;
					x[ib].im = xtmp_im;
				}
			}
		}
	}
}

void ophAS::ifftshift(coder::array<creal_T, 2U>& x)
{
	
	int vlend2;
	int npages;
	int k;
	int vspread;
	int midoffset;
	int i2;
	int i;
	int i1;
	int j;
	int ia;
	int ib;
	double xtmp_re;
	double xtmp_im;
	int ic;
	if (x.size(0) > 1) {
		vlend2 = x.size(0) / 2;
		if (vlend2 << 1 == x.size(0)) {
			eml_fftshift(x, 1);
		}
		else {
			npages = 1;
			for (k = 2; k < 3; k++) {
				npages *= x.size(1);
			}

			vspread = x.size(0) - 1;
			midoffset = vlend2 - 1;
			i2 = -1;
			for (i = 0; i < npages; i++) {
				i1 = i2 + 1;
				i2 += vspread;
				for (j = 0; j < 1; j++) {
					i1++;
					i2++;
					ia = i1 + midoffset;
					ib = i2;
					xtmp_re = x[i2].re;
					xtmp_im = x[i2].im;
					for (k = 0; k < vlend2; k++) {
						ia--;
						ic = ib;
						ib--;
						x[ic] = x[ia];
						x[ia] = x[ib];
					}

					x[ib].re = xtmp_re;
					x[ib].im = xtmp_im;
				}
			}
		}
	}

	if (x.size(1) > 1) {
		vlend2 = x.size(1) / 2;
		if (vlend2 << 1 == x.size(1)) {
			eml_fftshift(x, 2);
		}
		else {
			npages = 1;
			for (k = 0; k < 1; k++) {
				npages *= x.size(0);
			}

			vspread = (x.size(1) - 1) * npages;
			midoffset = vlend2 * npages - 1;
			i2 = -1;
			for (i = 0; i < 1; i++) {
				i1 = i2 + 1;
				i2 += vspread;
				for (j = 0; j < npages; j++) {
					i1++;
					i2++;
					ia = i1 + midoffset;
					ib = i2;
					xtmp_re = x[i2].re;
					xtmp_im = x[i2].im;
					for (k = 0; k < vlend2; k++) {
						ia -= npages;
						ic = ib;
						ib -= npages;
						x[ic] = x[ia];
						x[ia] = x[ib];
					}

					x[ib].re = xtmp_re;
					x[ib].im = xtmp_im;
				}
			}
		}
	}
	
}

int ophAS::save(const char * fname, uint8_t bitsperpixel, uchar * src, uint px, uint py)
{
	if (fname == nullptr) return -1;

	uchar* source = src;
	ivec2 p(px, py);

	if (src == nullptr)
		source = m_lpNormalized[0];
	if (px == 0 && py == 0)
		p = ivec2(context_.pixel_number[_X], context_.pixel_number[_Y]);

	if (checkExtension(fname, ".bmp")) 	// when the extension is bmp
		return Openholo::saveAsImg(fname, bitsperpixel, source, p[_X], p[_Y]);
	else {									// when extension is not .ohf, .bmp - force bmp
		char buf[256];
		memset(buf, 0x00, sizeof(char) * 256);
		sprintf_s(buf, "%s.bmp", fname);

		return Openholo::saveAsImg(buf, bitsperpixel, source, p[_X], p[_Y]);
	}
}

void ophAS::save(const char * fname)
{
	save(fname, 8, res, w, h);
	
}
