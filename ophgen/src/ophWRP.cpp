#include "ophwrp.h"

ophWRP::ophWRP(void) 
	: ophGen()
{
	n_points = -1;
	p_wrp_ = nullptr;
}

ophWRP::~ophWRP(void)
{
}


int ophWRP::loadPointCloud(const char* pc_file)
{
	n_points = ophGen::loadPointCloud(pc_file, &obj_);

	return n_points;

}

bool ophWRP::readConfig(const char* cfg_file)
{
	if (!ophGen::readConfig(cfg_file, pc_config_))
		return false;

	return true;
}

void ophWRP::encodefield(void)
{
	const int size = context_.pixel_number.v[_X] * context_.pixel_number.v[_Y];

	/*	initialize	*/
	int encode_size = size;
	if (holo_encoded != nullptr) delete[] holo_encoded;
	holo_encoded = new Real[size];
	memset(holo_encoded, 0, sizeof(double) * size);

	if (holo_normalized != nullptr) delete[] holo_normalized;
	holo_normalized = new uchar[size];
	memset(holo_normalized, 0, sizeof(uchar) * size);

	int pnx = context_.pixel_number[_X];
	int pny = context_.pixel_number[_Y];

	int i = 0;
#pragma omp parallel for private(i)	
	for (i = 0; i < pnx*pny; i++) {
		holo_encoded[i] = holo_gen[i].angle();
	}


}

void ophWRP::normalize(void)
{
	oph::normalize((Real*)holo_encoded, holo_normalized, context_.pixel_number[_X], context_.pixel_number[_Y]);
}

void ophWRP::initialize()
{

	if (holo_gen) delete[] holo_gen;
	holo_gen = new oph::Complex<Real>[context_.pixel_number[_X] * context_.pixel_number[_Y]];
	memset(holo_gen, 0.0, sizeof(oph::Complex<Real>) * context_.pixel_number[_X] * context_.pixel_number[_Y]);

	if (holo_encoded) delete[] holo_encoded;
	holo_encoded = new Real[context_.pixel_number[_X] * context_.pixel_number[_Y]];
	memset(holo_encoded, 0.0, sizeof(Real) * context_.pixel_number[_X] * context_.pixel_number[_Y]);

	if (holo_normalized) delete[] holo_normalized;
	holo_normalized = new uchar[context_.pixel_number[_X] * context_.pixel_number[_Y]];
	memset(holo_normalized, 0.0, sizeof(uchar) * context_.pixel_number[_X] * context_.pixel_number[_Y]);

}

void ophWRP::addPixel2WRP(int x, int y, Complex<Real> temp)
{
	long long int Nx = context_.pixel_number.v[0];
	long long int Ny = context_.pixel_number.v[1];
//	oph::Complex<double> *p = holo_gen;

	if (x >= 0 && x<Nx && y >= 0 && y< Ny) {
		long long int adr = x + y*Nx;
		if (adr == 0) std::cout << ".0";
//		p[adr] = p[adr] + temp;
		p_wrp_[adr] = p_wrp_[adr] + temp;
	}

}

void ophWRP::addPixel2WRP(int x, int y, oph::Complex<Real> temp, oph::Complex<Real>* wrp)
{
	long long int Nx = context_.pixel_number.v[0];
	long long int Ny = context_.pixel_number.v[1];

	if (x >= 0 && x<Nx && y >= 0 && y< Ny) {
		long long int adr = x + y*Nx;
		wrp[adr] += temp[adr];
	}
}

oph::Complex<Real>* ophWRP::calSubWRP(double wrp_d, Complex<Real>* wrp, OphPointCloudData* pc)
{

	Real wave_num = context_.k;   // wave_number
	Real wave_len = context_.lambda;  //wave_length

	int Nx = context_.pixel_number.v[0]; //slm_pixelNumberX
	int Ny = context_.pixel_number.v[1]; //slm_pixelNumberY

	Real wpx = context_.pixel_pitch.v[0];//wrp pitch
	Real wpy = context_.pixel_pitch.v[1];


	int Nx_h = Nx >> 1;
	int Ny_h = Ny >> 1;

	int num = n_points;


#ifdef _OPENMP
	omp_set_num_threads(omp_get_num_threads());
#pragma omp parallel for
#endif

	for (int k = 0; k < num; k++) {

		uint idx = 3 * k;
		Real x = pc->vertex[idx + _X];
		Real y = pc->vertex[idx + _Y];
		Real z = pc->vertex[idx + _Z];

		float dz = wrp_d - z;
		//	float tw = (int)fabs(wave_len*dz / wpx / wpx / 2 + 0.5) * 2 - 1;
		float tw = fabs(dz)*wave_len / wpx / wpx / 2;

		int w = (int)tw;

		int tx = (int)(x / wpx) + Nx_h;
		int ty = (int)(y / wpy) + Ny_h;

		printf("num=%d, tx=%d, ty=%d, w=%d\n", k, tx, ty, w);

		for (int wy = -w; wy < w; wy++) {
			for (int wx = -w; wx<w; wx++) {//WRP coordinate

				double dx = wx*wpx;
				double dy = wy*wpy;
				double dz = wrp_d - z;

				double sign = (dz>0.0) ? (1.0) : (-1.0);
				double r = sign*sqrt(dx*dx + dy*dy + dz*dz);

				//double tmp_re,tmp_im;
				Complex<Real> tmp;

				tmp._Val[_RE] = cosf(wave_num*r) / (r + 0.05);
				tmp._Val[_IM] = sinf(wave_num*r) / (r + 0.05);

				if (tx + wx >= 0 && tx + wx < Nx && ty + wy >= 0 && ty + wy < Ny)
				{
					addPixel2WRP(wx + tx, wy + ty, tmp, wrp);
				}
			}
		}
	}

	return wrp;
}

double ophWRP::calculateWRP(void)
{
	initialize();

	Real wave_num = context_.k;   // wave_number
	Real wave_len = context_.lambda;  //wave_length

	int Nx = context_.pixel_number.v[0]; //slm_pixelNumberX
	int Ny = context_.pixel_number.v[1]; //slm_pixelNumberY

	Real wpx = context_.pixel_pitch.v[0];//wrp pitch
	Real wpy = context_.pixel_pitch.v[1];


	int Nx_h = Nx >> 1;
	int Ny_h = Ny >> 1;

	OphPointCloudData pc = obj_;
	Real wrp_d = pc_config_.wrp_location;

	// Memory Location for Result Image

	if (p_wrp_) delete[] p_wrp_;
	p_wrp_ = new oph::Complex<Real>[context_.pixel_number[_X] * context_.pixel_number[_Y]];
	memset(p_wrp_, 0.0, sizeof(oph::Complex<Real>) * context_.pixel_number[_X] * context_.pixel_number[_Y]);

	int num = n_points;
	auto time_start = CUR_TIME;

#ifdef _OPENMP
	omp_set_num_threads(omp_get_num_threads());
#pragma omp parallel for
#endif

	for (int k = 0; k < num; ++k) {
		uint idx = 3 * k;
		Real x = pc.vertex[idx + _X];
		Real y = pc.vertex[idx + _Y];
		Real z = pc.vertex[idx + _Z];

		float dz = wrp_d - z;
		float tw = (int)fabs(wave_len*dz / wpx / wpx / 2 + 0.5) * 2 - 1;
		//	float tw = fabs(dz)*wave_len / wpx / wpx / 2;

		int w = (int)tw;

		int tx = (int)(x / wpx) + Nx_h;
		int ty = (int)(y / wpy) + Ny_h;

		printf("num=%d, tx=%d, ty=%d, w=%d\n", k, tx, ty, w);

		for (int wy = -w; wy < w; wy++) {
			for (int wx = -w; wx<w; wx++) {//WRP coordinate

				double dx = wx*wpx;
				double dy = wy*wpy;
				double dz = wrp_d - z;

				double sign = (dz>0.0) ? (1.0) : (-1.0);
				double r = sign*sqrt(dx*dx + dy*dy + dz*dz);

				//double tmp_re,tmp_im;
				oph::Complex<Real> tmp;
				tmp._Val[_RE] = cosf(wave_num*r) / (r + 0.05);
				tmp._Val[_IM] = sinf(wave_num*r) / (r + 0.05);

				if (tx + wx >= 0 && tx + wx < Nx && ty + wy >= 0 && ty + wy < Ny)
					addPixel2WRP(wx + tx, wy + ty, tmp);

			}
		}
	}

	auto time_finish = CUR_TIME;
	return ((std::chrono::duration<Real>)(time_finish - time_start)).count();

}

void ophWRP::fresnelPropagation(Complex<Real>* in, Complex<Real>* out, Real distance) {

	int Nx = context_.pixel_number[_X];
	int Ny = context_.pixel_number[_Y];

	Real dx = context_.pixel_pitch[_X];
	Real dy = context_.pixel_pitch[_Y];

	Real k = context_.k;

	Real fx = 1 / (Nx*dx);
	Real fy = 1 / (Ny*dy);

	Complex<Real>* in2x = new Complex<Real>[Nx*Ny];
	Complex<Real> zero(0, 0);
	oph::memsetArr<Complex<Real>>(in2x, zero, 0, Nx*Ny);

	uint idxIn = 0;

	for (idxIn = 0; idxIn<Nx*Ny; idxIn++)
		in2x[idxIn] = in[idxIn];

	Real* x = new Real[Nx*Ny];
	Real* y = new Real[Nx*Ny];

	uint i = 0;
	for (uint idy = (1 - Ny / 2); idy < (1 + Ny / 2); idy++) {
		for (uint idx = (1 - Nx / 2); idx < (1 + Nx / 2); idx++) {
			x[i] = idx;
			y[i] = idy;
			i++;
		}
	}

	Complex<Real>* prop = new Complex<Real>[Nx*Ny];
	fft2({ Nx, Ny }, in2x, OPH_FORWARD, OPH_ESTIMATE);
	fftExecute(prop);

	Complex<Real> part;

	Complex<Real>* temp2 = new Complex<Real>[Nx*Ny];

	for (uint i = 0; i < Nx*Ny; i++) {

		Real kk = M_PI*context_.lambda *distance *(x[i] * x[i] + y[i] * y[i]);
		part._Val[_RE] = cos(k*distance)*cos(kk);
		part._Val[_IM] = sin(k*distance)*sin(M_PI*context_.lambda*distance*(x[i] * x[i] + y[i] * y[i]));

		temp2[i]._Val[_RE] = prop[i]._Val[_RE] * part._Val[_RE];
		temp2[i]._Val[_IM] = prop[i]._Val[_IM] * part._Val[_IM];
	}

	fft2({ Nx, Ny }, temp2, OPH_BACKWARD, OPH_ESTIMATE);
	fftExecute(holo_gen);

}

void ophWRP::generateHologram(void)
{
	printf("Generating Hologram\n");
	Real distance = pc_config_.propagation_distance;
	fresnelPropagation(p_wrp_, holo_gen, distance);
	printf("Hologram Generated!\n");

}

oph::Complex<Real>** ophWRP::calculateMWRP(void)
{
	int wrp_num = pc_config_.num_wrp;

	if (wrp_num < 1)
		return nullptr;

	oph::Complex<Real>** wrp_list;

	Real wave_num = context_.k;   // wave_number
	Real wave_len = context_.lambda;  //wave_length

	int Nx = context_.pixel_number.v[0]; //slm_pixelNumberX
	int Ny = context_.pixel_number.v[1]; //slm_pixelNumberY

	Real wpx = context_.pixel_pitch.v[0];//wrp pitch
	Real wpy = context_.pixel_pitch.v[1];


	int Nx_h = Nx >> 1;
	int Ny_h = Ny >> 1;

	oph::Complex<Real>* wrp;

	// Memory Location for Result Image
	if (wrp != nullptr) free(wrp);
	wrp = (oph::Complex<Real>*)calloc(1, sizeof(oph::Complex<Real>) * Nx * Ny);

//	double wrp_d = pc_config_.offset_depth / wrp_num;

	OphPointCloudData pc = obj_;

	for (int i = 0; i<wrp_num; i++)
	{
//		wrp = calSubWRP(wrp_d, wrp, &pc);
		wrp_list[i] = wrp;
	}

	return wrp_list;
}

void ophWRP::ophFree(void)
{
	//	delete[] obj_.vertex;
	//	delete[] obj_.color;

}

