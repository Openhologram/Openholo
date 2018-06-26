#include "ophwrp.h"

ophWRP::ophWRP(void) :ophGen()
{
	n_points = -1;
	obj_ = new OphPointCloudData();
	p_wrp_ = nullptr;
}

/*ophWRP::~ophWRP(void)
{
}*/



OphPointCloudData* ophWRP::vector2pointer(std::vector<OphPointCloudData> vec)
{
	if (n_points<0)
	{
		std::cerr << "Invail operation";
	}

	int num = n_points;

	//ophObjPoint *op = (ophObjPoint*)malloc(num * sizeof(ophObjPoint));
	OphPointCloudData *op = (OphPointCloudData*)malloc(num * sizeof(OphPointCloudData));

	for (int i = 0; i < num; i++)
		op[i] = vec[i];

	return op;

}

int ophWRP::pobj2vecobj()
{
	if (obj_ == NULL)
		return -1;

	for (int n = 0; n<n_points; n++)
		vec_obj.push_back(obj_[n]);

	return 0;
}


int ophWRP::loadwPointCloud(const char* pc_file, bool colorinfo)
{

	//	if(colorinfo==true)
	//	n_points = ophGen::loadPointCloud(pc_file,obj_, PC_XYZ | PC_RGB);
	//	if(colorinfo==false)
	n_points = ophGen::loadPointCloud(pc_file, obj_, PC_XYZ);

	return n_points;

}

bool ophWRP::readConfig(const char* cfg_file)
{
	if (!ophGen::readConfig(cfg_file, pc_config_))
		return false;

	return true;
}



void ophWRP::AddPixel2WRP(int x, int y, Complex<real> temp)
{
	long long int Nx = context_.pixel_number.v[0];
	long long int Ny = context_.pixel_number.v[1];
	Complex<real> *p = getWRPBuff();

	if (x >= 0 && x<Nx && y >= 0 && y< Ny) {
		long long int adr = x + y*Nx;
		p[adr].re += temp.re;
		p[adr].im += temp.im;
	}

}

void ophWRP::AddPixel2WRP(int x, int y, oph::Complex<real> temp, oph::Complex<real>* wrp)
{
	long long int Nx = context_.pixel_number.v[0];
	long long int Ny = context_.pixel_number.v[1];

	if (x >= 0 && x<Nx && y >= 0 && y< Ny) {
		long long int adr = x + y*Nx;
		wrp[adr].re += temp.re;
		wrp[adr].im += temp.im;
	}

}

oph::Complex<real>* ophWRP::subWRP_calcu(double wrp_d, Complex<real>* wrp, OphPointCloudData* pc)
{

	real wave_num = context_.k;   // wave_number
	real wave_len = context_.lambda;  //wave_length

	int Nx = context_.pixel_number.v[0]; //slm_pixelNumberX
	int Ny = context_.pixel_number.v[1]; //slm_pixelNumberY

	real wpx = context_.pixel_pitch.v[0];//wrp pitch
	real wpy = context_.pixel_pitch.v[1];


	int Nx_h = Nx >> 1;
	int Ny_h = Ny >> 1;

	int num = n_points;


#ifdef _OPENMP
	omp_set_num_threads(omp_get_num_threads());
#pragma omp parallel for
#endif

	for (int k = 0; k < num; k++) {

		real x = pc->location[k][_X] * pc_config_.scale[_X];
		real y = pc->location[k][_Y] * pc_config_.scale[_Y];
		real z = pc->location[k][_Z] * pc_config_.scale[_Z] + pc_config_.offset_depth;


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
				Complex<real> tmp;
				tmp.re = cosf(wave_num*r) / (r + 0.05);
				tmp.im = sinf(wave_num*r) / (r + 0.05);

				if (tx + wx >= 0 && tx + wx < Nx && ty + wy >= 0 && ty + wy < Ny)
					AddPixel2WRP(wx + tx, wy + ty, tmp, wrp);

			}
		}
	}

	return wrp;
}

double ophWRP::calculateWRP(double wrp_d)
{
	real wave_num = context_.k;   // wave_number
	real wave_len = context_.lambda;  //wave_length

	int Nx = context_.pixel_number.v[0]; //slm_pixelNumberX
	int Ny = context_.pixel_number.v[1]; //slm_pixelNumberY

	real wpx = context_.pixel_pitch.v[0];//wrp pitch
	real wpy = context_.pixel_pitch.v[1];


	int Nx_h = Nx >> 1;
	int Ny_h = Ny >> 1;

	OphPointCloudData *pc = obj_;

	// Memory Location for Result Image
	if (p_wrp_ != nullptr) free(p_wrp_);
	p_wrp_ = (oph::Complex<real>*)calloc(1, sizeof(oph::Complex<real>) * Nx * Ny);

	int num = n_points;
	std::chrono::system_clock::time_point time_start = std::chrono::system_clock::now();

#ifdef _OPENMP
	omp_set_num_threads(omp_get_num_threads());
#pragma omp parallel for
#endif

	for (int k = 0; k < num; k++) {

		real x = pc->location[k][_X] * pc_config_.scale[_X];
		real y = pc->location[k][_Y] * pc_config_.scale[_Y];
		real z = pc->location[k][_Z] * pc_config_.scale[_Z] + pc_config_.offset_depth;

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
				Complex<real> tmp;
				tmp.re = cosf(wave_num*r) / (r + 0.05);
				tmp.im = sinf(wave_num*r) / (r + 0.05);

				if (tx + wx >= 0 && tx + wx < Nx && ty + wy >= 0 && ty + wy < Ny)
					AddPixel2WRP(wx + tx, wy + ty, tmp);

			}
		}
	}

	std::chrono::system_clock::time_point time_finish = std::chrono::system_clock::now();
	return ((std::chrono::duration<real>)(time_finish - time_start)).count();

	//	return 0;
}

oph::Complex<real>** ophWRP::calculateWRP(int wrp_num)
{

	if (wrp_num < 1)
		return nullptr;

	Complex<real>** wrp_list;

	real wave_num = context_.k;   // wave_number
	real wave_len = context_.lambda;  //wave_length

	int Nx = context_.pixel_number.v[0]; //slm_pixelNumberX
	int Ny = context_.pixel_number.v[1]; //slm_pixelNumberY

	real wpx = context_.pixel_pitch.v[0];//wrp pitch
	real wpy = context_.pixel_pitch.v[1];


	int Nx_h = Nx >> 1;
	int Ny_h = Ny >> 1;

	//OphPointCloudData *pc = obj_;
	oph::Complex<real>* wrp;

	// Memory Location for Result Image
	if (wrp != nullptr) free(wrp);
	wrp = (oph::Complex<real>*)calloc(1, sizeof(oph::Complex<real>) * Nx * Ny);

	pobj2vecobj();

	double wrp_d = pc_config_.offset_depth / wrp_num;

	OphPointCloudData* pc = obj_;

	for (int i = 0; i<wrp_num; i++)
	{
		wrp = subWRP_calcu(wrp_d, wrp, pc);
		wrp_list[i] = wrp;
	}

	return wrp_list;
}
