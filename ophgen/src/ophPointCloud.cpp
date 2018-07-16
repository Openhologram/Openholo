#include "ophPointCloud.h"
#include "include.h"

#include <sys.h>

ophPointCloud::ophPointCloud(void)
	: ophGen()
{
	setMode(false);
	n_points = -1;
}

ophPointCloud::ophPointCloud(const char* pc_file, const char* cfg_file)
	: ophGen()
{
	setMode(false);
	n_points = loadPointCloud(pc_file);
	if (n_points == -1) std::cerr << "OpenHolo Error : Failed to load Point Cloud Data File(*.dat)" << std::endl;

	bool b_read = readConfig(cfg_file);
	if (!b_read) std::cerr << "OpenHolo Error : Failed to load Config Specification Data File(*.config)" << std::endl;
}

ophPointCloud::~ophPointCloud(void)
{
}

void ophPointCloud::initialize(void)
{
	// Output Image Size
	int n_x = context_.pixel_number[_X];
	int n_y = context_.pixel_number[_Y];

	// Memory Location for Result Image
	if (holo_gen != nullptr) delete[] holo_gen;
	holo_gen = new oph::Complex<Real>[n_x * n_y];
	memset(holo_gen, 0, sizeof(Complex<Real>) * n_x * n_y);

	if (holo_encoded != nullptr) delete[] holo_encoded;
	holo_encoded = new Real[n_x * n_y];
	memset(holo_encoded, 0, sizeof(Real) * n_x * n_y);

	if (holo_normalized != nullptr) delete[] holo_normalized;
	holo_normalized = new uchar[n_x * n_y];
	memset(holo_normalized, 0, sizeof(uchar) * n_x * n_y);
}

void ophPointCloud::setMode(bool is_CPU)
{
	this->is_CPU = is_CPU;
}

int ophPointCloud::loadPointCloud(const char* pc_file)
{
	n_points = ophGen::loadPointCloud(pc_file, &pc_data_);

	return n_points;
}

bool ophPointCloud::readConfig(const char* cfg_file)
{
	if (!ophGen::readConfig(cfg_file, pc_config_))
		return false;

	initialize();

	return true;
}

Real ophPointCloud::generateHologram(uint diff_flag)
{
	auto start_time = CUR_TIME;

	// Create CGH Fringe Pattern by 3D Point Cloud
	if (is_CPU == true) { //Run CPU
#ifdef _OPENMP
		std::cout << "Generate Hologram with Multi Core CPU" << std::endl;
#else
		std::cout << "Generate Hologram with Single Core CPU" << std::endl;
#endif
		genCghPointCloudCPU(diff_flag); /// 홀로그램 데이터 Complex data로 변경 시 holo_gen으로
	}
	else { //Run GPU
		std::cout << "Generate Hologram with GPU" << std::endl;

		genCghPointCloudGPU(diff_flag);
		std::cout << ">>> CUDA GPGPU" << std::endl;
	}

	auto end_time = CUR_TIME;

	auto during_time = ((std::chrono::duration<Real>)(end_time - start_time)).count();

	LOG("Implement time : %.5lf sec\n", during_time);

	return during_time;
}

void ophPointCloud::encode(void)
{
	encodeSideBand(is_CPU, ivec2(0, 1));
}

void ophPointCloud::genCghPointCloudCPU(uint diff_flag)
{
	// Output Image Size
	ivec2 pn;
	pn[_X] = context_.pixel_number[_X];
	pn[_Y] = context_.pixel_number[_Y];

	// Tilt Angle
	Real thetaX = RADIAN(pc_config_.tilt_angle[_X]);
	Real thetaY = RADIAN(pc_config_.tilt_angle[_Y]);

	// Wave Number
	Real k = context_.k;

	// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
	vec2 pp;
	pp[_X] = context_.pixel_pitch[_X];
	pp[_Y] = context_.pixel_pitch[_Y];

	// Length (Width) of complex field at eyepiece plane (by simple magnification)
	vec2 ss;
	ss[_X] = context_.ss[_X];
	ss[_Y] = context_.ss[_Y];

	Complex<Real> lambda(1, context_.lambda);

	int j; // private variable for Multi Threading
#ifdef _OPENMP
	int num_threads = 0;
#pragma omp parallel
{
	num_threads = omp_get_num_threads(); // get number of Multi Threading
#pragma omp for private(j)
#endif
	for (j = 0; j < n_points; ++j) { //Create Fringe Pattern
		uint idx = 3 * j;
		uint color_idx = pc_data_.n_colors * j;
		Real pcx = pc_data_.vertex[idx + _X] * pc_config_.scale[_X];
		Real pcy = pc_data_.vertex[idx + _Y] * pc_config_.scale[_Y];
		Real pcz = pc_data_.vertex[idx + _Z] * pc_config_.scale[_Z] + pc_config_.offset_depth;
		Real amplitude = pc_data_.color[color_idx];

		switch (diff_flag)
		{
		case PC_DIFF_RS_ENCODED:
			diffractEncodedRS_CPU(pn, pp, ss, vec3(pcx, pcy, pcz), k, amplitude, vec2(thetaX, thetaY));
			break;
		case PC_DIFF_RS_NOT_ENCODED:
			diffractNotEncodedRS_CPU(pn, pp, ss, vec3(pcx, pcy, pcz), k, amplitude, lambda);
			break;
		case PC_DIFF_FRESNEL_ENCODED:
			diffractEncodedFrsn_CPU();
			break;
		case PC_DIFF_FRESNEL_NOT_ENCODED:
			diffractNotEncodedFrsn_CPU();
			break;
		}
	}
#ifdef _OPENMP
	}
	std::cout << ">>> All " << num_threads << " threads" << std::endl;
#endif
}

void ophPointCloud::diffractEncodedRS_CPU(ivec2 pn, vec2 pp, vec2 ss, vec3 vertex, Real k, Real amplitude, vec2 theta)
{
	for (int row = 0; row < pn[_Y]; ++row) {
		// Y coordinate of the current pixel : Note that pcy index is reversed order
		Real SLM_y = (ss[_Y] / 2) - ((Real)row + 0.5f) * pp[_Y];

		for (int col = 0; col < pn[_X]; ++col) {
			// X coordinate of the current pixel
			Real SLM_x = ((Real)col + 0.5) * pp[_X] - (ss[_X] / 2);

			Real r = sqrt((SLM_x - vertex[_X])*(SLM_x - vertex[_X]) + (SLM_y - vertex[_Y])*(SLM_y - vertex[_Y]) + vertex[_Z] * vertex[_Z]);
			Real phi = k * r - k * SLM_x*sin(theta[_X]) - k * SLM_y*sin(theta[_Y]); // Phase for printer
			Real result = amplitude * cos(phi);

			holo_encoded[col + row * pn[_X]] += result; //R-S Integral
		}
	}
}

void ophPointCloud::diffractNotEncodedRS_CPU(ivec2 pn, vec2 pp, vec2 ss, vec3 vertex, Real k, Real amplitude, Complex<Real> lambda)
{
	Real tx = context_.lambda / (2 * pp[_X]);
	Real ty = context_.lambda / (2 * pp[_Y]);

	Real xbound[2] = { vertex[_X] + abs(tx / sqrt(1 - pow(tx, 2)) * vertex[_Z]), vertex[_X] - abs(tx / sqrt(1 - pow(tx, 2)) * vertex[_Z]) };
	Real ybound[2] = { vertex[_Y] + abs(ty / sqrt(1 - pow(ty, 2)) * vertex[_Z]), vertex[_Y] - abs(ty / sqrt(1 - pow(ty, 2)) * vertex[_Z]) };

	Real Xbound[2] = { floor((xbound[0] + ss[_X] / 2) / pp[_X]) + 1, floor((xbound[1] + ss[_X] / 2) / pp[_X]) + 1 };
	Real Ybound[2] = { pn[_Y] - floor((ybound[1] + ss[_Y] / 2) / pp[_Y]), pn[_Y] - floor((ybound[0] + ss[_Y] / 2) / pp[_Y]) };

	if (Xbound[0] > pn[_X])	Xbound[0] = pn[_X];
	if (Xbound[1] < 0)		Xbound[1] = 0;
	if (Ybound[0] > pn[_Y]) Ybound[0] = pn[_Y];
	if (Ybound[1] < 0)		Ybound[1] = 0;

	for (int xxtr = Xbound[1]; xxtr < Xbound[0]; xxtr++)
	{
		for (int yytr = Ybound[1]; yytr < Ybound[0]; yytr++)
		{
			Real xxx = (-ss[_X]) / 2 + (xxtr - 1) * pp[_X];
			Real yyy = (-ss[_Y]) / 2 + (pn[_Y] - yytr) * pp[_Y];
			Real r = sqrt(pow(xxx - vertex[_X], 2) + pow(yyy - vertex[_Y], 2) + pow(vertex[_Z], 2));

			Real range_x[2] = {
				vertex[_X] + abs(tx / sqrt(1 - pow(tx, 2)) * sqrt(pow(yyy - vertex[_Y], 2) + pow(vertex[_Z], 2))),
				vertex[_X] - abs(tx / sqrt(1 - pow(tx, 2)) * sqrt(pow(yyy - vertex[_Y], 2) + pow(vertex[_Z], 2)))
			};

			Real range_y[2] = {
				vertex[_Y] + abs(ty / sqrt(1 - pow(ty, 2)) * sqrt(pow(xxx - vertex[_X], 2) + pow(vertex[_Z], 2))),
				vertex[_X] - abs(ty / sqrt(1 - pow(ty, 2)) * sqrt(pow(xxx - vertex[_X], 2) + pow(vertex[_Z], 2)))
			};

			Complex<Real> kr(1, k * r);

			if ((xxx < range_x[0] && xxx > range_x[1]) && (yyy < range_y[0] && yyy > range_y[1])) {
				auto res = (amplitude * (-vertex[_Z]) / lambda) * (kr.exp() / pow(r, 2));

				holo_gen[xxtr + yytr * pn[_X]][_RE] = res[_RE];
				holo_gen[xxtr + yytr * pn[_X]][_IM] = res[_IM];

				LOG("(%3d, %3d) [%7d] : ", xxtr, yytr, xxtr + yytr * pn[_X]);
				LOG("holo=(%15.5lf + %20.10lf * i )\n", holo_gen[xxtr + yytr * pn[_X]][_RE], holo_gen[xxtr + yytr * pn[_X]][_IM]);
			}
		}
	}
}

void ophPointCloud::diffractEncodedFrsn_CPU(void)
{
}

void ophPointCloud::diffractNotEncodedFrsn_CPU(void)
{
}

void ophPointCloud::ophFree(void)
{
	delete[] pc_data_.vertex;
	delete[] pc_data_.color;
	delete[] pc_data_.phase;
}