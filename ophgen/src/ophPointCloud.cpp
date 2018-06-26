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

void ophPointCloud::setMode(bool IsCPU)
{
	IsCPU_ = IsCPU;
}

int ophPointCloud::loadPointCloud(const char* pc_file, uint flag)
{
	n_points = ophGen::loadPointCloud(pc_file, &pc_data_, flag);

	return n_points;
}

bool ophPointCloud::readConfig(const char* cfg_file)
{
	if (!ophGen::readConfig(cfg_file, pc_config_))
		return false;

	initialize();

	return true;
}

Real ophPointCloud::generateHologram()
{
	auto start_time = _cur_time;

	// Create CGH Fringe Pattern by 3D Point Cloud
	if (IsCPU_ == true) { //Run CPU
#ifdef _OPENMP
		std::cout << "Generate Hologram with Multi Core CPU" << std::endl;
#else
		std::cout << "Generate Hologram with Single Core CPU" << std::endl;
#endif
		genCghPointCloud(holo_encoded); /// 홀로그램 데이터 Complex data로 변경 시 holo_gen으로
	}
	else { //Run GPU
		std::cout << "Generate Hologram with GPU" << std::endl;

		genCghPointCloud_cuda(holo_encoded);
		std::cout << ">>> CUDA GPGPU" << std::endl;
	}

	auto end_time = _cur_time;

	auto during_time = ((std::chrono::duration<Real>)(end_time - start_time)).count();

	LOG("Implement time : %.5lf\n", during_time);

	return during_time;
}

double ophPointCloud::diffract(void)
{
	auto start_time = _cur_time;

	initialize();

	if (IsCPU_) {
#ifdef _OPENMP
		std::cout << "Generate Hologram with Multi Core CPU" << std::endl;
#else
		std::cout << "Generate Hologram with Single Core CPU" << std::endl;
#endif
		genCghPointCloud(holo_encoded);
	}
	else {
		std::cout << "Generate Hologram with GPU" << std::endl;

		genCghPointCloud_cuda(holo_encoded);
		std::cout << ">>> CUDA GPGPU" << std::endl;
	}

	auto end_time = _cur_time;

	auto during_time = ((std::chrono::duration<Real>)(end_time - start_time)).count();

	LOG("Implement time : %.5lf\n", during_time);

	return during_time;
}

void ophPointCloud::encode(void)
{
	encodeSideBand(IsCPU_, ivec2(0, 1));
}

void ophPointCloud::genCghPointCloud(Real* dst)
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

	int j; // private variable for Multi Threading
#ifdef _OPENMP
	int num_threads = 0;
#pragma omp parallel
	{
	num_threads = omp_get_num_threads(); // get number of Multi Threading
#pragma omp for private(j)
#endif
		for (j = 0; j < n_points; ++j) { //Create Fringe Pattern
			Real pcx = pc_data_.location[j][_X] * pc_config_.scale[_X];
			Real pcy = pc_data_.location[j][_Y] * pc_config_.scale[_Y];
			Real pcz = pc_data_.location[j][_Z] * pc_config_.scale[_Z] + pc_config_.offset_depth;
			Real amplitude = pc_data_.amplitude[j];

			for (int row = 0; row < pn[_Y]; ++row) {
				// Y coordinate of the current pixel : Note that pcy index is reversed order
				Real SLM_y = (ss[_Y] / 2) - ((Real)row + 0.5f) * pp[_Y];

				for (int col = 0; col < pn[_X]; ++col) {
					// X coordinate of the current pixel
					Real SLM_x = ((Real)col + 0.5f) * pp[_X] - (ss[_X] / 2);

					Real r = sqrtf((SLM_x - pcx)*(SLM_x - pcx) + (SLM_y - pcy)*(SLM_y - pcy) + pcz * pcz);
					Real phi = k * r - k * SLM_x*sinf(thetaX) - k * SLM_y*sinf(thetaY); // Phase for printer
					Real result = amplitude * cosf(phi);

					*(dst + col + row * pn[_X]) += result; //R-S Integral
				}
			}

			/// <<
			//Real tx = context_.lambda / (2 * pp[_X]);
			//Real ty = context_.lambda / (2 * pp[_Y]);

			//Real xbound[2] = { pcx + abs(tx / sqrt(1 - pow(tx, 2)) * pcz), pcx - abs(tx / sqrt(1 - pow(tx, 2)) * pcz) };
			//Real ybound[2] = { pcy + abs(ty / sqrt(1 - pow(ty, 2)) * pcz), pcy - abs(ty / sqrt(1 - pow(ty, 2)) * pcz) };

			//Real Xbound[2] = { floor((xbound[0] + ss[_X] / 2) / pp[_X]) + 1, floor((xbound[1] + ss[_X] / 2) / pp[_X]) + 1 };
			//Real Ybound[2] = { pn[_Y] - floor((ybound[1] + ss[_Y] / 2) / pp[_Y]), pn[_Y] - floor((ybound[0] + ss[_Y] / 2) / pp[_Y]) };

			//if (Xbound[0] > pn[_X])
			//	Xbound[0] = pn[_X];
			//if (Xbound[1] < 0)
			//	Xbound[1] = 0;
			//if (Ybound[0] > pn[_Y])
			//	Ybound[0] = pn[_Y];
			//if (Ybound[1] < 0)
			//	Ybound[1] = 0;


			//for (int xxtr = Xbound[1]; xxtr < Xbound[0]; xxtr++)
			//{
			//	for (int yytr = Ybound[1]; yytr < Ybound[0]; yytr++)
			//	{
			//		auto xxx = -ss[_X] / 2 + (xxtr - 1) * pp[_X];
			//		auto yyy = -ss[_Y] / 2 + (pn[_Y] - yytr) * pp[_Y];
			//		auto r = sqrt(pow(xxx - pcx, 2) + pow(yyy - pcy, 2) + pow(pcz, 2));

			//		Real range_x[2] = { pcx + abs(tx / sqrt(1 - pow(tx, 2)) * sqrt(pow(yyy - pcy, 2) + pow(pcz, 2))), pcx - abs(tx / sqrt(1 - pow(tx, 2)) * sqrt(pow(yyy - pcy, 2) + pow(pcz, 2))) };
			//		Real range_y[2] = { pcy + abs(ty / sqrt(1 - pow(ty, 2)) * sqrt(pow(xxx - pcx, 2) + pow(pcz, 2))), pcx - abs(ty / sqrt(1 - pow(ty, 2)) * sqrt(pow(xxx - pcx, 2) + pow(pcz, 2))) };

			//		if ((xxx < range_x[0] && xxx > range_x[1]) && (yyy < range_y[0] && yyy > range_y[1]))
			//			*(holo_gen + xxtr + yytr * pn[_X]) += amplitude * -pcz / (context_.lambda/* * j*/) * exp(/*-i **/ k * r) / pow(r, 2);
			//	}
			//}
		}
#ifdef _OPENMP
	}
	std::cout << ">>> All " << num_threads << " threads" << std::endl;
#endif
}

void ophPointCloud::genCghPointCloud_cuda(Real* dst)
{
	int _bx = context_.pixel_number.v[0] / THREAD_X;
	int _by = context_.pixel_number.v[1] / THREAD_Y;
	int block_x = 2;
	int block_y = 2;

	//blocks number
	while (1) {
		if ((block_x >= _bx) && (block_y >= _by)) break;
		if (block_x < _bx) block_x *= 2;
		if (block_y < _by) block_y *= 2;
	}

	//threads number
	const ulonglong bufferSize = context_.pixel_number.v[0] * context_.pixel_number.v[1] * sizeof(Real);

	//Host Memory Location
	float3 *HostPointCloud = (float3*)pc_data_.location;
	Real *hostAmplitude = (Real*)pc_data_.amplitude;

	//Initializa Config for CUDA Kernel
	oph::GpuConst HostConfig; {
		HostConfig.n_points = n_points;
		HostConfig.scaleX = pc_config_.scale.v[0];
		HostConfig.scaleY = pc_config_.scale.v[1];
		HostConfig.scaleZ = pc_config_.scale.v[2];
		HostConfig.offsetDepth = pc_config_.offset_depth;

		// Output Image Size
		HostConfig.Nx = _bx;
		HostConfig.Ny = _by;

		// Tilt Angle
		Real thetaX = RADIAN(pc_config_.tilt_angle.v[0]);
		Real thetaY = RADIAN(pc_config_.tilt_angle.v[1]);
		HostConfig.sin_thetaX = sinf(thetaX);
		HostConfig.sin_thetaY = sinf(thetaY);

		// Wave Number
		HostConfig.k = (2.f * CUDART_PI_F) / context_.lambda;

		// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
		HostConfig.pixel_x = context_.pixel_pitch.v[0];
		HostConfig.pixel_y = context_.pixel_pitch.v[1];

		// Length (Width) of complex field at eyepiece plane (by simple magnification)
		Real Length_x = HostConfig.pixel_x * HostConfig.Nx;
		Real Length_y = HostConfig.pixel_y * HostConfig.Ny;
		HostConfig.halfLength_x = Length_x / 2.f;
		HostConfig.halfLength_y = Length_y / 2.f;
	}

	//Device(GPU) Memory Location
	float3 *DevicePointCloud;
	cudaMalloc((void**)&DevicePointCloud, n_points * 3 * sizeof(Real));
	cudaMemcpy(DevicePointCloud, HostPointCloud, n_points * 3 * sizeof(Real), cudaMemcpyHostToDevice);

	Real *deviceAmplitude;
	cudaMalloc((void**)&deviceAmplitude, n_points * sizeof(Real));
	cudaMemcpy(deviceAmplitude, hostAmplitude, n_points * sizeof(Real), cudaMemcpyHostToDevice);

	GpuConst *DeviceConfig;
	cudaMalloc((void**)&DeviceConfig, sizeof(GpuConst));
	cudaMemcpy(DeviceConfig, &HostConfig, sizeof(HostConfig), cudaMemcpyHostToDevice);

	Real *deviceDst;
	cudaMalloc((void**)&deviceDst, bufferSize);

	{
		cudaPointCloudKernel(block_x, block_y, THREAD_X, THREAD_Y, DevicePointCloud, deviceAmplitude, DeviceConfig, deviceDst);
		cudaMemcpy(dst, deviceDst, bufferSize, cudaMemcpyDeviceToHost);
	}

	//Device(GPU) Memory Delete
	cudaFree(DevicePointCloud);
	cudaFree(deviceAmplitude);
	cudaFree(deviceDst);
	cudaFree(DeviceConfig);
}

void ophPointCloud::ophFree(void)
{
	delete[] pc_data_.location;
	delete[] pc_data_.color;
	delete[] pc_data_.amplitude;
	delete[] pc_data_.phase;
}
