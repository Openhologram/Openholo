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

void ophPointCloud::setMode(bool IsCPU)
{
	IsCPU_ = IsCPU;
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

	return true;
}

real ophPointCloud::generateHologram()
{
	auto start_time = _cur_time;

	// Output Image Size
	int n_x = context_.pixel_number[_X];
	int n_y = context_.pixel_number[_Y];

	// Memory Location for Result Image
	if (holo_gen != nullptr) delete[] holo_gen;
	holo_gen = new oph::Complex<real>[n_x * n_y];

	if (holo_encoded != nullptr) delete[] holo_encoded;
	holo_encoded = new real[n_x * n_y];

	if (holo_normalized != nullptr) delete[] holo_normalized;
	holo_normalized = new uchar[n_x * n_y];

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

	return ((std::chrono::duration<real>)(end_time - start_time)).count();
}

void ophPointCloud::genCghPointCloud(real* dst)
{
	// Output Image Size
	int n_x = context_.pixel_number[_X];
	int n_y = context_.pixel_number[_Y];

	// Tilt Angle
	real thetaX = RADIAN(pc_config_.tilt_angle[_X]);
	real thetaY = RADIAN(pc_config_.tilt_angle[_Y]);

	// Wave Number
	real k = context_.k;

	// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
	real pixel_x = context_.pixel_pitch.v[0];
	real pixel_y = context_.pixel_pitch.v[1];

	// Length (Width) of complex field at eyepiece plane (by simple magnification)
	real Length_x = pixel_x * n_x;
	real Length_y = pixel_y * n_y;

	int j; // private variable for Multi Threading
#ifdef _OPENMP
	int num_threads = 0;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads(); // get number of Multi Threading
#pragma omp for private(j)
#endif
		for (j = 0; j < n_points; ++j) { //Create Fringe Pattern
			real x = pc_data_.location[j][_X] * pc_config_.scale[_X];
			real y = pc_data_.location[j][_Y] * pc_config_.scale[_Y];
			real z = pc_data_.location[j][_Z] * pc_config_.scale[_Z] + pc_config_.offset_depth;
			real amplitude = pc_data_.amplitude[j];

			for (int row = 0; row < n_y; ++row) {
				// Y coordinate of the current pixel : Note that y index is reversed order
				real SLM_y = (Length_y / 2) - ((real)row + 0.5f) * pixel_y;

				for (int col = 0; col < n_x; ++col) {
					// X coordinate of the current pixel
					real SLM_x = ((real)col + 0.5f) * pixel_x - (Length_x / 2);

					real r = sqrtf((SLM_x - x)*(SLM_x - x) + (SLM_y - y)*(SLM_y - y) + z * z);
					real phi = k * r - k * SLM_x*sinf(thetaX) - k * SLM_y*sinf(thetaY); // Phase for printer
					real result = amplitude * cosf(phi);

					*(dst + col + row * n_x) += result; //R-S Integral
				}
			}
		}
#ifdef _OPENMP
	}
	std::cout << ">>> All " << num_threads << " threads" << std::endl;
#endif
}

void ophPointCloud::genCghPointCloud_cuda(real* dst)
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
	const ulonglong bufferSize = context_.pixel_number.v[0] * context_.pixel_number.v[1] * sizeof(real);

	//Host Memory Location
	float3 *HostPointCloud = (float3*)pc_data_.location;
	real *hostAmplitude = (real*)pc_data_.amplitude;

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
		real thetaX = RADIAN(pc_config_.tilt_angle.v[0]);
		real thetaY = RADIAN(pc_config_.tilt_angle.v[1]);
		HostConfig.sin_thetaX = sinf(thetaX);
		HostConfig.sin_thetaY = sinf(thetaY);

		// Wave Number
		HostConfig.k = (2.f * CUDART_PI_F) / context_.lambda;

		// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
		HostConfig.pixel_x = context_.pixel_pitch.v[0];
		HostConfig.pixel_y = context_.pixel_pitch.v[1];

		// Length (Width) of complex field at eyepiece plane (by simple magnification)
		real Length_x = HostConfig.pixel_x * HostConfig.Nx;
		real Length_y = HostConfig.pixel_y * HostConfig.Ny;
		HostConfig.halfLength_x = Length_x / 2.f;
		HostConfig.halfLength_y = Length_y / 2.f;
	}

	//Device(GPU) Memory Location
	float3 *DevicePointCloud;
	cudaMalloc((void**)&DevicePointCloud, n_points * 3 * sizeof(real));
	cudaMemcpy(DevicePointCloud, HostPointCloud, n_points * 3 * sizeof(real), cudaMemcpyHostToDevice);

	real *deviceAmplitude;
	cudaMalloc((void**)&deviceAmplitude, n_points * sizeof(real));
	cudaMemcpy(deviceAmplitude, hostAmplitude, n_points * sizeof(real), cudaMemcpyHostToDevice);

	GpuConst *DeviceConfig;
	cudaMalloc((void**)&DeviceConfig, sizeof(GpuConst));
	cudaMemcpy(DeviceConfig, &HostConfig, sizeof(HostConfig), cudaMemcpyHostToDevice);

	real *deviceDst;
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
