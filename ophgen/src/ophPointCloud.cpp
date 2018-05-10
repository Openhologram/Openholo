#include "ophPointCloud.h"

ophPointCloud::ophPointCloud(void)
{
	setMode(false);
	n_points = -1;
}

ophPointCloud::ophPointCloud(const std::string pc_file, const std::string cfg_file)
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

int ophPointCloud::loadPointCloud(const std::string pc_file)
{
	n_points = ophGen::loadPointCloud(pc_file, &vertex_array_, &amplitude_array_, &phase_array_);

	return n_points;
}

bool ophPointCloud::readConfig(const std::string cfg_file)
{
	if (!ophGen::readConfig(cfg_file, pc_config_))
		return false;

	return true;
}

void ophPointCloud::setPointCloudModel(const std::vector<real> &vertex_array, const std::vector<real> &amplitude_array, const std::vector<real> &phase_array)
{
	vertex_array_ = vertex_array;
	amplitude_array_ = amplitude_array;
	phase_array_ = phase_array;
}

void ophPointCloud::getPointCloudModel(std::vector<real> &vertex_array, std::vector<real> &amplitude_array, std::vector<real> &phase_array)
{
	getModelVertexArray(vertex_array);
	getModelAmplitudeArray(amplitude_array);
	getModelPhaseArray(phase_array);
}

void ophPointCloud::getModelVertexArray(std::vector<real>& vertex_array)
{
	vertex_array = vertex_array_;
}

void ophPointCloud::getModelAmplitudeArray(std::vector<real>& amplitude_array)
{
	amplitude_array = amplitude_array_;
}

void ophPointCloud::getModelPhaseArray(std::vector<real>& phase_array)
{
	phase_array = phase_array_;
}

int ophPointCloud::getNumberOfPoints()
{
	return n_points;
}

real ophPointCloud::generateHologram()
{
	// Output Image Size
	int n_x = context_.pixel_number.v[0];
	int n_y = context_.pixel_number.v[1];

	// Memory Location for Result Image
	//if (data_hologram != nullptr) free(data_hologram);
	if (p_hologram != nullptr) free(p_hologram);
	p_hologram = (uchar*)calloc(1, sizeof(uchar)*n_x*n_y);
	real *data_fringe = (real*)calloc(1, sizeof(real)*n_x*n_y);

	// Create CGH Fringe Pattern by 3D Point Cloud
	real time = 0.0;
	if (IsCPU_ == true) { //Run CPU
#ifdef _OPENMP
		std::cout << "Generate Hologram with Multi Core CPU" << std::endl;
#else
		std::cout << "Generate Hologram with Single Core CPU" << std::endl;
#endif
		time = genCghPointCloud(data_fringe);
	}
	else { //Run GPU
		std::cout << "Generate Hologram with GPU" << std::endl;

		time = genCghPointCloud_cuda(data_fringe);
		std::cout << ">>> CUDA GPGPU" << std::endl;
	}

	// Normalization data_fringe to data_hologram
	oph::normalize(data_fringe, (uchar*)p_hologram, n_x, n_y);

	free(data_fringe);
	return time;
}

real ophPointCloud::genCghPointCloud(real * dst)
{
	// Output Image Size
	int n_x = context_.pixel_number.v[0];
	int n_y = context_.pixel_number.v[1];

	// Tilt Angle
	real thetaX = RADIAN(pc_config_.tilt_angle.v[0]);
	real thetaY = RADIAN(pc_config_.tilt_angle.v[1]);

	// Wave Number
	real k = context_.k;

	// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
	real pixel_x = context_.pixel_pitch.v[0];
	real pixel_y = context_.pixel_pitch.v[1];

	// Length (Width) of complex field at eyepiece plane (by simple magnification)
	real Length_x = pixel_x * n_x;
	real Length_y = pixel_y * n_y;

	std::chrono::system_clock::time_point time_start = std::chrono::system_clock::now();
	int j; // private variable for Multi Threading
#ifdef _OPENMP
	int num_threads = 0;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads(); // get number of Multi Threading
#pragma omp for private(j)
#endif
		for (j = 0; j < n_points; ++j) { //Create Fringe Pattern
			real x = vertex_array_[3 * j + 0] * pc_config_.scale.v[0];
			real y = vertex_array_[3 * j + 1] * pc_config_.scale.v[1];
			real z = vertex_array_[3 * j + 2] * pc_config_.scale.v[2] + pc_config_.offset_depth;
			real amplitude = amplitude_array_[j];

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
	std::chrono::system_clock::time_point time_finish = std::chrono::system_clock::now();
	return ((std::chrono::duration<real>)(time_finish - time_start)).count();
}

real ophPointCloud::genCghPointCloud_cuda(real * dst)
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
	float3 *HostPointCloud = (float3*)vertex_array_.data();
	real *hostAmplitude = (real*)amplitude_array_.data();

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
	cudaMalloc((void**)&DevicePointCloud, vertex_array_.size() * sizeof(real));
	cudaMemcpy(DevicePointCloud, HostPointCloud, vertex_array_.size() * sizeof(real), cudaMemcpyHostToDevice);

	real *deviceAmplitude;
	cudaMalloc((void**)&deviceAmplitude, amplitude_array_.size() * sizeof(real));
	cudaMemcpy(deviceAmplitude, hostAmplitude, amplitude_array_.size() * sizeof(real), cudaMemcpyHostToDevice);

	GpuConst *DeviceConfig;
	cudaMalloc((void**)&DeviceConfig, sizeof(GpuConst));
	cudaMemcpy(DeviceConfig, &HostConfig, sizeof(HostConfig), cudaMemcpyHostToDevice);

	real *deviceDst;
	cudaMalloc((void**)&deviceDst, bufferSize);

	std::chrono::system_clock::time_point time_start = std::chrono::system_clock::now();
	{
		cudaPointCloudKernel(block_x, block_y, THREAD_X, THREAD_Y, DevicePointCloud, deviceAmplitude, DeviceConfig, deviceDst);
		cudaMemcpy(dst, deviceDst, bufferSize, cudaMemcpyDeviceToHost);
	}
	std::chrono::system_clock::time_point time_finish = std::chrono::system_clock::now();

	//Device(GPU) Memory Delete
	cudaFree(DevicePointCloud);
	cudaFree(deviceAmplitude);
	cudaFree(deviceDst);
	cudaFree(DeviceConfig);
	return ((std::chrono::duration<real>)(time_finish - time_start)).count();
}

void ophPointCloud::ophFree(void)
{
	vertex_array_.clear();
	amplitude_array_.clear();
	phase_array_.clear();
}
