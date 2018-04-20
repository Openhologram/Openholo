#include "ophPointCloud.h"

ophPointCloud::ophPointCloud(void)
{
	setMode(false);
	this->InputSrcFile = "";
	this->InputConfigFile = "";
	this->n_points = -1;
	this->data_hologram = nullptr;
}

ophPointCloud::ophPointCloud(const std::string InputModelFile, const std::string InputConfigFile)
{
	setMode(false);
	this->InputSrcFile = InputModelFile;
	this->n_points = loadPointCloud(InputModelFile);
	if (n_points == -1) std::cerr << "OpenHolo Error : Failed to load Point Cloud Data File(*.dat)" << std::endl;

	this->InputConfigFile = InputConfigFile;
	bool ok = readConfig(InputConfigFile);
	if (!ok) std::cerr << "OpenHolo Error : Failed to load Config Specification Data File(*.config)" << std::endl;

	this->data_hologram = nullptr;
}

ophPointCloud::~ophPointCloud(void)
{
}

void ophPointCloud::setMode(bool isCPU)
{
	this->bIsCPU = isCPU;
}

int ophPointCloud::loadPointCloud(const std::string InputModelFile)
{
	//std::ifstream File(InputModelFile, std::ios::in);
	//if (!File.is_open()) {
	//	File.close();
	//	return -1;
	//}

	//std::string Line;
	//std::getline(File, Line);
	//int n_pts = atoi(Line.c_str());
	//this->n_points = n_pts;

	//// parse input point cloud file
	//for (int i = 0; i < n_pts; ++i) {
	//	int idx;
	//	float pX, pY, pZ, phase, amplitude;
	//	std::getline(File, Line);
	//	sscanf(Line.c_str(), "%d %f %f %f %f %f\n", &idx, &pX, &pY, &pZ, &phase, &amplitude);

	//	if (idx == i) {
	//		this->VertexArray.push_back(pX);
	//		this->VertexArray.push_back(pY);
	//		this->VertexArray.push_back(pZ);
	//		this->PhaseArray.push_back(phase);
	//		this->AmplitudeArray.push_back(amplitude);
	//		//this->ModelData.push_back(PointCloud(pX, pY, pZ, amplitude, phase));
	//	}
	//	else {
	//		File.close();
	//		return -1;
	//	}
	//}
	//File.close();
	this->InputSrcFile = InputModelFile;
	this->n_points = ophGen::loadPointCloudData(InputModelFile, &this->VertexArray, &this->AmplitudeArray, &this->PhaseArray);

	return this->n_points;
}

bool ophPointCloud::readConfig(const std::string InputConfigFile)
{
	if (!ophGen::readConfigFile(InputConfigFile, this->ConfigParams))
		return false;

	this->InputConfigFile = InputConfigFile;
	return true;
}

void ophPointCloud::setPointCloudModel(const std::vector<float>& VertexArray, const std::vector<float>& AmplitudeArray, const std::vector<float>& PhaseArray)
{
	this->VertexArray = VertexArray;
	this->AmplitudeArray = AmplitudeArray;
	this->PhaseArray = PhaseArray;
}

void ophPointCloud::getPointCloudModel(std::vector<float>& VertexArray, std::vector<float>& AmplitudeArray, std::vector<float>& PhaseArray)
{
	getModelVertexArray(VertexArray);
	getModelAmplitudeArray(AmplitudeArray);
	getModelPhaseArray(PhaseArray);
}

void ophPointCloud::getModelVertexArray(std::vector<float>& VertexArray)
{
	VertexArray = this->VertexArray;
}

void ophPointCloud::getModelAmplitudeArray(std::vector<float>& AmplitudeArray)
{
	AmplitudeArray = this->AmplitudeArray;
}

void ophPointCloud::getModelPhaseArray(std::vector<float>& PhaseArray)
{
	PhaseArray = this->PhaseArray;
}

int ophPointCloud::getNumberOfPoints()
{
	return this->n_points;
}

uchar * ophPointCloud::getHologramBufferData()
{
	return this->data_hologram;
}

void ophPointCloud::setConfigParams(const oph::ConfigParams & InputConfig)
{
	this->ConfigParams = InputConfig;
}

oph::ConfigParams ophPointCloud::getConfigParams()
{
	return this->ConfigParams;
}

double ophPointCloud::generateHologram()
{
	// Output Image Size
	int n_x = ConfigParams.nx;
	int n_y = ConfigParams.ny;

	// Memory Location for Result Image
	if (this->data_hologram != nullptr) free(data_hologram);
	this->data_hologram = (uchar*)calloc(1, sizeof(uchar)*n_x*n_y);
	float *data_fringe = (float*)calloc(1, sizeof(float)*n_x*n_y);

	// Create CGH Fringe Pattern by 3D Point Cloud
	double time = 0.0;
	if (this->bIsCPU == true) { //Run CPU
#ifdef _OPENMP
		std::cout << "Generate Hologram with Multi Core CPU" << std::endl;
#else
		std::cout << "Generate Hologram with Single Core CPU" << std::endl;
#endif
		time = genCghPointCloud(this->VertexArray, this->AmplitudeArray, data_fringe);
	}
	else { //Run GPU
		std::cout << "Generate Hologram with GPU" << std::endl;

		time = genCghPointCloud_cuda(this->VertexArray, this->AmplitudeArray, data_fringe);
		std::cout << ">>> CUDA GPGPU" << std::endl;
	}

	// Normalization data_fringe to data_hologram
	normalize(data_fringe, this->data_hologram, n_x, n_y);
	free(data_fringe);
	return time;
}

void ophPointCloud::saveFileBmp(std::string OutputFileName)
{
	Openholo::createBitmapFile(this->data_hologram,
		this->ConfigParams.nx, 
		this->ConfigParams.ny, 
		OPH_Bitsperpixel, 
		OutputFileName.c_str());

	//OutputFileName.append(".bmp");
	//FILE *fp = fopen(OutputFileName.c_str(), "wb");
	//bitmap *pbitmap = (bitmap*)calloc(1, sizeof(bitmap));
	//memset(pbitmap, 0x00, sizeof(bitmap));

	//// File Header
	//pbitmap->fileheader.signature[0] = 'B';
	//pbitmap->fileheader.signature[1] = 'M';
	//pbitmap->fileheader.filesize = _filesize;
	//pbitmap->fileheader.fileoffset_to_pixelarray = sizeof(bitmap);

	//// Initialize pallets : to Grayscale
	//for (int i = 0; i < 256; i++) {
	//	pbitmap->rgbquad[i].rgbBlue = i;
	//	pbitmap->rgbquad[i].rgbGreen = i;
	//	pbitmap->rgbquad[i].rgbRed = i;
	//}

	//// Image Header
	//pbitmap->bitmapinfoheader.dibheadersize = sizeof(bitmapinfoheader);
	//pbitmap->bitmapinfoheader.width = _width;
	//pbitmap->bitmapinfoheader.height = _height;
	//pbitmap->bitmapinfoheader.planes = OPH_Planes;
	//pbitmap->bitmapinfoheader.bitsperpixel = OPH_Bitsperpixel;
	//pbitmap->bitmapinfoheader.compression = OPH_Compression;
	//pbitmap->bitmapinfoheader.imagesize = _pixelbytesize;
	//pbitmap->bitmapinfoheader.ypixelpermeter = OPH_Ypixelpermeter;
	//pbitmap->bitmapinfoheader.xpixelpermeter = OPH_Xpixelpermeter;
	//pbitmap->bitmapinfoheader.numcolorspallette = 256;
	//fwrite(pbitmap, 1, sizeof(bitmap), fp);

	//fwrite(this->data_hologram, 1, _pixelbytesize, fp);
	//fclose(fp);
	//ophFree(pbitmap);
}

void ophPointCloud::setScaleFactor(const float scaleX, const float scaleY, const float scaleZ)
{
	this->ConfigParams.pointCloudScaleX = scaleX;
	this->ConfigParams.pointCloudScaleY = scaleY;
	this->ConfigParams.pointCloudScaleZ = scaleZ;
}

void ophPointCloud::getScaleFactor(float & scaleX, float & scaleY, float & scaleZ)
{
	scaleX = this->ConfigParams.pointCloudScaleX;
	scaleY = this->ConfigParams.pointCloudScaleY;
	scaleZ = this->ConfigParams.pointCloudScaleZ;
}

void ophPointCloud::setOffsetDepth(const float offsetDepth)
{
	this->ConfigParams.offsetDepth = offsetDepth;
}

float ophPointCloud::getOffsetDepth()
{
	return this->ConfigParams.offsetDepth;
}

void ophPointCloud::setSamplingPitch(const float pitchX, const float pitchY)
{
	this->ConfigParams.samplingPitchX = pitchX;
	this->ConfigParams.samplingPitchY = pitchY;
}

void ophPointCloud::getSamplingPitch(float & pitchX, float & pitchY)
{
	pitchX = this->ConfigParams.samplingPitchX;
	pitchY = this->ConfigParams.samplingPitchY;
}

void ophPointCloud::setImageSize(const int n_x, const int n_y)
{
	this->ConfigParams.nx = n_x;
	this->ConfigParams.ny = n_y;
}

void ophPointCloud::getImageSize(int & n_x, int & n_y)
{
	n_x = this->ConfigParams.nx;
	n_y = this->ConfigParams.ny;
}

void ophPointCloud::setWaveLength(const float lambda)
{
	this->ConfigParams.lambda = lambda;
}

float ophPointCloud::getWaveLength()
{
	return this->ConfigParams.lambda;
}

void ophPointCloud::setTiltAngle(const float tiltAngleX, const float tiltAngleY)
{
	this->ConfigParams.tiltAngleX = tiltAngleX;
	this->ConfigParams.tiltAngleY = tiltAngleY;
}

void ophPointCloud::getTiltAngle(float & tiltAngleX, float & tiltAngleY)
{
	tiltAngleX = this->ConfigParams.tiltAngleX;
	tiltAngleY = this->ConfigParams.tiltAngleY;
}

double ophPointCloud::genCghPointCloud(const std::vector<float>& VertexArray, const std::vector<float>& AmplitudeArray, float * dst)
{
	// Output Image Size
	int n_x = ConfigParams.nx;
	int n_y = ConfigParams.ny;

	// Tilt Angle
	float thetaX = RADIAN_F(ConfigParams.tiltAngleX);
	float thetaY = RADIAN_F(ConfigParams.tiltAngleY);

	// Wave Number
	float k = (2.f * (float)M_PI) / ConfigParams.lambda;

	// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
	float pixel_x = ConfigParams.samplingPitchX;
	float pixel_y = ConfigParams.samplingPitchY;

	// Length (Width) of complex field at eyepiece plane (by simple magnification)
	float Length_x = pixel_x * n_x;
	float Length_y = pixel_y * n_y;

	std::chrono::system_clock::time_point time_start = std::chrono::system_clock::now();
	int j; // private variable for Multi Threading
#ifdef _OPENMP
	int num_threads = 0;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads(); // get number of Multi Threading
#pragma omp for private(j)
#endif
		for (j = 0; j < this->n_points; ++j) { //Create Fringe Pattern
			float x = VertexArray[3 * j + 0] * ConfigParams.pointCloudScaleX;
			float y = VertexArray[3 * j + 1] * ConfigParams.pointCloudScaleY;
			float z = VertexArray[3 * j + 2] * ConfigParams.pointCloudScaleZ + ConfigParams.offsetDepth;
			float amplitude = AmplitudeArray[j];

			for (int row = 0; row < n_y; ++row) {
				// Y coordinate of the current pixel : Note that y index is reversed order
				float SLM_y = (Length_y / 2) - ((float)row + 0.5f) * pixel_y;

				for (int col = 0; col < n_x; ++col) {
					// X coordinate of the current pixel
					float SLM_x = ((float)col + 0.5f) * pixel_x - (Length_x / 2);

					float r = sqrtf((SLM_x - x)*(SLM_x - x) + (SLM_y - y)*(SLM_y - y) + z * z);
					float phi = k * r - k * SLM_x*sinf(thetaX) - k * SLM_y*sinf(thetaY); // Phase for printer
					float result = amplitude * cosf(phi);

					*(dst + col + row * n_x) += result; //R-S Integral
				}
			}
		}
#ifdef _OPENMP
	}
	std::cout << ">>> All " << num_threads << " threads" << std::endl;
#endif
	std::chrono::system_clock::time_point time_finish = std::chrono::system_clock::now();
	return ((std::chrono::duration<double>)(time_finish - time_start)).count();
}

double ophPointCloud::genCghPointCloud_cuda(const std::vector<float>& VertexArray, const std::vector<float>& AmplitudeArray, float * dst)
{
	int _bx = ConfigParams.nx / THREAD_X;
	int _by = ConfigParams.ny / THREAD_Y;
	int block_x = 2;
	int block_y = 2;

	//blocks number
	while (1) {
		if ((block_x >= _bx) && (block_y >= _by)) break;
		if (block_x < _bx) block_x *= 2;
		if (block_y < _by) block_y *= 2;
	}

	//threads number
	const ulonglong bufferSize = ConfigParams.nx * ConfigParams.ny * sizeof(float);

	//Host Memory Location
	float3 *HostPointCloud = (float3*)VertexArray.data();
	float *hostAmplitude = (float*)AmplitudeArray.data();

	//Initializa Config for CUDA Kernel
	oph::GpuConst HostConfig; {
		HostConfig.n_points = this->n_points;
		HostConfig.scaleX = ConfigParams.pointCloudScaleX;
		HostConfig.scaleY = ConfigParams.pointCloudScaleY;
		HostConfig.scaleZ = ConfigParams.pointCloudScaleZ;
		HostConfig.offsetDepth = ConfigParams.offsetDepth;

		// Output Image Size
		HostConfig.Nx = ConfigParams.nx;
		HostConfig.Ny = ConfigParams.ny;

		// Tilt Angle
		float thetaX = RADIAN_F(ConfigParams.tiltAngleX);
		float thetaY = RADIAN_F(ConfigParams.tiltAngleY);
		HostConfig.sin_thetaX = sinf(thetaX);
		HostConfig.sin_thetaY = sinf(thetaY);

		// Wave Number
		HostConfig.k = (2.f * CUDART_PI_F) / ConfigParams.lambda;

		// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
		HostConfig.pixel_x = ConfigParams.samplingPitchX;
		HostConfig.pixel_y = ConfigParams.samplingPitchY;

		// Length (Width) of complex field at eyepiece plane (by simple magnification)
		float Length_x = HostConfig.pixel_x * HostConfig.Nx;
		float Length_y = HostConfig.pixel_y * HostConfig.Ny;
		HostConfig.halfLength_x = Length_x / 2.f;
		HostConfig.halfLength_y = Length_y / 2.f;
	}

	//Device(GPU) Memory Location
	float3 *DevicePointCloud;
	cudaMalloc((void**)&DevicePointCloud, VertexArray.size() * sizeof(float));
	cudaMemcpy(DevicePointCloud, HostPointCloud, VertexArray.size() * sizeof(float), cudaMemcpyHostToDevice);

	float *deviceAmplitude;
	cudaMalloc((void**)&deviceAmplitude, AmplitudeArray.size() * sizeof(float));
	cudaMemcpy(deviceAmplitude, hostAmplitude, AmplitudeArray.size() * sizeof(float), cudaMemcpyHostToDevice);

	GpuConst *DeviceConfig;
	cudaMalloc((void**)&DeviceConfig, sizeof(GpuConst));
	cudaMemcpy(DeviceConfig, &HostConfig, sizeof(HostConfig), cudaMemcpyHostToDevice);

	float *deviceDst;
	cudaMalloc((void**)&deviceDst, bufferSize);

	std::chrono::system_clock::time_point time_start = std::chrono::system_clock::now();
	{
		//cudaPointCloudKernel(block_x, block_y, THREAD_X, THREAD_Y, DevicePointCloud, deviceAmplitude, DeviceConfig, deviceDst);
		cudaMemcpy(dst, deviceDst, bufferSize, cudaMemcpyDeviceToHost);
	}
	std::chrono::system_clock::time_point time_finish = std::chrono::system_clock::now();

	//Device(GPU) Memory Delete
	cudaFree(DevicePointCloud);
	cudaFree(deviceAmplitude);
	cudaFree(deviceDst);
	cudaFree(DeviceConfig);
	return ((std::chrono::duration<double>)(time_finish - time_start)).count();
}

void ophPointCloud::normalize(float *src, uchar *dst, const int nx, const int ny) {
	float minVal, maxVal;
	for (int ydx = 0; ydx < ny; ydx++) {
		for (int xdx = 0; xdx < nx; xdx++) {
			float *temp_pos = src + xdx + ydx * nx;
			if ((xdx == 0) && (ydx == 0)) {
				minVal = *(temp_pos);
				maxVal = *(temp_pos);
			}
			else {
				if (*(temp_pos) < minVal) minVal = *(temp_pos);
				if (*(temp_pos) > maxVal) maxVal = *(temp_pos);
			}
		}
	}

	for (int ydx = 0; ydx < ny; ydx++) {
		for (int xdx = 0; xdx < nx; xdx++) {
			float *src_pos = src + xdx + ydx * nx;
			uchar *res_pos = dst + xdx + (ny - ydx - 1)*nx;	// Flip image vertically to consider flipping by Fourier transform and projection geometry

			*(res_pos) = (uchar)(((*(src_pos)-minVal) / (maxVal - minVal)) * 255 + 0.5);
		}
	}
}

void ophPointCloud::ophFree(void)
{
}
