#include "CUDA.h"
#include "define.h"
#include <iostream>
#include <string>
#include <iomanip>

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}} 


CUDA* CUDA::instance = nullptr;
std::mutex CUDA::mtx;

CUDA::CUDA()
	: m_nThread(512)
	, num_gpu(1)
	, cur_gpu(0)
{
	initGPU();
}


CUDA::~CUDA()
{
}


void CUDA::initGPU()
{
	cudaGetDeviceCount(&num_gpu);
	LOG("%d GPU Detected.\n", num_gpu);
	active_gpus = num_gpu;
	cudaDeviceProp devProp;

	for (int i = 0; i < num_gpu && i < MAX_GPU; i++)
	{
		int devID;
		cudaError_t res = cudaGetDevice(&devID);
		if (res == cudaSuccess) {
			cudaSetDevice(i);
			cur_gpu = i;
			HANDLE_ERROR(cudaGetDeviceProperties(&devProp, cur_gpu));
			devProps[i] = devProp;
			int smPerCore = getSMPerCore(devProp.major, devProp.minor);
			cuda_cores[i] = smPerCore * devProp.multiProcessorCount;
		}
		else {
			LOG("<FAILED> cudaGetDevice(%d)\n", i);
		}
	}
	printDevInfo();
}

bool CUDA::printDevInfo()
{
	cudaDeviceProp devProp;

	for (int i = 0; i < num_gpu && i < MAX_GPU; i++)
	{
		devProp = devProps[i];
		LOG("%d] GPU Spec : %s\n", i, devProp.name);
#ifdef _WIN64
		LOG(" - Global Memory : %llu\n", devProp.totalGlobalMem);
		LOG(" - Const Memory : %llu\n", devProp.totalConstMem);
#else
		LOG(" - Global Memory : %zu\n", devProp.totalGlobalMem);
		LOG(" - Const Memory : %zu\n", devProp.totalConstMem);
#endif
		LOG("  - MP(Multiprocessor) Count : %d\n", devProp.multiProcessorCount);
		LOG("  - Total MP Count: %d\n", cuda_cores[i]);
		LOG("  - Maximum Threads per MP : %d\n", devProp.maxThreadsPerMultiProcessor);
		LOG("  - Warp Size : %u\n", devProp.warpSize);
#ifdef _WIN64
		LOG("  - Shared Memory per MP : %llu\n", devProp.sharedMemPerMultiprocessor);
#else
		LOG("  - Shared Memory per MP : %zu\n", devProp.sharedMemPerMultiprocessor);
#endif
		LOG("   - Block per MP : %d\n", devProp.maxThreadsPerMultiProcessor / devProp.maxThreadsPerBlock);
#ifdef _WIN64
		LOG("   - Shared Memory per Block : %llu\n", devProp.sharedMemPerBlock);
#else
		LOG("   - Shared Memory per Block : %zu\n", devProp.sharedMemPerBlock);
#endif
		LOG("   - Maximum Threads per Block : %d\n", devProp.maxThreadsPerBlock);
		LOG("   - Maximum Threads of each Dimension of a Block (X: %d / Y: %d / Z: %d)\n",
			devProp.maxThreadsDim[_X], devProp.maxThreadsDim[_Y], devProp.maxThreadsDim[_Z]);
		LOG("   - Maximum Blocks of each Dimension of a Grid, (X: %d / Y: %d / Z: %d)\n",
			devProp.maxGridSize[_X], devProp.maxGridSize[_Y], devProp.maxGridSize[_Z]);
		LOG("   - Device supports allocating Managed Memory on this system : %d\n\n", devProp.managedMemory);
	}

	return true;
}

void CUDA::printMemoryInfo(int idx)
{
	cudaError_t ret = cudaSuccess;

	int old_idx = 0;
	if (cudaGetDevice(&old_idx) == cudaSuccess) {
		if (idx < active_gpus && idx < MAX_GPU) {
			if (cudaSetDevice(idx) == cudaSuccess) {
				size_t free, total;
				if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
					uint64_t gb = 1024 * 1024 * 1024;
					LOG("%d] CUDA Memory: %.1f/%.1fGB\n", idx, static_cast<double>(total - free) / gb, static_cast<double>(total) / gb);
				}
			}
		}
		cudaSetDevice(old_idx);
	}
}

int CUDA::getSMPerCore(int major, int minor)
{
	int smPerMultiproc = 0;

	switch (major) {
	case 2: // Fermi
		smPerMultiproc = 32;
		break;
	case 3: // Kepler
		smPerMultiproc = 192;
		break;
	case 5: // Maxwell
		smPerMultiproc = 128;
		break;
	case 6: // Pascal
		smPerMultiproc = (minor == 1) ? 128 : 64;
		break;
	case 7: // Volta, Turing
		smPerMultiproc = 64;
		break;
	case 8: // Ampere, Ada Lovelace
		smPerMultiproc = 128;
		break;
	default:
		LOG("<FAILED> Unsupported CUDA architecture");
	}

	return smPerMultiproc;
}


void CUDA::setWorkload(int size)
{
	int total_cores = 0;
	int max_core = 0;
	int max_idx = 0;
	int total_workload = 0;
	for (int i = 0; i < active_gpus; i++)
	{
		work_load[i] = 0;
		total_cores += cuda_cores[i];
		if (cuda_cores[i] > max_core) {
			max_core = cuda_cores[i];
			max_idx = i;
		}
	}
	// distributed data
	for (int i = 0; i < active_gpus; i++)
	{
		work_load[i] = (size * cuda_cores[i]) / total_cores;
		total_workload += work_load[i];
	}
	// added loss data
	work_load[max_idx] += (size - total_workload);	
}