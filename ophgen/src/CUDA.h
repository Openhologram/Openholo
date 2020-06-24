#pragma once
#include <cuda_runtime.h>
#include <sys.h> //for LOG() macro
#include <atomic>
#include <mutex>

class CUDA
{
private:
	CUDA();
	~CUDA();
	static CUDA *instance;
	static std::mutex mutex;
	cudaDeviceProp devProp;

public:
	static CUDA* getInstance() {
		std::lock_guard<std::mutex> lock(mutex);
		if (instance == nullptr) {
			instance = new CUDA();
			atexit(releaseInstance);
		}
		return instance;
	}

	static void releaseInstance() {
		if (instance) {
			delete instance;
			instance = nullptr;
		}
	}

	int getMaxThreads() { return devProp.maxThreadsPerBlock; }
	int getWarpSize() { return devProp.warpSize; }

private:
	bool init();
	bool printDevInfo();

public:

private:
};
std::mutex CUDA::mutex;
CUDA* CUDA::instance = nullptr;