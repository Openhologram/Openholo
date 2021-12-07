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

	int m_nThread;
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

	void setCurThreads(int thread) { m_nThread = thread; }
	int getCurThreads() { return m_nThread; }
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