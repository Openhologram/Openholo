#pragma once
#include <cuda_runtime_api.h>
#include <sys.h> //for LOG() macro
#include <atomic>
#include <mutex>

class CUDA
{
private:
	CUDA();
	~CUDA();
	static CUDA *instance;
	cudaDeviceProp devProp;

	int m_nThread;
public:
	static CUDA* getInstance() {
		if (instance == nullptr) {
			instance = new CUDA();
			atexit(releaseInstance);
		}
		return instance;
	}

	static void releaseInstance() {
		if (instance != nullptr) {
			delete instance;
			instance = nullptr;
		}
	}

	void setCurThreads(int thread) { m_nThread = thread; }
	int getCurThreads() { return m_nThread; }
	int getMaxThreads() { return devProp.maxThreadsPerBlock; }
	int getWarpSize() { return devProp.warpSize; }

private:
	bool printDevInfo();

public:

private:
};