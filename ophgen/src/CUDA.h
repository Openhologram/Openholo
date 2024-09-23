#pragma once
#include <cuda_runtime_api.h>
#include <sys.h> //for LOG() macro
#include <atomic>
#include <mutex>

#define MAX_GPU 16

class CUDA
{
private:
	CUDA();
	~CUDA();
	static CUDA *instance;
	static std::mutex mtx;

	cudaDeviceProp devProps[MAX_GPU];
	int num_gpu;
	int m_nThread;
	int cur_gpu;
	int work_load[MAX_GPU];
	int cuda_cores[MAX_GPU];
	int active_gpus;

public:
	static CUDA* getInstance() {
		if (instance == nullptr) {
			std::lock_guard<std::mutex> lock(mtx);
			if (instance == nullptr) {
				instance = new CUDA();
				atexit(releaseInstance);
			}
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
	int getMaxThreads(int idx) { return devProps[idx].maxThreadsPerBlock; }
	int getWarpSize(int idx) { return devProps[idx].warpSize; }
	void printMemoryInfo(int idx);
	int getCurrentGPU() { return cur_gpu; }
	int getNumGPU() { return num_gpu; }
	
	void setWorkload(int size);
	int getWorkload(int idx) { return work_load[idx]; }
	
	int getActiveGPUs() { return active_gpus; }
	bool setActiveGPUs(int gpus) { 
		if (gpus <= MAX_GPU && gpus <= num_gpu) {
			active_gpus = gpus;
			return true;
		}
		else
			return false;
	}
	

private:
	void initGPU();
	bool printDevInfo();
	int getSMPerCore(int major, int minor);

public:

private:
};