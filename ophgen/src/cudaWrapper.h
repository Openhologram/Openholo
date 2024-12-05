#pragma once
#ifndef CUDAWRAPPER_H
#define CUDAWRAPPER_H
#include "ophGen.h"
#include <cuda_runtime_api.h>
#include <atomic>
#include <mutex>

#define MAX_GPU 16
#define MAX_INFO 14
#define MAX_LENGTH 256


class GEN_DLL cudaWrapper
{
public:
	enum DEVICE_INFO : int {
		DEVICE_NAME = 0,
		GLOBAL_MEMORY = 1,
		CONSTANT_MEMORY = 2,
		MANAGED_MEMORY = 3,
		MP_COUNT = 4,
		TOTAL_MP_COUNT = 5,
		MAX_THREADS_PER_MP = 6,
		WARP_SIZE = 7,
		BLOCK_PER_MP = 8,
		SHARED_MEMORY_PER_MP = 9,
		SHARED_MEMORY_PER_BLOCK = 10,
		MAX_THREADS_PER_BLOCK = 11,
		MAX_THREADS_DIMENSION = 12,
		MAX_GRID_SIZE = 13
	};

private:
	cudaWrapper();
	~cudaWrapper();
	static cudaWrapper *instance;
	static std::mutex mtx;

	cudaDeviceProp devProps[MAX_GPU];
	char devInfos[MAX_GPU][MAX_INFO][MAX_LENGTH];
	int num_gpu;
	int m_nThread;
	int cur_gpu;
	int work_load[MAX_GPU];
	int cuda_cores[MAX_GPU];
	int active_gpus;

public:
	static cudaWrapper* getInstance() {
		if (instance == nullptr) {
			std::lock_guard<std::mutex> lock(mtx);
			if (instance == nullptr) {
				instance = new cudaWrapper();
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
	cudaDeviceProp* getDeviceProps(int idx) { return &devProps[idx]; }
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

#endif // cudaWrapperWRAPPER_H