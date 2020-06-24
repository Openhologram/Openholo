#pragma once
#include <CL/cl.h>
#include <sys.h> //for LOG() macro
#include <atomic>
#include <mutex>
#define MAX_KERNEL_NAME 1024
#define checkError(E, S) errorCheck(E,S,__FILE__,__LINE__)
class OpenCL
{
private:
	OpenCL();
	~OpenCL();
	static OpenCL *instance;
	static std::mutex mutex;

public:
	static OpenCL* getInstance() {
		std::lock_guard<std::mutex> lock(mutex);
		if (instance == nullptr) {
			instance = new OpenCL();
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

private:
	bool init();
	bool printDevInfo(cl_device_id device_id);
	
	/**
	* @brief get the kernel name
	* @param[in] iKernel index of kernel source
	* @param[out] kernel kernel name
	*/
	void getKernelName(cl_int iKernel, char *kernel);

public:
	cl_context &getContext() { return context; }
	cl_command_queue &getCommand() { return commands; }
	cl_program &getProgram() { return program; }
	cl_kernel* getKernel() { return kernel; }
	cl_uint getNumOfKernel() { return nKernel; }
	bool LoadKernel(char *path);
	void errorCheck(cl_int err, const char *operation, char *filename, int line);


private:
	cl_context			context;				// compute context
	cl_command_queue	commands;				// compute command queue
	cl_program			program;				// compute program
	cl_kernel			*kernel;					// compute kernel
	cl_device_id		device_id;				// compute device id
	cl_uint				nPlatforms;
	cl_platform_id		*platform;
	cl_uint				nKernel;
	cl_uint				nUnits, nDimensions;
	size_t				group;
	size_t*				item;
};
std::mutex OpenCL::mutex;
OpenCL* OpenCL::instance = nullptr;