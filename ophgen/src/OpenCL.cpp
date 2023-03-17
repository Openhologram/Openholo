#include "OpenCL.h"
//#include <istream>
//#include <iostream>
//#include <string>
#include <sstream>
#include "ophPCKernel.cl"
#include <stdio.h>


OpenCL::OpenCL()
{
	nKernel = 0;
	nPlatforms = 0;
	device_id = nullptr;
	platform = nullptr;
	item = nullptr;
	kernel_source = nullptr;
	init();
}


OpenCL::~OpenCL()
{
	for (int i = 0; i < nKernel; i++)
	{
		clReleaseProgram(program[i]);
		clReleaseKernel(kernel[i]);
	}
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	delete[] item;
	delete[] kernel;
	delete[] platform;
	item = nullptr;
	kernel = nullptr;
	platform = nullptr;
}

const char *err_code(cl_int err_in)
{
	switch (err_in) {
	case CL_SUCCESS:
		return (char*)"CL_SUCCESS";
	case CL_DEVICE_NOT_FOUND:
		return (char*)"CL_DEVICE_NOT_FOUND";
	case CL_DEVICE_NOT_AVAILABLE:
		return (char*)"CL_DEVICE_NOT_AVAILABLE";
	case CL_COMPILER_NOT_AVAILABLE:
		return (char*)"CL_COMPILER_NOT_AVAILABLE";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		return (char*)"CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case CL_OUT_OF_RESOURCES:
		return (char*)"CL_OUT_OF_RESOURCES";
	case CL_OUT_OF_HOST_MEMORY:
		return (char*)"CL_OUT_OF_HOST_MEMORY";
	case CL_PROFILING_INFO_NOT_AVAILABLE:
		return (char*)"CL_PROFILING_INFO_NOT_AVAILABLE";
	case CL_MEM_COPY_OVERLAP:
		return (char*)"CL_MEM_COPY_OVERLAP";
	case CL_IMAGE_FORMAT_MISMATCH:
		return (char*)"CL_IMAGE_FORMAT_MISMATCH";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		return (char*)"CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case CL_BUILD_PROGRAM_FAILURE:
		return (char*)"CL_BUILD_PROGRAM_FAILURE";
	case CL_MAP_FAILURE:
		return (char*)"CL_MAP_FAILURE";
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:
		return (char*)"CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
		return (char*)"CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case CL_INVALID_VALUE:
		return (char*)"CL_INVALID_VALUE";
	case CL_INVALID_DEVICE_TYPE:
		return (char*)"CL_INVALID_DEVICE_TYPE";
	case CL_INVALID_PLATFORM:
		return (char*)"CL_INVALID_PLATFORM";
	case CL_INVALID_DEVICE:
		return (char*)"CL_INVALID_DEVICE";
	case CL_INVALID_CONTEXT:
		return (char*)"CL_INVALID_CONTEXT";
	case CL_INVALID_QUEUE_PROPERTIES:
		return (char*)"CL_INVALID_QUEUE_PROPERTIES";
	case CL_INVALID_COMMAND_QUEUE:
		return (char*)"CL_INVALID_COMMAND_QUEUE";
	case CL_INVALID_HOST_PTR:
		return (char*)"CL_INVALID_HOST_PTR";
	case CL_INVALID_MEM_OBJECT:
		return (char*)"CL_INVALID_MEM_OBJECT";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		return (char*)"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case CL_INVALID_IMAGE_SIZE:
		return (char*)"CL_INVALID_IMAGE_SIZE";
	case CL_INVALID_SAMPLER:
		return (char*)"CL_INVALID_SAMPLER";
	case CL_INVALID_BINARY:
		return (char*)"CL_INVALID_BINARY";
	case CL_INVALID_BUILD_OPTIONS:
		return (char*)"CL_INVALID_BUILD_OPTIONS";
	case CL_INVALID_PROGRAM:
		return (char*)"CL_INVALID_PROGRAM";
	case CL_INVALID_PROGRAM_EXECUTABLE:
		return (char*)"CL_INVALID_PROGRAM_EXECUTABLE";
	case CL_INVALID_KERNEL_NAME:
		return (char*)"CL_INVALID_KERNEL_NAME";
	case CL_INVALID_KERNEL_DEFINITION:
		return (char*)"CL_INVALID_KERNEL_DEFINITION";
	case CL_INVALID_KERNEL:
		return (char*)"CL_INVALID_KERNEL";
	case CL_INVALID_ARG_INDEX:
		return (char*)"CL_INVALID_ARG_INDEX";
	case CL_INVALID_ARG_VALUE:
		return (char*)"CL_INVALID_ARG_VALUE";
	case CL_INVALID_ARG_SIZE:
		return (char*)"CL_INVALID_ARG_SIZE";
	case CL_INVALID_KERNEL_ARGS:
		return (char*)"CL_INVALID_KERNEL_ARGS";
	case CL_INVALID_WORK_DIMENSION:
		return (char*)"CL_INVALID_WORK_DIMENSION";
	case CL_INVALID_WORK_GROUP_SIZE:
		return (char*)"CL_INVALID_WORK_GROUP_SIZE";
	case CL_INVALID_WORK_ITEM_SIZE:
		return (char*)"CL_INVALID_WORK_ITEM_SIZE";
	case CL_INVALID_GLOBAL_OFFSET:
		return (char*)"CL_INVALID_GLOBAL_OFFSET";
	case CL_INVALID_EVENT_WAIT_LIST:
		return (char*)"CL_INVALID_EVENT_WAIT_LIST";
	case CL_INVALID_EVENT:
		return (char*)"CL_INVALID_EVENT";
	case CL_INVALID_OPERATION:
		return (char*)"CL_INVALID_OPERATION";
	case CL_INVALID_GL_OBJECT:
		return (char*)"CL_INVALID_GL_OBJECT";
	case CL_INVALID_BUFFER_SIZE:
		return (char*)"CL_INVALID_BUFFER_SIZE";
	case CL_INVALID_MIP_LEVEL:
		return (char*)"CL_INVALID_MIP_LEVEL";
	case CL_INVALID_GLOBAL_WORK_SIZE:
		return (char*)"CL_INVALID_GLOBAL_WORK_SIZE";
	case CL_INVALID_PROPERTY:
		return (char*)"CL_INVALID_PROPERTY";

	default:
		return (char*)"UNKNOWN ERROR";
	}
}

void OpenCL::errorCheck(cl_int err, const char *operation, char *filename, int line)
{
	if (err != CL_SUCCESS)
	{
		fprintf(stderr, "Error during operation '%s', ", operation);
		fprintf(stderr, "in '%s' on line %d\n", filename, line);
		fprintf(stderr, "Error code was \"%s\" (%d)\n", err_code(err), err);
		exit(EXIT_FAILURE);
	}
}


bool OpenCL::LoadKernel()
{
	cl_int nErr;
	nKernel = sizeof(pKernel) / sizeof(pKernel[0]);

	// Create the compute kernel from the program
	kernel = new cl_kernel[nKernel];
	program = new cl_program[nKernel];


	char kname[MAX_KERNEL_NAME] = { 0, };
	size_t workSize;

	for (cl_uint i = 0; i < nKernel; i++) {
		kernel_source = const_cast<char*>(pKernel[i]);

		// Create the compute program from the source buffer
		program[i] = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, nullptr, &nErr);
		checkError(nErr, "Creating program");

		// Build the program
		nErr = clBuildProgram(program[i], 1, &device_id, nullptr, nullptr, nullptr);
		if (nErr != CL_SUCCESS) {
			size_t len;
			clGetProgramBuildInfo(program[i], device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
			char* buf = (char*)calloc(len + 1, sizeof(char));
			clGetProgramBuildInfo(program[i], device_id, CL_PROGRAM_BUILD_LOG, len + 1, buf, nullptr);
			LOG("\n=> %s\n", buf);
			free(buf);
			return false;
		}

		getKernelName(i, kname);
		kernel[i] = clCreateKernel(program[i], kname, &nErr);

		LOG("kernel[%d] : %s\n", i, kname);
		//nErr = clEnqueueNDRangeKernel(commands, kernel[i], 2, nullptr, global, local, 0, nullptr, nullptr);

		nErr = clGetKernelWorkGroupInfo(kernel[i], device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
			sizeof(size_t), &workSize, nullptr);
		LOG("%d] => Work Group Size: %d\n", i, workSize);
		nErr = clGetKernelWorkGroupInfo(kernel[i], device_id, CL_KERNEL_WORK_GROUP_SIZE,
			sizeof(size_t), &workSize, nullptr);
		work_size = workSize;

		LOG("%d] => Max Work Group Size: %d\n", i, workSize);
	}
	checkError(nErr, "Creating kernel");
	return true;
}

void OpenCL::getKernelName(cl_int iKernel, char *kernel)
{
	if (pKernel == nullptr) return;

	using namespace std;
	stringstream ss(pKernel[iKernel]);
	string item;
	size_t found;
	bool bFound = false;
	memset(kernel, 0, MAX_KERNEL_NAME);

	while (getline(ss, item, '\n')) {
		if (item.find("__kernel") != string::npos) {
			ss.seekg(-item.length(), ios_base::cur);
			while (getline(ss, item, ' ')) {
				if ((found = item.find("(")) != string::npos) {
					item = item.substr(0, found);
					bFound = true;
					break;
				}
			}
		}
		if (bFound) break;
	}
	if (!bFound) item = "Not Found";
	memcpy(kernel, item.c_str(), item.length());
}


bool OpenCL::printDevInfo(cl_device_id device_id)
{
	int				nErr;
	cl_device_id	id = nullptr;
	cl_device_type	type;
	cl_char			vendorName[1024] = { 0, };
	cl_char			deviceName[1024] = { 0, };

	nErr = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(deviceName), &deviceName, nullptr);
	if (nErr != CL_SUCCESS) {
		LOG("Error: Failed to access device name!\n");
		return false;
	}

	LOG("Device Name: %s\n", deviceName);

	nErr = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(type), &type, nullptr);
	if (nErr != CL_SUCCESS) {
		LOG("Error: Failed to access device type information!\n");
		return false;
	}
	LOG("Device Type: %s\n", type == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU");

	nErr = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendorName), &vendorName, nullptr);
	if (nErr != CL_SUCCESS) {
		LOG("Error: Failed to access device vendor name!\n");
		return false;
	}
	LOG("Vendor Name: %s\n", vendorName);

	nErr = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &nUnits, nullptr);
	if (nErr != CL_SUCCESS) {
		LOG("Error: Failed to access device number of compute units!\n");
		return false;
	}
	LOG("Max Units: %d\n", nUnits);

	nErr = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr);
	if (nErr != CL_SUCCESS) {
		LOG("Error: Failed to access device work group size!\n");
		return false;
	}
	LOG("Max Work Group Size: %d\n", group);


	nErr = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &nDimensions, nullptr);
	if (nErr != CL_SUCCESS) {
		LOG("Error: Failed to access device number of dimensions!\n");
		return false;
	}
	LOG("Max Work Item Deimesions: %d\n", nDimensions);

	if (item != nullptr) {
		delete[] item;
		item = nullptr;
	}
	item = new size_t[nDimensions];
	nErr = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * nDimensions, item, nullptr);
	if (nErr != CL_SUCCESS) {
		LOG("Error: Failed to access device work item size!\n");
		return false;
	}
	LOG("Max Work Item Size: [ ");
	for (int i = 0; i < nDimensions; i++) {
		LOG("%d", item[i]);
		if (i != nDimensions - 1)
			LOG(" / ");
	}
	LOG(" ]\n");
	return true;
}

bool OpenCL::init()
{
	int nErr;

	nErr = clGetPlatformIDs(0, nullptr, &nPlatforms);
	checkError(nErr, "Finding platforms");

	if (nPlatforms == 0) return false;

	if(platform == nullptr)
		platform = new cl_platform_id[nPlatforms];
	ZeroMemory(platform, nPlatforms);

	nErr = clGetPlatformIDs(nPlatforms, platform, nullptr);
	checkError(nErr, "Getting platforms");

	// Secure a GPU
	for (int i = 0; i < nPlatforms; i++)
	{
		nErr = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
		if (nErr == CL_SUCCESS) break;
	}

	if (device_id == nullptr) checkError(nErr, "Finding a device");

	printDevInfo(device_id);

	// Create a compute context
	context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &nErr);
	checkError(nErr, "Creating context");

	// Create a command queue
	commands = clCreateCommandQueue(context, device_id, 0, &nErr);
	checkError(nErr, "Creating command queue");
	

	return true;
}
