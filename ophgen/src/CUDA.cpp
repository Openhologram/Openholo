#include "CUDA.h"
#include "define.h"

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
CUDA::CUDA()
{
	printDevInfo();
}


CUDA::~CUDA()
{
}

bool CUDA::printDevInfo()
{
	int devID;
	HANDLE_ERROR(cudaGetDevice(&devID));
	HANDLE_ERROR(cudaGetDeviceProperties(&devProp, devID));

	LOG("GPU Spec : %s\n", devProp.name);
	LOG(" - Global Memory : %u\n", devProp.totalGlobalMem);
	LOG(" - Const Memory : %u\n", devProp.totalConstMem);
	LOG("  - MP(Multiprocessor) Count : %d\n", devProp.multiProcessorCount);
	LOG("  - Maximum Threads per MP : %d\n", devProp.maxThreadsPerMultiProcessor);
	LOG("  - Warp Size : %u\n", devProp.warpSize);
	LOG("  - Shared Memory per MP : %u\n", devProp.sharedMemPerMultiprocessor);
	LOG("   - Block per MP : %d\n", devProp.maxThreadsPerMultiProcessor / devProp.maxThreadsPerBlock);

	LOG("   - Shared Memory per Block : %u\n", devProp.sharedMemPerBlock);
	LOG("   - Maximum Threads per Block : %d\n", devProp.maxThreadsPerBlock);
	LOG("   - Maximum Threads of each Dimension of a Block (X: %d / Y: %d / Z: %d)\n",
		devProp.maxThreadsDim[_X], devProp.maxThreadsDim[_Y], devProp.maxThreadsDim[_Z]);
	LOG("   - Maximum Blocks of each Dimension of a Grid, (X: %d / Y: %d / Z: %d)\n",
		devProp.maxGridSize[_X], devProp.maxGridSize[_Y], devProp.maxGridSize[_Z]);
	LOG("   - Device supports allocating Managed Memory on this system : %d\n\n", devProp.managedMemory);


	return true;
}

bool CUDA::init()
{

	return true;
}
