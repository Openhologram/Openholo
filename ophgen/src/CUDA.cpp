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


CUDA* CUDA::instance = nullptr;

CUDA::CUDA()
{
	printDevInfo();
	m_nThread = 512;
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
#ifdef _WIN64
	LOG(" - Global Memory : %llu\n", devProp.totalGlobalMem);
	LOG(" - Const Memory : %llu\n", devProp.totalConstMem);
#else
	LOG(" - Global Memory : %zu\n", devProp.totalGlobalMem);
	LOG(" - Const Memory : %zu\n", devProp.totalConstMem);
#endif
	LOG("  - MP(Multiprocessor) Count : %d\n", devProp.multiProcessorCount);
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


	return true;
}