#include <cuda.h>

extern CUresult cuDeviceGetAttributes(int* retVal, CUdevice_attribute* attrs, int n, CUdevice dev);

extern CUresult cuLaunchAndSync(CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
		unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
		unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra);

extern CUresult cuAllocAndCopy(CUdeviceptr* mem, const void* src, size_t bytesize);

typedef enum {
	fn_setCurrent,

	fn_mallocD,
	fn_mallocH,
	fn_mallocManaged,
	fn_memfreeD,
	fn_memfreeH,

	fn_memcpy,
	fn_memcpyHtoD,
	fn_memcpyDtoH,
	fn_memcpyDtoD,

	fn_memcpyHtoDAsync,
	fn_memcpyDtoHAsync,
	fn_memcpyDtoDAsync,

	fn_launchKernel,
	fn_sync,
	fn_launchAndSync,

	fn_allocAndCopy,
} batchFn;

typedef struct fnargs {
	batchFn fn;

	CUcontext ctx;

	CUdeviceptr devPtr0;
	CUdeviceptr devPtr1;

	uintptr_t ptr0;
	uintptr_t ptr1;	

	CUfunction f;

	unsigned int gridDimX;
	unsigned int gridDimY;
	unsigned int gridDimZ;

	unsigned int blockDimX;
	unsigned int blockDimY;
	unsigned int blockDimZ;

	unsigned int sharedMemBytes;

	uintptr_t kernelParams;
	uintptr_t extra;

	size_t size;
	CUstream stream;

} fnargs_t;

extern CUresult processFn(fnargs_t* args);
extern void process(CUcontext ctx, uintptr_t* args, CUresult* retVal, int count);
// extern void process(uintptr_t* args, CUresult* retVal, int count);
// extern CUresult batchMalloc(uintptr_t* args, CUdeviceptr* ptrs, int count);