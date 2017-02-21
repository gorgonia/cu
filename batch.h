#include <cuda.h>

extern CUresult cuDeviceGetAttributes(int* retVal, CUdevice_attribute* attrs, int n, CUdevice dev);

extern CUresult cuLaunchAndSync(CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, 
		unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, 
		unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra);