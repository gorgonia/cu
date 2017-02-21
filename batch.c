#include <cuda.h>
#include "batch.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

CUresult cuDeviceGetAttributes(int* retVal, CUdevice_attribute* attrs, int n, CUdevice dev) {
	CUresult ret;
	int val;
	for (int i = 0; i < n; i++){
		ret = cuDeviceGetAttribute(&val, attrs[i], dev);

		if (ret != CUDA_SUCCESS) {
			return ret;
		}

		retVal[i] = val;
	}
	return CUDA_SUCCESS;
}

CUresult cuLaunchAndSync(CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) {
	CUresult ret;

	ret = cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimY, sharedMemBytes, hStream, kernelParams, extra);
	if (ret != CUDA_SUCCESS) {
		return ret;
	}

	return cuCtxSynchronize();
}