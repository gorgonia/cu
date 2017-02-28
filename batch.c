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

CUresult processFn(fnargs_t* args){
	CUresult ret;
	switch (args->fn) {
	case fn_mallocD:
		abort();
		break;
	case fn_mallocH:
		abort();
		break;
	case fn_mallocManaged:
		abort();
		break;
	case fn_memfreeD:
		ret = cuMemFree(args->devPtr0);
		break;
	case fn_memfreeH:
		ret = cuMemFreeHost((void*)(args->ptr0));
		break;
	case fn_memcpy:
		ret = cuMemcpy(args->devPtr0, args->devPtr1, args->size);
		break;
	case fn_memcpyHtoD:
		ret = cuMemcpyHtoD(args->devPtr0, (void*)(args->ptr0), args->size);
		// fprintf(stderr, "ret memcpyHtoD %d args->ptr0 %d\n", ret, args->ptr0);
		break;
	case fn_memcpyDtoH:
		ret = cuMemcpyDtoH((void*)(args->ptr0), args->devPtr0, args->size);
		break;
	case fn_memcpyDtoD:
		abort();
		break;
	case fn_memcpyHtoDAsync:
		abort();
		break;
	case fn_memcpyDtoHAsync:
		abort();
		break;
	case fn_memcpyDtoDAsync:
		abort();
		break;
	case fn_launchKernel:
		ret = cuLaunchKernel(args->f, args->gridDimX, args->gridDimY, args->gridDimZ, 
			args->blockDimX, args->blockDimY, args->blockDimZ, 
			args->sharedMemBytes, args->stream, 
			(void**)(args->kernelParams), (void**)(args->extra));
		break;
	case fn_sync:
		ret = cuCtxSynchronize();
		break;
	case fn_lauchAndSync:
		abort();
		break;
	}
	return ret;
}

void process(fnargs_t* args, CUresult* retVal, int count){
	// fprintf(stderr,"Processing: %d functions \n", count);
	for (int i = 0; i < count; ++i) {
		// fprintf(stderr, "Processing function %d\n", i);
		CUresult ret;
		ret = processFn(&args[i]);
		// fprintf(stderr, "ret %d\n",ret);

		retVal[i] = ret;
	}
}

CUresult batchMalloc(fnargs_t* args, CUdeviceptr* ptrs, int count){
	CUresult ret;
	for (int i = 0; i < count; ++i){
		switch (args[i].fn){
		case fn_mallocD:
			ret = cuMemAlloc(&args[i].devPtr0, args[i].size);
			break;
		case fn_mallocH:
			abort();
			break;
		case fn_mallocManaged:
			abort();
			break;
		}

		if (ret != CUDA_SUCCESS) {
			return ret;
		}
	}
	return ret;
}