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

CUresult cuAllocAndCopy(CUdeviceptr* mem, const void* src, size_t bytesize) {
	CUresult retVal;

	retVal = cuMemAlloc(mem, bytesize);
	if (retVal != CUDA_SUCCESS){
		return retVal;
	}

	retVal = cuMemcpyHtoD(*mem, src, bytesize);
	return retVal;
}

CUresult processFn(fnargs_t* args){
	CUresult ret;
	switch (args->fn) {
	case fn_setCurrent:
		ret =  cuCtxSetCurrent(args->ctx);
		break;
	case fn_mallocD:
		// fprintf(stderr, "mallocD %d\n", args->size);
		ret = cuMemAlloc(&args->devPtr0, args->size);
		// fprintf(stderr, "ret %d\n", ret);
		break;
	case fn_mallocH:
		abort();
		break;
	case fn_mallocManaged:
		ret = cuMemAllocManaged(&args->devPtr0, args->size, CU_MEM_ATTACH_GLOBAL);
		break;
	case fn_memfreeD:
		// fprintf(stderr, "memfree %p\n", args->devPtr0);
		ret = cuMemFree(args->devPtr0);
		// fprintf(stderr, "ret %d\n", ret);
		break;
	case fn_memfreeH:
		ret = cuMemFreeHost((void*)(args->ptr0));
		break;
	case fn_memcpy:
		// fprintf(stderr, "memCpy(%p, %p, %d) \n", args->devPtr0, args->devPtr1, args->size);
		ret = cuMemcpy(args->devPtr0, args->devPtr1, args->size);
		// fprintf(stderr, "Ret %d\n", ret);
		break;
	case fn_memcpyHtoD:
		// fprintf(stderr, "memcpyHtoD(%p, %p, %d)\n", args->devPtr0, args->ptr0, args->size);
		ret = cuMemcpyHtoD(args->devPtr0, (void*)(args->ptr0), args->size);
		// fprintf(stderr,"ret %d\n", ret);
		break;
	case fn_memcpyDtoH:
		// fprintf(stderr, "memcpyDtoH(%p, %p, %d)\n", args->ptr0, args->devPtr0, args->size);
		ret = cuMemcpyDtoH((void*)(args->ptr0), args->devPtr0, args->size);
		// fprintf(stderr,"ret %d\n", ret);
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
		// fprintf(stderr, "launch kernel. Kernel Params: %p\n", args->kernelParams);
		ret = cuLaunchKernel(args->f, args->gridDimX, args->gridDimY, args->gridDimZ, 
			args->blockDimX, args->blockDimY, args->blockDimZ, 
			args->sharedMemBytes, args->stream, 
			(void**)(args->kernelParams), (void**)(args->extra));
		// fprintf(stderr, "ret %d\n", ret);
		break;
	case fn_sync:
		// fprintf(stderr, "sync\n");
		ret = cuCtxSynchronize();
		// fprintf(stderr, "ret %d\n", ret);
		break;
	case fn_launchAndSync:
		abort();
		break;
	case fn_allocAndCopy:
		// fprintf(stderr, "alloc and copy\n");
		ret = cuAllocAndCopy(&args->devPtr0, (void*)(args->ptr0), args->size);
		// fprintf(stderr, "ret %d\n", ret);
		break;
	}
	return ret;
}

void process(CUcontext ctx, uintptr_t* args, CUresult* retVal, int count){
	cuCtxSetCurrent(ctx);
	// fprintf(stderr,"Processing: %d functions \n", count);
	for (int i = 0; i < count; ++i) {
		// // fprintf(stderr, "Processing function %d\n", i);
		CUresult ret;
		fnargs_t* toProc = (fnargs_t*)(args[i]);
		ret = processFn(toProc);
		// // fprintf(stderr, "ret %d\n",ret);

		retVal[i] = ret;
	}
	// fprintf(stderr, "DONE %d\n", count );
}

// CUresult batchMalloc(uintptr_t* args, CUdeviceptr* ptrs, int count){
// 	CUresult ret;
// 	for (int i = 0; i < count; ++i){
// 		switch (args[i].fn){
// 		case fn_mallocD:
// 			ret = cuMemAlloc(&args[i].devPtr0, args[i].size);
// 			break;
// 		case fn_mallocH:
// 			abort();
// 			break;
// 		case fn_mallocManaged:
// 			abort();
// 			break;
// 		}

// 		if (ret != CUDA_SUCCESS) {
// 			return ret;
// 		}
// 	}
// 	return ret;
// }