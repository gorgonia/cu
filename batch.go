package cu

// #cgo CFLAGS: -g -O3
// #include <cuda.h>
// #include "batch.h"
import "C"
import "unsafe"

type call struct {
	fnargs *fnargs
}

// fnargs is a representation of function and arguments to the function
// it's a super huge struct because it has to contain all the possible things that can be passed into a function
type fnargs struct {
	size C.size_t

	devptr0 C.CUdeviceptr
	devptr1 C.CUdeviceptr
	ptr1    unsafe.Pointer
	ptr0    unsafe.Pointer

	gridDimX, gridDimY, gridDimZ    C.uint
	blockDimX, blockDimY, blockDimZ C.uint
	sharedMemBytes                  C.uint
	kernelParams                    unsafe.Pointer

	stream C.CUstream // for async
}

type BatchedContext struct {
	ctx Context
	dev Device

	queue []call
}

// Attributes gets multiple attributes as provided
func (dev Device) Attributes(attrs ...DeviceAttribute) ([]int, error) {
	if len(attrs) == 0 {
		return nil, nil
	}
	cAttrs := make([]C.CUdevice_attribute, len(attrs))
	cRetVal := make([]C.int, len(attrs))
	size := C.int(len(attrs))

	for i, v := range attrs {
		cAttrs[i] = C.CUdevice_attribute(v)
	}

	err := result(C.cuDeviceGetAttributes(&cRetVal[0], &cAttrs[0], size, C.CUdevice(dev)))
	retVal := make([]int, len(attrs))
	for i, v := range cRetVal {
		retVal[i] = int(v)
	}

	return retVal, err
}

// LaunchAndSync launches the kernel and synchronizes the context
func (fn Function) LaunchAndSync(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes int, stream Stream, kernelParams []unsafe.Pointer) error {
	// err := fn.LaunchKernel(gridDimX, gridDimY, gridDimZ,
	// 	blockDimX, blockDimY, blockDimZ,
	// 	sharedMemBytes, stream, kernelParams)
	// if err != nil {
	// 	return err
	// }
	// return Synchronize()

	argv := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	argp := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	defer C.free(argv)
	defer C.free(argp)
	for i := range kernelParams {
		*((*unsafe.Pointer)(offset(argp, i))) = offset(argv, i)       // argp[i] = &argv[i]
		*((*uint64)(offset(argv, i))) = *((*uint64)(kernelParams[i])) // argv[i] = *kernelParams[i]
	}

	f := C.CUfunction(unsafe.Pointer(uintptr(fn)))
	err := result(C.cuLaunchAndSync(
		f,
		C.uint(gridDimX),
		C.uint(gridDimY),
		C.uint(gridDimZ),
		C.uint(blockDimX),
		C.uint(blockDimY),
		C.uint(blockDimZ),
		C.uint(sharedMemBytes),
		C.CUstream(unsafe.Pointer(uintptr(stream))),
		(*unsafe.Pointer)(argp),
		(*unsafe.Pointer)(unsafe.Pointer(uintptr(0)))))
	return err
}
