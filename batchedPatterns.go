package cu

// #include <cuda.h>
// #include "batch.h"
import "C"
import (
	"unsafe"

	"github.com/pkg/errors"
)

/* COMMON PATTERNS */

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
	argv := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	argp := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	defer C.free(argv)
	defer C.free(argp)
	for i := range kernelParams {
		*((*unsafe.Pointer)(offset(argp, i))) = offset(argv, i)       // argp[i] = &argv[i]
		*((*uint64)(offset(argv, i))) = *((*uint64)(kernelParams[i])) // argv[i] = *kernelParams[i]
	}

	err := result(C.cuLaunchAndSync(
		fn.fn,
		C.uint(gridDimX),
		C.uint(gridDimY),
		C.uint(gridDimZ),
		C.uint(blockDimX),
		C.uint(blockDimY),
		C.uint(blockDimZ),
		C.uint(sharedMemBytes),
		stream.c(),
		(*unsafe.Pointer)(argp),
		(*unsafe.Pointer)(nil)))
	return err
}

// AllocAndCopy abstracts away the common pattern of allocating and then copying a Go slice to the GPU
func AllocAndCopy(p unsafe.Pointer, bytesize int64) (DevicePtr, error) {
	if bytesize == 0 {
		return 0, errors.Wrapf(InvalidValue, "Cannot allocate memory with size 0")
	}

	var d C.CUdeviceptr
	if err := result(C.cuAllocAndCopy(&d, p, C.size_t(bytesize))); err != nil {
		return 0, errors.Wrapf(err, "AllocAndCopy")
	}
	return DevicePtr(d), nil
}
