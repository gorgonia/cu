package cu

// #include <cuda.h>
import "C"
import "unsafe"

// Function represents a CUDA function
type Function uintptr

const pointerSize = 8 // sorry, 64 bits only.

// Attribute returns the attribute information about a Function
func (fn Function) Attribute(attr FunctionAttribute) (int, error) {
	var pi C.int
	f := C.CUfunction(unsafe.Pointer(uintptr(fn)))
	a := C.CUfunction_attribute(attr)
	if err := result(C.cuFuncGetAttribute(&pi, a, f)); err != nil {
		return 0, err
	}
	return int(pi), nil
}

// SetCacheConfig sets the preferred cache configuration for a device function
func (fn Function) SetCacheConfig(conf FuncCacheConfig) error {
	f := C.CUfunction(unsafe.Pointer(uintptr(fn)))
	return result(C.cuFuncSetCacheConfig(f, C.CUfunc_cache(conf)))
}

// SetSharedMemConfig sets the shared memory configuration for a device function
func (fn Function) SetSharedMemConfig(conf SharedConfig) error {
	f := C.CUfunction(unsafe.Pointer(uintptr(fn)))
	return result(C.cuFuncSetSharedMemConfig(f, C.CUsharedconfig(conf)))
}

// LaunchKernel launches a CUDA function
func (fn Function) LaunchKernel(gridDimX, gridDimY, gridDimZ int, blockDimX, blockDimY, blockDimZ int, sharedMemBytes int, stream Stream, kernelParams []unsafe.Pointer) error {
	// Since Go 1.6, a cgo argument cannot have a Go pointer to Go pointer,
	// so we copy the argument values go C memory first.
	argv := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	argp := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	defer C.free(argv)
	defer C.free(argp)
	for i := range kernelParams {
		*((*unsafe.Pointer)(offset(argp, i))) = offset(argv, i)       // argp[i] = &argv[i]
		*((*uint64)(offset(argv, i))) = *((*uint64)(kernelParams[i])) // argv[i] = *kernelParams[i]
	}

	f := C.CUfunction(unsafe.Pointer(uintptr(fn)))
	err := result(C.cuLaunchKernel(
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

func offset(ptr unsafe.Pointer, i int) unsafe.Pointer {
	return unsafe.Pointer(uintptr(ptr) + pointerSize*uintptr(i))
}
