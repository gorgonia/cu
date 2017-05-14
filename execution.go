package cu

// #include <cuda.h>
import "C"
import "unsafe"

// Function represents a CUDA function
type Function uintptr

func makeFunction(fn C.CUfunction) Function {
	return Function(uintptr(unsafe.Pointer(fn)))
}

func (fn Function) c() C.CUfunction {
	return C.CUfunction(unsafe.Pointer(uintptr(fn)))
}

const pointerSize = 8 // sorry, 64 bits only.

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

	f := fn.c()
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

func (ctx *Ctx) LaunchKernel(fn Function, gridDimX, gridDimY, gridDimZ int, blockDimX, blockDimY, blockDimZ int, sharedMemBytes int, stream Stream, kernelParams []unsafe.Pointer) {
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

	function := fn.c()
	f := func() error {
		return result(C.cuLaunchKernel(
			function,
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
	}

	ctx.err = ctx.Do(f)
}
