package cu

// #cgo CFLAGS: -g -O3 -std=c99
// #include <cuda.h>
// #include "batch.h"
import "C"
import (
	"runtime"
	"unsafe"
)

const workBufLen = 7

type call struct {
	fnargs   *fnargs
	blocking bool
}

// fnargs is a representation of function and arguments to the function
// it's a super huge struct because it has to contain all the possible things that can be passed into a function
type fnargs struct {
	fn C.batchFn

	devptr0 C.CUdeviceptr
	devptr1 C.CUdeviceptr

	ptr0 unsafe.Pointer
	ptr1 unsafe.Pointer

	f C.CUfunction

	gridDimX, gridDimY, gridDimZ    C.uint
	blockDimX, blockDimY, blockDimZ C.uint
	sharedMemBytes                  C.uint

	kernelParams *unsafe.Pointer // void* stuff
	extra        *unsafe.Pointer

	size   C.size_t
	stream C.CUstream // for async
}

func (fn *fnargs) c() C.fnargs_t {
	return *(*C.fnargs_t)(unsafe.Pointer(fn))
}

// BatchedContext is a CUDA context where the cgo calls are batched up.
type BatchedContext struct {
	Context
	Device

	queue   []call
	fns     []C.fnargs_t
	results []C.CUresult
	frees   []unsafe.Pointer
	retVals []interface{}
}

func NewBatchedContext(c Context, d Device) *BatchedContext {
	return &BatchedContext{
		Context: c,
		Device:  d,

		queue:   make([]call, 0, workBufLen+1),
		fns:     make([]C.fnargs_t, workBufLen+1, workBufLen+1),
		results: make([]C.CUresult, workBufLen+1),
		frees:   make([]unsafe.Pointer, 0, workBufLen*2),
	}
}

func (ctx *BatchedContext) enqueue(c call) {
	if len(ctx.queue) == workBufLen-1 || c.blocking {
		ctx.queue = append(ctx.queue, c)
		ctx.DoWork()
		return
	}
	ctx.queue = append(ctx.queue, c)
}

func (ctx *BatchedContext) DoWork() {
	runtime.LockOSThread()
	for i, c := range ctx.queue {
		// log.Printf("B4 %d: %# v", i, pretty.Formatter(c.fnargs))
		ctx.fns[i] = c.fnargs.c()
		// log.Printf("AT %d: %# v", i, pretty.Formatter(ctx.fns[i]))
	}
	C.process(&ctx.fns[0], &ctx.results[0], C.int(len(ctx.queue)))

	for _, f := range ctx.frees {
		C.free(f)
	}

	// clear queue
	ctx.queue = ctx.queue[:0]
	ctx.frees = ctx.frees[:0]
	runtime.UnlockOSThread()
}

func (ctx *BatchedContext) Memcpy(dst, src DevicePtr, byteCount int64) {
	fn := &fnargs{
		fn:      C.fn_memcpy,
		devptr0: C.CUdeviceptr(dst),
		devptr1: C.CUdeviceptr(src),
		size:    C.size_t(byteCount),
	}
	c := call{fn, false}
	ctx.enqueue(c)
}

func (ctx *BatchedContext) MemcpyHtoD(dst DevicePtr, src unsafe.Pointer, byteCount int64) {
	fn := &fnargs{
		fn:      C.fn_memcpyHtoD,
		devptr0: C.CUdeviceptr(dst),
		ptr0:    src,
		size:    C.size_t(byteCount),
	}
	c := call{fn, false}
	ctx.enqueue(c)
}

func (ctx *BatchedContext) MemcpyDtoH(dst unsafe.Pointer, src DevicePtr, byteCount int64) {
	fn := &fnargs{
		fn:      C.fn_memcpyDtoH,
		devptr0: C.CUdeviceptr(src),
		ptr0:    dst,
		size:    C.size_t(byteCount),
	}
	c := call{fn, false}
	ctx.enqueue(c)
}

func (ctx *BatchedContext) LaunchKernel(function Function, gridDimX, gridDimY, gridDimZ int, blockDimX, blockDimY, blockDimZ int, sharedMemBytes int, stream Stream, kernelParams []unsafe.Pointer) {
	argv := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	argp := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	ctx.frees = append(ctx.frees, argv)
	ctx.frees = append(ctx.frees, argp)

	for i := range kernelParams {
		*((*unsafe.Pointer)(offset(argp, i))) = offset(argv, i)       // argp[i] = &argv[i]
		*((*uint64)(offset(argv, i))) = *((*uint64)(kernelParams[i])) // argv[i] = *kernelParams[i]
	}
	f := C.CUfunction(unsafe.Pointer(uintptr(function)))
	fn := &fnargs{
		fn:             C.fn_launchKernel,
		f:              f,
		gridDimX:       C.uint(gridDimX),
		gridDimY:       C.uint(gridDimY),
		gridDimZ:       C.uint(gridDimZ),
		blockDimX:      C.uint(blockDimX),
		blockDimY:      C.uint(blockDimY),
		blockDimZ:      C.uint(blockDimZ),
		sharedMemBytes: C.uint(sharedMemBytes),
		stream:         C.CUstream(unsafe.Pointer(uintptr(stream))),
		kernelParams:   (*unsafe.Pointer)(argp),
		extra:          (*unsafe.Pointer)(unsafe.Pointer(uintptr(0))),
	}
	c := call{fn, false}
	ctx.enqueue(c)
}

func (ctx *BatchedContext) Synchronize() {
	fn := &fnargs{
		fn: C.fn_sync,
	}
	c := call{fn, false}
	ctx.enqueue(c)
}

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
