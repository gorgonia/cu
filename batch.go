package cu

/*
This file deals with the batching of CUDA calls. The design of the batch calls is very much
inspired, and later, copied directly from the golang.org/x/mobile/gl package (it turns out that the
design was much better than what I had originally been inspired with).

There are some differences and modifications made, due to the nature of the intended use of this package.

The gl package is licenced under the Go licence.

*/

// #cgo CFLAGS: -g -O3 -std=c99
// #include <cuda.h>
// #include "batch.h"
import "C"
import (
	"bytes"
	"fmt"
	"log"
	"runtime"
	"unsafe"
)

const workBufLen = 64

type call struct {
	fnargs   *fnargs
	blocking bool
}

// fnargs is a representation of function and arguments to the function
// it's a super huge struct because it has to contain all the possible things that can be passed into a function
type fnargs struct {
	fn C.batchFn

	ctx C.CUcontext

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

func (fn *fnargs) String() string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "%s. ", batchFnString[fn.fn])
	switch fn.fn {
	case C.fn_setCurrent:
		fmt.Fprintf(&buf, "Current Context %d", fn.ctx)
	case C.fn_mallocD:
		fmt.Fprintf(&buf, "Size %d", fn.size)
	case C.fn_mallocH:
		fmt.Fprintf(&buf, "Size %d", fn.size)
	case C.fn_mallocManaged:
		fmt.Fprintf(&buf, "Size %d", fn.size)
	case C.fn_memfreeD:
		fmt.Fprintf(&buf, "mem: 0x%x", fn.devptr0)
	case C.fn_memfreeH:
		fmt.Fprintf(&buf, "mem: 0x%x", fn.devptr0)
	case C.fn_memcpy:
		fmt.Fprintf(&buf, "dest: 0x%x, src: 0x%x, size: %v", fn.devptr0, fn.devptr1, fn.size)
	case C.fn_memcpyHtoD:
		fmt.Fprintf(&buf, "dest: 0x%x, src: 0x%x, size: %v", fn.devptr0, fn.ptr0, fn.size)
	case C.fn_memcpyDtoH:
		fmt.Fprintf(&buf, "dest: 0x%x, src: 0x%x, size: %v", fn.ptr0, fn.devptr0, fn.size)
	case C.fn_memcpyDtoD:

	case C.fn_memcpyHtoDAsync:

	case C.fn_memcpyDtoHAsync:

	case C.fn_memcpyDtoDAsync:

	case C.fn_launchKernel:
		fmt.Fprintf(&buf, "KernelParams: %v", fn.kernelParams)
	case C.fn_sync:
		fmt.Fprintf(&buf, "Current Context %d", fn.ctx)
	case C.fn_launchAndSync:

	case C.fn_allocAndCopy:
		fmt.Fprintf(&buf, "Size: %v, src: %v", fn.size, fn.ptr0)
	}
	return buf.String()
}

func (fn *fnargs) c() C.uintptr_t {
	return C.uintptr_t(uintptr(unsafe.Pointer(fn)))
}

// BatchedContext is a CUDA context where the CUDA calls are batched up.
//
// Typically a locked OS thread is made to execute the CUDA calls like so:
// 		func main() {
//			ctx := NewBatchedContext(...)
//
//			runtime.LockOSThread()
//			defer runtime.UnlockOSThread()
//
//			workAvailable := ctx.WorkAvailable()
//			go doWhatever(ctx)
//			for {
//				select {
//					case <- workAvailable:
//						ctx.DoWork()
//						err := ctx.Errors()
//						handleErrors(err)
//					case ...:
//				}
//			}
//		}
//
//		func doWhatever(ctx *BatchedContext) {
//			ctx.Memcpy(...)
//			// et cetera
//			// et cetera
//		}
//
// For the moment, BatchedContext only supports a limited number of CUDA Runtime APIs.
// Feel free to send a pull request with more APIs.
type BatchedContext struct {
	Context
	Device

	workAvailable chan struct{} // an empty struct is sent down workAvailable when there is work
	work          chan call     // queue of calls to exec

	queue   []call
	fns     []C.uintptr_t
	results []C.CUresult
	frees   []unsafe.Pointer
	retVal  chan DevicePtr

	// sync.Mutex

	initialized bool
}

// NewBatchedContext creates a batched CUDA context.
func NewBatchedContext(c Context, d Device) *BatchedContext {
	return &BatchedContext{
		Context: c,
		Device:  d,

		workAvailable: make(chan struct{}, 1),
		work:          make(chan call, workBufLen),
		queue:         make([]call, 0, workBufLen),
		fns:           make([]C.uintptr_t, 0, workBufLen),
		results:       make([]C.CUresult, workBufLen),
		frees:         make([]unsafe.Pointer, 0, 2*workBufLen),
		retVal:        make(chan DevicePtr),
		initialized:   true,
	}
}

func (ctx *BatchedContext) IsInitialized() bool { return ctx.initialized }

// enqueue puts a CUDA call into the queue (which is the `work` channel).
//
// Here a difference between this package and package `gl` exists.
func (ctx *BatchedContext) enqueue(c call) (retVal DevicePtr, err error) {
	if len(ctx.work) >= workBufLen-1 {
		ctx.workAvailable <- struct{}{}
	}
	ctx.work <- c

	// where in package `gl` a signal is opportunistically
	// sent to the `workAvailable` channel, here it isn't. This is because
	// the intended use of this package's batch processing capability is
	// to only process when the queue is full, or when a call is blocking.

	if c.blocking {
		select {
		case ctx.workAvailable <- struct{}{}:
		default:
		}
		retVal = <-ctx.retVal
		return retVal, ctx.errors()
	}
	return 0, ctx.errors()
}

// WorkAvailable returns the chan where work availability is broadcasted on.
func (ctx *BatchedContext) WorkAvailable() <-chan struct{} { return ctx.workAvailable }

// DoWork waits for work to come in from the queue. If it's blocking, the entire queue will be processed immediately.
// Otherwise it will be added to the batch queue.
func (ctx *BatchedContext) DoWork() {
	for {
		select {
		case w := <-ctx.work:
			ctx.queue = append(ctx.queue, w)
		default:
			if len(ctx.queue) == 0 {
				return
			}
		}

		blocking := ctx.queue[len(ctx.queue)-1].blocking

	enqueue:
		for len(ctx.queue) < cap(ctx.queue) && !blocking {
			select {
			case w := <-ctx.work:
				ctx.queue = append(ctx.queue, w)
				blocking = ctx.queue[len(ctx.queue)-1].blocking
			default:
				break enqueue
			}
		}

		for _, c := range ctx.queue {
			ctx.fns = append(ctx.fns, c.fnargs.c())
		}

		// debug and instrumentation related stuff
		logCaller("DoWork()")
		logf(ctx.introspect())
		addQueueLength(len(ctx.queue))
		addBlockingCallers()

		cctx := ctx.CUDAContext().ctx
		ctx.results = ctx.results[:cap(ctx.results)]                         // make sure of the maximum availability for ctx.results
		C.process(cctx, &ctx.fns[0], &ctx.results[0], C.int(len(ctx.queue))) // process the queue
		ctx.results = ctx.results[:len(ctx.queue)]                           // then  truncate it to the len of queue for reporting purposes

		if ctx.checkResults() {
			log.Printf("Errors found %v", ctx.checkResults())
			log.Printf("Errors: \n%v", ctx.errors())
			log.Printf(ctx.introspect())
		}

		if blocking {
			b := ctx.queue[len(ctx.queue)-1]
			var retVal *fnargs
			switch b.fnargs.fn {
			case C.fn_mallocD:
				retVal = (*fnargs)(unsafe.Pointer(uintptr(ctx.fns[len(ctx.fns)-1])))
				ctx.retVal <- DevicePtr(retVal.devptr0)
			case C.fn_mallocH:
			case C.fn_mallocManaged:
				retVal = (*fnargs)(unsafe.Pointer(uintptr(ctx.fns[len(ctx.fns)-1])))
				ctx.retVal <- DevicePtr(retVal.devptr0)
			case C.fn_allocAndCopy:
				retVal = (*fnargs)(unsafe.Pointer(uintptr(ctx.fns[len(ctx.fns)-1])))
				ctx.retVal <- DevicePtr(retVal.devptr0)
			}
			logf("\t[RET] %v", DevicePtr(retVal.devptr0))
		}

		// clear queue
		ctx.queue = ctx.queue[:0]
		ctx.fns = ctx.fns[:0]
	}
}

// Run manages the running of the BatchedContext. Because it's expected to run in a goroutine, an error channel is to be passed in
func (ctx *BatchedContext) Run(errChan chan error) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	for {
		select {
		case <-ctx.workAvailable:
			ctx.DoWork()
			if err := ctx.Errors(); err != nil {
				if errChan == nil {
					return err
				}
				errChan <- err

			}
		case w := <-ctx.Work():
			ctx.ErrChan() <- w()
		}
	}
}

// Cleanup is the cleanup function. It cleans up all the ancilliary allocations that has happened for all the batched calls.
// This method should be called when the context is done with - otherwise there'd be a lot of leaked memory.
//
// The main reason why this method exists is because there is no way to reliably free memory without causing weird issues in the CUDA calls.
func (ctx *BatchedContext) Cleanup() {
	for i, f := range ctx.frees {
		C.free(f)
		ctx.frees[i] = nil
	}
	ctx.frees = ctx.frees[:0]
}

// Close closes the batched context
func (ctx *BatchedContext) Close() error {
	ctx.initialized = false
	return ctx.Context.Close()
}

// Errors returns any errors that may have occured during a batch processing
func (ctx *BatchedContext) Errors() error { return ctx.errors() }

// FirstError returns the first error if there was any
func (ctx *BatchedContext) FirstError() error {
	for i, v := range ctx.results {
		if cuResult(v) != Success {
			return result(v)
		}
		ctx.results[i] = C.CUDA_SUCCESS
	}
	return nil
}

// SetCurrent sets the current context. This is usually unnecessary because SetCurrent will be called before batch processing the calls.
func (ctx *BatchedContext) SetCurrent() {
	fn := &fnargs{
		fn:  C.fn_setCurrent,
		ctx: ctx.CUDAContext().ctx,
	}
	c := call{fn, false}
	ctx.enqueue(c)
}

// MemAlloc allocates memory. It is a blocking call.
func (ctx *BatchedContext) MemAlloc(bytesize int64) (retVal DevicePtr, err error) {
	fn := &fnargs{
		fn:   C.fn_mallocD,
		size: C.size_t(bytesize),
	}
	c := call{fn, true}
	return ctx.enqueue(c)
}

func (ctx *BatchedContext) MemAllocManaged(bytesize int64, flags MemAttachFlags) (retVal DevicePtr, err error) {
	fn := &fnargs{
		fn:   C.fn_mallocManaged,
		size: C.size_t(bytesize),
	}
	c := call{fn, true}
	return ctx.enqueue(c)
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

func (ctx *BatchedContext) MemFree(mem DevicePtr) {
	fn := &fnargs{
		fn:      C.fn_memfreeD,
		devptr0: C.CUdeviceptr(mem),
	}
	c := call{fn, false}
	ctx.enqueue(c)
}

func (ctx *BatchedContext) MemFreeHost(p unsafe.Pointer) {
	fn := &fnargs{
		fn:   C.fn_memfreeH,
		ptr0: p,
	}
	c := call{fn, false}
	ctx.enqueue(c)
}

func (ctx *BatchedContext) LaunchKernel(function Function, gridDimX, gridDimY, gridDimZ int, blockDimX, blockDimY, blockDimZ int, sharedMemBytes int, stream Stream, kernelParams []unsafe.Pointer) {
	argv := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	argp := C.malloc(C.size_t(len(kernelParams) * pointerSize))
	for i := range kernelParams {
		*((*unsafe.Pointer)(offset(argp, i))) = offset(argv, i)       // argp[i] = &argv[i]
		*((*uint64)(offset(argv, i))) = *((*uint64)(kernelParams[i])) // argv[i] = *kernelParams[i]
	}

	// ctx.frees = append(ctx.frees, argv)
	// ctx.frees = append(ctx.frees, argp)

	fn := &fnargs{
		fn:             C.fn_launchKernel,
		f:              function.fn,
		gridDimX:       C.uint(gridDimX),
		gridDimY:       C.uint(gridDimY),
		gridDimZ:       C.uint(gridDimZ),
		blockDimX:      C.uint(blockDimX),
		blockDimY:      C.uint(blockDimY),
		blockDimZ:      C.uint(blockDimZ),
		sharedMemBytes: C.uint(sharedMemBytes),
		stream:         stream.c(),
		kernelParams:   (*unsafe.Pointer)(argp),
		extra:          (*unsafe.Pointer)(nil),
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

func (ctx *BatchedContext) LaunchAndSync(function Function, gridDimX, gridDimY, gridDimZ int, blockDimX, blockDimY, blockDimZ int, sharedMemBytes int, stream Stream, kernelParams []unsafe.Pointer) {
	ctx.LaunchKernel(function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams)
	ctx.Synchronize()
}

func (ctx *BatchedContext) AllocAndCopy(p unsafe.Pointer, bytesize int64) (retVal DevicePtr, err error) {
	fn := &fnargs{
		fn:   C.fn_allocAndCopy,
		size: C.size_t(bytesize),
		ptr0: p,
	}
	c := call{fn, true}
	logf("Alloc And Copy")
	return ctx.enqueue(c)
}

/* PRIVATE METHODS */

// checkResults returns true if an error has occured while processing the queue
func (ctx *BatchedContext) checkResults() bool {
	for _, v := range ctx.results {
		if v != C.CUDA_SUCCESS {
			return true
		}
	}
	return false
}

// errors convert ctx.results into errors
func (ctx *BatchedContext) errors() error {
	if !ctx.checkResults() {
		return nil
	}
	err := make(errorSlice, len(ctx.results))
	for i, res := range ctx.results {
		err[i] = result(res)
	}
	return err
}

// introspect is useful for finding out what calls are going to be made in the batched call
func (ctx *BatchedContext) introspect() string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "Queue: %d", len(ctx.queue))
	for _, v := range ctx.queue {
		fmt.Fprintf(&buf, "\n\t[QUEUE] %s", v.fnargs)
	}
	return buf.String()
}

var batchFnString = map[C.batchFn]string{
	C.fn_setCurrent:      "setCurrent",
	C.fn_mallocD:         "mallocD",
	C.fn_mallocH:         "mallocH",
	C.fn_mallocManaged:   "mallocManaged",
	C.fn_memfreeD:        "memfreeD",
	C.fn_memfreeH:        "memfreeH",
	C.fn_memcpy:          "memcpy",
	C.fn_memcpyHtoD:      "memcpyHtoD",
	C.fn_memcpyDtoH:      "memcpyDtoH",
	C.fn_memcpyDtoD:      "memcpyDtoD",
	C.fn_memcpyHtoDAsync: "memcpyHtoDAsync",
	C.fn_memcpyDtoHAsync: "memcpyDtoHAsync",
	C.fn_memcpyDtoDAsync: "memcpyDtoDAsync",
	C.fn_launchKernel:    "launchKernel",
	C.fn_sync:            "sync",
	C.fn_launchAndSync:   "lauchAndSync",

	C.fn_allocAndCopy: "allocAndCopy",
}
