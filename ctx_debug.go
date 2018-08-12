// +build debug

package cu

// #include <cuda.h>
import "C"
import (
	"runtime"
	"unsafe"
)

// Ctx is a standalone CUDA Context that is threadlocked.
type Ctx struct {
	CUContext
	work    chan (func() error)
	errChan chan error
	err     error

	device Device
	flags  ContextFlags
	locked bool
}

// NewContext creates a new context, and runs a listener locked to an OSThread. All work is piped through that goroutine
func NewContext(d Device, flags ContextFlags) *Ctx {
	var cctx C.CUcontext
	err := result(C.cuCtxCreate(&cctx, C.uint(flags), C.CUdevice(d)))
	if err != nil {
		panic(err)
	}
	ctx := newContext(makeContext(cctx))
	ctx.device = d
	ctx.flags = flags

	errChan := make(chan error)
	go ctx.Run(errChan)
	if err := <-errChan; err != nil {
		panic(err)
	}

	return ctx
}

// NewManuallyManagedContext creates a new context, but the Run() method which locks a goroutine to an OS thread, has to be manually run
func NewManuallyManagedContext(d Device, flags ContextFlags) *Ctx {
	var cctx C.CUcontext
	err := result(C.cuCtxCreate(&cctx, C.uint(flags), C.CUdevice(d)))
	if err != nil {
		panic(err)
	}
	ctx := newContext(makeContext(cctx))
	ctx.device = d
	ctx.flags = flags

	return ctx
}

// CtxFromCUContext is another way of buildinga *Ctx.
//
// Typical example:
//	cuctx, err := dev.MakeContext(SchedAuto)
// 	if err != nil {
//		..error handling..
//	}
// 	ctx := CtxFroMCUContext(d, cuctx)
func CtxFromCUContext(d Device, cuctx CUContext, flags ContextFlags) *Ctx {
	ctx := newContext(cuctx)
	ctx.device = d
	ctx.flags = flags
	return ctx
}

func newContext(c CUContext) *Ctx {
	ctx := &Ctx{
		CUContext: c,
		work:      make(chan func() error),
		errChan:   make(chan error),
	}
	logf("Created %p", ctx)
	runtime.SetFinalizer(ctx, finalizeCtx)
	return ctx
}

// Close destroys the CUDA context and associated resources that has been created. Additionally, all channels of communications will be closed.
func (ctx *Ctx) Close() error {
	logf("Closing Ctx %v | ", ctx)
	logCaller("Ctx.Close")
	var empty C.CUcontext
	if ctx.CUContext.ctx == empty {
		return nil
	}

	if ctx.errChan != nil {
		close(ctx.errChan)
	}

	if ctx.work != nil {
		close(ctx.work)
	}

	err := result(C.cuCtxDestroy(C.CUcontext(unsafe.Pointer(ctx.CUContext.ctx))))
	ctx.CUContext.ctx = empty
	ctx.errChan = nil
	ctx.work = nil
	return err
}

func (ctx *Ctx) Do(fn func() error) error {
	ctx.work <- fn
	err := <-ctx.errChan
	return err
}

// CUDAContext returns the CUDA Context
func (ctx *Ctx) CUDAContext() CUContext { return ctx.CUContext }

// Error returns the errors that may have occured during the calls.
func (ctx *Ctx) Error() error { return ctx.err }

// Work returns the channel where work will be passed in. In most cases you don't need this. Use Run instead.
func (ctx *Ctx) Work() <-chan func() error { return ctx.work }

// ErrChan returns the internal error channel used
func (ctx *Ctx) ErrChan() chan error { return ctx.errChan }

// Run locks the goroutine to the OS thread and ties the CUDA context to the OS thread. For most cases, this would suffice
//
// Note: errChan that is passed in should NOT be the same errChan as the one used internally for signalling.
// The main reasoning for passing in an error channel is to support two different kinds of run modes:
//
// The typical use example is as such:
//
/*
	func A() {
			ctx := NewContext(d, SchedAuto)
			errChan := make(chan error)
			go ctx.Run(errChan)
			if err := <- errChan; err != nil {
				// handleError
			}
			doSomethingWithCtx(ctx)
	}
*/
// And yet another run mode supported is running of the context in the main thread:
//
/*
	func main() {
		ctx := NewContext(d, SchedAuto)
		go doSomethingWithCtx(ctx)
		if err := ctx.Run(nil); err != nil{
			// handle error
		}
	}
*/
func (ctx *Ctx) Run(errChan chan error) error {
	runtime.LockOSThread()

	// set current, which locks the context to the OS thread
	if err := SetCurrentContext(ctx.CUContext); err != nil {
		if errChan != nil {
			errChan <- err
		} else {
			return err
		}
		return nil
	}
	close(errChan)

	// wait for Do()s
	for w := range ctx.work {
		current, _ := CurrentContext()
		logf("Current Context %v", current)
		ctx.errChan <- w()
	}
	runtime.UnlockOSThread()
	return nil
}

func finalizeCtx(ctx *Ctx) {
	logf("Finalizing %p", ctx)
	ctx.Close()
}

/* Manually Written Methods */
