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

func (ctx *Ctx) Do(fn func() error) error {
	ctx.work <- fn
	return <-ctx.errChan
}

// CUDAContext returns the CUDA Context
func (ctx *Ctx) CUDAContext() CUContext { return ctx.CUContext }

// Error returns the errors that may have occured during the calls.
func (ctx *Ctx) Err() error { return ctx.err }

// Work returns the channel where work will be passed in. In most cases you don't need this. Use Run instead.
func (ctx *Ctx) Work() chan func() error { return ctx.work }

// Run locks the goroutine to the OS thread and ties the CUDA context to the OS thread. For most cases, this would suffice
//
// The debug version of Run() will print the context every time it prints
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
	if ctx.CUContext == 0 {
		close(ctx.errChan)
		close(ctx.work)
		return
	}

	f := func() error {
		return result(C.cuCtxDestroy(C.CUcontext(unsafe.Pointer(&ctx.CUContext))))
	}
	if err := ctx.Do(f); err != nil {
		panic(err)
	}
	close(ctx.errChan)
	close(ctx.work)
}

/* Manually Written Methods */
