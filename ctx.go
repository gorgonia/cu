// +build !debug

package cu

// #include <cuda.h>
import "C"
import (
	"runtime"
	"sync"
	"time"
	"unsafe"

	"github.com/pkg/errors"
)

// Ctx is a standalone CUDA Context that is threadlocked.
type Ctx struct {
	CUContext
	work    chan (func() error)
	errChan chan error
	done    chan struct{}
	err     error
	mu      sync.Mutex // done chan's mutexc

	device     Device
	flags      ContextFlags
	locked     bool
	doneClosed bool // set to true when done is closed.
}

// NewContext creates a new context, and runs a listener locked to an OSThread. All work is piped through that goroutine
func NewContext(d Device, flags ContextFlags) *Ctx {
	var cctx C.CUcontext
	err := result(C.cuCtxCreate(&cctx, C.uint(flags), C.CUdevice(d)))
	if err != nil {
		panic(err)
	}
	ctx := newContext(CUContext{cctx})
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
	ctx := newContext(CUContext{cctx})
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
		done:      make(chan struct{}),
		errChan:   make(chan error, 100),
	}
	runtime.SetFinalizer(ctx, finalizeCtx)
	return ctx
}

// Close destroys the CUDA context and associated resources that has been created. Additionally, all channels of communications will be closed.
func (ctx *Ctx) Close() error {
	var empty C.CUcontext
	if ctx.CUContext.ctx == empty {
		return nil
	}

	if ctx.errChan != nil {
		close(ctx.errChan)
		ctx.errChan = nil
	}

	if ctx.work != nil {
		close(ctx.work)
		ctx.work = nil
	}

	// don't close done

	err := result(C.cuCtxDestroy(C.CUcontext(unsafe.Pointer(ctx.CUContext.ctx))))
	ctx.CUContext.ctx = empty
	return err
}

// Do does one function at a time.
func (ctx *Ctx) Do(fn func() error) error {
	ctx.work <- fn
	return <-ctx.errChan
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
loop:
	for {
		select {
		case w := <-ctx.work:
			ctx.errChan <- w()
		case <-ctx.done:
			break loop
		}
	}

	runtime.UnlockOSThread()
	return nil
}

/* context.Context implementation */

// Deadline returns the dealine for this context. There is no deadline set.
func (ctx *Ctx) Deadline() (deadline time.Time, ok bool) { return time.Time{}, false }

// Done implements context.Context
func (ctx *Ctx) Done() <-chan struct{} { return ctx.done }

// Err returns an error.
// If Done is not yet closed, Err returns nil.
// If Done is closed, Err returns a non-nil error explaining why:
// Canceled if the context was canceled
// or DeadlineExceeded if the context's deadline passed.
// After Err returns a non-nil error, successive calls to Err return the same error.
func (ctx *Ctx) Err() error {
	ctx.mu.Lock()
	if ctx.doneClosed {
		if ctx.err != nil {
			ctx.mu.Unlock()
			return errors.Wrap(ctx.err, "Context Canceled")
		}
		ctx.mu.Unlock()
		return errors.New("Context Canceled")
	}
	ctx.mu.Unlock()
	return nil
}

// Value always returns nil.
func (ctx *Ctx) Value(key interface{}) interface{} { return nil }

// Cancel is a context.CancelFunc
func (ctx *Ctx) Cancel() {
	ctx.mu.Lock()
	if ctx.doneClosed {
		ctx.mu.Unlock()
		return
	}
	close(ctx.done)
	ctx.doneClosed = true
	ctx.mu.Unlock()
}

func finalizeCtx(ctx *Ctx) { ctx.Close() }

/* Manually Written Methods */

func (ctx *Ctx) ResetL2Cache() {
	ctx.Do(ctx.CUContext.ResetL2Cache)
}
