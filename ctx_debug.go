// +build debug

package cu

// #include <cuda.h>
import "C"
import (
	"runtime"
	"unsafe"
)

func newContext(c CUContext) *Ctx {
	ctx := &Ctx{
		CUContext: c,
		work:      make(chan func() error),
		done:      make(chan struct{}),
		errChan:   make(chan error, 100),
	}
	logf("Created Ctx: %p", ctx)
	runtime.SetFinalizer(ctx, finalizeCtx)
	return ctx
}

// Close destroys the CUDA context and associated resources that has been created. Additionally, all channels of communications will be closed.
func (ctx *Ctx) Close() error {
	logCaller("*Ctx.Close")
	ctx.mu.Lock()

	if ctx.doneClosed {
		ctx.mu.Unlock()
		return nil
	}

	var empty C.CUcontext
	if ctx.CUContext.ctx == empty {
		ctx.mu.Unlock()
		return nil
	}

	// close all the channels but do not nil them because we still need to drain them.

	if ctx.errChan != nil {
		close(ctx.errChan)
	}

	if ctx.work != nil {
		close(ctx.work)
	}
	ctx.mu.Unlock()
	ctx.Cancel() // there's a lock in Cancel, so we need to unlock it first

	ctx.mu.Lock()
	err := result(C.cuCtxDestroy(C.CUcontext(unsafe.Pointer(ctx.CUContext.ctx))))
	ctx.CUContext.ctx = empty
	ctx.mu.Unlock()

	return err
}

func finalizeCtx(ctx *Ctx) {
	logf("Finalizing %p", ctx)
	ctx.Close()
}
