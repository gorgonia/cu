// +build !debug

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

func finalizeCtx(ctx *Ctx) { ctx.Close() }
