package cublas

import "C"
import (
	"runtime"

	"github.com/chewxy/cu"
	"github.com/gonum/blas"
)

var (
	_ blas.Float32    = &Standalone{}
	_ blas.Float64    = &Standalone{}
	_ blas.Complex64  = &Standalone{}
	_ blas.Complex128 = &Standalone{}
)

type Standalone struct {
	cu.Context
	h C.cublasHandle_t

	o Order
	m PointerMode
	e error
}

func NewImplementation(opts ...ConsOpt) *Standalone {
	var handle C.cublasHandle_t
	C.cublasCreate(&handle)
	impl := &Standalone{
		h: handle,
	}
	runtime.SetFinalizer(impl, finalizeImpl)
	return impl
}

func (impl *Standalone) Err() error { return impl.e }

func finalizeImpl(impl *Standalone) {
	C.cublasDestroy(&impl.h)
}
