package cublas

// #include <cublas_v2.h>
import "C"
import (
	"runtime"

	"github.com/gonum/blas"
	"gorgonia.org/cu"
)

var (
	_ blas.Float32    = &Standard{}
	_ blas.Float64    = &Standard{}
	_ blas.Complex64  = &Standard{}
	_ blas.Complex128 = &Standard{}
)

// BLAS is the interface for all cuBLAS implementaions
type BLAS interface {
	cu.Context
	blas.Float32
	blas.Float64
	blas.Complex64
	blas.Complex128
}

type Standard struct {
	h C.cublasHandle_t
	o Order
	m PointerMode
	e error

	cu.Context
	dataOnDev bool
}

func NewStandardImplementation(ctx cu.Context) *Standard {
	var handle C.cublasHandle_t
	if err := status(C.cublasCreate(&handle)); err != nil {
		panic(err)
	}
	impl := &Standard{
		h:       handle,
		Context: ctx,
	}
	runtime.SetFinalizer(impl, finalizeImpl)
	return impl
}

func (impl *Standard) Err() error { return impl.e }

func finalizeImpl(impl *Standard) {
	C.cublasDestroy(impl.h)
}
