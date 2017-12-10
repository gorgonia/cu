package cublas

// #include <cublas_v2.h>
import "C"
import (
	"runtime"

	"github.com/chewxy/cu"
	"github.com/gonum/blas"
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

// Standard is the standard cuBLAS handler.
// By default it assumes that the data is in  RowMajor, DESPITE the fact that cuBLAS
// takes ColMajor only. This is done for the ease of use of developers writing in Go.
//
// Use NewStandardImplementation to create a new BLAS handler.
// Use the various ConsOpts to set the options
type Standard struct {
	h C.cublasHandle_t
	o Order
	m PointerMode
	e error

	cu.Context
	dataOnDev bool
}

func NewStandardImplementation(opts ...ConsOpt) *Standard {
	var handle C.cublasHandle_t
	if err := status(C.cublasCreate(&handle)); err != nil {
		panic(err)
	}

	impl := &Standard{
		h: handle,
	}

	for _, opt := range opts {
		opt(impl)
	}

	runtime.SetFinalizer(impl, finalizeImpl)
	return impl
}

func (impl *Standard) Err() error { return impl.e }

func finalizeImpl(impl *Standard) {
	C.cublasDestroy(impl.h)
}
