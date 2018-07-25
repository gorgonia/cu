package cublas

// #include <cublas_v2.h>
import "C"
import (
	"sync"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/blas"
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

// Standard is the standard cuBLAS handler.
// By default it assumes that the data is in  RowMajor, DESPITE the fact that cuBLAS
// takes ColMajor only. This is done for the ease of use of developers writing in Go.
//
// Use New to create a new BLAS handler.
// Use the various ConsOpts to set the options
type Standard struct {
	h C.cublasHandle_t
	o Order
	m PointerMode
	e error

	cu.Context
	dataOnDev bool

	sync.Mutex
}

func New(opts ...ConsOpt) *Standard {
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

	return impl
}

func (impl *Standard) Init(opts ...ConsOpt) error {
	impl.Lock()
	defer impl.Unlock()

	var handle C.cublasHandle_t
	if err := status(C.cublasCreate(&handle)); err != nil {
		return errors.Wrapf(err, "Failed to initialize Standard implementation of CUBLAS")
	}
	impl.h = handle

	for _, opt := range opts {
		opt(impl)
	}
	return nil
}

func (impl *Standard) Err() error { return impl.e }

func (impl *Standard) Close() error {
	impl.Lock()
	defer impl.Unlock()

	var empty C.cublasHandle_t
	if impl.h == empty {
		return nil
	}
	if err := status(C.cublasDestroy(impl.h)); err != nil {
		return err
	}
	impl.h = empty
	impl.Context = nil
	return nil
}
