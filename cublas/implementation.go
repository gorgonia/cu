package cublas

// #include <cublas_v2.h>
import "C"
import "github.com/gonum/blas"

// Type check assertions:
var (
	_ blas.Float32    = &Implementation{}
	_ blas.Float64    = &Implementation{}
	_ blas.Complex64  = &Implementation{}
	_ blas.Complex128 = &Implementation{}
)

// Implementation is a BLAS implementation that fulfils the gonum/blas interface
type Implementation struct {
	h C.cublasHandle_t
	e error
}

// NewImplementation creates a new implementation.
func NewImplementation() *Implementation {
	var handle C.cublasHandle_t
	C.cublasCreate(&handle)
	return &Implementation{
		h: handle,
	}
}

// Err returns the error if there is any
func (impl *Implementation) Err() error { return impl.e }
