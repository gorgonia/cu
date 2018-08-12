package cublas

import (
	"reflect"
	"unsafe"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/blas"
	"gorgonia.org/cu"
	"gorgonia.org/tensor"
)

func testSetup() (dev cu.Device, err error) {
	devices, _ := cu.NumDevices()

	if devices == 0 {
		err = errors.Errorf("NoDevice")
		return
	}

	dev = cu.Device(0)
	return
}

type Engine struct {
	tensor.StdEng
	ctx cu.Context
	*Standard
}

func newEngine() *Engine {
	ctx := cu.NewContext(cu.Device(0), cu.SchedAuto)
	blas := New(WithContext(ctx))
	return &Engine{
		ctx:      ctx,
		Standard: blas,
	}
}

func (e *Engine) AllocAccessible() bool { return true }

func (e *Engine) Alloc(size int64) (tensor.Memory, error) {
	return e.ctx.MemAllocManaged(size, cu.AttachGlobal)
}

func (e *Engine) AllocFlags() (tensor.MemoryFlag, tensor.DataOrder) {
	return tensor.MakeMemoryFlag(tensor.ManuallyManaged), tensor.ColMajor
}

func (e *Engine) Free(mem tensor.Memory, size int64) error {
	e.ctx.MemFree(mem.(cu.DevicePtr))
	return nil
}

func (e *Engine) Memset(mem tensor.Memory, val interface{}) error {
	panic("not implemented")
}

func (e *Engine) Memclr(mem tensor.Memory) {
	panic("not implemented")
}

func (e *Engine) Memcpy(dst tensor.Memory, src tensor.Memory) error {
	panic("not implemented")
}

func (e *Engine) Accessible(mem tensor.Memory) (tensor.Memory, error) {
	// panic("not implemented")
	size := mem.MemSize()
	retVal := make([]byte, int(size))
	e.ctx.MemcpyDtoH(unsafe.Pointer(&retVal[0]), cu.DevicePtr(mem.Uintptr()), int64(size))
	l := int(size / 8)
	foo2 := &reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&retVal[0])),
		Len:  l,
		Cap:  l,
	}
	return *(*foomem)(unsafe.Pointer(foo2)), e.ctx.Error()

}

func (e *Engine) WorksWith(order tensor.DataOrder) bool { return true }

func (e *Engine) NonStdAlloc() {}

func (e *Engine) ContextErr() error { return e.ctx.Error() }

type foomem []float64

func (m foomem) Uintptr() uintptr        { return uintptr(unsafe.Pointer(&m[0])) }
func (m foomem) Pointer() unsafe.Pointer { return unsafe.Pointer(&m[0]) }
func (m foomem) MemSize() uintptr        { return uintptr(len(m) * 8) }

func (e *Engine) checkThreeFloat(a, b, ret tensor.Tensor) (ad, bd, retVal *tensor.Dense, err error) {
	if /*a.IsNativelyAccessible() &&*/ !a.IsManuallyManaged() {
		return nil, nil, nil, errors.New("CUDA Engine only takes non-natively accessible memory (memory on graphics cards). a isn't.")
	}

	if /* b.IsNativelyAccessible() && */ !b.IsManuallyManaged() {
		return nil, nil, nil, errors.New("CUDA Engine only takes non-natively accessible memory (memory on graphics cards). b isn't")
	}

	if /* ret.IsNativelyAccessible() && */ !ret.IsManuallyManaged() {
		return nil, nil, nil, errors.New("CUDA Engine only takes non-natively accessible memory (memory on graphics cards). ret isn't")
	}

	if a.Dtype() != b.Dtype() || b.Dtype() != ret.Dtype() {
		return nil, nil, nil, errors.New("Expected a and b and retVal all to have the same Dtype")
	}
	var ok bool
	if ad, ok = a.(*tensor.Dense); !ok {
		return nil, nil, nil, errors.New("Expected a to be a *tensor.Dense")
	}
	if bd, ok = b.(*tensor.Dense); !ok {
		return nil, nil, nil, errors.New("Expected b to be a *tensor.Dense")
	}
	if retVal, ok = ret.(*tensor.Dense); !ok {
		return nil, nil, nil, errors.New("Expected ret to be a *tensor.Dense")
	}
	return
}

func (e *Engine) MatVecMul(a, b, prealloc tensor.Tensor) (err error) {
	var ad, bd, pd *tensor.Dense = a.(*tensor.Dense), b.(*tensor.Dense), prealloc.(*tensor.Dense)

	// if ad, bd, pd, err = e.checkThreeFloat(a, b, prealloc); err != nil {
	// 	return errors.Wrapf(err, "MatVecMul failed pre check")
	// }

	tA := blas.Trans
	do := a.DataOrder()
	z := do.IsTransposed()

	m := a.Shape()[0]
	n := a.Shape()[1]

	var lda int
	switch {
	case do.IsRowMajor() && z:
		tA = blas.NoTrans
		lda = m
	case do.IsRowMajor() && !z:
		lda = n
		m, n = n, m
	case do.IsColMajor() && z:
		tA = blas.Trans
		lda = n
		m, n = n, m
	case do.IsColMajor() && !z:
		lda = m
		tA = blas.NoTrans
	}

	incX, incY := 1, 1 // step size

	// ASPIRATIONAL TODO: different incX and incY
	// TECHNICAL DEBT. TECHDEBT. TECH DEBT
	// Example use case:
	// log.Printf("a %v %v", ad.Strides(), ad.ostrides())
	// log.Printf("b %v", b.Strides())
	// incX := a.Strides()[0]
	// incY = b.Strides()[0]

	switch ad.Dtype() {
	case tensor.Float64:
		A := ad.Float64s()
		X := bd.Float64s()
		Y := pd.Float64s()
		alpha, beta := float64(1), float64(0)
		e.Standard.Dgemv(tA, m, n, alpha, A, lda, X, incX, beta, Y, incY)
	case tensor.Float32:
		A := ad.Float32s()
		X := bd.Float32s()
		Y := pd.Float32s()
		alpha, beta := float32(1), float32(0)
		e.Standard.Sgemv(tA, m, n, alpha, A, lda, X, incX, beta, Y, incY)
	default:
		return errors.New("Unsupported Dtype")
	}
	return e.Standard.Err()
}
