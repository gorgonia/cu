package cublas_test

import (
	"reflect"
	"runtime"
	"unsafe"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/blas"
	"gorgonia.org/cu"
	cublas "gorgonia.org/cu/blas"
	"gorgonia.org/tensor"
)

type Engine struct {
	tensor.StdEng
	ctx cu.Context
	*cublas.Standard
}

func newEngine() *Engine {
	ctx := cu.NewContext(cu.Device(0), cu.SchedAuto)
	blas := cublas.New(cublas.WithContext(ctx))
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

func (m foomem) Uintptr() uintptr { return uintptr(unsafe.Pointer(&m[0])) }
func (m foomem) MemSize() uintptr { return uintptr(len(m) * 8) }

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
	var ad, bd, pd *tensor.Dense
	if ad, bd, pd, err = e.checkThreeFloat(a, b, prealloc); err != nil {
		return errors.Wrapf(err, "MatVecMul failed pre check")
	}

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

func (e *Engine) MatMul(a, b, prealloc tensor.Tensor) (err error) {
	var ad, bd, pd *tensor.Dense
	if ad, bd, pd, err = e.checkThreeFloat(a, b, prealloc); err != nil {
		return errors.Wrapf(err, "MatVecMul failed pre check")
	}

	ado := a.DataOrder()
	bdo := b.DataOrder()
	if !ado.HasSameOrder(bdo) {
		return errors.Errorf("a does not have the same data order as b. a is %v. b is %v", a.DataOrder(), b.DataOrder())
	}

	// get result shapes. k is the shared dimension
	// a is (m, k)
	// b is (k, n)
	// c is (m, n)
	var m, n, k int
	m = ad.Shape()[0]
	k = ad.Shape()[1]
	n = bd.Shape()[1]

	// // wrt the strides, we use the original strides, because that's what BLAS needs, instead of calling .Strides()
	// // lda in colmajor = number of rows;
	// // lda in row major = number of cols
	var lda, ldb, ldc int
	tA, tB := blas.Trans, blas.Trans
	za := ado.IsTransposed()
	zb := bdo.IsTransposed()

	// swapping around the operands if they are row major (a becomes b, and b becomes a)
	switch {
	case ado.IsColMajor() && bdo.IsColMajor() && !za && !zb:
		lda = m
		ldb = k
		ldc = prealloc.Shape()[0]
		tA, tB = blas.NoTrans, blas.NoTrans
	case ado.IsColMajor() && bdo.IsColMajor() && za && !zb:
		lda = k
		ldb = k
		ldc = prealloc.Shape()[0]
		tA, tB = blas.Trans, blas.NoTrans
	case ado.IsColMajor() && bdo.IsColMajor() && za && zb:
		lda = k
		ldb = n
		ldc = prealloc.Shape()[0]
		tA, tB = blas.Trans, blas.Trans
	case ado.IsColMajor() && bdo.IsColMajor() && !za && zb:
		lda = m
		ldb = n
		ldc = prealloc.Shape()[0]
		tA, tB = blas.NoTrans, blas.Trans
	case ado.IsRowMajor() && bdo.IsRowMajor() && !za && !zb:
		lda = k
		ldb = n
		ldc = prealloc.Shape()[1]
		tA, tB = blas.NoTrans, blas.NoTrans

		// magic swappy thingy
		m, n = n, m
		lda, ldb = ldb, lda
		ad, bd = bd, ad
	case ado.IsRowMajor() && bdo.IsRowMajor() && za && !zb:
		lda = m
		ldb = n
		ldc = prealloc.Shape()[1]
		tA, tB = blas.Trans, blas.NoTrans

		// magic swappy thingy
		m, n = n, m
		lda, ldb = ldb, lda
		tA, tB = tB, tA
		ad, bd = bd, ad
	case ado.IsRowMajor() && bdo.IsRowMajor() && za && zb:
		lda = m
		ldb = k
		ldc = prealloc.Shape()[1]
		tA, tB = blas.Trans, blas.Trans

		// magic swappy thingy
		m, n = n, m
		lda, ldb = ldb, lda
		ad, bd = bd, ad
	case ado.IsRowMajor() && bdo.IsRowMajor() && !za && zb:
		lda = k
		ldb = k
		ldc = prealloc.Shape()[1]
		tA, tB = blas.NoTrans, blas.Trans

		// magic swappy thingy
		m, n = n, m
		lda, ldb = ldb, lda
		tA, tB = tB, tA
		ad, bd = bd, ad

	default:
		panic("Unreachable")
	}

	switch ad.Dtype() {
	case tensor.Float64:
		A := ad.Float64s()
		B := bd.Float64s()
		C := pd.Float64s()
		alpha, beta := float64(1), float64(0)
		e.Standard.Dgemm(tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)

	case tensor.Float32:
		A := ad.Float32s()
		B := bd.Float32s()
		C := pd.Float32s()
		alpha, beta := float32(1), float32(0)
		e.Standard.Sgemm(tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
	default:
		return errors.Errorf("Unsupported Dtype %v", ad.Dtype())
	}
	return e.Standard.Err()
}

func (e *Engine) Outer(a, b, prealloc tensor.Tensor) (err error) {
	var ad, bd, pd *tensor.Dense
	if ad, bd, pd, err = e.checkThreeFloat(a, b, prealloc); err != nil {
		return errors.Wrapf(err, "MatVecMul failed pre check")
	}
	m := ad.Size()
	n := bd.Size()
	pdo := pd.DataOrder()

	var lda int
	switch {
	case pdo.IsColMajor():
		lda = pd.Shape()[0]
	case pdo.IsRowMajor():
		aShape := a.Shape().Clone()
		bShape := b.Shape().Clone()
		if err = a.Reshape(aShape[0], 1); err != nil {
			return err
		}
		if err = b.Reshape(1, bShape[0]); err != nil {
			return err
		}

		if err = e.MatMul(a, b, prealloc); err != nil {
			return err
		}

		if err = b.Reshape(bShape...); err != nil {
			return
		}
		if err = a.Reshape(aShape...); err != nil {
			return
		}
		return nil
	}
	incX, incY := 1, 1
	switch ad.Dtype() {
	case tensor.Float64:
		x := ad.Float64s()
		y := bd.Float64s()
		A := pd.Float64s()
		alpha := float64(1)
		e.Standard.Dger(m, n, alpha, x, incX, y, incY, A, lda)
	case tensor.Float32:
		x := ad.Float32s()
		y := bd.Float32s()
		A := pd.Float32s()
		alpha := float32(1)
		e.Standard.Sger(m, n, alpha, x, incX, y, incY, A, lda)
	}
	return e.Standard.Err()
}
func Example() {
	// debug.SetGCPercent(-1)
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	matVecMulColmajorNonTransposed()
	matVecMulColmajorTransposed()
	matVecMulRowMajorNonTransposed()
	matVecMulRowMajorTransposed()

	matMulColmajorNTNT()
	matMulColmajorTNT()
	matMulColmajorTT()
	matMulColmajorNTT()

	matMulRowmajorNTNT()
	matMulRowmajorTNT()
	matMulRowmajorTT()
	matMulRowmajorNTT()

	outerColMajor()
	outerRowMajor()

	// Output:
	// ColMajor Non Transposed
	// A:
	// ⎡1  2  3⎤
	// ⎣4  5  6⎦
	//
	// B:[1  2  3]
	// C:[1000  1000]
	// C:
	// [14 32]
	// ==========
	// ColMajor Transposed
	// A:
	// ⎡1  4⎤
	// ⎢2  5⎥
	// ⎣3  6⎦
	//
	// B:[1  2]
	// C[1000  1000  1000]
	// C:
	// [9 12 15]
	// ==========
	// RowMajor Non Transposed
	// A:
	// ⎡1  2  3⎤
	// ⎣4  5  6⎦
	//
	// B:[1  2  3]
	// C[1000  1000]
	// C:
	// [14 32]
	// ==========
	// RowMajor Transposed
	// A:
	// ⎡1  4⎤
	// ⎢2  5⎥
	// ⎣3  6⎦
	//
	// B:[1  2]
	// C[1000  1000  1000]
	// C:
	// [9 12 15]
	// ==========
	// ColMajor Non Transposed Non Transposed
	// A:
	// ⎡1  2  3⎤
	// ⎣4  5  6⎦
	//
	// B:
	// ⎡ 0   1   2   3⎤
	// ⎢ 4   5   6   7⎥
	// ⎣ 8   9  10  11⎦
	//
	// C:
	// ⎡1000  1000  1000  1000⎤
	// ⎣1000  1000  1000  1000⎦
	//
	// C:
	// [32 68 38 83 44 98 50 113]
	// ==========
	// ColMajor Transposed Non Transposed
	// A:
	// ⎡1  4⎤
	// ⎢2  5⎥
	// ⎣3  6⎦
	//
	// B:
	// ⎡0  1  2  3⎤
	// ⎣4  5  6  7⎦
	//
	// C:
	// ⎡1000  1000  1000  1000⎤
	// ⎢1000  1000  1000  1000⎥
	// ⎣1000  1000  1000  1000⎦
	//
	// C:
	// [16 20 24 21 27 33 26 34 42 31 41 51]
	// ==========
	// ColMajor Transposed Transposed
	// A:
	// ⎡1  4⎤
	// ⎢2  5⎥
	// ⎣3  6⎦
	//
	// B:
	// ⎡0  2  4  6⎤
	// ⎣1  3  5  7⎦
	//
	// C:
	// ⎡1000  1000  1000  1000⎤
	// ⎢1000  1000  1000  1000⎥
	// ⎣1000  1000  1000  1000⎦
	//
	// C:
	// [4 5 6 14 19 24 24 33 42 34 47 60]
	// ==========
	// ColMajor Non Transposed Transposed
	// A:
	// ⎡1  2  3⎤
	// ⎣4  5  6⎦
	//
	// B:
	// ⎡ 0   3   6   9⎤
	// ⎢ 1   4   7  10⎥
	// ⎣ 2   5   8  11⎦
	//
	// C:
	// ⎡1000  1000  1000  1000⎤
	// ⎣1000  1000  1000  1000⎦
	//
	// C:
	// [8 17 26 62 44 107 62 152]
	// ==========
	// RowMajor Non Transposed Non Transposed
	// A:
	// ⎡1  2  3⎤
	// ⎣4  5  6⎦
	//
	// B:
	// ⎡ 0   1   2   3⎤
	// ⎢ 4   5   6   7⎥
	// ⎣ 8   9  10  11⎦
	//
	// C:
	// ⎡1000  1000  1000  1000⎤
	// ⎣1000  1000  1000  1000⎦
	//
	// C:
	// [32 38 44 50 68 83 98 113]
	// ==========
	// RowMajor Transposed Non Transposed
	// A:
	// ⎡1  3  5⎤
	// ⎣2  4  6⎦
	//
	// B:
	// ⎡ 0   1   2   3⎤
	// ⎢ 4   5   6   7⎥
	// ⎣ 8   9  10  11⎦
	//
	// C:
	// ⎡1000  1000  1000  1000⎤
	// ⎣1000  1000  1000  1000⎦
	//
	// C:
	// [52 61 70 79 64 76 88 100]
	// ==========
	// RowMajor Transposed Non Transposed
	// A:
	// ⎡1  3  5⎤
	// ⎣2  4  6⎦
	//
	// B:
	// ⎡ 0   3   6   9⎤
	// ⎢ 1   4   7  10⎥
	// ⎣ 2   5   8  11⎦
	//
	// C:
	// ⎡1000  1000  1000  1000⎤
	// ⎣1000  1000  1000  1000⎦
	//
	// C:
	// [13 40 67 94 16 52 88 124]
	// ==========
	// RowMajor Transposed Non Transposed
	// A:
	// ⎡1  2  3⎤
	// ⎣4  5  6⎦
	//
	// B:
	// ⎡ 0   3   6   9⎤
	// ⎢ 1   4   7  10⎥
	// ⎣ 2   5   8  11⎦
	//
	// C:
	// ⎡1000  1000  1000  1000⎤
	// ⎣1000  1000  1000  1000⎦
	//
	// C:
	// [8 26 44 62 17 62 107 152]
	// ==========
	// RowMajor Non Transposed
	// A:
	// [1  2  3]
	// B:[0  1]
	// C
	// ⎡1000  1000⎤
	// ⎢1000  1000⎥
	// ⎣1000  1000⎦
	//
	// C:
	// [0 0 0 1 2 3]
	// ==========
	// RowMajor Non Transposed
	// A:
	// [1  2  3]
	// B:[0  1]
	// C
	// ⎡1000  1000⎤
	// ⎢1000  1000⎥
	// ⎣1000  1000⎦
	//
	// C:
	// [0 1 0 2 0 3]
	// ==========
}
