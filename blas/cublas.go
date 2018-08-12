package cublas

// #include <cublas_v2.h>
import "C"
import "gonum.org/v1/gonum/blas"

// Order is used to specify the matrix storage format. We still interact with
// an API that allows client calls to specify order, so this is here to document that fact.
type Order byte

const (
	RowMajor Order = iota // Row Major
	ColMajor              // Column Major (cublas assumes all matrices be stored in this order)
)

// PointerMode
type PointerMode byte

const (
	Host PointerMode = iota
	Device
)

const (
	NoTrans   = C.CUBLAS_OP_N // NoTrans represents the no-transpose operation
	Trans     = C.CUBLAS_OP_T // Trans represents the transpose operation
	ConjTrans = C.CUBLAS_OP_C // ConjTrans represents the conjugate transpose operation

	Upper = C.CUBLAS_FILL_MODE_UPPER // Upper is used to specify that the matrix is an upper triangular matrix
	Lower = C.CUBLAS_FILL_MODE_LOWER // Lower is used to specify that the matrix is an lower triangular matrix

	NonUnit = C.CUBLAS_DIAG_NON_UNIT // NonUnit is used to specify that the matrix is not a unit triangular matrix
	Unit    = C.CUBLAS_DIAG_UNIT     // Unit is used to specify that the matrix is a unit triangular matrix

	Left  = C.CUBLAS_SIDE_LEFT  // Left is used to specify a multiplication op is performed from the left
	Right = C.CUBLAS_SIDE_RIGHT // Right is used to specify a multiplication op is performed from the right
)

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func trans2cublasTrans(t blas.Transpose) C.cublasOperation_t {
	switch t {
	case blas.NoTrans:
		return NoTrans
	case blas.Trans:
		return Trans
	case blas.ConjTrans:
		return ConjTrans
	}
	panic("Unreachable")
}

func side2cublasSide(s blas.Side) C.cublasSideMode_t {
	switch s {
	case blas.Left:
		return Left
	case blas.Right:
		return Right
	}
	panic("Unreachable")
}

func diag2cublasDiag(d blas.Diag) C.cublasDiagType_t {
	switch d {
	case blas.Unit:
		return Unit
	case blas.NonUnit:
		return NonUnit
	}
	panic("Unreachable")
}

func uplo2cublasUplo(u blas.Uplo) C.cublasFillMode_t {
	switch u {
	case blas.Upper:
		return Upper
	case blas.Lower:
		return Lower
	}
	panic("Unreachable")
}
