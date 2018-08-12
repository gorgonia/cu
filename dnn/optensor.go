package cudnn

// #include <cudnn.h>
import "C"
import (
	"runtime"
	"unsafe"

	"github.com/pkg/errors"
)

// Op is a tuple describing the operation that needs to be done
type Op struct {
	internal C.cudnnOpTensorDescriptor_t

	op             OpTensorOp     // The Operation that needs to be done
	dataType       DataType       // The Data type
	nanPropagation NanPropagation // NaN propagation strategies
}

// NewOp creates a new Op with the provided settings
func NewOp(op OpTensorOp, dt DataType, prop NanPropagation) (*Op, error) {
	var internal C.cudnnOpTensorDescriptor_t
	if err := result(C.cudnnCreateOpTensorDescriptor(&internal)); err != nil {
		return nil, err
	}

	if err := result(C.cudnnSetOpTensorDescriptor(internal, op.C(), dt.C(), prop.C())); err != nil {
		return nil, err
	}
	retVal := &Op{
		internal:       internal,
		op:             op,
		dataType:       dt,
		nanPropagation: prop,
	}
	runtime.SetFinalizer(retVal, destroyOp)
	return retVal, nil
}

// Op returns the operation that needs to be done.
func (op *Op) Op() OpTensorOp { return op.op }

// DataType returns the data type that the operation is supposed to work on.
func (op *Op) DataType() DataType { return op.dataType }

// NaNPropagation returns the NaN propagation strategy.
func (op *Op) NaNPropagation() NanPropagation { return op.nanPropagation }

// DoOp actually performs the operation.
func (ctx *Context) DoOp(op *Op,
	alpha1 float64, aDesc *TensorDescriptor, aData Memory,
	alpha2 float64, bDesc *TensorDescriptor, bData Memory,
	beta float64, cDesc *TensorDescriptor, cData Memory) error {

	// dtype checks
	if !(aDesc.dataType == bDesc.dataType && bDesc.dataType == cDesc.dataType) {
		return errors.Errorf(dtypeMismatch3, cDesc.dataType, aDesc.dataType, bDesc.dataType)
	}

	if cDesc.dataType == Double && op.dataType != cDesc.dataType {
		return errors.Errorf(dtypeMismatch3, Double, cDesc.dataType, op.dataType)
	}

	if op.dataType != Float && op.dataType != Double {
		return errors.Errorf(dtypeMismatch2, Float, Double, op.dataType)
	}

	// shapecheck
	if !(shapeEq(aDesc.shape, bDesc.shape) && shapeEq(bDesc.shape, cDesc.shape)) {
		return errors.Errorf(shapeMismatch3, aDesc.shape, bDesc.shape, cDesc.shape)
	}

	// location check
	if bData.Uintptr() == cData.Uintptr() && aData.Uintptr() != cData.Uintptr() {
		// If the input tensor B is the same tensor as the destination tensor C,
		// then the input tensor A also must be the same tensor as the destination tensor C.
		return errors.Errorf(memoryError3, cData.Uintptr(), aData.Uintptr(), bData.Uintptr())
	}

	// alpha beta generation
	var alpha1C, alpha2C, betaC unsafe.Pointer
	if op.dataType == Float {
		var a1, a2, b C.float
		a1 = C.float(float32(alpha1))
		a2 = C.float(float32(alpha2))
		b = C.float(float32(beta))

		alpha1C = unsafe.Pointer(&a1)
		alpha2C = unsafe.Pointer(&a2)
		betaC = unsafe.Pointer(&b)
	} else {
		var a1, a2, b C.double
		a1 = C.double(alpha1)
		a2 = C.double(alpha2)
		b = C.double(beta)

		alpha1C = unsafe.Pointer(&a1)
		alpha2C = unsafe.Pointer(&a2)
		betaC = unsafe.Pointer(&b)
	}

	res := C.cudnnOpTensor(ctx.internal, op.internal,
		alpha1C, aDesc.internal, aData.Pointer(),
		alpha2C, bDesc.internal, bData.Pointer(),
		betaC, cDesc.internal, cData.Pointer(),
	)
	return result(res)
}

func destroyOp(obj *Op) { C.cudnnDestroyOpTensorDescriptor(obj.internal) }
