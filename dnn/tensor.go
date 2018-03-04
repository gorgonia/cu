package cudnn

// #include <cudnn_v7.h>
import "C"
import "github.com/pkg/errors"

type TensorDescriptor struct {
	desc C.cudnnTensorDescriptor_t // ptr to struct

	// internal data for fast
	format   TensorFormat
	dataType DataType
	shape    []int // NCHW format for 4-tensors
	strides  []int
}

func NewTensorDescriptor(format TensorFormat, dt DataType, shape, strides []int) (*TensorDescriptor, error) {
	var t C.cudnnTensorDescriptor_t
	if err := result(C.cudnnCreateTensorDescriptor(&t)); err != nil {
		return nil, err
	}

	desc := &TensorDescriptor{
		desc:     t,
		format:   format,
		dataType: dt,
		shape:    shape,
		strides:  strides,
	}

	err := desc.set(t)
	return desc, err
}

func (desc TensorDescriptor) set(t C.cudnnTensorDescriptor_t) error {
	switch len(desc.shape) {
	case 4:
		if len(desc.strides) == 4 {
			// use explicit
			res := C.cudnnSetTensor4dDescriptorEx(t, desc.dataType.c(),
				C.int(desc.shape[0]),   // N
				C.int(desc.shape[1]),   // C
				C.int(desc.shape[2]),   // H
				C.int(desc.shape[3]),   // W
				C.int(desc.strides[0]), // NStrides
				C.int(desc.strides[1]), // Cstrides
				C.int(desc.strides[2]), // HStrides
				C.int(desc.strides[3]), // WStrides
			)
			return result(res)
		}

		res := C.cudnnSetTensor4dDescriptor(t, desc.format.c(), desc.dataType.c(),
			C.int(desc.shape[0]), // N
			C.int(desc.shape[1]), // C
			C.int(desc.shape[2]), // H
			C.int(desc.shape[3]), // W
		)
		return result(res)

	}
	return errors.Errorf(nyi, "set for len == ", len(desc.shape))

}
