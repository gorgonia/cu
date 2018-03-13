package cudnn

// #include <cudnn_v7.h>
import "C"

type Filter struct {
	internal C.cudnnFilterDescriptor_t

	dataType DataType
	shape    []int
}

func NewFilter(dt DataType, format TensorFormat, shape []int) (*Filter, error) {
	var internal C.cudnnFilterDescriptor_t
	if err := result(C.cudnnCreateFilterDescriptor(&internal)); err != nil {
		return nil, err
	}

	switch len(shape) {
	case 4:
		k, c, h, w := C.int(shape[0]), C.int(shape[1]), C.int(shape[2]), C.int(shape[3])
		if err := result(C.cudnnSetFilter4dDescriptor(internal, dt.c(), format.c(), k, c, h, w)); err != nil {
			return nil, err
		}
	default:
		if err := result(C.cudnnSetFilterNdDescriptor, internal, dt.c(), format.c(), C.int(len(shape)), &shape[0]); err != nil {
			return nil, err
		}
	}
	return &Filter{
		internal: internal,

		dataType: dt,
		shape:    shape,
	}
}
