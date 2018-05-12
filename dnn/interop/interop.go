package t2cudnn

import (
	cudnn "gorgonia.org/cu/dnn"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	_ Tensor = &tensor.Dense{}
	_ Tensor = &gorgonia.Node{}
	_ Tensor = tensor.Tensor(nil)
)

type Tensor interface {
	Shape() tensor.Shape
	Strides() []int
	Dtype() tensor.Dtype
}

// Describe extracts the metadata from a tensor.Dense and returns a cuDNN TensorDescriptor
func Describe(t Tensor) (*cudnn.TensorDescriptor, error) {
	shape := t.Shape().Clone()
	strides := make([]int, len(shape))
	copy(strides, t.Strides())
	return cudnn.NewTensorDescriptor(cudnn.NCHW, Dtype2DataType(t.Dtype()), shape, strides)
}

func DescribeAsFilter(t Tensor, format cudnn.TensorFormat) (*cudnn.Filter, error) {
	shape := t.Shape().Clone()
	return cudnn.NewFilter(Dtype2DataType(t.Dtype()), format, shape)
}

// Dtype2DataType converts a tensor.Dtype to a cudnnDataType.
func Dtype2DataType(t tensor.Dtype) cudnn.DataType {
	switch t.Name() {
	case "float64":
		return cudnn.Double
	case "float32":
		return cudnn.Float
	case "float16":
		return cudnn.Half
	case "int8":
		return cudnn.Int8
	case "int32":
		return cudnn.Int32
	case "int128":
		return cudnn.Int8x4
	}
	panic("Unreachable")
}
