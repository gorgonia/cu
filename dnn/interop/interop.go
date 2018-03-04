package t2cudnn

import (
	cudnn "gorgonia.org/cu/dnn"
	"gorgonia.org/tensor"
)

func DescriptorFromTensor(t *tensor.Dense) cudnn.TensorDescrptor {
	shape := t.Shape().Clone()
	var strides []int
	return cudnn.TensorDescriptor{
		DataType: Dtype2DataType(t.Dtype()),
		Shape:    shape,
		Strides:  strides,
	}
}

func Dtype2DataType(t tensor.Dtype) cudnn.DataType {
	switch t.Name() {
	case "float64":
		return Double
	case "float32":
		return Float
	case "float16":
		return Half
	case "int8":
		return Int8
	case "int32":
		return Int32
	case "int128":
		return Int8x4
	}
	panic("Unreachable")
}
