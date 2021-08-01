package cudnn

// #include <cudnn.h>
import "C"
import (
	"unsafe"
)

func (ctx *Context) GetRNNLinLayerBiasParams(rnnDesc *RNN, pseudoLayer int, xDesc *TensorDescriptor, wDesc *Filter, w Memory, linLayerID int, linLayerBiasDesc *Filter, linLayerBias TODO) error {
	// call cudnnGetRNNLinLayerBiasParams
	return result(C.cudnnGetRNNLinLayerBiasParams(ctx.internal, rnnDesc.internal, C.int(pseudoLayer), xDesc.internal, wDesc.internal, unsafe.Pointer(w.Uintptr()), C.int(linLayerID), linLayerBiasDesc.internal, linLayerBias))
}
func (ctx *Context) GetRNNLinLayerMatrixParams(rnnDesc *RNN, pseudoLayer int, xDesc *TensorDescriptor, wDesc *Filter, w Memory, linLayerID int, linLayerMatDesc *Filter, linLayerMat TODO) error {
	// call cudnnGetRNNLinLayerMatrixParams
	return result(C.cudnnGetRNNLinLayerMatrixParams(ctx.internal, rnnDesc.internal, C.int(pseudoLayer), xDesc.internal, wDesc.internal, unsafe.Pointer(w.Uintptr()), C.int(linLayerID), linLayerMatDesc.internal, linLayerMat))
}
