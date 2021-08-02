package cudnn

// #include <cudnn.h>
import "C"

// TODO

/*
func (ctx *Context) GetRNNLinLayerBiasParams(rnnDesc *RNN, pseudoLayer int, xDesc *TensorDescriptor, wDesc *Filter, w Memory, linLayerID int, linLayerBiasDesc *Filter, linLayerBias TODO) error {
	// call cudnnGetRNNLinLayerBiasParams
	return result(C.cudnnGetRNNLinLayerBiasParams(ctx.internal, rnnDesc.internal, C.int(pseudoLayer), xDesc.internal, wDesc.internal, unsafe.Pointer(w.Uintptr()), C.int(linLayerID), linLayerBiasDesc.internal, linLayerBias))
}
func (ctx *Context) GetRNNLinLayerMatrixParams(rnnDesc *RNN, pseudoLayer int, xDesc *TensorDescriptor, wDesc *Filter, w Memory, linLayerID int, linLayerMatDesc *Filter, linLayerMat TODO) error {
	// call cudnnGetRNNLinLayerMatrixParams
	return result(C.cudnnGetRNNLinLayerMatrixParams(ctx.internal, rnnDesc.internal, C.int(pseudoLayer), xDesc.internal, wDesc.internal, unsafe.Pointer(w.Uintptr()), C.int(linLayerID), linLayerMatDesc.internal, linLayerMat))
}

// Input. Handle to a previously created cuDNN context. For more information, see cudnnHandle_t.
func (co *Context) CTCLoss(probsDesc *TensorDescriptor, probs Memory, hostLabels TODO, hostLabelLengths TODO, hostInputLengths TODO, costs Memory, gradientsDesc *TensorDescriptor, gradients Memory, algo CTCLossAlgo, ctcLossDesc *CTCLoss, workspace Memory, workSpaceSizeInBytes uintptr) error {
	// DOUBLECHECK: "cudnnCTCLoss" returns Memory type in Parameter 8
	// call cudnnCTCLoss
	return result(C.cudnnCTCLoss(co.internal, probsDesc.internal, unsafe.Pointer(probs.Uintptr()), hostLabels, hostLabelLengths, hostInputLengths, unsafe.Pointer(costs.Uintptr()), gradientsDesc.internal, unsafe.Pointer(gradients.Uintptr()), algo.C(), ctcLossDesc.internal, unsafe.Pointer(workspace.Uintptr()), C.size_t(workSpaceSizeInBytes)))
}
*/
