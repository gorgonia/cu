#include <cudnn.h>

cudnnStatus_t gocudnnNewConvolution(cudnnConvolutionDescriptor_t *retVal,
	cudnnMathType_t mathType, const int groupCount,
	const int size, const int* padding,
	const int* filterStrides,
	const int* dilation,
	cudnnConvolutionMode_t convolutionMode, cudnnDataType_t dataType) {

	cudnnStatus_t status ;
	status = cudnnCreateConvolutionDescriptor(retVal);
	if (status != CUDNN_STATUS_SUCCESS) {
		return status;
	}

	status = cudnnSetConvolutionMathType(*retVal, mathType);
	if (status != CUDNN_STATUS_SUCCESS) {
		return status;
	}

	status = cudnnSetConvolutionGroupCount(*retVal, groupCount);
	if (status != CUDNN_STATUS_SUCCESS) {
		return status;
	}

	int padH;
	int padW;
	int u;
	int v;
	int dilationH;
	int dilationW;
	switch (size) {
	case 0:
	case 1:
		return CUDNN_STATUS_BAD_PARAM;
	case 2:
		padH = padding[0];
		padW = padding[1];
		u = filterStrides[0];
		v = filterStrides[1];
		dilationH = dilation[0];
		dilationW = dilation[1];

		status = cudnnSetConvolution2dDescriptor(*retVal,
			padH, padW,
			u, v,
			dilationH, dilationW,
			convolutionMode, dataType);
		break;
	default:
		status = cudnnSetConvolutionNdDescriptor(*retVal, size, padding, filterStrides, dilation, convolutionMode, dataType);
		break;
	}
	return status;
}
