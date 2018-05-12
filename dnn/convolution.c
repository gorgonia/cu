#include <cudnn_v7.h>
#include <stdio.h>

cudnnStatus_t gocudnnNewConvolution(cudnnConvolutionDescriptor_t *retVal,
	cudnnMathType_t mathType, const int groupCount, 
	const int size, const int* padding,
	const int* filterStrides, 
	const int* dilation,
	cudnnConvolutionMode_t convolutionMode, cudnnDataType_t dataType) {

	cudnnStatus_t status ;
	status = cudnnCreateConvolutionDescriptor(retVal);
	if (status != CUDNN_STATUS_SUCCESS) {
		puts("CANNOT CREATE \n");
		return status; 
	}

	status = cudnnSetConvolutionMathType(*retVal, mathType);
	if (status != CUDNN_STATUS_SUCCESS) {
		puts("CANNOT set MathType \n");
		return status;
	}

	status = cudnnSetConvolutionGroupCount(*retVal, groupCount); 
	if (status != CUDNN_STATUS_SUCCESS) {
		puts("CANNOT SetGroupCount \n");
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
		puts("Case 0,1\n");
		return CUDNN_STATUS_BAD_PARAM;
	case 2:
		puts("Case 2\n");
		padH = padding[0];
		padW = padding[1];
		u = filterStrides[0];
		v = filterStrides[1];
		dilationH = dilation[0];
		dilationW = dilation[1]; 
		printf("padH %d padW %d u %d v %d dH %d dW %d \n", padH, padW, u, v, dilationH,dilationW);

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
	puts("REady\n");
	return status;
}