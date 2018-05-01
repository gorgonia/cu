#include <cudnn_v7.h>

extern cudnnStatus_t gocudnnNewConvolution(cudnnMathType_t mathType, const int groupCount, 
	const int paddingSize, const int* padding,
	const int strideSize, const int* filterStrides, 
	const int dilSize, const int* dilation,
	cudnnConvolutionMode_t convolutionMode, cudnnDataType_t dataType, 
	cudnnConvolutionDescriptor_t *retVal);