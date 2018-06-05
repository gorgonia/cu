extern cudnnStatus_t gocudnnNewConvolution(cudnnConvolutionDescriptor_t *retVal,
	cudnnMathType_t mathType, const int groupCount, 
	const int size, const int* padding,
	const int* filterStrides, 
	const int* dilation,
	cudnnConvolutionMode_t convolutionMode, cudnnDataType_t dataType);