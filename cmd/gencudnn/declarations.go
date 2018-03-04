package main

var empty struct{}

var ignoredEnums = map[string]struct{}{
	"cudnnStatus_t":                         empty,
	"cudnnConvolutionFwdPreference_t":       empty,
	"cudnnConvolutionFwdAlgo_t":             empty,
	"cudnnConvolutionBwdFilterPreference_t": empty,
	"cudnnConvolutionBwdFilterAlgo_t":       empty,
	"cudnnConvolutionBwdDataPreference_t":   empty,
	"cudnnConvolutionBwdDataAlgo_t":         empty,
}

var ignoredTypes = map[string]struct{}{
	"cudnnHandle_t":           {},
	"cudnnRuntimeTag_t":       {},
	"cudnnTensorDescriptor_t": {},
	// "cudnnConvolutionDescriptor_t":        {},
	// "cudnnPoolingDescriptor_t":            {},
	// "cudnnFilterDescriptor_t":             {},
	// "cudnnLRNDescriptor_t":                {},
	// "cudnnActivationDescriptor_t":         {},
	// "cudnnSpatialTransformerDescriptor_t": {},
	"cudnnOpTensorDescriptor_t": {},
	// "cudnnReduceTensorDescriptor_t":       {},
	// "cudnnCTCLossDescriptor_t":            {},
	// "cudnnConvolutionFwdAlgoPerf_t":       {},
	// "cudnnConvolutionBwdFilterAlgoPerf_t": {},
	// "cudnnConvolutionBwdDataAlgoPerf_t":   {},
	// "cudnnDropoutDescriptor_t":            {},
	// "cudnnRNNDescriptor_t":                {},
	// "cudnnPersistentRNNPlan_t":            {},
}

var ctypes2GoTypes = map[string]string{
	"cudnnHandle_t": "Context",
	// "cudnnRuntimeTag_t":                   "",
	"cudnnTensorDescriptor_t":             "TensorDescriptor",
	"cudnnConvolutionDescriptor_t":        "Convolution",
	"cudnnPoolingDescriptor_t":            "Pooling",
	"cudnnFilterDescriptor_t":             "Filter",
	"cudnnLRNDescriptor_t":                "LRN",
	"cudnnActivationDescriptor_t":         "Activation",
	"cudnnSpatialTransformerDescriptor_t": "SpatialTransformer",
	"cudnnOpTensorDescriptor_t":           "Op",
	"cudnnReduceTensorDescriptor_t":       "Reduction",
	"cudnnCTCLossDescriptor_t":            "CTCLoss",
	"cudnnConvolutionFwdAlgoPerf_t":       "ConvolutionFwdPerf",
	"cudnnConvolutionBwdFilterAlgoPerf_t": "ConvolutionBwdPerf",
	"cudnnConvolutionBwdDataAlgoPerf_t":   "ConvolutionBwdDataPerf",
	"cudnnDropoutDescriptor_t":            "Dropout",
	"cudnnRNNDescriptor_t":                "RNN",
	"cudnnPersistentRNNPlan_t":            "PersistentRNNPlan",
}

var builtins = map[string]string{
	"float":              "float32",
	"double":             "float64",
	"int":                "int",
	"unsigned":           "uint",
	"unsigned long":      "uint32",
	"unsigned long long": "uint64",
}
