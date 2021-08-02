package main

import "strings"

var empty struct{}

var ignoredEnums = map[string]struct{}{
	"cudnnStatus_t":                         empty,
	"cudnnConvolutionFwdPreference_t":       empty,
	"cudnnConvolutionFwdAlgo_t":             empty,
	"cudnnConvolutionBwdFilterPreference_t": empty,
	"cudnnConvolutionBwdFilterAlgo_t":       empty,
	"cudnnConvolutionBwdDataPreference_t":   empty,
	"cudnnConvolutionBwdDataAlgo_t":         empty,
	"cudnnConvolutionMode_t":                empty,
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

	// cuda11
	"cudnnFusedOpsVariantParamPack_t":  "FusedOpVariantParams",
	"cudnnFusedOpsConstParamPack_t":    "FusedOpConsts",
	"cudnnSeqDataDescriptor_t":         "SeqData",
	"cudnnTensorTransformDescriptor_t": "TensorTransform",
	"cudnnAlgorithmDescriptor_t":       "AlgorithmDescriptor",
	"cudnnAlgorithmPerformance_t":      "AlgorithmPerformance",
	"cudnnBackendDescriptor_t":         "Backend",
	"cudnnRNNDataDescriptor_t":         "RNNData",
	"cudnnAttnDescriptor_t":            "Attention",
}

var alphaBetaParams = []string{
	"alpha", "alpha1", "alpha2", "alpha3", "beta", "beta1",
	"alphaDataDiff", "alphaParamDiff", "betaDataDiff", "betaParamDiff",
}

var builtins = map[string]string{
	"float":              "float32",
	"double":             "float64",
	"int":                "int",
	"unsigned":           "uint",
	"unsigned long":      "uint32",
	"unsigned long long": "uint64",

	"size_t": "uintptr",

	"int64_t": "int64",
}

var go2cBuiltins = map[string]string{
	"float32": "float",
	"float64": "double",
	"int":     "int",
	"uint":    "uint",
	"uint32":  "ulong",
	"uint64":  "ulonglong",

	"uintptr": "size_t",

	"int64": "int64_t",
}

var nonPrimitives = map[string]string{
	"void*": "Memory",
}

var go2cNonPrimitives = map[string]string{
	"void*": "Memory",
}

// special for manual checks. The types are Go types.
var fnParamTypes = map[string]map[string]string{
	"cudnnCTCLoss":                                  {"labels": "[]int", "labelLengths": "[]int", "inputLengths": "[]int"},
	"cudnnGetCTCLossWorkspaceSize":                  {"labels": "[]int", "labelLengths": "[]int", "inputLengths": "[]int"},
	"cudnnFindConvolutionForwardAlgorithm":          {"returnedAlgoCount": "int"},
	"cudnnFindConvolutionForwardAlgorithmEx":        {"returnedAlgoCount": "int"},
	"cudnnFindConvolutionBackwardFilterAlgorithm":   {"returnedAlgoCount": "int"},
	"cudnnFindConvolutionBackwardFilterAlgorithmEx": {"returnedAlgoCount": "int"},
	"cudnnFindConvolutionBackwardDataAlgorithm":     {"returnedAlgoCount": "int"},
	"cudnnFindConvolutionBackwardDataAlgorithmEx":   {"returnedAlgoCount": "int"},
}

var deprecated = make(map[string]struct{})

func init() {
	for n, doc := range docs {
		if strings.Contains(doc, "has been deprecated in cuDNN 8.0.") {
			deprecated[n] = struct{}{}
		}
	}
}
