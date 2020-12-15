package main

var ignored = map[string]struct{}{
	// "cudnnActivationBackward":                            {}, //
	// "cudnnActivationForward":                             {}, //
	// "cudnnAddTensor":                                     {}, //
	// "cudnnBatchNormalizationBackward":                    {}, //
	// "cudnnBatchNormalizationForwardInference":            {}, //
	// "cudnnBatchNormalizationForwardTraining":             {}, //
	// "cudnnCTCLoss":                                       {}, //
	// "cudnnConvolutionBackwardBias":                       {}, //
	// "cudnnConvolutionBackwardData":                       {}, //
	// "cudnnConvolutionBackwardFilter":                     {}, //
	// "cudnnConvolutionBiasActivationForward":              {}, //
	// "cudnnConvolutionForward":                            {}, //
	"cudnnCreate": {},
	// "cudnnCreateActivationDescriptor":                    {}, //
	// "cudnnCreateCTCLossDescriptor":                       {}, //
	"cudnnCreateConvolutionDescriptor": {},
	// "cudnnCreateDropoutDescriptor":                       {}, //
	// "cudnnCreateFilterDescriptor":                        {}, //
	// "cudnnCreateLRNDescriptor":                           {}, //
	// "cudnnCreateOpTensorDescriptor":                      {}, //
	// "cudnnCreatePersistentRNNPlan":                       {}, //
	// "cudnnCreatePoolingDescriptor":                       {}, //
	// "cudnnCreateRNNDescriptor":                           {}, //
	// "cudnnCreateReduceTensorDescriptor":                  {}, //
	// "cudnnCreateSpatialTransformerDescriptor":            {}, //
	// "cudnnCreateTensorDescriptor":                        {}, //
	// "cudnnDeriveBNTensorDescriptor":                      {}, //
	"cudnnDestroy": {},
	// "cudnnDestroyActivationDescriptor":                   {}, //
	// "cudnnDestroyCTCLossDescriptor":                      {}, //
	"cudnnDestroyConvolutionDescriptor": {},
	// "cudnnDestroyDropoutDescriptor":                      {}, //
	// "cudnnDestroyFilterDescriptor":                       {}, //
	// "cudnnDestroyLRNDescriptor":                          {}, //
	// "cudnnDestroyOpTensorDescriptor":                     {}, //
	// "cudnnDestroyPersistentRNNPlan":                      {}, //
	// "cudnnDestroyPoolingDescriptor":                      {}, //
	// "cudnnDestroyRNNDescriptor":                          {}, //
	// "cudnnDestroyReduceTensorDescriptor":                 {}, //
	// "cudnnDestroySpatialTransformerDescriptor":           {}, //
	// "cudnnDestroyTensorDescriptor":                       {}, //
	// "cudnnDivisiveNormalizationBackward":                 {}, //
	// "cudnnDivisiveNormalizationForward":                  {}, //
	// "cudnnDropoutBackward":                               {}, //
	// "cudnnDropoutForward":                                {}, //
	// "cudnnDropoutGetReserveSpaceSize":                    {}, //
	// "cudnnDropoutGetStatesSize":                          {}, //
	// "cudnnFindConvolutionBackwardDataAlgorithm":          {}, //
	// "cudnnFindConvolutionBackwardDataAlgorithmEx":        {}, //
	// "cudnnFindConvolutionBackwardFilterAlgorithm":        {}, //
	// "cudnnFindConvolutionBackwardFilterAlgorithmEx":      {}, //
	// "cudnnFindConvolutionForwardAlgorithm":               {}, //
	// "cudnnFindConvolutionForwardAlgorithmEx":             {}, //
	"cudnnGetActivationDescriptor":                       {},
	"cudnnGetCTCLossDescriptor":                          {},
	"cudnnGetCTCLossWorkspaceSize":                       {},
	"cudnnGetConvolution2dDescriptor":                    {},
	"cudnnGetConvolution2dForwardOutputDim":              {},
	"cudnnGetConvolutionBackwardDataAlgorithm":           {},
	"cudnnGetConvolutionBackwardDataAlgorithmMaxCount":   {},
	"cudnnGetConvolutionBackwardDataAlgorithm_v7":        {},
	"cudnnGetConvolutionBackwardDataWorkspaceSize":       {},
	"cudnnGetConvolutionBackwardFilterAlgorithm":         {},
	"cudnnGetConvolutionBackwardFilterAlgorithmMaxCount": {},
	"cudnnGetConvolutionBackwardFilterAlgorithm_v7":      {},
	"cudnnGetConvolutionBackwardFilterWorkspaceSize":     {},
	"cudnnGetConvolutionForwardAlgorithm":                {},
	"cudnnGetConvolutionForwardAlgorithmMaxCount":        {},
	"cudnnGetConvolutionForwardAlgorithm_v7":             {},
	"cudnnGetConvolutionForwardWorkspaceSize":            {},
	"cudnnGetConvolutionGroupCount":                      {},
	"cudnnGetConvolutionMathType":                        {},
	"cudnnGetConvolutionNdDescriptor":                    {},
	"cudnnGetConvolutionNdForwardOutputDim":              {},
	"cudnnGetCudartVersion":                              {},
	"cudnnGetDropoutDescriptor":                          {},
	"cudnnGetErrorString":                                {},
	"cudnnGetFilter4dDescriptor":                         {},
	"cudnnGetFilterNdDescriptor":                         {},
	"cudnnGetLRNDescriptor":                              {},
	"cudnnGetOpTensorDescriptor":                         {},
	"cudnnGetPooling2dDescriptor":                        {},
	"cudnnGetPooling2dForwardOutputDim":                  {},
	"cudnnGetPoolingNdDescriptor":                        {},
	"cudnnGetPoolingNdForwardOutputDim":                  {},
	"cudnnGetProperty":                                   {},
	"cudnnGetRNNDescriptor":                              {},
	// "cudnnGetRNNLinLayerBiasParams":                      {}, //
	// "cudnnGetRNNLinLayerMatrixParams":                    {}, //
	// "cudnnGetRNNParamsSize":                              {}, //
	// "cudnnGetRNNTrainingReserveSize":                     {}, //
	// "cudnnGetRNNWorkspaceSize":                           {}, //
	"cudnnGetReduceTensorDescriptor": {},
	// "cudnnGetReductionIndicesSize":                       {}, //
	// "cudnnGetReductionWorkspaceSize":                     {}, //
	"cudnnGetStream":             {},
	"cudnnGetTensor4dDescriptor": {},
	"cudnnGetTensorNdDescriptor": {},
	"cudnnGetTensorSizeInBytes":  {},
	"cudnnGetVersion":            {},
	// "cudnnIm2Col":                                        {}, //
	// "cudnnLRNCrossChannelBackward":                       {}, //
	// "cudnnLRNCrossChannelForward":                        {}, //
	// "cudnnOpTensor":                                      {}, //
	// "cudnnPoolingBackward":                               {}, //
	// "cudnnPoolingForward":                                {}, //
	"cudnnQueryRuntimeError": {},
	// "cudnnRNNBackwardData":                               {}, //
	// "cudnnRNNBackwardWeights":                            {}, //
	// "cudnnRNNForwardInference":                           {}, //
	// "cudnnRNNForwardTraining":                            {}, //
	// "cudnnReduceTensor":                                  {}, //
	// "cudnnRestoreDropoutDescriptor":                      {}, //
	// "cudnnScaleTensor":                                   {}, //
	// "cudnnSetActivationDescriptor":                       {}, //
	// "cudnnSetCTCLossDescriptor":                          {}, //
	"cudnnSetConvolution2dDescriptor": {},
	"cudnnSetConvolutionGroupCount":   {},
	"cudnnSetConvolutionMathType":     {},
	"cudnnSetConvolutionNdDescriptor": {},
	// "cudnnSetDropoutDescriptor":                          {}, //
	// "cudnnSetFilter4dDescriptor":                         {}, //
	// "cudnnSetFilterNdDescriptor":                         {}, //
	// "cudnnSetLRNDescriptor":                              {}, //
	// "cudnnSetOpTensorDescriptor":                         {}, //
	// "cudnnSetPersistentRNNPlan":                          {}, //
	// "cudnnSetPooling2dDescriptor":                        {}, //
	// "cudnnSetPoolingNdDescriptor":                        {}, //
	// "cudnnSetRNNDescriptor":                              {}, //
	// "cudnnSetRNNDescriptor_v5":                           {}, //
	// "cudnnSetRNNDescriptor_v6":                           {}, //
	// "cudnnSetRNNMatrixMathType":                          {}, //
	// "cudnnSetReduceTensorDescriptor":                     {}, //
	// "cudnnSetSpatialTransformerNdDescriptor":             {}, //
	"cudnnSetStream": {},
	// "cudnnSetTensor":                                     {}, //
	// "cudnnSetTensor4dDescriptor":                         {}, //
	// "cudnnSetTensor4dDescriptorEx":                       {}, //
	// "cudnnSetTensorNdDescriptor":                         {}, //
	// "cudnnSetTensorNdDescriptorEx":                       {}, //
	// "cudnnSoftmaxBackward":                               {}, //
	// "cudnnSoftmaxForward":                                {}, //
	// "cudnnSpatialTfGridGeneratorBackward":                {}, //
	// "cudnnSpatialTfGridGeneratorForward":                 {}, //
	// "cudnnSpatialTfSamplerBackward":                      {}, //
	// "cudnnSpatialTfSamplerForward":                       {}, //
	// "cudnnTransformTensor":                               {}, //
}

func init() {
	fnNameMap = map[string]string{
		"cudnnActivationBackward":                            "ActivationBackward",
		"cudnnActivationForward":                             "ActivationForward",
		"cudnnAddTensor":                                     "AddTensor",
		"cudnnBatchNormalizationBackward":                    "BatchNormalizationBackward",
		"cudnnBatchNormalizationForwardInference":            "BatchNormalizationForwardInference",
		"cudnnBatchNormalizationForwardTraining":             "BatchNormalizationForwardTraining",
		"cudnnCTCLoss":                                       "CTCLoss",
		"cudnnConvolutionBackwardBias":                       "ConvolutionBackwardBias",
		"cudnnConvolutionBackwardData":                       "ConvolutionBackwardData",
		"cudnnConvolutionBackwardFilter":                     "ConvolutionBackwardFilter",
		"cudnnConvolutionBiasActivationForward":              "ConvolutionBiasActivationForward",
		"cudnnConvolutionForward":                            "ConvolutionForward",
		"cudnnCreate":                                        "Create",
		"cudnnCreateActivationDescriptor":                    "CreateActivationDescriptor",
		"cudnnCreateCTCLossDescriptor":                       "CreateCTCLossDescriptor",
		"cudnnCreateConvolutionDescriptor":                   "CreateConvolutionDescriptor",
		"cudnnCreateDropoutDescriptor":                       "CreateDropoutDescriptor",
		"cudnnCreateFilterDescriptor":                        "CreateFilterDescriptor",
		"cudnnCreateLRNDescriptor":                           "CreateLRNDescriptor",
		"cudnnCreateOpTensorDescriptor":                      "CreateOpTensorDescriptor",
		"cudnnCreatePersistentRNNPlan":                       "CreatePersistentRNNPlan",
		"cudnnCreatePoolingDescriptor":                       "CreatePoolingDescriptor",
		"cudnnCreateRNNDescriptor":                           "CreateRNNDescriptor",
		"cudnnCreateReduceTensorDescriptor":                  "CreateReduceTensorDescriptor",
		"cudnnCreateSpatialTransformerDescriptor":            "CreateSpatialTransformerDescriptor",
		"cudnnCreateTensorDescriptor":                        "CreateTensorDescriptor",
		"cudnnDeriveBNTensorDescriptor":                      "DeriveBNTensorDescriptor",
		"cudnnDestroy":                                       "Destroy",
		"cudnnDestroyActivationDescriptor":                   "DestroyActivationDescriptor",
		"cudnnDestroyCTCLossDescriptor":                      "DestroyCTCLossDescriptor",
		"cudnnDestroyConvolutionDescriptor":                  "DestroyConvolutionDescriptor",
		"cudnnDestroyDropoutDescriptor":                      "DestroyDropoutDescriptor",
		"cudnnDestroyFilterDescriptor":                       "DestroyFilterDescriptor",
		"cudnnDestroyLRNDescriptor":                          "DestroyLRNDescriptor",
		"cudnnDestroyOpTensorDescriptor":                     "DestroyOpTensorDescriptor",
		"cudnnDestroyPersistentRNNPlan":                      "DestroyPersistentRNNPlan",
		"cudnnDestroyPoolingDescriptor":                      "DestroyPoolingDescriptor",
		"cudnnDestroyRNNDescriptor":                          "DestroyRNNDescriptor",
		"cudnnDestroyReduceTensorDescriptor":                 "DestroyReduceTensorDescriptor",
		"cudnnDestroySpatialTransformerDescriptor":           "DestroySpatialTransformerDescriptor",
		"cudnnDestroyTensorDescriptor":                       "DestroyTensorDescriptor",
		"cudnnDivisiveNormalizationBackward":                 "DivisiveNormalizationBackward",
		"cudnnDivisiveNormalizationForward":                  "DivisiveNormalizationForward",
		"cudnnDropoutBackward":                               "DropoutBackward",
		"cudnnDropoutForward":                                "DropoutForward",
		"cudnnDropoutGetReserveSpaceSize":                    "DropoutGetReserveSpaceSize",
		"cudnnDropoutGetStatesSize":                          "DropoutGetStatesSize",
		"cudnnFindConvolutionBackwardDataAlgorithm":          "FindConvolutionBackwardDataAlgorithm",
		"cudnnFindConvolutionBackwardDataAlgorithmEx":        "FindConvolutionBackwardDataAlgorithmEx",
		"cudnnFindConvolutionBackwardFilterAlgorithm":        "FindConvolutionBackwardFilterAlgorithm",
		"cudnnFindConvolutionBackwardFilterAlgorithmEx":      "FindConvolutionBackwardFilterAlgorithmEx",
		"cudnnFindConvolutionForwardAlgorithm":               "FindConvolutionForwardAlgorithm",
		"cudnnFindConvolutionForwardAlgorithmEx":             "FindConvolutionForwardAlgorithmEx",
		"cudnnGetActivationDescriptor":                       "GetActivationDescriptor",
		"cudnnGetCTCLossDescriptor":                          "GetCTCLossDescriptor",
		"cudnnGetCTCLossWorkspaceSize":                       "GetCTCLossWorkspaceSize",
		"cudnnGetConvolution2dDescriptor":                    "GetConvolution2dDescriptor",
		"cudnnGetConvolution2dForwardOutputDim":              "GetConvolution2dForwardOutputDim",
		"cudnnGetConvolutionBackwardDataAlgorithm":           "GetConvolutionBackwardDataAlgorithm",
		"cudnnGetConvolutionBackwardDataAlgorithmMaxCount":   "GetConvolutionBackwardDataAlgorithmMaxCount",
		"cudnnGetConvolutionBackwardDataAlgorithm_v7":        "GetConvolutionBackwardDataAlgorithm_v7",
		"cudnnGetConvolutionBackwardDataWorkspaceSize":       "GetConvolutionBackwardDataWorkspaceSize",
		"cudnnGetConvolutionBackwardFilterAlgorithm":         "GetConvolutionBackwardFilterAlgorithm",
		"cudnnGetConvolutionBackwardFilterAlgorithmMaxCount": "GetConvolutionBackwardFilterAlgorithmMaxCount",
		"cudnnGetConvolutionBackwardFilterAlgorithm_v7":      "GetConvolutionBackwardFilterAlgorithm_v7",
		"cudnnGetConvolutionBackwardFilterWorkspaceSize":     "GetConvolutionBackwardFilterWorkspaceSize",
		"cudnnGetConvolutionForwardAlgorithm":                "GetConvolutionForwardAlgorithm",
		"cudnnGetConvolutionForwardAlgorithmMaxCount":        "GetConvolutionForwardAlgorithmMaxCount",
		"cudnnGetConvolutionForwardAlgorithm_v7":             "GetConvolutionForwardAlgorithm_v7",
		"cudnnGetConvolutionForwardWorkspaceSize":            "GetConvolutionForwardWorkspaceSize",
		"cudnnGetConvolutionGroupCount":                      "GetConvolutionGroupCount",
		"cudnnGetConvolutionMathType":                        "GetConvolutionMathType",
		"cudnnGetConvolutionNdDescriptor":                    "GetConvolutionNdDescriptor",
		"cudnnGetConvolutionNdForwardOutputDim":              "GetConvolutionNdForwardOutputDim",
		"cudnnGetCudartVersion":                              "GetCudartVersion",
		"cudnnGetDropoutDescriptor":                          "GetDropoutDescriptor",
		"cudnnGetErrorString":                                "GetErrorString",
		"cudnnGetFilter4dDescriptor":                         "GetFilter4dDescriptor",
		"cudnnGetFilterNdDescriptor":                         "GetFilterNdDescriptor",
		"cudnnGetLRNDescriptor":                              "GetLRNDescriptor",
		"cudnnGetOpTensorDescriptor":                         "GetOpTensorDescriptor",
		"cudnnGetPooling2dDescriptor":                        "GetPooling2dDescriptor",
		"cudnnGetPooling2dForwardOutputDim":                  "GetPooling2dForwardOutputDim",
		"cudnnGetPoolingNdDescriptor":                        "GetPoolingNdDescriptor",
		"cudnnGetPoolingNdForwardOutputDim":                  "GetPoolingNdForwardOutputDim",
		"cudnnGetProperty":                                   "GetProperty",
		"cudnnGetRNNDescriptor":                              "GetRNNDescriptor",
		"cudnnGetRNNLinLayerBiasParams":                      "GetRNNLinLayerBiasParams",
		"cudnnGetRNNLinLayerMatrixParams":                    "GetRNNLinLayerMatrixParams",
		"cudnnGetRNNParamsSize":                              "GetRNNParamsSize",
		"cudnnGetRNNTrainingReserveSize":                     "GetRNNTrainingReserveSize",
		"cudnnGetRNNWorkspaceSize":                           "GetRNNWorkspaceSize",
		"cudnnGetReduceTensorDescriptor":                     "GetReduceTensorDescriptor",
		"cudnnGetReductionIndicesSize":                       "GetReductionIndicesSize",
		"cudnnGetReductionWorkspaceSize":                     "GetReductionWorkspaceSize",
		"cudnnGetStream":                                     "GetStream",
		"cudnnGetTensor4dDescriptor":                         "GetTensor4dDescriptor",
		"cudnnGetTensorNdDescriptor":                         "GetTensorNdDescriptor",
		"cudnnGetTensorSizeInBytes":                          "GetTensorSizeInBytes",
		"cudnnGetVersion":                                    "GetVersion",
		"cudnnIm2Col":                                        "Im2Col",
		"cudnnLRNCrossChannelBackward":                       "LRNCrossChannelBackward",
		"cudnnLRNCrossChannelForward":                        "LRNCrossChannelForward",
		"cudnnOpTensor":                                      "OpTensor",
		"cudnnPoolingBackward":                               "PoolingBackward",
		"cudnnPoolingForward":                                "PoolingForward",
		"cudnnQueryRuntimeError":                             "QueryRuntimeError",
		"cudnnRNNBackwardData":                               "RNNBackwardData",
		"cudnnRNNBackwardWeights":                            "RNNBackwardWeights",
		"cudnnRNNForwardInference":                           "RNNForwardInference",
		"cudnnRNNForwardTraining":                            "RNNForwardTraining",
		"cudnnReduceTensor":                                  "ReduceTensor",
		"cudnnRestoreDropoutDescriptor":                      "RestoreDropoutDescriptor",
		"cudnnScaleTensor":                                   "ScaleTensor",
		"cudnnSetActivationDescriptor":                       "SetActivationDescriptor",
		"cudnnSetCTCLossDescriptor":                          "SetCTCLossDescriptor",
		"cudnnSetConvolution2dDescriptor":                    "SetConvolution2dDescriptor",
		"cudnnSetConvolutionGroupCount":                      "SetConvolutionGroupCount",
		"cudnnSetConvolutionMathType":                        "SetConvolutionMathType",
		"cudnnSetConvolutionNdDescriptor":                    "SetConvolutionNdDescriptor",
		"cudnnSetDropoutDescriptor":                          "SetDropoutDescriptor",
		"cudnnSetFilter4dDescriptor":                         "SetFilter4dDescriptor",
		"cudnnSetFilterNdDescriptor":                         "SetFilterNdDescriptor",
		"cudnnSetLRNDescriptor":                              "SetLRNDescriptor",
		"cudnnSetOpTensorDescriptor":                         "SetOpTensorDescriptor",
		"cudnnSetPersistentRNNPlan":                          "SetPersistentRNNPlan",
		"cudnnSetPooling2dDescriptor":                        "SetPooling2dDescriptor",
		"cudnnSetPoolingNdDescriptor":                        "SetPoolingNdDescriptor",
		"cudnnSetRNNDescriptor":                              "SetRNNDescriptor",
		"cudnnSetRNNDescriptor_v5":                           "SetRNNDescriptor_v5",
		"cudnnSetRNNDescriptor_v6":                           "SetRNNDescriptor_v6",
		"cudnnSetRNNMatrixMathType":                          "SetRNNMatrixMathType",
		"cudnnSetReduceTensorDescriptor":                     "SetReduceTensorDescriptor",
		"cudnnSetSpatialTransformerNdDescriptor":             "SetSpatialTransformerNdDescriptor",
		"cudnnSetStream":                                     "SetStream",
		"cudnnSetTensor":                                     "SetTensor",
		"cudnnSetTensor4dDescriptor":                         "SetTensor4dDescriptor",
		"cudnnSetTensor4dDescriptorEx":                       "SetTensor4dDescriptorEx",
		"cudnnSetTensorNdDescriptor":                         "SetTensorNdDescriptor",
		"cudnnSetTensorNdDescriptorEx":                       "SetTensorNdDescriptorEx",
		"cudnnSoftmaxBackward":                               "SoftmaxBackward",
		"cudnnSoftmaxForward":                                "SoftmaxForward",
		"cudnnSpatialTfGridGeneratorBackward":                "SpatialTfGridGeneratorBackward",
		"cudnnSpatialTfGridGeneratorForward":                 "SpatialTfGridGeneratorForward",
		"cudnnSpatialTfSamplerBackward":                      "SpatialTfSamplerBackward",
		"cudnnSpatialTfSamplerForward":                       "SpatialTfSamplerForward",
		"cudnnTransformTensor":                               "TransformTensor",
	}
	enumMappings = map[string]string{
		"cudnnActivationMode_t":                 "ActivationMode",
		"cudnnBatchNormMode_t":                  "BatchNormMode",
		"cudnnCTCLossAlgo_t":                    "CTCLossAlgo",
		"cudnnConvolutionBwdDataAlgo_t":         "ConvolutionBwdDataAlgo",
		"cudnnConvolutionBwdDataPreference_t":   "ConvolutionBwdDataPreference", //
		"cudnnConvolutionBwdDataPreference_t":   "ConvolutionPreference",
		"cudnnConvolutionBwdFilterAlgo_t":       "ConvolutionBwdFilterAlgo",
		"cudnnConvolutionBwdFilterPreference_t": "ConvolutionBwdFilterPreference", //
		"cudnnConvolutionBwdFilterPreference_t": "ConvolutionPreference",
		"cudnnConvolutionFwdAlgo_t":             "ConvolutionFwdAlgo",
		"cudnnConvolutionFwdPreference_t":       "ConvolutionFwdPreference", //
		"cudnnConvolutionFwdPreference_t":       "ConvolutionPreference",
		"cudnnConvolutionMode_t":                "ConvolutionMode",
		"cudnnDataType_t":                       "DataType",
		"cudnnDeterminism_t":                    "Determinism",
		"cudnnDirectionMode_t":                  "DirectionMode",
		"cudnnDivNormMode_t":                    "DivNormMode",
		"cudnnErrQueryMode_t":                   "ErrQueryMode",
		"cudnnIndicesType_t":                    "IndicesType",
		"cudnnLRNMode_t":                        "LRNMode",
		"cudnnMathType_t":                       "MathType",
		"cudnnNanPropagation_t":                 "NanPropagation",
		"cudnnOpTensorOp_t":                     "OpTensorOp",
		"cudnnPoolingMode_t":                    "PoolingMode",
		"cudnnRNNAlgo_t":                        "RNNAlgo",
		"cudnnRNNInputMode_t":                   "RNNInputMode",
		"cudnnRNNMode_t":                        "RNNMode",
		"cudnnReduceTensorIndices_t":            "ReduceTensorIndices",
		"cudnnReduceTensorOp_t":                 "ReduceTensorOp",
		"cudnnSamplerType_t":                    "SamplerType",
		"cudnnSoftmaxAlgorithm_t":               "SoftmaxAlgorithm",
		"cudnnSoftmaxMode_t":                    "SoftmaxMode",
		"cudnnTensorFormat_t":                   "TensorFormat",
		"cudnnStatus_t":                         "Status",
	}

	alphaBetas = map[string]map[int]string{
		"cudnnActivationBackward":                 {9: "beta", 2: "alpha"},
		"cudnnActivationForward":                  {5: "beta", 2: "alpha"},
		"cudnnAddTensor":                          {4: "beta", 1: "alpha"},
		"cudnnBatchNormalizationBackward":         {5: "betaParamDiff", 4: "alphaParamDiff", 3: "betaDataDiff", 2: "alphaDataDiff"},
		"cudnnBatchNormalizationForwardInference": {3: "beta", 2: "alpha"},
		"cudnnBatchNormalizationForwardTraining":  {3: "beta", 2: "alpha"},
		"cudnnConvolutionBackwardBias":            {4: "beta", 1: "alpha"},
		"cudnnConvolutionBackwardData":            {10: "beta", 1: "alpha"},
		"cudnnConvolutionBackwardFilter":          {10: "beta", 1: "alpha"},
		"cudnnConvolutionBiasActivationForward":   {10: "alpha2", 1: "alpha1"},
		"cudnnConvolutionForward":                 {10: "beta", 1: "alpha"},
		"cudnnDivisiveNormalizationBackward":      {10: "beta", 3: "alpha"},
		"cudnnDivisiveNormalizationForward":       {9: "beta", 3: "alpha"},
		"cudnnLRNCrossChannelBackward":            {10: "beta", 3: "alpha"},
		"cudnnLRNCrossChannelForward":             {6: "beta", 3: "alpha"},
		"cudnnOpTensor":                           {8: "beta", 5: "alpha2", 2: "alpha1"},
		"cudnnPoolingBackward":                    {9: "beta", 2: "alpha"},
		"cudnnPoolingForward":                     {5: "beta", 2: "alpha"},
		"cudnnReduceTensor":                       {9: "beta", 6: "alpha"},
		"cudnnScaleTensor":                        {3: "alpha"},
		"cudnnSoftmaxBackward":                    {8: "beta", 3: "alpha"},
		"cudnnSoftmaxForward":                     {6: "beta", 3: "alpha"},
		"cudnnSpatialTfSamplerBackward":           {5: "beta", 2: "alpha"},
		"cudnnSpatialTfSamplerForward":            {6: "beta", 2: "alpha"},
		"cudnnTransformTensor":                    {4: "beta", 1: "alpha"},
	}

	creations = map[string][]string{
		"cudnnActivationDescriptor_t":         {"cudnnCreateActivationDescriptor"},
		"cudnnCTCLossDescriptor_t":            {"cudnnCreateCTCLossDescriptor"},
		"cudnnDropoutDescriptor_t":            {"cudnnCreateDropoutDescriptor"},
		"cudnnFilterDescriptor_t":             {"cudnnCreateFilterDescriptor"},
		"cudnnHandle_t":                       {"cudnnCreate"},
		"cudnnLRNDescriptor_t":                {"cudnnCreateLRNDescriptor"},
		"cudnnOpTensorDescriptor_t":           {"cudnnCreateOpTensorDescriptor"},
		"cudnnPersistentRNNPlan_t":            {"cudnnCreatePersistentRNNPlan"},
		"cudnnPoolingDescriptor_t":            {"cudnnCreatePoolingDescriptor"},
		"cudnnRNNDescriptor_t":                {"cudnnCreateRNNDescriptor"},
		"cudnnReduceTensorDescriptor_t":       {"cudnnCreateReduceTensorDescriptor"},
		"cudnnSpatialTransformerDescriptor_t": {"cudnnCreateSpatialTransformerDescriptor"},
		"cudnnTensorDescriptor_t":             {"cudnnCreateTensorDescriptor"},
		"cudnnConvolutionDescriptor_t":        {"cudnnCreateConvolutionDescriptor"},
	}

	setFns = map[string][]string{
		"cudaStream_t":                        {"cudnnSetStream"},
		"cudnnActivationDescriptor_t":         {"cudnnSetActivationDescriptor"},
		"cudnnCTCLossDescriptor_t":            {"cudnnSetCTCLossDescriptor"},
		"cudnnConvolutionDescriptor_t":        {"cudnnSetConvolutionMathType", "cudnnSetConvolutionGroupCount", "cudnnSetConvolution2dDescriptor", "cudnnSetConvolutionNdDescriptor"},
		"cudnnDropoutDescriptor_t":            {"cudnnSetDropoutDescriptor"},
		"cudnnFilterDescriptor_t":             {"cudnnSetFilter4dDescriptor", "cudnnSetFilterNdDescriptor"},
		"cudnnLRNDescriptor_t":                {"cudnnSetLRNDescriptor"},
		"cudnnPoolingDescriptor_t":            {"cudnnSetPooling2dDescriptor", "cudnnSetPoolingNdDescriptor"},
		"cudnnRNNDescriptor_t":                {"cudnnSetPersistentRNNPlan", "cudnnSetRNNDescriptor", "cudnnSetRNNMatrixMathType", "cudnnSetRNNDescriptor_v6", "cudnnSetRNNDescriptor_v5"},
		"cudnnReduceTensorDescriptor_t":       {"cudnnSetReduceTensorDescriptor"},
		"cudnnSpatialTransformerDescriptor_t": {"cudnnSetSpatialTransformerNdDescriptor"},
		"cudnnTensorDescriptor_t":             {"cudnnSetTensor4dDescriptor", "cudnnSetTensor4dDescriptorEx", "cudnnSetTensorNdDescriptor", "cudnnSetTensorNdDescriptorEx", "cudnnSetTensor"},
		"cudnnOpTensorDescriptor_t":           {"cudnnSetOpTensorDescriptor"},
	}

	destructions = map[string][]string{
		"cudnnActivationDescriptor_t":         {"cudnnDestroyActivationDescriptor"},
		"cudnnCTCLossDescriptor_t":            {"cudnnDestroyCTCLossDescriptor"},
		"cudnnConvolutionDescriptor_t":        {"cudnnDestroyConvolutionDescriptor"},
		"cudnnDropoutDescriptor_t":            {"cudnnDestroyDropoutDescriptor"},
		"cudnnFilterDescriptor_t":             {"cudnnDestroyFilterDescriptor"},
		"cudnnHandle_t":                       {"cudnnDestroy"},
		"cudnnLRNDescriptor_t":                {"cudnnDestroyLRNDescriptor"},
		"cudnnOpTensorDescriptor_t":           {"cudnnDestroyOpTensorDescriptor"},
		"cudnnPersistentRNNPlan_t":            {"cudnnDestroyPersistentRNNPlan"},
		"cudnnPoolingDescriptor_t":            {"cudnnDestroyPoolingDescriptor"},
		"cudnnRNNDescriptor_t":                {"cudnnDestroyRNNDescriptor"},
		"cudnnSpatialTransformerDescriptor_t": {"cudnnDestroySpatialTransformerDescriptor"},
		"cudnnTensorDescriptor_t":             {"cudnnDestroyTensorDescriptor"},
		"cudnnReduceTensorDescriptor_t":       {"cudnnDestroyReduceTensorDescriptor"},
	}

	methods = map[string][]string{
		"cudnnActivationDescriptor_t":   {"cudnnGetActivationDescriptor"},
		"cudnnCTCLossDescriptor_t":      {"cudnnGetCTCLossDescriptor"},
		"cudnnDropoutDescriptor_t":      {"cudnnRestoreDropoutDescriptor", "cudnnGetDropoutDescriptor"},
		"cudnnFilterDescriptor_t":       {"cudnnGetFilter4dDescriptor", "cudnnGetFilterNdDescriptor"},
		"cudnnHandle_t":                 {"cudnnTransformTensor", "cudnnAddTensor", "cudnnOpTensor", "cudnnGetReductionIndicesSize", "cudnnGetReductionWorkspaceSize", "cudnnReduceTensor", "cudnnScaleTensor", "cudnnFindConvolutionForwardAlgorithm", "cudnnFindConvolutionForwardAlgorithmEx", "cudnnGetConvolutionForwardAlgorithm", "cudnnGetConvolutionForwardAlgorithm_v7", "cudnnGetConvolutionForwardWorkspaceSize", "cudnnConvolutionForward", "cudnnConvolutionBiasActivationForward", "cudnnConvolutionBackwardBias", "cudnnGetConvolutionBackwardFilterAlgorithmMaxCount", "cudnnFindConvolutionBackwardFilterAlgorithm", "cudnnFindConvolutionBackwardFilterAlgorithmEx", "cudnnGetConvolutionBackwardFilterAlgorithm", "cudnnGetConvolutionBackwardFilterAlgorithm_v7", "cudnnGetConvolutionBackwardFilterWorkspaceSize", "cudnnConvolutionBackwardFilter", "cudnnGetConvolutionBackwardDataAlgorithmMaxCount", "cudnnFindConvolutionBackwardDataAlgorithm", "cudnnFindConvolutionBackwardDataAlgorithmEx", "cudnnGetConvolutionBackwardDataAlgorithm", "cudnnGetConvolutionBackwardDataAlgorithm_v7", "cudnnGetConvolutionBackwardDataWorkspaceSize", "cudnnConvolutionBackwardData", "cudnnIm2Col", "cudnnSoftmaxForward", "cudnnSoftmaxBackward", "cudnnPoolingForward", "cudnnPoolingBackward", "cudnnActivationForward", "cudnnActivationBackward", "cudnnLRNCrossChannelForward", "cudnnLRNCrossChannelBackward", "cudnnDivisiveNormalizationForward", "cudnnDivisiveNormalizationBackward", "cudnnBatchNormalizationForwardTraining", "cudnnBatchNormalizationForwardInference", "cudnnBatchNormalizationBackward", "cudnnSpatialTfGridGeneratorForward", "cudnnSpatialTfGridGeneratorBackward", "cudnnSpatialTfSamplerForward", "cudnnSpatialTfSamplerBackward", "cudnnDropoutGetStatesSize", "cudnnDropoutForward", "cudnnDropoutBackward", "cudnnGetRNNDescriptor", "cudnnGetRNNWorkspaceSize", "cudnnGetRNNTrainingReserveSize", "cudnnGetRNNParamsSize", "cudnnGetRNNLinLayerMatrixParams", "cudnnGetRNNLinLayerBiasParams", "cudnnRNNForwardInference", "cudnnRNNForwardTraining", "cudnnRNNBackwardData", "cudnnRNNBackwardWeights", "cudnnCTCLoss", "cudnnGetCTCLossWorkspaceSize"},
		"cudnnLRNDescriptor_t":          {"cudnnGetLRNDescriptor"},
		"cudnnPoolingDescriptor_t":      {"cudnnGetPooling2dDescriptor", "cudnnGetPoolingNdDescriptor", "cudnnGetPoolingNdForwardOutputDim", "cudnnGetPooling2dForwardOutputDim"},
		"cudnnReduceTensorDescriptor_t": {"cudnnGetReduceTensorDescriptor"},
		"cudnnTensorDescriptor_t":       {"cudnnGetTensor4dDescriptor", "cudnnGetTensorNdDescriptor", "cudnnGetTensorSizeInBytes", "cudnnDeriveBNTensorDescriptor", "cudnnDropoutGetReserveSpaceSize"},
		"cudnnOpTensorDescriptor_t":     {"cudnnGetOpTensorDescriptor"},
	}
}
