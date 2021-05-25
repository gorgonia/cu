## Potential Nils ##
These functions have a `*T` return value, but a possible null exception error might happen

* `NewCTCLoss`
* `NewDropoutWithContext`

## Unconverted C Functions ##

* `cudnnAdvInferVersionCheck`
* `cudnnAdvTrainVersionCheck`
* `cudnnBackendExecute`
* `cudnnBackendFinalize`
* `cudnnBackendGetAttribute`
* `cudnnBackendInitialize`
* `cudnnBatchNormalizationBackwardEx`
* `cudnnBatchNormalizationForwardTrainingEx`
* `cudnnBuildRNNDynamic`
* `cudnnCTCLoss_v8`
* `cudnnCnnInferVersionCheck`
* `cudnnCnnTrainVersionCheck`
* `cudnnCopyAlgorithmDescriptor`
* `cudnnCreateCTCLossDescriptor`
* `cudnnCreateConvolutionDescriptor`
* `cudnnCreateFusedOpsPlan`
* `cudnnCreateRNNDescriptor`
* `cudnnDeriveNormTensorDescriptor`
* `cudnnDestroyFusedOpsPlan`
* `cudnnFindRNNBackwardDataAlgorithmEx`
* `cudnnFindRNNBackwardWeightsAlgorithmEx`
* `cudnnFindRNNForwardInferenceAlgorithmEx`
* `cudnnFindRNNForwardTrainingAlgorithmEx`
* `cudnnFusedOpsExecute`
* `cudnnGetActivationDescriptor`
* `cudnnGetAlgorithmDescriptor`
* `cudnnGetAlgorithmPerformance`
* `cudnnGetAlgorithmSpaceSize`
* `cudnnGetAttnDescriptor`
* `cudnnGetBatchNormalizationBackwardExWorkspaceSize`
* `cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize`
* `cudnnGetBatchNormalizationTrainingExReserveSpaceSize`
* `cudnnGetCTCLossDescriptor`
* `cudnnGetCTCLossDescriptorEx`
* `cudnnGetCTCLossDescriptor_v8`
* `cudnnGetCTCLossWorkspaceSize`
* `cudnnGetCTCLossWorkspaceSize_v8`
* `cudnnGetCallback`
* `cudnnGetConvolution2dDescriptor`
* `cudnnGetConvolutionBackwardDataAlgorithmMaxCount`
* `cudnnGetConvolutionBackwardDataAlgorithm_v7`
* `cudnnGetConvolutionBackwardDataWorkspaceSize`
* `cudnnGetConvolutionBackwardFilterAlgorithmMaxCount`
* `cudnnGetConvolutionBackwardFilterAlgorithm_v7`
* `cudnnGetConvolutionBackwardFilterWorkspaceSize`
* `cudnnGetConvolutionForwardAlgorithmMaxCount`
* `cudnnGetConvolutionForwardAlgorithm_v7`
* `cudnnGetConvolutionForwardWorkspaceSize`
* `cudnnGetConvolutionGroupCount`
* `cudnnGetConvolutionMathType`
* `cudnnGetConvolutionNdDescriptor`
* `cudnnGetConvolutionReorderType`
* `cudnnGetCudartVersion`
* `cudnnGetDropoutDescriptor`
* `cudnnGetErrorString`
* `cudnnGetFilter4dDescriptor`
* `cudnnGetFilterNdDescriptor`
* `cudnnGetFilterSizeInBytes`
* `cudnnGetFoldedConvBackwardDataDescriptors`
* `cudnnGetFusedOpsConstParamPackAttribute`
* `cudnnGetFusedOpsVariantParamPackAttribute`
* `cudnnGetLRNDescriptor`
* `cudnnGetMultiHeadAttnBuffers`
* `cudnnGetMultiHeadAttnWeights`
* `cudnnGetNormalizationBackwardWorkspaceSize`
* `cudnnGetNormalizationForwardTrainingWorkspaceSize`
* `cudnnGetNormalizationTrainingReserveSpaceSize`
* `cudnnGetOpTensorDescriptor`
* `cudnnGetPooling2dDescriptor`
* `cudnnGetPoolingNdDescriptor`
* `cudnnGetProperty`
* `cudnnGetRNNBackwardDataAlgorithmMaxCount`
* `cudnnGetRNNBackwardWeightsAlgorithmMaxCount`
* `cudnnGetRNNBiasMode`
* `cudnnGetRNNDataDescriptor`
* `cudnnGetRNNDescriptor_v6`
* `cudnnGetRNNDescriptor_v8`
* `cudnnGetRNNForwardInferenceAlgorithmMaxCount`
* `cudnnGetRNNForwardTrainingAlgorithmMaxCount`
* `cudnnGetRNNMatrixMathType`
* `cudnnGetRNNPaddingMode`
* `cudnnGetRNNProjectionLayers`
* `cudnnGetRNNTempSpaceSizes`
* `cudnnGetRNNWeightParams`
* `cudnnGetRNNWeightSpaceSize`
* `cudnnGetReduceTensorDescriptor`
* `cudnnGetSeqDataDescriptor`
* `cudnnGetStream`
* `cudnnGetTensor4dDescriptor`
* `cudnnGetTensorNdDescriptor`
* `cudnnGetTensorSizeInBytes`
* `cudnnGetTensorTransformDescriptor`
* `cudnnGetVersion`
* `cudnnInitTransformDest`
* `cudnnMakeFusedOpsPlan`
* `cudnnMultiHeadAttnBackwardData`
* `cudnnMultiHeadAttnBackwardWeights`
* `cudnnMultiHeadAttnForward`
* `cudnnNormalizationBackward`
* `cudnnNormalizationForwardInference`
* `cudnnNormalizationForwardTraining`
* `cudnnOpsInferVersionCheck`
* `cudnnOpsTrainVersionCheck`
* `cudnnQueryRuntimeError`
* `cudnnRNNBackwardDataEx`
* `cudnnRNNBackwardData_v8`
* `cudnnRNNBackwardWeightsEx`
* `cudnnRNNBackwardWeights_v8`
* `cudnnRNNForward`
* `cudnnRNNForwardInferenceEx`
* `cudnnRNNForwardTrainingEx`
* `cudnnRNNGetClip`
* `cudnnRNNGetClip_v8`
* `cudnnRNNSetClip`
* `cudnnRNNSetClip_v8`
* `cudnnReorderFilterAndBias`
* `cudnnRestoreAlgorithm`
* `cudnnSaveAlgorithm`
* `cudnnSetCTCLossDescriptor`
* `cudnnSetCTCLossDescriptorEx`
* `cudnnSetCTCLossDescriptor_v8`
* `cudnnSetCallback`
* `cudnnSetConvolution2dDescriptor`
* `cudnnSetConvolutionGroupCount`
* `cudnnSetConvolutionMathType`
* `cudnnSetConvolutionNdDescriptor`
* `cudnnSetConvolutionReorderType`
* `cudnnSetRNNAlgorithmDescriptor`
* `cudnnSetRNNBiasMode`
* `cudnnSetRNNDescriptor_v6`
* `cudnnSetRNNDescriptor_v8`
* `cudnnSetRNNPaddingMode`
* `cudnnSetRNNProjectionLayers`
* `cudnnSetStream`
* `cudnnSetTensor`
* `cudnnTransformFilter`
* `cudnnTransformTensorEx`

## Unconverted/Unused C Types ##

* `cudnnAlgorithm_t`
* `cudnnCallback_t`
* `cudnnConvolutionBwdDataAlgoPerf_t`
* `cudnnConvolutionBwdFilterAlgoPerf_t`
* `cudnnConvolutionFwdAlgoPerf_t`
* `cudnnDebug_t`
* `cudnnFusedOpsPlan_t`
* `cudnnRuntimeTag_t`
* `cudnnStatus_t`


# Build Errors/TODO
```
# gorgonia.org/cu/dnn
./algorithm.go:9:2: type struct {} is not an expression
./generated_API.go:13:54: *Activation is not a type
./generated_API.go:34:53: *Activation is not a type
./generated_API.go:147:82: undefined: TODO
./generated_API.go:214: *Activation is not a type
./generated_API.go:308:61: cannot use _Ctype_ulong(sizeInBytes) (type _Ctype_ulong) as type *_Ctype_ulong in assignment
./generated_API.go:312:143: cannot use _Ctype_int(returnedAlgoCount) (type _Ctype_int) as type *_Ctype_int in assignment
./generated_API.go:316:203: cannot use _Ctype_int(returnedAlgoCount) (type _Ctype_int) as type *_Ctype_int in assignment
./generated_API.go:320:145: cannot use _Ctype_int(returnedAlgoCount) (type _Ctype_int) as type *_Ctype_int in assignment
./generated_API.go:324:204: cannot use _Ctype_int(returnedAlgoCount) (type _Ctype_int) as type *_Ctype_int in assignment
./generated_API.go:328:138: cannot use _Ctype_int(returnedAlgoCount) (type _Ctype_int) as type *_Ctype_int in assignment
./generated_API.go:328:138: cannot use perfResults.internal (type _Ctype_cudnnConvolutionFwdAlgo_t) as type *_Ctype_struct_cudnnConvolutionFwdAlgoPerfStruct in assignment
./generated_API.go:334:181: undefined: TODO
./generated_API.go:332:195: cannot use _Ctype_int(returnedAlgoCount) (type _Ctype_int) as type *_Ctype_int in assignment
./generated_API.go:332:195: cannot use perfResults.internal (type _Ctype_cudnnConvolutionFwdAlgo_t) as type *_Ctype_struct_cudnnConvolutionFwdAlgoPerfStruct in assignment
./generated_API.go:338:181: undefined: TODO
./generated_API.go:344:91: cannot use _Ctype_ulong(sizeInBytes) (type _Ctype_ulong) as type *_Ctype_ulong in assignment
./generated_API.go:348:118: cannot use _Ctype_ulong(sizeInBytes) (type _Ctype_ulong) as type *_Ctype_ulong in assignment
./generated_API.go:348:156: cannot use _cgo3 (type _Ctype_cudnnTensorDescriptor_t) as type *_Ctype_cudnnTensorDescriptor_t in argument to _Cfunc_cudnnGetRNNTrainingReserveSize
./generated_API.go:352:112: cannot use _Ctype_ulong(sizeInBytes) (type _Ctype_ulong) as type *_Ctype_ulong in assignment
./generated_API.go:352:150: cannot use _cgo3 (type _Ctype_cudnnTensorDescriptor_t) as type *_Ctype_cudnnTensorDescriptor_t in argument to _Cfunc_cudnnGetRNNWorkspaceSize
./generated_API.go:356:123: cannot use _Ctype_ulong(sizeInBytes) (type _Ctype_ulong) as type *_Ctype_ulong in assignment
./generated_API.go:360:125: cannot use _Ctype_ulong(sizeInBytes) (type _Ctype_ulong) as type *_Ctype_ulong in assignment
./generated_API.go:477: cannot use _cgo3 (type _Ctype_cudnnTensorDescriptor_t) as type *_Ctype_cudnnTensorDescriptor_t in argument to _Cfunc_cudnnRNNBackwardData
./generated_API.go:477: cannot use _cgo5 (type _Ctype_cudnnTensorDescriptor_t) as type *_Ctype_cudnnTensorDescriptor_t in argument to _Cfunc_cudnnRNNBackwardData
./generated_API.go:477: cannot use _cgo17 (type _Ctype_cudnnTensorDescriptor_t) as type *_Ctype_cudnnTensorDescriptor_t in argument to _Cfunc_cudnnRNNBackwardData
./generated_API.go:481: cannot use _cgo3 (type _Ctype_cudnnTensorDescriptor_t) as type *_Ctype_cudnnTensorDescriptor_t in argument to _Cfunc_cudnnRNNBackwardWeights
./generated_API.go:481: cannot use _cgo7 (type _Ctype_cudnnTensorDescriptor_t) as type *_Ctype_cudnnTensorDescriptor_t in argument to _Cfunc_cudnnRNNBackwardWeights
./generated_API.go:485: cannot use _cgo3 (type _Ctype_cudnnTensorDescriptor_t) as type *_Ctype_cudnnTensorDescriptor_t in argument to _Cfunc_cudnnRNNForwardInference
./generated_API.go:485: cannot use _cgo11 (type _Ctype_cudnnTensorDescriptor_t) as type *_Ctype_cudnnTensorDescriptor_t in argument to _Cfunc_cudnnRNNForwardInference
./generated_API.go:489: cannot use _cgo3 (type _Ctype_cudnnTensorDescriptor_t) as type *_Ctype_cudnnTensorDescriptor_t in argument to _Cfunc_cudnnRNNForwardTraining
./generated_API.go:489: cannot use _cgo11 (type _Ctype_cudnnTensorDescriptor_t) as type *_Ctype_cudnnTensorDescriptor_t in argument to _Cfunc_cudnnRNNForwardTraining
./generated_API.go:645:61: undefined: xDesc
./generated_API.go:649:67: cannot use _Ctype_ulong(sizeInBytes) (type _Ctype_ulong) as type *_Ctype_ulong in assignment
./generated_algorithmdescriptor.go:13:12: undefined: TODO
./generated_algorithmdescriptor.go:17:39: undefined: TODO
./generated_algorithmperformance.go:14:11: undefined: Status
./generated_algorithmperformance.go:20:68: undefined: Status
./generated_algorithmperformance.go:22:52: not enough arguments in call to _Cfunc_cudnnCreateAlgorithmPerformance
	have (*_Ctype_cudnnAlgorithmPerformance_t)
	want (*_Ctype_cudnnAlgorithmPerformance_t, _Ctype_int)
./generated_algorithmperformance.go:42:76: a.algoPerf undefined (type *AlgorithmPerformance has no field or method algoPerf)
./generated_algorithmperformance.go:48:41: undefined: Status
./generated_algorithmperformance.go:57:36: not enough arguments in call to _Cfunc_cudnnDestroyAlgorithmPerformance
	have (_Ctype_cudnnAlgorithmPerformance_t)
	want (*_Ctype_cudnnAlgorithmPerformance_t, _Ctype_int)
./generated_backend.go:25:49: not enough arguments in call to _Cfunc_cudnnBackendCreateDescriptor
	have (*_Ctype_cudnnBackendDescriptor_t)
	want (_Ctype_cudnnBackendDescriptorType_t, *_Ctype_cudnnBackendDescriptor_t)
./generated_enums.go:259:2: Activation redeclared in this block
	previous declaration at ./generated_activation.go:10:6
./generated_enums.go:563:2: Activation redeclared in this block
	previous declaration at ./generated_enums.go:259:31
./generated_enums.go:574:2: Activation redeclared in this block
	previous declaration at ./generated_enums.go:563:24
./generated_enums.go:575:2: AddActivation redeclared in this block
	previous declaration at ./generated_enums.go:260:31
./generated_enums.go:595:6: PointwiseMode redeclared in this block
	previous declaration at ./generated_enums.go:25:63
./generated_enums.go:598:2: Add redeclared in this block
	previous declaration at ./generated_enums.go:584:20
./generated_enums.go:599:2: Mul redeclared in this block
	previous declaration at ./generated_enums.go:585:20
./generated_enums.go:600:2: Min redeclared in this block
	previous declaration at ./generated_enums.go:586:20
./generated_enums.go:601:2: Max redeclared in this block
	previous declaration at ./generated_enums.go:587:20
./generated_enums.go:602:2: Sqrt redeclared in this block
	previous declaration at ./generated_enums.go:588:20
./generated_enums.go:627:2: Standard redeclared in this block
	previous declaration at ./generated_enums.go:553:22
./generated_enums.go:630:2: Count redeclared in this block
	previous declaration at ./generated_enums.go:221:38
./generated_enums.go:651:2: None redeclared in this block
	previous declaration at ./generated_enums.go:501:34
./generated_enums.go:776:2: Channel redeclared in this block
	previous declaration at ./generated_enums.go:564:24
./generated_enums.go:796:2: Add redeclared in this block
	previous declaration at ./generated_enums.go:598:29
./generated_enums_strings.go:20:2: type PointwiseMode is not an expression
./generated_enums_strings.go:195:2: cannot use Count (type RNNAlgo) as type BackendLayoutType in map key
./generated_enums_strings.go:222:2: cannot use Activation (type NormOps) as type BatchNormOps in map key
./generated_enums_strings.go:223:2: cannot use AddActivation (type NormOps) as type BatchNormOps in map key
./generated_enums_strings.go:410:2: cannot use None (type RNNClipMode) as type LossNormalizationMode in map key
./generated_enums_strings.go:446:2: cannot use Standard (type RNNAlgo) as type NormAlgo in map key
./generated_enums_strings.go:453:2: cannot use Activation (type NormOps) as type NormMode in map key
./generated_enums_strings.go:454:2: cannot use Channel (type SoftmaxMode) as type NormMode in map key
./generated_enums_strings.go:468:2: cannot use Add (type WgradMode) as type OpTensorOp in map key
./generated_enums_strings.go:469:2: cannot use Mul (type PointwiseMode) as type OpTensorOp in map key
./generated_enums_strings.go:470:2: cannot use Min (type PointwiseMode) as type OpTensorOp in map key
./generated_enums_strings.go:471:2: cannot use Max (type PointwiseMode) as type OpTensorOp in map key
./generated_enums_strings.go:472:2: cannot use Sqrt (type PointwiseMode) as type OpTensorOp in map key
./generated_fusedopconsts.go:20:54: not enough arguments in call to _Cfunc_cudnnCreateFusedOpsConstParamPack
	have (*_Ctype_cudnnFusedOpsConstParamPack_t)
	want (*_Ctype_cudnnFusedOpsConstParamPack_t, _Ctype_cudnnFusedOps_t)
./generated_fusedopconsts.go:24:92: param.Pointer undefined (type Memory has no field or method Pointer)
./generated_fusedopvariantparams.go:23:56: not enough arguments in call to _Cfunc_cudnnCreateFusedOpsVariantParamPack
	have (*_Ctype_cudnnFusedOpsVariantParamPack_t)
	want (*_Ctype_cudnnFusedOpsVariantParamPack_t, _Ctype_cudnnFusedOps_t)
./generated_fusedopvariantparams.go:27:107: ptr.Pointer undefined (type Memory has no field or method Pointer)
./generated_seqdata.go:38:97: cannot use axes.C() (type _Ctype_cudnnSeqDataAxis_t) as type *_Ctype_cudnnSeqDataAxis_t in assignment
./generated_tensortransform.go:51:77: cannot use t.internal (type _Ctype_cudnnTensorTransformDescriptor_t) as type *_Ctype_cudnnTensorTransformDescriptor_t in return argument
./optensor.go:52:24: *TensorDescriptor is not a type
./optensor.go:53:24: *TensorDescriptor is not a type
./optensor.go:54:22: *TensorDescriptor is not a type
./optensor.go:106:26: cData.Pointer undefined (type Memory has no field or method Pointer)
./optensor.go:104:33: aData.Pointer undefined (type Memory has no field or method Pointer)
./optensor.go:105:28: bData.Pointer undefined (type Memory has no field or method Pointer)
./pooling.go:95:37: *TensorDescriptor is not a type
./pooling.go:103:41: *TensorDescriptor is not a type
./tensor.go:11:6: TensorDescriptor redeclared in this block
	previous declaration at ./generated_enums.go:163:71
```
