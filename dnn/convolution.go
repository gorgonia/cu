package cudnn

// #include <cudnn_v7.h>
import "C"

type ConvolutionType byte

const (
	Fwd ConvolutionType = 0
	//  combination of these indicate are used for bwd functions
	BwdFilter = 1 << 7
	BwdData   = 1<<7 | 1<<6
)

// ConvolutionPreference represents the preference for the algorithm to work with.
//
// Coincidentally ALL the preferences share the same three enum numbers, so we roll them into
// one Go type.
type ConvolutionPreference byte

const (
	NoWorkspace ConvolutionPreference = iota
	PreferFastest
	SpecifyWorkspaceLimit
)

// MakeConvolutionPreference allows the creation of a tagged preference - whether it's fwd, bwd or data or filter
func MakeConvolutionPreference(t ConvolutionType, pref ConvolutionPreference) ConvolutionPreference {
	return ConvolutionPreference(byte(t) | byte(pref))
}

func (c ConvolutionPreference) IsBwd() bool    { return byte(c)>>7&byte(1) == 1 }
func (c ConvolutionPreference) IsFwd() bool    { return !c.IsBwd() }
func (c ConvolutionPreference) IsData() bool   { return byte(c)>>6&byte(1) == 1 }
func (c ConvolutionPreference) IsFilter() bool { return !c.IsData() }
func (c ConvolutionPreference) Pref() ConvolutionPreference {
	retVal := byte(c) & ^(byte(1) << 6) & ^(byte(1) << 7)
	return ConvolutionPreference(retVal)
}

// C returns the C representation.
// Note this is only OK to do because all the ConvolutionPreferences share the ssame enum ints.
//
// This may break
func (c ConvolutionPreference) C() C.int { return C.int(int(c.Pref())) }

// type ConvolutionFwdPreference int

// const (
// 	NoWorkspace           ConvolutionFwdPreference = C.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE
// 	PreferFastest         ConvolutionFwdPreference = C.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST
// 	SpecifyWorkspaceLimit ConvolutionFwdPreference = C.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
// )

type ConvolutionFwdAlgo int

const (
	ConvolutionFwdAlgoImplicitGemm        ConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
	ConvolutionFwdAlgoImplicitPrecompGemm ConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
	ConvolutionFwdAlgoGemm                ConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_GEMM
	ConvolutionFwdAlgoDirect              ConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
	ConvolutionFwdAlgoFFT                 ConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_FFT
	ConvolutionFwdAlgoFFTTiling           ConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
	ConvolutionFwdAlgoWinograd            ConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
	ConvolutionFwdAlgoWinogradNonfused    ConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
	ConvolutionFwdAlgoCount               ConvolutionFwdAlgo = C.CUDNN_CONVOLUTION_FWD_ALGO_COUNT
)

// type ConvolutionBwdFilterPreference int

// const (
// 	NoWorkspace           ConvolutionBwdFilterPreference = C.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE
// 	PreferFastest         ConvolutionBwdFilterPreference = C.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST
// 	SpecifyWorkspaceLimit ConvolutionBwdFilterPreference = C.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
// )

type ConvolutionBwdFilterAlgo int

const (
	ConvolutionBwdFilterAlgo0                ConvolutionBwdFilterAlgo = C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0
	ConvolutionBwdFilterAlgo1                ConvolutionBwdFilterAlgo = C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1
	ConvolutionBwdFilterAlgoFFT              ConvolutionBwdFilterAlgo = C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT
	ConvolutionBwdFilterAlgo3                ConvolutionBwdFilterAlgo = C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3
	ConvolutionBwdFilterAlgoWinograd         ConvolutionBwdFilterAlgo = C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD
	ConvolutionBwdFilterAlgoWinogradNonfused ConvolutionBwdFilterAlgo = C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED
	ConvolutionBwdFilterAlgoFFTTiling        ConvolutionBwdFilterAlgo = C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING
	ConvolutionBwdFilterAlgoCount            ConvolutionBwdFilterAlgo = C.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT
)

// type ConvolutionBwdDataPreference int
// const (
// NoWorkspace ConvolutionBwdDataPreference = C.CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE
// PreferFastest ConvolutionBwdDataPreference = C.CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST
// SpecifyWorkspaceLimit ConvolutionBwdDataPreference = C.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT
// )

type ConvolutionBwdDataAlgo int

const (
	ConvolutionBwdDataAlgo0                ConvolutionBwdDataAlgo = C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
	ConvolutionBwdDataAlgo1                ConvolutionBwdDataAlgo = C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
	ConvolutionBwdDataAlgoFFT              ConvolutionBwdDataAlgo = C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT
	ConvolutionBwdDataAlgoFFTTiling        ConvolutionBwdDataAlgo = C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING
	ConvolutionBwdDataAlgoWinograd         ConvolutionBwdDataAlgo = C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD
	ConvolutionBwdDataAlgoWinogradNonfused ConvolutionBwdDataAlgo = C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
	ConvolutionBwdDataAlgoCount            ConvolutionBwdDataAlgo = C.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT
)
