package cudnn

// #include <cudnn.h>
// #include "convolution.h"
import "C"
import (
	"runtime"
	"unsafe"

	"github.com/pkg/errors"
)

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

func (c ConvolutionFwdAlgo) C() C.cudnnConvolutionFwdAlgo_t { return C.cudnnConvolutionFwdAlgo_t(c) }

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

func (c ConvolutionBwdFilterAlgo) C() C.cudnnConvolutionBwdFilterAlgo_t {
	return C.cudnnConvolutionBwdFilterAlgo_t(c)
}

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

func (c ConvolutionBwdDataAlgo) C() C.cudnnConvolutionBwdDataAlgo_t {
	return C.cudnnConvolutionBwdDataAlgo_t(c)
}

type ConvolutionMode int

const (
	StandardConvolution ConvolutionMode = C.CUDNN_CONVOLUTION
	CrossCorrelation    ConvolutionMode = C.CUDNN_CROSS_CORRELATION
)

// C returns the C representation of ConvolutionMode
func (e ConvolutionMode) C() C.cudnnConvolutionMode_t { return C.cudnnConvolutionMode_t(e) }

// Convolution is a struct describing the convolution operations. Internally it holds a cudnnConvolutionDescriptor_t, which will be passed around when making cgo calls.
type Convolution struct {
	internal C.cudnnConvolutionDescriptor_t

	mathType     MathType
	groupCount   int
	padding      []int
	filterStride []int
	dilation     []int

	// cache of outputShape
	dims        int
	inputTensor []int
	inputFilter []int
	outputShape []int
}

func NewConvolution(mathType MathType, groupCount int, padding, filterStride, dilation []int, convolutionMode ConvolutionMode, datatype DataType) (retVal *Convolution, err error) {
	// checks
	if !(len(padding) == len(filterStride) && len(filterStride) == len(dilation)) {
		return nil, errors.Errorf("Unmatching inputs: padding %v, filterStride %v, dilation %v", padding, filterStride, dilation)
	}
	if len(padding) < 2 {
		return nil, errors.Errorf("Convolution expects 4 dimensional inputs")
	}

	var internal C.cudnnConvolutionDescriptor_t
	padA, padAManaged := ints2CIntPtr(padding)
	defer returnManaged(padAManaged)

	filterStrideA, filterStrideAManaged := ints2CIntPtr(filterStride)
	defer returnManaged(filterStrideAManaged)
	dilationA, dilationAManaged := ints2CIntPtr(dilation)
	defer returnManaged(dilationAManaged)
	if err = result(C.gocudnnNewConvolution(&internal, mathType.C(), C.int(groupCount), C.int(len(padding)), padA, filterStrideA, dilationA, convolutionMode.C(), datatype.C())); err != nil {
		return nil, err
	}

	retVal = &Convolution{
		internal: internal,

		mathType:     mathType,
		groupCount:   groupCount,
		padding:      padding,
		filterStride: filterStride,
		dilation:     dilation,
	}
	runtime.SetFinalizer(retVal, destroyConvolution)
	return retVal, nil
}

func (c *Convolution) MathType() MathType  { return c.mathType }
func (c *Convolution) GroupCount() int     { return c.groupCount }
func (c *Convolution) Padding() []int      { return cloneShape(c.padding) }
func (c *Convolution) FilterStride() []int { return cloneShape(c.filterStride) }
func (c *Convolution) Dilation() []int     { return cloneShape(c.dilation) }

func (c *Convolution) ForwardOutputShape(input *TensorDescriptor, filter *Filter, dims int) (retVal []int, err error) {
	if c.dims == dims && shapeEq(c.inputTensor, input.shape) && shapeEq(c.inputFilter, filter.shape) {
		return cloneShape(c.outputShape), nil
	}
	return c.CalcForwardOutputShape(input, filter, dims)
}

func (c *Convolution) CalcForwardOutputShape(input *TensorDescriptor, filter *Filter, dims int) (retVal []int, err error) {
	c.inputTensor = cloneShape(input.shape)
	c.inputFilter = cloneShape(filter.shape)
	c.dims = dims
	switch dims {
	case 0, 1:
		return nil, errors.Errorf("Only 2+ dims can be inferred")
	case 2:
		c.outputShape = make([]int, 4)
		n := (*C.int)(unsafe.Pointer(&c.outputShape[0]))
		c_ := (*C.int)(unsafe.Pointer(&c.outputShape[1]))
		h := (*C.int)(unsafe.Pointer(&c.outputShape[2]))
		w := (*C.int)(unsafe.Pointer(&c.outputShape[3]))
		if err = result(C.cudnnGetConvolution2dForwardOutputDim(c.internal, input.internal, filter.internal, n, c_, h, w)); err != nil {
			return nil, err
		}
	default:
		c.outputShape = make([]int, dims)
		ptr, ptrManaged := ints2CIntPtr(c.outputShape)
		defer returnManaged(ptrManaged)
		if err = result(C.cudnnGetConvolutionNdForwardOutputDim(c.internal, input.internal, filter.internal, C.int(dims), ptr)); err != nil {
			return nil, err
		}
	}
	return cloneShape(c.outputShape), nil
}

func destroyConvolution(obj *Convolution) { C.cudnnDestroyConvolutionDescriptor(obj.internal) }

// TODO
type ConvolutionFwdPerf struct {
	internal    C.cudnnConvolutionFwdAlgo_t
	Algo        ConvolutionFwdAlgo
	Time        float64
	Memory      uintptr // size
	Determinism Determinism
	MathType    MathType
	Err         error
}

func convolutionFwdPerfFromC(p C.cudnnConvolutionFwdAlgo_t) *ConvolutionFwdPerf {
	retVal := &ConvolutionFwdPerf{}
	return retVal
}

type ConvolutionBwdPerf struct {
	internal *C.cudnnConvolutionBwdFilterAlgoPerf_t
	Err      error

	Algo        ConvolutionBwdFilterAlgo
	Time        float64
	Memory      uintptr // size
	Determinism Determinism
	MathType    MathType
}

func convolutionBwdPerfFromC(p C.cudnnConvolutionBwdFilterAlgoPerf_t) *ConvolutionBwdPerf {
	retVal := &ConvolutionBwdPerf{}
	return retVal
}

type ConvolutionBwdDataPerf struct {
	internal *C.cudnnConvolutionBwdDataAlgoPerf_t
	Algo     ConvolutionBwdDataAlgo
	Err      error

	Time        float64
	Memory      uintptr // size
	Determinism Determinism
	MathType    MathType
}

func ConvolutionBwdDataPerfFromC(p C.cudnnConvolutionBwdDataAlgoPerf_t) *ConvolutionBwdDataPerf {
	retVal := &ConvolutionBwdDataPerf{}
	return retVal
}
