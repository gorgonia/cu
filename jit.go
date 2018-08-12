package cu

// #include <cuda.h>
import "C"

import (
	"unsafe"
)

type LinkState struct {
	state     C.CUlinkState
	keepalive [][]JITOption
}

type JITOption interface {
	arguments() []jitoption
}

type jitoption struct {
	option C.CUjit_option
	value  uintptr
}

type (
	// Max number of registers that a thread may use.
	JITMaxRegisters struct{ Value uint }
	// Specifies minimum number of threads per block to target compilation
	JITThreadsPerBlock struct{ Value uint }
	// Overwrites the option value with the total wall clock time, in
	// milliseconds, spent in the compiler and linker.
	JITWallTime struct{ Result float32 }

	// Buffer in which to print any log messages that are informational in nature.
	JITInfoLogBuffer struct{ Buffer []byte }
	// Buffer in which to print any log messages that reflect errors
	JITErrorLogBuffer struct{ Buffer []byte }
	// Level of optimizations to apply to generated code (0 - 4)
	JITOptimizationLevel struct{ Value uint }
	// Determines the target based on the current attached context (default)
	JITTargetFromContext struct{}
	// Target is chosen based on supplied Value
	JITTarget struct{ Value JITTargetOption }
	// Specifies choice of fallback strategy if matching cubin is not found.
	JITFallbackStrategy struct{ Value JITFallbackOption }
	// Specifies whether to create debug information in output (-g)
	JITGenerateDebugInfo struct{ Enabled bool }
	// Generate verbose log messages (-v)
	JITLogVerbose struct{ Enabled bool }
	// Generate line number information (-lineinfo)
	JITGenerateLineInfo struct{ Enabled bool }
	// Specifies whether to enable caching explicitly (-dlcm)
	JITCacheMode struct{ Value JITCacheModeOption }
)

func (opt *JITMaxRegisters) arguments() []jitoption {
	return []jitoption{
		{C.CU_JIT_MAX_REGISTERS, uintptr(opt.Value)},
	}
}
func (opt *JITThreadsPerBlock) arguments() []jitoption {
	return []jitoption{
		{C.CU_JIT_THREADS_PER_BLOCK, uintptr(opt.Value)},
	}
}

func (opt *JITWallTime) arguments() []jitoption {
	return []jitoption{
		{C.CU_JIT_WALL_TIME, uintptr(unsafe.Pointer(&opt.Result))},
	}
}
func (opt *JITInfoLogBuffer) arguments() []jitoption {
	return []jitoption{
		{C.CU_JIT_INFO_LOG_BUFFER, uintptr(unsafe.Pointer(&opt.Buffer[0]))},
		{C.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, uintptr(len(opt.Buffer))},
	}
}
func (opt *JITErrorLogBuffer) arguments() []jitoption {
	return []jitoption{
		{C.CU_JIT_ERROR_LOG_BUFFER, uintptr(unsafe.Pointer(&opt.Buffer[0]))},
		{C.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, uintptr(len(opt.Buffer))},
	}
}
func (opt *JITOptimizationLevel) arguments() []jitoption {
	return []jitoption{
		{C.CU_JIT_OPTIMIZATION_LEVEL, uintptr(opt.Value)},
	}
}
func (opt *JITTargetFromContext) arguments() []jitoption {
	return []jitoption{
		{C.CU_JIT_TARGET_FROM_CUCONTEXT, 0},
	}
}
func (opt *JITTarget) arguments() []jitoption {
	return []jitoption{
		{C.CU_JIT_TARGET, uintptr(opt.Value)},
	}
}
func (opt *JITFallbackStrategy) arguments() []jitoption {
	return []jitoption{
		{C.CU_JIT_FALLBACK_STRATEGY, uintptr(opt.Value)},
	}
}
func (opt *JITGenerateDebugInfo) arguments() []jitoption {
	return []jitoption{
		{C.CU_JIT_GENERATE_DEBUG_INFO, jitBooleanOption(opt.Enabled)},
	}
}
func (opt *JITLogVerbose) arguments() []jitoption {
	return []jitoption{
		{C.CU_JIT_LOG_VERBOSE, jitBooleanOption(opt.Enabled)},
	}
}
func (opt *JITGenerateLineInfo) arguments() []jitoption {
	return []jitoption{
		{C.CU_JIT_GENERATE_LINE_INFO, jitBooleanOption(opt.Enabled)},
	}
}
func (opt *JITCacheMode) arguments() []jitoption {
	return []jitoption{
		{C.CU_JIT_CACHE_MODE, uintptr(opt.Value)},
	}
}

func jitBooleanOption(b bool) uintptr {
	if b {
		return 1
	}
	return 0
}

type JITTargetOption uint64

const (
	// JITTarget10 JITTargetOption = C.CU_TARGET_COMPUTE_10
	// JITTarget11 JITTargetOption = C.CU_TARGET_COMPUTE_11
	// JITTarget12 JITTargetOption = C.CU_TARGET_COMPUTE_12
	// JITTarget13 JITTargetOption = C.CU_TARGET_COMPUTE_13
	JITTarget20 JITTargetOption = C.CU_TARGET_COMPUTE_20
	JITTarget21 JITTargetOption = C.CU_TARGET_COMPUTE_21
	JITTarget30 JITTargetOption = C.CU_TARGET_COMPUTE_30
	JITTarget32 JITTargetOption = C.CU_TARGET_COMPUTE_32
	JITTarget35 JITTargetOption = C.CU_TARGET_COMPUTE_35
	JITTarget37 JITTargetOption = C.CU_TARGET_COMPUTE_37
	JITTarget50 JITTargetOption = C.CU_TARGET_COMPUTE_50
	JITTarget52 JITTargetOption = C.CU_TARGET_COMPUTE_52
	JITTarget53 JITTargetOption = C.CU_TARGET_COMPUTE_53
	JITTarget60 JITTargetOption = C.CU_TARGET_COMPUTE_60
	JITTarget61 JITTargetOption = C.CU_TARGET_COMPUTE_61
	JITTarget62 JITTargetOption = C.CU_TARGET_COMPUTE_62
)

// Cubin matching fallback strategies
type JITFallbackOption uint64

const (
	// Prefer to compile ptx if exact binary match not found
	JITPreferPTX JITFallbackOption = C.CU_PREFER_PTX
	// Prefer to fall back to compatible binary code if exact match not found
	JITPreferBinary JITFallbackOption = C.CU_PREFER_BINARY
)

// Caching modes for dlcm
type JITCacheModeOption uint64

const (
	// Compile with no -dlcm flag specified
	JITCacheNone JITCacheModeOption = C.CU_JIT_CACHE_OPTION_NONE
	// Compile with L1 cache disabled
	JITCacheCG JITCacheModeOption = C.CU_JIT_CACHE_OPTION_CG
	// Compile with L1 cache enabled
	JITCacheCA JITCacheModeOption = C.CU_JIT_CACHE_OPTION_CA
)

type JITInputType uint64

const (
	// Compiled device-class-specific device code
	JITInputCUBIN JITInputType = C.CU_JIT_INPUT_CUBIN
	// PTX source code
	JITInputPTX JITInputType = C.CU_JIT_INPUT_PTX
	// Bundle of multiple cubins and/or PTX of some device code
	JITInputFatBinary JITInputType = C.CU_JIT_INPUT_FATBINARY
	// Host object with embedded device code
	JITInputObject JITInputType = C.CU_JIT_INPUT_OBJECT
	// Archive of host objects with embedded device code
	JITInputLibrary JITInputType = C.CU_JIT_INPUT_LIBRARY
)

// Creates a pending JIT linker invocation.
func NewLink(options ...JITOption) (*LinkState, error) {
	link := &LinkState{}

	argcount, args, argvals := link.encodeArguments(options)
	err := result(C.cuLinkCreate(argcount, args, argvals, &link.state))
	return link, err
}

func (link *LinkState) encodeArguments(options []JITOption) (C.uint, *C.CUjit_option, *unsafe.Pointer) {
	if len(options) > 0 {
		link.keepalive = append(link.keepalive, options)
	}
	return encodeArguments(options)
}

func encodeArguments(options []JITOption) (C.uint, *C.CUjit_option, *unsafe.Pointer) {
	if len(options) == 0 {
		return 0, nil, nil
	}

	result := []C.CUjit_option{}
	values := []uintptr{}
	for _, option := range options {
		for _, arg := range option.arguments() {
			result = append(result, arg.option)
			values = append(values, arg.value)
		}
	}

	return C.uint(len(result)), &result[0], (*unsafe.Pointer)((unsafe.Pointer)(&values[0]))
}

// Add an input to a pending linker invocation
func (link *LinkState) AddData(input JITInputType, data string, name string, options ...JITOption) error {
	argcount, args, argvals := link.encodeArguments(options)

	cname := C.CString(name)
	bytes := []byte(data)
	err := result(C.cuLinkAddData(
		link.state, C.CUjitInputType(input),
		unsafe.Pointer(&bytes[0]), C.size_t(len(bytes)),
		cname,
		argcount, args, argvals,
	))
	C.free(unsafe.Pointer(cname))
	return err
}

// Add a file input to a pending linker invocation
func (link *LinkState) AddFile(input JITInputType, path string, options ...JITOption) error {
	argcount, args, argvals := link.encodeArguments(options)

	cpath := C.CString(path)
	err := result(C.cuLinkAddFile(
		link.state, C.CUjitInputType(input),
		cpath,
		argcount, args, argvals,
	))
	C.free(unsafe.Pointer(cpath))
	return err
}

// Complete a pending linker invocation
func (link *LinkState) Complete() (string, error) {
	var data unsafe.Pointer
	var datasize C.size_t

	err := result(C.cuLinkComplete(link.state, &data, &datasize))
	if err != nil {
		return "", err
	}

	size := int(datasize)
	buffer := make([]byte, size)
	copy(buffer, (*[20 << 30]byte)(data)[:size])

	return string(buffer), nil
}

// Destroys state for a JIT linker invocation.
func (link *LinkState) Destroy() error {
	return result(C.cuLinkDestroy(link.state))
}
