package nvrtc

//#include <nvrtc.h>
import "C"

type nvrtcResult int

func (err nvrtcResult) Error() string  { return err.String() }
func (err nvrtcResult) String() string { return resString[err] }

func result(x C.nvrtcResult) error {
	err := nvrtcResult(x)
	if err == Success {
		return nil
	}
	if err > InternalError {
		return InternalError
	}

	return err
}

const (
	Success                           nvrtcResult = C.NVRTC_SUCCESS
	OutOfMemory                       nvrtcResult = C.NVRTC_ERROR_OUT_OF_MEMORY
	ProgramCreationFailure            nvrtcResult = C.NVRTC_ERROR_PROGRAM_CREATION_FAILURE
	InvalidInput                      nvrtcResult = C.NVRTC_ERROR_INVALID_INPUT
	InvalidProgram                    nvrtcResult = C.NVRTC_ERROR_INVALID_PROGRAM
	InvalidOption                     nvrtcResult = C.NVRTC_ERROR_INVALID_OPTION
	Compilation                       nvrtcResult = C.NVRTC_ERROR_COMPILATION
	BuiltinOperationFailure           nvrtcResult = C.NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
	NoNameExpressionsAfterCompilation nvrtcResult = C.NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
	NoLoweredNamesBeforeCompilation   nvrtcResult = C.NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
	NameExpressionNotValid            nvrtcResult = C.NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID
	InternalError                     nvrtcResult = C.NVRTC_ERROR_INTERNAL_ERROR
)

var resString = map[nvrtcResult]string{
	Success:                           "Success",
	OutOfMemory:                       "OutOfMemory",
	ProgramCreationFailure:            "ProgramCreationFailure",
	InvalidInput:                      "InvalidInput",
	InvalidProgram:                    "InvalidProgram",
	InvalidOption:                     "InvalidOption",
	Compilation:                       "Compilation",
	BuiltinOperationFailure:           "BuiltinOperationFailure",
	NoNameExpressionsAfterCompilation: "NoNameExpressionsAfterCompilation",
	NoLoweredNamesBeforeCompilation:   "NoLoweredNamesBeforeCompilation",
	NameExpressionNotValid:            "NameExpressionNotValid",
	InternalError:                     "InternalError",
}
