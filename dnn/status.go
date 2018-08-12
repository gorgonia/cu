package cudnn

// #include <cudnn.h>
import "C"

type cudnnStatus int

func (err cudnnStatus) Error() string  { return err.String() }
func (err cudnnStatus) String() string { return resString[err] }

func result(x C.cudnnStatus_t) error {
	err := cudnnStatus(x)
	if err == Success {
		return nil
	}
	if err > RuntimeFpOverflow {
		return NotSupported
	}
	return err
}

const (
	Success                    cudnnStatus = C.CUDNN_STATUS_SUCCESS
	NotInitialized             cudnnStatus = C.CUDNN_STATUS_NOT_INITIALIZED
	AllocFailed                cudnnStatus = C.CUDNN_STATUS_ALLOC_FAILED
	BadParam                   cudnnStatus = C.CUDNN_STATUS_BAD_PARAM
	InternalError              cudnnStatus = C.CUDNN_STATUS_INTERNAL_ERROR
	InvalidValue               cudnnStatus = C.CUDNN_STATUS_INVALID_VALUE
	ArchMismatch               cudnnStatus = C.CUDNN_STATUS_ARCH_MISMATCH
	MappingError               cudnnStatus = C.CUDNN_STATUS_MAPPING_ERROR
	ExecutionFailed            cudnnStatus = C.CUDNN_STATUS_EXECUTION_FAILED
	NotSupported               cudnnStatus = C.CUDNN_STATUS_NOT_SUPPORTED
	LicenseError               cudnnStatus = C.CUDNN_STATUS_LICENSE_ERROR
	RuntimePrerequisiteMissing cudnnStatus = C.CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING
	RuntimeInProgress          cudnnStatus = C.CUDNN_STATUS_RUNTIME_IN_PROGRESS
	RuntimeFpOverflow          cudnnStatus = C.CUDNN_STATUS_RUNTIME_FP_OVERFLOW
)

var resString = map[cudnnStatus]string{
	Success:                    "Success",
	NotInitialized:             "NotInitialized",
	AllocFailed:                "AllocFailed",
	BadParam:                   "BadParam",
	InternalError:              "cuDNN InternalError",
	InvalidValue:               "cuDNN InvalidValue",
	ArchMismatch:               "ArchMismatch",
	MappingError:               "MappingError",
	ExecutionFailed:            "ExecutionFailed",
	NotSupported:               "NotSupported",
	LicenseError:               "LicenseError",
	RuntimePrerequisiteMissing: "RuntimePrerequisiteMissing",
	RuntimeInProgress:          "RuntimeInProgress",
	RuntimeFpOverflow:          "RuntimeFpOverflow",
}

const (
	dtypeMismatch3 = "Dtype Mismatch. These three should match %v, %v and %v."
	dtypeMismatch2 = "Dtype Mismatch. Expected %v or %v. Got %v instead."
	shapeMismatch3 = "Shape Mismatch. These three should match %v, %v and %v"
	memoryError3   = "Memory Error. A: %p B: %p C: %p"
	nyi            = "Not yet implemented. %v %v"
)
