package cudnn

// #include <cudnn_v7.h>
import "C"

// Status is the cudnn status
type Status int

const (
	Success Status = iota
	NotInitialized
	AllocFailed
	BadParam
	InternalError
	InvalidValue
	ArchMismatch
	MappingError
	ExecutionFailed
	NotSupported
	LicenseError
	RuntimePrerequisiteMissing
	RuntimeInProgress
	RuntimeFpOverflow
)

func (err Status) Error() string  { return err.String() }
func (err Status) String() string { return resString[int(err)] }

func result(s C.cudnnStatus_t) error {
	err := Status(s)
	if err == Success {
		return nil
	}
	if err > RuntimeFpOverflow {
		return NotSupported
	}
	return err
}

var resString = [...]string{
	"Success",
	"NotInitialized",
	"AllocFailed",
	"BadParam",
	"InternalError",
	"InvalidValue",
	"ArchMismatch",
	"MappingError",
	"ExecutionFailed",
	"NotSupported",
	"LicenseError",
	"RuntimePrerequisiteMissing",
	"RuntimeInProgress",
	"RuntimeFpOverflow",
}

var (
	dtypeMismatch2 = "Dtype Mismatch. Expected %v or %v. Got %v"
	dtypeMismatch3 = "Dtype Mismatch. Expected %v. Got %v and %v"
	memoryError3   = "Memory error. Expected location to be %p. Got %p and %p instead"

	shapeMismatch3 = "Shape Mismatch. Expected %v. Got %v and %v"
	nyi            = "Not Yet Implemented"
)
