package cu

//#include <cuda.h>
import "C"
import (
	"fmt"
	"unsafe"
)

// Device is the representation of a CUDA device
type Device int

const (
	CPU       Device = -1
	BadDevice Device = -2
)

// Name returns the name of the device.
//
// Wrapper over cuDeviceGetName: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gef75aa30df95446a845f2a7b9fffbb7f
func (d Device) Name() (string, error) {
	size := 256
	buf := make([]byte, 256)
	cstr := C.CString(string(buf))
	defer C.free(unsafe.Pointer(cstr))
	if err := result(C.cuDeviceGetName(cstr, C.int(size), C.CUdevice(d))); err != nil {
		return "", err
	}
	return C.GoString(cstr), nil
}

// String implementes fmt.Stringer (and runtime.stringer)
func (d Device) String() string {
	if d == CPU {
		return "CPU"
	}
	if d < 0 {
		return "Invalid Device"
	}
	return fmt.Sprintf("GPU(%d)", int(d))
}

// IsGPU returns true if the device is a GPU.
func (d Device) IsGPU() bool {
	if d < 0 {
		return false
	}
	return true
}
