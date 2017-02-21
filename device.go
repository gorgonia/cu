package cu

//#include <cuda.h>
import "C"
import "fmt"

// Device is the representation of a CUDA device
type Device int

const (
	CPU       Device = -1
	BadDevice Device = -2
)

// GetDevice returns a handle to the compute device with the provided ID
//
// Wrapper over cuDeviceGet: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb
func GetDevice(id int) (d Device, err error) {
	var dev C.CUdevice
	if err = result(C.cuDeviceGet(&dev, C.int(id))); err != nil {
		return
	}
	return Device(dev), nil
}

// NumDevices returns the number of compute capable devices
//
// Wrapper over cuDeviceGetCount: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74
func NumDevices() (int, error) {
	var c C.int
	if err := result(C.cuDeviceGetCount(&c)); err != nil {
		return 0, err
	}
	return int(c), nil
}

// Attribute returns information about the device.
//
// Wrapper over cuDeviceGetAttribute: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266
func (d Device) Attribute(attr DeviceAttribute) (int, error) {
	var a C.int
	if err := result(C.cuDeviceGetAttribute(&a, C.CUdevice_attribute(attr), C.CUdevice(d))); err != nil {
		return 0, err
	}
	return int(a), nil
}

// Name returns the name of the device.
//
// Wrapper over cuDeviceGetName: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gef75aa30df95446a845f2a7b9fffbb7f
func (d Device) Name() (string, error) {
	size := 256
	buf := make([]byte, 256)
	cstr := C.CString(string(buf))
	if err := result(C.cuDeviceGetName(cstr, C.int(size), C.CUdevice(d))); err != nil {
		return "", err
	}
	return C.GoString(cstr), nil
}

// TotalMem returns the total amount of memories on the device (in bytes)
//
// Wrapper over cuDeviceTotalMem http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc6a0d6551335a3780f9f3c967a0fde5d
func (d Device) TotalMem() (int64, error) {
	var size C.size_t
	if err := result(C.cuDeviceTotalMem(&size, C.CUdevice(d))); err != nil {
		return 0, err
	}
	return int64(size), nil
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
