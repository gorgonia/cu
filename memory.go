package cu

//#include <cuda.h>
import "C"
import (
	"fmt"

	"github.com/pkg/errors"
)

// DevicePtr is a pointer to the device memory. It is equivalent to CUDA's CUdeviceptr
type DevicePtr uintptr

func (d DevicePtr) String() string { return fmt.Sprintf("0x%x", uintptr(d)) }

func (d DevicePtr) AddressRange() (size int64, base DevicePtr, err error) {
	var s C.size_t
	var b C.CUdeviceptr
	if err = result(C.cuMemGetAddressRange(&b, &s, C.CUdeviceptr(d))); err != nil {
		err = errors.Wrapf(err, "MemGetAddressRange")
		return
	}
	return int64(s), DevicePtr(b), nil
}

// Uintptr returns the pointer in form of a uintptr
func (d DevicePtr) Uintptr() uintptr { return uintptr(d) }

// IsCUDAMemory returns true.
func (d DevicePtr) IsCUDAMemory() bool { return true }
