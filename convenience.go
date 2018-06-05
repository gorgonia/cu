package cu

// #include <cuda.h>
import "C"
import (
	"log"
	"unsafe"

	"github.com/pkg/errors"
)

// This file lists all the convenience functions and methods, not necessarily stuff that is covered in the API

// MemoryType returns the MemoryType of the memory
func (mem DevicePtr) MemoryType() (typ MemoryType, err error) {
	var p unsafe.Pointer
	if p, err = mem.PtrAttribute(MemoryTypeAttr); err != nil {
		return
	}
	t := *(*uint64)(p)
	typ = MemoryType(byte(t))
	return
}

// MemSize returns the size of the memory slab in bytes. Returns 0 if errors occured
func (mem DevicePtr) MemSize() uintptr {
	size, _, err := mem.AddressRange()
	if err != nil {
		log.Printf("MEMSIZE ERR %v", err)
	}
	return uintptr(size)
}

// Pointer returns the pointer in form of unsafe.pointer. You shouldn't use it though, as the pointer is typically on the device
func (mem DevicePtr) Pointer() unsafe.Pointer {
	return unsafe.Pointer(uintptr(mem))
}

// ComputeCapability returns the compute capability of the device.
// This method is a convenience method for the deprecated API call cuDeviceComputeCapability.
func (d Device) ComputeCapability() (major, minor int, err error) {
	var attrs []int
	if attrs, err = d.Attributes(ComputeCapabilityMajor, ComputeCapabilityMinor); err != nil {
		err = errors.Wrapf(err, "Failed to get ComputeCapability")
		return
	}
	major = attrs[0]
	minor = attrs[1]
	return
}
