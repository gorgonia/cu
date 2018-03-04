package cudnn

import "unsafe"

// Memory represents an instance of CUDA memory
type Memory interface {
	Uintptr() uintptr
	Pointer() unsafe.Pointer

	IsNativelyAccessible() bool
}
