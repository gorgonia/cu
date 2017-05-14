package cu

// include <cuda.h>
import "C"
import "unsafe"

// Event represents a CUDA event
type Event uintptr

func (e Event) c() C.CUevent {
	return C.CUevent(unsafe.Pointer(uintptr(e)))
}
