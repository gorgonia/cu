package cu

// #include <cuda.h>
import "C"
import "unsafe"

type SurfRef uintptr

func (s SurfRef) c() C.CUsurfref {
	return C.CUsurfref(unsafe.Pointer(uintptr(s)))
}
