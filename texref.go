package cu

// #include <cuda.h>
import "C"
import "unsafe"

type TexRef uintptr

func (t TexRef) c() C.CUtexref {
	return C.CUtexref(unsafe.Pointer(uintptr(t)))
}
