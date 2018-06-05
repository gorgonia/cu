package cu

// #include <cuda.h>
import "C"

type TexRef struct {
	ref C.CUtexref
}

func (t TexRef) c() C.CUtexref {
	return t.ref
}
