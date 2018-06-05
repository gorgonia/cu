package cu

// #include <cuda.h>
import "C"

type SurfRef struct {
	ref C.CUsurfref
}

func (s SurfRef) c() C.CUsurfref {
	return s.ref
}
