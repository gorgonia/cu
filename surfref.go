package cu

// #include <cuda.h>
import "C"
import "unsafe"

type SurfRef struct {
	ref C.CUsurfref
}

func (s SurfRef) c() C.CUsurfref {
	return s.ref
}

func (mod Module) SurfRef(name string) (SurfRef, error) {
	var ref SurfRef
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	err := result(C.cuModuleGetSurfRef(&ref.ref, mod.mod, cname))
	return ref, err
}

func (ctx *Ctx) ModuleSurfRef(mod Module, name string) (SurfRef, error) {
	var ref SurfRef
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	err := result(C.cuModuleGetSurfRef(&ref.ref, mod.mod, cname))
	return ref, err
}
