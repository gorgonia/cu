package cu

// #include <cuda.h>
import "C"
import "unsafe"

type TexRef struct {
	ref C.CUtexref
}

func (t TexRef) c() C.CUtexref {
	return t.ref
}

func (mod Module) TexRef(name string) (TexRef, error) {
	var ref TexRef
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	err := result(C.cuModuleGetTexRef(&ref.ref, mod.mod, cname))
	return ref, err
}

func (ctx *Ctx) ModuleTexRef(mod Module, name string) (TexRef, error) {
	var ref TexRef
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	err := result(C.cuModuleGetTexRef(&ref.ref, mod.mod, cname))
	return ref, err
}
