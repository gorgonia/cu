package cu

// #include <cuda.h>
import "C"
import (
	"unsafe"

	"github.com/pkg/errors"
)

// Module represents a CUDA Module, which is a pointer to a CUmod_st struct
type Module uintptr

func makeModule(mod C.CUmodule) Module {
	return Module(uintptr(unsafe.Pointer(mod)))
}

func (m Module) c() C.CUmodule {
	return C.CUmodule(unsafe.Pointer(uintptr(m)))
}

// Load loaads a module into the current context.
// The CUDA driver API does not attempt to lazily allocate the resources needed by a module;
// if the memory for functions and data (constant and global) needed by the module cannot be allocated, `Load()` fails.
//
// The file should be a cubin file as output by nvcc, or a PTX file either as output by nvcc or handwritten, or a fatbin file as output by nvcc from toolchain 4.0 or late
func Load(name string) (Module, error) {
	var mod C.CUmodule
	if err := result(C.cuModuleLoad(&mod, C.CString(name))); err != nil {
		return 0, err
	}
	return makeModule(mod), nil
}

// LoadData loads a module from a input string.
func LoadData(image string) (Module, error) {
	var mod C.CUmodule
	s := C.CString(image) // void* == unsafe.Pointer
	if err := result(C.cuModuleLoadData(&mod, unsafe.Pointer(s))); err != nil {
		return 0, err
	}
	return makeModule(mod), nil
}

// Function returns a pointer to the function in the module by the name. If it's not found, the error NotFound is returned
func (m Module) Function(name string) (Function, error) {
	var fn C.CUfunction
	mod := C.CUmodule(unsafe.Pointer(uintptr(m)))
	str := C.CString(name)
	if err := result(C.cuModuleGetFunction(&fn, mod, str)); err != nil {
		return 0, err
	}
	return makeFunction(fn), nil
}

// Global returns a global pointer as defined in a module. It returns a pointer to the memory in the device.
func (m Module) Global(name string) (DevicePtr, int64, error) {
	var d C.CUdeviceptr
	var size C.size_t
	mod := m.c()
	str := C.CString(name)
	if err := result(C.cuModuleGetGlobal(&d, &size, mod, str)); err != nil {
		return 0, 0, err
	}
	return DevicePtr(d), int64(size), nil
}

func (ctx *Ctx) Load(name string) (m Module, err error) {
	var mod C.CUmodule
	f := func() error { return result(C.cuModuleLoad(&mod, C.CString(name))) }
	if err = ctx.Do(f); err != nil {
		err = errors.Wrap(err, "LoadModule")
		return
	}
	m = makeModule(mod)
	return
}

func (ctx *Ctx) ModuleFunction(m Module, name string) (function Function, err error) {
	var fn C.CUfunction
	mod := m.c()
	str := C.CString(name)
	f := func() error { return result(C.cuModuleGetFunction(&fn, mod, str)) }
	if err = ctx.Do(f); err != nil {
		err = errors.Wrap(err, "ModuleFunction")
		return
	}
	function = makeFunction(fn)
	return
}

func (ctx *Ctx) ModuleGlobal(m Module, name string) (dptr DevicePtr, size int64, err error) {
	var d C.CUdeviceptr
	var s C.size_t

	mod := m.c()
	str := C.CString(name)
	f := func() error { return result(C.cuModuleGetGlobal(&d, &s, mod, str)) }
	if err = ctx.Do(f); err != nil {
		err = errors.Wrap(err, "ModuleGlobal")
		return
	}
	size = int64(s)
	return
}
