package cu

// #include <cuda.h>
import "C"
import (
	"unsafe"

	"github.com/pkg/errors"
)

// Module represents a CUDA Module
type Module struct {
	mod C.CUmodule
}

func (m Module) c() C.CUmodule { return m.mod }

// Load loads a module into the current context.
// The CUDA driver API does not attempt to lazily allocate the resources needed by a module;
// if the memory for functions and data (constant and global) needed by the module cannot be allocated, `Load()` fails.
//
// The file should be a cubin file as output by nvcc, or a PTX file either as output by nvcc or handwritten, or a fatbin file as output by nvcc from toolchain 4.0 or late
func Load(name string) (Module, error) {
	var mod C.CUmodule
	if err := result(C.cuModuleLoad(&mod, C.CString(name))); err != nil {
		return Module{}, err
	}
	return Module{mod}, nil
}

// LoadData loads a module from a input string.
func LoadData(image string) (Module, error) {
	var mod C.CUmodule
	s := C.CString(image) // void* == unsafe.Pointer
	if err := result(C.cuModuleLoadData(&mod, unsafe.Pointer(s))); err != nil {
		return Module{}, err
	}
	return Module{mod}, nil
}

// Function returns a pointer to the function in the module by the name. If it's not found, the error NotFound is returned
func (m Module) Function(name string) (Function, error) {
	var fn C.CUfunction
	str := C.CString(name)
	if err := result(C.cuModuleGetFunction(&fn, m.mod, str)); err != nil {
		return Function{}, err
	}
	return Function{fn}, nil
}

// Global returns a global pointer as defined in a module. It returns a pointer to the memory in the device.
func (m Module) Global(name string) (DevicePtr, int64, error) {
	var d C.CUdeviceptr
	var size C.size_t
	str := C.CString(name)
	if err := result(C.cuModuleGetGlobal(&d, &size, m.mod, str)); err != nil {
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
	m = Module{mod}
	return
}

func (ctx *Ctx) ModuleFunction(m Module, name string) (function Function, err error) {
	var fn C.CUfunction
	str := C.CString(name)
	f := func() error { return result(C.cuModuleGetFunction(&fn, m.mod, str)) }
	if err = ctx.Do(f); err != nil {
		err = errors.Wrap(err, "ModuleFunction")
		return
	}
	function = Function{fn}
	return
}

func (ctx *Ctx) ModuleGlobal(m Module, name string) (dptr DevicePtr, size int64, err error) {
	var d C.CUdeviceptr
	var s C.size_t

	str := C.CString(name)
	f := func() error { return result(C.cuModuleGetGlobal(&d, &s, m.mod, str)) }
	if err = ctx.Do(f); err != nil {
		err = errors.Wrap(err, "ModuleGlobal")
		return
	}
	size = int64(s)
	return
}
