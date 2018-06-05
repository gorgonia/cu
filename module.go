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
	var mod Module
	cstr := C.CString(name)
	defer C.free(unsafe.Pointer(cstr))
	err := result(C.cuModuleLoad(&mod.mod, cstr))
	return mod, err
}

// LoadData loads a module from a input string.
func LoadData(image string) (Module, error) {
	var mod Module
	cstr := C.CString(image)
	defer C.free(unsafe.Pointer(cstr))
	err := result(C.cuModuleLoadData(&mod.mod, unsafe.Pointer(cstr)))
	return mod, err
}

// LoadDataEx loads a module from a input string.
func LoadDataEx(image string, options ...JITOption) (Module, error) {
	var mod Module
	cstr := C.CString(image)
	defer C.free(unsafe.Pointer(cstr))

	argcount, args, argvals := encodeArguments(options)
	err := result(C.cuModuleLoadDataEx(&mod.mod, unsafe.Pointer(cstr), argcount, args, argvals))
	return mod, err
}

// LoadFatBinary loads a module from a input string.
func LoadFatBinary(image string) (Module, error) {
	var mod Module
	cstr := C.CString(image)
	defer C.free(unsafe.Pointer(cstr))
	err := result(C.cuModuleLoadFatBinary(&mod.mod, unsafe.Pointer(cstr)))
	return mod, err
}

// Function returns a pointer to the function in the module by the name. If it's not found, the error NotFound is returned
func (m Module) Function(name string) (Function, error) {
	var fn Function
	cstr := C.CString(name)
	defer C.free(unsafe.Pointer(cstr))
	err := result(C.cuModuleGetFunction(&fn.fn, m.mod, cstr))
	return fn, err
}

// Global returns a global pointer as defined in a module. It returns a pointer to the memory in the device.
func (m Module) Global(name string) (DevicePtr, int64, error) {
	var d C.CUdeviceptr
	var size C.size_t
	cstr := C.CString(name)
	defer C.free(unsafe.Pointer(cstr))
	if err := result(C.cuModuleGetGlobal(&d, &size, m.mod, cstr)); err != nil {
		return 0, 0, err
	}
	return DevicePtr(d), int64(size), nil
}

func (ctx *Ctx) Load(name string) (m Module, err error) {
	var mod C.CUmodule
	cstr := C.CString(name)
	defer C.free(unsafe.Pointer(cstr))
	f := func() error { return result(C.cuModuleLoad(&mod, cstr)) }
	if err = ctx.Do(f); err != nil {
		err = errors.Wrap(err, "LoadModule")
		return
	}
	m = Module{mod}
	return
}

func (ctx *Ctx) ModuleFunction(m Module, name string) (function Function, err error) {
	var fn C.CUfunction
	cstr := C.CString(name)
	defer C.free(unsafe.Pointer(cstr))
	f := func() error { return result(C.cuModuleGetFunction(&fn, m.mod, cstr)) }
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

	cstr := C.CString(name)
	defer C.free(unsafe.Pointer(cstr))
	f := func() error { return result(C.cuModuleGetGlobal(&d, &s, m.mod, cstr)) }
	if err = ctx.Do(f); err != nil {
		err = errors.Wrap(err, "ModuleGlobal")
		return
	}
	size = int64(s)
	return
}
