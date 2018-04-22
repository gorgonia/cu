package cudnn

// #include <cudnn_v7.h>
import "C"
import (
	"runtime"
	"unsafe"
)

// Memory represents an instance of CUDA memory
type Memory interface {
	Uintptr() uintptr
	Pointer() unsafe.Pointer

	IsNativelyAccessible() bool
}

// Context represents the context in which cuDNN operations are performed in.
//
// Internally, the Context holds a cudnnHandle_t
type Context struct {
	internal C.cudnnHandle_t
}

// NewContext creates a new Context. This is the only function that will panic if it is unable to create the context.
func NewContext() (retVal *Context) {
	var internal C.cudnnHandle_t
	if err := result(C.cudnnCreate(&internal)); err != nil {
		panic(err)
	}
	retVal = &Context{internal}
	runtime.SetFinalizer(retVal, destroyContext)
	return retVal
}

func destroyContext(obj *Context) { C.cudnnDestroy(obj.internal) }
