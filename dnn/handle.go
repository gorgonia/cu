package cudnn

// #include <cudnn.h>
import "C"
import (
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
//
// Once the context has been finished, do remember to call `Close` on the context.
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
	return retVal
}

//  Close destroys the underlying context.
func (ctx *Context) Close() error {
	var empty C.cudnnHandle_t
	if ctx.internal == empty {
		return nil
	}

	if err := result(C.cudnnDestroy(ctx.internal)); err != nil {
		return err
	}
	ctx.internal = empty
	return nil
}
