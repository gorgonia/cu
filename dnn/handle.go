package cudnn

// #include <cudnn_v7.h>
import "C"

type Context struct {
	internal C.cudnnHandle_t
}

func NewContext() *Context {
	var h C.cudnnHandle_t
	if err := result(C.cudnnCreate(&h)); err != nil {
		panic(err)
	}
	return &Context{
		internal: h,
	}
}

func (ctx *Context) Destroy() error { return result(C.cudnnDestroy(ctx.internal)) }
