package cu

// #include <cuda.h>
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

// CUContext is a CUDA context
type CUContext struct{ ctx C.CUcontext }

func (ctx CUContext) String() string { return fmt.Sprintf("0x%x", uintptr(unsafe.Pointer(ctx.ctx))) }

func makeContext(ctx C.CUcontext) CUContext { return CUContext{ctx} }

// C returns the CUContext as its C version
func (ctx CUContext) c() C.CUcontext { return ctx.ctx }

func (d Device) MakeContext(flags ContextFlags) (CUContext, error) {
	var ctx CUContext
	err := result(C.cuCtxCreate(&ctx.ctx, C.uint(flags), C.CUdevice(d)))
	return ctx, err
}

// Lock ties the calling goroutine to an OS thread, then ties the CUDA context to the thread.
// Do not call in a goroutine.
//
// Good:
/*
	func main() {
		dev, _ := GetDevice(0)
		ctx, _ := dev.MakeContext()
		if err := ctx.Lock(); err != nil{
			// handle error
		}

		mem, _ := MemAlloc(1024)
	}
*/
// Bad:
/*
	func main() {
		dev, _ := GetDevice(0)
		ctx, _ := dev.MakeContext()
		go ctx.Lock() // this will tie the goroutine that calls ctx.Lock to the OS thread, while the main thread does not get the lock
		mem, _ := MemAlloc(1024)
	}
*/
func (ctx CUContext) Lock() error {
	runtime.LockOSThread()
	return SetCurrentContext(ctx)
}

// Unlock unlocks unbinds the goroutine from the OS thread
func (ctx CUContext) Unlock() error {
	if err := Synchronize(); err != nil {
		return err
	}
	runtime.UnlockOSThread()
	return nil
}

// Destroy destroys the context. It returns an error if it wasn't properly destroyed
//
// Wrapper over cuCtxDestroy: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e
func (ctx *CUContext) Destroy() error {
	err := result(C.cuCtxDestroy(ctx.ctx))
	*ctx = CUContext{}
	return err
}

// RetainPrimaryCtx retains the primary context on the GPU, creating it if necessary, increasing its usage count.
//
// The caller must call d.ReleasePrimaryCtx() when done using the context.
// Unlike MakeContext() the newly created context is not pushed onto the stack.
//
// Context creation will fail with error `UnknownError` if the compute mode of the device is CU_COMPUTEMODE_PROHIBITED.
// The function cuDeviceGetAttribute() can be used with CU_DEVICE_ATTRIBUTE_COMPUTE_MODE to determine the compute mode of the device.
// The nvidia-smi tool can be used to set the compute mode for devices. Documentation for nvidia-smi can be obtained by passing a -h option to it.
// Please note that the primary context always supports pinned allocations. Other flags can be specified by cuDevicePrimaryCtxSetFlags().
func (d Device) RetainPrimaryCtx() (primaryContext CUContext, err error) {
	if err = result(C.cuDevicePrimaryCtxRetain(&primaryContext.ctx, C.CUdevice(d))); err != nil {
		return
	}
	return primaryContext, nil
}
