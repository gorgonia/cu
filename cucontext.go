package cu

// #include <cuda.h>
import "C"
import (
	"fmt"
	"unsafe"
)

// CUContext is a CUDA context
type CUContext uintptr

func (ctx CUContext) String() string { return fmt.Sprintf("0x%x", uintptr(ctx)) }

func makeContext(ctx C.CUcontext) CUContext {
	return CUContext(uintptr(unsafe.Pointer(ctx)))
}

// C returns the CUContext as its C version
func (ctx CUContext) c() C.CUcontext { return C.CUcontext(unsafe.Pointer(uintptr(ctx))) }

func (d Device) MakeContext(flags ContextFlags) (CUContext, error) {
	var ctx C.CUcontext
	if err := result(C.cuCtxCreate(&ctx, C.uint(flags), C.CUdevice(d))); err != nil {
		return 0, err
	}
	return makeContext(ctx), nil
}

// DestroyContext destroys the context. It returns an error if it wasn't properly destroyed
//
// Wrapper over cuCtxDestroy: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e
func DestroyContext(ctx *CUContext) error {
	if err := result(C.cuCtxDestroy(C.CUcontext(unsafe.Pointer(uintptr(*ctx))))); err != nil {
		return err
	}
	*ctx = 0
	return nil
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
	var ctx C.CUcontext
	if err = result(C.cuDevicePrimaryCtxRetain(&ctx, C.CUdevice(d))); err != nil {
		return
	}
	return makeContext(ctx), nil
}
