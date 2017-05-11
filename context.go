package cu

// #include <cuda.h>
import "C"
import (
	"fmt"
	"unsafe"

	"github.com/pkg/errors"
)

// Context is a CUDA context
type Context uintptr

func (ctx Context) String() string { return fmt.Sprintf("0x%x", uintptr(ctx)) }

// DestroyContext destroys the context. It returns an error if it wasn't properly destroyed
//
// Wrapper over cuCtxDestroy: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e
func DestroyContext(ctx *Context) error {
	if err := result(C.cuCtxDestroy(C.CUcontext(unsafe.Pointer(uintptr(*ctx))))); err != nil {
		return err
	}
	*ctx = 0
	return nil
}

// MakeContext creates a new context on the device
//
// Wrapper over cuCtxCreate: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g65dc0012348bc84810e2103a40d8e2cf
func (d Device) MakeContext(flags ContextFlags) (Context, error) {
	var ctx C.CUcontext
	if err := result(C.cuCtxCreate(&ctx, C.uint(flags), C.CUdevice(d))); err != nil {
		return 0, err
	}
	return Context(uintptr(unsafe.Pointer(ctx))), nil
}

// APIVersion returns the API version used to create this context.
//
// Wrapper over cuCtxGetApiVersion: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb
func (ctx Context) APIVersion() (int, error) {
	var v C.uint
	if err := result(C.cuCtxGetApiVersion(C.CUcontext(unsafe.Pointer(uintptr(ctx))), &v)); err != nil {
		return -1, err
	}
	return int(v), nil
}

// CurrentCacheConfig returns the preferred cache configuration for the current context.
//
// Wrapper over cuCtxGetCacheConfig: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360
func CurrentCacheConfig() (FuncCacheConfig, error) {
	var fcc C.CUfunc_cache
	if err := result(C.cuCtxGetCacheConfig(&fcc)); err != nil {
		return 0, err
	}
	return FuncCacheConfig(fcc), nil
}

// CurrentContext returns the CUDA context bound to the calling CPU thread.
//
// Wrapper over cuCtxGetCurrent: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g8f13165846b73750693640fb3e8380d0
func CurrentContext() (Context, error) {
	var ctx C.CUcontext
	if err := result(C.cuCtxGetCurrent(&ctx)); err != nil {
		return 0, err
	}
	if uintptr(unsafe.Pointer(ctx)) == 0 {
		return 0, errors.Errorf("WTF")
	}
	return Context(uintptr(unsafe.Pointer(ctx))), nil
}

// CurrentDevice returns the device ID for the current context.
//
// Wrapper over cuCtxGetDevice: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e
func CurrentDevice() (Device, error) {
	var dev C.CUdevice
	if err := result(C.cuCtxGetDevice(&dev)); err != nil {
		return 0, err
	}
	return Device(dev), nil
}

// CurrentFlags returns the flags for the current context.
//
// Wrapper over cuCtxGetFlags: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d
func CurrentFlags() (ContextFlags, error) {
	var f C.uint
	if err := result(C.cuCtxGetFlags(&f)); err != nil {
		return 0, err
	}
	return ContextFlags(f), nil
}

// Limits returns resource limits. This allows for querying for information regarding the current context
//
// Wrapper over cuCtxGetLimit: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8
func Limits(l Limit) (int64, error) {
	var size C.size_t
	if err := result(C.cuCtxGetLimit(&size, C.CUlimit(l))); err != nil {
		return 0, err
	}
	return int64(size), nil
}

// SharedMemConfig returns the current shared memory configuration for the current context.
//
// Wrapper over cuCtxGetSharedMemConfig: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g17153a1b8b8c756f7ab8505686a4ad74
func SharedMemConfig() (SharedConfig, error) {
	var c C.CUsharedconfig
	if err := result(C.cuCtxGetSharedMemConfig(&c)); err != nil {
		return 0, err
	}
	return SharedConfig(c), nil
}

// // StreamPriorityRange numerical values that correspond to the least and greatest stream priorities.
// //
// // Wrapper over cuCtxGetStreamPriorityRange: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091
// func StreamPriorityRange() (leastPriority, greatestPriority int, err error) {

// }

// PopCurrentCtx pops the current CUDA context from the current CPU thread.
//
// Wrapper over cuCtxPopCurrent: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902
func PopCurrentCtx() (Context, error) {
	var ctx C.CUcontext
	if err := result(C.cuCtxPopCurrent(&ctx)); err != nil {
		return 0, err
	}
	return Context(uintptr(unsafe.Pointer(ctx))), nil
}

// PushCurrentCtx pushes a context on the current CPU thread.
//
// Wrapper over cuCtxPushCurrent: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba
func PushCurrentCtx(ctx Context) error {
	cctx := C.CUcontext(unsafe.Pointer(uintptr(ctx)))
	return result(C.cuCtxPushCurrent(cctx))
}

// SetCurrentCacheConfig sets the preferred cache configuration for the current context.
//
// Wrapper over cuCtxSetCacheConfig: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3
func SetCurrentCacheConfig(c FuncCacheConfig) error {
	fcc := C.CUfunc_cache(c)
	return result(C.cuCtxSetCacheConfig(fcc))
}

// SetCurrent binds the specified CUDA context to the calling CPU thread.
//
// Wrapper over cuCtxSetCurrent: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gbe562ee6258b4fcc272ca6478ca2a2f7
func SetCurrent(ctx Context) error {
	cctx := C.CUcontext(unsafe.Pointer(uintptr(ctx)))
	return result(C.cuCtxSetCurrent(cctx))
}

// SetLimit set resoucer limits.
//
// Wrapper over cuCtxSetLimit: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a
func SetLimit(l Limit, size int64) error {
	cSize := C.size_t(size)
	cLimit := C.CUlimit(l)
	return result(C.cuCtxSetLimit(cLimit, cSize))
}

// SetShharedMemConfig sets the shared memory configuration for the current context.
func SetSharedMemConfig(c SharedConfig) error {
	conf := C.CUsharedconfig(c)
	return result(C.cuCtxSetSharedMemConfig(conf))
}

// Synchronize blocks for a context's tasks to complete.
func Synchronize() error {
	return result(C.cuCtxSynchronize())
}

/* Primary Context Management */

// PrimaryCtxState gets the state of the primary context
func (d Device) PrimaryCtxState() (flags ContextFlags, active bool, err error) {
	var f C.uint
	var a C.int
	if err = result(C.cuDevicePrimaryCtxGetState(C.CUdevice(d), &f, &a)); err != nil {
		return
	}
	flags = ContextFlags(f)
	act := int(a)
	if act == 1 {
		active = true
	}
	return
}

// ReleasePrimaryCtx releases the primary context on the GPU.
//
// Releases the primary context interop on the device by decreasing the usage count by 1.
// If the usage drops to 0 the primary context of device dev will be destroyed regardless of how many threads it is current to.
// Please note that unlike cuCtxDestroy() this method does not pop the context from stack in any circumstances.
func (d Device) ReleasePrimaryCtx() error {
	return result(C.cuDevicePrimaryCtxRelease(C.CUdevice(d)))
}

// ResetPrimaryCtx destroys all allocations and reset all state on the primary context.
// Explicitly destroys and cleans up all resources associated with the current device in the current process.
//
// Note that it is responsibility of the calling function to ensure that no other module in the process is using the device any more.
// For that reason it is recommended to use d.ReleasePrimaryCtx() in most cases.
// However it is safe for other modules to call d.ReleasePrimaryCtx() even after resetting the device.
func (d Device) ResetPrimaryCtx() error {
	return result(C.cuDevicePrimaryCtxReset(C.CUdevice(d)))
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
func (d Device) RetainPrimaryCtx() (primaryContext Context, err error) {
	var ctx C.CUcontext
	if err = result(C.cuDevicePrimaryCtxRetain(&ctx, C.CUdevice(d))); err != nil {
		return
	}
	return Context(uintptr(unsafe.Pointer(ctx))), nil
}

// SetPrimaryCtxFlags Sets the flags for the primary context on the device overwriting perviously set ones.
// If the primary context is already created, error `PrimaryContextActive` will be returned.
func (d Device) SetPrimaryCtxFlags(flags ContextFlags) error {
	return result(C.cuDevicePrimaryCtxSetFlags(C.CUdevice(d), C.uint(flags)))
}
