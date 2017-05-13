package cu

//#include <cuda.h>
import "C"
import (
	"unsafe"

	"github.com/pkg/errors"
)

// Make3DArray creates a 3D CUDA array.
func (ctx *Standalone) Make3DArray(desc Array3Desc) (Array, error) {
	var arr C.CUarray
	cdesc := desc.cstruct()

	f := func() error {
		return result(C.cuArray3DCreate(&arr, cdesc))
	}

	if err := ctx.Do(f); err != nil {
		return 0, errors.Wrapf(err, "Make3DArray")
	}
	return cuArrayToArray(&arr), nil
}

// MakeArray creates a CUDA array
func (ctx *Standalone) MakeArray(desc ArrayDesc) (Array, error) {
	var arr C.CUarray
	cdesc := desc.cstruct()

	f := func() error {
		return result(C.cuArrayCreate(&arr, cdesc))
	}

	if err := ctx.Do(f); err != nil {
		return 0, errors.Wrapf(err, "MakeArray")
	}
	return cuArrayToArray(&arr), nil
}

// DestroyArray destroys a CUDA array
func (ctx *Standalone) DestroyArray(arr Array) error {
	hArray := arr.c()
	f := func() error { return result(C.cuArrayDestroy(hArray)) }
	return ctx.Do(f)
}

// MemAlloc allocates device memory.
//
// Allocates bytesize bytes of linear memory on the device and returns in a pointer to the allocated memory.
// The allocated memory is suitably aligned for any kind of variable. The memory is not cleared.
//
// If bytesize is 0, cuMemAlloc() returns error `InvalidValue`.
func (ctx *Standalone) MemAlloc(bytesize int64) (retVal DevicePtr, err error) {
	var d C.CUdeviceptr
	f := func() (err error) {
		return result(C.cuMemAlloc(&d, C.size_t(bytesize)))
	}

	if err = ctx.Do(f); err != nil {
		err = errors.Wrapf(err, "MemAlloc")
		return
	}

	retVal = DevicePtr(d)
	return
}

// MemAllocHost allocates bytesize bytes of host memory that is page-locked and accessible to the device.
// The driver tracks the virtual memory ranges allocated with this function and automatically accelerates calls to functions such as `Memcpy()`.
// Since the memory can be accessed directly by the device, it can be read or written with much higher bandwidth than
// pageable memory obtained with functions such as `malloc()`.
//
// Allocating excessive amounts of memory with `MemAllocHost()` may degrade system performance, since it reduces the amount of memory available to the system for paging.
// As a result, this function is best used sparingly to allocate staging areas for data exchange between host and device.
//
//Note all host memory allocated using MemHostAlloc() will automatically be immediately accessible to all contexts on all devices which support unified addressing (as may be queried using CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING). The device pointer that may be used to access this host memory from those contexts is always equal to the returned host pointer *pp. See Unified Addressing for additional details.
func (ctx *Standalone) MemAllocHost(byteSize int64) (unsafe.Pointer, error) {
	var p unsafe.Pointer

	f := func() (err error) { return result(C.cuMemAllocHost(&p, C.size_t(byteSize))) }
	if err := ctx.Do(f); err != nil {
		return nil, errors.Wrapf(err, "MemAllocHost")
	}
	return p, nil
}

// MemAllocManaged allocates memory that will be automatically managed by the Unified Memory system.
//
// Allocates bytesize bytes of managed memory on the device and returns in a pointer to the allocated memory.
// If the device doesn't support allocating managed memory, the error `NotSupported` is returned.
// Support for managed memory can be queried using the device attribute `ManagedMemory`.
// The allocated memory is suitably aligned for any kind of variable. The memory is not cleared.
// The pointer is valid on the CPU and on all GPUs in the system that support managed memory. All accesses to this pointer must obey the Unified Memory programming model.
//
// If bytesize is 0, cuMemAllocManaged returns error `InvalidValue`.
//
// More information: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32
func (ctx *Standalone) MemAllocManaged(bytesize int64, flags MemAttachFlags) (DevicePtr, error) {
	if bytesize == 0 {
		return 0, errors.Wrapf(InvalidValue, "Cannot allocate memory with size 0")
	}

	var d C.CUdeviceptr

	f := func() (err error) { return result(C.cuMemAllocManaged(&d, C.size_t(bytesize), C.uint(flags))) }
	if err := ctx.Do(f); err != nil {
		return 0, errors.Wrapf(err, "MemAllocManaged")
	}
	return DevicePtr(d), nil
}

// MemAllocPitch allocates pitched device memory.
// MemAllocPitch allocates at least `widthInBytes * height` bytes of linear memory on the device and returns in a pointer to the allocated memory.
// The function may pad the allocation to ensure that corresponding pointers in any given row will continue to meet the alignment requirements for coalescing as the address is updated from row to row.
//
// `elementSizeBytes` specifies the size of the largest reads and writes that will be performed on the memory range.
// `elementSizeBytes` may be 4, 8 or 16 (since coalesced memory transactions are not possible on other data sizes).
// If `elementSizeBytes` is smaller than the actual read/write size of a kernel, the kernel will run correctly, but possibly at reduced speed.
//
// The pitch returned in pitch by MemAllocPitch() is the width in bytes of the allocation.
// The intended usage of pitch is as a separate parameter of the allocation, used to compute addresses within the 2D array.
// Given the row and column of an array element of type T, the address is computed as:
//
// 		T* pElement = (T*)((char*)BaseAddress + Row * Pitch) + Column;
//
// The pitch returned by MemAllocPitch() is guaranteed to work with Memcpy2D() under all circumstances.
// For allocations of 2D arrays, it is recommended that programmers consider performing pitch allocations using MemAllocPitch().
// Due to alignment restrictions in the hardware, this is especially true if the application will be performing 2D memory copies between different regions of device memory (whether linear memory or CUDA arrays).
//
// The byte alignment of the pitch returned by `MemAllocPitch()` is guaranteed to match or exceed the alignment requirement for texture binding with `TexRefSetAddress2D()`.
func (ctx *Standalone) MemAllocPitch(widthInBytes, height int64, elementSizeBytes uint) (DevicePtr, int64, error) {
	var d C.CUdeviceptr
	var p C.size_t

	f := func() (err error) {
		return result(C.cuMemAllocPitch(&d, &p, C.size_t(widthInBytes), C.size_t(height), C.uint(elementSizeBytes)))
	}
	if err := ctx.Do(f); err != nil {
		return 0, 0, errors.Wrapf(err, "MemAllocPitch")
	}
	return DevicePtr(d), int64(p), nil
}

// MemFree frees device memory.
func (ctx *Standalone) MemFree(d DevicePtr) error {
	if d == DevicePtr(uintptr(0)) {
		return nil // Allready freed
	}

	f := func() (err error) { return result(C.cuMemFree(C.CUdeviceptr(d))) }
	if err := ctx.Do(f); err != nil {
		return errors.Wrapf(err, "MemFree")
	}
	return nil
}

// MemFreeHost frees the paged-locked memory space pointed to by p, which must have been returned by a previous call to `MemAllocHost`.
func (ctx *Standalone) MemFreeHost(p unsafe.Pointer) error {
	f := func() error { return result(C.cuMemFreeHost(p)) }
	return ctx.Do(f)
}

// AddressRange get information on memory allocations.
//
// Returns the base address and size of the allocation by MemAlloc() or MemAllocPitch() that contains the input pointer d.
func (ctx *Standalone) AddressRange(d DevicePtr) (size int64, base DevicePtr, err error) {
	var s C.size_t
	var b C.CUdeviceptr
	f := func() error { return result(C.cuMemGetAddressRange(&b, &s, C.CUdeviceptr(d))) }
	if err = ctx.Do(f); err != nil {
		err = errors.Wrapf(err, "MemGetAddressRange")
		return
	}
	return int64(s), DevicePtr(b), nil
}

// MemInfo gets free and total memory.
func (ctx *Standalone) MemInfo() (free, total int64, err error) {
	var f, t C.size_t

	fn := func() error { return result(C.cuMemGetInfo(&f, &t)) }
	if err = ctx.Do(fn); err != nil {
		err = errors.Wrapf(err, "MemGetInfo")
		return
	}
	return int64(f), int64(t), nil
}

// Memcpy copies memory.
//
// Copies data between two pointers. dst and src are base pointers of the destination and source, respectively.
// `byteCount` specifies the number of bytes to copy.
//
// Note that this function infers the type of the transfer (host to host, host to device, device to device, or device to host) from the pointer values.
// This function is only allowed in contexts which support unified addressing.
func (ctx *Standalone) Memcpy(dst, src DevicePtr, byteCount int64) error {
	s := C.CUdeviceptr(src)
	d := C.CUdeviceptr(dst)
	f := func() error { return result(C.cuMemcpy(s, d, C.size_t(byteCount))) }
	return ctx.Do(f)
}

/*

// Memcpy2D copies memory for 2D arrays.
func (ctx *Standalone) Memcpy2D(CUDA_MEMCPY2D *pCopy) error {
	f := func()(err error) {
 return result(C.cuMemcpy2D())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "Memcpy2D") }
	return nil
}

// Memcpy2DAsync copies memory for 2D arrays.
func (ctx *Standalone) Memcpy2DAsync(pCopy *CUDA_MEMCPY2D, hStream CUstream) error {
	f := func()(err error) {
 return result(C.cuMemcpy2DAsync())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "Memcpy2DAsync") }
	return nil
}

// Memcpy2DUnaligned copies memory for 2D arrays.
func (ctx *Standalone) Memcpy2DUnaligned(CUDA_MEMCPY2D *pCopy) error {
	f := func()(err error) {
 return result(C.cuMemcpy2DUnaligned())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "Memcpy2DUnaligned") }
	return nil
}

// Memcpy3D copies memory for 3D arrays.
func (ctx *Standalone) Memcpy3D(CUDA_MEMCPY3D *pCopy) error {
	f := func()(err error) {
 return result(C.cuMemcpy3D())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "Memcpy3D") }
	return nil
}

// Memcpy3DAsync copies memory for 3D arrays.
func (ctx *Standalone) Memcpy3DAsync(pCopy *CUDA_MEMCPY3D, hStream CUstream) error {
	f := func()(err error) {
 return result(C.cuMemcpy3DAsync())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "Memcpy3DAsync") }
	return nil
}

// Memcpy3DPeer copies memory between contexts.
func (ctx *Standalone) Memcpy3DPeer(CUDA_MEMCPY3D_PEER *pCopy) error {
	f := func()(err error) {
 return result(C.cuMemcpy3DPeer())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "Memcpy3DPeer") }
	return nil
}

// Memcpy3DPeerAsync copies memory between contexts asynchronously.
func (ctx *Standalone) Memcpy3DPeerAsync(pCopy *CUDA_MEMCPY3D_PEER, hStream CUstream) error {
	f := func()(err error) {
 return result(C.cuMemcpy3DPeerAsync())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "Memcpy3DPeerAsync") }
	return nil
}

*/

// MemcpyAsync copies data between two pointers.
//
// `dst` and `src` are base pointers of the destination and source, respectively.
// `byteCount` specifies the number of bytes to copy.
// Note that this function infers the type of the transfer (host to host, host to device, device to device, or device to host) from the pointer values.
//
// This function is only allowed in contexts which support unified addressing.
func (ctx *Standalone) MemcpyAsync(dst, src DevicePtr, byteCount int64, hStream Stream) error {
	d := C.CUdeviceptr(dst)
	s := C.CUdeviceptr(src)
	size := C.size_t(byteCount)
	stream := C.CUstream(unsafe.Pointer(uintptr(hStream)))

	f := func() error { return result(C.cuMemcpyAsync(d, s, size, stream)) }
	return ctx.Do(f)
}

// // MemcpyAtoA copies memory from Array to Array.
// func (ctx *Standalone) MemcpyAtoA(dst Array, dstOffset int64, src Array, srcOffset int64, byteCount int64) error {

// 	if err := result(C.cuMemcpyAtoA()); err != nil {
// 		return errors.Wrapf(err, "MemcpyAtoA")
// 	}
// 	return nil
// }

// // MemcpyAtoD copies memory from Array to Device.
// func (ctx *Standalone) MemcpyAtoD(dstDevice CUdeviceptr, CUarray srcArray, srcOffset size_t, ByteCount size_t) error {
// 	if err := result(C.cuMemcpyAtoD()); err != nil {
// 		return errors.Wrapf(err, "MemcpyAtoD")
// 	}
// 	return nil
// }

// MemCpyDtoD copies a number of bytes from host to device.
func (ctx *Standalone) MemcpyDtoD(dst, src DevicePtr, byteCount int64) error {
	d := C.CUdeviceptr(dst)
	s := C.CUdeviceptr(src)
	size := C.size_t(byteCount)
	f := func() error {
		return result(C.cuMemcpyDtoD(d, s, size))
	}
	return ctx.Do(f)
}

// MemcpyDtoDAsync asynchronously copies a number of bytes from host to device.
func (ctx *Standalone) MemcpyDtoDAsync(dst, src DevicePtr, byteCount int64, hStream Stream) error {
	d := C.CUdeviceptr(dst)
	s := C.CUdeviceptr(src)
	size := C.size_t(byteCount)
	stream := C.CUstream(unsafe.Pointer(uintptr(hStream)))
	f := func() error { return result(C.cuMemcpyDtoDAsync(d, s, size, stream)) }
	return ctx.Do(f)
}

// MemcpyHtoD copies a number of bytes from host to device.
func (ctx *Standalone) MemcpyHtoD(dst DevicePtr, src unsafe.Pointer, byteCount int64) error {
	d := C.CUdeviceptr(dst)
	size := C.size_t(byteCount)
	f := func() error { return result(C.cuMemcpyHtoD(d, src, size)) }
	return ctx.Do(f)
}

// MemcpyHtoDAsync asynchronously copies a number of bytes from host to device.
// The host memory must be page-locked (see MemRegister)
func (ctx *Standalone) MemcpyHtoDAsync(dst DevicePtr, src unsafe.Pointer, byteCount int64, hStream Stream) error {
	d := C.CUdeviceptr(dst)
	size := C.size_t(byteCount)
	stream := C.CUstream(unsafe.Pointer(uintptr(hStream)))
	f := func() error { return result(C.cuMemcpyHtoDAsync(d, src, size, stream)) }
	return ctx.Do(f)
}

// MemcpyDtoH copies a number of bytes from device to host.
func (ctx *Standalone) MemcpyDtoH(dst unsafe.Pointer, src DevicePtr, byteCount int64) error {
	s := C.CUdeviceptr(src)
	size := C.size_t(byteCount)
	f := func() error { return result(C.cuMemcpyDtoH(dst, s, size)) }
	return ctx.Do(f)
}

// MemcpyDtoHAsync asynchronously copies a number of bytes device host to host.
// The host memory must be page-locked (see MemRegister)
func (ctx *Standalone) MemcpyDtoHAsync(dst unsafe.Pointer, src DevicePtr, byteCount int64, hStream Stream) error {
	s := C.CUdeviceptr(src)
	size := C.size_t(byteCount)
	stream := C.CUstream(unsafe.Pointer(uintptr(hStream)))
	f := func() error { return result(C.cuMemcpyDtoHAsync(dst, s, size, stream)) }
	return ctx.Do(f)
}

// MemcpyPeer copies from device memory in one context (device) to another.
func (ctx *Standalone) MemcpyPeer(dst DevicePtr, dstCtx CUContext, src DevicePtr, srcCtx CUContext, byteCount int64) error {
	d := C.CUdeviceptr(dst)
	s := C.CUdeviceptr(src)
	dctx := C.CUcontext(unsafe.Pointer(uintptr(dstCtx)))
	sctx := C.CUcontext(unsafe.Pointer(uintptr(srcCtx)))
	size := C.size_t(byteCount)
	f := func() error { return result(C.cuMemcpyPeer(d, dctx, s, sctx, size)) }
	return ctx.Do(f)
}

// MemcpyPeerAsync asynchronously copies from device memory in one context (device) to another.
func (ctx *Standalone) MemcpyPeerAsync(dst DevicePtr, dstCtx CUContext, src DevicePtr, srcCtx CUContext, byteCount int64, hStream Stream) error {
	d := C.CUdeviceptr(dst)
	s := C.CUdeviceptr(src)
	dctx := C.CUcontext(unsafe.Pointer(uintptr(dstCtx)))
	sctx := C.CUcontext(unsafe.Pointer(uintptr(srcCtx)))
	size := C.size_t(byteCount)
	stream := C.CUstream(unsafe.Pointer(uintptr(hStream)))
	f := func() error { return result(C.cuMemcpyPeerAsync(d, dctx, s, sctx, size, stream)) }
	return ctx.Do(f)
}

// MemsetD32 sets the first N 32-bit values of dst array to value.
// Asynchronous.
func (ctx *Standalone) MemsetD32(mem DevicePtr, value uint32, N int64) error {
	d := C.CUdeviceptr(mem)
	v := C.uint(value)
	n := C.size_t(N)
	f := func() error { return result(C.cuMemsetD32(d, v, n)) }
	return ctx.Do(f)
}

// MemsetD32Async asynchronously sets the first N 32-bit values of dst array to value.
func (ctx *Standalone) MemsetD32Async(mem DevicePtr, value uint32, N int64, hStream Stream) error {
	d := C.CUdeviceptr(mem)
	v := C.uint(value)
	n := C.size_t(N)
	s := C.CUstream(unsafe.Pointer(uintptr(hStream)))
	f := func() error { return result(C.cuMemsetD32Async(d, v, n, s)) }
	return ctx.Do(f)
}

// MemsetD8 sets the first N 8-bit values of dst array to value.
// Asynchronous.
func (ctx *Standalone) MemsetD8(mem DevicePtr, value uint8, N int64) error {
	d := C.CUdeviceptr(mem)
	v := C.uchar(value)
	n := C.size_t(N)
	f := func() error { return result(C.cuMemsetD8(d, v, n)) }
	return ctx.Do(f)
}

// MemsetD8Async asynchronously sets the first N 32-bit values of dst array to value.
func (ctx *Standalone) MemsetD8Async(mem DevicePtr, value uint8, N int64, hStream Stream) error {
	d := C.CUdeviceptr(mem)
	v := C.uchar(value)
	n := C.size_t(N)
	s := C.CUstream(unsafe.Pointer(uintptr(hStream)))
	f := func() error { return result(C.cuMemsetD8Async(d, v, n, s)) }
	return ctx.Do(f)
}

/*

// DeviceGetByPCIBusId returns a handle to a compute device.
func DeviceGetByPCIBusId(dev *CUdevice, pciBusId *char) error {
	f := func()(err error) {
 return result(C.cuDeviceGetByPCIBusId())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "DeviceGetByPCIBusId") }
	return nil
}

// DeviceGetPCIBusId returns a PCI Bus Id string for the device.
func DeviceGetPCIBusId(pciBusId *char, int len, CUdevice dev) error {
	f := func()(err error) {
 return result(C.cuDeviceGetPCIBusId())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "DeviceGetPCIBusId") }
	return nil
}

// IpcCloseMemHandle close memory mapped with cuIpcOpenMemHandle.
func IpcCloseMemHandle(dptr CUdeviceptr) error {
	f := func()(err error) {
 return result(C.cuIpcCloseMemHandle())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "IpcCloseMemHandle") }
	return nil
}

// IpcGetEventHandle gets an interprocess handle for a previously allocated event.
func IpcGetEventHandle(pHandle *CUipcEventHandle, CUevent event) error {
	f := func()(err error) {
 return result(C.cuIpcGetEventHandle())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "IpcGetEventHandle") }
	return nil
}

// IpcGetMemHandle gets an interprocess memory handle for an existing device memory allocation.
func IpcGetMemHandle(pHandle *CUipcMemHandle, dptr CUdeviceptr) error {
	f := func()(err error) {
 return result(C.cuIpcGetMemHandle())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "IpcGetMemHandle") }
	return nil
}

// IpcOpenEventHandle opens an interprocess event handle for use in the current process.
func IpcOpenEventHandle(phEvent *CUevent, CUipcEventHandle handle) error {
	f := func()(err error) {
 return result(C.cuIpcOpenEventHandle())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "IpcOpenEventHandle") }
	return nil
}

// IpcOpenMemHandle opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.
func IpcOpenMemHandle(pdptr *CUdeviceptr, CUipcMemHandle handle, uint Flags) error {
	f := func()(err error) {
 return result(C.cuIpcOpenMemHandle())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "IpcOpenMemHandle") }
	return nil
}










// MemHostGetDevicePointer passes back device pointer of mapped pinned memory.
func (ctx *Standalone) MemHostGetDevicePointer(pdptr *CUdeviceptr, p *void, uint Flags) error {
	f := func()(err error) {
 return result(C.cuMemHostGetDevicePointer())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "MemHostGetDevicePointer") }
	return nil
}

// MemHostGetFlags passes back flags that were used for a pinned allocation.
func (ctx *Standalone) MemHostGetFlags(pFlags *uint, p *void) error {
	f := func()(err error) {
 return result(C.cuMemHostGetFlags())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "MemHostGetFlags") }
	return nil
}

// MemHostRegister registers an existing host memory range for use by CUDA.
func (ctx *Standalone) MemHostRegister(p *void, bytesize size_t, uint Flags) error {
	f := func()(err error) {
 return result(C.cuMemHostRegister())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "MemHostRegister") }
	return nil
}

// MemHostUnregister unregisters a memory range that was registered with cuMemHostRegister.
func (ctx *Standalone) MemHostUnregister(void *p) error {
	f := func()(err error) {
 return result(C.cuMemHostUnregister())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "MemHostUnregister") }
	return nil
}






















// MemcpyAtoH copies memory from Array to Host.
func (ctx *Standalone) MemcpyAtoH(dstHost *void, CUarray srcArray, srcOffset size_t, ByteCount size_t) error {
	f := func()(err error) {
 return result(C.cuMemcpyAtoH())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "MemcpyAtoH") }
	return nil
}

// MemcpyAtoHAsync copies memory from Array to Host.
func (ctx *Standalone) MemcpyAtoHAsync(dstHost *void, CUarray srcArray, srcOffset size_t, ByteCount size_t, hStream CUstream) error {
	f := func()(err error) {
 return result(C.cuMemcpyAtoHAsync())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "MemcpyAtoHAsync") }
	return nil
}

// MemcpyDtoA copies memory from Device to Array.
func (ctx *Standalone) MemcpyDtoA(CUarray dstArray, dstOffset size_t, srcDevice CUdeviceptr, ByteCount size_t) error {
	f := func()(err error) {
 return result(C.cuMemcpyDtoA())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "MemcpyDtoA") }
	return nil
}


// MemcpyHtoA copies memory from Host to Array.
func (ctx *Standalone) MemcpyHtoA(srcHost *CUarray, dstArray, dstOffset size_t, void, ByteCount size_t) error {
	f := func()(err error) {
 return result(C.cuMemcpyHtoA())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "MemcpyHtoA") }
	return nil
}

// MemcpyHtoAAsync copies memory from Host to Array.
func (ctx *Standalone) MemcpyHtoAAsync(srcHost *CUarray, dstArray, dstOffset size_t, void, ByteCount size_t, hStream CUstream) error {
	f := func()(err error) {
 return result(C.cuMemcpyHtoAAsync())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "MemcpyHtoAAsync") }
	return nil
}










// MemsetD16 initializes device memory.
func (ctx *Standalone) MemsetD16(dstDevice CUdeviceptr, us uint16, N size_t) error {
	f := func()(err error) {
 return result(C.cuMemsetD16())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "MemsetD16") }
	return nil
}

// MemsetD16Async sets device memory.
func (ctx *Standalone) MemsetD16Async(dstDevice CUdeviceptr, us uint16, N size_t, hStream CUstream) error {
	f := func()(err error) {
 return result(C.cuMemsetD16Async())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "MemsetD16Async") }
	return nil
}

// MemsetD2D16 initializes device memory.
func (ctx *Standalone) MemsetD2D16(dstDevice CUdeviceptr, dstPitch size_t, us uint16, Width size_t, Height size_t) error {
	f := func()(err error) {
 return result(C.cuMemsetD2D16())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "MemsetD2D16") }
	return nil
}

// MemsetD2D16Async sets device memory.
func (ctx *Standalone) MemsetD2D16Async(dstDevice CUdeviceptr, dstPitch size_t, us uint16, Width size_t, Height size_t, CUstream hStream) error {
	f := func()(err error) {
 return result(C.cuMemsetD2D16Async())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "MemsetD2D16Async") }
	return nil
}

// MemsetD2D32 initializes device memory.
func (ctx *Standalone) MemsetD2D32(dstDevice CUdeviceptr, dstPitch size_t, uint ui, Width size_t, Height size_t) error {
	f := func()(err error) {
 return result(C.cuMemsetD2D32())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "MemsetD2D32") }
	return nil
}

// MemsetD2D32Async sets device memory.
func (ctx *Standalone) MemsetD2D32Async(dstDevice CUdeviceptr, dstPitch size_t, uint ui, Width size_t, Height size_t, hStream CUstream) error {
	f := func()(err error) {
 return result(C.cuMemsetD2D32Async())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "MemsetD2D32Async") }
	return nil
}

// MemsetD2D8 initializes device memory.
func (ctx *Standalone) MemsetD2D8(dstDevice CUdeviceptr, dstPitch size_t, uc uint8, Width size_t, Height size_t) error {
	f := func()(err error) {
 return result(C.cuMemsetD2D8())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "MemsetD2D8") }
	return nil
}

// MemsetD2D8Async sets device memory.
func (ctx *Standalone) MemsetD2D8Async(dstDevice CUdeviceptr, dstPitch size_t, uc uint8, Width size_t, Height size_t, hStream CUstream) error {
	f := func()(err error) {
 return result(C.cuMemsetD2D8Async())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "MemsetD2D8Async") }
	return nil
}









// MipmappedArrayCreate creates a CUDA mipmapped array.
func MipmappedArrayCreate(pHandle *CUmipmappedArray, pMipmappedArrayDesc *CUDA_ARRAY3D_DESCRIPTOR, uint numMipmapLevels) error {
	f := func()(err error) {
 return result(C.cuMipmappedArrayCreate())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "MipmappedArrayCreate") }
	return nil
}

// MipmappedArrayDestroy destroys a CUDA mipmapped array.
func MipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) error {
	f := func()(err error) {
 return result(C.cuMipmappedArrayDestroy())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "MipmappedArrayDestroy") }
	return nil
}

// MipmappedArrayGetLevel gets a mipmap level of a CUDA mipmapped array.
func MipmappedArrayGetLevel(pLevelArray *CUarray, CUmipmappedArray hMipmappedArray, uint level) error {
	f := func()(err error) {
 return result(C.cuMipmappedArrayGetLevel())}
 if err := ctx.Do(f); err != nil { return 0, errors.Wrapf(err, "MipmappedArrayGetLevel") }
	return nil
}
*/
