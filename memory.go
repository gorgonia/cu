package cu

//#include <cuda.h>
import "C"
import (
	"fmt"
	"unsafe"

	"github.com/pkg/errors"
)

// DevicePtr is a pointer to the device memory. It is equivalent to CUDA's CUdeviceptr
type DevicePtr uintptr

func (d DevicePtr) String() string { return fmt.Sprintf("0x%x", uintptr(d)) }

// Make3DArray creates a 3D CUDA array.
func Make3DArray(desc Array3Desc) (Array, error) {
	var arr C.CUarray
	cdesc := desc.cstruct()
	if err := result(C.cuArray3DCreate(&arr, cdesc)); err != nil {
		return 0, errors.Wrapf(err, "Array3DCreate")
	}
	return Array(uintptr(unsafe.Pointer(&arr))), nil
}

// MakeArray creates a CUDA array
func MakeArray(desc ArrayDesc) (Array, error) {
	var arr C.CUarray
	cdesc := desc.cstruct()
	if err := result(C.cuArrayCreate(&arr, cdesc)); err != nil {
		return 0, errors.Wrapf(err, "ArrayCreate")
	}
	return Array(uintptr(unsafe.Pointer(&arr))), nil
}

// Descriptor3 get a 3D CUDA array descriptor
func (arr Array) Descriptor3() (Array3Desc, error) {
	hArray := arr.cuda()
	var desc C.CUDA_ARRAY3D_DESCRIPTOR
	if err := result(C.cuArray3DGetDescriptor(&desc, hArray)); err != nil {
		return Array3Desc{}, errors.Wrapf(err, "Array3DGetDescriptor")
	}
	return goArray3Desc(&desc), nil
}

// Descriptor gets a 1D or 2D CUDA array descriptor
func (arr Array) Descriptor() (ArrayDesc, error) {
	hArray := arr.cuda()
	var desc C.CUDA_ARRAY_DESCRIPTOR
	if err := result(C.cuArrayGetDescriptor(&desc, hArray)); err != nil {
		return ArrayDesc{}, errors.Wrapf(err, "ArrayGetDescriptor")
	}
	return goArrayDesc(&desc), nil

}

// DestroyArray destroys a CUDA array
func DestroyArray(arr Array) error {
	hArray := arr.cuda()
	return result(C.cuArrayDestroy(hArray))
}

// MemAlloc allocates device memory.
//
// Allocates bytesize bytes of linear memory on the device and returns in a pointer to the allocated memory.
// The allocated memory is suitably aligned for any kind of variable. The memory is not cleared.
//
// If bytesize is 0, cuMemAlloc() returns error `InvalidValue`.
func MemAlloc(bytesize int64) (DevicePtr, error) {
	if bytesize == 0 {
		return 0, errors.Wrapf(InvalidValue, "Cannot allocate memory with size 0")
	}
	var d C.CUdeviceptr
	if err := result(C.cuMemAlloc(&d, C.size_t(bytesize))); err != nil {
		return 0, errors.Wrapf(err, "MemAlloc")
	}
	return DevicePtr(d), nil
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
func MemAllocHost(byteSize int64) (unsafe.Pointer, error) {
	var p unsafe.Pointer
	if err := result(C.cuMemAllocHost(&p, C.size_t(byteSize))); err != nil {
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
func MemAllocManaged(bytesize int64, flags MemAttachFlags) (DevicePtr, error) {
	if bytesize == 0 {
		return 0, errors.Wrapf(InvalidValue, "Cannot allocate memory with size 0")
	}

	var d C.CUdeviceptr
	if err := result(C.cuMemAllocManaged(&d, C.size_t(bytesize), C.uint(flags))); err != nil {
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
func MemAllocPitch(widthInBytes, height int64, elementSizeBytes uint) (DevicePtr, int64, error) {
	var d C.CUdeviceptr
	var p C.size_t
	if err := result(C.cuMemAllocPitch(&d, &p, C.size_t(widthInBytes), C.size_t(height), C.uint(elementSizeBytes))); err != nil {
		return 0, 0, errors.Wrapf(err, "MemAllocPitch")
	}
	return DevicePtr(d), int64(p), nil
}

// MemFree frees device memory.
func MemFree(d DevicePtr) error {
	if d == DevicePtr(uintptr(0)) {
		return nil // Allready freed
	}
	if err := result(C.cuMemFree(C.CUdeviceptr(d))); err != nil {
		return errors.Wrapf(err, "MemFree")
	}
	return nil
}

// MemFreeHost frees the paged-locked memory space pointed to by p, which must have been returned by a previous call to `MemAllocHost`.
func MemFreeHost(p unsafe.Pointer) error {
	return result(C.cuMemFreeHost(p))
}

// AddressRange get information on memory allocations.
//
// Returns the base address and size of the allocation by MemAlloc() or MemAllocPitch() that contains the input pointer d.
func (d DevicePtr) AddressRange() (size int64, base DevicePtr, err error) {
	var s C.size_t
	var b C.CUdeviceptr
	if err = result(C.cuMemGetAddressRange(&b, &s, C.CUdeviceptr(d))); err != nil {
		err = errors.Wrapf(err, "MemGetAddressRange")
		return
	}
	return int64(s), DevicePtr(b), nil
}

// MemInfo gets free and total memory.
func MemInfo() (free, total int64, err error) {
	var f, t C.size_t
	if err = result(C.cuMemGetInfo(&f, &t)); err != nil {
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
func Memcpy(dst, src DevicePtr, byteCount int64) error {
	s := C.CUdeviceptr(src)
	d := C.CUdeviceptr(dst)
	return result(C.cuMemcpy(s, d, C.size_t(byteCount)))
}

/*

// Memcpy2D copies memory for 2D arrays.
func Memcpy2D(CUDA_MEMCPY2D *pCopy) error {
	if err := result(C.cuMemcpy2D()); err != nil {
		return errors.Wrapf(err, "Memcpy2D")
	}
	return nil
}

// Memcpy2DAsync copies memory for 2D arrays.
func Memcpy2DAsync(pCopy *CUDA_MEMCPY2D, hStream CUstream) error {
	if err := result(C.cuMemcpy2DAsync()); err != nil {
		return errors.Wrapf(err, "Memcpy2DAsync")
	}
	return nil
}

// Memcpy2DUnaligned copies memory for 2D arrays.
func Memcpy2DUnaligned(CUDA_MEMCPY2D *pCopy) error {
	if err := result(C.cuMemcpy2DUnaligned()); err != nil {
		return errors.Wrapf(err, "Memcpy2DUnaligned")
	}
	return nil
}

// Memcpy3D copies memory for 3D arrays.
func Memcpy3D(CUDA_MEMCPY3D *pCopy) error {
	if err := result(C.cuMemcpy3D()); err != nil {
		return errors.Wrapf(err, "Memcpy3D")
	}
	return nil
}

// Memcpy3DAsync copies memory for 3D arrays.
func Memcpy3DAsync(pCopy *CUDA_MEMCPY3D, hStream CUstream) error {
	if err := result(C.cuMemcpy3DAsync()); err != nil {
		return errors.Wrapf(err, "Memcpy3DAsync")
	}
	return nil
}

// Memcpy3DPeer copies memory between contexts.
func Memcpy3DPeer(CUDA_MEMCPY3D_PEER *pCopy) error {
	if err := result(C.cuMemcpy3DPeer()); err != nil {
		return errors.Wrapf(err, "Memcpy3DPeer")
	}
	return nil
}

// Memcpy3DPeerAsync copies memory between contexts asynchronously.
func Memcpy3DPeerAsync(pCopy *CUDA_MEMCPY3D_PEER, hStream CUstream) error {
	if err := result(C.cuMemcpy3DPeerAsync()); err != nil {
		return errors.Wrapf(err, "Memcpy3DPeerAsync")
	}
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
func MemcpyAsync(dst, src DevicePtr, byteCount int64, hStream Stream) error {
	d := C.CUdeviceptr(dst)
	s := C.CUdeviceptr(src)
	size := C.size_t(byteCount)
	stream := C.CUstream(unsafe.Pointer(uintptr(hStream)))
	return result(C.cuMemcpyAsync(d, s, size, stream))
}

// // MemcpyAtoA copies memory from Array to Array.
// func MemcpyAtoA(dst Array, dstOffset int64, src Array, srcOffset int64, byteCount int64) error {

// 	if err := result(C.cuMemcpyAtoA()); err != nil {
// 		return errors.Wrapf(err, "MemcpyAtoA")
// 	}
// 	return nil
// }

// // MemcpyAtoD copies memory from Array to Device.
// func MemcpyAtoD(dstDevice CUdeviceptr, CUarray srcArray, srcOffset size_t, ByteCount size_t) error {
// 	if err := result(C.cuMemcpyAtoD()); err != nil {
// 		return errors.Wrapf(err, "MemcpyAtoD")
// 	}
// 	return nil
// }

// MemCpyDtoD copies a number of bytes from host to device.
func MemcpyDtoD(dst, src DevicePtr, byteCount int64) error {
	d := C.CUdeviceptr(dst)
	s := C.CUdeviceptr(src)
	size := C.size_t(byteCount)
	return result(C.cuMemcpyDtoD(d, s, size))
}

// MemcpyDtoDAsync asynchronously copies a number of bytes from host to device.
func MemcpyDtoDAsync(dst, src DevicePtr, byteCount int64, hStream Stream) error {
	d := C.CUdeviceptr(dst)
	s := C.CUdeviceptr(src)
	size := C.size_t(byteCount)
	stream := C.CUstream(unsafe.Pointer(uintptr(hStream)))
	return result(C.cuMemcpyDtoDAsync(d, s, size, stream))
}

// MemcpyHtoD copies a number of bytes from host to device.
func MemcpyHtoD(dst DevicePtr, src unsafe.Pointer, byteCount int64) error {
	d := C.CUdeviceptr(dst)
	size := C.size_t(byteCount)
	return result(C.cuMemcpyHtoD(d, src, size))
}

// MemcpyHtoDAsync asynchronously copies a number of bytes from host to device.
// The host memory must be page-locked (see MemRegister)
func MemcpyHtoDAsync(dst DevicePtr, src unsafe.Pointer, byteCount int64, hStream Stream) error {
	d := C.CUdeviceptr(dst)
	size := C.size_t(byteCount)
	stream := C.CUstream(unsafe.Pointer(uintptr(hStream)))
	return result(C.cuMemcpyHtoDAsync(d, src, size, stream))
}

// MemcpyDtoH copies a number of bytes from device to host.
func MemcpyDtoH(dst unsafe.Pointer, src DevicePtr, byteCount int64) error {
	s := C.CUdeviceptr(src)
	size := C.size_t(byteCount)
	return result(C.cuMemcpyDtoH(dst, s, size))
}

// MemcpyDtoHAsync asynchronously copies a number of bytes device host to host.
// The host memory must be page-locked (see MemRegister)
func MemcpyDtoHAsync(dst unsafe.Pointer, src DevicePtr, byteCount int64, hStream Stream) error {
	s := C.CUdeviceptr(src)
	size := C.size_t(byteCount)
	stream := C.CUstream(unsafe.Pointer(uintptr(hStream)))
	return result(C.cuMemcpyDtoHAsync(dst, s, size, stream))
}

// MemcpyPeer copies from device memory in one context (device) to another.
func MemcpyPeer(dst DevicePtr, dstCtx Context, src DevicePtr, srcCtx Context, byteCount int64) error {
	d := C.CUdeviceptr(dst)
	s := C.CUdeviceptr(src)
	dctx := C.CUcontext(unsafe.Pointer(uintptr(dstCtx)))
	sctx := C.CUcontext(unsafe.Pointer(uintptr(srcCtx)))
	size := C.size_t(byteCount)
	return result(C.cuMemcpyPeer(d, dctx, s, sctx, size))
}

// MemcpyPeerAsync asynchronously copies from device memory in one context (device) to another.
func MemcpyPeerAsync(dst DevicePtr, dstCtx Context, src DevicePtr, srcCtx Context, byteCount int64, hStream Stream) error {
	d := C.CUdeviceptr(dst)
	s := C.CUdeviceptr(src)
	dctx := C.CUcontext(unsafe.Pointer(uintptr(dstCtx)))
	sctx := C.CUcontext(unsafe.Pointer(uintptr(srcCtx)))
	size := C.size_t(byteCount)
	stream := C.CUstream(unsafe.Pointer(uintptr(hStream)))
	return result(C.cuMemcpyPeerAsync(d, dctx, s, sctx, size, stream))
}

// MemsetD32 sets the first N 32-bit values of dst array to value.
// Asynchronous.
func MemsetD32(mem DevicePtr, value uint32, N int64) error {
	d := C.CUdeviceptr(mem)
	v := C.uint(value)
	n := C.size_t(N)
	return result(C.cuMemsetD32(d, v, n))
}

// MemsetD32Async asynchronously sets the first N 32-bit values of dst array to value.
func MemsetD32Async(mem DevicePtr, value uint32, N int64, hStream Stream) error {
	d := C.CUdeviceptr(mem)
	v := C.uint(value)
	n := C.size_t(N)
	s := C.CUstream(unsafe.Pointer(uintptr(hStream)))
	return result(C.cuMemsetD32Async(d, v, n, s))
}

// MemsetD8 sets the first N 8-bit values of dst array to value.
// Asynchronous.
func MemsetD8(mem DevicePtr, value uint8, N int64) error {
	d := C.CUdeviceptr(mem)
	v := C.uchar(value)
	n := C.size_t(N)
	return result(C.cuMemsetD8(d, v, n))
}

// MemsetD8Async asynchronously sets the first N 32-bit values of dst array to value.
func MemsetD8Async(mem DevicePtr, value uint8, N int64, hStream Stream) error {
	d := C.CUdeviceptr(mem)
	v := C.uchar(value)
	n := C.size_t(N)
	s := C.CUstream(unsafe.Pointer(uintptr(hStream)))
	return result(C.cuMemsetD8Async(d, v, n, s))
}

/*

// DeviceGetByPCIBusId returns a handle to a compute device.
func DeviceGetByPCIBusId(dev *CUdevice, pciBusId *char) error {
	if err := result(C.cuDeviceGetByPCIBusId()); err != nil {
		return errors.Wrapf(err, "DeviceGetByPCIBusId")
	}
	return nil
}

// DeviceGetPCIBusId returns a PCI Bus Id string for the device.
func DeviceGetPCIBusId(pciBusId *char, int len, CUdevice dev) error {
	if err := result(C.cuDeviceGetPCIBusId()); err != nil {
		return errors.Wrapf(err, "DeviceGetPCIBusId")
	}
	return nil
}

// IpcCloseMemHandle close memory mapped with cuIpcOpenMemHandle.
func IpcCloseMemHandle(dptr CUdeviceptr) error {
	if err := result(C.cuIpcCloseMemHandle()); err != nil {
		return errors.Wrapf(err, "IpcCloseMemHandle")
	}
	return nil
}

// IpcGetEventHandle gets an interprocess handle for a previously allocated event.
func IpcGetEventHandle(pHandle *CUipcEventHandle, CUevent event) error {
	if err := result(C.cuIpcGetEventHandle()); err != nil {
		return errors.Wrapf(err, "IpcGetEventHandle")
	}
	return nil
}

// IpcGetMemHandle gets an interprocess memory handle for an existing device memory allocation.
func IpcGetMemHandle(pHandle *CUipcMemHandle, dptr CUdeviceptr) error {
	if err := result(C.cuIpcGetMemHandle()); err != nil {
		return errors.Wrapf(err, "IpcGetMemHandle")
	}
	return nil
}

// IpcOpenEventHandle opens an interprocess event handle for use in the current process.
func IpcOpenEventHandle(phEvent *CUevent, CUipcEventHandle handle) error {
	if err := result(C.cuIpcOpenEventHandle()); err != nil {
		return errors.Wrapf(err, "IpcOpenEventHandle")
	}
	return nil
}

// IpcOpenMemHandle opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.
func IpcOpenMemHandle(pdptr *CUdeviceptr, CUipcMemHandle handle, uint Flags) error {
	if err := result(C.cuIpcOpenMemHandle()); err != nil {
		return errors.Wrapf(err, "IpcOpenMemHandle")
	}
	return nil
}










// MemHostGetDevicePointer passes back device pointer of mapped pinned memory.
func MemHostGetDevicePointer(pdptr *CUdeviceptr, p *void, uint Flags) error {
	if err := result(C.cuMemHostGetDevicePointer()); err != nil {
		return errors.Wrapf(err, "MemHostGetDevicePointer")
	}
	return nil
}

// MemHostGetFlags passes back flags that were used for a pinned allocation.
func MemHostGetFlags(pFlags *uint, p *void) error {
	if err := result(C.cuMemHostGetFlags()); err != nil {
		return errors.Wrapf(err, "MemHostGetFlags")
	}
	return nil
}

// MemHostRegister registers an existing host memory range for use by CUDA.
func MemHostRegister(p *void, bytesize size_t, uint Flags) error {
	if err := result(C.cuMemHostRegister()); err != nil {
		return errors.Wrapf(err, "MemHostRegister")
	}
	return nil
}

// MemHostUnregister unregisters a memory range that was registered with cuMemHostRegister.
func MemHostUnregister(void *p) error {
	if err := result(C.cuMemHostUnregister()); err != nil {
		return errors.Wrapf(err, "MemHostUnregister")
	}
	return nil
}






















// MemcpyAtoH copies memory from Array to Host.
func MemcpyAtoH(dstHost *void, CUarray srcArray, srcOffset size_t, ByteCount size_t) error {
	if err := result(C.cuMemcpyAtoH()); err != nil {
		return errors.Wrapf(err, "MemcpyAtoH")
	}
	return nil
}

// MemcpyAtoHAsync copies memory from Array to Host.
func MemcpyAtoHAsync(dstHost *void, CUarray srcArray, srcOffset size_t, ByteCount size_t, hStream CUstream) error {
	if err := result(C.cuMemcpyAtoHAsync()); err != nil {
		return errors.Wrapf(err, "MemcpyAtoHAsync")
	}
	return nil
}

// MemcpyDtoA copies memory from Device to Array.
func MemcpyDtoA(CUarray dstArray, dstOffset size_t, srcDevice CUdeviceptr, ByteCount size_t) error {
	if err := result(C.cuMemcpyDtoA()); err != nil {
		return errors.Wrapf(err, "MemcpyDtoA")
	}
	return nil
}


// MemcpyHtoA copies memory from Host to Array.
func MemcpyHtoA(srcHost *CUarray, dstArray, dstOffset size_t, void, ByteCount size_t) error {
	if err := result(C.cuMemcpyHtoA()); err != nil {
		return errors.Wrapf(err, "MemcpyHtoA")
	}
	return nil
}

// MemcpyHtoAAsync copies memory from Host to Array.
func MemcpyHtoAAsync(srcHost *CUarray, dstArray, dstOffset size_t, void, ByteCount size_t, hStream CUstream) error {
	if err := result(C.cuMemcpyHtoAAsync()); err != nil {
		return errors.Wrapf(err, "MemcpyHtoAAsync")
	}
	return nil
}










// MemsetD16 initializes device memory.
func MemsetD16(dstDevice CUdeviceptr, us uint16, N size_t) error {
	if err := result(C.cuMemsetD16()); err != nil {
		return errors.Wrapf(err, "MemsetD16")
	}
	return nil
}

// MemsetD16Async sets device memory.
func MemsetD16Async(dstDevice CUdeviceptr, us uint16, N size_t, hStream CUstream) error {
	if err := result(C.cuMemsetD16Async()); err != nil {
		return errors.Wrapf(err, "MemsetD16Async")
	}
	return nil
}

// MemsetD2D16 initializes device memory.
func MemsetD2D16(dstDevice CUdeviceptr, dstPitch size_t, us uint16, Width size_t, Height size_t) error {
	if err := result(C.cuMemsetD2D16()); err != nil {
		return errors.Wrapf(err, "MemsetD2D16")
	}
	return nil
}

// MemsetD2D16Async sets device memory.
func MemsetD2D16Async(dstDevice CUdeviceptr, dstPitch size_t, us uint16, Width size_t, Height size_t, CUstream hStream) error {
	if err := result(C.cuMemsetD2D16Async()); err != nil {
		return errors.Wrapf(err, "MemsetD2D16Async")
	}
	return nil
}

// MemsetD2D32 initializes device memory.
func MemsetD2D32(dstDevice CUdeviceptr, dstPitch size_t, uint ui, Width size_t, Height size_t) error {
	if err := result(C.cuMemsetD2D32()); err != nil {
		return errors.Wrapf(err, "MemsetD2D32")
	}
	return nil
}

// MemsetD2D32Async sets device memory.
func MemsetD2D32Async(dstDevice CUdeviceptr, dstPitch size_t, uint ui, Width size_t, Height size_t, hStream CUstream) error {
	if err := result(C.cuMemsetD2D32Async()); err != nil {
		return errors.Wrapf(err, "MemsetD2D32Async")
	}
	return nil
}

// MemsetD2D8 initializes device memory.
func MemsetD2D8(dstDevice CUdeviceptr, dstPitch size_t, uc uint8, Width size_t, Height size_t) error {
	if err := result(C.cuMemsetD2D8()); err != nil {
		return errors.Wrapf(err, "MemsetD2D8")
	}
	return nil
}

// MemsetD2D8Async sets device memory.
func MemsetD2D8Async(dstDevice CUdeviceptr, dstPitch size_t, uc uint8, Width size_t, Height size_t, hStream CUstream) error {
	if err := result(C.cuMemsetD2D8Async()); err != nil {
		return errors.Wrapf(err, "MemsetD2D8Async")
	}
	return nil
}









// MipmappedArrayCreate creates a CUDA mipmapped array.
func MipmappedArrayCreate(pHandle *CUmipmappedArray, pMipmappedArrayDesc *CUDA_ARRAY3D_DESCRIPTOR, uint numMipmapLevels) error {
	if err := result(C.cuMipmappedArrayCreate()); err != nil {
		return errors.Wrapf(err, "MipmappedArrayCreate")
	}
	return nil
}

// MipmappedArrayDestroy destroys a CUDA mipmapped array.
func MipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) error {
	if err := result(C.cuMipmappedArrayDestroy()); err != nil {
		return errors.Wrapf(err, "MipmappedArrayDestroy")
	}
	return nil
}

// MipmappedArrayGetLevel gets a mipmap level of a CUDA mipmapped array.
func MipmappedArrayGetLevel(pLevelArray *CUarray, CUmipmappedArray hMipmappedArray, uint level) error {
	if err := result(C.cuMipmappedArrayGetLevel()); err != nil {
		return errors.Wrapf(err, "MipmappedArrayGetLevel")
	}
	return nil
}
*/
