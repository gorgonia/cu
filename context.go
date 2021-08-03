package cu

// #include <cuda.h>
import "C"
import (
	"context"
	"unsafe"
)

var (
	_ Context = &Ctx{}
	_ Context = &BatchedContext{}
)

// Context interface. Typically you'd just embed *Ctx. Rarely do you need to use CUContext
type Context interface {
	// Operational stuff
	CUDAContext() CUContext
	Error() error
	Run(chan error) error
	Do(fn func() error) error
	Work() <-chan func() error
	ErrChan() chan error
	Close() error // Close closes all resources associated with the context

	// context.Context

	context.Context
	Cancel() // context.CancelFunc

	// actual methods
	Address(hTexRef TexRef) (pdptr DevicePtr, err error)
	AddressMode(hTexRef TexRef, dim int) (pam AddressMode, err error)
	Array(hTexRef TexRef) (phArray Array, err error)
	AttachMemAsync(hStream Stream, dptr DevicePtr, length int64, flags uint)
	BorderColor(hTexRef TexRef) (pBorderColor [3]float32, err error)
	CurrentCacheConfig() (pconfig FuncCacheConfig, err error)
	CurrentDevice() (device Device, err error)
	CurrentFlags() (flags ContextFlags, err error)
	Descriptor(hArray Array) (pArrayDescriptor ArrayDesc, err error)
	Descriptor3(hArray Array) (pArrayDescriptor Array3Desc, err error)
	DestroyArray(hArray Array)
	DestroyEvent(event *Event)
	DestroyStream(hStream *Stream)
	DisablePeerAccess(peerContext CUContext)
	Elapsed(hStart Event, hEnd Event) (pMilliseconds float64, err error)
	EnablePeerAccess(peerContext CUContext, Flags uint)
	FilterMode(hTexRef TexRef) (pfm FilterMode, err error)
	Format(hTexRef TexRef) (pFormat Format, pNumChannels int, err error)
	FunctionAttribute(fn Function, attrib FunctionAttribute) (pi int, err error)
	GetArray(hSurfRef SurfRef) (phArray Array, err error)
	LaunchKernel(fn Function, gridDimX, gridDimY, gridDimZ int, blockDimX, blockDimY, blockDimZ int, sharedMemBytes int, stream Stream, kernelParams []unsafe.Pointer)
	Limits(limit Limit) (pvalue int64, err error)
	Load(name string) (m Module, err error)
	MakeEvent(flags EventFlags) (event Event, err error)
	MakeStream(flags StreamFlags) (stream Stream, err error)
	MakeStreamWithPriority(priority int, flags StreamFlags) (stream Stream, err error)
	MaxAnisotropy(hTexRef TexRef) (pmaxAniso int, err error)
	MemAlloc(bytesize int64) (dptr DevicePtr, err error)
	MemAllocManaged(bytesize int64, flags MemAttachFlags) (dptr DevicePtr, err error)
	MemAllocPitch(WidthInBytes int64, Height int64, ElementSizeBytes uint) (dptr DevicePtr, pPitch int64, err error)
	MemFree(dptr DevicePtr)
	MemFreeHost(p unsafe.Pointer)
	MemInfo() (free int64, total int64, err error)
	Memcpy(dst DevicePtr, src DevicePtr, ByteCount int64)
	Memcpy2D(pCopy Memcpy2dParam)
	Memcpy2DAsync(pCopy Memcpy2dParam, hStream Stream)
	Memcpy2DUnaligned(pCopy Memcpy2dParam)
	Memcpy3D(pCopy Memcpy3dParam)
	Memcpy3DAsync(pCopy Memcpy3dParam, hStream Stream)
	Memcpy3DPeer(pCopy Memcpy3dPeerParam)
	Memcpy3DPeerAsync(pCopy Memcpy3dPeerParam, hStream Stream)
	MemcpyAsync(dst DevicePtr, src DevicePtr, ByteCount int64, hStream Stream)
	MemcpyAtoA(dstArray Array, dstOffset int64, srcArray Array, srcOffset int64, ByteCount int64)
	MemcpyAtoD(dstDevice DevicePtr, srcArray Array, srcOffset int64, ByteCount int64)
	MemcpyAtoH(dstHost unsafe.Pointer, srcArray Array, srcOffset int64, ByteCount int64)
	MemcpyAtoHAsync(dstHost unsafe.Pointer, srcArray Array, srcOffset int64, ByteCount int64, hStream Stream)
	MemcpyDtoA(dstArray Array, dstOffset int64, srcDevice DevicePtr, ByteCount int64)
	MemcpyDtoD(dstDevice DevicePtr, srcDevice DevicePtr, ByteCount int64)
	MemcpyDtoDAsync(dstDevice DevicePtr, srcDevice DevicePtr, ByteCount int64, hStream Stream)
	MemcpyDtoH(dstHost unsafe.Pointer, srcDevice DevicePtr, ByteCount int64)
	MemcpyDtoHAsync(dstHost unsafe.Pointer, srcDevice DevicePtr, ByteCount int64, hStream Stream)
	MemcpyHtoA(dstArray Array, dstOffset int64, srcHost unsafe.Pointer, ByteCount int64)
	MemcpyHtoAAsync(dstArray Array, dstOffset int64, srcHost unsafe.Pointer, ByteCount int64, hStream Stream)
	MemcpyHtoD(dstDevice DevicePtr, srcHost unsafe.Pointer, ByteCount int64)
	MemcpyHtoDAsync(dstDevice DevicePtr, srcHost unsafe.Pointer, ByteCount int64, hStream Stream)
	MemcpyPeer(dstDevice DevicePtr, dstContext CUContext, srcDevice DevicePtr, srcContext CUContext, ByteCount int64)
	MemcpyPeerAsync(dstDevice DevicePtr, dstContext CUContext, srcDevice DevicePtr, srcContext CUContext, ByteCount int64, hStream Stream)
	MemsetD16(dstDevice DevicePtr, us uint16, N int64)
	MemsetD16Async(dstDevice DevicePtr, us uint16, N int64, hStream Stream)
	MemsetD2D16(dstDevice DevicePtr, dstPitch int64, us uint16, Width int64, Height int64)
	MemsetD2D16Async(dstDevice DevicePtr, dstPitch int64, us uint16, Width int64, Height int64, hStream Stream)
	MemsetD2D32(dstDevice DevicePtr, dstPitch int64, ui uint, Width int64, Height int64)
	MemsetD2D32Async(dstDevice DevicePtr, dstPitch int64, ui uint, Width int64, Height int64, hStream Stream)
	MemsetD2D8(dstDevice DevicePtr, dstPitch int64, uc byte, Width int64, Height int64)
	MemsetD2D8Async(dstDevice DevicePtr, dstPitch int64, uc byte, Width int64, Height int64, hStream Stream)
	MemsetD32(dstDevice DevicePtr, ui uint, N int64)
	MemsetD32Async(dstDevice DevicePtr, ui uint, N int64, hStream Stream)
	MemsetD8(dstDevice DevicePtr, uc byte, N int64)
	MemsetD8Async(dstDevice DevicePtr, uc byte, N int64, hStream Stream)
	ModuleFunction(m Module, name string) (function Function, err error)
	ModuleGlobal(m Module, name string) (dptr DevicePtr, size int64, err error)
	Priority(hStream Stream) (priority int, err error)
	QueryEvent(hEvent Event)
	QueryStream(hStream Stream)
	Record(hEvent Event, hStream Stream)
	SetAddress(hTexRef TexRef, dptr DevicePtr, bytes int64) (ByteOffset int64, err error)
	SetAddress2D(hTexRef TexRef, desc ArrayDesc, dptr DevicePtr, Pitch int64)
	SetAddressMode(hTexRef TexRef, dim int, am AddressMode)
	SetBorderColor(hTexRef TexRef, pBorderColor [3]float32)
	SetCacheConfig(fn Function, config FuncCacheConfig)
	SetCurrentCacheConfig(config FuncCacheConfig)
	SetFilterMode(hTexRef TexRef, fm FilterMode)
	SetFormat(hTexRef TexRef, fmt Format, NumPackedComponents int)
	SetFunctionSharedMemConfig(fn Function, config SharedConfig)
	SetLimit(limit Limit, value int64)
	SetMaxAnisotropy(hTexRef TexRef, maxAniso uint)
	SetMipmapFilterMode(hTexRef TexRef, fm FilterMode)
	SetMipmapLevelBias(hTexRef TexRef, bias float64)
	SetMipmapLevelClamp(hTexRef TexRef, minMipmapLevelClamp float64, maxMipmapLevelClamp float64)
	SetSharedMemConfig(config SharedConfig)
	SetTexRefFlags(hTexRef TexRef, Flags TexRefFlags)
	SharedMemConfig() (pConfig SharedConfig, err error)
	StreamFlags(hStream Stream) (flags uint, err error)
	StreamPriorityRange() (leastPriority int, greatestPriority int, err error)
	SurfRefSetArray(hSurfRef SurfRef, hArray Array, Flags uint)
	Synchronize()
	SynchronizeEvent(hEvent Event)
	SynchronizeStream(hStream Stream)
	TexRefFlags(hTexRef TexRef) (pFlags uint, err error)
	TexRefSetArray(hTexRef TexRef, hArray Array, Flags uint)
	Unload(hmod Module)
	Wait(hStream Stream, hEvent Event, Flags uint)
	WaitOnValue32(stream Stream, addr DevicePtr, value uint32, flags uint)
	WriteValue32(stream Stream, addr DevicePtr, value uint32, flags uint)
}
