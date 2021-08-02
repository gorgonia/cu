package cu

/*
#include <cuda.h>

void CallHostFunc(void* fn){
	handleCUDACB(fn);
};
*/
import "C"
import "unsafe"

// KernelNodeParams represents the parameters to launch a kernel in a graph node.
type KernelNodeParams struct {
	Func           Function
	GridDimX       uint
	GridDimY       uint
	GridDimZ       uint
	BlockDimX      uint
	BlockDimY      uint
	BlockDimZ      uint
	SharedMemBytes uint

	Params []*KernelNodeParams
}

func (p *KernelNodeParams) c() *C.CUDA_KERNEL_NODE_PARAMS {
	// here anonymous initialization of struct fields is used because `func` is a keyword.
	// see also: https://github.com/golang/go/issues/41968
	retVal := &C.CUDA_KERNEL_NODE_PARAMS{
		p.Func.fn,
		C.uint(p.GridDimX),
		C.uint(p.GridDimY),
		C.uint(p.GridDimZ),
		C.uint(p.BlockDimX),
		C.uint(p.BlockDimY),
		C.uint(p.BlockDimZ),
		C.uint(p.SharedMemBytes),
		nil,
		nil,
	}
	return retVal
}

// HostNodeParams are parameters passed in to a node that will call a host function (i.e. a function written in Go)
type HostNodeParams struct {
	Func HostFunction
	Data unsafe.Pointer

	registered bool
	ptr        unsafe.Pointer
}

func (p *HostNodeParams) c() *C.CUDA_HOST_NODE_PARAMS {
	var ptr unsafe.Pointer
	if p.registered {
		ptr = p.ptr
	} else {
		ptr = RegisterFunc(p.Func)
		p.ptr = ptr
		p.registered = true
	}

	return &C.CUDA_HOST_NODE_PARAMS{
		fn:       C.CUhostFn(C.CallHostFunc),
		userData: ptr, // userData is basically the Go function to call.
	}
}

type CopyParams struct {
	SrcXInBytes  uint64
	SrcY         uint64
	SrcZ         uint64
	SrcLOD       uint64
	SrcType      MemoryType
	SrcHost      unsafe.Pointer
	SrcDevicePtr DevicePtr
	SrcArray     Array
	Reserved0    unsafe.Pointer
	SrcPitch     uint64
	SrcHeight    uint64

	DstXInBytes  uint64
	DstY         uint64
	DstZ         uint64
	DstLOD       uint64
	DstType      MemoryType
	DstHost      unsafe.Pointer
	DstDevicePtr DevicePtr
	DstArray     Array
	Reserved1    unsafe.Pointer
	DstPitch     uint64
	DstHeight    uint64

	WidthInBytes uint64
	Height       uint64
	Depth        uint64
}

func (p *CopyParams) c() *C.CUDA_MEMCPY3D {
	return &C.CUDA_MEMCPY3D{
		srcXInBytes:   C.size_t(p.SrcXInBytes),
		srcY:          C.size_t(p.SrcY),
		srcZ:          C.size_t(p.SrcZ),
		srcLOD:        C.size_t(p.SrcLOD),
		srcMemoryType: C.CUmemorytype(p.SrcType),
		srcHost:       p.SrcHost,
		srcDevice:     C.CUdeviceptr(p.SrcDevicePtr),
		srcArray:      p.SrcArray.c(),
		reserved0:     nil,
		srcPitch:      C.size_t(p.SrcPitch),
		srcHeight:     C.size_t(p.SrcHeight),
		dstXInBytes:   C.size_t(p.DstXInBytes),
		dstY:          C.size_t(p.DstY),
		dstZ:          C.size_t(p.DstZ),
		dstLOD:        C.size_t(p.DstLOD),
		dstMemoryType: C.CUmemorytype(p.DstType),
		dstHost:       p.DstHost,
		dstDevice:     C.CUdeviceptr(p.DstDevicePtr),
		dstArray:      p.DstArray.c(),
		reserved1:     nil,
		dstPitch:      C.size_t(p.DstPitch),
		dstHeight:     C.size_t(p.DstHeight),
		WidthInBytes:  C.size_t(p.WidthInBytes),
		Height:        C.size_t(p.Height),
		Depth:         C.size_t(p.Depth),
	}
}

type MemsetParams struct {
	Dst         DevicePtr
	Pitch       uint64
	Value       uint
	ElementSize uint
	Width       uint64
	Height      uint64
}

func (p *MemsetParams) c() *C.CUDA_MEMSET_NODE_PARAMS {
	return &C.CUDA_MEMSET_NODE_PARAMS{
		dst:         C.CUdeviceptr(p.Dst),
		pitch:       C.size_t(p.Pitch),
		value:       C.uint(p.Value),
		elementSize: C.uint(p.ElementSize),
		width:       C.size_t(p.Width),
		height:      C.size_t(p.Height),
	}
}
