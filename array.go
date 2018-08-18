package cu

// #include <cuda.h>
import "C"
import "unsafe"

// Array is the pointer to a CUDA array. The name is a bit of a misnomer,
// as it would lead one to imply that it's rangeable. It's not.
type Array struct {
	arr *C.CUarray
}

func goArray(arr *C.CUarray) Array {
	return Array{arr}
}

func (arr Array) c() C.CUarray {
	return *arr.arr
}

// Array3Desc is the descriptor for CUDA 3D arrays, which is used to determine what to allocate.
//
// From the docs:
//	 Width, Height, and Depth are the width, height, and depth of the CUDA array (in elements); the following types of CUDA arrays can be allocated:
//		- A 1D array is allocated if Height and Depth extents are both zero.
//		- A 2D array is allocated if only Depth extent is zero.
//		- A 3D array is allocated if all three extents are non-zero.
//		- A 1D layered CUDA array is allocated if only Height is zero and the CUDA_ARRAY3D_LAYERED flag is set. Each layer is a 1D array. The number of layers is determined by the depth extent.
//		- A 2D layered CUDA array is allocated if all three extents are non-zero and the CUDA_ARRAY3D_LAYERED flag is set. Each layer is a 2D array. The number of layers is determined by the depth extent.
//		- A cubemap CUDA array is allocated if all three extents are non-zero and the CUDA_ARRAY3D_CUBEMAP flag is set. Width must be equal to Height, and Depth must be six. A cubemap is a special type of 2D layered CUDA array, where the six layers represent the six faces of a cube. The order of the six layers in memory is the same as that listed in CUarray_cubemap_face.
//		- A cubemap layered CUDA array is allocated if all three extents are non-zero, and both, CUDA_ARRAY3D_CUBEMAP and CUDA_ARRAY3D_LAYERED flags are set. Width must be equal to Height, and Depth must be a multiple of six. A cubemap layered CUDA array is a special type of 2D layered CUDA array that consists of a collection of cubemaps. The first six layers represent the first cubemap, the next six layers form the second cubemap, and so on.
type Array3Desc struct {
	Width, Height, Depth uint
	Format               Format
	NumChannels          uint
	Flags                uint
}

func (desc Array3Desc) c() *C.CUDA_ARRAY3D_DESCRIPTOR {
	return &C.CUDA_ARRAY3D_DESCRIPTOR{
		Width:       C.size_t(desc.Width),
		Height:      C.size_t(desc.Height),
		Depth:       C.size_t(desc.Depth),
		Format:      C.CUarray_format(desc.Format),
		NumChannels: C.uint(desc.NumChannels),
		Flags:       C.uint(desc.Flags),
	}
}

func goArray3Desc(desc *C.CUDA_ARRAY3D_DESCRIPTOR) Array3Desc {
	return Array3Desc{
		Width:       uint(desc.Width),
		Height:      uint(desc.Height),
		Depth:       uint(desc.Depth),
		Format:      Format(desc.Format),
		NumChannels: uint(desc.NumChannels),
		Flags:       uint(desc.Flags),
	}
}

// ArrayDesc is the descriptor for CUDA arrays, which is used to determine what to allocate.
//
// From the docs:
// 	Width, and Height are the width, and height of the CUDA array (in elements); the CUDA array is one-dimensional if height is 0, two-dimensional otherwise;
type ArrayDesc struct {
	Width, Height uint
	Format        Format
	NumChannels   uint
}

func (desc ArrayDesc) c() *C.CUDA_ARRAY_DESCRIPTOR {
	return &C.CUDA_ARRAY_DESCRIPTOR{
		Width:       C.size_t(desc.Width),
		Height:      C.size_t(desc.Height),
		Format:      C.CUarray_format(desc.Format),
		NumChannels: C.uint(desc.NumChannels),
	}
}

func goArrayDesc(desc *C.CUDA_ARRAY_DESCRIPTOR) ArrayDesc {
	return ArrayDesc{
		Width:       uint(desc.Width),
		Height:      uint(desc.Height),
		Format:      Format(desc.Format),
		NumChannels: uint(desc.NumChannels),
	}
}

// // Descriptor3 get a 3D CUDA array descriptor
// func (arr Array) Descriptor3() (Array3Desc, error) {
// 	hArray := arr.c()
// 	var desc C.CUDA_ARRAY3D_DESCRIPTOR
// 	if err := result(C.cuArray3DGetDescriptor(&desc, hArray)); err != nil {
// 		return Array3Desc{}, errors.Wrapf(err, "Array3DGetDescriptor")
// 	}
// 	return goArray3Desc(&desc), nil
// }

// // Descriptor gets a 1D or 2D CUDA array descriptor
// func (arr Array) Descriptor() (ArrayDesc, error) {
// 	hArray := arr.c()
// 	var desc C.CUDA_ARRAY_DESCRIPTOR
// 	if err := result(C.cuArrayGetDescriptor(&desc, hArray)); err != nil {
// 		return ArrayDesc{}, errors.Wrapf(err, "ArrayGetDescriptor")
// 	}
// 	return goArrayDesc(&desc), nil

// }

// Memcpy2dParam is a struct representing the params of a 2D memory copy instruction.
// To aid usability, the fields are ordered as per the documentation (the actual struct is laid out differently).
type Memcpy2dParam struct {
	Height        int64
	WidthInBytes  int64
	DstArray      Array
	DstDevice     DevicePtr
	DstHost       unsafe.Pointer
	DstMemoryType MemoryType
	DstPitch      int64
	DstXInBytes   int64
	DstY          int64
	SrcArray      Array
	SrcDevice     DevicePtr
	SrcHost       unsafe.Pointer
	SrcMemoryType MemoryType
	SrcPitch      int64
	SrcXInBytes   int64
	SrcY          int64
}

func (cpy Memcpy2dParam) c() *C.CUDA_MEMCPY2D {
	return &C.CUDA_MEMCPY2D{
		srcXInBytes:   C.size_t(cpy.SrcXInBytes),
		srcY:          C.size_t(cpy.SrcY),
		srcMemoryType: C.CUmemorytype(cpy.SrcMemoryType),
		srcHost:       cpy.SrcHost,
		srcDevice:     C.CUdeviceptr(cpy.SrcDevice),
		srcArray:      cpy.SrcArray.c(),
		srcPitch:      C.size_t(cpy.SrcPitch),
		dstXInBytes:   C.size_t(cpy.DstXInBytes),
		dstY:          C.size_t(cpy.DstY),
		dstMemoryType: C.CUmemorytype(cpy.DstMemoryType),
		dstHost:       cpy.DstHost,
		dstDevice:     C.CUdeviceptr(cpy.DstDevice),
		dstArray:      cpy.DstArray.c(),
		dstPitch:      C.size_t(cpy.DstPitch),
		WidthInBytes:  C.size_t(cpy.WidthInBytes),
		Height:        C.size_t(cpy.Height),
	}
}

// Memcpy3dParam is a struct representing the params of a 3D memory copy instruction.
// To aid usability, the fields are ordered as per the documentation (the actual struct is laid out differently).
type Memcpy3dParam struct {
	Depth         int64
	Height        int64
	WidthInBytes  int64
	DstArray      Array
	DstDevice     DevicePtr
	DstHeight     int64
	DstHost       unsafe.Pointer
	DstLOD        int64
	DstMemoryType MemoryType
	DstPitch      int64
	DstXInBytes   int64
	DstY          int64
	DstZ          int64
	SrcArray      Array
	SrcDevice     DevicePtr
	SrcHeight     int64
	SrcHost       unsafe.Pointer
	SrcLOD        int64
	SrcMemoryType MemoryType
	SrcPitch      int64
	SrcXInBytes   int64
	SrcY          int64
	SrcZ          int64
}

func (cpy Memcpy3dParam) c() *C.CUDA_MEMCPY3D {
	return &C.CUDA_MEMCPY3D{
		srcXInBytes:   C.size_t(cpy.SrcXInBytes),
		srcY:          C.size_t(cpy.SrcY),
		srcZ:          C.size_t(cpy.SrcZ),
		srcLOD:        C.size_t(cpy.SrcLOD),
		srcMemoryType: C.CUmemorytype(cpy.SrcMemoryType),
		srcHost:       cpy.SrcHost,
		srcDevice:     C.CUdeviceptr(cpy.SrcDevice),
		srcArray:      cpy.SrcArray.c(),
		reserved0:     nil,
		srcPitch:      C.size_t(cpy.SrcPitch),
		srcHeight:     C.size_t(cpy.SrcHeight),
		dstXInBytes:   C.size_t(cpy.DstXInBytes),
		dstY:          C.size_t(cpy.DstY),
		dstZ:          C.size_t(cpy.DstZ),
		dstLOD:        C.size_t(cpy.DstLOD),
		dstMemoryType: C.CUmemorytype(cpy.DstMemoryType),
		dstHost:       cpy.DstHost,
		dstDevice:     C.CUdeviceptr(cpy.DstDevice),
		dstArray:      cpy.DstArray.c(),
		reserved1:     nil,
		dstPitch:      C.size_t(cpy.DstPitch),
		dstHeight:     C.size_t(cpy.DstHeight),
		WidthInBytes:  C.size_t(cpy.WidthInBytes),
		Height:        C.size_t(cpy.Height),
		Depth:         C.size_t(cpy.Depth),
	}
}

// Memcpy3dParam is a struct representing the params of a 3D memory copy instruction across contexts.
// To aid usability, the fields are ordered as per the documentation (the actual struct is laid out differently).
type Memcpy3dPeerParam struct {
	Depth         int64
	Height        int64
	WidthInBytes  int64
	DstArray      Array
	DstContext    CUContext
	DstDevice     DevicePtr
	DstHeight     int64
	DstHost       unsafe.Pointer
	DstLOD        int64
	DstMemoryType MemoryType
	DstPitch      int64
	DstXInBytes   int64
	DstY          int64
	DstZ          int64
	SrcArray      Array
	SrcContext    CUContext
	SrcDevice     DevicePtr
	SrcHeight     int64
	SrcHost       unsafe.Pointer
	SrcLOD        int64
	SrcMemoryType MemoryType
	SrcPitch      int64
	SrcXInBytes   int64
	SrcY          int64
	SrcZ          int64
}

func (cpy *Memcpy3dPeerParam) c() *C.CUDA_MEMCPY3D_PEER {
	return &C.CUDA_MEMCPY3D_PEER{
		srcXInBytes:   C.size_t(cpy.SrcXInBytes),
		srcY:          C.size_t(cpy.SrcY),
		srcZ:          C.size_t(cpy.SrcZ),
		srcLOD:        C.size_t(cpy.SrcLOD),
		srcMemoryType: C.CUmemorytype(cpy.SrcMemoryType),
		srcHost:       cpy.SrcHost,
		srcDevice:     C.CUdeviceptr(cpy.SrcDevice),
		srcArray:      cpy.SrcArray.c(),
		srcContext:    cpy.SrcContext.c(),
		srcPitch:      C.size_t(cpy.SrcPitch),
		srcHeight:     C.size_t(cpy.SrcHeight),
		dstXInBytes:   C.size_t(cpy.DstXInBytes),
		dstY:          C.size_t(cpy.DstY),
		dstZ:          C.size_t(cpy.DstZ),
		dstLOD:        C.size_t(cpy.DstLOD),
		dstMemoryType: C.CUmemorytype(cpy.DstMemoryType),
		dstHost:       cpy.DstHost,
		dstDevice:     C.CUdeviceptr(cpy.DstDevice),
		dstArray:      cpy.DstArray.c(),
		dstContext:    cpy.DstContext.c(),
		dstPitch:      C.size_t(cpy.DstPitch),
		dstHeight:     C.size_t(cpy.DstHeight),
		WidthInBytes:  C.size_t(cpy.WidthInBytes),
		Height:        C.size_t(cpy.Height),
		Depth:         C.size_t(cpy.Depth),
	}
}

func MakeArray(pAllocateArray ArrayDesc) (pHandle Array, err error) {
	var CpHandle C.CUarray
	CpAllocateArray := pAllocateArray.c()
	err = result(C.cuArrayCreate(&CpHandle, CpAllocateArray))
	pHandle = Array{&CpHandle}
	return
}

func Make3DArray(pAllocateArray Array3Desc) (pHandle Array, err error) {
	var CpHandle C.CUarray
	CpAllocateArray := pAllocateArray.c()
	err = result(C.cuArray3DCreate(&CpHandle, CpAllocateArray))
	pHandle = Array{&CpHandle}
	return
}
