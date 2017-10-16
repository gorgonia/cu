package cu

// #include <cuda.h>
import "C"
import "unsafe"

// Array is the pointer to a CUDA array. The name is a bit of a misnomer,
// as it would lead one to imply that it's rangeable. It's not.
type Array uintptr

func goArray(arr *C.CUarray) Array {
	return Array(uintptr(unsafe.Pointer(arr)))
}

func (arr Array) c() C.CUarray {
	return *(*C.CUarray)(unsafe.Pointer(uintptr(arr)))
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
	a := &array3Desc{
		Width:       C.ulong(desc.Width),
		Height:      C.ulong(desc.Height),
		Depth:       C.ulong(desc.Depth),
		Format:      C.CUarray_format(desc.Format),
		NumChannels: C.uint(desc.NumChannels),
		Flags:       C.uint(desc.Flags),
	}
	return a.c()
}

// array3Desc is the unexported, C-comnpatible version of the Array3Desc
//
// Note: the CUDA API notes that the "Width", "Height", "Depth" fields are unsigned int, but a bit of experimentation shows that they're actually 32bits
// so ulong is used instead. This probably works for all 64bit platforms, but if there is a need to extend this to other platforms,
// this data type should be put into its own file with a build tag on top
type array3Desc struct {
	Width, Height, Depth C.ulong
	Format               C.CUarray_format
	NumChannels          C.uint
	Flags                C.uint
}

// cstruct casts the descriptor to a C struct for CUDA consumption.
func (desc *array3Desc) c() *C.CUDA_ARRAY3D_DESCRIPTOR {
	return (*C.CUDA_ARRAY3D_DESCRIPTOR)(unsafe.Pointer(desc))
}

func goArray3Desc(desc *C.CUDA_ARRAY3D_DESCRIPTOR) Array3Desc {
	a3d := (*array3Desc)(unsafe.Pointer(desc))
	return Array3Desc{
		Width:       uint(a3d.Width),
		Height:      uint(a3d.Height),
		Depth:       uint(a3d.Depth),
		Format:      Format(a3d.Format),
		NumChannels: uint(a3d.NumChannels),
		Flags:       uint(a3d.Flags),
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
	a := &arrayDesc{
		Width:       C.ulong(desc.Width),
		Height:      C.ulong(desc.Height),
		Format:      C.CUarray_format(desc.Format),
		NumChannels: C.uint(desc.NumChannels),
	}
	return a.c()
}

// arrayDesc is the unexported, C-compatible version of the Array3Desc
//
// Note: the CUDA API notes that the "Width", "Height" fields are unsigned int, but a bit of experimentation shows that they're actually 32bits
// so ulong is used instead. This probably works for all 64bit platforms, but if there is a need to extend this to other platforms,
// this data type should be put into its own file with a build tag on top
type arrayDesc struct {
	Width       C.ulong
	Height      C.ulong
	Format      C.CUarray_format
	NumChannels C.uint
}

func (desc *arrayDesc) c() *C.CUDA_ARRAY_DESCRIPTOR {
	return (*C.CUDA_ARRAY_DESCRIPTOR)(unsafe.Pointer(desc))
}

func goArrayDesc(desc *C.CUDA_ARRAY_DESCRIPTOR) ArrayDesc {
	ad := (*arrayDesc)(unsafe.Pointer(desc))
	return ArrayDesc{
		Width:       uint(ad.Width),
		Height:      uint(ad.Height),
		Format:      Format(ad.Format),
		NumChannels: uint(ad.NumChannels),
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
	cpyInstr := &cudamemcpy2d{
		SrcXInBytes:   C.size_t(cpy.SrcXInBytes),
		SrcY:          C.size_t(cpy.SrcY),
		SrcMemoryType: C.CUmemorytype(cpy.SrcMemoryType),
		SrcHost:       cpy.SrcHost,
		SrcDevice:     C.CUdeviceptr(cpy.SrcDevice),
		SrcArray:      cpy.SrcArray.c(),
		SrcPitch:      C.size_t(cpy.SrcPitch),
		DstXInBytes:   C.size_t(cpy.DstXInBytes),
		DstY:          C.size_t(cpy.DstY),
		DstMemoryType: C.CUmemorytype(cpy.DstMemoryType),
		DstHost:       cpy.DstHost,
		DstDevice:     C.CUdeviceptr(cpy.DstDevice),
		DstArray:      cpy.DstArray.c(),
		DstPitch:      C.size_t(cpy.DstPitch),
		WidthInBytes:  C.size_t(cpy.WidthInBytes),
		Height:        C.size_t(cpy.Height),
	}
	return cpyInstr.c()
}

// cudamemcpy2d is a struct that is laid out exactly like the CUDA_MEMCPY2D struct, which allows for easy conversion.
type cudamemcpy2d struct {
	SrcXInBytes   C.size_t
	SrcY          C.size_t
	SrcMemoryType C.CUmemorytype
	SrcHost       unsafe.Pointer
	SrcDevice     C.CUdeviceptr
	SrcArray      C.CUarray
	SrcPitch      C.size_t
	DstXInBytes   C.size_t
	DstY          C.size_t
	DstMemoryType C.CUmemorytype
	DstHost       unsafe.Pointer
	DstDevice     C.CUdeviceptr
	DstArray      C.CUarray
	DstPitch      C.size_t
	WidthInBytes  C.size_t
	Height        C.size_t
}

func (cpy *cudamemcpy2d) c() *C.CUDA_MEMCPY2D {
	return (*C.CUDA_MEMCPY2D)(unsafe.Pointer(cpy))
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
	instr := &cudamemcpy3d{
		SrcXInBytes:   C.size_t(cpy.SrcXInBytes),
		SrcY:          C.size_t(cpy.SrcY),
		SrcZ:          C.size_t(cpy.SrcZ),
		SrcLOD:        C.size_t(cpy.SrcLOD),
		SrcMemoryType: C.CUmemorytype(cpy.SrcMemoryType),
		SrcHost:       cpy.SrcHost,
		SrcDevice:     C.CUdeviceptr(cpy.SrcDevice),
		SrcArray:      cpy.SrcArray.c(),
		Reserved0:     nil,
		SrcPitch:      C.size_t(cpy.SrcPitch),
		SrcHeight:     C.size_t(cpy.SrcHeight),
		DstXInBytes:   C.size_t(cpy.DstXInBytes),
		DstY:          C.size_t(cpy.DstY),
		DstZ:          C.size_t(cpy.DstZ),
		DstLOD:        C.size_t(cpy.DstLOD),
		DstMemoryType: C.CUmemorytype(cpy.DstMemoryType),
		DstHost:       cpy.DstHost,
		DstDevice:     C.CUdeviceptr(cpy.DstDevice),
		DstArray:      cpy.DstArray.c(),
		Reserved1:     nil,
		DstPitch:      C.size_t(cpy.DstPitch),
		DstHeight:     C.size_t(cpy.DstHeight),
		WidthInBytes:  C.size_t(cpy.WidthInBytes),
		Height:        C.size_t(cpy.Height),
		Depth:         C.size_t(cpy.Depth),
	}
	return instr.c()
}

type cudamemcpy3d struct {
	SrcXInBytes   C.size_t
	SrcY          C.size_t
	SrcZ          C.size_t
	SrcLOD        C.size_t
	SrcMemoryType C.CUmemorytype
	SrcHost       unsafe.Pointer
	SrcDevice     C.CUdeviceptr
	SrcArray      C.CUarray
	Reserved0     unsafe.Pointer // must be nil
	SrcPitch      C.size_t
	SrcHeight     C.size_t
	DstXInBytes   C.size_t
	DstY          C.size_t
	DstZ          C.size_t
	DstLOD        C.size_t
	DstMemoryType C.CUmemorytype
	DstHost       unsafe.Pointer
	DstDevice     C.CUdeviceptr
	DstArray      C.CUarray
	Reserved1     unsafe.Pointer // must be nil
	DstPitch      C.size_t
	DstHeight     C.size_t
	WidthInBytes  C.size_t
	Height        C.size_t
	Depth         C.size_t
}

func (cpy *cudamemcpy3d) c() *C.CUDA_MEMCPY3D {
	return (*C.CUDA_MEMCPY3D)(unsafe.Pointer(cpy))
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
	instr := &cudaMemcpy3dPeer{
		SrcXInBytes:   C.size_t(cpy.SrcXInBytes),
		SrcY:          C.size_t(cpy.SrcY),
		SrcZ:          C.size_t(cpy.SrcZ),
		SrcLOD:        C.size_t(cpy.SrcLOD),
		SrcMemoryType: C.CUmemorytype(cpy.SrcMemoryType),
		SrcHost:       cpy.SrcHost,
		SrcDevice:     C.CUdeviceptr(cpy.SrcDevice),
		SrcArray:      cpy.SrcArray.c(),
		SrcContext:    cpy.SrcContext.c(),
		SrcPitch:      C.size_t(cpy.SrcPitch),
		SrcHeight:     C.size_t(cpy.SrcHeight),
		DstXInBytes:   C.size_t(cpy.DstXInBytes),
		DstY:          C.size_t(cpy.DstY),
		DstZ:          C.size_t(cpy.DstZ),
		DstLOD:        C.size_t(cpy.DstLOD),
		DstMemoryType: C.CUmemorytype(cpy.DstMemoryType),
		DstHost:       cpy.DstHost,
		DstDevice:     C.CUdeviceptr(cpy.DstDevice),
		DstArray:      cpy.DstArray.c(),
		DstContext:    cpy.DstContext.c(),
		DstPitch:      C.size_t(cpy.DstPitch),
		DstHeight:     C.size_t(cpy.DstHeight),
		WidthInBytes:  C.size_t(cpy.WidthInBytes),
		Height:        C.size_t(cpy.Height),
		Depth:         C.size_t(cpy.Depth),
	}
	return instr.c()
}

type cudaMemcpy3dPeer struct {
	SrcXInBytes   C.size_t
	SrcY          C.size_t
	SrcZ          C.size_t
	SrcLOD        C.size_t
	SrcMemoryType C.CUmemorytype
	SrcHost       unsafe.Pointer
	SrcDevice     C.CUdeviceptr
	SrcArray      C.CUarray
	SrcContext    C.CUcontext
	SrcPitch      C.size_t
	SrcHeight     C.size_t
	DstXInBytes   C.size_t
	DstY          C.size_t
	DstZ          C.size_t
	DstLOD        C.size_t
	DstMemoryType C.CUmemorytype
	DstHost       unsafe.Pointer
	DstDevice     C.CUdeviceptr
	DstArray      C.CUarray
	DstContext    C.CUcontext
	DstPitch      C.size_t
	DstHeight     C.size_t
	WidthInBytes  C.size_t
	Height        C.size_t
	Depth         C.size_t
}

func (cpy *cudaMemcpy3dPeer) c() *C.CUDA_MEMCPY3D_PEER {
	return (*C.CUDA_MEMCPY3D_PEER)(unsafe.Pointer(cpy))
}

func MakeArray(pAllocateArray ArrayDesc) (pHandle Array, err error) {
	var CpHandle C.CUarray
	CpAllocateArray := pAllocateArray.c()
	err = result(C.cuArrayCreate(&CpHandle, CpAllocateArray))
	pHandle = goArray(&CpHandle)
	return
}

func Make3DArray(pAllocateArray Array3Desc) (pHandle Array, err error) {
	var CpHandle C.CUarray
	CpAllocateArray := pAllocateArray.c()
	err = result(C.cuArray3DCreate(&CpHandle, CpAllocateArray))
	pHandle = goArray(&CpHandle)
	return
}
