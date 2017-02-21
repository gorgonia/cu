package cu

// #include <cuda.h>
import "C"
import "unsafe"

// Format is the type of array (think array types)
type Format byte

const (
	Uint8   Format = C.CU_AD_FORMAT_UNSIGNED_INT8  // Unsigned 8-bit integers
	Uint16  Format = C.CU_AD_FORMAT_UNSIGNED_INT16 // Unsigned 16-bit integers
	Uin32   Format = C.CU_AD_FORMAT_UNSIGNED_INT32 // Unsigned 32-bit integers
	Int8    Format = C.CU_AD_FORMAT_SIGNED_INT8    // Signed 8-bit integers
	Int16   Format = C.CU_AD_FORMAT_SIGNED_INT16   // Signed 16-bit integers
	Int32   Format = C.CU_AD_FORMAT_SIGNED_INT32   // Signed 32-bit integers
	Float16 Format = C.CU_AD_FORMAT_HALF           // 16-bit floating point
	Float32 Format = C.CU_AD_FORMAT_FLOAT          // 32-bit floating point
)

// Array is the pointer to a CUDA array. The name is a bit of a misnomer, as it would lead one to imply that it's rangeable. It's not.
type Array uintptr

func (arr Array) cuda() C.CUarray {
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

func (desc Array3Desc) cstruct() *C.CUDA_ARRAY3D_DESCRIPTOR {
	a := &array3Desc{
		Width:       C.ulong(desc.Width),
		Height:      C.ulong(desc.Height),
		Depth:       C.ulong(desc.Depth),
		Format:      C.CUarray_format(desc.Format),
		NumChannels: C.uint(desc.NumChannels),
		Flags:       C.uint(desc.Flags),
	}
	return a.cstruct()
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
func (desc *array3Desc) cstruct() *C.CUDA_ARRAY3D_DESCRIPTOR {
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

func (desc ArrayDesc) cstruct() *C.CUDA_ARRAY_DESCRIPTOR {
	a := &arrayDesc{
		Width:       C.ulong(desc.Width),
		Height:      C.ulong(desc.Height),
		Format:      C.CUarray_format(desc.Format),
		NumChannels: C.uint(desc.NumChannels),
	}
	return a.cstruct()
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

func (desc *arrayDesc) cstruct() *C.CUDA_ARRAY_DESCRIPTOR {
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
