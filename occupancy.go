package cu

// #include <cuda.h>
import "C"

// MaxActiveBlocksPerMultiProcessor returns the number of the maximum active blocks per streaming multiprocessor.
func (fn Function) MaxActiveBlocksPerMultiProcessor(blockSize int, dynamicSmemSize int64) (int, error) {
	bs := C.int(blockSize)
	dss := C.size_t(dynamicSmemSize)

	var numBlocks C.int
	if err := result(C.cuOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, fn.fn, bs, dss)); err != nil {
		return 0, err
	}
	return int(numBlocks), nil
}

// MaxActiveBlocksPerMultiProcessorWithFlags returns the number of the maximum active blocks per streaming multiprocessor.
// The flags control how special cases are handled.
func (fn Function) MaxActiveBlocksPerMultiProcessorWithFlags(blockSize int, dynamicSmemSize int64, flags OccupancyFlags) (int, error) {
	bs := C.int(blockSize)
	dss := C.size_t(dynamicSmemSize)
	of := C.uint(flags)

	var numBlocks C.int
	if err := result(C.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&numBlocks, fn.fn, bs, dss, of)); err != nil {
		return 0, err
	}
	return int(numBlocks), nil
}

// MaxPotentialBlockSize suggest a reasonable block size that can achieve the maximum occupancy (or, the maximum number of active warps with the fewest blocks per multiprocessor
// and the minimum grid size to achieve the maximum occupancy.
//
// If blockSizeLimit is 0, the configurator will use the maximum block size permitted by the device / function instead.
//
// If per-block dynamic shared memory allocation is not needed, the user should leave both bs2ds and dynamicSMemSize as 0.
//
// If per-block dynamic shared memory allocation is needed, then if the dynamic shared memory size is constant regardless of block size, the size should be passed through dynamicSMemSize, and blockSizeToDynamicSMemSize should be NULL.
//
// Otherwise, if the per-block dynamic shared memory size varies with different block sizes, the user needs to provide a unary function through blockSizeToDynamicSMemSize that computes the dynamic shared memory needed by func for any given block size.
// dynamicSMemSize is ignored. An example signature is:
//		// Take block size, returns dynamic shared memory needed
//          size_t blockToSmem(int blockSize);
// func (fn Function) MaxPotentialBlockSize(blockSizeToDynamicSMemSize int, dynamicSmemSize int64, blockSizeLimit int) (minGridSize, blockSize int, err error) {
// 	bs2dsm := C.CUoccupancyB2DSize(blockSizeToDynamicSMemSize)
// 	dss := C.size_t(dynamicSmemSize)
// 	f := C.CUfunction(fn)
// 	bsl := C.int(blockSizeLimit)

// 	var mgs, bs C.int
// 	if err = result(C.cuOccupancyMaxPotentialBlockSize(&mgs, &bs, f, bs2dsm, dss, bsl)); err != nil {
// 		return
// 	}
// 	minGridSize = int(mgs)
// 	blockSize = int(bs)
// 	return
// }

// MaxPotentialBlockSizeWithFlags suggest a reasonable block size that can achieve the maximum occupancy (or, the maximum number of active warps with the fewest blocks per multiprocessor
// and the minimum grid size to achieve the maximum occupancy.

// If blockSizeLimit is 0, the configurator will use the maximum block size permitted by the device / function instead.

// If per-block dynamic shared memory allocation is not needed, the user should leave both bs2ds and dynamicSMemSize as 0.

// If per-block dynamic shared memory allocation is needed, then if the dynamic shared memory size is constant regardless of block size, the size should be passed through dynamicSMemSize, and blockSizeToDynamicSMemSize should be NULL.

// Otherwise, if the per-block dynamic shared memory size varies with different block sizes, the user needs to provide a unary function through blockSizeToDynamicSMemSize that computes the dynamic shared memory needed by func for any given block size.
// dynamicSMemSize is ignored. An example signature is:
// 		// Take block size, returns dynamic shared memory needed
//          size_t blockToSmem(int blockSize);
// func (fn Function) MaxPotentialBlockSizeWithFlags(blockSizeToDynamicSMemSize int, dynamicSmemSize int64, blockSizeLimit int, flags OccupancyFlags) (minGridSize, blockSize int, err error) {
// 	bs2dsm := C.CUoccupancyB2DSize(blockSizeToDynamicSMemSize)
// 	dss := C.size_t(dynamicSmemSize)
// 	f := C.CUfunction(fn)
// 	bsl := C.int(blockSizeLimit)
// 	of := CUoccupancy_flags(flags)

// 	var mgs, bs C.int
// 	if err = result(C.cuOccupancyMaxPotentialBlockSize(&mgs, &bs, f, bs2dsm, dss, bsl, of)); err != nil {
// 		return
// 	}
// 	minGridSize = int(mgs)
// 	blockSize = int(bs)
// 	return
// }
