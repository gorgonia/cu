// Package cu provides an idiomatic interface to the CUDA Driver API.
package cu

// This file implements CUDA driver context management

//#include <cuda.h>
import "C"

func init() {
	// Given that the flags must be 0, the CUDA driver is initialized at the package level
	// http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html
	if err := result(C.cuInit(C.uint(0))); err != nil {
		panic(err)
	}

}

// Version returns the version of the CUDA driver
func Version() int {
	var v C.int
	if err := result(C.cuDriverGetVersion(&v)); err != nil {
		return -1
	}
	return int(v)
}
