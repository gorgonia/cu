package cu

// #include <cuda.h>
import "C"

type KernelNodeParams struct {
	C.CUDA_KERNAL_NODE_PARAMS
}
