/*
 * Module to test CUDA module loading and execution.
 * To be compiled with:
 * nvcc -ptx module_test.cu
 */

#ifdef __cplusplus
extern "C" {
#endif

/// Sets the first N elements of array to value.
__global__ void testMemset(float* array, float value, int N){
	int i = ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if(i < N){
		array[i] = value;
	}
}
#ifdef __cplusplus
}
#endif