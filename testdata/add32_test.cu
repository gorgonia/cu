extern "C" 
__global__ void add32(float* A, float *B, int size) {
    int block = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int index = block * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if(index >= size) return;

    A[index] = A[index] + B[index];
}