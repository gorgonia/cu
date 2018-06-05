__global__ void assignTID(int *data)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = tid;
}