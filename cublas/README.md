# cublas #

Package `cublas` implements a Go API for CUDA's cuBLAS. It matches the [gonum/blas](https://github.com/gonum/blas) interface. 

# How To Use # 

To install: `go get -u github.com/chewxy/cu`

The [CUDA Toolkit 8.0](https://developer.nvidia.com/cuda-toolkit) is required. LDFlags and CFlags may not be quite accurate. File an issue if you find one, thank you.

# How This Package Is Developed #

The majority of the CUDA interface was generated with the `cublasgen` program. The `cublasgen` program was adapted from the `cgo` generator from the `gonum/blas` package.

The `cudagen.h` file was generated based off the propietary header from nvidia, then further edited (several variable names were renamed) to match the cblas interface in order to quickly generate the API.