# cublas #

Package `cublas` implements a Go API for CUDA's cuBLAS. It matches the [gonum/blas](https://github.com/gonum/blas) interface. 

# How To Use # 

To install: `go get -u gorgonia.org/cu`

The [CUDA Toolkit 8.0](https://developer.nvidia.com/cuda-toolkit) is required. LDFlags and CFlags may not be quite accurate. File an issue if you find one, thank you.

Bear in mind that cublas only supports FORTRAN ordered matrices. Most Go matrices are created with the C ordering (gonum/matrix, gorgonia/tensor), therefore care must be applied.

For example, here's how to use `Dgemm`:

```go 
func main() {
	dev := cu.Device(0)
	ctx, err := dev.MakeContext(cu.SchedAuto)
	if err != nil {
		log.Fatal(err)
	}
	defer cu.DestroyContext(&ctx)

	dt := tensor.Float64
	s0 := tensor.Shape{5, 10}
	s1 := tensor.Shape{10, 12}
	s2 := tensor.Shape{5, 12}

	memsize0 := calcMemsize(dt, s0)
	mem0, err := cu.MemAllocManaged(memsize0, cu.AttachGlobal)
	if err != nil {
		log.Fatal(err)
	}
	mat0 := tensor.New(tensor.Of(dt), tensor.WithShape(s0...), tensor.FromMemory(uintptr(mem0), uintptr(memsize0)))
	d0 := mat0.Data().([]float64)
	for i := range d0 {
		d0[i] = float64(i + 1)
	}
	fmt.Printf("A: \n%#v\n", mat0)

	memsize1 := calcMemsize(dt, s1)
	mem1, err := cu.MemAllocManaged(memsize1, cu.AttachGlobal)
	if err != nil {
		log.Fatal(err)
	}
	mat1 := tensor.New(tensor.Of(dt), tensor.WithShape(s1...), tensor.FromMemory(uintptr(mem1), uintptr(memsize1)))
	d1 := mat1.Data().([]float64)
	for i := range d1 {
		d1[i] = float64(i + 1)
	}
	fmt.Printf("B: \n%#v\n", mat1)

	memsize2 := calcMemsize(dt, s2)
	mem2, err := cu.MemAllocManaged(memsize2, cu.AttachGlobal)
	if err != nil {
		log.Fatal(err)
	}
	mat2 := tensor.New(tensor.Of(dt), tensor.WithShape(s2...), tensor.FromMemory(uintptr(mem2), uintptr(memsize2)))
	d2 := mat2.Data().([]float64)
	fmt.Printf("C: \n%#v\n", mat2)

	impl := cublas.NewImplementation()

	m := s0[0]
	k := s0[1]
	n := s1[1]
	lda := mat0.Strides()[0]
	ldb := mat1.Strides()[0]
	ldc := mat2.Strides()[0]
	alpha := 1.0
	beta := 0.0
	impl.Dgemm(blas.NoTrans, blas.NoTrans, n, m, k, alpha, d1, ldn, d0, lda, beta, d2, ldc)
	if err := cu.Synchronize(); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("C: \n%#v\n", mat2)
	cu.MemFree(mem0)
	cu.MemFree(mem1)
	cu.MemFree(mem2)
}
```

These are things to note: To do a A×B, you need to essentially do Bᵀ×Aᵀ.

# How This Package Is Developed #

The majority of the CUDA interface was generated with the `cublasgen` program. The `cublasgen` program was adapted from the `cgo` generator from the `gonum/blas` package.

The `cudagen.h` file was generated based off the propietary header from nvidia, then further edited (several variable names were renamed) to match the cblas interface in order to quickly generate the API.