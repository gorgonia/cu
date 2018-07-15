# How to Use cuBLAS with Tensors #

This is an example repository for guides on how to use cuBLAS with Gorgonia's `tensor` library.


Because a `*tensor.Dense` can execute code pertaining to the tensor as long as an appropriate engine is provided, 

The key to this is in the `Engine` data structure, defined as such:

```
type Engine struct {
	tensor.StdEng
	ctx cu.Context
	*cublas.Standard
}
```

Look into `engine.go` for a minimalist implementation of a tensor `Engine`. When it allocates memory, it uses CUDA's ManagedMemory - this allows for Go to also read the memory (as long as the thread is locked)

For implementation of matrix multiplication and various other linear algebra things, look into `main.go` - the interfaces of this engine is implemented there.

For how to use with the provided engine, look into `tests.go`. Typical usage pattern looks like this:

```
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	e := newEngine()
	a := tensor.New(tensor.WithShape(2, 3), tensor.WithEngine(e), tensor.Of(tensor.Float64))
	b := tensor.New(tensor.WithShape(2), tensor.WithEngine(e), tensor.Of(tensor.Float64))
	c := tensor.New(tensor.WithShape(3), tensor.WithEngine(e), tensor.Of(tensor.Float64))

	defer e.ctx.Close()
	defer e.Free(cu.DevicePtr(a.Uintptr()), int64(a.MemSize()))
	defer e.Free(cu.DevicePtr(b.Uintptr()), int64(b.MemSize()))
	defer e.Free(cu.DevicePtr(c.Uintptr()), int64(c.MemSize()))

	_, err = tensor.MatVecMul(a, b, tensor.WithReuse(c))
	if err != nil || e.Standard.Err() != nil {
		fmt.Println(err)
		fmt.Println(e.Standard.Err())
	} else {
		if ravel, err := e.Accessible(c); err != nil {
			fmt.Printf("Error %v", err)
		} else {
			fmt.Printf("C:\n%v\nRaveled: %v\n\n", c, ravel)
		}
	}
```