package main

import (
	"fmt"
	"runtime"

	"gorgonia.org/cu"
	"gorgonia.org/tensor"
)

func matVecMulRowMajorNonTransposed() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	fmt.Println("RowMajor Non Transposed")
	e := newEngine()
	a := tensor.New(tensor.WithShape(2, 3), tensor.WithEngine(e), tensor.Of(tensor.Float64))
	b := tensor.New(tensor.WithShape(3), tensor.WithEngine(e), tensor.Of(tensor.Float64))
	c := tensor.New(tensor.WithShape(2), tensor.WithEngine(e), tensor.Of(tensor.Float64))

	defer e.ctx.Close()
	defer e.Free(cu.DevicePtr(a.Uintptr()), int64(a.MemSize()))
	defer e.Free(cu.DevicePtr(b.Uintptr()), int64(b.MemSize()))
	defer e.Free(cu.DevicePtr(c.Uintptr()), int64(c.MemSize()))

	ad := a.Data().([]float64)
	for i := range ad {
		ad[i] = float64(i + 1)
	}

	bd := b.Data().([]float64)
	for i := range bd {
		bd[i] = float64(i + 1)
	}
	fmt.Printf("A: \n%v\nB: %v\n", a, b)

	cd := c.Data().([]float64)
	for i := range cd {
		cd[i] = float64(1000)
	}
	fmt.Printf("C%v\n", c)

	// var c2 tensor.Tensor
	var err error
	// c2, err = tensor.MatVecMul(a, b, tensor.WithReuse(c))
	err = e.MatVecMul(a, b, c)
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
}

func matVecMulRowMajorTransposed() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	fmt.Println("RowMajor Transposed")
	e := newEngine()
	a := tensor.New(tensor.WithShape(2, 3), tensor.WithEngine(e), tensor.Of(tensor.Float64))
	b := tensor.New(tensor.WithShape(2), tensor.WithEngine(e), tensor.Of(tensor.Float64))
	c := tensor.New(tensor.WithShape(3), tensor.WithEngine(e), tensor.Of(tensor.Float64))

	defer e.ctx.Close()
	defer e.Free(cu.DevicePtr(a.Uintptr()), int64(a.MemSize()))
	defer e.Free(cu.DevicePtr(b.Uintptr()), int64(b.MemSize()))
	defer e.Free(cu.DevicePtr(c.Uintptr()), int64(c.MemSize()))

	ad := a.Data().([]float64)
	for i := range ad {
		ad[i] = float64(i + 1)
	}
	a.T()

	bd := b.Data().([]float64)
	for i := range bd {
		bd[i] = float64(i + 1)
	}
	fmt.Printf("A: \n%v\nB: %v\n", a, b)

	cd := c.Data().([]float64)
	for i := range cd {
		cd[i] = float64(1000)
	}
	fmt.Printf("C%v\n", c)

	// var c2 tensor.Tensor
	var err error
	// c2, err = tensor.MatVecMul(a, b, tensor.WithReuse(c))
	err = e.MatVecMul(a, b, c)
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
}

func matVecMulColmajorNonTransposed() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	fmt.Println("ColMajor Non Transposed")
	e := newEngine()
	a := tensor.New(tensor.WithShape(2, 3), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))
	b := tensor.New(tensor.WithShape(3), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))
	c := tensor.New(tensor.WithShape(2), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))

	defer e.ctx.Close()
	defer e.Free(cu.DevicePtr(a.Uintptr()), int64(a.MemSize()))
	defer e.Free(cu.DevicePtr(b.Uintptr()), int64(b.MemSize()))
	defer e.Free(cu.DevicePtr(c.Uintptr()), int64(c.MemSize()))

	ad := a.Data().([]float64)
	dataA := []float64{1, 4, 2, 5, 3, 6}
	for i := range ad {
		ad[i] = dataA[i]
	}

	bd := b.Data().([]float64)
	for i := range bd {
		bd[i] = float64(i + 1)
	}
	fmt.Printf("A: \n%v\nB: %v\n", a, b)

	cd := c.Data().([]float64)
	for i := range cd {
		cd[i] = float64(1000)
	}
	fmt.Printf("C%v\n", c)

	// var c2 tensor.Tensor
	var err error
	// c2, err = tensor.MatVecMul(a, b, tensor.WithReuse(c))
	err = e.MatVecMul(a, b, c)
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

}

func matVecMulColmajorTransposed() {
	fmt.Println("ColMajor Transposed")
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	e := newEngine()
	a := tensor.New(tensor.WithShape(2, 3), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))
	b := tensor.New(tensor.WithShape(2), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))
	c := tensor.New(tensor.WithShape(3), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))

	defer e.ctx.Close()
	defer e.Free(cu.DevicePtr(a.Uintptr()), int64(a.MemSize()))
	defer e.Free(cu.DevicePtr(b.Uintptr()), int64(b.MemSize()))
	defer e.Free(cu.DevicePtr(c.Uintptr()), int64(c.MemSize()))

	ad := a.Data().([]float64)
	dataA := []float64{1, 4, 2, 5, 3, 6}
	for i := range ad {
		ad[i] = dataA[i]
	}
	a.T()

	bd := b.Data().([]float64)
	for i := range bd {
		bd[i] = float64(i + 1)
	}
	fmt.Printf("A: \n%v\nB: %v\n", a, b)

	cd := c.Data().([]float64)
	for i := range cd {
		cd[i] = float64(1000)
	}
	fmt.Printf("C%v\n", c)

	// var c2 tensor.Tensor
	var err error
	// c2, err = tensor.MatVecMul(a, b, tensor.WithReuse(c))
	err = e.MatVecMul(a, b, c)
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
}

func matMulColmajorNTNT() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	fmt.Println("ColMajor Non Transposed Non Transposed")
	e := newEngine()
	a := tensor.New(tensor.WithShape(2, 3), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))
	b := tensor.New(tensor.WithShape(3, 4), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))
	c := tensor.New(tensor.WithShape(2, 4), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))

	defer e.ctx.Close()
	defer e.Free(cu.DevicePtr(a.Uintptr()), int64(a.MemSize()))
	defer e.Free(cu.DevicePtr(b.Uintptr()), int64(b.MemSize()))
	defer e.Free(cu.DevicePtr(c.Uintptr()), int64(c.MemSize()))

	ad := a.Data().([]float64)
	dataA := []float64{1, 4, 2, 5, 3, 6}
	for i := range ad {
		ad[i] = dataA[i]
	}

	bd := b.Data().([]float64)
	dataB := []float64{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11}
	for i := range bd {
		bd[i] = dataB[i]
	}
	fmt.Printf("A: \n%v\nB: \n%v\n", a, b)

	cd := c.Data().([]float64)
	for i := range cd {
		cd[i] = float64(1000)
	}
	fmt.Printf("C:\n%v\n", c)

	// var c2 tensor.Tensor
	var err error
	_, err = tensor.MatMul(a, b, tensor.WithReuse(c))
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

}

func matMulColmajorTNT() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	fmt.Println("ColMajor Transposed Non Transposed")
	e := newEngine()
	a := tensor.New(tensor.WithShape(2, 3), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))
	b := tensor.New(tensor.WithShape(2, 4), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))
	c := tensor.New(tensor.WithShape(3, 4), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))

	defer e.ctx.Close()
	defer e.Free(cu.DevicePtr(a.Uintptr()), int64(a.MemSize()))
	defer e.Free(cu.DevicePtr(b.Uintptr()), int64(b.MemSize()))
	defer e.Free(cu.DevicePtr(c.Uintptr()), int64(c.MemSize()))

	ad := a.Data().([]float64)
	dataA := []float64{1, 4, 2, 5, 3, 6}
	for i := range ad {
		ad[i] = dataA[i]
	}
	a.T()

	bd := b.Data().([]float64)
	dataB := []float64{0, 4, 1, 5, 2, 6, 3, 7}
	for i := range bd {
		bd[i] = dataB[i]
	}
	fmt.Printf("A: \n%v\nB: \n%v\n", a, b)

	cd := c.Data().([]float64)
	for i := range cd {
		cd[i] = float64(1000)
	}
	fmt.Printf("C:\n%v\n", c)

	// var c2 tensor.Tensor
	var err error
	_, err = tensor.MatMul(a, b, tensor.WithReuse(c))
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

}

func matMulColmajorTT() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	fmt.Println("ColMajor Transposed Transposed")
	e := newEngine()
	a := tensor.New(tensor.WithShape(2, 3), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))
	b := tensor.New(tensor.WithShape(4, 2), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))
	c := tensor.New(tensor.WithShape(3, 4), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))

	defer e.ctx.Close()
	defer e.Free(cu.DevicePtr(a.Uintptr()), int64(a.MemSize()))
	defer e.Free(cu.DevicePtr(b.Uintptr()), int64(b.MemSize()))
	defer e.Free(cu.DevicePtr(c.Uintptr()), int64(c.MemSize()))

	ad := a.Data().([]float64)
	dataA := []float64{1, 4, 2, 5, 3, 6}
	for i := range ad {
		ad[i] = dataA[i]
	}
	a.T()

	bd := b.Data().([]float64)
	dataB := []float64{0, 2, 4, 6, 1, 3, 5, 7}
	for i := range bd {
		bd[i] = dataB[i]
	}
	b.T()
	fmt.Printf("A: \n%v\nB: \n%v\n", a, b)

	cd := c.Data().([]float64)
	for i := range cd {
		cd[i] = float64(1000)
	}
	fmt.Printf("C:\n%v\n", c)

	// var c2 tensor.Tensor
	var err error
	_, err = tensor.MatMul(a, b, tensor.WithReuse(c))
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

}

func matMulColmajorNTT() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	fmt.Println("ColMajor Non Transposed Transposed")
	e := newEngine()
	a := tensor.New(tensor.WithShape(2, 3), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))
	b := tensor.New(tensor.WithShape(4, 3), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))
	c := tensor.New(tensor.WithShape(2, 4), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))

	defer e.ctx.Close()
	defer e.Free(cu.DevicePtr(a.Uintptr()), int64(a.MemSize()))
	defer e.Free(cu.DevicePtr(b.Uintptr()), int64(b.MemSize()))
	defer e.Free(cu.DevicePtr(c.Uintptr()), int64(c.MemSize()))

	ad := a.Data().([]float64)
	dataA := []float64{1, 4, 2, 5, 3, 6}
	for i := range ad {
		ad[i] = dataA[i]
	}

	bd := b.Data().([]float64)
	dataB := []float64{0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11}
	for i := range bd {
		bd[i] = dataB[i]
	}
	b.T()
	fmt.Printf("A: \n%v\nB: \n%v\n", a, b)

	cd := c.Data().([]float64)
	for i := range cd {
		cd[i] = float64(1000)
	}
	fmt.Printf("C:\n%v\n", c)

	// var c2 tensor.Tensor
	var err error
	_, err = tensor.MatMul(a, b, tensor.WithReuse(c))
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

}

func matMulRowmajorNTNT() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	fmt.Println("RowMajor Non Transposed Non Transposed")
	e := newEngine()
	a := tensor.New(tensor.WithShape(2, 3), tensor.WithEngine(e), tensor.Of(tensor.Float64))
	b := tensor.New(tensor.WithShape(3, 4), tensor.WithEngine(e), tensor.Of(tensor.Float64))
	c := tensor.New(tensor.WithShape(2, 4), tensor.WithEngine(e), tensor.Of(tensor.Float64))

	defer e.ctx.Close()
	defer e.Free(cu.DevicePtr(a.Uintptr()), int64(a.MemSize()))
	defer e.Free(cu.DevicePtr(b.Uintptr()), int64(b.MemSize()))
	defer e.Free(cu.DevicePtr(c.Uintptr()), int64(c.MemSize()))

	ad := a.Data().([]float64)
	for i := range ad {
		ad[i] = float64(i + 1)
	}

	bd := b.Data().([]float64)
	for i := range bd {
		bd[i] = float64(i)
	}
	fmt.Printf("A: \n%v\nB: \n%v\n", a, b)

	cd := c.Data().([]float64)
	for i := range cd {
		cd[i] = float64(1000)
	}
	fmt.Printf("C:\n%v\n", c)

	// var c2 tensor.Tensor
	var err error
	_, err = tensor.MatMul(a, b, tensor.WithReuse(c))
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

}

func matMulRowmajorTNT() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	fmt.Println("RowMajor Transposed Non Transposed")
	e := newEngine()
	a := tensor.New(tensor.WithShape(3, 2), tensor.WithEngine(e), tensor.Of(tensor.Float64))
	b := tensor.New(tensor.WithShape(3, 4), tensor.WithEngine(e), tensor.Of(tensor.Float64))
	c := tensor.New(tensor.WithShape(2, 4), tensor.WithEngine(e), tensor.Of(tensor.Float64))

	defer e.ctx.Close()
	defer e.Free(cu.DevicePtr(a.Uintptr()), int64(a.MemSize()))
	defer e.Free(cu.DevicePtr(b.Uintptr()), int64(b.MemSize()))
	defer e.Free(cu.DevicePtr(c.Uintptr()), int64(c.MemSize()))

	ad := a.Data().([]float64)
	for i := range ad {
		ad[i] = float64(i + 1)
	}
	a.T()

	bd := b.Data().([]float64)
	for i := range bd {
		bd[i] = float64(i)
	}
	fmt.Printf("A: \n%v\nB: \n%v\n", a, b)

	cd := c.Data().([]float64)
	for i := range cd {
		cd[i] = float64(1000)
	}
	fmt.Printf("C:\n%v\n", c)

	// var c2 tensor.Tensor
	var err error
	_, err = tensor.MatMul(a, b, tensor.WithReuse(c))
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

}

func matMulRowmajorTT() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	fmt.Println("RowMajor Transposed Non Transposed")
	e := newEngine()
	a := tensor.New(tensor.WithShape(3, 2), tensor.WithEngine(e), tensor.Of(tensor.Float64))
	b := tensor.New(tensor.WithShape(4, 3), tensor.WithEngine(e), tensor.Of(tensor.Float64))
	c := tensor.New(tensor.WithShape(2, 4), tensor.WithEngine(e), tensor.Of(tensor.Float64))

	defer e.ctx.Close()
	defer e.Free(cu.DevicePtr(a.Uintptr()), int64(a.MemSize()))
	defer e.Free(cu.DevicePtr(b.Uintptr()), int64(b.MemSize()))
	defer e.Free(cu.DevicePtr(c.Uintptr()), int64(c.MemSize()))

	ad := a.Data().([]float64)
	for i := range ad {
		ad[i] = float64(i + 1)
	}
	a.T()

	bd := b.Data().([]float64)
	for i := range bd {
		bd[i] = float64(i)
	}
	b.T()

	fmt.Printf("A: \n%v\nB: \n%v\n", a, b)

	cd := c.Data().([]float64)
	for i := range cd {
		cd[i] = float64(1000)
	}
	fmt.Printf("C:\n%v\n", c)

	// var c2 tensor.Tensor
	var err error
	_, err = tensor.MatMul(a, b, tensor.WithReuse(c))
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

}

func matMulRowmajorNTT() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	fmt.Println("RowMajor Transposed Non Transposed")
	e := newEngine()
	a := tensor.New(tensor.WithShape(2, 3), tensor.WithEngine(e), tensor.Of(tensor.Float64))
	b := tensor.New(tensor.WithShape(4, 3), tensor.WithEngine(e), tensor.Of(tensor.Float64))
	c := tensor.New(tensor.WithShape(2, 4), tensor.WithEngine(e), tensor.Of(tensor.Float64))

	defer e.ctx.Close()
	defer e.Free(cu.DevicePtr(a.Uintptr()), int64(a.MemSize()))
	defer e.Free(cu.DevicePtr(b.Uintptr()), int64(b.MemSize()))
	defer e.Free(cu.DevicePtr(c.Uintptr()), int64(c.MemSize()))

	ad := a.Data().([]float64)
	for i := range ad {
		ad[i] = float64(i + 1)
	}

	bd := b.Data().([]float64)
	for i := range bd {
		bd[i] = float64(i)
	}
	b.T()

	fmt.Printf("A: \n%v\nB: \n%v\n", a, b)

	cd := c.Data().([]float64)
	for i := range cd {
		cd[i] = float64(1000)
	}
	fmt.Printf("C:\n%v\n", c)

	// var c2 tensor.Tensor
	var err error
	_, err = tensor.MatMul(a, b, tensor.WithReuse(c))
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

}

func outerColMajor() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	fmt.Println("RowMajor Non Transposed")
	e := newEngine()
	a := tensor.New(tensor.WithShape(3), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))
	b := tensor.New(tensor.WithShape(2), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))
	c := tensor.New(tensor.WithShape(3, 2), tensor.WithEngine(e), tensor.Of(tensor.Float64), tensor.AsFortran(nil))

	defer e.ctx.Close()
	defer e.Free(cu.DevicePtr(a.Uintptr()), int64(a.MemSize()))
	defer e.Free(cu.DevicePtr(b.Uintptr()), int64(b.MemSize()))
	defer e.Free(cu.DevicePtr(c.Uintptr()), int64(c.MemSize()))

	ad := a.Data().([]float64)
	for i := range ad {
		ad[i] = float64(i + 1)
	}

	bd := b.Data().([]float64)
	for i := range bd {
		bd[i] = float64(i)
	}
	fmt.Printf("A: \n%v\nB: %v\n", a, b)

	cd := c.Data().([]float64)
	for i := range cd {
		cd[i] = float64(1000)
	}
	fmt.Printf("C\n%v\n", c)

	// var c2 tensor.Tensor
	var err error
	_, err = tensor.Outer(a, b, tensor.WithReuse(c))
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
}

func outerRowMajor() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	fmt.Println("RowMajor Non Transposed")
	e := newEngine()
	a := tensor.New(tensor.WithShape(3), tensor.WithEngine(e), tensor.Of(tensor.Float64))
	b := tensor.New(tensor.WithShape(2), tensor.WithEngine(e), tensor.Of(tensor.Float64))
	c := tensor.New(tensor.WithShape(3, 2), tensor.WithEngine(e), tensor.Of(tensor.Float64))

	defer e.ctx.Close()
	defer e.Free(cu.DevicePtr(a.Uintptr()), int64(a.MemSize()))
	defer e.Free(cu.DevicePtr(b.Uintptr()), int64(b.MemSize()))
	defer e.Free(cu.DevicePtr(c.Uintptr()), int64(c.MemSize()))

	ad := a.Data().([]float64)
	for i := range ad {
		ad[i] = float64(i + 1)
	}

	bd := b.Data().([]float64)
	for i := range bd {
		bd[i] = float64(i)
	}
	fmt.Printf("A: \n%v\nB: %v\n", a, b)

	cd := c.Data().([]float64)
	for i := range cd {
		cd[i] = float64(1000)
	}
	fmt.Printf("C\n%v\n", c)

	// var c2 tensor.Tensor
	var err error
	_, err = tensor.Outer(a, b, tensor.WithReuse(c))
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
}
