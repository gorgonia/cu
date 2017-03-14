package cu

import (
	"runtime"
	"testing"
	"unsafe"
)

func TestBatchContext(t *testing.T) {
	var err error
	var dev Device
	var ctx Context
	var mod Module
	var fn Function

	if dev, ctx, err = testSetup(); err != nil {
		if err.Error() == "NoDevice" {
			return
		}
		t.Fatal(err)
	}

	if mod, err = LoadData(add32PTX); err != nil {
		t.Fatalf("Cannot load add32: %v", err)
	}

	if fn, err = mod.Function("add32"); err != nil {
		t.Fatalf("Cannot get add32(): %v", err)
	}

	bctx := NewBatchedContext(ctx, dev)

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	doneChan := make(chan struct{})

	a := make([]float32, 1000)
	b := make([]float32, 1000)
	go func() {
		for i := range b {
			a[i] = 1
			b[i] = 1
		}

		size := int64(len(a) * 4)

		var memA, memB DevicePtr
		if memA, err = bctx.AllocAndCopy(unsafe.Pointer(&a[0]), size); err != nil {
			t.Fatalf("Cannot allocate A: %v", err)

		}

		if memB, err = bctx.MemAlloc(size); err != nil {
			t.Fatalf("Cannot allocate B: %v", err)
		}

		args := []unsafe.Pointer{
			unsafe.Pointer(&memA),
			unsafe.Pointer(&memB),
			unsafe.Pointer(&size),
		}

		bctx.MemcpyHtoD(memB, unsafe.Pointer(&b[0]), size)
		bctx.LaunchKernel(fn, 1, 1, 1, len(a), 1, 1, 0, Stream(0), args)
		bctx.Synchronize()
		bctx.MemcpyDtoH(unsafe.Pointer(&a[0]), memA, size)
		bctx.MemcpyDtoH(unsafe.Pointer(&b[0]), memB, size)
		bctx.MemFree(memA)
		bctx.MemFree(memB)
		bctx.workAvailable <- struct{}{}
		doneChan <- struct{}{}
	}()

loop:
	for {
		select {
		case <-bctx.workAvailable:
			bctx.DoWork()
		case <-doneChan:
			break loop
		default:
		}
	}

	for _, v := range a {
		if v != float32(2) {
			t.Errorf("Expected all values to be 2. %v", a)
			break
		}
	}

	Unload(mod)
	DestroyContext(&ctx)
}

func TestLargeBatch(t *testing.T) {
	var err error
	var dev Device
	var ctx Context
	var mod Module
	var fn Function

	if dev, ctx, err = testSetup(); err != nil {
		if err.Error() == "NoDevice" {
			return
		}
		t.Fatal(err)
	}

	if mod, err = LoadData(add32PTX); err != nil {
		t.Fatalf("Cannot load add32: %v", err)
	}

	if fn, err = mod.Function("add32"); err != nil {
		t.Fatalf("Cannot get add32(): %v", err)
	}

	dev.TotalMem()

	beforeFree, _, _ := MemInfo()
	bctx := NewBatchedContext(ctx, dev)

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	doneChan := make(chan struct{})

	a := make([]float32, 1000)
	b := make([]float32, 1000)
	for i := range b {
		a[i] = 1
		b[i] = 1
	}
	size := int64(len(a) * 4)

	var frees []DevicePtr
	go func() {
		var memA, memB DevicePtr

		for i := 0; i < 104729; i++ {
			if memA, err = bctx.AllocAndCopy(unsafe.Pointer(&a[0]), size); err != nil {
				t.Fatalf("Cannot allocate A: %v", err)

			}

			if memB, err = bctx.MemAlloc(size); err != nil {
				t.Fatalf("Cannot allocate B: %v", err)
			}

			args := []unsafe.Pointer{
				unsafe.Pointer(&memA),
				unsafe.Pointer(&memB),
				unsafe.Pointer(&size),
			}

			bctx.MemcpyHtoD(memB, unsafe.Pointer(&b[0]), size)
			bctx.LaunchKernel(fn, 1, 1, 1, len(a), 1, 1, 0, Stream(0), args)
			bctx.Synchronize()

			if i%13 == 0 {
				frees = append(frees, memA)
				frees = append(frees, memB)
			} else {
				bctx.MemFree(memA)
				bctx.MemFree(memB)
			}
		}

		bctx.MemcpyDtoH(unsafe.Pointer(&a[0]), memA, size)
		bctx.MemcpyDtoH(unsafe.Pointer(&b[0]), memB, size)
		for _, free := range frees {
			bctx.MemFree(free)
		}
		bctx.workAvailable <- struct{}{}
		doneChan <- struct{}{}
	}()

loop:
	for {
		select {
		case <-bctx.workAvailable:
			bctx.DoWork()
		case <-doneChan:
			break loop
		default:
		}
	}

	for _, v := range a {
		if v != float32(2) {
			t.Errorf("Expected all values to be 2. %v", a)
			break
		}
	}

	afterFree, _, _ := MemInfo()

	if afterFree != beforeFree {
		t.Errorf("Before: Freemem: %v. After %v", beforeFree, afterFree)
	}

	Unload(mod)
	DestroyContext(&ctx)

}

func BenchmarkNoBatching(bench *testing.B) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var err error
	var ctx Context
	var mod Module
	var fn Function

	if _, ctx, err = testSetup(); err != nil {
		if err.Error() == "NoDevice" {
			return
		}
		bench.Fatal(err)
	}

	if mod, err = LoadData(add32PTX); err != nil {
		bench.Fatalf("Cannot load add32: %v", err)
	}

	if fn, err = mod.Function("add32"); err != nil {
		bench.Fatalf("Cannot get add32(): %v", err)
	}

	a := make([]float32, 1000000)
	b := make([]float32, 1000000)
	for i := range b {
		a[i] = 1
		b[i] = 1
	}

	size := int64(len(a) * 4)

	var memA, memB DevicePtr
	if memA, err = MemAlloc(size); err != nil {
		bench.Fatalf("Failed to allocate for a: %v", err)
	}
	if memB, err = MemAlloc(size); err != nil {
		bench.Fatalf("Failed to allocate for b: %v", err)
	}

	args := []unsafe.Pointer{
		unsafe.Pointer(&memA),
		unsafe.Pointer(&memB),
		unsafe.Pointer(&size),
	}

	// ACTUAL BENCHMARK STARTS HERE
	for i := 0; i < bench.N; i++ {
		for j := 0; j < 100; j++ {
			if err = MemcpyHtoD(memA, unsafe.Pointer(&a[0]), size); err != nil {
				bench.Fatalf("Failed to copy memory from a: %v", err)
			}

			if err = MemcpyHtoD(memB, unsafe.Pointer(&b[0]), size); err != nil {
				bench.Fatalf("Failed to copy memory from b: %v", err)
			}

			if err = fn.LaunchAndSync(100, 10, 1, 1000, 1, 1, 1, Stream(0), args); err != nil {
				bench.Error("Launch and Sync Failed: %v", err)
			}

			if err = MemcpyDtoH(unsafe.Pointer(&a[0]), memA, size); err != nil {
				bench.Fatalf("Failed to copy memory to a: %v", err)
			}

			if err = MemcpyDtoH(unsafe.Pointer(&b[0]), memB, size); err != nil {
				bench.Fatalf("Failed to copy memory to b: %v", err)
			}
		}
	}
	MemFree(memA)
	MemFree(memB)
	Unload(mod)
	DestroyContext(&ctx)

}

func BenchmarkBatching(bench *testing.B) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var err error
	var dev Device
	var ctx Context
	var mod Module
	var fn Function

	if dev, ctx, err = testSetup(); err != nil {
		if err.Error() == "NoDevice" {
			return
		}
		bench.Fatal(err)
	}

	if mod, err = LoadData(add32PTX); err != nil {
		bench.Fatalf("Cannot load add32: %v", err)
	}

	if fn, err = mod.Function("add32"); err != nil {
		bench.Fatalf("Cannot get add32(): %v", err)
	}

	a := make([]float32, 1000000)
	b := make([]float32, 1000000)
	for i := range b {
		a[i] = 1
		b[i] = 1
	}

	size := int64(len(a) * 4)

	var memA, memB DevicePtr
	if memA, err = MemAlloc(size); err != nil {
		bench.Fatalf("Failed to allocate for a: %v", err)
	}
	if memB, err = MemAlloc(size); err != nil {
		bench.Fatalf("Failed to allocate for b: %v", err)
	}

	bctx := NewBatchedContext(ctx, dev)

	args := []unsafe.Pointer{
		unsafe.Pointer(&memA),
		unsafe.Pointer(&memB),
		unsafe.Pointer(&size),
	}

	// ACTUAL BENCHMARK STARTS HERE
	workAvailable := bctx.WorkAvailable()
	for i := 0; i < bench.N; i++ {
		for j := 0; j < 100; j++ {
			select {
			case <-workAvailable:
				bctx.DoWork()
			default:
				bctx.MemcpyHtoD(memA, unsafe.Pointer(&a[0]), size)
				bctx.MemcpyHtoD(memB, unsafe.Pointer(&b[0]), size)
				bctx.LaunchKernel(fn, 100, 10, 1, 1000, 1, 1, 0, Stream(0), args)
				bctx.Synchronize()
				bctx.MemcpyDtoH(unsafe.Pointer(&a[0]), memA, size)
				bctx.MemcpyDtoH(unsafe.Pointer(&b[0]), memB, size)
			}
		}
	}

	MemFree(memA)
	MemFree(memB)
	Unload(mod)
	DestroyContext(&ctx)

}
