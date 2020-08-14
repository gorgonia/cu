package cu

import (
	"log"
	"runtime"
	"sync/atomic"
	"testing"
	"unsafe"

	_ "net/http/pprof"
)

func TestBatchContext(t *testing.T) {
	log.Print("BatchContext")
	var err error
	var dev Device
	var cuctx CUContext
	var mod Module
	var fn Function

	if dev, cuctx, err = testSetup(); err != nil {
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
	ctx := newContext(cuctx)
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
		bctx.LaunchKernel(fn, 1, 1, 1, len(a), 1, 1, 0, Stream{}, args)
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
		}
	}
	if err = Synchronize(); err != nil {
		t.Errorf("Failed to Sync %v", err)
	}

	for _, v := range a {
		if v != float32(2) {
			t.Errorf("Expected all values to be 2. %v", a)
			break
		}
	}

	mod.Unload()
	cuctx.Destroy()
}

func TestLargeBatch(t *testing.T) {
	log.Printf("Large batch")
	var err error
	var dev Device
	var cuctx CUContext
	var mod Module
	var fn Function

	if dev, cuctx, err = testSetup(); err != nil {
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

	ctx := newContext(cuctx)
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

	var freeCount uint32
	go func(fc *uint32) {
		var memA, memB DevicePtr
		var frees []DevicePtr

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
			bctx.LaunchKernel(fn, 1, 1, 1, len(a), 1, 1, 0, Stream{}, args)
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
		atomic.AddUint32(fc, uint32(len(frees)))
		bctx.workAvailable <- struct{}{}
		doneChan <- struct{}{}
	}(&freeCount)

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

	bctx.DoWork()
	if err = Synchronize(); err != nil {
		t.Errorf("Failed to Sync %v", err)
	}

	for _, v := range a {
		if v != float32(2) {
			t.Errorf("Expected all values to be 2. %v", a)
			break
		}
	}
	mod.Unload()
	afterFree, _, _ := MemInfo()
	cuctx.Destroy()
	runtime.GC()

	if freeCount != 16114 {
		t.Errorf("Expected 16114 frees. Got %d instead", freeCount)
	}
	if afterFree != beforeFree {
		t.Logf("Before: Freemem: %v. After %v | Diff %v", beforeFree, afterFree, (beforeFree-afterFree)/1024)
	}

}

func BenchmarkNoBatching(bench *testing.B) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var err error
	var ctx CUContext
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

			if err = fn.LaunchAndSync(100, 10, 1, 1000, 1, 1, 1, Stream{}, args); err != nil {
				bench.Errorf("Launch and Sync Failed: %v", err)
			}

			if err = MemcpyDtoH(unsafe.Pointer(&a[0]), memA, size); err != nil {
				bench.Fatalf("Failed to copy memory to a: %v", err)
			}

			if err = MemcpyDtoH(unsafe.Pointer(&b[0]), memB, size); err != nil {
				bench.Fatalf("Failed to copy memory to b: %v", err)
			}
		}
		// useful for checking results
		// if i == 0 {
		// 	bench.Logf("%v", a[:10])
		// }
	}
	MemFree(memA)
	MemFree(memB)
	mod.Unload()
	ctx.Destroy()
}

func BenchmarkBatching(bench *testing.B) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var err error
	var dev Device
	var cuctx CUContext
	var mod Module
	var fn Function

	if dev, cuctx, err = testSetup(); err != nil {
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

	ctx := newContext(cuctx)
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
			done := make(chan struct{}, 1)
			go func(done chan struct{}) {
				bctx.MemcpyHtoD(memA, unsafe.Pointer(&a[0]), size)
				bctx.MemcpyHtoD(memB, unsafe.Pointer(&b[0]), size)
				bctx.LaunchKernel(fn, 100, 10, 1, 1000, 1, 1, 0, Stream{}, args)
				bctx.Synchronize()
				bctx.MemcpyDtoH(unsafe.Pointer(&a[0]), memA, size)
				bctx.MemcpyDtoH(unsafe.Pointer(&b[0]), memB, size)
				bctx.Signal()
				done <- struct{}{}
			}(done)

		work:
			for {
				select {
				case <-workAvailable:
					bctx.DoWork()
				case <-done:
					break work
				}
			}

		}

		if err := bctx.Errors(); err != nil {
			bench.Fatalf("Failed with errors in benchmark %d. Error: %v", i, err)
		}

		// useful for checking results
		// if i == 0 {
		// 	bench.Logf("%v", a[:10])
		// }
	}
	MemFree(memA)
	MemFree(memB)
	mod.Unload()
	cuctx.Destroy()
}
