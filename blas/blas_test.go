package cublas

import (
	"log"
	"reflect"
	"testing"
	"unsafe"

	"gonum.org/v1/gonum/blas"
	"gorgonia.org/cu"
)

func TestMultithreadedCalls(t *testing.T) {
	dev, err := testSetup()
	if err != nil {
		t.Fatal(err)
	}
	ctx := cu.NewContext(dev, cu.SchedAuto)
	impl := New(WithContext(ctx))

	// (20, 20) matrix of float32
	mem0, err := ctx.MemAlloc(400 * 4)
	if err != nil {
		t.Fatal(err)
	}

	// (20, 20) matrix of float32
	mem1, err := ctx.MemAlloc(400 * 4)
	if err != nil {
		t.Fatal(err)
	}

	// (20, 20) matrix of float32
	mem2, err := ctx.MemAlloc(400 * 4)
	if err != nil {
		t.Fatal(err)
	}

	mem0Hdr := reflect.SliceHeader{uintptr(mem0), 400, 400}
	mem0Slice := *(*[]float32)(unsafe.Pointer(&mem0Hdr))
	mem1Hdr := reflect.SliceHeader{uintptr(mem1), 400, 400}
	mem1Slice := *(*[]float32)(unsafe.Pointer(&mem1Hdr))
	mem2Hdr := reflect.SliceHeader{uintptr(mem2), 400, 400}
	mem2Slice := *(*[]float32)(unsafe.Pointer(&mem2Hdr))

	log.Printf("%d %d %p", len(mem1Slice), len(mem0Slice), unsafe.Pointer(&mem1Slice[0]))

	// SGEMM is running on a separate OS thread than the CUDA context, and it should still work fine
	impl.Sgemm(blas.NoTrans, blas.NoTrans, 20, 20, 20, 1, mem0Slice, 20, mem1Slice, 20, 0, mem2Slice, 20)
	if err = impl.Err(); err != nil {
		t.Fatal(err)
	}

	ctx.MemFree(mem0)
	if err = ctx.Error(); err != nil {
		t.Error(err)
	}

	ctx.MemFree(mem1)
	if err = ctx.Error(); err != nil {
		t.Error(err)
	}

	ctx.MemFree(mem2)
	if err = ctx.Error(); err != nil {
		t.Error(err)
	}
}
