package cu

import (
	"path/filepath"
	"testing"
	"unsafe"
)

func TestModule(t *testing.T) {
	devices, _ := NumDevices()
	if devices == 0 {
		t.Log("No Devices Found")
		return
	}
	ctx, err := Device(0).MakeContext(SchedAuto)
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Destroy()

	mod, err := Load(filepath.Join("testdata", "module_test.ptx"))
	if err != nil {
		t.Fatal(err)
	}
	defer mod.Unload()

	f, err := mod.Function("testMemset")
	if err != nil {
		t.Fatal(err)
	}

	N := 1000
	N4 := 4 * int64(N)
	a := make([]float32, N)
	A, err := MemAlloc(N4)
	if err != nil {
		t.Fatal(err)
	}
	defer MemFree(A)
	aptr := unsafe.Pointer(&a[0])

	if err = MemcpyHtoD(A, aptr, N4); err != nil {
		t.Fatal(err)
	}

	var value float32
	value = 42

	var n int
	n = N / 2

	block := 128
	grid := DivUp(N, block)
	shmem := 0
	args := []unsafe.Pointer{unsafe.Pointer(&A), unsafe.Pointer(&value), unsafe.Pointer(&n)}
	if err = f.Launch(grid, 1, 1, block, 1, 1, shmem, Stream{}, args); err != nil {
		t.Fatal(err)
	}

	if err = MemcpyDtoH(aptr, A, N4); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < N/2; i++ {
		if a[i] != 42 {
			t.Fail()
		}
	}
	for i := N / 2; i < N; i++ {
		if a[i] != 0 {
			t.Fail()
		}
	}
}

// Integer division rounded up.
func DivUp(x, y int) int {
	return ((x - 1) / y) + 1
}
