package cu

import (
	"math"
	"testing"
	"unsafe"
)

func TestAttributes(t *testing.T) {
	var dev Device
	var ctx Context
	var err error

	if dev, ctx, err = testSetup(); err != nil {
		if err.Error() == "NoDevice" {
			return
		}
		t.Fatal(err)
	}

	var mtpb, maj, min int
	if mtpb, err = dev.Attribute(MaxThreadsPerBlock); err != nil {
		t.Fatalf("Failed while getting MaxThreadsPerBlock: %v", err)
	}

	if maj, err = dev.Attribute(ComputeCapabilityMajor); err != nil {
		t.Fatalf("Failed while getting Compute Capability Major: %v", err)
	}

	if min, err = dev.Attribute(ComputeCapabilityMinor); err != nil {
		t.Fatalf("Failed while getting Compute Capability Minor: %v", err)
	}

	var attrs []int
	if attrs, err = dev.Attributes(MaxThreadsPerBlock, ComputeCapabilityMajor, ComputeCapabilityMinor); err != nil {
		t.Error(err)
	}

	if attrs[0] != mtpb {
		t.Errorf("Expected MaxThreadsPerBlock to be %v. Got %v instead", mtpb, attrs[0])
	}
	if attrs[1] != maj {
		t.Errorf("Expected ComputeCapabilityMajor to be %v. Got %v instead", maj, attrs[1])
	}
	if attrs[2] != min {
		t.Errorf("Expected ComputeCapabilityMinor to be %v. Got %v instead", min, attrs[2])
	}

	DestroyContext(&ctx)
}

func TestLaunchAndSync(t *testing.T) {
	var err error
	var ctx Context
	var mod Module
	var fn Function

	if _, ctx, err = testSetup(); err != nil {
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

	a := make([]float32, 1000)
	b := make([]float32, 1000)
	for i := range b {
		a[i] = 1
		b[i] = 1
	}

	size := int64(len(a) * 4)

	var memA, memB DevicePtr
	if memA, err = MemAlloc(size); err != nil {
		t.Fatalf("Failed to allocate for a: %v", err)
	}
	if memB, err = MemAlloc(size); err != nil {
		t.Fatalf("Failed to allocate for b: %v", err)
	}

	if err = MemcpyHtoD(memA, unsafe.Pointer(&a[0]), size); err != nil {
		t.Fatalf("Failed to copy memory from a: %v", err)
	}

	if err = MemcpyHtoD(memB, unsafe.Pointer(&b[0]), size); err != nil {
		t.Fatalf("Failed to copy memory from b: %v", err)
	}

	args := []unsafe.Pointer{
		unsafe.Pointer(&memA),
		unsafe.Pointer(&memB),
		unsafe.Pointer(&size),
	}

	if err = fn.LaunchAndSync(1, 1, 1, len(a), 1, 1, 1, Stream(0), args); err != nil {
		t.Error("Launch and Sync Failed: %v", err)
	}

	if err = MemcpyDtoH(unsafe.Pointer(&a[0]), memA, size); err != nil {
		t.Fatalf("Failed to copy memory to a: %v", err)
	}

	if err = MemcpyDtoH(unsafe.Pointer(&b[0]), memB, size); err != nil {
		t.Fatalf("Failed to copy memory to b: %v", err)
	}

	for _, v := range a {
		if v != float32(2) {
			t.Error("Expected all values to be 2.")
			break
		}
	}

	MemFree(memA)
	MemFree(memB)
	Unload(mod)
	DestroyContext(&ctx)
}

func TestAllocAndCopy(t *testing.T) {
	var err error
	var ctx Context
	var mem DevicePtr

	if _, ctx, err = testSetup(); err != nil {
		if err.Error() == "NoDevice" {
			return
		}
		t.Fatal(err)
	}

	SetCurrent(ctx)

	a := make([]float32, 1024)
	p := unsafe.Pointer(&a[0])
	bytesize := int64(len(a) * 4)
	if mem, err = AllocAndCopy(p, bytesize); err != nil {
		t.Fatalf("%+v", err)
	}

	if err = MemsetD32(mem, math.Float32bits(1.0), 512); err != nil {
		t.Fatalf("errored while memsetting the first 512 elems to 1.0: %v", err)
	}

	if err = MemcpyDtoH(p, mem, bytesize); err != nil {
		t.Fatalf("errored while copying from device back to slice: %v", err)
	}

	for i, v := range a {
		if i < 512 && v != float32(1) {
			t.Errorf("Expected a[%d] to be 1.0. Got %v instead", i, v)
			break
		}

		if i >= 512 && v != 0 {
			t.Errorf("Expected a[%d] to be 0.0. Got %v instead", i, v)
			break
		}
	}

	MemFree(mem)
	DestroyContext(&ctx)
}
