package cu

import (
	"testing"
	"unsafe"
)

func TestAttributes(t *testing.T) {
	devices, _ := NumDevices()

	if devices == 0 {
		return
	}

	d := Device(0)
	mtpb, err := d.Attribute(MaxThreadsPerBlock)
	if err != nil {
		t.Fatalf("Failed while getting MaxThreadsPerBlock: %v", err)
	}

	maj, err := d.Attribute(ComputeCapabilityMajor)
	if err != nil {
		t.Fatalf("Failed while getting Compute Capability Major: %v", err)
	}

	min, err := d.Attribute(ComputeCapabilityMinor)
	if err != nil {
		t.Fatalf("Failed while getting Compute Capability Minor: %v", err)
	}

	attrs, err := d.Attributes(MaxThreadsPerBlock, ComputeCapabilityMajor, ComputeCapabilityMinor)
	if err != nil {
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
}

func TestLaunchAndSync(t *testing.T) {
	devices, _ := NumDevices()

	if devices == 0 {
		return
	}

	var err error
	var ctx Context
	var mod Module
	var fn Function

	d := Device(0)
	if ctx, err = d.MakeContext(SchedAuto); err != nil {
		t.Fatal(err)
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

	if mod, err = LoadData(add32PTX); err != nil {
		t.Fatalf("Cannot load add32: %v", err)
	}

	if fn, err = mod.Function("add32"); err != nil {
		t.Fatalf("Cannot get add32(): %v", err)
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

	if err = MemcpyDtoH(unsafe.Pointer(&b[0]), memA, size); err != nil {
		t.Fatalf("Failed to copy memory to b: %v", err)
	}

	for _, v := range a {
		if v != float32(2) {
			t.Error("Expected all values to be 2.")
			break
		}
	}

	Unload(mod)
	DestroyContext(&ctx)
}
