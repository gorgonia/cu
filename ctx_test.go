package cu

import "testing"

func TestContext(t *testing.T) {
	ctx := NewContext(Device(0), SchedAuto)
	mem, err := ctx.MemAlloc(24)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%v", mem)
}

func TestMultipleContext(t *testing.T) {
	d := Device(0)
	ctx0 := NewManuallyManagedContext(d, SchedAuto)
	ctx1 := NewManuallyManagedContext(d, SchedAuto)

	errChan0 := make(chan error)
	errChan1 := make(chan error)
	go ctx0.Run(errChan0)
	go ctx1.Run(errChan1)

	if err := <-errChan0; err != nil {
		t.Fatalf("err while initializing run of ctx0 %v", err)
	}
	if err := <-errChan1; err != nil {
		t.Fatalf("err while initializing run of ctx1 %v", err)
	}

	var mem0, mem1 DevicePtr
	var err error
	if mem0, err = ctx0.MemAlloc(1024); err != nil {
		t.Errorf("Err while alloc in ctx0: %v", err)
	}

	if mem1, err = ctx1.MemAlloc(1024); err != nil {
		t.Errorf("Err while alloc in ctx1: %v", err)
	}

	t.Logf("Mem0: %v", mem0)
	t.Logf("Mem1: %v", mem1)
	ctx0.MemFree(mem0)
	ctx1.MemFree(mem1)

	if err = ctx0.Error(); err != nil {
		t.Errorf("Error while freeing %v", err)
	}
	if err = ctx1.Error(); err != nil {
		t.Errorf("Error while freeing %v", err)
	}

	// runtime.GC()
}
