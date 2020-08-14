package cu

import (
	"runtime"
	"testing"
	"unsafe"
)

func TestCUContext(t *testing.T) {
	devices, _ := NumDevices()
	if devices == 0 {
		return
	}

	d := Device(0)
	ctx, err := d.MakeContext(SchedAuto)
	if err != nil {
		t.Fatal(err)
	}

	maj, _, err := d.ComputeCapability()
	if err != nil {
		t.Error(err)
	}

	version, err := ctx.APIVersion()
	if err != nil {
		t.Error(err)
	}
	t.Logf("API Version: %v", version)

	current, err := CurrentContext()
	if err != nil {
		t.Fatal(err)
	}
	if current != ctx {
		t.Error("Expected current context to be ctx")
	}

	currentDevice, err := CurrentDevice()
	if err != nil {
		t.Fatal(err)
	}
	if currentDevice != d {
		t.Error("Expected currentDevice to be the same as d")
	}

	// flags - should be the same as the one we set up
	currentFlags, err := CurrentFlags()
	if err != nil {
		t.Fatal(err)
	}
	if currentFlags != SchedAuto {
		t.Error(err)
	}

	if maj >= 3 {
		defaultSharedConf, err := SharedMemConfig()
		if err != nil {
			t.Fatal(err)
		}

		var newBankSize SharedConfig

		for _, c := range []SharedConfig{FourByteBankSize, EightByteBankSize} {
			if c != defaultSharedConf {
				newBankSize = c
				break
			}
		}

		// shared conf
		if err := SetSharedMemConfig(newBankSize); err != nil {
			t.Fatal(err)
		}

		sharedConf, err := SharedMemConfig()
		if err != nil {
			t.Fatal(err)
		}

		if sharedConf != newBankSize && sharedConf != defaultSharedConf {
			t.Errorf("Expected sharedMemConf to be SharedConfig of %v or %v. Got %v instead", newBankSize, defaultSharedConf, sharedConf)
		}

		if sharedConf == defaultSharedConf {
			t.Logf("The graphics card does not have a configurable shared memory banks")
		}

		// cache config
		if err = SetCurrentCacheConfig(PreferEqual); err != nil {
			t.Fatal(err)
		}

		cacheconf, err := CurrentCacheConfig()
		if err != nil {
			t.Fatal(err)
		}
		if cacheconf != PreferEqual {
			t.Error("expected cache config to be PreferEqual")
		}
	}

	// push pop
	popped, err := PopCurrentCtx()
	if err != nil {
		t.Fatal(err)
	}
	if popped != ctx {
		t.Error("Expected popped context to be the same as ctx")
	}

	empty, err := CurrentDevice()
	if err == nil {
		t.Error("Expected an error when there is no current context")
	}
	if empty != d {
		t.Errorf("Expected empty to be 0. Empty: %v d: %v", empty, d)
	}

	if err = PushCurrentCtx(popped); err != nil {
		t.Fatal(err)
	}

	// get set limit
	if err = SetLimit(StackSize, 64); err != nil {
		t.Fatal(err)
	}
	ss, err := Limits(StackSize)
	if err != nil {
		t.Fatal(err)
	}
	if ss != int64(64) {
		t.Errorf("Expected stack size should be 64, ss: %v", ss)
	}

	// finally we destroy the context
	if err = ctx.Destroy(); err != nil {
		t.Error(err)
	}

	if (ctx != CUContext{}) {
		t.Error("expected ctx to be set to 0")
	}
}

func TestMultipleContextSingleHostThread(t *testing.T) {
	var err error
	var dev Device
	var ctx0, ctx1 CUContext
	var mod0, mod1 Module
	var fn0, fn1 Function
	var mem0, mem1 DevicePtr

	// prepare data
	data := make([]float32, 1000)
	result := make([]float32, 1000)
	for i := range data {
		data[i] = float32(i)
	}
	size := int64(len(data) * 4)

	// tests start
	if dev, err = GetDevice(0); err != nil {
		t.Fatal(err)
	}

	if ctx0, err = dev.MakeContext(SchedAuto); err != nil {
		t.Fatal(err)
	}
	if ctx1, err = dev.MakeContext(SchedAuto); err != nil {
		t.Fatal(err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// set current context to ctx0
	if err = SetCurrentContext(ctx0); err != nil {
		t.Fatal(err)
	}

	if mod0, err = LoadData(add32PTX); err != nil {
		t.Fatalf("Cannot load module for ctx0: %v", err)
	}

	if fn0, err = mod0.Function("add32"); err != nil {
		t.Fatalf("Cannot get add32(): %v", err)
	}

	if mem0, err = AllocAndCopy(unsafe.Pointer(&data[0]), size); err != nil {
		t.Fatalf("Cannot alloc and copy %v", err)
	}

	args := []unsafe.Pointer{
		unsafe.Pointer(&mem0),
		unsafe.Pointer(&mem0),
		unsafe.Pointer(&size),
	}

	if err = fn0.LaunchAndSync(1, 1, 1, len(data), 1, 1, 0, Stream{}, args); err != nil {
		t.Errorf("Failed to launcj add32: %v", err)
	}

	if err = MemcpyDtoH(unsafe.Pointer(&result[0]), mem0, size); err != nil {
		t.Errorf("Memcpy failed %v", err)
	}

	// repeat the same for ctx1
	if err = SetCurrentContext(ctx1); err != nil {
		t.Fatal(err)
	}

	if mod1, err = LoadData(add32PTX); err != nil {
		t.Fatalf("Cannot load module for ctx0: %v", err)
	}

	if fn1, err = mod1.Function("add32"); err != nil {
		t.Fatalf("Cannot get add32(): %v", err)
	}

	if mem1, err = AllocAndCopy(unsafe.Pointer(&data[0]), size); err != nil {
		t.Fatalf("Cannot alloc and copy %v", err)
	}

	args = []unsafe.Pointer{
		unsafe.Pointer(&mem1),
		unsafe.Pointer(&mem1),
		unsafe.Pointer(&size),
	}

	if err = fn1.LaunchAndSync(1, 1, 1, len(data), 1, 1, 0, Stream{}, args); err != nil {
		t.Errorf("Failed to launcj add32: %v", err)
	}

	if err = MemcpyDtoH(unsafe.Pointer(&result[0]), mem0, size); err != nil {
		t.Errorf("Memcpy failed %v", err)
	}

	// TIME TO MIX IT UP:

	// calling fn0 when the current context is ctx1
	args = []unsafe.Pointer{
		unsafe.Pointer(&mem1),
		unsafe.Pointer(&mem1),
		unsafe.Pointer(&size),
	}

	if err = fn0.LaunchAndSync(1, 1, 1, len(data), 1, 1, 0, Stream{}, args); err == nil {
		t.Errorf("Expected error when launching a kernel defined in a different context")
	}
	t.Log(err)

	// switch back to the first context
	if err = SetCurrentContext(ctx0); err != nil {
		t.Errorf("Failed to swtch to ctx0 %v", err)
	}
	if err = fn0.LaunchAndSync(1, 1, 1, len(data), 1, 1, 0, Stream{}, args); err != nil {
		t.Errorf("fn0 errored while using memory declared in ctx1: %v", err)
	}
	t.Log(err)
}
