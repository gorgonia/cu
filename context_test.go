package cu

import "testing"

func TestContext(t *testing.T) {
	devices, _ := NumDevices()
	if devices == 0 {
		return
	}

	d := Device(0)
	ctx, err := d.MakeContext(SchedAuto)
	if err != nil {
		t.Fatal(err)
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

	// shared conf
	if err := SetSharedMemConfig(EightByteBankSize); err != nil {
		t.Fatal(err)
	}

	sharedConf, err := SharedMemConfig()
	if err != nil {
		t.Fatal(err)
	}

	if sharedConf != EightByteBankSize {
		t.Error("Expected sharedMemConf to be EightByteBankSize")
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
	if err = DestroyContext(&ctx); err != nil {
		t.Error(err)
	}

	if ctx != 0 {
		t.Error("expected ctx to be set to 0")
	}
}
