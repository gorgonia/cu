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
