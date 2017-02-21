package cu

import (
	"math"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
)

func TestArray(t *testing.T) {
	devices, _ := NumDevices()
	if devices == 0 {
		t.Log("No Devices Found")
		return
	}

	assert := assert.New(t)
	desc := Array3Desc{
		Width:       64,
		Height:      64,
		NumChannels: 1,
		Format:      Float32,
	}
	ctx, _ := Device(0).MakeContext(SchedAuto)
	defer DestroyContext(&ctx)

	arr, err := Make3DArray(desc)
	assert.Nil(err)

	desc2, err := arr.Descriptor3()
	assert.Nil(err)
	assert.Equal(desc, desc2)

	desc3, err := arr.Descriptor()
	assert.Nil(err)
	assert.Equal(desc3.Format, desc.Format)
	assert.Equal(desc3.Width, desc.Width)
	assert.Equal(desc3.Height, desc.Height)
	assert.Equal(desc3.NumChannels, desc.NumChannels)

	err = DestroyArray(arr)
	assert.Nil(err)
}

func TestMalloc(t *testing.T) {
	devices, _ := NumDevices()
	if devices == 0 {
		t.Log("No Devices Found")
		return
	}
	ctx, _ := Device(0).MakeContext(SchedAuto)
	defer DestroyContext(&ctx)

	for i := 0; i < 1024; i++ {
		pointer, err := MemAlloc(16 * 1024 * 1024)
		if err != nil {
			t.Fatal(err)
		}
		if err = MemFree(pointer); err != nil {
			t.Fatal(err)
		}
	}
}

func TestDevicePtr_AddressRange(t *testing.T) {
	devices, _ := NumDevices()
	if devices == 0 {
		t.Log("No Devices Found")
		return
	}
	ctx, _ := Device(0).MakeContext(SchedAuto)
	defer DestroyContext(&ctx)

	// actual test

	N := int64(12345)
	ptr, _ := MemAlloc(N)
	size, base, err := ptr.AddressRange()
	if err != nil {
		t.Fatal(err)
	}
	if base != ptr {
		t.Errorf("Expected base to be the same as ptr")
	}
	if size != N {
		t.Errorf("Expected size to be %d", N)
	}
}

func TestMemInfo(t *testing.T) {
	devices, _ := NumDevices()
	if devices == 0 {
		t.Log("No Devices Found")
		return
	}
	ctx, _ := Device(0).MakeContext(SchedAuto)
	defer DestroyContext(&ctx)

	// actual test starts

	free, total, err := MemInfo()
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("MemGetInfo: %v / %v KiB", free, total)
	if free > total {
		t.Fail()
	}
	if total == 0 {
		t.Fail()
	}
}

func TestMemcpy(t *testing.T) {
	devices, _ := NumDevices()
	if devices == 0 {
		t.Log("No Devices Found")
		return
	}
	ctx, _ := Device(0).MakeContext(SchedAuto)
	defer DestroyContext(&ctx)

	// actual test starts

	var err error
	var dev1, dev2 DevicePtr
	N := int64(32 * 1024)
	host1 := make([]float32, N)
	for i := range host1 {
		host1[i] = float32(i)
	}
	host2 := make([]float32, N)
	if dev1, err = MemAlloc(int64(4 * N)); err != nil {
		t.Fatal(err)
	}
	if dev2, err = MemAlloc(int64(4 * N)); err != nil {
		t.Fatal(err)
	}

	if err = MemcpyHtoD(dev1, (unsafe.Pointer(&host1[0])), 4*N); err != nil {
		t.Fatal(err)
	}
	if err = MemcpyDtoD(dev2, dev1, 4*N); err != nil {
		t.Fatal(err)
	}

	if err = MemcpyDtoH((unsafe.Pointer(&host2[0])), dev2, 4*N); err != nil {
		t.Fatal(err)
	}
	for i := range host2 {
		if host2[i] != float32(i) {
			t.Fail()
		}
	}
	MemFree(dev1)
	MemFree(dev2)
}

func TestMemcpyAsync(t *testing.T) {
	devices, _ := NumDevices()
	if devices == 0 {
		t.Log("No Devices Found")
		return
	}
	ctx, _ := Device(0).MakeContext(SchedAuto)
	defer DestroyContext(&ctx)

	// actual test starts

	var err error
	N := int64(32 * 1024)
	host1 := make([]float32, N)
	for i := range host1 {
		host1[i] = float32(i)
	}
	host2 := make([]float32, N)
	dev1, _ := MemAlloc(int64(4 * N))
	dev2, _ := MemAlloc(int64(4 * N))
	stream, _ := MakeStream(DefaultStream)
	if err = MemcpyHtoDAsync(dev1, (unsafe.Pointer(&host1[0])), 4*N, stream); err != nil {
		t.Fatal(err)
	}
	if err = MemcpyDtoDAsync(dev2, dev1, 4*N, stream); err != nil {
		t.Fatal(err)
	}
	if err = MemcpyDtoHAsync((unsafe.Pointer(&host2[0])), dev2, 4*N, stream); err != nil {
		t.Fatal(err)
	}
	if err = stream.Synchronize(); err != nil {
		t.Fatal(err)
	}

	for i := range host2 {
		if host2[i] != float32(i) {
			t.Fail()
		}
	}
	MemFree(dev1)
	MemFree(dev2)
}

func TestMemset(t *testing.T) {
	devices, _ := NumDevices()
	if devices == 0 {
		t.Log("No Devices Found")
		return
	}
	ctx, _ := Device(0).MakeContext(SchedAuto)
	defer DestroyContext(&ctx)

	var err error
	var dev1 DevicePtr

	N := int64(32 * 1024)
	host1 := make([]float32, N)
	for i := range host1 {
		host1[i] = float32(i)
	}
	host2 := make([]float32, N)

	if dev1, err = MemAlloc(int64(4 * N)); err != nil {
		t.Fatal(err)
	}
	if err = MemcpyHtoD(dev1, (unsafe.Pointer(&host1[0])), 4*N); err != nil {
		t.Fatal(err)
	}
	if err = MemsetD32(dev1, math.Float32bits(42), N); err != nil {
		t.Fatal(err)
	}
	if err = MemsetD32(dev1, math.Float32bits(21), N/2); err != nil {
		t.Fatal(err)
	}
	if err = MemcpyDtoH((unsafe.Pointer(&host2[0])), dev1, 4*N); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < len(host2)/2; i++ {
		if host2[i] != 21 {
			t.Fail()
		}
	}
	for i := len(host2) / 2; i < len(host2); i++ {
		if host2[i] != 42 {
			t.Fail()
		}
	}
	if err = MemFree(dev1); err != nil {
		t.Fatal(err)
	}
}

func BenchmarkMallocFree1B(b *testing.B) {
	devices, _ := NumDevices()
	if devices == 0 {
		b.Log("No Devices Found")
		return
	}
	ctx, _ := Device(0).MakeContext(SchedAuto)
	defer DestroyContext(&ctx)

	var m DevicePtr
	var err error
	for i := 0; i < b.N; i++ {
		if m, err = MemAlloc(1); err != nil {
			b.Error(err)
			return
		}
		if err = MemFree(m); err != nil {
			b.Errorf("Error while freeing %v", err)
			return
		}

	}
}

func BenchmarkMallocFree1kB(b *testing.B) {
	devices, _ := NumDevices()
	if devices == 0 {
		b.Log("No Devices Found")
		return
	}
	ctx, _ := Device(0).MakeContext(SchedAuto)
	defer DestroyContext(&ctx)

	var m DevicePtr
	var err error
	for i := 0; i < b.N; i++ {
		if m, err = MemAlloc(1024); err != nil {
			b.Error(err)
			return
		}

		if err = MemFree(m); err != nil {
			b.Errorf("Error while freeing %v", err)
			return
		}

	}
}

func BenchmarkMallocFree1MB(b *testing.B) {
	devices, _ := NumDevices()
	if devices == 0 {
		b.Log("No Devices Found")
		return
	}
	ctx, _ := Device(0).MakeContext(SchedAuto)
	defer DestroyContext(&ctx)

	var m DevicePtr
	var err error
	for i := 0; i < b.N; i++ {
		if m, err = MemAlloc(1024 * 1024); err != nil {
			b.Error(err)
			return
		}

		if err = MemFree(m); err != nil {
			b.Errorf("Error while freeing %v", err)
			return
		}
	}
}

func BenchmarkMemcpy(b *testing.B) {
	devices, _ := NumDevices()
	if devices == 0 {
		b.Log("No Devices Found")
		return
	}
	b.SkipNow() // skip for now

	var dev1, dev2 DevicePtr
	var err error
	var ctx Context
	if ctx, err = Device(0).MakeContext(SchedAuto); err != nil {
		b.Fatal(err)
	}
	defer DestroyContext(&ctx)

	b.StopTimer()
	N := int64(32 * 1024 * 1024)
	host1 := make([]float32, N)
	host2 := make([]float32, N)
	if dev1, err = MemAlloc(int64(4 * N)); err != nil {
		b.Fatal(err)
	}
	defer MemFree(dev1)

	if dev2, err = MemAlloc(int64(4 * N)); err != nil {
		b.Fatal(err)
	}
	defer MemFree(dev2)

	b.SetBytes(4 * N)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		if err = MemcpyHtoD(dev1, (unsafe.Pointer(&host1[0])), 4*N); err != nil {
			b.Fatal(err)
		}
		if err = MemcpyDtoD(dev2, dev1, 4*N); err != nil {
			b.Fatal(err)
		}
		if err = MemcpyDtoH((unsafe.Pointer(&host2[0])), dev2, 4*N); err != nil {
			b.Fatal(err)
		}
	}
}
