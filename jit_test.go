package cu

import (
	"testing"
	"unsafe"
)

func TestJIT(t *testing.T) {
	const (
		nBlockSize = 256
		nGridSize  = 64
		nMemBytes  = nBlockSize * nGridSize * 4
	)

	device, err := GetDevice(0)
	if err != nil {
		t.Fatal(err)
	}

	ctx, err := device.MakeContext(SchedAuto)
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Destroy()

	module, kernel := compileJIT(t)
	defer module.Unload()

	hostData := make([]int32, nBlockSize*nGridSize)
	deviceData, err := MemAlloc(nMemBytes)
	if err != nil {
		t.Fatal(err)
	}
	defer MemFree(deviceData)

	err = kernel.Launch(
		nGridSize, 1, 1,
		nBlockSize, 1, 1,
		0, Stream{},
		[]unsafe.Pointer{
			unsafe.Pointer(&deviceData),
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	MemcpyDtoH(unsafe.Pointer(&hostData[0]), deviceData, nMemBytes)

	bad := 0
	for i, v := range hostData {
		if int32(i) != v {
			t.Errorf("Error at %v got %v\n", i, v)
			bad++
			if bad > 10 {
				t.Fatal("too many errors")
			}
		}
	}
}

func compileJIT(t *testing.T) (Module, Function) {
	walltime := &JITWallTime{0}
	logbuffer := make([]byte, 10<<10)
	errorbuffer := make([]byte, 10<<10)

	link, err := NewLink(
		walltime,
		&JITInfoLogBuffer{logbuffer},
		&JITErrorLogBuffer{errorbuffer},
		&JITLogVerbose{true},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer link.Destroy()

	err = link.AddData(JITInputPTX, myPtx64, "ptx64")
	if err != nil {
		t.Fatal(err)
	}

	binary, err := link.Complete()
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Complete %vms\n", walltime.Result)
	t.Logf("Linker output: %s\n", string(logbuffer))
	t.Logf("Error output: %s\n", string(errorbuffer))

	module, err := LoadData(binary)
	if err != nil {
		t.Fatal(err)
	}

	function, err := module.Function("assignTID")
	if err != nil {
		t.Fatal(err)
	}

	return module, function
}

/*
__global__ void assignTID(int *data)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = tid;
}
*/

const myPtx64 = `
.version 5.0
.target sm_20
.address_size 64

	// .globl	assignTID

.visible .entry assignTID(
	.param .u64 _Z9assignTIDPi_param_0
)
{
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [_Z9assignTIDPi_param_0];
	cvta.to.global.u64 	%rd2, %rd1;
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r2, %r1, %r3;
	mul.wide.s32 	%rd3, %r4, 4;
	add.s64 	%rd4, %rd2, %rd3;
	st.global.u32 	[%rd4], %r4;
	ret;
}
`
