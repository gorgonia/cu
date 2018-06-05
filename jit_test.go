package cu

import (
	"testing"
	"unsafe"
)

func TestJIT(t *testing.T) {
	const (
		nBlockSize = 256
		nGridSize  = 64
		nMemBytes  = nBlockSize * nGridSize * int64(unsafe.Sizeof(int(0)))
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

	hostData := make([]int, nBlockSize*nGridSize)
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
		if i != v {
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
	check(err)
	defer link.Destroy()

	if unsafe.Sizeof(int(0)) == 4 {
		err := link.AddData(JITInputPTX, myPtx32, "ptx32")
		check(err)
	} else {
		err := link.AddData(JITInputPTX, myPtx64, "ptx64")
		check(err)
	}

	binary, err := link.Complete()
	check(err)

	t.Logf("Complete %vms\n", walltime.Result)
	t.Logf("Linker output: %s\n", string(logbuffer))
	t.Logf("Error output: %s\n", string(errorbuffer))

	module, err := LoadData(binary)
	check(err)

	function, err := module.Function("assignTID")
	check(err)

	return module, function
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}

/*
__global__ void assignTID(int *data)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = tid;
}
*/

const myPtx64 = `
.version 3.2
.target sm_20
.address_size 64
.visible .entry assignTID(
	.param .u64 data
)
{
	.reg .s32 	%r<5>;
	.reg .s64 	%rd<5>;
	ld.param.u64 	%rd1, [data];
	cvta.to.global.u64 	%rd2, %rd1;
	.loc 1 3 1
	mov.u32 	%r1, %ntid.x;
	mov.u32 	%r2, %ctaid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r1, %r2, %r3;
	mul.wide.s32 	%rd3, %r4, 4;
	add.s64 	%rd4, %rd2, %rd3;
	.loc 1 4 1
	st.global.u32 	[%rd4], %r4;
	.loc 1 5 2
	ret;
}
`

const myPtx32 = `
.version 3.2
.target sm_20
.address_size 32
.visible .entry assignTID(
	.param .u32 data
)
{
	.reg .s32 	%r<9>;
	ld.param.u32 	%r1, [data];
	cvta.to.global.u32 	%r2, %r1;
	.loc 1 3 1
	mov.u32 	%r3, %ntid.x;
	mov.u32 	%r4, %ctaid.x;
	mov.u32 	%r5, %tid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	.loc 1 4 1
	shl.b32 	%r7, %r6, 2;
	add.s32 	%r8, %r2, %r7;
	st.global.u32 	[%r8], %r6;
	.loc 1 5 2
	ret;
}
`
