package nvrtc_test

import (
	"testing"

	"gorgonia.org/cu/nvrtc"
)

func TestCompile(t *testing.T) {
	program, err := nvrtc.CreateProgram(`
		extern "C" __global__
		void saxpy(float a, float *x, float *y, float *out, size_t n) {
			size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
			if (tid < n) {
				out[tid] = a * x[tid] + y[tid];
			}
		}
	`, `saxpy.cu`)
	if err != nil {
		t.Fatalf("failed to create program: %v", err)
	}

	err = program.AddNameExpression(`saxpy`)
	if err != nil {
		t.Fatalf("failed to AddNameExpression: %v", err)
	}

	err = program.Compile()
	if err != nil {
		t.Fatalf("failed to Compile: %v", err)
	}

	loweredName, err := program.GetLoweredName(`saxpy`)
	if err != nil {
		t.Fatalf("failed to GetLoweredName: %v", err)
	}
	t.Logf("lowered name: %v", loweredName)

	ptx, err := program.GetPTX()
	if err != nil {
		t.Fatalf("failed to GetPTX: %v", err)
	}
	t.Logf("ptx: %v", ptx)

	programLog, err := program.GetLog()
	if err != nil {
		t.Fatalf("failed to GetLog: %v", err)
	}
	t.Logf("program log: %v", programLog)
}
