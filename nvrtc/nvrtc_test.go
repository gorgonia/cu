package nvrtc_test

import (
	"testing"

	"gorgonia.org/cu/nvrtc"
)

func TestCompile(t *testing.T) {
	program, err := nvrtc.CreateProgram(`
		void add(float *A, float *B, int N) {
			int block = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
			int index = block * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
			if(index >= size) return;

			A[index] = A[index] + B[index];
		}
	`, `add.cu`)
	if err != nil {
		t.Fatalf("failed to create program: %v", err)
	}

	err = program.AddNameExpression(`add`)
	if err != nil {
		t.Fatalf("failed to AddNameExpression: %v", err)
	}

	err = program.Compile()
	if err != nil {
		t.Fatalf("failed to Compile: %v", err)
	}

	loweredName, err := program.GetLoweredName(`add`)
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
