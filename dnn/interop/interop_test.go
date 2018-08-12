package t2cudnn

import (
	"testing"

	"gorgonia.org/tensor"
)

func TestDescribe(t *testing.T) {
	T := tensor.New(tensor.WithShape(2, 3, 4, 5), tensor.Of(tensor.Float64))
	desc, err := Describe(T)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%v", desc)

	T2 := tensor.New(tensor.WithShape(2, 3), tensor.Of(tensor.Float32))
	_, err = Describe(T2)
	if err == nil {
		t.Errorf("Expected an error when a tensor < 4D is passed in")
	}
}
