package cu

import (
	"bytes"
	"fmt"
	"testing"
)

func TestDevice(t *testing.T) {
	devices, err := NumDevices()
	if err != nil {
		t.Fatal(err)
	}
	if devices == 0 {
		return
	}

	buf := new(bytes.Buffer)

	for id := 0; id < devices; id++ {
		d, err := GetDevice(id)
		if err != nil {
			t.Fatal(err)
		}

		name, err := d.Name()
		if err != nil {
			t.Fatal(err)
		}

		cr, err := d.Attribute(ClockRate)
		if err != nil {
			t.Fatal(err)
		}

		mem, err := d.TotalMem()
		if err != nil {
			t.Fatal(err)
		}

		maj, err := d.Attribute(ComputeCapabilityMajor)
		if err != nil {
			t.Fatal(err)
		}

		min, err := d.Attribute(ComputeCapabilityMinor)
		if err != nil {
			t.Fatal(err)
		}

		fmt.Fprintf(buf, "Device %d\n========\nName      :\t%q\n", d, name)
		fmt.Fprintf(buf, "Clock Rate:\t%v kHz\n", cr)
		fmt.Fprintf(buf, "Memory    :\t%v bytes\n", mem)
		fmt.Fprintf(buf, "Compute   : \t%d.%d\n", maj, min)
		t.Log(buf.String())

		buf.Reset()
	}
}

func TestVersion(t *testing.T) {
	t.Logf("CUDA Toolkit version: %v", Version())
}
