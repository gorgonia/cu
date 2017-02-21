package cu

import "testing"

func TestAttributes(t *testing.T) {
	devices, _ := NumDevices()

	if devices == 0 {
		return
	}

	d := Device(0)
	mtpb, err := d.Attribute(MaxThreadsPerBlock)
	if err != nil {
		t.Fatalf("Failed while getting MaxThreadsPerBlock: %v", err)
	}

	maj, err := d.Attribute(ComputeCapabilityMajor)
	if err != nil {
		t.Fatalf("Failed while getting Compute Capability Major: %v", err)
	}

	min, err := d.Attribute(ComputeCapabilityMinor)
	if err != nil {
		t.Fatalf("Failed while getting Compute Capability Minor: %v", err)
	}

	attrs, err := d.Attributes(MaxThreadsPerBlock, ComputeCapabilityMajor, ComputeCapabilityMinor)
	if err != nil {
		t.Error(err)
	}

	if attrs[0] != mtpb {
		t.Errorf("Expected MaxThreadsPerBlock to be %v. Got %v instead", mtpb, attrs[0])
	}
	if attrs[1] != maj {
		t.Errorf("Expected ComputeCapabilityMajor to be %v. Got %v instead", maj, attrs[1])
	}
	if attrs[2] != min {
		t.Errorf("Expected ComputeCapabilityMinor to be %v. Got %v instead", min, attrs[2])
	}
}
