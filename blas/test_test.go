package cublas

import (
	"github.com/pkg/errors"
	"gorgonia.org/cu"
)

func testSetup() (dev cu.Device, err error) {
	devices, _ := cu.NumDevices()

	if devices == 0 {
		err = errors.Errorf("NoDevice")
		return
	}

	dev = cu.Device(0)
	return
}
