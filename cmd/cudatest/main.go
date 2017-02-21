package main

import (
	"fmt"

	"github.com/chewxy/cu"
)

func main() {
	fmt.Printf("CUDA version: %v\n", cu.Version())
	devices, _ := cu.NumDevices()
	fmt.Printf("CUDA devices: %v\n", devices)

	for d := 0; d < devices; d++ {
		name, _ := cu.Device(d).Name()
		cr, _ := cu.Device(d).Attribute(cu.ClockRate)
		mem, _ := cu.Device(d).TotalMem()
		maj, _ := cu.Device(d).Attribute(cu.ComputeCapabilityMajor)
		min, _ := cu.Device(d).Attribute(cu.ComputeCapabilityMinor)
		fmt.Printf("Device %d\n========\nName      :\t%q\n", d, name)
		fmt.Printf("Clock Rate:\t%v kHz\n", cr)
		fmt.Printf("Memory    :\t%v bytes\n", mem)
		fmt.Printf("Compute   : \t%d.%d\n", maj, min)
	}

}
