// cudatest tests the existence of CUDA by running a simple Go program that uses CUDA.
package main

import (
	"fmt"

	"gorgonia.org/cu"
	"os"
	"github.com/cloudflare/cfssl/log"
)

func main() {
	fmt.Printf("\nCUDA version: %v\n", cu.Version())
	devices, err := cu.NumDevices()
	if err != nil {
		log.Info("issue found: %+v", err)
		os.Exit(1)
	}
	fmt.Printf("CUDA devices: %v\n\n", devices)

	for d := 0; d < devices; d++ {
		name, _ := cu.Device(d).Name()
		cr, _ := cu.Device(d).Attribute(cu.ClockRate)
		mem, _ := cu.Device(d).TotalMem()
		maj, _ := cu.Device(d).Attribute(cu.ComputeCapabilityMajor)
		min, _ := cu.Device(d).Attribute(cu.ComputeCapabilityMinor)
		fmt.Printf("Device %d\n========\nName      :\t%q\n", d, name)
		fmt.Printf("Clock Rate:\t%v kHz\n", cr)
		fmt.Printf("Memory    :\t%v bytes\n", mem)
		fmt.Printf("Compute   : \t%d.%d\n\n", maj, min)
	}
}
