package cu

// This file provides CGO flags to find CUDA libraries and headers.

//#cgo LDFLAGS:-lcuda
//
////default location:
//#cgo LDFLAGS:-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib
//#cgo CFLAGS: -I/usr/local/cuda/include/
//
////default location if not properly symlinked:
//#cgo LDFLAGS:-L/usr/local/cuda-6.0/lib64 -L/usr/local/cuda-6.0/lib
//#cgo LDFLAGS:-L/usr/local/cuda-5.5/lib64 -L/usr/local/cuda-5.5/lib
//#cgo LDFLAGS:-L/usr/local/cuda-5.0/lib64 -L/usr/local/cuda-5.0/lib
//#cgo CFLAGS: -I/usr/local/cuda-6.0/include/
//#cgo CFLAGS: -I/usr/local/cuda-5.5/include/
//#cgo CFLAGS: -I/usr/local/cuda-5.0/include/
//
////Ubuntu 15.04:
//#cgo LDFLAGS:-L/usr/lib/x86_64-linux-gnu/
//#cgo CFLAGS: -I/usr/include
//
////arch linux:
//#cgo LDFLAGS:-L/opt/cuda/lib64 -L/opt/cuda/lib
//#cgo CFLAGS: -I/opt/cuda/include
//
////WINDOWS:
//#cgo windows LDFLAGS:-LC:/cuda/v5.0/lib/x64 -LC:/cuda/v5.5/lib/x64 -LC:/cuda/v6.0/lib/x64 -LC:/cuda/v6.5/lib/x64 -LC:/cuda/v7.0/lib/x64
//#cgo windows CFLAGS: -IC:/cuda/v5.0/include -IC:/cuda/v5.5/include -IC:/cuda/v6.0/include -IC:/cuda/v6.5/include -IC:/cuda/v7.0/include
import "C"
