package cublas

// #cgo CFLAGS: -I/usr/local/cuda-9.0/targets/x86_64-linux/include -I/usr/local/cuda/include
// #cgo LDFLAGS: -lcublas
// #cgo LDFLAGS: -L/usr/local/cuda-9.0/targets/x86_64-linux/lib -L/usr/local/cuda/lib64 
import "C"
