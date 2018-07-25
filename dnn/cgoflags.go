package cudnn

// #cgo LDFLAGS:-lcuda
// #cgo LDFLAGS:-lcudnn
//
// // default locs:
// #cgo LDFLAGS:-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib
// #cgo CFLAGS: -I/usr/include/x86_64-linux-gnu -I/usr/local/cuda-9.0/targets/x86_64-linux/include -I/usr/local/cuda/include
import "C"
