package cu

/*
#include <stdlib.h>
void handleCUDACB(void* v);
*/
import "C"
import (
	"sync"
	"unsafe"
)

// fake.go handles faking of C pointers of Go functions

var fakepointers = make(map[unsafe.Pointer]HostFunction)
var lock sync.RWMutex

// RegisterFunc is used to register a Go based callback such that it may be called by CUDA.
func RegisterFunc(fn HostFunction) unsafe.Pointer {
	var ptr unsafe.Pointer = C.malloc(C.size_t(1))
	if ptr == nil {
		panic("Cannot allocate a fake pointer")
	}

	lock.Lock()
	fakepointers[ptr] = fn
	lock.Unlock()

	return ptr
}

func getHostFn(ptr unsafe.Pointer) HostFunction {
	if ptr == nil {
		return nil
	}
	lock.RLock()
	retVal := fakepointers[ptr]
	lock.RUnlock()
	return retVal
}

func deregisterFunc(ptr unsafe.Pointer) {
	if ptr == nil {
		return
	}

	lock.Lock()
	delete(fakepointers, ptr)
	lock.Unlock()

	C.free(ptr)
}

//export handleCUDACB
func handleCUDACB(fn unsafe.Pointer) {
	callback := getHostFn(fn)
	callback()
}
