package cu

/*
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

func registerFunc(fn HostFunction) unsafe.Pointer {
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
		return HostFunction{}
	}
	lock.Rlock()
	retVal := fakepointers[ptr]
	lock.RUnlock()
	return retVal
}

func deregisterFunc(ptr unsafe.Pointer) {
	if ptr == nil {
		return
	}

	mutex.Lock()
	delete(fakepointers, ptr)
	mutex.Unlock()

	C.free(ptr)
}

// export handleCUDACB
func handleCUDACB(fn unsafe.Pointer) {
	callback := getHostFn(fn)
	callback.Func(callback.UserData...)
}
