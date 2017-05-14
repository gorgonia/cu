package cu

// #include <cuda.h>
import "C"
import "unsafe"

// Event represents a CUDA event
type Event uintptr

func (e Event) c() C.CUevent {
	return C.CUevent(unsafe.Pointer(uintptr(e)))
}

func MakeEvent(flags EventFlags) (event Event, err error) {
	CFlags := C.uint(flags)
	var CEvent C.CUevent
	err = result(C.cuEventCreate(&CEvent, CFlags))
	event = Event(uintptr(unsafe.Pointer(CEvent)))
	return
}

func DestroyEvent(event *Event) (err error) {
	e := event.c()
	err = result(C.cuEventDestroy(e))
	*event = 0
	return
}
