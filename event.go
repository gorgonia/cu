package cu

// #include <cuda.h>
import "C"
import (
	"unsafe"

	"github.com/pkg/errors"
)

// Event represents a CUDA event
type Event uintptr

func makeEvent(event C.CUevent) Event { return Event(uintptr(unsafe.Pointer(event))) }

func (e Event) c() C.CUevent {
	return C.CUevent(unsafe.Pointer(uintptr(e)))
}

func MakeEvent(flags EventFlags) (event Event, err error) {
	CFlags := C.uint(flags)
	var CEvent C.CUevent
	err = result(C.cuEventCreate(&CEvent, CFlags))
	event = makeEvent(CEvent)
	return
}

func DestroyEvent(event *Event) (err error) {
	e := event.c()
	err = result(C.cuEventDestroy(e))
	*event = 0
	return
}

func (ctx *Ctx) MakeEvent(flags EventFlags) (event Event, err error) {
	CFlags := C.uint(flags)
	var CEvent C.CUevent
	f := func() error { return result(C.cuEventCreate(&CEvent, CFlags)) }
	if err = ctx.Do(f); err != nil {
		err = errors.Wrap(err, "MakeEvent")
		return
	}
	event = makeEvent(CEvent)
	return
}

func (ctx *Ctx) DestroyEvent(event *Event) {
	e := event.c()
	f := func() error { return result(C.cuEventDestroy(e)) }
	ctx.err = ctx.Do(f)
	*event = 0
	return
}
