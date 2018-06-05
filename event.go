package cu

// #include <cuda.h>
import "C"
import (
	"github.com/pkg/errors"
)

// Event represents a CUDA event
type Event struct {
	ev C.CUevent
}

func makeEvent(event C.CUevent) Event { return Event{event} }

func (e Event) c() C.CUevent { return e.ev }

func MakeEvent(flags EventFlags) (event Event, err error) {
	CFlags := C.uint(flags)
	err = result(C.cuEventCreate(&event.ev, CFlags))
	return
}

func DestroyEvent(event *Event) (err error) {
	err = result(C.cuEventDestroy(event.ev))
	*event = Event{}
	return
}

func (ctx *Ctx) MakeEvent(flags EventFlags) (event Event, err error) {
	CFlags := C.uint(flags)
	f := func() error { return result(C.cuEventCreate(&event.ev, CFlags)) }
	if err = ctx.Do(f); err != nil {
		err = errors.Wrap(err, "MakeEvent")
		return
	}
	return
}

func (ctx *Ctx) DestroyEvent(event *Event) {
	f := func() error { return result(C.cuEventDestroy(event.ev)) }
	ctx.err = ctx.Do(f)
	*event = Event{}
	return
}
