package cu

// #include <cuda.h>
import "C"
import (
	"unsafe"

	"github.com/pkg/errors"
)

// Stream represents a CUDA stream.
type Stream uintptr

func makeStream(s C.CUstream) Stream { return Stream(uintptr(unsafe.Pointer(s))) }
func (s Stream) c() C.CUstream       { return C.CUstream(unsafe.Pointer(uintptr(s))) }

// MakeStream creates a stream. The flags determines the behaviors of the stream.
func MakeStream(flags StreamFlags) (Stream, error) {
	var s C.CUstream
	if err := result(C.cuStreamCreate(&s, C.uint(flags))); err != nil {
		return 0, err
	}
	return makeStream(s), nil
}

// MakeStreamWithPriority creates a stream with the given priority. The flags determines the behaviors of the stream.
// This API alters the scheduler priority of work in the stream. Work in a higher priority stream may preempt work already executing in a low priority stream.
//
// `priority` follows a convention where lower numbers represent higher priorities. '0' represents default priority.
//
// The range of meaningful numerical priorities can be queried using `StreamPriorityRange`.
// If the specified priority is outside the numerical range returned by `StreamPriorityRange`,
// it will automatically be clamped to the lowest or the highest number in the range.
func MakeStreamWithPriority(priority int, flags StreamFlags) (Stream, error) {
	var s C.CUstream
	if err := result(C.cuStreamCreateWithPriority(&s, C.uint(flags), C.int(priority))); err != nil {
		return 0, err
	}
	return makeStream(s), nil
}

// DestroyStream destroys the stream specified by hStream.
//
// In case the device is still doing work in the stream hStream when DestroyStrea() is called,
// the function will return immediately and the resources associated with hStream will be released automatically once the device has completed all work in hStream.
func DestroyStream(hStream *Stream) error {
	stream := *hStream
	s := stream.c()
	*hStream = 0
	return result(C.cuStreamDestroy(s))
}

func (ctx *Ctx) MakeStream(flags StreamFlags) (stream Stream, err error) {
	var s C.CUstream

	f := func() error { return result(C.cuStreamCreate(&s, C.uint(flags))) }
	if err = ctx.Do(f); err != nil {
		err = errors.Wrap(err, "MakeStream")
		return
	}
	stream = makeStream(s)
	return
}

func (ctx *Ctx) MakeStreamWithPriority(priority int, flags StreamFlags) (stream Stream, err error) {
	var s C.CUstream

	f := func() error { return result(C.cuStreamCreateWithPriority(&s, C.uint(flags), C.int(priority))) }
	if err = ctx.Do(f); err != nil {
		err = errors.Wrap(err, "MakeStream With Priority")
		return
	}
	stream = makeStream(s)
	return
}

func (ctx *Ctx) DestroyStream(hStream *Stream) {
	stream := *hStream
	s := stream.c()

	f := func() error { return result(C.cuStreamDestroy(s)) }
	ctx.err = ctx.Do(f)
}
