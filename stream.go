package cu

// #include <cuda.h>
import "C"
import "unsafe"

// Stream represents a CUDA stream.
type Stream uintptr

// MakeStream creates a stream. The flags determines the behaviors of the stream.
func MakeStream(flags StreamFlags) (Stream, error) {
	var s C.CUstream
	if err := result(C.cuStreamCreate(&s, C.uint(flags))); err != nil {
		return 0, err
	}
	return Stream(uintptr(unsafe.Pointer(s))), nil
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
	return Stream(uintptr(unsafe.Pointer(s))), nil
}

// DestroyStream destroys the stream specified by hStream.
//
// In case the device is still doing work in the stream hStream when DestroyStrea() is called,
// the function will return immediately and the resources associated with hStream will be released automatically once the device has completed all work in hStream.
func DestroyStream(hStream *Stream) error {
	stream := *hStream
	s := C.CUstream(unsafe.Pointer(uintptr(stream)))
	*hStream = 0
	return result(C.cuStreamDestroy(s))
}

//Flags queries the flags of a given stream.
func (s Stream) Flags() (StreamFlags, error) {
	var f C.uint
	cs := C.CUstream(unsafe.Pointer(uintptr(s)))
	if err := result(C.cuStreamGetFlags(cs, &f)); err != nil {
		return 0, err
	}
	return StreamFlags(f), nil
}

// Priority returns the priority of a stream created.
//
// Note that if the stream was created with a priority outside the numerical range returned by `StreamPriorityRange`,
// this function returns the clamped priority.
func (s Stream) Priority() (int, error) {
	var p C.int
	cs := C.CUstream(unsafe.Pointer(uintptr(s)))
	if err := result(C.cuStreamGetPriority(cs, &p)); err != nil {
		return -1, err
	}
	return int(p), nil
}

// Query checks the status of a compute stream. It returns nil if all operations have completed, error NotReady otherwise
func (s Stream) Query() error {
	cs := C.CUstream(unsafe.Pointer(uintptr(s)))
	return result(C.cuStreamQuery(cs))
}

// Synchronize waits until the device has completed all operations in the stream.
// If the context was created with the SchedBlockingSync flag, the CPU thread will block until the stream is finished with all of its tasks.
func (s Stream) Synchronize() error {
	cs := C.CUstream(unsafe.Pointer(uintptr(s)))
	return result(C.cuStreamSynchronize(cs))
}

/* TODO */

// func (s Stream) AddCallback()
// func (s Stream) AttachMemAsync()
// func (s Stream) WaitEvent()
