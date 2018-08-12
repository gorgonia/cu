package cudnn

import "C"
import (
	"fmt"
	"sync"
	"unsafe"
)

var cintsize, gointsize int

func init() {
	cintsize = int(C.sizeof_int)
	gointsize = int(unsafe.Sizeof(int(1)))
}

var int32sPool = &sync.Pool{
	New: func() interface{} { return make([]int32, 0, 8) },
}

func returnManaged(a interface{}) {
	if a == nil {
		return
	}

	switch x := a.(type) {
	case []int32:
		for i := range x {
			x[i] = 0
		}
		x = x[:0]
		int32sPool.Put(x)
	}
}

func ints2CIntPtr(a []int) (cPtr *C.int, managed interface{}) {
	if cintsize == gointsize {
		return (*C.int)(unsafe.Pointer(&a[0])), nil
	}
	switch {
	case cintsize == 4 && gointsize == 8:
		b := int32sPool.Get().([]int32)
		for _, v := range a {
			b = append(b, int32(v))
		}
		return (*C.int)(unsafe.Pointer(&b[0])), b
	default:
		panic(fmt.Sprintf("UNHANDLED: cintsize: %v gointsize: %v", cintsize, gointsize))
	}
}
