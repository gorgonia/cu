package cu

/*
void CallHostFunc(void* v){
	handleCUDACB(v);
};
*/
import "C"
import "unsafe"

type HostFunction struct{ ptr unsafe.Pointer }

type Callback struct {
	Func     func(...interface{})
	UserData []interface{}
}
