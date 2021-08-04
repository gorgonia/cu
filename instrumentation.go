// +build instrumentation

package cu

import (
	"log"
	"reflect"
	"runtime"
	"sync"
)

/* Operational statistics related debugging */

var ql = new(sync.Mutex)
var q = make([]int, 0, 1000) // 1000 calls to DoWork
var blockingCallers = make(map[string]int)

// addQueueLength adds the queue length to the operational statistics.
func addQueueLength(l int) {
	ql.Lock()
	q = append(q, l)
	ql.Unlock()
}

// QueueLengths return the queue lengths recorded
func QueueLengths() []int {
	return q
}

// AverageQueueLength returns the average queue length recorded. This allows for optimizations.
func AverageQueueLength() int {
	ql.Lock()
	var s int
	for _, l := range q {
		s += l
	}
	avg := s / len(q) // yes, it's an integer division
	ql.Unlock()
	return avg
}

// addBlockingCallers adds to the list of blocking callers for instrumentation.
func addBlockingCallers() {
	pc, _, _, _ := runtime.Caller(3)
	fn := runtime.FuncForPC(pc)
	ql.Lock()
	blockingCallers[fn.Name()]++
	ql.Unlock()
}

// BlockingCallers returns the recorded list of blocking callers.
func BlockingCallers() map[string]int {
	return blockingCallers
}

// QUEUE returns the queue of a *BatchedContext for introspection.
func (ctx *BatchedContext) QUEUE() []call {
	log.Println(len(ctx.queue))
	return ctx.queue
}

// Introspect returns the introspection code.
func (ctx *BatchedContext) Introspect() string {
	return ctx.introspect()
}
