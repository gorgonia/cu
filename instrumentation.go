// +build instrumentation
// +build !debug

package cu

import (
	"bytes"
	"fmt"
	"log"
	"runtime"
	"sync"
)

const DEBUG = false

var tc uint32

// var _logger_ = log.New(os.Stderr, "", 0)
// var replacement = "\n"

func tabcount() int                             { return 0 }
func enterLoggingContext()                      {}
func leaveLoggingContext()                      {}
func logf(format string, others ...interface{}) {}

// introspect is useful for finding out what calls are going to be made in the batched call
func (ctx *BatchedContext) introspect() string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "Queue: %d", len(ctx.queue))
	for _, v := range ctx.queue {
		fmt.Fprintf(&buf, "\n\t[QUEUE] %s", v.fnargs)
	}
	return buf.String()
}

/* Operational statistics related debugging */

var ql = new(sync.Mutex)
var q = make([]int, 0, 1000) // 1000 calls to DoWork
var blockingCallers = make(map[string]int)

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

func addBlockingCallers() {
	pc, _, _, _ := runtime.Caller(3)
	fn := runtime.FuncForPC(pc)
	ql.Lock()
	blockingCallers[fn.Name()]++
	ql.Unlock()
}

func BlockingCallers() map[string]int {
	return blockingCallers
}

func (ctx *BatchedContext) QUEUE() []call {
	log.Println(len(ctx.queue))
	return ctx.queue
}

func (ctx *BatchedContext) Introspect() string {
	return ctx.introspect()
}
