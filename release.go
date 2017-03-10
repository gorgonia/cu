// +build !debug
// +build !instrumentation

package cu

const DEBUG = false

var tc uint32

// var _logger_ = log.New(os.Stderr, "", 0)
// var replacement = "\n"

func tabcount() int                             { return 0 }
func enterLoggingContext()                      {}
func leaveLoggingContext()                      {}
func logf(format string, others ...interface{}) {}

/* Debugging Utility Methods */

// introspect is useful for finding out what calls are going to be made in the batched call
func (ctx *BatchedContext) introspect() string { return "" }

// instrumentation related functions

func addQueueLength(l int) {}

// QueueLengths return the queue lengths recorded
func QueueLengths() []int { return nil }

// AverageQueueLength returns the average queue length recorded. This allows for optimizations.
func AverageQueueLength() int { return 0 }

func addBlockingCallers() {}

func BlockingCallers() map[string]int { return nil }
