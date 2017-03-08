// +build !debug

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
