// +build !instrumentation

package cu

// addQueueLength adds the queue length to the operational statistics. NOOP.
func addQueueLength(l int) {}

// QueueLengths return the queue lengths recorded.
// Returns nil in non-instrumentation mode.
func QueueLengths() []int { return nil }

// AverageQueueLength returns the average queue length recorded. This allows for optimizations.
// Returns 0 in non-instrumentation mode.
func AverageQueueLength() int { return 0 }

// addBlockingCallers adds to the list of blocking callers for instrumentation.
// NOOP in non-instrumentation mode.
func addBlockingCallers() {}

// BlockingCallers returns the recorded list of blocking callers.
// Returns nil in non-instrumentation mode.
func BlockingCallers() map[string]int { return nil }

// QUEUE returns the queue of a *BatchedContext for introspection.
// Returns nil in non-instrumentation mode
func (ctx *BatchedContext) QUEUE() []call { return nil }

// Introspect returns the introspection code.
// Returns "" in non-instrumentation mode.
func (ctx *BatchedContext) Introspect() string { return "" }
