package cu

import (
	"github.com/pkg/errors"
	"time"
)

/* context.Context implementation */

// Deadline returns the dealine for this context. There is no deadline set.
func (ctx *Ctx) Deadline() (deadline time.Time, ok bool) { return time.Time{}, false }

// Done implements context.Context
func (ctx *Ctx) Done() <-chan struct{} { return ctx.done }

// Err returns an error.
// If Done is not yet closed, Err returns nil.
// If Done is closed, Err returns a non-nil error explaining why:
// Canceled if the context was canceled
// or DeadlineExceeded if the context's deadline passed.
// After Err returns a non-nil error, successive calls to Err return the same error.
func (ctx *Ctx) Err() error {
	ctx.mu.Lock()
	if ctx.doneClosed {
		if ctx.err != nil {
			ctx.mu.Unlock()
			return errors.Wrap(ctx.err, "Context Canceled")
		}
		ctx.mu.Unlock()
		return errors.New("Context Canceled")
	}
	ctx.mu.Unlock()
	return nil
}

// Value always returns nil.
func (ctx *Ctx) Value(key interface{}) interface{} { return nil }

// Cancel is a context.CancelFunc
func (ctx *Ctx) Cancel() {
	ctx.mu.Lock()
	if ctx.doneClosed {
		ctx.mu.Unlock()
		return
	}
	close(ctx.done)
	ctx.doneClosed = true
	ctx.mu.Unlock()
}
