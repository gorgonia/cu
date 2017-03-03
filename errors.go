package cu

import (
	"bytes"
	"fmt"
)

// ErrorLister is the interface for a slice of error
type ErrorLister interface {
	ListErrors() []error
}

type errorSlice []error

func (err errorSlice) Error() string {
	var buf bytes.Buffer
	for i, v := range err {
		fmt.Fprintf(&buf, "[%d]: %v\n", i, v)
	}
	return buf.String()
}

func (err errorSlice) ListErrors() []error { return []error(err) }
