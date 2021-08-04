// +build debug
// +build !linux

package cu

// logtid is noop in non-linux builds
func logtid(category string, logcaller int) {}
