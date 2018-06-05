package cudnn

func isScalar(a []int) bool {
	return len(a) == 0 || (len(a) == 1 && a[0] == 1)
}

// shapeEq is adapted from tensor
func shapeEq(a, b []int) bool {
	if isScalar(a) && isScalar(b) {
		return true
	}

	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if b[i] != v {
			return false
		}
	}
	return true
}

func cloneShape(a []int) []int {
	retVal := make([]int, len(a))
	copy(retVal, a)
	return retVal
}
