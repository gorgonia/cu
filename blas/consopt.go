package cublas

import "gorgonia.org/cu"

type ConsOpt func(impl *Standard)

func WithContext(ctx cu.Context) ConsOpt {
	f := func(impl *Standard) {
		impl.Context = ctx
	}
	return f
}

func WithNativeData() ConsOpt {
	f := func(impl *Standard) {
		impl.dataOnDev = false
	}
	return f
}
