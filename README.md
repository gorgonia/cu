# cu [![GoDoc](https://godoc.org/gorgonia.org/cu?status.svg)](https://godoc.org/gorgonia.org/cu)

Package `cu` is a package that interfaces with the [CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/). This package was directly inspired by Arne Vansteenkiste's [`cu` package](https://github.com/barnex/cuda5).

# Why Write This Package? #
The main reason why this package was written (as opposed to just using the already-excellent [`cu` package](https://github.com/barnex/cuda5)) was because of errors. Specifically, the main difference between this package and Arne's package is that this package returns errors instead of panicking.

Additionally another goal for this package is to have an idiomatic interface for CUDA. For example, instead of exposing `cuCtxCreate` to be `CtxCreate`, a nicer, more idiomatic name `MakeContext` is used. The primary goal is to make calling the CUDA API as comfortable as calling Go functions or methods. Additional convenience functions and methods are also created in this package in the pursuit of that goal.

Lastly, this package uses the latest [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) whereas the original package `cu` uses a number of deprecated APIs.

# Installation #

This package is go-gettable: `go get -u gorgonia.org/cu`

This package mostly depends on built-in packages. There are two external dependencies:

* [errors](https://github.com/pkg/errors), which is licenced under a [MIT-like](https://github.com/pkg/errors/blob/master/LICENSE) licence. This package is used for wrapping errors and providing a debug trail.
* [assert](https://github.com/stretchr/testify), which is licenced under a [MIT-like](https://github.com/stretchr/testify/blob/master/LICENSE) licence. This package is used for quick and easy testing.

However, package `cu` DOES depend on one major external dependency: CUDA. Specifically, it requires the CUDA driver. Thankfully nvidia has made this rather simple - everything that is required can be installed with one click: [CUDA Toolkit 8.0](https://developer.nvidia.com/cuda-toolkit)


To verify that this library works, install and run the `cudatest` program, which accompanies this package:

```
go install gorgonia.org/cu/cmd/cudatest
cudatest
```

You should see something like this if successful:

```
CUDA version: 8000
CUDA devices: 1
Device 0
========
Name      :	"GeForce GTX 960"
Clock Rate:	1177500 kHz
Memory    :	4233297920 bytes
Compute   : 5.2
```

## Windows ##

To setup CUDA in Windows:

1. Install CUDA Toolkit
2. Add `%CUDA_PATH%/bin` to your `%PATH%` environment variable (running `nvcc` from console should work)
3. Make a symlink `mklink /D C:\cuda "c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"` (alternatively, install CUDA toolkit to `C:\cuda\`)

To setup the compiler:

1. Install MSYS2 (see https://www.msys2.org/)
2. In `c:\msys64\msys2_shell.cmd` uncomment the line with `set MSYS2_PATH_TYPE=inherit` (this makes Windows PATH variable visible)
3. Install `go` in MSYS2 (64 bit) with `pacman -S go`

# Progress #
The work to fully represent the CUDA Driver API is a work in progress. At the moment, it is not complete. However, most of the API that are required for GPGPU purposes are complete. None of the texture, surface and graphics related APIs are handled yet. Please feel free to send a pull request.

## Roadmap ##

* [ ] Remaining API to be ported over
* [x] All texture, surface and graphics related API have an equivalent Go prototype.
* [x] Batching of common operations (see for example `Device.Attributes(...)`
* [x] Generic queueing/batching of API calls (by some definition of generic)


# Contributing #
This author loves pull requests from everyone. Here's how to contribute to this package:

1. Fork then clone this repo:
    `git clone git@github.com:YOUR_USERNAME/cu.git`
2. Work on your edits.
3. Commit with a good commit message.
4. Push to your fork then [submit a pull request](https://gorgonia.org/cu/compare/).

We understand that this package is an interfacing package with a third party API. As such, tests may not always be viable. However, please do try to include as much tests as possible.


# Licence #
The package is licenced with a MIT-like licence. Ther is one file (`cgoflags.go`) where code is directly copied  and two files (`execution.go` and `memory.go`) where code was partially copied from Arne Vansteenkiste's package, which is unlicenced (but to be safe, just assume a GPL-like licence, as [mumax/3](https://github.com/mumax/3) is licenced under GPL).