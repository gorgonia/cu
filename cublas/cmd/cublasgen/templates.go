package main

import "text/template"

const handwrittenRaw = `// Do not manually edit this file. It was created by the cublasgen program.
// The header file was generated from {{.}}.

// Copyright ©2017 Xuanyi Chew. Adapted from the cgo BLAS library by
// The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cublas

/*
#cgo CFLAGS: -g -O3
#include <cublas_v2.h>
*/
import "C"

import (
	"unsafe"

	"github.com/gonum/blas"
)


// Special cases...

type srotmParams struct {
	flag float32
	h    [4]float32
}

type drotmParams struct {
	flag float64
	h    [4]float64
}

func (impl *Implementation) Srotg(a float32, b float32) (c float32, s float32, r float32, z float32) {
	impl.e = status(C.cublasSrotg(C.cublasHandle_t(impl.h), (*C.float)(&a), (*C.float)(&b), (*C.float)(&c), (*C.float)(&s)))
	return c, s, a, b
}
func (impl *Implementation) Srotmg(d1 float32, d2 float32, b1 float32, b2 float32) (p blas.SrotmParams, rd1 float32, rd2 float32, rb1 float32) {
	if impl.e != nil {
			return
	}
	var pi srotmParams
	impl.e = status(C.cublasSrotmg(C.cublasHandle_t(impl.h), (*C.float)(&d1), (*C.float)(&d2), (*C.float)(&b1), (*C.float)(&b2), (*C.float)(unsafe.Pointer(&pi))))
	return blas.SrotmParams{Flag: blas.Flag(pi.flag), H: pi.h}, d1, d2, b1
}

func (impl *Implementation) Srotm(n int, x []float32, incX int, y []float32, incY int, p blas.SrotmParams) {
	if impl.e != nil {
			return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incY == 0 {
		panic("blas: zero y index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if p.Flag < blas.Identity || p.Flag > blas.Diagonal {
		panic("blas: illegal blas.Flag value")
	}
	if n == 0 {
		return
	}
	pi := srotmParams{
		flag: float32(p.Flag),
		h:    p.H,
	}
	impl.e = status(C.cublasSrotm(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(&x[0]), C.int(incX), (*C.float)(&y[0]), C.int(incY), (*C.float)(unsafe.Pointer(&pi))))
}

func (impl *Implementation) Drotg(a float64, b float64) (c float64, s float64, r float64, z float64) {
	if impl.e != nil {
			return
	}
	impl.e = status(C.cublasDrotg(C.cublasHandle_t(impl.h), (*C.double)(&a), (*C.double)(&b), (*C.double)(&c), (*C.double)(&s)))
	return c, s, a, b
}

func (impl *Implementation) Drotmg(d1 float64, d2 float64, b1 float64, b2 float64) (p blas.DrotmParams, rd1 float64, rd2 float64, rb1 float64) {
	if impl.e != nil {
			return
	}
	var pi drotmParams
	impl.e = status(C.cublasDrotmg(C.cublasHandle_t(impl.h), (*C.double)(&d1), (*C.double)(&d2), (*C.double)(&b1), (*C.double)(&b2), (*C.double)(unsafe.Pointer(&pi))))
	return blas.DrotmParams{Flag: blas.Flag(pi.flag), H: pi.h}, d1, d2, b1
}

func (impl *Implementation) Drotm(n int, x []float64, incX int, y []float64, incY int, p blas.DrotmParams) {
	if impl.e != nil {
			return
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incY == 0 {
		panic("blas: zero y index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if p.Flag < blas.Identity || p.Flag > blas.Diagonal {
		panic("blas: illegal blas.Flag value")
	}
	if n == 0 {
		return
	}
	pi := drotmParams{
		flag: float64(p.Flag),
		h:    p.H,
	}
	impl.e = status(C.cublasDrotm(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(&x[0]), C.int(incX), (*C.double)(&y[0]), C.int(incY), (*C.double)(unsafe.Pointer(&pi))))
}

func (impl *Implementation) Cdotu(n int, x []complex64, incX int, y []complex64, incY int) (dotu complex64) {
	if impl.e != nil {
			return
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incY == 0 {
		panic("blas: zero y index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if n == 0 {
		return 0
	}
	impl.e = status(C.cublasCdotu(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&y[0])), C.int(incY), (*C.cuComplex)(unsafe.Pointer(&dotu))))
	return dotu
}
func (impl *Implementation) Cdotc(n int, x []complex64, incX int, y []complex64, incY int) (dotc complex64) {
	if impl.e != nil {
			return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incY == 0 {
		panic("blas: zero y index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if n == 0 {
		return 0
	}
	impl.e = status(C.cublasCdotc(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&y[0])), C.int(incY), (*C.cuComplex)(unsafe.Pointer(&dotc))))
	return dotc
}
func (impl *Implementation) Zdotu(n int, x []complex128, incX int, y []complex128, incY int) (dotu complex128) {
	if impl.e != nil {
			return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incY == 0 {
		panic("blas: zero y index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if n == 0 {
		return 0
	}
	impl.e = status(C.cublasZdotu(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&y[0])), C.int(incY), (*C.cuDoubleComplex)(unsafe.Pointer(&dotu))))
	return dotu
}
func (impl *Implementation) Zdotc(n int, x []complex128, incX int, y []complex128, incY int) (dotc complex128) {
	if impl.e != nil {
			return
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incY == 0 {
		panic("blas: zero y index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if n == 0 {
		return 0
	}
	impl.e = status(C.cublasZdotc(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&y[0])), C.int(incY), (*C.cuDoubleComplex)(unsafe.Pointer(&dotc))))
	return dotc
}

func (impl *Implementation) Sdsdot(n int, alpha float32, x []float32, incX int, y []float32, incY int) float32 {
	panic("Unimplemented in cuBLAS. Please contact nvidia.")
}

func (impl *Implementation) Dsdot(n int, x []float32, incX int, y []float32, incY int) float64 {
	panic("Unimplemented in cuBLAS. Please contact nvidia.")
}

func (impl *Implementation) Strmm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int){
	panic("Unimplemented in cuBLAS. Please contact nvidia.")	
}

func (impl *Implementation) Dtrmm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int){
	panic("Unimplemented in cuBLAS. Please contact nvidia.")	
}

func (impl *Implementation) Ctrmm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha complex64, a []complex64, lda int, b []complex64, ldb int){
	panic("Unimplemented in cuBLAS. Please contact nvidia.")	
}

func (impl *Implementation) Ztrmm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int){
	panic("Unimplemented in cuBLAS. Please contact nvidia.")	
}

// Generated cases ...

`

// TODO: complex scale
const complexScales = `
func (impl *Implementation) Cscal(n int, alpha complex64, x []complex64, incX int) {}
func (impl *Implementation) Zscal(n int, alpha complex64, x []complex128, incX int){}
func (impl *Implementation) Csscal(n int, alpha float32, x []complex64, incX int) {}
func (impl *Implementation) Zsscal(n int, alpha float64, x []complex128, incX int){}
`

const amaxRaw = `

`

const batchedCHeaderRaw = `#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>

typedef enum {
	fn_undefined,

	{{range . -}}
	fn_{{.Name}}, 
	{{end }}
} cublasFn;
`

var (
	batchedCHeader *template.Template
	handwritten    *template.Template
)

func init() {
	batchedCHeader = template.Must(template.New("batchedCHeader").Parse(batchedCHeaderRaw))
	handwritten = template.Must(template.New("handwritten").Parse(handwrittenRaw))
}
