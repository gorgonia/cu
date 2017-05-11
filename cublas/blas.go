// Do not manually edit this file. It was created by the cublasgen program.
// The header file was generated from cublasgen.h.

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

func (impl *Implementation) Strmm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) {
	panic("Unimplemented in cuBLAS. Please contact nvidia.")
}

func (impl *Implementation) Dtrmm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
	panic("Unimplemented in cuBLAS. Please contact nvidia.")
}

func (impl *Implementation) Ctrmm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha complex64, a []complex64, lda int, b []complex64, ldb int) {
	panic("Unimplemented in cuBLAS. Please contact nvidia.")
}

func (impl *Implementation) Ztrmm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int) {
	panic("Unimplemented in cuBLAS. Please contact nvidia.")
}

// Generated cases ...

// Snrm2 computes the Euclidean norm of a vector,
//  sqrt(\sum_i x[i] * x[i]).
// This function returns 0 if incX is negative.
func (impl *Implementation) Snrm2(n int, x []float32, incX int) (retVal float32) {
	// declared at cublasgen.h:137:17 enum CUBLAS_STATUS { ... } cublasSnrm2 ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incX < 0 {
		return 0
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasSnrm2(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(&x[0]), C.int(incX), (*C.float)(&retVal)))
	return retVal
}

// Dnrm2 computes the Euclidean norm of a vector,
//  sqrt(\sum_i x[i] * x[i]).
// This function returns 0 if incX is negative.
func (impl *Implementation) Dnrm2(n int, x []float64, incX int) (retVal float64) {
	// declared at cublasgen.h:143:17 enum CUBLAS_STATUS { ... } cublasDnrm2 ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incX < 0 {
		return 0
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasDnrm2(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(&x[0]), C.int(incX), (*C.double)(&retVal)))
	return retVal
}

func (impl *Implementation) Scnrm2(n int, x []complex64, incX int) (retVal float32) {
	// declared at cublasgen.h:149:17 enum CUBLAS_STATUS { ... } cublasScnrm2 ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incX < 0 {
		return 0
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasScnrm2(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.float)(&retVal)))
	return retVal
}

func (impl *Implementation) Dznrm2(n int, x []complex128, incX int) (retVal float64) {
	// declared at cublasgen.h:155:17 enum CUBLAS_STATUS { ... } cublasDznrm2 ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incX < 0 {
		return 0
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasDznrm2(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.double)(&retVal)))
	return retVal
}

// Sdot computes the dot product of the two vectors
//  \sum_i x[i]*y[i]
func (impl *Implementation) Sdot(n int, x []float32, incX int, y []float32, incY int) (retVal float32) {
	// declared at cublasgen.h:186:17 enum CUBLAS_STATUS { ... } cublasSdot ...
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
		return
	}
	impl.e = status(C.cublasSdot(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(&x[0]), C.int(incX), (*C.float)(&y[0]), C.int(incY), (*C.float)(&retVal)))
	return retVal
}

// Ddot computes the dot product of the two vectors
//  \sum_i x[i]*y[i]
func (impl *Implementation) Ddot(n int, x []float64, incX int, y []float64, incY int) (retVal float64) {
	// declared at cublasgen.h:194:17 enum CUBLAS_STATUS { ... } cublasDdot ...
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
		return
	}
	impl.e = status(C.cublasDdot(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(&x[0]), C.int(incX), (*C.double)(&y[0]), C.int(incY), (*C.double)(&retVal)))
	return retVal
}

// Sscal scales x by alpha.
//  x[i] *= alpha
// Sscal has no effect if incX < 0.
func (impl *Implementation) Sscal(n int, alpha float32, x []float32, incX int) {
	// declared at cublasgen.h:245:17 enum CUBLAS_STATUS { ... } cublasSscal ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incX < 0 {
		return
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasSscal(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(&alpha), (*C.float)(&x[0]), C.int(incX)))
}

// Dscal scales x by alpha.
//  x[i] *= alpha
// Dscal has no effect if incX < 0.
func (impl *Implementation) Dscal(n int, alpha float64, x []float64, incX int) {
	// declared at cublasgen.h:251:17 enum CUBLAS_STATUS { ... } cublasDscal ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incX < 0 {
		return
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasDscal(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(&alpha), (*C.double)(&x[0]), C.int(incX)))
}

func (impl *Implementation) Cscal(n int, alpha complex64, x []complex64, incX int) {
	// declared at cublasgen.h:257:17 enum CUBLAS_STATUS { ... } cublasCscal ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incX < 0 {
		return
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasCscal(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX)))
}

func (impl *Implementation) Csscal(n int, alpha float32, x []complex64, incX int) {
	// declared at cublasgen.h:263:17 enum CUBLAS_STATUS { ... } cublasCsscal ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incX < 0 {
		return
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasCsscal(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(&alpha), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX)))
}

func (impl *Implementation) Zscal(n int, alpha complex128, x []complex128, incX int) {
	// declared at cublasgen.h:269:17 enum CUBLAS_STATUS { ... } cublasZscal ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incX < 0 {
		return
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasZscal(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX)))
}

func (impl *Implementation) Zdscal(n int, alpha float64, x []complex128, incX int) {
	// declared at cublasgen.h:275:17 enum CUBLAS_STATUS { ... } cublasZdscal ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasZdscal(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(&alpha), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX)))
}

// Saxpy adds alpha times x to y
//  y[i] += alpha * x[i] for all i
func (impl *Implementation) Saxpy(n int, alpha float32, x []float32, incX int, y []float32, incY int) {
	// declared at cublasgen.h:296:17 enum CUBLAS_STATUS { ... } cublasSaxpy ...
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
		return
	}
	impl.e = status(C.cublasSaxpy(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(&alpha), (*C.float)(&x[0]), C.int(incX), (*C.float)(&y[0]), C.int(incY)))
}

// Daxpy adds alpha times x to y
//  y[i] += alpha * x[i] for all i
func (impl *Implementation) Daxpy(n int, alpha float64, x []float64, incX int, y []float64, incY int) {
	// declared at cublasgen.h:304:17 enum CUBLAS_STATUS { ... } cublasDaxpy ...
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
		return
	}
	impl.e = status(C.cublasDaxpy(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(&alpha), (*C.double)(&x[0]), C.int(incX), (*C.double)(&y[0]), C.int(incY)))
}

func (impl *Implementation) Caxpy(n int, alpha complex64, x []complex64, incX int, y []complex64, incY int) {
	// declared at cublasgen.h:312:17 enum CUBLAS_STATUS { ... } cublasCaxpy ...
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
		return
	}
	impl.e = status(C.cublasCaxpy(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&y[0])), C.int(incY)))
}

func (impl *Implementation) Zaxpy(n int, alpha complex128, x []complex128, incX int, y []complex128, incY int) {
	// declared at cublasgen.h:320:17 enum CUBLAS_STATUS { ... } cublasZaxpy ...
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
		return
	}
	impl.e = status(C.cublasZaxpy(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&y[0])), C.int(incY)))
}

// Scopy copies the elements of x into the elements of y.
//  y[i] = x[i] for all i
func (impl *Implementation) Scopy(n int, x []float32, incX int, y []float32, incY int) {
	// declared at cublasgen.h:328:17 enum CUBLAS_STATUS { ... } cublasScopy ...
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
		return
	}
	impl.e = status(C.cublasScopy(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(&x[0]), C.int(incX), (*C.float)(&y[0]), C.int(incY)))
}

// Dcopy copies the elements of x into the elements of y.
//  y[i] = x[i] for all i
func (impl *Implementation) Dcopy(n int, x []float64, incX int, y []float64, incY int) {
	// declared at cublasgen.h:335:17 enum CUBLAS_STATUS { ... } cublasDcopy ...
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
		return
	}
	impl.e = status(C.cublasDcopy(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(&x[0]), C.int(incX), (*C.double)(&y[0]), C.int(incY)))
}

func (impl *Implementation) Ccopy(n int, x []complex64, incX int, y []complex64, incY int) {
	// declared at cublasgen.h:342:17 enum CUBLAS_STATUS { ... } cublasCcopy ...
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
		return
	}
	impl.e = status(C.cublasCcopy(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&y[0])), C.int(incY)))
}

func (impl *Implementation) Zcopy(n int, x []complex128, incX int, y []complex128, incY int) {
	// declared at cublasgen.h:349:17 enum CUBLAS_STATUS { ... } cublasZcopy ...
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
		return
	}
	impl.e = status(C.cublasZcopy(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&y[0])), C.int(incY)))
}

// Sswap exchanges the elements of two vectors.
//  x[i], y[i] = y[i], x[i] for all i
func (impl *Implementation) Sswap(n int, x []float32, incX int, y []float32, incY int) {
	// declared at cublasgen.h:356:17 enum CUBLAS_STATUS { ... } cublasSswap ...
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
		return
	}
	impl.e = status(C.cublasSswap(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(&x[0]), C.int(incX), (*C.float)(&y[0]), C.int(incY)))
}

// Dswap exchanges the elements of two vectors.
//  x[i], y[i] = y[i], x[i] for all i
func (impl *Implementation) Dswap(n int, x []float64, incX int, y []float64, incY int) {
	// declared at cublasgen.h:363:17 enum CUBLAS_STATUS { ... } cublasDswap ...
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
		return
	}
	impl.e = status(C.cublasDswap(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(&x[0]), C.int(incX), (*C.double)(&y[0]), C.int(incY)))
}

func (impl *Implementation) Cswap(n int, x []complex64, incX int, y []complex64, incY int) {
	// declared at cublasgen.h:370:17 enum CUBLAS_STATUS { ... } cublasCswap ...
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
		return
	}
	impl.e = status(C.cublasCswap(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&y[0])), C.int(incY)))
}

func (impl *Implementation) Zswap(n int, x []complex128, incX int, y []complex128, incY int) {
	// declared at cublasgen.h:377:17 enum CUBLAS_STATUS { ... } cublasZswap ...
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
		return
	}
	impl.e = status(C.cublasZswap(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&y[0])), C.int(incY)))
}

// Isamax returns the index of an element of x with the largest absolute value.
// If there are multiple such indices the earliest is returned.
// Isamax returns -1 if n == 0.
func (impl *Implementation) Isamax(n int, x []float32, incX int) (retVal int) {
	// declared at cublasgen.h:384:17 enum CUBLAS_STATUS { ... } cublasIsamax ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if n == 0 || incX < 0 {
		return -1
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	var ret C.int
	impl.e = status(C.cublasIsamax(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(&x[0]), C.int(incX), &ret))
	return int(ret)
}

// Idamax returns the index of an element of x with the largest absolute value.
// If there are multiple such indices the earliest is returned.
// Idamax returns -1 if n == 0.
func (impl *Implementation) Idamax(n int, x []float64, incX int) (retVal int) {
	// declared at cublasgen.h:390:17 enum CUBLAS_STATUS { ... } cublasIdamax ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if n == 0 || incX < 0 {
		return -1
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	var ret C.int
	impl.e = status(C.cublasIdamax(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(&x[0]), C.int(incX), &ret))
	return int(ret)
}

func (impl *Implementation) Icamax(n int, x []complex64, incX int) (retVal int) {
	// declared at cublasgen.h:396:17 enum CUBLAS_STATUS { ... } cublasIcamax ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if n == 0 || incX < 0 {
		return -1
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	var ret C.int
	impl.e = status(C.cublasIcamax(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), &ret))
	return int(ret)
}

func (impl *Implementation) Izamax(n int, x []complex128, incX int) (retVal int) {
	// declared at cublasgen.h:402:17 enum CUBLAS_STATUS { ... } cublasIzamax ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if n == 0 || incX < 0 {
		return -1
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	var ret C.int
	impl.e = status(C.cublasIzamax(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), &ret))
	return int(ret)
}

func (impl *Implementation) Isamin(n int, x []float32, incX int) (retVal int) {
	// declared at cublasgen.h:408:17 enum CUBLAS_STATUS { ... } cublasIsamin ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if n == 0 || incX < 0 {
		return -1
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	var ret C.int
	impl.e = status(C.cublasIsamin(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(&x[0]), C.int(incX), &ret))
	return int(ret)
}

func (impl *Implementation) Idamin(n int, x []float64, incX int) (retVal int) {
	// declared at cublasgen.h:414:17 enum CUBLAS_STATUS { ... } cublasIdamin ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if n == 0 || incX < 0 {
		return -1
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	var ret C.int
	impl.e = status(C.cublasIdamin(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(&x[0]), C.int(incX), &ret))
	return int(ret)
}

func (impl *Implementation) Icamin(n int, x []complex64, incX int) (retVal int) {
	// declared at cublasgen.h:420:17 enum CUBLAS_STATUS { ... } cublasIcamin ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if n == 0 || incX < 0 {
		return -1
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	var ret C.int
	impl.e = status(C.cublasIcamin(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), &ret))
	return int(ret)
}

func (impl *Implementation) Izamin(n int, x []complex128, incX int) (retVal int) {
	// declared at cublasgen.h:426:17 enum CUBLAS_STATUS { ... } cublasIzamin ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if n == 0 || incX < 0 {
		return -1
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	var ret C.int
	impl.e = status(C.cublasIzamin(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), &ret))
	return int(ret)
}

// Sasum computes the sum of the absolute values of the elements of x.
//  \sum_i |x[i]|
// Sasum returns 0 if incX is negative.
func (impl *Implementation) Sasum(n int, x []float32, incX int) (retVal float32) {
	// declared at cublasgen.h:432:17 enum CUBLAS_STATUS { ... } cublasSasum ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incX < 0 {
		return 0
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasSasum(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(&x[0]), C.int(incX), (*C.float)(&retVal)))
	return retVal
}

// Dasum computes the sum of the absolute values of the elements of x.
//  \sum_i |x[i]|
// Dasum returns 0 if incX is negative.
func (impl *Implementation) Dasum(n int, x []float64, incX int) (retVal float64) {
	// declared at cublasgen.h:438:17 enum CUBLAS_STATUS { ... } cublasDasum ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incX < 0 {
		return 0
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasDasum(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(&x[0]), C.int(incX), (*C.double)(&retVal)))
	return retVal
}

func (impl *Implementation) Scasum(n int, x []complex64, incX int) (retVal float32) {
	// declared at cublasgen.h:444:17 enum CUBLAS_STATUS { ... } cublasScasum ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incX < 0 {
		return 0
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasScasum(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.float)(&retVal)))
	return retVal
}

func (impl *Implementation) Dzasum(n int, x []complex128, incX int) (retVal float64) {
	// declared at cublasgen.h:450:17 enum CUBLAS_STATUS { ... } cublasDzasum ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if incX < 0 {
		return 0
	}
	if incX > 0 && (n-1)*incX >= len(x) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasDzasum(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.double)(&retVal)))
	return retVal
}

// Srot applies a plane transformation.
//  x[i] = c * x[i] + s * y[i]
//  y[i] = c * y[i] - s * x[i]
func (impl *Implementation) Srot(n int, x []float32, incX int, y []float32, incY int, cScalar, sScalar float32) {
	// declared at cublasgen.h:456:17 enum CUBLAS_STATUS { ... } cublasSrot ...
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
		return
	}
	impl.e = status(C.cublasSrot(C.cublasHandle_t(impl.h), C.int(n), (*C.float)(&x[0]), C.int(incX), (*C.float)(&y[0]), C.int(incY), (*C.float)(&cScalar), (*C.float)(&sScalar)))
}

// Drot applies a plane transformation.
//  x[i] = c * x[i] + s * y[i]
//  y[i] = c * y[i] - s * x[i]
func (impl *Implementation) Drot(n int, x []float64, incX int, y []float64, incY int, cScalar, sScalar float64) {
	// declared at cublasgen.h:465:17 enum CUBLAS_STATUS { ... } cublasDrot ...
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
		return
	}
	impl.e = status(C.cublasDrot(C.cublasHandle_t(impl.h), C.int(n), (*C.double)(&x[0]), C.int(incX), (*C.double)(&y[0]), C.int(incY), (*C.double)(&cScalar), (*C.double)(&sScalar)))
}

func (impl *Implementation) Crot(n int, x []complex64, incX int, y []complex64, incY int, cScalar float32, sScalar []complex64) {
	// declared at cublasgen.h:474:17 enum CUBLAS_STATUS { ... } cublasCrot ...
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
		return
	}
	impl.e = status(C.cublasCrot(C.cublasHandle_t(impl.h), C.int(n), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&y[0])), C.int(incY), (*C.float)(&cScalar), (*C.cuComplex)(unsafe.Pointer(&sScalar))))
}

func (impl *Implementation) Zrot(n int, x []complex128, incX int, y []complex128, incY int, cScalar float64, sScalar complex128) {
	// declared at cublasgen.h:492:17 enum CUBLAS_STATUS { ... } cublasZrot ...
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
		return
	}
	impl.e = status(C.cublasZrot(C.cublasHandle_t(impl.h), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&y[0])), C.int(incY), (*C.double)(&cScalar), (*C.cuDoubleComplex)(unsafe.Pointer(&sScalar))))
}

// Sgemv computes
//  y = alpha * a * x + beta * y if tA = blas.NoTrans
//  y = alpha * A^T * x + beta * y if tA = blas.Trans or blas.ConjTrans
// where A is an m×n dense matrix, x and y are vectors, and alpha is a scalar.
func (impl *Implementation) Sgemv(tA blas.Transpose, m, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) {
	// declared at cublasgen.h:567:17 enum CUBLAS_STATUS { ... } cublasSgemv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
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
	var lenX, lenY int
	if tA == blas.NoTrans {
		lenX, lenY = n, m
	} else {
		lenX, lenY = m, n
	}
	if (incX > 0 && (lenX-1)*incX >= len(x)) || (incX < 0 && (1-lenX)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (lenY-1)*incY >= len(y)) || (incY < 0 && (1-lenY)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasSgemv(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), C.int(m), C.int(n), (*C.float)(&alpha), (*C.float)(&a[0]), C.int(lda), (*C.float)(&x[0]), C.int(incX), (*C.float)(&beta), (*C.float)(&y[0]), C.int(incY)))
}

// Dgemv computes
//  y = alpha * a * x + beta * y if tA = blas.NoTrans
//  y = alpha * A^T * x + beta * y if tA = blas.Trans or blas.ConjTrans
// where A is an m×n dense matrix, x and y are vectors, and alpha is a scalar.
func (impl *Implementation) Dgemv(tA blas.Transpose, m, n int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) {
	// declared at cublasgen.h:580:17 enum CUBLAS_STATUS { ... } cublasDgemv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
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
	var lenX, lenY int
	if tA == blas.NoTrans {
		lenX, lenY = n, m
	} else {
		lenX, lenY = m, n
	}
	if (incX > 0 && (lenX-1)*incX >= len(x)) || (incX < 0 && (1-lenX)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (lenY-1)*incY >= len(y)) || (incY < 0 && (1-lenY)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasDgemv(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), C.int(m), C.int(n), (*C.double)(&alpha), (*C.double)(&a[0]), C.int(lda), (*C.double)(&x[0]), C.int(incX), (*C.double)(&beta), (*C.double)(&y[0]), C.int(incY)))
}

func (impl *Implementation) Cgemv(tA blas.Transpose, m, n int, alpha complex64, a []complex64, lda int, x []complex64, incX int, beta complex64, y []complex64, incY int) {
	// declared at cublasgen.h:593:17 enum CUBLAS_STATUS { ... } cublasCgemv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
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
	var lenX, lenY int
	if tA == blas.NoTrans {
		lenX, lenY = n, m
	} else {
		lenX, lenY = m, n
	}
	if (incX > 0 && (lenX-1)*incX >= len(x)) || (incX < 0 && (1-lenX)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (lenY-1)*incY >= len(y)) || (incY < 0 && (1-lenY)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasCgemv(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), C.int(m), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(unsafe.Pointer(&y[0])), C.int(incY)))
}

func (impl *Implementation) Zgemv(tA blas.Transpose, m, n int, alpha complex128, a []complex128, lda int, x []complex128, incX int, beta complex128, y []complex128, incY int) {
	// declared at cublasgen.h:606:17 enum CUBLAS_STATUS { ... } cublasZgemv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
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
	var lenX, lenY int
	if tA == blas.NoTrans {
		lenX, lenY = n, m
	} else {
		lenX, lenY = m, n
	}
	if (incX > 0 && (lenX-1)*incX >= len(x)) || (incX < 0 && (1-lenX)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (lenY-1)*incY >= len(y)) || (incY < 0 && (1-lenY)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZgemv(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), C.int(m), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(unsafe.Pointer(&y[0])), C.int(incY)))
}

// Sgbmv computes
//  y = alpha * A * x + beta * y if tA == blas.NoTrans
//  y = alpha * A^T * x + beta * y if tA == blas.Trans or blas.ConjTrans
// where a is an m×n band matrix kL subdiagonals and kU super-diagonals, and
// m and n refer to the size of the full dense matrix it represents.
// x and y are vectors, and alpha and beta are scalars.
func (impl *Implementation) Sgbmv(tA blas.Transpose, m, n, kl, ku int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) {
	// declared at cublasgen.h:619:17 enum CUBLAS_STATUS { ... } cublasSgbmv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
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
	var lenX, lenY int
	if tA == blas.NoTrans {
		lenX, lenY = n, m
	} else {
		lenX, lenY = m, n
	}
	if (incX > 0 && (lenX-1)*incX >= len(x)) || (incX < 0 && (1-lenX)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (lenY-1)*incY >= len(y)) || (incY < 0 && (1-lenY)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasSgbmv(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), C.int(m), C.int(n), C.int(kl), C.int(ku), (*C.float)(&alpha), (*C.float)(&a[0]), C.int(lda), (*C.float)(&x[0]), C.int(incX), (*C.float)(&beta), (*C.float)(&y[0]), C.int(incY)))
}

// Dgbmv computes
//  y = alpha * A * x + beta * y if tA == blas.NoTrans
//  y = alpha * A^T * x + beta * y if tA == blas.Trans or blas.ConjTrans
// where a is an m×n band matrix kL subdiagonals and kU super-diagonals, and
// m and n refer to the size of the full dense matrix it represents.
// x and y are vectors, and alpha and beta are scalars.
func (impl *Implementation) Dgbmv(tA blas.Transpose, m, n, kl, ku int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) {
	// declared at cublasgen.h:634:17 enum CUBLAS_STATUS { ... } cublasDgbmv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
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
	var lenX, lenY int
	if tA == blas.NoTrans {
		lenX, lenY = n, m
	} else {
		lenX, lenY = m, n
	}
	if (incX > 0 && (lenX-1)*incX >= len(x)) || (incX < 0 && (1-lenX)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (lenY-1)*incY >= len(y)) || (incY < 0 && (1-lenY)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasDgbmv(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), C.int(m), C.int(n), C.int(kl), C.int(ku), (*C.double)(&alpha), (*C.double)(&a[0]), C.int(lda), (*C.double)(&x[0]), C.int(incX), (*C.double)(&beta), (*C.double)(&y[0]), C.int(incY)))
}

func (impl *Implementation) Cgbmv(tA blas.Transpose, m, n, kl, ku int, alpha complex64, a []complex64, lda int, x []complex64, incX int, beta complex64, y []complex64, incY int) {
	// declared at cublasgen.h:649:17 enum CUBLAS_STATUS { ... } cublasCgbmv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
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
	var lenX, lenY int
	if tA == blas.NoTrans {
		lenX, lenY = n, m
	} else {
		lenX, lenY = m, n
	}
	if (incX > 0 && (lenX-1)*incX >= len(x)) || (incX < 0 && (1-lenX)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (lenY-1)*incY >= len(y)) || (incY < 0 && (1-lenY)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasCgbmv(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), C.int(m), C.int(n), C.int(kl), C.int(ku), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(unsafe.Pointer(&y[0])), C.int(incY)))
}

func (impl *Implementation) Zgbmv(tA blas.Transpose, m, n, kl, ku int, alpha complex128, a []complex128, lda int, x []complex128, incX int, beta complex128, y []complex128, incY int) {
	// declared at cublasgen.h:664:17 enum CUBLAS_STATUS { ... } cublasZgbmv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
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
	var lenX, lenY int
	if tA == blas.NoTrans {
		lenX, lenY = n, m
	} else {
		lenX, lenY = m, n
	}
	if (incX > 0 && (lenX-1)*incX >= len(x)) || (incX < 0 && (1-lenX)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (lenY-1)*incY >= len(y)) || (incY < 0 && (1-lenY)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZgbmv(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), C.int(m), C.int(n), C.int(kl), C.int(ku), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(unsafe.Pointer(&y[0])), C.int(incY)))
}

// Strmv computes
//  x = A * x if tA == blas.NoTrans
//  x = A^T * x if tA == blas.Trans or blas.ConjTrans
// A is an n×n Triangular matrix and x is a vector.
func (impl *Implementation) Strmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, a []float32, lda int, x []float32, incX int) {
	// declared at cublasgen.h:680:17 enum CUBLAS_STATUS { ... } cublasStrmv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasStrmv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), (*C.float)(&a[0]), C.int(lda), (*C.float)(&x[0]), C.int(incX)))
}

// Dtrmv computes
//  x = A * x if tA == blas.NoTrans
//  x = A^T * x if tA == blas.Trans or blas.ConjTrans
// A is an n×n Triangular matrix and x is a vector.
func (impl *Implementation) Dtrmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, a []float64, lda int, x []float64, incX int) {
	// declared at cublasgen.h:690:17 enum CUBLAS_STATUS { ... } cublasDtrmv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasDtrmv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), (*C.double)(&a[0]), C.int(lda), (*C.double)(&x[0]), C.int(incX)))
}

func (impl *Implementation) Ctrmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, a []complex64, lda int, x []complex64, incX int) {
	// declared at cublasgen.h:700:17 enum CUBLAS_STATUS { ... } cublasCtrmv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasCtrmv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX)))
}

func (impl *Implementation) Ztrmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, a []complex128, lda int, x []complex128, incX int) {
	// declared at cublasgen.h:710:17 enum CUBLAS_STATUS { ... } cublasZtrmv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZtrmv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX)))
}

// Stbmv computes
//  x = A * x if tA == blas.NoTrans
//  x = A^T * x if tA == blas.Trans or blas.ConjTrans
// where A is an n×n triangular banded matrix with k diagonals, and x is a vector.
func (impl *Implementation) Stbmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n, k int, a []float32, lda int, x []float32, incX int) {
	// declared at cublasgen.h:721:17 enum CUBLAS_STATUS { ... } cublasStbmv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+k+1 > len(a) || lda < k+1 {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasStbmv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), C.int(k), (*C.float)(&a[0]), C.int(lda), (*C.float)(&x[0]), C.int(incX)))
}

// Dtbmv computes
//  x = A * x if tA == blas.NoTrans
//  x = A^T * x if tA == blas.Trans or blas.ConjTrans
// where A is an n×n triangular banded matrix with k diagonals, and x is a vector.
func (impl *Implementation) Dtbmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n, k int, a []float64, lda int, x []float64, incX int) {
	// declared at cublasgen.h:732:17 enum CUBLAS_STATUS { ... } cublasDtbmv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+k+1 > len(a) || lda < k+1 {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasDtbmv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), C.int(k), (*C.double)(&a[0]), C.int(lda), (*C.double)(&x[0]), C.int(incX)))
}

func (impl *Implementation) Ctbmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n, k int, a []complex64, lda int, x []complex64, incX int) {
	// declared at cublasgen.h:743:17 enum CUBLAS_STATUS { ... } cublasCtbmv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+k+1 > len(a) || lda < k+1 {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasCtbmv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), C.int(k), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX)))
}

func (impl *Implementation) Ztbmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n, k int, a []complex128, lda int, x []complex128, incX int) {
	// declared at cublasgen.h:754:17 enum CUBLAS_STATUS { ... } cublasZtbmv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+k+1 > len(a) || lda < k+1 {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZtbmv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), C.int(k), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX)))
}

// Stpmv computes
//  x = A * x if tA == blas.NoTrans
//  x = A^T * x if tA == blas.Trans or blas.ConjTrans
// where A is an n×n unit triangular matrix in packed format, and x is a vector.
func (impl *Implementation) Stpmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, aP, x []float32, incX int) {
	// declared at cublasgen.h:766:17 enum CUBLAS_STATUS { ... } cublasStpmv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasStpmv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), (*C.float)(&aP[0]), (*C.float)(&x[0]), C.int(incX)))
}

// Dtpmv computes
//  x = A * x if tA == blas.NoTrans
//  x = A^T * x if tA == blas.Trans or blas.ConjTrans
// where A is an n×n unit triangular matrix in packed format, and x is a vector.
func (impl *Implementation) Dtpmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, aP, x []float64, incX int) {
	// declared at cublasgen.h:775:17 enum CUBLAS_STATUS { ... } cublasDtpmv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasDtpmv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), (*C.double)(&aP[0]), (*C.double)(&x[0]), C.int(incX)))
}

func (impl *Implementation) Ctpmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, aP, x []complex64, incX int) {
	// declared at cublasgen.h:784:17 enum CUBLAS_STATUS { ... } cublasCtpmv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasCtpmv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), (*C.cuComplex)(unsafe.Pointer(&aP[0])), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX)))
}

func (impl *Implementation) Ztpmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, aP, x []complex128, incX int) {
	// declared at cublasgen.h:793:17 enum CUBLAS_STATUS { ... } cublasZtpmv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasZtpmv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&aP[0])), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX)))
}

// Strsv solves
//  A * x = b if tA == blas.NoTrans
//  A^T * x = b if tA == blas.Trans or blas.ConjTrans
// A is an n×n triangular matrix and x is a vector.
// At entry to the function, x contains the values of b, and the result is
// stored in place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func (impl *Implementation) Strsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, a []float32, lda int, x []float32, incX int) {
	// declared at cublasgen.h:803:17 enum CUBLAS_STATUS { ... } cublasStrsv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasStrsv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), (*C.float)(&a[0]), C.int(lda), (*C.float)(&x[0]), C.int(incX)))
}

// Dtrsv solves
//  A * x = b if tA == blas.NoTrans
//  A^T * x = b if tA == blas.Trans or blas.ConjTrans
// A is an n×n triangular matrix and x is a vector.
// At entry to the function, x contains the values of b, and the result is
// stored in place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func (impl *Implementation) Dtrsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, a []float64, lda int, x []float64, incX int) {
	// declared at cublasgen.h:813:17 enum CUBLAS_STATUS { ... } cublasDtrsv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasDtrsv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), (*C.double)(&a[0]), C.int(lda), (*C.double)(&x[0]), C.int(incX)))
}

func (impl *Implementation) Ctrsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, a []complex64, lda int, x []complex64, incX int) {
	// declared at cublasgen.h:823:17 enum CUBLAS_STATUS { ... } cublasCtrsv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasCtrsv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX)))
}

func (impl *Implementation) Ztrsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, a []complex128, lda int, x []complex128, incX int) {
	// declared at cublasgen.h:833:17 enum CUBLAS_STATUS { ... } cublasZtrsv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZtrsv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX)))
}

// Stpsv solves
//  A * x = b if tA == blas.NoTrans
//  A^T * x = b if tA == blas.Trans or blas.ConjTrans
// where A is an n×n triangular matrix in packed format and x is a vector.
// At entry to the function, x contains the values of b, and the result is
// stored in place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func (impl *Implementation) Stpsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, aP, x []float32, incX int) {
	// declared at cublasgen.h:844:17 enum CUBLAS_STATUS { ... } cublasStpsv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasStpsv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), (*C.float)(&aP[0]), (*C.float)(&x[0]), C.int(incX)))
}

// Dtpsv solves
//  A * x = b if tA == blas.NoTrans
//  A^T * x = b if tA == blas.Trans or blas.ConjTrans
// where A is an n×n triangular matrix in packed format and x is a vector.
// At entry to the function, x contains the values of b, and the result is
// stored in place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func (impl *Implementation) Dtpsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, aP, x []float64, incX int) {
	// declared at cublasgen.h:853:17 enum CUBLAS_STATUS { ... } cublasDtpsv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasDtpsv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), (*C.double)(&aP[0]), (*C.double)(&x[0]), C.int(incX)))
}

func (impl *Implementation) Ctpsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, aP, x []complex64, incX int) {
	// declared at cublasgen.h:862:17 enum CUBLAS_STATUS { ... } cublasCtpsv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasCtpsv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), (*C.cuComplex)(unsafe.Pointer(&aP[0])), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX)))
}

func (impl *Implementation) Ztpsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, aP, x []complex128, incX int) {
	// declared at cublasgen.h:871:17 enum CUBLAS_STATUS { ... } cublasZtpsv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasZtpsv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&aP[0])), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX)))
}

// Stbsv solves
//  A * x = b
// where A is an n×n triangular banded matrix with k diagonals in packed format,
// and x is a vector.
// At entry to the function, x contains the values of b, and the result is
// stored in place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func (impl *Implementation) Stbsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n, k int, a []float32, lda int, x []float32, incX int) {
	// declared at cublasgen.h:880:17 enum CUBLAS_STATUS { ... } cublasStbsv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+k+1 > len(a) || lda < k+1 {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasStbsv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), C.int(k), (*C.float)(&a[0]), C.int(lda), (*C.float)(&x[0]), C.int(incX)))
}

// Dtbsv solves
//  A * x = b
// where A is an n×n triangular banded matrix with k diagonals in packed format,
// and x is a vector.
// At entry to the function, x contains the values of b, and the result is
// stored in place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func (impl *Implementation) Dtbsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n, k int, a []float64, lda int, x []float64, incX int) {
	// declared at cublasgen.h:891:17 enum CUBLAS_STATUS { ... } cublasDtbsv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+k+1 > len(a) || lda < k+1 {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasDtbsv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), C.int(k), (*C.double)(&a[0]), C.int(lda), (*C.double)(&x[0]), C.int(incX)))
}

func (impl *Implementation) Ctbsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n, k int, a []complex64, lda int, x []complex64, incX int) {
	// declared at cublasgen.h:902:17 enum CUBLAS_STATUS { ... } cublasCtbsv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+k+1 > len(a) || lda < k+1 {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasCtbsv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), C.int(k), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX)))
}

func (impl *Implementation) Ztbsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n, k int, a []complex128, lda int, x []complex128, incX int) {
	// declared at cublasgen.h:913:17 enum CUBLAS_STATUS { ... } cublasZtbsv ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+k+1 > len(a) || lda < k+1 {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZtbsv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(n), C.int(k), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX)))
}

// Ssymv computes
//    y = alpha * A * x + beta * y,
// where a is an n×n symmetric matrix, x and y are vectors, and alpha and
// beta are scalars.
func (impl *Implementation) Ssymv(ul blas.Uplo, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) {
	// declared at cublasgen.h:925:17 enum CUBLAS_STATUS { ... } cublasSsymv ...
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
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasSsymv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.float)(&alpha), (*C.float)(&a[0]), C.int(lda), (*C.float)(&x[0]), C.int(incX), (*C.float)(&beta), (*C.float)(&y[0]), C.int(incY)))
}

// Dsymv computes
//    y = alpha * A * x + beta * y,
// where a is an n×n symmetric matrix, x and y are vectors, and alpha and
// beta are scalars.
func (impl *Implementation) Dsymv(ul blas.Uplo, n int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) {
	// declared at cublasgen.h:937:17 enum CUBLAS_STATUS { ... } cublasDsymv ...
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
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasDsymv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.double)(&alpha), (*C.double)(&a[0]), C.int(lda), (*C.double)(&x[0]), C.int(incX), (*C.double)(&beta), (*C.double)(&y[0]), C.int(incY)))
}

func (impl *Implementation) Csymv(ul blas.Uplo, n int, alpha complex64, a []complex64, lda int, x []complex64, incX int, beta complex64, y []complex64, incY int) {
	// declared at cublasgen.h:949:17 enum CUBLAS_STATUS { ... } cublasCsymv ...
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
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasCsymv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(unsafe.Pointer(&y[0])), C.int(incY)))
}

func (impl *Implementation) Zsymv(ul blas.Uplo, n int, alpha complex128, a []complex128, lda int, x []complex128, incX int, beta complex128, y []complex128, incY int) {
	// declared at cublasgen.h:961:17 enum CUBLAS_STATUS { ... } cublasZsymv ...
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
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZsymv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(unsafe.Pointer(&y[0])), C.int(incY)))
}

func (impl *Implementation) Chemv(ul blas.Uplo, n int, alpha complex64, a []complex64, lda int, x []complex64, incX int, beta complex64, y []complex64, incY int) {
	// declared at cublasgen.h:973:17 enum CUBLAS_STATUS { ... } cublasChemv ...
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
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasChemv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(unsafe.Pointer(&y[0])), C.int(incY)))
}

func (impl *Implementation) Zhemv(ul blas.Uplo, n int, alpha complex128, a []complex128, lda int, x []complex128, incX int, beta complex128, y []complex128, incY int) {
	// declared at cublasgen.h:985:17 enum CUBLAS_STATUS { ... } cublasZhemv ...
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
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZhemv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(unsafe.Pointer(&y[0])), C.int(incY)))
}

// Ssbmv performs
//  y = alpha * A * x + beta * y
// where A is an n×n symmetric banded matrix, x and y are vectors, and alpha
// and beta are scalars.
func (impl *Implementation) Ssbmv(ul blas.Uplo, n, k int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) {
	// declared at cublasgen.h:998:17 enum CUBLAS_STATUS { ... } cublasSsbmv ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
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
	if lda*(n-1)+k+1 > len(a) || lda < k+1 {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasSsbmv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), C.int(k), (*C.float)(&alpha), (*C.float)(&a[0]), C.int(lda), (*C.float)(&x[0]), C.int(incX), (*C.float)(&beta), (*C.float)(&y[0]), C.int(incY)))
}

// Dsbmv performs
//  y = alpha * A * x + beta * y
// where A is an n×n symmetric banded matrix, x and y are vectors, and alpha
// and beta are scalars.
func (impl *Implementation) Dsbmv(ul blas.Uplo, n, k int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) {
	// declared at cublasgen.h:1011:17 enum CUBLAS_STATUS { ... } cublasDsbmv ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
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
	if lda*(n-1)+k+1 > len(a) || lda < k+1 {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasDsbmv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), C.int(k), (*C.double)(&alpha), (*C.double)(&a[0]), C.int(lda), (*C.double)(&x[0]), C.int(incX), (*C.double)(&beta), (*C.double)(&y[0]), C.int(incY)))
}

func (impl *Implementation) Chbmv(ul blas.Uplo, n, k int, alpha complex64, a []complex64, lda int, x []complex64, incX int, beta complex64, y []complex64, incY int) {
	// declared at cublasgen.h:1024:17 enum CUBLAS_STATUS { ... } cublasChbmv ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
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
	if lda*(n-1)+k+1 > len(a) || lda < k+1 {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasChbmv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), C.int(k), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(unsafe.Pointer(&y[0])), C.int(incY)))
}

func (impl *Implementation) Zhbmv(ul blas.Uplo, n, k int, alpha complex128, a []complex128, lda int, x []complex128, incX int, beta complex128, y []complex128, incY int) {
	// declared at cublasgen.h:1037:17 enum CUBLAS_STATUS { ... } cublasZhbmv ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
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
	if lda*(n-1)+k+1 > len(a) || lda < k+1 {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZhbmv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), C.int(k), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(unsafe.Pointer(&y[0])), C.int(incY)))
}

// Sspmv performs
//    y = alpha * A * x + beta * y,
// where A is an n×n symmetric matrix in packed format, x and y are vectors
// and alpha and beta are scalars.
func (impl *Implementation) Sspmv(ul blas.Uplo, n int, alpha float32, aP, x []float32, incX int, beta float32, y []float32, incY int) {
	// declared at cublasgen.h:1051:17 enum CUBLAS_STATUS { ... } cublasSspmv ...
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
		return
	}
	impl.e = status(C.cublasSspmv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.float)(&alpha), (*C.float)(&aP[0]), (*C.float)(&x[0]), C.int(incX), (*C.float)(&beta), (*C.float)(&y[0]), C.int(incY)))
}

// Dspmv performs
//    y = alpha * A * x + beta * y,
// where A is an n×n symmetric matrix in packed format, x and y are vectors
// and alpha and beta are scalars.
func (impl *Implementation) Dspmv(ul blas.Uplo, n int, alpha float64, aP, x []float64, incX int, beta float64, y []float64, incY int) {
	// declared at cublasgen.h:1062:17 enum CUBLAS_STATUS { ... } cublasDspmv ...
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
		return
	}
	impl.e = status(C.cublasDspmv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.double)(&alpha), (*C.double)(&aP[0]), (*C.double)(&x[0]), C.int(incX), (*C.double)(&beta), (*C.double)(&y[0]), C.int(incY)))
}

func (impl *Implementation) Chpmv(ul blas.Uplo, n int, alpha complex64, aP, x []complex64, incX int, beta complex64, y []complex64, incY int) {
	// declared at cublasgen.h:1073:17 enum CUBLAS_STATUS { ... } cublasChpmv ...
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
		return
	}
	impl.e = status(C.cublasChpmv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&aP[0])), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(unsafe.Pointer(&y[0])), C.int(incY)))
}

func (impl *Implementation) Zhpmv(ul blas.Uplo, n int, alpha complex128, aP, x []complex128, incX int, beta complex128, y []complex128, incY int) {
	// declared at cublasgen.h:1084:17 enum CUBLAS_STATUS { ... } cublasZhpmv ...
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
		return
	}
	impl.e = status(C.cublasZhpmv(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&aP[0])), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(unsafe.Pointer(&y[0])), C.int(incY)))
}

// Sger performs the rank-one operation
//  A += alpha * x * y^T
// where A is an m×n dense matrix, x and y are vectors, and alpha is a scalar.
func (impl *Implementation) Sger(m, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32, lda int) {
	// declared at cublasgen.h:1096:17 enum CUBLAS_STATUS { ... } cublasSger ...
	if impl.e != nil {
		return
	}

	if m < 0 {
		panic("blas: m < 0")
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
	if (incX > 0 && (m-1)*incX >= len(x)) || (incX < 0 && (1-m)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasSger(C.cublasHandle_t(impl.h), C.int(m), C.int(n), (*C.float)(&alpha), (*C.float)(&x[0]), C.int(incX), (*C.float)(&y[0]), C.int(incY), (*C.float)(&a[0]), C.int(lda)))
}

// Dger performs the rank-one operation
//  A += alpha * x * y^T
// where A is an m×n dense matrix, x and y are vectors, and alpha is a scalar.
func (impl *Implementation) Dger(m, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64, lda int) {
	// declared at cublasgen.h:1107:17 enum CUBLAS_STATUS { ... } cublasDger ...
	if impl.e != nil {
		return
	}

	if m < 0 {
		panic("blas: m < 0")
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
	if (incX > 0 && (m-1)*incX >= len(x)) || (incX < 0 && (1-m)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasDger(C.cublasHandle_t(impl.h), C.int(m), C.int(n), (*C.double)(&alpha), (*C.double)(&x[0]), C.int(incX), (*C.double)(&y[0]), C.int(incY), (*C.double)(&a[0]), C.int(lda)))
}

func (impl *Implementation) Cgeru(m, n int, alpha complex64, x []complex64, incX int, y []complex64, incY int, a []complex64, lda int) {
	// declared at cublasgen.h:1118:17 enum CUBLAS_STATUS { ... } cublasCgeru ...
	if impl.e != nil {
		return
	}

	if m < 0 {
		panic("blas: m < 0")
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
	if (incX > 0 && (m-1)*incX >= len(x)) || (incX < 0 && (1-m)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasCgeru(C.cublasHandle_t(impl.h), C.int(m), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&y[0])), C.int(incY), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda)))
}

func (impl *Implementation) Cgerc(m, n int, alpha complex64, x []complex64, incX int, y []complex64, incY int, a []complex64, lda int) {
	// declared at cublasgen.h:1129:17 enum CUBLAS_STATUS { ... } cublasCgerc ...
	if impl.e != nil {
		return
	}

	if m < 0 {
		panic("blas: m < 0")
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
	if (incX > 0 && (m-1)*incX >= len(x)) || (incX < 0 && (1-m)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasCgerc(C.cublasHandle_t(impl.h), C.int(m), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&y[0])), C.int(incY), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda)))
}

func (impl *Implementation) Zgeru(m, n int, alpha complex128, x []complex128, incX int, y []complex128, incY int, a []complex128, lda int) {
	// declared at cublasgen.h:1140:17 enum CUBLAS_STATUS { ... } cublasZgeru ...
	if impl.e != nil {
		return
	}

	if m < 0 {
		panic("blas: m < 0")
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
	if (incX > 0 && (m-1)*incX >= len(x)) || (incX < 0 && (1-m)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZgeru(C.cublasHandle_t(impl.h), C.int(m), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&y[0])), C.int(incY), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda)))
}

func (impl *Implementation) Zgerc(m, n int, alpha complex128, x []complex128, incX int, y []complex128, incY int, a []complex128, lda int) {
	// declared at cublasgen.h:1151:17 enum CUBLAS_STATUS { ... } cublasZgerc ...
	if impl.e != nil {
		return
	}

	if m < 0 {
		panic("blas: m < 0")
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
	if (incX > 0 && (m-1)*incX >= len(x)) || (incX < 0 && (1-m)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if (incY > 0 && (n-1)*incY >= len(y)) || (incY < 0 && (1-n)*incY >= len(y)) {
		panic("blas: y index out of range")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZgerc(C.cublasHandle_t(impl.h), C.int(m), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&y[0])), C.int(incY), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda)))
}

// Ssyr performs the rank-one update
//  a += alpha * x * x^T
// where a is an n×n symmetric matrix, and x is a vector.
func (impl *Implementation) Ssyr(ul blas.Uplo, n int, alpha float32, x []float32, incX int, a []float32, lda int) {
	// declared at cublasgen.h:1163:17 enum CUBLAS_STATUS { ... } cublasSsyr ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasSsyr(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.float)(&alpha), (*C.float)(&x[0]), C.int(incX), (*C.float)(&a[0]), C.int(lda)))
}

// Dsyr performs the rank-one update
//  a += alpha * x * x^T
// where a is an n×n symmetric matrix, and x is a vector.
func (impl *Implementation) Dsyr(ul blas.Uplo, n int, alpha float64, x []float64, incX int, a []float64, lda int) {
	// declared at cublasgen.h:1172:17 enum CUBLAS_STATUS { ... } cublasDsyr ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasDsyr(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.double)(&alpha), (*C.double)(&x[0]), C.int(incX), (*C.double)(&a[0]), C.int(lda)))
}

func (impl *Implementation) Csyr(ul blas.Uplo, n int, alpha complex64, x []complex64, incX int, a []complex64, lda int) {
	// declared at cublasgen.h:1181:17 enum CUBLAS_STATUS { ... } cublasCsyr ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasCsyr(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda)))
}

func (impl *Implementation) Zsyr(ul blas.Uplo, n int, alpha complex128, x []complex128, incX int, a []complex128, lda int) {
	// declared at cublasgen.h:1190:17 enum CUBLAS_STATUS { ... } cublasZsyr ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZsyr(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda)))
}

func (impl *Implementation) Cher(ul blas.Uplo, n int, alpha float32, x []complex64, incX int, a []complex64, lda int) {
	// declared at cublasgen.h:1199:17 enum CUBLAS_STATUS { ... } cublasCher ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasCher(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.float)(&alpha), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda)))
}

func (impl *Implementation) Zher(ul blas.Uplo, n int, alpha float64, x []complex128, incX int, a []complex128, lda int) {
	// declared at cublasgen.h:1208:17 enum CUBLAS_STATUS { ... } cublasZher ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZher(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.double)(&alpha), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda)))
}

// Sspr computes the rank-one operation
//  a += alpha * x * x^T
// where a is an n×n symmetric matrix in packed format, x is a vector, and
// alpha is a scalar.
func (impl *Implementation) Sspr(ul blas.Uplo, n int, alpha float32, x []float32, incX int, aP []float32) {
	// declared at cublasgen.h:1218:17 enum CUBLAS_STATUS { ... } cublasSspr ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasSspr(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.float)(&alpha), (*C.float)(&x[0]), C.int(incX), (*C.float)(&aP[0])))
}

// Dspr computes the rank-one operation
//  a += alpha * x * x^T
// where a is an n×n symmetric matrix in packed format, x is a vector, and
// alpha is a scalar.
func (impl *Implementation) Dspr(ul blas.Uplo, n int, alpha float64, x []float64, incX int, aP []float64) {
	// declared at cublasgen.h:1226:17 enum CUBLAS_STATUS { ... } cublasDspr ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasDspr(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.double)(&alpha), (*C.double)(&x[0]), C.int(incX), (*C.double)(&aP[0])))
}

func (impl *Implementation) Chpr(ul blas.Uplo, n int, alpha float32, x []complex64, incX int, aP []complex64) {
	// declared at cublasgen.h:1234:17 enum CUBLAS_STATUS { ... } cublasChpr ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasChpr(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.float)(&alpha), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&aP[0]))))
}

func (impl *Implementation) Zhpr(ul blas.Uplo, n int, alpha float64, x []complex128, incX int, aP []complex128) {
	// declared at cublasgen.h:1242:17 enum CUBLAS_STATUS { ... } cublasZhpr ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (n-1)*incX >= len(x)) || (incX < 0 && (1-n)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if n == 0 {
		return
	}
	impl.e = status(C.cublasZhpr(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.double)(&alpha), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&aP[0]))))
}

// Ssyr2 performs the symmetric rank-two update
//  A += alpha * x * y^T + alpha * y * x^T
// where A is a symmetric n×n matrix, x and y are vectors, and alpha is a scalar.
func (impl *Implementation) Ssyr2(ul blas.Uplo, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32, lda int) {
	// declared at cublasgen.h:1251:17 enum CUBLAS_STATUS { ... } cublasSsyr2 ...
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
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasSsyr2(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.float)(&alpha), (*C.float)(&x[0]), C.int(incX), (*C.float)(&y[0]), C.int(incY), (*C.float)(&a[0]), C.int(lda)))
}

// Dsyr2 performs the symmetric rank-two update
//  A += alpha * x * y^T + alpha * y * x^T
// where A is a symmetric n×n matrix, x and y are vectors, and alpha is a scalar.
func (impl *Implementation) Dsyr2(ul blas.Uplo, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64, lda int) {
	// declared at cublasgen.h:1262:17 enum CUBLAS_STATUS { ... } cublasDsyr2 ...
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
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasDsyr2(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.double)(&alpha), (*C.double)(&x[0]), C.int(incX), (*C.double)(&y[0]), C.int(incY), (*C.double)(&a[0]), C.int(lda)))
}

func (impl *Implementation) Csyr2(ul blas.Uplo, n int, alpha complex64, x []complex64, incX int, y []complex64, incY int, a []complex64, lda int) {
	// declared at cublasgen.h:1273:17 enum CUBLAS_STATUS { ... } cublasCsyr2 ...
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
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasCsyr2(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&y[0])), C.int(incY), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda)))
}

func (impl *Implementation) Zsyr2(ul blas.Uplo, n int, alpha complex128, x []complex128, incX int, y []complex128, incY int, a []complex128, lda int) {
	// declared at cublasgen.h:1283:17 enum CUBLAS_STATUS { ... } cublasZsyr2 ...
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
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZsyr2(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&y[0])), C.int(incY), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda)))
}

func (impl *Implementation) Cher2(ul blas.Uplo, n int, alpha complex64, x []complex64, incX int, y []complex64, incY int, a []complex64, lda int) {
	// declared at cublasgen.h:1295:17 enum CUBLAS_STATUS { ... } cublasCher2 ...
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
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasCher2(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&y[0])), C.int(incY), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda)))
}

func (impl *Implementation) Zher2(ul blas.Uplo, n int, alpha complex128, x []complex128, incX int, y []complex128, incY int, a []complex128, lda int) {
	// declared at cublasgen.h:1305:17 enum CUBLAS_STATUS { ... } cublasZher2 ...
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
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZher2(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&y[0])), C.int(incY), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda)))
}

// Sspr2 performs the symmetric rank-2 update
//  A += alpha * x * y^T + alpha * y * x^T,
// where A is an n×n symmetric matrix in packed format, x and y are vectors,
// and alpha is a scalar.
func (impl *Implementation) Sspr2(ul blas.Uplo, n int, alpha float32, x []float32, incX int, y []float32, incY int, aP []float32) {
	// declared at cublasgen.h:1317:17 enum CUBLAS_STATUS { ... } cublasSspr2 ...
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
		return
	}
	impl.e = status(C.cublasSspr2(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.float)(&alpha), (*C.float)(&x[0]), C.int(incX), (*C.float)(&y[0]), C.int(incY), (*C.float)(&aP[0])))
}

// Dspr2 performs the symmetric rank-2 update
//  A += alpha * x * y^T + alpha * y * x^T,
// where A is an n×n symmetric matrix in packed format, x and y are vectors,
// and alpha is a scalar.
func (impl *Implementation) Dspr2(ul blas.Uplo, n int, alpha float64, x []float64, incX int, y []float64, incY int, aP []float64) {
	// declared at cublasgen.h:1327:17 enum CUBLAS_STATUS { ... } cublasDspr2 ...
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
		return
	}
	impl.e = status(C.cublasDspr2(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.double)(&alpha), (*C.double)(&x[0]), C.int(incX), (*C.double)(&y[0]), C.int(incY), (*C.double)(&aP[0])))
}

func (impl *Implementation) Chpr2(ul blas.Uplo, n int, alpha complex64, x []complex64, incX int, y []complex64, incY int, aP []complex64) {
	// declared at cublasgen.h:1338:17 enum CUBLAS_STATUS { ... } cublasChpr2 ...
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
		return
	}
	impl.e = status(C.cublasChpr2(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&y[0])), C.int(incY), (*C.cuComplex)(unsafe.Pointer(&aP[0]))))
}

func (impl *Implementation) Zhpr2(ul blas.Uplo, n int, alpha complex128, x []complex128, incX int, y []complex128, incY int, aP []complex128) {
	// declared at cublasgen.h:1348:17 enum CUBLAS_STATUS { ... } cublasZhpr2 ...
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
		return
	}
	impl.e = status(C.cublasZhpr2(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&y[0])), C.int(incY), (*C.cuDoubleComplex)(unsafe.Pointer(&aP[0]))))
}

// Sgemm computes
//  C = beta * C + alpha * A * B,
// where A, B, and C are dense matrices, and alpha and beta are scalars.
// tA and tB specify whether A or B are transposed.
func (impl *Implementation) Sgemm(tA, tB blas.Transpose, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	// declared at cublasgen.h:1361:17 enum CUBLAS_STATUS { ... } cublasSgemm ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if tB != blas.NoTrans && tB != blas.Trans && tB != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	var rowA, colA, rowB, colB int
	if tA == blas.NoTrans {
		rowA, colA = m, k
	} else {
		rowA, colA = k, m
	}
	if tB == blas.NoTrans {
		rowB, colB = k, n
	} else {
		rowB, colB = n, k
	}
	if lda*(rowA-1)+colA > len(a) || lda < max(1, colA) {
		panic("blas: index of a out of range")
	}
	if ldb*(rowB-1)+colB > len(b) || ldb < max(1, colB) {
		panic("blas: index of b out of range")
	}
	if ldc*(m-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasSgemm(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), trans2cublasTrans(tB), C.int(m), C.int(n), C.int(k), (*C.float)(&alpha), (*C.float)(&a[0]), C.int(lda), (*C.float)(&b[0]), C.int(ldb), (*C.float)(&beta), (*C.float)(&c[0]), C.int(ldc)))
}

// Dgemm computes
//  C = beta * C + alpha * A * B,
// where A, B, and C are dense matrices, and alpha and beta are scalars.
// tA and tB specify whether A or B are transposed.
func (impl *Implementation) Dgemm(tA, tB blas.Transpose, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	// declared at cublasgen.h:1376:17 enum CUBLAS_STATUS { ... } cublasDgemm ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if tB != blas.NoTrans && tB != blas.Trans && tB != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	var rowA, colA, rowB, colB int
	if tA == blas.NoTrans {
		rowA, colA = m, k
	} else {
		rowA, colA = k, m
	}
	if tB == blas.NoTrans {
		rowB, colB = k, n
	} else {
		rowB, colB = n, k
	}
	if lda*(rowA-1)+colA > len(a) || lda < max(1, colA) {
		panic("blas: index of a out of range")
	}
	if ldb*(rowB-1)+colB > len(b) || ldb < max(1, colB) {
		panic("blas: index of b out of range")
	}
	if ldc*(m-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasDgemm(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), trans2cublasTrans(tB), C.int(m), C.int(n), C.int(k), (*C.double)(&alpha), (*C.double)(&a[0]), C.int(lda), (*C.double)(&b[0]), C.int(ldb), (*C.double)(&beta), (*C.double)(&c[0]), C.int(ldc)))
}

func (impl *Implementation) Cgemm(tA, tB blas.Transpose, m, n, k int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta complex64, c []complex64, ldc int) {
	// declared at cublasgen.h:1391:17 enum CUBLAS_STATUS { ... } cublasCgemm ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if tB != blas.NoTrans && tB != blas.Trans && tB != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	var rowA, colA, rowB, colB int
	if tA == blas.NoTrans {
		rowA, colA = m, k
	} else {
		rowA, colA = k, m
	}
	if tB == blas.NoTrans {
		rowB, colB = k, n
	} else {
		rowB, colB = n, k
	}
	if lda*(rowA-1)+colA > len(a) || lda < max(1, colA) {
		panic("blas: index of a out of range")
	}
	if ldb*(rowB-1)+colB > len(b) || ldb < max(1, colB) {
		panic("blas: index of b out of range")
	}
	if ldc*(m-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasCgemm(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), trans2cublasTrans(tB), C.int(m), C.int(n), C.int(k), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&b[0])), C.int(ldb), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

func (impl *Implementation) Cgemm3m(tA, tB blas.Transpose, m, n, k int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta complex64, c []complex64, ldc int) {
	// declared at cublasgen.h:1406:17 enum CUBLAS_STATUS { ... } cublasCgemm3m ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if tB != blas.NoTrans && tB != blas.Trans && tB != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasCgemm3m(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), trans2cublasTrans(tB), C.int(m), C.int(n), C.int(k), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&b[0])), C.int(ldb), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

func (impl *Implementation) Zgemm(tA, tB blas.Transpose, m, n, k int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int) {
	// declared at cublasgen.h:1437:17 enum CUBLAS_STATUS { ... } cublasZgemm ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if tB != blas.NoTrans && tB != blas.Trans && tB != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	var rowA, colA, rowB, colB int
	if tA == blas.NoTrans {
		rowA, colA = m, k
	} else {
		rowA, colA = k, m
	}
	if tB == blas.NoTrans {
		rowB, colB = k, n
	} else {
		rowB, colB = n, k
	}
	if lda*(rowA-1)+colA > len(a) || lda < max(1, colA) {
		panic("blas: index of a out of range")
	}
	if ldb*(rowB-1)+colB > len(b) || ldb < max(1, colB) {
		panic("blas: index of b out of range")
	}
	if ldc*(m-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasZgemm(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), trans2cublasTrans(tB), C.int(m), C.int(n), C.int(k), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&b[0])), C.int(ldb), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

func (impl *Implementation) Zgemm3m(tA, tB blas.Transpose, m, n, k int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int) {
	// declared at cublasgen.h:1452:17 enum CUBLAS_STATUS { ... } cublasZgemm3m ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if tB != blas.NoTrans && tB != blas.Trans && tB != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZgemm3m(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), trans2cublasTrans(tB), C.int(m), C.int(n), C.int(k), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&b[0])), C.int(ldb), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

// Ssyrk performs the symmetric rank-k operation
//  C = alpha * A * A^T + beta*C
// C is an n×n symmetric matrix. A is an n×k matrix if tA == blas.NoTrans, and
// a k×n matrix otherwise. alpha and beta are scalars.
func (impl *Implementation) Ssyrk(ul blas.Uplo, t blas.Transpose, n, k int, alpha float32, a []float32, lda int, beta float32, c []float32, ldc int) {
	// declared at cublasgen.h:1548:17 enum CUBLAS_STATUS { ... } cublasSsyrk ...
	if impl.e != nil {
		return
	}

	if t != blas.NoTrans && t != blas.Trans && t != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	var row, col int
	if t == blas.NoTrans {
		row, col = n, k
	} else {
		row, col = k, n
	}
	if lda*(row-1)+col > len(a) || lda < max(1, col) {
		panic("blas: index of a out of range")
	}
	if ldc*(n-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasSsyrk(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(t), C.int(n), C.int(k), (*C.float)(&alpha), (*C.float)(&a[0]), C.int(lda), (*C.float)(&beta), (*C.float)(&c[0]), C.int(ldc)))
}

// Dsyrk performs the symmetric rank-k operation
//  C = alpha * A * A^T + beta*C
// C is an n×n symmetric matrix. A is an n×k matrix if tA == blas.NoTrans, and
// a k×n matrix otherwise. alpha and beta are scalars.
func (impl *Implementation) Dsyrk(ul blas.Uplo, t blas.Transpose, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int) {
	// declared at cublasgen.h:1560:17 enum CUBLAS_STATUS { ... } cublasDsyrk ...
	if impl.e != nil {
		return
	}

	if t != blas.NoTrans && t != blas.Trans && t != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	var row, col int
	if t == blas.NoTrans {
		row, col = n, k
	} else {
		row, col = k, n
	}
	if lda*(row-1)+col > len(a) || lda < max(1, col) {
		panic("blas: index of a out of range")
	}
	if ldc*(n-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasDsyrk(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(t), C.int(n), C.int(k), (*C.double)(&alpha), (*C.double)(&a[0]), C.int(lda), (*C.double)(&beta), (*C.double)(&c[0]), C.int(ldc)))
}

func (impl *Implementation) Csyrk(ul blas.Uplo, t blas.Transpose, n, k int, alpha complex64, a []complex64, lda int, beta complex64, c []complex64, ldc int) {
	// declared at cublasgen.h:1572:17 enum CUBLAS_STATUS { ... } cublasCsyrk ...
	if impl.e != nil {
		return
	}

	if t != blas.NoTrans && t != blas.Trans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	var row, col int
	if t == blas.NoTrans {
		row, col = n, k
	} else {
		row, col = k, n
	}
	if lda*(row-1)+col > len(a) || lda < max(1, col) {
		panic("blas: index of a out of range")
	}
	if ldc*(n-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasCsyrk(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(t), C.int(n), C.int(k), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

func (impl *Implementation) Zsyrk(ul blas.Uplo, t blas.Transpose, n, k int, alpha complex128, a []complex128, lda int, beta complex128, c []complex128, ldc int) {
	// declared at cublasgen.h:1584:17 enum CUBLAS_STATUS { ... } cublasZsyrk ...
	if impl.e != nil {
		return
	}

	if t != blas.NoTrans && t != blas.Trans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	var row, col int
	if t == blas.NoTrans {
		row, col = n, k
	} else {
		row, col = k, n
	}
	if lda*(row-1)+col > len(a) || lda < max(1, col) {
		panic("blas: index of a out of range")
	}
	if ldc*(n-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasZsyrk(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(t), C.int(n), C.int(k), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

func (impl *Implementation) Cherk(ul blas.Uplo, t blas.Transpose, n, k int, alpha float32, a []complex64, lda int, beta float32, c []complex64, ldc int) {
	// declared at cublasgen.h:1626:17 enum CUBLAS_STATUS { ... } cublasCherk ...
	if impl.e != nil {
		return
	}

	if t != blas.NoTrans && t != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	var row, col int
	if t == blas.NoTrans {
		row, col = n, k
	} else {
		row, col = k, n
	}
	if lda*(row-1)+col > len(a) || lda < max(1, col) {
		panic("blas: index of a out of range")
	}
	if ldc*(n-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasCherk(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(t), C.int(n), C.int(k), (*C.float)(&alpha), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.float)(&beta), (*C.cuComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

func (impl *Implementation) Zherk(ul blas.Uplo, t blas.Transpose, n, k int, alpha float64, a []complex128, lda int, beta float64, c []complex128, ldc int) {
	// declared at cublasgen.h:1638:17 enum CUBLAS_STATUS { ... } cublasZherk ...
	if impl.e != nil {
		return
	}

	if t != blas.NoTrans && t != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	var row, col int
	if t == blas.NoTrans {
		row, col = n, k
	} else {
		row, col = k, n
	}
	if lda*(row-1)+col > len(a) || lda < max(1, col) {
		panic("blas: index of a out of range")
	}
	if ldc*(n-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasZherk(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(t), C.int(n), C.int(k), (*C.double)(&alpha), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.double)(&beta), (*C.cuDoubleComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

// Ssyr2k performs the symmetric rank 2k operation
//  C = alpha * A * B^T + alpha * B * A^T + beta * C
// where C is an n×n symmetric matrix. A and B are n×k matrices if
// tA == NoTrans and k×n otherwise. alpha and beta are scalars.
func (impl *Implementation) Ssyr2k(ul blas.Uplo, t blas.Transpose, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	// declared at cublasgen.h:1682:17 enum CUBLAS_STATUS { ... } cublasSsyr2k ...
	if impl.e != nil {
		return
	}

	if t != blas.NoTrans && t != blas.Trans && t != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	var row, col int
	if t == blas.NoTrans {
		row, col = n, k
	} else {
		row, col = k, n
	}
	if lda*(row-1)+col > len(a) || lda < max(1, col) {
		panic("blas: index of a out of range")
	}
	if ldb*(row-1)+col > len(b) || ldb < max(1, col) {
		panic("blas: index of b out of range")
	}
	if ldc*(n-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasSsyr2k(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(t), C.int(n), C.int(k), (*C.float)(&alpha), (*C.float)(&a[0]), C.int(lda), (*C.float)(&b[0]), C.int(ldb), (*C.float)(&beta), (*C.float)(&c[0]), C.int(ldc)))
}

// Dsyr2k performs the symmetric rank 2k operation
//  C = alpha * A * B^T + alpha * B * A^T + beta * C
// where C is an n×n symmetric matrix. A and B are n×k matrices if
// tA == NoTrans and k×n otherwise. alpha and beta are scalars.
func (impl *Implementation) Dsyr2k(ul blas.Uplo, t blas.Transpose, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	// declared at cublasgen.h:1696:17 enum CUBLAS_STATUS { ... } cublasDsyr2k ...
	if impl.e != nil {
		return
	}

	if t != blas.NoTrans && t != blas.Trans && t != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	var row, col int
	if t == blas.NoTrans {
		row, col = n, k
	} else {
		row, col = k, n
	}
	if lda*(row-1)+col > len(a) || lda < max(1, col) {
		panic("blas: index of a out of range")
	}
	if ldb*(row-1)+col > len(b) || ldb < max(1, col) {
		panic("blas: index of b out of range")
	}
	if ldc*(n-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasDsyr2k(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(t), C.int(n), C.int(k), (*C.double)(&alpha), (*C.double)(&a[0]), C.int(lda), (*C.double)(&b[0]), C.int(ldb), (*C.double)(&beta), (*C.double)(&c[0]), C.int(ldc)))
}

func (impl *Implementation) Csyr2k(ul blas.Uplo, t blas.Transpose, n, k int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta complex64, c []complex64, ldc int) {
	// declared at cublasgen.h:1710:17 enum CUBLAS_STATUS { ... } cublasCsyr2k ...
	if impl.e != nil {
		return
	}

	if t != blas.NoTrans && t != blas.Trans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	var row, col int
	if t == blas.NoTrans {
		row, col = n, k
	} else {
		row, col = k, n
	}
	if lda*(row-1)+col > len(a) || lda < max(1, col) {
		panic("blas: index of a out of range")
	}
	if ldb*(row-1)+col > len(b) || ldb < max(1, col) {
		panic("blas: index of b out of range")
	}
	if ldc*(n-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasCsyr2k(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(t), C.int(n), C.int(k), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&b[0])), C.int(ldb), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

func (impl *Implementation) Zsyr2k(ul blas.Uplo, t blas.Transpose, n, k int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int) {
	// declared at cublasgen.h:1724:17 enum CUBLAS_STATUS { ... } cublasZsyr2k ...
	if impl.e != nil {
		return
	}

	if t != blas.NoTrans && t != blas.Trans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	var row, col int
	if t == blas.NoTrans {
		row, col = n, k
	} else {
		row, col = k, n
	}
	if lda*(row-1)+col > len(a) || lda < max(1, col) {
		panic("blas: index of a out of range")
	}
	if ldb*(row-1)+col > len(b) || ldb < max(1, col) {
		panic("blas: index of b out of range")
	}
	if ldc*(n-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasZsyr2k(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(t), C.int(n), C.int(k), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&b[0])), C.int(ldb), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

func (impl *Implementation) Cher2k(ul blas.Uplo, t blas.Transpose, n, k int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta float32, c []complex64, ldc int) {
	// declared at cublasgen.h:1738:17 enum CUBLAS_STATUS { ... } cublasCher2k ...
	if impl.e != nil {
		return
	}

	if t != blas.NoTrans && t != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	var row, col int
	if t == blas.NoTrans {
		row, col = n, k
	} else {
		row, col = k, n
	}
	if lda*(row-1)+col > len(a) || lda < max(1, col) {
		panic("blas: index of a out of range")
	}
	if ldb*(row-1)+col > len(b) || ldb < max(1, col) {
		panic("blas: index of b out of range")
	}
	if ldc*(n-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasCher2k(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(t), C.int(n), C.int(k), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&b[0])), C.int(ldb), (*C.float)(&beta), (*C.cuComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

func (impl *Implementation) Zher2k(ul blas.Uplo, t blas.Transpose, n, k int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta float64, c []complex128, ldc int) {
	// declared at cublasgen.h:1752:17 enum CUBLAS_STATUS { ... } cublasZher2k ...
	if impl.e != nil {
		return
	}

	if t != blas.NoTrans && t != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	var row, col int
	if t == blas.NoTrans {
		row, col = n, k
	} else {
		row, col = k, n
	}
	if lda*(row-1)+col > len(a) || lda < max(1, col) {
		panic("blas: index of a out of range")
	}
	if ldb*(row-1)+col > len(b) || ldb < max(1, col) {
		panic("blas: index of b out of range")
	}
	if ldc*(n-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasZher2k(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(t), C.int(n), C.int(k), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&b[0])), C.int(ldb), (*C.double)(&beta), (*C.cuDoubleComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

func (impl *Implementation) Ssyrkx(ul blas.Uplo, t blas.Transpose, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	// declared at cublasgen.h:1766:17 enum CUBLAS_STATUS { ... } cublasSsyrkx ...
	if impl.e != nil {
		return
	}

	if t != blas.NoTrans && t != blas.Trans && t != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	if lda*(n-1)+k+1 > len(a) || lda < k+1 {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasSsyrkx(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(t), C.int(n), C.int(k), (*C.float)(&alpha), (*C.float)(&a[0]), C.int(lda), (*C.float)(&b[0]), C.int(ldb), (*C.float)(&beta), (*C.float)(&c[0]), C.int(ldc)))
}

func (impl *Implementation) Dsyrkx(ul blas.Uplo, t blas.Transpose, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	// declared at cublasgen.h:1780:17 enum CUBLAS_STATUS { ... } cublasDsyrkx ...
	if impl.e != nil {
		return
	}

	if t != blas.NoTrans && t != blas.Trans && t != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	if lda*(n-1)+k+1 > len(a) || lda < k+1 {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasDsyrkx(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(t), C.int(n), C.int(k), (*C.double)(&alpha), (*C.double)(&a[0]), C.int(lda), (*C.double)(&b[0]), C.int(ldb), (*C.double)(&beta), (*C.double)(&c[0]), C.int(ldc)))
}

func (impl *Implementation) Csyrkx(ul blas.Uplo, t blas.Transpose, n, k int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta complex64, c []complex64, ldc int) {
	// declared at cublasgen.h:1794:17 enum CUBLAS_STATUS { ... } cublasCsyrkx ...
	if impl.e != nil {
		return
	}

	if t != blas.NoTrans && t != blas.Trans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	if lda*(n-1)+k+1 > len(a) || lda < k+1 {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasCsyrkx(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(t), C.int(n), C.int(k), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&b[0])), C.int(ldb), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

func (impl *Implementation) Zsyrkx(ul blas.Uplo, t blas.Transpose, n, k int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int) {
	// declared at cublasgen.h:1808:17 enum CUBLAS_STATUS { ... } cublasZsyrkx ...
	if impl.e != nil {
		return
	}

	if t != blas.NoTrans && t != blas.Trans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	if lda*(n-1)+k+1 > len(a) || lda < k+1 {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZsyrkx(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(t), C.int(n), C.int(k), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&b[0])), C.int(ldb), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

func (impl *Implementation) Cherkx(ul blas.Uplo, t blas.Transpose, n, k int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta float32, c []complex64, ldc int) {
	// declared at cublasgen.h:1822:17 enum CUBLAS_STATUS { ... } cublasCherkx ...
	if impl.e != nil {
		return
	}

	if t != blas.NoTrans && t != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	if lda*(n-1)+k+1 > len(a) || lda < k+1 {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasCherkx(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(t), C.int(n), C.int(k), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&b[0])), C.int(ldb), (*C.float)(&beta), (*C.cuComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

func (impl *Implementation) Zherkx(ul blas.Uplo, t blas.Transpose, n, k int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta float64, c []complex128, ldc int) {
	// declared at cublasgen.h:1836:17 enum CUBLAS_STATUS { ... } cublasZherkx ...
	if impl.e != nil {
		return
	}

	if t != blas.NoTrans && t != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if k < 0 {
		panic("blas: k < 0")
	}
	if lda*(n-1)+k+1 > len(a) || lda < k+1 {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZherkx(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), trans2cublasTrans(t), C.int(n), C.int(k), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&b[0])), C.int(ldb), (*C.double)(&beta), (*C.cuDoubleComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

// Ssymm performs one of
//  C = alpha * A * B + beta * C, if side == blas.Left,
//  C = alpha * B * A + beta * C, if side == blas.Right,
// where A is an n×n or m×m symmetric matrix, B and C are m×n matrices, and alpha
// is a scalar.
func (impl *Implementation) Ssymm(s blas.Side, ul blas.Uplo, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	// declared at cublasgen.h:1850:17 enum CUBLAS_STATUS { ... } cublasSsymm ...
	if impl.e != nil {
		return
	}

	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	var k int
	if s == blas.Left {
		k = m
	} else {
		k = n
	}
	if lda*(k-1)+k > len(a) || lda < max(1, k) {
		panic("blas: index of a out of range")
	}
	if ldb*(m-1)+n > len(b) || ldb < max(1, n) {
		panic("blas: index of b out of range")
	}
	if ldc*(m-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasSsymm(C.cublasHandle_t(impl.h), side2cublasSide(s), uplo2cublasUplo(ul), C.int(m), C.int(n), (*C.float)(&alpha), (*C.float)(&a[0]), C.int(lda), (*C.float)(&b[0]), C.int(ldb), (*C.float)(&beta), (*C.float)(&c[0]), C.int(ldc)))
}

// Dsymm performs one of
//  C = alpha * A * B + beta * C, if side == blas.Left,
//  C = alpha * B * A + beta * C, if side == blas.Right,
// where A is an n×n or m×m symmetric matrix, B and C are m×n matrices, and alpha
// is a scalar.
func (impl *Implementation) Dsymm(s blas.Side, ul blas.Uplo, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	// declared at cublasgen.h:1864:17 enum CUBLAS_STATUS { ... } cublasDsymm ...
	if impl.e != nil {
		return
	}

	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	var k int
	if s == blas.Left {
		k = m
	} else {
		k = n
	}
	if lda*(k-1)+k > len(a) || lda < max(1, k) {
		panic("blas: index of a out of range")
	}
	if ldb*(m-1)+n > len(b) || ldb < max(1, n) {
		panic("blas: index of b out of range")
	}
	if ldc*(m-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasDsymm(C.cublasHandle_t(impl.h), side2cublasSide(s), uplo2cublasUplo(ul), C.int(m), C.int(n), (*C.double)(&alpha), (*C.double)(&a[0]), C.int(lda), (*C.double)(&b[0]), C.int(ldb), (*C.double)(&beta), (*C.double)(&c[0]), C.int(ldc)))
}

func (impl *Implementation) Csymm(s blas.Side, ul blas.Uplo, m, n int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta complex64, c []complex64, ldc int) {
	// declared at cublasgen.h:1878:17 enum CUBLAS_STATUS { ... } cublasCsymm ...
	if impl.e != nil {
		return
	}

	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	var k int
	if s == blas.Left {
		k = m
	} else {
		k = n
	}
	if lda*(k-1)+k > len(a) || lda < max(1, k) {
		panic("blas: index of a out of range")
	}
	if ldb*(m-1)+n > len(b) || ldb < max(1, n) {
		panic("blas: index of b out of range")
	}
	if ldc*(m-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasCsymm(C.cublasHandle_t(impl.h), side2cublasSide(s), uplo2cublasUplo(ul), C.int(m), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&b[0])), C.int(ldb), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

func (impl *Implementation) Zsymm(s blas.Side, ul blas.Uplo, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int) {
	// declared at cublasgen.h:1892:17 enum CUBLAS_STATUS { ... } cublasZsymm ...
	if impl.e != nil {
		return
	}

	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	var k int
	if s == blas.Left {
		k = m
	} else {
		k = n
	}
	if lda*(k-1)+k > len(a) || lda < max(1, k) {
		panic("blas: index of a out of range")
	}
	if ldb*(m-1)+n > len(b) || ldb < max(1, n) {
		panic("blas: index of b out of range")
	}
	if ldc*(m-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasZsymm(C.cublasHandle_t(impl.h), side2cublasSide(s), uplo2cublasUplo(ul), C.int(m), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&b[0])), C.int(ldb), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

func (impl *Implementation) Chemm(s blas.Side, ul blas.Uplo, m, n int, alpha complex64, a []complex64, lda int, b []complex64, ldb int, beta complex64, c []complex64, ldc int) {
	// declared at cublasgen.h:1907:17 enum CUBLAS_STATUS { ... } cublasChemm ...
	if impl.e != nil {
		return
	}

	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	var k int
	if s == blas.Left {
		k = m
	} else {
		k = n
	}
	if lda*(k-1)+k > len(a) || lda < max(1, k) {
		panic("blas: index of a out of range")
	}
	if ldb*(m-1)+n > len(b) || ldb < max(1, n) {
		panic("blas: index of b out of range")
	}
	if ldc*(m-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasChemm(C.cublasHandle_t(impl.h), side2cublasSide(s), uplo2cublasUplo(ul), C.int(m), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&b[0])), C.int(ldb), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

func (impl *Implementation) Zhemm(s blas.Side, ul blas.Uplo, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int, beta complex128, c []complex128, ldc int) {
	// declared at cublasgen.h:1921:17 enum CUBLAS_STATUS { ... } cublasZhemm ...
	if impl.e != nil {
		return
	}

	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	var k int
	if s == blas.Left {
		k = m
	} else {
		k = n
	}
	if lda*(k-1)+k > len(a) || lda < max(1, k) {
		panic("blas: index of a out of range")
	}
	if ldb*(m-1)+n > len(b) || ldb < max(1, n) {
		panic("blas: index of b out of range")
	}
	if ldc*(m-1)+n > len(c) || ldc < max(1, n) {
		panic("blas: index of c out of range")
	}
	impl.e = status(C.cublasZhemm(C.cublasHandle_t(impl.h), side2cublasSide(s), uplo2cublasUplo(ul), C.int(m), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&b[0])), C.int(ldb), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

// Strsm solves
//  A * X = alpha * B,   if tA == blas.NoTrans side == blas.Left,
//  A^T * X = alpha * B, if tA == blas.Trans or blas.ConjTrans, and side == blas.Left,
//  X * A = alpha * B,   if tA == blas.NoTrans side == blas.Right,
//  X * A^T = alpha * B, if tA == blas.Trans or blas.ConjTrans, and side == blas.Right,
// where A is an n×n or m×m triangular matrix, X is an m×n matrix, and alpha is a
// scalar.
//
// At entry to the function, X contains the values of B, and the result is
// stored in place into X.
//
// No check is made that A is invertible.
func (impl *Implementation) Strsm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) {
	// declared at cublasgen.h:1936:17 enum CUBLAS_STATUS { ... } cublasStrsm ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	var k int
	if s == blas.Left {
		k = m
	} else {
		k = n
	}
	if lda*(k-1)+k > len(a) || lda < max(1, k) {
		panic("blas: index of a out of range")
	}
	if ldb*(m-1)+n > len(b) || ldb < max(1, n) {
		panic("blas: index of b out of range")
	}
	impl.e = status(C.cublasStrsm(C.cublasHandle_t(impl.h), side2cublasSide(s), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(m), C.int(n), (*C.float)(&alpha), (*C.float)(&a[0]), C.int(lda), (*C.float)(&b[0]), C.int(ldb)))
}

// Dtrsm solves
//  A * X = alpha * B,   if tA == blas.NoTrans side == blas.Left,
//  A^T * X = alpha * B, if tA == blas.Trans or blas.ConjTrans, and side == blas.Left,
//  X * A = alpha * B,   if tA == blas.NoTrans side == blas.Right,
//  X * A^T = alpha * B, if tA == blas.Trans or blas.ConjTrans, and side == blas.Right,
// where A is an n×n or m×m triangular matrix, X is an m×n matrix, and alpha is a
// scalar.
//
// At entry to the function, X contains the values of B, and the result is
// stored in place into X.
//
// No check is made that A is invertible.
func (impl *Implementation) Dtrsm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
	// declared at cublasgen.h:1950:17 enum CUBLAS_STATUS { ... } cublasDtrsm ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	var k int
	if s == blas.Left {
		k = m
	} else {
		k = n
	}
	if lda*(k-1)+k > len(a) || lda < max(1, k) {
		panic("blas: index of a out of range")
	}
	if ldb*(m-1)+n > len(b) || ldb < max(1, n) {
		panic("blas: index of b out of range")
	}
	impl.e = status(C.cublasDtrsm(C.cublasHandle_t(impl.h), side2cublasSide(s), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(m), C.int(n), (*C.double)(&alpha), (*C.double)(&a[0]), C.int(lda), (*C.double)(&b[0]), C.int(ldb)))
}

func (impl *Implementation) Ctrsm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha complex64, a []complex64, lda int, b []complex64, ldb int) {
	// declared at cublasgen.h:1963:17 enum CUBLAS_STATUS { ... } cublasCtrsm ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	var k int
	if s == blas.Left {
		k = m
	} else {
		k = n
	}
	if lda*(k-1)+k > len(a) || lda < max(1, k) {
		panic("blas: index of a out of range")
	}
	if ldb*(m-1)+n > len(b) || ldb < max(1, n) {
		panic("blas: index of b out of range")
	}
	impl.e = status(C.cublasCtrsm(C.cublasHandle_t(impl.h), side2cublasSide(s), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(m), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&b[0])), C.int(ldb)))
}

func (impl *Implementation) Ztrsm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha complex128, a []complex128, lda int, b []complex128, ldb int) {
	// declared at cublasgen.h:1976:17 enum CUBLAS_STATUS { ... } cublasZtrsm ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	var k int
	if s == blas.Left {
		k = m
	} else {
		k = n
	}
	if lda*(k-1)+k > len(a) || lda < max(1, k) {
		panic("blas: index of a out of range")
	}
	if ldb*(m-1)+n > len(b) || ldb < max(1, n) {
		panic("blas: index of b out of range")
	}
	impl.e = status(C.cublasZtrsm(C.cublasHandle_t(impl.h), side2cublasSide(s), uplo2cublasUplo(ul), trans2cublasTrans(tA), diag2cublasDiag(d), C.int(m), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&b[0])), C.int(ldb)))
}

func (impl *Implementation) Sgeam(tA, tB blas.Transpose, m, n int, alpha float32, a []float32, lda int, beta float32, b []float32, ldb int, c []float32, ldc int) {
	// declared at cublasgen.h:2247:17 enum CUBLAS_STATUS { ... } cublasSgeam ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if tB != blas.NoTrans && tB != blas.Trans && tB != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasSgeam(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), trans2cublasTrans(tB), C.int(m), C.int(n), (*C.float)(&alpha), (*C.float)(&a[0]), C.int(lda), (*C.float)(&beta), (*C.float)(&b[0]), C.int(ldb), (*C.float)(&c[0]), C.int(ldc)))
}

func (impl *Implementation) Dgeam(tA, tB blas.Transpose, m, n int, alpha float64, a []float64, lda int, beta float64, b []float64, ldb int, c []float64, ldc int) {
	// declared at cublasgen.h:2261:17 enum CUBLAS_STATUS { ... } cublasDgeam ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if tB != blas.NoTrans && tB != blas.Trans && tB != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasDgeam(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), trans2cublasTrans(tB), C.int(m), C.int(n), (*C.double)(&alpha), (*C.double)(&a[0]), C.int(lda), (*C.double)(&beta), (*C.double)(&b[0]), C.int(ldb), (*C.double)(&c[0]), C.int(ldc)))
}

func (impl *Implementation) Cgeam(tA, tB blas.Transpose, m, n int, alpha complex64, a []complex64, lda int, beta complex64, b []complex64, ldb int, c []complex64, ldc int) {
	// declared at cublasgen.h:2275:17 enum CUBLAS_STATUS { ... } cublasCgeam ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if tB != blas.NoTrans && tB != blas.Trans && tB != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasCgeam(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), trans2cublasTrans(tB), C.int(m), C.int(n), (*C.cuComplex)(unsafe.Pointer(&alpha)), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&beta)), (*C.cuComplex)(unsafe.Pointer(&b[0])), C.int(ldb), (*C.cuComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

func (impl *Implementation) Zgeam(tA, tB blas.Transpose, m, n int, alpha complex128, a []complex128, lda int, beta complex128, b []complex128, ldb int, c []complex128, ldc int) {
	// declared at cublasgen.h:2289:17 enum CUBLAS_STATUS { ... } cublasZgeam ...
	if impl.e != nil {
		return
	}

	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if tB != blas.NoTrans && tB != blas.Trans && tB != blas.ConjTrans {
		panic("blas: illegal transpose")
	}
	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZgeam(C.cublasHandle_t(impl.h), trans2cublasTrans(tA), trans2cublasTrans(tB), C.int(m), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&alpha)), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&beta)), (*C.cuDoubleComplex)(unsafe.Pointer(&b[0])), C.int(ldb), (*C.cuDoubleComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

func (impl *Implementation) Sdgmm(mode blas.Side, m, n int, a []float32, lda int, x []float32, incX int, c []float32, ldc int) {
	// declared at cublasgen.h:2614:17 enum CUBLAS_STATUS { ... } cublasSdgmm ...
	if impl.e != nil {
		return
	}

	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (m-1)*incX >= len(x)) || (incX < 0 && (1-m)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasSdgmm(C.cublasHandle_t(impl.h), side2cublasSide(mode), C.int(m), C.int(n), (*C.float)(&a[0]), C.int(lda), (*C.float)(&x[0]), C.int(incX), (*C.float)(&c[0]), C.int(ldc)))
}

func (impl *Implementation) Ddgmm(mode blas.Side, m, n int, a []float64, lda int, x []float64, incX int, c []float64, ldc int) {
	// declared at cublasgen.h:2625:17 enum CUBLAS_STATUS { ... } cublasDdgmm ...
	if impl.e != nil {
		return
	}

	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (m-1)*incX >= len(x)) || (incX < 0 && (1-m)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasDdgmm(C.cublasHandle_t(impl.h), side2cublasSide(mode), C.int(m), C.int(n), (*C.double)(&a[0]), C.int(lda), (*C.double)(&x[0]), C.int(incX), (*C.double)(&c[0]), C.int(ldc)))
}

func (impl *Implementation) Cdgmm(mode blas.Side, m, n int, a []complex64, lda int, x []complex64, incX int, c []complex64, ldc int) {
	// declared at cublasgen.h:2636:17 enum CUBLAS_STATUS { ... } cublasCdgmm ...
	if impl.e != nil {
		return
	}

	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (m-1)*incX >= len(x)) || (incX < 0 && (1-m)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasCdgmm(C.cublasHandle_t(impl.h), side2cublasSide(mode), C.int(m), C.int(n), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

func (impl *Implementation) Zdgmm(mode blas.Side, m, n int, a []complex128, lda int, x []complex128, incX int, c []complex128, ldc int) {
	// declared at cublasgen.h:2647:17 enum CUBLAS_STATUS { ... } cublasZdgmm ...
	if impl.e != nil {
		return
	}

	if m < 0 {
		panic("blas: m < 0")
	}
	if n < 0 {
		panic("blas: n < 0")
	}
	if incX == 0 {
		panic("blas: zero x index increment")
	}
	if (incX > 0 && (m-1)*incX >= len(x)) || (incX < 0 && (1-m)*incX >= len(x)) {
		panic("blas: x index out of range")
	}
	if lda*(m-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZdgmm(C.cublasHandle_t(impl.h), side2cublasSide(mode), C.int(m), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&x[0])), C.int(incX), (*C.cuDoubleComplex)(unsafe.Pointer(&c[0])), C.int(ldc)))
}

func (impl *Implementation) Stpttr(ul blas.Uplo, n int, aP, a []float32, lda int) {
	// declared at cublasgen.h:2659:17 enum CUBLAS_STATUS { ... } cublasStpttr ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasStpttr(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.float)(&aP[0]), (*C.float)(&a[0]), C.int(lda)))
}

func (impl *Implementation) Dtpttr(ul blas.Uplo, n int, aP, a []float64, lda int) {
	// declared at cublasgen.h:2666:17 enum CUBLAS_STATUS { ... } cublasDtpttr ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasDtpttr(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.double)(&aP[0]), (*C.double)(&a[0]), C.int(lda)))
}

func (impl *Implementation) Ctpttr(ul blas.Uplo, n int, aP, a []complex64, lda int) {
	// declared at cublasgen.h:2673:17 enum CUBLAS_STATUS { ... } cublasCtpttr ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasCtpttr(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.cuComplex)(unsafe.Pointer(&aP[0])), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda)))
}

func (impl *Implementation) Ztpttr(ul blas.Uplo, n int, aP, a []complex128, lda int) {
	// declared at cublasgen.h:2680:17 enum CUBLAS_STATUS { ... } cublasZtpttr ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZtpttr(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&aP[0])), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda)))
}

func (impl *Implementation) Strttp(ul blas.Uplo, n int, a []float32, lda int, aP []float32) {
	// declared at cublasgen.h:2687:17 enum CUBLAS_STATUS { ... } cublasStrttp ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasStrttp(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.float)(&a[0]), C.int(lda), (*C.float)(&aP[0])))
}

func (impl *Implementation) Dtrttp(ul blas.Uplo, n int, a []float64, lda int, aP []float64) {
	// declared at cublasgen.h:2694:17 enum CUBLAS_STATUS { ... } cublasDtrttp ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasDtrttp(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.double)(&a[0]), C.int(lda), (*C.double)(&aP[0])))
}

func (impl *Implementation) Ctrttp(ul blas.Uplo, n int, a []complex64, lda int, aP []complex64) {
	// declared at cublasgen.h:2701:17 enum CUBLAS_STATUS { ... } cublasCtrttp ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasCtrttp(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.cuComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuComplex)(unsafe.Pointer(&aP[0]))))
}

func (impl *Implementation) Ztrttp(ul blas.Uplo, n int, a []complex128, lda int, aP []complex128) {
	// declared at cublasgen.h:2708:17 enum CUBLAS_STATUS { ... } cublasZtrttp ...
	if impl.e != nil {
		return
	}

	if n < 0 {
		panic("blas: n < 0")
	}
	if lda*(n-1)+n > len(a) || lda < max(1, n) {
		panic("blas: index of a out of range")
	}
	impl.e = status(C.cublasZtrttp(C.cublasHandle_t(impl.h), uplo2cublasUplo(ul), C.int(n), (*C.cuDoubleComplex)(unsafe.Pointer(&a[0])), C.int(lda), (*C.cuDoubleComplex)(unsafe.Pointer(&aP[0]))))
}
