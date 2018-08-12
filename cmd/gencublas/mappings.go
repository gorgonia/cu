package main

import (
	"text/template"

	"github.com/cznic/cc"
	bg "github.com/gorgonia/bindgen"
)

var skip = map[string]bool{
	"cublasErrprn":    true,
	"cublasSrotg":     true,
	"cublasSrotmg":    true,
	"cublasSrotm":     true,
	"cublasDrotg":     true,
	"cublasDrotmg":    true,
	"cublasDrotm":     true,
	"cublasCrotg":     true,
	"cublasZrotg":     true,
	"cublasCdotu_sub": true,
	"cublasCdotc_sub": true,
	"cublasZdotu_sub": true,
	"cublasZdotc_sub": true,

	// ATLAS extensions.
	"cublasCsrot": true,
	"cublasZdrot": true,

	// trmm
	"cublasStrmm": true,
	"cublasDtrmm": true,
	"cublasZtrmm": true,
	"cublasCtrmm": true,
}

var cToGoType = map[string]string{
	"int":    "int",
	"float":  "float32",
	"double": "float64",
}

var blasEnums = map[string]bg.Template{
	"CUBLAS_ORDER":     bg.Pure(template.Must(template.New("order").Parse("order"))),
	"CUBLAS_DIAG":      bg.Pure(template.Must(template.New("diag").Parse("blas.Diag"))),
	"CUBLAS_TRANSPOSE": bg.Pure(template.Must(template.New("trans").Parse("blas.Transpose"))),
	"CUBLAS_UPLO":      bg.Pure(template.Must(template.New("uplo").Parse("blas.Uplo"))),
	"CUBLAS_SIDE":      bg.Pure(template.Must(template.New("side").Parse("blas.Side"))),
}

var cgoEnums = map[string]bg.Template{
	"CUBLAS_ORDER":     bg.Pure(template.Must(template.New("order").Parse("C.enum_CBLAS_ORDER(rowMajor)"))),
	"CUBLAS_DIAG":      bg.Pure(template.Must(template.New("diag").Parse("diag2cublasDiag({{.}})"))),
	"CUBLAS_TRANSPOSE": bg.Pure(template.Must(template.New("trans").Parse("trans2cublasTrans({{.}})"))),
	"CUBLAS_UPLO":      bg.Pure(template.Must(template.New("uplo").Parse("uplo2cublasUplo({{.}})"))),
	"CUBLAS_SIDE":      bg.Pure(template.Must(template.New("side").Parse("side2cublasSide({{.}})"))),
}

var (
	complex64Type = map[bg.TypeKey]bg.Template{
		{Kind: cc.FloatComplex, IsPointer: true}: bg.Pure(template.Must(template.New("void*").Parse(
			`{{if eq . "alpha" "beta"}}complex64{{else}}[]complex64{{end}}`,
		)))}

	complex128Type = map[bg.TypeKey]bg.Template{
		{Kind: cc.DoubleComplex, IsPointer: true}: bg.Pure(template.Must(template.New("void*").Parse(
			`{{if eq . "alpha" "beta"}}complex128{{else}}[]complex128{{end}}`,
		)))}
)

var names = map[string]string{
	"uplo":   "ul",
	"trans":  "t",
	"transA": "tA",
	"transB": "tB",
	"side":   "s",
	"diag":   "d",
}
