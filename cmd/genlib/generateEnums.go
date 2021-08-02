package main

import (
	"fmt"
	"io"
	"strings"

	"github.com/gorgonia/bindgen"
	"modernc.org/cc"
)

// genCUresult represents a list of enums we want to generate
var genCUreuslt = map[bindgen.TypeKey]struct{}{
	{Kind: cc.Enum, Name: "CUresult"}: {},
}

var cuResultMappings = map[bindgen.TypeKey]string{
	{Kind: cc.Enum, Name: "CUresult"}: "cuResult",
}

func goRenameCUResult(a string) string {
	a = strings.TrimPrefix(a, "CUDA_")
	a = strings.TrimPrefix(a, "ERROR_")
	splits := strings.Split(a, "_")
	for i, s := range splits {
		splits[i] = strings.Title(strings.ToLower(s))
	}
	return strings.Join(splits, "")
}

func generateResultEnums(f io.Writer) {
	t, err := bindgen.Parse(bindgen.Model(), "cuda.h")
	if err != nil {
		panic(err)
	}

	enums := func(decl *cc.Declarator) bool {
		name := bindgen.NameOf(decl)
		kind := decl.Type.Kind()
		tk := bindgen.TypeKey{Kind: kind, Name: name}
		if _, ok := genCUreuslt[tk]; ok {
			return true
		}
		return false
	}
	decls, err := bindgen.Get(t, enums)
	if err != nil {
		panic(err)
	}

	var m []string
	for _, d := range decls {
		e := d.(*bindgen.Enum)
		tk := bindgen.TypeKey{Kind: cc.Enum, Name: e.Name}
		fmt.Fprintf(f, "type %v int\nconst (\n", cuResultMappings[tk])

		// then write the const definitions:
		// 	const(...)

		for _, a := range e.Type.EnumeratorList() {
			enumName := string(a.DefTok.S())
			goName := goRenameCUResult(enumName)
			m = append(m, goName)
			fmt.Fprintf(f, "%v %v = C.%v\n", goName, cuResultMappings[tk], enumName)
		}
		f.Write([]byte(")\n"))
	}
	fmt.Fprintf(f, "var resString = map[cuResult]string{\n")
	for _, s := range m {
		fmt.Fprintf(f, "%v: %q,\n", s, s)
	}
	f.Write([]byte("}\n"))

}
