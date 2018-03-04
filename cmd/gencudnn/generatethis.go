package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/cznic/cc"
	"github.com/gorgonia/bindgen"
)

// generate this contains function to generate for THIS package (main)

// generateMappings is used to generate the mappings
func generateMappings(appendCurrent bool) {
	hdr := "package main\n"

	initfn := `
	func init() {


	`
	t, err := bindgen.Parse(model, hdrfile)
	handleErr(err)

	var buf io.WriteCloser
	if appendCurrent {
		buf, err = os.OpenFile("mappings.go", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	} else {
		buf, err = os.OpenFile("mappings.go", os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	}
	handleErr(err)
	defer buf.Close()

	if !appendCurrent {
		fmt.Fprintln(buf, hdr)
		bindgen.GenIgnored(buf, t, functions)
		fmt.Fprintln(buf, initfn)
		bindgen.GenNameMap(buf, t, "fnNameMap", processNameBasic, functions, true)
		bindgen.GenNameMap(buf, t, "enumMappings", processNameBasic, enums, true)
		generateContextualNameMap(buf, t)
		generateCreations(buf, t)
		generateAlphaBeta(buf, t)
		generateRetvals(buf, t)
		fmt.Fprintln(buf, "}\n")
	}

	fmt.Fprintln(buf, initfn)
	fmt.Fprintln(buf, "}\n")
}

func generateContextualNameMap(buf io.Writer, t *cc.TranslationUnit) {
	filter := func(d *cc.Declarator) bool {
		if !functions(d) {
			return false
		}
		ps, _ := d.Type.Parameters()
		for _, p := range ps {
			if bindgen.NameOf(p.Declarator) == "handle" {
				return true
			}
		}
		return false
	}

	decls, err := bindgen.Get(t, filter)
	handleErr(err)
	fmt.Fprint(buf, "contextual = map[string]struct{} {\n")
	for _, d := range decls {
		fmt.Fprintf(buf, "%q: {},\n", d.(*bindgen.CSignature).Name)
	}
	fmt.Fprint(buf, "}\n")
}

func generateRetvals(buf io.Writer, t *cc.TranslationUnit) {
	decls, err := bindgen.Get(t, functions)
	handleErr(err)
	fmt.Fprint(buf, "retVals = map[string]map[int]string {\n")
	for _, d := range decls {
		cs := d.(*bindgen.CSignature)
		params := cs.Parameters()
		lname := strings.ToLower(cs.Name)

		if !strings.Contains(lname, "get") && !strings.Contains(lname, "create") {
			continue
		}

		for i := len(params) - 1; i >= 0; i-- {
			p := params[i]
			if !bindgen.IsConstType(p.Type()) && bindgen.IsPointer(p.Type()) {
				fmt.Fprintf(buf, "%q: {%d: %q},\n", cs.Name, i, p.Name())
				break
			}
		}
	}
	fmt.Fprint(buf, "}\n")
}

func generateCreations(buf io.Writer, t *cc.TranslationUnit) {
	decls, err := bindgen.Get(t, functions)
	handleErr(err)
	fmt.Fprint(buf, "creations = map[string]string {\n")
	for _, d := range decls {
		cs := d.(*bindgen.CSignature)
		params := cs.Parameters()

		if !strings.Contains(strings.ToLower(cs.Name), "create") {
			continue
		}

		for i := len(params) - 1; i >= 0; i-- {
			p := params[i]
			if !bindgen.IsConstType(p.Type()) && bindgen.IsPointer(p.Type()) {
				// fmt.Fprintf(buf, "%q: {%d: %q},\n", cs.Name, i, p.Name())
				setFn := strings.Replace(cs.Name, "Create", "Set", -1)

				if searchByName(decls, setFn) == nil {
					setFn = ""
				}
				fmt.Fprintf(buf, "%q: %q,\n", nameOfType(p.Type()), setFn)

				break
			}
		}
	}
	fmt.Fprint(buf, "}\n")
}

func generateAlphaBeta(buf io.Writer, t *cc.TranslationUnit) {
	decls, err := bindgen.Get(t, functions)
	handleErr(err)
	fmt.Fprint(buf, "alphaBetas = map[string]map[int]string {\n")
	for _, d := range decls {
		cs := d.(*bindgen.CSignature)
		params := cs.Parameters()

		var printedName bool
		for i := len(params) - 1; i >= 0; i-- {
			p := params[i]
			if !(bindgen.IsConstType(p.Type()) && bindgen.IsPointer(p.Type())) {
				continue
			}

			switch p.Name() {
			case "alpha", "alpha1", "alpha2", "alpha3", "beta", "beta1":
				if !printedName {
					printedName = true
					fmt.Fprintf(buf, "%q: {", cs.Name)
				}
				fmt.Fprintf(buf, "%d: %q, ", i, p.Name())
			default:
			}

		}
		if printedName {
			fmt.Fprint(buf, "},\n") // close the first
		}
	}
	fmt.Fprint(buf, "}\n")
}
